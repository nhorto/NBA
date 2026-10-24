"""Late swap: re-solve the unlocked slots of entered lineups (docs/PLAYERS.md §2.5, §5.6; T2.14).

The owner has entries on DraftKings (our saved lineups, or their own lineups from DK's entries
CSV). Partway through a Classic slate some games have tipped. For each entry:

* **fixed slots**: the player's game has started (tip <= now), or DK's own file says so (a
  ``(LOCKED)`` cell, or "In Progress" / "Final" in the file's player list). A fixed player is
  never moved or replaced: he is pinned to his slot in the CP-SAT model and every result is
  checked again afterwards (`check_entry`, which raises `SwapError` rather than return a lineup
  that touches a fixed slot);
* **unlocked slots** are re-solved under the entry's objective (cash: mu - 0.3 (mu - p25);
  Top N / GPP: mu) with the cap left after the fixed salaries, the latest stored projections and
  the ruled-out players banned: DK status O, a late-news status change to Out, and the §5.3 rule
  (P(play) < 0.85 when no later swap run comes before his game). Unlocked players may move between
  unlocked slots (positional flex: a PG/SG in G can move to PG so that a SG-only replacement takes
  G); a second small solve keeps every player that stays in the slot he had when possible, so the
  edit touches as few cells as it can;
* **portfolio checks** when several entries are swapped together: entries of the same contest
  stay at least `min_diff` players apart (identical entries are solved once and stay identical),
  optional max shared players (`max_overlap`) and per-player exposure caps (`max_exposure`),
  all counted against the fixed parts, which cannot change. A cap the fixed parts already break
  is reported, never "fixed" by touching a locked slot;
* an entry is only changed when a ruled-out player has to go, or the objective gains at least
  `min_gain` (default 0.5 DK points), so projection noise does not churn lineups.

Output: a DK edit CSV (``Entry ID, Contest Name, Contest ID, Entry Fee, <slots>``, `export.
to_edit_csv`). Fixed and unchanged cells are written back exactly as DK wrote them in the uploaded
file. Nothing here talks to DraftKings: the owner uploads the file by hand.

DK entries CSV (what "Edit Entries" -> download gives; the same layout pydfs-lineup-optimizer
reads): header ``Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL``,
then an empty column, ``Instructions`` and, further right and down, DK's player list (``Position,
Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame``). Slot
cells are ``Name (ID)``; we also accept bare numeric ids (our own export's default) and plain
names. Which numeric id DK's uploader wants in each column is NOT verified (open item in
HANDOFF.md): for a new player we use the ID from DK's own player list in the uploaded file, else
the draftable id of that slot (as `export.py` does).

Two layers:

* `resolve_targets` is the runner's swap loop (moved here from ``runner._resolve_swaps``,
  unchanged), used by the automatic ``kind="swap"`` runs;
* `plan` is the owner-facing re-solver (CLI ``courtside dfs swap`` and the slate page's
  "Late swap" action). It reads only: stored projections, captures and late news.

CLI::

    courtside dfs swap --slate 154575                       # our saved lineups, now
    courtside dfs swap --slate 154575 --entries DKEntries.csv --now 2026-10-20T23:05:00Z
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import logging
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import floor
from typing import Iterable, Mapping, Sequence

from ortools.sat.python import cp_model

from . import export as X
from . import optimizer as O
from . import rules as R

log = logging.getLogger("courtside.proj.dfs.swap")

CASH_LAMBDA = 0.3              # runner.CASH_LAMBDA
SOLVE_LIMIT_S = 20.0           # runner.SOLVE_LIMIT_S, per CP-SAT solve
TOTAL_LIMIT_S = 30.0           # the whole plan (PLAYERS.md §5.8)
MIN_GAIN = 0.5                 # objective gain (DK points) that justifies changing an entry
P_PLAY_MIN = 0.85
SCOPES = ("all", "news")
OBJECTIVES = ("auto", "cash", "mean")
ENTRY_COLS = ("Entry ID", "Contest Name", "Contest ID", "Entry Fee")
CASH_CONTEST = re.compile(r"double[\s-]*up|50\s*/\s*50|fifty|head[\s-]*to[\s-]*head|\bh2h\b|"
                          r"triple[\s-]*up|quintuple|\bcash\b", re.I)
_CELL_ID = re.compile(r"^(?P<name>.*?)\s*\((?P<id>\d+)\)\s*(?P<rest>.*)$")
_LOCKED = re.compile(r"\(\s*locked\s*\)|\blocked\b", re.I)
MAX_ENTRIES = 500


class SwapError(RuntimeError):
    """A result would change a fixed slot, or the input cannot be used (shown to the owner)."""


# ============================================================================== runner layer

def resolve_targets(pool, projs, rules, targets, bans, now, min_diff):
    """The automatic swap run's loop (`runner._resolve_swaps`, moved here unchanged).

    Re-solve each target saved lineup with the started players of its game fixed (``existing=``
    + ``now=``) under the objective it was built with. Earlier results are cut off (min_diff) so
    Top N lineups do not collapse into one; dropped if that is infeasible."""
    from . import runner                       # the runner's constants, so the two never drift
    lam, limit = runner.CASH_LAMBDA, runner.SOLVE_LIMIT_S
    by_id = pool.by_id()
    started = pool.locked_ids(now)
    free_bans = set(bans) - started
    produced: dict[str, list] = {"cash": [], "topn": []}
    out = []
    for t in targets:
        pairs = [(s, pid) for s, pid in t["pairs"] if pid in by_id]
        for s, pid in pairs:
            if pid in started and pid not in projs:        # fixed players need a number
                p = by_id[pid]
                projs[pid] = O.Projection(p.fppg if p.fppg is not None else 0.0)
        mode = "cash" if t["mode"] == "cash" else "mean"
        lu = None
        for cuts in (True, False):
            model = O.model_for(pool, projs, rules, mode=mode, lam=lam, bans=free_bans,
                                now=now, existing=pairs, time_limit_s=limit)
            if cuts:
                for prev in produced[t["mode"]]:
                    model.add_no_good(prev, min_diff)
            lu = model.solve()
            if lu is not None:
                break
        if lu is None:
            out.append({"target": t, "lineup": None})
            continue
        produced[t["mode"]].append(lu)
        before = set(t["player_ids"])
        out.append({"target": t, "lineup": lu,
                    "in": [_name_of(by_id, i) for i in sorted(lu.player_ids - before)],
                    "out": [_name_of(by_id, i) for i in sorted(before - lu.player_ids)]})
    return out


def _name_of(by_id: Mapping, pid: str) -> str:
    p = by_id.get(pid)
    return p.name if p else pid


# ============================================================================== small helpers

def _utc(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def _norm(name) -> str:
    from .standings import _norm as snorm
    return snorm(name)


def objective_for(contest_name: str | None, mode: str | None = None) -> str:
    """'cash' for a cash-style entry (our cash lineup, a double-up / 50-50 / head-to-head
    contest), else 'mean' (Top N, GPP)."""
    if mode == "cash":
        return "cash"
    if mode in ("topn", "mean"):
        return "mean"
    return "cash" if contest_name and CASH_CONTEST.search(contest_name) else "mean"


def news_reason(item: Mapping) -> str:
    """Plain English for a `runner.late_news` item (same wording as the slate page)."""
    parts = []
    for r in item.get("reasons") or ():
        if r == "status":
            st = item.get("report_status") or item.get("status") or {}
            parts.append(f"status {st.get('from') or 'none'} → {st.get('to') or 'none'}")
        elif r == "unexpected_starter":
            parts.append("named a starter")
        elif r == "unexpected_non_starter":
            parts.append("not in the starting lineup")
        elif r == "minutes":
            m = item.get("minutes") or {}
            parts.append(f"projected minutes {m.get('from')} → {m.get('to')}")
        else:
            parts.append(str(r).replace("_", " "))
    return "; ".join(parts)


def _news_is_out(item: Mapping) -> bool:
    for k in ("status", "report_status"):
        to = str(((item.get(k) or {}).get("to") or "")).strip().lower()
        if to in ("o", "out"):
            return True
    return False


# ============================================================================== slate data

@dataclass
class SlateIndex:
    """Every draftable the slate ever listed (all captures, latest row per draftable), so a
    lineup can be read even when a player has since been greyed out of the pool."""
    by_draftable: dict[str, dict]
    by_player: dict[str, dict[str, dict]]       # dk player id -> slot -> row
    by_name: dict[str, set[str]]                # normalised name -> dk player ids

    def row(self, pid: str, slot: str | None = None) -> dict | None:
        rows = self.by_player.get(pid) or {}
        if slot and slot in rows:
            return rows[slot]
        return next(iter(rows.values()), None)


def slate_index(conn: sqlite3.Connection, slate_id) -> SlateIndex:
    by_d: dict[str, dict] = {}
    for r in conn.execute("SELECT * FROM dfs_salaries WHERE slate_id=? AND draftable_id IS NOT NULL "
                          "ORDER BY captured_at, id", (str(slate_id),)):
        by_d[str(r["draftable_id"])] = dict(r)
    by_p: dict[str, dict[str, dict]] = defaultdict(dict)
    by_n: dict[str, set[str]] = defaultdict(set)
    for d, r in by_d.items():
        pid = str(r["site_player_id"]) if r["site_player_id"] is not None else None
        if pid is None:
            continue
        by_p[pid][r["roster_slot"] or ""] = r
        by_n[_norm(r["player_name"])].add(pid)
    return SlateIndex(by_d, dict(by_p), dict(by_n))


def guess_slate(conn: sqlite3.Connection, ids: Iterable[str]) -> tuple[str | None, dict]:
    """The slate whose draftables hold most of `ids` (draftable or DK player ids) -> (slate id,
    {slate: hits})."""
    ids = sorted({str(i) for i in ids if i})
    hits: Counter = Counter()
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        q = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT slate_id, COUNT(DISTINCT draftable_id) FROM dfs_salaries "
                              f"WHERE draftable_id IN ({q}) GROUP BY slate_id", chunk):
            hits[str(r[0])] += int(r[1])
    if not hits:
        return None, {}
    return hits.most_common(1)[0][0], dict(hits)


def load_projections(conn: sqlite3.Connection, slate_id, pool, run_id: str | None = None
                     ) -> tuple[dict, dict]:
    """({dk id: Projection(mean, p25, p10, p50, p90)}, info) from the newest stored projection
    run of the slate (else the newest run that covers its games), DK 'dk' stat, cond 'all' (the
    P(play)-weighted mixture the runner optimises). Without any: DK FPPG ('fppg-fallback').
    info: run_id, model_version, created_at, source, p_play {dk id: p}. `run_id` pins the run.
    """
    sid = str(slate_id)
    try:
        if run_id is None:
            r = conn.execute(
                "SELECT r.run_id FROM dfs_runs r WHERE r.slate_id=? AND r.ok=1 AND EXISTS (SELECT "
                "1 FROM player_projections p WHERE p.run_id=r.run_id) ORDER BY r.created_at DESC, "
                "r.rowid DESC LIMIT 1", (sid,)).fetchone()
            run_id = r[0] if r else None
        if run_id is None:
            games = sorted({p.game_id for p in pool.players if p.game_id})
            if games:
                q = ",".join("?" * len(games))
                r = conn.execute(f"SELECT run_id FROM player_projections WHERE run_id IS NOT NULL "
                                 f"AND game_id IN ({q}) ORDER BY created_at DESC LIMIT 1",
                                 games).fetchone()
                run_id = r[0] if r else None
    except sqlite3.OperationalError:
        run_id = None
    info: dict = {"run_id": run_id, "model_version": None, "created_at": None,
                  "source": "fppg", "p_play": {}}
    projs: dict[str, O.Projection] = {}
    if run_id is not None:
        dk_of: dict[int, list[str]] = defaultdict(list)
        for a, b in conn.execute("SELECT source_id, player_id FROM player_ids WHERE source='dk' "
                                 "AND player_id IS NOT NULL"):
            dk_of[int(b)].append(str(a))
        rows = conn.execute(
            "SELECT p.player_id, p.p_play, p.model_version, p.created_at, s.mean, s.q25, s.q10, "
            "s.q50, s.q90 FROM player_projections p JOIN player_projection_stats s ON "
            "s.projection_id=p.id WHERE p.run_id=? AND s.stat='dk' AND s.cond='all' "
            "ORDER BY p.created_at, p.id", (run_id,)).fetchall()
        in_pool = {p.dk_player_id for p in pool.players}
        for r in rows:
            if r["mean"] is None:
                continue
            pr = O.Projection(float(r["mean"]), *(None if r[k] is None else float(r[k])
                                                  for k in ("q25", "q10", "q50", "q90")))
            for dk in dk_of.get(int(r["player_id"]), []):
                if dk in in_pool:
                    projs[dk] = pr
                    if r["p_play"] is not None:
                        info["p_play"][dk] = float(r["p_play"])
            info["model_version"] = info["model_version"] or r["model_version"]
            info["created_at"] = info["created_at"] or r["created_at"]
        if projs:
            info["source"] = "model"
    if not projs:
        info["run_id"] = None
        info["model_version"] = "fppg-fallback"
        projs = {p.dk_player_id: O.Projection(p.fppg) for p in pool.players if p.fppg is not None}
    return projs, info


def slate_news(conn: sqlite3.Connection, slate_id, now: datetime) -> list[dict]:
    """Late news for the slate as of `now`: the live `runner.late_news` against the latest build
    run, plus what the slate's runs since that build recorded (live wins per player)."""
    from . import runner
    out: dict[str, dict] = {}
    runs = runner._run_rows(conn, slate_id)
    base = next((r for r in reversed(runs) if r["kind"] != "swap"
                 and not r["detail_obj"].get("skipped")), None)
    for r in runs:
        if base is not None and r["created_at"] < base["created_at"]:
            continue
        for n in runner._loads(r.get("late_news_json"), []) or []:
            if isinstance(n, dict) and n.get("player_id") is not None:
                out[str(n["player_id"])] = n
    try:
        for n in runner.late_news(conn, slate_id, now, runner.prev_run(conn, slate_id, now)):
            out[str(n["player_id"])] = n
    except Exception:  # noqa: BLE001 - stored news is still useful
        log.exception("late news unavailable; using the stored lists")
    return list(out.values())


def ruled_out(conn, slate, pool, tips, now, news, p_play) -> dict[str, str]:
    """{dk id: reason} of players who must not be picked for an unlocked slot."""
    from . import runner
    out: dict[str, str] = {}
    by_news = {str(n["player_id"]): n for n in news}
    avail = {}
    for p in pool.players:
        pp = p_play.get(p.dk_player_id)
        avail[p.dk_player_id] = {"p": runner.dk_status_p(p.status) if pp is None else pp,
                                 "status": p.status or "", "source": "projection"
                                 if pp is not None else "dk_status"}
    nxt = runner.next_run_at("swap", slate, tips, now)
    bans, detail = runner.availability_bans(pool, avail, nxt, now)
    for d in detail:
        pid = d["player_id"]
        if d["status"] == "O":
            out[pid] = "DraftKings lists him Out"
        else:
            out[pid] = f"chance to play {d['p_play']:.0%}, no later swap before his game"
    for pid, n in by_news.items():
        if _news_is_out(n):
            out.setdefault(pid, news_reason(n) or "ruled out")
    return out


# ============================================================================== entries

@dataclass
class SwapEntry:
    """One lineup to swap. `pairs` [(slot, dk player id | None)] in DK column order; `cells`
    the original CSV cells (None for our saved lineups); `dk_locked` slot indexes DK's file
    marks locked."""
    key: str
    label: str
    pairs: list[tuple[str, str | None]]
    names: list[str]
    mode: str | None = None                  # 'cash' | 'topn' (saved), None (upload: by contest)
    entry_id: str | None = None
    contest_name: str | None = None
    contest_id: str | None = None
    entry_fee: str | None = None
    cells: list[str] | None = None
    dk_locked: set[int] = field(default_factory=set)
    source: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    slots_meta: list[dict] = field(default_factory=list)    # saved slots_json rows

    @property
    def group(self) -> str:
        """Entries of the same group must stay distinct (contest, or our saved mode)."""
        return f"c:{self.contest_id}" if self.contest_id else f"m:{self.mode or 'upload'}"


def saved_entries(conn: sqlite3.Connection, slate_id) -> list[SwapEntry]:
    """Our saved lineups as they stand (`runner.current_lineups`: swaps overlaid)."""
    from . import runner
    out = []
    for d in runner.current_lineups(conn, slate_id):
        slots = runner._loads(d.get("slots_json"), []) or []
        label = "Cash lineup" if d["mode"] == "cash" else f"Top N #{d['rank']}"
        out.append(SwapEntry(
            key=f"{d['mode']}-{d['rank']}", label=label, pairs=list(d["pairs"]),
            names=[s.get("name") or str(s.get("player_id")) for s in slots], mode=d["mode"],
            source={"kind": "saved", "lineup_id": d["id"], "run_id": d["run_id"],
                    "via": d["via"]}, slots_meta=slots))
    return out


@dataclass
class ParsedFile:
    entries: list[SwapEntry]
    slots: list[str]
    formats: Counter
    dk_ids: dict[str, str]           # dk player id -> the ID DK's own player list uses
    started_ids: set[str]            # players DK's list shows In Progress / Final
    warnings: list[str]
    showdown: bool = False


def _find(header: list[str], *names) -> int | None:
    want = {n.casefold() for n in names}
    for i, h in enumerate(header):
        if (h or "").strip().lstrip("﻿").casefold() in want:
            return i
    return None


def read_rows(data: bytes | str) -> list[list[str]]:
    from .standings import decode_text
    text = data if isinstance(data, str) else decode_text(data)
    return list(csv.reader(io.StringIO(text, newline="")))


def entry_ids_in(data: bytes | str) -> set[str]:
    """Every numeric id in the slot cells (for `guess_slate`)."""
    rows = read_rows(data)
    out = set()
    for r in rows[1:]:
        for c in r[4:20]:
            m = _CELL_ID.match(c.strip())
            if m:
                out.add(m["id"])
            elif c.strip().isdigit():
                out.add(c.strip())
    return out


def parse_entries_csv(data: bytes | str, idx: SlateIndex, rules: R.Rules) -> ParsedFile:
    """DK's entries / edit-entries CSV -> `SwapEntry`s resolved against the slate.

    Cells: ``Name (ID)`` (DK's own format, optionally followed by ``(LOCKED)``), a bare numeric
    id, or a player name. An id is matched to a draftable of the slate (any slot), then to a DK
    player id; a name must be unique in the slate. Rows without a numeric Entry ID are skipped
    (the instructions and DK's player list share the file)."""
    rows = read_rows(data)
    hi = next((i for i, r in enumerate(rows[:10]) if _find(r, "Entry ID", "Entry_ID")
               is not None), None)
    if hi is None:
        raise SwapError("This does not look like a DraftKings entries file: no Entry ID column. "
                        "On DraftKings use Lineups, then Edit Entries, then download the CSV.")
    head = rows[hi]
    ei = _find(head, "Entry ID", "Entry_ID")
    ci, cid, fee = (_find(head, "Contest Name"), _find(head, "Contest ID"),
                    _find(head, "Entry Fee"))
    names = set(rules.slot_names)
    slot_cols: list[int] = []
    start = max(i for i in (ei, ci, cid, fee) if i is not None) + 1
    for i in range(start, len(head)):
        h = (head[i] or "").strip().upper()
        if h in names or h in ("CPT", "UTIL", "FLEX"):
            slot_cols.append(i)
        elif slot_cols:
            break
    slots = [head[i].strip().upper() for i in slot_cols]
    pf = ParsedFile([], slots, Counter(), {}, set(), [])
    pf.showdown = "CPT" in slots
    if not slots:
        raise SwapError("No lineup columns (PG, SG, SF, PF, C, G, F, UTIL) in this file.")
    if not pf.showdown and Counter(slots) != Counter(rules.slot_names):
        raise SwapError(f"This file's lineup columns ({', '.join(slots)}) do not match this "
                        f"slate ({', '.join(rules.slot_names)}).")

    # DK's player list (right-hand block): ID -> player, and which games are under way
    for i, r in enumerate(rows):
        idc = _find(r, "ID")
        ta = _find(r, "TeamAbbrev")
        if idc is None or ta is None:
            continue
        gi, nm = _find(r, "Game Info"), _find(r, "Name")
        for r2 in rows[i + 1:]:
            if idc >= len(r2) or not r2[idc].strip().isdigit():
                continue
            did = r2[idc].strip()
            hit = idx.by_draftable.get(did)
            pid = str(hit["site_player_id"]) if hit and hit["site_player_id"] is not None else None
            if pid is None and nm is not None and nm < len(r2):
                ps = idx.by_name.get(_norm(r2[nm]), set())
                pid = next(iter(ps)) if len(ps) == 1 else None
            if pid is None:
                continue
            pf.dk_ids.setdefault(pid, did)
            g = r2[gi].strip().lower() if gi is not None and gi < len(r2) else ""
            if g in ("in progress", "final") or g.startswith("final"):
                pf.started_ids.add(pid)
        break

    for rn, r in enumerate(rows[hi + 1:], start=hi + 2):
        eid = r[ei].strip() if ei < len(r) else ""
        if not eid.isdigit():
            continue
        if len(pf.entries) >= MAX_ENTRIES:
            pf.warnings.append(f"Only the first {MAX_ENTRIES} entries were read.")
            break

        def cell(i):
            return r[i].strip() if i is not None and i < len(r) else ""
        e = SwapEntry(key=eid, label=f"Entry {eid}", pairs=[], names=[], entry_id=eid,
                      contest_name=cell(ci) or None, contest_id=cell(cid) or None,
                      entry_fee=cell(fee) or None, cells=[], source={"kind": "upload", "row": rn})
        for k, (col, slot) in enumerate(zip(slot_cols, slots)):
            raw = cell(col)
            e.cells.append(raw)
            pid, nm, fmt, locked = _resolve_cell(raw, slot, idx)
            pf.formats[fmt] += 1
            if locked:
                e.dk_locked.add(k)
            if raw and pid is None:
                e.problems.append(f"{slot}: '{raw}' is not on this slate")
            e.pairs.append((slot, pid))
            e.names.append(nm or raw)
        pf.entries.append(e)
    if not pf.entries:
        raise SwapError("No entries with an Entry ID in this file.")
    return pf


def _resolve_cell(raw: str, slot: str, idx: SlateIndex) -> tuple[str | None, str | None, str, bool]:
    """(dk player id, name, format, locked flag) of one slot cell."""
    s = (raw or "").strip()
    if not s:
        return None, None, "empty", False
    locked = bool(_LOCKED.search(s))
    m = _CELL_ID.match(s)
    if m:
        did, name, fmt = m["id"], m["name"].strip(), "name_id"
    elif s.isdigit():
        did, name, fmt = s, None, "id"
    else:
        did, name, fmt = None, _LOCKED.sub("", s).strip(), "name"
    pid = None
    if did is not None:
        hit = idx.by_draftable.get(did)
        if hit is not None and hit["site_player_id"] is not None:
            pid = str(hit["site_player_id"])
        elif did in idx.by_player:
            pid = did
    if pid is None and name:
        ps = idx.by_name.get(_norm(name), set())
        if len(ps) == 1:
            pid = next(iter(ps))
    if pid is not None and not name:
        row = idx.row(pid, slot)
        name = row["player_name"] if row else None
    return pid, name, fmt, locked


# ============================================================================== results

@dataclass
class SlotChange:
    slot: str
    fixed: bool
    fixed_why: str | None
    before: dict | None              # {player_id, name, team, salary, mean, out}
    after: dict | None
    reason: str | None = None        # why the before player left

    @property
    def changed(self) -> bool:
        if self.after is None:                       # nothing proposed (no legal swap, ...)
            return False
        return (self.before or {}).get("player_id") != self.after.get("player_id")


@dataclass
class EntryResult:
    entry: SwapEntry
    status: str                      # swapped | unchanged | all_locked | no_legal | unusable
    objective: str
    slots: list[SlotChange]
    before_mean: float | None
    after_mean: float | None
    before_salary: int | None
    after_salary: int | None
    after_pairs: list[tuple[str, str]] | None
    notes: list[str] = field(default_factory=list)
    lineup: O.Lineup | None = None

    @property
    def delta(self) -> float | None:
        if self.before_mean is None or self.after_mean is None:
            return None
        return self.after_mean - self.before_mean

    @property
    def n_changed(self) -> int:
        return sum(1 for s in self.slots if s.changed)


@dataclass
class SwapPlan:
    slate_id: str
    now: datetime
    rules: R.Rules
    results: list[EntryResult]
    games: list[dict]                # {label, start, started}
    projection: dict
    ruled_out: dict[str, str]
    news: list[dict]
    portfolio: dict
    warnings: list[str]
    seconds: float
    source: str                      # saved | upload
    formats: Counter = field(default_factory=Counter)
    dk_ids: dict[str, str] = field(default_factory=dict)
    by_id: dict = field(default_factory=dict)
    contest: dict | None = None      # D8 contest-aware run info (switch ON only)

    @property
    def n_swapped(self) -> int:
        return sum(1 for r in self.results if r.status == "swapped")

    def to_csv(self, style: str = "name_id", only_changed: bool = False) -> str:
        """The DK edit CSV: one row per entry (all entries, so the file mirrors DK's), fixed and
        unchanged cells as DK wrote them."""
        rows = []
        for r in self.results:
            if only_changed and r.status != "swapped":
                continue
            e = r.entry
            cells = []
            pairs = r.after_pairs if r.status == "swapped" and r.after_pairs else None
            for k, slot in enumerate(e.pairs):
                orig = e.cells[k] if e.cells and k < len(e.cells) else None
                if pairs is None or pairs[k][1] == slot[1]:
                    if orig:                                  # unchanged: DK's own cell
                        cells.append({"cell": orig})
                        continue
                    pid = slot[1]
                else:
                    pid = pairs[k][1]
                cells.append(self._slot_dict(pairs[k][0] if pairs else slot[0], pid, e))
            rows.append({"entry_id": e.entry_id or "", "contest_name": e.contest_name or "",
                         "contest_id": e.contest_id or "", "entry_fee": e.entry_fee or "",
                         "slots": cells})
        return X.to_edit_csv(rows, header=list(self.rules.slot_names), style=style)

    def _slot_dict(self, slot: str, pid: str | None, entry: SwapEntry) -> dict:
        if pid is None:
            return {"cell": ""}
        p = self.by_id.get(pid)
        did = self.dk_ids.get(pid) or (p.draftable_ids.get(slot) if p else None)
        if did is None:
            for m in entry.slots_meta:
                if str(m.get("player_id")) == pid and m.get("draftable_id"):
                    did = m["draftable_id"]
        return {"slot": slot, "player_id": pid, "name": p.name if p else pid,
                "draftable_id": did}

    def summary(self) -> dict:
        return {"slate_id": self.slate_id, "now": iso(self.now), "source": self.source,
                "seconds": round(self.seconds, 2), "entries": len(self.results),
                "swapped": self.n_swapped,
                "status": dict(Counter(r.status for r in self.results)),
                "projection": {k: v for k, v in self.projection.items() if k != "p_play"},
                "ruled_out": len(self.ruled_out), "warnings": self.warnings,
                "portfolio": self.portfolio,
                **({"contest": self.contest} if self.contest is not None else {})}


# ============================================================================== checks

def check_entry(rules: R.Rules, by_id: Mapping[str, R.PoolPlayer], before: Sequence,
                fixed: Mapping[int, str], after: Sequence[tuple[str, str]], *,
                started: set[str], bans: set[str]) -> None:
    """Raise `SwapError` unless `after` keeps every fixed slot exactly, uses only open players
    in open slots, and is a legal DK lineup (slots, eligibility, cap, games, teams)."""
    if [s for s, _ in after] != [s for s, _ in before]:
        raise SwapError("slot order changed")
    for k, pid in fixed.items():
        if after[k] != (before[k][0], pid):
            raise SwapError(f"fixed slot {before[k][0]} changed ({pid} -> {after[k][1]})")
    fixed_ids = set(fixed.values())
    ids = [pid for _, pid in after]
    if len(set(ids)) != len(ids):
        raise SwapError("a player appears twice")
    for k, (slot, pid) in enumerate(after):
        if k in fixed:
            continue
        if pid in started or pid in fixed_ids:
            raise SwapError(f"{slot}: a player from a started game was put in an open slot")
        if pid in bans:
            raise SwapError(f"{slot}: a ruled-out player was picked")
        p = by_id.get(pid)
        if p is None or slot not in p.slots:
            raise SwapError(f"{slot}: player {pid} is not eligible")
    sal = sum(by_id[pid].salary_for(s, rules) for s, pid in after)
    if sal > rules.salary_cap:
        raise SwapError(f"salary {sal} over the cap")
    if rules.min_teams and len({by_id[p].team for p in ids}) < rules.min_teams:
        raise SwapError("too few teams")
    if rules.min_games and len({O.game_key(by_id[p]) for p in ids}) < rules.min_games:
        raise SwapError("too few games")


def _legal(rules, by_id, pairs) -> bool:
    try:
        check_entry(rules, by_id, pairs, {}, pairs, started=set(), bans=set())
        return True
    except (SwapError, KeyError):
        return False


# ============================================================================== the solve

def _synth(row: Mapping, slot: str) -> R.PoolPlayer:
    """A pool player from a stored draftable row (a fixed player DK has since greyed out)."""
    return R.PoolPlayer(
        dk_player_id=str(row["site_player_id"]), dk_dk_id=row.get("site_player_dk_id"),
        name=row["player_name"], team=row.get("team"),
        positions=R.split_positions(row.get("positions")), salary=int(row.get("salary") or 0),
        slots=(slot,), salary_by_slot={slot: int(row.get("salary") or 0)},
        draftable_ids={slot: str(row["draftable_id"])}, game_id=row.get("game_id"),
        competition_id=row.get("competition_id"), game_label=row.get("game_label"),
        home=row.get("home"), away=row.get("away"), start_time=row.get("start_time"),
        status=R._status(row.get("status")), news_status=None, starter=None, swappable=None,
        fppg=row.get("fppg"))


def _with_slot(p: R.PoolPlayer, slot: str, row: Mapping | None) -> R.PoolPlayer:
    sal = dict(p.salary_by_slot)
    dids = dict(p.draftable_ids)
    sal.setdefault(slot, int(row["salary"]) if row and row.get("salary") else p.salary)
    if row and row.get("draftable_id"):
        dids.setdefault(slot, str(row["draftable_id"]))
    return dataclasses.replace(p, slots=tuple(p.slots) + (slot,), salary_by_slot=sal,
                               draftable_ids=dids)


def _keep_slots(rules, by_id, fixed: dict[int, str], before, chosen: list[tuple[str, str]],
                ) -> list[tuple[str, str]]:
    """Re-seat the chosen open players so that as many as possible keep their old slot (fewer
    cells to edit on DK). Fixed slots are untouched; the solved seating is the fallback."""
    open_k = [k for k in range(len(before)) if k not in fixed]
    players = [pid for k, (s, pid) in enumerate(chosen) if k not in fixed]
    old = {before[k][1]: before[k][0] for k in open_k if before[k][1]}
    m = cp_model.CpModel()
    x = {}
    for pid in players:
        for k in open_k:
            if before[k][0] in by_id[pid].slots:
                x[pid, k] = m.NewBoolVar(f"x{pid}_{k}")
    for pid in players:
        m.AddExactlyOne(v for (p, _), v in x.items() if p == pid)
    for k in open_k:
        m.AddExactlyOne(v for (_, kk), v in x.items() if kk == k)
    m.Add(sum(by_id[pid].salary_for(before[k][0], rules) * v for (pid, k), v in x.items())
          + sum(by_id[pid].salary_for(before[k][0], rules) for k, pid in fixed.items())
          <= rules.salary_cap)
    m.Maximize(sum(v for (pid, k), v in x.items() if old.get(pid) == before[k][0]
                   and before[k][1] == pid) * 2
               + sum(v for (pid, k), v in x.items() if old.get(pid) == before[k][0]))
    s = cp_model.CpSolver()
    s.parameters.num_workers = 1
    s.parameters.max_time_in_seconds = 5.0
    if s.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return chosen
    out = list(chosen)
    for (pid, k), v in x.items():
        if s.Value(v):
            out[k] = (before[k][0], pid)
    return out




def _seat(before: Sequence, fixed: Mapping[int, str], lu: O.Lineup) -> list[tuple[str, str]]:
    """The solved lineup in the entry's column order: fixed columns keep their player, the open
    columns take the solved players of the same slot name."""
    fixed_ids = set(fixed.values())
    queues: dict[str, list[str]] = defaultdict(list)
    for slot, pid in lu.slot_pairs():
        if pid not in fixed_ids:
            queues[slot].append(pid)
    out = []
    for k, (slot, _) in enumerate(before):
        out.append((slot, fixed[k]) if k in fixed else (slot, queues[slot].pop(0)))
    return out


def _no_good(model: O.LineupModel, pairs, min_diff: int) -> None:
    """At least `min_diff` players different from `pairs` (players outside the model cannot be
    picked anyway, so the bound is on the ones that can)."""
    ys = [model.y[pid] for _, pid in pairs if pid in model.y]
    if ys:
        model.model.Add(sum(ys) <= model.rules.n_slots - max(1, int(min_diff)))


def _ok_vs_others(pairs, others, n_slots: int, min_diff: int, max_overlap: int | None) -> bool:
    ids = {pid for _, pid in pairs}
    for _, o in others:
        shared = len(ids & {pid for _, pid in o})
        if shared > n_slots - max(1, int(min_diff)):
            return False
        if max_overlap is not None and shared > max_overlap:
            return False
    return True


def _objective(rules, scores, pairs) -> float:
    return sum(scores.get(pid, 0.0) * rules.slot_multiplier(s) for s, pid in pairs if pid)


def _result(e: SwapEntry, status: str, objective: str, fixed: Mapping[int, str],
            fixed_why: Mapping[int, str], after, info, rules, *, notes=None, out_reason=None,
            news_by=None, lineup=None) -> EntryResult:
    out_reason, news_by = out_reason or {}, news_by or {}
    before_ids = [pid for _, pid in e.pairs]
    after_slot = {pid: s for s, pid in (after or [])}
    slots = []
    for k, (slot, pid) in enumerate(e.pairs):
        b = info(pid, slot) if pid else None
        if b is None and k < len(e.names) and e.names[k]:
            b = {"player_id": None, "name": e.names[k], "team": None, "salary": None,
                 "mean": None, "proj": None, "out": False, "status": None}
        a = info(after[k][1], slot) if after else None
        reason = None
        if after and after[k][1] != pid:
            if pid is None:
                reason = "Empty slot"
            elif pid in after_slot:
                reason = f"Moved to {after_slot[pid]}"
            elif pid in out_reason:
                reason = out_reason[pid]
            elif pid in news_by:
                reason = news_reason(news_by[pid]) or "Late news"
            else:
                reason = "Higher projection elsewhere"
        slots.append(SlotChange(slot, k in fixed, fixed_why.get(k), b, a, reason))

    def total(rows, key):
        vals = [(r or {}).get(key) for r in rows]
        return None if not rows or any(v is None for v in vals) else sum(vals)
    bm = total([s.before for s in slots], "mean")
    am = total([s.after for s in slots], "mean") if after else None
    bs = total([s.before for s in slots], "salary")
    as_ = total([s.after for s in slots], "salary") if after else None
    del before_ids
    return EntryResult(e, status, objective, slots, bm, am, bs, as_,
                       list(after) if after else None, list(notes or []), lineup)


def portfolio_report(entries: Sequence[SwapEntry], results: Sequence[EntryResult],
                     limit_n: int | None, max_overlap: int | None) -> dict:
    """Exposure and overlap of the entry set before and after the swap."""
    n = max(len(entries), 1)
    before, after, fixed = Counter(), Counter(), Counter()
    for r in results:
        e = r.entry
        for k, (_, pid) in enumerate(e.pairs):
            if pid:
                before[pid] += 1
            if any(s.fixed for s in r.slots[k:k + 1]) and pid:
                fixed[pid] += 1
        for _, pid in (r.after_pairs if r.status == "swapped" else e.pairs) or []:
            if pid:
                after[pid] += 1
    exp = [{"player_id": pid, "before": before[pid] / n, "after": after[pid] / n,
            "fixed": fixed[pid]} for pid in set(before) | set(after)]
    exp.sort(key=lambda d: (-d["after"], -d["before"], d["player_id"]))
    groups: dict[str, list[EntryResult]] = defaultdict(list)
    for r in results:
        groups[r.entry.group].append(r)

    def overlap(rs, use_after):
        best, dupes = 0, 0
        sets = []
        for r in rs:
            pairs = r.after_pairs if use_after and r.status == "swapped" else r.entry.pairs
            sets.append(frozenset(pid for _, pid in pairs if pid))
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                sh = len(sets[i] & sets[j])
                best = max(best, sh)
                dupes += sets[i] == sets[j]
        return best, dupes
    by_group = {}
    for g, rs in groups.items():
        if len(rs) < 2:
            continue
        b, db = overlap(rs, False)
        a, da = overlap(rs, True)
        by_group[g] = {"entries": len(rs), "max_shared_before": b, "max_shared_after": a,
                       "identical_before": db, "identical_after": da}
    out = {"exposure": exp[:12], "groups": by_group, "max_exposure_n": limit_n,
           "max_overlap": max_overlap}
    if limit_n is not None:
        out["over_cap"] = [d["player_id"] for d in exp if d["after"] * n > limit_n]
        out["fixed_over"] = {pid: c for pid, c in fixed.items() if c > limit_n}
    return out


def plan(conn: sqlite3.Connection, slate_id, *, now: datetime | None = None,
         entries: list[SwapEntry] | None = None, csv_data: bytes | str | None = None,
         scope: str = "all", objective: str = "auto", min_diff: int = 1,
         max_overlap: int | None = None, max_exposure: float | None = None,
         min_gain: float = MIN_GAIN, time_limit_s: float = TOTAL_LIMIT_S,
         projections: Mapping | None = None, extra_out: Mapping[str, str] | None = None,
         contest_aware: bool | None = None, locked_points: Mapping[str, float] | None = None,
         run_id: str | None = None, sim_rows=None, seed: int = 0) -> SwapPlan:
    """Late-swap proposals for a slate's entries at `now` (see the module docstring).

    `entries` (default: our saved lineups) or `csv_data` (DK's entries CSV). `scope` 'all'
    re-solves every entry, 'news' only those with a ruled-out or late-news player in an open
    slot. `objective` 'auto' (cash for our cash lineup and cash-style contests, else mean).
    `projections` ({dk id: Projection | float}) replaces the stored ones and `extra_out`
    ({dk id: reason}) rules more players out (tests, what-ifs).
    `contest_aware` (None: the COURTSIDE_SWAP_CONTEST switch, `swap_contest`, default OFF):
    choose each entry's open slots by its contest (D8). ON only: `locked_points` {dk id: points}
    of started players (default none: their sims), `sim_rows` the stored sim rows it may use
    (replays), `seed`. `run_id` pins the projection run (default the newest).
    Read-only: nothing is written to the database."""
    import numpy as np

    from . import runner
    from . import swap_contest as SC
    t0 = time.perf_counter()
    now = _utc(now)
    sid = str(slate_id)
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    slate = runner._slate(conn, sid)
    if slate is None:
        raise SwapError(f"There is no slate {sid}.")
    pool = R.build_pool(conn, sid)
    rules = pool.rules
    if rules is None:
        raise SwapError(f"No DraftKings rules stored for slate {sid}.")
    if runner.is_showdown(slate) or rules.is_showdown or rules.allow_late_swap is False:
        raise SwapError("Showdown has no late swap on DraftKings: lineups lock at the start.")
    tips = runner.slate_tips(conn, sid)
    idx = slate_index(conn, sid)
    warnings: list[str] = []
    formats: Counter = Counter()
    dk_ids: dict[str, str] = {}
    dk_started: set[str] = set()
    source = "saved"
    if csv_data is not None:
        pf = parse_entries_csv(csv_data, idx, rules)
        if pf.showdown:
            raise SwapError("Showdown has no late swap on DraftKings: lineups lock at the start.")
        entries, formats, dk_ids, dk_started = pf.entries, pf.formats, pf.dk_ids, pf.started_ids
        warnings += pf.warnings
        source = "upload"
    elif entries is None:
        entries = saved_entries(conn, sid)
    if projections is not None:
        projs = {str(k): O.as_projection(v) for k, v in projections.items()}
        pinfo = {"run_id": None, "model_version": "given", "created_at": None,
                 "source": "given", "p_play": {}}
    else:
        projs, pinfo = load_projections(conn, sid, pool, run_id=run_id)
    projs = dict(projs)
    news = slate_news(conn, sid, now)
    out_reason = ruled_out(conn, slate, pool, tips, now, news, pinfo["p_play"])
    out_reason.update({str(k): v for k, v in (extra_out or {}).items()})
    news_by = {str(n["player_id"]): n for n in news}

    players: dict[str, R.PoolPlayer] = {p.dk_player_id: p for p in pool.players}
    started = {pid for pid, p in players.items() if p.is_locked(now)}
    started_games = {O.game_key(players[p]) for p in dk_started if p in players}
    started |= {pid for pid, p in players.items() if O.game_key(p) in started_games}

    def start_of(pid: str):
        p = players.get(pid)
        if p is not None:
            return p.start
        row = idx.row(pid)
        return R._parse_ts(row["start_time"]) if row else None

    # ---- fixed slots per entry (players DK greyed out are added back so they can stay put)
    fixed_of: dict[str, dict[int, str]] = {}
    fixed_why: dict[str, dict[int, str]] = {}
    for e in entries:
        fx, why = {}, {}
        for k, (slot, pid) in enumerate(e.pairs):
            if pid is None:
                continue
            st = start_of(pid)
            if pid in started or (st is not None and st <= now):
                fx[k], why[k] = pid, "Game started"
            elif k in e.dk_locked or pid in dk_started:
                fx[k], why[k] = pid, "DraftKings shows it locked"
            else:
                continue
            p = players.get(pid)
            if p is None:
                row = idx.row(pid, slot)
                if row is None:
                    e.problems.append(f"{slot}: locked player {pid} is unknown")
                    continue
                players[pid] = _synth(row, slot)
            elif slot not in p.slots:
                players[pid] = _with_slot(p, slot, idx.row(pid, slot))
        fixed_of[e.key], fixed_why[e.key] = fx, why
    for fx in fixed_of.values():
        for pid in fx.values():
            st = start_of(pid)
            if st is not None and st <= now:
                started.add(pid)
            if pid not in projs:
                p = players[pid]
                projs[pid] = O.Projection(p.fppg if p.fppg is not None else 0.0)
    plist = list(players.values())
    by_id = players
    score_tab = {m: O.player_scores(plist, projs, mode=m, lam=CASH_LAMBDA)
                 for m in ("cash", "mean")}
    means = score_tab["mean"][1]

    def info(pid: str | None, slot: str) -> dict | None:
        if pid is None:
            return None
        p = by_id.get(pid)
        row = idx.row(pid, slot) if p is None else None
        mean = means.get(pid)
        out = pid in out_reason
        return {"player_id": pid, "name": p.name if p else (row["player_name"] if row else pid),
                "team": p.team if p else (row["team"] if row else None),
                "salary": p.salary_for(slot, rules) if p is not None else
                (int(row["salary"]) if row and row.get("salary") else None),
                "mean": 0.0 if out and mean is not None else mean, "proj": mean, "out": out,
                "status": p.status if p else None}

    # ---- identical entries (same lineup, same fixed slots, same objective and group) solve once
    groups: dict[tuple, list[SwapEntry]] = {}
    for e in entries:
        mode = objective_for(e.contest_name, e.mode) if objective == "auto" else objective
        gk = (tuple(e.pairs), tuple(sorted(fixed_of[e.key].items())), mode, e.group)
        groups.setdefault(gk, []).append(e)
    limit_n = None if max_exposure is None else max(0, floor(max_exposure * len(entries) + 1e-9))
    finals: dict[tuple, list[tuple[str, str]]] = {}
    results: dict[str, EntryResult] = {}

    def needs(e: SwapEntry) -> bool:
        fx = fixed_of[e.key]
        return any(k not in fx and (pid is None or pid in out_reason or pid in news_by
                                    or pid not in by_id or pid in started)
                   for k, (_, pid) in enumerate(e.pairs))

    def settle(gk, status, after, notes=(), lineup=None):
        finals[gk] = list(after) if after else list(gk[0])
        for x in groups[gk]:
            results[x.key] = _result(x, status, gk[2], fixed_of[x.key], fixed_why[x.key], after,
                                     info, rules, notes=list(notes), out_reason=out_reason,
                                     news_by=news_by, lineup=lineup)

    targets = []
    for gk, es in groups.items():
        e = es[0]
        if any("not on this slate" in p or "unknown" in p for p in e.problems):
            settle(gk, "unusable", None, e.problems)
        elif len(fixed_of[e.key]) == len(e.pairs):
            settle(gk, "all_locked", list(e.pairs))
        elif scope == "news" and not needs(e):
            settle(gk, "unchanged", list(e.pairs), ["No late news in its open slots."])
        else:
            targets.append(gk)

    def committed(skip) -> Counter:
        c: Counter = Counter()
        for g2, es in groups.items():
            if g2 == skip:
                continue
            ids = [pid for _, pid in finals[g2]] if g2 in finals else \
                list(fixed_of[es[0].key].values())
            for pid in ids:
                if pid:
                    c[pid] += len(es)
        return c

    deadline = t0 + time_limit_s
    ca, pending = None, []                    # D8 contest-aware swap (swap_contest; ON only)
    if (SC.contest_swap() if contest_aware is None else contest_aware) and targets:
        ca, why = SC.prepare(conn, sid, run_id or pinfo.get("run_id"), news=news,
                             pool_players=pool.players, locked_points=locked_points,
                             rows=sim_rows, seed=seed)
        if ca is None:
            warnings.append(f"Contest-aware swap is on but {why}: entries were re-solved by "
                            f"their mean.")
        else:
            deadline = max(deadline, ca.deadline)

    def make_model(sc_scores, mns, locks, free_bans, others, gk, es, use_cuts, use_caps, limit):
        model = O.LineupModel(plist, rules, sc_scores, mns, locks=locks, bans=free_bans,
                              now=now, time_limit_s=limit)
        if use_cuts:
            for _, pairs in others:
                _no_good(model, pairs, min_diff)
        if use_caps:
            if max_overlap is not None:
                for _, pairs in others:
                    ys = [model.y[p] for _, p in pairs if p in model.y]
                    if ys:
                        model.model.Add(sum(ys) <= max_overlap)
            if limit_n is not None:
                for pid, used in committed(gk).items():
                    if used + len(es) > limit_n and pid not in locks:
                        model.ban(pid)
        return model

    def finish(gk, lu, notes, *, mean_gain: bool = True):
        """Seat, gain check (mean objective), safety check and settle one target."""
        e = groups[gk][0]
        fx = fixed_of[e.key]
        before = list(e.pairs)
        scores = score_tab[gk[2]][0]
        mns = score_tab[gk[2]][1]
        others = [(g2, finals[g2]) for g2 in finals if g2[3] == gk[3] and g2 != gk]
        locks = {pid: before[k][0] for k, pid in fx.items()}
        chosen = before if lu is None else _keep_slots(rules, by_id, fx, before,
                                                       _seat(before, fx, lu))
        must = needs(e)
        if mean_gain and not must and _legal(rules, by_id, before) and _ok_vs_others(
                before, others, rules.n_slots, min_diff, max_overlap):
            gain = _objective(rules, scores, chosen) - _objective(rules, scores, before)
            if {p for _, p in chosen} == {p for _, p in before}:
                chosen = before
            elif gain < min_gain:
                notes.append(f"Kept: the best change gains {gain:.1f} points, under "
                             f"{min_gain:g}.")
                chosen = before
        if not mean_gain and not must and {p for _, p in chosen} == {p for _, p in before}:
            chosen = before
        if chosen == before:
            settle(gk, "unchanged", before, notes)
            return
        try:
            check_entry(rules, by_id, before, fx, chosen, started=started - set(locks),
                        bans=set(out_reason) - set(locks))
        except SwapError as err:        # never hand out a lineup that breaks a locked slot
            log.error("swap result for %s failed its check: %s", e.label, err)
            warnings.append(f"{e.label}: the proposed swap failed a safety check ({err}); "
                            f"left unchanged.")
            settle(gk, "no_legal", None, [f"Safety check failed: {err}."])
            return
        settle(gk, "swapped", chosen, notes, O.build_lineup(rules, by_id, chosen, mns, mns))

    for gk in targets:
        es = groups[gk]
        e = es[0]
        fx = fixed_of[e.key]
        mode = gk[2]
        scores, mns = score_tab[mode]
        before = list(e.pairs)
        locks = {pid: before[k][0] for k, pid in fx.items()}
        free_bans = (set(out_reason) | started) - set(locks)
        others = [(g2, finals[g2]) for g2 in finals if g2[3] == gk[3] and g2 != gk]
        if time.perf_counter() > deadline:
            settle(gk, "unchanged", before, ["Not re-solved: the 30-second budget ran out."])
            continue
        caps = max_overlap is not None or limit_n is not None
        levels = ([("caps", True, True)] if caps else []) + [("cuts", True, False),
                                                              ("none", False, False)]
        lu, dropped, level = None, [], None
        for name, use_cuts, use_caps in levels:
            left = max(1.0, deadline - time.perf_counter())
            try:
                model = make_model(scores, mns, locks, free_bans, others, gk, es, use_cuts,
                                   use_caps, min(SOLVE_LIMIT_S, left))
            except ValueError as err:
                log.warning("swap model for %s: %s", e.label, err)
                break
            lu = model.solve()
            if lu is not None:
                level = (use_cuts, use_caps)
                break
            dropped.append({"caps": "the overlap and exposure caps",
                            "cuts": "the different-lineup rule"}.get(name, ""))
        if lu is None:
            settle(gk, "no_legal", None, ["No legal lineup fills the open slots with the cap "
                                          "left."])
            continue
        notes = []
        if dropped:
            notes.append("Relaxed " + " and ".join(d for d in dropped if d) +
                         ": the locked players leave no other choice.")
        if ca is None:
            finish(gk, lu, notes)
            continue
        # ---- D8 ON: scenario candidates under the same constraints; the choice comes later
        cands, seen = [lu], {frozenset(lu.player_ids)}
        try:
            contest = ca.contest_for(e.contest_id, mode)
            rng = np.random.default_rng(seed + len(pending))
            for sc_scores in SC.scenario_scores(ca, ca.get(contest), scores, SC.N_SCEN, rng):
                if time.perf_counter() > deadline:
                    break
                try:
                    m2 = make_model(sc_scores, mns, locks, free_bans, others, gk, es, *level,
                                    min(SOLVE_LIMIT_S, max(1.0, deadline - time.perf_counter())))
                except ValueError:
                    break
                l2 = m2.solve()
                if l2 is not None and frozenset(l2.player_ids) not in seen:
                    seen.add(frozenset(l2.player_ids))
                    cands.append(l2)
        except Exception as err:  # noqa: BLE001 - a field / sims failure: today's answer
            log.exception("contest-aware candidates failed for %s", e.label)
            warnings.append(f"{e.label}: contest-aware swap failed ({type(err).__name__}); "
                            f"re-solved by its mean.")
            finish(gk, lu, notes)
            continue
        # provisional: later entries are cut against this one's mean solve, as OFF cuts them
        # against its final (the choice below replaces it)
        finals[gk] = _seat(before, fx, lu)
        stay = not any(k not in fx and (pid is None or pid in out_reason or pid not in by_id
                                        or pid in started) for k, (_, pid) in enumerate(before))
        pending.append((SC.Pending(gk=gk, mode="cash" if mode == "cash" else "mean",
                                   contest=contest, copies=len(es), before=before,
                                   keep_ok=stay and _legal(rules, by_id, before),
                                   cands=cands, group=gk[3]), notes))

    if ca is not None and pending:
        pend_gk = {it.gk for it, _ in pending}
        fixed_groups: dict[str, list] = defaultdict(list)
        for g2, es2 in groups.items():
            if g2 not in pend_gk:
                fixed_groups[g2[3]].append((finals.get(g2, list(g2[0])), len(es2)))
        try:
            dec = SC.decide(ca, [it for it, _ in pending], fixed_groups, n_slots=rules.n_slots,
                            min_diff=min_diff, max_overlap=max_overlap)
        except Exception as err:  # noqa: BLE001 - fall back to today's answer, never fail
            log.exception("contest-aware swap failed; mean re-solve used")
            warnings.append(f"Contest-aware swap failed ({type(err).__name__}); entries were "
                            f"re-solved by their mean.")
            dec = None
        for it, notes in pending:
            if dec is None:
                finish(it.gk, it.cands[0], notes)
                continue
            d = dec[it.gk]
            finish(it.gk, d["lineup"], notes + [SC.note(d)], mean_gain=False)
        ca.info["decisions"] = {str(groups[gk][0].key): {k: v for k, v in d.items()
                                                         if k != "lineup"}
                                for gk, d in (dec or {}).items()}

    res_list = [results[e.key] for e in entries]
    port = portfolio_report(entries, res_list, limit_n, max_overlap)
    for pid, n_fixed in (port.get("fixed_over") or {}).items():
        warnings.append(f"{_name_of(by_id, pid)} is locked into {n_fixed} entries, above the "
                        f"exposure cap. Locked slots cannot change.")
    games = []
    for g in sorted(pool.games.values(), key=lambda g: g.get("start_time") or ""):
        st = R._parse_ts(g.get("start_time"))
        key = g.get("competition_id") or g.get("game_id") or g.get("label") or ""
        games.append({"label": g.get("label"), "start": st,
                      "started": bool(st and st <= now) or key in started_games})
    return SwapPlan(slate_id=sid, now=now, rules=rules, results=res_list, games=games,
                    projection=pinfo, ruled_out=out_reason, news=news, portfolio=port,
                    warnings=warnings, seconds=time.perf_counter() - t0, source=source,
                    formats=formats, dk_ids=dk_ids, by_id=by_id,
                    contest=None if ca is None else {
                        "version": SC.VERSION, "run_id": ca.run_id, "rows": ca.rows_info,
                        "outs_known": sorted(ca.outs), "locked_points": len(ca.locked_points),
                        **ca.info})


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("swap", help="late swap: re-solve the unlocked slots of saved lineups or "
                                    "of a DK entries CSV and write a DK edit CSV")
    p.add_argument("--slate", required=True, help="draft group id (Classic)")
    p.add_argument("--entries", default=None, help="DK entries CSV (Edit Entries download); "
                                                    "default: our saved lineups")
    p.add_argument("--now", default=None, help="ISO time to swap as of (default: now)")
    p.add_argument("--scope", choices=SCOPES, default="all",
                   help="all entries, or only those with late news in an open slot")
    p.add_argument("--objective", choices=OBJECTIVES, default="auto")
    p.add_argument("--min-diff", type=int, default=1, help="entries of a contest differ by >= d")
    p.add_argument("--max-overlap", type=int, default=None, help="max shared players per pair")
    p.add_argument("--max-exposure", type=float, default=None, help="max share of entries, 0-1")
    p.add_argument("--min-gain", type=float, default=MIN_GAIN,
                   help=f"change an entry only for this many points (default {MIN_GAIN})")
    p.add_argument("--fmt", choices=X.STYLES, default="name_id", help="new cells: Name (ID) or id")
    p.add_argument("--out", default=None, help="CSV path (default var/exports/...)")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.add_argument("--contest-aware", action="store_true", default=None,
                   help="choose open slots by each entry's contest (D8; default: the "
                        "COURTSIDE_SWAP_CONTEST switch, off)")
    return {"swap": _swap_cmd}


def _swap_cmd(args) -> int:
    from pathlib import Path

    from ... import config, db
    conn = db.connect()
    now = None
    if args.now:
        now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    data = Path(args.entries).read_bytes() if args.entries else None
    try:
        sp = plan(conn, args.slate, now=now, csv_data=data, scope=args.scope,
                  objective=args.objective, min_diff=args.min_diff, max_overlap=args.max_overlap,
                  max_exposure=args.max_exposure, min_gain=args.min_gain,
                  contest_aware=getattr(args, "contest_aware", None))
    except SwapError as e:
        print(f"error: {e}")
        return 1
    if not sp.results:
        print("no saved lineups for this slate (and no --entries file)")
        return 1
    out = Path(args.out) if args.out else (
        config.data_dir() / "exports" / f"dk_swap_{sp.slate_id}_{sp.now:%Y%m%dT%H%M%SZ}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sp.to_csv(style=args.fmt), encoding="utf-8")
    if args.json:
        print(json.dumps({**sp.summary(), "csv": str(out)}, indent=2, default=str))
        return 0
    print(f"slate {sp.slate_id} as of {iso(sp.now)}: {len(sp.results)} entries, "
          f"{sp.n_swapped} swapped, {sp.seconds:.1f} s; projections "
          f"{sp.projection.get('model_version')} ({sp.projection.get('source')})")
    print("games: " + ", ".join(f"{g['label']} {'started' if g['started'] else 'open'}"
                                for g in sp.games))
    for r in sp.results:
        d = f"{r.delta:+.1f}" if r.delta is not None else "n/a"
        print(f"  {r.entry.label:<18} {r.status:<10} {r.n_changed} slot(s) changed, "
              f"projection {d}")
        for s in r.slots:
            if s.changed or s.fixed:
                b = (s.before or {}).get("name") or "-"
                a = (s.after or {}).get("name") or "-"
                tag = "LOCKED" if s.fixed else "      "
                line = f"      {s.slot:<5}{tag} {b}" + (f" -> {a}" if s.changed else "")
                print(line + (f"   ({s.reason})" if s.changed and s.reason else ""))
        for n in r.notes:
            print(f"      note: {n}")
    for w in sp.warnings:
        print(f"warning: {w}")
    print(f"DK edit CSV: {out}")
    return 0
