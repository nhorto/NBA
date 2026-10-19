"""Real DraftKings results: contest standings CSVs, the owner's own entries (docs/PLAYERS.md §7.4,
§8 Phase 3, task T3.2).

Everything here is REAL data the owner uploads or types in. It never mixes with simulated or
backtested numbers: the tables written are ``dfs_uploads``, ``dfs_contest_results``,
``dfs_contest_ownership`` (the field) and ``dfs_entries`` (the owner's own entries, ``source``
'upload' or 'manual'). The grader (`grade.py`) reads ``dfs_contest_results`` for real cash lines;
`my_results` is the real P&L.

Input: the file an entrant downloads after a contest ("Export lineups to CSV"), usually named
``contest-standings-<contest id>.csv`` (a ``.zip`` of them is accepted too)::

    Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
    1,4601234567,someuser (1/20),0,321.5,"C Joel Embiid F LeBron James G Tyrese Maxey ...",,Joel Embiid,C/PF,45.2%,63.5

The entry block (columns up to ``Lineup``) and the player block (``Player`` onwards) share rows
but are independent; either may be longer. ``Lineup`` is a run of ``SLOT name`` pairs (slots PG SG
SF PF C G F UTIL; Showdown CPT UTIL), or ``LOCKED`` while a contest has not started.

What is stored (one standings file = one contest)

* archive first: the bytes go to ``var/raw/draftkings/uploads/`` (sha256-named) before anything is
  parsed; ``dfs_uploads.sha256`` is UNIQUE, so the same file twice is a no-op ("duplicate").
* ``dfs_contest_results``: one row per ranked entry. ``entry_hash`` = sha256(EntryId)[:24];
  ``points`` and ``rank`` as in the file; ``lineup_text`` the Lineup cell; ``payout`` = the contest's
  payout table at that rank (``dfs_contests.payout_json``, [{min_pos, max_pos, payout}]) when
  known, else NULL. Entry NAMES of other users are never stored (only the owner's, in
  ``dfs_entries``, matched by the configured DK username). Re-uploading a contest with a newer
  file replaces its rows.
* ``dfs_contest_ownership``: one row per NBA player, ``pct_drafted`` in PERCENT (45.2 = 45.2%) and
  ``fpts``; Showdown stores the UTIL row (the CPT percentages are kept in the upload's
  ``detail``). Names without an NBA id are listed in ``detail.unmatched`` (and shown on the page).
* ``dfs_contests``: a row is created when the contest was never captured (slate inferred from the
  players when possible), so `grade.contest_results` finds the contest through its slate.
* Final standings only: a file whose entries still have time remaining is refused (a partial
  cash line would pollute the grades).

Derived per contest (`contest_summary`): field size, winning score, top-1% score, median, the
REAL CASH LINE (score at the last paid place of the payout table; falls back to the lowest paid
row), and the ownership distribution.

Owner detection: a configured DK username (``COURTSIDE_DK_USERNAME``, or the file
``<data>/dk_username.txt``, comma separated for several) is matched against ``EntryName`` with the
" (3/20)" multi-entry suffix removed, case-insensitively. Their rows become ``dfs_entries``
(fee from ``dfs_contests``, payout from the table, ``lineup_id`` when the player set equals one of
our stored lineups of that slate). `add_manual_entry` and `import_history` (DK's Contest History
CSV) cover entries without a standings file.

CLI (register ``courtside.proj.dfs.standings`` in ``DFS_MODULES`` of ``proj/cli.py``)::

    courtside dfs results-import FILE [--contest-id N] [--username NAME] [--slate S]
    courtside dfs results-list
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ... import archive, config

log = logging.getLogger("courtside.proj.dfs.standings")

MAX_UPLOAD_BYTES = 60 * 1024 * 1024           # one file / one zip
MAX_UNZIPPED_BYTES = 250 * 1024 * 1024
MAX_ZIP_MEMBERS = 24
CLASSIC_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
SLOT_TOKENS = frozenset(CLASSIC_SLOTS) | {"CPT"}
SLATE_COVER_MIN = 0.95          # share of the observed players a slate pool must contain
USERNAME_ENV = "COURTSIDE_DK_USERNAME"
USERNAME_FILE = "dk_username.txt"
MAX_NAME_LEN = 120


class StandingsError(ValueError):
    """The upload cannot be used; the message is shown to the owner."""


# ============================================================================== small helpers

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _loads(s, default=None):
    if s is None or s == "":
        return default
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return default


def _num(x) -> float | None:
    """'1,234.5' / '$12.00' / '45.2%' / '' -> float or None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip().replace(",", "").replace("$", "").replace("%", "").strip()
        if not s or s.upper() in ("N/A", "NA", "-", "--"):
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    return v if math.isfinite(v) else None


def _int(x) -> int | None:
    v = _num(x)
    return None if v is None else int(v)


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def entry_hash(entry_id) -> str:
    return hashlib.sha256(str(entry_id).strip().encode()).hexdigest()[:24]


def _safe_name(name: str | None) -> str:
    base = os.path.basename((name or "upload").replace("\\", "/")) or "upload"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:80].strip("._") or "upload"
    return base


def _norm(name) -> str:
    from .. import ids as IDS
    return IDS.norm_name(name)


def _root(data_dir) -> Path:
    return Path(data_dir) if data_dir else config.data_dir()


def upload_dir(data_dir=None) -> Path:
    return _root(data_dir) / "raw" / "draftkings" / "uploads"


# ============================================================================== owner username

def owner_usernames(data_dir=None) -> list[str]:
    """Configured DK usernames: $COURTSIDE_DK_USERNAME, else ``<data>/dk_username.txt``
    (comma separated)."""
    raw = os.environ.get(USERNAME_ENV)
    if not raw:
        p = _root(data_dir) / USERNAME_FILE
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            raw = ""
    return [u.strip() for u in re.split(r"[,\n]", raw or "") if u.strip()]


def set_owner_username(name: str, data_dir=None) -> None:
    """Remember the owner's DK username (the upload form's "remember" box)."""
    name = (name or "").strip()[:MAX_NAME_LEN]
    if not name or "\n" in name:
        raise StandingsError("Enter your DraftKings username.")
    root = _root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / USERNAME_FILE).write_text(name + "\n", encoding="utf-8")


_MULTI = re.compile(r"\s*\(\s*\d+\s*/\s*\d+\s*\)\s*$")


def base_username(entry_name: str | None) -> str:
    """'someuser (3/20)' -> 'someuser' (casefolded)."""
    return _MULTI.sub("", (entry_name or "").strip()).strip().casefold()


# ============================================================================== parsing

@dataclass
class EntryRow:
    rank: int | None
    entry_id: str
    entry_name: str
    time_remaining: float | None
    points: float | None
    lineup: str | None            # None when empty / LOCKED


@dataclass
class PlayerRow:
    name: str
    position: str
    pct_drafted: float | None     # percent points (45.2 = 45.2%)
    fpts: float | None


@dataclass
class Standings:
    filename: str
    contest_id: str | None
    entries: list[EntryRow] = field(default_factory=list)
    players: list[PlayerRow] = field(default_factory=list)
    game_type: str = "classic"
    locked: int = 0               # entries whose lineup is hidden (LOCKED)
    skipped: int = 0              # entry rows without a rank
    live: bool = False            # entries with time remaining
    warnings: list[str] = field(default_factory=list)


def contest_id_from_filename(filename: str | None) -> str | None:
    """'contest-standings-196456028.csv' -> '196456028' (first run of >= 6 digits)."""
    m = re.search(r"(\d{6,})", os.path.basename(filename or ""))
    return m.group(1) if m else None


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def parse_lineup(text: str | None) -> list[tuple[str, str]] | None:
    """'C Joel Embiid F LeBron James ...' -> [('C', 'Joel Embiid'), ('F', 'LeBron James'), ...].
    None for empty / 'LOCKED' / text that does not start with a slot token."""
    s = (text or "").strip()
    if not s or s.upper() == "LOCKED":
        return None
    out: list[list[str]] = []
    for tok in s.split():
        if tok in SLOT_TOKENS and (not out or out[-1][1:]):
            out.append([tok])
        elif out:
            out[-1].append(tok)
        else:
            return None
    if any(len(o) < 2 for o in out):
        return None
    return [(o[0], " ".join(o[1:])) for o in out]


def _colmap(header: list[str]) -> dict[str, int]:
    names = {}
    want = {"rank": ("rank",), "entry_id": ("entryid", "entry id", "entry_id"),
            "entry_name": ("entryname", "entry name", "entry_name"),
            "time": ("timeremaining", "time remaining"), "points": ("points",),
            "lineup": ("lineup",), "player": ("player",),
            "position": ("roster position", "rosterposition", "position"),
            "pct": ("%drafted", "% drafted", "drafted"), "fpts": ("fpts", "fantasy points")}
    for i, h in enumerate(header):
        k = h.strip().lstrip("﻿").casefold()
        for key, alts in want.items():
            if k in alts and key not in names:
                names[key] = i
    return names


def sniff_kind(text: str) -> str:
    """'standings' | 'history' | 'unknown' from the header line."""
    head = text[:4000].casefold()
    first = head.splitlines()[0] if head.splitlines() else ""
    flat = re.sub(r"[\s_\"]", "", first)
    if "entryid" in flat and "rank" in flat:
        return "standings"
    if "entrykey" in flat and "contestkey" in flat:
        return "history"
    for line in head.splitlines()[:6]:       # header not on line 1
        f = re.sub(r"[\s_\"]", "", line)
        if "entryid" in f and "rank" in f:
            return "standings"
        if "entrykey" in f and "contestkey" in f:
            return "history"
    return "unknown"


def parse_standings(data: bytes | str, filename: str = "", contest_id: str | None = None
                    ) -> Standings:
    text = data if isinstance(data, str) else decode_text(data)
    rows = list(csv.reader(io.StringIO(text, newline="")))
    hi = next((i for i, r in enumerate(rows[:10])
               if any(c.strip().casefold() in ("entryid", "entry id", "entry_id") for c in r)), None)
    if hi is None:
        raise StandingsError("This does not look like a DraftKings contest standings file: "
                             "no EntryId column.")
    cm = _colmap(rows[hi])
    for need in ("entry_id", "points"):
        if need not in cm:
            raise StandingsError(f"The standings file has no {need.replace('_', ' ')} column.")
    st = Standings(filename=filename, contest_id=contest_id or contest_id_from_filename(filename))

    def cell(r, key):
        i = cm.get(key)
        return r[i].strip() if i is not None and i < len(r) else ""

    pct_has_sign = False
    pct_vals: list[float] = []
    seen_ids: set[str] = set()
    for r in rows[hi + 1:]:
        if not any(c.strip() for c in r):
            continue
        eid = cell(r, "entry_id")
        if eid:
            if eid in seen_ids:
                st.warnings.append(f"duplicate EntryId {eid} ignored")
            else:
                seen_ids.add(eid)
                rank = _int(cell(r, "rank"))
                pts = _num(cell(r, "points"))
                lineup_raw = cell(r, "lineup")
                lineup = None if (not lineup_raw or lineup_raw.upper() == "LOCKED") else lineup_raw
                if lineup is None and lineup_raw:
                    st.locked += 1
                tr = _num(cell(r, "time"))
                if tr is not None and tr > 0:
                    st.live = True
                if rank is None or pts is None:
                    st.skipped += 1
                else:
                    st.entries.append(EntryRow(rank, eid, cell(r, "entry_name"), tr, pts, lineup))
        pname = cell(r, "player")
        if pname and "player" in cm:
            raw_pct = cell(r, "pct")
            pct_has_sign = pct_has_sign or "%" in raw_pct
            pv = _num(raw_pct)
            if pv is not None:
                pct_vals.append(pv)
            st.players.append(PlayerRow(pname, cell(r, "position"), pv, _num(cell(r, "fpts"))))
    if pct_vals and not pct_has_sign and max(pct_vals) <= 1.0:      # fractions, not percents
        for p in st.players:
            if p.pct_drafted is not None:
                p.pct_drafted *= 100.0
    if any(parse_lineup(e.lineup) and any(s == "CPT" for s, _ in parse_lineup(e.lineup))
           for e in st.entries[:50]) or any("CPT" in p.position.upper() for p in st.players[:50]):
        st.game_type = "showdown"
    if not st.entries:
        raise StandingsError(
            "No ranked entries in this file. Export the standings after the contest is over "
            "(\"Export lineups to CSV\" on the finished contest).")
    return st


# ============================================================================== contest metadata

def payout_at(table: list[dict], rank: int) -> float:
    for t in table or []:
        try:
            if int(t["min_pos"]) <= rank <= int(t["max_pos"]):
                return float(t.get("payout") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0


def contest_table(conn, contest_id: str) -> list[dict]:
    r = conn.execute("SELECT payout_json FROM dfs_contests WHERE contest_id=?",
                     (str(contest_id),)).fetchone()
    t = _loads(r[0], []) if r else []
    return [x for x in (t or []) if isinstance(x, dict) and "min_pos" in x and "max_pos" in x]


def paid_places(table: list[dict]) -> int:
    return max((int(t["max_pos"]) for t in table if (t.get("payout") or 0) > 0), default=0)


def _slate_pool(conn, slate_id) -> list[tuple[str, str, str | None]]:
    """[(dk player id, name, team)] over every capture of the slate."""
    out, seen = [], set()
    for r in conn.execute("SELECT DISTINCT site_player_id, player_name, team FROM dfs_salaries "
                          "WHERE slate_id=? AND site_player_id IS NOT NULL", (str(slate_id),)):
        k = (str(r[0]), r[1])
        if k not in seen:
            seen.add(k)
            out.append((str(r[0]), r[1], r[2]))
    return out


def infer_slate(conn, names: set[str], game_type: str) -> tuple[str | None, str]:
    """(slate_id, how): the SMALLEST slate pool (of the right type) that holds >= 95% of the
    observed player names. None when no slate does, or two equally small ones tie."""
    keys = {_norm(n) for n in names if n}
    keys.discard("")
    if len(keys) < 3:
        return None, "too few players to infer a slate"
    best: list[tuple[int, str]] = []
    for s in conn.execute("SELECT slate_id, tag, game_type_id FROM dfs_slates"):
        sd = s["tag"] == "showdown" or s["game_type_id"] == 81
        if sd != (game_type == "showdown"):
            continue
        pool = {_norm(n) for _, n, _ in _slate_pool(conn, s["slate_id"])}
        if not pool:
            continue
        if len(keys & pool) / len(keys) >= SLATE_COVER_MIN:
            best.append((len(pool), str(s["slate_id"])))
    if not best:
        return None, "no stored slate contains these players"
    best.sort()
    if len(best) > 1 and best[0][0] == best[1][0]:
        return None, "several slates fit these players"
    return best[0][1], "inferred from the players"


def resolve_players(conn, names, slate_id: str | None) -> dict[str, dict]:
    """{name: {'player_id': NBA id | None, 'dk_id', 'how'}} via the slate's DK pool
    (dfs_salaries -> player_ids), falling back to proj/ids name matching league-wide."""
    from .. import ids as IDS
    by_key: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    if slate_id:
        for dk, nm, team in _slate_pool(conn, slate_id):
            by_key[_norm(nm)].add((dk, team))
    dk2nba = IDS.nba_ids(conn, IDS.DK)
    league = {"pool": None, "aliases": None}

    def league_resolve(name: str, team: str):
        if league["pool"] is None:
            try:
                IDS.load_seed(conn)
                league["pool"] = IDS.current_pool(conn)
                league["aliases"] = IDS._aliases(conn, IDS.DK)
            except Exception as e:  # noqa: BLE001
                log.warning("league name pool unavailable (%s)", e)
                league["pool"] = False
        if not league["pool"]:
            return None, "no_pool"
        pid, how, _ = IDS.resolve_name(name, team or "", league["pool"], league["aliases"])
        return pid, how

    out: dict[str, dict] = {}
    for nm in dict.fromkeys(names):
        hits = by_key.get(_norm(nm), set())
        dks = {d for d, _ in hits}
        if len(dks) == 1:
            dk, team = next(iter(hits))
            nba = dk2nba.get(dk)
            how = "pool"
            if nba is None:
                nba, how = league_resolve(nm, IDS._team(team) if team else "")
            out[nm] = {"player_id": nba, "dk_id": dk, "how": how if nba else "unmatched"}
        else:
            nba, how = league_resolve(nm, "")
            out[nm] = {"player_id": nba, "dk_id": None, "how": how if nba else "unmatched"}
    return out


# ============================================================================== lineups

def our_lineup_for(conn, slate_id: str | None, names: list[tuple[str, str]]) -> int | None:
    """dfs_lineups.id of OUR stored lineup (any ok run of the slate, newest run first) whose
    player set equals `names` ([(slot, name)]); Showdown also needs the same captain."""
    if not slate_id or not names:
        return None
    want = sorted(_norm(n) for _, n in names)
    cpt = next((_norm(n) for s, n in names if s == "CPT"), None)
    for r in conn.execute(
            "SELECT l.id, l.slots_json FROM dfs_lineups l JOIN dfs_runs r ON r.run_id=l.run_id "
            "WHERE r.slate_id=? AND r.ok=1 ORDER BY r.created_at DESC, l.id DESC", (str(slate_id),)):
        slots = _loads(r["slots_json"], []) or []
        if sorted(_norm(s.get("name")) for s in slots) != want:
            continue
        if cpt is not None and next((_norm(s.get("name")) for s in slots
                                     if s.get("slot") == "CPT"), None) != cpt:
            continue
        return int(r["id"])
    return None


def _slots_json(pairs: list[tuple[str, str]], resolved: dict[str, dict]) -> str:
    return json.dumps([{"slot": s, "name": n, "player_id": (resolved.get(n) or {}).get("dk_id"),
                        "nba_player_id": (resolved.get(n) or {}).get("player_id")}
                       for s, n in pairs], separators=(",", ":"))


# ============================================================================== entries (owner)

def _upsert_entry(conn, *, contest_id, slate_id, dk_entry_id, lineup_id, slots_json, entry_fee,
                  points, rank, field_size, payout, source, note, now) -> int:
    """Insert or update on (contest_id, dk_entry_id); an existing fee / payout / lineup / note
    is never replaced by NULL. A manual entry (no EntryId) of the same player set is claimed."""
    row = conn.execute("SELECT entry_id FROM dfs_entries WHERE contest_id=? AND dk_entry_id=?",
                       (contest_id, dk_entry_id)).fetchone() if dk_entry_id else None
    if row is None and dk_entry_id and slots_json:
        want = sorted(_norm(s.get("name")) for s in (_loads(slots_json, []) or []))
        for r in conn.execute("SELECT entry_id, slots_json FROM dfs_entries WHERE contest_id=? "
                              "AND dk_entry_id IS NULL AND source='manual'", (contest_id,)):
            have = sorted(_norm(s.get("name")) for s in (_loads(r["slots_json"], []) or []))
            if have and have == want:
                row = r
                break
    if row is None:
        cur = conn.execute(
            "INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, lineup_id, slots_json, "
            "entry_fee, points, rank, field_size, payout, source, created_at, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (contest_id, slate_id, dk_entry_id, lineup_id, slots_json, entry_fee, points, rank,
             field_size, payout, source, now, note))
        return int(cur.lastrowid)
    conn.execute(
        "UPDATE dfs_entries SET slate_id=COALESCE(?, slate_id), dk_entry_id=COALESCE(dk_entry_id, ?), "
        "lineup_id=COALESCE(?, lineup_id), slots_json=COALESCE(?, slots_json), "
        "entry_fee=COALESCE(entry_fee, ?), points=COALESCE(?, points), rank=COALESCE(?, rank), "
        "field_size=COALESCE(?, field_size), payout=COALESCE(?, payout), "
        "note=COALESCE(note, ?) WHERE entry_id=?",
        (slate_id, dk_entry_id, lineup_id, slots_json, entry_fee, points, rank, field_size,
         payout, note, row[0]))
    return int(row[0])


def _apply_owner(conn, st: Standings, *, contest_id: str, slate_id: str | None,
                 usernames: list[str], resolved: dict[str, dict], table: list[dict],
                 field_size: int, now: str) -> list[dict]:
    """dfs_entries rows for the owner's rows of a standings file. Returns summaries."""
    want = {u.casefold() for u in usernames}
    if not want:
        return []
    fee_row = conn.execute("SELECT entry_fee FROM dfs_contests WHERE contest_id=?",
                           (contest_id,)).fetchone()
    fee = fee_row[0] if fee_row else None
    out = []
    for e in st.entries:
        if base_username(e.entry_name) not in want:
            continue
        pairs = parse_lineup(e.lineup)
        lineup_id = our_lineup_for(conn, slate_id, pairs) if pairs else None
        payout = payout_at(table, e.rank) if table else None
        eid = _upsert_entry(
            conn, contest_id=contest_id, slate_id=slate_id, dk_entry_id=e.entry_id,
            lineup_id=lineup_id, slots_json=_slots_json(pairs, resolved) if pairs else None,
            entry_fee=fee, points=e.points, rank=e.rank, field_size=field_size, payout=payout,
            source="upload", note=e.entry_name or None, now=now)
        out.append({"entry_id": eid, "dk_entry_id": e.entry_id, "rank": e.rank,
                    "points": e.points, "payout": payout, "lineup_id": lineup_id})
    return out


# ============================================================================== import

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive(data: bytes, sha: str, filename: str, data_dir=None) -> Path:
    d = upload_dir(data_dir)
    path = d / f"{sha[:16]}__{_safe_name(filename)}"
    have = sorted(d.glob(f"{sha[:16]}__*")) if d.is_dir() else []
    if have:                    # the same bytes under another name: keep the first copy
        return have[0]
    archive._atomic_write(path, data)
    return path


def _find_upload(conn, sha: str):
    return conn.execute("SELECT * FROM dfs_uploads WHERE sha256=?", (sha,)).fetchone()


def import_upload(conn: sqlite3.Connection, data: bytes, filename: str, *,
                  contest_id: str | None = None, slate_id: str | None = None,
                  owner_names: list[str] | None = None, allow_live: bool = False,
                  now: str | None = None, data_dir=None) -> dict:
    """Archive and import one uploaded file (standings CSV, DK contest-history CSV or a zip of
    CSVs). Returns {'status': 'ok' | 'duplicate', 'kind', 'upload_id', ...}; raises
    `StandingsError` with a readable message for unusable files. `owner_names` default to the
    configured DK username(s)."""
    if not data:
        raise StandingsError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise StandingsError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    now = now or _now()
    owners = owner_usernames(data_dir) if owner_names is None else [n for n in owner_names if n]
    sha = _sha(data)
    path = _archive(data, sha, filename, data_dir)  # archive first, whatever happens next
    if data[:2] == b"PK":
        return _import_zip(conn, data, filename, sha, path, contest_id=contest_id,
                           slate_id=slate_id, owners=owners, allow_live=allow_live, now=now,
                           data_dir=data_dir)
    text = decode_text(data)
    kind = sniff_kind(text)
    if kind == "history":
        return import_history(conn, text, filename=filename, sha=sha, raw_path=path, now=now)
    if kind != "standings":
        raise StandingsError("Unrecognised file. Upload the standings CSV from a finished "
                             "DraftKings contest (\"Export lineups to CSV\") or your Contest "
                             "History CSV.")
    return _import_standings(conn, text, filename, sha, path, contest_id=contest_id,
                             slate_id=slate_id, owners=owners, allow_live=allow_live, now=now)


def _import_zip(conn, data, filename, sha, path, *, contest_id, slate_id, owners, allow_live, now,
                data_dir=None):
    dup = _find_upload(conn, sha)
    if dup is not None:
        return {"status": "duplicate", "kind": "zip", "upload_id": dup["upload_id"],
                "filename": filename}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise StandingsError("The zip file is damaged.") from e
    infos = [i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".csv")
             and not os.path.basename(i.filename).startswith(".")]
    if not infos:
        raise StandingsError("The zip contains no CSV files.")
    if len(infos) > MAX_ZIP_MEMBERS:
        raise StandingsError(f"The zip holds more than {MAX_ZIP_MEMBERS} files.")
    if sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
        raise StandingsError("The zip expands to too much data.")
    members = []
    for i in infos:
        member = zf.read(i)
        try:
            members.append({"filename": os.path.basename(i.filename),
                            **import_upload(conn, member, os.path.basename(i.filename),
                                            contest_id=contest_id if len(infos) == 1 else None,
                                            slate_id=slate_id, owner_names=owners,
                                            allow_live=allow_live, now=now, data_dir=data_dir)})
        except StandingsError as e:
            members.append({"filename": os.path.basename(i.filename), "status": "error",
                            "error": str(e)})
    with conn:
        cur = conn.execute(
            "INSERT INTO dfs_uploads (kind, contest_id, filename, sha256, raw_path, rows, "
            "uploaded_at, detail) VALUES ('zip', NULL, ?, ?, ?, ?, ?, ?)",
            (filename, sha, str(path), len(members), now,
             json.dumps({"members": [{k: m.get(k) for k in ("filename", "status", "kind",
                                                             "contest_id", "error")}
                                     for m in members]})))
    return {"status": "ok", "kind": "zip", "upload_id": cur.lastrowid, "filename": filename,
            "members": members}


def _import_standings(conn, text, filename, sha, path, *, contest_id, slate_id, owners,
                      allow_live, now) -> dict:
    dup = _find_upload(conn, sha)
    st = None
    if dup is not None:
        res = {"status": "duplicate", "kind": "standings", "upload_id": dup["upload_id"],
               "contest_id": dup["contest_id"], "filename": filename}
        if owners and dup["contest_id"]:
            # the same file again, now with a username: pick up the owner's rows from the archive
            st = parse_standings(text, filename, dup["contest_id"])
            cid = str(dup["contest_id"])
            det = _loads(dup["detail"], {}) or {}
            sid = det.get("slate_id")
            resolved = resolve_players(conn, {n for e in st.entries if (p := parse_lineup(e.lineup))
                                              for _, n in p}, sid)
            with conn:
                mine = _apply_owner(conn, st, contest_id=cid, slate_id=sid, usernames=owners,
                                    resolved=resolved, table=contest_table(conn, cid),
                                    field_size=len(st.entries), now=now)
            res["owner_entries"] = len(mine)
        return res
    st = parse_standings(text, filename, contest_id)
    if not st.contest_id:
        raise StandingsError("Could not find the contest id in the file name. Enter it in the "
                             "form (it is the number in the contest URL).")
    if st.live and not allow_live:
        raise StandingsError("This contest is still in progress (entries have time remaining). "
                             "Upload the standings after the last game has finished.")
    cid = str(st.contest_id)
    meta = conn.execute("SELECT slate_id, entry_fee, field_size FROM dfs_contests "
                        "WHERE contest_id=?", (cid,)).fetchone()
    slate_how = None
    if slate_id:
        slate_how = "chosen on upload"
    elif meta and meta["slate_id"]:
        slate_id, slate_how = str(meta["slate_id"]), "contest was captured"
    names = {p.name for p in st.players}
    if not names:
        names = {n for e in st.entries[:2000] if (pl := parse_lineup(e.lineup)) for _, n in pl}
    if not slate_id:
        slate_id, slate_how = infer_slate(conn, names, st.game_type)
    all_names = set(names)
    for e in st.entries:
        if base_username(e.entry_name) in {o.casefold() for o in owners}:
            if pl := parse_lineup(e.lineup):
                all_names |= {n for _, n in pl}
    resolved = resolve_players(conn, all_names, slate_id)
    table = contest_table(conn, cid)
    n_entries = len(st.entries)

    res_rows = [(cid, e.rank, entry_hash(e.entry_id), e.points, e.lineup,
                 payout_at(table, e.rank) if table else None, now) for e in st.entries]
    own_rows, cpt, unmatched, dup_names = {}, {}, [], 0
    for p in st.players:
        r = resolved.get(p.name) or {}
        pid = r.get("player_id")
        is_cpt = "CPT" in p.position.upper()
        if pid is None:
            unmatched.append({"name": p.name, "pct": p.pct_drafted})
            continue
        if is_cpt and st.game_type == "showdown":
            cpt[str(pid)] = {"name": p.name, "pct": p.pct_drafted, "fpts": p.fpts}
            continue
        if pid in own_rows:
            dup_names += 1
            continue
        own_rows[pid] = (cid, int(pid), p.name, p.pct_drafted, p.fpts)

    with conn:
        if meta is None:
            conn.execute(
                "INSERT INTO dfs_contests (contest_id, slate_id, name, field_size, attrs_json, "
                "captured_at, raw_path) VALUES (?,?,?,?,?,?,?)",
                (cid, slate_id, None, n_entries, json.dumps({"source": "upload"}), now, str(path)))
        elif slate_id and not meta["slate_id"]:
            conn.execute("UPDATE dfs_contests SET slate_id=? WHERE contest_id=?", (slate_id, cid))
        conn.execute("DELETE FROM dfs_contest_results WHERE contest_id=?", (cid,))
        conn.executemany(
            "INSERT OR REPLACE INTO dfs_contest_results (contest_id, rank, entry_hash, points, "
            "lineup_text, payout, uploaded_at) VALUES (?,?,?,?,?,?,?)", res_rows)
        conn.execute("DELETE FROM dfs_contest_ownership WHERE contest_id=?", (cid,))
        conn.executemany("INSERT OR REPLACE INTO dfs_contest_ownership (contest_id, player_id, "
                         "name, pct_drafted, fpts) VALUES (?,?,?,?,?)", list(own_rows.values()))
        mine = _apply_owner(conn, st, contest_id=cid, slate_id=slate_id, usernames=owners,
                            resolved=resolved, table=table, field_size=n_entries, now=now)
        detail = {"entries": n_entries, "players": len(st.players), "ownership_rows": len(own_rows),
                  "unmatched": unmatched, "slate_id": slate_id, "slate_how": slate_how,
                  "game_type": st.game_type, "locked": st.locked, "skipped_unranked": st.skipped,
                  "warnings": st.warnings[:20], "final": not st.live,
                  "owner": {"configured": bool(owners), "matched": len(mine)},
                  "payout_table": bool(table), "cpt_pct_drafted": cpt or None,
                  "duplicate_player_rows": dup_names}
        cur = conn.execute(
            "INSERT INTO dfs_uploads (kind, contest_id, filename, sha256, raw_path, rows, "
            "uploaded_at, detail) VALUES ('standings', ?, ?, ?, ?, ?, ?, ?)",
            (cid, filename, sha, str(path), n_entries, now, json.dumps(detail)))
    return {"status": "ok", "kind": "standings", "upload_id": cur.lastrowid, "contest_id": cid,
            "filename": filename, "entries": n_entries, "ownership_rows": len(own_rows),
            "unmatched": unmatched, "slate_id": slate_id, "slate_how": slate_how,
            "game_type": st.game_type, "owner_entries": len(mine), "owner_configured": bool(owners),
            "locked": st.locked, "warnings": st.warnings[:20], "payout_table": bool(table)}


# ============================================================================== DK contest history

HISTORY_COLS = ("Sport", "Game_Type", "Entry_Key", "Entry", "Contest_Key", "Contest_Date_EST",
                "Place", "Points", "Winnings_Non_Ticket", "Winnings_Ticket", "Contest_Entries",
                "Entry_Fee", "Prize_Pool", "Places_Paid")


def import_history(conn, text: str, *, filename: str = "", sha: str | None = None,
                   raw_path=None, now: str | None = None) -> dict:
    """DraftKings "Contest History" CSV (columns Sport, Game_Type, Entry_Key, Entry,
    Contest_Key, Contest_Date_EST, Place, Points, Winnings_Non_Ticket, Winnings_Ticket,
    Contest_Entries, Entry_Fee, Prize_Pool, Places_Paid). All rows are the owner's own entries;
    only NBA rows are kept. The history's winnings are authoritative and replace a payout
    computed from a payout table. (Column names are from DK's export as we know it; unverified
    against a live file, hence the tolerant header handling.)"""
    now = now or _now()
    rows = list(csv.reader(io.StringIO(text, newline="")))
    hi = next((i for i, r in enumerate(rows[:10])
               if {"entry_key", "contest_key"} <= {re.sub(r"[\s]", "_", c.strip().casefold())
                                                   for c in r}), None)
    if hi is None:
        raise StandingsError("This does not look like a DraftKings contest history file.")
    hdr = {re.sub(r"[\s]", "_", c.strip().casefold()): i for i, c in enumerate(rows[hi])}

    def g(r, k):
        i = hdr.get(k.casefold())
        return r[i].strip() if i is not None and i < len(r) else ""

    entries, skipped = [], 0
    for r in rows[hi + 1:]:
        if not any(c.strip() for c in r):
            continue
        sport = g(r, "Sport")
        if sport and sport.upper() != "NBA":
            skipped += 1
            continue
        key, cid = g(r, "Entry_Key"), g(r, "Contest_Key")
        if not key or not cid:
            skipped += 1
            continue
        entries.append({
            "contest_id": cid, "dk_entry_id": key, "fee": _num(g(r, "Entry_Fee")),
            "points": _num(g(r, "Points")), "rank": _int(g(r, "Place")),
            "field": _int(g(r, "Contest_Entries")),
            "payout": _num(g(r, "Winnings_Non_Ticket")),
            "ticket": g(r, "Winnings_Ticket"), "label": g(r, "Entry"), "date": g(r, "Contest_Date_EST"),
            "game_type": g(r, "Game_Type")})
    if not entries:
        raise StandingsError("No NBA entries in the contest history file.")
    sha = sha or _sha(text.encode())
    path = raw_path or ""
    dup = _find_upload(conn, sha)
    if dup is not None:
        return {"status": "duplicate", "kind": "history", "upload_id": dup["upload_id"],
                "filename": filename}
    n_new = 0
    with conn:
        for e in entries:
            c = conn.execute("SELECT slate_id FROM dfs_contests WHERE contest_id=?",
                             (e["contest_id"],)).fetchone()
            before = conn.execute("SELECT entry_id FROM dfs_entries WHERE contest_id=? AND "
                                  "dk_entry_id=?", (e["contest_id"], e["dk_entry_id"])).fetchone()
            eid = _upsert_entry(
                conn, contest_id=e["contest_id"], slate_id=c[0] if c else None,
                dk_entry_id=e["dk_entry_id"], lineup_id=None, slots_json=None,
                entry_fee=e["fee"], points=e["points"], rank=e["rank"], field_size=e["field"],
                payout=None, source="upload", note=f"history: {e['label']}"[:200], now=now)
            if e["payout"] is not None:         # the history's winnings are authoritative
                conn.execute("UPDATE dfs_entries SET payout=?, entry_fee=COALESCE(?, entry_fee) "
                             "WHERE entry_id=?", (e["payout"], e["fee"], eid))
            n_new += before is None
        cur = conn.execute(
            "INSERT INTO dfs_uploads (kind, contest_id, filename, sha256, raw_path, rows, "
            "uploaded_at, detail) VALUES ('history', NULL, ?, ?, ?, ?, ?, ?)",
            (filename, sha, str(path), len(entries), now,
             json.dumps({"entries": len(entries), "new": n_new, "skipped": skipped})))
    return {"status": "ok", "kind": "history", "upload_id": cur.lastrowid, "filename": filename,
            "entries": len(entries), "new_entries": n_new, "skipped": skipped}


# ============================================================================== manual entries

def add_manual_entry(conn, *, contest_id: str, entry_fee: float | None = None,
                     points: float | None = None, rank: int | None = None,
                     payout: float | None = None, field_size: int | None = None,
                     lineup_id: int | None = None, lineup_text: str | None = None,
                     slate_id: str | None = None, note: str | None = None,
                     now: str | None = None) -> int:
    """A hand-entered contest entry (source 'manual'). `lineup_id` copies one of OUR stored
    lineups; `lineup_text` is pasted text ("C Joel Embiid F LeBron James ..." or one name per
    line). `payout` None means the winnings are not known yet (pending in the P&L)."""
    cid = re.sub(r"\D", "", str(contest_id or ""))
    if not cid:
        raise StandingsError("Enter the contest id (the number in the contest's address).")
    for nm, v in (("entry fee", entry_fee), ("payout", payout), ("points", points)):
        if v is not None and (not math.isfinite(v) or v < 0):
            raise StandingsError(f"The {nm} must be a number of zero or more.")
    if rank is not None and rank < 1:
        raise StandingsError("The rank must be 1 or more.")
    meta = conn.execute("SELECT slate_id, entry_fee, field_size FROM dfs_contests WHERE contest_id=?",
                        (cid,)).fetchone()
    slots_json, pairs = None, None
    if lineup_id:
        r = conn.execute("SELECT l.slots_json, r.slate_id FROM dfs_lineups l JOIN dfs_runs r "
                         "ON r.run_id=l.run_id WHERE l.id=?", (int(lineup_id),)).fetchone()
        if r is None:
            raise StandingsError("That lineup does not exist.")
        slots_json = r["slots_json"]
        slate_id = slate_id or r["slate_id"]
    elif lineup_text and lineup_text.strip():
        pairs = parse_lineup(" ".join(lineup_text.split()))
        if pairs is None:
            lines = [ln.strip() for ln in lineup_text.splitlines() if ln.strip()]
            if len(lines) >= 6 and all(len(ln.split()) <= 5 for ln in lines):
                pairs = [(None, ln) for ln in lines]
        if pairs is None:
            raise StandingsError("Could not read the lineup. Paste it like the DraftKings "
                                 "export (\"C Joel Embiid F LeBron James ...\") or one player "
                                 "per line.")
        slate_id = slate_id or (meta["slate_id"] if meta else None)
        resolved = resolve_players(conn, [n for _, n in pairs], slate_id)
        slots_json = json.dumps([{"slot": s, "name": n,
                                  "player_id": resolved[n]["dk_id"],
                                  "nba_player_id": resolved[n]["player_id"]} for s, n in pairs])
        lineup_id = our_lineup_for(conn, slate_id, [(s or "", n) for s, n in pairs])
    slate_id = slate_id or (meta["slate_id"] if meta else None)
    fee = entry_fee if entry_fee is not None else (meta["entry_fee"] if meta else None)
    if field_size is None and meta:
        field_size = meta["field_size"]
    with conn:
        cur = conn.execute(
            "INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, lineup_id, slots_json, "
            "entry_fee, points, rank, field_size, payout, source, created_at, note) "
            "VALUES (?,?,NULL,?,?,?,?,?,?,?,'manual',?,?)",
            (cid, slate_id, lineup_id, slots_json, fee, points, rank, field_size, payout,
             now or _now(), (note or "").strip()[:300] or None))
    return int(cur.lastrowid)


def update_entry(conn, entry_id: int, *, entry_fee: float | None = None,
                 payout: float | None = None, clear_payout: bool = False) -> bool:
    """Set the fee and / or the winnings of an entry (e.g. once the winnings are known)."""
    for nm, v in (("entry fee", entry_fee), ("winnings", payout)):
        if v is not None and (not math.isfinite(v) or v < 0):
            raise StandingsError(f"The {nm} must be a number of zero or more.")
    with conn:
        cur = conn.execute(
            "UPDATE dfs_entries SET entry_fee=COALESCE(?, entry_fee), "
            "payout=CASE WHEN ? THEN NULL ELSE COALESCE(?, payout) END WHERE entry_id=?",
            (entry_fee, 1 if clear_payout else 0, payout, int(entry_id)))
    return cur.rowcount > 0


def link_contest_slate(conn, contest_id: str, slate_id: str) -> None:
    """Attach an uploaded contest (and the owner's entries in it) to a slate, so the grader can
    use its real cash line."""
    cid = str(contest_id)
    if conn.execute("SELECT 1 FROM dfs_slates WHERE slate_id=?", (str(slate_id),)).fetchone() is None:
        raise StandingsError("That slate does not exist.")
    with conn:
        if conn.execute("SELECT 1 FROM dfs_contests WHERE contest_id=?", (cid,)).fetchone():
            conn.execute("UPDATE dfs_contests SET slate_id=? WHERE contest_id=?", (str(slate_id), cid))
        else:
            conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, captured_at, attrs_json) "
                         "VALUES (?,?,?,?)", (cid, str(slate_id), _now(),
                                              json.dumps({"source": "upload"})))
        conn.execute("UPDATE dfs_entries SET slate_id=? WHERE contest_id=?", (str(slate_id), cid))


def refresh_contest_payouts(conn, contest_id: str) -> int:
    """Re-derive ``dfs_contest_results.payout`` from the contest's payout table (use after the
    table was captured later than the upload). Returns rows updated."""
    cid = str(contest_id)
    table = contest_table(conn, cid)
    if not table:
        return 0
    rows = conn.execute("SELECT rowid, rank FROM dfs_contest_results WHERE contest_id=?",
                        (cid,)).fetchall()
    with conn:
        conn.executemany("UPDATE dfs_contest_results SET payout=? WHERE rowid=?",
                         [(payout_at(table, r["rank"]), r["rowid"]) for r in rows])
        conn.execute("UPDATE dfs_contests SET slate_id=slate_id WHERE contest_id=?", (cid,))
        for e in conn.execute("SELECT entry_id, rank FROM dfs_entries WHERE contest_id=? AND "
                              "source='upload' AND payout IS NULL AND rank IS NOT NULL", (cid,)).fetchall():
            conn.execute("UPDATE dfs_entries SET payout=? WHERE entry_id=?",
                         (payout_at(table, e["rank"]), e["entry_id"]))
    return len(rows)


# ============================================================================== derived numbers

def _quantile_desc(sorted_desc: list[float], q_top: float) -> float | None:
    """Score of the entry at the top-`q_top` share (0.01 = the 1% line) of a descending list."""
    if not sorted_desc:
        return None
    k = max(1, math.ceil(q_top * len(sorted_desc)))
    return sorted_desc[min(k, len(sorted_desc)) - 1]


def contest_summary(conn, contest_id: str) -> dict | None:
    """Derived numbers of an uploaded contest (None if nothing was uploaded for it)."""
    cid = str(contest_id)
    rows = conn.execute("SELECT rank, points, payout FROM dfs_contest_results WHERE contest_id=? "
                        "AND points IS NOT NULL", (cid,)).fetchall()
    if not rows:
        return None
    pts = sorted((float(r["points"]) for r in rows), reverse=True)
    n = len(pts)
    meta = conn.execute("SELECT c.*, s.nba_day, s.tag, s.game_type_id FROM dfs_contests c "
                        "LEFT JOIN dfs_slates s ON s.slate_id=c.slate_id WHERE c.contest_id=?",
                        (cid,)).fetchone()
    table = contest_table(conn, cid)
    places = paid_places(table)
    line, how = None, None
    if table and places:
        line, how = (pts[-1] if places >= n else pts[places - 1]), "payout table"
    else:
        paid = [float(r["points"]) for r in rows if (r["payout"] or 0) > 0]
        if paid:
            line, how = min(paid), "paid rows"
    own = conn.execute("SELECT name, pct_drafted, fpts FROM dfs_contest_ownership WHERE contest_id=? "
                       "ORDER BY pct_drafted DESC", (cid,)).fetchall()
    pcts = [float(o["pct_drafted"]) for o in own if o["pct_drafted"] is not None]
    up = conn.execute("SELECT upload_id, uploaded_at, filename, detail FROM dfs_uploads "
                      "WHERE kind='standings' AND contest_id=? ORDER BY upload_id DESC LIMIT 1",
                      (cid,)).fetchone()
    det = _loads(up["detail"], {}) if up else {}
    return {
        "contest_id": cid, "name": meta["name"] if meta else None,
        "slate_id": meta["slate_id"] if meta else None, "nba_day": meta["nba_day"] if meta else None,
        "entry_fee": meta["entry_fee"] if meta else None, "field_size": n,
        "field_size_listed": meta["field_size"] if meta else None,
        "winning_score": pts[0], "top1_score": _quantile_desc(pts, 0.01),
        "top10_score": _quantile_desc(pts, 0.10), "median_score": (pts[(n - 1) // 2] + pts[n // 2]) / 2.0,
        "min_score": pts[-1], "cash_line": line, "cash_line_source": how,
        "paid_places": places or None, "payout_table": bool(table),
        "ownership": {"players": len(own), "sum_pct": sum(pcts) if pcts else None,
                      "max_pct": max(pcts) if pcts else None,
                      "top": [{"name": o["name"], "pct": o["pct_drafted"], "fpts": o["fpts"]}
                              for o in own[:6]]},
        "unmatched": det.get("unmatched") or [], "game_type": det.get("game_type"),
        "uploaded_at": up["uploaded_at"] if up else None,
        "filename": up["filename"] if up else None, "slate_how": det.get("slate_how"),
        "final": det.get("final", True)}


def list_contests(conn) -> list[dict]:
    """Uploaded contests, newest upload first, with their derived numbers."""
    if not _has_table(conn, "dfs_uploads"):
        return []
    ids = [r[0] for r in conn.execute(
        "SELECT contest_id FROM dfs_uploads WHERE kind='standings' AND contest_id IS NOT NULL "
        "GROUP BY contest_id ORDER BY MAX(upload_id) DESC")]
    return [s for cid in ids if (s := contest_summary(conn, cid))]


def my_results(conn) -> dict:
    """The owner's REAL entries and totals. Pending = winnings not known (not counted in net)."""
    if not _has_table(conn, "dfs_entries"):
        return {"entries": [], "totals": _totals([])}
    rows = conn.execute(
        "SELECT e.*, c.name AS contest_name, s.nba_day, s.tag FROM dfs_entries e "
        "LEFT JOIN dfs_contests c ON c.contest_id=e.contest_id "
        "LEFT JOIN dfs_slates s ON s.slate_id=e.slate_id ORDER BY COALESCE(s.nba_day, e.created_at) "
        "DESC, e.entry_id DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        fee, pay = d["entry_fee"], d["payout"]
        d["net"] = None if (pay is None or fee is None) else pay - fee
        d["percentile"] = (None if not (d["rank"] and d["field_size"])
                           else 100.0 * (d["rank"] - 0.5) / d["field_size"])
        d["slots"] = _loads(d.get("slots_json"), []) or []
        out.append(d)
    return {"entries": out, "totals": _totals(out)}


def _totals(entries: list[dict]) -> dict:
    real = [e for e in entries if (e.get("source") or "").lower() not in
            ("sim", "simulated", "proxy", "synthetic", "backtest")]
    settled = [e for e in real if e.get("payout") is not None]
    fees = sum(e["entry_fee"] or 0.0 for e in real)
    fees_settled = sum(e["entry_fee"] or 0.0 for e in settled)
    wins = sum(e["payout"] for e in settled)
    net = wins - fees_settled
    return {"entries": len(real), "settled": len(settled), "pending": len(real) - len(settled),
            "fees": fees, "fees_settled": fees_settled, "winnings": wins, "net": net,
            "roi": (net / fees_settled) if fees_settled > 0 else None,
            "cashed": sum(1 for e in settled if (e["payout"] or 0) > 0),
            "no_fee": sum(1 for e in real if e.get("entry_fee") is None)}


# ============================================================================== ownership data

def ownership_dataset(conn):
    """Training data for the ownership model (T2.11): one row per (uploaded contest, player)
    joining the REAL ``pct_drafted`` with what we knew before lock for the same slate.

    Columns: contest_id, slate_id, nba_day, game_type, field_size, entry_fee, player_id, name,
    pct_drafted (percent), fpts (actual DK points in the contest), dk_id, team, salary,
    salary_rank, status (DK status at the capture), starter, fppg (DK FPPG), fppg_rank,
    value_fppg (fppg per $1k), run_id, run_kind, proj_dk, proj_dk_sd, proj_min, p_play,
    proj_rank, value_proj (proj per $1k), pool_size.

    The salary row is the latest capture at or before the slate's lock (else the latest capture);
    the projection is the slate's lock10 run (else lock45, else the latest non-swap run) made at
    or before that lock. Rows without a salary or projection keep NaN there. Returns a DataFrame
    (empty, with the columns, when nothing was uploaded)."""
    import numpy as np
    import pandas as pd
    from .. import ids as IDS
    cols = ["contest_id", "slate_id", "nba_day", "game_type", "field_size", "entry_fee",
            "player_id", "name", "pct_drafted", "fpts", "dk_id", "team", "salary", "salary_rank",
            "status", "starter", "fppg", "fppg_rank", "value_fppg", "run_id", "run_kind",
            "proj_dk", "proj_dk_sd", "proj_min", "p_play", "proj_rank", "value_proj", "pool_size"]
    own = pd.read_sql_query(
        "SELECT o.contest_id, o.player_id, o.name, o.pct_drafted, o.fpts, c.slate_id, "
        "c.entry_fee, s.nba_day, s.lock_utc, s.tag, s.game_type_id "
        "FROM dfs_contest_ownership o JOIN dfs_contests c ON c.contest_id=o.contest_id "
        "JOIN dfs_slates s ON s.slate_id=c.slate_id", conn)
    if own.empty:
        return pd.DataFrame(columns=cols)
    nba2dk = {v: k for k, v in IDS.nba_ids(conn, IDS.DK).items()}
    sizes = dict(conn.execute("SELECT contest_id, COUNT(*) FROM dfs_contest_results "
                              "GROUP BY contest_id").fetchall())
    frames = []
    for sid, g in own.groupby("slate_id"):
        lock = g["lock_utc"].iloc[0]
        caps = [r[0] for r in conn.execute("SELECT DISTINCT captured_at FROM dfs_salaries WHERE "
                                           "slate_id=? ORDER BY captured_at", (str(sid),))]
        pick = None
        if caps:
            pick = max((c for c in caps if lock and c <= lock), default=caps[-1])
        sal = pd.DataFrame(columns=["dk_id", "team", "salary", "status", "starter", "fppg"])
        if pick:
            q = conn.execute(
                "SELECT site_player_id AS dk_id, team, salary, status, starter, fppg, roster_slot "
                "FROM dfs_salaries WHERE slate_id=? AND captured_at=? AND disabled=0 "
                "AND site_player_id IS NOT NULL", (str(sid), pick)).fetchall()
            sal = pd.DataFrame([dict(r) for r in q])
            if len(sal):
                sal["dk_id"] = sal["dk_id"].astype(str)
                sal = sal.sort_values("roster_slot", key=lambda s: s.eq("CPT")).drop_duplicates(
                    "dk_id").drop(columns="roster_slot")
        run = _reference_run_before(conn, str(sid), lock)
        proj = pd.DataFrame(columns=["player_id", "proj_dk", "proj_dk_sd", "proj_min", "p_play"])
        if run:
            pq = conn.execute(
                "SELECT pp.player_id, pp.p_play, "
                "(SELECT mean FROM player_projection_stats s WHERE s.projection_id=pp.id AND "
                " s.stat='dk' AND s.cond='all') AS proj_dk, "
                "(SELECT sd FROM player_projection_stats s WHERE s.projection_id=pp.id AND "
                " s.stat='dk' AND s.cond='all') AS proj_dk_sd, "
                "(SELECT mean FROM player_projection_stats s WHERE s.projection_id=pp.id AND "
                " s.stat='min' AND s.cond='all') AS proj_min "
                "FROM player_projections pp WHERE pp.run_id=?", (run["run_id"],)).fetchall()
            proj = pd.DataFrame([dict(r) for r in pq]) if pq else proj
        d = g.copy()
        d["dk_id"] = d["player_id"].map(nba2dk)
        d = d.merge(sal, on="dk_id", how="left")
        if len(proj):
            d = d.merge(proj.drop_duplicates("player_id"), on="player_id", how="left")
        else:
            for c in ("proj_dk", "proj_dk_sd", "proj_min", "p_play"):
                d[c] = np.nan
        d["run_id"] = run["run_id"] if run else None
        d["run_kind"] = run["kind"] if run else None
        d["pool_size"] = len(sal)
        d["game_type"] = np.where((d["tag"] == "showdown") | (d["game_type_id"] == 81),
                                  "showdown", "classic")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["field_size"] = out["contest_id"].map(sizes)
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    out["fppg"] = pd.to_numeric(out["fppg"], errors="coerce")
    for c in ("proj_dk", "proj_dk_sd", "proj_min", "p_play"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    grp = out.groupby("contest_id")
    out["salary_rank"] = grp["salary"].rank(ascending=False, method="min")
    out["fppg_rank"] = grp["fppg"].rank(ascending=False, method="min")
    out["proj_rank"] = grp["proj_dk"].rank(ascending=False, method="min")
    out["value_fppg"] = out["fppg"] / (out["salary"] / 1000.0)
    out["value_proj"] = out["proj_dk"] / (out["salary"] / 1000.0)
    return out[cols]


def _reference_run_before(conn, slate_id: str, lock: str | None):
    """lock10 run, else lock45, else the latest non-swap run made at or before `lock`."""
    runs = [dict(r) for r in conn.execute(
        "SELECT run_id, kind, created_at FROM dfs_runs WHERE slate_id=? AND ok=1 "
        "ORDER BY created_at", (slate_id,))]
    if lock:
        early = [r for r in runs if r["created_at"] <= lock]
        runs = early or runs
    for kind in ("lock10", "lock45"):
        c = [r for r in runs if r["kind"] == kind]
        if c:
            return c[-1]
    c = [r for r in runs if r["kind"] != "swap"]
    return c[-1] if c else None


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("results-import", help="import a DraftKings contest standings CSV / zip or "
                                              "a Contest History CSV (real results)")
    p.add_argument("file")
    p.add_argument("--contest-id", default=None, help="when the file name has no contest id")
    p.add_argument("--slate", default=None, help="draft group id, when it cannot be inferred")
    p.add_argument("--username", default=None, help="your DK username (default: "
                                                    "$COURTSIDE_DK_USERNAME / dk_username.txt)")
    p = sub.add_parser("results-list", help="uploaded contests, your real entries and P&L")
    return {"results-import": _import_cmd, "results-list": _list_cmd}


def _conn():
    from ... import db
    return db.connect()


def _import_cmd(args) -> int:
    path = Path(args.file)
    try:
        res = import_upload(_conn(), path.read_bytes(), path.name, contest_id=args.contest_id,
                            slate_id=args.slate,
                            owner_names=[args.username] if args.username else None)
    except (StandingsError, OSError) as e:
        print(f"import failed: {e}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    return 0


def _money(v):
    return "-" if v is None else f"${v:,.2f}"


def _list_cmd(args) -> int:
    conn = _conn()
    cs = list_contests(conn)
    print(f"{len(cs)} uploaded contest(s)")
    for c in cs:
        print(f"  {c['contest_id']}  {c['nba_day'] or '?':<10} field {c['field_size']:>6} win "
              f"{c['winning_score']:.1f} top1% {c['top1_score']:.1f} cash line "
              f"{'-' if c['cash_line'] is None else format(c['cash_line'], '.1f')}")
    r = my_results(conn)
    t = r["totals"]
    print(f"\nMy real results: {t['entries']} entries, fees {_money(t['fees'])}, winnings "
          f"{_money(t['winnings'])}, net {_money(t['net'])}, ROI "
          f"{'-' if t['roi'] is None else format(t['roi'], '.1%')}, {t['pending']} pending")
    return 0
