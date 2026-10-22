"""Lock-relative DFS runs, late news and late swap (docs/PLAYERS.md §2.5, §5.3, §5.6; task T1.6).

A watcher (`tick`, every 5 minutes from the scheduler) looks at the slates in ``dfs_slates`` and
fires `run_slate` when a window is hit:

    lock45   now in [lock-47 min, lock-40 min]
    lock10   now in [lock-12 min, lock-7 min]
    swap     Classic slates only, for each LATER game tip T (T > lock): now in [T-12, T-7], and
             only if `late_news` touches a player of a saved lineup of that slate

Outside a window `tick` reads ``dfs_slates`` / ``dfs_salaries`` / ``dfs_runs`` and returns: no
network request, no write. Each run is idempotent on (slate, kind) (swap: on (slate, kind, tip));
only an ``ok = 1`` run counts, so a failed run is retried by the next tick of its window.

One run (`run_slate`)
  1. refreshes the inputs: the official injury report and ESPN injuries (once per tick) and the
     slate's draftables (`dfs_capture.refresh_slate`, which draws on the daily DK budget; a spent
     budget or a failing source only means older inputs, never a failed run);
  2. projects the slate with ``courtside.proj.pipeline.project_slate`` (imported lazily). If it
     is missing or raises, DK's FPPG is the projection and ``model_version`` is 'fppg-fallback';
  3. applies the §5.3 availability rule: a player with P(play) < 0.85 is excluded only when his
     game locks before the NEXT scheduled run (lock45: the lock10 run, so effectively nobody;
     lock10 / manual / swap: the next swap run, so first-game Q players are out and later-game Q
     players are allowed and recorded as swap candidates with their best replacement); DK "O" is
     always excluded;
  4. solves the cash lineup (mu - 0.3 (mu - p25)) and the Top N (default 10, min different 1);
  5. saves ``dfs_runs`` (+ ``late_news_json``) and ``dfs_lineups`` (modes ``cash`` / ``topn``;
     a swap run saves mode ``swap`` rows whose ``why_json.from`` names the lineup they replace).

Late news (`late_news`): relative to the previous run of the slate (or, with none, to 90 minutes
before lock) a player is flagged when his DK status or his injury-report status changed, DK named
him a starter / non-starter against our expectation (his start share in his last 10 games), or
his projected minutes moved >= 3. Players whose game has started are not flagged.

Public API
  run_slate(conn, slate_id, kind, *, now=None, n=10, min_diff=1, force=False, ...) -> dict
  tick(conn, now=None, *, dry_run=False, ...) -> list[dict]
  late_news(conn, slate_id, now, prev_run, ...) -> list[dict]
  current_lineups(conn, slate_id) -> list[dict]      saved lineups, swaps overlaid
  lineups_with_news(conn, slate_id, news) -> list[dict]   for the "N lineups contain late news" banner
  availability_for(conn, slate, pool, now) -> {dk_id: {"p", "status", "source"}}
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from ... import config
from ...sources import dfs_capture

log = logging.getLogger("courtside.proj.dfs.runner")

KINDS = ("lock45", "lock10", "swap", "manual")
# seconds before the event: fire when  near - TOLERANCE_S <= (event - now) <= far
LOCK45_WINDOW = (47 * 60, 40 * 60)
LOCK10_WINDOW = (12 * 60, 7 * 60)
SWAP_WINDOW = LOCK10_WINDOW
LOCK10_MAX_LEAD = 25 * 60      # adaptive lock10: never start earlier than lock - 25 min
LOCK10_FINISH_BY = 3 * 60      # ... and aim to finish this long before the lock
LOCK10_LATE_MIN = 2 * 60       # a missed lock10 may still start until lock - 2 min ...
LOCK10_LATE_PAD = 60           # ... when 1.25 x its build estimate + this fits before the lock
TICK_INTERVAL_S = 5 * 60       # launchd fires the tick every 5 min (a busy tick skips a fire)
SWAP_EST_S = 30.0              # a swap run's time, for the pull-forward check
TOLERANCE_S = 30               # launchd / systemd start a few seconds late
SWAP_LEAD = timedelta(minutes=10)
FALLBACK_VERSION = "fppg-fallback"
CASH_LAMBDA = 0.3
P_PLAY_MIN = 0.85
NEWS_WINDOW = timedelta(minutes=90)
MINUTES_MOVE = 3.0
START_UNEXPECTED_HI = 0.65     # expected starter DK says is not starting
START_UNEXPECTED_LO = 0.35     # expected non-starter DK names a starter
SOLVE_LIMIT_S = 20.0           # per CP-SAT solve
TOPN_LIMIT_S = 90.0            # whole Top N
SHOWDOWN_GAME_TYPE = 81
SWAP_CANDIDATE_CAP = 3         # replacement solves per run
_DK_STATUS_P = {"O": 0.0, "Q": 0.5, "GTD": 0.5}


# ============================================================================== small helpers

def _utc(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s) -> datetime | None:
    """'2026-10-21T01:30:00.0000000Z' / '...:00Z' -> aware UTC."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", str(s))
    if not m:
        return None
    return datetime.strptime(f"{m[1]}T{m[2]}", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _norm_status(s) -> str | None:
    return None if s in (None, "", "None") else str(s)


def _loads(s, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


def is_showdown(slate: dict) -> bool:
    return slate.get("tag") == "showdown" or slate.get("game_type_id") == SHOWDOWN_GAME_TYPE


# ============================================================================== slates and runs

def _slate(conn: sqlite3.Connection, slate_id) -> dict | None:
    r = conn.execute("SELECT * FROM dfs_slates WHERE slate_id=?", (str(slate_id),)).fetchone()
    return dict(r) if r else None


def slate_tips(conn: sqlite3.Connection, slate_id) -> list[datetime]:
    """Distinct game start times of a slate (latest capture), ascending."""
    sid = str(slate_id)
    rows = conn.execute(
        "SELECT DISTINCT start_time FROM dfs_salaries WHERE slate_id=? AND captured_at="
        "(SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=?)", (sid, sid)).fetchall()
    return sorted({t for t in (_parse_ts(r[0]) for r in rows) if t})


def slate_lock(slate: dict, tips: list[datetime]) -> datetime | None:
    return _parse_ts(slate.get("lock_utc")) or (tips[0] if tips else None)


def _run_rows(conn: sqlite3.Connection, slate_id, *, kind: str | None = None) -> list[dict]:
    sql = "SELECT * FROM dfs_runs WHERE slate_id=? AND ok=1"
    args: list = [str(slate_id)]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    out = []
    for r in conn.execute(sql + " ORDER BY created_at, rowid", args):
        d = dict(r)
        d["detail_obj"] = _loads(d.get("detail"), {}) or {}
        if not isinstance(d["detail_obj"], dict):
            d["detail_obj"] = {}
        out.append(d)
    return out


def run_done(conn: sqlite3.Connection, slate_id, kind: str, trigger: str | None = None
             ) -> dict | None:
    """The successful run that makes (slate, kind[, trigger]) idempotent, if any."""
    for r in _run_rows(conn, slate_id, kind=kind):
        if (r["detail_obj"].get("trigger") or None) == trigger:
            return r
    return None


def prev_run(conn: sqlite3.Connection, slate_id, before: datetime) -> dict | None:
    """The latest successful run of the slate before `before` that was a real build (a swap run
    that found nothing to do is not): the state its lineups were built on."""
    cut = iso(before)
    best = None
    for r in _run_rows(conn, slate_id):
        if r["created_at"] < cut and not r["detail_obj"].get("skipped"):
            best = r
    return best


# ============================================================================== saved lineups

def _lineup_rows(conn: sqlite3.Connection, run_id: str, modes) -> list[dict]:
    q = ",".join("?" * len(modes))
    out = []
    for r in conn.execute(f"SELECT * FROM dfs_lineups WHERE run_id=? AND mode IN ({q}) "
                          f"ORDER BY mode, rank, id", (run_id, *modes)):
        d = dict(r)
        slots = _loads(d["slots_json"], []) or []
        d["pairs"] = [(s["slot"], str(s["player_id"])) for s in slots]
        d["player_ids"] = [pid for _, pid in d["pairs"]]
        d["why"] = _loads(d.get("why_json"), {}) or {}
        out.append(d)
    return out


def current_lineups(conn: sqlite3.Connection, slate_id) -> list[dict]:
    """The slate's saved lineups as they stand: the latest build run's cash / Top N lineups,
    each replaced by its newest swap version (swap runs after that build). Every item has
    ``mode`` (cash / topn: the lineup it stands for), ``rank``, ``via`` ('build' | 'swap'),
    ``run_id``, ``id``, ``pairs`` [(slot, dk player id)], ``player_ids``, ``why``."""
    runs = _run_rows(conn, slate_id)
    base = None
    for r in reversed(runs):
        if r["kind"] != "swap":
            rows = _lineup_rows(conn, r["run_id"], ("cash", "topn"))
            if rows:
                base = (r, rows)
                break
    if base is None:
        return []
    brun, rows = base
    cur = {(d["mode"], d["rank"]): {**d, "via": "build"} for d in rows}
    for r in runs:
        if r["kind"] != "swap" or r["created_at"] <= brun["created_at"]:
            continue
        for d in _lineup_rows(conn, r["run_id"], ("swap",)):
            frm = d["why"].get("from") or {}
            key = (frm.get("mode"), frm.get("rank"))
            if key in cur:
                cur[key] = {**d, "mode": key[0], "rank": key[1], "via": "swap"}
    return sorted(cur.values(), key=lambda d: (d["mode"] != "cash", d["rank"] or 0))


def lineups_with_news(conn: sqlite3.Connection, slate_id, news: list[dict] | None = None,
                      *, now: datetime | None = None) -> list[dict]:
    """The saved lineups that contain a late-news player (for the page banner:
    "2 lineups contain late-news players"). `news` defaults to the ``late_news_json`` of the
    slate's newest run. With `now`, only news about players whose game has not started."""
    if news is None:
        runs = _run_rows(conn, slate_id)
        news = _loads(runs[-1].get("late_news_json"), []) if runs else []
    flagged = {}
    for n in news or []:
        if now is not None:
            gs = _parse_ts(n.get("game_start"))
            if gs is not None and gs <= _utc(now):
                continue
        flagged[str(n["player_id"])] = n
    out = []
    for lu in current_lineups(conn, slate_id):
        hit = [pid for pid in lu["player_ids"] if pid in flagged]
        if hit:
            out.append({"id": lu["id"], "run_id": lu["run_id"], "mode": lu["mode"],
                        "rank": lu["rank"], "via": lu["via"], "players": hit,
                        "news": [flagged[p] for p in hit]})
    return out


# ============================================================================== late news

@dataclass(frozen=True)
class PState:
    """One DK player of a slate at one capture."""
    dk_id: str
    name: str
    team: str | None
    status: str | None            # DK: None | Q | O | GTD (or 'disabled' when DK greyed him out)
    starter: int | None           # DK "in starting lineup": 1 / 0 (lineups out) / None (unknown)
    game_start: str | None


def states_at(conn: sqlite3.Connection, slate_id, at: datetime, *, fallback_first: bool = True
              ) -> tuple[str | None, dict[str, PState]]:
    """(captured_at, {dk id: PState}) of the latest ``dfs_salaries`` capture at or before `at`.
    Captures are stored only when something moved, so that capture IS the state at `at`.
    With none that early, the first capture ever taken (`fallback_first`)."""
    sid = str(slate_id)
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (sid, iso(at))).fetchone()[0]
    if cap is None and fallback_first:
        cap = conn.execute("SELECT MIN(captured_at) FROM dfs_salaries WHERE slate_id=?",
                           (sid,)).fetchone()[0]
    if cap is None:
        return None, {}
    out: dict[str, PState] = {}
    for r in conn.execute(
            "SELECT site_player_id, player_name, team, status, starter, start_time, "
            "MIN(disabled) AS dis FROM dfs_salaries WHERE slate_id=? AND captured_at=? AND "
            "site_player_id IS NOT NULL GROUP BY site_player_id", (sid, cap)):
        st = _norm_status(r["status"])
        if r["dis"] and st is None:
            st = "disabled"
        out[str(r["site_player_id"])] = PState(
            str(r["site_player_id"]), r["player_name"], r["team"], st,
            None if r["starter"] is None else int(r["starter"]), r["start_time"])
    return cap, out


def _report_statuses(conn: sqlite3.Connection, players: dict[str, PState], at: datetime
                     ) -> dict[str, str] | None:
    """{dk id: injury-report status} as known at `at` (official over ESPN, `injuries.resolve`);
    '' = not listed. None when no report batch exists at all by then."""
    from ... import injuries
    res = injuries.resolve(conn, iso(at))
    if not res["fresh"]:
        return None
    by_team: dict[str, dict[str, str]] = {}
    for team, t in res["teams"].items():
        by_team[team] = {injuries.norm_name(r["player_name"]): (r.get("status") or "").strip()
                         for r in t["rows"]}
    return {dk: by_team.get(p.team or "", {}).get(injuries.norm_name(p.name), "")
            for dk, p in players.items()}


def _run_minutes(conn: sqlite3.Connection, run_id: str | None) -> dict[str, float]:
    """{dk id: projected minutes when he plays} of a pipeline run (``player_projection_stats``
    stat 'min', cond 'played' else 'all'). {} when the run stored none."""
    if not run_id:
        return {}
    try:
        rows = conn.execute(
            "SELECT pp.player_id, s.cond, s.mean FROM player_projections pp JOIN "
            "player_projection_stats s ON s.projection_id=pp.id WHERE pp.run_id=? AND "
            "s.stat='min' AND s.mean IS NOT NULL", (run_id,)).fetchall()
        dk_of: dict[int, list[str]] = {}
        for r in conn.execute("SELECT source_id, player_id FROM player_ids WHERE source='dk' "
                              "AND player_id IS NOT NULL"):
            dk_of.setdefault(int(r[1]), []).append(str(r[0]))
    except sqlite3.OperationalError:
        return {}
    best: dict[int, tuple[int, float]] = {}
    for pid, cond, mean in rows:
        rank = 0 if cond == "played" else 1
        if pid not in best or rank < best[pid][0]:
            best[pid] = (rank, float(mean))
    return {dk: m for pid, (_, m) in best.items() for dk in dk_of.get(int(pid), [])}


def start_shares(conn: sqlite3.Connection, dk_ids, day: date) -> dict[str, float]:
    """{dk id: share of his last 10 games (since ~8 months back) in which he started}; players
    without a history or an NBA id are absent (no expectation, so never an 'unexpected' starter)."""
    from .. import ids
    mapping = ids.nba_ids(conn, ids.DK, list(dk_ids))
    if not mapping:
        return {}
    by_nba: dict[int, list[str]] = {}
    for dk, nba in mapping.items():
        by_nba.setdefault(nba, []).append(dk)
    lo = (day - timedelta(days=240)).isoformat()
    last: dict[int, list[int]] = {}
    ph = ",".join("?" * len(by_nba))
    for pid, started in conn.execute(
            f"SELECT pg.player_id, ps.player_id IS NOT NULL FROM player_games pg "
            f"LEFT JOIN player_starts ps ON ps.game_id=pg.game_id AND ps.player_id=pg.player_id "
            f"WHERE pg.player_id IN ({ph}) AND pg.game_date<? AND pg.game_date>=? AND pg.min>0 "
            f"ORDER BY pg.game_date DESC", (*by_nba, day.isoformat(), lo)):
        v = last.setdefault(int(pid), [])
        if len(v) < 10:
            v.append(int(started))
    return {dk: sum(v) / len(v) for pid, v in last.items() if v for dk in by_nba[pid]}


def diff_news(prev: dict[str, PState], cur: dict[str, PState], *, now: datetime,
              p_start: dict[str, float] | None = None,
              report_prev: dict[str, str] | None = None,
              report_cur: dict[str, str] | None = None,
              minutes_prev: dict[str, float] | None = None,
              minutes_now: dict[str, float] | None = None,
              min_move: float = MINUTES_MOVE) -> list[dict]:
    """The pure core of `late_news`: compare two player states (see there for the rules).
    A player missing from `prev` has no previous status / starter flag / minutes to compare."""
    from ... import injuries
    p_start = p_start or {}
    out = []
    for dk, c in cur.items():
        gs = _parse_ts(c.game_start)
        if gs is not None and gs <= now:
            continue                                   # started: nothing left to react to
        p = prev.get(dk)
        reasons: list[str] = []
        entry: dict = {"player_id": dk, "name": c.name, "team": c.team,
                       "game_start": iso(gs) if gs else None}
        if p is not None and c.status != p.status:
            reasons.append("status")
            entry["status"] = {"from": p.status, "to": c.status, "source": "draftkings"}
        if report_prev is not None and report_cur is not None and dk in report_cur:
            a, b = report_prev.get(dk, ""), report_cur[dk]
            if injuries.SEVERITY.get(a.lower(), 0) != injuries.SEVERITY.get(b.lower(), 0):
                if "status" not in reasons:
                    reasons.append("status")
                entry["report_status"] = {"from": a or None, "to": b or None}
        if c.starter is not None and (p is None or p.starter != c.starter):
            share = p_start.get(dk)
            kind = None
            if share is not None:
                if c.starter == 1 and share <= START_UNEXPECTED_LO:
                    kind = "unexpected_starter"
                elif c.starter == 0 and share >= START_UNEXPECTED_HI:
                    kind = "unexpected_non_starter"
            if kind:
                reasons.append(kind)
                entry["starter"] = {"dk": c.starter, "expected_share": round(share, 2)}
        m1, m0 = (minutes_now or {}).get(dk), (minutes_prev or {}).get(dk)
        if m1 is not None and m0 is not None and abs(m1 - m0) >= min_move:
            reasons.append("minutes")
            entry["minutes"] = {"from": round(m0, 1), "to": round(m1, 1),
                                "delta": round(m1 - m0, 1)}
        if reasons:
            entry["reasons"] = reasons
            out.append(entry)
    out.sort(key=lambda e: (e["game_start"] or "", e["name"] or ""))
    return out


def late_news(conn: sqlite3.Connection, slate_id, now: datetime, prev_run: dict | None, *,
              run_id: str | None = None, minutes: dict[str, float] | None = None,
              prev_minutes: dict[str, float] | None = None,
              p_start: dict[str, float] | None = None) -> list[dict]:
    """Players of the slate with late news (PLAYERS.md §2.5), each
    ``{player_id (DK), name, team, game_start, reasons, status?, report_status?, starter?,
    minutes?}``. Baseline = the state at the previous run (`prev_run`, a ``dfs_runs`` row),
    else 90 minutes before lock. A player is flagged when, since the baseline:

    * his DK status (None / Q / GTD / O) or his injury-report status (official over ESPN,
      matched by name) changed -> ``status``;
    * DK now flags him a starter (``unexpected_starter``) although he started <= 35% of his last
      10 games, or confirms him a non-starter (``unexpected_non_starter``) although he started
      >= 65% -- only when that DK flag is new since the baseline;
    * his projected minutes (`minutes` for run `run_id`, else read from ``player_projection_stats``;
      against the previous run's) moved by >= 3 -> ``minutes``.

    Players whose game has started are never flagged. `minutes` / `prev_minutes` / `p_start`
    ({dk id: value}) override what is read from the database."""
    now = _utc(now)
    slate = _slate(conn, slate_id) or {}
    tips = slate_tips(conn, slate_id)
    lock = slate_lock(slate, tips)
    if prev_run is not None and _parse_ts(prev_run.get("created_at")):
        base_t = _parse_ts(prev_run["created_at"])
    else:
        base_t = (lock or now) - NEWS_WINDOW
    _, cur = states_at(conn, slate_id, now, fallback_first=False)
    _, prev = states_at(conn, slate_id, base_t)
    rep_prev = rep_cur = None
    try:
        rep_prev = _report_statuses(conn, cur, base_t)
        rep_cur = _report_statuses(conn, cur, now)
        if rep_prev is None or rep_cur is None:
            rep_prev = rep_cur = None
    except Exception:  # noqa: BLE001 - the report is one signal among several
        log.exception("late_news: injury report comparison failed")
    if p_start is None:
        try:
            day = date.fromisoformat(slate["nba_day"]) if slate.get("nba_day") else config.nba_day(now)
            p_start = start_shares(conn, [dk for dk, c in cur.items() if c.starter is not None], day)
        except Exception:  # noqa: BLE001
            log.exception("late_news: start shares unavailable")
            p_start = {}
    if minutes is None:
        minutes = _run_minutes(conn, run_id)
    if prev_minutes is None:
        prev_minutes = _run_minutes(conn, (prev_run or {}).get("run_id"))
    news = diff_news(prev, cur, now=now, p_start=p_start, report_prev=rep_prev,
                     report_cur=rep_cur, minutes_prev=prev_minutes, minutes_now=minutes)
    for n in news:
        n["since"] = iso(base_t)
    return news


# ============================================================================== inputs

def refresh_inputs(conn: sqlite3.Connection, now: datetime) -> dict:
    """Re-read the official injury report and ESPN injuries (once per tick, shared by every
    slate that fires). Each source is isolated; the result is a small summary per source."""
    from ... import collect
    day = config.nba_day(now)
    steps = (("official_injury_report", collect._official_report_step, (conn, day)),
             ("espn_injuries", collect._espn_injuries_step, (conn, day, iso(now))))
    out: dict = {}
    for name, fn, args in steps:
        try:
            res = fn(*args) or {}
            out[name] = {"status": res.get("status", "ok"), "rows": res.get("rows"),
                         **({"reason": res["reason"]} if "reason" in res else {})}
        except Exception as e:  # noqa: BLE001
            log.warning("refresh %s failed: %s", name, e)
            out[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    return out


def dk_status_p(status: str | None) -> float:
    return _DK_STATUS_P.get(status or "", 1.0)


def availability_for(conn: sqlite3.Connection, slate: dict, pool, now: datetime) -> dict[str, dict]:
    """{dk player id: {"p": P(play), "status", "source"}} for the pool: the calibrated model
    (`availability.p_play_slate`, latest report at `now`, DK starters at 0.99) where the player is
    matched to an NBA id and listed in a game of the day, else DK's own flag (O 0, Q / GTD 0.5,
    otherwise 1). DK "O" is always 0."""
    out = {p.dk_player_id: {"p": dk_status_p(p.status), "status": p.status or "",
                            "source": "dk_status"} for p in pool.players}
    try:
        from .. import availability as A
        from .. import ids
        dk_ids = [p.dk_player_id for p in pool.players]
        mapping = ids.nba_ids(conn, ids.DK, dk_ids)
        if len(mapping) < 0.9 * len(dk_ids):
            ids.resolve_dk(conn, since=pool.captured_at, write_check=False)
            mapping = ids.nba_ids(conn, ids.DK, dk_ids)
        starters = {mapping[p.dk_player_id] for p in pool.players
                    if p.starter and p.dk_player_id in mapping}
        df = A.p_play_slate(conn, slate["nba_day"], now, starters=starters)
        by_nba = {int(r.player_id): (float(r.p_play), str(r.status)) for r in df.itertuples()}
        for p in pool.players:
            hit = by_nba.get(mapping.get(p.dk_player_id, -1))
            if hit is not None:
                out[p.dk_player_id] = {"p": hit[0], "status": hit[1], "source": "availability"}
    except Exception:  # noqa: BLE001 - the DK flags are a safe floor
        log.exception("availability model failed; using DK status flags")
    for p in pool.players:
        if p.status == "O":
            out[p.dk_player_id] = {"p": 0.0, "status": "O", "source": "dk_status"}
    return out


def next_run_at(kind: str, slate: dict, tips: list[datetime], now: datetime) -> datetime | None:
    """When the next scheduled run that could still change a lineup happens, from a run of
    `kind` at `now`: lock45 -> the lock10 run; otherwise the next swap run (a later game's
    tip - 10 min), None when there is none (Showdown, or no later game left)."""
    lock = slate_lock(slate, tips)
    if kind == "lock45" and lock is not None:
        return lock - SWAP_LEAD
    if is_showdown(slate) or lock is None:
        return None
    later = sorted(t - SWAP_LEAD for t in tips if t > lock and t - SWAP_LEAD > now)
    return later[0] if later else None


def availability_bans(pool, avail: dict[str, dict], nxt: datetime | None, now: datetime
                      ) -> tuple[set[str], list[dict]]:
    """(bans, excluded details) under the §5.3 rule: P(play) < 0.85 and his game locks before
    the next run (always, when there is none); DK "O" always."""
    bans: set[str] = set()
    detail = []
    for p in pool.players:
        a = avail.get(p.dk_player_id)
        if a is None:
            continue
        start = p.start
        out_ = p.status == "O" or (a["p"] < P_PLAY_MIN and (nxt is None or (start is not None and start < nxt)))
        if out_:
            bans.add(p.dk_player_id)
            detail.append({"player_id": p.dk_player_id, "name": p.name, "p_play": round(a["p"], 3),
                           "status": a["status"], "game_start": iso(start) if start else None})
    return bans, detail


# ============================================================================== projection

def project(conn: sqlite3.Connection, pool, slate: dict, now: datetime, run_id: str
            ) -> tuple[dict, str, dict]:
    """({dk id: Projection}, model_version, info). ``courtside.proj.pipeline.project_slate`` when
    it imports and runs, else DK's FPPG (``model_version`` 'fppg-fallback', the reason in info)."""
    from .optimizer import Projection, as_projection
    info: dict = {}
    try:
        pipe = importlib.import_module("courtside.proj.pipeline")
        raw = pipe.project_slate(conn, slate["slate_id"], now, run_id=run_id)
        projs = {str(k): as_projection(v) for k, v in raw.items()}
        if not projs:
            raise RuntimeError("project_slate returned no projections")
        mv = None
        try:
            r = conn.execute("SELECT model_version FROM player_projections WHERE run_id=? "
                             "LIMIT 1", (run_id,)).fetchone()
            mv = r[0] if r else None
        except sqlite3.OperationalError:
            pass
        mv = mv or getattr(pipe, "MODEL_VERSION", None) or "pipeline"
        return projs, mv, info
    except ImportError as e:
        info["pipeline"] = f"unavailable: {e}"
    except Exception as e:  # noqa: BLE001 - a crashed model must not leave the slate without lineups
        log.exception("project_slate failed; falling back to FPPG")
        info["pipeline"] = f"error: {type(e).__name__}: {e}"
    projs = {p.dk_player_id: Projection(p.fppg) for p in pool.players if p.fppg is not None}
    return projs, FALLBACK_VERSION, info


# ============================================================================== solving

def _name_of(pool_by_id: dict, pid: str) -> str:
    p = pool_by_id.get(pid)
    return p.name if p else pid


def _lineup_dict(lu) -> dict:
    return {"slots": json.loads(lu.slots_json()), "salary": lu.salary, "mean": lu.projected}


def _q_players(lu, avail: dict[str, dict]) -> list[dict]:
    return [{"player_id": e.player.dk_player_id, "name": e.player.name,
             "p_play": round(avail[e.player.dk_player_id]["p"], 3),
             "game_start": iso(e.player.start) if e.player.start else None}
            for e in lu.entries
            if e.player.dk_player_id in avail and avail[e.player.dk_player_id]["p"] < P_PLAY_MIN]


def _swap_candidates(O, pool, projs, rules, cash, avail, bans, now) -> list[dict]:
    """For each allowed Q player of the cash lineup (a later-game player): who replaces him if
    he turns out Out, i.e. the cash lineup re-solved with him banned."""
    out = []
    by_id = pool.by_id()
    for q in _q_players(cash, avail)[:SWAP_CANDIDATE_CAP]:
        try:
            alt = O.optimize(pool, projs, rules, mode="cash", lam=CASH_LAMBDA,
                             bans=bans | {q["player_id"]}, now=now, time_limit_s=SOLVE_LIMIT_S)
        except Exception:  # noqa: BLE001
            log.exception("swap candidate solve failed")
            alt = None
        entry = dict(q)
        if alt is not None:
            entry["replacement"] = [{"player_id": pid, "name": _name_of(by_id, pid)}
                                    for pid in sorted(alt.player_ids - cash.player_ids)]
            entry["replacement_mean"] = round(alt.projected, 2)
        out.append(entry)
    return out


def _build(O, T, pool, projs, rules, bans, now, n, min_diff, avail):
    cash = O.optimize(pool, projs, rules, mode="cash", lam=CASH_LAMBDA, bans=bans, now=now,
                      time_limit_s=SOLVE_LIMIT_S)
    res = T.top_n(pool, projs, rules, n, min_diff=min_diff, mode="mean", bans=bans, now=now,
                  time_limit_s=SOLVE_LIMIT_S, total_time_limit_s=TOPN_LIMIT_S)
    return cash, res


def _resolve_swaps(O, pool, projs, rules, targets, bans, now, min_diff):
    """Re-solve each target saved lineup with the started players of its game fixed, under the
    objective it was built with (the loop lives in `swap.resolve_targets`, shared with the
    owner-facing late-swap re-solver `swap.plan`). `O` is kept for the call signature."""
    from . import swap
    return swap.resolve_targets(pool, projs, rules, targets, bans, now, min_diff)


def _insert_lineup(conn, run_id: str, mode: str, rank: int, lu, why: dict | None) -> None:
    conn.execute(
        "INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean, why_json) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, mode, rank, lu.slots_json(), lu.salary, lu.projected,
         json.dumps(why, default=str) if why else None))


# ============================================================================== lineup sims

PCT_COLS = (("p10", 10), ("p50", 50), ("p90", 90))
CAPTAIN_SLOT = "CPT"


class RunSims:
    """A run's per-sim DK points keyed by DK player id: ``fp`` (S, n) float16 and ``col`` {dk id:
    column}. Shared by the lineup range stats and (later) the field model (proj/dfs/field.py)."""

    def __init__(self, fp, dk_ids, path: str | None = None):
        self.fp = fp
        self.path = path
        self.col: dict[str, int] = {}
        for i, d in enumerate(dk_ids):
            d = str(d)
            if d and d not in ("None", "nan") and d not in self.col:
                self.col[d] = i
        self.n_sims = int(fp.shape[0])

    def lineup_totals(self, slots: list[dict]):
        """(S,) float32 sum of the lineup's players' sims (CPT x1.5), or None when any player has
        no column (the stats then stay NULL rather than understate the lineup)."""
        import numpy as np
        from ...sources import dfs as _dfs
        idx, w = [], []
        for s in slots:
            c = self.col.get(str(s.get("player_id")))
            if c is None:
                return None
            idx.append(c)
            w.append(_dfs.SHOWDOWN_CAPTAIN_MULTIPLIER if s.get("slot") == CAPTAIN_SLOT else 1.0)
        if not idx:
            return None
        return self.fp[:, idx].astype(np.float32) @ np.asarray(w, np.float32)


def load_run_sims(conn: sqlite3.Connection, run_id: str) -> RunSims | None:
    """The run's stored sims (``dfs_runs.sim_path``), or None: no path (FPPG fallback), a missing
    or unreadable file, or an npz without DK ids."""
    from ..store import load_sims
    r = conn.execute("SELECT sim_path FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
    path = r[0] if r else None
    if not path:
        return None
    try:
        z = load_sims(path)
    except (OSError, ValueError):
        log.warning("sims for run %s unreadable at %s", run_id, path)
        return None
    if "fp" not in z or "dk_id" not in z:
        return None
    return RunSims(z["fp"], z["dk_id"], str(path))


def lineup_stats(totals) -> dict[str, float]:
    """mean, sd, p10, p50, p90 of a (S,) per-sim lineup total."""
    import numpy as np
    q = np.percentile(totals, [q for _, q in PCT_COLS])
    return {"mean": float(totals.mean()), "sd": float(totals.std()),
            **{c: float(v) for (c, _), v in zip(PCT_COLS, q)}}


def annotate_lineups(conn: sqlite3.Connection, run_id: str, sims: RunSims | None = None) -> int:
    """Fill ``mean`` (sim mean), ``p10`` / ``p50`` / ``p90`` of the run's ``dfs_lineups`` rows from
    the stored sims and add ``{"sim": {"sd", "n_sims"}}`` to ``why_json``. Lineups with a player
    missing from the sims, and runs without sims (FPPG fallback), are left as they are (NULL).
    Returns the number of lineups updated. Does not commit.

    HOOK (T2.11): proj/dfs/field.py fills p_cash / p_top1 / e_payout / dupes on the same rows;
    call it from `post_lineups` with the same `RunSims`."""
    sims = sims or load_run_sims(conn, run_id)
    if sims is None:
        return 0
    n = 0
    for lid, slots_json, why_json in conn.execute(
            "SELECT id, slots_json, why_json FROM dfs_lineups WHERE run_id=?",
            (run_id,)).fetchall():
        tot = sims.lineup_totals(_loads(slots_json, []) or [])
        if tot is None:
            continue
        st = lineup_stats(tot)
        why = _loads(why_json, {}) or {}
        if not isinstance(why, dict):
            why = {}
        why["sim"] = {"sd": round(st["sd"], 2), "n_sims": sims.n_sims}
        conn.execute("UPDATE dfs_lineups SET mean=?, p10=?, p50=?, p90=?, why_json=? WHERE id=?",
                     (st["mean"], st["p10"], st["p50"], st["p90"], json.dumps(why, default=str),
                      lid))
        n += 1
    return n


def post_lineups(conn: sqlite3.Connection, run_id: str) -> dict:
    """Everything computed from a run's saved lineups and sims. Never raises: a failure only
    leaves the extra columns NULL."""
    out: dict = {}
    try:
        sims = load_run_sims(conn, run_id)
        out["lineup_sims"] = annotate_lineups(conn, run_id, sims) if sims else 0
        # HOOK (T2.11): field.annotate(conn, run_id, sims) -> p_cash, p_top1, e_payout, dupes
    except Exception as e:  # noqa: BLE001
        log.exception("lineup sim stats failed for %s", run_id)
        out["lineup_sims_error"] = f"{type(e).__name__}: {e}"
    return out


def backfill_lineup_ranges(conn: sqlite3.Connection, *, only_missing: bool = True) -> dict:
    """One-off: annotate the lineups of every existing run whose sims file exists. By default only
    runs with a lineup lacking p10 are touched. Returns {runs, lineups, skipped}."""
    where = "AND l.p10 IS NULL" if only_missing else ""
    runs = [r[0] for r in conn.execute(
        f"SELECT DISTINCT r.run_id, r.created_at FROM dfs_runs r JOIN dfs_lineups l "
        f"ON l.run_id=r.run_id WHERE r.sim_path IS NOT NULL {where} "
        f"ORDER BY r.created_at").fetchall()]
    res = {"runs": 0, "lineups": 0, "skipped": 0}
    for rid in runs:
        k = annotate_lineups(conn, rid)
        conn.commit()
        if k:
            res["runs"] += 1
            res["lineups"] += k
        else:
            res["skipped"] += 1
    return res


# ============================================================================== run_slate

def _next_tip(slate: dict, tips: list[datetime], now: datetime) -> datetime | None:
    lock = slate_lock(slate, tips)
    later = [t for t in tips if lock is not None and t > lock and t > now]
    return later[0] if later else None


def run_slate(conn: sqlite3.Connection, slate_id, kind: str, *, now: datetime | None = None,
              n: int = 10, min_diff: int = 1, force: bool = False, refresh=True,
              refresh_global: bool = True, tip: datetime | str | None = None,
              resolve_all: bool = False) -> dict:
    """One lock-relative run of a slate; returns a summary dict with ``status``:

    ok | already_done | skipped_no_news (swap with nothing to swap) | nothing_to_swap |
    unknown_slate | error.

    `refresh`: True (default) refreshes the inputs, False skips it, a callable
    ``f(conn, slate, kind, now) -> dict`` replaces it. `refresh_global=False` skips the
    shared injury-report / ESPN refresh (the caller did it once for several slates).
    `tip` (swap): the later game this swap is for (default: the next one after `now`).
    `resolve_all` (swap): re-solve every saved lineup, not only those with late news.
    `force` runs again even when an ok run exists."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r} ({'|'.join(KINDS)})")
    t0 = time.perf_counter()
    now = _utc(now)
    sid = str(slate_id)
    slate = _slate(conn, sid)
    if slate is None:
        return {"status": "unknown_slate", "slate_id": sid, "kind": kind}
    tips = slate_tips(conn, sid)
    trigger = None
    if kind == "swap":
        if is_showdown(slate):
            return {"status": "nothing_to_swap", "slate_id": sid, "kind": kind,
                    "reason": "Showdown has no late swap"}
        t = _parse_ts(tip) if isinstance(tip, str) else (_utc(tip) if tip else _next_tip(slate, tips, now))
        if t is None or (slate_lock(slate, tips) is not None and t <= slate_lock(slate, tips)):
            return {"status": "nothing_to_swap", "slate_id": sid, "kind": kind,
                    "reason": "no later game"}
        trigger = iso(t)
    if not force:
        done = run_done(conn, sid, kind, trigger)
        if done is not None:
            return {"status": "already_done", "slate_id": sid, "kind": kind,
                    "run_id": done["run_id"], "trigger": trigger}
    run_id = uuid.uuid4().hex[:12]
    detail: dict = {"trigger": trigger, "now": iso(now)}
    try:
        res = _execute(conn, slate, tips, kind, now, run_id, detail, n, min_diff, refresh,
                       refresh_global, trigger, resolve_all)
    except Exception as e:  # noqa: BLE001
        log.exception("dfs run %s/%s failed", sid, kind)
        with contextlib.suppress(Exception):
            conn.rollback()
        detail["error"] = f"{type(e).__name__}: {e}"
        from . import optimizer as O
        O.save_run(conn, slate_id=sid, kind=kind, model_version=detail.get("model_version"),
                   seconds=time.perf_counter() - t0, ok=False, detail=detail, path="A",
                   run_id=run_id, created_at=iso(now))
        return {"status": "error", "slate_id": sid, "kind": kind, "run_id": run_id,
                "error": detail["error"]}
    res.setdefault("slate_id", sid)
    res.setdefault("kind", kind)
    res["run_id"] = run_id
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def _execute(conn, slate, tips, kind, now, run_id, detail, n, min_diff, refresh, refresh_global,
             trigger, resolve_all) -> dict:
    from . import optimizer as O
    from . import rules as R
    from . import topn as T
    t0 = time.perf_counter()
    sid = slate["slate_id"]

    # 1. inputs ------------------------------------------------------------------------
    if callable(refresh):
        detail["refresh"] = refresh(conn, slate, kind, now)
    elif refresh:
        r: dict = {}
        if refresh_global:
            r["inputs"] = refresh_inputs(conn, now)
        r["draftables"] = dfs_capture.refresh_slate(conn, sid, reason=kind, now=now)
        detail["refresh"] = r
    tips = slate_tips(conn, sid) or tips

    pool = R.build_pool(conn, sid)
    rules = pool.rules
    if rules is None:
        raise LookupError(f"no rules for slate {sid}")
    prev = prev_run(conn, sid, now)
    current: list[dict] = []
    targets: list[dict] = []
    news: list[dict] = []

    # 2. swap: is there anything to do? --------------------------------------------------
    if kind == "swap":
        current = current_lineups(conn, sid)
        news = late_news(conn, sid, now, prev)
        hit = {(h["mode"], h["rank"]) for h in lineups_with_news(conn, sid, news, now=now)}
        targets = [c for c in current if (c["mode"], c["rank"]) in hit or resolve_all]
        if not targets:
            detail.update(skipped="no late news touches a saved lineup" if current
                          else "no saved lineups", late_news=len(news))
            _save_run(conn, run_id, sid, kind, now, None, t0, detail, news)
            return {"status": "skipped_no_news", "late_news": len(news),
                    "saved_lineups": len(current)}

    # 3. projections -------------------------------------------------------------------
    projs, model_version, info = project(conn, pool, slate, now, run_id)
    detail.update(info)
    detail["model_version"] = model_version
    if not projs:
        raise RuntimeError("no projections for the slate")

    # 4. availability --------------------------------------------------------------------
    avail = availability_for(conn, slate, pool, now)
    nxt = next_run_at(kind, slate, tips, now)
    bans, excluded = availability_bans(pool, avail, nxt, now)
    detail["availability"] = {"next_run_at": iso(nxt) if nxt else None, "excluded": excluded,
                              "sources": _count(a["source"] for a in avail.values())}

    # 5. solve ---------------------------------------------------------------------------
    counts = {}
    if kind == "swap":
        news = late_news(conn, sid, now, prev, run_id=run_id)
        solved = _resolve_swaps(O, pool, projs, rules, targets, bans, now, min_diff)
        k = 0
        for s in solved:
            t = s["target"]
            if s["lineup"] is None:
                continue
            why = {"from": {"mode": t["mode"], "rank": t["rank"], "lineup_id": t["id"]},
                   "in": s["in"], "out": s["out"],
                   "late_news": [x["name"] for x in news if x["player_id"] in t["player_ids"]]}
            _insert_lineup(conn, run_id, "swap", t["rank"], s["lineup"], why)
            k += 1
        counts["swap"] = k
        detail["swap"] = {"targets": len(targets),
                          "unsolved": sum(1 for s in solved if s["lineup"] is None)}
        if k == 0:
            raise RuntimeError("swap: no legal lineup after the started players were fixed")
    else:
        cash, res = _build(O, T, pool, projs, rules, bans, now, n, min_diff, avail)
        if cash is None and not len(res):
            raise RuntimeError("no legal lineup (rules, bans and projections leave none)")
        news = late_news(conn, sid, now, prev, run_id=run_id)
        if cash is not None:
            why = {"q_players": _q_players(cash, avail)} if _q_players(cash, avail) else None
            _insert_lineup(conn, run_id, "cash", 1, cash, why)
            counts["cash"] = 1
            cands = _swap_candidates(O, pool, projs, rules, cash, avail, bans, now) if why else []
            if cands:
                detail["swap_candidates"] = cands
        for r in res:
            q = _q_players(r.lineup, avail)
            _insert_lineup(conn, run_id, "topn", r.rank, r.lineup, {"q_players": q} if q else None)
        counts["topn"] = len(res)
        detail["topn"] = {"requested": n, "min_diff": min_diff, "stopped": res.stopped}
    detail["late_news"] = len(news)
    detail.update(post_lineups(conn, run_id))
    if kind != "swap":
        from . import field as FLD   # proxy ownership + simulated field; never fails the run
        detail["field"] = FLD.score_run_safe(conn, run_id, slate_id=sid, commit=False)
    _save_run(conn, run_id, sid, kind, now, model_version, t0, detail, news)
    return {"status": "ok", "model_version": model_version, "lineups": counts,
            "late_news": len(news), "excluded": len(excluded)}


def _count(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _save_run(conn, run_id, sid, kind, now, model_version, t0, detail, news) -> None:
    from . import optimizer as O
    O.save_run(conn, slate_id=sid, kind=kind, model_version=model_version,
               seconds=time.perf_counter() - t0, ok=True, detail=detail, path="A",
               run_id=run_id, created_at=iso(now))     # commits with the lineups inserted above
    conn.execute("UPDATE dfs_runs SET late_news_json=? WHERE run_id=?",
                 (json.dumps(news, default=str), run_id))
    conn.commit()


# ============================================================================== tick

def _in_window(delta_s: float, window: tuple[int, int]) -> bool:
    far, near = window
    return near - TOLERANCE_S <= delta_s <= far


def build_estimate(conn: sqlite3.Connection, slate_ids: list, lock: datetime, *,
                   fallback: bool = False) -> float:
    """Seconds to build `slate_ids` one after another: each slate's latest successful lock45
    (else any non-swap) build from the 6 h before `lock`. A slate with no such build counts 0,
    or with `fallback` the slowest non-swap build of any slate in those 6 h (a slate's first
    build of the day has no history of its own)."""
    since = iso(lock - timedelta(hours=6))
    worst = None
    est = 0.0
    for sid in slate_ids:
        r = conn.execute(
            "SELECT seconds FROM dfs_runs WHERE slate_id=? AND ok=1 AND kind!='swap' "
            "AND created_at>=? AND seconds IS NOT NULL "
            "ORDER BY (kind='lock45') DESC, created_at DESC LIMIT 1", (str(sid), since)).fetchone()
        if r:
            est += float(r[0])
        elif fallback:
            if worst is None:
                w = conn.execute(
                    "SELECT MAX(seconds) FROM dfs_runs WHERE ok=1 AND kind!='swap' AND "
                    "created_at>=? AND seconds IS NOT NULL", (since,)).fetchone()
                worst = float(w[0]) if w and w[0] is not None else 0.0
            est += worst
    return est


def lock10_window(conn: sqlite3.Connection, slate_ids: list, lock: datetime) -> tuple[int, int]:
    """The lock10 window for the slates that share `lock`, widened when builds are slow.

    The tick runs those slates one after another, so the estimate is the sum of each slate's
    latest successful lock45 (else any non-swap) build time from the last 6 h. When
    1.25 x that + LOCK10_FINISH_BY exceeds the default 12 min lead, the window starts that much
    earlier (capped at LOCK10_MAX_LEAD) and stays 5 min wide. The Mac is shared with CI: on
    2026-10-20 a lock45 build took 574 s at load ~150 (27 s unloaded), which would have
    finished the default lock10 run at the lock."""
    est = build_estimate(conn, slate_ids, lock)
    far = int(min(LOCK10_MAX_LEAD, max(LOCK10_WINDOW[0], 1.25 * est + LOCK10_FINISH_BY)))
    if far <= LOCK10_WINDOW[0]:
        return LOCK10_WINDOW
    return far, far - (LOCK10_WINDOW[0] - LOCK10_WINDOW[1])


def _slate_locks(conn: sqlite3.Connection, now: datetime):
    """([(slate, tips, lock)], {lock: [slate_id]}) for slates locking from 14 h ago to 1 day out."""
    rows = conn.execute("SELECT * FROM dfs_slates WHERE lock_utc>=? AND lock_utc<=? "
                        "ORDER BY lock_utc, slate_id",
                        (iso(now - timedelta(hours=14)), iso(now + timedelta(days=1)))).fetchall()
    slates = []
    for r in rows:
        slate = dict(r)
        tips = slate_tips(conn, slate["slate_id"])
        slates.append((slate, tips, slate_lock(slate, tips)))
    by_lock: dict = {}
    for slate, _, lock in slates:
        if lock is not None:
            by_lock.setdefault(lock, []).append(slate["slate_id"])
    return slates, by_lock


def _late_from(conn: sqlite3.Connection, slate_ids: list, lock: datetime) -> int | None:
    """Seconds before the lock until which a missed lock10 may still start (None: no room
    left after the window)."""
    est = build_estimate(conn, slate_ids, lock)
    late = int(max(LOCK10_LATE_MIN, 1.25 * est + LOCK10_LATE_PAD))
    return late if late < lock10_window(conn, slate_ids, lock)[1] - TOLERANCE_S else None


def pull_forward(conn: sqlite3.Connection, now: datetime, todo: list[dict]) -> list[dict]:
    """lock10 runs to start NOW, ahead of `todo`, because running `todo` first would carry the
    tick past their window (the tick runs one build after another and a busy tick skips the next
    launchd fire). On 2026-10-21 a 7:30 PM slate's lock45 builds held the tick from 18:45 to
    18:50 ET and the 7:00 PM main slate's lock10 window (18:46-18:51) passed unrun.

    Only lock10s whose window has not opened yet, that are not done, and that are no more than
    LOCK10_MAX_LEAD before their lock."""
    now = _utc(now)
    rest = [d for d in todo if d["kind"] != "lock10"]
    if not rest:
        return []
    est = 0.0
    for d in rest:
        if d["kind"] == "swap":
            est += SWAP_EST_S
        else:
            est += build_estimate(conn, [d["slate_id"]], _parse_ts(d["event"]), fallback=True)
    busy_until = now + timedelta(seconds=est + TICK_INTERVAL_S)
    queued = {(d["slate_id"], d["kind"]) for d in todo}
    out = []
    slates, by_lock = _slate_locks(conn, now)
    tag = {sl["slate_id"]: sl["tag"] for sl, _, _ in slates}
    for lock, sids in sorted(by_lock.items()):
        lead = (lock - now).total_seconds()
        far, near = lock10_window(conn, sids, lock)
        if not (far < lead <= LOCK10_MAX_LEAD):         # window open / passed, or too early
            continue
        if lock - timedelta(seconds=near) > busy_until:  # a later tick still lands in the window
            continue
        for sid in sids:
            if (sid, "lock10") in queued or run_done(conn, sid, "lock10", None) is not None:
                continue
            out.append({"slate_id": sid, "kind": "lock10", "trigger": None, "event": iso(lock),
                        "tag": tag.get(sid), "done": False,
                        "note": f"pulled forward {int(lead - far)} s: other builds would run "
                                "past its window"})
    return out


_RUN_ORDER = {"lock10": 0, "swap": 1, "manual": 2, "lock45": 3}


def due_runs(conn: sqlite3.Connection, now: datetime) -> list[dict]:
    """The (slate, kind[, tip]) whose window contains `now`, with ``done`` set when an ok run
    already exists. Reads ``dfs_slates`` / ``dfs_salaries`` / ``dfs_runs`` only."""
    now = _utc(now)
    slates, by_lock = _slate_locks(conn, now)
    out = []
    l10 = {lock: lock10_window(conn, sids, lock) for lock, sids in by_lock.items()}
    late = {lock: _late_from(conn, sids, lock) for lock, sids in by_lock.items()}
    for slate, tips, lock in slates:
        sid = slate["slate_id"]
        if lock is None:
            continue
        d = (lock - now).total_seconds()
        for kind, win in (("lock45", LOCK45_WINDOW), ("lock10", l10[lock])):
            if _in_window(d, win):
                out.append({"slate_id": sid, "kind": kind, "trigger": None,
                            "event": iso(lock), "tag": slate["tag"]})
        if late[lock] is not None and late[lock] <= d < l10[lock][1] - TOLERANCE_S:
            # the lock10 window passed (a long tick ate it): catch up while a build still
            # finishes before the lock (only fires when no lock10 run exists, see below)
            out.append({"slate_id": sid, "kind": "lock10", "trigger": None,
                        "event": iso(lock), "tag": slate["tag"], "late": True})
        if not is_showdown(slate):
            for t in tips:
                if t > lock and _in_window((t - now).total_seconds(), SWAP_WINDOW):
                    out.append({"slate_id": sid, "kind": "swap", "trigger": iso(t),
                                "event": iso(t), "tag": slate["tag"]})
    for d in out:
        d["done"] = run_done(conn, d["slate_id"], d["kind"], d["trigger"]) is not None
    out = [d for d in out if not (d.get("late") and d["done"])]     # catch-up only when missed
    for d in out:
        if d["kind"] == "swap" and not d["done"] and not current_lineups(conn, d["slate_id"]):
            d["done"] = True
            d["note"] = "no saved lineups to swap"
    return out


@contextlib.contextmanager
def _tick_lock():
    config.ensure_dirs()
    fh = open(config.data_dir() / ".dfs-tick.lock", "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def tick(conn: sqlite3.Connection, now: datetime | None = None, *, dry_run: bool = False,
         refresh=True, n: int = 10, min_diff: int = 1) -> list[dict]:
    """Fire what is due at `now`; returns one dict per (slate, kind) whose window contains it
    (``action``: ran | already_done | would_run | locked, plus the `run_slate` summary).

    No window hit -> [] without a request or a write. Otherwise the official report and ESPN
    injuries are refreshed once, then each due slate runs (`run_slate`, which refreshes its own
    draftables). `dry_run` reports what would run and touches nothing."""
    now = _utc(now)
    due = due_runs(conn, now)
    todo = [d for d in due if not d["done"]]
    out = [{**d, "action": "already_done"} for d in due if d["done"]]
    if not todo:
        return out
    # the lock nearest in time first, lock10s before swaps before lock45s; plus any lock10 the
    # rest of the tick would push out of its window
    todo = pull_forward(conn, now, todo) + todo
    todo.sort(key=lambda d: (d["event"], _RUN_ORDER.get(d["kind"], 9)))
    if dry_run:
        return out + [{**d, "action": "would_run"} for d in todo]
    with _tick_lock() as got:
        if not got:
            log.warning("another dfs tick is running; skipping")
            return out + [{**d, "action": "locked"} for d in todo]
        global_refresh = None
        if refresh is True:
            global_refresh = refresh_inputs(conn, now)
        for d in todo:
            res = run_slate(conn, d["slate_id"], d["kind"], now=now, n=n, min_diff=min_diff,
                            refresh=refresh, refresh_global=False,
                            tip=_parse_ts(d["trigger"]) if d["trigger"] else None)
            out.append({**d, "action": "ran", "result": res,
                        **({"inputs": global_refresh} if global_refresh is not None else {})})
    return out


if __name__ == "__main__":      # python -m courtside.proj.dfs.runner backfill-ranges
    import sys
    if sys.argv[1:] == ["backfill-ranges"]:
        from ... import db
        print(json.dumps(backfill_lineup_ranges(db.connect())))
    else:
        sys.exit("usage: python -m courtside.proj.dfs.runner backfill-ranges")
