"""Light daily DraftKings capture (docs/PLAYERS.md §3.2, T0.7): slates, payout tables, refresh.

DraftKings' terms forbid automated collection, so this is deliberately small:

* no crawl, no login, no leaderboards / standings / ownership, nothing that enters a contest
  or touches a lineup;
* the requests this module ADDS (a contest's payout table, a lock-relative slate refresh) are
  logged in ``dk_requests`` and capped at `DAILY_ADDED_BUDGET` per Eastern calendar day
  (`budget_remaining`); the always-on requests (lobby, rules, draftables of the collect step)
  are logged too, for the record, but do not draw on that budget;
* each contest's payout table is fetched ONCE, only before its slate locks.

``dk_requests`` is created by db migration 10 (and defensively on first use).

Per-run request counts (opening night 2026-10-20: 3 Classic + 3 Showdown + others in the lobby):
base 1 lobby + <= 5 draftables per pregame run (+ 2 rules on the first run of a day); added:
<= 40 contests per day in total (first runs, then nothing: already captured contests are not
fetched again) and <= 20 refreshes (the budget's remainder).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .. import archive, db
from . import dfs

log = logging.getLogger(__name__)

DAILY_ADDED_BUDGET = 60
ADDED_KINDS = ("contest", "refresh")
CONTEST_PAUSE_S = 1.0            # between payout-table fetches (on top of the host throttle)
CONTEST_MAX_CONSECUTIVE_FAILURES = 3
REFRESH_ATTEMPTS = 2
REFRESH_MIN_INTERVAL_S = 120
ET = ZoneInfo("America/New_York")
_sleep = time.sleep              # monkeypatch in tests


class BudgetExhausted(RuntimeError):
    pass


def _utc(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- request budget

def ensure_log(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dk_requests (
            id INTEGER PRIMARY KEY,
            ts TEXT NOT NULL,
            et_day TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref TEXT,
            ok INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_dk_requests_day ON dk_requests(et_day, kind);
    """)


def et_day(now: datetime) -> str:
    return _utc(now).astimezone(ET).date().isoformat()


def log_request(conn: sqlite3.Connection, kind: str, ref: str | None = None, *,
                now: datetime | None = None, ok: bool | None = None) -> int:
    """Record one DK request (call BEFORE sending it; `mark_request` settles ok)."""
    ensure_log(conn)
    now = _utc(now)
    cur = conn.execute("INSERT INTO dk_requests (ts, et_day, kind, ref, ok) VALUES (?,?,?,?,?)",
                       (iso(now), et_day(now), kind, ref, None if ok is None else int(ok)))
    conn.commit()
    return cur.lastrowid


def mark_request(conn: sqlite3.Connection, req_id: int, ok: bool) -> None:
    conn.execute("UPDATE dk_requests SET ok=? WHERE id=?", (int(ok), req_id))
    conn.commit()


def requests_today(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    """{"day", "total", "added", "budget", "remaining", "by_kind"} for the Eastern day of `now`."""
    ensure_log(conn)
    day = et_day(_utc(now))
    by_kind = {k: n for k, n in conn.execute(
        "SELECT kind, COUNT(*) FROM dk_requests WHERE et_day=? GROUP BY kind", (day,))}
    added = sum(by_kind.get(k, 0) for k in ADDED_KINDS)
    return {"day": day, "total": sum(by_kind.values()), "added": added,
            "budget": DAILY_ADDED_BUDGET, "remaining": max(0, DAILY_ADDED_BUDGET - added),
            "by_kind": by_kind}


def budget_remaining(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    return requests_today(conn, now)["remaining"]


def counted_fetch(conn: sqlite3.Connection, kind: str, ref: str, fn, *,
                  now: datetime | None = None, attempts: int = 1):
    """Call `fn()` (one HTTP request) for an ADDED request `kind`, logging each attempt first and
    refusing (`BudgetExhausted`) once the daily budget is spent. Retries only on transient
    errors (5xx / 429 / network / blocked), never on a 404 / other 4xx."""
    assert kind in ADDED_KINDS
    last: Exception | None = None
    for attempt in range(attempts):
        if budget_remaining(conn, now) <= 0:
            raise BudgetExhausted(f"daily DK request budget ({DAILY_ADDED_BUDGET}) is spent")
        rid = log_request(conn, kind, ref, now=now)
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            mark_request(conn, rid, False)
            last = e
            status = getattr(e, "status", None)
            if status is not None and 400 <= status < 500 and status not in (403, 429):
                raise
            if attempt + 1 < attempts:
                _sleep(2.0 * (attempt + 1))
            continue
        mark_request(conn, rid, True)
        return res
    assert last is not None
    raise last


# --------------------------------------------------------------------------- slates

def _lock_utc(payload: dict, fallback: str | None) -> str | None:
    starts = [c["start_time"] for c in dfs.competitions(payload).values() if c.get("start_time")]
    first = min(starts) if starts else fallback
    return (first[:19] + "Z") if first else None


def upsert_slate(conn: sqlite3.Connection, *, slate_id, day: date, game_type_id: int, tag: str,
                 game_ids: list[str], lock_utc: str | None, rules: dict | None,
                 captured_at: str, raw_path: str | None) -> None:
    conn.execute(
        """INSERT INTO dfs_slates (slate_id, site, nba_day, game_type_id, tag, game_ids_json,
                                   lock_utc, rules_json, captured_at, raw_path)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(slate_id) DO UPDATE SET
             nba_day=excluded.nba_day, game_type_id=excluded.game_type_id,
             tag=COALESCE(excluded.tag, dfs_slates.tag),
             game_ids_json=excluded.game_ids_json,
             lock_utc=COALESCE(excluded.lock_utc, dfs_slates.lock_utc),
             rules_json=COALESCE(excluded.rules_json, dfs_slates.rules_json),
             captured_at=excluded.captured_at, raw_path=excluded.raw_path""",
        (str(slate_id), dfs.SITE, day.isoformat(), game_type_id, tag,
         json.dumps(game_ids), lock_utc, json.dumps(rules) if rules is not None else None,
         captured_at, raw_path))
    conn.commit()


def _fingerprint(rows: list[dict]) -> tuple:
    return tuple(sorted((r["draftable_id"], r["salary"], r["status"], r["disabled"],
                         r.get("swappable"), r.get("starter"), r.get("news_status"))
                        for r in rows))


def games_by_teams(conn: sqlite3.Connection, day: date) -> dict[tuple, str]:
    return {(r["home"], r["away"]): r["game_id"] for r in conn.execute(
        "SELECT game_id, home, away FROM games WHERE nba_day=?", (day.isoformat(),))}


def store_draftables(conn: sqlite3.Connection, payload: dict, *, slate_id, game_type_id: int,
                     day: date, tag: str | None, rules: dict | None, lobby: dict | None,
                     start_utc: str | None, captured_at: str, now: datetime,
                     games: dict[tuple, str] | None = None) -> dict:
    """Archive-first store of one draft group's draftables: raw snapshot, a new
    ``dfs_salaries`` capture when anything moved (salary, status, disabled, swappable, starter,
    news), and the ``dfs_slates`` upsert. Returns the per-slate summary of the collect step."""
    path = archive.save_raw("draftkings", "draftgroup", day, f"draftgroup_{slate_id}", payload,
                            snapshot=True, now=now)
    rows = dfs.parse_draftables(payload, draft_group_id=slate_id, game_type_id=game_type_id,
                                rules=rules, lobby=lobby)
    games = games if games is not None else games_by_teams(conn, day)
    for r in rows:
        r["game_id"] = games.get((r["home"], r["away"]))
        r.update(captured_at=captured_at, raw_path=str(path))
    last = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE site=? AND slate_id=?",
                        (dfs.SITE, str(slate_id))).fetchone()[0]
    prev = [dict(r) for r in conn.execute(
        "SELECT draftable_id, salary, status, disabled, swappable, starter, news_status "
        "FROM dfs_salaries WHERE site=? AND slate_id=? AND captured_at=?",
        (dfs.SITE, str(slate_id), last))] if last else []
    unchanged = bool(prev) and _fingerprint(prev) == _fingerprint(rows)
    n = 0 if unchanged else db.insert_rows(conn, "dfs_salaries", rows)
    game_ids = sorted({r["game_id"] for r in rows if r["game_id"]})
    upsert_slate(conn, slate_id=slate_id, day=day, game_type_id=game_type_id,
                 tag=tag or "other", game_ids=game_ids, lock_utc=_lock_utc(payload, start_utc),
                 rules=rules, captured_at=captured_at, raw_path=str(path))
    return {"draftables": len(rows), "players": len({r["site_player_id"] for r in rows}),
            "inserted": n, "unchanged": unchanged, "tag": tag, "raw_path": str(path),
            "starters": len({r["site_player_id"] for r in rows if r["starter"]}),
            "games": [f'{x["away"]}@{x["home"]}' for x in dfs.slate_games(payload)]}


# --------------------------------------------------------------------------- contests

def _slates_of_day(conn: sqlite3.Connection, day: date) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM dfs_slates WHERE nba_day=? ORDER BY lock_utc, slate_id",
        (day.isoformat(),))]


def select_for_day(conn: sqlite3.Connection, lobby: dict, day: date, *,
                   now: datetime) -> tuple[list[dict], dict]:
    """The contests still to fetch for `day` (not in ``dfs_contests``, slate not locked) and a
    small context dict. Caps count what is already captured, so repeated runs never exceed
    `dfs.DAY_CONTEST_CAP` (40) contests, `dfs.MAX_SHOWDOWN_CONTESTS` (3) Showdowns."""
    now = _utc(now)
    slates = _slates_of_day(conn, day)
    open_ = [s for s in slates if s["lock_utc"] and s["lock_utc"] > iso(now)]
    main = next((s for s in open_ if s["tag"] == "main" and s["game_type_id"] == 70), None)
    showdowns = [s["slate_id"] for s in open_ if s["tag"] == "showdown"]
    ids = [s["slate_id"] for s in slates]
    have: dict[str, str] = {}
    if ids:
        have = {r[0]: r[1] for r in conn.execute(
            f"SELECT contest_id, slate_id FROM dfs_contests WHERE slate_id IN "
            f"({','.join('?' * len(ids))})", ids)}
    show_ids = {s["slate_id"] for s in slates if s["tag"] == "showdown"}
    main_ids = {s["slate_id"] for s in slates if s["tag"] == "main"}
    have_main = sum(1 for v in have.values() if v in main_ids)
    have_show_slates = {v for v in have.values() if v in show_ids}
    picks: list[dict] = []
    if main is not None:
        sel = dfs.select_main_contests(dfs.lobby_contests(lobby, main["slate_id"]))
        fresh = [c for c in sel if c["contest_id"] not in have]
        picks += fresh[:max(0, dfs.MAIN_CONTEST_CAP - have_main)]
    todo_show = [s for s in showdowns if s not in have_show_slates]
    room = dfs.MAX_SHOWDOWN_CONTESTS - len(have_show_slates)
    picks += [c for c in dfs.select_showdown_contests(lobby, todo_show, limit=room)
              if c["contest_id"] not in have]
    for c in picks:
        c["slate_id"] = str(c["draft_group_id"])
    ctx = {"main_slate": main["slate_id"] if main else None, "showdown_slates": showdowns,
           "already_captured": len(have)}
    return picks, ctx


def _store_contest(conn: sqlite3.Connection, row: dict) -> bool:
    cols = list(row)
    cur = conn.execute(f"INSERT OR IGNORE INTO dfs_contests ({','.join(cols)}) "
                       f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
    conn.commit()
    return cur.rowcount > 0


def capture_contests(conn: sqlite3.Connection, lobby: dict, day: date, *,
                     now: datetime | None = None, pause: float | None = None,
                     fetch=None) -> dict:
    """Fetch (once) and store the payout tables selected by `dfs.select_*` for `day`.

    Archive-first (``raw/draftkings/contest/<day>/contest_<id>.json.gz``; an existing archive
    file is reused without a request). A failing contest is logged and skipped; three
    consecutive failures stop the run (no hammering a blocked host). Only slates whose lock is
    still ahead are touched, and every request draws on the daily budget."""
    now = _utc(now)
    pause = CONTEST_PAUSE_S if pause is None else pause
    fetch = fetch or dfs.fetch_contest
    picks, ctx = select_for_day(conn, lobby, day, now=now)
    out = {**ctx, "selected": len(picks), "stored": 0, "requests": 0, "reused_archive": 0,
           "failed": {}, "stopped": None, "by_type": {}, "fees": None}
    stored_fees = []
    streak = 0
    for i, c in enumerate(picks):
        cid = c["contest_id"]
        path = archive.raw_path("draftkings", "contest", day, f"contest_{cid}")
        try:
            if path.exists():
                payload = archive.load_raw(path)
                out["reused_archive"] += 1
            else:
                if i and out["requests"] and pause:
                    _sleep(pause)
                payload = counted_fetch(conn, "contest", cid, lambda cid=cid: fetch(cid), now=now)
                out["requests"] += 1
                archive.save_raw("draftkings", "contest", day, f"contest_{cid}", payload)
            row = dfs.parse_contest(payload, lobby_row=c, slate_id=c["slate_id"],
                                    captured_at=iso(now), raw_path=str(path))
            if _store_contest(conn, row):
                out["stored"] += 1
                stored_fees.append(row["entry_fee"])
                why = "+".join(c["selected_as"])
                out["by_type"][why] = out["by_type"].get(why, 0) + 1
            streak = 0
        except BudgetExhausted as e:
            out["stopped"] = f"budget: {e}"
            log.warning("dk contest capture stopped: %s", e)
            break
        except Exception as e:  # noqa: BLE001 - payouts must never fail the salary capture
            streak += 1
            out["failed"][cid] = f"{type(e).__name__}: {e}"
            log.warning("dk contest %s failed: %s", cid, e)
            if streak >= CONTEST_MAX_CONSECUTIVE_FAILURES:
                out["stopped"] = f"{streak} consecutive failures"
                break
    if stored_fees:
        out["fees"] = [min(stored_fees), max(stored_fees)]
    return out


# --------------------------------------------------------------------------- refresh

def refresh_slate(conn: sqlite3.Connection, slate_id, *, reason: str,
                  now: datetime | None = None, fetch=None, force: bool = False,
                  min_interval_s: int = REFRESH_MIN_INTERVAL_S) -> dict:
    """Re-fetch ONE slate's draftables (status, disabled, swappable, starter, news) and store a
    new ``dfs_salaries`` capture if anything moved. For the lock-relative runs (T1.6).

    Idempotent: nothing is written when the payload is unchanged, and the same
    (slate, reason) within `min_interval_s` is a no-op without a request (unless `force`).
    Draws on the daily budget (`REFRESH_ATTEMPTS` tries at most). Never raises for budget or
    fetch trouble: look at ``status`` ('ok' | 'recent' | 'budget_exhausted' | 'error' |
    'unknown_slate')."""
    now = _utc(now)
    sid = str(slate_id)
    row = conn.execute("SELECT * FROM dfs_slates WHERE slate_id=?", (sid,)).fetchone()
    if row is None:
        return {"status": "unknown_slate", "slate_id": sid, "reason": reason}
    slate = dict(row)
    ref = f"{sid}:{reason}"
    ensure_log(conn)
    if not force:
        last = conn.execute("SELECT MAX(ts) FROM dk_requests WHERE kind='refresh' AND ref=? "
                            "AND ok=1", (ref,)).fetchone()[0]
        if last and last > iso(now - timedelta(seconds=min_interval_s)):
            return {"status": "recent", "slate_id": sid, "reason": reason, "last": last}
    fetch = fetch or (lambda s: dfs.http.fetch_json(
        f"{dfs.API}/draftgroups/v1/draftgroups/{s}/draftables", retries=0))
    try:
        payload = counted_fetch(conn, "refresh", ref, lambda: fetch(sid), now=now,
                                attempts=REFRESH_ATTEMPTS)
    except BudgetExhausted as e:
        return {"status": "budget_exhausted", "slate_id": sid, "reason": reason, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        log.warning("dk refresh %s failed: %s", sid, e)
        return {"status": "error", "slate_id": sid, "reason": reason,
                "error": f"{type(e).__name__}: {e}"}
    rules = json.loads(slate["rules_json"]) if slate["rules_json"] else None
    res = store_draftables(conn, payload, slate_id=sid, game_type_id=slate["game_type_id"],
                           day=date.fromisoformat(slate["nba_day"]), tag=slate["tag"],
                           rules=rules, lobby=None, start_utc=slate["lock_utc"],
                           captured_at=iso(now), now=now)
    return {"status": "ok", "slate_id": sid, "reason": reason, **res,
            "locked": bool(slate["lock_utc"] and slate["lock_utc"] <= iso(now)),
            "budget_remaining": budget_remaining(conn, now)}
