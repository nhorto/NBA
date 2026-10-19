"""Collection jobs: `run_pregame(day)` and `run_postgame(day)`.

Both jobs: take a process-wide non-blocking flock, log to stdout and
`<data>/logs/collect.log`, record a `collect_runs` row, and run every step in its own
try/except so one broken source never stops the others. Each step archives raw data first,
then parses it into SQLite. The returned summary is JSON-serialisable.

Conventions
-----------
* The schedule is cached per *today's* NBA day (`nba_cdn/schedule/<today>/scheduleLeagueV2__*`).
  A cached (older) schedule never regresses a game that is already live/final in the DB.
* The ESPN summary is archived keyed by NBA game id (like the other per-game kinds):
  `raw/espn/summary/<nba_day>/<game_id>.json.gz`.
* Postgame for a day with no `games` rows (any past season) falls back to the game ids in
  `team_games` (2000-01 onward): boxscore + play-by-play are archived for them (no ESPN summary:
  the ESPN event id is unknown, so ESPN events are reported as `unmatched`).
* After archiving, per-game analytics facts go to `game_facts`; for a day in the current season
  the season's game logs are re-fetched (keeps `team_games`/`player_games` current for the model
  and trends); finally the Brief for day + 1 (the morning-of Brief) is built. Pregame ends with
  `predict_day` and a rebuild of today's Brief.
* Pre-tip extras (props, referees, action_network, dfs_salaries) are archive-first sources
  whose data disappears at tip-off; see the "pre-tip extras" block below for their conventions
  (raw layout, change-log semantics, request volume).
* Official injury report: archived as `raw/nba_official/injury_report/<report_et_date>/
  Injury-Report_YYYY-MM-DD_HH_MMPM.pdf` (stable key => never re-downloaded). Rows are stored
  with `player_name = player_name_norm` and `captured_at = report publication time (UTC)`,
  since that is when the information became public. An archived PDF that was never parsed
  (parser was missing) is parsed from disk once the parser appears.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

from . import archive, config, db, teams
from .http import FetchError
from .sources import action_network, dfs, dfs_capture, espn, espn_props, nba_cdn, referees

log = logging.getLogger("courtside.collect")

SCHEDULE_KEY = "scheduleLeagueV2"
OFFICIAL_SOURCE = "nba_official"
_RANK = {"scheduled": 0, "live": 1, "final": 2}


# ------------------------------------------------------------------ infrastructure

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextlib.contextmanager
def _process_lock():
    """Yield True if the exclusive non-blocking flock was acquired, else False."""
    config.ensure_dirs()
    fh = open(config.data_dir() / ".collect.lock", "a+")
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


@contextlib.contextmanager
def _logging_ctx():
    """Attach stdout + file handlers to the `courtside` logger for the duration of a run."""
    root = logging.getLogger("courtside")
    logs = config.data_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(logs / "collect.log")]
    old_level = root.level
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.setLevel(old_level)
        for h in handlers:
            root.removeHandler(h)
            h.close()


class _Run:
    """Collects per-step results and errors."""

    def __init__(self, job: str, day: date):
        self.summary: dict = {"job": job, "day": day.isoformat(), "ok": True, "steps": {},
                              "errors": []}

    def step(self, name: str, fn, *args, **kw):
        log.info("step %s: start", name)
        try:
            res = fn(*args, **kw)
        except Exception as e:  # noqa: BLE001 - one broken source must not stop the others
            log.exception("step %s failed", name)
            msg = f"{type(e).__name__}: {e}"
            self.summary["steps"][name] = {"status": "error", "error": msg}
            self.summary["errors"].append({"step": name, "error": msg})
            self.summary["ok"] = False
            return None
        res = res if isinstance(res, dict) else {}
        self.summary["steps"][name] = {"status": res.pop("status", "ok"), **res}
        log.info("step %s: %s", name, self.summary["steps"][name])
        return res

    def error(self, where: str, msg: str):
        log.error("%s: %s", where, msg)
        self.summary["errors"].append({"step": where, "error": msg})
        self.summary["ok"] = False


def _run_job(job: str, day: date, body) -> dict:
    with _process_lock() as got:
        if not got:
            with _logging_ctx():
                log.warning("%s: another collect job holds the lock; skipping", job)
            return {"skipped": "locked"}
        with _logging_ctx():
            conn = db.connect()
            run_id = db.start_run(conn, job)
            run = _Run(job, day)
            run.summary["run_id"] = run_id
            try:
                body(conn, run, day)
            except Exception as e:  # noqa: BLE001
                log.exception("%s crashed", job)
                run.error("job", f"{type(e).__name__}: {e}")
            finally:
                try:
                    db.finish_run(conn, run_id, run.summary["ok"],
                                  json.dumps(run.summary, default=str))
                finally:
                    conn.close()
            log.info("%s done: %s", job, json.dumps(run.summary, default=str))
            return run.summary


# ------------------------------------------------------------------ games table helpers

_UPSERT_SQL = """
INSERT INTO games (game_id, espn_id, season, nba_day, tipoff_utc, home, away, status,
                   home_score, away_score, updated_at)
VALUES (:game_id, :espn_id, :season, :nba_day, :tipoff_utc, :home, :away, :status,
        :home_score, :away_score, :updated_at)
ON CONFLICT(game_id) DO UPDATE SET
    espn_id    = COALESCE(excluded.espn_id,    games.espn_id),
    season     = COALESCE(excluded.season,     games.season),
    nba_day    = COALESCE(excluded.nba_day,    games.nba_day),
    tipoff_utc = COALESCE(excluded.tipoff_utc, games.tipoff_utc),
    home       = COALESCE(excluded.home,       games.home),
    away       = COALESCE(excluded.away,       games.away),
    -- a stale 'scheduled' (cached schedule) must not undo live/final
    status     = CASE WHEN excluded.status = 'scheduled' AND games.status IN ('live', 'final')
                      THEN games.status ELSE excluded.status END,
    home_score = COALESCE(excluded.home_score, games.home_score),
    away_score = COALESCE(excluded.away_score, games.away_score),
    updated_at = excluded.updated_at
"""


def _upsert_games(conn: sqlite3.Connection, rows: list[dict]) -> dict:
    """Bulk upsert nba_cdn game rows in ONE transaction (None never overwrites)."""
    before = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    now = _iso(datetime.now(timezone.utc))
    params = [{**{c: None for c in ("game_id", *db.GAME_COLUMNS)}, **r,
               "status": r.get("status") or "scheduled", "updated_at": now} for r in rows]
    with conn:
        conn.executemany(_UPSERT_SQL, params)
    after = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    return {"rows": len(rows), "new": after - before}


def _set_game_result(conn: sqlite3.Connection, game_id: str, status: str,
                     home_score: int | None, away_score: int | None) -> None:
    """Update status/scores, never downgrading a final game (postponed may not overwrite final)."""
    cur = conn.execute("SELECT status FROM games WHERE game_id=?", (game_id,)).fetchone()
    if cur is None:
        return
    if cur["status"] == "final" and status != "final":
        return
    with conn:
        conn.execute(
            "UPDATE games SET status=?, home_score=COALESCE(?, home_score), "
            "away_score=COALESCE(?, away_score), updated_at=? WHERE game_id=?",
            (status, home_score, away_score, _iso(datetime.now(timezone.utc)), game_id))


# ------------------------------------------------------------------ shared steps

def _schedule_step(conn: sqlite3.Connection) -> dict:
    today = config.today_nba_day()
    folder = archive.raw_path("nba_cdn", "schedule", today, SCHEDULE_KEY).parent
    payload = None
    cached_path: Path | None = None
    for p in sorted(folder.glob(f"{SCHEDULE_KEY}__*.json.gz"), reverse=True):
        try:
            payload, cached_path = archive.load_raw(p), p
            break
        except Exception:  # noqa: BLE001 - corrupt cache file: try older, else refetch
            log.warning("unreadable cached schedule %s", p)
    if payload is not None:
        out = {"cached": True, "path": str(cached_path)}
    else:
        payload = nba_cdn.fetch_schedule()
        path = archive.save_raw("nba_cdn", "schedule", today, SCHEDULE_KEY, payload, snapshot=True)
        out = {"cached": False, "path": str(path)}
    rows = nba_cdn.schedule_games(payload)
    out.update(_upsert_games(conn, rows))
    return out


def _espn_scoreboard_step(conn: sqlite3.Connection, day: date, *, apply_results: bool,
                          insert_odds: bool, captured_at: str) -> dict:
    payload = espn.fetch_scoreboard(day)
    path = archive.save_raw("espn", "scoreboard", day, "scoreboard", payload, snapshot=True)
    events = espn.scoreboard_events(payload)
    by_espn: dict[str, str] = {}
    unmatched: list[str] = []
    for ev in events:
        row = conn.execute("SELECT game_id FROM games WHERE nba_day=? AND home=? AND away=?",
                           (ev["nba_day"], ev["home"], ev["away"])).fetchone()
        if row is None:
            unmatched.append(f'{ev["espn_id"]} {ev["away"]}@{ev["home"]} {ev["nba_day"]}')
            continue
        gid = row["game_id"]
        by_espn[ev["espn_id"]] = gid
        with conn:
            conn.execute("UPDATE games SET espn_id=? WHERE game_id=? "
                         "AND COALESCE(espn_id,'') != ?", (ev["espn_id"], gid, ev["espn_id"]))
        if apply_results and ev["status"] in ("final", "postponed"):
            _set_game_result(conn, gid, ev["status"], ev["home_score"], ev["away_score"])
    if unmatched:
        log.warning("espn events with no matching game: %s", unmatched)
    out: dict = {"path": str(path), "events": len(events), "matched": len(by_espn),
                 "unmatched": unmatched}
    if insert_odds:
        final_ids = {e["espn_id"] for e in events if e["status"] == "final"}
        inserted, skipped = 0, []
        for o in espn.scoreboard_odds(payload):
            gid = by_espn.get(o["espn_id"])
            if gid is None:
                skipped.append(o["espn_id"])
                log.warning("espn odds for unmatched event %s (%s@%s) skipped",
                            o["espn_id"], o["away"], o["home"])
                continue
            if o["espn_id"] in final_ids:
                continue
            db.insert_odds_snapshot(
                conn, game_id=gid, captured_at=captured_at, source=o["source"],
                provider=o["provider"], home_spread=o["home_spread"], total=o["total"],
                home_ml=o["home_ml"], away_ml=o["away_ml"], raw_path=str(path))
            inserted += 1
        out.update(odds_inserted=inserted, odds_skipped=skipped)
    return out


def _monitor_hook(conn: sqlite3.Connection, run: _Run, job: str, day: date) -> None:
    """MLOps monitoring after a job (data checks, metrics, drift, alerts; docs/MLOPS.md). Never
    fatal and never marks the job failed: problems show up as checks/alerts, not exit codes."""
    try:
        from .mlops import monitor
        res = monitor.run_after_job(conn, job, day, run.summary)
        run.summary["health"] = {k: v for k, v in res.items() if k != "line"}
    except Exception:  # noqa: BLE001
        log.exception("monitoring hook failed (ignored)")


# ------------------------------------------------------------------ pregame steps

def _nba_scoreboard_step(conn: sqlite3.Connection, day: date) -> dict:
    payload = nba_cdn.fetch_scoreboard()
    path = archive.save_raw("nba_cdn", "scoreboard", day, "scoreboard", payload, snapshot=True)
    return {"path": str(path), **_upsert_games(conn, nba_cdn.scoreboard_games(payload))}


def _nba_odds_step(conn: sqlite3.Connection, day: date, captured_at: str) -> dict:
    payload = nba_cdn.fetch_odds()
    path = archive.save_raw("nba_cdn", "odds", day, "odds", payload, snapshot=True)
    known = {r[0] for r in conn.execute("SELECT game_id FROM games")}
    inserted, skipped = 0, []
    for r in nba_cdn.odds_rows(payload):
        if r["game_id"] not in known:
            skipped.append(r["game_id"])
            continue
        db.insert_odds_snapshot(conn, captured_at=captured_at, raw_path=str(path), **r)
        inserted += 1
    return {"path": str(path), "odds_inserted": inserted, "odds_skipped": skipped}


def _espn_injuries_step(conn: sqlite3.Connection, day: date, captured_at: str) -> dict:
    payload = espn.fetch_injuries()
    path = archive.save_raw("espn", "injuries", day, "injuries", payload, snapshot=True)
    rows = [{**r, "captured_at": captured_at, "raw_path": str(path)}
            for r in espn.injury_rows(payload)]
    # one shared captured_at is set above; db helper only inserts what it is given
    n = db.insert_injury_snapshots(conn, rows) if rows else 0
    return {"path": str(path), "rows": n}


def _official_report_step(conn: sqlite3.Connection, day: date) -> dict:
    try:
        from .sources import injury_report as ir
    except ImportError:
        log.warning("injury_report module not available; skipping official report")
        return {"status": "skipped", "reason": "module_missing"}
    unavailable = getattr(ir, "ParserUnavailable", None) or ()

    found = ir.find_latest_report()
    if not found:
        return {"status": "skipped", "reason": "no_report"}
    url, report_dt = found
    if report_dt.tzinfo is None:
        report_dt = report_dt.replace(tzinfo=config.NBA_TZ)
    et = report_dt.astimezone(config.NBA_TZ)
    key = "Injury-Report_" + et.strftime("%Y-%m-%d_%I_%M%p")
    path = archive.raw_path(OFFICIAL_SOURCE, "injury_report", et.date(), key, ext=".pdf")
    out: dict = {"key": key, "path": str(path)}

    if path.exists():
        pdf, out["archived"] = path.read_bytes(), "existing"
        already = conn.execute("SELECT 1 FROM injury_snapshots WHERE raw_path=? LIMIT 1",
                               (str(path),)).fetchone()
        if already:
            return {**out, "status": "skipped", "reason": "already_collected"}
    else:
        pdf = ir.fetch_report(url)
        archive.save_raw_bytes(OFFICIAL_SOURCE, "injury_report", et.date(), key, pdf, ext=".pdf")
        out["archived"] = "new"

    if not ir.parser_available():
        return {**out, "parsed": False, "reason": "parser_unavailable"}
    try:
        parsed = ir.parse_report(pdf)
    except unavailable:
        return {**out, "parsed": False, "reason": "parser_unavailable"}
    captured = _iso(report_dt)
    rows = [{"captured_at": captured, "source": OFFICIAL_SOURCE, "team": r["team"],
             "player_name": r.get("player_name_norm") or r["player_name"], "player_id": None,
             "status": r.get("status"), "reason": r.get("reason"), "raw_path": str(path)}
            for r in parsed if r.get("player_name_norm") or r.get("player_name")]
    # "NOT YET SUBMITTED" placeholder rows have no player; player_name is NOT NULL in the table
    n = db.insert_injury_snapshots(conn, rows) if rows else 0
    return {**out, "parsed": True, "rows": n, "skipped_unnamed": len(parsed) - len(rows)}


# ------------------------------------------------------------------ pre-tip extras
#
# props          ESPN/DraftKings player props     raw/espn/props/<day>/<game>_<provider>_pNN__<ts>
# referees       official.nba.com crews (>= 9:00 ET) raw/nba_official/referee_assignments/<day>/...html
# action_network per-book lines + opening line    raw/actionnetwork/scoreboard/<day>/scoreboard__<ts>
# dfs_salaries   DraftKings DFS lobby + salaries  raw/draftkings/{lobby,draftgroup,rules}/<day>/...
#
# Request volume per pregame run (6 runs/day): props ~ 5 per upcoming game (providers + 2 pages
# of 1000 + 2 rosters the first time a team is seen), referees 1, action_network 1, dfs 1 lobby +
# 1 per Classic/Showdown draft group (+ rules once per day) + the payout tables of <= 40
# selected contests per day, each once (<= 60 ADDED requests a day, logged in dk_requests).
# Each source is polite (the shared per-host throttle) and its failure never blocks the others.

PROPS_ATHLETE_LOOKUP_CAP = 40     # athlete-document fallbacks per game (rosters cover ~all)
DFS_MAX_GROUPS = 12
REFEREES_AFTER_ET = dtime(9, 0)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _props_names(conn: sqlite3.Connection, game: sqlite3.Row, ids: set[str], day: date,
                 now: datetime, reqs: dict) -> dict[str, str]:
    """ESPN athlete id -> name: DB cache, else the two team rosters, else athlete documents."""
    names: dict[str, str] = {}
    if not ids:
        return names
    q = ",".join("?" * len(ids))
    for aid, nm in conn.execute(
            f"SELECT espn_athlete_id, player_name FROM prop_snapshots WHERE player_name IS NOT NULL "
            f"AND espn_athlete_id IN ({q}) GROUP BY espn_athlete_id", sorted(ids)):
        names[aid] = nm
    if ids - names.keys():
        for tri in (game["home"], game["away"]):
            try:
                roster = espn_props.fetch_roster(teams.by_tricode(tri).espn_abbr)
                reqs["roster"] += 1
                archive.save_raw("espn", "roster", day, f"{game['game_id']}_{tri}", roster,
                                 snapshot=True, now=now)
                names.update(espn_props.roster_names(roster))
            except Exception as e:  # noqa: BLE001 - names are best effort
                log.warning("espn roster %s failed: %s", tri, e)
    for aid in sorted(ids - names.keys())[:PROPS_ATHLETE_LOOKUP_CAP]:
        try:
            nm = espn_props.athlete_name(espn_props.fetch_athlete(aid))
            reqs["athlete"] += 1
            if nm:
                names[aid] = nm
        except Exception as e:  # noqa: BLE001
            log.warning("espn athlete %s lookup failed: %s", aid, e)
    return names


def _props_changed(conn: sqlite3.Connection, game_id: str, rows: list[dict]) -> list[dict]:
    """Rows whose (line, over, under) differ from the latest stored state of the same market
    (prop_snapshots is a change log; the raw pages keep every poll)."""
    last: dict[tuple, tuple] = {}
    for r in conn.execute(
            "SELECT provider, espn_athlete_id, espn_team_id, market_id, kind, line, over_odds, "
            "under_odds FROM prop_snapshots WHERE game_id=? ORDER BY captured_at, id", (game_id,)):
        key = (r["provider"], r["espn_athlete_id"], r["espn_team_id"], r["market_id"], r["kind"],
               r["line"] if r["kind"] == "milestone" else None)
        last[key] = (r["line"], r["over_odds"], r["under_odds"])
    return [r for r in rows
            if last.get(espn_props.row_key(r)) != espn_props.row_state(r)]


def _props_step(conn: sqlite3.Connection, run: _Run, day: date, captured_at: str) -> dict:
    now = _utcnow()
    games = conn.execute(
        "SELECT game_id, espn_id, home, away, tipoff_utc FROM games WHERE nba_day=? "
        "AND status='scheduled' AND espn_id IS NOT NULL AND (tipoff_utc IS NULL OR tipoff_utc>?) "
        "ORDER BY tipoff_utc, game_id", (day.isoformat(), _iso(now))).fetchall()
    if not games:
        return {"status": "skipped", "reason": "no upcoming games with an ESPN event id"}
    reqs = {"providers": 0, "pages": 0, "roster": 0, "athlete": 0}
    per_game: dict[str, dict] = {}
    failed: dict[str, str] = {}
    total_rows = total_new = 0
    for g in games:
        gid, eid = g["game_id"], g["espn_id"]
        try:
            prov_payload = espn_props.fetch_providers(eid)
            reqs["providers"] += 1
            archive.save_raw("espn", "props", day, f"{gid}_providers", prov_payload,
                             snapshot=True, now=now)
            provs = [p for p in espn_props.providers(prov_payload) if p["has_props"]]
            provs.sort(key=lambda p: (p["id"] != "100", p["id"]))   # DraftKings first
            rows: list[dict] = []
            first_path: dict[str, str] = {}
            for p in provs:
                def _save(n, payload, _p=p):
                    path = archive.save_raw("espn", "props", day, f"{gid}_{_p['id']}_p{n:02d}",
                                            payload, snapshot=True, now=now)
                    first_path.setdefault(_p["id"], str(path))
                    reqs["pages"] += 1
                pages = espn_props.fetch_propbets(eid, p["id"], on_page=_save)
                rows += [{**r, "raw_path": first_path.get(p["id"])}
                         for r in espn_props.parse_propbets(
                             pages, provider_id=p["id"], provider_name=p["name"])]
            names = _props_names(conn, g, espn_props.athlete_ids(rows), day, now, reqs)
            espn_props.attach_names(rows, names)
            fresh = _props_changed(conn, gid, rows)
            for r in fresh:
                r.update(game_id=gid, captured_at=captured_at)
            db.insert_rows(conn, "prop_snapshots", fresh)
            total_rows += len(rows)
            total_new += len(fresh)
            per_game[gid] = {"providers": [p["id"] for p in provs], "rows": len(rows),
                             "inserted": len(fresh),
                             "unnamed": sum(1 for r in rows if r["subject"] == "player"
                                            and not r.get("player_name"))}
        except Exception as e:  # noqa: BLE001 - one game must not stop the others
            log.exception("props for %s failed", gid)
            failed[gid] = f"{type(e).__name__}: {e}"
            run.error(f"props:{gid}", failed[gid])
    return {"games": len(games), "rows": total_rows, "inserted": total_new,
            "requests": sum(reqs.values()), "requests_by_kind": reqs, "per_game": per_game,
            "failed": failed}


def _referees_step(conn: sqlite3.Connection, day: date, captured_at: str) -> dict:
    now = _utcnow()
    if day != config.today_nba_day():
        return {"status": "skipped", "reason": "day is not today"}
    if now.astimezone(config.NBA_TZ).time() < REFEREES_AFTER_ET:
        return {"status": "skipped", "reason": "before 9:00 ET (assignments not posted yet)"}
    html = referees.fetch_page()
    path = archive.save_raw_bytes("nba_official", "referee_assignments", day,
                                  "referee-assignments", html, snapshot=True, now=now, ext=".html")
    parsed = referees.parse_assignments(html)
    out: dict = {"path": str(path), "page_day": parsed["nba_day"].isoformat()
                 if parsed["nba_day"] else None, "games": len(parsed["games"])}
    if not parsed["games"]:
        return {**out, "inserted": 0, "empty": True}
    if parsed["nba_day"] is not None and parsed["nba_day"] != day:
        return {**out, "status": "skipped", "inserted": 0,
                "reason": f"page is for {parsed['nba_day']}, not {day}"}
    rows = referees.assignment_rows({**parsed, "nba_day": day})
    unmatched = []
    for r in rows:
        r["game_id"] = None
        if r["home"] and r["away"]:
            g = conn.execute("SELECT game_id FROM games WHERE nba_day=? AND home=? AND away=?",
                             (day.isoformat(), r["home"], r["away"])).fetchone()
            r["game_id"] = g["game_id"] if g else None
            if g is None and r["game_label"] not in unmatched:
                unmatched.append(r["game_label"])
    sig = lambda rs: sorted((x["game_label"] or "", x["role"], x["official_name"]) for x in rs)
    prev = conn.execute(
        "SELECT game_label, role, official_name FROM referee_assignments WHERE nba_day=? AND "
        "captured_at=(SELECT MAX(captured_at) FROM referee_assignments WHERE nba_day=?)",
        (day.isoformat(), day.isoformat())).fetchall()
    prev_sig = sorted((x["game_label"] or "", x["role"], x["official_name"]) for x in prev)
    out["unmatched"] = unmatched
    if prev_sig == sig(rows):
        return {**out, "inserted": 0, "unchanged": True}
    for r in rows:
        r.update(captured_at=captured_at, raw_path=str(path))
    n = db.insert_rows(conn, "referee_assignments", [
        {k: r[k] for k in ("nba_day", "game_id", "game_label", "home", "away", "official_name",
                           "official_number", "role", "captured_at", "raw_path")} for r in rows])
    return {**out, "inserted": n}


def _action_network_step(conn: sqlite3.Connection, day: date, captured_at: str) -> dict:
    now = _utcnow()
    payload = action_network.fetch_scoreboard(day)
    path = archive.save_raw("actionnetwork", "scoreboard", day, "scoreboard", payload,
                            snapshot=True, now=now)
    games = action_network.scoreboard_games(payload)
    inserted, started, unmatched = 0, [], []
    for g in games:
        row = conn.execute("SELECT game_id, tipoff_utc FROM games WHERE nba_day=? AND home=? "
                           "AND away=?", (day.isoformat(), g["home"], g["away"])).fetchone()
        if row is None:
            unmatched.append(f'{g["away"]}@{g["home"]}')
            continue
        if g["status"] != "scheduled" or (row["tipoff_utc"] and row["tipoff_utc"] <= _iso(now)):
            started.append(row["game_id"])   # in-play lines are not pre-game lines
            continue
        rows = [{**b, "game_id": row["game_id"], "captured_at": captured_at,
                 "source": action_network.SOURCE, "source_game_id": str(g["an_game_id"]),
                 "raw_path": str(path)}
                for b in g["books"]
                if any(b[k] is not None for k in ("home_spread", "total", "home_ml", "away_ml"))]
        inserted += db.insert_rows(conn, "line_snapshots", rows)
    if unmatched:
        log.warning("action network games with no matching game: %s", unmatched)
    return {"path": str(path), "games": len(games), "inserted": inserted,
            "started_skipped": started, "unmatched": unmatched}


def _dfs_step(conn: sqlite3.Connection, day: date, captured_at: str) -> dict:
    """DraftKings light capture (docs/PLAYERS.md §3.2): lobby, rules, draftables (salary, status,
    swappable, starter) of every upcoming Classic/Showdown group, one ``dfs_slates`` row per
    group, then the payout tables of the day's selected contests (once each, before lock; at
    most 60 added requests a day, see sources/dfs_capture.py). A payout failure is recorded in
    the summary under ``payouts`` and never fails the salary capture."""
    now = _utcnow()
    dfs_capture.ensure_log(conn)
    rid = dfs_capture.log_request(conn, "lobby", now=now)
    lobby = dfs.fetch_lobby()
    dfs_capture.mark_request(conn, rid, True)
    lobby_path = archive.save_raw("draftkings", "lobby", day, "lobby", lobby, snapshot=True,
                                  now=now)
    day_groups = dfs.lobby_draft_groups(lobby, day=day, game_types=dfs.SALARY_GAME_TYPES)
    tags = dfs.classify_slates(day_groups)
    groups = [g for g in day_groups
              if g["start_utc"] and g["start_utc"] > _iso(now)][:DFS_MAX_GROUPS]
    out: dict = {"lobby_path": str(lobby_path),
                 "all_nba_groups": len(dfs.lobby_draft_groups(lobby, day=day)),
                 "groups": len(groups), "requests": 1, "slates": {}, "failed": {},
                 "tags": {str(k): v for k, v in tags.items() if k in
                          {g["draft_group_id"] for g in day_groups}}}
    if not groups:
        return {**out, "status": "skipped", "reason": "no upcoming Classic/Showdown groups"}
    rules_by_type: dict[int, dict] = {}
    for g in groups:
        gt = g["game_type_id"]
        if gt in rules_by_type:
            continue
        key = f"gametype_{gt}"
        rpath = archive.raw_path("draftkings", "rules", day, key)
        try:
            if rpath.exists():
                rules_by_type[gt] = archive.load_raw(rpath)
            else:
                dfs_capture.log_request(conn, "rules", str(gt), now=now, ok=True)
                rules_by_type[gt] = dfs.fetch_rules(gt)
                out["requests"] += 1
                archive.save_raw("draftkings", "rules", day, key, rules_by_type[gt])
        except Exception as e:  # noqa: BLE001 - slot names are nice-to-have
            log.warning("dk rules %s failed: %s", gt, e)
    games_today = dfs_capture.games_by_teams(conn, day)
    total = 0
    for g in groups:
        sid = g["draft_group_id"]
        try:
            dfs_capture.log_request(conn, "draftables", str(sid), now=now, ok=True)
            payload = dfs.fetch_draftables(sid)
            out["requests"] += 1
            res = dfs_capture.store_draftables(
                conn, payload, slate_id=sid, game_type_id=g["game_type_id"], day=day,
                tag=tags.get(sid), rules=rules_by_type.get(g["game_type_id"]), lobby=lobby,
                start_utc=g["start_utc"], captured_at=captured_at, now=now, games=games_today)
            total += res["inserted"]
            out["slates"][str(sid)] = {"game_type": g["game_type"], "label": g["suffix"], **res}
        except Exception as e:  # noqa: BLE001 - one slate must not stop the others
            log.exception("dfs draft group %s failed", sid)
            out["failed"][str(sid)] = f"{type(e).__name__}: {e}"
    out["inserted"] = total
    if out["failed"] and not out["slates"]:
        raise RuntimeError(f"all {len(groups)} draft groups failed: {out['failed']}")
    try:
        pay = dfs_capture.capture_contests(conn, lobby, day, now=now)
        out["requests"] += pay["requests"]
        out["payouts"] = pay
    except Exception as e:  # noqa: BLE001 - payout tables are extra; salaries matter more
        log.exception("dfs payout capture failed")
        out["payouts"] = {"error": f"{type(e).__name__}: {e}"}
    out["dk_requests_today"] = dfs_capture.requests_today(conn, now)
    return out


PREGAME_STEPS = ("schedule", "nba_scoreboard", "espn_scoreboard", "nba_odds", "espn_injuries",
                 "official_injury_report", "props", "referees", "action_network",
                 "dfs_salaries", "predict", "brief")


def run_pregame(day: date, only: set[str] | list[str] | tuple[str, ...] | None = None) -> dict:
    """Pregame snapshot job. `only` restricts it to the named steps (see PREGAME_STEPS), e.g.
    ``{"props", "referees"}`` for a props-only poll; the monitoring hook still runs."""
    want = set(only) if only else None
    if want is not None and want - set(PREGAME_STEPS):
        raise ValueError(f"unknown pregame step(s): {sorted(want - set(PREGAME_STEPS))}")

    def on(name: str) -> bool:
        return want is None or name in want

    def body(conn: sqlite3.Connection, run: _Run, day: date) -> None:
        captured_at = _iso(datetime.now(timezone.utc))
        if on("schedule"):
            run.step("schedule", _schedule_step, conn)
        if on("nba_scoreboard"):
            if day == config.today_nba_day():
                run.step("nba_scoreboard", _nba_scoreboard_step, conn, day)
            else:
                run.summary["steps"]["nba_scoreboard"] = {"status": "skipped",
                                                          "reason": "day is not today"}
        if on("espn_scoreboard"):
            run.step("espn_scoreboard", _espn_scoreboard_step, conn, day, apply_results=False,
                     insert_odds=True, captured_at=captured_at)
        if on("nba_odds"):
            run.step("nba_odds", _nba_odds_step, conn, day, captured_at)
        if on("espn_injuries"):
            run.step("espn_injuries", _espn_injuries_step, conn, day, captured_at)
        if on("official_injury_report"):
            run.step("official_injury_report", _official_report_step, conn, day)
        # pre-tip archive sources (each isolated; a failure never touches the steps around it)
        if on("props"):
            run.step("props", _props_step, conn, run, day, captured_at)
        if on("referees"):
            run.step("referees", _referees_step, conn, day, captured_at)
        if on("action_network"):
            run.step("action_network", _action_network_step, conn, day, captured_at)
        if on("dfs_salaries"):
            run.step("dfs_salaries", _dfs_step, conn, day, captured_at)
        if on("predict"):
            run.step("predict", _predict_step, conn, day)
        if on("brief"):
            run.step("brief", _brief_step, conn, day)
        _monitor_hook(conn, run, "pregame", day)

    return _run_job("pregame", day, body)


# ------------------------------------------------------------------ postgame

def _box_scores(box: dict) -> tuple[int | None, int | None]:
    g = (box or {}).get("game") or {}

    def sc(t):
        try:
            return int((g.get(t) or {}).get("score"))
        except (TypeError, ValueError):
            return None
    return sc("homeTeam"), sc("awayTeam")


def _espn_summary_complete(payload: dict) -> bool:
    """False only when the payload positively says the game is not completed."""
    comps = ((payload or {}).get("header") or {}).get("competitions") or []
    if not comps:
        return True
    t = ((comps[0].get("status") or {}).get("type")) or {}
    return bool(t.get("completed", True))


def _archive_game(conn: sqlite3.Connection, run: _Run, g: sqlite3.Row, day: date,
                  now: datetime) -> str:
    """Fetch whatever is missing for one game. Returns 'archived' | 'pending'."""
    gid, espn_id = g["game_id"], g["espn_id"]
    need_box = not archive.exists_final("nba_cdn", "boxscore", day, gid)
    need_pbp = not archive.exists_final("nba_cdn", "playbyplay", day, gid)
    need_sum = bool(espn_id) and not archive.exists_final("espn", "summary", day, gid)
    if not (need_box or need_pbp or need_sum):
        return "already"
    tip = g["tipoff_utc"]
    if need_box and tip and tip > _iso(now):
        return "pending"  # has not started

    pending = False
    if need_box:
        try:
            box = nba_cdn.fetch_boxscore(gid)
        except FetchError as e:
            if e.status == 404:
                return "pending"
            run.error(f"{gid}:boxscore", str(e))
            return "pending"
        if not nba_cdn.is_final(box):
            return "pending"
        archive.save_raw("nba_cdn", "boxscore", day, gid, box)
        hs, as_ = _box_scores(box)
        _set_game_result(conn, gid, "final", hs, as_)
        log.info("%s boxscore archived", gid)
    if need_pbp:
        try:
            archive.save_raw("nba_cdn", "playbyplay", day, gid, nba_cdn.fetch_playbyplay(gid))
            log.info("%s playbyplay archived", gid)
        except Exception as e:  # noqa: BLE001
            run.error(f"{gid}:playbyplay", f"{type(e).__name__}: {e}")
            pending = True
    if need_sum:
        try:
            summ = espn.fetch_summary(espn_id)
            if _espn_summary_complete(summ):
                archive.save_raw("espn", "summary", day, gid, summ)
                log.info("%s espn summary (event %s) archived", gid, espn_id)
            else:
                pending = True
        except Exception as e:  # noqa: BLE001
            run.error(f"{gid}:espn_summary", f"{type(e).__name__}: {e}")
            pending = True
    return "pending" if pending else "archived"


def _games_step(conn: sqlite3.Connection, run: _Run, day: date) -> dict:
    now = datetime.now(timezone.utc)
    games = conn.execute(
        "SELECT game_id, espn_id, status, tipoff_utc FROM games WHERE nba_day=? "
        "ORDER BY tipoff_utc, game_id", (day.isoformat(),)).fetchall()
    from_team_games = False
    if not games:
        games = _team_games_fallback(conn, day)
        from_team_games = bool(games)
    archived, fetched, pending, postponed = [], [], [], []
    for g in games:
        if g["status"] == "postponed":
            postponed.append(g["game_id"])
            continue
        # re-read: the ESPN step may have changed espn_id/status; row is from after it already
        try:
            res = _archive_game(conn, run, g, day, now)
        except Exception as e:  # noqa: BLE001
            run.error(g["game_id"], f"{type(e).__name__}: {e}")
            res = "pending"
        if res == "pending":
            pending.append(g["game_id"])
        else:
            archived.append(g["game_id"])
            if res == "archived":
                fetched.append(g["game_id"])
    run.summary.update(archived=archived, fetched=fetched, pending=pending,
                       postponed=postponed)
    return {"games": len(games), "archived": len(archived), "fetched": len(fetched),
            "pending": len(pending), "postponed": len(postponed),
            "source": "team_games" if from_team_games else "games"}


def _team_games_fallback(conn: sqlite3.Connection, day: date) -> list[dict]:
    """Pseudo `games` rows for a day that has none, from `team_games` (finished games only:
    the game log only lists played games). espn_id/tipoff are unknown."""
    ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_id FROM team_games WHERE game_date=? ORDER BY game_id",
        (day.isoformat(),))]
    return [{"game_id": g, "espn_id": None, "status": "final", "tipoff_utc": None} for g in ids]


def pregame_spread(conn: sqlite3.Connection, game_id: str) -> float | None:
    """Home spread (negative = home favoured) used as the win-probability prior: the consensus
    closing line if there is one, else the latest odds snapshot captured before tip-off, else
    None (the model then assumes a pick'em)."""
    from . import odds_history
    line = odds_history.consensus_line(conn, game_id)
    if line and line.get("home_spread") is not None:
        return float(line["home_spread"])
    row = conn.execute(
        "SELECT o.home_spread FROM odds_snapshots o LEFT JOIN games g ON g.game_id = o.game_id "
        "WHERE o.game_id = ? AND o.home_spread IS NOT NULL "
        "AND (g.tipoff_utc IS NULL OR o.captured_at <= g.tipoff_utc) "
        "ORDER BY o.captured_at DESC LIMIT 1", (game_id,)).fetchone()
    return float(row[0]) if row else None


def _facts_step(conn: sqlite3.Connection, run: _Run, day: date, *, force: bool = False) -> dict:
    """game_facts for every archived final game of `day`. A failure on one game is logged and
    recorded (never fatal, and does not mark the job failed)."""
    from .analytics import summarize
    ids = list(run.summary.get("archived", []))
    fetched = set(run.summary.get("fetched", []))
    have = {r[0] for r in conn.execute("SELECT game_id FROM game_facts WHERE nba_day=?",
                                       (day.isoformat(),))}
    done, failed, skipped = [], {}, []
    for gid in ids:
        if gid in have and gid not in fetched and not force:
            skipped.append(gid)
            continue
        try:
            box = archive.load_raw(archive.raw_path("nba_cdn", "boxscore", day, gid))
            pbp = archive.load_raw(archive.raw_path("nba_cdn", "playbyplay", day, gid))
            spread = pregame_spread(conn, gid)
            facts = (summarize.game_facts(box, pbp) if spread is None
                     else summarize.game_facts(box, pbp, pregame_spread=spread))
            payload = json.dumps(facts, allow_nan=False, default=str)
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO game_facts (game_id, nba_day, facts_json, "
                    "computed_at) VALUES (?,?,?,?)",
                    (gid, day.isoformat(), payload, _iso(datetime.now(timezone.utc))))
            done.append(gid)
        except Exception as e:  # noqa: BLE001 - one game's facts must never stop the job
            log.exception("game_facts failed for %s", gid)
            failed[gid] = f"{type(e).__name__}: {e}"
    return {"computed": len(done), "already": len(skipped), "failed": failed}


def _season_types_for(day: date) -> tuple[str, ...]:
    """Only ask stats.nba.com for season types that can have games around `day`."""
    types = ["Regular Season"]
    if day.month in (4, 5, 6):
        types += ["PlayIn", "Playoffs"]
    return tuple(types)


def _refresh_gamelogs_step(conn: sqlite3.Connection, day: date) -> dict:
    """Re-fetch the current season's game logs (a handful of stats.nba.com calls) so
    team_games / player_games include last night. No-op for days outside the current season."""
    from . import backfill
    from .trends.common import season_of
    season = season_of(day)
    if season != season_of(config.today_nba_day()):
        return {"status": "skipped", "reason": "day is not in the current season",
                "season": season}
    res = backfill.backfill_gamelogs([season], season_types=_season_types_for(day), force=True,
                                     conn=conn)
    if res["failed"]:
        raise RuntimeError(f"{res['failed']} gamelog fetches failed: {res['errors'][:3]}")
    return {"season": season, "fetched": res["fetched"], "rows": res["rows"]}


def _brief_step(conn: sqlite3.Connection, brief_day: date) -> dict:
    from .brief import assemble
    b = assemble.build_brief(conn, brief_day)
    return {"day": brief_day.isoformat(), "path": b["meta"].get("path"),
            "last_night_games": len(b["last_night"]["games"]),
            "tonight_games": len(b["tonight"]["games"])}


def _predict_step(conn: sqlite3.Connection, day: date) -> dict:
    from . import predict
    rows = predict.predict_day(conn, day)
    return {"model": predict.PRODUCTION_MODEL, "games": len(rows),
            "with_market": sum(1 for r in rows if r["market_home_prob"] is not None)}


def run_postgame(day: date) -> dict:
    def body(conn: sqlite3.Connection, run: _Run, day: date) -> None:
        run.step("schedule", _schedule_step, conn)
        run.step("espn_scoreboard", _espn_scoreboard_step, conn, day, apply_results=True,
                 insert_odds=False, captured_at=_iso(datetime.now(timezone.utc)))
        run.summary.update(archived=[], fetched=[], pending=[], postponed=[])
        run.step("games", _games_step, conn, run, day)
        run.step("game_facts", _facts_step, conn, run, day)
        run.step("refresh_gamelogs", _refresh_gamelogs_step, conn, day)
        run.step("brief", _brief_step, conn, day + timedelta(days=1))
        _monitor_hook(conn, run, "postgame", day)

    return _run_job("postgame", day, body)
