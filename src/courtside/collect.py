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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import archive, config, db
from .http import FetchError
from .sources import espn, nba_cdn

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


def run_pregame(day: date) -> dict:
    def body(conn: sqlite3.Connection, run: _Run, day: date) -> None:
        captured_at = _iso(datetime.now(timezone.utc))
        run.step("schedule", _schedule_step, conn)
        if day == config.today_nba_day():
            run.step("nba_scoreboard", _nba_scoreboard_step, conn, day)
        else:
            run.summary["steps"]["nba_scoreboard"] = {"status": "skipped",
                                                      "reason": "day is not today"}
        run.step("espn_scoreboard", _espn_scoreboard_step, conn, day, apply_results=False,
                 insert_odds=True, captured_at=captured_at)
        run.step("nba_odds", _nba_odds_step, conn, day, captured_at)
        run.step("espn_injuries", _espn_injuries_step, conn, day, captured_at)
        run.step("official_injury_report", _official_report_step, conn, day)
        run.step("predict", _predict_step, conn, day)
        run.step("brief", _brief_step, conn, day)

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

    return _run_job("postgame", day, body)
