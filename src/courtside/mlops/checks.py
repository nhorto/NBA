"""Data-quality and pipeline-health checks -> `data_checks`.

`run_checks(conn, job, day, run_summary)` runs after each collect job. Every check returns
status ok | warn | fail | skip with the measured `value`, the `threshold` it was compared to
and a plain-English `detail`. Games-dependent checks `skip` on days without games.

Check names (stable; alerts key off them):
  both jobs   `<job>.job_steps`, `stale.pregame`, `stale.postgame`
  pregame     `pregame.odds_coverage`, `pregame.injury_report`, `pregame.injury_match`,
              `pregame.prediction_coverage`, `pregame.brief_built`
  postgame    `postgame.finals_archived`, `postgame.game_facts`, `postgame.facts_validation`,
              `postgame.gamelogs_fresh`, `postgame.brief_next_day`
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, time, timedelta, timezone

from .. import archive, config

log = logging.getLogger("courtside.mlops.checks")

# Odds coverage of today's games
ODDS_WARN, ODDS_FAIL = 0.90, 0.50
# Official report is published ~11:00-13:00 ET on game days; look from 13:30 ET
INJURY_REPORT_AFTER = time(13, 30)
INJURY_MATCH_WARN = 0.85
INJURY_BATCH_MAX_AGE_H = 48
# Archived finals / facts: fraction of games complete
ARCHIVE_WARN, ARCHIVE_FAIL = 1.0, 0.80
FACTS_WARN, FACTS_FAIL = 1.0, 0.90
VALID_WARN, VALID_FAIL = 1.0, 0.75
# Hours since the last successful run: (warn, fail). Schedule: pregame 10:00-22:00 ET (max gap
# 12 h), postgame 03:00 and 09:00 ET (max gap 18 h).
STALE_HOURS = {"pregame": (16.0, 36.0), "postgame": (26.0, 50.0)}
SEASON_GAP_DAYS = 5     # no games within this many days of now = offseason: staleness is skipped


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _res(name: str, status: str, value=None, threshold=None, detail: str = "") -> dict:
    return {"check_name": name, "status": status,
            "value": None if value is None else round(float(value), 4),
            "threshold": None if threshold is None else float(threshold), "detail": detail}


def _ladder(name: str, frac: float, warn_below: float, fail_below: float, detail_ok: str,
            detail_bad: str) -> dict:
    """ok when frac >= warn_below, fail when frac < fail_below, else warn."""
    if frac >= warn_below:
        return _res(name, "ok", frac, warn_below, detail_ok)
    return _res(name, "fail" if frac < fail_below else "warn", frac, warn_below, detail_bad)


# ------------------------------------------------------------------ helpers

def _day_games(conn: sqlite3.Connection, day: date) -> list[dict]:
    """Non-postponed games of an NBA day; falls back to `team_games` ids (treated as final)
    for days with no `games` rows (the same rule collect.py uses)."""
    rows = [dict(r) for r in conn.execute(
        "SELECT game_id, espn_id, status, tipoff_utc FROM games WHERE nba_day=? "
        "AND status != 'postponed' ORDER BY tipoff_utc, game_id", (day.isoformat(),))]
    if rows:
        return rows
    if conn.execute("SELECT 1 FROM games WHERE nba_day=? LIMIT 1", (day.isoformat(),)).fetchone():
        return []   # only postponed games
    return [{"game_id": r[0], "espn_id": None, "status": "final", "tipoff_utc": None}
            for r in conn.execute("SELECT DISTINCT game_id FROM team_games WHERE game_date=? "
                                  "ORDER BY game_id", (day.isoformat(),))]


def _skip_all(names: list[str], why: str) -> list[dict]:
    return [_res(n, "skip", detail=why) for n in names]


def _last_run(conn: sqlite3.Connection, job: str, ok_only: bool = False):
    sql = "SELECT * FROM collect_runs WHERE job=? AND finished_at IS NOT NULL"
    if ok_only:
        sql += " AND ok=1"
    return conn.execute(sql + " ORDER BY finished_at DESC, id DESC LIMIT 1", (job,)).fetchone()


# ------------------------------------------------------------------ job health

def _job_steps(conn: sqlite3.Connection, job: str, summary: dict | None) -> dict:
    name = f"{job}.job_steps"
    if summary is None:
        row = _last_run(conn, job)
        if row is None:
            return _res(name, "skip", detail=f"no finished {job} run recorded yet")
        try:
            summary = json.loads(row["detail"]) if row["detail"] else {}
        except ValueError:
            summary = {}
        summary.setdefault("ok", bool(row["ok"]))
        when = f" (run finished {row['finished_at']})"
    else:
        when = ""
    steps = summary.get("steps") or {}
    failed = {s: v.get("error", "") for s, v in steps.items() if v.get("status") == "error"}
    errors = summary.get("errors") or []
    other = [e for e in errors if e.get("step") not in failed]
    if summary.get("ok", True) and not failed:
        return _res(name, "ok", 0, 0, f"all {len(steps)} {job} steps succeeded{when}")
    bits = [f"{s} ({str(m)[:80]})" for s, m in list(failed.items())[:4]]
    bits += [f"{e.get('step')} ({str(e.get('error'))[:60]})" for e in other[:3]]
    n_bad = len(failed) + len(other)
    return _res(name, "fail", n_bad, 0,
                f"{job} reported {n_bad} problem(s): {'; '.join(bits) or 'job not ok'}{when}")


def _in_season(conn: sqlite3.Connection, now: datetime) -> bool:
    today = now.astimezone(config.NBA_TZ).date()
    r = conn.execute("SELECT 1 FROM games WHERE nba_day BETWEEN ? AND ? LIMIT 1",
                     ((today - timedelta(days=SEASON_GAP_DAYS)).isoformat(),
                      (today + timedelta(days=SEASON_GAP_DAYS)).isoformat())).fetchone()
    return r is not None


def _stale(conn: sqlite3.Connection, job: str, now: datetime) -> dict:
    name = f"stale.{job}"
    warn_h, fail_h = STALE_HOURS[job]
    if not _in_season(conn, now):
        return _res(name, "skip", detail=f"no games scheduled within {SEASON_GAP_DAYS} days")
    row = _last_run(conn, job, ok_only=True)
    if row is None:
        if _last_run(conn, job) is None:
            return _res(name, "skip", detail=f"no {job} run recorded yet")
        return _res(name, "fail", None, warn_h, f"no {job} run has ever succeeded")
    age = (now - datetime.strptime(row["finished_at"], "%Y-%m-%dT%H:%M:%SZ")
           .replace(tzinfo=timezone.utc)).total_seconds() / 3600
    detail = f"last successful {job} finished {age:.1f} h ago (expected within {warn_h:.0f} h)"
    status = "fail" if age > fail_h else "warn" if age > warn_h else "ok"
    return _res(name, status, age, warn_h, detail)


# ------------------------------------------------------------------ pregame

def _production_model() -> str:
    from .. import predict
    return getattr(predict, "PRODUCTION_MODEL", "gbm_market")


def _pregame(conn: sqlite3.Connection, day: date, now: datetime) -> list[dict]:
    names = ["pregame.odds_coverage", "pregame.injury_report", "pregame.injury_match",
             "pregame.prediction_coverage", "pregame.brief_built"]
    games = _day_games(conn, day)
    out: list[dict] = []
    brief = conn.execute("SELECT built_at FROM briefs WHERE nba_day=?",
                         (day.isoformat(),)).fetchone()
    if not games:
        why = f"no games on {day}"
        out += _skip_all(names[:4], why)
        out.append(_res(names[4], "ok", 1, 1, f"brief for {day} built {brief['built_at']}")
                   if brief else _res(names[4], "skip", detail=why + ", no brief"))
        return out
    ids = [g["game_id"] for g in games]
    q = ",".join("?" * len(ids))

    # odds coverage
    have = {r[0] for r in conn.execute(
        f"SELECT DISTINCT game_id FROM odds_snapshots WHERE game_id IN ({q}) "
        f"AND (home_spread IS NOT NULL OR home_ml IS NOT NULL OR total IS NOT NULL)", ids)}
    frac = len(have) / len(ids)
    missing = [g for g in ids if g not in have]
    out.append(_ladder(names[0], frac, ODDS_WARN, ODDS_FAIL,
                       f"odds captured for all {len(ids)} games" if frac == 1 else
                       f"odds captured for {len(have)} of {len(ids)} games",
                       f"odds missing for {len(missing)} of {len(ids)} games today "
                       f"({', '.join(missing[:4])}{'...' if len(missing) > 4 else ''})"))

    # official injury report (published ~11:00-13:00 ET on game days)
    cutoff = datetime.combine(day, INJURY_REPORT_AFTER, tzinfo=config.NBA_TZ)
    if now < cutoff:
        out.append(_res(names[1], "skip", detail="too early: the official report is checked "
                        "from 13:30 ET on game days"))
    else:
        lo = datetime.combine(day, time(0, 0), tzinfo=config.NBA_TZ)
        n = conn.execute("SELECT COUNT(DISTINCT captured_at) FROM injury_snapshots "
                         "WHERE source='nba_official' AND captured_at>=? AND captured_at<?",
                         (_iso(lo), _iso(lo + timedelta(days=1)))).fetchone()[0]
        out.append(_res(names[1], "ok" if n else "warn", n, 1,
                        f"{n} official injury report(s) captured on {day}" if n else
                        f"no official NBA injury report captured on {day} (games today; "
                        f"expected by 13:30 ET)"))

    out.append(_injury_match(conn, now))

    # prediction coverage: every game has a champion prediction made at/before tip-off
    model = _production_model()
    pred = {r[0] for r in conn.execute(
        f"SELECT DISTINCT p.game_id FROM predictions p LEFT JOIN games g ON g.game_id=p.game_id "
        f"WHERE p.game_id IN ({q}) AND p.model=? AND (p.role IS NULL OR p.role='champion') "
        f"AND p.home_win_prob IS NOT NULL "
        f"AND (g.tipoff_utc IS NULL OR p.created_at <= g.tipoff_utc)", [*ids, model])}
    frac = len(pred) / len(ids)
    miss = [g for g in ids if g not in pred]
    out.append(_ladder(names[3], frac, 1.0, 0.5,
                       f"all {len(ids)} games have a {model} prediction made before tip-off",
                       f"{len(miss)} of {len(ids)} games have no {model} champion prediction "
                       f"before tip-off ({', '.join(miss[:4])}{'...' if len(miss) > 4 else ''})"))

    out.append(_res(names[4], "ok", 1, 1, f"brief for {day} built {brief['built_at']}")
               if brief else _res(names[4], "fail", 0, 1, f"no brief stored for {day}"))
    return out


def _injury_match(conn: sqlite3.Connection, now: datetime) -> dict:
    """Name-match rate of the newest official batch (else the newest ESPN batch) from the last
    48 h, via `injuries.match_report` (the same matcher the model uses)."""
    name = "pregame.injury_match"
    from .. import injuries
    oldest = _iso(now - timedelta(hours=INJURY_BATCH_MAX_AGE_H))
    newest = _iso(now)
    parts, primary = [], None
    for src in (injuries.OFFICIAL, injuries.ESPN):
        bs = [b for b in injuries.batches(conn, src, newest) if b >= oldest]
        if not bs:
            continue
        rep = injuries.match_report(conn, [(src, bs[-1])])["total"]
        if not rep["rows"]:
            continue
        parts.append(f"{src} {rep['matched']}/{rep['rows']} ({rep['rate']:.0%})")
        if primary is None:
            primary = rep["rate"]
    if primary is None:
        return _res(name, "skip", detail="no injury batch captured in the last 48 h")
    status = "ok" if primary >= INJURY_MATCH_WARN else "warn"
    return _res(name, status, primary, INJURY_MATCH_WARN,
                f"injury names matched to players: {'; '.join(parts)}"
                + ("" if status == "ok" else " - unmatched names are not priced into predictions"))


# ------------------------------------------------------------------ postgame

def _postgame(conn: sqlite3.Connection, day: date, now: datetime) -> list[dict]:
    names = ["postgame.finals_archived", "postgame.game_facts", "postgame.facts_validation",
             "postgame.gamelogs_fresh", "postgame.brief_next_day"]
    games = _day_games(conn, day)
    nxt = day + timedelta(days=1)
    brief = conn.execute("SELECT built_at FROM briefs WHERE nba_day=?",
                         (nxt.isoformat(),)).fetchone()
    brief_row = (_res(names[4], "ok", 1, 1, f"brief for {nxt} built {brief['built_at']}")
                 if brief else None)
    if not games:
        why = f"no games on {day}"
        return _skip_all(names[:4], why) + [brief_row or _res(names[4], "skip",
                                                              detail=why + ", no brief")]
    out: list[dict] = []
    # all expected games archived (boxscore + play-by-play, + ESPN summary when there is an id)
    incomplete = []
    for g in games:
        gid = g["game_id"]
        miss = [k for k, ok in (
            ("boxscore", archive.exists_final("nba_cdn", "boxscore", day, gid)),
            ("playbyplay", archive.exists_final("nba_cdn", "playbyplay", day, gid)),
            ("summary", not g["espn_id"] or archive.exists_final("espn", "summary", day, gid)))
            if not ok]
        if g["status"] != "final":
            miss.insert(0, f"not final ({g['status']})")
        if miss:
            incomplete.append(f"{gid} [{', '.join(miss)}]")
    frac = 1 - len(incomplete) / len(games)
    out.append(_ladder(names[0], frac, ARCHIVE_WARN, ARCHIVE_FAIL,
                       f"all {len(games)} games final and archived (boxscore, play-by-play, summary)",
                       f"{len(incomplete)} of {len(games)} games on {day} not fully archived: "
                       f"{'; '.join(incomplete[:3])}{'...' if len(incomplete) > 3 else ''}"))

    # game_facts for the finals
    finals = [g["game_id"] for g in games if g["status"] == "final"]
    facts = {}
    if finals:
        q = ",".join("?" * len(finals))
        for r in conn.execute(f"SELECT game_id, facts_json FROM game_facts WHERE game_id IN ({q})",
                              finals):
            facts[r["game_id"]] = r["facts_json"]
    if not finals:
        out.append(_res(names[1], "skip", detail=f"no final games on {day}"))
        out.append(_res(names[2], "skip", detail=f"no final games on {day}"))
    else:
        frac = len(facts) / len(finals)
        miss = [g for g in finals if g not in facts]
        out.append(_ladder(names[1], frac, FACTS_WARN, FACTS_FAIL,
                           f"game_facts present for all {len(finals)} finals",
                           f"game_facts missing for {len(miss)} of {len(finals)} finals "
                           f"({', '.join(miss[:3])}{'...' if len(miss) > 3 else ''})"))
        bad, checked = [], 0
        for gid, js in facts.items():
            try:
                v = (json.loads(js) or {}).get("validation")
            except ValueError:
                v = None
            if not isinstance(v, dict) or "ok" not in v:
                continue
            checked += 1
            if not v["ok"]:
                bad.append(f"{gid} ({v.get('minutes_mismatches', 0)} minutes, "
                           f"{v.get('plus_minus_mismatches', 0)} plus-minus mismatches)")
        if not checked:
            out.append(_res(names[2], "skip", detail="no game_facts carry a validation block"))
        else:
            frac = 1 - len(bad) / checked
            out.append(_ladder(names[2], frac, VALID_WARN, VALID_FAIL,
                               f"stint reconciliation passed for all {checked} games",
                               f"stint reconciliation failed for {len(bad)} of {checked} games: "
                               f"{'; '.join(bad[:3])}"))

    # game logs refreshed (regular-season days only)
    if not any(g["game_id"].startswith("002") for g in games):
        out.append(_res(names[3], "skip", detail=f"{day} had no regular-season games"))
    else:
        mx = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
        behind = (day - date.fromisoformat(mx)).days if mx else None
        ok = mx is not None and mx >= day.isoformat()
        out.append(_res(names[3], "ok" if ok else "fail", max(behind, 0) if behind is not None
                        else None, 0,
                        f"team_games runs through {mx}" if ok else
                        f"team_games ends {mx}, before {day}: the model and trends are missing "
                        f"last night's games"))
    out.append(brief_row or _res(names[4], "fail", 0, 1, f"no brief stored for {nxt}"))
    return out


# ------------------------------------------------------------------ entry point

def run_checks(conn: sqlite3.Connection, job: str, day, run_summary: dict | None = None,
               *, now: datetime | None = None) -> list[dict]:
    """Run every check for `job` ("pregame" | "postgame") about NBA day `day`, store them in
    `data_checks` (run_at = now) and return the rows."""
    if job not in ("pregame", "postgame"):
        raise ValueError(f"job must be pregame or postgame, not {job!r}")
    now = now or datetime.now(timezone.utc)
    day = _as_date(day)
    rows: list[dict] = [_job_steps(conn, job, run_summary),
                        _stale(conn, "pregame", now), _stale(conn, "postgame", now)]
    rows += (_pregame if job == "pregame" else _postgame)(conn, day, now)
    run_at = _iso(now)
    for r in rows:
        r["run_at"] = run_at
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO data_checks (run_at, check_name, status, value, threshold, "
            "detail) VALUES (:run_at, :check_name, :status, :value, :threshold, :detail)", rows)
    return rows


def latest(conn: sqlite3.Connection) -> list[dict]:
    """The most recent result of every check name."""
    return [dict(r) for r in conn.execute(
        "SELECT d.* FROM data_checks d JOIN (SELECT check_name, MAX(run_at) AS m FROM data_checks "
        "GROUP BY check_name) x ON x.check_name=d.check_name AND x.m=d.run_at "
        "ORDER BY d.check_name")]
