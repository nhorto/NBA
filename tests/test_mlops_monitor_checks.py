"""mlops.checks: statuses on game days, off days, and failures."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from mlops_helpers import conn_for, game, odds, pred

from courtside import archive, db
from courtside.mlops import checks

DAY = date(2026, 1, 20)
AFTER_REPORT = datetime(2026, 1, 20, 19, 0, tzinfo=timezone.utc)      # 14:00 ET
BEFORE_REPORT = datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc)     # 10:00 ET


def by_name(rows):
    return {r["check_name"]: r for r in rows}


def finished_run(conn, job, ok, finished, detail=None):
    conn.execute("INSERT INTO collect_runs (job, started_at, finished_at, ok, detail) "
                 "VALUES (?,?,?,?,?)", (job, finished, finished, int(ok), detail))
    conn.commit()


def test_off_day_is_skip_not_fail(tmp_path):
    conn = conn_for(tmp_path)
    for job, fn in (("pregame", "pregame"), ("postgame", "postgame")):
        rows = by_name(checks.run_checks(conn, job, DAY, None, now=AFTER_REPORT))
        for name, r in rows.items():
            if name.startswith(f"{job}.") and not name.endswith("job_steps"):
                assert r["status"] == "skip", (name, r)
        assert not any(r["status"] == "fail" for r in rows.values())


def test_pregame_full_coverage_is_ok(tmp_path):
    conn = conn_for(tmp_path)
    for i in range(3):
        game(conn, f"g{i}", DAY.isoformat(), None, None, status="scheduled",
             tip="2026-01-21T00:00:00Z")
        odds(conn, f"g{i}", -150, 130, spread=-3)
        pred(conn, f"g{i}", DAY.isoformat(), 0.6, created="2026-01-20T15:00:00Z")
    conn.execute("INSERT INTO briefs (nba_day, brief_json, built_at) VALUES (?,?,?)",
                 (DAY.isoformat(), "{}", "2026-01-20T15:01:00Z"))
    db.insert_injury_snapshots(conn, [{"captured_at": "2026-01-20T17:00:00Z",
                                       "source": "nba_official", "team": "BOS",
                                       "player_name": "nobody here", "status": "Out"}])
    rows = by_name(checks.run_checks(conn, "pregame", DAY, {"ok": True, "steps": {"a": {}}},
                                     now=AFTER_REPORT))
    assert rows["pregame.odds_coverage"]["status"] == "ok"
    assert rows["pregame.odds_coverage"]["value"] == 1.0
    assert rows["pregame.prediction_coverage"]["status"] == "ok"
    assert rows["pregame.injury_report"]["status"] == "ok"
    assert rows["pregame.brief_built"]["status"] == "ok"
    assert rows["pregame.job_steps"]["status"] == "ok"
    stored = conn.execute("SELECT COUNT(*) FROM data_checks").fetchone()[0]
    assert stored == len(rows)


def test_pregame_failures_and_warnings(tmp_path):
    conn = conn_for(tmp_path)
    for i in range(4):
        game(conn, f"g{i}", DAY.isoformat(), None, None, status="scheduled",
             tip="2026-01-21T00:00:00Z")
    odds(conn, "g0", -150, 130)                                  # 1 of 4 = 25% -> fail
    pred(conn, "g0", DAY.isoformat(), 0.6, created="2026-01-20T15:00:00Z")
    pred(conn, "g1", DAY.isoformat(), 0.6, created="2026-01-21T01:00:00Z")   # after tip: not counted
    summary = {"ok": False, "steps": {"nba_odds": {"status": "error", "error": "HTTP 500"},
                                      "schedule": {"status": "ok"}},
               "errors": [{"step": "nba_odds", "error": "HTTP 500"}]}
    rows = by_name(checks.run_checks(conn, "pregame", DAY, summary, now=AFTER_REPORT))
    assert rows["pregame.odds_coverage"]["status"] == "fail"
    assert rows["pregame.injury_report"]["status"] == "warn"
    assert rows["pregame.prediction_coverage"]["status"] == "fail"       # 1 of 4 < 50%
    assert rows["pregame.brief_built"]["status"] == "fail"
    assert rows["pregame.job_steps"]["status"] == "fail"
    assert "nba_odds" in rows["pregame.job_steps"]["detail"]
    # before 13:30 ET the official report is not required yet
    early = by_name(checks.run_checks(conn, "pregame", DAY, None, now=BEFORE_REPORT))
    assert early["pregame.injury_report"]["status"] == "skip"


def test_odds_partial_coverage_warns(tmp_path):
    conn = conn_for(tmp_path)
    for i in range(10):
        game(conn, f"g{i}", DAY.isoformat(), None, None, status="scheduled")
    for i in range(8):                                           # 80%: warn (>= 50%, < 90%)
        odds(conn, f"g{i}", -150, 130)
    rows = by_name(checks.run_checks(conn, "pregame", DAY, None, now=AFTER_REPORT))
    assert rows["pregame.odds_coverage"]["status"] == "warn"


def test_job_steps_from_latest_collect_run(tmp_path):
    conn = conn_for(tmp_path)
    assert by_name(checks.run_checks(conn, "pregame", DAY, None,
                                     now=AFTER_REPORT))["pregame.job_steps"]["status"] == "skip"
    finished_run(conn, "pregame", False, "2026-01-20T12:00:00Z",
                 json.dumps({"ok": False, "steps": {"x": {"status": "error", "error": "boom"}}}))
    rows = by_name(checks.run_checks(conn, "pregame", DAY, None, now=AFTER_REPORT))
    assert rows["pregame.job_steps"]["status"] == "fail"
    assert "boom" in rows["pregame.job_steps"]["detail"]


def test_stale_job_detection(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g0", DAY.isoformat(), None, None, status="scheduled")
    finished_run(conn, "pregame", True, "2026-01-20T18:00:00Z")        # 1 h ago: ok
    finished_run(conn, "postgame", True, "2026-01-18T09:00:00Z")       # 58 h ago: fail
    rows = by_name(checks.run_checks(conn, "pregame", DAY, None, now=AFTER_REPORT))
    assert rows["stale.pregame"]["status"] == "ok"
    assert rows["stale.postgame"]["status"] == "fail"
    # a postgame run that is merely late (30 h) warns
    finished_run(conn, "postgame", True, "2026-01-19T13:00:00Z")
    rows = by_name(checks.run_checks(conn, "postgame", DAY, None, now=AFTER_REPORT))
    assert rows["stale.postgame"]["status"] == "warn"


def test_stale_skipped_in_offseason(tmp_path):
    conn = conn_for(tmp_path)
    finished_run(conn, "pregame", True, "2025-12-01T12:00:00Z")
    rows = by_name(checks.run_checks(conn, "pregame", DAY, None, now=AFTER_REPORT))
    assert rows["stale.pregame"]["status"] == "skip"


def _final(conn, gid, day, facts=None, archived=("boxscore", "playbyplay"), espn_id=None):
    game(conn, gid, day.isoformat(), 100, 90)
    if espn_id:
        conn.execute("UPDATE games SET espn_id=? WHERE game_id=?", (espn_id, gid))
        conn.commit()
    for kind in archived:
        archive.save_raw("nba_cdn", kind, day, gid, {"x": 1})
    if facts is not None:
        conn.execute("INSERT INTO game_facts (game_id, nba_day, facts_json, computed_at) "
                     "VALUES (?,?,?,?)", (gid, day.isoformat(), json.dumps(facts), "t"))
        conn.commit()


def test_postgame_all_ok(tmp_path):
    conn = conn_for(tmp_path)
    _final(conn, "0022500001", DAY, {"validation": {"ok": True}})
    _final(conn, "0022500002", DAY, {"validation": {"ok": True}})
    conn.execute("INSERT INTO team_games (game_id, team, season, game_date, is_home, min, pts) "
                 "VALUES ('0022500002','BOS','2025-26','2026-01-20',1,240,100)")
    conn.execute("INSERT INTO briefs (nba_day, brief_json, built_at) VALUES ('2026-01-21','{}','t')")
    conn.commit()
    rows = by_name(checks.run_checks(conn, "postgame", DAY, {"ok": True, "steps": {}},
                                     now=AFTER_REPORT))
    for n in ("finals_archived", "game_facts", "facts_validation", "gamelogs_fresh",
              "brief_next_day"):
        assert rows[f"postgame.{n}"]["status"] == "ok", rows[f"postgame.{n}"]


def test_postgame_failures(tmp_path):
    conn = conn_for(tmp_path)
    _final(conn, "0022500001", DAY, {"validation": {"ok": False, "minutes_mismatches": 2}})
    _final(conn, "0022500002", DAY, None, archived=("boxscore",))     # no pbp, no facts
    rows = by_name(checks.run_checks(conn, "postgame", DAY, None, now=AFTER_REPORT))
    assert rows["postgame.finals_archived"]["status"] == "fail"       # 1 of 2 = 50% < 80%
    assert "playbyplay" in rows["postgame.finals_archived"]["detail"]
    assert rows["postgame.game_facts"]["status"] == "fail"            # 1 of 2 < 90%
    assert rows["postgame.facts_validation"]["status"] == "fail"      # 0 of 1 ok
    assert rows["postgame.gamelogs_fresh"]["status"] == "fail"        # no team_games at all
    assert rows["postgame.brief_next_day"]["status"] == "fail"


def test_summary_archive_required_only_with_espn_id(tmp_path):
    conn = conn_for(tmp_path)
    _final(conn, "0022500001", DAY, {"validation": {"ok": True}}, espn_id="401")
    rows = by_name(checks.run_checks(conn, "postgame", DAY, None, now=AFTER_REPORT))
    assert rows["postgame.finals_archived"]["status"] == "fail"
    assert "summary" in rows["postgame.finals_archived"]["detail"]


def test_gamelogs_skipped_for_non_regular_season(tmp_path):
    conn = conn_for(tmp_path)
    _final(conn, "0042500001", DAY, {"validation": {"ok": True}})
    rows = by_name(checks.run_checks(conn, "postgame", DAY, None, now=AFTER_REPORT))
    assert rows["postgame.gamelogs_fresh"]["status"] == "skip"


def test_bad_job_name(tmp_path):
    with pytest.raises(ValueError):
        checks.run_checks(conn_for(tmp_path), "nightly", DAY)
