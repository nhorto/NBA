"""Pre-tip archive sources wired into `run_pregame`: props, referees, action_network,
dfs_salaries. Everything offline (fixtures via tests/pretip_helpers.py)."""

from __future__ import annotations

import copy
import json
import sqlite3
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pretip_helpers import J, NOW, referee_page_for, stub_pretip_sources

from courtside import collect, config, db
from courtside.http import FetchError
from courtside.mlops import checks
from courtside.sources import action_network, dfs, espn, espn_props, nba_cdn, referees

TODAY = date(2026, 10, 20)
STEPS = {"schedule", "espn_scoreboard", "props", "referees", "action_network", "dfs_salaries"}
NEW_TABLES = ("prop_snapshots", "referee_assignments", "line_snapshots", "dfs_salaries")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(config, "today_nba_day", lambda: TODAY)
    monkeypatch.setattr(nba_cdn, "fetch_schedule", lambda: copy.deepcopy(J("nba_cdn/schedule.json")))
    monkeypatch.setattr(espn, "fetch_scoreboard",
                        lambda day: copy.deepcopy(J("espn/scoreboard_20261020.json")))
    return stub_pretip_sources(monkeypatch)


def q(sql, *a):
    conn = db.connect()
    try:
        return [tuple(r) for r in conn.execute(sql, a).fetchall()]
    finally:
        conn.close()


def count(table):
    return q(f"SELECT COUNT(*) FROM {table}")[0][0]


def run(only=STEPS):
    return collect.run_pregame(TODAY, only=only)


# ------------------------------------------------------------------ migration

def test_migration_creates_tables_and_is_idempotent():
    conn = db.connect()
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == \
        len(db.MIGRATIONS)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(NEW_TABLES) <= tables
    idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_props_game_captured", "idx_refs_day_game", "idx_lines_game_captured",
            "idx_dfs_slate_captured"} <= idx
    cols = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")} for t in NEW_TABLES}
    assert {"game_id", "captured_at", "provider", "espn_athlete_id", "player_name", "market",
            "line", "over_odds", "under_odds", "raw_path"} <= cols["prop_snapshots"]
    assert {"nba_day", "game_id", "official_name", "role", "captured_at"} <= \
        cols["referee_assignments"]
    assert {"game_id", "captured_at", "source", "book", "home_spread", "total", "home_ml",
            "away_ml"} <= cols["line_snapshots"]
    assert {"captured_at", "site", "slate_id", "game_type", "player_name", "site_player_id",
            "team", "positions", "salary"} <= cols["dfs_salaries"]
    n = db.migrate(conn)
    assert n == len(db.MIGRATIONS)
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(db.MIGRATIONS)


def test_insert_rows_rejects_unknown_columns():
    conn = db.connect()
    with pytest.raises(TypeError):
        db.insert_rows(conn, "line_snapshots", [{"game_id": "x", "bogus": 1}])
    assert db.insert_rows(conn, "line_snapshots", []) == 0


# ------------------------------------------------------------------ happy path

def test_extras_happy_path(env):
    s = run()
    assert s["ok"], s["errors"]
    steps = s["steps"]
    assert all(steps[n]["status"] == "ok" for n in STEPS), steps

    # props: 3 games, DraftKings, names resolved from the rosters / athlete fallback
    p = steps["props"]
    assert p["games"] == 3 and p["inserted"] == p["rows"] == count("prop_snapshots") > 0
    assert env["props.providers"] == 3 and env["props.page"] == 6      # 2 pages per game
    assert p["requests"] == sum(p["requests_by_kind"].values())
    assert q("SELECT DISTINCT provider, provider_name FROM prop_snapshots") == [
        ("100", "DraftKings")]
    assert q("SELECT COUNT(*) FROM prop_snapshots WHERE subject='player' AND player_name IS NULL"
             ) == [(0,)]
    assert q("SELECT DISTINCT player_name FROM prop_snapshots WHERE espn_athlete_id='4065648'"
             ) == [("Jayson Tatum",)]
    assert q("SELECT COUNT(*) FROM prop_snapshots WHERE raw_path IS NULL") == [(0,)]
    raw = config.data_dir() / "raw" / "espn" / "props" / str(TODAY)
    assert len(list(raw.glob("*_100_p0[12]__*.json.gz"))) == 6
    assert len(list(raw.glob("*_providers__*.json.gz"))) == 3

    # referees: html archived, crews stored per official/role and matched to our game ids
    r = steps["referees"]
    assert r["games"] == 3 and r["inserted"] == count("referee_assignments") == 3 * 4 + 2
    assert q("SELECT DISTINCT game_id FROM referee_assignments WHERE game_id IS NOT NULL "
             "ORDER BY 1") == [("0022600001",), ("0022600002",), ("0022600003",)]
    assert q("SELECT official_name, official_number FROM referee_assignments "
             "WHERE game_id='0022600001' AND role='crew_chief'") == [("Zach Zarba", 15)]
    assert q("SELECT COUNT(*) FROM referee_assignments WHERE role='replay_center' "
             "AND game_id IS NULL") == [(2,)]
    html = list((config.data_dir() / "raw" / "nba_official" / "referee_assignments"
                 / str(TODAY)).glob("referee-assignments__*.html"))
    assert len(html) == 1 and b"<table" in html[0].read_bytes()

    # action network: 4 books x 3 scheduled games; the in-play copy is not stored
    a = steps["action_network"]
    assert a["games"] == 4 and a["inserted"] == count("line_snapshots") == 12
    assert len(a["started_skipped"]) == 1
    assert q("SELECT home_spread, total, home_ml, away_ml FROM line_snapshots "
             "WHERE game_id='0022600002' AND book='draftkings'") == [(-4.5, 231.5, -185, 154)]
    assert q("SELECT COUNT(*) FROM line_snapshots WHERE book='open' AND is_open=1") == [(3,)]

    # dfs: classic + showdown slates, salaries with game ids and slot names
    d = steps["dfs_salaries"]
    assert d["groups"] == 2 and set(d["slates"]) == {"154575", "154825"}
    assert d["inserted"] == count("dfs_salaries") > 0
    assert env["dfs.lobby"] == 1 and env["dfs.draftables"] == 2 and env["dfs.rules"] == 2
    assert q("SELECT DISTINCT game_type FROM dfs_salaries ORDER BY 1") == [("classic",),
                                                                           ("showdown",)]
    assert q("SELECT salary, roster_slot, game_id, team FROM dfs_salaries "
             "WHERE player_name='Victor Wembanyama' ORDER BY roster_slot") == [
        (10500, "C", "0022600003", "SAS"), (10500, "UTIL", "0022600003", "SAS")]
    lobby = list((config.data_dir() / "raw" / "draftkings" / "lobby" / str(TODAY)).glob("*"))
    assert len(lobby) == 1
    assert q("SELECT COUNT(*) FROM dfs_salaries WHERE roster_slot IS NULL") == [(0,)]

    # one shared captured_at per run across the new tables
    stamps = {r[0] for r in q("SELECT captured_at FROM prop_snapshots UNION "
                              "SELECT captured_at FROM referee_assignments UNION "
                              "SELECT captured_at FROM line_snapshots UNION "
                              "SELECT captured_at FROM dfs_salaries")}
    assert len(stamps) == 1


def test_second_run_writes_only_changes(env):
    run()
    before = {t: count(t) for t in NEW_TABLES}
    s = run()
    assert s["ok"], s["errors"]
    assert s["steps"]["props"]["inserted"] == 0                     # change log: nothing moved
    assert s["steps"]["referees"]["inserted"] == 0 and s["steps"]["referees"]["unchanged"]
    assert s["steps"]["dfs_salaries"]["inserted"] == 0
    assert all(v["unchanged"] for v in s["steps"]["dfs_salaries"]["slates"].values())
    assert count("line_snapshots") == before["line_snapshots"] + 12  # lines: full snapshots
    for t in ("prop_snapshots", "referee_assignments", "dfs_salaries"):
        assert count(t) == before[t]
    # raw archive still keeps every poll
    raw = config.data_dir() / "raw" / "espn" / "props" / str(TODAY)
    assert len(list(raw.glob("*_100_p01__*.json.gz"))) >= 3


def test_props_change_log_records_only_moved_markets(env, monkeypatch):
    run()
    moved = J("espn_props/propbets_page1.json")
    item = next(i for i in moved["items"]
                if i["type"]["name"] == "Total Points"
                and i["athlete"]["$ref"].split("/")[-1].startswith("4251"))
    item["odds"]["american"]["value"] = "-135"            # over price of Paul George 15.5
    monkeypatch.setattr(espn_props, "fetch_propbets_page",
                        lambda eid, pid, page=1, limit=1000: moved if page == 1
                        else J("espn_props/propbets_page2.json"))
    monkeypatch.setattr(collect, "_utcnow", lambda: datetime(2026, 10, 20, 15, 0,
                                                             tzinfo=timezone.utc))
    s = run({"props"})
    assert s["steps"]["props"]["inserted"] == 3      # the same fixture is served for each game
    rows = q("SELECT over_odds, under_odds FROM prop_snapshots WHERE espn_athlete_id='4251' "
             "AND market='Total Points' AND game_id='0022600001' ORDER BY id")
    assert rows == [(-120, -106), (-135, -106)]


def test_props_only_for_games_not_yet_tipped(env, monkeypatch):
    # 19:30Z: BOS@DET (19:00Z) has tipped; the other two have not
    monkeypatch.setattr(collect, "_utcnow", lambda: datetime(2026, 10, 20, 19, 30,
                                                             tzinfo=timezone.utc))
    s = run({"schedule", "espn_scoreboard", "props"})
    assert s["steps"]["props"]["games"] == 2 and env["props.providers"] == 2
    assert {r[0] for r in q("SELECT DISTINCT game_id FROM prop_snapshots")} == {
        "0022600002", "0022600003"}


def test_props_skipped_when_nothing_upcoming(env, monkeypatch):
    monkeypatch.setattr(collect, "_utcnow", lambda: datetime(2026, 10, 21, 3, 0,
                                                             tzinfo=timezone.utc))
    s = run({"schedule", "espn_scoreboard", "props", "dfs_salaries"})
    assert s["steps"]["props"]["status"] == "skipped" and env["props.providers"] == 0
    assert s["steps"]["dfs_salaries"]["status"] == "skipped"      # all slates locked
    assert s["ok"]


def test_name_cache_avoids_repeat_lookups(env, monkeypatch):
    run({"schedule", "espn_scoreboard", "props"})
    first = env["props.roster"]
    assert first > 0
    moved = J("espn_props/propbets_page1.json")
    moved["items"][0]["odds"]["american"]["value"] = "+500"      # force a new row
    monkeypatch.setattr(espn_props, "fetch_propbets_page",
                        lambda eid, pid, page=1, limit=1000: moved if page == 1
                        else J("espn_props/propbets_page2.json"))
    run({"props"})
    assert env["props.roster"] == first                          # every id known from the DB


# ------------------------------------------------------------------ referees edge cases

def test_referees_not_before_9_et_or_for_other_days(env, monkeypatch):
    monkeypatch.setattr(collect, "_utcnow", lambda: datetime(2026, 10, 20, 12, 59,
                                                             tzinfo=timezone.utc))   # 8:59 ET
    s = run({"schedule", "referees"})
    assert s["steps"]["referees"]["status"] == "skipped" and env["referees.page"] == 0
    monkeypatch.setattr(collect, "_utcnow", lambda: NOW)
    s = collect.run_pregame(date(2026, 10, 21), only={"referees"})
    assert s["steps"]["referees"]["reason"] == "day is not today" and env["referees.page"] == 0


def test_referees_empty_table_stores_nothing_but_archives(env, monkeypatch):
    html = (Path(__file__).parent / "fixtures/referees/empty_20261019.html").read_text()
    monkeypatch.setattr(referees, "fetch_page", lambda *a, **k: html.encode())
    s = run({"schedule", "referees"})
    r = s["steps"]["referees"]
    assert s["ok"] and r["empty"] and r["inserted"] == 0 and count("referee_assignments") == 0
    assert list((config.data_dir() / "raw" / "nba_official").rglob("*.html"))


def test_referees_stale_page_is_not_stored(env, monkeypatch):
    monkeypatch.setattr(referees, "fetch_page",
                        lambda *a, **k: referee_page_for("October 19, 2026").encode())
    s = run({"schedule", "espn_scoreboard", "referees"})
    r = s["steps"]["referees"]
    assert r["status"] == "skipped" and "2026-10-19" in r["reason"]
    assert count("referee_assignments") == 0


def test_referees_changed_crew_is_a_new_snapshot(env, monkeypatch):
    run({"schedule", "espn_scoreboard", "referees"})
    n = count("referee_assignments")
    changed = referee_page_for().replace("Zach Zarba", "Scott Foster")
    monkeypatch.setattr(referees, "fetch_page", lambda *a, **k: changed.encode())
    s = run({"referees"})
    assert s["steps"]["referees"]["inserted"] == n
    assert q("SELECT DISTINCT official_name FROM referee_assignments "
             "WHERE game_id='0022600001' AND role='crew_chief' ORDER BY 1") == [
        ("Scott Foster",), ("Zach Zarba",)]


# ------------------------------------------------------------------ step isolation

BROKEN = {
    "props": (espn_props, "fetch_providers"),
    "referees": (referees, "fetch_page"),
    "action_network": (action_network, "fetch_scoreboard"),
    "dfs_salaries": (dfs, "fetch_lobby"),
}


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_one_failing_source_does_not_break_the_others(env, monkeypatch, broken):
    mod, attr = BROKEN[broken]

    def boom(*a, **k):
        raise FetchError("503 boom", status=503)

    monkeypatch.setattr(mod, attr, boom)
    s = run()
    assert s["ok"] is False
    for name in BROKEN:
        if name == broken:
            continue
        assert s["steps"][name]["status"] == "ok", (broken, name, s["steps"][name])
    if broken == "props":    # per-game isolation: recorded as errors, step itself completes
        assert len(s["steps"]["props"]["failed"]) == 3
        assert {e["step"] for e in s["errors"]} == {f"props:00226000{i:02d}" for i in (1, 2, 3)}
    else:
        assert s["steps"][broken]["status"] == "error" and "boom" in s["steps"][broken]["error"]
        assert s["errors"][0]["step"] == broken
    # the surrounding pipeline still ran and the run was recorded
    assert s["steps"]["schedule"]["status"] == "ok"
    assert q("SELECT job, ok FROM collect_runs") == [("pregame", 0)]
    healthy = [t for t, b in zip(NEW_TABLES, ("props", "referees", "action_network",
                                              "dfs_salaries")) if b != broken]
    assert all(count(t) > 0 for t in healthy)


def test_props_one_bad_game_does_not_stop_the_rest(env, monkeypatch):
    real = espn_props.fetch_providers

    def flaky(eid):
        if eid == "401909089":
            raise FetchError("500", status=500)
        return real(eid)

    monkeypatch.setattr(espn_props, "fetch_providers", flaky)
    s = run()
    p = s["steps"]["props"]
    assert list(p["failed"]) == ["0022600002"] and set(p["per_game"]) == {"0022600001",
                                                                          "0022600003"}
    assert {r[0] for r in q("SELECT DISTINCT game_id FROM prop_snapshots")} == {
        "0022600001", "0022600003"}


def test_props_names_fall_back_to_athlete_documents(env, monkeypatch):
    def broken_roster(abbr):
        raise FetchError("roster down", status=500)

    monkeypatch.setattr(espn_props, "fetch_roster", broken_roster)
    s = run({"schedule", "espn_scoreboard", "props"})
    assert s["ok"], s["errors"]
    assert env["props.athlete"] > 0
    assert q("SELECT player_name FROM prop_snapshots WHERE espn_athlete_id='4065648' LIMIT 1"
             ) == [("Athlete 4065648",)]


def test_unknown_step_name_rejected():
    with pytest.raises(ValueError):
        collect.run_pregame(TODAY, only={"props", "bogus"})


def test_cli_steps_option():
    from courtside import cli
    args = cli.build_parser().parse_args(
        ["collect", "pregame", "--date", "2026-10-20", "--steps", "props,referees"])
    assert args.steps == "props,referees"


# ------------------------------------------------------------------ data checks

def _checks(now):
    conn = db.connect()
    try:
        rows = checks._pretip_extras(conn, TODAY, now,
                                     [r[0] for r in conn.execute(
                                         "SELECT game_id FROM games WHERE nba_day=?",
                                         (TODAY.isoformat(),))])
        return {r["check_name"]: r for r in rows}
    finally:
        conn.close()


def test_checks_ok_after_collection(env):
    run()
    got = _checks(NOW)
    assert {k: v["status"] for k, v in got.items()} == {
        "pregame.props_coverage": "ok", "pregame.referee_assignments": "ok",
        "pregame.dfs_salaries": "ok"}
    assert got["pregame.props_coverage"]["value"] == 1.0


def test_checks_warn_when_nothing_captured(env):
    run({"schedule", "espn_scoreboard"})
    got = _checks(NOW)
    assert {k: v["status"] for k, v in got.items()} == {
        "pregame.props_coverage": "warn", "pregame.referee_assignments": "warn",
        "pregame.dfs_salaries": "warn"}
    assert "posted ~9:00 ET" in got["pregame.referee_assignments"]["detail"]
    early = _checks(datetime(2026, 10, 20, 13, 59, tzinfo=timezone.utc))      # 9:59 ET
    assert early["pregame.referee_assignments"]["status"] == "skip"


def test_run_checks_includes_new_checks_and_skips_on_empty_day():
    conn = db.connect()
    rows = {r["check_name"]: r for r in checks.run_checks(
        conn, "pregame", date(2026, 3, 1), {"ok": True, "steps": {}}, now=NOW)}
    for n in checks.PRETIP_CHECKS:
        assert rows[n]["status"] == "skip"
