import copy
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from courtside import archive, backfill_players as bp, cli, db
from courtside.http import BlockedError, FetchError
from courtside.sources import nba_players as N

FIX = Path(__file__).parent / "fixtures" / "nba_players"
KIND = {"matchups": "boxscorematchupsv3", "tracking": "boxscoreplayertrackv3",
        "hustle": "boxscorehustlev2"}
GAME_FIX = {"boxscorematchupsv3": "game_matchups", "boxscoreplayertrackv3": "game_tracking",
            "boxscorehustlev2": "game_hustle"}


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def seed_games(conn, games):
    """games: [(game_id, game_date, season, season_type)] -> team_games rows (one per game)."""
    db.upsert_rows(conn, "team_games", [
        {"game_id": g, "team": "AAA", "season": s, "season_type": t, "game_date": d}
        for g, d, s, t in games])


def make_fetch(calls, empty=()):
    def fetch(kind, gid):
        calls.append((kind, gid))
        p = copy.deepcopy(load("game_matchups_empty" if gid in empty else GAME_FIX[kind]))
        p[N.GAME_KINDS[kind]]["gameId"] = gid
        return p
    return fetch


class NoSleep:
    def __init__(self):
        self.slept = []
        self.resets = 0

    def guard(self, waits=(5, 10, 20)):
        return bp.Guard(sleep=self.slept.append, reset=self._reset, waits=waits, log=lambda m: None)

    def _reset(self):
        self.resets += 1


@pytest.fixture
def conn():
    c = db.connect()
    yield c
    c.close()


OLD = date(2026, 10, 19)


def test_migration_creates_tables(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"player_index", "team_rosters", "team_coaches", "matchup_rollup", "season_matchups",
            "pt_defend", "pt_shot_defdist", "synergy_playtypes", "hustle_season",
            "player_matchups", "player_tracking", "player_hustle"} <= names
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == len(db.MIGRATIONS)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_matchups)")}
    assert {"off_player_id", "def_player_id", "matchup_min", "partial_poss", "switches",
            "help_blk", "sfl", "ftm", "fta"} <= cols
    db.migrate(conn)   # idempotent


def test_per_game_archives_resumes_and_respects_floors(conn):
    seed_games(conn, [("0021800001", "2018-11-01", "2018-19", "Regular Season"),
                      ("0021900001", "2019-11-01", "2019-20", "Regular Season"),
                      ("0042400001", "2025-05-01", "2024-25", "Playoffs"),
                      ("0012500001", "2025-10-01", "2025-26", "Preseason")])   # never fetched
    calls = []
    s1 = bp.run_per_game("all", conn=conn, fetch=make_fetch(calls), today=OLD,
                         guard=NoSleep().guard())
    m, t, h = (s1["by_kind"][KIND[k]] for k in ("matchups", "tracking", "hustle"))
    assert (m["games"], t["games"], h["games"]) == (3, 2, 2)    # tracking/hustle start 2019-20
    assert m["fetched"] == 3 and t["fetched"] == 2 and not s1["aborted"]
    assert [k for k, _ in calls] == [KIND["matchups"]] * 3 + [KIND["tracking"]] * 2 + \
        [KIND["hustle"]] * 2                                   # matchups, then tracking, then hustle
    assert archive.raw_path("nba_stats", "boxscorematchupsv3", "2018-11-01", "0021800001").exists()
    assert conn.execute("SELECT COUNT(*) FROM player_matchups").fetchone()[0] == 3 * 12
    assert conn.execute("SELECT COUNT(*) FROM player_tracking").fetchone()[0] == 2 * 4
    assert conn.execute("SELECT COUNT(*) FROM player_hustle").fetchone()[0] == 2 * 4
    assert conn.execute("SELECT COUNT(*) FROM collect_runs WHERE job LIKE 'backfill_players%' "
                        "AND ok=1").fetchone()[0] == 1

    # rerun: everything archived and in the DB -> no network, nothing re-parsed
    calls.clear()
    s2 = bp.run_per_game("all", conn=conn, fetch=make_fetch(calls), today=OLD,
                         guard=NoSleep().guard())
    assert calls == [] and s2["by_kind"][KIND["matchups"]]["cached"] == 3

    # DB wiped but archive intact: rebuilt from the archive, still no network
    conn.execute("DELETE FROM player_matchups")
    conn.commit()
    s3 = bp.run_per_game("matchups", conn=conn, fetch=make_fetch(calls), today=OLD,
                         guard=NoSleep().guard())
    assert calls == [] and s3["by_kind"][KIND["matchups"]]["reparsed"] == 3
    assert conn.execute("SELECT COUNT(*) FROM player_matchups").fetchone()[0] == 36


def test_from_to_and_limit(conn):
    seed_games(conn, [(f"002190000{i}", f"2019-11-0{i}", "2019-20", "Regular Season")
                      for i in range(1, 5)] +
               [("0022400001", "2024-11-01", "2024-25", "Regular Season")])
    calls = []
    bp.run_per_game("hustle", start="2024-25", conn=conn, fetch=make_fetch(calls), today=OLD,
                    guard=NoSleep().guard())
    assert calls == [(KIND["hustle"], "0022400001")]
    calls.clear()
    s = bp.run_per_game("hustle", end="2019-20", limit=2, conn=conn, fetch=make_fetch(calls),
                        today=OLD, guard=NoSleep().guard())
    assert len(calls) == 2 and s["by_kind"][KIND["hustle"]]["fetched"] == 2


def test_empty_answer_archived_unless_game_is_recent(conn):
    old, fresh = "0021900001", "0021900002"
    today = date(2026, 10, 19)
    seed_games(conn, [(old, "2019-11-01", "2019-20", "Regular Season"),
                      (fresh, (today - timedelta(days=1)).isoformat(), "2019-20", "Regular Season")])
    calls = []
    s = bp.run_per_game("matchups", conn=conn, fetch=make_fetch(calls, empty={old, fresh}),
                        today=today, guard=NoSleep().guard())
    assert s["by_kind"][KIND["matchups"]]["empty"] == 2
    assert archive.raw_path("nba_stats", "boxscorematchupsv3", "2019-11-01", old).exists()
    fresh_day = (today - timedelta(days=1)).isoformat()
    assert not archive.raw_path("nba_stats", "boxscorematchupsv3", fresh_day, fresh).exists()
    calls.clear()                                   # the old empty is final, the fresh one is retried
    bp.run_per_game("matchups", conn=conn, fetch=make_fetch(calls), today=today,
                    guard=NoSleep().guard())
    assert calls == [(KIND["matchups"], fresh)]


def test_guard_backs_off_on_blocked_responses_then_continues():
    ns = NoSleep()
    g = ns.guard(waits=(120, 300, 600))
    attempts = iter([BlockedError("HTML/blocked response"), FetchError("403", status=403),
                     "payload"])

    def fn(x):
        r = next(attempts)
        if isinstance(r, Exception):
            raise r
        return r

    assert g.call(fn, 1) == "payload"
    assert ns.slept == [120, 300] and ns.resets == 2      # escalating waits, fresh session each time
    assert g.level == 0 and g.blocks == 2                 # a success resets the escalation


def test_guard_aborts_when_blocks_never_clear():
    ns = NoSleep()
    g = ns.guard(waits=(1, 2))

    def fn():
        raise BlockedError("HTML/blocked response")

    with pytest.raises(bp.Aborted):
        g.call(fn)
    assert ns.slept == [1, 2]


def test_guard_timeouts_count_as_blocks_but_404_does_not():
    ns = NoSleep()
    g = ns.guard(waits=(7,))
    seq = iter([FetchError("timeout"), "ok"])

    def fn():
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r

    assert g.call(fn) == "ok" and ns.slept == [7]
    with pytest.raises(FetchError):
        g.call(lambda: (_ for _ in ()).throw(FetchError("gone", status=404)))
    assert ns.slept == [7]


def test_guard_backs_off_after_consecutive_server_errors():
    ns = NoSleep()
    g = ns.guard(waits=(30, 60))
    for _ in range(bp.MAX_CONSECUTIVE_FAILS):
        with pytest.raises(FetchError):
            g.call(lambda: (_ for _ in ()).throw(FetchError("boom", status=500)))
    assert ns.slept == [30]


def test_per_game_run_survives_a_block_and_resumes_after_abort(conn):
    seed_games(conn, [(f"002190000{i}", f"2019-11-0{i}", "2019-20", "Regular Season")
                      for i in range(1, 4)])
    calls, state = [], {"n": 0}
    good = make_fetch(calls)

    def flaky(kind, gid):
        state["n"] += 1
        if state["n"] == 2:                        # second request is blocked once
            raise BlockedError("HTML/blocked response")
        return good(kind, gid)

    ns = NoSleep()
    s = bp.run_per_game("tracking", conn=conn, fetch=flaky, today=OLD, guard=ns.guard((9,)))
    c = s["by_kind"][KIND["tracking"]]
    assert c["fetched"] == 3 and c["failed"] == 0 and ns.slept == [9] and not s["aborted"]

    # permanently blocked from game 2 on: aborts cleanly, run recorded as not ok, then resumes
    conn.execute("DELETE FROM player_tracking")
    for p in (archive.raw_path("nba_stats", "boxscoreplayertrackv3", f"2019-11-0{i}",
                               f"002190000{i}") for i in (2, 3)):
        p.unlink()
    conn.commit()

    def blocked(kind, gid):
        raise BlockedError("HTML/blocked response")

    s2 = bp.run_per_game("tracking", conn=conn, fetch=blocked, today=OLD, guard=NoSleep().guard((1,)))
    assert s2["aborted"] and s2["by_kind"][KIND["tracking"]]["failed"] == 0
    last = conn.execute("SELECT ok FROM collect_runs WHERE job='backfill_players_games' "
                        "ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert last == 0
    calls.clear()
    s3 = bp.run_per_game("tracking", conn=conn, fetch=make_fetch(calls), today=OLD,
                         guard=NoSleep().guard())
    assert len(calls) == 2 and s3["by_kind"][KIND["tracking"]]["fetched"] == 2
    assert conn.execute("SELECT COUNT(*) FROM player_tracking").fetchone()[0] == 12


def test_progress_logged_to_file(conn):
    seed_games(conn, [(f"00219000{i:02d}", "2019-11-01", "2019-20", "Regular Season")
                      for i in range(1, 6)])
    bp.run_per_game("hustle", conn=conn, fetch=make_fetch([]), today=OLD, guard=NoSleep().guard(),
                    log_every=2)
    text = bp.log_path().read_text()
    assert bp.log_path().name == "backfill_players.log"
    assert "fetched=2/5" in text and "fetched=4/5" in text and "games/min" in text


# --------------------------------------------------------------------- season tables

@pytest.fixture
def fake_season_fetch(monkeypatch):
    calls = []

    def rec(name, fixture):
        def f(*a, **kw):
            calls.append((name, a))
            return load(fixture(*a) if callable(fixture) else fixture)
        monkeypatch.setattr(N, name, f)
    rec("fetch_player_index", "playerindex")
    rec("fetch_roster", "roster")
    rec("fetch_matchups_rollup", "matchupsrollup")
    rec("fetch_league_season_matchups", "leagueseasonmatchups")
    rec("fetch_pt_defend", "ptdefend")
    rec("fetch_pt_shot", "ptshot")
    rec("fetch_synergy", lambda s, t, e, g, p: "synergy_player" if e == "P" else "synergy_team")
    rec("fetch_hustle_season", "hustle_season")
    return calls


def test_season_jobs_enumeration():
    jobs = list(bp.season_jobs("2024-25", "2025-26", ("Regular Season", "Playoffs")))
    by = {}
    for j in jobs:
        by[j.table] = by.get(j.table, 0) + 1
    assert by == {"index": 1, "roster": 60, "rollup": 4, "lsm": 4, "ptdefend": 16, "ptshot": 16,
                  "synergy": 160, "hustle": 4}
    keys = [(j.kind, j.day, j.key) for j in jobs]
    assert len(keys) == len(set(keys))                       # archive paths are unique
    # per-table default starts when --from is not given
    d = list(bp.season_jobs(None, "2025-26", ("Regular Season",), tables=("roster", "rollup")))
    assert sum(j.table == "roster" for j in d) == 30 * 11    # 2015-16 .. 2025-26
    assert sum(j.table == "rollup" for j in d) == 9          # 2017-18 .. 2025-26


def test_season_tables_load_archive_and_rerun_without_network(conn, fake_season_fetch):
    kw = dict(start="2025-26", end="2025-26", season_types=("Regular Season",), conn=conn,
              guard=NoSleep().guard())
    s1 = bp.run_season_tables(**kw)
    assert s1["failed"] == 0 and not s1["aborted"] and s1["cached"] == 0
    n = len(fake_season_fetch)
    assert n == s1["fetched"] == 1 + 30 + 1 + 1 + 4 + 4 + 40 + 1
    rows = s1["rows"]
    assert rows["player_index"] == 4 and rows["team_coaches"] == 30 * 4
    assert rows["matchup_rollup"] == 4 and rows["season_matchups"] == 4
    assert rows["pt_defend"] == 4 * 3 and rows["pt_shot_defdist"] == 4 * 3
    assert conn.execute("SELECT COUNT(*) FROM team_rosters").fetchone()[0] == 4   # same ids x30 teams
    assert conn.execute("SELECT COUNT(*) FROM synergy_playtypes").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM hustle_season").fetchone()[0] == 3
    assert archive.raw_path("nba_stats", "playerindex", "2025-10-01", "all_2025-26").exists()
    assert archive.raw_path("nba_stats", "commonteamroster", "2025-10-01",
                            "2025-26_1610612744").exists()

    s2 = bp.run_season_tables(**kw)                  # archived: nothing fetched, tables rebuilt
    assert len(fake_season_fetch) == n and s2["fetched"] == 0 and s2["cached"] == n
    conn.execute("DELETE FROM pt_defend")
    conn.commit()
    bp.run_season_tables(tables=("ptdefend",), **kw)
    assert conn.execute("SELECT COUNT(*) FROM pt_defend").fetchone()[0] == 12 and len(
        fake_season_fetch) == n

    s3 = bp.run_season_tables(force=True, tables=("rollup",), **kw)
    assert s3["fetched"] == 1 and len(fake_season_fetch) == n + 1


def test_season_table_failure_is_counted_not_fatal(conn, monkeypatch):
    def boom(*a, **k):
        raise FetchError("HTTP 500", status=500)
    monkeypatch.setattr(N, "fetch_matchups_rollup", boom)
    monkeypatch.setattr(N, "fetch_hustle_season", lambda s, t: load("hustle_season"))
    s = bp.run_season_tables(start="2025-26", end="2025-26", season_types=("Regular Season",),
                             tables=("rollup", "hustle"), conn=conn, guard=NoSleep().guard())
    assert s["failed"] == 1 and s["fetched"] == 1 and s["rows"] == {"hustle_season": 3}
    assert conn.execute("SELECT ok FROM collect_runs WHERE job='backfill_players_season'"
                        ).fetchone()[0] == 0


# ------------------------------------------------------------------------------- CLI

def test_cli_players(monkeypatch):
    season, game = [], []

    def fake_season(*a, **k):
        season.append((a, k))
        return {"fetched": 0, "cached": 0, "failed": 0, "aborted": False, "rows": {}, "errors": []}

    def fake_game(*a, **k):
        game.append((a, k))
        return {"by_kind": {}, "aborted": False, "errors": []}

    monkeypatch.setattr(bp, "run_season_tables", fake_season)
    monkeypatch.setattr(bp, "run_per_game", fake_game)
    assert cli.main(["backfill", "players"]) == 2                       # needs a mode
    assert cli.main(["backfill", "players", "--season-tables", "--from", "bad"]) == 2
    assert cli.main(["backfill", "players", "--season-tables", "--tables", "nope"]) == 2
    assert not season and not game
    assert cli.main(["backfill", "players", "--season-tables", "--per-game", "all",
                     "--from", "2017-18"]) == 0
    assert season[0][0] == ("2017-18", "2025-26") and game[0][0] == ("all", "2017-18", "2025-26")
    assert cli.main(["backfill", "players", "--per-game", "tracking", "--limit", "5"]) == 0
    assert len(season) == 1 and game[1][0] == ("tracking", None, "2025-26")
    assert game[1][1]["limit"] == 5
