"""courtside backfill stats / shots: one season's stats-site data, network mocked."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

from courtside import archive, backfill_stats, cli, db
from courtside.sources import nba_players

FIX = Path(__file__).parent / "fixtures"
GID, DAY, SEASON = "0022500001", "2025-10-21", "2025-26"


GAME_FIX = {"boxscorematchupsv3": ("game_matchups", "boxScoreMatchups"),
            "boxscoreplayertrackv3": ("game_tracking", "boxScorePlayerTrack"),
            "boxscorehustlev2": ("game_hustle", "boxScoreHustle")}


def J(rel):
    return json.loads((FIX / rel).read_text())


def _seed():
    archive.save_raw("nba_cdn", "playbyplay", DAY, GID,
                     archive.load_raw(FIX / "analytics/playbyplay_0022500001.json.gz"))
    archive.save_raw("nba_cdn", "boxscore", DAY, GID,
                     archive.load_raw(FIX / "analytics/boxscore_0022500001.json.gz"))
    conn = db.connect()
    base = dict(game_id=GID, season=SEASON, season_type="Regular Season", game_date=DAY)
    db.upsert_rows(conn, "team_games", [{**base, "team": "OKC", "opp": "HOU", "is_home": 1},
                                        {**base, "team": "HOU", "opp": "OKC", "is_home": 0}])
    return conn


def _mock_stats(monkeypatch):
    calls = Counter()

    def game(kind, gid):
        calls["game"] += 1
        calls[kind] += 1
        fix, root = GAME_FIX[kind]
        p = copy.deepcopy(J(f"nba_players/{fix}.json"))
        p[root]["gameId"] = gid
        return p

    def pt_day(day, season, stype, measure):
        calls["pt_day"] += 1
        return J(f"nba_players/pt_day_{measure.lower()}.json")

    def syn(s, t, e, g, p):
        calls["synergy"] += 1
        return J("nba_players/synergy_player.json" if e == "P" else "nba_players/synergy_team.json")

    def lsm(s, t):
        calls["lsm"] += 1
        return J("nba_players/leagueseasonmatchups.json")

    monkeypatch.setattr(nba_players, "fetch_game", game)
    monkeypatch.setattr(nba_players, "fetch_pt_day", pt_day)
    monkeypatch.setattr(nba_players, "fetch_synergy", syn)
    monkeypatch.setattr(nba_players, "fetch_league_season_matchups", lsm)
    return calls


def test_season_types_for():
    assert backfill_stats.season_types_for("2026-27", date(2026, 10, 25)) == ("Regular Season",)
    assert backfill_stats.season_types_for("2025-26", date(2026, 10, 25)) == (
        "Regular Season", "Playoffs")


def test_offline_fills_shots_and_stints_idempotently():
    conn = _seed()
    s = backfill_stats.run(SEASON, offline=True, conn=conn)
    assert s["shots"]["stored_games"] == 1 and s["stints"]["stored"] == 1
    assert "season_tables" not in s and backfill_stats.ok(s), s
    assert s["matchups"]["boxscorematchupsv3"]["fetched"] == 0          # offline: archive only
    assert s["tracking_ext"]["requests"] == 0 and s["tracking_ext"]["pending"] == [DAY]
    assert s["shot_xfg"]["fga"] == conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    n = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    s = backfill_stats.run(SEASON, offline=True, conn=conn)
    assert s["shots"]["skipped"] == 1 and s["stints"]["already"] == 1
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == n


def test_network_parts_resume_from_the_archive(monkeypatch):
    conn = _seed()
    calls = _mock_stats(monkeypatch)
    s = backfill_stats.run(SEASON, conn=conn)
    assert backfill_stats.ok(s), s
    want = {"game": 3, "boxscorematchupsv3": 1, "boxscoreplayertrackv3": 1,
            "boxscorehustlev2": 1, "pt_day": 3, "synergy": 80, "lsm": 2}   # RS + Playoffs
    assert calls == want
    for t in ("player_matchups", "player_tracking", "player_hustle", "player_tracking_ext"):
        assert conn.execute(f"SELECT COUNT(DISTINCT game_id) FROM {t}").fetchone()[0] == 1, t
    assert conn.execute("SELECT COUNT(*) FROM synergy_playtypes").fetchone()[0] > 0
    # a past season archives the day's dashboards; the BOS row has no game that day
    assert all(archive.exists_final("nba_stats", "leaguedashptstats", DAY,
                                    f"{m}_regular_season")
               for m in ("passing", "drives", "possessions"))
    assert s["tracking_ext"]["done"] == 1 and s["tracking_ext"]["rows"] == 12
    assert tuple(conn.execute("SELECT COUNT(*), COUNT(pot_ast), COUNT(drives), "
                              "COUNT(paint_touches) FROM player_tracking_ext").fetchone()) \
        == (4, 4, 4, 4)
    s = backfill_stats.run(SEASON, conn=conn)                  # everything archived: no calls
    assert calls == want
    assert s["season_tables"]["cached"] == 82 and s["tracking_ext"]["cached"] == 3


def test_current_season_keeps_retrying_unprocessed_games(monkeypatch):
    """2026-27: an empty tracking payload / dashboard is never archived (nba.com lags days)."""
    gid, day = "0022600008", "2026-10-23"
    conn = db.connect()
    base = dict(game_id=gid, season="2026-27", season_type="Regular Season", game_date=day)
    db.upsert_rows(conn, "team_games", [{**base, "team": "OKC", "opp": "HOU", "is_home": 1},
                                        {**base, "team": "HOU", "opp": "OKC", "is_home": 0}])
    calls = _mock_stats(monkeypatch)
    empty = {"resultSets": [{"name": "x", "headers": ["PLAYER_ID", "GP"], "rowSet": []}]}

    def shell(kind, g):
        calls[kind] += 1
        root = GAME_FIX[kind][1]
        return {root: {"gameId": g, "homeTeam": {"players": []}, "awayTeam": {"players": []}}}

    monkeypatch.setattr(nba_players, "fetch_game", shell)
    monkeypatch.setattr(nba_players, "fetch_pt_day", lambda *a: calls.update(["pt_day"]) or empty)
    monkeypatch.setattr(backfill_stats.config, "today_nba_day", lambda: date(2026, 10, 25))
    import courtside.backfill_players as bp
    real = bp.run_per_game
    monkeypatch.setattr(bp, "run_per_game",
                        lambda *a, **k: real(*a, **{**k, "today": date(2026, 10, 25)}))
    s = backfill_stats.run("2026-27", conn=conn)
    assert s["tracking"]["boxscoreplayertrackv3"]["empty"] == 1
    assert s["tracking_ext"]["pending"] == [day]
    assert not archive.exists_final("nba_stats", "boxscoreplayertrackv3", day, gid)
    assert not archive.exists_final("nba_stats", "leaguedashptstats", day, "passing_regular_season")
    backfill_stats.run("2026-27", conn=conn)                   # retried, not cached
    assert calls["boxscoreplayertrackv3"] == 2 and calls["pt_day"] == 6


def test_cli_backfill_shots(capsys):
    _seed()
    assert cli.main(["backfill", "shots", "--season", SEASON]) == 0
    assert "1 games stored" in capsys.readouterr().out
    assert cli.main(["backfill", "stats", "--season", SEASON, "--offline"]) == 0
    assert cli.main(["backfill", "stats", "--season", "2025-27"]) == 2


def test_offline_backfills_starters_idempotently_from_the_boxscore():
    conn = _seed()
    # a stale sdv row for the same game is replaced, not duplicated
    conn.execute("INSERT INTO player_starts VALUES (?,?,?,?)", (GID, 1, "OKC", "sdv"))
    conn.commit()
    s = backfill_stats.run(SEASON, offline=True, conn=conn)
    assert s["starts"]["stored_games"] == 1 and s["starts"]["failed"] == {}
    rows = conn.execute("SELECT team, source, COUNT(*) FROM player_starts WHERE game_id=? "
                        "GROUP BY team, source ORDER BY team", (GID,)).fetchall()
    assert [tuple(r) for r in rows] == [("HOU", "cdn", 5), ("OKC", "cdn", 5)]
    box = archive.load_raw(FIX / "analytics/boxscore_0022500001.json.gz")["game"]
    want = {p["personId"] for t in ("homeTeam", "awayTeam") for p in box[t]["players"]
            if p["starter"] == "1"}
    assert {r[0] for r in conn.execute("SELECT player_id FROM player_starts")} == want
    s = backfill_stats.run(SEASON, offline=True, conn=conn)
    assert s["starts"]["stored_games"] == 0 and s["starts"]["already"] == 1
    assert conn.execute("SELECT COUNT(*) FROM player_starts").fetchone()[0] == 10


def test_starters_skip_non_final_or_incomplete_boxscores():
    from courtside.analytics import starts_store
    box = archive.load_raw(FIX / "analytics/boxscore_0022500001.json.gz")
    live = copy.deepcopy(box)
    live["game"]["gameStatus"] = 2
    assert starts_store.parse_starters(live) == []
    short = copy.deepcopy(box)
    short["game"]["homeTeam"]["players"][0]["starter"] = "0"
    assert starts_store.parse_starters(short) == []
    assert len(starts_store.parse_starters(box)) == 10
