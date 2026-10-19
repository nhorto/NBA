"""Stints store (T0.4): stored rows round-trip, reconcile with the boxscore, build is idempotent."""

from __future__ import annotations

import copy
import json
import sqlite3
from functools import lru_cache

import pytest

from courtside import archive, db
from courtside.analytics import stints_store as ss
from courtside.analytics.pbp import AWAY, HOME, Event, game_seconds, normalize
from courtside.analytics.stints import Stint, box_players, build_rotation
from test_analytics_stints import GAMES, load

DAY = "2025-10-22"


@pytest.fixture
def conn():
    c = db.connect()
    yield c
    c.close()


def archive_games(ids=tuple(GAMES), *, with_box=True):
    for gid in ids:
        box, pbp = load(gid)
        archive.save_raw("nba_cdn", "playbyplay", DAY, gid, pbp)
        if with_box:
            archive.save_raw("nba_cdn", "boxscore", DAY, gid, box)


@lru_cache(maxsize=None)
def _truth(gid):
    box, pbp = load(gid)
    events = normalize(pbp, box)
    return box, events, build_rotation(events, box)


# ------------------------------------------------------------------------- build + round trip

def test_build_stores_every_fixture_game(conn):
    archive_games()
    s = ss.build(conn)
    assert s["archived"] == s["attempted"] == s["validated"] == len(GAMES)
    assert s["strict"] == len(GAMES)          # the fixtures reconcile at 0.0 s / exact +/-
    assert s["failures"] == []
    assert ss.stored_games(conn) == set(GAMES)
    assert s["check"]["check_name"] == "stints_store" and s["check"]["status"] == "ok"
    row = conn.execute("SELECT * FROM data_checks WHERE check_name='stints_store'").fetchone()
    assert row["status"] == "ok" and row["value"] == 1.0


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_stored_stints_round_trip(conn, gid):
    archive_games([gid])
    ss.build(conn)
    _, _, rot = _truth(gid)
    got = ss.load_stints(conn, gid)
    assert len(got) == len(rot.stints)
    for n, (g, s) in enumerate(zip(got, rot.stints)):
        assert g["stint_no"] == n
        assert (g["period"], g["start_sec"], g["end_sec"]) == (s.period, s.start, s.end)
        assert g["home5"] == list(s.home) and g["away5"] == list(s.away)
        assert (g["home_pts"], g["away_pts"]) == (s.home_pts, s.away_pts)
        assert (g["home_poss"], g["away_poss"]) == (s.home_poss, s.away_poss)


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_five_per_side_sorted_and_tiling(conn, gid):
    archive_games([gid])
    ss.build(conn)
    box, events, _ = _truth(gid)
    got = ss.load_stints(conn, gid)
    for g in got:
        assert len(g["home5"]) == 5 and len(g["away5"]) == 5
        assert g["home5"] == sorted(g["home5"]) and g["away5"] == sorted(g["away5"])
        assert not set(g["home5"]) & set(g["away5"])
    assert [g["stint_no"] for g in got] == list(range(len(got)))
    assert got[0]["start_sec"] == 0.0 and got[-1]["end_sec"] == game_seconds(events)
    assert all(a["end_sec"] == b["start_sec"] for a, b in zip(got, got[1:]))
    assert sum(g["home_pts"] for g in got) == box["game"]["homeTeam"]["score"]
    assert sum(g["away_pts"] for g in got) == box["game"]["awayTeam"]["score"]
    # the stored text is plain JSON, as documented
    raw = conn.execute("SELECT home5 FROM stints WHERE game_id=? AND stint_no=0", (gid,)).fetchone()[0]
    assert json.loads(raw) == got[0]["home5"] and " " not in raw


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_on_court_seconds_equal_boxscore_minutes(conn, gid):
    archive_games([gid])
    ss.build(conn)
    box, events, _ = _truth(gid)
    players = box_players(box)
    rows = {r["player_id"]: r for r in conn.execute(
        "SELECT * FROM player_game_oncourt WHERE game_id=?", (gid,))}
    assert set(rows) == {pid for pid, p in players.items() if p["played"]}
    for pid, r in rows.items():
        total = r["sec_q13"] + r["sec_q4"] + r["sec_ot"]
        assert abs(total - players[pid]["seconds"]) <= 1.0, (pid, total, players[pid]["seconds"])
        assert r["team"] == players[pid]["team"]
        assert bool(r["started"]) == players[pid]["starter"]
        assert 0 <= r["sec_garbage"] <= r["sec_q4"] + 1e-6
    ot_games = {"0022500001", "0022500095"}
    assert (sum(r["sec_ot"] for r in rows.values()) > 0) == (gid in ot_games)
    assert sum(r["started"] for r in rows.values()) == 10


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_on_court_tallies_are_consistent_with_the_box(conn, gid):
    """Every team event has five own-team and five opposing players on the floor."""
    archive_games([gid])
    ss.build(conn)
    box, _, rot = _truth(gid)
    g = box["game"]
    rows = [dict(r) for r in conn.execute("SELECT * FROM player_game_oncourt WHERE game_id=?", (gid,))]
    for key, other in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        tri, st, ost = g[key]["teamTricode"], g[key]["statistics"], g[other]["statistics"]
        mine = [r for r in rows if r["team"] == tri]
        fga, fgm = st["fieldGoalsAttempted"], st["fieldGoalsMade"]
        assert sum(r["team_fga_on"] for r in mine) == 5 * fga
        assert sum(r["team_fgm_on"] for r in mine) == 5 * fgm
        assert sum(r["team_misses_on"] for r in mine) == 5 * (fga - fgm)
        assert sum(r["tm_fgm_on"] for r in mine) == 4 * fgm
        assert sum(r["opp_misses_on"] for r in mine) == 5 * (ost["fieldGoalsAttempted"] - ost["fieldGoalsMade"])
        assert sum(r["opp_tov_on"] for r in mine) == 5 * ost["turnoversTotal"]
        assert sum(r["opp_2pa_on"] for r in mine) == 5 * (
            ost["fieldGoalsAttempted"] - ost["threePointersAttempted"])
        side = HOME if key == "homeTeam" else AWAY
        assert sum(r["on_poss"] for r in mine) == 5 * sum(s.poss(side) for s in rot.stints)
        opp = AWAY if side == HOME else HOME
        assert sum(r["opp_on_poss"] for r in mine) == 5 * sum(s.poss(opp) for s in rot.stints)
        for r in mine:
            assert 0 <= r["tm_fgm_on"] <= r["team_fgm_on"] <= r["team_fga_on"] <= fga


# ------------------------------------------------------------------------- idempotency etc.

def test_second_build_is_a_noop_and_force_recomputes(conn):
    archive_games()
    first = ss.build(conn)
    snap = lambda: (  # noqa: E731
        [tuple(r) for r in conn.execute("SELECT * FROM stints ORDER BY game_id, stint_no")],
        [tuple(r) for r in conn.execute("SELECT * FROM player_game_oncourt ORDER BY game_id, player_id")])
    before = snap()
    n_checks = conn.execute("SELECT COUNT(*) FROM data_checks").fetchone()[0]
    second = ss.build(conn)
    assert second["attempted"] == 0 and second["skipped"] == len(GAMES) == first["archived"]
    assert second["check"] is None
    assert snap() == before
    assert conn.execute("SELECT COUNT(*) FROM data_checks").fetchone()[0] == n_checks
    forced = ss.build(conn, force=True)
    assert forced["attempted"] == forced["validated"] == len(GAMES)
    assert snap() == before                      # replaced, not duplicated


def test_filters_by_season_and_game_ids(conn):
    archive_games()
    assert ss.build(conn, "2024-25")["archived"] == 0
    assert ss.stored_games(conn) == set()
    s = ss.build(conn, ["2025-26"], game_ids=["0022500083", "0022500095", "0029999999"])
    assert s["attempted"] == 2 and ss.stored_games(conn) == {"0022500083", "0022500095"}
    s = ss.build(conn, "2025-26", limit=1)
    assert s["attempted"] == 1 and len(ss.stored_games(conn)) == 3


def test_game_without_boxscore_is_not_attempted(conn):
    archive_games(["0022500083"])
    archive_games(["0022500095"], with_box=False)
    s = ss.build(conn)
    assert s["archived"] == s["attempted"] == 1 and ss.stored_games(conn) == {"0022500083"}
    archive_games(["0022500095"])                      # the boxscore lands later: picked up
    s = ss.build(conn)
    assert s["skipped"] == 1 and s["attempted"] == 1 and len(ss.stored_games(conn)) == 2


def test_failing_game_is_reported_not_stored_and_retried(conn):
    archive_games(["0022500083"])
    box, pbp = copy.deepcopy(load("0022500095"))
    p = box["game"]["homeTeam"]["players"][0]
    p["statistics"]["plusMinusPoints"] = float(p["statistics"]["plusMinusPoints"]) + 9
    p2 = next(x for x in box["game"]["awayTeam"]["players"] if x["statistics"]["minutes"] != "PT00M00.00S")
    p2["statistics"]["minutes"] = "PT01M00.00S"        # way off the play-by-play
    archive.save_raw("nba_cdn", "playbyplay", DAY, "0022500095", pbp)
    archive.save_raw("nba_cdn", "boxscore", DAY, "0022500095", box)
    s = ss.build(conn)
    assert s["attempted"] == 2 and s["validated"] == 1
    (f,) = s["failures"]
    assert f["game_id"] == "0022500095" and set(f["causes"]) == {"minutes", "plus_minus"}
    assert s["failure_causes"] == {"minutes": 1, "plus_minus": 1}
    assert ss.stored_games(conn) == {"0022500083"}
    assert conn.execute("SELECT COUNT(*) FROM stints WHERE game_id='0022500095'").fetchone()[0] == 0
    chk = conn.execute("SELECT * FROM data_checks WHERE check_name='stints_store'").fetchone()
    assert chk["status"] == "fail" and chk["value"] == pytest.approx(0.5)
    assert "0022500095" in chk["detail"]
    # retried (not skipped) on the next build, now with a good boxscore
    archive_games(["0022500095"])
    s = ss.build(conn)
    assert s["attempted"] == 1 and s["validated"] == 1 and len(ss.stored_games(conn)) == 2


def test_unreadable_and_empty_payloads_do_not_crash(conn):
    archive_games(["0022500083"])
    archive.save_raw("nba_cdn", "playbyplay", DAY, "0022500001", {"game": {"actions": []}})
    archive.save_raw("nba_cdn", "boxscore", DAY, "0022500001", load("0022500001")[0])
    p = archive.raw_path("nba_cdn", "playbyplay", DAY, "0022500095")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not gzip")
    archive.save_raw("nba_cdn", "boxscore", DAY, "0022500095", load("0022500095")[0])
    s = ss.build(conn)
    assert s["validated"] == 1
    assert {f["game_id"]: f["causes"] for f in s["failures"]} == {
        "0022500001": ["no_actions"], "0022500095": ["unreadable"]}


def test_season_of():
    assert ss.season_of("0021900001") == "2019-20"
    assert ss.season_of("0042400131") == "2024-25"
    assert ss.season_of("0022500001") == "2025-26"
    assert ss.season_of("0029900001") == "2099-00"


# ------------------------------------------------------------------------- garbage time

def _ev(idx, period, clock, home, away, elapsed=None):
    length = 720.0 if period <= 4 else 300.0
    start = 720.0 * (period - 1) if period <= 4 else 2880.0 + 300.0 * (period - 5)
    return Event(idx=idx, action_number=idx, order=idx, period=period, clock=clock,
                 elapsed=start + length - clock if elapsed is None else elapsed, team_id=None,
                 team=None, side=None, player_id=None, action_type="period", sub_type="",
                 descriptor=None, qualifiers=(), description="", points=0,
                 score_home=home, score_away=away)


def _stint(period, start, end, first, last, home, away):
    return Stint(period=period, start=start, end=end, first_idx=first, last_idx=last,
                 home=tuple(home), away=tuple(away), home_pts=0, away_pts=0, home_poss=0, away_poss=0)


HOME5, AWAY5 = (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)
BENCH_H, BENCH_A = (6, 7, 8, 9, 10), (16, 17, 18, 19, 20)


def _garbage(margin, lineup_home, lineup_away, starters, period=4):
    start = 720.0 * (period - 1)
    events = [_ev(0, period, 720.0, 100 + margin, 100), _ev(1, period, 0.0, 100 + margin, 100)]
    stints = [_stint(period, start, start + 720.0, 0, 2, lineup_home, lineup_away)]
    return ss.garbage_seconds(events, stints, starters)[0]


def test_garbage_time_definition_on_a_synthetic_game():
    starters = HOME5 + AWAY5
    # margin 30 all fourth quarter, bench units on the floor: the whole quarter is garbage
    assert _garbage(30, BENCH_H, BENCH_A, starters) == pytest.approx(720.0)
    # margin 15: needs >= 20 above 6:00, so only the last 6:00 (>= 10) counts
    assert _garbage(15, BENCH_H, BENCH_A, starters) == pytest.approx(360.0)
    # margin 22: >= 25 never above 9:00; >= 20 from 9:00 on
    assert _garbage(22, BENCH_H, BENCH_A, starters) == pytest.approx(540.0)
    # margin 9 is never garbage; neither is a 30-point margin in the third quarter
    assert _garbage(9, BENCH_H, BENCH_A, starters) == 0.0
    assert _garbage(30, BENCH_H, BENCH_A, starters, period=3) == 0.0
    # more than two starters on the floor (here 3) cancels it; two is still garbage
    assert _garbage(30, (1, 2, 3, 9, 10), BENCH_A, starters) == 0.0
    assert _garbage(30, (1, 11, 8, 9, 10), BENCH_A, starters) == pytest.approx(720.0)
    # the trailing team being ahead is symmetric
    assert _garbage(-30, BENCH_H, BENCH_A, starters) == pytest.approx(720.0)


def test_garbage_seconds_reach_the_player_rows(conn):
    """The blowout fixture has garbage time; the 1-point finish does not."""
    archive_games(["0022500622", "0022500083"])
    ss.build(conn)
    blow = conn.execute("SELECT SUM(sec_garbage) s, MAX(sec_garbage) m FROM player_game_oncourt "
                        "WHERE game_id='0022500622'").fetchone()
    close = conn.execute("SELECT SUM(sec_garbage) FROM player_game_oncourt "
                         "WHERE game_id='0022500083'").fetchone()[0]
    assert blow["s"] > 0 and blow["m"] > 60
    assert close == 0


# ------------------------------------------------------------------------- legacy dialects

def _sub(order, pid, team_id, **kw):
    return {"actionType": "substitution", "subType": "out", "personId": pid, "period": 1,
            "clock": "PT05M00.00S", "orderNumber": order, "teamId": team_id, **kw}


def test_repair_adds_the_missing_in_action_for_one_sided_substitutions():
    pbp = {"game": {"actions": [
        _sub(10000, 7, 1, subsInPersonId=9, subsInPlayerName="Nine"),
        {"actionType": "2pt", "subType": "jumpshot", "personId": 9, "period": 1,
         "clock": "PT04M00.00S", "orderNumber": 20000, "shotResult": ": Made"}]}}
    out = ss.repair_legacy_pbp(pbp)
    acts = out["game"]["actions"]
    assert [(a["subType"], a["personId"]) for a in acts[:2]] == [("out", 7), ("in", 9)]
    assert acts[1]["orderNumber"] == 10001 and acts[2]["shotResult"] == "Made"
    assert pbp["game"]["actions"][1]["shotResult"] == ": Made"      # input untouched
    ev = normalize(out)
    assert [(e.sub_dir, e.player_id) for e in ev if e.is_sub] == [("out", 7), ("in", 9)]
    assert next(e for e in ev if e.is_fga).made is True
    # already two-sided / modern payloads are returned as they came
    modern = {"game": {"actions": [_sub(10000, 7, 1), {**_sub(10001, 9, 1), "subType": "in"}]}}
    assert ss.repair_legacy_pbp(modern) is modern


def test_repair_resolves_sub_for_descriptions_by_family_name():
    box = {"game": {"homeTeam": {"teamId": 1, "players": [
        {"personId": 7, "familyName": "Out", "nameI": "A. Out", "name": "A Out"},
        {"personId": 9, "familyName": "Jones Jr.", "nameI": "B. Jones Jr.", "name": "B Jones Jr."}]},
        "awayTeam": {"teamId": 2, "players": [{"personId": 9, "familyName": "Jones Jr."}]}}}
    pbp = {"game": {"actions": [_sub(10000, 7, 1, description="SUB: Jones Jr. FOR Out")]}}
    acts = ss.repair_legacy_pbp(pbp, box)["game"]["actions"]
    assert [(a["subType"], a["personId"]) for a in acts] == [("out", 7), ("in", 9)]
    # a family name shared by two teammates stays unresolved
    box["game"]["homeTeam"]["players"].append({"personId": 10, "familyName": "Jones Jr."})
    assert ss.repair_legacy_pbp(pbp, box) is pbp


def test_stub_boxscore_and_missing_plus_minus(conn):
    box, pbp = copy.deepcopy(load("0022500083"))
    stub = {"game": {"gameId": "0022500095", "homeTeam": {"players": [{}]}}}
    assert ss.compute_game(stub, pbp, "0022500095").causes == ["stub_boxscore"]
    # plusMinusPoints all zero (the Feb-Mar 2020 'finalbox' boxscores): minutes still validated,
    # but the game is not 'strict'
    for t in ("homeTeam", "awayTeam"):
        for p in box["game"][t]["players"]:
            p["statistics"]["plusMinusPoints"] = 0.0
    r = ss.compute_game(box, pbp, "0022500083")
    assert r.ok and not r.strict
    # a game still in progress is refused
    box["game"]["gameStatus"] = 2
    assert ss.compute_game(box, pbp, "0022500083").causes == ["not_final"]
