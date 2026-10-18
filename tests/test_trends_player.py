"""Player pulse on synthetic leagues: impact metric, streaks, role changes, hot/cold, highs."""

import sqlite3

import pandas as pd
import pytest

from courtside.trends import player as P
from courtside.trends.synthetic import day_date, make_league, player_id


def league(player_fn=None, **kw):
    conn = sqlite3.connect(":memory:")
    make_league(conn, player_fn=player_fn, **kw)
    return conn


def test_impact_metric_by_hand():
    row = dict(game_id="g", team="A", season="2024-25", pts=30.0, fgm=11.0, fga=20.0, ftm=6.0,
               fta=8.0, fg3m=2.0, oreb=1.0, dreb=5.0, stl=2.0, ast=6.0, blk=1.0, pf=3.0,
               tov=3.0, min=36.0, plus_minus=10.0, usg_pct=0.30)
    mate = dict(row, pts=0.0, fgm=0.0, fga=0.0, ftm=0.0, fta=0.0, fg3m=0.0, oreb=0.0, dreb=0.0,
                stl=0.0, ast=0.0, blk=0.0, pf=0.0, tov=0.0, min=204.0, plus_minus=40.0,
                usg_pct=0.2)
    df = pd.DataFrame([row, mate])
    ci = P.impact_per_game(df).iloc[0]
    gmsc = 30 + 0.4 * 11 - 0.7 * 20 - 0.4 * 2 + 0.7 * 1 + 0.3 * 5 + 2 + 0.7 * 6 + 0.7 - 0.4 * 3 - 3
    lg_ts = 30 / (2 * (20 + 0.44 * 8))            # league TS from the frame
    par = lg_ts - 0.25 * (0.30 - 0.20)
    pop = 30 - 2 * (20 + 0.44 * 8) * par
    rel = 10 - 36 * (50 / 240)                    # team +10 margin => +50 summed on-court
    assert ci == pytest.approx(gmsc + 0.5 * pop + 0.15 * rel)


def test_player_streaks():
    def fn(season, day, team, slot, row):
        if team == "NYK" and slot == 3 and season == "2024-25":
            row["pts"] = 25 if day >= 24 else 12
        if team == "DEN" and slot == 4 and season == "2024-25" and day >= 25:
            row["pts"], row["reb"] = 14, 12
        return row

    conn = league(fn)
    pp = P.player_pulse(conn, day_date("2024-25", 30))
    s = {(x["player"], x["kind"]): x for x in pp["streaks"]}
    assert s[("NYK Player 3", "20_point_games")]["games"] == 6
    assert s[("NYK Player 3", "20_point_games")]["since"] == day_date("2024-25", 24).isoformat()
    assert s[("DEN Player 4", "double_doubles")]["games"] == 5


def test_streak_carries_over_from_prior_season():
    def fn(season, day, team, slot, row):
        if team == "BOS" and slot == 0:
            row["pts"] = 22 if (season == "2024-25" or day >= 25) else 10
        return row

    conn = league(fn)
    pp = P.player_pulse(conn, day_date("2024-25", 6))
    st = next(x for x in pp["streaks"] if x["player"] == "BOS Player 0"
              and x["kind"] == "20_point_games")
    assert st["games"] == 11 and st["games_this_season"] == 6 and not st["at_least"]


def test_role_changes():
    def fn(season, day, team, slot, row):
        if season != "2024-25":
            return row
        if team == "LAL" and slot == 8:          # deep bench -> rotation
            if day < 25:
                return None if day % 2 else {**row, "min": 4.0}
            return {**row, "min": 32.0}
        if team == "GSW" and slot == 1 and day >= 26:   # starter vanishes
            return None
        if team == "DEN" and slot == 6 and day >= 25:   # 22 -> 33 mpg
            return {**row, "min": 33.0}
        return row

    conn = league(fn)
    pp = P.player_pulse(conn, day_date("2024-25", 30))
    rc = {}
    for r in pp["role_changes"]:
        rc.setdefault(r["player"], []).append(r)
    kinds = lambda name: {r["kind"] for r in rc.get(name, [])}
    assert "new_in_rotation" in kinds("LAL Player 8")
    out = next(r for r in rc["GSW Player 1"] if r["kind"] == "out_of_rotation")
    assert out["team_games_missed"] == 4
    assert out["last_played"] == day_date("2024-25", 25).isoformat()
    assert "minutes_up" in kinds("DEN Player 6")
    mu = next(r for r in rc["DEN Player 6"] if r["kind"] == "minutes_up")
    assert mu["change"] == pytest.approx(11.0)
    # unchanged regulars produce nothing
    assert not rc.get("BOS Player 0")


def test_hot_player_and_early_suppression():
    def fn(season, day, team, slot, row):
        if team == "MIA" and slot == 2 and season == "2024-25" and day >= 25:
            return {**row, "pts": 38, "fgm": 14, "fga": 22, "fg3m": 4, "ftm": 6, "fta": 7}
        return row

    conn = league(fn)
    pp = P.player_pulse(conn, day_date("2024-25", 30))
    hot = {h["player"]: h for h in pp["hot"]}
    assert "MIA Player 2" in hot
    h = hot["MIA Player 2"]
    assert h["z"] >= P.HOT_Z and h["impact_recent"] > h["impact_baseline"]
    assert h["recent"]["pts"] == 38.0
    early = P.player_pulse(conn, day_date("2024-25", 4))
    assert early["hot"] == [] and early["cold"] == []
    assert any("early season" in n for n in early["notes"])


def test_career_high():
    def fn(season, day, team, slot, row):
        if team == "BOS" and slot == 0 and season == "2024-25" and day == 10:
            return {**row, "pts": 45}
        return row

    conn = league(fn)
    pp = P.player_pulse(conn, day_date("2024-25", 20))
    highs = [h for h in pp["career_highs"] if h["player_id"] == player_id("BOS", 0)
             and h["stat"] == "pts"]
    assert highs and highs[0]["value"] == 45
    assert highs[0]["previous_high"] is not None and highs[0]["previous_high"] < 45
    # before the game it isn't there
    pp0 = P.player_pulse(conn, day_date("2024-25", 10))
    assert not [h for h in pp0["career_highs"] if h["value"] == 45]


def test_leaders_last10():
    conn = league()
    pp = P.player_pulse(conn, day_date("2024-25", 20))
    lead = pp["leaders_last10"]
    assert set(lead) == {"impact", "pts", "reb", "ast", "stocks", "fg3m", "ts_pct"}
    assert len(lead["pts"]) == 5
    vals = [r["value"] for r in lead["pts"]]
    assert vals == sorted(vals, reverse=True)
    assert all(r["games"] == 10 for r in lead["pts"])
