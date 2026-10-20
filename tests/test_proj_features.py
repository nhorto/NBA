"""Feature store (Phase 2F F0-F2): no lookahead per group, shrinkage, F1 roles, determinism."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import features as FX
from courtside.proj import state as ST
from courtside.proj.features import common as CM
from courtside.proj.features import opp_avail as F1
from courtside.proj.features import opp_defense as F2

SEASONS = {"2021-22": date(2021, 10, 20), "2022-23": date(2022, 10, 20),
           "2023-24": date(2023, 10, 20)}
TEAMS = ("AAA", "BBB", "CCC", "DDD")
N_DAYS = 36
GRP = ["G", "G", "G", "G", "F", "F", "F", "C", "C", "C"]
STARTERS = (0, 1, 4, 5, 7)
BASE = {"G": dict(pts=.45, fg3m=.08, oreb=.02, dreb=.10, ast=.18, stl=.04, blk=.01, tov=.07,
                  fga=.38, fta=.10, pf=.07),
        "F": dict(pts=.42, fg3m=.05, oreb=.05, dreb=.16, ast=.08, stl=.03, blk=.02, tov=.05,
                  fga=.34, fta=.10, pf=.08),
        "C": dict(pts=.40, fg3m=.01, oreb=.10, dreb=.25, ast=.06, stl=.02, blk=.06, tov=.05,
                  fga=.30, fta=.12, pf=.10)}
# opponent effects the context layer can find: DDD gives up boards to centres, CCC forces TOs
OPP_EFFECT = {("DDD", "C", "dreb"): 1.4, ("DDD", "C", "oreb"): 1.4, ("CCC", None, "tov"): 1.35}
SCHED = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]


def make_league(seed: int = 0) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    rng = np.random.default_rng(seed)
    prow, trow, crow, rost = [], [], [], []
    gid = 0
    for season, start in SEASONS.items():
        for ti, t in enumerate(TEAMS):
            for j in range(10):
                rost.append((season, ti + 1, t, 100 * (ti + 1) + j, GRP[j]))
        for k in range(N_DAYS):
            day = start + timedelta(days=2 * k)
            for hi, ai in SCHED[k % 3]:
                gid += 1
                g = f"00{season[2:4]}{gid:05d}"
                box = {}
                for side, (ti, oi) in enumerate(((hi, ai), (ai, hi))):
                    t, o = TEAMS[ti], TEAMS[oi]
                    rows = []
                    for j in range(10):
                        if rng.random() < 0.08:            # sits
                            continue
                        mins = float(np.clip(rng.normal(32 if j in STARTERS else 16, 3), 4, 42))
                        x = {}
                        for s_, r in BASE[GRP[j]].items():
                            eff = OPP_EFFECT.get((o, GRP[j], s_), OPP_EFFECT.get((o, None, s_), 1))
                            x[s_] = int(rng.poisson(r * mins * eff))
                        x["reb"] = x["oreb"] + x["dreb"]
                        x["fga"] = max(x["fga"], x["fg3m"])
                        fgm = min(x["fga"], int(0.45 * x["fga"]) + x["fg3m"])
                        rows.append((g, 100 * (ti + 1) + j, f"P{ti}{j}", t, season,
                                     "Regular Season", day.isoformat(), mins, fgm, x["fga"],
                                     x["fg3m"], x["fg3m"] * 3, 0, x["fta"], x["oreb"], x["dreb"],
                                     x["reb"], x["ast"], x["stl"], x["blk"], x["tov"], x["pf"],
                                     x["pts"], 0.0))
                    prow += rows
                    a = np.array([r[8:23] for r in rows], float).sum(0)
                    box[t] = (o, side == 0, a)
                for t, (o, home, a) in box.items():
                    ob = box[o][2]
                    trow.append((g, t, season, "Regular Season", day.isoformat(), o, int(home),
                                 "W" if a[14] > ob[14] else "L", 240.0, *a[:15], 0.0))
                crow.append((g, "test", "test", 0.0, -2.0, 205.0, 205.0 + rng.normal(0, 4)))
    c.executemany("INSERT INTO player_games(game_id, player_id, player_name, team, season, "
                  "season_type, game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, "
                  "ast, stl, blk, tov, pf, pts, plus_minus) VALUES (" + ",".join("?" * 24) + ")",
                  prow)
    c.executemany("INSERT INTO team_games(game_id, team, season, season_type, game_date, opp, "
                  "is_home, wl, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, "
                  "blk, tov, pf, pts, plus_minus) VALUES (" + ",".join("?" * 25) + ")", trow)
    c.executemany("INSERT INTO closing_lines(game_id, source, provider, home_spread_open, "
                  "home_spread_close, total_open, total_close) VALUES (?,?,?,?,?,?,?)", crow)
    c.executemany("INSERT INTO team_rosters(season, team_id, team, player_id, position) "
                  "VALUES (?,?,?,?,?)", rost)
    c.commit()
    return c


def params() -> dict:
    p = ST.default_params()
    p["history_seasons"] = 1
    return p


SEASON_LIST = list(SEASONS)


def feature_rows(conn, seasons=SEASON_LIST) -> pd.DataFrame:
    sp = params()
    wf = CM.base_rows(conn, seasons, sp)
    tg = FX.team_game_features(conn, wf, sp)
    return FX.attach(wf, tg)


def feature_cols(fr: pd.DataFrame) -> list[str]:
    from courtside.proj.features import player
    return [c for c in fr.columns if c.startswith(("f1_", "f2_")) or c in player.COLS]


@pytest.fixture(scope="module")
def league():
    return make_league()


@pytest.fixture(scope="module")
def frame(league):
    return feature_rows(league)


# ------------------------------------------------------------------------- primitives

def test_ew_asof_strictly_before_and_weights():
    h_key = np.array(["a", "a", "a", "b"])
    h_day = np.array([0, 10, 20, 5])
    h_sidx = np.array([0, 0, 1, 0])
    vals = np.array([[1.0], [2.0], [4.0], [8.0]])
    q_key = np.array(["a", "a", "b", "c"])
    q_day = np.array([10, 30, 5, 99])
    q_sidx = np.array([0, 1, 0, 0])
    s = CM.ew_asof(h_key, h_day, h_sidx, vals, q_key, q_day, q_sidx, half_life=10.0, rho=0.5)
    # day-10 query sees only the day-0 row (same day excluded), at half weight
    assert s[0, 0] == pytest.approx(0.5)
    # day-30 (season 1): day 0 (x 2^-3 x 0.5), day 10 (x 2^-2 x 0.5), day 20 (x 2^-1)
    assert s[1, 0] == pytest.approx(1 * 0.125 * 0.5 + 2 * 0.25 * 0.5 + 4 * 0.5)
    assert s[2, 0] == 0.0 and s[3, 0] == 0.0


def test_regulars_rule_and_trade():
    rows = []
    for k in range(12):
        rows.append(("g%02d" % k, "AAA", 1, k, 2023))
        if k < 6:
            rows.append(("g%02d" % k, "AAA", 2, k, 2023))      # then traded to BBB
        else:
            rows.append(("h%02d" % k, "BBB", 2, k, 2023))
        if k in (0, 1, 2, 3, 4, 5, 6, 7):
            rows.append(("g%02d" % k, "AAA", 3, k, 2023))      # then sits
    box = pd.DataFrame(rows, columns=["game_id", "team", "player_id", "day", "sidx"])
    r = CM.regulars(box)
    a = r[r["team"] == "AAA"]
    # player 3 sat in games 8.. and is a regular (7 of the last 10) -> absent rows exist
    p3 = a[(a["player_id"] == 3) & ~a["played"]]
    assert len(p3) and p3["regular"].all()
    # player 2's latest earlier appearance is for BBB from game 7 on -> never absent for AAA
    p2 = a[(a["player_id"] == 2) & ~a["played"]]
    assert set(p2["game_id"]) <= {"g06"}


# ------------------------------------------------------------------------- no lookahead

def _scramble_from(conn, cut: str, seed: int = 1) -> sqlite3.Connection:
    """Copy of the DB with every counting stat on/after `cut` permuted (minutes and who played
    kept: the avail oracle), team boxes and closing totals re-randomised."""
    c2 = sqlite3.connect(":memory:")
    conn.backup(c2)
    rng = np.random.default_rng(seed)
    cols = ["fgm", "fga", "fg3m", "fg3a", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk",
            "tov", "pf", "pts"]
    rows = c2.execute("SELECT rowid FROM player_games WHERE game_date >= ?", (cut,)).fetchall()
    for (rid,) in rows:
        vals = [int(v) for v in rng.integers(0, 12, len(cols))]
        c2.execute("UPDATE player_games SET " + ", ".join(f"{k}=?" for k in cols)
                   + " WHERE rowid=?", (*vals, rid))
    c2.execute("UPDATE team_games SET pts = pts + 7, oreb = oreb + 3, tov = tov + 2 "
               "WHERE game_date >= ?", (cut,))
    c2.execute("UPDATE closing_lines SET total_close = total_close + 9 WHERE game_id IN "
               "(SELECT game_id FROM team_games WHERE game_date > ?)", (cut,))
    c2.commit()
    return c2


@pytest.mark.parametrize("prefix", ["p_", "f1_", "f2_"])
def test_no_lookahead_scramble(league, frame, prefix):
    """L1 per group: scrambling every stat on/after the cut leaves every feature of the games
    on the cut day (and before) identical. (The closing total of the cut day's own games is
    kept: it is a pre-tip input.)"""
    cut = (SEASONS["2023-24"] + timedelta(days=30)).isoformat()
    other = feature_rows(_scramble_from(league, cut))
    cols = [c for c in feature_cols(frame) if c.startswith(prefix)]
    assert cols
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other[other["game_date"] <= cut].sort_values(key)
    assert len(a) == len(b) and len(a) > 1000
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())
    # and the scramble did change later features (the test has teeth)
    if prefix == "f2_":
        a2 = frame[frame["game_date"] > cut].sort_values(key)
        b2 = other[other["game_date"] > cut].sort_values(key)
        assert not np.allclose(a2[cols].to_numpy(), b2[cols].to_numpy())


def test_no_lookahead_truncation(league, frame):
    """Physically removing every row after the cut leaves features up to the cut identical."""
    cut = (SEASONS["2023-24"] + timedelta(days=40)).isoformat()
    c2 = sqlite3.connect(":memory:")
    league.backup(c2)
    for t in ("player_games", "team_games"):
        c2.execute(f"DELETE FROM {t} WHERE game_date > ?", (cut,))
    c2.commit()
    other = feature_rows(c2)
    cols = feature_cols(frame)
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other.sort_values(key)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


# ------------------------------------------------------------------------- shrinkage

def test_dvp_shrinkage():
    """No history -> index 0 (league); heavy K -> towards 0; light K -> towards raw Y/E."""
    sums = {"league": np.zeros((2, 2)), "team": np.array([[0.0, 0.0], [30.0, 20.0]]),
            "G": np.array([[0.0, 0.0], [30.0, 20.0]]),
            "F": np.zeros((2, 2)), "C": np.zeros((2, 2))}
    scale = {"reb": (10.0, 3.0)}
    heavy = F2.dvp_indices(sums, scale, {"reb": 1e6}, {"reb": 1e6}, stats=("reb",))
    light = F2.dvp_indices(sums, scale, {"reb": 1e-6}, {"reb": 1e-6}, stats=("reb",))
    mid = F2.dvp_indices(sums, scale, {"reb": 2.0}, {"reb": 5.0}, stats=("reb",))
    assert heavy.iloc[0].abs().max() == 0.0                 # no history at all
    assert abs(heavy.loc[1, "f2_dvp_reb_G"]) < 1e-4
    assert light.loc[1, "f2_dvp_reb_G"] == pytest.approx(np.log(1.5), rel=1e-4)
    assert 0 < mid.loc[1, "f2_dvp_reb_all"] < np.log(1.5)
    # an empty group shrinks to its team's level
    assert light.loc[1, "f2_dvp_reb_F"] == pytest.approx(light.loc[1, "f2_dvp_reb_all"])
    # league-relative: a league-wide drift (everybody allows 1.5x the EB expectation) is not
    # opponent information -> an opponent at the league level has index 0
    drift = dict(sums, league=np.array([[0.0, 0.0], [3000.0, 2000.0]]))
    rel = F2.dvp_indices(drift, scale, {"reb": 2.0}, {"reb": 5.0}, stats=("reb",))
    assert abs(rel.loc[1, "f2_dvp_reb_all"]) < 1e-3 and abs(rel.loc[1, "f2_dvp_reb_G"]) < 1e-3


def test_box_rates_shrink_to_league(league):
    t = CM.load_team_games(league, "2021-22", "2023-24")
    q = pd.DataFrame({"team": ["AAA", "ZZZ"], "day": [t["day"].max() + 1] * 2,
                      "sidx": [2023, 2023]})
    br = F2.box_rates(t, q)
    assert np.allclose(br.iloc[1].to_numpy(), 0.0)          # no games -> the league rate
    assert br.iloc[0].abs().max() > 0


def test_opponent_effects_visible(frame):
    """The DvP index finds the planted effects: DDD gives up boards to centres, CCC forces
    turnovers (late in the last season, after shrinkage)."""
    late = frame[(frame["season"] == "2023-24") & (frame["day"] >= frame["day"].max() - 20)]
    vs_d = late[(late["opp"] == "DDD") & (late["grp"] == 2)]["f2_dvp_dreb_grp"].mean()
    vs_other = late[(late["opp"] != "DDD") & (late["grp"] == 2)]["f2_dvp_dreb_grp"].mean()
    assert vs_d > vs_other + 0.05
    tov_c = late[late["opp"] == "CCC"]["f2_dvp_tov_all"].mean()
    tov_o = late[late["opp"] != "CCC"]["f2_dvp_tov_all"].mean()
    assert tov_c > tov_o + 0.05


# ------------------------------------------------------------------------- F1

def test_f1_roles():
    R = pd.DataFrame({
        "key": ["k"] * 5,
        "absent": [False, False, True, False, False],
        "mpg": [34.0, 30.0, 28.0, 22.0, 12.0],
        "grp": [0, 1, 2, 2, 2],
        "r_blk": [0.01, 0.02, 0.08, 0.03, 0.2],          # the 12-mpg player is ignored
        "r_dreb": [0.10, 0.15, 0.30, 0.20, 0.5],
        "r_ast": [0.25, 0.08, 0.05, 0.04, 0.0],
        "r_stl": [0.05, 0.03, 0.02, 0.02, 0.0],
        "r_pts": [0.6, 0.5, 0.4, 0.3, 0.3]})
    f = F1.team_features(R).loc["k"]
    assert f["f1_n_abs"] == 1 and f["f1_vac_min"] == 28.0 and f["f1_vac_C"] == 28.0
    assert f["f1_rim_top"] == 1 and f["f1_reb_top"] == 1
    assert f["f1_bh_top"] == 0 and f["f1_perim_top"] == 0
    blk = np.array([0.34, 0.6, 2.24, 0.66])
    assert f["f1_rim_share"] == pytest.approx(2.24 / blk.sum())
    assert f["f1_depth"] == 3


def test_f1_scenarios_match_build(league, frame):
    """scenario_features (the sim hook) on the opponent's regulars with the actual absent set
    equals the team-game table built from the box."""
    sp = params()
    wf = CM.base_rows(league, SEASON_LIST, sp)
    reg = CM.regulars(wf[["game_id", "team", "player_id", "day", "sidx"]])
    row = frame[frame["season"] == "2023-24"].iloc[200]
    gid, opp = row["game_id"], row["opp"]
    R = reg[(reg["game_id"] == gid) & (reg["team"] == opp) & reg["regular"]]
    q = pd.DataFrame({"player_id": R["player_id"].astype(np.int64),
                      "game_date": row["game_date"]})
    st = ST.rate_frame(league, q, sp, games=ST.load_games(league))
    st["absent"] = ~R["played"].to_numpy()
    sc = F1.scenario_features(st, np.array([st["absent"].to_numpy(),
                                            np.zeros(len(st), bool)]))
    for c in F1.TEAM_COLS:
        assert sc.loc[0, c] == pytest.approx(row[c]), c
    assert sc.loc[1, "f1_n_abs"] == 0 and sc.loc[1, "f1_vac_min"] == 0


def test_deterministic(league, frame):
    again = feature_rows(league)
    cols = feature_cols(frame)
    pd.testing.assert_frame_equal(frame[cols].reset_index(drop=True),
                                  again[cols].reset_index(drop=True))


def test_registry_columns_exist(frame):
    for s in CM.DK_STATS:
        for g in ("P", "F1", "F2"):          # F3 / F5 / F6: extend_frame (test_proj_features2)
            for c in FX.model_columns(s, [g]):
                assert c in frame.columns, (s, g, c)
        mono = FX.monotone_for(s, ["P", "F1", "F2"])
        assert all(c in FX.model_columns(s, ["P", "F1", "F2"]) for c in mono)


def test_day_features_match_frame(league, frame):
    """The pipeline hook (one day's games as queries, history that includes later days)
    reproduces the frame's F2 columns for that day."""
    sp = params()
    wf = CM.base_rows(league, SEASON_LIST, sp)
    t = CM.load_team_games(league, SEASON_LIST[0], SEASON_LIST[-1])
    day = (SEASONS["2023-24"] + timedelta(days=50)).isoformat()
    tg = t[(t["game_date"] == day) & (t["is_home"] == 1)]
    games = pd.DataFrame({"game_id": tg["game_id"], "home": tg["team"], "away": tg["opp"]})
    out = FX.day_features(league, wf, t, day, games)
    cols = list(F2.TEAM_COLS)
    ref = (frame[frame["game_date"] == day].drop_duplicates(["game_id", "team"])
           .set_index(["game_id", "team"])[cols])
    got = out.set_index(["game_id", "team"]).loc[ref.index, cols]
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-10, atol=1e-12)
