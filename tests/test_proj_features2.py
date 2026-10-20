"""Phase 2F second feature wave: F3 tracking, F5 matchups, F6 schedule. Leak tests in the
L1 / L5 style (scrambling or deleting everything on/after a cut leaves every feature of the
games up to the cut identical), shrinkage / era handling, planted signals, the registry, and
the per-day (pipeline) hooks reproducing the frame."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside.proj import features as FX
from courtside.proj import state as ST
from courtside.proj.features import common as CM
from courtside.proj.features import matchups as F5
from courtside.proj.features import schedule as F6
from courtside.proj.features import tracking as F3

from test_proj_features import SEASONS, make_league, params

SEASON_LIST = list(SEASONS)
BAD_DEF = 400          # DDD's first guard: gives up 1.6x the expected points
GUARD_PAIR = (100, 400)  # AAA's 100 is always guarded by DDD's 400 when they meet


def add_tracking(conn: sqlite3.Connection, seed: int = 0) -> sqlite3.Connection:
    """Tracking, hustle and matchup rows for every player-game of a `make_league` DB:
    rebound chances ~ 2 x rebounds, passes ~ 6 x assists, deflections ~ 2.5 x steals; matchups
    pair each attacker with the opposing players of the same slot (100 <-> 400 always), and
    BAD_DEF allows 1.6x points."""
    rng = np.random.default_rng(seed)
    pg = pd.read_sql_query("SELECT * FROM player_games", conn).fillna(0)
    trk, hus = [], []
    for r in pg.itertuples():
        m = float(r.min)
        trk.append((r.game_id, r.player_id, r.team, m, 4.2, m / 12,
                    int(r.oreb * 2 + rng.poisson(1)), int(r.dreb * 2 + rng.poisson(1)),
                    0, int(m * 1.5 + rng.poisson(3)), int(rng.poisson(0.5)), 0,
                    int(r.ast * 6 + rng.poisson(4)), int(r.ast), int(r.fgm // 2),
                    int(r.fga // 2), int(r.fgm - r.fgm // 2), int(r.fga - r.fga // 2),
                    int(rng.poisson(1)), int(rng.poisson(2))))
        hus.append((r.game_id, r.player_id, r.team, m, int(r.pts), int(rng.poisson(m / 6)),
                    int(rng.poisson(m / 9)), int(rng.poisson(m / 18)),
                    int(r.stl * 2.5 + rng.poisson(0.5)), 0, int(rng.poisson(0.5)), 0, 0, 0,
                    int(rng.poisson(0.5)), 0, int(rng.poisson(1)), 0, 0, int(rng.poisson(1))))
    conn.executemany("INSERT INTO player_tracking VALUES (" + ",".join("?" * 20) + ")", trk)
    conn.executemany("INSERT INTO player_hustle VALUES (" + ",".join("?" * 20) + ")", hus)
    mrows = []
    for gid, g in pg.groupby("game_id"):
        teams = g["team"].unique()
        for t in teams:
            att = g[g["team"] == t]
            dfn = g[g["team"] != t]
            for a in att.itertuples():
                slot = a.player_id % 100
                for d in dfn.itertuples():
                    same = (d.player_id % 100) == slot
                    if (a.player_id, d.player_id) == GUARD_PAIR:
                        frac = 2.0
                    else:
                        frac = 0.5 if same else 0.5 / max(len(dfn) - 1, 1)
                    mm = a.min * frac * 0.45
                    if mm < 0.05:
                        continue
                    poss = mm * 2.1
                    lift = 1.6 if d.player_id == BAD_DEF else 1.0
                    pts = int(rng.poisson(0.5 * poss * lift))
                    fga = int(rng.poisson(0.4 * poss))
                    mrows.append((gid, a.player_id, d.player_id, t, d.team, mm, poss, 0.1, 0.1,
                                  0.1, 0, pts, pts * 3, int(rng.poisson(0.1 * poss)), 0,
                                  int(rng.poisson(0.06 * poss)), 0, min(fga, pts // 2), fga, 0,
                                  int(rng.poisson(0.1 * poss)), 0, 0, 0, 0,
                                  int(rng.poisson(0.1 * poss)), 0))
    conn.executemany("INSERT INTO player_matchups VALUES (" + ",".join("?" * 27) + ")", mrows)
    conn.commit()
    return conn


def frame_rows(conn) -> pd.DataFrame:
    sp = params()
    wf = CM.base_rows(conn, SEASON_LIST, sp)
    tg = FX.team_game_features(conn, wf, sp)
    fr = FX.attach(wf, tg)
    return FX.extend_frame(conn, fr, ["F3", "F5"], cache=False)


def cols_of(fr, prefix):
    return [c for c in fr.columns if c.startswith(prefix)]


@pytest.fixture(scope="module")
def league():
    return add_tracking(make_league())


@pytest.fixture(scope="module")
def frame(league):
    return frame_rows(league)


def _copy(conn):
    c2 = sqlite3.connect(":memory:")
    conn.backup(c2)
    return c2


def _scramble(conn, cut: str):
    """Every tracking / hustle / matchup / box stat on or after `cut` re-randomised (who played
    and minutes kept: the avail oracle)."""
    c2 = _copy(conn)
    ids = [r[0] for r in c2.execute(
        "SELECT DISTINCT game_id FROM player_games WHERE game_date >= ?", (cut,))]
    q = ",".join("?" * len(ids))
    for t, cols in (("player_tracking", ("reb_chances_off", "reb_chances_def", "passes",
                                         "touches", "cont_fga", "uncont_fga", "def_rim_fga")),
                    ("player_hustle", ("deflections", "contested_2pt", "box_outs",
                                       "screen_assists")),
                    ("player_matchups", ("matchup_min", "partial_poss", "pts", "fga", "fgm",
                                         "ast", "tov"))):
        for c in cols:
            c2.execute(f"UPDATE {t} SET {c} = abs(random() % 13) WHERE game_id IN ({q})", ids)
    c2.execute(f"UPDATE player_games SET reb = reb + 4, ast = ast + 3, stl = stl + 2, "
               f"blk = blk + 1 WHERE game_id IN ({q})", ids)
    c2.commit()
    return c2


# ------------------------------------------------------------------------- no lookahead

@pytest.mark.parametrize("prefix", ["t3_", "f5_"])
def test_no_lookahead_scramble(league, frame, prefix):
    """L1 for F3 / F5: scrambling every tracking / matchup / box stat on or after the cut leaves
    the features of every game up to the cut day identical, and changes later ones."""
    cut = (SEASONS["2023-24"] + timedelta(days=30)).isoformat()
    other = frame_rows(_scramble(league, cut))
    cols = cols_of(frame, prefix)
    assert len(cols) > 5
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other[other["game_date"] <= cut].sort_values(key)
    assert len(a) == len(b) and len(a) > 1000
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())
    a2 = frame[frame["game_date"] > cut].sort_values(key)
    b2 = other[other["game_date"] > cut].sort_values(key)
    assert not np.allclose(a2[cols].to_numpy(), b2[cols].to_numpy(), equal_nan=True)


def test_no_lookahead_truncation(league, frame):
    """L5 for F3 / F5: physically deleting every row after the cut leaves features up to it
    identical (F3 gaps included: they read the EB state, itself strictly pre-day)."""
    cut = (SEASONS["2023-24"] + timedelta(days=40)).isoformat()
    c2 = _copy(league)
    ids = [r[0] for r in c2.execute(
        "SELECT DISTINCT game_id FROM player_games WHERE game_date > ?", (cut,))]
    q = ",".join("?" * len(ids))
    for t in ("player_tracking", "player_hustle", "player_matchups", "player_games",
              "team_games"):
        c2.execute(f"DELETE FROM {t} WHERE game_id IN ({q})", ids)
    c2.commit()
    other = frame_rows(c2)
    cols = cols_of(frame, "t3_") + cols_of(frame, "f5_")
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other.sort_values(key)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


def test_f6_only_earlier_games():
    """F6 of a game depends only on the team's earlier games and the game's own date / site:
    dropping later games and changing every result changes nothing."""
    g = _schedule()
    full = F6.from_games(g)
    cut = g["game_date"].iloc[len(g) // 2]
    part = g[g["game_date"] <= cut].copy()
    part["home_pts"], part["away_pts"] = 0, 0
    trunc = F6.from_games(part)
    key = ["game_id", "team"]
    a = full[full["game_id"].isin(part["game_id"])].sort_values(key).reset_index(drop=True)
    b = trunc.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------------- F3

def test_f3_era_and_prior(league):
    """Before any tracking data: NaN (missing, not 'average'); a player without his own history
    inside the era sits at the group prior (0) with no evidence."""
    pos = ST.load_positions(league)
    h = F3.load_history(league, pos=pos)
    first = int(h["day"].min())
    q = pd.DataFrame({"player_id": [100, 999999, 100], "day": [first, first + 200,
                                                               first + 200],
                      "sidx": [2021, 2022, 2022], "grp": [0, 0, 0]})
    f = F3.player_features(h, q)
    assert f.iloc[0].isna().all()
    assert np.allclose(f.loc[1, list(F3.RATE_COLS + F3.CONV_COLS)].to_numpy(float), 0.0)
    assert f.loc[1, "t3_ln_min"] == 0.0
    assert f.loc[2, "t3_ln_min"] > 3.0


def test_f3_gap_tracks_opportunity(frame):
    """Players with more rebound chances than their EB rebound rate implies get a positive
    rebound gap; the implied rates are on the per-minute scale of the EB rates."""
    late = frame[(frame["season"] == "2023-24")].dropna(subset=["t3_gap_reb"])
    assert len(late) > 500
    ratio = late["t3_impl_reb"] / late["r_reb"]
    assert 0.5 < ratio.median() < 2.0
    assert np.corrcoef(np.log(late["t3_impl_reb"]), np.log(late["r_reb"]))[0, 1] > 0.5


# ------------------------------------------------------------------------- F5

def test_f5_assignment_and_bad_defender(frame):
    """The planted assignment is found (AAA's 100 vs DDD: 400 is the likely primary, a high
    share) and the attacker facing the bad defender gets a positive points-quality index."""
    late = frame[(frame["season"] == "2023-24") & (frame["day"] > frame["day"].max() - 40)]
    a = late[(late["player_id"] == GUARD_PAIR[0]) & (late["opp"] == "DDD")]
    o = late[(late["player_id"] == GUARD_PAIR[0]) & (late["opp"] != "DDD")]
    assert len(a) and len(o)
    assert a["f5_prim_share"].mean() > o["f5_prim_share"].mean() + 0.1
    assert a["f5_prim_pts"].mean() > 0.05
    assert a["f5_q_pts"].mean() > o["f5_q_pts"].mean() + 0.02


def test_f5_quality_shrinks_without_history(league):
    """A defender with no matchup history has quality exactly 0 (the league ratio)."""
    h = F5.History(league)
    dq = F5._expected(h.m)
    q = pd.DataFrame({"player_id": [123456789, BAD_DEF], "day": [h.m["day"].max() + 1] * 2,
                      "sidx": [2023, 2023]})
    qual = F5.defender_quality(dq, q)
    assert np.allclose(qual.iloc[0].to_numpy(), 0.0)
    assert qual.loc[1, "pts"] > 0.1


# ------------------------------------------------------------------------- F6

def _schedule() -> pd.DataFrame:
    """A small real-tricode schedule: DEN, BOS, LAL, UTA; LAL plays a back-to-back at DEN."""
    from courtside.model import sites
    rows = []
    d0 = date(2023, 10, 24)
    plan = [(0, "DEN", "BOS"), (0, "LAL", "UTA"), (2, "BOS", "LAL"), (2, "UTA", "DEN"),
            (3, "DEN", "LAL"), (5, "BOS", "UTA"), (6, "LAL", "BOS"), (7, "UTA", "DEN"),
            (9, "DEN", "BOS"), (9, "LAL", "UTA")]
    for i, (k, h, a) in enumerate(plan):
        s = sites.home_site(h, "2023-24")
        rows.append({"game_id": f"00223{i:05d}", "season": "2023-24",
                     "season_type": "Regular Season",
                     "game_date": pd.Timestamp(d0 + timedelta(days=k)), "home": h, "away": a,
                     "home_raw": h, "away_raw": a, "home_pts": 110, "away_pts": 100,
                     "is_neutral": False, "site": s.name, "site_lat": s.lat,
                     "site_lon": s.lon, "site_tz": s.tz})
    return pd.DataFrame(rows)


def test_f6_values():
    f = F6.from_games(_schedule()).set_index(["game_id", "team"])
    r = f.loc[("0022300004", "LAL")]                    # LAL at DEN, day 3, after day-2 game
    assert r["f6_home"] == 0 and r["f6_b2b"] == 1 and r["f6_rest"] == 0
    assert r["f6_alt"] == pytest.approx(1.609) and r["f6_alt_vis"] == pytest.approx(1.609)
    assert r["f6_travel"] > 1.0                          # Boston -> Denver
    assert r["f6_tz"] == pytest.approx(-2.0)              # Eastern -> Mountain
    d = f.loc[("0022300004", "DEN")]
    assert d["f6_home"] == 1 and d["f6_alt_vis"] == 0.0 and d["f6_o_b2b"] == 1
    assert d["f6_rest_diff"] == d["f6_rest"] - r["f6_rest"]
    assert f.loc[("0022300000", "DEN"), "f6_stage"] == 0.0


# ------------------------------------------------------------------------- registry / hooks

def test_registry_columns_exist(frame):
    fr = frame.copy()
    for c in F6.TEAM_COLS:              # F6 needs real tricodes: its own tests cover the table
        fr[c] = 0.0
    for s in CM.DK_STATS:
        for grp in ("F3", "F5", "F6"):
            cols = FX.model_columns(s, [grp])
            assert cols
            for c in cols:
                assert c in fr.columns, (s, grp, c)
        mono = FX.monotone_for(s, ["P", "F2", "F3", "F5", "F6"])
        assert all(c in FX.model_columns(s, ["P", "F2", "F3", "F5", "F6"]) for c in mono)


def test_live_extras_match_frame(league, frame):
    """The pipeline hook (`LiveExtras.day`: one day's games, rosters with pre-game minutes
    weights, history that includes later days) reproduces the frame's F3 / F5 columns."""
    ex = FX.LiveExtras(league, "2023-24", ("F3", "F5"))
    day = (SEASONS["2023-24"] + timedelta(days=50)).isoformat()
    d = frame[frame["game_date"] == day]
    assert len(d) > 20
    rows = d[["game_id", "team", "player_id", "grp"]].copy()
    rows["w"] = np.maximum(d["mpg"].to_numpy(float), 1.0)
    g = d.drop_duplicates("game_id")
    tg = pd.read_sql_query("SELECT game_id, team, opp FROM team_games WHERE is_home = 1 AND "
                           "game_date = ?", league, params=(day,))
    games = pd.DataFrame({"game_id": tg["game_id"], "home": tg["team"], "away": tg["opp"]})
    assert set(games["game_id"]) == set(g["game_id"])
    tf, pf = ex.day(league, day, games, rows)
    cols = list(F3.TEAM_COLS + F5.TEAM_COLS)
    got = pf.set_index(["game_id", "player_id"]).loc[
        pd.MultiIndex.from_frame(d[["game_id", "player_id"]]), cols]
    np.testing.assert_allclose(got.to_numpy(float), d[cols].to_numpy(float), rtol=1e-9,
                               atol=1e-12)
    assert set(zip(tf["game_id"], tf["team"])) == set(zip(d["game_id"], d["team"]))
