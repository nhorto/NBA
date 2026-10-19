"""EB per-minute rate state (T1.1): no lookahead, shrinkage, carry-over, determinism, storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import state

STAT_COLS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
             "blk", "tov", "pf", "pts"]
SEASONS = {"2022-23": date(2022, 10, 20), "2023-24": date(2023, 10, 20),
           "2024-25": date(2024, 10, 20)}
N_GAMES = 60
STAR = 1          # high-rate regular
NEWBIE = 2        # only 2 games, in the last two game days of the final season
BASE_RATE = 0.4   # league points per minute


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    return c


def _fill(c: sqlite3.Connection, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_players = 24
    pids = list(range(1, n_players + 1))
    mult = {p: 1.0 for p in pids}
    mult[STAR] = 2.0
    mins = {p: float(rng.integers(14, 34)) for p in pids}
    mins[STAR] = 33.0
    rows, gid = [], 0
    for season, start in SEASONS.items():
        for k in range(N_GAMES):
            day = start + timedelta(days=2 * k)
            gid += 1
            last_two = season == "2024-25" and k >= N_GAMES - 2
            for p in pids:
                if p == NEWBIE and not last_two:
                    continue
                m = float(np.clip(rng.normal(mins[p], 3), 6, 40))
                pts = int(rng.poisson(BASE_RATE * mult[p] * m * (2.0 if p == NEWBIE else 1.0)))
                other = {k_: int(rng.poisson(r * m)) for k_, r in
                         (("fg3m", 0.06), ("oreb", 0.05), ("dreb", 0.13), ("ast", 0.1),
                          ("stl", 0.03), ("blk", 0.02), ("tov", 0.06), ("fga", 0.35),
                          ("fta", 0.1), ("pf", 0.09))}
                rows.append((f"G{gid:05d}", p, f"P{p}", "AAA" if p <= 12 else "BBB", season,
                             "Regular Season", day.isoformat(), m, 0, other["fga"], other["fg3m"],
                             0, 0, other["fta"], other["oreb"], other["dreb"],
                             other["oreb"] + other["dreb"], other["ast"], other["stl"],
                             other["blk"], other["tov"], other["pf"], pts, 0.0))
    c.executemany(
        "INSERT INTO player_games(game_id, player_id, player_name, team, season, season_type, "
        "game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, "
        "pts, plus_minus) VALUES (" + ",".join("?" * 24) + ")", rows)
    pos = ["G", "F", "C"]
    for season in SEASONS:
        c.executemany("INSERT INTO team_rosters(season, team_id, team, player_id, position) "
                      "VALUES (?,?,?,?,?)",
                      [(season, 1 if p <= 12 else 2, "AAA" if p <= 12 else "BBB", p, pos[p % 3])
                       for p in pids])
    c.commit()


def _wide(c, as_of, params=None):
    return state.to_wide(state.build_state(c, as_of, params), params).sort_index()


def test_no_lookahead(conn):
    cut = date(2024, 12, 1)
    full = state.build_state(conn, cut)
    conn.execute("DELETE FROM player_games WHERE game_date >= ?", (cut.isoformat(),))
    conn.commit()
    trimmed = state.build_state(conn, cut)
    pd.testing.assert_frame_equal(full, trimmed, check_exact=True)
    # walk-forward rows of the last game day also ignore later days
    wf1 = state.walk_forward(conn, "2024-25")
    conn.execute("DELETE FROM player_games WHERE game_date >= '2024-11-15'")
    conn.commit()
    wf2 = state.walk_forward(conn, "2024-25")
    key = wf1["game_date"] < "2024-11-15"
    cols = [c for c in wf1.columns if c.startswith(("r_", "n_", "p_"))] + ["mpg", "tier"]
    a = wf1[key].reset_index(drop=True)[cols]
    b = wf2.reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_walk_forward_matches_rate_frame_and_build_state(conn):
    wf = state.walk_forward(conn, "2024-25")
    row = wf[(wf["player_id"] == STAR)].iloc[20]
    q = pd.DataFrame({"player_id": [STAR], "game_date": [row["game_date"]]})
    rf = state.rate_frame(conn, q)
    for c in ("r_pts", "n_pts", "p_pts", "r_dk", "mpg"):
        assert rf[c].iloc[0] == pytest.approx(row[c], rel=1e-12)
    st = state.build_state(conn, date.fromisoformat(row["game_date"]))
    r = st[(st["player_id"] == STAR) & (st["stat"] == "pts")].iloc[0]
    assert r["rate"] == pytest.approx(row["r_pts"], rel=1e-12)
    assert r["prior"] == pytest.approx(row["p_pts"], rel=1e-12)
    # the same-day game is never part of the state: first game of the history has no past
    day1 = state.rate_frame(conn, pd.DataFrame({"player_id": [STAR], "game_date": ["2022-10-20"]}))
    assert day1["n_pts"].iloc[0] == 0.0 and day1["r_pts"].iloc[0] == day1["p_pts"].iloc[0]


def test_shrinkage_two_games_near_prior_big_sample_near_own_rate(conn):
    as_of = date(2025, 3, 1)
    w = _wide(conn, as_of)
    prior = w["p_pts"]
    # the newbie: 2 games at twice the league rate, ~60 minutes of evidence
    new = w.loc[NEWBIE]
    assert new["n_pts"] < 120
    assert abs(new["r_pts"] - new["p_pts"]) < 0.5 * abs(0.8 - new["p_pts"])
    # the star: thousands of minutes at 0.8 pts/min
    star = w.loc[STAR]
    assert star["n_pts"] > 800
    assert star["r_pts"] > 0.68
    assert abs(star["r_pts"] - 0.8) < abs(star["r_pts"] - star["p_pts"]) / 2
    # ordinary players sit near the league rate; the prior itself is the league rate
    ordinary = w.drop(index=[STAR, NEWBIE])
    assert ordinary["r_pts"].mean() == pytest.approx(BASE_RATE, abs=0.04)
    assert np.all(np.isfinite(prior))


def test_prior_only_for_player_without_games(conn):
    q = pd.DataFrame({"player_id": [999], "game_date": ["2024-12-01"]})
    r = state.rate_frame(conn, q)
    assert r["n_pts"].iloc[0] == 0.0
    assert r["r_pts"].iloc[0] == pytest.approx(r["p_pts"].iloc[0])


def test_season_carry_over_decays_by_halflife_and_season_break(conn):
    params = state.default_params()
    h, rho = params["stats"]["pts"]["H"], params["stats"]["pts"]["rho"]
    end = date(2023, 4, 1)          # after the last 2022-23 game (day 118 after Oct 20)
    last = date(2022, 10, 20) + timedelta(days=2 * (N_GAMES - 1))
    assert last < end
    new_start = date(2023, 10, 20)
    a = _wide(conn, end, params)
    b = _wide(conn, new_start, params)           # no 2023-24 game yet
    assert b.loc[STAR, "n_pts"] > 0
    expected = 2.0 ** (-(new_start - end).days / h) * rho
    assert (b["n_pts"] / a["n_pts"]).dropna().to_numpy() == pytest.approx(expected, rel=1e-9)
    # early in the next season his rate still remembers last season (0.8), not just the league's
    assert b.loc[STAR, "r_pts"] > b.loc[STAR, "p_pts"] + 0.1
    assert b.loc[STAR, "r_pts"] < a.loc[STAR, "r_pts"]       # less evidence -> pulled to prior
    # the prior is already season-aware on day one (previous season's cell rates)
    assert b["p_pts"].between(0.3, 0.55).all()


def test_deterministic(conn):
    a = state.build_state(conn, date(2025, 1, 15))
    b = state.build_state(conn, date(2025, 1, 15))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    w1 = state.walk_forward(conn, "2023-24")
    w2 = state.walk_forward(conn, "2023-24")
    pd.testing.assert_frame_equal(w1, w2, check_exact=True)


def test_dk_rate_is_the_weighted_sum(conn):
    st = _wide(conn, date(2025, 1, 15))
    exp = sum(w * st[f"r_{s}"] for s, w in state.DK_WEIGHTS.items())
    pd.testing.assert_series_equal(st["r_dk"], exp, check_names=False)
    assert np.all(st["tier"].between(0, state.N_TIER - 1))


def test_write_state_idempotent_and_prunes(conn):
    d = date(2025, 1, 15)
    n1 = state.write_state(conn, d)
    rows1 = conn.execute("SELECT * FROM rate_state ORDER BY as_of, player_id, stat").fetchall()
    n2 = state.write_state(conn, d)
    rows2 = conn.execute("SELECT * FROM rate_state ORDER BY as_of, player_id, stat").fetchall()
    assert n1 == n2 == len(rows1) == len(rows2)
    assert [tuple(r) for r in rows1] == [tuple(r) for r in rows2]
    assert {r["stat"] for r in rows1} == set(state.STATS) | {"dk", "mpg"}
    w = state.read_state(conn)
    assert w.attrs["as_of"] == d.isoformat()
    ref = _wide(conn, d)
    assert w["r_pts"].to_numpy() == pytest.approx(ref["r_pts"].to_numpy(), rel=1e-12)
    assert (w["tier"] == ref["tier"]).all()
    state.write_state(conn, date(2025, 1, 16), keep=1)
    assert [r[0] for r in conn.execute("SELECT DISTINCT as_of FROM rate_state")] == ["2025-01-16"]


def test_minutes_multiplier_default_is_one_and_interpolates():
    p = state.default_params()
    assert np.all(state.minutes_multiplier(p, "pts", [10.0, 30.0], [20.0, 30.0]) == 1.0)
    p["minutes_adj"] = {"centers": [0.5, 1.0, 1.5], "mult": {"pts": [0.8, 1.0, 1.2]}}
    got = state.minutes_multiplier(p, "pts", [10.0, 20.0, 30.0, 60.0], [20.0] * 4)
    assert got == pytest.approx([0.8, 1.0, 1.2, 1.2])
    assert state.minutes_multiplier(p, "ast", [10.0], [20.0])[0] == 1.0


def test_nb_logpmf_normalises_and_reduces_to_poisson():
    y = np.arange(0, 200)
    for mu, phi in ((3.0, 0.2), (12.0, 0.05)):
        assert np.exp(state.nb_logpmf(y, mu, phi)).sum() == pytest.approx(1.0, abs=1e-6)
    from scipy.stats import poisson
    assert state.nb_logpmf(4, 3.0, 1e-6) == pytest.approx(poisson.logpmf(4, 3.0), abs=1e-4)


def test_fit_params_smoke(conn):
    p = state.fit_params(conn, ["2023-24", "2024-25"], history_seasons=1, h_grid=(60.0, 240.0),
                         rho_grid=(0.6, 1.0), kappa_grid=(25.0, 100.0, 400.0), passes=1)
    p2 = json.loads(json.dumps(p))
    assert p2 == p
    for s in state.STATS:
        v = p["stats"][s]
        assert v["H"] in (60.0, 240.0) and v["rho"] in (0.6, 1.0)
        assert v["kappa"] in (25.0, 100.0, 400.0)
        assert len(v["phi"]) == state.N_TIER and min(v["phi"]) > 0
        assert len(p["minutes_adj"]["mult"][s]) == len(p["minutes_adj"]["centers"])
    # the fitted params drive build_state
    st = state.build_state(conn, date(2025, 2, 1), p)
    assert set(st["stat"]) == set(state.STATS) | {"dk", "mpg"}
    assert st["rate"].notna().all()
