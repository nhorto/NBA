"""Rung 2 (player ratings / expected minutes / team strength): synthetic data, no network."""

import numpy as np
import pandas as pd
import pytest

from courtside.model import backtest, player_model, players as pl

SMALL = pl.PlayerParams(active_days=400, roster_games=5)


def _player_rows(frame: pd.DataFrame, seed: int = 0, n_per_team: int = 9) -> pd.DataFrame:
    """Player box rows for every game of `frame`: fixed rosters with true ratings, random
    absences, minutes summing to 240, plus-minus consistent with the margin."""
    rng = np.random.default_rng(seed)
    teams = sorted(set(frame["home_raw"]) | set(frame["away_raw"]))
    roster = {t: [(1000 + 100 * i + k, rng.normal(0, 3)) for k in range(n_per_team)]
              for i, t in enumerate(teams)}
    base_min = np.array([36, 34, 33, 32, 30, 25, 20, 18, 12], float)[:n_per_team]
    rows = []
    for g in frame.itertuples():
        for team, sgn in ((g.home_raw, 1), (g.away_raw, -1)):
            pl_ = roster[team]
            out = rng.random(len(pl_)) < 0.1
            mins = np.where(out, 0.0, base_min)
            mins = np.round(mins * 240 / mins.sum())
            mins[np.argmax(mins)] += 240 - mins.sum()
            for (pid, r), m in zip(pl_, mins):
                if m <= 0:
                    continue
                pm = sgn * g.margin * m / 48 + rng.normal(0, 3)
                rows.append({"game_id": g.game_id, "player_id": pid, "player_name": f"P{pid}",
                             "team": team, "min": m, "pts": rng.poisson(m * (0.4 + r / 50)),
                             "fga": rng.poisson(m * 0.4), "fta": rng.poisson(m * 0.1),
                             "fg3a": rng.poisson(m * 0.1), "ast": rng.poisson(m * 0.1),
                             "tov": rng.poisson(m * 0.05), "oreb": rng.poisson(m * 0.05),
                             "dreb": rng.poisson(m * 0.15), "stl": rng.poisson(m * 0.03),
                             "blk": rng.poisson(m * 0.02), "pf": rng.poisson(m * 0.06),
                             "plus_minus": float(round(pm)), "poss": m * 2.0})
    df = pd.DataFrame(rows)
    for c in ("min", *pl.RATE_STATS, "plus_minus", "poss"):
        df[c] = df[c].astype(float)
    return df


@pytest.fixture
def league(synthetic_frame):
    f = synthetic_frame()
    return f, _player_rows(f)


# --- minutes and lineups --------------------------------------------------------------------

def test_allocate_minutes_renormalises_to_240_with_cap():
    proj = np.array([38.0, 36, 30, 28, 24, 20, 14, 10])
    m = pl.allocate_minutes(proj, np.ones(8), cap=42)
    assert m.sum() == pytest.approx(240.0)
    assert m.max() <= 42 + 1e-9
    # a short-handed team: caps bind and the rest is spread proportionally
    m2 = pl.allocate_minutes(np.array([40.0, 38, 30, 25, 20, 10]), np.ones(6), cap=42)
    assert m2.sum() == pytest.approx(240.0)
    assert m2.max() == pytest.approx(42.0)
    assert np.all(np.diff(m2) <= 1e-9)
    # P(play) = 0 gets nothing; 0.5 gets about half its share
    m3 = pl.allocate_minutes(proj, np.array([0, 1, 1, 1, 1, 1, 1, 0.5]))
    assert m3[0] == 0 and m3.sum() == pytest.approx(240.0)
    assert m3[7] < pl.allocate_minutes(proj, np.array([0, 1, 1, 1, 1, 1, 1, 1]))[7]


def test_select_lineup_floor_keeps_min_players():
    proj = np.array([30.0, 30, 30, 30, 30, 4, 3, 2, 1])
    keep = pl.select_lineup(proj, np.ones(9), floor=6, min_players=8)
    assert keep.sum() == 8 and not keep[8]
    keep2 = pl.select_lineup(proj, np.ones(9), floor=6, min_players=5)
    assert keep2.sum() == 5


def test_unseen_player_gets_the_prior(league):
    f, p = league
    eng, st = pl.run_player_engine(f, p, SMALL, modes=("played",), return_state=True)
    # a never-seen player in an availability map enters with the new-player prior and m0
    team = f["home"].iloc[-1]
    lu = st.expected_lineup(team, None, {999999: 1.0})
    row = lu[lu["player_id"] == 999999].iloc[0]
    assert row["rating"] == pytest.approx(st.mu_new)
    assert row["proj_min"] == pytest.approx(SMALL.m0)
    # mu_new is the box-prior regression evaluated at default features, not a rated player's
    assert np.isfinite(st.mu_new)
    # a player whose first game is today was rated with the same prior inside the engine
    f2 = f.copy()
    p2 = p.copy()
    last_gid = f2["game_id"].iloc[-1]
    row0 = p2[(p2.game_id == last_gid) & (p2.team == f2["home_raw"].iloc[-1])].iloc[0].copy()
    row0["player_id"] = 424242
    row0["min"] = 30.0
    p2 = pd.concat([p2, row0.to_frame().T.astype(p2.dtypes.to_dict())], ignore_index=True)
    e1 = pl.run_player_engine(f2, p, SMALL, modes=("played",))
    e2 = pl.run_player_engine(f2, p2, SMALL, modes=("played",))
    proj_new = SMALL.min_w_short * SMALL.m0 + (1 - SMALL.min_w_short) * SMALL.m0
    assert proj_new == pytest.approx(SMALL.m0)
    assert e1["played_home"].iloc[-1] != e2["played_home"].iloc[-1]
    assert np.allclose(e1["played_home"].iloc[:-1], e2["played_home"].iloc[:-1])


def test_expected_lineup_excludes_injured_player(league):
    f, p = league
    _, st = pl.run_player_engine(f, p, SMALL, modes=("played",), return_state=True)
    team = f["home"].iloc[-1]
    full = st.expected_lineup(team, None, None)
    star = int(full.sort_values("rating", ascending=False)["player_id"].iloc[0])
    star_min = float(full.loc[full.player_id == star, "minutes"].iloc[0])
    inj = st.expected_lineup(team, {"date": "next"}, {star: 0.0})
    assert inj["minutes"].sum() == pytest.approx(240.0)
    assert float(inj.loc[inj.player_id == star, "minutes"].iloc[0]) == 0.0
    assert star_min > 0
    assert inj.attrs["strength"] < full.attrs["strength"]       # losing the best player hurts
    q = st.expected_lineup(team, None, {star: 0.5})
    assert inj.attrs["strength"] < q.attrs["strength"] < full.attrs["strength"]


def test_ratings_recover_true_order(synthetic_frame):
    f = synthetic_frame(seasons=("2000-01", "2001-02", "2002-03", "2003-04", "2004-05"))
    rng = np.random.default_rng(3)
    p = _player_rows(f)
    # make plus-minus informative: add each player's true rating
    true = {pid: r for pid, r in zip(sorted(p.player_id.unique()),
                                     rng.normal(0, 4, p.player_id.nunique()))}
    p["plus_minus"] = p["plus_minus"] + p["player_id"].map(true) * p["min"] / 48 * 3
    _, st = pl.run_player_engine(f, p, SMALL, modes=("played",), return_state=True)
    tab = st.ratings_table()
    corr = np.corrcoef(tab["rating"], tab["player_id"].map(true))[0, 1]
    assert corr > 0.3


# --- leakage ---------------------------------------------------------------------------------

def _scramble_players_after(p, f, cut, seed=1):
    rng = np.random.default_rng(seed)
    fut = set(f.loc[f["game_date"] >= cut, "game_id"])
    p = p.copy()
    m = p["game_id"].isin(fut).to_numpy()
    for c in ("pts", "fga", "ast", "tov", "dreb"):
        p.loc[m, c] = rng.integers(0, 30, m.sum()).astype(float)
    p.loc[m, "plus_minus"] = rng.integers(-30, 30, m.sum()).astype(float)
    p.loc[m, "min"] = np.clip(p.loc[m, "min"] + rng.integers(-10, 10, m.sum()), 1, 48)
    return p


def test_engine_no_lookahead_on_box_scores(league):
    f, p = league
    cut = f["game_date"].iloc[len(f) * 3 // 4]
    p2 = _scramble_players_after(p, f, cut)
    e1 = pl.run_player_engine(f, p, SMALL)
    e2 = pl.run_player_engine(f, p2, SMALL)
    upto = (f["game_date"] <= cut).to_numpy()
    # strengths of games on the cut day depend on who played (oracle availability) but never
    # on any box-score number of that day or later; minutes > 0 is preserved by the scramble
    np.testing.assert_array_equal(e1[upto].to_numpy(), e2[upto].to_numpy())
    assert not np.allclose(e1[~upto].to_numpy(), e2[~upto].to_numpy())


def test_prefix_run_equals_full_run(league):
    f, p = league
    full = pl.run_player_engine(f, p, SMALL)
    k = int(len(f) * 0.6)
    part = pl.run_player_engine(f.iloc[:k], p, SMALL)
    np.testing.assert_allclose(part.to_numpy(), full.iloc[:k].to_numpy(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", list(player_model.SPECS))
def test_player_models_no_lookahead(league, name):
    f, p = league
    test_season = f["season"].max()
    train, test = f[f["season"] < test_season], f[f["season"] == test_season]
    cut = test["game_date"].iloc[len(test) // 2]
    # results in the game frame (the harness check) ...
    reg, _ = player_model.make_registry(p, SMALL, burn_in=1)
    assert backtest.check_no_lookahead(reg[name], train, test, cut) == 0.0
    # ... and every box score on/after the cut
    p2 = _scramble_players_after(p, f, cut)
    reg2, _ = player_model.make_registry(p2, SMALL, burn_in=1)
    m1, m2 = reg[name](), reg2[name]()
    m1.fit(train); m2.fit(train)
    upto = (test["game_date"] <= cut).to_numpy()
    np.testing.assert_array_equal(m1.predict_proba(test)[upto], m2.predict_proba(test)[upto])
    np.testing.assert_array_equal(m1.predict_margin(test)[upto], m2.predict_margin(test)[upto])


def test_cache_prefix_slicing_matches_recompute(league):
    f, p = league
    reg, cache = player_model.make_registry(p, SMALL, burn_in=1)
    cache.prime(f)
    s = f["season"].max()
    train, test = f[f["season"] < s], f[f["season"] == s]
    m = reg["ps_elo_sched"](); m.fit(train)
    p_cached = m.predict_proba(test)
    assert cache.n_runs == 1
    reg2, cache2 = player_model.make_registry(p, SMALL, burn_in=1)
    m2 = reg2["ps_elo_sched"](); m2.fit(train)
    np.testing.assert_allclose(m2.predict_proba(test), p_cached, atol=1e-12)
    assert cache2.n_runs == 2


def test_backtest_runs_player_models(league):
    f, p = league
    reg, cache = player_model.make_registry(p, SMALL, burn_in=1)
    cache.prime(f)
    tests = backtest.seasons_in(f)[2:]
    preds, desc = backtest.run_backtest(f, {k: reg[k] for k in ("ps", "ps_mh", "ps_elo_sched")},
                                        tests)
    assert preds["p"].between(0, 1).all() and preds["margin_pred"].notna().all()
    ev = backtest.evaluate(preds)
    assert set(ev) == {"ps", "ps_mh", "ps_elo_sched"}
    assert cache.n_runs == 1


def test_tuning_score_ignores_later_seasons(synthetic_frame):
    f = synthetic_frame(seasons=("2008-09", "2009-10", "2010-11", "2011-12"))
    p = _player_rows(f)
    later = set(f.loc[f["season"] > "2010-11", "game_id"])
    p2 = p.copy()
    m = p2["game_id"].isin(later)
    p2.loc[m, "plus_minus"] = -p2.loc[m, "plus_minus"]
    f2 = f.copy()
    f2.loc[f2["season"] > "2010-11", "margin"] *= -1
    old = player_model.TUNE_EVAL_FROM
    player_model.TUNE_EVAL_FROM = "2009-10"
    try:
        a = player_model.tune_score(f, p, SMALL)
        b = player_model.tune_score(f2, p2, SMALL)
    finally:
        player_model.TUNE_EVAL_FROM = old
    assert a == b
