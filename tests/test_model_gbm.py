"""Rung 3 (stacked GBM): synthetic data, no network."""

import numpy as np
import pandas as pd
import pytest

from courtside.model import backtest, features as ft, gbm, players as pl, rung3
from test_model_players import _player_rows, _scramble_players_after

SMALL = pl.PlayerParams(active_days=400, roster_games=5)
TINY_GRID = (gbm.GBMConfig(max_iter=40, min_data_in_leaf=20, num_leaves=4),
             gbm.GBMConfig(max_iter=40, min_data_in_leaf=20, num_leaves=4, half_life=2.0))
SEASONS = ("2000-01", "2001-02", "2002-03", "2003-04", "2004-05")


def _with_box_and_market(f: pd.DataFrame, seed: int = 0, p_line: float = 0.7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = f.copy()
    n = len(f)
    for s in ("h", "a"):
        f[f"{s}_poss"] = rng.normal(98, 4, n)
        f[f"{s}_ortg"] = rng.normal(110, 8, n)
        f[f"{s}_drtg"] = rng.normal(110, 8, n)
        f[f"{s}_efg"] = rng.normal(0.52, 0.05, n)
        f[f"{s}_tov"] = rng.normal(0.13, 0.02, n)
        f[f"{s}_orb"] = rng.normal(0.25, 0.05, n)
        f[f"{s}_ftr"] = rng.normal(0.25, 0.05, n)
    has = rng.random(n) < p_line
    f["mkt_margin"] = np.where(has, 0.4 * f["margin"] + rng.normal(0, 6, n), np.nan)
    f["mkt_total"] = np.where(has, 205 + rng.normal(0, 6, n), np.nan)
    return f


@pytest.fixture
def league(synthetic_frame):
    f = _with_box_and_market(synthetic_frame(seasons=SEASONS))
    return f, _player_rows(f)


def _registry(p, **kw):
    kw = {"params": SMALL, "frozen": False, "burn_in": 1, "n_inner": 2, "grid": TINY_GRID,
          "disk_dir": None, **kw}
    disk = kw.pop("disk_dir")
    params = kw.pop("params")
    _, fc = gbm.make_caches(p, params=params, disk_dir=disk)
    fc.engine.disk_dir = None
    reg, fc = gbm.make_registry(p, params=params, fcache=fc, **kw)
    return reg, fc


def _split(f):
    s = f["season"].max()
    return f[f["season"] < s], f[f["season"] == s]


def _scramble_frame_after(f, cut, seed=2):
    """Every outcome column of games on/after `cut`: score, margin, win and the box columns."""
    rng = np.random.default_rng(seed)
    g = f.copy()
    m = (g["game_date"] >= cut).to_numpy()
    k = int(m.sum())
    hp = rng.integers(70, 140, k)
    ap = rng.integers(70, 140, k)
    ap = np.where(ap == hp, ap + 1, ap)
    g.loc[m, "home_pts"], g.loc[m, "away_pts"] = hp, ap
    g.loc[m, "margin"] = hp - ap
    g.loc[m, "home_win"] = (hp > ap).astype(int)
    for c in ft.BOX_OUTCOME_COLUMNS:
        g.loc[m, c] = rng.normal(100, 30, k)
    return g


# --- leakage ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [gbm.PRODUCTION_WINPROB, gbm.PRODUCTION_OPINION])
def test_gbm_no_lookahead(league, name):
    f, p = league
    train, test = _split(f)
    cut = test["game_date"].iloc[len(test) // 2]
    reg, _ = _registry(p)
    # harness check: scores/margins/wins of every test game on/after the cut
    assert backtest.check_no_lookahead(reg[name], train, test, cut) == 0.0
    # stronger: box columns and player box scores too, fresh caches on both sides
    test2 = _scramble_frame_after(test, cut)
    p2 = _scramble_players_after(p, f, cut)
    reg2, _ = _registry(p2)
    m1, m2 = reg[name](), reg2[name]()
    m1.fit(train)
    m2.fit(train)
    upto = (test["game_date"] <= cut).to_numpy()
    for fn in ("predict_proba", "predict_margin", "predict_total"):
        a, b = getattr(m1, fn)(test), getattr(m2, fn)(test2)
        np.testing.assert_array_equal(a[upto], b[upto])
    assert not np.allclose(m1.predict_margin(test)[~upto], m2.predict_margin(test2)[~upto])


def test_features_are_causal(league):
    f, p = league
    cut = f["game_date"].iloc[len(f) * 2 // 3]
    _, fc1 = _registry(p)
    _, fc2 = _registry(_scramble_players_after(p, f, cut))
    F1 = fc1.get(f)
    F2 = fc2.get(_scramble_frame_after(f, cut))
    upto = (f["game_date"] <= cut).to_numpy()
    np.testing.assert_array_equal(F1[upto].to_numpy(), F2[upto].to_numpy())
    # rolling form and league environment do react to the scrambled results afterwards
    after = ~upto
    assert not np.allclose(F1.loc[after, "net_diff_r10"], F2.loc[after, "net_diff_r10"],
                           equal_nan=True)


def test_feature_cache_prefix_slicing(league):
    f, p = league
    _, fc = _registry(p)
    fc.prime(f)
    k = int(len(f) * 0.6)
    sliced = fc.get(f.iloc[:k])
    assert fc.n_runs == 1
    _, fc2 = _registry(p)
    fresh = fc2.get(f.iloc[:k])
    np.testing.assert_allclose(sliced.to_numpy(), fresh.to_numpy(), equal_nan=True, atol=1e-9)


def test_league_trailing_uses_only_earlier_days():
    day = np.array([0, 0, 1, 1, 2, 3])
    val = np.array([1.0, 3.0, 5.0, 7.0, 100.0, 0.0])
    out = ft._league_trailing(day, val, np.ones(6, bool), 3)
    assert np.isnan(out[0]) and np.isnan(out[1])          # nothing before day 0
    assert out[2] == pytest.approx(2.0) and out[3] == pytest.approx(2.0)
    assert out[4] == pytest.approx((3 + 5 + 7) / 3)       # last 3 games before day 2
    assert out[5] == pytest.approx((5 + 7 + 100) / 3)


# --- market features ------------------------------------------------------------------------

def test_nan_market_features_fall_back(league):
    f, p = league
    train, test = _split(f)
    reg, _ = _registry(p)
    m = reg[gbm.PRODUCTION_WINPROB]()
    m.fit(train)
    base = m.predict_margin(test)
    assert np.isfinite(base).all() and np.isfinite(m.predict_proba(test)).all()
    # no line at all for the test season (columns NaN) and no columns at all
    t_nan = test.assign(mkt_margin=np.nan, mkt_total=np.nan)
    t_none = test.drop(columns=["mkt_margin", "mkt_total"])
    a, b = m.predict_margin(t_nan), m.predict_margin(t_none)
    assert np.isfinite(a).all() and np.isfinite(m.predict_total(t_nan)).all()
    np.testing.assert_allclose(a, b)
    has = test["mkt_margin"].notna().to_numpy()
    assert not np.allclose(a[has], base[has])               # the line is used when present
    # the no-market model ignores the line entirely
    o = reg[gbm.PRODUCTION_OPINION]()
    o.fit(train)
    np.testing.assert_array_equal(o.predict_margin(test), o.predict_margin(t_nan))


def test_market_model_trains_with_no_lines_at_all(league):
    f, p = league
    f = f.assign(mkt_margin=np.nan, mkt_total=np.nan)
    train, test = _split(f)
    reg, _ = _registry(p)
    m = reg[gbm.PRODUCTION_WINPROB]()
    m.fit(train)
    assert np.isfinite(m.predict_proba(test)).all()


# --- explain ---------------------------------------------------------------------------------

def test_explain_shape_and_determinism(league):
    f, p = league
    train, test = _split(f)
    out = []
    for _ in range(2):
        reg, _ = _registry(p)
        m = reg[gbm.PRODUCTION_OPINION]()
        m.fit(train)
        out.append(m.explain(test.iloc[7], test, top=4))
    r = out[0]
    assert r == out[1]
    assert 1 <= len(r) <= 4
    for x in r:
        assert set(x) == {"factor", "team", "impact_pts", "home_margin_pts"}
        assert x["team"] in (test["home"].iloc[7], test["away"].iloc[7])
        assert isinstance(x["impact_pts"], float)
    mags = [abs(x["home_margin_pts"]) for x in r]
    assert mags == sorted(mags, reverse=True)
    # contributions are an exact decomposition of the predicted margin
    C = m.contributions(test)
    np.testing.assert_allclose(C.sum(axis=1).to_numpy(), m.predict_margin(test), atol=1e-8)
    # by game_id gives the same answer
    assert m.explain(test["game_id"].iloc[7], test, top=4) == r


def test_reasons_sign_convention():
    c = pd.Series({"abs_loss_home": -3.1, "abs_loss_away": 1.5, "ps_diff": -2.0,
                   "home_court": 2.4, "_bias": 1.0})
    r = gbm.reasons_from_contrib(c, "LAL", "BOS", top=10)
    by = {(x["factor"], x["team"]): x["impact_pts"] for x in r}
    assert by[("absences", "LAL")] == pytest.approx(-3.1)    # LAL's absences cost LAL 3.1
    assert by[("absences", "BOS")] == pytest.approx(-1.5)    # BOS's absences help LAL
    assert by[("player_strength", "BOS")] == pytest.approx(2.0)
    assert by[("home_court", "LAL")] == pytest.approx(2.4)


# --- calibration -----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", gbm.CALIBRATORS)
def test_calibration_is_monotone(kind):
    rng = np.random.default_rng(0)
    mu = rng.normal(2, 9, 4000)
    post = (rng.random(4000) < 0.1).astype(float)
    y = (mu + rng.normal(0, 12, 4000) > 0).astype(int)
    c = gbm.Calibrator(kind).fit(mu, y, post)
    grid = np.linspace(-30, 30, 301)
    for po in (0.0, 1.0):
        pr = c.predict(grid, np.full_like(grid, po))
        assert np.all(np.diff(pr) >= -1e-12)
        assert pr[0] < 0.2 and pr[-1] > 0.8
    cal, scores = gbm.choose_calibrator(mu, y, post, np.repeat([1, 2, 3, 4], 1000))
    assert cal.kind in gbm.CALIBRATORS and set(scores) == set(gbm.CALIBRATORS)


def test_fitted_model_probability_is_monotone_in_margin(league):
    f, p = league
    train, test = _split(f)
    reg, _ = _registry(p)
    m = reg[gbm.PRODUCTION_OPINION]()
    m.fit(train)
    rs = (test["season_type"] == "Regular Season").to_numpy()
    mu, pr = m.predict_margin(test)[rs], m.predict_proba(test)[rs]
    o = np.argsort(mu, kind="stable")
    assert np.all(np.diff(pr[o]) >= -1e-12)


# --- engine and reporting helpers ---------------------------------------------------------

def test_tight_modes_only_change_postseason(league):
    f, p = league
    e = pl.run_player_engine(f, p, SMALL, modes=("played", "played_tight", "avail",
                                                  "avail_tight", "roster", "roster_tight"))
    rs = (f["season_type"] == "Regular Season").to_numpy()
    np.testing.assert_array_equal(e.loc[rs, "roster_home"], e.loc[rs, "roster_tight_home"])
    np.testing.assert_array_equal(e.loc[rs, "played_home"], e.loc[rs, "played_tight_home"])
    np.testing.assert_array_equal(e.loc[rs, "avail_home"], e.loc[rs, "avail_tight_home"])


def test_avail_ignores_fringe_participation(league):
    """The leak fix: whether a fringe player (projection < REGULAR_MIN) logged minutes in a
    game (DNP-CD vs garbage time, which depends on the game's flow) must not move that game's
    'avail' strength; it does move the full-oracle 'played' strength."""
    f, p = league
    g = f.iloc[-1]
    rows = p[(p["game_id"] == g["game_id"]) & (p["team"] == g["home_raw"])]
    _, st = pl.run_player_engine(f.iloc[:-1], p, SMALL, modes=("played",), return_state=True)
    proj = {int(pid): st.proj_min[st.index_of(pid)] for pid in rows["player_id"]}
    fringe = [pid for pid, m in proj.items() if m < pl.REGULAR_MIN]
    regular = [pid for pid, m in proj.items() if m >= max(pl.REGULAR_MIN, SMALL.floor)]
    assert fringe and regular
    drop = p.index[(p["game_id"] == g["game_id"]) & p["player_id"].isin(fringe[:1])]
    e1 = pl.run_player_engine(f, p, SMALL, modes=("played", "avail"))
    e2 = pl.run_player_engine(f, p.drop(drop), SMALL, modes=("played", "avail"))
    assert e1["avail_home"].iloc[-1] == e2["avail_home"].iloc[-1]
    # a regular's absence is (pre-game, injury-report) information and does count
    drop_r = p.index[(p["game_id"] == g["game_id"]) & p["player_id"].isin(regular[:1])]
    e3 = pl.run_player_engine(f, p.drop(drop_r), SMALL, modes=("avail",))
    assert e3["avail_home"].iloc[-1] != e1["avail_home"].iloc[-1]


def test_record_and_wilson():
    pred = np.array([5.0, 5.0, -5.0, 1.0, 0.0])
    line = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    act = np.array([3.0, -2.0, -1.0, 0.0, 4.0])   # win, loss, win, push, no pick
    r = rung3.record(pred, line, act, edges=(0.0, 2.0))
    assert r[0]["n"] == 3 and r[0]["wins"] == 2
    assert r[1]["n"] == 3
    lo, hi = rung3.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)


def test_with_injuries_lowers_the_injured_teams_lineup(league):
    import sqlite3
    f, p = league
    train, test = _split(f)
    nxt = test.iloc[[-1]].copy()
    nxt["game_id"] = "0029999999"
    nxt["game_date"] = test["game_date"].max() + pd.Timedelta(days=2)
    for c in ("home_pts", "away_pts", "margin", "home_win"):
        nxt[c] = 0
    reg, _ = _registry(p)
    m = reg[gbm.PRODUCTION_OPINION]()
    m.prepare(f, nxt)
    m.fit(f)
    team = nxt["home"].iloc[0]
    st = m.fc.engine.state(pd.concat([f, nxt]))
    star = st.expected_lineup(team).sort_values("contribution", ascending=False).iloc[0]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE injury_snapshots (captured_at TEXT, source TEXT, team TEXT, "
                 "player_name TEXT, status TEXT)")
    conn.execute("INSERT INTO injury_snapshots VALUES ('2099-01-01T00:00:00Z', 'nba_official', ?, "
                 "?, 'Out')", (team, star["name"]))
    out = m.with_injuries(conn, nxt, "2099-12-31T00:00:00Z")
    assert np.isnan(out["lineup_away"].iloc[0])           # no snapshot for the away team
    full = st.expected_lineup(team).attrs["strength"]
    assert out["lineup_home"].iloc[0] < full
    assert m.predict_margin(out)[0] < m.predict_margin(nxt)[0]
    reasons = m.explain(out.iloc[0], out, top=10)
    ab = [r for r in reasons if r["factor"] == "absences" and r["team"] == team]
    assert ab and ab[0]["impact_pts"] < 0
    # absences are measured from a healthy roster: with nobody out the linear part is 0
    base = m.heads_["margin"][0].base
    b_ps = base.coef_fb[base.cols_fb.index("ps_diff")]
    abs_away = m._predict_all(nxt)["F"]["abs_loss_away"].iloc[0]
    assert m.contributions(nxt)["abs_loss_away_lin"].iloc[0] == pytest.approx(b_ps * abs_away)


def test_frozen_registry_skips_the_search(league):
    f, p = league
    train, test = _split(f)
    reg, _ = _registry(p, frozen=True)
    m = reg[gbm.PRODUCTION_OPINION]()
    m.fit(train)
    for t, (cfg, k) in gbm.FROZEN[gbm.PRODUCTION_OPINION].items():
        assert m.chosen()[t][1] <= k and m.chosen()[t][0].num_leaves == cfg.num_leaves
    assert np.isfinite(m.predict_total(test)).all()


# --- season-start totals ----------------------------------------------------------------------

def _paced_league(synthetic_frame, seed=3):
    """Synthetic league whose teams differ in pace and offence (persistent across seasons), so a
    game's total depends on who plays. Points and box columns follow from the team traits."""
    codes = ("BOS", "NYK", "LAL", "GSW", "DEN", "MIA", "CHI", "PHX", "SEA", "UTA", "ATL", "POR")
    f = _with_box_and_market(synthetic_frame(seasons=SEASONS, codes=codes), seed=seed,
                             p_line=0.0)
    rng = np.random.default_rng(seed)
    teams = sorted(set(f["home"]) | set(f["away"]))
    pace = dict(zip(teams, np.linspace(-10, 10, len(teams)) + rng.normal(0, 0.5, len(teams))))
    off = dict(zip(teams, rng.permutation(np.linspace(-9, 9, len(teams)))))
    n = len(f)
    h, a = f["home"].to_numpy(), f["away"].to_numpy()
    poss = 98 + (np.vectorize(pace.get)(h) + np.vectorize(pace.get)(a)) / 2 + rng.normal(0, 1.5, n)
    ho = 112 + np.vectorize(off.get)(h) + rng.normal(0, 3, n)
    ao = 112 + np.vectorize(off.get)(a) + rng.normal(0, 3, n)
    f["h_poss"] = f["a_poss"] = poss
    f["h_ortg"], f["a_ortg"], f["h_drtg"], f["a_drtg"] = ho, ao, ao, ho
    f["home_pts"], f["away_pts"] = np.round(poss * ho / 100), np.round(poss * ao / 100)
    f["margin"] = f["home_pts"] - f["away_pts"]
    f["home_win"] = (f["margin"] > 0).astype(int)
    expect = 98 + (np.vectorize(pace.get)(h) + np.vectorize(pace.get)(a)) / 2
    f["exp_total"] = expect * (224 + np.vectorize(off.get)(h) + np.vectorize(off.get)(a)) / 100
    return f


def _season_opener(f, days=1):
    s = f["season"].max()
    cur = f[f["season"] == s]
    d0 = cur["game_date"].min()
    first = cur[cur["game_date"] < d0 + pd.Timedelta(days=days)]      # opening day: gp == 0
    return f[f["game_date"] < d0], first


def test_season_start_rolling_totals_use_last_seasons_prior(synthetic_frame):
    f = _paced_league(synthetic_frame)
    _, fc = _registry(_player_rows(f))
    F = fc.get(f)
    s = f["season"].max()
    first = (F["gp_home"] == 0) & (F["gp_away"] == 0) & (f["season"] == s).to_numpy()
    assert first.any()
    for c in ("tot_r10_home", "pace_r10_sum", "ortg_r10_home", "drtg_r10_away"):
        assert F.loc[first, c].notna().all()           # was NaN before the prior carry-over
    assert F.loc[first, "tot_r10_home"].std() > 1.0    # priors differ by team
    # the very first season of the frame has no last season: still unknown there
    s0 = (f["season"] == f["season"].min()).to_numpy() & (F["gp_home"] == 0).to_numpy()
    assert F.loc[s0, "tot_r10_home"].isna().all()


@pytest.mark.parametrize("name", [gbm.PRODUCTION_OPINION, gbm.PRODUCTION_WINPROB])
def test_season_opener_totals_vary_with_team_pace(synthetic_frame, name):
    f = _paced_league(synthetic_frame)
    train, opener = _season_opener(f)
    reg, _ = _registry(_player_rows(f))
    m = reg[name]()
    m.fit(train)
    t = m.predict_total(opener.assign(mkt_total=np.nan, mkt_margin=np.nan))
    assert np.isfinite(t).all() and len(t) >= 6
    assert t.std() > 2.0, f"opening-day totals collapsed to the mean (SD {t.std():.2f})"
    exp = f.loc[opener.index, "exp_total"].to_numpy()
    assert np.corrcoef(t, exp)[0, 1] > 0.5


def test_season_opener_total_tracks_the_market_line(synthetic_frame):
    f = _paced_league(synthetic_frame)
    train, opener = _season_opener(f)
    rng = np.random.default_rng(7)
    # a line that moves with the teams but is not a function of our features
    line = f.loc[train.index, "exp_total"] + rng.normal(0, 2, len(train))
    train = train.assign(mkt_total=line.to_numpy(), mkt_margin=0.3 * train["margin"].to_numpy())
    mkt = opener["exp_total"].to_numpy() + rng.normal(0, 4, len(opener))
    reg, _ = _registry(_player_rows(f))
    m = reg[gbm.PRODUCTION_WINPROB]()
    m.fit(train)
    t = m.predict_total(opener.assign(mkt_total=mkt, mkt_margin=0.0))
    assert t.std() > 2.0
    assert np.corrcoef(t, mkt)[0, 1] > 0.8
    assert np.abs(t - mkt).mean() < 4.0
