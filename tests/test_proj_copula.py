"""Path A v0 copula pieces (T1.4): reconciliation, copula recovery, marginals, scenarios,
simulate_game invariants on a synthetic league."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import reconcile as RC
from courtside.proj import state as ST

from test_proj_minutes import _league


# ------------------------------------------------------------------------- reconciliation

def test_reconcile_hits_target_and_flags_gap():
    rng = np.random.default_rng(0)
    raw = rng.gamma(2.0, 3.0, size=(4000, 12)) * (rng.random((4000, 12)) > 0.1)
    oth = raw * 0.9
    g = 1.0 + 0.1 * rng.standard_normal(4000)
    target = raw.sum(1).mean() + 3.0
    r = RC.reconcile_team(raw, oth, g, 0.5, target)
    assert abs(r["pts"].sum(1).mean() - target) <= RC.TOL and r["ok"]
    assert abs(r["resid"]) < 1e-6
    assert not r["data_check"] and r["gap"] == pytest.approx(-3.0)
    # pace part: other stats scaled by c^alpha g^alpha
    assert np.allclose(r["oth"], oth * (g ** 0.5)[:, None] * r["scale"] ** 0.5)
    assert np.allclose(r["pace"], (g * r["scale"]) ** 0.5)
    off = RC.reconcile_team(raw, oth, g, 0.5, target, weight=0.0)
    assert off["scale"] == pytest.approx(1.0) and np.allclose(off["pts"], raw)
    far = RC.reconcile_team(raw, oth, g, 0.5, raw.sum(1).mean() - 9.0)
    assert far["data_check"] and far["ok"]
    rows = RC.check_rows([dict(far, team="AAA")], "run1")
    assert rows[0]["status"] == "warn" and rows[0]["value"] == pytest.approx(9.0, abs=1e-6)
    # capped scale fails acceptance and is reported
    capped = RC.reconcile_team(raw, oth, g, 0.5, raw.sum(1).mean() * 2)
    assert capped["scale"] == RC.SCALE_BOUNDS[1] and not capped["ok"]


def test_implied_points_prefers_team_totals():
    assert RC.implied_points(4.0, 220.0) == (112.0, 108.0)
    assert RC.implied_points(4.0, 220.0, 113.5, 107.0) == (113.5, 107.0)
    assert RC.implied_points(4.0, 220.0, None, 107.0) == (112.0, 108.0)


# ------------------------------------------------------------------------- copula

def _cp(**kw):
    cp = C.default_copula_params()
    cp.rho_team = {C.pair_key(a, b): 0.0 for a in C.TIERS for b in C.TIERS}
    cp.rho_team.update(kw.get("rho_team", {}))
    cp.rho_opp = kw.get("rho_opp", 0.0)
    return cp


def test_copula_correlation_recovered():
    team = np.array(["H"] * 6 + ["A"] * 6)
    tiers = np.array([0, 0, 1, 1, 2, 2] * 2)
    cp = _cp(rho_team={"star|star": 0.3, "star|role": -0.1, "bench|bench": 0.2}, rho_opp=0.05)
    R = C.corr_matrix(team, tiers, cp)
    assert np.allclose(np.diag(R), 1.0) and np.allclose(R, R.T)
    assert R[0, 1] == pytest.approx(0.3) and R[0, 2] == pytest.approx(-0.1)
    assert R[4, 5] == pytest.approx(0.2) and R[0, 6] == pytest.approx(0.05)
    z = C.draw_copula(R, 200_000, np.random.default_rng(1))
    emp = np.corrcoef(z.T)
    assert np.abs(emp - R).max() < 0.012
    # through the marginals (normal stars, gamma bench) the FP correlation keeps sign/size
    mu = np.array([45, 40, 28, 26, 12, 9] * 2, float)
    normal = mu >= 25
    fp = C.marginal_fp(mu[None, :], z, C.cv_of(cp, mu)[None, :], normal[None, :])
    c = np.corrcoef(fp.T)
    assert c[0, 1] == pytest.approx(0.3, abs=0.015)
    assert c[4, 5] == pytest.approx(0.2, abs=0.03)
    assert c[0, 6] == pytest.approx(0.05, abs=0.015)


def test_nearest_corr_makes_psd():
    R = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1.0]])
    P = C.nearest_corr(R)
    assert np.linalg.eigvalsh(P).min() > 0 and np.allclose(np.diag(P), 1.0)
    np.linalg.cholesky(P)


def test_marginals_mean_and_sd():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((400_000, 1))
    for mu, cv, normal in ((8.0, 0.7, False), (18.0, 0.45, False), (40.0, 0.28, True)):
        x = C.marginal_fp(np.full((1, 1), mu), z, np.full((1, 1), cv), np.array([[normal]]))
        assert x.mean() == pytest.approx(mu, rel=0.01)
        assert x.std() == pytest.approx(cv * mu, rel=0.06)
        if not normal:
            assert x.min() >= 0


def test_tier_pair_corr_recovers_block_correlation():
    rng = np.random.default_rng(5)
    rows = []
    rho_tm, rho_op = 0.15, 0.05
    for g in range(4000):
        shared = rng.standard_normal()
        for t in ("A", "B"):
            team = rng.standard_normal()
            for i in range(8):
                z = (np.sqrt(rho_op) * shared + np.sqrt(rho_tm - rho_op) * team
                     + np.sqrt(1 - rho_tm) * rng.standard_normal())
                rows.append((f"g{g}", t, i % 3, z))
    df = pd.DataFrame(rows, columns=["game_id", "team", "tier", "z"])
    rho, opp, n = C.tier_pair_corr(df)
    assert opp == pytest.approx(rho_op, abs=0.01)
    for k, v in rho.items():
        assert v == pytest.approx(rho_tm, abs=0.02), k
    assert n["opp"] == 4000 * 64


# ------------------------------------------------------------------------- scenarios

def test_scenarios_exact_marginals_cap_and_certain_outs():
    p = np.array([1.0, 1.0, 0.0, 0.5, 0.7, 0.9, 0.3, 0.6, 0.8, 0.4, 0.95, 0.85])
    sc = C.draw_scenarios(p, 20_000, np.random.default_rng(0))
    ab = sc["absent"]
    assert ab[:, 2].all() and not ab[:, :2].any()
    assert np.allclose(1 - ab.mean(0), p, atol=0.012)
    assert len(sc["kept"]) <= C.MAX_SCENARIOS
    # every sim's kept set is a subset of its drawn absent set
    for s in range(0, 20_000, 97):
        kept = sc["kept"][sc["assign"][s]]
        assert not (kept & ~ab[s]).any()
    assert sc["collapsed"] > 0
    one = C.draw_scenarios(np.array([1.0, 0.0, 1.0]), 10, np.random.default_rng(0))
    assert len(one["kept"]) == 1 and one["absent"][:, 1].all()


# ------------------------------------------------------------------------- simulate_game

@pytest.fixture(scope="module")
def game():
    pg, tg, pos = _league(seed=1)
    h = M.build_history(pg, tg, pos)
    last = h.tg[h.tg["season"] == "2022-23"].sort_values("date")
    g = last.iloc[-1]
    day = int(g["date"])
    opp = h.tg[(h.tg["game_id"] == g["game_id"]) & (h.tg["team"] != g["team"])].iloc[0]["team"]
    teams = {}
    sp = ST.default_params()
    for side, team in (("home", g["team"]), ("away", opp)):
        t = pd.DataFrame({"key": [team], "team": [team], "date": [day], "season": ["2022-23"]})
        panel = M.pregame_panel(h, t)
        n = len(panel)
        rng = np.random.default_rng(len(team))
        pl = pd.DataFrame({"player_id": panel["player_id"].to_numpy(np.int64),
                           "p_avail": 1.0})
        for s, r in ST.DK_WEIGHTS.items():
            pl[f"r_{s}"] = ST.DEFAULT_RATE[s] * rng.uniform(0.6, 1.6, n)
        pl["mpg"] = 0.5 * panel["ew_f"].to_numpy() + 0.5 * panel["ew_s"].to_numpy()
        pl["tier_state"] = ST.tier_of(pl["mpg"])
        pl["star"] = False
        pl.loc[np.argsort(-pl["mpg"].to_numpy())[:2], "star"] = True
        pl["rotation"] = pl["mpg"] >= 15
        teams[side] = (team, panel, pl)
    return h, teams, sp


def _spec(game, p_out=None, p_half=None, implied=(112.0, 108.0)):
    h, teams, sp = game
    mp = M.default_params()
    specs = {}
    for k, (side, (team, panel, pl)) in enumerate(teams.items()):
        pl = pl.copy()
        if p_out is not None and side == "home":
            pl.loc[p_out, "p_avail"] = 0.0
        if p_half is not None and side == "home":
            pl.loc[p_half, "p_avail"] = 0.5
        specs[side] = C.TeamSpec(team=team, players=pl, minutes=C.ScenarioMinutes(panel, mp),
                                 implied=implied[k])
    return C.GameSpec(game_id="G1", home=specs["home"], away=specs["away"], mu_margin=4.0,
                      mu_total=220.0), mp, sp


def test_simulate_game_invariants(game):
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    a = C.simulate_game(spec, cp, mp, sp, 3000, 11)
    b = C.simulate_game(spec, cp, mp, sp, 3000, 11)
    c = C.simulate_game(spec, cp, mp, sp, 3000, 12)
    assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    assert not np.array_equal(a.fp, c.fp)
    home = a.info["team"].to_numpy() == spec.home.team
    hp = spec.home.players["player_id"].to_numpy()
    j0 = int(np.flatnonzero(a.player_id == hp[0])[0])
    assert (a.fp[:, j0] == 0).all() and (a.minutes[:, j0] == 0).all()
    # not playing <=> 0 FP; the P(available)=.5 players sit about half the sims
    assert (a.fp[a.minutes == 0] == 0).all()
    j1 = int(np.flatnonzero(a.player_id == hp[1])[0])
    assert 0.4 < (a.minutes[:, j1] > 0).mean() < 0.55
    # reconciliation: sum of expected points per team = implied
    for team, info in a.teams.items():
        assert abs(info["resid"]) <= RC.TOL
    assert a.teams[spec.home.team]["n_projected"] >= 4       # 2 uncertain players -> 4 sets
    pts_home = a.info.loc[home, "pts_all"].sum()
    assert pts_home == pytest.approx(112.0, abs=0.5)
    # team minutes per sim = 240 (+25 per OT) - reserve (reserve <= 30 here)
    tot = a.minutes[:, home].sum(1)
    ot = np.abs(a.margin) < C.OT_BAND
    assert tot[~ot].max() <= 240 + 1e-3 and np.median(tot[~ot]) >= 235   # 40-min cap can bind
    assert tot[ot].max() <= 265 + 1e-3 and np.median(tot[ot]) >= 260
    # cond all = P(play)-weighted mixture with 0
    for j in range(a.fp.shape[1]):
        pl_ = a.minutes[:, j] > 0
        if pl_.any():
            assert a.fp[:, j].mean() == pytest.approx(pl_.mean() * a.fp[pl_, j].mean(), rel=1e-4,
                                                      abs=1e-6)


def test_blowouts_cost_starters_minutes(game):
    """Shared margin draw: the top starter's minutes fall with |margin| (fitted-like g)."""
    spec, mp, sp = _spec(game)
    nt, nb = len(M.TIER_EDGES) + 1, len(M.MARGIN_BINS) - 1
    tab = np.zeros((nt, 2, nb))
    tab[3:, :, 4:] = -4.0
    mp.g_table = tab.tolist()
    cp = C.default_copula_params()
    s = C.simulate_game(spec, cp, mp, sp, 6000, 3)
    j = int(np.argmax(s.minutes.mean(0)))
    blow = np.abs(s.margin) >= 22
    assert s.minutes[blow, j].mean() < s.minutes[~blow, j].mean() - 1.0
