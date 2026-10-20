"""Path A copula v1 (T2.10): count tables, exact DK scoring, component sims (CRN with v0,
reconciliation, DD/TD jumps, correlations), the v1 fits, the pipeline switch and versioning,
per-stat sims storage and the props reader."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from courtside.proj import copula as C
from courtside.proj import pipeline as P
from courtside.proj import props as PR
from courtside.proj import reconcile as RC
from courtside.proj import state as ST
from courtside.proj import store as SO
from courtside.proj.dfs import rules as R

from test_proj_copula import _spec, game  # noqa: F401  (module fixture)


# ------------------------------------------------------------------------- tables, scoring

def test_count_tables_exact_mean_and_nb_variance():
    v1 = C.default_copula_v1_params()
    T = C.count_tables(v1)
    assert C.count_tables(v1) is T                               # cached per dispersion
    rng = np.random.default_rng(0)
    n = 300_000
    for s, m in (("pts", 3.0), ("pts", 21.7), ("reb", 7.4), ("blk", 0.6), ("tov", 0.0)):
        y = T.draw(s, np.full(n, m), rng.random(n), rng.random(n)).astype(float)
        c2 = np.interp(m, v1.disp_mu[s], v1.disp_c2[s])
        assert y.mean() == pytest.approx(m, abs=0.02 + 0.006 * m)
        assert y.var() == pytest.approx(m + c2 * m * m, rel=0.03, abs=1e-9)
        assert y.dtype == float and y.min() >= 0
    # monotone in u (a copula needs a non-decreasing quantile function)
    u = np.linspace(0, 1, 2001)[:-1]
    y = T.draw("pts", np.full(len(u), 18.0), u, np.zeros(len(u)))
    assert np.all(np.diff(y.astype(int)) >= 0)
    # Hermite-1 coefficient ~ sd for a near-normal count, positive everywhere it matters
    assert T.h("pts", 25.0) == pytest.approx(np.sqrt(25 + 25 ** 2 * np.interp(
        25.0, v1.disp_mu["pts"], v1.disp_c2["pts"])), rel=0.03)
    assert T.h("blk", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_dk_from_components_matches_rules_with_real_bonuses():
    rng = np.random.default_rng(1)
    n = 3000
    st = {s: rng.integers(0, 16, n) for s in C.COMP_STATS}
    fp = C.dk_from_components(st)
    for i in range(0, n, 7):
        row = {s: int(st[s][i]) for s in C.COMP_STATS}
        assert fp[i] == pytest.approx(R.dk_points(row))
    td = {"pts": 12, "reb": 11, "ast": 10, "stl": 0, "blk": 0, "fg3m": 0, "tov": 0}
    one = C.dk_from_components({k: np.array([v]) for k, v in td.items()})
    assert one[0] == pytest.approx(12 + 13.75 + 15 + 4.5)


# ------------------------------------------------------------------------- simulate_game_v1

def test_simulate_v1_invariants_and_crn_with_v0(game):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    v1 = C.default_copula_v1_params(cp=cp)
    a = C.simulate_game_v1(spec, cp, v1, mp, sp, 3000, 11)
    b = C.simulate_game_v1(spec, cp, v1, mp, sp, 3000, 11)
    v0 = C.simulate_game(spec, cp, mp, sp, 3000, 11)
    assert np.array_equal(a.fp, b.fp) and all(np.array_equal(a.stats[s], b.stats[s])
                                              for s in C.COMP_STATS)
    # common random numbers: same margins, totals, availability and minutes as v0
    assert np.array_equal(a.margin, v0.margin) and np.array_equal(a.minutes, v0.minutes)
    assert list(a.info["tier"]) == list(v0.info["tier"])
    played = a.minutes > 0
    # DK is scored exactly from the counts; DD / TD are discrete jumps
    exact = np.where(played, C.dk_from_components(a.stats), 0.0)
    assert np.allclose(a.fp, exact, atol=1e-4)
    base = sum(C.dk_scoring()[s] * a.stats[s].astype(float) for s in C.COMP_STATS)
    jump = np.round(a.fp - base, 4)
    assert set(np.unique(jump[played])) <= {0.0, 1.5, 4.5}
    for s in C.COMP_STATS:
        assert a.stats[s].dtype == np.uint8 and (a.stats[s][~played] == 0).all()
    hp = spec.home.players["player_id"].to_numpy()
    j0 = int(np.flatnonzero(a.player_id == hp[0])[0])
    assert (a.fp[:, j0] == 0).all()
    # reconciliation unchanged; the drawn points still sum to the implied total
    for info in a.teams.values():
        assert abs(info["resid"]) <= RC.TOL
    home = a.info["team"].to_numpy() == spec.home.team
    assert a.info.loc[home, "pts_all"].sum() == pytest.approx(112.0, abs=0.8)
    # same mean DK as v0 up to the bonus approximation and Monte Carlo
    assert a.fp.mean(0).sum() == pytest.approx(v0.fp.mean(0).sum(), rel=0.03)


def test_draw_components_within_and_across_correlation():
    S, n = 60_000, 4
    v1 = C.default_copula_v1_params()
    Rw = np.eye(C.N_COMP)
    i, j = C.COMP_STATS.index("pts"), C.COMP_STATS.index("reb")
    Rw[i, j] = Rw[j, i] = 0.5
    v1.r_within = {t: Rw.tolist() for t in C.TIERS}
    T = C.count_tables(v1)
    lv = {"pts": 20.0, "fg3m": 2.0, "reb": 7.0, "ast": 5.0, "stl": 1.0, "blk": 0.8, "tov": 2.0}
    means = {s: np.full((S, n), v) for s, v in lv.items()}
    played = np.ones((S, n), bool)
    Rp = np.eye(n)
    Rp[0, 1] = Rp[1, 0] = 0.3
    rng = np.random.default_rng(3)
    w = C.draw_copula(Rp, S, rng)
    cnt = C.draw_components(means, played, w, np.zeros(n, int), v1, T,
                            np.random.default_rng(4), np.random.default_rng(5))
    for k in range(n):
        assert np.corrcoef(cnt["pts"][:, k], cnt["reb"][:, k])[0, 1] == pytest.approx(0.47,
                                                                                      abs=0.04)
        assert cnt["pts"][:, k].mean() == pytest.approx(20.0, rel=0.01)
    dk = C.dk_from_components(cnt)
    c = np.corrcoef(dk.T)
    assert c[0, 1] == pytest.approx(0.29, abs=0.03)              # the player matrix, in DK
    assert abs(c[0, 2]) < 0.02 and abs(c[2, 3]) < 0.02


def test_wl_factor_lookup():
    v1 = C.default_copula_v1_params()
    assert np.all(C.wl_factor(v1, "ast", np.array([0, 2]), np.array([-30.0, 0, 30])) == 1.0)
    v1.wl_centres = [-20.0, 0.0, 20.0]
    v1.wl_factor = {s: {t: [0.9, 1.0, 1.1] for t in C.TIERS} for s in C.COMP_STATS}
    f = C.wl_factor(v1, "ast", np.array([0, 2]), np.array([-30.0, -10.0, 0.0, 30.0]))
    assert f.shape == (4, 2) and np.allclose(f[:, 0], [0.9, 0.95, 1.0, 1.1])


# ------------------------------------------------------------------------- fits

def _frame(n=40_000, seed=0, rho=0.4, wl=-0.15):
    """Synthetic v1_frame rows: NB2 counts with a known c2, a pts-fg3m latent link rho and an
    assist shift wl on losing teams (margin < -10)."""
    rng = np.random.default_rng(seed)
    v1 = C.default_copula_v1_params()
    for s in C.COMP_STATS:
        v1.disp_mu[s] = [1.0, 10.0]
        v1.disp_c2[s] = [0.05, 0.05]
    Rw = np.eye(C.N_COMP)
    a, b = C.COMP_STATS.index("pts"), C.COMP_STATS.index("fg3m")
    Rw[a, b] = Rw[b, a] = rho
    T = C.count_tables(v1)
    margin = rng.normal(0, 13, n)
    lv = {"pts": 14.0, "fg3m": 1.6, "reb": 5.0, "ast": 3.5, "stl": 0.9, "blk": 0.6, "tov": 1.5}
    m = {s: v * rng.uniform(0.4, 1.8, n) for s, v in lv.items()}
    z = rng.standard_normal((n, C.N_COMP)) @ np.linalg.cholesky(Rw).T
    from scipy.special import ndtr
    f = pd.DataFrame({"game_id": [f"g{i // 8}" for i in range(n)],
                      "team": ["A" if (i // 4) % 2 == 0 else "B" for i in range(n)],
                      "season": "2021-22", "margin": margin, "g": 1.0,
                      "tier": rng.integers(0, 3, n)})
    for k, s in enumerate(C.COMP_STATS):
        mt = m[s] * np.where((s == "ast") & (margin < -10), 1 + wl, 1.0)
        f[f"y_{s}"] = T.draw(s, mt, ndtr(z[:, k]), rng.random(n)).astype(float)
        f[f"m_{s}"] = m[s]
    f["dk_act"] = C.dk_from_components({s: f[f"y_{s}"].to_numpy() for s in C.COMP_STATS})
    f["mu_dk"] = sum(C.DK_W[s] * f[f"m_{s}"] for s in C.COMP_STATS)
    return f


def test_fit_wl_dispersion_and_within_recover_truth():
    f = _frame()
    cen, fac, _ = C.fit_wl(f)
    lose = [k for k, c in enumerate(cen) if c < -12]
    mid = [k for k, c in enumerate(cen) if abs(c) < 3]
    for t in C.TIERS:
        assert all(fac["ast"][t][k] < 0.93 for k in lose)
        assert all(abs(fac["ast"][t][k] - 1.0) < 0.08 for k in mid)
        assert all(abs(fac["reb"][t][k] - 1.0) < 0.05 for k in lose)
    v1 = C.default_copula_v1_params()
    v1.wl_centres, v1.wl_factor = cen, fac
    f2 = C.apply_wl(f, v1)
    mus, c2s = C.fit_dispersion(f2)
    for s in ("pts", "reb"):
        assert np.median(c2s[s]) == pytest.approx(0.05, abs=0.02)
    v1.disp_mu, v1.disp_c2 = mus, c2s
    Rw, diag = C.fit_within(f2, C.count_tables(v1), iters=3)
    a, b = C.COMP_STATS.index("pts"), C.COMP_STATS.index("fg3m")
    for t in C.TIERS:
        assert Rw[t][a][b] == pytest.approx(0.4, abs=0.07)
        assert abs(Rw[t][C.COMP_STATS.index("reb")][a]) < 0.06
        assert np.linalg.eigvalsh(np.asarray(Rw[t])).min() > 0
        co, cs = np.asarray(diag[t]["corr_obs"]), np.asarray(diag[t]["corr_sim"])
        assert np.abs(co - cs).max() < 0.04


def test_params_json_round_trip_and_defaults():
    v1 = C.default_copula_v1_params("2024-25")
    back = C.CopulaV1Params.from_json(v1.to_json())
    assert back == v1 and back.corr["source"] == "copula-v0"
    R_ = C.corr_matrix_v1(np.array(["H", "H", "A"]), np.array([0, 1, 0]), None, v1)
    assert R_[0, 1] == pytest.approx(v1.corr["rho_team"]["star|role"])
    v1.corr["rho_team_cell"] = {"star:G|role:C": 0.2}
    R_ = C.corr_matrix_v1(np.array(["H", "H"]), np.array([1, 0]), np.array([2, 0]), v1)
    assert R_[0, 1] == pytest.approx(0.2)


# ------------------------------------------------------------------------- pipeline switch

@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA)
    yield
    P.set_copula(state[2])
    P.set_context(state[0], state[1])


def test_versions_compose(restore_switches):
    base = P.MODEL_VERSION_BASE
    assert P._versions(False, False, "v0") == (P._VS_BASE, base)
    assert P._versions(False, False, "v1") == (P._VS_BASE + "_c1", base + "+c1")
    assert P._versions(True, False, "v1")[1] == P.CONTEXT_MODEL_VERSION + "+c1"
    assert P._versions(True, True, "v1") == (P._VS_BASE + "_cx2_c1",
                                             P.CONTEXT_EXT_MODEL_VERSION + "+c1")
    assert P._versions(True, True) == P._versions(True, True, "v0")
    P.set_context(True, True)
    P.set_copula("v1")
    assert P.MODEL_VERSION == "pathA-v0.4+c1" and P._VS.endswith("_cx2_c1")
    P.set_context(False, False)
    assert P.MODEL_VERSION == "pathA-v0.2+c1"
    P.set_copula("v0")
    assert P.MODEL_VERSION == "pathA-v0.2" and P.SEED_VERSION == "pathA-v0.2"
    with pytest.raises(ValueError):
        P.set_copula("v2")
    assert P._copula_flag("NO_SUCH_ENV_VAR_XYZ") == "v0"


def test_bundle_json_keeps_v0_format():
    b = P.Bundle(season="2022-23", minutes=P.M.default_params("2022-23"),
                 state=ST.default_params(), copula=C.default_copula_params("2022-23"))
    assert "copula_v1" not in json.loads(b.to_json())
    b.copula_v1 = C.default_copula_v1_params("2022-23")
    back = P.Bundle.from_json(b.to_json())
    assert back.copula_v1 == b.copula_v1


# ------------------------------------------------------------------------- store + props

def test_stat_sims_round_trip_and_props_reader(tmp_path):
    rng = np.random.default_rng(0)
    S = 6000
    sim_path = tmp_path / "slate" / "run1.npz"
    sim_path.parent.mkdir()
    pid = np.array([11, 12, 13])
    stats = {s: rng.poisson(lam, (S, 3)).astype(np.uint8) for s, lam in
             (("pts", 15), ("fg3m", 2), ("reb", 6), ("ast", 4), ("stl", 1), ("blk", 1),
              ("tov", 2))}
    minutes = np.where(rng.random((S, 3)) < 0.9, 30.4, 0.0)
    minutes[:, 2] = 0.0
    minutes[:150, 2] = 12.0                                     # too few played sims
    p = SO.save_stat_sims(sim_path, pid, stats, minutes, game_id=["G1", "G1", "G2"])
    assert p == SO.stat_sims_path(sim_path) and p.name == "run1.stats.npz"
    z = SO.load_sims(p)
    assert z["pts"].shape == (SO.STAT_SIMS, 3) and z["pts"].dtype == np.uint8
    assert z["min"][0, 0] in (0, 31)
    run = {"run_id": "run1", "sim_path": str(sim_path)}
    d = PR.run_stat_sims(run, "G1", cache={})
    assert set(d) == {11, 12} and d[11].source == "sims"
    played = z["min"][:, 0] > 0
    assert d[11].means["pts"] == pytest.approx(z["pts"][played, 0].mean())
    pra = (z["pts"][played, 0].astype(int) + z["reb"][played, 0] + z["ast"][played, 0])
    assert d[11].means["pra"] == pytest.approx(pra.mean())
    above, push, below = PR.line_probs(d[11].pmf("pts"), 14.5)
    assert above == pytest.approx((z["pts"][played, 0] >= 15).mean())
    assert PR.run_stat_sims(run, "G2", cache={}) == {}
    assert PR.run_stat_sims({"run_id": "x", "sim_path": str(tmp_path / "none.npz")}) == {}
    assert PR.run_stat_sims({"run_id": "x"}) == {}


# ------------------------------------------------------------------------- pipeline end to end

def test_pipeline_v1_end_to_end_and_v0_unchanged(tmp_path, restore_switches):
    import sqlite3

    from courtside import db
    from test_proj_pipeline import SEASON, _fill, _late_day
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    _fill(conn)
    b = P.Bundle(season=SEASON, minutes=P.M.default_params(SEASON), state=ST.default_params(),
                 copula=C.default_copula_params(SEASON))
    ctx = P.Context(conn, SEASON)
    day = _late_day(conn)
    P.set_copula("v0")
    r0 = P.project_historical(conn, day, n_sims=1500, ctx=ctx, bundle=b)
    P.set_copula("v1")
    r1 = P.project_historical(conn, day, n_sims=1500, ctx=ctx, bundle=b)
    assert P.MODEL_VERSION.endswith("+c1")
    assert all(g.stats is not None for g in r1.sims)
    assert np.array_equal(r0.sims[0].minutes, r1.sims[0].minutes)        # CRN
    st = r1.stats
    playing = set(np.flatnonzero(r1.players["p_play"].to_numpy() > 0))
    pts = st[(st["stat"] == "pts") & (st["cond"] == "played") & st["key"].isin(playing)]
    assert len(pts) and pts["sd"].notna().all() and pts["q90"].notna().all()
    for t in r1.teams:
        assert abs(t["resid"]) <= RC.TOL
    meta = P.save_run(conn, r1, run_id="r-v1", slate_id=None, kind="manual")
    sp = SO.stat_sims_path(meta["sim_path"])
    assert sp.exists()
    z = SO.load_sims(sp)
    assert z["pts"].shape[1] == len(r1.players)
    d = PR.run_stat_sims({"run_id": "r-v1", "sim_path": meta["sim_path"]},
                         r1.sims[0].game_id, cache={})
    assert d and all(v.source == "sims" for v in d.values())
    # back to v0: identical to the first v0 run (the switch leaves no state behind)
    P.set_copula("v0")
    r0b = P.project_historical(conn, day, n_sims=1500, ctx=ctx, bundle=b)
    assert all(np.array_equal(x.fp, y.fp) for x, y in zip(r0.sims, r0b.sims))
    pd.testing.assert_frame_equal(r0.stats, r0b.stats)
    assert all(g.stats is None for g in r0b.sims)


def test_fit_copula_v1_smoke_falls_back_without_t27(restore_switches):
    import sqlite3

    from courtside import db
    from test_proj_pipeline import SEASON, _fill
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    _fill(conn)
    cp = C.fit_copula(conn, SEASON, ST.default_params(), n_seasons=2)
    v1 = C.fit_copula_v1(conn, SEASON, ST.default_params(), cp)
    assert v1.train_seasons and all(s < SEASON for s in v1.train_seasons)
    assert v1.corr["source"] == "copula-v0"                  # no corr_<S>.json, no extracts
    for t in C.TIERS:
        R_ = np.asarray(v1.r_within[t])
        assert np.allclose(np.diag(R_), 1.0) and np.linalg.eigvalsh(R_).min() > -1e-9
    for s in C.COMP_STATS:
        assert len(v1.disp_mu[s]) == len(v1.disp_c2[s]) >= 1
        assert min(v1.disp_c2[s]) >= C.C2_MIN
        for t in C.TIERS:
            assert all(C.WL_BOUNDS[0] <= x <= C.WL_BOUNDS[1] for x in v1.wl_factor[s][t])
    assert C.CopulaV1Params.from_json(v1.to_json()) == v1
    assert "dk_by_level" in v1.diagnostics
