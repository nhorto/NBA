"""Blowout minutes (COURTSIDE_BLOWOUT, `+bo`; `minutes.BO_*`, `fit_v1(blowout=True)`).

The switch is off by default and changes nothing when off (params JSON, minute draws, versions);
ON adds a blowout stage fit on the training seasons (P(plays) and reserve offsets by margin, the
margin curve calibrated by simulation) without touching anything fit before it; versions /
bundle suffixes compose after +mf and before +mr / +c1 / +p2; the bundle loader refits the
minutes params only and caches them."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (fixture)
from test_proj_minutes import V1_KW, _league, _stints

from courtside.proj import minutes as M
from courtside.proj import pipeline as P

K, B = len(M.BO_P_EDGES), len(M.MARGIN_BINS) - 1


@pytest.fixture(scope="module")
def league_bo():
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    prm0 = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=V1_KW)
    prm1 = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=dict(V1_KW, blowout=True))
    return h, prm0, prm1


@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX,
             P.BLOWOUT)
    yield
    P.set_copula(state[2])
    P.set_pplay(state[3])
    P.set_minratio(state[4])
    P.set_preseason(state[5])
    P.set_minfix(state[6])
    P.set_blowout(state[7])
    P.set_context(state[0], state[1])


def _proj():
    return pd.DataFrame({"mean": [35.0, 32, 28, 24, 18, 12, 8, 5, 3, 0.0],
                         "p_play": [1.0, 0.995, 0.99, 0.97, 0.9, 0.7, 0.5, 0.3, 0.15, 0.0],
                         "pf36": 3.5, "reserve": 2.0})


def _bo(prm, p_off=None, res_off=None):
    return {"version": M.BO_VERSION,
            "p_off": (np.zeros((K, 2, B)) if p_off is None else p_off).tolist(),
            "res_off": [0.0] * B if res_off is None else list(res_off),
            "g_table": prm.g_table, "g_center": prm.g_center}


def test_default_off_and_env_flag(monkeypatch):
    assert P.BLOWOUT_DEFAULT is False
    monkeypatch.setenv(P.BLOWOUT_ENV, "1")
    assert P._env_flag(P.BLOWOUT_ENV, P.BLOWOUT_DEFAULT) is True
    monkeypatch.setenv(P.BLOWOUT_ENV, "0")
    assert P._env_flag(P.BLOWOUT_ENV, True) is False
    monkeypatch.delenv(P.BLOWOUT_ENV)
    assert P._env_flag(P.BLOWOUT_ENV, P.BLOWOUT_DEFAULT) is False


def test_versions_compose(restore_switches):
    # OFF: every arm's strings are the ones before the switch existed
    for on in (False, True):
        for mr in (False, True):
            for pre in (False, True):
                for mf in (False, True):
                    for cop in ("v0", "v1"):
                        for pp in ("v1", "v2"):
                            assert P._versions(on, False, cop, pp, mr, pre, mf) == \
                                P._versions(on, False, cop, pp, mr, pre, mf, False)
    assert P._versions(True, False, "v0", "v2", False, False, False, True) == (
        "_v1_cx_bo", "pathA-v0.3+bo+p2")
    assert P._versions(True, False, "v0", "v2", False, False, True, True) == (
        "_v1_cx_mf_bo", "pathA-v0.3+mf+bo+p2")
    assert P._versions(True, False, "v1", "v1", True, True, True, True) == (
        "_v1_cx_pre_mf_bo_mr_c1", "pathA-v0.3+pre+mf+bo+mr+c1")
    P.set_blowout(True)
    assert "+bo" in P.MODEL_VERSION and P._VS.endswith("_bo")
    P.set_blowout(False)
    assert "+bo" not in P.MODEL_VERSION and "_bo" not in P._VS


def test_params_json_off_identity(league_bo):
    _, prm0, prm1 = league_bo
    assert prm0.blowout == {} and "blowout" not in prm0.to_json()
    assert "blowout" not in M.default_params().to_json()
    assert M.MinutesParams.from_json(prm0.to_json()) == prm0
    assert prm1.blowout["version"] == M.BO_VERSION
    # the stage changes nothing fit before it
    d0, d1 = json.loads(prm0.to_json()), json.loads(prm1.to_json())
    d1.pop("blowout")
    d0.pop("diagnostics"), d1.pop("diagnostics")
    assert d0 == d1
    assert M.MinutesParams.from_json(prm1.to_json()).blowout == json.loads(
        json.dumps(prm1.blowout))


def test_fitted_stage_shapes_and_centring(league_bo):
    _, _, prm1 = league_bo
    bo = prm1.blowout
    a, n = np.asarray(bo["p_off"]), np.asarray(bo["p_n"], float)
    assert a.shape == (K, 2, B) and n.shape == (K, 2, B)
    for k in range(K):          # each band keeps its level (count-weighted mean offset 0)
        if n[k].sum():
            assert abs((a[k] * n[k]).sum() / n[k].sum()) < 1e-9
    assert len(bo["res_off"]) == B
    assert np.asarray(bo["g_table"]).shape == np.asarray(prm1.g_table).shape
    assert len(bo["calib"]) == M.BO_CAL_ITERS + 1


def test_zero_stage_draws_equal_off():
    """With zero offsets and the arm's own curve the blowout path draws exactly the OFF minutes
    (same random stream: common random numbers)."""
    prm = M.default_params()
    rng = np.random.default_rng(3)
    nt = len(M.TIER_EDGES) + 1
    prm.g_table = rng.normal(0, 2, (nt, 2, B)).tolist()
    m = rng.normal(2.0, 14.0, 800)
    ot = (np.abs(m) < 1).astype(float)
    x0 = M.sample_minutes(_proj(), prm, m, np.random.default_rng(9), n_ot=ot)
    prm1 = M.MinutesParams.from_json(prm.to_json())
    prm1.blowout = _bo(prm)
    x1 = M.sample_minutes(_proj(), prm1, m, np.random.default_rng(9), n_ot=ot)
    assert np.array_equal(x0, x1)


def test_offsets_move_the_bench_and_stars_with_the_margin():
    prm = M.default_params()
    p_off = np.zeros((K, 2, B))
    p_off[:2, :, -3:] = 1.5            # deep bench plays more in blowouts
    p_off[:2, :, :2] = -1.0            # ... and less in close games
    prm1 = M.MinutesParams.from_json(prm.to_json())
    prm1.blowout = _bo(prm, p_off, res_off=np.r_[np.zeros(B - 2), 1.5, 2.0])
    S = 4000
    close, blow = np.zeros(S), np.full(S, 34.0)
    xc = M.sample_minutes(_proj(), prm1, close, np.random.default_rng(1))
    xb = M.sample_minutes(_proj(), prm1, blow, np.random.default_rng(1))
    x0 = M.sample_minutes(_proj(), prm, blow, np.random.default_rng(1))
    deep = [7, 8]
    assert (xb[:, deep] > 0).mean() > (x0[:, deep] > 0).mean() > (xc[:, deep] > 0).mean()
    assert xb[:, 0].mean() < x0[:, 0].mean()           # the star gives minutes back
    assert np.isclose(np.median(xb.sum(1)), 240.0 - 2.0 - 2.0)   # reserve offset at 34: +2
    assert (xb[:, -1] == 0).all()                      # absent stays absent


def test_bo_pplay_bounds():
    bo = {"p_off": np.full((K, 2, B), 3.0).tolist()}
    p = np.array([0.0, 0.2, 0.6, 0.99, 1.0])
    q = M.bo_pplay(bo, p, np.array([-30.0, 0.0, 12.0]))
    assert q.shape == (3, 5)
    assert (q[:, 0] == 0).all() and (q[:, -1] == 1).all()
    assert (q[:, 1:4] > p[None, 1:4]).all()
    bo0 = {"p_off": np.zeros((K, 2, B)).tolist()}
    assert np.allclose(M.bo_pplay(bo0, p, np.array([5.0, -40.0])), p[None, :])


def test_reserve_offsets_centred():
    r = pd.DataFrame({"key": np.repeat([f"g{i}" for i in range(400)], 2),
                      "margin": np.repeat(np.linspace(-40, 40, 400), 2),
                      "n_ot": 0.0, "act_min": 118.0, "reserve": 3.0})
    r.loc[r["margin"].abs() > 25, "act_min"] = 117.0      # 2 more outside minutes in blowouts
    v = np.asarray(M._bo_fit_reserve(r))
    b = M._bo_bin(r.drop_duplicates("key")["margin"])
    n = np.bincount(b, minlength=B)
    assert abs((v * n).sum()) < 1e-9
    assert v[-1] > v[0]


def test_load_bundle_bo_refits_minutes_only(monkeypatch, tmp_path, restore_switches):
    """The +bo bundle is the arm's bundle with only the minutes params replaced (fit < S); with
    MINFIX on the refit carries both."""
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    calls = {}

    def fake_fit(h_, season, **kw):
        calls.setdefault("kw", []).append(kw.get("v1_kw"))
        prm = M.default_params(season)
        prm.blowout = {"version": 1}
        prm.minfix = {"version": 1} if (kw.get("v1_kw") or {}).get("minfix") else {}
        return prm

    class Ctx:
        def __init__(self):
            self.h = h

        def train_panel(self):
            return M.oracle_panel(h, M.eval_targets(h, ["2019-20", "2020-21", "2021-22"]))

    base = P.Bundle(season="2022-23", minutes=M.default_params("2022-23"), state={"x": 1},
                    copula=P.C.default_copula_params("2022-23"))
    monkeypatch.setattr(P, "load_bundle", lambda *a, **k: base)
    monkeypatch.setattr(P.M, "fit", fake_fit)
    P.set_context(False)
    P.set_minfix(False)
    P.set_blowout(True)
    b = P._load_bundle_bo(None, "2022-23", Ctx(), refit=False, log=None)
    assert calls["kw"] == [{"blowout": True}]
    assert b.minutes.blowout and b.state == base.state and b.copula == base.copula
    assert (P.cache_dir() / "minutes_params_2022-23_v1_bo.json").exists()
    assert (P.cache_dir() / "bundle_2022-23_v1_bo.json").exists()
    b2 = P._load_bundle_bo(None, "2022-23", Ctx(), refit=False, log=None)   # cached
    assert len(calls["kw"]) == 1 and b2.minutes == b.minutes
    P.set_minfix(True)
    b3 = P._load_bundle_bo(None, "2022-23", Ctx(), refit=False, log=None)
    assert calls["kw"][-1] == {"blowout": True, "minfix": True}
    assert b3.minutes.minfix and (P.cache_dir() / "minutes_params_2022-23_v1_mf_bo.json").exists()


def test_off_switch_dispatch_untouched(monkeypatch, restore_switches):
    """With BLOWOUT off load_bundle never reaches the blowout loader."""
    P.set_blowout(False)
    monkeypatch.setattr(P, "_load_bundle_bo", lambda *a, **k: pytest.fail("bo loader"))
    monkeypatch.setattr(P, "_load_bundle_mf", lambda *a, **k: "mf")
    P.set_minfix(True)
    assert P.load_bundle(None, "2022-23") == "mf"


def test_bo_sigma_used_only_under_bo():
    prm = M.default_params()
    m = np.random.default_rng(4).normal(0, 14, 2000)
    x0 = M.sample_minutes(_proj(), prm, m, np.random.default_rng(2))
    prm1 = M.MinutesParams.from_json(prm.to_json())
    prm1.blowout = dict(_bo(prm), sigma=list(prm.sigma))
    assert np.array_equal(x0, M.sample_minutes(_proj(), prm1, m, np.random.default_rng(2)))
    prm1.blowout["sigma"] = [0.05] * len(prm.sigma)
    x1 = M.sample_minutes(_proj(), prm1, m, np.random.default_rng(2))
    assert x1[:, 0].std() < x0[:, 0].std()


def test_fitted_stage_keeps_the_noise_level(league_bo):
    _, prm0, prm1 = league_bo
    bo = prm1.blowout
    assert len(bo["sigma"]) == len(prm0.sigma) and len(bo["cov0"]) == len(prm0.sigma)
    assert prm1.sigma == prm0.sigma          # the arm's own noise is untouched


def test_minutes_factor_and_lookup_wrapper():
    from courtside.proj import copula as C
    prm = M.default_params()
    nt = len(M.TIER_EDGES) + 1
    tab = np.zeros((nt, 2, B))
    tab[4, :, -2:] = -6.0                    # stars lose 6 minutes in blowouts
    prm.blowout = dict(_bo(prm), g_table=tab.tolist(), g_center=[0.0] * nt)
    proj = _proj()
    k = M.bo_minutes_factor(prm, proj, np.array([0.0, 40.0, -40.0]))
    assert np.allclose(k[0], 1.0)            # no margin effect: f read at the minutes
    assert np.isclose(k[1, 0], 35.0 / 29.0) and np.isclose(k[2, 0], 35.0 / 29.0)
    assert k[1, -1] == 1.0                   # absent player
    grid = np.tile(np.arange(C.GRID_MAX + 1, dtype=float) ** 1.5, (len(proj), 1))

    def look(key, rl, xx):
        return C._lookup(grid, None, xx)
    x = np.full((3, len(proj)), 20.0)
    played = x > 0
    lk = C._bo_look(look, np.ones_like(x), played)
    assert np.allclose(lk("pts", None, x), look("pts", None, x))
    lk = C._bo_look(look, k, played)
    v = lk("pts", None, x)
    assert np.allclose(v, look("pts", None, x * k) / k)
    assert v[1, 0] > look("pts", None, x)[1, 0]      # grid convex: a higher per-minute rate


def test_zero_stage_simulate_game_equals_off(game):  # noqa: F811  (the fixture)
    """Copula v0 under the blowout arm: a zero stage (no offsets, the arm's curve) draws exactly
    the OFF sims; a real curve moves the stars' minutes and keeps each team's points."""
    from courtside.proj import copula as C
    spec, mp, sp = _spec(game)
    cp = C.default_copula_params()
    a = C.simulate_game(spec, cp, mp, sp, 3000, 11)
    mp_bo = M.MinutesParams.from_json(mp.to_json())
    mp_bo.blowout = _bo(mp)
    b = C.simulate_game(spec, cp, mp_bo, sp, 3000, 11)
    assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    nt = len(M.TIER_EDGES) + 1
    tab = np.zeros((nt, 2, B))
    tab[3:, :, -3:] = -5.0
    mp_bo.blowout = dict(_bo(mp), g_table=tab.tolist(), g_center=[0.0] * nt)
    c = C.simulate_game(spec, cp, mp_bo, sp, 3000, 11)
    assert not np.array_equal(a.minutes, c.minutes)
    for team, info in c.teams.items():
        assert info["team"] == team
