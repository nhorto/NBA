"""Player-specific minutes spread (COURTSIDE_MIN_SPREAD, `+ms`; `minutes.fit_spread`,
IMPROVEMENTS.md P6).

OFF by default and nothing of it runs when off (params JSON, minute draws, scenario frames,
versions); ON each player's log-minutes noise is a two-piece normal with his own lower / upper
scale (LightGBM quantiles, seasons < S) re-centred to the tier's mean multiplier, drawn from the
same random numbers; the bundle / version suffixes compose after `+ep`."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
from test_proj_minutes import V1_KW, _league, _stints

from courtside.proj import copula as C
from courtside.proj import early_prior as EP
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

# minutes.sample_minutes OFF digest from HEAD 881877e (var/tmp/p46/golden.py on the HEAD tree)
GOLDEN_SAMPLE = "0508501d795e19901e9f846869438c33138a2885"


@pytest.fixture
def restore():
    state = (P.MIN_SPREAD, EP.EARLY_PRIOR, P.MINFIX, P.CONTEXT_ON, P.PPLAY)
    yield
    P.set_min_spread(state[0])
    P.set_early_prior(state[1])
    P.set_minfix(state[2])
    P.set_pplay(state[4])
    P.set_context(state[3])


@pytest.fixture(scope="module")
def league_ms():
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                v1_kw=dict(V1_KW, minfix=True))
    panel = M.oracle_panel(h, M.eval_targets(h, prm.train_seasons))
    sp = M.fit_spread(prm, panel, M.PairShares(M.pair_events(panel)), log=lambda s: None,
                      max_rounds=V1_KW["max_rounds"], gbm_params=V1_KW["gbm_params"], rounds=40,
                      spread_gbm={"min_data_in_leaf": 20}, min_fold_rows=50)
    return h, prm, sp, panel


def _proj():
    return pd.DataFrame({"mean": [34.0, 28.0, 22.0, 15.0, 8.0, 3.0, 0.0],
                         "p_play": [0.99, 0.97, 0.9, 0.8, 0.6, 0.3, 0.0],
                         "pf36": [3.0, 4.0, 3.5, 5.0, 4.2, 6.0, 3.0], "reserve": 2.0})


def _params():
    mp = M.default_params()
    mp.mu = [0.0, 0.02, -0.01, 0.01, 0.0]
    mp.sigma = [0.4, 0.3, 0.25, 0.2, 0.15]
    return mp


def _draw(mp, proj, seed=3, S=4000):
    return M.sample_minutes(proj, mp, np.linspace(-25, 25, S), np.random.default_rng(seed),
                            n_ot=(np.arange(S) % 17 == 0).astype(float))


# ------------------------------------------------------------------------------ switch

def test_default_on_and_env_flag(monkeypatch):
    assert P.MIN_SPREAD_DEFAULT is True                  # shipped 2026-10-24 (conftest pins it off)
    monkeypatch.setenv(P.MIN_SPREAD_ENV, "1")
    assert P._env_flag(P.MIN_SPREAD_ENV, P.MIN_SPREAD_DEFAULT) is True
    monkeypatch.setenv(P.MIN_SPREAD_ENV, "0")
    assert P._env_flag(P.MIN_SPREAD_ENV, P.MIN_SPREAD_DEFAULT) is False


def test_versions_compose(restore):
    P.set_context(True)
    P.set_pplay("v2")
    P.set_minfix(True)
    P.set_early_prior(True)
    P.set_min_spread(False)
    vs, mv = P._VS, P.MODEL_VERSION
    assert (vs, mv.split("+p2")[0]) == ("_v1_cx_mf_ep", "pathA-v0.3+mf+ep")
    P.set_min_spread(True)
    assert P._VS == "_v1_cx_mf_ep_ms"
    assert P.MODEL_VERSION == mv.replace("+ep+p2", "+ep+ms+p2")
    P.set_min_spread(False)
    assert (P._VS, P.MODEL_VERSION) == (vs, mv)


def test_json_omits_empty_spread():
    mp = M.default_params()
    assert mp.spread == {} and "spread" not in mp.to_json()
    mp.spread = {"version": 1}
    assert M.MinutesParams.from_json(mp.to_json()).spread == {"version": 1}


# ------------------------------------------------------------------------------ OFF

@pytest.mark.skipif(sys.platform != "linux", reason="exact float golden: Linux only")
def test_sample_minutes_off_matches_head_golden():
    x = _draw(_params(), _proj())
    assert hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest() == GOLDEN_SAMPLE


def test_off_ignores_spread_columns():
    """Columns alone do nothing: the params must carry the spread."""
    pr = _proj().assign(sp_lo=2.0, sp_hi=0.5)
    assert np.array_equal(_draw(_params(), _proj()), _draw(_params(), pr))


def test_scenario_minutes_columns_only_on(league_ms):
    h, prm, sp, panel = league_ms
    key = panel["key"].iloc[0]
    tp = panel[panel["key"] == key].drop(columns=["absent"])
    absent = np.zeros(len(tp), bool)
    off = C.ScenarioMinutes(tp, prm).project(absent)
    on = C.ScenarioMinutes(tp, sp).project(absent)
    assert not any(c in off for c in M.SPREAD_COLS)
    assert all(c in on for c in M.SPREAD_COLS)
    lo, hi = M.SPREAD_BOUNDS
    assert on["sp_lo"].between(lo, hi).all() and on["sp_hi"].between(lo, hi).all()


# ------------------------------------------------------------------------------ ON

def test_unit_spread_is_the_tier_noise():
    mp = _params()
    mp.spread = {"version": 1}
    pr = _proj().assign(sp_lo=1.0, sp_hi=1.0)
    a, b = _draw(_params(), _proj()), _draw(mp, pr)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10)
    assert np.array_equal(a > 0, b > 0)                       # same DNP draws


def test_two_piece_keeps_mean_and_moves_tails():
    S = 400_000
    xi = np.random.default_rng(0).standard_normal((S, 3))
    mu, sg = np.array([0.0, 0.05, -0.02]), np.array([0.3, 0.2, 0.25])
    lo, hi = np.array([0.5, 1.0, 2.0]), np.array([2.0, 1.0, 0.5])
    z = np.exp(M.spread_noise(mu, sg, lo, hi, xi))
    assert np.allclose(z.mean(0), np.exp(mu + sg ** 2 / 2), rtol=4e-3)
    q = np.quantile(np.log(z), [0.1, 0.5, 0.9], axis=0)
    up, down = q[2] - q[1], q[1] - q[0]
    assert up[0] > 3 * down[0] and down[2] > 3 * up[2]
    assert up[1] == pytest.approx(down[1], rel=0.03)


def test_same_draws_on_off():
    mp = _params()
    mp.spread = {"version": 1}
    pr = _proj().assign(sp_lo=[0.5, 1, 2, 1.5, 0.7, 1, 1], sp_hi=[2, 1, 0.5, 0.8, 1.6, 1, 1])
    a, b = _draw(_params(), _proj()), _draw(mp, pr)
    assert np.array_equal(a > 0, b > 0)
    assert not np.allclose(a, b)


def test_spread_scales_sorted_and_bounded():
    q = np.array([[0.1, -0.2, -0.05], [-0.3, 0.0, 0.2], [-0.01, 0.0, 0.0]])
    s_lo, s_hi = M.spread_scales(q, np.array([0, 1, 2]), [0.1] * 5, [0.1] * 5)
    srt = np.sort(q, axis=1)
    assert s_lo[0] == pytest.approx((srt[0, 1] - srt[0, 0]) / 1.2815515655446004 / 0.1)
    assert s_lo.min() >= M.SPREAD_BOUNDS[0] and s_hi.max() <= M.SPREAD_BOUNDS[1]


def test_fit_spread_keeps_the_arm(league_ms):
    h, prm, sp, panel = league_ms
    a, b = json.loads(prm.to_json()), json.loads(sp.to_json())
    for k in a:
        if k not in ("mu", "sigma", "diagnostics"):
            assert a[k] == b[k], k
    s = sp.spread
    assert s["version"] == M.SPREAD_VERSION and len(s["models"]) == len(M.SPREAD_QUANTS)
    assert s["mu_arm"] == prm.mu and s["sigma_arm"] == prm.sigma
    assert len(s["ref_lo"]) == len(M.TIER_EDGES) + 1 and min(s["ref_lo"]) > 0
    assert set(sp.diagnostics["spread"]["pinball_oof"]) == {"q10", "q50", "q90"}
    assert M.MinutesParams.from_json(sp.to_json()).spread["ref_hi"] == s["ref_hi"]
