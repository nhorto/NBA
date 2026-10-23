"""Minutes fix (COURTSIDE_MINFIX, `+mf`; `minutes.MF_*`, `fit_v1(minfix=...)`).

The switch is off by default and changes nothing when off (params JSON, projections, versions);
ON refits P(plays | available) with the GBM plan and recalibrates the noise by simulation, all
on the training seasons; versions / bundle suffixes compose with +pre / +mr / +c1 / +p2; the
debut reserve's driver is bounded to the training range."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from courtside.proj import minutes as M
from courtside.proj import pipeline as P

from test_proj_minutes import V1_KW, _league, _stints


@pytest.fixture(scope="module")
def league_mf():
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    prm0 = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=V1_KW)
    prm1 = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=dict(V1_KW, minfix=True))
    return h, prm0, prm1


@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX)
    yield
    P.set_copula(state[2])
    P.set_pplay(state[3])
    P.set_minratio(state[4])
    P.set_preseason(state[5])
    P.set_minfix(state[6])
    P.set_context(state[0], state[1])


def test_default_on_and_env_flag(monkeypatch):
    assert P.MINFIX_DEFAULT is True                                   # shipped 2026-10-23
    monkeypatch.setenv(P.MINFIX_ENV, "1")
    assert P._env_flag(P.MINFIX_ENV, P.MINFIX_DEFAULT) is True
    monkeypatch.setenv(P.MINFIX_ENV, "0")
    assert P._env_flag(P.MINFIX_ENV, P.MINFIX_DEFAULT) is False
    monkeypatch.delenv(P.MINFIX_ENV)
    assert P._env_flag(P.MINFIX_ENV, P.MINFIX_DEFAULT) is True


def test_versions_compose(restore_switches):
    # OFF: every arm's strings are the ones before the switch existed
    for on in (False, True):
        for mr in (False, True):
            for pre in (False, True):
                for cop in ("v0", "v1"):
                    for pp in ("v1", "v2"):
                        assert P._versions(on, False, cop, pp, mr, pre) == \
                            P._versions(on, False, cop, pp, mr, pre, False)
    assert P._versions(True, False, "v0", "v2", False, False, True) == ("_v1_cx_mf",
                                                                       "pathA-v0.3+mf+p2")
    assert P._versions(True, False, "v0", "v2", True, False, True) == ("_v1_cx_mf_mr",
                                                                      "pathA-v0.3+mf+mr+p2")
    assert P._versions(True, False, "v0", "v2", False, True, True) == ("_v1_cx_pre_mf",
                                                                      "pathA-v0.3+pre+mf+p2")
    assert P._versions(True, False, "v1", "v1", True, True, True) == (
        "_v1_cx_pre_mf_mr_c1", "pathA-v0.3+pre+mf+mr+c1")
    P.set_minfix(True)
    assert "+mf" in P.MODEL_VERSION and P._VS.endswith("_mf")
    P.set_minfix(False)
    assert "+mf" not in P.MODEL_VERSION and "_mf" not in P._VS


def test_params_json_off_identity(league_mf):
    h, prm0, prm1 = league_mf
    assert prm0.minfix == {} and "minfix" not in prm0.to_json()
    assert "minfix" not in M.default_params().to_json()
    assert prm1.minfix["version"] == M.MF_VERSION
    assert prm1.minfix["pplay"] == M.MF_DEFAULT["pplay"]
    assert M.MinutesParams.from_json(prm1.to_json()) == prm1
    assert M.MinutesParams(**json.loads(prm1.to_json())) == prm1
    # the GBM base, the v0.5 parts and the stage-2 inputs are shared; only P(plays), the
    # stage-2 choice it feeds and the noise move
    for k in ("gbm_model", "base_coef", "pplay_coef", "pplay_coef_v1", "reserve_coef"):
        assert getattr(prm0, k) == getattr(prm1, k), k


def test_off_projection_unchanged_by_the_code(league_mf):
    """OFF params never reach the minfix path: base_and_pplay equals the v1 logit exactly."""
    h, prm0, _ = league_mf
    t = M.eval_targets(h, ["2022-23"]).iloc[:40]
    panel = M.oracle_panel(h, t)
    ab = panel["absent"].to_numpy(bool)
    base, pp = M.base_and_pplay(panel, prm0, ab)
    q = panel.copy()
    b0 = np.clip(M.design(q, prm0.base_features) @ np.asarray(prm0.base_coef), 0.5, 44.0)
    q["avail_rank"], q["n_avail"] = M.avail_rank(q, b0, ab)
    z = M.design(M.forced_as_roster(q), prm0.pplay_features) @ np.asarray(prm0.pplay_coef_v1)
    assert np.allclose(pp, np.clip(1 / (1 + np.exp(-z)), 0.01, 0.995))


def test_on_pplay_reads_the_plan(league_mf):
    """ON: P(plays) moves with the GBM plan at fixed history (a lower plan, a lower P)."""
    h, prm0, prm1 = league_mf
    t = M.eval_targets(h, ["2022-23"]).iloc[:20]
    panel = M.oracle_panel(h, t)
    ab = panel["absent"].to_numpy(bool)
    q = panel.copy()
    b0 = np.clip(M.design(q, prm1.base_features) @ np.asarray(prm1.base_coef), 0.5, 44.0)
    q["avail_rank"], q["n_avail"] = M.avail_rank(q, b0, ab)
    base = np.asarray(M.base_and_pplay(panel, prm1, ab)[0], float)
    lo = M.mf_pplay(q, prm1, base * 0.6, ab)
    hi = M.mf_pplay(q, prm1, base, ab)
    assert lo.shape == hi.shape == (len(panel),)
    assert ((lo >= 0.01) & (lo <= 0.995)).all()
    assert lo.mean() < hi.mean()
    _, pp1 = M.base_and_pplay(panel, prm1, ab)
    assert np.allclose(pp1, hi)


@pytest.mark.parametrize("kind", ["logit", "logit_rank"])
def test_pplay_variants_fit(league_mf, kind):
    h, _, _ = league_mf
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                v1_kw=dict(V1_KW, minfix={"pplay": kind, "noise": "net"}))
    assert prm.minfix["pplay"] == kind
    n_rank = len(M.MF_RANKS) if kind == "logit_rank" else 0
    assert len(prm.minfix["pplay_coef"]) == (len(prm.pplay_features) + len(M.MF_PLAN_FEATURES)
                                             + n_rank)
    out = M.project_team(h, prm, "BOS", pd.Timestamp("2023-01-15"), season="2022-23")
    assert np.isclose((out["p_play"] * out["mean"]).sum(), 240.0 - out["reserve"].iloc[0])


def test_unknown_variant_rejected(league_mf):
    h, _, _ = league_mf
    with pytest.raises(ValueError):
        M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
              v1_kw=dict(V1_KW, minfix={"pplay": "gbm"}))


def test_noise_calibration_by_simulation(league_mf):
    """ON: per-tier noise location / scale are recalibrated on the training projections and
    the calibration history is stored; the margin curve is the OFF fit's shape."""
    h, prm0, prm1 = league_mf
    assert prm1.minfix["noise"] == "calib"
    cal = prm1.minfix["calibration"]
    assert len(cal) == M.MF_CAL_ITERS
    assert len(prm1.mu) == len(prm1.sigma) == len(M.TIER_EDGES) + 1
    assert all(0.02 <= s <= 2.0 for s in prm1.sigma)
    assert prm1.mu != prm0.mu or prm1.sigma != prm0.sigma


def test_g_rows_matches_g_value():
    prm = M.default_params()
    rng = np.random.default_rng(0)
    nt, nb = len(M.TIER_EDGES) + 1, len(M.MARGIN_BINS) - 1
    prm.g_table = rng.normal(0, 2, (nt, 2, nb)).tolist()
    prm.g_center = rng.normal(0, 0.3, nt).tolist()
    tiers = np.array([0, 1, 2, 3, 4, 4, 2])
    m = np.array([-30.0, -3.0, 0.0, 7.5, 12.0, 40.0, -18.0])
    exp = np.array([M.g_value(prm, np.array([t]), mm)[0] for t, mm in zip(tiers, m)])
    assert np.allclose(M._g_rows(prm, tiers, m), exp)


def test_load_bundle_mf_refits_minutes_only(monkeypatch, tmp_path, restore_switches):
    """The +mf bundle is the arm's bundle with only the minutes params replaced (fit < S)."""
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    calls = {}
    real_fit = M.fit

    def fake_fit(h_, season, **kw):
        calls.setdefault("kw", []).append(kw.get("v1_kw"))
        return real_fit(h_, season, window=3, log=lambda s: None, f_grid=(0.6,),
                        version="v1", v1_kw=dict(V1_KW, **(kw.get("v1_kw") or {})))

    class Ctx:
        def __init__(self):
            self.h = h

        def train_panel(self):
            return M.oracle_panel(h, M.eval_targets(h, ["2019-20", "2020-21", "2021-22"]))

    base = P.Bundle(season="2022-23", minutes=M.default_params("2022-23"), state={"x": 1},
                    copula=P.C.default_copula_params("2022-23"))
    monkeypatch.setattr(P, "load_bundle", lambda *a, **k: base if not P.MINFIX else
                        P._load_bundle_mf(*a, refit=False, log=None))
    monkeypatch.setattr(P.M, "fit", fake_fit)
    P.set_context(False)
    P.set_minfix(True)
    b = P._load_bundle_mf(None, "2022-23", Ctx(), refit=False, log=None)
    assert calls["kw"] == [{"minfix": True}]
    assert b.minutes.minfix and b.state == base.state and b.copula == base.copula
    assert (P.cache_dir() / "minutes_params_2022-23_v1_mf.json").exists()
    assert (P.cache_dir() / "bundle_2022-23_v1_mf.json").exists()
    # cached: no second fit
    b2 = P._load_bundle_mf(None, "2022-23", Ctx(), refit=False, log=None)
    assert len(calls["kw"]) == 1 and b2.minutes == b.minutes


def test_debut_driver_top2_and_old_params():
    """The debut reserve reads the driver its params were fit with: the first design's params
    (no driver code) read the camp sum, new params (code 2) the two largest new players."""
    p = pd.DataFrame({"tgs": [0, 0, 3], "pre_new_tg": [120.0, 30.0, 30.0],
                      "pre_new_top2": [50.0, 30.0, 30.0]})
    old = M.default_params()
    old.reserve_debut = [-6.0, 1.0, 0.3, 0.1, 0.5, 0.1, 0.0, 0.0, 70.0]
    new = M.default_params()
    new.reserve_debut = [-6.0, 1.0, 0.3, 0.1, 0.5, 0.1, 0.0, 0.0, 52.5, 2.0]
    assert np.allclose(M.debut_reserve(old, p), [29.0, 9.0, 4.0])      # min(120, 70) x 0.5 - 6
    assert np.allclose(M.debut_reserve(new, p), [19.0, 9.0, 4.0])      # min(50, 52.5) x 0.5 - 6
    assert M.MinutesParams.from_json(new.to_json()) == new


def test_new_player_top2_is_bounded_by_two_players():
    from test_proj_preseason_features import _preseason
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    pre = _preseason(pg, tg)
    # three camp players with no box row: 30, 20, 10 minutes per game in all 4 BOS games
    first = pre.teams[(pre.teams["team"] == "BOS") & (pre.teams["season"] == "2022-23")]
    rows = [{"season": "2022-23", "player_id": 900 + i, "team": "BOS", "game_id": g,
             "date": d, "min": m} for i, m in enumerate((30.0, 20.0, 10.0))
            for g, d in zip(first["game_id"], first["date"])]
    h.pre = M.Preseason(pd.concat([pre.players, pd.DataFrame(rows)], ignore_index=True),
                        pre.teams)
    t = M.eval_targets(h, ["2022-23"])
    t = t[(t["team"] == "BOS")].iloc[:1]
    p = M.pregame_panel(h, t)
    assert np.isclose(p["pre_new_top2"].iloc[0], 50.0)
    assert np.isclose(p["pre_new_tg"].iloc[0], 60.0)
