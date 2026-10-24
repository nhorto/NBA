"""Weighted reconciliation (COURTSIDE_RECON_WEIGHTED, `+rw`; `reconcile._reconcile_weighted`,
IMPROVEMENTS.md P4).

OFF by default and nothing of it runs when off (same function results, same sims, same versions).
ON the team gap to the implied total is split by Gaussian conditioning on the team sum (weights
theta e_i e + V_i): the team still hits the target, players with little rate evidence absorb
more, theta -> infinity is the uniform factor, the minutes draws are unchanged, and a gap above
GAP_CHECK is logged as a projection error."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC
from courtside.proj import state as ST

DAY = "2024-01-10"
# P.run_games OFF digest from HEAD 881877e (var/tmp/p46/golden.py on the HEAD tree)
GOLDEN_OFF = "72ef1442686f9c490bffda9222869e399b8d53d8"


@pytest.fixture
def restore():
    state = RC.RECON_WEIGHTED
    yield
    P.set_recon_weighted(state)


def _head_reconcile(raw_pts, raw_oth, g, alpha, target, weight=1.0):
    """`reconcile.reconcile_team` as of HEAD 881877e (frozen copy)."""
    g = np.maximum(np.asarray(g, float), 0.05) ** weight
    baseline = float(raw_pts.sum(1).mean()) if raw_pts.size else 0.0
    tgt = baseline + weight * (float(target) - baseline) if np.isfinite(target) else baseline
    gp = raw_pts * g[:, None]
    e = float(gp.sum(1).mean()) if raw_pts.size else 0.0
    c_raw = tgt / e if e > 0 else 1.0
    c = float(np.clip(c_raw, *RC.SCALE_BOUNDS))
    pts = gp * c
    pace = (g * c) ** alpha
    oth = raw_oth * pace[:, None]
    post = float(pts.sum(1).mean()) if raw_pts.size else 0.0
    resid = post - tgt
    return {"pts": pts, "oth": oth, "pace": pace, "target": tgt, "baseline": baseline,
            "gap": baseline - float(target), "scale": c, "scale_uncapped": float(c_raw),
            "post": post, "resid": resid, "ok": bool(abs(resid) <= RC.TOL),
            "data_check": bool(abs(baseline - target) > RC.GAP_CHECK)}


def _arrays(seed=0, S=500, n=9):
    rng = np.random.default_rng(seed)
    lvl = rng.uniform(2, 25, n)
    raw = lvl[None, :] * rng.gamma(4, 0.25, (S, n)) * (rng.random((S, n)) > 0.1)
    oth = raw * rng.uniform(0.6, 1.4, n)[None, :]
    g = rng.normal(1.0, 0.08, S)
    return raw, oth, g


def _bundle(mp, sp):
    return SimpleNamespace(copula=C.default_copula_params("2023-24"), minutes=mp, state=sp,
                           season="2023-24", copula_v1=None)


def _spec(game, implied=(118.0, 103.0), evidence=None):  # noqa: F811
    h, teams, sp = game
    mp = M.default_params()
    specs = {}
    for k, (side, (team, panel, pl)) in enumerate(teams.items()):
        pl = pl.copy()
        if side == "home":
            pl.loc[[0], "p_avail"] = 0.0
            pl.loc[[1, 3], "p_avail"] = 0.5
        if evidence is not None:
            pl["n_pts"] = evidence(len(pl))
            pl["k_pts"] = 300.0
        specs[side] = C.TeamSpec(team=team, players=pl, minutes=C.ScenarioMinutes(panel, mp),
                                 implied=implied[k])
    return C.GameSpec(game_id="G1", home=specs["home"], away=specs["away"], mu_margin=4.0,
                      mu_total=221.0), mp, sp


def _digest(res) -> str:
    x = hashlib.sha1()
    for g in res.sims:
        for a in (g.fp, g.minutes, np.asarray(g.player_id)):
            x.update(np.ascontiguousarray(a).tobytes())
    x.update(pd.util.hash_pandas_object(res.players, index=True).to_numpy().tobytes())
    x.update(pd.util.hash_pandas_object(res.stats, index=True).to_numpy().tobytes())
    meta = {k: v for k, v in res.meta.items() if not k.startswith("seconds")}
    x.update(json.dumps(meta, sort_keys=True, default=str).encode())
    return x.hexdigest()


# ------------------------------------------------------------------------------ switch

def test_default_off_and_env_flag(monkeypatch):
    assert RC.RECON_WEIGHTED_DEFAULT is False
    monkeypatch.setenv(RC.RECON_WEIGHTED_ENV, "1")
    assert RC._flag(RC.RECON_WEIGHTED_ENV, RC.RECON_WEIGHTED_DEFAULT) is True
    monkeypatch.setenv(RC.RECON_WEIGHTED_ENV, "0")
    assert RC._flag(RC.RECON_WEIGHTED_ENV, RC.RECON_WEIGHTED_DEFAULT) is False


def test_version_tag(restore):
    P.set_recon_weighted(False)
    vs, mv = P._VS, P.MODEL_VERSION
    assert "+rw" not in mv
    P.set_recon_weighted(True)
    assert P.MODEL_VERSION == mv + "+rw" and P._VS == vs          # model version only
    st = RC.RECON_WEIGHTED, P.TR.TAIL_REMAP
    try:
        P.set_tail_remap(True)
        assert P.MODEL_VERSION.endswith("+tr+rw")
    finally:
        P.set_tail_remap(st[1])
    P.set_recon_weighted(False)
    assert P.MODEL_VERSION == mv


# ------------------------------------------------------------------------------ OFF

@pytest.mark.parametrize("weight,target", [(1.0, 110.0), (0.5, 90.0), (1.0, 400.0),
                                           (1.0, float("nan"))])
def test_off_reconcile_is_head(weight, target):
    raw, oth, g = _arrays()
    a = RC.reconcile_team(raw, oth, g, 0.6, target, weight)
    b = _head_reconcile(raw, oth, g, 0.6, target, weight)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k


def test_run_games_off_never_runs(game, restore, monkeypatch):  # noqa: F811
    P.set_recon_weighted(False)
    monkeypatch.setattr(RC, "_reconcile_weighted",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    spec, mp, sp = _spec(game)
    r = P.run_games([spec], _bundle(mp, sp), 1500, 7, P._as_date(DAY))
    assert all("weighted" not in t for t in r.teams)


@pytest.mark.skipif(sys.platform != "linux", reason="exact float golden: Linux only")
def test_run_games_off_matches_head_golden(game, restore):  # noqa: F811
    P.set_recon_weighted(False)
    spec, mp, sp = _spec(game)
    a = P.run_games([spec], _bundle(mp, sp), 3000, 13, P._as_date(DAY))
    assert _digest(a) == GOLDEN_OFF


# ------------------------------------------------------------------------------ ON, unit

def test_weighted_hits_target_and_bounds():
    raw, oth, g = _arrays(1)
    n = raw.shape[1]
    rw = {"r_pts": np.full(n, 0.5), "n_pts": np.linspace(0, 3000, n), "kappa": 300.0,
          "theta": 1e-4}
    base = float((raw * np.maximum(g, .05)[:, None]).sum(1).mean())
    for tgt in (base - 15, base - 3, base + 4, base + 12):
        r = RC.reconcile_team(raw, oth, g, 0.6, tgt, rw=rw)
        assert abs(r["resid"]) <= 1e-6 and r["ok"]
        c = np.asarray(r["c"])
        assert ((c >= RC.PLAYER_BOUNDS[0] - 1e-12) & (c <= RC.PLAYER_BOUNDS[1] + 1e-12)).all()
        assert r["pace"].shape == raw.shape
        assert np.allclose(r["oth"], oth * (g[:, None] * c[None, :]) ** 0.6)
        assert np.sum(r["delta"]) == pytest.approx(tgt - base, abs=1e-6)


def test_theta_limits_and_low_evidence_absorbs():
    e = np.array([25.0, 20.0, 15.0, 10.0, 6.0, 3.0])
    r = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.35])
    n = np.array([3000, 2500, 1500, 800, 0, 0.0])
    v = RC.rate_variance(e, r, n, 300.0)
    tgt = e.sum() + 5.0
    cu = RC.weighted_factors(e, v, tgt, 1e9)
    assert np.allclose(cu, tgt / e.sum(), atol=1e-6)              # theta -> inf: uniform c
    c0 = RC.weighted_factors(e, v, tgt, 0.0)
    assert (e * c0).sum() == pytest.approx(tgt)
    assert np.allclose((c0 - 1) * e / v, (c0[0] - 1) * e[0] / v[0])   # delta_i oc V_i
    assert c0[4] > c0[0] and c0[5] > c0[1]                         # no evidence absorbs more
    assert np.all(c0[4:] > cu[4:]) and np.all(c0[:2] < cu[:2])


def test_bounds_hand_the_rest_on():
    e = np.array([20.0, 2.0])
    v = np.array([0.0, 5.0])                       # all of the gap goes to the second player
    c = RC.weighted_factors(e, v, 30.0, 0.0)
    assert c[1] == pytest.approx(RC.PLAYER_BOUNDS[1])
    assert (e * c).sum() < 30.0                    # the second is at its bound, the first has w 0
    raw = np.tile(e, (10, 1))
    out = RC.reconcile_team(raw, raw, np.ones(10), 0.5, 30.0,
                            rw={"r_pts": np.ones(2), "n_pts": np.array([1e12, 0.0]),
                                "kappa": 1.0, "theta": 0.0})
    assert out["post"] == pytest.approx(30.0)      # the uniform factor takes the remainder


def test_rw_theta_walk_forward(tmp_path, monkeypatch):
    f = tmp_path / "rw.json"
    f.write_text(json.dumps({"seasons": {"2023-24": {"theta": 1e-5},
                                         "2025-26": {"theta": 3e-4}}}))
    monkeypatch.setattr(RC, "RW_PATH", f)
    RC.rw_theta.cache_clear()
    try:
        assert RC.rw_theta("2023-24") == 1e-5
        assert RC.rw_theta("2024-25") == 1e-5           # latest earlier season
        assert RC.rw_theta("2026-27") == 3e-4
        assert RC.rw_theta("2019-20") == RC.RW_FALLBACK_THETA
    finally:
        RC.rw_theta.cache_clear()


def test_shipped_params_are_walk_forward():
    if not RC.RW_PATH.exists():
        pytest.skip("no recon_weighted_params.json")
    d = json.loads(RC.RW_PATH.read_text())
    for s, v in d["seasons"].items():
        assert all(t.split(":")[1] < s for t in v["train"]), s
        assert v["theta"] in d["grid"]


# ------------------------------------------------------------------------------ ON, simulate

def test_simulate_on_same_minutes_and_targets(game, restore):  # noqa: F811
    spec_off, mp, sp = _spec(game)
    spec_on, _, _ = _spec(game, evidence=lambda k: np.linspace(0.0, 2500.0, k))
    cp = C.default_copula_params("2023-24")
    P.set_recon_weighted(False)
    a = C.simulate_game(spec_off, cp, mp, sp, 3000, 11)
    P.set_recon_weighted(True)
    b = C.simulate_game(spec_on, cp, mp, sp, 3000, 11)
    assert np.array_equal(a.minutes, b.minutes)                    # same draws
    for t, info in b.teams.items():
        assert info["weighted"] and abs(info["resid"]) <= RC.TOL
    home = b.info["team"].to_numpy() == spec_on.home.team
    assert b.info.loc[home, "pts_all"].sum() == pytest.approx(118.0, abs=0.5)
    assert not np.allclose(a.info["pts_all"], b.info["pts_all"])
    assert np.isfinite(b.info["reb_all"]).all() and (b.fp[b.minutes == 0] == 0).all()


def test_gap_error_logged(game, restore, caplog):  # noqa: F811
    spec, mp, sp = _spec(game, implied=(150.0, 103.0))
    P.set_recon_weighted(True)
    with caplog.at_level(logging.WARNING, logger=RC.__name__):
        s = C.simulate_game(spec, C.default_copula_params("2023-24"), mp, sp, 1500, 3)
    info = s.teams[spec.home.team]
    assert info["data_check"] and info["gap_error"]["gap"] < -RC.GAP_CHECK
    assert len(info["gap_error"]["absorbed"]) == 3
    assert any("projection error" in r.message for r in caplog.records)
    row = RC.check_rows(list(s.teams.values()), "t")[0]
    assert row["status"] == "warn" and "projection_error" in row["detail"]
    json.dumps(info["gap_error"])
