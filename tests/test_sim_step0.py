"""T2.5: synthetic checks of the sim step-0 estimators (research/sim_step0.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "research" / "sim_step0.py"
_spec = importlib.util.spec_from_file_location("sim_step0", _PATH)
step0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step0)


def _record(bh, ba, spread=0.0, total=220.0, periods=4):
    z = np.zeros(9)
    return {"bh": np.asarray(bh, float), "ba": np.asarray(ba, float),
            "final_h": float(bh[-1]), "final_a": float(ba[-1]), "periods": periods,
            "poss_h": z + 12, "poss_a": z + 12,
            "pp": {t: np.array([1.0, 2.0]) for t in step0.START_TYPES},
            "reb": {k: {c: 1 for c in step0.REB_CATS} for k in ("fg", "ft")},
            "spread": spread, "total": total}


def _synthetic_games(n, rho, b_final, seed=0):
    """Segment margins with dm_k = rho * m_{k-1} + line drift + noise; final = b * m3 + noise."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        spread = float(rng.choice(np.arange(-12, 12.5, 0.5)))
        e = -spread
        m = np.zeros(9)
        for k in range(1, 9):
            rest = rho * m[k - 1] if k >= 2 else 0.0
            m[k] = m[k - 1] + rest + e / 8 + rng.normal(0, 4.5)
        # overwrite the final margin with the b4 model (after Q3 the rho-walk is not used for b4)
        m[8] = b_final * m[6] + 0.25 * e + rng.normal(0, 7.0)
        bh = np.linspace(0, 110, 9) + m / 2
        ba = np.linspace(0, 110, 9) - m / 2
        recs.append(_record(np.round(bh * 2) / 2, np.round(ba * 2) / 2, spread=spread))
    return recs


def _evaluate(recs, B=0):
    F = pd.DataFrame([step0.game_features(r) for r in recs]).fillna(0.0)
    return step0.evaluate(F, np.array(["s"] * len(F)), B)


def test_boundary_margins_take_last_event_at_or_before_boundary():
    el = np.array([0, 100, 360, 360.5, 700, 720, 1500, 2880])
    sh = np.array([0, 2, 5, 5, 9, 10, 30, 100])
    sa = np.array([0, 0, 3, 6, 6, 8, 25, 90])
    h, a = step0.boundary_margins(el, sh, sa)
    assert h[:3].tolist() == [0, 5, 10]          # 360 includes the event at exactly 6:00
    assert a[:3].tolist() == [0, 3, 8]
    assert h[4] == 10 and h[5] == 30 and h[8] == 100 and a[8] == 90


def test_segment_of():
    assert step0.segment_of(0.5) == 0
    assert step0.segment_of(360.0) == 0
    assert step0.segment_of(360.1) == 1
    assert step0.segment_of(2880.0) == 7
    assert step0.segment_of(2900.0) == 8


@pytest.mark.parametrize("rho", [-0.08, 0.0])
def test_restoring_force_recovers_rho(rho):
    recs = _synthetic_games(6000, rho=rho, b_final=0.9, seed=1)
    res = _evaluate(recs)
    for k in range(2, 8):                         # segment 8 uses the overwritten final
        assert res[f"rho_seg{k}"][0] == pytest.approx(rho, abs=0.03)


def test_b4_recovers_slope_and_line_coef():
    recs = _synthetic_games(6000, rho=0.0, b_final=0.85, seed=2)
    res = _evaluate(recs)
    assert res["b4_regulation"][0] == pytest.approx(0.85, abs=0.03)
    assert res["b4"][0] == pytest.approx(0.85, abs=0.03)
    assert res["b4_line_coef"][0] == pytest.approx(0.25, abs=0.05)
    assert res["b4_resid_sd"][0] == pytest.approx(7.0, abs=0.3)


def test_bootstrap_ci_covers_and_line_variances():
    rng = np.random.default_rng(3)
    recs = []
    for _ in range(3000):
        # H and A residuals with Var 125 and Cov 31 around a 220 total / pick'em line
        rh, ra = rng.multivariate_normal([0, 0], [[125, 31], [31, 125]])
        H, A = 110 + rh, 110 + ra
        recs.append(_record(np.linspace(0, H, 9), np.linspace(0, A, 9)))
    res = _evaluate(recs, B=200)
    v, lo, hi, _ = res["var_team_line"]
    assert lo < 125 < hi and lo < v < hi
    c, lo, hi, _ = res["cov_HA_line"]
    assert lo < 31 < hi
    assert res["sd_margin_line"][0] == pytest.approx(np.sqrt(2 * (125 - 31)), rel=0.05)


def test_leader_bins_pool_every_segment_of_a_game():
    """Each game contributes every segment start whose |m| falls in the bin (not just the last)."""
    # m = 0, 2, 4, 3, 1, 8, ... : segments 2..5 start in the 0-5 bin with leader dm = +2, -1, -2, +7
    m = np.array([0, 2, 4, 3, 1, 8, 8, 8, 8], float)
    rec = _record(110 + m / 2, 110 - m / 2)
    rec["bh"][0] = rec["ba"][0] = 0.0
    f = step0.game_features(rec)
    assert f["lead_q13_0_n"] == 4                  # starts at k = 1..4 (m = 2, 4, 3, 1)
    assert f["lead_q13_0_d"] == pytest.approx(2 - 1 - 2 + 7)
    assert f["lead_q13_1_n"] == 1                  # k = 5, m = 8 -> dm 0
    assert f["lead_q4_1_n"] == 2                   # k = 6, 7 (m = 8)


def test_total_restoring_force_recovers_slope():
    rng = np.random.default_rng(4)
    recs = []
    for _ in range(5000):
        tot = float(rng.choice(np.arange(205, 240, 0.5)))
        T = np.zeros(9)
        for k in range(1, 9):
            dev = T[k - 1] - tot * (k - 1) / 8
            T[k] = T[k - 1] + tot / 8 + (-0.1 * dev if k >= 2 else 0) + rng.normal(0, 6)
        recs.append(_record(T / 2, T / 2, total=tot))
    res = _evaluate(recs)
    assert res["rhoT_pooled"][0] == pytest.approx(-0.1, abs=0.02)
