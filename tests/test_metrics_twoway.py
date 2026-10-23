"""paired_cluster2: two-way cluster-robust SE (date x player), consistent with the harness."""

import numpy as np
import pytest

from courtside.proj import backtest as BT
from courtside.proj import metrics as MT


def _data(seed=0, n=600, nd=25, npl=40):
    rng = np.random.default_rng(seed)
    day = rng.integers(0, nd, n)
    pl = rng.integers(0, npl, n)
    d = rng.normal(0.1, 1, n) + rng.normal(0, 0.5, npl)[pl] + rng.normal(0, 0.3, nd)[day]
    return d, day, pl


def test_matches_backtest_two_way():
    for seed in range(5):
        d, day, pl = _data(seed)
        m = d.mean()
        var = BT._two_way_var((d - m) / len(d), day, pl)
        r = MT.paired_cluster2(d, day, pl)
        assert r["se"] == pytest.approx(np.sqrt(var), rel=1e-12)
        assert r["mean"] == pytest.approx(m)
        assert r["n"] == len(d) and r["n_c1"] == len(set(day)) and r["n_c2"] == len(set(pl))


def test_reduces_to_one_way_when_c2_unique():
    d, day, _ = _data(1)
    r = MT.paired_cluster2(d, day, np.arange(len(d)))
    one = MT.paired_cluster(d, day)
    assert r["se"] == pytest.approx(one["se"], rel=1e-12)
    assert r["se_c1"] == pytest.approx(one["se"], rel=1e-12)
    assert r["n_clusters"] == one["n_clusters"]


def test_negative_variance_falls_back_to_max_one_way():
    # c1 and c2 are a perfect 2x2 grid with within-cell sums cancelling in V1, V2 but not V12
    d = np.array([1.0, -1.0, -1.0, 1.0])
    c1 = np.array([0, 0, 1, 1])
    c2 = np.array([0, 1, 0, 1])
    # V1 = V2 = 0 (each cluster sums to 0); V12 = 2 * 4*... > 0 -> V = V1+V2-V12 < 0
    r = MT.paired_cluster2(d, c1, c2)
    assert r["se_c1"] == 0.0 and r["se_c2"] == 0.0
    assert r["se"] == 0.0  # fallback max(V1, V2) = 0, not sqrt of a negative / NaN
    # a case with a positive fallback
    d2 = np.array([2.0, -1.0, -1.0, 1.0, 3.0, -1.0])
    c1b = np.array([0, 0, 1, 1, 2, 2])
    c2b = np.array([0, 1, 0, 1, 2, 3])
    r2 = MT.paired_cluster2(d2, c1b, c2b)
    assert np.isfinite(r2["se"]) and r2["se"] >= 0


def test_fallback_value_is_max_v1_v2():
    # a case where V1 + V2 - V12 < 0 with positive one-way variances
    d = np.array([0.126, -0.132, 0.64, 0.105, -0.536, 0.362, 1.304, 0.947])
    c1 = np.array([1, 1, 1, 2, 0, 2, 2, 0])
    c2 = np.array([1, 2, 1, 0, 2, 2, 2, 0])
    z = (d - d.mean()) / len(d)
    v1, v2 = MT._cluster_var(z, c1), MT._cluster_var(z, c2)
    v12 = MT._cluster_var(z, np.array(["%d|%d" % p for p in zip(c1, c2)]))
    assert v1 + v2 - v12 < 0 and max(v1, v2) > 0
    r = MT.paired_cluster2(d, c1, c2)
    assert r["se"] == pytest.approx(np.sqrt(max(v1, v2)))


def test_empty():
    r = MT.paired_cluster2([], [], [], by=[])
    assert r["n"] == 0 and np.isnan(r["mean"]) and np.isnan(r["se"])
    assert r["n_c1"] == 0 and r["n_c2"] == 0 and r["by_mean"] == {}


def test_single_cluster_is_nan():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    r = MT.paired_cluster2(d, np.zeros(4, int), np.arange(4))
    assert np.isnan(r["se"]) and np.isnan(r["se_c1"]) and r["n_c1"] == 1
    r = MT.paired_cluster2(d, np.arange(4), np.zeros(4, int))
    assert np.isnan(r["se"]) and np.isnan(r["se_c2"]) and r["n_c2"] == 1


def test_by_group_means():
    d, day, pl = _data(2, n=100)
    by = np.where(day < 12, "2023-24", "2024-25")
    r = MT.paired_cluster2(d, day, pl, by=by)
    assert r["by_mean"]["2023-24"] == pytest.approx(d[day < 12].mean())
    assert r["by_mean"]["2024-25"] == pytest.approx(d[day >= 12].mean())
    assert "by_mean" not in MT.paired_cluster2(d, day, pl)
