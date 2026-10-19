"""proj/metrics.py: pure evaluation metrics on known cases (no DB, no pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from courtside.proj import evallite as E
from courtside.proj import metrics as MT


# --------------------------------------------------------------------------- clustered stats

def test_stats_are_shared_with_evallite():
    assert E.paired_cluster is MT.paired_cluster
    assert E.gain_pct is MT.gain_pct and E.rate_se is MT.rate_se


def test_paired_cluster_known_case():
    d = np.array([1.0, 3.0, -2.0, 4.0, 0.0, 2.0])
    cl = np.array(["a", "a", "b", "b", "c", "c"])
    r = MT.paired_cluster(d, cl)
    s = np.array([(1 - d.mean()) + (3 - d.mean()), (-2 - d.mean()) + (4 - d.mean()),
                  (0 - d.mean()) + (2 - d.mean())])
    assert r["mean"] == pytest.approx(d.mean())
    assert r["se"] == pytest.approx(np.sqrt(3 / 2 * (s ** 2).sum()) / 6)
    assert MT.paired_cluster([], [])["n"] == 0
    assert np.isnan(MT.paired_cluster([1.0, 2.0], ["a", "a"])["se"])      # one cluster


def test_gain_pct_and_rate_se():
    g = MT.gain_pct([10.0] * 4, [8.0] * 4, [1, 1, 2, 2])
    assert g["gain_pct"] == pytest.approx(20.0) and g["mae_model"] == 8.0
    r = MT.rate_se([1, 0, 1, 1], [1, 1, 2, 2])
    assert r["rate"] == pytest.approx(0.75) and r["n"] == 4


def test_paired_summary_wins_ties():
    s = MT.paired_summary([3, 2, 5, 1], [1, 2, 6, 0], [1, 2, 3, 4])
    assert s["win_rate"] == 0.5 and s["tie_rate"] == 0.25
    assert s["mean_diff"] == pytest.approx((2 + 0 - 1 + 1) / 4)
    assert s["n"] == 4 and s["z"] is not None


# --------------------------------------------------------------------------- CRPS

def test_crps_samples_known_cases():
    assert MT.crps_samples(np.array([5.0, 5.0, 5.0]), 8.0) == pytest.approx(3.0)   # point forecast
    # {0, 2} against 1: E|X-y| = 1, E|X-X'| = 1 -> 0.5
    assert MT.crps_samples(np.array([0.0, 2.0]), 1.0) == pytest.approx(0.5)
    # order does not matter
    assert MT.crps_samples(np.array([2.0, 0.0]), 1.0) == pytest.approx(0.5)


def test_crps_samples_matches_normal_closed_form():
    sd = 4.0
    x = stats.norm.ppf((np.arange(4000) + 0.5) / 4000, loc=20, scale=sd)
    for y in (12.0, 20.0, 31.0):
        assert MT.crps_samples(x, y) == pytest.approx(float(MT.crps_normal(20, sd, y)), rel=2e-3)
    assert float(MT.crps_normal(0, 1, 0)) == pytest.approx(0.23369, abs=1e-4)


def test_crps_samples_matrix_equals_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(15, 5, (300, 6))
    y = rng.normal(15, 5, 6)
    got = MT.crps_samples_matrix(X, y)
    want = [MT.crps_samples(X[:, j], y[j]) for j in range(6)]
    assert np.allclose(got, want)
    from courtside.proj import pipeline as P
    assert MT.crps_samples(X[:, 0], y[0]) == pytest.approx(P.crps_samples(X[:, 0], y[0]))


def test_crps_quantiles_approximates_normal_crps():
    levels = (np.arange(99) + 1) / 100
    mu, sd = np.array([20.0, 35.0]), np.array([5.0, 8.0])
    q = stats.norm.ppf(levels[None, :], loc=mu[:, None], scale=sd[:, None])
    y = np.array([18.0, 40.0])
    assert np.allclose(MT.crps_quantiles(q, levels, y), MT.crps_normal(mu, sd, y), rtol=0.03)
    # a degenerate forecast: every quantile at 7 -> CRPS = |7 - y|
    assert MT.crps_quantiles(np.full((1, 5), 7.0), (.1, .25, .5, .75, .9), np.array([10.0])
                             )[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------- coverage and PIT

def test_interval_hits_inclusive_and_coverage_of_a_calibrated_normal():
    assert list(MT.interval_hits([0, 0, 0], [1, 1, 1], [0, 1, 1.01])) == [1.0, 1.0, 0.0]
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 20000)
    z = stats.norm.ppf(0.9)
    assert MT.interval_hits(np.full(y.size, -z), np.full(y.size, z), y).mean() == pytest.approx(
        0.8, abs=0.01)


def test_pit_extremes_and_ties():
    X = np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), (1, 3))
    y = np.array([0.0, 2.0, 9.0])
    u = MT.pit_randomized(X, y, np.array([0.5, 0.5, 0.5]))
    assert u[0] == 0.0 and u[2] == 1.0
    assert u[1] == pytest.approx(0.25 + 0.5 * 0.25)        # F(2-)=.25, F(2)=.5


def test_randomised_pit_is_uniform_for_discrete_outcomes_with_a_point_mass():
    rng = np.random.default_rng(2)
    n, S = 6000, 400
    # forecast: 30% DNP (0) else Poisson(25); outcomes drawn from the same law
    def draw(size):
        return np.where(rng.random(size) < 0.3, 0.0, rng.poisson(25, size).astype(float))
    X = draw((S, n))
    y = draw(n)
    h = MT.pit_histogram(MT.pit_randomized(X, y, rng.random(n)), 10)
    assert h.sum() == pytest.approx(1.0) and np.all(np.abs(h - 0.1) < 0.02)
    # without randomisation the DNP mass piles into one bin (why S3 is needed)
    plain = MT.pit_histogram((X < y[None, :]).mean(0), 10)
    assert plain[0] > 0.25


def test_pit_detects_a_biased_forecast():
    rng = np.random.default_rng(3)
    n, S = 4000, 300
    X = rng.normal(20, 5, (S, n))
    y = rng.normal(25, 5, n)                    # outcomes run high
    h = MT.pit_histogram(MT.pit_randomized(X, y, rng.random(n)), 10)
    assert h[-1] > 0.2 and h[0] < 0.05
    assert np.isnan(MT.pit_histogram(np.array([np.nan]), 10)).all()


# --------------------------------------------------------------------------- tables

def test_tier_table_known_case():
    df = pd.DataFrame({"tier": ["a", "a", "b", "b"], "mean": [10.0, 20.0, 30.0, 30.0],
                       "y": [12.0, 18.0, 30.0, 36.0], "q10": [5.0, 10.0, 25.0, 25.0],
                       "q90": [15.0, 25.0, 35.0, 35.0], "q25": [8.0, 15.0, 28.0, 28.0],
                       "q75": [12.0, 22.0, 32.0, 32.0], "base": [11.0, 11.0, 30.0, 30.0]})
    t = MT.tier_table(df, "tier", base="base").set_index("tier")
    assert t.loc["a", "n"] == 2 and t.loc["a", "mae"] == pytest.approx(2.0)
    assert t.loc["a", "bias"] == pytest.approx(0.0) and t.loc["b", "bias"] == pytest.approx(-3.0)
    assert t.loc["a", "mae_base"] == pytest.approx((1 + 7) / 2)
    assert t.loc["a", "cov80"] == 1.0 and t.loc["b", "cov80"] == 0.5
    assert t.loc["a", "cov50"] == 1.0 and t.loc["b", "cov50"] == 0.5


def test_naive_normal_reference_uses_only_the_rows_it_is_fit_on():
    rng = np.random.default_rng(4)
    g = np.array(["star"] * 300 + ["bench"] * 300)
    mean = np.where(g == "star", 35.0, 10.0)
    y = mean + rng.normal(0, 1, 600) * np.where(g == "star", 0.3, 0.6) * mean
    cv = MT.fit_group_cv(g, mean, y)
    assert cv["star"] == pytest.approx(0.3, abs=0.04) and cv["bench"] == pytest.approx(0.6, abs=0.06)
    assert None in cv                                   # pooled fallback
    c = MT.naive_normal_crps(np.array(["star", "unknown"]), np.array([35.0, 10.0]),
                             np.array([35.0, 10.0]), cv)
    assert c[0] == pytest.approx(0.2337 * 0.3 * 35, rel=0.15) and c[1] > 0
    assert np.isnan(MT.naive_normal_crps(np.array(["a"]), np.array([5.0]), np.array([5.0]), {})).all()


def test_percentile_rank_mid_rank():
    ref = np.array([1.0, 2.0, 2.0, 3.0])
    assert MT.percentile_rank(0.0, ref) == 0.0 and MT.percentile_rank(9.0, ref) == 1.0
    assert MT.percentile_rank(2.0, ref) == pytest.approx((1 + 0.5 * 2) / 4)
    v = np.array([0.0, 2.0, 2.5, 9.0])
    assert np.allclose(MT.percentile_ranks(v, ref), [MT.percentile_rank(x, ref) for x in v])
    assert np.isnan(MT.percentile_rank(1.0, np.array([])))


def test_tier_table_baseline_mae_ignores_missing_baselines():
    df = pd.DataFrame({"tier": ["a"] * 3, "mean": [10.0, 10.0, 10.0], "y": [10.0, 12.0, 14.0],
                       "base": [8.0, np.nan, 10.0], "q10": 0.0, "q90": 99.0, "q25": 0.0, "q75": 99.0})
    t = MT.tier_table(df, "tier", base="base")
    assert t["mae_base"].iloc[0] == pytest.approx((2 + 4) / 2)
