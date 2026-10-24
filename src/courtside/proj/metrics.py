"""Pure evaluation metrics for projections and lineups (docs/PLAYERS.md §7.2, T2.1).

No database, no pipeline: arrays in, numbers out, so every function is unit-tested on known cases.

  paired_cluster(d, clusters)          mean of paired differences + cluster-robust SE (by date)
  gain_pct(base_err, model_err, cl)    % MAE gain of the model over a baseline, SE in the same %
  rate_se(ind, clusters)               mean of a 0/1 indicator (coverage) + clustered SE
  crps_samples(x, y)                   CRPS of an empirical sample (sorted formula)
  crps_samples_matrix(X, y)            the same for many players at once (X is (S, n))
  crps_normal(mu, sd, y)               closed-form CRPS of a normal forecast
  crps_quantiles(q, levels, y)         CRPS approximated from forecast quantiles (pinball)
  interval_hits(lo, hi, y)             0/1 coverage indicators
  pit_randomized(X, y, v)              randomised PIT for discrete / mixed outcomes (S3)
  pit_histogram(u, bins)               PIT bin shares
  tier_table(df, by, ...)              n, mean prediction, mean actual, MAE, bias, coverage by tier
  fit_group_cv / naive_normal_crps     the "naive mean + tier-CV normal" CRPS reference
  percentile_rank(value, reference)    mid-rank percentile of a lineup among reference lineups
  paired_summary(a, b, clusters)       mean diff, SE, z, win and tie rates
  dfs_population(salary, *proj)        E4: salary >= $4k or projected >= 15 DK (DFS-relevant rows)
  salary_implied(y, salary, slate)     E4: per-slate OLS line of DK points on salary
  corr_cluster(x, y, clusters)         Pearson r with a cluster-robust SE (influence function)
  corr_paired_cluster(xa, xb, y, cl)   r_a - r_b on the same rows, clustered SE of the difference
  value_ranking(preds, y, salary, ...) E4: corr(pred - salary-implied, actual - salary-implied)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# ================================================================== clustered statistics


def paired_cluster(d: Sequence[float] | np.ndarray, clusters: Sequence | np.ndarray) -> dict:
    """Mean of the paired differences `d` with the cluster-robust SE of the mean:
    sqrt(G/(G-1) * sum_g (sum_{i in g}(d_i - dbar))^2) / n  (one-way, e.g. by date)."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0, "n_clusters": 0}
    cl = np.asarray(clusters)
    mean = float(d.mean())
    s = pd.Series(d - mean).groupby(cl).sum().to_numpy()
    g = len(s)
    se = float(np.sqrt(g / (g - 1) * (s ** 2).sum()) / n) if g > 1 else float("nan")
    return {"mean": mean, "se": se, "n": int(n), "n_clusters": int(g)}


def gain_pct(base_err: np.ndarray, model_err: np.ndarray, clusters) -> dict:
    """% improvement of mean |model error| over mean |baseline error| on the same rows, with the
    clustered SE of the paired difference expressed in the same %."""
    base_err = np.asarray(base_err, float)
    model_err = np.asarray(model_err, float)
    pc = paired_cluster(base_err - model_err, clusters)
    b = float(base_err.mean()) if len(base_err) else float("nan")
    return {"gain_pct": 100.0 * pc["mean"] / b, "se_pct": 100.0 * pc["se"] / b,
            "mae_model": float(model_err.mean()) if len(model_err) else float("nan"),
            "mae_base": b, "n": pc["n"], "n_clusters": pc["n_clusters"]}


def rate_se(ind: np.ndarray, clusters) -> dict:
    """Mean of a 0/1 indicator (coverage) with the cluster-robust SE."""
    pc = paired_cluster(np.asarray(ind, float), clusters)
    return {"rate": pc["mean"], "se": pc["se"], "n": pc["n"]}


def paired_summary(a: np.ndarray, b: np.ndarray, clusters) -> dict:
    """a - b per row: mean, clustered SE, z, share of rows a > b (wins) and a == b (ties)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    pc = paired_cluster(diff, clusters)
    z = pc["mean"] / pc["se"] if pc["se"] and np.isfinite(pc["se"]) else None
    return {"mean_diff": pc["mean"], "se": pc["se"], "z": z,
            "win_rate": float((diff > 0).mean()) if len(diff) else float("nan"),
            "tie_rate": float((diff == 0).mean()) if len(diff) else float("nan"),
            "n": pc["n"], "n_clusters": pc["n_clusters"]}


# ================================================================== CRPS


def crps_samples(x: np.ndarray, y: float) -> float:
    """CRPS of an empirical sample x for outcome y: E|X - y| - E|X - X'|/2 (sorted formula)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    e1 = np.abs(x - y).mean()
    i = np.arange(1, n + 1)
    e2 = 2.0 * ((2 * i - n - 1) * x).sum() / (n * n)
    return float(e1 - 0.5 * e2)


def crps_samples_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`crps_samples` for each column of X (S, n) against y (n,)."""
    X = np.sort(np.asarray(X, float), axis=0)
    y = np.asarray(y, float)
    s = X.shape[0]
    e1 = np.abs(X - y[None, :]).mean(0)
    i = np.arange(1, s + 1)[:, None]
    e2 = 2.0 * ((2 * i - s - 1) * X).sum(0) / (s * s)
    return e1 - 0.5 * e2


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _norm_cdf(z):
    from scipy.special import ndtr
    return ndtr(np.asarray(z, float))


def crps_normal(mu, sd, y) -> np.ndarray:
    """Closed-form CRPS of N(mu, sd^2) at y: sd * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))."""
    mu, sd, y = (np.asarray(v, float) for v in (mu, sd, y))
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return sd * (z * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * _norm_pdf(z) - 1.0 / np.sqrt(np.pi))


def crps_quantiles(q: np.ndarray, levels: Sequence[float], y: np.ndarray) -> np.ndarray:
    """CRPS approximated from forecast quantiles: 2 * mean over levels of the pinball loss
    (CRPS = 2 * integral over tau of the quantile score). q is (n, k) at `levels` (k), y (n,).
    Accurate when the levels are dense; with 5 levels it is a coarse, slightly low estimate."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)[:, None]
    tau = np.asarray(levels, float)[None, :]
    d = y - q
    pin = np.where(d >= 0, tau * d, (tau - 1.0) * d)
    return 2.0 * pin.mean(axis=1)


# ================================================================== intervals and PIT


def interval_hits(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    lo, hi, y = (np.asarray(v, float) for v in (lo, hi, y))
    return ((lo <= y) & (y <= hi)).astype(float)


def pit_randomized(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Randomised PIT u = F(y-) + V * (F(y) - F(y-)) with F the empirical CDF of each column of
    X (S, n) and V ~ U(0,1) (`v`, length n). Handles the point mass at 0 of a DNP and the
    discreteness of box-score points; uniform on [0,1] when the forecast is calibrated."""
    X = np.asarray(X)
    y = np.asarray(y, float)
    s = X.shape[0]
    f_lo = (X < y[None, :]).sum(0) / s
    f_hi = (X <= y[None, :]).sum(0) / s
    return f_lo + np.asarray(v, float) * (f_hi - f_lo)


def pit_histogram(u: np.ndarray, bins: int = 10) -> np.ndarray:
    """Share of PIT values in each of `bins` equal bins (all 1/bins when calibrated)."""
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    if not len(u):
        return np.full(bins, np.nan)
    h, _ = np.histogram(np.clip(u, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return h / len(u)


# ================================================================== tables


def tier_table(df: pd.DataFrame, by: str, *, pred: str = "mean", y: str = "y",
               lo80: str | None = "q10", hi80: str | None = "q90", lo50: str | None = "q25",
               hi50: str | None = "q75", base: str | None = None) -> pd.DataFrame:
    """Per group of column `by`: n, mean prediction, mean actual, bias (pred - actual), MAE,
    optional MAE of a baseline column on the same rows, and 80/50% coverage."""
    rows = []
    for name, g in df.groupby(by, observed=True):
        yy = g[y].to_numpy(float)
        pp = g[pred].to_numpy(float)
        r = {by: name, "n": int(len(g)), "pred": float(pp.mean()), "actual": float(yy.mean()),
             "bias": float((pp - yy).mean()), "mae": float(np.abs(pp - yy).mean())}
        if base is not None and base in g:
            b = g[base].to_numpy(float)
            fin = np.isfinite(b)
            r["mae_base"] = float(np.abs(b[fin] - yy[fin]).mean()) if fin.any() else float("nan")
        if lo80 and hi80 and lo80 in g:
            r["cov80"] = float(interval_hits(g[lo80], g[hi80], yy).mean())
        if lo50 and hi50 and lo50 in g:
            r["cov50"] = float(interval_hits(g[lo50], g[hi50], yy).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def fit_group_cv(groups: np.ndarray, mean: np.ndarray, y: np.ndarray, *, min_n: int = 30
                 ) -> dict:
    """Coefficient of variation by group, sd(y - mean) / mean(mean), fit on the rows given
    (callers pass seasons < S only). Groups with fewer than `min_n` rows get the pooled value
    under the key None."""
    groups = np.asarray(groups)
    mean = np.asarray(mean, float)
    y = np.asarray(y, float)
    ok = np.isfinite(mean) & np.isfinite(y)
    out: dict = {}
    if ok.sum() >= 2:
        out[None] = float(np.std(y[ok] - mean[ok]) / max(mean[ok].mean(), 1e-9))
    for g in pd.unique(groups[ok]):
        m = ok & (groups == g)
        if m.sum() >= min_n:
            out[g] = float(np.std(y[m] - mean[m]) / max(mean[m].mean(), 1e-9))
    return out


def naive_normal_crps(groups: np.ndarray, mean: np.ndarray, y: np.ndarray, cv: dict) -> np.ndarray:
    """CRPS of the naive forecast N(mean, (cv[group] * mean)^2); NaN when `cv` is empty."""
    if not cv:
        return np.full(len(mean), np.nan)
    pooled = cv.get(None, np.nan)
    c = np.array([cv.get(g, pooled) for g in groups], float)
    mean = np.asarray(mean, float)
    return crps_normal(mean, c * np.maximum(mean, 1.0), y)


# ================================================================== lineups


def percentile_rank(value: float, reference: np.ndarray) -> float:
    """Mid-rank percentile in [0, 1]: P(ref < value) + 0.5 P(ref == value)."""
    r = np.asarray(reference, float)
    if not len(r):
        return float("nan")
    return float(((r < value).sum() + 0.5 * (r == value).sum()) / len(r))


def percentile_ranks(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`percentile_rank` of each of `values` against one reference sample (vectorised)."""
    r = np.sort(np.asarray(reference, float))
    v = np.asarray(values, float)
    if not len(r):
        return np.full(len(v), np.nan)
    lo = np.searchsorted(r, v, side="left")
    hi = np.searchsorted(r, v, side="right")
    return (lo + 0.5 * (hi - lo)) / len(r)


# ================================================================== two-way clustered statistics


def _cluster_var(z: np.ndarray, labels: np.ndarray) -> float:
    """G/(G-1) * sum_g (sum_{i in g} z_i)^2 (NaN with a single cluster)."""
    s = pd.Series(z).groupby(labels).sum().to_numpy()
    g = len(s)
    return float(g / (g - 1) * (s ** 2).sum()) if g > 1 else float("nan")


def paired_cluster2(d: Sequence[float] | np.ndarray, c1: Sequence | np.ndarray,
                    c2: Sequence | np.ndarray, by: Sequence | np.ndarray | None = None) -> dict:
    """Mean of `d` with the two-way cluster-robust SE of the mean (Cameron-Gelbach-Miller, e.g.
    c1 = date, c2 = player): V = V1 + V2 - V12, V12 clustered on the (c1, c2) intersection, the
    same math as backtest._two_way_var. If V <= 0 it falls back to max(V1, V2). SE is NaN when
    either dimension has a single cluster. Keys as `paired_cluster` (n_clusters = n_c1) plus
    se_c1, se_c2 (the one-way SEs), n_c1, n_c2; with `by` (e.g. season labels) also `by_mean`,
    a {label: mean of d} dict, so callers can report the per-group sign."""
    d = np.asarray(d, float)
    n = len(d)
    nan = float("nan")
    if n == 0:
        out = {"mean": nan, "se": nan, "n": 0, "n_clusters": 0, "se_c1": nan, "se_c2": nan,
               "n_c1": 0, "n_c2": 0}
        if by is not None:
            out["by_mean"] = {}
        return out
    a1, a2 = np.asarray(c1), np.asarray(c2)
    mean = float(d.mean())
    z = (d - mean) / n
    both = pd.Series(a1).astype(str) + "|" + pd.Series(a2).astype(str)
    v1, v2 = _cluster_var(z, a1), _cluster_var(z, a2)
    v12 = _cluster_var(z, both.to_numpy())
    v = v1 + v2 - v12
    if np.isfinite(v) and v <= 0:
        v = max(v1, v2)
    out = {"mean": mean, "se": float(np.sqrt(v)) if np.isfinite(v) else nan, "n": int(n),
           "n_clusters": int(pd.Series(a1).nunique()),
           "se_c1": float(np.sqrt(v1)) if np.isfinite(v1) else nan,
           "se_c2": float(np.sqrt(v2)) if np.isfinite(v2) else nan,
           "n_c1": int(pd.Series(a1).nunique()), "n_c2": int(pd.Series(a2).nunique())}
    if by is not None:
        out["by_mean"] = {k: float(m) for k, m in pd.Series(d).groupby(np.asarray(by)).mean().items()}
    return out


# ================================================================== DFS-relevant rows and value ranking (E4)

DFS_SAL_MIN = 4000.0       # E4: salary >= $4k ...
DFS_PROJ_MIN = 15.0        # ... or projected >= 15 DK points (all-row CRPS is diluted by bench zeros)


def dfs_population(salary, *proj, sal_min: float = DFS_SAL_MIN,
                   proj_min: float = DFS_PROJ_MIN) -> np.ndarray:
    """E4's DFS-relevant rows: salary >= `sal_min` (where a salary is known) OR any of the given
    projections >= `proj_min`. Pass every forecast being compared (ours, DK FPPG, an arm and the
    champion) so a paired comparison is not selected on one side's own numbers."""
    sal = np.asarray(salary, float) if salary is not None else None
    out = None
    if sal is not None:
        out = np.isfinite(sal) & (sal >= sal_min)
    for p in proj:
        p = np.asarray(p, float)
        hit = np.isfinite(p) & (p >= proj_min)
        out = hit if out is None else (out | hit)
    if out is None:
        raise ValueError("dfs_population needs a salary or a projection")
    return out


def salary_implied(y, salary, groups, *, min_rows: int = 5) -> np.ndarray:
    """Per group (slate): the OLS line of DK points `y` on `salary`, evaluated at each row's
    salary. NaN where the salary or y is missing, or the group has fewer than `min_rows` usable
    rows / no salary spread. It is fit on the actual points of the slate (hindsight), so it is the
    same line for every forecast being compared."""
    y = np.asarray(y, float)
    sal = np.asarray(salary, float)
    g = np.asarray(groups)
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y) & np.isfinite(sal)
    for key in pd.unique(g[ok]):
        m = ok & (g == key)
        if m.sum() < min_rows:
            continue
        x = sal[m]
        if np.ptp(x) <= 0:
            continue
        b, a = np.polyfit(x, y[m], 1)
        out[m] = a + b * x
    return out


def _corr_if(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Pearson r and its per-row influence function u v - r (u^2 + v^2) / 2 (u, v standardised
    with the sample means and population sds); mean(IF) = 0, var(r) ~ var(sum IF) / n^2."""
    u = (x - x.mean()) / x.std()
    v = (y - y.mean()) / y.std()
    r = float((u * v).mean())
    return r, u * v - 0.5 * r * (u * u + v * v)


def corr_cluster(x, y, clusters) -> dict:
    """Pearson correlation of x and y with the cluster-robust SE (influence-function / delta
    method, clusters e.g. dates). NaN SE with fewer than 2 clusters."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], cl[ok]
    nan = float("nan")
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return {"r": nan, "se": nan, "n": int(len(x)),
                "n_clusters": int(pd.Series(cl).nunique()) if len(cl) else 0}
    r, inf = _corr_if(x, y)
    pc = paired_cluster(inf, cl)
    return {"r": r, "se": pc["se"], "n": pc["n"], "n_clusters": pc["n_clusters"]}


def corr_paired_cluster(xa, xb, y, clusters) -> dict:
    """corr(xa, y) - corr(xb, y) on the rows where all three are finite, with the clustered SE of
    the difference (the two influence functions differenced row by row, so the shared y and
    the shared clusters are accounted for). Also each r and its own SE, and z."""
    xa, xb, y = (np.asarray(v, float) for v in (xa, xb, y))
    cl = np.asarray(clusters)
    ok = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y, cl = xa[ok], xb[ok], y[ok], cl[ok]
    nan = float("nan")
    out = {"r_a": nan, "r_b": nan, "se_a": nan, "se_b": nan, "diff": nan, "se": nan, "z": None,
           "n": int(len(y)), "n_clusters": int(pd.Series(cl).nunique()) if len(cl) else 0}
    if len(y) < 3 or xa.std() == 0 or xb.std() == 0 or y.std() == 0:
        return out
    ra, ia = _corr_if(xa, y)
    rb, ib = _corr_if(xb, y)
    pa, pb, pd_ = paired_cluster(ia, cl), paired_cluster(ib, cl), paired_cluster(ia - ib, cl)
    se = pd_["se"]
    out.update({"r_a": ra, "r_b": rb, "se_a": pa["se"], "se_b": pb["se"], "diff": ra - rb,
                "se": se, "z": (ra - rb) / se if np.isfinite(se) and se > 0 else None})
    return out


def value_ranking(preds: dict, y, salary, slate, clusters, *, base: str | None = None,
                  min_rows: int = 5) -> dict:
    """E4 value-ranking metric. Per slate the salary-implied DK points s (`salary_implied`, OLS
    of actual DK on salary over the given rows); for each forecast p in `preds`
    ({name: array}) r = corr(p - s, y - s) pooled over every slate with the clustered SE
    (`corr_cluster`, clusters e.g. dates: players of a night share games), plus the mean of the
    per-slate correlations. With `base`, every other forecast minus `base` (paired, same rows:
    `corr_paired_cluster`). Rows with any forecast, y, salary or s missing are dropped, so all
    forecasts are scored on identical rows."""
    y = np.asarray(y, float)
    sal = np.asarray(salary, float)
    slate = np.asarray(slate)
    cl = np.asarray(clusters)
    P = {k: np.asarray(v, float) for k, v in preds.items()}
    ok = np.isfinite(y) & np.isfinite(sal)
    for v in P.values():
        ok &= np.isfinite(v)
    s = np.full(len(y), np.nan)
    s[ok] = salary_implied(y[ok], sal[ok], slate[ok], min_rows=min_rows)
    ok &= np.isfinite(s)
    ry = y[ok] - s[ok]
    out: dict = {"n": int(ok.sum()), "n_slates": int(pd.Series(slate[ok]).nunique()),
                 "n_clusters": int(pd.Series(cl[ok]).nunique()), "forecasts": {}, "vs_base": {}}
    for k, v in P.items():
        rv = v[ok] - s[ok]
        c = corr_cluster(rv, ry, cl[ok])
        per = []
        for key in pd.unique(slate[ok]):
            m = slate[ok] == key
            if m.sum() >= 3 and rv[m].std() > 0 and ry[m].std() > 0:
                per.append(float(np.corrcoef(rv[m], ry[m])[0, 1]))
        out["forecasts"][k] = {**c, "slate_mean_r": float(np.mean(per)) if per else float("nan"),
                               "slate_mean_se": (float(np.std(per, ddof=1) / np.sqrt(len(per)))
                                                 if len(per) > 1 else float("nan")),
                               "n_slates_r": len(per)}
    if base is not None and base in P:
        for k, v in P.items():
            if k == base:
                continue
            out["vs_base"][k] = {"base": base, **corr_paired_cluster(
                v[ok] - s[ok], P[base][ok] - s[ok], ry, cl[ok])}
    return out
