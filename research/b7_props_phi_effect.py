"""B7: how much does fitting the props PHI on seasons < S (instead of 2023-26) move the stored
per-stat calibration of the props approximation (docs/PLAYERS.md "Per-stat calibration")?

Reads the copula_v1_eval per-date `*.stats.parquet` (rows: y, mean_approx, line_std, pit_approx)
and re-scores each row's NB1 with the live PHI vs the season's out-of-sample PHI, on the stored
means, with the same uniform PIT draw v. The run's minutes draws are not stored, so the pmf is
NB1 at the stored mean with no minutes mixture (`stat_pmf(mu, None, phi)`); the absolute level
therefore differs from the stored pit_approx, the DELTA (live PHI -> OOS PHI) is what this reports.

    python research/b7_props_phi_effect.py STATS_DIR/2023-24 PHI_OOS_JSON   # {"pts": 2.355, ...}
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

from courtside.proj import props as PR

STATS = ("pts", "reb", "ast", "fg3m")


def pit(pmf: np.ndarray, y: int, v: float) -> float:
    cdf = np.cumsum(pmf)
    lo = float(cdf[y - 1]) if 0 < y <= len(cdf) else (0.0 if y <= 0 else 1.0)
    hi = float(cdf[y]) if 0 <= y < len(cdf) else 1.0
    return lo + v * (hi - lo)


def spiegelhalter(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(((y - p) * (1 - 2 * p)).sum() / np.sqrt((((1 - 2 * p) ** 2) * p * (1 - p)).sum()))


def main(argv=None) -> None:
    d, phi_json = sys.argv[1:3]
    oos = json.loads(phi_json)
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/*.stats.parquet"))],
                   ignore_index=True)
    rng = np.random.default_rng(1)
    out = {}
    for s in STATS:
        x = df[(df["stat"] == s) & np.isfinite(df["mean_approx"])]
        v = rng.random(len(x))
        res = {"n": int(len(x))}
        for name, phi in (("live", PR.PHI[s]), ("oos", oos[s])):
            u, ps, ys, ll = [], [], [], []
            cache = {}
            for mu, y, L, vv in zip(x["mean_approx"], x["y"], x["line_std"], v):
                k = round(float(mu), 2)
                pmf = cache.get(k)
                if pmf is None:
                    pmf = cache[k] = PR.stat_pmf(max(k, 1e-3), None, phi)
                u.append(pit(pmf, int(y), vv))
                if np.isfinite(L):
                    ps.append(PR.line_probs(pmf, float(L))[0])
                    ys.append(float(y > L))
            h = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
            ps, ys = np.array(ps), np.array(ys)
            res[name] = {"phi": phi, "pit_maxdev": float(np.abs(h - 0.1).max()),
                         "p_std": float(ps.mean()), "freq": float(ys.mean()),
                         "z": spiegelhalter(ps, ys),
                         "logloss": float(-np.mean(ys * np.log(np.clip(ps, 1e-6, 1))
                                                   + (1 - ys) * np.log(np.clip(1 - ps, 1e-6, 1)))),
                         "pit": [round(float(a), 4) for a in h]}
        res["stored_pit_maxdev_with_minutes"] = float(np.abs(np.histogram(
            x["pit_approx"], bins=10, range=(0, 1))[0] / len(x) - 0.1).max())
        out[s] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
