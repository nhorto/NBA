"""B7: restate the p2-vs-v1 slice comparison with out-of-sample fringe params.

Same rule as the stored `courtside-compute/pq/cmp.py` (the −6.9 / −7.4 SE of HANDOFF): rows of the
T2.4 slice present in both arms with a CRPS, paired difference of the second arm minus the first,
SE clustered by date (one-way) -- plus the two-way (date x player) SE of `backtest._two_way_var`
and, as a fixed population, rows with E[min] >= 10 in either arm. Negative = the second arm better.

    python research/b7_p2_compare.py V1_PARQUET_DIR_OR_FILE ... : see main()
    python research/b7_p2_compare.py live V1_DIR P2_IN_DIR P2_OOS_DIR     # per protocol
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from courtside.proj import backtest as B

K = ["date", "player_id", "game_id"]


def one_way(d: np.ndarray, date: np.ndarray) -> tuple[float, float]:
    s = pd.Series(d).groupby(date).agg(["sum", "size"])
    mu = float(s["sum"].sum() / s["size"].sum())
    r = s["sum"] - mu * s["size"]
    return mu, float(np.sqrt((r ** 2).sum()) / s["size"].sum())


def two_way(d: np.ndarray, date: np.ndarray, pid: np.ndarray) -> tuple[float, float]:
    m = float(d.mean())
    var = B._two_way_var((d - m) / len(d), date, pid)
    return m, float(np.sqrt(var))


def cell(m: pd.DataFrame, col_b: str, col_a: str) -> dict:
    d = (m[col_b] - m[col_a]).to_numpy(float)
    mu, se1 = one_way(d, m["date"].to_numpy())
    _, se2 = two_way(d, m["date"].to_numpy(), m["player_id"].to_numpy())
    return {"n": int(len(m)), "dates": int(m["date"].nunique()),
            "players": int(m["player_id"].nunique()), "diff": mu, "se1": se1, "z1": mu / se1,
            "se2": se2, "ratio": se2 / se1, "z2": mu / se2}


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """b minus a."""
    m = a.merge(b, on=K, suffixes=("_a", "_b"))
    m = m[m["crps_a"].notna() & m["crps_b"].notna()].copy()
    m["ae_a"] = (m["mean_a"] - m["y_a"]).abs()
    m["ae_b"] = (m["mean_b"] - m["y_a"]).abs()
    out = {"all": {"crps": cell(m, "crps_b", "crps_a"), "mae": cell(m, "ae_b", "ae_a")},
           "tier": {}, "p_play_mean": [float(m["p_play_a"].mean()), float(m["p_play_b"].mean())]}
    U = m[(m["exp_min_a"] >= 10) | (m["exp_min_b"] >= 10)]
    out["fixed_pop"] = {"crps": cell(U, "crps_b", "crps_a"), "mae": cell(U, "ae_b", "ae_a")}
    for t in sorted(m["tier_b"].dropna().unique()):
        x = m[m["tier_b"] == t]
        out["tier"][str(t)] = {"crps": cell(x, "crps_b", "crps_a")}
    return out


def main(argv=None) -> None:
    proto, v1, p2in, p2oos = sys.argv[1:5]
    v1 = pd.read_parquet(f"{v1}/players_pathA-v0.3_{proto}.parquet")
    pin = pd.read_parquet(f"{p2in}/players_pathA-v0.3+p2_{proto}.parquet")
    poos = pd.read_parquet(f"{p2oos}/players_pathA-v0.3+p2_{proto}.parquet")
    res = {"proto": proto,
           "p2_in_sample_vs_v1": compare(v1, pin),
           "p2_oos_vs_v1": compare(v1, poos),
           "p2_oos_vs_p2_in_sample": compare(pin, poos)}
    print(json.dumps(res, indent=1))
    for k in ("p2_in_sample_vs_v1", "p2_oos_vs_v1", "p2_oos_vs_p2_in_sample"):
        c = res[k]["all"]["crps"]
        f = res[k]["fixed_pop"]["crps"]
        print(f"{proto:6s} {k:24s} all: n={c['n']} d={c['diff']:+.4f} se1={c['se1']:.4f} "
              f"z1={c['z1']:+.2f} se2={c['se2']:.4f} z2={c['z2']:+.2f} | fixed-pop n={f['n']} "
              f"d={f['diff']:+.4f} z1={f['z1']:+.2f} z2={f['z2']:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
