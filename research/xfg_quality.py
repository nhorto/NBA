"""F4 xFG model quality, walk-forward: for each test season S the xFG model (LightGBM on pre-shot
features) and the distance-only baseline (make rate per 2/3 x 1-ft bin) are fit on the shots of
seasons 2019-20 .. S-1 and scored on every FGA of S: log-loss, Brier, ECE, calibration slope /
intercept, by 2s / 3s. Also: the year-over-year stability of player shooting luck vs skill.

    python research/xfg_quality.py [--seasons 2022-23 2023-24 2024-25 2025-26] [--out FILE]
    python research/xfg_quality.py --persistence 2024-25

--persistence S: how much of a player's prior shooting-over-expected (made - xFG per attempt,
recency-weighted at several half-lives, >= 50 attempts) carries into each game of S (weighted
slope; 1 = all skill, 0 = all luck), 2s and 3s, xFG of seasons < S.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj.features import shots as F4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--persistence", default=None, metavar="S")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db or config.db_path()}?mode=ro", uri=True)
    if a.persistence:
        return persistence(conn, a.persistence)
    sh = F4.load_shots(conn, F4.SHOT_START, a.seasons[-1], log=print)
    fga = F4._fga(sh).reset_index(drop=True)
    res = {}
    for s in a.seasons:
        t0 = time.time()
        tr, te = fga[fga["season"] < s], fga[fga["season"] == s]
        m = F4.fit_xfg(tr, log=print)
        p = m.predict(te)
        b = F4.predict_baseline(F4.fit_baseline(tr), te)
        y = te["made"].to_numpy(float)
        r = {"train": sorted(tr["season"].unique()), "rounds": m.n_rounds}
        for lab, mk in (("all", np.ones(len(te), bool)), ("2pt", (te["kind"] == 2).to_numpy()),
                        ("3pt", (te["kind"] == 3).to_numpy())):
            r[lab] = {"xfg": F4.xfg_quality(y[mk], p[mk]), "dist": F4.xfg_quality(y[mk], b[mk])}
        # importance
        import lightgbm as lgb
        bst = lgb.Booster(model_str=m.booster)
        g = bst.feature_importance("gain")
        r["importance"] = dict(sorted(zip(bst.feature_name(), (g / g.sum()).round(4).tolist()),
                                      key=lambda kv: -kv[1])[:12])
        res[s] = r
        print(f"== {s} (fit {r['train'][0]}..{r['train'][-1]}, {m.n_rounds} rounds, "
              f"{time.time() - t0:.0f}s)")
        for lab in ("all", "2pt", "3pt"):
            x, d = r[lab]["xfg"], r[lab]["dist"]
            print(f"  {lab:4s} n={x['n']:7d}  logloss xFG {x['logloss']:.5f} vs dist {d['logloss']:.5f} "
                  f"(gain {d['logloss'] - x['logloss']:+.5f})  Brier {x['brier']:.5f} / {d['brier']:.5f}  "
                  f"ECE {x['ece']:.4f} / {d['ece']:.4f}  slope {x['cal_slope']:.3f} / {d['cal_slope']:.3f}  "
                  f"mean p {x['mean_p']:.4f} / {d['mean_p']:.4f} vs y {x['mean_y']:.4f}")
        print("  importance:", r["importance"])
    out = a.out or str(config.data_dir() / "reports" / "f4" / "xfg_quality.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)
    return 0


def persistence(conn, season: str) -> int:
    from courtside.proj.features.common import ew_asof
    m = F4.xfg_for_season(conn, season)
    h = F4.load_history(conn, m, last=season).h
    q = h[h["season"] == season].reset_index(drop=True)
    for lab, (n, mm, x) in {"3pt": ("n3", "m3", "x3"), "2pt": ("n2", "m2", "x2")}.items():
        y, w = (q[mm] - q[x]).to_numpy(float), q[n].to_numpy(float)
        for hl in (60, 120, 365, 730):
            S = ew_asof(h["player_id"].to_numpy(), h["day"].to_numpy(), h["sidx"].to_numpy(),
                        h[[n, mm, x]].to_numpy(float), q["player_id"].to_numpy(),
                        q["day"].to_numpy(), q["sidx"].to_numpy(), half_life=hl, rho=0.9)
            k = (w > 0) & (S[:, 0] >= 50)
            xx = (S[k, 1] - S[k, 2]) / S[k, 0]
            yy, ww = y[k] / w[k], w[k]
            xm, ym = np.average(xx, weights=ww), np.average(yy, weights=ww)
            b = np.sum(ww * (xx - xm) * (yy - ym)) / np.sum(ww * (xx - xm) ** 2)
            print(f"{season} {lab} half-life {hl:3d} d: n={int(k.sum())} carry-over slope {b:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
