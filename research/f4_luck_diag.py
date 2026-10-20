"""F4 diagnostic: does shooting luck inside the EB rates regress? For test season S (regular
season), bin the rows by f4_lk_<s> (the share of the EB per-minute rate that is made-minus-xFG
shooting) and report actual / expected count for pts and fg3m, against the base E (EB x actual
minutes x market) and against E x the P+F2+F3+F6 layer's multiplier (saved by the ablation run),
and the same with the P+F2+F3+F6+F4 layer. A downward slope = hot shooters fall back.

Second part: the frame metrics (NB LL, DK MAE / CRPS given actual minutes) of +F4 vs P+F2+F3+F6
with the multipliers RECONCILED per team-game as the pipeline does (points rescaled so the
team's expected points are unchanged, the other stats by c^0.5): what survives the market total.

    python research/f4_luck_diag.py 2024-25 [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj import context as CX
from courtside.proj import features as FX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("season")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{config.db_path()}?mode=ro", uri=True)
    sp = CX.state_params_for(conn, a.season)
    fr = FX.build_frame(conn, a.season, sp)
    fr = FX.extend_frame(conn, fr, ["F3", "F6", "F4"], season=a.season)
    te = fr[(fr["season"] == a.season) & (fr["season_type"] == "Regular Season")].reset_index(
        drop=True)
    mult = {}
    for v in ("P+F2+F3+F6", "P+F2+F3+F6+F4"):
        p = CX.model_path(a.season, v)
        if p.exists():
            mult[v] = CX.load(p).predict(te)
    res = {}
    lines = [f"== {a.season}: actual / expected by decile of the EB-embedded luck share"]
    for s, col in (("pts", "f4_lk_pts"), ("fg3m", "f4_lk_fg3m")):
        x = te[col].to_numpy(float)
        ok = np.isfinite(x) & (te["mpg"].to_numpy(float) >= 15)
        q = np.quantile(x[ok], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 9)
        y, E = te[s].to_numpy(float), te[f"E_{s}"].to_numpy(float)
        rows = []
        for k in range(10):
            mk = ok & (b == k)
            r = {"decile": k, "lk_mean": float(x[mk].mean()), "n": int(mk.sum()),
                 "base": float(y[mk].sum() / E[mk].sum())}
            for v, m in mult.items():
                r[v] = float(y[mk].sum() / (E[mk] * m[s][mk]).sum())
            rows.append(r)
        res[s] = rows
        lines.append(f"  {s} (mpg >= 15, n={int(ok.sum())}):  decile  luck share  base  "
                     + "  ".join(mult))
        for r in rows:
            lines.append(f"    {r['decile']:2d}  {r['lk_mean']:+.3f}  {r['base']:.3f}  "
                         + "  ".join(f"{r[v]:.3f}" for v in mult))
    # reconciled per team-game (the pipeline's reconciliation to the implied total)
    if len(mult) == 2:
        key = (te["game_id"].astype(str) + "|" + te["team"].astype(str)).to_numpy()
        rec = {}
        for v, m in mult.items():
            Ep = te["E_pts"].to_numpy(float)
            c = (pd.Series(Ep).groupby(key).transform("sum")
                 / pd.Series(Ep * m["pts"]).groupby(key).transform("sum")).to_numpy()
            rec[v] = {s: m[s] * (c if s == "pts" else c ** 0.5) for s in m}
        cv = CX.cv_table(fr[(fr["season"] < a.season)
                            & (fr["season_type"] == "Regular Season")])
        names = list(mult)
        ev = CX.evaluate(te, {**{f"{v} raw": mult[v] for v in names},
                              **{f"{v} rec": rec[v] for v in names}}, cv,
                         pairs=[(f"{names[1]} raw", f"{names[0]} raw"),
                                (f"{names[1]} rec", f"{names[0]} rec")])
        res["reconciled"] = ev["pairs"]
        lines.append("  +F4 vs P+F2+F3+F6, raw vs reconciled per team-game (positive = +F4 better):")
        for p_, d in ev["pairs"].items():
            x = d["all"]
            z = {m_: x[m_]["diff"] / max(x[m_]["se"], x[m_]["se_date"]) for m_ in ("ll", "mae",
                                                                                   "crps")}
            lines.append(f"    {p_:40s} LL {x['ll']['diff']:+.5f} (z {z['ll']:+.1f})  MAE "
                         f"{x['mae']['diff']:+.4f} (z {z['mae']:+.1f})  CRPS "
                         f"{x['crps']['diff']:+.4f} (z {z['crps']:+.1f})  pts LL "
                         f"{x['ll_by_stat']['pts']['diff']:+.5f}  fg3m LL "
                         f"{x['ll_by_stat']['fg3m']['diff']:+.5f}")
    txt = "\n".join(lines)
    print(txt)
    out = Path(a.out) if a.out else config.data_dir() / "reports" / "f4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"luck_diag_{a.season}.json").write_text(json.dumps(res, indent=1))
    (out / f"luck_diag_{a.season}.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
