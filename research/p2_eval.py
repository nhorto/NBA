"""P(play) v2 evaluation on the T2.4 live-like slice: v0.3 (off) vs v0.3+p2, lock-45 and lock-10.

Reads OUT/players_<arm>_<protocol>.parquet (research/t24_slice.py run) and the candidate rows
OUT/rows_slice_live45.parquet / rows_slice_live10.parquet (research/p2_rows.py: roster flag,
report status, fringe features), and prints / writes:

* P(play) calibration by group (listed not Out / fringe / roster unlisted / Out) x role
  (availability.role_of on avg10): mean P(play) off, p2, share played, gaps with SEs;
* DK MAE / CRPS / bias / coverage and minutes MAE, paired p2 - off on a FIXED population
  (rows with E[min] >= 10 in either arm, DNP = 0; the slice's own-arm populations would drop
  the fringe rows p2 pushes below 10), overall, by tier (the off arm's tier) and by group;
  SEs two-way clustered by day and player;
* the slice's standard cells (`backtest.slice_report`, each arm's own population).

    python research/p2_eval.py OUT [--off pathA-v0.3] [--on pathA-v0.3+p2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from courtside.proj import availability as AV
from courtside.proj import backtest as B

LISTED = {"doubtful", "questionable", "probable", "available", "removed", "daytoday"}
KEY = ["date", "player_id"]


def groups(r: pd.DataFrame) -> pd.Series:
    st = r["status"].fillna("")
    g = np.where(st == "out", "out",
                 np.where(st.isin(LISTED), "listed",
                          np.where(r["roster"] == 0, "fringe", "roster_unlisted")))
    return pd.Series(g, index=r.index)


def paired(x: pd.DataFrame, col_a: str, col_b: str) -> dict:
    d = (x[col_a] - x[col_b]).to_numpy(float)
    ok = np.isfinite(d)
    x, d = x[ok], d[ok]
    if not len(d):
        return {"diff": None, "se": None, "n": 0}
    m = float(d.mean())
    var = B._two_way_var((d - m) / len(d), x["date"].to_numpy(), x["player_id"].to_numpy())
    se = float(np.sqrt(var)) if np.isfinite(var) else None
    return {"diff": m, "se": se, "n": int(len(d)), "off": float(x[col_b].mean()),
            "p2": float(x[col_a].mean())}


def mean_se(x: pd.DataFrame, col: str) -> tuple[float, float]:
    v = x[col].to_numpy(float)
    m = float(v.mean())
    var = B._two_way_var((v - m) / len(v), x["date"].to_numpy(), x["player_id"].to_numpy())
    return m, float(np.sqrt(var))


def compare(off: pd.DataFrame, on: pd.DataFrame, rows: pd.DataFrame) -> dict:
    a = B.slice_rows(off)
    b = B.slice_rows(on)
    cols = ["mean", "y", "q10", "q90", "crps", "exp_min", "minutes", "p_play", "min_played",
            "tier", "game_id", "p_avail"]
    m = a[KEY + cols + ["status"]].merge(b[KEY + cols], on=KEY, suffixes=("_off", "_p2"))
    m = m.merge(rows[KEY + ["roster", "avg10", "app_season", "team_gp"]].drop_duplicates(KEY),
                on=KEY, how="left")
    m["group"] = groups(m)
    m["role"] = [AV.role_of(v if np.isfinite(v) and v > 0 else None) for v in m["avg10"]]
    m["played"] = (m["minutes_off"] > 0).astype(float)
    y = m["y_off"].to_numpy(float)
    for s in ("off", "p2"):
        m[f"ae_{s}"] = np.abs(m[f"mean_{s}"] - y)
        m[f"bias_{s}"] = m[f"mean_{s}"] - y
        m[f"cov80_{s}"] = ((y >= m[f"q10_{s}"]) & (y <= m[f"q90_{s}"])).astype(float)
        m[f"minae_{s}"] = np.abs(m[f"exp_min_{s}"] - m["minutes_off"])
        m[f"pp_{s}"] = m[f"p_play_{s}"]
    U = m[(m["exp_min_off"] >= 10) | (m["exp_min_p2"] >= 10)]
    out = {"n_rows": int(len(m)), "n_U": int(len(U)), "overall": {}, "tier": {}, "group": {},
           "pplay": []}
    for k, (ca, cb) in {"dk_mae": ("ae_p2", "ae_off"), "dk_crps": ("crps_p2", "crps_off"),
                        "dk_bias": ("bias_p2", "bias_off"), "cov80": ("cov80_p2", "cov80_off"),
                        "min_mae": ("minae_p2", "minae_off")}.items():
        out["overall"][k] = paired(U, ca, cb)
        out["tier"][k] = {t: paired(U[U["tier_off"] == t], ca, cb)
                          for t in sorted(U["tier_off"].dropna().unique())}
        out["group"][k] = {g: paired(U[U["group"] == g], ca, cb)
                           for g in sorted(U["group"].unique())}
    for (g, r), x in m.groupby(["group", "role"]):
        if len(x) < 5:
            continue
        row = {"group": g, "role": r, "n": int(len(x)), "played": float(x["played"].mean())}
        for s in ("off", "p2"):
            x = x.assign(e=x[f"pp_{s}"] - x["played"])
            gm, gse = mean_se(x, "e")
            row[f"p_{s}"] = float(x[f"pp_{s}"].mean())
            row[f"gap_{s}"], row[f"gap_se_{s}"] = gm, gse
        out["pplay"].append(row)
    for g, x in m.groupby("group"):
        row = {"group": g, "role": "all", "n": int(len(x)), "played": float(x["played"].mean())}
        for s in ("off", "p2"):
            x = x.assign(e=x[f"pp_{s}"] - x["played"])
            gm, gse = mean_se(x, "e")
            row[f"p_{s}"] = float(x[f"pp_{s}"].mean())
            row[f"gap_{s}"], row[f"gap_se_{s}"] = gm, gse
            row[f"brier_{s}"] = float(((x[f"pp_{s}"] - x["played"]) ** 2).mean())
        out["pplay"].append(row)
    return out


def fmt(res: dict, proto: str) -> str:
    L = [f"\n=== {proto}: {res['n_rows']} candidate rows, fixed DK population "
         f"(E[min] >= 10 either arm) {res['n_U']}"]
    L.append("P(play) calibration (P off / P p2 / played; gap = P - played (SE))")
    for r in sorted(res["pplay"], key=lambda r: (r["group"], r["role"])):
        L.append(f"  {r['group']:<16} {r['role']:<9} n={r['n']:<6} off {r['p_off']:.3f} "
                 f"({r['gap_off']:+.3f} {r['gap_se_off']:.3f})  p2 {r['p_p2']:.3f} "
                 f"({r['gap_p2']:+.3f} {r['gap_se_p2']:.3f})  played {r['played']:.3f}"
                 + (f"  Brier {r['brier_off']:.4f} -> {r['brier_p2']:.4f}" if "brier_off" in r
                    else ""))
    L.append("Paired p2 - off (SE): overall")
    for k, v in res["overall"].items():
        L.append(f"  {k:<8} off {v['off']:.4f} p2 {v['p2']:.4f} diff {v['diff']:+.4f} "
                 f"({v['se'] or float('nan'):.4f}) n={v['n']}")
    for part in ("tier", "group"):
        L.append(f"by {part}:")
        for k in ("dk_mae", "dk_crps", "dk_bias", "min_mae"):
            cells = "  ".join(f"{t} {v['diff']:+.3f} ({v['se'] or float('nan'):.3f}) n={v['n']}"
                              for t, v in res[part][k].items() if v["n"])
            L.append(f"  {k:<8} {cells}")
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--off", default="pathA-v0.3")
    ap.add_argument("--on", default="pathA-v0.3+p2")
    a = ap.parse_args(argv)
    od = Path(a.out)
    rep = {}
    frames = {}
    for proto, rows_file in (("live", "rows_slice_live45.parquet"),
                             ("live10", "rows_slice_live10.parquet")):
        fo, fp = od / f"players_{a.off}_{proto}.parquet", od / f"players_{a.on}_{proto}.parquet"
        if not (fo.exists() and fp.exists()):
            continue
        off, on = pd.read_parquet(fo), pd.read_parquet(fp)
        rows = pd.read_parquet(od / rows_file)
        rep[proto] = compare(off, on, rows)
        print(fmt(rep[proto], proto))
        frames[(a.off, proto)] = off
        frames[(a.on, proto)] = on
    sr = B.slice_report(frames, reference="live", primary="live", base_arm=a.off)
    print("\n" + B.format_slice_report(sr))
    (od / "p2_eval.json").write_text(json.dumps(B.E._clean({"paired": rep, "slice": sr}),
                                                indent=1))
    print("wrote", od / "p2_eval.json")


if __name__ == "__main__":
    sys.exit(main())
