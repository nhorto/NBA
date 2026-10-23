"""Role-aware blend (`COURTSIDE_ROLE_BLEND`, `+rb`; IMPROVEMENTS.md P1 + D1): fit and evaluation.

Docs: docs/research/role-blend.md.  Code under test: `proj/role_blend.py`, its hooks in
`pipeline.build_games` / `run_games` / `save_run`, `portfolio.truth_sims(blended=...)`.

    python research/role_blend_eval.py harness S PROTO [--lineups]
        backtest rows (4,000 sims) of the arm in the environment -> OUT/rows_<P>_<S>_<MV>.parquet
        (+ OUT/lineups_<P>_<S>_<MV>.parquet with --lineups: RotoGuru cash / Top-N / hindsight)
    python research/role_blend_eval.py fit ROWS_DIR [--arm ARM]
        per-season beta(r) by CRPS proxy, walk-forward on the OFF rows of seasons < S
        -> src/courtside/proj/role_blend_params.json
    python research/role_blend_eval.py dk OFF ON LABEL
        paired ON - OFF DK CRPS / MAE (two-way date x player SEs), tiers, ratio buckets
    python research/role_blend_eval.py lineups OFF ON LABEL
        paired ON - OFF RotoGuru lineup results (actual points, ratio to hindsight)

Environment: COURTSIDE_DATA (bundles + DB), RB_OUT (default var/tmp/rb/out). Heavy: Mac scratch
(`courtside-compute/rb`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RB_OUT", ROOT / "var" / "tmp" / "rb" / "out"))
ARM = "pathA-v0.3+mf+p2"
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
# training datasets (OFF rows) -> their season; targets get every dataset of seasons < S
DATASETS = (("avail", "2018-19"), ("avail", "2019-20"), ("avail", "2020-21"),
            ("avail", "2023-24"), ("avail", "2024-25"), ("live", "2025-26"))
TARGETS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
           "2026-27")


def say(m: str) -> None:
    print(f"[rb] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import db
    return db.connect()


# ============================================================================== harness

def cmd_harness(season: str, protocol: str, lineups: bool) -> None:
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000,
                    modes=("cash", "topn") if lineups else (), baselines=B.BASELINES,
                    pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, ld, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                                date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    if lineups and ld is not None and len(ld):
        ld.to_parquet(OUT / f"lineups_{protocol}_{season}_{P.MODEL_VERSION}.parquet")
    say(f"harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, "
        f"{0 if ld is None else len(ld)} lineup rows, skipped {len(meta['skipped'])}, "
        f"{time.time() - t0:.0f}s -> {f}")


# ============================================================================== frames

def role_frame(rows: pd.DataFrame, conn, season: str) -> pd.DataFrame:
    """Harness rows -> the role_blend fit frame (mu_p, planned, l10, l10min, missed, y, q..)."""
    from courtside.proj import role_blend as RB
    d = rows[rows["has_proj"].astype(bool) & (rows["p_play"] > 0.02) & rows["y"].notna()
             & rows["player_id"].notna() & rows["team"].notna()].copy()
    d["mu_p"] = d["mean"] / d["p_play"]
    d["planned"] = d["exp_min"] / d["p_play"]
    h = RB.RoleHistory(conn, season)
    x = [h.one(int(p), t, str(dd)[:10]) for p, t, dd in zip(d["player_id"], d["team"], d["date"])]
    a = np.array(x, float).reshape(-1, 3)
    d["l10"], d["l10min"], d["missed"] = a[:, 0], a[:, 1], a[:, 2]
    return d.reset_index(drop=True)


def cmd_fit(rows_dir: str, arm: str) -> None:
    from courtside.proj import role_blend as RB
    conn = _conn()
    curves = {}
    for proto, s in DATASETS:
        f = Path(rows_dir) / f"rows_{proto}_{s}_{arm}.parquet"
        if not f.exists():
            say(f"missing {f}; skipped")
            continue
        fr = role_frame(pd.read_parquet(f), conn, s)
        curves[(proto, s)] = RB.loss_curves(fr)
        say(f"{proto} {s}: {len(fr)} rows; n by bucket "
            f"{ {k: v['n'] for k, v in curves[(proto, s)].items()} }")
    out = {"anchor": "l10", "edges": list(RB.EDGES), "buckets": list(RB.BUCKETS),
           "grid": list(RB.GRID), "min_rows": RB.MIN_ROWS, "arm": arm,
           "method": "CRPS proxy (q10..q90) argmin per bucket, walk-forward on harness rows of "
                     "seasons < S; returning (>= 3 missed) and no-history = 0",
           "seasons": {}}
    for S in TARGETS:
        tr = [k for k in curves if k[1] < S]
        if not tr:
            say(f"{S}: no training rows; not written (falls back to DEFAULT_BETA)")
            continue
        c = {}
        for k in tr:
            c = RB.add_curves(c, curves[k]) if c else curves[k]
        b = RB.fit_betas(c)
        out["seasons"][S] = {"beta": b, "train": [f"{p}:{s}" for p, s in tr],
                             "n": {k: v["n"] for k, v in c.items()}, "curves": c}
        say(f"{S}: beta {b} on {[f'{p}:{s}' for p, s in tr]}")
    RB.RB_PATH.write_text(json.dumps(out, indent=1) + "\n")
    say(f"wrote {RB.RB_PATH}")


# ============================================================================== paired DK

def _pair(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    k = ["date", "player_id"]
    a = off.dropna(subset=["player_id"]).drop_duplicates(k)
    b = on.dropna(subset=["player_id"]).drop_duplicates(k)
    cols = ["mean", "q10", "q90", "crps", "exp_min", "p_play"]
    d = a.merge(b[k + cols], on=k, suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool) & d["crps_off"].notna() & d["crps_on"].notna()]
    return d.reset_index(drop=True)


def _metrics(d: pd.DataFrame) -> dict:
    from courtside.proj.metrics import paired_cluster2
    y = d["y"].to_numpy(float)
    dates, pids = d["date"].astype(str).to_numpy(), d["player_id"].to_numpy()
    out = {"n": int(len(d))}
    if len(d) < 20:
        return out
    for k, f in (("crps", lambda a: d[f"crps_{a}"].to_numpy(float)),
                 ("mae", lambda a: np.abs(d[f"mean_{a}"].to_numpy(float) - y)),
                 ("bias", lambda a: d[f"mean_{a}"].to_numpy(float) - y)):
        z0, z1 = f("off"), f("on")
        pc = paired_cluster2(z1 - z0, dates, pids)
        out[k] = {"off": float(z0.mean()), "diff": pc["mean"], "se": pc["se"],
                  "se_date": pc["se_c1"], "z": pc["mean"] / pc["se"] if pc["se"] > 0 else np.nan}
    return out


def cmd_dk(off: str, on: str, label: str) -> None:
    from courtside.proj import role_blend as RB
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    d = _pair(a, b)
    conn = _conn()
    from courtside.proj import pipeline as P
    from datetime import date
    season = P.season_of(date.fromisoformat(str(d["date"].iloc[0])[:10]))
    h = RB.RoleHistory(conn, season)
    x = np.array([h.one(int(p), t, str(dd)[:10]) for p, t, dd in
                  zip(d["player_id"], d["team"], d["date"])], float).reshape(-1, 3)
    d["l10"], d["l10min"], d["missed"] = x[:, 0], x[:, 1], x[:, 2]
    d["planned"] = d["exp_min_off"] / d["p_play_off"].clip(lower=0.02)
    d["bucket"] = RB.bucket_of(d["planned"].to_numpy(float), d["l10min"].to_numpy(float))
    d["returning"] = d["missed"] >= RB.RETURNING_MISSED
    d["tier4"] = "bench"
    mc = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    for name, lo, hi in TIERS:
        d.loc[(mc >= lo) & (mc < hi), "tier4"] = name
    d["pop"] = (d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)
    d["shift"] = d["mean_on"] - d["mean_off"]
    res = {"label": label, "season": season, "beta": RB.betas(season)}
    for popn, q in (("pop10", d[d["pop"]]), ("all", d)):
        r = {"all": _metrics(q)}
        for t, _, _ in TIERS:
            r[f"tier:{t}"] = _metrics(q[q["tier4"] == t])
        for t in ("star", "role", "bench"):
            r[f"ctier:{t}"] = _metrics(q[q["tier"] == t])
        for k in RB.BUCKETS + ("nohist",):
            r[f"r:{k}"] = _metrics(q[(q["bucket"] == k) & ~q["returning"]])
        r["promoted(>1.05)"] = _metrics(q[q["bucket"].isin(["1.05-1.25", ">1.25"])])
        r["returning"] = _metrics(q[q["returning"]])
        r["_shift"] = {"mean": float(q["shift"].mean()), "mean_abs": float(q["shift"].abs().mean()),
                       "share_shifted": float((q["shift"].abs() > 1e-3).mean())}
        res[popn] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label} (ON - OFF; z two-way date x player)  beta {res['beta']}")
    for popn in ("pop10", "all"):
        print(f"### {popn}  shift {res[popn]['_shift']}")
        for k, v in res[popn].items():
            if k.startswith("_") or "crps" not in v:
                continue
            c, m = v["crps"], v["mae"]
            print(f"  {k:18s} n={v['n']:6d}  CRPS {c['off']:.4f} {c['diff']:+.4f} ({c['se']:.4f}) "
                  f"z {c['z']:+.2f}   MAE {m['off']:.4f} {m['diff']:+.4f} ({m['se']:.4f}) "
                  f"z {m['z']:+.2f}   bias {v['bias']['off']:+.3f}->{v['bias']['off'] + v['bias']['diff']:+.3f}")


# ============================================================================== lineups

def cmd_lineups(off: str, on: str, label: str) -> None:
    from courtside.proj import metrics as MT
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    out = {"label": label}

    def singles(x):
        s = x[(x["rank"] == 1) & x["kind"].isin(["single", "hindsight"])]
        return s.pivot_table(index="date", columns="method", values="actual", aggfunc="first")

    wa, wb = singles(a), singles(b)
    idx = wa.index.intersection(wb.index)
    wa, wb = wa.loc[idx], wb.loc[idx]
    hs = wa["hindsight"]
    for m in ("ours_cash", "ours_mean"):
        ok = wa[m].notna() & wb[m].notna() & hs.notna()
        pc = MT.paired_cluster((wb[m] - wa[m])[ok].to_numpy(float), np.asarray(idx[ok]))
        out[m] = {"n_dates": int(ok.sum()), "off": float(wa[m][ok].mean()),
                  "on": float(wb[m][ok].mean()), "diff": pc["mean"], "se": pc["se"],
                  "ratio_hs_off": float(wa[m][ok].sum() / hs[ok].sum()),
                  "ratio_hs_on": float(wb[m][ok].sum() / hs[ok].sum()),
                  "ge300_off": float((wa[m][ok] >= 300).mean()),
                  "ge300_on": float((wb[m][ok] >= 300).mean())}
    ta, tb = a[a["kind"] == "topn"], b[b["kind"] == "topn"]
    for m in ("ours_topn",):
        ga = ta[ta["method"] == m].groupby("date")["actual"].agg(["mean", "max"])
        gb = tb[tb["method"] == m].groupby("date")["actual"].agg(["mean", "max"])
        j = ga.index.intersection(gb.index)
        for st in ("mean", "max"):
            pc = MT.paired_cluster((gb.loc[j, st] - ga.loc[j, st]).to_numpy(float), np.asarray(j))
            out[f"{m}_{st}"] = {"n_dates": int(len(j)), "off": float(ga.loc[j, st].mean()),
                                "on": float(gb.loc[j, st].mean()), "diff": pc["mean"],
                                "se": pc["se"],
                                "ratio_hs_off": float(ga.loc[j, st].sum() / hs.reindex(j).sum()),
                                "ratio_hs_on": float(gb.loc[j, st].sum() / hs.reindex(j).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lineups_{label}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n## lineups {label} (ON - OFF, SE by date)")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:16s} dates {v['n_dates']:4d}  {v['off']:.2f} -> {v['on']:.2f}  "
                  f"diff {v['diff']:+.2f} ({v['se']:.2f})  /hindsight {v['ratio_hs_off']:.4f} -> "
                  f"{v['ratio_hs_on']:.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    h.add_argument("--lineups", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("rows_dir")
    f.add_argument("--arm", default=ARM)
    for c in ("dk", "lineups"):
        d = sub.add_parser(c)
        d.add_argument("off")
        d.add_argument("on")
        d.add_argument("label")
    x = ap.parse_args(argv)
    if x.cmd == "harness":
        cmd_harness(x.season, x.protocol, x.lineups)
    elif x.cmd == "fit":
        cmd_fit(x.rows_dir, x.arm)
    elif x.cmd == "dk":
        cmd_dk(x.off, x.on, x.label)
    else:
        cmd_lineups(x.off, x.on, x.label)


if __name__ == "__main__":
    main()
