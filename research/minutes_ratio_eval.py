"""Minutes-ratio arm (COURTSIDE_MINRATIO=1, `proj/minutes_ratio.py`) vs OFF: paired evaluation.

Reads per-player projection rows of both arms (same seeds: common random numbers, so the minutes
draws are identical and every difference is the per-minute table):

  avail (T2.1 harness, `proj/backtest.py`, 4,000 sims, closing lines, bundles fit < S):
      OUT/rows_avail_<S>_off.parquet / _on.parquet   S in 2023-24, 2024-25 (+ RotoGuru 2019-20,
      2020-21: DK pool rows only)
  live-like slice (T2.4, `research/t24_slice.py`, P(play) v2, 2025-12-22 .. 2026-04-12):
      OUT/players_pathA-v0.3+p2_<live|live10>.parquet (off) / players_pathA-v0.3+mr+p2_<..> (on)

and the return-state rows of the return-minutes study (var/tmp/ret3/out/rows_*.parquet:
pre_min, games since return, absence length, report reason) to cut them.

Population (as `backtest.player_report`): projected rows with E[min] >= 10, DNP scored 0. Every
difference is ON - OFF on identical rows, SE clustered by date. Metrics: DK MAE, CRPS, bias,
80 % coverage, PIT max deviation, minutes MAE (must be identical); by tier of E[min | plays]
(star >= 30, starter 24-30, rotation 15-24, bench < 15); the per-minute DK error by E[min | plays]
/ last-10 minutes (the finding the arm addresses); promoted rows (ratio > 1.2); T2.7's star-out
top-6 gap; return rows (games 0-2, long absences, soft tissue).

    python research/minutes_ratio_eval.py pull         # rsync rows from the Mac scratch (mr/out)
    python research/minutes_ratio_eval.py run [--raw var/tmp/mr/raw] [--out var/tmp/mr/out]
    python research/minutes_ratio_eval.py proxy S [S ..]   # table designs at realised minutes
                                                           # (needs the DB and the bundles of S)

Designs: the switch runs design C, f_u by tier of the plan P (`minutes_ratio.PL_EDGES`; Mac
`mr/out_c`, pulled to var/tmp/mr2/raw_c with the shared OFF rows). Design B (one pooled f_u,
`mr/out_b` -> var/tmp/mr/raw_b) and the g_p variant (`mr/out`) were dropped (docs/PLAYERS.md).
The report adds bench cuts (E[min | plays] < 15) by the realised m / P and by P / last-10.

`proxy` fits each candidate table on the training rows of the 5 seasons < S (walk-forward state,
plans attached) and scores season S's rows at their REALISED minutes (no sims): DK MAE, pts
Poisson deviance and the calibration RMS over (P tier x m / P bin) cells, against the OFF
f(m / mpg) table of S's bundle. This is how the P tiers were chosen.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "var" / "tmp" / "mr"
RET3 = ROOT / "var" / "tmp" / "ret3" / "out"
DB = ROOT / "var" / "courtside.db"
MAC = "nicholashorton@mac-mini"
SCR = "Developer/courtside-compute/mr/out/"
AVAIL = ("2019-20", "2020-21", "2023-24", "2024-25")
LIVE = ("live", "live10")
RET3_AVAIL = {"2019-20": "avail-1920", "2020-21": "avail-2021", "2023-24": "avail-2324",
              "2024-25": "avail-2425"}
RET3_LIVE = {"live": "live+p2", "live10": "live10+p2"}
KEY = ["date", "player_id", "game_id"]
P_MIN = 0.05
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
RATIO_BINS = [0, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4, 10]


# ============================================================================== io

def pull() -> None:
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include", "rows_*.parquet", "--include", "players_*.parquet",
                    "--include", "*.json", "--exclude", "*", f"{MAC}:{SCR}", str(WORK / "raw")],
                   check=True)


def load_pairs(raw: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out = {}
    for s in AVAIL:
        a, b = raw / f"rows_avail_{s}_off.parquet", raw / f"rows_avail_{s}_on.parquet"
        if a.exists() and b.exists():
            out[f"avail {s}"] = (pd.read_parquet(a), pd.read_parquet(b))
    for p in LIVE:
        a = raw / f"players_pathA-v0.3+p2_{p}.parquet"
        b = raw / f"players_pathA-v0.3+mr+p2_{p}.parquet"
        if a.exists() and b.exists():
            out[f"{p}+p2 2025-26"] = (pd.read_parquet(a), pd.read_parquet(b))
    return out


def state_rows(name: str) -> pd.DataFrame | None:
    if name.startswith("avail"):
        f = RET3 / f"rows_{RET3_AVAIL[name.split()[1]]}.parquet"
    else:
        f = RET3 / f"rows_{RET3_LIVE[name.split('+')[0]]}.parquet"
    if not f.exists():
        return None
    cols = ["pre_min", "gsr_b", "abs_b", "tier_pre", "team_change", "first_season", "reason_g",
            "minutes"]
    d = pd.read_parquet(f)
    keep = KEY + [c for c in cols if c in d]
    return d[keep].rename(columns={"minutes": "box_minutes"}).drop_duplicates(KEY)


def paired_frame(off: pd.DataFrame, on: pd.DataFrame, st: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["y", "minutes", "has_proj", "mean", "q10", "q25", "q50", "q75", "q90", "exp_min",
            "p_play", "crps", "pit", "team"]
    a = off[KEY + cols].drop_duplicates(KEY)
    b = on[KEY + ["mean", "q10", "q90", "crps", "pit", "exp_min", "p_play"]].drop_duplicates(KEY)
    d = a.merge(b, on=KEY, suffixes=("_off", "_on"), how="inner")
    if st is not None:
        d = d.merge(st, on=KEY, how="left")
        if "box_minutes" in d:            # RotoGuru minutes -> box minutes where known
            d["minutes"] = np.where(d["box_minutes"].notna(), d["box_minutes"], d["minutes"])
    d = d[d["has_proj"].astype(bool) & (d["exp_min_off"] >= 10)].reset_index(drop=True)
    d["min_c"] = d["exp_min_off"] / d["p_play_off"].clip(lower=P_MIN)
    d["tier4"] = "bench"
    for name, lo, hi in TIERS:
        d.loc[(d["min_c"] >= lo) & (d["min_c"] < hi), "tier4"] = name
    return d


# ============================================================================== stats

def cl_mean(z: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    z = np.asarray(z, float)
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    m = float(z.mean())
    s = pd.Series(z - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    return m, float(np.sqrt(g / max(g - 1, 1) * (s ** 2).sum()) / n)


def pit_dev(p: np.ndarray) -> float:
    h = np.histogram(np.clip(p, 0, 1 - 1e-12), bins=10, range=(0, 1))[0] / max(len(p), 1)
    return float(np.abs(h - 0.1).max())


def metrics(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(float)
    cl = d["date"].to_numpy()
    out = {"n": int(len(d))}
    for k, f in (("mae", lambda a: np.abs(y - d[f"mean_{a}"].to_numpy(float))),
                 ("crps", lambda a: d[f"crps_{a}"].to_numpy(float)),
                 ("bias", lambda a: d[f"mean_{a}"].to_numpy(float) - y),
                 ("cov80", lambda a: ((d[f"q10_{a}"].to_numpy(float) <= y)
                                      & (y <= d[f"q90_{a}"].to_numpy(float))).astype(float))):
        z0, z1 = f("off"), f("on")
        m0, se0 = cl_mean(z0, cl)
        dd, se = cl_mean(z1 - z0, cl)
        out[k] = {"off": m0, "off_se": se0, "on": float(z1.mean()), "diff": dd, "se": se,
                  "z": dd / se if se and se > 0 else float("nan")}
    out["pit_dev"] = {"off": pit_dev(d["pit_off"].to_numpy(float)),
                      "on": pit_dev(d["pit_on"].to_numpy(float))}
    mm = np.abs(d["exp_min_off"].to_numpy(float) - d["minutes"].to_numpy(float))
    out["min_mae"] = {"off": float(mm.mean()),
                      "identical": bool(np.array_equal(d["exp_min_off"].to_numpy(),
                                                       d["exp_min_on"].to_numpy())
                                        and np.array_equal(d["p_play_off"].to_numpy(),
                                                           d["p_play_on"].to_numpy()))}
    return out


def played_bias(d: pd.DataFrame) -> dict:
    """DK bias given played (actual - E[DK | plays]), off / on / paired diff."""
    q = d[(d["minutes"] > 0) & (d["p_play_off"] >= P_MIN)]
    if len(q) < 2:
        return {"n": int(len(q))}
    cl = q["date"].to_numpy()
    r0 = q["y"] - q["mean_off"] / q["p_play_off"].clip(lower=P_MIN)
    r1 = q["y"] - q["mean_on"] / q["p_play_off"].clip(lower=P_MIN)
    m0, s0 = cl_mean(r0.to_numpy(), cl)
    m1, s1 = cl_mean(r1.to_numpy(), cl)
    dd, sd = cl_mean((r1 - r0).to_numpy(), cl)
    return {"n": int(len(q)), "off": m0, "off_se": s0, "on": m1, "on_se": s1, "diff": dd,
            "diff_se": sd}


def per_minute(d: pd.DataFrame, arm: str) -> tuple[float, float]:
    """Per-minute DK error sum(y)/sum(min) / (sum E[DK|plays] / sum E[min|plays]) - 1 (played)."""
    q = d[(d["minutes"] > 0) & (d["p_play_off"] >= P_MIN)]
    if len(q) < 2:
        return float("nan"), float("nan")
    dk_c = q[f"mean_{arm}"] / q["p_play_off"].clip(lower=P_MIN)
    ra = q["y"].sum() / q["minutes"].sum()
    rp = dk_c.sum() / q["min_c"].sum()
    z = (q["y"] - ra * q["minutes"]).to_numpy() / q["minutes"].mean() / rp
    _m, se = cl_mean(z, q["date"].to_numpy())
    return float(ra / rp - 1), se


# ============================================================================== star out

def star_out(seasons) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "research"))
    from dfs_correlations_v2 import star_out_frame
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return star_out_frame(conn, list(seasons))


def star_out_rows(d: pd.DataFrame, so: pd.DataFrame) -> dict:
    x = d.merge(so, on=["game_id", "team"], how="left")
    x = x[x["n_star_out"].notna()].copy()
    x["top6"] = x.groupby(["game_id", "team"])["exp_min_off"].rank(ascending=False,
                                                                   method="first") <= 6
    t = x[x["top6"]]
    out = {"star_out": played_bias(t[t["n_star_out"] >= 1]),
           "no_star_out": played_bias(t[t["n_star_out"] == 0])}
    for arm in ("off", "on"):
        a, b = out["star_out"].get(arm), out["no_star_out"].get(arm)
        out[f"gap_{arm}"] = None if a is None or b is None else a - b
    so_rows = t[t["n_star_out"] >= 1]
    out["mae"] = metrics(so_rows)["mae"] if len(so_rows) > 1 else None
    return out


# ============================================================================== report

def cuts(d: pd.DataFrame) -> dict[str, pd.Series]:
    ok = pd.Series(True, index=d.index)
    if "pre_min" in d:
        ratio = d["min_c"] / d["pre_min"].where(d["pre_min"] >= 5)
        core = (d["tier_pre"].isin(["star", "starter", "rotation"])
                & (d["team_change"] != 1) & (d["first_season"] == 0))
        gsr02 = d["gsr_b"].isin(["0", "1", "2"]) & core
        out = {"ratio < 0.8": ratio < 0.8, "ratio 0.8-1.2": (ratio >= 0.8) & (ratio <= 1.2),
               "promoted (ratio > 1.2)": ratio > 1.2,
               "return gsr 0": (d["gsr_b"] == "0") & core, "return gsr 0-2": gsr02,
               "long absence 16+, gsr 0-2": gsr02 & (d["abs_b"] == "16+")}
        if "reason_g" in d:
            out["soft tissue, gsr 0-2"] = gsr02 & (d["reason_g"] == "soft tissue")
        return out
    return {"all": ok}


BENCH_U_BINS = [-1, 0.001, 0.4, 0.7, 1.0, 1.3, 1.7, 99]
BENCH_V_BINS = [0, 0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 99]


def bench_cuts(d: pd.DataFrame) -> dict:
    """Bench rows (E[min | plays] < 15) by the realised m / E[min | plays] (0 = did not play) and
    by E[min | plays] / last-10 minutes: n, mean shift ON - OFF, DK MAE and CRPS deltas, bias."""
    b = d[d["tier4"] == "bench"]
    out = {}
    cuts_ = {"m/P": pd.cut(b["minutes"] / b["min_c"], BENCH_U_BINS)}
    if "pre_min" in b:
        cuts_["P/last10"] = pd.cut(b["min_c"] / b["pre_min"].where(b["pre_min"] >= 1),
                                   BENCH_V_BINS)
    for name, c in cuts_.items():
        for k, q in b.groupby(c, observed=True):
            y = q["y"].to_numpy(float)
            out[f"{name} {k}"] = {
                "n": int(len(q)),
                "dmean": float((q["mean_on"] - q["mean_off"]).mean()),
                "dmae": float((np.abs(y - q["mean_on"]) - np.abs(y - q["mean_off"])).mean()),
                "dcrps": float((q["crps_on"] - q["crps_off"]).mean()),
                "bias_off": float((q["mean_off"] - y).mean()),
                "bias_on": float((q["mean_on"] - y).mean())}
    return out


def report(pairs: dict, so: pd.DataFrame | None) -> tuple[str, dict]:
    md, js = [], {}
    for name, (off, on) in pairs.items():
        d = paired_frame(off, on, state_rows(name))
        r = {"all": metrics(d), "tiers": {}, "cuts": {}, "per_minute": {}}
        for t, _, _ in TIERS:
            q = d[d["tier4"] == t]
            r["tiers"][t] = {**metrics(q), "played_bias": played_bias(q)} if len(q) > 1 else {}
        for k, m in cuts(d).items():
            q = d[m.fillna(False).to_numpy()]
            mq = metrics(q) if len(q) > 1 else None
            r["cuts"][k] = ({"n": int(len(q)), "mae": mq["mae"], "crps": mq["crps"],
                             "played_bias": played_bias(q)} if mq else {"n": int(len(q))})
        r["bench_cuts"] = bench_cuts(d)
        if "pre_min" in d:
            ratio = d["min_c"] / d["pre_min"].where(d["pre_min"] >= 5)
            rb = pd.cut(ratio, RATIO_BINS)
            for b in rb.cat.categories:
                q = d[(rb == b).to_numpy()]
                r["per_minute"][str(b)] = {"n": int(len(q)), "off": per_minute(q, "off"),
                                           "on": per_minute(q, "on")}
        if so is not None:
            r["star_out"] = star_out_rows(d, so)
        js[name] = r
        md.append(fmt(name, r))
    return "\n".join(md), js


def _f(x, k=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{k}f}"


def fmt(name: str, r: dict) -> str:
    a = r["all"]
    L = [f"\n## {name} (n {a['n']})\n",
         "| | OFF | ON − OFF (SE) | z |", "|---|---|---|---|"]
    for k in ("mae", "crps", "bias", "cov80"):
        L.append(f"| {k} | {a[k]['off']:.4f} | {a[k]['diff']:+.4f} ({a[k]['se']:.4f}) | "
                 f"{a[k]['z']:+.1f} |")
    L.append(f"| PIT max dev | {a['pit_dev']['off']:.4f} | → {a['pit_dev']['on']:.4f} | |")
    L.append(f"| minutes MAE | {a['min_mae']['off']:.3f} | identical: {a['min_mae']['identical']} | |")
    L += ["\n| tier | n | DK MAE Δ (SE) | CRPS Δ (SE) | z CRPS | bias|played OFF → ON |",
          "|---|---|---|---|---|---|"]
    for t, v in r["tiers"].items():
        if not v:
            continue
        pb = v["played_bias"]
        L.append(f"| {t} | {v['n']} | {v['mae']['diff']:+.4f} ({v['mae']['se']:.4f}) | "
                 f"{v['crps']['diff']:+.4f} ({v['crps']['se']:.4f}) | {v['crps']['z']:+.1f} | "
                 f"{_f(pb.get('off'), 2)} → {_f(pb.get('on'), 2)} |")
    if r["cuts"]:
        L += ["\n| rows | n | DK MAE Δ (SE) | CRPS Δ (SE) | DK bias|played OFF → ON (SE) |",
              "|---|---|---|---|---|"]
        for k, v in r["cuts"].items():
            if "mae" not in v:
                L.append(f"| {k} | {v.get('n', 0)} | | | |")
                continue
            pb = v["played_bias"]
            L.append(f"| {k} | {v['n']} | "
                     f"{v['mae']['diff']:+.4f} ({v['mae']['se']:.4f}) | "
                     f"{v['crps']['diff']:+.4f} ({v['crps']['se']:.4f}) | "
                     f"{_f(pb.get('off'), 2)} → {_f(pb.get('on'), 2)} ({_f(pb.get('on_se'), 2)}) |")
    if r["per_minute"]:
        L += ["\n| E[min∣plays] / last-10 min | n | per-minute DK error OFF | ON |", "|---|---|---|---|"]
        for b, v in r["per_minute"].items():
            o, n_ = v["off"], v["on"]
            L.append(f"| {b} | {v['n']} | {o[0]:+.1%} ({o[1]:.1%}) | {n_[0]:+.1%} ({n_[1]:.1%}) |")
    if r.get("bench_cuts"):
        L += ["\n| bench rows | n | mean Δ | DK MAE Δ | CRPS Δ | bias OFF → ON |",
              "|---|---|---|---|---|---|"]
        for k, v in r["bench_cuts"].items():
            L.append(f"| {k} | {v['n']} | {v['dmean']:+.3f} | {v['dmae']:+.4f} | "
                     f"{v['dcrps']:+.4f} | {v['bias_off']:+.2f} → {v['bias_on']:+.2f} |")
    so = r.get("star_out")
    if so:
        a_, b_ = so["star_out"], so["no_star_out"]
        L.append(f"\nStar out, remaining top 6 (played, n {a_.get('n')}): DK bias actual − proj "
                 f"OFF {_f(a_.get('off'), 2)} → ON {_f(a_.get('on'), 2)} (paired "
                 f"{_f(a_.get('diff'), 3)} ± {_f(a_.get('diff_se'), 3)}); gap vs no-star-out nights "
                 f"OFF {_f(so['gap_off'], 2)} → ON {_f(so['gap_on'], 2)}")
    return "\n".join(L)


def run(out: Path, raw: Path = WORK / "raw") -> None:
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(raw)
    seasons = sorted({k.split()[1] for k in pairs if k.startswith("avail")} | {"2025-26"})
    so = star_out(seasons) if DB.exists() else None
    md, js = report(pairs, so)
    (out / "tables.md").write_text(md + "\n")
    (out / "results.json").write_text(json.dumps(js, indent=1, default=float))
    print(md)


PROXY_DESIGNS = {"pooled (B)": (), "P tiers 15": (15.0,), "P tiers 12/18": (12.0, 18.0),
                 "P tiers 10..30": (10.0, 15.0, 20.0, 25.0, 30.0),
                 "P tiers 12..30 (C)": (12.0, 18.0, 24.0, 30.0)}


def proxy(seasons) -> None:
    """Table designs at realised minutes (module doc): fit on the 5 seasons < S, score S."""
    from courtside import config, db
    from courtside.proj import minutes_ratio as MR
    from courtside.proj import state as ST
    conn = db.connect()
    W = ST.DK_WEIGHTS
    for S in seasons:
        off = json.loads((config.data_dir() / "proj" / f"bundle_{S}_v1_cx.json").read_text())
        off = off["state"]
        fit = [MR._minus(S, k) for k in range(MR.FIT_SEASONS, 0, -1)]
        pl = MR.planned_rows(conn, fit + [S], compute=False)
        d = MR.training_frame(conn, S, off, pl, seasons=fit + [S])
        d = d[np.isfinite(d["planned"]) & (d["min"] > 0)].reset_index(drop=True)
        tr, te = d[d["season"] < S], d[d["season"] == S].reset_index(drop=True)
        m, mpg, P = (te[c].to_numpy(float) for c in ("min", "mpg", "planned"))
        y = sum(W[s] * te[s].to_numpy(float) for s in W)
        ypts = te["pts"].to_numpy(float)
        tier = pd.cut(te["planned"], [0, 10, 15, 24, 30, 99]).astype(str)
        ub = pd.cut(m / np.maximum(P, 1), [0, 0.4, 0.7, 0.85, 1.0, 1.15, 1.4, 99]).astype(str)
        sps = {"OFF f(m/mpg)": None}
        sps.update({k: dict(off, minutes_adj_mr=MR.fit_table(tr, pl_edges=e))
                    for k, e in PROXY_DESIGNS.items()})
        rows = {}
        for name, sp in sps.items():
            def mult(s):
                return (ST.minutes_multiplier(off, s, m, mpg) if sp is None
                        else ST.minutes_multiplier(sp, s, m, mpg, planned=P))
            p = sum(W[s] * te[f"r_{s}"].to_numpy(float) * m * mult(s) for s in W)
            pp = te["r_pts"].to_numpy(float) * m * mult("pts")
            dev = 2 * (np.where(ypts > 0, ypts * np.log(np.maximum(ypts, 1e-9) / pp), 0)
                       - (ypts - pp))
            g = pd.DataFrame({"y": y, "p": p, "t": tier, "u": ub, "ae": np.abs(y - p)})
            cal = g.groupby(["t", "u"]).agg(y=("y", "sum"), p=("p", "sum"), n=("y", "size"))
            r = {"DK MAE": g["ae"].mean(), "pts deviance": dev.mean(),
                 "cal RMS": float(np.sqrt(np.average((cal["y"] / cal["p"] - 1) ** 2,
                                                     weights=cal["n"])))}
            for t_, q in g.groupby("t"):
                r[f"MAE P {t_}"] = q["ae"].mean()
                r[f"err P {t_}"] = q["y"].sum() / q["p"].sum() - 1
            rows[name] = r
        print(f"\n## {S} (test rows {len(te)})\n" + pd.DataFrame(rows).round(4).to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull")
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(WORK / "out"))
    r.add_argument("--raw", default=str(WORK / "raw"),
                   help="rows dir (OFF rows + one design's ON rows; the g_p variant's ON rows "
                        "were kept in a separate dir)")
    px = sub.add_parser("proxy")
    px.add_argument("seasons", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "pull":
        pull()
    elif a.cmd == "proxy":
        proxy(a.seasons)
    else:
        run(Path(a.out), Path(a.raw))


if __name__ == "__main__":
    main()
