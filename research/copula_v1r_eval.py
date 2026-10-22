"""Copula v1 retune (`COURTSIDE_COPULA=v1r`, `+c1r`): diagnosis of v1's over-dispersion and the
evaluation of the arms OFF, +c1, +c1r, +mf, +mf+c1r (docs/PLAYERS.md §0.4 / results block).

Copula v1 (`+c1`) draws per-stat counts (NB2 by level) per player and sim. Its ranges were too
wide (2024-25 DK 80 % coverage 0.829 vs 0.824 for v0). This script measures where the width comes
from, per stat (pts / reb / ast / fg3m, what props would use, plus DK) and tier:

  harness S PROTOCOL
      T2.1 harness (avail) / T2.4 slice (live, live10; P(play) cells leave-one-month-out, real
      tip times from the scoreboard archive in the data dir) for the arm of the environment
      switches, 4,000 sims. Besides the harness's DK rows it writes, per projected player-game
      that played (actual minutes > 0, >= 200 played sims) and per stat (pts, reb, ast, fg3m when
      the run has component sims; DK always), from the played sims:
        mean, sd, randomised PIT, sample CRPS, the variance split by minutes (within 1-minute
        bins of the sim's minutes = count / per-minute noise; between = minutes noise), and the
        same stat CONDITIONAL on the sim's minutes being near the actual ones (+-1, widened to
        +-3 until >= 60 sims): mean / sd / PIT at the actual minutes (= count noise only)
      -> OUT/rows_<protocol>_<S>_<version>[-<diag>].parquet, OUT/stats_... (same name)
      env C1R_DIAG=nowl: a diagnostic copula v1 run without the winner / loser shift (set
      COURTSIDE_BT_CODE_TAG too, so the harness's date cache is not the plain run's)
  fdiag S
      count noise given ACTUAL minutes and team points, on the frame rows of S (test) and of the
      training seasons: per stat x level, the c^2 the arm's copula v1 params use vs the c^2 of
      the residuals around the EB rates (what v1 is fit on) and around the context arm's rates
      (`copula.v1r_frame`), and the NB coverage given minutes -> OUT/fdiag_<S>.json
  report BASE ON --label L
      paired ON - BASE on identical harness rows (E[min] >= 10 in either arm): DK CRPS / MAE /
      80 % coverage / PIT by tier; per-stat calibration of ON (and BASE when it has component
      sims): 50 / 80 % coverage by randomised PIT, sd ratio (predictive sd / RMS error), PIT
      max deviation, CRPS, by stat x tier; SE clustered by date -> OUT/rep_<L>.json
  summary L [L ...]
      markdown tables of several rep_<L>.json (the PLAYERS.md results block)

Tiers by the BASE arm's E[min | plays] (research/minutes_fix_eval.py): star >= 30, starter
24-30, rotation 15-24, bench < 15. Per-stat coverage uses the randomised PIT (a count's q10..q90
over-covers by construction): covered80 = 0.1 <= PIT <= 0.9.

Environment: COURTSIDE_DATA (bundles, DB), C1R_OUT (default var/tmp/c1r/out). Heavy: Mac scratch
(`courtside-compute/c1r`).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("C1R_OUT", ROOT / "var" / "tmp" / "c1r" / "out"))
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
TIER_NAMES = tuple(t[0] for t in TIERS)
PSTATS = ("pts", "reb", "ast", "fg3m")
MIN_PLAYED_SIMS = 200
AT_WIN = (1.0, 2.0, 3.0)
AT_MIN_SIMS = 60
_LAST: dict = {}


def tier4(min_c) -> np.ndarray:
    x = np.asarray(min_c, float)
    out = np.full(len(x), "bench", dtype=object)
    for name, lo, hi in TIERS:
        out[(x >= lo) & (x < hi)] = name
    return out


def cl_mean(z, cl) -> tuple[float, float]:
    from courtside.proj.metrics import paired_cluster
    r = paired_cluster(np.asarray(z, float), np.asarray(cl))
    return r["mean"], r["se"]


# ============================================================================== harness

def _actuals(conn) -> dict:
    a = pd.read_sql_query("SELECT game_id, player_id, min, pts, reb, ast, fg3m, stl, blk, tov "
                          "FROM player_games WHERE season >= '2022-23'", conn)
    cats = sum((a[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    w = {"pts": 1.0, "fg3m": 0.5, "reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5}
    a["dk"] = sum(w[s] * a[s].astype(float) for s in w) + 1.5 * (cats >= 2) + 3.0 * (cats >= 3)
    return {(r.game_id, int(r.player_id)): r for r in a.itertuples(index=False)}


def _one(x: np.ndarray, mins: np.ndarray, y: float, y_min: float, v: float, v2: float) -> tuple:
    """Per-player per-stat summary of played sims x (counts or DK) with their minutes."""
    from courtside.proj.metrics import crps_samples
    n = len(x)
    mean = float(x.mean())
    var = float(x.var())
    k = np.clip(np.rint(mins), 0, 70).astype(np.int64)
    nb = np.bincount(k)
    s1 = np.bincount(k, x)
    s2 = np.bincount(k, x * x)
    ok = nb > 0
    within = float((s2[ok] - s1[ok] ** 2 / nb[ok]).sum() / n)
    lo, eq = float((x < y).mean()), float((x == y).mean())
    pit = lo + v * eq
    crps = float(crps_samples(x, y))
    at = (np.nan,) * 4
    for w in AT_WIN:
        sel = np.abs(mins - y_min) <= w
        if sel.sum() >= AT_MIN_SIMS:
            xa = x[sel]
            at = (float(xa.mean()), float(xa.std()),
                  float((xa < y).mean() + v2 * (xa == y).mean()), float(sel.sum()))
            break
    return (mean, float(np.sqrt(var)), within, var - within, pit, crps, *at)


S_COLS = ("mean", "sd", "var_within", "var_between", "pit", "crps", "at_mean", "at_sd", "at_pit",
          "at_n")


def stat_rows(res, day: str, acts: dict) -> list:
    """Per-stat rows of one harness date (module doc: `harness`)."""
    from courtside.proj import pipeline as P
    rng = np.random.default_rng(P.stable_seed(day, "c1r-stat-pit", 1))
    out = []
    for g in res.sims:
        for j, pid in enumerate(g.player_id):
            a = acts.get((g.game_id, int(pid)))
            if a is None or not (a.min or 0) > 0:
                continue
            pl = g.minutes[:, j] > 0
            if pl.sum() < MIN_PLAYED_SIMS:
                continue
            mins = g.minutes[pl, j].astype(float)
            base = {"date": day, "game_id": g.game_id, "player_id": int(pid),
                    "tier3": g.info.at[j, "tier"], "min_played": float(mins.mean()),
                    "p_play": float(pl.mean()), "y_min": float(a.min)}
            items = [("dk", g.fp[pl, j].astype(float), float(a.dk))]
            if g.stats is not None:
                items += [(s, g.stats[s][pl, j].astype(float), float(getattr(a, s)))
                          for s in PSTATS]
            for s, x, y in items:
                vals = _one(x, mins, y, float(a.min), float(rng.random()), float(rng.random()))
                out.append({**base, "stat": s, "y": y, **dict(zip(S_COLS, vals))})
    return out


def _patch_capture() -> None:
    from courtside.proj import backtest as B
    from courtside.proj import pipeline as P
    if getattr(B.SeasonRun.run_date, "_c1r", False):
        return
    ph, ll, orig = P.project_historical, B.live_like_day, B.SeasonRun.run_date

    def ph2(*a, **k):
        r = ph(*a, **k)
        _LAST["res"] = r
        return r

    def ll2(*a, **k):
        r = ll(*a, **k)
        _LAST["res"] = r
        return r

    def run_date(self, day):
        players, lineups, meta = orig(self, day)
        res = _LAST.pop("res", None)
        if res is not None:
            _LAST.setdefault("stats", []).extend(stat_rows(res, day, _LAST["acts"]))
        _LAST["tip_unknown"] = _LAST.get("tip_unknown", 0) + int(meta.get("tip_unknown") or 0)
        return players, lineups, meta

    run_date._c1r = True
    P.project_historical, B.live_like_day, B.SeasonRun.run_date = ph2, ll2, run_date


def _diag_patch() -> str:
    """Diagnostic variants of a copula v1 run (env C1R_DIAG, comma list), tagged in the file
    names: `nowl` = no winner / loser shift in the sims (factors 1)."""
    from courtside.proj import copula as C
    tags = [t for t in os.environ.get("C1R_DIAG", "").split(",") if t]
    if "nowl" in tags:
        C.wl_factor = lambda v1, stat, tiers, margin: np.ones((len(margin), len(tiers)))
    return "".join(f"-{t}" for t in tags)


def cmd_harness(season: str, protocol: str) -> None:
    from courtside import db
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    tag = _diag_patch()
    _patch_capture()
    conn = db.connect()
    _LAST["acts"] = _actuals(conn)
    _LAST["stats"] = []
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_{protocol}_{season}_{P.MODEL_VERSION}{tag}.parquet"
    pl.to_parquet(f)
    pd.DataFrame(_LAST["stats"]).to_parquet(
        OUT / f"stats_{protocol}_{season}_{P.MODEL_VERSION}{tag}.parquet")
    print(f"[c1r] harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, "
          f"{len(_LAST['stats'])} stat rows, skipped {len(meta['skipped'])}, tip_unknown "
          f"{_LAST.get('tip_unknown', 0)}, {time.time() - t0:.0f}s -> {f}", flush=True)


# ============================================================================== frame diag

def _frame_table(f: pd.DataFrame, v1, label: str, rng) -> list:
    from courtside.proj import copula as C
    rows = []
    for s in C.COMP_STATS:
        y, m = f[f"y_{s}"].to_numpy(float), f[f"m_{s}"].to_numpy(float)
        ok = m > 0.01
        y, m, tier = y[ok], m[ok], f["tier"].to_numpy(int)[ok]
        c2 = np.maximum(np.interp(m, v1.disp_mu[s], v1.disp_c2[s]), C.C2_MIN)
        pit = C.nb_pit(y, m, c2, rng.random(len(y)))
        for t in (-1, 0, 1, 2):
            sel = np.ones(len(y), bool) if t < 0 else tier == t
            if sel.sum() < 200:
                continue
            r = y[sel] - m[sel]
            vmod = m[sel] + c2[sel] * m[sel] ** 2
            c2e = float((r.var() - m[sel].mean()) / np.mean(m[sel] ** 2))
            c2m = float(np.mean(c2[sel] * m[sel] ** 2) / np.mean(m[sel] ** 2))
            rows.append({"frame": label, "stat": s, "tier": "all" if t < 0 else C.TIERS[t],
                         "n": int(sel.sum()), "mean_m": float(m[sel].mean()),
                         "bias": float(r.mean()), "sd_ratio": float(np.sqrt(vmod.mean() / np.mean(r ** 2))),
                         "c2_emp": c2e, "c2_model": c2m,
                         "cov80": float(((pit[sel] >= 0.1) & (pit[sel] <= 0.9)).mean()),
                         "cov50": float(((pit[sel] >= 0.25) & (pit[sel] <= 0.75)).mean())})
    return rows


def _wl(f: pd.DataFrame, v1) -> pd.DataFrame:
    """The frame's means with v1's winner / loser shift (v1r without a shift: unchanged)."""
    from courtside.proj import copula as C
    return C.apply_wl(f, v1) if v1.wl_factor else f


def cmd_fdiag(season: str) -> None:
    """Module doc: `fdiag`. Arm = the environment switches (COPULA v1 or v1r)."""
    from courtside import db
    from courtside.proj import copula as C
    from courtside.proj import pipeline as P
    conn = db.connect()
    b = P.load_bundle(conn, season, None, log=print)
    v1 = b.copula_v1
    rng = np.random.default_rng(5)
    rows = []
    s0 = int(season[:4])
    nxt = f"{s0 + 1}-{(s0 + 2) % 100:02d}"
    prev = f"{s0 - 1}-{s0 % 100:02d}"
    # test season S: EB rates (v1's construction) and the context arm's rates (OOF: model < S)
    f_eb = C.v1_frame(conn, nxt, b.state, b.copula.alpha, n_seasons=1)
    rows += _frame_table(_wl(f_eb, v1), v1, f"test {season} EB", rng)
    f_cx = C.v1r_frame(conn, season, b.state, b.copula.alpha, b.context, seasons=[season])
    rows += _frame_table(_wl(f_cx, v1), v1, f"test {season} ctx", rng)
    # training season S-1: EB, context in-sample (model of S), context OOF (model of S-1)
    f_tr = C.v1r_frame(conn, season, b.state, b.copula.alpha, b.context, seasons=[prev])
    rows += _frame_table(_wl(f_tr, v1), v1, f"train {prev} ctx in-sample", rng)
    f_tre = C.v1r_frame(conn, season, b.state, b.copula.alpha, None, seasons=[prev])
    rows += _frame_table(_wl(f_tre, v1), v1, f"train {prev} EB", rng)
    pb = P.cache_dir() / f"bundle_{prev}_v1_cx.json"
    if pb.exists():
        bp = P.Bundle.from_json(pb.read_text())
        f_oof = C.v1r_frame(conn, prev, bp.state, bp.copula.alpha, bp.context, seasons=[prev])
        rows += _frame_table(_wl(f_oof, v1), v1, f"train {prev} ctx OOF", rng)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fdiag_{season}_{P.MODEL_VERSION}.json").write_text(d.to_json(orient="records"))
    pd.set_option("display.width", 200)
    print(d.round(4).to_string(index=False))


# ============================================================================== report

def _pitdev(u) -> float:
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    if not len(u):
        return float("nan")
    h = np.histogram(np.clip(u, 0, 1 - 1e-12), bins=10, range=(0, 1))[0]
    return float(np.abs(h / h.sum() - 0.1).max())


def pair_rows(base: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    a = base.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    b = on.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    cols = ["mean", "q10", "q25", "q75", "q90", "crps", "pit", "exp_min", "p_play"]
    d = a.merge(b[["date", "player_id"] + cols], on=["date", "player_id"],
                suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool)].reset_index(drop=True)
    d = d[(d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)].reset_index(drop=True)
    d["tier4"] = tier4(d["exp_min_off"] / d["p_play_off"].clip(lower=0.05))
    return d


def paired(d: pd.DataFrame, f) -> dict:
    z0, z1 = np.asarray(f("off"), float), np.asarray(f("on"), float)
    ok = np.isfinite(z0) & np.isfinite(z1)
    cl = d["date"].to_numpy()[ok]
    m0, _ = cl_mean(z0[ok], cl)
    dd, se = cl_mean(z1[ok] - z0[ok], cl)
    return {"off": m0, "on": float(z1[ok].mean()), "diff": dd, "se": se,
            "z": dd / se if se > 0 else float("nan"), "n": int(ok.sum())}


def dk_metrics(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(float)
    out = {"n": len(d)}
    defs = {"dk_crps": lambda a: d[f"crps_{a}"],
            "dk_mae": lambda a: np.abs(y - d[f"mean_{a}"]),
            "dk_bias": lambda a: d[f"mean_{a}"] - y,
            "dk_cov80": lambda a: ((d[f"q10_{a}"] <= y) & (y <= d[f"q90_{a}"])).astype(float),
            "dk_cov50": lambda a: ((d[f"q25_{a}"] <= y) & (y <= d[f"q75_{a}"])).astype(float),
            "dk_width80": lambda a: d[f"q90_{a}"] - d[f"q10_{a}"]}
    if len(d) >= 2:
        for k, f in defs.items():
            out[k] = paired(d, f)
    for a in ("off", "on"):
        out[f"pitdev_{a}"] = _pitdev(d[f"pit_{a}"])
    return out


def stat_metrics(s: pd.DataFrame) -> dict:
    """Calibration of one arm's per-stat rows (played; randomised-PIT coverage)."""
    if not len(s):
        return {}
    from courtside.proj.metrics import rate_se
    cl = s["date"].to_numpy()
    pit = s["pit"].to_numpy(float)
    c80 = rate_se(((pit >= 0.1) & (pit <= 0.9)).astype(float), cl)
    c50 = rate_se(((pit >= 0.25) & (pit <= 0.75)).astype(float), cl)
    err2 = (s["y"] - s["mean"]) ** 2
    var = s["sd"] ** 2
    out = {"n": int(len(s)), "cov80": c80["rate"], "cov80_se": c80["se"], "cov50": c50["rate"],
           "pit_mean": float(np.nanmean(pit)), "pitdev": _pitdev(pit),
           "sd_ratio": float(np.sqrt(var.mean() / err2.mean())),
           "bias": float((s["mean"] - s["y"]).mean()), "crps": float(s["crps"].mean()),
           "share_minutes_var": float(s["var_between"].sum() / max(var.sum(), 1e-9))}
    at = s[s["at_pit"].notna()]
    if len(at):
        ap = at["at_pit"].to_numpy(float)
        out.update({"at_n": int(len(at)),
                    "at_cov80": float(((ap >= 0.1) & (ap <= 0.9)).mean()),
                    "at_cov50": float(((ap >= 0.25) & (ap <= 0.75)).mean()),
                    "at_sd_ratio": float(np.sqrt((at["at_sd"] ** 2).mean()
                                                 / ((at["y"] - at["at_mean"]) ** 2).mean())),
                    "at_bias": float((at["at_mean"] - at["y"]).mean())})
    return out


def stat_paired(s0: pd.DataFrame, s1: pd.DataFrame) -> dict:
    """Per-stat CRPS ON - BASE on the same played player-games."""
    k = ["date", "player_id", "stat"]
    d = s0[k + ["crps", "tier4"]].merge(s1[k + ["crps"]], on=k, suffixes=("_off", "_on"))
    out = {}
    for s, g in d.groupby("stat"):
        out[s] = paired(g, lambda a, g=g: g[f"crps_{a}"])
    return out


def cmd_report(base: str, on: str, label: str) -> None:
    a, b = pd.read_parquet(base), pd.read_parquet(on)
    d = pair_rows(a, b)
    res = {"base": Path(base).name, "on": Path(on).name, "all": dk_metrics(d)}
    for t in TIER_NAMES:
        res[t] = dk_metrics(d[d["tier4"] == t])
    tier_of = dict(zip(zip(d["date"], d["player_id"]), d["tier4"]))
    st = {}
    for arm, f in (("off", base), ("on", on)):
        p = Path(f).with_name(Path(f).name.replace("rows_", "stats_"))
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        s["tier4"] = [tier_of.get((dd, pp), "") for dd, pp in zip(s["date"], s["player_id"])]
        s = s[s["tier4"] != ""]
        st[arm] = s
        tab = {}
        for stat, g in s.groupby("stat"):
            tab[stat] = {"all": stat_metrics(g)}
            for t in TIER_NAMES:
                tab[stat][t] = stat_metrics(g[g["tier4"] == t])
        res[f"stats_{arm}"] = tab
    if "off" in st and "on" in st:
        res["stats_paired"] = stat_paired(st["off"], st["on"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rep_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label}: {Path(base).name} -> {Path(on).name} (rows {len(d)})")
    keys = ["dk_crps", "dk_mae", "dk_bias", "dk_cov80", "dk_cov50", "dk_width80"]
    rows = []
    for n in ("all",) + TIER_NAMES:
        r = res[n]
        row = {"rows": n, "n": r["n"]}
        for k in keys:
            if k in r:
                row[k] = f"{r[k]['off']:.3f} {r[k]['diff']:+.4f} ({r[k]['z']:+.1f})"
        row["pitdev"] = f"{r['pitdev_off']:.4f}>{r['pitdev_on']:.4f}"
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("rows").to_string())
    for arm in ("off", "on"):
        tab = res.get(f"stats_{arm}")
        if not tab:
            continue
        print(f"\nper-stat ({arm}), played: cov80 / cov50 / sd ratio / PIT dev / minutes share "
              f"of variance; at the actual minutes: cov80 / sd ratio")
        rr = []
        for stat, t in tab.items():
            for n in ("all",) + TIER_NAMES:
                m = t.get(n) or {}
                if not m:
                    continue
                rr.append({"stat": stat, "tier": n, "n": m["n"], "cov80": round(m["cov80"], 3),
                           "cov50": round(m["cov50"], 3), "sd_ratio": round(m["sd_ratio"], 3),
                           "pitdev": round(m["pitdev"], 4), "bias": round(m["bias"], 3),
                           "min_share": round(m["share_minutes_var"], 3),
                           "at_cov80": round(m.get("at_cov80", np.nan), 3),
                           "at_sd_ratio": round(m.get("at_sd_ratio", np.nan), 3),
                           "crps": round(m["crps"], 4)})
        print(pd.DataFrame(rr).to_string(index=False))
    if "stats_paired" in res:
        print("\nper-stat CRPS ON - BASE (played): " + ", ".join(
            f"{s} {r['diff']:+.4f} ({r['z']:+.1f})" for s, r in res["stats_paired"].items()))


def cmd_summary(labels: list) -> None:
    rs = [json.loads((OUT / f"rep_{lb}.json").read_text()) for lb in labels]

    def cell(r, k, fmt="{off:.3f} / {diff:+.4f} ({z:+.1f})"):
        return fmt.format(**r[k]) if k in r else ""

    def cov(r, k="dk_cov80"):
        return f"{r[k]['off']:.3f} → {r[k]['on']:.3f}" if k in r else ""

    rows = [("rows", [str(r["all"]["n"]) for r in rs]),
            ("DK CRPS base / Δ (z)", [cell(r["all"], "dk_crps") for r in rs]),
            ("DK MAE base / Δ (z)", [cell(r["all"], "dk_mae") for r in rs]),
            ("DK 80% cov base → ON", [cov(r["all"]) for r in rs]),
            ("DK 50% cov base → ON", [cov(r["all"], "dk_cov50") for r in rs]),
            ("DK PIT max dev", [f"{r['all']['pitdev_off']:.4f} → {r['all']['pitdev_on']:.4f}"
                                for r in rs]),
            ("CRPS z star/starter/rotation/bench",
             [" / ".join(f"{r[t]['dk_crps']['z']:+.1f}" for t in TIER_NAMES) for r in rs]),
            ("80% cov star/starter/rotation/bench, ON",
             [" / ".join(f"{r[t]['dk_cov80']['on']:.3f}" for t in TIER_NAMES) for r in rs])]
    for s in PSTATS:
        vals = []
        for r in rs:
            m = (r.get("stats_on") or {}).get(s, {}).get("all")
            vals.append(f"{m['cov80']:.3f} / {m['cov50']:.3f} / {m['sd_ratio']:.3f} / "
                        f"{m['pitdev']:.3f}" if m else "")
        rows.append((f"{s}: cov80 / cov50 / sd ratio / PIT dev (ON, played)", vals))
    print("| | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for name, vals in rows:
        print(f"| {name} | " + " | ".join(vals) + " |")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    fd = sub.add_parser("fdiag")
    fd.add_argument("season")
    rp = sub.add_parser("report")
    rp.add_argument("base")
    rp.add_argument("on")
    rp.add_argument("--label", default="cmp")
    su = sub.add_parser("summary")
    su.add_argument("labels", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "harness":
        cmd_harness(a.season, a.protocol)
    elif a.cmd == "fdiag":
        cmd_fdiag(a.season)
    elif a.cmd == "report":
        cmd_report(a.base, a.on, a.label)
    else:
        cmd_summary(a.labels)


if __name__ == "__main__":
    main()
