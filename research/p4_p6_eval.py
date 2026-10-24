"""P4 weighted reconciliation (`COURTSIDE_RECON_WEIGHTED`, `+rw`) and P6 player-specific minutes
spread (`COURTSIDE_MIN_SPREAD`, `+ms`): fit and evaluation (IMPROVEMENTS.md P4 / P6).

Docs: docs/research/p4-p6.md.  Code under test: `proj/reconcile.py` (`_reconcile_weighted`),
`proj/copula.py` (`_reconcile`, `ScenarioMinutes.project`), `proj/minutes.py` (`fit_spread`,
`sample_minutes`), `proj/pipeline.py` (`_load_bundle_ms`, `build_games`, `_versions`).

    python research/p4_p6_eval.py harness S PROTO [--capture]
        backtest rows (4,000 sims) of the arm in the environment, with the minutes columns
        (`minutes_fix_eval._patch_capture`: sims' minutes mean / q10 / q90 / PIT / CRPS)
        -> OUT/rows_<P>_<S>_<MV>.parquet; --capture also keeps each team-game's reconciliation
        input (e_i = E[rate x minutes x env] per player, EB points rate, evidence minutes, kappa,
        the target) -> OUT/recon_<P>_<S>_<MV>.parquet (the run itself is unchanged)
    python research/p4_p6_eval.py rwfit DIR [--arm ARM]
        theta per season S on the captures of seasons < S (walk-forward): the grid value with the
        least squared error of the reconciled expected points vs the actual points
        -> src/courtside/proj/recon_weighted_params.json
    python research/p4_p6_eval.py fitbundle S
        the arm's bundle of S (fits `_ms` when COURTSIDE_MIN_SPREAD=1)
    python research/p4_p6_eval.py dk OFF ON LABEL
        paired ON - OFF DK CRPS / MAE / bias and minutes CRPS / MAE (two-way date x player SEs),
        by tier; minutes PIT deviation and 80 % coverage by tier

Environment: COURTSIDE_DATA (bundles + DB), P46_OUT (default var/tmp/p46/out). Heavy: Mac scratch
(`courtside-compute/p46`).
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
sys.path.insert(0, str(Path(__file__).parent))
OUT = Path(os.environ.get("P46_OUT", ROOT / "var" / "tmp" / "p46" / "out"))
ARM = "pathA-v0.3+mf+ep+p2+rx+rb+tr"
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
# capture datasets -> their season; target S gets every dataset of seasons < S
DATASETS = (("avail", "2018-19"), ("avail", "2019-20"), ("avail", "2020-21"),
            ("avail", "2023-24"), ("avail", "2024-25"), ("live", "2025-26"))
TARGETS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
           "2026-27")
THETA_GRID = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e6)


def say(m: str) -> None:
    print(f"[p46] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import db
    return db.connect()


# ============================================================================== capture

_CAP: dict = {"rows": [], "rates": {}, "kappa": None, "day": None}


def _patch_recon_capture() -> None:
    """Record every team's reconciliation input. Wraps `state.rate_frame` (the evidence minutes
    n_pts of the panel's players, by player id) and `copula._reconcile` (calls the original, then
    stores e_i); nothing of the run changes."""
    from courtside.proj import backtest as B
    from courtside.proj import copula as C
    from courtside.proj import state as ST
    if getattr(C._reconcile, "_p46", False):
        return
    rf, orig, rd = ST.rate_frame, C._reconcile, B.SeasonRun.run_date

    def rate_frame(conn, queries, params=None, *a, **k):
        w = rf(conn, queries, params, *a, **k)
        if "n_pts" in w:
            _CAP["rates"].update(zip(w["player_id"].astype(np.int64).tolist(),
                                     w["n_pts"].astype(float).tolist()))
            p = params or ST.DEFAULT_PARAMS
            _CAP["kappa"] = float(p["stats"]["pts"]["kappa"])
        return w

    def reconcile(ts, ro, raw_pts, raw_oth, g, cp):
        rec = orig(ts, ro, raw_pts, raw_oth, g, cp)
        gg = np.maximum(np.asarray(g, float), 0.05) ** cp.env_weight
        e = (raw_pts * gg[:, None]).mean(0) if raw_pts.size else np.zeros(raw_pts.shape[1])
        pid = ts.players["player_id"].to_numpy(np.int64)
        _CAP["rows"].append(pd.DataFrame({
            "date": _CAP["day"], "team": ts.team, "player_id": pid, "e": e,
            "r_pts": ts.players["r_pts"].to_numpy(float),
            "n_pts": [_CAP["rates"].get(int(p), np.nan) for p in pid],
            "kappa": _CAP["kappa"], "target": float(rec["target"]),
            "scale": float(rec["scale"]), "season": cp.season}))
        return rec

    def run_date(self, day):
        _CAP["day"], _CAP["rates"] = str(day)[:10], {}
        return rd(self, day)

    reconcile._p46 = True
    ST.rate_frame, C._reconcile, B.SeasonRun.run_date = rate_frame, reconcile, run_date


def _actual_pts(conn, cap: pd.DataFrame) -> np.ndarray:
    dates = sorted(cap["date"].unique())
    g = pd.read_sql_query(
        f"SELECT player_id, game_date AS date, pts FROM player_games WHERE game_date IN "
        f"({','.join('?' * len(dates))})", conn, params=dates)
    g = g.groupby(["player_id", "date"], as_index=False)["pts"].sum()
    m = cap[["player_id", "date"]].merge(g, on=["player_id", "date"], how="left")
    return m["pts"].fillna(0.0).to_numpy(float)


# ============================================================================== harness

def cmd_harness(season: str, protocol: str, capture: bool, theta: float | None = None,
                unit_spread: bool = False) -> None:
    """`theta` (ablation): every season's theta of the weighted reconciliation set to it;
    `unit_spread` (ablation): the +ms bundle's mu / sigma with every player's s_lo = s_hi = 1.
    Both tag the output file (`_th<theta>`, `_unit`)."""
    import minutes_fix_eval as MFE

    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    from courtside.proj import reconcile as RC
    M.GBM_PARAMS["num_threads"] = 1
    MFE._patch_capture()
    tag = ""
    if theta is not None:
        RC.rw_theta = lambda season: float(theta)
        tag += f"_th{theta:g}"
    if unit_spread:
        M.spread_columns = lambda p, params, absent, plan: (np.ones(len(p)), np.ones(len(p)))
        tag += "_unit"
    if capture:
        _patch_recon_capture()
    conn = _conn()
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
    if capture:
        if not _CAP["rows"]:
            say("capture: no rows (every date came from the cache?)")
        else:
            cap = pd.concat(_CAP["rows"], ignore_index=True)
            cap["y_pts"] = _actual_pts(conn, cap)
            cap.to_parquet(OUT / f"recon_{protocol}_{season}_{P.MODEL_VERSION}.parquet")
            say(f"capture: {len(cap)} player rows, {cap.groupby(['date', 'team']).ngroups} "
                f"team-games")
    say(f"harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
        f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}")


def cmd_fitbundle(season: str) -> None:
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    t0 = time.time()
    b = P.load_bundle(_conn(), season, None, log=lambda m: print(m, flush=True))
    say(f"bundle {season} ({P.MODEL_VERSION}) in {time.time() - t0:.0f}s; spread "
        f"{bool(b.minutes.spread)} mu {np.round(b.minutes.mu, 3).tolist()} sigma "
        f"{np.round(b.minutes.sigma, 3).tolist()}; diag "
        f"{json.dumps(b.minutes.diagnostics.get('spread', {}))}")


# ============================================================================== rw fit

def rw_loss(cap: pd.DataFrame, theta: float) -> tuple[float, float, int]:
    """(sum of squared errors of the reconciled expected points, the same with the uniform c,
    team-games) over the captured team-games."""
    from courtside.proj import reconcile as RC
    sse = sse_u = 0.0
    n = 0
    for _, d in cap.groupby(["date", "team"], sort=False):
        e = d["e"].to_numpy(float)
        tot = e.sum()
        if tot <= 0:
            continue
        t = float(d["target"].iloc[0])
        y = d["y_pts"].to_numpy(float)
        v = RC.rate_variance(e, d["r_pts"].to_numpy(float), d["n_pts"].to_numpy(float),
                             float(d["kappa"].iloc[0]))
        c = RC.weighted_factors(e, v, t, theta)
        left = float((e * c).sum())
        c = c * (float(np.clip(t / left, *RC.SCALE_BOUNDS)) if left > 0 else 1.0)
        cu = float(np.clip(t / tot, *RC.SCALE_BOUNDS))
        sse += float(((y - e * c) ** 2).sum())
        sse_u += float(((y - e * cu) ** 2).sum())
        n += 1
    return sse, sse_u, n


def cmd_rwfit(rows_dir: str, arm: str) -> None:
    from courtside.proj import reconcile as RC
    caps = {}
    for proto, s in DATASETS:
        fs = sorted(Path(rows_dir).glob(f"recon_{proto}_{s}_*.parquet"))
        fs = [f for f in fs if arm in f.name] or fs
        if not fs:
            say(f"missing recon_{proto}_{s}; skipped")
            continue
        caps[(proto, s)] = pd.read_parquet(fs[0])
        say(f"{proto} {s}: {len(caps[(proto, s)])} rows ({fs[0].name})")
    curves = {}
    for k, cap in caps.items():
        curves[k] = {th: rw_loss(cap, th) for th in THETA_GRID}
        u = curves[k][THETA_GRID[0]]
        say(f"{k}: team-games {u[2]}; sse per row " + " ".join(
            f"{th:g}:{v[0] / len(cap):.4f}" for th, v in curves[k].items())
            + f" uniform {u[1] / len(cap):.4f}")
    out = {"version": "rw-v1", "grid": list(THETA_GRID), "arm": arm,
           "bounds": list(RC.PLAYER_BOUNDS),
           "method": "theta = argmin over the grid of the summed squared error of the reconciled "
                     "expected points (e_i c_i) vs the actual points, team-games of the captured "
                     "datasets of seasons < S (walk-forward); 1e6 = the uniform factor",
           "seasons": {}}
    for S in TARGETS:
        tr = [k for k in curves if k[1] < S]
        if not tr:
            say(f"{S}: no training rows; not written (fallback theta)")
            continue
        tot = {th: sum(curves[k][th][0] for k in tr) for th in THETA_GRID}
        uni = sum(curves[k][THETA_GRID[0]][1] for k in tr)
        nrows = sum(len(caps[k]) for k in tr)
        best = min(THETA_GRID, key=lambda th: tot[th])
        out["seasons"][S] = {"theta": best, "train": [f"{p}:{s}" for p, s in tr],
                             "rows": nrows,
                             "mse": {f"{th:g}": round(v / nrows, 5) for th, v in tot.items()},
                             "mse_uniform": round(uni / nrows, 5)}
        say(f"{S}: theta {best:g} on {[f'{p}:{s}' for p, s in tr]}; mse {tot[best] / nrows:.4f} "
            f"vs uniform {uni / nrows:.4f}")
    RC.RW_PATH.write_text(json.dumps(out, indent=1) + "\n")
    RC.rw_theta.cache_clear()
    say(f"wrote {RC.RW_PATH}")


# ============================================================================== paired

MIN_COLS = ("min_mean", "min_p", "min_q10", "min_q90", "min_pit", "min_crps")


def _pair(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    k = ["date", "player_id"]
    a = off.dropna(subset=["player_id"]).drop_duplicates(k)
    b = on.dropna(subset=["player_id"]).drop_duplicates(k)
    cols = ["mean", "q10", "q90", "crps", "pit", "exp_min", "p_play"] + \
        [c for c in MIN_COLS if c in a and c in b]
    d = a.merge(b[k + cols], on=k, suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool) & d["crps_off"].notna() & d["crps_on"].notna()]
    return d.reset_index(drop=True)


def _metrics(d: pd.DataFrame) -> dict:
    from courtside.proj.metrics import paired_cluster2
    out = {"n": int(len(d))}
    if len(d) < 20:
        return out
    y = d["y"].to_numpy(float)
    m = d["minutes"].fillna(0).to_numpy(float)
    dates, pids = d["date"].astype(str).to_numpy(), d["player_id"].to_numpy()
    defs = {"crps": lambda a: d[f"crps_{a}"].to_numpy(float),
            "mae": lambda a: np.abs(d[f"mean_{a}"].to_numpy(float) - y),
            "bias": lambda a: d[f"mean_{a}"].to_numpy(float) - y}
    if "min_crps_off" in d:
        defs.update({
            "min_crps": lambda a: d[f"min_crps_{a}"].to_numpy(float),
            "min_mae": lambda a: np.abs(d[f"min_mean_{a}"].to_numpy(float) - m),
            "min_cov80": lambda a: ((d[f"min_q10_{a}"] <= m) & (m <= d[f"min_q90_{a}"]))
            .to_numpy(float)})
    for k, f in defs.items():
        z0, z1 = f("off"), f("on")
        ok = np.isfinite(z0) & np.isfinite(z1)
        if ok.sum() < 20:
            continue
        pc = paired_cluster2(z1[ok] - z0[ok], dates[ok], pids[ok])
        out[k] = {"off": float(z0[ok].mean()), "diff": pc["mean"], "se": pc["se"],
                  "se_date": pc["se_c1"], "z": pc["mean"] / pc["se"] if pc["se"] > 0 else np.nan}
    if "min_pit_off" in d:
        for a in ("off", "on"):
            u = d.loc[d["minutes"] > 0, f"min_pit_{a}"].dropna().clip(0, 1 - 1e-12)
            h = np.histogram(u, bins=10, range=(0, 1))[0]
            out[f"min_pit_{a}"] = {"maxdev": float(np.abs(h / max(h.sum(), 1) - 0.1).max()),
                                   "lt10": float((u < 0.1).mean()), "gt90": float((u > 0.9).mean()),
                                   "n_played": int(len(u))}
    return out


EV_BANDS = (("ev<300", 0, 300), ("ev300-1500", 300, 1500), ("ev>=1500", 1500, 1e12))


def cmd_dk(off: str, on: str, label: str, recon: str | None = None) -> None:
    d = _pair(pd.read_parquet(off), pd.read_parquet(on))
    if recon:                     # the OFF capture: evidence minutes of each row's player
        cap = pd.read_parquet(recon)[["date", "player_id", "n_pts"]]
        cap = cap.drop_duplicates(["date", "player_id"])
        d["date_k"] = d["date"].astype(str).str[:10]
        d = d.merge(cap.rename(columns={"date": "date_k"}).astype({"player_id": float}),
                    on=["date_k", "player_id"], how="left")
    d["tier4"] = "bench"
    mc = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    for name, lo, hi in TIERS:
        d.loc[(mc >= lo) & (mc < hi), "tier4"] = name
    d["pop"] = (d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)
    d["shift"] = d["mean_on"] - d["mean_off"]
    res = {"label": label, "off": off, "on": on}
    for popn, q in (("pop10", d[d["pop"]]), ("all", d)):
        r = {"all": _metrics(q)}
        for t, _, _ in TIERS:
            r[f"tier:{t}"] = _metrics(q[q["tier4"] == t])
        if "n_pts" in q:
            nv = q["n_pts"].fillna(0.0)
            for name, lo, hi in EV_BANDS:
                r[name] = _metrics(q[(nv >= lo) & (nv < hi)])
        r["_shift"] = {"mean": float(q["shift"].mean()), "mean_abs": float(q["shift"].abs().mean())}
        res[popn] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label} (ON - OFF; z two-way date x player)")
    for popn in ("pop10", "all"):
        print(f"### {popn}  shift {res[popn]['_shift']}")
        for k, v in res[popn].items():
            if k.startswith("_") or "crps" not in v:
                continue
            line = f"  {k:14s} n={v['n']:6d}"
            for m in ("crps", "mae", "bias", "min_crps", "min_mae", "min_cov80"):
                if m in v:
                    line += f"  {m} {v[m]['off']:.4f} {v[m]['diff']:+.4f} z {v[m]['z']:+.2f}"
            for a in ("off", "on"):
                if f"min_pit_{a}" in v:
                    p = v[f"min_pit_{a}"]
                    line += f"  pit[{a}] dev {p['maxdev']:.3f} <.1 {p['lt10']:.3f} >.9 {p['gt90']:.3f}"
            print(line)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol")
    h.add_argument("--capture", action="store_true")
    h.add_argument("--theta", type=float, default=None)
    h.add_argument("--unit-spread", action="store_true")
    f = sub.add_parser("rwfit")
    f.add_argument("rows_dir")
    f.add_argument("--arm", default=ARM)
    b = sub.add_parser("fitbundle")
    b.add_argument("season")
    k = sub.add_parser("dk")
    k.add_argument("off")
    k.add_argument("on")
    k.add_argument("label")
    k.add_argument("--recon", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "harness":
        cmd_harness(a.season, a.protocol, a.capture, a.theta, a.unit_spread)
    elif a.cmd == "rwfit":
        cmd_rwfit(a.rows_dir, a.arm)
    elif a.cmd == "fitbundle":
        cmd_fitbundle(a.season)
    else:
        cmd_dk(a.off, a.on, a.label, a.recon)


if __name__ == "__main__":
    main()
