"""P8 usage redistribution (`COURTSIDE_USAGE`, `+us`) and E3 forward selection over the OFF
candidate switches on top of the live stack: harness runs and paired metrics.

Docs: docs/research/p8-e3.md. Code under test: `proj/usage.py` (live switch, `live_model`,
`team_matrix`), `proj/copula.py` (`_usage_means`, `TeamSpec.usage`), `proj/pipeline.py`
(`build_games`, `_versions`, `set_usage`). The candidate switches are set by environment
(Mac scratch `run.sh`), so one harness command serves every arm.

    python research/p8_e3_eval.py harness S PROTO [--tag T]
        backtest rows (4,000 sims) of the arm in the environment with the minutes columns
        (`minutes_fix_eval._patch_capture`) -> OUT/rows_<P>_<S>_<MV><T>.parquet, and every
        team-game's candidate list as projected (player, P(available), mpg, top-2 star,
        regular) -> OUT/absent_<P>_<S>_<MV><T>.parquet (the run itself is unchanged)
    python research/p8_e3_eval.py fitbundle S
        the arm's bundle of S (fits `_bo` / `_mr` / `_c1r` when those switches are on)
    python research/p8_e3_eval.py usagefit S
        usage v1 params of S for the arm's rate state (`usage.live_model`), printed
    python research/p8_e3_eval.py dk LABEL ABSENT OFF ON [ABSENT OFF ON ...]
        paired ON - OFF on identical rows (pooled over the given sets): DK CRPS / MAE / bias,
        minutes CRPS (two-way date x player SEs), by tier, on the focus rows (teammates of an
        absent rotation player; split: a top-2 star absent / only non-stars absent; none
        absent), and the E4 value-ranking metric (salary-implied on RotoGuru sets, DK-FPPG-
        implied elsewhere) on the DFS population, overall and on the focus rows
        -> OUT/dk_<LABEL>.json

Environment: COURTSIDE_DATA (bundles + DB), P8E3_OUT (default var/tmp/p8e3/out). Heavy: Mac
scratch (`courtside-compute/p8e3`).
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
OUT = Path(os.environ.get("P8E3_OUT", ROOT / "var" / "tmp" / "p8e3" / "out"))
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
ROT_MPG = 15.0            # "rotation player": a regular with mpg >= 15
ABSENT_P = 0.5            # absent at the run: P(available) <= 0.5 (avail oracle: 0)
EMIN = 10.0


def say(m: str) -> None:
    print(f"[p8e3] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import db
    return db.connect()


# ============================================================================== capture

_ABS: list = []


def _patch_absent_capture() -> None:
    """Record every team's candidate list as `build_games` hands it to the sim (wraps
    `pipeline.build_games`; nothing of the run changes)."""
    from courtside.proj import pipeline as P
    if getattr(P.build_games, "_p8", False):
        return
    orig = P.build_games

    def build_games(ctx, b, conn, day, games, panel, *a, **k):
        specs = orig(ctx, b, conn, day, games, panel, *a, **k)
        pn = panel.reset_index(drop=True)
        reg = (((pn["app10"].to_numpy(float) >= 7) | (pn["missed_now"].to_numpy(float) < 3))
               & pn["last_date"].notna().to_numpy())
        regd = dict(zip(zip(pn["key"], pn["player_id"].astype(np.int64)), reg))
        for gs in specs:
            for ts in (gs.home, gs.away):
                pl = ts.players
                key = f"{gs.game_id}|{ts.team}"
                pid = pl["player_id"].to_numpy(np.int64)
                _ABS.append(pd.DataFrame({
                    "date": str(day)[:10], "game_id": gs.game_id, "team": ts.team,
                    "player_id": pid, "p_avail": pl["p_avail"].to_numpy(float),
                    "mpg": pl["mpg"].to_numpy(float), "star": pl["star"].to_numpy(bool),
                    "regular": [bool(regd.get((key, int(p)), False)) for p in pid]}))
        return specs

    build_games._p8 = True
    P.build_games = build_games


# ============================================================================== harness

def _patch_pq_lomo(path: str) -> None:
    """+pq on the 2025-26 slice with the offsets fit leaving the month out (the packaged
    `pplay_q_params.json` was fit on the slice itself), as `pplay_q_refit.py slice` does."""
    from courtside.proj import availability as AV
    lomo = {m: AV.QRefitParams.from_json(d) for m, d in json.loads(Path(path).read_text()).items()}
    orig = AV.q_refit_frame

    def patched(df, hist, day, starters, params=None):
        return orig(df, hist, day, starters, lomo[day.strftime("%Y-%m")])

    AV.q_refit_frame = patched
    say(f"pq offsets leave-one-month-out: months {sorted(lomo)}")


def cmd_harness(season: str, protocol: str, tag: str = "", pq_lomo: str | None = None) -> None:
    import minutes_fix_eval as MFE

    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    MFE._patch_capture()
    _patch_absent_capture()
    if pq_lomo:
        _patch_pq_lomo(pq_lomo)
    conn = _conn()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{protocol}_{season}_{P.MODEL_VERSION}{tag}"
    f = OUT / f"rows_{stem}.parquet"
    pl.to_parquet(f)
    if _ABS:
        pd.concat(_ABS, ignore_index=True).to_parquet(OUT / f"absent_{stem}.parquet")
    else:
        say("absent capture: no rows (every date came from the cache?)")
    say(f"harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
        f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}")


def cmd_fitbundle(season: str) -> None:
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    t0 = time.time()
    P.load_bundle(_conn(), season, None, log=lambda m: print(m, flush=True))
    say(f"bundle {season} ({P.MODEL_VERSION}, {P._VS}) in {time.time() - t0:.0f}s")


def cmd_usagefit(season: str) -> None:
    from courtside.proj import pipeline as P
    from courtside.proj import usage as U
    conn = _conn()
    b = P.load_bundle(conn, season, None, log=lambda m: print(m, flush=True))
    p, D = U.live_model(conn, season, b.state, log=lambda m: print(m, flush=True))
    say(f"usage {season}: {U.live_params_path(season, b.state)}")
    print(json.dumps({k: getattr(p, k) for k in ("beta", "alpha", "lam", "k_stint", "k_box")},
                     default=float))


# ============================================================================== paired

def focus_flags(absent: pd.DataFrame) -> pd.DataFrame:
    """Per (date, team, player): flags from the team's candidate list at the run. A teammate is
    an 'absent rotation player' when he is a regular with mpg >= ROT_MPG and P(available) <=
    ABSENT_P. focus = at least one such teammate; f_star = one of them is a top-2 star (the
    tier bump fires); f_nonstar = focus without an absent star (only usage v1 reacts);
    f_none = no absent regular with mpg >= 10 at all."""
    a = absent.drop_duplicates(["date", "team", "player_id"]).copy()
    out_ = a["p_avail"] <= ABSENT_P
    a["ar"] = out_ & a["regular"] & (a["mpg"] >= ROT_MPG)
    a["as"] = out_ & a["regular"] & a["star"]
    a["a10"] = out_ & a["regular"] & (a["mpg"] >= 10.0)
    g = a.groupby(["date", "team"])[["ar", "as", "a10"]].transform("sum")
    n_ar = g["ar"] - a["ar"]
    n_as = g["as"] - a["as"]
    n_10 = g["a10"] - a["a10"]
    return pd.DataFrame({"date": a["date"], "player_id": a["player_id"].astype(float),
                         "n_abs_rot": n_ar, "focus": n_ar > 0, "f_star": n_as > 0,
                         "f_nonstar": (n_ar > 0) & (n_as == 0), "f_none": n_10 == 0,
                         "self_out": out_})


def _pair(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    k = ["date", "player_id"]
    a = off.dropna(subset=["player_id"]).drop_duplicates(k)
    b = on.dropna(subset=["player_id"]).drop_duplicates(k)
    cols = ["mean", "q10", "q90", "crps", "pit", "exp_min", "p_play"] + \
        [c for c in ("min_mean", "min_crps") if c in a and c in b]
    d = a.merge(b[k + cols], on=k, suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool) & d["crps_off"].notna() & d["crps_on"].notna()]
    return d.reset_index(drop=True)


def _metrics(d: pd.DataFrame) -> dict:
    from courtside.proj.metrics import paired_cluster2
    out = {"n": int(len(d))}
    if len(d) < 20:
        return out
    y = d["y"].to_numpy(float)
    dates, pids = d["dkey"].to_numpy(), d["player_id"].to_numpy()
    defs = {"crps": lambda a: d[f"crps_{a}"].to_numpy(float),
            "mae": lambda a: np.abs(d[f"mean_{a}"].to_numpy(float) - y),
            "bias": lambda a: d[f"mean_{a}"].to_numpy(float) - y}
    if "min_crps_off" in d:
        defs["min_crps"] = lambda a: d[f"min_crps_{a}"].to_numpy(float)
    for k, f in defs.items():
        z0, z1 = f("off"), f("on")
        ok = np.isfinite(z0) & np.isfinite(z1)
        if ok.sum() < 20:
            continue
        pc = paired_cluster2(z1[ok] - z0[ok], dates[ok], pids[ok], by=d["set"].to_numpy()[ok])
        out[k] = {"off": float(z0[ok].mean()), "diff": pc["mean"], "se": pc["se"],
                  "se_date": pc["se_c1"], "z": pc["mean"] / pc["se"] if pc["se"] > 0 else np.nan,
                  "by_set": pc.get("by_mean", {})}
    return out


def _value(d: pd.DataFrame) -> dict:
    """E4 value ranking ON vs OFF on the DFS population of each set (anchor: the real salary on
    RotoGuru sets, DK FPPG to date elsewhere), the anchor's line fit per date on the whole
    population; overall and on the focus rows. SE clustered by date."""
    from courtside.proj import metrics as MT
    parts = []
    for sname, q in d.groupby("set", sort=False):
        rg = q["salary"].notna().mean() > 0.5
        anchor = q["salary"] if rg else q["b_fppg"]
        if rg:
            pop = MT.dfs_population(q["salary"], q["mean_off"], q["mean_on"])
        else:
            pop = MT.dfs_population(None, q["mean_off"], q["mean_on"], q["b_fppg"])
        # as E4 on live grades: players known out at the run (P(available) <= 0.5; the avail
        # oracle's absent regulars) are nobody's DFS pick, so they leave the population
        live_ok = (q["p_avail"].fillna(1.0) > ABSENT_P).to_numpy()
        q = q[pop & live_ok & anchor.notna().to_numpy() & q["y"].notna().to_numpy()].copy()
        q["anchor"] = (q["salary"] if rg else q["b_fppg"]).astype(float)
        q["s"] = MT.salary_implied(q["y"], q["anchor"], q["dkey"])
        q["kind"] = "salary" if rg else "fppg"
        parts.append(q[np.isfinite(q["s"])])
    if not parts:
        return {}
    v = pd.concat(parts, ignore_index=True)
    out = {"anchor": sorted(v["kind"].unique().tolist())}
    for name, m in (("all", np.ones(len(v), bool)), ("focus", v["focus"].fillna(False).to_numpy(bool)),
                    ("f_nonstar", v["f_nonstar"].fillna(False).to_numpy(bool)),
                    ("f_star", v["f_star"].fillna(False).to_numpy(bool))):
        q = v[m]
        if len(q) < 30:
            continue
        r = MT.corr_paired_cluster(q["mean_on"] - q["s"], q["mean_off"] - q["s"],
                                   q["y"] - q["s"], q["dkey"])
        out[name] = {k: r[k] for k in ("r_a", "r_b", "diff", "se", "z", "n", "n_clusters")}
        pc = MT.paired_cluster2(q["crps_on"].to_numpy(float) - q["crps_off"].to_numpy(float),
                                q["dkey"].to_numpy(), q["player_id"].to_numpy())
        out[name]["crps"] = {"diff": pc["mean"], "se": pc["se"],
                             "z": pc["mean"] / pc["se"] if pc["se"] > 0 else np.nan}
    return out


def load_set(absent: str, off: str, on: str, sname: str) -> pd.DataFrame:
    d = _pair(pd.read_parquet(off), pd.read_parquet(on))
    d["date"] = d["date"].astype(str).str[:10]
    if absent and absent != "-":
        ff = focus_flags(pd.read_parquet(absent))
        d = d.merge(ff, on=["date", "player_id"], how="left")
    for c in ("focus", "f_star", "f_nonstar", "f_none"):
        if c not in d:
            d[c] = False
        d[c] = d[c].fillna(False).astype(bool)
    d["set"] = sname
    d["dkey"] = sname + ":" + d["date"]
    return d


def cmd_dk(label: str, triples: list[str]) -> None:
    if len(triples) % 3:
        raise SystemExit("dk LABEL ABSENT OFF ON [ABSENT OFF ON ...]")
    sets = []
    for i in range(0, len(triples), 3):
        ab, off, on = triples[i:i + 3]
        sname = Path(off).name.split("_")[1] + "_" + Path(off).name.split("_")[2]
        sets.append(load_set(ab, off, on, sname))
    d = pd.concat(sets, ignore_index=True)
    d["tier4"] = "bench"
    mc = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    for name, lo, hi in TIERS:
        d.loc[(mc >= lo) & (mc < hi), "tier4"] = name
    d["pop"] = (d["exp_min_off"] >= EMIN) | (d["exp_min_on"] >= EMIN)
    d["shift"] = d["mean_on"] - d["mean_off"]
    res = {"label": label, "files": triples}
    q = d[d["pop"]]
    r = {"all": _metrics(q)}
    for t, _, _ in TIERS:
        r[f"tier:{t}"] = _metrics(q[q["tier4"] == t])
    for c in ("focus", "f_star", "f_nonstar", "f_none"):
        r[c] = _metrics(q[q[c]])
        r[c]["share"] = float(q[c].mean())
    for t, _, _ in TIERS:
        r[f"focus:{t}"] = _metrics(q[q["focus"] & (q["tier4"] == t)])
    r["_shift"] = {"mean": float(q["shift"].mean()), "mean_abs": float(q["shift"].abs().mean()),
                   "focus_mean_abs": float(q.loc[q["focus"], "shift"].abs().mean())
                   if q["focus"].any() else None}
    res["pop10"] = r
    res["value"] = _value(d)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label} (ON - OFF; z two-way date x player); shift {r['_shift']}")
    for k, v in r.items():
        if k.startswith("_") or "crps" not in v:
            continue
        line = f"  {k:16s} n={v['n']:6d}"
        for m in ("crps", "mae", "bias", "min_crps"):
            if m in v:
                line += f"  {m} {v[m]['diff']:+.4f} z {v[m]['z']:+.2f}"
        if len(v["crps"].get("by_set", {})) > 1:
            line += "  by set " + " ".join(f"{s}:{x:+.4f}" for s, x in v["crps"]["by_set"].items())
        print(line)
    for k, v in res["value"].items():
        if k == "anchor":
            print(f"  value anchor {v}")
            continue
        z = v["z"]
        print(f"  value {k:10s} n={v['n']:6d} r on {v['r_a']:.4f} off {v['r_b']:.4f} diff "
              f"{v['diff']:+.4f} (SE {v['se']:.4f}, z {z if z is None else round(z, 2)}); DFS-pop "
              f"CRPS {v['crps']['diff']:+.4f} z {v['crps']['z']:+.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol")
    h.add_argument("--tag", default="")
    h.add_argument("--pq-lomo", default=None, help="PQ_LOMO json (pplay_q_refit.py fit)")
    b = sub.add_parser("fitbundle")
    b.add_argument("season")
    u = sub.add_parser("usagefit")
    u.add_argument("season")
    k = sub.add_parser("dk")
    k.add_argument("label")
    k.add_argument("triples", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "harness":
        cmd_harness(a.season, a.protocol, a.tag, a.pq_lomo)
    elif a.cmd == "fitbundle":
        cmd_fitbundle(a.season)
    elif a.cmd == "usagefit":
        cmd_usagefit(a.season)
    else:
        cmd_dk(a.label, a.triples)


if __name__ == "__main__":
    main()
