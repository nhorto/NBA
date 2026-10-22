"""+mf+mr with the minutes-ratio table refit on the minutes-fix engine's plans: evaluation.

The minutes-fix agent found +mf+mr (design B) the strongest arm, but its table had been fit on
the OFF minutes engine's training plans while the sims read P from the +mf engine. Since
`minutes_ratio.plans_tag`, a +mf+mr bundle is fit on the +mf engine's walk-forward plans
(`planned_<X>_mf.parquet`, season X's `fit_v1(minfix=True)` params fit on the 5 seasons < X).
This driver runs the arms and the paired tables (docs/PLAYERS.md, minutes-fix results):

  plans S [S ..]          the +mf engine's planned minutes of those seasons (cached; fits the
                          season's +mf minutes params when no `minutes_params_<S>_v1_mf.json`)
  fitbundle S             pipeline.load_bundle(S) for the arm of the environment switches
  harness S PROTOCOL      T2.1 harness (avail) / T2.4 slice (live, live10; P(play) cells leave-
                          one-month-out), 4,000 sims, with the minutes columns of
                          research/minutes_fix_eval.py; RotoGuru seasons also build the cash
                          lineups -> OUT/rows_<protocol>_<S>_<TAG>.parquet (+ lineups_...)
                          TAG = $MFMR_TAG or the MODEL_VERSION
  report [--sets ..]      paired tables per set: +mf+mr(refit) vs OFF, vs +mf, and +mf vs OFF
                          (and the old table, tag `mfmr_old`, when its rows exist); tiers and
                          demotion flags by the OFF arm; SE clustered by date; cash lineups
                          (actual DK paired by date, ratio to the hindsight lineup)
                          -> OUT/mfmr_report.json + markdown on stdout

Rows file tags (env MFMR_TAG at harness time): off, mf, mfmr, mfmr_old.
Population (as the earlier evals): projected rows with E[min] >= 10 in either arm of the pair,
DNP = 0. Tiers by the OFF arm's E[min | plays]: star >= 30, starter 24-30, rotation 15-24,
bench < 15. Environment: COURTSIDE_DATA, MFMR_OUT (default var/tmp/mfmr/out). Heavy: Mac scratch
(`courtside-compute/mfmr`).
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
sys.path.insert(0, str(ROOT / "research"))
OUT = Path(os.environ.get("MFMR_OUT", ROOT / "var" / "tmp" / "mfmr" / "out"))
os.environ.setdefault("MF_OUT", str(OUT))

import minutes_fix_eval as MF  # noqa: E402

SETS = (("2023-24", "avail"), ("2024-25", "avail"), ("2025-26", "live"),
        ("2025-26", "live10"), ("2019-20", "avail"), ("2020-21", "avail"))
TIERS = tuple(t[0] for t in MF.TIERS)


def _log(m):
    print(m, flush=True)


def cmd_plans(*seasons) -> None:
    from courtside import db
    from courtside.proj import minutes as M
    from courtside.proj import minutes_ratio as MR
    M.GBM_PARAMS["num_threads"] = 1
    conn = db.connect()
    for s in seasons:
        t = time.time()
        d = MR.planned_rows(conn, [s], log=_log, plans=MR.PLANS_MF)
        _log(f"[mfmr] plans {s} (mf engine): {len(d)} rows, {time.time() - t:.0f}s")


def cmd_fitbundle(season: str) -> None:
    from courtside import db
    from courtside.proj import minutes as M
    from courtside.proj import minutes_ratio as MR
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    conn = db.connect()
    t = time.time()
    b = P.load_bundle(conn, season, None, log=_log)
    t_mr = b.state.get("minutes_adj_mr") or {}
    _log(f"[mfmr] bundle {season} {P.MODEL_VERSION} ({P._VS}) in {time.time() - t:.0f}s; "
         f"minfix {bool(b.minutes.minfix)}, table plans {t_mr.get('plans', '') or 'OFF'}, "
         f"expected {MR.plans_tag(b.minutes) or 'OFF'}")


def _tag() -> str:
    from courtside.proj import pipeline as P
    return os.environ.get("MFMR_TAG") or P.MODEL_VERSION


def cmd_harness(season: str, protocol: str) -> None:
    from courtside import db
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    MF._patch_capture()
    conn = db.connect()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    rg = B.is_rotoguru(season) and not live
    st = B.Settings(protocol=protocol, n_sims=4000, modes=("cash",) if rg else (),
                    baselines=B.BASELINES, pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, ln, meta = B.run_season(conn, season, st, log=_log, date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = _tag()
    f = OUT / f"rows_{protocol}_{season}_{tag}.parquet"
    pl.to_parquet(f)
    if ln is not None and len(ln):
        ln.to_parquet(OUT / f"lineups_{protocol}_{season}_{tag}.parquet")
    _log(f"[mfmr] harness {season} {protocol} {P.MODEL_VERSION} (tag {tag}): {len(pl)} rows, "
         f"lineups {0 if ln is None else len(ln)}, skipped {len(meta['skipped'])}, "
         f"tip_unknown {meta.get('tip_unknown', 'n/a')}, {time.time() - t0:.0f}s -> {f}")


# ============================================================================== report

def _rows(protocol: str, season: str, tag: str) -> pd.DataFrame | None:
    f = OUT / f"rows_{protocol}_{season}_{tag}.parquet"
    if not f.exists():
        return None
    a = pd.read_parquet(f)
    return a.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"]) \
        .reset_index(drop=True)


def _off_keys(off: pd.DataFrame, l10: np.ndarray) -> pd.DataFrame:
    """Tier and demotion flag of every (date, player) by the OFF arm."""
    k = off[["date", "player_id"]].copy()
    min_c = off["exp_min"] / off["p_play"].clip(lower=0.05)
    k["tier_off"] = "bench"
    for name, lo, hi in MF.TIERS:
        k.loc[(min_c >= lo) & (min_c < hi), "tier_off"] = name
    v = min_c / pd.Series(l10, index=off.index).where(pd.Series(l10, index=off.index) >= 1)
    k["dem_off"] = np.where(v < 0.9, "demoted", np.where(v > 1.1, "promoted", "steady"))
    return k


def compare(base: pd.DataFrame, other: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """Paired other - base (`minutes_fix_eval.dk_metrics`), overall, by tier (OFF), and the
    bench / rotation demoted / promoted rows (OFF)."""
    d = MF.dk_pair(base, other)
    d = d.merge(keys, on=["date", "player_id"], how="left")
    d["tier4"], d["dem"] = d["tier_off"].fillna(d["tier4"]), d["dem_off"].fillna("steady")
    q = d[d["pop"]]
    res = {"all": MF.dk_metrics(q)}
    for t in TIERS:
        res[t] = MF.dk_metrics(q[q["tier4"] == t])
    for t in ("bench", "rotation"):
        for dm in ("demoted", "promoted"):
            x = q[(q["tier4"] == t) & (q["dem"] == dm)]
            if len(x) > 20:
                res[f"{t} {dm}"] = MF.dk_metrics(x)
    return res


def lineup_stats(protocol: str, season: str, tags: list) -> dict:
    """Cash lineups of each arm (RotoGuru seasons): mean actual DK, ratio to the hindsight
    lineup, and paired by date vs each other arm."""
    from courtside.proj import backtest as B
    lns = {}
    for t in tags:
        f = OUT / f"lineups_{protocol}_{season}_{t}.parquet"
        if f.exists():
            lns[t] = pd.read_parquet(f)
    if not lns:
        return {}
    out: dict = {"arms": {}, "paired": {}}
    cash = {}
    for t, ld in lns.items():
        rep = B.lineup_report(ld)
        m = rep.get("methods", {})
        if "ours_cash" in m:
            out["arms"][t] = {"cash_actual": m["ours_cash"]["mean_actual"],
                              "cash_se": m["ours_cash"]["se"],
                              "cash_ratio_hindsight": m["ours_cash"]["ratio_to_hindsight"],
                              "mean_ratio_hindsight": m.get("ours_mean", {}).get(
                                  "ratio_to_hindsight", float("nan")),
                              "hindsight": m.get("hindsight", {}).get("mean_actual"),
                              "n_dates": rep.get("n_dates")}
            c = ld[(ld["method"] == "ours_cash") & (ld["rank"] == 1)]
            cash[t] = c.set_index("date")["actual"]
    for a in cash:
        for b in cash:
            if a == b:
                continue
            j = pd.concat([cash[a], cash[b]], axis=1, keys=["a", "b"]).dropna()
            if len(j) < 5:
                continue
            dd = (j["b"] - j["a"]).to_numpy(float)
            out["paired"][f"{b}-{a}"] = {"n": int(len(j)), "diff": float(dd.mean()),
                                         "se": float(dd.std(ddof=1) / np.sqrt(len(dd)))}
    return out


def _cell(r: dict, k: str, fmt: str = "{off:.3f} / {diff:+.4f} ({z:+.1f})") -> str:
    return fmt.format(**r[k]) if k in r else ""


def _table(res: dict, sets: list, cmp: str) -> str:
    cols = [f"{s} {p}" if p != "avail" else s for s, p in sets]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def row(name, f):
        vals = []
        for s, p in sets:
            r = res.get(f"{p}|{s}", {}).get(cmp)
            vals.append(f(r) if r else "")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    row("rows", lambda r: str(r["all"]["n"]))
    row("DK MAE base / Δ (z)", lambda r: _cell(r["all"], "dk_mae"))
    row("DK CRPS base / Δ (z)", lambda r: _cell(r["all"], "dk_crps"))
    row("80% cov base → new", lambda r: f"{r['all']['dk_cov80']['off']:.3f} → "
        f"{r['all']['dk_cov80']['on']:.3f}")
    row("DK PIT max dev", lambda r: f"{r['all']['dk_pitdev_off']:.4f} → "
        f"{r['all']['dk_pitdev_on']:.4f}")
    row("CRPS z star/starter/rotation/bench", lambda r: " / ".join(
        f"{r[t]['dk_crps']['z']:+.1f}" for t in TIERS))
    row("DK bias star/starter/rotation/bench", lambda r: " / ".join(
        f"{r[t]['dk_bias']['off']:+.2f}→{r[t]['dk_bias']['on']:+.2f}" for t in TIERS))
    row("bench demoted (n) DK CRPS Δ (z)", lambda r: (
        f"({r['bench demoted']['n']}) {r['bench demoted']['dk_crps']['diff']:+.3f} "
        f"({r['bench demoted']['dk_crps']['z']:+.1f})") if "bench demoted" in r else "")
    row("rotation promoted (n) DK CRPS Δ (z)", lambda r: (
        f"({r['rotation promoted']['n']}) {r['rotation promoted']['dk_crps']['diff']:+.3f} "
        f"({r['rotation promoted']['dk_crps']['z']:+.1f})") if "rotation promoted" in r else "")
    return "\n".join(lines)


def cmd_report(sets: list) -> None:
    conn = MF._conn()
    res: dict = {}
    for season, protocol in sets:
        arms = {t: _rows(protocol, season, t) for t in ("off", "mf", "mfmr", "mfmr_old")}
        if arms["off"] is None:
            _log(f"[mfmr] {season} {protocol}: no OFF rows, skipped")
            continue
        keys = _off_keys(arms["off"], MF._last10(conn, arms["off"]))
        r: dict = {}
        for name, a, b in (("mfmr-off", "off", "mfmr"), ("mfmr-mf", "mf", "mfmr"),
                           ("mf-off", "off", "mf"), ("old-off", "off", "mfmr_old"),
                           ("mfmr-old", "mfmr_old", "mfmr")):
            if arms[a] is not None and arms[b] is not None:
                r[name] = compare(arms[a], arms[b], keys)
        r["lineups"] = lineup_stats(protocol, season, ["off", "mf", "mfmr", "mfmr_old"])
        res[f"{protocol}|{season}"] = r
        _log(f"[mfmr] {season} {protocol}: " + ", ".join(
            f"{k} CRPS {v['all']['dk_crps']['diff']:+.4f} ({v['all']['dk_crps']['z']:+.1f})"
            for k, v in r.items() if k != "lineups"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mfmr_report.json").write_text(json.dumps(res, indent=1, default=float))
    for cmp, title in (("mfmr-off", "+mf+mr (refit) vs OFF"), ("mfmr-mf", "+mf+mr (refit) vs +mf"),
                       ("mf-off", "+mf vs OFF"), ("old-off", "+mf+mr (old table) vs OFF"),
                       ("mfmr-old", "+mf+mr refit vs old table")):
        if any(cmp in v for v in res.values()):
            print(f"\n**{title}**\n")
            print(_table(res, sets, cmp))
    print("\n**Cash lineups (RotoGuru): actual DK, ratio to hindsight; paired by date**\n")
    for k, v in res.items():
        lu = v.get("lineups") or {}
        if not lu.get("arms"):
            continue
        arms = ", ".join(f"{t} {a['cash_actual']:.1f} ({a['cash_ratio_hindsight']:.3f})"
                         for t, a in lu["arms"].items())
        pr = ", ".join(f"{p} {x['diff']:+.1f} ± {x['se']:.1f}" for p, x in lu["paired"].items()
                       if p in ("mfmr-off", "mfmr-mf", "mf-off", "mfmr-mfmr_old"))
        print(f"- {k}: {arms}; {pr}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plans")
    a.add_argument("seasons", nargs="+")
    b = sub.add_parser("fitbundle")
    b.add_argument("season")
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    r = sub.add_parser("report")
    r.add_argument("--sets", nargs="*", help="season:protocol ... (default: the six sets)")
    x = ap.parse_args(argv)
    if x.cmd == "plans":
        cmd_plans(*x.seasons)
    elif x.cmd == "fitbundle":
        cmd_fitbundle(x.season)
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    else:
        sets = [tuple(s.split(":")) for s in x.sets] if x.sets else list(SETS)
        cmd_report(sets)


if __name__ == "__main__":
    main()
