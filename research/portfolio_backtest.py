"""T2.12 acceptance backtest: greedy GPP portfolio vs "top N by individual E[payout]".

Walk-forward on RotoGuru slates (docs/PLAYERS.md §7.3, §8 T2.12). Per slate:

1. our projections and sims from the T2.1 harness path (`backtest.SeasonRun`: bundle fit on
   seasons < S, causal context, `pipeline.project_historical(protocol='avail')`);
2. a field proxy from public information only (`field.hist_inputs`: season-to-date DK average as
   the public projection, RotoGuru starters, availability known), large-GPP archetype, medium
   strength, public projection 0.5 ours + 0.5 season average as the live proxy blends DK FPPG
   with ours (`--field-public-only`: season average alone), for a preset GPP ladder (`field.preset_contest('gpp', M)`, top 0.1 % / 1 % / 20 %
   sharing 86 % of the fees); the field sample is exactly the M - N other entries;
3. `portfolio.build_portfolio(..., baselines=True)` with the calibrated scoring truth
   (`portfolio.TRUTH_BETA` / `TRUTH_TAU`; `--beta` / `--tau` override): the greedy portfolio (default caps: gamma 5,
   max exposure 60 %) and, from the same candidates and sims, the N candidates with the largest
   individual E[payout] (no caps), the same picked under the portfolio's caps, and the greedy
   with no caps (isolates the joint-marginal selection from the caps);
4. each set is scored jointly against the same field with the slate's ACTUAL DK points
   (`portfolio.realised`): the realised proxy payout.

Report: calibration (simulated / realised E[payout], and the same sets scored on the raw sims
= "before"), paired difference greedy - baseline in entry fees per slate (mean ± SE over slates, and
the share of slates won), plus the in-sample (simulated) E[payout] of each. One realised outcome
per slate is very noisy for a top-heavy GPP ladder: read the SE.

    uv run python research/portfolio_backtest.py --seasons 2019-20 2020-21 --dates 50 \\
        --n 20 --field 5000 --sims 4000 --out var/reports/portfolio_backtest.csv

Resumable: rows already in --out are skipped. Single process; CP-SAT uses --threads (default 2).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import backtest as BT
from courtside.proj import pipeline as P
from courtside.proj.dfs import field as F
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import portfolio as PF

METHODS = ("greedy", "top_individual", "top_individual_capped", "greedy_uncapped", "top_mean")
TAU_GRID = (0.0, 0.35, 0.7)            # extra greedy-set rescoring at beta = TRUTH_BETA


def slate_sims(res, rg_of_nba: dict[int, str]) -> dict:
    """Stored-sims-like dict (fp S x n, dk_id = RotoGuru id) from a DayResult."""
    fps, ids = [], []
    S = min(g.fp.shape[0] for g in res.sims)
    for g in res.sims:
        fps.append(np.asarray(g.fp[:S], np.float32))
        ids.extend(rg_of_nba.get(int(p), "") for p in g.player_id)
    return {"fp": np.concatenate(fps, axis=1), "dk_id": np.array(ids)}


def run_date(conn, sr: BT.SeasonRun, day: str, args) -> dict | None:
    t0 = time.time()
    pool = O.hist_pool(conn, day, sr.rules)
    inp = F.hist_inputs(conn, pool)
    if inp["fppg"].notna().sum() < 40:
        return None
    rows = sr.rg[sr.rg["game_date"] == day]
    rg_of = {int(r.player_id): str(r.rg_id) for r in rows.itertuples(index=False)
             if not pd.isna(r.player_id)}
    res = P.project_historical(conn, day, protocol="avail", n_sims=args.sims, ctx=sr.ctx,
                               bundle=sr.bundle)
    pl = res.players.drop_duplicates("player_id").set_index("player_id")
    in_pool = {p.dk_player_id for p in pool.players}
    mu = {rg_of[int(i)]: float(q["dk_mean"]) for i, q in pl.iterrows()
          if int(i) in rg_of and rg_of[int(i)] in in_pool}
    if len(mu) < 3 * sr.rules.n_slots:
        return None
    sims = slate_sims(res, rg_of)
    if not args.field_public_only:   # live-like public projection: 0.5 ours + 0.5 season avg
        inp["proj"] = inp["pid"].map(mu)
    M = int(args.field)
    contest = F.preset_contest("gpp", M, 1.0)
    s = PF.Settings(n=args.n, n_sims=None, n_field=M - args.n, threads=args.threads,
                    seed=P.stable_seed(day, "portfolio-bt"), beta=args.beta, tau=args.tau)
    pf = PF.build_portfolio(pool, inp, sims, contest, s, mu=mu, rules=sr.rules, baselines=True)
    actual = O.hist_projections(conn, pool, "actual")
    act = np.array([actual[p].mean if p in actual else 0.0 for p in pf.field.pids], np.float32)
    out = {"date": day, "season": sr.season, "games": len(res.sims), "pool": len(pool.players),
           "projected": len(mu), "candidates": pf.info["candidates_total"],
           "entries": len(pf.lineups), "cost": float(args.n), "field": M,
           "field_blend": not args.field_public_only, "beta": s.beta, "tau": s.tau,
           "seconds": 0.0}
    sets = {"greedy": pf.info["greedy_pairs"],
            **{k: v["pairs"] for k, v in pf.info["baselines"].items()}}
    insample = {"greedy": pf.summary["e_payout"],
                **{k: v["e_payout"] for k, v in pf.info["baselines"].items()}}
    for k, pairs in sets.items():
        r = PF.realised(pf.field, contest, pairs, act, seed=1)
        out[f"{k}_payout"] = r["payout"]
        out[f"{k}_cashed"] = r["cashed"]
        out[f"{k}_best_rank"] = r["best_rank"]
        out[f"{k}_sim_e"] = insample[k]
        out[f"{k}_mode"] = r["mode"]
    # "before": the same sets scored on our raw sims (truth = our projection, beta = tau = 0)
    for k in ("greedy", "top_individual", "top_mean"):
        if k in sets:
            out[f"{k}_sim_e_raw"] = PF.sim_payout(pf.field, contest, pf.fp_raw, sets[k], seed=1)
    for tau in TAU_GRID:
        fp_t, _ = PF.truth_sims(pf.fp_raw, pf.field.pids, inp, PF.TRUTH_BETA, tau,
                                np.random.default_rng(3))
        out[f"greedy_sim_e_tau{tau:g}"] = PF.sim_payout(pf.field, contest, fp_t, sets["greedy"],
                                                         seed=1)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(df: pd.DataFrame) -> str:
    lines = [f"slates {len(df)} (seasons {', '.join(sorted(df.season.unique()))}), N "
             f"{int(df.entries.median())}, field {int(df.field.median())}, preset GPP ladder; "
             f"payout in entry fees, realised = ACTUAL DK points vs the field proxy"]
    methods = [k for k in METHODS if f"{k}_payout" in df]
    for k in methods:
        lines.append(f"  {k:<24} realised mean {df[f'{k}_payout'].mean():8.2f} "
                     f"(ROI {df[f'{k}_payout'].sum() / df.cost.sum() - 1:+.1%}); simulated "
                     f"E {df[f'{k}_sim_e'].mean():8.2f}; cashed/slate "
                     f"{df[f'{k}_cashed'].mean():.2f}")
    lines.append("  calibration: simulated / realised E[payout] (sum over slates; bootstrap 90% "
                 "interval over slates)")
    rng = np.random.default_rng(0)
    cols = [(f"{k}_sim_e", k) for k in methods] + \
        [(f"{k}_sim_e_raw", k) for k in methods if f"{k}_sim_e_raw" in df] + \
        [(c, "greedy") for c in df.columns if c.startswith("greedy_sim_e_tau")]
    for col, k in cols:
        if col not in df:
            continue
        ok = df[col].notna()
        sim, real = df.loc[ok, col].to_numpy(), df.loc[ok, f"{k}_payout"].to_numpy()
        boot = []
        for _ in range(500):
            i = rng.integers(0, len(sim), len(sim))
            boot.append(sim[i].sum() / max(real[i].sum(), 1e-9))
        lines.append(f"    {col:<26} {sim.sum() / real.sum():6.2f}  "
                     f"[{np.percentile(boot, 5):.2f}, {np.percentile(boot, 95):.2f}]  "
                     f"({ok.sum()} slates)")
    pairs = [("greedy", b) for b in methods[1:] if b != "greedy_uncapped"]
    if "greedy_uncapped" in methods:
        pairs.append(("greedy_uncapped", "top_individual"))
    for a, b in pairs:
        d = df[f"{a}_payout"] - df[f"{b}_payout"]
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        lines.append(f"  {a} - {b:<22} paired {d.mean():+8.2f} ± {se:.2f} (SE) "
                     f"= {d.mean() / se if se else float('nan'):+.2f} SE; {a} ahead on "
                     f"{(d > 0).mean():.0%}, behind on {(d < 0).mean():.0%} of slates; median "
                     f"{d.median():+.2f}")
        ds = df[f"{a}_sim_e"] - df[f"{b}_sim_e"]
        lines.append(f"  {'':<33} in-sample (simulated) {ds.mean():+8.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", default=["2019-20", "2020-21"])
    ap.add_argument("--dates", type=int, default=50, help="dates per season (evenly spaced)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--field", type=int, default=5000)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--field-public-only", action="store_true",
                    help="field public projection = season average only (default: 0.5 ours + "
                         "0.5 season average, as live)")
    ap.add_argument("--beta", type=float, default=None, help="truth shift (default TRUTH_BETA)")
    ap.add_argument("--tau", type=float, default=None, help="model error (default TRUTH_TAU)")
    args = ap.parse_args(argv)
    conn = db.connect(args.db) if args.db else db.connect()
    out = Path(args.out or (Path(os.environ.get("COURTSIDE_DATA", "var")) / "reports"
                            / "portfolio_backtest.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    have = set(done["date"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    settings = BT.Settings(modes=("cash",), n_sims=args.sims)
    for season in args.seasons:
        sr = BT.SeasonRun(conn, season, settings, log=lambda m: print(m, flush=True))
        for day in sr.dates(args.dates):
            if day in have:
                continue
            try:
                r = run_date(conn, sr, day, args)
            except Exception as e:  # noqa: BLE001 - one bad slate must not stop the run
                print(f"{day}: failed {type(e).__name__}: {e}", flush=True)
                continue
            if r is None:
                print(f"{day}: skipped (pool / projections too thin)", flush=True)
                continue
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{day}: greedy {r['greedy_payout']:.1f} top_ind {r['top_individual_payout']:.1f}"
                  f" capped {r['top_individual_capped_payout']:.1f} (sim E {r['greedy_sim_e']:.1f}"
                  f" / {r['top_individual_sim_e']:.1f}) {r['seconds']}s", flush=True)
        del sr
    df = pd.DataFrame(rows)
    if len(df):
        txt = summarize(df)
        print("\n" + txt)
        out.with_suffix(".txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
