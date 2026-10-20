"""T2.13 check: is the simulated P(cash) of our cash lineup calibrated against the proxy cash line?

Walk-forward on RotoGuru slates, the same projection path as `research/portfolio_backtest.py`
(`backtest.SeasonRun`, `pipeline.project_historical(protocol='avail')`). Per slate:

* our cash lineup (runner objective: mu - 0.3 (mu - p25)) and the Top-1 by mean;
* a preset double-up (`field.preset_contest('double_up', M)`, 44.4 % paid) and a cash-archetype
  field proxy from public information, live-like (public projection = 1/2 ours + 1/2 season
  average), at the medium information strength and at the sharp one (75 % ours);
* simulated P(cash) on our raw sims ("before") and on the calibrated truth
  (`portfolio.truth_sims`, "after"), vs the realised rate: the slate's ACTUAL points scored
  against the same field, opponents redrawn `--draws` times (the field-sampling part).

Acceptance (PLAYERS.md T2.13): mean simulated P(cash) within +/- 5 pp of the realised rate.

    uv run python research/cash_calibration.py --seasons 2019-20 2020-21 --dates 56 \\
        --db var/tmp/t212/courtside.db --out var/reports/cash_calibration.csv
"""

from __future__ import annotations

import argparse
import math
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

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_backtest import slate_sims  # noqa: E402

STRENGTH_W = {"medium": 0.5, "sharp": 0.75}


def run_date(conn, sr, day: str, args) -> dict | None:
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
    proj = {rg_of[int(i)]: O.Projection(float(q["dk_mean"]), p25=float(q["dk_q25"]))
            for i, q in pl.iterrows() if int(i) in rg_of and rg_of[int(i)] in in_pool}
    if len(proj) < 3 * sr.rules.n_slots:
        return None
    inp["proj"] = inp["pid"].map({k: v.mean for k, v in proj.items()})
    sims = slate_sims(res, rg_of)
    cash = O.optimize(pool, proj, sr.rules, mode="cash", lam=0.3, time_limit_s=20)
    top = O.optimize(pool, proj, sr.rules, mode="mean", time_limit_s=20)
    lus = [lu.slot_pairs() for lu in (cash, top) if lu is not None]
    if len(lus) < 2:
        return None
    contest = F.preset_contest("double_up", args.field, 5.0)
    actual = O.hist_projections(conn, pool, "actual")
    out = {"date": day, "season": sr.season, "games": len(res.sims), "field": args.field}
    seed = P.stable_seed(day, "cash-cal")
    params, ver = F.load_params()
    for sname, w in STRENGTH_W.items():
        op = F.ownership_proxy(inp, pool, sr.rules, archetype="cash",
                               params={**params, "w_ours": w}, version=ver)
        f = F.build_field(pool, inp, contest, rules=sr.rules, ownership=op, seed=seed)
        fp_raw, _ = F.align_sims(sims, f.pids)
        fp_t, _ = PF.truth_sims(fp_raw, f.pids, inp, PF.TRUTH_BETA, PF.TRUTH_TAU,
                                np.random.default_rng(seed + 7))
        act = np.array([actual[p].mean if p in actual else 0.0 for p in f.pids], np.float32)
        ev_raw = F.evaluate(f, fp_raw, contest, lus, seed=seed, tiers=False)
        ev_t = F.evaluate(f, fp_t, contest, lus, seed=seed, tiers=False)
        ev_a = F.evaluate(f, np.repeat(act[None, :], args.draws, 0), contest, lus, seed=seed,
                          tiers=False)
        for k, name in enumerate(("cash", "top")):
            out[f"{sname}_{name}_sim_raw"] = ev_raw.lineups[k]["p_cash"]
            out[f"{sname}_{name}_sim"] = ev_t.lineups[k]["p_cash"]
            out[f"{sname}_{name}_real"] = ev_a.lineups[k]["p_cash"]
        out[f"{sname}_cash_line_real"] = float(np.median(ev_a.cash_line))
        out[f"{sname}_cash_line_sim_p50"] = float(np.median(ev_t.cash_line))
    out["cash_actual"] = float(sum(actual[p].mean for _, p in lus[0] if p in actual))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(df: pd.DataFrame) -> str:
    lines = [f"slates {len(df)} ({', '.join(sorted(df.season.unique()))}); preset double-up "
             f"{int(df.field.median())} entries, 44.4 % paid; realised = actual points vs the "
             f"proxy field (opponents redrawn); T2.13 bar: |sim - realised| <= 5 pp"]
    for s in STRENGTH_W:
        for k in ("cash", "top"):
            real = df[f"{s}_{k}_real"]
            for tag, col in (("before", f"{s}_{k}_sim_raw"), ("after", f"{s}_{k}_sim")):
                d = df[col] - real
                se = d.std(ddof=1) / math.sqrt(len(d))
                lines.append(f"  {s:<6} {k:<4} {tag:<6} sim {df[col].mean():.3f}  realised "
                             f"{real.mean():.3f}  diff {d.mean() * 100:+5.1f} ± {se * 100:.1f} pp")
    # reliability of the calibrated medium cash numbers
    b = pd.cut(df["medium_cash_sim"], [0, .5, .6, .7, .8, .9, 1.0])
    g = df.groupby(b, observed=True).agg(n=("medium_cash_sim", "size"),
                                         sim=("medium_cash_sim", "mean"),
                                         real=("medium_cash_real", "mean"))
    lines.append("  medium cash, calibrated, by simulated P(cash): " + "; ".join(
        f"{i}: n {r.n}, sim {r.sim:.2f}, real {r.real:.2f}" for i, r in g.iterrows()))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", default=["2019-20", "2020-21"])
    ap.add_argument("--dates", type=int, default=56)
    ap.add_argument("--field", type=int, default=500)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=200, help="opponent redraws of the actual row")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="var/reports/cash_calibration.csv")
    args = ap.parse_args(argv)
    conn = db.connect(args.db) if args.db else db.connect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    have = set(done["date"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    for season in args.seasons:
        sr = BT.SeasonRun(conn, season, BT.Settings(modes=("cash",), n_sims=args.sims),
                          log=lambda m: print(m, flush=True))
        for day in sr.dates(args.dates):
            if day in have:
                continue
            try:
                r = run_date(conn, sr, day, args)
            except Exception as e:  # noqa: BLE001
                print(f"{day}: failed {type(e).__name__}: {e}", flush=True)
                continue
            if r is None:
                print(f"{day}: skipped", flush=True)
                continue
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{day}: medium cash sim {r['medium_cash_sim_raw']:.2f}->{r['medium_cash_sim']:.2f}"
                  f" real {r['medium_cash_real']:.2f} | sharp sim {r['sharp_cash_sim']:.2f} real "
                  f"{r['sharp_cash_real']:.2f} ({r['seconds']}s)", flush=True)
        del sr
    df = pd.DataFrame(rows)
    if len(df):
        txt = summarize(df)
        print("\n" + txt)
        out.with_suffix(".txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
