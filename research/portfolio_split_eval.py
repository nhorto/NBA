"""Contest builder sim split, ON vs OFF (IMPROVEMENTS.md B1; docs/research/portfolio-split.md).

`portfolio.build_portfolio(split=False | True)` on stored slates:

* **slates**  per (slate, contest): the chosen portfolio's REPORTED E[payout] / P(any cash) /
  P(any top 1 %) / relative score, the share of scenario-type lineups among those selected and the
  runtime, OFF vs ON. With `--holdout H` both builds see only the first S - H stored sim rows and
  both chosen sets are also scored on the last H rows (fresh truth noise, same field, same dupe
  model): an honest estimate neither arm selected on, i.e. how much of each arm's report is
  in-sample optimism and whether ON's picks are worse out of sample.
* **opener**  the 2026-10-20 opener (slate 154575, run c15398ddd87f) against its four contests
  with real uploaded standings (`real_standings_calib.real_contests`): ON / OFF builds, then the
  chosen lineups scored on the ACTUAL DK points vs the real standings (rank, cashed, payout) next
  to what the build reported, and the PIT of the night in each arm's reported distribution.

Read-only on the DB (opened mode=ro). Sim paths in dfs_runs are the Mac's; they are remapped to
$COURTSIDE_DATA/sims. Run on a scratch copy:

    COURTSIDE_DATA=var/tmp/b1 PYTHONPATH=src python research/portfolio_split_eval.py slates \\
        --pairs 154575:196456028,155482:197024714 --n 20 --holdout 5000 --out var/tmp/b1/slates.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from courtside.proj import store as SO
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF

sys.path.insert(0, str(Path(__file__).parent))
import real_standings_calib as RS        # noqa: E402

MAC_VAR = "/Users/nicholashorton/Developer/NBA/var"
SCEN = ("scenario", "min_locked", "refill", "curve_refill")


def connect() -> sqlite3.Connection:
    p = Path(os.environ["COURTSIDE_DATA"]) / "courtside.db"
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def pick_run(conn, slate: str) -> str:
    """The newest ok run with sims at or before the slate's lock (else the newest)."""
    lock = conn.execute("SELECT lock_utc FROM dfs_slates WHERE slate_id=?", (slate,)).fetchone()
    q = ("SELECT run_id FROM dfs_runs WHERE slate_id=? AND ok=1 AND sim_path IS NOT NULL "
         "AND kind!='swap' {} ORDER BY created_at DESC LIMIT 1")
    r = None
    if lock and lock[0]:
        r = conn.execute(q.format("AND created_at<=?"), (slate, lock[0])).fetchone()
    r = r or conn.execute(q.format(""), (slate,)).fetchone()
    if r is None:
        raise SystemExit(f"slate {slate}: no run with sims")
    return r[0]


def load(conn, slate: str, run: str | None) -> dict:
    run = run or pick_run(conn, slate)
    ts = conn.execute("SELECT created_at FROM dfs_runs WHERE run_id=?", (run,)).fetchone()[0]
    pool, inp, info = F.slate_inputs(conn, slate, run, as_of=ts)
    path = str(info["sim_path"]).replace(MAC_VAR, os.environ["COURTSIDE_DATA"])
    return {"slate": slate, "run": run, "pool": pool, "inputs": inp, "sims": SO.load_sims(path)}


THREADS = 1
ON_MODE = "split"                      # the ON arm's portfolio split mode ("split" | "gs")


def mode_of(arm: str) -> str:
    return ON_MODE if arm == "on" else "off"


def settings(n: int, seed: int, n_sims: int | None) -> PF.Settings:
    return PF.Settings(n=n, seed=seed, n_sims=n_sims, threads=THREADS)


def honest(pf: PF.Portfolio, ctx: dict, hold: np.ndarray, contest: F.Contest, seed: int) -> dict:
    """The chosen set scored on held-out stored rows (fresh truth noise), with the build's dupe
    model (expected dupes the field sample lacks as a Poisson count per sim)."""
    fld, n = pf.field, len(pf.lineups)
    sims = {**ctx["sims"], "fp": ctx["sims"]["fp"][hold]}
    fp, _ = F.align_sims(sims, fld.pids)
    fp, _ = PF.truth_sims(fp, fld.pids, ctx["inputs"], pf.settings.beta, pf.settings.tau,
                          np.random.default_rng(seed + 99))
    pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in pf.slots()]
    ev = F.evaluate(fld, fp, contest, pairs, seed=seed, tiers=False, own_entries=n, per_sim=True)
    m_o = max(1, contest.field_size - n)
    tied = np.rint(ev.per_sim["tied"]).astype(np.int64)
    extra = np.array([max(0.0, float(lu["dupes"]) * m_o / max(1, contest.field_size - 1) -
                          _in_sample(fld, p, m_o)) for lu, p in zip(pf.lineups, pairs)])
    rng = np.random.default_rng(seed + 98)
    if (extra > 1e-4).any():
        tied += rng.poisson(extra[None, :], size=tied.shape)
    lad = PF.Ladder(contest)
    pay, rank = PF.joint_payouts(lad, ev.per_sim["score"],
                                 np.rint(ev.per_sim["above"]).astype(np.int64), tied)
    sm = PF._summary(lad, pay, rank, contest)
    return {**{k: sm[k] for k in ("e_payout", "e_payout_se", "p_any_cash", "p_any_top1",
                                  "n_sims")}, "_tot": pay.sum(1)}


def _in_sample(fld: F.Field, pairs, m_o: int) -> float:
    _, rows, _ = F._cand_matrix(fld, [pairs], fld.slots)
    row = rows[0]
    if not (row >= 0).all():
        return 0.0
    cpt = fld.slots[0] == "CPT"
    cnt = {tuple(int(x) for x in r): int(c) for r, c in zip(F._canon(fld.lineups, cpt), fld.counts)}
    return cnt.get(tuple(int(x) for x in F._key(row, cpt)), 0) * m_o / max(1, fld.n)


def row_of(pf: PF.Portfolio, secs: float) -> dict:
    sm = pf.summary
    src = pf.sources or []
    return {"version": pf.info["version"], "n": len(pf.lineups), "seconds": round(secs, 1),
            "e_payout": sm["e_payout"], "e_payout_se": sm["e_payout_se"], "roi": sm["roi"],
            "p_any_cash": sm["p_any_cash"], "p_any_top1": sm["p_any_top1"],
            "relative": sm.get("relative"), "n_sims": sm["n_sims"],
            "p_cash_mean": float(np.mean([x["p_cash"] for x in pf.lineups])),
            "p_top1_mean": float(np.mean([x["p_top1"] for x in pf.lineups])),
            "scenario_share": float(np.mean([s in SCEN for s in src])) if src else None,
            "sources": PF._count(src), "split": pf.info.get("split"),
            "candidates": pf.info["candidates_total"], "refills": pf.info["refills"],
            "timing": pf.info["timing"]}


def cmd_slates(args, conn) -> dict:
    out = []
    for item in args.pairs.split(","):
        slate, cid, *rest = item.split(":")
        ctx = load(conn, slate, rest[0] if rest else None)
        contest = F.load_contest(conn, cid)
        S = ctx["sims"]["fp"].shape[0]
        sims, hold = ctx["sims"], None
        if args.holdout:
            hold = np.arange(S - args.holdout, S)
            sims = {**sims, "fp": sims["fp"][:S - args.holdout]}
        n = min(args.n, contest.field_size - 1)
        res = {"slate": slate, "contest": cid, "name": contest.name, "M": contest.field_size,
               "fee": contest.entry_fee, "run": ctx["run"], "sims_build": sims["fp"].shape[0],
               "holdout": args.holdout}
        tots = {}
        for arm in ("off", "on"):
            for seed in range(args.seed0, args.seed0 + args.seeds):
                t = time.perf_counter()
                pf = PF.build_portfolio(ctx["pool"], ctx["inputs"], sims, contest,
                                        settings(n, seed, None), split=mode_of(arm))
                r = row_of(pf, time.perf_counter() - t)
                r["seed"] = seed
                if hold is not None:
                    r["holdout"] = honest(pf, ctx, hold, contest, seed)
                    tots[arm, seed] = r["holdout"].pop("_tot")
                res.setdefault(arm, []).append(r)
                print(json.dumps({"slate": slate, "contest": cid, "arm": arm, "seed": seed,
                                  **{k: r[k] for k in ("e_payout", "p_any_cash", "p_any_top1",
                                                       "scenario_share", "seconds")},
                                  "holdout": r.get("holdout")}), flush=True)
        if hold is not None:            # paired (same held-out rows and truth noise per seed)
            res["holdout_on_minus_off"] = []
            for seed in range(args.seed0, args.seed0 + args.seeds):
                d = tots["on", seed] - tots["off", seed]
                res["holdout_on_minus_off"].append({"seed": seed, "mean": float(d.mean()),
                                                    "se": float(d.std() / math.sqrt(len(d)))})
            print(json.dumps({"slate": slate, "paired": res["holdout_on_minus_off"]}), flush=True)
        out.append(res)
    return {"pairs": out}


def cmd_opener(args, conn) -> dict:
    ctx = load(conn, "154575", args.run)
    act_pts = RS.actual_points(conn, "154575", ctx["run"], ctx["pool"])
    out = []
    for rc in RS.real_contests(conn, "154575"):
        c = rc["contest"]
        single = "single entry" in (c.name or "").lower()
        n = 1 if single else min(args.n, c.field_size - 1)
        pts = np.sort(rc["points"])[::-1]
        line = float(pts[c.paid - 1])
        top1 = float(pts[max(1, int(math.ceil(0.01 * c.field_size))) - 1])
        lad_real = c.ladder()
        res = {"contest": c.contest_id, "name": c.name, "M": c.field_size, "n": n,
               "real_cash_line": line, "real_top1_line": top1, "source": rc["source"]}
        for arm in ("off", "on"):
            arms = []
            for seed in range(args.seeds):
                t = time.perf_counter()
                pf = PF.build_portfolio(ctx["pool"], ctx["inputs"], ctx["sims"], c,
                                        settings(n, seed, None), split=mode_of(arm))
                r = row_of(pf, time.perf_counter() - t)
                pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in pf.slots()]
                act = np.array([sum(act_pts.get(p, 0.0) for _, p in lu) for lu in pairs])
                # our k entries among the real M: entry j's rank = real entries above it + our
                # own entries above it + 1 (the M real entries stand in for the M - k others)
                above = np.array([int((pts > a + 1e-9).sum()) for a in act])
                rank = PF.joint_ranks(np.rint(act * 64).astype(np.int64)[None, :],
                                      above[None, :])[0]
                pay = np.array([lad_real[r] if r <= c.field_size else 0.0 for r in rank])
                # each arm's reported distribution (R rows ON, all rows OFF) of lineup - cash line
                fp = pf.fp
                if arm == "on":
                    fp = fp[PF.split_rows(fp.shape[0], seed, ON_MODE)["R"]]
                ev = F.evaluate(pf.field, fp, c, pairs, seed=seed, tiers=False, own_entries=n,
                                per_sim=True)
                sc = (ev.per_sim["score"].astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
                lad = PF.Ladder(c)
                spay, _ = PF.joint_payouts(lad, ev.per_sim["score"],
                                           np.rint(ev.per_sim["above"]).astype(np.int64),
                                           np.rint(ev.per_sim["tied"]).astype(np.int64))
                stot, rtot = spay.sum(1), float(pay.sum())
                pit = ((sc - ev.cash_line[:, None]) < (act - line)[None, :]).mean(0)
                cashed = act >= line
                r.update({
                    "pred_p_cash_mean": r["p_cash_mean"], "real_cashed": float(cashed.mean()),
                    "pred_p_any_cash": r["p_any_cash"], "real_any_cash": bool(cashed.any()),
                    "pred_p_top1_mean": r["p_top1_mean"], "real_top1": float((act >= top1).mean()),
                    "pred_e_payout": r["e_payout"], "real_payout": rtot,
                    "payout_pct": float((stot < rtot).mean() + 0.5 * (stot == rtot).mean()),
                    "pit_median": float(np.median(pit)), "pit_mean": float(pit.mean()),
                    "actual_mean": float(act.mean()), "best_real_rank": int(rank.min()),
                    "sim_mean": float(sc.mean())})
                arms.append(r)
                print(json.dumps({"contest": c.contest_id, "arm": arm, "seed": seed,
                                  **{k: r[k] for k in ("pred_p_cash_mean", "real_cashed",
                                                       "pred_e_payout", "real_payout",
                                                       "payout_pct", "pit_median",
                                                       "scenario_share", "seconds")}}),
                      flush=True)
            res[arm] = arms
        out.append(res)
    return {"run": ctx["run"], "contests": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("slates")
    a.add_argument("--pairs", required=True, help="slate:contest[:run],...")
    a.add_argument("--holdout", type=int, default=0)
    a.add_argument("--seed0", type=int, default=0)
    a.add_argument("--split-r", type=float, default=None, help="override portfolio.SPLIT_R")
    b = sub.add_parser("opener")
    b.add_argument("--run", default="c15398ddd87f")
    for p in (a, b):
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--seeds", type=int, default=1)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--on-mode", default="split", choices=("split", "gs"))
    args = ap.parse_args(argv)
    global THREADS, ON_MODE
    THREADS, ON_MODE = args.threads, args.on_mode
    if getattr(args, "split_r", None) is not None:
        PF.SPLIT_R = args.split_r
    conn = connect()
    res = (cmd_slates if args.cmd == "slates" else cmd_opener)(args, conn)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
