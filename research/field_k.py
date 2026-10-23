"""Ownership uncertainty in the field (IMPROVEMENTS.md D5; docs/research/field-k.md).

Commands (read-only on the DB it is pointed at; run on a scratch copy):

* **fit**   The log-odds sd of the ownership noise (`field.OWN_SD`) that reproduces the proxy's
  measured errors on the 2026-10-20 uploads (slate 154575, lock-45 run c15398ddd87f). For each
  archetype the proxy is built from the run's inputs; for each sd on a grid, R noisy draws
  (`field.perturb_ownership`, i.e. log-odds noise then Sinkhorn back to the slot totals and the
  row cap) are scored against the un-noised proxy with the same metric as the real errors
  (`real_standings_calib.own_metrics`: players above 0.1 % in either; top 10 = by the "real",
  here the draw). sd is interpolated where the mean metric equals the real one: large GPP MAE
  (vs the 29.7k 4PP) and cash top-10 MAE (mean over the three double-ups). Also reported: the
  direct log-odds residual sd of real vs proxy, the other metric each sd implies (cross-check),
  and how much of the noise the sampled fields realise.
* **pit**   Our 50 lineups of the slate vs each real contest, OFF (one field) vs ON (K fields):
  P(cash), P(top 1 %), and the night's PIT of (lineup - cash line) inside the simulated
  distribution (as `real_standings_calib.py chances`), plus the across-field band.
* **bench** Contest-builder runtime OFF vs ON (`portfolio.build_for_slate`, the web's settings:
  strengths on, curve, default sims and split), N = 20 / 150 on a real slate. No save.

    COURTSIDE_DATA=<scratch var> PYTHONPATH=src:research python research/field_k.py fit --out X
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from courtside import db                                    # noqa: E402
from courtside.proj.dfs import field as F                   # noqa: E402
from courtside.proj.dfs import late_news_tag as LNT         # noqa: E402

import real_standings_calib as RS                           # noqa: E402

GRID = tuple(round(x, 2) for x in np.arange(0.10, 3.01, 0.05))


def _logit(x):
    x = np.clip(np.asarray(x, float), 1e-3, 1 - 1e-3)
    return np.log(x / (1 - x))


def draw_metrics(op: F.OwnershipProxy, sd: float, R: int, seed: int,
                 unc: dict | None = None) -> dict:
    """Mean own_metrics(proxy, draw) over R draws ("real" = the draw)."""
    rng = np.random.default_rng(seed)
    sdv = F.own_sd_for(op, unc, sd)
    base = op.as_dict()
    ms = []
    for _ in range(R):
        q = F.perturb_ownership(op, sdv, rng)
        ms.append(RS.own_metrics(base, q.as_dict()))
    return {k: float(np.mean([m[k] for m in ms])) for k in ("mae_pp", "top_mae_pp",
                                                             "top_bias_pp", "spearman")}


def _interp(xs, ys, target):
    xs, ys = np.asarray(xs), np.asarray(ys)
    i = int(np.searchsorted(ys, target))       # ys increasing in sd
    if i <= 0:
        return float(xs[0])
    if i >= len(xs):
        return float(xs[-1])
    t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(xs[i - 1] + t * (xs[i] - xs[i - 1]))


def cmd_fit(args, conn) -> dict:
    ctx = RS.setup(conn, args.slate, args.run)
    rcs = RS.real_contests(conn, args.slate)
    out = {"slate": args.slate, "run": args.run, "R": args.draws, "real": [], "fit": {}}
    ops = {a: F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=a)
           for a in ("cash", "large_gpp")}
    tgt = {"cash": [], "large_gpp": []}
    for rc in rcs:
        c = rc["contest"]
        arch = "cash" if c.archetype == "cash" else "large_gpp"
        op = ops[arch]
        m = RS.own_metrics(op.as_dict(), rc["own"])
        keys = sorted(set(op.as_dict()) | set(rc["own"]))
        p = np.array([op.as_dict().get(k, 0.0) for k in keys])
        y = np.array([rc["own"].get(k, 0.0) for k in keys])
        keep = (p > 0.001) | (y > 0.001)
        resid = _logit(y[keep]) - _logit(p[keep])
        row = {"contest": c.contest_id, "name": c.name, "M": c.field_size, "archetype": arch,
               **{k: m[k] for k in ("mae_pp", "top_mae_pp", "top_bias_pp", "spearman", "n")},
               "logit_resid_sd": float(np.std(resid)),
               "logit_resid_mad_sd": float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
               "logit_resid_sd_both_gt1pct": float(np.std((_logit(y) - _logit(p))[
                   (p > 0.01) & (y > 0.01)]))}
        out["real"].append(row)
        tgt[arch].append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
    # the D5 targets: large GPP on the overall MAE (4.6 pp), cash on the double-ups' top-10 MAE
    # (~27 pp); every metric's implied sd is reported as a cross-check
    for arch in ("large_gpp", "cash"):
        curve = []
        for sd in GRID:
            dm = draw_metrics(ops[arch], sd, args.draws, args.seed)
            curve.append({"sd": sd, **dm})
        real = {m: float(np.mean([r[m] for r in tgt[arch]])) for m in
                ("mae_pp", "top_mae_pp", "spearman", "logit_resid_sd")}
        implied = {}
        for m in ("mae_pp", "top_mae_pp"):
            ys = [c[m] for c in curve]
            implied[m] = (_interp(GRID, ys, real[m]) if max(ys) >= real[m]
                          else f">{GRID[-1]} (max {max(ys):.1f})")
        rho = [-c["spearman"] for c in curve]
        implied["spearman"] = _interp(GRID, rho, -real["spearman"])
        metric = "top_mae_pp" if arch == "cash" else "mae_pp"
        sd_fit = implied[metric] if not isinstance(implied[metric], str) else implied["mae_pp"]
        at = draw_metrics(ops[arch], sd_fit, args.draws * 2, args.seed + 1)
        out["fit"][arch] = {"metric": metric, "real": real, "sd": sd_fit,
                            "implied_sd": implied, "at_fit": at, "curve": curve}
        print(json.dumps({"arch": arch, "real": {k: round(v, 3) for k, v in real.items()},
                          "sd": round(sd_fit, 3), "implied_sd": implied,
                          "at_fit": {k: round(v, 2) for k, v in at.items()}}, default=str),
              flush=True)
    # what a sampled field realises of the noise (fields at the fitted sd vs the base field)
    if args.fields:
        for arch, contest in (("cash", F.preset_contest("double_up", 1149, 2.0)),
                              ("large_gpp", F.preset_contest("gpp", 29726, 5.0))):
            sd = out["fit"][arch]["sd"]
            fs = F.build_fields(ctx["pool"], ctx["inputs"], contest, k=args.fields,
                                ownership=ops[arch], sd=sd, seed=args.seed)
            base = F.build_field(ctx["pool"], ctx["inputs"], contest, ownership=ops[arch],
                                 seed=args.seed)
            b = base.realized_ownership()
            draw_mae, real_mae, real_vs_base = [], [], []
            for f in fs.fields:
                pr = ops[arch].as_dict()
                draw_mae.append(RS.own_metrics(pr, f.ownership.as_dict())["mae_pp"])
                real_mae.append(RS.own_metrics(pr, dict(zip(f.pids, f.realized_ownership())))
                                ["mae_pp"])
                real_vs_base.append(RS.own_metrics(dict(zip(base.pids, b)),
                                                   dict(zip(f.pids, f.realized_ownership())))
                                    ["mae_pp"])
            out["fit"][arch]["fields"] = {
                "k": args.fields, "draw_mae_pp": draw_mae, "realized_vs_proxy_mae_pp": real_mae,
                "realized_vs_base_field_mae_pp": real_vs_base,
                "base_field_vs_proxy_mae_pp": RS.own_metrics(ops[arch].as_dict(),
                                                             dict(zip(base.pids, b)))["mae_pp"],
                "seconds": fs.seconds}
            print(json.dumps({"arch": arch, **{k: (np.round(v, 2).tolist()
                                                   if isinstance(v, list) else v)
                                               for k, v in out["fit"][arch]["fields"].items()}}),
                  flush=True)
    return out


def cmd_pit(args, conn) -> dict:
    """P(cash) / P(top 1 %) / PIT of our lineups vs each real contest, OFF vs ON."""
    from courtside.proj.dfs import portfolio as PF
    ctx = RS.setup(conn, args.slate, args.run)
    rcs = RS.real_contests(conn, args.slate)
    lus = RS.our_lineups(conn, args.slate)
    unc = LNT.own_uncertainty(LNT.run_flags(conn, args.run))
    act = np.array([sum(ctx["actual"].get(p, 0.0) for _, p in d["pairs"]) for d in lus])
    res = {"run": args.run, "n_lineups": len(lus), "own_unc": unc, "k": args.k,
           "contests": []}
    for rc in rcs:
        c = rc["contest"]
        pts = np.sort(rc["points"])[::-1]
        line = float(pts[c.paid - 1])
        top1 = float(pts[max(1, int(math.ceil(0.01 * c.field_size))) - 1])
        crow = {"contest": c.contest_id, "name": c.name, "M": c.field_size,
                "archetype": c.archetype, "real_cash_line": line, "real_top1_line": top1,
                "real_cashed": float(np.mean(act >= line - 1e-9)),
                "real_top1": float(np.mean(act >= top1 - 1e-9)), "arms": {}}
        for arm in ("off", "on"):
            sv = []
            for sd in range(args.seeds):
                t0 = time.perf_counter()
                op = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=c.archetype)
                if arm == "on":
                    f = F.build_fields(ctx["pool"], ctx["inputs"], c, k=args.k, own_unc=unc,
                                       n=args.n, ownership=op, seed=sd)
                else:
                    f = F.build_field(ctx["pool"], ctx["inputs"], c, n=args.n, ownership=op,
                                      seed=sd)
                fp, _ = F.align_sims(ctx["sims"], f.pids, args.sims, sd)
                fp, _ = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                                      np.random.default_rng(sd + 7))
                ev = F.evaluate(f, fp, c, [d["pairs"] for d in lus], seed=sd, tiers=False,
                                per_sim=True)
                sc = (ev.per_sim["score"].astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
                dsim = sc - ev.cash_line[:, None]
                pit = (dsim < (act - line)[None, :]).mean(0)
                # PIT of the lineup's real score vs the simulated top-1 % line gap, and of the
                # real cash line's position vs our lineups' own simulated scores is not used
                v = {"p_cash": np.array([x["p_cash"] for x in ev.lineups]),
                     "p_top1": np.array([x["p_top1"] for x in ev.lineups]),
                     "pit": pit, "dsim_sd": dsim.std(0),
                     "cash_line_sd": float(ev.cash_line.std()),
                     "cash_p50": float(np.median(ev.cash_line)),
                     "seconds": time.perf_counter() - t0}
                if arm == "on":
                    v["p_cash_lo"] = np.array([x["p_cash_lo"] for x in ev.lineups])
                    v["p_cash_hi"] = np.array([x["p_cash_hi"] for x in ev.lineups])
                    v["p_top1_lo"] = np.array([x["p_top1_lo"] for x in ev.lineups])
                    v["p_top1_hi"] = np.array([x["p_top1_hi"] for x in ev.lineups])
                    v["cash_line_by_field"] = [float(np.median(ev.cash_line[f.rows_of(
                        len(ev.cash_line), j)])) for j in range(f.k)]
                sv.append(v)
            agg = {k: np.mean([v[k] for v in sv], axis=0) for k in sv[0]
                   if k not in ("cash_line_by_field",)}
            pit = agg["pit"]
            row = {"p_cash_mean": float(agg["p_cash"].mean()),
                   "p_top1_mean": float(agg["p_top1"].mean()),
                   "pit_mean": float(pit.mean()), "pit_median": float(np.median(pit)),
                   "pit_q10_q90": [float(np.quantile(pit, 0.1)), float(np.quantile(pit, 0.9))],
                   "pit_mid50_share": float(np.mean((pit >= 0.25) & (pit <= 0.75))),
                   "pit_ks_uniform": float(np.max(np.abs(np.sort(pit) - (np.arange(len(pit))
                                                                         + 0.5) / len(pit)))),
                   "pit_extreme_share": float(np.mean((pit < 0.1) | (pit > 0.9))),
                   "dsim_sd_mean": float(agg["dsim_sd"].mean()),
                   "cash_line_sd": float(agg["cash_line_sd"]),
                   "cash_p50": float(agg["cash_p50"]),
                   "brier_cash": float(np.mean((agg["p_cash"] - (act >= line - 1e-9)) ** 2)),
                   "seconds": float(agg["seconds"]), "pit": pit.tolist(),
                   "p_cash": agg["p_cash"].tolist()}
            if arm == "on":
                row["p_cash_band_mean"] = [float(agg["p_cash_lo"].mean()),
                                           float(agg["p_cash_hi"].mean())]
                row["p_top1_band_mean"] = [float(agg["p_top1_lo"].mean()),
                                           float(agg["p_top1_hi"].mean())]
                row["p_cash_band_width_mean"] = float((agg["p_cash_hi"] - agg["p_cash_lo"]).mean())
                row["p_top1_band_width_mean"] = float((agg["p_top1_hi"] - agg["p_top1_lo"]).mean())
                row["cash_line_by_field"] = [v["cash_line_by_field"] for v in sv]
            crow["arms"][arm] = row
            print(json.dumps({"contest": c.contest_id, "arm": arm,
                              **{k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in row.items() if k not in ("pit", "p_cash",
                                                                      "cash_line_by_field")},
                              "real_cashed": round(crow["real_cashed"], 3)}), flush=True)
        res["contests"].append(crow)
    return res


def cmd_bench(args, conn) -> dict:
    from datetime import datetime, timedelta, timezone
    from courtside.proj.dfs import portfolio as PF
    lock = conn.execute("SELECT lock_utc FROM dfs_slates WHERE slate_id=?",
                        (str(args.slate),)).fetchone()[0]
    now = datetime.strptime(lock[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc) - timedelta(minutes=30)          # before the first game: no bans
    out = []
    for n in [int(x) for x in args.ns.split(",")]:
        for arm in args.arms.split(","):
            F.set_field_k(args.k if arm == "on" else 0)
            s = PF.Settings(n=n, strengths=True, seed=args.seed)
            t0 = time.perf_counter()
            pf = PF.build_for_slate(conn, args.slate, args.contest, s, run_id=args.run,
                                    now=now)
            sec = time.perf_counter() - t0
            row = {"n": n, "arm": arm, "k": F.field_k(), "seconds": round(sec, 1),
                   "timing": pf.info["timing"], "e_payout": pf.summary["e_payout"],
                   "p_any_cash": pf.summary["p_any_cash"],
                   "p_any_top1": pf.summary["p_any_top1"],
                   "mean_p_cash": float(np.mean([x["p_cash"] for x in pf.lineups])),
                   "mean_p_top1": float(np.mean([x["p_top1"] for x in pf.lineups])),
                   "field_k": pf.summary.get("field_k"),
                   "version": pf.info["version"], "field_version": pf.info["field"]["version"]}
            if "field_k" in pf.info["field"]:
                row["field_info"] = pf.info["field"]["field_k"]
            out.append(row)
            print(json.dumps(row, default=float), flush=True)
    F.set_field_k(0)
    return {"slate": args.slate, "contest": args.contest, "run": args.run, "rows": out}


def cmd_lines(args, conn) -> dict:
    """Field realism at the ACTUAL box scores (as `real_standings_calib.py field`): each real
    contest's cash / top-1 % / winning score and concentration (Σ own², real %Drafted) next to
    the one field (OFF) and each of the K fields (ON), every lineup scored with the actual DK
    points."""
    ctx = RS.setup(conn, args.slate, args.run)
    rcs = RS.real_contests(conn, args.slate)
    op0 = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype="large_gpp")
    tab = RS.pool_table(ctx, op0)
    act = tab["actual"].to_numpy()
    out = []
    for rc in rcs:
        c = rc["contest"]
        pts = np.sort(rc["points"])[::-1]
        real = {"cash": float(pts[c.paid - 1]),
                "top1": float(pts[max(1, int(math.ceil(0.01 * c.field_size))) - 1]),
                "first": float(pts[0]),
                "sum_own2": float(sum(v * v for v in rc["own"].values()))}
        row = {"contest": c.contest_id, "name": c.name, "M": c.field_size, "real": real}
        for arm in ("off", "on"):
            vals = []
            for sd in range(args.seeds):
                op = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=c.archetype)
                fl = ([F.build_field(ctx["pool"], ctx["inputs"], c, ownership=op, seed=sd)]
                      if arm == "off" else
                      F.build_fields(ctx["pool"], ctx["inputs"], c, k=args.k, ownership=op,
                                     seed=sd).fields)
                for f in fl:
                    Lf, wf = RS.field_lineups(f, tab)
                    sc = act[Lf].sum(1)
                    ln = RS.lines(sc, wf, c, {})
                    own = f.realized_ownership()
                    # the winner among M entries: mean of the max over draws of M lineups
                    rng = np.random.default_rng(sd)
                    pr = wf / wf.sum()
                    mx = [float(sc[rng.choice(len(sc), c.field_size, p=pr)].max())
                          for _ in range(20)]
                    vals.append({"cash": ln["cash_line"], "top1": ln["top1"],
                                 "first": float(np.mean(mx)), "sum_own2": float(np.sum(own ** 2))})
            row[arm] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            row[arm + "_sd"] = {k: float(np.std([v[k] for v in vals])) for k in vals[0]}
        out.append(row)
        print(json.dumps({"contest": c.contest_id, "M": c.field_size, "real": real,
                          "off": {k: round(v, 2) for k, v in row["off"].items()},
                          "on": {k: round(v, 2) for k, v in row["on"].items()},
                          "on_sd": {k: round(v, 2) for k, v in row["on_sd"].items()}}),
              flush=True)
    return {"rows": out}


def cmd_payout(args, conn) -> dict:
    """Why E[payout] moves: the OFF build's chosen set scored on the same truth rows against the
    one field, K fields with sd = 0 (seed variation only) and K fields at OWN_SD (and x2)."""
    from datetime import datetime, timedelta, timezone
    from courtside.proj.dfs import portfolio as PF
    lock = conn.execute("SELECT lock_utc FROM dfs_slates WHERE slate_id=?",
                        (str(args.slate),)).fetchone()[0]
    now = datetime.strptime(lock[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc) - timedelta(minutes=30)
    F.set_field_k(0)
    out = {"rows": []}
    for n in [int(x) for x in args.ns.split(",")]:
        s = PF.Settings(n=n, strengths=False, curve=False, seed=args.seed)
        pf = PF.build_for_slate(conn, args.slate, args.contest, s, run_id=args.run, now=now)
        pairs = [[(d["slot"], str(d["player_id"])) for d in json.loads(x["slots_json"])]
                 for x in pf.lineups]
        fp, c = pf.fp, pf.contest
        ctx = F.prepare(conn, args.slate, args.run, n_sims=s.n_sims, seed=s.seed)
        lad = PF.Ladder(c)
        arms = {"off": pf.field}
        for name, sd in (("k_sd0", 0.0), ("k_fit", None), ("k_fit_x2", 2 * F.OWN_SD[
                c.archetype])):
            arms[name] = F.build_fields(ctx["pool"], ctx["inputs"], c, k=args.k,
                                        ownership=pf.field.ownership, sd=sd, seed=s.seed,
                                        threads=2)
        for name, f in arms.items():
            t0 = time.perf_counter()
            ev = F.evaluate(f, fp, c, pairs, seed=s.seed, tiers=False, own_entries=n,
                            per_sim=True)
            ps = ev.per_sim
            pay, rank = PF.joint_payouts(lad, ps["score"], np.rint(ps["above"]).astype(np.int64),
                                         np.rint(ps["tied"]).astype(np.int64))
            tot = pay.sum(1)
            row = {"n": n, "arm": name, "e_payout": float(tot.mean()),
                   "e_payout_se": float(tot.std() / math.sqrt(len(tot))),
                   "p_any_cash": float((rank <= lad.paid).any(1).mean()),
                   "p_any_top1": float((rank <= lad.top1).any(1).mean()),
                   "p_first": float((rank == 1).any(1).mean()),
                   "mean_p_cash": float((rank <= lad.paid).mean()),
                   "cash_line_p50": float(np.median(ev.cash_line)),
                   "top1_line_p50": float(np.median(ev.top1_line)),
                   "top1_line_sd": float(np.std(ev.top1_line)),
                   "seconds": round(time.perf_counter() - t0, 1)}
            if name != "off":
                k = f.k
                row["e_payout_k"] = [float(tot[f.rows_of(len(tot), j)].mean()) for j in range(k)]
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fit", "pit", "bench", "payout", "lines"])
    ap.add_argument("--slate", default="154575")
    ap.add_argument("--run", default="c15398ddd87f")
    ap.add_argument("--draws", type=int, default=60, help="fit: draws per sd")
    ap.add_argument("--fields", type=int, default=4, help="fit: realised-field check (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="pit: seeds per arm")
    ap.add_argument("--k", type=int, default=F.FIELD_K_ON)
    ap.add_argument("--n", type=int, default=None, help="pit: field sample")
    ap.add_argument("--sims", type=int, default=10_000, help="pit: sims subsample")
    ap.add_argument("--contest", default="196981120", help="bench: contest id")
    ap.add_argument("--ns", default="20,150", help="bench: entry counts")
    ap.add_argument("--arms", default="off,on", help="bench: arms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    conn = db.connect()
    res = {"fit": cmd_fit, "pit": cmd_pit, "bench": cmd_bench,
           "payout": cmd_payout, "lines": cmd_lines}[args.cmd](args, conn)
    txt = json.dumps(res, indent=1, default=float)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
