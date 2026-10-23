"""D2 evaluation: cash lineup by P(cash) (`proj/dfs/cash_pcash.py`, COURTSIDE_CASH_PCASH) vs the
current rule (mu - 0.3 (mu - p25)). See docs/research/cash-pcash.md.

Commands (read-only on the DB COURTSIDE_DATA points at; run on a scratch copy):

  rg       RotoGuru walk-forward (2019-20, 2020-21: real salaries), the T2.13 harness of
           `research/cash_calibration.py`: `pipeline.project_historical(protocol='avail')`, a
           preset double-up (`--field` entries, 44.4 % paid) and its cash-archetype field proxy
           from public information (public = 1/2 ours + 1/2 season average). Candidates = Top N
           (`--topn`, mean) + the λ grid; pick on the S rows of the calibrated truth, report on R
           (`cash_pcash.split`). Realised = the slate's ACTUAL points against the same field,
           opponents redrawn `--draws` times (the cash line of cash_calibration.py).
  opener   2026-10-20 (slate 154575): the stored lock-45 / lock-10 runs (their projections, sims
           and saved lineups) against each real double-up (field = the production cash field for
           that contest), the pick's actual points vs the real cash lines.
  timing   the ON overhead of a production build on stored runs (`slate:run` pairs): `choose`
           (field, solves, scoring) and `score_run` with / without the field reuse.
  offcheck dump `runner._build` lineups of stored runs from the tree on PYTHONPATH (run it with
           a `git show HEAD:` tree and the working tree, then compare the JSON files).

    PYTHONPATH=src python research/cash_pcash_eval.py rg --season 2019-20 --out out/rg_1920.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
try:
    from courtside.proj.dfs import cash_pcash as CP
except ImportError:          # offcheck on a `git show HEAD:` tree (before D2)
    CP = None
from courtside.proj.dfs import field as F
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import topn as T

sys.path.insert(0, str(Path(__file__).parent))

W_OURS = 0.5            # live-like public projection (cash_calibration.py "medium")
REAL_DU = {"196970732": 237.25, "196456023": 247.75, "196456027": 238.75}


# ============================================================================== shared

def realised(f, contest, lineups, act: np.ndarray, draws: int, seed: int
             ) -> tuple[list[float], float]:
    """P(cash) of each lineup at the actual points, opponents redrawn `draws` times."""
    ev = F.evaluate(f, np.repeat(act[None, :], draws, 0), contest, lineups, seed=seed,
                    tiers=False)
    return [float(x["p_cash"]) for x in ev.lineups], float(np.median(ev.cash_line))


def decide(cands, p_s, res, rules):
    """(pick index, reranked Top N, index of each reranked Top N lineup in cands)."""
    k = CP.pick(cands, p_s)
    idx = {CP._ident(rules, lu): i for i, (_, lu) in enumerate(cands)}
    p_top = [p_s[idx[CP._ident(rules, r.lineup)]] for r in res]
    new = T.rank_by_mean(res, p_top)
    return k, new, [idx[CP._ident(rules, r.lineup)] for r in new]


# ============================================================================== rg

def rg_date(conn, sr, day: str, args) -> dict | None:
    from courtside.proj import pipeline as P
    from courtside.proj.dfs import portfolio as PF
    from portfolio_backtest import slate_sims
    t0 = time.time()
    pool = O.hist_pool(conn, day, sr.rules)
    inp = F.hist_inputs(conn, pool)
    if inp["fppg"].notna().sum() < 40:
        return None
    rows = sr.rg[sr.rg["game_date"] == day]
    rg_of = {int(r.player_id): str(r.rg_id) for r in rows.itertuples(index=False)
             if not pd.isna(r.player_id)}
    res_p = P.project_historical(conn, day, protocol="avail", n_sims=args.sims, ctx=sr.ctx,
                                 bundle=sr.bundle)
    pl = res_p.players.drop_duplicates("player_id").set_index("player_id")
    in_pool = {p.dk_player_id for p in pool.players}
    proj = {rg_of[int(i)]: O.Projection(float(q["dk_mean"]), p25=float(q["dk_q25"]))
            for i, q in pl.iterrows() if int(i) in rg_of and rg_of[int(i)] in in_pool}
    if len(proj) < 3 * sr.rules.n_slots:
        return None
    inp["proj"] = inp["pid"].map({k: v.mean for k, v in proj.items()})
    sims = slate_sims(res_p, rg_of)
    t1 = time.time()
    cash = O.optimize(pool, proj, sr.rules, mode="cash", lam=CP.RULE_LAMBDA, time_limit_s=20)
    res = T.top_n(pool, proj, sr.rules, args.topn, mode="mean", time_limit_s=20,
                  total_time_limit_s=90)
    if cash is None or not len(res):
        return None
    cands = CP.candidates(O, pool, proj, sr.rules, set(), None, cash, res, solve_limit_s=20)
    t2 = time.time()
    contest = F.preset_contest("double_up", args.field, 5.0)
    seed = P.stable_seed(day, "cash-cal")
    params, ver = F.load_params()
    op = F.ownership_proxy(inp, pool, sr.rules, archetype="cash",
                           params={**params, "w_ours": W_OURS}, version=ver)
    f = F.build_field(pool, inp, contest, rules=sr.rules, ownership=op, seed=seed)
    fp_raw, _ = F.align_sims(sims, f.pids)
    fp, _ = PF.truth_sims(fp_raw, f.pids, inp, PF.TRUTH_BETA, PF.TRUTH_TAU,
                          np.random.default_rng(seed + 7))
    rs, rr = CP.split(fp.shape[0])
    lus = [lu.slot_pairs() for _, lu in cands]
    p_s, p_r = CP.score(f, fp, contest, [lu for _, lu in cands], rs, rr)
    actual = O.hist_projections(conn, pool, "actual")
    act = np.array([actual[p].mean if p in actual else 0.0 for p in f.pids], np.float32)
    p_a, line = realised(f, contest, lus, act, args.draws, seed)
    pts = [float(sum(actual[p].mean for _, p in lu if p in actual)) for lu in lus]
    k, new, order = decide(cands, p_s, res, sr.rules)
    i_top1 = next(i for i, (s, _) in enumerate(cands) if s == "topn1") \
        if any(s == "topn1" for s, _ in cands) else 0
    out = {"date": day, "season": sr.season, "games": len(res_p.sims), "field": args.field,
           "n_cands": len(cands), "pick": cands[k][0], "changed": int(k != 0),
           "rule_real": p_a[0], "on_real": p_a[k], "top1_real": p_a[i_top1],
           "rule_pts": pts[0], "on_pts": pts[k], "top1_pts": pts[i_top1],
           "rule_sim_r": p_r[0], "on_sim_r": p_r[k], "on_sim_s": p_s[k], "rule_sim_s": p_s[0],
           "new1_real": p_a[order[0]], "new1_pts": pts[order[0]],
           "topn_reordered": int([r.lineup.player_ids for r in new]
                                 != [r.lineup.player_ids for r in res]),
           "new1_changed": int(new.lineups[0].lineup.player_ids != res.lineups[0].lineup.player_ids),
           "cash_line_real": line, "cash_line_sim_p50": float(np.median(
               F.evaluate(f, fp[rr], contest, [], seed=seed, tiers=False).cash_line)),
           "cands": json.dumps([{"src": s, "mean": round(lu.projected, 2), "ps": round(p_s[i], 4),
                                 "pr": round(p_r[i], 4), "real": round(p_a[i], 4),
                                 "pts": pts[i]} for i, (s, lu) in enumerate(cands)]),
           "sec_proj": round(t1 - t0, 1), "sec_solve": round(t2 - t1, 1),
           "seconds": round(time.time() - t0, 1)}
    return out


def cmd_rg(args, conn) -> int:
    from courtside.proj import backtest as BT
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    have = set(done["date"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    sr = BT.SeasonRun(conn, args.season, BT.Settings(modes=("cash",), n_sims=args.sims),
                      log=lambda m: print(m, flush=True))
    for day in sr.dates(args.dates):
        if day in have:
            continue
        if args.stop_at and time.strftime("%H%M") >= args.stop_at:
            print(f"stop-at {args.stop_at} reached", flush=True)
            break
        try:
            r = rg_date(conn, sr, day, args)
        except Exception as e:  # noqa: BLE001
            print(f"{day}: failed {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"{day}: skipped", flush=True)
            continue
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{day}: pick {r['pick']:<8} rule {r['rule_real']:.2f} on {r['on_real']:.2f} "
              f"({r['seconds']}s)", flush=True)
    return 0


def paired(d: pd.Series) -> str:
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.4f} ± {se:.4f} (z {d.mean() / se if se else float('nan'):+.2f})"


def cmd_summary(args, conn=None) -> int:
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    lines = [f"slates {len(df)} ({', '.join(sorted(df.season.unique()))}); preset double-up "
             f"{int(df.field.median())} entries, 44.4 % paid; realised = actual points vs the "
             f"proxy field (opponents redrawn)"]
    for name, g in [("all", df)] + list(df.groupby("season")):
        lines.append(f"[{name}] n {len(g)}, pick changed {g.changed.mean():.1%}; realised cash "
                     f"rule {g.rule_real.mean():.4f}, ON {g.on_real.mean():.4f}, Top-N #1 "
                     f"{g.top1_real.mean():.4f}")
        lines.append(f"   ON - rule: {paired(g.on_real - g.rule_real)};  points "
                     f"{paired(g.on_pts - g.rule_pts)}")
        lines.append(f"   ON - Top-N #1: {paired(g.on_real - g.top1_real)}")
        lines.append(f"   new Top-N #1 - old #1: {paired(g.new1_real - g.top1_real)} "
                     f"(#1 changed {g.new1_changed.mean():.1%}, reordered "
                     f"{g.topn_reordered.mean():.1%})")
        lines.append(f"   sim P(cash) on R: rule {g.rule_sim_r.mean():.4f}, ON "
                     f"{g.on_sim_r.mean():.4f} (S {g.on_sim_s.mean():.4f}); realised − sim R: "
                     f"ON {paired(g.on_real - g.on_sim_r)}")
        ch = g[g.changed == 1]
        if len(ch) > 1:
            lines.append(f"   changed slates only (n {len(ch)}): ON - rule "
                         f"{paired(ch.on_real - ch.rule_real)}; sim R gain "
                         f"{paired(ch.on_sim_r - ch.rule_sim_r)}")
    if "rule_pts" in df:
        lines.append(f"fixed line 300 (backtest.py cash_rate_proxy_ge_300): rule "
                     f"{(df.rule_pts >= 300).mean():.3f}, ON {(df.on_pts >= 300).mean():.3f}, "
                     f"diff {paired((df.on_pts >= 300).astype(float) - (df.rule_pts >= 300))}")
    lines.append("picks: " + ", ".join(f"{k} {v}" for k, v in df.pick.value_counts().items()))
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    return 0


# ============================================================================== stored runs

def stored_build(conn, slate: str, run_id: str):
    """(pool, projs, cash, Top N result, bans) of a stored run, re-solved from its stored
    projections: the salary capture the run saw and the players its availability rule excluded (no `now`:
    nothing is locked)."""
    from courtside.proj.dfs import runner
    from courtside.proj.dfs import rules as R
    row = conn.execute("SELECT created_at, detail FROM dfs_runs WHERE run_id=?",
                       (run_id,)).fetchone()
    detail = json.loads(row[1] or "{}")
    bans = {str(e["player_id"]) for e in (detail.get("availability") or {}).get("excluded", [])}
    pr = F._run_projections(conn, run_id)
    projs = {str(r.pid): O.Projection(float(r.proj), p25=None if pd.isna(r.p25) else float(r.p25))
             for r in pr.itertuples() if not pd.isna(r.proj)}
    pool = R.build_pool(conn, slate, captured_at=F.capture_at(conn, slate, row[0]))
    bans &= {p.dk_player_id for p in pool.players}
    cash, res = runner._build(O, T, pool, projs, pool.rules, bans, None, 10, 1, {})
    return pool, projs, cash, res, bans


def cmd_offcheck(args, conn) -> int:
    out = {}
    for pair in args.runs.split(","):
        slate, run = pair.split(":")
        _, _, cash, res, _ = stored_build(conn, slate, run)
        stored = conn.execute("SELECT mode, rank, slots_json FROM dfs_lineups WHERE run_id=? "
                              "AND mode IN ('cash', 'topn') ORDER BY mode, rank", (run,)).fetchall()
        ids = lambda js: sorted(x["player_id"] for x in json.loads(js))  # noqa: E731
        rebuilt = [("cash", 1, cash.slots_json())] + [("topn", r.rank, r.lineup.slots_json())
                                                      for r in res]
        out[pair] = {"cash": cash.slots_json() if cash else None,
                     "topn": [[r.rank, r.lineup.slots_json(), r.lineup.projected,
                               r.diff.d_projected] for r in res],
                     "same_players_as_stored": sum(
                         1 for a, b in zip(stored, rebuilt) if a[0] == b[0] and a[1] == b[1]
                         and ids(a[2]) == ids(b[2])), "stored": len(stored)}
    txt = json.dumps(out, indent=0, sort_keys=True)
    Path(args.out).write_text(txt)
    print(f"{len(out)} runs -> {args.out} ({sum(len(v['topn']) + 1 for v in out.values())} lineups)")
    return 0


def cmd_timing(args, conn) -> int:
    rows = []
    for pair in args.runs.split(","):
        slate, run = pair.split(":")
        pool, projs, cash, res, bans = stored_build(conn, slate, run)
        t0 = time.perf_counter()
        _, _, info = CP.choose(conn, slate, run, O, T, pool, projs, pool.rules, bans, None,
                                 cash, res, solve_limit_s=20.0)
        t_choose = time.perf_counter() - t0
        t0 = time.perf_counter()
        with CP.reuse_fields(run):
            F.score_run(conn, run, slate_id=slate, write=False, commit=False)
        t_reuse = time.perf_counter() - t0
        t0 = time.perf_counter()
        F.score_run(conn, run, slate_id=slate, write=False, commit=False)
        t_plain = time.perf_counter() - t0
        r = {"slate": slate, "run": run, "pool": len(pool.players),
             "choose_s": round(t_choose, 2), **{f"choose_{k}": v for k, v in
                                                info.get("seconds", {}).items()},
             "score_run_reuse_s": round(t_reuse, 2), "score_run_plain_s": round(t_plain, 2),
             "added_s": round(t_choose + t_reuse - t_plain, 2), "pick": info.get("pick"),
             "changed": info.get("changed"), "n_cands": len(info.get("candidates", []))}
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def cmd_opener(args, conn) -> int:
    from courtside.proj.dfs import portfolio as PF
    import real_standings_calib as RS
    out = []
    contests = {rc["contest"].contest_id: rc for rc in RS.real_contests(conn, args.slate)}
    for run in args.opener_runs.split(","):
        pool, projs, cash, res, bans = stored_build(conn, args.slate, run)
        stored = {(r["mode"], r["rank"]): r["slots_json"] for r in conn.execute(
            "SELECT mode, rank, slots_json FROM dfs_lineups WHERE run_id=?", (run,))}
        same = stored.get(("cash", 1)) and {s["player_id"] for s in json.loads(stored[("cash", 1)])} \
            == set(cash.player_ids)
        cands = CP.candidates(O, pool, projs, pool.rules, bans, None, cash, res, solve_limit_s=20)
        ctx = F.prepare(conn, args.slate, run)
        actual = RS.actual_points(conn, args.slate, run, pool)
        pts = [sum(actual.get(p, 0.0) for _, p in lu.slot_pairs()) for _, lu in cands]
        for cid, line in REAL_DU.items():
            contest = contests[cid]["contest"] if cid in contests else F.load_contest(conn, cid)
            f = F.field_for(ctx, contest, n=CP.FIELD_N, seed=0)
            fp, _ = F.align_sims(ctx["sims"], f.pids)
            fp, _ = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                                  np.random.default_rng(7))
            rs, rr = CP.split(fp.shape[0])
            p_s, p_r = CP.score(f, fp, contest, [lu for _, lu in cands], rs, rr)
            k, _, order = decide(cands, p_s, res, pool.rules)
            r = {"run": run, "contest": cid, "entries": contest.field_size,
                 "paid": contest.paid, "real_line": line, "rebuild_matches_stored_cash": bool(same),
                 "rule_pts": pts[0], "rule_cash": pts[0] >= line, "rule_sim_r": p_r[0],
                 "pick": cands[k][0], "on_pts": pts[k], "on_cash": pts[k] >= line,
                 "on_sim_r": p_r[k], "new1": cands[order[0]][0], "new1_pts": pts[order[0]],
                 "cands": [{"src": s, "mean": round(lu.projected, 2), "ps": round(p_s[i], 4),
                            "pts": pts[i], "cash": pts[i] >= line}
                           for i, (s, lu) in enumerate(cands)]}
            out.append(r)
            print(json.dumps({k2: v for k2, v in r.items() if k2 != "cands"}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["rg", "summary", "opener", "timing", "offcheck"])
    ap.add_argument("--season", default="2019-20")
    ap.add_argument("--dates", type=int, default=0, help="rg: evenly spaced dates (0 = all)")
    ap.add_argument("--field", type=int, default=500)
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--stop-at", default=None, help="rg: local HHMM to stop before")
    ap.add_argument("--csv", nargs="+", default=[])
    ap.add_argument("--slate", default="154575")
    ap.add_argument("--opener-runs", default="42da71bcf37a,c15398ddd87f")
    ap.add_argument("--runs", default="")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "summary":
        return cmd_summary(args)
    conn = db.connect(args.db) if args.db else db.connect()
    return {"rg": cmd_rg, "opener": cmd_opener, "timing": cmd_timing,
            "offcheck": cmd_offcheck}[args.cmd](args, conn)


if __name__ == "__main__":
    sys.exit(main())
