"""D8 accept test: replay stored late-swap nights, contest-aware vs mean re-solve
(IMPROVEMENTS.md D8; docs/research/d8-d9.md).

Per night (slate, swap time T) and seed:
* entries, as the owner could have had them at T:
  - "builder": a 20-entry contest-builder portfolio (`portfolio.build_portfolio`, shipped
    settings) for the night's main GPP, built on the last pre-lock run with sims;
  - "saved": the site's saved lineups as they stood at T (the latest build's cash / Top N, with
    swaps saved before T overlaid); in `swap.plan` they have no contest, so both arms score them
    in the slate's default contests (`field.default_contests`: largest double-up / GPP);
* the projection run = the newest run with sims at or before T (the swap run's own re-projection
  when it ran); locked players' realised points = their graded actual DK points;
* the stored sim rows are shuffled (seed) into A (selection: ON may only use these) and B
  (held out);
* arms: "mean" = `swap.plan(contest_aware=False)` (today), "ca" = `swap.plan(contest_aware=True,
  locked_points=..., sim_rows=A)`, "keep" = no swap;
* scored on B: the same conditioning (Outs known at T via the scenario rows, locked players fixed
  at their realised points), the contest's medium field, the builder's dupe model; GPP groups:
  joint E[payout] of the group's entries; cash: mean P(cash). Paired per sim (same rows, same
  truth noise): ca - mean, with the SE over B rows.

Read-only DB (mode=ro).

    COURTSIDE_DATA=... PYTHONPATH=src:research python research/d8_replay.py \\
        --nights 155553:2026-10-24T01:51:10Z:197042080 --seeds 2 --out d8.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from courtside.proj import role_blend as RB
from courtside.proj import store as SO
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import runner
from courtside.proj.dfs import swap
from courtside.proj.dfs import swap_contest as SC

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_split_eval as PSE        # noqa: E402  (connect, load, MAC_VAR)
import real_standings_calib as RS         # noqa: E402  (actual_points)

ARMS = ("keep", "mean", "ca")


def T(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run_at(conn, slate: str, at: str, *, pre_lock: bool = False) -> str:
    """The newest ok run with sims at or before `at` (pre_lock: a non-swap run at or before the
    slate's lock)."""
    if pre_lock:
        lock = conn.execute("SELECT lock_utc FROM dfs_slates WHERE slate_id=?", (slate,)).fetchone()
        at = min(at, lock[0]) if lock and lock[0] else at
    r = conn.execute("SELECT run_id FROM dfs_runs WHERE slate_id=? AND ok=1 AND sim_path IS NOT "
                     "NULL AND created_at<=? " + ("AND kind!='swap' " if pre_lock else "") +
                     "ORDER BY created_at DESC LIMIT 1", (slate, at)).fetchone()
    if r is None:
        raise SystemExit(f"{slate}: no run with sims before {at}")
    return r[0]


def saved_at(conn, slate: str, at: str) -> list[swap.SwapEntry]:
    """`swap.saved_entries` as of `at`: the latest build run (created before `at`) with cash /
    Top N lineups, swaps saved before `at` overlaid."""
    runs = [r for r in runner._run_rows(conn, slate) if r["created_at"] < at]
    base = None
    for r in reversed(runs):
        if r["kind"] != "swap":
            rows = runner._lineup_rows(conn, r["run_id"], ("cash", "topn"))
            if rows:
                base = (r, rows)
                break
    if base is None:
        return []
    brun, rows = base
    cur = {(d["mode"], d["rank"]): d for d in rows}
    for r in runs:
        if r["kind"] != "swap" or r["created_at"] <= brun["created_at"]:
            continue
        for d in runner._lineup_rows(conn, r["run_id"], ("swap",)):
            frm = d["why"].get("from") or {}
            key = (frm.get("mode"), frm.get("rank"))
            if key in cur:
                cur[key] = {**d, "mode": key[0], "rank": key[1]}
    out = []
    for d in sorted(cur.values(), key=lambda d: (d["mode"] != "cash", d["rank"] or 0)):
        slots = runner._loads(d.get("slots_json"), []) or []
        out.append(swap.SwapEntry(
            key=f"{d['mode']}-{d['rank']}", label=f"{d['mode']} {d['rank']}",
            pairs=list(d["pairs"]), names=[s.get("name") or str(s.get("player_id"))
                                           for s in slots], mode=d["mode"], slots_meta=slots))
    return out


def builder_entries(conn, slate: str, at: str, cid: str, n: int, seed: int) -> list:
    run = run_at(conn, slate, at, pre_lock=True)
    ctx = PSE.load(conn, slate, run)
    contest = F.load_contest(conn, cid)
    s = PF.Settings(n=min(n, contest.field_size - 1), seed=seed, threads=2, curve=False)
    pf = PF.build_portfolio(ctx["pool"], ctx["inputs"], ctx["sims"], contest, s,
                            blended=RB.run_blended(conn, run))
    out = []
    for i, sl in enumerate(pf.slots()):
        out.append(swap.SwapEntry(
            key=f"pf-{i + 1}", label=f"builder {i + 1}", entry_id=str(9000 + i),
            pairs=[(x["slot"], str(x["player_id"])) for x in sl],
            names=[x.get("name") or str(x["player_id"]) for x in sl],
            contest_id=cid, contest_name=contest.name, entry_fee=str(contest.entry_fee),
            slots_meta=sl))
    return out, {"run": run, "version": pf.info["version"], "e_payout": pf.summary["e_payout"]}


def finals(sp) -> dict[str, list]:
    return {r.entry.key: (r.after_pairs if r.status == "swapped" else list(r.entry.pairs))
            for r in sp.results}


def score_group(ctx: SC.Context, contest: F.Contest, lineups: list[list], mode: str) -> np.ndarray:
    """Per-sim value of a group of our entries on the context's rows: GPP joint total payout,
    cash the mean P(cash) over the entries."""
    cs = ctx.get(contest)
    lad = cs.lad if mode != "cash" else SC.cash_ladder(cs.lad)
    n = len(lineups)
    ev = F.evaluate(cs.field, cs.fp, contest, lineups, seed=ctx.seed, tiers=False,
                    own_entries=n, per_sim=True)
    P = PF.dupe_pool(cs.field, ev, lineups, contest, max(1, contest.field_size - n),
                     np.random.default_rng(ctx.seed + 3))
    pay, _ = PF.joint_payouts(lad, P.score, P.above, P.tied)
    return pay.sum(1) / (n if mode == "cash" else 1)


def night(conn, slate: str, at: str, cid: str, seed: int, n_builder: int) -> dict:
    t0 = time.perf_counter()
    now = T(at)
    run = run_at(conn, slate, at)
    pool = R.build_pool(conn, slate)
    acts = RS.actual_points(conn, slate, run, pool)
    started = {p.dk_player_id for p in pool.players if p.is_locked(now)}
    locked = {pid: v for pid, v in acts.items() if pid in started}
    path = conn.execute("SELECT sim_path FROM dfs_runs WHERE run_id=?", (run,)).fetchone()[0]
    import os
    S = SO.load_sims(str(path).replace(PSE.MAC_VAR, os.environ["COURTSIDE_DATA"]))["fp"].shape[0]
    perm = np.random.default_rng(500 + seed).permutation(S)
    A, B = np.sort(perm[: S // 2]), np.sort(perm[S // 2:])
    bld, binfo = builder_entries(conn, slate, at, cid, n_builder, seed)
    saved = saved_at(conn, slate, at)
    entries = bld + saved
    plans = {}
    for arm in ("mean", "ca"):
        t = time.perf_counter()
        plans[arm] = swap.plan(conn, slate, now=now, entries=entries, run_id=run,
                               contest_aware=arm == "ca", locked_points=locked, sim_rows=A,
                               seed=seed)
        plans[arm].seconds = time.perf_counter() - t
    fin = {"keep": {e.key: list(e.pairs) for e in entries},
           "mean": finals(plans["mean"]), "ca": finals(plans["ca"])}
    # held-out scoring context: rows B, same conditioning
    ev_ctx, why = SC.prepare(conn, slate, run, news=plans["ca"].news, pool_players=pool.players,
                             locked_points=locked, rows=B, n_sims=None, seed=1000 + seed)
    if ev_ctx is None:
        raise SystemExit(f"{slate}: {why}")
    cash_c, gpp_c = F.default_contests(conn, slate)
    groups = {"builder": ([e.key for e in bld], F.load_contest(conn, cid), "mean"),
              "saved_topn": ([e.key for e in saved if e.mode == "topn"], gpp_c, "mean"),
              "saved_cash": ([e.key for e in saved if e.mode == "cash"], cash_c, "cash")}
    out = {"slate": slate, "at": at, "run": run, "seed": seed, "builder": binfo,
           "locked_players": len(locked), "rows": {"A": int(len(A)), "B": int(len(B))},
           "ca_info": {k: v for k, v in (plans["ca"].contest or {}).items() if k != "decisions"},
           "status": {arm: dict(plans[arm].summary()["status"]) for arm in ("mean", "ca")},
           "seconds": {arm: round(plans[arm].seconds, 1) for arm in ("mean", "ca")},
           "warnings": {arm: plans[arm].warnings for arm in ("mean", "ca")}, "groups": {}}
    for g, (keys, contest, mode) in groups.items():
        if not keys or contest is None:
            continue
        vals = {arm: score_group(ev_ctx, contest, [fin[arm][k] for k in keys], mode)
                for arm in ARMS}
        changed = {arm: sum(fin[arm][k] != fin["keep"][k] for k in keys) for arm in ("mean", "ca")}
        res = {"contest": contest.contest_id, "name": contest.name, "M": contest.field_size,
               "fee": contest.entry_fee, "mode": mode, "entries": len(keys), "changed": changed,
               "n_sims": int(len(vals["keep"]))}
        for arm in ARMS:
            res[arm] = float(vals[arm].mean())
        for a, b in (("ca", "mean"), ("mean", "keep"), ("ca", "keep")):
            d = vals[a] - vals[b]
            res[f"{a}_minus_{b}"] = {"mean": float(d.mean()),
                                     "se": float(d.std() / math.sqrt(len(d)))}
        out["groups"][g] = res
    out["seconds_total"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", required=True, help="slate:ISO-time:main-gpp-contest,...")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    conn = PSE.connect()
    res = []
    for item in args.nights.split(","):
        slate, rest = item.split(":", 1)
        at, cid = rest.rsplit(":", 1)
        for seed in range(args.seed0, args.seed0 + args.seeds):
            r = night(conn, slate, at, cid, seed, args.n)
            res.append(r)
            print(json.dumps({k: r[k] for k in ("slate", "at", "seed", "locked_players",
                                                 "status", "seconds", "seconds_total")}
                             | {"groups": {g: {k: v for k, v in x.items()
                                               if k not in ("name",)}
                                           for g, x in r["groups"].items()}}), flush=True)
            Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
