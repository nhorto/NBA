"""D10: Showdown, contest builder vs Top N by mean, replayed on the stored Showdown slates.

IMPROVEMENTS.md D10 (depends on B6); write-up docs/research/b6-d10.md.

Per stored Showdown slate (its reference run: lock10, else lock45, made before lock; stored sims;
inputs as the public saw them at lock) and its captured contest (else a preset GPP of the median
captured Showdown shape), for N in --ns:

1. `portfolio.build_portfolio(..., baselines=True)` with the shipped settings (sim split "gs",
   selection objective "blend", field-proxy-1.2 at the contest's archetype, ownership captain
   term OFF = live). The Top N by mean is the builder's own baseline: the first N `topn.top_n`
   lineups (min different 1), the same lineups the slate page lists.
2. Both sets are scored JOINTLY (our N entries compete with each other) on held-out sims:
   * `R_infield`: the build's report rows (split "gs" R, 45 %), the build's own field sample
     (what the page reports; the field sample is shared with the selection);
   * `R_fresh_off`: the same R rows of the raw sims with a fresh truth-noise draw
     (`truth_sims`, new seed) and a fresh field sample (new seed), ownership OFF;
   * `R_fresh_cpt`: as R_fresh_off with the B6 captain term ON in the field (mis-specification
     check: is the ordering robust to how captains are owned?);
   * `actual`: the games' actual DK points (one draw per slate) vs the fresh OFF field.
   Per set: E[payout] (joint), P(any entry top 1 %), mean per-entry P(top 1 %), P(any cash);
   the builder - Top N paired difference per sim with its SE (iid over sims).
3. Pooled over slates (per N): mean ROI difference (payout / fees), SE from the sims and the
   between-slate SE.

   e2e      build_for_slate -> save -> load -> CSV export -> builder page view model, on one
            Showdown slate of the scratch DB (end-to-end check; writes the SCRATCH DB only).
   replay   the comparison above (resumable JSON lines in --out).

Usage (scratch DB, never prod):
  COURTSIDE_DATA=<scratch data> python research/d10_showdown_replay.py e2e --slate 155558
  COURTSIDE_DATA=<scratch data> python research/d10_showdown_replay.py replay --ns 1 5 20 \\
      --out out/d10_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF
from courtside.proj.dfs import standings as ST


def sd_slates(conn) -> list[dict]:
    out = []
    for r in conn.execute("SELECT slate_id, nba_day, lock_utc, game_ids_json FROM dfs_slates "
                          "WHERE tag='showdown' OR game_type_id=81 ORDER BY lock_utc"):
        sid = str(r["slate_id"])
        run = ST._reference_run_before(conn, sid, r["lock_utc"])
        if run is None:
            continue
        sp = conn.execute("SELECT sim_path FROM dfs_runs WHERE run_id=?",
                          (run["run_id"],)).fetchone()
        if not (sp and sp[0] and Path(sp[0]).exists()):
            continue
        out.append({"slate_id": sid, "day": r["nba_day"], "lock": r["lock_utc"],
                    "run_id": run["run_id"], "kind": run["kind"], "run_at": run["created_at"],
                    "game_ids": json.loads(r["game_ids_json"] or "[]")})
    return out


def contest_for(conn, sid: str) -> F.Contest:
    cs = F.slate_contests(conn, sid)
    gpp = [c for c in cs if c.archetype != "cash"]
    if gpp:
        return max(gpp, key=lambda c: c.field_size)
    # no captured contest: a preset GPP of the captured Showdown median shape (~3,100 entries)
    return F.preset_contest("gpp", 3137, 15.0)


def actual_points(conn, sl: dict, pool) -> dict[str, float]:
    from courtside.proj.dfs import grade as G
    day = date.fromisoformat(sl["day"])
    slate = next((s for s in G.day_slates(conn, day) if s["slate_id"] == sl["slate_id"]), None)
    if slate is None:
        return {}
    states = G.game_states(conn, day, slate["game_ids"])
    if not slate["game_ids"] or any(s != "final" for s in states.values()):
        return {}
    box = G.load_box(conn, slate["game_ids"])
    A = G.Actuals(conn, slate, day, box)
    return A.for_players(list(pool.players))


def joint(field, contest, fp, pairs, seed) -> dict:
    """Per-sim joint payout total, any-top-1 %, mean per-entry top-1 %, any-cash of a set."""
    lad = PF.Ladder(contest)
    ev = F.evaluate(field, fp, contest, list(pairs), seed=seed, tiers=False,
                    own_entries=len(pairs), per_sim=True)
    ps = ev.per_sim
    pay, rank = PF.joint_payouts(lad, ps["score"], np.rint(ps["above"]).astype(np.int64),
                                 np.rint(ps["tied"]).astype(np.int64))
    return {"pay": pay.sum(1), "any_top1": (rank <= lad.top1).any(1).astype(float),
            "top1": (rank <= lad.top1).mean(1), "any_cash": (rank <= lad.paid).any(1).astype(float)}


def compare(a: dict, b: dict, cost: float) -> dict:
    out = {}
    for k in ("pay", "any_top1", "top1", "any_cash"):
        x, y = a[k], b[k]
        d = x - y
        n = len(d)
        out[k] = {"builder": float(x.mean()), "topn": float(y.mean()), "diff": float(d.mean()),
                  "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")}
    out["roi_diff"] = out["pay"]["diff"] / cost if cost else None
    out["roi_diff_se"] = out["pay"]["se"] / cost if cost else None
    return out


def run_one(conn, sl: dict, n: int, args) -> dict:
    t0 = time.time()
    ctx = F.prepare(conn, sl["slate_id"], sl["run_id"], as_of=sl["lock"])
    pool, inp, sims = ctx["pool"], ctx["inputs"], ctx["sims"]
    contest = contest_for(conn, sl["slate_id"])
    r = conn.execute("SELECT max_per_user FROM dfs_contests WHERE contest_id=?",
                     (str(contest.contest_id),)).fetchone()
    max_user = int(r[0]) if r and r[0] else PF.MAX_ENTRIES
    n = max(1, min(n, max_user, contest.field_size - 1))
    seed = args.seed
    s = PF.Settings(n=n, seed=seed, threads=args.threads, curve=False, strengths=False,
                    n_sims=args.sims)
    pf = PF.build_portfolio(pool, inp, sims, contest, s, rules=pool.rules, baselines=True,
                            slate_id=sl["slate_id"], run_id=sl["run_id"],
                            now=datetime.fromisoformat(sl["run_at"].replace("Z", "+00:00")))
    sets = {"builder": pf.info["greedy_pairs"], "topn": pf.info["baselines"]["top_mean"]["pairs"]}
    cost = n * float(contest.entry_fee or 0)
    out = {"slate_id": sl["slate_id"], "run_id": sl["run_id"], "kind": sl["kind"], "n": n,
           "contest_id": contest.contest_id, "contest": contest.name, "field_size":
           contest.field_size, "fee": contest.entry_fee, "archetype": contest.archetype,
           "source": contest.source, "cost": cost, "pool": len(pool.players),
           "version": pf.info["version"], "ownership_version": pf.info["ownership_version"],
           "candidates": pf.info["candidates_total"], "entries": len(pf.lineups),
           "relative": pf.summary.get("relative"), "relative_se": pf.summary.get("relative_se"),
           "same_sets": sorted(map(str, sets["builder"])) == sorted(map(str, sets["topn"])),
           "overlap_lineups": len({str(sorted(x)) for x in sets["builder"]}
                                  & {str(sorted(x)) for x in sets["topn"]}),
           "cpt_builder": sorted({p for x in sets["builder"] for sl_, p in x if sl_ == "CPT"}),
           "cpt_topn": sorted({p for x in sets["topn"] for sl_, p in x if sl_ == "CPT"})}
    R = PF.split_rows(pf.fp.shape[0], s.seed, "gs")["R"]
    # R_infield: the build's own field and truth rows
    res = {"R_infield": compare(joint(pf.field, contest, pf.fp[R], sets["builder"], seed + 1),
                                joint(pf.field, contest, pf.fp[R], sets["topn"], seed + 1), cost)}
    # fresh truth noise + fresh field samples
    fp_t, _ = PF.truth_sims(pf.fp_raw[R], pf.field.pids, inp, s.beta, s.tau,
                            np.random.default_rng(seed + 9001))
    for name, cpt in (("R_fresh_off", False), ("R_fresh_cpt", True)):
        op = F.ownership_proxy(inp, pool, archetype=contest.archetype, cpt=cpt)
        fld = F.build_field(pool, inp, contest, n=None, ownership=op, seed=seed + 77,
                            threads=args.threads)
        if fld.pids != pf.field.pids:
            col = {p: j for j, p in enumerate(pf.field.pids)}
            fpx = np.stack([fp_t[:, col[p]] if p in col else np.zeros(len(fp_t), np.float32)
                            for p in fld.pids], 1)
        else:
            fpx = fp_t
        res[name] = compare(joint(fld, contest, fpx, sets["builder"], seed + 2),
                            joint(fld, contest, fpx, sets["topn"], seed + 2), cost)
        if name == "R_fresh_off":
            act = actual_points(conn, sl, pool)
            if act:
                a = np.array([act.get(p, 0.0) for p in fld.pids], np.float32).reshape(1, -1)
                rb = joint(fld, contest, a, sets["builder"], seed + 3)
                rt = joint(fld, contest, a, sets["topn"], seed + 3)
                res["actual"] = {k: {"builder": float(rb[k][0]), "topn": float(rt[k][0]),
                                     "diff": float(rb[k][0] - rt[k][0])}
                                 for k in ("pay", "any_top1", "top1", "any_cash")}
    out["eval"] = res
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarise(rows: list[dict]) -> dict:
    out = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        d = {"slates": len(rs), "same_sets": sum(r["same_sets"] for r in rs)}
        for ev in ("R_infield", "R_fresh_off", "R_fresh_cpt"):
            rr = [r["eval"][ev] for r in rs if ev in r["eval"]]
            if not rr:
                continue
            roi = np.array([x["roi_diff"] for x in rr], float)
            se_sim = math.sqrt(sum(x["roi_diff_se"] ** 2 for x in rr)) / len(rr)
            se_slate = float(roi.std(ddof=1) / math.sqrt(len(roi))) if len(roi) > 1 else float("nan")
            e = {"roi_diff": float(roi.mean()), "se_sim": se_sim, "se_slates": se_slate,
                 "slates_builder_better": int((roi > 0).sum())}
            for k in ("any_top1", "top1", "any_cash"):
                v = np.array([x[k]["diff"] for x in rr])
                e[k] = {"builder": float(np.mean([x[k]["builder"] for x in rr])),
                        "topn": float(np.mean([x[k]["topn"] for x in rr])),
                        "diff": float(v.mean()),
                        "se_sim": math.sqrt(sum(x[k]["se"] ** 2 for x in rr)) / len(rr),
                        "se_slates": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1
                        else float("nan")}
            e["builder_roi"] = float(np.mean([x["pay"]["builder"] / r["cost"] - 1
                                              for x, r in zip(rr, rs)]))
            e["topn_roi"] = float(np.mean([x["pay"]["topn"] / r["cost"] - 1
                                           for x, r in zip(rr, rs)]))
            d[ev] = e
        ac = [r for r in rs if "actual" in r["eval"]]
        if ac:
            v = np.array([r["eval"]["actual"]["pay"]["diff"] / r["cost"] for r in ac])
            d["actual"] = {"slates": len(ac), "roi_diff": float(v.mean()),
                           "se_slates": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1
                           else float("nan"),
                           "builder_payout": float(sum(r["eval"]["actual"]["pay"]["builder"]
                                                       for r in ac)),
                           "topn_payout": float(sum(r["eval"]["actual"]["pay"]["topn"]
                                                    for r in ac)),
                           "cost": float(sum(r["cost"] for r in ac))}
        out[str(n)] = d
    return out


def e2e(conn, slate_id: str) -> dict:
    """build_for_slate -> save -> load -> CSV -> page view model (scratch DB only)."""
    from courtside.proj.dfs import export as X
    from courtside.web import dfs_contest as WC
    sl = next(s for s in sd_slates(conn) if s["slate_id"] == slate_id)
    c = contest_for(conn, slate_id)
    s = PF.Settings(n=20, threads=2)
    now = datetime.fromisoformat(sl["run_at"].replace("Z", "+00:00"))
    t0 = time.time()
    pf = PF.build_for_slate(conn, slate_id, c.contest_id, s, run_id=sl["run_id"], now=now)
    pid = PF.save(conn, pf)
    d = PF.load(conn, pid)
    csv_text = X.to_csv([lu["slots"] for lu in d["lineups"]], style="id")
    lines = csv_text.strip().splitlines()
    by_id = F.prepare(conn, slate_id, sl["run_id"])["pool"].by_id()
    rules = pf.field.slots
    probs = []
    for lu in d["lineups"]:
        slots = lu["slots"]
        if [x["slot"] for x in slots] != ["CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL"]:
            probs.append(f"slot order {[x['slot'] for x in slots]}")
        sal = sum(int(x["salary"]) for x in slots)
        if sal > 50000:
            probs.append(f"salary {sal}")
        cpt = slots[0]
        p = by_id[str(cpt["player_id"])]
        if int(cpt["salary"]) != p.salary_by_slot.get("CPT"):
            probs.append(f"CPT salary {cpt['salary']} vs {p.salary_by_slot.get('CPT')}")
        if len({x["player_id"] for x in slots}) != 6:
            probs.append("duplicate player")
        if len({by_id[str(x['player_id'])].team for x in slots}) < 2:
            probs.append("one team")
    view = WC.build_page(conn, sl["day"], slate_id, now=now, portfolio_id=pid,
                         builds=WC.Builds())
    pv = view.get("portfolio") or {}
    return {"portfolio_id": pid, "seconds": round(time.time() - t0, 1), "status": d["status"],
            "lineups": len(d["lineups"]), "csv_header": lines[0], "csv_rows": len(lines) - 1,
            "csv_first": lines[1] if len(lines) > 1 else None, "problems": probs,
            "version": pf.info["version"], "relative": pf.summary.get("relative"),
            "view_status": pv.get("status"), "view_headline": pv.get("headline"),
            "view_lineup0": (pv.get("lineups") or [{}])[0].get("names"), "slots": rules,
            "view_is_showdown": view.get("slate", {}).get("is_showdown"),
            "view_gamma_choices": view.get("gamma_choices")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("e2e", "replay", "summary"))
    ap.add_argument("--slate", default=None)
    ap.add_argument("--slates", nargs="*", default=None)
    ap.add_argument("--ns", type=int, nargs="*", default=[1, 5, 20])
    ap.add_argument("--sims", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="d10_rows.jsonl")
    a = ap.parse_args(argv)
    conn = db.connect()
    if a.cmd == "e2e":
        print(json.dumps(e2e(conn, a.slate), indent=1, default=str))
        return 0
    path = Path(a.out)
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if a.cmd == "replay":
        done = {(r["slate_id"], r["n_requested"]) for r in rows}
        for sl in sd_slates(conn):
            if a.slates and sl["slate_id"] not in a.slates:
                continue
            for n in a.ns:
                if (sl["slate_id"], n) in done:
                    continue
                try:
                    r = run_one(conn, sl, n, a)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    r = {"slate_id": sl["slate_id"], "n": n, "error": f"{type(e).__name__}: {e}"}
                r["n_requested"] = n
                with path.open("a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                rows.append(r)
                ev = r.get("eval", {}).get("R_fresh_off", {})
                print(sl["slate_id"], n, r.get("error") or
                      f"roi diff {ev.get('roi_diff'):+.4f} ± {ev.get('roi_diff_se'):.4f} "
                      f"top1 {ev.get('any_top1', {}).get('diff', 0):+.4f} {r['seconds']}s",
                      flush=True)
    ok = [r for r in rows if "error" not in r]
    print(json.dumps(summarise(ok), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
