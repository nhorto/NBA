"""D9 selection stability: does a smoothed selection ladder pick the same portfolio on two sim
seeds more often, without losing held-out E[payout]? (IMPROVEMENTS.md D9; docs/research/d8-d9.md)

Per (slate, contest, N) and replicate r:
* the build's inputs exactly as `portfolio.build_portfolio` makes them (medium field proxy,
  scoring truth `truth_sims`), with the sim rows shuffled (seed r) into
  G (2,500: candidate objectives), A and B (two disjoint selection halves), C (held out);
* ONE candidate pool (`portfolio.candidates` on G), scored on A, B and C with the build's dupe
  model (expected dupes the field sample lacks as a Poisson count per sim);
* for each objective (off / flat / blend, `portfolio.selection_ladder`) the greedy picks N on A
  and N on B (same caps, gamma; no refills);
* stability = lineup overlap |pick_A & pick_B| / N and player-exposure overlap
  sum_p min(exp_A, exp_B) / 8; quality = joint E[payout] (REAL ladder) of each pick on C,
  paired vs off on the same C rows (SE over C rows).

The candidate pool is fixed within a replicate, so the overlap measures selection noise only
(two builds with different seeds also differ in candidates). Read-only DB (mode=ro).

    COURTSIDE_DATA=... PYTHONPATH=src:research python research/d9_stability.py \\
        --pairs 154575:196456028,155482:197024714 --n 20 --reps 4 --out d9_n20.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from courtside.proj import role_blend as RB
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_split_eval as PSE       # noqa: E402  (connect, load: run at lock, sims remap)

MODES = ("off", "flat", "blend")
G_ROWS = 2_500


def score(fld, fp, contest, lineups, n, seed, rng) -> PF._Pool:
    """`build_portfolio`'s score(): per-sim score / above / tied plus the Poisson dupes."""
    pairs = [lu.slot_pairs() for lu in lineups]
    ev = F.evaluate(fld, fp, contest, pairs, seed=seed, tiers=False, own_entries=n, per_sim=True)
    ps = ev.per_sim
    m_o = max(1, contest.field_size - n)
    _, rows, _ = F._cand_matrix(fld, pairs, fld.slots)
    cpt = fld.slots[0] == "CPT"
    cnt = {tuple(int(x) for x in r): int(c) for r, c in zip(F._canon(fld.lineups, cpt), fld.counts)}
    extra = []
    for i, row in enumerate(rows):
        d = float(ev.lineups[i]["dupes"]) * m_o / max(1, contest.field_size - 1)
        ins = cnt.get(tuple(int(x) for x in F._key(row, cpt)), 0) * m_o / max(1, fld.n) \
            if (row >= 0).all() else 0.0
        extra.append(max(0.0, d - ins))
    extra = np.array(extra)
    tied = np.rint(ps["tied"]).astype(np.int32)
    if (extra > 1e-4).any():
        tied += rng.poisson(extra[None, :], size=tied.shape).astype(np.int32)
    return PF._Pool(ps["score"], np.rint(ps["above"]).astype(np.int32), tied)


def exposure(inc, chosen):
    e = inc[chosen].sum(0).astype(float)
    return e / max(1, len(chosen))


def one(conn, ctx, contest, n, rep, threads, blended) -> dict:
    t0 = time.perf_counter()
    pool, inputs, sims = ctx["pool"], ctx["inputs"], ctx["sims"]
    players, rules = pool.players, pool.rules
    by_id = {p.dk_player_id: p for p in players}
    s = PF.Settings(n=n, seed=rep, n_sims=None, threads=threads)
    mu = {str(p): float(v) for p, v in zip(inputs["pid"], inputs["proj"])
          if v is not None and np.isfinite(v)}
    status = dict(zip(inputs["pid"].astype(str), inputs["status"].fillna("").astype(str)
                      .str.upper()))
    bans = {p for p in by_id if status.get(p) in F.OUT_STATUSES}
    mu = {p: v for p, v in mu.items() if p in by_id and v >= PF.MIN_PROJ}
    fld, op = PF.strength_field(players, inputs, contest, rules, "medium", s, seed=0)
    fp_raw, _ = F.align_sims(sims, fld.pids, None, 0)
    fp, _ = PF.truth_sims(fp_raw, fld.pids, inputs, s.beta, s.tau,
                          np.random.default_rng(rep + 7), blended=blended)
    S = fp.shape[0]
    perm = np.random.default_rng(1000 + rep).permutation(S)
    h = (S - G_ROWS) // 3
    rows = {"G": np.sort(perm[:G_ROWS]), "A": np.sort(perm[G_ROWS:G_ROWS + h]),
            "B": np.sort(perm[G_ROWS + h:G_ROWS + 2 * h]), "C": np.sort(perm[G_ROWS + 2 * h:])}
    fcol = {p: j for j, p in enumerate(fld.pids)}
    rng = np.random.default_rng(rep)
    cands, cinfo = PF.candidates(players, rules, mu, fp[rows["G"]], fcol, op.as_dict(), s,
                                 bans=bans, rng=rng)
    t_c = time.perf_counter() - t0
    pools = {k: score(fld, fp[rows[k]], contest, cands, n, rep, np.random.default_rng(rep + 50 + i))
             for i, k in enumerate(("A", "B", "C"))}
    pids = sorted({p for lu in cands for _, p in lu.slot_pairs()} | set(by_id))
    pidx = {p: i for i, p in enumerate(pids)}
    inc = PF._incidence(cands, pidx)
    cap_d, low_d = PF.exposure_limits(s, by_id)
    cap = np.array([cap_d.get(p, n) for p in pids])
    low = np.array([low_d.get(p, 0) for p in pids])
    lad = PF.Ladder(contest)
    C = pools["C"]
    out = {"rep": rep, "candidates": len(cands), "by_source": cinfo["by_source"],
           "rows": {k: int(len(v)) for k, v in rows.items()}, "modes": {}}
    tots = {}
    for mode in MODES:
        sl = PF.selection_ladder(lad, mode)
        picks = {}
        for half in ("A", "B"):
            ch, gi = PF.greedy(sl, PF.CandSet(pools[half], inc), n, gamma=s.gamma, cap=cap, low=low)
            picks[half] = ch
            pay, rank = PF.joint_payouts(lad, C.score[:, ch], C.above[:, ch], C.tied[:, ch])
            tots[mode, half] = pay.sum(1)
            out["modes"].setdefault(mode, {})[half] = {
                "n": len(ch), "stopped": gi["stopped"],
                "e_payout_C": float(pay.sum(1).mean()),
                "e_payout_C_se": float(pay.sum(1).std() / math.sqrt(len(pay))),
                "p_any_top1_C": float((rank <= lad.top1).any(1).mean()),
                "p_any_cash_C": float((rank <= lad.paid).any(1).mean())}
        a, b = picks["A"], picks["B"]
        ea, eb = exposure(inc, a), exposure(inc, b)
        out["modes"][mode]["overlap_lineups"] = len(set(a) & set(b)) / max(1, n)
        out["modes"][mode]["overlap_exposure"] = float(np.minimum(ea, eb).sum() / rules.n_slots)
    for mode in MODES[1:]:
        ds = [tots[mode, h] - tots["off", h] for h in ("A", "B")]
        d = 0.5 * (ds[0] + ds[1])
        base = 0.5 * (tots["off", "A"].mean() + tots["off", "B"].mean())
        out["modes"][mode]["paired_vs_off"] = {"mean": float(d.mean()),
                                               "se": float(d.std() / math.sqrt(len(d))),
                                               "pct": float(d.mean() / base) if base else None}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["seconds_candidates"] = round(t_c, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="slate:contest[:run],...")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    conn = PSE.connect()
    res = []
    for item in args.pairs.split(","):
        slate, cid, *rest = item.split(":")
        ctx = PSE.load(conn, slate, rest[0] if rest else None)
        contest = F.load_contest(conn, cid)
        n = min(args.n, contest.field_size - 1)
        blended = RB.run_blended(conn, ctx["run"])
        r = {"slate": slate, "contest": cid, "name": contest.name, "M": contest.field_size,
             "fee": contest.entry_fee, "n": n, "run": ctx["run"], "blended": blended, "reps": []}
        for rep in range(args.rep0, args.rep0 + args.reps):
            x = one(conn, ctx, contest, n, rep, args.threads, blended)
            r["reps"].append(x)
            print(json.dumps({"slate": slate, "contest": cid, "n": n, "rep": rep,
                              **{m: {k: v for k, v in x["modes"][m].items() if k not in ("A", "B")}
                                 for m in MODES},
                              "seconds": x["seconds"]}), flush=True)
            Path(args.out).write_text(json.dumps(res + [r], indent=1, default=float))
        res.append(r)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
