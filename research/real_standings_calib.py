"""Real DK standings vs our modelled field (docs/research/real-standings-2026-10-20.md).

The first real data: full standings of the 4 contests the owner played on the 2026-10-20 opener
(main slate 154575, 3 games). This script answers, on that one night:

1. **field**  Field realism at the ACTUAL box scores. For each real contest it builds our modelled
   field the way production does (`field.ownership_proxy` + `field.build_field` at the contest's
   archetype; strengths and new parameters as a grid), scores every field lineup with the players'
   actual DK points and compares with the real standings: cash line, median, top-1 % line,
   winning score, share above each real cash line, plus lineup STRUCTURE (salary used, public /
   our projection, unique lineups and dupes, players per team / game, share holding each
   top-owned player).
2. **own**    Ownership proxy per archetype vs real %Drafted (MAE, rank correlation, top-10 MAE)
   and the big misses with their inputs (DK FPPG, our projection, starter / status, salary).
3. **chances**  P(cash) / P(top 1 %) / E[payout] of all our lineups on the slate against the
   current and the recalibrated field, next to what really happened (real ranks).
4. **slates** Before/after on other slates (no standings): the slate page's cash line, P(cash),
   P(top 1 %) (`field.score_run(write=False)`) and the contest builder's chances by number of
   entries (`portfolio.build_portfolio` curve) for a double-up and a large GPP.
5. **hist**   RotoGuru history: the modelled double-up cash line at the actual points under the
   old and new settings (`field.backtest_cash_lines`), i.e. how far the recalibration moves it.

"old" = field-proxy-1.0 / own-proxy-1.0 patched back in (`field_1_0`), "new" = the module's
current defaults. The cash lineup vs Top-N #1 history is in var/reports/dfsbt_*.txt and
cash_calibration_*.txt (no rerun needed).

Read-only on the DB it is pointed at (COURTSIDE_DATA); run it on a scratch copy:

    COURTSIDE_DATA=<scratch var> PYTHONPATH=src python research/real_standings_calib.py field \\
        --slate 154575 --run c15398ddd87f --out var/reports/real_standings

Every number against the field is "simulated against a modelled field" (label proxy); the real
standings are real. One slate cannot fit a model: see the doc for what it can and cannot show.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import ids as IDS
from courtside.proj import store as SO
from courtside.proj.dfs import field as F
from courtside.proj.dfs import standings as ST

PAID_DU = 0.435            # DK double-ups pay round(0.435 * entries) (125/287, 500/1149, 200/459)
TOP_OWNED = 12


# ============================================================================== loading

def run_row(conn, run_id: str) -> dict:
    r = conn.execute("SELECT * FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
    if r is None:
        raise SystemExit(f"unknown run {run_id}")
    return dict(r)


def setup(conn, slate: str, run_id: str, *, as_of: str | None = "run") -> dict:
    """Pool and ownership inputs as of the run (the capture the run saw), its sims, and the
    players' actual DK points (grades of the run; DK FPTS of the standings for the rest)."""
    run = run_row(conn, run_id)
    ts = run["created_at"] if as_of == "run" else as_of
    pool, inp, info = F.slate_inputs(conn, slate, run_id, as_of=ts)
    sims = SO.load_sims(info["sim_path"]) if info.get("sim_path") and \
        Path(info["sim_path"]).exists() else None
    actual = actual_points(conn, slate, run_id, pool)
    return {"slate": str(slate), "run": run, "pool": pool, "inputs": inp, "info": info,
            "sims": sims, "actual": actual}


def actual_points(conn, slate: str, run_id: str, pool) -> dict[str, float]:
    """{DK id: actual DK points}: the run's player grades (box scores), else the standings' FPTS
    (max over the slate's contests), else 0 (did not play / not drafted)."""
    dk2nba = IDS.nba_ids(conn, IDS.DK, [p.dk_player_id for p in pool.players])
    g = dict(conn.execute("SELECT player_id, actual_dk FROM player_projection_grades WHERE "
                          "run_id=?", (run_id,)).fetchall())
    fp = dict(conn.execute(
        "SELECT o.player_id, MAX(o.fpts) FROM dfs_contest_ownership o JOIN dfs_contests c ON "
        "c.contest_id=o.contest_id WHERE c.slate_id=? GROUP BY o.player_id",
        (str(slate),)).fetchall())
    out = {}
    for p in pool.players:
        n = dk2nba.get(p.dk_player_id)
        v = g.get(n) if n is not None else None
        if v is None and n is not None:
            v = fp.get(n)
        out[p.dk_player_id] = float(v) if v is not None else 0.0
    return out


def real_contests(conn, slate: str) -> list[dict]:
    """Uploaded contests of the slate: a field.Contest (payout table; double-ups without one get
    round(0.435 M) paid places at 2x the fee, labelled 'inferred'), real scores (desc) and lineups
    (DK id lists), real ownership {DK id: fraction}."""
    out = []
    cids = [r[0] for r in conn.execute(
        "SELECT DISTINCT r.contest_id FROM dfs_contest_results r JOIN dfs_contests c ON "
        "c.contest_id=r.contest_id WHERE c.slate_id=?", (str(slate),))]
    hist = {}
    for r in conn.execute("SELECT contest_id, entry_fee, field_size, note FROM dfs_entries "
                          "WHERE slate_id=?", (str(slate),)):
        hist.setdefault(r["contest_id"], dict(r))
    nba2dk = {v: k for k, v in IDS.nba_ids(conn, IDS.DK).items()}
    for cid in cids:
        c = F.load_contest(conn, cid)
        rows = conn.execute("SELECT rank, points, lineup_text FROM dfs_contest_results WHERE "
                            "contest_id=? ORDER BY rank", (cid,)).fetchall()
        m = len(rows)
        h = hist.get(cid) or {}
        name = (c.name if c and c.name else (h.get("note") or "").replace("history: ", "")
                .split(" (")[0]) or cid
        fee = (c.entry_fee if c and c.entry_fee else h.get("entry_fee")) or 0.0
        source = "captured"
        if c is None or not c.payouts:
            if "double up" in name.lower():
                paid = int(round(PAID_DU * m))
                c = F.Contest(cid, name, float(fee), m,
                              [{"min_pos": 1, "max_pos": paid, "payout": 2 * float(fee)}],
                              "cash", "inferred")
                source = "inferred"
            else:
                raise SystemExit(f"contest {cid}: no payout table")
        else:
            c = F.Contest(c.contest_id, name, float(fee), m, c.payouts, c.archetype, c.source)
        own = {}
        for pid, pct in conn.execute("SELECT player_id, pct_drafted FROM dfs_contest_ownership "
                                     "WHERE contest_id=?", (cid,)):
            dk = nba2dk.get(int(pid))
            if dk is not None:
                own[str(dk)] = own.get(str(dk), 0.0) + float(pct) / 100.0
        out.append({"contest": c, "source": source,
                    "points": np.array([r["points"] for r in rows], float),
                    "rank": np.array([r["rank"] for r in rows], int),
                    "lineup_text": [r["lineup_text"] for r in rows], "own": own})
    out.sort(key=lambda d: d["contest"].field_size)
    return out


def lineup_ids(texts, pool) -> tuple[np.ndarray, int]:
    """(entries x 8) DK ids (object array) parsed from the standings' Lineup cells by name;
    rows with an unknown name are dropped (count returned)."""
    by_name: dict[str, str] = {}
    for p in pool.players:
        by_name.setdefault(IDS.norm_name(p.name), p.dk_player_id)
    rows, bad = [], 0
    for t in texts:
        pr = ST.parse_lineup(t)
        ids = [by_name.get(IDS.norm_name(n)) for _, n in (pr or [])]
        if not pr or len(ids) != 8 or any(i is None for i in ids):
            bad += 1
            rows.append(None)
            continue
        rows.append(ids)
    return rows, bad


# ============================================================================== structure

def structure(L: np.ndarray, w: np.ndarray | None, tab: pd.DataFrame, top: list[str]) -> dict:
    """Lineup-structure summary of a set of lineups (rows of pool indices into `tab`), each
    weighted `w` (entries). `tab` columns: salary, pub, proj, team, game, actual."""
    w = np.ones(len(L)) if w is None else np.asarray(w, float)
    W = w.sum()
    sal = tab["salary"].to_numpy()[L].sum(1)
    pub = tab["pub"].to_numpy()[L].sum(1)
    ours = np.nan_to_num(tab["proj"].to_numpy()[L]).sum(1)
    act = tab["actual"].to_numpy()[L].sum(1)
    team = tab["team_code"].to_numpy()[L]
    game = tab["game_code"].to_numpy()[L]

    def maxcount(codes):
        s = np.sort(codes, axis=1)
        best = np.ones(len(s), int)
        run = np.ones(len(s), int)
        for j in range(1, s.shape[1]):
            same = s[:, j] == s[:, j - 1]
            run = np.where(same, run + 1, 1)
            best = np.maximum(best, run)
        return best

    mt, mg = maxcount(team), maxcount(game)
    key = [tuple(sorted(r)) for r in L.tolist()]
    cnt: dict = {}
    for k, x in zip(key, w):
        cnt[k] = cnt.get(k, 0.0) + x
    dup_entries = sum(v for v in cnt.values() if v > 1.0001)
    pos = {p: i for i, p in enumerate(tab["pid"])}
    hold = {}
    for p in top:
        i = pos.get(p)
        hold[p] = float((w * (L == i).any(1)).sum() / W) if i is not None else float("nan")
    q = lambda x, p: float(np.quantile(np.repeat(x, np.maximum(1, np.rint(w).astype(int))), p))  # noqa
    return {"entries": float(W), "unique": len(cnt), "unique_share": len(cnt) / W,
            "dupe_entry_share": dup_entries / W, "max_dupes": max(cnt.values()),
            "salary_mean": float((w * sal).sum() / W),
            "salary_ge_49500": float((w * (sal >= 49500)).sum() / W),
            "salary_ge_49000": float((w * (sal >= 49000)).sum() / W),
            "pub_mean": float((w * pub).sum() / W), "ours_mean": float((w * ours).sum() / W),
            "max_team_mean": float((w * mt).sum() / W),
            "team4plus": float((w * (mt >= 4)).sum() / W),
            "max_game_mean": float((w * mg).sum() / W),
            "actual_mean": float((w * act).sum() / W), "hold": hold}


def lines(scores: np.ndarray, w: np.ndarray | None, contest: F.Contest,
          real_lines: dict[str, float]) -> dict:
    """Score lines of a (weighted) set: cash line (score at the paid fraction of the field),
    median, top-1 % line, max, and the share at or above each real cash line."""
    w = np.ones(len(scores)) if w is None else np.asarray(w, float)
    o = np.argsort(-scores)
    s, ww = scores[o], w[o]
    cw = np.cumsum(ww) / ww.sum()

    def at(frac):
        return float(s[min(len(s) - 1, int(np.searchsorted(cw, frac)))])

    out = {"cash_line": at(contest.paid / contest.field_size), "median": at(0.5),
           "top1": at(0.01), "top10": at(0.10), "max": float(s[0]),
           "mean": float((s * ww).sum() / ww.sum())}
    for k, v in real_lines.items():
        out[f"share_ge_{k}"] = float(ww[s >= v - 1e-9].sum() / ww.sum())
    return out


def pool_table(ctx: dict, op: F.OwnershipProxy) -> pd.DataFrame:
    tab = op.inputs[["pid", "name", "team", "game", "salary", "proj", "fppg", "status",
                     "starter", "pub_proj"]].rename(columns={"pub_proj": "pub"}).copy()
    tab["actual"] = tab["pid"].map(ctx["actual"]).fillna(0.0)
    tab["team_code"] = pd.factorize(tab["team"].fillna("?"))[0]
    tab["game_code"] = pd.factorize(tab["game"].fillna("?"))[0]
    return tab.reset_index(drop=True)


def real_matrix(rc: dict, tab: pd.DataFrame, pool) -> tuple[np.ndarray, np.ndarray, int]:
    """(entries x 8) pool indices of the real lineups, their real points, dropped count."""
    rows, bad = lineup_ids(rc["lineup_text"], pool)
    pos = {p: i for i, p in enumerate(tab["pid"])}
    L, pts = [], []
    for r, p in zip(rows, rc["points"]):
        if r is None or any(x not in pos for x in r):
            continue
        L.append([pos[x] for x in r])
        pts.append(p)
    return np.array(L, int), np.array(pts, float), bad + (len(rows) - bad - len(L))


# ============================================================================== field at actual

def draws(Lf: np.ndarray, wf: np.ndarray, sc: np.ndarray, contest: F.Contest, m: int,
          reps: int = 40, seed: int = 0) -> dict:
    """The field as a real contest would see it: `m` entries drawn from the sampled field
    (weights = counts), `reps` times. Uniqueness / dupes at the real entry count (comparable
    with the standings) and the spread of the cash line across same-night contests."""
    rng = np.random.default_rng(seed)
    p = wf / wf.sum()
    key = np.unique(np.sort(Lf, axis=1), axis=0, return_inverse=True)[1].ravel()
    uq, dup, mx, cl = [], [], [], []
    paid = max(1, int(round(contest.paid / contest.field_size * m)))
    for _ in range(reps):
        idx = rng.choice(len(Lf), m, p=p)
        _, cnt = np.unique(key[idx], return_counts=True)
        uq.append(len(cnt) / m)
        dup.append(cnt[cnt > 1].sum() / m)
        mx.append(cnt.max() / m)
        cl.append(float(np.sort(sc[idx])[::-1][paid - 1]))
    return {"unique_share": float(np.mean(uq)), "dupe_entry_share": float(np.mean(dup)),
            "max_dupe_share": float(np.mean(mx)), "cash_line_mean": float(np.mean(cl)),
            "cash_line_sd": float(np.std(cl))}


def field_lineups(f: F.Field, tab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Field lineups as rows of `tab` indices and their weights (sample counts)."""
    pos = {p: i for i, p in enumerate(tab["pid"])}
    m = np.array([pos[p] for p in f.pids])
    return m[f.lineups], f.counts.astype(float)


def build(ctx, contest, *, archetype=None, mix=None, w_ours=None, params=None, seed=0,
          n=None, n_solves=F.N_SOLVES, **kw):
    """Ownership proxy + field for a contest (production path, overridable knobs)."""
    arch = archetype or contest.archetype
    kw.pop("name", None)
    P = dict(params or {})
    if w_ours is not None:
        P["w_ours"] = w_ours
    patch = {k: kw.pop(k) for k in list(kw) if k in F.ARCHETYPES[arch]}
    floors = kw.pop("floors", None)
    saved = dict(F.ARCHETYPES[arch])
    saved_fl = (F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT)
    F.ARCHETYPES[arch].update(patch)
    if floors:
        F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT = floors
    try:
        lp, lv = F.load_params()
        op = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=arch, params={**lp, **P},
                               version=lv, **{k: v for k, v in kw.items()
                                              if k in ("row_cap", "spend")})
        fkw = {k: v for k, v in kw.items() if k not in ("row_cap", "spend")}
        f = F.build_field(ctx["pool"], ctx["inputs"], contest, n=n, mix=mix, ownership=op,
                          archetype=arch, n_solves=n_solves, seed=seed, **fkw)
    finally:
        F.ARCHETYPES[arch].clear()
        F.ARCHETYPES[arch].update(saved)
        F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT = saved_fl
    return op, f


def own_metrics(pred: dict[str, float], real: dict[str, float], top: int = 10) -> dict:
    keys = sorted(set(pred) | set(real))
    p = np.array([pred.get(k, 0.0) for k in keys])
    y = np.array([real.get(k, 0.0) for k in keys])
    keep = (p > 0.001) | (y > 0.001)
    p, y = p[keep], y[keep]
    o = np.argsort(-y)[:top]
    return {"mae_pp": float(np.mean(np.abs(p - y)) * 100),
            "corr": float(np.corrcoef(p, y)[0, 1]) if len(p) > 2 else float("nan"),
            "spearman": F._spearman(p, y),
            "top_mae_pp": float(np.mean(np.abs(p[o] - y[o])) * 100),
            "top_bias_pp": float(np.mean(p[o] - y[o]) * 100),
            "sum_top": float(y[o].sum()), "sum_top_pred": float(p[o].sum()), "n": int(len(p))}


# ============================================================================== commands

def cmd_own(args, conn) -> dict:
    ctx = setup(conn, args.slate, args.run)
    rcs = real_contests(conn, args.slate)
    out = {"run": args.run, "contests": []}
    tabs = []
    for rc in rcs:
        c = rc["contest"]
        row = {"contest": c.contest_id, "name": c.name, "M": c.field_size,
               "archetype": c.archetype}
        for arch in ("cash", "small_gpp", "large_gpp"):
            op = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=arch)
            row[arch] = own_metrics(op.as_dict(), rc["own"])
        out["contests"].append(row)
        op = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype=c.archetype)
        t = op.table()
        t["real"] = t["pid"].map(rc["own"]).fillna(0.0)
        t["contest"] = c.contest_id
        t["actual"] = t["pid"].map(ctx["actual"])
        tabs.append(t)
    big = pd.concat(tabs)
    big = big[(big["own"] > 0.03) | (big["real"] > 0.03)]
    cols = ["contest", "name", "team", "salary", "fppg", "proj", "pub_proj", "pub_value",
            "status", "starter", "replacement", "proj_min", "recent_min", "own", "real",
            "actual"]
    out["table"] = big[cols].sort_values(["contest", "real"], ascending=[True, False]) \
        .to_dict("records")
    return out


def cmd_field(args, conn) -> dict:
    """Grid of field settings per real contest, scored at the actual box scores."""
    ctx = setup(conn, args.slate, args.run)
    rcs = real_contests(conn, args.slate)
    res = {"run": args.run, "slate": args.slate, "real": [], "grid": []}
    op0 = F.ownership_proxy(ctx["inputs"], ctx["pool"], archetype="large_gpp")
    tab = pool_table(ctx, op0)
    rl = {}
    for rc in rcs:
        c = rc["contest"]
        L, pts, dropped = real_matrix(rc, tab, ctx["pool"])
        top = sorted(rc["own"], key=lambda k: -rc["own"][k])[:TOP_OWNED]
        rl[c.contest_id] = float(np.sort(rc["points"])[::-1][c.paid - 1])
        rc["L"], rc["top"] = L, top
        # the real lineups re-scored at our actuals (checks the actual-points join)
        resc = tab["actual"].to_numpy()[L].sum(1)
        res["real"].append({
            "contest": c.contest_id, "name": c.name, "M": c.field_size, "paid": c.paid,
            "fee": c.entry_fee, "archetype": c.archetype, "payouts": rc["source"],
            "parsed": int(len(L)), "dropped": int(dropped),
            "rescore_mae": float(np.mean(np.abs(resc - pts))),
            "lines": lines(rc["points"], None, c, {}),
            "structure": structure(L, None, tab, top),
            "at_m": draws(L, np.ones(len(L)), pts, c, min(c.field_size, 3000)),
            "own_top": {tab.loc[tab.pid == p, "name"].iloc[0]: rc["own"][p] for p in top
                        if (tab.pid == p).any()}})
    for rc in rcs:
        c = rc["contest"]
        if args.contests and c.contest_id not in args.contests.split(","):
            continue
        for setting in grid(args, c):
            t0 = time.perf_counter()
            reps = []
            for sd in range(args.seeds):
                op, f = build(ctx, c, seed=sd, n=args.n, **setting)
                Lf, wf = field_lineups(f, tab)
                sc = tab["actual"].to_numpy()[Lf].sum(1)
                st = structure(Lf, wf, tab, rc["top"])
                st["at_m"] = draws(Lf, wf, sc, c, min(c.field_size, 3000), seed=sd)
                reps.append((lines(sc, wf, c, rl), st,
                             own_metrics(op.as_dict(), rc["own"]),
                             own_metrics(dict(zip(f.pids, f.realized_ownership())), rc["own"])))
            row = {"contest": c.contest_id, "setting": {k: v for k, v in setting.items()},
                   "lines": _avg([r[0] for r in reps]), "structure": _avg([r[1] for r in reps]),
                   "own_proxy": _avg([r[2] for r in reps]),
                   "own_field": _avg([r[3] for r in reps]),
                   "seconds": round(time.perf_counter() - t0, 1)}
            res["grid"].append(row)
            print(json.dumps({"contest": row["contest"], **row["setting"],
                              "cash": round(row["lines"]["cash_line"], 2),
                              "real": rl[c.contest_id],
                              "med": round(row["lines"]["median"], 1),
                              "top1": round(row["lines"]["top1"], 1),
                              "uniq_m": round(row["structure"]["at_m"]["unique_share"], 3),
                              "dupe_m": round(row["structure"]["at_m"]["dupe_entry_share"], 3),
                              "maxd_m": round(row["structure"]["at_m"]["max_dupe_share"], 4),
                              "cl_sd": round(row["structure"]["at_m"]["cash_line_sd"], 2),
                              "sal": round(row["structure"]["salary_mean"]),
                              "pub": round(row["structure"]["pub_mean"], 1),
                              "ours": round(row["structure"]["ours_mean"], 1),
                              "top_mae": round(row["own_field"]["top_mae_pp"], 1),
                              "s": row["seconds"]}), flush=True)
    return res


# field-proxy-1.0 / own-proxy-1.0 settings (before this calibration), to compare against the
# live defaults: archetype params, expected spend 0.99, spend floors 0.98 / 0.96 / 0.92 (0.7)
FIELD_1_0 = {"cash": {"conc": 1.25, "noise": 0.10, "full_spend": 0.90, "mix": 0.40},
             "small_gpp": {"conc": 1.00, "noise": 0.15, "full_spend": 0.82, "mix": 0.40},
             "large_gpp": {"conc": 0.90, "noise": 0.18, "full_spend": 0.78, "mix": 0.40}}
GLOBAL_1_0 = {"spend": 0.99, "floors": (0.98, 0.96, 0.92, 0.7)}


def setting_for(name: str, arch: str) -> dict:
    """'old' = field-proxy-1.0 (patched back), 'new' = the module's current defaults."""
    if name == "old":
        return {"archetype": arch, "mix": 0.40, "w_ours": 0.5, **FIELD_1_0[arch],
                **GLOBAL_1_0}
    return {"archetype": arch}


def our_lineups(conn, slate: str) -> list[dict]:
    """Unique cash / Top N lineups of the slate's ok non-swap runs (first occurrence), with the
    runs and ranks they appeared in, and the owner's entries."""
    rows = conn.execute(
        "SELECT l.id, l.run_id, l.mode, l.rank, l.slots_json, l.p_cash, r.kind, r.created_at "
        "FROM dfs_lineups l JOIN dfs_runs r ON r.run_id=l.run_id WHERE r.slate_id=? AND r.ok=1 "
        "AND l.mode IN ('cash','topn') AND r.kind!='swap' ORDER BY r.created_at, l.mode, l.rank",
        (str(slate),)).fetchall()
    ent = {}
    for r in conn.execute("SELECT lineup_id, contest_id, points, rank FROM dfs_entries WHERE "
                          "slate_id=?", (str(slate),)):
        ent.setdefault(r["lineup_id"], []).append(r["contest_id"])
    out: dict = {}
    for r in rows:
        pairs = [(d["slot"], str(d["player_id"])) for d in json.loads(r["slots_json"])]
        key = tuple(sorted(p for _, p in pairs))
        d = out.setdefault(key, {"id": r["id"], "pairs": pairs, "tags": [], "entered": [],
                                 "stored_p_cash": r["p_cash"]})
        d["tags"].append(f"{r['run_id']}:{r['kind']}:{r['mode']}{r['rank']}")
        d["entered"] += ent.get(r["id"], [])
    return list(out.values())


def cmd_chances(args, conn) -> dict:
    """P(cash) / P(top 1 %) / E[payout] of our lineups vs each real contest, old vs new field,
    next to the real outcome, plus the night's PIT of (lineup - cash line)."""
    from courtside.proj.dfs import portfolio as PF
    ctx = setup(conn, args.slate, args.run)
    rcs = real_contests(conn, args.slate)
    lus = our_lineups(conn, args.slate)
    act = np.array([sum(ctx["actual"].get(p, 0.0) for _, p in d["pairs"]) for d in lus])
    res = {"run": args.run, "n_lineups": len(lus), "contests": []}
    for rc in rcs:
        c = rc["contest"]
        if args.contests and c.contest_id not in args.contests.split(","):
            continue
        pts = np.sort(rc["points"])[::-1]
        line = float(pts[c.paid - 1])
        top1 = float(pts[max(1, int(math.ceil(0.01 * c.field_size))) - 1])
        rank = np.array([int((pts > a + 1e-9).sum()) + 1 for a in act])
        lad = c.ladder()
        real_pay = np.array([lad[min(r, c.field_size)] if r <= c.field_size else 0.0
                             for r in rank])
        crow = {"contest": c.contest_id, "name": c.name, "M": c.field_size, "fee": c.entry_fee,
                "real_cash_line": line, "real_top1_line": top1,
                "real_cashed": float(np.mean(act >= line - 1e-9)),
                "real_top1": float(np.mean(act >= top1 - 1e-9)),
                "real_payout_mean": float(real_pay.mean()), "settings": {}}
        for name in args.settings.split(","):
            sv = []
            for sd in range(args.seeds):
                st = setting_for(name, "cash" if c.archetype == "cash" else
                                 ("small_gpp" if c.field_size <= F.SMALL_GPP_MAX else "large_gpp"))
                op, f = build(ctx, c, seed=sd, n=args.n, **st)
                fp, _ = F.align_sims(ctx["sims"], f.pids, args.sims, sd)
                fp, _ = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                                      np.random.default_rng(sd + 7))
                ev = F.evaluate(f, fp, c, [d["pairs"] for d in lus], seed=sd, tiers=False,
                                per_sim=True)
                sc = (ev.per_sim["score"].astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
                dsim = sc - ev.cash_line[:, None]
                dreal = act - line
                pit = (dsim < dreal[None, :]).mean(0)
                sv.append({"p_cash": np.array([x["p_cash"] for x in ev.lineups]),
                           "p_top1": np.array([x["p_top1"] for x in ev.lineups]),
                           "e_payout": np.array([x["e_payout"] for x in ev.lineups]),
                           "pit": pit, "cash_line_sim": ev.cash_line_summary(),
                           "lineup_sim_mean": sc.mean(0), "dsim_mean": dsim.mean(0)})
            agg = {k: np.mean([v[k] for v in sv], axis=0) for k in
                   ("p_cash", "p_top1", "e_payout", "pit", "lineup_sim_mean", "dsim_mean")}
            crow["settings"][name] = {
                "p_cash_mean": float(agg["p_cash"].mean()),
                "p_top1_mean": float(agg["p_top1"].mean()),
                "e_payout_mean": float(agg["e_payout"].mean()),
                "pit_median": float(np.median(agg["pit"])),
                "pit_mean": float(agg["pit"].mean()),
                "dsim_mean": float(agg["dsim_mean"].mean()),
                "cash_line_sim": _avg([v["cash_line_sim"] for v in sv]),
                "per_lineup": [{"id": d["id"], "tags": d["tags"][:3], "entered": d["entered"],
                                "actual": float(a), "rank": int(r), "p_cash": float(pc),
                                "p_top1": float(pt), "e_payout": float(ep), "pit": float(pi),
                                "sim_mean": float(sm)}
                               for d, a, r, pc, pt, ep, pi, sm in zip(
                                   lus, act, rank, agg["p_cash"], agg["p_top1"],
                                   agg["e_payout"], agg["pit"], agg["lineup_sim_mean"])]}
            print(json.dumps({"contest": c.contest_id, "setting": name,
                              "p_cash": round(float(agg["p_cash"].mean()), 3),
                              "real_cashed": round(crow["real_cashed"], 3),
                              "pit_med": round(float(np.median(agg["pit"])), 3),
                              "sim_cash_p50": round(crow["settings"][name]["cash_line_sim"]
                                                    ["cash_p50"], 1),
                              "real_line": line}), flush=True)
        res["contests"].append(crow)
    return res


@contextlib.contextmanager
def field_1_0():
    """Run the production code with the field-proxy-1.0 / own-proxy-1.0 settings."""
    saved = {a: dict(v) for a, v in F.ARCHETYPES.items()}
    sv = (F.SPEND, F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT)
    for a, v in FIELD_1_0.items():
        F.ARCHETYPES[a].update(v)
    F.SPEND = GLOBAL_1_0["spend"]
    F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT = GLOBAL_1_0["floors"]
    try:
        yield
    finally:
        for a, v in saved.items():
            F.ARCHETYPES[a].clear()
            F.ARCHETYPES[a].update(v)
        F.SPEND, F.FLOOR_FULL, F.FLOOR_MOST, F.FLOOR_REST, F.FLOOR_SPLIT = sv


def _patched(name: str):
    return field_1_0() if name == "old" else contextlib.nullcontext()


def cmd_slates(args, conn) -> dict:
    """Before / after on slates without standings: the slate-page numbers of the run's lineups
    (`field.score_run(write=False)`: cash line, P(cash), P(top 1 %), dupes) and the contest
    builder's chances by number of entries (`portfolio.build_for_slate`, curve) for the given
    contests. Nothing is written."""
    from datetime import datetime
    from courtside.proj.dfs import portfolio as PF
    out = {"slates": []}
    for item in args.runs.split(","):
        slate, run = item.split(":")
        row = {"slate": slate, "run": run, "score_run": {}, "curves": {}}
        created = run_row(conn, run)["created_at"]
        for name in args.settings.split(","):
            with _patched(name):
                r = F.score_run(conn, run, slate_id=slate, write=False, n_sims=args.sims,
                                as_of=created)
            pl = r.get("per_lineup") or []
            row["score_run"][name] = {
                "cash": {k: r["cash"][k] for k in ("contest_id", "name", "field_size",
                                                   "cash_line")},
                "gpp": {k: r["gpp"][k] for k in ("contest_id", "name", "field_size",
                                                 "cash_line")},
                "cash_field": {k: r["cash"]["field"].get(k) for k in ("unique", "sampled",
                                                                       "salary_use")},
                "lineups": [{k: x[k] for k in ("mode", "rank", "p_cash", "p_top1", "dupes")}
                            for x in pl]}
            print(json.dumps({"slate": slate, "setting": name,
                              "cash_p50": round(r["cash"]["cash_line"]["cash_p50"], 1),
                              "p_cash_mean": round(float(np.mean([x["p_cash"] for x in pl])), 3)
                              if pl else None,
                              "p_top1_mean": round(float(np.mean([x["p_top1"] for x in pl])), 4)
                              if pl else None}), flush=True)
        for cid in (args.curve_contests.split(",") if args.curve_contests and slate ==
                    args.curve_slate else []):
            for name in args.settings.split(","):
                s = PF.Settings(n=1, n_sims=args.sims, seed=0, threads=2)
                r = conn.execute("SELECT max_per_user FROM dfs_contests WHERE contest_id=?",
                                 (cid,)).fetchone()
                s.n = int(min(r[0] or 1, args.curve_max))
                s.curve_to = s.n
                t0 = time.perf_counter()
                ctxp = F.prepare(conn, slate, run, n_sims=s.n_sims, as_of=created)
                contest = F.load_contest(conn, cid)
                with _patched(name):          # = portfolio.build_for_slate at the run's capture
                    pf = PF.build_portfolio(ctxp["pool"], ctxp["inputs"], ctxp["sims"], contest,
                                            s, now=datetime.fromisoformat(
                                                created.replace("Z", "+00:00")),
                                            slate_id=slate, run_id=run)
                cv = pf.summary.get("curve") or {}
                row["curves"].setdefault(cid, {})[name] = {
                    "contest": pf.summary.get("contest"),
                    "points": [{k: p.get(k) for k in ("entries", "p_any_cash", "p_any_top1",
                                                      "p_any_top01", "p_first", "cost")}
                               for p in cv.get("points", [])],
                    "relative": pf.summary.get("relative"),
                    "seconds": round(time.perf_counter() - t0, 1)}
                print(json.dumps({"contest": cid, "setting": name,
                                  "curve": [(p["entries"], round(p["p_any_cash"], 3),
                                             round(p["p_any_top1"], 4))
                                            for p in cv.get("points", [])],
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
        out["slates"].append(row)
    return out


def cmd_hist(args, conn) -> dict:
    """RotoGuru history (no standings): the double-up proxy cash line at the ACTUAL points under
    the old and new field settings (`field.backtest_cash_lines`: public = season-to-date DK
    average, starters known, availability known), i.e. how much the recalibration moves the
    modelled cash line over many slates. Says nothing about real lines by itself."""
    from courtside.proj.dfs import optimizer as O
    dates = O.hist_dates(conn)
    dates = [d for d in dates if d >= args.hist_from]
    if args.hist and len(dates) > args.hist:
        idx = np.linspace(0, len(dates) - 1, args.hist).round().astype(int)
        dates = [dates[i] for i in sorted(set(idx))]
    out = {"dates": len(dates), "rows": {}}
    for name in args.settings.split(","):
        with _patched(name):
            df = F.backtest_cash_lines(conn, dates, n=3000, paid_frac=0.435, field_size=1000,
                                       n_solves=100, seed=0)
        out["rows"][name] = df.to_dict("records")
        print(json.dumps({"setting": name, "n": len(df),
                          "cash_line_mean": round(float(df["cash_line"].mean()), 2),
                          "top1_mean": round(float(df["top1_line"].mean()), 2),
                          "field_mean": round(float(df["field_mean"].mean()), 2),
                          "optimal_mean": round(float(df["optimal"].mean()), 2),
                          "share_ge_98": round(float(df["share_ge_98"].mean()), 3)}), flush=True)
    if {"old", "new"} <= set(out["rows"]):
        a = pd.DataFrame(out["rows"]["old"]).set_index("date")
        b = pd.DataFrame(out["rows"]["new"]).set_index("date")
        j = a.join(b, lsuffix="_old", rsuffix="_new", how="inner")
        for k in ("cash_line", "top1_line", "field_mean"):
            d = j[f"{k}_new"] - j[f"{k}_old"]
            out[f"diff_{k}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) /
                                                                      math.sqrt(len(d))),
                                "n": int(len(d))}
        print(json.dumps({k: v for k, v in out.items() if k.startswith("diff_")}), flush=True)
    return out


def _avg(ds: list[dict]) -> dict:
    out = {}
    for k, v in ds[0].items():
        if isinstance(v, dict):
            out[k] = _avg([d[k] for d in ds])
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = [d[k] for d in ds]
            out[k] = float(np.mean(vals))
            if len(vals) > 1:
                out[k + "_sd"] = float(np.std(vals, ddof=1))
        else:
            out[k] = v
    return out


def grid(args, c: F.Contest) -> list[dict]:
    g = json.loads(args.grid) if args.grid else None
    if g:
        return g
    if args.settings_grid:
        arch = "cash" if c.archetype == "cash" else (
            "small_gpp" if c.field_size <= F.SMALL_GPP_MAX else "large_gpp")
        return [{"name": nm, **setting_for(nm, arch)} for nm in args.settings.split(",")]
    arch = "cash" if c.archetype == "cash" else "large_gpp"
    out = [{"archetype": arch, "mix": mix, "w_ours": w}
           for mix in (0.2, 0.4, 0.7) for w in (0.25, 0.5, 0.75)]
    if arch == "cash":
        out += [{"archetype": arch, "mix": mix, "w_ours": 0.5, "noise": nz}
                for mix in (0.4, 0.7) for nz in (0.05, 0.02)]
        out += [{"archetype": arch, "mix": 0.4, "w_ours": 0.5, "conc": 2.0}]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["own", "field", "chances", "slates", "hist"])
    ap.add_argument("--slate", default="154575")
    ap.add_argument("--run", default="c15398ddd87f")
    ap.add_argument("--n", type=int, default=None, help="field sample (default production's)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--grid", default=None, help="JSON list of settings")
    ap.add_argument("--contests", default=None, help="comma-separated contest ids (default all)")
    ap.add_argument("--settings", default="old,new", help="chances: field settings to compare")
    ap.add_argument("--settings-grid", action="store_true",
                    help="field: use the named --settings (old / new) as the grid")
    ap.add_argument("--sims", type=int, default=10_000, help="chances: sims subsample")
    ap.add_argument("--runs", default="155078:d514cb25b711,155316:9418d9ee3778",
                    help="slates: slate:run pairs")
    ap.add_argument("--curve-slate", default="155078")
    ap.add_argument("--curve-contests", default="196981149,196981122",
                    help="slates: contests for the chances-by-entries table")
    ap.add_argument("--curve-max", type=int, default=20, help="slates: largest entry count")
    ap.add_argument("--hist", type=int, default=60, help="hist: number of RotoGuru dates")
    ap.add_argument("--hist-from", default="2023-10-01", help="hist: first date")
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args(argv)
    conn = db.connect()
    fn = {"own": cmd_own, "field": cmd_field, "chances": cmd_chances,
          "slates": cmd_slates, "hist": cmd_hist}[args.cmd]
    res = fn(args, conn)
    txt = json.dumps(res, indent=1, default=float)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    else:
        print(txt[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
