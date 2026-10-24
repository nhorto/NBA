"""X factor v2 (top-1 % leverage) vs v1: evaluation (IMPROVEMENTS.md D3; docs/research/xfactor-v2.md).

Commands (all read-only on the DB they are pointed at; run on scratch copies):

* **slate**  `--run R [--run R2 ...]`: v1 and v2 tags of a stored run side by side (proxy and
  modelled-field popularity, leverage, top-1 % share, P(play), public value rank, v1 chance) and,
  where graded, the actual DK points. Sim paths are remapped to `--sims-root`.
* **opener** `--slate 154575 --contest 196456024 --run R ...`: the 2026-10-20 uploaded GPP (4PP,
  29,726 entries, real %Drafted). v2 leverage is computed against THAT contest's modelled field
  (K = 4); the real top-1 % entries (rank <= ceil(1 % of entries), standings ranks) give each
  player's real top-1 % share; realised leverage = real share / real %Drafted. Reported for the
  v1 tags, the v2 tags, and (v1 had none on the opener) the v1 rule without its 1-in-3 chance
  threshold (top 5 by chance), plus the rank correlation of predicted vs realised leverage.
* **rg**     `--season S --dates N`: walk-forward on RotoGuru with the proxy ownership. Per date:
  projections and sims from the T2.1 harness path (`backtest.SeasonRun`, bundle fit on seasons
  < S, `protocol='avail'`), the public-information proxy (`field.hist_inputs`, public = 1/2 ours
  + 1/2 season average as live), K = 4 noisy large-GPP fields (preset 10,000-entry GPP), v2 tags
  from the truth sims; v1 tags from the raw sims (tier bench / role, proxy band Low, calibrated
  P(DK >= 6 x salary/1000) >= 1/3, our value rank > 10, top 5). **Hindsight top-1 %:** the same K
  modelled fields scored on the ACTUAL DK points; per field the top 1 % of its entries
  (count-weighted line, ties in), averaged over the fields; a player's actual top-1 % share vs
  his ownership (the fields' realised share of entries, which is what the proxy field drafts).
  Writes one row per player and date (CSV) plus a per-date JSON line.
* **builder** `--pairs slate:contest[:run],... --n 20 --seeds 3`: the contest builder with the
  switch OFF, ON (leverage-tilted candidate solves) and a control "ctl" (the same extra scenario
  solves from the same stream with leverage 1, i.e. no tilt: separates "more candidates" from
  "leverage"), same seed = same G / S / R rows ('gs' split) and truth noise. E[payout] on the R rows (the build's report) and on held-out stored
  rows neither arm saw (`--holdout`), paired ON - OFF per row with its SE.

    COURTSIDE_DATA=var/tmp/d3/data PYTHONPATH=src python research/xfactor_v2_eval.py opener \\
        --run c15398ddd87f --run 42da71bcf37a --out var/tmp/d3/opener.json
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
import pandas as pd

from courtside.proj import store as SO
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF
from courtside.proj.dfs import xfactor as XF

sys.path.insert(0, str(Path(__file__).parent))

MAC_VAR = "/Users/nicholashorton/Developer/NBA/var"


def connect(path: str | None = None) -> sqlite3.Connection:
    p = Path(path) if path else Path(os.environ["COURTSIDE_DATA"]) / "courtside.db"
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def spearman(a, b) -> float:
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 2 else float("nan")


# ============================================================================== shared

def v1_rule(rows: list[dict], *, chance_min: float = XF.P_MIN, cap: int = XF.CAP) -> set:
    """v1 tags among `rows` (dicts with chance, band, p_play, tier, value_rank, own, id)."""
    ok = [r for r in rows if r["chance"] is not None and np.isfinite(r["chance"])
          and r["chance"] >= chance_min and r["band"] in XF.OWN_BANDS
          and (r["p_play"] or 0) >= XF.P_PLAY_MIN and r["tier"] in XF.TIERS
          and not (r["value_rank"] is not None and r["value_rank"] <= XF.VALUE_TOP_EXCLUDE)]
    ok.sort(key=lambda r: (-r["chance"], r["own"] if r["own"] is not None else 99.0, r["id"]))
    return {r["id"] for r in ok[:cap]}


def v2_rule(rows: list[dict], cap: int = XF.CAP) -> set:
    ok = [r for r in rows if XF.is_xfactor_v2(r["lev"], r["share"], r["lev_own"], r["p_play"],
                                              r["pub_value_rank"])]
    ok.sort(key=lambda r: (-r["lev"], r["lev_own"], r["id"]))
    return {r["id"] for r in ok[:cap]}


def hindsight(fields, actual: np.ndarray, top: float = XF.LEV_TOP) -> tuple[np.ndarray, np.ndarray]:
    """(actual top-1 % share, field ownership) per pid: each field scored on the actual points
    (one row), averaged over the fields."""
    sh, ow = [], []
    for f in getattr(fields, "fields", [fields]):
        s, o, _ = XF.top1_counts(f, actual[None, :].astype(np.float32), top)
        sh.append(s)
        ow.append(o)
    return np.mean(sh, 0), np.mean(ow, 0)


def tag_stats(ids: set, share: dict, own: dict, actual: dict | None = None,
              salary: dict | None = None) -> dict:
    """Pooled realised leverage of a tag set: Σ share / Σ own, plus the per-tag mean."""
    ids = [i for i in ids if i in share and own.get(i)]
    if not ids:
        return {"n": 0}
    s, o = sum(share[i] for i in ids), sum(own[i] for i in ids)
    out = {"n": len(ids), "share_sum": s, "own_sum": o, "lev_pooled": s / o if o else None,
           "lev_mean": float(np.mean([share[i] / own[i] for i in ids])),
           "in_top1_any": int(sum(share[i] > 0 for i in ids))}
    if actual is not None and salary is not None:
        out["big6"] = int(sum(actual.get(i, 0.0) >= 6 * salary[i] / 1000 for i in ids))
        out["actual_mean"] = float(np.mean([actual.get(i, 0.0) for i in ids]))
    return out


# ============================================================================== slate

def _remap(conn, run_id: str, root: str | None) -> None:
    pass


def run_rows(conn, run_id: str, sims_root: str | None, contest=None) -> tuple[list[dict], dict]:
    """Per DK id of a stored run: v1 table fields + v2 leverage (K fields of `contest`, default
    the slate's reference GPP)."""
    XF._CACHE.clear()
    saved = XF.XF_V2
    XF.set_xf_v2(False)
    try:
        v1 = XF.run_table(conn, run_id)
    finally:
        XF.XF_V2 = saved
    lv = XF.run_leverage(conn, run_id, contest=contest)
    names = {str(a): b for a, b in conn.execute(
        "SELECT DISTINCT site_player_id, player_name FROM dfs_salaries s JOIN dfs_runs r ON "
        "r.slate_id=s.slate_id WHERE r.run_id=?", (run_id,))}
    rows = []
    for d, r in v1.items():
        L = lv["players"].get(d)
        if L is None:
            continue
        rows.append({"id": d, "name": names.get(d, d), "salary": r["salary"],
                     "own": r["own"], "band": r["band"], "tier": r["tier"],
                     "p_play": r["p_play"], "chance": r["chance"], "value_rank": r["value_rank"],
                     "mean": r["mean"], "x_v1": bool(r["x"]), **L,
                     "splits": r.get("splits")})
    return rows, lv


def cmd_slate(args, conn) -> dict:
    out = {}
    for run in args.run:
        rows, lv = run_rows(conn, run, args.sims_root)
        t2 = v2_rule(rows)
        t1 = {r["id"] for r in rows if r["x_v1"]}
        grade = {}
        try:
            from courtside.proj import ids as IDS
            g = dict(conn.execute("SELECT player_id, actual_dk FROM player_projection_grades "
                                  "WHERE run_id=?", (run,)).fetchall())
            dk2nba = IDS.nba_ids(conn, IDS.DK, [r["id"] for r in rows])
            grade = {d: g.get(n) for d, n in dk2nba.items() if g.get(n) is not None}
        except Exception:  # noqa: BLE001
            pass
        show = []
        for r in sorted(rows, key=lambda r: -r["lev"]):
            if r["id"] in t1 | t2:
                show.append({k: r[k] for k in ("name", "salary", "own", "lev_own", "share", "lev",
                                               "lev_raw", "p_play", "chance", "tier",
                                               "pub_value_rank", "value_rank")}
                            | {"v1": r["id"] in t1, "v2": r["id"] in t2,
                               "actual": grade.get(r["id"])})
        out[run] = {"contest": lv["contest"], "seconds": lv["seconds"], "tags": show,
                    "own_unc": lv["own_unc"]}
        print(f"\n== run {run}: {lv['contest']['name']} ({lv['contest']['field_size']}), "
              f"{lv['seconds']} s")
        for x in show:
            print(f"  {'v1' if x['v1'] else '  '} {'v2' if x['v2'] else '  '} {x['name'][:22]:22s}"
                  f" ${x['salary']:>6.0f} proxy {x['own'] or 0:5.1f}% field "
                  f"{x['lev_own'] * 100:5.1f}% top1 {x['share'] * 100:5.1f}% lev {x['lev']:.2f} "
                  f"chance {x['chance']:.2f} pub#{x['pub_value_rank']} our#{x['value_rank']} "
                  f"{x['tier']} actual {x['actual']}")
    return out


# ============================================================================== opener

def cmd_opener(args, conn) -> dict:
    import real_standings_calib as RS
    rcs = {str(rc["contest"].contest_id): rc for rc in RS.real_contests(conn, args.slate)}
    rc = rcs[str(args.contest)]
    out = {"slate": args.slate, "contest": args.contest, "name": rc["contest"].name,
           "entries": rc["contest"].field_size, "runs": {}}
    for run in args.run:
        ctx = RS.setup(conn, args.slate, run)
        pool = ctx["pool"]
        rows, lv = run_rows(conn, run, None, contest=rc["contest"])
        L, bad = RS.lineup_ids(rc["lineup_text"], pool)
        M = len(L)
        n1 = int(math.ceil(XF.LEV_TOP * M))
        ranks = rc["rank"]
        top = [L[i] for i in range(M) if ranks[i] <= ranks[min(n1, M) - 1] and L[i] is not None]
        cnt: dict[str, int] = {}
        for lu in top:
            for p in lu:
                cnt[p] = cnt.get(p, 0) + 1
        real_share = {p: c / len(top) for p, c in cnt.items()}
        real_own = rc["own"]
        rs = {r["id"]: real_share.get(r["id"], 0.0) for r in rows}
        ro = {r["id"]: real_own.get(r["id"], 0.0) for r in rows}
        t1 = {r["id"] for r in rows if r["x_v1"]}
        t1x = v1_rule(rows, chance_min=0.0)
        t2 = v2_rule(rows)
        act = ctx["actual"]
        sal = {r["id"]: r["salary"] for r in rows}
        elig = [r for r in rows if (r["p_play"] or 0) >= XF.P_PLAY_MIN and ro[r["id"]] > 0.001]
        res = {"top1_entries": len(top), "unparsed": bad, "contest_model": lv["contest"],
               "v1": tag_stats(t1, rs, ro, act, sal),
               "v1_no_chance_min": tag_stats(t1x, rs, ro, act, sal),
               "v2": tag_stats(t2, rs, ro, act, sal),
               "spearman_lev_vs_real_lev": spearman(
                   [r["lev"] for r in elig],
                   [(rs[r["id"]] + XF.LEV_PRIOR) / (ro[r["id"]] + XF.LEV_PRIOR) for r in elig]),
               "spearman_share_vs_real_share": spearman([r["share"] for r in elig],
                                                        [rs[r["id"]] for r in elig]),
               "spearman_chance_vs_real_lev": spearman(
                   [r["chance"] for r in elig],
                   [(rs[r["id"]] + XF.LEV_PRIOR) / (ro[r["id"]] + XF.LEV_PRIOR) for r in elig]),
               "n_elig": len(elig), "players": []}
        for r in sorted(rows, key=lambda r: -r["lev"]):
            if r["id"] in t1 | t1x | t2 or rs[r["id"]] >= 0.2:
                res["players"].append({
                    "name": r["name"], "salary": r["salary"], "proxy_own": r["own"],
                    "field_own": r["lev_own"], "pred_share": r["share"], "lev": r["lev"],
                    "chance": r["chance"], "p_play": r["p_play"], "tier": r["tier"],
                    "real_own": ro[r["id"]], "real_top1_share": rs[r["id"]],
                    "actual": act.get(r["id"]), "v1": r["id"] in t1, "v1x": r["id"] in t1x,
                    "v2": r["id"] in t2})
        out["runs"][run] = res
        print(json.dumps({k: v for k, v in res.items() if k != "players"}, default=str),
              flush=True)
        for x in res["players"]:
            print("  ", json.dumps(x, default=lambda v: round(v, 4) if isinstance(v, float)
                                   else str(v)))
    return out


# ============================================================================== rg

def rg_date(conn, sr, day: str, args) -> tuple[list[dict], dict] | None:
    from courtside.proj import pipeline as P
    from courtside.proj.dfs import optimizer as O
    from research_portfolio_backtest import slate_sims
    t0 = time.time()
    pool = O.hist_pool(conn, day, sr.rules)
    inp = F.hist_inputs(conn, pool)
    if inp["fppg"].notna().sum() < 40:
        return None
    rg = sr.rg[sr.rg["game_date"] == day]
    rg_of = {int(r.player_id): str(r.rg_id) for r in rg.itertuples(index=False)
             if not pd.isna(r.player_id)}
    res = P.project_historical(conn, day, protocol="avail", n_sims=args.sims, ctx=sr.ctx,
                               bundle=sr.bundle)
    pl = res.players.drop_duplicates("player_id").set_index("player_id")
    in_pool = {p.dk_player_id for p in pool.players}
    info = {}
    for i, q in pl.iterrows():
        d = rg_of.get(int(i))
        if d in in_pool:
            info[d] = (float(q["dk_mean"]), q.get("tier"), float(q.get("p_play", np.nan)))
    if len(info) < 3 * sr.rules.n_slots:
        return None
    mu = {d: v[0] for d, v in info.items()}
    sims = slate_sims(res, rg_of)
    inp["proj"] = inp["pid"].map(mu)
    contest = F.preset_contest("gpp", args.field, 1.0)
    op = F.ownership_proxy(inp, pool, archetype=contest.archetype)
    seed = P.stable_seed(day, "xf-v2")
    fs = F.build_fields(pool, inp, contest, k=XF.LEV_K, n=args.field, ownership=op, seed=seed,
                        threads=1)
    fp_raw, _ = F.align_sims(sims, fs.pids)
    fp, _ = PF.truth_sims(fp_raw, fs.pids, inp, PF.TRUTH_BETA, PF.TRUTH_TAU,
                          np.random.default_rng(seed + 7))
    lv = XF.leverage(fs, fp)
    pvr = XF.public_value_ranks(op)
    sal = {p.dk_player_id: float(p.salary) for p in pool.players}
    ids = list(fs.pids)
    tiers = [info.get(d, (0, None, 0))[1] for d in ids]
    chance = XF.calibrate(XF.raw_chances(fp_raw, [sal[d] for d in ids]), tiers)
    means = fp_raw.astype(np.float32).mean(0)
    pp = [info.get(d, (0, None, np.nan))[2] for d in ids]
    vr = XF.value_ranks(ids, means, [sal[d] for d in ids], pp)
    actual = O.hist_projections(conn, pool, "actual")
    act = np.array([actual[d].mean if d in actual else 0.0 for d in ids], np.float32)
    ash, aown = hindsight(fs, act)
    rows = []
    for i, d in enumerate(ids):
        if op.own[i] <= 0 or d not in info:
            continue
        rows.append({"date": day, "season": sr.season, "id": d, "salary": sal[d],
                     "own": float(op.own[i] * 100), "band": F.band(float(op.own[i])),
                     "tier": tiers[i], "p_play": pp[i], "chance": float(chance[i]),
                     "value_rank": vr.get(d), "pub_value_rank": pvr.get(d),
                     "share": float(lv["share"][i]), "lev_own": float(lv["own"][i]),
                     "lev": float(lv["lev"][i]), "mean": float(means[i]),
                     "actual": float(act[i]), "act_share": float(ash[i]),
                     "act_own": float(aown[i])})
    t1, t2 = v1_rule(rows), v2_rule(rows)
    for r in rows:
        r["x1"], r["x2"] = r["id"] in t1, r["id"] in t2
    sh = {r["id"]: r["act_share"] for r in rows}
    ow = {r["id"]: r["act_own"] for r in rows}
    a = {r["id"]: r["actual"] for r in rows}
    s_ = {r["id"]: r["salary"] for r in rows}
    summ = {"date": day, "season": sr.season, "games": len(res.sims), "players": len(rows),
            "v1": tag_stats(t1, sh, ow, a, s_), "v2": tag_stats(t2, sh, ow, a, s_),
            "seconds": round(time.time() - t0, 1)}
    return rows, summ


def cmd_rg(args, conn) -> dict:
    from courtside import db
    from courtside.proj import backtest as BT
    sys.modules.setdefault("research_portfolio_backtest",
                           __import__("portfolio_backtest"))
    wconn = db.connect(args.db) if args.db else db.connect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    have = set(done["date"]) if len(done) else set()
    all_rows = done.to_dict("records") if len(done) else []
    sr = BT.SeasonRun(wconn, args.season, BT.Settings(modes=("cash",), n_sims=args.sims),
                      log=lambda m: print(m, flush=True))
    with out.with_suffix(".jsonl").open("a") as fh:
        for day in sr.dates(args.dates):
            if day in have:
                continue
            try:
                r = rg_date(wconn, sr, day, args)
            except Exception as e:  # noqa: BLE001 - one bad date must not stop the run
                print(f"{day}: failed {type(e).__name__}: {e}", flush=True)
                continue
            if r is None:
                print(f"{day}: skipped", flush=True)
                continue
            rows, summ = r
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(csv, index=False)
            fh.write(json.dumps(summ, default=str) + "\n")
            fh.flush()
            print(json.dumps(summ, default=str), flush=True)
    return {"rows": len(all_rows)}


def summarize_rg(paths: list[str]) -> dict:
    """Pooled over dates: tags' actual top-1 % share vs their (field) ownership, v1 vs v2, with
    a date-clustered SE of the pooled ratio (delta method)."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out = {}
    for season, d in list(df.groupby("season")) + [("pooled", df)]:
        res = {"dates": int(d["date"].nunique())}
        for tag in ("x1", "x2"):
            t = d[d[tag]]
            g = t.groupby("date").agg(s=("act_share", "sum"), o=("act_own", "sum"),
                                      n=("id", "size"))
            S, O = g["s"].sum(), g["o"].sum()
            ratio = S / O if O else float("nan")
            k = len(g)
            se = (math.sqrt(((g["s"] - ratio * g["o"]) ** 2).sum() * k / max(k - 1, 1)) / O
                  if O and k > 1 else float("nan"))
            big = (t["actual"] >= 6 * t["salary"] / 1000).mean() if len(t) else float("nan")
            res[tag] = {"tags": int(len(t)), "per_date": float(len(t) / max(1, res["dates"])),
                        "share_sum": float(S), "own_sum": float(O), "lev": float(ratio),
                        "lev_se": float(se), "share_mean": float(t["act_share"].mean()),
                        "own_mean": float(t["act_own"].mean()),
                        "proxy_own_mean": float(t["own"].mean()),
                        "pred_lev_mean": float(t["lev"].mean()),
                        "pred_share_mean": float(t["share"].mean()),
                        "in_top1_rate": float((t["act_share"] > 0).mean()),
                        "big6_rate": float(big), "actual_mean": float(t["actual"].mean()),
                        "salary_mean": float(t["salary"].mean())}
        e = d[(d["p_play"] >= XF.P_PLAY_MIN) & (d["act_own"] > 0.001)]
        alev = (e["act_share"] + XF.LEV_PRIOR) / (e["act_own"] + XF.LEV_PRIOR)
        res["spearman_pred_lev_vs_actual_lev"] = spearman(e["lev"], alev)
        res["spearman_chance_vs_actual_lev"] = spearman(e["chance"], alev)
        res["spearman_pred_share_vs_actual_share"] = spearman(e["share"], e["act_share"])
        res["all_eligible_lev"] = float(e["act_share"].sum() / e["act_own"].sum())
        res["paired_v2_minus_v1"] = _paired(d, "x1", "x2")
        res["variants"] = _variants(d)
        out[season] = res
    return out


def _pooled_ratio(t: pd.DataFrame) -> float:
    o = t["act_own"].sum()
    return float(t["act_share"].sum() / o) if o else float("nan")


def _paired(d: pd.DataFrame, a: str, b: str, reps: int = 2000) -> dict:
    """Pooled realised leverage of tag set b minus a, SE by a bootstrap over dates."""
    dates = d["date"].unique()
    by = {k: g for k, g in d.groupby("date")}
    est = _pooled_ratio(d[d[b]]) - _pooled_ratio(d[d[a]])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(reps):
        pick = pd.concat([by[x] for x in rng.choice(dates, len(dates))])
        bs.append(_pooled_ratio(pick[pick[b]]) - _pooled_ratio(pick[pick[a]]))
    bs = np.array(bs)
    return {"diff": est, "se": float(np.nanstd(bs)),
            "p_le_0": float(np.nanmean(bs <= 0))}


def _variants(d: pd.DataFrame) -> dict:
    """The v2 rule re-applied per date with other constants (from the stored per-player rows)."""
    out = {}
    raw = d["share"] / d["lev_own"].where(d["lev_own"] > 0)
    for name, lev_min, share_min, prior, own_max in (
            ("shipped", XF.LEV_MIN, XF.SHARE_MIN, XF.LEV_PRIOR, XF.OWN_MAX),
            ("lev1.3", 1.3, XF.SHARE_MIN, XF.LEV_PRIOR, XF.OWN_MAX),
            ("lev2.0", 2.0, XF.SHARE_MIN, XF.LEV_PRIOR, XF.OWN_MAX),
            ("share1%", XF.LEV_MIN, 0.01, XF.LEV_PRIOR, XF.OWN_MAX),
            ("share3%", XF.LEV_MIN, 0.03, XF.LEV_PRIOR, XF.OWN_MAX),
            ("share0", XF.LEV_MIN, 0.0, XF.LEV_PRIOR, XF.OWN_MAX),
            ("prior0", XF.LEV_MIN, XF.SHARE_MIN, 0.0, XF.OWN_MAX),
            ("prior0_share0", XF.LEV_MIN, 0.0, 0.0, XF.OWN_MAX),
            ("prior1%", XF.LEV_MIN, XF.SHARE_MIN, 0.01, XF.OWN_MAX),
            ("own20%", XF.LEV_MIN, XF.SHARE_MIN, XF.LEV_PRIOR, 0.20),
            ("no_pub_excl", XF.LEV_MIN, XF.SHARE_MIN, XF.LEV_PRIOR, XF.OWN_MAX)):
        lev = (d["share"] + prior) / (d["lev_own"] + prior) if prior else raw
        ok = ((lev >= lev_min) & (d["share"] >= share_min) & (d["p_play"] >= XF.P_PLAY_MIN)
              & (d["lev_own"] < own_max))
        if name != "no_pub_excl":
            ok &= ~(d["pub_value_rank"].fillna(1e9) <= XF.VALUE_TOP_EXCLUDE)
        t = d.assign(_lev=lev, _ok=ok)[ok]
        t = t.sort_values(["date", "_lev"], ascending=[True, False]).groupby("date").head(XF.CAP)
        big = (t["actual"] >= 6 * t["salary"] / 1000).mean() if len(t) else float("nan")
        out[name] = {"tags": int(len(t)), "per_date": float(len(t) / d["date"].nunique()),
                     "lev": _pooled_ratio(t), "own_mean": float(t["act_own"].mean()),
                     "share_mean": float(t["act_share"].mean()),
                     "pred_lev_mean": float(t["_lev"].mean()) if len(t) else float("nan"),
                     "in_top1_rate": float((t["act_share"] > 0).mean()) if len(t) else None,
                     "big6_rate": float(big)}
    return out


# ============================================================================== builder

def report_payout(pf: PF.Portfolio, fp: np.ndarray, contest: F.Contest, seed: int
                  ) -> np.ndarray:
    """Per-row total payout of the chosen set on truth rows `fp` (columns = pf.field.pids) with
    the build's dupe model (Poisson extra ties), seeded the same in both arms."""
    import portfolio_split_eval as PSE
    fld, n = pf.field, len(pf.lineups)
    pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in pf.slots()]
    ev = F.evaluate(fld, fp, contest, pairs, seed=seed, tiers=False, own_entries=n, per_sim=True)
    m_o = max(1, contest.field_size - n)
    tied = np.rint(ev.per_sim["tied"]).astype(np.int64)
    base = fld.fields[0] if isinstance(fld, F.FieldSet) else fld
    extra = np.array([max(0.0, float(lu["dupes"]) * m_o / max(1, contest.field_size - 1) -
                          PSE._in_sample(base, p, m_o)) for lu, p in zip(pf.lineups, pairs)])
    rng = np.random.default_rng(seed + 98)
    if (extra > 1e-4).any():
        tied += rng.poisson(extra[None, :], size=tied.shape)
    lad = PF.Ladder(contest)
    pay, _ = PF.joint_payouts(lad, ev.per_sim["score"],
                              np.rint(ev.per_sim["above"]).astype(np.int64), tied)
    return pay.sum(1)


def cmd_builder(args, conn) -> dict:
    import portfolio_split_eval as PSE
    out = []
    for item in args.pairs.split(","):
        slate, cid, *rest = item.split(":")
        ctx = PSE.load(conn, slate, rest[0] if rest else None)
        contest = F.load_contest(conn, cid)
        S = ctx["sims"]["fp"].shape[0]
        sims, hold = ctx["sims"], None
        if args.holdout:
            hold = np.arange(S - args.holdout, S)
            sims = {**sims, "fp": sims["fp"][:S - args.holdout]}
        n = min(args.n, contest.field_size - 1)
        res = {"slate": slate, "contest": cid, "name": contest.name, "M": contest.field_size,
               "fee": contest.entry_fee, "run": ctx["run"], "n": n, "seeds": []}
        for seed in range(args.seed0, args.seed0 + args.seeds):
            arms = {}
            real_lev = XF.builder_leverage
            for arm in ("off", "ctl", "on"):
                XF.set_xf_v2(arm != "off")
                if arm == "ctl":         # the same extra scenario solves, no tilt (lev = 1)
                    XF.builder_leverage = lambda *a, **k: (
                        {p: 1.0 for p in real_lev(*a, **k)[0]}, {"version": "ctl", "k": 0})
                t = time.perf_counter()
                try:
                    pf = PF.build_portfolio(ctx["pool"], ctx["inputs"], sims, contest,
                                            PF.Settings(n=n, seed=seed, n_sims=None, threads=1),
                                            split="gs")
                finally:
                    XF.builder_leverage = real_lev
                secs = time.perf_counter() - t
                R = PF.split_rows(pf.fp.shape[0], seed, "gs")["R"]
                rep = report_payout(pf, pf.fp[R], contest, seed)
                hp = None
                if hold is not None:
                    h = PSE.honest(pf, ctx, hold, contest, seed)
                    hp = h.pop("_tot")
                src = PF._count(pf.sources or [])
                arms[arm] = {"pf": pf, "rep": rep, "hold": hp, "secs": secs, "src": src}
                print(json.dumps({"slate": slate, "arm": arm, "seed": seed, "n": n,
                                  "e_payout_R": pf.summary["e_payout"],
                                  "rescored_R": float(rep.mean()),
                                  "holdout": None if hp is None else float(hp.mean()),
                                  "selected_sources": src, "seconds": round(secs, 1),
                                  "xf2": pf.info.get("xfactor_v2")}, default=str), flush=True)
            XF.set_xf_v2(False)
            d = arms["on"]["rep"] - arms["off"]["rep"]
            dc = arms["on"]["rep"] - arms["ctl"]["rep"]
            row = {"seed": seed, "ctl_R": float(arms["ctl"]["pf"].summary["e_payout"]),
                   "on_minus_ctl_R": float(dc.mean()),
                   "on_minus_ctl_R_se": float(dc.std() / math.sqrt(len(dc))),
                   "off_R": float(arms["off"]["pf"].summary["e_payout"]),
                   "on_R": float(arms["on"]["pf"].summary["e_payout"]),
                   "diff_R": float(d.mean()), "diff_R_se": float(d.std() / math.sqrt(len(d))),
                   "off_p_any_top1": arms["off"]["pf"].summary["p_any_top1"],
                   "on_p_any_top1": arms["on"]["pf"].summary["p_any_top1"],
                   "levtilt_selected": arms["on"]["src"].get("levtilt", 0),
                   "lineups_changed": n - len(
                       {lu["slots_json"] for lu in arms["on"]["pf"].lineups}
                       & {lu["slots_json"] for lu in arms["off"]["pf"].lineups}),
                   "secs_off": round(arms["off"]["secs"], 1),
                   "secs_on": round(arms["on"]["secs"], 1)}
            if hold is not None:
                dh = arms["on"]["hold"] - arms["off"]["hold"]
                dhc = arms["on"]["hold"] - arms["ctl"]["hold"]
                row.update(off_hold=float(arms["off"]["hold"].mean()),
                           on_hold=float(arms["on"]["hold"].mean()),
                           ctl_hold=float(arms["ctl"]["hold"].mean()),
                           diff_hold=float(dh.mean()),
                           diff_hold_se=float(dh.std() / math.sqrt(len(dh))),
                           on_minus_ctl_hold=float(dhc.mean()),
                           on_minus_ctl_hold_se=float(dhc.std() / math.sqrt(len(dhc))))
            res["seeds"].append(row)
            print(json.dumps({"slate": slate, "paired": row}), flush=True)
        out.append(res)
    return {"pairs": out}


# ============================================================================== main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("slate")
    a.add_argument("--run", action="append", required=True)
    a.add_argument("--sims-root", default=None)
    b = sub.add_parser("opener")
    b.add_argument("--slate", default="154575")
    b.add_argument("--contest", default="196456024")
    b.add_argument("--run", action="append", required=True)
    c = sub.add_parser("rg")
    c.add_argument("--season", required=True)
    c.add_argument("--dates", type=int, default=60)
    c.add_argument("--sims", type=int, default=4000)
    c.add_argument("--field", type=int, default=10_000)
    c.add_argument("--db", default=None)
    d = sub.add_parser("rgsum")
    d.add_argument("csv", nargs="+")
    e = sub.add_parser("builder")
    e.add_argument("--pairs", required=True)
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--seed0", type=int, default=0)
    e.add_argument("--holdout", type=int, default=5000)
    for p in (a, b, c, d, e):
        p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "rgsum":
        res = summarize_rg(args.csv)
    else:
        conn = None if args.cmd == "rg" else connect()
        res = {"slate": cmd_slate, "opener": cmd_opener, "rg": cmd_rg,
               "builder": cmd_builder}[args.cmd](args, conn)
    if args.cmd != "rg":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    if args.cmd == "rgsum":
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
