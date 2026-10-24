"""B6: the Showdown captain (CPT) ownership term, measured on the stored Showdown slates.

IMPROVEMENTS.md B6 / PLAYERS.md §5.7; write-up docs/research/b6-d10.md.

No Showdown standings have been uploaded (the 4 uploads are slate 154575, Classic) and there is
no public Showdown ownership in our data, so the accuracy check (`courtside dfs showdown-own`,
`showdown.ownership_eval`) cannot run yet. This script does what can be done now:

  measure   per stored Showdown slate (its reference run, inputs as the public saw them at lock),
            the ownership proxy with the captain term OFF and ON: the salary shadow price
            lambda (the §9 question: is lambda < 0, i.e. does exp(-lambda (s - mean)) favour
            expensive players, more so at CPT?), and how CPT ownership is distributed: share on
            the top-1 / top-3 public projections, HHI, rank correlation with the public
            projection, value per $1k and salary, CPT / total ownership of the top-3 projections
            vs the top value plays, and the CPT allocation at lambda = 0 (what the utility alone
            gives).
  eval      `showdown.ownership_eval` (CPT and UTIL MAE / rank corr OFF vs ON with SEs) once
            Showdown standings exist.
  identity  OFF (COURTSIDE_SD_CPT unset) vs cpt=False vs HEAD tree (PYTHONPATH) hashes of the
            proxy and the field on every stored Showdown slate: run twice, diff the JSON.

Usage (scratch DB, never prod):
  COURTSIDE_DATA=<scratch data> python research/b6_showdown_own.py measure --out out/measure.json
  COURTSIDE_DATA=<scratch data> python research/b6_showdown_own.py identity --out out/id_X.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import standings as ST


def sd_slates(conn) -> list[dict]:
    out = []
    for r in conn.execute("SELECT slate_id, lock_utc FROM dfs_slates WHERE tag='showdown' OR "
                          "game_type_id=81 ORDER BY lock_utc"):
        run = ST._reference_run_before(conn, str(r["slate_id"]), r["lock_utc"])
        if run is None:
            continue
        has = conn.execute("SELECT sim_path FROM dfs_runs WHERE run_id=?",
                           (run["run_id"],)).fetchone()
        out.append({"slate_id": str(r["slate_id"]), "lock": r["lock_utc"],
                    "run_id": run["run_id"], "kind": run["kind"], "sims": bool(has and has[0])})
    return out


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def describe(op: F.OwnershipProxy) -> dict:
    k_c, k_u = op.slot_names.index("CPT"), op.slot_names.index("UTIL")
    cpt, util = op.by_slot[:, k_c], op.by_slot[:, k_u]
    inp = op.inputs
    pub = inp["pub_proj"].to_numpy(float)
    val = inp["pub_value"].to_numpy(float)
    sal = inp["salary"].to_numpy(float)
    act = op.own > 0
    order = np.argsort(-np.where(act, pub, -1e9))
    top1, top3 = order[:1], order[:3]
    # top value plays: the 3 best public value among active players outside the top-6 projections
    rest = [i for i in np.argsort(-np.where(act, val, -1e9)) if i not in set(order[:6])][:3]
    tot = op.own
    return {
        "lam": float(op.lam), "n_active": int(act.sum()),
        "cpt_top1": float(cpt[top1].sum()), "cpt_top3": float(cpt[top3].sum()),
        "cpt_hhi": float((cpt ** 2).sum()), "cpt_max": float(cpt.max()),
        "cpt_n_over_5pct": int((cpt > 0.05).sum()),
        "util_top3": float(util[top3].sum() / 5), "util_max": float(util.max()),
        "rc_cpt_pub": _spear(cpt[act], pub[act]), "rc_cpt_value": _spear(cpt[act], val[act]),
        "rc_cpt_salary": _spear(cpt[act], sal[act]), "rc_util_pub": _spear(util[act], pub[act]),
        "cpt_share_top3": float(cpt[top3].sum() / max(1e-9, tot[top3].sum())),
        "cpt_share_value3": float(cpt[rest].sum() / max(1e-9, tot[rest].sum())) if rest else None,
        # salary-weighted mean (UTIL salary) of a captain pick vs a UTIL pick
        "cpt_mean_salary": float((cpt * sal).sum() / max(1e-9, cpt.sum())),
        "util_mean_salary": float((util * sal).sum() / max(1e-9, util.sum())),
        "top": [{"name": inp["name"].iloc[i], "salary": int(sal[i]), "pub": round(float(pub[i]), 1),
                 "cpt": round(float(cpt[i]), 4), "util": round(float(util[i]), 4)}
                for i in order[:8]],
    }


def at_lambda0(op: F.OwnershipProxy, pool, cpt: bool) -> dict:
    """The allocation with the salary shadow price off (lambda = 0): what the utility gives."""
    inp = op.inputs
    P = op.params
    u, active, feats = F.utility(inp, P, op.archetype)
    by_id = pool.by_id()
    names, counts, E, S = F._eligibility(inp, by_id, pool.rules)
    E &= active[:, None]
    umax = np.max(u[active])
    base = np.repeat(np.where(active, np.exp(np.clip(u - umax, -50, 0)), 0.0)[:, None],
                     len(names), axis=1)
    if cpt:
        uc = F.cpt_utility(inp, P, op.archetype, feats, active)
        base[:, names.index("CPT")] = np.where(active, np.exp(np.clip(uc - np.max(uc[active]),
                                                                       -50, 0)), 0.0)
    X = F.sinkhorn(base * E, counts, op.cap)
    k_c = names.index("CPT")
    sal = inp["salary"].to_numpy(float)
    spend = float((X * S).sum())
    pub = feats["pub"]
    order = np.argsort(-np.where(active, pub, -1e9))
    return {"spend_frac": spend / pool.rules.salary_cap,
            "cpt_top1": float(X[order[:1], k_c].sum()), "cpt_top3": float(X[order[:3], k_c].sum()),
            "cpt_mean_salary": float((X[:, k_c] * sal).sum())}


def measure(conn, archetypes=("large_gpp", "small_gpp")) -> dict:
    out = []
    for sl in sd_slates(conn):
        try:
            pool, inp, info = F.slate_inputs(conn, sl["slate_id"], sl["run_id"], as_of=sl["lock"])
        except Exception as e:  # noqa: BLE001
            out.append({**sl, "error": f"{type(e).__name__}: {e}"})
            continue
        rec = {**sl, "n_projected": info["n_projected"], "n_totals": info["n_totals"]}
        for arch in archetypes:
            for arm in ("off", "cpt"):
                op = F.ownership_proxy(inp, pool, archetype=arch, cpt=(arm == "cpt"))
                d = describe(op)
                d["lam0"] = at_lambda0(op, pool, arm == "cpt")
                d["version"] = op.version
                rec[f"{arch}/{arm}"] = d
        out.append(rec)
        print(sl["slate_id"], {k: round(v["cpt_top1"], 3) for k, v in rec.items()
                               if isinstance(v, dict) and "cpt_top1" in v}, flush=True)
    return {"slates": out, "cpt_params": F.CPT_PARAMS}


def summarise(m: dict) -> dict:
    rows = [r for r in m["slates"] if "error" not in r]
    keys = ["lam", "cpt_top1", "cpt_top3", "cpt_hhi", "cpt_max", "cpt_n_over_5pct", "util_top3",
            "util_max", "rc_cpt_pub", "rc_cpt_value", "rc_cpt_salary", "rc_util_pub",
            "cpt_share_top3", "cpt_share_value3", "cpt_mean_salary", "util_mean_salary"]
    out = {"n_slates": len(rows)}
    for arm in [k for k in rows[0] if "/" in k] if rows else []:
        d = {}
        for k in keys:
            v = np.array([r[arm][k] for r in rows if r[arm].get(k) is not None], float)
            d[k] = {"mean": float(np.nanmean(v)),
                    "se": float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()))}
        for k in ("cpt_top1", "cpt_top3", "spend_frac", "cpt_mean_salary"):
            v = np.array([r[arm]["lam0"][k] for r in rows], float)
            d[f"lam0_{k}"] = {"mean": float(v.mean()),
                              "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
        d["lam_negative"] = int(sum(r[arm]["lam"] < 0 for r in rows))
        out[arm] = d
    return out


def identity(conn) -> dict:
    """Hashes of the OFF proxy (both archetypes) and a small field per stored Showdown slate."""
    out = {}
    kw = {"cpt": False} if "cpt" in F.ownership_proxy.__code__.co_varnames else {}
    for sl in sd_slates(conn):
        pool, inp, _ = F.slate_inputs(conn, sl["slate_id"], sl["run_id"], as_of=sl["lock"])
        for arch in ("large_gpp", "small_gpp", "cash"):
            op = F.ownership_proxy(inp, pool, archetype=arch, **kw)
            out[f"{sl['slate_id']}/{arch}"] = (
                hashlib.sha256(np.ascontiguousarray(op.by_slot).tobytes()).hexdigest()[:16]
                + f" {op.lam!r} {op.version}")
        op = F.ownership_proxy(inp, pool, archetype="large_gpp", **kw)
        f = F.build_field(pool, inp, F.preset_contest("gpp", 3000), n=3000, n_solves=20,
                          seed=5, threads=1, ownership=op)
        out[f"{sl['slate_id']}/field"] = (
            hashlib.sha256(np.ascontiguousarray(f.lineups).tobytes()
                           + np.ascontiguousarray(f.counts).tobytes()).hexdigest()[:16]
            + " " + f.version)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "eval", "identity"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    conn = db.connect()
    if a.cmd == "measure":
        m = measure(conn)
        res = {"measure": m, "summary": summarise(m)}
    elif a.cmd == "eval":
        from courtside.proj.dfs import showdown as SD
        res = SD.ownership_eval(conn)
    else:
        res = identity(conn)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(txt)
    print(json.dumps(res.get("summary", res), indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
