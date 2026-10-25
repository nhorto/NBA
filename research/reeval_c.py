"""Re-evaluation workstream C: field and ownership vs real DK standings (H8).

docs/REEVAL_2026-10.md (H8), docs/research/reeval-C.md. Read-only on the DB it is pointed at
(opened with ``?mode=ro``); writes only under ``--out`` (default var/tmp/reeval-c).

Commands:

* **data**   Per uploaded Classic contest: the ownership-model inputs as the public saw them at
  lock (salary capture at or before the lock, the slate's last lock10 / lock45 run made before
  the lock: `standings._reference_run_before`), the proxy at the contest's real type (cash =
  double-up, large GPP > 1,000 entries) and the real %Drafted, one row per pool player, plus
  diagnosis features (salary tier, value ranks, teammate-out news, recent form). Cached as a
  pickle in --out.
* **diag**   Proxy vs real by contest type, salary tier, public value rank and news; residual
  log-odds regressions (concentration vs features).
* **loso**   Leave-one-slate-out refit of the concentration / news parameters (the ON variant
  `field.OWN_CAL_PARAMS`) and its MAE, rank correlation and top-10 bias, OFF vs ON.
* **lines**  The modelled field scored at the actual box scores, OFF vs ON: cash and top-1 %
  lines per real contest (as research/real_standings_calib.py `field`).

    PYTHONPATH=src:research nice -n 10 uv run python research/reeval_c.py data --db var/courtside.db
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj import ids as IDS                       # noqa: E402
from courtside.proj.dfs import field as F                   # noqa: E402
from courtside.proj.dfs import standings as ST              # noqa: E402

OUT = Path("var/tmp/reeval-c")
SLATES = ("154575", "155547", "155576")
SAL_TIERS = ((0, 4000, "<4k"), (4000, 6000, "4-6k"), (6000, 8000, "6-8k"), (8000, 99999, "8k+"))


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================== data

def contest_type(conn, cid: str, n_results: int) -> tuple[str, str]:
    """(name, archetype): cash for double-ups, else by the standings' entry count."""
    c = F.load_contest(conn, cid)
    name = c.name if c and c.name else None
    if not name:
        r = conn.execute("SELECT note FROM dfs_entries WHERE contest_id=? AND note LIKE "
                         "'history:%' LIMIT 1", (cid,)).fetchone()
        name = (r[0] if r else "").replace("history:", "").split(" (")[0].strip() or cid
    if F.is_double_up(name):
        return name, "cash"
    return name, "small_gpp" if n_results <= F.SMALL_GPP_MAX else "large_gpp"


def teammate_out(conn, inp: pd.DataFrame, pool, day: str) -> pd.DataFrame:
    """Per pool player: the recent minutes (last 15 games) of teammates who are OUT at the
    lock capture (DK status O / OUT, or disabled / absent from the pool while they played
    >= 15 minutes in their last games), and the count of such teammates with >= 20 minutes."""
    st = inp["status"].fillna("").astype(str).str.upper()
    out_rec = inp.loc[st.isin(list(F.OUT_STATUSES)), ["team", "recent_min"]]
    agg = out_rec.assign(rm=out_rec["recent_min"].fillna(0.0)).groupby("team")["rm"]
    lost = agg.sum().to_dict()
    n20 = out_rec.assign(b=out_rec["recent_min"].fillna(0) >= 20).groupby("team")["b"].sum() \
        .to_dict()
    return pd.DataFrame({"pid": inp["pid"],
                         "tm_out_min": inp["team"].map(lost).fillna(0.0).to_numpy(),
                         "tm_out_n20": inp["team"].map(n20).fillna(0).to_numpy()})


def build_data(conn) -> dict:
    nba2dk = {v: k for k, v in IDS.nba_ids(conn, IDS.DK).items()}
    out = {"slates": {}, "contests": []}
    for sid in SLATES:
        lock = F._slate_lock(conn, sid)
        run = ST._reference_run_before(conn, sid, lock)
        pool, inp, info = F.slate_inputs(conn, sid, run["run_id"], as_of=lock, cal=False)
        dk2nba = IDS.nba_ids(conn, IDS.DK, [p.dk_player_id for p in pool.players])
        day = info["nba_day"]
        rdk = F.recent_dk(conn, dk2nba.values(), day)
        inp["recent_dk"] = pd.to_numeric(inp["pid"].map(dk2nba).map(rdk), errors="coerce")
        out["slates"][sid] = {"lock": lock, "run_id": run["run_id"], "run_kind": run["kind"],
                              "run_at": run["created_at"], "day": day, "pool": pool,
                              "inputs": inp, "dk2nba": dk2nba, "recent_dk3": rdk,
                              "tm": teammate_out(conn, inp, pool, day)}
        cids = [r[0] for r in conn.execute(
            "SELECT DISTINCT o.contest_id FROM dfs_contest_ownership o JOIN dfs_contests c ON "
            "c.contest_id=o.contest_id WHERE c.slate_id=?", (sid,))]
        for cid in cids:
            n = conn.execute("SELECT COUNT(*) FROM dfs_contest_results WHERE contest_id=?",
                             (cid,)).fetchone()[0]
            name, arch = contest_type(conn, cid, n)
            own: dict[str, float] = {}
            for pid, pct in conn.execute("SELECT player_id, pct_drafted FROM "
                                         "dfs_contest_ownership WHERE contest_id=?", (cid,)):
                k = nba2dk.get(int(pid))
                if k is not None and pct is not None:
                    own[str(k)] = own.get(str(k), 0.0) + float(pct) / 100.0
            pts = np.array([r[0] for r in conn.execute(
                "SELECT points FROM dfs_contest_results WHERE contest_id=? AND points IS NOT "
                "NULL ORDER BY points DESC", (cid,))], float)
            texts = [r[0] for r in conn.execute(
                "SELECT lineup_text FROM dfs_contest_results WHERE contest_id=?", (cid,))]
            out["contests"].append({"contest_id": cid, "slate_id": sid, "name": name,
                                    "archetype": arch, "entries": int(n), "own": own,
                                    "points": pts, "lineup_text": texts})
    return out


def player_table(D: dict, cal: dict | None = None) -> pd.DataFrame:
    """One row per (contest, pool player): proxy (OFF, or ON with `cal` params) vs real."""
    rows = []
    for c in D["contests"]:
        S = D["slates"][c["slate_id"]]
        kw = {} if cal is None else {"cal": cal}
        op = F.ownership_proxy(S["inputs"], S["pool"], archetype=c["archetype"], **kw)
        t = op.inputs.copy()
        t["own"] = op.own
        t["real"] = t["pid"].map(c["own"]).fillna(0.0)
        t = t.merge(S["tm"], on="pid", how="left")
        t["nba_id"] = t["pid"].map(S["dk2nba"])
        t["dk3"] = t["nba_id"].map(S["recent_dk3"])
        t["contest_id"], t["slate_id"], t["archetype"] = c["contest_id"], c["slate_id"], \
            c["archetype"]
        t["cname"], t["entries"] = c["name"], c["entries"]
        rows.append(t)
    T = pd.concat(rows, ignore_index=True)
    T["tier"] = pd.cut(T["salary"], [a for a, _, _ in SAL_TIERS] + [99999],
                       labels=[n for _, _, n in SAL_TIERS], right=False)
    act = T["pub_proj"] >= F.MIN_PUB_PROJ
    T["value_rank"] = T.where(act).groupby("contest_id")["pub_value"].rank(ascending=False)
    T["proj_value"] = T["proj"] / (T["salary"] / 1000.0)
    T["ours_value_rank"] = T.where(act).groupby("contest_id")["proj_value"].rank(ascending=False)
    T["proj_rank"] = T.where(act).groupby("contest_id")["pub_proj"].rank(ascending=False)
    T["news"] = (T["tm_out_n20"].fillna(0) > 0) & ((T["proj_min"] - T["recent_min"]) >= 3)
    return T


# ============================================================================== metrics

def own_metrics(p: np.ndarray, y: np.ndarray, top: int = 10) -> dict:
    """Same definition as field.calibration_report / real_standings_calib.own_metrics."""
    keep = (p > 0.001) | (y > 0.001)
    p, y = p[keep], y[keep]
    o = np.argsort(-y)[:top]
    lp, ly = _logit(p), _logit(y)
    tail = y < 0.10
    po = np.argsort(-p)[:top]
    return {"mae_pp": float(np.mean(np.abs(p - y)) * 100), "spearman": F._spearman(p, y),
            "top10_mae_pp": float(np.mean(np.abs(p[o] - y[o])) * 100),
            "top10_bias_pp": float(np.mean(p[o] - y[o]) * 100),
            "ptop10_bias_pp": float(np.mean(p[po] - y[po]) * 100),
            "logit_slope": float(np.polyfit(lp, ly, 1)[0]),
            "tail_mae_pp": float(np.mean(np.abs(p[tail] - y[tail])) * 100),
            "logit_mae": float(np.mean(np.abs(lp - ly))),
            "sum_sq_real": float((y ** 2).sum()), "sum_sq_pred": float((p ** 2).sum()),
            "n": int(len(p))}


def _logit(x):
    x = np.clip(np.asarray(x, float), 0.005, 0.995)
    return np.log(x / (1 - x))


def by_contest(T: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid, g in T.groupby("contest_id", sort=False):
        m = own_metrics(g["own"].to_numpy(), g["real"].to_numpy())
        out.append({"contest_id": cid, "slate": g["slate_id"].iloc[0],
                    "name": g["cname"].iloc[0][:28], "type": g["archetype"].iloc[0],
                    "entries": g["entries"].iloc[0], **m})
    return pd.DataFrame(out)


def group_bias(T: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean proxy - real (pp) and mean |err| by `col` and contest type, rows above 0.1 %."""
    k = T[(T["own"] > 0.001) | (T["real"] > 0.001)]
    g = k.groupby(["archetype", col], observed=True)
    return pd.DataFrame({"n": g.size(), "real": g["real"].mean() * 100,
                         "proxy": g["own"].mean() * 100,
                         "bias_pp": (g["own"].mean() - g["real"].mean()) * 100,
                         "mae_pp": g.apply(lambda d: np.mean(np.abs(d["own"] - d["real"])) * 100,
                                           include_groups=False)}).round(2)


# ============================================================================== fit

# nested variants: which spec entries are free (the rest stay at the 1.2 values / 0)
VARIANTS = {
    "temp": ("conc",),                                     # concentration only
    "temp+w": ("conc", "w_ours"),
    "feat": ("w_ours", "b_sal", "b_form", "b_news"),      # features, 1.2 concentration
    "full": ("conc", "w_ours", "b_sal", "b_form", "b_news"),
    "full+b": ("conc", "w_ours", "b_sal", "b_form", "b_news", "b_value", "b_proj"),
}
BOUNDS = {"conc": (0.2, 4.0), "w_ours": (0.0, 1.0), "b_sal": (-2.0, 3.0),
          "b_form": (-3.0, 4.0), "b_news": (-3.0, 4.0), "b_value": (0.0, 4.0),
          "b_proj": (-1.0, 4.0)}


def base_spec() -> dict:
    """The 1.2 proxy written as a spec (the research starting point; = OFF numerically)."""
    return {"params": {"b_sal": 0.0, "b_form": 0.0, "b_news": 0.0, "b_value": 1.10,
                       "b_proj": 0.70, "w_ours": 0.50},
            "arch_params": {},
            "conc": {"cash": 1.60, "small_gpp": 1.00, "large_gpp": 0.90}}


def spec_from(theta, keys, start: dict, arch: str) -> dict:
    sp = json.loads(json.dumps(start))
    for k, v in zip(keys, theta):
        lo, hi = BOUNDS[k]
        v = float(np.clip(v, lo, hi))
        if k == "conc":
            sp["conc"][arch] = v
            if arch == "large_gpp":
                sp["conc"]["small_gpp"] = v * 1.00 / 0.90     # keep the 1.2 small / large ratio
        else:
            sp["params"][k] = v
    return sp


def proxies(D: dict, contests: list[dict], spec) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """(pred, real, contest) per contest; one proxy per (slate, archetype)."""
    cache = {}
    out = []
    for c in contests:
        key = (c["slate_id"], c["archetype"])
        S = D["slates"][c["slate_id"]]
        if key not in cache:
            op = F.ownership_proxy(S["inputs"], S["pool"], archetype=c["archetype"],
                                   cal=spec if spec is not None else False)
            cache[key] = (op.pids, op.own)
        pids, own = cache[key]
        y = np.array([c["own"].get(p, 0.0) for p in pids])
        out.append((own, y, c))
    return out


def loss_fn(pairs, kind: str = "logit") -> float:
    """Mean over slates of the mean over that slate's contests of the per-player error
    (rows above 0.1 % in either): |log-odds| (E8) or |pp|."""
    per = {}
    for p, y, c in pairs:
        k = (p > 0.001) | (y > 0.001)
        e = np.abs(_logit(p[k]) - _logit(y[k])) if kind == "logit" else np.abs(p[k] - y[k])
        per.setdefault(c["slate_id"], []).append(float(e.mean()))
    return float(np.mean([np.mean(v) for v in per.values()]))


def fit(D, contests, keys, start, arch, kind="logit", maxiter=600):
    from scipy.optimize import minimize
    x0 = np.array([start["conc"][arch] if k == "conc" else start["params"][k] for k in keys])

    def f(th):
        return loss_fn(proxies(D, contests, spec_from(th, keys, start, arch)), kind)
    step = {"conc": 0.25, "w_ours": 0.2, "b_value": 0.3, "b_proj": 0.3}
    sim = [x0] + [x0 + np.eye(len(keys))[i] * step.get(k, 0.4) for i, k in enumerate(keys)]
    r = minimize(f, x0, method="Nelder-Mead",
                 options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6,
                          "initial_simplex": np.array(sim)})
    return spec_from(r.x, keys, start, arch), float(r.fun)


def metrics_of(pairs) -> dict:
    """Per-slate means (each contest once, slates weighted equally) of the own_metrics."""
    per = {}
    for p, y, c in pairs:
        per.setdefault(c["slate_id"], []).append(own_metrics(p, y))
    keys = ("mae_pp", "spearman", "top10_mae_pp", "top10_bias_pp", "ptop10_bias_pp",
            "tail_mae_pp", "logit_mae", "logit_slope")
    sl = {s: {k: float(np.mean([m[k] for m in ms])) for k in keys} for s, ms in per.items()}
    return {"per_slate": sl, **{k: float(np.mean([v[k] for v in sl.values()])) for k in keys}}


# ============================================================================== commands

def cmd_data(args):
    conn = connect(args.db)
    t0 = time.perf_counter()
    D = build_data(conn)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "data.pkl", "wb") as fh:
        pickle.dump(D, fh)
    for sid, S in D["slates"].items():
        print(sid, S["day"], S["run_kind"], S["run_id"], S["run_at"], "lock", S["lock"],
              "pool", len(S["inputs"]))
    for c in D["contests"]:
        print(c["slate_id"], c["contest_id"], c["archetype"], c["entries"], c["name"],
              f"{sum(c['own'].values()):.2f}")
    print(f"{time.perf_counter() - t0:.1f}s")


def cmd_diag(args):
    """Diagnosis tables (OFF and the shipped ON spec, in-sample): per contest, by salary tier,
    public value rank, public projection rank and teammate-out news; residual log-odds
    regression on candidate drivers (SEs clustered by player-slate)."""
    D = load_data(args.out)
    pd.set_option("display.width", 200)
    out = {}
    for tag, cal in (("off", False), ("on", True)):
        T = player_table(D, cal)
        T["value_rank_b"] = pd.cut(T["value_rank"], [0, 5, 10, 20, 40, 999],
                                   labels=["1-5", "6-10", "11-20", "21-40", "41+"])
        T["proj_rank_b"] = pd.cut(T["proj_rank"], [0, 5, 10, 20, 40, 999],
                                  labels=["1-5", "6-10", "11-20", "21-40", "41+"])
        T["gap"] = pd.cut(((T["proj"] - T["fppg"]) / T["fppg"].clip(lower=5)),
                          [-9, -0.1, 0.1, 9], labels=["ours < FPPG", "~", "ours > FPPG"])
        bc = by_contest(T)
        print(f"== {tag}: per contest")
        print(bc.round(3).to_string())
        out[tag] = {"contests": bc.to_dict("records")}
        for col in ("tier", "value_rank_b", "proj_rank_b", "news", "replacement", "gap"):
            g = group_bias(T, col)
            print(f"== {tag}: by {col}")
            print(g.to_string())
            out[tag][col] = g.reset_index().astype({col: str}).to_dict("records")
        if tag == "off":
            out["regression"] = residual_regression(T)
            for arch, rows in out["regression"].items():
                print(f"== residual log-odds regression, {arch}")
                for r in rows:
                    print(f"   {r['term']:<10} {r['b']:+.3f} (SE {r['se']:.3f}) z {r['z']:+.1f}")
            misses = T.assign(err=T["own"] - T["real"]).sort_values("err")
            cols = ["slate_id", "cname", "archetype", "name", "salary", "fppg", "proj",
                    "recent_dk", "proj_min", "recent_min", "news", "value_rank", "own", "real"]
            mm = misses.drop_duplicates(["slate_id", "archetype", "name"]).head(15)[cols]
            print(mm.round(2).to_string())
            out["misses"] = mm.to_dict("records")
    (args.out / "diag.json").write_text(json.dumps(out, indent=1, default=str))


def residual_regression(T: pd.DataFrame) -> dict:
    k = T[(T["own"] > 0.001) | (T["real"] > 0.001)].copy()
    k["res"] = _logit(k["real"].to_numpy()) - _logit(k["own"].to_numpy())
    feats = {
        "lp": lambda g: _logit(g["own"].to_numpy()),
        "ours_gap": lambda g: ((g["proj"] - g["fppg"]) / g["fppg"].clip(lower=5)).clip(-1, 1)
        .fillna(0).to_numpy(),
        "form": lambda g: ((g["recent_dk"] - g["pub_proj"]) / g["pub_proj"].clip(lower=10))
        .clip(-1, 1).fillna(0).to_numpy(),
        "news": lambda g: g["news"].astype(float).to_numpy(),
        "repl": lambda g: g["replacement"].astype(float).to_numpy(),
        "z_salary": lambda g: ((g["salary"] - 6000) / 2000).to_numpy(),
        "bench": lambda g: (g["starter"] == False).astype(float).to_numpy(),  # noqa: E712
    }
    out = {}
    for arch, g in k.groupby("archetype"):
        X = np.column_stack([np.ones(len(g))] + [f(g) for f in feats.values()])
        y = g["res"].to_numpy()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        XtXi = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1],) * 2)
        grp = (g["slate_id"] + ":" + g["pid"]).to_numpy()
        for u in np.unique(grp):
            m = grp == u
            sc = X[m].T @ e[m]
            meat += np.outer(sc, sc)
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        out[arch] = [{"term": t, "b": float(bb), "se": float(ss), "z": float(bb / ss)}
                     for t, bb, ss in zip(["const", *feats], b, se)]
        out[arch + "_n"] = [{"term": "rows", "b": float(len(g)), "se": 1.0, "z": 0.0}]
    return out


def cmd_loso(args):
    """GPP: every variant fit on 2 slates, scored on the third (and in-sample on all 3). Cash
    (10-20 only): conc fit in-sample on the 3 double-ups given the GPP fit."""
    D = load_data(args.out)
    gpp = [c for c in D["contests"] if c["archetype"] != "cash"]
    cash = [c for c in D["contests"] if c["archetype"] == "cash"]
    res = {"off": {"gpp": metrics_of(proxies(D, gpp, None)),
                   "cash": metrics_of(proxies(D, cash, None))}}
    slates = sorted({c["slate_id"] for c in gpp})
    for kind in args.loss.split(","):
        for name, keys in VARIANTS.items():
            if args.variants and name not in args.variants.split(","):
                continue
            t0 = time.perf_counter()
            held_pairs, folds = [], {}
            for s in slates:
                tr = [c for c in gpp if c["slate_id"] != s]
                te = [c for c in gpp if c["slate_id"] == s]
                sp, _ = fit(D, tr, keys, base_spec(), "large_gpp", kind)
                folds[s] = sp
                held_pairs += proxies(D, te, sp)
            sp_all, fun = fit(D, gpp, keys, base_spec(), "large_gpp", kind)
            sp_cash, _ = fit(D, cash, ("conc",), sp_all, "cash", kind)
            row = {"keys": keys, "loss": kind, "spec": sp_cash, "folds": folds,
                   "gpp_loso": metrics_of(held_pairs),
                   "gpp_in": metrics_of(proxies(D, gpp, sp_all)),
                   "cash_in": metrics_of(proxies(D, cash, sp_cash)),
                   "cash_gpp_spec": metrics_of(proxies(D, cash, sp_all)),
                   "seconds": round(time.perf_counter() - t0, 1)}
            res[f"{name}|{kind}"] = row
            print(name, kind, json.dumps({k: round(v, 3) for k, v in row["gpp_loso"].items()
                                          if k != "per_slate"}),
                  "cash_in", json.dumps({k: round(v, 3) for k, v in row["cash_in"].items()
                                         if k != "per_slate"}),
                  "conc", {a: round(v, 3) for a, v in sp_cash["conc"].items()},
                  {k: round(v, 3) for k, v in sp_cash["params"].items()}, row["seconds"], "s",
                  flush=True)
    print("off gpp", json.dumps({k: round(v, 3) for k, v in res["off"]["gpp"].items()
                                 if k != "per_slate"}))
    print("off cash", json.dumps({k: round(v, 3) for k, v in res["off"]["cash"].items()
                                  if k != "per_slate"}))
    (args.out / "loso.json").write_text(json.dumps(res, indent=1, default=float))


def cmd_grid(args):
    """(GPP conc, w_ours) grid: per-slate metrics of every cell (no fitting: a held-out slate's
    cell is out of sample by construction; LOSO picks the cell on the other two slates)."""
    D = load_data(args.out)
    gpp = [c for c in D["contests"] if c["archetype"] != "cash"]
    cash = [c for c in D["contests"] if c["archetype"] == "cash"]
    concs = [round(x, 2) for x in np.arange(0.50, 1.21, 0.05)]
    ws = [round(x, 2) for x in np.arange(0.30, 1.01, 0.05)]
    cells = []
    for cg in concs:
        for w in ws:
            sp = base_spec()
            sp["conc"]["large_gpp"], sp["conc"]["small_gpp"] = cg, cg / 0.9
            sp["params"]["w_ours"] = w
            m = metrics_of(proxies(D, gpp, sp))
            mc = metrics_of(proxies(D, cash, sp))
            cells.append({"conc": cg, "w_ours": w, "gpp": m, "cash": mc,
                          "loss_logit": {s: None for s in m["per_slate"]}})
    (args.out / "grid.json").write_text(json.dumps(cells, default=float))
    slates = sorted({c["slate_id"] for c in gpp})
    for crit in ("mae_pp", "logit_mae"):
        print("criterion", crit)
        held = {}
        for s in slates:
            tr = [x for x in slates if x != s]
            best = min(cells, key=lambda c: np.mean([c["gpp"]["per_slate"][t][crit] for t in tr]))
            held[s] = best
            off = next(c for c in cells if c["conc"] == 0.9 and c["w_ours"] == 0.5)
            print(f"  hold {s}: pick conc {best['conc']} w {best['w_ours']}; held-out",
                  {k: (round(off["gpp"]["per_slate"][s][k], 3),
                       round(best["gpp"]["per_slate"][s][k], 3))
                   for k in ("mae_pp", "spearman", "logit_mae", "top10_bias_pp",
                             "ptop10_bias_pp", "logit_slope")})
        allb = min(cells, key=lambda c: c["gpp"][crit])
        print("  all 3:", allb["conc"], allb["w_ours"])


def _draws(op, sd: float, R: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = F.own_sd_for(op, None, sd)
    return np.array([F.perturb_ownership(op, v, rng).own for _ in range(R)])


def _interp(xs, ys, target):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    i = int(np.searchsorted(ys, target))
    if i <= 0:
        return float(xs[0])
    if i >= len(xs):
        return float(xs[-1])
    t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(xs[i - 1] + t * (xs[i] - xs[i - 1]))


def noise_check(op, contests, draws: np.ndarray) -> dict:
    """Real ownership against R noisy draws of the proxy: 90 % / 50 % interval coverage and the
    mean PIT (players above 0.1 % in proxy or real), and the top-10 ownership sum (draws' mean
    vs real: does the noisy field reach the real chalk?)."""
    cov90, cov50, pit, t10d, t10r, ssd, ssr = [], [], [], [], [], [], []
    for c in contests:
        y = np.array([c["own"].get(p, 0.0) for p in op.pids])
        k = (op.own > 0.001) | (y > 0.001)
        lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
        q1, q3 = np.quantile(draws, [0.25, 0.75], axis=0)
        cov90.append(float(((y >= lo) & (y <= hi))[k].mean()))
        cov50.append(float(((y >= q1) & (y <= q3))[k].mean()))
        pit.append(float(((draws < y).mean(0) + 0.5 * (draws == y).mean(0))[k].mean()))
        t10d.append(float(np.mean(np.sort(draws, axis=1)[:, ::-1][:, :10].sum(1))))
        t10r.append(float(np.sort(y)[::-1][:10].sum()))
        ssd.append(float(np.mean((draws ** 2).sum(1))))
        ssr.append(float((y ** 2).sum()))
    return {"cov90": float(np.mean(cov90)), "cov50": float(np.mean(cov50)),
            "pit_mean": float(np.mean(pit)), "top10_sum_draws": float(np.mean(t10d)),
            "top10_sum_real": float(np.mean(t10r)), "sumsq_draws": float(np.mean(ssd)),
            "sumsq_real": float(np.mean(ssr))}


SD_GRID = tuple(round(x, 2) for x in np.arange(0.2, 2.01, 0.1))


def cmd_noise(args):
    """D5 noise refit (OWN_SD, large GPP) per slate for the OFF and the ON proxy: the sd whose
    draws reproduce the slate's real MAE (as research/field_k.py fit), then leave-one-slate-out
    coverage of the real ownership by draws at the sd fit on the other two slates."""
    D = load_data(args.out)
    gpp = [c for c in D["contests"] if c["archetype"] != "cash"]
    slates = sorted({c["slate_id"] for c in gpp})
    res = {}
    for tag, spec in (("off", False), ("on", True)):
        ops = {s: F.ownership_proxy(D["slates"][s]["inputs"], D["slates"][s]["pool"],
                                    archetype="large_gpp", cal=spec) for s in slates}
        fit_sd = {}
        for s in slates:
            cs = [c for c in gpp if c["slate_id"] == s]
            real = np.mean([own_metrics(ops[s].own, np.array([c["own"].get(p, 0.0)
                                                              for p in ops[s].pids]))["mae_pp"]
                            for c in cs])
            curve = []
            for sd in SD_GRID:
                dr = _draws(ops[s], sd, args.draws, 11)
                curve.append(np.mean([own_metrics(ops[s].own, d)["mae_pp"] for d in dr]))
            fit_sd[s] = _interp(SD_GRID, curve, real)
            print(tag, s, "real MAE %.2f" % real, "sd %.2f" % fit_sd[s], flush=True)
        rows = {}
        for s in slates:
            sd_tr = float(np.mean([fit_sd[t] for t in slates if t != s]))
            cs = [c for c in gpp if c["slate_id"] == s]
            for name, sd in (("loso", sd_tr), ("own_sd_1.2", F.OWN_SD["large_gpp"]),
                             ("in_slate", fit_sd[s])):
                rows.setdefault(name, {})[s] = {"sd": sd, **noise_check(
                    ops[s], cs, _draws(ops[s], sd, 400, 5))}
        res[tag] = {"fit_sd": fit_sd, "sd_all": float(np.mean(list(fit_sd.values()))),
                    "check": rows}
        for name, r in rows.items():
            print(tag, name, {s: {k: round(v, 3) for k, v in x.items()} for s, x in r.items()})
    # cash: one slate (10-20), in-sample only
    cash = [c for c in D["contests"] if c["archetype"] == "cash"]
    S = D["slates"][cash[0]["slate_id"]]
    opc = F.ownership_proxy(S["inputs"], S["pool"], archetype="cash", cal=False)
    res["cash_in_sample"] = {str(sd): noise_check(opc, cash, _draws(opc, sd, 400, 5))
                             for sd in (1.5, 2.0, 2.65, 3.0)}
    for sd, r in res["cash_in_sample"].items():
        print("cash", sd, {k: round(v, 3) for k, v in r.items()})
    (args.out / "noise.json").write_text(json.dumps(res, indent=1, default=float))


CONFIGS = {"off": (False, 0), "on": (True, 0), "off+k": (False, 4), "on+k": (True, 4)}


def _wq(scores, w, frac):
    o = np.argsort(-scores)
    s, ww = scores[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    return float(s[min(len(s) - 1, int(np.searchsorted(cw, frac)))])


def cmd_lines(args):
    """The modelled field of each real contest scored at the actual box scores: cash line (the
    paid fraction) and top-1 % line vs the real standings, plus the field's realised top-10
    ownership and Σ own² vs real. K fields: lines of the pooled mixture and the between-field
    sd of the cash line."""
    import real_standings_calib as RS
    from courtside.proj.dfs import real_calib as RC
    from courtside.proj.dfs import standings as ST2
    conn = connect(args.db)
    D = load_data(args.out)
    hist = ST2.history_contests(conn)
    res = []
    acts = {}
    for c in D["contests"]:
        S = D["slates"][c["slate_id"]]
        if c["slate_id"] not in acts:
            acts[c["slate_id"]] = RS.actual_points(conn, c["slate_id"], S["run_id"], S["pool"])
        act = acts[c["slate_id"]]
        info = RC.contest_info(conn, c["contest_id"], hist)
        M = c["entries"]
        paid = info["places_paid"] or (F.du_paid_places(M) if c["archetype"] == "cash" else None)
        pts = c["points"]
        real = {"cash": float(pts[paid - 1]), "top1": float(pts[max(1, math.ceil(0.01 * M)) - 1]),
                "top10_own": float(np.sort(list(c["own"].values()))[::-1][:10].sum()),
                "sumsq": float((np.array(list(c["own"].values())) ** 2).sum())}
        contest = F.Contest(c["contest_id"], c["name"], 1.0, M,
                            [{"min_pos": 1, "max_pos": paid, "payout": 2.0}], c["archetype"],
                            "reeval")
        row = {"contest_id": c["contest_id"], "slate": c["slate_id"], "name": c["name"],
               "type": c["archetype"], "M": M, "paid": paid, "real": real}
        cfgs = dict(CONFIGS)
        if args.fold_spec:                  # a leave-one-slate-out spec: "conc,w_ours"
            cg, w = (float(x) for x in args.fold_spec.split(","))
            sp = json.loads(json.dumps(F.OWN_CAL_SPEC))
            sp["conc"]["large_gpp"], sp["conc"]["small_gpp"] = cg, cg / 0.9
            sp["params"]["w_ours"] = w
            cfgs = {"fold": (sp, 0), "fold+k": (sp, 4)}
        if args.slates and c["slate_id"] not in args.slates.split(","):
            continue
        for name, (cal, k) in cfgs.items():
            if args.variants and name not in args.variants.split(","):
                continue
            reps = []
            for sd in range(args.seeds):
                op = F.ownership_proxy(S["inputs"], S["pool"], archetype=c["archetype"], cal=cal)
                if k:
                    fs = F.build_fields(S["pool"], S["inputs"], contest, k=k, ownership=op,
                                        seed=sd)
                    fl = fs.fields
                else:
                    fl = [F.build_field(S["pool"], S["inputs"], contest, ownership=op, seed=sd)]
                scs, ws, per_cash, own_r = [], [], [], []
                for f in fl:
                    a = np.array([act.get(p, 0.0) for p in f.pids])
                    sc = a[f.lineups].sum(1)
                    w = f.counts.astype(float)
                    scs.append(sc)
                    ws.append(w / w.sum())
                    per_cash.append(_wq(sc, w, paid / M))
                    own_r.append(f.realized_ownership())
                sc, w = np.concatenate(scs), np.concatenate(ws)
                ro = np.mean([np.sort(o)[::-1][:10].sum() for o in own_r])
                ss = np.mean([(o ** 2).sum() for o in own_r])
                reps.append({"cash": _wq(sc, w, paid / M), "top1": _wq(sc, w, 0.01),
                             "cash_between_sd": float(np.std(per_cash)) if k else 0.0,
                             "top10_own": float(ro), "sumsq": float(ss),
                             "version": fl[0].version})
            row[name] = {kk: (float(np.mean([r[kk] for r in reps])) if kk != "version"
                              else reps[0][kk]) for kk in reps[0]}
        res.append(row)
        print(json.dumps({"c": c["contest_id"], "type": c["archetype"], "real": {
            k2: round(v, 2) for k2, v in real.items()}, **{n: {k2: (round(v, 2) if
                                                               isinstance(v, float) else v)
                                                               for k2, v in row[n].items()}
                                                           for n in cfgs if n in row}}),
              flush=True)
    name = "lines_fold.json" if args.fold_spec else "lines.json"
    (args.out / name).write_text(json.dumps(res, indent=1, default=float))


def _local_sims(out: Path):
    """Point `field.slate_inputs`' sim_path at the local copies in <out>/sims (the stored paths
    are the Mac's)."""
    orig = F.slate_inputs

    def patched(*a, **kw):
        pool, inp, info = orig(*a, **kw)
        sp = info.get("sim_path")
        if sp:
            loc = out / "sims" / Path(sp).parent.name / Path(sp).name
            info = {**info, "sim_path": str(loc) if loc.exists() else
                    (sp if Path(sp).exists() else None)}
        return pool, inp, info
    F.slate_inputs = patched
    return orig


def cmd_chances(args):
    """`real_calib.build_report` with modelled-field sims (`--sims`), per config: simulated
    P(cash) of the owner's real entries vs the real in-the-money rate, and the PITs."""
    from courtside.proj.dfs import real_calib as RC
    _local_sims(args.out)
    conn = connect(args.db)
    orig_ff = F.field_for
    res = {}
    for name, (cal, k) in CONFIGS.items():
        if args.variants and name not in args.variants.split(","):
            continue
        F.set_own_cal(cal)

        def ff(ctx, contest, k_=k, **kw):
            kw["k"] = k_
            return orig_ff(ctx, contest, **kw)
        F.field_for = ff
        t0 = time.perf_counter()
        rep = RC.build_report(conn, sims=args.sims, ownership=False, until="2026-10-24")
        F.field_for = orig_ff
        F.set_own_cal(False)
        sm = rep["summary"]
        per = []
        for b in rep["slates"]:
            for c in b["contests"]:
                sim = c.get("sim") or {}
                for e in c["entries"]:
                    d = (sim.get("lineups") or {}).get(str(e["lineup_id"])) or {}
                    if d:
                        per.append({"slate": b["slate_id"], "contest": c["contest_id"],
                                    "type": c["type"], "cashed": e["cashed"],
                                    "p_cash": d.get("p_cash"), "pit_line": d.get("pit_line"),
                                    "pit_rank": d.get("pit_rank"),
                                    "sim_line_p50": (sim.get("sim_cash_line") or {}).get("p50"),
                                    "real_line": (c.get("cash_line") or {}).get("value")})
        res[name] = {"summary": {k2: sm[k2] for k2 in ("du_pit_line", "du_cash_rate",
                                                       "du_sim_p_cash", "gpp_pit_rank",
                                                       "gpp_cash_rate", "gpp_sim_p_cash",
                                                       "gpp_mean_beat")},
                     "entries": per, "seconds": round(time.perf_counter() - t0, 1)}
        print(name, json.dumps(res[name]["summary"], default=float), res[name]["seconds"], "s",
              flush=True)
    (args.out / "chances.json").write_text(json.dumps(res, indent=1, default=float))


BUILDS = (("155547", "197052780", "685785c2c04b"),     # $40K Sharpshooter, 20 max, lock10
          ("155576", "197082729", "1f04e52b1979"))     # $50K Four Point Play, 20 max, lock10


def cmd_build(args):
    """One contest-builder build per slate (`portfolio.build_for_slate`, N = 20, low sims, no
    curve / strengths, not saved) with the recalibration OFF vs ON: the chosen set's simulated
    chances and how the lineups differ."""
    from courtside.proj.dfs import portfolio as PF
    _local_sims(args.out)
    conn = connect(args.db)
    res = {}
    builds = [tuple(x.split(":")) for x in args.builds.split(",")] if args.builds else BUILDS
    for sid, cid, run in builds:
        r = conn.execute("SELECT created_at FROM dfs_runs WHERE run_id=?", (run,)).fetchone()
        from datetime import datetime
        now = datetime.fromisoformat(r[0].replace("Z", "+00:00"))
        for name in args.variants.split(",") if args.variants else ("off", "on"):
            cal, k = CONFIGS[name]
            F.set_own_cal(cal)
            F.set_field_k(k)
            s_ = PF.Settings(n=args.n, n_sims=args.sims, curve=args.full,
                             strengths=args.full, threads=2)
            t0 = time.perf_counter()
            pf = PF.build_for_slate(conn, sid, cid, s_, run_id=run, now=now)
            F.set_own_cal(False)
            F.set_field_k(0)
            sm = pf.summary
            keys = [tuple(sorted(d["player_id"] for d in lu)) for lu in pf.slots()]
            pc = [x.get("p_cash") for x in pf.lineups if x.get("p_cash") is not None]
            res[f"{sid}|{name}"] = {"summary": {k2: sm.get(k2) for k2 in (
                "p_any_cash", "p_any_top1", "e_cashing", "e_payout", "field")},
                "mean_lineup_p_cash": float(np.mean(pc)) if pc else None,
                "lineups": keys, "seconds": round(time.perf_counter() - t0, 1)}
            print(sid, name, json.dumps({**{k2: sm.get(k2) for k2 in (
                "p_any_cash", "p_any_top1", "e_cashing")}, "mean_lineup_p_cash":
                res[f"{sid}|{name}"]["mean_lineup_p_cash"]}, default=float),
                  res[f"{sid}|{name}"]["seconds"], "s", flush=True)
        import real_standings_calib as RS
        pool = F.slate_inputs(conn, sid, run)[0]
        act = RS.actual_points(conn, sid, run, pool)
        for name in args.variants.split(",") if args.variants else ("off", "on"):
            sc = [sum(act.get(p, 0.0) for p in lu) for lu in res[f"{sid}|{name}"]["lineups"]]
            res[f"{sid}|{name}"]["actual"] = sc
            print(sid, name, "actual mean %.1f best %.1f" % (np.mean(sc), max(sc)), flush=True)
        a, b = res[f"{sid}|off"]["lineups"], res[f"{sid}|on"]["lineups"]
        print(sid, "lineups in both", len(set(a) & set(b)), "of", len(a), flush=True)
    (args.out / "build.json").write_text(json.dumps(res, indent=1, default=str))


def load_data(out: Path) -> dict:
    with open(out / "data.pkl", "rb") as fh:
        return pickle.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "diag", "loso", "grid", "noise", "lines",
                                       "chances", "build"])
    ap.add_argument("--db", default="var/courtside.db")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--loss", default="logit,pp")
    ap.add_argument("--variants", default="")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--fold-spec", default="")
    ap.add_argument("--slates", default="")
    ap.add_argument("--builds", default="", help="sid:contest:run,... (default BUILDS)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--full", action="store_true", help="builder with curve + strengths")
    args = ap.parse_args(argv)
    return {"data": cmd_data, "diag": cmd_diag, "loso": cmd_loso, "grid": cmd_grid,
            "noise": cmd_noise, "lines": cmd_lines,
            "chances": cmd_chances, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
