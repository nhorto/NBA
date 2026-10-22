"""Lock-10 Questionable P(play) refit + a "returning" split for Questionable (HANDOFF next #2).

Docs: docs/research/pplay-q-refit.md.  Code under test: `availability.PPLAYQ` (switch
COURTSIDE_PPLAYQ=1, default OFF), `availability.q_refit_p`.

    python research/pplay_q_refit.py samples    # build + cache candidate rows (local DB, read-only)
    python research/pplay_q_refit.py diagnose   # step 1: predicted vs observed by cell, why
    python research/pplay_q_refit.py evaluate   # step 2: walk-forward on/off, log loss / Brier
    python research/pplay_q_refit.py fit        # write src/courtside/proj/pplay_q_params.json
                                                # (+ $PQ_OUT/pq_lomo.json for the slice)
    python research/pplay_q_refit.py identity --ref OLD_availability.py --day D --slate ID
                                                # OFF == reference p_play_slate frames
    python research/pplay_q_refit.py slice live10   # T2.4 slice arm (COURTSIDE_PPLAYQ=1 = ON)

Everything reads the DB through a read-only URI.  Rows are cached in $PQ_OUT (default
var/tmp/pq).  Only `slice` simulates (4,000 sims; ~5 min per arm on the Mac scratch box).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj import availability as AV

OUT = Path(os.environ.get("PQ_OUT", config.REPO_ROOT / "var" / "tmp" / "pq"))
LO, HI = AV.TRAIN_LO, AV.RS_END
OFFSETS = (10, 45, 300)


def say(m: str) -> None:
    print(f"[pq] {time.strftime('%H:%M:%S')} {m}", flush=True)


def ro_conn(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.db_path()
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


# ----------------------------------------------------------------------------- samples

def cmd_samples(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = ro_conn(a.db)
    t0 = time.time()
    hist = AV.load_history(conn, LO)
    say(f"history {time.time() - t0:.0f}s")
    T = AV.build_samples(conn, LO, HI, mode="bucket", hist=hist)
    say(f"bucket rows {len(T)} ({time.time() - t0:.0f}s)")
    S = AV.build_samples(conn, LO, HI, mode="scenario", offsets_min=OFFSETS, hist=hist)
    say(f"scenario rows {len(S)} ({time.time() - t0:.0f}s)")
    for name, df in (("T", T), ("S", S)):
        df = df.copy()
        for c in ("at", "tip"):
            if c in df:
                df[c] = df[c].astype(str)
        df.to_parquet(OUT / f"{name}.parquet")
    say(f"wrote {OUT}")


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{name}.parquet")
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


# ----------------------------------------------------------------------------- enrichment

def starts_frame(conn) -> pd.DataFrame:
    lo = (LO - timedelta(days=400)).isoformat()
    pg = pd.read_sql_query(
        "SELECT pg.game_id, pg.player_id, pg.team, pg.game_date, pg.min, "
        "(ps.player_id IS NOT NULL) AS started FROM player_games pg LEFT JOIN player_starts ps "
        "ON ps.game_id = pg.game_id AND ps.player_id = pg.player_id "
        "WHERE pg.game_date >= ? AND pg.min > 0", conn, params=(lo,))
    return pg.sort_values(["game_date", "game_id"], kind="mergesort")


def enrich(df: pd.DataFrame, conn) -> pd.DataFrame:
    """+ started (actual box starter), start10 (start share of the last 10 appearances strictly
    before the day), missed (team games since his last appearance, strictly before the day),
    lineup_known (the game has 10 recorded starters)."""
    pg = starts_frame(conn)
    hist = AV.load_history(conn, LO)
    by_p = {int(p): (g["game_date"].tolist(), g["started"].astype(int).tolist())
            for p, g in pg.groupby("player_id", sort=False)}
    starters = set(zip(pg.loc[pg["started"] == 1, "game_id"], pg.loc[pg["started"] == 1, "player_id"]))
    n_st = pg[pg["started"] == 1].groupby("game_id").size().to_dict()
    from bisect import bisect_left
    s10, miss = [], []
    for pid, team, d in zip(df["player_id"], df["team"], df["day"]):
        ds = d.isoformat()
        rec = by_p.get(int(pid))
        if rec:
            i = bisect_left(rec[0], ds)
            v = rec[1][max(0, i - 10):i]
            s10.append(float(np.mean(v)) if v else np.nan)
        else:
            s10.append(np.nan)
        miss.append(AV.games_missed(hist, int(pid), team, d))
    out = df.copy()
    out["start10"] = s10
    out["missed"] = miss
    out["started"] = [int((g, int(p)) in starters) for g, p in zip(out["game_id"], out["player_id"])]
    out["lineup_known"] = [n_st.get(g, 0) >= 10 for g in out["game_id"]]
    return out


# ----------------------------------------------------------------------------- folds

def month(d: date) -> str:
    return d.strftime("%Y-%m")


def cells_p(train_T: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    prm = AV.fit_from_samples(train_T, m1=20.0, m=AV.DEFAULT_M)
    return AV.predict(rows, prm)["p_play"].to_numpy()


def oof_cell(T: pd.DataFrame, S: pd.DataFrame, scheme: str) -> np.ndarray:
    """Report-cell P for every S row, never fit on the row's own month.
    scheme 'lomo' = every other month (what the T2.4 slice does); 'fwd' = months before only
    (rows of the first month get NaN)."""
    out = np.full(len(S), np.nan)
    tm = np.array([month(d) for d in T["day"]])
    sm = np.array([month(d) for d in S["day"]])
    for m in sorted(set(sm)):
        tr = T[tm != m] if scheme == "lomo" else T[tm < m]
        if scheme == "fwd" and tr.empty:
            continue
        idx = np.where(sm == m)[0]
        out[idx] = cells_p(tr, S.iloc[idx])
    return out


def clustered(x: np.ndarray, groups) -> tuple[float, float]:
    """mean and cluster-robust SE of x (clusters = groups)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = x.mean()
    g = pd.Series(x - m).groupby(np.asarray(groups)).sum()
    return float(m), float(np.sqrt((g ** 2).sum()) / n)


def gap_table(df: pd.DataFrame, pcol: str, by: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(by, observed=True, dropna=False):
        gap, se = clustered(g[pcol] - g["played"], g["day"])
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), n=len(g),
                         p=g[pcol].mean(), played=g["played"].mean(), gap=gap, se=se))
    return pd.DataFrame(rows)


def lead_rows(conn=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regular-season T (bucket) and enriched S (scenario) rows, with oof cell P columns."""
    cache = OUT / "S_enriched.parquet"
    T = load("T")
    T = T[T["season_type"] == "Regular Season"].reset_index(drop=True)
    if cache.exists():
        S = pd.read_parquet(cache)
        S["day"] = pd.to_datetime(S["day"]).dt.date
        return T, S
    S = load("S")
    S = S[S["season_type"] == "Regular Season"].reset_index(drop=True)
    S = enrich(S, conn or ro_conn())
    S["p_lomo"] = oof_cell(T, S, "lomo")
    S["p_fwd"] = oof_cell(T, S, "fwd")
    S.to_parquet(cache)
    return T, S


def with_override(S: pd.DataFrame, pcol: str) -> np.ndarray:
    """The live path at lock-10: DK-flagged starters get 0.99 (here: the box starters)."""
    p = S[pcol].to_numpy().copy()
    m = (S["scenario"] == "T-10m").to_numpy() & (S["started"] == 1).to_numpy()
    p[m] = 0.99
    return p


def cmd_diagnose(a) -> None:
    T, S = lead_rows()
    S = S.assign(p_live=with_override(S, "p_lomo"))
    pd.set_option("display.width", 200)
    L = S[S["status"].isin(["questionable", "probable", "available", "doubtful", "removed"])]
    print("== listed, by scenario x status (LOMO cells; lock-10 starters at 0.99)")
    print(gap_table(L, "p_live", ["scenario", "status"]).round(3).to_string(index=False))
    q10 = L[(L["scenario"] == "T-10m")]
    print("\n== lock-10, by status x started")
    print(gap_table(q10, "p_live", ["status", "started"]).round(3).to_string(index=False))
    ns = q10[q10["started"] == 0].copy()
    ns["usual"] = np.where(ns["start10"] >= 0.5, "usual_starter", "usual_bench")
    print("\n== lock-10 non-starters by status x usual start x role")
    print(gap_table(ns, "p_live", ["status", "usual"]).round(3).to_string(index=False))
    print(gap_table(ns, "p_live", ["status", "role"]).round(3).to_string(index=False))
    print(gap_table(ns[ns.status == "questionable"], "p_live", ["bucket", "reason"]).round(3).to_string(index=False))
    print("\n== Q by scenario x bucket")
    print(gap_table(L[L.status == "questionable"], "p_live", ["scenario", "bucket"]).round(3).to_string(index=False))
    q45 = L[(L["scenario"] == "T-45m") & (L["status"] == "questionable")].copy()
    for N in (1, 2, 3, 5, 8, 10, 15):
        q45[f"ret{N}"] = q45["missed"] >= N
        t = gap_table(q45, "p_lomo", [f"ret{N}"])
        print(f"\n== lock-45 Q returning (missed >= {N})\n" + t.round(3).to_string(index=False))
    q45["role"] = q45["role"]
    q45["ret3"] = q45["missed"] >= 3
    print(gap_table(q45, "p_lomo", ["ret3", "role"]).round(3).to_string(index=False))
    print(gap_table(q45, "p_lomo", ["ret3", "reason"]).round(3).to_string(index=False))


# ----------------------------------------------------------------------------- evaluation

LISTED_EVAL = ("questionable", "probable", "available", "doubtful", "removed", "daytoday")
RET_GRID = (1, 2, 3, 5, 8)


def prep(S: pd.DataFrame) -> pd.DataFrame:
    """Lineups are known at lock-10 only (the live lock-10 run has the DK starter flags; the
    T2.4 live10 protocol uses the box starters for them)."""
    S = S[S["status"].isin(LISTED_EVAL)].copy()
    S["lineup_known"] = (S["scenario"] == "T-10m").to_numpy() & S["lineup_known"].to_numpy()
    S["started"] = S["started"].astype(bool) & S["lineup_known"]
    S["month"] = [month(d) for d in S["day"]]
    return S.reset_index(drop=True)


def override(p: np.ndarray, S: pd.DataFrame) -> np.ndarray:
    p = np.asarray(p, float).copy()
    p[S["started"].to_numpy()] = 0.99
    return p


def logloss_vec(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), AV.CLAMP[0], AV.CLAMP[1])
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_rows_mask(S: pd.DataFrame, groups) -> np.ndarray:
    """Rows that train the offsets: ns groups at lock-10, ret groups at lock-45 only (one row
    per player-game per group family)."""
    g = np.array([x or "" for x in groups], object)
    sc = S["scenario"].to_numpy()
    ns = np.char.startswith(g.astype(str), "ns|") & (sc == "T-10m")
    rt = np.char.startswith(g.astype(str), "ret|") & (sc == "T-45m")
    return ns | rt


def inner_cells(T: pd.DataFrame, S: pd.DataFrame, months: list[str]) -> np.ndarray:
    """Cell P for S rows in `months`, each month predicted from the other `months` only."""
    out = np.full(len(S), np.nan)
    tm = np.array([month(d) for d in T["day"]])
    for m in months:
        idx = np.where(S["month"].to_numpy() == m)[0]
        if len(idx):
            tr = T[np.isin(tm, [x for x in months if x != m])]
            out[idx] = cells_p(tr, S.iloc[idx])
    return out


def choose_ret_n(S_tr: pd.DataFrame, p_tr: np.ndarray, months: list[str], ridge: float) -> int:
    """ret_n by leave-one-month-out log loss inside the training months (lock-45 Q rows)."""
    if len(months) < 2:
        return AV.Q_RET_N
    best, best_ll = AV.Q_RET_N, np.inf
    sel = ((S_tr["scenario"] == "T-45m") & (S_tr["status"] == "questionable")).to_numpy()
    for n in RET_GRID:
        g = np.array([x or "" for x in AV.q_groups(S_tr, n)], object)
        ll = []
        for m in months:
            te = sel & (S_tr["month"].to_numpy() == m)
            tr = sel & (S_tr["month"].to_numpy() != m) & np.char.startswith(g.astype(str), "ret|")
            offs, _ = AV.fit_q_offsets(p_tr[tr], S_tr["played"].to_numpy()[tr], g[tr], ridge)
            prm = AV.QRefitParams(offsets=offs, ret_n=n)
            ll.append(logloss_vec(S_tr["played"].to_numpy()[te],
                                  AV.q_apply(p_tr[te], list(g[te]), prm)))
        v = float(np.concatenate(ll).mean())
        if v < best_ll - 1e-12:
            best, best_ll = n, v
    return best


def walk(T: pd.DataFrame, S: pd.DataFrame, scheme: str, ridge: float = AV.Q_RIDGE,
         ret_n: int | None = None, families=("ns", "ret"), inner: bool = False,
         relabel=None) -> pd.DataFrame:
    """Out-of-sample base / new P for every S row (scheme 'lomo' or 'fwd')."""
    months = sorted(S["month"].unique())
    base = np.full(len(S), np.nan)
    new = np.full(len(S), np.nan)
    grp = np.empty(len(S), object)
    info = []
    for m in months:
        tr_m = [x for x in months if (x != m if scheme == "lomo" else x < m)]
        if not tr_m:
            continue
        te = np.where(S["month"].to_numpy() == m)[0]
        tm = np.array([month(d) for d in T["day"]])
        p_te = cells_p(T[np.isin(tm, tr_m)], S.iloc[te])
        S_tr = S[S["month"].isin(tr_m)].reset_index(drop=True)
        # offsets are fit against the training months' own cells (in-sample): the effects are
        # splits inside a cell (started / returning), which the cells cannot absorb, and a
        # nested fit leaves the first forward folds with prior-only cells.
        p_tr = (inner_cells(T, S_tr, tr_m) if inner else
                cells_p(T[np.isin(tm, tr_m)], S_tr))
        n = ret_n if ret_n is not None else choose_ret_n(S_tr, p_tr, tr_m, ridge)
        g_tr = AV.q_groups(S_tr, n)
        g_tr = [x if x and x.split("|")[0] in families else None for x in g_tr]
        if relabel:
            g_tr = [relabel(x) for x in g_tr]
        fm = fit_rows_mask(S_tr, g_tr)
        offs, ns = AV.fit_q_offsets(p_tr[fm], S_tr["played"].to_numpy()[fm],
                                    np.array(g_tr, object)[fm], ridge)
        prm = AV.QRefitParams(offsets=offs, n=ns, ret_n=n)
        g_te = AV.q_groups(S.iloc[te], n)
        g_te = [x if x and x.split("|")[0] in families else None for x in g_te]
        if relabel:
            g_te = [relabel(x) for x in g_te]
        base[te] = p_te
        new[te] = AV.q_apply(p_te, g_te, prm)
        grp[te] = g_te
        info.append({"month": m, "train": tr_m, "ret_n": n,
                     "offsets": {k: round(v, 3) for k, v in offs.items()}, "n": ns})
    out = S.assign(p_base=override(base, S), p_new=override(new, S), group=grp)
    out.attrs["info"] = info
    return out[np.isfinite(base)].reset_index(drop=True)


def paired_ll(R: pd.DataFrame) -> dict:
    y = R["played"].to_numpy()
    lb, ln = logloss_vec(y, R["p_base"]), logloss_vec(y, R["p_new"])
    bb, bn = (R["p_base"] - y) ** 2, (R["p_new"] - y) ** 2
    d_ll, se_ll = clustered(ln - lb, R["day"])
    d_br, se_br = clustered(bn - bb, R["day"])
    return {"n": len(R), "ll_base": lb.mean(), "ll_new": ln.mean(), "d_ll": d_ll, "se_ll": se_ll,
            "z_ll": d_ll / se_ll if se_ll else np.nan, "br_base": bb.mean(), "br_new": bn.mean(),
            "d_br": d_br, "se_br": se_br}


def calib(R: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in R.groupby(by, observed=True, dropna=False):
        gb, sb = clustered(g["p_base"] - g["played"], g["day"])
        gn, sn = clustered(g["p_new"] - g["played"], g["day"])
        y = g["played"].to_numpy()
        dl, sl = clustered(logloss_vec(y, g["p_new"]) - logloss_vec(y, g["p_base"]), g["day"])
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)), n=len(g),
                         played=y.mean(), p_base=g["p_base"].mean(), p_new=g["p_new"].mean(),
                         gap_base=gb, se_b=sb, gap_new=gn, se_n=sn, d_ll=dl, se_dll=sl))
    return pd.DataFrame(rows)


def report_walk(R: pd.DataFrame, title: str) -> str:
    out = [f"\n#### {title}"]
    for i in R.attrs.get("info", []):
        out.append(f"  fold {i['month']}: ret_n={i['ret_n']} offsets={i['offsets']}")
    rows = []
    for sc in ("T-10m", "T-45m", "T-300m", "all"):
        x = R if sc == "all" else R[R["scenario"] == sc]
        rows.append({"scenario": sc, **paired_ll(x)})
        xq = x[x["status"] == "questionable"]
        rows.append({"scenario": sc + " Q", **paired_ll(xq)})
    out.append(pd.DataFrame(rows).round(4).to_string(index=False))
    R = R.assign(grp=R["group"].fillna("-"))
    out.append("\ncalibration by refit group x scenario:")
    out.append(calib(R, ["scenario", "grp"]).round(3).to_string(index=False))
    out.append("\ncalibration by status x role x started (all scenarios):")
    out.append(calib(R, ["scenario", "status", "started"]).round(3).to_string(index=False))
    return "\n".join(out)


def cmd_evaluate(a) -> None:
    T, S = lead_rows()
    S = prep(S)
    pd.set_option("display.width", 220)
    txt = []
    res = {}
    for scheme in ("fwd", "lomo"):
        R = walk(T, S, scheme)
        res[scheme] = R
        txt.append(report_walk(R, f"{scheme}: both families, ridge {AV.Q_RIDGE}, ret_n nested"))
        R.drop(columns=[c for c in R.columns if R[c].dtype == object and c not in
                        ("status", "role", "scenario", "group", "reason", "bucket", "team",
                         "game_id", "month")]).to_parquet(OUT / f"walk_{scheme}.parquet")
    for fam in (("ns",), ("ret",)):
        R = walk(T, S, "fwd", families=fam)
        txt.append(report_walk(R, f"fwd: families {fam} only"))
    variants = {
        "ret pooled over roles": lambda x: "ret|q" if x and x.startswith("ret|") else x,
        "ret pre-absence >= 15 min (rotation+starter; return-minutes cut)":
            lambda x: (None if x == "ret|q|bench" else "ret|q|pre15")
            if x and x.startswith("ret|") else x,
    }
    for name, f in variants.items():
        for n in (None, 3):
            R = walk(T, S, "fwd", families=("ret",), relabel=f, ret_n=n)
            txt.append(report_walk(R, f"fwd: {name}, ret_n {n or 'nested'}").split("calibration by status")[0])
    for ridge in (1.0, 2.0, 8.0, 16.0):
        R = walk(T, S, "fwd", ridge=ridge)
        x = paired_ll(R)
        txt.append(f"\nfwd ridge {ridge}: d_ll {x['d_ll']:.4f} ({x['se_ll']:.4f}) z {x['z_ll']:.2f}")
    for n in RET_GRID:
        R = walk(T, S, "fwd", ret_n=n, families=("ret",))
        x = paired_ll(R[R["scenario"] == "T-45m"])
        txt.append(f"fwd ret only, ret_n {n}: lock-45 d_ll {x['d_ll']:.4f} ({x['se_ll']:.4f})")
    s = "\n".join(txt)
    print(s)
    (OUT / "evaluate.txt").write_text(s)


# ----------------------------------------------------------------------------- fit / export

SHIP_FAMILIES = ("ns",)        # "ret" (returning Questionable) did not pass walk-forward


def fit_params(T: pd.DataFrame, S: pd.DataFrame, months: list[str] | None = None,
               families=SHIP_FAMILIES, meta: dict | None = None) -> AV.QRefitParams:
    """Offsets fit on the S rows of `months` (default all) against those months' cells."""
    months = months or sorted(S["month"].unique())
    tm = np.array([month(d) for d in T["day"]])
    S_tr = S[S["month"].isin(months)].reset_index(drop=True)
    p_tr = cells_p(T[np.isin(tm, months)], S_tr)
    g = [x if x and x.split("|")[0] in families else None for x in AV.q_groups(S_tr, AV.Q_RET_N)]
    fm = fit_rows_mask(S_tr, g)
    offs, ns = AV.fit_q_offsets(p_tr[fm], S_tr["played"].to_numpy()[fm], np.array(g, object)[fm],
                                AV.Q_RIDGE)
    return AV.QRefitParams(offsets=offs, n=ns, ret_n=AV.Q_RET_N, meta=meta or {})


def cmd_fit(a) -> None:
    T, S = lead_rows()
    S = prep(S)
    meta = {"fit_lo": LO.isoformat(), "fit_hi": HI.isoformat(), "families": list(SHIP_FAMILIES),
            "ridge": AV.Q_RIDGE, "lineup_min": AV.Q_LINEUP_MIN,
            "rows": "lock-10 listed non-starters (box starters = DK flags), 2025-26 regular season",
            "cells": "availability_params.json (fit on the same window)",
            "source": "research/pplay_q_refit.py fit; docs/research/pplay-q-refit.md"}
    prm = fit_params(T, S, meta=meta)
    print(json.dumps(prm.to_json(), indent=1))
    if not a.no_write:
        print("wrote", prm.save())
    lomo = {}
    months = sorted(S["month"].unique())
    for m in months:
        lomo[m] = fit_params(T, S, [x for x in months if x != m],
                             meta={"held_out_month": m}).to_json()
    (OUT / "pq_lomo.json").write_text(json.dumps(lomo, indent=1))
    print("wrote", OUT / "pq_lomo.json")


# ----------------------------------------------------------------------------- identity

def load_ref(path: str):
    """Load another copy of availability.py (e.g. `git show HEAD:...`) as its own module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("courtside.proj._av_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.PARAMS_PATH = AV.PARAMS_PATH          # the same shipped files (unchanged by pq)
    mod.FRINGE_PATH = AV.FRINGE_PATH
    return mod


def frame_hash(df: pd.DataFrame) -> str:
    import hashlib
    d = df.sort_values(["game_id", "team", "player_id"], kind="mergesort").reset_index(drop=True)
    h = hashlib.sha1()
    h.update(d["player_id"].to_numpy(np.int64).tobytes())
    h.update(d["p_play"].to_numpy(np.float64).tobytes())
    h.update(pd.util.hash_pandas_object(d.astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def dk_starters(conn, slate_id: str, at_iso: str) -> set[int]:
    """NBA ids DK flags as starters in the slate's latest capture at or before `at`."""
    from courtside.proj import ids as IDS
    r = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                     "captured_at<=?", (slate_id, at_iso)).fetchone()
    if not r or not r[0]:
        return set()
    rows = conn.execute("SELECT site_player_id FROM dfs_salaries WHERE slate_id=? AND "
                        "captured_at=? AND starter=1", (slate_id, r[0])).fetchall()
    m = IDS.nba_ids(conn, "dk", [str(x[0]) for x in rows])
    return {int(v) for v in m.values()}


def cmd_identity(a) -> None:
    """OFF (this tree, switch off) vs a reference availability.py: identical p_play_slate frames.
    Cases: --day D (each tip - 45 / tip - 10, box starters at lock-10) and --slate ID (the
    slate's games at lock-45 / lock-10 with the DK starter flags of the time)."""
    from datetime import timezone
    conn = ro_conn(a.db)
    ref = load_ref(a.ref)
    out = []
    cases = []
    for d in a.day or []:
        day = date.fromisoformat(d)
        tips = AV.day_games(conn, day)
        st = {int(r[0]) for r in conn.execute(
            "SELECT player_id FROM player_starts WHERE game_id IN (SELECT game_id FROM team_games "
            "WHERE game_date=?)", (d,))}
        for tip in sorted(tips["tip"].unique()):
            for off in (45, 10):
                cases.append((f"{d} tip {pd.Timestamp(tip):%H:%M}Z lock-{off}", day,
                              pd.Timestamp(tip).to_pydatetime() - timedelta(minutes=off),
                              st if off == 10 else None))
    for sid in a.slate or []:
        r = conn.execute("SELECT nba_day, game_ids_json FROM dfs_slates WHERE slate_id=?",
                         (sid,)).fetchone()
        day = date.fromisoformat(str(r[0])[:10])
        gids = set(json.loads(r[1])) if r[1] else None
        tips = AV.day_games(conn, day)
        if gids:
            tips = tips[tips["game_id"].isin(gids)]
        for tip in sorted(tips["tip"].unique()):
            for off in (45, 10):
                at = pd.Timestamp(tip).to_pydatetime() - timedelta(minutes=off)
                st = dk_starters(conn, sid, at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                cases.append((f"slate {sid} tip {pd.Timestamp(tip):%H:%M}Z lock-{off} "
                              f"({len(st)} DK starters)", day, at, st or None))
    for name, day, at, st in cases:
        AV.set_pplayq(False)
        f_off = AV.p_play_slate(conn, day, at, starters=st)
        f_ref = ref.p_play_slate(conn, day, at, starters=st)
        AV.set_pplayq(True)
        f_on = AV.p_play_slate(conn, day, at, starters=st)
        AV.set_pplayq(False)
        same = frame_hash(f_off) == frame_hash(f_ref) and \
            np.array_equal(f_off["p_play"].to_numpy(), f_ref["p_play"].to_numpy())
        j = f_off.merge(f_on[["game_id", "team", "player_id", "status", "p_play"]],
                        on=["game_id", "team", "player_id", "status"],
                        suffixes=("", "_on"))
        ch = j[j["p_play"] != j["p_play_on"]]
        out.append(f"{name}: rows {len(f_off)}, OFF {frame_hash(f_off)} ref {frame_hash(f_ref)} "
                   f"{'IDENTICAL' if same else 'DIFFERENT'}; ON changes {len(ch)} rows")
        for r in ch.itertuples(index=False):
            out.append(f"    {r.team} {r.player or r.player_id} {r.status} {r.role}: "
                       f"{r.p_play:.3f} -> {r.p_play_on:.3f}")
    s = "\n".join(out)
    print(s)
    if a.save:
        Path(a.save).write_text(s + "\n")


# ----------------------------------------------------------------------------- DFS slice

def cmd_slice(a) -> None:
    """T2.4 slice (research/t24_slice.py run) with the pq offsets fit leaving the month out
    (PQ_LOMO json from `fit`), like the slice's P(play) cells.  The arm is the environment:
    COURTSIDE_PPLAYQ=1 = ON; unset = OFF (the patch is then never called).  Run each arm in its
    own COURTSIDE_DATA so the per-date caches cannot collide."""
    lomo = {m: AV.QRefitParams.from_json(d)
            for m, d in json.loads(Path(os.environ["PQ_LOMO"]).read_text()).items()}
    orig = AV.q_refit_frame

    def patched(df, hist, day, starters, params=None):
        return orig(df, hist, day, starters, lomo[day.strftime("%Y-%m")])

    AV.q_refit_frame = patched
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import t24_slice
    # the scratch box's no-heavy-work windows (ET) on top of the live-lock blackouts
    t24_slice.BLACKOUTS_ET = tuple(t24_slice.BLACKOUTS_ET) + (((18, 0), (19, 20)),
                                                             ((21, 15), (22, 45)))
    say(f"PPLAYQ={AV.PPLAYQ} lomo months {sorted(lomo)}")
    t24_slice.main(["run", a.protocol, "--sims", str(a.sims), "--chunk", "4"]
                   + (["--dates", str(a.dates)] if a.dates else []))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["samples", "diagnose", "evaluate", "fit", "slice",
                                    "identity"])
    ap.add_argument("--ref", default=None, help="identity: reference availability.py")
    ap.add_argument("--day", action="append", help="identity: NBA day (repeatable)")
    ap.add_argument("--slate", action="append", help="identity: DK slate id (repeatable)")
    ap.add_argument("--save", default=None)
    ap.add_argument("protocol", nargs="?", default="live10")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--dates", type=int, default=0)
    ap.add_argument("--db", default=None)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    {"samples": cmd_samples, "diagnose": cmd_diagnose, "evaluate": cmd_evaluate,
     "fit": cmd_fit, "slice": cmd_slice, "identity": cmd_identity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
