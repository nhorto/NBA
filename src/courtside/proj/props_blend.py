"""G3: the pre-registered props blend test (docs/research/g3-preregistration.md).

Report-only and read-only: it opens the database read-only, never writes a table, and nothing in
the app reads it. Props are information only; this module only measures log loss.

Rows (`day_rows`): one per (game, player, stat) at the **lock-45 state**: the newest ok
``lock45`` run created before tip-off with projections for the game, our chance from
``props.compare_game(run=<that run>, market_at=<its created_at>)`` (on the market's no-push
basis, ``props.comparable``), the market's proportional no-vig chance at the same snapshot
state, and the box-score outcome (the ``props.grade_day`` rule).

Test (`evaluate`): ``logit p = a * logit(m) + b * logit(o)``, fit walk-forward on earlier NBA days
only; primary metric = mean LL(blend) - LL(market) with SEs clustered two-way by game and player
(`metrics.paired_cluster2`). Slates from ``PREREG["first_scored_day"]`` on are scored; the
decision is made once, on the first ``PREREG["decision_slates"]`` scored slates. Before that
every number is flagged interim. The pre-registration slates (earlier days) only train; their
walk-forward numbers are shown as a dry run, not part of the test.

De-vig comparison (`devig_compare`, reported only): multiplicative / power / additive, each raw
and with an over-side shade term ``logit p = logit(m_k) + c``, and the blend on each, once
``PREREG["devig_min_rows"]`` eligible graded rows exist (``--devig`` forces it, flagged interim).

Ledger (``--ledger PATH``): the per-row inputs of every fully graded slate, frozen in a CSV. A
slate already in the ledger is read back, never recomputed. Without ``--ledger`` nothing is
written anywhere.

CLI: ``courtside proj props-blend-eval [--from DAY] [--to DAY] [--db PATH] [--ledger PATH]
[--devig] [--json]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import metrics as MT
from . import props as P

VERSION = "props-blend-eval-v1"
PREREG_DOC = Path(__file__).resolve().parents[3] / "docs" / "research" / "g3-preregistration.md"
PREREG_SHA256 = "4c7f0a431acfc8643908ad47d0b51a1f3cb811c0b917cc965eddd1d0c8ae3ad2"   # amended 2026-10-24 (first scored day 10-25)
PREREG = {
    "registered": "2026-10-24",
    "stats": ("pts", "reb", "ast", "pra", "fg3m"),
    "run_kind": "lock45",
    "clip": 0.001,
    "hold_range": (0.0, 0.25),
    "train_min_rows": 1000,
    "season_start": "2026-10-20",
    "first_scored_day": "2026-10-25",
    "decision_slates": 60,
    "z_pass": -2.0,
    "z_guard": 2.0,
    "line_dist_edges": (0.0, 0.1, 0.2, 0.3, 0.5),
    "disagree_edges": (0.0, 0.05, 0.10, 0.20, 1.0),
    "devig_min_rows": 10000,
}

LEDGER_COLS = (
    "nba_day", "game_id", "tipoff_utc", "projection_id", "player_id", "player_name",
    "espn_athlete_id", "stat", "line", "over_odds", "under_odds", "captured_at", "provider",
    "run_id", "run_created_at", "model_version", "dist_source", "our_mean", "our_p_above",
    "our_p_push", "our_p_below", "p_ours", "p_mkt", "hold", "game_state", "actual", "result",
    "y", "evaluator")
_STR_COLS = ("nba_day", "game_id", "tipoff_utc", "player_name", "espn_athlete_id", "stat",
             "captured_at", "provider", "run_id", "run_created_at", "model_version",
             "dist_source", "game_state", "result", "evaluator")


# ============================================================================ rows

def _rows(conn: sqlite3.Connection, sql: str, args=()) -> list[dict]:
    cur = conn.execute(sql, tuple(args))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def lock45_run(conn: sqlite3.Connection, game_id: str, tip: str | None) -> dict | None:
    """The newest ok lock-45 run created at or before tip-off with projections for the game."""
    sql = ("SELECT r.* FROM dfs_runs r WHERE r.kind = ? AND COALESCE(r.ok, 1) = 1 "
           "AND EXISTS (SELECT 1 FROM player_projections p WHERE p.run_id = r.run_id "
           "AND p.game_id = ?)")
    args: list = [PREREG["run_kind"], game_id]
    if tip:
        sql += " AND r.created_at <= ?"
        args.append(tip)
    sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1"
    rows = _rows(conn, sql, args)
    return rows[0] if rows else None


def _game_states(conn: sqlite3.Connection, day: str, ids: list[str]) -> dict[str, str]:
    from .dfs.grade import game_states
    return game_states(conn, date.fromisoformat(day), ids)


def day_rows(conn: sqlite3.Connection, day: str, cache: dict | None = None) -> dict:
    """{rows: [ledger rows], games, no_run, unjoined: {reason: n}} of one NBA day at the lock-45
    state. Read-only."""
    cache = cache if cache is not None else {}
    games = P.day_games(conn, day)
    states = _game_states(conn, day, [g["game_id"] for g in games]) if games else {}
    out: dict = {"rows": [], "games": 0, "no_run": 0, "unjoined": {},
                 "pending_games": sum(states.get(g["game_id"]) == "pending" for g in games)}
    final = [g["game_id"] for g in games if states.get(g["game_id"]) == "final"]
    box: dict = {}
    if final:
        ph = ",".join("?" * len(final))
        box = {(r[0], int(r[1])): r for r in conn.execute(
            f"SELECT game_id, player_id, min, pts, reb, ast, fg3m FROM player_games "
            f"WHERE game_id IN ({ph})", final)}
    for g in games:
        gid, tip = g["game_id"], g.get("tipoff_utc")
        run = lock45_run(conn, gid, tip)
        if run is None:
            if P.market_state(conn, gid, tip):
                out["no_run"] += 1
            continue
        res = P.compare_game(conn, g, run=run, market_at=run["created_at"], cache=cache)
        if not res["props"]:
            continue
        out["games"] += 1
        for u in res["unjoined"]:
            if not u.get("joined"):
                out["unjoined"][u["reason"]] = out["unjoined"].get(u["reason"], 0) + 1
        dists = cache.get(("dists", run["run_id"], gid), {})
        st = states.get(gid, "pending")
        for r in res["rows"]:
            pid = int(r["player_id"])
            actual = result = y = None
            if st == "final":
                b = box.get((gid, pid))
                if b is None or not (b[2] or 0) > 0:
                    result = "void"
                else:
                    vals = {"pts": b[3], "reb": b[4], "ast": b[5], "fg3m": b[6]}
                    vals["pra"] = (b[3] or 0) + (b[4] or 0) + (b[5] or 0)
                    actual = float(vals[r["stat"]] or 0)
                    line = float(r["line"])
                    result = "above" if actual > line else "below" if actual < line else "push"
                    y = None if result == "push" else int(result == "above")
            d = dists.get(pid)
            out["rows"].append({
                "nba_day": day, "game_id": gid, "tipoff_utc": tip,
                "projection_id": r["projection_id"], "player_id": pid,
                "player_name": r.get("player_name"), "espn_athlete_id": r.get("espn_athlete_id"),
                "stat": r["stat"], "line": r["line"], "over_odds": r.get("over_odds"),
                "under_odds": r.get("under_odds"), "captured_at": r["captured_at"],
                "provider": r.get("provider"), "run_id": r["run_id"],
                "run_created_at": r.get("run_created_at"), "model_version": r.get("model_version"),
                "dist_source": getattr(d, "source", None), "our_mean": r.get("our_mean"),
                "our_p_above": r["our_p_above"], "our_p_push": r["our_p_push"],
                "our_p_below": r["our_p_below"],
                "p_ours": P.comparable(r["our_p_above"], r["our_p_push"]),
                "p_mkt": r["mkt_p_above"], "hold": r["hold"], "game_state": st,
                "actual": actual, "result": result, "y": y, "evaluator": VERSION})
    return out


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(LEDGER_COLS))
    for c in _STR_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def read_ledger(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={c: str for c in _STR_COLS}, keep_default_na=True)
    return df[[c for c in LEDGER_COLS if c in df.columns]]


def write_ledger(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.sort_values(["nba_day", "game_id", "player_id", "stat", "line", "captured_at"]
                   ).to_csv(tmp, index=False, columns=list(LEDGER_COLS))
    os.replace(tmp, p)


def build_frame(conn: sqlite3.Connection, lo: str, hi: str, ledger: pd.DataFrame | None = None,
                today: str | None = None) -> tuple[pd.DataFrame, dict]:
    """All rows of the days lo..hi (ledger days read back, the rest computed) and an info dict
    {days, computed, from_ledger, frozen_new: [days now fully graded], no_run, unjoined}."""
    today = today or config.today_nba_day().isoformat()
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT nba_day FROM games WHERE nba_day BETWEEN ? AND ? ORDER BY 1", (lo, hi))]
    have = set(ledger["nba_day"].unique()) if ledger is not None and len(ledger) else set()
    parts, info = [], {"days": days, "computed": [], "from_ledger": [], "frozen_new": [],
                       "no_run": 0, "unjoined": {}}
    if have:
        parts.append(ledger[ledger["nba_day"].isin(days)])
    cache: dict = {}
    for d in days:
        if d in have:
            info["from_ledger"].append(d)
            continue
        res = day_rows(conn, d, cache)
        info["computed"].append(d)
        info["no_run"] += res["no_run"]
        for k, v in res["unjoined"].items():
            info["unjoined"][k] = info["unjoined"].get(k, 0) + v
        if res["rows"]:
            parts.append(_frame(res["rows"]))
            if d < today and not res["pending_games"]:     # every game of the day graded
                info["frozen_new"].append(d)
    df = pd.concat(parts, ignore_index=True) if parts else _frame([])
    return df, info


# ============================================================================ eligibility

def eligibility(df: pd.DataFrame) -> pd.Series:
    """Per row: 'ok' or the first exclusion reason (price, hold, dup, not_final, void, push).
    Duplicates of (game, player, stat) keep the line with the market chance closest to 0.5, then
    the newest capture, then the lower line (decided before the outcome is looked at)."""
    lo, hi = PREREG["hold_range"]
    why = pd.Series("ok", index=df.index, dtype=object)
    pm = pd.to_numeric(df["p_mkt"], errors="coerce")
    po = pd.to_numeric(df["p_ours"], errors="coerce")
    hold = pd.to_numeric(df["hold"], errors="coerce")
    why[pm.isna() | po.isna()] = "price"
    why[(why == "ok") & ~hold.between(lo, hi)] = "hold"
    ok = df[why == "ok"].assign(_d=(pm - 0.5).abs(), _line=pd.to_numeric(df["line"]))
    ok = ok.sort_values(["_d", "captured_at", "_line"], ascending=[True, False, True],
                        kind="mergesort")
    dup = ok.duplicated(["game_id", "player_id", "stat"], keep="first")
    why[dup[dup].index] = "dup"
    st = df["game_state"].fillna("pending")
    res = df["result"]
    why[(why == "ok") & (st != "final")] = "not_final"
    why[(why == "ok") & (res == "void")] = "void"
    why[(why == "ok") & (res == "push")] = "push"
    return why


# ============================================================================ maths

def _clip(p) -> np.ndarray:
    c = PREREG["clip"]
    return np.clip(np.asarray(p, float), c, 1.0 - c)


def logit(p) -> np.ndarray:
    q = _clip(p)
    return np.log(q / (1.0 - q))


def _sig(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logloss(p, y) -> np.ndarray:
    q, y = _clip(p), np.asarray(y, float)
    return -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))


def fit_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
              start: np.ndarray | None = None, iters: int = 100) -> np.ndarray:
    """Unpenalised logistic MLE (no implicit intercept) by Newton with step halving."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1]) if start is None else np.asarray(start, float).copy()

    def nll(beta):
        z = X @ beta + off
        return float(np.sum(np.logaddexp(0.0, z) - y * z))
    f = nll(b)
    for _ in range(iters):
        p = _sig(X @ b + off)
        g = X.T @ (y - p)
        H = X.T @ (X * (p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while t > 1e-6:
            nb = b + t * step
            nf = nll(nb)
            if nf <= f + 1e-12:
                break
            t /= 2.0
        b, f_old, f = nb, f, nf
        if np.max(np.abs(t * step)) < 1e-10 or abs(f_old - f) < 1e-12:
            break
    return b


def sandwich2(X: np.ndarray, y: np.ndarray, beta: np.ndarray, c1, c2,
              offset: np.ndarray | None = None) -> np.ndarray:
    """Two-way (c1, c2) cluster-robust covariance of a logistic fit: B^-1 (M1 + M2 - M12) B^-1,
    each M with the G/(G-1) factor; falls back to the larger one-way M if M is not PSD."""
    X = np.asarray(X, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    p = _sig(X @ beta + off)
    S = X * (np.asarray(y, float) - p)[:, None]
    B = X.T @ (X * (p * (1.0 - p))[:, None])
    Binv = np.linalg.pinv(B)

    def meat(labels):
        lab = pd.Series(np.asarray(labels)).astype(str).to_numpy()
        g = pd.DataFrame(S).groupby(lab).sum().to_numpy()
        G = len(g)
        return (G / (G - 1)) * g.T @ g if G > 1 else np.full((X.shape[1],) * 2, np.nan)
    a1, a2 = np.asarray(c1).astype(str), np.asarray(c2).astype(str)
    M1, M2, M12 = meat(a1), meat(a2), meat(np.char.add(np.char.add(a1, "|"), a2))
    M = M1 + M2 - M12
    if np.all(np.isfinite(M)) and np.min(np.linalg.eigvalsh((M + M.T) / 2)) < 0:
        M = M1 if np.trace(Binv @ M1 @ Binv) >= np.trace(Binv @ M2 @ Binv) else M2
    return Binv @ M @ Binv


def walk_forward(df: pd.DataFrame, X: np.ndarray, *, offset: np.ndarray | None = None,
                 min_rows: int | None = None) -> tuple[np.ndarray, dict]:
    """Per row the prediction of a fit on every row with an earlier nba_day (NaN when the
    training set is below `min_rows`), and {day: coefficients}."""
    min_rows = PREREG["train_min_rows"] if min_rows is None else min_rows
    days = df["nba_day"].to_numpy()
    y = df["y"].to_numpy(float)
    off = np.zeros(len(df)) if offset is None else np.asarray(offset, float)
    pred = np.full(len(df), np.nan)
    coefs: dict = {}
    start = None
    for d in sorted(set(days)):
        tr, te = days < d, days == d
        if tr.sum() < min_rows:
            continue
        b = fit_logit(X[tr], y[tr], off[tr], start=start)
        start = b
        coefs[d] = b
        pred[te] = _sig(X[te] @ b + off[te])
    return pred, coefs


def blend_X(m, o) -> np.ndarray:
    return np.column_stack([logit(m), logit(o)])


# ---------------------------------------------------------------------------- de-vig methods

def implied_arr(odds) -> np.ndarray:
    a = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(float)
    out = np.full(len(a), np.nan)
    pos, neg = a >= 100, a <= -100
    out[pos] = 100.0 / (a[pos] + 100.0)
    out[neg] = -a[neg] / (-a[neg] + 100.0)
    return out


def devig(qo: np.ndarray, qu: np.ndarray, method: str) -> np.ndarray:
    """Over-side chance from the two implied chances: 'mult' q_o / (q_o + q_u), 'add' q_o - hold/2,
    'power' q_o^k with q_o^k + q_u^k = 1 (bisection)."""
    qo, qu = np.asarray(qo, float), np.asarray(qu, float)
    if method == "mult":
        return qo / (qo + qu)
    if method == "add":
        return qo - (qo + qu - 1.0) / 2.0
    if method == "power":
        lo, hi = np.full(len(qo), 1e-3), np.full(len(qo), 20.0)
        for _ in range(80):
            k = (lo + hi) / 2.0
            s = qo ** k + qu ** k
            big = s > 1.0           # sum falls with k: need a larger k
            lo = np.where(big, k, lo)
            hi = np.where(big, hi, k)
        return qo ** ((lo + hi) / 2.0)
    raise ValueError(method)


# ============================================================================ the report

def _diff(d, g, pl) -> dict:
    r = MT.paired_cluster2(np.asarray(d, float), g, pl)
    z = r["mean"] / r["se"] if r["se"] and np.isfinite(r["se"]) and r["se"] > 0 else None
    return {"mean": r["mean"], "se": r["se"], "z": z, "n": r["n"], "se_game": r["se_c1"],
            "se_player": r["se_c2"], "games": r["n_c1"], "players": r["n_c2"]}


def _coef(X, y, g, pl, names=("a_market", "b_ours")) -> dict:
    if len(y) < len(names) + 1:
        return {}
    b = fit_logit(X, y)
    V = sandwich2(X, y, b, g, pl)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {n: {"coef": float(c), "se": float(s)} for n, c, s in zip(names, b, se)}


def _buckets(x: np.ndarray, edges) -> np.ndarray:
    i = np.searchsorted(np.asarray(edges[1:-1]), x, side="right")
    return np.array([f"[{edges[k]:g}, {edges[k + 1]:g}{']' if k == len(edges) - 2 else ')'}"
                     for k in i], dtype=object)


def _by(sub: pd.DataFrame, labels, order=None) -> dict:
    out = {}
    keys = order if order is not None else sorted(set(labels))
    for k in keys:
        s = sub[np.asarray(labels) == k]
        if len(s):
            out[str(k)] = _diff(s["ll_blend"] - s["ll_mkt"], s["game_id"], s["player_id"])
    return out


def block(sub: pd.DataFrame) -> dict:
    """Primary metric and secondary checks of scored rows (columns p_blend, p_mkt, p_ours, y)."""
    if not len(sub):
        return {"n": 0}
    sub = sub.copy()
    y = sub["y"].to_numpy(float)
    m, o, pb = (sub[c].to_numpy(float) for c in ("p_mkt", "p_ours", "p_blend"))
    sub["ll_blend"], sub["ll_mkt"], sub["ll_ours"] = logloss(pb, y), logloss(m, y), logloss(o, y)
    g, pl = sub["game_id"].to_numpy(), sub["player_id"].to_numpy()
    prim = _diff(sub["ll_blend"] - sub["ll_mkt"], g, pl)
    out = {
        "n": int(len(sub)), "slates": int(sub["nba_day"].nunique()),
        "games": int(sub["game_id"].nunique()), "players": int(sub["player_id"].nunique()),
        "above_rate": float(y.mean()),
        "ll_market": float(sub["ll_mkt"].mean()), "ll_blend": float(sub["ll_blend"].mean()),
        "ll_ours": float(sub["ll_ours"].mean()),
        "primary": prim,
        "ours_vs_market": _diff(sub["ll_ours"] - sub["ll_mkt"], g, pl),
        "brier": {"market": float(np.mean((_clip(m) - y) ** 2)),
                  "blend": float(np.mean((pb - y) ** 2)),
                  "ours": float(np.mean((_clip(o) - y) ** 2)),
                  "blend_minus_market": _diff((pb - y) ** 2 - (_clip(m) - y) ** 2, g, pl)},
        "by_stat": _by(sub, sub["stat"].to_numpy(), PREREG["stats"]),
    }
    e1, e2 = PREREG["line_dist_edges"], PREREG["disagree_edges"]
    out["by_line_distance"] = _by(sub, _buckets(np.abs(o - 0.5), e1),
                                  list(dict.fromkeys(_buckets(np.array(e1[:-1]), e1))))
    out["by_disagreement"] = _by(sub, _buckets(np.abs(o - m), e2),
                                 list(dict.fromkeys(_buckets(np.array(e2[:-1]), e2))))
    months = sub["nba_day"].str[:7].to_numpy()
    out["by_month"] = {k: {"mean": float(v), "n": int((months == k).sum())}
                       for k, v in (sub["ll_blend"] - sub["ll_mkt"]).groupby(months).mean().items()}
    out["coef_fit_on_these_rows"] = _coef(blend_X(m, o), y, g, pl)
    bins = np.minimum((pb * 10).astype(int), 9)
    out["calibration_blend"] = [
        {"bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int((bins == k).sum()),
         "pred": float(pb[bins == k].mean()), "obs": float(y[bins == k].mean())}
        for k in range(10) if (bins == k).any()]
    tip = pd.to_datetime(sub["tipoff_utc"], utc=True, errors="coerce")
    cap = pd.to_datetime(sub["captured_at"], utc=True, errors="coerce")
    run = pd.to_datetime(sub["run_created_at"], utc=True, errors="coerce")
    h_cap, h_run = (tip - cap).dt.total_seconds() / 3600, (tip - run).dt.total_seconds() / 3600
    out["hours_to_tip"] = {"market_capture": _quart(h_cap), "run": _quart(h_run)}
    out["model_versions"] = {str(k): int(v) for k, v in sub["model_version"].value_counts().items()}
    out["dist_source"] = {str(k): int(v)
                          for k, v in sub["dist_source"].fillna("?").value_counts().items()}
    return out


def _quart(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return {}
    q = s.quantile([0.1, 0.5, 0.9])
    return {"p10": float(q.iloc[0]), "median": float(q.iloc[1]), "p90": float(q.iloc[2])}


def devig_compare(el: pd.DataFrame, scored_mask: np.ndarray) -> dict:
    """Reported only (pre-registration section 8): de-vig methods raw / with an over-side shade
    term / in the blend, walk-forward like the test, scored on the same rows, each minus the
    multiplicative raw market (two-way SE)."""
    qo, qu = implied_arr(el["over_odds"]), implied_arr(el["under_odds"])
    y = el["y"].to_numpy(float)
    o = el["p_ours"].to_numpy(float)
    one = np.ones((len(el), 1))
    base = None
    arms: dict[str, np.ndarray] = {}
    for meth in ("mult", "power", "add"):
        m = devig(qo, qu, meth)
        arms[f"{meth}"] = _clip(m)
        arms[f"{meth}+shade"], _ = walk_forward(el, one, offset=logit(m))
        arms[f"blend_{meth}"], _ = walk_forward(el, blend_X(m, o))
        arms[f"blend_{meth}+shade"], _ = walk_forward(
            el, np.column_stack([blend_X(m, o), one]))
        if meth == "mult":
            base = arms["mult"]
    ok = scored_mask.copy()
    out = {"n": int(ok.sum()), "slates": int(el["nba_day"][ok].nunique()),
           "ll_mult_raw": float(logloss(base[ok], y[ok]).mean()) if ok.any() else None,
           "vs_mult_raw": {}}
    for k, v in arms.items():               # each arm on the rows it scores (fits need training)
        if k == "mult":
            continue
        rk = ok & np.isfinite(v)
        if rk.any():
            out["vs_mult_raw"][k] = _diff(logloss(v[rk], y[rk]) - logloss(base[rk], y[rk]),
                                          el["game_id"].to_numpy()[rk],
                                          el["player_id"].to_numpy()[rk])
        else:
            out["vs_mult_raw"][k] = {"n": 0, "note": "no rows with enough training"}
    g, pl = el["game_id"].to_numpy()[ok], el["player_id"].to_numpy()[ok]
    if ok.any():
        sh = {}
        for meth in ("mult", "power", "add"):
            m = devig(qo, qu, meth)[ok]
            c = fit_logit(one[ok], y[ok], offset=logit(m))
            V = sandwich2(one[ok], y[ok], c, g, pl, offset=logit(m))
            sh[meth] = {"coef": float(c[0]), "se": float(np.sqrt(max(V[0, 0], 0.0)))}
        out["shade_fit_on_these_rows"] = sh
    return out


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def evaluate(df: pd.DataFrame, *, devig: bool = False) -> dict:
    """The status report of the pre-registered test on a frame of ledger rows."""
    first, need = PREREG["first_scored_day"], PREREG["decision_slates"]
    why = eligibility(df) if len(df) else pd.Series(dtype=object)
    excl = {str(k): int(v) for k, v in why.value_counts().items()} if len(df) else {}
    el = df[why == "ok"].copy() if len(df) else df.copy()
    el["y"] = pd.to_numeric(el["y"], errors="coerce")
    el = el.sort_values(["nba_day", "game_id", "player_id", "stat"]).reset_index(drop=True)
    for c in ("p_mkt", "p_ours", "line"):
        el[c] = pd.to_numeric(el[c], errors="coerce")
    sha = _sha(PREREG_DOC)
    rep: dict = {
        "version": VERSION, "prereg": {**{k: v for k, v in PREREG.items()},
                                       "doc": str(PREREG_DOC), "sha256": sha,
                                       "sha256_pinned": PREREG_SHA256,
                                       "doc_unchanged": (sha == PREREG_SHA256) if sha else None},
        "rows": int(len(df)), "eligible": int(len(el)), "excluded": excl,
        "graded_eligible_rows": int(len(el)),
    }
    slates = sorted(el["nba_day"].unique()) if len(el) else []
    pre = [d for d in slates if d < first]
    post = [d for d in slates if d >= first]
    rep["slates"] = {"pre_registration": pre, "n_pre_registration": len(pre),
                     "n_graded_since_registration": len(post)}
    if not len(el):
        rep.update(status="collecting", n_scored_slates=0, minimum=need,
                   note="no eligible graded rows yet")
        return rep
    X = blend_X(el["p_mkt"], el["p_ours"])
    pb, coefs = walk_forward(el, X)
    el["p_blend"] = pb
    scored = el[(el["nba_day"] >= first) & np.isfinite(pb)]
    sdays = sorted(scored["nba_day"].unique())
    rep["n_scored_slates"], rep["minimum"] = len(sdays), need
    rep["scored_days"] = sdays
    rep["last_walk_forward_coef"] = (
        {"day": max(coefs), "a_market": float(coefs[max(coefs)][0]),
         "b_ours": float(coefs[max(coefs)][1])} if coefs else None)
    if len(sdays) < need:
        rep["status"] = "collecting"
        rep["decision"] = None
        rep["interim"] = {"flag": f"INTERIM: {len(sdays)} of {need} scored slates; no decision "
                                  f"before {need}", **block(scored)}
    else:
        dec_days = sdays[:need]
        dec = scored[scored["nba_day"].isin(dec_days)]
        b = block(dec)
        prim = b["primary"]
        passed = prim["mean"] < 0 and prim["z"] is not None and prim["z"] <= PREREG["z_pass"]
        guard = [s for s, v in b["by_stat"].items()
                 if v.get("z") is not None and v["z"] >= PREREG["z_guard"]]
        result = "FAIL" if not passed else ("PASS (stat guard)" if guard else "PASS")
        rep["status"] = "decided"
        rep["decision"] = {"result": result, "decision_days": [dec_days[0], dec_days[-1]],
                           "stat_guard": guard, **b}
        later = scored[~scored["nba_day"].isin(dec_days)]
        if len(later):
            rep["follow_up"] = {"flag": "after the decision set; does not change the decision",
                                **block(later)}
    dry = el[(el["nba_day"] < first) & np.isfinite(pb)]
    rep["dry_run_pre_registration"] = {
        "flag": "DRY RUN on the pre-registration slates (walk-forward); not part of the test",
        **block(dry)} if len(dry) else {"n": 0, "flag": "no pre-registration slate has "
                                                       f"{PREREG['train_min_rows']}+ training rows"}
    pre_el = el[el["nba_day"] < first]
    if len(pre_el):
        y = pre_el["y"].to_numpy(float)
        rep["pre_registration_descriptive"] = {
            "flag": "pre-registration slates, in sample: descriptive only",
            "n": int(len(pre_el)), "slates": len(pre),
            "ll_market": float(logloss(pre_el["p_mkt"], y).mean()),
            "ll_ours": float(logloss(pre_el["p_ours"], y).mean()),
            "ours_vs_market": _diff(logloss(pre_el["p_ours"], y) - logloss(pre_el["p_mkt"], y),
                                    pre_el["game_id"], pre_el["player_id"]),
            "coef_in_sample": _coef(blend_X(pre_el["p_mkt"], pre_el["p_ours"]), y,
                                    pre_el["game_id"], pre_el["player_id"])}
    n_dv, need_dv = len(el), PREREG["devig_min_rows"]
    if devig or n_dv >= need_dv:
        mask = (el["nba_day"] >= first).to_numpy()
        if not mask.any():
            mask = np.ones(len(el), bool)
            where = "all eligible rows (no scored slate yet)"
        else:
            where = "scored slates"
        rep["devig"] = {"flag": ("reported only" if n_dv >= need_dv else
                                 f"INTERIM: {n_dv} of {need_dv} graded rows; reported only"),
                        "rows": where, **devig_compare(el, mask)}
    else:
        rep["devig"] = {"status": f"waiting: {n_dv} of {need_dv} graded rows"}
    return rep


def prop_compare_check(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    """Cross-check the recomputed rows against the stored prop_compare rows with the same key
    (projection_id, market, line, captured_at): our chance, the market's and the grade."""
    if not len(df):
        return {"matched": 0}
    try:
        pc = pd.read_sql_query(
            "SELECT projection_id, market AS stat, line, captured_at, our_p_above AS pc_ours, "
            "mkt_p_above AS pc_mkt, result AS pc_result FROM prop_compare", conn)
    except Exception:  # noqa: BLE001 - a DB without the table
        return {"matched": 0}
    d = df[["projection_id", "stat", "line", "captured_at", "our_p_above", "p_mkt", "result"]
           ].copy()
    d["projection_id"] = pd.to_numeric(d["projection_id"], errors="coerce")
    d["line"] = pd.to_numeric(d["line"], errors="coerce")
    j = d.merge(pc, on=["projection_id", "stat", "line", "captured_at"])
    if not len(j):
        return {"matched": 0}
    graded = j["pc_result"].notna() & j["result"].notna()
    return {"matched": int(len(j)),
            "max_abs_diff_ours": float(np.nanmax(np.abs(pd.to_numeric(j["our_p_above"])
                                                        - j["pc_ours"]))),
            "max_abs_diff_market": float(np.nanmax(np.abs(pd.to_numeric(j["p_mkt"])
                                                          - j["pc_mkt"]))),
            "grade_mismatches": int((j["pc_result"][graded] != j["result"][graded]).sum()),
            "graded_matched": int(graded.sum())}


def run(conn: sqlite3.Connection, lo: str | None = None, hi: str | None = None, *,
        ledger: str | Path | None = None, devig: bool = False, today: str | None = None) -> dict:
    """Build the frame (ledger days read back), evaluate, and (only with `ledger`) freeze newly
    fully graded slates into the ledger file. Never writes the database."""
    today = today or config.today_nba_day().isoformat()
    lo = lo or PREREG["season_start"]
    hi = hi or today
    led = read_ledger(ledger) if ledger else None
    df, info = build_frame(conn, lo, hi, led, today=today)
    rep = evaluate(df, devig=devig)
    rep["from"], rep["to"], rep["today"] = lo, hi, today
    rep["build"] = {k: info[k] for k in ("computed", "from_ledger", "frozen_new", "no_run",
                                         "unjoined")}
    rep["build"]["games_with_props_no_lock45_run"] = rep["build"].pop("no_run")
    rep["checks"] = {"prop_compare": prop_compare_check(conn, df[df["nba_day"].isin(
        info["computed"])])}
    if ledger and info["frozen_new"]:
        new = df[df["nba_day"].isin(info["frozen_new"])]
        allrows = pd.concat([led, new], ignore_index=True) if led is not None else new
        write_ledger(allrows, ledger)
        rep["build"]["ledger_written"] = str(ledger)
    return _clean(rep)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


# ============================================================================ CLI

def _n(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def _fmt_diff(v: dict | None, d=4) -> str:
    if not v:
        return "-"
    return (f"{_n(v.get('mean'), d)} (SE {_n(v.get('se'), d)}, z {_n(v.get('z'), 2)}, "
            f"n {v.get('n')})")


def _fmt_block(L: list[str], b: dict, ind: str = "    ") -> None:
    if not b or not b.get("n"):
        L.append(f"{ind}no rows")
        return
    L.append(f"{ind}n {b['n']} rows, {b['slates']} slates, {b['games']} games, "
             f"{b['players']} players; above rate {_n(b['above_rate'], 3)}")
    L.append(f"{ind}LL market {_n(b['ll_market'])}, blend {_n(b['ll_blend'])}, "
             f"ours {_n(b['ll_ours'])}")
    L.append(f"{ind}blend - market: {_fmt_diff(b['primary'])}")
    L.append(f"{ind}ours - market:  {_fmt_diff(b['ours_vs_market'])}")
    for name in ("by_stat", "by_line_distance", "by_disagreement"):
        L.append(f"{ind}{name}: " + "; ".join(
            f"{k} {_n(v['mean'])} (z {_n(v['z'], 1)}, n {v['n']})" for k, v in b[name].items()))
    c = b.get("coef_fit_on_these_rows") or {}
    if c:
        L.append(f"{ind}fit on these rows: a {_n(c['a_market']['coef'], 3)} "
                 f"(SE {_n(c['a_market']['se'], 3)}), b {_n(c['b_ours']['coef'], 3)} "
                 f"(SE {_n(c['b_ours']['se'], 3)})")


def format_report(rep: dict) -> str:
    pr = rep["prereg"]
    L = [f"G3 props blend test ({rep['version']}), {rep['from']} .. {rep['to']}",
         f"  pre-registration {pr['registered']}: doc unchanged = {pr['doc_unchanged']}",
         f"  status: {rep['status']}; scored slates {rep.get('n_scored_slates', 0)} of "
         f"{rep['minimum']} (pre-registration slates {rep['slates']['n_pre_registration']}, "
         f"training only)",
         f"  rows {rep['rows']}, eligible graded {rep['eligible']}; excluded "
         f"{json.dumps({k: v for k, v in rep['excluded'].items() if k != 'ok'})}"]
    b = rep.get("build") or {}
    if b:
        L.append(f"  games with props but no lock-45 run: "
                 f"{b.get('games_with_props_no_lock45_run')}; unjoined props "
                 f"{json.dumps(b.get('unjoined'))}")
    ck = (rep.get("checks") or {}).get("prop_compare")
    if ck:
        L.append(f"  prop_compare cross-check: {json.dumps(ck)}")
    if rep.get("last_walk_forward_coef"):
        c = rep["last_walk_forward_coef"]
        L.append(f"  last walk-forward fit ({c['day']}): a {_n(c['a_market'], 3)}, "
                 f"b {_n(c['b_ours'], 3)}")
    if rep.get("decision"):
        d = rep["decision"]
        L.append(f"  DECISION: {d['result']} on {d['decision_days'][0]} .. "
                 f"{d['decision_days'][1]}")
        _fmt_block(L, d)
    if rep.get("interim"):
        L.append(f"  {rep['interim']['flag']}")
        _fmt_block(L, rep["interim"])
    if rep.get("follow_up"):
        L.append(f"  follow-up ({rep['follow_up']['flag']})")
        _fmt_block(L, rep["follow_up"])
    dr = rep.get("dry_run_pre_registration")
    if dr:
        L.append(f"  {dr['flag']}")
        _fmt_block(L, dr)
    pd_ = rep.get("pre_registration_descriptive")
    if pd_:
        L.append(f"  {pd_['flag']}: n {pd_['n']}, LL market {_n(pd_['ll_market'])}, ours "
                 f"{_n(pd_['ll_ours'])}; ours - market {_fmt_diff(pd_['ours_vs_market'])}")
        c = pd_.get("coef_in_sample") or {}
        if c:
            L.append(f"    in-sample fit: a {_n(c['a_market']['coef'], 3)} "
                     f"(SE {_n(c['a_market']['se'], 3)}), b {_n(c['b_ours']['coef'], 3)} "
                     f"(SE {_n(c['b_ours']['se'], 3)})")
    dv = rep.get("devig") or {}
    if dv.get("status"):
        L.append(f"  de-vig comparison: {dv['status']}")
    elif dv:
        L.append(f"  de-vig comparison ({dv['flag']}; {dv['rows']}): n {dv['n']}, LL mult raw "
                 f"{_n(dv['ll_mult_raw'])}")
        for k, v in dv["vs_mult_raw"].items():
            L.append(f"    {k} - mult raw: {_fmt_diff(v, 5) if v.get('n') else v.get('note')}")
        for k, v in (dv.get("shade_fit_on_these_rows") or {}).items():
            L.append(f"    over-side shade c ({k}, fit on these rows): {_n(v['coef'], 3)} "
                     f"(SE {_n(v['se'], 3)})")
    return "\n".join(L)


def add_commands(sub) -> dict:
    p = sub.add_parser("props-blend-eval", help="G3: the pre-registered props blend test "
                                                "(read-only status report)")
    p.add_argument("--from", dest="lo", default=None)
    p.add_argument("--to", dest="hi", default=None)
    p.add_argument("--db", default=None, help="database path (opened read-only)")
    p.add_argument("--ledger", default=None,
                   help="CSV of frozen per-row inputs: read back, and new fully graded slates "
                        "appended (nothing is written without it)")
    p.add_argument("--devig", action="store_true",
                   help="run the de-vig comparison before its 10k-row minimum (interim)")
    p.add_argument("--json", action="store_true")
    return {"props-blend-eval": _cmd}


def _cmd(args) -> int:
    path = Path(args.db) if args.db else config.db_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rep = run(conn, args.lo, args.hi, ledger=args.ledger, devig=args.devig)
    finally:
        conn.close()
    print(json.dumps(rep, indent=1) if args.json else format_report(rep))
    return 0
