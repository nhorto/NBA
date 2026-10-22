"""Game-model calibration study: is gbm_market overconfident in season, where, and does a
walk-forward recalibration layer help?  (docs/research/game-calibration.md)

Inputs
* `--preds`  walk-forward per-game predictions from `courtside model gbm-backtest`
  (rung3_preds_*.csv.gz: one fold per test season, fit on earlier seasons only).
* `--db`     a database copy (read-only) for game dates/teams and the de-vigged closing
  moneyline (`model.market.load_closing`).
* `--replay` optional day-by-day production replays (`research/game_calibration_replay.py`
  output, or the original 838-game replay pulled from `predictions`), `label=path`.

Statistics
* Log loss / Brier / accuracy; paired differences with **date-clustered** SEs (all games of an
  NBA day form one cluster).
* Logistic recalibration y ~ a + b·logit(p) (slope b < 1 = overconfident) with a
  date-clustered sandwich SE; Spiegelhalter z exactly as `mlops.monitor.spiegelhalter_z`
  (z > 0 = overconfident), plus a cluster-robust variant.
* Disagreement regression y ~ a + b1·logit(p_mkt) + b2·(logit p − logit p_mkt): b2 < 1 means
  the model's departures from the close are larger than the outcomes support.

Candidate fixes (all walk-forward: the layer for test season S is fit on the walk-forward
predictions of seasons < S only; first fixed season = `--fix-from`):
* `temp`        one global Platt layer (a, b) on logit(p).
* `temp_phase`  a Platt layer per season phase (team games 0-10 / 11-25 / 26+).
* `shrink`      toward the close with one weight: logit p' = logit p_mkt + w·(logit p −
  logit p_mkt) where a moneyline exists; unchanged elsewhere.
* `market_only` reference row (not a candidate): the de-vigged close wherever it exists.
* `shrink_phase` toward the close: logit p' = logit p_mkt + w_phase·(logit p − logit p_mkt)
  where a moneyline exists (w fit by maximum likelihood), the global Platt layer elsewhere.

    uv run python research/game_calibration.py --preds var/tmp/gcal/rung3_preds.csv.gz \\
        --db var/courtside.db --replay orig=var/tmp/gcal/replay_orig.csv.gz \\
        --out var/tmp/gcal/game_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from courtside.mlops.monitor import calibration_fit, spiegelhalter_z
from courtside.model import dataset, market

EPS = 1e-15
PHASES = ((0, 10, "0-10"), (11, 25, "11-25"), (26, 10**6, "26+"))
FAV_BINS = ((0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"),
            (0.8, 1.01, "80%+"))


# ------------------------------------------------------------------ basic statistics

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_mean_se(x, cl) -> tuple[float, float]:
    """Mean of x and its SE with clusters `cl` (CR1 small-sample factor)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else float("nan"), float("nan")
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return float(m), float("nan")
    return float(m), float(np.sqrt((s ** 2).sum() * g / (g - 1)) / n)


def logistic_fit(X, y, cl=None, w0=None):
    """Logistic MLE by Newton; returns beta and (date-clustered if `cl`) sandwich covariance."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, float)
    for _ in range(100):
        mu = sigmoid(X @ beta)
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = sigmoid(X @ beta)
    H = X.T @ (X * (mu * (1 - mu))[:, None])
    Hi = np.linalg.inv(H)
    if cl is None:
        return beta, Hi
    sc = pd.DataFrame(X * (y - mu)[:, None]).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(sc)
    meat = sc.T @ sc * g / max(g - 1, 1)
    return beta, Hi @ meat @ Hi


def calib(y, p, cl) -> dict:
    """Recalibration slope/intercept (clustered SE), MLOPS z, cluster-robust z, metrics."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y))}
    if len(y) < 30 or y.min() == y.max():
        return out
    X = np.column_stack([np.ones(len(p)), logit(p)])
    b, V = logistic_fit(X, y, cl)
    mon = calibration_fit(y, p) or {}
    u = (y - p) * (1 - 2 * p)
    uc = pd.Series(u).groupby(np.asarray(cl)).sum().to_numpy()
    out.update({
        "log_loss": float(ll_vec(y, p).mean()), "brier": float(np.mean((y - p) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "intercept": float(b[0]), "intercept_se": float(np.sqrt(V[0, 0])),
        "slope": float(b[1]), "slope_se": float(np.sqrt(V[1, 1])),
        "slope_se_iid": mon.get("slope_se"),
        "z": spiegelhalter_z(y, p),
        "z_cluster": float(u.sum() / np.sqrt((uc ** 2).sum())) if (uc ** 2).sum() > 0 else None,
        "mean_p": float(p.mean()), "mean_y": float(y.mean())})
    return out


def disagreement(y, p, pm, cl) -> dict:
    """y ~ a + b1 logit(pm) + b2 (logit p - logit pm), clustered SEs."""
    X = np.column_stack([np.ones(len(p)), logit(pm), logit(p) - logit(pm)])
    b, V = logistic_fit(X, y, cl)
    se = np.sqrt(np.diag(V))
    d = logit(p) - logit(pm)
    return {"n": int(len(y)), "a": float(b[0]), "b_market": float(b[1]),
            "b_market_se": float(se[1]), "b_disagree": float(b[2]),
            "b_disagree_se": float(se[2]), "sd_disagree_logit": float(d.std()),
            "mean_abs_dp": float(np.mean(np.abs(p - pm)))}


def paired(y, p_new, p_old, cl) -> dict:
    """New minus old: log loss, Brier (mean diff, clustered SE), accuracy."""
    dl = ll_vec(y, p_new) - ll_vec(y, p_old)
    db = (y - p_new) ** 2 - (y - p_old) ** 2
    m, s = clustered_mean_se(dl, cl)
    mb, sb = clustered_mean_se(db, cl)
    return {"n": int(len(y)), "d_ll": m, "d_ll_se": s, "d_brier": mb, "d_brier_se": sb,
            "z_ll": m / s if s and s > 0 else None,
            "acc_new": float(np.mean((p_new > 0.5) == (y == 1))),
            "acc_old": float(np.mean((p_old > 0.5) == (y == 1)))}


def slope_diff_boot(y, p1, p2, cl, reps: int = 1000, seed: int = 0) -> dict:
    """slope(p1) - slope(p2) on the same games, SE from a date-cluster bootstrap."""
    y, p1, p2, cl = map(np.asarray, (y, p1, p2, cl))
    days, inv = np.unique(cl, return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(days))]

    def slope(rows, p):
        X = np.column_stack([np.ones(len(rows)), logit(p[rows])])
        return logistic_fit(X, y[rows])[0][1]
    allr = np.arange(len(y))
    d0 = slope(allr, p1) - slope(allr, p2)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(reps):
        rows = np.concatenate([idx[k] for k in rng.integers(0, len(days), len(days))])
        ds.append(slope(rows, p1) - slope(rows, p2))
    return {"diff": float(d0), "se": float(np.std(ds, ddof=1))}


# ------------------------------------------------------------------ data

def season_of(day: pd.Timestamp) -> str:
    s = day.year if day.month >= 7 else day.year - 1
    return f"{s}-{(s + 1) % 100:02d}"


def game_table(conn) -> pd.DataFrame:
    """game_id, season, season_type, game_date, home, away, y, team games-into-season (gis =
    the smaller of the two teams' prior RS games this season), market p and spread."""
    g = dataset.load_games(conn)
    g = g[["game_id", "season", "season_type", "game_date", "home", "away", "home_win"]].copy()
    g = g.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    rs = g["season_type"] == "Regular Season"
    n = {}
    gh, ga = np.full(len(g), -1), np.full(len(g), -1)
    for i, (s, h, a, r) in enumerate(zip(g["season"], g["home"], g["away"], rs)):
        if not r:
            continue
        gh[i], ga[i] = n.get((s, h), 0), n.get((s, a), 0)
        n[(s, h)] = gh[i] + 1
        n[(s, a)] = ga[i] + 1
    g["gp_home"], g["gp_away"] = gh, ga
    g["gis"] = np.minimum(gh, ga)
    cl = market.load_closing(conn)
    g = g.merge(cl[["game_id", "p_mkt", "home_spread_close", "source"]], on="game_id",
                how="left")
    return g


def phase_of(gis: np.ndarray) -> np.ndarray:
    out = np.empty(len(gis), dtype=object)
    for lo, hi, lab in PHASES:
        out[(gis >= lo) & (gis <= hi)] = lab
    return out


def load_wf(path: str, model: str, games: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"game_id": str})
    d["game_id"] = d["game_id"].str.zfill(10)
    d = d[(d["model"] == model) & (d["season_type"] == "Regular Season")]
    d = d[["game_id", "p", "margin_pred"]].merge(games, on="game_id", how="inner")
    d["y"] = d["home_win"].astype(float)
    return d.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)


def finish(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d")
    d["month"] = pd.to_datetime(d["game_date"]).dt.strftime("%m")
    d["phase"] = phase_of(d["gis"].to_numpy())
    d["has_mkt"] = d["p_mkt"].notna()
    ref = np.where(d["has_mkt"], d["p_mkt"], d["p"])
    fav = np.maximum(ref, 1 - ref)
    d["fav_bin"] = None
    for lo, hi, lab in FAV_BINS:
        d.loc[(fav >= lo) & (fav < hi), "fav_bin"] = lab
    return d


# ------------------------------------------------------------------ analysis blocks

def by_group(d: pd.DataFrame, col: str, with_market: bool = True) -> list[dict]:
    rows = []
    for k, g in d.groupby(col, sort=True):
        r = {col: k, "model": calib(g["y"], g["p"], g["date"])}
        gm = g[g["has_mkt"]]
        if with_market and len(gm) >= 30:
            r["model_on_mkt_games"] = calib(gm["y"], gm["p"], gm["date"])
            r["market"] = calib(gm["y"], gm["p_mkt"], gm["date"])
            r["vs_market"] = paired(gm["y"].to_numpy(), gm["p"].to_numpy(),
                                    gm["p_mkt"].to_numpy(), gm["date"])
            r["disagreement"] = disagreement(gm["y"].to_numpy(), gm["p"].to_numpy(),
                                             gm["p_mkt"].to_numpy(), gm["date"])
            r["slope_model_minus_market"] = slope_diff_boot(gm["y"], gm["p"], gm["p_mkt"],
                                                            gm["date"], reps=300)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ fixes

def _fit_platt(y, p):
    X = np.column_stack([np.ones(len(p)), logit(p)])
    b, _ = logistic_fit(X, y)
    return b


def _apply_platt(b, p):
    return sigmoid(b[0] + b[1] * logit(p))


def _fit_shrink(y, p, pm):
    """w maximising the likelihood of sigmoid(logit pm + w (logit p - logit pm))."""
    lm, dd = logit(pm), logit(p) - logit(pm)

    def nll(w):
        q = sigmoid(lm + w[0] * dd)
        return ll_vec(y, q).sum()
    return float(minimize(nll, [1.0], method="L-BFGS-B", bounds=[(0.0, 3.0)]).x[0])


def apply_fix(kind: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Recalibrated p for `test` from a layer fit on `train` (earlier seasons only)."""
    p = test["p"].to_numpy(float)
    if kind == "temp":
        return _apply_platt(_fit_platt(train["y"], train["p"]), p)
    if kind == "temp_phase":
        out = p.copy()
        for _, _, lab in PHASES:
            tr, m = train[train["phase"] == lab], (test["phase"] == lab).to_numpy()
            if m.any():
                out[m] = _apply_platt(_fit_platt(tr["y"], tr["p"]), p[m])
        return out
    if kind == "temp_early":
        # only the first phase gets its own layer; later games unchanged
        out = p.copy()
        lab = PHASES[0][2]
        tr, m = train[train["phase"] == lab], (test["phase"] == lab).to_numpy()
        if m.any():
            out[m] = _apply_platt(_fit_platt(tr["y"], tr["p"]), p[m])
        return out
    if kind == "market_only":
        out = p.copy()
        m = test["has_mkt"].to_numpy()
        out[m] = test["p_mkt"].to_numpy(float)[m]
        return out
    if kind == "shrink":
        out = p.copy()
        tr, m = train[train["has_mkt"]], test["has_mkt"].to_numpy()
        if m.any():
            w = _fit_shrink(tr["y"].to_numpy(), tr["p"].to_numpy(), tr["p_mkt"].to_numpy())
            pm = test["p_mkt"].to_numpy(float)[m]
            out[m] = sigmoid(logit(pm) + w * (logit(p[m]) - logit(pm)))
        return out
    if kind == "shrink_phase":
        out = _apply_platt(_fit_platt(train["y"], train["p"]), p)
        for _, _, lab in PHASES:
            tr = train[(train["phase"] == lab) & train["has_mkt"]]
            m = ((test["phase"] == lab) & test["has_mkt"]).to_numpy()
            if m.any() and len(tr) > 100:
                w = _fit_shrink(tr["y"].to_numpy(), tr["p"].to_numpy(), tr["p_mkt"].to_numpy())
                pm = test["p_mkt"].to_numpy(float)[m]
                out[m] = sigmoid(logit(pm) + w * (logit(p[m]) - logit(pm)))
        return out
    raise ValueError(kind)


FIXES = ("temp", "temp_phase", "temp_early", "shrink", "shrink_phase", "market_only")


def fix_params(kind: str, train: pd.DataFrame) -> dict:
    if kind == "temp":
        b = _fit_platt(train["y"], train["p"])
        return {"a": float(b[0]), "b": float(b[1])}
    if kind in ("temp_phase", "temp_early"):
        labs = [lab for _, _, lab in PHASES] if kind == "temp_phase" else [PHASES[0][2]]
        return {lab: dict(zip(("a", "b"), map(float, _fit_platt(
            train.loc[train["phase"] == lab, "y"], train.loc[train["phase"] == lab, "p"]))))
            for lab in labs}
    if kind == "shrink":
        tr = train[train["has_mkt"]]
        return {"w": _fit_shrink(tr["y"].to_numpy(), tr["p"].to_numpy(), tr["p_mkt"].to_numpy())}
    if kind == "shrink_phase":
        out = {}
        for _, _, lab in PHASES:
            tr = train[(train["phase"] == lab) & train["has_mkt"]]
            out[lab] = _fit_shrink(tr["y"].to_numpy(), tr["p"].to_numpy(),
                                   tr["p_mkt"].to_numpy())
        return out
    return {}


def evaluate_fixes(d: pd.DataFrame, fix_from: str) -> dict:
    seasons = sorted(d["season"].unique())
    test_seasons = [s for s in seasons if s >= fix_from]
    res = {}
    for kind in FIXES:
        per, newp = [], []
        for s in test_seasons:
            tr, te = d[d["season"] < s], d[d["season"] == s]
            q = apply_fix(kind, tr, te)
            pr = paired(te["y"].to_numpy(), q, te["p"].to_numpy(), te["date"])
            pr.update({"season": s, "slope_new": calib(te["y"], q, te["date"]).get("slope"),
                       "slope_old": calib(te["y"], te["p"], te["date"]).get("slope"),
                       "params": fix_params(kind, tr)})
            per.append(pr)
            newp.append(pd.DataFrame({"game_id": te["game_id"], "q": q}))
        nq = pd.concat(newp).merge(d, on="game_id")
        pooled = paired(nq["y"].to_numpy(), nq["q"].to_numpy(), nq["p"].to_numpy(),
                        nq["date"])
        phase = {lab: paired(g["y"].to_numpy(), g["q"].to_numpy(), g["p"].to_numpy(),
                             g["date"]) for lab, g in nq.groupby("phase")}
        worst = max(per, key=lambda r: (r["d_ll"] / r["d_ll_se"]) if r["d_ll_se"] else 0)
        bar = (pooled["d_ll"] <= -2 * pooled["d_ll_se"]) and all(
            r["d_ll"] <= r["d_ll_se"] for r in per)
        res[kind] = {"pooled": pooled, "by_phase": phase, "by_season": per,
                     "worst_season": worst["season"],
                     "worst_season_z": worst["d_ll"] / worst["d_ll_se"],
                     "pooled_slope_new": calib(nq["y"], nq["q"], nq["date"]).get("slope"),
                     "pooled_slope_old": calib(nq["y"], nq["p"], nq["date"]).get("slope"),
                     "clears_bar": bool(bar)}
    return res


# ------------------------------------------------------------------ report

def fmt_cal(c: dict) -> str:
    if "slope" not in c:
        return f"n={c.get('n')}"
    return (f"n={c['n']:5d} LL {c['log_loss']:.4f} slope {c['slope']:.3f}±{c['slope_se']:.3f} "
            f"int {c['intercept']:+.3f} z {c['z']:+.2f} (zc {c['z_cluster']:+.2f})")


def _c(c: dict, key: str = "slope") -> str:
    if key not in c:
        return "—"
    return f"{c[key]:.2f} ({c[key + '_se']:.2f})"


def tables_md(res: dict) -> str:
    """Markdown tables for docs/research/game-calibration.md."""
    L = ["### Calibration by season (Regular Season, walk-forward)", "",
         "| season | n | model LL | model slope (SE) | model z | market LL | market slope (SE) "
         "| market z | model − market slope (SE) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["by_season"]:
        m, k = r["model"], r.get("market", {})
        dd = r.get("slope_model_minus_market", {})
        L.append(f"| {r['season']} | {m['n']} | {m['log_loss']:.4f} | {_c(m)} | {m['z']:+.2f} | "
                 f"{k.get('log_loss', float('nan')):.4f} | {_c(k)} | {k.get('z', float('nan')):+.2f}"
                 f" | {dd.get('diff', float('nan')):+.3f} ({dd.get('se', float('nan')):.3f}) |")
    pm, pk = res["pooled"]["model"], res["pooled"]["market"]
    dd = res["pooled"]["slope_model_minus_market"]
    L.append(f"| **all** | {pm['n']} | {pm['log_loss']:.4f} | {_c(pm)} | {pm['z']:+.2f} | "
             f"{pk['log_loss']:.4f} | {_c(pk)} | {pk['z']:+.2f} | {dd['diff']:+.3f} "
             f"({dd['se']:.3f}) |")
    for col, title in (("phase", "team games into the season"), ("month", "month"),
                       ("fav_bin", "favourite size (market favourite's probability)")):
        L += ["", f"### By {title} (all seasons)", "",
              "| group | n | model slope (SE) | model z | market slope (SE) | market z | "
              "model − market slope (SE) | LL model − market (SE) | b_disagree (SE) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res[f"by_{col}"]:
            if "market" not in r:
                continue
            m, k, v = r["model"], r["market"], r["vs_market"]
            dd, ds = r["slope_model_minus_market"], r["disagreement"]
            L.append(f"| {r[col]} | {m['n']} | {_c(m)} | {m['z']:+.2f} | {_c(k)} | "
                     f"{k['z']:+.2f} | {dd['diff']:+.3f} ({dd['se']:.3f}) | "
                     f"{v['d_ll']:+.4f} ({v['d_ll_se']:.4f}) | {ds['b_disagree']:.2f} "
                     f"({ds['b_disagree_se']:.2f}) |")
    L += ["", "### 2025-26 by window, phase and month", "",
          "| group | n | model slope (SE) | model z | market slope (SE) | market z | "
          "model − market slope (SE) |", "|---|---|---|---|---|---|---|"]
    for key, col in (("current_by_window", "window"), ("current_by_phase", "phase"),
                     ("current_by_month", "month")):
        for r in res[key]:
            if "market" not in r:
                continue
            m, k, dd = r["model"], r["market"], r["slope_model_minus_market"]
            L.append(f"| {r[col]} | {m['n']} | {_c(m)} | {m['z']:+.2f} | {_c(k)} | "
                     f"{k['z']:+.2f} | {dd['diff']:+.3f} ({dd['se']:.3f}) |")
    if res.get("replays"):
        L += ["", "### Day-by-day replays of 2025-26", "",
              "| replay | games | arm | n | LL | slope (SE) | z |", "|---|---|---|---|---|---|---|"]
        for lab, o in res["replays"].items():
            for arm, key in (("replay", "replay"), ("walk-forward, same games",
                                                   "walk_forward_same_games"),
                             ("market, same games", "market_same_games")):
                c = o[key]
                L.append(f"| {lab} | {o['first']} → {o['last']} | {arm} | {c['n']} | "
                         f"{c['log_loss']:.4f} | {_c(c)} | {c['z']:+.2f} |")
            ig = o.get("injury_games")
            if ig:
                for arm in ("replay", "walk_forward", "market"):
                    c = ig[arm]
                    L.append(f"| {lab} | injury report, both teams ({o['injury_both_sides']}) | "
                             f"{arm} | {c['n']} | {c['log_loss']:.4f} | {_c(c)} | {c['z']:+.2f} |")
    L += ["", "### Candidate fixes (walk-forward; layer fit on earlier seasons only)", "",
          "| fix | LL change (SE) | z | Brier change (SE) | accuracy new / old | slope new / old "
          "| worst season (z) | clears bar |", "|---|---|---|---|---|---|---|---|"]
    for k, f in res["fixes"].items():
        p = f["pooled"]
        L.append(f"| {k} | {p['d_ll']:+.5f} ({p['d_ll_se']:.5f}) | {p['z_ll']:+.2f} | "
                 f"{p['d_brier']:+.5f} ({p['d_brier_se']:.5f}) | {p['acc_new']:.4f} / "
                 f"{p['acc_old']:.4f} | {f['pooled_slope_new']:.3f} / {f['pooled_slope_old']:.3f}"
                 f" | {f['worst_season']} ({f['worst_season_z']:+.2f}) | "
                 f"{'yes' if f['clears_bar'] else 'no'} |")
    L += ["", "### Per-season LL change for the shrink-toward-the-close fix", "",
          "| season | w (fit on earlier seasons) | LL change (SE) | z | slope old → new |",
          "|---|---|---|---|---|"]
    for r in res["fixes"]["shrink"]["by_season"]:
        L.append(f"| {r['season']} | {r['params']['w']:.2f} | {r['d_ll']:+.5f} "
                 f"({r['d_ll_se']:.5f}) | {r['d_ll'] / r['d_ll_se']:+.2f} | "
                 f"{r['slope_old']:.3f} → {r['slope_new']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default="gbm_market")
    ap.add_argument("--replay", action="append", default=[], help="label=path (csv)")
    ap.add_argument("--fix-from", default="2014-15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None, help="also write markdown tables here")
    ap.add_argument("--window-end", default="2026-02-20",
                    help="split the latest season here (the original replay's last day)")
    a = ap.parse_args()

    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    games = game_table(conn)
    d = finish(load_wf(a.preds, a.model, games))
    res: dict = {"model": a.model, "preds": a.preds, "n": int(len(d))}

    print(f"== {a.model}: walk-forward, Regular Season, {len(d)} games")
    res["by_season"] = by_group(d, "season")
    for r in res["by_season"]:
        print(f"{r['season']}  model {fmt_cal(r['model'])}")
        if "market" in r:
            print(f"         market {fmt_cal(r['market'])}  (model on same: "
                  f"slope {r['model_on_mkt_games']['slope']:.3f})")
    dm = d[d["has_mkt"]]
    res["pooled"] = {"model": calib(d["y"], d["p"], d["date"]),
                     "model_on_mkt_games": calib(dm["y"], dm["p"], dm["date"]),
                     "market": calib(dm["y"], dm["p_mkt"], dm["date"]),
                     "disagreement": disagreement(dm["y"].to_numpy(), dm["p"].to_numpy(),
                                                  dm["p_mkt"].to_numpy(), dm["date"])}
    print("pooled model", fmt_cal(res["pooled"]["model"]))
    print("pooled mkt  ", fmt_cal(res["pooled"]["market"]))
    res["pooled"]["slope_model_minus_market"] = slope_diff_boot(dm["y"], dm["p"], dm["p_mkt"],
                                                                dm["date"], reps=300)
    print("pooled slope(model)-slope(mkt)", res["pooled"]["slope_model_minus_market"])
    print("disagreement", res["pooled"]["disagreement"])
    for col in ("phase", "month", "fav_bin", "has_mkt"):
        res[f"by_{col}"] = by_group(d, col, with_market=col != "has_mkt")
        print(f"-- by {col}")
        for r in res[f"by_{col}"]:
            line = f"{str(r[col]):>8}  model {fmt_cal(r['model'])}"
            if "market" in r:
                line += (f" | mkt slope {r['market']['slope']:.3f}±{r['market']['slope_se']:.3f}"
                         f" dLL {r['vs_market']['d_ll']:+.4f}±{r['vs_market']['d_ll_se']:.4f}"
                         f" b2 {r['disagreement']['b_disagree']:.2f}"
                         f"±{r['disagreement']['b_disagree_se']:.2f}"
                         f" dslope {r['slope_model_minus_market']['diff']:+.3f}"
                         f"±{r['slope_model_minus_market']['se']:.3f}")
            print(line)
    # recent seasons by phase
    rec = d[d["season"] >= "2023-24"]
    res["by_phase_last3"] = by_group(rec, "phase")
    print("-- by phase, 2023-24..2025-26")
    for r in res["by_phase_last3"]:
        print(f"{r['phase']:>8}  model {fmt_cal(r['model'])} | mkt slope "
              f"{r['market']['slope']:.3f}±{r['market']['slope_se']:.3f}")
    # 2025-26 by phase and the replay window
    cur = d[d["season"] == d["season"].max()]
    res["current_by_phase"] = by_group(cur, "phase")
    res["current_by_month"] = by_group(cur, "month")
    print(f"-- {cur['season'].iloc[0]} by phase / month / replay window")
    for key, col in (("current_by_phase", "phase"), ("current_by_month", "month")):
        for r in res[key]:
            if "market" in r:
                print(f"{str(r[col]):>8}  model {fmt_cal(r['model'])} | mkt slope "
                      f"{r['market']['slope']:.3f}±{r['market']['slope_se']:.3f}")
    cut = a.window_end      # last day of the original 838-game replay
    cur = cur.assign(window=np.where(cur["date"] <= cut, f"to {cut[5:]}", f"after {cut[5:]}"))
    res["current_by_window"] = by_group(cur, "window")
    for r in res["current_by_window"]:
        print(f"{r['window']:>8}  model {fmt_cal(r['model'])} | mkt "
              f"{fmt_cal(r['market'])} | slope diff "
              f"{r['slope_model_minus_market']['diff']:+.3f}"
              f"±{r['slope_model_minus_market']['se']:.3f}")

    # replays
    res["replays"] = {}
    for spec in a.replay:
        lab, path = spec.split("=", 1)
        r = pd.read_csv(path, dtype={"game_id": str})
        r["game_id"] = r["game_id"].str.zfill(10)
        if "model" in r:
            r = r[r["model"] == a.model]
        r = r.rename(columns={"p": "p_rep"})[["game_id", "p_rep"] + [
            c for c in ("basis_home", "basis_away") if c in r]]
        m = r.merge(d, on="game_id", how="inner")
        out = {"n": int(len(m)), "first": m["date"].min(), "last": m["date"].max(),
               "replay": calib(m["y"], m["p_rep"], m["date"]),
               "walk_forward_same_games": calib(m["y"], m["p"], m["date"])}
        mm = m[m["has_mkt"]]
        out["market_same_games"] = calib(mm["y"], mm["p_mkt"], mm["date"])
        out["replay_on_mkt_games"] = calib(mm["y"], mm["p_rep"], mm["date"])
        out["replay_vs_market"] = paired(mm["y"].to_numpy(), mm["p_rep"].to_numpy(),
                                         mm["p_mkt"].to_numpy(), mm["date"])
        out["replay_vs_wf"] = paired(m["y"].to_numpy(), m["p_rep"].to_numpy(),
                                     m["p"].to_numpy(), m["date"])
        if "basis_home" in m:
            inj = (m["basis_home"] == "injury_report") & (m["basis_away"] == "injury_report")
            out["injury_both_sides"] = int(inj.sum())
            sub = m[inj]
            if len(sub) >= 30:
                out["injury_games"] = {
                    "replay": calib(sub["y"], sub["p_rep"], sub["date"]),
                    "walk_forward": calib(sub["y"], sub["p"], sub["date"]),
                    "replay_vs_wf": paired(sub["y"].to_numpy(), sub["p_rep"].to_numpy(),
                                           sub["p"].to_numpy(), sub["date"])}
                sm = sub[sub["has_mkt"]]
                out["injury_games"]["market"] = calib(sm["y"], sm["p_mkt"], sm["date"])
                out["injury_games"]["replay_vs_market"] = paired(
                    sm["y"].to_numpy(), sm["p_rep"].to_numpy(), sm["p_mkt"].to_numpy(),
                    sm["date"])
        out["replay_minus_market_slope"] = slope_diff_boot(
            mm["y"], mm["p_rep"], mm["p_mkt"], mm["date"])
        res["replays"][lab] = out
        print(f"== replay {lab}: {out['n']} games {out['first']}..{out['last']}")
        print("  replay        ", fmt_cal(out["replay"]))
        print("  walk-forward  ", fmt_cal(out["walk_forward_same_games"]))
        print("  market        ", fmt_cal(out["market_same_games"]))
        print("  replay-wf dLL  {d_ll:+.4f}±{d_ll_se:.4f}".format(**out["replay_vs_wf"]))
        print("  replay-mkt dLL {d_ll:+.4f}±{d_ll_se:.4f}".format(**out["replay_vs_market"]),
              " slope(replay)-slope(mkt) {diff:+.3f}±{se:.3f}".format(
                  **out["replay_minus_market_slope"]))
        if "injury_games" in out:
            ig = out["injury_games"]
            print(f"  injury-report games ({out['injury_both_sides']}):")
            for k in ("replay", "walk_forward", "market"):
                print(f"    {k:<13}", fmt_cal(ig[k]))
            print("    replay-wf dLL {d_ll:+.4f}±{d_ll_se:.4f}".format(**ig["replay_vs_wf"]))

    res["fixes"] = evaluate_fixes(d, a.fix_from)
    print(f"== fixes (walk-forward, test seasons >= {a.fix_from})")
    for k, f in res["fixes"].items():
        p = f["pooled"]
        print(f"{k:<13} dLL {p['d_ll']:+.5f}±{p['d_ll_se']:.5f} (z {p['z_ll']:+.2f}) "
              f"dBrier {p['d_brier']:+.5f}±{p['d_brier_se']:.5f} acc {p['acc_new']:.4f} vs "
              f"{p['acc_old']:.4f} slope {f['pooled_slope_new']:.3f} vs "
              f"{f['pooled_slope_old']:.3f} worst season {f['worst_season']} "
              f"z {f['worst_season_z']:+.2f} bar={f['clears_bar']}")
        for lab, ph in f["by_phase"].items():
            print(f"     phase {lab:>6} dLL {ph['d_ll']:+.5f}±{ph['d_ll_se']:.5f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {a.out}")
    if a.md:
        Path(a.md).write_text(tables_md(res))
        print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
