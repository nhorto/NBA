"""Probability and margin metrics. Pure numpy; y is 1 for a home win, p = P(home win)."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def _arr(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, p


def log_loss(y, p, eps: float = EPS) -> float:
    y, p = _arr(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = _arr(y, p)
    return float(np.mean((p - y) ** 2))


def accuracy(y, p) -> float:
    """Share of games where the side with p >= 0.5 (home on an exact tie) won."""
    y, p = _arr(y, p)
    return float(np.mean((p >= 0.5) == (y == 1)))


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Equal-width probability bins on [0, 1]. Each row: lo, hi, n, mean_p, frac_pos.
    Empty bins are included with n=0 and None means."""
    y, p = _arr(y, p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "mean_p": float(p[m].mean()) if n else None,
                     "frac_pos": float(y[m].mean()) if n else None})
    return rows


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: sum_b (n_b / N) * |mean_p_b - frac_pos_b|."""
    y, p = _arr(y, p)
    tab = calibration_table(y, p, bins)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["mean_p"] - r["frac_pos"]) for r in tab if r["n"]))


def mae(actual, pred) -> float:
    a, q = _arr(actual, pred)
    return float(np.mean(np.abs(a - q)))


def rmse(actual, pred) -> float:
    a, q = _arr(actual, pred)
    return float(np.sqrt(np.mean((a - q) ** 2)))


def summarize(y, p, margin=None, margin_pred=None, bins: int = 10) -> dict:
    out = {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p),
           "accuracy": accuracy(y, p), "ece": ece(y, p, bins), "mean_p": float(np.mean(p)),
           "home_win_rate": float(np.mean(y))}
    if margin is not None and margin_pred is not None:
        out["margin_mae"] = mae(margin, margin_pred)
        out["margin_rmse"] = rmse(margin, margin_pred)
    return out
