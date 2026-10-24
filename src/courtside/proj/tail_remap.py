"""Upper-tail quantile remap of each player's played-sim DK distribution (tr; IMPROVEMENTS.md D6,
docs/research/tail-remap.md; the evidence: docs/research/xfactor.md §2).

The simulated upper tails are too wide: conditional on playing, a bench player at 16-24 minutes
ends above his played-sims q99 about 0.4 % of the time instead of 1 %. The remap fixes the shape
of the played distribution per model tier x minutes band, fit on seasons < S (walk-forward):

    q_c(v) = v-quantile of U for v in LEVELS,     U = F_played(y) on rows where the player played
    x'     = Q_played(g_c(F_played(x)))           for each played sim x of a player in cell c

* **U** is the PIT of the outcome in the player's played-sim distribution (harness rows:
  u_c = (pit - (1 - p_play)) / p_play, exact for y > 0 since the DNP sims are 0).
* **g_c**: piecewise linear through (0, 0), (0.5, 0.5), (v, q_c(v)) for v in LEVELS, (1, 1), i.e.
  the inverse of h_c(t) = P(U <= t), exact at the grid levels: the remapped v-quantile is the old
  q_c(v)-quantile, which the outcome exceeds with share 1 - v. Monotone increasing; identity at and
  below the median (g(0.5) = 0.5 is imposed), so the median and everything under it are untouched;
  it only re-labels the upper half of the player's own sims, so each sim keeps its rank and the
  joint (copula) rank structure is unchanged.
* **cells**: tier (copula info `tier`: bench / role / star) x minutes band of E[min | plays]
  (MIN_BANDS, as xfactor_calibration). A cell's q is shrunk toward its tier's, the tier's toward
  the pooled one, with prior weight SHRINK rows (cells with few rows end up near the tier).
* Applied in `pipeline.run_games` after the draws and after the role blend (if on), before the
  summary, so the stored projections, the sims npz, the cards, the X factor and the contest builder
  all see it (in place on GameSims.fp; info gets `tr_shift` = mean change of the played sims and
  `mu_played` moves by it).
* Switch `COURTSIDE_TAIL_REMAP` (default OFF; `pipeline.set_tail_remap`): model version
  `<arm>+tr` (after `+tm`), same bundle, same seeds. OFF: nothing of it runs (bit-identical).
* Params: `tail_remap_params.json`, written by `research/tail_remap_eval.py fit`.

Public API
  TAIL_REMAP, set_tail_remap(on), cells(season) -> {cell: q (len(LEVELS))}
  cell_of(tier, min_played) -> "tier|band"; mapping(q) -> (V, T) knots of g
  remap_column(x, V, T) -> remapped played values (same order)
  apply(sims, day) -> {remapped, mean_shift, mean_abs_shift}
  fit_cells(frame) -> {"cells", "tiers", "pooled", "n"}   (frame: tier, min_played, uc)
"""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TAIL_REMAP_DEFAULT = True             # shipped 2026-10-24 (owner OK; env still overrides)
TAIL_REMAP_ENV = "COURTSIDE_TAIL_REMAP"
TAIL_REMAP_TAG = "+tr"
TR_PATH = Path(__file__).with_name("tail_remap_params.json")

LEVELS = (0.75, 0.90, 0.95, 0.98, 0.99)          # knots above the fixed median
MEDIAN = 0.5
MIN_BANDS = ((0, 8, "<8"), (8, 16, "8-16"), (16, 24, "16-24"), (24, 32, "24-32"),
             (32, 999, "32+"))
TIERS = ("bench", "role", "star")
SHRINK = 2000.0          # prior weight (rows) of the parent level when a cell's h is estimated
MIN_PLAYED = 20          # a player needs this many played sims to be remapped
EPS = 1e-4               # strict monotonicity of the knots


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


TAIL_REMAP = _flag(TAIL_REMAP_ENV, TAIL_REMAP_DEFAULT)


def set_tail_remap(on: bool) -> None:
    """Flip the switch at run time (`pipeline.set_tail_remap` also updates the model version)."""
    global TAIL_REMAP
    TAIL_REMAP = bool(on)


# ============================================================================== parameters

def _read(path: Path | None = None) -> dict:
    try:
        return json.loads(Path(path or TR_PATH).read_text())
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=None)
def cells(season: str) -> dict:
    """{cell: h} for season S (fit on seasons < S); a season missing from the file uses the latest
    earlier one (still walk-forward); none at all -> {} (no remap)."""
    seasons = _read().get("seasons", {})
    key = season if season in seasons else max((s for s in seasons if s < season), default=None)
    if key is None:
        return {}
    e = seasons[key]
    out = {f"{t}|*": list(map(float, h)) for t, h in e.get("tiers", {}).items()}
    out.update({c: list(map(float, v["q"])) for c, v in e.get("cells", {}).items()})
    return out


def band_of(min_played: float) -> str | None:
    if min_played is None or not np.isfinite(min_played):
        return None
    for lo, hi, lab in MIN_BANDS:
        if lo <= min_played < hi:
            return lab
    return None


def cell_of(tier, min_played: float) -> str:
    return f"{tier}|{band_of(min_played)}"


def lookup(table: dict, tier, min_played: float) -> list | None:
    """q of the player's cell, else of his tier ('tier|*'), else None (left as is)."""
    return table.get(cell_of(tier, min_played)) or table.get(f"{tier}|*")


def mapping(q) -> tuple[np.ndarray, np.ndarray]:
    """Knots (V, T) of g: V = (0, .5, LEVELS..., 1), T = (0, .5, q..., 1), T strictly increasing
    (q is clipped into (0.5, 1) and made monotone)."""
    t = [0.0, MEDIAN]
    for i, x in enumerate(q):
        t.append(float(np.clip(x, t[-1] + EPS, 1.0 - EPS * (len(LEVELS) - i))))
    return np.array([0.0, MEDIAN, *LEVELS, 1.0]), np.array(t + [1.0])


def remap_column(x: np.ndarray, V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """x' = Q(g(F(x))) on one player's played values (any order, returned in the same order):
    F = mid-rank empirical CDF (tied values stay tied), Q = linear interpolation of the sorted
    values at position g(u)·n − 0.5. Values at or below the median are returned unchanged."""
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return x.copy()
    s = np.sort(x.astype(np.float64))
    lo = np.searchsorted(s, x, "left")
    hi = np.searchsorted(s, x, "right")
    u = 0.5 * (lo + hi) / n
    up = u > MEDIAN
    out = x.astype(np.float64).copy()
    if up.any():
        pos = np.interp(u[up], V, T) * n - 0.5
        out[up] = np.interp(pos, np.arange(n, dtype=np.float64), s)
    return out


def apply(sims: list, day) -> dict:
    """Remap each player's played sims in place (GameSims.fp; info gets `tr_shift` and
    `mu_played` + shift). DNP sims stay 0; players with < MIN_PLAYED played sims or no cell are
    left as they are."""
    from .pipeline import season_of
    d = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    table = cells(season_of(d))
    n_re, tot, tot_abs = 0, 0.0, 0.0
    for g in sims:
        played = g.minutes > 0
        npl = played.sum(0)
        tiers = g.info["tier"].to_numpy() if "tier" in g.info else np.full(len(npl), None)
        shift = np.zeros(len(npl))
        if table:
            fp = np.array(g.fp, dtype=np.float32, copy=True)
            for j in range(fp.shape[1]):
                if npl[j] < MIN_PLAYED:
                    continue
                m = played[:, j]
                mp = float(g.minutes[m, j].astype(np.float64).mean())
                q = lookup(table, tiers[j], mp)
                if q is None:
                    continue
                V, T = mapping(q)
                x = fp[m, j]
                y = remap_column(x, V, T).astype(np.float32)
                shift[j] = float(y.astype(np.float64).mean() - x.astype(np.float64).mean())
                fp[m, j] = y
                n_re += 1
            g.fp = fp
        g.info = g.info.copy()
        if "mu_played" in g.info:
            g.info["mu_played"] = g.info["mu_played"].to_numpy(float) + shift
        g.info["tr_shift"] = shift
        tot += float(shift.sum())
        tot_abs += float(np.abs(shift).sum())
    return {"remapped": n_re, "mean_shift": tot / n_re if n_re else 0.0,
            "mean_abs_shift": tot_abs / n_re if n_re else 0.0}


# ============================================================================== fitting

def cond_pit(pit, p_play) -> np.ndarray:
    """PIT in the played distribution from the harness's unconditional randomised PIT (rows where
    the player played): (pit − (1 − p)) / p, clipped to [0, 1]."""
    p = np.clip(np.asarray(p_play, float), 1e-6, 1.0)
    return np.clip((np.asarray(pit, float) - (1.0 - p)) / p, 0.0, 1.0)


def _h(u: np.ndarray) -> np.ndarray:
    """Quantiles of U at LEVELS."""
    return (np.quantile(u, LEVELS) if len(u) else np.full(len(LEVELS), np.nan))


def fit_cells(fr: pd.DataFrame, shrink: float = SHRINK) -> dict:
    """q per cell from a frame with columns tier, min_played, uc (played rows): pooled -> tier ->
    cell, each level shrunk toward its parent with `shrink` prior rows."""
    fr = fr[np.isfinite(fr["uc"].to_numpy(float)) & fr["tier"].isin(TIERS)]
    u = fr["uc"].to_numpy(float)
    pooled = _h(u)
    tiers, cells_, n = {}, {}, {"pooled": int(len(fr))}
    band = fr["min_played"].map(band_of)
    for t in TIERS:
        m = (fr["tier"] == t).to_numpy()
        k = int(m.sum())
        ht = (k * _h(u[m]) + shrink * pooled) / (k + shrink) if k else pooled
        tiers[t] = [float(x) for x in ht]
        n[f"{t}|*"] = k
        for _, _, lab in MIN_BANDS:
            mc = m & (band == lab).to_numpy()
            kc = int(mc.sum())
            if not kc:
                continue
            hc = (kc * _h(u[mc]) + shrink * ht) / (kc + shrink)
            cells_[f"{t}|{lab}"] = {"n": kc, "q": [float(x) for x in hc]}
    return {"cells": cells_, "tiers": tiers, "pooled": [float(x) for x in pooled], "n": n}
