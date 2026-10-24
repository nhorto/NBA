"""Environment and reconciliation to team totals (docs/PLAYERS.md §4.4, T1.4).

* **Implied team points** (N1): the median across books of the book team totals at the run
  time; without them (T + M)/2 and (T - M)/2 from the market spread/total (closing line in a
  backtest), else from `gbm_market`.
* **Scaling.** In the Path A sim, each player's per-sim points are rate x minutes x g_s, with
  g_s = H_s / E[H] the sim's team-points draw relative to its mean, and his other DK stats are
  scaled by g_s^alpha (alpha = pace share: the elasticity of the non-points DK stats to team
  points, fit on seasons < S). Then one team factor c makes
  sum_i E[pts_i] = implied exactly (c on points, c^alpha on the other stats: the alpha split of
  pace vs efficiency). Because the factor is computed on the same sims it applies to, this is
  the §4.4 pilot + Newton step without the pilot: the residual is 0 up to float error.
* **Acceptance:** |sum_i E[pts_i] - target| <= 0.5 for every team-game; a pre-reconciliation gap
  (sum of rate x minutes points before any scaling vs the target) > 6 raises a data check
  (stale report, mis-matched name, missing player); c is capped to [0.75, 1.33] (the remainder
  is reported and fails acceptance).
* **Reserve** (rr, `RECON_RESERVE`, default OFF; env ``COURTSIDE_RECON_RESERVE=1``,
  `set_recon_reserve`; model version `<arm>+rr`, same bundle and seeds; IMPROVEMENTS.md B2,
  docs/research/recon-reserve.md). The minutes engine keeps E[reserve] team minutes for players
  outside the candidate set, so the candidates should not score the whole implied total: the
  target becomes implied - E[reserve minutes] x the league reserve points per minute (box rows
  outside a team's last-10-game players, pooled over the 3 seasons < S:
  `recon_reserve_params.json`, `research/recon_reserve.py fit`). OFF: target = implied.
* **Weighted** (rw, `RECON_WEIGHTED`, default OFF; env ``COURTSIDE_RECON_WEIGHTED=1``,
  `set_recon_weighted`; model version `<arm>+rw`, same bundle and seeds; IMPROVEMENTS.md P4,
  docs/research/p4-p6.md). The gap is not spread by one factor c but by Gaussian conditioning on
  the team sum: player i's expected points e_i carry a team-common relative error (variance
  theta x e_i^2, the market's pace / efficiency news) and his own rate error (variance
  V_i = e_i^2 / (r_i (n_i + kappa)): the EB points rate r_i, its evidence minutes n_i and prior
  minutes kappa), so E[delta_i | gap] = gap x (theta e_i e + V_i) / (theta e^2 + sum V), e = sum
  e_i. New players (n small) and promoted ones (e_i large for their n) absorb most of it; theta
  -> infinity is the uniform c. Per player c_i = 1 + delta_i / e_i in PLAYER_BOUNDS (the rest
  goes to the unclipped players; a remainder after that gets one uniform factor in SCALE_BOUNDS),
  the other DK stats move by c_i^alpha. theta per season S is fit on seasons < S
  (`recon_weighted_params.json`, `research/p4_p6_eval.py rwfit`). A pre-reconciliation |gap| >
  GAP_CHECK is logged as a projection error with the players that absorbed it. OFF: nothing of
  it runs.

Public API
  implied_points(mu_margin, mu_total, home_tt=None, away_tt=None) -> (home, away)
  reconcile_team(raw_pts, raw_oth, g, alpha, target) -> dict(pts, oth, scale, baseline, ...)
  acceptance(team_infos) -> dict;  check_rows(team_infos, run_label) -> data_checks rows
  reserve_target(target, reserve_min, season) -> float;  bench_ppm(season) -> float
  rate_variance(e, r_pts, n_pts, kappa) -> V;  rw_theta(season) -> float
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

TOL = 0.5
GAP_CHECK = 6.0
SCALE_BOUNDS = (0.75, 1.33)

RECON_RESERVE_DEFAULT = False          # shipping = True here (env still overrides)
RECON_RESERVE_ENV = "COURTSIDE_RECON_RESERVE"
RECON_RESERVE_TAG = "+rr"
RR_PATH = Path(__file__).with_name("recon_reserve_params.json")
RR_FALLBACK_PPM = 0.42                 # no table / no season <= S in it

RECON_WEIGHTED_DEFAULT = False         # shipping = True here (env still overrides)
RECON_WEIGHTED_ENV = "COURTSIDE_RECON_WEIGHTED"
RECON_WEIGHTED_TAG = "+rw"
RW_PATH = Path(__file__).with_name("recon_weighted_params.json")
RW_FALLBACK_THETA = 1e-4               # no table / no season <= S in it
PLAYER_BOUNDS = (0.5, 2.0)             # per-player factor c_i (weighted reconciliation)
RW_MIN_R = 0.05                        # floor of the points rate in V_i
RW_ITERS = 8
log = logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


RECON_RESERVE = _flag(RECON_RESERVE_ENV, RECON_RESERVE_DEFAULT)


def set_recon_reserve(on: bool) -> None:
    """Flip the reconciliation-reserve switch at run time (evaluation scripts, tests)."""
    global RECON_RESERVE
    RECON_RESERVE = bool(on)


RECON_WEIGHTED = _flag(RECON_WEIGHTED_ENV, RECON_WEIGHTED_DEFAULT)


def set_recon_weighted(on: bool) -> None:
    """Flip the weighted-reconciliation switch at run time (evaluation scripts, tests)."""
    global RECON_WEIGHTED
    RECON_WEIGHTED = bool(on)


@lru_cache(maxsize=None)
def rw_theta(season: str) -> float:
    """The team-common relative variance theta for season S (fit on seasons < S); a season
    missing from the table uses the latest earlier one (still walk-forward)."""
    try:
        th = json.loads(RW_PATH.read_text())["seasons"]
    except (OSError, ValueError, KeyError):
        return RW_FALLBACK_THETA
    if season in th:
        return float(th[season]["theta"])
    prev = [s for s in th if s < season]
    return float(th[max(prev)]["theta"]) if prev else RW_FALLBACK_THETA


def rate_variance(e: np.ndarray, r_pts: np.ndarray, n_pts: np.ndarray, kappa: float
                  ) -> np.ndarray:
    """V_i (up to a constant): the variance of player i's expected points e_i from his EB points
    rate, e_i^2 x Var(r_i) / r_i^2 with Var(r_i) ~ r_i / (n_i + kappa) (n_i = his decayed
    evidence minutes, kappa the prior's pseudo-minutes)."""
    e = np.asarray(e, float)
    r = np.maximum(np.nan_to_num(np.asarray(r_pts, float), nan=RW_MIN_R), RW_MIN_R)
    n = np.maximum(np.nan_to_num(np.asarray(n_pts, float), nan=0.0), 0.0)
    return e * e / (r * (n + max(float(kappa), 1.0)))


def weighted_factors(e: np.ndarray, v: np.ndarray, target: float, theta: float,
                     bounds: tuple = PLAYER_BOUNDS, iters: int = RW_ITERS) -> np.ndarray:
    """Per-player factors c (n,) with sum e_i c_i = target: the gap split in proportion to
    w_i = theta e_i e + V_i (Gaussian conditioning on the team sum), c_i clipped to `bounds`
    and the remainder re-split over the players not at a bound. Players with e_i <= 0 keep 1."""
    e = np.asarray(e, float)
    w_all = theta * e * max(float(e.sum()), 1e-9) + np.asarray(v, float)
    c = np.ones(len(e))
    free = e > 0
    for _ in range(iters):
        rem = float(target) - float((e * c).sum())
        if abs(rem) < 1e-9 or not free.any():
            break
        w = np.where(free, w_all, 0.0)
        sw = float(w.sum())
        if sw <= 0:
            break
        c = np.where(free, c + rem * w / (np.maximum(e, 1e-12) * sw), c)
        lo, hi = c < bounds[0], c > bounds[1]
        c = np.clip(c, *bounds)
        free &= ~(lo | hi)
    return c


@lru_cache(maxsize=None)
def bench_ppm(season: str) -> float:
    """League reserve points per minute for season S (fit on seasons < S); a season missing
    from the table uses the latest earlier one (still walk-forward)."""
    try:
        rates = json.loads(RR_PATH.read_text())["rates"]
    except (OSError, ValueError, KeyError):
        return RR_FALLBACK_PPM
    if season in rates:
        return float(rates[season]["ppm"])
    prev = [s for s in rates if s < season]
    return float(rates[max(prev)]["ppm"]) if prev else RR_FALLBACK_PPM


def reserve_target(target: float, reserve_min: float, season: str) -> float:
    """The candidates' share of the implied total: implied - E[reserve minutes] x bench rate."""
    if not np.isfinite(target):
        return target
    return float(target) - max(float(reserve_min), 0.0) * bench_ppm(season)


def implied_points(mu_margin: float, mu_total: float, home_tt: float | None = None,
                   away_tt: float | None = None) -> tuple[float, float]:
    """Target team points: book team totals when both are known, else the spread/total split."""
    if home_tt is not None and away_tt is not None and np.isfinite(home_tt) \
            and np.isfinite(away_tt):
        return float(home_tt), float(away_tt)
    return (float(mu_total + mu_margin) / 2.0, float(mu_total - mu_margin) / 2.0)


def reconcile_team(raw_pts: np.ndarray, raw_oth: np.ndarray, g: np.ndarray, alpha: float,
                   target: float, weight: float = 1.0, rw: dict | None = None) -> dict:
    """Scale a team's per-sim points (S, n) and other DK (S, n) to the environment and target.

    raw_*: rate x minutes (0 when the player sits); g: (S,) team-points draw / its mean.
    `weight` (ablation; 1 = the model) shrinks both the per-sim environment (g^weight) and the
    target towards the unscaled baseline. Returns pts, oth (scaled arrays), pace ((S,) factor
    applied to the non-points stats) and the diagnostics: target, baseline (E[sum raw pts]),
    gap = baseline - target, scale (c), post (E[sum pts] after), resid, ok, data_check.
    `rw` (weighted reconciliation, RECON_WEIGHTED): {"r_pts", "n_pts", "kappa", "theta"} per
    player / team -> per-player factors (`_reconcile_weighted`; pace is then (S, n))."""
    if rw is not None:
        return _reconcile_weighted(raw_pts, raw_oth, g, alpha, target, weight, rw)
    g = np.maximum(np.asarray(g, float), 0.05) ** weight
    baseline = float(raw_pts.sum(1).mean()) if raw_pts.size else 0.0
    tgt = baseline + weight * (float(target) - baseline) if np.isfinite(target) else baseline
    gp = raw_pts * g[:, None]
    e = float(gp.sum(1).mean()) if raw_pts.size else 0.0
    c_raw = tgt / e if e > 0 else 1.0
    c = float(np.clip(c_raw, *SCALE_BOUNDS))
    pts = gp * c
    pace = (g * c) ** alpha
    oth = raw_oth * pace[:, None]
    post = float(pts.sum(1).mean()) if raw_pts.size else 0.0
    resid = post - tgt
    return {"pts": pts, "oth": oth, "pace": pace, "target": tgt, "baseline": baseline,
            "gap": baseline - float(target), "scale": c, "scale_uncapped": float(c_raw),
            "post": post, "resid": resid, "ok": bool(abs(resid) <= TOL),
            "data_check": bool(abs(baseline - target) > GAP_CHECK)}


def _reconcile_weighted(raw_pts: np.ndarray, raw_oth: np.ndarray, g: np.ndarray, alpha: float,
                        target: float, weight: float, rw: dict) -> dict:
    """`reconcile_team` with per-player factors (module doc, Weighted): c_i on points and
    c_i^alpha on the other stats; one uniform factor in SCALE_BOUNDS takes any remainder the
    player bounds leave. Extra diagnostics: c (n,), delta (n,) = e_i (c_i - 1)."""
    g = np.maximum(np.asarray(g, float), 0.05) ** weight
    baseline = float(raw_pts.sum(1).mean()) if raw_pts.size else 0.0
    tgt = baseline + weight * (float(target) - baseline) if np.isfinite(target) else baseline
    gp = raw_pts * g[:, None]
    e_i = gp.mean(0) if raw_pts.size else np.zeros(raw_pts.shape[1])
    e = float(e_i.sum())
    c_raw = tgt / e if e > 0 else 1.0
    v = rate_variance(e_i, rw["r_pts"], rw["n_pts"], rw["kappa"])
    c_i = weighted_factors(e_i, v, tgt, float(rw["theta"])) if e > 0 else np.ones(len(e_i))
    left = float((e_i * c_i).sum())
    c0 = float(np.clip(tgt / left, *SCALE_BOUNDS)) if left > 0 else 1.0
    c_i = c_i * c0
    pts = gp * c_i[None, :]
    pace = (g[:, None] * c_i[None, :]) ** alpha
    oth = raw_oth * pace
    post = float(pts.sum(1).mean()) if raw_pts.size else 0.0
    resid = post - tgt
    gap = baseline - float(target)
    out = {"pts": pts, "oth": oth, "pace": pace, "target": tgt, "baseline": baseline,
           "gap": gap, "scale": float(np.clip(c_raw, *SCALE_BOUNDS)),
           "scale_uncapped": float(c_raw), "post": post, "resid": resid,
           "ok": bool(abs(resid) <= TOL), "data_check": bool(abs(gap) > GAP_CHECK),
           "weighted": True, "c": c_i.tolist(), "delta": (e_i * (c_i - 1.0)).tolist()}
    return out


def log_gap_error(info: dict, team: str, player_ids) -> dict | None:
    """Weighted reconciliation: a pre-reconciliation |gap| > GAP_CHECK is a projection error
    (stale report, mis-matched name, missing player, a wrong rate). Logs it with the three
    players that absorbed most of it and returns that record (also kept in the team info)."""
    if not info.get("data_check") or "delta" not in info:
        return None
    d = np.asarray(info["delta"], float)
    pid = np.asarray(player_ids)
    top = np.argsort(-np.abs(d), kind="stable")[:3]
    rec = {"team": team, "gap": round(float(info["gap"]), 2),
           "target": round(float(info["target"]), 1),
           "absorbed": [[int(pid[j]), round(float(d[j]), 2),
                         round(float(np.asarray(info["c"])[j]), 3)] for j in top]}
    log.warning("projection error: %s pre-reconciliation gap %+.1f (target %.1f); absorbed by %s",
                team, rec["gap"], rec["target"], rec["absorbed"])
    return rec


def acceptance(team_infos: dict | list) -> dict:
    """Summary over team-games: all within TOL, worst residual, teams with a data check."""
    rows = list(team_infos.values()) if isinstance(team_infos, dict) else list(team_infos)
    if not rows:
        return {"n": 0, "ok": True, "max_abs_resid": 0.0, "data_checks": []}
    res = [abs(r["resid"]) for r in rows]
    return {"n": len(rows), "ok": all(r["ok"] for r in rows), "max_abs_resid": max(res),
            "max_abs_gap": max(abs(r["gap"]) for r in rows),
            "data_checks": [r.get("team") for r in rows if r["data_check"]]}


def check_rows(team_infos: dict | list, label: str) -> list[dict]:
    """`data_checks` rows: one 'proj_recon_gap:<label>' row (warn when any team-game's
    pre-reconciliation gap > 6 or the residual exceeds 0.5)."""
    rows = list(team_infos.values()) if isinstance(team_infos, dict) else list(team_infos)
    if not rows:
        return []
    acc = acceptance(rows)
    bad = [{"team": r.get("team"), "gap": round(r["gap"], 2), "target": round(r["target"], 1),
            "scale": round(r["scale"], 3),
            **({"projection_error": r["gap_error"]["absorbed"]} if r.get("gap_error") else {})}
           for r in rows if r["data_check"] or not r["ok"]]
    return [{"check_name": f"proj_recon_gap:{label}",
             "status": "warn" if bad else "ok", "value": float(acc["max_abs_gap"]),
             "threshold": GAP_CHECK,
             "detail": json.dumps({"max_abs_resid": round(acc["max_abs_resid"], 4),
                                   "teams": bad})}]
