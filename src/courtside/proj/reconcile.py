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

Public API
  implied_points(mu_margin, mu_total, home_tt=None, away_tt=None) -> (home, away)
  reconcile_team(raw_pts, raw_oth, g, alpha, target) -> dict(pts, oth, scale, baseline, ...)
  acceptance(team_infos) -> dict;  check_rows(team_infos, run_label) -> data_checks rows
  reserve_target(target, reserve_min, season) -> float;  bench_ppm(season) -> float
"""

from __future__ import annotations

import json
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
                   target: float, weight: float = 1.0) -> dict:
    """Scale a team's per-sim points (S, n) and other DK (S, n) to the environment and target.

    raw_*: rate x minutes (0 when the player sits); g: (S,) team-points draw / its mean.
    `weight` (ablation; 1 = the model) shrinks both the per-sim environment (g^weight) and the
    target towards the unscaled baseline. Returns pts, oth (scaled arrays), pace ((S,) factor
    applied to the non-points stats) and the diagnostics: target, baseline (E[sum raw pts]),
    gap = baseline - target, scale (c), post (E[sum pts] after), resid, ok, data_check."""
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
            "scale": round(r["scale"], 3)} for r in rows if r["data_check"] or not r["ok"]]
    return [{"check_name": f"proj_recon_gap:{label}",
             "status": "warn" if bad else "ok", "value": float(acc["max_abs_gap"]),
             "threshold": GAP_CHECK,
             "detail": json.dumps({"max_abs_resid": round(acc["max_abs_resid"], 4),
                                   "teams": bad})}]
