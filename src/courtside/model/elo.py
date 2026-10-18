"""FiveThirtyEight-style NBA Elo, strictly sequential.

For each game, in chronological order:
    dr     = elo_home + hca_eff - elo_away + adj          (hca_eff = 0 at neutral sites/bubble)
    E_home = 1 / (1 + 10 ** (-dr / 400))
    mult   = (|margin| + mov_add) ** mov_exp / (ac_base + ac_slope * dr_winner)
    shift  = K * mult * (home_win - E_home);  elo_home += shift;  elo_away -= shift
where dr_winner = dr from the winner's side. The denominator is the autocorrelation correction:
favourites win by more, so their big wins are discounted. `adj` is the optional rest/travel
adjustment. Between seasons each rating regresses: r <- mean + carry * (r - mean). Teams in the
first season of the data start at `init`; franchises appearing later (2004-05 Bobcats) start at
`expansion`.

The pre-game columns of game i are computed before game i's result is applied, so they depend
only on earlier games (a team plays at most once per day, so same-day order is irrelevant).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import metrics

LN10_400 = math.log(10) / 400.0


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hca: float = 100.0
    carry: float = 0.75            # share of a rating's deviation from `mean` kept over summer
    mean: float = 1505.0
    init: float = 1500.0
    expansion: float = 1300.0
    mov_exp: float = 0.8
    mov_add: float = 3.0
    ac_base: float = 7.5
    ac_slope: float = 0.006
    playoff_k_mult: float = 1.0
    b2b_elo: float = 0.0           # Elo penalty for playing the 2nd night of a back-to-back
    travel_elo_per_1000km: float = 0.0   # Elo penalty per 1000 km travelled since last game
    pts_per_elo: float = 28.0      # default Elo-points per game-point for the implied spread

    def to_dict(self) -> dict:
        return asdict(self)


# Tuned on 2000-01..2010-11 only (log-loss on 2001-02..2010-11, all season types) with
# `courtside model tune-elo` (elo.tune_all, 75 configs), then frozen. Tuning log-loss 0.60224
# (538's published k=20, hca=100, carry=0.75 scores 0.60310 on the same window).
TUNED = EloParams(k=20.0, hca=90.0, carry=0.65)
# Rest-adjusted variant: TUNED plus the best (b2b, travel) on the same window (0.60083).
# A travel penalty made tuning log-loss worse at every b2b level, so it stays 0.
TUNED_REST = replace(TUNED, b2b_elo=40.0, travel_elo_per_1000km=0.0)


def _adjustment(games: pd.DataFrame, p: EloParams) -> np.ndarray:
    """Rest/travel adjustment to dr (home perspective). Needs schedule columns if non-zero."""
    if p.b2b_elo == 0 and p.travel_elo_per_1000km == 0:
        return np.zeros(len(games))
    adj = (-p.b2b_elo * (games["home_b2b"].to_numpy() - games["away_b2b"].to_numpy())
           - p.travel_elo_per_1000km / 1000.0
           * (games["home_travel_km"].to_numpy() - games["away_travel_km"].to_numpy()))
    return adj.astype(float)


def run_elo(games: pd.DataFrame, params: EloParams = TUNED,
            return_ratings: bool = False):
    """Run Elo over `games` (must be chronological). Returns a DataFrame aligned with `games`:
    elo_home_pre, elo_away_pre, elo_dr (incl. HCA/adjustment), elo_prob (P home win),
    elo_logit, elo_margin (expected home margin, pts_per_elo scale), elo_spread (= -elo_margin,
    betting convention: negative means home favoured).
    """
    p = params
    d = games["game_date"].to_numpy()
    if len(d) > 1 and (np.diff(d.astype("datetime64[D]").astype(np.int64)) < 0).any():
        raise ValueError("games must be sorted by game_date")
    home = games["home"].to_numpy()
    away = games["away"].to_numpy()
    season = games["season"].to_numpy()
    neutral = games["is_neutral"].to_numpy(dtype=bool)
    playoff = (games["season_type"].to_numpy() != "Regular Season")
    margin = games["margin"].to_numpy(dtype=float)
    adj = _adjustment(games, p)
    n = len(games)
    eh = np.empty(n)
    ea = np.empty(n)
    dr_out = np.empty(n)
    ratings: dict[str, float] = {}
    first_season = season[0] if n else None
    cur_season = first_season
    for i in range(n):
        s = season[i]
        if s != cur_season:
            for t in ratings:
                ratings[t] = p.mean + p.carry * (ratings[t] - p.mean)
            cur_season = s
        h, a = home[i], away[i]
        if h not in ratings:
            ratings[h] = p.init if s == first_season else p.expansion
        if a not in ratings:
            ratings[a] = p.init if s == first_season else p.expansion
        rh, ra = ratings[h], ratings[a]
        dr = rh - ra + (0.0 if neutral[i] else p.hca) + adj[i]
        eh[i], ea[i], dr_out[i] = rh, ra, dr
        e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        m = margin[i]
        win = 1.0 if m > 0 else 0.0
        dr_w = dr if m > 0 else -dr
        mult = (abs(m) + p.mov_add) ** p.mov_exp / max(p.ac_base + p.ac_slope * dr_w, 1.0)
        k = p.k * (p.playoff_k_mult if playoff[i] else 1.0)
        shift = k * mult * (win - e_home)
        ratings[h] = rh + shift
        ratings[a] = ra - shift
    out = pd.DataFrame({
        "elo_home_pre": eh, "elo_away_pre": ea, "elo_dr": dr_out,
        "elo_prob": 1.0 / (1.0 + 10.0 ** (-dr_out / 400.0)),
        "elo_logit": dr_out * LN10_400,
        "elo_margin": dr_out / p.pts_per_elo,
    }, index=games.index)
    out["elo_spread"] = -out["elo_margin"]
    return (out, ratings) if return_ratings else out


def fit_pts_per_elo(dr: np.ndarray, margin: np.ndarray) -> float:
    """Least-squares slope (through the origin) of margin on Elo difference, as Elo per point."""
    slope = float(np.dot(dr, margin) / np.dot(dr, dr))
    return 1.0 / slope


# --- tuning --------------------------------------------------------------------------------

TUNE_SEASONS = ("2000-01", "2010-11")   # inclusive; data window used for tuning
TUNE_EVAL_FROM = "2001-02"              # skip the cold-start season when scoring


def default_grid() -> list[EloParams]:
    """36 core configs (k x hca x carry)."""
    return [EloParams(k=k, hca=h, carry=c)
            for k in (15.0, 20.0, 25.0, 30.0)
            for h in (60.0, 80.0, 100.0)
            for c in (0.6, 0.75, 0.9)]


def refine_grid(best: EloParams) -> list[EloParams]:
    """27 configs around a stage-1 winner (the stage-1 optimum sat on a grid edge)."""
    return [replace(best, k=best.k + dk, hca=best.hca + dh, carry=round(best.carry + dc, 3))
            for dk in (-2.0, 0.0, 2.0) for dh in (-10.0, 0.0, 10.0) for dc in (-0.05, 0.0, 0.05)]


def rest_grid(base: EloParams) -> list[EloParams]:
    return [replace(base, b2b_elo=b, travel_elo_per_1000km=t)
            for b in (0.0, 20.0, 40.0, 60.0) for t in (0.0, 10.0, 20.0)]


def tune(games: pd.DataFrame, grid: list[EloParams]) -> list[tuple[float, EloParams]]:
    """Score each config by log-loss on TUNE_EVAL_FROM..TUNE_SEASONS[1]. Uses only games in
    the tuning window. Returns [(log_loss, params)] best first."""
    lo, hi = TUNE_SEASONS
    w = games[(games["season"] >= lo) & (games["season"] <= hi)]
    ev = (w["season"] >= TUNE_EVAL_FROM).to_numpy()
    y = w["home_win"].to_numpy()[ev]
    res = []
    for prm in grid:
        e = run_elo(w, prm)
        res.append((metrics.log_loss(y, e["elo_prob"].to_numpy()[ev]), prm))
    res.sort(key=lambda r: r[0])
    return res


def tune_all(games: pd.DataFrame) -> dict:
    """Stage 1 (36 core configs), stage 2 (27 around the winner), stage 3 (12 rest/travel
    configs on top). Needs schedule columns. ~75 Elo runs over 11 seasons: a few seconds."""
    s1 = tune(games, default_grid())
    s2 = tune(games, refine_grid(s1[0][1]))
    best = min(s1[:1] + s2, key=lambda r: r[0])
    s3 = tune(games, rest_grid(best[1]))
    return {"stage1": s1, "stage2": s2, "best": best, "rest": s3}
