"""Minutes-ratio arm (`COURTSIDE_MINRATIO=1`, `pipeline.MINRATIO`): key the per-minute minutes
table on PLANNED minutes (docs/research/return-minutes.md, docs/PLAYERS.md Phase 2 results).

The rate state's table f_stat(min / mpg) (`state.fit_params`, `minutes_adj`) is fit on REALISED
minutes ratios: a player pulled early (foul trouble, an injury exit, a blowout) produces less per
minute, and f carries that over. The sims apply it at the drawn minutes over the player's usual
minutes, so a PLANNED change in minutes - a ramp after an injury, a promotion when a starter sits -
gets the same per-minute penalty or boost as an unplanned one, which it does not have in reality
(per-minute DK under-projected +2..9 % when projected well below his recent minutes, over-projected
-2..-6 % above 1.2x).

Here the table is keyed on the plan (`state.minutes_multiplier_mr`):

    E[stat | m] = rate x m x f_u(m / P) [x g_p(P / mpg)]

* P = the minutes engine's pre-game E[min | plays] (regulation, before the margin): in the sims
  the `mean` of the sim's own availability scenario (`copula._PlanGrids`), in training the same
  engine walk-forward under the avail oracle (`planned_season`: season X's minutes params, fit on
  the 5 seasons < X, projecting X's team-games; exactly what that season's backtest sims use);
* f_u carries the unplanned deviation (early exits, blowouts, overtime, coach's whims);
* g_p (only with PLAN_FACTOR = True, a research option) would carry a per-minute change of the
  plan itself. The switch runs the plain re-key, **g_p = 1** (PLAN_FACTOR = False): the
  fitted g_p (+2..7 % per minute at P = 1.3..1.5 x mpg in training) made the promoted rows' over-
  projection worse walk-forward and the star tier worse in 2023-24 (+1.3 SE CRPS), while the plain
  re-key passed the bar (docs/PLAYERS.md, minutes-ratio results). The table keeps `mult_p`
  (all ones) and records `plan_factor`, so both variants read the same way.

Both are binned ratio-of-sums tables fit on the regular-season rows of the FIT_SEASONS seasons
< S with a plan (`fit_table`; backfitted when g_p is fitted), interpolated linearly between bin
centres (as `minutes_adj`); g_p is normalised to 1 at P = mpg. Rows without a plan (call-ups
outside the engine's candidates) fall back to P = mpg.

Design C (research option, `PL_EDGES = PL_TIERS_C`; the switch runs design B, PL_EDGES = ()):
f_u per tier of P (table version 2, tiers 12 / 18 / 24 / 30 minutes; each (tier, m/P bin) cell
shrunk to the pooled f_u with SHRINK_ROWS pseudo-rows), read linearly in P between the tiers'
mean plans. The pooled f_u is dominated by starters' rows, whose short nights are early exits;
a bench plan's short nights are spot and garbage-time minutes, so its f_u is flatter below
m / P ~ 0.6 and lower around 0.8..1.0, and star plans' are flatter too. Out of sample at realised
minutes C cuts the calibration error over (P tier x m/P) cells by a quarter to a third
(`research/minutes_ratio_eval.py proxy`) and the harness CRPS gain grows, but its steeper
mid-P curves widen the rotation / bench intervals, which the sims' minutes already make too wide:
the rotation tier is worse on the live slice (+1.6 / +2.3 SE CRPS) and the bench tier stays worse
in 2019-20, so it was not adopted (docs/PLAYERS.md, minutes-ratio results).

The arm's bundle (`bundle_<S><arm>_mr.json`, `fit_bundle`) is the arm's bundle with, all fit on
seasons < S:
1. state: + `minutes_adj_mr`; the usage bump refit on top of it (it is a ratio of sums given the
   minutes table); EB half-lives / kappa / phi unchanged;
2. copula (v0): refit (alpha, CV-by-level and correlations are residuals given the minutes table);
3. context layer: refit with its Poisson offset E_s re-keyed (`adjust_offsets`). Its F2 features
   keep the f(m/mpg) expectations of `features.common.base_rows` - they are inputs, built the same
   way in training and live, and live has no plan for past games of other teams.
Copula v1 params (only with COURTSIDE_COPULA=v1) are kept from the arm's bundle (fit on the old
table; the combination is wired but not evaluated).

Cache: planned minutes per season under `<data>/proj/minutes_ratio/planned_<X>.parquet`, the
minutes params of seasons without a bundle under `.../minutes_params_<X>.json`.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from . import minutes as M
from . import state as ST

TABLE_VERSION = 2                  # 2: f_u by tier of the plan P (PL_EDGES); 1: one pooled f_u
FIT_SEASONS = 5                    # the minutes engine's window
U_EDGES = ST.RATIO_EDGES           # m / P
P_EDGES = (0.6, 0.75, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5)   # P / mpg
PL_TIERS_C = (12.0, 18.0, 24.0, 30.0)  # design C: f_u by tier of P (minutes); research option
PL_EDGES: tuple = ()               # the switch's design: () = one pooled f_u (design B, version 1)
SHRINK_ROWS = 300                  # a tier's bin is shrunk to the pooled f_u with this many rows
ITERS = 4
MIN_BIN = 300
PLAN_FACTOR = False                # True: also fit g_p(P / mpg) (research variant; see the docstring)
CLIP = (0.5, 2.0)
HISTORY_SEASONS = 7                # as pipeline.Context
WINDOW = 5                         # pipeline.MINUTES_WINDOW


def cache_dir() -> Path:
    return config.data_dir() / "proj" / "minutes_ratio"


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def _minus(season: str, k: int) -> str:
    return season_label(int(season[:4]) - k)


# ============================================================================== planned minutes

def minutes_params_for(conn: sqlite3.Connection, season: str, h: M.History | None = None,
                       panel_tr: pd.DataFrame | None = None, log=None) -> M.MinutesParams:
    """Season X's v1 minutes params (fit on the 5 seasons < X): the pipeline's cached bundle /
    minutes params when present, else a cached or fresh fit (the pipeline's own recipe)."""
    say = log or (lambda m: None)
    pdir = config.data_dir() / "proj"
    for f in (pdir / f"bundle_{season}_v1.json", pdir / f"minutes_params_{season}_v1.json",
              cache_dir() / f"minutes_params_{season}.json"):
        if f.exists():
            d = json.loads(f.read_text())
            d = d.get("minutes", d)
            mp = M.MinutesParams(**d)
            if mp.version == "v1":
                return mp
    if h is None or panel_tr is None:
        raise ValueError(f"no minutes params for {season} and no history to fit them")
    t0 = time.time()
    mp = M.fit(h, season, window=WINDOW, log=say, panel=panel_tr,
               shares=M.PairShares(M.pair_events(panel_tr)), version="v1")
    cache_dir().mkdir(parents=True, exist_ok=True)
    (cache_dir() / f"minutes_params_{season}.json").write_text(mp.to_json())
    say(f"minutes params {season} fitted in {time.time() - t0:.0f}s")
    return mp


def planned_season(conn: sqlite3.Connection, season: str, log=None) -> pd.DataFrame:
    """Walk-forward planned minutes of every played, non-absent regular-season player-game of
    `season`: the season's minutes engine (params fit on seasons < X) on the avail-oracle panel,
    pair shares from the training window plus the season (lookups strictly before each date),
    as `pipeline.Context` / `copula.ScenarioMinutes` build them. Columns: game_id, team,
    player_id, season, planned (E[min | plays], regulation, before the margin), p_play."""
    say = log or (lambda m: None)
    t0 = time.time()
    h = M.load_history(conn, since=_minus(season, HISTORY_SEASONS))
    train = sorted(x for x in h.tg["season"].unique() if x < season)[-WINDOW:]
    panel_tr = M.oracle_panel(h, M.eval_targets(h, train))
    mp = minutes_params_for(conn, season, h, panel_tr, log=say)
    panel = M.oracle_panel(h, M.eval_targets(h, [season]))
    shares = M.PairShares(pd.concat([M.pair_events(panel_tr), M.pair_events(panel)],
                                    ignore_index=True))
    r = M.run_projection(panel, mp, shares=shares)
    ok = r["played"].to_numpy(bool) & ~r["absent"].to_numpy(bool)
    out = pd.DataFrame({"game_id": r.loc[ok, "game_id"].to_numpy(),
                        "team": r.loc[ok, "team"].to_numpy(),
                        "player_id": r.loc[ok, "player_id"].to_numpy(np.int64),
                        "season": season,
                        "planned": r.loc[ok, "mean0"].to_numpy(float),
                        "p_play": r.loc[ok, "p_play"].to_numpy(float)})
    say(f"planned minutes {season}: {len(out)} rows ({time.time() - t0:.0f}s)")
    return out


def planned_rows(conn: sqlite3.Connection, seasons, log=None, compute: bool = True
                 ) -> pd.DataFrame:
    """`planned_season` for each season, cached as parquet (computed when missing)."""
    parts = []
    for s in sorted(set(seasons)):
        f = cache_dir() / f"planned_{s}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
            continue
        if not compute:
            raise FileNotFoundError(f)
        d = planned_season(conn, s, log=log)
        f.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(f)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "team", "player_id", "season", "planned", "p_play"])


def attach(wf: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    """wf + a `planned` column (NaN where the engine had no plan), row order kept."""
    pl = planned[["game_id", "team", "player_id", "planned"]].drop_duplicates(
        ["game_id", "team", "player_id"])
    out = wf.merge(pl, on=["game_id", "team", "player_id"], how="left")
    assert len(out) == len(wf)
    return out


# ============================================================================== the table

def _centers(x: np.ndarray, edges) -> tuple[np.ndarray, np.ndarray]:
    b = np.searchsorted(np.asarray(edges), x, side="right")
    c = np.array([x[b == k].mean() if np.any(b == k) else
                  float(edges[min(k, len(edges) - 1)]) for k in range(len(edges) + 1)])
    return b, c


def _ratio_bins(y: np.ndarray, pred: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    out = np.full(k, np.nan)
    for j in range(k):
        m = b == j
        p = float(pred[m].sum())
        if m.sum() >= MIN_BIN and p > 0:
            out[j] = float(np.clip(y[m].sum() / p, *CLIP))
    if np.all(np.isnan(out)):
        return np.ones(k)
    idx = np.flatnonzero(np.isfinite(out))
    return np.interp(np.arange(k), idx, out[idx])        # empty bins: the nearest neighbours


def _tier_bins(y: np.ndarray, pred: np.ndarray, bu: np.ndarray, bt: np.ndarray, fu: np.ndarray,
               n_tiers: int) -> np.ndarray:
    """(n_tiers, len(fu)) ratio of sums per (tier of P, bin of m / P), each cell shrunk to the
    pooled bin `fu` with SHRINK_ROWS pseudo-rows of the cell's own mean prediction:
    f = (sum y + lam f_pool) / (sum pred + lam), lam = SHRINK_ROWS x sum pred / n. Empty cells
    are the pooled value."""
    out = np.tile(np.asarray(fu, float), (n_tiers, 1))
    for t in range(n_tiers):
        for j in range(len(fu)):
            c = (bt == t) & (bu == j)
            n = int(c.sum())
            p = float(pred[c].sum())
            if n == 0 or p <= 0:
                continue
            lam = SHRINK_ROWS * p / n
            out[t, j] = float(np.clip((y[c].sum() + lam * fu[j]) / (p + lam), *CLIP))
    return out


def fit_table(rows: pd.DataFrame, stats=ST.STATS, plan_factor: bool | None = None,
              pl_edges=None) -> dict:
    """The planned-minutes table from training rows (min > 0, a plan, the pre-game state and the
    actual stats): ratio-of-sums per bin of u = m / P; with `plan_factor` (default PLAN_FACTOR)
    also per bin of v = P / mpg, backfitted. Without it `mult_p` is all ones.

    Tiers (`pl_edges`, default PL_EDGES; () = one pooled f_u, the version-1 table): f_u is also
    fit per tier of P (minutes) - `mult_up[stat][tier]` on the tier's own bin centres
    `centers_up[tier]`, each cell shrunk to the pooled f_u (`_tier_bins`), read linearly in P
    between the tiers' mean plans `centers_pl` (`state.minutes_multiplier_mr`). A bench plan's
    short nights are mostly spot / garbage-time minutes, a starter's are early exits (fouls,
    injuries): the pooled f_u, dominated by starters' rows, is too steep for a bench plan."""
    plan_factor = PLAN_FACTOR if plan_factor is None else bool(plan_factor)
    pl_edges = tuple(PL_EDGES if pl_edges is None else pl_edges)
    rows = rows[np.isfinite(rows["planned"].to_numpy(float))
                & (rows["planned"].to_numpy(float) > 0) & (rows["min"].to_numpy(float) > 0)]
    m = rows["min"].to_numpy(float)
    pl = np.maximum(rows["planned"].to_numpy(float), 1.0)
    mpg = np.maximum(rows["mpg"].to_numpy(float), 1.0)
    u, v = m / pl, pl / mpg
    bu, cu = _centers(u, U_EDGES)
    bv, cv = _centers(v, P_EDGES)
    out = {"version": TABLE_VERSION if pl_edges else 1, "edges_u": list(U_EDGES),
           "edges_p": list(P_EDGES),
           "centers_u": [float(c) for c in cu], "centers_p": [float(c) for c in cv],
           "mult_u": {}, "mult_p": {}, "n": int(len(rows)), "plan_factor": plan_factor,
           "n_u": [int((bu == k).sum()) for k in range(len(cu))],
           "n_p": [int((bv == k).sum()) for k in range(len(cv))]}
    if pl_edges:
        bt, cpl = _centers(pl, pl_edges)
        nt = len(cpl)
        cup = [[float(u[(bt == t) & (bu == j)].mean()) if np.any((bt == t) & (bu == j))
                else float(cu[j]) for j in range(len(cu))] for t in range(nt)]
        out.update({"edges_pl": list(pl_edges), "centers_pl": [float(c) for c in cpl],
                    "centers_up": cup, "mult_up": {}, "shrink_rows": SHRINK_ROWS,
                    "n_up": [[int(((bt == t) & (bu == j)).sum()) for j in range(len(cu))]
                             for t in range(nt)]})
    for s in stats:
        y = rows[s].to_numpy(float)
        base = rows[f"r_{s}"].to_numpy(float) * m
        fu, gp = np.ones(len(cu)), np.ones(len(cv))
        for _ in range(ITERS if plan_factor else 1):
            fu = _ratio_bins(y, base * np.interp(v, cv, gp), bu, len(cu))
            if plan_factor:
                gp = _ratio_bins(y, base * np.interp(u, cu, fu), bv, len(cv))
        g1 = float(np.interp(1.0, cv, gp))
        out["mult_u"][s] = [float(x) for x in fu * g1]
        out["mult_p"][s] = [float(x) for x in gp / g1]
        if pl_edges:
            ft = _tier_bins(y, base * np.interp(v, cv, gp), bu, bt, fu, nt)
            out["mult_up"][s] = [[float(x) for x in r * g1] for r in ft]
    return out


def table_matches(t: dict | None) -> bool:
    """A cached table was fit with the current design: PLAN_FACTOR (tables without the field
    count as g_p-fitted unless every `mult_p` entry is 1) and the P tiers (PL_EDGES; a table
    without `edges_pl` is the pooled version-1 design)."""
    if not t:
        return False
    pf = t.get("plan_factor")
    if pf is None:
        pf = any(abs(x - 1.0) > 1e-12 for v in t.get("mult_p", {}).values() for x in v)
    tiers = tuple(float(x) for x in t.get("edges_pl", ()))
    return (bool(pf) == PLAN_FACTOR and tiers == tuple(float(x) for x in PL_EDGES)
            and float(t.get("shrink_rows", SHRINK_ROWS)) == float(SHRINK_ROWS))


def training_frame(conn: sqlite3.Connection, season: str, sp: dict, planned: pd.DataFrame,
                   seasons=None, games: pd.DataFrame | None = None, pos=None) -> pd.DataFrame:
    """Regular-season walk-forward rows (state params `sp`) of `seasons` (default: the
    FIT_SEASONS seasons < S) with the plan attached."""
    seasons = seasons or [_minus(season, k) for k in range(FIT_SEASONS, 0, -1)]
    if games is None:
        games = ST.load_games(conn, since_season=_minus(seasons[0], sp.get("history_seasons", 6)),
                              until_season=seasons[-1])
    pos = pos if pos is not None else ST.load_positions(conn)
    wf = pd.concat([ST.walk_forward(conn, s, sp, games=games, pos=pos) for s in seasons],
                   ignore_index=True)
    wf = wf[wf["season_type"] == "Regular Season"].reset_index(drop=True)
    return attach(wf, planned)


def fit_state(conn: sqlite3.Connection, season: str, sp: dict, planned: pd.DataFrame,
              log=None) -> dict:
    """The arm's rate-state params for S: `sp` + `minutes_adj_mr` fit on the FIT_SEASONS seasons
    < S, and the usage bump refit on the last 3 of them on top of the new table."""
    say = log or (lambda m: None)
    t0 = time.time()
    wf = training_frame(conn, season, sp, planned)
    have = np.isfinite(wf["planned"].to_numpy(float))
    sp2 = copy.deepcopy(sp)
    sp2["minutes_adj_mr"] = fit_table(wf[have])
    sp2["minutes_adj_mr"]["fit_seasons"] = sorted(wf["season"].unique().tolist())
    sp2["minutes_adj_mr"]["plan_share"] = float(have.mean())
    last3 = sorted(wf["season"].unique())[-3:]
    sp2["usage_bump"] = ST.fit_usage_bump(conn, sp2, last3, wf=wf[wf["season"].isin(last3)])
    t = sp2["minutes_adj_mr"]
    say(f"minutes-ratio table {season}: {t['n']} rows ({have.mean():.1%} with a plan), "
        f"pts f_u {np.round(t['mult_u']['pts'], 3).tolist()} g_p "
        f"{np.round(t['mult_p']['pts'], 3).tolist()} ({time.time() - t0:.0f}s)")
    for k, c in enumerate(t.get("centers_pl", [])):
        say(f"  tier P ~ {c:.1f}: pts f_u {np.round(t['mult_up']['pts'][k], 3).tolist()}")
    return sp2


# ============================================================================== context offsets

def adjust_offsets(conn: sqlite3.Connection, frame: pd.DataFrame, sp: dict,
                   planned: pd.DataFrame) -> pd.DataFrame:
    """The context layer's Poisson offsets re-keyed: E0_s x f_mr / f_old per row (the bump and
    the rates unchanged), then the market reconciliation c recomputed on the new team sums
    (`common.env_factor`). Features are untouched. Rows without a plan use P = mpg."""
    from .features.common import ALL_STATS, ENV_ALPHA, env_factor
    f = attach(frame, planned)
    mins = f["min"].to_numpy(float)
    mpg = f["mpg"].to_numpy(float)
    pl = f["planned"].to_numpy(float)
    new = {}
    for s in ALL_STATS:
        old = ST.minutes_multiplier(sp, s, mins, mpg)
        mr = ST.minutes_multiplier(sp, s, mins, mpg, planned=pl)
        new[f"E0_{s}"] = f[f"E0_{s}"].to_numpy(float) * np.where(old > 0, mr / old, 1.0)
    out = frame.copy()
    for k, v in new.items():
        out[k] = v
    c = env_factor(conn, out, sorted(out["season"].unique()))
    out["c_env"] = c
    for s in ALL_STATS:
        out[f"E_{s}"] = out[f"E0_{s}"].to_numpy() * (c if s == "pts" else c ** ENV_ALPHA)
    return out


# ============================================================================== bundle

def plan_seasons(season: str, n_context: int) -> list[str]:
    """Seasons whose plans the arm's bundle for S needs: the context layer's training seasons
    (n_context < S) cover the table's (FIT_SEASONS), the bump's and the copula's."""
    return [_minus(season, k) for k in range(max(n_context, FIT_SEASONS), 0, -1)]


def fit_bundle(conn: sqlite3.Connection, season: str, base, *, fit_context: Callable | None,
               log=None):
    """The arm's bundle from the arm's own bundle `base` (pipeline.Bundle, everything fit on
    seasons < S): new state (table + bump), copula v0 refit, context layer refit with re-keyed
    offsets (when `base` has one; `fit_context(sp, frame)` -> ContextModel). Minutes params and
    copula v1 params are kept."""
    from . import context as CX
    from . import copula as C
    from . import features as FX
    say = log or (lambda m: None)
    t0 = time.time()
    n_cx = int(base.context.meta.get("n_train_seasons", CX.LIVE_N_TRAIN)) if base.context else 0
    planned = planned_rows(conn, plan_seasons(season, n_cx), log=say)
    sp = fit_state(conn, season, base.state, planned, log=say)
    cp = C.fit_copula(conn, season, sp, log=say, planned=planned)
    b = copy.copy(base)
    b.state, b.copula = sp, cp
    if base.context is not None and fit_context is not None:
        fr = FX.build_frame(conn, season, sp, n_train=n_cx, log=say)
        fr = adjust_offsets(conn, fr, sp, planned)
        b.context = fit_context(sp, fr)
    say(f"minutes-ratio bundle {season} fitted in {time.time() - t0:.0f}s")
    return b
