"""Slate projection pipeline (docs/PLAYERS.md §4, §5.3, §6, §7; T1.4).

Inputs per game, all as known at the run time (`at`) or strictly before the game date:

* **Environment** (`environment`): live = median over books of each book's latest
  `line_snapshots` row at or before `at` (spread, total, team totals; the 'open' book is
  skipped), else the latest `gbm_market` row of `predictions` at or before `at`, else the
  closing line, else a league default (data check). Backtests use the closing line.
* **Availability:** live = `availability.p_play_slate` at `at` (DK starter flags = P 0.99).
  Report statuses (out ... available, removed, day-to-day) give P(available) for the scenario
  draw; players the report does not list get 1.0 because their DNP-CD probability is already
  the minutes engine's P(plays | available). Backtest protocol "avail" (§7.1 L4) = the oracle:
  regulars who did not play are out, everybody else available; protocol "reports" = the
  official reports at the first tip - 45 min (2025-12-22+).
* **Minutes** (`minutes.pregame_panel`, scenario re-runs in `copula.ScenarioMinutes`), **rates**
  (`state.rate_frame`: EB per-minute rates before the day), **usage bump** (`state.bump_for`)
  for the team's top-2 regular stars (by DK/min x mpg; appeared in >= 7 of the last 10 team
  games or in one of the last 3).
* **Parameters** (`Bundle`): minutes params, rate-state params and copula params, all fit on
  seasons < S (S = the season of the day), cached as `<data_dir>/proj/bundle_<S>.json`.
* **Context layer** (Phase 2F P + F2, `context.py`), behind `CONTEXT_ON` (default OFF; env
  ``COURTSIDE_CONTEXT=1``): the bundle carries a LightGBM per-minute rate multiplier per DK stat
  (fit on seasons < S, clipped to [0.8, 1.25]) applied to the EB rates right after the stars are
  picked. ON runs use their own bundle file (`bundle_<S>_v1_cx.json`) and MODEL_VERSION
  (`pathA-v0.3`); with the switch OFF nothing of it runs and projections are bit-identical.
  The extended layer P + F2 + F3 (tracking) + F6 (schedule) is a second opt-in on top,
  `CONTEXT_EXT` (default OFF; env ``COURTSIDE_CONTEXT_EXT=1`` together with
  ``COURTSIDE_CONTEXT=1``): `bundle_<S>_v1_cx2.json`, `pathA-v0.4`, its per-day inputs from
  `Context.context_extras`. With CONTEXT_EXT off, ON is exactly `pathA-v0.3`.
* **Copula v1** (T2.10, `copula.simulate_game_v1`: per-stat component sims, exact DD/TD,
  winner/loser shift, T2.7 matrices), behind `COPULA` (default "v0"; env ``COURTSIDE_COPULA=v1``).
  It composes with the context switches as a suffix: bundle file `<arm's bundle>_c1.json` (the
  arm's v0 bundle + `copula_v1` params fit on seasons < S), MODEL_VERSION `<arm's version>+c1`
  (`pathA-v0.2+c1`, `pathA-v0.3+c1`, `pathA-v0.4+c1`). Seeds stay on SEED_VERSION, so a v1 run
  shares the environment / availability / minutes draws of its v0 arm. With COPULA "v0" none
  of it runs and every arm is bit-identical to the code before it.
  **Copula v1 retune** (v1r, ``COURTSIDE_COPULA=v1r``): the same simulator with params from
  `copula.fit_copula_v1r` (dispersion and within-player correlation fit on the context arm's
  residuals, a per-stat shape map of the count marginals, no winner/loser shift unless the
  minutes carry the blowout stage). Bundle `<arm's bundle>_c1r.json`, version `<arm>+c1r`
  (`pathA-v0.3+c1r+p2`, `pathA-v0.3+mf+c1r+p2`); seeds unchanged. "v0" / "v1": nothing of it
  runs.
* **P(play) v2** (p2, 2026-10-21, `pplay_v2_fix`), behind `PPLAY` (default "v1"; env
  ``COURTSIDE_PPLAY=v2``): on the live candidate paths only, (1) a listed, not-Out player's
  P(plays | available) is 1, so his total P(play) is the report cell (fit on *played*; the engine's
  own DNP-CD probability on top counted it twice), and (2) a forced DK-pool player outside the
  team's last 10 games and not on the report gets the fringe model's P(plays | available)
  (`availability.fringe_features` / `load_fringe`, DK Classic salary in the season's first
  games, DK starter flag 0.99) instead of being scored as an active roster player. Version
  suffix `+p2` on any arm (`pathA-v0.3+p2`, `pathA-v0.3+c1+p2`); bundle and seeds unchanged.
  With "v1" nothing of it runs (bit-identical); the avail oracle is identical either way.
* **Minutes ratio** (mr, 2026-10-21, `minutes_ratio.py`), behind `MINRATIO` (default OFF; env
  ``COURTSIDE_MINRATIO=1``): the per-minute minutes table is keyed on the minutes engine's
  planned E[min | plays] of each sim's availability scenario, f_u(m / P) (g_p = 1), instead
  of f(m / mpg) (which gave planned changes - ramps, promotions - the per-minute penalty of
  unplanned ones). Suffix on any arm: bundle `<arm's bundle>_mr.json` (the arm's bundle with the
  table, the usage bump, copula v0 and the context layer refit on seasons < S), version
  `<arm>+mr` (`pathA-v0.3+mr+p2`); seeds unchanged. OFF: nothing of it runs (bit-identical).
  With MINFIX the table's training plans are the +mf engine's (`minutes_ratio.plans_tag`).
* **Preseason minutes** (item #2, `minutes.V1_PRESEASON`), behind `PRESEASON` (default OFF; env
  ``COURTSIDE_PRESEASON=1``): the Context loads the preseason game logs (migration 15 tables; the
  current season's too, so the live path reads them) and the minutes LightGBM base adds the
  pre_* columns (preseason mpg, last-2 mpg, share and rank on the team, vs prior-season mpg and
  teammates, did-not-play flag), refit on seasons < S. Suffix on any arm: bundle
  `<arm's bundle>_pre.json` (the arm's bundle, minutes params replaced), version `<arm>+pre`
  (`pathA-v0.3+pre+p2`); seeds unchanged. OFF: no preseason table is read (bit-identical).
* **Minutes fix** (mf, `minutes.MF_*`), behind `MINFIX` (default OFF; env ``COURTSIDE_MINFIX=1``):
  the minutes params are refit on seasons < S with P(plays | available) reading the GBM's plan for
  the night (players planned below / above their recent minutes) and the noise fit net of the
  margin curve. Suffix on any arm: bundle `<arm's bundle>_mf.json` (after `_pre`, before `_mr` /
  `_c1`; the arm's bundle, minutes params replaced: `minutes_params_<S>_v1[_pre]_mf.json`),
  version `<arm>+mf` (`pathA-v0.3+mf+p2`); seeds unchanged. OFF: bit-identical.
* **Blowout minutes** (bo, `minutes.BO_*`), behind `BLOWOUT` (default OFF; env
  ``COURTSIDE_BLOWOUT=1``): the minutes params are refit on seasons < S with a blowout stage on
  top (P(plays | available) and the reserve move with the sim's margin, the margin curve is
  recalibrated by simulation so stars lose / the bench gains the minutes they really do at each
  margin, each tier's level and spread unchanged); the copula reads the per-minute multiplier
  net of the margin curve (copula v1's winner / loser shift is fit the same way). Suffix on any arm: bundle `<arm's bundle>_bo.json` (after `_mf`, before `_mr` / `_c1`; the
  arm's bundle, minutes params replaced: `minutes_params_<S>_v1[_pre][_mf]_bo.json`), version
  `<arm>+bo` (`pathA-v0.3+mf+bo+p2`); seeds unchanged. OFF: bit-identical.
* **Lock-10 Questionable refit** (pq, `availability.PPLAYQ`, default OFF; env
  ``COURTSIDE_PPLAYQ=1``; `set_pplayq`): logit offsets on listed non-starters once a lineup is
  out. Version `<model version>+pq` (`pathA-v0.3+p2+pq`); same bundle and seeds.
* **Reconciliation reserve** (rr, `reconcile.RECON_RESERVE`, default OFF; env
  ``COURTSIDE_RECON_RESERVE=1``; `set_recon_reserve`): the reconcile target is the implied total
  minus E[reserve minutes] x the league bench points per minute (fit on seasons < S). Version
  `<model version>+rr` (after `+pq`); same bundle and seeds.
* **Reserve extras** (rx, `minutes.RESERVE_EXTRAS`, default OFF; env
  ``COURTSIDE_RESERVE_EXTRAS=1``; `set_reserve_extras`): the minutes reserve matches the run-time
  candidates. With the DK pool in them (`candidate_panel` marks `forced`), only career debutants
  are left out: the reserve is their minutes by team games into the season (seasons < S,
  `minutes.pool_reserve`); else the forced extras' planned minutes come off the fitted reserve.
  Version `<model version>+rx` (before `+rr`); same bundle and seeds.
* **Role blend** (rb, `role_blend.ROLE_BLEND`, default OFF; env ``COURTSIDE_ROLE_BLEND=1``;
  `set_role_blend`): `run_games` moves each player's played sims by beta(r) x (last-10 DK -
  mu_played), r = planned / last-10 minutes (per-bucket beta fit on seasons < S,
  `role_blend_params.json`), so the stored projections and sims carry the blended mean; the
  contest builder's flat TRUTH_BETA shift is skipped. Version `<model version>+rb` (after
  `+rr`); same bundle and seeds.
* **Tier match** (tm, `copula.TIER_MATCH`, default OFF; env ``COURTSIDE_TIER_MATCH=1``;
  `set_tier_match`): the draw assigns the rho / CV tiers from the fit's inputs (r_dk x mpg, mpg;
  players who play in no sim count as bench) instead of mu_played / min_played. Version
  `<model version>+tm` (after `+rb`); same bundle and seeds. `docs/research/tier-match.md`.
* **Tail remap** (tr, `tail_remap.TAIL_REMAP`, default OFF; env ``COURTSIDE_TAIL_REMAP=1``;
  `set_tail_remap`): `run_games` re-labels the upper half of each player's played sims through a
  per tier x minutes-band quantile map fit on seasons < S (after the role blend), so stored sims,
  cards, X factor and builder see it. Version `<model version>+tr` (after `+tm`); same bundle and
  seeds. `docs/research/tail-remap.md`.
* **Minutes restriction** (rs, `min_restrict.MIN_RESTRICT`, default OFF; env
  ``COURTSIDE_MIN_RESTRICT=1``; `set_min_restrict`): players an ESPN injury comment of the last 72 h
  before the run puts on a minutes restriction (`sources/espn_restrict.py`) get their planned minutes
  capped (parsed cap, else 0.75 x normal minutes) via the panel's `rs_cap` column
  (`project_games` -> `build_games(rs_caps=)`). Version `<model version>+rs` (after `+tr`); same
  bundle and seeds. OFF: bit-identical. `docs/research/min-restrict.md`.
* **Minutes spread** (ms, `minutes.fit_spread`, IMPROVEMENTS.md P6), behind `MIN_SPREAD` (default
  OFF; env ``COURTSIDE_MIN_SPREAD=1``): the minutes noise gets a per-player lower / upper scale
  from LightGBM quantile models (seasons < S) instead of the tier's alone. Suffix on any arm:
  bundle `<arm's bundle>_ms.json` (after `_ep`, before `_mr` / `_c1`; minutes params
  `minutes_params_<S>_v1[_pre][_mf][_bo]_ms.json`), version `<arm>+ms`; seeds unchanged. OFF:
  bit-identical. `docs/research/p4-p6.md`.
* **Weighted reconciliation** (rw, `reconcile.RECON_WEIGHTED`, default OFF; env
  ``COURTSIDE_RECON_WEIGHTED=1``; `set_recon_weighted`): the team gap to the implied total is
  split by each player's rate uncertainty (EB evidence minutes, `n_pts` / `k_pts` added to the
  TeamSpec players) instead of one factor c; theta fit on seasons < S. Version
  `<model version>+rw` (after `+rs`); same bundle and seeds. `docs/research/p4-p6.md`.
* **Early-season prior** (ep, `early_prior.EARLY_PRIOR`, default OFF; env
  ``COURTSIDE_EARLY_PRIOR=1``; `set_early_prior`): the rate state adds the season's preseason
  games (weight and per-minute level per stat, fit on seasons < S) to the current-season
  evidence (`state.rate_frame(..., ep=)`). Suffix on any arm: bundle `<arm's bundle>_ep.json`
  (after `_bo`, before `_mr` / `_c1`; the arm's bundle + `state["early_prior"]`), version
  `<arm>+ep`; seeds unchanged. OFF: bit-identical. `docs/research/early-prior.md`.
* **Usage redistribution** (us, `usage.USAGE`, IMPROVEMENTS.md P8, default OFF; env
  ``COURTSIDE_USAGE=1``; `set_usage`): the sim's per-minute rates use usage v1's per-player
  with/without factors for every absent regular in each availability draw (`usage.team_matrix`
  on `TeamSpec.usage`, `copula._usage_means`) instead of the top-2-star tier bump; params fit
  on seasons < S per rate state (`usage.live_model`). Version `<model version>+us` (after
  `+rw`); same bundle and seeds. OFF: bit-identical. `docs/research/p8-e3.md`.

Seeds (common random numbers): `stable_seed(slate_or_day, kind, SEED_VERSION)` (= the
MODEL_VERSION without the context layer, so ON and OFF share draws); every game
draws from SeedSequence([seed, crc32(game_id)]), so a game's sims do not depend on which other
games are on the slate.

Public API
  project_slate(conn, slate_id, at, *, run_id, n_sims=25_000, seed=None, kind=None, ctx=None,
                bundle=None, store=True) -> ProjectionSet {DK playerId (str): Projection}
  project_day(conn, nba_day, at, *, n_sims=25_000, seed=None, ...) -> DayResult (NBA ids)
  project_historical(conn, game_date, *, protocol="avail" | "reports", ...) -> DayResult
  load_bundle(conn, season, ctx) / Context(conn, season)
  backtest(conn, season, n_dates=60, ...) -> dict  (quick walk-forward evaluation)
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sqlite3
import time
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from . import availability as AV
from . import context as CX
from . import copula as C
from . import early_prior as EP
from . import min_restrict as MRS
from . import minutes as M
from . import reconcile as RC
from . import role_blend as RB
from . import state as ST
from . import tail_remap as TR
from . import usage as US
from . import store as SO
from .dfs.optimizer import Projection

# Minutes engine used by the live/backtest pipeline (T2.2: v1 = LightGBM base with learned
# promotion; walk-forward MAE 4.49 / 4.40 vs v0.5 4.63 / 4.58, DK gate 8.049 vs 8.181).
MINUTES_VERSION = "v1"                 # "v0.5" | "v1"
_VS_BASE = "" if MINUTES_VERSION == "v0.5" else f"_{MINUTES_VERSION}"
MODEL_VERSION_BASE = C.MODEL_VERSION if MINUTES_VERSION == "v0.5" else "pathA-v0.2"

# Learned context layer (Phase 2F P + F2). OFF by default: the live model stays MODEL_VERSION_BASE
# until the owner ships it. COURTSIDE_CONTEXT=1 (read at import; `set_context` at run time) turns
# it on with its own bundle file suffix and model version, so ON and OFF runs never share a
# bundle, a backtest cache or a dfs_runs / player_projections model_version (they do share
# seeds, see SEED_VERSION).
CONTEXT_DEFAULT = True                 # shipped 2026-10-21 (owner OK after opener grading; env still overrides)
CONTEXT_ENV = "COURTSIDE_CONTEXT"      # "1"/"true"/"on" forces ON, "0"/"false"/"off" forces OFF
CONTEXT_SUFFIX = "_cx"
CONTEXT_MODEL_VERSION = "pathA-v0.3"
# Extended layer (Phase 2F second ablation: P + F2 + the groups in `context.LIVE_GROUPS_EXT`).
# Opt-in on top of CONTEXT_ON: COURTSIDE_CONTEXT=1 COURTSIDE_CONTEXT_EXT=1. Own bundle suffix and
# model version; with CONTEXT_EXT off, ON behaves exactly as P + F2 (pathA-v0.3).
CONTEXT_EXT_DEFAULT = False
CONTEXT_EXT_ENV = "COURTSIDE_CONTEXT_EXT"
CONTEXT_EXT_SUFFIX = "_cx2"
CONTEXT_EXT_MODEL_VERSION = "pathA-v0.4"
# Path A copula v1 (T2.10). "v0" by default (env COURTSIDE_COPULA=v1 / v0 overrides; `set_copula`
# at run time). v1 = a suffix on whatever context arm is active: bundle `<arm>_c1`, version
# `<arm version>+c1`; seeds unchanged (common random numbers with the arm's v0 runs).
COPULA_DEFAULT = "v1r"                 # +c1r shipped 2026-10-24 (owner OK; env still overrides)
COPULA_ENV = "COURTSIDE_COPULA"
COPULA_V1_SUFFIX = "_c1"
COPULA_V1_TAG = "+c1"
# Copula v1 retune (v1r): v1's simulator with params refit on the context arm's residuals
# (`copula.fit_copula_v1r`). COURTSIDE_COPULA=v1r; bundle `<arm>_c1r`, version `<arm>+c1r`;
# seeds unchanged. With "v0" / "v1" none of it runs.
COPULA_V1R_SUFFIX = "_c1r"
COPULA_V1R_TAG = "+c1r"
COPULA_V1_VERSIONS = ("v1", "v1r")     # copula versions that run `copula.simulate_game_v1`
# P(play) v2 (p2, 2026-10-21; `pplay_v2_fix`). "v1" by default (env COURTSIDE_PPLAY=v2 / v1
# overrides; `set_pplay` at run time). v2 = a version suffix on whatever arm is active
# (`pathA-v0.3+p2`, `pathA-v0.3+c1+p2`); the bundle is unchanged (the fringe model ships in the
# package, `availability.FRINGE_PATH`) and seeds are unchanged (common random numbers).
# It only acts on the live candidate paths (`project_games`, `backtest.live_like_day`): the
# avail oracle has no report and no forced pool players, so it is identical either way.
PPLAY_DEFAULT = "v2"                   # shipped 2026-10-21 (owner OK; env still overrides)
PPLAY_ENV = "COURTSIDE_PPLAY"
PPLAY_V2_TAG = "+p2"
# Minutes ratio (mr, `minutes_ratio.py`). OFF by default (env COURTSIDE_MINRATIO=1 / 0 overrides;
# `set_minratio` at run time). ON = a suffix on whatever arm is active: bundle `<arm>_mr` (before
# a copula `_c1`), version `<arm>+mr` (before `+c1` / `+p2`); seeds unchanged (common random
# numbers with the arm's OFF runs).
MINRATIO_DEFAULT = False               # shipping = True here (env still overrides)
MINRATIO_ENV = "COURTSIDE_MINRATIO"
MINRATIO_SUFFIX = "_mr"
MINRATIO_TAG = "+mr"
# Preseason minutes features (item #2, N2; `minutes.V1_PRESEASON`). OFF by default (env
# COURTSIDE_PRESEASON=1 / 0 overrides; `set_preseason` at run time). ON = a suffix on whatever arm
# is active: bundle `<arm>_pre` (before `_mr` / `_c1`) = the arm's bundle with the minutes params
# refit with the pre_* columns (`minutes_params_<S>_v1_pre.json`), version `<arm>+pre`; the
# Context loads the preseason logs (`minutes.attach_preseason`); seeds unchanged (common random
# numbers with the arm's OFF runs). OFF: no preseason table is read (bit-identical).
PRESEASON_DEFAULT = False              # shipping = True here (env still overrides)
PRESEASON_ENV = "COURTSIDE_PRESEASON"
PRESEASON_SUFFIX = "_pre"
PRESEASON_TAG = "+pre"
# Minutes fix (mf; `minutes.MF_*`). OFF by default (env COURTSIDE_MINFIX=1 / 0 overrides;
# `set_minfix` at run time). ON = a suffix on whatever arm is active: bundle `<arm>_mf` (after
# `_pre`, before `_mr` / `_c1`) = the arm's bundle with the minutes params refit with
# `fit_v1(minfix=True)` on seasons < S, version `<arm>+mf`; seeds unchanged (common random
# numbers with the arm's OFF runs). OFF: nothing of it runs (bit-identical).
MINFIX_DEFAULT = True                  # shipped 2026-10-23 (owner OK; env still overrides)
MINFIX_ENV = "COURTSIDE_MINFIX"
MINFIX_SUFFIX = "_mf"
MINFIX_TAG = "+mf"
# Blowout minutes (bo; `minutes.BO_*`). OFF by default (env COURTSIDE_BLOWOUT=1 / 0 overrides;
# `set_blowout` at run time). ON = a suffix on whatever arm is active: bundle `<arm>_bo` (after
# `_mf`, before `_mr` / `_c1`) = the arm's bundle with the minutes params refit with
# `fit_v1(blowout=True)` on seasons < S, version `<arm>+bo`; seeds unchanged (common random
# numbers with the arm's OFF runs). OFF: nothing of it runs (bit-identical).
BLOWOUT_DEFAULT = False                # shipping = True here (env still overrides)
BLOWOUT_ENV = "COURTSIDE_BLOWOUT"
BLOWOUT_SUFFIX = "_bo"
BLOWOUT_TAG = "+bo"
# Player-specific minutes spread (ms; `minutes.fit_spread`, IMPROVEMENTS.md P6). OFF by default
# (env COURTSIDE_MIN_SPREAD=1 / 0 overrides; `set_min_spread` at run time). ON = a suffix on
# whatever arm is active: bundle `<arm>_ms` (after `_ep`, before `_mr` / `_c1`) = the arm's
# bundle with the spread models added to its minutes params (`minutes_params_<S>_v1[_pre][_mf]
# [_bo]_ms.json`; mu / sigma recalibrated), version `<arm>+ms`; seeds unchanged (common random
# numbers with the arm's OFF runs). OFF: nothing of it runs (bit-identical).
MIN_SPREAD_DEFAULT = True              # shipped 2026-10-24 (owner OK; env still overrides)
MIN_SPREAD_ENV = "COURTSIDE_MIN_SPREAD"
MIN_SPREAD_SUFFIX = "_ms"
MIN_SPREAD_TAG = "+ms"


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _copula_flag(name: str, default: str = "v0") -> str:
    v = os.environ.get(name, "").strip().lower()
    if v == "v1r":
        return "v1r"
    if v in ("v1", "1", "true", "yes", "on"):
        return "v1"
    if v in ("v0", "0", "false", "no", "off"):
        return "v0"
    return default


def _pplay_flag(name: str, default: str = "v1") -> str:
    v = os.environ.get(name, "").strip().lower()
    if v in ("v2", "2", "p2", "true", "yes", "on", "1"):
        return "v2"
    if v in ("v1", "p1", "false", "no", "off", "0"):
        return "v1"
    return default


# Line freshness (lf, default OFF; env COURTSIDE_LINE_FRESH=1; `set_line_fresh`): `environment()`
# drops a book's line older than LINE_FRESH_MAX_H at `at`, or more than LINE_FRESH_REL_H older
# than the newest book's line (always keeps the newest book). Model version only (`+lf`).
LINE_FRESH_DEFAULT = False
LINE_FRESH_ENV = "COURTSIDE_LINE_FRESH"
LINE_FRESH_TAG = "+lf"
LINE_FRESH_MAX_H = 3.0
LINE_FRESH_REL_H = 1.0
LINE_FRESH = _env_flag(LINE_FRESH_ENV, LINE_FRESH_DEFAULT)
# Ghost candidates (gh, default OFF; env COURTSIDE_GHOST_DROP=1; `set_ghost_drop`;
# docs/research/reeval-A.md). The minutes candidates are the players of a team's last 10 games
# whose last team it is; at the start of a season (and after waivers) that keeps last April's
# two-way / 10-day / tank-minutes players who are on no NBA roster any more (live 2026-10-20 ..
# 24: 191 projected rows, P(play) 0.33, none played, ~12 expected minutes per team-game taken
# from the real players). ON, live path only (`project_games` with env_mode "live"):
# `drop_moved_players(strict=True)` also drops a candidate who is not forced in (DK pool /
# report) and is on no team of the season's `team_rosters` snapshot (no snapshot -> no-op).
# Model version only (`+gh`), same bundle and seeds. OFF: unchanged (bit-identical).
GHOST_DROP_DEFAULT = False
GHOST_DROP_ENV = "COURTSIDE_GHOST_DROP"
GHOST_DROP_TAG = "+gh"
GHOST_DROP = _env_flag(GHOST_DROP_ENV, GHOST_DROP_DEFAULT)


def _versions(on: bool, ext: bool = False, copula: str = "v0", pplay: str = "v1",
              mr: bool = False, pre: bool = False, mf: bool = False, bo: bool = False
              ) -> tuple[str, str]:
    """(bundle file suffix, MODEL_VERSION) for the context switch (+ the extended layer), the
    copula version (v1 appends `_c1` / `+c1` to whichever arm is active), the P(play)
    version (v2 appends `+p2` to the model version only: same bundle) and the minutes-ratio
    switch (`_mr` / `+mr`, before the copula's); the preseason (`_pre`), minutes-fix (`_mf`)
    and blowout (`_bo`) switches come before it."""
    vs, mv = _versions_bundle(on, ext, copula, mr, pre, mf, bo)
    mv = mv + PPLAY_V2_TAG if pplay == "v2" else mv
    # the lock-10 Questionable refit (availability.PPLAYQ, `+pq`): model version only, same
    # bundle and seeds; its offsets live in availability's own params file
    mv = mv + AV.PPLAYQ_TAG if AV.PPLAYQ else mv
    # official injury rows keyed by game date (availability.INJ_GAMEDAY, `+gd`): model version
    # only, same bundle and seeds; its refit cells live in availability's own params file
    mv = mv + AV.INJ_GAMEDAY_TAG if AV.INJ_GAMEDAY else mv
    # report names matched on the current roster too (availability.REPORT_MATCH, `+rm`) and
    # ghost candidates dropped (GHOST_DROP, `+gh`): model version only, same bundle and seeds
    mv = mv + AV.REPORT_MATCH_TAG if AV.REPORT_MATCH else mv
    mv = mv + GHOST_DROP_TAG if GHOST_DROP else mv
    # reserve extras (minutes.RESERVE_EXTRAS, `+rx`): model version only, same bundle and seeds
    mv = mv + M.RESERVE_EXTRAS_TAG if M.RESERVE_EXTRAS else mv
    # lineup known (minutes.LINEUP, `+lu`): model version only, same bundle and seeds; its
    # weights live in minutes' own params file
    mv = mv + M.LINEUP_TAG if M.LINEUP else mv
    # the reconciliation reserve (reconcile.RECON_RESERVE, `+rr`): model version only, same
    # bundle and seeds; its bench rates live in reconcile's own params file
    mv = mv + RC.RECON_RESERVE_TAG if RC.RECON_RESERVE else mv
    # the role-aware blend (role_blend.ROLE_BLEND, `+rb`): model version only, same bundle and
    # seeds; its betas live in role_blend's own params file
    mv = mv + RB.ROLE_BLEND_TAG if RB.ROLE_BLEND else mv
    # the tier match (copula.TIER_MATCH, `+tm`): model version only, same bundle and seeds
    mv = mv + C.TIER_MATCH_TAG if C.TIER_MATCH else mv
    # the tail remap (tail_remap.TAIL_REMAP, `+tr`): model version only, same bundle and seeds
    mv = mv + TR.TAIL_REMAP_TAG if TR.TAIL_REMAP else mv
    # the minutes restriction (min_restrict.MIN_RESTRICT, `+rs`): model version only
    mv = mv + MRS.MIN_RESTRICT_TAG if MRS.MIN_RESTRICT else mv
    # the weighted reconciliation (reconcile.RECON_WEIGHTED, `+rw`): model version only, same
    # bundle and seeds; its theta lives in reconcile's own params file
    mv = mv + RC.RECON_WEIGHTED_TAG if RC.RECON_WEIGHTED else mv
    # usage v1 redistribution (usage.USAGE, `+us`, P8): model version only, same bundle and
    # seeds; its params live in usage's own per-season files
    mv = mv + US.USAGE_TAG if US.USAGE else mv
    # line freshness (LINE_FRESH, `+lf`): model version only, same bundle and seeds
    return vs, (mv + LINE_FRESH_TAG if LINE_FRESH else mv)


def _versions_bundle(on: bool, ext: bool = False, copula: str = "v0", mr: bool = False,
                     pre: bool = False, mf: bool = False, bo: bool = False) -> tuple[str, str]:
    if not on:
        vs, mv = _VS_BASE, MODEL_VERSION_BASE
    elif ext:
        mv = (CONTEXT_EXT_MODEL_VERSION if MINUTES_VERSION == "v1"
              else f"{MODEL_VERSION_BASE}+cx2")
        vs = _VS_BASE + CONTEXT_EXT_SUFFIX
    else:
        mv = CONTEXT_MODEL_VERSION if MINUTES_VERSION == "v1" else f"{MODEL_VERSION_BASE}+cx"
        vs = _VS_BASE + CONTEXT_SUFFIX
    if pre:
        vs, mv = vs + PRESEASON_SUFFIX, mv + PRESEASON_TAG
    if mf:
        vs, mv = vs + MINFIX_SUFFIX, mv + MINFIX_TAG
    if bo:
        vs, mv = vs + BLOWOUT_SUFFIX, mv + BLOWOUT_TAG
    if EP.EARLY_PRIOR:          # early-season prior (+ep): its own bundle (state["early_prior"])
        vs, mv = vs + EP.EARLY_PRIOR_SUFFIX, mv + EP.EARLY_PRIOR_TAG
    if MIN_SPREAD:              # player-specific minutes spread (+ms): its own minutes params
        vs, mv = vs + MIN_SPREAD_SUFFIX, mv + MIN_SPREAD_TAG
    if mr:
        vs, mv = vs + MINRATIO_SUFFIX, mv + MINRATIO_TAG
    if copula == "v1":
        return vs + COPULA_V1_SUFFIX, mv + COPULA_V1_TAG
    if copula == "v1r":
        return vs + COPULA_V1R_SUFFIX, mv + COPULA_V1R_TAG
    return vs, mv


CONTEXT_ON = _env_flag(CONTEXT_ENV, CONTEXT_DEFAULT)
CONTEXT_EXT = _env_flag(CONTEXT_EXT_ENV, CONTEXT_EXT_DEFAULT)
COPULA = _copula_flag(COPULA_ENV, COPULA_DEFAULT)
PPLAY = _pplay_flag(PPLAY_ENV, PPLAY_DEFAULT)
MINRATIO = _env_flag(MINRATIO_ENV, MINRATIO_DEFAULT)
PRESEASON = _env_flag(PRESEASON_ENV, PRESEASON_DEFAULT)
MINFIX = _env_flag(MINFIX_ENV, MINFIX_DEFAULT)
BLOWOUT = _env_flag(BLOWOUT_ENV, BLOWOUT_DEFAULT)
MIN_SPREAD = _env_flag(MIN_SPREAD_ENV, MIN_SPREAD_DEFAULT)
_VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                               MINFIX, BLOWOUT)
# Seeds use the base version in both states: ON and OFF runs of the same slate/day draw common
# random numbers, so their difference is the layer, not Monte Carlo noise (OFF is unchanged).
SEED_VERSION = MODEL_VERSION_BASE


def set_context(on: bool, ext: bool | None = None) -> None:
    """Flip the context switch at run time (evaluation scripts, tests): updates CONTEXT_ON (and
    CONTEXT_EXT when `ext` is given), the bundle suffix and MODEL_VERSION together. Modules that
    copied MODEL_VERSION at import keep their copy; `pipeline.MODEL_VERSION` is read at call
    time everywhere in proj/."""
    global CONTEXT_ON, CONTEXT_EXT, _VS, MODEL_VERSION
    CONTEXT_ON = bool(on)
    if ext is not None:
        CONTEXT_EXT = bool(ext)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_copula(version: str) -> None:
    """Flip the copula switch at run time ("v0" | "v1" | "v1r"): updates COPULA, the bundle
    suffix and MODEL_VERSION together (the context switches are kept)."""
    global COPULA, _VS, MODEL_VERSION
    if version not in ("v0", "v1", "v1r"):
        raise ValueError(f"unknown copula version {version!r}")
    COPULA = version
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_pplay(version: str) -> None:
    """Flip the P(play) switch at run time ("v1" | "v2"): updates PPLAY and MODEL_VERSION
    together (the bundle suffix does not change)."""
    global PPLAY, _VS, MODEL_VERSION
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown P(play) version {version!r}")
    PPLAY = version
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_minratio(on: bool) -> None:
    """Flip the minutes-ratio switch at run time: updates MINRATIO, the bundle suffix and
    MODEL_VERSION together (the other switches are kept)."""
    global MINRATIO, _VS, MODEL_VERSION
    MINRATIO = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_preseason(on: bool) -> None:
    """Flip the preseason switch at run time: updates PRESEASON, the bundle suffix and
    MODEL_VERSION together (the other switches are kept). A Context built while it was off has no
    preseason logs; `load_bundle` / `Context` attach them when it is on."""
    global PRESEASON, _VS, MODEL_VERSION
    PRESEASON = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_minfix(on: bool) -> None:
    """Flip the minutes-fix switch at run time: updates MINFIX, the bundle suffix and
    MODEL_VERSION together (the other switches are kept)."""
    global MINFIX, _VS, MODEL_VERSION
    MINFIX = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_blowout(on: bool) -> None:
    """Flip the blowout-minutes switch at run time: updates BLOWOUT, the bundle suffix and
    MODEL_VERSION together (the other switches are kept)."""
    global BLOWOUT, _VS, MODEL_VERSION
    BLOWOUT = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_min_spread(on: bool) -> None:
    """Flip the player-specific minutes spread: updates MIN_SPREAD, the bundle suffix and
    MODEL_VERSION together (the other switches are kept)."""
    global MIN_SPREAD, _VS, MODEL_VERSION
    MIN_SPREAD = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_pplayq(on: bool) -> None:
    """Flip the lock-10 Questionable refit (availability.PPLAYQ) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    AV.set_pplayq(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_inj_gameday(on: bool) -> None:
    """Flip game-day keying of injury rows (availability.INJ_GAMEDAY) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    AV.set_inj_gameday(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_report_match(on: bool) -> None:
    """Flip current-roster report matching (availability.REPORT_MATCH) and MODEL_VERSION."""
    global _VS, MODEL_VERSION
    AV.set_report_match(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_ghost_drop(on: bool) -> None:
    """Flip the ghost-candidate drop (GHOST_DROP) and MODEL_VERSION together."""
    global GHOST_DROP, _VS, MODEL_VERSION
    GHOST_DROP = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_line_fresh(on: bool) -> None:
    """Flip the line-freshness bound (LINE_FRESH) and MODEL_VERSION together."""
    global LINE_FRESH, _VS, MODEL_VERSION
    LINE_FRESH = bool(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_recon_reserve(on: bool) -> None:
    """Flip the reconciliation reserve (reconcile.RECON_RESERVE) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    RC.set_recon_reserve(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_recon_weighted(on: bool) -> None:
    """Flip the weighted reconciliation (reconcile.RECON_WEIGHTED) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    RC.set_recon_weighted(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_reserve_extras(on: bool) -> None:
    """Flip the reserve extras (minutes.RESERVE_EXTRAS) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    M.set_reserve_extras(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_lineup(on: bool) -> None:
    """Flip the lineup-known adjustments (minutes.LINEUP) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    M.set_lineup(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_tier_match(on: bool) -> None:
    """Flip the draw's fit-matched tiers (copula.TIER_MATCH) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    C.set_tier_match(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_role_blend(on: bool) -> None:
    """Flip the role-aware blend (role_blend.ROLE_BLEND) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    RB.set_role_blend(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_tail_remap(on: bool) -> None:
    """Flip the tail remap (tail_remap.TAIL_REMAP) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    TR.set_tail_remap(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_early_prior(on: bool) -> None:
    """Flip the early-season prior (early_prior.EARLY_PRIOR), the bundle suffix and
    MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    EP.set_early_prior(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_usage(on: bool) -> None:
    """Flip usage v1 redistribution (usage.USAGE, `+us`) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    US.set_usage(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def set_min_restrict(on: bool) -> None:
    """Flip the minutes restriction (min_restrict.MIN_RESTRICT) and MODEL_VERSION together."""
    global _VS, MODEL_VERSION
    MRS.set_min_restrict(on)
    _VS, MODEL_VERSION = _versions(CONTEXT_ON, CONTEXT_EXT, COPULA, PPLAY, MINRATIO, PRESEASON,
                                   MINFIX, BLOWOUT)


def context_groups() -> tuple:
    """Feature groups of the layer the current switch fits (P + F2, or the extended set)."""
    return tuple(CX.LIVE_GROUPS_EXT if CONTEXT_EXT else CX.LIVE_GROUPS)


HISTORY_SEASONS = 7            # box seasons loaded for the minutes engine (window 5 + warm-up)
MINUTES_WINDOW = 5
LISTED = set(AV.LISTED) | {"removed", "daytoday"}
DEFAULT_TOTAL = 225.0
DEFAULT_HOME_EDGE = 2.0
STATS_OUT = ("min", "dk")


# ============================================================================== small helpers

def season_of(day: date) -> str:
    y = day.year if day.month >= 8 else day.year - 1
    if day.year == 2020 and 8 <= day.month <= 10:   # 2019-20 bubble restart ran to 2020-10-11
        y = 2019
    return f"{y}-{(y + 1) % 100:02d}"


def _prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _season_minus(season: str, k: int) -> str:
    y = int(season[:4]) - k
    return f"{y}-{(y + 1) % 100:02d}"


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _slate_day(pool) -> date:
    """The NBA (Eastern) day of a slate without a ``dfs_slates`` row: its earliest tip in ET.
    A 9:30 PM ET tip is the next day in UTC, so the UTC date of any one player is wrong."""
    # DK start times look like "2026-10-20T19:00:00.0000000Z" (UTC, 7-digit fraction).
    starts = [datetime.fromisoformat(p.start_time[:19]).replace(tzinfo=timezone.utc)
              for p in pool.players if p.start_time]
    return min(starts).astimezone(config.NBA_TZ).date()


def _aware(at) -> datetime:
    if isinstance(at, str):
        at = datetime.fromisoformat(at.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("`at` must be timezone-aware")
    return at.astimezone(timezone.utc)


def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_seed(*parts) -> int:
    """Process-independent 63-bit seed from the parts (slate/day, run kind, model version)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def game_seed(seed: int, game_id: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed) & (2 ** 63 - 1), zlib.crc32(str(game_id).encode())])


def cache_dir() -> Path:
    return config.data_dir() / "proj"


# ============================================================================== context, bundle

class Context:
    """Data loaded once per process for one season: the minutes History (box seasons S-7..),
    the rate-state game frame and positions, and pair shares (events of the training seasons
    plus season S; every lookup is strictly before the projected date)."""

    def __init__(self, conn: sqlite3.Connection, season: str, *, history: M.History | None = None,
                 log: Callable[[str], None] | None = None):
        self.season = season
        self.log = log or (lambda m: None)
        s0 = int(season[:4])
        since = f"{s0 - HISTORY_SEASONS}-{(s0 - HISTORY_SEASONS + 1) % 100:02d}"
        t0 = time.time()
        self.h = history if history is not None else M.load_history(conn, since=since)
        if PRESEASON and self.h.pre is None:
            M.attach_preseason(self.h, conn)
        st_first = f"{s0 - 1 - 6}-{(s0 - 6) % 100:02d}"
        self.games = ST.load_games(conn, since_season=st_first)
        self.pos = ST.load_positions(conn)
        self._shares: M.PairShares | None = None
        self.train_seasons = sorted(x for x in self.h.tg["season"].unique()
                                    if x < season)[-MINUTES_WINDOW:]
        # context layer inputs (only when the switch is on): team-game box history of S-2..S
        # now, the EB base rows on first use (they need the bundle's rate-state params)
        self.cx_t: pd.DataFrame | None = None
        self._cx_wf: dict = {}
        self._cx_ext: dict = {}
        if CONTEXT_ON:
            from .features.common import load_team_games
            self.cx_t = load_team_games(conn, _season_minus(season, 2), season)
        self.seconds_load = time.time() - t0

    def context_inputs(self, conn: sqlite3.Connection, sp: dict
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(base rows of seasons S-2..S under the rate-state params `sp`, team games S-2..S) for
        `features.day_features`; built once per (season, sp). Rows on or after a projected day
        are ignored by construction (as-of sums strictly before the day)."""
        from .features.common import base_rows, load_team_games
        if self.cx_t is None:
            self.cx_t = load_team_games(conn, _season_minus(self.season, 2), self.season)
        key = hashlib.sha1(json.dumps(sp, sort_keys=True, default=str).encode()).hexdigest()
        if key not in self._cx_wf:
            t0 = time.time()
            seasons = [_season_minus(self.season, k) for k in (2, 1, 0)]
            self._cx_wf = {key: base_rows(conn, seasons, sp, games=self.games, pos=self.pos)}
            self.log(f"context base rows {seasons[0]}..{seasons[-1]}: {len(self._cx_wf[key])} "
                     f"({time.time() - t0:.0f}s)")
        return self._cx_wf[key], self.cx_t

    def context_extras(self, conn: sqlite3.Connection, groups: tuple):
        """Per-season inputs of the extension groups (F3 / F5 / F6) of the layer, built once
        (`features.LiveExtras`; every use strictly before the projected day)."""
        from .features import LiveExtras
        key = tuple(groups)
        if key not in self._cx_ext:
            self._cx_ext[key] = LiveExtras(conn, self.season, key, pos=self.pos, log=self.log)
        return self._cx_ext[key]

    def train_panel(self) -> pd.DataFrame:
        return M.oracle_panel(self.h, M.eval_targets(self.h, self.train_seasons))

    @property
    def shares(self) -> M.PairShares:
        if self._shares is None:
            t0 = time.time()
            ev = [M.pair_events(self.train_panel())]
            cur = M.eval_targets(self.h, [self.season])
            if len(cur):
                ev.append(M.pair_events(M.oracle_panel(self.h, cur)))
            self._shares = M.PairShares(pd.concat(ev, ignore_index=True))
            self.log(f"pair shares built in {time.time() - t0:.1f}s")
        return self._shares


@dataclass
class Bundle:
    season: str
    minutes: M.MinutesParams
    state: dict
    copula: C.CopulaParams
    context: CX.ContextModel | None = None      # Phase 2F layer (used only when CONTEXT_ON)
    copula_v1: C.CopulaV1Params | None = None   # T2.10 v1 params (used only when COPULA v1)

    def to_json(self) -> str:
        d = {"season": self.season, "minutes": json.loads(self.minutes.to_json()),
             "state": self.state, "copula": json.loads(self.copula.to_json()),
             "model_version": MODEL_VERSION}
        if self.context is not None:          # absent key: the pre-context format, byte for byte
            d["context"] = json.loads(self.context.to_json())
        if self.copula_v1 is not None:        # absent key: the v0 format, byte for byte
            d["copula_v1"] = json.loads(self.copula_v1.to_json())
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Bundle":
        d = json.loads(s)
        cx = d.get("context")
        c1 = d.get("copula_v1")
        return cls(season=d["season"], minutes=M.MinutesParams(**d["minutes"]),
                   state=d["state"], copula=C.CopulaParams(**d["copula"]),
                   context=CX.ContextModel(**cx) if cx else None,
                   copula_v1=C.CopulaV1Params(**c1) if c1 else None)


def fit_state_params(conn: sqlite3.Connection, season: str, log=None) -> dict:
    """Rate-state params for seasons < S: DEFAULT_PARAMS when it was fit exactly on those, a
    cached `state_params_<S>.json`, else fit_params on 8 seasons + the usage bump on 3."""
    path = cache_dir() / f"state_params_{season}.json"
    if path.exists():
        return json.loads(path.read_text())
    if ST.DEFAULT_PARAMS.get("fit_seasons", [None, None])[-1] == _prev_season(season):
        return ST.DEFAULT_PARAMS
    s0 = int(season[:4])
    seasons = [f"{y}-{(y + 1) % 100:02d}" for y in range(s0 - 8, s0)]
    p = ST.fit_params(conn, seasons, log=log)
    p["usage_bump"] = ST.fit_usage_bump(conn, p, seasons[-3:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ST.params_to_json(p))
    return p


# One LightGBM thread for the live layer's fit: with 3 threads the early-stopping round can
# differ between two fits of the same frame; with 1 a refit is byte-identical on any machine.
CONTEXT_FIT_PARAMS = {"num_threads": 1}


def _fit_context(conn: sqlite3.Connection, season: str, sp: dict, say) -> CX.ContextModel:
    return CX.fit_season(conn, season, sp, groups=context_groups(), params=CONTEXT_FIT_PARAMS,
                         log=say)


def load_bundle(conn: sqlite3.Connection, season: str, ctx: Context | None = None, *,
                refit: bool = False, log=None) -> Bundle:
    """The fitted Bundle for season S (everything fit on seasons < S), cached on disk. With the
    context switch on: `bundle_<S>_v1_cx.json` = the OFF bundle of S (reused from disk when
    present, so ON and OFF differ only by the layer) plus the context layer fit on seasons < S.
    With COPULA v1: `<the arm's file>_c1.json` = the arm's v0 bundle (loaded or fitted the v0
    way) plus `copula_v1` fit on seasons < S. With MINRATIO: `<that file>_mr.json` (before a
    `_c1`) = that arm's bundle refit for the planned-minutes table (`_load_bundle_mr`)."""
    if MINRATIO:
        return _load_bundle_mr(conn, season, ctx, refit=refit, log=log)
    if COPULA == "v1":
        return _load_bundle_v1(conn, season, ctx, refit=refit, log=log)
    if COPULA == "v1r":
        return _load_bundle_v1r(conn, season, ctx, refit=refit, log=log)
    if MIN_SPREAD:
        return _load_bundle_ms(conn, season, ctx, refit=refit, log=log)
    if EP.EARLY_PRIOR:
        return _load_bundle_ep(conn, season, ctx, refit=refit, log=log)
    if BLOWOUT:
        return _load_bundle_bo(conn, season, ctx, refit=refit, log=log)
    if MINFIX:
        return _load_bundle_mf(conn, season, ctx, refit=refit, log=log)
    if PRESEASON:
        return _load_bundle_pre(conn, season, ctx, refit=refit, log=log)
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if CONTEXT_ON and b.context is None:
            b.context = _fit_context(conn, season, b.state, say)
            path.write_text(b.to_json())
        return b
    if CONTEXT_ON and not refit:
        base = cache_dir() / f"bundle_{season}{_VS_BASE}.json"
        if base.exists():
            t0 = time.time()
            b = Bundle.from_json(base.read_text())
            b.context = _fit_context(conn, season, b.state, say)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(b.to_json())
            say(f"bundle {season} + context fitted in {time.time() - t0:.0f}s -> {path}")
            return b
    t0 = time.time()
    sp = fit_state_params(conn, season, log=say)
    mpath = cache_dir() / f"minutes_params_{season}{_VS_BASE}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        panel = ctx.train_panel()
        mp = M.fit(ctx.h, season, window=MINUTES_WINDOW, log=say, panel=panel,
                   shares=M.PairShares(M.pair_events(panel)), version=MINUTES_VERSION)
    cp = C.fit_copula(conn, season, sp, log=say)
    b = Bundle(season=season, minutes=mp, state=sp, copula=cp)
    if CONTEXT_ON:
        b.context = _fit_context(conn, season, sp, say)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} fitted in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_v1(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if b.copula_v1 is not None:
            return b
    set_copula("v0")
    try:
        b = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_copula("v1")
    t0 = time.time()
    if b.minutes.blowout:       # blowout arm: the shift is fit with f read net of the margin
        frame = C.v1_frame(conn, season, b.state, b.copula.alpha, mp=b.minutes)
        b.copula_v1 = C.fit_copula_v1(conn, season, b.state, b.copula, frame=frame, log=say)
    else:
        b.copula_v1 = C.fit_copula_v1(conn, season, b.state, b.copula, log=say)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} + copula v1 fitted in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_v1r(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                     refit: bool, log) -> Bundle:
    """The copula v1 retune's bundle: the arm's v0 bundle (loaded or fitted the v0 way) plus
    `copula_v1` = `copula.fit_copula_v1r` on seasons < S (the arm's context layer and minutes
    params), cached as `bundle_<S><arm>_c1r.json`."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if b.copula_v1 is not None and b.copula_v1.version == C.COPULA_V1R_VERSION:
            return b
    set_copula("v0")
    try:
        b = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_copula("v1r")
    t0 = time.time()
    b.copula_v1 = C.fit_copula_v1r(conn, season, b.state, b.copula, b.context, b.minutes,
                                   log=say)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} + copula v1r fitted in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_pre(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                     refit: bool, log) -> Bundle:
    """The preseason arm's bundle: the same arm's bundle with PRESEASON off (loaded or fitted the
    usual way) with its minutes params replaced by a refit on seasons < S that adds
    `minutes.V1_PRESEASON` to the LightGBM base and the season-start debut reserve
    (`minutes.debut_reserve`; `minutes_params_<S>_v1_pre.json`), cached as
    `bundle_<S><arm>_pre.json`. Rate state, copula and context layer are the arm's."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if M.uses_preseason(b.minutes):
            return b
    set_preseason(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_preseason(True)
    t0 = time.time()
    mpath = cache_dir() / f"minutes_params_{season}{_VS_BASE}{PRESEASON_SUFFIX}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        if ctx.h.pre is None:
            M.attach_preseason(ctx.h, conn)
        panel = ctx.train_panel()
        mp = M.fit(ctx.h, season, window=MINUTES_WINDOW, log=say, panel=panel,
                   shares=M.PairShares(M.pair_events(panel)), version=MINUTES_VERSION,
                   v1_kw={"features": M.preseason_features()}, debut_reserve=True)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(mp.to_json())
    b = Bundle(season=season, minutes=mp, state=base.state, copula=base.copula,
               context=base.context, copula_v1=base.copula_v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (preseason minutes) in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_mf(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    """The minutes-fix arm's bundle: the same arm's bundle with MINFIX off (loaded or fitted the
    usual way; the preseason arm's when PRESEASON is on) with its minutes params replaced by a
    refit on seasons < S with `fit_v1(minfix=True)` (+ the preseason features and debut reserve
    when PRESEASON is on; `minutes_params_<S>_v1[_pre]_mf.json`), cached as
    `bundle_<S><arm>_mf.json`. Rate state, copula and context layer are the arm's."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if b.minutes.minfix and (M.uses_preseason(b.minutes) == PRESEASON):
            return b
    set_minfix(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_minfix(True)
    t0 = time.time()
    tag = _VS_BASE + (PRESEASON_SUFFIX if PRESEASON else "") + MINFIX_SUFFIX
    mpath = cache_dir() / f"minutes_params_{season}{tag}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        if PRESEASON and ctx.h.pre is None:
            M.attach_preseason(ctx.h, conn)
        panel = ctx.train_panel()
        kw: dict = {"minfix": True}
        if PRESEASON:
            kw["features"] = M.preseason_features()
        mp = M.fit(ctx.h, season, window=MINUTES_WINDOW, log=say, panel=panel,
                   shares=M.PairShares(M.pair_events(panel)), version=MINUTES_VERSION,
                   v1_kw=kw, debut_reserve=PRESEASON)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(mp.to_json())
    b = Bundle(season=season, minutes=mp, state=base.state, copula=base.copula,
               context=base.context, copula_v1=base.copula_v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (minutes fix) in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_bo(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    """The blowout arm's bundle: the same arm's bundle with BLOWOUT off (loaded or fitted the
    usual way; the minutes-fix / preseason arm's when those are on) with its minutes params
    replaced by a refit on seasons < S with `fit_v1(blowout=True)` (+ minfix when MINFIX is on,
    + the preseason features and debut reserve when PRESEASON is on;
    `minutes_params_<S>_v1[_pre][_mf]_bo.json`), cached as `bundle_<S><arm>_bo.json`. Rate state,
    copula and context layer are the arm's."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if (b.minutes.blowout and bool(b.minutes.minfix) == MINFIX
                and M.uses_preseason(b.minutes) == PRESEASON):
            return b
    set_blowout(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_blowout(True)
    t0 = time.time()
    tag = (_VS_BASE + (PRESEASON_SUFFIX if PRESEASON else "")
           + (MINFIX_SUFFIX if MINFIX else "") + BLOWOUT_SUFFIX)
    mpath = cache_dir() / f"minutes_params_{season}{tag}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        if PRESEASON and ctx.h.pre is None:
            M.attach_preseason(ctx.h, conn)
        panel = ctx.train_panel()
        kw: dict = {"blowout": True}
        if MINFIX:
            kw["minfix"] = True
        if PRESEASON:
            kw["features"] = M.preseason_features()
        mp = M.fit(ctx.h, season, window=MINUTES_WINDOW, log=say, panel=panel,
                   shares=M.PairShares(M.pair_events(panel)), version=MINUTES_VERSION,
                   v1_kw=kw, debut_reserve=PRESEASON)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(mp.to_json())
    b = Bundle(season=season, minutes=mp, state=base.state, copula=base.copula,
               context=base.context, copula_v1=base.copula_v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (blowout minutes) in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_ep(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    """The early-season prior's bundle: the same arm's bundle with EARLY_PRIOR off (loaded or
    fitted the usual way) plus `state["early_prior"]` = `early_prior.fit` on seasons < S with
    the arm's rate-state params, cached as `bundle_<S><arm>_ep.json`. Minutes, copula and
    context layer are the arm's."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if (EP.has_ep(b.state) and bool(b.minutes.minfix) == MINFIX
                and bool(b.minutes.blowout) == BLOWOUT
                and M.uses_preseason(b.minutes) == PRESEASON):
            return b
    set_early_prior(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_early_prior(True)
    t0 = time.time()
    sp = dict(base.state)
    sp["early_prior"] = EP.fit(conn, season, base.state, log=say)
    b = Bundle(season=season, minutes=base.minutes, state=sp, copula=base.copula,
               context=base.context, copula_v1=base.copula_v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (early-season prior) in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_ms(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    """The minutes-spread arm's bundle: the same arm's bundle with MIN_SPREAD off (loaded or
    fitted the usual way) with `minutes.fit_spread` added to its minutes params on seasons < S
    (`minutes_params_<S>_v1[_pre][_mf][_bo]_ms.json`), cached as `bundle_<S><arm>_ms.json`. Rate
    state, copula and context layer are the arm's."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if (b.minutes.spread and bool(b.minutes.minfix) == MINFIX
                and bool(b.minutes.blowout) == BLOWOUT
                and M.uses_preseason(b.minutes) == PRESEASON):
            return b
    set_min_spread(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_min_spread(True)
    t0 = time.time()
    tag = (_VS_BASE + (PRESEASON_SUFFIX if PRESEASON else "") + (MINFIX_SUFFIX if MINFIX else "")
           + (BLOWOUT_SUFFIX if BLOWOUT else "") + MIN_SPREAD_SUFFIX)
    mpath = cache_dir() / f"minutes_params_{season}{tag}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        if PRESEASON and ctx.h.pre is None:
            M.attach_preseason(ctx.h, conn)
        panel = ctx.train_panel()
        mp = M.fit_spread(base.minutes, panel, M.PairShares(M.pair_events(panel)), log=say)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(mp.to_json())
    b = Bundle(season=season, minutes=mp, state=base.state, copula=base.copula,
               context=base.context, copula_v1=base.copula_v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (minutes spread) in {time.time() - t0:.0f}s -> {path}")
    return b


def _load_bundle_mr(conn: sqlite3.Connection, season: str, ctx: Context | None, *,
                    refit: bool, log) -> Bundle:
    """The minutes-ratio arm's bundle: the same arm's bundle with MINRATIO off (loaded or fitted
    the usual way), then `minutes_ratio.fit_bundle` (table + bump, copula v0, context layer with
    re-keyed offsets; all on seasons < S), cached as `bundle_<S><arm>_mr[_c1].json`. The table's
    training plans are the arm's own minutes engine's (`minutes_ratio.plans_tag`: the +mf
    engine's with MINFIX, `bundle_<S><arm>_mf_mr.json`; else the OFF engine's), and a cached
    bundle keyed on the other engine's plans (or with minutes of another arm) is refit."""
    from . import minutes_ratio as MR
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}{_VS}.json"
    if path.exists() and not refit:
        b = Bundle.from_json(path.read_text())
        if (ST.has_mr(b.state) and bool(b.minutes.minfix) == MINFIX
                and MR.table_matches(b.state["minutes_adj_mr"], MR.plans_tag(b.minutes))):
            return b
    set_minratio(False)
    try:
        base = load_bundle(conn, season, ctx, refit=refit, log=log)
    finally:
        set_minratio(True)

    def fit_cx(sp, frame):
        return CX.fit_season(conn, season, sp, groups=tuple(base.context.groups),
                             n_train=int(base.context.meta.get("n_train_seasons",
                                                               CX.LIVE_N_TRAIN)),
                             params=CONTEXT_FIT_PARAMS, frame=frame, log=say)
    b = MR.fit_bundle(conn, season, base, fit_context=fit_cx, log=say)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} (minutes ratio) -> {path}")
    return b


# ============================================================================== environment

def _age_min(captured_at: str, at: datetime) -> float:
    """Minutes from a line snapshot's `captured_at` to `at` (naive timestamps are UTC)."""
    t = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
    t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    return (at.astimezone(timezone.utc) - t).total_seconds() / 60.0


def environment(conn: sqlite3.Connection, game_id: str, at: datetime | None = None, *,
                mode: str = "live") -> dict:
    """Expected home margin / total and implied team points for a game.

    mode 'live': books at `at` -> gbm_market at `at` -> closing line -> default.
    mode 'closing': closing line -> default (backtests)."""
    if mode == "live" and at is not None:
        ts = _iso(at)
        rows = conn.execute(
            "SELECT book, home_spread, total, home_team_total, away_team_total, captured_at "
            "FROM line_snapshots WHERE game_id=? AND captured_at<=? AND book!='open' "
            "ORDER BY captured_at", (game_id, ts)).fetchall()
        if rows:
            latest = {}
            for r in rows:
                latest[r[0]] = r
            v = list(latest.values())
            ages = {r[0]: _age_min(r[5], at) for r in v}      # minutes before `at`, per book
            if LINE_FRESH and len(v) > 1:
                newest = min(ages.values())
                keep = [r for r in v if ages[r[0]] <= LINE_FRESH_MAX_H * 60
                        and ages[r[0]] - newest <= LINE_FRESH_REL_H * 60]
                v = keep or [r for r in v if ages[r[0]] == newest]   # newest always stays
            used = {r[0] for r in v}

            def med(i):
                x = [r[i] for r in v if r[i] is not None]
                return float(np.median(x)) if x else None
            hs, tot, htt, att = med(1), med(2), med(3), med(4)
            if hs is not None and tot is not None:
                h, a = RC.implied_points(-hs, tot, htt, att)
                return {"mu_margin": -hs, "mu_total": tot, "home_pts": h, "away_pts": a,
                        "source": f"books:{len(v)}",
                        "captured_at": max(r[5] for r in v),
                        "line_ages_min": ages,
                        "line_age_max_min": max(ages[b] for b in used)}
        r = conn.execute(
            "SELECT home_margin, total, created_at FROM predictions WHERE game_id=? AND "
            "model='gbm_market' AND created_at<=? AND (role IS NULL OR role='champion') "
            "ORDER BY created_at DESC LIMIT 1", (game_id, ts)).fetchone()
        if r is not None and r[0] is not None and r[1] is not None:
            h, a = RC.implied_points(r[0], r[1])
            return {"mu_margin": float(r[0]), "mu_total": float(r[1]), "home_pts": h,
                    "away_pts": a, "source": "gbm_market", "captured_at": r[2]}
    r = conn.execute("SELECT AVG(home_spread_close), AVG(total_close) FROM closing_lines "
                     "WHERE game_id=? AND home_spread_close IS NOT NULL AND total_close IS NOT "
                     "NULL", (game_id,)).fetchone()
    if r is not None and r[0] is not None:
        h, a = RC.implied_points(-r[0], r[1])
        return {"mu_margin": -float(r[0]), "mu_total": float(r[1]), "home_pts": h,
                "away_pts": a, "source": "closing"}
    h, a = RC.implied_points(DEFAULT_HOME_EDGE, DEFAULT_TOTAL)
    return {"mu_margin": DEFAULT_HOME_EDGE, "mu_total": DEFAULT_TOTAL, "home_pts": h,
            "away_pts": a, "source": "default"}


# ============================================================================== inputs

def _day_int(day: date) -> int:
    return int(np.datetime64(day, "D").astype(np.int64))


def _targets(games: pd.DataFrame, day: date) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        for team, opp in ((g.home, g.away), (g.away, g.home)):
            rows.append({"key": f"{g.game_id}|{team}", "game_id": g.game_id, "team": team,
                         "opp": opp, "date": _day_int(day), "season": season_of(day)})
    return pd.DataFrame(rows)


def _stars(panel: pd.DataFrame, rates: pd.DataFrame) -> np.ndarray:
    """Top-2 regular stars of each team-game (key) by DK/min x mpg (fit_usage_bump's rule)."""
    score = (rates["r_dk"] * rates["mpg"]).to_numpy(float)
    regular = ((panel["app10"].to_numpy(float) >= 7)
               | (panel["missed_now"].to_numpy(float) < 3)) & panel["last_date"].notna().to_numpy()
    out = np.zeros(len(panel), bool)
    for _, idx in panel.groupby("key", sort=False).indices.items():
        cand = idx[regular[idx]]
        top = cand[np.argsort(-score[cand], kind="stable")[:2]]
        out[top] = True
    return out


def _rb_history(ctx: Context, conn: sqlite3.Connection) -> RB.RoleHistory:
    """The role blend's box history, loaded once per Context (ON only)."""
    h = getattr(ctx, "_rb_hist", None)
    if h is None:
        h = ctx._rb_hist = RB.RoleHistory(conn, ctx.season)
    return h


def _ep_input(ctx: Context, b: Bundle, conn: sqlite3.Connection):
    """The early-season prior's input for `state.rate_frame` (None unless EARLY_PRIOR is on and
    the bundle carries its params); bio and preseason logs loaded once per Context."""
    if not (EP.EARLY_PRIOR and EP.has_ep(b.state)):
        return None
    data = getattr(ctx, "_ep_data", None)
    if data is None:
        data = ctx._ep_data = EP.load_data(conn)
    return EP.EPInput(b.state["early_prior"], data)


def build_games(ctx: Context, b: Bundle, conn: sqlite3.Connection, day: date,
                games: pd.DataFrame, panel: pd.DataFrame, p_avail: dict, env: dict,
                starters: set | None = None, pp_fix: dict | None = None,
                rs_caps: dict | None = None) -> list[C.GameSpec]:
    """GameSpecs from a candidate panel (pregame_panel rows keyed game_id|team, optional
    `absent` column) and P(available) {(key, player_id): p} (default 1.0; absent -> 0).
    `pp_fix` (P(play) v2, `pplay_v2_fix`): {(key, player_id): P(plays | available)} replacing
    the minutes engine's value for those players (None = the engine everywhere)."""
    panel = panel.reset_index(drop=True)
    q = pd.DataFrame({"player_id": panel["player_id"].astype(np.int64),
                      "game_date": day.isoformat()})
    ep = _ep_input(ctx, b, conn)                  # early-season prior (+ep); None when OFF
    rates = ST.rate_frame(conn, q, b.state, games=ctx.games, pos=ctx.pos,
                          **({"ep": ep} if ep is not None else {}))
    panel["_star"] = _stars(panel, rates)
    usage = US.live_model(conn, ctx.season, b.state) if US.USAGE else None   # +us (P8)
    if CONTEXT_ON and b.context is not None and len(panel):
        rates = apply_context(ctx, b, conn, day, games, panel, rates, env, p_avail=p_avail)
    shares = ctx.shares
    out = []
    by_key = panel.groupby("key", sort=False).indices
    for g in games.itertuples(index=False):
        teams = {}
        e = env[g.game_id]
        for side, team in (("home", g.home), ("away", g.away)):
            key = f"{g.game_id}|{team}"
            idx = by_key.get(key, np.zeros(0, int))
            p = panel.iloc[idx].reset_index(drop=True)
            r = rates.iloc[idx].reset_index(drop=True)
            if "absent" in p:
                pa = np.where(p["absent"].to_numpy(bool), 0.0, 1.0)
            else:
                pa = np.ones(len(p))
            pa = np.array([p_avail.get((key, int(pid)), v)
                           for pid, v in zip(p["player_id"], pa)], float)
            pl = pd.DataFrame({"player_id": p["player_id"].astype(np.int64), "p_avail": pa})
            for s in C.DK_STATS:
                pl[f"r_{s}"] = r[f"r_{s}"].to_numpy(float)
            pl["r_dk"] = r["r_dk"].to_numpy(float)
            pl["mpg"] = r["mpg"].to_numpy(float)
            pl["tier_state"] = r["tier"].to_numpy(int)
            pl["star"] = p["_star"].to_numpy(bool)
            pl["rotation"] = pl["mpg"].to_numpy() >= 15.0
            if RC.RECON_WEIGHTED:             # weighted reconciliation (+rw): rate evidence
                pl["n_pts"] = r["n_pts"].to_numpy(float)
                pl["k_pts"] = float(b.state["stats"]["pts"]["kappa"])
            if RB.ROLE_BLEND:                 # role blend (+rb): L10 / last-10 minutes inputs
                RB.attach(pl, _rb_history(ctx, conn), team, day)
            if COPULA in COPULA_V1_VERSIONS:  # position group for the T2.7 tier x position cells
                pl["grp"] = r["grp"].to_numpy(int)
            if usage is not None:             # usage v1 (+us): regulars, position groups
                pl["regular"] = (((p["app10"].to_numpy(float) >= 7)
                                  | (p["missed_now"].to_numpy(float) < 3))
                                 & p["last_date"].notna().to_numpy())
                if "grp" not in pl:
                    pl["grp"] = r["grp"].to_numpy(int)
            pm = p.drop(columns=["_star"])
            if pp_fix:
                pm["pp_fix"] = [pp_fix.get((key, int(pid)), np.nan) for pid in pm["player_id"]]
            if rs_caps and any(k == key for k, _ in rs_caps):   # +rs: planned-minutes caps
                pm["rs_cap"] = MRS.rs_column(pm, rs_caps)
            sm = C.ScenarioMinutes(pm, b.minutes, shares,
                                   starters=_team_starters(starters, p))
            teams[side] = C.TeamSpec(team=team, players=pl, minutes=sm,
                                     implied=e["home_pts"] if side == "home" else e["away_pts"])
            if usage is not None:
                teams[side].usage = US.team_matrix(usage, pl, day)
        out.append(C.GameSpec(game_id=g.game_id, home=teams["home"], away=teams["away"],
                              mu_margin=e["mu_margin"], mu_total=e["mu_total"],
                              env_source=e.get("source", "")))
    return out


def apply_context(ctx: Context, b: Bundle, conn: sqlite3.Connection, day: date,
                  games: pd.DataFrame, panel: pd.DataFrame, rates: pd.DataFrame,
                  env: dict, p_avail: dict | None = None) -> pd.DataFrame:
    """The context layer on the EB rates of the panel rows (`context.apply_rates`): F2 features
    of the day's games (as of strictly before `day`; the pace term uses the environment's market
    total, none for a league-default environment) and P from the rates themselves. A layer with
    extension groups (the CONTEXT_EXT bundle) also gets their per-day features
    (`Context.context_extras`): F3 per player, F5 over each opponent's candidates weighted by
    minutes state x P(available), F6 per team-game."""
    from . import features as FX
    wf, t = ctx.context_inputs(conn, b.state)
    totals = {g: float(e["mu_total"]) for g, e in env.items()
              if e.get("source") != "default" and e.get("mu_total") is not None}
    gm = games[["game_id", "home", "away"]]
    tf = FX.day_features(conn, wf, t, day, gm, totals=totals)
    key = panel["key"].str.split("|", n=1, expand=True)
    pf = None
    ext = tuple(g for g in b.context.groups if g in FX.EXTENSION_GROUPS)
    if ext:
        if "absent" in panel:
            pa = np.where(panel["absent"].to_numpy(bool), 0.0, 1.0)
        else:
            pa = np.ones(len(panel))
        pa = np.array([(p_avail or {}).get((k, int(pid)), v)
                       for k, pid, v in zip(panel["key"], panel["player_id"], pa)], float)
        rows = pd.DataFrame({"game_id": key[0].to_numpy(), "team": key[1].to_numpy(),
                             "player_id": panel["player_id"].astype(np.int64).to_numpy(),
                             "grp": rates["grp"].to_numpy(int),
                             "w": np.maximum(rates["mpg"].to_numpy(float), 1.0) * pa})
        tfx, pf = ctx.context_extras(conn, ext).day(conn, day, gm, rows)
        if "F6" in ext:
            tf = tf.merge(tfx, on=["game_id", "team"], how="left")
    return CX.apply_rates(b.context, rates, tf, key[0].to_numpy(), key[1].to_numpy(),
                          player_feats=pf)


def pplay_v2_fix(ctx: Context, panel: pd.DataFrame, p_avail: dict | None,
                 listed: set | None, starters: set | None = None,
                 salaries: dict | None = None) -> dict:
    """P(play) v2 (`PPLAY == "v2"`): {(key, player_id): P(plays | available)} overrides of the
    minutes engine on the live candidate path (`build_games(pp_fix=...)`).

    1. **Listed, not Out** (`listed`: the report's players with status in LISTED other than
       'out'): 1.0. Their P(available) is the report cell, which is fit on *played* (it already
       contains coach's-decision DNPs), so total P(play) = the calibrated cell. Multiplying it
       by the engine's own P(plays | available) counted the DNP twice (T2.4: listed bench 0.29
       vs 0.45 played).
    2. **Fringe pool players**: forced in (`roster == 0`: not in the team's last 10 games) and
       not on the report at all: the fringe model `availability.load_fringe()` on
       `availability.fringe_features` (box history strictly before the day; the DK salary
       `salaries` {player_id: salary} in the season's first games; a player DK prices at
       >= `availability.EARLY_ENGINE_SALARY` there keeps the engine's value); DK starter flag
       -> 0.99.
       The engine scores these as active roster players (~0.7 vs ~0.25 played).
    3. **Lineup known** (`minutes.LINEUP`, `+lu`, default OFF): a DK-flagged starter who is not
       on the report gets `minutes.LU_P_STARTER`.
    Everybody else keeps the engine's value."""
    out: dict = {}
    if not len(panel):
        return out
    listed = {int(x) for x in (listed or ())}
    reported = {int(x) for x in (p_avail or {})}
    starters = {int(x) for x in (starters or ())}
    keys = panel["key"].to_numpy(object)
    pids = panel["player_id"].to_numpy(np.int64)
    roster = panel["roster"].to_numpy(int) if "roster" in panel else np.ones(len(panel), int)
    fringe = np.array([r == 0 and int(p) not in reported for r, p in zip(roster, pids)], bool)
    if fringe.any():
        f = panel[fringe].reset_index(drop=True)
        X = AV.fringe_features(f, ctx.h, salaries)
        pf = AV.load_fringe().p(X)
        priced = ((X["regime"] == "early") & (X["salary"] >= AV.EARLY_ENGINE_SALARY)).to_numpy()
        for k, pid, v, keep in zip(f["key"], f["player_id"], pf, priced):
            if int(pid) in starters:
                out[(k, int(pid))] = AV.FRINGE_STARTER_P
            elif not keep:              # season start, DK-priced player: the engine's value
                out[(k, int(pid))] = float(v)
    for k, pid in zip(keys, pids):
        if int(pid) in listed:
            out[(k, int(pid))] = 1.0
    if M.LINEUP and starters:   # +lu (default OFF): confirmed starters off the report
        for k, pid in zip(keys, pids):
            if int(pid) in starters and int(pid) not in listed and (k, int(pid)) not in out:
                out[(k, int(pid))] = M.LU_P_STARTER
    return out


def _team_starters(starters: set | None, p: pd.DataFrame):
    """Confirmed starters of a team (only when exactly five are flagged for it)."""
    if not starters:
        return None
    s = [int(x) for x in p["player_id"] if int(x) in starters]
    return s if len(s) == 5 else None


# ============================================================================== results

@dataclass
class DayResult:
    day: date
    sims: list                      # GameSims per game
    players: pd.DataFrame           # one row per player (summary)
    stats: pd.DataFrame             # long: key (players index), stat, cond, mean, sd, q..
    teams: list                     # reconciliation info per team-game
    meta: dict = field(default_factory=dict)

    def projections(self, key: str = "player_id") -> dict:
        out = {}
        for r in self.players.itertuples(index=False):
            k = getattr(r, key)
            if k is None or (isinstance(k, float) and not np.isfinite(k)):
                continue
            out[k] = Projection(mean=round(float(r.dk_mean), 3), p25=round(float(r.dk_q25), 3),
                                p10=round(float(r.dk_q10), 3), p50=round(float(r.dk_q50), 3),
                                p90=round(float(r.dk_q90), 3))
        return out

    def fp_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(player_id (n,), fp (S, n) float32) over all games of the day."""
        if not self.sims:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        return (np.concatenate([g.player_id for g in self.sims]),
                np.concatenate([g.fp for g in self.sims], axis=1))


class ProjectionSet(dict):
    """{DK playerId (str): Projection} plus `.meta` (run_id, sim_path, seconds, checks) and
    `.result` (the DayResult, NBA ids)."""
    meta: dict
    result: DayResult | None


def summarize_games(sims: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    prow, srows = [], []
    for g in sims:
        played = g.minutes > 0
        sd = C.summarize(g.fp, played)
        sm = C.summarize(g.minutes, played)
        # v1: full per-stat summaries (sd, quantiles) from the component sims
        sst = ({s: C.summarize_counts(g.stats[s], played) for s in C.PROP_STATS}
               if g.stats is not None else None)
        info = g.info.reset_index(drop=True)
        for j in range(len(info)):
            r = info.iloc[j].to_dict()
            r["game_id"] = g.game_id
            a = sd[(sd["col"] == j) & (sd["cond"] == "all")].iloc[0]
            pd_ = sd[(sd["col"] == j) & (sd["cond"] == "played")].iloc[0]
            ma = sm[(sm["col"] == j) & (sm["cond"] == "all")].iloc[0]
            mpl = sm[(sm["col"] == j) & (sm["cond"] == "played")].iloc[0]
            for c in ("mean", "sd", "q05", "q10", "q25", "q50", "q75", "q90", "q95"):
                r[f"dk_{c}"] = a[c]
                r[f"dkp_{c}"] = pd_[c]
            r["min_mean"] = ma["mean"]
            r["min_played"] = mpl["mean"]
            key = len(prow)
            prow.append(r)
            for stat, frame in (("dk", sd), ("min", sm)):
                for cond in ("played", "all"):
                    row = frame[(frame["col"] == j) & (frame["cond"] == cond)].iloc[0].to_dict()
                    row.pop("col")
                    row.update({"key": key, "stat": stat})
                    srows.append(row)
            for s in C.PROP_STATS:
                if sst is not None:
                    for cond in ("played", "all"):
                        f = sst[s]
                        row = f[(f["col"] == j) & (f["cond"] == cond)].iloc[0].to_dict()
                        row.pop("col")
                        row.update({"key": key, "stat": s})
                        srows.append(row)
                    continue
                srows.append({"key": key, "stat": s, "cond": "all", "mean": r[f"{s}_all"]})
                srows.append({"key": key, "stat": s, "cond": "played",
                              "mean": r[f"{s}_played"] if r["p_play"] > 0 else None})
    return pd.DataFrame(prow), pd.DataFrame(srows)


def run_games(specs: list, b: Bundle, n_sims: int, seed: int, day: date,
              log: Callable[[str], None] | None = None) -> DayResult:
    t0 = time.time()
    if COPULA in COPULA_V1_VERSIONS:
        v1 = b.copula_v1 or C.default_copula_v1_params(b.season, b.copula)
        sims = [C.simulate_game_v1(gs, b.copula, v1, b.minutes, b.state, n_sims,
                                   game_seed(seed, gs.game_id)) for gs in specs]
    else:
        sims = [C.simulate_game(gs, b.copula, b.minutes, b.state, n_sims,
                                game_seed(seed, gs.game_id)) for gs in specs]
    t_sim = time.time() - t0
    rb = RB.apply(sims, specs, day) if RB.ROLE_BLEND else None     # role blend (+rb)
    tr = TR.apply(sims, day) if TR.TAIL_REMAP else None             # tail remap (+tr)
    players, stats = summarize_games(sims)
    teams = [dict(v, game_id=g.game_id) for g in sims for v in g.teams.values()]
    meta = {"seconds_sim": round(t_sim, 2), "seconds_summary": round(time.time() - t0 - t_sim, 2),
            "n_sims": n_sims, "seed": seed, "games": len(specs),
            "recon": RC.acceptance(teams), **({"role_blend": rb} if rb is not None else {}),
            **({"tail_remap": tr} if tr is not None else {}),
            "env": {g.game_id: {"mu_margin": g.mu_margin, "mu_total": g.mu_total,
                                "source": g.env_source} for g in specs}}
    return DayResult(day=day, sims=sims, players=players, stats=stats, teams=teams, meta=meta)


# ============================================================================== live

def report_avail(conn: sqlite3.Connection, day: date, at: datetime, games: pd.DataFrame,
                 starters: set | None = None) -> tuple[dict, dict, dict]:
    """From the official reports at `at`: P(available) {player_id: p} for report statuses
    (unlisted players are left to the minutes engine), report players to force into the
    candidates {(game_id, team): [player_id]} (everybody not 'out') and a summary."""
    av = AV.p_play_slate(conn, day, at, starters=starters)
    av = av[av["game_id"].isin(set(games["game_id"]))]
    p_avail, extras = {}, {}
    for r in av.itertuples(index=False):
        if r.status in LISTED:
            p_avail[int(r.player_id)] = float(r.p_play)
        # Force only players the report itself lists (it names their current team); its
        # "unlisted" regulars are placed on their last box-score team, which is stale for a
        # traded player, and are candidates through the minutes panel anyway.
        if r.status in LISTED and r.status != "out":
            extras.setdefault((r.game_id, r.team), []).append(int(r.player_id))
    n_listed = int(av["status"].isin(list(LISTED)).sum()) if len(av) else 0
    return p_avail, extras, {"n_report_rows": int(len(av)), "n_listed": n_listed}


def listed_of(report_extras: dict | None) -> set:
    """Player ids of `report_avail`'s forced report players (= listed, not 'out')."""
    return {int(p) for v in (report_extras or {}).values() for p in v}


def merge_extras(primary: dict | None, report: dict | None) -> dict:
    """Forced candidates {(game_id, team): [player_id]}. `primary` (the DK pool: each player's
    CURRENT team) wins over `report`, which places unlisted players on their last box-score
    team (a traded player's old team); `candidate_panel` keeps a player's first occurrence."""
    ex = {k: list(v) for k, v in (primary or {}).items()}
    placed = {int(p) for v in ex.values() for p in v}
    for k, v in (report or {}).items():
        rest = [p for p in v if int(p) not in placed]
        if rest:
            ex.setdefault(k, []).extend(rest)
    return ex


def candidate_panel(ctx: Context, day: date, games: pd.DataFrame,
                    extras: dict | None = None) -> pd.DataFrame:
    """Pregame minutes candidates of each team-game (players of the team's last 10 games whose
    last team it is) + `extras` {(game_id, team): [player_id]} (report / DK pool players with
    box history; a player forced onto a team is removed from any other team's candidates).
    Players without any box history (rookies before their debut) are not projected."""
    tg = _targets(games, day)
    ex = [{"key": f"{gid}|{team}", "player_id": int(p)}
          for (gid, team), pids in (extras or {}).items() for p in pids]
    exdf = pd.DataFrame(ex, columns=["key", "player_id"]).drop_duplicates("player_id")
    exdf = exdf[exdf["key"].isin(set(tg["key"]))]
    panel = M.pregame_panel(ctx.h, tg, extra=exdf if len(exdf) else None)
    panel = panel[(panel["roster"] == 1) | panel["last_date"].notna()]
    if len(exdf):
        fteam = dict(zip(exdf["player_id"], exdf["key"].str.split("|").str[1]))
        keep = [fteam.get(int(p), t) == t for p, t in zip(panel["player_id"], panel["team"])]
        panel = panel[np.asarray(keep, bool)]
    if M.RESERVE_EXTRAS:        # reserve extras (+rx): which candidates were forced in
        fk = set(zip(exdf["key"], exdf["player_id"].astype(int)))
        panel = panel.assign(forced=[(k, int(p)) in fk
                                     for k, p in zip(panel["key"], panel["player_id"])])
    panel = panel.merge(tg[["key", "game_id"]], on="key", how="left")
    return panel.reset_index(drop=True)


def current_teams(conn: sqlite3.Connection, season: str) -> dict[int, str]:
    """{player_id: team} from the season's `team_rosters` snapshot ({} if not loaded). Only
    for live days: historical seasons' snapshots are taken at fetch time, not on the day."""
    rows = conn.execute("SELECT player_id, team FROM team_rosters WHERE season=?",
                        (season,)).fetchall()
    return {int(p): t for p, t in rows if p is not None and t}


def drop_moved_players(panel: pd.DataFrame, current: dict[int, str],
                       extras: dict | None = None, strict: bool = False) -> pd.DataFrame:
    """Remove candidates whose current roster team differs from the team they are a candidate
    for (an offseason or mid-season move: last season's box games put a traded player on his
    old team). Players forced onto a team via `extras` (the DK pool, the report) are kept.
    `strict` (GHOST_DROP, `+gh`): a candidate on no team of `current` is removed too."""
    if not current or not len(panel):
        return panel
    forced = {(team, int(p)) for (_, team), pids in (extras or {}).items() for p in pids}
    if strict:                  # +gh (default OFF: this branch never runs)
        keep = [(t, int(p)) in forced or current.get(int(p)) == t
                for p, t in zip(panel["player_id"], panel["team"])]
        return panel[np.asarray(keep, bool)].reset_index(drop=True)
    keep = [(t, int(p)) in forced or current.get(int(p), t) == t
            for p, t in zip(panel["player_id"], panel["team"])]
    return panel[np.asarray(keep, bool)].reset_index(drop=True)


def project_games(conn: sqlite3.Connection, day, games: pd.DataFrame, *, ctx: Context,
                  bundle: Bundle, n_sims: int, seed: int, p_avail: dict | None = None,
                  at: datetime | None = None, env_mode: str = "live",
                  extras: dict | None = None, starters: set | None = None,
                  listed: set | None = None, salaries: dict | None = None,
                  log: Callable[[str], None] | None = None,
                  rs_at: datetime | None = None) -> DayResult:
    """Core of the live and report paths: candidates (+ extras), P(available) {player_id: p}
    (default 1; 0 = out), environment per `env_mode` at `at`, simulate, summarise.
    `listed` (the report's listed, not-Out players) and `salaries` ({player_id: DK salary}):
    P(play) v2 only (`pplay_v2_fix`). `rs_at`: the run time the minutes restriction (+rs)
    reads the ESPN comments at when `at` is None (the report path)."""
    day = _as_date(day)
    panel = candidate_panel(ctx, day, games, extras)
    if env_mode == "live":
        if GHOST_DROP:          # +gh (default OFF: the else line is the unchanged call)
            panel = drop_moved_players(panel, current_teams(conn, season_of(day)), extras,
                                       strict=True)
        else:
            panel = drop_moved_players(panel, current_teams(conn, season_of(day)), extras)
    pa = {(k, int(pid)): float(p_avail[int(pid)])
          for k, pid in zip(panel["key"], panel["player_id"]) if int(pid) in (p_avail or {})}
    env = {g: environment(conn, g, at, mode=env_mode) for g in games["game_id"]}
    rs = (MRS.restrict_caps(conn, panel, at if at is not None else rs_at)
          if MRS.MIN_RESTRICT else None)
    if PPLAY == "v2":
        fix = pplay_v2_fix(ctx, panel, p_avail, listed, starters, salaries)
        specs = build_games(ctx, bundle, conn, day, games, panel, pa, env, starters, pp_fix=fix,
                            rs_caps=rs)
    else:
        specs = build_games(ctx, bundle, conn, day, games, panel, pa, env, starters,
                            rs_caps=rs)
    return run_games(specs, bundle, n_sims, seed, day, log)


def _day_games(conn: sqlite3.Connection, day: date, game_ids=None) -> pd.DataFrame:
    g = AV.day_games(conn, day)
    if game_ids is not None:
        g = g[g["game_id"].isin(set(game_ids))]
    return g.reset_index(drop=True)


def project_day(conn: sqlite3.Connection, nba_day, at, *, n_sims: int = 25_000,
                seed: int | None = None, kind: str = "manual", game_ids=None,
                ctx: Context | None = None, bundle: Bundle | None = None,
                extras: dict | None = None, starters: set | None = None,
                salaries: dict | None = None, dk_out: set | None = None,
                log: Callable[[str], None] | None = None) -> DayResult:
    """All games of an NBA day (or `game_ids`) as known at `at`, keyed by NBA player_id.
    `salaries` {player_id: DK salary}: read by P(play) v2 only. `dk_out` (player_ids DK marks
    'O'): read only with availability.REPORT_MATCH (`+rm`): a player the report does not list
    gets P(available) 0."""
    t0 = time.time()
    day = _as_date(nba_day)
    at = _aware(at)
    season = season_of(day)
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    games = _day_games(conn, day, game_ids)
    if seed is None:
        seed = stable_seed(day.isoformat(), kind, SEED_VERSION)
    p_avail, rep_extras, avinfo = report_avail(conn, day, at, games, starters)
    if AV.REPORT_MATCH and dk_out:      # +rm (default OFF): DK 'O' off the report = out
        for pid in dk_out:
            p_avail.setdefault(int(pid), 0.0)
    ex = merge_extras(extras, rep_extras)
    res = project_games(conn, day, games, ctx=ctx, bundle=b, n_sims=n_sims, seed=seed,
                        p_avail=p_avail, at=at, env_mode="live", extras=ex, starters=starters,
                        listed=listed_of(rep_extras), salaries=salaries, log=log)
    res.meta.update({"seconds_total": round(time.time() - t0, 2), "at": _iso(at),
                     "kind": kind, **avinfo})
    return res


def project_slate(conn: sqlite3.Connection, slate_id, at, *, run_id: str,
                  n_sims: int = 25_000, seed: int | None = None, kind: str | None = None,
                  ctx: Context | None = None, bundle: Bundle | None = None,
                  store: bool = True, log: Callable[[str], None] | None = None
                  ) -> ProjectionSet:
    """Projections for a DK slate as known at `at`: {DK playerId (str, = PoolPlayer
    .dk_player_id): Projection(mean, p25, p10, p50, p90)} with cond all = the P(play)-weighted
    mixture with 0. Writes player_projections / player_projection_stats, the sims npz and
    dfs_runs(run_id).sim_path when `store`. Seed default: stable_seed(slate, kind, version)."""
    from . import ids as IDS
    from .dfs import rules as R
    t0 = time.time()
    at = _aware(at)
    sid = str(slate_id)
    row = conn.execute("SELECT nba_day, game_ids_json FROM dfs_slates WHERE slate_id=?",
                       (sid,)).fetchone()
    pool = R.build_pool(conn, sid)
    if kind is None:
        k = conn.execute("SELECT kind FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
        kind = k[0] if k else "manual"
    game_ids = json.loads(row[1]) if row and row[1] else sorted(
        {p.game_id for p in pool.players if p.game_id})
    day = _as_date(row[0]) if row else _slate_day(pool)
    dk_ids = [p.dk_player_id for p in pool.players]
    dk2nba = IDS.nba_ids(conn, "dk", dk_ids)
    if len(dk2nba) < 0.9 * len(dk_ids):
        # a new slate's draftables are not mapped yet: map them BEFORE projecting (2026-10-21: the
        # first run of a new slate projected with no DK ids and the optimizer found no lineup)
        IDS.resolve_dk(conn, since=pool.captured_at, write_check=False)
        dk2nba = IDS.nba_ids(conn, "dk", dk_ids)
    extras: dict = {}
    starters: set = set()
    salaries: dict = {}
    dk_out: set = set()
    for p in pool.players:
        nid = dk2nba.get(str(p.dk_player_id))
        if nid is None or not p.game_id or not p.team:
            continue
        extras.setdefault((p.game_id, p.team), []).append(nid)
        if p.starter:
            starters.add(nid)
        if AV.REPORT_MATCH and getattr(p, "status", None) == "O":     # +rm only
            dk_out.add(nid)
        if p.salary and pool.game_type_id == CLASSIC_GAME_TYPE:
            salaries.setdefault(nid, int(p.salary))
    if seed is None:
        seed = stable_seed(sid, kind, SEED_VERSION)
    if PPLAY == "v2" and pool.game_type_id != CLASSIC_GAME_TYPE:
        salaries = classic_salaries(conn, day)          # Showdown prices are another scale
    res = project_day(conn, day, at, n_sims=n_sims, seed=seed, kind=kind, game_ids=game_ids,
                      ctx=ctx, bundle=bundle, extras=extras, starters=starters or None,
                      salaries=salaries if PPLAY == "v2" else None,
                      dk_out=dk_out if AV.REPORT_MATCH else None, log=log)
    nba2dk = {}
    for dk, nid in dk2nba.items():
        nba2dk.setdefault(nid, dk)
    res.players["dk_id"] = [nba2dk.get(int(p)) for p in res.players["player_id"]]
    res.players["slate_id"] = sid
    out = ProjectionSet(res.projections(key="dk_id"))
    out.result = res
    meta = {"run_id": run_id, "slate_id": sid, "kind": kind, "seed": seed,
            "model_version": MODEL_VERSION, "n_players": len(out),
            "dk_pool": len(pool.players),
            "dk_unprojected": sorted(set(p.dk_player_id for p in pool.players) - set(out))}
    if store:
        meta.update(save_run(conn, res, run_id=run_id, slate_id=sid, kind=kind,
                             seconds=time.time() - t0))
    meta["seconds"] = round(time.time() - t0, 2)
    meta.update(res.meta)
    out.meta = meta
    return out


CLASSIC_GAME_TYPE = 70                  # DK NBA Classic (the fringe model's salary scale)


def classic_salaries(conn: sqlite3.Connection, day: date) -> dict[int, int]:
    """{NBA player_id: DK Classic salary} over the day's Classic slates (latest capture wins);
    P(play) v2 reads it for Showdown slates, whose prices are on another scale."""
    from . import ids as IDS
    rows = conn.execute(
        "SELECT d.site_player_id, d.salary FROM dfs_salaries d JOIN dfs_slates s "
        "ON s.slate_id = d.slate_id WHERE s.nba_day=? AND s.game_type_id=? AND d.salary>0 "
        "AND d.site_player_id IS NOT NULL ORDER BY d.captured_at, d.id",
        (_as_date(day).isoformat(), CLASSIC_GAME_TYPE)).fetchall()
    if not rows:
        return {}
    dk2nba = IDS.nba_ids(conn, "dk", sorted({str(r[0]) for r in rows}))
    out: dict[int, int] = {}
    for dk, sal in rows:
        nid = dk2nba.get(str(dk))
        if nid is not None:
            out[int(nid)] = int(sal)
    return out


def save_run(conn: sqlite3.Connection, res: DayResult, *, run_id: str, slate_id: str | None,
             kind: str, seconds: float | None = None) -> dict:
    """Store a DayResult: projections + stats, the sims npz, dfs_runs, data checks."""
    pl = res.players.copy()
    proj = pd.DataFrame({
        "game_id": pl["game_id"], "nba_day": res.day.isoformat(), "slate_id": slate_id,
        "player_id": pl["player_id"].astype(np.int64), "team": pl["team"], "run_id": run_id,
        "model_version": MODEL_VERSION, "role": pl["tier"], "p_play": pl["p_play"]}, index=pl.index)
    proj["minutes_json"] = [{"mean_all": r.min_mean, "mean_played": r.min_played}
                            for r in pl.itertuples(index=False)]
    proj["inputs_json"] = [
        {"p_avail": r.p_avail, "normal": bool(r.normal), "mu_played": r.mu_played,
         "dk_id": getattr(r, "dk_id", None), "env": res.meta.get("env", {}).get(r.game_id),
         **({"rb_shift": float(getattr(r, "rb_shift", 0.0))} if RB.ROLE_BLEND else {}),
         **({"tr_shift": float(getattr(r, "tr_shift", 0.0))} if TR.TAIL_REMAP else {})}
        for r in pl.itertuples(index=False)]
    SO.write_projections(conn, proj, res.stats)
    pid, fp = res.fp_matrix()
    dk = list(pl["dk_id"]) if "dk_id" in pl else None
    gids = np.concatenate([np.full(len(g.player_id), g.game_id, dtype=object) for g in res.sims]) \
        if res.sims else []
    path = SO.save_sims(slate_id or res.day.isoformat(), run_id, pid, fp, dk_id=dk, game_id=gids,
                        extra=SO.scenario_arrays(res.sims))      # P10 scenarios
    if res.sims and all(g.stats is not None for g in res.sims):     # v1: per-stat sims too
        SO.save_stat_sims(path, pid,
                          {s: np.concatenate([g.stats[s] for g in res.sims], axis=1)
                           for s in C.COMP_STATS},
                          np.concatenate([g.minutes for g in res.sims], axis=1),
                          dk_id=dk, game_id=gids)
    SO.record_run(conn, run_id, slate_id=slate_id, kind=kind, model_version=MODEL_VERSION,
                  sim_path=str(path), seconds=seconds,
                  detail={"n_sims": res.meta.get("n_sims"), "seed": res.meta.get("seed")})
    checks = RC.check_rows(res.teams, run_id)
    if checks:
        from .dfs.rules import store_checks
        store_checks(conn, checks)
    return {"sim_path": str(path), "checks": checks}


# ============================================================================== historical

def _hist_games(ctx: Context, day: date) -> pd.DataFrame:
    tg = ctx.h.tg
    d = tg[(tg["date"] == _day_int(day)) & (tg["season_type"].isin(
        ["Regular Season", "PlayIn", "Playoffs"]))]
    return d


def _home_away(conn: sqlite3.Connection, day: date) -> pd.DataFrame:
    rows = conn.execute("SELECT game_id, team, is_home FROM team_games WHERE game_date=?",
                        (day.isoformat(),)).fetchall()
    teams: dict = {}
    for gid, team, is_home in rows:
        teams.setdefault(gid, {})[team] = int(bool(is_home))
    out = []
    for gid, t in sorted(teams.items()):
        if len(t) != 2:
            continue
        home = [k for k, v in t.items() if v]
        if len(home) == 1:
            out.append({"game_id": gid, "home": home[0],
                        "away": next(k for k in t if k != home[0])})
        else:
            # Neutral site (Cup knockouts, international games): team_games marks both teams
            # away and the nominal home team isn't stored, so the order is alphabetical and the
            # sign of a closing spread for these few games is uncertain.
            a, b = sorted(t)
            out.append({"game_id": gid, "home": a, "away": b})
    return pd.DataFrame(out, columns=["game_id", "home", "away"])


def historical_inputs(conn: sqlite3.Connection, ctx: Context, day: date, *,
                      protocol: str = "avail", at: datetime | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(games, panel, p_avail) for a past day using only data before it, except the
    availability oracle (protocol 'avail': regulars who did not play are out, §7.1 L4)."""
    games = _home_away(conn, day)
    if protocol == "avail":
        tg = ctx.h.tg[ctx.h.tg["date"] == _day_int(day)]
        t = tg[["game_id", "team", "date", "season", "margin", "n_ot"]].copy()
        t["key"] = t["game_id"] + "|" + t["team"]
        panel = M.oracle_panel(ctx.h, t)
        return games, panel.reset_index(drop=True), {}
    if protocol != "reports":
        raise ValueError(f"unknown protocol {protocol!r}")
    raise RuntimeError("use project_historical(protocol='reports')")


def project_historical(conn: sqlite3.Connection, game_date, *, protocol: str = "avail",
                       n_sims: int = 5_000, seed: int | None = None, ctx: Context | None = None,
                       bundle: Bundle | None = None, at: datetime | None = None,
                       log: Callable[[str], None] | None = None) -> DayResult:
    """Backtest projection of every game on `game_date` from data before it (closing lines as
    the market; availability per `protocol`: 'avail' oracle or 'reports' at first tip - 45)."""
    t0 = time.time()
    day = _as_date(game_date)
    season = season_of(day)
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    if seed is None:
        seed = stable_seed(day.isoformat(), f"hist-{protocol}", SEED_VERSION)
    if protocol == "reports":
        games = _home_away(conn, day)
        if at is None:
            g = AV.day_games(conn, day)
            at = (g["tip"].min() if len(g) else AV._default_tip(day)) - timedelta(minutes=45)
        at = _aware(at)
        p_avail, extras, avinfo = report_avail(conn, day, at, games)
        res = project_games(conn, day, games, ctx=ctx, bundle=b, n_sims=n_sims, seed=seed,
                            p_avail=p_avail, at=None, env_mode="closing", extras=extras,
                            listed=listed_of(extras), log=log, rs_at=at)
        res.meta.update({"at": _iso(at), **avinfo})
    else:
        games, panel, p_avail = historical_inputs(conn, ctx, day, protocol=protocol)
        env = {g: environment(conn, g, None, mode="closing") for g in games["game_id"]}
        specs = build_games(ctx, b, conn, day, games, panel, p_avail, env)
        res = run_games(specs, b, n_sims, seed, day, log)
    res.meta.update({"protocol": protocol, "seconds_total": round(time.time() - t0, 2)})
    return res


# ============================================================================== evaluation

def crps_samples(x: np.ndarray, y: float) -> float:
    """CRPS of an empirical sample x for outcome y: E|X - y| - E|X - X'|/2 (sorted formula)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    e1 = np.abs(x - y).mean()
    i = np.arange(1, n + 1)
    e2 = 2.0 * ((2 * i - n - 1) * x).sum() / (n * n)
    return float(e1 - 0.5 * e2)


def _actuals(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    a = pd.read_sql_query(
        "SELECT game_id, player_id, game_date, min, pts, fg3m, reb, ast, stl, blk, tov FROM "
        "player_games WHERE season <= ? AND season >= ?", conn,
        params=(season, _prev_season(season)))
    cats = sum((a[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    a["dk"] = (sum(C.DK_W[s] * a[s].astype(float) for s in C.DK_STATS)
               + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
    a["season"] = [season_of(date.fromisoformat(d)) for d in a["game_date"]]
    return a


def _baselines(acts: pd.DataFrame, rows: pd.DataFrame, panel_base: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date DK average (FPPG convention: last season's average before the 5th game)
    and EWMA minutes (rung-2, avail water-fill) x season-to-date DK per minute, per row
    (player_id, game_date)."""
    a = acts[acts["min"] > 0].sort_values(["player_id", "game_date"])
    out = []
    by = {pid: g for pid, g in a.groupby("player_id")}
    for r in rows.itertuples(index=False):
        g = by.get(int(r.player_id))
        std = dpm = np.nan
        if g is not None:
            prev = g[g["game_date"] < r.game_date]
            cur = prev[prev["season"] == r.season]
            last = prev[prev["season"] == _prev_season(r.season)]
            src = cur if len(cur) >= 5 else (last if len(last) else cur)
            if len(src):
                std = float(src["dk"].mean())
                dpm = float(src["dk"].sum() / max(src["min"].sum(), 1.0))
        out.append((std, dpm))
    o = np.array(out, float).reshape(-1, 2)
    res = rows.copy()
    res["base_std"] = o[:, 0]
    res["base_dpm"] = o[:, 1]
    res = res.merge(panel_base.drop_duplicates(["game_id", "player_id"]),
                    on=["game_id", "player_id"], how="left")
    res["base_ewma"] = res["base_min"].fillna(0.0) * res["base_dpm"]
    return res


def backtest(conn: sqlite3.Connection, season: str = "2024-25", *, n_dates: int = 60,
             n_sims: int = 4_000, protocol: str = "avail", ctx: Context | None = None,
             bundle: Bundle | None = None, log: Callable[[str], None] = print,
             dates: list | None = None) -> dict:
    """Quick walk-forward evaluation (fit < S) on `n_dates` regular-season dates spread evenly
    over S: DK FP MAE vs season-to-date and EWMA-minutes x season rate, CRPS, 50/80% coverage,
    calibration by tier, teammate/opponent correlation (sim vs actual)."""
    from .minutes_eval import baseline as rung2_baseline
    t0 = time.time()
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    tg = ctx.h.tg
    rs = tg[(tg["season"] == season) & (tg["season_type"] == "Regular Season")]
    all_days = sorted(rs["date"].unique())
    if dates is None:
        pick = np.unique(np.linspace(0, len(all_days) - 1, n_dates).round().astype(int))
        dates = [date(1970, 1, 1) + timedelta(days=int(all_days[i])) for i in pick]
    acts = _actuals(conn, season)
    act_key = acts.set_index(["game_id", "player_id"])
    frames, times, corr_rows = [], [], []
    for k, day in enumerate(dates):
        t1 = time.time()
        res = project_historical(conn, day, protocol=protocol, n_sims=n_sims, ctx=ctx, bundle=b)
        times.append((time.time() - t1, len(res.sims)))
        rows = []
        for g in res.sims:
            rows += _eval_rows(g, act_key, day, season)
            corr_rows.append(_sim_pair_corr(g))
        if not rows:
            continue
        d = pd.DataFrame(rows)
        # rung-2 EWMA minutes baseline (avail water-fill) on the same day's oracle panel
        _, panel, _ = historical_inputs(conn, ctx, day, protocol="avail")
        panel = panel.copy()
        panel["base_min"] = rung2_baseline(panel)
        bl = _baselines(acts, d[["game_id", "player_id", "game_date", "season"]],
                        panel[["game_id", "player_id", "base_min"]])
        for c in ("base_std", "base_ewma", "base_min"):
            d[c] = bl[c].to_numpy()
        frames.append(d)
        if log and (k % 10 == 0 or k == len(dates) - 1):
            log(f"  {day} ({k + 1}/{len(dates)}): {len(res.sims)} games in {times[-1][0]:.1f}s; "
                f"rss {peak_rss_mb():.0f} MB")
    df = pd.concat(frames, ignore_index=True)
    out = {"season": season, "protocol": protocol, "n_dates": len(dates), "n_sims": n_sims,
           "seconds": round(time.time() - t0, 1),
           "seconds_per_game": round(sum(t for t, _ in times) / max(sum(n for _, n in times), 1),
                                     3),
           "copula": json.loads(b.copula.to_json())}
    out.update(evaluate_rows(df))
    out["sim_corr"] = _pool_corr(corr_rows)
    out["actual_corr"] = _actual_pair_corr(df)
    out["frame"] = df
    return out


def _eval_rows(g, act_key: pd.DataFrame, day: date, season: str, sub: int = 2000) -> list:
    """One evaluation row per simulated player of a game (actual DK, sim quantiles, CRPS)."""
    info = g.info.reset_index(drop=True)
    sub = min(sub, g.fp.shape[0])
    rows = []
    for j in range(len(info)):
        pid = int(info.at[j, "player_id"])
        try:
            a = act_key.loc[(g.game_id, pid)]
            y, ymin = float(a["dk"]), float(a["min"] or 0.0)
        except KeyError:
            y, ymin = 0.0, 0.0
        x = g.fp[:, j].astype(float)
        mins = g.minutes[:, j]
        xp = x[mins > 0]
        q = np.quantile(x, (.1, .25, .5, .75, .9))
        r = {"game_id": g.game_id, "player_id": pid, "game_date": day.isoformat(),
             "season": season, "team": info.at[j, "team"], "tier": info.at[j, "tier"],
             "p_play": float(info.at[j, "p_play"]), "y": y, "played": ymin > 0,
             "act_min": ymin, "mean": float(x.mean()), "q10": q[0], "q25": q[1], "q50": q[2],
             "q75": q[3], "q90": q[4], "crps": crps_samples(x[:sub], y),
             "exp_min": float(mins.mean()),
             "min_played": float(mins[mins > 0].mean()) if (mins > 0).any() else 0.0,
             "pmean": np.nan}
        if len(xp) >= 50:
            qp = np.quantile(xp, (.1, .25, .75, .9))
            r.update({"pmean": float(xp.mean()), "pq10": qp[0], "pq25": qp[1], "pq75": qp[2],
                      "pq90": qp[3], "pcrps": crps_samples(xp[:sub], y)})
        rows.append(r)
    return rows


def _sim_pair_corr(g) -> dict:
    """Mean simulated DK correlation of teammate / opponent pairs (both p_play >= .9, played
    minutes >= 15) for one game."""
    info = g.info.reset_index(drop=True)
    ok = np.flatnonzero((info["p_play"] >= 0.9) & (info["min_played"] >= 15))
    if len(ok) < 4:
        return {"team": [], "opp": []}
    c = np.corrcoef(g.fp[:, ok].astype(float).T)
    team = info["team"].to_numpy()[ok]
    tm, op = [], []
    for a in range(len(ok)):
        for b_ in range(a + 1, len(ok)):
            (tm if team[a] == team[b_] else op).append(c[a, b_])
    return {"team": tm, "opp": op}


def _pool_corr(rows: list) -> dict:
    t = [x for r in rows for x in r["team"]]
    o = [x for r in rows for x in r["opp"]]
    return {"team": float(np.mean(t)) if t else None, "opp": float(np.mean(o)) if o else None,
            "n_team": len(t), "n_opp": len(o)}


def _actual_pair_corr(df: pd.DataFrame) -> dict:
    """Pooled correlation of standardised actual residuals (y - mean)/sd(sim) over the same
    kind of pairs (played, projected played minutes >= 15)."""
    d = df[(df["played"]) & (df["min_played"] >= 15) & (df["p_play"] >= 0.9)].copy()
    sd = (d["q90"] - d["q10"]) / 2.563
    d["z"] = (d["y"] - d["mean"]) / sd.clip(lower=1.0)
    d["z"] = d["z"] - d["z"].mean()
    v = float((d["z"] ** 2).mean())
    tm = op = 0.0
    nt = no = 0
    for gid, g in d.groupby("game_id"):
        z = g["z"].to_numpy()
        t = g["team"].to_numpy()
        for a in range(len(z)):
            for b_ in range(a + 1, len(z)):
                if t[a] == t[b_]:
                    tm += z[a] * z[b_]
                    nt += 1
                else:
                    op += z[a] * z[b_]
                    no += 1
    return {"team": tm / max(nt, 1) / v, "opp": op / max(no, 1) / v, "n_team": nt, "n_opp": no}


def _cl_se(d: np.ndarray, cl: np.ndarray) -> float:
    d = d - d.mean()
    s = pd.Series(d).groupby(cl).sum().to_numpy()
    return float(np.sqrt((s ** 2).sum()) / len(d))


def evaluate_rows(df: pd.DataFrame) -> dict:
    """Metrics on backtest rows: unconditional (exp minutes >= 10, DNP = 0) and conditional
    (played, projected played minutes >= 15)."""
    out: dict = {}
    u = df[(df["exp_min"] >= 10) & df["base_std"].notna()].copy()
    u["base_ewma"] = u["base_ewma"].fillna(u["base_std"])
    y = u["y"].to_numpy()
    err = {"model_mean": np.abs(u["mean"] - y), "model_p50": np.abs(u["q50"] - y),
           "season_to_date": np.abs(u["base_std"] - y),
           "ewma_min_x_season_rate": np.abs(u["base_ewma"] - y)}
    gid = u["game_id"].to_numpy()
    out["uncond"] = {"n": int(len(u)), "dnp_rate": float((~u["played"]).mean()),
                     "mae": {k: float(v.mean()) for k, v in err.items()},
                     "crps_model": float(u["crps"].mean()),
                     "gain_vs_std_pct": float((1 - err["model_mean"].mean()
                                               / err["season_to_date"].mean()) * 100),
                     "gain_vs_std_se_pct": _cl_se((err["model_mean"] - err["season_to_date"])
                                                  .to_numpy(), gid)
                     / err["season_to_date"].mean() * 100,
                     "gain_vs_ewma_pct": float((1 - err["model_mean"].mean()
                                                / err["ewma_min_x_season_rate"].mean()) * 100),
                     "cov80": float(((u["q10"] <= y) & (y <= u["q90"])).mean()),
                     "cov50": float(((u["q25"] <= y) & (y <= u["q75"])).mean()),
                     "bias": float((u["mean"] - y).mean())}
    c = df[df["played"] & (df["min_played"] >= 15) & df["pmean"].notna()]
    yc = c["y"].to_numpy()
    out["cond_played"] = {"n": int(len(c)), "mae": float(np.abs(c["pmean"] - yc).mean()),
                          "crps": float(c["pcrps"].mean()),
                          "cov80": float(((c["pq10"] <= yc) & (yc <= c["pq90"])).mean()),
                          "cov50": float(((c["pq25"] <= yc) & (yc <= c["pq75"])).mean()),
                          "bias": float((c["pmean"] - yc).mean())}
    tiers = {}
    u2 = u.copy()
    u2["lvl"] = pd.cut(u2["mean"], [-1, 10, 20, 30, 40, 200],
                       labels=["<10", "10-20", "20-30", "30-40", "40+"])
    for name, g in list(u2.groupby("tier")) + list(u2.groupby("lvl", observed=True)):
        yy = g["y"].to_numpy()
        tiers[str(name)] = {"n": int(len(g)), "pred": float(g["mean"].mean()),
                            "actual": float(yy.mean()),
                            "mae": float(np.abs(g["mean"] - yy).mean()),
                            "mae_std": float(np.abs(g["base_std"] - yy).mean()),
                            "cov80": float(((g["q10"] <= yy) & (yy <= g["q90"])).mean()),
                            "cov50": float(((g["q25"] <= yy) & (yy <= g["q75"])).mean())}
    out["calibration"] = tiers
    return out


def format_backtest(res: dict) -> str:
    u, c = res["uncond"], res["cond_played"]
    lines = [f"walk-forward {res['season']} ({res['protocol']}), {res['n_dates']} dates, "
             f"{res['n_sims']} sims; {res['seconds']}s ({res['seconds_per_game']}s/game)",
             f"unconditional (exp min >= 10, DNP = 0): n={u['n']} dnp={u['dnp_rate']:.3f}",
             "  MAE " + ", ".join(f"{k} {v:.3f}" for k, v in u["mae"].items()),
             f"  gain vs season-to-date {u['gain_vs_std_pct']:+.2f}% "
             f"(SE {u['gain_vs_std_se_pct']:.2f}), vs EWMA x rate {u['gain_vs_ewma_pct']:+.2f}%",
             f"  CRPS {u['crps_model']:.3f}  cov80 {u['cov80']:.3f}  cov50 {u['cov50']:.3f}  "
             f"bias {u['bias']:+.3f}",
             f"conditional (played, proj min >= 15): n={c['n']} MAE {c['mae']:.3f} CRPS "
             f"{c['crps']:.3f} cov80 {c['cov80']:.3f} cov50 {c['cov50']:.3f} bias {c['bias']:+.3f}",
             "calibration (unconditional):"]
    for k, v in res["calibration"].items():
        lines.append(f"  {k:<6} n={v['n']:<5} pred {v['pred']:6.2f} actual {v['actual']:6.2f} "
                     f"MAE {v['mae']:.2f} (std {v['mae_std']:.2f}) cov80 {v['cov80']:.3f} "
                     f"cov50 {v['cov50']:.3f}")
    sc, ac = res["sim_corr"], res["actual_corr"]
    lines.append(f"pair corr teammates sim {sc['team']:.3f} vs actual {ac['team']:.3f}; "
                 f"opponents sim {sc['opp']:.3f} vs actual {ac['opp']:.3f}")
    return "\n".join(lines)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
