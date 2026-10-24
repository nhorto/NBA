"""GPP portfolio: the contest builder's N-entry lineup set (docs/PLAYERS.md §2.3, §5.5; task T2.12).

**Everything here is simulated against a modelled field** (`field.py`: a PROXY of ownership and
of the other entries). Expected payout, ROI and P(any cash) are "simulated, not real results".

Pipeline (`build_portfolio`):

1. **Candidates** (§5.5.1, S11): CP-SAT solves (`optimizer.optimize`, dominated players pruned as
   in `field._opt_lineups`) on a thread pool, deduplicated:
   * `n_scenario` scenario objectives  w·mu_i + (1 - w)·FP_{i,s} for a random sim s (w = 0.5);
   * `n_ownpen` ownership-penalised     mu_i - lambda·own_i,  lambda ~ U(0, 0.5·sd(mu)) (own = proxy);
   * `n_noise` noise objectives         mu_i·exp(N(0, 0.12²));
   * the Top-`n_topmean` by mean (`topn.top_n`, min different 1);
   * per player with a user minimum exposure, a few scenario solves with him locked.
   Counts scale with N above 50 (`Settings.scale`). Locks are forced and excludes banned in every
   solve; players whose game has started are banned (late builds).
2. **Field proxy** (`build_field` at the contest's archetype; strengths soft / medium / sharp
   differ by how informed the field's projection is, `STRENGTH_W`) and a calibrated **scoring
   truth** (`truth_sims`: our sims shifted toward the public number by TRUTH_BETA and widened by a
   per-sim model error TRUTH_TAU, both from the T2.1 harness residuals), then **scoring** (`field.evaluate(per_sim=True, own_entries=N)`): per sim and
   candidate the score, the number of the M - N other entries above it and tied with it, plus the
   tier lines. Expected dupes the field sample does not already contain (the product-form part of
   `field.dupes`) are added to the ties as a Poisson count per sim, so dupes split the prize.
3. **Greedy selection** (§5.5.5): add the candidate with the largest **marginal expected payout of
   the whole portfolio**. Our entries occupy ranks jointly: in sim s, entry j finishes at rank
   above_j + (our entries above j) + 1 and splits the ladder over its ties, so a new entry
   that beats one of ours pushes it one place down (the loss is counted in the marginal).
   Constraints, all exact:
   * overlap: at most `gamma` shared players with every chosen lineup (default 5 of 8; at least
     the number of locks);
   * exposure: a player appears in at most floor(max·N) lineups (default max 60 %, applied from
     N >= 5 so 2-4 entry sets are not forced apart; per-player max overrides it, 0 = exclude;
     a per-player minimum above the default cap raises that player's cap),
     at least ceil(min·N) (per-player min; a lineup is only allowed if every unmet minimum can
     still be met in the remaining picks); locked players are in every lineup.
   If the candidate pool runs dry, up to `refills` batches of new scenario solves are made under
   the constraints (CP-SAT no-good cuts Σ_{i∈L} x_i <= gamma per chosen L, capped players
   banned) and scored against the same field.
4. **Output** (`Portfolio`): per lineup P(cash), P(top 1 %), E[payout] (joint, so they add up to
   the portfolio), dupes and band; per portfolio E[payout] ± SE, ROI, P(any cash), P(profit), the
   **relative score** (E[payout] / that of the Top-N-by-projection set on the same sims and
   field) and the exposure table; optionally E[payout] under the soft and sharp fields.
   **Display policy:** until `calibration_slates` >= 10 (uploaded standings), pages and the CLI
   lead with the relative score and probability bands (`prob_band`); dollar E[payout] and ROI
   are "details (simulated, uncalibrated)" only.
5. **Entries curve** (`entries_curve`, `summary["curve"]`): chances that at least one entry is in
   the money / top 1 % / top 0.1 % / 1st, and the fees, for N = 1, 3, 5, 10, 20, 50, 150 (up to
   the contest's max entries) on the SAME sims, so entries that bust together count once.
   Points are nested sets under their own N's caps (smaller: subsets of the build; larger: the
   build plus more greedy picks). E[payout] per dollar is stored but never headlined.

**Sim split** (`COURTSIDE_PORTFOLIO_SPLIT`, OFF by default; see SPLIT_ENV): steps 1, 3 and 4/5
run on disjoint rows of the scoring truth (generation G / selection S / report R), so the
reported chances are not scored on the sims that generated and picked the lineups.

**Selection objective** (`COURTSIDE_PORTFOLIO_OBJ`, OFF by default; see OBJ_ENV, D9): the greedy
picks on a ladder with the top prizes smoothed ('flat' / 'blend'); the report uses the real one.
The entries curve's points carry simulation SEs (`curve_se`, display only).

Storage (migration 13): `dfs_portfolios` (one row per build, status building | ok | error,
settings and summary JSON) and the lineups in `dfs_lineups` with ``run_id`` = the portfolio id
(``pf-...``), ``mode='gpp'`` and the contest id. They are kept out of `dfs_runs` on purpose: the
runner, grading and monitoring read every run of a slate, and a portfolio is not a projection run.

CLI: ``courtside dfs portfolio --slate ID --contest ID --n 20 [--gamma 5] [--max-exp 0.6]
[--lock NAME_OR_ID ...] [--exclude ...] [--min PLAYER=0.3 ...] [--max PLAYER=0.4 ...] [--no-save]``.
No DraftKings request is made here; the CSV is uploaded by hand (`export.to_csv`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field as dc_field, replace as dc_replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import role_blend as RB
from . import field as F
from .rules import PoolPlayer, Rules, SlatePool

log = logging.getLogger(__name__)

VERSION = "gpp-portfolio-1.0"
LABEL = "simulated against a modelled field, not real results"
MODE = "gpp"
ID_PREFIX = "pf-"
MIN_PROJ = 2.0                 # players projected under 2 DK points are never rostered
MAX_ENTRIES = 150              # §2.3 cap
STALE_BUILD_S = 15 * 60        # a 'building' row older than this was interrupted
MIN_CAL_SLATES = F.MIN_FIT_SLATES   # dollar E[payout] / ROI headlines only from 10 uploaded slates

# Scoring "truth" (calibration, docs/PLAYERS.md §0.4): our sims are centred on our projection,
# but the public number carries information ours lacks. On the T2.1 harness rows (2019-20,
# 2020-21, ~46k player-games) the regression of (actual - ours) on (FPPG - ours) gives
# beta = 0.16 +/- 0.02 and 0.20 +/- 0.01; the residual sd (9.6) exceeds the sims' sd (9.45) by
# about 0.18 of a player's sd. Each sim is scored on
#     truth_{s,i} = fp_{s,i} + 1[plays] * (beta * (fppg_i - mu_played_i) + tau * sd_i * z_{s,i}).
TRUTH_BETA = 0.18
TRUTH_TAU = 0.18
# Field strength = how informed the field's projection is: the share of ours in its public blend
# (default 0.5) moves -0.25 / 0 / +0.25; the optimizer-built share stays at the archetype's
# default (field.ARCHETYPES "mix": 40 % cash / small GPP, 20 % large GPP since field-proxy-1.1).
# The first real standings (2026-10-20, docs/research/real-standings-2026-10-20.md) put the
# medium field's cash line at the ACTUAL box scores within the real contests' spread (double-ups
# 248 vs 237-248 real, GPP 253 vs 253), so medium stays the headline.
# PLAYERS.md S12's mix-based strengths (20/80, 40/60, 70/30) do not order: the optimizer part is
# resampled from ~200 solves, so a 70 % optimizer field had 4.4k unique lineups vs 11.3k soft, a
# lower top tail, and scored as the EASIEST field for a fixed set (soft 748 / medium 563 / sharp
# 658 on the 2026-10-20 $60K Zone); by information it orders 880 / 563 / 416.
STRENGTH_W = {"soft": -0.25, "medium": 0.0, "sharp": 0.25}
# Sim split (IMPROVEMENTS.md B1, docs/research/portfolio-split.md). OFF by default (env
# COURTSIDE_PORTFOLIO_SPLIT=1 / 0 overrides, read at import; `set_split` or
# `build_portfolio(split=...)` at run time). OFF, one set of sim rows generates the scenario
# candidates, selects (greedy) and reports, so scenario lineups are scored on "their" own row and
# the max over ~350 candidates is reported on the sims that picked it (winner's curse). ON, the
# scoring truth's rows are split at random (seeded by Settings.seed) into three disjoint sets:
#   G = SPLIT_G of the rows (2,500 of 25,000): scenario / min-locked / refill objectives;
#   S = half of the rest (11,250): candidate scoring for the greedy and the curve's picks;
#   R = the other half (11,250): everything reported (E[payout], P(cash), P(top 1 %), dupes,
#       relative score, entries curve, strengths, baselines).
# Version `gpp-portfolio-1.0+split` in info.version. OFF: nothing of it runs (bit-identical,
# tests/test_portfolio_split.py).
# Mode "gs" (COURTSIDE_PORTFOLIO_SPLIT=gs, version `+split-gs`): a report-only split. G and S are
# ONE set (the first 1 - SPLIT_R of the same permutation, 13,750 of 25,000), generation draws
# from it uniformly with replacement as OFF does, the greedy selects on it, and R (the same
# 11,250 rows as mode "split") reports. Same R share as "split" so the report noise is unchanged.
SPLIT_DEFAULT = "gs"                   # shipped 2026-10-23 (owner OK); "off" | "split" | "gs" (env overrides)
SPLIT_ENV = "COURTSIDE_PORTFOLIO_SPLIT"
SPLIT_TAG = "+split"
SPLIT_GS_TAG = "+split-gs"
SPLIT_MODES = ("off", "split", "gs")
SPLIT_G = 0.10                         # share of the rows for generation (G)
SPLIT_R = 0.45                         # share of the rows for the report (R); S = the rest
SPLIT_SEED = 11                        # row permutation: default_rng(Settings.seed + SPLIT_SEED)


def split_mode(v: Any, default: str = "off") -> str:
    """'off' | 'split' | 'gs' from an env string / bool / mode: 1 true yes on split -> 'split',
    gs -> 'gs', 0 false no off -> 'off', anything else -> `default`."""
    if v is None:
        return default
    if isinstance(v, bool):
        return "split" if v else "off"
    x = str(v).strip().lower()
    if x in ("1", "true", "yes", "on", "split"):
        return "split"
    if x == "gs":
        return "gs"
    if x in ("0", "false", "no", "off", ""):
        return "off" if x else default
    return default


PORTFOLIO_SPLIT = split_mode(os.environ.get(SPLIT_ENV), SPLIT_DEFAULT)

# Selection objective (IMPROVEMENTS.md D9, docs/research/d8-d9.md). OFF by default (env
# COURTSIDE_PORTFOLIO_OBJ, read at import; `set_objective` or `build_portfolio(objective=...)`).
# In a big GPP P(1st) per lineup is ~1e-4, so 25k sims hold a handful of 1st places and the
# greedy's marginal E[payout] is driven by them: picks move with the sim seed. ON, the greedy
# (and the entries curve's picks) score a SMOOTHED ladder; everything reported still uses the
# real one. Both keep the ladder's total money:
#   "flat"  : the top 0.1 % of ranks (max(1, ceil(0.001 M)), the curve's top-0.1 % rank) all pay
#             the mean of their prizes (E[payout] with the top prizes flattened);
#   "blend" : ranks in the top 1 % pay (1 - OBJ_BLEND_W) x prize + OBJ_BLEND_W x the mean top-1 %
#             prize (E[payout] blended with P(top 1 %) priced at the mean top-1 % prize).
# Version tag `+obj-flat` / `+obj-blend` on info.version. OFF: the greedy gets the real ladder
# object itself (bit-identical).
OBJ_ENV = "COURTSIDE_PORTFOLIO_OBJ"
OBJ_DEFAULT = "blend"                  # shipped 2026-10-24 (owner OK; env still overrides)
OBJ_MODES = ("off", "flat", "blend")
OBJ_TAG = "+obj-"
OBJ_BLEND_W = 0.5


def objective_mode(v: Any, default: str = OBJ_DEFAULT) -> str:
    """'off' | 'flat' | 'blend' from an env string (0 / false / no / off / '' -> 'off'; 1 / on ->
    'flat'); anything else -> `default`."""
    if v is None:
        return default
    x = str(v).strip().lower()
    if x in OBJ_MODES:
        return x
    if x in ("1", "true", "yes", "on"):
        return "flat"
    if x in ("0", "false", "no", ""):
        return "off"
    return default


PORTFOLIO_OBJ = objective_mode(os.environ.get(OBJ_ENV), OBJ_DEFAULT)


def set_objective(mode: Any) -> None:
    """Set the selection objective for later builds (the env is read once at import)."""
    global PORTFOLIO_OBJ
    PORTFOLIO_OBJ = objective_mode(mode)


def set_split(on: bool | str) -> None:
    """Set the sim split mode for later builds (the env is read once at import): True / 'split',
    'gs', or False / 'off'."""
    global PORTFOLIO_SPLIT
    PORTFOLIO_SPLIT = split_mode(on)


def split_rows(S: int, seed: int, mode: str = "split") -> dict[str, np.ndarray]:
    """Disjoint sorted row indices {G, S, R} of an S-row scoring truth (see SPLIT_G): a seeded
    permutation, G = the first round(SPLIT_G * S) rows, R = the last round(SPLIT_R * S), S the
    rest in between (each at least 1). mode 'gs': G = S = everything but R (same R rows)."""
    if S < 3:
        raise PortfolioError(f"the sim split needs at least 3 sims (have {S})")
    perm = np.random.default_rng(int(seed) + SPLIT_SEED).permutation(S)
    g = min(S - 2, max(1, int(round(SPLIT_G * S))))
    r = min(S - g - 1, max(1, int(round(SPLIT_R * S))))
    if mode == "gs":
        gs = np.sort(perm[:S - r])
        return {"G": gs, "S": gs, "R": np.sort(perm[S - r:])}
    return {"G": np.sort(perm[:g]), "S": np.sort(perm[g:S - r]), "R": np.sort(perm[S - r:])}


# ============================================================================== settings

@dataclass
class Settings:
    n: int = 20
    gamma: int = 5                         # max shared players with every chosen lineup
    max_exposure: float | None = 0.60      # default per-player cap (fraction of N)
    exposure_from: int = 5                 # ... applied from N >= this
    min_exp: dict = dc_field(default_factory=dict)    # {dk id: fraction}
    max_exp: dict = dc_field(default_factory=dict)    # {dk id: fraction}
    locks: tuple = ()
    excludes: tuple = ()
    n_scenario: int = 150
    n_ownpen: int = 100
    n_noise: int = 50
    n_topmean: int = 50
    n_min_locked: int = 8                  # scenario solves per min-exposure player
    scen_w: float = 0.5
    noise_sd: float = 0.12
    ownpen_max: float = 0.5                # lambda ~ U(0, ownpen_max * sd(mu))
    n_sims: int | None = 25_000
    n_field: int | None = None             # field sample (default: field.build_field's)
    field_solves: int = F.N_SOLVES
    mix: Any = "medium"
    threads: int = 2
    solve_limit_s: float = 1.0
    refills: int = 3
    strengths: bool = False                # also soft / sharp E[payout]
    strength_sims: int = 10_000
    beta: float = None                     # truth shift toward the public number (TRUTH_BETA)
    tau: float = None                      # per-sim model error, x player sd (TRUTH_TAU)
    seed: int = 0
    curve: bool = True                     # the "how many entries" curve (entries_curve)
    curve_to: int | None = None            # its largest entry count (None: up to N only)
    curve_refills: int = 2                 # refill batches for curve points above N
    curve_select_sims: int | None = 8_000  # sims used to PICK curve lineups above N (all score)

    def __post_init__(self):
        if self.beta is None:
            self.beta = TRUTH_BETA
        if self.tau is None:
            self.tau = TRUTH_TAU

    @property
    def scale(self) -> float:
        return max(1.0, self.n / 50.0)

    def counts(self) -> dict:
        k = self.scale
        return {"scenario": int(round(self.n_scenario * k)), "ownpen": int(round(self.n_ownpen * k)),
                "noise": int(round(self.n_noise * k)), "topmean": max(self.n_topmean, self.n)}

    def as_json(self) -> dict:
        d = asdict(self)
        d["locks"], d["excludes"] = list(self.locks), list(self.excludes)
        return d


class PortfolioError(ValueError):
    """Settings that cannot produce a portfolio (bad caps, unknown player, no sims...)."""


def exposure_limits(s: Settings, pids: Iterable[str]) -> tuple[dict[str, int], dict[str, int]]:
    """({pid: max lineups}, {pid: min lineups}) for N = s.n (exact integer caps)."""
    n = int(s.n)
    locks = set(s.locks)
    cap: dict[str, int] = {}
    low: dict[str, int] = {}
    for p in pids:
        if p in locks:
            cap[p] = n
        elif p in s.max_exp:
            cap[p] = int(math.floor(float(s.max_exp[p]) * n + 1e-9))
        elif s.max_exposure is not None and n >= s.exposure_from:
            cap[p] = max(1, int(math.floor(float(s.max_exposure) * n + 1e-9)))
        else:
            cap[p] = n
        lo = n if p in locks else int(math.ceil(float(s.min_exp.get(p, 0.0)) * n - 1e-9))
        if p not in s.max_exp:            # a player's own minimum overrides the default cap
            cap[p] = max(cap[p], lo)
        if lo > cap[p]:
            raise PortfolioError(f"player {p}: minimum exposure ({lo} of {n}) is above the "
                                 f"maximum ({cap[p]} of {n})")
        if lo:
            low[p] = lo
    return cap, low


# ============================================================================== candidates

def _identity(lu, showdown: bool) -> tuple:
    pairs = lu.slot_pairs()
    if showdown:
        cpt = [p for s, p in pairs if s == "CPT"]
        return tuple(cpt) + tuple(sorted(p for s, p in pairs if s != "CPT"))
    return tuple(sorted(p for _, p in pairs))


def _solver(players: Sequence[PoolPlayer], rules: Rules, *, locks=(), bans=(), now=None,
            limit: float = 1.0, cuts: Sequence = (), gamma: int = 8):
    """solve(score dict, extra_locks) -> Lineup | None, pruning dominated players first."""
    from . import optimizer as O
    by_id = {p.dk_player_id: p for p in players}
    names = rules.distinct_slot_names
    prune_ok = len(set(rules.slot_multiplier(s) for s in names)) == 1 and not cuts

    def solve(scores: Mapping[str, float], extra_locks=()) -> Any:
        lk = set(locks) | set(extra_locks)
        ids = [i for i in scores if i in by_id and i not in bans]
        if prune_ok and len(ids) > rules.n_slots:
            v = np.array([scores[i] for i in ids])
            sal = np.array([by_id[i].salary for i in ids], float)
            masks = np.array([sum(1 << names.index(s) for s in by_id[i].slots if s in names)
                              for i in ids], np.int64)
            keep = F.undominated(v, sal, masks, rules.n_slots)
            ids = [i for j, i in enumerate(ids) if keep[j] or i in lk]
        sc = {i: round(float(scores[i]), 3) for i in ids}
        if any(i not in sc for i in lk):
            return None
        try:
            model = O.model_for([by_id[i] for i in ids], sc, rules, locks=sorted(lk),
                                now=now, time_limit_s=limit)
        except ValueError:
            return None
        for lu in cuts:
            model.add_no_good(lu, rules.n_slots - gamma)
        return model.solve()

    return solve


def candidates(pool_players: Sequence[PoolPlayer], rules: Rules, mu: Mapping[str, float],
               fp: np.ndarray, fp_col: Mapping[str, int], own: Mapping[str, float],
               s: Settings, *, now: datetime | None = None, bans: Iterable[str] = (),
               rng: np.random.Generator, tags: list | None = None,
               draw: Callable[[], int] | None = None, lev: Mapping[str, float] | None = None,
               lev_rng: np.random.Generator | None = None) -> tuple[list, dict]:
    """Unique candidate lineups (`optimizer.Lineup`) and per-source counts (§5.5.1). `tags`
    (a list) receives each returned lineup's source, in order; `draw()` gives the sim row of
    each scenario objective (default: uniform over `fp`'s rows from `rng`). `lev` (xf-v2 ON
    only, {DK id: shrunk top-1 % leverage}, `xfactor.builder_leverage`): also
    round(xfactor.LEV_N_SOLVES x scale) leverage-tilted SCENARIO solves
    w mu_i + (1 - w) FP_{i,s} + lambda clip(log lev_i), s a uniform row of `fp`, lambda ~
    U(LEV_TILT_MIN, LEV_TILT_MAX) sd(mu), both from `lev_rng`, appended after every other job (so
    the other candidates and their order are unchanged). A tilt of mu alone moves every solve
    along one direction and collapses to a few lineups (like the ownpen solves, which add ~0
    unique candidates on real slates); the scenario row makes each solve different."""
    from . import topn as T
    t0 = time.perf_counter()
    bans = set(bans)
    ids = [p for p in mu if p not in bans]
    if len(ids) < rules.n_slots:
        raise PortfolioError("too few projected players for a lineup")
    base = np.array([mu[i] for i in ids], float)
    sd_mu = float(np.std(base[base >= 8])) if (base >= 8).sum() > 3 else float(np.std(base))
    cols = np.array([fp_col.get(i, -1) for i in ids])
    o = np.array([float(own.get(i, 0.0)) for i in ids])
    c = s.counts()
    jobs: list[tuple[str, dict, tuple]] = []
    S = fp.shape[0]
    draw = draw or (lambda: int(rng.integers(0, S)))
    for _ in range(c["scenario"]):
        r = draw()
        x = np.where(cols >= 0, fp[r, np.maximum(cols, 0)].astype(float), base)
        jobs.append(("scenario", dict(zip(ids, s.scen_w * base + (1 - s.scen_w) * x)), ()))
    for _ in range(c["ownpen"]):
        lam = float(rng.uniform(0, s.ownpen_max * sd_mu))
        jobs.append(("ownpen", dict(zip(ids, base - lam * o)), ()))
    for _ in range(c["noise"]):
        jobs.append(("noise", dict(zip(ids, base * np.exp(rng.normal(0, s.noise_sd, len(ids))))),
                     ()))
    for p, f in s.min_exp.items():
        if f > 0 and p in mu and p not in s.locks:
            for _ in range(s.n_min_locked):
                r = draw()
                x = np.where(cols >= 0, fp[r, np.maximum(cols, 0)].astype(float), base)
                jobs.append(("min_locked", dict(zip(ids, s.scen_w * base + (1 - s.scen_w) * x)),
                             (p,)))
    if lev:
        from . import xfactor as XF
        lr = lev_rng if lev_rng is not None else np.random.default_rng(s.seed + XF.LEV_SEED)
        for _ in range(int(round(XF.LEV_N_SOLVES * s.scale))):
            r = int(lr.integers(0, S))
            x = np.where(cols >= 0, fp[r, np.maximum(cols, 0)].astype(float), base)
            lam = float(lr.uniform(XF.LEV_TILT_MIN, XF.LEV_TILT_MAX) * sd_mu)
            jobs.append(("levtilt", XF.tilt_scores(ids, s.scen_w * base + (1 - s.scen_w) * x,
                                                   lev, lam), ()))
    players = [p for p in pool_players if p.dk_player_id in mu and p.dk_player_id not in bans]
    solve = _solver(players, rules, locks=s.locks, bans=bans, now=now, limit=s.solve_limit_s)

    def top_job():
        if not c["topmean"]:
            return []
        try:
            r = T.top_n(players, {i: mu[i] for i in ids}, rules, c["topmean"], min_diff=1,
                        locks=list(s.locks), now=now, time_limit_s=s.solve_limit_s * 2,
                        total_time_limit_s=max(10.0, 0.5 * c["topmean"]))
        except ValueError:
            return []
        return [x.lineup for x in r.lineups]

    out: list = []
    src: dict[str, int] = {}
    seen: set = set()
    showdown = rules.is_showdown

    def add(tag, lu):
        if lu is None:
            return
        k = _identity(lu, showdown)
        if k in seen:
            return
        seen.add(k)
        out.append(lu)
        src[tag] = src.get(tag, 0) + 1
        if tags is not None:
            tags.append(tag)

    with ThreadPoolExecutor(max(1, s.threads)) as ex:
        tf = ex.submit(top_job)
        res = list(ex.map(lambda j: (j[0], solve(j[1], j[2])), jobs))
        for lu in tf.result():
            add("topmean", lu)
    for tag, lu in res:
        add(tag, lu)
    info = {"solves": len(jobs) + c["topmean"], "unique": len(out), "by_source": src,
            "seconds": round(time.perf_counter() - t0, 2)}
    return out, info


# ============================================================================== joint payout

class Ladder:
    """Payout by rank with tie splitting: g(lo, t) = mean payout over ranks lo..lo+t."""

    def __init__(self, contest: F.Contest):
        M = int(contest.field_size)
        pay = np.zeros(M + 2)
        pay[1:M + 1] = contest.ladder()[1:M + 1]
        self.top = M + 1
        self.cum = np.concatenate([[0.0], np.cumsum(pay[1:])])    # cum[r] = Σ_{1..r}
        self.paid = max(1, min(contest.paid, M))
        self.top1 = max(1, int(math.ceil(0.01 * M)))
        self.M = M

    def g(self, lo, t):
        lo = np.clip(lo, 1, self.top)
        hi = np.minimum(lo + t, self.top)
        return (self.cum[hi] - self.cum[lo - 1]) / (t + 1)

    def pay(self) -> np.ndarray:
        """Payout by rank 1..M + 1 (index 0 = rank 1; the last rank pays 0)."""
        return np.diff(self.cum)


def selection_ladder(lad: Ladder, mode: str) -> Ladder:
    """The ladder the greedy scores under the D9 selection objective (OBJ_ENV): `lad` itself
    when 'off', else a copy whose top ranks are smoothed (total money unchanged)."""
    if mode == "off":
        return lad
    pay = lad.pay()
    if mode == "flat":
        k = min(max(1, int(math.ceil(0.001 * lad.M))), lad.paid)
        pay[:k] = pay[:k].mean()
    elif mode == "blend":
        k = min(lad.top1, lad.paid)
        pay[:k] = (1 - OBJ_BLEND_W) * pay[:k] + OBJ_BLEND_W * pay[:k].mean()
    else:
        raise PortfolioError(f"unknown selection objective {mode!r}")
    out = Ladder.__new__(Ladder)
    out.__dict__.update(lad.__dict__)
    out.cum = np.concatenate([[0.0], np.cumsum(pay)])
    out.mode = mode
    return out


def joint_ranks(score: np.ndarray, above: np.ndarray) -> np.ndarray:
    """(S x k) finishing rank of each of our k entries: other entries above + our entries above +
    1. Equal scores among ours: the later entry ranks first (it does not change the total)."""
    S, k = score.shape
    key = score.astype(np.int64) * 4096 + np.arange(k)
    order = np.argsort(-key, axis=1, kind="stable")
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(k), (S, k)), axis=1)
    return above.astype(np.int64) + pos + 1


def joint_payouts(lad: Ladder, score, above, tied) -> tuple[np.ndarray, np.ndarray]:
    """(payout (S x k), rank (S x k)) of a set of our entries."""
    r = joint_ranks(score, above)
    return lad.g(r, tied.astype(np.int64)), r


@dataclass
class _Pool:
    """Per-sim candidate inputs: score / above / tied (S x C, int) and the live (sim, cand)
    pairs whose entry can still be paid (above < paid)."""
    score: np.ndarray
    above: np.ndarray
    tied: np.ndarray

    def live(self, paid: int):
        s, c = np.nonzero(self.above < paid)
        return (s.astype(np.int32), c.astype(np.int32), self.above[s, c].astype(np.int64),
                self.tied[s, c].astype(np.int64), self.score[s, c])


def _incidence(lineups: Sequence, pidx: Mapping[str, int]) -> np.ndarray:
    inc = np.zeros((len(lineups), len(pidx)), np.int16)
    for r, lu in enumerate(lineups):
        for _, p in lu.slot_pairs():
            inc[r, pidx[p]] = 1
    return inc


@dataclass
class CandSet:
    """Candidates as the greedy sees them: per-sim inputs `P` and the (C x players) incidence
    `inc`. `on_empty` callbacks may replace both (refills). `R`: the same candidates scored on
    the report rows (sim split ON, see SPLIT_ENV); None = report on `P`."""
    P: _Pool
    inc: np.ndarray
    R: _Pool | None = None

    @property
    def report(self) -> _Pool:
        return self.P if self.R is None else self.R


def greedy(lad: Ladder, cs: CandSet, n: int, *, gamma: int,
           cap: np.ndarray, low: np.ndarray, start: Sequence[int] = (),
           on_empty: Callable[[list[int], np.ndarray], bool] | None = None,
           objective: np.ndarray | None = None) -> tuple[list[int], dict]:
    """Greedy portfolio: indices of chosen candidates (in pick order) and info.

    Marginal of candidate c given chosen set P (per sim, then averaged):
        g(above_c + A_c + 1, tied_c) - Σ_{j ∈ P below c} [g(r_j, t_j) - g(r_j + 1, t_j)]
    with A_c = our entries strictly above c. `objective` (C,) replaces the marginal by a fixed
    score (the "top N by individual E[payout]" baseline, same constraints). `on_empty(chosen,
    counts)` may add candidates (returns True) when none is allowed."""
    S = cs.P.score.shape[0]
    chosen: list[int] = list(start)
    counts = np.zeros(cs.inc.shape[1], np.int64)
    marg_trace = []
    stopped = None
    ls = lc = la = lt = lx = lA = None

    def rebuild():
        nonlocal ls, lc, la, lt, lx, lA
        ls, lc, la, lt, lx = cs.P.live(lad.paid)
        lA = np.zeros(len(ls), np.int64)
        for j in chosen:
            lA += cs.P.score[ls, j] > lx

    rebuild()
    for j in chosen:
        counts += cs.inc[j]
    while len(chosen) < n:
        P, inc = cs.P, cs.inc
        C = inc.shape[0]
        k = len(chosen)
        rem = n - k
        allowed = np.ones(C, bool)
        allowed[chosen] = False
        full = counts >= cap
        if full.any():
            allowed &= ~(inc[:, full] > 0).any(1)
        if k:
            shared = inc.astype(np.int32) @ inc[chosen].T.astype(np.int32)
            allowed &= (shared <= gamma).all(1)
        need = low - counts
        urgent = need >= rem
        if urgent.any():
            allowed &= (inc[:, urgent] > 0).all(1)
        if not allowed.any():
            if on_empty is not None and on_empty(chosen, counts):
                rebuild()
                continue
            stopped = "exhausted"
            break
        if objective is not None:
            em = np.where(allowed, objective[:C], -np.inf)
        else:
            if k:
                sc, ab, td = P.score[:, chosen], P.above[:, chosen], P.tied[:, chosen]
                key = sc.astype(np.int64) * 4096 + np.arange(k)
                order = np.argsort(-key, axis=1, kind="stable")
                ab_s = np.take_along_axis(ab, order, 1).astype(np.int64)
                td_s = np.take_along_axis(td, order, 1).astype(np.int64)
                r = ab_s + np.arange(k) + 1
                D = lad.g(r, td_s) - lad.g(r + 1, td_s)
                bsuf = np.zeros((S, k + 1))
                bsuf[:, :k] = np.cumsum(D[:, ::-1], axis=1)[:, ::-1]
                loss = bsuf[ls, lA]
            else:
                loss = 0.0
            m = lad.g(la + lA + 1, lt) - loss
            em = np.bincount(lc, weights=m, minlength=C)[:C] / S
            em = np.where(allowed, em, -np.inf)
        e = int(np.argmax(em))
        marg_trace.append(float(em[e]))
        chosen.append(e)
        counts += inc[e]
        lA += P.score[ls, e] > lx
    return chosen, {"stopped": stopped, "marginals": marg_trace}


def truth_sims(fp: np.ndarray, pids: Sequence[str], inputs: pd.DataFrame, beta: float,
               tau: float, rng: np.random.Generator, *, blended: bool | None = None
               ) -> tuple[np.ndarray, dict]:
    """Per-sim DK points to score on (see TRUTH_BETA): our sims shifted toward the public
    number by `beta` and widened by a per-sim model error `tau` x the player's sd, only in sims
    where the player plays. Players without a public number (or out) are left as they are.
    `blended` (None: the process switch `role_blend.ROLE_BLEND`): the sims already carry the
    role-aware blend (+rb, IMPROVEMENTS.md P1 / D1), so the flat beta shift is skipped (beta = 0;
    tau is kept)."""
    if blended is None:
        blended = RB.ROLE_BLEND
    if blended:
        beta = 0.0
    if not beta and not tau:
        return fp, {"beta": 0.0, "tau": 0.0}
    x = fp.astype(np.float32)
    plays = x != 0
    cnt = plays.sum(0)
    mu_p = np.where(cnt > 0, x.sum(0) / np.maximum(cnt, 1), 0.0)
    sd = np.sqrt(np.maximum(0.0, np.where(cnt > 1, (x * x).sum(0) / np.maximum(cnt, 1) - mu_p ** 2,
                                          0.0)))
    fppg = dict(zip(inputs["pid"].astype(str), pd.to_numeric(inputs["fppg"], errors="coerce")))
    f = np.array([fppg.get(str(p), np.nan) for p in pids], float)
    ok = np.isfinite(f) & (f > 0) & (cnt > 0)
    shift = np.where(ok, beta * (f - mu_p), 0.0).astype(np.float32)
    add = np.broadcast_to(shift, x.shape).copy()
    if tau:
        add += (rng.standard_normal(x.shape, dtype=np.float32) * (tau * sd).astype(np.float32))
    x += np.where(plays, add, 0.0).astype(np.float32)
    return x, {"beta": beta, "tau": tau, "shifted": int(ok.sum()),
               "mean_shift": float(np.abs(shift[ok]).mean()) if ok.any() else 0.0,
               **({"role_blend": True} if blended else {})}


def strength_field(players, inputs, contest: F.Contest, rules: Rules, name: str, s: "Settings",
                   *, seed: int, own_unc: Mapping[str, float] | None = None
                   ) -> tuple[F.Field, F.OwnershipProxy]:
    """The field proxy at strength soft | medium | sharp (see STRENGTH_W). With
    COURTSIDE_FIELD_K on (`field.field_k()`), K fields (`field.build_fields`, ownership
    uncertainty with the late-news `own_unc` multipliers); `op` stays the un-perturbed proxy."""
    params, version = F.load_params()
    w = float(np.clip(params.get("w_ours", 0.5) + STRENGTH_W[name], 0.0, 1.0))
    op = F.ownership_proxy(inputs, players, rules, archetype=contest.archetype,
                           params={**params, "w_ours": w}, version=version)
    if F.field_k():
        fld = F.build_fields(players, inputs, contest, k=F.field_k(), own_unc=own_unc,
                             rules=rules, n=s.n_field, mix=s.mix, ownership=op,
                             n_solves=s.field_solves, seed=seed, threads=s.threads)
        return fld, op
    fld = F.build_field(players, inputs, contest, rules=rules, n=s.n_field, mix=s.mix,
                        ownership=op, n_solves=s.field_solves, seed=seed, threads=s.threads)
    return fld, op


def prob_band(p: float | None) -> str:
    """A probability as a band (until payouts are calibrated on uploads): '~15-25%'."""
    if p is None or not np.isfinite(p):
        return "n/a"
    edges = (0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97, 1.0)
    if p < 0.01:
        return "below 1%"
    if p >= 0.97:
        return "above 97%"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= p < hi:
            return f"~{lo * 100:.0f}\u2013{hi * 100:.0f}%"
    return "n/a"


def one_in(p: float, sig: int = 1) -> str:
    """'~1 in 3,000': a rare chance as odds, rounded to `sig` significant figures."""
    x = (1.0 / p) * (1 + 1e-9)                          # 1 / 0.00004 is 24999.999...
    d = max(0, int(math.floor(math.log10(x))) + 1 - sig)
    x = int(math.floor(x / 10 ** d + 0.5)) * 10 ** d          # half up, not to even
    return f"~1 in {x:,}"


def se_text(p: float | None, se: float | None) -> str:
    """A chance's simulation SE for display (D9): '± 0.4 pt' (percentage points) from 1 % up,
    '± 35%' (relative to the chance) below; '' when there is none (or the chance is 0)."""
    if p is None or se is None or not np.isfinite(p) or not np.isfinite(se) or p <= 0:
        return ""
    if p >= 0.01:
        pts = se * 100
        return f"± {pts:.1f} pt" if pts >= 0.05 else "± <0.1 pt"
    return f"± {min(se / p, 9.99):.0%}"


def chance_text(p: float | None, calibrated: bool = False, n_sims: int | None = None) -> str:
    """A chance for display: from 1 % up, `prob_band` until payouts are calibrated (a plain
    percentage after); below 1 %, odds ('~1 in 3,000', 1 significant figure until calibrated,
    2 after). Never seen in `n_sims` sims: 'under 1 in {n_sims}'."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p <= 0:
        return f"under 1 in {n_sims:,}" if n_sims else "not seen"
    if p >= 0.01:
        if not calibrated:
            return prob_band(p)
        return "above 99%" if p > 0.99 else f"{p * 100:.0f}%"
    return one_in(p, 2 if calibrated else 1)


def payout_calibrated(conn: sqlite3.Connection | None) -> bool:
    """The one display gate for simulated payout numbers (contest builder: dollar E[payout] /
    ROI; slate page: P(cash)): headline only once >= MIN_CAL_SLATES slates of real standings are
    uploaded. Until then they live in "details (simulated, uncalibrated)" toggles."""
    return conn is not None and calibration_slates(conn) >= MIN_CAL_SLATES


def calibration_slates(conn: sqlite3.Connection) -> int:
    """Slates with uploaded DK standings (the payout model's calibration data)."""
    try:
        r = conn.execute("SELECT COUNT(DISTINCT c.slate_id) FROM dfs_contest_results x "
                         "JOIN dfs_contests c ON c.contest_id = x.contest_id").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(r[0] or 0)


# ============================================================================== result

@dataclass
class Portfolio:
    slate_id: str | None
    contest: F.Contest
    settings: Settings
    lineups: list[dict]                  # per entry (pick order = rank)
    summary: dict
    exposure: list[dict]
    info: dict
    run_id: str | None = None
    label: str = LABEL
    field: Any = None                    # the medium field proxy (not stored)
    fp: Any = None                       # scoring truth (S x field.pids), not stored
    fp_raw: Any = None                   # our sims before the truth model, not stored
    sources: Any = None                  # candidate source per lineup (pick order), not stored

    def slots(self) -> list[list[dict]]:
        return [json.loads(x["slots_json"]) for x in self.lineups]


def _contest_json(c: F.Contest) -> dict:
    return {"contest_id": c.contest_id, "name": c.name, "entry_fee": c.entry_fee,
            "field_size": c.field_size, "paid": c.paid, "archetype": c.archetype,
            "source": c.source, "top_prize": max((float(t["payout"]) for t in c.payouts),
                                                 default=0.0)}


def _summary(lad: Ladder, pay: np.ndarray, rank: np.ndarray, contest: F.Contest) -> dict:
    S, k = pay.shape
    tot = pay.sum(1)
    fee = float(contest.entry_fee or 0.0)
    e = float(tot.mean())
    se = float(tot.std() / math.sqrt(S)) if S > 1 else float("nan")
    cost = fee * k
    return {"entries": k, "e_payout": e, "e_payout_se": se, "cost": cost,
            "roi": (e / cost - 1) if cost else None, "roi_se": (se / cost) if cost else None,
            "p_any_cash": float((rank <= lad.paid).any(1).mean()) if k else 0.0,
            "p_profit": float((tot > cost).mean()) if cost else None,
            "p_any_top1": float((rank <= lad.top1).any(1).mean()) if k else 0.0,
            "e_cashing": float((rank <= lad.paid).sum(1).mean()) if k else 0.0,
            "payout_p50": float(np.median(tot)), "payout_p90": float(np.percentile(tot, 90)),
            "n_sims": S}


def build_portfolio(pool: SlatePool | Sequence[PoolPlayer], inputs: pd.DataFrame,
                    sims: Mapping[str, np.ndarray], contest: F.Contest, s: Settings, *,
                    mu: Mapping[str, float] | None = None, rules: Rules | None = None,
                    now: datetime | None = None, slate_id: str | None = None,
                    run_id: str | None = None, baselines: bool = False,
                    progress: Callable[[str], None] | None = None,
                    split: bool | str | None = None, blended: bool | None = None,
                    own_unc: Mapping[str, float] | None = None,
                    objective: str | None = None) -> Portfolio:
    """The portfolio for one contest. `inputs` is the field.py ownership input table (live:
    `field.slate_inputs`; history: `field.hist_inputs`), `sims` a stored-sims dict (``fp`` S x n,
    ``dk_id``), `mu` our mean projections by DK id (default: ``inputs.proj``). `baselines=True`
    adds the "top N by individual E[payout]" picks (with and without the caps) to `info`.
    `split` (None: PORTFOLIO_SPLIT; True / 'split' / 'gs' / False / 'off') = generate / select /
    report on disjoint sims (SPLIT_ENV). `blended`: the sims carry the role blend (+rb; None =
    the process switch), see `truth_sims`. `own_unc`: late-news ownership-uncertainty
    multipliers {DK id: x} for the K-field mode (COURTSIDE_FIELD_K, `field.build_fields`).
    `objective` (None: PORTFOLIO_OBJ; 'off' | 'flat' | 'blend'): the greedy's selection ladder
    (D9, OBJ_ENV)."""
    t0 = time.perf_counter()
    obj = PORTFOLIO_OBJ if objective is None else objective_mode(objective)
    mode = PORTFOLIO_SPLIT if split is None else split_mode(split)
    split = mode != "off"
    say = progress or (lambda m: None)
    rules = rules or (pool.rules if isinstance(pool, SlatePool) else None)
    if rules is None:
        raise PortfolioError("no rules for the slate")
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    by_id = {p.dk_player_id: p for p in players}
    n = int(s.n)
    if n < 1:
        raise PortfolioError("N must be at least 1")
    if n > int(contest.field_size) - 1:
        raise PortfolioError(f"N = {n} leaves no other entries in a {contest.field_size}-entry "
                             f"contest")
    for p in list(s.locks) + list(s.excludes) + list(s.min_exp) + list(s.max_exp):
        if p not in by_id:
            raise PortfolioError(f"player {p} is not in the slate pool")
    if set(s.locks) & set(s.excludes):
        raise PortfolioError("a player is both locked and excluded")
    if len(s.locks) > rules.n_slots:
        raise PortfolioError("more locks than roster slots")
    rng = np.random.default_rng(s.seed)
    timing: dict[str, float] = {}

    # ---- projections, bans
    if mu is None:
        mu = {str(p): float(v) for p, v in zip(inputs["pid"], inputs["proj"])
              if v is not None and np.isfinite(v)}
    status = dict(zip(inputs["pid"].astype(str), inputs["status"].fillna("").astype(str)
                      .str.upper()))
    bans = set(s.excludes) | {p for p, f in s.max_exp.items() if float(f) <= 0}
    bans |= {p for p in by_id if status.get(p) in F.OUT_STATUSES and p not in s.locks}
    mu = {p: float(v) for p, v in mu.items() if p in by_id and (v >= MIN_PROJ or p in s.locks)}
    for p in s.locks:
        if p not in mu:
            raise PortfolioError(f"locked player {by_id[p].name} has no projection")
    cap_d, low_d = exposure_limits(s, by_id)

    # ---- ownership proxy + field proxy (medium) + sims
    say("field")
    t = time.perf_counter()
    fld, op = strength_field(players, inputs, contest, rules, "medium", s, seed=s.seed,
                             **({"own_unc": own_unc} if F.field_k() else {}))
    fp_raw, ainfo = F.align_sims(sims, fld.pids, s.n_sims, s.seed)
    fp, tinfo = truth_sims(fp_raw, fld.pids, inputs, s.beta, s.tau,
                           np.random.default_rng(s.seed + 7), blended=blended)
    timing["field"] = time.perf_counter() - t
    rows = split_rows(fp.shape[0], s.seed, mode) if split else None
    fp_gen, fp_sel, fp_rep = ((fp[rows["G"]], fp[rows["S"]], fp[rows["R"]]) if split
                              else (fp, fp, fp))
    # scenario rows: OFF uniform with replacement (as before); ON without replacement through
    # G (reshuffled when used up), so the small G set does not repeat objectives
    # ('gs': uniform with replacement through G + S, as OFF through all rows)
    draw = (_cycler(len(rows["G"]), s.seed + SPLIT_SEED + 1) if mode == "split"
            else (lambda: int(rng.integers(0, fp_gen.shape[0]))))
    fcol = {p: j for j, p in enumerate(fld.pids)}
    missing = set(ainfo["missing"])
    if any(p in missing for p in mu):
        log.info("portfolio: %d projected players have no sims column",
                 sum(p in missing for p in mu))

    # ---- xf-v2 (COURTSIDE_XF_V2 ON only): top-1 % leverage on the generation rows
    from . import xfactor as XF
    lev, xf2 = None, None
    if XF.xf_v2():
        say("leverage")
        t = time.perf_counter()
        try:
            lev, xf2 = XF.builder_leverage(fld, players, inputs, contest, rules, fp_gen,
                                           own_unc=own_unc, seed=s.seed, threads=s.threads)
        except Exception as e:  # noqa: BLE001 - the extra solves must not fail a build
            log.exception("portfolio: xf-v2 leverage failed")
            lev, xf2 = None, {"version": XF.VERSION_V2, "error": f"{type(e).__name__}: {e}"}
        timing["leverage"] = time.perf_counter() - t

    # ---- candidates
    say("candidates")
    t = time.perf_counter()
    sources: list[str] = []
    cands, cinfo = candidates(players, rules, mu, fp_gen, fcol, op.as_dict(), s, now=now,
                              bans=bans, rng=rng, tags=sources, draw=draw,
                              **({"lev": lev, "lev_rng": np.random.default_rng(
                                  s.seed + XF.LEV_SEED)} if lev else {}))
    timing["candidates"] = time.perf_counter() - t
    if not cands:
        raise PortfolioError("no legal lineup under the locks and excludes")

    # ---- score against the field
    say("scoring")
    t = time.perf_counter()
    lad = Ladder(contest)
    lad_sel = selection_ladder(lad, obj)             # = lad with the D9 objective OFF
    m_o = max(1, contest.field_size - n)

    def score(lineups: list, fp=fp) -> tuple[_Pool, list[dict], F.FieldEval]:
        ev = F.evaluate(fld, fp, contest, [lu.slot_pairs() for lu in lineups], seed=s.seed,
                        tiers=True, own_entries=n, per_sim=True)
        if isinstance(fld, F.FieldSet):
            return _score_k(fld, ev, lineups, contest, m_o, rng), ev.lineups, ev
        return (dupe_pool(fld, ev, [lu.slot_pairs() for lu in lineups], contest, m_o, rng),
                ev.lineups, ev)

    if split:                     # select on S, report on R (`single` / ev0: R); see SPLIT_ENV
        P, single_sel, _ = score(cands, fp_sel)
        R0, single, ev0 = score(cands, fp_rep)
    else:
        P, single, ev0 = score(cands)
        R0 = single_sel = None
    timing["scoring"] = time.perf_counter() - t

    # ---- greedy
    say("selection")
    t = time.perf_counter()
    pids = sorted({p for lu in cands for _, p in lu.slot_pairs()} | set(by_id))
    pidx = {p: i for i, p in enumerate(pids)}
    inc = _incidence(cands, pidx)
    cap = np.array([cap_d.get(p, n) for p in pids])
    low = np.array([low_d.get(p, 0) for p in pids])
    gamma = max(int(s.gamma), len(s.locks))
    cset = CandSet(P, inc, R0)
    state = {"cands": cands, "single": single, "single_sel": single_sel, "refills": 0,
             "curve_refills": 0}

    def refiller(n_to: int, cap: np.ndarray, low: np.ndarray, key: str, budget: int):
        """`greedy`'s on_empty for a set of `n_to` entries under (cap, low); `key` counts the
        refill batches against `budget`."""
        return lambda chosen, counts: on_empty(chosen, counts, n_to, cap, low, key, budget)

    def on_empty(chosen: list[int], counts: np.ndarray, n: int, cap: np.ndarray,
                 low: np.ndarray, key: str = "refills", budget: int = s.refills) -> bool:
        """Refill: new scenario solves that satisfy the overlap cap against every chosen lineup
        (no-good cuts) with capped players banned and urgent minimums locked."""
        if state[key] >= budget:
            return False
        state[key] += 1
        say(f"refill {state[key]}" if key == "refills" else f"curve refill {state[key]}")
        full = {pids[i] for i in np.flatnonzero(counts >= cap)}
        need = {pids[i] for i in np.flatnonzero(low - counts >= n - len(chosen))}
        cur = state["cands"]
        out = bans | full
        ids = [p for p in mu if p not in out]
        solve = _solver([by_id[p] for p in ids], rules, locks=tuple(set(s.locks) | need),
                        bans=out, now=now, limit=s.solve_limit_s * 2,
                        cuts=[cur[j] for j in chosen], gamma=gamma)
        base = np.array([mu[p] for p in ids])
        cols = np.array([fcol.get(p, -1) for p in ids])
        jobs = []
        for _ in range(max(20, 2 * (n - len(chosen)))):
            r = draw()
            x = np.where(cols >= 0, fp_gen[r, np.maximum(cols, 0)].astype(float), base)
            jobs.append(dict(zip(ids, s.scen_w * base + (1 - s.scen_w) * x)))
        with ThreadPoolExecutor(max(1, s.threads)) as ex:
            new = [lu for lu in ex.map(solve, jobs) if lu is not None]
        have = {_identity(lu, rules.is_showdown) for lu in cur}
        fresh = []
        for lu in new:
            k = _identity(lu, rules.is_showdown)
            if k not in have:
                have.add(k)
                fresh.append(lu)
        if not fresh:
            return False
        if split:
            P2, sel2, _ = score(fresh, fp_sel)
            R2, single2, _ = score(fresh, fp_rep)
            state["single_sel"] = state["single_sel"] + sel2
            cset.R = _cat(cset.R, R2)
        else:
            P2, single2, _ = score(fresh)
        state["cands"] = cur + fresh
        state["single"] = state["single"] + single2
        sources.extend([key[:-1]] * len(fresh))            # "refill" / "curve_refill"
        cset.inc = np.concatenate([cset.inc, _incidence(fresh, pidx)])
        cset.P = _cat(cset.P, P2)
        return True

    chosen, ginfo = greedy(lad_sel, cset, n, gamma=gamma, cap=cap, low=low,
                           on_empty=refiller(n, cap, low, "refills", s.refills))
    timing["selection"] = time.perf_counter() - t
    cands, single, inc, P = state["cands"], state["single"], cset.inc, cset.P
    Rp = cset.report                     # = P with the split OFF

    # ---- output (on the report rows)
    sc, ab, td = Rp.score[:, chosen], Rp.above[:, chosen], Rp.tied[:, chosen]
    pay, rank = (joint_payouts(lad, sc, ab, td) if chosen
                 else (np.zeros((fp_rep.shape[0], 0)),) * 2)
    summ = _summary(lad, pay, rank, contest)
    summ["stopped"] = ginfo["stopped"]
    summ["requested"] = n
    summ.update(_relative(lad, Rp, pay, cinfo["by_source"].get("topmean", 0), n, contest))
    kband = _k_bands(fld, lad, pay, rank, contest) if isinstance(fld, F.FieldSet) else None
    if kband is not None:
        summ["field_k"] = kband["portfolio"]
    denom = F.UNIT * F.WUNIT
    lineups = []
    for k, j in enumerate(chosen):
        lu = cands[j]
        sj = single[j]
        tot = (sc[:, k].astype(np.float64) - F.BIAS) / denom
        lineups.append({
            "rank": k + 1, "slots_json": lu.slots_json(), "salary": int(lu.salary),
            "mean": float(tot.mean()), "proj": float(lu.projected),
            "p10": float(np.percentile(tot, 10)), "p50": float(np.percentile(tot, 50)),
            "p90": float(np.percentile(tot, 90)),
            "p_cash": float((rank[:, k] <= lad.paid).mean()),
            "p_top1": float((rank[:, k] <= lad.top1).mean()),
            "e_payout": float(pay[:, k].mean()),
            "e_payout_single": float(sj["e_payout"]), "p_cash_single": float(sj["p_cash"]),
            "dupes": float(sj["dupes"]), "dupe_band": sj["dupe_band"],
            "marginal": ginfo["marginals"][k] if k < len(ginfo["marginals"]) else None})
        if kband is not None:
            lineups[-1].update(kband["lineups"][k])
    exposure = exposure_table(lineups, by_id, op, mu, cap_d, low_d, n, s)
    ranges = None
    if s.strengths and chosen:
        say("strengths")
        t = time.perf_counter()
        ranges = strength_range(players, inputs, _report_sims(sims, s, rows) if split else sims,
                                contest, rules, [cands[j].slot_pairs() for j in chosen], s,
                                blended=blended,
                                **({"own_unc": own_unc} if F.field_k() else {}))
        ranges["medium"] = {"e_payout": summ["e_payout"], "roi": summ["roi"]}
        timing["strengths"] = time.perf_counter() - t
    summ["strengths"] = ranges
    summ["curve"] = None
    if s.curve and chosen:
        say("curve")
        t = time.perf_counter()
        summ["curve"] = entries_curve(lad, cset, chosen, s, pids, contest, m_o=m_o, gamma=gamma,
                                      built_stopped=ginfo["stopped"],
                                      refill=lambda k, c, lo: refiller(k, c, lo, "curve_refills",
                                                                       s.curve_refills),
                                      **({"sel_lad": lad_sel} if obj != "off" else {}))
        timing["curve"] = time.perf_counter() - t
    info = {"version": VERSION + {"off": "", "split": SPLIT_TAG, "gs": SPLIT_GS_TAG}[mode],
            "label": LABEL,
            "candidates": cinfo, "refills": state["refills"],
            "candidates_total": len(cands), "gamma": gamma,
            "field": fld.summary(), "field_label": F.LABEL, "ownership_version": op.version,
            "ownership_calibrated": F.calibrated(),
            "cash_line": ev0.cash_line_summary(), "tier_ranks": ev0.info["tier_ranks"],
            "eval_mode": ev0.info["mode"], "n_sims": int(fp.shape[0]),
            "missing_sim_players": len(missing), "truth": tinfo,
            "timing": {k: round(v, 2) for k, v in timing.items()},
            "seconds": round(time.perf_counter() - t0, 2)}
    if obj != "off":
        info["version"] += OBJ_TAG + obj
    if xf2 is not None:
        info["version"] += "+xf2"
        xf2["selected_levtilt"] = sum(1 for j in chosen if sources[j] == "levtilt")
        info["xfactor_v2"] = xf2
    if split:
        info["split"] = {k: int(len(v)) for k, v in rows.items()}
        if mode == "gs":
            info["split"]["mode"] = "gs"
        info["split"]["by_source_selected"] = _count([sources[j] for j in chosen])
    if baselines:
        info["baselines"] = _baselines(lad, P, inc, n, state["single_sel"] if split else single,
                                       gamma, cap, low, contest, cands, report=cset.R)
        info["greedy_pairs"] = [cands[j].slot_pairs() for j in chosen]
    return Portfolio(slate_id, contest, s, lineups, summ, exposure, info, run_id, field=fld,
                     fp=fp, fp_raw=fp_raw, sources=[sources[j] for j in chosen])


def dupe_pool(fld: F.Field, ev: F.FieldEval, pairs: Sequence, contest: F.Contest, m_o: int,
              rng: np.random.Generator) -> _Pool:
    """The per-sim pool of an `F.evaluate(..., per_sim=True)` on one field: the expected dupes
    the field sample does not already contain are added to the ties as a Poisson count per sim
    (so dupes split the prize). `pairs`: the evaluated lineups' (slot, id) pairs, in order."""
    ps = ev.per_sim
    _, rows, _ = F._cand_matrix(fld, list(pairs), fld.slots)
    cpt_first = fld.slots[0] == "CPT"
    cnt = {tuple(int(x) for x in r): int(c)
           for r, c in zip(F._canon(fld.lineups, cpt_first), fld.counts)}
    F_n = max(1, fld.n)
    extra = []
    for i, row in enumerate(rows):
        d = float(ev.lineups[i]["dupes"]) * m_o / max(1, contest.field_size - 1)
        in_sample = cnt.get(tuple(int(x) for x in F._key(row, cpt_first)), 0) * m_o / F_n \
            if (row >= 0).all() else 0.0
        extra.append(max(0.0, d - in_sample))
    extra = np.array(extra)
    tied = np.rint(ps["tied"]).astype(np.int32)
    if (extra > 1e-4).any():
        tied += rng.poisson(extra[None, :], size=tied.shape).astype(np.int32)
    return _Pool(ps["score"], np.rint(ps["above"]).astype(np.int32), tied)


def _score_k(fld: "F.FieldSet", ev: F.FieldEval, lineups: list, contest: F.Contest, m_o: int,
             rng: np.random.Generator) -> _Pool:
    """`build_portfolio`'s per-sim pool against K fields (D5): as for one field, the expected
    dupes a field's sample does not contain are added to the ties as a Poisson count, per field
    on its own sim rows (`FieldSet.rows_of`)."""
    ps = ev.per_sim
    S = ps["score"].shape[0]
    tied = np.rint(ps["tied"]).astype(np.int32)
    pairs = [lu.slot_pairs() for lu in lineups]
    for j, f in enumerate(fld.fields):
        _, rows, _ = F._cand_matrix(f, pairs, f.slots)
        cpt_first = f.slots[0] == "CPT"
        cnt = {tuple(int(x) for x in r): int(c)
               for r, c in zip(F._canon(f.lineups, cpt_first), f.counts)}
        F_n = max(1, f.n)
        extra = np.zeros(len(rows))
        for i, row in enumerate(rows):
            d = float(ev.lineups[i]["dupes_k"][j]) * m_o / max(1, contest.field_size - 1)
            in_sample = cnt.get(tuple(int(x) for x in F._key(row, cpt_first)), 0) * m_o / F_n \
                if (row >= 0).all() else 0.0
            extra[i] = max(0.0, d - in_sample)
        r = fld.rows_of(S, j) if S >= fld.k else np.arange(S)
        if (extra > 1e-4).any() and len(r):
            tied[r] += rng.poisson(extra[None, :] / (1 if S >= fld.k else fld.k),
                                   size=(len(r), len(extra))).astype(np.int32)
    return _Pool(ps["score"], np.rint(ps["above"]).astype(np.int32), tied)


def _k_bands(fld: "F.FieldSet", lad: Ladder, pay: np.ndarray, rank: np.ndarray,
             contest: F.Contest) -> dict:
    """Across-field spread of the chosen set's chances (display only): per field k, on its own
    report rows, each lineup's P(cash) / P(top 1 %) and the portfolio's P(any cash) /
    P(any top 1 %) / E[payout]; lo / hi = min / max over the K fields."""
    S, n = rank.shape
    per = []
    for j in range(fld.k):
        r = fld.rows_of(S, j) if S >= fld.k else np.arange(S)
        per.append(_summary(lad, pay[r], rank[r], contest) | {
            "p_cash": (rank[r] <= lad.paid).mean(0), "p_top1": (rank[r] <= lad.top1).mean(0)})
    port = {"k": fld.k, "version": fld.version,
            "sd": float(F.OWN_SD.get(fld.ownership.archetype, np.nan)),
            "own_unc_players": int(sum(1 for v in fld.own_unc.values() if float(v) != 1.0))}
    for key in ("p_any_cash", "p_any_top1", "e_payout"):
        v = [float(x[key]) for x in per]
        port[key + "_k"] = v
        port[key + "_lo"], port[key + "_hi"] = min(v), max(v)
    lus = []
    for i in range(n):
        d = {}
        for key in ("p_cash", "p_top1"):
            v = [float(x[key][i]) for x in per]
            d[key + "_lo"], d[key + "_hi"] = min(v), max(v)
        lus.append(d)
    return {"portfolio": port, "lineups": lus}


def _cycler(n: int, seed: int) -> Callable[[], int]:
    """draw() -> row index in 0..n-1 without replacement, reshuffled after every n draws."""
    rng = np.random.default_rng(seed)
    st = {"perm": rng.permutation(n), "i": 0}

    def draw() -> int:
        if st["i"] >= n:
            st["perm"], st["i"] = rng.permutation(n), 0
        st["i"] += 1
        return int(st["perm"][st["i"] - 1])
    return draw


def _cat(a: _Pool, b: _Pool) -> _Pool:
    """Candidates `b` appended to `a` (same sim rows)."""
    return _Pool(np.concatenate([a.score, b.score], 1), np.concatenate([a.above, b.above], 1),
                 np.concatenate([a.tied, b.tied], 1))


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _aligned_rows(S: int, n_sims: int | None, seed: int) -> np.ndarray:
    """The stored-sims rows behind `field.align_sims(sims, pids, n_sims, seed)` (same rule)."""
    if n_sims and n_sims < S:
        return np.sort(np.random.default_rng(seed).choice(S, n_sims, replace=False))
    return np.arange(S)


def _report_sims(sims: Mapping[str, np.ndarray], s: Settings,
                 rows: Mapping[str, np.ndarray]) -> dict:
    """The stored sims restricted to the build's report rows R (split ON): the soft / sharp
    strengths are then scored on nights the greedy never saw."""
    stored = _aligned_rows(sims["fp"].shape[0], s.n_sims, s.seed)[rows["R"]]
    return {**sims, "fp": sims["fp"][stored]}


# ============================================================================== entries curve

CURVE_POINTS = (1, 3, 5, 10, 20, 50, 150)
CURVE_VERSION = "entries-curve-1.0"


def curve_points(n_built: int, top: int | None) -> list[int]:
    """Entry counts of the curve: CURVE_POINTS up to `top` (the contest's max entries, at most
    MAX_ENTRIES) and `top` itself, plus the build's own N. `top` None: up to the build's N."""
    n_built = max(1, int(n_built))
    top = n_built if top is None else max(n_built, min(int(top), MAX_ENTRIES))
    return sorted({k for k in CURVE_POINTS if k <= top} | {n_built, top})


def _caps_for(s: Settings, k: int, pids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    cap_d, low_d = exposure_limits(dc_replace(s, n=int(k)), pids)
    return (np.array([cap_d.get(p, k) for p in pids]), np.array([low_d.get(p, 0) for p in pids]))


def _rescaled(x: np.ndarray, f: float) -> np.ndarray:
    return x.astype(np.int64) if f == 1.0 else np.rint(x * f).astype(np.int64)


def curve_stats(lad: Ladder, P: _Pool, idx: Sequence[int], contest: F.Contest, *,
                m_o: int) -> dict:
    """Chances of a set of our entries `idx` (columns of P) on the same sims, scored jointly:
    in every sim our entries push each other down, so the chances of "at least one" come from
    the sims themselves, not from 1 - prod(1 - p). `P` was scored with `m_o` other entries; a
    set of k entries has M - k others, so the counts above / tied are rescaled by
    (M - k) / m_o (exactly 1 for the build's own N)."""
    k = len(idx)
    M = int(contest.field_size)
    f = max(1, M - k) / max(1, m_o)
    sc = P.score[:, idx]
    ab, td = _rescaled(P.above[:, idx], f), _rescaled(P.tied[:, idx], f)
    pay, rank = joint_payouts(lad, sc, ab, td)
    best = rank.min(1)
    top01 = max(1, int(math.ceil(0.001 * M)))
    fee = float(contest.entry_fee or 0.0)
    e = float(pay.sum(1).mean())
    alone = ab + 1                                     # each entry's rank if it were our only one
    indep = {name: float(1 - np.prod(1 - (alone <= r).mean(0)))
             for name, r in (("p_any_cash", lad.paid), ("p_any_top1", lad.top1))}
    out = {"entries": k, "cost": fee * k,
           "p_any_cash": float((best <= lad.paid).mean()),
           "p_any_top1": float((best <= lad.top1).mean()),
           "p_any_top01": float((best <= top01).mean()),
           "p_first": float((best <= 1).mean()),
           "e_payout": e, "e_payout_per_dollar": (e / (fee * k)) if fee else None,
           "if_independent": indep, "top01_rank": top01, "n_sims": int(sc.shape[0])}
    out["se"] = curve_se(out, pay.sum(1))
    return out


def binom_se(p: float | None, n: int | None) -> float | None:
    """Simulation SE of a chance estimated as a share of `n` independent sims (binomial):
    sqrt(p (1 - p) / n). The sims of a build are iid rows, so this is the exact MC error of the
    curve's chances (not of the model, the field proxy or the picking)."""
    if p is None or not n or not np.isfinite(p):
        return None
    return float(math.sqrt(max(0.0, p * (1.0 - p)) / int(n)))


def curve_se(point: Mapping, tot: np.ndarray | None = None) -> dict:
    """D9: SEs of a curve point's numbers (display only; the numbers are unchanged). Chances:
    binomial over the point's sims; E[payout]: sd of the per-sim total payout / sqrt(S) (the sims
    are iid rows), per dollar by the same factor."""
    n = point.get("n_sims")
    out = {k: binom_se(point.get(k), n) for k in ("p_any_cash", "p_any_top1", "p_any_top01",
                                                   "p_first")}
    if tot is not None and len(tot) > 1:
        se = float(np.std(tot) / math.sqrt(len(tot)))
        out["e_payout"] = se
        cost = float(point.get("cost") or 0.0)
        out["e_payout_per_dollar"] = se / cost if cost else None
    return out


def _cap_subset(parent: Sequence[int], first: Sequence[int], k: int, inc: np.ndarray,
                cap: np.ndarray, low: np.ndarray, time_s: float = 2.0) -> list[int] | None:
    """A k-subset of `parent` that meets the caps (CP-SAT), keeping as many of the greedy's own
    picks `first` and as early (in the parent's pick order) as it can; None if none exists."""
    from ortools.sat.python import cp_model
    L = len(parent)
    sub = inc[list(parent)].astype(np.int64)
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(L)]
    m.Add(sum(x) == k)
    for p in np.flatnonzero(sub.sum(0) > cap):
        m.Add(sum(x[i] for i in np.flatnonzero(sub[:, p])) <= int(cap[p]))
    for p in np.flatnonzero(low > 0):
        m.Add(sum(x[i] for i in np.flatnonzero(sub[:, p])) >= int(low[p]))
    keep = set(first)
    m.Maximize(sum(((L - i) + (L if parent[i] in keep else 0)) * x[i] for i in range(L)))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = time_s
    sv.parameters.num_workers = 1
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [parent[i] for i in range(L) if sv.Value(x[i])]


def _fill_least_over(pick: list[int], parent: Sequence[int], k: int, inc: np.ndarray,
                     cap: np.ndarray) -> list[int]:
    """Complete a subset whose caps cannot all be met: add, one at a time, the parent lineup
    that pushes the players' counts least above their caps (ties: earlier pick)."""
    pick = list(pick)
    counts = inc[pick].sum(0).astype(np.int64) if pick else np.zeros(inc.shape[1], np.int64)
    while len(pick) < k:
        rest = [j for j in parent if j not in pick]
        over = [int(np.maximum(0, counts + inc[j] - cap).sum()) for j in rest]
        j = rest[int(np.argmin(over))]
        pick.append(j)
        counts += inc[j]
    return pick


def entries_curve(lad: Ladder, cs: CandSet, chosen: Sequence[int], s: Settings,
                  pids: Sequence[str], contest: F.Contest, *, m_o: int, gamma: int,
                  built_stopped: str | None = None,
                  refill: Callable[[int, np.ndarray, np.ndarray], Callable] | None = None,
                  sel_lad: Ladder | None = None) -> dict:
    """The "how many entries" curve: chances of at least one entry in the money / top 1 % /
    top 0.1 % / 1st and the entry cost for N = 1, 3, 5, 10, 20, 50, 150 (up to `s.curve_to`)
    and the build's own N, all on the SAME simulated nights as the build.

    Nesting. The greedy marginal depends only on the chosen set, but its constraints do not:
    the default exposure cap is floor(max * N) from N >= exposure_from and minimum exposures are
    ceil(min * N), so the first k picks of an N-entry build can break the k-entry caps (and an
    independent k-entry build can differ from them). Each point is therefore the greedy
    selection under ITS OWN N's caps, constrained to nest:
      * k < N: picked from the next larger point's set (top down from the build), so every
        smaller set is a subset of the build (`build_ranks` says which lineups);
      * k = N: the build itself (same numbers as its summary);
      * k > N: the build plus greedy picks from the candidates under k's caps (refilled with
        new solves if the pool runs dry; `refill` makes the on_empty callback), picked on the
        first `s.curve_select_sims` sims to keep the web build fast (a 20-entry build extended
        to 150 spent 34 s here on all 25,000 sims) and scored on all of them. These extra
        lineups are not part of the build or its CSV.
    Nested sets make every "at least one" chance non-decreasing in N on the same sims: an entry
    added above one of ours takes a rank no worse than the one it pushes down (the best rank
    of a set is 1 + the fewest other entries above any of its lineups). If the greedy gets
    stuck on a subset's caps, a CP-SAT subset meets them keeping the earliest picks; if no
    subset can, the least-overage fill is used (`caps_relaxed`). `sel_lad` (D9 selection
    objective ON): the ladder the picks are made under; the chances use `lad`."""
    t0 = time.perf_counter()
    pick_lad = lad if sel_lad is None else sel_lad
    chosen = [int(j) for j in chosen]
    N = len(chosen)
    if not N:
        return {"version": CURVE_VERSION, "points": [], "n_build": 0}
    pts = curve_points(N, s.curve_to if built_stopped is None else N)
    sets: dict[int, list[int]] = {N: chosen}
    meta: dict[int, dict] = {N: {"source": "build", "caps_relaxed": False}}
    pos = {j: r + 1 for r, j in enumerate(chosen)}          # candidate -> rank in the build
    # ---- k < N: top down, each a subset of the next larger set
    parent = chosen
    for k in sorted((p for p in pts if p < N), reverse=True):
        try:
            cap, low = _caps_for(s, k, pids)
        except PortfolioError:
            cap, low = np.full(len(pids), k), np.zeros(len(pids), int)
        sub = CandSet(_Pool(cs.P.score[:, parent], cs.P.above[:, parent], cs.P.tied[:, parent]),
                      cs.inc[parent])
        got, _ = greedy(pick_lad, sub, k, gamma=gamma, cap=cap, low=low)
        pick = [parent[i] for i in got]
        relaxed = False
        if len(pick) < k:                  # greedy painted itself into a corner: exact subset
            pick = _cap_subset(parent, pick, k, cs.inc, cap, low)
            if pick is None:               # no subset meets the caps: least overage
                relaxed = True
                pick = _fill_least_over([parent[i] for i in got], parent, k, cs.inc, cap)
        sets[k] = sorted(pick, key=lambda j: pos[j])
        meta[k] = {"source": "subset", "caps_relaxed": relaxed}
        parent = sets[k]
    # ---- k > N: bottom up from the build, picked on the first `curve_select_sims` sims (the
    # greedy step costs ~S x k; the sims are exchangeable) and scored on all of them below
    S = cs.P.score.shape[0]
    rows = slice(0, min(S, int(s.curve_select_sims or S)))
    sel = CandSet(_Pool(cs.P.score[rows], cs.P.above[rows], cs.P.tied[rows]), cs.inc)

    def on_empty_for(k, cap, low):
        inner = refill(k, cap, low) if refill else None
        if inner is None:
            return None

        def go(ch, counts):
            ok = inner(ch, counts)
            if ok:                                           # new candidates: refresh the view
                sel.P = _Pool(cs.P.score[rows], cs.P.above[rows], cs.P.tied[rows])
                sel.inc = cs.inc
            return ok
        return go

    cur = chosen
    for k in sorted(p for p in pts if p > N):
        try:
            cap, low = _caps_for(s, k, pids)
        except PortfolioError:
            break
        got, gi = greedy(pick_lad, sel, k, gamma=gamma, cap=cap, low=low, start=cur,
                         on_empty=on_empty_for(k, cap, low))
        sets[k] = list(got)
        meta[k] = {"source": "extended", "caps_relaxed": False, "stopped": gi["stopped"]}
        cur = sets[k]
        if gi["stopped"]:
            break
    points = []
    for k in sorted(sets):
        idx = sets[k]
        row = {"requested": k, **curve_stats(lad, cs.report, idx, contest, m_o=m_o), **meta[k]}
        if meta[k]["source"] != "extended":
            row["build_ranks"] = [pos[j] for j in idx]
            row["is_prefix"] = row["build_ranks"] == list(range(1, len(idx) + 1))
        else:
            row["build_lineups"] = N
        points.append(row)
    top = pts[-1]
    return {"version": CURVE_VERSION, "label": LABEL, "n_build": N, "points": points,
            "select_sims": rows.stop if any(p > N for p in pts) else None,
            "requested_points": pts, "complete": max(sets) >= top and
            len(sets[max(sets)]) >= top,
            "nested_prefix": all(p.get("is_prefix", True) for p in points),
            "curve_to": s.curve_to, "seconds": round(time.perf_counter() - t0, 2)}


def _relative(lad: Ladder, P: _Pool, pay: np.ndarray, n_top: int, n: int,
              contest: F.Contest) -> dict:
    """The portfolio's E[payout] relative to the Top-N-by-projection set (the first N top-mean
    candidates: `topn.top_n` on our means, min different 1), both scored jointly on the same
    sims and field. Ratio SE by the delta method on the paired per-sim totals."""
    if n_top < n or pay.shape[1] == 0:
        return {"relative": None, "baseline": None}
    base = list(range(n))
    bpay, brank = joint_payouts(lad, P.score[:, base], P.above[:, base], P.tied[:, base])
    bs = _summary(lad, bpay, brank, contest)
    a, b = pay.sum(1), bpay.sum(1)
    if b.mean() <= 0:
        return {"relative": None, "baseline": {"e_payout": 0.0}}
    r = float(a.mean() / b.mean())
    se = float((a - r * b).std() / (math.sqrt(len(a)) * b.mean()))
    return {"relative": r, "relative_se": se,
            "baseline": {"name": "Top N by projection", "e_payout": bs["e_payout"],
                         "p_any_cash": bs["p_any_cash"], "p_any_top1": bs["p_any_top1"]}}


def sim_payout(field: F.Field, contest: F.Contest, fp: np.ndarray, lineup_pairs: Sequence,
               *, seed: int = 0) -> float:
    """Simulated joint E[payout] of a set of our lineups on per-sim points `fp` (columns =
    `field.pids`)."""
    ev = F.evaluate(field, fp, contest, list(lineup_pairs), seed=seed, tiers=False,
                    own_entries=len(lineup_pairs), per_sim=True)
    ps = ev.per_sim
    pay, _ = joint_payouts(Ladder(contest), ps["score"], np.rint(ps["above"]).astype(np.int64),
                           np.rint(ps["tied"]).astype(np.int64))
    return float(pay.sum(1).mean())


def _baselines(lad, P, inc, n, single, gamma, cap, low, contest, cands, report=None) -> dict:
    """'top N by individual E[payout]' (no caps) and the same with the portfolio's caps, scored
    jointly on the sims (for the T2.12 comparison). `report`: score the picks there (split ON)."""
    ind = np.array([x["e_payout"] for x in single])
    out = {}
    top = list(np.argsort(-ind, kind="stable")[:n])
    cs = CandSet(P, inc)
    capped, _ = greedy(lad, cs, n, gamma=gamma, cap=cap, low=low, objective=ind)
    topmean = list(range(n))             # candidates start with the Top-N by mean
    # the joint-marginal greedy with no exposure cap and any two lineups merely distinct
    open_, _ = greedy(lad, cs, n, gamma=inc.shape[1], cap=np.full(inc.shape[1], n),
                      low=np.zeros(inc.shape[1], int))
    for name, ch in (("top_individual", top), ("top_individual_capped", capped),
                     ("greedy_uncapped", open_), ("top_mean", topmean)):
        Q = P if report is None else report
        pay, rank = joint_payouts(lad, Q.score[:, ch], Q.above[:, ch], Q.tied[:, ch])
        sm = _summary(lad, pay, rank, contest)
        out[name] = {"pairs": [cands[int(c)].slot_pairs() for c in ch], "e_payout": sm["e_payout"],
                     "e_payout_se": sm["e_payout_se"], "roi": sm["roi"],
                     "p_any_cash": sm["p_any_cash"]}
    return out


def exposure_table(lineups: list[dict], by_id: Mapping[str, PoolPlayer], op: F.OwnershipProxy,
                   mu: Mapping[str, float], cap: Mapping[str, int], low: Mapping[str, int],
                   n: int, s: Settings) -> list[dict]:
    cnt: dict[str, int] = {}
    for lu in lineups:
        for slot in json.loads(lu["slots_json"]):
            cnt[str(slot["player_id"])] = cnt.get(str(slot["player_id"]), 0) + 1
    own = op.as_dict()
    k = max(1, len(lineups))
    rows = []
    for p, c in cnt.items():
        pl = by_id.get(p)
        o = own.get(p, 0.0)
        rows.append({"player_id": p, "name": pl.name if pl else p, "team": pl.team if pl else None,
                     "positions": "/".join(pl.positions) if pl else "",
                     "salary": pl.salary if pl else None, "mean": mu.get(p),
                     "count": c, "share": c / k, "own_proxy": o, "own_band": F.band(o),
                     "cap": cap.get(p, n), "min": low.get(p, 0),
                     "locked": p in s.locks})
    rows.sort(key=lambda r: (-r["count"], -(r["mean"] or 0), r["name"]))
    return rows


def strength_range(players, inputs, sims, contest, rules, chosen_pairs, s: Settings, *,
                   blended: bool | None = None,
                   own_unc: Mapping[str, float] | None = None) -> dict:
    """E[payout] / ROI of the chosen portfolio against the soft and sharp field strengths
    (same scoring truth, fewer sims)."""
    out = {}
    lad = Ladder(contest)
    n = len(chosen_pairs)
    for name in ("soft", "sharp"):
        f, _ = strength_field(players, inputs, contest, rules, name, s, seed=s.seed + 1,
                              **({"own_unc": own_unc} if F.field_k() else {}))
        fp, _ = F.align_sims(sims, f.pids, min(s.strength_sims, s.n_sims or 10**9), s.seed + 1)
        fp, _ = truth_sims(fp, f.pids, inputs, s.beta, s.tau, np.random.default_rng(s.seed + 8),
                           blended=blended)
        ev = F.evaluate(f, fp, contest, chosen_pairs, seed=s.seed + 1, tiers=False,
                        own_entries=n, per_sim=True)
        ps = ev.per_sim
        pay, rank = joint_payouts(lad, ps["score"], np.rint(ps["above"]).astype(np.int64),
                                  np.rint(ps["tied"]).astype(np.int64))
        sm = _summary(lad, pay, rank, contest)
        out[name] = {"e_payout": sm["e_payout"], "roi": sm["roi"], "n_sims": sm["n_sims"]}
    return out


# ============================================================================== realised (backtest)

def realised(field: F.Field, contest: F.Contest, lineup_pairs: Sequence, actual: np.ndarray,
             *, seed: int = 0) -> dict:
    """Joint payout of a set of our lineups when the slate's ACTUAL DK points (one row, columns =
    `field.pids`) are scored against the field proxy: what the set would have won against the
    modelled field. A `field.FieldSet` (K fields): the mean over its fields."""
    if isinstance(field, F.FieldSet):
        per = [realised(f, contest, lineup_pairs, actual, seed=seed + j)
               for j, f in enumerate(field.fields)]
        out = {key: float(np.mean([r[key] for r in per])) for key in ("payout", "cashed",
                                                                      "best_rank")}
        cost = per[0]["cost"]
        out.update({"cost": cost, "roi": (out["payout"] / cost - 1) if cost else None,
                    "mode": per[0]["mode"], "field_k": len(per),
                    "payout_k": [r["payout"] for r in per]})
        return out
    lad = Ladder(contest)
    n = len(lineup_pairs)
    ev = F.evaluate(field, np.asarray(actual, np.float32).reshape(1, -1), contest,
                    list(lineup_pairs), seed=seed, tiers=False, own_entries=n, per_sim=True)
    ps = ev.per_sim
    pay, rank = joint_payouts(lad, ps["score"], np.rint(ps["above"]).astype(np.int64),
                              np.rint(ps["tied"]).astype(np.int64))
    cost = n * float(contest.entry_fee or 0)
    tot = float(pay.sum())
    return {"payout": tot, "cost": cost, "roi": (tot / cost - 1) if cost else None,
            "cashed": int((rank <= lad.paid).sum()), "best_rank": int(rank.min()),
            "mode": ev.info["mode"]}


# ============================================================================== storage

def new_id() -> str:
    return ID_PREFIX + uuid.uuid4().hex[:10]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start_row(conn: sqlite3.Connection, portfolio_id: str, slate_id, contest_id, s: Settings,
              *, run_id: str | None = None) -> None:
    conn.execute("INSERT INTO dfs_portfolios (portfolio_id, slate_id, contest_id, run_id, "
                 "created_at, status, n_requested, settings_json) VALUES (?,?,?,?,?,?,?,?)",
                 (portfolio_id, str(slate_id), None if contest_id is None else str(contest_id),
                  run_id, _now_iso(), "building", int(s.n), json.dumps(s.as_json())))
    conn.commit()


def fail_row(conn: sqlite3.Connection, portfolio_id: str, error: str, seconds=None) -> None:
    conn.execute("UPDATE dfs_portfolios SET status='error', error=?, seconds=? "
                 "WHERE portfolio_id=?", (error[:2000], seconds, portfolio_id))
    conn.commit()


def save(conn: sqlite3.Connection, pf: Portfolio, portfolio_id: str | None = None) -> str:
    """Write (or complete) the `dfs_portfolios` row and the `dfs_lineups` rows (mode 'gpp')."""
    pid = portfolio_id or new_id()
    exists = conn.execute("SELECT 1 FROM dfs_portfolios WHERE portfolio_id=?",
                          (pid,)).fetchone()
    if not exists:
        start_row(conn, pid, pf.slate_id, pf.contest.contest_id, pf.settings, run_id=pf.run_id)
    summary = {"summary": pf.summary, "exposure": pf.exposure, "contest": _contest_json(pf.contest),
               "info": pf.info, "label": pf.label}
    conn.execute("DELETE FROM dfs_lineups WHERE run_id=? AND mode=?", (pid, MODE))
    for lu in pf.lineups:
        why = {k: lu[k] for k in ("e_payout_single", "p_cash_single", "dupe_band", "marginal",
                                  "proj")}
        conn.execute(
            "INSERT INTO dfs_lineups (run_id, mode, contest_id, rank, slots_json, salary, mean, "
            "p10, p50, p90, p_cash, p_top1, e_payout, dupes, why_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, MODE, pf.contest.contest_id, lu["rank"], lu["slots_json"], lu["salary"],
             lu["mean"], lu["p10"], lu["p50"], lu["p90"], round(lu["p_cash"], 5),
             round(lu["p_top1"], 5), round(lu["e_payout"], 4), round(lu["dupes"], 4),
             json.dumps(why, default=float)))
    conn.execute("UPDATE dfs_portfolios SET status='ok', run_id=COALESCE(?, run_id), n_built=?, "
                 "summary_json=?, seconds=?, error=NULL WHERE portfolio_id=?",
                 (pf.run_id, len(pf.lineups), json.dumps(summary, default=_json_default),
                  pf.info.get("seconds"), pid))
    conn.commit()
    return pid


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load(conn: sqlite3.Connection, portfolio_id: str) -> dict | None:
    """The stored portfolio: row fields + parsed settings / summary + lineups (slots parsed)."""
    try:
        r = conn.execute("SELECT * FROM dfs_portfolios WHERE portfolio_id=?",
                         (portfolio_id,)).fetchone()
    except sqlite3.OperationalError:          # read-only DB before migration 13
        return None
    if r is None:
        return None
    d = dict(r)
    d["settings"] = json.loads(d.get("settings_json") or "{}")
    d["data"] = json.loads(d.get("summary_json") or "{}") if d.get("summary_json") else {}
    rows = conn.execute("SELECT * FROM dfs_lineups WHERE run_id=? AND mode=? ORDER BY rank, id",
                        (portfolio_id, MODE)).fetchall()
    d["lineups"] = []
    for x in rows:
        lu = dict(x)
        lu["slots"] = json.loads(lu["slots_json"] or "[]")
        lu["why"] = json.loads(lu["why_json"] or "{}")
        d["lineups"].append(lu)
    return d


def recent(conn: sqlite3.Connection, slate_id, limit: int = 10) -> list[dict]:
    try:
        rows = conn.execute("SELECT portfolio_id, contest_id, created_at, status, n_requested, "
                            "n_built, seconds, error FROM dfs_portfolios WHERE slate_id=? "
                            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                            (str(slate_id), limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


# ============================================================================== live entry point

def resolve_players(pool: SlatePool, items: Iterable[str]) -> list[str]:
    """DK ids from ids or (unique, case-insensitive) name fragments."""
    out = []
    by_id = pool.by_id()
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        if it in by_id:
            out.append(it)
            continue
        hits = [p for p in pool.players if it.lower() in p.name.lower()]
        exact = [p for p in hits if p.name.lower() == it.lower()]
        hits = exact or hits
        if len({p.dk_player_id for p in hits}) != 1:
            raise PortfolioError(f"'{it}' matches {len(hits)} players" if hits
                                 else f"no player matches '{it}'")
        out.append(hits[0].dk_player_id)
    return out


def build_for_slate(conn: sqlite3.Connection, slate_id, contest_id, s: Settings, *,
                    run_id: str | None = None, now: datetime | None = None,
                    progress: Callable[[str], None] | None = None) -> Portfolio:
    """Live build: the slate's newest projection run with sims (or `run_id`), the captured
    contest, N capped at the contest's max entries per user and 150."""
    ctx = F.prepare(conn, slate_id, run_id, n_sims=s.n_sims, seed=s.seed)
    if ctx["sims"] is None:
        raise PortfolioError("this slate has no simulated projection run yet (the contest "
                             "builder needs the sims)")
    contest = F.load_contest(conn, str(contest_id))
    if contest is None:
        raise PortfolioError(f"unknown contest {contest_id}")
    r = conn.execute("SELECT slate_id, max_per_user FROM dfs_contests WHERE contest_id=?",
                     (str(contest_id),)).fetchone()
    if r is not None and r["slate_id"] is not None and str(r["slate_id"]) != str(slate_id):
        raise PortfolioError(f"contest {contest_id} is not on slate {slate_id}")
    max_user = int(r["max_per_user"] or MAX_ENTRIES) if r is not None else MAX_ENTRIES
    s.n = max(1, min(int(s.n), max_user, MAX_ENTRIES, contest.field_size - 1))
    if s.curve and s.curve_to is None:
        s.curve_to = max(1, min(max_user, MAX_ENTRIES, contest.field_size - 1))
    pool = ctx["pool"]
    pf = build_portfolio(pool, ctx["inputs"], ctx["sims"], contest, s,
                         now=now or datetime.now(timezone.utc), slate_id=str(slate_id),
                         run_id=ctx["info"]["run_id"], progress=progress,
                         blended=RB.run_blended(conn, ctx["info"]["run_id"]),
                         **({"own_unc": F.run_own_uncertainty(conn, ctx["info"]["run_id"])}
                            if F.field_k() else {}))
    pf.summary["xfactor"] = xfactor_summary(conn, pf)
    return pf


def xfactor_summary(conn: sqlite3.Connection, pf: Portfolio) -> dict | None:
    """How many of the portfolio's lineups include at least one X factor player of the run
    (`xfactor.py`; reporting only, the selection does not use it). None when the run has no
    sims or the table cannot be read."""
    from . import xfactor as XF
    try:
        table = XF.run_table(conn, pf.run_id)
    except Exception:  # noqa: BLE001 - a report line must not fail a build
        log.exception("xfactor summary failed for run %s", pf.run_id)
        return None
    if not table:
        return None
    ids = XF.xfactor_ids(table)
    slots = pf.slots()
    names = sorted({str(x.get("name")) for sl in slots for x in sl
                    if str(x.get("player_id")) in ids})
    return {"version": XF.VERSION, "n_players": len(ids),
            "lineups_with": XF.lineup_count(slots, ids), "entries": len(slots),
            "players_used": names}


def contest_options(conn: sqlite3.Connection, slate_id) -> list[dict]:
    """The slate's captured contests for the builder (fee, field, max entries, top prize)."""
    rows = conn.execute("SELECT contest_id, name, entry_fee, field_size, max_per_user, "
                        "total_prizes, payout_json FROM dfs_contests WHERE slate_id=? "
                        "ORDER BY field_size DESC", (str(slate_id),)).fetchall()
    out = []
    for r in rows:
        try:
            pays = json.loads(r["payout_json"] or "[]")
        except ValueError:
            pays = []
        if not pays or not r["field_size"] or int(r["field_size"]) < 2:
            continue
        top = max((float(t.get("payout") or 0) for t in pays), default=0.0)
        paid = max((int(t.get("max_pos") or 0) for t in pays), default=0)
        arch = F.classify_contest({"payouts": pays, "field_size": int(r["field_size"])})
        out.append({"contest_id": str(r["contest_id"]), "name": (r["name"] or "").strip(),
                    "entry_fee": float(r["entry_fee"] or 0), "field_size": int(r["field_size"]),
                    "max_per_user": int(r["max_per_user"] or 1), "top_prize": top,
                    "total_prizes": float(r["total_prizes"] or 0), "paid": paid,
                    "archetype": arch})
    return out


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("portfolio", help="GPP contest portfolio for a slate (SIMULATED against "
                                         "a modelled field, not real results)")
    p.add_argument("--slate", required=True, help="DK draft group id")
    p.add_argument("--contest", required=True, help="captured contest id (dfs_contests)")
    p.add_argument("--n", type=int, default=20, help="entries (capped at the contest's max)")
    p.add_argument("--run", default=None, help="projection run id (default: newest with sims)")
    p.add_argument("--gamma", type=int, default=5, help="max shared players between lineups")
    p.add_argument("--max-exp", type=float, default=0.60, help="default max exposure (0-1)")
    p.add_argument("--lock", action="append", default=[], help="player id or name (repeat)")
    p.add_argument("--exclude", action="append", default=[], help="player id or name (repeat)")
    p.add_argument("--min", action="append", default=[], metavar="PLAYER=FRAC")
    p.add_argument("--max", action="append", default=[], metavar="PLAYER=FRAC")
    p.add_argument("--sims", type=int, default=25_000)
    p.add_argument("--field-n", type=int, default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strengths", action="store_true", help="also soft / sharp field E[payout]")
    p.add_argument("--baselines", action="store_true",
                   help="also report 'top N by individual E[payout]' on the same sims")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--csv", default=None, help="write the DK upload CSV here")
    return {"portfolio": _cmd}


def _pairs(items: Sequence[str], pool: SlatePool) -> dict[str, float]:
    out = {}
    for it in items:
        if "=" not in it:
            raise PortfolioError(f"expected PLAYER=FRACTION, got '{it}'")
        k, v = it.rsplit("=", 1)
        out[resolve_players(pool, [k])[0]] = float(v)
    return out


def _cmd(args) -> int:
    from ... import db
    from . import export as X
    from . import rules as R
    conn = db.connect()
    try:
        pool = R.build_pool(conn, str(args.slate))
        s = Settings(n=args.n, gamma=args.gamma, max_exposure=args.max_exp,
                     locks=tuple(resolve_players(pool, args.lock)),
                     excludes=tuple(resolve_players(pool, args.exclude)),
                     min_exp=_pairs(args.min, pool), max_exp=_pairs(args.max, pool),
                     n_sims=args.sims, n_field=args.field_n, threads=args.threads,
                     seed=args.seed, strengths=args.strengths)
        t0 = time.perf_counter()
        ctx = F.prepare(conn, args.slate, args.run, n_sims=s.n_sims, seed=s.seed)
        if ctx["sims"] is None:
            raise PortfolioError("no simulated projection run for this slate")
        contest = F.load_contest(conn, args.contest)
        if contest is None:
            raise PortfolioError(f"unknown contest {args.contest}")
        r = conn.execute("SELECT max_per_user FROM dfs_contests WHERE contest_id=?",
                         (str(args.contest),)).fetchone()
        mx = int(r[0] or MAX_ENTRIES) if r else MAX_ENTRIES
        if s.n > mx:
            print(f"N capped at the contest's max entries per user: {mx}")
        s.n = max(1, min(s.n, mx, MAX_ENTRIES))
        s.curve_to = max(1, min(mx, MAX_ENTRIES, contest.field_size - 1))
        pf = build_portfolio(ctx["pool"], ctx["inputs"], ctx["sims"], contest, s,
                             now=datetime.now(timezone.utc), slate_id=str(args.slate),
                             run_id=ctx["info"]["run_id"], baselines=args.baselines,
                             progress=lambda m: print(f"  ... {m} ({time.perf_counter() - t0:.0f}s)"),
                             blended=RB.run_blended(conn, ctx["info"]["run_id"]),
                             **({"own_unc": F.run_own_uncertainty(conn, ctx["info"]["run_id"])}
                                if F.field_k() else {}))
    except PortfolioError as e:
        print(f"error: {e}")
        return 1
    pf.summary["xfactor"] = xfactor_summary(conn, pf)
    _print(pf)
    if args.baselines and "baselines" in pf.info:
        print("\nBASELINES on the same sims (simulated):")
        print(f"  greedy portfolio             E[payout] ${pf.summary['e_payout']:,.2f} "
              f"± {pf.summary['e_payout_se']:.2f}")
        for k, b in pf.info["baselines"].items():
            print(f"  {k:<28} E[payout] ${b['e_payout']:,.2f} ± {b['e_payout_se']:.2f}  "
                  f"P(any cash) {b['p_any_cash']:.1%}")
    if not args.no_save:
        pid = save(conn, pf)
        print(f"\nsaved portfolio {pid}")
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(X.to_csv(pf.slots()))
        print(f"wrote {args.csv}")
    return 0


def _print(pf: Portfolio) -> None:
    c, sm = pf.contest, pf.summary
    print(f"\nCONTEST PORTFOLIO  {c.name}  (fee ${c.entry_fee:g}, field {c.field_size:,}, paid "
          f"{c.paid:,}, {c.archetype})")
    print(f"  {LABEL}; ownership and field are proxies ({pf.info['ownership_version']})")
    print(f"  {sm['entries']} of {sm['requested']} entries"
          f"{' (stopped: ' + sm['stopped'] + ')' if sm['stopped'] else ''}; "
          f"{pf.info['candidates_total']} candidates; {pf.info['n_sims']:,} sims; "
          f"built in {pf.info['seconds']:.1f}s {pf.info['timing']}")
    rel = sm.get("relative")
    print(f"  vs Top {sm['requested']} by projection (same sims and field): "
          + (f"{(rel - 1):+.0%} ± {sm['relative_se']:.0%}" if rel is not None else "n/a"))
    print(f"  any entry cashes {prob_band(sm['p_any_cash'])}; any top-1% finish "
          f"{prob_band(sm['p_any_top1'])}  (simulated, uncalibrated ranges)")
    roi = sm["roi"]
    print(f"  details (simulated, uncalibrated, not real results): E[payout] "
          f"${sm['e_payout']:,.2f} ± {sm['e_payout_se']:.2f} for ${sm['cost']:,.2f}"
          + (f"; ROI {roi:+.1%}" if roi is not None else "")
          + f"; P(any cash) {sm['p_any_cash']:.1%}; P(profit) {sm['p_profit'] or 0:.1%}; "
          f"truth beta {pf.info['truth'].get('beta')} tau {pf.info['truth'].get('tau')}")
    if sm.get("strengths"):
        st = sm["strengths"]
        print("  details: E[payout] by field strength (proxy): " + ", ".join(
            f"{k} ${v['e_payout']:,.0f}" for k, v in st.items()))
    xf = sm.get("xfactor")
    if xf:
        print(f"  X factor players (estimate; reporting only): {xf['lineups_with']} of "
              f"{xf['entries']} lineups include at least one of the slate's {xf['n_players']}"
              + (f" ({', '.join(xf['players_used'])})" if xf["players_used"] else ""))
    cv = sm.get("curve")
    if cv and cv.get("points"):
        print("\n  CHANCES BY NUMBER OF ENTRIES (same simulated nights, nested sets; simulated, "
              "uncalibrated)")
        print(f"  {'N':>4} {'fees':>9} {'in the money':>24} {'top 1%':>22} {'top 0.1%':>22} "
              f"{'1st':>24}  set")
        for r in cv["points"]:
            n_txt = (f"{r['entries']}" if r["entries"] == r["requested"]
                     else f"{r['entries']}/{r['requested']}")
            se = r.get("se") or curve_se(r)

            def cell(k, w):
                t = se_text(r[k], se.get(k))
                return format(chance_text(r[k]) + (f" ({t})" if t else ""), f">{w}")
            print(f"  {n_txt:>4} {'$' + format(r['cost'], ',.0f'):>9} "
                  f"{cell('p_any_cash', 24)} {cell('p_any_top1', 22)} "
                  f"{cell('p_any_top01', 22)} {cell('p_first', 24)}  "
                  f"{r['source']}{' (caps relaxed)' if r.get('caps_relaxed') else ''}")
        print("  (\u00b1 = simulation noise, one standard error across the simulated nights)")
        print(f"  curve built in {cv['seconds']:.1f}s; first k of the build = the k-entry set: "
              f"{'yes' if cv['nested_prefix'] else 'no (exposure caps differ by N)'}")
    print(f"\n  {'#':>3} {'mean':>6} {'cash':>10} {'top 1%':>10} {'dupes':>6}  players")
    for lu in pf.lineups:
        names = ", ".join(x["name"].split()[-1] for x in json.loads(lu["slots_json"]))
        print(f"  {lu['rank']:>3} {lu['mean']:6.1f} {prob_band(lu['p_cash']):>10} "
              f"{prob_band(lu['p_top1']):>10} {lu['dupe_band']:>6}  {names}")
    print("\n  EXPOSURE")
    for r in pf.exposure[:25]:
        print(f"  {r['name'][:24]:<25}{(r['team'] or ''):<5}{r['count']:>4} "
              f"({r['share']:.0%}, cap {r['cap']})  own proxy {r['own_band']}"
              f"{'  locked' if r['locked'] else ''}")
