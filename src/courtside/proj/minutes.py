"""Minutes engine v0.5 (docs/PLAYERS.md §4.2, T1.2): EWMA + rules, scenario-aware, margin-aware.

Pipeline for one team-game (team T, day d, scenario = absent player_ids, optional starters):

1. **State** (strictly games before d): per player, fast/slow EWMAs of regulation-equivalent
   minutes (alpha 0.2 / 0.02; OT minutes scaled by 240/(240+25 n_OT)), last game, median of the
   last 5, started last game, starts in the last 10 appearances (`player_starts`), team games
   missed right now, games since returning from >= 3 missed, PF/36, DK/min; per team: rest/b2b
   and an EWMA of its starters' minutes per position group (G/F/C) shrunk to the league value.
   Candidates: players who appeared for T in its last 10 games and whose last team is T, plus
   `extra` ids (new signings, long absences coming back).
2. **Base** minutes when playing: a linear model on that state (return-from-injury restriction,
   b2b, new season, few games ...), fit on the `window` seasons before S.
   **P(plays | available)** (coach decision / DNP-CD): logistic on the state plus the depth rank
   among *available* players, so the deep bench plays more when regulars sit.
3. **Value-play promotion (S14)**, per absent expected starter j (top 5 by starts in last 10,
   then EWMA), in descending base. Next man up target: 0.5 x the team's starter minutes for j's
   group + 0.5 x his own EWMA.
   - `nmu_mode="softmax"` (default): every available non-starter u gets P(u replaces j) from a
     conditional logit on (same group, secondary group, adjacent group, EWMA, starts in last
     10, his shrunk historical share of j's minutes when j sat, best EWMA in group), fit on
     seasons < S; expected lift = P x (target - base)+. The literal deterministic rule picks
     the top of j's group but is the actual replacement only ~50% of the time and
     over-projects the pick by ~2x (see minutes_eval).
   - `nmu_mode="top"`: the literal §4.2 rule (top of j's group, then adjacent groups, by
     starts in last 10, EWMA, share).
   - Confirmed starters (lock-10): the unexpected starter of j's group (else any) gets the
     full lift; an unexpected starter not matched to an absent starter is lifted as well; an
     expected starter coming off the bench is cut to 0.5 x base + 0.5 x bench target.
   Leftover: `k_add` x the rest of j's minutes goes `f_same` to j's group and the remainder to
   the adjacent groups (G<->F<->C), split by p x minutes x (1 + 4 x share of j's minutes).
4. **Team total** 240 (+ 25 per OT period) minus a fitted reserve for players outside the
   candidate set (call-ups; it grows when few candidates are available). `recon="variance"`
   (default) moves each player's expected minutes in proportion to his variance
   p a (0.2^2 + 1 - p), capped at 40 (+5/OT): deep-bench and DNP-prone players absorb the slack.
   `recon="proportional"` is `players.allocate_minutes` water-filling (the §4.2 text); it
   transmits errors to the starters and is ~0.1 min MAE worse on every test season.
   The conditional mean is expected minutes / p.
5. **Margin.** minutes(m) = mean + g_tier(|m|, won/lost) - c_tier, then re-normalised. g is the
   mean residual by tier x |margin| bin x result on seasons < S (c_tier centres it on the
   training margin mix). `expected_minutes` integrates over m ~ N(spread, 12.4^2)
   (Gauss-Hermite); `sample_minutes` evaluates it per simulated margin.
6. **Noise** (simulator): minutes x exp(mu_tier + sigma_tier xi); early exit with probability
   q0 + q1 x PF/36, scaled by U(0.35, 0.85); DNP-CD with probability 1 - p; then the team
   total per sim (proportional water-filling).

**v1 (T2.2, `MinutesParams.version == "v1"`, `fit(..., version="v1")`).** Steps 1, 3-6 are
kept; step 2's base becomes a LightGBM E[regulation minutes | plays] (L2, refit per season on
the 5 seasons < S, leave-one-season-out early stopping) over the §4.2 groups: role (EWMAs incl.
alpha 0.5, last, median 5, volatility, minutes as starter / off the bench, season mean, starts,
team minutes rank, last game's |margin| and fouls), return (missed now, games since return,
absence length), schedule (rest, b2b, games in the last 3 days, age, team games into the
season, new season / new team), coach (N4: `CoachState`, coach by date from the curated
`COACH_STINTS` + `team_coaches`, coach-level top-7 share and 30+ minute players with
COACH_K = 40 pseudo-games, carried across teams, plus the team's rotation EWMA), his group's
starter target, and the scenario (`scenario_features`: vacated minutes in total / by group / in
his groups / newly absent / of absent expected starters, expected starters out, depth rank after
the removals overall and in his group, rank gain, minutes slack, available rotation size) - the
learned promotion. Stage 2 on top (chosen on out-of-fold training predictions, fixed
population): the softmax next man up scaled by `nmu_scale`, the pair-share term `share_coef` x
sum_j share[i, j] base_j, the leftover `k_add` and the reconciliation `recon_cv`. g and noise
are refit on the OOF projections; the reserve stays the v0.5 model on the v0.5 base
(`V1Base.s_rows`: refit on the GBM base it extrapolated to 25-35 lost team minutes on
opening-night live panels). P(plays | available): the v0.5 logit refit (`pplay_coef_v1`) on
training candidates the live path would have (`departed` removed: live drops moved players, so
they must not teach "new season -> DNP"), with forced extras scored as roster players
(`forced_as_roster`). S15: players
without box history or with < V1_MIN_GP games, and panels without the v1 columns, keep the v0.5
base (`V1_STATS` counts them). News features are not in the GBM (reports exist from
2025-12-22 only).

**Preseason (N2, item #2; `pipeline.PRESEASON`, env COURTSIDE_PRESEASON=1, default OFF).** With
`History.pre` loaded (`attach_preseason`: migration 15 preseason game logs 2010-11+, scrimmages
dropped, games against international clubs kept), the v1 panel gains `V1_PRESEASON` (the row
team's preseason only: games, mpg, last-2-games mpg, share of team minutes, rank, vs prior-season
mpg and vs teammates, did-not-play flag; strictly before the date) and the team-level
`pre_new_tg` (preseason minutes of camp players with no box row yet) and `pre_new_top2` (the
same for the camp's two largest such players). The ON fit
(`fit(..., v1_kw={"features": preseason_features()}, debut_reserve=True)`) adds them to the GBM
(LightGBM learns the decay through gp_season / tgs) and splits the reserve: minutes of players
with no box row before the date (`debut_reserve`: team games into the season x DEBUT_DRIVER,
`pre_new_top2` capped at the training maximum since 2026-10-22, `pre_new_tg` before;
season openers lose ~13 minutes per team to debutants, the season-average reserve gave ~3) and
everybody else outside the candidates (the usual spline, refit without the debutants).
With `History.pre` None (OFF) no column, parameter or JSON key changes.

**Minutes fix (mf; `pipeline.MINFIX`, env COURTSIDE_MINFIX=1, default OFF).** `fit(...,
v1_kw={"minfix": True})` (MF_DEFAULT) changes two fitted parts, both on the 5 seasons < S:
P(plays | available) is a logit on the v1 logit's columns plus the GBM's plan for the night
(MF_PLAN_FEATURES: log plan, log plan / EWMAs / median of 5) and dummies of the depth rank by
that plan (MF_RANKS), fit on the out-of-fold plans of the training panel (`mf_pplay`); and the
noise (mu, sigma per tier) is fit net of the margin curve, then recalibrated by simulation on the
training projections at each game's actual margin so that the simulated minutes of players who
play match the actual minutes on average and their 10-90 % interval covers 80 % (`_mf_calibrate_
noise`). A LightGBM classifier for P(plays) was tested and dropped: it learns the avail oracle's
own rule (available regulars always play), which does not exist live. With `minfix` empty (OFF)
nothing changes: the key is left out of the JSON and no code path reads it.

Everything learned lives in `MinutesParams` (JSON-serialisable; v1 stores the LightGBM model
text), produced by `fit(history, S)` from seasons < S only. Structural constants (EWMA alphas, RECON_CV, SHARE_K, tiers, margin
bins, the 50% shrink) are fixed; RECON_CV and the variance form were chosen on the dev season
2013-14 (fit < 2013-14), before every reported test season.

Public API (consumed by T1.4 copula/pipeline)
  load_history(conn, since="2000-01") -> History            (~8 s, ~0.8 GB peak)
  fit(history, season, window=5, log=print, version="v0.5") -> MinutesParams
                                                              (~40-60 s; v1 +~90 s)
  build_shares(history, seasons) -> PairShares               (pair "minutes when j sat")
  project_team(history, params, team, day, absent=(), starters=None, extra=(), spread=None,
               season=None, shares=None, n_ot=None) -> DataFrame
      one row per candidate: player_id, g1, g2, base, p_play (P plays | available; 0 if
      absent), mean (E[min | plays], regulation, before the margin), exp_min (margin-
      integrated), lift, pstart, promoted, exp_start, pf36, dkpm, reserve ...
  expected_minutes(proj, params, spread=0.0, sd=12.4, n_ot=None) -> np.ndarray
  sample_minutes(proj, params, margins, rng, n_ot=None, noise=True) -> (sims, players) array
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from courtside.model.players import REGULAR_MIN, allocate_minutes

FIRST_SEASON = "2000-01"
ALPHA_FAST, ALPHA_SLOW = 0.2, 0.02
BASE_ALPHA_SHORT, BASE_ALPHA_LONG, BASE_M0 = 0.2, 0.02, 12.0   # rung-2 EWMA (players.TUNED)
TEAM_MIN = 240.0
OT_TEAM_MIN = 25.0
CAP = 40.0
MARGIN_SD = 12.4
ROTATION_MIN = 12.0
WINDOW_GAMES = 10
TARGET_ALPHA, TARGET_K = 0.05, 10.0
RECON_CV = 0.2          # reconciliation: Var_i ~ p a (cv^2 + 1 - p), frozen on dev 2013-14
SHARE_K = 3.0           # pseudo-games shrinking a pair's "minutes gained when j sat" share
GROUPS = ("G", "F", "C")
ADJ = {0: (1,), 1: (0, 2), 2: (1,)}
TIER_EDGES = (12.0, 20.0, 26.0, 32.0)          # 5 tiers of the projected mean
MARGIN_BINS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 200.0)
EXIT_U = (0.35, 0.85)
DK_COLS = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")

BASE_FEATURES = ("one", "ew_f", "ew_s", "med5", "last", "st10", "start_last", "st10_ewf",
                 "ret0", "ret0_ews", "ret12_ews", "short_miss", "b2b", "b2b_ewf", "new_season",
                 "new_season_ews", "new_season_ewf", "inv_gp", "inv_gp_ewf", "early_season")
PPLAY_FEATURES = ("one", "ew_f", "ew_s", "app10", "missed_now_c", "ret0", "new_season",
                  "log_gp", "st10", "last", "rk_le7", "rk8", "rk9", "rk10", "rk11", "rk12",
                  "n_avail")


def tier_of(x) -> np.ndarray:
    return np.searchsorted(np.asarray(TIER_EDGES), np.asarray(x, float), side="right")


# --- params ----------------------------------------------------------------------------------

@dataclass
class MinutesParams:
    season: str                       # S: fit on seasons < S
    train_seasons: list
    base_coef: list
    pplay_coef: list
    starter_league: list              # league starter minutes per group (G, F, C)
    bench_target: float               # league minutes of the top bench player
    f_same: float                     # share of leftover vacated minutes kept in j's group
    g_table: list                     # [tier][result 0=lost 1=won][margin bin] (minutes)
    g_center: list                    # per tier: E_train[g]
    mu: list                          # per tier: mean log ratio
    sigma: list                       # per tier
    q0: float
    q1: float
    mean_n_ot: float
    reserve_coef: list = field(default_factory=lambda: [0.8, 0.0, 0.0, 0.0])
    share_weight: float = 20.0        # "top" next-man-up rank: EWMA minutes per unit of share
    k_add: float = 0.6                # share of leftover vacated minutes targeted explicitly
    recon: str = "variance"           # "variance" or "proportional" (players.allocate_minutes)
    nmu_mode: str = "softmax"         # "softmax" (fit) or "top" (literal deterministic rule)
    nmu_coef: list = field(default_factory=lambda: [2.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0])
    base_features: list = field(default_factory=lambda: list(BASE_FEATURES))
    pplay_features: list = field(default_factory=lambda: list(PPLAY_FEATURES))
    diagnostics: dict = field(default_factory=dict)
    # --- v1 (T2.2). version "v0.5" ignores everything below.
    version: str = "v0.5"             # "v0.5" (linear base + rules) or "v1" (LightGBM base)
    gbm_model: str = ""               # LightGBM model text (E[min | plays] before the 240)
    gbm_features: list = field(default_factory=list)
    nmu_scale: float = 1.0            # v1: share of the softmax next-man-up lift kept on top
    share_coef: float = 0.0           # v1: minutes per (share x vacated minutes) on top
    recon_cv: float = RECON_CV        # reconciliation variance form (v0.5: frozen 0.2)
    pplay_coef_v1: list = field(default_factory=list)   # v1 P(plays | available), see fit_v1
    # season-start debut reserve (COURTSIDE_PRESEASON, `fit(debut_reserve=True)`): minutes of
    # players with no box row before the date, by team games into the season and the team's
    # debutants' preseason minutes. Empty = off (and the key is left out of the JSON).
    reserve_debut: list = field(default_factory=list)
    # minutes fix (COURTSIDE_MINFIX, `fit(minfix=True)`): P(plays | available) that reads the
    # GBM plan, the noise fit net of the margin curve (see MF_* below). Empty = off (and the key
    # is left out of the JSON).
    minfix: dict = field(default_factory=dict)
    # blowout minutes (COURTSIDE_BLOWOUT, `fit(v1_kw={"blowout": True})`): P(plays) offsets,
    # reserve offsets and the margin curve calibrated by simulation (see BO_* below). Empty =
    # off (and the key is left out of the JSON).
    blowout: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        if not d["reserve_debut"]:          # absent key: the format before it, byte for byte
            del d["reserve_debut"]
        if not d["minfix"]:                 # likewise
            del d["minfix"]
        if not d["blowout"]:                # likewise
            del d["blowout"]
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "MinutesParams":
        return cls(**json.loads(s))


def default_params(season: str = "2099-00") -> MinutesParams:
    """Uninformative params (tests / cold start): base = blend of the EWMAs."""
    coef = np.zeros(len(BASE_FEATURES))
    coef[BASE_FEATURES.index("ew_f")] = 0.5
    coef[BASE_FEATURES.index("ew_s")] = 0.5
    pc = np.zeros(len(PPLAY_FEATURES))
    pc[0] = -3.0
    pc[PPLAY_FEATURES.index("ew_f")] = 0.3
    pc[PPLAY_FEATURES.index("rk_le7")] = 3.0
    nt = len(TIER_EDGES) + 1
    nb = len(MARGIN_BINS) - 1
    return MinutesParams(season=season, train_seasons=[], base_coef=coef.tolist(),
                         pplay_coef=pc.tolist(), starter_league=[32.0, 31.0, 28.0],
                         bench_target=24.0, f_same=0.6,
                         g_table=np.zeros((nt, 2, nb)).tolist(), g_center=[0.0] * nt,
                         mu=[0.0] * nt, sigma=[0.25] * nt, q0=0.02, q1=0.0, mean_n_ot=0.06)


# --- data ------------------------------------------------------------------------------------

@dataclass
class History:
    hist: pd.DataFrame        # one row per player-game (min > 0), post-game state, sorted
    sched: dict               # team -> {"date": int days, "season": array, "gid": array}
    targets: pd.DataFrame     # team-game starter targets per group (post-game EWMA)
    positions: pd.DataFrame   # player_id, season, g1, g2 (season NaN = snapshot)
    tg: pd.DataFrame          # team-games (game_id, team, date, season, season_type, margin, n_ot)
    # v1 extras (optional; absent -> the v1 coach/age features are missing, not wrong)
    coach: "CoachState | None" = None
    births: pd.DataFrame | None = None    # player_id, birth (int days)
    shares: "PairShares | None" = None    # pair shares for the v1 share features (optional)
    # preseason game logs (COURTSIDE_PRESEASON, `load_preseason`); None -> no pre_* columns
    pre: "Preseason | None" = None


def _days(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]").astype(np.int64)


def _grp(pos) -> tuple[int, int]:
    if not isinstance(pos, str) or not pos:
        return -1, -1
    parts = [p[:1].upper() for p in pos.replace("/", "-").split("-") if p]
    gs = [GROUPS.index(p) for p in parts if p in GROUPS]
    if not gs:
        return -1, -1
    return gs[0], (gs[1] if len(gs) > 1 else gs[0])


def load_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """player_id, season (None = player_index snapshot), g1, g2 (0=G 1=F 2=C)."""
    tr = pd.read_sql_query("SELECT player_id, season, position FROM team_rosters", conn)
    pi = pd.read_sql_query("SELECT player_id, position FROM player_index", conn)
    pi["season"] = None
    df = pd.concat([tr, pi], ignore_index=True)
    g = np.array([_grp(p) for p in df["position"]], dtype=int).reshape(-1, 2)
    df["g1"], df["g2"] = g[:, 0], g[:, 1]
    df = df[df["g1"] >= 0].drop_duplicates(["player_id", "season"])
    return df[["player_id", "season", "g1", "g2"]].reset_index(drop=True)


def load_box(conn: sqlite3.Connection, since: str = FIRST_SEASON
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player rows with min > 0 + started flag, team-games with margin and n_ot)."""
    pg = pd.read_sql_query(
        "SELECT game_id, player_id, team, season, season_type, game_date, min, pf, "
        + ", ".join(DK_COLS) + " FROM player_games WHERE season >= ? AND min > 0",
        conn, params=(since,))
    st = pd.read_sql_query("SELECT game_id, player_id FROM player_starts", conn)
    st["started"] = 1
    pg = pg.merge(st, on=["game_id", "player_id"], how="left")
    pg["started"] = pg["started"].fillna(0).astype(np.int8)
    tg = pd.read_sql_query(
        "SELECT game_id, team, season, season_type, game_date, min, pts FROM team_games "
        "WHERE season >= ?", conn, params=(since,))
    return pg, tg


def _team_games(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    opp = tg[["game_id", "team", "pts"]].rename(columns={"team": "opp", "pts": "opp_pts"})
    tg = tg.merge(opp, on="game_id")
    tg = tg[tg["team"] != tg["opp"]].drop_duplicates(["game_id", "team"])
    tg["margin"] = (tg["pts"] - tg["opp_pts"]).astype(float)
    tm = tg["min"].astype(float)
    tg["n_ot"] = np.clip(np.round((tm.where(tm > 0, TEAM_MIN) - TEAM_MIN) / OT_TEAM_MIN), 0, 9)
    tg["date"] = _days(tg["game_date"])
    tg = tg.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    tg["tgi"] = tg.groupby("team").cumcount()
    return tg[["game_id", "team", "opp", "season", "season_type", "date", "tgi", "margin",
               "n_ot"]]


def _ewm(s: pd.Series, by, alpha: float) -> pd.Series:
    return s.groupby(by, sort=False).transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean())


def player_history(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Post-game state per player-game (each row uses only that player's rows up to it)."""
    h = pg.merge(tg[["game_id", "team", "date", "tgi", "n_ot", "margin"]],
                 on=["game_id", "team"], how="inner")
    h = h.sort_values(["player_id", "date", "game_id"], kind="mergesort").reset_index(drop=True)
    pid = h["player_id"].to_numpy()
    by = h["player_id"]
    m = h["min"].astype(float)
    h["min_r"] = m * TEAM_MIN / (TEAM_MIN + OT_TEAM_MIN * h["n_ot"])
    h["ew_f"] = _ewm(h["min_r"], by, ALPHA_FAST)
    h["ew_s"] = _ewm(h["min_r"], by, ALPHA_SLOW)
    h["med5"] = h.groupby("player_id", sort=False)["min_r"].transform(
        lambda x: x.rolling(5, min_periods=1).median())
    h["st10"] = h.groupby("player_id", sort=False)["started"].transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    pf_e = _ewm(h["pf"].astype(float), by, 0.05)
    mn_e = _ewm(m, by, 0.05)
    h["pf36"] = 36.0 * pf_e / mn_e.clip(lower=1.0)
    d = h[list(DK_COLS)].astype(float)
    cats = (d[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    dk = (d.pts + 0.5 * d.fg3m + 1.25 * d.reb + 1.5 * d.ast + 2 * d.stl + 2 * d.blk
          - 0.5 * d.tov + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
    h["dk"] = dk
    h["dkpm"] = _ewm(dk, by, 0.05) / mn_e.clip(lower=1.0)
    # rung-2 baseline EWMA: init m0, adjust=False, raw minutes, blend 0.5/0.5
    first = np.r_[True, pid[1:] != pid[:-1]]
    n = np.arange(len(h)) - np.maximum.accumulate(np.where(first, np.arange(len(h)), 0)) + 1
    h["n_gp"] = n
    a = _ewm_from(m.to_numpy(), first, BASE_ALPHA_SHORT, BASE_M0)
    b = _ewm_from(m.to_numpy(), first, BASE_ALPHA_LONG, BASE_M0)
    h["base_ew"] = 0.5 * a + 0.5 * b
    h["season_gp"] = h.groupby(["player_id", "season"], sort=False).cumcount() + 1
    # missed team games before this one (same team schedule), games since return
    prev_date = h.groupby("player_id", sort=False)["date"].shift(1)
    missed = np.zeros(len(h))
    for team, idx in h.groupby("team", sort=False).indices.items():
        tdates = tg.loc[tg["team"] == team, "date"].to_numpy()
        pdt = prev_date.to_numpy()[idx]
        ok = ~np.isnan(pdt)
        cnt = np.searchsorted(tdates, np.where(ok, pdt, -1e18), side="right")
        missed[idx] = np.where(ok, h["tgi"].to_numpy()[idx] - cnt, 0)
    h["missed_before"] = np.maximum(missed, 0)
    ret = (h["missed_before"] >= 3).astype(int)
    seg = ret.groupby(by, sort=False).cumsum()
    pre = h.groupby([h["player_id"], seg], sort=False).cumcount()
    pre = np.where(seg > 0, np.minimum(pre, 10), 10)
    h["gsr_post"] = np.minimum(pre + 1, 10)
    # v1 role/return state (post-game, same rows only)
    mr = h["min_r"].to_numpy(float)
    h["ew_vf"] = _ewm_from(mr, first, 0.5, np.nan)
    dev = (h["min_r"] - h["ew_s"]) ** 2
    h["min_sd"] = np.sqrt(_ewm_from(dev.to_numpy(float), first, 0.1, np.nan))
    stt = h["started"].to_numpy() == 1
    h["ew_start"] = _ewm_masked(mr, first, stt, 0.2)
    h["ew_bench"] = _ewm_masked(mr, first, ~stt, 0.2)
    gs = h.groupby(["player_id", "season"], sort=False)["min_r"]
    h["sea_mean"] = gs.cumsum() / (gs.cumcount() + 1)
    ab = np.where(h["missed_before"].to_numpy() >= 3, h["missed_before"].to_numpy(float), np.nan)
    h["abs_len"] = pd.Series(ab).groupby(pid).ffill().fillna(0.0).to_numpy()
    keep = ["player_id", "game_id", "team", "season", "season_type", "date", "tgi", "min",
            "min_r", "started", "ew_f", "ew_s", "med5", "st10", "pf36", "dk", "dkpm", "n_gp",
            "base_ew", "season_gp", "missed_before", "gsr_post", "n_ot", "ew_vf", "min_sd",
            "ew_start", "ew_bench", "sea_mean", "abs_len", "last_absm", "last_pf"]
    h["last_absm"] = np.abs(h["margin"].to_numpy(float))
    h["last_pf"] = h["pf"].astype(float)
    out = h[keep].copy()
    for c in ("min", "min_r", "ew_f", "ew_s", "med5", "st10", "pf36", "dk", "dkpm", "base_ew",
              "ew_vf", "min_sd", "ew_start", "ew_bench", "sea_mean", "abs_len"):
        out[c] = out[c].astype(np.float32)
    return out


def _ewm_from(x: np.ndarray, first: np.ndarray, alpha: float, init: float) -> np.ndarray:
    """EWMA with y_-1 = init per segment (adjust=False), segments start where first is True."""
    out = np.empty(len(x))
    y = init
    for i in range(len(x)):
        if first[i]:
            y = init
        y = x[i] if y != y else y + alpha * (x[i] - y)    # NaN init: start at the first value
        out[i] = y
    return out


def _ewm_masked(x: np.ndarray, first: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA over the rows where `mask` holds, carried forward on the others (NaN before the
    first masked row of a segment)."""
    out = np.empty(len(x))
    y = np.nan
    for i in range(len(x)):
        if first[i]:
            y = np.nan
        if mask[i]:
            y = x[i] if y != y else y + alpha * (x[i] - y)
        out[i] = y
    return out


def _starter_targets(hist, tg, positions):
    st = hist[hist["started"] == 1][["game_id", "team", "player_id", "season", "min_r"]]
    if positions is not None:
        st = _attach_groups(st, positions)
    rows = st.groupby(["game_id", "team", "g1"])["min_r"].mean().unstack("g1")
    rows = rows.reindex(columns=range(3))
    t = tg[["game_id", "team", "date", "tgi"]].merge(
        rows.reset_index(), on=["game_id", "team"], how="left")
    t = t.sort_values(["team", "tgi"]).reset_index(drop=True)
    for g in range(3):
        x = t[g].astype(float)
        has = x.notna().astype(float)
        num = (x.fillna(0.0)).groupby(t["team"]).transform(
            lambda s: _ewsum(s.to_numpy(), 1 - TARGET_ALPHA))
        den = has.groupby(t["team"]).transform(lambda s: _ewsum(s.to_numpy(), 1 - TARGET_ALPHA))
        t[f"tgt_num{g}"] = num
        t[f"tgt_den{g}"] = den
    return t[["team", "date", "tgi"] + [f"tgt_num{g}" for g in range(3)]
             + [f"tgt_den{g}" for g in range(3)]]


def _ewsum(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty(len(x))
    s = 0.0
    for i, v in enumerate(x):
        s = s * decay + v
        out[i] = s
    return out


def _attach_groups(df: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Add g1/g2 by (player, season): that season's roster position, else the latest earlier
    roster season, else the player_index snapshot, else F."""
    pos_s = positions[positions["season"].notna()].sort_values(["player_id", "season"])
    snap = positions[positions["season"].isna()].set_index("player_id")[["g1", "g2"]]
    key = df[["player_id", "season"]].drop_duplicates()
    m = key.merge(pos_s, on=["player_id", "season"], how="left")
    miss = m["g1"].isna()
    if miss.any() and len(pos_s):
        # latest earlier season (seasons sort lexicographically: 'YYYY-YY')
        sub = m.loc[miss, ["player_id", "season"]].copy()
        cand = sub.merge(pos_s, on="player_id", suffixes=("", "_r"))
        cand = cand[cand["season_r"] < cand["season"]].sort_values("season_r")
        cand = cand.drop_duplicates(["player_id", "season"], keep="last")
        m = m.merge(cand[["player_id", "season", "g1", "g2"]], on=["player_id", "season"],
                    how="left", suffixes=("", "_e"))
        m["g1"] = m["g1"].fillna(m["g1_e"])
        m["g2"] = m["g2"].fillna(m["g2_e"])
        m = m.drop(columns=["g1_e", "g2_e"])
    miss = m["g1"].isna()
    if miss.any():
        s = snap.reindex(m.loc[miss, "player_id"].to_numpy())
        m.loc[miss, "g1"] = s["g1"].to_numpy()
        m.loc[miss, "g2"] = s["g2"].to_numpy()
    m["g1"] = m["g1"].fillna(1).astype(int)
    m["g2"] = m["g2"].fillna(m["g1"]).astype(int)
    return df.merge(m[["player_id", "season", "g1", "g2"]], on=["player_id", "season"],
                    how="left")


def build_history(pg: pd.DataFrame, tg_raw: pd.DataFrame, positions: pd.DataFrame,
                  stints: pd.DataFrame | None = None, births: pd.DataFrame | None = None
                  ) -> History:
    tg = _team_games(tg_raw)
    hist = player_history(pg, tg)
    targets = _starter_targets(hist, tg, positions)
    sched = {t: {"date": d["date"].to_numpy(), "season": d["season"].to_numpy(),
                 "gid": d["game_id"].to_numpy()}
             for t, d in tg.groupby("team", sort=False)}
    coach = CoachState.build(hist, tg, stints) if stints is not None else None
    return History(hist=hist, sched=sched, targets=targets, positions=positions, tg=tg,
                   coach=coach, births=births)


def load_births(conn: sqlite3.Connection) -> pd.DataFrame:
    """player_id, birth (int days) from team_rosters (any season)."""
    b = pd.read_sql_query("SELECT player_id, MIN(birth_date) birth FROM team_rosters "
                          "WHERE birth_date IS NOT NULL GROUP BY player_id", conn)
    d = pd.to_datetime(b["birth"], errors="coerce")
    b = b[d.notna()].copy()
    b["birth"] = _days(d[d.notna()])
    return b


def load_history(conn: sqlite3.Connection, since: str = FIRST_SEASON) -> History:
    pg, tg = load_box(conn, since)
    try:
        stints = coach_stints(load_coach_snapshots(conn))
        births = load_births(conn)
    except Exception:  # pragma: no cover - old DBs without team_coaches / birth dates
        stints, births = None, None
    return build_history(pg, tg, load_positions(conn), stints=stints, births=births)


# --- preseason (N2, item #2; COURTSIDE_PRESEASON) ---------------------------------------------
#
# Source: preseason_player_games / preseason_team_games (migration 15; stats.nba.com game logs
# 2010-11+, whole minutes, no starter flag). Used: NBA teams' preseason games, INCLUDING games
# against international clubs (vs_non_nba: ~6% of rows; they are part of the team's camp
# rotation, and the per-team-game share normalises their length), EXCLUDING `scrimmage` rows
# (the July 2020 bubble scrimmages filed under 2019-20: 40-minute games ~65 games into the
# season, not the camp before an opener). Every feature uses preseason games dated strictly
# before the projected date (the preseason ends before the opener, so in practice the whole
# preseason of the row's season), and the prior-season mpg uses regular seasons < S only.

class Preseason:
    """Preseason game logs: `players` (season, player_id, team, game_id, date int days, min) and
    `teams` (season, team, game_id, date, min) of NBA teams, scrimmages removed."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame):
        self.players = players.reset_index(drop=True)
        self.teams = teams.reset_index(drop=True)

    def truncate(self, d: int) -> "Preseason":
        return Preseason(self.players[self.players["date"] < d], self.teams[self.teams["date"] < d])


def load_preseason(conn: sqlite3.Connection, since: str = FIRST_SEASON) -> Preseason:
    """Preseason logs of seasons >= since (NBA teams only, `scrimmage` = 0). Empty frames when
    the tables do not exist (DB before migration 15)."""
    nba = "(SELECT DISTINCT team FROM team_games)"
    try:
        pl = pd.read_sql_query(
            "SELECT season, player_id, team, game_id, game_date, min FROM preseason_player_games "
            f"WHERE season >= ? AND scrimmage = 0 AND team IN {nba}", conn, params=(since,))
        tm = pd.read_sql_query(
            "SELECT season, team, game_id, game_date, min FROM preseason_team_games "
            f"WHERE season >= ? AND scrimmage = 0 AND team IN {nba}", conn, params=(since,))
    except Exception:  # pragma: no cover - no migration 15
        pl = pd.DataFrame(columns=["season", "player_id", "team", "game_id", "game_date", "min"])
        tm = pd.DataFrame(columns=["season", "team", "game_id", "game_date", "min"])
    for d in (pl, tm):
        d["date"] = _days(d["game_date"]) if len(d) else np.zeros(0, np.int64)
        d.drop(columns=["game_date"], inplace=True)
        d["min"] = pd.to_numeric(d["min"], errors="coerce").fillna(0.0).astype(float)
    pl["player_id"] = pl["player_id"].astype(np.int64)
    tm["min"] = tm["min"].where(tm["min"] > 0, TEAM_MIN)
    return Preseason(pl, tm)


def attach_preseason(h: History, conn: sqlite3.Connection, since: str | None = None) -> History:
    """`h` with its preseason logs (h.pre) loaded from the seasons of its box history on."""
    if since is None:
        since = str(h.tg["season"].min()) if len(h.tg) else FIRST_SEASON
    h.pre = load_preseason(conn, since)
    return h


V1_PRESEASON = ("pre_gp", "pre_mpg", "pre_mpg_l2", "pre_l2_gp", "pre_share", "pre_rank",
                "pre_rel_prior", "pre_rel_team", "pre_none")
# team-level (same on every row of a team-game; input of the debut reserve, not a GBM feature):
# preseason minutes per team preseason game of the team's camp players with no box row before
# the date (rookies / new to the league: never minutes-engine candidates)
# pre_new_top2: the same for the team's two largest such players only (the debut reserve's driver
# from 2026-10-22: bounded by construction, see DEBUT_DRIVER)
PRE_TEAM_COLS = ("pre_new_tg", "pre_new_top2")
DEBUT_BINS = (0, 1, 5, 15)      # team games into the season: {0}, {1-4}, {5-14}, {15+}
PRE_MAX_TGS = 20     # V1_PRESEASON set to NaN from this many team games on (None: never)


def preseason_features() -> list:
    """The v1 GBM feature list of the preseason arm (COURTSIDE_PRESEASON)."""
    return list(V1_FEATURES) + list(V1_PRESEASON)


def uses_preseason(params: "MinutesParams") -> bool:
    return any(f in V1_PRESEASON for f in (params.gbm_features or ()))


def _preseason_agg(pl: pd.DataFrame, tm: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Per (player, team) of one season's (already cut) preseason: games played for that team,
    mpg, share of the team's preseason minutes x 5 (DNP = 0), mean minutes in the team's last 2
    games (DNP = 0) and games played among them, rank on the team by share, mpg minus the team's
    median mpg (players with >= 2 games). Also the set of teams with games. Only the row team's
    games count, so a player who changed teams (a departed player in a historical panel built
    from last season's roster) reads as `pre_none`, not as his new team's role."""
    cols = ["player_id", "team", "pre_gp", "pre_mpg", "pre_share", "pre_mpg_l2", "pre_l2_gp",
            "pre_rank", "pre_rel_team"]
    teams = set(tm["team"].unique())
    pl = pl[pl["min"] > 0]
    if not len(pl) or not len(tm):
        return pd.DataFrame(columns=cols), teams
    tm = tm.sort_values(["team", "date", "game_id"], kind="mergesort")
    tmin = tm.groupby("team")["min"].sum()
    last2 = tm.groupby("team", sort=False).tail(2)
    l2_n = last2.groupby("team").size()
    l2 = set(zip(last2["team"], last2["game_id"]))
    g = pl.groupby(["player_id", "team"])["min"]
    a = pd.DataFrame({"pre_gp": g.size().astype(float), "pre_mpg": g.mean(), "tot": g.sum()})
    a = a.reset_index()
    a = a[a["team"].isin(teams)]
    a["pre_share"] = 5.0 * a["tot"] / tmin.reindex(a["team"]).to_numpy()
    inl2 = np.array([(t, gid) in l2 for t, gid in zip(pl["team"], pl["game_id"])], bool)
    ql2 = pl[inl2].groupby(["player_id", "team"])["min"]
    idx = pd.MultiIndex.from_frame(a[["player_id", "team"]])
    a["pre_l2_gp"] = ql2.size().reindex(idx).fillna(0.0).to_numpy(float)
    a["pre_mpg_l2"] = (ql2.sum().reindex(idx).fillna(0.0).to_numpy(float)
                       / l2_n.reindex(a["team"]).to_numpy())
    a["pre_rank"] = a.groupby("team")["pre_share"].rank(ascending=False, method="min")
    med = a[a["pre_gp"] >= 2].groupby("team")["pre_mpg"].median()
    a["pre_rel_team"] = a["pre_mpg"] - med.reindex(a["team"]).to_numpy()
    return a[cols].reset_index(drop=True), teams


def prior_season_mpg(h: History, seasons: Iterable[str] = ()) -> pd.DataFrame:
    """player_id, season S, prev_mpg = regular-season mean regulation minutes in his latest
    season < S, for every season S in h.tg and `seasons` (earlier seasons only)."""
    r = h.hist[h.hist["season_type"] == "Regular Season"]
    m = r.groupby(["player_id", "season"])["min_r"].mean().reset_index()
    m["y"] = m["season"].str[:4].astype(int)
    frames = []
    for s in sorted(set(h.tg["season"]) | set(seasons)):
        q = m[m["y"] < int(s[:4])].sort_values("y", kind="mergesort")
        q = q.groupby("player_id", sort=False).tail(1)
        frames.append(pd.DataFrame({"player_id": q["player_id"].to_numpy(np.int64),
                                    "season": s, "prev_mpg": q["min_r"].to_numpy(float)}))
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "prev_mpg"])
    return pd.concat(frames, ignore_index=True)


def preseason_columns(h: History, p: pd.DataFrame) -> pd.DataFrame:
    """Add V1_PRESEASON and PRE_TEAM_COLS to a panel (rows: player_id, team, season, date) from
    h.pre: the aggregates of the row's team's preseason games of the row's season dated strictly
    before the row's date. pre_none = 1 when the row's team has preseason games but the player
    played in none of them (injury, rest, late signing, a different team), 0 when he did, NaN
    when the team has none (no data). pre_rel_prior = pre_mpg / max(prior-season mpg, 5)."""
    pre = h.pre
    p = p.copy()
    n = len(p)
    out = {c: np.full(n, np.nan) for c in V1_PRESEASON + PRE_TEAM_COLS}
    if n and pre is not None and len(pre.teams):
        dates = p["date"].to_numpy(np.int64)
        seas = p["season"].to_numpy(object)
        pids = p["player_id"].to_numpy(np.int64)
        tms = p["team"].to_numpy(object)
        tsea = {s: d for s, d in pre.teams.groupby("season", sort=False)}
        psea = {s: d for s, d in pre.players.groupby("season", sort=False)}
        for s in pd.unique(seas):
            if s not in tsea:
                continue
            T, Pl = tsea[s], psea.get(s, pre.players.iloc[:0])
            end = int(T["date"].max()) + 1
            rows = np.flatnonzero(seas == s)
            cut = np.minimum(dates[rows], end)
            for c in np.unique(cut):
                ri = rows[cut == c]
                a, teams = _preseason_agg(Pl[Pl["date"] < c], T[T["date"] < c])
                out["pre_new_tg"][ri], out["pre_new_top2"][ri] = _new_player_minutes(
                    h, Pl[Pl["date"] < c], T[T["date"] < c], p.iloc[ri])
                a = a.set_index(["player_id", "team"])
                key = pd.MultiIndex.from_arrays([pids[ri], tms[ri]])
                has = key.isin(a.index)
                team_has = np.array([t in teams for t in tms[ri]], bool)
                out["pre_none"][ri] = np.where(has, 0.0, np.where(team_has, 1.0, np.nan))
                out["pre_gp"][ri] = np.where(has, np.nan, np.where(team_has, 0.0, np.nan))
                if not has.any():
                    continue
                sub = a.reindex(key[has])
                for col in ("pre_gp", "pre_mpg", "pre_share", "pre_mpg_l2", "pre_l2_gp",
                            "pre_rank", "pre_rel_team"):
                    out[col][ri[has]] = sub[col].to_numpy(float)
        if np.isfinite(out["pre_mpg"]).any():
            pm = prior_season_mpg(h, pd.unique(seas))
            q = p[["player_id", "season"]].merge(pm, on=["player_id", "season"], how="left")
            out["pre_rel_prior"] = out["pre_mpg"] / np.maximum(q["prev_mpg"].to_numpy(float), 5.0)
    if PRE_MAX_TGS is not None and "tgs" in p:
        late = p["tgs"].to_numpy(float) >= PRE_MAX_TGS
        for c in V1_PRESEASON:
            out[c][late] = np.nan
    for c in V1_PRESEASON + PRE_TEAM_COLS:
        p[c] = out[c]
    return p


def _new_player_minutes(h: History, pl: pd.DataFrame, tm: pd.DataFrame, rows: pd.DataFrame
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per row (team, date): sum over the team's preseason players with no box row strictly
    before the date of their preseason minutes / the team's preseason games (DNP = 0), and the
    same over the two largest of them. NaN for a team without preseason games."""
    out = np.full(len(rows), np.nan)
    top = np.full(len(rows), np.nan)
    if not len(tm):
        return out, top
    n_tg = tm.groupby("team").size()
    q = pl[pl["min"] > 0].groupby(["team", "player_id"])["min"].sum().reset_index()
    q["mtg"] = q["min"] / n_tg.reindex(q["team"]).to_numpy()
    first = h.hist.groupby("player_id")["date"].min()
    q["first"] = first.reindex(q["player_id"]).to_numpy(float)     # NaN = never in the box
    keys = rows[["team", "date"]].drop_duplicates()
    m = keys.merge(q[["team", "mtg", "first"]], on="team", how="left")
    new = ~(m["first"] < m["date"])                    # no box row before the date (NaN first)
    m["v"] = np.where(new & m["mtg"].notna(), m["mtg"], 0.0)
    s = m.groupby(["team", "date"])["v"].sum()
    s2 = m.sort_values("v", ascending=False, kind="mergesort").groupby(
        ["team", "date"])["v"].apply(lambda x: float(x.iloc[:2].sum()))
    has = set(n_tg.index)
    for i, (t, d) in enumerate(zip(rows["team"], rows["date"])):
        if t in has:
            out[i] = float(s.get((t, d), 0.0))
            top[i] = float(s2.get((t, d), 0.0))
    return out, top


DEBUT_D_Q = 0.95                # the new players' preseason minutes are winsorised at this
DEBUT_MAX = 40.0                # training quantile (stored in reserve_debut); reserve cap
# Driver of the debut reserve (2026-10-22, docs/PLAYERS.md minutes-fix results): the camp's two
# largest new players' preseason minutes per team game, capped at the training maximum. The sum
# over all new players (`pre_new_tg`, the first design) grows with camp invitees who never play
# (OKC 2026-27: 126 vs a training p95 of 70); two players cannot, and they predict the debut
# minutes as well or better leave-one-season-out (game-0 MAE 11.0 vs 11.2 over 2019-26, 11.2 vs
# 12.2 on the 28 openers of 2026-27). reserve_debut stores the driver code after the cap:
# 2 = top-2 (absent: the sum, the first design's params read unchanged).
DEBUT_DRIVER = "pre_new_top2"
DEBUT_DRIVER_Q = 1.0
_DEBUT_CODES = {2.0: "pre_new_top2"}


def _debut_design(tgs, d, cap: float | None = None) -> np.ndarray:
    """Columns: one-hot of DEBUT_BINS (team games into the season) and the same x the team's
    new players' preseason minutes `d` (NaN -> 0; capped at `cap`: a camp that rests its
    regulars gives huge values, e.g. OKC 2026-27 126/game vs a training p95 of ~60)."""
    tgs = np.asarray(tgs, float)
    d = np.nan_to_num(np.asarray(d, float))
    if cap is not None:
        d = np.minimum(d, cap)
    b = np.clip(np.searchsorted(np.asarray(DEBUT_BINS), tgs, side="right") - 1, 0,
                len(DEBUT_BINS) - 1)
    oh = np.eye(len(DEBUT_BINS))[b]
    return np.column_stack([oh, oh * d[:, None]])


def debut_reserve(params: MinutesParams, p: pd.DataFrame) -> np.ndarray:
    """Per row: team minutes expected to go to players with no box row before the date
    (params.reserve_debut = coefficients of `_debut_design` + the cap of its input [+ the driver
    code, DEBUT_DRIVER; absent = the first design's `pre_new_tg`]; clipped to [0, DEBUT_MAX]).
    Rows without `tgs` get 0."""
    if not params.reserve_debut or "tgs" not in p or not len(p):
        return np.zeros(len(p))
    k = 2 * len(DEBUT_BINS)
    coef = np.asarray(params.reserve_debut[:k], float)
    cap = params.reserve_debut[k] if len(params.reserve_debut) > k else None
    col = (_DEBUT_CODES.get(float(params.reserve_debut[k + 1]), "pre_new_tg")
           if len(params.reserve_debut) > k + 1 else "pre_new_tg")
    d = p[col].to_numpy(float) if col in p else np.zeros(len(p))
    return np.clip(_debut_design(p["tgs"].to_numpy(float), d, cap) @ coef, 0.0, DEBUT_MAX)


def debut_minutes(h: History, panel: pd.DataFrame) -> pd.Series:
    """Per panel key (game_id|team): regulation minutes of box rows of that team-game by players
    with no box row before the game's date (the debut reserve's training target)."""
    keys = panel[["key", "date"]].drop_duplicates("key")
    gt = keys["key"].str.split("|", n=1, expand=True)
    keys = keys.assign(game_id=gt[0].to_numpy(), team=gt[1].to_numpy())
    hh = h.hist[["game_id", "team", "player_id", "date", "min_r"]].merge(
        keys[["game_id", "team", "key"]], on=["game_id", "team"])
    first = h.hist.groupby("player_id")["date"].min()
    deb = first.reindex(hh["player_id"]).to_numpy() >= hh["date"].to_numpy()
    s = hh[deb].groupby("key")["min_r"].sum()
    return s.reindex(keys["key"]).fillna(0.0).set_axis(keys["key"].to_numpy())


def truncate(h: History, day) -> History:
    """History restricted to games strictly before `day` (for no-lookahead checks)."""
    d = int(np.datetime64(pd.Timestamp(day).date(), "D").astype(np.int64))
    hist = h.hist[h.hist["date"] < d]
    tg = h.tg[h.tg["date"] < d]
    sched = {t: {k: v[v_ok] for k, v in s.items()}
             for t, s in h.sched.items() for v_ok in [s["date"] < d]}
    return History(hist=hist, sched=sched, targets=h.targets[h.targets["date"] < d],
                   positions=h.positions, tg=tg,
                   coach=h.coach.truncate(d) if h.coach is not None else None,
                   births=h.births, shares=h.shares,
                   pre=h.pre.truncate(d) if h.pre is not None else None)


# --- coach by date (N4) ----------------------------------------------------------------------
#
# Source. `team_coaches` (commonteamroster) has one staff snapshot per (season, team) from
# 2015-16, and the snapshot labelled S is the staff of the summer AFTER season S (2018-19 lists
# Beilein/CLE, Vogel/LAL, Jenkins/MEM; 2024-25 lists Ott/PHX, hired for 2025-26). It has no
# dates, so mid-season changes cannot be read from the DB. COACH_STINTS below is the head
# coach by date for 2015-16 -> 2025-26, curated from the public record (first day the coach ran
# the team; interim coaches included when they coached a meaningful stretch; dates are the
# firing / hiring day, +-1-2 days). 2025-26 in-season changes other than POR are not curated
# (unknown here). Seasons after 2025-26 use the season's snapshot at Oct 1 (live: refreshed by
# the roster collector, so an in-season change shows up once the snapshot is re-fetched).
# Coach ids: team_coaches.coach_id by name; coaches never in a snapshot get a stable
# synthetic negative id.

CURATED_COACH_LAST = "2025-26"
COACH_STINTS = {
    "ATL": [("2015-10-01", "Mike Budenholzer"), ("2018-10-01", "Lloyd Pierce"),
            ("2021-03-01", "Nate McMillan"), ("2023-02-22", "Quin Snyder")],
    "BOS": [("2015-10-01", "Brad Stevens"), ("2021-10-01", "Ime Udoka"),
            ("2022-10-01", "Joe Mazzulla")],
    "BKN": [("2015-10-01", "Lionel Hollins"), ("2016-01-11", "Tony Brown"),
            ("2016-10-01", "Kenny Atkinson"), ("2020-03-08", "Jacque Vaughn"),
            ("2020-10-01", "Steve Nash"), ("2022-11-02", "Jacque Vaughn"),
            ("2024-02-20", "Kevin Ollie"), ("2024-10-01", "Jordi Fernandez")],
    "CHA": [("2015-10-01", "Steve Clifford"), ("2018-10-01", "James Borrego"),
            ("2022-10-01", "Steve Clifford"), ("2024-10-01", "Charles Lee")],
    "CHI": [("2015-10-01", "Fred Hoiberg"), ("2018-12-03", "Jim Boylen"),
            ("2020-10-01", "Billy Donovan")],
    "CLE": [("2015-10-01", "David Blatt"), ("2016-01-23", "Tyronn Lue"),
            ("2018-10-29", "Larry Drew"), ("2019-10-01", "John Beilein"),
            ("2020-02-19", "J.B. Bickerstaff"), ("2024-10-01", "Kenny Atkinson")],
    "DAL": [("2015-10-01", "Rick Carlisle"), ("2021-10-01", "Jason Kidd")],
    "DEN": [("2015-10-01", "Michael Malone"), ("2025-04-09", "David Adelman")],
    "DET": [("2015-10-01", "Stan Van Gundy"), ("2018-10-01", "Dwane Casey"),
            ("2023-10-01", "Monty Williams"), ("2024-10-01", "J.B. Bickerstaff")],
    "GSW": [("2015-10-01", "Luke Walton"), ("2016-01-22", "Steve Kerr")],
    "HOU": [("2015-10-01", "Kevin McHale"), ("2015-11-19", "J.B. Bickerstaff"),
            ("2016-10-01", "Mike D'Antoni"), ("2020-10-01", "Stephen Silas"),
            ("2023-10-01", "Ime Udoka")],
    "IND": [("2015-10-01", "Frank Vogel"), ("2016-10-01", "Nate McMillan"),
            ("2020-10-01", "Nate Bjorkgren"), ("2021-10-01", "Rick Carlisle")],
    "LAC": [("2015-10-01", "Doc Rivers"), ("2020-10-01", "Tyronn Lue")],
    "LAL": [("2015-10-01", "Byron Scott"), ("2016-10-01", "Luke Walton"),
            ("2019-10-01", "Frank Vogel"), ("2022-10-01", "Darvin Ham"),
            ("2024-10-01", "JJ Redick")],
    "MEM": [("2015-10-01", "David Joerger"), ("2016-10-01", "David Fizdale"),
            ("2017-11-27", "J.B. Bickerstaff"), ("2019-10-01", "Taylor Jenkins"),
            ("2025-03-29", "Tuomas Iisalo")],
    "MIA": [("2015-10-01", "Erik Spoelstra")],
    "MIL": [("2015-10-01", "Jason Kidd"), ("2018-01-23", "Joe Prunty"),
            ("2018-10-01", "Mike Budenholzer"), ("2023-10-01", "Adrian Griffin"),
            ("2024-01-24", "Joe Prunty"), ("2024-01-29", "Doc Rivers")],
    "MIN": [("2015-10-01", "Sam Mitchell"), ("2016-10-01", "Tom Thibodeau"),
            ("2019-01-07", "Ryan Saunders"), ("2021-02-23", "Chris Finch")],
    "NOP": [("2015-10-01", "Alvin Gentry"), ("2020-10-01", "Stan Van Gundy"),
            ("2021-10-01", "Willie Green")],
    "NYK": [("2015-10-01", "Derek Fisher"), ("2016-02-09", "Kurt Rambis"),
            ("2016-10-01", "Jeff Hornacek"), ("2018-10-01", "David Fizdale"),
            ("2019-12-07", "Mike Miller"), ("2020-10-01", "Tom Thibodeau"),
            ("2025-10-01", "Mike Brown")],
    "OKC": [("2015-10-01", "Billy Donovan"), ("2020-10-01", "Mark Daigneault")],
    "ORL": [("2015-10-01", "Scott Skiles"), ("2016-10-01", "Frank Vogel"),
            ("2018-10-01", "Steve Clifford"), ("2021-10-01", "Jamahl Mosley")],
    "PHI": [("2015-10-01", "Brett Brown"), ("2020-10-01", "Doc Rivers"),
            ("2023-10-01", "Nick Nurse")],
    "PHX": [("2015-10-01", "Jeff Hornacek"), ("2016-02-02", "Earl Watson"),
            ("2017-10-23", "Jay Triano"), ("2018-10-01", "Igor Kokoskov"),
            ("2019-10-01", "Monty Williams"), ("2023-10-01", "Frank Vogel"),
            ("2024-10-01", "Mike Budenholzer"), ("2025-10-01", "Jordan Ott")],
    "POR": [("2015-10-01", "Terry Stotts"), ("2021-10-01", "Chauncey Billups"),
            ("2025-10-24", "Tiago Splitter")],
    "SAC": [("2015-10-01", "George Karl"), ("2016-10-01", "David Joerger"),
            ("2019-10-01", "Luke Walton"), ("2021-11-22", "Alvin Gentry"),
            ("2022-10-01", "Mike Brown"), ("2024-12-28", "Doug Christie")],
    "SAS": [("2015-10-01", "Gregg Popovich"), ("2024-11-03", "Mitch Johnson")],
    "TOR": [("2015-10-01", "Dwane Casey"), ("2018-10-01", "Nick Nurse"),
            ("2023-10-01", "Darko Rajakovic")],
    "UTA": [("2015-10-01", "Quin Snyder"), ("2022-10-01", "Will Hardy")],
    "WAS": [("2015-10-01", "Randy Wittman"), ("2016-10-01", "Scott Brooks"),
            ("2021-10-01", "Wes Unseld"), ("2024-01-26", "Brian Keefe")],
}
COACH_K = 40.0          # pseudo-games shrinking a coach's rotation concentration to the league
COACH_DECAY = 0.995     # per coach game (half-life ~140 games)
TEAM_ROT_ALPHA = 0.1    # team rotation-shape EWMA


def load_coach_snapshots(conn: sqlite3.Connection) -> pd.DataFrame:
    """season, team, coach_id, coach_name of the head coach per team_coaches snapshot."""
    return pd.read_sql_query(
        "SELECT season, team, coach_id, coach_name, sort_sequence FROM team_coaches "
        "WHERE coach_type = 'Head Coach' ORDER BY season, team, sort_sequence", conn)


def _coach_id(name: str, ids: dict) -> int:
    if name in ids:
        return int(ids[name])
    import zlib
    return -int(zlib.crc32(name.encode()) % 1_000_000) - 1


def coach_stints(snap: pd.DataFrame | None = None) -> pd.DataFrame:
    """team, date (int days), coach_id, coach_name, source: the head coach from `date` on.
    Curated through CURATED_COACH_LAST; later seasons from the snapshot at Oct 1."""
    snap = snap if snap is not None else pd.DataFrame(columns=["season", "team", "coach_id",
                                                               "coach_name"])
    ids = dict(zip(snap["coach_name"], snap["coach_id"]))
    rows = [{"team": t, "day": d, "coach_name": n, "source": "curated"}
            for t, st in COACH_STINTS.items() for d, n in st]
    later = snap[snap["season"] > CURATED_COACH_LAST].drop_duplicates(["season", "team"])
    for r in later.itertuples(index=False):
        rows.append({"team": r.team, "day": f"{r.season[:4]}-10-01", "coach_name": r.coach_name,
                     "source": "snapshot"})
    df = pd.DataFrame(rows)
    df["date"] = _days(df["day"])
    df["coach_id"] = [_coach_id(n, ids) for n in df["coach_name"]]
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    # drop snapshot rows that repeat the current coach
    same = (df["team"] == df["team"].shift()) & (df["coach_id"] == df["coach_id"].shift())
    return df[~same][["team", "date", "coach_id", "coach_name", "source"]].reset_index(drop=True)


def coach_snapshot_check(stints: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Consistency of the curated stints with the snapshots: for each (season S, team) the
    snapshot coach should be the curated coach at the end of S or at the start of S+1."""
    out = []
    for r in snap.drop_duplicates(["season", "team"]).itertuples(index=False):
        if r.season > CURATED_COACH_LAST:
            continue
        y = int(r.season[:4])
        s = stints[stints["team"] == r.team]
        end = s[s["date"] <= _days(pd.Series([f"{y + 1}-06-30"]))[0]]
        nxt = s[s["date"] <= _days(pd.Series([f"{y + 1}-10-02"]))[0]]
        a = end["coach_id"].iloc[-1] if len(end) else None
        b = nxt["coach_id"].iloc[-1] if len(nxt) else None
        out.append({"season": r.season, "team": r.team, "snapshot": r.coach_name,
                    "ok": r.coach_id in (a, b)})
    return pd.DataFrame(out)


class CoachState:
    """Coach id per team-date and causal rotation-concentration state.

    Per regular-season team-game: top-7 share of the team's minutes (regulation-scaled) and the
    number of 30+ minute players. Coach level: decayed (COACH_DECAY per coach game) sums over
    the coach's games on any team, shrunk to the causal league mean with COACH_K pseudo-games
    (so a coach's habits move with him). Team level: EWMA (TEAM_ROT_ALPHA). Every lookup is
    strictly before the target date."""

    def __init__(self, stints, games, league):
        self.stints = stints      # team, date, coach_id
        self.games = games        # per team-game post state (sorted by date)
        self.league = league      # date, l_top7, l_n30 (post-day expanding means)

    @classmethod
    def build(cls, hist: pd.DataFrame, tg: pd.DataFrame, stints: pd.DataFrame) -> "CoachState":
        reg = hist[hist["season_type"] == "Regular Season"][["game_id", "team", "min_r"]]
        srt = reg.sort_values(["game_id", "team", "min_r"], ascending=[True, True, False])
        rk = srt.groupby(["game_id", "team"], sort=False).cumcount().to_numpy()
        srt = srt.assign(top=np.where(rk < 7, srt["min_r"].to_numpy(float), 0.0),
                         n30=(srt["min_r"].to_numpy(float) >= 30.0).astype(float))
        a = srt.groupby(["game_id", "team"]).agg(tot=("min_r", "sum"), top=("top", "sum"),
                                                 n30=("n30", "sum")).reset_index()
        a = a[a["tot"] > 100]
        a["top7"] = a["top"] / a["tot"]
        g = tg[["game_id", "team", "date"]].merge(a[["game_id", "team", "top7", "n30"]],
                                                  on=["game_id", "team"])
        g = g.sort_values(["date", "game_id", "team"], kind="mergesort").reset_index(drop=True)
        st = stints[["team", "date", "coach_id"]].sort_values("date")
        g = pd.merge_asof(g, st, on="date", by="team", direction="backward")
        g["coach_id"] = g["coach_id"].fillna(0).astype(np.int64)    # 0 = unknown coach
        # league expanding means by day (post-day)
        d = g.groupby("date").agg(s7=("top7", "sum"), s30=("n30", "sum"), n=("top7", "size"))
        cn = d["n"].cumsum()
        league = pd.DataFrame({"date": d.index.to_numpy(),
                               "l_top7": (d["s7"].cumsum() / cn).to_numpy(),
                               "l_n30": (d["s30"].cumsum() / cn).to_numpy()})
        # coach decayed sums (post-game), by coach in date order
        g["c_num7"] = 0.0
        g["c_num30"] = 0.0
        g["c_den"] = 0.0
        for cid, idx in g.groupby("coach_id", sort=False).indices.items():
            if cid == 0:
                continue
            g.loc[idx, "c_num7"] = _ewsum(g["top7"].to_numpy()[idx], COACH_DECAY)
            g.loc[idx, "c_num30"] = _ewsum(g["n30"].to_numpy()[idx], COACH_DECAY)
            g.loc[idx, "c_den"] = _ewsum(np.ones(len(idx)), COACH_DECAY)
        g["ct_n"] = g.groupby(["coach_id", "team"]).cumcount() + 1.0
        for c, src in (("t_top7", "top7"), ("t_n30", "n30")):
            g[c] = g.groupby("team", sort=False)[src].transform(
                lambda x: x.ewm(alpha=TEAM_ROT_ALPHA, adjust=True).mean())
        return cls(stints[["team", "date", "coach_id"]].copy(), g, league)

    def truncate(self, d: int) -> "CoachState":
        return CoachState(self.stints, self.games[self.games["date"] < d],
                          self.league[self.league["date"] < d])

    def features(self, team: np.ndarray, date: np.ndarray) -> pd.DataFrame:
        """Pre-game coach features for (team, date) rows (same order as the input)."""
        q = pd.DataFrame({"team": np.asarray(team), "date": np.asarray(date, np.int64)})
        q["i"] = np.arange(len(q))
        q = q.sort_values("date", kind="mergesort")
        st = self.stints.sort_values("date")
        q = pd.merge_asof(q, st, on="date", by="team", direction="backward")
        q["coach_id"] = q["coach_id"].fillna(0).astype(np.int64)
        q["date_q"] = q["date"] - 1
        lg = self.league.rename(columns={"date": "l_date"}).sort_values("l_date")
        q = pd.merge_asof(q.sort_values("date_q"), lg, left_on="date_q", right_on="l_date",
                          direction="backward")
        g = self.games.sort_values("date", kind="mergesort")
        cs = g[g["coach_id"] != 0][["coach_id", "date", "c_num7", "c_num30", "c_den"]].rename(
            columns={"date": "c_date"})
        q = pd.merge_asof(q.sort_values("date_q"), cs.sort_values("c_date"), left_on="date_q",
                          right_on="c_date", by="coach_id", direction="backward")
        ct = g[g["coach_id"] != 0][["coach_id", "team", "date", "ct_n"]].rename(
            columns={"date": "ct_date"})
        q = pd.merge_asof(q.sort_values("date_q"), ct.sort_values("ct_date"), left_on="date_q",
                          right_on="ct_date", by=["coach_id", "team"], direction="backward")
        tt = g[["team", "date", "t_top7", "t_n30"]].rename(columns={"date": "t_date"})
        q = pd.merge_asof(q.sort_values("date_q"), tt.sort_values("t_date"), left_on="date_q",
                          right_on="t_date", by="team", direction="backward")
        q = q.sort_values("i")
        known = (q["coach_id"] != 0).to_numpy()
        den = q["c_den"].fillna(0.0).to_numpy(float)
        l7 = q["l_top7"].fillna(0.75).to_numpy(float)
        l30 = q["l_n30"].fillna(2.0).to_numpy(float)
        c7 = (q["c_num7"].fillna(0.0).to_numpy(float) + COACH_K * l7) / (den + COACH_K)
        c30 = (q["c_num30"].fillna(0.0).to_numpy(float) + COACH_K * l30) / (den + COACH_K)
        return pd.DataFrame({
            "coach_id": q["coach_id"].to_numpy(),
            "c_top7": np.where(known, c7, np.nan), "c_n30": np.where(known, c30, np.nan),
            "c_games": np.where(known, den, np.nan),
            "c_team_n": np.where(known, q["ct_n"].fillna(0.0).to_numpy(float), np.nan),
            "t_top7": q["t_top7"].to_numpy(float), "t_n30": q["t_n30"].to_numpy(float)})


# --- pre-game panel --------------------------------------------------------------------------

def pregame_panel(h: History, targets: pd.DataFrame, extra: pd.DataFrame | None = None,
                  window: int = WINDOW_GAMES) -> pd.DataFrame:
    """Candidates and their pre-game features for each target team-game.

    targets: columns team, date (int days or datetime), season, and `key` (any id).
    extra: optional key, player_id rows forced into the candidate set.
    Every feature uses rows with date < target date only."""
    tgt = targets.copy()
    if not np.issubdtype(tgt["date"].dtype, np.integer):
        tgt["date"] = _days(tgt["date"])
    hist = h.hist
    out_key, out_pid, out_app = [], [], []
    team_rows = {t: idx for t, idx in hist.groupby("team", sort=False).indices.items()}
    h_tgi = hist["tgi"].to_numpy()
    h_pid = hist["player_id"].to_numpy()
    h_date = hist["date"].to_numpy()
    k_of = np.zeros(len(tgt), dtype=np.int64)
    for i, (key, team, date) in enumerate(zip(tgt["key"], tgt["team"], tgt["date"])):
        s = h.sched.get(team)
        k = int(np.searchsorted(s["date"], date, side="left")) if s is not None else 0
        k_of[i] = k
        idx = team_rows.get(team)
        if idx is None or k == 0:
            continue
        ti = h_tgi[idx]
        sel = idx[(ti >= k - window) & (ti < k) & (h_date[idx] < date)]
        if not len(sel):
            continue
        u, c = np.unique(h_pid[sel], return_counts=True)
        out_key.append(np.full(len(u), key, dtype=object))
        out_pid.append(u)
        out_app.append(c)
    tgt["k"] = k_of
    cand = pd.DataFrame({"key": np.concatenate(out_key) if out_key else [],
                         "player_id": np.concatenate(out_pid) if out_pid else [],
                         "app10": np.concatenate(out_app) if out_app else []})
    cand["roster"] = 1
    if extra is not None and len(extra):
        ex = extra[["key", "player_id"]].copy()
        ex["app10"] = 0
        ex["roster"] = 0
        cand = pd.concat([cand, ex], ignore_index=True)
        cand = cand.sort_values("roster", ascending=False).drop_duplicates(["key", "player_id"])
        cand["app10"] = cand.groupby(["key", "player_id"])["app10"].transform("max")
    cand["player_id"] = cand["player_id"].astype(np.int64)
    cand["app10"] = cand["app10"].astype(float)
    cand = cand.merge(tgt[["key", "team", "date", "season", "k"]], on="key", how="left")
    # last state strictly before the target date
    v1c = [c for c in V1_HIST_COLS if c in hist]
    st = hist[["player_id", "date", "team", "season", "min_r", "started", "ew_f", "ew_s",
               "med5", "st10", "pf36", "dkpm", "n_gp", "base_ew", "season_gp", "gsr_post"]
              + v1c]
    st = st.rename(columns={"date": "last_date", "team": "last_team", "season": "last_season",
                            "min_r": "last", "started": "start_last"})
    cand = cand.sort_values("date", kind="mergesort").reset_index(drop=True)
    st = st.sort_values("last_date", kind="mergesort")
    cand["date_q"] = cand["date"] - 1    # strictly before: last_date <= date - 1
    p = pd.merge_asof(cand, st, left_on="date_q", right_on="last_date", by="player_id",
                      direction="backward")
    # roster rule: last team must be this team (extras kept)
    p = p[(p["roster"] == 0) | (p["last_team"] == p["team"])].copy()
    has = p["last_date"].notna()
    # team games missed since his last appearance (this team's schedule)
    missed = np.zeros(len(p))
    for team, idx in p.groupby("team", sort=False).indices.items():
        s = h.sched.get(team)
        if s is None:
            continue
        ld = p["last_date"].to_numpy()[idx]
        cnt = np.searchsorted(s["date"], np.where(np.isnan(ld), -1e18, ld), side="right")
        missed[idx] = np.where(np.isnan(ld), 0, p["k"].to_numpy()[idx] - cnt)
    p["missed_now"] = np.maximum(missed, 0)
    p["gsr_now"] = np.where(p["missed_now"] >= 3, 0, p["gsr_post"].fillna(10))
    p["new_season"] = (has & (p["last_season"] != p["season"])).astype(float)
    p["gp_season"] = np.where(p["new_season"] > 0, 0, p["season_gp"].fillna(0))
    for c, v in (("ew_f", BASE_M0), ("ew_s", BASE_M0), ("med5", BASE_M0), ("last", BASE_M0),
                 ("st10", 0.0), ("start_last", 0.0), ("pf36", 3.5), ("dkpm", 0.9),
                 ("n_gp", 0.0), ("base_ew", BASE_M0)):
        p[c] = p[c].astype(float).fillna(v)
    # team rest
    rest = np.full(len(p), 3.0)
    for team, idx in p.groupby("team", sort=False).indices.items():
        s = h.sched.get(team)
        if s is None:
            continue
        k = p["k"].to_numpy()[idx]
        prev = np.where(k > 0, s["date"][np.maximum(k - 1, 0)], -10**6)
        rest[idx] = np.clip(p["date"].to_numpy()[idx] - prev, 0, 10)
    p["rest"] = rest
    p["b2b"] = (rest == 1).astype(float)
    # team starter targets per group (EWMA of the team's starter minutes, last game < date)
    tt = h.targets.rename(columns={"date": "t_date"}).sort_values("t_date", kind="mergesort")
    p = p.sort_values("date_q", kind="mergesort")
    p = pd.merge_asof(p, tt.drop(columns=["tgi"]), left_on="date_q", right_on="t_date",
                      by="team", direction="backward")
    p = _attach_groups(p, h.positions)
    p = p.drop(columns=["date_q", "t_date", "season_gp", "gsr_post"])
    p = p.sort_values(["key", "player_id"]).reset_index(drop=True)
    if v1c:
        p = _v1_panel_columns(h, p)
    return p


V1_HIST_COLS = ("ew_vf", "min_sd", "ew_start", "ew_bench", "sea_mean", "abs_len", "last_absm",
                "last_pf")


def _v1_panel_columns(h: History, p: pd.DataFrame) -> pd.DataFrame:
    """Scenario-free v1 features (all from rows strictly before the target date):
    return (current absence length), new team, season-to-date mean, age, team schedule
    density, team games into the season, coach (N4) and team rotation concentration, the
    player's minutes rank among the team-game's candidates, the share pairs (if h.shares)."""
    p = p.copy()
    mn = p["missed_now"].to_numpy(float)
    gsr = p["gsr_now"].to_numpy(float)
    post = p["abs_len"].to_numpy(float) if "abs_len" in p else np.zeros(len(p))
    p["abs_len_now"] = np.where(mn >= 3, mn, np.where(gsr < 10, np.nan_to_num(post), 0.0))
    has = p["last_date"].notna().to_numpy()
    p["new_team"] = (has & (p["last_team"] != p["team"]).to_numpy()).astype(float)
    p["sea_mean_now"] = np.where(p["new_season"].to_numpy() > 0, np.nan,
                                 p["sea_mean"].to_numpy(float))
    if h.births is not None and len(h.births):
        b = p[["player_id"]].merge(h.births, on="player_id", how="left")["birth"].to_numpy(float)
        p["age"] = (p["date"].to_numpy(float) - b) / 365.25
    else:
        p["age"] = np.nan
    # team schedule: games in the 3 days before, team games already played this season
    g3 = np.zeros(len(p))
    tgs = np.zeros(len(p))
    for team, idx in p.groupby("team", sort=False).indices.items():
        s = h.sched.get(team)
        if s is None:
            continue
        dt = p["date"].to_numpy()[idx]
        g3[idx] = (np.searchsorted(s["date"], dt, side="left")
                   - np.searchsorted(s["date"], dt - 3, side="left"))
        sea = p["season"].to_numpy()[idx]
        k = p["k"].to_numpy()[idx]
        first = {x: int(np.searchsorted(s["season"], x, side="left")) for x in set(sea)}
        tgs[idx] = np.maximum(k - np.array([first[x] for x in sea]), 0)
    p["g_last3d"] = g3
    p["tgs"] = tgs
    if h.pre is not None:
        p = preseason_columns(h, p)
    if h.coach is not None:
        cf = h.coach.features(p["team"].to_numpy(), p["date"].to_numpy())
        for c in cf.columns:
            p[c] = cf[c].to_numpy()
    else:
        for c in ("coach_id", "c_top7", "c_n30", "c_games", "c_team_n", "t_top7", "t_n30"):
            p[c] = np.nan
    ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
    p["team_rank"] = pd.Series(ew).groupby(p["key"].to_numpy()).rank(
        ascending=False, method="first").to_numpy()
    if h.shares is not None:
        p["share_map"] = share_maps(h.shares, p)
    return p


def share_maps(shares: "PairShares", p: pd.DataFrame) -> list:
    """Per row: {teammate player_id: share of that teammate's minutes this player absorbed when
    the teammate sat (games strictly before the date)}, nonzero entries only."""
    out = [None] * len(p)
    pids = p["player_id"].to_numpy(np.int64)
    dates = p["date"].to_numpy()
    for _, idx in p.groupby("key", sort=False).indices.items():
        m = shares.full_matrix(int(dates[idx[0]]), pids[idx])
        for a, i in enumerate(idx):
            nz = np.flatnonzero(m[a])
            out[i] = {int(pids[idx[j]]): float(m[a, j]) for j in nz}
    return out


def design(p: pd.DataFrame, names: Iterable[str] = BASE_FEATURES) -> np.ndarray:
    ew_f = p["ew_f"].to_numpy(float)
    ew_s = p["ew_s"].to_numpy(float)
    gsr = p["gsr_now"].to_numpy(float)
    ret0 = (gsr == 0).astype(float)
    ret12 = ((gsr >= 1) & (gsr <= 2)).astype(float)
    ns = p["new_season"].to_numpy(float)
    b2b = p["b2b"].to_numpy(float)
    n = p["n_gp"].to_numpy(float)
    inv = 1.0 / (1.0 + n)
    st10 = p["st10"].to_numpy(float)
    mn = p["missed_now"].to_numpy(float)
    cols = {
        "one": np.ones(len(p)), "ew_f": ew_f, "ew_s": ew_s, "med5": p["med5"].to_numpy(float),
        "last": p["last"].to_numpy(float), "st10": st10,
        "start_last": p["start_last"].to_numpy(float), "st10_ewf": st10 * ew_f,
        "ret0": ret0, "ret0_ews": ret0 * ew_s, "ret12_ews": ret12 * ew_s,
        "short_miss": ((mn >= 1) & (mn < 3)).astype(float), "b2b": b2b, "b2b_ewf": b2b * ew_f,
        "new_season": ns, "new_season_ews": ns * ew_s, "new_season_ewf": ns * ew_f,
        "inv_gp": inv, "inv_gp_ewf": inv * ew_f,
        "early_season": (p["gp_season"].to_numpy(float) < 5).astype(float),
        "app10": p["app10"].to_numpy(float), "missed_now_c": np.minimum(mn, 10.0),
        "log_gp": np.log1p(n),
    }
    if "avail_rank" in p:
        rk = p["avail_rank"].to_numpy(float)
        cols.update({"rk_le7": (rk <= 7).astype(float), "rk8": (rk == 8).astype(float),
                     "rk9": (rk == 9).astype(float), "rk10": (rk == 10).astype(float),
                     "rk11": (rk == 11).astype(float), "rk12": (rk == 12).astype(float),
                     "n_avail": np.minimum(p["n_avail"].to_numpy(float), 18.0)})
    return np.column_stack([cols[c] for c in names])


# --- pair shares: who absorbed j's minutes when j sat ----------------------------------------

def pair_events(panel: pd.DataFrame) -> pd.DataFrame:
    """From an oracle panel: one row per (team-game, absent regular j, available player i who
    played) with i's gain over his own EWMA (regulation minutes), split across the absent
    players in proportion to their EWMAs. No fitted parameter is used."""
    ew = 0.5 * panel["ew_f"].to_numpy(float) + 0.5 * panel["ew_s"].to_numpy(float)
    d = panel[["key", "date", "team", "player_id", "absent", "played", "act_min_r"]].copy()
    d["ew"] = ew
    ab = d[d["absent"]][["key", "player_id", "ew"]].rename(
        columns={"player_id": "j", "ew": "vac"})
    if ab.empty:
        return pd.DataFrame(columns=["date", "team", "j", "i", "gain", "vac"])
    ab["vac_tot"] = ab.groupby("key")["vac"].transform("sum")
    pl = d[d["played"] & ~d["absent"]][["key", "date", "team", "player_id", "act_min_r", "ew"]]
    pl = pl.rename(columns={"player_id": "i"})
    e = pl.merge(ab, on="key")
    e["gain"] = (e["act_min_r"] - e["ew"]) * e["vac"] / e["vac_tot"]
    return e[["date", "team", "j", "i", "gain", "vac"]].reset_index(drop=True)


class PairShares:
    """share[i, j] at day d = sum of i's gains in games before d that j missed /
    (sum of j's EWMA in those games + SHARE_K x 20), clipped to [-0.2, 1]."""

    def __init__(self, events: pd.DataFrame):
        e = events.sort_values(["j", "i", "date"], kind="mergesort")
        self._j = e["j"].to_numpy(np.int64)
        self._i = e["i"].to_numpy(np.int64)
        self._d = e["date"].to_numpy(np.int64)
        self._g = e["gain"].to_numpy(float)
        self._v = e["vac"].to_numpy(float)
        key = self._j * 10_000_000_000 + self._i
        self._key = key
        self._cg = np.cumsum(self._g)
        self._cv = np.cumsum(self._v)
        # first index of each pair
        self._uk, self._start = np.unique(key, return_index=True)
        # (pair rank, date) packed into one sorted int64 for vectorised lookups
        self._d0 = int(self._d.min()) - 1 if len(self._d) else 0
        rank = np.searchsorted(self._uk, key)
        self._comb = rank * (1 << 22) + np.clip(self._d - self._d0, 0, (1 << 22) - 1)

    def lookup(self, i: np.ndarray, j: np.ndarray, date: int) -> np.ndarray:
        """share[i, j] for arrays of player ids (i absorbed j's minutes), games before date."""
        i = np.asarray(i, np.int64)
        j = np.asarray(j, np.int64)
        out = np.zeros(len(i))
        if not len(self._uk) or not len(i):
            return out
        keys = j * 10_000_000_000 + i
        r = np.searchsorted(self._uk, keys)
        ok = r < len(self._uk)
        ok[ok] &= self._uk[r[ok]] == keys[ok]
        if not ok.any():
            return out
        r = r[ok]
        st = self._start[r]
        q = r * (1 << 22) + min(max(int(date) - self._d0, 0), (1 << 22) - 1)
        k = np.searchsorted(self._comb, q, side="left")
        has = k > st
        g = np.zeros(len(r))
        v = np.zeros(len(r))
        kk, ss = k[has], st[has]
        base_g = np.where(ss > 0, self._cg[np.maximum(ss - 1, 0)], 0.0)
        base_v = np.where(ss > 0, self._cv[np.maximum(ss - 1, 0)], 0.0)
        g[has] = self._cg[kk - 1] - base_g
        v[has] = self._cv[kk - 1] - base_v
        val = np.where(has, np.clip(g / (v + SHARE_K * 20.0), -0.2, 1.0), 0.0)
        out[np.flatnonzero(ok)] = val
        return out

    def full_matrix(self, date: int, pids: np.ndarray) -> np.ndarray:
        """share[a, b] for every ordered pair of `pids` (diagonal 0)."""
        pids = np.asarray(pids, np.int64)
        n = len(pids)
        a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a, b = a.ravel(), b.ravel()
        m = a != b
        out = np.zeros((n, n))
        out[a[m], b[m]] = self.lookup(pids[a[m]], pids[b[m]], date)
        return out

    def matrix(self, date: int, pids: np.ndarray, absent: np.ndarray) -> np.ndarray:
        n = len(pids)
        out = np.zeros((n, n))
        js = np.flatnonzero(absent)
        if not len(js) or not len(self._uk):
            return out
        av = np.flatnonzero(~absent)
        for j in js:
            keys = int(pids[j]) * 10_000_000_000 + pids[av].astype(np.int64)
            pos = np.searchsorted(self._uk, keys)
            ok = (pos < len(self._uk))
            ok[ok] &= self._uk[pos[ok]] == keys[ok]
            for a, ps in zip(av[ok], pos[ok]):
                st = self._start[ps]
                en = self._start[ps + 1] if ps + 1 < len(self._start) else len(self._key)
                k = st + np.searchsorted(self._d[st:en], date, side="left")  # dates < date
                if k == st:
                    continue
                base_g = self._cg[st - 1] if st > 0 else 0.0
                base_v = self._cv[st - 1] if st > 0 else 0.0
                g = self._cg[k - 1] - base_g
                v = self._cv[k - 1] - base_v
                out[a, j] = np.clip(g / (v + SHARE_K * 20.0), -0.2, 1.0)
        return out


# --- projection core -------------------------------------------------------------------------

def _team_target(p: pd.DataFrame, params: MinutesParams) -> np.ndarray:
    """(n, 3) starter-minute targets per group, shrunk to the league value (TARGET_K)."""
    lg = np.asarray(params.starter_league, float)
    out = np.empty((len(p), 3))
    for g in range(3):
        num = p[f"tgt_num{g}"].to_numpy(float) if f"tgt_num{g}" in p else np.zeros(len(p))
        den = p[f"tgt_den{g}"].to_numpy(float) if f"tgt_den{g}" in p else np.zeros(len(p))
        num = np.nan_to_num(num)
        den = np.nan_to_num(den)
        out[:, g] = (num + TARGET_K * lg[g]) / (den + TARGET_K)
    return out


def avail_rank(p: pd.DataFrame, base: np.ndarray, absent: np.ndarray, key: str = "key"
               ) -> tuple[np.ndarray, np.ndarray]:
    """Depth rank by base among available candidates of the team-game (1 = most minutes) and
    the number available. Absent players get rank 99."""
    b = pd.Series(np.where(absent, -np.inf, base), index=p.index)
    k = p[key] if key in p else pd.Series(0, index=p.index)
    rk = b.groupby(k.to_numpy()).rank(ascending=False, method="first").to_numpy()
    n = pd.Series(~absent, index=p.index).groupby(k.to_numpy()).transform("sum").to_numpy()
    return np.where(absent, 99.0, rk), n.astype(float)


def base_and_pplay(p: pd.DataFrame, params: MinutesParams, absent=None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Base minutes when playing and P(plays | available); the latter depends on the scenario
    through the depth rank among available players.

    v1 (params.version == "v1"): the base is the LightGBM E[min | plays] on the panel's v1
    columns plus the scenario features of `absent` (learned promotion); players failing
    `v1_ok` (no box history, < V1_MIN_GP games) and panels without v1 columns keep the v0.5
    linear base (S15 fallback). P(plays | available): the v0.5 logit, or in v1 its refit
    `pplay_coef_v1` with forced extras scored as roster players (see fit_v1)."""
    X = design(p, params.base_features)
    base = np.clip(X @ np.asarray(params.base_coef), 0.5, 44.0)
    if absent is None:
        absent = p["absent"].to_numpy(bool) if "absent" in p else np.zeros(len(p), bool)
    absent = np.asarray(absent, bool)
    q = p.copy()
    q["avail_rank"], q["n_avail"] = avail_rank(q, base, absent)
    Z = design(q, params.pplay_features)
    pp = 1.0 / (1.0 + np.exp(-(Z @ np.asarray(params.pplay_coef))))
    pp = np.clip(pp, 0.01, 0.995)
    if params.version == "v1" and params.gbm_model:
        s_rows = np.where(absent, 0.0, pp * base)      # the v0.5 reserve input, per row
        if params.pplay_coef_v1:
            Z1 = design(forced_as_roster(q), params.pplay_features)
            pp = np.clip(1.0 / (1.0 + np.exp(-(Z1 @ np.asarray(params.pplay_coef_v1)))),
                         0.01, 0.995)
        base = v1_base(p, params, absent, base).view(V1Base)
        base.s_rows = s_rows
        if params.minfix:
            pp = mf_pplay(q, params, np.asarray(base), absent)
    if params.reserve_debut:
        if not isinstance(base, V1Base):
            base = base.view(V1Base)
        base.res_add = debut_reserve(params, p)
    return base, pp


def forced_as_roster(p: pd.DataFrame) -> pd.DataFrame:
    """v1 P(plays): candidates forced in from outside the recent roster (roster == 0 with box
    history: DK pool / report players, new signings, returns from long absences) are scored as
    active roster players (10 of the last 10 team games, nothing missed). In training such rows
    exist only when they played (the oracle), so their app10 = 0 / missed-10 profile otherwise
    reads as a deep-bench DNP (live: P(plays) 0.05-0.5 for new-team rotation players)."""
    if "roster" not in p:
        return p
    f = ((p["roster"].to_numpy() == 0) & p["last_date"].notna().to_numpy()) \
        if "last_date" in p else p["roster"].to_numpy() == 0
    if not f.any():
        return p
    q = p.copy()
    q.loc[f, "app10"] = 10.0
    q.loc[f, "missed_now"] = 0.0
    q.loc[f, "gsr_now"] = 10.0
    return q


# --- minutes fix (COURTSIDE_MINFIX) ----------------------------------------------------------
#
# Diagnosis (docs/PLAYERS.md, minutes-fix results): on the avail harness the v1 P(plays |
# available) logit reads only the players' history (EWMAs, appearances, missed games, the v0.5
# depth rank), not the GBM's plan for the night. Players the GBM plans below their recent minutes
# (E[min | plays] < 0.9 x last-10: a starter or teammate back, the deep bench after DNPs) play
# less often than it says (P(play) +0.05..0.06), players planned above them more often (-0.06..
# -0.07); the minutes when they play are close. MF_PLAN_FEATURES add the plan to the logit (log
# plan and log plan / EWMAs / median of 5), fit on the out-of-fold GBM plans of the training
# panel. The noise model is fit net of the margin curve: the sims add g(margin) and then the
# noise, but the v1 noise was fit on log(actual / plan), which already holds the margin effect.

MF_VERSION = 1
MF_PLAN_FEATURES = ("mf_lb", "mf_lrf", "mf_lrs", "mf_lrm")
MF_RANKS = (5, 7, 8, 9, 10, 11, 12)
# fit options (`fit_v1(minfix=...)`; True = MF_DEFAULT): pplay "logit" (plan features) |
# "logit_rank" (+ dummies of the depth rank by the plan); noise "net" (MLE net of the margin
# curve) | "calib" (then location / scale per tier calibrated by simulation on the training
# projections). A LightGBM classifier was tried and dropped (it learns the oracle's "available
# regulars always play" rule: base_ew >= REGULAR_MIN defines who can be absent).
MF_DEFAULT = {"pplay": "logit_rank", "noise": "calib"}
MF_CAL_SIMS = 60
MF_CAL_ITERS = 4


def mf_plan_columns(p: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """(n, 4) plan features: log plan, log plan / fast EWMA, / slow EWMA, / median of 5."""
    b = np.maximum(np.asarray(base, float), 0.5)
    lb = np.log(b)
    cols = [lb]
    for c in ("ew_f", "ew_s", "med5"):
        cols.append(lb - np.log(np.maximum(p[c].to_numpy(float), 1.0)))
    return np.column_stack(cols)


def mf_design(q: pd.DataFrame, params: MinutesParams, base: np.ndarray, absent: np.ndarray,
              kind: str) -> np.ndarray:
    """The minfix P(plays) design of the available-and-absent panel q (avail_rank / n_avail
    set): the v1 logit's columns (forced extras as roster) + the plan [+ dummies of the depth
    rank by the plan among the available ("logit_rank")]."""
    cols = [design(forced_as_roster(q), params.pplay_features), mf_plan_columns(q, base)]
    if kind == "logit_rank":
        rk, _ = avail_rank(q, np.asarray(base, float), np.asarray(absent, bool))
        cols.append(np.column_stack([(rk <= MF_RANKS[0]).astype(float)]
                                    + [(rk == k).astype(float) for k in MF_RANKS[1:]]))
    return np.column_stack(cols)


def mf_pplay(q: pd.DataFrame, params: MinutesParams, base: np.ndarray, absent: np.ndarray
             ) -> np.ndarray:
    """P(plays | available) of the minfix model (q: panel with avail_rank / n_avail)."""
    mf = params.minfix
    X = mf_design(q, params, base, absent, mf["pplay"])
    pr = 1.0 / (1.0 + np.exp(-(X @ np.asarray(mf["pplay_coef"], float))))
    return np.clip(pr, 0.01, 0.995)


def _mf_fit_pplay(prm: MinutesParams, q: pd.DataFrame, plan: np.ndarray, absent: np.ndarray,
                  keep: np.ndarray, y: np.ndarray, kind: str) -> dict:
    """Fit the minfix P(plays) logit on the training rows `keep` (plans = out-of-fold GBM
    bases; the depth rank by plan among the training panel's available rows)."""
    if kind not in ("logit", "logit_rank"):
        raise ValueError(f"unknown minfix pplay {kind!r}")
    X = mf_design(q, prm, plan, absent, kind)[keep]
    return {"pplay_coef": _logit(X, y).tolist()}


def _mf_sim_stats(prm: MinutesParams, r: pd.DataFrame, sims: int, seed: int
                  ) -> pd.DataFrame:
    """Per played training row: the mean and 10 / 90 % quantiles of its simulated minutes given
    that it plays (`sample_minutes` at the game's actual margin and overtime), the actual box
    minutes and the tier of its plan."""
    rng = np.random.default_rng(seed)
    out = []
    cols = ["p_play", "pf36", "reserve", "mean0"]
    ms = r["margin"].to_numpy(float)
    ot = r["n_ot"].fillna(0).to_numpy(float)
    for _, idx in r.groupby("key", sort=False).indices.items():
        d = r.iloc[idx]
        pl = d["played"].to_numpy(bool) & (d["mean0"].to_numpy(float) > 0)
        if not pl.any():
            continue
        proj = d[cols].rename(columns={"mean0": "mean"})
        x = sample_minutes(proj, prm, np.full(sims, ms[idx[0]]), rng,
                           n_ot=np.full(sims, ot[idx[0]]))
        x = np.where(x > 0, x, np.nan)[:, pl]
        ok = np.isfinite(x).sum(0) >= 5
        if not ok.any():
            continue
        with np.errstate(all="ignore"):
            q = np.nanquantile(x[:, ok], [0.1, 0.9], axis=0)
            mu = np.nanmean(x[:, ok], axis=0)
        out.append(pd.DataFrame({"mean0": d["mean0"].to_numpy(float)[pl][ok], "sim": mu,
                                 "q10": q[0], "q90": q[1],
                                 "y": d["act_min"].to_numpy(float)[pl][ok]}))
    f = pd.concat(out, ignore_index=True)
    f["tier"] = tier_of(f["mean0"].to_numpy(float))
    return f


def _mf_calibrate_noise(prm: MinutesParams, r: pd.DataFrame, log: Callable[[str], None],
                        sims: int = MF_CAL_SIMS, iters: int = MF_CAL_ITERS) -> dict:
    """Location (mu) and scale (sigma) of the noise per tier, calibrated by simulation on the
    training projections `r` (out-of-fold plans, the oracle scenario, the actual margin and
    overtime of each game): per tier, the simulated minutes given that a player plays match the
    actual box minutes of the rows that played on average, and their 10-90 % interval covers
    80 % of them. The margin curve and the early exits stay as fitted."""
    from scipy.stats import norm
    nt = len(TIER_EDGES) + 1
    z80 = float(norm.ppf(0.9))
    mu = np.asarray(prm.mu, float).copy()
    sg = np.asarray(prm.sigma, float).copy()
    hist = []
    for it in range(iters):
        prm.mu, prm.sigma = mu.tolist(), sg.tolist()
        f = _mf_sim_stats(prm, r, sims, seed=17 + it)
        step = []
        for t in range(nt):
            q = f[f["tier"] == t]
            if len(q) < 200:
                step.append((t, len(q), 1.0, -1.0))           # too few rows: unchanged
                continue
            ratio = float(q["y"].mean() / max(q["sim"].mean(), 1e-6))
            cov = float(((q["q10"] <= q["y"]) & (q["y"] <= q["q90"])).mean())
            z = float(norm.ppf(np.clip((1.0 + cov) / 2.0, 0.55, 0.995)))
            mu[t] += np.log(np.clip(ratio, 0.8, 1.25))
            sg[t] = float(np.clip(sg[t] * np.clip(z80 / z, 0.7, 1.4), 0.02, 2.0))
            step.append((t, len(q), ratio, cov))
        hist.append(step)
        log(f"  fit_v1 minfix: noise calibration {it}: " + ", ".join(
            f"t{t} n {n} mean x{rt:.3f} cov {cv:.3f}" for t, n, rt, cv in step))
    prm.mu, prm.sigma = mu.tolist(), sg.tolist()
    return {"calib": [[[int(a), int(b), round(float(c), 4), round(float(d), 4)]
                       for a, b, c, d in st] for st in hist]}


# --- blowout minutes (bo; pipeline.BLOWOUT, env COURTSIDE_BLOWOUT=1, default OFF) ----------
# The margin curve g is fit on box residuals and has the right shape (the top tier +1.5 min in
# close games, -6 at 30+), but the sims lose most of it: P(plays | available) does not move with
# the margin (the deep bench that plays in blowouts is drawn DNP at its average rate) and the
# per-sim proportional water-filling to 240 hands the stars' lost minutes back (2024-25, OFF,
# minutes at the actual margin: stars -2.2 min too many at |m| >= 26, +1.6 too few at 0-9). The
# blowout stage, fit on the 5 seasons < S on the same out-of-fold training projections as the
# noise: (1) P(plays) logit offsets by P band x won / lost x |margin| bin (BO_P_EDGES; each
# band's offsets centred, so its level is the arm's), (2) the team's minutes outside the
# candidate set by |margin| bin (centred), (3) the margin curve recalibrated by simulation
# (sample_minutes at each game's actual margin, regulation games) so that the simulated minutes
# of the players who play match the actual ones by tier x won / lost x bin around each tier's
# level, which stays the arm's own simulated level (shape only: mean-preserving), and (4) the
# noise scale per tier rescaled so that the margin-integrated spread of the minutes given play
# stays the arm's (variance moves from the noise to the margin). The copula then reads the
# per-minute multiplier f(m / mpg) at the minutes net of the margin curve (`bo_minutes_factor`):
# f was fit on realised minutes, where a low ratio is mostly an early exit with a lower rate,
# and garbage-time losses keep the per-minute rate.

BO_VERSION = 1
BO_P_EDGES = (0.0, 0.3, 0.6, 0.85, 0.95)     # P(plays | available) bands of the offsets
BO_RIDGE = 2.0                               # ridge of each offset (logit units^-2)
BO_RES_K = 50.0                              # pseudo team-games of the reserve offsets
BO_CELL_K = 200.0                            # pseudo-rows of a margin-curve cell's update
BO_CAL_SIMS = 60
BO_CAL_ITERS = 5


def _bo_mids() -> np.ndarray:
    edges = np.asarray(MARGIN_BINS, float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[-1] = edges[-2] + 4.0
    return mids


def _bo_bin(margins) -> np.ndarray:
    return np.clip(np.searchsorted(MARGIN_BINS, np.abs(np.asarray(margins, float)),
                                   side="right") - 1, 0, len(MARGIN_BINS) - 2)


def _bo_band(p) -> np.ndarray:
    return np.clip(np.searchsorted(BO_P_EDGES, np.asarray(p, float), side="right") - 1, 0,
                   len(BO_P_EDGES) - 1)


def bo_pplay(bo: dict, p: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """(S, n) P(plays | available) at each sim's team margin: logit(p) + the offset of p's band
    and the margin's side, linear in |m| between bin centres; p = 0 / 1 stay 0 / 1."""
    p = np.asarray(p, float)
    m = np.atleast_1d(np.asarray(margins, float))
    a = np.asarray(bo["p_off"], float)                  # (K, 2, B)
    band = _bo_band(p)
    mids = _bo_mids()
    am, won = np.abs(m), (m > 0).astype(int)
    off = np.zeros((len(m), len(p)))
    for r in (0, 1):
        sel = won == r
        if not sel.any():
            continue
        for k in np.unique(band):
            off[np.ix_(sel, band == k)] = np.interp(am[sel], mids, a[k, r])[:, None]
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    out = 1.0 / (1.0 + np.exp(-(np.log(pc / (1.0 - pc))[None, :] + off)))
    return np.where(p[None, :] <= 0.0, 0.0, np.where(p[None, :] >= 1.0, 1.0, out))


def bo_reserve(bo: dict, margins: np.ndarray) -> np.ndarray:
    """(S,) extra team minutes outside the candidate set at each sim's |margin| (centred)."""
    return np.interp(np.abs(np.asarray(margins, float)), _bo_mids(),
                     np.asarray(bo["res_off"], float))


def bo_minutes_factor(params: MinutesParams, proj: pd.DataFrame, margins: np.ndarray
                      ) -> np.ndarray:
    """(S, n) plan / (plan + g_bo(margin)): the factor that takes a sim's minutes back to the
    minutes it would have had without the margin curve (the copula reads the per-minute
    multiplier f(m / mpg) there, so a margin-driven change - garbage time - keeps the player's
    per-minute rate; f stays for the unplanned part, early exits and noise). 1 where the plan
    is 0."""
    bo = params.blowout
    mean = proj["mean"].to_numpy(float)
    gv = g_value(params, tier_of(mean), np.asarray(margins, float), bo["g_table"],
                 bo["g_center"]).reshape(len(margins), len(mean))
    lvl = np.maximum(mean[None, :] + gv, 0.5)
    return np.where(mean[None, :] > 0, np.clip(mean[None, :] / lvl, 0.5, 2.0), 1.0)


def bo_minutes_factor_rows(params: MinutesParams, plan: np.ndarray, margins: np.ndarray
                           ) -> np.ndarray:
    """`bo_minutes_factor` per row (each row its own plan and team margin): plan / (plan +
    g_bo(margin)), 1 where the plan is 0 (training frames: the plan is the player's mpg)."""
    bo = params.blowout
    plan = np.nan_to_num(np.asarray(plan, float))
    gv = _g_rows(params, tier_of(plan), margins, bo["g_table"], bo["g_center"])
    lvl = np.maximum(plan + gv, 0.5)
    return np.where(plan > 0, np.clip(plan / lvl, 0.5, 2.0), 1.0)


def _bo_fit_pplay(r: pd.DataFrame, iters: int = 25) -> tuple[list, list]:
    """Per (P band, side, |margin| bin): the logit offset on top of each available training
    row's P(plays) (separable 1-D ridge Newton per cell), then each band's offsets centred
    (count-weighted) so the band keeps its level. Returns (offsets, counts)."""
    q = r[~r["absent"].to_numpy(bool) & (r["mean0"].to_numpy(float) > 0)
          & (r["p_play"].to_numpy(float) > 0)]
    p = np.clip(q["p_play"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    lp = np.log(p / (1.0 - p))
    y = q["played"].to_numpy(float)
    m = q["margin"].to_numpy(float)
    K, B = len(BO_P_EDGES), len(MARGIN_BINS) - 1
    c = (_bo_band(p) * 2 + (m > 0).astype(int)) * B + _bo_bin(m)
    a = np.zeros(K * 2 * B)
    n = np.bincount(c, minlength=len(a)).astype(float)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(lp + a[c])))
        g = np.bincount(c, y - mu, len(a)) - BO_RIDGE * a
        hs = np.bincount(c, mu * (1.0 - mu), len(a)) + BO_RIDGE
        a = a + g / hs
    a = a.reshape(K, 2, B)
    n = n.reshape(K, 2, B)
    for k in range(K):
        if n[k].sum() > 0:
            a[k] -= (a[k] * n[k]).sum() / n[k].sum()
    return a.tolist(), n.astype(int).tolist()


def _bo_fit_reserve(r: pd.DataFrame) -> list:
    """Per |margin| bin: the mean of (team minutes outside the candidates - the reserve), shrunk
    to the overall mean (BO_RES_K team-games), centred (count-weighted)."""
    k = r.assign(_y=r["act_min"].fillna(0.0)).groupby("key").agg(
        margin=("margin", "first"), n_ot=("n_ot", "first"), act=("_y", "sum"),
        res=("reserve", "first"))
    out = TEAM_MIN + OT_TEAM_MIN * k["n_ot"].fillna(0).to_numpy(float) - k["act"].to_numpy(float)
    e = out - k["res"].to_numpy(float)
    b = _bo_bin(k["margin"].to_numpy(float))
    B = len(MARGIN_BINS) - 1
    n = np.bincount(b, minlength=B).astype(float)
    tot = np.bincount(b, e, B)
    mean = float(e.mean())
    v = (tot + BO_RES_K * mean) / (n + BO_RES_K)
    v = v - (v * n).sum() / max(n.sum(), 1.0)
    return v.tolist()


def _bo_sim_stats(prm: MinutesParams, r: pd.DataFrame, sims: int, seed: int,
                  margins: dict | None = None) -> pd.DataFrame:
    """Per played row of the regulation training games: the mean of its simulated minutes given
    that it plays (`sample_minutes` at the game's actual margin, no overtime), the box minutes,
    the tier of its plan, the side and the |margin| bin. `margins` ((sims,) per key): simulate
    at these margins instead (draws from the margin mix of the training games) and add the 10 /
    90 % quantiles of the minutes given play (rows with >= 10 played sims)."""
    rng = np.random.default_rng(seed)
    rr = r[r["n_ot"].fillna(0).to_numpy(float) == 0]
    cols = ["p_play", "pf36", "reserve", "mean0"]
    ms = rr["margin"].to_numpy(float)
    keys = rr["key"].to_numpy()
    out = []
    for idx in rr.groupby("key", sort=False).indices.values():
        d = rr.iloc[idx]
        pl = d["played"].to_numpy(bool) & (d["mean0"].to_numpy(float) > 0)
        if not pl.any():
            continue
        m = np.full(sims, ms[idx[0]]) if margins is None else margins[keys[idx[0]]]
        x = sample_minutes(d[cols].rename(columns={"mean0": "mean"}), prm, m, rng)[:, pl]
        k = (x > 0).sum(0)
        ok = k >= (3 if margins is None else 10)
        if not ok.any():
            continue
        fr = pd.DataFrame({"mean0": d["mean0"].to_numpy(float)[pl][ok],
                           "sim": x.sum(0)[ok] / k[ok],
                           "y": d["act_min"].to_numpy(float)[pl][ok], "m": ms[idx[0]]})
        if margins is not None:
            with np.errstate(all="ignore"):
                q = np.nanquantile(np.where(x[:, ok] > 0, x[:, ok], np.nan), [0.1, 0.9], axis=0)
            fr["q10"], fr["q90"] = q[0], q[1]
        out.append(fr)
    f = pd.concat(out, ignore_index=True)
    f["tier"] = tier_of(f["mean0"].to_numpy(float))
    f["won"] = (f["m"].to_numpy(float) > 0).astype(int)
    f["bin"] = _bo_bin(f["m"].to_numpy(float))
    return f


def _bo_cov(f: pd.DataFrame, nt: int) -> np.ndarray:
    """Per tier: share of played rows whose box minutes are inside the sims' 10-90 % (given
    play); NaN for tiers with < 200 rows."""
    ok = f["q10"].notna() & f["q90"].notna()
    c = ((f["q10"] <= f["y"]) & (f["y"] <= f["q90"]))[ok].groupby(f["tier"][ok])
    n, v = c.size(), c.mean()
    return np.array([float(v[t]) if t in v.index and n[t] >= 200 else np.nan for t in range(nt)])


def _bo_fit(prm: MinutesParams, r: pd.DataFrame, log: Callable[[str], None],
            sims: int = BO_CAL_SIMS, iters: int = BO_CAL_ITERS) -> dict:
    """The blowout stage on the projected training panel `r` (out-of-fold plans, the oracle
    scenario, each game's actual margin): P(plays) offsets, reserve offsets, then the margin
    curve calibrated by simulation (module comment above) and the noise scale per tier
    rescaled so that the minutes given play, simulated over the margin mix of the training games
    (each sim at a random training game's margin), keep the arm's own 10-90 % coverage: the
    variance the margin now explains comes out of the noise (the arm's noise was fit with the
    margin effect in it). Returns `MinutesParams.blowout`."""
    from scipy.stats import norm
    nt = len(TIER_EDGES) + 1
    rr = r[r["n_ot"].fillna(0).to_numpy(float) == 0]
    km = rr.groupby("key", sort=False)["margin"].first()
    pool = km.to_numpy(float)
    rg = np.random.default_rng(41)
    shuf = {k: rg.choice(pool, sims) for k in km.index}      # each sim: a training game's margin
    p_off, p_n = _bo_fit_pplay(r)
    res_off = _bo_fit_reserve(r)
    log(f"  fit_v1 blowout: reserve offsets {np.round(res_off, 2).tolist()}")
    for k in range(len(BO_P_EDGES)):
        log(f"  fit_v1 blowout: P offsets band {k} lost {np.round(p_off[k][0], 2).tolist()} "
            f"won {np.round(p_off[k][1], 2).tolist()}")
    saved = prm.blowout
    prm.blowout = {}
    f0 = _bo_sim_stats(prm, r, sims, seed=29)             # the arm's own sims: tier levels
    lvl0 = f0.groupby("tier")["sim"].mean().reindex(range(nt)).to_numpy(float)
    cov0 = _bo_cov(_bo_sim_stats(prm, r, sims, seed=30, margins=shuf), nt)
    tab = np.asarray(prm.g_table, float).copy()
    sg = np.asarray(prm.sigma, float).copy()
    prm.blowout = {"version": BO_VERSION, "p_off": p_off, "res_off": res_off,
                   "g_table": tab.tolist(), "g_center": list(prm.g_center),
                   "sigma": sg.tolist()}
    hist = []
    for it in range(iters + 1):
        fs = _bo_sim_stats(prm, r, sims, seed=131 + it, margins=shuf)
        cov = _bo_cov(fs, nt)
        if it < iters:
            for t in range(nt):
                if np.isfinite(cov[t]) and np.isfinite(cov0[t]):
                    z0 = norm.ppf(np.clip((1.0 + cov0[t]) / 2.0, 0.55, 0.995))
                    z1 = norm.ppf(np.clip((1.0 + cov[t]) / 2.0, 0.55, 0.995))
                    sg[t] = float(np.clip(sg[t] * np.clip(z0 / z1, 0.8, 1.25), 0.02, 2.0))
            prm.blowout["sigma"] = sg.tolist()
        log(f"  fit_v1 blowout: noise {it}: coverage at the margin mix "
            f"{np.round(cov, 3).tolist()} vs the arm's {np.round(cov0, 3).tolist()}, sigma "
            f"{np.round(sg, 3).tolist()}")
        f = _bo_sim_stats(prm, r, sims, seed=31 + it)
        g = f.groupby(["tier", "won", "bin"])
        n = g.size()
        d = (g["y"].mean() - g["sim"].mean())
        lt = f.groupby("tier")
        lv_y, lv_s = lt["y"].mean(), lt["sim"].mean()
        worst = 0.0
        for (t, w, b), cnt in n.items():
            if t >= nt or not np.isfinite(lvl0[t]):
                continue
            upd = (d[(t, w, b)] - (lv_y[t] - lv_s[t])) + (lvl0[t] - lv_s[t])
            worst = max(worst, abs(upd) * min(1.0, cnt / 400.0))
            if it < iters:
                tab[t, w, b] += cnt / (cnt + BO_CELL_K) * upd
        hist.append(round(worst, 3))
        log(f"  fit_v1 blowout: g calibration {it}: worst cell gap {worst:.3f}, tier level "
            f"sim - arm {np.round(lv_s.reindex(range(nt)).to_numpy() - lvl0, 3).tolist()}")
        if it < iters:
            prm.blowout["g_table"] = tab.tolist()
    out = dict(prm.blowout, p_n=p_n, calib=hist, level0=np.round(lvl0, 4).tolist(),
               cov0=np.round(cov0, 4).tolist())
    prm.blowout = saved
    return out


def departed(h: History, panel: pd.DataFrame) -> np.ndarray:
    """Training candidates that the live path would not have (pipeline.drop_moved_players
    removes players whose current roster team differs): the player's next regular-season
    appearance on/after the date in this season is for another team, or (at his first game of
    a new season) he does not appear for anyone this season. Uses later box rows only to
    reconstruct the game-day roster, never as a feature."""
    hs = h.hist[h.hist["season_type"] == "Regular Season"][["player_id", "date", "team", "season"]]
    hs = hs.rename(columns={"date": "n_date", "team": "n_team", "season": "n_season"})
    q = pd.DataFrame({"player_id": panel["player_id"].to_numpy(np.int64),
                      "date": panel["date"].to_numpy(np.int64), "i": np.arange(len(panel))})
    q = pd.merge_asof(q.sort_values("date"), hs.sort_values("n_date").astype(
        {"player_id": np.int64, "n_date": np.int64}), left_on="date", right_on="n_date",
        by="player_id", direction="forward").sort_values("i")
    same = (q["n_season"].to_numpy() == panel["season"].to_numpy())
    moved = same & (q["n_team"].to_numpy() != panel["team"].to_numpy())
    gone = ~same & (panel["new_season"].to_numpy() > 0)
    roster = panel["roster"].to_numpy() == 1 if "roster" in panel else np.ones(len(panel), bool)
    return roster & (moved | gone)


class V1Base(np.ndarray):
    """v1 base minutes carrying `s_rows` = P(plays | available) x the v0.5 linear base (0 if
    absent): the input of the reserve (minutes of players outside the candidate set), which v1
    keeps from v0.5. project_arrays reads it, so the copula path (base_and_pplay ->
    project_arrays) needs no change. Slices lose it; run_projection passes it explicitly."""
    s_rows = None
    res_add = None      # per row: the team debut reserve (reserve_debut), see project_arrays


# --- v1: LightGBM base with scenario features (T2.2) ------------------------------------------

V1_MIN_GP = 3            # S15: fewer career games (or none) -> the v0.5 base for that player
V1_ROLE = ("ew_f", "ew_s", "ew_vf", "last", "med5", "min_sd", "st10", "start_last", "ew_start",
           "ew_bench", "sea_mean_now", "gp_season", "n_gp", "pf36", "app10", "team_rank",
           "base_ew", "last_absm", "last_pf")
V1_RETURN = ("missed_now", "gsr_now", "abs_len_now")
V1_SCHEDULE = ("rest", "b2b", "g_last3d", "age", "tgs", "new_season", "new_team")
V1_COACH = ("c_top7", "c_n30", "c_games", "c_team_n", "t_top7", "t_n30")
V1_GROUP = ("g1", "g2", "tgt_g")
V1_SCENARIO = ("vac_tot", "vac_same", "vac_g0", "vac_g1", "vac_g2", "vac_new", "n_st_out",
               "vac_st_same", "rank_after", "grp_rank_after", "rank_gain", "slack",
               "n_rot_avail", "exp_start", "top_in_grp")
V1_SHARE = ("sv", "sh_max")
V1_FEATURES = V1_ROLE + V1_RETURN + V1_SCHEDULE + V1_COACH + V1_GROUP + V1_SCENARIO
GBM_PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 5.0, "num_threads": 4, "seed": 7,
              "deterministic": True, "force_col_wise": True, "verbose": -1}
_BOOSTERS: dict = {}
V1_STATS = {"rows": 0, "fallback_rows": 0, "panel_fallback_calls": 0}


def has_v1_columns(p: pd.DataFrame) -> bool:
    return all(c in p for c in ("ew_vf", "abs_len_now", "team_rank", "c_top7", "tgs"))


def v1_ok(p: pd.DataFrame) -> np.ndarray:
    """Players the GBM projects (the rest fall back to v0.5, S15)."""
    return (p["last_date"].notna() & (p["n_gp"].to_numpy(float) >= V1_MIN_GP)).to_numpy()


def scenario_features(p: pd.DataFrame, absent: np.ndarray, key: str = "key") -> pd.DataFrame:
    """Per row, given the absent set: vacated minutes (EWMA blend of the absent players) in
    total, by group (G/F/C), in his own groups, of newly absent players (played the team's last
    game) and of absent expected starters in his groups; number of expected starters out
    (top 5 by starts in last 10, then EWMA, over all candidates); his depth rank among the
    available (overall and in his primary group) and the gain vs the healthy rank; minutes
    slack (240 - sum of available EWMAs); available rotation players (EWMA >= 15); and, if the
    panel has `share_map`, sv = sum_j share[i, j] x EWMA_j over absent j and the max share."""
    ab = np.asarray(absent, bool)
    n = len(p)
    ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
    kk = p[key].to_numpy() if key in p else np.zeros(n)
    codes, uniq = pd.factorize(kk)
    nk = max(len(uniq), 1)
    g1 = p["g1"].to_numpy(int)
    g2 = p["g2"].to_numpy(int)
    w = np.where(ab, ew, 0.0)

    def tot(x):
        return np.bincount(codes, x, nk)[codes]

    vg = np.stack([tot(w * (g1 == g)) for g in range(3)], 1)
    vac_same = vg[np.arange(n), np.clip(g1, 0, 2)] + np.where(
        g2 != g1, vg[np.arange(n), np.clip(g2, 0, 2)], 0.0)
    order = p["st10"].to_numpy(float) * 100.0 + ew
    r_all = pd.Series(order).groupby(codes).rank(ascending=False, method="first").to_numpy()
    exp_start = r_all <= 5
    st_out = ab & exp_start
    vs = np.stack([tot(np.where(st_out & (g1 == g), ew, 0.0)) for g in range(3)], 1)
    vac_st_same = vs[np.arange(n), np.clip(g1, 0, 2)] + np.where(
        g2 != g1, vs[np.arange(n), np.clip(g2, 0, 2)], 0.0)
    av_ew = pd.Series(np.where(ab, np.nan, ew))
    rank_after = av_ew.groupby(codes).rank(ascending=False, method="first").to_numpy()
    grp_rank = av_ew.groupby([codes, g1]).rank(ascending=False, method="first").to_numpy()
    team_rank = p["team_rank"].to_numpy(float) if "team_rank" in p else \
        pd.Series(ew).groupby(codes).rank(ascending=False, method="first").to_numpy()
    newly = p["missed_now"].to_numpy(float) == 0 if "missed_now" in p else np.ones(n, bool)
    out = {"vac_tot": tot(w), "vac_g0": vg[:, 0], "vac_g1": vg[:, 1], "vac_g2": vg[:, 2],
           "vac_same": vac_same, "vac_new": tot(np.where(newly, w, 0.0)),
           "n_st_out": tot(st_out.astype(float)), "vac_st_same": vac_st_same,
           "rank_after": rank_after, "grp_rank_after": grp_rank,
           "rank_gain": team_rank - rank_after,
           "slack": TEAM_MIN - tot(np.where(ab, 0.0, ew)),
           "n_rot_avail": tot(((~ab) & (ew >= 15)).astype(float)),
           "exp_start": exp_start.astype(float), "top_in_grp": (grp_rank == 1).astype(float)}
    if "share_map" in p:
        sv = np.zeros(n)
        smax = np.zeros(n)
        pid = p["player_id"].to_numpy(np.int64)
        for _, idx in pd.Series(np.arange(n)).groupby(codes).indices.items():
            a = idx[ab[idx]]
            if not len(a):
                continue
            ew_ab = dict(zip(pid[a].tolist(), ew[a].tolist()))
            for i in idx[~ab[idx]]:
                m = p["share_map"].iat[i]
                if not m:
                    continue
                vals = [(s, ew_ab[j]) for j, s in m.items() if j in ew_ab]
                if vals:
                    sv[i] = sum(s * e for s, e in vals)
                    smax[i] = max(s for s, _ in vals)
        out["sv"], out["sh_max"] = sv, smax
    else:
        out["sv"] = np.full(n, np.nan)
        out["sh_max"] = np.full(n, np.nan)
    return pd.DataFrame(out, index=p.index)


def v1_matrix(p: pd.DataFrame, absent: np.ndarray, names) -> np.ndarray:
    """Feature matrix (rows of p) in the order `names`."""
    sf = scenario_features(p, absent)
    cols = []
    for c in names:
        if c in sf:
            cols.append(sf[c].to_numpy(float))
        elif c == "tgt_g":
            g = np.clip(p["g1"].to_numpy(int), 0, 2)
            num = np.stack([p[f"tgt_num{k}"].to_numpy(float) if f"tgt_num{k}" in p
                            else np.full(len(p), np.nan) for k in range(3)], 1)
            den = np.stack([p[f"tgt_den{k}"].to_numpy(float) if f"tgt_den{k}" in p
                            else np.full(len(p), np.nan) for k in range(3)], 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = num[np.arange(len(p)), g] / den[np.arange(len(p)), g]
            cols.append(np.where(np.isfinite(r), r, np.nan))
        elif c == "n_gp":
            cols.append(np.log1p(p["n_gp"].to_numpy(float)))
        elif c in p:
            cols.append(pd.to_numeric(p[c], errors="coerce").to_numpy(float))
        else:
            cols.append(np.full(len(p), np.nan))
    return np.column_stack(cols) if cols else np.zeros((len(p), 0))


def _booster(model: str):
    b = _BOOSTERS.get(model)
    if b is None:
        import lightgbm as lgb
        if len(_BOOSTERS) > 8:
            _BOOSTERS.clear()
        b = lgb.Booster(model_str=model)
        _BOOSTERS[model] = b
    return b


def gbm_predict(params: MinutesParams, X: np.ndarray) -> np.ndarray:
    if not len(X):
        return np.zeros(0)
    nt = 4 if len(X) > 20_000 else 1
    return _booster(params.gbm_model).predict(X, num_threads=nt)


def v1_base(p: pd.DataFrame, params: MinutesParams, absent: np.ndarray, base0: np.ndarray
            ) -> np.ndarray:
    """GBM E[min | plays] where v1_ok, else base0 (the v0.5 linear base). A training-time
    column `_v1_oof` (out-of-fold predictions) is used verbatim when present."""
    if not has_v1_columns(p):
        V1_STATS["panel_fallback_calls"] += 1
        V1_STATS["rows"] += len(p)
        V1_STATS["fallback_rows"] += len(p)
        return base0
    ok = v1_ok(p) & ~absent
    base = base0.copy()
    if "_v1_oof" in p:
        o = p["_v1_oof"].to_numpy(float)
        ok &= np.isfinite(o)
        base[ok] = o[ok]
    elif ok.any():
        X = v1_matrix(p, absent, params.gbm_features)
        base[ok] = gbm_predict(params, X[ok])
    V1_STATS["rows"] += int((~absent).sum())
    V1_STATS["fallback_rows"] += int((~absent & ~ok).sum())
    return np.clip(base, 0.5, 44.0)


def reserve_design(s_avail: np.ndarray, n_avail: np.ndarray) -> np.ndarray:
    """Spline features for minutes going to players outside the candidate set (call-ups, new
    signings): s_avail = sum of p x base over available candidates, n_avail = their count."""
    s_avail = np.atleast_1d(np.asarray(s_avail, float))
    n_avail = np.atleast_1d(np.asarray(n_avail, float))
    return np.column_stack([np.ones(len(s_avail)), np.maximum(0, 220 - s_avail),
                            np.maximum(0, 190 - s_avail), np.maximum(0, 10 - n_avail)])


def reserve_minutes(params: MinutesParams, s_avail, n_avail) -> np.ndarray:
    r = reserve_design(s_avail, n_avail) @ np.asarray(params.reserve_coef, float)
    return np.clip(r, 0.0, 150.0)


NMU_FEATURES = ("same1", "same2", "adjacent", "ew", "st10", "share", "top_in_group")


def nmu_features(g1, g2, ew, st10, share_j, gj: int, cand: np.ndarray) -> np.ndarray:
    """Next-man-up choice features of candidates `cand` (indices) for absent starter of group
    gj: same primary group, same secondary group, adjacent group, EWMA, starts in last 10,
    share of j's minutes when j sat, best EWMA among same-group candidates."""
    c = cand
    same1 = (g1[c] == gj).astype(float)
    same2 = ((g2[c] == gj) & (g1[c] != gj)).astype(float)
    adjacent = (np.isin(g1[c], ADJ[gj]) & (same1 == 0) & (same2 == 0)).astype(float)
    sg = (same1 + same2) > 0
    top = np.zeros(len(c))
    if sg.any():
        top[np.flatnonzero(sg)[np.argmax(ew[c][sg])]] = 1.0
    sh = share_j[c] if share_j is not None else np.zeros(len(c))
    return np.column_stack([same1, same2, adjacent, ew[c] / 10.0, st10[c], sh, top])


def project_arrays(base, p_avail, absent, g1, g2, st10, ew, tgt, params: MinutesParams,
                   starters=None, share=None, n_ot: float = 0.0, detail: bool = False,
                   s_reserve: float | None = None, reserve_add: float | None = None):
    """Scenario projection for one team-game.

    base, p_avail (P plays | available), g1/g2 (groups), st10, ew (minutes EWMA blend): arrays
    over the candidates; absent: bool mask; tgt: (n, 3) team starter targets (identical rows);
    starters: optional bool mask of confirmed starters; share: optional (n, n) matrix,
    share[i, j] = i's historical share of j's minutes when j sat.
    Returns mean minutes conditional on playing (0 for absent) [and a detail dict].
    s_reserve: the reserve input (sum p x base of the available); default `base.s_rows` (v1,
    the v0.5 base) when present, else sum p x base. reserve_add: team minutes of debutants
    added to the reserve (params.reserve_debut); default `base.res_add` when present, else 0.

    Next man up (S14), per absent expected starter j (in descending base):
      params.nmu_mode == "top": the top candidate of j's group (then adjacent groups) by
        starts in last 10, EWMA and share gets the full lift (the literal §4.2 rule);
      "softmax" (default): candidates get P(u replaces j) from a conditional logit fit on
        seasons < S, and an expected lift P x (0.5 target_g + 0.5 EWMA_u - base_u)+.
      Confirmed starters (lock-10): the unexpected starter of j's group (else any) gets the
        full lift; non-starters get none.
    The rest of j's minutes: f_same to j's group, the remainder to adjacent groups, by
    p x minutes x (1 + 4 x share)."""
    n = len(base)
    absent = np.asarray(absent, bool)
    avail = ~absent
    adj = base.astype(float).copy()
    p = np.where(avail, p_avail, 0.0)
    order_key = st10 * 100.0 + ew
    exp_start = np.zeros(n, bool)
    exp_start[np.argsort(-order_key, kind="stable")[:5]] = True
    lift = np.zeros(n)
    pstart = np.zeros(n)            # P(u starts in place of an absent starter)
    t = tgt[0] if len(tgt) else np.asarray(params.starter_league)
    in_g = lambda g: (g1 == g) | (g2 == g)  # noqa: E731
    starters = None if starters is None else np.asarray(starters, bool)
    unexpected = (starters & ~exp_start & avail) if starters is not None else np.zeros(n, bool)
    used = np.zeros(n, bool)
    if starters is not None:
        demote = exp_start & avail & ~starters
        adj[demote] = np.minimum(adj[demote], 0.5 * adj[demote] + 0.5 * params.bench_target)
    js = np.flatnonzero(absent & exp_start & (base >= ROTATION_MIN))
    js = js[np.argsort(-base[js], kind="stable")]
    pool = avail & (p > 0.05) & ~exp_start
    mass = np.ones(n)
    beta = np.asarray(params.nmu_coef, float)
    for j in js:
        vac = base[j]
        gj = g1[j]
        sh_j = share[:, j] if share is not None else None
        probs = np.zeros(n)
        if starters is not None:
            pu = np.flatnonzero(unexpected & ~used & in_g(gj))
            if not len(pu):
                pu = np.flatnonzero(unexpected & ~used)
            if len(pu):
                u = int(pu[np.argmax(ew[pu])])
                used[u] = True
                probs[u] = 1.0
        else:
            cand = np.flatnonzero(pool & (mass > 1e-6))
            if len(cand):
                if params.nmu_mode == "top":
                    for groups in ((gj,), ADJ[gj]):
                        m = np.isin(g1[cand], groups) | np.isin(g2[cand], groups)
                        if m.any():
                            c2 = cand[m]
                            sh = sh_j[c2] if sh_j is not None else 0.0
                            key = st10[c2] * 100.0 + ew[c2] + params.share_weight * sh
                            u = int(c2[np.argmax(key)])
                            probs[u] = 1.0
                            mass[u] = 0.0
                            break
                else:
                    X = nmu_features(g1, g2, ew, st10, sh_j, gj, cand)
                    z = X @ beta
                    w = np.exp(z - z.max()) * mass[cand]
                    if w.sum() > 0:
                        probs[cand] = w / w.sum()
                        mass[cand] *= (1.0 - probs[cand])
        if probs.any():
            new = 0.5 * t[gj] + 0.5 * ew
            gain = np.maximum(new - base, 0.0) * probs
            if starters is None:
                gain = gain * params.nmu_scale     # v1: the GBM already promotes
            gain = np.where(probs > 0, gain, 0.0)
            # never more than the vacated minutes
            if gain.sum() > vac:
                gain *= vac / gain.sum()
            lift += gain
            pstart += probs
            vac -= gain.sum()
        # leftover: f_same to j's group, rest to adjacent groups, by p*adj*(1+share)
        _spread_leftover(adj, lift, p, avail, g1, g2, gj, vac * params.k_add, sh_j, params)
    # absent bench rotation players: leftover only (no individual lift)
    for j in np.flatnonzero(absent & ~exp_start & (base >= ROTATION_MIN)):
        _spread_leftover(adj, lift, p, avail, g1, g2, g1[j], base[j] * params.k_add,
                         share[:, j] if share is not None else None, params)
    # v1: learned pair-share term (who absorbed the absent players' minutes before)
    if params.share_coef and share is not None and absent.any():
        sv = (np.clip(share[:, absent], 0.0, 1.0) * base[absent][None, :]).sum(1)
        lift += np.where(avail, params.share_coef * sv, 0.0)
    # confirmed unexpected starters not used as anyone's replacement: lift to starter level
    if starters is not None:
        for i in np.flatnonzero(unexpected & ~used):
            new = 0.5 * t[g1[i]] + 0.5 * ew[i]
            if new > adj[i] + lift[i]:
                lift[i] += new - (adj[i] + lift[i])
            pstart[i] = 1.0
    adj = adj + lift
    if s_reserve is None:
        sr = getattr(base, "s_rows", None)
        s_reserve = float(np.sum(sr)) if sr is not None and len(sr) == n else None
    s_av = float((p * base).sum()) if s_reserve is None else float(s_reserve)
    reserve = float(reserve_minutes(params, s_av, float(avail.sum()))[0])
    if reserve_add is None:
        ra = getattr(base, "res_add", None)
        reserve_add = float(ra[0]) if ra is not None and len(ra) == n and n else None
    if reserve_add:
        reserve = float(np.clip(reserve + reserve_add, 0.0, 150.0))
    total = TEAM_MIN + OT_TEAM_MIN * n_ot - reserve
    cap = CAP + 5.0 * n_ot
    if params.recon == "proportional":
        exp_min = allocate_minutes(np.where(avail, adj, 0.0), p, total=total, cap=cap)
    else:
        exp_min = reconcile(np.where(avail, adj, 0.0), p, total, cap, cv=params.recon_cv)
    mean = np.where(p > 0, exp_min / np.maximum(p, 1e-9), 0.0)
    mean = np.minimum(mean, cap)
    if not detail:
        return mean
    return mean, {"adj": adj, "p": p, "lift": lift, "promoted": pstart > 0.5,
                  "pstart": np.minimum(pstart, 1.0), "exp_start": exp_start,
                  "reserve": reserve}


def reconcile(a: np.ndarray, p: np.ndarray, total: float, cap: float = CAP,
              cv: float = RECON_CV) -> np.ndarray:
    """Expected minutes e = p x a moved to sum to `total`, each player's move proportional to
    his variance p a (cv^2 + 1 - p) (DNP-prone and bench players absorb most of the slack;
    proportional water-filling over-transmits errors to the starters), with 0 <= e <= p cap."""
    a = np.maximum(np.asarray(a, float), 0.0)
    p = np.asarray(p, float)
    e = p * a
    v = np.where(p > 0, p * np.maximum(a, 1.0) * (cv ** 2 + 1.0 - p), 0.0)
    for _ in range(len(a) + 1):
        if v.sum() <= 0:
            break
        e2 = e + v / v.sum() * (total - e.sum())
        lo, hi = e2 < 0, e2 > cap * p
        e2 = np.clip(e2, 0.0, cap * p)
        e = e2
        if abs(e.sum() - total) < 1e-6:
            break
        v = np.where(lo | hi, 0.0, v)
    if v.sum() <= 0 and e.sum() > 0 and abs(e.sum() - total) > 1e-6:
        e = allocate_minutes(e, np.ones(len(e)), total=total, cap=cap)  # fallback
    return e


def _spread_leftover(adj, lift, p, avail, g1, g2, gj, vac, sh_j, params) -> None:
    if vac <= 0:
        return
    w = p * (adj + lift) * (1.0 + (np.clip(sh_j, 0, 1) * 4.0 if sh_j is not None else 0.0))
    w = np.where(avail, w, 0.0)
    same = (g1 == gj) | (g2 == gj)
    other = (np.isin(g1, ADJ[gj]) | np.isin(g2, ADJ[gj])) & ~same
    ws, wo = w * same, w * other
    fs = params.f_same if wo.sum() > 0 else 1.0
    if ws.sum() <= 0:
        fs = 0.0
    add = np.zeros(len(adj))
    if ws.sum() > 0:
        add += vac * fs * ws / ws.sum()
    if wo.sum() > 0:
        add += vac * (1 - fs) * wo / wo.sum()
    if add.sum() == 0 and w.sum() > 0:
        add = vac * w / w.sum()
    adj += add


# --- margin curve and noise ------------------------------------------------------------------

def g_value(params: MinutesParams, tiers: np.ndarray, margin, tab=None, cen=None
            ) -> np.ndarray:
    """g_tier(|m|, won) - c_tier, linear in |m| between bin centres. margin: scalar or (S,)
    array (team perspective); returns (S, n) (or (n,) for a scalar). tab / cen: another
    (T, 2, B) table and centre (the blowout arm's), default the params' own."""
    tab = np.asarray(params.g_table if tab is None else tab, float)        # (T, 2, B)
    cen = np.asarray(params.g_center if cen is None else cen, float)
    edges = np.asarray(MARGIN_BINS, float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[-1] = edges[-2] + 4.0
    m = np.atleast_1d(np.asarray(margin, float))
    am = np.abs(m)
    won = (m > 0).astype(int)
    out = np.empty((len(m), len(tiers)))
    for r in (0, 1):
        sel = won == r
        if not sel.any():
            continue
        for t in np.unique(tiers):
            cols = tiers == t
            v = np.interp(am[sel], mids, tab[t, r]) - cen[t]
            out[np.ix_(sel, cols)] = v[:, None]
    return out[0] if np.ndim(margin) == 0 else out


def _renorm(x: np.ndarray, total: np.ndarray, cap: np.ndarray, iters: int = 6) -> np.ndarray:
    """Row-wise water-filling of expected minutes x (S, n) to `total` with per-row cap."""
    x = np.maximum(x, 0.0)
    total = np.broadcast_to(np.asarray(total, float).reshape(-1, 1), (x.shape[0], 1))
    cap = np.broadcast_to(np.asarray(cap, float).reshape(-1, 1), (x.shape[0], 1))
    fixed = np.zeros_like(x, bool)
    for _ in range(iters):
        free = ~fixed
        s_free = np.where(free, x, 0.0).sum(1, keepdims=True)
        s_fix = np.where(fixed, x, 0.0).sum(1, keepdims=True)
        f = np.where(s_free > 0, (total - s_fix) / np.maximum(s_free, 1e-9), 1.0)
        x = np.where(free, x * f, x)
        over = free & (x > cap)
        if not over.any():
            break
        x = np.where(over, cap, x)
        fixed |= over
    return x


def expected_minutes(proj: pd.DataFrame, params: MinutesParams, spread: float | None = 0.0,
                     sd: float = MARGIN_SD, n_ot: float | None = None, nodes: int = 21
                     ) -> np.ndarray:
    """E[minutes | plays] integrating g over margin ~ N(spread, sd^2) (spread = expected
    margin, + = this team favoured). n_ot None -> the fitted mean number of OT periods."""
    mean = proj["mean"].to_numpy(float)
    p = proj["p_play"].to_numpy(float) * (mean > 0)
    tiers = tier_of(mean)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    m = (0.0 if spread is None or not np.isfinite(spread) else spread) + sd * x
    lvl = np.maximum(mean[None, :] + g_value(params, tiers, m), 0.0) * (mean > 0)
    ot = params.mean_n_ot if n_ot is None else n_ot
    tot = TEAM_MIN + OT_TEAM_MIN * ot - _reserve_of(proj)
    e = _renorm(lvl * p[None, :], np.full(len(m), tot), np.full(len(m), CAP + 5.0 * ot))
    cond = np.where(p > 0, e / np.maximum(p, 1e-9), 0.0)
    return (w[:, None] * cond).sum(0)


def sample_minutes(proj: pd.DataFrame, params: MinutesParams, margins: np.ndarray,
                   rng: np.random.Generator, n_ot: np.ndarray | None = None,
                   noise: bool = True) -> np.ndarray:
    """Per-sim minutes (S, n): mean + g(margin_s), x exp(mu + sigma xi), early exits,
    DNP-CD draws (1 - p_play), then water-filled to 240 + 25 n_OT_s with a 40 (+5/OT) cap.
    Absent players (mean 0) get 0."""
    mean = proj["mean"].to_numpy(float)
    p = proj["p_play"].to_numpy(float) * (mean > 0)
    pf = proj["pf36"].to_numpy(float) if "pf36" in proj else np.full(len(mean), 3.5)
    margins = np.asarray(margins, float)
    S, n = len(margins), len(mean)
    tiers = tier_of(mean)
    bo = params.blowout
    if bo:      # blowout arm: its calibrated margin curve (same draws below)
        gv = g_value(params, tiers, margins, bo["g_table"], bo["g_center"])
        x = np.maximum(mean[None, :] + gv.reshape(S, n), 0.0)
    else:
        x = np.maximum(mean[None, :] + g_value(params, tiers, margins).reshape(S, n), 0.0)
    if noise:
        mu = np.asarray(params.mu)[tiers]
        sg = np.asarray(bo["sigma"] if bo and "sigma" in bo else params.sigma)[tiers]
        x = x * np.exp(mu[None, :] + sg[None, :] * rng.standard_normal((S, n)))
        q = np.clip(params.q0 + params.q1 * pf, 0.0, 0.5)
        ex = rng.random((S, n)) < q[None, :]
        x = np.where(ex, x * rng.uniform(*EXIT_U, size=(S, n)), x)
    ot = np.zeros(S) if n_ot is None else np.asarray(n_ot, float)
    if bo:      # P(plays) and the reserve move with the sim's margin (common random numbers)
        plays = rng.random((S, n)) < bo_pplay(bo, p, margins)
        x = np.where(plays, x, 0.0)
        return _renorm(x, TEAM_MIN + OT_TEAM_MIN * ot - _reserve_of(proj)
                       - bo_reserve(bo, margins), CAP + 5.0 * ot)
    plays = rng.random((S, n)) < p[None, :]
    x = np.where(plays, x, 0.0)
    return _renorm(x, TEAM_MIN + OT_TEAM_MIN * ot - _reserve_of(proj), CAP + 5.0 * ot)


def _reserve_of(proj: pd.DataFrame) -> float:
    """Team minutes reserved for players outside the candidate set (column `reserve`)."""
    if "reserve" in proj and len(proj):
        return float(proj["reserve"].iloc[0])
    return 0.0


# --- fitting ---------------------------------------------------------------------------------

def _ols(X, y, ridge=1e-3):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _logit(X, y, iters=30, ridge=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ b
        pr = 1 / (1 + np.exp(-z))
        W = pr * (1 - pr)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(len(b))
        g = X.T @ (y - pr) - ridge * b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def eval_targets(h: History, seasons: Iterable[str], regular_only: bool = True) -> pd.DataFrame:
    """Team-game targets (key = game_id|team) for seasons, with the actual results attached."""
    tg = h.tg[h.tg["season"].isin(list(seasons))]
    if regular_only:
        tg = tg[tg["season_type"] == "Regular Season"]
    t = tg[["game_id", "team", "date", "season", "margin", "n_ot"]].copy()
    t["key"] = t["game_id"] + "|" + t["team"]
    return t.reset_index(drop=True)


def oracle_panel(h: History, targets: pd.DataFrame) -> pd.DataFrame:
    """pregame_panel + the avail oracle (players.REGULAR_MIN on the rung-2 EWMA): candidates =
    roster U regulars who played; absent = regular candidates who did not play. Adds actual
    minutes, started and the margin."""
    hist = h.hist
    played = hist.merge(targets[["game_id", "team", "key"]], on=["game_id", "team"])
    # regulars who played but are not on the recent roster -> extras
    p0 = pregame_panel(h, targets, extra=played[["key", "player_id"]])
    act = played[["key", "player_id", "min", "min_r", "started"]].rename(
        columns={"min": "act_min", "min_r": "act_min_r", "started": "act_started"})
    p0 = p0.merge(act, on=["key", "player_id"], how="left")
    p0["played"] = p0["act_min"].notna()
    reg = p0["base_ew"] >= REGULAR_MIN
    keep = (p0["roster"] == 1) | (p0["played"] & reg)
    p0 = p0[keep].copy()
    p0["absent"] = reg[keep] & ~p0["played"]
    p0 = p0.merge(targets[["key", "game_id", "margin", "n_ot"]], on="key", how="left")
    return p0.reset_index(drop=True)


def run_projection(panel: pd.DataFrame, params: MinutesParams, starters: bool = False,
                   absent_col: str = "absent", shares: "PairShares | None" = None,
                   detail: bool = False) -> pd.DataFrame:
    """Project every team-game of an (oracle) panel. Adds base, p_play, mean (+ lift info)."""
    base, pp = base_and_pplay(panel, params, panel[absent_col].to_numpy(bool))
    tgt = _team_target(panel, params)
    out_mean = np.zeros(len(panel))
    out_lift = np.zeros(len(panel))
    out_prom = np.zeros(len(panel), bool)
    out_ps = np.zeros(len(panel))
    out_exp = np.zeros(len(panel), bool)
    out_res = np.zeros(len(panel))
    out_adj = np.zeros(len(panel))
    g1 = panel["g1"].to_numpy(int)
    g2 = panel["g2"].to_numpy(int)
    st10 = panel["st10"].to_numpy(float)
    ew = 0.5 * panel["ew_f"].to_numpy(float) + 0.5 * panel["ew_s"].to_numpy(float)
    ab = panel[absent_col].to_numpy(bool)
    stt = panel["act_started"].fillna(0).to_numpy(bool) if starters else None
    pids = panel["player_id"].to_numpy()
    dates = panel["date"].to_numpy()
    for key, idx in panel.groupby("key", sort=False).indices.items():
        share = shares.matrix(int(dates[idx[0]]), pids[idx], ab[idx]) if shares is not None \
            else None
        sr = getattr(base, "s_rows", None)
        ra = getattr(base, "res_add", None)
        res = project_arrays(np.asarray(base[idx]), pp[idx], ab[idx], g1[idx], g2[idx],
                             st10[idx], ew[idx], tgt[idx], params,
                             starters=None if stt is None else stt[idx], share=share,
                             detail=True,
                             s_reserve=None if sr is None else float(sr[idx].sum()),
                             reserve_add=None if ra is None else float(ra[idx][0]))
        out_mean[idx] = res[0]
        out_lift[idx] = res[1]["lift"]
        out_prom[idx] = res[1]["promoted"]
        out_ps[idx] = res[1]["pstart"]
        out_exp[idx] = res[1]["exp_start"]
        out_res[idx] = res[1]["reserve"]
        out_adj[idx] = res[1]["adj"]
    r = panel.copy()
    r["reserve"] = out_res
    r["adj"] = out_adj

    r["base"] = base
    r["p_play"] = np.where(ab, 0.0, pp)
    r["mean0"] = out_mean           # before the margin integration
    r["lift"] = out_lift
    r["promoted"] = out_prom
    r["pstart"] = out_ps
    r["exp_start"] = out_exp
    return r


def _fit_noise(r: np.ndarray, tiers: np.ndarray, pf: np.ndarray) -> tuple:
    """MLE of per-tier (mu, sigma) and shared (q0, q1) for log ratio r with an early-exit
    mixture: r ~ (1-q) N(mu, s^2) + q [N(mu, s^2) + log U(0.35, 0.85)]."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    nt = len(TIER_EDGES) + 1
    us = np.log(np.linspace(EXIT_U[0], EXIT_U[1], 11))

    def nll(th):
        mu = th[:nt][tiers]
        sg = np.exp(th[nt:2 * nt])[tiers]
        q = np.clip(th[2 * nt] + th[2 * nt + 1] * pf, 1e-4, 0.6)
        f0 = norm.pdf(r, mu, sg)
        f1 = norm.pdf(r[:, None], mu[:, None] + us[None, :], sg[:, None]).mean(1)
        return -np.log((1 - q) * f0 + q * f1 + 1e-300).sum()

    th0 = np.r_[np.zeros(nt), np.log(np.full(nt, 0.2)), 0.05, 0.0]
    res = minimize(nll, th0, method="L-BFGS-B",
                   bounds=[(-1, 1)] * nt + [(np.log(0.02), np.log(2.0))] * nt
                   + [(0.0, 0.5), (-0.1, 0.1)])
    th = res.x
    return (th[:nt].tolist(), np.exp(th[nt:2 * nt]).tolist(), float(th[2 * nt]),
            float(th[2 * nt + 1]), bool(res.success))


def _fit_nmu(r: pd.DataFrame, shares: "PairShares | None", ridge: float = 1.0) -> list:
    """Conditional logit for P(u replaces absent expected starter j) on a projected training
    panel (needs exp_start, p_play, act_started). Choice set: available non-starters with
    p > 0.05; the chosen one: the actual unexpected starter of j's group (best EWMA), else any
    unexpected starter. Starters are matched to absent j in descending base."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    sets = []
    base = r["base"].to_numpy(float)
    g1 = r["g1"].to_numpy(int)
    g2 = r["g2"].to_numpy(int)
    ew = 0.5 * r["ew_f"].to_numpy(float) + 0.5 * r["ew_s"].to_numpy(float)
    st10 = r["st10"].to_numpy(float)
    ab = r["absent"].to_numpy(bool)
    es = r["exp_start"].to_numpy(bool)
    pp = r["p_play"].to_numpy(float)
    ast = r["act_started"].fillna(0).to_numpy(bool)
    pids = r["player_id"].to_numpy()
    dates = r["date"].to_numpy()
    for _, idx in r.groupby("key", sort=False).indices.items():
        jm = ab[idx] & es[idx] & (base[idx] >= ROTATION_MIN)
        if not jm.any():
            continue
        js = np.flatnonzero(jm)
        js = js[np.argsort(-base[idx][js], kind="stable")]
        cand = np.flatnonzero(~ab[idx] & (pp[idx] > 0.05) & ~es[idx])
        pos = set(cand[ast[idx][cand]].tolist())
        if not len(cand) or not pos:
            continue
        sh = shares.matrix(int(dates[idx[0]]), pids[idx], ab[idx]) if shares is not None \
            else None
        G1, G2, EW, ST = g1[idx], g2[idx], ew[idx], st10[idx]
        for j in js:
            if not pos:
                break
            gj = G1[j]
            pl = sorted(pos, key=lambda u: -EW[u])
            same = [u for u in pl if G1[u] == gj or G2[u] == gj]
            u = same[0] if same else pl[0]
            X = nmu_features(G1, G2, EW, ST, sh[:, j] if sh is not None else None, gj, cand)
            sets.append((X, int(np.flatnonzero(cand == u)[0])))
            pos.discard(u)
            cand = cand[cand != u]
            if not len(cand):
                break
    if not sets:
        return default_params().nmu_coef
    k = sets[0][0].shape[1]

    def nll(b):
        f, g = 0.0, np.zeros(k)
        for X, y in sets:
            z = X @ b
            lse = logsumexp(z)
            pr = np.exp(z - lse)
            f -= z[y] - lse
            g -= X[y] - pr @ X
        return f + 0.5 * ridge * b @ b, g + ridge * b

    res = minimize(nll, np.zeros(k), jac=True, method="L-BFGS-B")
    return res.x.tolist()


def fit(h: History, season: str, window: int = 5, log: Callable[[str], None] = print,
        f_grid=(0.4, 0.6, 0.8), panel: pd.DataFrame | None = None,
        shares: "PairShares | None" = None, version: str = "v0.5",
        v1_kw: dict | None = None, debut_reserve: bool = False) -> MinutesParams:
    """Fit every parameter on the `window` seasons before `season` (strictly < season).

    panel / shares: optional precomputed oracle panel of those seasons and pair shares
    (events from seasons < season only). version "v1": the v0.5 fit, then `fit_v1` on top
    (the v0.5 base / P(play) / next-man-up stay as the S15 fallback and the stage-2 inputs).
    debut_reserve (COURTSIDE_PRESEASON): the reserve is split into players with no box row
    before the date (`debut_minutes`, OLS on `_debut_design`: team games into the season x the
    team's new players' preseason minutes -> `reserve_debut`) and everybody else outside the
    candidate set (the usual spline on the available weight, refit without the debutants)."""
    seasons = sorted(s for s in h.tg["season"].unique() if s < season)[-window:]
    if not seasons:
        raise ValueError(f"no seasons before {season}")
    tr = eval_targets(h, seasons)
    if panel is None:
        panel = oracle_panel(h, tr)
    if shares is None:
        shares = PairShares(pair_events(panel))
    pl = panel[panel["played"]]
    X = design(pl, BASE_FEATURES)
    coef = _ols(X, pl["act_min_r"].to_numpy(float))
    base_tr = np.clip(design(panel, BASE_FEATURES) @ coef, 0.5, 44.0)
    panel = panel.copy()
    panel["avail_rank"], panel["n_avail"] = avail_rank(panel, base_tr,
                                                       panel["absent"].to_numpy(bool))
    nonab = panel[~panel["absent"]]
    Z = design(nonab, PPLAY_FEATURES)
    pcoef = _logit(Z, nonab["played"].to_numpy(float))
    # minutes to players outside the candidate set, as a function of the available weight
    pp_tr = 1.0 / (1.0 + np.exp(-(Z @ pcoef)))
    rcoef = _fit_reserve(panel, base_tr, pp_tr)
    rdeb: list = []
    if debut_reserve:
        deb = debut_minutes(h, panel)
        rcoef = _fit_reserve(panel, base_tr, pp_tr, minus=deb)
        kk = panel.drop_duplicates("key")
        d = kk[DEBUT_DRIVER].to_numpy(float) if DEBUT_DRIVER in kk else np.zeros(len(kk))
        d0 = d[(kk["tgs"].to_numpy(float) == 0) & np.isfinite(d)]
        cap = float(np.quantile(d0, DEBUT_DRIVER_Q)) if len(d0) >= 20 else 1e9
        code = {v: c for c, v in _DEBUT_CODES.items()}[DEBUT_DRIVER]
        rdeb = _ols(_debut_design(kk["tgs"].to_numpy(float), d, cap),
                    deb.reindex(kk["key"]).to_numpy(float), ridge=1.0).tolist() + [cap, code]
        log(f"  fit {season}: debut reserve {np.round(rdeb, 3).tolist()}")
    # league starter minutes per group and the top bench player's minutes
    hs = h.hist[h.hist["season"].isin(seasons) & (h.hist["season_type"] == "Regular Season")]
    hs = _attach_groups(hs[["player_id", "season", "game_id", "team", "min_r", "started"]],
                        h.positions)
    sl = [float(hs[(hs["started"] == 1) & (hs["g1"] == g)]["min_r"].mean()) for g in range(3)]
    bench = hs[hs["started"] == 0].groupby(["game_id", "team"])["min_r"].max()
    prm = default_params(season)
    prm.train_seasons = seasons
    prm.base_coef = coef.tolist()
    prm.pplay_coef = pcoef.tolist()
    prm.reserve_coef = rcoef
    prm.reserve_debut = rdeb
    prm.starter_league = sl
    prm.bench_target = float(bench.mean())
    prm.mean_n_ot = float(tr["n_ot"].mean())
    # next-man-up choice model (conditional logit)
    r0 = run_projection(panel, prm, shares=shares)
    prm.nmu_coef = _fit_nmu(r0, shares)
    log(f"  fit {season}: nmu_coef {np.round(prm.nmu_coef, 3).tolist()}")
    # f_same: grid on the training window (minutes MAE of played rotation players)
    best = None
    for f in f_grid:
        prm.f_same = f
        r = run_projection(panel, prm, shares=shares)
        m = r["played"] & (r["mean0"] >= 15)
        mae = float(np.abs(r.loc[m, "mean0"] - r.loc[m, "act_min_r"]).mean())
        log(f"  fit {season}: f_same={f} train MAE {mae:.3f}")
        if best is None or mae < best[0]:
            best = (mae, f, r)
    prm.f_same = best[1]
    for k in (0.3, 0.8):
        prm.k_add = k
        r = run_projection(panel, prm, shares=shares)
        m = r["played"] & (r["mean0"] >= 15)
        mae = float(np.abs(r.loc[m, "mean0"] - r.loc[m, "act_min_r"]).mean())
        log(f"  fit {season}: k_add={k} train MAE {mae:.3f}")
        if mae < best[0]:
            best = (mae, best[1], r, k)
    prm.k_add = best[3] if len(best) > 3 else 0.6
    r = best[2]
    cnt_min, ok = _fit_margin_noise(prm, r)
    prm.diagnostics = {"train_rows": int(len(pl)), "train_team_games": int(len(tr)),
                       "train_mae_ge15": round(best[0], 4), "noise_fit_ok": ok,
                       "g_cells_min_n": cnt_min}
    if version == "v1":
        prm = fit_v1(h, prm, panel, shares, log=log, **(v1_kw or {}))
    return prm


def fit_v1(h: History, prm05: MinutesParams, panel: pd.DataFrame,
           shares: "PairShares | None", log: Callable[[str], None] = print,
           use_share: bool = False, max_rounds: int = 3000, gbm_params: dict | None = None,
           nmu_grid=(0.0, 0.5, 1.0), k_grid=(0.0, 0.15, 0.3), share_grid=(0.0, 0.15, 0.3),
           cv_grid=(0.2, 0.4, 0.6), features=None, minfix: bool | dict = False,
           blowout: bool = False) -> MinutesParams:
    """v1 on top of a v0.5 fit, all on the same training panel (seasons < S, oracle scenario).

    1. LightGBM (L2) for E[regulation minutes | plays] on V1_FEATURES (+ V1_SHARE if
       use_share; the panel then needs `share_map`, i.e. History.shares set when it was built)
       over played, non-absent, v1_ok rows. Leave-one-season-out folds give out-of-fold (OOF)
       predictions and the number of rounds (early stopping on the held-out season, mean x 1.1);
       the final model is refit on all training seasons.
    2. On the OOF base (the reserve stays the v0.5 model on the v0.5 base), choose
       (coordinate-wise) the reconciliation
       cv, nmu_scale, k_add and share_coef (stage 2 on top of the learned promotion) by the
       training MAE on a FIXED population: played rows whose OOF GBM base is >= 10. (The
       v0.5 criterion, rows projected >= 15, depends on the candidate itself: pushing bench
       players below 15 shrinks it, and it picked a leftover that over-projects starters.)
    3. Refit the margin curve g and the noise model on the OOF projections.
    minfix (COURTSIDE_MINFIX): P(plays | available) is refit with the plan (MF_PLAN_FEATURES on
    the out-of-fold GBM base, `prm.minfix["pplay_coef"]`) before step 2, and step 3 fits the
    noise net of the margin curve.
    blowout (COURTSIDE_BLOWOUT): after step 3 (and the minfix noise calibration), the blowout
    stage (`_bo_fit`: P(plays) and reserve offsets by margin, the margin curve calibrated by
    simulation) on the same out-of-fold projections -> `prm.blowout`; nothing before it
    changes."""
    import lightgbm as lgb
    import time as _time
    t0 = _time.time()
    prm = MinutesParams.from_json(prm05.to_json())
    prm.version = "v1"
    if not has_v1_columns(panel):
        raise ValueError("fit_v1: the training panel has no v1 columns (rebuild the History)")
    names = list(features) if features is not None else \
        list(V1_FEATURES) + (list(V1_SHARE) if use_share else [])
    if use_share and "share_map" not in panel:
        raise ValueError("fit_v1(use_share=True) needs panel['share_map'] (History.shares)")
    gp = dict(GBM_PARAMS, **(gbm_params or {}))
    ab = panel["absent"].to_numpy(bool)
    X = v1_matrix(panel, ab, names)
    y = panel["act_min_r"].to_numpy(float)
    ok = v1_ok(panel)
    rows = panel["played"].to_numpy(bool) & ~ab & ok
    seas = panel["season"].to_numpy()
    oof = np.full(len(panel), np.nan)
    iters, folds = [], {}
    for s in sorted(set(seas)):
        tr, va = rows & (seas != s), rows & (seas == s)
        if va.sum() < 100 or tr.sum() < 100:
            continue
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=False)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(gp, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        it = m.best_iteration or max_rounds
        iters.append(it)
        pr = ~ab & ok & (seas == s)
        oof[pr] = m.predict(X[pr], num_iteration=it, num_threads=gp["num_threads"])
        e = np.abs(oof[va] - y[va])
        folds[s] = {"rounds": int(it), "mae_played": round(float(e.mean()), 4)}
        log(f"  fit_v1 fold {s}: rounds {it}, OOF MAE (played) {e.mean():.3f}")
    n_rounds = int(round(np.mean(iters) * 1.1)) if iters else 300
    final = lgb.train(gp, lgb.Dataset(X[rows], y[rows], feature_name=names),
                      num_boost_round=n_rounds)
    prm.gbm_model = final.model_to_string()
    prm.gbm_features = names
    imp = final.feature_importance("gain")
    imp = imp / max(imp.sum(), 1e-9)
    top = sorted(zip(names, imp.tolist()), key=lambda t: -t[1])[:20]
    t_gbm = _time.time() - t0
    # P(plays | available) for the live candidate set: departed players removed (they are
    # dropped live, so they must not teach "new season -> DNP"), forced extras as roster
    dep = departed(h, panel)
    b0 = np.clip(design(panel, prm.base_features) @ np.asarray(prm.base_coef), 0.5, 44.0)
    q = panel.copy()
    q["avail_rank"], q["n_avail"] = avail_rank(q, b0, ab | dep)
    keep = ~ab & ~dep
    Z1 = design(forced_as_roster(q[keep]), prm.pplay_features)
    prm.pplay_coef_v1 = _logit(Z1, q.loc[keep, "played"].to_numpy(float)).tolist()
    if minfix:
        opts = dict(MF_DEFAULT, **(minfix if isinstance(minfix, dict) else {}))
        # the plan each training row had out of fold (the v0.5 base where the GBM falls back)
        bmf = np.where(np.isfinite(oof), np.clip(oof, 0.5, 44.0), b0)
        mf = _mf_fit_pplay(prm, q, bmf, ab | dep, keep, q.loc[keep, "played"].to_numpy(float),
                           opts["pplay"])
        prm.minfix = {"version": MF_VERSION, "pplay": opts["pplay"], "noise": opts["noise"],
                      "noise_net_of_margin": True, **mf}
        log(f"  fit_v1 minfix: P(plays) {opts['pplay']}, plan coef "
            f"{np.round(mf['pplay_coef'][len(prm.pplay_features):], 3).tolist()}")
    # stage 2 on the OOF base
    tp = panel.copy()
    tp["_v1_oof"] = oof
    base, pp = base_and_pplay(tp, prm, ab)
    base = np.asarray(base)
    # the reserve stays the v0.5 model on the v0.5 base (V1Base.s_rows): refit on the GBM
    # base it had slope ~1 below sum p x base = 220, a range training rarely reaches (1%) but
    # opening-night live panels do (P(plays) skew) -> 25-35 team minutes lost (T2.2 follow-up)

    fixed = panel["played"].to_numpy(bool) & ~ab & (base >= 10.0)

    def mae_of(q):
        r = run_projection(tp, q, shares=shares)
        y_ = r["act_min_r"].to_numpy(float)[fixed]
        return float(np.abs(r["mean0"].to_numpy(float)[fixed] - y_).mean()), r

    grid = {}
    best = None
    for name, vals in (("recon_cv", cv_grid), ("nmu_scale", nmu_grid), ("k_add", k_grid),
                       ("share_coef", share_grid)):
        cur = None
        for v in vals:
            setattr(prm, name, v)
            mae, r = mae_of(prm)
            grid[f"{name}={v}"] = round(mae, 4)
            log(f"  fit_v1 {name}={v}: train (OOF) MAE {mae:.4f}")
            if cur is None or mae < cur[0] - 1e-6:
                cur = (mae, v, r)
        setattr(prm, name, cur[1])
        best = cur
    cnt_min, nok = _fit_margin_noise(prm, best[2])
    if prm.minfix.get("noise") == "calib":
        prm.minfix["calibration"] = _mf_calibrate_noise(prm, best[2], log)["calib"]
    if blowout:
        prm.blowout = _bo_fit(prm, best[2], log)
    prm.diagnostics = dict(prm.diagnostics, v1={
        "n_rounds": n_rounds, "folds": folds, "train_rows": int(rows.sum()),
        "fallback_share_train": round(float((~ok & ~ab).sum() / max((~ab).sum(), 1)), 4),
        "pplay_departed_rows_removed": int((dep & ~ab).sum()),
        "oof_train_mae_ge15": round(best[0], 4), "grid": grid,
        "importance_gain_top20": [[n, round(v, 4)] for n, v in top],
        "noise_fit_ok": nok, "g_cells_min_n": cnt_min, "seconds_gbm": round(t_gbm, 1),
        "seconds_total": round(_time.time() - t0, 1), "use_share": bool(use_share)})
    r_b = best[2]
    m15 = r_b["played"].to_numpy(bool) & (r_b["mean0"].to_numpy(float) >= 15)
    prm.diagnostics["v1"]["oof_train_mae_ge15"] = round(float(np.abs(
        r_b["mean0"].to_numpy(float)[m15] - r_b["act_min_r"].to_numpy(float)[m15]).mean()), 4)
    prm.diagnostics["v1"]["oof_train_mae_fixed_pop"] = round(best[0], 4)
    log(f"  fit_v1: {n_rounds} rounds, cv {prm.recon_cv}, nmu_scale {prm.nmu_scale}, "
        f"k_add {prm.k_add}, "
        f"share_coef {prm.share_coef}, OOF MAE {best[0]:.4f} ({_time.time() - t0:.0f}s)")
    return prm


def _fit_reserve(panel: pd.DataFrame, base: np.ndarray, pp_avail: np.ndarray,
                 minus: pd.Series | None = None) -> list:
    """OLS of the minutes outside the candidate set on the available weight (sum p x base);
    `minus` (per key): minutes taken out of the target first (the debut reserve's)."""
    av = ~panel["absent"].to_numpy(bool)
    w = np.zeros(len(panel))
    w[np.flatnonzero(av)] = pp_avail * base[av]
    k = pd.DataFrame({"key": panel["key"], "w": w, "a": panel["act_min_r"].fillna(0.0),
                      "av": av}).groupby("key").agg(s=("w", "sum"), act=("a", "sum"),
                                                     n=("av", "sum"))
    y = TEAM_MIN - k["act"].to_numpy()
    if minus is not None:
        y = y - minus.reindex(k.index).fillna(0.0).to_numpy(float)
    return _ols(reserve_design(k["s"].to_numpy(), k["n"].to_numpy()),
                np.clip(y, 0, None)).tolist()


def _fit_margin_noise(prm: MinutesParams, r: pd.DataFrame) -> tuple[int, bool]:
    """g table (margin curve) and the noise model from a projected training panel (mean0)."""
    # margin curve g: mean residual by tier x result x |margin| bin (played, regulation games)
    rr = r[r["played"] & (r["n_ot"] == 0) & (r["mean0"] > 0)]
    res = rr["act_min_r"].to_numpy(float) - rr["mean0"].to_numpy(float)
    tiers = tier_of(rr["mean0"].to_numpy(float))
    mg = rr["margin"].to_numpy(float)
    b = np.clip(np.searchsorted(MARGIN_BINS, np.abs(mg), side="right") - 1, 0,
                len(MARGIN_BINS) - 2)
    won = (mg > 0).astype(int)
    nt, nb = len(TIER_EDGES) + 1, len(MARGIN_BINS) - 1
    tab = np.zeros((nt, 2, nb))
    cnt = np.zeros((nt, 2, nb))
    np.add.at(tab, (tiers, won, b), res)
    np.add.at(cnt, (tiers, won, b), 1)
    tier_mean = np.array([res[tiers == t].mean() if (tiers == t).any() else 0.0
                          for t in range(nt)])
    # shrink thin cells to the tier mean (20 pseudo-obs)
    tab = (tab + 20.0 * tier_mean[:, None, None]) / (cnt + 20.0)
    prm.g_table = tab.tolist()
    # centre: E_train[g] per tier under the training margin mix -> 0
    gv = tab[tiers, won, b]
    prm.g_center = [float(gv[tiers == t].mean()) if (tiers == t).any() else 0.0
                    for t in range(nt)]
    # noise on log ratio, played, mean >= 5 (minfix: net of the margin curve, as the sims draw
    # minutes = (plan + g(margin)) x noise)
    rn = r[r["played"] & (r["mean0"] >= 5)]
    lvl = rn["mean0"].to_numpy(float)
    if prm.minfix.get("noise_net_of_margin"):
        lvl = np.maximum(lvl + _g_rows(prm, tier_of(lvl), rn["margin"].to_numpy(float)), 1.0)
    lr = np.log(np.maximum(rn["act_min_r"].to_numpy(float), 0.5) / lvl)
    sub = np.random.default_rng(0).permutation(len(lr))[:80000]
    mu, sg, q0, q1, ok = _fit_noise(lr[sub], tier_of(rn["mean0"].to_numpy(float))[sub],
                                    rn["pf36"].to_numpy(float)[sub])
    prm.mu, prm.sigma, prm.q0, prm.q1 = mu, sg, q0, q1
    return int(cnt.min()), ok


def _g_rows(prm: MinutesParams, tiers: np.ndarray, margins: np.ndarray, tab=None, cen=None
            ) -> np.ndarray:
    """g_tier(|m|, won) - c_tier per row (each row its own tier and margin), as g_value (tab /
    cen: another table and centre, default the params' own)."""
    tab = np.asarray(prm.g_table if tab is None else tab, float)
    cen = np.asarray(prm.g_center if cen is None else cen, float)
    edges = np.asarray(MARGIN_BINS, float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[-1] = edges[-2] + 4.0
    m = np.nan_to_num(np.asarray(margins, float))
    am, won = np.abs(m), (m > 0).astype(int)
    out = np.zeros(len(m))
    for t in np.unique(tiers):
        for r in (0, 1):
            sel = (tiers == t) & (won == r)
            if sel.any():
                out[sel] = np.interp(am[sel], mids, tab[t, r]) - cen[t]
    return out


# --- live / single team-game API -------------------------------------------------------------

def build_shares(h: History, seasons: Iterable[str]) -> "PairShares":
    """Pair shares from the regular seasons `seasons` (lookups stay strictly before a date)."""
    return PairShares(pair_events(oracle_panel(h, eval_targets(h, seasons))))


def project_team(h: History, params: MinutesParams, team: str, day, absent=(),
                 starters=None, extra=(), spread: float | None = None,
                 season: str | None = None, n_ot: float | None = None,
                 shares: "PairShares | None" = None) -> pd.DataFrame:
    """Minutes for `team` on `day` under a scenario (absent ids; optional confirmed starters).

    Returns one row per candidate: player_id, g1, g2, base, p_play (P plays | available; 0 if
    absent), mean (E[min | plays], before the margin integration), exp_min (margin-integrated
    E[min | plays] with spread ~ N(spread, 12.4^2)), lift, promoted, exp_start, pf36, dkpm."""
    d = int(np.datetime64(pd.Timestamp(day).date(), "D").astype(np.int64))
    if season is None:
        ts = pd.Timestamp(day)
        y = ts.year if ts.month >= 8 else ts.year - 1
        season = f"{y}-{(y + 1) % 100:02d}"
    tgt = pd.DataFrame({"key": ["x"], "team": [team], "date": [d], "season": [season]})
    ex = pd.DataFrame({"key": "x", "player_id": [int(x) for x in extra]}) if extra else None
    p = pregame_panel(h, tgt, extra=ex)
    if ex is not None and len(ex):
        p = p[(p["roster"] == 1) | p["player_id"].isin(ex["player_id"])]
    p = p.reset_index(drop=True)
    absent = set(int(a) for a in absent)
    p["absent"] = p["player_id"].isin(absent)
    base, pp = base_and_pplay(p, params, p["absent"].to_numpy(bool))
    ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
    stt = None
    if starters is not None:
        stt = p["player_id"].isin(set(int(s) for s in starters)).to_numpy()
    ab = p["absent"].to_numpy()
    share = shares.matrix(d, p["player_id"].to_numpy(), ab) if shares is not None else None
    mean, det = project_arrays(base, pp, ab, p["g1"].to_numpy(int),
                               p["g2"].to_numpy(int), p["st10"].to_numpy(float), ew,
                               _team_target(p, params), params, starters=stt, share=share,
                               detail=True)
    out = pd.DataFrame({"player_id": p["player_id"], "g1": p["g1"], "g2": p["g2"],
                        "base": base, "p_play": np.where(p["absent"], 0.0, pp), "mean": mean,
                        "lift": det["lift"], "pstart": det["pstart"],
                        "promoted": det["promoted"],
                        "exp_start": det["exp_start"], "pf36": p["pf36"], "dkpm": p["dkpm"],
                        "ew_f": p["ew_f"], "ew_s": p["ew_s"], "st10": p["st10"],
                        "reserve": det["reserve"]})
    out["exp_min"] = expected_minutes(out, params, spread=spread, n_ot=n_ot)
    return out.sort_values("exp_min", ascending=False).reset_index(drop=True)
