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

Everything learned lives in `MinutesParams` (JSON-serialisable), produced by `fit(history, S)`
from seasons < S only. Structural constants (EWMA alphas, RECON_CV, SHARE_K, tiers, margin
bins, the 50% shrink) are fixed; RECON_CV and the variance form were chosen on the dev season
2013-14 (fit < 2013-14), before every reported test season.

Public API (consumed by T1.4 copula/pipeline)
  load_history(conn, since="2000-01") -> History            (~8 s, ~0.8 GB peak)
  fit(history, season, window=5, log=print) -> MinutesParams  (~40-60 s)
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

    def to_json(self) -> str:
        return json.dumps(asdict(self))

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
    h = pg.merge(tg[["game_id", "team", "date", "tgi", "n_ot"]], on=["game_id", "team"],
                 how="inner")
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
    keep = ["player_id", "game_id", "team", "season", "season_type", "date", "tgi", "min",
            "min_r", "started", "ew_f", "ew_s", "med5", "st10", "pf36", "dk", "dkpm", "n_gp",
            "base_ew", "season_gp", "missed_before", "gsr_post", "n_ot"]
    out = h[keep].copy()
    for c in ("min", "min_r", "ew_f", "ew_s", "med5", "st10", "pf36", "dk", "dkpm", "base_ew"):
        out[c] = out[c].astype(np.float32)
    return out


def _ewm_from(x: np.ndarray, first: np.ndarray, alpha: float, init: float) -> np.ndarray:
    """EWMA with y_-1 = init per segment (adjust=False), segments start where first is True."""
    out = np.empty(len(x))
    y = init
    for i in range(len(x)):
        if first[i]:
            y = init
        y = y + alpha * (x[i] - y)
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


def build_history(pg: pd.DataFrame, tg_raw: pd.DataFrame, positions: pd.DataFrame) -> History:
    tg = _team_games(tg_raw)
    hist = player_history(pg, tg)
    targets = _starter_targets(hist, tg, positions)
    sched = {t: {"date": d["date"].to_numpy(), "season": d["season"].to_numpy(),
                 "gid": d["game_id"].to_numpy()}
             for t, d in tg.groupby("team", sort=False)}
    return History(hist=hist, sched=sched, targets=targets, positions=positions, tg=tg)


def load_history(conn: sqlite3.Connection, since: str = FIRST_SEASON) -> History:
    pg, tg = load_box(conn, since)
    return build_history(pg, tg, load_positions(conn))


def truncate(h: History, day) -> History:
    """History restricted to games strictly before `day` (for no-lookahead checks)."""
    d = int(np.datetime64(pd.Timestamp(day).date(), "D").astype(np.int64))
    hist = h.hist[h.hist["date"] < d]
    tg = h.tg[h.tg["date"] < d]
    sched = {t: {k: v[v_ok] for k, v in s.items()}
             for t, s in h.sched.items() for v_ok in [s["date"] < d]}
    return History(hist=hist, sched=sched, targets=h.targets[h.targets["date"] < d],
                   positions=h.positions, tg=tg)


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
    st = hist[["player_id", "date", "team", "season", "min_r", "started", "ew_f", "ew_s",
               "med5", "st10", "pf36", "dkpm", "n_gp", "base_ew", "season_gp", "gsr_post"]]
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
    return p.sort_values(["key", "player_id"]).reset_index(drop=True)


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
    through the depth rank among available players."""
    X = design(p, params.base_features)
    base = np.clip(X @ np.asarray(params.base_coef), 0.5, 44.0)
    if absent is None:
        absent = p["absent"].to_numpy(bool) if "absent" in p else np.zeros(len(p), bool)
    p = p.copy()
    p["avail_rank"], p["n_avail"] = avail_rank(p, base, np.asarray(absent, bool))
    Z = design(p, params.pplay_features)
    pp = 1.0 / (1.0 + np.exp(-(Z @ np.asarray(params.pplay_coef))))
    return base, np.clip(pp, 0.01, 0.995)


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
                   starters=None, share=None, n_ot: float = 0.0, detail: bool = False):
    """Scenario projection for one team-game.

    base, p_avail (P plays | available), g1/g2 (groups), st10, ew (minutes EWMA blend): arrays
    over the candidates; absent: bool mask; tgt: (n, 3) team starter targets (identical rows);
    starters: optional bool mask of confirmed starters; share: optional (n, n) matrix,
    share[i, j] = i's historical share of j's minutes when j sat.
    Returns mean minutes conditional on playing (0 for absent) [and a detail dict].

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
    # confirmed unexpected starters not used as anyone's replacement: lift to starter level
    if starters is not None:
        for i in np.flatnonzero(unexpected & ~used):
            new = 0.5 * t[g1[i]] + 0.5 * ew[i]
            if new > adj[i] + lift[i]:
                lift[i] += new - (adj[i] + lift[i])
            pstart[i] = 1.0
    adj = adj + lift
    s_av = float((p * base).sum())
    reserve = float(reserve_minutes(params, s_av, float(avail.sum()))[0])
    total = TEAM_MIN + OT_TEAM_MIN * n_ot - reserve
    cap = CAP + 5.0 * n_ot
    if params.recon == "proportional":
        exp_min = allocate_minutes(np.where(avail, adj, 0.0), p, total=total, cap=cap)
    else:
        exp_min = reconcile(np.where(avail, adj, 0.0), p, total, cap)
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

def g_value(params: MinutesParams, tiers: np.ndarray, margin) -> np.ndarray:
    """g_tier(|m|, won) - c_tier, linear in |m| between bin centres. margin: scalar or (S,)
    array (team perspective); returns (S, n) (or (n,) for a scalar)."""
    tab = np.asarray(params.g_table, float)        # (T, 2, B)
    cen = np.asarray(params.g_center, float)
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
    x = np.maximum(mean[None, :] + g_value(params, tiers, margins).reshape(S, n), 0.0)
    if noise:
        mu = np.asarray(params.mu)[tiers]
        sg = np.asarray(params.sigma)[tiers]
        x = x * np.exp(mu[None, :] + sg[None, :] * rng.standard_normal((S, n)))
        q = np.clip(params.q0 + params.q1 * pf, 0.0, 0.5)
        ex = rng.random((S, n)) < q[None, :]
        x = np.where(ex, x * rng.uniform(*EXIT_U, size=(S, n)), x)
    plays = rng.random((S, n)) < p[None, :]
    x = np.where(plays, x, 0.0)
    ot = np.zeros(S) if n_ot is None else np.asarray(n_ot, float)
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
        res = project_arrays(base[idx], pp[idx], ab[idx], g1[idx], g2[idx], st10[idx], ew[idx],
                             tgt[idx], params, starters=None if stt is None else stt[idx],
                             share=share, detail=True)
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
        shares: "PairShares | None" = None) -> MinutesParams:
    """Fit every parameter on the `window` seasons before `season` (strictly < season).

    panel / shares: optional precomputed oracle panel of those seasons and pair shares
    (events from seasons < season only)."""
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
    w = np.zeros(len(panel))
    w[np.flatnonzero(~panel["absent"].to_numpy())] = pp_tr * base_tr[~panel["absent"].to_numpy()]
    k = pd.DataFrame({"key": panel["key"], "w": w, "a": panel["act_min_r"].fillna(0.0),
                      "av": ~panel["absent"]}).groupby("key").agg(
        s=("w", "sum"), act=("a", "sum"), n=("av", "sum"))
    rcoef = _ols(reserve_design(k["s"].to_numpy(), k["n"].to_numpy()),
                 np.clip(TEAM_MIN - k["act"].to_numpy(), 0, None))
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
    prm.reserve_coef = rcoef.tolist()
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
    # noise on log ratio, played, mean >= 5
    rn = r[r["played"] & (r["mean0"] >= 5)]
    lr = np.log(np.maximum(rn["act_min_r"].to_numpy(float), 0.5) / rn["mean0"].to_numpy(float))
    sub = np.random.default_rng(0).permutation(len(lr))[:80000]
    mu, sg, q0, q1, ok = _fit_noise(lr[sub], tier_of(rn["mean0"].to_numpy(float))[sub],
                                    rn["pf36"].to_numpy(float)[sub])
    prm.mu, prm.sigma, prm.q0, prm.q1 = mu, sg, q0, q1
    prm.diagnostics = {"train_rows": int(len(pl)), "train_team_games": int(len(tr)),
                       "train_mae_ge15": round(best[0], 4), "noise_fit_ok": ok,
                       "g_cells_min_n": int(cnt.min())}
    return prm


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
