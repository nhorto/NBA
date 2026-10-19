"""Empirical-Bayes per-minute rate state (docs/PLAYERS.md §4.3, T1.1).

For every player and stat in `STATS` (plus the derived DK-per-minute `dk`, bonuses excluded) the
state as of day d is a shrunk per-minute rate that uses only games dated before d::

    r = (Sx + kappa * mu) / (Sn + kappa)         Sx = sum_g w_g x_g,   Sn = sum_g w_g min_g
    w_g = 2 ** (-(d - day_g) / H)  *  rho ** (season_d - season_g)

* **Recency weights** are exponential in calendar days (half-life `H`, so the off-season decays
  by itself) times an extra season-break discount `rho` per season boundary crossed. `H`, `rho`
  and `kappa` are chosen per stat by `fit_params` on seasons < S (maximum NB log-likelihood of
  the next game's count given its actual minutes). Regular season, play-in and playoffs all feed
  the state (the DK FPPG convention).
* **Prior** `mu` is the season-to-date league rate of the player's cell = position group
  (G/F/C/unknown, `team_rosters` then the `player_index` snapshot) x tier of his own decayed
  minutes per game (5 tiers, edges 12/20/26/32), measured on games before d and blended with
  the previous season's final cell rate (`m0` pseudo-minutes), so early-season rows have a
  prior. Cells use each game's *pre-game* tier, so the prior is causal too.
* **Dispersion**: a negative-binomial `phi` per (stat, tier) (Var = mu + phi mu^2) fit on seasons
  < S from the walk-forward predictions, for the copula / marginal distributions.
* **Usage bump** (`fit_usage_bump`): role-tier per-minute multipliers when a regular star sat,
  fit on natural experiments in seasons < S (the Phase 1 part of §4.3).

Everything is one vectorised pass: per-player exclusive cumulative sums of `x * exp(clock)`, where
clock = lambda * days + beta * season_index, so weights are exp(-(clock_t - clock_g)). Query rows
(a player, a day) are inserted into the same sorted frame, which makes `build_state` (nightly),
`rate_frame` (any list of player-days, e.g. DNP candidates) and `walk_forward` (every player-game
of a season) one code path; with queries sorted before the same-day real row, a state can never
see its own day.

Public API (consumed by T1.4 copula / reconcile)
  DEFAULT_PARAMS                         fitted on seasons 2018-19..2025-26 (JSON-serialisable dict)
  fit_params(conn, seasons, ...) -> params
  build_state(conn, as_of, params=None) -> long DataFrame (as_of, player_id, stat, rate, n_eff,
                                           prior, kappa); stat 'mpg' carries the minutes-per-game
                                           state (rate) and decayed games (n_eff)
  write_state(conn, as_of, params=None, keep=30) -> rows written
  read_state(conn, as_of=None) -> wide DataFrame (see `to_wide`)
  to_wide(long_df, params) -> index player_id; r_<stat>, p_<stat>, n_<stat>, mpg, tier
  rate_frame(conn, queries[player_id, game_date], params, games=None) -> wide, one row per query
  walk_forward(conn, season, params, games=None) -> wide, one row per player-game of the season
                                           (pre-game state) with the actual stats next to it
  phi_for(params, stat, tier) -> NB overdispersion;  nb_logpmf(y, mu, phi)
  dk_rate(rates_by_stat) -> DK points per minute (no DD/TD bonus)
  bump_for(params, role, stat) -> multiplier for a role in {'star', 'rotation'} when a star sat
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln

STATS = ("pts", "fg3m", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fta", "pf")
DK_WEIGHTS = {"pts": 1.0, "fg3m": 0.5, "reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5}
TIER_EDGES = (12.0, 20.0, 26.0, 32.0)
N_TIER = len(TIER_EDGES) + 1
N_GRP = 4                         # G, F, C, unknown
HISTORY_SEASONS = 6               # seasons of games before the first evaluated season
LN2 = math.log(2.0)
EPOCH = np.datetime64("1970-01-01", "D")

# league per-minute rates, only used when a cell has neither current nor previous-season data
DEFAULT_RATE = {"pts": 0.45, "fg3m": 0.06, "reb": 0.19, "oreb": 0.055, "dreb": 0.135, "ast": 0.11,
                "stl": 0.033, "blk": 0.02, "tov": 0.06, "fga": 0.38, "fta": 0.11, "pf": 0.09}

_MPG = {"H": 90.0, "rho": 0.6, "k0": 2.0, "mpg0": 14.0}   # fixed state behind the tier
# per-minute production depends on how long the player actually plays relative to his usual
# minutes (foul trouble / injury cut games short and cost efficiency; close games lift it):
# E[stat | minutes] = rate * minutes * f_stat(minutes / mpg), f a piecewise-linear table.
RATIO_EDGES = (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)


# ============================================================================== small helpers

def season_of(day: date) -> str:
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def season_label(sidx: int) -> str:
    return f"{sidx}-{(sidx + 1) % 100:02d}"


def sidx_of(label: str) -> int:
    return int(label[:4])


def tier_of(mpg, edges=TIER_EDGES) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(mpg, float), side="right")


def _to_day(s) -> np.ndarray:
    return (pd.to_datetime(pd.Series(s)).to_numpy().astype("datetime64[D]") - EPOCH).astype(np.int64)


def _day_of(d: date | str) -> int:
    return int((np.datetime64(pd.Timestamp(d).date(), "D") - EPOCH).astype(np.int64))


def nb_logpmf(y, mu, phi) -> np.ndarray:
    """log NB2 pmf with mean mu and Var = mu + phi mu^2 (phi -> 0 is Poisson)."""
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    phi = np.maximum(np.asarray(phi, float), 1e-6)
    r = 1.0 / phi
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + y * np.log(mu * phi / (1.0 + mu * phi)) - r * np.log1p(mu * phi))


def dk_rate(rates: dict) -> np.ndarray:
    """DK points per minute from per-minute rates (dict of stat -> array); no DD/TD bonus."""
    return sum(DK_WEIGHTS[s] * np.asarray(rates[s], float) for s in DK_WEIGHTS)


# ============================================================================== params

def _stat_default(stat: str) -> dict:
    return {"H": 120.0, "rho": 0.6, "kappa": 300.0, "phi": [0.1] * N_TIER}


def default_params() -> dict:
    """Uninformative params (cold start / tests): 120-day half-life, last season x0.6, kappa 300."""
    return {"version": 1, "fit_seasons": [], "history_seasons": HISTORY_SEASONS, "m0": 1500.0,
            "relative": False, "tier_edges": list(TIER_EDGES), "mpg": dict(_MPG),
            "stats": {s: _stat_default(s) for s in STATS}, "minutes_adj": {}, "usage_bump": {},
            "diagnostics": {}}


def params_to_json(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def params_from_json(s: str) -> dict:
    return json.loads(s)


def phi_for(params: dict, stat: str, tier) -> np.ndarray:
    phi = np.asarray(params["stats"][stat]["phi"], float)
    return phi[np.asarray(tier, int)]


def minutes_multiplier(params: dict, stat: str, minutes, mpg) -> np.ndarray:
    """f_stat(minutes / mpg): multiplier on rate x minutes for a game of `minutes` by a player
    whose minutes-per-game state is `mpg` (1.0 when the table is not fitted)."""
    adj = params.get("minutes_adj")
    if not adj or stat not in adj["mult"]:
        return np.ones(np.broadcast(np.asarray(minutes), np.asarray(mpg)).shape)
    ratio = np.asarray(minutes, float) / np.maximum(np.asarray(mpg, float), 1.0)
    return np.interp(ratio, adj["centers"], adj["mult"][stat])


def bump_for(params: dict, role: str, stat: str) -> float:
    """Per-minute multiplier for `role` ('star' = the other star, 'rotation') when a regular
    star sat (1.0 if not fitted)."""
    return float(params.get("usage_bump", {}).get(role, {}).get(stat, 1.0))


# ============================================================================== loading

def load_games(conn: sqlite3.Connection, since_season: str | None = None,
               until_season: str | None = None, before: date | None = None) -> pd.DataFrame:
    """player_games rows with min > 0 (all season types) as a DataFrame with `day`, `sidx`."""
    sql = ("SELECT game_id, player_id, team, season, season_type, game_date, min, "
           + ", ".join(STATS) + " FROM player_games WHERE min > 0")
    args: list = []
    if since_season:
        sql += " AND season >= ?"
        args.append(since_season)
    if until_season:
        sql += " AND season <= ?"
        args.append(until_season)
    if before is not None:
        sql += " AND game_date < ?"
        args.append(before.isoformat())
    df = pd.read_sql_query(sql, conn, params=args)
    df[list(STATS)] = df[list(STATS)].fillna(0).astype(float)
    df["min"] = df["min"].astype(float)
    df["day"] = _to_day(df["game_date"]) if len(df) else np.zeros(0, np.int64)
    df["sidx"] = df["season"].str[:4].astype(int) if len(df) else np.zeros(0, int)
    return df


def _grp_code(pos) -> int:
    if not isinstance(pos, str) or not pos:
        return 3
    return {"G": 0, "F": 1, "C": 2}.get(pos.strip()[:1].upper(), 3)


def load_positions(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_id, sidx, grp) from team_rosters, (player_id, grp_snap) from player_index.
    Position group = the first listed position (G, F, C); 3 = unknown."""
    tr = pd.read_sql_query("SELECT player_id, season, position FROM team_rosters", conn)
    pi = pd.read_sql_query("SELECT player_id, position FROM player_index", conn)
    tr["sidx"] = tr["season"].str[:4].astype(int)
    tr["grp"] = [_grp_code(p) for p in tr["position"]]
    tr = tr[tr["grp"] < 3].drop_duplicates(["player_id", "sidx"])[["player_id", "sidx", "grp"]]
    pi["grp_snap"] = [_grp_code(p) for p in pi["position"]]
    pi = pi[pi["grp_snap"] < 3].drop_duplicates("player_id")[["player_id", "grp_snap"]]
    return tr, pi


def _attach_grp(pid: np.ndarray, sidx: np.ndarray, pos) -> np.ndarray:
    tr, pi = pos
    key = pd.DataFrame({"player_id": pid, "sidx": sidx})
    m = key.merge(tr, on=["player_id", "sidx"], how="left").merge(pi, on="player_id", how="left")
    return m["grp"].fillna(m["grp_snap"]).fillna(3).to_numpy().astype(int)


# ============================================================================== the frame

class _Walk:
    """Exclusive, per-player exponentially weighted sums over a (pid, day)-sorted frame."""

    def __init__(self, pid: np.ndarray, day: np.ndarray, sidx: np.ndarray):
        n = len(pid)
        self.n = n
        self.start = np.ones(n, bool)
        if n > 1:
            self.start[1:] = pid[1:] != pid[:-1]
        self.gid = np.cumsum(self.start) - 1
        first = np.flatnonzero(self.start)
        self.day = day.astype(float)
        self.sidx = sidx.astype(float)
        self.day0 = self.day[first][self.gid]
        self.sidx0 = self.sidx[first][self.gid]

    def clock(self, half_life: float, rho: float) -> np.ndarray:
        return ((self.day - self.day0) * (LN2 / half_life)
                + (self.sidx - self.sidx0) * (-math.log(rho)))

    def sums(self, vals: np.ndarray, clock: np.ndarray) -> np.ndarray:
        """E[i] = sum over earlier rows j of the same player of vals[j] * exp(-(clock_i - clock_j))."""
        if self.n == 0:
            return np.zeros_like(vals)
        t = vals * np.exp(clock)[:, None]
        c = pd.DataFrame(t).groupby(self.gid, sort=False).cumsum().to_numpy()
        e = np.empty_like(c)
        e[0] = 0.0
        e[1:] = c[:-1]
        e[self.start] = 0.0
        return e * np.exp(-clock)[:, None]


@dataclass
class _Frame:
    pid: np.ndarray
    day: np.ndarray
    sidx: np.ndarray
    isq: np.ndarray            # query row (no stats of its own)
    rep: np.ndarray            # rows reported back
    orig: np.ndarray           # position of the reported row in the caller's input
    grp: np.ndarray
    vals: np.ndarray           # (n, 1 + len(STATS)): min then STATS (zeros on query rows)
    walk: _Walk
    mpg: np.ndarray
    geff: np.ndarray
    tier: np.ndarray
    cell: np.ndarray
    prior: np.ndarray          # (n, len(STATS)): the cell prior in force at every row
    rep_idx: np.ndarray
    df: pd.DataFrame           # real + query rows in frame order (ids, season_type, ...)


def _prepare(games: pd.DataFrame, pos, params: dict, queries: pd.DataFrame | None = None
             ) -> _Frame:
    """Sort, attach groups, tiers (from the fixed mpg state) and causal cell priors."""
    g = games
    parts = [pd.DataFrame({
        "pid": g["player_id"].to_numpy(), "day": g["day"].to_numpy(),
        "sidx": g["sidx"].to_numpy(), "isq": False, "rep": queries is None,
        "orig": np.arange(len(g)), "min": g["min"].to_numpy(),
        **{s: g[s].to_numpy() for s in STATS}})]
    if queries is not None and len(queries):
        qd = pd.to_datetime(queries["game_date"]).dt.date
        qsidx = np.array([sidx_of(season_of(d)) for d in qd], int)
        q = pd.DataFrame({
            "pid": queries["player_id"].to_numpy(), "day": _to_day(queries["game_date"]),
            "sidx": qsidx, "isq": True, "rep": True, "orig": np.arange(len(queries)),
            "min": 0.0, **{s: 0.0 for s in STATS}})
        parts.append(q)
    df = pd.concat(parts, ignore_index=True)
    order = np.lexsort((~df["isq"].to_numpy() * 1, df["day"].to_numpy(), df["pid"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    pid, day, sidx = df["pid"].to_numpy(), df["day"].to_numpy(), df["sidx"].to_numpy()
    walk = _Walk(pid, day, sidx)
    vals = df[["min", *STATS]].to_numpy(float)
    grp = _attach_grp(pid, sidx, pos)

    mp = params.get("mpg", _MPG)
    e = walk.sums(np.column_stack([vals[:, 0], (~df["isq"].to_numpy()).astype(float)]),
                  walk.clock(mp["H"], mp["rho"]))
    mpg = (e[:, 0] + mp["k0"] * mp["mpg0"]) / (e[:, 1] + mp["k0"])
    tier = tier_of(mpg, params.get("tier_edges", TIER_EDGES))
    cell = grp * N_TIER + tier
    rep = df["rep"].to_numpy()
    rep_idx = np.flatnonzero(rep)
    isq = df["isq"].to_numpy()
    prior = _cell_priors(sidx, cell, day, vals, isq, np.arange(len(pid)), params.get("m0", 1500.0))
    return _Frame(pid=pid, day=day, sidx=sidx, isq=isq, rep=rep, orig=df["orig"].to_numpy(),
                  grp=grp, vals=vals, walk=walk, mpg=mpg, geff=e[:, 1], tier=tier, cell=cell,
                  prior=prior, rep_idx=rep_idx, df=df)


def _cell_priors(sidx, cell, day, vals, isq, rep_idx, m0: float) -> np.ndarray:
    """Season-to-date league per-minute rate of each reported row's cell (games strictly before
    its day), blended with the previous season's final cell rate (m0 pseudo-minutes)."""
    cols = ["min", *STATS]
    real = ~isq
    d = pd.DataFrame(vals[real], columns=cols)
    d["sidx"], d["cell"], d["day"] = sidx[real], cell[real], day[real]
    daily = d.groupby(["sidx", "cell", "day"], sort=True)[cols].sum().reset_index()
    cum = daily.groupby(["sidx", "cell"])[cols].cumsum()
    cum_df = daily[["sidx", "cell", "day"]].copy()
    cum_df[cols] = cum.to_numpy()
    tot = cum_df.groupby(["sidx", "cell"])[cols].last().reset_index()     # last cumsum = total
    nq = len(rep_idx)
    q = pd.DataFrame({"sidx": sidx[rep_idx], "cell": cell[rep_idx], "day": day[rep_idx],
                      "qi": np.arange(nq)})
    q = q.sort_values("day", kind="stable")
    m = pd.merge_asof(q, cum_df.sort_values("day", kind="stable"), on="day",
                      by=["sidx", "cell"], allow_exact_matches=False)
    m = m.sort_values("qi")
    cur = m[cols].fillna(0.0).to_numpy()
    prev_t = tot.copy()
    prev_t["sidx"] = prev_t["sidx"] + 1
    pv = q[["sidx", "cell", "qi"]].merge(prev_t, on=["sidx", "cell"], how="left").sort_values("qi")
    prev = pv[cols].to_numpy(float)
    prev_n = prev[:, 0]
    default = np.array([DEFAULT_RATE[s] for s in STATS])
    ok = np.isfinite(prev_n) & (prev_n > 0)
    mu_prev = np.tile(default, (nq, 1))
    mu_prev[ok] = prev[ok][:, 1:] / prev_n[ok][:, None]
    return (cur[:, 1:] + m0 * mu_prev) / (cur[:, [0]] + m0)


# ============================================================================== core evaluation

def _stat_values(fr: _Frame, params: dict) -> np.ndarray:
    """Per-row contribution columns [min, stats...]. With params['relative'] each stat is divided
    by the cell prior in force at that game (an era-adjusted count), so a league-wide drift in a
    rate does not read as a change in the player."""
    if not params.get("relative", False):
        return fr.vals
    v = fr.vals.copy()
    v[:, 1:] = v[:, 1:] / fr.prior
    return v


def _rates(fr: _Frame, params: dict, stats: Iterable[str] = STATS):
    """Shrunk per-minute rate, effective minutes and prior for every reported row.

    absolute:  r = (Sx + kappa mu) / (Sn + kappa)
    relative:  r = mu (Sz + kappa) / (Sn + kappa),  z_g = x_g / mu_g   (mu_g = prior at game g)"""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    groups: dict[tuple[float, float], list[str]] = {}
    for s in stats:
        p = params["stats"][s]
        groups.setdefault((p["H"], p["rho"]), []).append(s)
    ri = fr.rep_idx
    rel = params.get("relative", False)
    v = _stat_values(fr, params)
    for (h, rho), ss in groups.items():
        idx = [0] + [1 + STATS.index(s) for s in ss]
        e = fr.walk.sums(v[:, idx], fr.walk.clock(h, rho))[ri]
        sn = e[:, 0]
        for j, s in enumerate(ss):
            kappa = params["stats"][s]["kappa"]
            pj = fr.prior[ri, STATS.index(s)]
            out[s] = (pj * (e[:, 1 + j] + kappa) / (sn + kappa) if rel
                      else (e[:, 1 + j] + kappa * pj) / (sn + kappa), sn, pj)
    return out


def _wide(fr: _Frame, params: dict, n_out: int) -> pd.DataFrame:
    rates = _rates(fr, params)
    ri = fr.rep_idx
    cols: dict[str, np.ndarray] = {}
    for s in STATS:
        r, sn, p = rates[s]
        cols[f"r_{s}"], cols[f"n_{s}"], cols[f"p_{s}"] = r, sn, p
    cols["r_dk"] = dk_rate({s: cols[f"r_{s}"] for s in DK_WEIGHTS})
    cols["p_dk"] = dk_rate({s: cols[f"p_{s}"] for s in DK_WEIGHTS})
    cols["n_dk"] = cols["n_pts"]
    cols["mpg"], cols["g_eff"] = fr.mpg[ri], fr.geff[ri]
    cols["tier"], cols["grp"] = fr.tier[ri], fr.grp[ri]
    out = pd.DataFrame(cols)
    out.insert(0, "player_id", fr.pid[ri])
    out.insert(1, "day", fr.day[ri])
    out["_orig"] = fr.orig[ri]
    out = out.sort_values("_orig").reset_index(drop=True)
    assert len(out) == n_out
    return out.drop(columns="_orig")


def _day_to_date(day) -> pd.Series:
    return pd.Series(EPOCH + np.asarray(day, np.int64).astype("timedelta64[D]")).dt.strftime("%Y-%m-%d")


def rate_frame(conn: sqlite3.Connection, queries: pd.DataFrame, params: dict | None = None,
               games: pd.DataFrame | None = None, pos=None) -> pd.DataFrame:
    """State before each query (player_id, game_date): wide frame in query order with columns
    player_id, game_date, r_/n_/p_<stat> (+ dk), mpg, g_eff, tier, grp. Players without any
    game get the prior (n = 0). `games` may be a preloaded `load_games` frame (rows on/after a
    query day are ignored by construction)."""
    params = params or DEFAULT_PARAMS
    queries = queries[["player_id", "game_date"]].reset_index(drop=True)
    qd = _to_day(queries["game_date"])
    if games is None:
        first = season_label(sidx_of(season_of(pd.Timestamp(queries["game_date"].min()).date()))
                             - params["history_seasons"])
        games = load_games(conn, since_season=first,
                           before=pd.Timestamp(queries["game_date"].max()).date())
    games = games[games["day"] <= qd.max()]
    pos = pos if pos is not None else load_positions(conn)
    fr = _prepare(games, pos, params, queries)
    w = _wide(fr, params, len(queries))
    w.insert(1, "game_date", queries["game_date"].to_numpy())
    return w.drop(columns="day")


def walk_forward(conn: sqlite3.Connection, season: str, params: dict | None = None,
                 games: pd.DataFrame | None = None, pos=None) -> pd.DataFrame:
    """Pre-game state for every player-game (min > 0, all season types) of `season`, using only
    earlier games. Columns: game_id, player_id, team, game_date, season_type, min, the actual
    stats, then r_/n_/p_<stat> (+ dk), mpg, g_eff, tier, grp."""
    params = params or DEFAULT_PARAMS
    if games is None:
        first = season_label(sidx_of(season) - params["history_seasons"])
        games = load_games(conn, since_season=first, until_season=season)
    pos = pos if pos is not None else load_positions(conn)
    games = games[games["season"] <= season].reset_index(drop=True)
    fr = _prepare(games, pos, params, None)
    # frame rows are all real & reported; keep the requested season
    w = _wide(fr, params, len(games))
    base = games[["game_id", "player_id", "team", "game_date", "season", "season_type", "min",
                  *STATS]].reset_index(drop=True)
    out = pd.concat([base, w.drop(columns=["player_id", "day"])], axis=1)
    return out[out["season"] == season].reset_index(drop=True)


# ============================================================================== nightly state

def build_state(conn: sqlite3.Connection, as_of: date, params: dict | None = None, *,
                games: pd.DataFrame | None = None, active_days: int = 400) -> pd.DataFrame:
    """Rate state as of `as_of` (games dated before it) for every player with a game in the last
    `active_days` days. Long format = the `rate_state` table, plus stat 'mpg' (rate = decayed
    minutes per game, n_eff = decayed games) and the derived stat 'dk'."""
    params = params or DEFAULT_PARAMS
    first = season_label(sidx_of(season_of(as_of)) - params["history_seasons"])
    if games is None:
        games = load_games(conn, since_season=first, before=as_of)
    else:
        games = games[games["day"] < _day_of(as_of)]
    cutoff = _day_of(as_of) - active_days
    pids = np.sort(games.loc[games["day"] >= cutoff, "player_id"].unique())
    if not len(pids):
        return pd.DataFrame(columns=["as_of", "player_id", "stat", "rate", "n_eff", "prior",
                                     "kappa"])
    q = pd.DataFrame({"player_id": pids, "game_date": as_of.isoformat()})
    w = rate_frame(conn, q, params, games=games)
    return _to_long(w, params, as_of.isoformat())


def _to_long(w: pd.DataFrame, params: dict, as_of: str) -> pd.DataFrame:
    frames = []
    for s in (*STATS, "dk"):
        kappa = params["stats"][s]["kappa"] if s in STATS else np.nan
        frames.append(pd.DataFrame({
            "as_of": as_of, "player_id": w["player_id"].to_numpy(), "stat": s,
            "rate": w[f"r_{s}"].to_numpy(), "n_eff": w[f"n_{s}"].to_numpy(),
            "prior": w[f"p_{s}"].to_numpy(), "kappa": kappa}))
    frames.append(pd.DataFrame({
        "as_of": as_of, "player_id": w["player_id"].to_numpy(), "stat": "mpg",
        "rate": w["mpg"].to_numpy(), "n_eff": w["g_eff"].to_numpy(), "prior": np.nan,
        "kappa": np.nan}))
    return pd.concat(frames, ignore_index=True)


def to_wide(long_df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Long state -> one row per player: r_<stat>, p_<stat>, n_<stat>, mpg, tier."""
    params = params or DEFAULT_PARAMS
    d = long_df[long_df["stat"] != "mpg"]
    out = pd.concat([d.pivot(index="player_id", columns="stat", values=c).add_prefix(p)
                     for c, p in (("rate", "r_"), ("prior", "p_"), ("n_eff", "n_"))], axis=1)
    m = long_df[long_df["stat"] == "mpg"].set_index("player_id")
    out["mpg"] = m["rate"]
    out["g_eff"] = m["n_eff"]
    out["tier"] = tier_of(out["mpg"].fillna(_MPG["mpg0"]), params.get("tier_edges", TIER_EDGES))
    return out


def write_state(conn: sqlite3.Connection, as_of: date, params: dict | None = None, *,
                keep: int | None = 30) -> int:
    """Store the state as of `as_of` in `rate_state` (idempotent: replaces that as_of). With
    `keep`, as_of dates older than the `keep` most recent are pruned. Returns rows written."""
    st = build_state(conn, as_of, params)
    key = as_of.isoformat()
    rows = [(r.as_of, int(r.player_id), r.stat, _f(r.rate), _f(r.n_eff), _f(r.prior), _f(r.kappa))
            for r in st.itertuples(index=False)]
    with conn:
        conn.execute("DELETE FROM rate_state WHERE as_of = ?", (key,))
        conn.executemany("INSERT INTO rate_state(as_of, player_id, stat, rate, n_eff, prior, "
                         "kappa) VALUES (?,?,?,?,?,?,?)", rows)
        if keep:
            conn.execute("DELETE FROM rate_state WHERE as_of NOT IN (SELECT as_of FROM "
                         "(SELECT DISTINCT as_of FROM rate_state ORDER BY as_of DESC LIMIT ?))",
                         (keep,))
    return len(rows)


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def read_state(conn: sqlite3.Connection, as_of: date | str | None = None,
               params: dict | None = None) -> pd.DataFrame:
    """Wide state from `rate_state` at the latest as_of <= `as_of` (default: latest)."""
    if as_of is None:
        row = conn.execute("SELECT MAX(as_of) FROM rate_state").fetchone()
    else:
        k = as_of.isoformat() if isinstance(as_of, date) else as_of
        row = conn.execute("SELECT MAX(as_of) FROM rate_state WHERE as_of <= ?", (k,)).fetchone()
    if not row or row[0] is None:
        return pd.DataFrame()
    long = pd.read_sql_query("SELECT * FROM rate_state WHERE as_of = ?", conn, params=(row[0],))
    w = to_wide(long, params)
    w.attrs["as_of"] = row[0]
    return w


# ============================================================================== fitting

KAPPA_GRID = tuple(float(x) for x in (15, 25, 40, 65, 100, 160, 250, 400, 650, 1000, 1600, 2600,
                                      4000))
H_GRID = (30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 360.0, 480.0, 720.0)
RHO_GRID = (0.3, 0.5, 0.7, 0.85, 1.0)


def _nb_ll_part(y, mu, phi_row):
    """mu-dependent part of the NB log-likelihood (constants in kappa dropped)."""
    mu = np.maximum(mu, 1e-9)
    return float(np.sum(y * np.log(mu) - (y + 1.0 / phi_row) * np.log1p(phi_row * mu)))


def _mom_phi(y, mu, tier) -> list:
    phi = []
    for t in range(N_TIER):
        m = tier == t
        den = float(np.sum(mu[m] ** 2))
        num = float(np.sum((y[m] - mu[m]) ** 2 - mu[m]))
        phi.append(float(np.clip(num / den, 1e-3, 5.0)) if den > 0 else 0.1)
    return phi


def fit_params(conn: sqlite3.Connection, seasons: list[str], *, history_seasons: int = HISTORY_SEASONS,
               m0: float = 1500.0, h_grid=H_GRID, rho_grid=RHO_GRID, kappa_grid=KAPPA_GRID,
               passes: int = 2, relative: bool = False, games: pd.DataFrame | None = None,
               log: Callable[[str], None] | None = None) -> dict:
    """Choose (half-life H, season discount rho, kappa) per stat and the NB dispersion phi per
    (stat, tier) on `seasons` (walk-forward: every training game is predicted from earlier
    games only, history from `history_seasons` before the first training season).

    Stage 1 fixes phi from a default state, picks (H, rho, kappa) per stat by the NB
    log-likelihood of the stat given actual minutes, then refits phi at the chosen state and
    repeats (`passes`)."""
    say = log or (lambda s: None)
    seasons = sorted(seasons)
    params = default_params()
    params.update({"fit_seasons": [seasons[0], seasons[-1]], "history_seasons": history_seasons,
                   "m0": m0, "relative": bool(relative)})
    first = season_label(sidx_of(seasons[0]) - history_seasons)
    if games is None:
        games = load_games(conn, since_season=first, until_season=seasons[-1])
    games = games[(games["season"] >= first) & (games["season"] <= seasons[-1])]
    games = games.reset_index(drop=True)
    pos = load_positions(conn)
    fr = _prepare(games, pos, params, None)
    train = fr.rep & (fr.sidx >= sidx_of(seasons[0]))
    tr = np.flatnonzero(train)                 # frame rows; frame rows == rep rows here
    ri = fr.rep_idx
    assert np.array_equal(ri, np.arange(len(ri)))
    minutes = fr.vals[tr, 0]
    tier = fr.tier[tr]
    y_all = fr.vals[tr, 1:]
    prior_tr = fr.prior[tr]
    say(f"fit: {len(tr)} training games, {len(fr.pid)} frame rows")

    rel = bool(params.get("relative", False))
    vv = _stat_values(fr, params)

    def sums_at(h, rho):
        return fr.walk.sums(vv, fr.walk.clock(h, rho))[tr]

    def shrunk(e, j, kappa):
        if rel:
            return prior_tr[:, j] * (e[:, 1 + j] + kappa) / (e[:, 0] + kappa)
        return (e[:, 1 + j] + kappa * prior_tr[:, j]) / (e[:, 0] + kappa)

    # preliminary phi from the default state
    e = sums_at(120.0, 0.6)
    phi = np.zeros((len(STATS), N_TIER))
    for j, s in enumerate(STATS):
        r = shrunk(e, j, 300.0)
        phi[j] = _mom_phi(y_all[:, j], r * minutes, tier)
    ratio = minutes / np.maximum(fr.mpg[tr], 1.0)
    rbin = np.searchsorted(RATIO_EDGES, ratio, side="right")
    centers = np.array([ratio[rbin == b].mean() if np.any(rbin == b) else
                        (RATIO_EDGES[min(b, len(RATIO_EDGES) - 1)]) for b in range(len(RATIO_EDGES) + 1)])
    mult = np.ones((len(STATS), len(RATIO_EDGES) + 1))      # f_stat per ratio bin
    best: dict[str, dict] = {}
    for it in range(passes):
        adj = [np.interp(ratio, centers, mult[j]) for j in range(len(STATS))]
        best = {s: {"ll": -np.inf} for s in STATS}
        for h in h_grid:
            for rho in rho_grid:
                e = sums_at(h, rho)
                sn = e[:, 0]
                for j, s in enumerate(STATS):
                    phi_row = phi[j][tier]
                    mm = minutes * adj[j]
                    for kappa in kappa_grid:
                        r = shrunk(e, j, kappa)
                        ll = _nb_ll_part(y_all[:, j], r * mm, phi_row)
                        if ll > best[s]["ll"]:
                            best[s] = {"ll": ll, "H": h, "rho": rho, "kappa": kappa}
        for j, s in enumerate(STATS):
            b = best[s]
            e = sums_at(b["H"], b["rho"])
            r = shrunk(e, j, b["kappa"])
            for k in range(mult.shape[1]):
                m = rbin == k
                pred = float(np.sum(r[m] * minutes[m]))
                mult[j, k] = float(np.clip(y_all[m, j].sum() / pred, 0.5, 2.0)) if pred > 0 else 1.0
            phi[j] = _mom_phi(y_all[:, j], r * minutes * np.interp(ratio, centers, mult[j]), tier)
        say(f"fit pass {it + 1}: " + ", ".join(
            f"{s} H{best[s]['H']:.0f}/r{best[s]['rho']}/k{best[s]['kappa']:.0f}" for s in STATS))
    for j, s in enumerate(STATS):
        b = best[s]
        params["stats"][s] = {"H": float(b["H"]), "rho": float(b["rho"]), "kappa": float(b["kappa"]),
                              "phi": [float(x) for x in phi[j]]}
    params["minutes_adj"] = {"edges": list(RATIO_EDGES), "centers": [float(c) for c in centers],
                             "mult": {s: [float(x) for x in mult[j]] for j, s in enumerate(STATS)}}
    params["diagnostics"] = {"n_train": int(len(tr)),
                             "ll_part": {s: best[s]["ll"] / len(tr) for s in STATS}}
    return params


# ============================================================================== usage bump

def fit_usage_bump(conn: sqlite3.Connection, params: dict, seasons: list[str], *,
                   star_top: int = 2, regular_games: int = 7, min_rotation_mpg: float = 15.0,
                   games: pd.DataFrame | None = None, wf: pd.DataFrame | None = None,
                   log: Callable[[str], None] | None = None) -> dict:
    """Role-tier per-minute multipliers when a regular star sat (the Phase 1 usage bump).

    A *star* is one of a team's `star_top` players by pre-game dk-per-minute x mpg state; he
    *sat* if he appeared in >= `regular_games` of the team's previous 10 games and has no row
    in this one. Each rotation player of that team-game (pre-game mpg >= `min_rotation_mpg`) is
    a 'star' (the other star) or 'rotation' player; the multiplier of a stat is
    sum(actual) / sum(pre-game rate x actual minutes) over those player-games (a ratio of sums
    with the minutes table already applied, so it is the per-minute change at fixed minutes). Returns {role: {stat: multiplier,
    'n': games, 'dk': multiplier}}; fit on `seasons` (all < S)."""
    seasons = sorted(seasons)
    if wf is None:
        parts = [walk_forward(conn, s, params, games=games) for s in seasons]
        wf = pd.concat(parts, ignore_index=True)
    wf = wf[wf["season_type"] == "Regular Season"].copy()
    wf["dk_act"] = sum(DK_WEIGHTS[s] * wf[s] for s in DK_WEIGHTS)
    wf["star_score"] = wf["r_dk"] * wf["mpg"]
    flags = star_absence_flags(wf, star_top=star_top, regular_games=regular_games)
    wf = wf.merge(flags, on=["game_id", "team", "player_id"], how="inner")
    wf = wf[(wf["mpg"] >= min_rotation_mpg) & (wf["min"] > 0)]
    wf["role"] = np.where(wf["is_star"], "star", "rotation")
    out: dict = {}
    for role, d in wf.groupby("role"):
        entry = {"n": int(len(d)), "games": int(d[["game_id", "team"]].drop_duplicates().shape[0])}
        mu = {s: d[f"r_{s}"].to_numpy() * d["min"].to_numpy()
                 * minutes_multiplier(params, s, d["min"].to_numpy(), d["mpg"].to_numpy())
              for s in STATS}
        for s in STATS:
            entry[s] = float(d[s].sum() / mu[s].sum()) if mu[s].sum() > 0 else 1.0
        entry["dk"] = float(d["dk_act"].sum() / sum(DK_WEIGHTS[s] * mu[s].sum() for s in DK_WEIGHTS))
        out[role] = entry
    return out


def star_absence_flags(wf: pd.DataFrame, *, star_top: int = 2, regular_games: int = 7
                       ) -> pd.DataFrame:
    """Regular-season team-games in which a regular star sat.

    `wf` is a `walk_forward` frame (needs `star_score`). A team's candidate stars before its
    k-th game are players who appeared in >= `regular_games` of its previous 10 games (or in one
    of the previous 3), scored by `star_score` at their latest appearance; the top `star_top`
    are the stars. Returns (game_id, team, player_id, is_star) for every player who played in
    such a team-game, `is_star` = he is one of those stars."""
    w = wf[wf["season_type"] == "Regular Season"]
    tg = (w[["team", "game_id", "game_date"]].drop_duplicates()
          .sort_values(["team", "game_date", "game_id"]))
    tg["k"] = tg.groupby("team").cumcount()
    w = w.merge(tg[["team", "game_id", "k"]], on=["team", "game_id"])
    hist: dict = {}                        # team -> {player: (ks, scores)}
    for (team, pid), d in w.sort_values("k").groupby(["team", "player_id"]):
        hist.setdefault(team, {})[pid] = (d["k"].to_numpy(), d["star_score"].to_numpy())
    played = w.groupby(["team", "game_id"])["player_id"].apply(list)
    out = []
    for team, gid, k in tg[["team", "game_id", "k"]].itertuples(index=False):
        if k < 10:
            continue
        cands = []
        for pid, (ks, sc) in hist[team].items():
            j = int(np.searchsorted(ks, k, side="left")) - 1      # latest appearance before k
            if j < 0:
                continue
            if int(np.sum(ks[max(j - 9, 0):j + 1] >= k - 10)) < regular_games and k - ks[j] > 3:
                continue
            cands.append((sc[j], pid))
        cands.sort(reverse=True)
        stars = {p for _, p in cands[:star_top]}
        players = played[(team, gid)]
        if stars.issubset(players):
            continue
        out.extend((gid, team, p, p in stars) for p in players)
    return pd.DataFrame(out, columns=["game_id", "team", "player_id", "is_star"])


# ============================================================================== CLI

def main(argv: list[str] | None = None) -> int:
    import argparse
    from .. import config, db

    ap = argparse.ArgumentParser(description="rate state (T1.1)")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--fit", action="store_true", help="print fitted params for seasons < as-of")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    conn = db.connect()
    d = date.fromisoformat(a.as_of) if a.as_of else date.today()
    if a.fit:
        s = sidx_of(season_of(d))
        seasons = [season_label(i) for i in range(s - 8, s)]
        print(params_to_json(fit_params(conn, seasons, log=print)))
        return 0
    n = 0 if a.no_write else write_state(conn, d)
    print(f"rate_state as_of={d} rows={n}")
    return 0


# Fitted by `python -m courtside.proj.state_eval --fit-default --bump` on seasons 2018-19..2025-26
# (8 seasons before 2026-27; refit each season and paste). Production default for build_state.
DEFAULT_PARAMS: dict = json.loads('''
{
 "diagnostics": {
  "ll_part": {
   "ast": 0.317625,
   "blk": -0.671997,
   "dreb": 1.04329,
   "fg3m": -0.589887,
   "fga": 11.0742,
   "fta": -0.620678,
   "oreb": -0.759994,
   "pf": -0.513473,
   "pts": 12.0656,
   "reb": 2.45796,
   "stl": -0.843956,
   "tov": -0.706021
  },
  "n_train": 217078
 },
 "fit_seasons": [
  "2018-19",
  "2025-26"
 ],
 "history_seasons": 6,
 "m0": 1500.0,
 "minutes_adj": {
  "centers": [
   0.239413,
   0.477018,
   0.630996,
   0.782319,
   0.929397,
   1.07603,
   1.21671,
   1.38507,
   1.89941
  ],
  "edges": [
   0.4,
   0.55,
   0.7,
   0.85,
   1.0,
   1.15,
   1.3,
   1.5
  ],
  "mult": {
   "ast": [
    0.924433,
    0.935657,
    0.954417,
    0.992737,
    1.02241,
    1.01077,
    1.01029,
    1.01687,
    1.05316
   ],
   "blk": [
    0.802531,
    0.930665,
    0.92281,
    0.959261,
    0.981116,
    0.97619,
    0.992765,
    0.982473,
    1.01428
   ],
   "dreb": [
    0.964576,
    0.979208,
    0.963519,
    0.989197,
    0.994922,
    0.99543,
    0.997574,
    1.00666,
    1.01115
   ],
   "fg3m": [
    0.709238,
    0.758368,
    0.818182,
    0.901293,
    0.972568,
    1.01118,
    1.03797,
    1.07119,
    1.11685
   ],
   "fga": [
    0.983954,
    0.980291,
    0.979906,
    0.988292,
    0.995818,
    1.00219,
    1.00193,
    1.01005,
    1.02921
   ],
   "fta": [
    0.938664,
    0.896471,
    0.933782,
    0.931895,
    0.968882,
    1.01386,
    1.03231,
    1.03286,
    1.02592
   ],
   "oreb": [
    0.922509,
    0.966553,
    0.996929,
    0.990537,
    0.999205,
    0.9988,
    1.0032,
    1.00463,
    1.01094
   ],
   "pf": [
    1.13204,
    1.21538,
    1.24228,
    1.14583,
    1.06141,
    0.986816,
    0.938596,
    0.909682,
    0.862697
   ],
   "pts": [
    0.774076,
    0.801196,
    0.845937,
    0.917839,
    0.972546,
    1.00689,
    1.02497,
    1.05029,
    1.0895
   ],
   "reb": [
    0.952858,
    0.975051,
    0.971176,
    0.988835,
    0.995375,
    0.995928,
    0.998588,
    1.00555,
    1.01028
   ],
   "stl": [
    0.850591,
    0.958917,
    0.929578,
    0.982815,
    1.0099,
    0.993521,
    0.991659,
    1.00569,
    1.03162
   ],
   "tov": [
    1.27099,
    1.20639,
    1.15911,
    1.08609,
    1.03091,
    0.99094,
    0.958267,
    0.93825,
    0.915629
   ]
  }
 },
 "mpg": {
  "H": 90.0,
  "k0": 2.0,
  "mpg0": 14.0,
  "rho": 0.6
 },
 "relative": false,
 "stats": {
  "ast": {
   "H": 90.0,
   "kappa": 40.0,
   "phi": [
    0.0906121,
    0.0512679,
    0.0427986,
    0.0251383,
    0.018926
   ],
   "rho": 1.0
  },
  "blk": {
   "H": 240.0,
   "kappa": 100.0,
   "phi": [
    0.199657,
    0.114376,
    0.0872316,
    0.0752314,
    0.101751
   ],
   "rho": 1.0
  },
  "dreb": {
   "H": 120.0,
   "kappa": 65.0,
   "phi": [
    0.0320541,
    0.0109509,
    0.0120522,
    0.0127146,
    0.0113942
   ],
   "rho": 1.0
  },
  "fg3m": {
   "H": 120.0,
   "kappa": 25.0,
   "phi": [
    0.0726858,
    0.00992282,
    0.0329873,
    0.0243906,
    0.0437915
   ],
   "rho": 1.0
  },
  "fga": {
   "H": 60.0,
   "kappa": 25.0,
   "phi": [
    0.001,
    0.001,
    0.001,
    0.001,
    0.001
   ],
   "rho": 1.0
  },
  "fta": {
   "H": 120.0,
   "kappa": 65.0,
   "phi": [
    0.88179,
    0.586872,
    0.417636,
    0.26141,
    0.16222
   ],
   "rho": 1.0
  },
  "oreb": {
   "H": 240.0,
   "kappa": 100.0,
   "phi": [
    0.127007,
    0.0948577,
    0.0772672,
    0.0774755,
    0.082813
   ],
   "rho": 0.85
  },
  "pf": {
   "H": 180.0,
   "kappa": 160.0,
   "phi": [
    0.001,
    0.001,
    0.001,
    0.001,
    0.001
   ],
   "rho": 1.0
  },
  "pts": {
   "H": 90.0,
   "kappa": 65.0,
   "phi": [
    0.211601,
    0.14523,
    0.111019,
    0.0757713,
    0.0547987
   ],
   "rho": 1.0
  },
  "reb": {
   "H": 180.0,
   "kappa": 65.0,
   "phi": [
    0.0314659,
    0.0181589,
    0.0150346,
    0.0143886,
    0.0121045
   ],
   "rho": 0.7
  },
  "stl": {
   "H": 360.0,
   "kappa": 250.0,
   "phi": [
    0.110168,
    0.0502598,
    0.0535873,
    0.0252777,
    0.020641
   ],
   "rho": 1.0
  },
  "tov": {
   "H": 180.0,
   "kappa": 100.0,
   "phi": [
    0.0513158,
    0.0177299,
    0.00543105,
    0.001,
    0.001
   ],
   "rho": 0.85
  }
 },
 "tier_edges": [
  12.0,
  20.0,
  26.0,
  32.0
 ],
 "usage_bump": {
  "rotation": {
   "ast": 1.07034,
   "blk": 0.982298,
   "dk": 1.03455,
   "dreb": 1.01504,
   "fg3m": 1.00782,
   "fga": 1.06186,
   "fta": 1.08942,
   "games": 4821,
   "n": 31932,
   "oreb": 0.995525,
   "pf": 0.974316,
   "pts": 1.04776,
   "reb": 1.01039,
   "stl": 1.01283,
   "tov": 1.08919
  },
  "star": {
   "ast": 1.087,
   "blk": 0.998779,
   "dk": 1.05695,
   "dreb": 1.032,
   "fg3m": 1.01913,
   "fga": 1.07629,
   "fta": 1.13385,
   "games": 4149,
   "n": 4149,
   "oreb": 0.999326,
   "pf": 0.946445,
   "pts": 1.07589,
   "reb": 1.0259,
   "stl": 0.999146,
   "tov": 1.14263
  }
 },
 "version": 1
}
''')


if __name__ == "__main__":
    raise SystemExit(main())
