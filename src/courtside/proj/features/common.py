"""Shared pieces of the feature store (docs/PLAYERS.md Phase 2F, F0).

* `ew_asof`: exponentially weighted sums per key over history rows strictly before each query
  day (recency half-life in days x a per-season-boundary carry-over factor), the one primitive
  every season-to-date / recency-weighted feature is built on. A query never sees a history row
  of its own day (queries sort before same-day history rows).
* `regulars`: the regulars of every team-game (appeared in >= 7 of the team's previous 10 games
  of the season or in one of the previous 3: the pipeline's `_stars` / usage rule) and whether
  each one played; a regular whose latest earlier appearance was for another team has left
  (traded) and is dropped. Only games before each game are read, plus who played in it (the
  backtest's `avail` oracle).
* `base_rows`: the player-game frame the context layer is fit on: the T1.1 EB pre-game state
  (`state.walk_forward`), the opponent, and per stat the expected count given ACTUAL minutes
  E0_s = r_s x min x f_s(min/mpg) x tier bump (the pipeline's mean, `copula._grids`) and
  E_s = E0_s x the market reconciliation (`env_factor`: points scaled so the team sums to the
  closing-line implied total, the other stats by its alpha power, as `reconcile.reconcile_team`
  does), the actual count y_s and the NB dispersion d_s. E_s is the base the context layer is
  fit and judged on, so it never re-learns what the market total already gives the pipeline.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Sequence

import numpy as np
import pandas as pd

from .. import state as ST

DK_STATS = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")
EXTRA_STATS = ("oreb", "dreb", "fta", "fga")          # used by the opponent indices only
ALL_STATS = DK_STATS + EXTRA_STATS
DK_W = dict(ST.DK_WEIGHTS)
GRP_NAMES = ("G", "F", "C")
LN2 = math.log(2.0)


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def seasons_between(first: str, last: str) -> list[str]:
    return [season_label(y) for y in range(int(first[:4]), int(last[:4]) + 1)]


# ============================================================================== EW as-of sums

def ew_asof(h_key: np.ndarray, h_day: np.ndarray, h_sidx: np.ndarray, h_vals: np.ndarray,
            q_key: np.ndarray, q_day: np.ndarray, q_sidx: np.ndarray, *,
            half_life: float, rho: float) -> np.ndarray:
    """For each query (key, day, season) the sums over history rows of the same key with
    day < query day of vals x 2^(-(day_q - day_h) / half_life) x rho^(season_q - season_h).

    h_vals is (n_h, k); returns (n_q, k) in query order. Keys may be any hashable codes."""
    h_vals = np.asarray(h_vals, float)
    if h_vals.ndim == 1:
        h_vals = h_vals[:, None]
    nq, k = len(q_key), h_vals.shape[1]
    if nq == 0:
        return np.zeros((0, k))
    keys = pd.factorize(np.concatenate([np.asarray(h_key), np.asarray(q_key)]))[0]
    nh = len(h_key)
    day = np.concatenate([np.asarray(h_day, float), np.asarray(q_day, float)])
    sidx = np.concatenate([np.asarray(h_sidx, float), np.asarray(q_sidx, float)])
    isq = np.zeros(nh + nq, np.int8)
    isq[nh:] = 1
    vals = np.zeros((nh + nq, k))
    vals[:nh] = h_vals
    # queries (isq=1 -> rank 0) before same-day history rows
    order = np.lexsort((1 - isq, day, keys))
    keys, day, sidx, vals = keys[order], day[order], sidx[order], vals[order]
    start = np.ones(len(keys), bool)
    start[1:] = keys[1:] != keys[:-1]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    clock = ((day - day[first][gid]) * (LN2 / half_life)
             + (sidx - sidx[first][gid]) * (-math.log(max(rho, 1e-12))))
    t = vals * np.exp(clock)[:, None]
    c = pd.DataFrame(t).groupby(gid, sort=False).cumsum().to_numpy()
    # exclusive: drop the row itself (query rows carry zeros, so inclusive == exclusive)
    e = (c - t) * np.exp(-clock)[:, None]
    out = np.empty((nq, k))
    qpos = np.flatnonzero(isq[order] == 1)
    out[order[qpos] - nh] = e[qpos]
    return out


def league_asof(h_day: np.ndarray, h_sidx: np.ndarray, h_vals: np.ndarray, q_day: np.ndarray,
                q_sidx: np.ndarray, m0: float) -> np.ndarray:
    """Season-to-date league sums before each query day blended with the previous season's
    final rate: returns (n_q, k) ratios num/den for column pairs (vals[:, 2j], vals[:, 2j+1]),
    i.e. (cum_num + m0 * prev_rate) / (cum_den + m0) with m0 in denominator units (per pair,
    scaled by the previous season's mean denominator per row so m0 means 'rows')."""
    h_vals = np.asarray(h_vals, float)
    k2 = h_vals.shape[1]
    d = pd.DataFrame(h_vals)
    d["sidx"], d["day"] = np.asarray(h_sidx), np.asarray(h_day)
    cols = list(range(k2))
    daily = d.groupby(["sidx", "day"], sort=True)[cols].sum().reset_index()
    cnt = d.groupby(["sidx", "day"], sort=True).size().to_numpy()
    cum = daily.groupby("sidx")[cols].cumsum()
    cum_df = daily[["sidx", "day"]].copy()
    cum_df[cols] = cum.to_numpy()
    cum_df["_n"] = pd.Series(cnt).groupby(daily["sidx"].to_numpy()).cumsum().to_numpy()
    tot = cum_df.groupby("sidx")[cols + ["_n"]].last()
    q = pd.DataFrame({"sidx": np.asarray(q_sidx), "day": np.asarray(q_day, float),
                      "qi": np.arange(len(q_day))}).sort_values("day", kind="stable")
    cum_df["day"] = cum_df["day"].astype(float)
    m = pd.merge_asof(q, cum_df.sort_values("day", kind="stable"), on="day", by="sidx",
                      allow_exact_matches=False).sort_values("qi")
    cur = m[cols].fillna(0.0).to_numpy()
    prev = tot.reindex(m["sidx"].to_numpy() - 1)
    pv = prev[cols].to_numpy(float)
    pn = prev["_n"].to_numpy(float)
    out = np.zeros((len(q_day), k2 // 2))
    for j in range(k2 // 2):
        num, den = cur[:, 2 * j], cur[:, 2 * j + 1]
        pr_num, pr_den = pv[:, 2 * j], pv[:, 2 * j + 1]
        ok = np.isfinite(pr_den) & (pr_den > 0)
        prev_rate = np.where(ok, pr_num / np.where(ok, pr_den, 1.0), np.nan)
        prev_den_row = np.where(ok, pr_den / np.maximum(pn, 1.0), 0.0)
        mm = m0 * prev_den_row
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (num + mm * np.nan_to_num(prev_rate)) / (den + mm)
        out[:, j] = r
    return out


# ============================================================================== regulars

def regulars(box: pd.DataFrame) -> pd.DataFrame:
    """Regulars of every team-game. `box` has game_id, team, player_id, day, sidx (one row per
    player who played). Returns (game_id, team, day, sidx, player_id, played, regular) for every
    player who is a regular or played; a regular appeared in >= 7 of the team's previous 10
    games of the season or in one of its previous 3. An absent regular whose latest earlier
    appearance was for another team (traded) is dropped (causal: no later game is read)."""
    w = box[["game_id", "team", "player_id", "day", "sidx"]]
    tg = (w[["team", "game_id", "day", "sidx"]].drop_duplicates()
          .sort_values(["team", "sidx", "day", "game_id"]).reset_index(drop=True))
    tg["k"] = tg.groupby(["team", "sidx"]).cumcount()
    w = w.merge(tg[["team", "game_id", "k"]], on=["team", "game_id"])
    out = []
    tg_by = {key: d for key, d in tg.groupby(["team", "sidx"], sort=False)}
    for (team, sidx), d in w.groupby(["team", "sidx"], sort=False):
        g = tg_by[(team, sidx)]
        G = len(g)
        pids = np.sort(d["player_id"].unique())
        pi = np.searchsorted(pids, d["player_id"].to_numpy())
        played = np.zeros((G, len(pids)), bool)
        played[d["k"].to_numpy(), pi] = True
        c = np.vstack([np.zeros((1, len(pids))), np.cumsum(played, 0)])
        ar = np.arange(G)
        app10 = c[ar] - c[np.maximum(ar - 10, 0)]
        app3 = c[ar] - c[np.maximum(ar - 3, 0)]
        reg = (app10 >= 7) | (app3 >= 1)
        kk, pp = np.nonzero(reg | played)
        out.append(pd.DataFrame({"game_id": g["game_id"].to_numpy()[kk], "team": team,
                                 "day": g["day"].to_numpy()[kk], "sidx": sidx,
                                 "player_id": pids[pp], "played": played[kk, pp],
                                 "regular": reg[kk, pp]}))
    if not out:
        return pd.DataFrame(columns=["game_id", "team", "day", "sidx", "player_id", "played",
                                     "regular"])
    r = pd.concat(out, ignore_index=True)
    # causal trade rule: an absent regular whose latest appearance before the game was for
    # another team has left (traded / waived and signed elsewhere)
    ab = r[~r["played"]].copy()
    app = (w[["player_id", "day", "team"]].rename(columns={"team": "last_team", "day": "lday"})
           .sort_values("lday"))
    ab = ab.sort_values("day")
    ab = pd.merge_asof(ab, app, left_on="day", right_on="lday", by="player_id",
                       direction="backward", allow_exact_matches=False)
    left = ab["last_team"].notna() & (ab["last_team"] != ab["team"])
    drop = ab.loc[left, ["game_id", "player_id"]]
    r = r.merge(drop.assign(_d=1), on=["game_id", "player_id"], how="left")
    r = r[r["_d"].isna()].drop(columns="_d")
    return r.sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)


# ============================================================================== base frame

def load_team_games(conn: sqlite3.Connection, first: str, last: str) -> pd.DataFrame:
    """team_games (+ team_games_adv poss) for seasons [first, last], with day / sidx and the
    opponent's box on the same row (prefix o_)."""
    cols = ["min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast",
            "stl", "blk", "tov", "pf", "pts"]
    t = pd.read_sql_query(
        "SELECT t.game_id, t.team, t.opp, t.season, t.season_type, t.game_date, t.is_home, "
        + ", ".join(f"t.{c}" for c in cols) + ", a.poss FROM team_games t LEFT JOIN "
        "team_games_adv a ON a.game_id = t.game_id AND a.team = t.team "
        "WHERE t.season >= ? AND t.season <= ?", conn, params=(first, last))
    t[cols] = t[cols].astype(float)
    est = t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    t["poss"] = t["poss"].astype(float).where(t["poss"].notna() & (t["poss"] > 0), est)
    o = t[["game_id", "team"] + cols + ["poss"]].rename(
        columns={"team": "opp", **{c: f"o_{c}" for c in cols + ["poss"]}})
    t = t.merge(o, on=["game_id", "opp"], how="inner")
    t["day"] = ST._to_day(t["game_date"]) if len(t) else np.zeros(0, np.int64)
    t["sidx"] = t["season"].str[:4].astype(int)
    return t.sort_values(["day", "game_id", "team"]).reset_index(drop=True)


def bump_roles(wf: pd.DataFrame) -> np.ndarray:
    """Tier-bump role per walk-forward row (0 none, 1 the other star, 2 rotation) by the
    pipeline's rule (`state.star_absence_flags`; regular season only)."""
    w = wf.copy()
    w["star_score"] = w["r_dk"] * w["mpg"]
    fl = ST.star_absence_flags(w)
    m = w[["game_id", "team", "player_id", "mpg"]].merge(
        fl, on=["game_id", "team", "player_id"], how="left")
    is_star = m["is_star"].fillna(False).astype(bool).to_numpy()
    flagged = m["is_star"].notna().to_numpy()
    return np.where(is_star, 1, np.where(flagged & (m["mpg"].to_numpy() >= 15.0), 2, 0))


def base_rows(conn: sqlite3.Connection, seasons: Sequence[str], sp: dict, *,
              games: pd.DataFrame | None = None, pos=None) -> pd.DataFrame:
    """Every player-game (min > 0, all season types) of `seasons` with the pre-game EB state
    (fit params `sp`), the opponent, and per stat y_<s>, E_<s> (expected count given the actual
    minutes, with the minutes table and the tier bump), d_<s> (NB phi by tier) and r_/p_/n_<s>."""
    seasons = sorted(seasons)
    if games is None:
        first = ST.season_label(ST.sidx_of(seasons[0]) - sp.get("history_seasons", 6))
        games = ST.load_games(conn, since_season=first, until_season=seasons[-1])
    pos = pos if pos is not None else ST.load_positions(conn)
    parts = [ST.walk_forward(conn, s, sp, games=games, pos=pos) for s in seasons]
    wf = pd.concat(parts, ignore_index=True)
    wf["day"] = ST._to_day(wf["game_date"])
    wf["sidx"] = wf["season"].str[:4].astype(int)
    opp = pd.read_sql_query(
        "SELECT game_id, team, opp FROM team_games WHERE season >= ? AND season <= ?", conn,
        params=(seasons[0], seasons[-1]))
    wf = wf.merge(opp, on=["game_id", "team"], how="left")
    wf = wf[wf["opp"].notna()].reset_index(drop=True)
    role = np.zeros(len(wf), int)
    rs = (wf["season_type"] == "Regular Season").to_numpy()
    if rs.any():
        role[rs] = bump_roles(wf[rs].reset_index(drop=True))
    wf["bump_role"] = role
    mins = wf["min"].to_numpy(float)
    mpg = wf["mpg"].to_numpy(float)
    tier = wf["tier"].to_numpy(int)
    new = {}
    for s in ALL_STATS:
        b = np.array([1.0, ST.bump_for(sp, "star", s), ST.bump_for(sp, "rotation", s)])
        new[f"E0_{s}"] = (wf[f"r_{s}"].to_numpy(float) * mins
                          * ST.minutes_multiplier(sp, s, mins, mpg) * b[role])
        new[f"d_{s}"] = ST.phi_for(sp, s, tier)
    wf = pd.concat([wf, pd.DataFrame(new)], axis=1)
    c = env_factor(conn, wf, seasons)
    wf["c_env"] = c
    env = {f"E_{s}": wf[f"E0_{s}"].to_numpy() * (c if s == "pts" else c ** ENV_ALPHA)
           for s in ALL_STATS}
    return pd.concat([wf, pd.DataFrame(env)], axis=1)


ENV_ALPHA = 0.5               # copula alpha (fit 0.49-0.52 on the 2019-20 .. 2025-26 bundles)
ENV_BOUNDS = (0.75, 1.33)     # reconcile.SCALE_BOUNDS


def env_factor(conn: sqlite3.Connection, wf: pd.DataFrame, seasons: Sequence[str]) -> np.ndarray:
    """The pipeline's market reconciliation given actual minutes: per team-game
    c = implied team points (closing spread / total) / sum of the team's expected points, so
    E_pts x c sums to the implied total and the other stats scale by c^alpha (reconcile.py).
    1.0 without a closing line."""
    cl = pd.read_sql_query(
        "SELECT game_id, AVG(home_spread_close) AS hs, AVG(total_close) AS tot FROM "
        "closing_lines WHERE home_spread_close IS NOT NULL AND total_close IS NOT NULL "
        "GROUP BY game_id", conn)
    hm = pd.read_sql_query(
        "SELECT game_id, team, is_home FROM team_games WHERE season >= ? AND season <= ?", conn,
        params=(min(seasons), max(seasons)))
    t = hm.merge(cl, on="game_id", how="inner")
    sign = np.where(t["is_home"].astype(int).to_numpy() == 1, 1.0, -1.0)
    t["implied"] = (t["tot"] - sign * t["hs"]) / 2.0
    tp = wf.groupby(["game_id", "team"])["E0_pts"].sum().rename("raw").reset_index()
    tp = tp.merge(t[["game_id", "team", "implied"]], on=["game_id", "team"], how="left")
    tp["c"] = np.clip(tp["implied"] / tp["raw"].clip(lower=1.0), *ENV_BOUNDS)
    tp["c"] = tp["c"].fillna(1.0)
    m = wf[["game_id", "team"]].merge(tp[["game_id", "team", "c"]], on=["game_id", "team"],
                                      how="left")
    return m["c"].fillna(1.0).to_numpy(float)
