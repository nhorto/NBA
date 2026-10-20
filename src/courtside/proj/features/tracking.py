"""F3: tracking role signals (docs/PLAYERS.md Phase 2F).

Per player, strictly before the game day (`common.ew_asof`: recency half-life H days and a
per-season carry-over RHO), shrunk to the league rate of the player's position group at that
date (league sums as of the same day, same weights):

* **Per-minute rates** from `player_tracking` (stats.nba.com boxscoreplayertrackv3, 2019-20 on)
  and `player_hustle` (boxscorehustlev2): rebound chances (off / def), touches, passes,
  secondary and free-throw assists, contested / uncontested FGA, FGA defended at the rim,
  distance; contested 2PT / 3PT shots, deflections, charges, screen assists, loose balls,
  box-outs. r = (S_q + K_MIN p_g) / (S_min + K_MIN); feature t3_<q> = log(r / p_g).
* **Conversions** (own box stat per tracking opportunity): OREB / off. chances, DREB / def.
  chances, AST / pass, uncontested and contested FG%, STL / deflection, BLK / contested 2PT;
  c = (S_num + K c_g) / (S_den + K); feature log(c / c_g) (heavily shrunk: these are the noisy,
  luck-heavy parts).
* **Tracking-implied rates vs the EB state** (row level, `row_features`): e.g. rebounds implied
  by the player's rebound-chance rates at the league conversion of his group vs his EB rebound
  rate: t3_gap_reb = log(implied / r_reb); likewise assists (passes), steals (deflections) and
  blocks (contested 2PT shots). This is the "opportunity stabilises faster than the outcome"
  signal; the GBM sees it next to the P descriptors.
* t3_ln_min = log(1 + weighted tracked minutes) (how much evidence there is).

Rows dated before the tracking era (no league tracking sums yet) are NaN (LightGBM's missing
branch), never the prior: "no data source" is not "a league-average player". A player with no
tracking history inside the era (rookie, new signing) gets the group prior (0) and ln_min 0.
Potential assists are not in the source (stats.nba.com leaves matchupPotentialAssists at 0) and
drives are not in the box-score tracking feed.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .. import state as ST
from .common import ew_asof

TRACK_START = "2019-20"
H, RHO = 90.0, 0.8
K_MIN = 240.0                       # prior minutes for the per-minute rates
# name: (source column, feed) ; feed 't' = player_tracking, 'h' = player_hustle
RATES = {
    "rbc_o": ("reb_chances_off", "t"), "rbc_d": ("reb_chances_def", "t"),
    "touch": ("touches", "t"), "pass": ("passes", "t"), "sast": ("secondary_ast", "t"),
    "ftast": ("ft_ast", "t"), "cfga": ("cont_fga", "t"), "ufga": ("uncont_fga", "t"),
    "drim": ("def_rim_fga", "t"), "dist": ("distance", "t"),
    "cs2": ("contested_2pt", "h"), "cs3": ("contested_3pt", "h"), "defl": ("deflections", "h"),
    "chg": ("charges_drawn", "h"), "scrn": ("screen_assists", "h"), "loose": ("loose_total", "h"),
    "boxo": ("box_outs", "h"),
}
# name: (numerator, denominator, K in denominator units, feed)
CONVS = {
    "oreb_c": ("oreb", "reb_chances_off", 40.0, "t"),
    "dreb_c": ("dreb", "reb_chances_def", 80.0, "t"),
    "ast_p": ("ast", "passes", 400.0, "t"),
    "ufg": ("uncont_fgm", "uncont_fga", 80.0, "t"),
    "cfg": ("cont_fgm", "cont_fga", 80.0, "t"),
    "stl_d": ("stl", "deflections", 40.0, "h"),
    "blk_c": ("blk", "contested_2pt", 60.0, "h"),
}
# implied per-minute rate of a DK stat = sum of (rate x league conversion of the group)
IMPLIED = {"reb": (("rbc_o", "oreb_c"), ("rbc_d", "dreb_c")), "ast": (("pass", "ast_p"),),
           "stl": (("defl", "stl_d"),), "blk": (("cs2", "blk_c"),)}

RATE_COLS = tuple(f"t3_{q}" for q in RATES)
CONV_COLS = tuple(f"t3_{c}" for c in CONVS)
IMPL_COLS = tuple(f"t3_impl_{s}" for s in IMPLIED)
TEAM_COLS = RATE_COLS + CONV_COLS + IMPL_COLS + ("t3_ln_min",)   # per (player, day)
ROW_COLS = tuple(f"t3_gap_{s}" for s in IMPLIED)
NEUTRAL = {c: 0.0 for c in RATE_COLS + CONV_COLS + ROW_COLS}

PER_STAT = {
    "pts": ("touch", "cfga", "ufga", "ufg", "cfg", "ftast", "scrn", "dist"),
    "fg3m": ("ufga", "cfga", "ufg", "touch", "dist"),
    "reb": ("rbc_o", "rbc_d", "oreb_c", "dreb_c", "boxo", "loose", "drim"),
    "ast": ("pass", "touch", "sast", "ftast", "ast_p", "scrn"),
    "stl": ("defl", "loose", "stl_d", "cs3", "dist"),
    "blk": ("drim", "cs2", "blk_c", "chg", "rbc_d"),
    "tov": ("touch", "pass", "cfga", "dist", "chg"),
}

_T_COLS = ("minutes", "reb_chances_off", "reb_chances_def", "touches", "passes",
           "secondary_ast", "ft_ast", "cont_fgm", "cont_fga", "uncont_fgm", "uncont_fga",
           "def_rim_fga", "distance")
_H_COLS = ("contested_2pt", "contested_3pt", "deflections", "charges_drawn", "screen_assists",
           "loose_total", "box_outs")


def load_history(conn: sqlite3.Connection, first: str | None = None,
                 last: str | None = None, before=None, pos=None) -> pd.DataFrame:
    """One row per tracked player-game (player_games x player_tracking, hustle left-joined) with
    day, sidx, grp, the box stats the conversions need and has_t / has_h feed flags."""
    first = max(first or TRACK_START, TRACK_START)
    sql = ("SELECT p.game_id, p.player_id, p.season, p.game_date, p.min, p.oreb, p.dreb, p.ast, "
           "p.stl, p.blk, " + ", ".join(f"t.{c} AS {c}" for c in _T_COLS) + ", "
           + ", ".join(f"h.{c} AS {c}" for c in _H_COLS)
           + ", h.player_id IS NOT NULL AS has_h FROM player_games p "
           "JOIN player_tracking t ON t.game_id = p.game_id AND t.player_id = p.player_id "
           "LEFT JOIN player_hustle h ON h.game_id = p.game_id AND h.player_id = p.player_id "
           "WHERE p.min > 0 AND p.season >= ?")
    args: list = [first]
    if last:
        sql += " AND p.season <= ?"
        args.append(last)
    if before is not None:
        sql += " AND p.game_date < ?"
        args.append(pd.Timestamp(before).date().isoformat())
    h = pd.read_sql_query(sql, conn, params=args)
    num = ["min", "oreb", "dreb", "ast", "stl", "blk", *_T_COLS, *_H_COLS]
    h[num] = h[num].astype(float).fillna(0.0)
    h["has_t"] = 1.0
    h["has_h"] = h["has_h"].astype(float)
    h["day"] = ST._to_day(h["game_date"]) if len(h) else np.zeros(0, np.int64)
    h["sidx"] = h["season"].str[:4].astype(int) if len(h) else np.zeros(0, int)
    pos = pos if pos is not None else ST.load_positions(conn)
    h["grp"] = ST._attach_grp(h["player_id"].to_numpy(), h["sidx"].to_numpy(), pos) if len(h) \
        else np.zeros(0, int)
    return h


def _pairs(h: pd.DataFrame) -> tuple[np.ndarray, list]:
    """(n, k) value matrix of numerator / denominator columns and their names."""
    cols, names = [], []
    tmin = h["min"].to_numpy(float) * h["has_t"].to_numpy(float)
    hmin = h["min"].to_numpy(float) * h["has_h"].to_numpy(float)
    cols += [tmin, hmin]
    names += ["_tmin", "_hmin"]
    for q, (src, _) in RATES.items():
        cols.append(h[src].to_numpy(float))
        names.append(q)
    for c, (num, den, _, feed) in CONVS.items():
        flag = h["has_t" if feed == "t" else "has_h"].to_numpy(float)
        cols += [h[num].to_numpy(float) * flag, h[den].to_numpy(float)]
        names += [f"{c}_n", f"{c}_d"]
    return np.column_stack(cols), names


def player_features(h: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """F3 per query (player_id, day, sidx, grp), strictly before each day. Returns the TEAM_COLS
    (per player-day) in query order."""
    nq = len(q)
    out = pd.DataFrame(index=range(nq), columns=list(TEAM_COLS), dtype=float)
    if nq == 0:
        return out
    V, names = _pairs(h)
    ix = {n: i for i, n in enumerate(names)}
    qd, qs = q["day"].to_numpy(), q["sidx"].to_numpy()
    S = ew_asof(h["player_id"].to_numpy(), h["day"].to_numpy(), h["sidx"].to_numpy(), V,
                q["player_id"].to_numpy(), qd, qs, half_life=H, rho=RHO)
    # league sums per position group (unknown group -> all players), same weights
    hg = pd.DataFrame(V, columns=names)
    hg["grp"], hg["day"], hg["sidx"] = h["grp"].to_numpy(), h["day"].to_numpy(), \
        h["sidx"].to_numpy()
    by_g = hg.groupby(["grp", "day", "sidx"], sort=False)[names].sum().reset_index()
    by_all = hg.groupby(["day", "sidx"], sort=False)[names].sum().reset_index()
    qg = np.asarray(q["grp"].to_numpy(), int)
    L = ew_asof(np.zeros(len(by_all), int), by_all["day"].to_numpy(), by_all["sidx"].to_numpy(),
                by_all[names].to_numpy(float), np.zeros(nq, int), qd, qs, half_life=H, rho=RHO)
    G = ew_asof(by_g["grp"].to_numpy(), by_g["day"].to_numpy(), by_g["sidx"].to_numpy(),
                by_g[names].to_numpy(float), qg, qd, qs, half_life=H, rho=RHO)
    known = (qg < 3) & (G[:, ix["_tmin"]] > 0)
    P = np.where(known[:, None], G, L)                  # the prior's sums
    era = L[:, ix["_tmin"]] > 0                         # any league tracking before the day
    with np.errstate(divide="ignore", invalid="ignore"):
        prior, conv_p = {}, {}
        for qn, (_, feed) in RATES.items():
            m = "_tmin" if feed == "t" else "_hmin"
            p = P[:, ix[qn]] / P[:, ix[m]]
            prior[qn] = p
            r = (S[:, ix[qn]] + K_MIN * p) / (S[:, ix[m]] + K_MIN)
            out[f"t3_{qn}"] = np.log(np.maximum(r, 1e-9) / np.maximum(p, 1e-9))
            prior[qn + "_r"] = r
        for cn, (_, _, K, _) in CONVS.items():
            cp = P[:, ix[f"{cn}_n"]] / P[:, ix[f"{cn}_d"]]
            conv_p[cn] = cp
            c = (S[:, ix[f"{cn}_n"]] + K * cp) / (S[:, ix[f"{cn}_d"]] + K)
            out[f"t3_{cn}"] = np.log(np.maximum(c, 1e-9) / np.maximum(cp, 1e-9))
        for s, parts in IMPLIED.items():
            out[f"t3_impl_{s}"] = sum(prior[r + "_r"] * conv_p[c] for r, c in parts)
    out["t3_ln_min"] = np.log1p(np.maximum(S[:, ix["_tmin"]], 0.0))
    out.loc[~era, :] = np.nan
    return out


def build(conn: sqlite3.Connection, rows: pd.DataFrame, *, h: pd.DataFrame | None = None,
          pos=None) -> pd.DataFrame:
    """F3 for frame rows (game_id, player_id, day, sidx, grp): returns game_id, player_id +
    TEAM_COLS (one row per distinct (game_id, player_id) of `rows`)."""
    if h is None:
        last = ST.season_label(int(rows["sidx"].max())) if len(rows) else None
        h = load_history(conn, last=last, pos=pos)
    q = rows[["game_id", "player_id", "day", "sidx", "grp"]].drop_duplicates(
        ["game_id", "player_id"]).reset_index(drop=True)
    f = player_features(h, q)
    f.insert(0, "game_id", q["game_id"].to_numpy())
    f.insert(1, "player_id", q["player_id"].to_numpy())
    return f


def row_features(rows: pd.DataFrame) -> pd.DataFrame:
    """t3_gap_<s> = log(tracking-implied per-minute rate / EB rate r_<s>)."""
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in IMPLIED:
            out[f"t3_gap_{s}"] = np.log(np.maximum(rows[f"t3_impl_{s}"].to_numpy(float), 1e-9)
                                        / np.maximum(rows[f"r_{s}"].to_numpy(float), 1e-9))
            out[f"t3_gap_{s}"] = np.where(np.isfinite(rows[f"t3_impl_{s}"].to_numpy(float)),
                                          out[f"t3_gap_{s}"], np.nan)
    return pd.DataFrame(out, index=rows.index)


def columns(stat: str) -> list[str]:
    cols = [f"t3_{q}" for q in PER_STAT[stat]]
    if stat in IMPLIED:
        cols.append(f"t3_gap_{stat}")
    return cols + ["t3_ln_min"]


def monotone(stat: str) -> dict:
    """More tracking-implied opportunity than the EB rate can only raise the rate."""
    return {f"t3_gap_{stat}": 1} if stat in IMPLIED else {}
