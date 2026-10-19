"""F1: opponent availability (docs/PLAYERS.md Phase 2F).

For a player of team T in a game against O, which of O's rotation regulars are absent and what
roles they fill. A *regular* appeared in >= 7 of O's previous 10 games of the season or in one of
its previous 3 (`common.regulars`), and counts here when his pre-game minutes state is >= 15 mpg.
Roles come from O's pre-game (season-to-date, recency-weighted, shrunk: the T1.1 EB state)
production per game = rate x mpg among those regulars:

  rim protector   blocks            rebounder       defensive rebounds
  ball-handler    assists           perimeter def.  steals among guards / forwards (a box proxy:
                                                    the box has no direct defensive measure)
  scorer          points

Features (team-game level, from T's side; all 0 when O is healthy):
  f1_n_abs, f1_vac_min           absent regulars and their minutes state
  f1_vac_G / _F / _C             vacated minutes by position group (+ row-level f1_vac_same:
                                 the group of the player himself)
  f1_<role>_share                absent share of O's regular production of the role's stat
  f1_<role>_top                  1 when O's top player in that role is absent
  f1_depth                       O's available rotation regulars

Fitting uses the actual absences (the backtest's `avail` protocol: regulars who did not play);
the sim evaluates the same function per availability scenario of the opponent
(`team_features(players, absent_masks)`).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .. import state as ST
from .common import regulars

MIN_MPG = 15.0
ROLES = {"rim": "blk", "reb": "dreb", "bh": "ast", "perim": "stl", "pts": "pts"}
TEAM_COLS = (["f1_n_abs", "f1_vac_min", "f1_vac_G", "f1_vac_F", "f1_vac_C", "f1_depth"]
             + [f"f1_{r}_share" for r in ROLES] + [f"f1_{r}_top" for r in ROLES if r != "pts"])
ROW_COLS = ("f1_vac_same",)
COLS = tuple(TEAM_COLS) + ROW_COLS


def team_features(R: pd.DataFrame) -> pd.DataFrame:
    """F1 values per `key` from a long frame of regulars: key, absent (bool), mpg, grp (0 G,
    1 F, 2 C, 3 unknown) and r_blk, r_dreb, r_ast, r_stl, r_pts (pre-game per-minute rates).
    Rows with mpg < MIN_MPG are ignored. Returns one row per key (index = key)."""
    R = R[R["mpg"].to_numpy(float) >= MIN_MPG]
    keys = pd.Index(pd.unique(R["key"]))
    k = keys.get_indexer(R["key"])
    nk = len(keys)
    ab = R["absent"].to_numpy(bool)
    mpg = R["mpg"].to_numpy(float)
    grp = R["grp"].to_numpy(int)
    out = {}

    def bsum(v):
        return np.bincount(k, weights=v, minlength=nk)

    out["f1_n_abs"] = bsum(ab.astype(float))
    out["f1_vac_min"] = bsum(mpg * ab)
    for g, name in enumerate(("G", "F", "C")):
        out[f"f1_vac_{name}"] = bsum(mpg * ab * (grp == g))
    out["f1_depth"] = bsum((~ab).astype(float))
    for role, stat in ROLES.items():
        prod = R[f"r_{stat}"].to_numpy(float) * mpg
        if role == "perim":
            prod = prod * (grp <= 1)
        tot = bsum(prod)
        out[f"f1_{role}_share"] = np.where(tot > 0, bsum(prod * ab) / np.maximum(tot, 1e-12), 0.0)
        if role != "pts":
            # top producer of the key: max prod; absent flag of that row
            order = np.lexsort((-prod, k))
            first = np.ones(len(order), bool)
            first[1:] = k[order][1:] != k[order][:-1]
            top = np.zeros(nk)
            idx = order[first]
            top[k[idx]] = (ab[idx] & (prod[idx] > 0)).astype(float)
            out[f"f1_{role}_top"] = top
    return pd.DataFrame(out, index=keys)[list(TEAM_COLS)]


def scenario_features(players: pd.DataFrame, absent_masks: np.ndarray,
                      regular: np.ndarray | None = None) -> pd.DataFrame:
    """F1 of one team (the opponent) for each availability scenario: `players` has mpg, grp,
    r_blk, r_dreb, r_ast, r_stl, r_pts; absent_masks (K, n) bool; `regular` (n,) marks the
    regulars (default: all). Returns K rows (index 0..K-1) -- the sim hook."""
    absent_masks = np.atleast_2d(np.asarray(absent_masks, bool))
    K, n = absent_masks.shape
    reg = np.ones(n, bool) if regular is None else np.asarray(regular, bool)
    base = players.reset_index(drop=True).loc[reg]
    R = pd.concat([base.assign(key=i, absent=absent_masks[i][reg]) for i in range(K)],
                  ignore_index=True)
    out = team_features(R).reindex(range(K)).fillna(0.0)
    return out


def build(conn: sqlite3.Connection, wf: pd.DataFrame, sp: dict, *,
          games: pd.DataFrame | None = None, pos=None) -> pd.DataFrame:
    """Team-game F1 table from the actual absences: one row per (game_id, team) = the team the
    features are FOR (its opponent's absences), columns TEAM_COLS. `wf` = `common.base_rows`
    (every player-game with its pre-game state); absent regulars get their state from
    `state.rate_frame` at the game date (strictly before it)."""
    box = wf[["game_id", "team", "player_id", "day", "sidx"]]
    reg = regulars(box)
    reg = reg[reg["regular"]].copy()
    st_cols = ["mpg", "grp", "r_blk", "r_dreb", "r_ast", "r_stl", "r_pts"]
    cur = reg[reg["played"]].merge(wf[["game_id", "player_id"] + st_cols],
                                   on=["game_id", "player_id"], how="left")
    ab = reg[~reg["played"]].reset_index(drop=True)
    if len(ab):
        q = pd.DataFrame({"player_id": ab["player_id"].astype(np.int64),
                          "game_date": ST._day_to_date(ab["day"].to_numpy())})
        rf = ST.rate_frame(conn, q, sp, games=games, pos=pos)
        for c in st_cols:
            ab[c] = rf[c].to_numpy()
    R = pd.concat([cur, ab], ignore_index=True)
    R["absent"] = ~R["played"].to_numpy(bool)
    R["key"] = R["game_id"] + "|" + R["team"]
    T = team_features(R)
    gid_team = T.index.to_series().str.split("|", n=1, expand=True)
    T = T.reset_index(drop=True)
    T.insert(0, "game_id", gid_team[0].to_numpy())
    T.insert(1, "opp", gid_team[1].to_numpy())
    # features for the OTHER team of each game: map (game_id, opp) -> (game_id, team)
    pairs = wf[["game_id", "team", "opp"]].drop_duplicates(["game_id", "team"])
    out = pairs.merge(T, on=["game_id", "opp"], how="left").drop(columns="opp")
    out[list(TEAM_COLS)] = out[list(TEAM_COLS)].fillna(0.0)
    return out.reset_index(drop=True)


def row_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Row-level F1 columns from the joined team-game columns and the player's group."""
    g = rows["grp"].to_numpy(int)
    vac = np.column_stack([rows["f1_vac_G"], rows["f1_vac_F"], rows["f1_vac_C"],
                           np.zeros(len(rows))])
    return pd.DataFrame({"f1_vac_same": vac[np.arange(len(rows)), np.clip(g, 0, 3)]},
                        index=rows.index)


def columns(stat: str) -> list[str]:
    return list(COLS)


def monotone(stat: str) -> dict:
    return {}
