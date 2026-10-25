"""F3x: tracking-dashboard opportunity (re-evaluation H4, docs/research/reeval-B.md).

Adds the `player_tracking_ext` fields (stats.nba.com leaguedashptstats per day: potential
assists, drives, touches by court area, time of possession; `analytics/tracking_ext_store.py`)
to F3, in F3's form (`tracking.py`): per-minute rates shrunk to the position group's league rate
as of the day (K_MIN prior minutes), conversions shrunk with K in denominator units, and the
tracking-implied assist rate vs the EB rate. All strictly before the game day (`common.ew_asof`).

  t3x_pot, t3x_apc, t3x_drv, t3x_dfga, t3x_dpts, t3x_dpass, t3x_fct, t3x_paint, t3x_post,
  t3x_elbow, t3x_top                       log(shrunk rate per minute / group rate)
  t3x_ast_pot  AST per potential assist     t3x_dpts_d  points per drive
  t3x_dtov_d   TOV per drive                (log shrunk conversion / group conversion)
  t3x_gap_ast  log(pot. assists rate x league AST/pot. assist / EB assist rate)   row level
  t3x_ln_min   log(1 + weighted tracked minutes)

Coverage: the table exists from migration 18 (2026-10-25) and is filled per day; history needs
the network backfill (`courtside backfill stats --season S`, ~600 requests a season). Without the
table, or before the first row, every column is NaN (LightGBM's missing branch), as F3 before
its era.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .. import state as ST
from .common import ew_asof

H, RHO = 90.0, 0.8
K_MIN = 240.0
RATES = {"pot": "pot_ast", "apc": "ast_pts_created", "drv": "drives", "dfga": "drive_fga",
         "dpts": "drive_pts", "dpass": "drive_passes", "fct": "front_ct_touches",
         "paint": "paint_touches", "post": "post_touches", "elbow": "elbow_touches",
         "top": "time_of_poss"}
CONVS = {"ast_pot": ("ast", "pot_ast", 150.0), "dpts_d": ("drive_pts", "drives", 100.0),
         "dtov_d": ("drive_tov", "drives", 150.0)}
RATE_COLS = tuple(f"t3x_{q}" for q in RATES)
CONV_COLS = tuple(f"t3x_{c}" for c in CONVS)
TEAM_COLS = RATE_COLS + CONV_COLS + ("t3x_impl_ast", "t3x_ln_min")
ROW_COLS = ("t3x_gap_ast",)
NEUTRAL = {c: 0.0 for c in RATE_COLS + CONV_COLS + ROW_COLS}
PER_STAT = {"pts": ("drv", "dfga", "dpts", "paint", "post", "elbow", "top", "dpts_d"),
            "fg3m": ("drv", "top", "fct"),
            "reb": ("paint", "post"),
            "ast": ("pot", "apc", "dpass", "top", "ast_pot"),
            "stl": ("top",),
            "blk": ("paint", "post"),
            "tov": ("drv", "top", "dtov_d", "post")}
_SRC = tuple(dict.fromkeys(list(RATES.values()) + ["ast", "drive_tov"]))


def has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = "
                        "'player_tracking_ext'").fetchone() is not None


def coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """Rows and filled measures per season (the H4 data check)."""
    if not has_table(conn):
        return pd.DataFrame(columns=["season", "rows", "pot_ast", "drives", "touches", "first",
                                     "last"])
    return pd.read_sql_query(
        "SELECT season, COUNT(*) AS rows, COUNT(pot_ast) AS pot_ast, COUNT(drives) AS drives, "
        "COUNT(touches) AS touches, MIN(game_date) AS first, MAX(game_date) AS last "
        "FROM player_tracking_ext GROUP BY season ORDER BY season", conn)


def load_history(conn: sqlite3.Connection, last: str | None = None, pos=None) -> pd.DataFrame:
    """One row per player-game with every dashboard measure filled (Passing, Drives,
    Possessions), minutes > 0."""
    cols = ["game_id", "player_id", "season", "game_date", "minutes", *_SRC]
    if not has_table(conn):
        return pd.DataFrame(columns=cols + ["day", "sidx", "grp"])
    sql = ("SELECT " + ", ".join(cols) + " FROM player_tracking_ext WHERE minutes > 0 AND "
           "pot_ast IS NOT NULL AND drives IS NOT NULL AND touches IS NOT NULL")
    args: list = []
    if last:
        sql += " AND season <= ?"
        args.append(last)
    h = pd.read_sql_query(sql, conn, params=args)
    num = ["minutes", *_SRC]
    h[num] = h[num].astype(float).fillna(0.0)
    h["day"] = ST._to_day(h["game_date"]) if len(h) else np.zeros(0, np.int64)
    h["sidx"] = h["season"].str[:4].astype(int) if len(h) else np.zeros(0, int)
    pos = pos if pos is not None else ST.load_positions(conn)
    h["grp"] = (ST._attach_grp(h["player_id"].to_numpy(), h["sidx"].to_numpy(), pos)
                if len(h) else np.zeros(0, int))
    return h


def player_features(h: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """TEAM_COLS per query (player_id, day, sidx, grp), strictly before each day."""
    nq = len(q)
    out = pd.DataFrame(np.nan, index=range(nq), columns=list(TEAM_COLS))
    if nq == 0 or not len(h):
        return out
    names = ["_min"] + list(RATES) + [f"{c}_{k}" for c in CONVS for k in ("n", "d")]
    V = np.column_stack([h["minutes"].to_numpy(float)]
                        + [h[src].to_numpy(float) for src in RATES.values()]
                        + [h[x].to_numpy(float) for c in CONVS.values() for x in c[:2]])
    ix = {n: i for i, n in enumerate(names)}
    qd, qs, qg = q["day"].to_numpy(), q["sidx"].to_numpy(), np.asarray(q["grp"], int)
    S = ew_asof(h["player_id"].to_numpy(), h["day"].to_numpy(), h["sidx"].to_numpy(), V,
                q["player_id"].to_numpy(), qd, qs, half_life=H, rho=RHO)
    hg = pd.DataFrame(V, columns=names)
    hg["grp"], hg["day"], hg["sidx"] = h["grp"].to_numpy(), h["day"].to_numpy(), \
        h["sidx"].to_numpy()
    by_g = hg.groupby(["grp", "day", "sidx"], sort=False)[names].sum().reset_index()
    by_all = hg.groupby(["day", "sidx"], sort=False)[names].sum().reset_index()
    L = ew_asof(np.zeros(len(by_all), int), by_all["day"].to_numpy(), by_all["sidx"].to_numpy(),
                by_all[names].to_numpy(float), np.zeros(nq, int), qd, qs, half_life=H, rho=RHO)
    G = ew_asof(by_g["grp"].to_numpy(), by_g["day"].to_numpy(), by_g["sidx"].to_numpy(),
                by_g[names].to_numpy(float), qg, qd, qs, half_life=H, rho=RHO)
    known = (qg < 3) & (G[:, ix["_min"]] > 0)
    P = np.where(known[:, None], G, L)
    era = L[:, ix["_min"]] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for qn in RATES:
            p = P[:, ix[qn]] / P[:, ix["_min"]]
            r = (S[:, ix[qn]] + K_MIN * p) / (S[:, ix["_min"]] + K_MIN)
            out[f"t3x_{qn}"] = np.log(np.maximum(r, 1e-9) / np.maximum(p, 1e-9))
        conv_p = {}
        for cn, (_, _, K) in CONVS.items():
            cp = P[:, ix[f"{cn}_n"]] / P[:, ix[f"{cn}_d"]]
            conv_p[cn] = cp
            c = (S[:, ix[f"{cn}_n"]] + K * cp) / (S[:, ix[f"{cn}_d"]] + K)
            out[f"t3x_{cn}"] = np.log(np.maximum(c, 1e-9) / np.maximum(cp, 1e-9))
        p = P[:, ix["pot"]] / P[:, ix["_min"]]
        r_pot = (S[:, ix["pot"]] + K_MIN * p) / (S[:, ix["_min"]] + K_MIN)
        out["t3x_impl_ast"] = r_pot * conv_p["ast_pot"]
    out["t3x_ln_min"] = np.log1p(np.maximum(S[:, ix["_min"]], 0.0))
    out.loc[~era, :] = np.nan
    return out


def build(conn: sqlite3.Connection, rows: pd.DataFrame, *, h: pd.DataFrame | None = None,
          pos=None) -> pd.DataFrame:
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
    imp = rows["t3x_impl_ast"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(np.maximum(imp, 1e-9) / np.maximum(rows["r_ast"].to_numpy(float), 1e-9))
    return pd.DataFrame({"t3x_gap_ast": np.where(np.isfinite(imp), g, np.nan)},
                        index=rows.index)


def columns(stat: str) -> list[str]:
    cols = []
    for q in PER_STAT[stat]:
        cols.append(f"t3x_{q}")
    if stat == "ast":
        cols.append("t3x_gap_ast")
    return cols + ["t3x_ln_min"]


def monotone(stat: str) -> dict:
    return {"t3x_gap_ast": 1} if stat == "ast" else {}
