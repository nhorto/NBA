"""Assembling the (player, game) feature frame (docs/PLAYERS.md Phase 2F, F0).

  team_game_features(conn, wf, sp, ...)  F1 + F2 per (game_id, team), from `common.base_rows`
  day_features(conn, wf, t, day, games)  F2 rows for one day's games (pipeline / live hook)
  attach(rows, tg)                       join to player rows (any frame with the EB state
                                         columns + game_id, team) and add the row-level columns
  build_frame(conn, season, sp, ...)     every player-game of seasons S-n_train .. S with
                                         y_/E_/d_<s> and all feature groups; cached as parquet
                                         under <data>/proj/features/
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from ... import config
from .. import state as ST
from . import opp_avail, opp_defense, player
from .common import ALL_STATS, base_rows, load_team_games, season_label

FRAME_VERSION = 4
N_TRAIN = 6


def cache_dir() -> Path:
    return config.data_dir() / "proj" / "features"


def team_game_features(conn: sqlite3.Connection, wf: pd.DataFrame, sp: dict, *,
                       games: pd.DataFrame | None = None, pos=None,
                       dvp_params: dict | None = None, t: pd.DataFrame | None = None,
                       log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """(game_id, team) -> F1 (actual absences of the opponent) and F2 columns."""
    say = log or (lambda m: None)
    t0 = time.time()
    f1 = opp_avail.build(conn, wf, sp, games=games, pos=pos)
    say(f"features: F1 {len(f1)} team-games ({time.time() - t0:.0f}s)")
    t0 = time.time()
    f2 = opp_defense.build(conn, wf, params=dvp_params, t=t)
    say(f"features: F2 {len(f2)} team-games ({time.time() - t0:.0f}s)")
    return f1.merge(f2, on=["game_id", "team"], how="outer")


def day_features(conn: sqlite3.Connection, wf: pd.DataFrame, t: pd.DataFrame, day,
                 games: pd.DataFrame, *, dvp_params: dict | None = None) -> pd.DataFrame:
    """F2 team-game rows for the games of one day (the pipeline / live path): `games` has
    game_id, home, away; `wf` = `common.base_rows` and `t` = `common.load_team_games` history
    (anything on or after `day` is ignored by construction). F1 is per availability scenario
    (`opp_avail.scenario_features`) and is not part of the shipped layer."""
    d = pd.Timestamp(day).date()
    dn = int(ST._day_of(d))
    sidx = ST.sidx_of(ST.season_of(d))
    q = pd.concat([pd.DataFrame({"game_id": games["game_id"], "team": games["home"],
                                 "opp": games["away"]}),
                   pd.DataFrame({"game_id": games["game_id"], "team": games["away"],
                                 "opp": games["home"]})], ignore_index=True)
    q["day"], q["sidx"] = dn, sidx
    return opp_defense.build(conn, wf, params=dvp_params, t=t, queries=q)


def attach(rows: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """rows (game_id, team, EB state columns) + team-game features + row-level columns."""
    out = rows.merge(tg, on=["game_id", "team"], how="left")
    for c in opp_avail.TEAM_COLS:
        if c in out:
            out[c] = out[c].fillna(0.0)
    extra = [player.row_features(out)]
    if "f1_vac_G" in out:
        extra.append(opp_avail.row_features(out))
    if "f2_dvp_pts_G" in out:
        extra.append(opp_defense.row_features(out))
    return pd.concat([out] + extra, axis=1)


def _key(season: str, sp: dict, n_train: int, dvp_params: dict | None) -> str:
    blob = json.dumps({"v": FRAME_VERSION, "season": season, "n": n_train,
                       "sp": sp.get("fit_seasons"), "kappa": {s: sp["stats"][s]["kappa"]
                                                              for s in sp["stats"]},
                       "dvp": dvp_params or opp_defense.DVP_PARAMS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_frame(conn: sqlite3.Connection, season: str, sp: dict, *, n_train: int = N_TRAIN,
                dvp_params: dict | None = None, cache: bool = True,
                log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Rows of seasons S-n_train .. S (all season types, min > 0) with the base expectation
    and every feature group. One extra season before is loaded as history for F2. `sp` =
    the rate-state params fit on seasons < S (the bundle's)."""
    say = log or (lambda m: None)
    path = cache_dir() / f"frame_{season}_{_key(season, sp, n_train, dvp_params)}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    s0 = int(season[:4])
    first_hist = season_label(s0 - n_train - 1)
    seasons = [season_label(y) for y in range(s0 - n_train - 1, s0 + 1)]
    games = ST.load_games(conn, since_season=season_label(s0 - n_train - 1
                                                          - sp.get("history_seasons", 6)),
                          until_season=season)
    pos = ST.load_positions(conn)
    wf = base_rows(conn, seasons, sp, games=games, pos=pos)
    say(f"features: base rows {len(wf)} ({time.time() - t0:.0f}s)")
    t = load_team_games(conn, first_hist, season)
    tg = team_game_features(conn, wf, sp, games=games, pos=pos, dvp_params=dvp_params, t=t,
                            log=say)
    keep = wf["season"] >= season_label(s0 - n_train)
    keep_cols = (["game_id", "player_id", "team", "opp", "game_date", "season", "season_type",
                  "day", "sidx", "min", "mpg", "g_eff", "tier", "grp", "bump_role", "r_dk", "p_dk"]
                 + [f"{p}_{s}" for s in ST.STATS for p in ("r", "p", "n")]
                 + list(ALL_STATS) + [f"E_{s}" for s in ALL_STATS] + [f"E0_{s}" for s in ALL_STATS]
                 + ["c_env"]
                 + [f"d_{s}" for s in ALL_STATS])
    keep_cols = list(dict.fromkeys(keep_cols))
    rows = wf.loc[keep, keep_cols].reset_index(drop=True)
    out = attach(rows, tg)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    say(f"features: frame {season} {len(out)} rows x {out.shape[1]} cols "
        f"({time.time() - t0:.0f}s)")
    return out
