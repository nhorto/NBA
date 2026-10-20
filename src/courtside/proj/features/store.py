"""Assembling the (player, game) feature frame (docs/PLAYERS.md Phase 2F, F0).

  team_game_features(conn, wf, sp, ...)  F1 + F2 per (game_id, team), from `common.base_rows`
  day_features(conn, wf, t, day, games)  F2 rows for one day's games (pipeline / live hook)
  attach(rows, tg)                       join to player rows (any frame with the EB state
                                         columns + game_id, team) and add the row-level columns
  build_frame(conn, season, sp, ...)     every player-game of seasons S-n_train .. S with
                                         y_/E_/d_<s> and all feature groups; cached as parquet
                                         under <data>/proj/features/
  extend_frame(conn, frame, groups, ...) adds F3 / F4 / F5 / F6 (not in the base frame) to a
                                         frame, cached per (frame, group)
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
from . import matchups, opp_avail, opp_defense, player, schedule, shots, tracking
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
                 games: pd.DataFrame, *, dvp_params: dict | None = None,
                 totals: dict | None = None) -> pd.DataFrame:
    """F2 team-game rows for the games of one day (the pipeline / live path): `games` has
    game_id, home, away; `wf` = `common.base_rows` and `t` = `common.load_team_games` history
    (anything on or after `day` is ignored by construction). `totals` {game_id: market total}
    (the pipeline's environment) replaces the closing totals of the pace term (default: the
    closing lines, as in training). F1 is per availability scenario
    (`opp_avail.scenario_features`) and is not part of the shipped layer."""
    d = pd.Timestamp(day).date()
    dn = int(ST._day_of(d))
    sidx = ST.sidx_of(ST.season_of(d))
    q = pd.concat([pd.DataFrame({"game_id": games["game_id"], "team": games["home"],
                                 "opp": games["away"]}),
                   pd.DataFrame({"game_id": games["game_id"], "team": games["away"],
                                 "opp": games["home"]})], ignore_index=True)
    q["day"], q["sidx"] = dn, sidx
    return opp_defense.build(conn, wf, params=dvp_params, t=t, queries=q, totals=totals)


def attach(rows: pd.DataFrame, tg: pd.DataFrame, pf: pd.DataFrame | None = None
           ) -> pd.DataFrame:
    """rows (game_id, team, EB state columns) + team-game features (+ per player-game features
    `pf` keyed game_id, player_id: F3 / F5) + row-level columns."""
    out = rows.merge(tg, on=["game_id", "team"], how="left")
    if pf is not None and len(pf.columns) > 2:
        out = out.merge(pf, on=["game_id", "player_id"], how="left")
    for c in opp_avail.TEAM_COLS:
        if c in out:
            out[c] = out[c].fillna(0.0)
    extra = [player.row_features(out)]
    if "f1_vac_G" in out:
        extra.append(opp_avail.row_features(out))
    if "f2_dvp_pts_G" in out:
        extra.append(opp_defense.row_features(out))
    if "t3_impl_reb" in out:
        extra.append(tracking.row_features(out))
    if "f4_lpts_s" in out:
        extra.append(shots.row_features(out))
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


EXT_VERSION = 1


def _ext_path(season: str, group: str, frame: pd.DataFrame) -> Path:
    blob = json.dumps({"v": EXT_VERSION, "g": group, "n": int(len(frame)),
                       "first": str(frame["game_date"].min()),
                       "last": str(frame["game_date"].max()),
                       "ids": int(pd.util.hash_pandas_object(
                           frame[["game_id", "player_id"]], index=False).sum() % (2 ** 61))},
                      sort_keys=True)
    return cache_dir() / f"ext_{season}_{group}_{hashlib.sha1(blob.encode()).hexdigest()[:10]}" \
        ".parquet"


def group_table(conn: sqlite3.Connection, frame: pd.DataFrame, group: str, *,
                season: str = "", log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """The raw feature table of an extension group for the rows of `frame`: (game_id, team) for
    F6, (game_id, player_id) for F3 / F4 / F5 (F4 with the xFG model of `season`, fit on the
    shots of seasons < S)."""
    say = log or (lambda m: None)
    t0 = time.time()
    pos = ST.load_positions(conn)
    if group == "F6":
        out = schedule.build(conn)
    elif group == "F3":
        out = tracking.build(conn, frame, pos=pos)
    elif group == "F5":
        out = matchups.build(conn, frame, pos=pos, log=say)
    elif group == "F4":
        out = shots.build(conn, frame, season=season or None, pos=pos, log=say)
    else:
        raise ValueError(group)
    say(f"features: {group} {len(out)} rows ({time.time() - t0:.0f}s)")
    return out


def extend_frame(conn: sqlite3.Connection, frame: pd.DataFrame, groups, *, season: str = "",
                 cache: bool = True, log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """`frame` (build_frame) + the columns of the extension groups in `groups` (F3 / F4 / F5 /
    F6; others are ignored) and their row-level columns. Row order is kept."""
    out = frame
    for g in groups:
        if g not in ("F3", "F4", "F5", "F6"):
            continue
        cols = {"F3": tracking.TEAM_COLS, "F5": matchups.TEAM_COLS,
                "F6": schedule.TEAM_COLS, "F4": shots.TEAM_COLS}[g]
        if all(c in out.columns for c in cols):
            continue
        path = _ext_path(season, g, frame)
        if cache and path.exists():
            tab = pd.read_parquet(path)
        else:
            tab = group_table(conn, frame, g, season=season, log=log)
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                tab.to_parquet(path)
        key = ["game_id", "team"] if g == "F6" else ["game_id", "player_id"]
        tab = tab.drop_duplicates(key)
        n = len(out)
        out = out.merge(tab[key + list(cols)], on=key, how="left")
        assert len(out) == n
        if g == "F3":
            out = pd.concat([out, tracking.row_features(out)], axis=1)
        if g == "F4":
            out = pd.concat([out, shots.row_features(out)], axis=1)
    return out


# ============================================================================== live hook

class LiveExtras:
    """Per-season inputs of the extension groups for the pipeline / live path, loaded once per
    process (seasons S-3..S; every use is strictly before the projected day by construction,
    and older history carries < 1% weight under the groups' half-lives). F4 reads the whole shot
    era (its skill memory is long) with the layer's own xFG model (`xfg`: the
    `ContextModel.meta["xfg"]` JSON; default the xFG model of S fit on seasons < S)."""

    def __init__(self, conn: sqlite3.Connection, season: str, groups, *, pos=None,
                 xfg=None, log: Callable[[str], None] | None = None):
        say = log or (lambda m: None)
        t0 = time.time()
        self.groups = tuple(g for g in groups if g in ("F3", "F4", "F5", "F6"))
        self.season = season
        pos = pos if pos is not None else ST.load_positions(conn)
        first = season_label(int(season[:4]) - 3)
        self.trk = self.mh = self.ph = self.dq = self.sch = self.shots = None
        if "F4" in self.groups:
            m = (shots.XFGModel.from_json(xfg) if xfg is not None
                 else shots.xfg_for_season(conn, season, log=say))
            self.shots = shots.load_history(conn, m, last=season, pos=pos, log=say)
        if "F3" in self.groups:
            self.trk = tracking.load_history(conn, first=first, last=season, pos=pos)
        if "F5" in self.groups:
            self.mh = matchups.History(conn, first=first, last=season, pos=pos)
            self.ph = matchups.pair_history(self.mh)
            self.dq = matchups._expected(self.mh.m)
        if "F6" in self.groups:
            self.sch = schedule.History(conn, since_season=season_label(int(season[:4]) - 1))
        say(f"context extras {'+'.join(self.groups) or '-'} {season} ({time.time() - t0:.0f}s)")

    def day(self, conn: sqlite3.Connection, day, games: pd.DataFrame, rows: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(team-game F6 rows keyed game_id, team; player F3 / F5 rows keyed game_id,
        player_id) for the games of `day`. `rows`: game_id, team, player_id, grp, w (expected
        weight: pre-game minutes x P(available); 0 = out) for every candidate of both teams."""
        dn = int(ST._day_of(day))
        sidx = ST.sidx_of(ST.season_of(pd.Timestamp(day).date()))
        tf = pd.DataFrame({"game_id": pd.concat([games["game_id"], games["game_id"]]),
                           "team": pd.concat([games["home"], games["away"]])}
                          ).reset_index(drop=True)
        if self.sch is not None:
            tf = self.sch.day_features(conn, day, games[["game_id", "home", "away"]])
        r = rows[["game_id", "team", "player_id", "grp", "w"]].copy().reset_index(drop=True)
        r["day"], r["sidx"] = dn, sidx
        pf = r[["game_id", "player_id"]].copy()
        if self.trk is not None:
            f3 = tracking.player_features(self.trk, r[["player_id", "day", "sidx", "grp"]])
            pf = pd.concat([pf, f3], axis=1)
        if self.mh is not None:
            f5 = matchups.features(self.mh, r, ph=self.ph, dq=self.dq)
            pf = pd.concat([pf, f5.drop(columns=["game_id", "player_id"])], axis=1)
        if self.shots is not None:
            opp = {**dict(zip(games["game_id"].astype(str) + "|" + games["home"].astype(str),
                              games["away"])),
                   **dict(zip(games["game_id"].astype(str) + "|" + games["away"].astype(str),
                              games["home"]))}
            q = r[["player_id", "day", "sidx", "grp"]].copy()
            q["opp"] = (r["game_id"].astype(str) + "|" + r["team"].astype(str)).map(opp)
            f4 = shots.player_features(self.shots, q)
            pf = pd.concat([pf, f4], axis=1)
        return tf, pf.drop_duplicates(["game_id", "player_id"])
