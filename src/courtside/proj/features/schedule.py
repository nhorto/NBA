"""F6: schedule context per team-game (docs/PLAYERS.md Phase 2F).

Home/away, rest, back-to-backs, 3-in-4 / 4-in-6, games in the last 7 days, travel distance and
time-zone change since the previous game, altitude of the venue, season stage, for the team and
for its opponent. Built on the game model's schedule features (`model.schedule`, era-aware sites
from `model.sites`): every value uses only the dates and sites of the team's EARLIER games plus
the game's own (pre-announced) date and site; no result is read, so the table is strictly
pre-tip by construction.

Columns (per (game_id, team): the team the features are FOR)
  f6_home        1 home, 0 away, 0.5 neutral site
  f6_rest        days of rest (capped at 4; season openers 4)
  f6_b2b, f6_3in4, f6_4in6, f6_g7      density
  f6_travel      km since the previous game's site / 1000
  f6_tz          time-zone change (hours, + = moved east)
  f6_alt         venue altitude in km (Denver 1.61, Salt Lake City 1.29, Mexico City 2.24, else 0)
  f6_alt_vis     f6_alt when the team is the visitor (the altitude effect is on visitors)
  f6_stage       days since the season's first game / 100
  f6_o_rest, f6_o_b2b, f6_o_3in4, f6_o_travel, f6_o_tz     the opponent's
  f6_rest_diff   own minus opponent rest
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

ALTITUDE_KM = {"DEN": 1.609, "UTA": 1.288}
MEXICO_ALT_KM = 2.24
OWN = ("rest", "b2b", "three_in_four", "four_in_six", "games_last7", "travel_km", "tz_change")
RENAME = {"rest": "f6_rest", "b2b": "f6_b2b", "three_in_four": "f6_3in4",
          "four_in_six": "f6_4in6", "games_last7": "f6_g7", "travel_km": "f6_travel",
          "tz_change": "f6_tz"}
OPP = ("rest", "b2b", "three_in_four", "travel_km", "tz_change")
TEAM_COLS = (("f6_home", "f6_rest", "f6_b2b", "f6_3in4", "f6_4in6", "f6_g7", "f6_travel",
              "f6_tz", "f6_alt", "f6_alt_vis", "f6_stage")
             + tuple(f"f6_o_{RENAME[c][3:]}" for c in OPP) + ("f6_rest_diff",))
NEUTRAL = {"f6_home": 0.5, "f6_rest": 1.0, "f6_b2b": 0.0, "f6_3in4": 0.0, "f6_4in6": 0.0,
           "f6_g7": 3.0, "f6_travel": 0.8, "f6_tz": 0.0, "f6_alt": 0.0, "f6_alt_vis": 0.0,
           "f6_o_rest": 1.0, "f6_o_b2b": 0.0, "f6_o_3in4": 0.0, "f6_o_travel": 0.8,
           "f6_o_tz": 0.0, "f6_rest_diff": 0.0}


def _altitude(games: pd.DataFrame) -> np.ndarray:
    site = games["site"].astype(str).to_numpy()
    home = games["home"].astype(str).to_numpy()
    neutral = games["is_neutral"].astype(bool).to_numpy()
    alt = np.array([ALTITUDE_KM.get(h, 0.0) for h in home])
    alt = np.where(neutral, 0.0, alt)
    return np.where(np.char.startswith(site.astype(str), "Mexico City"), MEXICO_ALT_KM, alt)


def from_games(games: pd.DataFrame) -> pd.DataFrame:
    """The F6 table for every game of `games` (`model.dataset.load_games` columns, optionally
    with scheduled games carrying placeholder results). One row per (game_id, team) with `team`
    = the tricode as recorded (team_games / player_games)."""
    from ...model import schedule as SCH
    if not len(games):
        return pd.DataFrame(columns=["game_id", "team", *TEAM_COLS])
    g = games.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    tf = SCH.team_schedule_features(g)
    alt = pd.Series(_altitude(g), index=g["game_id"].to_numpy())
    first = g.groupby("season")["game_date"].transform("min")
    stage = pd.Series(((g["game_date"] - first).dt.days / 100.0).to_numpy(),
                      index=g["game_id"].to_numpy())
    neutral = pd.Series(g["is_neutral"].astype(bool).to_numpy(), index=g["game_id"].to_numpy())
    raw = {"home": pd.Series(g["home_raw"].to_numpy(), index=g["game_id"].to_numpy()),
           "away": pd.Series(g["away_raw"].to_numpy(), index=g["game_id"].to_numpy())}
    sides = {}
    for side in ("home", "away"):
        s = tf[tf["side"] == side].set_index("game_id")
        sides[side] = s
    out = []
    for side, other in (("home", "away"), ("away", "home")):
        me, op = sides[side], sides[other].reindex(sides[side].index)
        gid = me.index.to_numpy()
        d = pd.DataFrame({"game_id": gid, "team": raw[side].reindex(gid).to_numpy()})
        nt = neutral.reindex(gid).to_numpy()
        d["f6_home"] = np.where(nt, 0.5, 1.0 if side == "home" else 0.0)
        for c in OWN:
            d[RENAME[c]] = me[c].to_numpy(float)
        d["f6_travel"] = d["f6_travel"] / 1000.0
        a = alt.reindex(gid).to_numpy(float)
        d["f6_alt"] = a
        d["f6_alt_vis"] = a if side == "away" else np.where(nt, a, 0.0)
        d["f6_stage"] = stage.reindex(gid).to_numpy(float)
        for c in OPP:
            v = op[c].to_numpy(float)
            d[f"f6_o_{RENAME[c][3:]}"] = v / 1000.0 if c == "travel_km" else v
        d["f6_rest_diff"] = d["f6_rest"] - d["f6_o_rest"]
        out.append(d)
    return pd.concat(out, ignore_index=True)[["game_id", "team", *TEAM_COLS]]


def build(conn: sqlite3.Connection) -> pd.DataFrame:
    """F6 for every completed game in team_games (history and backtests)."""
    from ...model import dataset
    return from_games(dataset.load_games(conn))


class History:
    """Completed games with sites (`model.dataset.load_games`), loaded once per process.
    `day_features` computes the day's games from the history strictly BEFORE the day plus the
    day's own games (date and site only): identical to the table for completed games, and the
    same code path live (games not played yet) and in backtests."""

    def __init__(self, conn: sqlite3.Connection, since_season: str | None = None):
        from ...model import dataset
        g = dataset.load_games(conn)
        if since_season is not None:
            g = g[g["season"] >= since_season]
        self.g = g.reset_index(drop=True)

    def _day_rows(self, day, games: pd.DataFrame) -> pd.DataFrame:
        from ... import teams
        from ...model import sites
        from ..state import season_of
        d = pd.Timestamp(day).normalize()
        season = season_of(d.date())
        have = self.g.set_index("game_id")
        rows = []
        for gid, h, a in zip(games["game_id"], games["home"], games["away"]):
            if gid in have.index:
                rows.append(have.loc[[gid]].reset_index())
                continue
            site = sites.flagged_neutral_site(d.date().isoformat()) or \
                sites.curated_neutral_site(d.date().isoformat(), h, a)
            neutral = site is not None
            site = site or sites.home_site(h, season)
            st = "Playoffs" if str(gid).startswith("004") else (
                "PlayIn" if str(gid).startswith("005") else "Regular Season")
            rows.append(pd.DataFrame([{
                "game_id": gid, "season": season, "season_type": st, "game_date": d,
                "home": teams.franchise(h, season), "away": teams.franchise(a, season),
                "home_raw": h, "away_raw": a, "is_neutral": neutral, "site": site.name,
                "site_lat": site.lat, "site_lon": site.lon, "site_tz": site.tz}]))
        return pd.concat(rows, ignore_index=True)

    def day_features(self, conn: sqlite3.Connection, day, games: pd.DataFrame) -> pd.DataFrame:
        """(game_id, team) F6 rows for `games` (game_id, home, away: tricodes as recorded) on
        `day`, from games strictly before the day plus the day's games."""
        d = pd.Timestamp(day).normalize()
        hist = self.g[self.g["game_date"] < d]
        cols = ["game_id", "season", "season_type", "game_date", "home", "away", "home_raw",
                "away_raw", "is_neutral", "site", "site_lat", "site_lon", "site_tz"]
        today = self._day_rows(d, games)[cols]
        f = from_games(pd.concat([hist[cols], today], ignore_index=True))
        keys = pd.concat([pd.DataFrame({"game_id": games["game_id"], "team": games["home"]}),
                          pd.DataFrame({"game_id": games["game_id"], "team": games["away"]})],
                         ignore_index=True)
        return keys.merge(f, on=["game_id", "team"], how="left")[["game_id", "team",
                                                                   *TEAM_COLS]]


def columns(stat: str) -> list[str]:
    return list(TEAM_COLS)


def monotone(stat: str) -> dict:
    return {}
