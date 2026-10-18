"""Schedule-derived pregame features: rest, back-to-backs, density, travel, time zones.

Leakage-safe by construction: every feature of a game uses only the dates and locations of
that team's *earlier* games plus the game's own (pre-announced) date and site. No results are
read. Locations come from model.sites (era-aware home arenas, bubble, neutral venues).
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import sites

REST_CAP = 4          # days of rest are capped here (season openers get the cap)
EARTH_KM = 6371.0088

TEAM_FEATURES = ("rest", "b2b", "three_in_four", "four_in_six", "games_last7",
                 "travel_km", "tz_change", "season_opener")


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


@lru_cache(maxsize=None)
def utc_offset_hours(tz: str, day: str) -> float:
    """UTC offset of `tz` at local noon on `day` (YYYY-MM-DD), DST-aware."""
    dt = datetime.fromisoformat(day + "T12:00:00").replace(tzinfo=ZoneInfo(tz))
    return dt.utcoffset().total_seconds() / 3600.0


def team_games_long(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home and away perspective) with site info, sorted by team and date."""
    base = games[["game_id", "season", "game_date", "site_lat", "site_lon", "site_tz"]]
    h = base.assign(team=games["home"].values, raw=games["home_raw"].values, side="home")
    a = base.assign(team=games["away"].values, raw=games["away_raw"].values, side="away")
    long = pd.concat([h, a], ignore_index=True)
    return long.sort_values(["team", "game_date", "game_id"], kind="mergesort").reset_index(drop=True)


def team_schedule_features(games: pd.DataFrame, rest_cap: int = REST_CAP) -> pd.DataFrame:
    """Per (game_id, side) schedule features. `games` needs the load_games() columns."""
    long = team_games_long(games)
    day = (long["game_date"].values.astype("datetime64[D]").astype(np.int64))
    long["_day"] = day
    out = []
    for _, t in long.groupby("team", sort=False):
        t = t.copy()
        d = t["_day"].to_numpy()
        new_season = (t["season"] != t["season"].shift()).to_numpy()
        gap = np.diff(d, prepend=d[0])
        rest = np.where(new_season, rest_cap, np.clip(gap - 1, 0, rest_cap))
        if (gap[~new_season] <= 0).any():
            raise ValueError(f"team {t['team'].iat[0]} has two games on one day")
        # counts of this team's games in day windows, from sorted day numbers
        def n_in(lo_off, hi_off):  # games with day in [d+lo_off, d+hi_off]
            return (np.searchsorted(d, d + hi_off, side="right")
                    - np.searchsorted(d, d + lo_off, side="left"))
        t["rest"] = rest
        t["b2b"] = ((rest == 0) & ~new_season).astype(int)
        t["three_in_four"] = (n_in(-3, 0) >= 3).astype(int)
        t["four_in_six"] = (n_in(-5, 0) >= 4).astype(int)
        t["games_last7"] = n_in(-7, -1)
        # previous location: last game's site, or the team's home arena at a season start
        plat = t["site_lat"].shift().to_numpy(copy=True)
        plon = t["site_lon"].shift().to_numpy(copy=True)
        ptz = t["site_tz"].shift().to_numpy(dtype=object, copy=True)
        for i in np.flatnonzero(new_season):
            hs = sites.home_site(t["raw"].iat[i], t["season"].iat[i])
            plat[i], plon[i], ptz[i] = hs.lat, hs.lon, hs.tz
        t["travel_km"] = haversine_km(plat, plon, t["site_lat"], t["site_lon"])
        ds = t["game_date"].dt.strftime("%Y-%m-%d").to_numpy()
        now = np.array([utc_offset_hours(z, x) for z, x in zip(t["site_tz"], ds)])
        prev = np.array([utc_offset_hours(z, x) for z, x in zip(ptz, ds)])
        t["tz_change"] = now - prev     # + = moved east (clock jumps forward)
        t["season_opener"] = new_season.astype(int)
        out.append(t)
    res = pd.concat(out)
    return res[["game_id", "side", *TEAM_FEATURES]]


def schedule_features(games: pd.DataFrame, rest_cap: int = REST_CAP) -> pd.DataFrame:
    """Wide per-game features: home_<f>, away_<f> for f in TEAM_FEATURES, plus rest_diff.
    Returned in the same row order as `games` (indexed like `games`)."""
    tf = team_schedule_features(games, rest_cap)
    wide = {}
    for side in ("home", "away"):
        s = tf[tf["side"] == side].set_index("game_id")[list(TEAM_FEATURES)]
        wide[side] = s.add_prefix(f"{side}_")
    w = wide["home"].join(wide["away"], how="inner")
    w = w.reindex(games["game_id"].values)
    w.index = games.index
    w["rest_diff"] = w["home_rest"] - w["away_rest"]
    return w


def add_schedule_features(games: pd.DataFrame, rest_cap: int = REST_CAP) -> pd.DataFrame:
    return pd.concat([games, schedule_features(games, rest_cap)], axis=1)
