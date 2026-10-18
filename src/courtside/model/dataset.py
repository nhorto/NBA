"""One row per game, built from the per-team `team_games` log, with integrity checks.

`team_games` has two rows per game (one per team) keyed by historical tricode and with no
team_id. `build_games` pairs them, validates the pairing, maps tricodes to current franchises
(teams.franchise) and tags where the game was played (model.sites).
"""

from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

from .. import teams
from . import sites

SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")
# Columns that reveal the result. Models must not use them for the game being predicted.
OUTCOME_COLUMNS = ("home_pts", "away_pts", "margin", "home_win")
_ID_PREFIX = {"Regular Season": "002", "Playoffs": "004", "PlayIn": "005"}


class DataError(ValueError):
    """team_games violates a structural invariant (e.g. not exactly two rows per game)."""


def load_team_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT game_id, team, season, season_type, game_date, opp, is_home, wl, min, pts "
        "FROM team_games", conn)


def load_games(conn: sqlite3.Connection,
               season_types: tuple[str, ...] | None = SEASON_TYPES) -> pd.DataFrame:
    """All completed games, one row each, sorted chronologically.

    Columns: game_id, season, season_type, game_date (datetime64), home, away (current franchise
    tricodes), home_raw, away_raw (tricodes as recorded), home_pts, away_pts, margin
    (home - away), home_win, ot (overtime periods), is_neutral, neutral_reason, site, site_lat,
    site_lon, site_tz. A dict of data anomalies is attached as `df.attrs["anomalies"]`.
    """
    rows = load_team_rows(conn)
    if season_types is not None:
        rows = rows[rows["season_type"].isin(season_types)]
    return build_games(rows)


def build_games(rows: pd.DataFrame) -> pd.DataFrame:
    """Pair team rows into games. Raises DataError on structural problems; softer anomalies are
    reported in `df.attrs["anomalies"]`."""
    an: dict = {}
    rows = rows.copy()

    # 1. Junk: games that never happened but have stat lines (e.g. BOS-IND 2013-04-16, cancelled
    #    after the Boston Marathon bombing) show 0 minutes.
    junk = rows["min"].isna() | (rows["min"] <= 0)
    junk_ids = set(rows.loc[junk, "game_id"])
    partial = [g for g in junk_ids if not junk[rows["game_id"] == g].all()]
    if partial:
        raise DataError(f"games with one 0-minute team row and one real row: {sorted(partial)}")
    an["junk_games_dropped"] = sorted(junk_ids)
    rows = rows[~rows["game_id"].isin(junk_ids)]

    # 2. Exactly two rows per game.
    counts = rows.groupby("game_id").size()
    bad = counts[counts != 2]
    if len(bad):
        raise DataError(f"{len(bad)} games without exactly two team rows, e.g. "
                        f"{bad.head(5).to_dict()}")

    # 3. Home designation: one "vs." row normally; both "@" = neutral (flagged by stats.nba.com
    #    from 2024-25). Two home rows is an error.
    rows["is_home"] = rows["is_home"].fillna(0).astype(int)
    home_n = rows.groupby("game_id")["is_home"].transform("sum")
    if (home_n > 1).any():
        raise DataError(f"games with two home teams: {sorted(set(rows.loc[home_n > 1, 'game_id']))[:5]}")
    rows["flag_neutral"] = home_n == 0
    # Home first; for flagged-neutral games the nominal home team is the alphabetically first.
    rows = rows.sort_values(["game_id", "is_home", "team"], ascending=[True, False, True])
    h = rows.groupby("game_id", sort=True).nth(0).set_index("game_id")
    a = rows.groupby("game_id", sort=True).nth(1).set_index("game_id")
    a = a.loc[h.index]

    # 4. Cross-consistency of the two rows.
    for col in ("season", "season_type", "game_date"):
        mism = h.index[h[col].values != a[col].values]
        if len(mism):
            raise DataError(f"{col} differs between the two rows of {list(mism[:5])}")
    if ((h["opp"].values != a["team"].values) | (a["opp"].values != h["team"].values)).any():
        raise DataError("team/opp fields do not cross-match")
    if (h["pts"].values == a["pts"].values).any():
        raise DataError("tied final score")
    wl_ok = (h["wl"].values == np.where(h["pts"].values > a["pts"].values, "W", "L"))
    an["wl_inconsistent"] = list(h.index[~wl_ok])

    g = pd.DataFrame({
        "game_id": h.index.values,
        "season": h["season"].values,
        "season_type": h["season_type"].values,
        "game_date": pd.to_datetime(h["game_date"].values),
        "home_raw": h["team"].values,
        "away_raw": a["team"].values,
        "home_pts": h["pts"].values.astype(int),
        "away_pts": a["pts"].values.astype(int),
        "min": h["min"].values,
        "flag_neutral": h["flag_neutral"].values,
    })
    g["home"] = [teams.franchise(t, s) for t, s in zip(g["home_raw"], g["season"])]
    g["away"] = [teams.franchise(t, s) for t, s in zip(g["away_raw"], g["season"])]
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    g["ot"] = np.round((g["min"] - 240) / 25).clip(lower=0).astype(int)

    # 5. Sites: bubble, flagged neutral, curated neutral, else the home arena.
    date_str = g["game_date"].dt.strftime("%Y-%m-%d")
    reason, site_objs = [], []
    matched_curated = set()
    for d, s, hr, ar, flagged in zip(date_str, g["season"], g["home_raw"], g["away_raw"],
                                     g["flag_neutral"]):
        cur = sites.curated_neutral_site(d, hr, ar)
        if sites.is_bubble(s, d):
            reason.append("bubble"); site_objs.append(sites.BUBBLE)
        elif flagged:
            reason.append("flagged"); site_objs.append(
                sites.flagged_neutral_site(d) or sites.home_site(hr, s))
        elif cur is not None:
            matched_curated.add((d, frozenset((hr, ar))))
            reason.append("curated"); site_objs.append(cur)
        else:
            reason.append(""); site_objs.append(sites.home_site(hr, s))
    g["neutral_reason"] = reason
    g["is_neutral"] = g["neutral_reason"] != ""
    g["site"] = [x.name for x in site_objs]
    g["site_lat"] = [x.lat for x in site_objs]
    g["site_lon"] = [x.lon for x in site_objs]
    g["site_tz"] = [x.tz for x in site_objs]
    seasons_present = set(g["season"])
    an["curated_neutral_unmatched"] = sorted(
        f"{d} {'-'.join(sorted(p))}" for d, p in sites.curated_entries()
        if (d, p) not in matched_curated
        and _season_of(d) in seasons_present)

    # 6. Softer anomalies worth knowing about.
    an["neutral_counts"] = dict(Counter(r for r in reason if r))
    an["flagged_neutral_games"] = sorted(
        f"{d} {x}-{y}" for d, x, y, f in zip(date_str, g["home_raw"], g["away_raw"],
                                             g["flag_neutral"]) if f)
    prefix_bad = [gid for gid, st in zip(g["game_id"], g["season_type"])
                  if _ID_PREFIX.get(st) and not gid.startswith(_ID_PREFIX[st])]
    an["game_id_prefix_mismatch"] = prefix_bad
    long = pd.concat([g[["game_date", "home"]].rename(columns={"home": "t"}),
                      g[["game_date", "away"]].rename(columns={"away": "t"})])
    dup = long[long.duplicated(keep=False)]
    an["team_twice_same_day"] = sorted({f"{d:%Y-%m-%d} {t}" for d, t in dup.itertuples(index=False)})
    an["ot_games"] = int((g["ot"] > 0).sum())
    an["games_by_season_type"] = (g.groupby(["season", "season_type"]).size()
                                  .rename("n").reset_index()
                                  .pivot(index="season", columns="season_type", values="n")
                                  .fillna(0).astype(int).to_dict(orient="index"))
    an["toronto_tampa_2020_21_home_games"] = int(((g["home_raw"] == "TOR")
                                                  & (g["season"] == "2020-21")).sum())

    g = g.drop(columns=["min", "flag_neutral"])
    g = g.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    cols = ["game_id", "season", "season_type", "game_date", "home", "away", "home_raw",
            "away_raw", "home_pts", "away_pts", "margin", "home_win", "ot", "is_neutral",
            "neutral_reason", "site", "site_lat", "site_lon", "site_tz"]
    g = g[cols]
    g.attrs["anomalies"] = an
    return g


def _season_of(day: str) -> str:
    y, m = int(day[:4]), int(day[5:7])
    start = y if m >= 8 else y - 1
    return f"{start}-{(start + 1) % 100:02d}"
