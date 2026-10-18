"""Shared helpers for the trends engine: seasons, read-only DB access, leak-free loaders.

Every loader takes ``before`` (a ``date``) and filters ``game_date < before`` *in SQL*, so nothing
from the as-of day or later can reach the analytics. Loaders are filtered to a season window (the
current and prior season) so a call never scans full history.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from courtside import teams

REGULAR = ("Regular Season",)


# --- seasons ---------------------------------------------------------------------------------
def season_of(d: date) -> str:
    """NBA season label containing date ``d`` ('2026-02-15' -> '2025-26'). July is the cut."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def prior_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def as_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def franchise_key(code: str, season: str) -> str:
    """Stable franchise id across relocations/renames (SEA->OKC, NJN->BKN, ...)."""
    try:
        return teams.franchise(code, season)
    except KeyError:
        return code


def connect_ro(path: str | Path | None = None) -> sqlite3.Connection:
    """Read-only connection (never runs migrations; safe while other jobs write the DB)."""
    if path is None:
        from courtside import config
        path = config.db_path()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


def _ph(items: Iterable[Any]) -> str:
    return ",".join("?" for _ in items)


# --- loaders ---------------------------------------------------------------------------------
def load_team_games(conn: sqlite3.Connection, seasons: Iterable[str], before: date,
                    season_types: tuple[str, ...] = REGULAR) -> pd.DataFrame:
    """One row per team-game, with the opponent's points and possessions joined.

    Columns: game_id, team, opp, season, game_date (Timestamp), h (+1 home / -1 road / 0 neutral),
    win, pts, opp_pts, poss (mean of both teams' possessions), ortg, drtg, net, margin, fr, opp_fr.
    Possessions come from ``team_games_adv``; when missing they are estimated from the box.
    """
    seasons = list(seasons)
    sql = f"""
        SELECT t.game_id, t.team, t.opp, t.season, t.game_date, t.is_home, t.wl, t.pts, t.min,
               t.fga, t.oreb, t.tov, t.fta, a.poss
        FROM team_games t LEFT JOIN team_games_adv a ON a.game_id = t.game_id AND a.team = t.team
        WHERE t.season IN ({_ph(seasons)}) AND t.season_type IN ({_ph(season_types)})
          AND t.game_date < ?
    """
    df = pd.read_sql_query(sql, conn, params=[*seasons, *season_types, before.isoformat()])
    if df.empty:
        return _empty_team_frame()
    est = df.fga - df.oreb + df.tov + 0.44 * df.fta
    df["poss"] = df["poss"].where(df["poss"].notna() & (df["poss"] > 0), est)
    # drop junk rows (e.g. the cancelled 2013 BOS-IND game: 0 minutes, 0 points)
    df = df[(df["min"].fillna(0) > 0) & (df["poss"] > 0)]
    opp = df[["game_id", "team", "pts", "poss", "is_home"]].rename(
        columns={"team": "opp", "pts": "opp_pts", "poss": "opp_poss", "is_home": "opp_home"})
    df = df.merge(opp, on=["game_id", "opp"], how="inner")
    df["poss"] = (df["poss"] + df["opp_poss"]) / 2.0
    # stats.nba.com marks neutral sites (2024-25+) with both rows away.
    df["h"] = np.where(df.is_home == 1, 1, np.where(df.opp_home == 1, -1, 0))
    df["win"] = (df.pts > df.opp_pts).astype(int)
    df["margin"] = df.pts - df.opp_pts
    df["ortg"] = 100.0 * df.pts / df.poss
    df["drtg"] = 100.0 * df.opp_pts / df.poss
    df["net"] = df.ortg - df.drtg
    df["game_date"] = pd.to_datetime(df.game_date)
    df["fr"] = [franchise_key(t, s) for t, s in zip(df.team, df.season)]
    df["opp_fr"] = [franchise_key(t, s) for t, s in zip(df.opp, df.season)]
    cols = ["game_id", "team", "opp", "season", "game_date", "h", "win", "pts", "opp_pts",
            "poss", "ortg", "drtg", "net", "margin", "fr", "opp_fr"]
    return df[cols].sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def _empty_team_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in [
        ("game_id", object), ("team", object), ("opp", object), ("season", object),
        ("game_date", "datetime64[ns]"), ("h", int), ("win", int), ("pts", float),
        ("opp_pts", float), ("poss", float), ("ortg", float), ("drtg", float), ("net", float),
        ("margin", float), ("fr", object), ("opp_fr", object)]})


PLAYER_COLS = ("game_id", "player_id", "player_name", "team", "season", "game_date", "min",
               "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast",
               "stl", "blk", "tov", "pf", "pts", "plus_minus")


def load_player_games(conn: sqlite3.Connection, seasons: Iterable[str], before: date,
                      season_types: tuple[str, ...] = REGULAR) -> pd.DataFrame:
    """One row per player appearance (``player_games`` has no DNP rows), with usage joined."""
    seasons = list(seasons)
    cols = ", ".join(f"p.{c}" for c in PLAYER_COLS)
    sql = f"""
        SELECT {cols}, a.usg_pct
        FROM player_games p
        LEFT JOIN player_games_adv a ON a.game_id = p.game_id AND a.player_id = p.player_id
        WHERE p.season IN ({_ph(seasons)}) AND p.season_type IN ({_ph(season_types)})
          AND p.game_date < ?
    """
    df = pd.read_sql_query(sql, conn, params=[*seasons, *season_types, before.isoformat()])
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in PLAYER_COLS[7:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    df["usg_pct"] = pd.to_numeric(df["usg_pct"], errors="coerce")
    return df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


# --- JSON hygiene ----------------------------------------------------------------------------
def r1(x: Any, nd: int = 1) -> float | None:
    """Round for output; NaN/inf/None -> None. Always a plain Python float."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and dates into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return r1(obj, 3) if math.isfinite(float(obj)) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
