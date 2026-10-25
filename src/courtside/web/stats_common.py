"""Shared helpers for the stats route modules (stats_games / stats_teams / stats_players /
stats_leaders). Owned by the foundation; page modules import from here.

    from .stats_common import open_db, season_args

    with open_db(store) as conn:              # read-only sqlite3.Connection (Row rows) or None
        ...
    season, window = season_args(request, today())   # ?season= validated against the window

Heavy aggregates: wrap them in `courtside.stats.db.cached(key, fn, conn)` so a warm render
stays under 50 ms (the cache drops everything when courtside.db changes).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Iterator

from fastapi import Request

from ..stats import db as sdb
from ..stats import seasons as S


@contextmanager
def open_db(store) -> Iterator[sqlite3.Connection | None]:
    """Read-only connection to the store's courtside.db (None when there is no database)."""
    with sdb.ro(store.db_path) as conn:
        yield conn


def season_args(request: Request, today: date) -> tuple[str, tuple[str, ...]]:
    """(season, window): `?season=` if it is in the 3-season window, else the current one."""
    window = S.seasons(today)
    return S.pick(request.query_params.get("season"), allowed=window), window


def season_type_arg(request: Request) -> str:
    """`?type=regular|playoffs` -> 'Regular Season' / 'Playoffs' (default regular)."""
    return S.season_type(request.query_params.get("type"))
