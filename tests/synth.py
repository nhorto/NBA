"""Shared helpers: a migrated in-memory DB filled with a synthetic league (no network)."""

from __future__ import annotations

import sqlite3

from courtside import db
from courtside.trends.synthetic import day_date, make_league  # noqa: F401  (re-exported)


def synth_conn(seasons=None, **kw) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.MIGRATIONS[0])          # games, odds, injuries, runs
    make_league(conn, seasons=seasons or {"2023-24": 40, "2024-25": 40}, **kw)   # migration 2
    conn.executescript(db.MIGRATIONS[2])          # closing_lines
    conn.executescript(db.MIGRATIONS[3])          # game_facts, predictions, briefs
    return conn
