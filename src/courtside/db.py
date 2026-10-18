"""SQLite access: connect(), append-only migrations, small upsert helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

MIGRATIONS: list[str] = [
    # 1: v1 tables
    """
    CREATE TABLE games (
        game_id TEXT PRIMARY KEY,
        espn_id TEXT,
        season TEXT,
        nba_day TEXT,
        tipoff_utc TEXT,
        home TEXT,
        away TEXT,
        status TEXT NOT NULL DEFAULT 'scheduled'
            CHECK (status IN ('scheduled','live','final','postponed')),
        home_score INTEGER,
        away_score INTEGER,
        updated_at TEXT
    );
    CREATE INDEX idx_games_nba_day ON games(nba_day);

    CREATE TABLE odds_snapshots (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        source TEXT NOT NULL,
        provider TEXT,
        home_spread REAL,
        total REAL,
        home_ml REAL,
        away_ml REAL,
        raw_path TEXT
    );
    CREATE INDEX idx_odds_game_captured ON odds_snapshots(game_id, captured_at);

    CREATE TABLE injury_snapshots (
        id INTEGER PRIMARY KEY,
        captured_at TEXT NOT NULL,
        source TEXT NOT NULL,
        team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        player_id TEXT,
        status TEXT,
        reason TEXT,
        raw_path TEXT
    );
    CREATE INDEX idx_injury_captured ON injury_snapshots(captured_at);

    CREATE TABLE collect_runs (
        id INTEGER PRIMARY KEY,
        job TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        ok INTEGER,
        detail TEXT
    );
    """,
    # 2: historical game logs (stats.nba.com)
    """
    CREATE TABLE team_games (
        game_id TEXT NOT NULL, team TEXT NOT NULL, season TEXT, season_type TEXT,
        game_date TEXT, opp TEXT, is_home INTEGER, wl TEXT, min REAL,
        fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER, fta INTEGER,
        oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER,
        tov INTEGER, pf INTEGER, pts INTEGER, plus_minus REAL,
        PRIMARY KEY (game_id, team)
    );
    CREATE INDEX idx_team_games_season ON team_games(season);
    CREATE INDEX idx_team_games_date ON team_games(game_date);
    CREATE INDEX idx_team_games_team_date ON team_games(team, game_date);

    CREATE TABLE player_games (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL, player_name TEXT, team TEXT,
        season TEXT, season_type TEXT, game_date TEXT, min REAL,
        fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER, fta INTEGER,
        oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER,
        tov INTEGER, pf INTEGER, pts INTEGER, plus_minus REAL,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE INDEX idx_player_games_season ON player_games(season);
    CREATE INDEX idx_player_games_date ON player_games(game_date);
    CREATE INDEX idx_player_games_player_date ON player_games(player_id, game_date);

    CREATE TABLE team_games_adv (
        game_id TEXT NOT NULL, team TEXT NOT NULL,
        off_rating REAL, def_rating REAL, net_rating REAL, pace REAL, poss REAL,
        efg_pct REAL, ts_pct REAL, tov_pct REAL, oreb_pct REAL, dreb_pct REAL,
        ast_pct REAL, pie REAL,
        PRIMARY KEY (game_id, team)
    );

    CREATE TABLE player_games_adv (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL,
        off_rating REAL, def_rating REAL, net_rating REAL, usg_pct REAL, ts_pct REAL,
        efg_pct REAL, ast_pct REAL, reb_pct REAL, pace REAL, poss REAL, pie REAL,
        PRIMARY KEY (game_id, player_id)
    );
    """,
]

GAME_COLUMNS = ("espn_id", "season", "nba_day", "tipoff_utc", "home", "away",
                "status", "home_score", "away_score")
ODDS_COLUMNS = ("game_id", "captured_at", "source", "provider", "home_spread",
                "total", "home_ml", "away_ml", "raw_path")
INJURY_COLUMNS = ("captured_at", "source", "team", "player_name", "player_id",
                  "status", "reason", "raw_path")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; return the current schema version."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version "
                 "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    current = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
    for version, sql in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        conn.executescript("BEGIN;\n" + sql + "\nCOMMIT;")
        conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                     (version, _now()))
        conn.commit()
    return max(current, len(MIGRATIONS))


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    if path is None:
        config.ensure_dirs()
        path = config.db_path()
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _check(fields: dict, allowed: tuple[str, ...], extra: tuple[str, ...] = ()) -> None:
    bad = set(fields) - set(allowed) - set(extra)
    if bad:
        raise TypeError(f"unknown column(s): {sorted(bad)}")


def upsert_game(conn: sqlite3.Connection, **fields) -> None:
    """Insert or update a game by game_id; None-valued fields never overwrite existing data."""
    game_id = fields.pop("game_id", None)
    if not game_id:
        raise ValueError("game_id is required")
    _check(fields, GAME_COLUMNS, ("updated_at",))
    fields.pop("updated_at", None)
    given = {k: v for k, v in fields.items() if v is not None}
    cols = ["game_id", *given, "updated_at"]
    vals = [game_id, *given.values(), _now()]
    sets = ", ".join(f"{c}=excluded.{c}" for c in [*given, "updated_at"])
    conn.execute(
        f"INSERT INTO games ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT(game_id) DO UPDATE SET {sets}", vals)
    conn.commit()


def insert_odds_snapshot(conn: sqlite3.Connection, **fields) -> int:
    _check(fields, ODDS_COLUMNS)
    cols = list(fields)
    cur = conn.execute(
        f"INSERT INTO odds_snapshots ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        list(fields.values()))
    conn.commit()
    return cur.lastrowid


def insert_injury_snapshots(conn: sqlite3.Connection, rows: list[dict]) -> int:
    for r in rows:
        _check(r, INJURY_COLUMNS)
    conn.executemany(
        f"INSERT INTO injury_snapshots ({', '.join(INJURY_COLUMNS)}) "
        f"VALUES ({', '.join(':' + c for c in INJURY_COLUMNS)})",
        [{c: r.get(c) for c in INJURY_COLUMNS} for r in rows])
    conn.commit()
    return len(rows)


def upsert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE dicts into `table`. Keys must be existing columns; missing keys -> NULL."""
    if not rows:
        return 0
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not cols:
        raise ValueError(f"unknown table {table!r}")
    for r in rows:
        _check(r, tuple(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)})",
        [{c: r.get(c) for c in cols} for r in rows])
    conn.commit()
    return len(rows)


def start_run(conn: sqlite3.Connection, job: str) -> int:
    cur = conn.execute("INSERT INTO collect_runs (job, started_at) VALUES (?, ?)", (job, _now()))
    conn.commit()
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, ok: bool, detail: str | None = None) -> None:
    conn.execute("UPDATE collect_runs SET finished_at=?, ok=?, detail=? WHERE id=?",
                 (_now(), int(ok), detail, run_id))
    conn.commit()
