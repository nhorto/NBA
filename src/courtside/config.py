"""Paths, time zone, and NBA-day helpers. Environment is read at call time."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
NBA_TZ = ZoneInfo("America/New_York")


def data_dir() -> Path:
    """Data root: $COURTSIDE_DATA or <repo>/var. Not created; see ensure_dirs()."""
    env = os.environ.get("COURTSIDE_DATA")
    return Path(env).expanduser() if env else REPO_ROOT / "var"


def db_path() -> Path:
    return data_dir() / "courtside.db"


def ensure_dirs() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    return d


# Import-time snapshots for convenience; prefer the functions above.
DATA_DIR = data_dir()
DB_PATH = db_path()


def nba_day(dt: datetime) -> date:
    """US/Eastern calendar date of an aware datetime. Naive datetimes are rejected."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("nba_day() requires a timezone-aware datetime")
    return dt.astimezone(NBA_TZ).date()


def today_nba_day() -> date:
    return nba_day(datetime.now(timezone.utc))


def yesterday_nba_day() -> date:
    return today_nba_day() - timedelta(days=1)
