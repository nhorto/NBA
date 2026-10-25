"""Seasons for the stats pages: "2026-27" runs October 2026 to June 2027.

July to September (the off-season) still counts as the season that just ended, so the pages
keep showing real numbers until the new season tips off. The site covers the current season
plus the two before it (owner decision, docs/STATS_SITE.md).

Season-type values in the tables: 'Regular Season', 'Playoffs' (and 'PlayIn', which the pages
fold into neither unless asked).
"""

from __future__ import annotations

import re
from datetime import date

REGULAR = "Regular Season"
PLAYOFFS = "Playoffs"
PLAYIN = "PlayIn"
# URL value -> table value (e.g. ?type=playoffs)
SEASON_TYPES = {"regular": REGULAR, "playoffs": PLAYOFFS}
TYPE_LABELS = {REGULAR: "Regular season", PLAYOFFS: "Playoffs", PLAYIN: "Play-in"}
N_SEASONS = 3
_RE = re.compile(r"^(\d{4})-(\d{2})$")


def start_year(season: str) -> int:
    """'2026-27' -> 2026. Raises ValueError for anything else."""
    m = _RE.match(season or "")
    if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
        raise ValueError(f"not a season: {season!r}")
    return int(m.group(1))


def is_season(season: str | None) -> bool:
    try:
        start_year(season or "")
        return True
    except ValueError:
        return False


def from_start_year(year: int) -> str:
    """2026 -> '2026-27'."""
    return f"{year}-{(year + 1) % 100:02d}"


def season_for(d: date) -> str:
    """The season a date belongs to (Oct-Dec: the one starting that year; Jan-Sep: the one
    that started the year before)."""
    return from_start_year(d.year if d.month >= 10 else d.year - 1)


def current_season(today: date | None = None) -> str:
    return season_for(today or date.today())


def seasons(today: date | None = None, n: int = N_SEASONS) -> tuple[str, ...]:
    """The last `n` seasons, newest first: ('2026-27', '2025-26', '2024-25')."""
    y = start_year(current_season(today))
    return tuple(from_start_year(y - i) for i in range(n))


# Snapshot at import time. A long-running server crossing October 1 keeps the old window
# until restart; route code should prefer `seasons(today())`.
SEASONS = seasons()


def prev_season(season: str) -> str:
    return from_start_year(start_year(season) - 1)


def label(season: str) -> str:
    """'2026-27' (the canonical label; kept as a function so pages never hand-format)."""
    return from_start_year(start_year(season))


def short_label(season: str) -> str:
    """'2026-27' -> '26-27'."""
    return label(season)[2:]


def date_range(season: str) -> tuple[date, date]:
    """(Oct 1, Sep 30): every game of a season, playoffs included, falls inside."""
    y = start_year(season)
    return date(y, 10, 1), date(y + 1, 9, 30)


def pick(value: str | None, today: date | None = None, allowed: tuple[str, ...] | None = None
         ) -> str:
    """A `?season=` value if it is one of `allowed` (default: the 3-season window), else the
    current season."""
    allowed = allowed or seasons(today)
    return value if value in allowed else allowed[0]


def season_type(value: str | None, default: str = REGULAR) -> str:
    """`?type=` value ('regular' / 'playoffs', or the table value itself) -> table value."""
    if value in SEASON_TYPES:
        return SEASON_TYPES[value]
    if value in (REGULAR, PLAYOFFS, PLAYIN):
        return value
    return default


def type_label(season_type_value: str) -> str:
    return TYPE_LABELS.get(season_type_value, season_type_value or "")
