"""Where each game was played: home arenas (era-aware), the 2020 bubble, neutral sites.

Used for home-court advantage (neutral games get none) and for travel/time-zone features.

Approximations (documented in docs/MODEL.md):
- Home games are placed at the franchise's *current* arena, except for the explicit overrides in
  `_HISTORICAL_HOME` (relocated franchises and temporary homes). Moves within a metro area
  (e.g. Oakland -> San Francisco, Newark/East Rutherford) are ignored (< 20 km).
- Neutral games before 2024-25 come from the curated `_NEUTRAL_GAMES` list below; each entry was
  verified to match a game in `team_games`. From 2024-25 on, stats.nba.com marks neutral games
  with "@" on both sides (is_home = 0 for both teams), and those are detected automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .. import teams


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    tz: str


LONDON = Site("London (O2 Arena)", 51.5030, 0.0032, "Europe/London")
PARIS = Site("Paris (Accor Arena)", 48.8386, 2.3786, "Europe/Paris")
BERLIN = Site("Berlin (Uber Arena)", 52.5053, 13.4436, "Europe/Berlin")
MEXICO_CITY = Site("Mexico City (Arena CDMX)", 19.4886, -99.1837, "America/Mexico_City")
LAS_VEGAS = Site("Las Vegas (T-Mobile Arena)", 36.1029, -115.1784, "America/Los_Angeles")
BUBBLE = Site("Orlando bubble (ESPN Wide World of Sports)", 28.3375, -81.5562,
              "America/New_York")

# Regular-season games at neutral sites before stats.nba.com started flagging them (2024-25).
# (date, team_a, team_b) with historical tricodes as stored in team_games.
_NEUTRAL_GAMES: dict[tuple[str, frozenset[str]], Site] = {
    (d, frozenset((a, b))): s for d, a, b, s in [
        ("2011-03-04", "NJN", "TOR", LONDON), ("2011-03-05", "NJN", "TOR", LONDON),
        ("2013-01-17", "DET", "NYK", LONDON), ("2014-01-16", "BKN", "ATL", LONDON),
        ("2015-01-15", "NYK", "MIL", LONDON), ("2016-01-14", "ORL", "TOR", LONDON),
        ("2017-01-12", "IND", "DEN", LONDON), ("2018-01-11", "PHI", "BOS", LONDON),
        ("2019-01-17", "NYK", "WAS", LONDON),
        ("2020-01-24", "CHA", "MIL", PARIS), ("2023-01-19", "CHI", "DET", PARIS),
        ("2024-01-11", "BKN", "CLE", PARIS),
        ("2014-11-12", "HOU", "MIN", MEXICO_CITY), ("2015-12-03", "BOS", "SAC", MEXICO_CITY),
        ("2017-01-12", "PHX", "DAL", MEXICO_CITY), ("2017-01-14", "PHX", "SAS", MEXICO_CITY),
        ("2017-12-07", "BKN", "OKC", MEXICO_CITY), ("2017-12-09", "BKN", "MIA", MEXICO_CITY),
        ("2018-12-13", "ORL", "CHI", MEXICO_CITY), ("2018-12-15", "ORL", "UTA", MEXICO_CITY),
        ("2019-12-12", "DAL", "DET", MEXICO_CITY), ("2019-12-14", "PHX", "SAS", MEXICO_CITY),
        ("2022-12-17", "MIA", "SAS", MEXICO_CITY), ("2023-11-09", "ATL", "ORL", MEXICO_CITY),
        # In-Season Tournament semifinals (the final does not count as a regular-season game).
        ("2023-12-07", "LAL", "NOP", LAS_VEGAS), ("2023-12-07", "IND", "MIL", LAS_VEGAS),
    ]
}

# Known venues for the auto-flagged (both-"@") games from 2024-25 on, for travel features.
# Unknown dates fall back to the nominal home team's arena.
_FLAGGED_NEUTRAL_VENUES: dict[str, Site] = {
    "2024-11-02": MEXICO_CITY, "2025-11-01": MEXICO_CITY,
    "2025-01-23": PARIS, "2025-01-25": PARIS,
    "2024-12-14": LAS_VEGAS, "2025-12-13": LAS_VEGAS,
    "2026-01-15": BERLIN, "2026-01-18": LONDON,
}

# (historical tricode, first season start year, last season start year) -> Site
_HISTORICAL_HOME: list[tuple[str, int, int, Site]] = [
    ("SEA", 1967, 2007, Site("Seattle (KeyArena)", 47.6221, -122.3540, "America/Los_Angeles")),
    ("VAN", 1995, 2000, Site("Vancouver (GM Place)", 49.2778, -123.1089, "America/Vancouver")),
    ("CHH", 1988, 2001, Site("Charlotte Coliseum", 35.2034, -80.9118, "America/New_York")),
    # Post-Katrina: the Hornets played most 2005-07 home games in Oklahoma City.
    ("NOK", 2005, 2006, Site("Oklahoma City (Ford Center)", 35.4634, -97.5151,
                             "America/Chicago")),
    ("NJN", 1977, 2011, Site("New Jersey (Meadowlands/Newark)", 40.7700, -74.1200,
                             "America/New_York")),
    # COVID season: the Raptors played 2020-21 home games in Tampa.
    ("TOR", 2020, 2020, Site("Tampa (Amalie Arena)", 27.9427, -82.4519, "America/New_York")),
]

BUBBLE_START = date(2020, 7, 30)


def home_site(tricode: str, season: str) -> Site:
    """Home arena of a (historical) tricode in a given season."""
    year = teams.season_start_year(season)
    code = tricode.upper()
    for tri, a, b, site in _HISTORICAL_HOME:
        if code == tri and a <= year <= b:
            return site
    t = teams.by_tricode(teams.franchise(code, season))
    return Site(t.arena, t.lat, t.lon, t.tz)


def is_bubble(season: str, game_date: date | str) -> bool:
    """2019-20 restart in the Orlando bubble: seeding games, play-in and playoffs."""
    if season != "2019-20":
        return False
    d = date.fromisoformat(game_date) if isinstance(game_date, str) else game_date
    return d >= BUBBLE_START


def curated_neutral_site(game_date: str, team_a: str, team_b: str) -> Site | None:
    return _NEUTRAL_GAMES.get((game_date, frozenset((team_a, team_b))))


def flagged_neutral_site(game_date: str) -> Site | None:
    return _FLAGGED_NEUTRAL_VENUES.get(game_date)


def curated_entries() -> list[tuple[str, frozenset[str]]]:
    return list(_NEUTRAL_GAMES)
