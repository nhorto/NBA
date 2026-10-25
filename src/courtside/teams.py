"""The 30 current NBA teams: ids, ESPN abbreviations, arena location, time zone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    tricode: str      # NBA tricode (canonical)
    espn_abbr: str    # ESPN site API abbreviation
    team_id: int      # NBA numeric team id
    city: str
    name: str
    arena: str
    lat: float
    lon: float
    tz: str           # IANA zone of the arena


def _t(tricode, espn, n, city, name, arena, lat, lon, tz) -> Team:
    return Team(tricode, espn, 1610612700 + n, city, name, arena, lat, lon, tz)


ET, CT, MT, PT = ("America/New_York", "America/Chicago", "America/Denver",
                  "America/Los_Angeles")

TEAMS: tuple[Team, ...] = (
    _t("ATL", "ATL", 37, "Atlanta", "Hawks", "State Farm Arena", 33.7573, -84.3963, ET),
    _t("BOS", "BOS", 38, "Boston", "Celtics", "TD Garden", 42.3662, -71.0621, ET),
    _t("CLE", "CLE", 39, "Cleveland", "Cavaliers", "Rocket Arena", 41.4965, -81.6882, ET),
    _t("NOP", "NO", 40, "New Orleans", "Pelicans", "Smoothie King Center", 29.9490, -90.0821, CT),
    _t("CHI", "CHI", 41, "Chicago", "Bulls", "United Center", 41.8807, -87.6742, CT),
    _t("DAL", "DAL", 42, "Dallas", "Mavericks", "American Airlines Center", 32.7905, -96.8103, CT),
    _t("DEN", "DEN", 43, "Denver", "Nuggets", "Ball Arena", 39.7487, -105.0077, MT),
    _t("GSW", "GS", 44, "Golden State", "Warriors", "Chase Center", 37.7680, -122.3877, PT),
    _t("HOU", "HOU", 45, "Houston", "Rockets", "Toyota Center", 29.7508, -95.3621, CT),
    _t("LAC", "LAC", 46, "Los Angeles", "Clippers", "Intuit Dome", 33.9958, -118.3419, PT),
    _t("LAL", "LAL", 47, "Los Angeles", "Lakers", "Crypto.com Arena", 34.0430, -118.2673, PT),
    _t("MIA", "MIA", 48, "Miami", "Heat", "Kaseya Center", 25.7814, -80.1870, ET),
    _t("MIL", "MIL", 49, "Milwaukee", "Bucks", "Fiserv Forum", 43.0451, -87.9174, CT),
    _t("MIN", "MIN", 50, "Minnesota", "Timberwolves", "Target Center", 44.9795, -93.2761, CT),
    _t("BKN", "BKN", 51, "Brooklyn", "Nets", "Barclays Center", 40.6826, -73.9754, ET),
    _t("NYK", "NY", 52, "New York", "Knicks", "Madison Square Garden", 40.7505, -73.9934, ET),
    _t("ORL", "ORL", 53, "Orlando", "Magic", "Kia Center", 28.5392, -81.3839, ET),
    _t("IND", "IND", 54, "Indiana", "Pacers", "Gainbridge Fieldhouse", 39.7640, -86.1555,
       "America/Indiana/Indianapolis"),
    _t("PHI", "PHI", 55, "Philadelphia", "76ers", "Wells Fargo Center", 39.9012, -75.1720, ET),
    _t("PHX", "PHX", 56, "Phoenix", "Suns", "Mortgage Matchup Center", 33.4457, -112.0712,
       "America/Phoenix"),
    _t("POR", "POR", 57, "Portland", "Trail Blazers", "Moda Center", 45.5316, -122.6668, PT),
    _t("SAC", "SAC", 58, "Sacramento", "Kings", "Golden 1 Center", 38.5802, -121.4997, PT),
    _t("SAS", "SA", 59, "San Antonio", "Spurs", "Frost Bank Center", 29.4270, -98.4375, CT),
    _t("OKC", "OKC", 60, "Oklahoma City", "Thunder", "Paycom Center", 35.4634, -97.5151, CT),
    _t("TOR", "TOR", 61, "Toronto", "Raptors", "Scotiabank Arena", 43.6435, -79.3791,
       "America/Toronto"),
    _t("UTA", "UTAH", 62, "Utah", "Jazz", "Delta Center", 40.7683, -111.9011, MT),
    _t("MEM", "MEM", 63, "Memphis", "Grizzlies", "FedExForum", 35.1382, -90.0505, CT),
    _t("WAS", "WSH", 64, "Washington", "Wizards", "Capital One Arena", 38.8981, -77.0209, ET),
    _t("DET", "DET", 65, "Detroit", "Pistons", "Little Caesars Arena", 42.3410, -83.0553, ET),
    _t("CHA", "CHA", 66, "Charlotte", "Hornets", "Spectrum Center", 35.2251, -80.8392, ET),
)

_BY_TRI = {t.tricode: t for t in TEAMS}
_BY_ESPN = {t.espn_abbr: t for t in TEAMS}
_BY_ID = {t.team_id: t for t in TEAMS}


def by_tricode(code: str) -> Team:
    return _BY_TRI[code.upper()]


def by_espn(abbr: str) -> Team:
    return _BY_ESPN[abbr.upper()]


def espn_to_tricode(abbr: str) -> str:
    return by_espn(abbr).tricode


def by_team_id(team_id: int | str) -> Team:
    return _BY_ID[int(team_id)]


# --- Conference / division (current alignment) ------------------------------------------------

DIVISIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Atlantic": ("East", ("BOS", "BKN", "NYK", "PHI", "TOR")),
    "Central": ("East", ("CHI", "CLE", "DET", "IND", "MIL")),
    "Southeast": ("East", ("ATL", "CHA", "MIA", "ORL", "WAS")),
    "Northwest": ("West", ("DEN", "MIN", "OKC", "POR", "UTA")),
    "Pacific": ("West", ("GSW", "LAC", "LAL", "PHX", "SAC")),
    "Southwest": ("West", ("DAL", "HOU", "MEM", "NOP", "SAS")),
}
CONFERENCES = ("East", "West")
_DIVISION_OF = {t: d for d, (_, ts) in DIVISIONS.items() for t in ts}


def division(tricode: str) -> str:
    """Current division of a team ('Atlantic', 'Central', 'Southeast', 'Northwest', 'Pacific',
    'Southwest')."""
    return _DIVISION_OF[by_tricode(tricode).tricode]


def conference(tricode: str) -> str:
    """Current conference of a team: 'East' or 'West'."""
    return DIVISIONS[division(tricode)][0]


def teams_in(*, conference: str | None = None, division: str | None = None) -> list[Team]:
    """Teams of a conference and/or division (case-insensitive), in TEAMS order."""
    def keep(t: Team) -> bool:
        d = _DIVISION_OF[t.tricode]
        return ((division is None or d.lower() == division.lower())
                and (conference is None or DIVISIONS[d][0].lower() == conference.lower()))
    return [t for t in TEAMS if keep(t)]


# --- Franchise continuity -------------------------------------------------------------------
#
# Historical tricodes -> current franchise tricode, by *roster continuity* (who the players
# were), which is what a rating model needs. This deliberately differs from the NBA's official
# record-keeping in one place: in 2014 the NBA reassigned the 1988-2002 Charlotte Hornets'
# history to today's Charlotte franchise, and stats.nba.com reports those CHH seasons with
# TEAM_ID 1610612766 (Charlotte). But that roster moved to New Orleans in 2002 (NOH -> NOK ->
# NOP), while Charlotte restarted as the 2004 expansion Bobcats. So the numeric team_id is NOT
# a safe continuity key (team_games stores no team_id anyway); use franchise(tricode, season).
#
# Also accepted: Basketball-Reference style codes (BRK, CHO, PHO, NOH, CHH) and ESPN
# abbreviations, so other sources can be joined.

_FRANCHISE_ALIASES: dict[str, str] = {
    "NJN": "BKN", "BRK": "BKN",            # New Jersey Nets -> Brooklyn (2012-13)
    "SEA": "OKC",                          # Seattle SuperSonics -> Oklahoma City (2008-09)
    "VAN": "MEM",                          # Vancouver Grizzlies -> Memphis (2001-02)
    "NOH": "NOP", "NOK": "NOP",            # New Orleans (/Oklahoma City 2005-07) Hornets
    "CHH": "NOP",                          # Charlotte Hornets 1988-2002 -> New Orleans
    "CHO": "CHA",                          # B-Ref code for Charlotte 2014-
    "PHO": "PHX", "GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA",
    "WSH": "WAS",
}

# Season (start year) in which the Charlotte expansion franchise (Bobcats) began. Any "CHA"
# before it can only be the original Hornets, i.e. today's Pelicans.
CHARLOTTE_EXPANSION_YEAR = 2004


def season_start_year(season: str | int | None) -> int | None:
    """'2004-05' -> 2004; 2004 -> 2004; None -> None."""
    if season is None:
        return None
    if isinstance(season, int):
        return season
    return int(str(season)[:4])


def franchise(tricode: str, season: str | int | None = None) -> str:
    """Map any historical NBA tricode to the current franchise's tricode.

    `season` ('2004-05' or start year) only matters for ambiguous codes: "CHA" before 2004-05
    means the original Hornets (-> NOP). Raises KeyError for unknown codes.
    """
    code = tricode.strip().upper()
    year = season_start_year(season)
    if code == "CHA" and year is not None and year < CHARLOTTE_EXPANSION_YEAR:
        return "NOP"
    code = _FRANCHISE_ALIASES.get(code, code)
    if code not in _BY_TRI:
        raise KeyError(f"unknown NBA tricode {tricode!r}")
    return code
