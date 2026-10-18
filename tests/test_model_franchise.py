import pytest

from courtside import teams

# Every distinct tricode in team_games after the 2000-01..2025-26 backfill (verified against
# var/courtside.db), with its season range and the expected current franchise.
OBSERVED = {
    "ATL": "ATL", "BKN": "BKN", "BOS": "BOS", "CHA": "CHA", "CHH": "NOP", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW", "HOU": "HOU",
    "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "NOP": "NOP", "NYK": "NYK",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHX": "PHX", "POR": "POR", "SAC": "SAC",
    "SAS": "SAS", "SEA": "OKC", "TOR": "TOR", "UTA": "UTA", "VAN": "MEM", "WAS": "WAS",
}
FIRST_SEASON = {"CHA": "2004-05", "CHH": "2000-01", "NOH": "2002-03", "NOK": "2005-06",
                "SEA": "2000-01", "VAN": "2000-01", "NJN": "2000-01", "BKN": "2012-13",
                "MEM": "2001-02", "OKC": "2008-09", "NOP": "2013-14"}


@pytest.mark.parametrize("code,expected", sorted(OBSERVED.items()))
def test_every_observed_tricode_maps(code, expected):
    season = FIRST_SEASON.get(code, "2010-11")
    assert teams.franchise(code, season) == expected
    assert teams.franchise(code.lower(), season) == expected


def test_observed_covers_all_thirty_franchises():
    assert set(OBSERVED.values()) == {t.tricode for t in teams.TEAMS}


def test_charlotte_roster_continuity():
    # 1988-2002 Hornets roster moved to New Orleans; 2004 Bobcats are today's Charlotte.
    assert teams.franchise("CHH", "2001-02") == "NOP"
    assert teams.franchise("CHA", "2001-02") == "NOP"
    assert teams.franchise("CHA", "2004-05") == "CHA"
    assert teams.franchise("CHA", 2014) == "CHA"
    assert teams.franchise("CHA") == "CHA"


def test_other_source_aliases():
    for a, b in {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "GS": "GSW", "WSH": "WAS",
                 "UTAH": "UTA", "NO": "NOP"}.items():
        assert teams.franchise(a, "2020-21") == b


def test_unknown_tricode_raises():
    with pytest.raises(KeyError):
        teams.franchise("XYZ", "2020-21")


def test_season_start_year():
    assert teams.season_start_year("2004-05") == 2004
    assert teams.season_start_year(1999) == 1999
    assert teams.season_start_year(None) is None
