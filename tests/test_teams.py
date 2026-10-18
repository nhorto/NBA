from courtside import teams


def test_thirty_unique():
    assert len(teams.TEAMS) == 30
    assert len({t.team_id for t in teams.TEAMS}) == 30
    assert len({t.tricode for t in teams.TEAMS}) == 30
    assert len({t.espn_abbr for t in teams.TEAMS}) == 30
    assert min(t.team_id for t in teams.TEAMS) == 1610612737
    assert max(t.team_id for t in teams.TEAMS) == 1610612766


def test_espn_mapping():
    for a, b in {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA",
                 "WSH": "WAS", "BOS": "BOS"}.items():
        assert teams.espn_to_tricode(a) == b


def test_lookups():
    assert teams.by_team_id(1610612738).tricode == "BOS"
    assert teams.by_team_id("1610612764").tricode == "WAS"
    assert teams.by_tricode("lal").name == "Lakers"


def test_geo_and_tz_valid():
    from zoneinfo import ZoneInfo
    for t in teams.TEAMS:
        ZoneInfo(t.tz)
        assert 25 < t.lat < 46 and -123 < t.lon < -70
