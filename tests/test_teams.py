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


def test_conference_and_division_cover_all_30_teams():
    assert set(teams.DIVISIONS) == {"Atlantic", "Central", "Southeast", "Northwest", "Pacific",
                                    "Southwest"}
    assigned = [t for _, (_, ts) in teams.DIVISIONS.items() for t in ts]
    assert sorted(assigned) == sorted(t.tricode for t in teams.TEAMS)      # each team once
    assert all(len(ts) == 5 for _, ts in teams.DIVISIONS.values())
    assert len(teams.teams_in(conference="East")) == len(teams.teams_in(conference="west")) == 15
    assert (teams.conference("BOS"), teams.division("BOS")) == ("East", "Atlantic")
    assert (teams.conference("OKC"), teams.division("okc")) == ("West", "Northwest")
    assert teams.division("NOP") == "Southwest" and teams.conference("MIN") == "West"
    assert [t.tricode for t in teams.teams_in(division="Pacific")] == [
        "GSW", "LAC", "LAL", "PHX", "SAC"]
