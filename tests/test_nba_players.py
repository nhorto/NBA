import json
from pathlib import Path

import pytest

from courtside.http import BlockedError
from courtside.sources import nba_players as N

FIX = Path(__file__).parent / "fixtures" / "nba_players"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def test_minutes_height_birthdate():
    assert N.minutes("38:36") == pytest.approx(38.6)
    assert N.minutes("PT34M21.00S") == pytest.approx(34.35)
    assert N.minutes(12) == 12.0 and N.minutes("") is None and N.minutes(None) is None
    assert N.minutes("garbage") is None
    assert N.height_inches("6-8") == 80 and N.height_inches("") is None
    assert N._birth_date("DEC 01, 1992") == "1992-12-01" and N._birth_date("") is None


def test_player_index():
    rows = N.parse_player_index(load("playerindex"), "2025-26")
    assert len(rows) == 4
    r = rows[0]
    assert r["player_id"] == 76001 and r["last_name"] == "Abdelnaby"
    assert r["height_in"] == 82 and r["weight"] == 240 and r["draft_year"] == 1990
    assert r["country"] == "USA" and r["snapshot_season"] == "2025-26"


def test_roster_and_coaches():
    roster, coaches = N.parse_roster(load("roster"), "2025-26")
    assert len(roster) == 4 and len(coaches) == 4
    assert roster[0]["birth_date"] == "1992-12-01" and roster[0]["exp"] == 9
    assert roster[0]["team"] == "GSW" and roster[0]["height_in"] == 74
    assert {r["exp"] for r in roster} == {9, 2, 0, 4}          # 'R' (rookie) -> 0
    head = coaches[0]
    assert head["coach_name"] == "Steve Kerr" and head["coach_type"] == "Head Coach"
    assert head["is_assistant"] == 1 and coaches[1]["coach_type"] == "Assistant Coach"
    assert all(c["season"] == "2025-26" and c["team"] == "GSW" for c in coaches)


def test_season_level_tables():
    ro = N.parse_matchups_rollup(load("matchupsrollup"), "2025-26", "Regular Season")[0]
    assert ro["def_player_id"] == 2544 and ro["position"] == "G"
    assert ro["matchup_min"] == pytest.approx(315 + 4 / 60) and ro["partial_poss"] == 1616.7

    lsm = N.parse_season_matchups(load("leagueseasonmatchups"), "2025-26", "Regular Season")[0]
    assert (lsm["off_player_id"], lsm["def_player_id"]) == (1628389, 1628976)
    assert lsm["matchup_min"] == pytest.approx(38 + 41 / 60) and lsm["fga"] == 36.0
    assert lsm["extra_json"] is None          # derived headers are not kept

    d = N.parse_pt_defend(load("ptdefend"), "2025-26", "Regular Season", "Overall")[0]
    assert d["category"] == "Overall" and d["d_fgm"] == 656 and d["pct_plusminus"] == -0.046

    s = N.parse_pt_shot(load("ptshot"), "2025-26", "Regular Season", "6+")[0]
    assert s["def_dist"] == "6+" and s["fga"] == 481 and s["fg3_pct"] == 0.403

    h = N.parse_hustle_season(load("hustle_season"), "2025-26", "Regular Season")[0]
    assert h["deflections"] == 16 and h["min"] == 173.0
    assert json.loads(h["extra_json"])["PCT_BOX_OUTS_REB"] == 0.5   # unmapped header kept


def test_synergy_player_and_team():
    p = N.parse_synergy(load("synergy_player"), "2025-26", "Regular Season", "P",
                        "Offensive", "Isolation")[0]
    assert p["entity"] == "P" and p["player_id"] == 1628983 and p["ppp"] == 1.163
    t = N.parse_synergy(load("synergy_team"), "2025-26", "Regular Season", "T",
                        "Defensive", "Isolation")[0]
    assert t["entity"] == "T" and t["player_id"] == 0 and t["team"] == "BOS"
    assert t["grouping"] == "Defensive" and t["poss"] == 487


def test_game_matchups_orientation_and_teams():
    rows = N.parse_game_matchups(load("game_matchups"))
    assert len(rows) == 12
    r = rows[0]
    # top-level player (Harrison Barnes, SAC) is the OFFENSIVE player, listed matchups defend him
    assert (r["off_player_id"], r["def_player_id"]) == (203084, 203076)
    assert (r["off_team"], r["def_team"]) == ("SAC", "LAL")
    assert r["matchup_min"] == pytest.approx(98.0 / 60) and r["partial_poss"] == 7.5
    assert r["team_pts"] == 10 and r["sfl"] == 0
    assert {x["off_team"] for x in rows} == {"SAC", "LAL"}
    assert all(x["game_id"] == "0022300100" for x in rows)
    assert len({(x["off_player_id"], x["def_player_id"]) for x in rows}) == 12


def test_game_tracking_and_hustle_skip_dnp():
    t = N.parse_game_tracking(load("game_tracking"))
    h = N.parse_game_hustle(load("game_hustle"))
    assert len(t) == 4 and len(h) == 4          # fixtures also hold a 0-minute DNP each
    assert t[0]["touches"] == 43 and t[0]["cont_fga"] == 4 and t[0]["team"] == "SAC"
    assert t[0]["minutes"] == pytest.approx(38.6)
    assert h[0]["contested_shots"] == 8 and h[0]["points"] == 8 and h[0]["box_outs"] == 1


def test_empty_shell_has_no_rows():
    payload = load("game_matchups_empty")
    assert N.check_game(payload, "boxscorematchupsv3") is payload
    assert N.parse_game_matchups(payload) == []
    assert not N.game_has_data(payload, "boxscorematchupsv3")


def test_check_game_rejects_foreign_payload():
    with pytest.raises(BlockedError):
        N.check_game({"unexpected": 1}, "boxscorehustlev2")
    with pytest.raises(BlockedError):
        N.check_game(load("game_hustle"), "boxscorematchupsv3")
