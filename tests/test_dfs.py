"""DraftKings DFS parsing (trimmed real payloads from 2026-10-19: lobby, draftables, rules)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from courtside.sources import dfs

FIX = Path(__file__).parent / "fixtures" / "dfs"


def J(name):
    return json.loads((FIX / name).read_text())


def test_lobby_draft_groups_filter_by_day_and_type():
    lobby = J("lobby.json")
    all_nba = dfs.lobby_draft_groups(lobby)
    assert {g["draft_group_id"] for g in all_nba} >= {154575, 154825, 154823, 150563}
    day = dfs.lobby_draft_groups(lobby, day=date(2026, 10, 20), game_types=dfs.SALARY_GAME_TYPES)
    assert [(g["draft_group_id"], g["game_type"]) for g in day] == [
        (154575, "classic"), (154825, "showdown")]
    g = day[0]
    assert g["start_utc"] == "2026-10-20T19:00:00Z" and g["start_et"].startswith("2026-10-20T15")
    assert g["game_count"] == 3
    # previous-evening group (the WNBA one) is excluded by the day filter
    assert all(x["draft_group_id"] != 155096 for x in
               dfs.lobby_draft_groups(lobby, day=date(2026, 10, 20)))


def test_rules_roster_construction():
    classic = dfs.parse_rules(J("rules_70.json"))
    assert classic["salary_cap"] == 50000 and classic["min_games"] == 2
    assert [s["name"] for s in classic["slots"]] == list(dfs.CLASSIC_ROSTER)
    show = dfs.parse_rules(J("rules_81.json"))
    assert [s["name"] for s in show["slots"]] == list(dfs.SHOWDOWN_ROSTER)
    assert show["slots"][0]["note"] == "Scores 1.5x Fantasy Points"
    assert dfs.slot_names(J("rules_70.json"))[465] == "UTIL"


def test_parse_classic_draftables():
    rules = J("rules_70.json")
    rows = dfs.parse_draftables(J("draftables_classic.json"), draft_group_id=154575,
                                game_type_id=70, rules=rules)
    assert rows and {r["game_type"] for r in rows} == {"classic"}
    assert {r["slate_id"] for r in rows} == {"154575"} and {r["site"] for r in rows} == {"dk"}
    wemby = [r for r in rows if r["player_name"] == "Victor Wembanyama"]
    assert {r["roster_slot"] for r in wemby} == {"C", "UTIL"}      # one row per eligible slot
    assert {r["salary"] for r in wemby} == {10500} and len(wemby) == 2
    w = wemby[0]
    assert (w["team"], w["positions"], w["home"], w["away"]) == ("SAS", "C", "SAS", "OKC")
    assert w["site_player_id"] == "1373356" and w["site_player_dk_id"] == "713754"
    assert w["draftable_id"] and w["fppg"] == 51.7 and w["game_label"] == "OKC @ SAS"
    assert w["start_time"] == "2026-10-21T01:30:00.0000000Z" and w["disabled"] == 0
    # combo-guard eligibility expands to PG, SG, G, UTIL
    assert {r["roster_slot"] for r in rows if r["positions"] == "PG/SG"} == \
        {"PG", "SG", "G", "UTIL"}
    assert {r["roster_slot"] for r in rows if r["positions"] == "PF/C"} == \
        {"PF", "C", "F", "UTIL"}


def test_parse_showdown_captain_salary():
    rows = dfs.parse_draftables(J("draftables_showdown.json"), draft_group_id=154825,
                                game_type_id=81, rules=J("rules_81.json"))
    assert {r["game_type"] for r in rows} == {"showdown"}
    by = {}
    for r in rows:
        by.setdefault(r["player_name"], {})[r["roster_slot"]] = r["salary"]
    assert by
    for slots in by.values():
        assert slots["CPT"] == round(slots["UTIL"] * dfs.SHOWDOWN_CAPTAIN_MULTIPLIER)


def test_parse_without_rules_keeps_slot_id():
    rows = dfs.parse_draftables(J("draftables_classic.json"), draft_group_id=1, game_type_id=70)
    assert all(r["roster_slot"] is None and r["roster_slot_id"] for r in rows)


def test_slate_games():
    games = dfs.slate_games(J("draftables_classic.json"))
    assert [(g["away"], g["home"]) for g in games] == [("BOS", "DET"), ("PHI", "NYK"),
                                                       ("OKC", "SAS")]


def test_classic_scoring_rules():
    # 30 pts, 3 threes, 10 reb, 5 ast, 2 stl, 1 blk, 4 tov: double-double bonus only
    assert dfs.classic_points(30, 3, 10, 5, 2, 1, 4) == \
        30 + 1.5 + 12.5 + 7.5 + 4 + 2 - 2 + 1.5
    # triple-double: both bonuses
    assert dfs.classic_points(10, 0, 10, 10) == 10 + 12.5 + 15 + 1.5 + 3
    assert dfs.classic_points() == 0
