import json
from pathlib import Path

import pytest

from courtside.sources import nba_stats as ns

FIX = Path(__file__).parent / "fixtures" / "nba_stats"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def test_result_rows_both_envelopes():
    a = {"resultSets": [{"headers": ["A", "B"], "rowSet": [[1, 2], [3, 4]]}]}
    b = {"resultSet": {"headers": ["A", "B"], "rowSet": [[1, 2], [3, 4]]}}
    assert ns.result_rows(a) == ns.result_rows(b) == [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
    assert ns.result_rows({"resultSets": []}) == []


def test_parse_minutes_and_matchup():
    assert ns.parse_minutes("28:30") == 28.5
    assert ns.parse_minutes(290) == 290.0
    assert ns.parse_minutes("21.7") == 21.7
    assert ns.parse_minutes(None) is None and ns.parse_minutes("") is None
    assert ns.parse_matchup("OKC vs. HOU") == (1, "HOU")
    assert ns.parse_matchup("GSW @ LAL") == (0, "LAL")
    assert ns.parse_matchup(None) == (None, None)


def test_team_game_rows():
    rows = ns.team_game_rows(load("team_gamelog"), "2025-26", "Regular Season")
    assert len(rows) == 10
    r = rows[0]
    assert r["game_id"] == "0022500001" and r["team"] == "OKC" and r["opp"] == "HOU"
    assert r["is_home"] == 1 and r["wl"] == "W" and r["pts"] == 125
    assert r["season"] == "2025-26" and r["game_date"] == "2025-10-21"
    assert isinstance(r["min"], float)
    assert {x["is_home"] for x in rows} == {0, 1}
    assert "fg_pct" not in r and "team_id" not in r


def test_player_game_rows():
    rows = ns.player_game_rows(load("player_gamelog"), "2025-26", "Regular Season")
    assert len(rows) == 10
    r = rows[0]
    assert r["player_id"] == 203110 and r["player_name"] == "Draymond Green"
    assert r["team"] == "GSW" and r["game_date"] == "2025-10-21" and r["min"] == 28.0
    assert r["pts"] == 8 and r["plus_minus"] == 20


def test_team_adv_rows():
    rows = ns.team_adv_rows(load("team_adv"))
    assert len(rows) == 10
    r = rows[0]
    assert r["off_rating"] == 136.0 and r["def_rating"] == 101.0  # non-estimated, not E_
    assert r["tov_pct"] == 0.1 and r["pace"] == 100.0 and r["poss"] == 100
    assert set(r) == {"game_id", "team", "off_rating", "def_rating", "net_rating", "pace", "poss",
                      "efg_pct", "ts_pct", "tov_pct", "oreb_pct", "dreb_pct", "ast_pct", "pie"}


def test_player_adv_rows():
    rows = ns.player_adv_rows(load("player_adv"))
    assert len(rows) == 10
    r = rows[0]
    assert r["player_id"] == 1630194 and r["off_rating"] == 146.0
    assert r["usg_pct"] == 0.232 and r["reb_pct"] == 0.146
    assert set(r) == {"game_id", "player_id", "off_rating", "def_rating", "net_rating", "usg_pct",
                      "ts_pct", "efg_pct", "ast_pct", "reb_pct", "pace", "poss", "pie"}


def test_fetchers_build_params(monkeypatch):
    calls = []
    monkeypatch.setattr(ns, "fetch_json", lambda url, params=None, **kw: (
        calls.append((url, params)) or load("team_gamelog")))
    ns.fetch_league_gamelog("2025-26", "PlayIn", "T")
    ns.fetch_gamelogs_advanced("2025-26", "Playoffs", "player")
    assert calls[0][0].endswith("/leaguegamelog")
    assert calls[0][1]["SeasonType"] == "PlayIn" and calls[0][1]["PlayerOrTeam"] == "T"
    assert calls[1][0].endswith("/playergamelogs") and calls[1][1]["MeasureType"] == "Advanced"
    with pytest.raises(ValueError):
        ns.fetch_league_gamelog("2025-26", "Regular Season", "X")


def test_checked_rejects_garbage(monkeypatch):
    from courtside.http import BlockedError
    monkeypatch.setattr(ns, "fetch_json", lambda *a, **k: {"message": "nope"})
    with pytest.raises(BlockedError):
        ns.fetch_league_gamelog("2025-26", "Regular Season", "T")
