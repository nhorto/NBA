"""ESPN source parsers. Fixtures only; never touches the network."""

from __future__ import annotations

import copy
import json
import logging
from datetime import date
from pathlib import Path

import pytest

from courtside.sources import espn

FIX = Path(__file__).parent / "fixtures" / "espn"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


@pytest.fixture
def pre() -> dict:
    return load("scoreboard_20261020.json")


@pytest.fixture
def done() -> dict:
    return load("scoreboard_20260410.json")


def test_fetch_functions_build_urls(monkeypatch):
    calls = []
    monkeypatch.setattr(espn.http, "fetch_json",
                        lambda url, **kw: calls.append((url, kw.get("params"))) or {})
    espn.fetch_scoreboard(date(2026, 10, 20))
    espn.fetch_summary(401811039)
    espn.fetch_injuries()
    assert calls == [
        (espn.BASE + "/scoreboard", {"dates": "20261020"}),
        (espn.BASE + "/summary", {"event": "401811039"}),
        (espn.BASE + "/injuries", None),
    ]


# ---------------------------------------------------------------- events

def test_pregame_events(pre):
    evs = espn.scoreboard_events(pre)
    assert [e["espn_id"] for e in evs] == ["401909088", "401909089", "401909090"]
    e = evs[0]
    assert (e["home"], e["away"]) == ("DET", "BOS")
    assert e["status"] == "scheduled"
    assert e["home_score"] is None and e["away_score"] is None
    assert e["tipoff_utc"] == "2026-10-20T19:00:00Z"
    assert e["nba_day"] == "2026-10-20"
    # 01:30Z on the 21st is still the 20th in US/Eastern
    assert evs[2]["tipoff_utc"] == "2026-10-21T01:30:00Z"
    assert evs[2]["nba_day"] == "2026-10-20"


def test_team_mapping(pre, done):
    pairs = {(e["away"], e["home"]) for e in espn.scoreboard_events(done)}
    # ESPN GS/NY/SA/UTAH/WSH/NO -> NBA tricodes
    assert ("GSW", "SAC") in pairs
    assert ("MEM", "UTA") in pairs
    assert ("TOR", "NYK") in pairs
    assert ("NOP", "BOS") in pairs
    assert ("MIA", "WAS") in pairs
    assert ("DAL", "SAS") in pairs
    assert ("PHI", "NYK") in {(e["away"], e["home"]) for e in espn.scoreboard_events(pre)}


def test_final_events(done):
    evs = {e["espn_id"]: e for e in espn.scoreboard_events(done)}
    e = evs["401811039"]  # GS @ SAC, final 118-124
    assert e["status"] == "final"
    assert (e["home"], e["away"]) == ("SAC", "GSW")
    assert (e["home_score"], e["away_score"]) == (124, 118)
    assert e["nba_day"] == "2026-04-10"  # tip 02:00Z on the 11th


@pytest.mark.parametrize("name,state,completed,expected", [
    ("STATUS_SCHEDULED", "pre", False, "scheduled"),
    ("STATUS_IN_PROGRESS", "in", False, "live"),
    ("STATUS_HALFTIME", "in", False, "live"),
    ("STATUS_END_PERIOD", "in", False, "live"),
    ("STATUS_FINAL", "post", True, "final"),
    ("STATUS_POSTPONED", "post", False, "postponed"),
    ("STATUS_CANCELED", "post", False, "postponed"),
])
def test_status_mapping(pre, name, state, completed, expected):
    p = copy.deepcopy(pre)
    ev = p["events"][0]
    ev["status"]["type"].update(name=name, state=state, completed=completed)
    for c in ev["competitions"][0]["competitors"]:
        c["score"] = "55"
    got = espn.scoreboard_events(p)[0]
    assert got["status"] == expected
    if expected in ("scheduled", "postponed"):
        assert got["home_score"] is None and got["away_score"] is None
    else:
        assert got["home_score"] == 55 and got["away_score"] == 55


def test_unmapped_teams_skipped_with_warning(pre, caplog):
    p = copy.deepcopy(pre)
    p["events"][1]["competitions"][0]["competitors"][0]["team"]["abbreviation"] = "WORLD"
    with caplog.at_level(logging.WARNING, logger="courtside.sources.espn"):
        evs = espn.scoreboard_events(p)
    assert [e["espn_id"] for e in evs] == ["401909088", "401909090"]
    assert "401909089" in caplog.text
    assert [o["espn_id"] for o in espn.scoreboard_odds(p)] == ["401909088", "401909090"]


# ---------------------------------------------------------------- odds

def test_odds_bos_at_det_sign(pre):
    rows = {r["espn_id"]: r for r in espn.scoreboard_odds(pre)}
    r = rows["401909088"]
    # BOS (away) favored by 1.5 => home DET is +1.5
    assert (r["home"], r["away"]) == ("DET", "BOS")
    assert r["home_spread"] == 1.5
    assert r["total"] == 222.5
    assert (r["home_ml"], r["away_ml"]) == (-108, -112)
    assert r["source"] == "espn"
    assert r["provider"] == "DraftKings"


def test_odds_home_favorite_negative(pre):
    r = {r["espn_id"]: r for r in espn.scoreboard_odds(pre)}["401909089"]  # PHI @ NY
    assert (r["home"], r["away"]) == ("NYK", "PHI")
    assert r["home_spread"] == -4.5
    assert (r["home_ml"], r["away_ml"]) == (-192, 160)
    assert r["total"] == 232.5


def _strip_to(o: dict, keep: tuple[str, ...]) -> dict:
    return {k: v for k, v in o.items() if k in keep}


def test_spread_sign_fallbacks(pre):
    """Without pointSpread, sign comes from favorite flags, then raw, then `details`."""
    base = pre["events"][0]["competitions"][0]["odds"][0]

    def mk(**o):
        p = copy.deepcopy(pre)
        p["events"] = p["events"][:1]
        p["events"][0]["competitions"][0]["odds"] = [o]
        return espn.scoreboard_odds(p)[0]

    # unsigned magnitude + away favorite flag -> home underdog (+)
    r = mk(spread=1.5, awayTeamOdds={"favorite": True}, homeTeamOdds={"favorite": False})
    assert r["home_spread"] == 1.5
    # unsigned magnitude + home favorite flag -> negative
    r = mk(spread=1.5, awayTeamOdds={"favorite": False}, homeTeamOdds={"favorite": True})
    assert r["home_spread"] == -1.5
    # details only: "BOS -1.5" with BOS away => home +1.5; "DET -3" => home -3
    assert mk(details="BOS -1.5", overUnder=200)["home_spread"] == 1.5
    assert mk(details="DET -3", overUnder=200)["home_spread"] == -3.0
    # legacy moneyline location
    r = mk(spread=-2.0, homeTeamOdds={"moneyLine": -130}, awayTeamOdds={"moneyLine": 110})
    assert (r["home_ml"], r["away_ml"]) == (-130, 110)
    assert base["pointSpread"]["home"]["close"]["line"] == "+1.5"


def test_no_odds_no_row(done, pre):
    assert espn.scoreboard_odds(done) == []
    p = copy.deepcopy(pre)
    del p["events"][0]["competitions"][0]["odds"]
    p["events"][1]["competitions"][0]["odds"] = []
    assert [r["espn_id"] for r in espn.scoreboard_odds(p)] == ["401909090"]


# ---------------------------------------------------------------- summary

def test_summary_parts():
    s = load("summary_401811039.json")
    parts = espn.summary_parts(s)
    assert parts["boxscore"] and parts["plays"] and parts["winprobability"]
    assert parts["pickcenter"] and parts["injuries"]
    assert parts["odds"] is False  # empty list on this game
    assert len(s["plays"]) == 30
    play = s["plays"][1]
    for k in ("awayScore", "homeScore", "clock", "period", "participants", "coordinate",
              "scoringPlay", "scoreValue", "text", "wallclock"):
        assert k in play
    assert {"homeWinPercentage", "playId"} <= set(s["winprobability"][0])
    empty = espn.summary_parts({})
    assert not any(empty.values())


# ---------------------------------------------------------------- injuries

def test_injury_rows():
    rows = espn.injury_rows(load("injuries.json"))
    assert len(rows) == 6
    by_name = {r["player_name"]: r for r in rows}
    moody = by_name["Moses Moody"]
    assert moody["team"] == "GSW"  # ESPN "GS"
    assert moody["player_id"] == "espn:4432171"
    assert moody["status"] == "Out"
    assert moody["source"] == "espn"
    assert moody["reason"].startswith("Knee, Left: Surgery; ret 2026-12-01")
    assert by_name["Josh Hart"]["team"] == "NYK"
    assert by_name["Josh Hart"]["status"] == "Day-To-Day"
    assert {r["team"] for r in rows} == {"ATL", "GSW", "NYK"}
    assert all(set(r) == {"team", "player_name", "player_id", "status", "reason", "source"}
               for r in rows)
    # no detail field -> still a sensible reason
    assert by_name["Kristaps Porzingis"]["reason"].startswith("Undisclosed")


def test_injury_unmapped_team_skipped(caplog):
    p = load("injuries.json")
    p["injuries"][0]["injuries"][0]["athlete"]["team"]["abbreviation"] = "XXX"
    with caplog.at_level(logging.WARNING, logger="courtside.sources.espn"):
        rows = espn.injury_rows(p)
    assert len(rows) == 5
    assert "XXX" in caplog.text
