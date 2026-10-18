"""nba_cdn parsers on saved fixtures; fetchers with fetch_json monkeypatched. No network."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from courtside.sources import nba_cdn

FIX = Path(__file__).parent / "fixtures" / "nba_cdn"


def load(name):
    return json.loads((FIX / name).read_text())


def by_id(rows):
    return {r["game_id"]: r for r in rows}


# ------------------------------------------------------------------ schedule

def test_schedule_excludes_preseason_and_tbd():
    rows = by_id(nba_cdn.schedule_games(load("schedule.json")))
    assert set(rows) == {"0022600001", "0022600002", "0022600003"}
    assert "0012600009" not in rows          # preseason
    assert "0062600001" not in rows          # Cup final, teams TBD
    assert "0022601201" not in rows          # TBD tricodes


def test_schedule_row_shape():
    r = by_id(nba_cdn.schedule_games(load("schedule.json")))["0022600001"]
    assert r == {
        "game_id": "0022600001", "season": "2026-27", "nba_day": "2026-10-20",
        "tipoff_utc": "2026-10-20T19:00:00Z", "home": "DET", "away": r["away"],
        "status": "scheduled", "home_score": None, "away_score": None,
    }
    assert len(r["away"]) == 3


def test_nba_day_late_night_et_game():
    # 01:30Z on the 21st is 9:30pm ET on the 20th.
    r = by_id(nba_cdn.schedule_games(load("schedule.json")))["0022600003"]
    assert r["tipoff_utc"] == "2026-10-21T01:30:00Z"
    assert r["nba_day"] == "2026-10-20"


def test_status_mapping_and_scores():
    p = load("schedule.json")
    games = [g for d in p["leagueSchedule"]["gameDates"] for g in d["games"]]
    g1 = next(g for g in games if g["gameId"] == "0022600001")
    g2 = next(g for g in games if g["gameId"] == "0022600002")
    g3 = next(g for g in games if g["gameId"] == "0022600003")
    g1.update(gameStatus=3, gameStatusText="Final")
    g1["homeTeam"]["score"], g1["awayTeam"]["score"] = 110, 105
    g2.update(gameStatus=2, gameStatusText="Q2 3:00")
    g2["homeTeam"]["score"], g2["awayTeam"]["score"] = 50, 48
    g3.update(gameStatus=1, gameStatusText="PPD")
    rows = by_id(nba_cdn.schedule_games(p))
    assert (rows["0022600001"]["status"], rows["0022600001"]["home_score"],
            rows["0022600001"]["away_score"]) == ("final", 110, 105)
    assert (rows["0022600002"]["status"], rows["0022600002"]["home_score"]) == ("live", 50)
    assert rows["0022600003"]["status"] == "postponed"
    assert rows["0022600003"]["home_score"] is None


def test_postponed_text_variants_and_flag():
    for text in ("PPD", "Postponed", "Game Postponed"):
        assert nba_cdn._status(1, text) == "postponed"
    assert nba_cdn._status(1, "7:00 pm ET", "Y") == "postponed"
    assert nba_cdn._status(1, "7:00 pm ET", "N") == "scheduled"


def test_prefix_rules():
    p = load("schedule.json")
    tmpl = next(g for d in p["leagueSchedule"]["gameDates"] for g in d["games"]
                if g["gameId"] == "0022600001")
    games = []
    for prefix in ("001", "002", "003", "004", "005", "006"):
        g = copy.deepcopy(tmpl)
        g["gameId"] = prefix + "2600001"
        games.append(g)
    payload = {"leagueSchedule": {"seasonYear": "2026-27",
                                  "gameDates": [{"gameDate": "x", "games": games}]}}
    ids = {r["game_id"][:3] for r in nba_cdn.schedule_games(payload)}
    assert ids == {"002", "004", "005", "006"}


def test_schedule_empty_payloads():
    assert nba_cdn.schedule_games({}) == []
    assert nba_cdn.schedule_games({"leagueSchedule": {"gameDates": None}}) == []


# ------------------------------------------------------------------ scoreboard

def test_scoreboard_empty_day():
    assert nba_cdn.scoreboard_games(load("scoreboard_empty.json")) == []


def test_scoreboard_games():
    rows = by_id(nba_cdn.scoreboard_games(load("scoreboard_games.json")))
    assert set(rows) == {"0022600001", "0022600002", "0022600003", "0022600004"}  # no preseason
    assert rows["0022600001"]["status"] == "final"
    assert (rows["0022600001"]["home_score"], rows["0022600001"]["away_score"]) == (101, 99)
    assert rows["0022600002"]["status"] == "live"
    assert rows["0022600003"]["status"] == "scheduled"
    assert rows["0022600003"]["home_score"] is None
    assert rows["0022600003"]["nba_day"] == "2026-10-20"
    assert rows["0022600003"]["season"] == "2026-27"
    assert rows["0022600004"]["status"] == "postponed"


# ------------------------------------------------------------------ box score

def test_is_final():
    box = load("boxscore_0022500001.json")
    assert nba_cdn.is_final(box) is True
    box["game"]["gameStatus"] = 2
    assert nba_cdn.is_final(box) is False
    assert nba_cdn.is_final({}) is False


# ------------------------------------------------------------------ odds

def test_decimal_to_american():
    f = nba_cdn.decimal_to_american
    assert f("2.000") == 100
    assert f("2.140") == 114
    assert f("1.714") == -140
    assert f("1.909") == -110
    assert f(-110) == -110 and f("150") == 150   # already American
    assert f(None) is None and f("x") is None and f("1.0") is None


def test_odds_rows_fixture():
    rows = nba_cdn.odds_rows(load("odds.json"))
    assert len(rows) == 2
    r = rows[0]
    assert r["game_id"] == "0012600008" and r["source"] == "nba_cdn"
    assert r["provider"] == "FanDuel"          # preferred book, same for every market
    assert r["home_spread"] == -1.5
    assert (r["home_ml"], r["away_ml"]) == (-112, -108)    # 1.893 / 1.926 decimal
    assert (rows[1]["home_spread"], rows[1]["home_ml"], rows[1]["away_ml"]) == (-6.5, -225, 180)
    assert r["total"] is None                   # no totals market in the real payload
    assert set(r) == {"game_id", "source", "provider", "home_spread", "total",
                      "home_ml", "away_ml"}


def test_odds_rows_fallback_provider_and_totals():
    def book(name, outs, cc="US"):
        return {"name": name, "countryCode": cc, "outcomes": outs}
    payload = {"games": [{"gameId": "0022600001", "markets": [
        {"name": "2way", "books": [book("Other", [
            {"type": "home", "odds": "2.500"}, {"type": "away", "odds": "1.570"}])]},
        {"name": "spread", "books": [book("Other", [
            {"type": "home", "odds": "1.9", "spread": "3.5"},
            {"type": "away", "odds": "1.9", "spread": "-3.5"}])]},
        {"name": "totals", "books": [book("Other", [
            {"type": "over", "odds": "1.9", "total": "224.5"},
            {"type": "under", "odds": "1.9", "total": "224.5"}])]},
    ]}]}
    (r,) = nba_cdn.odds_rows(payload)
    assert r["provider"] == "Other"
    assert (r["home_spread"], r["total"]) == (3.5, 224.5)
    assert (r["home_ml"], r["away_ml"]) == (150, -175)


def test_odds_rows_empty_cases():
    assert nba_cdn.odds_rows({}) == []
    assert nba_cdn.odds_rows({"games": []}) == []
    assert nba_cdn.odds_rows({"games": None}) == []
    assert nba_cdn.odds_rows({"games": [{"gameId": "1", "markets": []}]}) == []
    assert nba_cdn.odds_rows({"games": [{"gameId": "1"}]}) == []
    assert nba_cdn.odds_rows(None) == []
    assert nba_cdn.odds_rows([]) == []


# ------------------------------------------------------------------ fetchers

def test_fetchers_use_expected_urls(monkeypatch):
    seen = []
    monkeypatch.setattr(nba_cdn, "fetch_json", lambda url, **kw: seen.append(url) or {})
    nba_cdn.fetch_schedule()
    nba_cdn.fetch_scoreboard()
    nba_cdn.fetch_boxscore("0022500001")
    nba_cdn.fetch_playbyplay("0022500001")
    nba_cdn.fetch_odds()
    assert seen == [
        nba_cdn.SCHEDULE_URL,
        nba_cdn.SCOREBOARD_URL,
        "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022500001.json",
        "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022500001.json",
        nba_cdn.ODDS_URL,
    ]
    assert nba_cdn.SCOREBOARD_URL.endswith("liveData/scoreboard/todaysScoreboard_00.json")
    assert nba_cdn.ODDS_URL.endswith("liveData/odds/odds_todaysGames.json")
