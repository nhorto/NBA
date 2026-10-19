"""Action Network scoreboard parsing and team mapping (trimmed real payload, 2026-10-20 slate)."""

from __future__ import annotations

import json
from pathlib import Path

from courtside.sources import action_network as an

FIX = Path(__file__).parent / "fixtures" / "action_network"


def payload():
    return json.loads((FIX / "scoreboard_20261020.json").read_text())


def test_games_map_to_our_tricodes():
    games = an.scoreboard_games(payload())
    assert [(g["away"], g["home"]) for g in games[:3]] == [
        ("BOS", "DET"), ("PHI", "NYK"), ("OKC", "SAS")]
    assert games[0]["an_game_id"] == 295136 and games[0]["status"] == "scheduled"
    assert games[3]["status"] == "inprogress"


def test_book_rows_and_sign_convention():
    g = an.scoreboard_games(payload())[1]    # PHI @ NYK: NYK favoured
    books = {b["book"]: b for b in g["books"]}
    assert set(books) == {"consensus", "open", "draftkings", "fanduel"}
    dk = books["draftkings"]
    assert dk["book_id"] == 68 and dk["is_open"] == 0
    assert dk["home_spread"] == -4.5            # negative = home favoured
    assert (dk["home_spread_price"], dk["away_spread_price"]) == (-112, -108)
    assert dk["total"] == 231.5 and (dk["over_price"], dk["under_price"]) == (-112, -108)
    assert (dk["home_ml"], dk["away_ml"]) == (-185, 154)
    assert dk["line_updated_at"].startswith("2026-10-19T10:21")
    assert all(isinstance(dk[k], (int, type(None))) for k in ("home_ml", "over_price"))


def test_opening_line_flagged():
    g = an.scoreboard_games(payload())[1]
    opn = next(b for b in g["books"] if b["book_id"] == 30)
    assert opn["book"] == "open" and opn["is_open"] == 1 and opn["home_spread"] == -5.5


def test_unmapped_team_and_unknown_book():
    p = payload()
    p["games"][0]["teams"][0]["abbr"] = "XXX"
    p["games"][0]["teams"][0]["full_name"] = "Nowhere Nothings"
    p["games"][1]["odds"][0]["book_id"] = 4242
    games = an.scoreboard_games(p)
    assert all(g["an_game_id"] != 295136 for g in games)       # skipped, not fatal
    assert games[0]["books"][0]["book"] == "book_4242"


def test_abbreviation_aliases():
    p = payload()
    for t in p["games"][2]["teams"]:
        t["abbr"] = {"SAS": "SA", "OKC": "OKC"}[t["abbr"]]     # ESPN-style 'SA'
    assert an.scoreboard_games(p)[2]["home"] == "SAS"


def test_fetch_requests_book_ids(monkeypatch):
    seen = {}
    monkeypatch.setattr(an.http, "fetch_json",
                        lambda url, params=None, **k: seen.update(url=url, params=params) or {})
    from datetime import date
    an.fetch_scoreboard(date(2026, 10, 20))
    assert seen["params"]["date"] == "20261020" and seen["params"]["period"] == "game"
    assert seen["params"]["bookIds"].split(",")[:2] == ["15", "30"]
