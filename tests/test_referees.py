"""Referee assignment page parsing. Fixtures: the live empty page of 2026-10-19 and trimmed
Wayback captures (2026-01-18 link-wrapped cells, 2026-04-20 plain cells)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from courtside.sources import referees

FIX = Path(__file__).parent / "fixtures" / "referees"


def page(name):
    return (FIX / name).read_text()


def test_empty_table_is_not_an_error():
    p = referees.parse_assignments(page("empty_20261019.html"))
    assert p == {"nba_day": date(2026, 10, 19), "games": [], "replay_center": []}
    assert referees.assignment_rows(p) == []


def test_garbage_and_missing_sections_do_not_raise():
    for html in ("", "<html></html>", b"\xff\xfe not html", "<div class='nba-refs-content'>"):
        p = referees.parse_assignments(html)
        assert p["games"] == [] and p["replay_center"] == []


def test_filled_linked_cells():
    p = referees.parse_assignments(page("filled_linked_20260118.html"))
    assert p["nba_day"] == date(2026, 1, 18)
    assert len(p["games"]) == 6      # the two G League rows are not NBA games
    assert all("Maine" not in g["label"] for g in p["games"])
    g = p["games"][0]
    assert (g["label"], g["away"], g["home"]) == ("Orlando @ Memphis", "ORL", "MEM")
    assert g["crew_chief"] == [{"name": "Courtney Kirkland", "number": 61}]
    assert g["referee"] == [{"name": "Scott Twardoski", "number": 52}]
    assert g["umpire"] == [{"name": "Marat Kogut", "number": 32}]
    assert g["alternate"] == []                       # empty cell
    assert [o["name"] for o in p["replay_center"]] == ["John Butler", "Simone Jelks",
                                                       "Suyash Mehta"]


def test_filled_plain_cells_with_alternate():
    p = referees.parse_assignments(page("filled_plain_20260420.html"))
    assert p["nba_day"] == date(2026, 4, 20) and len(p["games"]) == 3
    g = p["games"][0]
    assert (g["away"], g["home"]) == ("ATL", "NYK")
    assert g["crew_chief"] == [{"name": "Zach Zarba", "number": 15}]
    assert g["alternate"] == [{"name": "Mousa Dagher", "number": 28}]
    assert [o["name"] for o in p["replay_center"]] == ["Natalie Sago", "Pat Fraher"]


def test_assignment_rows_flatten():
    rows = referees.assignment_rows(referees.parse_assignments(page("filled_plain_20260420.html")))
    games = [r for r in rows if r["role"] != "replay_center"]
    assert len(games) == 3 * 4 and {r["nba_day"] for r in rows} == {"2026-04-20"}
    assert {r["role"] for r in games} == {"crew_chief", "referee", "umpire", "alternate"}
    first = games[0]
    assert (first["official_name"], first["official_number"], first["home"], first["away"]) == \
        ("Zach Zarba", 15, "NYK", "ATL")
    rc = [r for r in rows if r["role"] == "replay_center"]
    assert len(rc) == 2 and rc[0]["game_label"] is None


def test_team_label_mapping():
    t = referees.team_tricode
    assert t("L.A. Lakers") == "LAL" and t("LA Clippers") == "LAC"
    assert t("New York") == "NYK" and t("Golden State") == "GSW"
    assert t("Oklahoma City") == "OKC" and t("Portland") == "POR" and t("Utah") == "UTA"
    assert t("Los Angeles") is None          # ambiguous city alone
    assert t("Salt Lake City") is None       # G League label, not an NBA team
    # every NBA city/name is covered
    from courtside import teams
    for tm in teams.TEAMS:
        if tm.city != "Los Angeles":
            assert t(tm.city) == tm.tricode, tm
        assert t(f"{tm.city} {tm.name}") == tm.tricode


def test_parse_officials_variants():
    assert referees.parse_officials("<td> <a href='x'> Zach Zarba (#15) </a></td>") == [
        {"name": "Zach Zarba", "number": 15}]
    assert referees.parse_officials("A B (#1), C D (#2)") == [
        {"name": "A B", "number": 1}, {"name": "C D", "number": 2}]
    assert referees.parse_officials("Pat Fraher") == [{"name": "Pat Fraher", "number": None}]
    assert referees.parse_officials("  ") == []
