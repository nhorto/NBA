"""ESPN props parsing + the over/under side semantics (fixtures are trimmed real payloads from
2026-10-19, DET-BOS event 401909088, DraftKings provider 100)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courtside.sources import espn_props as ep

FIX = Path(__file__).parent / "fixtures" / "espn_props"


def J(name):
    return json.loads((FIX / name).read_text())


def imp(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


@pytest.fixture
def pages():
    return [J("propbets_page1.json"), J("propbets_page2.json")]


@pytest.fixture
def rows(pages):
    return ep.parse_propbets(pages, provider_id="100", provider_name="DraftKings")


def find(rows, athlete=None, market=None, team=None, kind=None, line=None):
    out = [r for r in rows
           if (athlete is None or r["espn_athlete_id"] == athlete)
           and (market is None or r["market"] == market)
           and (team is None or r["espn_team_id"] == team)
           and (kind is None or r["kind"] == kind)
           and (line is None or r["line"] == line)]
    return out


# ------------------------------------------------------------------ providers / helpers

def test_providers_lists_draftkings_with_props():
    assert ep.providers(J("odds_list.json")) == [
        {"id": "100", "name": "DraftKings", "has_props": True}]


def test_ref_id_and_american():
    assert ep.ref_id({"$ref": "http://x/casinos/100?lang=en&region=us"}) == "100"
    assert ep.ref_id(None) is None
    assert [ep.american(x) for x in ("-120", "+105", "EVEN", 110, None, "")] == \
        [-120, 105, 100, 110, None, None]


def test_roster_names_flat_and_grouped():
    assert ep.roster_names(J("roster_BOS.json")) == {"3078576": "Derrick White",
                                                     "4065648": "Jayson Tatum"}
    grouped = {"athletes": [{"position": "G", "items": [{"id": "1", "displayName": "A B"}]}]}
    assert ep.roster_names(grouped) == {"1": "A B"}


# ------------------------------------------------------------------ pagination

def test_fetch_propbets_follows_page_count_and_archives_each_page(monkeypatch):
    served = {1: J("propbets_page1.json"), 2: J("propbets_page2.json")}
    asked = []
    monkeypatch.setattr(ep.http, "fetch_json",
                        lambda url, params=None, **k: asked.append((url, params))
                        or served[params["page"]])
    seen = []
    pages = ep.fetch_propbets("401909088", "100", on_page=lambda n, p: seen.append(n))
    assert len(pages) == 2 and seen == [1, 2]
    assert all(a[1]["limit"] == ep.PAGE_LIMIT for a in asked)
    assert asked[0][0].endswith("/events/401909088/competitions/401909088/odds/100/propBets")


def test_fetch_propbets_stops_at_max_pages(monkeypatch):
    monkeypatch.setattr(ep, "MAX_PAGES", 3)
    monkeypatch.setattr(ep.http, "fetch_json",
                        lambda url, params=None, **k: {"items": [], "pageCount": 50})
    assert len(ep.fetch_propbets("1", "100")) == 3


# ------------------------------------------------------------------ parsing

def test_item_pairs_collapse_to_one_row_per_line(pages, rows):
    n_items = sum(len(p["items"]) for p in pages)
    assert len(rows) < n_items
    kinds = {r["kind"] for r in rows}
    assert kinds == {"ou", "yesno", "milestone", "single"}
    assert {r["provider"] for r in rows} == {"100"}
    assert {r["provider_name"] for r in rows} == {"DraftKings"}


def test_over_under_pair_row(rows):
    r, = find(rows, "4251", "Total Points")
    assert (r["kind"], r["subject"], r["market_id"]) == ("ou", "player", "1")
    assert r["line"] == 15.5 and r["open_line"] == 15.5
    # first item of the pair is the over, second the under (module docstring)
    assert (r["over_odds"], r["under_odds"]) == (-120, -106)
    assert (r["open_over_odds"], r["open_under_odds"]) == (-124, -103)
    assert r["last_updated"] == "2026-10-19T14:39Z" and r["side_basis"] == "payload_order"


def test_positive_prices_parse_as_ints(rows):
    r, = find(rows, "3908845", "Total Points")
    assert r["over_odds"] == 100 and r["under_odds"] == -127


def test_milestone_rows_are_single_prices(rows):
    ms = find(rows, "4065648", "Points Milestones")
    assert ms and all(r["kind"] == "milestone" and r["under_odds"] is None for r in ms)
    assert sorted(r["line"] for r in ms) == [r["line"] for r in ms]   # distinct targets
    assert all(r["over_odds"] is not None for r in ms)


def test_team_and_game_level_props(rows):
    team = [r for r in rows if r["subject"] == "team" and r["market"] == "Team Total Points"]
    assert len(team) == 2 and all(r["kind"] == "ou" and r["line"] == 111.5 for r in team)
    assert {r["espn_team_id"] for r in team} == {"2", "8"}
    half, = [r for r in rows if r["market"] == "1st Half Total"]
    assert half["subject"] == "game" and half["kind"] == "ou" and half["line"] == 114.5
    ot, = [r for r in rows if r["market"] == "Will There Be Overtime?"]
    assert ot["kind"] == "yesno" and ot["line"] is None and ot["over_odds"] > 0 > ot["under_odds"]
    ml = [r for r in rows if r["market"] == "1st Half Moneyline"]
    assert len(ml) == 2 and all(r["kind"] == "single" for r in ml)


def test_single_market_prices(rows):
    fb = [r for r in rows if r["market"] == "First Basket"]
    assert fb and all(r["kind"] == "single" and r["under_odds"] is None for r in fb)


def test_names_attach_and_missing_stay_none(rows):
    names = {**ep.roster_names(J("roster_BOS.json")), **ep.roster_names(J("roster_DET.json"))}
    ep.attach_names(rows, names)
    assert find(rows, "4065648", "Total Points")[0]["player_name"] == "Jayson Tatum"
    assert all(r["player_name"] is None for r in rows if r["subject"] != "player")
    assert ep.athlete_ids(rows) >= {"4251", "4065648"}


def test_row_key_and_state_distinguish_milestone_lines(rows):
    ms = find(rows, "4065648", "Points Milestones")
    assert len({ep.row_key(r) for r in ms}) == len(ms)
    ou, = find(rows, "4251", "Total Points")
    assert ep.row_state(ou) == (15.5, -120, -106)


def test_empty_and_odd_payloads():
    assert ep.parse_propbets([]) == []
    assert ep.parse_propbets([{"items": []}]) == []
    triple = {"items": [
        {"athlete": {"$ref": "x/athletes/9"}, "type": {"id": "1", "name": "Total Points"},
         "odds": {"american": {"value": v, "open": v}}, "current": {"target": {"value": 9.5}},
         "open": {"target": {"value": 9.5}}} for v in ("-110", "-110", "-120")]}
    out = ep.parse_propbets([triple], provider_id="100")
    assert [r["kind"] for r in out] == ["single"] * 3     # >2 items: stored unpaired, not guessed


# ------------------------------------------------------------------ side semantics (evidence)

def test_side_line_move_evidence(rows):
    """Line went UP => the over gets cheaper (implied probability falls) vs its open price;
    the under gets dearer. Holds for the first item of every pair on core markets, which is
    what makes first == over."""
    core = {"Total Points", "Total Rebounds", "Total Assists"}
    moved = [r for r in rows if r["kind"] == "ou" and r["market"] in core
             and r["line"] != r["open_line"] and r["open_over_odds"] is not None]
    assert moved, "fixture needs at least one moved line"
    for r in moved:
        up = r["line"] > r["open_line"]
        d_over = imp(r["over_odds"]) - imp(r["open_over_odds"])
        d_under = imp(r["under_odds"]) - imp(r["open_under_odds"])
        assert (d_over < 0) == up and (d_under > 0) == up, r


def test_side_derrick_white_example(rows):
    """Real example: line 14.5 -> 16.5, item 1 went -123 -> -104 (cheaper = over)."""
    r, = find(rows, "3078576", "Total Points")
    assert (r["open_line"], r["line"]) == (14.5, 16.5)
    assert (r["open_over_odds"], r["over_odds"]) == (-123, -104)
    assert (r["open_under_odds"], r["under_odds"]) == (-104, -123)


def test_side_milestone_cross_check(rows):
    """'Points Milestones 26' means 26 or more = over 25.5: its price matches the FIRST item
    of the Total Points 25.5 pair, not the second."""
    ou, = find(rows, "4065648", "Total Points")
    assert ou["line"] == 25.5 and ou["over_odds"] != ou["under_odds"]
    ms, = find(rows, "4065648", "Points Milestones", line=26.0)
    assert abs(imp(ms["over_odds"]) - imp(ou["over_odds"])) < \
        abs(imp(ms["over_odds"]) - imp(ou["under_odds"]))
