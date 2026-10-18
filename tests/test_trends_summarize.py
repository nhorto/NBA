"""league_facts: JSON-serializable, leak-free, sane early in the season."""

import json
import sqlite3

from courtside.trends.summarize import league_facts
from courtside.trends.synthetic import day_date, make_league


def league(**kw):
    conn = sqlite3.connect(":memory:")
    make_league(conn, **kw)
    return conn


def test_json_serializable_at_every_stage():
    conn = league()
    for d in (day_date("2024-25", 0), day_date("2024-25", 3), day_date("2024-25", 20),
              day_date("2024-25", 40)):
        doc = league_facts(conn, d)
        s = json.dumps(doc, allow_nan=False)
        assert json.loads(s) == doc
        assert doc["as_of"] == d.isoformat() and doc["season"] == "2024-25"


def test_no_lookahead_in_facts():
    conn = league()
    d = day_date("2024-25", 18)
    before = league_facts(conn, d)
    conn.execute("UPDATE player_games SET pts = pts * 3, plus_minus = plus_minus + 40 "
                 "WHERE game_date >= ?", (d.isoformat(),))
    conn.execute("UPDATE team_games SET pts = pts + 50 WHERE game_date >= ?", (d.isoformat(),))
    assert league_facts(conn, d) == before
    assert before["data_through"] == day_date("2024-25", 17).isoformat()


def test_early_season_is_suppressed():
    conn = league()
    doc = league_facts(conn, day_date("2024-25", 4))
    assert doc["early_season"] is True
    assert doc["teams"]["rising"] == [] and doc["teams"]["sliding"] == []
    assert doc["players"]["hot"] == [] and doc["players"]["role_changes"] == []
    assert any("early season" in n for n in doc["notes"])
    assert len(doc["teams"]["power_top10"]) == 6


def test_before_first_game_has_prior_power_only():
    conn = league()
    doc = league_facts(conn, day_date("2024-25", 0))
    assert doc["data_through"] is None and doc["games_per_team_median"] == 0
    assert doc["teams"]["power_top10"][0]["team"] == "MIA"
