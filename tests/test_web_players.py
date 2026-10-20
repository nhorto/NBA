"""The /players page (Tonight player cards + props vs the market, T2.9).

Synthetic day from tests/props_helpers.py in a temporary courtside.db; the app reads it
read-only. Includes the owner's wording rule: information only, no advice words.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.brief import factcheck  # noqa: E402
from courtside.web import players_view  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

import props_helpers as H  # noqa: E402

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
# the owner's list (binding) on top of the Brief's betting / DFS-tipster lists
ADVICE_RE = re.compile(r"\b(?:over|overs|edges?|lean(?:s|ing)?|value|smash\w*|locks?|"
                       r"best bet|pick(?:s)?)\b", re.I)


@pytest.fixture
def client(data_dir):
    conn = db.connect()
    H.populate(conn)
    conn.close()
    app = create_app(data_dir)
    app.state.players_now = lambda: NOW
    return TestClient(app)


def _main_text(html: str) -> str:
    """Visible text of <main> (scripts/styles dropped, tags to spaces)."""
    m = re.search(r"<main[^>]*>(.*)</main>", html, re.S)
    body = m.group(1) if m else html
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    body = body.replace("&rsquo;", "'").replace("&ldquo;", '"').replace("&rdquo;", '"')
    return re.sub(r"\s+", " ", body)


def test_route_renders_cards_and_props(client):
    r = client.get(f"/players/{H.DAY}")
    assert r.status_code == 200
    t = _main_text(r.text)
    assert "Tonight's players" in t
    for name in ("Ann Alder", "Nic Claxton", "Bo Birch", "Di Dogwood", "Ed Elm"):
        assert name in t
    # the canonical wording, both percentages
    assert re.search(r"Points 20\.5: our model \d{1,3}% / market 50%", t)
    assert re.search(r"Rebounds 5: our model \d{1,3}% / market \d{1,3}%", t)
    assert "a push" in t                                    # integer line: push shown
    # projection with 10th-90th ranges, availability
    assert re.search(r"Pts 20\.0 range \d+–\d+", t)
    assert "Questionable" in t and "Left ankle sprain" in t
    assert "chance he plays" in t
    assert "Information, not betting advice" in t
    # unjoined names do not get a card
    assert "Qq Nobody" not in t and "Zed Zeta" not in t


def test_filter_and_sort(client):
    r = client.get(f"/players/{H.DAY}?game={H.G2}&sort=reb")
    t = _main_text(r.text)
    assert "Ed Elm" in t and "Ann Alder" not in t
    assert t.index("Ed Elm") < t.index("Di Dogwood")       # 9 reb before 4 reb
    r = client.get(f"/players/{H.DAY}?sort=name")
    t = _main_text(r.text)
    assert t.index("Ann Alder") < t.index("Nic Claxton")    # Alder < Claxton (G1)
    # a bad game id or sort falls back to everything / the default
    r = client.get(f"/players/{H.DAY}?game=nope&sort=bogus")
    assert r.status_code == 200 and "Ed Elm" in r.text and "Ann Alder" in r.text


def test_redirect_empty_day_and_bad_date(client):
    r = client.get("/players", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("/players/")
    r = client.get("/players/2026-10-25")
    assert r.status_code == 200 and "No games on" in r.text
    assert client.get("/players/2026-13-45").status_code == 404
    assert client.get("/players/yesterday").status_code == 404


def test_nav_links(client):
    html = client.get(f"/players/{H.DAY}").text
    assert 'href="/players"' in html and 'class="top-link is-active">Players' in html
    assert f'href="/dfs/{H.DAY}"' in html


def test_no_advice_words(client):
    """Owner rule: information only. Every advice / tipster word is banned on the page; the
    one allowed mention of betting is the disclaimer that this is *not* betting advice."""
    for q in ("", f"?game={H.G1}&sort=pts", "?sort=name"):
        t = _main_text(client.get(f"/players/{H.DAY}{q}").text)
        t = t.replace(players_view.DISCLAIMER.replace("’", "'"), " ")
        t = t.replace("not betting advice", " ")
        assert not factcheck.BETTING_RE.search(t), factcheck.BETTING_RE.search(t)
        assert not factcheck.STORY_BANNED_RE.search(t), factcheck.STORY_BANNED_RE.search(t)
        assert not ADVICE_RE.search(t), ADVICE_RE.search(t)


def test_template_source_has_no_advice_words():
    from pathlib import Path
    src = (Path(players_view.__file__).parent / "templates" / "players.html").read_text()
    src = re.sub(r"\{#.*?#\}", " ", src, flags=re.S)          # template comments
    assert not ADVICE_RE.search(src), ADVICE_RE.search(src)


def test_build_page_is_cached_and_read_only(data_dir):
    conn = db.connect()
    H.populate(conn)
    conn.close()
    import sqlite3
    ro = sqlite3.connect(f"file:{data_dir / 'courtside.db'}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    cache: dict = {}
    v1 = players_view.build_page(ro, H.DAY, now=NOW, cache=cache)
    assert v1["state"] == "ok" and v1["n_games"] == 2 and v1["n_props"] == H.N_JOINED
    assert "games" in cache
    v2 = players_view.build_page(ro, H.DAY, now=NOW, params={"game": H.G1}, cache=cache)
    assert [g["game_id"] for g in v2["games"]] == [H.G1]
    card = next(c for g in v1["games"] for c in g["cards"] if c["name"] == "Ann Alder")
    assert [p["stat"] for p in card["props"]] == ["pts", "reb"]
    keys = [s["key"] for s in card["stats"]]
    assert keys == ["min", "pts", "reb", "ast", "fg3m", "dk"]
    assert all(s["lo"] is not None and s["hi"] is not None for s in card["stats"])
    fp1 = players_view.fingerprint(ro, H.DAY)
    ro.close()
    rw = db.connect()
    rw.execute("UPDATE prop_snapshots SET over_odds=-120 WHERE id=1")
    rw.execute("INSERT INTO prop_snapshots (game_id, captured_at, provider, espn_athlete_id, "
               "player_name, market_id, market, kind, line, over_odds, under_odds) VALUES "
               "(?, '2026-10-20T21:00:00Z', '100', 'a401', 'Ed Elm', 'x', 'Total Rebounds', "
               "'ou', 8.5, -110, -110)", (H.G2,))
    rw.commit()
    ro = sqlite3.connect(f"file:{data_dir / 'courtside.db'}?mode=ro", uri=True)
    assert players_view.fingerprint(ro, H.DAY) != fp1         # a new capture busts the cache


def test_no_database(tmp_path):
    v = players_view.build_page(None, H.DAY, now=NOW)
    assert v["state"] == "no_db"
