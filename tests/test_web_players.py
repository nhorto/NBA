"""The /tonight page (Tonight player cards + props vs the market, T2.9; formerly /players/<day>).

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
    r = client.get(f"/tonight/{H.DAY}")
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
    r = client.get(f"/tonight/{H.DAY}?game={H.G2}&sort=reb")
    t = _main_text(r.text)
    assert "Ed Elm" in t and "Ann Alder" not in t
    assert t.index("Ed Elm") < t.index("Di Dogwood")       # 9 reb before 4 reb
    r = client.get(f"/tonight/{H.DAY}?sort=name")
    t = _main_text(r.text)
    assert t.index("Ann Alder") < t.index("Nic Claxton")    # Alder < Claxton (G1)
    # a bad game id or sort falls back to everything / the default
    r = client.get(f"/tonight/{H.DAY}?game=nope&sort=bogus")
    assert r.status_code == 200 and "Ed Elm" in r.text and "Ann Alder" in r.text


def test_redirect_empty_day_and_bad_date(client):
    r = client.get("/tonight", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("/tonight/")
    r = client.get("/tonight/2026-10-25")
    assert r.status_code == 200 and "No games on" in r.text
    assert client.get("/tonight/2026-13-45").status_code == 404
    assert client.get("/tonight/yesterday").status_code == 404
    # the old URL moved permanently (query kept); /players is now the player directory
    r = client.get(f"/players/{H.DAY}?sort=name", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == f"/tonight/{H.DAY}?sort=name"
    assert "Ann Alder" in client.get(f"/players/{H.DAY}").text


def test_nav_links(client):
    html = client.get(f"/tonight/{H.DAY}").text
    assert 'href="/tonight" aria-current="page" class="top-link is-active">Tonight' in html
    assert f'href="/dfs/{H.DAY}"' in html


def test_no_advice_words(client):
    """Owner rule: information only. Every advice / tipster word is banned on the page; the
    one allowed mention of betting is the disclaimer that this is *not* betting advice."""
    for q in ("", f"?game={H.G1}&sort=pts", "?sort=name"):
        t = _main_text(client.get(f"/tonight/{H.DAY}{q}").text)
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


# ------------------------------------------------------------------ research panels
# "Teammate out" with/without splits and the market context strip (stats/tonight.py).

from datetime import date  # noqa: E402

from courtside.stats import tonight as TN  # noqa: E402

HIST = [("0022600801", "2026-10-10"), ("0022600802", "2026-10-12"), ("0022600803", "2026-10-14"),
        ("0022600804", "2026-10-16"), ("0022600805", "2026-10-18")]


def _add_research(conn) -> None:
    """Claxton (102, AAA) listed Out; five earlier AAA games: he plays the first three, has 0
    minutes in the fourth and no row in the fifth; Alder (101) plays all five. A 2025-26 AAA
    game without Claxton (he was not on AAA then) and an Alder game for another team must be
    ignored. Plus a pre-tip line for G1."""
    conn.execute("INSERT INTO injury_snapshots (captured_at, source, team, player_name, "
                 "player_id, status, reason) VALUES ('2026-10-20T17:00:00Z', 'nba_official', "
                 "'AAA', 'Nicolas Claxton', 102, 'Out', 'Injury/Illness - Right Knee; Soreness')")
    for k, (gid, d) in enumerate(HIST):
        # Alder: 30 + k minutes, 20 + 2k points, 5 reb, 4 ast; usage 0.20 + 0.01k
        conn.execute("INSERT INTO player_games (game_id, player_id, team, season, season_type, "
                     "game_date, min, pts, reb, ast, fg3m, stl, blk, tov) VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (gid, 101, "AAA", "2026-27", "Regular Season", d, 30 + k, 20 + 2 * k, 5, 4,
                      0, 0, 0, 0))
        conn.execute("INSERT INTO player_games_adv (game_id, player_id, usg_pct) VALUES (?,?,?)",
                     (gid, 101, 0.20 + 0.01 * k))
        if k < 4:
            conn.execute("INSERT INTO player_games (game_id, player_id, team, season, "
                         "season_type, game_date, min, pts, reb, ast) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (gid, 102, "AAA", "2026-27", "Regular Season", d, 25 if k < 3 else 0,
                          10, 8, 2))
    conn.execute("INSERT INTO player_games (game_id, player_id, team, season, season_type, "
                 "game_date, min, pts, reb, ast) VALUES ('0022500100', 101, 'AAA', '2025-26', "
                 "'Regular Season', '2026-01-10', 40, 50, 10, 10)")
    conn.execute("INSERT INTO player_games (game_id, player_id, team, season, season_type, "
                 "game_date, min, pts, reb, ast) VALUES ('0022600806', 101, 'ZZZ', '2026-27', "
                 "'Regular Season', '2026-10-19', 40, 50, 10, 10)")
    conn.execute("INSERT INTO odds_snapshots (game_id, captured_at, source, provider, "
                 "home_spread, total, home_ml, away_ml) VALUES (?, '2026-10-20T16:00:00Z', "
                 "'espn', 'DraftKings', -4.5, 220.5, -190, 160)", (H.G1,))
    conn.commit()


@pytest.fixture
def research_db(data_dir):
    conn = db.connect()
    H.populate(conn)
    _add_research(conn)
    conn.close()
    import sqlite3
    from courtside.stats import db as sdb
    sdb.clear()
    ro = sqlite3.connect(f"file:{data_dir / 'courtside.db'}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    yield ro
    ro.close()


def test_with_without_aggregation(research_db):
    ww = TN.with_without(research_db, "AAA", 102, [101, 102], date(2026, 10, 20))
    assert ww["a_games"] == 3 and ww["seasons"] == ["2026-27"]
    (row,) = ww["rows"]                         # the absent player is never his own teammate
    assert row["player_id"] == 101
    w, wo, d = row["with"], row["without"], row["delta"]
    assert (w["n"], wo["n"]) == (3, 2)          # 0 minutes and no row both count as without
    assert w["min"] == pytest.approx(31.0) and wo["min"] == pytest.approx(33.5)
    assert w["pts"] == pytest.approx(22.0) and wo["pts"] == pytest.approx(27.0)
    assert w["usg"] == pytest.approx(0.21) and wo["usg"] == pytest.approx(0.235)
    assert w["reb"] == 5 and w["ast"] == 4
    # DK: pts + 1.25 reb + 1.5 ast (no double-double)
    assert w["dk"] == pytest.approx(22.0 + 6.25 + 6.0)
    assert d["pts"] == pytest.approx(5.0) and d["min"] == pytest.approx(2.5)
    assert row["small"] is True                 # 2 games without him < 3


def test_small_sample_flag():
    def data(n_without: int) -> dict:
        bp = {1: {}, 2: {}}
        season_of = {}
        for k in range(3 + n_without):
            g = f"g{k}"
            season_of[g] = "2026-27"
            line = {"min": 30.0, "usg": 0.2, "pts": 20.0, "reb": 5.0, "ast": 4.0, "dk": 30.0}
            bp[2][g] = line
            if k < 3:
                bp[1][g] = dict(line)
        return {"by_player": bp, "season_of": season_of}
    assert TN.split(data(2), 1, 2)["small"] is True
    sp = TN.split(data(3), 1, 2)
    assert sp["small"] is False and sp["without"]["n"] == 3 and sp["with"]["n"] == 3
    assert TN.split(data(0), 1, 2)["without"]["min"] is None       # no games without him
    assert TN.split(data(2), 9, 2) is None                          # absent never played


def test_implied_totals():
    assert TN.implied_totals(228.5, -11.5) == (120.0, 108.5)     # home favoured by 11.5
    assert TN.implied_totals(220.0, 3.0) == (108.5, 111.5)       # away favoured by 3
    assert TN.implied_totals(None, -4.5) == (None, None)
    assert TN.implied_totals(210.0, None) == (105.0, 105.0)


def test_market_context_and_panel_on_page(data_dir):
    conn = db.connect()
    H.populate(conn)
    _add_research(conn)
    conn.close()
    from courtside.stats import db as sdb
    sdb.clear()
    app = create_app(data_dir)
    app.state.players_now = lambda: NOW
    c = TestClient(app)
    html = c.get(f"/tonight/{H.DAY}").text
    t = _main_text(html)
    # market strip: the Brief's line and the implied totals (BBB home -4.5, total 220.5)
    assert "Market context" in t and "BBB −4.5" in t and "220.5" in t
    assert "AAA 108.0" in t and "BBB 112.5" in t
    assert "Context, not advice" in t
    # teammate out panel: collapsed, Claxton's absence, Alder's split with a small sample
    assert '<details class="tr-tout">' in html
    assert "Teammate out: with and without" in t and "Without Nicolas Claxton" in t
    assert "3 / 2" in t and "small sample" in t
    assert 'class="is-small"' in html
    assert "31.0 &rarr; 33.5" in t              # minutes with -> without
    # the no-advice rules still hold with the panels on the page
    tt = t.replace(players_view.DISCLAIMER.replace("’", "'"), " ").replace(
        "not betting advice", " ")
    assert not factcheck.BETTING_RE.search(tt), factcheck.BETTING_RE.search(tt)
    assert not factcheck.STORY_BANNED_RE.search(tt), factcheck.STORY_BANNED_RE.search(tt)
    assert not ADVICE_RE.search(tt), ADVICE_RE.search(tt)
