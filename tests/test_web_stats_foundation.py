"""Stats dashboard foundation (docs/STATS_SITE.md): site menu, /tonight move, /api/search,
the stub stats routes and their 404s, the macros/filters, seasons and the stats.db cache.

Tiny fixture DB built in the test (conftest's data_dir points COURTSIDE_DATA at a tmp dir).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import warnings
from datetime import date

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import search as SR  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.web import fmt  # noqa: E402
from courtside.web.app import create_app, make_env  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
OLD = S.prev_season(S.prev_season(S.prev_season(CUR)))     # outside the 3-season window
JOKIC, TRADED, RETIRED, BOXONLY = 203999, 1001, 1002, 1003
NAV = ["/", "/tonight", "/games", "/teams", "/players", "/leaders", "/finder", "/dfs", "/health"]


def _pg(conn, gid, pid, name, team, season, day, pts=10):
    conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, season, "
                 "season_type, game_date, min, pts) VALUES (?,?,?,?,?,?,?,?,?)",
                 (gid, pid, name, team, season, S.REGULAR, day, 30.0, pts))


def populate(conn: sqlite3.Connection) -> None:
    y = S.start_year(CUR)
    _pg(conn, "g1", JOKIC, "Nikola Jokic", "DEN", CUR, f"{y}-10-24", 35)
    _pg(conn, "g0", JOKIC, "Nikola Jokic", "DEN", PREV, f"{y}-03-01", 30)
    _pg(conn, "g2", TRADED, "Tom Trade", "BOS", PREV, f"{y}-02-01")
    _pg(conn, "g3", TRADED, "Tom Trade", "NYK", PREV, f"{y}-04-01")      # most recent: NYK
    _pg(conn, "g4", RETIRED, "Old Timer", "LAL", OLD, f"{y - 3}-01-05")
    _pg(conn, "g5", BOXONLY, "Box Only", "MIA", CUR, f"{y}-10-23")
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team) "
                     "VALUES (?,?,?,?)",
                     [(JOKIC, "Nikola", "Jokić", "DEN"), (TRADED, "Tom", "Trade", "NYK"),
                      (RETIRED, "Old", "Timer", "LAL")])
    conn.commit()


@pytest.fixture
def client(data_dir):
    sdb.clear()
    conn = db.connect()
    populate(conn)
    conn.close()
    return TestClient(create_app(data_dir))


def _nav(html: str) -> list[str]:
    nav = re.search(r'<nav class="top-nav"[^>]*>(.*?)</nav>', html, re.S).group(1)
    return re.findall(r'<a href="([^"]+)"', nav)


def _active(html: str) -> str | None:
    m = re.search(r'<a href="([^"]+)" aria-current="page" class="top-link is-active">', html)
    return m.group(1) if m else None


# ------------------------------------------------------------------ chrome
def test_nav_links_order_active_and_footer(client):
    html = client.get("/").text
    assert _nav(html) == NAV
    assert _active(html) == "/"
    assert 'id="gsearch-q"' in html and 'action="/players"' in html
    assert "/static/pro.css?v=" in html and "/static/stats.js?v=" in html
    foot = re.search(r'<nav class="foot-links"[^>]*>(.*?)</nav>', html, re.S).group(1)
    for href in NAV[:-1] + ["/archive", "/health"]:
        assert f'href="{href}"' in foot
    assert "theme-toggle" not in html and "cs-theme" not in html   # one dark theme, no toggle
    for url, want in [("/teams", "/teams"), ("/team/DEN", "/teams"), ("/games", "/games"),
                      ("/players", "/players"), (f"/player/{JOKIC}", "/players"),
                      ("/leaders", "/leaders"), (f"/tonight/{TODAY}", "/tonight"),
                      ("/health", "/health")]:
        assert _active(client.get(url).text) == want, url


# ------------------------------------------------------------------ /tonight move
def test_tonight_route_and_players_redirect(client):
    r = client.get("/tonight", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == f"/tonight/{TODAY}"
    r = client.get("/tonight/2026-10-20")
    assert r.status_code == 200 and "Tonight&rsquo;s players" in r.text
    r = client.get("/players/2026-10-20?sort=name&game=x", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "/tonight/2026-10-20?sort=name&game=x"
    assert client.get("/tonight/2026-13-45").status_code == 404
    assert client.get("/players/yesterday").status_code == 404


# ------------------------------------------------------------------ /api/search
def test_api_search_shape_and_accents(client):
    r = client.get("/api/search", params={"q": "jokic"})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"q", "results"} and d["q"] == "jokic"
    top = d["results"][0]
    assert top == {"type": "player", "id": JOKIC, "name": "Nikola Jokić", "team": "DEN",
                   "season": CUR, "last_game": top["last_game"], "url": f"/player/{JOKIC}",
                   "sub": "DEN"}
    for q in ("JOKIĆ", "Jokić", "nik jok", "nikola"):
        assert client.get("/api/search", params={"q": q}).json()["results"][0]["id"] == JOKIC
    den = client.get("/api/search", params={"q": "den"}).json()["results"][0]
    assert den == {"type": "team", "id": "DEN", "name": "Denver Nuggets", "team": "DEN",
                   "url": "/team/DEN", "sub": "Team"}
    assert client.get("/api/search", params={"q": "nuggets"}).json()["results"][0]["id"] == "DEN"
    assert client.get("/api/search", params={"q": ""}).json()["results"] == []
    assert client.get("/api/search").json()["results"] == []


def test_search_window_team_and_names(data_dir):
    sdb.clear()
    conn = db.connect()
    populate(conn)
    conn.close()
    ro = sdb.connect_ro(data_dir / "courtside.db")
    tom = SR.search(ro, "tom", today=TODAY)
    assert [h["id"] for h in tom] == [TRADED]
    assert tom[0]["team"] == "NYK" and tom[0]["sub"] == f"NYK · {PREV}"   # last game's team
    assert SR.search(ro, "old timer", today=TODAY) == []                 # outside the window
    box = SR.search(ro, "box", today=TODAY)
    assert box[0]["name"] == "Box Only"                                  # box-score name
    assert len(SR.search(ro, "a", today=TODAY)) <= SR.LIMIT
    assert SR.norm("D'Angelo Russell") == "dangelo russell"
    assert SR.norm("P.J. Tucker") == "pj tucker" and SR.norm("Đorđević") == "dordevic"
    ro.close()


# ------------------------------------------------------------------ stub pages + 404s
@pytest.mark.parametrize("url", ["/games", "/games/2026-10-24", "/teams", f"/teams?season={PREV}",
                                 "/team/DEN", "/players", "/players?q=jok", f"/player/{JOKIC}",
                                 "/leaders", "/leaders?season=1999-00"])
def test_stub_routes_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, url
    assert 'class="page-' in r.text and "pro-page" in r.text and '<div class="pro">' in r.text
    assert "None" not in re.sub(r"<script.*?</script>", "", r.text, flags=re.S)


def test_stub_content(client):
    assert "Denver Nuggets" in client.get("/team/DEN").text
    assert "Nikola Jokić" in client.get("/players?q=jokic").text
    page = client.get(f"/player/{JOKIC}").text
    assert "Nikola Jokić" in page or "Nikola Jokic" in page
    html = client.get(f"/teams?season={PREV}").text
    assert f'<a href="/teams?season={PREV}" class="on" aria-current="true">{PREV}</a>' in html


def test_404s(client):
    for url in ("/team/XXX", "/team/12", "/player/999999", "/player/abc", "/games/2026-13-45",
                "/games/yesterday"):
        r = client.get(url)
        assert r.status_code == 404, url
        assert "error-page" in r.text or "page-error" in r.text, url
    r = client.get("/team/den?season=x", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/team/DEN?season=x"


# ------------------------------------------------------------------ filters + macros
def test_rank_color_and_ordinal():
    assert fmt.rank_color(1, 30) == "#2a6fc4"
    assert fmt.rank_color(30, 30) == "#c44a4a"
    assert fmt.rank_color(None) == "#3a3f4b"
    mid = fmt.rank_color(15.5, 30)
    assert mid == "#3a3f4b"
    assert fmt.rank_color(1, 1) == "#2a6fc4"                  # n=1 never divides by zero
    assert [fmt.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th"]
    assert fmt.ordinal(None) == ""
    assert fmt.dollars(11400) == "$11,400" and fmt.dollars(None) == fmt.EM


def test_macros_render():
    env = make_env()
    t = env.from_string(
        '{% import "_stats.html" as s %}'
        '{{ s.rank_chip(3) }}|{{ s.rank_chip(None) }}|{{ s.tile("Pts", 27.4|num(1), rank=2) }}|'
        '{% call s.card("Roster", sub="per game", cls="span-8", flush=True) %}<table></table>{% endcall %}|'
        '{{ s.seg("box", [("trad", "Traditional"), ("adv", "Advanced")], "adv") }}|'
        '{% call s.panel("box", "trad dk", "adv") %}T{% endcall %}|'
        '{{ s.season_seg("2025-26", base="/teams", options=["2026-27", "2025-26"]) }}|'
        '{{ s.empty() }}|{{ s.hbar(5, 10, "5.0") }}|{{ s.range_bar(20, 40, 30, 35) }}|'
        '{{ s.ntd(0.612, kind="pct", nd=1, rank=1) }}|{{ s.hero("Denver Nuggets", badge_team="DEN") }}')
    out = t.render()
    chip = f'<span class="rk" style="background:{fmt.rank_color(3)}" title="Rank 3 of 30">3rd</span>'
    assert out.startswith(chip + "||")                        # None -> no chip
    assert ">2nd</span>" in out and 'class="num-big">27.4<' in out
    assert '<section class="card span-8">' in out and '<span class="sub">per game</span>' in out
    assert '<div class="scroll"><table></table></div>' in out
    assert 'data-seg-target="adv" class="on" aria-pressed="true"' in out
    assert 'data-seg-panel="box" data-seg-value="trad dk" hidden' in out
    assert '<a href="/teams?season=2025-26" class="on" aria-current="true">2025-26</a>' in out
    assert "Not collected yet." in out and 'style="width:50.0%"' in out
    assert 'class="band"' in out and 'class="act"' in out
    assert 'data-v="0.612">61.2%' in out and ">1st</span>" in out
    assert 'class="hero-badge team"' in out and "<h1>Denver Nuggets</h1>" in out
    assert 'class="right"' not in out                         # hero without a call body


# ------------------------------------------------------------------ seasons
def test_seasons():
    assert S.season_for(date(2026, 10, 1)) == "2026-27"
    assert S.season_for(date(2027, 6, 30)) == "2026-27"
    assert S.season_for(date(2027, 8, 15)) == "2026-27"      # off-season: the one just ended
    assert S.season_for(date(2027, 10, 1)) == "2027-28"
    assert S.seasons(date(2026, 10, 25)) == ("2026-27", "2025-26", "2024-25")
    assert S.seasons(date(2000, 1, 5)) == ("1999-00", "1998-99", "1997-98")
    assert S.pick("2025-26", date(2026, 10, 25)) == "2025-26"
    assert S.pick("2019-20", date(2026, 10, 25)) == "2026-27"
    assert S.pick(None, date(2026, 10, 25)) == "2026-27"
    assert S.is_season("1999-00") and not S.is_season("2026-28") and not S.is_season("x")
    assert S.season_type("playoffs") == "Playoffs" and S.season_type(None) == S.REGULAR
    assert S.date_range("2026-27") == (date(2026, 10, 1), date(2027, 9, 30))


# ------------------------------------------------------------------ stats.db cache
def _mkdb(path):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t (x INTEGER)")
    c.execute("INSERT INTO t VALUES (1)")
    c.commit()
    c.close()


def test_cache_invalidates_on_db_change(tmp_path):
    sdb.clear()
    p = tmp_path / "x.db"
    _mkdb(p)
    calls = []

    def fn(conn):
        calls.append(1)
        return conn.execute("SELECT SUM(x) FROM t").fetchone()[0]

    assert sdb.cached(("sum",), fn, p) == 1
    assert sdb.cached(("sum",), fn, p) == 1 and len(calls) == 1          # warm hit
    ro = sdb.connect_ro(p)
    assert sdb.cached(("sum",), fn, ro) == 1 and len(calls) == 1         # conn: same file
    ro.close()
    assert sdb.cached(("other",), fn, str(p)) == 1 and len(calls) == 2   # new key
    w = sqlite3.connect(p)
    w.execute("INSERT INTO t VALUES (41)")
    w.commit()
    w.close()
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))       # coarse clocks
    assert sdb.cached(("sum",), fn, p) == 42 and len(calls) == 3         # dropped, recomputed
    assert sdb.cached(("sum",), fn, p) == 42 and len(calls) == 3
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))         # mtime alone
    assert sdb.cached(("sum",), fn, p) == 42 and len(calls) == 4
    # no file / in-memory: computed, never cached
    assert sdb.cached(("sum",), fn, tmp_path / "missing.db") is None
    mem = sqlite3.connect(":memory:")
    mem.execute("CREATE TABLE t (x INTEGER)")
    mem.execute("INSERT INTO t VALUES (7)")
    assert sdb.cached(("sum",), fn, mem) == 7 and sdb.cached(("sum",), fn, mem) == 7
    assert len(calls) == 6


def test_cache_is_thread_safe(tmp_path):
    sdb.clear()
    p = tmp_path / "y.db"
    _mkdb(p)
    calls = []

    def slow(conn):
        calls.append(1)
        time.sleep(0.05)
        return conn.execute("SELECT x FROM t").fetchone()[0]

    out = []
    ts = [threading.Thread(target=lambda: out.append(sdb.cached(("slow",), slow, p)))
          for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [1] * 8 and len(calls) == 1
