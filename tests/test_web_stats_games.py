"""Games section (docs/STATS_SITE.md): stats/games.py queries, /games scoreboard, and the box
score on /game/<game_id>. Tiny fixture DB built in the test (conftest points COURTSIDE_DATA at
a tmp dir); the sample Brief supplies the existing game page (0022500792, HOU at CHA)."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj.dfs.rules import dk_points  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import games as G  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

SAMPLE = Path(__file__).parent / "fixtures" / "brief" / "sample_brief.json"
BRIEF_DAY = "2026-02-20"
OLD_GAME, OLD_HOME, OLD_AWAY = "0022500792", "CHA", "HOU"   # in the sample Brief (last night)
D1, D2, D3 = "2026-10-23", "2026-10-24", "2026-10-25"
G1, G2, G3, GOT = "0022600001", "0022600002", "0022600003", "0022600004"
NOPID, JOKIC, MURRAY, EDDIE, BENCH = 1628369, 203999, 1627750, 5001, 5002
TIP = "2026-10-25T01:00:00Z"


def _ins(conn, table, **kw):
    cols = ",".join(kw)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values()))


def _team(conn, gid, team, opp, home, day, pts, wl, season="2026-27", **extra):
    row = dict(fgm=40, fga=90, fg3m=10, fg3a=30, ftm=20, fta=25, oreb=10, dreb=34, reb=44, ast=25,
               stl=7, blk=5, tov=12, pf=20, plus_minus=0.0, min=240.0)
    row.update(extra)
    _ins(conn, "team_games", game_id=gid, team=team, season=season, season_type="Regular Season",
         game_date=day, opp=opp, is_home=home, wl=wl, pts=pts, **row)


def _player(conn, gid, pid, name, team, day, minutes, pts, reb=5, ast=3, season="2026-27", **kw):
    row = dict(fgm=8, fga=15, fg3m=2, fg3a=5, ftm=2, fta=2, oreb=1, dreb=reb - 1, stl=1, blk=0,
               tov=2, pf=2, plus_minus=5.0)
    row.update(kw)
    _ins(conn, "player_games", game_id=gid, player_id=pid, player_name=name, team=team,
         season=season, season_type="Regular Season", game_date=day, min=minutes, pts=pts,
         reb=reb, ast=ast, **row)


def populate(conn: sqlite3.Connection) -> None:
    # 2026-27 schedule: night 1 (D1) one final, night 2 (D2) two finals (one OT), night 3 tonight
    games = [(G1, D1, "2026-10-24T00:00:00Z", "DEN", "NOP", "final", 110, 100),
             (G2, D2, "2026-10-24T23:00:00Z", "BOS", "NYK", "final", 99, 101),
             (GOT, D2, "2026-10-25T01:00:00Z", "LAL", "GSW", "final", 120, 118),
             (G3, D3, "2026-10-25T23:30:00Z", "MIA", "ORL", "scheduled", None, None)]
    for gid, day, tip, home, away, st, hs, as_ in games:
        _ins(conn, "games", game_id=gid, season="2026-27", nba_day=day, tipoff_utc=tip, home=home,
             away=away, status=st, home_score=hs, away_score=as_)
    # records come from the games table: DEN 1-0 after D1, NOP 0-1
    facts = {"home": {"periods": [30, 25, 30, 25]}, "away": {"periods": [25, 25, 25, 25]},
             "overtimes": 0,
             "top_performers": [
                 {"id": MURRAY, "name": "Jamal Murray", "team": "DEN", "game_score": 20.0,
                  "line": "30 pts (10-20 FG), 3 reb, 6 ast"},
                 {"id": JOKIC, "name": "Nikola Jokić", "team": "DEN", "game_score": 30.0,
                  "line": "35 pts (12-23 FG), 15 reb, 10 ast"}]}
    _ins(conn, "game_facts", game_id=G1, nba_day=D1, facts_json=json.dumps(facts),
         computed_at="2026-10-24T07:00:00Z")
    ot = {"home": {"periods": [20, 20, 20, 20, 12, 8, 0]}, "away": {"periods": [20, 20, 20, 20, 10, 9, 0]},
          "overtimes": 2, "top_performers": []}                   # a 0-0 seventh period is ignored
    _ins(conn, "game_facts", game_id=GOT, nba_day=D2, facts_json=json.dumps(ot),
         computed_at="2026-10-25T07:00:00Z")
    # box score of G1 (DEN 110, NOP 100)
    _team(conn, G1, "DEN", "NOP", 1, D1, 110, "W", fgm=44, fga=88, fg3m=12, fg3a=30, ftm=10, fta=22,
          oreb=10, dreb=35, reb=45, tov=10)
    _team(conn, G1, "NOP", "DEN", 0, D1, 100, "L", fgm=38, fga=90, fg3m=8, fg3a=28, ftm=16, fta=20,
          oreb=8, dreb=30, reb=38, tov=14)
    _ins(conn, "team_games_adv", game_id=G1, team="DEN", off_rating=115.0, def_rating=105.0,
         net_rating=10.0, pace=96.0, efg_pct=0.57, ts_pct=0.6, tov_pct=0.09, oreb_pct=0.25)
    _player(conn, G1, JOKIC, "Nikola Jokic", "DEN", D1, 36.5, 35, 15, 10, fgm=12, fga=23)
    _player(conn, G1, MURRAY, "Jamal Murray", "DEN", D1, 34.0, 30, 3, 6)
    _player(conn, G1, BENCH, "Bench Guy", "DEN", D1, 12.0, 2, 1, 0)
    _player(conn, G1, EDDIE, "Eddie Nop", "NOP", D1, 30.0, 20, 4, 2)
    _ins(conn, "player_games", game_id=G1, player_id=7001, player_name="Idle Guy", team="DEN",
         season="2026-27", season_type="Regular Season", game_date=D1, min=0.0)
    _ins(conn, "player_games_adv", game_id=G1, player_id=JOKIC, off_rating=130.0, def_rating=100.0,
         net_rating=30.0, usg_pct=0.3, ts_pct=0.7, efg_pct=0.65, ast_pct=0.4, reb_pct=0.3, pie=0.3)
    _ins(conn, "player_starts", game_id=G1, player_id=JOKIC, team="DEN", source="sdv")
    # DFS: Jokic is mapped by the DK id map (salary + projection + ownership); Murray only by name
    _ins(conn, "dfs_slates", slate_id="900", site="dk", nba_day=D1, game_type_id=70, tag="main",
         game_ids_json=json.dumps([G1]), lock_utc="2026-10-24T00:00:00Z")
    _ins(conn, "dfs_slates", slate_id="901", site="dk", nba_day=D1, game_type_id=81, tag="showdown",
         game_ids_json=json.dumps([G1]), lock_utc="2026-10-24T00:00:00Z")
    _ins(conn, "player_ids", source="dk", source_id="dk-jokic", player_id=JOKIC, name="Nikola Jokic")
    for slate, gt, pid, name, sal, cap in [("900", 70, "dk-jokic", "Nikola Jokic", 11000, "2026-10-23T20:00:00Z"),
                                           ("900", 70, "dk-jokic", "Nikola Jokic", 11400, "2026-10-23T23:30:00Z"),
                                           ("900", 70, "dk-jokic", "Nikola Jokic", 99999, "2026-10-24T05:00:00Z"),
                                           ("900", 70, "dk-x", "Jamal Murray", 8000, "2026-10-23T23:30:00Z"),
                                           ("901", 81, "dk-s", "Eddie Nop", 1234, "2026-10-23T23:30:00Z")]:
        _ins(conn, "dfs_salaries", captured_at=cap, site="dk", slate_id=slate, game_type="x",
             game_type_id=gt, player_name=name, site_player_id=pid, salary=sal, game_id=G1)
    for rid, created, ok in [("r-early", "2026-10-23T20:00:00Z", 1), ("r-late", "2026-10-23T23:00:00Z", 1),
                             ("r-bad", "2026-10-23T23:30:00Z", 0), ("r-after", "2026-10-24T01:00:00Z", 1)]:
        _ins(conn, "dfs_runs", run_id=rid, slate_id="900", kind="lock10", created_at=created, ok=ok)
        for i, (pid, mean) in enumerate([(JOKIC, {"r-early": 50.0, "r-late": 55.5, "r-bad": 1.0, "r-after": 2.0}[rid])]):
            cur = conn.execute(
                "INSERT INTO player_projections (game_id, player_id, run_id, created_at, model_version) "
                "VALUES (?,?,?,?,?)", (G1, pid, rid, created, "m"))
            _ins(conn, "player_projection_stats", projection_id=cur.lastrowid, stat="dk", cond="played", mean=mean)
    for cid, pct in [("c1", 40.0), ("c2", 60.0)]:
        _ins(conn, "dfs_contests", contest_id=cid, slate_id="900", name=cid, captured_at="x")
        _ins(conn, "dfs_contest_ownership", contest_id=cid, player_id=JOKIC, name="Nikola Jokic", pct_drafted=pct)
    # an older game (the sample Brief's) with a box score but no `games` row
    _team(conn, OLD_GAME, OLD_HOME, OLD_AWAY, 1, "2026-02-19", 105, "W", season="2025-26")
    _team(conn, OLD_GAME, OLD_AWAY, OLD_HOME, 0, "2026-02-19", 99, "L", season="2025-26")
    _player(conn, OLD_GAME, 9001, "Charlie Hornet", OLD_HOME, "2026-02-19", 30.0, 25, season="2025-26")
    _player(conn, OLD_GAME, 9002, "Hank Rocket", OLD_AWAY, "2026-02-19", 28.0, 22, season="2025-26")
    conn.commit()


@pytest.fixture
def conn(data_dir):
    sdb.clear()
    c = db.connect()
    populate(c)
    yield c
    c.close()


@pytest.fixture
def client(data_dir):
    sdb.clear()
    c = db.connect()
    populate(c)
    c.close()
    d = data_dir / "briefs"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE, d / f"{BRIEF_DAY}.json")
    return TestClient(create_app(data_dir))


def visible(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ------------------------------------------------------------------ queries
def test_scoreboard_cards(conn):
    sb = G.scoreboard(conn, D2)
    assert [g["game_id"] for g in sb] == [G2, GOT]
    g = sb[0]
    assert (g["home"], g["away"], g["home_score"], g["away_score"]) == ("BOS", "NYK", 99, 101)
    assert g["winner"] == "away" and g["status"] == "final" and g["ot"] == 0
    assert sb[1]["ot"] == 2 and G.ot_label(sb[1]["ot"]) == "2OT"          # the 0-0 period is not an OT
    first = G.scoreboard(conn, D1)[0]
    assert first["winner"] == "home" and first["ot"] == 0
    assert [p["name"] for p in first["top"]] == ["Nikola Jokić", "Jamal Murray"]   # by game score
    assert first["top"][0]["pts"] == "35"


def test_records_to_date(conn):
    g = G.scoreboard(conn, D1)[0]
    assert (g["home_rec"], g["away_rec"]) == ("1-0", "0-1")
    # a night after: DEN and NOP records do not move; BOS/NYK/LAL/GSW are one game in
    sb = G.scoreboard(conn, D2)
    assert (sb[0]["home_rec"], sb[0]["away_rec"]) == ("0-1", "1-0")
    # tonight's scheduled game shows records going in
    tonight = G.scoreboard(conn, D3)[0]
    assert tonight["status"] == "scheduled" and tonight["winner"] is None
    assert tonight["home_rec"] is None and tonight["top"] == []
    assert G.records(conn, "2026-27", D1, True)["DEN"] == (1, 0)
    assert "DEN" not in G.records(conn, "2026-27", D1, False)


def test_scoreboard_falls_back_to_team_games(conn):
    sb = G.scoreboard(conn, "2026-02-19")
    assert len(sb) == 1 and sb[0]["game_id"] == OLD_GAME
    assert (sb[0]["home"], sb[0]["away"], sb[0]["home_score"]) == (OLD_HOME, OLD_AWAY, 105)
    assert G.scoreboard(conn, "2026-01-01") == [] and G.scoreboard(None, D1) == []


def test_days(conn):
    assert G.latest_final_day(conn) == D2
    assert G.adjacent_days(conn, D2) == (D1, D3)
    assert G.adjacent_days(conn, "2026-10-26")[0] == D3
    assert G.adjacent_days(conn, "2026-10-01")[0] == "2026-02-19"
    assert G.latest_final_day(None) is None


def test_box_score_rows_and_totals(conn):
    b = G.game_box(conn, G1)
    assert (b["home"], b["away"], b["day"]) == ("DEN", "NOP", D1)
    away, home = b["teams"]
    assert (away["team"], home["team"]) == ("NOP", "DEN")
    assert (away["score"], home["score"]) == (100, 110) and home["wl"] == "W"
    # starters first, then minutes; the zero-minute player is listed as DNP
    assert [p["name"] for p in home["players"]] == ["Nikola Jokic", "Jamal Murray", "Bench Guy"]
    assert home["players"][0]["starter"] and not home["players"][1]["starter"]
    assert home["dnp"] == ["Idle Guy"] and b["has_starts"]
    assert home["players"][0]["adv"]["usg_pct"] == 0.3 and home["players"][1]["adv"] == {}
    assert home["tot"]["pts"] == 110
    assert b["periods"] == {"labels": ["Q1", "Q2", "Q3", "Q4"], "home": [30, 25, 30, 25],
                            "away": [25, 25, 25, 25]}


def test_box_dk_points_use_the_scoring_rules(conn):
    home = G.game_box(conn, G1)["teams"][1]
    row = conn.execute("SELECT * FROM player_games WHERE game_id=? AND player_id=?", (G1, JOKIC)).fetchone()
    assert home["players"][0]["dk"] == dk_points(row) > 50          # a triple-double
    assert home["dk_total"] == pytest.approx(sum(p["dk"] for p in home["players"]))


def test_four_factors(conn):
    b = G.game_box(conn, G1)
    ff = {f["label"]: f for f in b["four_factors"]}
    # NOP has no adv row: computed from the box; DEN uses the stored adv values
    assert ff["eFG%"]["away"] == pytest.approx((38 + 0.5 * 8) / 90)
    assert ff["eFG%"]["home"] == 0.57 and ff["eFG%"]["better"] == "home"
    assert ff["TOV%"]["away"] == pytest.approx(14 / (90 + 0.44 * 20 + 14)) and ff["TOV%"]["better"] == "home"
    assert ff["OREB%"]["away"] == pytest.approx(8 / (8 + 35)) and ff["OREB%"]["home"] == 0.25
    assert ff["FT rate"]["away"] == pytest.approx(20 / 90) and ff["FT rate"]["home"] == pytest.approx(22 / 88)
    assert ff["Off rating"]["away"] is None and ff["Off rating"]["home"] == 115.0


def test_dfs_layer(conn):
    home = G.game_box(conn, G1)["teams"][1]
    jok, mur = home["players"][0], home["players"][1]
    assert jok["salary"] == 11400                      # newest capture at or before tip-off, classic only
    assert jok["proj"] == 55.5                         # latest ok run created before tip-off
    assert jok["own"] == pytest.approx(50.0)           # mean over the slate's contests
    assert jok["diff"] == pytest.approx(jok["dk"] - 55.5)
    assert jok["value"] == pytest.approx(jok["dk"] / 11.4)
    assert mur["salary"] == 8000                       # matched by normalised name
    assert mur["proj"] is None and mur["own"] is None and mur["diff"] is None
    assert G.game_box(conn, G1)["has_dfs"]
    away = G.game_box(conn, G1)["teams"][0]["players"][0]
    assert away["salary"] is None                      # only a showdown price exists for him


def test_box_missing(conn):
    assert G.game_box(conn, "0029999999") is None and G.game_box(None, G1) is None
    assert G.dfs_for_game(conn, "0029999999") == {}


# ------------------------------------------------------------------ scoreboard routes
def test_scoreboard_page(client):
    r = client.get(f"/games/{D1}")
    assert r.status_code == 200 and "pro-page" in r.text and '<div class="pro">' in r.text
    assert "/static/pro-games.css?v=" in r.text
    t = visible(r.text)
    assert "DEN" in t and "NOP" in t and "110" in t and "Final" in t
    assert f'href="/game/{G1}"' in r.text
    assert "1–0" in t and "0–1" in t                                   # records
    assert "Jokić 35" in t                                              # top performer
    assert 'class="row win"' in r.text and 'class="row lose"' in r.text
    assert f'href="/games/{D2}"' in r.text and "None" not in t


def test_scoreboard_overtime_and_links(client):
    html = client.get(f"/games/{D2}").text
    assert "Final/2OT" in visible(html) and "Final/OT" not in visible(html)
    assert f'href="/games/{D1}"' in html and f'href="/games/{D3}"' in html


def test_default_night_is_the_latest_with_finals(client):
    html = client.get("/games").text
    assert f'href="/games/{D2}"' in html and 'aria-current="true">Sat, Oct 24' in html
    assert f'/game/{GOT}' in html and f'/game/{G3}' not in html


def test_scoreboard_scheduled_night(client):
    r = client.get(f"/games/{D3}")
    assert r.status_code == 200
    t = visible(r.text)
    assert "ORL" in t and "MIA" in t and "PM ET" in t and "Preview" in t
    assert f'href="/game/{G3}"' in r.text and "to play" in t
    assert 'aria-disabled="true"' in r.text                             # no later night


def test_scoreboard_empty_state(client):
    r = client.get("/games/2026-12-25")
    assert r.status_code == 200 and "empty-state" in r.text and "No games on this night" in r.text
    assert client.get("/games/2026-13-45").status_code == 404


def test_scoreboard_without_database(data_dir):
    r = TestClient(create_app(data_dir)).get("/games")
    assert r.status_code == 200 and "No games on this night" in r.text


# ------------------------------------------------------------------ game page
def test_game_page_has_box_score_and_keeps_every_section(client):
    html = client.get(f"/game/{OLD_GAME}").text
    assert 'id="box-score"' in html and "/static/pro-games.css?v=" in html
    for marker in ("game-hero", "Game flow", "How it unfolded", "Scoring runs", 'id="impact-title"',
                   "Unsung hero candidates", 'id="clutch-title"', "Notable lines", "Incidents"):
        assert marker in html, marker
    assert html.index("game-hero") < html.index('id="box-score"') < html.index("flow-title")
    assert "Charlie Hornet" in html and "Hank Rocket" in html and 'href="/player/9001"' in html
    assert 'href="/team/CHA"' in html and 'href="/team/HOU"' in html


def test_game_page_box_views_and_sorting(client):
    # the box is only on pages the Brief knows a game of: use the Brief's game
    html = client.get(f"/game/{OLD_GAME}").text
    assert html.count("data-sortable") == 6                              # 3 views x 2 teams
    for v in ("trad", "adv", "dfs"):
        assert f'data-seg-target="{v}"' in html
    assert re.search(r'data-seg-value="trad">', html)
    assert re.search(r'data-seg-value="adv" hidden>', html) and re.search(r'data-seg-value="dfs" hidden>', html)
    assert 'data-v="25"' in html                                         # sortable numeric cells
    assert "Four factors" in html and "Team totals" in html
    assert "No DraftKings salary" in html                                # no DFS data for that game
    assert "None" not in visible(html)


def test_game_box_hook_and_dfs_view(client, data_dir):
    from courtside.web import stats_games
    store = client.app.state.store
    box = stats_games.game_box(store, G1)
    assert box is not None and box["has_dfs"]
    assert stats_games.game_box(store, "0029999999") is None
    # render the partial for a game with DFS data through the real template environment
    from courtside.web.app import make_env
    html = make_env(False).get_template("_game_box.html").render(box=box)
    t = visible(html)
    assert "$11,400" in t and "55.5" in t and "50.0%" in t and "Nikola Jokic" in t
    assert 'class="stm"' in html and "Did not play: Idle Guy" in t
    assert "href=\"/team/DEN\"" in html and f'href="/player/{JOKIC}"' in html
    assert "–" in t                                                       # missing values are en dashes


def test_preview_page_gets_no_box(client):
    r = client.get("/game/0022500802")                                   # tonight's game in the Brief
    assert r.status_code == 200 and 'id="box-score"' not in r.text
