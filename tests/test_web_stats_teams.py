"""Teams + Leaders stats pages (docs/STATS_SITE.md): league table ranks and possession-weighted
ratings, defense vs position buckets, on/off and lineups from stints, the leaders qualifier and
modes, the routes, empty states and the 404.

Tiny fixture league built in the test (conftest's data_dir points COURTSIDE_DATA at a tmp dir):
three teams in the previous season (DEN 2 games, BOS 3, LAL 1), one DEN game this season
(no stints: "not collected yet"), DK salaries for this season only.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import timedelta

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import leaders as L  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.stats import teams as ST  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
Y = S.start_year(CUR)
D1, D2, D3, D4, D5, D6 = 1, 2, 3, 4, 5, 6           # DEN (D1 = the guard who scores)
BG, BF, BC, B4, B5 = 11, 12, 13, 14, 15              # BOS: G, F, C by player_index
LG = 21                                              # LAL guard
BOX = dict(fgm=40, fga=85, fg3m=10, fg3a=30, ftm=15, fta=20, oreb=10, dreb=30, reb=40, ast=25,
           stl=7, blk=5, tov=12, pf=18)


def _tg(conn, gid, team, opp, home, pts, opp_pts, poss, season=PREV, day=None, tov=12):
    b = dict(BOX, tov=tov)
    conn.execute(
        "INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, is_home, "
        "wl, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, pts, "
        "plus_minus) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, team, season, S.REGULAR, day, opp, int(home), "W" if pts > opp_pts else "L",
         240.0, b["fgm"], b["fga"], b["fg3m"], b["fg3a"], b["ftm"], b["fta"], b["oreb"],
         b["dreb"], b["reb"], b["ast"], b["stl"], b["blk"], b["tov"], b["pf"], pts,
         float(pts - opp_pts)))
    conn.execute("INSERT INTO team_games_adv (game_id, team, poss) VALUES (?,?,?)",
                 (gid, team, poss))


def _game(conn, gid, home, away, hp, ap, poss, season=PREV, day=None, tov=None):
    tov = tov or {}
    _tg(conn, gid, home, away, True, hp, ap, poss, season, day, tov.get(home, 12))
    _tg(conn, gid, away, home, False, ap, hp, poss, season, day, tov.get(away, 12))


def _pg(conn, gid, pid, name, team, day, pts, mins=30.0, season=PREV, **kw):
    cols = dict(game_id=gid, player_id=pid, player_name=name, team=team, season=season,
                season_type=S.REGULAR, game_date=day, min=mins, pts=pts, fga=10, fta=0, **kw)
    conn.execute(f"INSERT INTO player_games ({','.join(cols)}) VALUES "
                 f"({','.join('?' * len(cols))})", tuple(cols.values()))
    conn.execute("INSERT INTO player_games_adv (game_id, player_id, usg_pct) VALUES (?,?,?)",
                 (gid, pid, 0.25))


def _stint(conn, gid, no, home5, away5, hp, ap, hposs, aposs, start, end):
    conn.execute("INSERT INTO stints (game_id, stint_no, period, start_sec, end_sec, home5, "
                 "away5, home_pts, away_pts, home_poss, away_poss) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (gid, no, 1, start, end, json.dumps(sorted(home5)), json.dumps(sorted(away5)),
                  hp, ap, hposs, aposs))


def populate(conn: sqlite3.Connection) -> None:
    d1, d2, d3 = f"{Y}-01-10", f"{Y}-01-12", f"{Y}-01-14"
    # g1: DEN 110-100 BOS (100 poss each), g2: BOS 90-120 DEN (120 poss), g3: LAL 100-105 BOS
    _game(conn, "g1", "DEN", "BOS", 110, 100, 100, day=d1, tov={"DEN": 20})
    _game(conn, "g2", "BOS", "DEN", 90, 120, 120, day=d2, tov={"DEN": 20})
    _game(conn, "g3", "LAL", "BOS", 100, 105, 100, day=d3)
    den = {D1: "Dee One", D2: "Dee Two", D3: "Dee Three", D4: "Dee Four", D5: "Dee Five",
           D6: "Dee Six"}
    for pid, nm in den.items():
        if pid != D6:
            _pg(conn, "g1", pid, nm, "DEN", d1, 25 if pid == D1 else 5)
        _pg(conn, "g2", pid, nm, "DEN", d2, 15 if pid == D1 else 5, mins=30.0 if pid != D6 else 10)
    for gid, day, g, f, c in (("g1", d1, 20, 10, 10), ("g2", d2, 10, 0, 4), ("g3", d3, 33, 2, 2)):
        _pg(conn, gid, BG, "Bos Guard", "BOS", day, g)
        _pg(conn, gid, BF, "Bos Forward", "BOS", day, f)
        _pg(conn, gid, BC, "Bos Center", "BOS", day, c, reb=10)    # 10 & 10 in g1: a double-double
    _pg(conn, "g3", LG, "Lal Guard", "LAL", d3, 8)
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team, position) "
                     "VALUES (?,?,?,?,?)",
                     [(BG, "Bos", "Guard", "BOS", "G"), (BF, "Bos", "Forward", "BOS", "F-C"),
                      (BC, "Bos", "Center", "BOS", "C"), (D1, "Dee", "One", "DEN", "G-F"),
                      (LG, "Lal", "Guard", "LAL", "G")])
    # stints: g1 DEN at home, g2 DEN away
    five, bench = [D1, D2, D3, D4, D5], [D1, D2, D3, D4, D6]
    bos = [BG, BF, BC, B4, B5]
    _stint(conn, "g1", 0, five, bos, 10, 5, 10, 10, 0, 300)
    _stint(conn, "g1", 1, bench, bos, 4, 8, 10, 10, 300, 600)
    _stint(conn, "g2", 0, bos, five, 6, 6, 5, 5, 0, 240)
    # this season: one DEN game (no stints yet), DK salaries say LG is an SG
    cday = (TODAY - timedelta(days=1)).isoformat()
    _game(conn, "c1", "DEN", "LAL", 100, 90, 95, season=CUR, day=cday)
    _pg(conn, "c1", D1, "Dee One", "DEN", cday, 30, season=CUR)
    _pg(conn, "c1", LG, "Lal Guard", "LAL", cday, 12, season=CUR)
    conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id, "
                 "player_name, site_player_id, team, positions, salary) VALUES "
                 "(?,?,?,?,?,?,?,?,?,?)", (f"{cday}T15:00:00Z", "dk", "s1", "classic", 70,
                                           "Lal Guard", "dk21", "LAL", "SG/SF", 5000))
    conn.execute("INSERT INTO player_ids (source, source_id, player_id, name) VALUES "
                 "('dk','dk21',?,'Lal Guard')", (LG,))
    conn.execute("INSERT INTO team_coaches (season, team_id, team, coach_id, coach_name, "
                 "coach_type, is_assistant, sort_sequence) VALUES (?,?,?,?,?,?,?,?)",
                 (PREV, 1610612743, "DEN", 9, "Head Honcho", "Head Coach", 1, 1))
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES ('n1', ?, ?, ?, 'BOS', 'DEN', 'scheduled')",
                 (CUR, (TODAY + timedelta(days=2)).isoformat(),
                  f"{(TODAY + timedelta(days=3)).isoformat()}T00:00:00Z"))
    conn.commit()


@pytest.fixture
def conn(data_dir):
    sdb.clear()
    c = db.connect()
    populate(c)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture
def client(conn, data_dir):
    sdb.clear()
    return TestClient(create_app(data_dir))


# ------------------------------------------------------------------ league table
def test_possession_weighted_ratings_and_rank_direction(conn):
    lg = ST.league(conn, PREV)
    den = lg["by_team"]["DEN"]
    assert den["gp"] == 2 and (den["w"], den["l"]) == (2, 0)
    # 100 * (110 + 120) / (100 + 120), not the mean of 110 and 100
    assert den["ortg"] == pytest.approx(100 * 230 / 220)
    assert den["drtg"] == pytest.approx(100 * 190 / 220)
    assert den["net"] == pytest.approx(den["ortg"] - den["drtg"])
    assert den["opp_pts"] == pytest.approx(95.0)
    assert den["pace"] == pytest.approx(48 * 220 / (480 / 5))
    assert den["efg"] == pytest.approx((40 + 5) / 85)
    assert den["home"] == "1-0" and den["road"] == "1-0" and den["l10"] == "2-0"
    assert den["streak"] == "W2"
    # rank 1 = best: high ORtg, LOW DRtg, LOW TOV%, LOW opp pts
    assert den["rk"]["ortg"] == 1 and den["rk"]["drtg"] == 1 and den["rk"]["opp_pts"] == 1
    assert den["rk"]["tov"] == 3                                  # DEN turns it over most
    bos, lal = lg["by_team"]["BOS"], lg["by_team"]["LAL"]
    assert bos["drtg"] == pytest.approx(100 * 330 / 320) and bos["rk"]["drtg"] == 2
    assert lal["rk"]["drtg"] == 3 and lal["rk"]["ortg"] == 2 and bos["rk"]["ortg"] == 3
    assert [r["team"] for r in lg["rows"]][0] == "DEN"            # sorted by net rating
    assert L.rank_values({"a": 3, "b": 3, "c": 1}) == {"a": 1, "b": 1, "c": 3}
    assert L.rank_values({"a": 3, "b": None, "c": 1}, higher_better=False) == {"c": 1, "a": 2}


def test_team_leaders_pick_best_by_direction(conn):
    tl = {c["key"]: c for c in ST.team_leaders(conn, PREV)}
    assert tl["drtg"]["best"]["team"] == "DEN" and tl["drtg"]["next"]["team"] == "BOS"
    assert tl["tov"]["best"]["team"] in ("BOS", "LAL")
    assert ST.team_leaders(conn, "1999-00") == []


# ------------------------------------------------------------------ defense vs position
def test_dvp_index_buckets_and_rank_most_allowed_first(conn):
    d = ST.dvp(conn, "DEN", PREV)
    assert d["basis"] == "index" and d["buckets"] == ("G", "F", "C")
    row = d["row"]
    assert row["all"]["G"] == pytest.approx((20 + 10) / 2)       # BOS guard vs DEN
    assert row["all"]["F"] == pytest.approx(5.0)
    assert row["all"]["C"] == pytest.approx((10 + 12.5 + 1.5 + 4 + 12.5) / 2)   # DD bonus
    # guards: LAL allowed 33/g, BOS (25+15+8)/3 = 16, DEN 15 -> rank 1 = most allowed
    assert ST.dvp(conn, "LAL", PREV)["row"]["rk"]["G"] == 1
    assert ST.dvp(conn, "BOS", PREV)["row"]["rk"]["G"] == 2
    assert row["rk"]["G"] == 3
    assert row["l10"]["G"] == row["all"]["G"]


def test_dvp_dk_positions_this_season(conn):
    pos = L.positions(conn, CUR)
    assert pos["basis"] == "dk" and pos["pos"] == {LG: "SG"}
    d = ST.dvp(conn, "DEN", CUR)
    assert d["buckets"] == L.DK_BUCKETS and d["row"]["all"]["SG"] == pytest.approx(12.0)
    assert d["row"]["all"]["PG"] == 0.0


# ------------------------------------------------------------------ on/off + lineups
def test_onoff_and_lineups_from_stints(conn):
    st = ST.stints(conn, "DEN", PREV)
    assert st["games"] == 2 and st["team_gp"] == 2
    t = st["team"]
    assert t["ortg"] == pytest.approx(100 * 20 / 25) and t["drtg"] == pytest.approx(100 * 19 / 25)
    by = {r["player_id"]: r for r in st["onoff"]}
    assert set(by) == {D1, D2, D3, D4, D5, D6}
    assert by[D1]["on_net"] == pytest.approx(4.0) and by[D1]["off_net"] is None
    assert by[D1]["diff"] is None
    assert by[D5]["on_net"] == pytest.approx(100 * (16 - 11) / 15)
    assert by[D5]["off_net"] == pytest.approx(-40.0)
    assert by[D5]["diff"] == pytest.approx(100 * 5 / 15 + 40)
    assert by[D6]["on_net"] == pytest.approx(-40.0) and by[D6]["name"] == "Dee Six"
    assert by[D1]["min"] == pytest.approx(840 / 60)
    assert st["min_poss"] == 10
    assert [u["ids"] for u in st["lineups"]] == [sorted([D1, D2, D3, D4, D5]),
                                                 sorted([D1, D2, D3, D4, D6])]
    assert st["lineups"][0]["poss"] == pytest.approx(15.0)
    assert st["lineups"][0]["net"] == pytest.approx(100 * 5 / 15)
    assert ST.lineup_min_poss(60) == 50 and ST.lineup_min_poss(3) == 10
    # BOS side of the same stints (home in g2, away in g1)
    bos = ST.stints(conn, "BOS", PREV)
    assert bos["team"]["ortg"] == pytest.approx(100 * 19 / 25)
    # no stints this season yet
    assert ST.stints(conn, "DEN", CUR)["games"] == 0


# ------------------------------------------------------------------ leaders
def test_leaders_qualifier_modes_and_windows(conn):
    b = L.leaders(conn, PREV, stat="pts")
    assert b["early"] and "2 games" in b["rule"]
    names = [r["name"] for r in b["rows"]]
    assert names[0] == "Bos Guard" and b["rows"][0]["value"] == pytest.approx(21.0)
    assert "Lal Guard" not in names                              # 1 game: not qualified
    assert "Dee Six" not in names                                # 1 game, 10 min
    tot = L.leaders(conn, PREV, stat="pts", mode="tot")
    assert tot["rows"][0]["value"] == pytest.approx(63) and "Lal Guard" in [
        r["name"] for r in tot["rows"]]
    p36 = L.leaders(conn, PREV, stat="pts", mode="p36")
    assert p36["rows"][0]["value"] == pytest.approx(63 / 90 * 36)
    l5 = L.leaders(conn, PREV, stat="pts", window="l5")
    assert l5["rows"][0]["gp"] == 3
    dd = L.leaders(conn, PREV, stat="dd")
    assert dd["rows"][0]["name"] == "Bos Center" and dd["rows"][0]["value"] == 1
    ts = L.leaders(conn, PREV, stat="ts")
    assert ts["rows"][0]["value"] == pytest.approx(21 / 20)
    dk = L.leaders(conn, PREV, stat="dk")
    assert dk["rows"][0]["value"] > 0
    assert L.leaders(conn, PREV, stat="nope")["stat"] == "pts"
    assert L.leaders(conn, "1999-00")["rows"] == []
    # later in the season: 40% of the team's games
    q = L.qualifier("pts", "pg", 20)
    assert not q["early"] and "40%" in q["text"]
    assert q["fn"]({"gp": 8, "min": 80}, 20) and not q["fn"]({"gp": 7, "min": 300}, 20)
    q36 = L.qualifier("pts", "p36", 20)
    assert not q36["fn"]({"gp": 10, "min": 100}, 20) and q36["fn"]({"gp": 10, "min": 150}, 20)
    assert L.qualifier("pts", "tot", 20)["fn"]({"gp": 1, "min": 1}, 20)


def test_roster_splits_by_team(conn):
    ro = ST.roster(conn, "DEN", PREV)
    assert ro[0]["player_id"] == D1 and ro[0]["gp"] == 2 and ro[0]["pts"] == pytest.approx(20)
    assert {r["player_id"] for r in ro} == {D1, D2, D3, D4, D5, D6}
    assert ST.roster(conn, "DEN", CUR)[0]["pts"] == pytest.approx(30)


# ------------------------------------------------------------------ routes
def test_teams_route(client):
    html = client.get(f"/teams?season={PREV}").text
    assert '<table data-sortable data-renumber class="tm-league">' in html
    assert 'href="/team/DEN?season=' in html and "Nuggets" in html
    assert 'class="rk"' in html and "/static/pro-teams.css?v=" in html
    assert "possession-weighted" in html
    assert f'<a href="/teams?season={PREV}" class="on" aria-current="true">{PREV}</a>' in html
    r = client.get(f"/teams?season={PREV}&type=playoffs")
    assert r.status_code == 200 and "No playoffs games" in r.text


def test_team_route_full_page(client):
    r = client.get(f"/team/DEN?season={PREV}")
    assert r.status_code == 200
    html = r.text
    assert "Denver Nuggets" in html and "Head Honcho" in html
    assert f'href="/player/{D1}"' in html and 'href="/game/g1"' in html
    assert "Defense vs position" in html and "1st = allows the most DK points" in html
    assert "player-index positions G / F / C" in html
    assert "On / off" in html and 'class="diverge"' in html and "Dee Six" in html
    assert "Top 5-man lineups" in html and "min 10 poss" in html
    assert "Off rating" in html and "Opp pts" in html


def test_team_route_current_season_empty_states(client):
    html = client.get("/team/DEN").text
    assert "Not collected yet." in html                            # no stints this season
    assert "DK roster positions" in html
    assert 'href="/game/n1"' in html and "Upcoming" in html
    html = client.get("/team/DEN?type=playoffs").text
    assert "No playoffs games for the Nuggets" in html
    assert client.get("/team/LAL").status_code == 200


def test_team_404_and_redirect(client):
    assert client.get("/team/XYZ").status_code == 404
    r = client.get(f"/team/den?season={PREV}", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == f"/team/DEN?season={PREV}"


def test_leaders_route(client):
    html = client.get(f"/leaders?season={PREV}").text
    assert "Points per game" in html and 'href="/player/11"' in html
    assert 'class="hbar"' in html and "early season" in html
    assert 'href="/team/BOS"' in html and "Team leaders" in html
    assert f'href="/leaders?season={PREV}&amp;stat=reb"' in html
    html = client.get(f"/leaders?season={PREV}&stat=pts&mode=tot&last=5").text
    assert "Points totals" in html and "Lal Guard" in html and "Last 5" in html
    html = client.get("/leaders?stat=ts&type=playoffs").text
    assert "No qualified players yet." in html
    assert client.get("/leaders?stat=zzz&mode=x&last=99").status_code == 200


# ------------------------------------------------------------------ standings
def test_standings_conference_division_and_gb(conn):
    st = ST.standings(conn, PREV)
    west = st["conferences"]["West"]
    assert [r["team"] for r in west] == ["DEN", "LAL"]           # 2-0, then 0-1
    assert west[0]["gb"] == 0 and west[1]["gb"] == pytest.approx(1.5)
    assert [r["team"] for r in st["conferences"]["East"]] == ["BOS"]
    assert [r["team"] for r in st["divisions"]["Northwest"]["rows"]] == ["DEN"]
    assert st["divisions"]["Atlantic"]["conference"] == "East"
    assert st["divisions"]["Pacific"]["rows"][0]["pos"] == 1
    assert st["rank"]["LAL"] == {"conf": 2, "div": 1, "conference": "West",
                                 "division": "Pacific"}
    assert ST.standings(conn, "1999-00")["rank"] == {}


def test_standings_views_and_team_header(client):
    html = client.get(f"/teams?season={PREV}&view=conf").text
    assert "Western Conference" in html and "Eastern Conference" in html
    assert '<table class="tm-stand">' in html and "1.5" in html and "W2" in html
    assert 'aria-current="true">Standings</a>' in html
    assert f'href="/teams?season={PREV}&amp;view=div"' in html
    assert "tiebreakers are not applied" in html
    html = client.get(f"/teams?season={PREV}&view=div").text
    assert "Northwest" in html and "Pacific" in html and "Atlantic" in html
    html = client.get(f"/teams?season={PREV}&view=bogus").text
    assert 'data-renumber class="tm-league"' in html                # falls back to the table
    assert "No playoffs games" in client.get(f"/teams?season={PREV}&type=playoffs&view=conf").text
    html = client.get(f"/team/DEN?season={PREV}").text
    assert "1st in Northwest" in html and "1st in the West" in html
    html = client.get(f"/team/LAL?season={PREV}").text
    assert "1st in Pacific" in html and "2nd in the West" in html
