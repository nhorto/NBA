"""Stats tools (docs/STATS_SITE.md): the stat finder (whitelist, every condition type, streak
screens, the result cap, the timeout guard) and the lineup explorer (on/off math, WOWY,
min-possession filtering), plus both routes.

Tiny fixture league: previous season, DEN vs BOS, 12 games with a fixed rest pattern
(rest None,0,1,2,0,1,2,0,1,2,0,1), DEN at home in even games, DEN loses games 0/3/6/9.
  Ace (A, DEN, starts) scores ACE_PTS; Bench (B, DEN, never starts) 8 a game, 25 in game 5;
  Big (BOS, starts) 12 & 12 every game, a triple-double in game 4.
This season: one DEN home game vs LAL (Ace 33, DK salary $7,500, real own 22.5%).
Stints: g00 (DEN home) and g01 (DEN away) with players A..F.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import warnings
from datetime import date, timedelta

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import finder as F  # noqa: E402
from courtside.stats import lineups as LU  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
WINDOW = S.seasons(TODAY)
Y = S.start_year(PREV)
A, B, C, D, E, FF = 101, 102, 103, 104, 105, 106     # DEN (A = Ace, B = Bench)
BIG = 201
ACE_PTS = [10, 22, 25, 30, 18, 21, 35, 24, 20, 26, 28, 40]
OFFSETS = [0, 1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 21]
REST = [None, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
CDAY = (TODAY - timedelta(days=1)).isoformat()


def _tg(conn, gid, team, opp, home, wl, season, day, pts):
    conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, "
                 "is_home, wl, min, pts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (gid, team, season, S.REGULAR, day, opp, int(home), wl, 240.0, pts))


def _pg(conn, gid, pid, name, team, season, day, **kw):
    cols = dict(game_id=gid, player_id=pid, player_name=name, team=team, season=season,
                season_type=S.REGULAR, game_date=day, min=30.0, pts=0, reb=0, ast=0, stl=0,
                blk=0, tov=0, fg3m=0, fg3a=0, fgm=4, fga=10, ftm=0, fta=0, plus_minus=0.0)
    cols.update(kw)
    conn.execute(f"INSERT INTO player_games ({','.join(cols)}) VALUES "
                 f"({','.join('?' * len(cols))})", tuple(cols.values()))


def _stint(conn, gid, no, home5, away5, hp, ap, hposs, aposs, start, end):
    conn.execute("INSERT INTO stints (game_id, stint_no, period, start_sec, end_sec, home5, "
                 "away5, home_pts, away_pts, home_poss, away_poss) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (gid, no, 1, start, end, json.dumps(sorted(home5)), json.dumps(sorted(away5)),
                  hp, ap, hposs, aposs))


def populate(conn: sqlite3.Connection) -> None:
    d0 = date(Y, 11, 1)
    for k in range(12):
        gid, day = f"g{k:02d}", (d0 + timedelta(days=OFFSETS[k])).isoformat()
        den_home, den_win = k % 2 == 0, k % 3 != 0
        _tg(conn, gid, "DEN", "BOS", den_home, "W" if den_win else "L", PREV, day, 110)
        _tg(conn, gid, "BOS", "DEN", not den_home, "L" if den_win else "W", PREV, day, 100)
        _pg(conn, gid, A, "Ace Scorer", "DEN", PREV, day, pts=ACE_PTS[k], reb=5, ast=5,
            fg3m=3 if k >= 7 else 1, fgm=15 if k == 11 else 8, fga=20 if k == 11 else 16)
        _pg(conn, gid, B, "Bench Guy", "DEN", PREV, day, pts=25 if k == 5 else 8, min=18.0)
        _pg(conn, gid, BIG, "Big Man", "BOS", PREV, day, pts=12, reb=12, ast=10 if k == 4 else 2,
            fgm=5)
        _pg(conn, gid, 999, "Did Not Play", "DEN", PREV, day, pts=0, min=0.0)
        conn.executemany("INSERT INTO player_starts (game_id, player_id, team, source) VALUES "
                         "(?,?,?, 'sdv')", [(gid, A, "DEN"), (gid, BIG, "BOS")])
    # this season: DEN 1-0 at home vs LAL
    _tg(conn, "c1", "DEN", "LAL", True, "W", CUR, CDAY, 120)
    _tg(conn, "c1", "LAL", "DEN", False, "L", CUR, CDAY, 100)
    _pg(conn, "c1", A, "Ace Scorer", "DEN", CUR, CDAY, pts=33, reb=5, ast=5, fgm=12, fga=20)
    conn.execute("INSERT INTO dfs_slates (slate_id, site, nba_day, game_type_id) VALUES "
                 "('s1', 'dk', ?, 70)", (CDAY,))
    conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id,"
                 " player_name, site_player_id, salary) VALUES (?,?,?,?,?,?,?,?)",
                 (f"{CDAY}T15:00:00Z", "dk", "s1", "classic", 70, "Ace Scorer", "dk101", 7500))
    conn.execute("INSERT INTO player_ids (source, source_id, player_id, name) VALUES "
                 "('dk', 'dk101', ?, 'Ace Scorer')", (A,))
    conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, captured_at) VALUES "
                 "('k1', 's1', ?)", (f"{CDAY}T23:00:00Z",))
    conn.execute("INSERT INTO dfs_contest_ownership (contest_id, player_id, pct_drafted) VALUES "
                 "('k1', ?, 22.5)", (A,))
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team) VALUES "
                     "(?,?,?,'DEN')", [(A, "Ace", "Scorer"), (B, "Bench", "Guy"),
                                       (C, "Cee", "Three"), (D, "Dee", "Four"),
                                       (E, "Eee", "Five"), (FF, "Eff", "Six")])
    bos = [BIG, 202, 203, 204, 205]
    # g00 DEN home: A-E +10/+5 on 10/10 poss (300 s); A-D+F 4/8 on 10/10 (300 s)
    _stint(conn, "g00", 0, [A, B, C, D, E], bos, 10, 5, 10, 10, 0, 300)
    _stint(conn, "g00", 1, [A, B, C, D, FF], bos, 4, 8, 10, 10, 300, 600)
    # g01 DEN away: B-F score 9, allow 6 on 5/5 poss (240 s)
    _stint(conn, "g01", 0, bos, [B, C, D, E, FF], 6, 9, 5, 5, 0, 240)
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


def _ctx(**kw) -> F.Context:
    kw.setdefault("seasons", WINDOW)
    return F.Context(**kw)


def _find(conn, conds=(), **kw):
    sort = kw.pop("sort", "date")
    group = kw.pop("group", False)
    cap = kw.pop("cap", F.CAP)
    conds = [c if isinstance(c, F.Cond) else F.Cond.make(*c) for c in conds]
    return F.game_search(conn, conds, _ctx(**kw), WINDOW, sort=sort, group=group, cap=cap)


# ------------------------------------------------------------------ finder: whitelist
def test_whitelist_rejects_unknown_columns_operators_and_values(conn):
    for bad in ("pts; DROP TABLE player_games", "pg.pts", "player_name", "", None, "PTS"):
        with pytest.raises(F.FinderError):
            F.Cond.make(bad, "ge", 1)
    for op in ("like", ">=", "or", "", None, "GE"):
        with pytest.raises(F.FinderError):
            F.Cond.make("pts", op, 1)
    for v in ("abc", "1 OR 1=1", "nan", "inf", "1e9", None, True):
        with pytest.raises(F.FinderError):
            F.Cond.make("pts", "ge", v)
    with pytest.raises(F.FinderError):
        F.Context(seasons=WINDOW, team="XXX").clauses()
    with pytest.raises(F.FinderError):
        F.Context(seasons=WINDOW, ha="1=1").clauses()
    with pytest.raises(F.FinderError):
        _find(conn, [("pts", "ge", 1)], sort="pts DESC; --")
    with pytest.raises(F.FinderError):
        F.game_search(conn, ["pts >= 0"], _ctx(), WINDOW)
    # the value is bound, never part of the SQL text
    frag, args = F.Cond.make("pts", "ge", "30").sql()
    assert frag == "pg.pts >= ?" and args == (30.0,)
    # the query-param parser drops bad rows with a notice
    conds, errs = F.parse_conds({"s1": "pts", "o1": "ge", "v1": "30", "s2": "evil", "v2": "1",
                                 "s3": "reb", "o3": "xx", "v3": "2", "s4": "salary", "v4": "1",
                                 "s5": "ast", "v5": "", "sal_max": "x"})
    assert conds == [F.Cond("pts", "ge", 30.0)]
    assert len(errs) == 4 and all("ignored" in e for e in errs)


# ------------------------------------------------------------------ finder: conditions
def test_stat_comparisons(conn):
    assert _find(conn, [("pts", "ge", 30)])["count"] == 4             # 30, 35, 40 + 33 this year
    assert _find(conn, [("pts", "gt", 35)])["count"] == 1
    assert _find(conn, [("pts", "eq", 22)])["count"] == 1
    assert _find(conn, [("pts", "lt", 10)])["count"] == 11             # Bench's 8s
    assert _find(conn, [("pts", "le", 10), ("min", "ge", 25)])["count"] == 1   # Ace's 10
    assert _find(conn, [("reb", "ge", 10), ("ast", "ge", 10)])["count"] == 1
    assert _find(conn, [("dd", "ge", 2)])["count"] == 12               # Big every game
    assert _find(conn, [("dd", "ge", 3)])["rows"][0]["game_id"] == "g04"
    assert _find(conn, [("pra", "ge", 50)])["count"] == 1
    assert _find(conn, [("fg3m", "ge", 3), ("pts", "ge", 20)])["count"] == 5
    r = _find(conn, [("fg_pct", "ge", 70)])                            # percent input
    assert [x["game_id"] for x in r["rows"]] == ["g11"]
    # DK points from proj.dfs.rules.dk_points: 40 + 1.5 + 6.25 + 7.5
    r = _find(conn, [("dk", "ge", 50)])
    assert r["count"] == 1 and r["rows"][0]["dk"] == pytest.approx(55.25)
    # + Ace's 35 (49.25), his 33 this season (46.75) and Big's triple-double (46.5)
    assert _find(conn, [("dk", "ge", 46.5)])["count"] == 4
    assert _find(conn, [("min", "ge", 0)])["count"] == 37               # DNPs never match


def test_context_filters(conn):
    p30 = [("pts", "ge", 30)]
    assert _find(conn, p30, ha="home")["count"] == 2                   # g06 + this season
    assert _find(conn, p30, ha="away")["count"] == 2
    assert _find(conn, p30, wl="l")["count"] == 2                      # g03, g06
    assert _find(conn, p30, opp="BOS")["count"] == 3
    assert _find(conn, p30, opp="LAL")["count"] == 1
    assert _find(conn, [("dd", "ge", 2)], team="BOS")["count"] == 12
    assert _find(conn, [("dd", "ge", 2)], team="DEN")["count"] == 0
    assert _find(conn, [("pts", "ge", 25)], role="bench")["count"] == 1
    assert _find(conn, [("pts", "ge", 25)], role="starter")["count"] == 6
    # c1 has no start data: neither starter nor bench
    assert _find(conn, [("pts", "ge", 33)], role="starter")["count"] == 2
    assert _find(conn, [("pts", "ge", 33)], role="bench")["count"] == 0
    ace = [("pts", "ge", 0), ("min", "ge", 25)]
    assert _find(conn, ace, team="DEN", rest="0")["count"] == 4        # games 1, 4, 7, 10
    assert _find(conn, ace, team="DEN", rest="1")["count"] == 4
    assert _find(conn, ace, team="DEN", rest="2")["count"] == 3        # 2+ days
    assert _find(conn, p30, seasons=(CUR,))["count"] == 1
    assert _find(conn, p30, seasons=(PREV,))["count"] == 3
    assert _find(conn, p30, stype=S.PLAYOFFS)["count"] == 0
    assert _find(conn, [("dd", "ge", 2)], player=BIG)["count"] == 12
    assert _find(conn, p30, player=B)["count"] == 0


def test_dfs_ranges_and_result_rows(conn):
    r = _find(conn, [("salary", "le", 7900)])
    assert r["count"] == 1
    row = r["rows"][0]
    assert row["salary"] == 7500 and row["own"] == pytest.approx(22.5)
    assert row["value"] == pytest.approx(row["dk"] / 7.5)
    assert row["opp"] == "LAL" and row["is_home"] == 1 and row["wl"] == "W"
    assert _find(conn, [("salary", "ge", 8000)])["count"] == 0
    assert _find(conn, [("own", "ge", 20), ("own", "le", 25)])["count"] == 1
    assert _find(conn, [("own", "ge", 30)])["count"] == 0
    conds, errs = F.parse_conds({"sal_min": "7,000", "sal_max": "$8000", "own_min": "10"})
    assert not errs and len(conds) == 3 and _find(conn, conds)["count"] == 1


def test_sort_group_and_cap(conn):
    r = _find(conn, [("pts", "ge", 20)], sort="pts")
    assert [x["pts"] for x in r["rows"][:3]] == [40, 35, 33]
    r = _find(conn, [("pts", "ge", 20)])                               # newest first
    assert r["rows"][0]["game_id"] == "c1"
    total = r["count"]
    capped = _find(conn, [("pts", "ge", 20)], cap=3)
    assert capped["capped"] and len(capped["rows"]) == 3 and capped["count"] == total
    assert not r["capped"] and len(r["rows"]) == total
    g = _find(conn, [("pts", "ge", 8)], group=True)
    by = {x["player_id"]: x for x in g["rows"]}
    assert by[A]["games"] == 13 and by[B]["games"] == 12 and by[BIG]["games"] == 12
    assert by[A]["last"] == CDAY and by[A]["team"] == "DEN"
    assert g["count"] == 37 and g["groups"] == 3
    assert [x["player_id"] for x in g["rows"]][0] == A                 # most games first
    gc = _find(conn, [("pts", "ge", 8)], group=True, cap=2)
    assert gc["capped"] and len(gc["rows"]) == 2 and gc["groups"] == 3
    assert F.CAP == 500


def test_timeout_guard(conn):
    slow = ("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) "
            "SELECT MAX(x) FROM r")
    t0 = time.monotonic()
    with pytest.raises(F.QueryTimeout):
        F._run(conn, slow, [], timeout=0.05)
    assert time.monotonic() - t0 < 5
    assert F._run(conn, "SELECT 1 AS one", [])[0]["one"] == 1        # handler removed after


# ------------------------------------------------------------------ finder: streaks
def test_streak_screens(conn):
    prev = F.Context(seasons=(PREV,))
    r = F.streaks(conn, F.Screen.make("pts", "ge", 20, 8, 10), prev, WINDOW)
    assert [x["player_id"] for x in r["rows"]] == [A]
    ace = r["rows"][0]
    assert (ace["hits"], ace["n"], ace["gp"], ace["run"]) == (9, 10, 12, 7)
    assert [s["v"] for s in ace["seq"]] == ACE_PTS[2:]                 # oldest -> newest
    assert [s["hit"] for s in ace["seq"]].count(False) == 1
    assert ace["avg"] == pytest.approx(sum(ACE_PTS[2:]) / 10)
    # across the window his last 10 includes this season's 33
    r = F.streaks(conn, F.Screen.make("pts", "ge", 20, 8, 10), _ctx(), WINDOW)
    assert r["rows"][0]["run"] == 8 and r["rows"][0]["last"] == CDAY
    assert F.streaks(conn, F.Screen.make("pts", "ge", 20, 10, 10), prev, WINDOW)["count"] == 0
    # "in 5 straight": k == n
    r = F.streaks(conn, F.Screen.make("pts", "le", 8, 5, 5), prev, WINDOW)
    assert [x["player_id"] for x in r["rows"]] == [B] and r["rows"][0]["run"] == 5
    r = F.streaks(conn, F.Screen.make("dd", "ge", 2, 4, 5), prev, WINDOW)
    assert [x["player_id"] for x in r["rows"]] == [BIG]
    # minimum games: 20 > the 12 anyone has
    assert F.streaks(conn, F.Screen.make("pts", "ge", 0, 1, 5, 20), prev, WINDOW)["count"] == 0
    assert F.streaks(conn, F.Screen.make("pts", "ge", 0, 1, 5), prev, WINDOW)["count"] == 3
    # team filter + cap
    r = F.streaks(conn, F.Screen.make("pts", "ge", 0, 1, 5), F.Context(seasons=(PREV,),
                  team="DEN"), WINDOW, cap=1)
    assert r["capped"] and len(r["rows"]) == 1 and r["count"] == 2
    for bad in (("evil", "ge", 1, 1, 1), ("pts", "gt", 1, 1, 1), ("pts", "ge", "x", 1, 1),
                ("pts", "ge", 1, 5, 4), ("pts", "ge", 1, 0, 4), ("pts", "ge", 1, 1, 99),
                ("pts", "ge", 1, "a", 4), ("salary", "ge", 1, 1, 1)):
        with pytest.raises(F.FinderError):
            F.Screen.make(*bad)
    assert F.Screen.make("pts", "ge", 20, 8, 10).label() == "PTS ≥ 20 in at least 8 of the last 10"
    assert F.Screen.make("fg3m", "ge", 3, 5, 5).label() == "3PM ≥ 3 in each of the last 5"


# ------------------------------------------------------------------ lineups
def test_lineup_on_off_math(conn):
    d = LU.load(conn, "DEN", PREV)
    assert d["games"] == 2 and d["team_gp"] == 12 and len(d["units"]) == 3
    t = LU.rtg(d["total"])                                             # 23-19 on 25/25 poss
    assert t["poss"] == 25 and t["ortg"] == pytest.approx(92) and t["drtg"] == pytest.approx(76)
    assert t["min"] == pytest.approx(840 / 60)
    s = LU.select(d, [A])
    assert s["sel"]["poss"] == 20 and s["sel"]["min"] == pytest.approx(10)
    assert s["sel"]["ortg"] == pytest.approx(70) and s["sel"]["drtg"] == pytest.approx(65)
    assert s["sel"]["net"] == pytest.approx(5) and s["sel"]["share"] == pytest.approx(0.8)
    assert s["rest"]["net"] == pytest.approx(60)                        # g01: 9-6 on 5/5
    assert s["diff"] == pytest.approx(5 - 16)
    assert LU.select(d, [A], [FF])["sel"]["net"] == pytest.approx(50)   # only the first stint
    assert LU.select(d, [E, FF])["sel"]["poss"] == 5                   # only g01
    assert LU.select(d, [], [A])["sel"]["net"] == pytest.approx(60)
    none = LU.select(d, [A], [B])["sel"]
    assert none["poss"] == 0 and none["net"] is None
    # away side read from away5 with the scores flipped
    assert LU.load(conn, "BOS", PREV)["total"][:4] == [19.0, 23.0, 25.0, 25.0]
    assert LU.load(conn, "DEN", CUR)["games"] == 0                     # not collected yet


def test_wowy(conn):
    d = LU.load(conn, "DEN", PREV)
    w = LU.wowy(d, A)
    assert w["on"]["net"] == pytest.approx(5) and w["off"]["net"] == pytest.approx(60)
    assert w["diff"] == pytest.approx(-55)
    by = {m["player_id"]: m for m in w["mates"]}
    assert set(by) == {B, C, D, E, FF}
    assert by[B]["both"]["poss"] == 20 and by[B]["a_only"]["poss"] == 0
    assert by[B]["a_only"]["net"] is None and by[B]["b_only"]["net"] == pytest.approx(60)
    assert by[E]["both"]["net"] == pytest.approx(50)                   # stint 1
    assert by[E]["a_only"]["net"] == pytest.approx(-40)                # A without E: stint 2
    assert by[E]["b_only"]["net"] == pytest.approx(60)                 # E without A: g01
    assert [m["player_id"] for m in w["mates"]][:3] == [B, C, D]       # most poss together


def test_combos_min_possessions(conn):
    d = LU.load(conn, "DEN", PREV)
    five = LU.combos(d, 5, 0)
    assert [round(u["net"]) for u in five["best"]] == [60, 50]           # 3 units: split
    assert [round(u["net"]) for u in five["worst"]] == [-40]
    top1 = LU.combos(d, 5, 0, top=1)
    assert round(top1["best"][0]["net"]) == 60 and round(top1["worst"][0]["net"]) == -40
    assert LU.combos(d, 5, 15)["n"] == 0                               # no unit has 15 poss
    # pairs: A+B/C/D on 20 poss, B/C/D pairs on 25, B/C/D + E or F on 15, A+E / A+F 10, E+F 5
    assert LU.combos(d, 2, 15)["n"] == 12
    pairs = LU.combos(d, 2, 16)
    assert pairs["n"] == 6 and all(u["poss"] >= 16 for u in pairs["best"])
    assert all(E not in u["ids"] and FF not in u["ids"] for u in pairs["best"])
    assert LU.combos(d, 2, 0)["n"] == 15
    assert LU.combos(d, 3, 25)["n"] == 1                               # B+C+D everywhere
    with pytest.raises(ValueError):
        LU.combos(d, 4, 0)


# ------------------------------------------------------------------ routes
def test_finder_routes(client):
    html = client.get("/finder").text
    assert "Saved screens" in html and "20+ pts in 8 of last 10" in html
    assert "/static/pro-tools.css?v=" in html and "Add a stat condition" in html
    assert 'href="/lineups"' in html and 'href="/compare"' in html
    html = client.get(f"/finder?mode=games&s1=pts&o1=ge&v1=30&season={PREV}").text
    assert "3 games" in html and 'href="/player/101"' in html and "Ace Scorer" in html
    assert "<table data-sortable" in html
    html = client.get("/finder?mode=games&s1=pts&o1=ge&v1=30&view=players").text
    assert "4 games" in html and "games &rsaquo;" in html
    html = client.get(f"/finder?mode=games&s1=pts&o1=ge&v1=0&player={BIG}").text
    assert "12 games" in html and "Big Man" in html and "all players" in html
    html = client.get("/finder?mode=games&s1=dk&o1=ge&v1=30&sal_max=7900").text
    assert "1 game" in html and "$7,500" in html and "22.5</td>" in html
    r = client.get("/finder?mode=games&s1=pts%3BDROP&o1=ge&v1=1&s2=pts&o2=zz&v2=3&team=XX"
                   "&season=1999-00&rest=9")
    assert r.status_code == 200
    assert "Condition 1 ignored" in r.text and "Condition 2 ignored" in r.text
    assert "is not one of the choices" in r.text and "outside the 3-season window" in r.text
    html = client.get(f"/finder?mode=streak&stat=pts&op=ge&thr=20&k=8&n=10&season={PREV}").text
    assert "1 player" in html and "9<span class=\"dim\">/10</span>" in html
    assert 'class="tl-strip"' in html
    html = client.get("/finder?mode=streak&stat=pts&thr=99&k=1&n=1").text
    assert "No players" in html
    assert "Screen not run" in client.get("/finder?mode=streak&stat=zzz").text
    html = client.get("/finder?mode=games&s1=dk&o1=ge&v1=50&sal_max=7900&sort=dk").text
    assert 'class="pill tl-on"' in html                                # the preset is marked
    assert not re.search(r"\b(over/under|bets?|betting|lean|edge)\b", html, re.I)


def test_lineups_routes(client):
    html = client.get("/lineups").text
    assert "Pick a team" in html and 'href="/lineups?team=DEN&amp;season=' in html
    r = client.get(f"/lineups?team=DEN&season={PREV}")
    assert r.status_code == 200
    assert "2 of 12 games with lineup data" in r.text and "Team overall" in r.text
    assert "Best and worst combos" in r.text and "/static/pro-tools.js?v=" in r.text
    html = client.get(f"/lineups?team=den&season={PREV}&on={A}&off={FF}&wowy={A}&min=0").text
    assert "Selection" in html and "on: Ace Scorer" in html and "off: Eff Six" in html
    assert "+50.0" in html                                             # selection net
    assert "With or without Ace Scorer" in html and "Teammate only" in html
    html = client.get(f"/lineups?team=DEN&season={PREV}&on={A}&off={A}&on=777&on=abc").text
    assert "be both on and off" in html and "Player 777 never played" in html
    assert "is not a player id" in html
    assert "Not collected yet." in client.get("/lineups?team=DEN").text  # no stints this season
    assert client.get("/lineups?team=XYZ").status_code == 404
    html = client.get(f"/lineups?team=DEN&season={PREV}&min=16").text
    assert "6 2-man units with 16+ possessions" in html
    assert "No 5-man unit has 16 possessions" in html


def test_tools_links_on_team_leaders_players(client):
    assert f'href="/lineups?team=DEN&amp;season={PREV}"' in client.get(
        f"/team/DEN?season={PREV}").text
    for url in ("/leaders", "/players"):
        html = client.get(url).text
        assert 'href="/finder"' in html and 'href="/lineups"' in html
