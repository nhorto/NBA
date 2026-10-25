"""Players section (docs/STATS_SITE.md): stats/players.py queries and the /players + /player/<id>
routes. Tiny fixture DB built in the test (conftest's data_dir points COURTSIDE_DATA at a tmp dir).
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.proj.dfs.rules import dk_points  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import players as P  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
WINDOW = S.seasons(TODAY)
Y = S.start_year(CUR)
JOKIC, BETA, GAMMA, DRAY, KEVON, NOPDEF, BENCH = 203999, 2, 3, 10, 11, 12, 4
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _game(conn, gid, day, season, stype, home, away, hpts, apts):
    for team, opp, is_home, pts, opts in ((home, away, 1, hpts, apts), (away, home, 0, apts, hpts)):
        conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, "
                     "is_home, wl, pts) VALUES (?,?,?,?,?,?,?,?,?)",
                     (gid, team, season, stype, day, opp, is_home, "W" if pts > opts else "L", pts))


def _pg(conn, gid, pid, name, team, season, stype, day, *, mn=34.0, pts=20, reb=10, ast=5,
        stl=1, blk=1, tov=3, fg3m=1, fgm=8, fga=16, fta=4, ftm=3, pm=5):
    conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, season, "
                 "season_type, game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, reb, ast, stl, blk, "
                 "tov, pts, plus_minus) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (gid, pid, name, team, season, stype, day, mn, fgm, fga, fg3m, 3, ftm, fta, reb,
                  ast, stl, blk, tov, pts, pm))


def populate(conn: sqlite3.Connection) -> None:
    py = Y - 1
    # previous season, regular: Jokic 5 games (30 pts), Beta 5 games (20 pts), Gamma 1 game (50)
    opps = ["GSW", "GSW", "NOP", "LAL", "BOS"]
    for i, opp in enumerate(opps, 1):
        gid, day = f"p{i}", f"{py}-11-{i:02d}"
        _game(conn, gid, day, PREV, S.REGULAR, "DEN", opp, 110, 100 + i)
        _pg(conn, gid, JOKIC, "Nikola Jokic", "DEN", PREV, S.REGULAR, day, pts=30, reb=12, ast=9)
        _pg(conn, gid, BETA, "Beta Big", opp, PREV, S.REGULAR, day, pts=20, reb=14, ast=2)
    _game(conn, "p9", f"{py}-12-01", PREV, S.REGULAR, "MIA", "PHX", 99, 90)
    _pg(conn, "p9", GAMMA, "Gamma Bench", "MIA", PREV, S.REGULAR, f"{py}-12-01", pts=50)
    _pg(conn, "p9", BENCH, "Bench Guy", "MIA", PREV, S.REGULAR, f"{py}-12-01", mn=0, pts=0)
    # previous season playoffs: one game vs GSW
    _game(conn, "p20", f"{Y}-04-25", PREV, S.PLAYOFFS, "GSW", "DEN", 101, 99)
    _pg(conn, "p20", JOKIC, "Nikola Jokic", "DEN", PREV, S.PLAYOFFS, f"{Y}-04-25", pts=40)
    # this season: one game vs NOP
    cday = f"{Y}-10-22"
    _game(conn, "c1", cday, CUR, S.REGULAR, "DEN", "NOP", 120, 110)
    _pg(conn, "c1", JOKIC, "Nikola Jokic", "DEN", CUR, S.REGULAR, cday, mn=44, pts=35, reb=15,
        ast=10, fg3m=3)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES ('c1', ?, ?, ?, 'DEN', 'NOP', 'final')", (CUR, cday, f"{cday}T23:00:00Z"))
    nxt = TODAY + timedelta(days=1)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES ('c2', ?, ?, ?, 'DEN', 'GSW', 'scheduled')",
                 (CUR, nxt.isoformat(), f"{nxt.isoformat()}T23:59:00Z"))
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team, jersey, "
                     "position, height, weight, country, draft_year, draft_round, draft_number) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     [(JOKIC, "Nikola", "Jokić", "DEN", "15", "C", "6-11", 284, "Serbia", 2014, 2,
                       41),
                      (BETA, "Beta", "Big", "GSW", "1", "F", "6-9", 240, "USA", None, None, None),
                      (DRAY, "Dray", "Defender", "GSW", "23", "F", "6-6", 230, "USA", None, None,
                       None),
                      (KEVON, "Kevon", "Looney", "GSW", "5", "C", "6-9", 222, "USA", None, None,
                       None),
                      (77, "No", "Games", "BOS", None, "G", None, None, None, None, None, None)])
    # who guarded him: p1 + p2 vs GSW, p3 vs NOP (must not count for GSW)
    mu = ("INSERT INTO player_matchups (game_id, off_player_id, def_player_id, def_team, matchup_min,"
          " partial_poss, pts, fgm, fga, fg3m, fg3a, ast, tov) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")
    conn.executemany(mu, [("p1", JOKIC, DRAY, "GSW", 6.0, 30.0, 12, 5, 9, 0, 1, 2, 1),
                          ("p1", JOKIC, KEVON, "GSW", 2.0, 10.0, 4, 2, 3, 0, 0, 0, 0),
                          ("p2", JOKIC, DRAY, "GSW", 4.0, 20.0, 8, 3, 6, 1, 2, 1, 0),
                          ("p3", JOKIC, NOPDEF, "NOP", 5.0, 40.0, 20, 8, 10, 0, 0, 3, 1)])
    # shots: Jokic prev regular (4 RA, 3 made; 2 threes missed); Beta (GSW) 2 RA, 1 made
    sh = ("INSERT INTO shots (game_id, action_number, player_id, team, season, season_type, "
          "game_date, x, y, made, zone) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    rows = [("p1", i, JOKIC, "DEN", PREV, S.REGULAR, f"{py}-11-01", x, y, m, z)
            for i, (x, y, m, z) in enumerate([(0, 0, 1, "Restricted Area"),
                                              (2, 3, 1, "Restricted Area"),
                                              (-3, 1, 1, "Restricted Area"),
                                              (1, -2, 0, "Restricted Area"),
                                              (0, 250, 0, "Above the Break 3"),
                                              (5, 252, 0, "Above the Break 3")], 1)]
    rows += [("p1", 100 + i, BETA, "GSW", PREV, S.REGULAR, f"{py}-11-01", 0, 5, m,
              "Restricted Area") for i, m in enumerate([1, 0])]
    conn.executemany(sh, rows)
    # play types (prev regular season)
    syn = ("INSERT INTO synergy_playtypes (season, season_type, entity, grouping, play_type, "
           "player_id, team_id, player_name, team, percentile, gp, poss, poss_pct, ppp, pts) "
           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    conn.executemany(syn, [
        (PREV, S.REGULAR, "P", "Offensive", "Postup", JOKIC, 1610612743, "Nikola Jokić", "DEN",
         0.8, 5, 100.0, 0.3, 1.1, 110.0),
        (PREV, S.REGULAR, "P", "Offensive", "Spotup", JOKIC, 1610612743, "Nikola Jokić", "DEN",
         0.4, 5, 50.0, 0.15, 0.9, 45.0),
        (PREV, S.REGULAR, "P", "Defensive", "Postup", JOKIC, 1610612743, "Nikola Jokić", "DEN",
         0.5, 5, 70.0, 0.2, 0.8, 56.0)])
    # DFS: DK id map, a classic slate on the current-season game day, salary, contests, runs
    conn.execute("INSERT INTO player_ids (source, source_id, player_id, name, team, how) VALUES "
                 "('dk', '830650', ?, 'Nikola Jokic', 'DEN', 'exact_team')", (JOKIC,))
    conn.execute("INSERT INTO dfs_slates (slate_id, site, nba_day, game_type_id, tag, lock_utc) "
                 "VALUES ('s1', 'dk', ?, 70, 'main', ?)", (cday, f"{cday}T23:00:00Z"))
    conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id, "
                 "player_name, site_player_id, team, positions, salary) VALUES "
                 "(?, 'dk', 's1', 'classic', 70, 'Nikola Jokic', '830650', 'DEN', 'C', 11400)",
                 (f"{cday}T15:00:00Z",))
    conn.executemany("INSERT INTO dfs_contests (contest_id, slate_id, name, field_size, "
                     "captured_at) VALUES (?, 's1', ?, ?, ?)",
                     [("k1", "Big GPP", 1000, f"{cday}T15:00:00Z"),
                      ("k2", "Small GPP", 100, f"{cday}T15:00:00Z")])
    conn.executemany("INSERT INTO dfs_contest_ownership (contest_id, player_id, name, pct_drafted, "
                     "fpts) VALUES (?, ?, 'Nikola Jokic', ?, 81.75)",
                     [("k1", JOKIC, 30.0), ("k2", JOKIC, 40.0)])
    runs = [("r1", f"{cday}T21:00:00Z", 1, "lock45"), ("r2", f"{cday}T22:30:00Z", 1, "lock10"),
            ("r3", f"{cday}T22:50:00Z", 0, "manual"), ("r4", f"{cday}T23:30:00Z", 1, "swap")]
    conn.executemany("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, ok) "
                     "VALUES (?, 's1', ?, ?, ?)", [(r, k, t, ok) for r, t, ok, k in runs])
    conn.executemany("INSERT INTO dfs_ownership_proxy (run_id, player_id, own) VALUES (?, ?, ?)",
                     [("r1", 830650, 20.0), ("r2", 830650, 25.0), ("r4", 830650, 99.0)])
    for n, (r, t, _, _) in enumerate(runs, 1):
        conn.execute("INSERT INTO player_projections (id, game_id, nba_day, slate_id, player_id, "
                     "team, run_id, created_at, p_play) VALUES (?, 'c1', ?, 's1', ?, 'DEN', ?, ?, "
                     "0.99)", (n, cday, JOKIC, r, t))
        for stat, mean, lo, hi in (("dk", 50.0 + n, 35.0, 70.0), ("min", 34.0, 28.0, 38.0),
                                   ("pts", 25.0, None, None), ("reb", 12.0, None, None)):
            conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean, "
                         "q10, q90) VALUES (?, ?, 'played', ?, ?, ?)", (n, stat, mean, lo, hi))
        conn.execute("INSERT INTO player_projection_grades (projection_id, game_id, player_id, "
                     "run_id, graded_at, played, actual_min, actual_dk, salary) VALUES "
                     "(?, 'c1', ?, ?, ?, 1, 44, 81.75, 11400)", (n, JOKIC, r, f"{cday}T09:00:00Z"))
    # injury report: Jokic questionable (ESPN, an hour ago)
    conn.execute("INSERT INTO injury_snapshots (captured_at, source, team, player_name, player_id, "
                 "status, reason) VALUES (?, 'espn', 'DEN', 'Nikola Jokic', 'espn:3112335', "
                 "'Questionable', 'Ankle')", (_iso(NOW - timedelta(hours=1)),))
    conn.commit()


@pytest.fixture
def dbconn(data_dir):
    sdb.clear()
    conn = db.connect()
    populate(conn)
    yield conn
    conn.close()


@pytest.fixture
def client(dbconn, data_dir):
    return TestClient(create_app(data_dir))


def _text(html: str) -> str:
    return re.sub(r"<script.*?</script>", "", html, flags=re.S)


# ------------------------------------------------------------------ league table + ranks
def test_rank_values_ties_and_direction():
    assert P.rank_values({"a": 3.0, "b": 5.0, "c": 5.0, "d": None}) == {"a": 3, "b": 1, "c": 1}
    assert P.rank_values({"a": 3.0, "b": 5.0}, higher_better=False) == {"a": 1, "b": 2}
    assert P.qualifier(5) == 2 and P.qualifier(70) == 28 and P.qualifier(0) == 2


def test_league_table_ranks_and_qualifier(dbconn):
    lt = P.league_table(dbconn, PREV, S.REGULAR)
    pl = lt["players"]
    assert lt["max_gp"] == 5 and lt["min_gp"] == 2 and lt["n"] == 2
    assert BENCH not in pl                                  # 0-minute rows are not games
    j, b, g = pl[JOKIC], pl[BETA], pl[GAMMA]
    assert j["gp"] == 5 and j["pts"] == 30 and j["name"] == "Nikola Jokić" and j["team"] == "DEN"
    assert j["ranks"]["pts"] == 1 and b["ranks"]["pts"] == 2 and b["ranks"]["reb"] == 1
    assert not g["qualified"] and g["ranks"]["pts"] is None          # 1 game: not ranked
    assert j["dk"] == pytest.approx(dk_points({"pts": 30, "reb": 12, "ast": 9, "stl": 1,
                                                "blk": 1, "tov": 3, "fg3m": 1}))
    assert j["ts"] == pytest.approx(30 / (2 * (16 + 0.44 * 4)))


def test_directory_filters_sort_and_pages(dbconn):
    d = P.directory(dbconn, PREV)
    assert [r["player_id"] for r in d["rows"]] == [GAMMA, JOKIC, BETA]        # by pts
    assert P.directory(dbconn, PREV, sort="reb")["rows"][0]["player_id"] == BETA
    assert [r["player_id"] for r in P.directory(dbconn, PREV, q="jokic")["rows"]] == [JOKIC]
    assert [r["player_id"] for r in P.directory(dbconn, PREV, team="mia")["rows"]] == [GAMMA]
    pg = P.directory(dbconn, PREV, per_page=2, page=2)
    assert pg["pages"] == 2 and pg["offset"] == 2 and len(pg["rows"]) == 1


# ------------------------------------------------------------------ header, lines, log
def test_header_bio_next_status_salary(dbconn):
    h = P.header(dbconn, JOKIC, today=TODAY, now=NOW)
    assert h["name"] == "Nikola Jokić" and h["team"] == "DEN" and h["initials"] == "NJ"
    assert h["draft"] == "drafted 2014 R2 #41" and h["height"] == "6-11"
    assert h["next"]["opp"] == "GSW" and h["next"]["is_home"] is True
    assert h["status"]["status"] == "Questionable"
    assert h["salary"]["salary"] == 11400
    assert P.header(dbconn, 999999, today=TODAY) is None
    assert P.header(dbconn, 77, today=TODAY)["name"] == "No Games"    # bio only
    assert P.header(dbconn, BETA, today=TODAY, now=NOW)["status"] is None


def test_season_lines_and_game_log(dbconn):
    lines = P.season_lines(dbconn, JOKIC, WINDOW)
    assert [(ln["season"], ln["season_type"]) for ln in lines] == [
        (CUR, S.REGULAR), (PREV, S.REGULAR), (PREV, S.PLAYOFFS)]
    log = P.game_log(dbconn, JOKIC, WINDOW)
    assert [r["game_id"] for r in log] == ["c1", "p20", "p5", "p4", "p3", "p2", "p1"]
    top = log[0]
    assert top["opp"] == "NOP" and top["is_home"] == 1 and top["wl"] == "W"
    assert top["salary"] == 11400 and top["own"] == pytest.approx(35.0)
    assert top["value"] == pytest.approx(top["dk"] / 11.4)
    assert top["dk"] == pytest.approx(dk_points(top))
    assert log[1]["salary"] is None and log[1]["own"] is None
    po = P.game_log(dbconn, JOKIC, WINDOW, season=PREV, season_type=S.PLAYOFFS)
    assert [r["game_id"] for r in po] == ["p20"]


# ------------------------------------------------------------------ projection
def test_pick_projection_latest_ok_run_before_tip(dbconn):
    tip = f"{Y}-10-22T23:00:00Z"
    p = P.pick_projection(dbconn, "c1", JOKIC, tip)
    assert p["run_id"] == "r2" and p["kind"] == "lock10"          # r3 not ok, r4 after tip
    assert p["stats"]["dk"]["mean"] == 52.0 and p["stats"]["dk"]["q10"] == 35.0
    assert p["grade"]["actual_dk"] == 81.75
    assert P.pick_projection(dbconn, "c1", JOKIC, None)["run_id"] == "r4"
    assert P.pick_projection(dbconn, "c1", JOKIC, f"{Y}-10-22T20:00:00Z") is None


def test_projection_card_graded_and_tonight(dbconn):
    card = P.projection_card(dbconn, JOKIC, None, NOW)
    assert card["mode"] == "graded" and card["proj"]["run_id"] == "r2"
    assert card["actual"]["pts"] == 35 and card["game"]["opp"] == "NOP"
    now = datetime(Y, 10, 22, 22, 45, tzinfo=timezone.utc)            # game day, before tip
    nxt = {"game_id": "c1", "nba_day": config.nba_day(now).isoformat(),
           "tipoff_utc": f"{Y}-10-22T23:00:00Z", "opp": "NOP", "is_home": True}
    tonight = P.projection_card(dbconn, JOKIC, nxt, now)
    assert tonight["mode"] == "tonight" and tonight["proj"]["run_id"] == "r2"
    assert tonight["actual"] is None
    assert P.projection_card(dbconn, BETA, None, NOW) is None


# ------------------------------------------------------------------ versus a team
def test_versus_and_who_guarded_him(dbconn):
    v = P.versus(dbconn, JOKIC, "GSW", WINDOW)
    assert [g["game_id"] for g in v["games"]] == ["p20", "p2", "p1"]
    assert v["avg"]["pts"] == pytest.approx((40 + 30 + 30) / 3)
    base = (35 + 40 + 5 * 30) / 7
    assert v["diff"]["pts"] == pytest.approx(v["avg"]["pts"] - base)
    g = {r["def_player_id"]: r for r in v["guarded"]}
    assert set(g) == {DRAY, KEVON}                                   # NOP defender excluded
    d = g[DRAY]
    assert d["name"] == "Dray Defender" and d["games"] == 2 and d["poss"] == 50
    assert (d["pts"], d["fgm"], d["fga"], d["ast"], d["tov"]) == (20, 8, 15, 3, 1)
    assert d["p100"] == pytest.approx(40.0) and v["guarded"][0]["def_player_id"] == DRAY
    empty = P.versus(dbconn, JOKIC, "SAS", WINDOW)
    assert empty["games"] == [] and empty["guarded"] == [] and empty["avg"] == {}
    assert [o["team"] for o in P.opponents(dbconn, JOKIC, WINDOW)] == ["BOS", "GSW", "LAL",
                                                                       "NOP"]


# ------------------------------------------------------------------ shots
def test_hexbin_pointy_bins():
    assert P.hex_center(0, 0) == (0, 0)
    bins = P.hexbin([(0, 0, 1), (2, 3, 0), (100, 100, 1), (None, 5, 1)])
    by = {(b["x"], b["y"]): b for b in bins}
    assert by[(0, 0)]["n"] == 2 and by[(0, 0)]["made"] == 1 and len(bins) == 2
    for b in bins:                     # every centre is on the pointy-top lattice
        dx, dy = P.HEX_R * 3 ** .5, P.HEX_R * 1.5
        row = round(b["y"] / dy)
        assert b["y"] == pytest.approx(row * dy)
        assert (b["x"] / dx - (row & 1) / 2) == pytest.approx(round(b["x"] / dx - (row & 1) / 2))
    assert P.fg_color(0.1) == P.FG_RAMP[0] and P.fg_color(0.9) == P.FG_RAMP[-1]
    assert len(P.hex_points(10, 10, 5).split()) == 6


def test_shot_chart_zones_vs_league(dbconn):
    sc = P.shot_chart(dbconn, JOKIC, PREV, S.REGULAR)
    assert sc["n"] == 6 and sc["made"] == 3 and sc["games"] == 1
    z = {r["zone"]: r for r in sc["zones"]}
    assert list(z) == ["Restricted Area", "Above the Break 3"]           # zone order
    ra = z["Restricted Area"]
    assert (ra["fga"], ra["fgm"]) == (4, 3) and ra["share"] == pytest.approx(4 / 6)
    assert ra["league"] == pytest.approx(4 / 6) and ra["diff"] == pytest.approx(0.75 - 4 / 6)
    assert sum(h["n"] for h in sc["hexes"]) == 6 and all("points" in h for h in sc["hexes"])
    lz = P.league_zones(dbconn, PREV, S.REGULAR)
    assert lz["Restricted Area"]["fga"] == 6 and lz["Above the Break 3"]["fgm"] == 0
    assert P.shot_chart(dbconn, JOKIC, CUR, S.REGULAR)["n"] == 0


def test_missing_tables_return_empty():
    mem = sqlite3.connect(":memory:")
    assert P.shot_chart(mem, JOKIC, PREV)["n"] == 0
    assert P.play_types(mem, JOKIC, PREV)["rows"] == []
    assert P.ownership(mem, JOKIC) == [] and P.versus(mem, JOKIC, "GSW", WINDOW)["games"] == []
    assert P.league_table(mem, PREV)["players"] == {} and P.header(mem, JOKIC, today=TODAY) is None


# ------------------------------------------------------------------ play types + ownership
def test_play_types_offense_only_and_thin(dbconn, monkeypatch):
    pt = P.play_types(dbconn, JOKIC, PREV)
    assert pt["thin"] is True                       # 2 offensive rows league-wide < threshold
    monkeypatch.setattr(P, "SYNERGY_THIN", 1)
    pt = P.play_types(dbconn, JOKIC, PREV)
    assert not pt["thin"] and [r["play_type"] for r in pt["rows"]] == ["Postup", "Spotup"]
    assert pt["rows"][0]["label"] == "Post-up" and pt["rows"][0]["pctile"] == 80
    assert P.play_types(dbconn, JOKIC, CUR)["rows"] == []


def test_ownership_rows_with_prelock_estimate(dbconn):
    rows = P.ownership(dbconn, JOKIC)
    assert [r["contest"] for r in rows] == ["Big GPP", "Small GPP"]
    assert rows[0]["own"] == 30.0 and rows[0]["field"] == 1000 and rows[0]["nba_day"] == f"{Y}-10-22"
    assert all(r["est"] == 25.0 for r in rows)      # r2: latest ok run before lock (not r4)
    assert P.ownership(dbconn, BETA) == []


# ------------------------------------------------------------------ routes
def test_player_page_renders_everything(client):
    r = client.get(f"/player/{JOKIC}")
    assert r.status_code == 200
    html = _text(r.text)
    assert "None" not in html and "/static/pro-players.css?v=" in html
    for want in ("Nikola Jokić", "drafted 2014 R2 #41", "$11,400", "vs GSW", "Questionable",
                 "Our projection", "last game, graded", "Season by season", "Game log",
                 "Versus a team", "Who guarded him", "Dray Defender", "Real DFS ownership",
                 "Big GPP", "Our est.", 'class="rk"'):
        assert want in html, want
    assert '<option value="GSW" selected>' in html                 # default: next opponent
    assert "No shots collected" in html                             # current season: no shots
    assert "Not enough play-type data yet" in html


def test_player_page_switches(client):
    html = _text(client.get(f"/player/{JOKIC}?season={PREV}&vs=NOP").text)
    assert "<polygon" in html and "Restricted Area" in html and "data-tip=" in html
    assert f'<a href="/player/{JOKIC}?season={PREV}&amp;type=playoffs&amp;vs=NOP">Playoffs' in html
    assert '<option value="NOP" selected>' in html and "No matchup data" not in html
    html = _text(client.get(f"/player/{JOKIC}?log={PREV}&logtype=playoffs").text)
    assert html.count('href="/team/GSW"') >= 1 and "vs NOP" not in html.split('id="log"')[1] \
        .split("</section>")[0]
    html = _text(client.get(f"/player/{JOKIC}?vs=SAS").text)
    assert "No games against San Antonio Spurs" in html


def test_player_page_empty_states(client):
    r = client.get("/player/77")
    assert r.status_code == 200
    html = _text(r.text)
    assert "None" not in html
    for want in ("No games in the last three seasons.", "No projection yet.",
                 "Not in any standings you uploaded yet.", "No shots collected",
                 "No opponents yet."):
        assert want in html, want


def test_player_404(client):
    for url in ("/player/999999", "/player/abc"):
        assert client.get(url).status_code == 404, url


def test_players_directory_route(client):
    r = client.get(f"/players?season={PREV}")
    assert r.status_code == 200
    html = _text(r.text)
    assert "None" not in html and "Player directory" in html
    assert html.index("Gamma Bench") < html.index("Nikola Jokić") < html.index("Beta Big")
    assert f'href="/player/{JOKIC}"' in html and 'class="rk"' in html
    html = _text(client.get(f"/players?season={PREV}&sort=reb").text)
    assert html.index("Beta Big") < html.index("Nikola Jokić")
    html = _text(client.get("/players?q=jokic").text)
    assert "Nikola Jokić" in html and "Beta Big" not in html
    html = _text(client.get(f"/players?season={PREV}&team=MIA").text)
    assert "Gamma Bench" in html and "Nikola Jokić" not in html
    html = _text(client.get("/players?q=zzzz").text)
    assert "players match" in html


# ================================================================== research tools
# (stats/player_tools.py: filters, splits, recent-games chart, minutes, clutch)
import gzip  # noqa: E402
import json  # noqa: E402

from courtside.stats import player_tools as T  # noqa: E402

MURRAY, SUBBY = 20, 21


def populate_tools(conn: sqlite3.Connection) -> None:
    """On top of populate(): a 1-day-rest game (p6), a teammate who plays p1-p3 (DNP in p4,
    absent in p5, p6, p20), start data for p1-p5 (bench in p5), a non-teammate in p1."""
    py = Y - 1
    _game(conn, "p6", f"{py}-11-07", PREV, S.REGULAR, "DEN", "MIA", 100, 90)
    _pg(conn, "p6", JOKIC, "Nikola Jokic", "DEN", PREV, S.REGULAR, f"{py}-11-07", mn=24, pts=10,
        reb=5, ast=3)
    for i in (1, 2, 3):
        _pg(conn, f"p{i}", MURRAY, "Jamal Murray", "DEN", PREV, S.REGULAR, f"{py}-11-{i:02d}",
            mn=30, pts=18)
    _pg(conn, "p4", MURRAY, "Jamal Murray", "DEN", PREV, S.REGULAR, f"{py}-11-04", mn=0, pts=0)
    conn.executemany("INSERT INTO player_starts (game_id, player_id, team, source) VALUES "
                     "(?, ?, 'DEN', 'sdv')", [(f"p{i}", JOKIC) for i in (1, 2, 3, 4)]
                     + [("p5", SUBBY)])
    conn.execute("INSERT INTO player_index (player_id, first_name, last_name, team) VALUES "
                 "(?, 'Jamal', 'Murray', 'DEN')", (MURRAY,))
    conn.commit()


@pytest.fixture
def tconn(data_dir):
    sdb.clear()
    conn = db.connect()
    populate(conn)
    populate_tools(conn)
    yield conn
    conn.close()


@pytest.fixture
def tclient(tconn, data_dir):
    return TestClient(create_app(data_dir))


def _by_id(gs):
    return {g["game_id"]: g for g in gs}


def test_tools_games_rest_and_starts(tconn):
    gs = T.games(tconn, JOKIC, WINDOW)
    assert [g["game_id"] for g in gs] == ["c1", "p20", "p6", "p5", "p4", "p3", "p2", "p1"]
    g = _by_id(gs)
    # rest = days off since his team's previous game in that season (None = first game)
    assert g["p1"]["rest"] is None and g["c1"]["rest"] is None
    assert [g[f"p{i}"]["rest"] for i in (2, 3, 4, 5)] == [0, 0, 0, 0]          # back-to-backs
    assert g["p6"]["rest"] == 1 and g["p20"]["rest"] >= 2
    # starts: p1-p4 started, p5 has start data without him (bench), others unknown
    assert [g[f"p{i}"]["started"] for i in (1, 2, 3, 4, 5)] == [True] * 4 + [False]
    assert g["p6"]["started"] is None and g["c1"]["started"] is None
    # DvP tiers: PREV has under 10 games per team -> no tiers for its games; CUR (also thin)
    # borrows PREV's ranks. Index basis there: his position is C; 8 teams -> all "bottom 10".
    assert all(g[f"p{i}"]["tier"] is None for i in (1, 2, 3, 4, 5, 6))
    assert g["c1"]["tier"] == "bottom" and g["c1"]["dvp_season"] == PREV and g["c1"]["pos"] == "C"
    f = T.parse_filters({"dt": "bottom"})
    assert [x["game_id"] for x in T.apply_filters(tconn, JOKIC, gs, f, WINDOW)] == ["c1"]
    assert T.apply_filters(tconn, JOKIC, gs, {"dt": "top"}, WINDOW) == []
    assert T.tier_of(1, 30) == "bottom" and T.tier_of(10, 30) == "bottom"
    assert T.tier_of(11, 30) == "mid" and T.tier_of(21, 30) == "top" and T.tier_of(None, 30) is None


def test_tools_filters_rest_role_home_last(tconn):
    gs = T.games(tconn, JOKIC, WINDOW)

    def ids(**q):
        f = T.parse_filters(q, opps={"GSW", "NOP", "MIA"}, mates={MURRAY})
        return [g["game_id"] for g in T.apply_filters(tconn, JOKIC, gs, f, WINDOW)]
    assert ids(rest="0") == ["p5", "p4", "p3", "p2"]
    assert ids(rest="1") == ["p6"] and ids(rest="2") == ["p20"]
    assert ids(role="start") == ["p4", "p3", "p2", "p1"] and ids(role="bench") == ["p5"]
    assert ids(ha="away") == ["p20"] and ids(wl="L") == ["p20"]
    assert ids(opp="GSW") == ["p20", "p2", "p1"] and ids(opp="XXX") == [g["game_id"] for g in gs]
    assert ids(rest="0", last="2") == ["p5", "p4"]             # last N applies after the rest
    assert T.parse_filters({"rest": "9", "ha": "x", "last": "abc", "tm": "77"}) == {}


def test_tools_with_and_without_teammate(tconn):
    gs = T.games(tconn, JOKIC, WINDOW)
    mates = T.teammates(tconn, JOKIC, WINDOW)
    assert [m["player_id"] for m in mates] == [MURRAY]          # min 0 rows and opponents: no
    assert mates[0]["games"] == 3 and mates[0]["name"] == "Jamal Murray"
    w = T.apply_filters(tconn, JOKIC, gs, {"tm": MURRAY, "tmode": "with"}, WINDOW)
    assert [g["game_id"] for g in w] == ["p3", "p2", "p1"]
    wo = T.apply_filters(tconn, JOKIC, gs, {"tm": MURRAY, "tmode": "without"}, WINDOW)
    # p4: a 0-minute row counts as missing; p5/p6/p20: no row. c1 (a season he did not play
    # for DEN) is not a "without" game.
    assert [g["game_id"] for g in wo] == ["p20", "p6", "p5", "p4"]
    f = T.parse_filters({"tm": str(MURRAY), "tmode": "without"}, mates={MURRAY})
    assert f == {"tm": MURRAY, "tmode": "without"}


def test_tools_splits_sample_sizes(tconn):
    gs = [g for g in T.games(tconn, JOKIC, WINDOW) if g["season"] == PREV
          and g["season_type"] == S.REGULAR]
    rows = T.splits(tconn, JOKIC, gs, window=WINDOW, mate={"player_id": MURRAY,
                                                             "name": "Jamal Murray"}, opp="GSW")
    by = {r["label"]: r for r in rows}
    assert by["All games"]["avg"]["gp"] == 6
    assert by["All games"]["avg"]["pts"] == pytest.approx((5 * 30 + 10) / 6)
    assert by["Back-to-back"]["avg"]["gp"] == 4 and not by["Back-to-back"]["avg"]["small"]
    assert by["1 day rest"]["avg"]["gp"] == 1 and by["1 day rest"]["avg"]["small"]
    assert by["2+ days rest"]["avg"]["gp"] == 0 and by["2+ days rest"]["avg"]["pts"] is None
    assert by["Bench"]["avg"]["gp"] == 1 and by["Starter"]["avg"]["gp"] == 4
    assert by["With Jamal Murray"]["avg"]["gp"] == 3
    assert by["Without Jamal Murray"]["avg"]["gp"] == 3          # p4, p5, p6
    assert by["vs GSW"]["avg"]["gp"] == 2 and by["vs GSW"]["key"] == "opp"
    pra = by["All games"]["avg"]["pra"]
    assert pra == pytest.approx((5 * (30 + 12 + 9) + (10 + 5 + 3)) / 6)


def test_tools_default_line_and_summary():
    assert T.default_line(26.3) == 26.5 and T.default_line(26.8) == 26.5
    assert T.default_line(26.0) == 26.5 and T.default_line(None) == 0.5
    assert T.threshold_label(24.5, True) == "25" and T.threshold_label(25, True) == "25"
    assert T.threshold_label(45.5, False) == "45.5" and T.threshold_label(40.0, False) == "40"
    s = T.summary(24.5, True, [("last 10", [25, 30, 10, 24, 26, 25, 25, 19, 40, 27]),
                               ("2026-27", [25] * 18 + [20] * 13), ("vs DAL", [30, 30, 10])])
    assert s == "Reached 25+ in 7 of last 10 (70%) · 2026-27 18 of 31 (58%) · vs DAL 2 of 3 (67%)"
    assert T.summary(5, True, [("last 8", [5, 1, 1, 1, 1, 1, 1, 1])]).endswith("(13%)")  # 12.5
    for bad in ("over", "under", "edge", "lean", "pick", "bet"):
        assert bad not in s.lower()


def test_tools_bar_chart_data(tconn):
    gs = T.games(tconn, JOKIC, WINDOW)
    c = T.bar_chart(gs, "pts", "10", None, season=CUR, stype=S.REGULAR, season_avg=35.0,
                    vs_opp="GSW")
    assert c["default"] == 35.5 and c["line"] == 35.5 and c["n"] == 8
    assert [b["v"] for b in c["bars"]] == [30, 30, 30, 30, 30, 10, 40, 35]     # oldest first
    assert [b["hit"] for b in c["bars"]].count(True) == 1 and c["bars"][6]["hit"]
    assert c["bars"][6]["opp"] == "GSW" and not c["bars"][6]["home"]
    assert "Apr" not in c["bars"][0]["tip"] and "34 min" in c["bars"][0]["tip"]
    assert c["summary"] == (f"Reached 36+ in 1 of last 8 (13%) · {CUR} 0 of 1 (0%) · "
                            f"vs GSW 1 of 3 (33%)")
    assert c["data"]["parts"][0]["values"] == [30, 30, 30, 30, 30, 10, 40, 35]
    assert c["top"] < c["line_y"] < c["bot"] and c["ymax"] >= 40
    # visitor line + other stats
    c = T.bar_chart(gs, "pra", "5", 51, season=PREV, stype=S.REGULAR, season_avg=None,
                    vs_opp=None)
    assert [b["v"] for b in c["bars"]] == [51, 51, 18, 55, 60]     # p4 p5 p6 p20 c1
    assert c["summary"] == f"Reached 51+ in 4 of last 5 (80%) · {PREV} 5 of 6 (83%)"
    c = T.bar_chart(gs, "dk", "season", None, season=PREV, stype=S.PLAYOFFS, season_avg=None,
                    vs_opp=None)
    assert c["n"] == 1 and not c["integer"] and c["summary"].startswith("Reached ")
    assert "1 game in" in c["summary"]
    assert T.stat_value({"stl": 2, "blk": 1}, "stocks") == 3
    assert "</" not in T.json_script({"x": "</script>"})


def test_tools_minutes_trend(tconn):
    gs = T.games(tconn, JOKIC, WINDOW)
    m = T.minutes_trend(gs, PREV, S.REGULAR)
    assert m["n"] == 8 and len(m["points"]) == 8
    assert m["s30"] == pytest.approx(7 / 8) and m["s36"] == pytest.approx(1 / 8)   # 44 min c1
    assert m["start_n"] == 4 and m["bench_n"] == 1 and m["bench_min"] == 34
    assert m["season_n"] == 6 and m["season_s30"] == pytest.approx(5 / 6)


# ------------------------------------------------------------------ clutch
def _act(n, period, clock, atype, team, pid, home, away, **kw):
    return {"actionNumber": n, "orderNumber": n * 10, "period": period,
            "clock": f"PT{clock[0]:02d}M{clock[1]:05.2f}S", "actionType": atype,
            "teamTricode": team, "personId": pid, "scoreHome": str(home),
            "scoreAway": str(away), "personIdsFilter": [pid] if pid else [], **kw}


def clutch_payload() -> dict:
    a = [_act(1, 1, (5, 0), "2pt", "DEN", JOKIC, 88, 86, shotResult="Made"),
         _act(2, 4, (8, 0), "2pt", "NOP", 99, 88, 88, shotResult="Made"),       # not clutch
         _act(3, 4, (4, 0), "3pt", "DEN", JOKIC, 91, 88, shotResult="Made"),
         _act(4, 4, (3, 0), "2pt", "NOP", 99, 91, 90, shotResult="Made"),
         _act(5, 4, (2, 0), "substitution", "DEN", JOKIC, 91, 90, subType="out"),
         _act(6, 4, (1, 0), "3pt", "NOP", 99, 91, 93, shotResult="Made"),       # he is off
         _act(7, 4, (0, 30), "substitution", "DEN", JOKIC, 91, 93, subType="in"),
         _act(8, 4, (0, 10), "freethrow", "DEN", JOKIC, 92, 93, shotResult="Made"),
         _act(9, 4, (0, 5), "2pt", "DEN", JOKIC, 92, 93, shotResult="Missed")]
    return {"game": {"actions": a}}


def test_clutch_game_line():
    ln = T.clutch_game(clutch_payload(), JOKIC, "DEN")
    assert (ln["pts"], ln["fgm"], ln["fga"], ln["fg3m"], ln["fg3a"], ln["ftm"], ln["fta"]) == \
        (4, 1, 2, 1, 1, 1, 1)
    assert ln["pm"] == 2 and ln["sec"] == pytest.approx(205) and ln["any"]
    blowout = {"game": {"actions": [
        _act(1, 1, (5, 0), "2pt", "DEN", JOKIC, 100, 80, shotResult="Made"),
        _act(2, 4, (1, 0), "2pt", "DEN", JOKIC, 102, 80, shotResult="Made")]}}
    ln = T.clutch_game(blowout, JOKIC, "DEN")
    assert ln["pts"] == 0 and ln["sec"] == 0 and not ln["any"]


def test_clutch_from_archived_pbp(tconn, data_dir):
    p = T.pbp_path(f"{Y}-10-22", "c1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(clutch_payload()).encode()))
    k = T.clutch(tconn, JOKIC, CUR, S.REGULAR, WINDOW)
    assert k["n_games"] == 1 and k["with_pbp"] == 1 and k["missing"] == 0
    assert k["tot"]["pts"] == 4 and k["tot"]["pm"] == 2 and k["games"][0]["opp"] == "NOP"
    k = T.clutch(tconn, JOKIC, PREV, S.REGULAR, WINDOW)
    assert k["n_games"] == 0 and k["missing"] == 6 and k["with_pbp"] == 0


# ------------------------------------------------------------------ routes
def test_player_page_research_tools(tclient):
    html = tclient.get(f"/player/{JOKIC}").text
    assert '<script src="/static/pro-players.js?v=' in html
    assert '<script type="application/json" id="pl2-data">' in html
    data = json.loads(html.split('id="pl2-data">')[1].split("</script>")[0])
    assert data["line"] == 35.5 and data["integer"] is True
    text = _text(html)
    assert "None" not in text
    want = f"Reached 36+ in 1 of last 8 (13%) · {CUR} 0 of 1 (0%) · vs GSW 1 of 3 (33%)"
    assert want in text                                           # server-side summary
    assert text.count('class="pl2-bar hit"') == 1 and text.count('class="pl2-bar"') == 7
    for w in ("Filters", "Recent games", "Minutes", "Splits", "Clutch", "Back-to-back",
              "Jamal Murray"):
        assert w in text, w
    for bad in (">Over", ">Under", "edge", " lean", "bet "):
        assert bad not in text, bad


def test_player_page_filters_drive_chart_and_log(tclient):
    html = _text(tclient.get(f"/player/{JOKIC}?tm={MURRAY}&tmode=without&line=20").text)
    assert "4</b> of 8 games match" in html and "Without Jamal Murray" in html
    log = html.split('id="log"')[1].split("</section>")[0]
    assert log.count("<tr") == 5                                  # header + 4 games
    assert "Reached 20+ in 3 of last 4 (75%)" in html
    html = _text(tclient.get(f"/player/{JOKIC}?rest=0&stat=reb&win=5").text)
    assert "Reached 16+ in 0 of last 4 (0%)" in html              # default: CUR reb 15 -> 15.5
    assert 'href="/team/MIA"' not in html.split('id="log"')[1].split("</section>")[0]
