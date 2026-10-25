"""Percentile bars (stats/percentiles.py, `pct_bar` in _stats.html) and the garbage-time filter
(stats/garbage.py, `?gt=0` on the team and player pages).

Tiny fixture built in the test: one previous-season game DEN 50-23 BOS with four stints; the
last two are garbage time (4th quarter, benches in, margin 25 then 31).
"""

from __future__ import annotations

import json
import sqlite3
import warnings

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import garbage as G  # noqa: E402
from courtside.stats import percentiles as PC  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.web.app import create_app, make_env  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
Y = S.start_year(CUR)
DEN5, DENB = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
BOS5, BOSB = [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]


# ------------------------------------------------------------------ percentiles
def test_pct_direction_and_ties():
    hi = PC.pct_values({"a": 10, "b": 20, "c": 30, "d": None})
    assert hi == {"a": 0.0, "b": 50.0, "c": 100.0}             # None: no percentile
    lo = PC.pct_values({"a": 10, "b": 20, "c": 30}, higher_better=False)
    assert lo == {"a": 100.0, "b": 50.0, "c": 0.0}             # TOV / DRtg: lower is better
    tie = PC.pct_values({"a": 5, "b": 5, "c": 1, "d": 9})
    assert tie["a"] == tie["b"] == pytest.approx(100 * (1 + 0.5) / 3)   # beats c, ties each other
    assert tie["c"] == 0.0 and tie["d"] == 100.0
    assert PC.pct_values({"a": 3, "b": 3}) == {"a": 50.0, "b": 50.0}
    assert PC.pct_values({"a": 3}) == {"a": None}              # a pool of one: no percentile


def _macro(src: str) -> str:
    env = make_env()
    return env.from_string('{% import "_stats.html" as s %}' + src).render()


def test_pct_bar_render():
    html = _macro("{{ s.pct_bar(92.4, of='30 teams') }}")
    assert 'class="pctb"' in html and "<b>92</b>" in html
    assert "pq4" in html and "width:92.4%" in html and "92nd percentile of 30 teams" in html
    low = _macro("{{ s.pct_bar(7) }}")
    assert "pq0" in low and "<b>7</b>" in low
    assert "pq2" in _macro("{{ s.pct_bar(40) }}") and "pq1" in _macro("{{ s.pct_bar(39.9) }}")
    assert "pctb-l" in _macro("{{ s.pct_bar(50, label='pct') }}")
    assert _macro("{{ s.pct_bar(None) }}").strip() == ""
    assert "<b>100</b>" in _macro("{{ s.pct_bar(130) }}")      # clipped to 0..100
    tile = _macro("{{ s.tile_pct('Pts', '27.7', rank=8, n=398, pct=98.2) }}")
    assert 'class="rk"' in tile and "pctb-tile" in tile        # the chip stays
    cell = _macro("{{ s.ntd_pct(1.5, rank=2, n=30, pct=96.5, group='g') }}")
    assert 'data-seg-value="rk"' in cell and 'data-seg-value="pct" hidden' in cell


# ------------------------------------------------------------------ garbage-time marking
@pytest.mark.parametrize("left,margin,expected", [
    (13 * 60, 40, False),          # before 12:00 left
    (720, 25, True), (720, 24, False), (541, 25, True), (541, 24, False),
    (540, 20, True), (540, 19, False), (361, 20, True), (361, 19, False),
    (360, 10, True), (360, 9, False), (5, -12, True),            # either team ahead
])
def test_thresholds(left, margin, expected):
    assert G.is_garbage_start(4, G.REG_END - left, margin) is expected


def test_threshold_quarter_and_starters():
    assert G.threshold(3, 2100) is None and G.threshold(5, 2900) is None
    assert G.threshold(4, 2160) == 25 and G.threshold(4, 2340) == 20 and G.threshold(4, 2520) == 10
    assert G.is_garbage_start(4, 2600, 30, starters_on=2) is True
    assert G.is_garbage_start(4, 2600, 30, starters_on=3) is False


def _st(no, period, start, end, h5, a5, hp, ap, hq=5.0, aq=5.0):
    return {"stint_no": no, "period": period, "start_sec": start, "end_sec": end,
            "home5": h5, "away5": a5, "home_pts": hp, "away_pts": ap,
            "home_poss": hq, "away_poss": aq}


def test_margin_accumulation():
    sts = [_st(0, 1, 0, 300, DEN5, BOS5, 10, 4), _st(1, 1, 300, 720, DEN5, BOS5, 3, 9),
           _st(2, 2, 720, 900, DEN5, BOS5, 0, 2), _st(3, 2, 900, 1440, DEN5, BOS5, 7, 0)]
    assert G.margins(sts) == [0, 6, 0, -2]


def test_mark_game_suffix_starters_and_reset():
    first = _st(0, 1, 0, 2160, DEN5, BOS5, 50, 25)                 # +25 entering the 4th
    # benches in at 12:00 with +25, still +25 at 8:00, +31 at 3:00 -> both 4th-q stints
    g = [first, _st(1, 4, 2160, 2400, DENB, BOSB, 2, 2), _st(2, 4, 2400, 2700, DENB, BOSB, 6, 0),
         _st(3, 4, 2700, 2880, DENB, BOSB, 0, 0)]
    assert G.mark_game(g) == {1, 2, 3}
    # starters still in (5 + 5): never garbage time
    g2 = [first, _st(1, 4, 2160, 2880, DEN5, BOSB, 2, 2)]
    assert G.mark_game(g2) == set()
    # a comeback inside the window resets the clock: only the final stretch counts
    g3 = [first, _st(1, 4, 2160, 2400, DENB, BOSB, 0, 18),          # 25 -> 7
          _st(2, 4, 2400, 2700, DENB, BOSB, 10, 0),                  # at 8:00: +7 < 20
          _st(3, 4, 2700, 2880, DENB, BOSB, 0, 0)]                   # at 3:00: +17 >= 10
    assert G.mark_game(g3) == {3}
    # overtime: the game got close, nothing is garbage
    g4 = [first, _st(1, 4, 2160, 2880, DENB, BOSB, 0, 25), _st(2, 5, 2880, 3180, DENB, BOSB, 3, 0)]
    assert G.mark_game(g4) == set()
    assert G.mark_game([]) == set()


# ------------------------------------------------------------------ fixture DB
def _tg(conn, gid, team, opp, home, pts, opp_pts, poss):
    conn.execute(
        "INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, is_home, "
        "wl, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, pts, "
        "plus_minus) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, team, PREV, S.REGULAR, f"{Y - 1}-12-01", opp, int(home),
         "W" if pts > opp_pts else "L", 240.0, 20, 45, 5, 15, 5, 8, 5, 20, 25, 10, 4, 3, 8, 15,
         pts, float(pts - opp_pts)))
    conn.execute("INSERT INTO team_games_adv (game_id, team, poss) VALUES (?,?,?)",
                 (gid, team, poss))


def _pg(conn, pid, name, team, pts, mins=30.0, gid="g1"):
    conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, season, "
                 "season_type, game_date, min, pts, reb, ast, fgm, fga, fta) "
                 "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (gid, pid, name, team, PREV, S.REGULAR, f"{Y - 1}-12-01", mins, pts, 5, 3,
                  3, 10, 2))


def _stint(conn, s):
    conn.execute("INSERT INTO stints (game_id, stint_no, period, start_sec, end_sec, home5, "
                 "away5, home_pts, away_pts, home_poss, away_poss) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 ("g1", s["stint_no"], s["period"], s["start_sec"], s["end_sec"],
                  json.dumps(s["home5"]), json.dumps(s["away5"]), s["home_pts"], s["away_pts"],
                  s["home_poss"], s["away_poss"]))


def populate(conn):
    _tg(conn, "g1", "DEN", "BOS", True, 50, 23, 43)
    _tg(conn, "g1", "BOS", "DEN", False, 23, 50, 43)
    _tg(conn, "g2", "DEN", "BOS", True, 50, 23, 43)                 # no stints for g2
    _tg(conn, "g2", "BOS", "DEN", False, 23, 50, 43)
    for s in (_st(0, 1, 0, 720, DEN5, BOS5, 30, 10, 20, 20),
              _st(1, 4, 2160, 2400, DENB, BOSB, 10, 5, 10, 10),     # +20 at 12:00: not yet
              _st(2, 4, 2400, 2700, DENB, BOSB, 8, 2, 8, 8),        # +25 at 8:00: garbage
              _st(3, 4, 2700, 2880, DENB, BOSB, 2, 6, 5, 5)):       # +31 at 3:00: garbage
        _stint(conn, s)
    for gid in ("g1", "g2"):                                        # 2 games: all qualify
        for pid in DEN5:
            _pg(conn, pid, f"Den {pid}", "DEN", 6, gid=gid)
        for pid in DENB:
            _pg(conn, pid, f"Den {pid}", "DEN", 4, mins=18.0, gid=gid)
        for pid in BOS5 + BOSB:
            _pg(conn, pid, f"Bos {pid}", "BOS", 2, mins=24.0, gid=gid)
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


def test_season_pass_and_ratings_with_and_without_garbage(conn):
    sp = G.season_pass(conn, PREV)
    assert sp["games"] == 1 and sp["garbage"] == {"g1": frozenset({2, 3})}
    den = sp["teams"]["DEN"]
    assert den["all"][:4] == [50, 23, 43, 43] and den["xgt"][:4] == [40, 15, 30, 30]
    inc = G.team_ratings(conn, PREV, exclude=False)["by_team"]["DEN"]
    exc = G.team_ratings(conn, PREV, exclude=True)["by_team"]["DEN"]
    assert inc["ortg"] == pytest.approx(100 * 50 / 43) and inc["drtg"] == pytest.approx(100 * 23 / 43)
    assert exc["ortg"] == pytest.approx(100 * 40 / 30) and exc["net"] == pytest.approx(100 * 25 / 30)
    assert exc["pace"] == pytest.approx(2880 * 30 / 960)            # poss per 48 of 16 min
    assert exc["gt_share"] == pytest.approx(13 / 43)
    assert exc["rk"]["drtg"] == 1 and exc["pct"]["drtg"] == 100.0   # lower DRtg is better
    bos = G.team_ratings(conn, PREV, exclude=True)["by_team"]["BOS"]
    assert bos["rk"]["net"] == 2 and bos["pct"]["net"] == 0.0
    # on/off + lineups without garbage time: the DEN bench unit only has stint 1 left
    st = G.team_stints(conn, "DEN", PREV, exclude=True)
    assert st["team"]["ortg"] == pytest.approx(100 * 40 / 30)
    full = G.team_stints(conn, "DEN", PREV, exclude=False)
    assert full["team"]["ortg"] == pytest.approx(100 * 50 / 43)
    bench = next(p for p in st["onoff"] if p["player_id"] == 6)
    assert bench["on_poss"] == pytest.approx(10)
    # player on-court net with and without garbage time
    on = G.player_onoff(conn, 6, PREV, exclude=False)
    off = G.player_onoff(conn, 6, PREV, exclude=True)
    assert on["on"]["net"] == pytest.approx(100 * (20 - 13) / 23)
    assert off["on"]["net"] == pytest.approx(100 * 5 / 10)
    assert off["teams"][0]["off"]["net"] == pytest.approx(100 * 20 / 20)
    assert G.player_onoff(conn, 6, CUR)["games"] == 0               # no stints: empty state


def test_player_and_team_pcts(conn):
    pp = PC.player_pcts(conn, PREV)
    # 20 qualified: 5 at 6 pts, 5 at 4, 10 at 2; ties share half of their tie group
    assert pp[1]["pts"] == pp[2]["pts"] == pytest.approx(100 * (15 + 2) / 19)
    assert pp[6]["pts"] == pytest.approx(100 * (10 + 2) / 19)
    assert pp[16]["pts"] == pytest.approx(100 * 4.5 / 19)
    tp = PC.team_pcts(conn, PREV)
    assert tp["DEN"]["ortg"] == 100.0 and tp["DEN"]["drtg"] == 100.0 and tp["BOS"]["net"] == 0.0


def test_routes_gt_param_and_percentiles(client):
    html = client.get(f"/team/DEN?season={PREV}").text
    assert "pctb" in html and "All minutes" in html and "Excl. garbage time" in html
    assert "excl. garbage time" not in html                         # default: include
    gt = client.get(f"/team/DEN?season={PREV}&gt=0")
    assert gt.status_code == 200
    assert "excl. garbage time" in gt.text and "133.3" in gt.text   # 100 x 40 / 30
    assert "30.2% of DEN possessions were garbage time" in gt.text
    assert "&amp;gt=0" in gt.text or "&gt=0" in gt.text             # links keep the filter
    cur = client.get("/team/DEN?gt=0")
    assert cur.status_code == 200
    teams = client.get(f"/teams?season={PREV}").text
    assert 'data-seg-group="tmpct"' in teams and "pctb" in teams
    p = client.get(f"/player/6?season={PREV}")
    assert p.status_code == 200 and "On / off" in p.text and "every minute included" in p.text
    p0 = client.get(f"/player/6?season={PREV}&gt=0")
    assert p0.status_code == 200 and "excl. garbage time" in p0.text
    assert "Cleaning the Glass" in p0.text
    assert client.get(f"/player/6?season={CUR}&gt=0").status_code == 200
