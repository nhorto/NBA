"""Compare tool (/compare) and the "Similar players" card (docs/STATS_SITE.md):
stats/compare.py, stats/similar.py, web/stats_compare.py. Tiny fixture DBs built in the test
(conftest's data_dir points COURTSIDE_DATA at a tmp dir).
"""

from __future__ import annotations

import re
import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import compare as C  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.stats import similar as SIM  # noqa: E402
from courtside.web import stats_compare as WC  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
Y = S.start_year(CUR)
JOKIC, SGA, CURRY = 203999, 1628983, 201939

_PG = ("INSERT INTO player_games (game_id, player_id, player_name, team, season, season_type, "
       "game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pts, "
       "plus_minus) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
_ADV = ("INSERT INTO player_games_adv (game_id, player_id, usg_pct, ast_pct, reb_pct) "
        "VALUES (?,?,?,?,?)")


def _team_game(conn, gid, day, season, home, away, hpts, apts, stype=S.REGULAR):
    for team, opp, is_home, pts, opts in ((home, away, 1, hpts, apts), (away, home, 0, apts, hpts)):
        conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, "
                     "is_home, wl, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, tov, "
                     "pts, plus_minus) VALUES (?,?,?,?,?,?,?,?,240,40,88,12,34,?,22,10,34,44,25,13,"
                     "?,?)",
                     (gid, team, season, stype, day, opp, is_home, "W" if pts > opts else "L",
                      pts - 92, pts, pts - opts))


def _line(conn, gid, pid, name, team, season, day, *, mn=34.0, pts=25, reb=10, ast=8, stl=1,
          blk=1, tov=3, fg3m=2, fg3a=5, fgm=9, fga=17, ftm=5, fta=6, usg=.28, astp=.35,
          rebp=.17, stype=S.REGULAR):
    conn.execute(_PG, (gid, pid, name, team, season, stype, day, mn, fgm, fga, fg3m, fg3a, ftm,
                       fta, 2, reb - 2, reb, ast, stl, blk, tov, pts, 4))
    conn.execute(_ADV, (gid, pid, usg, astp, rebp))


def populate(conn) -> None:
    """DEN (Jokic) vs OKC (SGA) twice this season, once last season; Curry (GSW) never meets
    SGA's OKC in the fixture but plays DEN once."""
    games = [("c1", f"{Y}-10-22", CUR, "DEN", "OKC", 120, 110),
             ("c2", f"{Y}-10-24", CUR, "OKC", "DEN", 118, 101),
             ("c3", f"{Y}-10-26", CUR, "GSW", "DEN", 99, 104),
             ("p1", f"{Y - 1}-12-01", PREV, "OKC", "DEN", 100, 98)]
    for gid, day, season, home, away, hp, ap in games:
        _team_game(conn, gid, day, season, home, away, hp, ap)
    for gid, day, season, home, away, _, _ in games:
        if "DEN" in (home, away):
            _line(conn, gid, JOKIC, "Nikola Jokic", "DEN", season, day, pts=30, reb=13, ast=10,
                  usg=.30, astp=.45, rebp=.20)
        if "OKC" in (home, away):
            _line(conn, gid, SGA, "Shai Gilgeous-Alexander", "OKC", season, day, pts=32, reb=5,
                  ast=6, fg3m=2, fgm=11, fga=20, ftm=8, fta=9, usg=.33, astp=.30, rebp=.08)
        if "GSW" in (home, away):
            _line(conn, gid, CURRY, "Stephen Curry", "GSW", season, day, pts=28, reb=4, ast=6,
                  fg3m=5, fg3a=12, usg=.31, astp=.30, rebp=.07)
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team, position, "
                     "height, height_in, from_year) VALUES (?,?,?,?,?,?,?,?)",
                     [(JOKIC, "Nikola", "Jokić", "DEN", "C", "6-11", 83, 2015),
                      (SGA, "Shai", "Gilgeous-Alexander", "OKC", "G", "6-6", 78, 2018),
                      (CURRY, "Stephen", "Curry", "GSW", "G", "6-2", 74, 2009)])
    sh = ("INSERT INTO shots (game_id, action_number, player_id, team, season, season_type, "
          "game_date, x, y, made, zone) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    rows = []
    for i in range(12):     # Jokic: 12 at the rim, 9 made; SGA: 12 at the rim, 6 made
        rows.append(("c1", i, JOKIC, "DEN", CUR, S.REGULAR, f"{Y}-10-22", 0, 5, int(i < 9),
                     "Restricted Area"))
        rows.append(("c1", 100 + i, SGA, "OKC", CUR, S.REGULAR, f"{Y}-10-22", 0, 5, int(i < 6),
                     "Restricted Area"))
    rows.append(("c1", 200, SGA, "OKC", CUR, S.REGULAR, f"{Y}-10-22", 150, 200, 1, "Mid-Range"))
    conn.executemany(sh, rows)
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


# ------------------------------------------------------------------ percentile + leader math
def test_percentile_rank_mid_rank_and_flip():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert C.percentile_rank(vals, 3.0) == pytest.approx(62.5)        # 2 below + half a tie
    assert C.percentile_rank(vals, 3.0, higher_better=False) == pytest.approx(37.5)
    assert C.percentile_rank(vals, 5.0) == 100.0 and C.percentile_rank(vals, 0.0) == 0.0
    assert C.percentile_rank(vals, 4.0) == pytest.approx(87.5)        # the best is not 100
    assert C.percentile_rank([1.0, 2.0, 2.0, 3.0], 2.0) == pytest.approx(50.0)   # ties
    assert C.percentile_rank(vals, None) is None and C.percentile_rank([], 1.0) is None


def test_leaders_direction_ties_and_missing():
    assert WC.leaders([1, 3, 3], 1) == [False, True, True]
    assert WC.leaders([1.5, 3, None], -1) == [True, False, False]
    assert WC.leaders([2, 2], 1) == [False, False]                     # all equal: nobody
    assert WC.leaders([5, None], 1) == [False, False]                  # one value: nobody
    assert WC.leaders([1, 9], 0) == [False, False]                     # neutral row


def test_pools_and_player_column(dbconn):
    pl = C.pools(dbconn, CUR)
    assert pl["_n"] == 2 and pl["pg.pts"] == [30.0, 32.0]         # Curry: 1 game, unqualified
    col = C.player(dbconn, SGA, CUR)
    assert col["played"] and col["values"]["pg.pts"] == 32
    assert col["pctile"]["pg.pts"] == pytest.approx(75.0)
    assert col["pctile"]["pg.reb"] == pytest.approx(25.0)
    assert col["pctile"]["pg.tov"] == pytest.approx(50.0)              # tie, flipped
    assert col["values"]["adv.usg"] == pytest.approx(.33)
    assert col["values"]["p36.pts"] == pytest.approx(32 / 34 * 36)
    assert col["form"]["n"] == 2 and col["form"]["avg"]["pts"] == 32
    assert {z["zone"] for z in col["zones"]} == {"Restricted Area", "Mid-Range"}
    none = C.player(dbconn, CURRY, PREV)
    assert not none["played"] and none["values"].get("pg.pts") is None


def test_head_to_head_only_opposite_teams(dbconn):
    h = C.head_to_head(dbconn, [JOKIC, SGA])
    assert [g["game_id"] for g in h["games"]] == ["c2", "c1", "p1"]   # newest first
    s = h["summary"][JOKIC]
    assert s["games"] == 3 and (s["w"], s["l"]) == (1, 2)
    assert h["games"][1]["home"] == "DEN" and h["games"][1]["home_pts"] == 120
    assert C.head_to_head(dbconn, [SGA, CURRY])["games"] == []        # never met
    assert C.head_to_head(dbconn, [SGA])["games"] == []


def test_team_factors_and_h2h(dbconn):
    ff = C.team_factors(dbconn, CUR)
    den = ff["by_team"]["DEN"]
    assert den["efg"] == pytest.approx((40 + 0.5 * 12) / 88)
    assert den["ftr"] == pytest.approx(22 / 88)
    assert den["drb"] == pytest.approx(1 - 10 / 44) and den["rk"]["efg"] == 1
    h = C.team_h2h(dbconn, "DEN", "OKC", (CUR, PREV))
    assert [b["season"] for b in h] == [CUR, PREV]
    assert (h[0]["a_w"], h[0]["b_w"]) == (1, 1) and h[1]["games"][0]["winner"] == "OKC"


# ------------------------------------------------------------------ routes
def test_compare_empty_page(client):
    r = client.get("/compare")
    assert r.status_code == 200
    assert "Pick two to four players" in r.text and 'aria-current="page"' in r.text
    assert re.search(r'class="top-link is-active">Players<', r.text)


def test_compare_players_page(client):
    r = client.get(f"/compare?p={JOKIC}&p={SGA}&season={CUR}")
    assert r.status_code == 200
    t = _text(r.text)
    for s in ("Nikola Jokić", "Shai Gilgeous-Alexander", "Side by side", "Per 36 minutes",
              "USG%", "AST%", "REB%", "DK value", "Last 10 games", "Head-to-head",
              "Shot zones", "cmp-pb", "cmp-lead", "best: "):
        assert s in t, s
    # Jokic leads rebounds (bold + marker); SGA's 9-of-12 vs 6-of-12 at the rim: Jokic leads
    assert re.search(r'cmp-lead"><div class="cmp-v"><span class="cmp-mk"[^>]*>▲</span>'
                     r'<span class="sr">best: </span>13\.0', t)
    assert "75.0%" in t and "OKC 118 @ DEN" not in t and "DEN 101 @ OKC 118" in t
    assert f'href="/compare?p={SGA}" aria-label="Remove Nikola Jokić"' in t   # remove link


def test_compare_bad_ids(client):
    assert client.get("/compare?p=abc").status_code == 404
    assert client.get("/compare?p=999999").status_code == 404
    assert client.get("/compare?t=XXX").status_code == 404
    r = client.get(f"/compare?p={JOKIC}&p=999999")
    assert r.status_code == 200 and "No player with id 999999" in r.text
    r = client.get(f"/compare?p={JOKIC}&p={JOKIC}")                     # dedup: one column
    assert r.status_code == 200 and "Add another player" in r.text


def test_compare_add_resolves_and_redirects(client):
    r = client.get(f"/compare?p={JOKIC}&add=p:{SGA}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/compare?p={JOKIC}&p={SGA}"
    r = client.get(f"/compare?p={JOKIC}&add=gilgeous", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith(f"p={SGA}")
    r = client.get("/compare?t=DEN&add=okc", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/compare?t=DEN&t=OKC"
    r = client.get(f"/compare?p={JOKIC}&add=zzzz")
    assert r.status_code == 200 and "No player matches" in r.text


def test_compare_teams_page(client):
    r = client.get(f"/compare?t=DEN&t=OKC&season={CUR}")
    assert r.status_code == 200
    t = _text(r.text)
    for s in ("Compare teams", "Denver Nuggets", "Oklahoma City Thunder", "Off rating",
              "Net rating", "Four factors", "Opp eFG%", "DREB%", "Head-to-head", "class=\"rk\""):
        assert s in t, s
    assert "DEN <b>1</b>&ndash;<b>1</b> OKC" in t and "<b>OKC 100</b>" in t   # last season
    r = client.get("/compare?t=DEN&t=OKC&t=BOS")
    assert r.status_code == 200 and "Two teams at a time" in r.text


def test_compare_buttons_on_player_and_team_pages(client):
    r = client.get(f"/player/{JOKIC}")
    assert r.status_code == 200 and f'href="/compare?p={JOKIC}&amp;season=' in r.text
    assert "Similar players" in r.text                     # card (thin/none state here)
    r = client.get("/team/DEN")
    assert r.status_code == 200 and 'href="/compare?t=DEN&amp;season=' in r.text


# ------------------------------------------------------------------ similarity
SIM_SEASON = "2010-11"


def _sim_populate(conn) -> None:
    """One season of synthetic player-seasons (30 games x 36 min = 1,080 min each): a target,
    a near-duplicate, the target's own other season, and ten different profiles."""
    rng = np.random.default_rng(7)
    base = dict(pts=24, reb=11, ast=7, stl=1, blk=1, tov=3, fg3m=1, fg3a=3, fgm=9, fga=18,
                ftm=5, fta=7, usg=.27, astp=.30, rebp=.17)
    players = {1: base, 2: {**base, "pts": 24, "reb": 11, "ast": 7}}   # 2 = near-duplicate
    for i in range(10):
        players[10 + i] = dict(pts=int(rng.integers(6, 30)), reb=int(rng.integers(2, 13)),
                               ast=int(rng.integers(1, 10)), stl=int(rng.integers(0, 3)),
                               blk=int(rng.integers(0, 3)), tov=int(rng.integers(1, 4)),
                               fg3m=int(rng.integers(0, 4)), fg3a=int(rng.integers(4, 10)),
                               fgm=int(rng.integers(3, 10)), fga=int(rng.integers(10, 20)),
                               ftm=int(rng.integers(0, 4)), fta=int(rng.integers(4, 6)),
                               usg=float(rng.uniform(.12, .32)), astp=float(rng.uniform(.05, .4)),
                               rebp=float(rng.uniform(.04, .2)))
    heights = {1: 81, 2: 81} | {10 + i: 74 + i for i in range(10)}
    for pid, st in players.items():
        conn.execute("INSERT INTO player_index (player_id, first_name, last_name, height_in, "
                     "from_year) VALUES (?, 'P', ?, ?, 2008)", (pid, f"Number{pid}", heights[pid]))
        for g in range(30):
            jitter = 1 if (pid == 2 and g % 10 == 0) else 0          # near, not identical
            _line(conn, f"s{pid}-{g}", pid, f"P{pid}", "BOS", SIM_SEASON,
                  f"2011-01-{g % 28 + 1:02d}", mn=36.0, **{**st, "pts": st["pts"] + jitter})
    # the target's own (identical) profile in another season must never be listed
    for g in range(30):
        _line(conn, f"o1-{g}", 1, "P1", "BOS", "2011-12", f"2012-01-{g % 28 + 1:02d}", mn=36.0,
              **base)
    conn.commit()


@pytest.fixture
def simconn(data_dir):
    sdb.clear()
    conn = db.connect()
    _sim_populate(conn)
    yield conn
    conn.close()


def test_distance_is_weighted_rms():
    z = np.array([[0.0] * len(SIM.FEATURES), [1.0] * len(SIM.FEATURES)])
    d = SIM.distances(z, np.zeros(len(SIM.FEATURES)))
    assert d[0] == 0 and d[1] == pytest.approx(1.0)
    assert SIM.score(0) == 100 and SIM.score(SIM.SCALE) == 0 and SIM.score(9) == 0


def test_similarity_near_duplicate_ranks_first(simconn):
    res = SIM.similar(simconn, 1, SIM_SEASON, TODAY)
    assert res["status"] == "ok" and len(res["rows"]) == SIM.TOP_N
    ids = [r["player_id"] for r in res["rows"]]
    assert ids[0] == 2 and 1 not in ids                     # near-duplicate first, never himself
    assert len(set(ids)) == len(ids)                         # each player once
    assert res["rows"][0]["score"] > 90 > res["rows"][-1]["score"]
    scores = [r["score"] for r in res["rows"]]
    assert scores == sorted(scores, reverse=True)
    t0 = time.perf_counter()
    SIM.similar(simconn, 1, SIM_SEASON, TODAY)               # warm: cached
    assert time.perf_counter() - t0 < 0.03


def test_similarity_states(simconn):
    assert SIM.similar(simconn, 1, "2005-06", TODAY)["status"] == "none"
    prof = SIM.season_profiles(simconn, SIM_SEASON)[1]
    assert prof["pts36"] == pytest.approx(24) and prof["usg"] == pytest.approx(.27)
    assert prof["age"] == pytest.approx(SIM.ROOKIE_AGE + 2) and prof["age_est"]
