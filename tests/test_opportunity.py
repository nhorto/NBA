"""stats/opportunity.py + analytics/xfg_store.py + tracking_ext_store parsing on tiny fixtures."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from courtside import db
from courtside.analytics import tracking_ext_store as tx, xfg_store
from courtside.sources import nba_players
from courtside.stats import opportunity as op

S = "2025-26"
FIX = Path(__file__).parent / "fixtures"


def gid(i: int) -> str:
    return f"00225{i:05d}"


def _pg(conn, g, pid, team, day, **box):
    db.upsert_rows(conn, "player_games", [{"game_id": g, "player_id": pid, "player_name": f"P{pid}",
                                           "team": team, "season": S, "season_type":
                                           "Regular Season", "game_date": day, "min": 30, **box}])


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.upsert_rows(c, "team_rosters", [
        {"season": S, "team_id": 1, "team": "AAA", "player_id": 1, "position": "G"},
        {"season": S, "team_id": 1, "team": "AAA", "player_id": 2, "position": "C-F"},
        {"season": S, "team_id": 1, "team": "AAA", "player_id": 3, "position": "F"}])
    for i in range(1, 13):
        g, day = gid(i), f"2025-11-{i:02d}"
        # player 1: 5 reb on 10 chances in games 1-10, then 8 on 10 in games 11-12
        reb = 8 if i > 10 else 5
        _pg(c, g, 1, "AAA", day, reb=reb, oreb=1, dreb=reb - 1, ast=6, stl=1, blk=0, fgm=5,
            fga=10, pts=12)
        db.upsert_rows(c, "player_tracking", [{
            "game_id": g, "player_id": 1, "team": "AAA", "minutes": 30,
            "reb_chances_off": 3, "reb_chances_def": 7, "reb_chances_total": 10,
            "passes": 50, "touches": 70, "secondary_ast": 1, "ft_ast": 0,
            "def_rim_fgm": 2, "def_rim_fga": 4}])
        db.upsert_rows(c, "player_hustle", [{"game_id": g, "player_id": 1, "team": "AAA",
                                             "deflections": 3, "contested_shots": 6,
                                             "contested_2pt": 4, "contested_3pt": 2,
                                             "charges_drawn": 0, "loose_total": 1, "box_outs": 2}])
        if i % 2 == 0:              # potential assists collected on even games only
            db.upsert_rows(c, "player_tracking_ext", [{"game_id": g, "player_id": 1, "team": "AAA",
                                                       "game_date": day, "season": S,
                                                       "pot_ast": 10, "drives": 8}])
        # player 2 (C): 12 reb on 20 chances every game
        _pg(c, g, 2, "AAA", day, reb=12, oreb=4, dreb=8, ast=2, stl=0, blk=2, fgm=6, fga=9,
            pts=14)
        db.upsert_rows(c, "player_tracking", [{
            "game_id": g, "player_id": 2, "team": "AAA", "minutes": 30,
            "reb_chances_off": 7, "reb_chances_def": 13, "reb_chances_total": 20,
            "passes": 30, "touches": 40, "def_rim_fgm": 3, "def_rim_fga": 8}])
    # player 3: a game with no tracking at all (must not count as 0 chances)
    _pg(c, gid(1), 3, "AAA", "2025-11-01", reb=4, oreb=0, dreb=4, ast=1, fgm=1, fga=3, pts=2)
    return c


# ------------------------------------------------------------------------------- rebounds

def test_rebound_opportunity_vs_his_season_and_league(conn):
    r = op.rebound_opportunity(conn, 1, [gid(11), gid(12)])
    assert r["season"] == S and r["position"] == "G"
    assert (r["reb"], r["chances"], r["conv"]) == (16, 20, 0.8)
    assert r["season_chances"] == 120 and r["season_conv"] == round(66 / 120, 3)
    assert r["league_conv_by_pos"]["G"] == round(66 / 120, 3)
    assert r["league_conv_by_pos"]["C"] == 0.6
    assert r["league_conv_by_pos"]["all"] == round((66 + 144) / (120 + 240), 3)
    base = (66 + op.K_CONV["reb"] * 66 / 120) / (120 + op.K_CONV["reb"])
    assert r["baseline_conv"] == round(base, 3)
    assert r["expected_reb"] == round(20 * base, 2)
    assert r["reb_over_expected"] == round(16 - 20 * base, 2)
    assert r["off"]["stat"] == 2 and r["off"]["opps"] == 6
    assert r["def"]["conv"] == round(14 / 14, 3)


def test_games_without_tracking_do_not_count_as_zero_chances(conn):
    r = op.rebound_opportunity(conn, 3, [gid(1)])
    assert r["games_selected"] == 1 and r["games"] == 0
    assert r["chances"] == 0 and r["conv"] is None and r["expected_reb"] is None


def test_whole_season_when_game_ids_is_none(conn):
    r = op.rebound_opportunity(conn, 2, None, season=S)
    assert (r["reb"], r["chances"], r["games"]) == (144, 240, 12)


# -------------------------------------------------------------------------------- assists

def test_assist_opportunity_uses_games_with_potential_assists(conn):
    a = op.assist_opportunity(conn, 1, [gid(i) for i in range(1, 13)])
    assert a["ast"] == 72 and a["passes"] == 600 and a["touches"] == 840
    assert a["pot_ast"] == 60 and a["pot_ast_games"] == 6
    assert a["ast_per_pot_ast"] == round(36 / 60, 3)          # 6 even games x 6 AST / 10
    assert a["ast_per_pass"] == round(72 / 600, 3)
    assert a["drives"] == 48 and a["secondary_ast"] == 12
    b = op.assist_opportunity(conn, 1, [gid(1)])               # odd game: none collected
    assert b["pot_ast"] is None and b["ast_per_pot_ast"] is None and b["passes"] == 50


# -------------------------------------------------------------------------------- defense

def test_defense_activity(conn):
    d = op.defense_activity(conn, 1, [gid(1), gid(2)])
    assert d["deflections"] == 6 and d["deflections_pg"] == 3.0 and d["contested_3pt"] == 4
    assert d["rim_fga_defended"] == 8 and d["rim_fg_allowed"] == 0.5
    assert d["league_rim_fg_allowed"] == 0.5                   # G group: player 1 only
    assert d["stl_per_deflection"] == round(2 / 6, 3) and d["stl"] == 2
    d2 = op.defense_activity(conn, 2, [gid(1)])
    assert d2["hustle_games"] == 0 and d2["deflections"] is None and d2["blk"] == 2


def test_missing_tables_give_the_empty_state():
    c = sqlite3.connect(":memory:")
    assert op.rebound_opportunity(c, 1, ["x"])["conv"] is None
    assert op.assist_opportunity(c, 1, ["x"])["ast"] == 0
    assert op.shot_quality(c, 1, ["x"])["fga"] == 0
    assert op.regression_flags(c, S, 5) == {"season": S, "last_n": 5, "hot": [], "cold": []}


# -------------------------------------------------------------------------------- xFG

def test_dist_band_matches_the_projection_models_bins():
    import pandas as pd
    from courtside.proj.features import shots as F4
    dists = [0, 0.4, 1, 3.99, 4, 13.5, 29.9, 30, 35.9, 36, 46.9, 47, 60, None]
    for is3 in (0, 1):
        bins = F4.dist_bin(pd.DataFrame({"dist": [float("nan") if d is None else d
                                                  for d in dists], "is3": is3}))
        labels = [xfg_store.dist_band(d, "3pt" if is3 else "2pt") for d in dists]
        # one label per F4 bin and vice versa
        assert len(set(zip(bins, labels))) == len(set(bins)) == len(set(labels))
    assert xfg_store.dist_band(2.7, "2pt") == "2pt 02"
    assert xfg_store.dist_band(25.2, "3pt") == "3pt 25"
    assert xfg_store.dist_band(40, "3pt") == "3pt 36-46"
    assert [xfg_store.family(s) for s in ("Jump Shot", "jumpshot", "DUNK", "Hook", "Tip", None)] \
        == ["jumpshot", "jumpshot", "dunk", "hook", "other", "other"]


def _shot(g, n, pid, team, zone, dist, stype, sub, made):
    return {"game_id": g, "action_number": n, "player_id": pid, "team": team, "season": S,
            "season_type": "Regular Season", "game_date": "2025-11-01", "zone": zone,
            "distance": dist, "shot_type": stype, "sub_type": sub, "made": made}


def _seed_shots(c):
    rows, n = [], 0
    # league: rim dunks 90%, rim layups 60%, corner threes 40%; player 9 shoots 15 dunks / 15 made
    for i in range(200):
        n += 1
        rows.append(_shot(gid(1), n, 8, "BBB", "Restricted Area", 1.0, "2pt", "Layup", int(i % 5 < 3)))
        n += 1
        rows.append(_shot(gid(1), n, 8, "BBB", "Left Corner 3", 22.5, "3pt", "Jump Shot",
                          int(i % 5 < 2)))
    for i in range(100):
        n += 1
        rows.append(_shot(gid(1), n, 7, "BBB", "Restricted Area", 1.2, "2pt", "DUNK",
                          int(i % 10 < 9)))
    for i in range(15):
        n += 1
        rows.append(_shot(gid(2), n, 9, "AAA", "Restricted Area", 0.5, "2pt", "DUNK", 1))
    db.upsert_rows(c, "shots", rows)


def test_xfg_fit_is_calibrated_and_ordered(conn):
    _seed_shots(conn)
    r = xfg_store.fit_season(conn, S)
    assert r["fga"] == 515 and r["cells"] == 4
    exp, act = conn.execute(
        "SELECT SUM(fga * xfg), SUM(fgm) FROM shot_xfg WHERE season=? AND dist_band<>'*' "
        "AND action_type<>'*'", (S,)).fetchone()
    assert math.isclose(exp, act, rel_tol=1e-9)                # league xFG% == FG% by construction
    t = xfg_store.XFGTable(conn, [S])
    dunk = t.xfg(S, "Restricted Area", 1.0, "2pt", "DUNK")
    lay = t.xfg(S, "Restricted Area", 1.0, "2pt", "Layup")
    c3 = t.xfg(S, "Left Corner 3", 22.5, "3pt", "Jump Shot")
    assert dunk > 0.85 and 0.55 < lay < 0.65 and 0.35 < c3 < 0.45
    # unseen family -> pooled band; unseen band -> zone; unseen zone -> league
    hook = t.xfg(S, "Restricted Area", 1.0, "2pt", "Hook")
    assert hook == conn.execute("SELECT xfg FROM shot_xfg WHERE zone='Restricted Area' AND "
                                "dist_band='2pt 01' AND action_type='*'").fetchone()[0]
    assert t.xfg(S, "Mid-Range", 15, "2pt", "Jump Shot") == pytest.approx(t.league[S])
    assert t.xfg("1999-00", "Mid-Range", 15, "2pt", "Jump Shot") is None


def test_shot_quality_player_team_and_allowed(conn):
    _seed_shots(conn)
    xfg_store.fit_season(conn, S)
    t = xfg_store.XFGTable(conn, [S])
    q = op.shot_quality(conn, 9, [gid(2)])
    x = t.xfg(S, "Restricted Area", 0.5, "2pt", "DUNK")
    assert q["fga"] == 15 and q["fgm"] == 15 and q["fg_pct"] == 1.0
    assert q["xfg_pct"] == round(x, 3) and q["xpts"] == round(15 * 2 * x, 2)
    assert q["making_pts"] == round(30 - 30 * x, 2) and q["games"] == 1
    assert q["making_per_game"] == round(30 - 30 * x, 2)
    assert q["by_type"]["2pt"]["fga"] == 15 and q["model"] == "shot_xfg"
    team = op.shot_quality(conn, team="BBB", season=S)
    assert team["fga"] == 500 and team["by_type"]["3pt"]["fga"] == 200
    allowed = op.shot_quality(conn, team="AAA", game_ids=[gid(1), gid(2)], allowed=True)
    assert allowed["fga"] == 0                                 # AAA did not play game 1
    allowed = op.shot_quality(conn, team="BBB", season=S, allowed=True)
    assert allowed["fga"] == 0
    league = op.shot_quality(conn, season=S)
    assert math.isclose(league["xfgm"], league["fgm"], abs_tol=0.01)


# -------------------------------------------------------------------------- regression

def test_regression_flags_hot_cold_and_sample_guard(conn):
    # player 4 (F): 5 / 10 per game for 20 games, then 9 / 10 for 5 games -> hot
    # player 5 (F): 6 / 10, then 2 / 10 -> cold;  player 6: hot but only 2 games (8 chances)
    for pid, team, early, late, n_late, ch in ((4, "AAA", 5, 9, 5, 10), (5, "AAA", 6, 2, 5, 10),
                                               (6, "AAA", 2, 4, 2, 4)):
        db.upsert_rows(conn, "team_rosters", [{"season": S, "team_id": 1, "team": team,
                                               "player_id": pid, "position": "F"}])
        for i in range(1, 21 + n_late):
            g, day = f"0022510{pid}{i:02d}", f"2025-12-{i:02d}"
            reb = late if i > 20 else early
            _pg(conn, g, pid, team, day, reb=reb, oreb=0, dreb=reb, ast=0)
            db.upsert_rows(conn, "player_tracking", [{"game_id": g, "player_id": pid, "team": team,
                                                      "reb_chances_total": ch,
                                                      "reb_chances_off": 0,
                                                      "reb_chances_def": ch}])
    f = op.regression_flags(conn, S, last_n=5, metrics=("reb",))
    hot = {x["player_id"]: x for x in f["hot"]}
    cold = {x["player_id"]: x for x in f["cold"]}
    assert 4 in hot and 5 in cold and 6 not in hot and 6 not in cold
    h = hot[4]
    assert (h["n"], h["d"], h["games"], h["rate"]) == (45, 50, 5, 0.9)
    assert 0.5 < h["baseline"] < 0.6 and h["z_shrunk"] < h["z"] and h["good"]
    p0 = h["baseline"]
    z = (45 - 50 * p0) / math.sqrt(50 * p0 * (1 - p0))
    assert h["z"] == pytest.approx(z, abs=0.02)
    assert h["z_shrunk"] == pytest.approx(z * 50 / (50 + op.K_CONV["reb"]), abs=0.02)
    assert cold[5]["rate"] == 0.2 and cold[5]["z_shrunk"] < -1.5


def test_regression_flags_fg_uses_shot_quality(conn):
    rows, n = [], 0
    for i in range(1, 31):                                     # player 11: 30 games x 10 threes
        g = f"00225200{i:02d}"
        _pg(conn, g, 11, "AAA", f"2026-01-{i:02d}", fgm=0, fga=10)
        for k in range(10):
            n += 1
            made = int(k < 9) if i > 25 else int(k < 3)        # 30% then 90% over the last 5
            rows.append({**_shot(g, n, 11, "AAA", "Above the Break 3", 25.0, "3pt",
                                 "Jump Shot", made), "game_date": f"2026-01-{i:02d}"})
    db.upsert_rows(conn, "shots", rows)
    xfg_store.fit_season(conn, S)
    f = op.regression_flags(conn, S, last_n=5, metrics=("fg",))
    assert [x["player_id"] for x in f["hot"]] == [11]
    h = f["hot"][0]
    assert h["d"] == 50 and h["n"] == 45 and h["metric"] == "fg" and h["baseline"] < 0.45


# ------------------------------------------------------------------ tracking dashboards

def test_parse_pt_day_and_match_to_games(conn):
    p = json.loads((FIX / "nba_players/pt_day_passing.json").read_text())
    rows = nba_players.parse_pt_day(p, "Passing")
    assert len(rows) == 5 and {"pot_ast", "passes_made", "ast_pts_created"} <= set(rows[0])
    assert rows[0]["team"] == "OKC" and rows[0]["pot_ast"] == 6
    db.upsert_rows(conn, "team_games", [
        {"game_id": "0022500001", "team": t, "season": S, "season_type": "Regular Season",
         "game_date": "2025-10-21"} for t in ("OKC", "HOU")])
    games = tx.day_games(conn, "2025-10-21")
    n, seen, unmatched = tx.store_rows(conn, "2025-10-21", "Passing", rows, games)
    assert n == 4 and seen == {"OKC", "HOU"} and unmatched == [1628369]       # BOS: no game
    d = json.loads((FIX / "nba_players/pt_day_drives.json").read_text())
    tx.store_rows(conn, "2025-10-21", "Drives", nba_players.parse_pt_day(d, "Drives"), games)
    # Drives fills its own columns and keeps Passing's
    r = conn.execute("SELECT pot_ast, drives, touches, season FROM player_tracking_ext "
                     "WHERE player_id=1628983").fetchone()
    assert tuple(r) == (6, 2, None, S)
    assert tx.measure_stored(conn, {"0022500001"}, "Passing")
    assert not tx.measure_stored(conn, {"0022500001"}, "Possessions")
    assert tx.days_missing(conn, ["2025-10-21"]) == ["2025-10-21"]


# ------------------------------------------------------------------------ player panel

def test_player_opportunity_bundle(conn):
    _seed_shots(conn)
    db.upsert_rows(conn, "shots", [_shot(gid(i), 1000 + i, 1, "AAA", "Left Corner 3", 22.5, "3pt",
                                         "Jump Shot", int(i % 2)) for i in range(1, 13)])
    xfg_store.fit_season(conn, S)
    p = op.player_opportunity(conn, 1, S, "Regular Season")
    assert p["windows"] == ["l5", "l10", "season"] and p["games"] == 12
    assert p["window_games"] == {"l5": 5, "l10": 10, "season": 12}
    assert p["has"] == {"tracking": True, "hustle": True, "pot_ast": True, "shots": True}
    assert p["position"] == "G"
    rb = p["rebounds"]
    assert rb["l5"]["reb"] == 5 * 3 + 8 * 2 and rb["l5"]["chances"] == 50
    assert rb["l5"]["chances_pg"] == 10.0 and rb["season"]["conv"] == round(66 / 120, 3)
    assert rb["baseline"]["own_raw"] == round(66 / 120, 3) and rb["baseline"]["league"] is not None
    a = p["assists"]
    assert a["l10"]["pot_ast_games"] == 5 and a["season"]["passes_pg"] == 50.0
    assert a["baseline"]["own"] == round(36 / 60, 3)
    sh = p["shooting"]
    assert sh["season"]["fga"] == 12 and sh["l5"]["fga"] == 5
    assert sh["baseline"]["league_fg_pct"] is not None
    assert p["defense"]["season"]["deflections_pg"] == 3.0
    assert p["defense"]["baseline"]["own"] == 0.5
    assert isinstance(p["flags"], list)
    assert op.player_opportunity(conn, 1, S, "Playoffs") is None
    assert op.player_opportunity(conn, 3, S, "Regular Season") is None     # no tracking/shots
    assert op.player_opportunity(conn, 999, S, "Regular Season") is None
