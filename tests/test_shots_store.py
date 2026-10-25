"""shots table (migration 17) and analytics/shots_store: pure parser, zones, backfill."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtside import archive, db
from courtside.analytics import shots_store as ss

FIX = Path(__file__).parent / "fixtures"
GID = "0022500001"


def _fga(n, x, y, *, at="2pt", res="Made", pid=101, team="OKC", **kw):
    return {"actionNumber": n, "clock": "PT07M15.00S", "period": 2, "teamTricode": team,
            "actionType": at, "subType": "Jump Shot", "descriptor": "pullup", "personId": pid,
            "xLegacy": x, "yLegacy": y, "shotDistance": None, "isFieldGoal": 1,
            "shotResult": res, **kw}


def _pbp(actions, gid="0022600005"):
    return {"game": {"gameId": gid, "actions": actions}}


@pytest.mark.parametrize("x,y,st,zone", [
    (0, 0, "2pt", "Restricted Area"),
    (30, 30, "2pt", "Restricted Area"),        # 4.2 ft: inside the NBA's labelled RA
    (0, 60, "2pt", "Paint (Non-RA)"),
    (70, 130, "2pt", "Paint (Non-RA)"),
    (90, 100, "2pt", "Mid-Range"),             # outside the lane
    (0, 150, "2pt", "Mid-Range"),              # beyond the free-throw line
    (-225, 20, "3pt", "Left Corner 3"),
    (226, 89, "3pt", "Right Corner 3"),
    (100, 220, "3pt", "Above the Break 3"),
    (-3, 620, "3pt", "Backcourt"),             # heave (the feed calls one of these 'Mid-Range')
    (None, 10, "2pt", None),
])
def test_zone_for(x, y, st, zone):
    assert ss.zone_for(x, y, st) == zone


def test_fmt_clock():
    assert ss.fmt_clock("PT11M39.00S") == "11:39"
    assert ss.fmt_clock("PT00M05.40S") == "0:05.4"
    assert ss.fmt_clock("PT00M00.00S") == "0:00"
    assert ss.fmt_clock(None) is None


def test_parse_shots_handmade():
    acts = [
        {"actionNumber": 1, "actionType": "period", "subType": "start"},
        _fga(4, -225, 20, at="3pt", res="Missed", pid=7),
        {"actionNumber": 5, "actionType": "rebound", "isFieldGoal": 0, "personId": 8},
        _fga(9, 0, 5, subType="DUNK"),
        {"actionNumber": 10, "actionType": "freethrow", "isFieldGoal": 0, "shotResult": "Made"},
        _fga(9, 0, 5, subType="DUNK"),                      # duplicated action: one row
    ]
    rows = ss.parse_shots(_pbp(acts), game_date="2026-10-22")
    assert [r["action_number"] for r in rows] == [4, 9]
    a, b = rows
    assert set(a) == set(ss.COLS)
    assert a == {**a, "game_id": "0022600005", "player_id": 7, "team": "OKC",
                 "season": "2026-27", "season_type": "Regular Season", "game_date": "2026-10-22",
                 "period": 2, "clock": "7:15", "x": -225.0, "y": 20.0, "made": 0,
                 "shot_type": "3pt", "action_type": "3pt", "sub_type": "Jump Shot",
                 "descriptor": "pullup", "zone": "Left Corner 3"}
    assert a["distance"] == pytest.approx(22.59, abs=0.01)     # from x / y when not given
    assert b["made"] == 1 and b["shot_type"] == "2pt" and b["zone"] == "Restricted Area"
    assert b["sub_type"] == "DUNK"


def test_parse_shots_legacy_dialect():
    """2019-20: no isFieldGoal key; Feb-Mar 2020: shotResult ': Made'."""
    acts = [{"actionNumber": 3, "actionType": "2pt", "subType": "dunk", "personId": 5,
             "teamTricode": "MIN", "xLegacy": -8, "yLegacy": 9, "shotDistance": 1.62,
             "shotResult": ": Made", "period": 1, "clock": "PT11M48.00S"},
            {"actionNumber": 4, "actionType": "substitution", "personId": 6}]
    (r,) = ss.parse_shots(_pbp(acts, "0041900101"))
    assert r["made"] == 1 and r["distance"] == 1.62 and r["season"] == "2019-20"
    assert r["season_type"] == "Playoffs" and r["zone"] == "Restricted Area"


def test_parse_real_game_matches_boxscore():
    pbp = archive.load_raw(FIX / "analytics/playbyplay_0022500001.json.gz")
    box = archive.load_raw(FIX / "analytics/boxscore_0022500001.json.gz")["game"]
    rows = ss.parse_shots(pbp, GID, "2025-10-21")
    for side in ("homeTeam", "awayTeam"):
        t, st = box[side]["teamTricode"], box[side]["statistics"]
        mine = [r for r in rows if r["team"] == t]
        assert len(mine) == st["fieldGoalsAttempted"]
        assert sum(r["made"] for r in mine) == st["fieldGoalsMade"]
        assert sum(r["shot_type"] == "3pt" for r in mine) == st["threePointersAttempted"]
    assert all(r["zone"] in ss.ZONES for r in rows)
    corners = [r for r in rows if r["zone"] in ("Left Corner 3", "Right Corner 3")]
    assert corners and all((r["x"] < 0) == (r["zone"] == "Left Corner 3") for r in corners)


def test_migration_creates_shots_table_and_indexes():
    conn = db.connect()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(shots)")]
    assert tuple(cols) == ss.COLS
    pk = [r[1] for r in sorted(conn.execute("PRAGMA table_info(shots)"), key=lambda r: r[5])
          if r[5]]
    assert pk == ["game_id", "action_number"]
    idx = {r[1]: [c[2] for c in conn.execute(f"PRAGMA index_info({r[1]})")]
           for r in conn.execute("PRAGMA index_list(shots)") if not r[1].startswith("sqlite_")}
    assert idx == {"idx_shots_player_season": ["player_id", "season"],
                   "idx_shots_team_season": ["team", "season"]}
    assert db.migrate(conn) == len(db.MIGRATIONS)         # re-run is a no-op


def _archive_game(gid, day, pbp):
    archive.save_raw("nba_cdn", "playbyplay", day, gid, pbp)


def test_backfill_is_idempotent_and_resumable():
    real = archive.load_raw(FIX / "analytics/playbyplay_0022500001.json.gz")
    _archive_game(GID, "2025-10-21", real)
    _archive_game("0022600005", "2026-10-22", _pbp([_fga(4, 0, 5)]))
    _archive_game("0022600006", "2026-10-22", _pbp([], "0022600006"))     # no field goals
    conn = db.connect()
    s = ss.backfill(conn, limit=1)                           # killed after one game
    assert s["stored_games"] == 1
    s = ss.backfill(conn)
    assert s["skipped"] == 1 and s["stored_games"] == 1 and s["empty"] == ["0022600006"]
    n = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert n == 104 + 97 + 1
    s = ss.backfill(conn)
    assert s["stored_games"] == 0 and s["skipped"] == 2
    s = ss.backfill(conn, force=True)                        # re-parse: same rows, no dupes
    assert s["stored_games"] == 2
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == n
    assert ss.backfill(conn, "2026-27", force=True)["archived"] == 2
    assert tuple(conn.execute("SELECT game_date, season FROM shots WHERE game_id='0022600005'"
                              ).fetchone()) == ("2026-10-22", "2026-27")


def test_store_game_replaces_rows():
    _archive_game("0022600005", "2026-10-22", _pbp([_fga(4, 0, 5), _fga(6, 0, 300)]))
    conn = db.connect()
    assert ss.store_game(conn, "0022600005", "2026-10-22") == 2
    assert ss.store_game(conn, "0022600005", "2026-10-22", _pbp([_fga(4, 0, 5)])) == 1
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1
