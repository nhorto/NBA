"""Boxscore backfill tests: fake fetch, tmp archive, no network."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from courtside import archive, backfill_pbp, db
from courtside.http import FetchError

FIX = Path(__file__).parent / "fixtures" / "analytics"


def _conn(tmp_path, games):
    """Create a test DB with team_games."""
    conn = db.connect(tmp_path / "t.db")
    for gid, season, stype, day in games:
        for team, home in (("AAA", 1), ("BBB", 0)):
            conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, "
                         "is_home) VALUES (?,?,?,?,?,?)", (gid, team, season, stype, day, home))
    conn.commit()
    return conn


GAMES = [("0021900001", "2019-20", "Regular Season", "2019-10-22"),
         ("0021900002", "2019-20", "Regular Season", "2019-10-23"),
         ("0021900003", "2019-20", "Regular Season", "2019-10-23"),
         ("0041900101", "2019-20", "Playoffs", "2020-08-18"),
         ("0022000001", "2020-21", "Regular Season", "2020-12-22")]


def _valid_boxscore(game_id: str) -> dict:
    """Return a valid boxscore payload for the given game_id."""
    return {
        "game": {
            "gameId": game_id,
            "homeTeam": {"players": [{"personId": 1, "firstName": "Home", "lastName": "Player"}]},
            "awayTeam": {"players": [{"personId": 2, "firstName": "Away", "lastName": "Player"}]},
        }
    }


def test_backfill_pbp_default_unchanged(tmp_path):
    """Ensure playbyplay backfill still works with default kind."""
    conn = _conn(tmp_path, GAMES[:2])
    calls = []

    def fetch(gid):
        calls.append(gid)
        return {"game": {"actions": [{"actionNumber": 1}]}}

    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn, fetch=fetch)
    assert s["fetched"] == 2
    assert archive.raw_path("nba_cdn", "playbyplay", "2019-10-22", "0021900001").exists()
    assert set(calls) == {"0021900001", "0021900002"}


def test_backfill_boxscore_skips_archived(tmp_path):
    """Boxscore backfill skips games already archived."""
    # First, archive some playbyplay games
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-23", "0021900002", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "boxscore", "2019-10-22", "0021900001", _valid_boxscore("0021900001"))

    calls = []

    def fetch(gid):
        calls.append(gid)
        return _valid_boxscore(gid)

    s = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch)
    assert s["cached"] == 1  # 0021900001 is cached
    assert s["fetched"] == 1  # 0021900002 is fetched
    assert set(calls) == {"0021900002"}


def test_backfill_boxscore_archives_new(tmp_path):
    """Boxscore backfill archives new boxscores under the correct path."""
    # Archive playbyplay so we know what games to backfill
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-23", "0021900002", {"game": {"actions": [1]}})

    calls = []

    def fetch(gid):
        calls.append(gid)
        return _valid_boxscore(gid)

    s = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch)
    assert s["fetched"] == 2
    assert not s["aborted"]

    # Check files are archived in the right place
    p1 = archive.raw_path("nba_cdn", "boxscore", "2019-10-22", "0021900001")
    p2 = archive.raw_path("nba_cdn", "boxscore", "2019-10-23", "0021900002")
    assert p1.exists() and p2.exists()
    assert archive.load_raw(p1)["game"]["homeTeam"]["players"]


def test_backfill_boxscore_rejects_mismatched_game_id(tmp_path):
    """Boxscore with mismatched game_id is rejected (counted as missing)."""
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})

    def fetch(gid):
        # Return a boxscore with wrong game_id
        return _valid_boxscore("0021900999")

    s = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch)
    assert s["missing"] == 1
    assert s["fetched"] == 0
    assert not archive.raw_path("nba_cdn", "boxscore", "2019-10-22", "0021900001").exists()


def test_backfill_boxscore_rejects_missing_teams(tmp_path):
    """Boxscore without both teams' players is rejected (counted as missing)."""
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})

    def fetch(gid):
        # Return a boxscore without away team
        return {
            "game": {
                "gameId": gid,
                "homeTeam": {"players": [{"personId": 1}]},
                "awayTeam": {},
            }
        }

    s = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch)
    assert s["missing"] == 1
    assert s["fetched"] == 0


def test_backfill_boxscore_resumes(tmp_path):
    """Boxscore backfill resumes where it left off."""
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-23", "0021900002", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-23", "0021900003", {"game": {"actions": [1]}})

    # First run: fetch and cache some
    calls1 = []

    def fetch1(gid):
        calls1.append(gid)
        if gid == "0021900002":
            raise FetchError("error", status=500)
        return _valid_boxscore(gid)

    s1 = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch1)
    assert s1["fetched"] == 2
    assert s1["failed"] == 1
    assert set(calls1) == {"0021900001", "0021900002", "0021900003"}

    # Second run: only refetch the failed one
    calls2 = []

    def fetch2(gid):
        calls2.append(gid)
        return _valid_boxscore(gid)

    s2 = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch2)
    assert s2["cached"] == 2
    assert s2["fetched"] == 1
    assert calls2 == ["0021900002"]


def test_backfill_boxscore_handles_missing_games(tmp_path):
    """Boxscore for missing games (403) is counted as missing, not failed."""
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-22", "0021900001", {"game": {"actions": [1]}})
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-23", "0021900002", {"game": {"actions": [1]}})

    def fetch(gid):
        if gid == "0021900001":
            return _valid_boxscore(gid)
        raise FetchError("404", status=404)

    s = backfill_pbp.backfill_pbp(["2019-20"], kind="boxscore", fetch=fetch)
    assert s["fetched"] == 1
    assert s["missing"] == 1
    assert s["failed"] == 0
    assert not s["aborted"]
