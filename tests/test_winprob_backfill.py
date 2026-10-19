"""PBP backfill (fake fetch, tmp archive, no network) and training-sample building."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from courtside import archive, backfill_pbp, db
from courtside.analytics import winprob as wp
from courtside.http import FetchError

FIX = Path(__file__).parent / "fixtures" / "analytics"


def _conn(tmp_path, games):
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
         ("0011900001", "2019-20", "Pre Season", "2019-10-01"),
         ("0022000001", "2020-21", "Regular Season", "2020-12-22")]


def test_games_for_filters_types_and_seasons(tmp_path):
    conn = _conn(tmp_path, GAMES)
    got = backfill_pbp.games_for(conn, ["2019-20"])
    assert [g[0] for g in got] == ["0021900001", "0021900002", "0021900003", "0041900101"]
    assert got[0] == ("0021900001", "2019-10-22", "2019-20")


def test_backfill_archives_skips_and_resumes(tmp_path):
    conn = _conn(tmp_path, GAMES)
    calls = []

    def fetch(gid):
        calls.append(gid)
        if gid == "0021900002":
            raise FetchError("403", status=403)
        return {"game": {"actions": [{"actionNumber": 1}]}}

    s = backfill_pbp.backfill_pbp(["2019-20", "2020-21"], conn=conn, fetch=fetch)
    assert (s["fetched"], s["missing"], s["failed"], s["cached"]) == (4, 1, 0, 0)
    assert not s["aborted"]
    p = archive.raw_path("nba_cdn", "playbyplay", "2019-10-22", "0021900001")
    assert p.exists() and archive.load_raw(p)["game"]["actions"]
    assert archive.raw_path("nba_cdn", "playbyplay", "2020-08-18", "0041900101").exists()
    assert set(backfill_pbp.archived_index()) == {"0021900001", "0021900003", "0041900101", "0022000001"}

    calls.clear()                                   # resume: archived games are never refetched
    s2 = backfill_pbp.backfill_pbp(["2019-20", "2020-21"], conn=conn, fetch=fetch)
    assert calls == ["0021900002"] and s2["cached"] == 4 and s2["missing"] == 1
    assert s2["by_season"]["2019-20"]["games"] == 4


def test_archived_under_another_day_counts_as_archived(tmp_path):
    conn = _conn(tmp_path, GAMES[:1])
    archive.save_raw("nba_cdn", "playbyplay", "2019-10-21", "0021900001", {"game": {"actions": [1]}})
    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn, fetch=lambda g: pytest.fail("refetched"))
    assert s["cached"] == 1 and s["fetched"] == 0


def test_backfill_survives_errors_and_aborts_on_an_outage(tmp_path, monkeypatch):
    many = [(f"00219{n:05d}", "2019-20", "Regular Season", "2019-10-22") for n in range(1, 40)]
    conn = _conn(tmp_path, many)

    def boom(gid):
        raise FetchError("HTTP 500", status=500)

    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn, fetch=boom)
    assert s["aborted"] and s["failed"] == backfill_pbp.MAX_CONSECUTIVE_FAILS
    assert s["fetched"] == 0 and not archive.raw_path(
        "nba_cdn", "playbyplay", "2019-10-22", many[0][0]).exists()

    # holes in the cdn are not an outage
    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn,
                                  fetch=lambda g: (_ for _ in ()).throw(FetchError("x", status=403)))
    assert not s["aborted"] and s["missing"] == len(many)


def test_empty_payload_is_missing_not_archived(tmp_path):
    conn = _conn(tmp_path, GAMES[:1])
    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn, fetch=lambda g: {"game": {"actions": []}})
    assert s["missing"] == 1 and not backfill_pbp.archived_index()


def test_fetcher_tells_missing_from_blocked(monkeypatch):
    log = []

    def get(url, **kw):
        log.append((url[-15:-5], kw))
        raise FetchError("403", status=403)

    # sentinel answers: the id is simply missing, no waiting
    def get_missing(url, **kw):
        if "0021900001" in url:
            return {"game": {"actions": [1]}}
        raise FetchError("403", status=403)

    waits, resets = [], []
    fetch = backfill_pbp.make_fetcher("0021900001", sleep=waits.append, get=get_missing,
                                      reset=lambda: resets.append(1))
    with pytest.raises(FetchError):
        fetch("0021900777")
    assert waits == []

    # sentinel is blocked too: wait with growing backoff, retry the game, then give up
    resets.clear()
    fetch = backfill_pbp.make_fetcher("0021900001", sleep=waits.append, get=get,
                                      reset=lambda: resets.append(1))
    with pytest.raises(backfill_pbp.Blocked):
        fetch("0021900777")
    assert waits == list(backfill_pbp.BLOCK_WAITS) and resets == [1]
    assert all(kw == {"retries": 0} for _, kw in log)

    # block lifts after one wait: the game is fetched
    state = {"n": 0}

    def get_recovering(url, **kw):
        state["n"] += 1
        if state["n"] <= 2:                     # game 403, sentinel 403
            raise FetchError("403", status=403)
        return {"game": {"actions": [1]}}

    waits.clear()
    resets.clear()
    fetch = backfill_pbp.make_fetcher("0021900001", sleep=waits.append, get=get_recovering,
                                      reset=lambda: resets.append(1))
    # a fresh session fixes it immediately, no waiting
    assert fetch("0021900777")["game"]["actions"] == [1] and waits == [] and resets == [1]


def test_blocked_stops_the_run_without_marking_games_missing(tmp_path):
    conn = _conn(tmp_path, GAMES[:3])

    def blocked(gid):
        raise backfill_pbp.Blocked("still blocked")

    s = backfill_pbp.backfill_pbp(["2019-20"], conn=conn, fetch=blocked)
    assert s["aborted"] and s["missing"] == 0 and s["fetched"] == 0


def test_training_sample_rows_from_a_fixture():
    with gzip.open(FIX / "playbyplay_0022501066.json.gz", "rt") as f:
        pbp = json.load(f)
    with gzip.open(FIX / "boxscore_0022501066.json.gz", "rt") as f:
        box = json.load(f)
    g = box["game"]
    rows = wp._sample_rows(pbp, g["homeTeam"]["teamTricode"], g["awayTeam"]["teamTricode"], 3.5)
    assert 100 < len(rows) < 600
    margin, t, exp, poss, period, elapsed, won = zip(*rows)
    assert set(won) == {0} and set(exp) == {3.5}          # IND (home) lost 113-114
    assert max(t) == 2880.0 and min(t) >= 0 and set(poss) <= {-1, 0, 1}
    assert max(margin) >= 20                               # IND led by 24
    assert wp._sample_rows(pbp, "XXX", "YYY", 0.0) is None
