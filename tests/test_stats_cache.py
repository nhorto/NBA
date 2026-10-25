"""stats.db domain-versioned cache (box / dfs), stale-while-revalidate and the stats/warm.py
background warmer. Tiny fixture DBs built in the test."""

from __future__ import annotations

import sqlite3
import sys
import threading
import time

import pytest

from courtside.stats import db as sdb
from courtside.stats import warm as W


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    sdb.clear()
    sdb.set_refresher(None)
    monkeypatch.setattr(sdb, "PROBE_INTERVAL", 0.0)
    yield
    W.stop()
    sdb.set_refresher(None)
    sdb.clear()


def _db(tmp_path):
    p = tmp_path / "courtside.db"
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
        CREATE TABLE player_games (game_id TEXT, player_id INTEGER, pts INTEGER,
                                   PRIMARY KEY (game_id, player_id));
        CREATE TABLE player_games_adv (game_id TEXT, player_id INTEGER, usg_pct REAL);
        CREATE TABLE dfs_runs (run_id TEXT PRIMARY KEY, created_at TEXT);
        CREATE TABLE player_projections (id INTEGER PRIMARY KEY, player_id INTEGER);
        CREATE TABLE odds_snapshots (game_id TEXT, total REAL);
        INSERT INTO player_games VALUES ('g1', 1, 10), ('g1', 2, 20);
        INSERT INTO dfs_runs VALUES ('r1', '2026-10-24');
    """)
    c.commit()
    return p, c


def _pts(c):
    return c.execute("SELECT SUM(pts) FROM player_games").fetchone()[0]


# ------------------------------------------------------------------ domain versions
def test_domain_tables_patterns():
    have = ["player_games", "player_games_adv", "team_games_adv", "dfs_runs", "games",
            "player_projections", "player_projection_stats", "odds_snapshots"]
    assert sdb.domain_tables("box", have) == ["player_games", "player_games_adv",
                                              "team_games_adv"]
    assert sdb.domain_tables("dfs", have) == ["dfs_runs", "player_projection_stats",
                                              "player_projections"]


def test_domain_version_moves_only_with_its_tables(tmp_path):
    p, c = _db(tmp_path)
    v0 = sdb.domain_versions(p, force=True)
    c.execute("INSERT INTO dfs_runs VALUES ('r2', '2026-10-25')")
    c.execute("INSERT INTO odds_snapshots VALUES ('g1', 220.5)")    # in no domain
    c.commit()
    v1 = sdb.domain_versions(p, force=True)
    assert v1["box"] == v0["box"]
    assert v1["dfs"] != v0["dfs"]
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 30)")
    c.commit()
    v2 = sdb.domain_versions(p, force=True)
    assert v2["box"] != v1["box"] and v2["dfs"] == v1["dfs"]
    # INSERT OR REPLACE re-inserts at a new rowid: a corrected box score moves the version
    c.execute("INSERT OR REPLACE INTO player_games VALUES ('g1', 1, 11)")
    c.commit()
    assert sdb.domain_versions(p, force=True)["box"] != v2["box"]
    # deleting the oldest rows (pruning) moves MIN(rowid)
    v3 = sdb.domain_versions(p, force=True)
    c.execute("DELETE FROM player_games WHERE rowid = (SELECT MIN(rowid) FROM player_games)")
    c.commit()
    assert sdb.domain_versions(p, force=True)["box"] != v3["box"]


def test_probe_is_throttled(tmp_path, monkeypatch):
    p, c = _db(tmp_path)
    monkeypatch.setattr(sdb, "PROBE_INTERVAL", 3600.0)
    v0 = sdb.domain_versions(p)
    assert sdb.domain_versions(p) is v0                      # unchanged file: no probe
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 30)")
    c.commit()
    assert sdb.domain_versions(p) is v0                      # changed, but inside the interval
    assert sdb.domain_versions(p, force=True)["box"] != v0["box"]


def test_missing_db_has_no_version(tmp_path):
    assert sdb.domain_versions(tmp_path / "nope.db") is None


# ------------------------------------------------------------------ cached(domain=...)
def test_box_entry_survives_dfs_writes_but_not_box_writes(tmp_path):
    p, c = _db(tmp_path)
    calls = []

    def fn(conn):
        calls.append(1)
        return _pts(conn)

    assert sdb.cached(("box-sum",), fn, p, domain="box") == 30
    assert sdb.cached(("file-sum",), fn, p) == 30
    c.execute("INSERT INTO dfs_runs VALUES ('r2', '2026-10-25')")
    c.commit()
    assert sdb.cached(("box-sum",), fn, p, domain="box") == 30 and len(calls) == 2  # kept
    assert sdb.cached(("file-sum",), fn, p) == 30 and len(calls) == 3              # dropped
    ro = sdb.connect_ro(p)
    assert sdb.cached(("box-sum",), fn, ro, domain="box") == 30 and len(calls) == 3
    ro.close()
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
    c.commit()
    assert sdb.cached(("box-sum",), fn, p, domain="box") == 35 and len(calls) == 4  # no SWR
    # a tuple of domains follows both
    assert sdb.cached(("both",), fn, p, domain=("dfs", "box")) == 35 and len(calls) == 5
    c.execute("INSERT INTO player_projections (player_id) VALUES (1)")
    c.commit()
    assert sdb.cached(("both",), fn, p, domain=("box", "dfs")) == 35 and len(calls) == 6
    with pytest.raises(ValueError):
        sdb.cached(("x",), fn, p, domain="odds")


def test_in_memory_and_missing_with_domain(tmp_path):
    mem = sqlite3.connect(":memory:")
    assert sdb.cached(("k",), lambda c: 7, mem, domain="box") == 7
    assert sdb.cached(("k",), lambda c: 1, tmp_path / "missing.db", domain="box") is None


# ------------------------------------------------------------------ stale-while-revalidate
def test_stale_while_revalidate_serves_old_value_then_new(tmp_path):
    p, c = _db(tmp_path)
    queued = []
    sdb.set_refresher(lambda bucket, key, compute: queued.append((key, compute)))
    calls = []

    def fn(conn):
        calls.append(1)
        return _pts(conn)

    assert sdb.cached(("sum",), fn, p, domain="box") == 30       # cold miss: computed inline
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
    c.commit()
    assert sdb.cached(("sum",), fn, p, domain="box") == 30       # stale value, at once
    assert len(calls) == 1 and [k for k, _ in queued] == [("sum",)]
    queued[0][1]()                                                # the background refresh
    assert len(calls) == 2
    assert sdb.cached(("sum",), fn, p, domain="box") == 35 and len(calls) == 2
    # the file flavour never serves stale values
    assert sdb.cached(("fsum",), fn, p) == 35
    c.execute("INSERT INTO player_games VALUES ('g3', 1, 1)")
    c.commit()
    assert sdb.cached(("fsum",), fn, p) == 36


def test_value_built_from_stale_input_is_refreshed_again(tmp_path):
    p, c = _db(tmp_path)
    queued = []
    sdb.set_refresher(lambda bucket, key, compute: queued.append((key, compute)))
    inner = lambda conn: sdb.cached(("inner",), _pts, conn, domain="box")  # noqa: E731
    assert inner(p) == 30
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
    c.commit()
    # a new outer key computed now sees the stale inner value ...
    assert sdb.cached(("outer",), lambda conn: inner(conn) + 1, p, domain="box") == 31
    # ... so it is stored as stale: the next read serves it and queues its refresh
    assert sdb.cached(("outer",), lambda conn: inner(conn) + 1, p, domain="box") == 31
    assert [k for k, _ in queued] == [("inner",), ("outer",)]
    for _, compute in queued:
        compute()
    assert sdb.cached(("outer",), lambda conn: inner(conn) + 1, p, domain="box") == 36


def test_failed_refresh_drops_the_entry(tmp_path):
    p, c = _db(tmp_path)
    queued = []
    sdb.set_refresher(lambda bucket, key, compute: queued.append(compute))
    state = {"fail": False}

    def fn(conn):
        if state["fail"]:
            raise RuntimeError("boom")
        return _pts(conn)

    assert sdb.cached(("k",), fn, p, domain="box") == 30
    c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
    c.commit()
    state["fail"] = True
    assert sdb.cached(("k",), fn, p, domain="box") == 30
    with pytest.raises(RuntimeError):
        queued[0]()
    state["fail"] = False
    assert sdb.cached(("k",), fn, p, domain="box") == 35          # recomputed in the request


# ------------------------------------------------------------------ the warmer
def _wait(cond, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_warmer_precomputes_and_rewarms_on_domain_change(tmp_path):
    p, c = _db(tmp_path)
    calls = []

    def build(conn):
        calls.append(1)
        return _pts(conn)

    def job(conn):
        return sdb.cached(("lg",), build, conn, domain="box")

    extra_ran = []
    switch = sys.getswitchinterval()
    w = W.Warmer(p, extra=[("templates", lambda: extra_ran.append(1))], poll=0.05,
                 job_list=lambda conn, day: [("league", job)]).start()
    try:
        assert w.ready.wait(10)
        assert calls == [1] and extra_ran == [1]
        assert sdb.cached(("lg",), build, p, domain="box") == 30     # warm hit
        c.execute("INSERT INTO dfs_runs VALUES ('r2', '2026-10-25')")       # other domain
        c.commit()
        assert _wait(lambda: len(w.cycles) >= 2)
        assert calls == [1]                                    # box entry still fresh
        c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
        c.commit()
        # until the warmer is done the old value is served; then the new one
        assert sdb.cached(("lg",), build, p, domain="box") in (30, 35)
        assert _wait(lambda: len(calls) == 2)                  # warmer or queued refresh
        assert _wait(lambda: sdb.cached(("lg",), build, p, domain="box") == 35)
        assert extra_ran == [1]                                # one-time jobs run once
        assert _wait(lambda: any("box" in cy["reason"] for cy in w.cycles))
        assert len(calls) == 2                                 # computed once, not twice
    finally:
        w.stop()
    assert not w.thread.is_alive()
    assert sdb._refresher is None
    assert sys.getswitchinterval() == switch                  # restored after each cycle


def test_warmer_runs_requests_stale_refreshes(tmp_path):
    p, c = _db(tmp_path)
    w = W.Warmer(p, poll=3600, job_list=lambda conn, day: []).start()
    try:
        assert w.ready.wait(10)
        fn = lambda cc: _pts(cc)  # noqa: E731
        assert sdb.cached(("pp",), fn, p, domain="box") == 30
        c.execute("INSERT INTO player_games VALUES ('g2', 1, 5)")
        c.commit()
        assert sdb.cached(("pp",), fn, p, domain="box") == 30     # stale, refresh queued
        assert _wait(lambda: w.refreshed == 1)
        assert sdb.cached(("pp",), fn, p, domain="box") == 35
    finally:
        w.stop()


def test_warmer_can_be_disabled(tmp_path, monkeypatch):
    p, _ = _db(tmp_path)
    monkeypatch.setenv("COURTSIDE_WARM", "0")
    assert not W.enabled()
    assert W.start(p) is None
    monkeypatch.setenv("COURTSIDE_WARM", "1")
    assert W.enabled()
    assert W.start(tmp_path / "missing.db") is None


def _warm_threads():
    return [t for t in threading.enumerate() if t.name == "courtside-stats-warm"]


def test_app_startup_hook_respects_the_switch(tmp_path, data_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from courtside.web.app import create_app
    data_dir.mkdir(parents=True, exist_ok=True)
    _db(data_dir)
    monkeypatch.setattr(W, "jobs", lambda conn, day: [])
    monkeypatch.setenv("COURTSIDE_WARM", "0")                  # the test default
    with TestClient(create_app(data_dir)) as client:
        assert client.get("/healthz").status_code == 200
        assert _warm_threads() == []
    monkeypatch.setenv("COURTSIDE_WARM", "1")
    with TestClient(create_app(data_dir)):
        assert len(_warm_threads()) == 1
        w = W._warmers[str((data_dir / "courtside.db").resolve())]
        assert w.ready.wait(10) and w.cycles[0]["reason"] == "start"
        assert "templates" in w.cycles[0]["jobs"]
    assert _wait(lambda: _warm_threads() == [])                # stopped on shutdown


def test_pinned_entries_survive_the_lru(tmp_path, monkeypatch):
    p, _ = _db(tmp_path)
    monkeypatch.setattr(sdb, "MAX_DOMAIN_ENTRIES", 3)
    with sdb.pinning():
        assert sdb.cached(("shared",), _pts, p, domain="box") == 30
    for i in range(6):
        sdb.cached(("player", i), lambda c, i=i: i, p, domain="box")
    calls = []
    assert sdb.cached(("shared",), lambda c: calls.append(1), p, domain="box") == 30
    assert calls == []                                          # never evicted
    (b,) = [b for b in sdb.stats()["buckets"] if b["domain"] == ("box",)]
    assert b["entries"] == 3 and b["pinned"] == 1
    sdb.unpin_all()
    for i in range(6, 9):
        sdb.cached(("player", i), lambda c, i=i: i, p, domain="box")
    assert sdb.cached(("shared",), lambda c: calls.append(1) or 0, p, domain="box") == 0
