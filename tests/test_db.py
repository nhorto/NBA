from courtside import db


def test_connect_creates_db_and_migrates(data_dir):
    conn = db.connect()
    assert (data_dir / "courtside.db").exists()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"games", "odds_snapshots", "injury_snapshots", "collect_runs", "schema_version"} <= tables
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_migrate_idempotent():
    conn = db.connect()
    assert db.migrate(conn) == len(db.MIGRATIONS)
    db.migrate(conn)
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert [r[0] for r in rows] == list(range(1, len(db.MIGRATIONS) + 1))


def test_upsert_game_partial_update():
    conn = db.connect()
    db.upsert_game(conn, game_id="0022600001", nba_day="2026-10-20", home="BOS", away="NYK",
                   tipoff_utc="2026-10-20T23:30:00Z")
    db.upsert_game(conn, game_id="0022600001", espn_id="401", status="final",
                   home_score=110, away_score=99, tipoff_utc=None)
    g = conn.execute("SELECT * FROM games").fetchone()
    assert g["tipoff_utc"] == "2026-10-20T23:30:00Z"  # not clobbered by None
    assert (g["espn_id"], g["status"], g["home_score"], g["home"]) == ("401", "final", 110, "BOS")
    assert g["updated_at"]
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1


def test_odds_injuries_runs():
    conn = db.connect()
    oid = db.insert_odds_snapshot(conn, game_id="g", captured_at="2026-10-20T14:00:00Z",
                                  source="espn", home_spread=-3.5, total=221.5)
    assert oid == 1
    n = db.insert_injury_snapshots(conn, [
        {"captured_at": "t", "source": "espn", "team": "BOS", "player_name": "A", "status": "Out"},
        {"captured_at": "t", "source": "espn", "team": "BOS", "player_name": "B"},
    ])
    assert n == 2
    rid = db.start_run(conn, "pregame")
    db.finish_run(conn, rid, True, "ok")
    r = conn.execute("SELECT * FROM collect_runs WHERE id=?", (rid,)).fetchone()
    assert r["ok"] == 1 and r["detail"] == "ok" and r["finished_at"]
