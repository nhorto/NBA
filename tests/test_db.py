import pytest

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


def test_migration_2_on_top_of_1_and_idempotent(data_dir):
    import sqlite3
    path = data_dir / "v1.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    conn.executescript("BEGIN;\n" + db.MIGRATIONS[0] + "\nCOMMIT;")
    conn.execute("INSERT INTO schema_version VALUES (1, 'x')")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version")]
    assert versions == list(range(1, len(db.MIGRATIONS) + 1)) and len(versions) >= 2
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"team_games", "player_games", "team_games_adv", "player_games_adv"} <= tables
    idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_team_games_team_date", "idx_player_games_player_date"} <= idx
    db.migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(versions)


def test_upsert_rows():
    conn = db.connect()
    row = {"game_id": "g1", "team": "BOS", "season": "2025-26", "pts": 100}
    assert db.upsert_rows(conn, "team_games", [row]) == 1
    db.upsert_rows(conn, "team_games", [{**row, "pts": 110}])
    rows = conn.execute("SELECT pts, opp FROM team_games").fetchall()
    assert len(rows) == 1 and rows[0]["pts"] == 110 and rows[0]["opp"] is None
    assert db.upsert_rows(conn, "team_games", []) == 0
    with pytest.raises(TypeError):
        db.upsert_rows(conn, "team_games", [{"game_id": "g", "team": "X", "bogus": 1}])
    with pytest.raises(ValueError):
        db.upsert_rows(conn, "nope", [{"a": 1}])


def test_migration_4_tables_and_latest_pretip_view():
    conn = db.connect()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"game_facts", "predictions", "briefs"} <= tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    assert {"game_id", "model", "created_at", "home_win_prob", "home_margin", "total",
            "market_home_spread", "market_total", "market_home_prob", "inputs_json"} <= cols
    assert {r[1] for r in conn.execute("PRAGMA table_info(game_facts)")} >= {
        "game_id", "nba_day", "facts_json", "computed_at"}
    assert {r[1] for r in conn.execute("PRAGMA table_info(briefs)")} >= {
        "nba_day", "brief_json", "built_at", "version"}

    db.upsert_game(conn, game_id="G1", nba_day="2026-10-20", home="BOS", away="NYK",
                   tipoff_utc="2026-10-20T23:00:00Z")
    ins = ("INSERT INTO predictions (game_id, model, created_at, home_win_prob, nba_day) "
           "VALUES (?,?,?,?,?)")
    for created, p in (("2026-10-20T12:00:00Z", 0.60), ("2026-10-20T18:00:00Z", 0.65),
                       ("2026-10-20T23:30:00Z", 0.99)):     # last one is after tip-off
        conn.execute(ins, ("G1", "m", created, p, "2026-10-20"))
    conn.execute(ins, ("G1", "other", "2026-10-20T12:00:00Z", 0.40, "2026-10-20"))
    # a game with no `games` row (past demo day): newest row counts
    conn.execute(ins, ("G2", "m", "2025-12-01T12:00:00Z", 0.30, "2025-12-01"))
    conn.execute(ins, ("G2", "m", "2025-12-01T15:00:00Z", 0.35, "2025-12-01"))
    conn.commit()
    got = {(r["game_id"], r["model"]): r["home_win_prob"] for r in conn.execute(
        "SELECT * FROM latest_pretip_predictions")}
    assert got == {("G1", "m"): 0.65, ("G1", "other"): 0.40, ("G2", "m"): 0.35}
    # history is append-only: nothing was collapsed
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 6


def test_earlier_migrations_unchanged_and_ordered():
    conn = db.connect()
    assert [r[0] for r in conn.execute("SELECT version FROM schema_version")] == [1, 2, 3, 4]
