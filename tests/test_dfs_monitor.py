"""T3.1: DFS registry specs and alerts on synthetic snapshots (no network)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from courtside import db
from courtside.mlops import alerts, dfs_monitor, registry

AS_OF = date(2026, 11, 5)
NOW = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)
D0 = date(2026, 10, 21)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(dfs_monitor, "_now", lambda: NOW)
    monkeypatch.delenv("COURTSIDE_LIVE_LIKE_MIN_MAE", raising=False)
    # no T2.4 baseline unless a test sets one (neither the data dir nor the shipped copy)
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(dfs_monitor, "PACKAGE_BASELINE", tmp_path / "no_baseline.json")
    return db.connect(tmp_path / "t.db")


def add_run(conn, run_id, slate, kind, created, version="pathA-v0.1", ok=1):
    conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
                 "VALUES (?,?,?,?,?,?)", (run_id, slate, kind, created, version, ok))
    conn.commit()


def add_rows(conn, n, *, proj_min=30.0, actual_min=30.0, in80=lambda k: 1 if k % 10 < 8 else 0,
             start=0, per_game=10,
             proj_dk=30.0):
    """n graded projection rows (one per game x player), days D0.. in blocks of per_game."""
    for k in range(start, start + n):
        day = (D0 + timedelta(days=k // 40)).isoformat()
        game = f"G{k // per_game}"
        cur = conn.execute(
            "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, run_id, "
            "created_at, model_version, p_play) VALUES (?,?,?,?,?,?,?,?)",
            (game, day, "9001", 1000 + k, "run1", day + "T20:00:00Z", "pathA-v0.1", 0.95))
        pid = cur.lastrowid
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?, 'min', 'played', ?)", (pid, proj_min))
        in80_v = in80(k) if callable(in80) else in80
        conn.execute(
            "INSERT INTO player_projection_grades (projection_id, game_id, player_id, run_id, "
            "graded_at, played, actual_min, actual_dk, proj_min, proj_dk, in80, in50) "
            "VALUES (?,?,?,?,?,1,?,?,?,?,?,?)",
            (pid, game, 1000 + k, "run1", day + "T23:00:00Z",
             actual_min(k) if callable(actual_min) else actual_min, 30.0, proj_min * 0.95, proj_dk,
             in80_v, in80_v))
    conn.commit()


def keys(conn, status="open"):
    return sorted(r[0] for r in conn.execute("SELECT key FROM alerts WHERE status=?", (status,)))


def evaluate(conn):
    return alerts.evaluate(conn, AS_OF)


# ------------------------------------------------------------------ registry

def test_registry_specs_follow_run_versions(conn):
    assert dfs_monitor.sync_registry(conn) == {}                     # no runs: nothing
    add_run(conn, "r0", "9001", "lock10", "2026-10-20T20:00:00Z", version="fppg-fallback")
    add_run(conn, "r1", "9001", "lock45", "2026-10-20T21:00:00Z", version="pathA-v0.1")
    add_run(conn, "rg", "rg:2021-01-01", "manual", "2026-10-20T21:30:00Z", version="hist-v9")
    add_run(conn, "rx", "9001", "lock10", "2026-10-20T21:40:00Z", version="pathA-v0.1", ok=0)
    res = dfs_monitor.sync_registry(conn)
    assert set(res) == set(dfs_monitor.COMPONENTS)
    for comp, r in res.items():
        v = registry.get_champion(conn, comp)
        assert v["version_id"] == r["champion"] and v["version_id"].startswith(f"{comp}@v1-")
        assert v["params"]["model_version"] == "pathA-v0.1"
    eng = registry.get_champion(conn, "dfs_engine")["params"]
    assert eng["cash_lambda"] == 0.3 and eng["p_play_min"] == 0.85
    again = dfs_monitor.sync_registry(conn)                          # idempotent
    assert again["player_minutes"]["champion"] == res["player_minutes"]["champion"]
    assert len(registry.list_versions(conn, "player_minutes")) == 1
    # a new pipeline version: new champion, the old one retired, events logged
    add_run(conn, "r2", "9001", "lock45", "2026-10-25T21:00:00Z", version="pathA-v0.2")
    dfs_monitor.sync_registry(conn)
    vs = registry.list_versions(conn, "player_dist")
    assert sorted(v["status"] for v in vs) == ["champion", "retired"]
    assert registry.get_champion(conn, "player_dist")["version_id"].startswith("player_dist@v2-")
    acts = [e["action"] for e in registry.events(conn, model_name="player_dist")]
    assert "promote" in acts and "retire" in acts
    # the game models' registry view is untouched
    assert registry.production_versions(conn)["champions"] == {}


# ------------------------------------------------------------------ minutes bias

def test_min_bias_alert_fires_dedupes_and_resolves(conn):
    add_rows(conn, 120, proj_min=30.0, actual_min=27.0)               # +3 min over-projection
    ev = evaluate(conn)
    assert keys(conn) == ["dfs:min_bias"]
    a = [e for e in ev if e["key"] == "dfs:min_bias"][0]
    assert a["event"] == "opened" and a["kind"] == "dfs" and "over-projecting" in a["message"]
    assert "3.0 min" in a["message"]
    ev2 = evaluate(conn)                                              # same state: one row, no new event
    assert keys(conn) == ["dfs:min_bias"]
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
    assert [e["event"] for e in ev2 if e["key"] == "dfs:min_bias"] == ["open"]
    # more rows that are fine pull the last-300 mean back inside +-1.0
    add_rows(conn, 300, proj_min=30.0, actual_min=30.2, start=120)
    ev3 = evaluate(conn)
    assert keys(conn) == [] and keys(conn, "resolved") == ["dfs:min_bias"]
    assert [e["event"] for e in ev3 if e["key"] == "dfs:min_bias"] == ["resolved"]


def test_min_bias_needs_enough_starters_and_ignores_bench(conn):
    add_rows(conn, 60, proj_min=30.0, actual_min=25.0)                # only 60 starter rows
    add_rows(conn, 200, proj_min=18.0, actual_min=12.0, start=60)     # bench rows do not count
    evaluate(conn)
    assert keys(conn) == []


def test_under_projection_is_worded_as_such(conn):
    add_rows(conn, 150, proj_min=30.0, actual_min=32.0)
    a = [e for e in evaluate(conn) if e["key"] == "dfs:min_bias"][0]
    assert "under-projecting" in a["message"]


# ------------------------------------------------------------------ 80% coverage

def test_cov80_alert(conn):
    add_rows(conn, 300, in80=lambda k: 1 if k % 10 < 6 else 0, actual_min=30.0)   # 60% coverage
    ev = evaluate(conn)
    a = [e for e in ev if e["key"] == "dfs:cov80"][0]
    assert "covered 60.0%" in a["message"] and "too narrow" in a["message"]
    assert a["details"]["n"] == 300
    # healthy coverage: no alert, and the open one resolves
    add_rows(conn, 300, in80=lambda k: 1 if k % 10 < 8 else 0, start=300, actual_min=30.0)
    evaluate(conn)
    assert "dfs:cov80" not in keys(conn)


def test_cov80_quiet_within_three_se_and_with_few_rows(conn):
    add_rows(conn, 300, in80=lambda k: 1 if k % 50 < 37 else 0, actual_min=30.0)  # 74%: < 3 SE
    evaluate(conn)
    assert "dfs:cov80" not in keys(conn)
    add_rows(conn, 150, in80=0, actual_min=30.0, start=300)           # very bad but only 150 rows
    conn.execute("DELETE FROM player_projection_grades WHERE projection_id <= 300")
    conn.commit()
    evaluate(conn)
    assert "dfs:cov80" not in keys(conn)


# ------------------------------------------------------------------ live minutes MAE

def test_live_minutes_mae_needs_baseline(conn, monkeypatch):
    add_rows(conn, 200, proj_min=30.0, actual_min=lambda k: 30.0 + (6 if k % 2 else -6))
    evaluate(conn)
    assert "dfs:live_min_mae" not in keys(conn)                       # T2.4 has not set it yet
    monkeypatch.setenv("COURTSIDE_LIVE_LIKE_MIN_MAE", "4.7")
    a = [e for e in evaluate(conn) if e["key"] == "dfs:live_min_mae"][0]
    assert "6.00 min" in a["message"] and "4.70" in a["message"]
    monkeypatch.setenv("COURTSIDE_LIVE_LIKE_MIN_MAE", "5.6")          # within +0.5 of 6.0 -> quiet
    evaluate(conn)
    assert "dfs:live_min_mae" not in keys(conn)


def test_live_baseline_from_file(data_dir, monkeypatch):
    monkeypatch.delenv("COURTSIDE_LIVE_LIKE_MIN_MAE", raising=False)
    monkeypatch.setattr(dfs_monitor, "PACKAGE_BASELINE", data_dir / "none.json")
    assert dfs_monitor.live_like_baseline() is None
    (data_dir / "proj").mkdir(parents=True)
    (data_dir / "proj" / "live_like_baseline.json").write_text('{"min_mae": 4.65}')
    assert dfs_monitor.live_like_baseline() == 4.65


# ------------------------------------------------------------------ missed lock run

def slate(conn, sid, lock, tag="main", day="2026-11-05"):
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, lock_utc) "
                 "VALUES (?,?,70,?,?)", (sid, day, tag, lock))
    conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id, "
                 "draftable_id, player_name, site_player_id, team, positions, roster_slot, salary, "
                 "status, disabled) VALUES ('2026-11-05T12:00:00Z','dk',?,'classic',70,'d','A B',"
                 "'1','AAA','PG','UTIL',5000,'None',0)", (sid,))
    conn.commit()


def test_missed_lock_run_critical_then_partial_then_ok(conn):
    slate(conn, "9001", "2026-11-05T14:30:00Z")                       # locked 30 min before NOW
    slate(conn, "9002", "2026-11-05T14:50:00Z", tag="late")           # inside the 15 min grace
    slate(conn, "9003", "2026-11-05T23:00:00Z", tag="early")          # not locked yet
    ev = evaluate(conn)
    assert keys(conn) == ["dfs:missed_lock:9001"]
    a = [e for e in ev if e["key"] == "dfs:missed_lock:9001"][0]
    assert a["severity"] == "critical" and "no lock-45 or lock-10 run" in a["message"]
    assert "9:30 AM ET" in a["message"]
    add_run(conn, "r1", "9001", "lock45", "2026-11-05T13:45:00Z")
    a = [e for e in evaluate(conn) if e["key"] == "dfs:missed_lock:9001"][0]
    assert a["severity"] == "warn" and a["event"] == "updated" and "lock-10" in a["message"]
    add_run(conn, "r2", "9001", "lock10", "2026-11-05T14:20:00Z")
    ev = evaluate(conn)
    assert keys(conn) == [] and [e["event"] for e in ev] == ["resolved"]


def test_missed_lock_counts_failed_attempts_and_skips_old_or_uncaptured(conn):
    slate(conn, "9001", "2026-11-05T14:30:00Z")
    add_run(conn, "r1", "9001", "lock10", "2026-11-05T14:21:00Z", ok=0)
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, tag, lock_utc) VALUES "
                 "('9100','2026-11-05','main','2026-11-05T14:00:00Z')")    # never captured
    slate(conn, "9200", "2026-10-25T14:00:00Z", day="2026-10-25")          # a week old
    conn.commit()
    a = [e for e in evaluate(conn) if e["key"].startswith("dfs:missed_lock")]
    assert [e["key"] for e in a] == ["dfs:missed_lock:9001"]
    assert "1 failed attempt" in a[0]["message"]


# ------------------------------------------------------------------ reconciliation, fallback, data checks

def check(conn, name, status, value=None, detail="", run_at="2026-11-05T09:00:00Z"):
    conn.execute("INSERT INTO data_checks (run_at, check_name, status, value, threshold, detail) "
                 "VALUES (?,?,?,?,?,?)", (run_at, name, status, value, 6.0, detail))
    conn.commit()


def test_reconciliation_gap_is_one_alert_not_one_per_run(conn):
    det = json.dumps({"max_abs_resid": 0.8, "teams": [{"team": "BOS", "gap": 7.2, "target": 112.0,
                                                       "scale": 1.05}]})
    check(conn, "proj_recon_gap:runA", "warn", 7.2, det)
    check(conn, "proj_recon_gap:runB", "warn", 9.1, det, run_at="2026-11-05T10:00:00Z")
    check(conn, "proj_recon_gap:runC", "ok", 1.0, "{}", run_at="2026-11-05T11:00:00Z")
    evaluate(conn)
    assert keys(conn) == ["dfs:recon_gap"]                            # not data:proj_recon_gap:*
    msg = conn.execute("SELECT message FROM alerts").fetchone()[0]
    assert "2 recent projection runs" in msg and "9.1 points" in msg and "BOS" in msg
    # old checks age out of the lookback
    assert alerts.evaluate(conn, date(2026, 11, 20)) and keys(conn) == []


def test_fppg_and_id_match_failures_use_the_data_check_alerts(conn):
    check(conn, "dfs_scoring_fppg", "fail", 0.6, "412/700 within 0.1 FP of DK FPPG")
    check(conn, "ids_dk", "fail", 0.97, "likely rotation players unmatched: A B")
    evaluate(conn)
    assert keys(conn) == ["data:dfs_scoring_fppg", "data:ids_dk"]
    msg = {r[0]: r[1] for r in conn.execute("SELECT key, message FROM alerts")}
    assert "412/700" in msg["data:dfs_scoring_fppg"]
    check(conn, "dfs_scoring_fppg", "ok", 0.99, "ok", run_at="2026-11-05T12:00:00Z")
    evaluate(conn)
    assert keys(conn) == ["data:ids_dk"]


def test_fallback_runs_alert(conn):
    add_run(conn, "r1", "9001", "lock45", "2026-11-04T20:00:00Z", version="fppg-fallback")
    add_run(conn, "r2", "9001", "lock10", "2026-11-04T22:00:00Z", version="pathA-v0.1")
    a = [e for e in evaluate(conn) if e["key"] == "dfs:fallback"][0]
    assert "1 of 2" in a["message"]


# ------------------------------------------------------------------ robustness

def test_a_failing_rule_keeps_its_open_alerts(conn, monkeypatch):
    add_rows(conn, 300, in80=lambda k: 1 if k % 10 < 6 else 0, actual_min=30.0)
    evaluate(conn)
    assert "dfs:cov80" in keys(conn)

    def boom(*a, **k):
        raise RuntimeError("rule broke")
    rules = tuple((p, n, boom if p == "dfs:cov80" else f) for p, n, f in dfs_monitor.RULES)
    monkeypatch.setattr(dfs_monitor, "RULES", rules)
    evaluate(conn)
    assert "dfs:cov80" in keys(conn) and keys(conn, "resolved") == []    # not spuriously resolved


def test_alerts_work_on_an_empty_dfs_database(conn):
    assert dfs_monitor.desired_alerts(conn, AS_OF, now=NOW) == []
    assert alerts.evaluate(conn, AS_OF) == []


def test_postgame_hook_syncs_registry(conn):
    from courtside.mlops import monitor
    add_run(conn, "r1", "9001", "lock45", "2026-11-04T20:00:00Z")
    out = monitor.run_after_job(conn, "postgame", date(2026, 11, 4))
    assert set(out["dfs"]["registry"]) == set(dfs_monitor.COMPONENTS)
    assert "errors" not in out
