"""T2.4 live-like baselines in the DFS monitor: the shipped baseline file, the data-dir override,
and the nightly range rule (dfs:live_range:<metric>, info) with its small-night widening."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from courtside import db
from courtside.mlops import alerts, dfs_monitor

AS_OF = date(2026, 10, 21)
NOW = datetime(2026, 10, 21, 7, 0, tzinfo=timezone.utc)
NIGHT = "2026-10-20"
SHIPPED = Path(dfs_monitor.__file__).with_name("live_like_baseline.json")

DAILY = {"dk_mae": {"p10": 7.0, "p50": 7.6, "p90": 8.2, "rows_median": 100},
         "cov80": {"p10": 0.74, "p50": 0.80, "p90": 0.86, "rows_median": 100},
         "min_mae_played15": {"p10": 4.2, "p50": 4.8, "p90": 5.4, "rows_median": 50},
         "pplay_cal_err": {"p10": -0.08, "p50": 0.0, "p90": 0.08, "rows_median": 20}}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(dfs_monitor, "_now", lambda: NOW)
    monkeypatch.delenv("COURTSIDE_LIVE_LIKE_MIN_MAE", raising=False)
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path / "data"))
    base = tmp_path / "pkg_baseline.json"
    base.write_text(json.dumps({"min_mae": 4.8, "model": "pathA-v0.2", "daily": DAILY}))
    monkeypatch.setattr(dfs_monitor, "PACKAGE_BASELINE", base)
    return db.connect(tmp_path / "t.db")


def night(conn, n, *, err=7.6, in80=lambda k: k % 10 < 8, min_err=4.8, p_avail=None,
          p_play=0.5, played=lambda k: 1, day=NIGHT, version="pathA-v0.2"):
    """n graded rows of one night: |proj - actual| DK = err, minutes error = min_err."""
    for k in range(n):
        game = f"{day}-G{k // 12}"
        inputs = json.dumps({"p_avail": p_avail(k) if callable(p_avail) else p_avail})
        cur = conn.execute(
            "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, run_id, "
            "created_at, model_version, p_play, inputs_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (game, day, "9001", 5000 + k, "run1", day + "T22:00:00Z", version, p_play, inputs))
        pid = cur.lastrowid
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?, 'min', 'played', ?)", (pid, 30.0))
        pl = played(k)
        conn.execute(
            "INSERT INTO player_projection_grades (projection_id, game_id, player_id, run_id, "
            "graded_at, played, actual_min, actual_dk, proj_min, proj_dk, in80, in50) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, game, 5000 + k, "run1", day + "T23:59:00Z", pl, (30.0 - min_err) if pl else 0,
             30.0, 28.0, 30.0 + (err if k % 2 else -err), int(in80(k)), 1))
    conn.commit()


def live_keys(conn):
    return sorted(a["key"] for a in dfs_monitor.desired_alerts(conn, AS_OF)
                  if a["key"].startswith("dfs:live_range"))


def test_shipped_baseline_file_is_valid_and_complete():
    v = json.loads(SHIPPED.read_text())
    assert 3.0 < v["min_mae"] < 7.0
    for m in dfs_monitor.RANGE_METRICS:
        r = v["daily"][m]
        assert r["p10"] <= r["p50"] <= r["p90"] and r["rows_median"] > 0
    assert {"pathA-v0.2", "pathA-v0.3", "pathA-v0.4"} <= set(v["per_model"])
    assert v["model"] == "pathA-v0.2" and v["protocol"] == "live"


def test_data_dir_file_overrides_the_shipped_copy(conn, tmp_path):
    assert dfs_monitor.live_like_baseline() == 4.8
    (tmp_path / "data" / "proj").mkdir(parents=True)
    (tmp_path / "data" / "proj" / "live_like_baseline.json").write_text('{"min_mae": 5.1}')
    assert dfs_monitor.live_like_baseline() == 5.1
    assert dfs_monitor.live_like_daily() is None                    # that file has no ranges


def test_quiet_inside_the_range(conn):
    night(conn, 100)
    assert live_keys(conn) == []


def test_high_dk_error_is_info_and_low_error_is_not_flagged(conn):
    night(conn, 100, err=9.0)
    a = [x for x in dfs_monitor.desired_alerts(conn, AS_OF) if x["key"] == "dfs:live_range:dk_mae"]
    assert len(a) == 1 and a[0]["severity"] == "info" and "above" in a[0]["message"]
    assert a[0]["details"]["k"] == pytest.approx(1.0)


def test_small_nights_widen_the_band_never_narrow_it(conn):
    night(conn, 25, err=8.6)                     # 25 rows vs 100 per slice night: x2 band
    assert live_keys(conn) == []
    lo, hi, k = dfs_monitor.range_band(DAILY["dk_mae"], 25)
    assert k == pytest.approx(2.0) and hi == pytest.approx(7.6 + 0.6 * 2)
    assert dfs_monitor.range_band(DAILY["dk_mae"], 400)[2] == 1.0


def test_low_coverage_and_pplay_both_sides(conn):
    night(conn, 100, in80=lambda k: k % 10 < 6, p_avail=lambda k: 0.5 if k < 30 else 1.0,
          p_play=0.5, played=lambda k: 1)
    ks = live_keys(conn)
    assert "dfs:live_range:cov80" in ks                            # 60% < 74%
    assert "dfs:live_range:pplay_cal_err" in ks                     # 0.5 - 1.0 = -0.5
    a = [x for x in dfs_monitor.desired_alerts(conn, AS_OF)
         if x["key"] == "dfs:live_range:pplay_cal_err"][0]
    assert a["details"]["n"] == 30 and a["details"]["value"] == pytest.approx(-0.5)


def test_minutes_range_and_stale_nights(conn):
    night(conn, 100, min_err=7.0, day="2026-10-17")                # 4 days before AS_OF
    assert live_keys(conn) == []
    night(conn, 100, min_err=7.0)
    assert live_keys(conn) == ["dfs:live_range:min_mae_played15"]


def test_per_model_ranges_are_used_for_their_version(conn, tmp_path):
    other = {m: dict(r) for m, r in DAILY.items()}
    other["dk_mae"] = {"p10": 9.0, "p50": 9.5, "p90": 10.0, "rows_median": 100}
    (tmp_path / "data" / "proj").mkdir(parents=True)
    (tmp_path / "data" / "proj" / "live_like_baseline.json").write_text(json.dumps(
        {"min_mae": 4.8, "daily": DAILY, "per_model": {"pathA-v0.9": {"daily": other}}}))
    night(conn, 100, err=9.0, version="pathA-v0.9")
    assert live_keys(conn) == []                                   # inside v0.9's own range
    assert dfs_monitor.live_like_daily("pathA-v0.2") == DAILY


def test_range_alerts_go_through_the_alerts_table_and_resolve(conn):
    night(conn, 100, err=9.0)
    alerts.evaluate(conn, AS_OF)
    rows = conn.execute("SELECT key, severity FROM alerts WHERE status='open' AND key LIKE "
                        "'dfs:live_range%'").fetchall()
    assert [tuple(r) for r in rows] == [("dfs:live_range:dk_mae", "info")]
    later = date(2026, 10, 25)                                      # night is now stale
    alerts.evaluate(conn, later)
    assert conn.execute("SELECT COUNT(*) FROM alerts WHERE status='open' AND key LIKE "
                        "'dfs:live_range%'").fetchone()[0] == 0


def test_no_baseline_no_range_rule(conn, monkeypatch, tmp_path):
    monkeypatch.setattr(dfs_monitor, "PACKAGE_BASELINE", tmp_path / "missing.json")
    night(conn, 100, err=20.0)
    assert live_keys(conn) == []
