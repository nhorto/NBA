"""mlops.alerts (rules, dedupe, escalation, resolution) and mlops.drift (PSI, min-n)."""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pytest
from mlops_helpers import conn_for, pred

from courtside.mlops import alerts, drift, monitor

AS_OF = date(2026, 2, 1)


def put_snapshot(conn, model="gbm_market", window="last300", as_of=AS_OF, n=300, version=None,
                 **kw):
    det = {"ll_se": 0.027, "home_bias_se": 0.02, "margin_bias_se": 0.7, "n_margin": n,
           "total_bias_se": 1.0, "n_total": n}
    det.update(kw.pop("details", {}))
    row = dict(log_loss=0.60, brier=0.21, accuracy=0.66, ece=0.03, home_pred=0.58,
               home_actual=0.58, margin_bias=0.0, margin_mae=10.0, total_bias=0.0, total_mae=14.0,
               mkt_n=n, mkt_log_loss=0.60, mkt_brier=0.21, ll_diff_vs_mkt=0.0, ll_diff_se=0.004)
    row.update(kw)
    conn.execute(
        "INSERT OR REPLACE INTO metric_snapshots (as_of, model, model_version, window, n, "
        "log_loss, brier, accuracy, ece, home_pred, home_actual, margin_bias, margin_mae, "
        "total_bias, total_mae, mkt_n, mkt_log_loss, mkt_brier, ll_diff_vs_mkt, ll_diff_se, "
        "details_json) VALUES (:as_of,:model,:v,:window,:n,:log_loss,:brier,:accuracy,:ece,"
        ":home_pred,:home_actual,:margin_bias,:margin_mae,:total_bias,:total_mae,:mkt_n,"
        ":mkt_log_loss,:mkt_brier,:ll_diff_vs_mkt,:ll_diff_se,:d)",
        {**row, "as_of": as_of.isoformat(), "model": model, "v": version, "window": window,
         "n": n, "d": json.dumps(det)})
    conn.commit()


def put_check(conn, name, status, run_at, detail="x"):
    conn.execute("INSERT OR REPLACE INTO data_checks (run_at, check_name, status, detail) "
                 "VALUES (?,?,?,?)", (run_at, name, status, detail))
    conn.commit()


def keys(rows, open_only=True):
    return {a["key"] for a in rows if not open_only or a["status"] == "open"}


def test_quiet_when_everything_is_fine(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn)
    assert alerts.evaluate(conn, AS_OF) == []


def test_total_bias_message_and_min_n(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, total_bias=2.1, details={"total_bias_se": 0.9})       # 2.33 SE
    out = alerts.evaluate(conn, AS_OF)
    a = [x for x in out if x["key"] == "bias:gbm_market:total"][0]
    assert "Over-predicting totals by 2.1 pts" in a["message"]
    assert "last 300 games" in a["message"]
    # below the bias n floor nothing fires
    conn.execute("DELETE FROM metric_snapshots")
    put_snapshot(conn, n=99, total_bias=5.0)
    assert [x for x in alerts.evaluate(conn, AS_OF) if x["status"] == "open"] == []


def test_bias_needs_two_se(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, total_bias=1.9, margin_bias=-1.0, home_pred=0.60, home_actual=0.57)
    assert alerts.evaluate(conn, AS_OF) == []         # 1.9/1.0, 1.0/0.7, 0.03/0.02: all < 2 SE
    put_snapshot(conn, margin_bias=-1.6, home_pred=0.60, home_actual=0.55)
    k = keys(alerts.evaluate(conn, AS_OF))
    assert k == {"bias:gbm_market:margin", "bias:gbm_market:home"}
    msgs = {a["key"]: a["message"] for a in alerts.list_alerts(conn)}
    assert msgs["bias:gbm_market:margin"].startswith("Under-predicting home margins by 1.6 pts")
    assert "gives them 60.0% to win" in msgs["bias:gbm_market:home"]


def test_performance_vs_market(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, ll_diff_vs_mkt=0.012, ll_diff_se=0.004)            # 3 SE worse
    assert keys(alerts.evaluate(conn, AS_OF)) == {"perf:gbm_market"}
    put_snapshot(conn, ll_diff_vs_mkt=0.006, ll_diff_se=0.004)            # 1.5 SE: clears
    out = alerts.evaluate(conn, AS_OF)
    assert [(a["key"], a["event"]) for a in out] == [("perf:gbm_market", "resolved")]
    # market-free models are not held to the market
    put_snapshot(conn, model="gbm_opinion", ll_diff_vs_mkt=0.02, ll_diff_se=0.004)
    assert alerts.evaluate(conn, AS_OF) == []
    # n floor
    put_snapshot(conn, n=99, mkt_n=99, ll_diff_vs_mkt=0.5, ll_diff_se=0.004)
    assert alerts.evaluate(conn, AS_OF) == []


def test_calibration_needs_300(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, ece=0.08, n=250)
    assert alerts.evaluate(conn, AS_OF) == []
    put_snapshot(conn, ece=0.08, n=300)
    assert keys(alerts.evaluate(conn, AS_OF)) == {"calib:gbm_market"}


def test_expected_band_uses_docs_constant_then_registry(tmp_path):
    conn = conn_for(tmp_path)
    # docs constant 0.5790; tolerance 2*sqrt(0.027^2+0.015^2) = 0.0618 -> fires above 0.6408
    put_snapshot(conn, log_loss=0.635)
    assert alerts.evaluate(conn, AS_OF) == []
    put_snapshot(conn, log_loss=0.645)
    assert keys(alerts.evaluate(conn, AS_OF)) == {"band:gbm_market"}
    # a registered benchmark replaces the constant
    conn.execute("INSERT INTO model_versions (version_id, model_name, spec_hash, status, "
                 "created_at, benchmark_json) VALUES ('v1','gbm_market','h','champion','t',?)",
                 (json.dumps({"walk_forward": {"log_loss": 0.62}}),))
    conn.commit()
    out = alerts.evaluate(conn, AS_OF)
    assert [(a["key"], a["event"]) for a in out] == [("band:gbm_market", "resolved")]


def test_dedupe_update_and_resolve(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, ece=0.08)
    first = alerts.evaluate(conn, AS_OF)
    assert first[0]["event"] == "opened"
    put_snapshot(conn, ece=0.09)
    second = alerts.evaluate(conn, AS_OF)
    assert second[0]["event"] == "updated" and "0.090" in second[0]["message"]
    rows = conn.execute("SELECT * FROM alerts").fetchall()
    assert len(rows) == 1 and rows[0]["status"] == "open"
    assert rows[0]["created_at"] <= rows[0]["updated_at"]
    third = alerts.evaluate(conn, AS_OF)
    assert third[0]["event"] in ("open", "updated")
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
    put_snapshot(conn, ece=0.02)
    out = alerts.evaluate(conn, AS_OF)
    assert out[0]["event"] == "resolved"
    row = conn.execute("SELECT * FROM alerts").fetchone()
    assert row["status"] == "resolved" and row["resolved_at"]
    # firing again opens a NEW row (the old one stays as history)
    put_snapshot(conn, ece=0.08)
    alerts.evaluate(conn, AS_OF)
    assert [r["status"] for r in conn.execute("SELECT status FROM alerts ORDER BY id")] == \
        ["resolved", "open"]


def test_data_check_escalation_and_resolution(tmp_path):
    conn = conn_for(tmp_path)
    name = "pregame.odds_coverage"
    put_check(conn, name, "ok", "2026-02-01T10:00:00Z")
    put_check(conn, name, "warn", "2026-02-01T13:30:00Z", "odds missing for 2 of 10 games")
    assert alerts.evaluate(conn, AS_OF) == []                         # one warning: quiet
    put_check(conn, name, "skip", "2026-02-01T15:00:00Z")             # skips are ignored
    put_check(conn, name, "warn", "2026-02-01T17:00:00Z", "odds missing for 2 of 10 games")
    a = alerts.evaluate(conn, AS_OF)[0]
    assert (a["key"], a["severity"]) == (f"data:{name}", "warn")      # 2 in a row
    assert "2 runs in a row" in a["message"] and "odds missing" in a["message"]
    put_check(conn, name, "fail", "2026-02-01T18:30:00Z", "odds missing for 7 of 10 games")
    a = alerts.evaluate(conn, AS_OF)[0]
    assert a["severity"] == "critical" and a["event"] == "updated"    # 3 in a row
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
    put_check(conn, name, "ok", "2026-02-01T20:00:00Z")
    assert alerts.evaluate(conn, AS_OF)[0]["event"] == "resolved"


def test_single_fail_warns_immediately(tmp_path):
    conn = conn_for(tmp_path)
    put_check(conn, "postgame.finals_archived", "fail", "2026-02-01T10:00:00Z", "2 missing")
    a = alerts.evaluate(conn, AS_OF)[0]
    assert a["severity"] == "warn" and a["kind"] == "data"


def test_drift_alert_levels(tmp_path):
    conn = conn_for(tmp_path)
    conn.executemany("INSERT INTO drift_snapshots (as_of, feature, psi, ref_label, cur_n) "
                     "VALUES ('2026-02-01',?,?, '2024-25 preds', 150)",
                     [("market_total", 0.30), ("elo_diff", 0.60), ("rest_diff", 0.10)])
    conn.commit()
    sev = {a["key"]: a["severity"] for a in alerts.evaluate(conn, AS_OF)}
    assert sev == {"drift:market_total": "warn", "drift:elo_diff": "critical"}
    # a stale (> 7 day old) drift snapshot is ignored and its alerts resolve
    out = alerts.evaluate(conn, date(2026, 2, 20))
    assert {a["event"] for a in out} == {"resolved"}


# ------------------------------------------------------------------ drift

def test_psi_two_bin_hand_value():
    ref = [0] * 50 + [1] * 50
    cur = [0] * 80 + [1] * 20
    expect = 0.3 * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert drift.psi(ref, cur, bins=2) == pytest.approx(expect)


def test_psi_identical_is_zero_and_empty_bins_are_floored():
    ref = np.arange(100)
    assert drift.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    cur = [0] * 50 + [99] * 50              # only the lowest and highest decile occupied
    expect = 2 * (0.5 - 0.1) * math.log(5) + 8 * (1e-4 - 0.1) * math.log(1e-4 / 0.1)
    assert drift.psi(ref, cur) == pytest.approx(expect, rel=1e-6)


def test_psi_constant_reference_and_empty():
    assert drift.psi([3] * 20, [3] * 20) == pytest.approx(0.0)
    assert drift.psi([3] * 20, [5] * 20) > 1.0
    assert drift.psi([], [1, 2]) is None


def _inputs(spread, total=220.0, elo=(1600, 1550), rest_diff=0, b2b=(0, 0)):
    return {"schedule": {"rest_diff": rest_diff, "home": {"b2b": b2b[0]}, "away": {"b2b": b2b[1]}},
            "elo": {"home": elo[0], "away": elo[1]},
            "market": {"home_spread": spread, "total": total},
            "explain": [{"factor": "player_strength", "home_margin_pts": 1.5},
                        {"factor": "absences", "home_margin_pts": -0.5},
                        {"factor": "absences", "home_margin_pts": -0.25}],
            "availability": {"home": {"lineup_strength": 1.0}, "away": {"lineup_strength": 0.4}}}


def _fill(conn, start, n, spread_fn):
    for i in range(n):
        d = date.fromordinal(start.toordinal() + i // 5).isoformat()
        pred(conn, f"{d}-{i}", d, 0.5 + (i % 7) / 20, total=215 + i % 9,
             inputs=_inputs(spread_fn(i)), created=f"{d}T12:00:00Z", mkt_spread=spread_fn(i))


def test_extract_features():
    f = drift.extract({"inputs_json": json.dumps(_inputs(-4.5, 221, rest_diff=1, b2b=(1, 0))),
                       "home_win_prob": 0.6, "total": 219.0, "market_home_spread": None,
                       "market_total": None})
    assert f["elo_diff"] == 50 and f["strength_diff"] == 1.5 and f["absences"] == -0.75
    assert f["rest_diff"] == 1 and f["b2b_total"] == 1
    assert f["market_spread"] == -4.5 and f["market_total"] == 221
    assert f["pred_home_prob"] == 0.6 and f["pred_total"] == 219
    assert f["lineup_strength_diff"] == pytest.approx(0.6)


def test_drift_min_n_suppression(tmp_path):
    conn = conn_for(tmp_path)
    _fill(conn, date(2025, 1, 10), 150, lambda i: -3.0 + (i % 10) / 2)       # last season
    _fill(conn, date(2026, 1, 2), 60, lambda i: 5.0)                         # only 60 live
    assert drift.compute_drift(conn, AS_OF) == []
    assert conn.execute("SELECT COUNT(*) FROM drift_snapshots").fetchone()[0] == 0
    assert len(drift.compute_drift(conn, AS_OF, min_n=50)) > 0


def test_drift_detects_shift_vs_previous_season(tmp_path):
    conn = conn_for(tmp_path)
    _fill(conn, date(2025, 1, 10), 150, lambda i: -3.0 + (i % 10) / 2)
    _fill(conn, date(2026, 1, 2), 150, lambda i: 6.0 + (i % 10) / 2)         # spreads flipped
    rows = {r["feature"]: r for r in drift.compute_drift(conn, AS_OF)}
    assert rows["market_spread"]["psi"] > 0.5
    assert rows["market_spread"]["ref_label"] == "2024-25 preds"
    assert rows["market_spread"]["cur_n"] == 150
    assert rows["pred_home_prob"]["psi"] < 0.05                              # same distribution
    stored = conn.execute("SELECT feature, psi FROM drift_snapshots WHERE as_of='2026-02-01'"
                          ).fetchall()
    assert {r["feature"] for r in stored} == set(rows)


def test_drift_falls_back_to_earlier_in_season(tmp_path):
    conn = conn_for(tmp_path)
    _fill(conn, date(2025, 10, 25), 150, lambda i: -3.0 + (i % 10) / 2)      # same season, early
    _fill(conn, date(2026, 1, 2), 150, lambda i: -3.0 + (i % 10) / 2)
    rows = drift.compute_drift(conn, AS_OF)
    assert rows and rows[0]["ref_label"] == "2025-26 before window"
    assert all(r["psi"] < 0.1 for r in rows)


def test_run_after_job_is_guarded_and_summarises(tmp_path):
    conn = conn_for(tmp_path)
    out = monitor.run_after_job(conn, "postgame", date(2026, 2, 1), {"ok": True, "steps": {}})
    assert out["line"].startswith("health[postgame 2026-02-01]")
    assert out["alerts"]["open"] == 0
    conn.execute("DROP TABLE alerts")            # a broken stage must not raise
    out = monitor.run_after_job(conn, "pregame", date(2026, 2, 1), None)
    assert "alerts" in out["errors"]


def test_stale_metric_snapshots_raise_no_alerts(tmp_path):
    conn = conn_for(tmp_path)
    put_snapshot(conn, ece=0.08)
    assert keys(alerts.evaluate(conn, AS_OF)) == {"calib:gbm_market"}
    out = alerts.evaluate(conn, date(2026, 3, 1))           # snapshot is 28 days old
    assert [(a["key"], a["event"]) for a in out] == [("calib:gbm_market", "resolved")]
