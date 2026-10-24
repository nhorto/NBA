"""IMPROVEMENTS.md E6: MLOps gating and monitoring fixes, all on synthetic data.

1 online gate "not clearly worse" (date-clustered SE)   2 offline gate: clustered SE, regular season
3 monthly / alert-triggered retrain   4 calibration alert vs the market, persistent
5 data fingerprint on the registry   6 PSI suppression for seasonal features early in a season.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlops_helpers import pred
from test_mlops_monitor_alerts import AS_OF, _inputs, put_snapshot

from courtside import db
from courtside.mlops import alerts, drift, monitor, registry, retrain


@pytest.fixture(autouse=True)
def no_mlflow(monkeypatch):
    monkeypatch.setenv("COURTSIDE_MLFLOW", "0")


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "e6.db")


# ------------------------------------------------------------------ 1. online gate

@pytest.mark.parametrize("n, diff, se, crit, action", [
    (99, 0.5, 0.001, 0, "wait"),          # the n floor stays
    (100, 0.0004, 0.001, 0, "promote"),   # worse, but inside 2 SE: not clearly worse
    (100, 0.0021, 0.001, 0, "reject"),    # worse by more than 2 SE
    (100, -0.01, 0.001, 0, "promote"),
    (150, 0.0004, 0.001, 1, "wait"),      # not clearly worse but a critical alert is open
    (150, 0.0100, 0.001, 1, "reject"),    # clearly worse is rejected even with an alert open
])
def test_shadow_decision_not_clearly_worse(n, diff, se, crit, action):
    assert retrain.shadow_decision(n, diff, crit, se)[0] == action


def test_shadow_decision_without_se_keeps_the_old_rule():
    assert retrain.shadow_decision(100, 0.0, 0)[0] == "promote"
    assert retrain.shadow_decision(299, 0.002, 0)[0] == "wait"
    assert retrain.shadow_decision(300, 0.002, 0)[0] == "reject"
    assert retrain.shadow_decision(120, 0.002, 0, float("nan"))[0] == "wait"


def _shadow_world(conn, n_games: int, jitter: float, seed=0):
    """Champion p = truth; shadow p = truth + noise(jitter). 4 games a night."""
    registry.bootstrap(conn, log_mlflow=False, log_fn=lambda s: None)
    p = registry.production_params("gbm_opinion")
    p["fixed"]["margin"][1] = 7
    sid = registry.register(conn, "gbm_opinion", p, registry.feature_list("gbm_opinion"),
                            status="shadow")
    champ = registry.get_champion(conn, "gbm_opinion")["version_id"]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_games):
        gid = f"00224{i:05d}"
        day = date.fromordinal(date(2024, 11, 1).toordinal() + i // 4).isoformat()
        truth = rng.uniform(0.25, 0.75)
        y = rng.random() < truth
        conn.execute("INSERT INTO games (game_id, nba_day, status, home_score, away_score, "
                     "tipoff_utc) VALUES (?,?,?,?,?,?)",
                     (gid, day, "final", 110 if y else 100, 100 if y else 110, f"{day}T23:00:00Z"))
        ps = float(np.clip(truth + rng.normal(0, jitter), 0.05, 0.95))
        for model, ver, role, pr in (("gbm_opinion", champ, "champion", truth),
                                     (sid, sid, "shadow", ps)):
            rows.append((gid, model, f"{day}T12:00:00Z", pr, "{}", day, "BOS", "NYK", ver, role))
    conn.executemany("INSERT INTO predictions (game_id, model, created_at, home_win_prob, "
                     "inputs_json, nba_day, home, away, model_version, role) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return sid, champ


def test_evaluate_shadows_promotes_a_slightly_worse_shadow(conn):
    sid, champ = _shadow_world(conn, 160, jitter=0.02, seed=2)
    (r,) = retrain.evaluate_shadows(conn, log_fn=lambda s: None)
    # the shadow is (noisily) worse on average yet within 2 date-clustered SE
    assert r["n"] == 160 and 0 < r["diff"] <= 2 * r["se"]
    assert r["action"] == "promote" and r["applied"]
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == sid


def test_evaluate_shadows_rejects_a_clearly_worse_shadow_at_n_160(tmp_path):
    conn = db.connect(tmp_path / "bad.db")
    sid, champ = _shadow_world(conn, 160, jitter=0.25)
    (r,) = retrain.evaluate_shadows(conn, log_fn=lambda s: None)
    assert r["diff"] > 2 * r["se"] and r["action"] == "reject" and r["applied"]
    assert registry.get_version(conn, sid)["status"] == "retired"
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == champ


# ------------------------------------------------------------------ 2. offline gate

def _gate_frames(n_days=300, per_day=8, night_sd=0.0, playoff_bad=False, seed=1):
    """Champion = noisy quote; challenger = the same quote with a 1.5x stretched logit. Outcomes
    carry a per-night effect (favorites over/under-perform on a whole night), so the loss
    difference of games on one night is correlated."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = date.fromordinal(date(2023, 10, 25).toordinal() + d).isoformat()
        ne = rng.normal(0, night_sd)
        for k in range(per_day):
            truth = rng.uniform(0.3, 0.7)
            champ = float(np.clip(truth + rng.normal(0, 0.05), 0.02, 0.98))
            y = int(rng.random() < np.clip(truth + np.sign(truth - 0.5) * ne, 0.02, 0.98))
            chal = float(1 / (1 + np.exp(-np.log(champ / (1 - champ)) * 1.5)))
            rows.append((f"g{d}-{k}", "2023-24", "Regular Season", day, y, champ, chal))
    if playoff_bad:
        for k in range(400):
            truth = rng.uniform(0.3, 0.7)
            y = int(rng.random() < truth)
            rows.append((f"p{k}", "2023-24", "Playoffs", "2024-05-01", y, truth, 1 - truth))
    df = pd.DataFrame(rows, columns=["game_id", "season", "season_type", "game_date", "y",
                                     "p_champ", "p_chal"])
    base = ["game_id", "season", "season_type", "game_date", "y"]
    return df[base].assign(p=df["p_champ"]), df[base].assign(p=df["p_chal"])


def test_offline_gate_uses_a_date_clustered_se():
    from courtside.proj.metrics import paired_cluster
    a, b = _gate_frames(night_sd=0.3)
    g = retrain.offline_gate(a, b)
    d = retrain.ll_vec(a["y"], b["p"]) - retrain.ll_vec(a["y"], a["p"])
    naive = float(d.std(ddof=1) / np.sqrt(len(d)))
    assert g["overall"]["n_clusters"] == 300
    assert g["overall"]["se"] > 1.5 * naive              # night-level effects inflate the SE
    pc = paired_cluster(d, a["game_date"].to_numpy())
    assert g["overall"]["se"] == pytest.approx(pc["se"])
    assert g["overall"]["diff"] == pytest.approx(pc["mean"])


def test_offline_gate_excludes_playoffs():
    a, b = _gate_frames(playoff_bad=True)
    g = retrain.offline_gate(a, b)
    assert g["n"] == 300 * 8                              # the 400 playoff games are not counted
    a2, b2 = _gate_frames(playoff_bad=False)
    assert retrain.offline_gate(a2, b2)["overall"]["diff"] == pytest.approx(g["overall"]["diff"])


def test_offline_gate_without_dates_is_the_per_game_se():
    a, b = _gate_frames()
    cols = ["game_id", "season", "y", "p"]
    g = retrain.offline_gate(a[cols], b[cols])
    d = retrain.ll_vec(a["y"], b["p"]) - retrain.ll_vec(a["y"], a["p"])
    assert g["overall"]["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(len(d)))


# ------------------------------------------------------------------ 3. retrain cadence

def _alert(conn, key, kind, status="open"):
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message, "
                 "status) VALUES (?, 't', 't', 'warn', ?, 'm', ?)", (key, kind, status))
    conn.commit()


def test_retrain_due_monthly_or_on_alert(conn):
    assert retrain.retrain_due(conn, date(2026, 11, 2))[0]            # first Monday of the month
    assert retrain.retrain_due(conn, date(2026, 11, 7))[0]
    due, why = retrain.retrain_due(conn, date(2026, 11, 9))           # second Monday: skipped
    assert not due and "no calibration/drift alert" in why
    _alert(conn, "perf:gbm_market", "performance")                    # not a trigger kind
    _alert(conn, "calib:gbm_opinion", "calibration", status="resolved")
    assert not retrain.retrain_due(conn, date(2026, 11, 9))[0]
    _alert(conn, "calib:gbm_market", "calibration")
    due, why = retrain.retrain_due(conn, date(2026, 11, 9))
    assert due and "calib:gbm_market" in why
    conn.execute("DELETE FROM alerts")
    _alert(conn, "drift:market_total", "drift")
    assert retrain.retrain_due(conn, date(2026, 11, 16))[0]


def test_weekly_job_calls_retrain_if_due_and_keeps_the_other_steps():
    sh = (Path(__file__).resolve().parents[1] / "deploy" / "run-job.sh").read_text()
    assert '"retrain --model all --if-due" "shadows" "card"' in sh


# ------------------------------------------------------------------ 4. calibration vs market

def _league(n, flatten, model_extra, seed=0):
    """Both the market and the model quote from q; outcomes come from q flattened toward 0.5 by
    `flatten` (a league running 'hot': favorites underperform). The model additionally stretches
    its logit by (1 + model_extra) beyond the market."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < 0.5 + (q - 0.5) * flatten).astype(float)
    p = 1 / (1 + np.exp(-np.log(q / (1 - q)) * (1 + model_extra)))
    days = np.repeat(np.arange(n // 4 + 1), 4)[:n].astype(str)
    return y, p, q, days


def test_calibration_vs_market_cancels_league_wide_noise():
    # model == market in a flat league: model-only z is large, model-minus-market has zero spread
    y, p, mkt, days = _league(3000, 0.7, 0.0)
    assert monitor.spiegelhalter_z(y, p) > 4
    assert monitor.calibration_vs_market(y, p, mkt, days) is None     # d == 0: no variance
    # a model stretched beyond the market in that league is overconfident relative to it
    y, p, mkt, days = _league(3000, 0.7, 0.4)
    c = monitor.calibration_vs_market(y, p, mkt, days)
    assert c["z"] > 3 and c["slope_diff"] < 0 and c["n"] == 3000
    # a model shrunk toward 0.5 relative to the market reads underconfident vs it
    y, p, mkt, days = _league(3000, 0.7, -0.3, seed=3)
    assert monitor.calibration_vs_market(y, p, mkt, days)["z"] < -3
    assert monitor.calibration_vs_market(y[:10], p[:10], mkt[:10], days[:10]) is None


def test_window_metrics_stores_cal_vs_mkt():
    y, p, mkt, days = _league(400, 0.7, 0.4)
    day_str = np.array([f"2026-01-{1 + int(d) % 28:02d}" for d in days.astype(int)], dtype="<U10")
    nan = np.full(len(y), np.nan)
    s = monitor.Series(label="gbm_market", model="gbm_market", days=day_str,
                       game_ids=np.arange(len(y)).astype(object), p=p, y=y, margin_pred=nan,
                       margin=nan, total_pred=nan, total=nan, mkt_p=mkt,
                       versions=np.array([""] * len(y), dtype=object))
    det = monitor.window_metrics(s, 0, len(y))["details"]
    assert det["cal_vs_mkt"]["n"] == 400 and det["cal_vs_mkt"]["z"] is not None


def _cvm_snaps(conn, zs, *, model="gbm_market", window="last300", n=300, step_days=1,
               model_only_z=4.0):
    """One snapshot per `step_days`, the last on AS_OF; `zs` oldest first."""
    k = len(zs)
    for i, z in enumerate(zs):
        put_snapshot(conn, model=model, window=window, n=n,
                     as_of=AS_OF - timedelta(days=(k - 1 - i) * step_days),
                     details={"spiegelhalter_z": model_only_z, "cal_slope": 0.8,
                              "cal_vs_mkt": {"n": n, "z": z, "slope_diff": -0.12}})


def calib_alerts(conn):
    return [a for a in alerts.evaluate(conn, AS_OF) if a["key"].startswith("calib:")]


def test_calibration_alert_needs_three_consecutive_breaches_vs_the_market(conn):
    # the model-only z (4.0) would page under the old rule; versus the market it is quiet
    _cvm_snaps(conn, [0.3, 0.5, 0.8])
    assert calib_alerts(conn) == []
    conn.execute("DELETE FROM metric_snapshots")
    _cvm_snaps(conn, [0.5, 2.4, 2.6])                       # only two consecutive breaches
    assert calib_alerts(conn) == []
    conn.execute("DELETE FROM metric_snapshots")
    _cvm_snaps(conn, [2.3, 2.6, 2.4])
    (a,) = calib_alerts(conn)
    assert (a["key"], a["severity"], a["kind"]) == ("calib:gbm_market", "warn", "calibration")
    assert "betting market" in a["message"] and "overconfident" in a["message"]
    assert "3 consecutive daily checks" in a["message"]
    assert [h["z"] for h in a["details"]["history"]] == [2.4, 2.6, 2.3]   # newest first


def test_calibration_alert_direction_span_and_severity(conn):
    _cvm_snaps(conn, [2.5, -2.5, 2.5])                      # sign flips: not persistent
    assert calib_alerts(conn) == []
    conn.execute("DELETE FROM metric_snapshots")
    _cvm_snaps(conn, [2.5, 2.5, 2.5], step_days=5)          # snapshots 10 days apart: stale
    assert calib_alerts(conn) == []
    conn.execute("DELETE FROM metric_snapshots")
    _cvm_snaps(conn, [2.5, 2.6, 2.7], n=150)                # below the n floor
    assert calib_alerts(conn) == []
    conn.execute("DELETE FROM metric_snapshots")
    _cvm_snaps(conn, [-3.8, -3.9, -3.6])                    # underconfident vs market, critical
    (a,) = calib_alerts(conn)
    assert a["severity"] == "critical" and "underconfident" in a["message"]


def test_calibration_alert_falls_back_to_model_only_without_market_games(conn):
    put_snapshot(conn, n=300, details={"spiegelhalter_z": 2.6, "cal_slope": 0.82})
    (a,) = calib_alerts(conn)
    assert a["key"] == "calib:gbm_market" and "z=+2.6" in a["message"]


# ------------------------------------------------------------------ 5. data fingerprint

def _game(conn, gid, day):
    db.upsert_game(conn, game_id=gid, season="2025-26", nba_day=day, tipoff_utc=None,
                   home="BOS", away="NYK", status="final", home_score=100, away_score=90)


def test_data_fingerprint_changes_on_backfill_but_not_the_version_id(conn):
    empty = registry.data_fingerprint(conn)
    assert empty["tables"]["games"] == [0, None] and len(empty["hash"]) == 64
    _game(conn, "g1", "2026-01-02")
    _game(conn, "g2", "2026-01-03")
    fp1 = registry.data_fingerprint(conn)
    assert fp1["hash"] != empty["hash"] and fp1["tables"]["games"] == [2, "2026-01-03"]
    assert registry.data_fingerprint(conn)["hash"] == fp1["hash"]        # deterministic

    params = registry.production_params("gbm_opinion")
    feats = registry.feature_list("gbm_opinion")
    vid = registry.register(conn, "gbm_opinion", params, feats, code="abc")
    expect = registry.spec_hash("gbm_opinion", params, feats, "abc")
    assert vid == f"gbm_opinion@v1-{expect[:8]}"                         # id formula unchanged
    assert registry.get_version(conn, vid)["spec_hash"] == expect
    assert registry.version_fingerprint(conn, vid) == fp1

    _game(conn, "g0", "2025-12-01")                                      # a backfill: older row
    fp2 = registry.data_fingerprint(conn)
    assert fp2["hash"] != fp1["hash"]
    diff = registry.fingerprint_diff(registry.version_fingerprint(conn, vid), fp2)
    assert len(diff) == 1 and diff[0].startswith("games: rows 2->3") and "backfill-like" in diff[0]
    # the same spec registered again is the same version; its recorded fingerprint is untouched
    assert registry.register(conn, "gbm_opinion", params, feats, code="abc") == vid
    assert registry.version_fingerprint(conn, vid) == fp1
    assert registry.fingerprint_diff(fp1, fp1) == [] and registry.fingerprint_diff(None, fp1) == []


def test_missing_tables_do_not_break_the_fingerprint(conn):
    conn.execute("DROP TABLE IF EXISTS dfs_contests")
    conn.commit()
    assert registry.data_fingerprint(conn)["tables"]["dfs_contests"] == [0, None]


# ------------------------------------------------------------------ 6. seasonal PSI

def _fill_elo(conn, start, n, elo_gap):
    for i in range(n):
        d = date.fromordinal(start.toordinal() + i // 5).isoformat()
        pred(conn, f"{d}-{i}", d, 0.5 + (i % 7) / 20, total=215 + i % 9,
             inputs=_inputs(-3.0 + (i % 10) / 2, elo=(1500 + elo_gap + (i % 11) * 4, 1500)),
             created=f"{d}T12:00:00Z", mkt_spread=-3.0 + (i % 10) / 2)


def test_seasonal_psi_suppressed_early_in_a_season_only(conn):
    # last season: wide elo gaps. This season (from Oct 22): compressed, a big raw PSI.
    _fill_elo(conn, date(2025, 1, 10), 150, 120)
    _fill_elo(conn, date(2025, 10, 22), 150, 0)
    early = date(2025, 11, 20)
    assert drift.seasonal_suppressed(early) and not drift.seasonal_suppressed(date(2026, 2, 1))
    assert not drift.seasonal_suppressed(date(2025, 12, 5))
    rows = {r["feature"]: r for r in drift.compute_drift(conn, early)}
    assert "elo_diff" not in rows and "lineup_strength_diff" not in rows
    assert "market_spread" in rows and "pred_total" in rows              # others still computed
    stored = {r[0] for r in conn.execute("SELECT feature FROM drift_snapshots")}
    assert "elo_diff" not in stored
    # after the suppression window elo_diff is judged again, with its real PSI
    _fill_elo(conn, date(2025, 11, 25), 150, 0)
    late = {r["feature"]: r for r in drift.compute_drift(conn, date(2025, 12, 20))}
    assert late["elo_diff"]["psi"] > 0.5
