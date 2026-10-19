"""mlops.monitor: metric math, pre-tip selection, shadow labelling, market pairing, windows."""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pytest
from mlops_helpers import conn_for, game, odds, pred

from courtside.mlops import monitor


def snap(conn, as_of, model, window):
    r = conn.execute("SELECT * FROM metric_snapshots WHERE as_of=? AND model=? AND window=?",
                     (as_of, model, window)).fetchone()
    return dict(r) if r else None


@pytest.fixture
def two_games(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90)       # home won by 10, total 190
    game(conn, "g2", "2026-01-11", 95, 100)       # away won by 5, total 195
    pred(conn, "g1", "2026-01-10", 0.8, margin=8, total=200)
    pred(conn, "g2", "2026-01-11", 0.6, margin=4, total=190)
    odds(conn, "g1", -200, 200, spread=-4, total=210)      # market p = 2/3
    odds(conn, "g2", -300, 300, spread=-6, total=185)      # market p = 0.75
    return conn


def test_metric_math_matches_hand_values(two_games):
    monitor.compute_metrics(two_games, date(2026, 1, 15))
    m = snap(two_games, "2026-01-15", "gbm_market", "season")
    assert m["n"] == 2
    assert m["log_loss"] == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2, abs=1e-5)
    assert m["brier"] == pytest.approx((0.04 + 0.36) / 2)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["home_pred"] == pytest.approx(0.7) and m["home_actual"] == pytest.approx(0.5)
    assert m["margin_bias"] == pytest.approx((8 - 10 + 4 - (-5)) / 2)       # 3.5
    assert m["margin_mae"] == pytest.approx((2 + 9) / 2)
    assert m["total_bias"] == pytest.approx((200 - 190 + 190 - 195) / 2)    # 2.5
    assert m["total_mae"] == pytest.approx((10 + 5) / 2)
    # market on the same games, paired difference and its standard error
    mk1, mk2 = -math.log(2 / 3), -math.log(0.25)
    d = np.array([-math.log(0.8) - mk1, -math.log(0.4) - mk2])
    assert m["mkt_n"] == 2
    assert m["mkt_log_loss"] == pytest.approx((mk1 + mk2) / 2, abs=1e-5)
    assert m["ll_diff_vs_mkt"] == pytest.approx(d.mean(), abs=1e-5)
    assert m["ll_diff_se"] == pytest.approx(d.std(ddof=1) / math.sqrt(2), abs=1e-5)
    det = json.loads(m["details_json"])
    assert sum(b["n"] for b in det["calibration"]) == 2
    b8 = [b for b in det["calibration"] if b["n"] and b["lo"] == 0.8][0]
    assert (b8["pred"], b8["actual"]) == (0.8, 1.0)
    assert det["total_bias_se"] == pytest.approx(np.std([10, -5], ddof=1) / math.sqrt(2), abs=1e-4)


def test_market_is_its_own_pseudo_model(two_games):
    monitor.compute_metrics(two_games, date(2026, 1, 15))
    m = snap(two_games, "2026-01-15", "market", "season")
    assert m["n"] == 2
    assert m["log_loss"] == pytest.approx((-math.log(2 / 3) - math.log(0.25)) / 2, abs=1e-5)
    assert m["ll_diff_vs_mkt"] is None
    # market margin is -spread: (4 - 10) and (6 - (-5)) -> mean 2.5; total line 210/185
    assert m["margin_bias"] == pytest.approx(((4 - 10) + (6 + 5)) / 2)
    assert m["total_bias"] == pytest.approx(((210 - 190) + (185 - 195)) / 2)


def test_market_falls_back_to_closing_line(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90)
    pred(conn, "g1", "2026-01-10", 0.7)
    conn.execute("INSERT INTO closing_lines (game_id, source, home_ml_close, away_ml_close, "
                 "home_spread_close, total_close) VALUES ('g1','espn',-200,200,-4,210)")
    conn.commit()
    monitor.compute_metrics(conn, date(2026, 1, 15))
    m = snap(conn, "2026-01-15", "market", "season")
    assert m["log_loss"] == pytest.approx(-math.log(2 / 3), abs=1e-5)


def test_game_without_market_is_unpaired(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90)
    game(conn, "g2", "2026-01-11", 95, 100)
    pred(conn, "g1", "2026-01-10", 0.8)
    pred(conn, "g2", "2026-01-11", 0.6)
    odds(conn, "g1", -200, 200)
    monitor.compute_metrics(conn, date(2026, 1, 15))
    m = snap(conn, "2026-01-15", "gbm_market", "season")
    assert (m["n"], m["mkt_n"]) == (2, 1)
    assert m["ll_diff_vs_mkt"] == pytest.approx(-math.log(0.8) + math.log(2 / 3), abs=1e-5)
    assert m["ll_diff_se"] is None            # one pair: no standard error


def test_latest_pretip_prediction_is_scored(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90, tip="2026-01-10T00:00:00Z")
    pred(conn, "g1", "2026-01-10", 0.50, created="2026-01-09T10:00:00Z")
    pred(conn, "g1", "2026-01-10", 0.70, created="2026-01-09T23:00:00Z")   # the one to score
    pred(conn, "g1", "2026-01-10", 0.10, created="2026-01-10T01:00:00Z")   # after tip-off
    monitor.compute_metrics(conn, date(2026, 1, 15))
    m = snap(conn, "2026-01-15", "gbm_market", "season")
    assert m["n"] == 1 and m["home_pred"] == pytest.approx(0.7)


def test_unfinished_and_future_games_are_not_scored(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", None, None, status="scheduled")
    game(conn, "g2", "2026-01-15", 100, 90)            # played on as_of itself: not before
    pred(conn, "g1", "2026-01-10", 0.7)
    pred(conn, "g2", "2026-01-15", 0.7)
    assert monitor.compute_metrics(conn, date(2026, 1, 15)) == []


def test_shadow_rows_are_labelled_separately(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90)
    pred(conn, "g1", "2026-01-10", 0.70, role="champion", version="v1", created="2026-01-10T12:00:00Z")
    pred(conn, "g1", "2026-01-10", 0.40, role="shadow", version="v2", created="2026-01-10T12:00:01Z")
    pred(conn, "g1", "2026-01-10", 0.30, model="old", role="retired", version="v0")   # ignored
    monitor.compute_metrics(conn, date(2026, 1, 15))
    champ = snap(conn, "2026-01-15", "gbm_market", "season")
    shadow = snap(conn, "2026-01-15", "gbm_market@v2", "season")
    assert champ["home_pred"] == pytest.approx(0.7) and champ["model_version"] == "v1"
    assert shadow["home_pred"] == pytest.approx(0.4) and shadow["model_version"] == "v2"
    models = {r[0] for r in conn.execute("SELECT DISTINCT model FROM metric_snapshots")}
    assert models == {"gbm_market", "gbm_market@v2"}


def test_legacy_null_role_counts_as_champion(tmp_path):
    conn = conn_for(tmp_path)
    game(conn, "g1", "2026-01-10", 100, 90)
    pred(conn, "g1", "2026-01-10", 0.70)
    monitor.compute_metrics(conn, date(2026, 1, 15))
    assert snap(conn, "2026-01-15", "gbm_market", "season")["n"] == 1


def test_result_from_team_games_when_no_games_row(tmp_path):
    conn = conn_for(tmp_path)
    for team, pts, home in (("BOS", 110, 1), ("NYK", 100, 0)):
        conn.execute("INSERT INTO team_games (game_id, team, season, game_date, is_home, min, pts)"
                     " VALUES ('x1', ?, '2025-26', '2026-01-10', ?, 240, ?)", (team, home, pts))
    conn.commit()
    pred(conn, "x1", "2026-01-10", 0.6, margin=7, total=205)
    monitor.compute_metrics(conn, date(2026, 1, 15))
    m = snap(conn, "2026-01-15", "gbm_market", "season")
    assert m["n"] == 1 and m["margin_bias"] == pytest.approx(7 - 10)
    assert m["total_bias"] == pytest.approx(205 - 210)


@pytest.fixture
def daily(tmp_path):
    """One game per day 2025-12-20 and 2026-01-01..12, one from last season; home wins on even
    days."""
    conn = conn_for(tmp_path)
    days = ["2025-06-20", "2025-12-20"] + [f"2026-01-{d:02d}" for d in range(1, 13)]
    for i, d in enumerate(days):
        game(conn, f"d{i}", d, 100 if i % 2 == 0 else 90, 90 if i % 2 == 0 else 100)
        pred(conn, f"d{i}", d, 0.6)
    return conn


def test_window_boundaries(daily):
    monitor.compute_metrics(daily, date(2026, 1, 13))
    n = {w: snap(daily, "2026-01-13", "gbm_market", w)["n"] for w in monitor.WINDOWS}
    # 7d = Jan 6..12; 30d = Dec 14..Jan 12 (adds Dec 20 and Jan 1..5); season = since 2025-07-01
    assert n == {"7d": 7, "30d": 13, "last100": 14, "last300": 14, "season": 13}
    # as_of 2026-01-12 excludes the Jan 12 game and starts 7d on Jan 5
    monitor.compute_metrics(daily, date(2026, 1, 12))
    assert snap(daily, "2026-01-12", "gbm_market", "7d")["n"] == 7
    assert snap(daily, "2026-01-12", "gbm_market", "season")["n"] == 12


def test_last_n_windows_cap(tmp_path):
    conn = conn_for(tmp_path)
    for i in range(130):
        d = date.fromordinal(date(2025, 11, 1).toordinal() + i // 2).isoformat()
        game(conn, f"g{i:03d}", d, 100, 90)
        pred(conn, f"g{i:03d}", d, 0.6)
    monitor.compute_metrics(conn, date(2026, 2, 1))
    assert snap(conn, "2026-02-01", "gbm_market", "last100")["n"] == 100
    assert snap(conn, "2026-02-01", "gbm_market", "last300")["n"] == 130


def test_replay_matches_single_day_compute(daily):
    daily.execute("DELETE FROM metric_snapshots")
    out = monitor.replay(daily, date(2026, 1, 8), date(2026, 1, 13))
    assert out["days"] == 6
    replayed = {(r["as_of"], r["model"], r["window"]): dict(r) for r in
                daily.execute("SELECT * FROM metric_snapshots")}
    assert {k[0] for k in replayed} == {f"2026-01-{d:02d}" for d in range(8, 14)}
    monitor.compute_metrics(daily, date(2026, 1, 11))
    single = dict(daily.execute("SELECT * FROM metric_snapshots WHERE as_of='2026-01-11' AND "
                                "model='gbm_market' AND window='30d'").fetchone())
    assert single == replayed[("2026-01-11", "gbm_market", "30d")]


def test_recompute_replaces_not_duplicates(two_games):
    monitor.compute_metrics(two_games, date(2026, 1, 15))
    a = two_games.execute("SELECT COUNT(*) FROM metric_snapshots").fetchone()[0]
    monitor.compute_metrics(two_games, date(2026, 1, 15))
    assert two_games.execute("SELECT COUNT(*) FROM metric_snapshots").fetchone()[0] == a


def test_ece_matches_model_metrics(two_games):
    from courtside.model import metrics
    monitor.compute_metrics(two_games, date(2026, 1, 15))
    m = snap(two_games, "2026-01-15", "gbm_market", "season")
    assert m["ece"] == pytest.approx(metrics.ece([1, 0], [0.8, 0.6], 10), abs=1e-5)


def test_cli_status_and_alerts_on_empty_and_filled_db(capsys, data_dir):
    from courtside import db
    from courtside.mlops import cli
    assert cli.main(["status", "--as-of", "2026-01-15"]) == 0
    assert "no metric snapshots yet" in capsys.readouterr().out
    conn = db.connect()
    game(conn, "g1", "2026-01-10", 100, 90)
    pred(conn, "g1", "2026-01-10", 0.8, margin=8, total=200)
    odds(conn, "g1", -200, 200)
    conn.close()
    assert cli.main(["monitor", "--as-of", "2026-01-15"]) == 0
    assert cli.main(["status", "--as-of", "2026-01-15"]) == 0
    out = capsys.readouterr().out
    assert "gbm_market" in out and "market" in out and "Open alerts" in out
    assert cli.main(["alerts", "--all"]) == 0
    assert cli.main(["checks", "--job", "pregame", "--date", "2026-01-15"]) in (0, 1)
