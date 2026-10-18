"""predict_day: no leakage, schedule-only upcoming games, market and injuries attached."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest
from synth import day_date, synth_conn

from courtside import predict
from courtside.model import baselines


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)


DAY = day_date("2024-25", 22)          # a day whose games are already in team_games


def _strip(rows):
    return [{k: v for k, v in r.items() if k != "created_at"} for r in rows]


def test_predict_past_day_from_team_games_and_stores_history():
    conn = synth_conn()
    rows = predict.predict_day(conn, DAY)
    n_games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM team_games WHERE game_date=?",
                           (DAY.isoformat(),)).fetchone()[0]
    assert len(rows) == n_games == 3
    for r in rows:
        assert 0 < r["home_win_prob"] < 1 and r["total"] > 150
        assert r["model"] == predict.PRODUCTION_MODEL
        assert r["created_at"] == f"{DAY.isoformat()}T12:00:00Z"       # past day: virtual time
        assert set(r["injuries"]) == {"home", "away"}
    stored = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert stored == 3
    predict.predict_day(conn, DAY)                                   # idempotent for past days
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 3
    latest = predict.latest_pretip(conn, day=DAY.isoformat())
    assert set(latest) == {r["game_id"] for r in rows}
    assert latest[rows[0]["game_id"]]["inputs"]["schedule"]["home"]["rest"] is not None


class SpyModel:
    """Delegates to the real model but records what it was fit and asked to predict on."""
    seen: list = []

    def __init__(self):
        self.inner = baselines.MODELS["elo_sched"]()
        self.name = "spy"

    def fit(self, train):
        SpyModel.seen.append(("fit", train["game_date"].max(), len(train),
                              int(train["game_id"].nunique())))
        self.inner.fit(train)

    def predict_proba(self, test):
        SpyModel.seen.append(("test", test["game_date"].min(), test["game_date"].max(),
                              float(test["margin"].abs().sum()),
                              float(test["home_pts"].abs().sum())))
        return self.inner.predict_proba(test)

    def predict_margin(self, test):
        return self.inner.predict_margin(test)


def test_fit_uses_only_games_before_day_and_test_results_are_masked(monkeypatch):
    conn = synth_conn()
    SpyModel.seen = []
    monkeypatch.setattr(predict, "model_factory", lambda name, conn=None: SpyModel)
    predict.predict_day(conn, DAY, model="spy", store=False)
    fit = next(s for s in SpyModel.seen if s[0] == "fit")
    test = next(s for s in SpyModel.seen if s[0] == "test")
    assert fit[1] < np.datetime64(DAY)                       # newest training game before DAY
    expected_train = conn.execute("SELECT COUNT(DISTINCT game_id) FROM team_games "
                                  "WHERE game_date < ?", (DAY.isoformat(),)).fetchone()[0]
    assert fit[3] == expected_train
    assert test[1] == test[2] == np.datetime64(DAY)
    assert test[3] == 0.0 and test[4] == 0.0                 # results are placeholders


def test_results_on_or_after_day_never_change_predictions():
    conn = synth_conn()
    before = predict.predict_day(conn, DAY, store=False)
    cut = DAY.isoformat()
    # scramble every result from DAY onward (the games being predicted and everything later)
    conn.execute("UPDATE team_games SET pts = pts + ((abs(random()) % 40) - 20), "
                 "wl = CASE WHEN abs(random()) % 2 = 0 THEN 'W' ELSE 'L' END "
                 "WHERE game_date >= ?", (cut,))
    # keep the pairs consistent for the loader: make it decisive and non-tied
    conn.execute("UPDATE team_games SET pts = pts + 1 WHERE is_home = 1 AND game_date >= ? "
                 "AND pts = (SELECT pts FROM team_games t2 WHERE t2.game_id = team_games.game_id "
                 "AND t2.is_home = 0)", (cut,))
    after = predict.predict_day(conn, DAY, store=False)
    assert _strip(before) == _strip(after)
    # and changing the result of an EARLIER game does change them (the test can fail)
    conn.execute("UPDATE team_games SET pts = pts + 30 WHERE is_home = 1 AND game_date = ?",
                 ((DAY - timedelta(days=1)).isoformat(),))
    changed = predict.predict_day(conn, DAY, store=False)
    assert _strip(changed) != _strip(before)


def test_upcoming_games_come_from_games_table_without_leakage():
    conn = synth_conn()
    last = day_date("2024-25", 39)
    nxt = last + timedelta(days=1)
    conn.executemany(
        "INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
        "VALUES (?,?,?,?,?,?,?)",
        [("0022400901", "2024-25", nxt.isoformat(), f"{nxt}T23:00:00Z", "BOS", "NYK", "scheduled"),
         ("0022400902", "2024-25", nxt.isoformat(), f"{nxt}T23:30:00Z", "LAL", "GSW", "scheduled"),
         ("0012400903", "2024-25", nxt.isoformat(), f"{nxt}T23:30:00Z", "DEN", "MIA", "scheduled"),
         ("0022400904", "2024-25", nxt.isoformat(), None, "DEN", "MIA", "postponed")])
    rows = predict.predict_day(conn, nxt, created_at="2024-12-01T15:00:00Z")
    assert [r["game_id"] for r in rows] == ["0022400901", "0022400902"]   # preseason + postponed skipped
    assert all(r["tipoff_utc"] for r in rows)
    # schedule features exist for games that are not in team_games (b2b etc. from history)
    assert rows[0]["inputs"]["schedule"]["home"]["rest"] in (0, 1, 2, 3, 4)
    # same prediction no matter what happens afterwards: add a result for the game, rerun
    again = predict.predict_day(conn, nxt, created_at="2024-12-01T15:00:00Z", store=False)
    assert _strip(rows) == _strip(again)
    # a later prediction row never replaces the earlier one (append-only)
    predict.predict_day(conn, nxt, created_at="2024-12-01T16:00:00Z")
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE game_id='0022400901'").fetchone()[0]
    assert n == 2


def test_market_line_is_pre_tip_devigged_and_falls_back_to_closing():
    conn = synth_conn()
    nxt = day_date("2024-25", 39) + timedelta(days=1)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES ('0022400901','2024-25',?,?,'BOS','NYK','scheduled')",
                 (nxt.isoformat(), f"{nxt}T23:00:00Z"))
    ins = ("INSERT INTO odds_snapshots (game_id, captured_at, source, provider, home_spread, "
           "total, home_ml, away_ml) VALUES ('0022400901',?,?,?,?,?,?,?)")
    conn.execute(ins, (f"{nxt}T14:00:00Z", "espn", "DK", -3.0, 220.0, -150, 130))
    conn.execute(ins, (f"{nxt}T20:00:00Z", "espn", "DK", -4.0, 221.0, -200, 170))
    conn.execute(ins, (f"{nxt}T23:30:00Z", "espn", "DK", -9.0, 230.0, -900, 600))   # after tip
    conn.commit()
    (r,) = predict.predict_day(conn, nxt, store=False, created_at=f"{nxt}T21:00:00Z")
    assert r["market_home_spread"] == -4.0 and r["market_total"] == 221.0
    ph, pa = 200 / 300, 100 / 270
    assert r["market_home_prob"] == pytest.approx(ph / (ph + pa), abs=1e-4)
    # no snapshots at all: closing_lines is used
    conn.execute("DELETE FROM odds_snapshots")
    gid = conn.execute("SELECT game_id FROM team_games WHERE game_date=? LIMIT 1",
                       (DAY.isoformat(),)).fetchone()[0]
    conn.execute("INSERT INTO closing_lines (game_id, source, provider, home_spread_close, "
                 "total_close, home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', 2.5, 215.5, "
                 "120, -140)", (gid,))
    rows = {x["game_id"]: x for x in predict.predict_day(conn, DAY, store=False)}
    assert rows[gid]["market_home_spread"] == 2.5 and rows[gid]["market_total"] == 215.5
    assert rows[gid]["market_home_prob"] < 0.5
    other = next(x for k, x in rows.items() if k != gid)
    assert other["market_home_prob"] is None and other["market_home_spread"] is None


def test_injuries_attached_from_latest_snapshot():
    conn = synth_conn()
    (g, *_) = predict.predict_day(conn, DAY, store=False)
    home = g["home"]
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
           "VALUES (?,?,?,?,?,?)")
    conn.execute(ins, (f"{DAY - timedelta(days=1)}T15:00:00Z", "espn", home, "Old Guy", "Out", "x"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", home, "Star Man", "Out", "knee"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", home, "Role Guy", "Probable", "x"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", g["away"], "Wing Man", "Questionable", "x"))
    conn.commit()
    rows = {r["game_id"]: r for r in predict.predict_day(conn, DAY, store=False)}
    inj = rows[g["game_id"]]["injuries"]
    assert [x["player"] for x in inj["home"]] == ["Star Man"]          # latest batch, relevant only
    assert [x["status"] for x in inj["away"]] == ["Questionable"]


def test_too_little_history_returns_nothing():
    conn = synth_conn({"2024-25": 3})
    assert predict.predict_day(conn, day_date("2024-25", 2), store=False) == []


def test_backfill_equals_single_day_and_is_walk_forward():
    conn = synth_conn()
    d0, d1 = day_date("2024-25", 10), day_date("2024-25", 14)
    res = predict.predict_days(conn, d0, d1)
    assert res["days"] == 5 and res["games"] == 15
    stored = {r["game_id"]: r["home_win_prob"] for r in conn.execute(
        "SELECT game_id, home_win_prob FROM predictions WHERE nba_day=?", (d1.isoformat(),))}
    single = {r["game_id"]: r["home_win_prob"]
              for r in predict.predict_day(conn, d1, store=False)}
    assert stored == pytest.approx(single, abs=1e-9)
    json.dumps([dict(r) for r in conn.execute("SELECT * FROM predictions")])


def test_registry_is_generic():
    conn = synth_conn()
    assert predict.model_factory("elo")().name == "elo"
    with pytest.raises(KeyError):
        predict.model_factory("nope")
    rows = predict.predict_day(conn, DAY, model="elo", store=False)
    assert rows and rows[0]["model"] == "elo"
