"""G1 news since the line (COURTSIDE_NEWS_SINCE_LINE): OFF is bit-identical, ON moves the
headline model's line by beta x the opinion model's value of the news since line capture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synth import day_date, synth_conn

from courtside import predict
from courtside.model import line_news

DAY = day_date("2024-25", 22)
INS = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
       "VALUES (?,?,?,?,?,?)")


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)
    monkeypatch.setattr(line_news, "ON", False)


def _closing_lines(conn, before, noise=4.0):
    rng = np.random.default_rng(3)
    rows = conn.execute("SELECT h.game_id, h.pts - a.pts AS m, h.pts + a.pts AS t FROM team_games h "
                        "JOIN team_games a ON a.game_id=h.game_id AND a.is_home=0 "
                        "WHERE h.is_home=1 AND h.game_date < ?", (before.isoformat(),)).fetchall()
    conn.executemany(
        "INSERT INTO closing_lines (game_id, source, provider, home_spread_close, total_close, "
        "home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', ?, ?, -150, 130)",
        [(r["game_id"], -0.7 * r["m"] + rng.normal(0, noise), r["t"] + rng.normal(0, noise))
         for r in rows])
    conn.commit()


def _snapshot(conn, at, spread=-4.0, total=221.0):
    for (gid,) in conn.execute("SELECT DISTINCT game_id FROM team_games WHERE game_date=?",
                               (DAY.isoformat(),)).fetchall():
        conn.execute("INSERT INTO odds_snapshots (game_id, captured_at, source, provider, "
                     "home_spread, total, home_ml, away_ml) VALUES (?,?,?,?,?,?,?,?)",
                     (gid, at, "espn", "DK", spread, total, -180, 155))
    conn.commit()


def _world(line_at=f"{DAY}T11:00:00Z", healthy_at=f"{DAY}T10:00:00Z",
           out_at=f"{DAY}T15:00:00Z", list_home_early=True):
    """Past day with closing history, one snapshot line and two official reports: the home
    star of the first game listed Available (healthy_at), then Out (out_at)."""
    conn = synth_conn()
    _closing_lines(conn, DAY)
    base = predict.predict_day(conn, DAY, store=False)
    g = base[0]
    home, away = g["home"], g["away"]
    early_team = home if list_home_early else away
    conn.execute(INS, (healthy_at, "nba_official", early_team, f"{early_team} Player 0",
                       "Available", "x"))
    conn.execute(INS, (out_at, "nba_official", home, f"{home} Player 0", "Out", "knee"))
    conn.execute(INS, (out_at, "nba_official", away, f"{away} Player 0", "Available", "x"))
    conn.commit()
    if line_at:
        _snapshot(conn, line_at)
    return conn, g["game_id"]


def _by_game(rows):
    return {r["game_id"]: r for r in rows}


def _numbers(rows):
    keys = ("home_win_prob", "home_margin", "total", "market_home_spread", "market_total",
            "market_home_prob")
    return [(r["game_id"], *(r[k] for k in keys), *(r["opinion"][k] for k in keys))
            for r in rows]


def _strip(rows):
    def one(r):
        out = {k: v for k, v in r.items() if k not in ("created_at", "opinion")}
        out["inputs"] = {k: v for k, v in r["inputs"].items() if k != "fit"}
        if r.get("opinion"):
            out["opinion"] = one(r["opinion"])
        return out
    return [one(r) for r in rows]


# --- switch plumbing ---------------------------------------------------------------------------

def test_default_is_off_and_env_parsing(monkeypatch):
    assert line_news.DEFAULT is False
    for v, want in (("1", True), ("on", True), ("true", True), ("0", False), ("off", False),
                    ("", False), ("junk", False)):
        monkeypatch.setenv(line_news.ENV, v)
        assert line_news._env_flag(line_news.ENV, line_news.DEFAULT) is want
    assert line_news.tagged("gbm_market@v1-abc") == "gbm_market@v1-abc+nsl"
    assert line_news.tagged(None) is None


def test_capture_time_only_for_snapshot_lines():
    assert line_news.capture_time({"kind": "snapshot", "captured_at": "2025-01-01T11:00:00Z"}) \
        == "2025-01-01T11:00:00Z"
    assert line_news.capture_time({"kind": "closing", "captured_at": None}) is None
    assert line_news.capture_time(None) is None


def test_guard_keeps_current_lineup_unless_listed_both_times():
    now, cap = np.array([1.0, 2.0, 3.0, 4.0]), np.array([9.0, 9.0, 9.0, np.nan])
    got = line_news.guarded(now, cap, [True, True, False, True], [True, False, True, True])
    assert got.tolist() == [9.0, 2.0, 3.0, 4.0]


def test_zero_shift_reproduces_the_features_exactly():
    G = pd.DataFrame({"mkt_margin": [3.5, np.nan], "mkt_total": [220.0, np.nan], "x": [1, 2]})
    s = {"shift_margin": np.zeros(2), "shift_total": np.zeros(2)}
    out = line_news.apply_shift(G, s)
    pd.testing.assert_frame_equal(out, G)


# --- OFF: bit-identical ---------------------------------------------------------------------------

def test_off_never_enters_the_new_code_and_is_deterministic(monkeypatch):
    conn, _ = _world()
    ref = predict.predict_day(conn, DAY, store=False)

    def boom(*a, **k):
        raise AssertionError("news-since-line code ran with the switch OFF")
    monkeypatch.setattr(predict, "_news_since_line", boom)
    monkeypatch.setattr(line_news, "shift", boom)
    monkeypatch.setattr(line_news, "apply_shift", boom)
    off = predict.predict_day(conn, DAY, store=False)
    assert _strip(off) == _strip(ref)
    for r in off:
        assert "news_since_line" not in r["inputs"]
        assert r["model_version"] == r["inputs"]["model_version"]
        assert r["model_version"] is None or not r["model_version"].endswith(line_news.TAG)


def test_on_with_zero_beta_prices_exactly_like_off(monkeypatch):
    conn, _ = _world()
    off = predict.predict_day(conn, DAY, store=False)
    monkeypatch.setattr(line_news, "ON", True)
    monkeypatch.setattr(line_news, "BETA_MARGIN", 0.0)
    monkeypatch.setattr(line_news, "BETA_TOTAL", 0.0)
    on = predict.predict_day(conn, DAY, store=False)
    assert _numbers(on) == _numbers(off)
    assert any(r["inputs"]["news_since_line"]["delta_margin"] != 0 for r in on)


# --- ON ---------------------------------------------------------------------------------------------

def test_on_moves_the_line_by_beta_times_the_opinion_news(monkeypatch):
    conn, gid = _world()
    off = _by_game(predict.predict_day(conn, DAY, store=False))
    # the opinion model's value of the news = its margin with the star out minus healthy
    after_tip = (pd.Timestamp(DAY) + pd.Timedelta(days=1, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn2, _ = _world(out_at=after_tip)                   # the Out report comes after the tip
    healthy = _by_game(predict.predict_day(conn2, DAY, store=False))
    news = off[gid]["opinion"]["home_margin"] - healthy[gid]["opinion"]["home_margin"]
    assert abs(news) > 0.05                               # the absence is worth something

    monkeypatch.setattr(line_news, "ON", True)
    monkeypatch.setattr(line_news, "BETA_MARGIN", 1.0)
    monkeypatch.setattr(line_news, "BETA_TOTAL", 0.5)
    on = _by_game(predict.predict_day(conn, DAY, store=False))
    rec = on[gid]["inputs"]["news_since_line"]
    assert rec["line_at"] == f"{DAY}T11:00:00Z"
    assert rec["sides"] == {"home": True, "away": False}  # away unlisted at line capture
    assert rec["delta_margin"] == pytest.approx(news, abs=2e-4)
    assert rec["shift_margin"] == pytest.approx(rec["delta_margin"], abs=1e-4)
    assert rec["shift_total"] == pytest.approx(0.5 * rec["delta_total"], abs=1e-4)
    # the opinion model and the stored market columns are untouched
    assert on[gid]["opinion"]["home_margin"] == off[gid]["opinion"]["home_margin"]
    assert "news_since_line" not in on[gid]["opinion"]["inputs"]
    assert on[gid]["market_home_spread"] == off[gid]["market_home_spread"]
    # the headline model priced the shifted line: same as OFF fed that line directly
    assert np.sign(on[gid]["home_margin"] - off[gid]["home_margin"]) == np.sign(news)
    sh = on[gid]["inputs"]["news_since_line"]
    conn.execute("UPDATE odds_snapshots SET home_spread=home_spread-?, total=total+? "
                 "WHERE game_id=?", (sh["shift_margin"], sh["shift_total"], gid))
    conn.commit()
    monkeypatch.setattr(line_news, "ON", False)
    fed = _by_game(predict.predict_day(conn, DAY, store=False))
    assert on[gid]["home_margin"] == pytest.approx(fed[gid]["home_margin"], abs=1e-3)
    assert on[gid]["total"] == pytest.approx(fed[gid]["total"], abs=1e-3)
    # games with no news since the line are not moved
    for g, r in on.items():
        if g != gid:
            assert r["inputs"]["news_since_line"]["shift_margin"] == 0
            assert r["home_margin"] == off[g]["home_margin"]


def test_on_no_shift_when_the_team_was_not_listed_at_capture(monkeypatch):
    conn, gid = _world(list_home_early=False)              # only the away team filed early
    off = _by_game(predict.predict_day(conn, DAY, store=False))
    monkeypatch.setattr(line_news, "ON", True)
    monkeypatch.setattr(line_news, "BETA_MARGIN", 1.0)
    on = _by_game(predict.predict_day(conn, DAY, store=False))
    rec = on[gid]["inputs"]["news_since_line"]
    assert rec["sides"]["home"] is False and rec["shift_margin"] == 0
    assert on[gid]["home_margin"] == off[gid]["home_margin"]


def test_on_no_shift_before_any_report_or_after_the_news(monkeypatch):
    monkeypatch.setattr(line_news, "BETA_MARGIN", 1.0)
    for line_at in (f"{DAY}T09:00:00Z",                    # before any report: nothing to diff
                    f"{DAY}T16:00:00Z"):                   # the line already saw the Out
        conn, gid = _world(line_at=line_at)
        monkeypatch.setattr(line_news, "ON", False)
        off = _by_game(predict.predict_day(conn, DAY, store=False))
        monkeypatch.setattr(line_news, "ON", True)
        on = _by_game(predict.predict_day(conn, DAY, store=False))
        assert on[gid]["inputs"]["news_since_line"]["shift_margin"] == 0, line_at
        assert _numbers(list(on.values())) == _numbers(list(off.values())), line_at


def test_on_closing_lines_are_never_moved(monkeypatch):
    conn, gid = _world(line_at=None)
    conn.execute("INSERT INTO closing_lines (game_id, source, provider, home_spread_close, "
                 "total_close, home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', -3.5, 221, "
                 "-160, 140)", (gid,))
    conn.commit()
    off = _by_game(predict.predict_day(conn, DAY, store=False))
    monkeypatch.setattr(line_news, "ON", True)
    monkeypatch.setattr(line_news, "BETA_MARGIN", 1.0)
    on = _by_game(predict.predict_day(conn, DAY, store=False))
    assert on[gid]["inputs"]["market"]["kind"] == "closing"
    rec = on[gid]["inputs"]["news_since_line"]
    assert rec["line_at"] is None and rec["shift_margin"] == 0
    assert _numbers(list(on.values())) == _numbers(list(off.values()))
