"""Line-freshness bound in pipeline.environment (lf, `pipeline.LINE_FRESH`, default OFF; B9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from courtside import db
from courtside.proj import pipeline as P

AT = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
GAME = "G1"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def switch_off():
    P.set_line_fresh(False)
    yield
    P.set_line_fresh(False)


def put(conn, book, mins_old, spread, total, htt=None, att=None, at=AT):
    ts = (at - timedelta(minutes=mins_old)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:
        conn.execute("INSERT INTO line_snapshots (game_id, captured_at, source, book, "
                     "home_spread, total, home_team_total, away_team_total) "
                     "VALUES (?, ?, 'test', ?, ?, ?, ?, ?)", (GAME, ts, book, spread, total,
                                                            htt, att))


def env(conn, on):
    P.set_line_fresh(on)
    return P.environment(conn, GAME, AT, mode="live")


def test_default_off_and_tag():
    assert P.LINE_FRESH is False and P.LINE_FRESH_TAG == "+lf"
    base = P.MODEL_VERSION
    assert "+lf" not in base
    P.set_line_fresh(True)
    assert P.MODEL_VERSION == base + "+lf" and P._VS == P._versions(
        P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX,
        P.BLOWOUT)[0]
    P.set_line_fresh(False)
    assert P.MODEL_VERSION == base


def test_all_fresh_identical_on_off(conn):
    put(conn, "a", 10, -3.5, 220.5, 111.5, 109.0)
    put(conn, "b", 30, -4.5, 222.5, 112.0, 110.5)
    put(conn, "c", 55, -3.0, 221.0, 111.0, 110.0)
    off, on = env(conn, False), env(conn, True)
    for k in ("mu_margin", "mu_total", "home_pts", "away_pts", "source", "captured_at",
              "line_ages_min", "line_age_max_min"):
        assert off[k] == on[k]
    assert off["source"] == "books:3"
    assert off["line_ages_min"] == pytest.approx({"a": 10, "b": 30, "c": 55})
    assert off["line_age_max_min"] == pytest.approx(55)


def test_off_keeps_stale_book_and_records_ages(conn):
    put(conn, "a", 5, -3.5, 220.0)
    put(conn, "b", 10, -3.5, 220.0)
    put(conn, "old", 600, -9.5, 240.0)
    e = env(conn, False)
    assert e["source"] == "books:3" and e["mu_total"] == 220.0   # median of 3 is not moved here
    assert e["line_ages_min"]["old"] == pytest.approx(600)
    assert e["line_age_max_min"] == pytest.approx(600)


def test_on_drops_book_older_than_3h(conn):
    put(conn, "a", 5, -3.5, 220.0)
    put(conn, "b", 200, -3.5, 230.0)      # 3h20m old -> dropped (also > 1h behind newest)
    put(conn, "c", 20, -4.5, 224.0)
    off, on = env(conn, False), env(conn, True)
    assert off["source"] == "books:3" and off["mu_total"] == 224.0
    assert on["source"] == "books:2" and on["mu_total"] == 222.0
    assert on["mu_margin"] == 4.0
    assert on["line_age_max_min"] == pytest.approx(20)
    assert on["line_ages_min"]["b"] == pytest.approx(200)      # ages cover every book seen


def test_on_relative_one_hour_rule(conn):
    put(conn, "a", 5, -3.5, 220.0)
    put(conn, "b", 60, -3.5, 222.0)       # exactly 55 min behind newest -> kept
    put(conn, "c", 75, -9.5, 240.0)       # 70 min behind newest, under 3h -> dropped
    on = env(conn, True)
    assert on["source"] == "books:2" and on["mu_total"] == 221.0
    assert on["line_age_max_min"] == pytest.approx(60)
    assert env(conn, False)["source"] == "books:3"


def test_on_newest_book_always_kept(conn):
    put(conn, "a", 400, -3.5, 220.0)      # everything older than 3h: newest stays alone
    put(conn, "b", 500, -9.5, 240.0)
    put(conn, "c", 700, -9.5, 240.0)
    on = env(conn, True)
    assert on["source"] == "books:1" and on["mu_total"] == 220.0 and on["mu_margin"] == 3.5
    assert on["line_age_max_min"] == pytest.approx(400)


def test_on_single_stale_book_is_kept(conn):
    put(conn, "a", 1000, -3.5, 220.0)
    assert env(conn, True)["source"] == "books:1"


def test_on_ties_for_newest_both_kept(conn):
    put(conn, "a", 400, -3.5, 220.0)
    put(conn, "b", 400, -4.5, 222.0)
    put(conn, "c", 450, -9.5, 240.0)
    on = env(conn, True)
    assert on["source"] == "books:2" and on["mu_total"] == 221.0


def test_team_totals_follow_the_kept_books(conn):
    put(conn, "a", 5, -3.5, 220.0, 111.0, 109.0)
    put(conn, "b", 10, -3.5, 220.0, 113.0, 107.0)
    put(conn, "stale", 100, -3.5, 220.0, 125.0, 95.0)
    off, on = env(conn, False), env(conn, True)
    assert on["source"] == "books:2" and off["source"] == "books:3"
    assert on["home_pts"] != pytest.approx(off["home_pts"])   # ON medians (111,113), OFF 113


def test_closing_mode_untouched(conn):
    put(conn, "a", 5, -3.5, 220.0)
    with conn:
        conn.execute("INSERT INTO closing_lines (game_id, source, home_spread_close, "
                     "total_close) VALUES (?, 's', -2.0, 210.0)", (GAME,))
    P.set_line_fresh(True)
    e = P.environment(conn, GAME, None, mode="closing")
    assert e["source"] == "closing" and "line_ages_min" not in e
