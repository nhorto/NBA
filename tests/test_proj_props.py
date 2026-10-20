"""proj/props.py (T2.9): market maths, our per-stat distribution, the join, storage, grading."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from courtside import collect, db
from courtside.proj import props as P

import props_helpers as H


# ------------------------------------------------------------------ market maths

def test_implied_and_no_vig():
    assert P.implied(-110) == pytest.approx(110 / 210)
    assert P.implied(150) == pytest.approx(0.4)
    assert P.implied(None) is None and P.implied(50) is None
    p, hold = P.no_vig(-110, -110)
    assert p == pytest.approx(0.5) and hold == pytest.approx(2 * 110 / 210 - 1)
    p, hold = P.no_vig(120, -150)           # 0.4545 / (0.4545 + 0.6)
    assert p == pytest.approx((100 / 220) / (100 / 220 + 0.6))
    assert hold > 0
    # the two no-vig sides sum to 1 whatever the margin
    po, _ = P.no_vig(-140, 115)
    pu, _ = P.no_vig(115, -140)
    assert po + pu == pytest.approx(1.0)
    assert P.no_vig(-110, None) == (None, None)


# ------------------------------------------------------------------ distribution

def test_minutes_draws_follow_the_quantiles():
    m = P.minutes_draws({"q05": 18, "q10": 21, "q25": 26, "q50": 30, "q75": 34, "q90": 37,
                         "q95": 39})
    assert len(m) == P.N_MINUTES and np.all(np.diff(m) >= 0)
    assert np.median(m) == pytest.approx(30, abs=0.6)
    assert 0.5 <= m.min() < 18 and 39 < m.max() <= P.MAX_MINUTES
    assert list(P.minutes_draws({}, 24.0)) == [24.0]        # no quantiles: the mean
    assert len(P.minutes_draws(None, None)) == 0


def test_pmf_mean_and_dispersion():
    pmf = P.stat_pmf(18.0, np.array([30.0]), 2.4)           # one minutes value: plain NB1
    k = np.arange(len(pmf))
    assert pmf.sum() == pytest.approx(1.0)
    mean = (k * pmf).sum()
    assert mean == pytest.approx(18.0, rel=1e-6)
    assert ((k - mean) ** 2 * pmf).sum() == pytest.approx(2.4 * 18.0, rel=1e-3)
    pois = P.stat_pmf(3.0, None, 1.0)
    assert (np.arange(len(pois)) * pois).sum() == pytest.approx(3.0)
    assert pois[0] == pytest.approx(math.exp(-3.0))
    # minutes uncertainty widens the distribution, the mean stays put
    wide = P.stat_pmf(18.0, P.minutes_draws({"q05": 10, "q10": 14, "q25": 22, "q50": 30,
                                             "q75": 34, "q90": 37, "q95": 39}), 2.4)
    kw = np.arange(len(wide))
    assert (kw * wide).sum() == pytest.approx(18.0, rel=1e-6)
    assert ((kw - 18) ** 2 * wide).sum() > ((k - 18) ** 2 * pmf).sum()
    assert list(P.stat_pmf(0.0, None, 2.0)) == [1.0]


def test_line_probs_half_line_has_no_push():
    pmf = P.stat_pmf(20.0, None, 2.4)
    above, push, below = P.line_probs(pmf, 20.5)
    assert push == 0.0
    assert above == pytest.approx(pmf[21:].sum())
    assert below == pytest.approx(pmf[:21].sum())
    assert above + below == pytest.approx(1.0)
    assert P.comparable(above, push) == pytest.approx(above)


def test_line_probs_integer_line_pushes():
    pmf = P.stat_pmf(5.2, None, 1.11)
    above, push, below = P.line_probs(pmf, 5.0)
    assert push == pytest.approx(pmf[5]) and push > 0.1
    assert above == pytest.approx(pmf[6:].sum())
    assert below == pytest.approx(pmf[:5].sum())
    assert above + push + below == pytest.approx(1.0)
    # the market refunds the push: ours on the same basis
    assert P.comparable(above, push) == pytest.approx(above / (above + below))
    # a line of 0: only the push and above
    a0, p0, b0 = P.line_probs(pmf, 0.0)
    assert b0 == 0.0 and p0 == pytest.approx(pmf[0])


def test_more_mean_more_chance_above():
    lo = P.line_probs(P.stat_pmf(18.0, None, 2.4), 20.5)[0]
    hi = P.line_probs(P.stat_pmf(23.0, None, 2.4), 20.5)[0]
    assert lo < 0.5 < hi


def test_player_dist_pra_and_ranges():
    stats = {(s, "played"): {"mean": v} for s, v in
             {"pts": 20.0, "reb": 5.0, "ast": 4.0, "fg3m": 2.0}.items()}
    stats[("min", "played")] = dict(zip(P.QCOLS, H.MINQ), mean=30.0)
    d = P.player_dist(stats)
    assert d.means["pra"] == pytest.approx(29.0)
    lo, hi = d.range("pts")
    assert lo < 20 < hi
    assert d.pmf("pts") is d.pmf("pts")                      # memoised
    stats.pop(("ast", "played"))
    assert P.player_dist(stats).means["pra"] is None


# ------------------------------------------------------------------ the join

@pytest.fixture
def conn():
    c = db.connect()
    H.populate(c)
    return c


def test_join_rate_runs_and_reasons(conn):
    res = P.compare_day(conn, H.DAY)
    s = res["summary"]
    assert s["props"] == H.N_MAIN and s["joined"] == H.N_JOINED
    assert s["join_rate"] == pytest.approx(H.N_JOINED / H.N_MAIN)
    assert s["by_reason"] == {P.R_NOT_PROJECTED: 1, P.R_NO_MATCH: 1}
    g1, g2 = res["games"]
    # G1: the failed lock45, the swap run and the post-tip lock10 are all ignored
    assert g1["run"]["run_id"] == "r_main"
    assert g2["run"]["run_id"] == "r_late"                   # newest before G2's tip
    rows = {(r["player_id"], r["stat"]): r for r in g1["rows"]}
    assert set(rows) == {(101, "pts"), (101, "reb"), (102, "ast"), (201, "pra")}
    # newest pre-tip line (20.5 at 20:00), not the post-tip 30.5, not the milestone
    assert rows[(101, "pts")]["line"] == 20.5
    assert rows[(101, "pts")]["captured_at"] == "2026-10-20T20:00:00Z"
    assert rows[(102, "ast")]["match"] == "nickname"         # "Nic" -> Nicolas Claxton
    assert rows[(201, "pra")]["our_mean"] == pytest.approx(31.0)
    r = rows[(101, "reb")]                                   # integer line: a push
    assert r["our_p_push"] > 0.1
    assert r["our_p_above"] + r["our_p_push"] + r["our_p_below"] == pytest.approx(1.0)
    assert rows[(101, "pts")]["mkt_p_above"] == pytest.approx(0.5)
    by_name = {u["player_name"]: u["reason"] for u in g1["unjoined"]}
    assert by_name == {"Zed Zeta": P.R_NOT_PROJECTED, "Qq Nobody": P.R_NO_MATCH}
    # G2 joined to r_late's (10% higher) means
    g2rows = {r["player_id"]: r for r in g2["rows"]}
    assert g2rows[401]["our_mean"] == pytest.approx(12.0 * 1.1)


def test_at_caps_runs_and_market(conn):
    # as of 16:00: the 20:00 line move has not happened, r_late does not exist yet
    res = P.compare_day(conn, H.DAY, at="2026-10-20T16:00:00Z")
    g1, g2 = res["games"]
    pts = next(r for r in g1["rows"] if r["player_id"] == 101 and r["stat"] == "pts")
    assert pts["line"] == 19.5
    assert g2["run"]["run_id"] == "r_main"
    # before any run: everything unjoined with the no-run reason
    res = P.compare_day(conn, H.DAY, at="2026-10-20T14:30:00Z")
    assert res["summary"]["joined"] == 0
    assert set(res["summary"]["by_reason"]) == {P.R_NO_RUN}


def test_no_props_no_rows(conn):
    conn.execute("DELETE FROM prop_snapshots")
    res = P.compare_day(conn, H.DAY)
    assert res["summary"]["props"] == 0 and res["summary"]["join_rate"] is None
    assert res["games"][0]["proj"]                            # cards still get projections


def test_alias_beats_a_missing_name(conn):
    conn.execute("INSERT INTO player_aliases (source, name_norm, team, player_id) VALUES "
                 "('espn', 'qq nobody', '', 202)")
    res = P.compare_day(conn, H.DAY)
    assert res["summary"]["joined"] == H.N_JOINED + 1


# ------------------------------------------------------------------ storage + grading

def test_migration_14_columns():
    c = db.connect(":memory:")
    cols = {r[1] for r in c.execute("PRAGMA table_info(prop_compare)")}
    assert {"game_id", "player_id", "run_id", "our_p_push", "our_p_below", "hold", "actual",
            "result", "went_above", "graded_at"} <= cols


def test_write_day_stores_run_time_and_latest_snapshots(conn):
    s = P.write_day(conn, H.DAY, at="2026-10-21T03:00:00Z")
    assert s["joined"] == H.N_JOINED
    rows = conn.execute("SELECT * FROM prop_compare WHERE player_id=101 AND market='pts' "
                        "ORDER BY captured_at").fetchall()
    # 19.5 (state at the 15:00 run) and 20.5 (newest pre-tip)
    assert [(r["line"], r["captured_at"]) for r in rows] == [
        (19.5, "2026-10-20T14:00:00Z"), (20.5, "2026-10-20T20:00:00Z")]
    assert all(r["run_id"] == "r_main" and r["run_created_at"] == "2026-10-20T15:00:00Z"
               for r in rows)
    n = conn.execute("SELECT COUNT(*) FROM prop_compare").fetchone()[0]
    P.write_day(conn, H.DAY, at="2026-10-21T03:00:00Z")        # idempotent
    assert conn.execute("SELECT COUNT(*) FROM prop_compare").fetchone()[0] == n


def test_grade_day(conn):
    P.write_day(conn, H.DAY, at="2026-10-21T03:00:00Z")
    assert P.grade_day(conn, H.DAY)["status"] == "pending"
    H.finish_g1(conn)
    g = P.grade_day(conn, H.DAY)
    assert g["status"] == "ok" and g["games"] == 1
    got = {(r["player_id"], r["market"], r["line"]): (r["actual"], r["result"], r["went_above"])
           for r in conn.execute("SELECT * FROM prop_compare WHERE game_id=?", (H.G1,))}
    assert got[(101, "pts", 20.5)] == (21.0, "above", 1)
    assert got[(101, "pts", 19.5)] == (21.0, "above", 1)
    assert got[(101, "reb", 5.0)] == (5.0, "push", None)
    assert got[(102, "ast", 2.5)] == (2.0, "below", 0)
    assert got[(201, "pra", 30.5)] == (None, "void", None)     # no box row: did not play
    # G2 is not final: untouched
    assert conn.execute("SELECT COUNT(*) FROM prop_compare WHERE game_id=? AND graded_at IS "
                        "NOT NULL", (H.G2,)).fetchone()[0] == 0
    # re-writing the comparison keeps the grades
    P.write_day(conn, H.DAY, at="2026-10-21T03:00:00Z")
    r = conn.execute("SELECT result FROM prop_compare WHERE player_id=101 AND market='reb'"
                     ).fetchone()
    assert r["result"] == "push"


def test_collect_steps(conn):
    s = collect._prop_compare_step(conn, date.fromisoformat(H.DAY))
    assert s["props"] == H.N_MAIN and s["joined"] == H.N_JOINED and s["rows_written"] > 0
    H.finish_g1(conn)
    s = collect._prop_grade_step(conn, date.fromisoformat(H.DAY))
    assert s["grade"]["status"] == "ok"
    empty = collect._prop_compare_step(conn, date(2026, 10, 25))
    assert empty["status"] == "skipped"
    assert collect.PREGAME_STEPS.index("prop_compare") == collect.PREGAME_STEPS.index("props") + 1
