"""Re-evaluation workstream A switches (docs/research/reeval-A.md), all default OFF:

* `+rm` (availability.REPORT_MATCH): a report name the box-history roster cannot resolve is
  matched on the season's `team_rosters` snapshot (second stage only), so a long-absent
  player's 'Out' row is kept; OFF and ON-without-snapshot frames are bit-identical.
* `+gh` (pipeline.GHOST_DROP): `drop_moved_players(strict=True)` also drops an unforced
  candidate who is on no current roster.
* `+lu` (minutes.LINEUP): lineup-known demotion / unexpected-starter weights (OFF 0.5 / 0.5 is
  bit-identical) and P(plays | available) 0.99 for confirmed starters off the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_proj_availability import DAY, conn, name, pid, report, status_of, world  # noqa: F401

from courtside.proj import availability as av
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

AT = "2026-03-20T22:00:00Z"
LONG_OUT = 1999                      # on AAA's current roster, no box row in the lookback


@pytest.fixture
def switches():
    state = (av.REPORT_MATCH, P.GHOST_DROP, M.LINEUP)
    yield
    P.set_report_match(state[0])
    P.set_ghost_drop(state[1])
    P.set_lineup(state[2])
    M._LU_OVERRIDE = None


def test_defaults_off_and_env_flags(monkeypatch):
    assert av.REPORT_MATCH_DEFAULT is False and P.GHOST_DROP_DEFAULT is False
    assert M.LINEUP_DEFAULT is False
    monkeypatch.setenv(M.LINEUP_ENV, "1")
    assert M._lu_flag() is True
    monkeypatch.setenv(M.LINEUP_ENV, "off")
    assert M._lu_flag() is False
    monkeypatch.setenv(av.REPORT_MATCH_ENV, "on")
    assert av._flag(av.REPORT_MATCH_ENV, False) is True
    monkeypatch.setenv(P.GHOST_DROP_ENV, "yes")
    assert P._env_flag(P.GHOST_DROP_ENV, False) is True


def test_version_tags_are_model_version_only(switches):
    P.set_report_match(False)
    P.set_ghost_drop(False)
    P.set_lineup(False)
    base_vs, base_mv = P._VS, P.MODEL_VERSION
    assert not any(t in base_mv for t in ("+rm", "+gh", "+lu"))
    P.set_report_match(True)
    P.set_ghost_drop(True)
    P.set_lineup(True)
    assert P._VS == base_vs                                    # same bundle
    for t in ("+rm", "+gh", "+lu"):
        assert t in P.MODEL_VERSION
    assert P.MODEL_VERSION.replace("+rm", "").replace("+gh", "").replace("+lu", "") == base_mv
    P.set_report_match(False)
    P.set_ghost_drop(False)
    P.set_lineup(False)
    assert (P._VS, P.MODEL_VERSION) == (base_vs, base_mv)


# --- +rm ------------------------------------------------------------------------------------

def _roster(conn, rows):
    conn.executemany("INSERT INTO team_rosters (season, team_id, team, player_id, player) "
                     "VALUES (?,?,?,?,?)", rows)
    conn.commit()


def test_rm_keeps_a_long_absent_players_out_row(conn, switches):  # noqa: F811
    world(conn)
    report(conn, "2026-03-20T20:00:00Z", [("AAA", "Zeta Longout", "Out", "Injury/Illness - Left Foot; Surgery"),
                                          ("AAA", name("AAA", 3), "Questionable", "Rest")])
    _roster(conn, [("2025-26", 1, "AAA", LONG_OUT, "Zeta Longout")]
            + [("2025-26", 1, "AAA", pid("AAA", i), name("AAA", i)) for i in range(1, 9)])
    P.set_report_match(False)
    off = av.p_play_slate(conn, DAY, AT)
    assert LONG_OUT not in set(off["player_id"])                # OFF: the row is dropped
    P.set_report_match(True)
    on = av.p_play_slate(conn, DAY, AT)
    st, p = status_of(on, "AAA", LONG_OUT - pid("AAA", 0))
    assert st == "out" and p < 0.02
    # every other row is exactly the OFF row (second-stage match only)
    rest = on[on["player_id"] != LONG_OUT].reset_index(drop=True)
    assert rest.equals(off.reset_index(drop=True))


def test_rm_without_a_roster_snapshot_is_bit_identical(conn, switches):  # noqa: F811
    world(conn)
    report(conn, "2026-03-20T20:00:00Z", [("AAA", "Zeta Longout", "Out", "Rest"),
                                          ("BBB", name("BBB", 2), "Probable", "Rest")])
    P.set_report_match(False)
    off = av.p_play_slate(conn, DAY, AT)
    P.set_report_match(True)
    on = av.p_play_slate(conn, DAY, AT)
    assert on.equals(off)


def test_rm_off_never_reads_the_roster_table(conn, switches, monkeypatch):  # noqa: F811
    world(conn)
    P.set_report_match(False)

    def boom(*a, **k):
        raise AssertionError("current_roster_extra must not run when +rm is off")

    monkeypatch.setattr(av, "current_roster_extra", boom)
    av.p_play_slate(conn, DAY, AT)


def test_current_roster_extra_lists_only_players_missing_from_the_box_roster(conn):  # noqa: F811
    _roster(conn, [("2025-26", 1, "AAA", 1, "One"), ("2025-26", 1, "AAA", 2, "Two"),
                   ("2024-25", 1, "AAA", 3, "Old Season")])
    out = av.current_roster_extra(conn, DAY, {"AAA": [(1, "One")]})
    assert out == {"AAA": [(2, "Two")]}
    assert av.current_roster_extra(conn, DAY.replace(year=2030), {}) == {}


# --- +gh ------------------------------------------------------------------------------------

def test_gh_strict_drops_unforced_candidates_on_no_current_roster():
    panel = pd.DataFrame({"player_id": [1, 2, 3, 4], "team": ["AAA"] * 4, "key": ["g|AAA"] * 4})
    current = {1: "AAA", 2: "BBB"}                      # 3, 4: on no current roster
    extras = {("g", "AAA"): [4]}                        # 4 is in the DK pool
    loose = P.drop_moved_players(panel, current, extras)
    strict = P.drop_moved_players(panel, current, extras, strict=True)
    assert loose["player_id"].tolist() == [1, 3, 4]
    assert strict["player_id"].tolist() == [1, 4]
    assert P.drop_moved_players(panel, {}, extras, strict=True).equals(panel)   # no snapshot


# --- +lu ------------------------------------------------------------------------------------

def _team(n=10):
    base = np.array([34, 32, 30, 29, 27, 22, 18, 14, 9, 5], float)[:n]
    ew = base.copy()
    st10 = np.array([10, 10, 10, 10, 10, 0, 0, 0, 0, 0], float)[:n]
    g1 = np.array([0, 0, 1, 1, 2, 0, 1, 2, 0, 1])[:n]
    p = np.full(n, 0.95)
    absent = np.zeros(n, bool)
    absent[1] = True                                    # an expected starter is out
    starters = np.zeros(n, bool)
    starters[[0, 2, 4, 5, 6]] = True                    # 3 (expected) benched; 5, 6 start
    tgt = np.tile([32.0, 31.0, 28.0], (n, 1))
    return base, p, absent, g1, g1.copy(), st10, ew, tgt, starters


def test_lu_weights_table_and_fallbacks(monkeypatch):
    M._lu_table.cache_clear()
    monkeypatch.setattr(M, "_lu_table", lambda: {"2024-25": {"w_demote": 0.8, "w_target": 0.6}})
    assert M.lineup_weights("2024-25") == (0.8, 0.6)
    assert M.lineup_weights("2026-27") == (0.8, 0.6)       # newest earlier season
    assert M.lineup_weights("2023-24") == M.LU_OFF         # nothing at or before S
    M._LU_OVERRIDE = (0.3, 0.9)
    try:
        assert M.lineup_weights("2023-24") == (0.3, 0.9)
    finally:
        M._LU_OVERRIDE = None


def test_lu_at_the_off_weights_is_bit_identical_and_moves_the_right_rows(switches):
    prm = M.default_params()
    base, p, absent, g1, g2, st10, ew, tgt, starters = _team()
    args = (base, p, absent, g1, g2, st10, ew, tgt, prm)
    P.set_lineup(False)
    off = M.project_arrays(*args, starters=starters)
    P.set_lineup(True)
    M._LU_OVERRIDE = M.LU_OFF
    same = M.project_arrays(*args, starters=starters)
    assert np.array_equal(off, same)
    M._LU_OVERRIDE = (1.0, 1.0)
    on = M.project_arrays(*args, starters=starters)
    assert on[3] < off[3]                               # demoted expected starter: fewer minutes
    assert on[5] > off[5] or on[6] > off[6]             # unexpected starters: more
    # without a lineup (lock-45) the switch changes nothing
    lock45_on = M.project_arrays(*args)
    P.set_lineup(False)
    assert np.array_equal(lock45_on, M.project_arrays(*args))


def test_lu_confirmed_starter_off_the_report_gets_099(switches):
    panel = pd.DataFrame({"key": ["g|AAA"] * 3, "player_id": [11, 12, 13], "roster": [1, 1, 1]})
    P.set_lineup(False)
    assert P.pplay_v2_fix(None, panel, {}, set(), starters={11, 12}) == {}
    P.set_lineup(True)
    out = P.pplay_v2_fix(None, panel, {12: 0.6}, {12}, starters={11, 12})
    assert out == {("g|AAA", 11): M.LU_P_STARTER, ("g|AAA", 12): 1.0}


def test_rm_reads_only_reports_published_by_at(conn, switches):  # noqa: F811
    """Leak check: with +rm on, a report (and the long-absent player's row in it) published
    after `at` changes nothing, bit for bit."""
    world(conn)
    _roster(conn, [("2025-26", 1, "AAA", LONG_OUT, "Zeta Longout")])
    report(conn, "2026-03-20T20:00:00Z", [("AAA", name("AAA", 3), "Questionable", "Rest")])
    P.set_report_match(True)
    before = av.p_play_slate(conn, DAY, AT)
    report(conn, "2026-03-20T22:30:00Z", [("AAA", "Zeta Longout", "Out", "Rest"),
                                          ("AAA", name("AAA", 3), "Out", "Rest")])
    after = av.p_play_slate(conn, DAY, AT)
    assert before.equals(after)
    assert LONG_OUT not in set(after["player_id"])
    late = av.p_play_slate(conn, DAY, "2026-03-20T23:00:00Z")
    assert status_of(late, "AAA", LONG_OUT - pid("AAA", 0))[0] == "out"
