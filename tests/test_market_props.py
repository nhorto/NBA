"""F9 market prop lines (proj/features/market_props.py): de-vig maths, the strictly pre-cutoff
as-of join, missing lines, several books, the live loader, and the group being off by
default. Synthetic DB only: nothing touches the network or var/."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import context as CX
from courtside.proj import features as FX
from courtside.proj.features import market_props as MP

DAY = "2026-10-20"
G1, G2 = "0022600901", "0022600902"
TIP1, TIP2 = "2026-10-20T23:00:00Z", "2026-10-21T01:30:00Z"
# player_id: (team, name, game)
PLAYERS = {101: ("AAA", "Ann Alder", G1), 102: ("AAA", "Ben Bench", G1),
           201: ("BBB", "Bo Birch", G1), 301: ("CCC", "Di Dogwood", G2)}


def _snap(conn, gid, cap, aid, name, market, line, o, u, provider="100", kind="ou",
          last_updated=None, open_line=None):
    conn.execute(
        "INSERT INTO prop_snapshots (game_id, captured_at, provider, provider_name, "
        "espn_athlete_id, player_name, subject, market_id, market, kind, line, over_odds, "
        "under_odds, open_line, last_updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, cap, provider, "Book" + provider, aid, name, "player", market, market, kind, line,
         o, u, open_line, last_updated))


@pytest.fixture
def conn():
    c = db.connect()
    for gid, tip, home, away in ((G1, TIP1, "BBB", "AAA"), (G2, TIP2, "DDD", "CCC")):
        c.execute("INSERT INTO games (game_id, nba_day, tipoff_utc, home, away, status) "
                  "VALUES (?,?,?,?,?, 'scheduled')", (gid, DAY, tip, home, away))
    ids = {"AAA": 1, "BBB": 2, "CCC": 3}
    for pid, (team, name, _) in PLAYERS.items():
        c.execute("INSERT INTO team_rosters (season, team_id, team, player_id, player) "
                  "VALUES ('2026-27', ?, ?, ?, ?)", (ids[team], team, pid, name))
    # Alder points: 19.5 early, 20.5 at 20:00Z (lock-45 = 22:15Z sees this), 30.5 at 22:30Z
    # (after lock-45, before tip), 40.5 after tip
    _snap(c, G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Total Points", 19.5, -115, -105,
          open_line=19.5)
    _snap(c, G1, "2026-10-20T20:00:00Z", "a101", "Ann Alder", "Total Points", 20.5, -110, -110,
          last_updated="2026-10-21T03:00Z", open_line=19.5)    # ESPN stamp after tip: ignored
    _snap(c, G1, "2026-10-20T22:30:00Z", "a101", "Ann Alder", "Total Points", 30.5, -110, -110)
    _snap(c, G1, "2026-10-20T23:10:00Z", "a101", "Ann Alder", "Total Points", 40.5, -110, -110)
    _snap(c, G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Total Rebounds", 5.0, -105, -115)
    _snap(c, G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Points Milestones", 25.0, 150,
          None, kind="milestone")                                # ignored
    # Birch: two books, and a market first seen only after the cutoff
    _snap(c, G1, "2026-10-20T14:00:00Z", "a201", "Bo Birch", "Total Assists", 6.5, -110, -110)
    _snap(c, G1, "2026-10-20T14:00:00Z", "a201", "Bo Birch", "Total Assists", 7.5, 120, -150,
          provider="200")
    _snap(c, G1, "2026-10-20T22:40:00Z", "a201", "Bo Birch", "Total Points", 18.5, -110, -110)
    # G2
    _snap(c, G2, "2026-10-20T14:00:00Z", "a301", "Di Dogwood", "Total 3-Point Field Goals", 2.5,
          130, -160)
    c.commit()
    return c


# ------------------------------------------------------------------ de-vig maths

def test_devig_is_proportional_and_sums_to_one():
    p, hold = MP.devig(-110, -110)
    assert p == pytest.approx(0.5) and hold == pytest.approx(2 * 110 / 210 - 1)
    po, _ = MP.devig(-140, 115)
    pu, _ = MP.devig(115, -140)
    assert po + pu == pytest.approx(1.0)
    assert MP.devig(-110, None) == (None, None)


def test_implied_mean_round_trips_the_no_vig_chance():
    for stat, line, p in (("pts", 20.5, 0.5), ("pts", 20.5, 0.58), ("reb", 7.5, 0.42),
                          ("ast", 2.5, 0.37), ("fg3m", 0.5, 0.66), ("pra", 33.5, 0.52)):
        mu = MP.implied_mean(stat, line, p)
        assert mu is not None and mu > 0
        # the chance above the line implied by that mean is p again
        j = int(np.searchsorted(MP.MU_GRID, mu))
        curve = MP._comparable_curve(stat, line)
        assert curve[j - 1] - 1e-9 <= p <= curve[j] + 1e-9


def test_implied_mean_monotone_and_fair_line_at_even_money():
    mus = [MP.implied_mean("pts", 20.5, p) for p in (0.3, 0.4, 0.5, 0.6, 0.7)]
    assert all(b > a for a, b in zip(mus, mus[1:]))
    # 50/50 on a half line: the fair (median) line is the line itself
    assert MP.fair_line("pts", MP.implied_mean("pts", 20.5, 0.5)) == pytest.approx(20.5, abs=1e-6)
    assert MP.fair_line("reb", MP.implied_mean("reb", 6.5, 0.5)) == pytest.approx(6.5, abs=1e-6)
    # right-skewed counts: mean above the median
    assert MP.implied_mean("pts", 20.5, 0.5) > 20.5
    # over favoured -> fair line above the posted one
    assert MP.fair_line("pts", MP.implied_mean("pts", 20.5, 0.6)) > 20.5


def test_integer_line_uses_the_push_refunded_chance():
    mu = MP.implied_mean("reb", 5.0, 0.5)
    pmf_cdf = MP._comparable_curve("reb", 5.0)
    assert mu is not None and 4.5 < mu < 6.5
    assert np.all(np.diff(pmf_cdf) >= 0)


def test_implied_mean_rejects_bad_inputs():
    assert MP.implied_mean("pts", 20.5, None) is None
    assert MP.implied_mean("pts", None, 0.5) is None
    assert MP.implied_mean("pts", 20.5, 1.0) is None
    assert MP.implied_mean("pts", 20.5, 0.0) is None
    assert MP.implied_mean("nope", 20.5, 0.5) is None
    assert MP.fair_line("pts", None) is None


# ------------------------------------------------------------------ as-of join

def test_cutoffs_lock45_lock10_and_slate_anchor(conn):
    c45 = MP.cutoffs_for(conn, [G1, G2], "lock45")
    assert c45 == {G1: "2026-10-20T22:15:00Z", G2: "2026-10-21T00:45:00Z"}
    assert MP.cutoffs_for(conn, [G1], "lock10")[G1] == "2026-10-20T22:50:00Z"
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, tag, game_ids_json, lock_utc) "
                 f"VALUES ('S1', '{DAY}', 'main', '[\"{G1}\", \"{G2}\"]', '{TIP1}')")
    assert MP.cutoffs_for(conn, [G2], "lock45", anchor="slate")[G2] == "2026-10-20T22:15:00Z"


def test_asof_never_uses_post_cutoff_snapshots(conn):
    snaps = MP.load_snapshots(conn)
    assert "milestone" not in set(snaps["market"].str.lower())      # o/u only
    cut = MP.cutoffs_for(conn, [G1, G2], "lock45")
    a = MP.asof(snaps, cut)
    assert (a["captured_at"] <= a["cutoff"]).all()
    alder = a[(a["espn_athlete_id"] == "a101") & (a["stat"] == "pts")].iloc[0]
    assert alder["line"] == 20.5                  # not 30.5 (after lock-45) or 40.5 (after tip)
    assert alder["first_line"] == 19.5 and alder["n_changes"] == 2
    assert alder["age"] == pytest.approx(135.0)  # 20:00 -> 22:15
    # Birch's points market appears only after the cutoff: absent
    assert not ((a["espn_athlete_id"] == "a201") & (a["stat"] == "pts")).any()
    # at lock-10 the 22:30 move is visible, the post-tip one still is not
    a10 = MP.asof(snaps, MP.cutoffs_for(conn, [G1], "lock10"))
    assert a10[(a10["espn_athlete_id"] == "a101") & (a10["stat"] == "pts")]["line"].item() == 30.5
    # a snapshot cut off entirely: no rows at all
    assert len(MP.asof(snaps, {G1: "2026-10-20T13:00:00Z"})) == 0


def test_asof_result_unchanged_by_adding_future_snapshots(conn):
    cut = MP.cutoffs_for(conn, [G1, G2], "lock45")
    keys = pd.DataFrame({"game_id": [G1, G1, G1, G2], "player_id": [101, 102, 201, 301]})
    before = MP.lines_asof(conn, [G1, G2], cut, keys=keys)
    for gid in (G1, G2):
        _snap(conn, gid, "2026-10-20T22:59:00Z", "a101", "Ann Alder", "Total Points", 99.5,
              -500, 300)
        _snap(conn, gid, "2026-10-21T02:00:00Z", "a301", "Di Dogwood",
              "Total 3-Point Field Goals", 9.5, -110, -110)
    _snap(conn, G1, "2026-10-20T22:20:00Z", "a102", "Ben Bench", "Total Points", 4.5, -110,
          -110)
    after = MP.lines_asof(conn, [G1, G2], cut, keys=keys)
    pd.testing.assert_frame_equal(before, after)


def test_lines_asof_wide_with_missing_lines_and_two_books(conn):
    cut = MP.cutoffs_for(conn, [G1, G2], "lock45")
    keys = pd.DataFrame({"game_id": [G1, G1, G1, G2], "player_id": [101, 102, 201, 301]})
    w = MP.lines_asof(conn, [G1, G2], cut, keys=keys).set_index("player_id")
    assert list(w.index) == [101, 102, 201, 301]
    # Alder: points and rebounds
    assert w.loc[101, "f9_has_pts"] == 1 and w.loc[101, "f9_line_pts"] == 20.5
    assert w.loc[101, "f9_pover_pts"] == pytest.approx(0.5)
    assert w.loc[101, "f9_fair_pts"] == pytest.approx(20.5, abs=1e-6)
    assert w.loc[101, "f9_mean_pts"] == pytest.approx(MP.implied_mean("pts", 20.5, 0.5))
    assert w.loc[101, "f9_move_pts"] == pytest.approx(1.0)
    assert w.loc[101, "f9_open_pts"] == 19.5
    assert w.loc[101, "f9_nmkts"] == 2 and w.loc[101, "f9_has_any"] == 1
    # bench player without any line: flags 0, numbers NaN
    assert w.loc[102, "f9_has_any"] == 0 and w.loc[102, "f9_nmkts"] == 0
    assert w.loc[102, [f"f9_has_{s}" for s in MP.STATS]].eq(0).all()
    assert np.isnan(w.loc[102, "f9_mean_pts"]) and np.isnan(w.loc[102, "f9_line_ast"])
    # Birch: two books -> median line, mean of the no-vig chances, nbooks 2
    assert w.loc[201, "f9_nbooks_ast"] == 2
    assert w.loc[201, "f9_line_ast"] == pytest.approx(7.0)
    p2, _ = MP.devig(120, -150)
    assert w.loc[201, "f9_pover_ast"] == pytest.approx((0.5 + p2) / 2)
    m1, m2 = MP.implied_mean("ast", 6.5, 0.5), MP.implied_mean("ast", 7.5, p2)
    assert w.loc[201, "f9_mean_ast"] == pytest.approx((m1 + m2) / 2)
    assert w.loc[201, "f9_has_pts"] == 0               # only after the cutoff
    assert w.loc[301, "f9_has_fg3m"] == 1
    # without keys: only players with a line
    assert set(MP.lines_asof(conn, [G1, G2], cut)["player_id"]) == {101, 201, 301}


def test_unmatched_names_are_dropped_not_guessed(conn):
    _snap(conn, G1, "2026-10-20T14:00:00Z", "a999", "Qq Nobody", "Total Points", 5.5, -110, -110)
    w = MP.lines_asof(conn, [G1], MP.cutoffs_for(conn, [G1], "lock45"))
    assert set(w["player_id"]) == {101, 201}


def test_live_lines_cap_at_tip(conn):
    # a run 'at' after G1's tip: G1 is capped at the tip (22:30 visible, 23:10 not)
    w = MP.live_lines(conn, [G1, G2], at="2026-10-20T23:30:00Z").set_index("player_id")
    assert w.loc[101, "f9_line_pts"] == 30.5
    assert w.loc[201, "f9_has_pts"] == 1               # 22:40 < tip
    # a run at 21:00: Alder at 20.5
    w = MP.live_lines(conn, [G1], at="2026-10-20T21:00:00Z").set_index("player_id")
    assert w.loc[101, "f9_line_pts"] == 20.5 and w.loc[101, "f9_age_pts"] == pytest.approx(60.0)


# ------------------------------------------------------------------ the group: off by default

def _frame():
    rows = []
    for pid, (team, _, gid) in PLAYERS.items():
        r = {"game_id": gid, "player_id": pid, "team": team, "mpg": 30.0}
        r.update({f"r_{s}": v for s, v in (("pts", 0.6), ("fg3m", 0.07), ("reb", 0.2),
                                            ("ast", 0.15), ("stl", 0.03), ("blk", 0.02),
                                            ("tov", 0.05))})
        rows.append(r)
    return pd.DataFrame(rows)


def test_group_registered_but_in_no_live_set():
    assert "F9" in FX.GROUPS and "F9" in FX.MARKET_GROUPS
    assert "F9" not in FX.EXTENSION_GROUPS
    for live in (CX.LIVE_GROUPS, CX.LIVE_GROUPS_EXT):
        assert "F9" not in live
        for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov"):
            assert not any(c.startswith("f9_") for c in FX.model_columns(s, live))
    # the F9 columns exist for every DK stat (so a fit with F9 alone is well defined)
    for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov"):
        assert FX.model_columns(s, ["F9"])
    assert FX.monotone_for("pts", ["F9"]) == {"f9_lr_pts": 1}


def test_off_by_default_changes_nothing(conn):
    fr = _frame()
    # extend_frame without F9: the very same frame
    assert FX.extend_frame(conn, fr, []) is fr
    # attach without F9 columns adds no f9 column
    tg = pd.DataFrame({"game_id": [G1, G1, G2], "team": ["AAA", "BBB", "CCC"]})
    fr2 = fr.assign(grp=1, g_eff=5.0, r_dk=1.0, p_dk=1.0,
                    **{f"{p}_{s}": 0.1 for s in ("oreb", "dreb", "fga", "fta")
                       for p in ("r", "p", "n")},
                    **{f"{p}_{s}": 0.1 for s in ("pts", "fg3m", "reb", "ast", "stl", "blk",
                                                 "tov") for p in ("p", "n")})
    out = FX.attach(fr2, tg)
    assert not any(c.startswith("f9_") for c in out.columns)


def test_extend_frame_adds_f9_keeps_rows_and_row_features(conn):
    fr = _frame().iloc[::-1].reset_index(drop=True)          # odd order: must be kept
    out = FX.extend_frame(conn, fr, ["F9"])
    assert len(out) == len(fr)
    assert list(out["player_id"]) == list(fr["player_id"])
    alder = out[out["player_id"] == 101].iloc[0]
    assert alder["f9_has_pts"] == 1
    assert alder["f9_lr_pts"] == pytest.approx(math.log(alder["f9_mean_pts"] / (0.6 * 30.0)))
    bench = out[out["player_id"] == 102].iloc[0]
    assert bench["f9_has_any"] == 0 and np.isnan(bench["f9_lr_pts"])
    assert np.isnan(bench["f9_lr_pra"])
    for s in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov"):
        assert all(c in out.columns for c in FX.model_columns(s, ["F9"]))


def test_extend_frame_with_no_props_is_all_missing():
    c = db.connect()
    c.execute("INSERT INTO games (game_id, nba_day, tipoff_utc, home, away, status) "
              f"VALUES ('{G1}', '{DAY}', '{TIP1}', 'BBB', 'AAA', 'final')")
    out = FX.extend_frame(c, _frame(), ["F9"])
    assert out["f9_has_any"].eq(0).all() and out["f9_lr_pts"].isna().all()


# ------------------------------------------------------------------ readiness

def test_readiness_counts_only_graded_played_rows(conn):
    r = MP.readiness(conn)
    assert r["markets"]["pts"]["lined"] == 1                  # Alder (Birch after cutoff)
    assert r["markets"]["pts"]["graded"] == 0                 # nothing final yet
    conn.execute("UPDATE games SET status='final' WHERE game_id=?", (G1,))
    r = MP.readiness(conn)
    assert r["markets"]["pts"]["graded"] == 0                 # final but no box score yet
    conn.execute("INSERT INTO player_games (game_id, player_id, team, season, game_date, min, "
                 "pts, reb, ast, fg3m) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (G1, 101, "AAA", "2026-27", DAY, 34, 21, 5, 3, 2))
    r = MP.readiness(conn)
    assert r["markets"]["pts"]["played"] == 1 and r["markets"]["ast"]["void_dnp"] == 1
    assert "Evaluation plan" in MP.format_readiness(r)
