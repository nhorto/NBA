"""Questionable / lock-10 P(play) refit (pq, `availability.PPLAYQ`, default OFF).

docs/research/pplay-q-refit.md; research/pplay_q_refit.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import availability as av

DAY = date(2026, 3, 20)
GAME = "G_TARGET"
TIP = "2026-03-21T00:00:00Z"          # 20:00 ET on DAY
LOCK10 = "2026-03-20T23:50:00Z"
MINUTES = {1: 32, 2: 30, 3: 22, 4: 20, 5: 18, 6: 9, 7: 8, 8: 6}    # 1-2 starter role, 3-5 rotation


def pid(team, i):
    return (1000 if team == "AAA" else 2000) + i


def name(team, i):
    return f"{'Alpha' if team == 'AAA' else 'Bravo'} Player{'abcdefgh'[i - 1]}"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def switch_off():
    av.set_pplayq(False)
    yield
    av.set_pplayq(False)


def add_game(conn, gid, day, played_fn=lambda team, i: True):
    for team, opp, home in (("AAA", "BBB", 1), ("BBB", "AAA", 0)):
        conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, "
                     "is_home) VALUES (?,?,?,?,?,?,?)",
                     (gid, team, "2025-26", "Regular Season", day.isoformat(), opp, home))
        for i, m in MINUTES.items():
            if played_fn(team, i):
                conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, "
                             "season, season_type, game_date, min) VALUES (?,?,?,?,?,?,?,?)",
                             (gid, pid(team, i), name(team, i), team, "2025-26", "Regular Season",
                              day.isoformat(), float(m)))
    conn.commit()


def world(conn, n=12, played_fn=lambda k, team, i: True):
    for k in range(n):
        add_game(conn, f"H{k:02d}", DAY - timedelta(days=n - k + 1),
                 lambda team, i, k=k: played_fn(k, team, i))
    add_game(conn, GAME, DAY)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?,?,?,?,?,?,?)", (GAME, "2025-26", DAY.isoformat(), TIP, "AAA", "BBB",
                                            "final"))
    conn.commit()


def report(conn, at, rows):
    conn.executemany(
        "INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
        "VALUES (?,?,?,?,?,?)", [(at, "nba_official", t, n, s, r) for t, n, s, r in rows])
    conn.commit()


def p_of(df, team, i):
    return float(df.loc[df["player_id"] == pid(team, i), "p_play"].iloc[0])


def lock10_report(conn):
    report(conn, "2026-03-20T23:00:00Z", [
        ("AAA", name("AAA", 1), "Questionable", "Injury/Illness - Left Ankle; Sprain"),
        ("AAA", name("AAA", 7), "Available", "Injury/Illness - Left Knee; Soreness"),
        ("BBB", name("BBB", 1), "Questionable", "Injury/Illness - Left Ankle; Sprain"),
    ])


AAA_STARTERS = [pid("AAA", i) for i in (2, 3, 4, 5, 6)]      # AAA's lineup is out, 1 is not in it


# --- switch ----------------------------------------------------------------------------------

def _flag_in_subprocess(value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "COURTSIDE_PPLAYQ"}
    if value is not None:
        env["COURTSIDE_PPLAYQ"] = value
    out = subprocess.run([sys.executable, "-c", "from courtside.proj import availability as a;"
                          "print(a.PPLAYQ, a.PPLAYQ_DEFAULT)"], env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def test_switch_defaults_on_and_reads_the_environment():
    assert av.PPLAYQ_DEFAULT is True                     # shipped 2026-10-24
    assert av.PPLAYQ_TAG == "+pq"
    assert _flag_in_subprocess(None) == "True True"
    assert _flag_in_subprocess("1") == "True True"
    assert _flag_in_subprocess("0") == "False True"


def test_off_never_calls_the_refit(conn, monkeypatch):
    world(conn)
    lock10_report(conn)

    def boom(*a, **k):
        raise AssertionError("q_refit_frame called with the switch off")

    monkeypatch.setattr(av, "q_refit_frame", boom)
    df = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    assert len(df) and p_of(df, "AAA", 2) == 0.99


def test_on_with_no_offsets_is_bit_identical_to_off(conn, monkeypatch):
    world(conn)
    lock10_report(conn)
    off = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    av.set_pplayq(True)
    monkeypatch.setattr(av, "load_q", lambda path=None: av.QRefitParams())
    on = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    assert off.equals(on)


# --- the refit on a slate ----------------------------------------------------------------------

def test_lock10_non_starters_drop_only_where_lineups_are_known(conn):
    world(conn)
    lock10_report(conn)
    off = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    av.set_pplayq(True)
    on = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    prm = av.load_q()
    # AAA: lineup out (5 flagged); Questionable starter-role player not in it -> much lower
    assert p_of(on, "AAA", 1) < p_of(off, "AAA", 1) - 0.15
    z = np.log(p_of(off, "AAA", 1) / (1 - p_of(off, "AAA", 1))) + prm.offsets["ns|q|starter"]
    assert p_of(on, "AAA", 1) == pytest.approx(1 / (1 + np.exp(-z)), abs=1e-12)
    # Available bench non-starter -> lower by the pa|bench offset
    assert p_of(on, "AAA", 7) < p_of(off, "AAA", 7)
    # flagged starters stay at 0.99; unlisted players and the other team are untouched
    for i in (2, 3, 4, 5, 6):
        assert p_of(on, "AAA", i) == p_of(off, "AAA", i) == 0.99
    assert p_of(on, "AAA", 8) == p_of(off, "AAA", 8)
    bbb_on, bbb_off = on[on["team"] == "BBB"], off[off["team"] == "BBB"]
    assert bbb_on.sort_values("player_id")["p_play"].tolist() == \
        bbb_off.sort_values("player_id")["p_play"].tolist()


def test_no_starters_given_means_no_lineup_refit(conn):
    world(conn)
    lock10_report(conn)
    off = av.p_play_slate(conn, DAY, LOCK10)
    av.set_pplayq(True)
    on = av.p_play_slate(conn, DAY, LOCK10)
    assert off.equals(on)          # shipped params carry no "ret" offsets


def test_a_partial_flag_set_does_not_count_as_a_lineup(conn):
    world(conn)
    lock10_report(conn)
    two = AAA_STARTERS[:av.Q_LINEUP_MIN - 1]
    off = av.p_play_slate(conn, DAY, LOCK10, starters=two)
    av.set_pplayq(True)
    on = av.p_play_slate(conn, DAY, LOCK10, starters=two)
    assert p_of(on, "AAA", 1) == p_of(off, "AAA", 1)


def test_returning_split_applies_only_without_lineups(conn, monkeypatch):
    # AAA player 1 missed the last 4 team games before DAY
    world(conn, played_fn=lambda k, team, i: not (team == "AAA" and i == 1 and k >= 8))
    lock10_report(conn)
    prm = av.QRefitParams(offsets={"ret|q|starter": 0.8}, ret_n=3)
    monkeypatch.setattr(av, "load_q", lambda path=None: prm)
    off = av.p_play_slate(conn, DAY, LOCK10)
    av.set_pplayq(True)
    on = av.p_play_slate(conn, DAY, LOCK10)
    assert p_of(on, "AAA", 1) > p_of(off, "AAA", 1)
    assert p_of(on, "BBB", 1) == p_of(off, "BBB", 1)       # Questionable but not returning
    known = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    av.set_pplayq(False)
    known_off = av.p_play_slate(conn, DAY, LOCK10, starters=AAA_STARTERS)
    assert p_of(known, "AAA", 1) == p_of(known_off, "AAA", 1)   # lineup out: no ns offset here


# --- pieces --------------------------------------------------------------------------------------

@pytest.mark.parametrize("args,group", [
    (("questionable", "starter", False, True, 0), "ns|q|starter"),
    (("questionable", "bench", True, True, 0), None),               # flagged starter: 0.99 path
    (("available", "bench", False, True, 0), "ns|pa|bench"),
    (("probable", "rotation", False, True, 0), "ns|pa|rotation"),
    (("removed", "starter", False, True, 0), "ns|removed"),
    (("out", "starter", False, True, 0), None),
    (("doubtful", "starter", False, True, 0), None),
    (("unlisted", "starter", False, True, 0), None),
    (("questionable", "starter", False, False, 3), "ret|q|starter"),
    (("questionable", "starter", False, False, 2), None),
    (("questionable", "starter", False, False, -1), None),          # never appeared
    (("available", "starter", False, False, 9), None),
])
def test_group_rules(args, group):
    assert av.q_group(*args, ret_n=3) == group


def test_apply_keeps_ungrouped_rows_exactly_and_clamps():
    p = np.array([0.123456789, 0.5, 0.999, 0.5])
    prm = av.QRefitParams(offsets={"a": 1.0, "b": -50.0})
    out = av.q_apply(p, [None, "a", "zzz", "b"], prm)
    assert out[0] == p[0] and out[2] == p[2]                     # no logit round trip
    assert out[1] == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert out[3] == av.CLAMP[0]


def test_offset_fit_recovers_a_known_shift_and_shrinks_with_little_data():
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.3, 0.8, 20_000)
    z = np.log(p0 / (1 - p0)) - 1.0
    y = rng.random(len(p0)) < 1 / (1 + np.exp(-z))
    offs, ns = av.fit_q_offsets(p0, y, ["g"] * len(p0))
    assert offs["g"] == pytest.approx(-1.0, abs=0.06) and ns["g"] == len(p0)
    small, _ = av.fit_q_offsets(p0[:6], np.zeros(6), ["g"] * 6, ridge=4.0)
    loose, _ = av.fit_q_offsets(p0[:6], np.zeros(6), ["g"] * 6, ridge=0.1)
    assert loose["g"] < small["g"] < 0                            # ridge pulls toward the cell


def test_games_missed_counts_team_games_since_the_last_appearance(conn):
    world(conn, played_fn=lambda k, team, i: not (team == "AAA" and i == 1 and k >= 8))
    h = av.load_history(conn, DAY)
    assert av.games_missed(h, pid("AAA", 1), "AAA", DAY) == 4
    assert av.games_missed(h, pid("AAA", 2), "AAA", DAY) == 0
    assert av.games_missed(h, 999999, "AAA", DAY) == -1
    # strictly before the day: his appearance ON the day (he plays the target game) only
    # counts from the next day on
    assert av.games_missed(h, pid("AAA", 1), "AAA", DAY + timedelta(days=1)) == 0


def test_shipped_params_are_the_lineup_family_only():
    d = json.loads(av.PQ_PATH.read_text())
    prm = av.load_q()
    assert prm.meta["families"] == ["ns"] == d["meta"]["families"]
    assert prm.offsets and all(k.startswith("ns|") for k in prm.offsets)
    assert set(prm.offsets) == set(prm.n) and all(v > 0 for v in prm.n.values())
    assert prm.offsets["ns|q|starter"] < -1.0                     # 0.63 vs 0.14 played
    assert prm.offsets["ns|pa|bench"] < 0
    assert abs(prm.offsets["ns|q|rotation"]) < 0.2
    assert av.QRefitParams.from_json(prm.to_json()).offsets == pytest.approx(prm.offsets)


def test_refit_frame_vector_matches_the_rule_table():
    h = av.History(pd.DataFrame(columns=["game_id", "player_id", "team", "game_date", "min"]),
                   pd.DataFrame(columns=["game_id", "team", "game_date"]))
    df = pd.DataFrame({"player_id": [1, 2, 3, 4, 5, 6, 7], "team": ["A"] * 5 + ["B"] * 2,
                       "status": ["questionable", "unlisted", "unlisted", "unlisted",
                                  "available", "questionable", "available"],
                       "role": ["starter", "starter", "rotation", "rotation", "bench",
                                "starter", "bench"],
                       "p_play": [0.6, 0.97, 0.97, 0.97, 0.7, 0.6, 0.7]})
    prm = av.QRefitParams(offsets={"ns|q|starter": -1.0, "ns|pa|bench": -0.5})
    out = av.q_refit_frame(df, h, DAY, {2, 3, 4}, prm)
    assert out[0] < 0.6 and out[4] < 0.7                         # team A lineup known
    assert out[5] == 0.6 and out[6] == 0.7                       # team B: no flags
    assert list(out[1:4]) == [0.97, 0.97, 0.97]


def test_pipeline_version_tag_follows_the_switch():
    from courtside.proj import availability as AV, pipeline as P
    was = AV.PPLAYQ
    try:
        P.set_pplayq(False)
        off = P.MODEL_VERSION
        assert not off.endswith(AV.PPLAYQ_TAG)
        vs_off = P._VS
        P.set_pplayq(True)
        assert P.MODEL_VERSION == off + AV.PPLAYQ_TAG and P._VS == vs_off   # same bundle
    finally:
        P.set_pplayq(was)
