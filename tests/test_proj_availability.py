"""P(play) recalibration (T1.3): cells, shrinkage, reason parsing, leakage, determinism."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from courtside import db, injuries
from courtside.proj import availability as av

DAY = date(2026, 3, 20)
GAME = "G_TARGET"
TIP = "2026-03-21T00:00:00Z"          # 20:00 ET on DAY
TEAMS = ("AAA", "BBB")


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def pid(team, i):
    return (1000 if team == "AAA" else 2000) + i


def name(team, i):
    return f"{'Alpha' if team == 'AAA' else 'Bravo'} Player{'abcdefgh'[i - 1]}"


MINUTES = {1: 32, 2: 30, 3: 22, 4: 20, 5: 18, 6: 9, 7: 8, 8: 6}      # 1-2 starters, 3-5 rotation


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


def add_history(conn, n=12):
    for k in range(n):
        add_game(conn, f"H{k:02d}", DAY - timedelta(days=n - k + 1))


def add_target(conn, played_fn=lambda team, i: True):
    add_game(conn, GAME, DAY, played_fn)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?,?,?,?,?,?,?)", (GAME, "2025-26", DAY.isoformat(), TIP, "AAA", "BBB",
                                            "final"))
    conn.commit()


def report(conn, at, rows, source="nba_official"):
    conn.executemany(
        "INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
        "VALUES (?,?,?,?,?,?)", [(at, source, t, n, s, r) for t, n, s, r in rows])
    conn.commit()


def status_of(df, team, i):
    r = df[df["player_id"] == pid(team, i)]
    return r["status"].iloc[0], float(r["p_play"].iloc[0])


# --- reason classes ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,cls", [
    ("Injury/Illness - Left Ankle; Sprain", "injury"),
    ("Injury/Illness - Right Knee; Injury Management", "management"),
    ("Injury/Illness - Illness; Illness", "illness"),
    ("Injury/Illness - N/A; Illness", "illness"),
    ("Rest", "rest"),
    ("Personal Reasons", "personal"),
    ("Not With Team", "personal"),
    ("G League - Two-Way", "gleague"),
    ("G League - On Assignment", "gleague"),
    ("League Suspension", "suspension"),
    ("Concussion; Protocol", "injury"),
    ("Trade Pending", "other"),
    ("", "none"),
    (None, "none"),
])
def test_reason_classification(text, cls):
    assert av.reason_class(text) == cls


def test_status_roles_buckets():
    assert av.canon_status("Game Time Decision") == "questionable"
    assert av.canon_status("Day-To-Day") == "daytoday"
    assert av.canon_status("OUT") == "out"
    assert [av.hours_bucket(h) for h in (0.1, 0.75, 1.9, 2, 5.9, 6, 30)] == \
        ["lt45m", "45m_2h", "45m_2h", "2h_6h", "2h_6h", "gt6h", "gt6h"]
    assert av.hours_bucket(None) == av.NOBUCKET
    assert [av.role_of(x) for x in (None, 5, 14.9, 15, 25.9, 26, 36)] == \
        ["bench", "bench", "bench", "rotation", "rotation", "starter", "starter"]


# --- Beta shrinkage ---------------------------------------------------------------------------

def test_sparse_cell_shrinks_to_its_parent_and_dense_cell_does_not():
    # status level: 1000 Questionable games, 300 played -> parent about 0.30
    cells = {"questionable": [1000, 300]}
    sparse = dict(cells, **{"questionable|gt6h": [2, 2]})           # 2 of 2: would be 1.0 raw
    dense = dict(cells, **{"questionable|gt6h": [2000, 2000]})
    p_sparse = av.AvailabilityParams(cells=sparse).chain("questionable", "gt6h", "injury", "bench")
    p_dense = av.AvailabilityParams(cells=dense).chain("questionable", "gt6h", "injury", "bench")
    parent = av.AvailabilityParams(cells=cells).chain("questionable", "gt6h", "injury", "bench")
    assert 0.29 < parent[0][0] < 0.31
    assert abs(p_sparse[1][0] - parent[1][0]) < 0.09          # n=2 barely moves it
    assert p_sparse[1][0] > parent[1][0]                      # ... but in the right direction
    assert p_dense[1][0] > 0.95                               # n=2000 dominates its parent
    # an empty cell is exactly its parent
    assert parent[3][0] == parent[2][0] == parent[1][0] == parent[0][0] or \
        abs(parent[3][0] - parent[0][0]) < 1e-9


def test_prior_is_the_hand_set_map_with_20_pseudo_observations():
    empty = av.AvailabilityParams()
    for s, hand in (("doubtful", 0.10), ("questionable", 0.5), ("probable", 0.9)):
        assert empty.chain(s, "gt6h", "injury", "starter")[0][0] == pytest.approx(hand, abs=0.02)
    # 20 pseudo-obs: after 20 observations all-played, Questionable (prior 0.5) is at 0.75
    p = av.AvailabilityParams(cells={"questionable": [20, 20]})
    assert p.chain("questionable", "gt6h", "injury", "bench")[0][0] == pytest.approx(0.75, abs=0.02)


def test_status_table_is_monotone_even_when_the_raw_cells_invert():
    # Available raw 0.60 < Probable raw 0.95 -> pooled, never inverted
    cells = {"available": [500, 300], "probable": [500, 475], "questionable": [500, 250],
             "doubtful": [500, 10], "out": [500, 1]}
    for s in av.LISTED:
        cells[f"{s}|2h_6h"] = list(cells[s])
    p = av.AvailabilityParams(cells=cells)
    vals = [p.p_status_bucket(s, "2h_6h") for s in av.LISTED]
    assert vals == sorted(vals)
    assert vals[3] == pytest.approx(vals[4])                  # pooled


def test_shipped_params_are_monotone_in_every_bucket():
    params = av.load_params()
    assert params.cells, "availability_params.json is missing or empty"
    for b in av.BUCKETS:
        vals = [params.p_status_bucket(s, b) for s in av.LISTED]
        assert vals == sorted(vals), (b, vals)
    for s in av.LISTED:
        for role in av.ROLES:
            assert 0.005 <= params.p(s, "gt6h", "injury", role) <= 0.995
    # the headline recalibrations: Doubtful is not 0.10, a Questionable starter plays > bench
    assert params.p("doubtful", "2h_6h", "injury", "starter") < 0.05
    assert params.p("questionable", "2h_6h", "injury", "starter") > \
        params.p("questionable", "2h_6h", "injury", "bench")
    assert params.p("unlisted", "2h_6h", "none", "starter") > 0.95


def test_params_json_round_trip(tmp_path):
    params = av.load_params()
    path = params.save(tmp_path / "p.json")
    again = av.AvailabilityParams.from_json(json.loads(path.read_text()))
    assert again.cells == {k: list(v) for k, v in params.cells.items()}
    assert again.p("questionable", "gt6h", "injury", "rotation") == \
        params.p("questionable", "gt6h", "injury", "rotation")


# --- slate: uses only reports published before `at` -------------------------------------------

def world(conn):
    add_history(conn)
    add_target(conn)


def test_uses_the_latest_report_at_or_before_at_and_never_a_later_one(conn):
    world(conn)
    n1, n2 = name("AAA", 1), name("AAA", 2)
    report(conn, "2026-03-20T17:00:00Z", [("AAA", n1, "Questionable", "Injury/Illness - Left Knee; Soreness"),
                                          ("AAA", n2, "Probable", "Injury/Illness - Left Foot; Soreness")])
    report(conn, "2026-03-20T23:30:00Z", [("AAA", n1, "Out", "Injury/Illness - Left Knee; Soreness"),
                                          ("AAA", n2, "Available", "")])
    early = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    late = av.p_play_slate(conn, DAY, "2026-03-20T23:45:00Z")
    assert status_of(early, "AAA", 1)[0] == "questionable"
    assert status_of(late, "AAA", 1)[0] == "out"
    assert status_of(early, "AAA", 2)[0] == "probable"
    assert early["report_time"].dropna().unique().tolist() == ["2026-03-20T17:00:00Z"]
    assert late["report_time"].dropna().unique().tolist() == ["2026-03-20T23:30:00Z"]
    # hours-to-tip bucket is the REPORT's age, not the decision's: 17:00Z -> 7 h before 00:00Z
    assert set(early.loc[early["status"] == "questionable", "bucket"]) == {"gt6h"}
    assert set(late.loc[late["status"] == "out", "bucket"]) == {"lt45m"}
    # a report published after `at` changes nothing, bit for bit
    before = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    report(conn, "2026-03-20T23:00:00Z", [("AAA", n1, "Available", ""), ("AAA", n2, "Out", "Rest")])
    after = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    assert before.equals(after)


def test_previous_day_report_is_only_an_old_fallback_never_todays_report(conn):
    world(conn)
    report(conn, "2026-03-19T23:00:00Z", [("AAA", name("AAA", 1), "Out", "Rest")])   # yesterday ET
    df = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    r = df[df["player_id"] == pid("AAA", 1)].iloc[0]
    assert (r["status"], r["source"], r["bucket"]) == ("out", "nba_official", "gt6h")
    assert r["report_time"] == "2026-03-19T23:00:00Z"
    b = df[df["team"] == "BBB"]
    assert set(b["status"]) == {"unlisted_noreport"} and set(b["source"]) == {"prior"}
    # once today's report exists it replaces the stale one for that team
    report(conn, "2026-03-20T20:00:00Z", [("AAA", name("AAA", 2), "Questionable", "Rest")])
    df2 = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    assert status_of(df2, "AAA", 1)[0] == "unlisted"
    assert status_of(df2, "AAA", 2)[0] == "questionable"


def test_unlisted_regulars_high_and_bench_lower_and_noreport_team_flagged(conn):
    world(conn)
    report(conn, "2026-03-20T20:00:00Z", [("AAA", name("AAA", 3), "Out", "Injury/Illness - Right Ankle; Sprain")])
    df = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    assert status_of(df, "AAA", 3)[0] == "out" and status_of(df, "AAA", 3)[1] < 0.02
    assert status_of(df, "AAA", 1) [0] == "unlisted"          # AAA has a report that omits him
    assert status_of(df, "BBB", 1)[0] == "unlisted_noreport"  # BBB has none yet
    p_star = status_of(df, "AAA", 1)[1]
    p_bench = status_of(df, "AAA", 8)[1]
    assert p_star > 0.9 > p_bench
    assert set(df["team"]) == set(TEAMS) and set(df["game_id"]) == {GAME}
    assert list(df.columns[:7]) == ["player_id", "team", "game_id", "status", "p_play", "source",
                                    "report_time"]


def test_espn_status_when_the_team_has_no_official_report(conn):
    world(conn)
    report(conn, "2026-03-20T15:00:00Z", [("BBB", name("BBB", 2), "Out", "Knee")], source="espn")
    df = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    r = df[df["player_id"] == pid("BBB", 2)].iloc[0]
    assert r["source"] == "espn" and r["status"] == "out" and r["p_play"] < 0.02
    assert df[df["team"] == "AAA"]["source"].eq("prior").all()


def test_dk_starter_flag_overrides_to_099(conn):
    world(conn)
    report(conn, "2026-03-20T20:00:00Z", [("AAA", name("AAA", 1), "Questionable", "Rest")])
    base = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z")
    over = av.p_play_slate(conn, DAY, "2026-03-20T22:00:00Z", starters=[pid("AAA", 1)])
    assert status_of(base, "AAA", 1)[1] < 0.8
    assert status_of(over, "AAA", 1)[1] == 0.99
    assert over["override"].sum() == 1


def test_p_play_matches_the_slate_and_is_deterministic(conn):
    world(conn)
    report(conn, "2026-03-20T20:00:00Z", [("AAA", name("AAA", 2), "Questionable", "Injury/Illness - Left Calf; Strain")])
    at = "2026-03-20T22:00:00Z"
    a = av.p_play_slate(conn, DAY, at)
    b = av.p_play_slate(conn, DAY, at)
    assert a.equals(b)
    assert av.p_play(conn, pid("AAA", 2), GAME, at) == pytest.approx(status_of(a, "AAA", 2)[1])
    assert av.p_play(conn, 987654, GAME, at) == av.load_params().p("unlisted", "gt6h", "none", "bench")
    with pytest.raises(KeyError):
        av.p_play(conn, 1, "NOPE", at)
    with pytest.raises(ValueError):
        av.p_play_slate(conn, DAY, __import__("datetime").datetime(2026, 3, 20, 22))   # naive


# --- fit on a synthetic season -----------------------------------------------------------------

def synthetic_season(conn, days=16):
    """AAA: star A1 Out every day (never plays), A2 Questionable every day (plays on even days),
    everyone else unlisted and always plays; BBB: no reports."""
    start = date(2026, 2, 1)
    for k in range(10):                                   # everyone played in January
        add_game(conn, f"J{k:02d}", start - timedelta(days=12 - k))
    for k in range(days):
        d = start + timedelta(days=k)
        gid = f"S{k:02d}"
        add_game(conn, gid, d, lambda team, i, k=k: not (team == "AAA" and (i == 1 or (i == 2 and k % 2))))
        report(conn, f"{d.isoformat()}T20:00:00Z",
               [("AAA", name("AAA", 1), "Out", "Injury/Illness - Left Knee; Surgery"),
                ("AAA", name("AAA", 2), "Questionable", "Injury/Illness - Left Ankle; Sprain")])
    return start, start + timedelta(days=days - 1)


def test_fit_learns_cells_from_reports_vs_who_played(conn):
    lo, hi = synthetic_season(conn)
    params = av.fit(conn, lo, hi)
    assert params.cells["out"][0] > 0 and params.cells["out"][1] == 0      # never played
    n_q, k_q = params.cells["questionable"]
    assert n_q > 0 and 0 < k_q < n_q
    assert params.p("out", "2h_6h", "injury", "starter") < 0.02
    assert 0.3 < params.p("questionable", "2h_6h", "injury", "starter") < 0.7
    assert params.meta["fit_lo"] == lo.isoformat()
    # deterministic: same data -> identical JSON
    again = av.fit(conn, lo, hi)
    assert json.dumps(again.to_json(), sort_keys=True) == json.dumps(params.to_json(), sort_keys=True)


def test_training_rows_use_only_pre_tip_reports_and_label_outcomes(conn):
    lo, hi = synthetic_season(conn, days=6)
    df = av.build_samples(conn, lo, hi, mode="bucket")
    assert (df["hours"].dropna() > 0).all()                    # never a post-tip report
    out = df[df["status"] == "out"]
    assert len(out) and out["played"].sum() == 0
    q = df[df["status"] == "questionable"].sort_values("day")
    assert set(q["played"]) == {0, 1}
    un = df[(df["status"] == "unlisted") & (df["team"] == "AAA")]
    assert len(un) and un["played"].mean() == 1.0              # the regulars the report omits
