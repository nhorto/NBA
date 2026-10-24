"""Minutes restriction (COURTSIDE_MIN_RESTRICT, `+rs`; IMPROVEMENTS.md P7).

* Parser (`sources/espn_restrict.py`): real ESPN comments from the archive (2026-10-18 onward),
  positives and the false-positive traps around them, plus the brief's phrasings.
* Prior (`proj/min_restrict.py`): the cap rule and the name match on a snapshot fixture.
* Engine (`minutes.base_and_pplay` / `project_arrays`): no `rs_cap` column = the HEAD code
  bit for bit (a `git show HEAD:` copy in var/tmp/p7, when present); with it the flagged plan is
  capped and the team total is kept.
* Pipeline: OFF by default, `+rs` on the model version only.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import min_restrict as MRS
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.sources import espn_restrict as ER

ROOT = Path(__file__).resolve().parents[1]
HEAD_COPY = ROOT / "var" / "tmp" / "p7" / "minutes_head.py"
FIXTURE = Path(__file__).parent / "fixtures" / "espn_injuries_restrict.json.gz"


@pytest.fixture
def restore():
    state = MRS.MIN_RESTRICT
    yield
    P.set_min_restrict(state)


# ============================================================================== parser

# (player, short, long) -> (flag, cap): real comments from var/raw/espn/injuries (2026-10-18 ..)
REAL = [
    ("Zach Collins",
     "Collins (knee) is expected to play in Wednesday's season opener against the Raptors.",
     "He missed the entire preseason and could potentially face restrictions early on. With "
     "Nicolas Claxton (hamstring) set to miss two weeks, Collins and Jalen Smith could see the "
     "bulk of the center minutes.", True, None),
    ("Zach Collins",
     "Collins (knee) is questionable for Friday's game against the Hawks.",
     "The veteran center had a minutes cap of 16 minutes in the season opener Wednesday in "
     "Toronto, and he'll likely carry a similar restriction.", True, 16.0),
    ("Adem Bona", "Bona (ankle) is listed as questionable for Tuesday's game against the Knicks.",
     "If Bona is cleared for Tuesday's season opener, he'll play limited minutes behind Joel "
     "Embiid.", True, None),
    ("Collin Sexton", "Sexton (hamstring) is questionable for Wednesday's game.",
     "He'll likely have some minutes restrictions if he's cleared to play.", True, None),
    ("Jarrett Allen", "Allen (knee) is questionable for Thursday's opener against Miami.",
     "The Cavaliers may keep a close eye on his minutes if he's cleared to suit up.", True, None),
    # --- traps --------------------------------------------------------------------------------
    ("Anthony Black", "Black (ankle) is out for Friday's game.",
     "The Magic did not come to terms with Black on a contract extension before the deadline, "
     "setting the guard up to become a restricted free agent next offseason.", False, None),
    ("Ariel Hukporti", "Hukporti (knee) will be re-evaluated in two weeks.",
     "He signed a one-year deal with the 76ers over the summer, so Hukporti will be an "
     "unrestricted free agent come season's end.", False, None),
    ("Jonathan Isaac", "Isaac (knee) is day-to-day.",
     "Isaac has been a limited participant at training camp and did not make any appearances in "
     "the preseason.", False, None),
    ("T.J. McConnell", "McConnell (hamstring) will miss the season opener.",
     "Hamstring issues limited McConnell to 56 games last season, so it's not great that the "
     "34-year-old veteran will begin the 2026-27 season on the sidelines.", False, None),
    ("Dean Wade", "Wade (ankle) is questionable.",
     "He was limited to individual work Monday and appears to be questionable at best for "
     "Opening Night against the Knicks.", False, None),
    ("Kon Knueppel", "Knueppel (hamstring) is out.",
     "With the 21-year-old swingman sidelined, Sion James, Grayson Allen (knee) and Royce O'Neale "
     "are candidates to see an uptick in minutes.", False, None),
    ("Jake LaRavia", "LaRavia (ankle) is questionable for Friday.",
     "LaRavia logged 10 points (4-5 FG, 2-3 3Pt), two rebounds, two assists and one steal in 18 "
     "minutes during Wednesday's 127-105 win over the Warriors.", False, None),
    ("Taurean Prince", "Prince (neck) is probable.",
     "The veteran forward may not play meaningful minutes even if he's available, though.",
     False, None),
    ("Tobias Harris", "Harris (ankle) is out.",
     '"He\'s ramping up. "He still needs some more progressing on the court." With Harris still '
     "not ready to return, the Spurs will likely utilize their three-guard lineup.", False, None),
    ("Dereck Lively II", "Lively (foot) is out.",
     "Given that there hasn't been any reporting of Lively ramping up his workload, it seems "
     "there's a real chance he won't be ready for the start of the 2026-27 season.", False, None),
]

# the brief's phrasings, and their negations / teammate forms
SYNTH = [
    ("John Doe", "Doe will be on a minutes restriction Friday.", True, None),
    ("John Doe", "Doe is expected to play limited minutes in his return.", True, None),
    ("John Doe", "Coach said Doe will be on a minutes limit against Boston.", True, None),
    ("John Doe", "Doe is expected to play ~20 minutes in his return.", True, 20.0),
    ("John Doe", "Doe will be eased in after missing six weeks.", True, None),
    ("John Doe", "Doe will be eased back into action Wednesday.", True, None),
    ("John Doe", "Doe will play around 15-20 minutes as he works back from the injury.", True, 17.5),
    ("John Doe", "Doe will be capped at 24 minutes Saturday.", True, 24.0),
    ("John Doe", "Doe will face a 20-minute limit in his return.", True, 20.0),
    ("John Doe", "Doe will not face a minutes restriction Friday.", False, None),
    ("John Doe", "Doe's minutes restriction has been lifted.", False, None),
    ("John Doe", "Doe no longer has a minutes limit.", False, None),
    ("John Doe", "Doe was on a minutes restriction last season after the surgery.", False, None),
    ("John Doe", "With Jim Roe (knee) on a minutes restriction, Doe could see more run.",
     False, None),
    ("John Doe", "With Roe on a minutes restriction, Doe should see more minutes.", False, None),
    ("John Doe", "Doe averaged 14.4 minutes in two preseason appearances.", False, None),
]


@pytest.mark.parametrize("name,short,long_,flag,cap", REAL)
def test_real_comments(name, short, long_, flag, cap):
    r = ER.parse_comment(short, long_, name)
    assert r.flag is flag
    assert r.cap == cap
    assert bool(r.phrase) is flag


@pytest.mark.parametrize("name,text,flag,cap", SYNTH)
def test_brief_phrasings(name, text, flag, cap):
    r = ER.parse_comment(text, None, name)
    assert (r.flag, r.cap) == (flag, cap)


def test_empty_and_none():
    assert ER.parse_comment(None, None) == ER.NONE
    assert ER.parse_comment("", "", "X") == ER.NONE
    assert ER.sentences("") == []


def test_sentences_keep_initials_together():
    s = ER.sentences("T.J. McConnell is out. He'll miss Friday. Jr. Smith is fine.")
    assert len(s) == 3 and s[0].startswith("T.J. McConnell")


def test_snapshot_fixture_rows():
    with gzip.open(FIXTURE, "rt") as fh:
        rows = ER.snapshot_rows(json.load(fh))
    by = {r["player_name"]: r for r in rows}
    assert set(by) == {"Grant Williams", "Zach Collins", "Anthony Black"}
    assert by["Zach Collins"]["flag"] and by["Zach Collins"]["cap"] == 16.0
    assert by["Zach Collins"]["team"] == "CHI"            # traded to Chicago
    assert not by["Anthony Black"]["flag"] and not by["Grant Williams"]["flag"]


def test_local_raw_path(monkeypatch, tmp_path):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    p = ER.local_raw_path("/Users/x/Developer/NBA/var/raw/espn/injuries/2026-10-22/a.json.gz")
    assert p == tmp_path / "raw" / "espn" / "injuries" / "2026-10-22" / "a.json.gz"


# ============================================================================== prior

def _snapshot_db(tmp_path: Path) -> sqlite3.Connection:
    raw = tmp_path / "raw" / "espn" / "injuries" / "2026-10-22"
    raw.mkdir(parents=True)
    shutil.copy(FIXTURE, raw / "injuries__20261023T020002Z.json.gz")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE injury_snapshots (id INTEGER PRIMARY KEY, captured_at TEXT, "
                 "source TEXT, team TEXT, player_name TEXT, player_id TEXT, status TEXT, "
                 "reason TEXT, raw_path TEXT)")
    conn.execute("INSERT INTO injury_snapshots (captured_at, source, team, player_name, raw_path)"
                 " VALUES ('2026-10-23T02:00:02Z', 'espn', 'CHI', 'Zach Collins', "
                 "'/Users/x/Developer/NBA/var/raw/espn/injuries/2026-10-22/"
                 "injuries__20261023T020002Z.json.gz')")
    conn.execute("CREATE TABLE player_index (player_id INTEGER PRIMARY KEY, first_name TEXT, "
                 "last_name TEXT)")
    conn.executemany("INSERT INTO player_index VALUES (?,?,?)",
                     [(1, "Zach", "Collins"), (2, "Jalen", "Smith"), (3, "Anthony", "Black")])
    return conn


def test_flags_at_and_caps(monkeypatch, tmp_path):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    ER._rows_of_file.cache_clear()
    conn = _snapshot_db(tmp_path)
    panel = pd.DataFrame({"key": ["G|CHI", "G|CHI", "H|ORL"], "player_id": [1, 2, 3],
                          "team": ["CHI", "CHI", "ORL"], "ew_f": [20.0, 25.0, 30.0],
                          "ew_s": [22.0, 25.0, 30.0]})
    at = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
    assert MRS.restrict_caps(conn, panel, at) == {("G|CHI", 1): 16.0}
    # before the snapshot: nothing known; None: nothing
    assert MRS.restrict_caps(conn, panel, datetime(2026, 10, 22, tzinfo=timezone.utc)) == {}
    assert MRS.restrict_caps(conn, panel, None) == {}
    # stale news (comment date 2026-10-22T20:07Z, > 72 h before)
    assert MRS.restrict_caps(conn, panel, datetime(2026, 10, 26, tzinfo=timezone.utc)) == {}
    col = MRS.rs_column(panel, {("G|CHI", 1): 16.0})
    assert col[0] == 16.0 and np.isnan(col[1:]).all()


def _payload(status: str, short: str, date: str) -> dict:
    return {"injuries": [{"id": "4", "displayName": "Chicago Bulls", "injuries": [{
        "status": status, "date": date, "shortComment": short, "longComment": "",
        "athlete": {"displayName": "Zach Collins", "team": {"abbreviation": "CHI"}}}]}]}


def _two_snapshot_db(tmp_path: Path, second: dict, played: bool = False) -> sqlite3.Connection:
    raw = tmp_path / "raw" / "espn" / "injuries" / "2026-10-20"
    raw.mkdir(parents=True)
    snaps = [("2026-10-20T14:00:00Z", _payload("Day-To-Day", "Collins will be on a minutes "
                                                "restriction in the opener.", "2026-10-20T13:00Z")),
             ("2026-10-21T14:00:00Z", second)]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE injury_snapshots (captured_at TEXT, source TEXT, raw_path TEXT)")
    for i, (cap, pl) in enumerate(snaps):
        f = raw / f"injuries__{i}.json.gz"
        with gzip.open(f, "wt") as fh:
            json.dump(pl, fh)
        conn.execute("INSERT INTO injury_snapshots VALUES (?, 'espn', ?)",
                     (cap, f"/elsewhere/var/raw/espn/injuries/2026-10-20/{f.name}"))
    conn.execute("CREATE TABLE player_index (player_id INTEGER, first_name TEXT, last_name TEXT)")
    conn.execute("INSERT INTO player_index VALUES (1, 'Zach', 'Collins')")
    conn.execute("CREATE TABLE player_games (game_id TEXT, player_id INTEGER, min REAL)")
    conn.execute("CREATE TABLE games (game_id TEXT, tipoff_utc TEXT)")
    if played:
        conn.execute("INSERT INTO player_games VALUES ('g0', 1, 12.0)")
        conn.execute("INSERT INTO games VALUES ('g0', '2026-10-20T23:30:00Z')")
    return conn


@pytest.mark.parametrize("second,played,want", [
    # the next comment drops the remark: the flag sticks (within 72 h, has not played since)
    (_payload("Day-To-Day", "Collins (knee) is expected to play Wednesday.", "2026-10-21T13:00Z"),
     False, {("G|CHI", 1): 18.0}),
    # ... unless he has played a game since the news
    (_payload("Day-To-Day", "Collins (knee) is expected to play Wednesday.", "2026-10-21T13:00Z"),
     True, {}),
    # ... or the newest snapshot lists him Out
    (_payload("Out", "Collins (knee) is out Wednesday.", "2026-10-21T13:00Z"), False, {}),
])
def test_flag_is_sticky_over_the_window(monkeypatch, tmp_path, second, played, want):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    ER._rows_of_file.cache_clear()
    conn = _two_snapshot_db(tmp_path, second, played)
    panel = pd.DataFrame({"key": ["G|CHI"], "player_id": [1], "team": ["CHI"],
                          "ew_f": [24.0], "ew_s": [24.0]})
    at = datetime(2026, 10, 21, 22, 45, tzinfo=timezone.utc)
    assert MRS.restrict_caps(conn, panel, at) == want
    # without the window only the newest snapshot counts: no flag
    _, rows = ER.flags_at(conn, "2026-10-21T22:45:00Z")
    assert rows == []


def test_cap_rule_without_a_number():
    panel = pd.DataFrame({"key": ["K"] * 2, "player_id": [1, 2], "ew_f": [30.0, 4.0],
                          "ew_s": [26.0, 4.0]})
    fl = [{"key": "K", "player_id": 1, "cap": None}, {"key": "K", "player_id": 2, "cap": None}]
    caps = MRS.caps_for(panel, fl)
    assert caps[("K", 1)] == pytest.approx(MRS.RS_FRAC * 28.0)
    assert caps[("K", 2)] == MRS.RS_FLOOR
    assert MRS.caps_for(panel, [{"key": "K", "player_id": 1, "cap": 18.0}]) == {("K", 1): 18.0}


# ============================================================================== engine

def _args(p, mp, absent):
    ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
    return (p["g1"].to_numpy(int), p["g2"].to_numpy(int), p["st10"].to_numpy(float), ew,
            M._team_target(p, mp), mp)


def _head():
    spec = importlib.util.spec_from_file_location("courtside.proj.minutes_head_p7", HEAD_COPY)
    head = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = head
    spec.loader.exec_module(head)
    return head


def test_no_column_is_untouched(game):  # noqa: F811
    _, teams, _ = game
    p = teams["home"][1]
    mp = M.default_params()
    absent = np.zeros(len(p), bool)
    base, pp = M.base_and_pplay(p, mp, absent)
    assert getattr(base, "rs_cap", None) is None
    # an all-NaN column changes nothing either
    q = p.assign(rs_cap=np.nan)
    b2, pp2 = M.base_and_pplay(q, mp, absent)
    m1 = M.project_arrays(base, pp, absent, *_args(p, mp, absent))
    m2 = M.project_arrays(b2, pp2, absent, *_args(q, mp, absent))
    assert np.array_equal(m1, m2) and np.array_equal(pp, pp2)


def test_cap_binds_and_team_total_kept(game):  # noqa: F811
    _, teams, _ = game
    p = teams["home"][1].reset_index(drop=True)
    mp = M.default_params()
    absent = np.zeros(len(p), bool)
    base, pp = M.base_and_pplay(p, mp, absent)
    m0, d0 = M.project_arrays(base, pp, absent, *_args(p, mp, absent), detail=True)
    top = int(np.argmax(m0))
    lim = round(0.6 * float(m0[top]), 1)
    q = p.assign(rs_cap=[lim if i == top else np.nan for i in range(len(p))])
    b1, pp1 = M.base_and_pplay(q, mp, absent)
    m1, d1 = M.project_arrays(b1, pp1, absent, *_args(q, mp, absent), detail=True)
    assert m1[top] <= lim + 1e-9 and m1[top] < m0[top]
    others = np.arange(len(p)) != top
    assert (m1[others] >= m0[others] - 1e-9).all()             # the freed minutes go to them
    assert float((d1["p"] * m1).sum()) == pytest.approx(float((d0["p"] * m0).sum()), abs=1e-6)
    # a cap above the plan changes nothing
    q2 = p.assign(rs_cap=[40.0 if i == top else np.nan for i in range(len(p))])
    b2, pp2 = M.base_and_pplay(q2, mp, absent)
    assert np.allclose(M.project_arrays(b2, pp2, absent, *_args(q2, mp, absent)), m0)


@pytest.mark.skipif(not HEAD_COPY.exists(), reason="HEAD copy of minutes.py not present")
def test_off_equals_head_copy_engine(game):  # noqa: F811
    """No `rs_cap` column: base, P(plays) and the scenario projection equal HEAD's code."""
    head = _head()
    _, teams, _ = game
    for side in ("home", "away"):
        p = teams[side][1]
        mp = M.default_params()
        hp = head.MinutesParams.from_json(mp.to_json())
        for k in range(3):
            absent = np.zeros(len(p), bool)
            absent[:k] = True
            b, pp = M.base_and_pplay(p, mp, absent)
            hb, hpp = head.base_and_pplay(p, hp, absent)
            assert np.array_equal(np.asarray(b), np.asarray(hb)) and np.array_equal(pp, hpp)
            m = M.project_arrays(b, pp, absent, *_args(p, mp, absent))
            hm = head.project_arrays(hb, hpp, absent, *_args(p, hp, absent))
            assert np.array_equal(m, hm)


@pytest.mark.skipif(not HEAD_COPY.exists(), reason="HEAD copy of minutes.py not present")
def test_off_simulation_frame_equals_head_copy(game, monkeypatch):  # noqa: F811
    """Frame level: a whole simulated game with the minutes engine swapped for HEAD's copy."""
    spec, mp, sp = _spec(game)
    cp = C.default_copula_params("2024-25")
    a = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    head = _head()
    monkeypatch.setattr(C, "M", head)
    hp = head.MinutesParams.from_json(mp.to_json())
    for t in (spec.home, spec.away):
        t.minutes = C.ScenarioMinutes(t.minutes.panel, hp)
    b = C.simulate_game(spec, cp, hp, sp, 2000, 5)
    assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    assert np.array_equal(a.margin, b.margin)


# sha256 of the float64 means of `test_golden_capped_plan` (taken on geekom1, Linux / OpenBLAS)
GOLDEN_CAPPED = "53fcc2e6aba731592d02d65a3a4a477f1472ef4ddead6677ae97a796884fcfd0"


@pytest.mark.skipif(sys.platform != "linux", reason="exact floats: Linux only")
def test_golden_capped_plan(game):  # noqa: F811
    _, teams, _ = game
    p = teams["away"][1].reset_index(drop=True)
    mp = M.default_params()
    absent = np.zeros(len(p), bool)
    q = p.assign(rs_cap=[16.0] + [np.nan] * (len(p) - 1))
    b, pp = M.base_and_pplay(q, mp, absent)
    m = M.project_arrays(b, pp, absent, *_args(q, mp, absent))
    assert m[0] <= 16.0 and np.isfinite(m).all()
    h = hashlib.sha256(np.ascontiguousarray(m, np.float64).tobytes()).hexdigest()
    assert h == GOLDEN_CAPPED, h


# ============================================================================== switch

def test_default_off_and_tag(restore, monkeypatch):
    assert MRS.MIN_RESTRICT_DEFAULT is False
    monkeypatch.setenv(MRS.MIN_RESTRICT_ENV, "1")
    assert MRS._env_flag(MRS.MIN_RESTRICT_ENV, False) is True
    P.set_min_restrict(False)
    vs, mv = P._VS, P.MODEL_VERSION
    assert "+rs" not in mv
    P.set_min_restrict(True)
    assert P._VS == vs and P.MODEL_VERSION == mv + "+rs"        # model version only
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE
    P.set_min_restrict(False)
    assert P.MODEL_VERSION == mv


def test_build_games_column_only_for_flagged_teams(restore):
    """`rs_caps` goes into the panel of the flagged player's team only."""
    caps = {("G|BKN", 1): 16.0}
    pm = pd.DataFrame({"key": ["G|BKN", "G|BKN"], "player_id": [1, 2]})
    assert any(k == "G|BKN" for k, _ in caps) and not any(k == "G|ORL" for k, _ in caps)
    col = MRS.rs_column(pm, caps)
    assert col[0] == 16.0 and np.isnan(col[1])


# ============================================================================== latency (part B)

def _latency_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE injury_snapshots (captured_at TEXT, source TEXT, team TEXT, "
                 "player_name TEXT, status TEXT)")
    conn.executemany("INSERT INTO injury_snapshots VALUES (?,?,?,?,?)", [
        ("2026-10-21T17:00:00Z", "nba_official", "CHI", "A", "Questionable"),   # baseline
        ("2026-10-21T22:30:00Z", "nba_official", "CHI", "A", "Out"),            # news
        ("2026-10-21T23:25:00Z", "nba_official", "CHI", "B", "Out"),            # new listing
        ("2026-10-21T23:25:00Z", "nba_official", "CHI", "A", "Out"),
    ])
    conn.execute("CREATE TABLE games (game_id TEXT, tipoff_utc TEXT, home TEXT, away TEXT)")
    conn.executemany("INSERT INTO games VALUES (?,?,?,?)",
                     [("g1", "2026-10-21T23:30:00Z", "TOR", "CHI"),
                      ("g2", "2026-10-23T20:00:00Z", "MIL", "CHI")])
    conn.execute("CREATE TABLE dfs_slates (slate_id TEXT, game_ids_json TEXT)")
    conn.execute("INSERT INTO dfs_slates VALUES ('s1', '[\"g1\"]')")
    conn.execute("CREATE TABLE dfs_runs (slate_id TEXT, kind TEXT, created_at TEXT, seconds REAL,"
                 " ok INTEGER, model_version TEXT)")
    conn.executemany("INSERT INTO dfs_runs VALUES (?,?,?,?,?,?)", [
        ("s1", "lock45", "2026-10-21T22:45:00Z", 60.0, 1, "m"),
        ("s1", "swap", "2026-10-21T22:50:00Z", 1.0, 1, None),      # no re-projection
        ("s1", "lock10", "2026-10-21T23:20:00Z", 30.0, 1, "m")])
    return conn


def test_news_latency_events_and_summary():
    from courtside.proj import news_latency as NL
    ev = NL.events(_latency_db(), "2026-10-01T00:00:00Z", as_of="2026-10-22T12:00:00Z")
    a = ev[ev["player_name"] == "A"].iloc[0]
    assert (a["old"], a["new"], a["game_id"], a["rerun_kind"]) == ("Questionable", "Out", "g1",
                                                                   "lock45")
    assert a["to_rerun_min"] == pytest.approx(16.0) and a["to_lock_min"] == pytest.approx(60.0)
    b = ev[ev["player_name"] == "B"].iloc[0]           # 5 min before tip, after the last run
    assert b["missed"] and np.isnan(b["to_rerun_min"]) and pd.isna(b["old"])
    s = NL.summary(ev)
    assert s["all"]["n"] == 2 and s["all"]["missed"] == 1 and s["pending"] == 0
    assert NL.events(None).empty
