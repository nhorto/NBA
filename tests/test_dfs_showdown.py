"""Showdown Captain Mode (proj/dfs/showdown.py, T2.15): our CP-SAT optimum equals
pydfs-lineup-optimizer's on Showdown slates built from RotoGuru single games and on the captured
DK Showdown slates; DK rules (CPT 1.5x salary and points, one CPT + five UTIL, a player once,
>= 2 teams, the cap to the dollar); Top N and the DK CSV for Showdown; the runner path.

The fixture games always run; the 30-game check runs on the real var/courtside.db when present."""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from courtside import db
from courtside.proj.dfs import export as X
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import runner, showdown as SD
from courtside.proj.dfs import topn as TN

pytest.importorskip("pydfs_lineup_optimizer")

FIX = Path(__file__).parent / "fixtures" / "dfs"
REAL_DB = Path(__file__).parents[1] / "var" / "courtside.db"
RULES81 = json.loads((FIX / "rules_81.json").read_text())
SHOW = R.Rules.from_payload(RULES81)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def load_hist_fixture(conn):
    data = json.loads(gzip.decompress((FIX / "hist_slates.json.gz").read_bytes()))
    rows = []
    for day, rs in data.items():
        for rg, name, pos, team, opp, home, starter, sal, pts in rs:
            rows.append({"game_date": day, "rg_id": rg, "name": name, "pos": pos, "team": team,
                         "opp": opp, "home": home, "starter": starter, "salary": sal,
                         "dk_pts": pts, "on_slate": 1})
    db.insert_rows(conn, "dfs_hist_salaries", rows)


def insert_real(conn, slate_id="154825"):
    data = json.loads(gzip.decompress((FIX / "real_slates_20261019.json.gz").read_bytes()))
    db.insert_rows(conn, "dfs_salaries", data[slate_id])
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, rules_json, "
                 "lock_utc) VALUES (?,?,?,?,?,?)", (slate_id, "2026-10-20", 81, "showdown",
                                                    json.dumps(RULES81), "2026-10-20T19:00:00Z"))
    conn.commit()


def mk(pid, team, salary):
    return R.PoolPlayer(str(pid), None, f"P{pid}", team, ("PG",), salary, ("CPT", "UTIL"),
                        {"CPT": SD.captain_salary(salary), "UTIL": salary},
                        {"CPT": f"9{pid}", "UTIL": f"8{pid}"}, "g", None, "A @ B", "B", "A",
                        None, None, None, None, None, None)


def brute(players, projs, rules):
    best = None
    for cpt in players:
        rest = [p for p in players if p is not cpt]
        for utils in itertools.combinations(rest, 5):
            pairs = [("CPT", cpt.dk_player_id)] + [("UTIL", u.dk_player_id) for u in utils]
            by = {p.dk_player_id: p for p in players}
            if SD.rule_problems(rules, by, pairs):
                continue
            v = SD.points(rules, projs, pairs)
            best = v if best is None or v > best else best
    return best


# =========================================================================== rules

def test_captain_salary_is_exact_1_5x():
    assert SD.captain_salary(10000) == 15000
    assert SD.captain_salary(8100) == 12150          # Classic $100 steps: $50 captains, exact
    assert SD.captain_salary(5200) == 7800           # DK Showdown $200 steps: always $300 steps
    assert SD.captain_salary(3333) == 5000           # never seen on DK: half up to the dollar
    assert SHOW.captain_multiplier == 1.5 and SHOW.min_teams == 2 and SHOW.salary_cap == 50000
    assert SHOW.allow_late_swap is False


def test_captured_cpt_rows_are_exactly_1_5x(conn):
    insert_real(conn)
    rep = SD.check_cpt_salaries(conn)
    assert rep["n"] >= 40 and rep["exact"] == rep["n"] and not rep["mismatches"]
    assert rep["needs_rounding"] == 0 and set(rep["util_price_steps"]) == {200}
    pool = R.build_pool(conn, "154825")
    for p in pool.players:                              # CPT and UTIL are separate draftables
        assert set(p.slots) == {"CPT", "UTIL"} and p.draftable_ids["CPT"] != p.draftable_ids["UTIL"]
        assert p.salary_for("CPT", pool.rules) == SD.captain_salary(p.salary)


def test_salary_cap_to_the_dollar_with_a_50_dollar_captain():
    """The optimum spends exactly $50,000 with a $12,150 CPT; a better lineup at $50,050 is
    illegal for both solvers."""
    ps = [mk(1, "A", 8100), mk(2, "B", 9000), mk(3, "A", 8000), mk(4, "B", 7500),
          mk(5, "A", 7000), mk(6, "B", 6350), mk(7, "A", 6400), mk(8, "B", 3000)]
    projs = {"1": 60.0, "2": 50.0, "3": 45.0, "4": 42.0, "5": 40.0, "6": 30.0, "7": 30.75,
             "8": 5.0}
    lu = O.optimize(ps, projs, SHOW)
    best, theirs, settings = SD.pydfs_best(ps, projs, SHOW)
    assert lu.salary == 50000 and lu.entries[0].player.dk_player_id == "1"
    assert lu.entries[0].salary == 12150
    assert "7" not in lu.player_ids                     # P7 instead of P6 would cost $50,050
    assert lu.projected == pytest.approx(best) == pytest.approx(brute(ps, projs, SHOW))
    assert sorted(theirs) == sorted(lu.slot_pairs())
    assert settings["budget"] == 50000 and settings["max_from_one_team"] == 5


def test_rule_problems_catch_illegal_lineups():
    ps = {p.dk_player_id: p for p in [mk(i, "A" if i < 7 else "B", 5000) for i in range(1, 9)]}
    ok = [("CPT", "1")] + [("UTIL", str(i)) for i in (2, 3, 4, 5, 7)]
    assert SD.rule_problems(SHOW, ps, ok) == []
    assert "fewer than 2 teams" in SD.rule_problems(SHOW, ps, [("CPT", "1")] + [
        ("UTIL", str(i)) for i in (2, 3, 4, 5, 6)])
    assert "a player is both CPT and UTIL" in SD.rule_problems(SHOW, ps, [("CPT", "1")] + [
        ("UTIL", str(i)) for i in (1, 3, 4, 5, 7)])
    big = {**ps, "9": mk(9, "B", 20000)}
    assert any("salary" in p for p in SD.rule_problems(SHOW, big, [("CPT", "9")] + [
        ("UTIL", str(i)) for i in (2, 3, 4, 5, 7)]))


def test_never_cpt_and_util_and_two_teams_on_synthetic_pools():
    import random
    for seed in range(6):
        rnd = random.Random(seed)
        ps = [mk(i, "A" if i % 3 else "B", rnd.randrange(10, 60) * 200) for i in range(1, 10)]
        projs = {p.dk_player_id: round(rnd.uniform(5, 55) * 4) / 4 for p in ps}
        lu = O.optimize(ps, projs, SHOW)
        assert len(lu.player_ids) == 6 and lu.dk_header() == ["CPT"] + ["UTIL"] * 5
        assert len({p.team for p in lu.players}) >= 2
        assert lu.projected == pytest.approx(brute(ps, projs, SHOW))
        assert lu.projected == pytest.approx(SD.pydfs_best(ps, projs, SHOW)[0])


# =========================================================================== the oracle

def test_pydfs_equal_on_every_fixture_game(conn):
    load_hist_fixture(conn)
    games = SD.hist_games(conn)
    assert len(games) >= 20
    n = 0
    for day, a, b in games:
        pool, projs = SD.hist_showdown_pool(conn, day, a, b, SHOW)
        c = SD.check_pool(pool, projs, SHOW, "rotoguru")
        assert c.ok, (c.slate, c.problems)
        n += 1
    assert n == len(games)


def test_pydfs_equal_on_the_captured_showdown_slate(conn):
    insert_real(conn)
    pool = R.build_pool(conn, "154825")
    projs = {p.dk_player_id: p.fppg for p in pool.players if p.fppg is not None}
    c = SD.check_pool(pool, projs, pool.rules, "captured")
    assert c.ok and c.players >= 40


def test_thirty_showdown_slates_from_the_real_db():
    """The T2.15 acceptance: 30 RotoGuru games (actual and noisy points) + every captured
    Showdown slate, all equal to pydfs."""
    if not REAL_DB.exists():
        pytest.skip("no var/courtside.db")
    ro = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        if len(SD.hist_games(ro)) < 30:
            pytest.skip("fewer than 30 RotoGuru games")
        rep = SD.run_check(ro, 30)
    finally:
        ro.close()
    assert rep["rotoguru"] == 30 and rep["passed"] == rep["total"]
    assert sum(c.source == "rotoguru-noisy" for c in rep["checks"]) == 30
    assert rep["slates_passed"] == rep["slates"] >= 30
    cs = rep["cpt_salaries"]
    assert cs["exact"] == cs["n"]


def test_cli_showdown_check(conn, capsys, monkeypatch):
    from courtside import config
    from courtside.proj import cli
    path = config.data_dir() / "courtside.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    c = db.connect(path)
    load_hist_fixture(c)
    insert_real(c)
    c.close()
    rc = cli.dfs_main(["showdown-check", "--n", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and "6/6 slates" in out and "11/11 comparisons" in out
    assert "CPT salary = 1.5 x UTIL on captured rows" in out


# =========================================================================== Top N, CSV, runner

def test_top_n_and_dk_csv_for_showdown(conn):
    load_hist_fixture(conn)
    day, a, b = SD.hist_games(conn)[0]
    pool, projs = SD.hist_showdown_pool(conn, day, a, b, SHOW)
    res = TN.top_n(pool, projs, SHOW, 8, min_diff=1)
    assert len(res) == 8
    by = pool.by_id()
    seen = set()
    prev = None
    for r in res:
        pairs = r.lineup.slot_pairs()
        assert SD.rule_problems(SHOW, by, pairs) == []
        assert frozenset(pairs) not in seen
        seen.add(frozenset(pairs))
        assert prev is None or r.lineup.objective <= prev + 1e-9
        prev = r.lineup.objective
    slots = [json.loads(r.lineup.slots_json()) for r in res]
    text = X.to_csv(slots)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL"]
    for row, r in zip(rows[1:], res):
        cpt = r.lineup.entries[0].player
        assert row[0] == cpt.draftable_ids["CPT"] and row[0].startswith("9")
        assert all(c.startswith("8") for c in row[1:])
    named = list(csv.reader(io.StringIO(X.to_csv(slots, style="name_id"))))
    assert named[1][0].endswith(f"({res.lineups[0].lineup.entries[0].player.draftable_ids['CPT']})")
    assert X.default_header(showdown=True) == rows[0]


def test_runner_builds_showdown_lineups_that_export(conn, monkeypatch):
    insert_real(conn)
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", None)       # FPPG fallback
    monkeypatch.setattr(runner, "availability_for", lambda c, s, pool, now: {
        p.dk_player_id: {"p": 1.0, "status": "", "source": "stub"} for p in pool.players})
    now = datetime(2026, 10, 20, 18, 15, tzinfo=timezone.utc)
    res = runner.run_slate(conn, "154825", "manual", now=now, refresh=False, n=5)
    assert res["status"] == "ok" and res["lineups"]["topn"] == 5
    rows = conn.execute("SELECT mode, slots_json FROM dfs_lineups ORDER BY mode, rank").fetchall()
    slots = [json.loads(r[1]) for r in rows if r[0] == "topn"]
    text = X.to_csv(slots)
    head = text.splitlines()[0].split(",")
    assert head == ["CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL"]
    pool = R.build_pool(conn, "154825")
    by = pool.by_id()
    for s in slots:
        pairs = [(x["slot"], str(x["player_id"])) for x in s]
        assert SD.rule_problems(pool.rules, by, pairs) == []
        assert s[0]["draftable_id"] == by[pairs[0][1]].draftable_ids["CPT"]
        assert s[0]["salary"] == SD.captain_salary(by[pairs[0][1]].salary)
    assert runner.run_slate(conn, "154825", "swap", now=now, refresh=False)["status"] == \
        "nothing_to_swap"
