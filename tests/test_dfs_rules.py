"""DraftKings rules, pool, scoring and the scoring verification checks (proj/dfs/rules.py).

Rules payloads and draftables are the trimmed real 2026-10-19 captures in fixtures/dfs/.
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from courtside import db
from courtside.proj import cli as proj_cli
from courtside.proj.dfs import rules as R
from courtside.sources import dfs

FIX = Path(__file__).parent / "fixtures" / "dfs"


def J(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


# --------------------------------------------------------------------------- rules

def test_classic_rules_from_real_payload():
    r = R.Rules.from_payload(J("rules_70.json"))
    assert r.salary_cap == 50000 and r.game_type_id == 70
    assert r.min_games == 2 and r.min_teams == 2
    assert r.slot_names == ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL") and r.n_slots == 8
    assert r.allow_late_swap is True and not r.is_showdown and r.captain_multiplier == 1.0


def test_showdown_rules_captain_multiplier_from_payload():
    r = R.Rules.from_payload(J("rules_81.json"))
    assert r.is_showdown and r.slot_names == ("CPT",) + ("UTIL",) * 5
    assert r.captain_multiplier == 1.5 and r.slot_multiplier("CPT") == 1.5
    assert r.slot_multiplier("UTIL") == 1.0
    assert r.min_games is None and r.min_teams == 2 and r.allow_late_swap is False
    assert r.distinct_slot_names == ("CPT", "UTIL")


def test_rules_are_read_from_the_payload_not_hardcoded():
    raw = copy.deepcopy(J("rules_70.json"))
    raw["salaryCap"]["maxValue"] = 60000
    raw["gameCount"]["minValue"] = 4
    raw["teamCount"]["minValue"] = 3
    raw["lineupTemplate"] = [t for t in raw["lineupTemplate"]
                             if t["rosterSlot"]["name"] != "SF"]
    r = R.Rules.from_payload(raw)
    assert (r.salary_cap, r.min_games, r.min_teams) == (60000, 4, 3)
    assert "SF" not in r.slot_names and r.n_slots == 7
    # a parsed payload (sources.dfs.parse_rules output) is accepted too
    assert R.Rules.from_payload(dfs.parse_rules(raw)).salary_cap == 60000


def test_unknown_slot_name_is_an_error():
    raw = copy.deepcopy(J("rules_70.json"))
    raw["lineupTemplate"][0]["rosterSlot"]["name"] = "QB"
    with pytest.raises(ValueError):
        R.Rules.from_payload(raw)


def test_eligibility_per_slot():
    r = R.Rules.from_payload(J("rules_70.json"))
    assert r.eligible_slots("PG") == ("PG", "G", "UTIL")
    assert r.eligible_slots("PG/SG") == ("PG", "SG", "G", "UTIL")
    assert r.eligible_slots("C") == ("C", "UTIL")
    assert r.eligible_slots("SF/PF") == ("SF", "PF", "F", "UTIL")
    assert r.eligible_slots("PF/C") == ("PF", "C", "F", "UTIL")
    assert r.eligible_slots("") == ()
    show = R.Rules.from_payload(J("rules_81.json"))
    assert show.eligible_slots("C") == ("CPT", "UTIL")


# --------------------------------------------------------------------------- pool

def _insert_slate(conn, *, draftables="draftables_classic.json", rules="rules_70.json",
                  slate_id=154575, game_type_id=70, captured_at="2026-10-19T12:28:34Z",
                  mutate=None):
    rows = dfs.parse_draftables(J(draftables), draft_group_id=slate_id,
                                game_type_id=game_type_id, rules=J(rules))
    for r in rows:
        r.update(captured_at=captured_at, raw_path="/nonexistent/draftgroup/2026-10-20/x.json.gz")
    if mutate:
        mutate(rows)
    db.insert_rows(conn, "dfs_salaries", rows)
    return rows


def test_pool_one_entry_per_player_with_eligibility(conn, monkeypatch, tmp_path):
    rows = _insert_slate(conn)
    pool = R.build_pool(conn, 154575, rules=R.Rules.from_payload(J("rules_70.json")))
    ids = {r["site_player_id"] for r in rows}
    assert len(pool.players) == len(ids)
    by = pool.by_id()
    wemby = next(p for p in pool.players if p.name == "Victor Wembanyama")
    assert wemby.positions == ("C",) and wemby.slots == ("C", "UTIL")
    assert wemby.salary == 10500 and wemby.team == "SAS" and wemby.fppg == 51.7
    assert wemby.home == "SAS" and wemby.away == "OKC" and wemby.opp == "OKC"
    assert wemby.start_time == "2026-10-21T01:30:00.0000000Z"
    assert wemby.draftable_ids["C"] != wemby.draftable_ids["UTIL"]
    combo = next(p for p in pool.players if p.positions == ("PG", "SG"))
    assert combo.slots == ("PG", "SG", "G", "UTIL")
    assert by[wemby.dk_player_id] is wemby
    assert len(pool.eligible("UTIL")) == len(pool.players)
    assert set(pool.games) and len(pool.games) == 3


def test_pool_uses_latest_capture_only(conn):
    _insert_slate(conn, captured_at="2026-10-19T10:00:00Z",
                  mutate=lambda rs: [r.update(salary=1000) for r in rs])
    _insert_slate(conn, captured_at="2026-10-19T12:00:00Z")
    pool = R.build_pool(conn, 154575, rules=R.Rules.from_payload(J("rules_70.json")))
    assert pool.captured_at == "2026-10-19T12:00:00Z"
    assert max(p.salary for p in pool.players) == 10500
    assert [s["slate_id"] for s in R.latest_slates(conn)] == ["154575"]
    assert R.latest_slates(conn)[0]["captured_at"] == "2026-10-19T12:00:00Z"


def test_pool_excludes_disabled_rows(conn):
    def disable(rows):
        for r in rows:
            if r["player_name"] == "Victor Wembanyama":
                r["disabled"] = 1
            if r["player_name"] == "Shai Gilgeous-Alexander" and r["roster_slot"] == "UTIL":
                r["disabled"] = 1
    _insert_slate(conn, mutate=disable)
    pool = R.build_pool(conn, 154575, rules=R.Rules.from_payload(J("rules_70.json")))
    names = {p.name for p in pool.players}
    assert "Victor Wembanyama" not in names                    # every row disabled
    sga = next(p for p in pool.players if p.name == "Shai Gilgeous-Alexander")
    assert sga.slots == ("PG", "G")                            # the disabled UTIL row is gone


def test_pool_showdown_captain_salary(conn):
    _insert_slate(conn, draftables="draftables_showdown.json", rules="rules_81.json",
                  slate_id=154825, game_type_id=81)
    rules = R.Rules.from_payload(J("rules_81.json"))
    pool = R.build_pool(conn, 154825, rules=rules)
    assert pool.game_type == "showdown"
    for p in pool.players:
        assert p.slots == ("CPT", "UTIL")
        assert p.salary_by_slot["CPT"] == round(p.salary * rules.captain_multiplier)
        assert p.salary_for("CPT", rules) == p.salary_by_slot["CPT"]


def test_load_rules_from_archive_dir(conn, tmp_path):
    import gzip
    day = tmp_path / "raw" / "draftkings" / "rules" / "2026-10-20"
    day.mkdir(parents=True)
    with gzip.open(day / "gametype_70.json.gz", "wt") as f:
        json.dump(J("rules_70.json"), f)
    rows = dfs.parse_draftables(J("draftables_classic.json"), draft_group_id=154575,
                                game_type_id=70, rules=J("rules_70.json"))
    for r in rows:
        r.update(captured_at="2026-10-19T12:28:34Z",
                 raw_path=str(tmp_path / "raw" / "draftkings" / "draftgroup" / "2026-10-20" / "x"))
    db.insert_rows(conn, "dfs_salaries", rows)
    rules = R.load_rules(conn, 154575)
    assert rules is not None and rules.min_teams == 2 and rules.salary_cap == 50000
    assert R.build_pool(conn, 154575).rules == rules
    # stored rules_json wins over the archive
    stored = copy.deepcopy(J("rules_70.json"))
    stored["salaryCap"]["maxValue"] = 55555
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, rules_json) "
                 "VALUES ('154575', '2026-10-20', 70, ?)", (json.dumps(stored),))
    assert R.load_rules(conn, 154575).salary_cap == 55555


def test_lock_by_start_time(conn):
    _insert_slate(conn)
    pool = R.build_pool(conn, 154575, rules=R.Rules.from_payload(J("rules_70.json")))
    utc = timezone.utc
    before = datetime(2026, 10, 20, 18, 59, tzinfo=utc)
    assert pool.locked_ids(before) == frozenset()
    after_first = datetime(2026, 10, 20, 19, 0, 1, tzinfo=utc)           # BOS @ DET tips
    locked = pool.locked_ids(after_first)
    assert locked and {p.team for p in pool.players if p.dk_player_id in locked} <= {"BOS", "DET"}
    assert len(pool.open_players(after_first)) == len(pool.players) - len(locked)
    assert pool.locked_ids(datetime(2026, 10, 21, 2, 0, tzinfo=utc)) == \
        frozenset(p.dk_player_id for p in pool.players)
    naive = datetime(2026, 10, 20, 19, 0, 1)                             # naive = UTC
    assert pool.locked_ids(naive) == locked


# --------------------------------------------------------------------------- scoring

def line(**kw):
    return {"pts": 0, "fg3m": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0, **kw}


def test_dk_points_basic():
    assert R.dk_points(line()) == 0
    # 30 pts, 3 threes, 10 reb, 5 ast, 2 stl, 1 blk, 4 tov: double-double only
    assert R.dk_points(line(pts=30, fg3m=3, reb=10, ast=5, stl=2, blk=1, tov=4)) == \
        30 + 1.5 + 12.5 + 7.5 + 4 + 2 - 2 + 1.5
    assert R.dk_points({"pts": None, "reb": 3}) == 3.75        # None and missing = 0


def test_dk_points_triple_double_is_4_5_total():
    td = line(pts=10, reb=10, ast=10)
    assert R.dk_points(td) == 10 + 12.5 + 15 + 4.5
    dd = line(pts=10, reb=10, ast=9)
    assert R.dk_points(dd) == 10 + 12.5 + 13.5 + 1.5
    assert R.dk_points(td) - R.dk_points(line(pts=10, reb=10, ast=10), R.TD_ALT) == 1.5


def test_dk_points_defensive_double_double_and_other_categories():
    assert R.dk_points(line(stl=10, blk=10)) == 20 + 20 + 1.5      # 10 steals + 10 blocks
    assert R.dk_points(line(pts=12, stl=10)) == 12 + 20 + 1.5
    assert R.dk_points(line(pts=10, reb=10, ast=10, stl=10)) == 10 + 12.5 + 15 + 20 + 4.5
    assert R.dk_points(line(pts=9, reb=9, ast=9, stl=9, blk=9)) == 9 + 11.25 + 13.5 + 18 + 18
    assert R.double_double_cats(line(pts=10, reb=10, ast=10, stl=10, blk=10)) == 5


def test_dk_points_matches_sources_classic_points():
    for kw in (dict(pts=22, fg3m=4, reb=11, ast=3, stl=1, tov=2),
               dict(pts=11, reb=10, ast=12, blk=1, tov=5), dict(pts=2, tov=6)):
        assert R.dk_points(line(**kw)) == dfs.classic_points(**kw)


# --------------------------------------------------------------------------- check 1: FPPG

SLATE_START = "2026-10-21T00:30:00.0000000Z"       # Eastern day 2026-10-20


def _player_game(conn, pid, name, team, date_, stype, gid, **stats):
    conn.execute(
        "INSERT INTO player_games (game_id, player_id, player_name, team, season, season_type, "
        "game_date, min, pts, fg3m, reb, ast, stl, blk, tov) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, pid, name, team, "2025-26", stype, date_, stats.pop("min", 30.0),
         stats.get("pts", 0), stats.get("fg3m", 0), stats.get("reb", 0), stats.get("ast", 0),
         stats.get("stl", 0), stats.get("blk", 0), stats.get("tov", 0)))


def _dk_row(name, pid, team, fppg, slot="UTIL"):
    return {"site": "dk", "slate_id": "9001", "game_type": "classic", "game_type_id": 70,
            "draftable_id": f"d{pid}", "player_name": name, "site_player_id": str(pid),
            "site_player_dk_id": str(pid), "team": team, "positions": "PG",
            "roster_slot_id": 465, "roster_slot": slot, "salary": 5000, "status": None,
            "fppg": fppg, "disabled": 0, "competition_id": "c1", "game_label": "NYK @ BOS",
            "home": "BOS", "away": "NYK", "start_time": SLATE_START,
            "captured_at": "2026-10-19T12:00:00Z", "raw_path": "x"}


def _synthetic_fppg(conn, *, n_players=60, n_off=3, td_fppg_total=4.5):
    """n_players with 3 games each (RS, RS, playoffs); the first n_off have a wrong DK FPPG.
    Player 0 is a triple-double machine (12 TDs) whose FPPG uses `td_fppg_total`."""
    rows = []
    for i in range(n_players):
        name, team = f"First{i:02d} Last{i:02d}", "BOS" if i % 2 else "NYK"
        stats = [dict(pts=10 + i, reb=3, ast=2), dict(pts=20, reb=5 + i % 4, ast=4, stl=1),
                 dict(pts=8, ast=8 + i % 3, tov=2)]
        for g, (st, kind) in enumerate(zip(stats, ("Regular Season",) * 2 + ("Playoffs",))):
            _player_game(conn, 100 + i, name, team, f"2026-0{g + 1}-1{i % 9}", kind,
                         f"00{i:02d}{g}", **st)
        mean = sum(R.dk_points(line(**s)) for s in stats) / 3
        rows.append(_dk_row(name, 100 + i, team, round(mean + (2.0 if i < n_off else 0), 1)))
    # triple-double star on top of the synthetic pool
    star_sc = {**dfs.CLASSIC_SCORING, "triple_double": td_fppg_total - 1.5}
    star = []
    for g in range(14):
        s = dict(pts=25, reb=12, ast=11) if g < 12 else dict(pts=14, reb=4, ast=3)
        star.append(s)
        _player_game(conn, 999, "Nikola Starman", "BOS", f"2026-02-{g + 1:02d}",
                     "Regular Season", f"99{g:02d}", **s)
    mean = sum(R.dk_points(line(**s), star_sc) for s in star) / 14
    rows.append(_dk_row("Nikola Starman", 999, "BOS", round(mean, 2)))
    db.insert_rows(conn, "dfs_salaries", rows)
    return rows


def test_check_fppg_hit_rate_window_and_td_evidence(conn):
    _synthetic_fppg(conn, n_players=60, n_off=4)
    rep = R.check_fppg(conn)
    assert rep["n"] == 61 and rep["within"] == 57 and rep["unmatched"] == []
    assert rep["frac"] == pytest.approx(57 / 61)
    assert {round(m["delta"], 1) for m in rep["misses"]} == {-2.0} and rep["n_miss"] == 4
    assert rep["window"].startswith("2026-27")
    assert rep["td"]["players"] == 1 and rep["td"]["verdict"] == "4.5"
    assert rep["td"]["mae"]["td_4.5"] < rep["td"]["mae"]["td_3.0"]
    row = R.fppg_check_row(rep)
    assert row["check_name"] == R.FPPG_CHECK and row["status"] == "warn"       # 93.4% < 95%
    assert row["value"] == pytest.approx(57 / 61, abs=1e-4)


def test_check_fppg_ok_when_all_match_and_fail_when_td_total_is_3(conn):
    _synthetic_fppg(conn, n_players=60, n_off=0)
    rep = R.check_fppg(conn)
    assert rep["within"] == rep["n"] == 61
    assert R.fppg_check_row(rep)["status"] == "ok"


def test_check_fppg_flags_a_3_point_td_bonus(tmp_path):
    c = db.connect(tmp_path / "t3.db")
    _synthetic_fppg(c, n_players=60, n_off=0, td_fppg_total=3.0)
    rep = R.check_fppg(c)
    assert rep["td"]["verdict"] == "3.0"
    row = R.fppg_check_row(rep)
    assert row["status"] == "fail" and "TD bonus" in row["detail"]


def test_check_fppg_needs_enough_draftables(conn):
    _synthetic_fppg(conn, n_players=10, n_off=0)
    assert R.fppg_check_row(R.check_fppg(conn))["status"] == "warn"


def test_check_fppg_uses_previous_season_before_five_games_and_current_after(conn):
    rows = [_dk_row("Hot Rookie", 1, "BOS", 50.0), _dk_row("Old Hand", 2, "NYK", 30.0)]
    db.insert_rows(conn, "dfs_salaries", rows)
    for g in range(6):                                         # Old Hand: 6 games this season
        conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, season, "
                     "season_type, game_date, min, pts) VALUES (?,?,?,?,?,?,?,?,?)",
                     (f"c{g}", 2, "Old Hand", "NYK", "2026-27", "Regular Season",
                      f"2026-10-0{g + 1}", 30.0, 10))
    for g in range(3):                                         # last season for both
        for pid, nm, tm in ((1, "Hot Rookie", "BOS"), (2, "Old Hand", "NYK")):
            _player_game(conn, pid, nm, tm, f"2026-01-0{g + 1}", "Regular Season",
                         f"p{pid}{g}", pts=50)
    conn.execute("UPDATE dfs_salaries SET start_time='2026-10-21T00:30:00.0000000Z'")
    rep = R.check_fppg(conn, day=date(2026, 10, 20))
    assert rep["n"] == 2 and rep["within"] == 1                 # Hot Rookie 50 ok; Old Hand 10 != 30
    assert rep["misses"][0]["name"] == "Old Hand" and rep["misses"][0]["season"] == "2026-27"


# --------------------------------------------------------------------------- check 2: RotoGuru

def _hist(conn, rg_id, date_, name, team, dk_pts, raw):
    conn.execute("INSERT INTO dfs_hist_salaries (game_date, rg_id, name, team, dk_pts, raw_stats) "
                 "VALUES (?,?,?,?,?,?)", (date_, rg_id, name, team, dk_pts, raw))


def test_check_rotoguru_join_and_own_stat_lines(conn):
    # joined: exact, exact triple-double, wrong (a box-score feed difference), unmatched name
    _player_game(conn, 1, "Stephen Curry", "GSW", "2014-10-29", "Regular Season", "g1",
                 pts=24, reb=10, ast=5, stl=6, tov=4, fg3m=2)
    _player_game(conn, 2, "Russell Westbrook", "OKC", "2014-10-29", "Regular Season", "g2",
                 pts=20, reb=11, ast=12, tov=3)
    _player_game(conn, 3, "Brandon Knight", "MIL", "2014-10-29", "Regular Season", "g3",
                 pts=22, reb=8, ast=13, blk=1, tov=4, fg3m=2)
    curry = 24 + 12.5 + 7.5 + 12 - 2 + 1 + 1.5
    _hist(conn, "a", "2014-10-29", "Stephen Curry", "GSW", curry, "24pt 10rb 5as 6st 4to 2trey")
    _hist(conn, "b", "2014-10-29", "Russell Westbrook", "OKC", 20 + 13.75 + 18 - 1.5 + 4.5,
          "20pt 11rb 12as 3to")
    _hist(conn, "c", "2014-10-29", "Knight, Brandon", "MIL", 99.0, "22pt 8rb 13as 1bl 4to 2trey")
    _hist(conn, "d", "2014-10-29", "Nobody Here", "BOS", 10.0, "10pt")
    rep = R.check_rotoguru(conn)
    assert rep["n_rows"] == 4 and rep["n"] == 3 and rep["unmatched"] == 1
    assert rep["exact"] == 2 and rep["tds"] == 1 and rep["tds_exact"] == 1
    assert rep["tds_exact_30"] == 0
    assert rep["misses"][0]["name"] == "Knight, Brandon" and rep["misses"][0]["rg"] == 99.0
    own = rep["own"]
    assert own["n"] == 4 and own["exact"] == 3 and own["tds"] == 1
    row = R.rotoguru_check_row(rep)
    assert row["check_name"] == R.ROTOGURU_CHECK and row["status"] in ("warn", "fail")


def test_check_rotoguru_ok_needs_td_count_and_exactness(conn):
    # 25 triple-doubles + 25 normal lines, all exact on both the join and RotoGuru's own lines
    for i in range(50):
        td = i < 25
        st = dict(pts=10, reb=10, ast=10) if td else dict(pts=15 + i, reb=3, ast=2)
        d = f"2014-11-{i % 28 + 1:02d}"
        _player_game(conn, 500 + i, f"Guy{i:02d} Name{i:02d}", "BOS", d, "Regular Season",
                     f"x{i}", **st)
        raw = " ".join(f"{v}{k}" for k, v in (("pt", st["pts"]), ("rb", st["reb"]),
                                              ("as", st["ast"])))
        _hist(conn, f"r{i}", d, f"Guy{i:02d} Name{i:02d}", "BOS", R.dk_points(line(**st)), raw)
    rep = R.check_rotoguru(conn)
    assert rep["n"] == 50 and rep["exact"] == 50 and rep["tds"] == 25
    assert rep["own"]["n"] == 50 and rep["own"]["exact"] == 50 and rep["own"]["tds"] == 25
    assert R.rotoguru_check_row(rep)["status"] == "ok"
    # fewer than 20 triple-doubles -> warn
    conn.executemany("DELETE FROM dfs_hist_salaries WHERE rg_id=?", [(f"r{i}",) for i in range(7)])
    assert R.rotoguru_check_row(R.check_rotoguru(conn))["status"] == "warn"


def test_check_rotoguru_skips_when_empty(conn):
    rep = R.check_rotoguru(conn)
    assert rep["n_rows"] == 0 and rep["n"] == 0
    assert R.rotoguru_check_row(rep)["status"] == "skip"


def test_parse_rg_stats():
    assert R.parse_rg_stats("24pt 10rb 5as 6st 4to 2trey 7-17fg 8-9ft") == \
        {"pts": 24, "reb": 10, "ast": 5, "stl": 6, "tov": 4, "fg3m": 2, "blk": 0}
    assert R.parse_rg_stats("2bl")["blk"] == 2 and R.parse_rg_stats(None) is None
    assert R.parse_rg_stats("") is None


# --------------------------------------------------------------------------- runner / CLI

def test_verify_stores_data_checks(conn):
    _synthetic_fppg(conn, n_players=60, n_off=0)
    rows, reps = R.verify(conn, now=datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc))
    assert {r["check_name"] for r in rows} == {R.FPPG_CHECK, R.ROTOGURU_CHECK}
    stored = {r["check_name"]: r for r in conn.execute("SELECT * FROM data_checks")}
    assert stored[R.FPPG_CHECK]["status"] == "ok" and stored[R.FPPG_CHECK]["run_at"] == \
        "2026-10-19T12:00:00Z"
    assert stored[R.ROTOGURU_CHECK]["status"] == "skip"
    assert reps["fppg"]["n"] == 61


def test_cli_commands_are_routed(conn, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "_conn", lambda: conn)
    _synthetic_fppg(conn, n_players=60, n_off=0)
    # reachable through `courtside dfs ...`
    assert proj_cli.dfs_main(["check-scoring", "--no-store"]) == 0
    out = capsys.readouterr().out
    assert "dfs_scoring_fppg" in out and "dfs_scoring_rotoguru" in out
    assert proj_cli.dfs_main(["rules", "--slate", "9001"]) == 0
    assert "slate 9001" in capsys.readouterr().out


def test_cli_check_scoring_returns_1_on_fail(tmp_path, monkeypatch, capsys):
    c = db.connect(tmp_path / "fail.db")
    _synthetic_fppg(c, n_players=60, n_off=0, td_fppg_total=3.0)
    monkeypatch.setattr(R, "_conn", lambda: c)
    assert proj_cli.dfs_main(["check-scoring", "--no-store"]) == 1
