import gzip
from pathlib import Path

import pytest

from courtside import archive, backfill_rotoguru as br, cli, db
from courtside.sources import rotoguru as rg

FIX = Path(__file__).parent / "fixtures" / "rotoguru"
DAYS = ("2014-10-29", "2016-11-15", "2019-12-01")


def page(day):
    return (FIX / f"{day}.html").read_bytes()


def rows_of(day):
    return rg.parse_page(page(day))


def seed_days(conn, days=DAYS, season_type="Regular Season"):
    db.upsert_rows(conn, "team_games", [
        {"game_id": f"G{d}", "team": "AAA", "season": "2016-17", "season_type": season_type,
         "game_date": d} for d in days])


# ------------------------------------------------------------------------------ parsing

def test_parse_rows_starters_and_teams():
    rows = rows_of("2016-11-15")
    assert len(rows) == 30
    assert len({r["rg_id"] for r in rows}) == 30
    assert {r["game_date"] for r in rows} == {"2016-11-15"}
    by_name = {r["name"]: r for r in rows}
    russell = by_name["D'Angelo Russell"]
    assert russell == {
        "game_date": "2016-11-15", "rg_id": "4799", "name": "D'Angelo Russell", "pos": "PG",
        "team": "LAL", "opp": "BKN", "home": 1, "starter": 1, "salary": 6200, "dk_pts": 56.0,
        "minutes": 31.0, "raw_stats": "32pt 8rb 4as 3st 3to 7trey 11-20fg 3-4ft"}
    assert by_name["Ivica Zubac"]["starter"] == 0
    assert {r["team"] for r in rows} == {"LAL", "BKN"}
    assert {r["opp"] for r in rows} == {"LAL", "BKN"}
    assert all(r["home"] == int(r["team"] == "LAL") for r in rows)
    assert sum(r["starter"] for r in rows) == 10


def test_parse_dnp_and_inactive_rows():
    rows = rows_of("2016-11-15")
    dnp = [r for r in rows if r["minutes"] == 0.0]
    na = [r for r in rows if r["minutes"] is None]
    assert dnp and na
    z = next(r for r in rows if r["name"] == "Ivica Zubac")
    assert (z["minutes"], z["dk_pts"], z["salary"], z["raw_stats"]) == (None, 0.0, 3000, None)
    assert all(r["raw_stats"] is None and r["dk_pts"] == 0 for r in dnp + na)
    assert all(r["salary"] is not None for r in rows)


def test_parse_off_slate_rows_have_no_salary():
    rows = rows_of("2019-12-01")
    assert {r["team"] for r in rows} == {"WAS", "LAC"}
    assert all(r["salary"] is None and r["pos"] is None for r in rows)
    kawhi = next(r for r in rows if r["name"] == "Kawhi Leonard")
    assert (kawhi["dk_pts"], kawhi["starter"], kawhi["home"], kawhi["opp"]) == (48.5, 1, 1, "WAS")


def test_parse_whole_minutes_in_2014():
    rows = rows_of("2014-10-29")
    curry = next(r for r in rows if r["name"] == "Stephen Curry")
    assert (curry["minutes"], curry["salary"], curry["team"], curry["opp"], curry["home"]) == (
        37.0, 10100, "GSW", "SAC", 0)


def test_helpers():
    assert rg.display_name("Jackson Jr., Jaren") == "Jaren Jackson Jr."
    assert rg.display_name("Nene") == "Nene"
    assert rg.parse_minutes("39:43") == pytest.approx(39 + 43 / 60)
    assert (rg.parse_minutes("37"), rg.parse_minutes("DNP"), rg.parse_minutes("NA")) == (
        37.0, 0.0, None)
    assert rg.parse_stats("28pt 5rb 9as 4st 1to 4trey 11-23fg 2-2ft 2bl") == {
        "pts": 28, "reb": 5, "ast": 9, "stl": 4, "tov": 1, "fg3m": 4, "blk": 2}
    assert [rg.team_code(c) for c in ("nor", "pho", "gsw", "was", "uta", "", "xyz")] == [
        "NOP", "PHX", "GSW", "WAS", "UTA", None, None]
    assert rg.day_url("2016-11-05") == (
        "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=11&day=5&year=2016")


def test_validate_page():
    rg.validate_page(page("2016-11-15"), "2016-11-15")
    assert rg.page_date(page("2014-10-29")) == "2014-10-29"
    with pytest.raises(rg.PageError, match="wanted"):
        rg.validate_page(page("2016-11-15"), "2016-11-16")
    with pytest.raises(rg.PageError, match="truncated"):
        rg.validate_page(page("2016-11-15")[:3000], "2016-11-15")


def test_dk_points_recomputed_from_stats():
    """The stat string rebuilds the page's DK points (incl. DD +1.5 / TD +3 bonuses)."""
    allrows = [r for d in DAYS for r in rows_of(d)]
    exact = [r for r in allrows if rg.recompute_points(r["raw_stats"]) == pytest.approx(r["dk_pts"])]
    rate = len(exact) / len(allrows)
    print(f"dk_pts exact-match on fixtures: {len(exact)}/{len(allrows)} = {rate:.1%}")
    assert rate == 1.0
    randle = next(r for r in rows_of("2016-11-15") if r["name"] == "Julius Randle")
    assert rg.is_triple_double(randle["raw_stats"])
    # 17 + 14*1.25 + 10*1.5 - 3*0.5 = 48.0, + DD 1.5 + TD 3.0
    assert randle["dk_pts"] == rg.recompute_points(randle["raw_stats"]) == 52.5


# ------------------------------------------------------------------------------ archive / skip

def test_page_path_and_roundtrip(data_dir):
    p = br.save_page("2016-11-15", page("2016-11-15"))
    assert p == data_dir / "raw" / "rotoguru" / "dk" / "2016-11-15.html.gz"
    assert br.load_page(p) == page("2016-11-15")
    assert br.archived_days() == {"2016-11-15": p}
    assert br.page_path("2016-11-15") == p


def test_pending_days_skips_archived(data_dir):
    br.save_page("2016-11-15", b"x")
    days = ["2014-10-29", "2016-11-15", "2019-12-01"]
    assert br.pending_days(days, br.archived_days()) == ["2014-10-29", "2019-12-01"]


def test_game_days_regular_season_only_and_floor():
    conn = db.connect()
    seed_days(conn, ("2014-10-28", "2014-10-29", "2015-01-10"))
    seed_days(conn, ("2015-04-20",), "Playoffs")
    assert br.game_days(conn) == ["2014-10-29", "2015-01-10"]
    assert br.game_days(conn, playoffs=True) == ["2014-10-29", "2015-01-10", "2015-04-20"]
    assert br.game_days(conn, "2015-01-01", "2015-02-01") == ["2015-01-10"]


def test_run_fetches_only_missing_and_reparses_offline(data_dir):
    conn = db.connect()
    seed_days(conn)
    br.save_page("2016-11-15", page("2016-11-15"))     # already archived: must not be fetched
    calls = []

    def fetch(day):
        calls.append(day)
        return page(day)

    s = br.run(conn=conn, fetch=fetch)
    assert calls == ["2014-10-29", "2019-12-01"]
    assert (s["days"], s["fetched"], s["reparsed"], s["failed"]) == (3, 2, 1, 0)
    assert s["rows"] == 30 + 14 + 16
    n = conn.execute("SELECT COUNT(*) FROM dfs_hist_salaries").fetchone()[0]
    assert n == 60
    assert {r[0] for r in conn.execute("SELECT DISTINCT raw_path FROM dfs_hist_salaries")} == {
        str(data_dir / "raw" / "rotoguru" / "dk" / f"{d}.html.gz") for d in DAYS}

    s = br.run(conn=conn, fetch=fetch)                  # everything archived and parsed
    assert calls == ["2014-10-29", "2019-12-01"] and s["cached"] == 3 and s["rows"] == 0

    conn.execute("UPDATE dfs_hist_salaries SET player_id=77 WHERE rg_id='4799'")
    conn.commit()

    def no_network(day):
        raise AssertionError("reparse must not fetch")

    s = br.run(conn=conn, fetch=no_network, reparse=True)
    assert s["reparsed"] == 3 and s["rows"] == 60 and s["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM dfs_hist_salaries").fetchone()[0] == 60
    assert conn.execute(
        "SELECT player_id FROM dfs_hist_salaries WHERE rg_id='4799'").fetchone()[0] == 77


def test_run_failed_day_is_retried_next_run(data_dir):
    conn = db.connect()
    seed_days(conn)

    def flaky(day):
        if day == "2016-11-15":
            raise rg.PageError("truncated")
        return page(day)

    s = br.run(conn=conn, fetch=flaky)
    assert (s["fetched"], s["failed"]) == (2, 1) and "2016-11-15" not in br.archived_days()
    s = br.run(conn=conn, fetch=page)
    assert (s["fetched"], s["cached"], s["failed"]) == (1, 2, 0)


def test_run_aborts_after_consecutive_failures(data_dir):
    conn = db.connect()
    days = [f"2015-01-{d:02d}" for d in range(10, 20)]
    seed_days(conn, days)

    def down(day):
        raise OSError("down")

    s = br.run(conn=conn, fetch=down)
    assert s["aborted"] and s["failed"] == br.MAX_CONSECUTIVE_FAILS


def test_fetch_page_retries_and_throttles(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(rg, "_sleep", sleeps.append)
    monkeypatch.setattr(rg, "_monotonic", lambda: 100.0)
    monkeypatch.setattr(rg, "_last_request", [])

    def get(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("timeout")
        return page("2016-11-15")

    monkeypatch.setattr(rg, "_get", get)
    assert rg.fetch_page("2016-11-15") == page("2016-11-15")
    assert len(calls) == 3
    assert sleeps[0] == 5 and sleeps[2] == 20     # throttle pause, retry wait, throttle, retry wait
    assert "mon=11&day=15&year=2016" in calls[0]


# ------------------------------------------------------------------------------ on_slate

def _team(conn, gid, day, team, opp, names, short=0, home=1):
    db.upsert_rows(conn, "team_games", [{"game_id": gid, "team": team, "opp": opp,
                                         "is_home": home, "game_date": day,
                                         "season_type": "Regular Season"}])
    db.upsert_rows(conn, "player_games", [
        {"game_id": gid, "player_id": hash((team, n)) % 10**6, "player_name": n, "team": team,
         "game_date": day, "min": 5.0 if i >= len(names) - short else 20.0}
        for i, n in enumerate(names)])


def _sal(team, names, salary=5000):
    return [{"team": team, "name": n, "salary": salary} for n in names]


def test_on_slate_rule_on_synthetic_db():
    conn = db.connect()
    day = "2016-11-15"
    names = {t: [f"{t} Player{i}" for i in range(12)] for t in ("BOS", "NYK", "LAL", "BKN",
                                                                 "CHI", "MIA")}
    names["BOS"][0] = "José Álvarez"       # accents / suffixes are normalised away
    names["NYK"][1] = "Jaren Jackson Jr."
    for gid, h, a in (("G1", "BOS", "NYK"), ("G2", "LAL", "BKN"), ("G3", "CHI", "MIA")):
        _team(conn, gid, day, h, a, names[h], short=2)       # 10 rotation + 2 deep bench each
        _team(conn, gid, day, a, h, names[a], short=2, home=0)
    rows = (_sal("BOS", ["Jose Alvarez"] + names["BOS"][1:10])          # 10/10
            + _sal("NYK", ["Jaren Jackson"] + names["NYK"][2:10])                                # 9/10
            + _sal("LAL", names["LAL"][:8])                             # 8/10 = 80%: on
            + _sal("BKN", names["BKN"][:7])                             # 7/10: off
            + _sal("CHI", names["CHI"][:10])                            # CHI on, MIA none: off
            + [{"team": "MIA", "name": n, "salary": None} for n in names["MIA"]])
    game_of, games = br.slate_flags(conn, day, rows)
    assert game_of["BOS"] == game_of["NYK"] == "G1"
    assert games["G1"]["on_slate"] == 1 and games["G1"]["share"] == {"BOS": 1.0, "NYK": 0.9}
    assert games["G2"]["on_slate"] == 0 and games["G2"]["share"] == {"LAL": 0.8, "BKN": 0.7}
    assert games["G3"]["on_slate"] == 0 and games["G3"]["share"]["MIA"] == 0.0
    assert all(g["tip_et"] is None for g in games.values())     # no tip times for history


def test_annotate_sets_game_id_on_slate_and_keeps_unresolved_null():
    conn = db.connect()
    day = "2016-11-15"
    ns = [f"P{i}" for i in range(10)]
    _team(conn, "G1", day, "LAL", "BKN", ns)
    _team(conn, "G1", day, "BKN", "LAL", ns, home=0)
    rows = [dict(r, rg_id=str(i)) for i, r in enumerate(
        _sal("LAL", ns) + _sal("BKN", ns) + _sal("UTA", ns[:1]))]
    games = br.annotate(conn, day, rows, "p")
    assert [r["game_id"] for r in rows[:20]] == ["G1"] * 20 and rows[20]["game_id"] is None
    assert [r["on_slate"] for r in rows[:20]] == [1] * 20 and rows[20]["on_slate"] is None
    assert games["G1"]["on_slate"] == 1 and rows[0]["raw_path"] == "p"


def test_tip_proxy_uses_games_table():
    conn = db.connect()
    db.upsert_game(conn, game_id="E1", nba_day="2026-10-20", home="BOS", away="NYK",
                   tipoff_utc="2026-10-20T23:30:00Z")        # 19:30 ET
    db.upsert_game(conn, game_id="E2", nba_day="2026-10-20", home="BOS", away="NYK",
                   tipoff_utc="2026-10-20T17:00:00Z")        # 13:00 ET
    assert (br.tip_proxy(conn, "E1"), br.tip_proxy(conn, "E2"), br.tip_proxy(conn, "nope")) == (
        True, False, None)


def test_run_end_to_end_slate_and_summary(data_dir):
    conn = db.connect()
    day = "2016-11-15"
    db.upsert_rows(conn, "team_games", [
        {"game_id": "G1", "team": t, "opp": o, "game_date": day, "season_type": "Regular Season"}
        for t, o in (("LAL", "BKN"), ("BKN", "LAL"))])
    played = [r for r in rows_of(day) if r["minutes"]]
    db.upsert_rows(conn, "player_games", [
        {"game_id": "G1", "player_id": i + 1, "player_name": r["name"], "team": r["team"],
         "game_date": day, "min": r["minutes"]} for i, r in enumerate(played)])
    s = br.run(conn=conn, fetch=page, start=day, end=day)
    assert (s["games"], s["games_on_slate"]) == (1, 1)
    assert {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM dfs_hist_salaries")} == {"G1"}
    out = br.summarize(conn, day, day)
    assert out["days_expected"] == out["days_archived"] == out["days_with_rows"] == 1
    assert out["rows"] == out["game_id_resolved"] == 30
    assert out["played_rows"] == out["played_matched"] == len(played)
    assert out["points_exact"] == out["points_rows"] == 30
    assert out["triple_doubles"] == 1 and out["games_on_slate"] == 1


def test_cli_parser_accepts_rotoguru(monkeypatch):
    args = cli.build_parser().parse_args(
        ["backfill", "rotoguru", "--from", "2015-01-01", "--to", "2015-02-01", "--reparse"])
    assert (args.what, args.reparse, args.playoffs) == ("rotoguru", True, False)
    seen = {}
    monkeypatch.setattr(br, "run", lambda *a, **k: seen.update(a=a, k=k) or {
        "failed": 0, "aborted": False, "errors": []})
    assert cli.main(["backfill", "rotoguru", "--from", "2015-01-01", "--reparse"]) == 0
    assert seen["a"] == ("2015-01-01", None) and seen["k"]["reparse"] is True


def test_pair_names_tiers():
    from courtside.injuries import norm_name as n
    rg_names = {n(x) for x in ("J.R. Smith", "Wes Matthews", "Enes Kanter", "Nene Hilario",
                               "Maurice Williams", "Kyle Lowry", "Zed Nobody")}
    box = {n(x) for x in ("JR Smith", "Wesley Matthews", "Enes Freedom", "Nene", "Mo Williams",
                          "Kyle Lowry")}
    got = br.pair_names(rg_names, box)
    assert got == {
        "kyle lowry": ("kyle lowry", 0), "j r smith": ("smith", 3),   # norm_name eats "JR" as a suffix
        "wes matthews": ("wesley matthews", 2), "enes kanter": ("enes freedom", 3),
        "nene hilario": ("nene", 3), "maurice williams": ("mo williams", 2)}
    assert br.pair_names({"cj miles"}, {"c j miles"}) == {"cj miles": ("c j miles", 1)}
    # two same-surname candidates: ambiguous, so no loose pairing
    assert br.pair_names({"jo smith"}, {"al smith", "bo smith"}) == {}
