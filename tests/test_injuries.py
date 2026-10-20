"""Injury snapshot state and the ~24 h diff."""

from __future__ import annotations

import sqlite3

import pytest

from synth import day_date

from courtside import db, injuries

T0 = "2026-02-19T22:00:00Z"
T1 = "2026-02-20T22:00:00Z"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.MIGRATIONS[0])
    return c


def snap(conn, at, source, rows):
    conn.executemany(
        "INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
        "VALUES (?,?,?,?,?,?)", [(at, source, t, n, s, "r") for t, n, s in rows])
    conn.commit()


def by_player(lst):
    return {c["player"]: c for c in lst}


def test_norm_name_handles_formats():
    n = injuries.norm_name
    assert n("Jaren Jackson Jr.") == n("Jackson Jr., Jaren") == "jaren jackson"
    assert n("Nikola Jokić") == n("nikola jokic") == "nikola jokic"
    assert n("Michael Porter Jr") == "michael porter"
    assert n(None) == "" and n("") == ""


def test_diff_new_return_and_status_change(conn):
    snap(conn, T0, "espn", [("BOS", "Back Soon", "Out"), ("BOS", "Gets Worse", "Questionable"),
                            ("BOS", "Gets Better", "Out"), ("NYK", "Always Out", "Out"),
                            ("NYK", "Healthy Probable", "Probable")])
    snap(conn, T1, "espn", [("BOS", "New Guy", "Out"), ("BOS", "Gets Worse", "Out"),
                            ("BOS", "Gets Better", "Questionable"), ("NYK", "Always Out", "Out"),
                            ("NYK", "Healthy Probable", "Questionable")])
    d = injuries.diff(conn, T1)
    assert d["available"] and d["latest"] == T1 and d["baseline"] == T0
    assert set(by_player(d["new_absences"])) == {"New Guy", "Healthy Probable"}
    assert set(by_player(d["returns"])) == {"Back Soon"}
    chg = by_player(d["status_changes"])
    assert set(chg) == {"Gets Worse", "Gets Better"}
    assert (chg["Gets Worse"]["from_status"], chg["Gets Worse"]["to_status"],
            chg["Gets Worse"]["direction"]) == ("Questionable", "Out", "worse")
    assert chg["Gets Better"]["direction"] == "better"
    assert by_player(d["returns"])["Back Soon"]["to_status"] == "Healthy"
    assert "Always Out" not in str(d)          # unchanged players are not news


def test_diff_needs_a_baseline_at_least_12h_older(conn):
    snap(conn, "2026-02-20T15:00:00Z", "espn", [("BOS", "A", "Out")])
    snap(conn, T1, "espn", [("BOS", "B", "Out")])
    d = injuries.diff(conn, T1)
    assert d["available"] is False and d["new_absences"] == [] and d["baseline"] is None


def test_diff_picks_baseline_closest_to_24h_and_ignores_later_snapshots(conn):
    snap(conn, "2026-02-19T10:00:00Z", "espn", [("BOS", "A", "Out")])       # 36 h earlier
    snap(conn, "2026-02-19T21:00:00Z", "espn", [("BOS", "B", "Out")])       # 25 h earlier: best
    snap(conn, "2026-02-20T14:00:00Z", "espn", [("BOS", "B", "Out"), ("BOS", "C", "Out")])
    snap(conn, T1, "espn", [("BOS", "B", "Out"), ("BOS", "C", "Out")])
    snap(conn, "2026-02-21T22:00:00Z", "espn", [("BOS", "Z", "Out")])       # after `at`
    d = injuries.diff(conn, T1)
    assert d["baseline"] == "2026-02-19T21:00:00Z"
    assert list(by_player(d["new_absences"])) == ["C"] and d["returns"] == []


def test_official_only_compares_teams_in_both_reports_and_wins_over_espn(conn):
    # official reports list only the teams playing that day
    snap(conn, T0, "nba_official", [("BOS", "Gone", "Out"), ("NYK", "Other Day", "Out")])
    snap(conn, T1, "nba_official", [("BOS", "Fresh", "Out"), ("LAL", "New Team", "Out")])
    snap(conn, T0, "espn", [("NYK", "Espn Guy", "Out"), ("BOS", "Espn Bos", "Out")])
    snap(conn, T1, "espn", [("NYK", "Espn Guy", "Questionable"), ("BOS", "Espn Bos", "Out")])
    d = injuries.diff(conn, T1)
    assert set(d["sources"]) == {"nba_official", "espn"}
    assert set(by_player(d["new_absences"])) == {"Fresh"}      # LAL not comparable
    assert set(by_player(d["returns"])) == {"Gone"}            # NYK "Other Day" not reported
    # ESPN fills in for a team the official reports did not both cover
    assert set(by_player(d["status_changes"])) == {"Espn Guy"}
    assert "Espn Bos" not in str(d)       # BOS is covered by the official comparison


def test_current_unions_sources_official_wins_and_drops_healthy(conn):
    snap(conn, T1, "espn", [("BOS", "Star", "Out"), ("BOS", "Cleared Officially", "Out"),
                            ("NYK", "Espn Only", "Day-To-Day"), ("NYK", "Fine", "Probable")])
    snap(conn, "2026-02-20T22:30:00Z", "nba_official",
         [("BOS", "Star", "Doubtful"), ("BOS", "Cleared Officially", "Available")])
    cur = injuries.current(conn, "2026-02-20T23:00:00Z")
    bos = {e["player"]: e for e in cur["by_team"]["BOS"]}
    assert set(bos) == {"Star"} and bos["Star"]["status"] == "Doubtful"
    assert bos["Star"]["source"] == "nba_official"
    nyk = cur["by_team"]["NYK"]
    assert [e["player"] for e in nyk] == ["Espn Only"] and nyk[0]["status"] == "Questionable"
    assert cur["batches"] == {"espn": T1, "nba_official": "2026-02-20T22:30:00Z"}
    # snapshots after `at` are invisible
    assert injuries.current(conn, "2026-02-20T10:00:00Z")["by_team"] == {}
    # an old official report no longer speaks for a team
    old = injuries.current(conn, "2026-02-23T23:00:00Z")
    assert old["batches"].get("nba_official") is None


# --- pricing: status -> P(plays), names -> player ids, official vs ESPN ------------------------

ROSTER = [(1966, "LeBron James"), (203999, "Nikola Jokić"), (1628991, "Jaren Jackson Jr."),
          (1630162, "Nic Claxton"), (1629027, "Cam Thomas"), (201935, "James Harden"),
          (203507, "Giannis Antetokounmpo"), (1627736, "P.J. Washington"),
          (11, "Marcus Smith"), (12, "Marcus Smyth"), (13, "Tim Hardaway Jr."),
          (14, "Gary Payton II")]


@pytest.mark.parametrize("status,p", [
    ("Out", 0.0), ("out", 0.0), ("Doubtful", 0.10), ("Questionable", 0.5),
    ("Day-To-Day", 0.6), ("Probable", 0.9), ("Available", 1.0), ("", 1.0), (None, 1.0),
    ("Game Time Decision", 0.5), ("Some New Wording", 0.5)])
def test_status_to_play_probability(status, p):
    assert injuries.play_probability(status) == p


@pytest.mark.parametrize("name,pid,how", [
    ("LeBron James", 1966, "exact"),
    ("Nikola Jokic", 203999, "exact"),                  # accent
    ("NIKOLA JOKIĆ", 203999, "exact"),
    ("Jaren Jackson", 1628991, "exact"),                # suffix dropped by the report
    ("Jackson Jr., Jaren", 1628991, "exact"),           # official "Last, First" layout
    ("Tim Hardaway Jr", 13, "exact"),
    ("Gary Payton III", 14, "exact"),                   # wrong suffix: same person on the roster
    ("PJ Washington", 1627736, "compact"),
    ("Giannis Antetokounmpo", 203507, "exact"),
    ("Nicolas Claxton", 1630162, "nickname"),           # nickname / long form
    ("Cameron Thomas", 1629027, "nickname"),
    ("Jim Harden", None, None),                          # not close enough
])
def test_name_matching_on_a_roster(name, pid, how):
    got, h = injuries.match_player(name, ROSTER)
    assert (got, h) == (pid, how)


def test_matching_is_restricted_to_the_roster_and_refuses_ambiguity():
    assert injuries.match_player("LeBron James", [(5, "Anthony Davis")]) == (None, None)
    # two plausible roster players for one typo: refuse rather than guess
    assert injuries.match_player("Marcus Smith", ROSTER) == (11, "exact")
    assert injuries.match_player("Marcus Smth", ROSTER) == (None, "ambiguous")
    twins = [(1, "Jalen Williams"), (2, "Jaylin Williams")]
    assert injuries.match_player("Williams", twins) == (None, None)
    assert injuries.match_player("Jalen Williams", twins) == (1, "exact")


def test_espn_ids_are_ignored_but_numeric_nba_ids_win_and_other_teams_match_exactly():
    # an ESPN id says nothing about the NBA id: match by name
    assert injuries.match_player("LeBron James", ROSTER, "espn:1966") == (1966, "exact")
    # a plain NBA id on the roster wins even if the name is odd
    assert injuries.match_player("King James", ROSTER, "1966") == (1966, "id")
    # traded / signed: not on the team's roster, found league-wide by exact name only
    league = ROSTER + [(7, "Trae Young")]
    assert injuries.match_player("Trae Young", ROSTER, None, league) == (7, "other_team")
    assert injuries.match_player("Tray Young", ROSTER, None, league) == (None, None)


def test_team_availability_keeps_unmatched_names_out_of_the_lineup():
    rows = [{"player_name": "LeBron James", "status": "Out", "player_id": "espn:1966"},
            {"player_name": "Jokić, Nikola", "status": "Questionable", "player_id": None},
            {"player_name": "Cam Thomas", "status": "Probable", "player_id": None},
            {"player_name": "Healthy Guy", "status": "Available", "player_id": None},
            {"player_name": "Ghost Rookie", "status": "Out", "player_id": None}]
    avail, recs = injuries.team_availability(rows, ROSTER, "espn")
    assert avail == {1966: 0.0, 203999: 0.5, 1629027: 0.9}
    ghost = next(r for r in recs if r["player"] == "Ghost Rookie")
    assert ghost["player_id"] is None and ghost["how"] is None and ghost["p_play"] == 0.0
    assert len(recs) == 5 and all(r["source"] == "espn" for r in recs)


def test_resolve_prefers_official_only_when_it_is_the_newer_report(conn):
    snap(conn, "2026-02-20T20:00:00Z", "espn", [("BOS", "Star", "Out"), ("NYK", "Knick", "Out")])
    snap(conn, "2026-02-20T21:00:00Z", "nba_official", [("BOS", "Star", "Probable")])
    res = injuries.resolve(conn, "2026-02-20T23:00:00Z")
    assert res["teams"]["BOS"]["source"] == "nba_official"        # newer official wins
    assert res["teams"]["NYK"]["source"] == "espn"                # official does not list NYK
    snap(conn, "2026-02-20T22:00:00Z", "espn", [("BOS", "Star", "Out")])
    res = injuries.resolve(conn, "2026-02-20T23:00:00Z")
    assert res["teams"]["BOS"]["source"] == "espn"                # newer ESPN beats older official
    assert res["teams"]["BOS"]["rows"][0]["status"] == "Out"
    # nothing at all before the first snapshot
    assert injuries.resolve(conn, "2026-02-19T00:00:00Z") == {"teams": {}, "batches": {},
                                                              "fresh": False}
    # stale ESPN is not "the current state" for pricing
    old = injuries.resolve(conn, "2026-02-25T00:00:00Z", espn_max_age_h=injuries.ESPN_MAX_AGE_H)
    assert old["fresh"] is False


def test_availability_from_the_database_roster(conn):
    from synth import synth_conn
    c = synth_conn()
    day = day_date("2024-25", 22)
    roster = injuries.recent_rosters(c, day)
    assert len(roster["BOS"]) >= 8
    pid, name = roster["BOS"][0]
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, player_id, "
           "status, reason) VALUES (?,?,?,?,?,?,?)")
    at = f"{day}T15:00:00Z"
    c.execute(ins, (at, "espn", "BOS", name.upper(), f"espn:{pid}", "Day-To-Day", "x"))
    c.execute(ins, (at, "espn", "BOS", "Not A Player", "espn:9", "Out", "x"))
    c.commit()
    out = injuries.availability(c, f"{day}T20:00:00Z", day, {"BOS", "NYK"})
    assert out["BOS"]["basis"] == "injury_report" and out["BOS"]["source"] == "espn"
    assert out["BOS"]["avail"] == {pid: 0.6}
    assert out["NYK"]["avail"] == {} and out["NYK"]["source"] is None   # unlisted = healthy
    assert injuries.availability(c, f"{day}T10:00:00Z", day, {"BOS"}) == {}   # before the batch


def test_match_report_counts_and_writes_the_unmatched_csv(tmp_path):
    from synth import synth_conn
    c = synth_conn()
    day = day_date("2024-25", 22)
    roster = injuries.recent_rosters(c, day)
    name = roster["BOS"][0][1]
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
           "reason) VALUES (?,?,?,?,?,?)")
    at = f"{day}T20:00:00Z"
    for team, n, st in (("BOS", name, "Out"), ("BOS", "Nobody Atall", "Out"),
                        ("NYK", roster["BOS"][1][1], "Questionable")):
        c.execute(ins, (at, "nba_official", team, n, st, "x"))
    c.commit()
    path = tmp_path / "unmatched.csv"
    r = injuries.match_report(c, csv_path=path)
    b = r["batches"][0]
    assert (b["rows"], b["matched"], b["unknown"]) == (3, 2, 1)       # the NYK row matches
    assert b["other_team"] == 1                                       # BOS player listed for NYK
    assert b["rate"] == pytest.approx(2 / 3, abs=1e-3)
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("source,captured_at,team,player_name") and len(lines) == 2
    assert "Nobody Atall" in lines[1] and "unknown" in lines[1]


def test_match_report_rookie_on_official_roster_is_no_history_not_a_miss():
    """A listed rookie / camp player on his team's official roster with no NBA minutes cannot be
    priced, so he is `no_history` and left out of the match rate (opener 2026-10-20: five Out
    rookies escalated the pregame alert to critical)."""
    from synth import synth_conn
    c = synth_conn()
    day = day_date("2024-25", 22)
    roster = injuries.recent_rosters(c, day)
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
           "reason) VALUES (?,?,?,?,?,?)")
    at = f"{day}T20:00:00Z"
    c.execute("CREATE TABLE IF NOT EXISTS team_rosters (season TEXT, team_id INTEGER, team TEXT, "
              "player_id INTEGER, player TEXT)")
    c.execute("INSERT INTO team_rosters VALUES ('2024-25', 1, 'BOS', 999001, 'Rookie Newguy')")
    for team, n in (("BOS", roster["BOS"][0][1]), ("BOS", "Rookie Newguy"),
                    ("NYK", "Rookie Newguy"), ("BOS", "Nobody Atall")):
        c.execute(ins, (at, "nba_official", team, n, "Out", "x"))
    c.commit()
    b = injuries.match_report(c)["batches"][0]
    assert (b["rows"], b["matched"], b["no_history"], b["unknown"]) == (4, 1, 1, 2)
    assert b["rate"] == pytest.approx(1 / 3, abs=1e-3)       # 1 of the 3 priceable rows
    assert b["relevant_no_history"] == 1
