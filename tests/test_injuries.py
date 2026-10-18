"""Injury snapshot state and the ~24 h diff."""

from __future__ import annotations

import sqlite3

import pytest

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
    snap(conn, "2026-02-20T21:00:00Z", "nba_official",
         [("BOS", "Star", "Doubtful"), ("BOS", "Cleared Officially", "Available")])
    cur = injuries.current(conn, "2026-02-20T23:00:00Z")
    bos = {e["player"]: e for e in cur["by_team"]["BOS"]}
    assert set(bos) == {"Star"} and bos["Star"]["status"] == "Doubtful"
    assert bos["Star"]["source"] == "nba_official"
    nyk = cur["by_team"]["NYK"]
    assert [e["player"] for e in nyk] == ["Espn Only"] and nyk[0]["status"] == "Questionable"
    assert cur["batches"] == {"espn": T1, "nba_official": "2026-02-20T21:00:00Z"}
    # snapshots after `at` are invisible
    assert injuries.current(conn, "2026-02-20T10:00:00Z")["by_team"] == {}
    # an old official report no longer speaks for a team
    old = injuries.current(conn, "2026-02-23T23:00:00Z")
    assert old["batches"].get("nba_official") is None
