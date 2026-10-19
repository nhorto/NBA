"""DK / RotoGuru id -> NBA player_id mapping (T0.5) on a synthetic in-memory DB."""

from __future__ import annotations

import csv
import sqlite3
from datetime import date

import pytest

from courtside import db
from courtside.proj import ids

SHIPPED_SEED = ids.SEED_PATH
AS_OF = date(2026, 10, 19)
CAP = "2026-10-19T12:00:00Z"

# pid, player_index first, last, team  (to_year 2026 = current)
INDEX = [
    (1, "Shai", "Gilgeous-Alexander", "OKC"),
    (2, "Jalen", "Williams", "OKC"),
    (3, "Jaylin", "Williams", "OKC"),
    (4, "Luka", "Dončić", "LAL"),
    (5, "Jaren", "Jackson Jr.", "MEM"),
    (6, "Cameron", "Johnson", "DEN"),
    (7, "Keldon", "Johnson", "SAS"),
    (8, "Gary", "Trent Jr.", "TOR"),
    (9, "Nicolas", "Claxton", "BKN"),
    (10, "Jalen", "Johnson", "ATL"),
    (11, "Jaden", "Ivey", "DET"),
    (12, "Wendell", "Carter Jr.", "ORL"),
    (13, "Jimmy", "Butler III", "GSW"),
    (14, "PJ", "Washington", "DAL"),
    (15, "Rookie", "Newguy", "BOS"),
]


@pytest.fixture(autouse=True)
def no_shipped_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(ids, "SEED_PATH", tmp_path / "none.csv")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    for pid, f, l, t in INDEX:
        c.execute("INSERT INTO player_index (player_id, first_name, last_name, team, to_year, "
                  "snapshot_season) VALUES (?,?,?,?,2026,'2025-26')", (pid, f, l, t))
    c.commit()
    return c


def dk(conn, rows):
    """rows: (site_player_id, name, team)"""
    conn.executemany(
        "INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, draftable_id, "
        "player_name, site_player_id, team) VALUES (?,?,?,?,?,?,?,?)",
        [(CAP, "dk", "s1", "classic", f"d{i}", n, sid, t) for i, (sid, n, t) in enumerate(rows)])
    conn.commit()


def stored(conn, source="dk"):
    return {r["source_id"]: (r["player_id"], r["how"]) for r in conn.execute(
        "SELECT * FROM player_ids WHERE source=?", (source,))}


def test_exact_team_and_accents_suffixes(conn):
    dk(conn, [("a", "Luka Doncic", "LAL"), ("b", "Jaren Jackson Jr.", "MEM"),
              ("c", "Gary Trent Jr.", "TOR"), ("d", "Wendell Carter Jr.", "ORL"),
              ("e", "Jimmy Butler", "GSW"), ("f", "P.J. Washington", "DAL"),
              ("g", "Shai Gilgeous-Alexander", "OKC")])
    s = ids.resolve_dk(conn, as_of=AS_OF)
    assert s["n"] == 7 and s["matched"] == 7 and s["rate"] == 1.0
    assert stored(conn) == {"a": (4, "exact_team"), "b": (5, "exact_team"),
                            "c": (8, "exact_team"), "d": (12, "exact_team"),
                            "e": (13, "exact_team"), "f": (14, "exact_team"),
                            "g": (1, "exact_team")}
    assert ids.nba_id(conn, "dk", "a") == 4
    assert ids.nba_id(conn, "dk", 4) is None
    assert ids.nba_ids(conn, "dk", ["a", "g", "zz"]) == {"a": 4, "g": 1}


def test_exact_league_when_team_differs(conn):
    dk(conn, [("a", "Keldon Johnson", "PHI")])       # traded: DK knows, rosters do not yet
    ids.resolve_dk(conn, as_of=AS_OF)
    assert stored(conn)["a"] == (7, "exact_league")


def test_nickname_and_fuzzy(conn):
    dk(conn, [("a", "Cam Johnson", "DEN"), ("b", "Nic Claxton", "BKN"),
              ("c", "Shai Gilgeous Alexandar", "OKC")])
    ids.resolve_dk(conn, as_of=AS_OF)
    got = stored(conn)
    assert got["a"] == (6, "fuzzy") and got["b"] == (9, "fuzzy") and got["c"] == (1, "fuzzy")
    conf = {r["source_id"]: r["confidence"] for r in conn.execute("SELECT * FROM player_ids")}
    assert 0.8 < conf["c"] < 1.0


def test_jalen_jaylin_same_team_disambiguation(conn):
    dk(conn, [("a", "Jalen Williams", "OKC"), ("b", "Jaylin Williams", "OKC"),
              ("c", "Jaylen Williams", "OKC")])      # a misspelling: ambiguous between the two
    s = ids.resolve_dk(conn, as_of=AS_OF)
    got = stored(conn)
    assert got["a"] == (2, "exact_team") and got["b"] == (3, "exact_team")
    assert got["c"] == (None, "unmatched")
    assert [u["source_id"] for u in s["unmatched"]] == ["c"]
    assert s["unmatched"][0]["reason"] in ("near_duplicate", "ambiguous")
    # the guard: a first name that is only *close* is not guessed when a namesake exists ...
    pool = ids.current_pool(conn, AS_OF)
    assert ids.resolve_name("Jaylen Williams", "OKC", pool)[0] is None
    # ... an exact first name with a surname typo is
    assert ids.resolve_name("Jalen Willams", "OKC", pool)[0] == 2
    # ... but without a same-last-name player on the team it is
    assert ids.resolve_name("Jalen Johnsn", "ATL", pool)[0] == 10


def test_alias_precedence_and_never_overwritten_by_weaker(conn):
    dk(conn, [("a", "Jalen Williams", "OKC"), ("b", "Jaylen Williams", "OKC")])
    ids.resolve_dk(conn, as_of=AS_OF)
    assert stored(conn)["b"][0] is None
    ids.add_alias(conn, "dk", "Jaylen Williams", "OKC", 3, "typo in DK feed")
    # an alias beats even an exact name match
    ids.add_alias(conn, "dk", "Jalen Williams", "OKC", 3, "forced")
    s = ids.resolve_dk(conn, as_of=AS_OF)
    assert stored(conn) == {"a": (3, "alias"), "b": (3, "alias")}
    assert s["matched"] == 2 and s["auto"] == 0 and s["rate"] == 1.0
    # the alias row disappearing does not downgrade the stored mapping to a weaker one
    conn.execute("DELETE FROM player_aliases WHERE name_norm='jalen williams'")
    ids.resolve_dk(conn, as_of=AS_OF)
    assert stored(conn)["a"] == (3, "alias")
    # ... and an unknown player id is refused
    with pytest.raises(ValueError):
        ids.add_alias(conn, "dk", "Nobody", "OKC", 999999)


def test_idempotent(conn):
    dk(conn, [("a", "Luka Doncic", "LAL"), ("b", "Cam Johnson", "DEN"),
              ("c", "Totally Unknown", "BOS"), ("d", "Rookie Newguy", "BOS")])
    ids.resolve_dk(conn, as_of=AS_OF)
    first = [tuple(r) for r in conn.execute("SELECT * FROM player_ids ORDER BY source_id")]
    ids.resolve_dk(conn, as_of=AS_OF)
    again = [tuple(r) for r in conn.execute("SELECT * FROM player_ids ORDER BY source_id")]
    assert first == again and len(first) == 4
    assert conn.execute("SELECT COUNT(*) FROM player_aliases").fetchone()[0] == 0


def test_unmatched_writes_data_check(conn):
    dk(conn, [("a", "Luka Doncic", "LAL"), ("b", "Zzyzx Qwerty", "BOS")])
    s = ids.resolve_dk(conn, as_of=AS_OF)
    assert s["matched"] == 1 and s["unmatched"][0]["name"] == "Zzyzx Qwerty"
    row = conn.execute("SELECT * FROM data_checks WHERE check_name='ids_dk'").fetchone()
    assert row["status"] == "fail" and "Zzyzx Qwerty" in row["detail"]
    assert row["value"] == 0.5
    assert stored(conn)["b"] == (None, "unmatched")


def test_likely_unmatched_fails(conn, monkeypatch):
    # many matched players keep the rate above the warn line; one rotation-level miss still fails the check
    monkeypatch.setattr(ids, "_last_season_minutes",
                        lambda c: {p: 24.0 for p in range(1, 16)})
    names = [("Shai Gilgeous-Alexander", "OKC"), ("Luka Doncic", "LAL"),
             ("Jaren Jackson Jr.", "MEM"), ("Gary Trent Jr.", "TOR"),
             ("Wendell Carter Jr.", "ORL"), ("Jimmy Butler", "GSW"), ("PJ Washington", "DAL"),
             ("Cameron Johnson", "DEN"), ("Keldon Johnson", "SAS"), ("Nicolas Claxton", "BKN"),
             ("Jalen Johnson", "ATL"), ("Jaden Ivey", "DET"), ("Rookie Newguy", "BOS"),
             ("Jalen Williams", "OKC"), ("Jaylin Williams", "OKC")]
    dk(conn, [(f"s{i}", n, t) for i, (n, t) in enumerate(names)] * 1)
    # 14 other matched players are not needed: 15 matched + 1 miss = 93.8%, below the warn line
    dk(conn, [("miss", "Jaylen Williams", "OKC")])
    s = ids.resolve_dk(conn, as_of=AS_OF)
    assert [u["source_id"] for u in s["likely_unmatched"]] == ["miss"]
    assert s["likely_total"] == 16
    chk = conn.execute("SELECT * FROM data_checks WHERE check_name='ids_dk'").fetchone()
    assert chk["status"] == "fail" and "Jaylen Williams" in chk["detail"]


def test_seed_file_loaded(conn, tmp_path):
    p = tmp_path / "seed.csv"
    p.write_text("source,name,team,player_id,note\n"
                 "# comment\ndk,Jaylen Williams,OKC,3,seed\n")
    assert ids.load_seed(conn, p) == 1
    assert ids.load_seed(conn, p) == 0
    assert conn.execute("SELECT player_id FROM player_aliases").fetchone()[0] == 3
    with SHIPPED_SEED.open(newline="") as f:           # the shipped seed parses
        rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
    assert all(r["source"] in ids.SOURCES and int(r["player_id"]) for r in rows)


# --- RotoGuru ----------------------------------------------------------------------------

def hist(conn, rows):
    """rows: (game_date, rg_id, name, team)"""
    conn.executemany("INSERT INTO dfs_hist_salaries (game_date, rg_id, name, team, salary) "
                     "VALUES (?,?,?,?,5000)", rows)
    conn.commit()


def games(conn, rows):
    """rows: (game_date, player_id, name, team)"""
    conn.executemany(
        "INSERT INTO player_games (game_id, player_id, player_name, team, season, season_type, "
        "game_date, min) VALUES (?,?,?,?,?,'Regular Season',?,30)",
        [(f"{d}-{p}", p, n, t, "2016-17", d) for d, p, n, t in rows])
    conn.commit()


def test_rotoguru_last_first_format_and_date_key(conn):
    d1, d2 = "2016-11-02", "2016-11-05"
    games(conn, [(d1, 201939, "Stephen Curry", "GSW"), (d1, 202695, "Kawhi Leonard", "SAS"),
                 (d1, 203999, "Nikola Jokić", "DEN"), (d1, 203076, "Anthony Davis", "NOP"),
                 (d1, 101, "Jaren Jackson Jr.", "MEM"),
                 (d1, 102, "Jalen Williams", "OKC"), (d1, 103, "Jaylin Williams", "OKC"),
                 (d2, 201939, "Stephen Curry", "GSW")])
    hist(conn, [(d1, "rg1", "Curry, Stephen", "gsw"), (d1, "rg2", "Leonard, Kawhi", "sas"),
                (d1, "rg3", "Jokic, Nikola", "den"), (d1, "rg4", "Davis, Anthony", "nor"),
                (d1, "rg5", "Jackson Jr., Jaren", "mem"),
                (d1, "rg6", "Williams, Jalen", "okc"), (d1, "rg7", "Williams, Jaylin", "okc"),
                (d2, "rg1", "Stephen Curry", "gsw"),
                (d2, "rg9", "Nobody, Real", "gsw")])
    s = ids.resolve_rotoguru(conn)
    pid = {(r["game_date"], r["rg_id"]): r["player_id"]
           for r in conn.execute("SELECT * FROM dfs_hist_salaries")}
    assert pid[(d1, "rg1")] == pid[(d2, "rg1")] == 201939
    assert pid[(d1, "rg2")] == 202695 and pid[(d1, "rg3")] == 203999
    assert pid[(d1, "rg4")] == 203076                   # RotoGuru's "nor" = New Orleans
    assert pid[(d1, "rg5")] == 101
    assert pid[(d1, "rg6")] == 102 and pid[(d1, "rg7")] == 103
    assert pid[(d2, "rg9")] is None
    assert s["n"] == 9 and s["matched"] == 8
    assert [u["name"] for u in s["unmatched"]] == ["Nobody, Real"]
    assert ids.nba_id(conn, "rotoguru", "rg1") == 201939
    assert ids.nba_id(conn, "rotoguru", "rg9") is None
    chk = conn.execute("SELECT * FROM data_checks WHERE check_name='ids_rotoguru'").fetchone()
    assert chk["status"] == "fail" and "Nobody, Real" in chk["detail"]
    # idempotent
    before = [tuple(r) for r in conn.execute("SELECT * FROM player_ids ORDER BY source_id")]
    ids.resolve_rotoguru(conn)
    assert before == [tuple(r) for r in conn.execute("SELECT * FROM player_ids ORDER BY source_id")]


def test_rotoguru_dnp_roster_fallback_and_rg_id_vote(conn):
    # day 1 he plays; day 2 he is salaried but did not play (not in that day's box score)
    games(conn, [("2016-11-01", 201939, "Stephen Curry", "GSW"),
                 ("2016-11-02", 202695, "Kawhi Leonard", "SAS")])
    hist(conn, [("2016-11-01", "rg1", "Curry, Stephen", "gsw"),
                ("2016-11-02", "rg1", "Curry, Stephen", "gsw"),      # DNP: roster as of the date
                ("2016-11-02", "rg2", "Leonard, Kawhi", "sas"),
                ("2016-11-02", "rg3", "Curry, Steph", "gsw")])        # nickname + DNP
    s = ids.resolve_rotoguru(conn)
    got = {(r["game_date"], r["rg_id"]): r["player_id"]
           for r in conn.execute("SELECT * FROM dfs_hist_salaries")}
    assert got[("2016-11-02", "rg1")] == 201939
    assert got[("2016-11-02", "rg3")] == 201939
    assert s["rate"] == 1.0
    assert ids.nba_id(conn, "rotoguru", "rg1") == 201939


def test_rotoguru_empty_table(conn):
    s = ids.resolve_rotoguru(conn)
    assert s["n"] == 0
    assert conn.execute("SELECT status FROM data_checks WHERE check_name='ids_rotoguru'"
                        ).fetchone()[0] == "skip"


def test_jr_does_not_match_retired_sr_and_leading_jr_is_a_first_name(conn):
    conn.execute("INSERT INTO player_index (player_id, first_name, last_name, team, to_year, "
                 "snapshot_season) VALUES (2749, 'Jameer', 'Nelson', NULL, 2019, '2025-26')")
    conn.execute("INSERT INTO player_index (player_id, first_name, last_name, team, to_year, "
                 "snapshot_season) VALUES (2747, 'JR', 'Smith', 'NYK', 2026, '2025-26')")
    conn.commit()
    dk(conn, [("a", "Jameer Nelson Jr.", "PHI"), ("b", "J.R. Smith", "NYK")])
    ids.resolve_dk(conn, as_of=AS_OF)
    assert stored(conn)["a"] == (None, "unmatched")          # not the father
    assert stored(conn)["b"] == (2747, "exact_team")
    assert ids.norm_name("JR Smith") == "jr smith" and ids.norm_name("Smith, JR") == "jr smith"
    assert ids.norm_name("Jackson Jr., Jaren") == "jaren jackson"
