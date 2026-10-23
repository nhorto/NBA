"""Game-day keying of official injury rows (IMPROVEMENTS B8; availability.INJ_GAMEDAY, +gd).

Schema 16 `injury_snapshots.game_date`, the writers, the PDF backfill, and the switch: OFF is
bit-identical whether or not game_date is filled; ON keys batches by the report's game date.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from courtside import backfill_injuries as bi
from courtside import collect, config, db
from courtside.proj import availability as av
from courtside.sources import injury_report as ir

ET = config.NBA_TZ
DAY = date(2026, 3, 20)
GAME = "G_TARGET"
TIP = "2026-03-21T00:00:00Z"          # 20:00 ET on DAY
T_10AM = "2026-03-20T14:00:00Z"       # 10:00 ET on DAY (the first pregame run)
T_5PM_PREV = "2026-03-19T22:00:00Z"   # 17:00 ET the day before
T_10PM_PREV = "2026-03-20T03:00:00Z"  # 22:00 ET the day before (still 03-19 ET)
MINUTES = {1: 32, 2: 30, 3: 22, 4: 20, 5: 18, 6: 9, 7: 8, 8: 6}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _switch_off():
    av.set_inj_gameday(False)
    yield
    av.set_inj_gameday(False)


def pid(team, i):
    return (1000 if team == "AAA" else 2000) + i


def name(team, i):
    return f"{'Alpha' if team == 'AAA' else 'Bravo'} Player{'abcdefgh'[i - 1]}"


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


def world(conn):
    for k in range(12):
        add_game(conn, f"H{k:02d}", DAY - timedelta(days=13 - k))
    add_game(conn, GAME, DAY)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?,?,?,?,?,?,?)", (GAME, "2025-26", DAY.isoformat(), TIP, "AAA", "BBB",
                                            "final"))
    conn.commit()


def report(conn, at, rows, raw_path=None):
    """rows: (team, name, status, reason, game_date)."""
    conn.executemany(
        "INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason, "
        "raw_path, game_date) VALUES (?,?,?,?,?,?,?,?)",
        [(at, "nba_official", t, n, s, r, raw_path, g) for t, n, s, r, g in rows])
    conn.commit()


def row(df, team, i):
    return df[df["player_id"] == pid(team, i)].iloc[0]


# --- schema + writers -------------------------------------------------------------------------

def test_schema_16_adds_nullable_game_date_and_the_writers_fill_it(conn, monkeypatch):
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(injury_snapshots)")}
    assert "game_date" in cols and cols["game_date"][3] == 0            # nullable
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] >= 16
    parsed = [{"team": "BOS", "player_name": "Tatum, Jayson", "player_name_norm": "Jayson Tatum",
               "status": "Out", "reason": "Injury/Illness - Achilles", "game_date": "2026-01-11"},
              {"team": "NYK", "player_name": None, "player_name_norm": None,
               "status": "NOT YET SUBMITTED", "reason": None, "game_date": "2026-01-10"}]
    report_dt = datetime(2026, 1, 10, 17, 30, tzinfo=ET)
    rows = bi.official_rows(parsed, report_dt, "/x/r.pdf")
    assert [r["game_date"] for r in rows] == ["2026-01-11"]
    db.insert_injury_snapshots(conn, rows)
    # ESPN rows carry no game date and still insert
    db.insert_injury_snapshots(conn, [{"captured_at": "2026-01-10T20:00:00Z", "source": "espn",
                                       "team": "BOS", "player_name": "X", "status": "Out"}])
    assert [tuple(r) for r in conn.execute("SELECT source, game_date FROM injury_snapshots "
                                           "ORDER BY id")] == [("nba_official", "2026-01-11"), ("espn", None)]
    # collect's live step writes the same thing
    conn.execute("DELETE FROM injury_snapshots")
    monkeypatch.setattr(ir, "find_latest_report", lambda *a, **k: ("http://x/r.pdf", report_dt))
    monkeypatch.setattr(ir, "fetch_report", lambda url: b"%PDF-fake")
    monkeypatch.setattr(ir, "parser_available", lambda: True)
    monkeypatch.setattr(ir, "parse_report", lambda pdf: parsed)
    collect._official_report_step(conn, date(2026, 1, 10))
    assert [tuple(r) for r in conn.execute("SELECT game_date FROM injury_snapshots")] == [("2026-01-11",)]


# --- PDF backfill -----------------------------------------------------------------------------

def _parsed(*rows):
    return [{"team": t, "player_name": n, "player_name_norm": n, "status": s, "reason": r,
             "game_date": g} for t, n, s, r, g in rows]


def test_backfill_game_dates_by_order_then_by_key_idempotent_and_dry_run(conn, tmp_path, monkeypatch):
    pdf_a, pdf_b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-a")
    pdf_b.write_bytes(b"%PDF-b")
    # A: rows in parse order -> positional match (a back-to-back player appears twice)
    report(conn, "2026-01-14T22:00:00Z", [("CLE", "A", "Out", "x", None), ("NYK", "B", "Out", "y", None),
                                          ("NYK", "B", "Questionable", "y", None)], str(pdf_a))
    # B: an extra DB row the PDF does not have -> key match; one row stays NULL
    report(conn, "2026-01-15T22:00:00Z", [("BOS", "C", "Out", "z", None), ("BOS", "D", "Out", "w", None)],
           str(pdf_b))
    # C: a PDF that is not archived anywhere
    report(conn, "2026-01-16T22:00:00Z", [("MIA", "E", "Out", "q", None)], str(tmp_path / "gone.pdf"))
    parsed = {b"%PDF-a": _parsed(("CLE", "A", "Out", "x", "2026-01-14"), ("NYK", "B", "Out", "y", "2026-01-14"),
                                 ("NYK", "B", "Questionable", "y", "2026-01-15")),
              b"%PDF-b": _parsed(("BOS", "D", "Out", "w", "2026-01-16"))}
    monkeypatch.setattr(ir, "parse_report", lambda pdf: parsed[pdf])

    dry = bi.backfill_game_dates(conn, dry_run=True)
    assert conn.execute("SELECT COUNT(*) FROM injury_snapshots WHERE game_date IS NOT NULL").fetchone()[0] == 0
    assert (dry["reports"], dry["rows_matched"], dry["rows_unmatched"], dry["pdf_missing"]) == (3, 4, 2, 1)

    st = bi.backfill_game_dates(conn)
    assert (st["reports_done"], st["by_order"], st["by_key"], st["pdf_missing"]) == (2, 1, 1, 1)
    assert (st["rows_matched"], st["rows_unmatched"]) == (4, 2)
    q = "SELECT player_name, status, game_date FROM injury_snapshots ORDER BY id"
    got = [tuple(r) for r in conn.execute(q)]
    assert got == [("A", "Out", "2026-01-14"), ("B", "Out", "2026-01-14"),
                   ("B", "Questionable", "2026-01-15"), ("C", "Out", None), ("D", "Out", "2026-01-16"),
                   ("E", "Out", None)]
    # idempotent: a second run only revisits the rows still NULL and writes nothing new
    again = bi.backfill_game_dates(conn)
    assert (again["reports"], again["rows_matched"]) == (2, 0)
    assert [tuple(r) for r in conn.execute(q)] == got


def test_backfill_relocates_a_pdf_archived_under_this_data_dir(conn, monkeypatch):
    slot = datetime(2026, 2, 19, 17, 0, tzinfo=ET)
    here = bi.report_path(slot)                                 # under COURTSIDE_DATA
    here.parent.mkdir(parents=True, exist_ok=True)
    here.write_bytes(b"%PDF-x")
    other = "/Users/someone/Developer/NBA/var/raw/nba_official/injury_report/2026-02-19/" + here.name
    report(conn, "2026-02-19T22:00:00Z", [("DEN", "F", "Out", "r", None)], other)
    monkeypatch.setattr(ir, "parse_report", lambda pdf: _parsed(("DEN", "F", "Out", "r", "2026-02-20")))
    st = bi.backfill_game_dates(conn)
    assert st["pdf_missing"] == 0 and st["rows_matched"] == 1
    assert conn.execute("SELECT game_date FROM injury_snapshots").fetchone()[0] == "2026-02-20"


def test_match_game_dates_leaves_rows_without_a_parsed_date_null():
    rows = [(1, "BOS", "A", "Out", "x"), (2, "BOS", "B", "Out", "y")]
    parsed = _parsed(("BOS", "A", "Out", "x", None), ("BOS", "B", "Out", "y", "2026-01-02"))
    upd, un, how = bi.match_game_dates(rows, parsed)
    assert (upd, un, how) == ([("2026-01-02", 2)], 1, "order")


# --- the switch -------------------------------------------------------------------------------

def test_day_before_5pm_report_is_the_latest_batch_for_the_next_days_game_at_10_et_when_on(conn):
    world(conn)
    n1, n2 = name("AAA", 1), name("AAA", 2)
    report(conn, T_5PM_PREV, [("AAA", n1, "Questionable", "Injury/Illness - Left Knee; Soreness", DAY.isoformat()),
                              ("AAA", n2, "Out", "Injury/Illness - Left Foot; Sprain", DAY.isoformat())])
    # ON: keyed to the game day, an ordered official batch 26 h before the tip
    av.set_inj_gameday(True)
    on = av.p_play_slate(conn, DAY, T_10AM)
    r1 = row(on, "AAA", 1)
    assert (r1["status"], r1["source"], r1["report_time"], r1["bucket"]) == \
        ("questionable", "nba_official", T_5PM_PREV, "gt6h")
    assert r1["hours_to_tip"] == pytest.approx(26.0)
    assert row(on, "AAA", 2)["status"] == "out"
    keyed = av.load_reports_gd(conn, DAY, DAY, av._aware(T_10AM))
    assert [b.ts for b in keyed[(DAY, "AAA")]] == [T_5PM_PREV]
    # OFF: filed under 03-19; the game day sees it only through the one-batch fallback
    av.set_inj_gameday(False)
    off = av.p_play_slate(conn, DAY, T_10AM)
    assert av.load_reports(conn, av._day_start_utc(DAY), av._aware(T_10AM)) == {}
    assert row(off, "AAA", 1)["status"] == "questionable"       # same status, via the fallback
    assert row(off, "AAA", 1)["report_time"] == T_5PM_PREV


def test_on_orders_day_before_batches_so_a_dropped_player_is_removed_and_the_10pm_report_wins(conn):
    world(conn)
    n1, n2 = name("AAA", 1), name("AAA", 2)
    report(conn, T_5PM_PREV, [("AAA", n1, "Questionable", "Injury/Illness - Knee", DAY.isoformat()),
                              ("AAA", n2, "Doubtful", "Injury/Illness - Foot", DAY.isoformat())])
    report(conn, T_10PM_PREV, [("AAA", n1, "Probable", "Injury/Illness - Knee", DAY.isoformat())])
    av.set_inj_gameday(True)
    on = av.p_play_slate(conn, DAY, T_10AM)
    assert row(on, "AAA", 1)["status"] == "probable" and row(on, "AAA", 1)["report_time"] == T_10PM_PREV
    assert row(on, "AAA", 2)["status"] == "removed"
    av.set_inj_gameday(False)
    off = av.p_play_slate(conn, DAY, T_10AM)
    assert row(off, "AAA", 1)["status"] == "probable"
    assert row(off, "AAA", 2)["status"] == "unlisted"           # one unordered batch: no history


def test_back_to_back_player_gets_the_row_of_his_own_game_date(conn):
    world(conn)
    prev = DAY - timedelta(days=1)
    gid = "G_PREV"
    add_game(conn, gid, prev)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?,?,?,?,?,?,?)", (gid, "2025-26", prev.isoformat(), "2026-03-19T23:30:00Z",
                                            "BBB", "AAA", "final"))
    conn.commit()
    n1 = name("AAA", 1)
    # one 5 PM ET report on 03-19 lists him twice: Out tonight, Questionable tomorrow
    report(conn, T_5PM_PREV, [("AAA", n1, "Out", "Injury/Illness - Knee", prev.isoformat()),
                              ("AAA", n1, "Questionable", "Injury/Illness - Knee", DAY.isoformat())])
    av.set_inj_gameday(True)
    assert row(av.p_play_slate(conn, DAY, T_10AM), "AAA", 1)["status"] == "questionable"
    assert row(av.p_play_slate(conn, prev, "2026-03-19T23:00:00Z"), "AAA", 1)["status"] == "out"
    av.set_inj_gameday(False)
    # OFF merges the two rows (most severe wins) for both days
    assert row(av.p_play_slate(conn, DAY, T_10AM), "AAA", 1)["status"] == "out"
    assert row(av.p_play_slate(conn, prev, "2026-03-19T23:00:00Z"), "AAA", 1)["status"] == "out"


def test_off_ignores_game_date_entirely(conn):
    world(conn)
    n1, n2 = name("AAA", 1), name("AAA", 2)
    report(conn, T_10PM_PREV, [("AAA", n1, "Out", "Injury/Illness - Knee", DAY.isoformat())])
    report(conn, "2026-03-20T17:00:00Z", [("AAA", n2, "Questionable", "Rest", DAY.isoformat()),
                                          ("BBB", name("BBB", 3), "Out", "Rest", DAY.isoformat())])
    ats = (T_10AM, "2026-03-20T18:00:00Z", "2026-03-20T23:50:00Z")
    filled = [av.p_play_slate(conn, DAY, at) for at in ats]
    T = [av.build_samples(conn, DAY, DAY, mode=m) for m in ("bucket", "scenario")]
    conn.execute("UPDATE injury_snapshots SET game_date=NULL")
    conn.commit()
    for at, f in zip(ats, filled):
        assert av.p_play_slate(conn, DAY, at).equals(f)
    for m, t in zip(("bucket", "scenario"), T):
        assert av.build_samples(conn, DAY, DAY, mode=m).equals(t)


def test_unfilled_rows_keep_the_capture_day_key_when_on(conn):
    world(conn)
    report(conn, "2026-03-20T17:00:00Z", [("AAA", name("AAA", 1), "Out", "Rest", None)])
    av.set_inj_gameday(True)
    assert row(av.p_play_slate(conn, DAY, "2026-03-20T18:00:00Z"), "AAA", 1)["status"] == "out"
    assert (DAY, "AAA") in av.load_reports_gd(conn, DAY, DAY, av._aware("2026-03-20T18:00:00Z"))


def test_build_samples_on_sees_day_before_batches_and_clock_mode(conn):
    world(conn)
    n1 = name("AAA", 1)
    report(conn, T_5PM_PREV, [("AAA", n1, "Out", "Injury/Illness - Knee", DAY.isoformat())])
    report(conn, "2026-03-20T20:00:00Z", [("AAA", n1, "Out", "Injury/Illness - Knee", DAY.isoformat())])
    off = av.build_samples(conn, DAY, DAY, mode="bucket")
    assert off["report_time"].dropna().unique().tolist() == ["2026-03-20T20:00:00Z"]
    av.set_inj_gameday(True)
    on = av.build_samples(conn, DAY, DAY, mode="bucket")
    # the day-before report is the gt6h batch of the game day (26 h), the 16:00 ET one 2h_6h
    o = on[on["player_id"] == pid("AAA", 1)].set_index("bucket")
    assert o.loc["gt6h", "report_time"] == T_5PM_PREV and o.loc["2h_6h", "report_time"] == "2026-03-20T20:00:00Z"
    clock = av.build_samples(conn, DAY, DAY, mode="clock")
    assert set(clock["scenario"]) == {"ET10:00", "ET13:30"}
    c10 = clock[(clock["scenario"] == "ET10:00") & (clock["player_id"] == pid("AAA", 1))].iloc[0]
    assert c10["report_time"] == T_5PM_PREV and c10["status"] == "out"
    assert c10["at"] == datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc)
    # a game that tipped before 13:30 ET has no 13:30 row
    conn.execute("UPDATE games SET tipoff_utc='2026-03-20T16:00:00Z' WHERE game_id=?", (GAME,))
    conn.commit()
    assert set(av.build_samples(conn, DAY, DAY, mode="clock")["scenario"]) == {"ET10:00"}


def test_load_params_serves_the_gd_file_only_when_on(tmp_path, monkeypatch):
    gd = tmp_path / "gd.json"
    gd.write_text(av.AvailabilityParams(m1=20.0, m=7.0, cells={}, meta={"keying": "game_date"})
                  .save(gd).read_text())
    monkeypatch.setattr(av, "GD_PARAMS_PATH", gd)
    av._PARAMS_CACHE.clear()
    assert av.load_params().m == av.AvailabilityParams.from_json(
        __import__("json").loads(av.PARAMS_PATH.read_text())).m if av.PARAMS_PATH.exists() else True
    av.set_inj_gameday(True)
    assert av.load_params().m == 7.0 and av.load_params().meta["keying"] == "game_date"
    av.set_inj_gameday(False)
    av._PARAMS_CACHE.clear()


def test_switch_defaults_off_and_has_its_tag():
    assert av.INJ_GAMEDAY_DEFAULT is False and av.INJ_GAMEDAY_TAG == "+gd"
    assert av.INJ_GAMEDAY_ENV == "COURTSIDE_INJ_GAMEDAY"


def test_shipped_gd_params_are_a_game_day_refit_and_monotone():
    assert av.GD_PARAMS_PATH.exists()
    prm = av.load_params(av.GD_PARAMS_PATH)
    assert prm.meta["keying"] == "game_date"
    assert prm.meta["rows"] > 90_000 and prm.meta["games"] > 700
    for b in av.BUCKETS:
        ps = [prm.p_status_bucket(s, b) for s in av.LISTED]
        assert ps == sorted(ps)
    # the two files are different fits of the same window (the cells did move)
    base = av.load_params(av.PARAMS_PATH)
    assert base.meta["fit_lo"] == prm.meta["fit_lo"] and base.meta["fit_hi"] == prm.meta["fit_hi"]
    assert prm.cells["removed"][0] > base.cells["removed"][0]
