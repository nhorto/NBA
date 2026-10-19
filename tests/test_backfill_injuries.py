"""backfill_injuries: no network. Fetch/parse are stubbed."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from courtside import backfill_injuries as bi
from courtside import collect, config, db
from courtside.http import FetchError
from courtside.sources import injury_report as ir

ET = config.NBA_TZ
PARSED = [
    {"team": "BOS", "player_name": "Tatum, Jayson", "player_name_norm": "Jayson Tatum",
     "status": "Out", "reason": "Injury/Illness - Achilles", "source": "nba_official"},
    {"team": "DET", "player_name": "Cunningham, Cade", "player_name_norm": None,
     "status": "Questionable", "reason": None, "source": "nba_official"},
    {"team": "NYK", "player_name": None, "player_name_norm": None,
     "status": "NOT YET SUBMITTED", "reason": None, "source": "nba_official"},
]


@pytest.fixture
def conn():
    c = db.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def stub_parse(monkeypatch):
    calls = []

    def parse(pdf):
        calls.append(pdf)
        return PARSED
    monkeypatch.setattr(ir, "parse_report", parse)
    return calls


class FakeServer:
    """URL -> bytes for published slots; everything else is 'missing' (None)."""

    def __init__(self, published):
        self.published = {ir.report_url(s) for s in published}
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return b"%PDF-fake" if url in self.published else None


def _team_game(conn, game_id, day, season="2025-26", stype="Regular Season"):
    conn.execute("INSERT INTO team_games(game_id, team, season, season_type, game_date) "
                 "VALUES (?,?,?,?,?)", (game_id, "BOS", season, stype, day))
    conn.commit()


# ------------------------------------------------------------------ day selection

def test_game_days_filters_season_types_and_range(conn):
    for gid, day, season, st in [
        ("1", "2025-12-21", "2025-26", "Regular Season"),
        ("2", "2025-12-22", "2025-26", "Regular Season"),
        ("3", "2025-12-22", "2025-26", "Regular Season"),     # duplicate day
        ("4", "2026-04-15", "2025-26", "PlayIn"),
        ("5", "2026-06-13", "2025-26", "Playoffs"),
        ("6", "2026-01-05", "2025-26", "Preseason"),          # not a counted type
        ("7", "2025-01-05", "2024-25", "Regular Season"),     # other season
    ]:
        _team_game(conn, gid, day, season, st)
    assert bi.game_days(conn) == [date(2025, 12, 21), date(2025, 12, 22),
                                  date(2026, 4, 15), date(2026, 6, 13)]
    assert bi.game_days(conn, start=date(2025, 12, 22), end=date(2026, 4, 15)) == [
        date(2025, 12, 22), date(2026, 4, 15)]


def test_day_slots_default_and_known_tip():
    pri, extras = bi.day_slots(date(2026, 1, 10))
    assert [(s.hour, s.minute) for s in pri] == [(13, 30), (17, 30), (18, 30)]
    assert [s.hour for s in extras] == list(range(12, 23)) and all(s.minute == 0 for s in extras)
    assert all(s.tzinfo is not None for s in pri + extras)
    # 7:00 PM ET tip (00:00Z) -> 6:45 PM report; 7:05 PM -> 7:00 PM
    assert bi.pretip_slot(date(2026, 1, 10), "2026-01-11T00:00:00Z").strftime("%H:%M") == "18:45"
    assert bi.pretip_slot(date(2026, 1, 10), "2026-01-11T00:05:00Z").strftime("%H:%M") == "19:00"


def test_first_tips_from_games_table(conn):
    conn.execute("INSERT INTO games(game_id, nba_day, tipoff_utc) VALUES ('a','2026-10-20','2026-10-20T23:00:00Z')")
    conn.execute("INSERT INTO games(game_id, nba_day, tipoff_utc) VALUES ('b','2026-10-20','2026-10-20T19:00:00Z')")
    conn.commit()
    assert bi.first_tips(conn) == {date(2026, 10, 20): "2026-10-20T19:00:00Z"}


def test_find_earliest_date_binary_search():
    cut = date(2025, 12, 22)
    probes = []

    def fetch(url):
        probes.append(url)
        d = date.fromisoformat(url.split("_")[1])
        return b"%PDF" if d >= cut else None
    assert bi.find_earliest_date(fetch) == cut
    assert len(probes) < 12 and all("_05_00PM" in u for u in probes)
    assert bi.find_earliest_date(lambda u: None) is None


# ------------------------------------------------------------------ walk-back

def test_walk_back_order():
    slot = datetime(2026, 1, 10, 13, 30, tzinfo=ET)
    assert [s.strftime("%H:%M") for s in bi.walk_back(slot)] == ["13:30", "13:15", "13:00", "12:45"]


def test_process_slot_walks_back_up_to_45_minutes(conn):
    slot = datetime(2026, 1, 10, 13, 30, tzinfo=ET)
    state, stats = bi.State(None), bi.Stats()
    srv = FakeServer([slot - timedelta(minutes=45)])
    got = bi.process_slot(conn, slot, srv, state, stats)
    assert got == slot - timedelta(minutes=45)
    assert bi.report_path(got).exists() and bi.report_path(got).name == "Injury-Report_2026-01-10_12_45PM.pdf"
    assert stats.slots_missing == 3 and stats.pdfs_new == 1


def test_process_slot_gives_up_after_45_minutes(conn):
    slot = datetime(2026, 1, 10, 13, 30, tzinfo=ET)
    srv = FakeServer([slot - timedelta(minutes=60)])
    stats = bi.Stats()
    assert bi.process_slot(conn, slot, srv, bi.State(None), stats) is None
    assert len(srv.calls) == 4 and stats.pdfs == 0 and stats.slots_missing == 4


def test_fetcher_treats_403_as_missing_and_other_errors_as_errors():
    def get(url):
        if "bad" in url:
            raise FetchError("HTTP 500", status=500, url=url)
        raise FetchError("403", status=403, url=url)
    f = bi.Fetcher(get=get, sleep=lambda s: None)
    assert f("http://x/missing.pdf") is None
    with pytest.raises(FetchError):
        f("http://x/bad.pdf")


# ------------------------------------------------------------------ resumability

def test_run_is_resumable(conn, tmp_path, stub_parse):
    for i, d in enumerate(["2026-01-10", "2026-01-11"]):
        _team_game(conn, str(i), d)
    pri10, _ = bi.day_slots(date(2026, 1, 10))
    pri11, _ = bi.day_slots(date(2026, 1, 11))
    # 01-10 has 13:30 (via 13:15), 17:30, 18:30; 01-11 has nothing at all
    srv = FakeServer([pri10[0] - timedelta(minutes=15), pri10[1], pri10[2]])
    state_path = tmp_path / "state.jsonl"
    kw = dict(start=date(2026, 1, 10), extras=False, conn=conn, state_path=state_path)
    r1 = bi.run(fetch=srv, **kw)
    assert r1["pdfs_new"] == 3 and r1["days_captured"] == 1 and r1["game_days"] == 2
    assert r1["days_none"] == ["2026-01-11"] and r1["coverage_pct"] == 50.0
    assert r1["rows"] == 6 and r1["unnamed_skipped"] == 3 and r1["parse_failures"] == 0
    n_rows = conn.execute("SELECT COUNT(*) FROM injury_snapshots").fetchone()[0]
    assert n_rows == 6 and len(stub_parse) == 3

    srv2 = FakeServer([pri10[0] - timedelta(minutes=15), pri10[1], pri10[2]])
    r2 = bi.run(fetch=srv2, **kw)
    assert srv2.calls == []                      # nothing re-fetched, even known-missing slots
    assert r2["pdfs_new"] == 0 and r2["pdfs_existing"] == 3 and r2["rows"] == 0
    assert conn.execute("SELECT COUNT(*) FROM injury_snapshots").fetchone()[0] == n_rows
    assert len(stub_parse) == 3                  # no re-parse either


def test_archived_but_unparsed_pdf_is_parsed_without_refetch(conn, stub_parse):
    slot = datetime(2026, 1, 10, 17, 30, tzinfo=ET)
    bi.archive.save_raw_bytes(bi.SOURCE, bi.KIND, "2026-01-10", bi.report_key(slot), b"%PDF-x", ext=".pdf")
    srv = FakeServer([slot])
    bi.process_slot(conn, slot, srv, bi.State(None), bi.Stats())
    assert srv.calls == [] and len(stub_parse) == 1
    assert conn.execute("SELECT COUNT(*) FROM injury_snapshots").fetchone()[0] == 2


def test_parse_failure_keeps_pdf_and_is_counted(conn, monkeypatch):
    def boom(pdf):
        raise ir.ReportParseError("no table")
    monkeypatch.setattr(ir, "parse_report", boom)
    slot = datetime(2026, 1, 10, 17, 30, tzinfo=ET)
    stats = bi.Stats()
    bi.process_slot(conn, slot, FakeServer([slot]), bi.State(None), stats)
    assert stats.parse_failures == 1 and bi.report_path(slot).exists()
    assert conn.execute("SELECT COUNT(*) FROM injury_snapshots").fetchone()[0] == 0


def test_extras_respect_pdf_cap(conn):
    _team_game(conn, "1", "2026-01-10")
    pri, extras = bi.day_slots(date(2026, 1, 10))
    srv = FakeServer(pri + extras)
    r = bi.run(start=date(2026, 1, 10), conn=conn, fetch=srv, max_pdfs=5, state_path=None)
    assert r["pdfs_new"] == 5


# ------------------------------------------------------------------ parity with collect.py

def test_rows_match_collect_official_step(conn, monkeypatch):
    report_dt = datetime(2026, 1, 10, 17, 30, tzinfo=ET)
    monkeypatch.setattr(ir, "find_latest_report", lambda *a, **k: ("http://x/r.pdf", report_dt))
    monkeypatch.setattr(ir, "fetch_report", lambda url: b"%PDF-fake")
    monkeypatch.setattr(ir, "parser_available", lambda: True)
    cols = "captured_at, source, team, player_name, player_id, status, reason, raw_path"
    q = f"SELECT {cols} FROM injury_snapshots ORDER BY team"

    out = collect._official_report_step(conn, date(2026, 1, 10))
    live = conn.execute(q).fetchall()
    path = bi.report_path(report_dt)
    assert out["path"] == str(path) and path.read_bytes() == b"%PDF-fake"

    conn.execute("DELETE FROM injury_snapshots")
    conn.commit()
    path.unlink()
    got = bi.process_slot(conn, report_dt, FakeServer([report_dt]), bi.State(None), bi.Stats())
    assert got == report_dt
    assert conn.execute(q).fetchall() == live
    assert [tuple(r)[:2] for r in live] == [("2026-01-10T22:30:00Z", "nba_official")] * 2
    assert {r["player_name"] for r in live} == {"Jayson Tatum", "Cunningham, Cade"}
