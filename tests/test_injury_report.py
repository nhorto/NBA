import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from courtside import http
from courtside.sources import injury_report as ir

ET = ZoneInfo("America/New_York")
FIX = Path(__file__).parent / "fixtures" / "injury_report"
BASE = "https://ak-static.cms.nba.com/referee/injury/"


# ------------------------------------------------------------------ URLs

@pytest.mark.parametrize("dt, name", [
    (datetime(2026, 4, 10, 17, 0, tzinfo=ET), "Injury-Report_2026-04-10_05_00PM.pdf"),
    (datetime(2026, 4, 10, 17, 22, 41, tzinfo=ET), "Injury-Report_2026-04-10_05_15PM.pdf"),
    (datetime(2026, 4, 10, 9, 45, tzinfo=ET), "Injury-Report_2026-04-10_09_45AM.pdf"),
    (datetime(2026, 4, 10, 12, 0, tzinfo=ET), "Injury-Report_2026-04-10_12_00PM.pdf"),
    (datetime(2026, 4, 10, 12, 14, tzinfo=ET), "Injury-Report_2026-04-10_12_00PM.pdf"),
    (datetime(2026, 4, 10, 0, 0, tzinfo=ET), "Injury-Report_2026-04-10_12_00AM.pdf"),
    (datetime(2026, 4, 10, 0, 30, tzinfo=ET), "Injury-Report_2026-04-10_12_30AM.pdf"),
    (datetime(2026, 4, 10, 23, 59, tzinfo=ET), "Injury-Report_2026-04-10_11_45PM.pdf"),
])
def test_report_url_formats(dt, name):
    assert ir.report_url(dt) == BASE + name


def test_report_url_converts_to_eastern():
    # 21:00Z in March (EDT) is 5pm ET
    dt = datetime(2026, 3, 15, 21, 0, tzinfo=timezone.utc)
    assert ir.report_url(dt).endswith("Injury-Report_2026-03-15_05_00PM.pdf")


def test_report_url_requires_aware():
    with pytest.raises(ValueError):
        ir.report_url(datetime(2026, 4, 10, 17, 0))


# ------------------------------------------------------------------ find_latest_report

def test_find_latest_returns_newest_published(monkeypatch):
    calls = []
    now = datetime(2026, 4, 10, 17, 40, tzinfo=ET)
    # only published up to 17:15; later slots do not exist
    published = {ir.report_url(datetime(2026, 4, 10, 17, 15, tzinfo=ET)),
                 ir.report_url(datetime(2026, 4, 10, 17, 0, tzinfo=ET))}

    def fake(url, **kw):
        calls.append(url)
        if url in published:
            return b"%PDF"
        raise http.BlockedError("403", status=403, url=url)

    monkeypatch.setattr(http, "fetch_bytes", fake)
    url, slot = ir.find_latest_report(now)
    assert url.endswith("Injury-Report_2026-04-10_05_15PM.pdf")
    assert slot == datetime(2026, 4, 10, 17, 15, tzinfo=ET)
    assert calls[0].endswith("05_30PM.pdf")  # starts at the newest grid slot <= now
    assert len(calls) == 2


def test_find_latest_falls_back_to_hourly_grid(monkeypatch):
    calls = []
    now = datetime(2026, 4, 10, 17, 40, tzinfo=ET)
    target = ir.report_url(datetime(2026, 4, 10, 11, 0, tzinfo=ET))

    def fake(url, **kw):
        calls.append(url)
        if url == target:
            return b"%PDF"
        raise http.FetchError("404", status=404, url=url)

    monkeypatch.setattr(http, "fetch_bytes", fake)
    url, _ = ir.find_latest_report(now)
    assert url == target
    assert len(calls) <= ir.MAX_PROBES
    # after the 2h fine window only :00 slots are probed
    assert all("_00" in c for c in calls[8:])


def test_find_latest_none_and_probe_cap(monkeypatch):
    calls = []

    def fake(url, **kw):
        calls.append(url)
        raise http.BlockedError("403", status=403, url=url)

    monkeypatch.setattr(http, "fetch_bytes", fake)
    assert ir.find_latest_report(datetime(2026, 10, 18, 12, 0, tzinfo=ET)) is None
    assert 0 < len(calls) <= ir.MAX_PROBES
    assert len(set(calls)) == len(calls)
    calls.clear()
    assert ir.find_latest_report(datetime(2026, 10, 18, 12, 0, tzinfo=ET), max_back_hours=1) is None
    assert len(calls) == 5  # 12:00 down to 11:00 on the 15-min grid


def test_find_latest_propagates_real_errors(monkeypatch):
    def fake(url, **kw):
        raise http.FetchError("HTTP 500", status=500, url=url)

    monkeypatch.setattr(http, "fetch_bytes", fake)
    with pytest.raises(http.FetchError):
        ir.find_latest_report(datetime(2026, 4, 10, 17, 0, tzinfo=ET))


def test_fetch_report_rejects_non_pdf(monkeypatch):
    monkeypatch.setattr(http, "fetch_bytes", lambda url, **kw: b"<xml/>")
    with pytest.raises(http.FetchError):
        ir.fetch_report("x")
    monkeypatch.setattr(http, "fetch_bytes", lambda url, **kw: b"%PDF-1.4 ...")
    assert ir.fetch_report("x").startswith(b"%PDF")


# ------------------------------------------------------------------ rows_from_table

def test_rows_from_table_carry_forward_and_not_submitted():
    table = [
        ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"],
        ["03/15/2026", "01:00 (ET)", "MIN@OKC", "Minnesota Timberwolves", "Edwards, Anthony",
         "Available", "Injury/Illness - Right Knee; Soreness"],
        ["", "", "", "", "Freeman, Enrique", "Out", "G League - Two-Way"],
        [None, None, None, "Oklahoma City Thunder", "Hartenstein, Isaiah", "Available", "-"],
        ["", "03:30 (ET)", "DAL@CLE", "LA Clippers", "Lively II, Dereck", "Questionable", "Rest"],
        ["", "", "", "Boston Celtics", "", "", ""],  # bare label, no player
        ["03/16/2026", "07:00 (ET)", "GSW@WAS", "Golden State Warriors", "NOT YET SUBMITTED", "", ""],
        ["", "", "", "Washington Wizards", "NOT YET SUBMITTED", "", ""],
        ["", "", "MEM@CHI", "Memphis Grizzlies", "", "", "NOT YET SUBMITTED"],
    ]
    rows = ir.rows_from_table(table)
    got = [(r["game_date"], r["game_time"], r["matchup"], r["team"], r["player_name"],
            r["player_name_norm"], r["status"], r["reason"]) for r in rows]
    assert got[0] == ("2026-03-15", "01:00 (ET)", "MIN@OKC", "MIN", "Edwards, Anthony",
                      "Anthony Edwards", "Available", "Injury/Illness - Right Knee; Soreness")
    assert got[1] == ("2026-03-15", "01:00 (ET)", "MIN@OKC", "MIN", "Freeman, Enrique",
                      "Enrique Freeman", "Out", "G League - Two-Way")
    assert got[2][3:] == ("OKC", "Hartenstein, Isaiah", "Isaiah Hartenstein", "Available", None)
    assert got[2][:3] == ("2026-03-15", "01:00 (ET)", "MIN@OKC")
    assert got[3][1:4] == ("03:30 (ET)", "DAL@CLE", "LAC")
    assert got[3][4:6] == ("Lively II, Dereck", "Dereck Lively II")
    # a bare team label emits nothing
    assert all(r["team"] != "BOS" for r in rows)
    ns = [r for r in rows if r["status"] == "NOT YET SUBMITTED"]
    assert [(r["team"], r["game_date"], r["player_name"], r["reason"]) for r in ns] == [
        ("GSW", "2026-03-16", None, None), ("WAS", "2026-03-16", None, None),
        ("MEM", "2026-03-16", None, None)]
    assert ns[2]["matchup"] == "MEM@CHI"
    assert all(r["source"] == "nba_official" for r in rows)


def test_rows_from_table_reason_continuation_across_pages():
    table = [
        ["03/15/2026", "07:00 (ET)", "MIA@WAS", "Washington Wizards", "Johnson, Tre", "Out",
         "Injury/Illness - Right Foot; Sprain;"],
        ["", "", "", "", "", "", "Low Back Strain"],
        ["", "", "", "", "Whitmore, Cam", "Out", "Rest"],
    ]
    rows = ir.rows_from_table(table)
    assert len(rows) == 2
    assert rows[0]["reason"] == "Injury/Illness - Right Foot; Sprain; Low Back Strain"
    assert rows[1]["team"] == "WAS" and rows[1]["reason"] == "Rest"


def test_team_and_name_helpers():
    assert ir.team_tricode("Boston Celtics") == "BOS"
    assert ir.team_tricode("LA Clippers") == "LAC"
    assert ir.team_tricode("Los Angeles Lakers") == "LAL"
    assert ir.team_tricode("Portland Trail Blazers") == "POR"
    assert ir.team_tricode("Nowhere Nobodies") is None
    assert ir.normalize_player_name("Jackson Jr., Jaren") == "Jaren Jackson Jr."
    assert ir.normalize_player_name("Nance") == "Nance"


def test_parser_unavailable_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pdfplumber":
            raise ImportError("no")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ir.ParserUnavailable):
        ir.parse_report(b"%PDF-1.4")


# ------------------------------------------------------------------ real PDFs

EXPECTED = json.loads((FIX / "expected.json").read_text())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_parse_fixture_pdf(name):
    pytest.importorskip("pdfplumber")
    assert ir.parser_available()
    rows = ir.parse_report((FIX / name).read_bytes())
    assert len(rows) == EXPECTED[name]["count"]
    by_key = {(r["matchup"], r["team"], r["player_name"], r["status"]): r for r in rows}
    for want in EXPECTED[name]["rows"]:
        key = (want["matchup"], want["team"], want["player_name"], want["status"])
        if want["player_name"] is None:
            got = next(r for r in rows if r["team"] == want["team"]
                       and r["status"] == want["status"] and r["matchup"] == want["matchup"])
        else:
            got = by_key[key]
        assert got == want
    assert all(r["team"] for r in rows)
    assert all(r["status"] in (*ir.STATUSES, ir.NOT_SUBMITTED) for r in rows)
