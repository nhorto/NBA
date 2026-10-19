"""collect jobs: everything offline, sources monkeypatched with fixtures / synthetic payloads."""

from __future__ import annotations

import copy
import fcntl
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import courtside.sources as sources_pkg
from courtside import archive, collect, config, db
from courtside.http import FetchError
from courtside.sources import espn, nba_cdn
from pretip_helpers import stub_pretip_sources

FIX = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 20)
PAST = date(2025, 10, 21)
GID = "0022500001"


def J(rel: str):
    return json.loads((FIX / rel).read_text())


class Net:
    def __init__(self):
        self.calls: Counter = Counter()
        self.schedule = J("nba_cdn/schedule.json")
        self.odds = J("nba_cdn/odds.json")
        self.odds["games"][0]["gameId"] = "0022600001"
        self.box = J("nba_cdn/boxscore_0022500001.json")
        self.pbp = J("nba_cdn/playbyplay_0022500001.json")
        self.espn_sb = J("espn/scoreboard_20261020.json")
        self.espn_inj = J("espn/injuries.json")
        self.summary = J("espn/summary_401811039.json")
        self.box_by_game: dict = {}
        self.fail: set[str] = set()
        self.box_404: set[str] = set()
        self.backfills: list = []

    def _c(self, name):
        self.calls[name] += 1
        if name in self.fail:
            raise FetchError(f"boom {name}", status=500)


@pytest.fixture
def net(monkeypatch):
    n = Net()
    monkeypatch.setattr(config, "today_nba_day", lambda: TODAY)

    def mk(mod, attr, fn):
        def wrapper(*a, **k):
            n._c(f"{mod.__name__.rsplit('.', 1)[-1]}.{attr}")
            return fn(*a, **k)
        monkeypatch.setattr(mod, attr, wrapper)

    mk(nba_cdn, "fetch_schedule", lambda: copy.deepcopy(n.schedule))
    mk(nba_cdn, "fetch_scoreboard", lambda: J("nba_cdn/scoreboard_games.json"))
    mk(nba_cdn, "fetch_odds", lambda: copy.deepcopy(n.odds))
    def box(gid):
        if gid in n.box_404:
            raise FetchError("404", status=404)
        return copy.deepcopy(n.box_by_game.get(gid, n.box))

    mk(nba_cdn, "fetch_boxscore", box)
    mk(nba_cdn, "fetch_playbyplay", lambda gid: n.pbp)
    mk(espn, "fetch_scoreboard", lambda day: n.espn_sb)
    mk(espn, "fetch_injuries", lambda: n.espn_inj)
    mk(espn, "fetch_summary", lambda eid: n.summary)
    monkeypatch.setitem(sys.modules, "courtside.sources.injury_report", None)
    monkeypatch.delattr(sources_pkg, "injury_report", raising=False)
    n.pretip = stub_pretip_sources(monkeypatch)   # props / referees / Action Network / DFS

    import courtside.backfill as backfill_mod

    def fake_backfill(seasons, **kw):
        n._c("backfill.backfill_gamelogs")
        n.backfills.append((list(seasons), kw.get("season_types"), kw.get("force")))
        return {"fetched": 4, "cached": 0, "failed": 0, "rows": {}, "errors": []}

    monkeypatch.setattr(backfill_mod, "backfill_gamelogs", fake_backfill)
    return n


def fake_ir(monkeypatch, *, available=True, parse_raises=None, rows=None, report_dt=None):
    class ParserUnavailable(RuntimeError):
        pass

    state = SimpleNamespace(fetches=0, parses=0, available=available)
    report_dt = report_dt or datetime(2026, 10, 20, 17, 0, tzinfo=config.NBA_TZ)
    rows = rows if rows is not None else [
        {"team": "BOS", "player_name": "Jayson Tatum", "player_name_norm": "jayson tatum",
         "status": "Out", "reason": "Injury/Illness - Achilles", "source": "nba_official"},
        {"team": "DET", "player_name": "Cade Cunningham", "player_name_norm": "cade cunningham",
         "status": "Questionable", "reason": "Knee", "source": "nba_official"},
    ]

    def fetch_report(url):
        state.fetches += 1
        return b"%PDF-fake"

    def parse_report(pdf):
        state.parses += 1
        if parse_raises:
            raise ParserUnavailable("no pdfplumber")
        return rows

    mod = SimpleNamespace(
        ParserUnavailable=ParserUnavailable,
        find_latest_report=lambda now=None, max_back_hours=24: ("http://x/r.pdf", report_dt),
        fetch_report=fetch_report, parse_report=parse_report,
        parser_available=lambda: state.available)
    monkeypatch.setitem(sys.modules, "courtside.sources.injury_report", mod)
    monkeypatch.setattr(sources_pkg, "injury_report", mod, raising=False)
    return state


def q(sql, *a):
    conn = db.connect()
    try:
        return [tuple(r) for r in conn.execute(sql, a).fetchall()]
    finally:
        conn.close()


def count(table):
    return q(f"SELECT COUNT(*) FROM {table}")[0][0]


def raw_files(kind_glob="*/*"):
    return sorted((config.data_dir() / "raw").glob(f"{kind_glob}/*/*"))


# ------------------------------------------------------------------ pregame

def test_pregame_happy_path(net):
    s = collect.run_pregame(TODAY)
    assert s["ok"], s["errors"]
    assert count("games") == 4  # preseason + TBD-team rows filtered by the source module
    assert q("SELECT espn_id FROM games WHERE game_id='0022600002'") == [("401909089",)]
    assert q("SELECT COUNT(*) FROM games WHERE espn_id IS NULL AND game_id LIKE '0022600%'")[0][0] \
        in (0, 1)
    # ESPN odds -> game ids resolved, raw_path set
    n_espn = s["steps"]["espn_scoreboard"]["odds_inserted"]
    assert q("SELECT COUNT(*) FROM odds_snapshots WHERE source='espn' AND raw_path IS NOT NULL "
             "AND game_id IN (SELECT game_id FROM games)")[0][0] == n_espn
    # NBA odds: fixture game 0 was rewritten to a known id; the preseason id is skipped
    assert s["steps"]["nba_odds"]["odds_inserted"] == 1
    assert len(s["steps"]["nba_odds"]["odds_skipped"]) == 1
    # one shared captured_at per run across odds + espn injuries
    stamps = {r[0] for r in q("SELECT captured_at FROM odds_snapshots UNION ALL "
                              "SELECT captured_at FROM injury_snapshots "
                              "WHERE source='espn'")}
    assert len(stamps) == 1
    assert count("injury_snapshots") == len(espn.injury_rows(net.espn_inj)) > 0
    kinds = {p.parent.parent.name for p in raw_files()}
    assert {"schedule", "scoreboard", "odds", "injuries"} <= kinds
    run = q("SELECT job, ok, finished_at FROM collect_runs")
    assert run[0][0] == "pregame" and run[0][1] == 1 and run[0][2]
    assert (config.data_dir() / "logs" / "collect.log").read_text().strip()


def test_espn_nba_matching_uses_tricode_map(net):
    collect.run_pregame(TODAY)
    got = dict(q("SELECT game_id, espn_id FROM games WHERE nba_day='2026-10-20' "
                 "OR nba_day='2026-10-21'"))
    # BOS@DET, PHI@NY (NYK), OKC@SA (SAS)
    assert got["0022600001"] == "401909088"
    assert got["0022600002"] == "401909089"
    assert got["0022600003"] == "401909090"


def test_unmatched_espn_events_logged_not_fatal(net):
    # schedule has no games for 2026-04-10: ESPN events cannot be matched
    net.espn_sb = J("espn/scoreboard_20260410.json")
    s = collect.run_pregame(date(2026, 4, 10))
    assert s["ok"]
    st = s["steps"]["espn_scoreboard"]
    assert st["matched"] == 0 and len(st["unmatched"]) == st["events"] > 0
    assert st.get("odds_inserted", 0) == 0
    assert s["steps"]["nba_scoreboard"]["status"] == "skipped"  # day != today


def test_schedule_daily_cache(net):
    collect.run_pregame(TODAY)
    s2 = collect.run_pregame(TODAY)
    assert net.calls["nba_cdn.fetch_schedule"] == 1
    assert s2["steps"]["schedule"]["cached"] is True
    assert len(list((config.data_dir() / "raw/nba_cdn/schedule" / str(TODAY)).glob(
        "scheduleLeagueV2__*.json.gz"))) == 1
    assert count("games") == 4


def test_cached_schedule_does_not_regress_final(net):
    collect.run_pregame(TODAY)
    conn = db.connect()
    conn.execute("UPDATE games SET status='final', home_score=100, away_score=90 "
                 "WHERE game_id='0022600001'")
    conn.commit()
    conn.close()
    collect.run_pregame(date(2026, 10, 21))  # not today: no live scoreboard step, cached schedule
    assert q("SELECT status, home_score, away_score FROM games WHERE game_id='0022600001'") \
        == [("final", 100, 90)]


def test_one_source_failing_does_not_stop_others(net):
    net.fail.add("espn.fetch_scoreboard")
    s = collect.run_pregame(TODAY)
    assert s["ok"] is False
    assert s["steps"]["espn_scoreboard"]["status"] == "error"
    assert [e["step"] for e in s["errors"]] == ["espn_scoreboard"]
    assert s["steps"]["nba_odds"]["status"] == "ok"
    assert s["steps"]["espn_injuries"]["rows"] > 0
    ok, detail = q("SELECT ok, detail FROM collect_runs")[0]
    assert ok == 0
    assert json.loads(detail)["errors"][0]["step"] == "espn_scoreboard"


def test_lock_contention(net):
    config.ensure_dirs()
    fh = open(config.data_dir() / ".collect.lock", "a+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert collect.run_pregame(TODAY) == {"skipped": "locked"}
        assert collect.run_postgame(PAST) == {"skipped": "locked"}
        assert sum(net.calls.values()) == 0
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    assert collect.run_pregame(TODAY)["ok"]


# ------------------------------------------------------------------ official injury report

def test_injury_report_key_dedupe(net, monkeypatch):
    st = fake_ir(monkeypatch)
    s1 = collect.run_pregame(TODAY)
    assert s1["ok"], s1["errors"]
    pdf = config.data_dir() / "raw/nba_official/injury_report/2026-10-20/Injury-Report_2026-10-20_05_00PM.pdf"
    assert pdf.read_bytes() == b"%PDF-fake"
    rows = q("SELECT team, player_name, player_id, status, captured_at FROM injury_snapshots "
             "WHERE source='nba_official' ORDER BY team")
    assert rows == [("BOS", "jayson tatum", None, "Out", "2026-10-20T21:00:00Z"),
                    ("DET", "cade cunningham", None, "Questionable", "2026-10-20T21:00:00Z")]
    s2 = collect.run_pregame(TODAY)
    assert st.fetches == 1 and st.parses == 1
    assert s2["steps"]["official_injury_report"]["status"] == "skipped"
    assert q("SELECT COUNT(*) FROM injury_snapshots WHERE source='nba_official'")[0][0] == 2


def test_injury_report_module_missing_is_graceful(net):
    s = collect.run_pregame(TODAY)  # net fixture makes the import fail
    assert s["ok"]
    assert s["steps"]["official_injury_report"]["reason"] == "module_missing"


def test_injury_report_parser_unavailable_then_reparsed_from_disk(net, monkeypatch):
    st = fake_ir(monkeypatch, available=False)
    s = collect.run_pregame(TODAY)
    assert s["ok"]
    step = s["steps"]["official_injury_report"]
    assert step["parsed"] is False and step["reason"] == "parser_unavailable"
    assert q("SELECT COUNT(*) FROM injury_snapshots WHERE source='nba_official'")[0][0] == 0
    assert st.fetches == 1
    st.available = True  # dependency installed later: parse the archived PDF, no refetch
    s = collect.run_pregame(TODAY)
    assert s["ok"] and st.fetches == 1
    assert q("SELECT COUNT(*) FROM injury_snapshots WHERE source='nba_official'")[0][0] == 2


def test_injury_report_parse_raises_parser_unavailable(net, monkeypatch):
    fake_ir(monkeypatch, parse_raises=True)
    s = collect.run_pregame(TODAY)
    assert s["ok"] and s["steps"]["official_injury_report"]["reason"] == "parser_unavailable"


def test_injury_report_none_found_is_not_error(net, monkeypatch):
    fake_ir(monkeypatch)
    monkeypatch.setattr(sys.modules["courtside.sources.injury_report"], "find_latest_report",
                        lambda *a, **k: None)
    s = collect.run_pregame(TODAY)
    assert s["ok"] and s["steps"]["official_injury_report"]["reason"] == "no_report"


# ------------------------------------------------------------------ postgame

def _seed_postgame_games():
    conn = db.connect()
    db.upsert_game(conn, game_id=GID, espn_id="401811039", season="2025-26",
                   nba_day=PAST.isoformat(), tipoff_utc="2025-10-21T23:30:00Z",
                   home="OKC", away="HOU", status="scheduled")
    conn.close()


@pytest.fixture
def post(net):
    net.espn_sb = {"events": []}
    _seed_postgame_games()
    return net


def test_postgame_archives_then_is_idempotent(post):
    s = collect.run_postgame(PAST)
    assert s["ok"], s["errors"]
    assert s["archived"] == [GID] and s["fetched"] == [GID] and s["pending"] == []
    for src, kind in (("nba_cdn", "boxscore"), ("nba_cdn", "playbyplay"), ("espn", "summary")):
        assert archive.exists_final(src, kind, PAST, GID)
    assert q("SELECT status, home_score, away_score FROM games WHERE game_id=?", GID) \
        == [("final", 125, post.box["game"]["awayTeam"]["score"])]
    before = Counter(post.calls)
    s2 = collect.run_postgame(PAST)
    assert s2["ok"] and s2["archived"] == [GID] and s2["fetched"] == []
    for name in ("nba_cdn.fetch_boxscore", "nba_cdn.fetch_playbyplay", "espn.fetch_summary"):
        assert post.calls[name] == before[name] == 1


def test_postgame_pending_postponed_and_future(post):
    conn = db.connect()
    base = dict(season="2025-26", nba_day=PAST.isoformat(), home="BOS", away="DET")
    db.upsert_game(conn, game_id="0022500002", tipoff_utc="2025-10-21T23:00:00Z",
                   status="live", **base)       # not final yet
    db.upsert_game(conn, game_id="0022500003", tipoff_utc="2999-01-01T00:00:00Z",
                   status="scheduled", **base)  # has not started: no fetch at all
    db.upsert_game(conn, game_id="0022500004", tipoff_utc="2025-10-21T23:00:00Z",
                   status="postponed", **base)
    db.upsert_game(conn, game_id="0022500005", tipoff_utc="2025-10-21T23:00:00Z",
                   status="scheduled", **base)  # boxscore 404
    conn.close()
    live = copy.deepcopy(post.box)
    live["game"]["gameStatus"] = 2
    post.box_by_game["0022500002"] = live
    post.box_404.add("0022500005")
    s = collect.run_postgame(PAST)
    assert s["ok"], s["errors"]
    assert s["archived"] == [GID]
    assert sorted(s["pending"]) == ["0022500002", "0022500003", "0022500005"]
    assert s["postponed"] == ["0022500004"]
    # pending games get no play-by-play / summary fetch
    assert post.calls["nba_cdn.fetch_playbyplay"] == 1
    assert not archive.exists_final("nba_cdn", "boxscore", PAST, "0022500002")
    assert q("SELECT status FROM games WHERE game_id='0022500002'") == [("live",)]


def test_postgame_missing_games_for_past_season_day(net):
    net.espn_sb = J("espn/scoreboard_20260410.json")
    s = collect.run_postgame(date(2026, 4, 10))
    assert s["ok"] and s["archived"] == [] and s["pending"] == []
    assert len(s["steps"]["espn_scoreboard"]["unmatched"]) > 0
    assert net.calls["nba_cdn.fetch_boxscore"] == 0


def test_postgame_espn_final_updates_game(net):
    net.espn_sb = J("espn/scoreboard_20261020.json")
    ev = net.espn_sb["events"][0]
    for c in ev["competitions"][0]["competitors"]:
        c["score"] = "101" if c["homeAway"] == "home" else "99"
    ev["status"] = {"type": {"name": "STATUS_FINAL", "state": "post", "completed": True}}
    s = collect.run_postgame(TODAY)
    assert s["steps"]["espn_scoreboard"]["matched"] == 3
    assert q("SELECT status, home_score, away_score FROM games WHERE espn_id=?", ev["id"]) \
        == [("final", 101, 99)]


# ------------------------------------------------------------------ postgame: past dates, facts, brief

def _seed_team_games_only():
    """The game exists in team_games (historical log) but there is no `games` row for it."""
    conn = db.connect()
    base = dict(game_id=GID, season="2025-26", season_type="Regular Season",
                game_date=PAST.isoformat(), min=265.0)
    db.upsert_rows(conn, "team_games", [
        {**base, "team": "OKC", "opp": "HOU", "is_home": 1, "wl": "W", "pts": 125},
        {**base, "team": "HOU", "opp": "OKC", "is_home": 0, "wl": "L", "pts": 124}])
    conn.close()


def test_postgame_past_date_falls_back_to_team_games(net):
    net.espn_sb = {"events": []}
    _seed_team_games_only()
    assert q("SELECT COUNT(*) FROM games WHERE nba_day=?", PAST.isoformat()) == [(0,)]
    s = collect.run_postgame(PAST)
    assert s["ok"], s["errors"]
    assert s["archived"] == [GID] and s["fetched"] == [GID] and s["pending"] == []
    assert s["steps"]["games"]["source"] == "team_games"
    assert archive.exists_final("nba_cdn", "boxscore", PAST, GID)
    assert archive.exists_final("nba_cdn", "playbyplay", PAST, GID)
    assert not archive.exists_final("espn", "summary", PAST, GID)    # espn event id unknown
    assert net.calls["espn.fetch_summary"] == 0
    # facts were computed from the archived payloads and stored
    ((nba_day, facts_json),) = q("SELECT nba_day, facts_json FROM game_facts WHERE game_id=?", GID)
    facts = json.loads(facts_json)
    assert nba_day == PAST.isoformat() and facts["game_id"] == GID
    assert s["steps"]["game_facts"]["computed"] == 1
    # past season: no game-log refresh; the morning-of Brief for day + 1 was built
    assert s["steps"]["refresh_gamelogs"]["status"] == "skipped" and net.backfills == []
    assert s["steps"]["brief"]["day"] == "2025-10-22"
    assert q("SELECT nba_day FROM briefs") == [("2025-10-22",)]
    assert (config.data_dir() / "briefs" / "2025-10-22.json").exists()
    # idempotent: second run fetches nothing and keeps the facts
    before = Counter(net.calls)
    s2 = collect.run_postgame(PAST)
    assert s2["ok"] and s2["fetched"] == [] and s2["archived"] == [GID]
    assert s2["steps"]["game_facts"] == {"status": "ok", "computed": 0, "already": 1, "failed": {}}
    assert net.calls["nba_cdn.fetch_boxscore"] == before["nba_cdn.fetch_boxscore"]


def test_postgame_with_no_games_anywhere_is_clean(net):
    net.espn_sb = {"events": []}
    s = collect.run_postgame(PAST)
    assert s["ok"] and s["archived"] == [] and s["steps"]["games"]["source"] == "games"
    assert s["steps"]["game_facts"]["computed"] == 0


def test_game_facts_failure_is_logged_not_fatal(post, monkeypatch):
    from courtside.analytics import summarize

    def boom(box, pbp):
        raise ValueError("bad pbp")
    monkeypatch.setattr(summarize, "game_facts", boom)
    s = collect.run_postgame(PAST)
    assert s["ok"], s["errors"]
    assert s["archived"] == [GID]
    assert list(s["steps"]["game_facts"]["failed"]) == [GID]
    assert "bad pbp" in s["steps"]["game_facts"]["failed"][GID]
    assert q("SELECT COUNT(*) FROM game_facts") == [(0,)]
    assert s["steps"]["brief"]["status"] == "ok"          # the Brief is still built


def test_gamelog_refresh_only_for_current_season(post):
    collect.run_postgame(PAST)                              # 2025-26 vs today 2026-27
    assert post.backfills == []
    post.espn_sb = J("espn/scoreboard_20261020.json")
    s = collect.run_postgame(date(2026, 10, 19))            # current season
    assert post.backfills == [(["2026-27"], ("Regular Season",), True)]
    assert s["steps"]["refresh_gamelogs"]["season"] == "2026-27"
    assert s["steps"]["brief"]["day"] == "2026-10-20"       # the morning-of Brief for day + 1


def test_gamelog_refresh_failure_is_reported_but_brief_still_built(post, monkeypatch):
    import courtside.backfill as backfill_mod
    monkeypatch.setattr(backfill_mod, "backfill_gamelogs",
                        lambda seasons, **kw: {"fetched": 1, "cached": 0, "failed": 2,
                                               "rows": {}, "errors": ["x", "y"]})
    s = collect.run_postgame(date(2026, 10, 19))
    assert not s["ok"] and s["steps"]["refresh_gamelogs"]["status"] == "error"
    assert s["steps"]["brief"]["status"] == "ok"


def test_pregame_ends_with_predictions_and_todays_brief(net):
    s = collect.run_pregame(TODAY)
    assert s["ok"], s["errors"]
    assert s["steps"]["predict"]["status"] == "ok"
    assert s["steps"]["brief"]["day"] == TODAY.isoformat()
    assert q("SELECT nba_day FROM briefs") == [(TODAY.isoformat(),)]


def test_injury_report_unnamed_placeholder_rows_do_not_break_the_step(net, monkeypatch):
    rows = [{"team": "BOS", "player_name": "Jayson Tatum", "player_name_norm": "Jayson Tatum",
             "status": "Out", "reason": "x", "source": "nba_official"},
            {"team": "NYK", "player_name": None, "player_name_norm": None,
             "status": None, "reason": "NOT YET SUBMITTED", "source": "nba_official"}]
    fake_ir(monkeypatch, rows=rows)
    s = collect.run_pregame(TODAY)
    assert s["ok"], s["errors"]
    assert s["steps"]["official_injury_report"]["rows"] == 1
    assert s["steps"]["official_injury_report"]["skipped_unnamed"] == 1
