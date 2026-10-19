"""Light DraftKings capture (T0.7): starter/swappable, slate tags, contest selection, payout
tables, the daily request budget and the lock-relative refresh. Offline: fixtures are trimmed
real payloads (lobby_contests.json = the 2026-10-19 lobby for opening night, contest_196456028
= one live contest probed 2026-10-19)."""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from courtside import archive, collect, config, db
from courtside.http import FetchError
from courtside.proj.dfs import capture_cli
from courtside.proj.dfs import rules as R
from courtside.sources import dfs, dfs_capture as cap

FIX = Path(__file__).parent / "fixtures" / "dfs"
DAY = date(2026, 10, 20)
NOW = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)      # 10:00 ET, 5 h before first tip


def J(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(cap, "_sleep", lambda s: None)


@pytest.fixture
def lobby():
    return J("lobby_contests.json")


def contest_payload(cid):
    p = J("contest_196456028.json")
    p["contestDetail"]["contestKey"] = str(cid)
    return p


class Fetcher:
    def __init__(self):
        self.ids: list[str] = []
        self.fail: set[str] = set()

    def __call__(self, cid):
        self.ids.append(str(cid))
        if str(cid) in self.fail:
            raise FetchError("HTTP 500", status=500)
        return contest_payload(cid)


def seed_slates(conn, lobby, *, now=NOW):
    """dfs_slates for the five opening-night groups from the real lobby (+ fixture draftables)."""
    groups = dfs.lobby_draft_groups(lobby, day=DAY, game_types=dfs.SALARY_GAME_TYPES)
    tags = dfs.classify_slates(groups)
    for g in groups:
        sid = g["draft_group_id"]
        pl = J("draftables_classic.json" if g["game_type_id"] == 70 else "draftables_showdown.json")
        cap.store_draftables(conn, pl, slate_id=sid, game_type_id=g["game_type_id"], day=DAY,
                             tag=tags[sid], rules=J(f"rules_{g['game_type_id']}.json"),
                             lobby=lobby, start_utc=g["start_utc"], captured_at=cap.iso(now),
                             now=now)
    return tags


# --------------------------------------------------------------------------- draftables extras

def with_starters(payload, n_per_team=3):
    p = copy.deepcopy(payload)
    seen: dict[str, set] = {}
    for d in p["draftables"]:
        s = seen.setdefault(d["teamAbbreviation"], set())
        if d["playerId"] in s or len(s) < n_per_team:
            s.add(d["playerId"])
            d["playerGameAttributes"] = [{"id": 100, "value": "true"}]
    return p


def test_starter_and_swappable_parse():
    rows = dfs.parse_draftables(with_starters(J("draftables_showdown.json")),
                                draft_group_id=1, game_type_id=81, rules=J("rules_81.json"))
    flagged = {r["site_player_id"] for r in rows if r["starter"] == 1}
    assert len(flagged) == 5                  # 3 BOS + 2 DET (all the fixture has), all rows of each
    assert {r["starter"] for r in rows} == {0, 1}              # lineups out: the rest are 0
    for pid in flagged:
        assert {r["starter"] for r in rows if r["site_player_id"] == pid} == {1}
    assert {r["swappable"] for r in rows} <= {0, 1}
    assert any(r["swappable"] == 1 for r in rows) and all(r["disabled"] == 0 for r in rows)


def test_starter_is_null_until_dk_publishes_lineups():
    rows = dfs.parse_draftables(J("draftables_classic.json"), draft_group_id=1, game_type_id=70)
    assert {r["starter"] for r in rows} == {None}
    # a couple of stray flags (< 5 in a game) do not turn everyone else into a 0
    p = J("draftables_classic.json")
    p["draftables"][0]["playerGameAttributes"] = [{"id": 100, "value": "true"}]
    rows = dfs.parse_draftables(p, draft_group_id=1, game_type_id=70)
    assert {r["starter"] for r in rows} == {None, 1}
    # a flag with value "false" or another attribute id is not a starter
    p["draftables"][0]["playerGameAttributes"] = [{"id": 100, "value": "false"}, {"id": 7}]
    assert {r["starter"] for r in dfs.parse_draftables(p, draft_group_id=1,
                                                       game_type_id=70)} == {None}


def test_swappable_false_and_missing():
    p = J("draftables_showdown.json")
    p["draftables"][0]["isSwappable"] = False
    del p["draftables"][1]["isSwappable"]
    rows = dfs.parse_draftables(p, draft_group_id=1, game_type_id=81)
    assert rows[0]["swappable"] == 0 and rows[1]["swappable"] is None


def test_store_draftables_records_new_capture_when_starters_change(conn, lobby):
    seed_slates(conn, lobby)
    n0 = conn.execute("SELECT COUNT(*) FROM dfs_salaries").fetchone()[0]
    again = seed_slates(conn, lobby)                          # identical payloads: no new rows
    assert again and conn.execute("SELECT COUNT(*) FROM dfs_salaries").fetchone()[0] == n0
    later = NOW + timedelta(hours=4)
    pl = with_starters(J("draftables_showdown.json"))
    res = cap.store_draftables(conn, pl, slate_id=154825, game_type_id=81, day=DAY,
                               tag="showdown", rules=J("rules_81.json"), lobby=lobby,
                               start_utc=None, captured_at=cap.iso(later), now=later)
    assert res["inserted"] > 0 and not res["unchanged"] and res["starters"] == 5
    pool = R.build_pool(conn, 154825)
    assert sum(1 for p in pool.players if p.starter) == 5


# --------------------------------------------------------------------------- slate tags

def test_slate_tags_opening_night(lobby):
    groups = dfs.lobby_draft_groups(lobby, day=DAY, game_types=dfs.SALARY_GAME_TYPES)
    tags = dfs.classify_slates(groups)
    assert tags == {154575: "main", 154825: "showdown", 154826: "showdown",
                    154827: "late", 154829: "showdown"}
    # other game types are "other"
    assert dfs.classify_slates(dfs.lobby_draft_groups(lobby, day=DAY))[154823] == "other"


def g(gid, count, start, suffix="", gt=70, tag="Featured"):
    return {"draft_group_id": gid, "game_type_id": gt, "game_count": count,
            "start_utc": start, "suffix": suffix, "tag": tag}


def test_slate_tag_logic():
    # a normal night: Early + full main + Late + Turbo
    t = dfs.classify_slates([
        g(1, 2, "2026-11-03T23:00:00Z", " (Early)"), g(2, 9, "2026-11-04T00:00:00Z", ""),
        g(3, 3, "2026-11-04T02:00:00Z", " (Late)"), g(4, 2, "2026-11-04T02:30:00Z", " (Turbo)"),
        g(5, 1, "2026-11-04T00:00:00Z", " (MIA @ BOS)", gt=81)])
    assert t == {1: "early", 2: "main", 3: "late", 4: "turbo", 5: "showdown"}
    # keyword-less subsets: before -> early, after -> late, same start fewer games -> other
    t = dfs.classify_slates([g(1, 8, "2026-11-04T00:00:00Z"), g(2, 3, "2026-11-03T23:00:00Z"),
                             g(3, 3, "2026-11-04T02:00:00Z"), g(4, 5, "2026-11-04T00:00:00Z")])
    assert t == {1: "main", 2: "early", 3: "late", 4: "other"}
    # the only Classic group has a suffix: still the main slate
    assert dfs.classify_slates([g(1, 4, "2026-11-04T00:00:00Z", " (Night)")]) == {1: "main"}
    # tie on games: the Featured one, then the earliest
    assert dfs.classify_slates([g(1, 4, "2026-11-04T01:00:00Z", tag=""),
                                g(2, 4, "2026-11-04T02:00:00Z")])[2] == "main"
    assert dfs.classify_slates([g(1, 4, "2026-11-04T02:00:00Z"),
                                g(2, 4, "2026-11-04T01:00:00Z")])[2] == "main"


# --------------------------------------------------------------------------- selection

def test_main_contest_selection_on_real_lobby(lobby):
    all_c = dfs.lobby_contests(lobby, 154575)
    assert len(all_c) == 493
    sel = dfs.select_main_contests(all_c)
    ids = [c["contest_id"] for c in sel]
    assert len(ids) == len(set(ids)) and 20 <= len(sel) <= dfs.MAIN_CONTEST_CAP
    for c in sel:                                 # filters: no free, no tiny, no leagues / satellites
        assert c["entry_fee"] >= 3 and c["field_size"] >= 100 and c["prizes"] > 0
        assert not c["attrs"].get("League") and not c["attrs"].get("IsQualifier")
        assert c["draft_group_id"] == 154575
    by = {c["contest_id"]: c for c in sel}
    kinds = {}
    for c in sel:
        for k in c["selected_as"]:
            kinds[k] = kinds.get(k, 0) + 1
    assert kinds["top_gpp"] == 10 and kinds["double_up"] == 8     # every eligible double-up
    assert kinds["single_entry"] >= 3 and kinds["three_max"] >= 1
    top = sorted((c for c in all_c if dfs.eligible_contest(c) and not c["double_up"]),
                 key=lambda c: -c["prizes"])[:10]
    assert {c["contest_id"] for c in top} <= set(by)
    assert by["196456057"]["prizes"] == 600000 and by["196456057"]["selected_as"] == ["top_gpp"]
    # a single-entry and a 3-max GPP are in even when outside the top 10
    assert any(c["max_per_user"] == 1 and "single_entry" in c["selected_as"] and
               "top_gpp" not in c["selected_as"] for c in sel)
    assert all(c["max_per_user"] == 3 for c in sel if c["selected_as"] == ["three_max"])
    # double-ups of every fee on offer
    assert {c["entry_fee"] for c in sel if "double_up" in c["selected_as"]} == {3, 5, 10, 25}


def test_selection_caps_and_fee_spread():
    cs = []
    for i in range(30):                         # 30 double-ups across 6 fees
        cs.append({"contest_id": f"d{i}", "name": "DU", "entry_fee": float([3, 5, 10, 25, 50, 100][i % 6]),
                   "field_size": 1000 - i, "max_per_user": 1, "prizes": 1000.0,
                   "draft_group_id": 1, "game_type_id": 70, "double_up": True,
                   "guaranteed": True, "attrs": {"IsDoubleUp": "true"}})
    for i in range(60):                         # 60 GPPs
        cs.append({"contest_id": f"g{i}", "name": "GPP", "entry_fee": 20.0, "field_size": 500,
                   "max_per_user": [1, 3, 20][i % 3], "prizes": 10000.0 + i, "draft_group_id": 1,
                   "game_type_id": 70, "double_up": False, "guaranteed": True, "attrs": {}})
    sel = dfs.select_main_contests(cs)
    assert len(sel) == dfs.MAIN_CONTEST_CAP
    du = [c for c in sel if "double_up" in c["selected_as"]]
    assert len(du) == dfs.MAIN_DOUBLE_UP_CAP
    assert {c["entry_fee"] for c in du} == {3, 5, 10, 25, 50, 100}     # a spread, not 14 cheap ones
    assert sum("top_gpp" in c["selected_as"] for c in sel) == 10
    assert sum("single_entry" in c["selected_as"] for c in sel) <= dfs.MAIN_SINGLE_ENTRY_CAP + 10
    small = dfs.select_main_contests(cs, cap=12)
    assert len(small) == 12 and sum("top_gpp" in c["selected_as"] for c in small) == 10


def test_ineligible_contests():
    base = {"contest_id": "1", "name": "x", "entry_fee": 5.0, "field_size": 500, "max_per_user": 1,
            "prizes": 100.0, "draft_group_id": 1, "game_type_id": 70, "double_up": False,
            "guaranteed": True, "attrs": {}}
    assert dfs.eligible_contest(base)
    assert not dfs.eligible_contest({**base, "entry_fee": 2.0})
    assert not dfs.eligible_contest({**base, "entry_fee": 0.0})
    assert not dfs.eligible_contest({**base, "field_size": 99})
    for flag in ("League", "IsQualifier", "IsBeginner", "IsWinnerTakeAll", "Multiplier"):
        assert not dfs.eligible_contest({**base, "attrs": {flag: "true"}})


def test_showdown_selection_is_one_per_slate_and_at_most_three(lobby):
    sd = [154825, 154826, 154829]
    sel = dfs.select_showdown_contests(lobby, sd, limit=dfs.MAX_SHOWDOWN_CONTESTS)
    assert [c["draft_group_id"] for c in sel] == [154829, 154825, 154826]    # richest first
    assert all(c["selected_as"] == ["showdown"] and c["entry_fee"] >= 3 for c in sel)
    assert len(dfs.select_showdown_contests(lobby, sd, limit=2)) == 2
    assert dfs.select_showdown_contests(lobby, sd, limit=0) == []
    many = copy.deepcopy(lobby)
    for k in range(5):                                     # 5 more showdown slates
        for c in [c for c in many["Contests"] if c["dg"] == 154825][:3]:
            many["Contests"].append({**c, "id": c["id"] + 1000 * (k + 1), "dg": 900 + k})
    assert len(dfs.select_showdown_contests(many, sd + [900 + k for k in range(5)],
                                            limit=3)) == 3


# --------------------------------------------------------------------------- payouts

def test_payout_normalization_from_real_contest():
    p = J("contest_196456028.json")
    tiers = dfs.normalize_payouts(p["contestDetail"])
    assert len(tiers) == 20
    assert tiers[0] == {"min_pos": 1, "max_pos": 1, "payout": 10000.0}
    assert tiers[6] == {"min_pos": 7, "max_pos": 8, "payout": 350.0}
    assert tiers[-1] == {"min_pos": 1826, "max_pos": 3325, "payout": 8.0}
    assert all(a["max_pos"] + 1 == b["min_pos"] for a, b in zip(tiers, tiers[1:]))
    paid = sum((t["max_pos"] - t["min_pos"] + 1) * t["payout"] for t in tiers)
    assert paid == p["contestDetail"]["totalPayouts"] == 60000.0
    row = dfs.parse_contest(p, lobby_row={"attrs": {"IsGuaranteed": "true"},
                                          "selected_as": ["top_gpp"]},
                            slate_id=154575, captured_at="2026-10-20T14:00:00Z", raw_path="r")
    assert row["contest_id"] == "196456028" and row["slate_id"] == "154575"
    assert (row["entry_fee"], row["field_size"], row["max_per_user"]) == (5, 14268, 150)
    assert row["guaranteed"] == 1 and row["total_prizes"] == 60000.0
    assert json.loads(row["payout_json"]) == tiers
    attrs = json.loads(row["attrs_json"])
    assert attrs["selected_as"] == ["top_gpp"] and attrs["lobby"] == {"IsGuaranteed": "true"}
    assert attrs["paid_places"] == 3325 and attrs["payout_sum"] == 60000.0
    assert attrs["entries_at_capture"] == 1711 and "logo" not in row["attrs_json"].lower()


def test_double_up_and_ticket_tiers():
    du = {"contestDetail": {"contestKey": "7", "entryFee": 5, "maximumEntries": 459,
                            "maximumEntriesPerUser": 1, "isGuaranteed": True,
                            "totalPayouts": 2000.0, "draftGroupId": 9, "name": "NBA $5 Double Up",
                            "payoutSummary": [{"minPosition": 1, "maxPosition": 200,
                                               "tierPayoutDescriptions": {"Cash": "$10.00"},
                                               "payoutDescriptions": [{"value": 10.0,
                                                                       "payoutDescription": "$10.00"}]}]}}
    assert dfs.normalize_payouts(du["contestDetail"]) == [
        {"min_pos": 1, "max_pos": 200, "payout": 10.0}]
    tix = {"payoutSummary": [
        {"minPosition": 1, "maxPosition": 1, "tierPayoutDescriptions": {"Cash": "$1,500.00"},
         "payoutDescriptions": [{"value": 0, "payoutDescription": "$1,500.00"}]},
        {"minPosition": 2, "maxPosition": 5, "tierPayoutDescriptions": {"Ticket": "NFL Ticket"},
         "payoutDescriptions": [{"value": 0, "payoutDescription": "NFL Ticket"}]}]}
    t = dfs.normalize_payouts(tix)
    assert t[0]["payout"] == 1500.0 and "non_cash" not in t[0]
    assert t[1]["payout"] == 0.0 and t[1]["non_cash"] == "NFL Ticket"
    with pytest.raises(ValueError):
        dfs.parse_contest({"contestDetail": {}, "errorStatus": {}}, captured_at="x")


# --------------------------------------------------------------------------- slates table

def test_dfs_slates_rows(conn, lobby):
    tags = seed_slates(conn, lobby)
    rows = {r["slate_id"]: dict(r) for r in conn.execute("SELECT * FROM dfs_slates")}
    assert set(rows) == {str(k) for k in tags}
    main = rows["154575"]
    assert (main["tag"], main["game_type_id"], main["nba_day"], main["site"]) == (
        "main", 70, "2026-10-20", "dk")
    assert main["lock_utc"] == "2026-10-20T19:00:00Z"      # earliest competition in the fixture
    assert rows["154825"]["tag"] == "showdown" and rows["154827"]["tag"] == "late"
    assert main["raw_path"] and Path(main["raw_path"]).exists() and main["captured_at"]
    assert R.Rules.from_payload(json.loads(main["rules_json"])).salary_cap == 50000
    assert R.load_rules(conn, 154575).slot_names == R.Rules.from_payload(J("rules_70.json")).slot_names
    seed_slates(conn, lobby)                                # upsert, not duplicate
    assert conn.execute("SELECT COUNT(*) FROM dfs_slates").fetchone()[0] == 5


# --------------------------------------------------------------------------- contest capture

def test_capture_contests_end_to_end(conn, lobby):
    seed_slates(conn, lobby)
    f = Fetcher()
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f)
    assert out["main_slate"] == "154575" and sorted(out["showdown_slates"]) == [
        "154825", "154826", "154829"]
    assert 20 <= out["stored"] <= dfs.DAY_CONTEST_CAP
    assert out["stored"] == out["selected"] == out["requests"]
    assert len(f.ids) == len(set(f.ids)) == out["stored"] and not out["failed"]
    rows = conn.execute("SELECT * FROM dfs_contests").fetchall()
    assert len(rows) == out["stored"]
    assert sum(1 for r in rows if r["slate_id"] in ("154825", "154826", "154829")) == 3
    assert all(r["slate_id"] in ("154575", "154825", "154826", "154829") for r in rows)
    assert all(r["raw_path"] and Path(r["raw_path"]).exists() for r in rows)
    assert all(json.loads(r["payout_json"]) for r in rows)
    assert out["fees"][0] >= 3
    # archive-first: the raw payload is on disk under raw/draftkings/contest/<day>/
    assert len(list((config.data_dir() / "raw/draftkings/contest/2026-10-20").glob("*.json.gz"))
               ) == out["stored"]
    assert cap.requests_today(conn, NOW)["by_kind"] == {"contest": out["requests"]}

    # fetch once: a second run (and a third, later) requests nothing and stores nothing
    f2 = Fetcher()
    again = cap.capture_contests(conn, lobby, DAY, now=NOW + timedelta(hours=3.5), pause=0,
                                 fetch=f2)
    assert f2.ids == [] and again["selected"] == 0 and again["stored"] == 0
    assert conn.execute("SELECT COUNT(*) FROM dfs_contests").fetchone()[0] == out["stored"]


def test_capture_skips_already_captured_ids(conn, lobby):
    seed_slates(conn, lobby)
    f = Fetcher()
    cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f)
    keep = f.ids[0]
    conn.execute("DELETE FROM dfs_contests WHERE contest_id=?", (keep,))
    conn.commit()
    f2 = Fetcher()
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f2)
    assert out["stored"] == 1 and out["reused_archive"] == 1 and f2.ids == []   # from the archive


def test_capture_only_before_lock(conn, lobby):
    seed_slates(conn, lobby)
    f = Fetcher()
    after = datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)      # exactly at the lock
    out = cap.capture_contests(conn, lobby, DAY, now=after, pause=0, fetch=f)
    assert f.ids == [] and out["selected"] == 0 and out["main_slate"] is None
    assert conn.execute("SELECT COUNT(*) FROM dfs_contests").fetchone()[0] == 0
    # a locked showdown is skipped while an open main is still captured
    conn.execute("UPDATE dfs_slates SET lock_utc='2026-10-20T10:00:00Z' WHERE tag='showdown'")
    conn.commit()
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f)
    assert out["stored"] > 0 and out["showdown_slates"] == []
    assert conn.execute("SELECT COUNT(*) FROM dfs_contests WHERE slate_id != '154575'"
                        ).fetchone()[0] == 0


def test_daily_budget_is_enforced(conn, lobby):
    seed_slates(conn, lobby)
    for _ in range(cap.DAILY_ADDED_BUDGET - 3):             # 57 used today (ET)
        cap.log_request(conn, "contest", "x", now=NOW, ok=True)
    f = Fetcher()
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f)
    assert len(f.ids) == 3 and out["stored"] == 3 and out["stopped"].startswith("budget")
    assert cap.budget_remaining(conn, NOW) == 0
    assert cap.requests_today(conn, NOW)["added"] == cap.DAILY_ADDED_BUDGET
    # base requests do not draw on it, and the next ET day starts fresh
    cap.log_request(conn, "lobby", now=NOW, ok=True)
    assert cap.requests_today(conn, NOW)["total"] == cap.DAILY_ADDED_BUDGET + 1
    assert cap.budget_remaining(conn, NOW + timedelta(days=1)) == cap.DAILY_ADDED_BUDGET
    f2 = Fetcher()
    cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f2)
    assert f2.ids == []                                   # still spent: nothing sent
    # ET day, not UTC: 03:00Z on the 21st is still the 20th in New York
    assert cap.et_day(datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)) == "2026-10-20"


def test_failures_do_not_raise_and_streak_stops(conn, lobby):
    seed_slates(conn, lobby)
    f = Fetcher()
    out0 = dfs.select_main_contests(dfs.lobby_contests(lobby, 154575))
    f.fail = {c["contest_id"] for c in out0[:2]}
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f)
    assert len(out["failed"]) == 2 and out["stored"] > 0 and out["stopped"] is None
    # everything failing stops after 3 in a row
    conn.execute("DELETE FROM dfs_contests")
    conn.commit()
    for p in (config.data_dir() / "raw/draftkings/contest/2026-10-20").glob("*"):
        p.unlink()
    f3 = Fetcher()
    f3.fail = {c["contest_id"] for c in dfs.lobby_contests(lobby)}
    out = cap.capture_contests(conn, lobby, DAY, now=NOW, pause=0, fetch=f3)
    assert out["stored"] == 0 and len(out["failed"]) == 3 and "consecutive" in out["stopped"]


def test_counted_fetch_retries_only_transient(conn):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise FetchError("HTTP 503", status=503)
        return {"ok": 1}

    assert cap.counted_fetch(conn, "refresh", "a", flaky, now=NOW, attempts=2) == {"ok": 1}
    assert len(calls) == 2 and cap.requests_today(conn, NOW)["added"] == 2   # both attempts count

    def gone():
        raise FetchError("404", status=404)

    with pytest.raises(FetchError):
        cap.counted_fetch(conn, "refresh", "b", gone, now=NOW, attempts=3)
    assert cap.requests_today(conn, NOW)["added"] == 3                         # no retry on 404


# --------------------------------------------------------------------------- refresh

def test_refresh_slate(conn, lobby):
    seed_slates(conn, lobby)
    base = conn.execute("SELECT COUNT(*) FROM dfs_salaries WHERE slate_id='154825'").fetchone()[0]
    later = NOW + timedelta(hours=8)
    changed = with_starters(J("draftables_showdown.json"))
    changed["draftables"][0]["status"] = "O"
    calls = []

    def fetch(sid):
        calls.append(sid)
        return copy.deepcopy(changed)

    r = cap.refresh_slate(conn, 154825, reason="lock45", now=later, fetch=fetch)
    assert r["status"] == "ok" and r["inserted"] == base and not r["unchanged"] and calls == ["154825"]
    assert r["starters"] == 5 and r["budget_remaining"] == cap.DAILY_ADDED_BUDGET - 1
    assert conn.execute("SELECT COUNT(*) FROM dfs_salaries WHERE slate_id='154825'"
                        ).fetchone()[0] == 2 * base
    pool = R.build_pool(conn, 154825)
    assert sum(1 for p in pool.players if p.starter) == 5 and pool.rules is not None
    # idempotent: the same reason again is a no-op without a request ...
    r2 = cap.refresh_slate(conn, 154825, reason="lock45", now=later + timedelta(seconds=30),
                           fetch=fetch)
    assert r2["status"] == "recent" and calls == ["154825"]
    # ... a forced / later one requests but finds nothing new, so stores nothing
    r3 = cap.refresh_slate(conn, 154825, reason="lock45", now=later + timedelta(minutes=5),
                           fetch=fetch)
    assert r3["status"] == "ok" and r3["unchanged"] and r3["inserted"] == 0
    assert conn.execute("SELECT COUNT(*) FROM dfs_salaries WHERE slate_id='154825'"
                        ).fetchone()[0] == 2 * base
    assert cap.requests_today(conn, later)["by_kind"]["refresh"] == 2
    # the slate row keeps its tag / rules / lock
    s = conn.execute("SELECT * FROM dfs_slates WHERE slate_id='154825'").fetchone()
    assert s["tag"] == "showdown" and s["rules_json"] and s["lock_utc"]


def test_refresh_budget_and_errors(conn, lobby):
    seed_slates(conn, lobby)
    assert cap.refresh_slate(conn, 999, reason="x", now=NOW)["status"] == "unknown_slate"

    def boom(sid):
        raise FetchError("HTTP 500", status=500)

    r = cap.refresh_slate(conn, 154825, reason="lock10", now=NOW, fetch=boom)
    assert r["status"] == "error" and cap.requests_today(conn, NOW)["added"] == cap.REFRESH_ATTEMPTS
    for _ in range(cap.DAILY_ADDED_BUDGET):
        cap.log_request(conn, "contest", "x", now=NOW, ok=True)
    called = []
    r = cap.refresh_slate(conn, 154825, reason="lock10", now=NOW, force=True,
                          fetch=lambda s: called.append(s))
    assert r["status"] == "budget_exhausted" and called == []


def test_capture_cli(conn, lobby, monkeypatch, capsys):
    seed_slates(conn, lobby)
    monkeypatch.setattr(config, "today_nba_day", lambda: DAY)
    sub = __import__("argparse").ArgumentParser().add_subparsers()
    handlers = capture_cli.add_commands(sub)
    assert set(handlers) == {"refresh", "capture-status"}
    monkeypatch.setattr(capture_cli, "_conn", lambda: conn)
    args = type("A", (), {"date": None})()
    assert handlers["capture-status"](args) == 0
    out = capsys.readouterr().out
    assert "154575" in out and "main" in out and "showdown" in out
    monkeypatch.setattr(cap, "refresh_slate", lambda c, s, *, reason, force: {
        "status": "ok", "slate_id": s, "reason": reason})
    assert handlers["refresh"](type("A", (), {"slate": "154575", "reason": "lock10",
                                              "force": False})()) == 0


# --------------------------------------------------------------------------- collect step

def _stub(monkeypatch, lobby, fetcher):
    from pretip_helpers import NOW as PNOW
    monkeypatch.setattr(collect, "_utcnow", lambda: PNOW)
    monkeypatch.setattr(config, "today_nba_day", lambda: DAY)
    monkeypatch.setattr(dfs, "fetch_lobby", lambda: copy.deepcopy(lobby))
    monkeypatch.setattr(dfs, "fetch_draftables", lambda sid: J(
        "draftables_classic.json" if int(sid) == 154575 else "draftables_showdown.json"))
    monkeypatch.setattr(dfs, "fetch_rules", lambda gt: J(f"rules_{int(gt)}.json"))
    monkeypatch.setattr(dfs, "fetch_contest", fetcher)
    monkeypatch.setattr(cap, "CONTEST_PAUSE_S", 0)


def test_collect_step_captures_slates_and_payouts(lobby, monkeypatch):
    f = Fetcher()
    _stub(monkeypatch, lobby, f)
    s = collect.run_pregame(DAY, only={"dfs_salaries"})
    d = s["steps"]["dfs_salaries"]
    assert s["ok"] and d["status"] == "ok", d
    assert d["tags"] == {"154575": "main", "154825": "showdown", "154826": "showdown",
                         "154827": "late", "154829": "showdown"}
    assert d["payouts"]["stored"] == len(f.ids) >= 20
    # lobby + 5 draftables + 2 rules + contests; only the contests are 'added'
    assert d["requests"] == 1 + 5 + 2 + len(f.ids)
    assert d["dk_requests_today"]["added"] == len(f.ids) <= cap.DAILY_ADDED_BUDGET
    assert d["dk_requests_today"]["by_kind"]["lobby"] == 1
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM dfs_slates").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM dfs_contests").fetchone()[0] == len(f.ids)
    conn.close()

    # second pregame run: nothing new to fetch, no new salaries, no new contests
    n = len(f.ids)
    s2 = collect.run_pregame(DAY, only={"dfs_salaries"})
    d2 = s2["steps"]["dfs_salaries"]
    assert d2["inserted"] == 0 and d2["payouts"]["stored"] == 0 and len(f.ids) == n


def test_payout_failure_does_not_fail_the_salary_step(lobby, monkeypatch):
    f = Fetcher()
    _stub(monkeypatch, lobby, f)

    def boom(*a, **k):
        raise RuntimeError("selection exploded")

    monkeypatch.setattr(cap, "capture_contests", boom)
    s = collect.run_pregame(DAY, only={"dfs_salaries"})
    d = s["steps"]["dfs_salaries"]
    assert s["ok"] and d["status"] == "ok" and d["inserted"] > 0
    assert "selection exploded" in d["payouts"]["error"]
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM dfs_slates").fetchone()[0] == 5
    conn.close()


def test_all_contests_failing_does_not_fail_the_step(lobby, monkeypatch):
    f = Fetcher()
    f.fail = {c["contest_id"] for c in dfs.lobby_contests(lobby)}
    _stub(monkeypatch, lobby, f)
    s = collect.run_pregame(DAY, only={"dfs_salaries"})
    d = s["steps"]["dfs_salaries"]
    assert s["ok"] and d["status"] == "ok" and d["payouts"]["stored"] == 0
    assert "consecutive" in d["payouts"]["stopped"] and len(f.ids) == 3
