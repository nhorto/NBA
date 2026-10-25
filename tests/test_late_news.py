"""Late-news tag (D4, `proj/dfs/late_news_tag.py`): news detection from the injury report,
beneficiaries from the run-to-run projection diff, storage on the run, the D5 multiplier, and
the tag on the DFS pool and the Tonight cards."""

from __future__ import annotations

import json
import re
from datetime import timedelta

import pytest

from courtside import db
from courtside.proj.dfs import late_news_tag as L

DAY = "2026-10-20"
SL = "7001"
LOCK = "2026-10-20T23:00:00Z"
MV = "pathA-test"

# NBA id: (DK id, name, team, base (dk, min, p_play), current (dk, min, p_play))
P = {
    1: ("d1", "Xavier Star", "AAA", (40.0, 32.0, 0.97), (0.1, 32.0, 0.0)),     # ruled out
    2: ("d2", "Tom One", "AAA", (20.0, 24.0, 0.97), (25.0, 30.0, 0.97)),       # +5 DK
    3: ("d3", "Tim Two", "AAA", (18.0, 22.0, 0.97), (21.5, 24.0, 0.97)),       # +3.5 DK
    4: ("d4", "Ted Three", "AAA", (10.0, 15.0, 0.97), (11.0, 18.1, 0.97)),     # +3.1 min only
    5: ("d5", "Tad Four", "AAA", (12.0, 18.0, 0.97), (13.0, 19.0, 0.97)),      # +1: no
    6: ("d6", "Tob Five", "AAA", (3.0, 12.0, 0.30), (13.0, 20.0, 0.30)),       # P(play) 0.3
    7: ("d7", "Tug Six", "AAA", (15.0, 20.0, 0.97), (19.0, 23.0, 0.97)),       # +4 DK
    8: ("d8", "Yuri Old", "AAA", (0.1, 20.0, 0.0), (0.1, 20.0, 0.0)),          # long-term Out
    9: ("d9", "Bob Other", "BBB", (30.0, 30.0, 0.97), (36.0, 34.0, 0.97)),     # other team
}


def _proj(conn, run_id: str, created: str, which: int, mv: str = MV, over=None) -> None:
    for pid, (dk_id, name, team, base, cur) in P.items():
        dk, mins, pp = (over or {}).get(pid, (base, cur)[which])
        c = conn.execute(
            "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, team, run_id,"
            " created_at, model_version, role, p_play, inputs_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("G1", DAY, SL, pid, team, run_id, created, mv, "role", pp,
             json.dumps({"dk_id": dk_id})))
        for stat, cond, mean in (("dk", "all", dk), ("dk", "played", dk / max(pp, 0.05)),
                                 ("min", "played", mins), ("min", "all", mins * pp)):
            conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                         "VALUES (?,?,?,?)", (c.lastrowid, stat, cond, mean))


def _run(conn, run_id, kind, created, mv=MV, detail=None):
    conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok, "
                 "detail) VALUES (?,?,?,?,?,1,?)", (run_id, SL, kind, created, mv,
                                                    json.dumps(detail) if detail else None))


def _inj(conn, source, at, team, name, status, game_date=None):
    db.insert_injury_snapshots(conn, [{"captured_at": at, "source": source, "team": team,
                                       "player_name": name, "status": status,
                                       "game_date": game_date}])


@pytest.fixture
def conn(data_dir):
    L._CACHE.clear()
    c = db.connect(data_dir / "courtside.db")
    c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
              "lock_utc) VALUES (?,?,70,'main','[\"G1\"]',?)", (SL, DAY, LOCK))
    for pid, (dk_id, name, team, _, _) in P.items():
        c.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, draftable_id, "
                  "player_name, site_player_id, team, roster_slot, salary, start_time) VALUES "
                  "('2026-10-20T12:00:00Z','dk',?,'classic',?,?,?,?,'UTIL',5000,?)",
                  (SL, "D" + dk_id, name, dk_id, team, LOCK))
    _run(c, "rb", "lock45", "2026-10-20T22:15:00Z")
    _proj(c, "rb", "2026-10-20T22:15:00Z", 0)
    _run(c, "rc", "lock10", "2026-10-20T22:50:00Z")
    _proj(c, "rc", "2026-10-20T22:50:00Z", 1)
    for d, (own, band) in (("d2", (4.0, "Low")), ("d3", (12.0, "Med")), ("d7", (2.0, "Low"))):
        c.execute("INSERT INTO dfs_ownership_proxy (run_id, player_id, own, band, version) "
                  "VALUES ('rc', ?, ?, ?, 't')", (d, own, band))
    # injury report: Yuri Out all day (old news); Xavier listed Out at 22:30 (30 min to lock)
    for at in ("2026-10-20T14:00:00Z", "2026-10-20T21:00:00Z"):
        _inj(c, "nba_official", at, "AAA", "Old, Yuri", "Out", DAY)
        _inj(c, "espn", at.replace(":00Z", ":05Z"), "AAA", "Yuri Old", "Out")
    _inj(c, "nba_official", "2026-10-20T22:30:00Z", "AAA", "Old, Yuri", "Out", DAY)
    _inj(c, "nba_official", "2026-10-20T22:30:00Z", "AAA", "Star, Xavier", "Out", DAY)
    c.commit()
    yield c
    c.close()


# ----------------------------------------------------------------------------- the rule

def test_news_events_old_and_new(conn):
    lock = L._ts(LOCK)
    ev = L.news_events(conn, ["AAA", "BBB"], DAY, lock - L.LOOKBACK, lock)
    x = ev[("AAA", "xavier star")]
    assert x["at"] == "2026-10-20T22:30:00Z" and x["source"] == "nba_official"
    assert x["status_from"] == "Not listed" and x["status_to"] == "Out"
    assert ev[("AAA", "yuri old")]["at"] is None          # Out since the first batch: old news


def test_flags_beneficiaries_cap_and_record(conn):
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["version"] == L.VERSION == "ln-v1"
    assert [n["name"] for n in res["news"]] == ["Xavier Star"]
    n = res["news"][0]
    assert n["rotation"] and n["taken_in"] and n["baseline_run"] == "rb"
    assert n["min_to_lock"] == pytest.approx(30.0)
    got = [(f["name"], f["gain_dk"]) for f in res["flags"]]
    # +5, +4, +3.5 DK; Ted (+3.1 min) is cut by the per-team cap; Tad +1; Tob P(play) 0.3;
    # Bob is on the other team; Yuri's news is old
    assert got == [("Tom One", 5.0), ("Tug Six", 4.0), ("Tim Two", 3.5)]
    f = res["flags"][0]
    assert f["news"][0]["name"] == "Xavier Star" and f["news_at"] == "2026-10-20T22:30:00Z"
    assert f["min_to_lock"] == pytest.approx(30.0) and f["gain_min"] == pytest.approx(6.0)
    assert (f["own"], f["band"]) == (4.0, "Low") and f["baseline_at"] == "2026-10-20T22:15:00Z"
    assert L.own_uncertainty(res) == {"d2": 2.0, "d7": 2.0, "d3": 2.0}
    assert set(L.flags_by_dk(res)) == {"d2", "d3", "d7"}
    assert set(L.flags_by_player(res)) == {2, 3, 7}
    res4 = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z", per_team=4)
    assert "Ted Three" in [f["name"] for f in res4["flags"]]       # qualifies on minutes


def test_window_and_run_time(conn):
    assert L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z",
                     window=timedelta(minutes=20))["flags"] == []
    # a run before the news sees nothing
    assert L.compute(conn, SL, "rb", "2026-10-20T22:15:00Z")["news"] == []


def test_doubtful_to_out_is_not_news(conn):
    _inj(conn, "nba_official", "2026-10-20T21:00:00Z", "AAA", "Star, Xavier", "Doubtful", DAY)
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["news"] == [] and res["flags"] == []


def test_questionable_to_out_is_news(conn):
    _inj(conn, "nba_official", "2026-10-20T21:00:00Z", "AAA", "Star, Xavier", "Questionable",
         DAY)
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["news"][0]["status_from"] == "Questionable" and len(res["flags"]) == 3


def test_other_game_date_rows_are_ignored(conn):
    conn.execute("DELETE FROM injury_snapshots WHERE player_name='Star, Xavier'")
    _inj(conn, "nba_official", "2026-10-20T22:30:00Z", "AAA", "Star, Xavier", "Out",
         "2026-10-21")                                     # tomorrow's game on the same PDF
    assert L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")["news"] == []


def test_espn_alone_counts_and_earliest_source_wins(conn):
    _inj(conn, "espn", "2026-10-20T22:40:00Z", "AAA", "Yuri Old", "Out")
    _inj(conn, "espn", "2026-10-20T22:40:00Z", "AAA", "Xavier Star", "Out")
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["news"][0]["at"] == "2026-10-20T22:30:00Z"      # official was first
    conn.execute("DELETE FROM injury_snapshots WHERE source='nba_official' AND "
                 "player_name='Star, Xavier'")
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["news"][0]["source"] == "espn" and res["news"][0]["at"] == "2026-10-20T22:40:00Z"


def test_non_rotation_player_has_no_beneficiaries(conn, monkeypatch):
    monkeypatch.setattr(L, "ROTATION_MIN", 40.0)
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert len(res["news"]) == 1 and not res["news"][0]["rotation"] and res["flags"] == []


def test_news_not_taken_in_by_the_run(conn):
    conn.execute("UPDATE player_projections SET p_play=0.9 WHERE run_id='rc' AND player_id=1")
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert not res["news"][0]["taken_in"] and res["flags"] == []


def test_baseline_needs_same_model_version(conn):
    conn.execute("UPDATE dfs_runs SET model_version='other' WHERE run_id='rb'")
    res = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")
    assert res["news"][0]["baseline_run"] is None and res["flags"] == []


def test_run_flags_stored_computed_and_swap(conn):
    res = L.run_flags(conn, "rc")                       # nothing stored: computed
    assert len(res["flags"]) == 3
    L._CACHE.clear()
    stored = {"version": L.VERSION, "flags": [], "own_uncertainty": {}, "news": []}
    conn.execute("UPDATE dfs_runs SET detail=? WHERE run_id='rc'",
                 (json.dumps({L.DETAIL_KEY: stored}),))
    assert L.run_flags(conn, "rc") == stored            # stored wins
    _run(conn, "rs", "swap", "2026-10-20T23:20:00Z")
    L._CACHE.clear()
    assert L.run_flags(conn, "rs") == stored            # a swap run reads its build
    assert L.run_flags(conn, None) == {} and L.run_flags(conn, "nope") == {}
    assert L.own_uncertainty(None) == {}


def test_compute_safe_never_raises(conn, monkeypatch):
    monkeypatch.setattr(L, "news_events", lambda *a, **k: 1 / 0)
    assert L.compute_safe(conn, SL, "rc", "2026-10-20T22:50:00Z") is None


BANNED = re.compile(r"\b(?:edges?|lean\w*|over|value|play|plays|smash|locks?|bet\w*|"
                    r"leverage|ownership|chalk)\b", re.I)


def test_label_is_information_only(conn):
    f = L.compute(conn, SL, "rc", "2026-10-20T22:50:00Z")["flags"][0]
    s = L.label(f)
    assert "Xavier Star listed Out" in s and "30 minutes before" in s
    assert "+5.0 DK points" in s and "+6 expected minutes" in s and "(Low)" in s
    assert "not advice" in s and not BANNED.search(s), BANNED.search(s)


# ----------------------------------------------------------------------------- runner hook

def test_runner_stores_the_tag_in_the_run_detail():
    import inspect

    from courtside.proj.dfs import runner
    src = inspect.getsource(runner._execute)
    i, j = src.index("LNT.compute_safe"), src.index("_save_run(conn, run_id, sid, kind, now, "
                                                    "model_version")
    k = src.index("FLD.score_run_safe")
    assert k < i < j          # after the lineups and the proxy ownership, before the save


# ----------------------------------------------------------------------------- pages

def test_dfs_pool_shows_the_tag(data_dir):
    from fastapi.testclient import TestClient

    from courtside.web.app import create_app
    from test_web_dfs import DAY as WDAY, NOW, SLATE, players, populate, text_of
    data_dir.mkdir(parents=True, exist_ok=True)
    c = db.connect(data_dir / "courtside.db")
    populate(c, model=True)
    ps = players()
    team = {500 + p["i"]: p["team"] for p in ps}
    for pid, t in team.items():
        c.execute("UPDATE player_projections SET team=?, inputs_json=? WHERE player_id=?",
                  (t, json.dumps({"dk_id": str(pid + 500)}), pid))
    news, ben = ps[24], ps[20]                     # same team (i % 4 == 0 -> AAA)
    assert news["team"] == ben["team"]
    c.execute("UPDATE player_projections SET p_play=0.0 WHERE player_id=?", (500 + news["i"],))
    # baseline run 30 minutes earlier: the beneficiary 6 DK points lower
    rows = c.execute("SELECT * FROM player_projections WHERE run_id='run1'").fetchall()
    _ = c.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
                  "VALUES ('run0', ?, 'manual', '2026-10-20T17:50:00Z', 'dfsmodel-v1', 1)",
                  (SLATE,))
    for r in rows:
        cur = c.execute("INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, "
                        "team, run_id, created_at, model_version, p_play, inputs_json) VALUES "
                        "(?,?,?,?,?,'run0',?,?,?,?)",
                        (r["game_id"], r["nba_day"], r["slate_id"], r["player_id"], r["team"],
                         "2026-10-20T17:50:00Z", r["model_version"],
                         0.97 if r["player_id"] == 500 + news["i"] else r["p_play"],
                         r["inputs_json"]))
        for s in c.execute("SELECT stat, cond, mean FROM player_projection_stats WHERE "
                           "projection_id=?", (r["id"],)).fetchall():
            m = s["mean"] - (6.0 if r["player_id"] == 500 + ben["i"] and s["stat"] == "dk"
                             else 0.0)
            c.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                      "VALUES (?,?,?,?)", (cur.lastrowid, s["stat"], s["cond"], m))
    last, first = news["name"].split(" ")
    for at in ("2026-10-20T17:00:00Z", "2026-10-20T18:15:00Z"):
        _inj(c, "nba_official", at, news["team"], "Somebody, Else", "Out", WDAY)
    _inj(c, "nba_official", "2026-10-20T18:15:00Z", news["team"], f"{first}, {last}", "Out", WDAY)
    c.commit()
    res = L.run_flags(c, "run1")
    c.close()
    assert [f["name"] for f in res["flags"]] == [ben["name"]]
    L._CACHE.clear()
    app = create_app(data_dir)
    app.state.dfs_now = lambda: NOW
    html = TestClient(app).get(f"/dfs/{WDAY}/{SLATE}?all=1").text
    assert 'class="dx-tag dx-ln"' in html and html.count("dx-ln") == 1
    t = text_of(html)
    assert f"{news['name']} listed Out" in t and "1 on this slate" in t
    for w in ("edge", "lean", "smash", "leverage", "chalk"):
        assert w not in t.lower(), w


def test_tonight_card_shows_the_tag(data_dir, monkeypatch):
    from fastapi.testclient import TestClient

    import props_helpers as H
    from courtside.web import players_view
    from courtside.web.app import create_app
    from test_web_players import ADVICE_RE, NOW, _main_text
    c = db.connect()
    H.populate(c)
    c.close()
    flag = {"player_id": 101, "dk_id": "x", "name": "Ann Alder", "team": "AAA",
            "news": [{"name": "Nicolas Claxton", "status_to": "Out", "source": "nba_official",
                      "at": "2026-10-20T22:30:00Z", "min_to_lock": 30.0}],
            "gain_dk": 4.2, "gain_min": 3.0, "baseline_at": "2026-10-20T22:15:00Z",
            "band": "Low"}
    monkeypatch.setattr(players_view.LN, "run_flags",
                        lambda conn, rid: {"flags": [flag]} if rid else {})
    app = create_app(data_dir)
    app.state.players_now = lambda: NOW
    html = TestClient(app).get(f"/players/{H.DAY}").text
    assert html.count('class="tag tag-hl pl-ln"') == 1
    t = _main_text(html)
    assert "Teammate out" in t and "Nicolas Claxton listed Out" in t
    t = t.replace(players_view.DISCLAIMER.replace("’", "'"), " ")
    t = t.replace("not betting advice", " ")
    assert not ADVICE_RE.search(t), ADVICE_RE.search(t)
