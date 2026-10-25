"""The /dfs page, DK CSV export and the templated "why this lineup" (T1.7).

A synthetic slate (30 invented players, 2 games) is written into a temporary courtside.db with
the same tables the collectors and the runner fill; nothing touches the network or var/.
"""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj.dfs import export as dfs_export  # noqa: E402
from courtside.proj.dfs import why as dfs_why  # noqa: E402
from courtside.web import dfs_view  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

DAY = "2026-10-20"
SLATE = "9001"
NOW = datetime(2026, 10, 20, 16, 50, tzinfo=timezone.utc)     # 12:50 ET, lock 15:00 ET
POS = ["PG", "SG", "SF", "PF", "C"]
FIRST = ["Ann", "Bo", "Cy", "Di", "Ed", "Flo"]
LAST = ["Alder", "Birch", "Cedar", "Dogwood", "Elm"]
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]


def pid(i: int) -> str:
    return str(1000 + i)


def players() -> list[dict]:
    out = []
    for i in range(30):
        g = i % 2
        team = ("AAA", "BBB", "CCC", "DDD")[g * 2 + (i // 2) % 2]
        out.append({"i": i, "id": pid(i), "name": f"{FIRST[i % 6]} {LAST[i % 5]}{i}",
                    "team": team, "pos": POS[i % 5], "salary": 3000 + 250 * i,
                    "fppg": round(14 + 1.1 * i, 1), "game": ("AAA @ BBB", "CCC @ DDD")[g],
                    "game_id": f"G{g}", "home": ("BBB", "DDD")[g], "away": ("AAA", "CCC")[g]})
    return out


def lineup_slots(ids: list[int]) -> list[dict]:
    ps = {p["i"]: p for p in players()}
    return [{"slot": SLOTS[k], "player_id": pid(i), "name": ps[i]["name"], "team": ps[i]["team"],
             "salary": ps[i]["salary"], "mean": ps[i]["fppg"], "draftable_id": f"D{pid(i)}{SLOTS[k]}"}
            for k, i in enumerate(ids)]


def populate(conn, *, run: bool = True, model: bool = False, late: bool = False,
             tag: str = "main", salary_only: bool = False, ranges: bool = True) -> None:
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
                 "lock_utc) VALUES (?,?,?,?,?,?)",
                 (SLATE, DAY, 70, tag, '["G0","G1"]', "2026-10-20T19:00:00Z"))
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
                 "lock_utc) VALUES (?,?,?,?,?,?)",
                 ("9002", DAY, 81, "showdown", '["G1"]', "2026-10-20T23:00:00Z"))
    for p in players():
        for slot in ("UTIL", p["pos"]):
            conn.execute(
                "INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id, "
                "draftable_id, player_name, site_player_id, team, positions, roster_slot, salary, "
                "status, fppg, disabled, game_id, game_label, home, away, start_time, starter) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?)",
                ("2026-10-20T12:00:00Z", "dk", SLATE, "classic", 70, f"D{p['id']}{slot}",
                 p["name"], p["id"], p["team"], p["pos"], slot, p["salary"],
                 "Q" if p["i"] == 3 else ("OUT" if p["i"] == 4 else "None"), p["fppg"],
                 p["game_id"], p["game"], p["home"], p["away"],
                 "2026-10-20T19:00:00.0000000Z" if p["game_id"] == "G0"
                 else "2026-10-20T23:00:00.0000000Z", 1 if p["i"] % 3 == 0 else None))
    conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, captured_at) "
                 "VALUES ('c1', ?, 'Test GPP', '2026-10-20T12:00:00Z')", (SLATE,))
    if model:
        mv = "dfsmodel-v1"
        for p in players():
            conn.execute("INSERT INTO player_ids (source, source_id, player_id) VALUES "
                         "('dk', ?, ?)", (p["id"], 500 + p["i"]))
            inputs = json.dumps({"ewma_min": 22.0, "absent": ["Zed Zeta"]}) if p["i"] == 20 else "{}"
            cur = conn.execute(
                "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, run_id, "
                "created_at, model_version, p_play, inputs_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (p["game_id"], DAY, SLATE, 500 + p["i"], "run1", "2026-10-20T14:10:00Z", mv,
                 0.8 if p["i"] == 3 else 0.97, inputs))
            mean = p["fppg"] * 1.05
            conn.execute("INSERT INTO player_projection_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (cur.lastrowid, "dk", "all", mean, 8.0, 0, 0, 0, mean, 0, 0, 0))
            conn.execute("INSERT INTO player_projection_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (cur.lastrowid, "dk", "played", mean, 8.0, mean - 14, mean - 11,
                          mean - 6, mean, mean + 6, mean + 13, mean + 17))
            conn.execute("INSERT INTO player_projection_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (cur.lastrowid, "min", "played", 33.0 if p["i"] == 20 else 24.0, 4.0,
                          None, None, None, None, None, None, None))
    if run:
        news = json.dumps([{"name": players()[20]["name"], "reason": "named a starter"},
                           {"name": "Zed Zeta", "team": "AAA", "reason": "status changed to Out"}]) \
            if late else None
        conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, "
                     "late_news_json, ok) VALUES ('run1', ?, 'lock45', '2026-10-20T18:20:00Z', ?, ?, 1)",
                     (SLATE, "dfsmodel-v1" if model else None, news))
        lineups = [[29, 28, 27, 26, 25, 24, 23, 20], [29, 28, 27, 26, 25, 24, 23, 22],
                   [29, 28, 27, 26, 25, 24, 21, 20]]
        conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean, "
                     "p10, p50, p90, p_cash) VALUES ('run1','cash',1,?,?,?,?,?,?,?)",
                     (json.dumps(lineup_slots(lineups[0])), 49500, 300.5,
                      *((270, 299, 330) if ranges else (None, None, None)), 0.55))
        for k, ids in enumerate(lineups, 1):
            conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean, "
                         "p10, p50, p90) VALUES ('run1','topn',?,?,?,?,?,?,?)",
                         (k, json.dumps(lineup_slots(ids)), 49000 + k, 310 - k,
                          *((280, 308, 340) if ranges else (None, None, None))))
    conn.commit()


@pytest.fixture
def make(data_dir):
    """make(**populate_kwargs) -> TestClient on a fresh synthetic database."""
    made = []

    def _make(**kw):
        d = data_dir.parent / f"data{len(made)}"
        made.append(d)
        d.mkdir(parents=True, exist_ok=True)
        conn = db.connect(d / "courtside.db")
        populate(conn, **kw)
        conn.close()
        app = create_app(d)
        app.state.dfs_now = lambda: NOW
        return TestClient(app)
    return _make


def text_of(html: str) -> str:
    body = re.sub(r"<(script|style|svg).*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def test_page_renders_full(make):
    c = make(model=True, late=True)
    r = c.get(f"/dfs/{DAY}")
    assert r.status_code == 200
    t = text_of(r.text)
    assert "Cash lineup" in t and "Top 3 lineups" in t and "Player pool" in t
    assert "Contest builder" in t and "1 contest captured" in t and "Open the contest builder" in t
    assert "Locks in 2h 10m" in t                             # 16:50Z -> 19:00Z
    assert "Last run: 45-minute run, 2:20 PM ET" in t
    assert "Projections: model dfsmodel-v1" in t and "Beta" not in t
    assert "Main" in t and "Showdown" in t
    assert r.text.index('href="/dfs/2026-10-20/9001"') < r.text.index('href="/dfs/2026-10-20/9002"')
    assert "<svg" in r.text and 'class="rng-dot"' in r.text      # ranges as inline SVG
    assert re.search(r"<table[^>]*>\s*<caption", r.text)
    assert "Why this lineup" in t
    assert "None" not in t and "nan" not in t.lower().split()
    # slot rows of the cash lineup
    for slot in SLOTS:
        assert f">{slot}</th>" in r.text


def test_lineup_ranges_render_when_present(make):
    c = make(model=True)
    html = c.get(f"/dfs/{DAY}").text
    t = text_of(html)
    assert "Range 270" in t and "330 (10th to 90th percentile, simulated)" in t   # cash
    assert t.count("Range 280") == 3 and "340" in t                               # each Top N card
    assert "Range is the 10th to 90th percentile of simulated lineup points" in t
    # the lineup bars are inline SVG with an accessible description, light/dark via CSS vars
    assert html.count("Simulated range, 10th to 90th percentile 270 to 330") == 2  # aria + title
    assert 'class="dfs-lu-bar"' in html and html.count('class="dfs-sum-bar"') == 3
    css = c.get("/static/style.css").text
    assert ".dfs-lu-bar" in css and "var(--accent)" in css


def test_lineup_ranges_hidden_when_absent(make):
    c = make(model=True, ranges=False)
    html = c.get(f"/dfs/{DAY}").text
    t = text_of(html)
    assert "percentile, simulated)" not in t and "simulated lineup points" not in t
    assert "Range 2" not in t
    assert "Simulated range" not in html
    assert 'class="dfs-lu-bar"' not in html and 'class="dfs-sum-bar"' not in html
    assert "Cash lineup" in t and "300.5" in t          # the rest of the card is unchanged


def test_fallback_badge_and_pool_without_model(make):
    c = make()   # run exists, no projections, model_version NULL
    t = text_of(c.get(f"/dfs/{DAY}").text)
    assert "Beta" in t and "Projections: DraftKings FPPG fallback" in t
    assert "FPPG, not a projection" in t


def test_no_run_yet_shows_pool_and_message(make):
    c = make(run=False)
    t = text_of(c.get(f"/dfs/{DAY}").text)
    assert "Lineups appear 45 minutes before lock" in t
    assert "No run yet" in t
    assert "Player pool" in t and "Ann Alder0" in t


def test_no_slate_today_empty_state(make):
    c = make()
    r = c.get("/dfs/2026-10-25")
    assert r.status_code == 200
    t = text_of(r.text)
    assert "No DraftKings slate for" in t
    assert "Previous slate day" in t
    assert c.get("/dfs/2026-10-20/nope").status_code == 404
    assert c.get("/dfs/not-a-day").status_code == 404


def test_redirect_and_nav(make):
    c = make()
    r = c.get("/dfs", follow_redirects=False)
    assert r.status_code == 307 and re.fullmatch(r"/dfs/\d{4}-\d\d-\d\d", r.headers["location"])
    assert 'href="/dfs"' in c.get("/archive").text


def test_empty_database(data_dir):
    c = TestClient(create_app(data_dir))
    r = c.get("/dfs/2026-10-20")
    assert r.status_code == 200 and "No DraftKings slate" in r.text


def test_late_news_banner(make):
    c = make(model=True, late=True)
    html = c.get(f"/dfs/{DAY}").text
    t = text_of(html)
    assert "3 lineups contain late-news players" in t
    assert "named a starter" in t and "status changed to Out" in t
    assert 'role="status"' in html
    quiet = make(model=True)
    assert "late-news" not in text_of(quiet.get(f"/dfs/{DAY}").text)


def test_pool_sort_filter_search(make):
    c = make(model=True)
    base = f"/dfs/{DAY}/{SLATE}"

    def names(qs):
        html = c.get(base + qs).text
        return re.findall(r'<span class="dfs-name">([^<]+)</span>',
                          html.split('id="pool"')[1])
    by_salary = names("?sort=salary&dir=asc&all=1")
    assert by_salary[0] == "Ann Alder0" and by_salary[-1].endswith("29")
    by_proj = names("?all=1")
    assert by_proj[0].endswith("29")
    c_only = names("?pos=C&all=1")
    assert c_only and all(int(n.rsplit("r", 1)[-1][-2:].lstrip("dm")) % 5 == 4
                          for n in c_only if n[-1].isdigit())
    assert names("?q=Alder0&all=1")[0] == "Ann Alder0"
    assert "Flo Elm4" not in names("?hide=out&all=1")      # player 4 is listed OUT
    # default page shows 40 of the pool (30 players: all) with sort links and aria-sort
    html = c.get(base).text
    assert 'aria-sort="descending"' in html
    assert "sort=salary" in html


def test_pool_value_is_points_per_thousand(make):
    c = make(model=True)
    html = c.get(f"/dfs/{DAY}/{SLATE}?sort=value&all=1").text
    pool = html.split('id="pool"')[1]
    rows = re.findall(r"\$([\d,]+)</td>\s*<td class=\"num\">([\d.]+)</td>\s*<td class=\"num\">([\d.]+)", pool)
    assert rows
    for sal, proj, val in rows:
        assert abs(float(proj) / (int(sal.replace(",", "")) / 1000) - float(val)) < 0.06


def test_export_csv(make):
    c = make()
    r = c.get(f"/dfs/{DAY}/{SLATE}/export.csv?mode=topn&n=2")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    assert len(rows) == 3
    assert rows[1][0] == "D1029PG" and rows[1][7] == "D1020UTIL"
    cash = list(csv.reader(io.StringIO(c.get(f"/dfs/{DAY}/{SLATE}/export.csv?mode=cash").text)))
    assert len(cash) == 2 and cash[1][0] == "D1029PG"
    nid = c.get(f"/dfs/{DAY}/{SLATE}/export.csv?mode=cash&fmt=name_id").text.splitlines()[1]
    assert nid.startswith("Ann Alder29") or "(D1029PG)" in nid
    assert c.get(f"/dfs/{DAY}/{SLATE}/export.csv?fmt=bad").status_code == 400
    assert c.get(f"/dfs/{DAY}/9002/export.csv").status_code == 404      # no run for that slate
    txt = c.get(f"/dfs/{DAY}/{SLATE}/lineup.txt?mode=topn&rank=2")
    assert txt.status_code == 200 and txt.text.startswith("Top N #2") and "Salary $" in txt.text


def test_export_missing_draftable_id():
    slots = [{"slot": "PG", "player_id": "1", "name": "X", "draftable_id": None}]
    with pytest.raises(dfs_export.ExportError):
        dfs_export.to_csv([slots])
    assert dfs_export.to_csv([], showdown=True).strip() == "CPT,UTIL,UTIL,UTIL,UTIL,UTIL"
    assert dfs_export.default_header() == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    sd = [{"slot": "CPT", "draftable_id": 5, "name": "A"}] + [
        {"slot": "UTIL", "draftable_id": 6 + i, "name": "B"} for i in range(5)]
    out = dfs_export.to_csv([sd]).splitlines()
    assert out == ["CPT,UTIL,UTIL,UTIL,UTIL,UTIL", "5,6,7,8,9,10"]


# ------------------------------------------------------------------------------- why.py

def why_inputs():
    pool = [{"id": str(i), "name": f"First{i} Last{i}", "team": "AAA", "salary": 3000 + 500 * i,
             "proj": 15.0 + 3 * i if i != 7 else 60.0, "game": "AAA @ BBB" if i < 6 else "CCC @ DDD",
             "p_play": 0.97, "status": ""} for i in range(10)]
    pool[2].update(p_play=0.72, status="Q")
    pool[5].update(min_mean=31.0, inputs={"ewma_min": 24.0, "absent": ["Big Out"]})
    pool[8].update(status="GTD", p_play=None)
    lineup = {"slots": [{"slot": SLOTS[k], "player_id": str(i), "name": pool[i]["name"],
                         "salary": pool[i]["salary"], "mean": pool[i]["proj"]}
                        for k, i in enumerate([7, 5, 2, 1, 0, 3, 8, 4])],
              "salary": sum(pool[i]["salary"] for i in [7, 5, 2, 1, 0, 3, 8, 4]),
              "mean": 200.0}
    return lineup, pool


NUM = re.compile(r"(?<![A-Za-z\d])\d[\d,]*\.?\d*")


def numbers(text: str) -> set[float]:
    return {float(m.rstrip(".").replace(",", "")) for m in NUM.findall(text)}


def fact_numbers(x) -> set[float]:
    out: set[float] = set()
    if isinstance(x, dict):
        for v in x.values():
            out |= fact_numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            out |= fact_numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.add(float(x))
    return out


def test_why_only_uses_numbers_from_the_data():
    lineup, pool = why_inputs()
    w = dfs_why.why_lineup(lineup, pool)
    assert 1 <= len(w["reasons"]) <= 3
    kinds = [r["kind"] for r in w["reasons"]]
    assert kinds[0] == "minutes" and "price" in kinds
    given = set()
    for p in pool:
        given |= {float(p["salary"]), float(p["salary"]) / 1000, p["proj"], round(p["proj"], 1)}
        given |= {float(p.get("min_mean") or 0), float((p.get("inputs") or {}).get("ewma_min", 0))}
    for r in w["reasons"]:
        f = fact_numbers(r["facts"])
        assert numbers(r["text"]) <= f, (r["text"], f)
    minutes = next(r for r in w["reasons"] if r["kind"] == "minutes")
    assert minutes["facts"]["minutes"] == 31 and minutes["facts"]["recent_avg"] == 24
    assert minutes["facts"]["delta_min"] == 7 and "Out" in minutes["text"]
    price = next(r for r in w["reasons"] if r["kind"] == "price")
    top = price["facts"]
    assert top["proj"] == 60.0 and top["salary"] == pool[7]["salary"]
    # flags: P(play) < 0.9 and a Q/GTD status without P(play); never for 0.97
    flagged = {f["name"] for f in w["flags"]}
    assert flagged == {pool[2]["name"], pool[8]["name"]}
    q = next(f for f in w["flags"] if f["name"] == pool[2]["name"])
    assert numbers(q["text"]) == {72.0}
    s = w["summary"]
    assert s["salary_used"] == lineup["salary"] and s["salary_left"] == 50000 - lineup["salary"]
    assert numbers(s["text"]) <= {float(s["salary_used"]), 50000.0, float(s["salary_left"])}
    assert json.loads(dfs_why.dumps(w)) == w
    assert dfs_why.loads(dfs_why.dumps(w)) == w and dfs_why.loads("{bad") is None


def test_why_degrades_without_optional_inputs():
    lineup, pool = why_inputs()
    for p in pool:
        p.pop("inputs", None)
        p.pop("min_mean", None)
        p["p_play"] = None
        p["status"] = ""
    w = dfs_why.why_lineup(lineup, pool)
    assert all(r["kind"] != "minutes" for r in w["reasons"])
    assert w["flags"] == []
    assert dfs_why.why_lineup({"slots": [], "salary": 0}, [])["reasons"] == []
    # minutes without a named absence: stated without a cause
    lineup, pool = why_inputs()
    pool[5]["inputs"] = {"ewma_min": 24.0}
    m = next(r for r in dfs_why.why_lineup(lineup, pool)["reasons"] if r["kind"] == "minutes")
    assert "with" not in m["text"] and "above his recent average" in m["text"]


# ------------------------------------------------------------------------------- voice

BANNED = [r"\block\s+of\b", r"\bedge\b", r"\blean(s|ing)?\b", r"\bover\b", r"\bbest bet\b",
          r"\bfade\b", r"\bsmash\b", r"\bvalue play\b", r"\bguarantee[sd]?\b", r"\bsure thing\b"]


def test_style_words_absent(make):
    c = make(model=True, late=True)
    pages = [c.get(f"/dfs/{DAY}").text, c.get(f"/dfs/{DAY}/9002").text,
             c.get("/dfs/2026-10-30").text]
    for html in pages:
        t = text_of(html).lower()
        for pat in BANNED:
            assert not re.search(pat, t), pat
    # every "lock" on the page is the DraftKings lock time, never a pick
    t = text_of(pages[0]).lower()
    for m in re.finditer(r"\block\w*", t):
        ctx = t[max(0, m.start() - 20): m.end() + 25]
        assert re.search(r"locks? (in|tue|at)|lock\)?\.|before lock|locked at|lock time", ctx), ctx
    lineup, pool = why_inputs()
    w = json.dumps(dfs_why.why_lineup(lineup, pool)).lower()
    for pat in BANNED:
        assert not re.search(pat, w), pat


def test_stored_why_wins(make):
    c = make(model=True)
    conn = db.connect(c.app.state.store.db_path)
    conn.execute("UPDATE dfs_lineups SET why_json=? WHERE mode='cash'", (json.dumps(
        {"version": 0, "reasons": [{"kind": "game", "text": "STORED REASON 7", "facts": {}}],
         "flags": [], "summary": {"text": "Uses $49,500 of $50,000"}}),))
    conn.commit()
    conn.close()
    assert "STORED REASON 7" in c.get(f"/dfs/{DAY}").text


def test_helpers():
    assert dfs_view.is_fallback(None) and dfs_view.is_fallback("fppg-fallback")
    assert not dfs_view.is_fallback("v1-abc")
    assert dfs_view.projection_badge("v1-beta")["beta"] and not dfs_view.projection_badge("v1")["beta"]
    now = NOW
    from datetime import timedelta
    assert dfs_view.lock_text(now + timedelta(minutes=35), now) == "Locks in 35m"
    assert dfs_view.lock_text(now - timedelta(minutes=5), now).startswith("Locked at")
    assert dfs_view.lock_text(now + timedelta(days=2), now).startswith("Locks ")
    assert dfs_view.parse_late_news('{"players": [{"player": "A B", "why": ["x", "y"]}]}') == [
        {"name": "A B", "id": None, "team": None, "reason": "x; y"}]
    assert dfs_view.parse_late_news(None) == []


def test_runner_late_news_entries_render_plain_english():
    from courtside.web import dfs_view
    txt = dfs_view._runner_reason({
        "reasons": ["status", "unexpected_starter", "minutes"],
        "status": {"from": "Q", "to": "O", "source": "draftkings"},
        "minutes": {"from": 24.1, "to": 31.0, "delta": 6.9}})
    assert txt == "status Q → O; named a starter; projected minutes 24.1 → 31.0"


# ------------------------------------------------------------------------------- research
# Boom / bust, leverage and DK points vs salary on the pool (stats/dfs_research.py).

import numpy as np  # noqa: E402

from courtside.stats import dfs_research as RS  # noqa: E402

QV = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 55.0]      # q05 q10 q25 q50 q75 q90 q95


def test_boom_bust_from_quantiles():
    # $5,000: boom mark 30 (= q50), bust mark 15 (= q10)
    up, dn = RS.boom_bust_quantiles(QV, 5000)
    assert up == pytest.approx(0.50) and dn == pytest.approx(0.10)
    # between quantiles: linear (35 is half-way q50 -> q75)
    assert RS.cdf(QV, 35.0) == pytest.approx(0.625)
    assert RS.tail_above(QV, 35.0) == pytest.approx(0.375)
    # beyond q95 the chance halves every q95 - q90 = 5 points: 60 -> 2.5 %, 65 -> 1.25 %
    up, dn = RS.boom_bust_quantiles(QV, 10000)
    assert up == pytest.approx(0.025) and dn == pytest.approx(0.50)
    assert RS.tail_above(QV, 65.0) == pytest.approx(0.0125)
    # below q05 likewise (q10 - q05 = 5): 5 -> 2.5 %
    assert RS.cdf(QV, 5.0) == pytest.approx(0.025)
    assert RS.boom_bust_quantiles([None] * 7, 5000) == (None, None)
    assert RS.boom_bust_quantiles(QV, None) == (None, None)


def test_boom_bust_from_sims(make, data_dir, tmp_path):
    # 10 sims, 2 players; player A ($5,000: boom >= 30, bust < 15) misses 2 sims (DNP)
    fp = np.array([[0, 30], [0, 31], [31, 10], [30, 40], [29, 5], [14, 20], [15, 35],
                   [40, 12], [10, 50], [50, 22]], dtype=np.float16)
    min_t = np.ones((2, 10), dtype=np.uint8) * 30
    min_t[0, :2] = 0
    path = tmp_path / "sims.npz"
    np.savez(path, fp=fp, dk_id=np.array(["1001", "1002"]), min_t=min_t)
    make()
    conn = db.connect(data_dir.parent / "data0" / "courtside.db")
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id='run1'", (str(path),))
    conn.commit()
    out = RS.sim_boom_bust(conn, "run1", {"1001": 5000.0, "1002": 4000.0})
    conn.close()
    boom, bust, n = out["1001"]
    assert n == 8                                       # only the sims in which he played
    assert boom == pytest.approx(4 / 8)                 # 31, 30, 40, 50
    assert bust == pytest.approx(2 / 8)                 # 14, 10 (not the two DNPs)
    boom, bust, n = out["1002"]                         # $4,000: boom >= 24, bust < 12
    assert (n, boom, bust) == (10, pytest.approx(0.5), pytest.approx(0.2))
    # a run whose file is gone: no sims (the page falls back to the quantiles)
    assert RS.sim_boom_bust(db.connect(data_dir.parent / "data0" / "courtside.db"),
                            "nope", {"1001": 5000.0}) == {}


def test_leverage_sign():
    assert RS.leverage(0.30, 10.0) == pytest.approx(20.0)        # in more lineups than the field
    assert RS.leverage(0.0, 12.5) == pytest.approx(-12.5)        # in none of ours
    assert RS.leverage(0.10, 25.0) == pytest.approx(-15.0)
    assert RS.leverage(0.5, None) is None                        # no popularity estimate
    shares, n = RS.lineup_shares([lineup_slots([1, 2, 3]), lineup_slots([1, 4, 5])])
    assert n == 2 and shares[pid(1)] == 1.0 and shares[pid(4)] == 0.5 and pid(9) not in shares


def test_value_vs_salary_fit_and_delta(make, data_dir):
    make()
    conn = db.connect(data_dir.parent / "data0" / "courtside.db")
    _add_grades(conn)
    lines = RS.compute_salary_lines(conn, DAY)
    conn.close()
    fit = lines["2026-27"]
    # y = 5 DK per $1,000 - 5 exactly
    assert fit["n"] == 60 and fit["b"] == pytest.approx(0.005) and fit["a"] == pytest.approx(-5)
    assert RS.expected_dk(fit, 6000) == pytest.approx(25.0)
    recent = {"last": [30.0, 34.0], "last_seasons": ["2026-27", "2025-26"], "cur": "2026-27",
              "season_mean": 30.0, "season_n": 1}
    v = RS.value_vs_salary(recent, fit, 6000)
    assert v["pm"] == pytest.approx(7.0) and v["last_n"] == 2 and v["last_prev"] == 1
    assert v["season_pm"] == pytest.approx(5.0)
    assert RS.value_vs_salary({"last": []}, fit, 6000) is None
    assert RS.value_vs_salary(recent, None, 6000) is None


def _add_grades(conn) -> None:
    """60 graded Classic player-games before DAY on the line DK = 5/1000 salary - 5, plus
    two box scores for players[29] (NBA id 529) and the popularity proxy of run1."""
    for k in range(60):
        sal = 3000 + 100 * k
        cur = conn.execute("INSERT INTO player_projections (game_id, nba_day, player_id, run_id, "
                           "created_at, model_version) VALUES (?,?,?,?,?,?)",
                           (f"00226{k:05d}", "2026-10-1" + str(k % 9), 700 + k, "old",
                            "2026-10-10T00:00:00Z", "m"))
        conn.execute("INSERT INTO player_projection_grades (projection_id, game_id, player_id, "
                     "run_id, graded_at, played, actual_dk, salary) VALUES (?,?,?,?,?,?,?,?)",
                     (cur.lastrowid, f"00226{k:05d}", 700 + k, "old", "2026-10-19T00:00:00Z", 1,
                      5 * sal / 1000 - 5, sal))
    for gid, d, pts in (("0022600001", "2026-10-15", 40), ("0022600002", "2026-10-17", 50)):
        conn.execute("INSERT INTO player_games (game_id, player_id, team, season, season_type, "
                     "game_date, min, pts, reb, ast, fg3m, stl, blk, tov) VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (gid, 529, "DDD", "2026-27", "Regular Season", d, 34, pts, 0, 0, 0, 0, 0, 0))
    for i, own in ((29, 60.0), (28, 5.0), (0, 30.0)):
        conn.execute("INSERT INTO dfs_ownership_proxy (run_id, player_id, own, band) VALUES "
                     "('run1', ?, ?, 'Med')", (int(pid(i)), own))
    conn.commit()


def test_pool_research_columns(make, data_dir):
    from courtside.stats import db as sdb
    sdb.clear()
    c = make(model=True)
    conn = db.connect(data_dir.parent / "data0" / "courtside.db")
    _add_grades(conn)
    conn.close()
    html = c.get(f"/dfs/{DAY}/{SLATE}?all=1").text
    for h in (">6×+<", ">Under 3×<", ">Ours − field<", ">vs salary<"):
        assert h in html, h
    assert "approximated from the run&rsquo;s stored 5th to 95th percentiles" in html
    assert 'data-tip="6×+: chance he scores at least 6× his salary' in html
    v = dfs_view.build_page(db.connect(data_dir.parent / "data0" / "courtside.db"), DAY, SLATE,
                            now=NOW, params={"all": "1"})
    rows = {r["id"]: r for r in v["pool"]["rows"]}
    assert v["pool"]["bb_method"] == "quantiles" and v["pool"]["lev_n"] == 3
    # players[29] is in all 3 Top N lineups (100 %) at a 60 % proxy: +40; players[28] too: +95
    assert rows[pid(29)]["lev"] == pytest.approx(40.0)
    assert rows[pid(28)]["lev"] == pytest.approx(95.0)
    assert rows[pid(0)]["lev"] == pytest.approx(-30.0)        # in none of them at 30 %
    assert rows[pid(5)]["lev"] is None                        # no proxy row
    assert "In 3 of 3 Top N lineups (100%)" in rows[pid(29)]["lev_tip"]
    # boom / bust from the stored played quantiles (mean-14 ... mean+17)
    r = rows[pid(29)]
    up, dn = RS.boom_bust_quantiles(r["qs"], r["salary"])
    assert r["boom"] == pytest.approx(up) and r["bust"] == pytest.approx(dn)
    assert r["bb_src"] == "quantiles" and 0 <= r["bust"] < r["boom"] <= 1
    assert rows[pid(4)]["boom"] is None                       # OUT
    # DK points vs salary: 529 scored 40 and 50 DK; $10,250 implies 46.25
    vpm = rows[pid(29)]["vpm"]
    assert vpm["last_n"] == 2 and vpm["pm"] == pytest.approx(45.0 - 46.25)
    assert "This season (2 games)" in rows[pid(29)]["vpm_tip"]
    assert rows[pid(5)]["vpm"] is None                        # no box scores
    # server-side sorts on the new columns
    for key in ("boom", "bust", "lev", "vpm"):
        assert c.get(f"/dfs/{DAY}/{SLATE}?sort={key}").status_code == 200
    def pool_text(q: str) -> str:
        html = c.get(f"/dfs/{DAY}/{SLATE}?{q}").text
        return text_of(html[html.index('id="pool"'):])
    t = pool_text("sort=lev&dir=desc")
    assert t.index(players()[28]["name"]) < t.index(players()[29]["name"]) \
        < t.index(players()[0]["name"])
    t = pool_text("sort=lev&dir=asc")
    assert t.index(players()[0]["name"]) < t.index(players()[29]["name"])
    # still none of the banned words with the new columns and tooltips
    for pat in BANNED:
        assert not re.search(pat, t.lower()), pat


def test_pool_research_absent_without_data(make):
    html = make().get(f"/dfs/{DAY}/{SLATE}").text           # FPPG pool, no proxy, no grades
    assert ">Boom<" not in html and ">Lev<" not in html and ">Value +/−<" not in html
