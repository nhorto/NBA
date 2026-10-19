"""/dfs/results (upload round trip, hand entries) and the DFS tab of /health (T3.1 / T3.2)."""

from __future__ import annotations

import json
import re
import warnings
from datetime import date, timedelta

import pytest
from dfs_results_helpers import (CONTEST, DAY, FNAME_FMT, NAMES, SLATE, TABLE, lineup_text,
                                 player_block, populate, standings_csv, ten_entries)

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.mlops import dfs_monitor  # noqa: E402
from courtside.web import health_dfs  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

FNAME = FNAME_FMT.format(CONTEST)
TODAY = date(2026, 10, 25)


def text_of(html: str) -> str:
    body = re.sub(r"<(script|style|svg).*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def clean(html: str) -> str:
    t = text_of(html)
    assert not re.search(r"\bNone\b", t), re.search(r".{40}\bNone\b.{40}", t).group(0)
    assert not re.search(r"\bnan\b", t, re.I)
    assert "undefined" not in t
    return t


@pytest.fixture
def make(data_dir):
    made = []

    def _make(*, populated=True, **kw):
        d = data_dir.parent / f"web{len(made)}"
        made.append(d)
        d.mkdir(parents=True, exist_ok=True)
        conn = db.connect(d / "courtside.db")
        if populated:
            populate(conn, **kw)
        conn.close()
        return TestClient(create_app(d)), d
    return _make


def upload(c, data=None, name=FNAME, **form):
    data = standings_csv(ten_entries(), player_block()) if data is None else data
    return c.post("/dfs/results/upload", files={"file": (name, data, "text/csv")}, data=form,
                  follow_redirects=True)


def rows(d, sql, args=()):
    conn = db.connect(d / "courtside.db")
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


# ------------------------------------------------------------------ the page

def test_results_page_empty_states(make):
    c, _ = make(populated=False)
    r = c.get("/dfs/results")
    assert r.status_code == 200
    t = clean(r.text)
    assert "Real results" in t and "No entries yet" in t and "No contests uploaded yet" in t
    assert "Upload a finished contest" in t and "Export lineups to CSV" in t
    assert 'enctype="multipart/form-data"' in r.text and 'type="file"' in r.text
    assert "required" in re.search(r'<input[^>]*name="username"[^>]*>', r.text).group(0)
    # not swallowed by the /dfs/<day> route; the DFS pages link to it
    assert c.get("/dfs/2026-10-20").status_code == 200
    assert 'href="/dfs/results"' in c.get("/dfs/2026-10-20").text


def test_results_page_without_any_database(data_dir):
    c = TestClient(create_app(data_dir.parent / "nodb"))
    r = c.get("/dfs/results")
    assert r.status_code == 200 and "No database yet" in clean(r.text)


def test_upload_round_trip(make):
    c, d = make()
    r = upload(c, username="nhorto", remember="1")
    assert r.status_code == 200
    t = clean(r.text)
    assert "Standings imported" in t and "Contest 196456028: 10 entries" in t
    assert f"Linked to slate {SLATE}" in t and "Found 1 of your entry" in t
    # my results: real P&L from the payout table
    assert "My results" in t and "$3.00" in t and "Net" in t and "+$3.00" in t and "100.0%" in t
    assert "NBA $20K Test" in t                                      # the captured contest name
    # uploaded contest card with the derived numbers
    assert "Winning score" in t and "300.0" in t and "Top 1% score" in t
    assert re.search(r"Cash line\s+260\.0", t) and "from the payout table" in t
    assert "Joel Embiid 60.0%" in t                                   # most owned
    # database and archive
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_contest_results")[0]["n"] == 10
    [e] = rows(d, "SELECT * FROM dfs_entries")
    assert e["source"] == "upload" and e["payout"] == 6.0
    assert len(list((d / "raw" / "draftkings" / "uploads").glob("*"))) == 1
    assert (d / "dk_username.txt").read_text().strip() == "nhorto"
    # the username is remembered: the next page asks no more for it
    page = c.get("/dfs/results")
    assert 'value="nhorto"' in page.text
    assert "required" not in re.search(r'<input[^>]*name="username"[^>]*>', page.text).group(0)
    # the same file again: nothing duplicated
    r2 = upload(c, username="nhorto")
    assert "Already imported" in clean(r2.text)
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_uploads")[0]["n"] == 1
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_entries")[0]["n"] == 1


def test_upload_without_username_explains_itself(make):
    c, d = make()
    t = clean(upload(c).text)
    assert "No DK username is set" in t
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_entries")[0]["n"] == 0
    # ... typing it and uploading the same file again picks the entries up
    t = clean(upload(c, username="nhorto").text)
    assert "Already imported" in t and "added" in t
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_entries")[0]["n"] == 1


def test_upload_errors_are_readable(make):
    c, d = make()
    t = clean(c.post("/dfs/results/upload", files={"file": ("x.csv", b"hello,world\n", "text/csv")},
                    follow_redirects=True).text)
    assert "Could not import that file" in t and "Unrecognised file" in t
    t = clean(c.post("/dfs/results/upload", data={"username": "a"}, files={"file": ("", b"")},
                     follow_redirects=True).text)
    assert "Choose a file first" in t
    t = clean(upload(c, standings_csv(ten_entries(), player_block()), name="standings.csv").text)
    assert "contest id" in t.lower()
    t = clean(upload(c, standings_csv(ten_entries(), player_block()), name="standings.csv",
                     contest_id="424242424").text)
    assert "Contest 424242424: 10 entries" in t
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_contest_results")[0]["n"] == 10


def test_oversize_and_cross_site_posts_are_refused(make, monkeypatch):
    from courtside.proj.dfs import standings as ST
    c, d = make()
    monkeypatch.setattr(ST, "MAX_UPLOAD_BYTES", 1000)
    t = clean(upload(c).text)                                  # parsed, then refused by the importer
    assert "larger than" in t
    from courtside.web import dfs_results
    monkeypatch.setattr(dfs_results, "BODY_SLACK", 0)
    assert "too large" in clean(upload(c).text)               # refused while reading the body
    monkeypatch.undo()
    bad = c.post("/dfs/results/upload", files={"file": (FNAME, b"x", "text/csv")},
                 headers={"Origin": "https://evil.example"}, follow_redirects=False)
    assert bad.status_code == 403
    bad = c.post("/dfs/results/entry", data={"contest_id": "1"},
                 headers={"Sec-Fetch-Site": "cross-site"}, follow_redirects=False)
    assert bad.status_code == 403
    ok = c.post("/dfs/results/entry", data={"contest_id": "1", "fee": "2"},
                headers={"Origin": "http://testserver"}, follow_redirects=False)
    assert ok.status_code == 303
    assert rows(d, "SELECT COUNT(*) AS n FROM dfs_entries")[0]["n"] == 1


def test_manual_entry_and_winnings_form(make):
    c, d = make()
    lid = rows(d, "SELECT id FROM dfs_lineups")[0]["id"]
    page = c.get("/dfs/results").text
    assert f'<option value="{lid}">' in page and "Cash #1" in page          # our lineups to pick
    r = c.post("/dfs/results/entry", data={
        "contest_id": CONTEST, "fee": "3", "points": "270.5", "rank": "4", "field_size": "10",
        "lineup_id": str(lid), "payout": "", "note": "test"}, follow_redirects=True)
    t = clean(r.text)
    assert "Entry added" in t and "no winnings recorded yet" in t
    [e] = rows(d, "SELECT * FROM dfs_entries")
    assert e["source"] == "manual" and e["payout"] is None and e["lineup_id"] == lid
    assert "our cash #1" in t and 'name="payout"' in r.text                   # the pending form
    r = c.post(f"/dfs/results/entry/{e['entry_id']}", data={"payout": "6"}, follow_redirects=True)
    t = clean(r.text)
    assert "+$3.00" in t and "no winnings recorded" not in t
    bad = clean(c.post("/dfs/results/entry", data={"contest_id": "5", "fee": "abc"},
                       follow_redirects=True).text)
    assert "not a number" in bad
    assert len(rows(d, "SELECT * FROM dfs_entries")) == 1


def test_unlinked_contest_can_be_linked(make):
    c, d = make(contest=False)
    pb = [("Nobody Atall%d" % i, "C", "1.0%", "1") for i in range(20)]
    ents = [(r, 9000 + r, f"u{r}", 0, 100 - r, "C Nobody Atall0 F Nobody Atall1") for r in range(1, 4)]
    t = clean(upload(c, standings_csv(ents, pb), name="contest-standings-888000222.csv").text)
    assert "Could not tell which slate" in t and "Which slate was this" in t
    assert "20 player name(s) could not be matched" in t
    r = c.post("/dfs/results/link", data={"contest_id": "888000222", "slate_id": SLATE},
               follow_redirects=True)
    t = clean(r.text)
    assert "Contest linked" in t and "Which slate was this" not in t
    assert rows(d, "SELECT slate_id FROM dfs_contests WHERE contest_id='888000222'")[0][
        "slate_id"] == SLATE


# ------------------------------------------------------------------ health DFS tab

def snap(conn, table, as_of, model, metric, window, value, n=20, se=None):
    conn.execute(f"INSERT INTO {table} (as_of, model, metric, \"window\", n, value, se, "
                 f"details_json) VALUES (?,?,?,?,?,?,?,'{{}}')",
                 (as_of, model, metric, window, n, value, se))


def fill_snapshots(conn, days=4):
    for k in range(days):
        as_of = (TODAY - timedelta(days=days - 1 - k)).isoformat()
        for w, nn in (("7d", 20), ("30d", 60), ("season", 90)):
            for m, base in (("cash", 270.0), ("topn", 275.0), ("fppg", 262.0), ("salary", 250.0),
                            ("hindsight", 360.0), ("random", 200.0)):
                snap(conn, "dfs_metric_snapshots", as_of, m, "actual_mean", w, base + k, nn, 6.0)
                snap(conn, "dfs_metric_snapshots", as_of, m, "ratio_hindsight", w,
                     (base + k) / 360.0, nn, 0.02)
            for m in ("cash", "topn"):
                snap(conn, "dfs_metric_snapshots", as_of, m, "pct_random", w, 0.93, nn, 0.01)
                snap(conn, "dfs_metric_snapshots", as_of, m, "diff_vs_fppg", w, 8.0 + k, nn, 3.0)
                snap(conn, "dfs_metric_snapshots", as_of, m, "would_cash_rate", w, 0.6, 4)
            for m, mae in (("ours", 6.1), ("fppg", 6.9), ("salary", 7.4)):
                snap(conn, "player_metric_snapshots", as_of, m, "dk_mae", w, mae - 0.02 * k, 400, 0.1)
            snap(conn, "player_metric_snapshots", as_of, "ours", "dk_mae_gain_vs_fppg", w, 11.5, 400, 1.5)
            snap(conn, "player_metric_snapshots", as_of, "ours", "dk_cov80", w, 0.79, 400, 0.02)
            snap(conn, "player_metric_snapshots", as_of, "ours", "dk_cov50", w, 0.49, 400, 0.02)
            snap(conn, "player_metric_snapshots", as_of, "ours", "dk_crps", w, 4.25, 400, 0.1)
            snap(conn, "player_metric_snapshots", as_of, "ours", "min_mae_played15", w, 4.6, 300, 0.2)
            snap(conn, "player_metric_snapshots", as_of, "ours", "min_bias", w, -0.31, 400, 0.1)
    conn.commit()


def test_health_dfs_tab_empty(make):
    c, _ = make(populated=False)
    r = c.get("/health/dfs")
    assert r.status_code == 200
    t = clean(r.text)
    assert "DFS tracking" in t and "Nothing has been graded yet" in t
    assert "October 20" in t.replace("Oct 20", "October 20") and "No graded lineups yet" in t
    assert "No open DFS alerts" in t and "No real entries yet" in t
    assert "Versions are registered after the first DFS run" in t
    assert 'href="/health"' in r.text and 'href="/dfs/results"' in r.text
    assert c.get("/health").status_code == 200


def test_health_dfs_tab_without_a_database(data_dir):
    c = TestClient(create_app(data_dir.parent / "nodb2"))
    r = c.get("/health/dfs")
    assert r.status_code == 200 and "Nothing has been graded yet" in clean(r.text)


def test_health_dfs_tab_with_data(make, monkeypatch):
    c, d = make()
    conn = db.connect(d / "courtside.db")
    fill_snapshots(conn)
    # slate runs: main slate ok, early missed; baselines snapshot for the main slate
    conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
                 "VALUES ('run0', ?, 'lock45', '2026-10-20T22:15:00Z', 'pathA-v0.1', 1)", (SLATE,))
    det = {"slate_id": SLATE, "hindsight": {"actual": 355.5}, "fppg": {"actual": 261.0},
           "salary": {"actual": 244.0}, "ours": [{"role": "cash", "rank": 1, "actual": 281.5}]}
    snap(conn, "dfs_metric_snapshots", DAY, f"slate:{SLATE}", "baselines", "1d", 355.5)
    conn.execute("UPDATE dfs_metric_snapshots SET details_json=? WHERE model LIKE 'slate:%'",
                 (json.dumps(det),))
    dfs_monitor.sync_registry(conn)
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message, status) "
                 "VALUES ('dfs:cov80','2026-10-24T09:00:00Z','2026-10-24T09:00:00Z','warn','dfs',"
                 "'The 80% range for DK points covered 70.0% of the last 300 player results.','open')")
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message, status) "
                 "VALUES ('perf:gbm_market','2026-10-24T09:00:00Z','2026-10-24T09:00:00Z','warn',"
                 "'performance','Game model alert.','open')")
    conn.commit()
    conn.close()
    # real entries via the upload path so the real block has data
    upload(c, username="nhorto", remember="1")
    from courtside.web import app as webapp
    monkeypatch.setattr(webapp.config, "today_nba_day", lambda: TODAY)
    r = c.get("/health/dfs")
    t = clean(r.text)
    assert r.status_code == 200
    # paper results
    assert "Our cash lineup" in t and "DK FPPG optimizer" in t and "Salary-implied optimizer" in t
    assert "Best possible (hindsight)" in t and "Share of the best possible score" in t
    assert "Paper: scored on real games, not entered" in t
    assert "Season" in t and "7 days" in t and "30 days" in t
    assert "DK points error per player" in t and "80% range coverage" in t
    assert re.search(r"Our projections\s+6\.1", t) or "Our projections" in t
    assert "Minutes error (players 15+ min)" in t and "CRPS" in t
    # charts: server-side svg, narrow + wide
    assert r.text.count('class="chart hm-chart is-wide"') == 2 and "is-narrow" in r.text
    # slates: lock runs ok / missed and the scores
    assert "Last slates" in t and "OK" in t and "281.5" in t and "355.5" in t
    # alerts: only the DFS ones on this tab
    assert "The 80% range for DK points covered 70.0%" in t and "Game model alert" not in t
    # versions
    assert "Player minutes" in t and "DFS engine" in t and "pathA-v0.1" in t
    # real block: separate, labelled, with the money
    i_real, i_paper = t.index("Real contests"), t.index("Our lineups vs the alternatives")
    assert i_paper < i_real
    real = t[i_real:]
    assert "Real money" in real and "$3.00" in real and "NBA $20K Test" in real
    assert "Winner" in real and "Cash line" in real and "260.0" in real
    assert "Paper" not in real
    # the Games tab does not list the DFS components and links here
    g = c.get("/health")
    assert g.status_code == 200 and "Player minutes" not in g.text and 'href="/health/dfs"' in g.text
    gt = clean(g.text)
    assert "The 80% range for DK points" in gt                       # the alert list shows all kinds


def test_health_dfs_view_model_pending_and_negative_money(make):
    c, d = make()
    conn = db.connect(d / "courtside.db")
    from courtside.proj.dfs import standings as ST
    ST.add_manual_entry(conn, contest_id="5", entry_fee=20, payout=0, points=100, rank=900,
                        field_size=1000, slate_id=SLATE)
    ST.add_manual_entry(conn, contest_id="6", entry_fee=5, slate_id=SLATE)
    vm = health_dfs.build(conn, TODAY)
    t = {r["label"]: [x["text"] for x in r["cells"]] for r in vm["real"]["table"]}
    assert t["Entries"][2] == "2" and t["Entry fees"][2] == "$25.00"
    assert t["Net"][2] == "−$20.00" and t["ROI"][2] == "−100.0%"
    assert t["Winnings not entered yet"][2] == "1"
    conn.close()
