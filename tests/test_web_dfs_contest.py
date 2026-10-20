"""The contest builder page (web/dfs_contest.py, T2.16).

The /dfs synthetic slate (test_web_dfs.populate) plus a captured GPP with a payout table and a
run with sims. The build itself is replaced by a fast fake that returns a `Portfolio` (the real
build is covered by test_dfs_portfolio.py); everything else is the real route: form parsing,
validation, the background thread, storage, the result page, labels and the DK CSV.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import warnings
from datetime import datetime, timezone

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj.dfs import field as F  # noqa: E402
from courtside.proj.dfs import portfolio as PF  # noqa: E402
from courtside.web import dfs_contest  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

from test_web_dfs import BANNED, DAY, NOW, SLATE, lineup_slots, pid, populate, text_of  # noqa: E402

URL = f"/dfs/{DAY}/{SLATE}/contest"
CID = "777"
SAME = {"origin": "http://testserver"}
PAYOUTS = [{"min_pos": 1, "max_pos": 1, "payout": 1000.0}, {"min_pos": 2, "max_pos": 20,
                                                             "payout": 20.0}]


def add_contest(conn):
    conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, entry_fee, field_size, "
                 "max_per_user, total_prizes, payout_json, captured_at) VALUES "
                 "(?,?,?,?,?,?,?,?,?)", (CID, SLATE, "NBA $1.4K Test Shot [20 Entry Max]", 3.0,
                                         600, 20, 1380.0, json.dumps(PAYOUTS),
                                         "2026-10-20T12:00:00Z"))
    conn.execute("UPDATE dfs_runs SET sim_path='/nonexistent/sims.npz' WHERE run_id='run1'")
    conn.commit()


@pytest.fixture
def client(data_dir, monkeypatch):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    populate(conn, model=True)
    add_contest(conn)
    conn.close()
    calls = []

    def fake_build(conn, slate_id, contest_id, s, *, run_id=None, now=None, progress=None):
        calls.append({"slate": slate_id, "contest": contest_id, "settings": s})
        if progress:
            progress("candidates")
        con = F.load_contest(conn, contest_id)
        ids = [[29, 28, 27, 26, 25, 24, 23, 20], [29, 28, 27, 26, 25, 22, 21, 19]][: s.n]
        lus = []
        for k, lu in enumerate(ids, 1):
            sl = lineup_slots(lu)
            lus.append({"rank": k, "slots_json": json.dumps(sl), "salary": 49000 + k,
                        "mean": 300.0 - k, "proj": 300.0, "p10": 260.0, "p50": 299.0, "p90": 340.0,
                        "p_cash": 0.31, "p_top1": 0.012, "e_payout": 4.5 - k,
                        "e_payout_single": 4.0, "p_cash_single": 0.3, "dupes": 0.1 * k,
                        "dupe_band": "Low", "marginal": 4.5 - k})
        summary = {"entries": len(lus), "requested": s.n, "stopped": None, "e_payout": 6.5,
                   "e_payout_se": 0.4, "cost": 3.0 * len(lus), "roi": 6.5 / (3.0 * len(lus)) - 1,
                   "p_any_cash": 0.45, "p_profit": 0.3, "p_any_top1": 0.02, "e_cashing": 0.6,
                   "n_sims": 25000, "relative": 1.12, "relative_se": 0.03,
                   "baseline": {"name": "Top N by projection", "e_payout": 5.8, "p_any_cash": 0.4,
                                "p_any_top1": 0.015},
                   "strengths": {"soft": {"roi": 0.2, "e_payout": 7.2},
                                 "sharp": {"roi": -0.1, "e_payout": 5.4},
                                 "medium": {"roi": 0.08, "e_payout": 6.5}}}
        exposure = [{"player_id": pid(29), "name": "Bo Elm29", "team": "DDD", "positions": "C",
                     "salary": 10250, "mean": 46.0, "count": 2, "share": 1.0, "own_proxy": 0.3,
                     "own_band": "High", "cap": 2, "min": 0, "locked": False}]
        info = {"candidates_total": 80, "n_sims": 25000, "field": {"sampled": 5000},
                "ownership_calibrated": False, "seconds": 3.2, "gamma": s.gamma}
        return PF.Portfolio(slate_id, con, s, lus, summary, exposure, info, "run1")

    monkeypatch.setattr(PF, "build_for_slate", fake_build)
    app = create_app(data_dir)
    app.state.dfs_now = lambda: NOW
    c = TestClient(app)
    c.calls = calls
    c.app_ = app
    return c


def wait_done(c, timeout=10.0):
    b = c.app_.state.contest_builds
    t0 = time.time()
    while b.busy() and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert not b.busy()


def test_builder_page_labels_and_contest_details(client):
    r = client.get(URL)
    assert r.status_code == 200
    t = text_of(r.text)
    assert "Simulated, not real results" in t
    assert "proxy" in t.lower()
    assert "Sensible stakes" in t and "entertainment" in t
    assert "NBA $1.4K Test Shot [20 Entry Max]" in t
    assert "$3.00" in t and "600" in t and "$1,000" in t          # fee, field, top prize
    assert re.search(r"\b20\b", t)                                # max entries
    assert 'name="contest"' in r.text and 'name="n"' in r.text and 'name="max_exp"' in r.text
    assert f'name="p_{pid(29)}"' in r.text                        # lock / exclude control
    assert "Build lineup set" in t and " disabled" not in r.text.split("Build lineup set")[0][-80:]
    low = t.lower()
    assert "kelly" not in low and not re.search(r"\bstake (\$|\d)", low)
    for pat in BANNED:
        assert not re.search(pat, low), pat


def test_slate_page_links_the_builder(client):
    r = client.get(f"/dfs/{DAY}/{SLATE}")
    assert f'href="{URL}"' in r.text
    t = text_of(r.text)
    assert "Simulated chance to cash ~50–65%" in t and "55%" not in t     # p_cash 0.55, banded


def test_dollar_headline_only_after_ten_uploaded_slates(client):
    r = client.post(URL, data={"contest": CID, "n": "2", "max_exp": "60", "gamma": "5"},
                    headers=SAME, follow_redirects=False)
    wait_done(client)
    conn = db.connect(client.app_.state.store.db_path)
    for k in range(10):
        conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, captured_at) "
                     "VALUES (?,?,?,?)", (f"u{k}", f"s{k}", "x", "2026-10-01T00:00:00Z"))
        conn.execute("INSERT INTO dfs_contest_results (contest_id, rank, entry_hash, "
                     "uploaded_at) VALUES (?,1,'h','2026-10-02T00:00:00Z')", (f"u{k}",))
    conn.commit()
    conn.close()
    t = text_of(client.get(r.headers["location"]).text)
    assert "Expected payout $6.50" in t and "ROI +8%" in t


def test_build_flow_result_and_csv(client):
    r = client.post(URL, data={"contest": CID, "n": "2", "max_exp": "60", "gamma": "5",
                               f"p_{pid(29)}": "lock", f"p_{pid(1)}": "exclude"},
                    headers=SAME, follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith(URL + "?p=pf-")
    wait_done(client)
    s = client.calls[0]["settings"]
    assert client.calls[0]["contest"] == CID and s.n == 2 and s.max_exposure == 0.6
    assert s.locks == (pid(29),) and s.excludes == (pid(1),) and s.strengths
    page = client.get(loc)
    assert page.status_code == 200 and 'http-equiv="refresh"' not in page.text
    t = text_of(page.text)
    for want in ("Simulated, not real results", "Any entry cashes", "vs Top 2 by proj.",
                 "+12%", "~35–50%", "~1–3%", "Exposure", "Field popularity (proxy)",
                 "Dupe risk", "Download DK CSV", "Bo Elm29", "Details (simulated, uncalibrated)"):
        assert want in t, want
    # no dollar payout / ROI headline before 10 uploaded slates: only inside the details toggle
    head = page.text.split('class="stats dfc-stats"')[1].split("</dl>")[0]
    assert "$" not in head and "ROI" not in head and "%" in head
    details = page.text.split('class="card dfc-details"')[1].split("</details>")[0]
    assert "$6.50" in details and "+8%" in details                # 6.5 / 6.0 - 1
    assert "$6.50" not in page.text.split('class="card dfc-details"')[0]
    pfid = loc.split("p=")[1]
    j = client.get(f"{URL}/{pfid}.json")
    assert j.status_code == 200 and "uncalibrated" in j.json()["label"]
    assert j.json()["summary"]["e_payout"] == 6.5 and len(j.json()["lineups"]) == 2
    assert client.get(f"{URL}/pf-nothere.json").status_code == 404
    csv_r = client.get(f"{URL}/{pfid}.csv")
    assert csv_r.status_code == 200 and csv_r.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv_r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(csv_r.text)))
    assert rows[0] == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    assert len(rows) == 3 and rows[1][0] == f"D{pid(29)}PG"
    named = list(csv.reader(io.StringIO(client.get(f"{URL}/{pfid}.csv?fmt=name_id").text)))
    assert named[1][0].endswith(f"(D{pid(29)}PG)")
    assert client.get(f"{URL}/{pfid}.csv?fmt=bad").status_code == 400
    # stored as dfs_lineups mode gpp under the portfolio id, not as a dfs_runs row
    conn = db.connect(client.app_.state.store.db_path)
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups WHERE run_id=? AND mode='gpp'",
                        (pfid,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs WHERE run_id=?", (pfid,)).fetchone()[0] == 0
    conn.close()
    # the builds list links it
    assert f"?p={pfid}" in client.get(URL).text


def test_validation_errors_and_cross_site(client):
    bad = client.post(URL, data={"contest": CID, "n": "21", "max_exp": "60", "gamma": "5"},
                      headers=SAME)
    assert bad.status_code == 400 and "at most 20 entries" in text_of(bad.text)
    assert client.post(URL, data={"contest": "nope", "n": "2", "max_exp": "60", "gamma": "5"},
                       headers=SAME).status_code == 400
    assert client.post(URL, data={"contest": CID, "n": "2", "max_exp": "5", "gamma": "5"},
                       headers=SAME).status_code == 400
    x = client.post(URL, data={"contest": CID, "n": "2", "max_exp": "60", "gamma": "5"},
                    headers={"origin": "http://evil.example"})
    assert x.status_code == 403
    assert client.calls == []


def test_locked_slate_cannot_build(client):
    client.app_.state.dfs_now = lambda: datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc)
    t = text_of(client.get(URL).text)
    assert "no longer takes new entries" in t
    r = client.post(URL, data={"contest": CID, "n": "2", "max_exp": "60", "gamma": "5"},
                    headers=SAME)
    assert r.status_code == 400 and client.calls == []


def test_building_state_refreshes_and_errors_show(client):
    conn = db.connect(client.app_.state.store.db_path)
    PF.start_row(conn, "pf-building1", SLATE, CID, PF.Settings(n=5))
    PF.start_row(conn, "pf-failed001", SLATE, CID, PF.Settings(n=5))
    PF.fail_row(conn, "pf-failed001", "no legal lineup under the locks and excludes")
    PF.start_row(conn, "pf-oldbuild", SLATE, CID, PF.Settings(n=5))
    conn.execute("UPDATE dfs_portfolios SET created_at='2026-10-20T16:49:00Z' "
                 "WHERE portfolio_id='pf-building1'")
    conn.execute("UPDATE dfs_portfolios SET created_at='2026-10-20T15:00:00Z' "
                 "WHERE portfolio_id='pf-oldbuild'")
    conn.commit()
    conn.close()
    r = client.get(URL + "?p=pf-building1")
    assert 'http-equiv="refresh"' in r.text and "Building" in text_of(r.text)
    assert client.get(f"{URL}/pf-building1.csv").status_code == 404
    r = client.get(URL + "?p=pf-failed001")
    assert "The build failed" in text_of(r.text) and "no legal lineup" in text_of(r.text)
    r = client.get(URL + "?p=pf-oldbuild")                # interrupted (server restart)
    assert "did not finish" in text_of(r.text) and 'http-equiv="refresh"' not in r.text


def test_unknown_build_and_slate_404(client):
    assert client.get(URL + "?p=pf-nothere").status_code == 404
    assert client.get(f"/dfs/{DAY}/424242/contest").status_code == 404
    assert client.get(f"{URL}/pf-nothere.csv").status_code == 404


def test_parse_build_settings():
    contests = {"1": {"max_n": 150}}
    cid, s = dfs_contest.parse_build({"contest": "1", "n": "150", "max_exp": "40", "gamma": "4",
                                      "p_a": "lock", "p_b": "exclude", "p_c": ""},
                                     contests, {"a", "b", "c"})
    assert cid == "1" and s.n == 150 and s.max_exposure == pytest.approx(0.4) and s.gamma == 4
    assert s.locks == ("a",) and s.excludes == ("b",)
    with pytest.raises(PF.PortfolioError):
        dfs_contest.parse_build({"contest": "1", "n": "5", "max_exp": "40", "gamma": "4",
                                 "p_zz": "lock"}, contests, {"a"})
    with pytest.raises(PF.PortfolioError):
        dfs_contest.parse_build({"contest": "1", "n": "5", "max_exp": "40", "gamma": "9"},
                                contests, set())
