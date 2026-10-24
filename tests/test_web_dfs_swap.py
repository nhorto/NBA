"""The slate page's "Late swap" action (web/dfs_swap.py, T2.14): saved lineups, a DK entries
upload, the edit CSV downloads, Showdown and error states, labels. Offline, synthetic slate."""

from __future__ import annotations

import csv
import io
import re
import threading
import warnings
from datetime import timedelta

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj.dfs import export as X  # noqa: E402
from courtside.proj.dfs import optimizer as O  # noqa: E402
from courtside.proj.dfs import rules as R  # noqa: E402
from courtside.proj.dfs import swap as SW  # noqa: E402
from courtside.proj.dfs import topn as TN  # noqa: E402
from courtside.web import dfs_swap as WS  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

from test_dfs_swap import G1, G2, G3, CLASSIC, T, capture, dk_file, payload  # noqa: E402

DAY = "2026-10-20"
NOW = T(G2) + timedelta(minutes=5)           # games 1 and 2 have started


@pytest.fixture
def client(data_dir, monkeypatch):
    monkeypatch.setattr(WS, "WAIT_S", 60.0)     # the fast path renders in the request
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    capture(conn, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
    conn.execute("UPDATE dfs_slates SET nba_day=?, tag='main' WHERE slate_id='S1'", (DAY,))
    pool = R.build_pool(conn, "S1")
    projs = {p.dk_player_id: p.fppg for p in pool.players}
    rid = O.save_run(conn, slate_id="S1", kind="lock10", ok=True, detail={},
                     created_at="2026-10-20T22:50:00Z")
    O.save_lineups(conn, rid, "cash", [O.optimize(pool, projs, CLASSIC, mode="cash")])
    res = TN.top_n(pool, projs, CLASSIC, 3)
    O.save_lineups(conn, rid, "topn", res)
    # a game-3 player of the saved lineups is ruled out after the build
    victim = next(pid for _, pid in res.lineups[0].lineup.slot_pairs()
                  if pool.by_id()[pid].start == T(G3))
    capture(conn, "S1", payload([G1, G2, G3], status={int(victim): "O"}),
            at="2026-10-21T00:30:00Z")
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, lock_utc) "
                 "VALUES ('SD', ?, 81, 'showdown', '2026-10-20T23:00:00Z')", (DAY,))
    conn.commit()
    lu = res.lineups[0].lineup
    cells = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    file_text = dk_file([("4601", "NBA GPP", "c2", "$3", cells),
                         ("4602", "NBA $5 Double Up", "c3", "$5", cells)], pool)
    victim_name = pool.by_id()[victim].name
    conn.close()
    app = create_app(data_dir)
    app.state.dfs_now = lambda: NOW
    c = TestClient(app)
    c.ctx = {"file": file_text, "victim": victim_name, "cells": cells}
    return c


def text_of(html: str) -> str:
    body = re.sub(r"<(script|style|svg).*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def test_slate_page_links_to_late_swap(client):
    r = client.get(f"/dfs/{DAY}/S1")
    assert r.status_code == 200
    assert f'href="/dfs/{DAY}/S1/swap"' in r.text and ">Late swap<" in r.text


def test_saved_lineups_page(client):
    r = client.get(f"/dfs/{DAY}/S1/swap")
    assert r.status_code == 200
    t = text_of(r.text)
    assert "Late swap: Main" in t and "Our saved lineups" in t
    assert "Started, locked" in t and "Open" in t
    assert "Ruled out for open slots" in t and client.ctx["victim"] in t
    assert "Swap proposed" in t and "Locked" in t and "Game started" in t
    assert "DraftKings lists him Out" in t
    assert "Download DK edit CSV" in t and "never signs in to DraftKings" in t
    assert "has not been confirmed with a real upload" in t
    for bad in ("bet", "wager", "lock of", "smash", "edge", "value play"):
        assert not re.search(rf"\b{bad}\b", t, re.I), bad
    r = client.get(f"/dfs/{DAY}/S1/swap?scope=news")
    assert "only lineups with late news" in text_of(r.text).lower()


def test_saved_lineups_csv(client):
    r = client.get(f"/dfs/{DAY}/S1/swap.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == list(X.EDIT_HEADER) + list(CLASSIC.slot_names)
    assert len(rows) == 5 and all(x[0] == "" for x in rows[1:])
    assert client.ctx["victim"] not in r.text
    r = client.get(f"/dfs/{DAY}/S1/swap.csv?fmt=id")
    assert all(c.isdigit() for c in list(csv.reader(io.StringIO(r.text)))[1][4:])
    assert client.get(f"/dfs/{DAY}/S1/swap.csv?fmt=xml").status_code == 400


def test_upload_entries_and_download(client):
    r = client.post(f"/dfs/{DAY}/S1/swap", files={"file": ("DKEntries.csv",
                                                           client.ctx["file"], "text/csv")},
                    data={"scope": "all"})
    assert r.status_code == 200
    t = text_of(r.text)
    assert "Your uploaded entries" in t and "Entry 4601" in t and "NBA $5 Double Up" in t
    assert "Read cells as Name (ID)" in t and "2 of 2 lineups with a swap" in t
    assert "Cash: favours a safer floor" in t and "Highest projection" in t
    href = re.search(r'href="([^"]+swap\.csv\?t=[^"]+)"', r.text).group(1).replace("&amp;", "&")
    d = client.get(href)
    assert d.status_code == 200 and "late-swap" in d.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(d.text)))
    assert [x[0] for x in rows[1:]] == ["4601", "4602"] and rows[1][1] == "NBA GPP"
    pool_cells = client.ctx["cells"]
    started = 0
    for k, cell in enumerate(rows[1][4:]):
        if cell == pool_cells[k]:
            started += 1
    assert started >= 4 and client.ctx["victim"] not in d.text
    assert client.get(f"/dfs/{DAY}/S1/swap.csv?t=nope").status_code == 410


def test_upload_errors_and_cross_site(client):
    r = client.post(f"/dfs/{DAY}/S1/swap", files={"file": ("x.csv", "Rank,EntryId\n1,2\n",
                                                           "text/csv")})
    assert r.status_code == 200 and "no Entry ID column" in r.text
    r = client.post(f"/dfs/{DAY}/S1/swap", files={"file": ("x.csv", b"", "text/csv")})
    assert "Choose your DraftKings entries CSV first" in r.text
    r = client.post(f"/dfs/{DAY}/S1/swap", files={"file": ("x.csv", client.ctx["file"],
                                                           "text/csv")},
                    headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_showdown_and_missing(client):
    r = client.get(f"/dfs/{DAY}/SD/swap")
    assert r.status_code == 200 and "Showdown has no late swap" in r.text
    assert client.get(f"/dfs/{DAY}/nope/swap").status_code == 404
    assert client.get("/dfs/2026-13-45/S1/swap").status_code == 404


def test_preview_time_and_all_open(client):
    r = client.get(f"/dfs/{DAY}/S1/swap?at=2026-10-20T21:00:00Z")
    t = text_of(r.text)
    assert "No game has started yet" in t and "Started, locked" not in t


# ------------------------------------------------------------------ background jobs

@pytest.fixture
def slow(client, monkeypatch):
    """`SW.plan` held on a gate, counting calls; requests do not wait for it."""
    real = SW.plan
    gate = threading.Event()
    calls = []

    def plan(*a, **k):
        calls.append(k.get("csv_data") is not None)
        gate.wait(30)
        if isinstance(slow_fail.get("raise"), BaseException):
            raise slow_fail["raise"]
        return real(*a, **k)

    slow_fail: dict = {}
    monkeypatch.setattr(SW, "plan", plan)
    monkeypatch.setattr(WS, "WAIT_S", 0.0)
    yield type("Slow", (), {"gate": gate, "calls": calls, "fail": slow_fail})
    gate.set()


def _jobs(client) -> WS.SwapJobs:
    return client.app.state.swap_jobs


def _finish(client, slow, job_id: str) -> None:
    slow.gate.set()
    assert _jobs(client).get(job_id).done.wait(30)


def _refresh(html: str) -> str:
    m = re.search(r'<meta http-equiv="refresh" content="\d+;url=([^"]+)">', html)
    assert m, "no self-refresh on the pending page"
    return m.group(1).replace("&amp;", "&")


def _norm(html: str) -> str:
    html = re.sub(r"Solved in [\d.]+ s", "", html)
    return re.sub(r"(&amp;|&)j=[\w-]+", "", html)


def test_pending_then_ready_matches_fast_path(client, slow, monkeypatch):
    fast = client.get(f"/dfs/{DAY}/S1/swap?at=2026-10-21T00:20:00Z")    # gate shut: pending
    assert fast.status_code == 200
    t = text_of(fast.text)
    assert "Working out the swap" in t and "usually under a minute" in t
    assert 'role="status"' in fast.text and "Download DK edit CSV" not in t
    url = _refresh(fast.text)
    assert url.startswith(f"/dfs/{DAY}/S1/swap?scope=all&at=") and "&job=" in url
    job_id = url.rsplit("job=", 1)[1]
    still = client.get(url)                              # refresh while running: same job
    assert "Working out the swap" in still.text and _refresh(still.text) == url
    _finish(client, slow, job_id)
    ready = client.get(url)
    assert ready.status_code == 200 and "Working out the swap" not in ready.text
    assert 'http-equiv="refresh"' not in ready.text
    assert "Swap proposed" in text_of(ready.text) and slow.calls == [False]
    # the same page rendered on the fast path (a fresh app, plan inline) is identical
    monkeypatch.setattr(WS, "WAIT_S", 60.0)
    _jobs(client).jobs.clear()
    inline = client.get(f"/dfs/{DAY}/S1/swap?at=2026-10-21T00:20:00Z")
    assert _norm(inline.text) == _norm(ready.text)


def test_csv_uses_finished_job(client, slow):
    r = client.get(f"/dfs/{DAY}/S1/swap")
    job_id = _refresh(r.text).rsplit("job=", 1)[1]
    d = client.get(f"/dfs/{DAY}/S1/swap.csv")           # same key: the running job
    assert d.status_code == 202 and "still being worked out" in d.text
    _finish(client, slow, job_id)
    ready = client.get(f"/dfs/{DAY}/S1/swap?job={job_id}")
    href = re.search(r'href="([^"]+swap\.csv\?scope=all&amp;j=[^"]+)"', ready.text)
    assert href, "saved CSV link carries the job"
    d = client.get(href.group(1).replace("&amp;", "&"))
    assert d.status_code == 200 and d.headers["content-type"].startswith("text/csv")
    assert len(list(csv.reader(io.StringIO(d.text)))) == 5
    assert client.get(f"/dfs/{DAY}/S1/swap.csv").status_code == 200    # reused, live window
    assert slow.calls == [False]                         # one plan, never recomputed


def test_same_key_reuses_job(client, slow):
    a = _refresh(client.get(f"/dfs/{DAY}/S1/swap?scope=news").text)
    b = _refresh(client.get(f"/dfs/{DAY}/S1/swap?scope=news").text)
    assert a == b and slow.calls == [False]
    c = _refresh(client.get(f"/dfs/{DAY}/S1/swap?scope=all").text)    # another key
    assert c != a and slow.calls == [False, False]
    _finish(client, slow, a.rsplit("job=", 1)[1])
    _jobs(client).get(c.rsplit("job=", 1)[1]).done.wait(30)
    r = client.get(f"/dfs/{DAY}/S1/swap?scope=news")     # finished: reused, no new plan
    assert "only lineups with late news" in text_of(r.text).lower()
    assert slow.calls == [False, False]


def test_upload_runs_in_background(client, slow):
    up = {"file": ("DKEntries.csv", client.ctx["file"], "text/csv")}
    r = client.post(f"/dfs/{DAY}/S1/swap", files=up, data={"scope": "all"})
    assert r.status_code == 200 and "Working out the swap" in r.text
    assert "DKEntries.csv" in text_of(r.text)
    url = _refresh(r.text)
    job_id = url.rsplit("job=", 1)[1]
    r2 = client.post(f"/dfs/{DAY}/S1/swap", files=up, data={"scope": "all"})
    assert _refresh(r2.text) == url and slow.calls == [True]      # same file: same job
    assert client.get(f"/dfs/{DAY}/S1/swap.csv?t={job_id}").status_code == 410   # not ready
    _finish(client, slow, job_id)
    t = text_of(client.get(url).text)
    assert "Your uploaded entries" in t and "Entry 4601" in t and "2 of 2 lineups" in t
    assert "From DKEntries.csv" in t
    d = client.get(f"/dfs/{DAY}/S1/swap.csv?t={job_id}")
    assert d.status_code == 200 and 'filename="DKEntries-late-swap.csv"' in \
        d.headers["content-disposition"]
    assert [x[0] for x in list(csv.reader(io.StringIO(d.text)))[1:]] == ["4601", "4602"]
    assert _jobs(client).get(job_id).fn is None          # the upload's bytes are dropped
    assert slow.calls == [True]


def test_background_errors(client, slow, monkeypatch):
    slow.fail["raise"] = SW.SwapError("There is no slate S1 any more.")
    r = client.get(f"/dfs/{DAY}/S1/swap")
    url = _refresh(r.text)
    _finish(client, slow, url.rsplit("job=", 1)[1])
    t = text_of(client.get(url).text)
    assert "Could not build a late swap" in t and "no slate S1 any more" in t
    monkeypatch.setattr(WS, "WAIT_S", 30.0)
    d = client.get(f"/dfs/{DAY}/S1/swap.csv")            # an error is not reused: re-run
    assert d.status_code == 409 and "no slate S1 any more" in d.text
    slow.fail["raise"] = RuntimeError("kaput")
    d = client.get(f"/dfs/{DAY}/S1/swap.csv")
    assert d.status_code == 409 and "RuntimeError: kaput" in d.text
    # expired / unknown job id
    t = text_of(client.get(f"/dfs/{DAY}/S1/swap?job=nope").text)
    assert "has expired" in t


def test_background_timeout(client, slow, monkeypatch):
    monkeypatch.setattr(WS, "JOB_TIMEOUT_S", 0.3)
    url = _refresh(client.get(f"/dfs/{DAY}/S1/swap").text)
    monkeypatch.setattr(WS, "WAIT_S", 5.0)               # waits only up to the timeout
    t = text_of(client.get(url).text)
    assert "more than two minutes" in t and "Try again" in t
    assert client.get(f"/dfs/{DAY}/S1/swap.csv?" + url.split("?", 1)[1].replace(
        "job=", "j=")).status_code == 409
    slow.gate.set()
    job = _jobs(client).get(url.rsplit("job=", 1)[1])
    assert job.done.wait(30) and job.state() == "error" and job.plan is None
    monkeypatch.setattr(WS, "JOB_TIMEOUT_S", 120.0)      # a retry starts a fresh job
    r = client.get(f"/dfs/{DAY}/S1/swap")
    assert "Swap proposed" in text_of(r.text) and len(slow.calls) == 2
