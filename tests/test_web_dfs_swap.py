"""The slate page's "Late swap" action (web/dfs_swap.py, T2.14): saved lineups, a DK entries
upload, the edit CSV downloads, Showdown and error states, labels. Offline, synthetic slate."""

from __future__ import annotations

import csv
import io
import re
import warnings
from datetime import timedelta

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj.dfs import export as X  # noqa: E402
from courtside.proj.dfs import optimizer as O  # noqa: E402
from courtside.proj.dfs import rules as R  # noqa: E402
from courtside.proj.dfs import topn as TN  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

from test_dfs_swap import G1, G2, G3, CLASSIC, T, capture, dk_file, payload  # noqa: E402

DAY = "2026-10-20"
NOW = T(G2) + timedelta(minutes=5)           # games 1 and 2 have started


@pytest.fixture
def client(data_dir):
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
