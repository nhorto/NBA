"""D10: the Showdown slate page points at the contest builder; the builder page speaks Showdown.

Display only (no lineup changes, no switch): a Showdown slate gets a guidance box with the builder
link, the builder first in the section nav, a note on the Top N, six-slot captions; the builder
page gets the Showdown guidance line, "(of 6)" and overlap choices below six. Classic pages are
unchanged. No "lock"/"edge"-style words, no stake sizing.
"""

from __future__ import annotations

import re
import warnings

warnings.filterwarnings("ignore", message="Using `httpx`")

from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.web import dfs_contest  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

from test_web_dfs import BANNED, DAY, NOW, SLATE, populate, text_of  # noqa: E402
from test_web_dfs_contest import URL, add_contest  # noqa: E402

SD_GUIDE = "Showdown is one game."


def _client(data_dir, monkeypatch, tag):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    populate(conn, model=True, tag=tag)
    add_contest(conn)
    conn.close()
    app = create_app(data_dir)
    app.state.dfs_now = lambda: NOW
    return TestClient(app)


def test_showdown_slate_page_points_at_the_builder(data_dir, monkeypatch):
    c = _client(data_dir, monkeypatch, "showdown")
    r = c.get(f"/dfs/{DAY}/{SLATE}")
    assert r.status_code == 200
    t = text_of(r.text)
    assert SD_GUIDE in t and "joint simulations" in t and "modelled field" in t
    box = r.text.split('class="dfs-sd-guide"')[1].split("</div>")[0]
    assert f'href="{URL}"' in box and "Open the contest builder" in box
    nav = r.text.split('class="secnav"')[1].split("</nav>")[0]
    assert nav.index("#contest") < nav.index("#cash")
    assert "Captain plus five" in t and "Eight slots" not in t
    assert "Cash lineup, six roster slots" in r.text
    assert "Ranked one lineup at a time by projected points" in t
    low = t.lower()
    for pat in BANNED:
        assert not re.search(pat, low), pat
    assert "kelly" not in low and not re.search(r"\bstake (\$|\d)", low)


def test_classic_slate_page_unchanged(data_dir, monkeypatch):
    c = _client(data_dir, monkeypatch, "main")
    r = c.get(f"/dfs/{DAY}/{SLATE}")
    t = text_of(r.text)
    assert SD_GUIDE not in t and 'class="dfs-sd-guide"' not in r.text
    assert "Eight slots" in t and "Cash lineup, eight roster slots" in r.text
    assert "Ranked one lineup at a time" not in t


def test_showdown_builder_page(data_dir, monkeypatch):
    c = _client(data_dir, monkeypatch, "showdown")
    r = c.get(URL)
    assert r.status_code == 200
    t = text_of(r.text)
    assert dfs_contest.GUIDANCE_SHOWDOWN[0] in t
    assert "(of 6)" in t and "(of 8)" not in t
    sel = r.text.split('name="gamma"')[1].split("</select>")[0]
    assert re.findall(r'value="(\d)"', sel) == ["3", "4", "5"]
    low = t.lower()
    for pat in BANNED:
        assert not re.search(pat, low), pat
    assert "kelly" not in low


def test_classic_builder_page_unchanged(data_dir, monkeypatch):
    c = _client(data_dir, monkeypatch, "main")
    t = text_of(c.get(URL).text)
    assert dfs_contest.GUIDANCE_SHOWDOWN[0] not in t and "(of 8)" in t
