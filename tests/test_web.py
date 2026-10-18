"""Web front end: routes, null-text fallbacks, empty Briefs, game pages, JSON passthrough.

No network: the app reads Briefs from a temporary data dir (conftest sets COURTSIDE_DATA).
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside.web import fmt, view  # noqa: E402
from courtside.web.app import create_app  # noqa: E402
from courtside.web.store import BriefStore  # noqa: E402

SAMPLE = Path(__file__).parent / "fixtures" / "brief" / "sample_brief.json"
DAY = "2026-02-20"


def sample() -> dict:
    return json.loads(SAMPLE.read_text())


def empty_brief(day: str = "2026-02-15") -> dict:
    """A schema-1.0 Brief for an off day: every list empty, every optional section absent."""
    b = sample()
    b["meta"].update(day=day, last_night_day="2026-02-14", notes=["no games scheduled"],
                     text_filled=False, path=f"briefs/{day}.json")
    b["headline"] = {"text": None}
    b["last_night"] = {"day": "2026-02-14", "games": [], "text": None,
                       "summary": {"games": 0, "final": 0, "picks_scored": 0, "picks_correct": 0,
                                   "market_favorites_scored": 0,
                                   "market_favorites_correct": 0}}
    b["team_pulse"] = {"available": False, "reason": "league_facts failed", "text": None}
    b["player_pulse"] = {"available": False, "reason": "league_facts failed", "text": None}
    b["injury_watch"] = {"available": False, "as_of": None, "latest_snapshot": None,
                         "baseline_snapshot": None, "diff_available": False, "sources": [],
                         "changes": {"new_absences": [], "returns": [], "status_changes": []},
                         "out_tonight": [], "text": None}
    b["tonight"] = {"day": day, "games": [], "upset_watch": [], "text": None,
                    "summary": {"games": 0, "with_prediction": 0, "with_market": 0,
                                "upset_watch": 0}}
    sc = b["scorecard"]
    for k in ("n", "correct", "incorrect"):
        sc[k] = 0
    for k in ("accuracy", "brier", "log_loss", "ece", "mean_pick_prob", "vs_market", "through"):
        sc[k] = None
    sc.update(calibration=[], by_confidence=[],
              recent={"last_night": {"n": 0, "correct": 0}, "last_7_days": {"n": 0, "correct": 0}})
    return b


def write(data_dir: Path, brief: dict) -> None:
    d = data_dir / "briefs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{brief['meta']['day']}.json").write_text(json.dumps(brief))


@pytest.fixture
def client(data_dir):
    d = data_dir / "briefs"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE, d / f"{DAY}.json")
    return TestClient(create_app(data_dir))


def no_none(html: str) -> None:
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert not re.search(r"\bNone\b", text), re.search(r".{40}\bNone\b.{40}", text).group(0)
    assert "nan" not in re.findall(r">\s*(nan)\s*<", text)
    assert "undefined" not in text


# ------------------------------------------------------------------ routes

@pytest.mark.parametrize("url", ["/", f"/brief/{DAY}", "/archive", "/healthz",
                                 f"/api/brief/{DAY}", "/api/brief/latest"])
def test_routes_200_on_sample(client, url):
    r = client.get(url)
    assert r.status_code == 200
    if "html" in r.headers["content-type"]:
        no_none(r.text)


def test_home_shows_every_section_and_structured_facts(client):
    html = client.get("/").text
    for sec in ("last-night", "teams", "players", "injuries", "tonight", "scorecard"):
        assert f'id="{sec}"' in html
    b = sample()
    for g in b["last_night"]["games"]:
        assert f'/game/{g["game_id"]}' in html
    assert "Upset watch" in html
    assert "The market is sharper" in html          # honest scorecard (model_beats_market false)
    assert "doesn&rsquo;t price injuries" in html     # model_uses_injuries false caveat
    assert "calib-chart" in html and "<svg" in html


def test_null_text_fallbacks_are_built_from_facts(client):
    html = client.get("/").text
    # masthead headline + per-game headline/story come from the facts when text is null
    assert "Clippers edge Nuggets" in html
    assert "turned a 74–64 deficit into an 81–79 lead" in html
    assert "Ten games last night; our picks went 6⁠–⁠4" in html
    assert 'class="gc-head"' in html and 'class="gc-story"' in html


def test_writer_text_is_preferred_when_present(data_dir):
    b = sample()
    b["headline"]["text"] = "A headline from the writer"
    b["last_night"]["games"][0]["text"] = {"headline": "Writer card head",
                                           "story": "Writer story text."}
    b["tonight"]["games"][0]["text"] = {"preview": "Writer preview."}
    b["team_pulse"]["rising"][0]["text"] = "Writer team blurb."
    write(data_dir, b)
    html = TestClient(create_app(data_dir)).get("/").text
    for s in ("A headline from the writer", "Writer card head", "Writer story text.",
              "Writer preview.", "Writer team blurb."):
        assert s in html
    no_none(html)


def test_empty_brief_renders_gracefully(data_dir):
    write(data_dir, empty_brief())
    c = TestClient(create_app(data_dir))
    r = c.get("/brief/2026-02-15")
    assert r.status_code == 200
    no_none(r.text)
    for msg in ("No games last night", "No games tonight", "Team trends are unavailable",
                "No injury reports yet", "No graded picks yet"):
        assert msg in r.text
    assert c.get("/").status_code == 200       # it is also the latest Brief
    assert c.get("/archive").status_code == 200


def test_empty_brief_from_build_brief_renders(data_dir):
    """A real Brief built on an empty database (the writer is off in tests)."""
    from courtside import db
    from courtside.brief import assemble
    conn = db.connect()
    assemble.build_brief(conn, "2026-02-15", run_writer=False)
    conn.close()
    r = TestClient(create_app(data_dir)).get("/")
    assert r.status_code == 200
    no_none(r.text)
    assert "No games last night" in r.text


def test_brief_with_missing_and_odd_values_still_renders(data_dir):
    b = sample()
    g0 = b["last_night"]["games"][0]
    g0.update(facts=None, facts_available=False, prediction=None, correct=None, market=None)
    b["last_night"]["games"][1]["facts"]["turning_point"] = None
    b["last_night"]["games"][1]["facts"]["unsung_hero"] = []
    t0 = b["tonight"]["games"][0]
    t0.update(prediction=None, market=None, edge=None, upset=None, upset_watch=False)
    t0["home"].update(record=None, last10=None, streak=None, power_rank=None)
    b["player_pulse"]["hot"][0]["recent"] = {}
    b["injury_watch"]["out_tonight"][0].update(mpg=None, reason=None)
    b["scorecard"]["vs_market"] = None
    write(data_dir, b)
    c = TestClient(create_app(data_dir))
    for url in ("/", f"/game/{g0['game_id']}", f"/game/{t0['game_id']}"):
        r = c.get(url)
        assert r.status_code == 200, url
        no_none(r.text)


# ------------------------------------------------------------------ game pages

@pytest.mark.parametrize("gid", [g["game_id"] for g in sample()["last_night"]["games"]])
def test_game_page_renders_for_every_game(client, gid):
    r = client.get(f"/game/{gid}")
    assert r.status_code == 200
    no_none(r.text)
    for s in ("Game flow", "Scoring runs", "Impact", "Clutch", "Notable lines"):
        assert s in r.text
    assert "lead-chart" in r.text


def test_tonight_game_page_is_a_preview(client):
    gid = sample()["tonight"]["games"][0]["game_id"]
    r = client.get(f"/game/{gid}")
    assert r.status_code == 200 and "Injury report" in r.text
    no_none(r.text)


def test_unknown_pages_are_404(client):
    assert client.get("/game/0000000000").status_code == 404
    assert client.get("/brief/2026-01-01").status_code == 404
    assert client.get("/brief/not-a-date").status_code == 404
    assert client.get("/nowhere").status_code == 404
    assert client.get("/api/brief/2026-01-01").status_code == 404


# ------------------------------------------------------------------ api / store

def test_api_json_matches_stored_brief(client):
    assert client.get(f"/api/brief/{DAY}").json() == sample()
    assert client.get("/api/brief/latest").json() == sample()


def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["ok"] is True and h["latest_day"] == DAY and h["briefs"] == 1


def test_store_reads_db_and_prefers_newer_build(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "courtside.db")
    conn.execute("CREATE TABLE briefs (nba_day TEXT PRIMARY KEY, brief_json TEXT, built_at TEXT, "
                 "version TEXT)")
    db_only = empty_brief("2026-02-10")
    db_only["meta"]["built_at"] = "2026-02-10T12:00:00Z"
    newer = sample()
    newer["meta"]["built_at"] = "2099-01-01T00:00:00Z"
    newer["headline"]["text"] = "From the database"
    for b in (db_only, newer):
        conn.execute("INSERT INTO briefs VALUES (?,?,?,?)",
                     (b["meta"]["day"], json.dumps(b), b["meta"]["built_at"], "1.0"))
    conn.commit()
    conn.close()
    write(data_dir, sample())
    store = BriefStore(data_dir)
    assert store.days() == ["2026-02-10", DAY]
    assert store.get(DAY)["headline"]["text"] == "From the database"
    assert store.neighbours(DAY) == ("2026-02-10", None)


def test_store_caches_by_mtime(data_dir):
    write(data_dir, sample())
    store = BriefStore(data_dir)
    a = store.get(DAY)
    assert store.get(DAY) is a                    # cached object, no re-parse
    b = sample()
    b["headline"]["text"] = "changed"
    p = data_dir / "briefs" / f"{DAY}.json"
    p.write_text(json.dumps(b) + " ")             # size changes even if mtime resolution is coarse
    assert store.get(DAY)["headline"]["text"] == "changed"


# ------------------------------------------------------------------ formatting / views

def test_number_formatting():
    assert fmt.signed(-3.8) == "−3.8" and fmt.signed(14.46) == "+14.5" and fmt.signed(0) == "0.0"
    assert fmt.signed(None) == "—"
    assert fmt.pct(0.6044) == "60%" and fmt.pct(0.6441, 1) == "64.4%"
    assert fmt.num(117.44) == "117.4" and fmt.ml(-192) == "−192" and fmt.ml(160) == "+160"
    assert fmt.spread_text(None, 4.5, "CHA", "HOU") == "HOU −4.5"
    assert fmt.spread_text(None, -2.5, "MEM", "UTA") == "MEM −2.5"
    assert fmt.margin_text(1.14, "CHA", "CLE") == "CHA by 1.1"
    assert fmt.injury_reason("Injury/Illness - Right Knee; MCL Sprain; Quad Contusion") == \
        "Right knee MCL sprain, quad contusion"


def test_view_handles_a_bare_brief():
    vm = view.brief_view({"meta": {"day": DAY}})
    assert vm["last_night"]["cards"] == [] and vm["tonight"]["games"] == []
    assert vm["masthead"]["headline"]


def test_fallback_card_text_uses_only_facts():
    b = sample()
    cards = [view.card_view(copy.deepcopy(g)) for g in b["last_night"]["games"]]
    for c, g in zip(cards, b["last_night"]["games"]):
        assert c["headline"] and c["story"] and c["headline_auto"] and c["story_auto"]
        assert c["home"]["nickname"] in c["headline"] or c["away"]["nickname"] in c["headline"]
        for n in re.findall(r"\d+", c["story"]):   # every number in the story is in the facts
            assert n in json.dumps(g, ensure_ascii=False), (n, c["story"])
