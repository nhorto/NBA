"""Brief "Player stories": post-game why (proj/why_v0) -> Brief section -> writer/factcheck ->
template fallback -> text and web rendering. The LLM is always a stub."""
from __future__ import annotations

import copy
import json
import warnings
from datetime import date
from pathlib import Path

from test_proj_why import DAY, GID, RATES, line, seed_game, seed_player, seeded

from courtside import db
from courtside.brief import assemble, factcheck, llm, show, templates, writer
from courtside.proj import why_v0 as W

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside.web import view  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

LN = date.fromisoformat(DAY)                       # the stories' night ...
BRIEF_DAY = date(2026, 11, 3)                      # ... is the next morning's "last night"
FIX = Path(__file__).parent / "fixtures" / "brief"


def stories_db(tmp_path, *, foul_out=True):
    conn, rates = seeded(tmp_path)
    # drop the scratched player so the game's two stories are Brown (below) and Duren (above)
    conn.execute("DELETE FROM player_projection_stats WHERE projection_id IN "
                 "(SELECT id FROM player_projections WHERE player_id=4)")
    conn.execute("DELETE FROM player_projections WHERE player_id=4")
    if foul_out:
        conn.execute("UPDATE player_games SET pf=6 WHERE player_id=2")
        conn.commit()
    W.build(conn, DAY, llm=False, rates=rates)
    return conn


def section(conn, day=LN, last_night=None):
    notes: list[str] = []
    return assemble.player_stories_section(conn, day, last_night or {"games": []}, notes), notes


def scope(e):
    return {**writer.story_facts(e), "unit_values": writer.story_units(e)}


def check(text, e):
    return factcheck.check(text, scope(e), kind="player_story")


# ============================================================================ assembly

def test_section_with_stories(tmp_path):
    conn = stories_db(tmp_path)
    sec, notes = section(conn)
    assert notes == [] and sec["available"] and sec["day"] == DAY and sec["text"] is None
    st = sec["stories"]
    assert [e["player"] for e in st] == ["Jaylen Brown", "Jalen Duren"]
    assert all(abs(a["dk"]["diff"]) >= abs(b["dk"]["diff"]) for a, b in zip(st, st[1:]))
    brown, duren = st
    assert duren["direction"] == "above" and brown["direction"] == "below"
    assert duren["minutes"] == {"actual": 38, "expected": 29, "diff": 9}
    assert duren["line"]["reb"] == 21 and duren["projected"]["reb"] == 8.5
    kinds = [r["kind"] for r in duren["reasons"]]
    assert "minutes" in kinds and "reb" in kinds and len(kinds) <= assemble.STORY_MAX_REASONS
    # every reason points the way the box does against the projection
    for e in st:
        for r in e["reasons"]:
            if r["kind"] in ("minutes", "pts", "reb", "ast", "fg3m", "stl", "blk"):
                assert (r["actual"] > r["projected"]) == (r["dk_points"] > 0)
            elif r["kind"] == "tov":
                assert (r["actual"] < r["projected"]) == (r["dk_points"] > 0)
    assert brown["context"].get("fouled_out") is True
    assert duren["context"]["overtime"] == 1 and duren["context"]["blowout_margin"] == 26
    assert duren["text"] is None and duren["why_text"].startswith("Jalen Duren: ")
    json.dumps(sec, allow_nan=False)


def test_section_limits_did_not_play_and_per_game(tmp_path):
    conn = db.connect(tmp_path / "w.db")
    seed_game(conn)
    seed_game(conn, gid="0022600002", home="NYK", away="MIA", hs=110, as_=108)
    for i in range(3):                               # three big scratches in one game
        seed_player(conn, GID, 10 + i, f"Scratch {i}", "DET", line(mn=0.0),
                    exp_min=30, exp_dk=40.0 + i, stats={"pts": 18.0, "reb": 6.0, "ast": 4.0,
                                                        "fg3m": 2.0})
    seed_player(conn, "0022600002", 20, "Big Night", "NYK",
                line(pts=40, fg3m=7, reb=8, ast=6, stl=2, blk=0, tov=2, mn=40),
                exp_min=34, exp_dk=36.0, stats={"pts": 22.0, "reb": 5.0, "ast": 4.0, "fg3m": 2.5})
    conn.commit()
    W.build(conn, DAY, llm=False, rates={p: dict(RATES) for p in (10, 11, 12, 20)})
    sec, _ = section(conn)
    names = [e["player"] for e in sec["stories"]]
    assert sum(e["did_not_play"] for e in sec["stories"]) == 1
    assert names == ["Scratch 2", "Big Night"]
    dnp = sec["stories"][0]
    assert dnp["line"] is None and dnp["reasons"] == [] and dnp["minutes"]["actual"] == 0
    # top_stories without the limit still returns them all (2 per game)
    assert len(W.top_stories(conn, DAY, n=5, per_game=2)) == 3


def test_section_without_stories_is_silent(tmp_path):
    conn = stories_db(tmp_path)
    sec, notes = section(conn, day=date(2026, 11, 5))           # no games / no projections
    assert sec["available"] is False and sec["stories"] == [] and notes == []
    # finals but no why rows yet (the postgame why step has not run)
    conn2, _ = seeded(tmp_path / "b")
    sec, notes = section(conn2)
    assert sec["available"] is False and sec["stories"] == [] and notes == []
    # a database without the projection tables at all
    bare = db.connect(tmp_path / "bare.db")
    bare.execute("DROP TABLE postgame_why_v0")
    bare.execute("DROP TABLE player_projections")
    sec, notes = section(bare)
    assert sec["available"] is False and notes == []


def test_build_brief_with_and_without_stories(tmp_path):
    conn = stories_db(tmp_path)
    b = assemble.build_brief(conn, BRIEF_DAY, write=False, run_writer=False)
    assert b["meta"]["schema_version"] == "1.2"
    ps = b["player_stories"]
    assert ps["available"] and ps["day"] == DAY and len(ps["stories"]) == 2
    assert not any("player stories" in n for n in b["meta"]["notes"])
    stored = json.loads(conn.execute("SELECT brief_json FROM briefs WHERE nba_day=?",
                                     (BRIEF_DAY.isoformat(),)).fetchone()[0])
    assert stored["player_stories"] == ps
    out = show.format_brief(b)
    assert "PLAYER STORIES" in out
    assert "Jalen Duren scored 59.2 DK points vs 34.0 projected" in out

    later = assemble.build_brief(conn, date(2026, 11, 10), write=False, run_writer=False)
    assert later["player_stories"]["available"] is False
    assert later["player_stories"]["stories"] == []
    assert "PLAYER STORIES" not in show.format_brief(later)


def test_rebuild_carries_story_text(tmp_path):
    conn = stories_db(tmp_path)
    b = assemble.build_brief(conn, BRIEF_DAY, write=False, run_writer=False)
    b["player_stories"]["stories"][0]["text"] = "Kept prose."
    with conn:
        conn.execute("UPDATE briefs SET brief_json=? WHERE nba_day=?",
                     (json.dumps(b), BRIEF_DAY.isoformat()))
    b2 = assemble.build_brief(conn, BRIEF_DAY, write=False, run_writer=False)
    assert b2["player_stories"]["stories"][0]["text"] == "Kept prose."
    assert b2["player_stories"]["stories"][1]["text"] is None


# ============================================================================ factcheck

def fixture_stories():
    return json.loads((FIX / "sample_brief_gbm.json").read_text())["player_stories"]["stories"]


def all_stories(tmp_path):
    conn = stories_db(tmp_path)
    sec, _ = section(conn)
    return sec["stories"] + fixture_stories()


def test_template_lines_pass_the_factcheck(tmp_path):
    for e in all_stories(tmp_path):
        t = templates.player_story(e)
        res = check(t, e)
        assert res.ok, (t, res.messages())
        assert "DK points" in t and e["player"] in t
        assert not factcheck.STORY_BANNED_RE.search(t)


def test_template_wording(tmp_path):
    conn = stories_db(tmp_path)
    brown, duren = section(conn)[0]["stories"]
    t = templates.player_story(duren)
    assert t.startswith("Jalen Duren scored 59.2 DK points vs 34.0 projected: he ")
    assert t.startswith("Jalen Duren scored 59.2 DK points vs 34.0 projected: he grabbed 21 "
                        "rebounds (8.5 projected), played 38 minutes (29 expected) and had 22 "
                        "points (13 projected).")
    t = templates.player_story(brown)
    assert t == ("Jaylen Brown scored 12.8 DK points vs 42.0 projected: he had only 8 points "
                 "(24 projected), played 27 minutes (34 expected) and handed out 2 assists (4 "
                 "projected). "
                 "He fouled out; the game was a 26-point blowout.")
    dnp = {**duren, "did_not_play": True, "line": None, "reasons": [],
           "dk": {"actual": 0.0, "expected": 28.0, "diff": -28.0}, "direction": "below",
           "minutes": {"actual": 0, "expected": 26, "diff": -26}, "context": {}}
    t = templates.player_story(dnp)
    assert t == "Jalen Duren did not play; we had projected 28.0 DK points and about 26 minutes."
    assert check(t, dnp).ok


def test_factcheck_covers_the_story_numbers(tmp_path):
    conn = stories_db(tmp_path)
    brown, duren = section(conn)[0]["stories"]
    good = ("Jalen Duren scored 59.2 DK points vs 34.0 projected: he played 38 minutes, "
            "9 more than expected, and grabbed 21 rebounds.")
    assert check(good, duren).ok, check(good, duren).messages()
    # a minutes figure passed off as DK points, and DK points passed off as minutes
    assert any("DK" in m for m in check("Jalen Duren scored 38 DK points.", duren).messages())
    assert not check("Jalen Duren played 59.2 minutes.", duren).ok
    # a real count that is not his
    assert not check("Jalen Duren grabbed 22 rebounds (8.5 projected).", duren).ok
    # threes: made and attempted are both his; anything else is not
    assert not check("Jalen Duren made 3 threes.", duren).ok
    # an invented number anywhere
    assert not check("Jalen Duren scored 59.2 DK points in 44 minutes.", duren).ok
    # DFS / betting words
    for bad in ("Jalen Duren hit the over with 59.2 DK points.",
                "Jalen Duren had the edge: 59.2 DK points.",
                "A lean toward Jalen Duren paid off with 59.2 DK points.",
                "Jalen Duren was great value at 59.2 DK points."):
        res = check(bad, duren)
        assert any(v.kind == "betting" for v in res.violations), bad
    # direction
    assert any(v.kind == "direction" for v in check(
        "Jaylen Brown beat our projection.", brown).violations)
    assert any(v.kind == "direction" for v in check(
        "Jaylen Brown outperformed us.", brown).violations)
    assert any(v.kind == "direction" for v in check(
        "Jalen Duren fell short of our projection.", duren).violations)
    # unknown names
    assert not check("Jalen Duren and Cade Cunningham scored 59.2 DK points.", duren).ok


ALLOWED = (
    "Jaylen Brown turned it over 4 times and scored 12.8 DK points vs 42.0 projected.",
    "Jaylen Brown scored 12.8 DK points, well under the 42.0 we projected.",
    "Jaylen Brown played 27 minutes, under the 34 we expected, and fouled out.",
)
ALLOWED_DUREN = (
    "Jalen Duren played 38 minutes, well over the 29 we expected, and scored 59.2 DK points.",
    "Jalen Duren grabbed 21 rebounds under the basket for 59.2 DK points vs 34.0 projected.",
    "Over 38 minutes, Jalen Duren scored 59.2 DK points vs 34.0 projected.",
)
BANNED = (
    "Jalen Duren hit the over with 59.2 DK points.",
    "Jalen Duren cleared the over: 59.2 DK points.",
    "The under never had a chance against Jalen Duren.",
    "Jalen Duren's over/under was 34.0 DK points.",
    "Jalen Duren's o/u was 34.0 DK points.",
    "Jalen Duren covered with 59.2 DK points.",
    "Jalen Duren was a smash at 59.2 DK points.",
    "Jalen Duren was a lock for 59.2 DK points.",
    "Jalen Duren had the edge: 59.2 DK points.",
    "A lean toward Jalen Duren paid off.",
)


def test_plain_over_and_under_are_allowed_betting_phrases_are_not(tmp_path):
    conn = stories_db(tmp_path)
    brown, duren = section(conn)[0]["stories"]
    for t in ALLOWED:
        assert check(t, brown).ok, (t, check(t, brown).messages())
    for t in ALLOWED_DUREN:
        assert check(t, duren).ok, (t, check(t, duren).messages())
    for t in BANNED:
        res = check(t, duren)
        assert any(v.kind == "betting" for v in res.violations), (t, res.messages())


def test_rotation_floor(tmp_path):
    conn = db.connect(tmp_path / "w.db")
    seed_game(conn)
    seed_game(conn, gid="0022600002", home="NYK", away="MIA", hs=110, as_=108)
    # garbage-time bench player: biggest gap of the night, but projected 3.0 DK in 4 minutes
    seed_player(conn, GID, 30, "Deep Bench", "DET",
                line(pts=24, fg3m=4, reb=6, ast=3, stl=2, blk=0, tov=1, mn=26),
                exp_min=4, exp_dk=3.0, stats={"pts": 1.5, "reb": 0.8, "ast": 0.4, "fg3m": 0.3})
    # rotation by minutes only (projected DK below 15)
    seed_player(conn, GID, 31, "Minutes Guy", "DET",
                line(pts=20, fg3m=2, reb=7, ast=2, stl=1, blk=1, tov=1, mn=30),
                exp_min=21, exp_dk=14.0, stats={"pts": 6.0, "reb": 3.0, "ast": 1.0, "fg3m": 0.5})
    # projected starter who sat: 'all' numbers cut by a Questionable tag (16 min, 13 DK: under
    # both floors; why_v0 itself only writes a DNP projected for >= 15 min), 30 if he plays
    pj = seed_player(conn, "0022600002", 32, "Late Scratch", "NYK", line(mn=0.0),
                     exp_min=16, exp_dk=13.0, stats={"pts": 6.0, "reb": 2.0, "ast": 2.0,
                                                     "fg3m": 0.5})
    conn.execute("UPDATE player_projections SET minutes_json=? WHERE id=?",
                 (json.dumps({"mean_all": 16, "mean_played": 30}), pj))
    # a fringe player who sat: not a story
    seed_player(conn, "0022600002", 33, "Fringe Sit", "NYK", line(mn=0.0),
                exp_min=16, exp_dk=14.5, stats={"pts": 7.0, "reb": 2.0, "ast": 1.0, "fg3m": 0.5})
    conn.commit()
    W.build(conn, DAY, llm=False, rates={p: dict(RATES) for p in (30, 31, 32, 33)})
    assert "Deep Bench" in [r["player"] for r in W.top_stories(conn, DAY)]   # no floor by default
    sec, _ = section(conn)
    names = [e["player"] for e in sec["stories"]]
    assert "Deep Bench" not in names and "Fringe Sit" not in names
    assert names == ["Minutes Guy", "Late Scratch"]
    assert W.is_rotation(15.0, 0, {}, did_not_play=False, min_dk=15, min_minutes=20)
    assert W.is_rotation(5.0, 20.0, {}, did_not_play=False, min_dk=15, min_minutes=20)
    assert not W.is_rotation(14.9, 19.9, {"exp_min_played": 30}, did_not_play=False,
                             min_dk=15, min_minutes=20)
    assert W.is_rotation(5.0, 5.0, {"p_start": 0.8}, did_not_play=True, min_dk=15,
                         min_minutes=20)


def test_unit_check_is_off_without_unit_values():
    # the existing kinds never carry unit_values, so their checks are unchanged
    res = factcheck.check("He played 41 minutes.", {"x": 41}, kind="blurb")
    assert res.ok


# ============================================================================ writer

class Stub:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt, system=None, timeout=None):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else llm.LLMError("stub exhausted")
        if isinstance(r, Exception):
            raise r
        return r


def stories_reply(d):
    return json.dumps({"stories": d})


def test_writer_llm_ok_and_retry(tmp_path):
    conn = stories_db(tmp_path)
    brown, duren = section(conn)[0]["stories"]
    k1, k2 = writer.story_key(duren), writer.story_key(brown)
    good1 = ("Jalen Duren scored 59.2 DK points vs 34.0 projected: 38 minutes instead of 29 "
             "and 21 rebounds did it.")
    bad2 = "Jaylen Brown was a lock to hit the under."
    good2 = "Jaylen Brown scored 12.8 DK points vs 42.0 projected and fouled out."
    assert check(good2, brown).ok, check(good2, brown).messages()
    stub = Stub(stories_reply({k1: good1, k2: bad2}), stories_reply({k2: good2}))
    rep: dict = {}
    out = writer.write_player_stories([duren, brown], complete=stub, report=rep)
    assert out[k1] == {"text": good1, "source": "llm_ok"}
    assert out[k2] == {"text": good2, "source": "llm_retry_ok"}
    assert any("betting" in p for p in rep[k2]["problems"])
    assert len(stub.prompts) == 2 and "Player stories" in stub.prompts[0]
    assert '"unit_values"' not in stub.prompts[0]          # checker-only, not shown to the LLM


def test_template_fallback_when_llm_is_down(tmp_path):
    conn = stories_db(tmp_path)
    b = assemble.build_brief(conn, BRIEF_DAY, write=False, run_writer=False)
    stub = Stub(*[llm.LLMError("no backend")] * 20)
    out = writer.fill(b, complete=stub)
    for e in out["player_stories"]["stories"]:
        assert e["text"] == templates.player_story(e)
        assert out["meta"]["writer"]["outcomes"][f"story:{writer.story_key(e)}"] == "template"
    assert b["player_stories"]["stories"][0]["text"] is None      # fill() works on a copy


def test_template_fallback_when_llm_output_fails_twice(tmp_path):
    conn = stories_db(tmp_path)
    brown, duren = section(conn)[0]["stories"]
    bad = stories_reply({writer.story_key(duren): "Jalen Duren smashed with 99 DK points.",
                         writer.story_key(brown): "Jaylen Brown had the edge."})
    out = writer.write_player_stories([duren, brown], complete=Stub(bad, bad))
    for e in (duren, brown):
        assert out[writer.story_key(e)] == {"text": templates.player_story(e),
                                           "source": "template"}


# ============================================================================ web

def write_brief(data_dir, b):
    d = data_dir / "briefs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{b['meta']['day']}.json").write_text(json.dumps(b))


def test_web_renders_stories_with_template_fallback_and_writer_text(data_dir):
    b = json.loads((FIX / "sample_brief_gbm.json").read_text())
    write_brief(data_dir, b)
    html = TestClient(create_app(data_dir)).get(f"/brief/{b['meta']['day']}").text
    assert 'id="player-stories"' in html and "Player stories" in html
    first = b["player_stories"]["stories"][0]
    assert templates.player_story(first) in html
    assert "None" not in html.split('id="player-stories"')[1].split("</section>")[0]

    b2 = copy.deepcopy(b)
    b2["player_stories"]["stories"][0]["text"] = "Writer prose wins."
    v = view.stories_view(b2)
    assert v["stories"][0]["text"] == "Writer prose wins."
    assert v["stories"][1]["text"] == templates.player_story(b2["player_stories"]["stories"][1])


def test_web_without_stories_renders_no_section(data_dir):
    b = json.loads((FIX / "sample_brief_gbm.json").read_text())
    b["player_stories"] = {"available": False, "day": "2026-02-19", "stories": [], "text": None}
    write_brief(data_dir, b)
    html = TestClient(create_app(data_dir)).get(f"/brief/{b['meta']['day']}").text
    assert 'id="player-stories"' not in html
    old = json.loads((FIX / "sample_brief.json").read_text())      # schema 1.0: no section
    assert view.stories_view(old) == {"day": None, "stories": []}
