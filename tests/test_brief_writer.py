"""brief/writer.py with a fake LLM: batching, fact-check, retry, template fallback, fill()."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from courtside.brief import llm, templates, writer

FIX = Path(__file__).parent / "fixtures" / "brief_writer"


def games():
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(FIX / "game_facts_*.json")))]


def league():
    return json.loads((FIX / "league_facts_2026-02-15.json").read_text())


class Fake:
    """Scripted LLM: each call pops the next reply (a str, a callable(prompt) or an exception)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt, *, system=None, timeout=180, **kw):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r(prompt) if callable(r) else r


def good_stories(fl):
    return json.dumps({"games": {f["game_id"]: templates.game(f) for f in fl}})


# ---- JSON parsing ------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Here you go:\n```\n{"a": 1}\n```\nHope that helps!',
    'Sure! {"a": 1}\n\nLet me know if you want changes.',
    '{"a": 1,}',
    '“{"a": 1}”'.replace("“", "").replace("”", ""),
])
def test_extract_json_tolerates_fences_preamble_and_trailing_text(text):
    assert writer.extract_json(text) == {"a": 1}


def test_extract_json_skips_leading_braces_in_prose_and_handles_arrays():
    assert writer.extract_json('I used {curly} words. [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        writer.extract_json("no json at all")


# ---- game stories ------------------------------------------------------------------------

def test_stories_one_call_all_llm_ok_with_messy_wrapper():
    fl = games()
    fake = Fake("Here are the recaps:\n```json\n" + good_stories(fl) + "\n```\nEnjoy.")
    report = {}
    out = writer.write_game_stories(fl, complete=fake, report=report)
    assert len(fake.prompts) == 1
    assert set(out) == {f["game_id"] for f in fl}
    assert {v["source"] for v in out.values()} == {"llm_ok"}
    assert all(r["problems"] == [] for r in report.values())
    # the prompt carries the style guide, a compacted facts doc and the output format
    p = fake.prompts[0]
    assert "HOUSE STYLE" in p and "OUTPUT FORMAT" in p and "Kawhi Leonard" in p
    assert "validation" not in p and "boxscore" not in p
    assert len(p) < 8000 * len(fl)


def test_invented_number_triggers_one_targeted_retry_then_llm_retry_ok():
    fl = games()
    target = "0022501066"
    first = json.loads(good_stories(fl))
    first["games"][target]["story"] = "Darius Garland scored 31 points as the Clippers rallied."
    seen = {}

    def second(prompt):
        seen["retry"] = prompt
        return json.dumps({"games": {target: templates.game(next(f for f in fl if f["game_id"] == target))}})

    fake = Fake(json.dumps(first), second)
    report = {}
    out = writer.write_game_stories(fl, complete=fake, report=report)
    assert len(fake.prompts) == 2
    assert out[target]["source"] == "llm_retry_ok"
    assert all(v["source"] == "llm_ok" for k, v in out.items() if k != target)
    retry = seen["retry"]
    assert "PROBLEMS" in retry and "'31'" in retry and target in retry
    assert report[target]["source"] == "llm_retry_ok"
    assert any("'31'" in p for p in report[target]["problems"])      # what the first draft got wrong


def test_still_bad_after_retry_falls_back_to_template():
    fl = games()
    target = "0022501066"
    bad = json.loads(good_stories(fl))
    bad["games"][target] = {"headline": "Stephen Curry rescues Clippers", "story": "Curry hit 9 threes."}
    fake = Fake(json.dumps(bad), json.dumps({"games": {target: bad["games"][target]}}))
    report = {}
    out = writer.write_game_stories(fl, complete=fake, report=report)
    assert len(fake.prompts) == 2                      # exactly one retry
    f = next(f for f in fl if f["game_id"] == target)
    assert out[target]["source"] == "template"
    assert out[target]["headline"] == templates.game(f)["headline"]
    assert out[target]["story"] == templates.game(f)["story"]
    assert any("Curry" in p for p in report[target]["problems"])


def test_missing_game_and_unparseable_reply_are_retried():
    fl = games()[:2]
    partial = json.dumps({"games": {fl[0]["game_id"]: templates.game(fl[0])}})
    fake = Fake(partial, "totally not json", )
    out = writer.write_game_stories(fl, complete=fake)
    assert out[fl[0]["game_id"]]["source"] == "llm_ok"
    assert out[fl[1]["game_id"]]["source"] == "template"
    assert len(fake.prompts) == 2


def test_llm_down_means_templates_and_a_single_attempt():
    fl = games()
    fake = Fake(llm.AllBackendsFailed({"claude": "Not logged in", "codex": "timeout"}))
    out = writer.write_game_stories(fl, complete=fake)
    assert len(fake.prompts) == 1                      # no point retrying a dead backend
    assert {v["source"] for v in out.values()} == {"template"}
    assert all(v["headline"] and v["story"] for v in out.values())


def test_game_failing_stint_validation_is_never_sent_to_the_model():
    fl = games()
    fl[0] = {**fl[0], "validation": {**fl[0]["validation"], "ok": False}}
    fake = Fake(good_stories(fl[1:]))
    out = writer.write_game_stories(fl, complete=fake)
    assert out[fl[0]["game_id"]]["source"] == "template"
    assert fl[0]["game_id"] not in fake.prompts[0].replace('"game_id"', "")  # id absent from facts
    assert len(fake.prompts) == 1


def test_stories_empty_input():
    assert writer.write_game_stories([], complete=Fake()) == {}


def test_compact_game_is_smaller_and_drops_noise():
    f = games()[-1]
    c = writer.compact_game(f)
    assert len(json.dumps(c)) < len(json.dumps(f)) * 0.8
    s = json.dumps(c)
    assert "validation" not in s and "on_off" not in s and "game_score" not in s
    assert c["comeback"]["deficit"] == 24 and c["late_key_plays"][-1]["decisive"] is True
    # a 2-point "deficit" is not a comeback
    assert "comeback" not in writer.compact_game(next(g for g in games() if g["game_id"] == "0022500622"))


def test_cache_means_second_build_costs_no_calls(monkeypatch):
    fl = games()
    calls = []

    def backend(prompt, system, timeout):
        calls.append(prompt)
        return good_stories(fl)

    monkeypatch.setattr(llm, "BACKENDS", {"claude": backend})
    monkeypatch.setenv("COURTSIDE_LLM", "claude")
    a = writer.write_game_stories(fl)
    b = writer.write_game_stories(fl)
    assert a == b and len(calls) == 1


# ---- pulse -------------------------------------------------------------------------------

def good_pulse(lf):
    teams, players = writer._team_entities(lf), writer._player_entities(lf)
    return {
        "headline": templates.pulse_headline(teams, players),
        "teams": {e["team"]: templates.team_blurb(e) for e in teams},
        "players": {e["key"]: templates.player_blurb(e) for e in players},
    }


def test_pulse_ok_and_shape():
    lf = league()
    fake = Fake(json.dumps(good_pulse(lf)))
    out = writer.write_pulse(lf, complete=fake)
    assert len(fake.prompts) == 1
    assert set(out) == {"headline", "teams", "players", "sources"}
    assert "NYK" in out["teams"] and out["headline"]
    assert set(out["sources"].values()) == {"llm_ok"}
    assert len(out["teams"]) <= 8 and len(out["players"]) <= 12
    p = fake.prompts[0]
    assert "61.7" in p and "76.0" in p                 # validity numbers reach the writer for hedging


def test_pulse_blurb_scoped_to_its_own_entity_and_retried():
    lf = league()
    g = good_pulse(lf)
    g["teams"]["NYK"] = "The Knicks are 8-2 and the Bulls are 1-9."          # Bulls: another entity
    bad_key = next(iter(g["players"]))
    g["players"][bad_key] = "He averaged 99.9 points."
    fixed = {"teams": {"NYK": good_pulse(lf)["teams"]["NYK"]},
             "players": {bad_key: good_pulse(lf)["players"][bad_key]}}
    fake = Fake(json.dumps(g), json.dumps(fixed))
    out = writer.write_pulse(lf, complete=fake)
    assert out["sources"]["team:NYK"] == "llm_retry_ok"
    assert out["sources"][f"player:{bad_key}"] == "llm_retry_ok"
    assert out["sources"]["headline"] == "llm_ok"
    assert len(fake.prompts) == 2


def test_pulse_with_nothing_to_say_makes_no_call():
    out = writer.write_pulse({"teams": {}, "players": {}, "method": {}}, complete=Fake())
    assert out["teams"] == {} and out["players"] == {} and out["headline"]


# ---- tonight -----------------------------------------------------------------------------

PREDS = [
    {"game_id": "0022500900", "home": "BOS", "away": "MIA", "home_win_prob": 0.71,
     "home_margin": 7.2, "total": 228.5, "market_home_spread": -4.5,
     "reasons": ["Miami is on the second night of a back-to-back"],
     "injuries": [{"player": "Bam Adebayo", "team": "MIA", "status": "out"}]},
    {"game_id": "0022500901", "home": "Denver Nuggets", "away": "PHX", "p_home": 38,
     "margin": -1.5, "spread": 1.0},
]


def test_normalize_prediction_is_tolerant():
    n = writer.normalize_prediction(PREDS[1])
    assert n["home"]["team"] == "DEN" and n["away"]["team"] == "PHX"
    assert n["home_win_prob"] == 0.38 and n["away_win_prob"] == 0.62
    assert n["projected_home_margin"] == -1.5 and n["our_number_minus_market_pts"] == -0.5
    n0 = writer.normalize_prediction(PREDS[0])
    assert n0["injuries"] == ["Bam Adebayo (Heat): out"] and n0["our_number_minus_market_pts"] == 2.7


def test_tonight_ok_betting_language_is_retried_then_template():
    good = {p["game_id"]: templates.tonight(writer._tonight_template_input(writer.normalize_prediction(p)))
            for p in PREDS}
    bet = dict(good)
    bet["0022500900"] = "Take the Celtics, a lock at -4.5."
    fake = Fake(json.dumps({"games": bet}), json.dumps({"games": {"0022500900": "Best bet: Boston."}}))
    rep = {}
    out = writer.write_tonight(PREDS, complete=fake, report=rep)
    assert rep["0022500900"]["source"] == "template" and rep["0022500901"]["source"] == "llm_ok"
    assert out["0022500900"] == good["0022500900"]
    assert len(fake.prompts) == 2
    assert "2.7" in fake.prompts[0]                    # the model-vs-market gap is precomputed


def test_tonight_sentence_overrun_and_invented_player_fail():
    long = "Boston is a 71% favorite. Miami is tired. The Heat lost. Again. Really."
    inv = "Boston by 7, with Jimmy Butler out."
    fake = Fake(json.dumps({"games": {"0022500900": long, "0022500901": inv}}),
                json.dumps({"games": {"0022500900": long, "0022500901": inv}}))
    rep = {}
    writer.write_tonight(PREDS, complete=fake, report=rep)
    assert {r["source"] for r in rep.values()} == {"template"}


# ---- brief headline ----------------------------------------------------------------------

def test_brief_headline_checked_against_summary():
    summary = {"games": [{"game_id": "1", "headline": "Clippers erase 24-point deficit to beat Pacers",
                          "home": "IND", "away": "LAC", "margin": 1, "comeback_deficit": 24}]}
    ok = writer.write_brief_headline(summary, complete=Fake('{"headline": "Clippers erase a 24-point hole in Indiana"}'))
    assert ok == {"headline": "Clippers erase a 24-point hole in Indiana", "source": "llm_ok"}
    bad = writer.write_brief_headline(summary, complete=Fake('{"headline": "Lakers stun Pacers by 30"}',
                                                             '{"headline": "Lakers stun Pacers by 30"}'))
    assert bad["source"] == "template" and bad["headline"] == "Clippers erase 24-point deficit to beat Pacers"


# ---- fill (docs/BRIEF.md shapes) ---------------------------------------------------------

GAME_SHAPED = {
    "game_id": "0022500950", "status": "scheduled",
    "home": {"team": "BOS", "name": "Boston Celtics", "record": "40-15", "last10": "8-2", "streak": "W3",
             "power_rank": 2},
    "away": {"team": "MIA", "name": "Miami Heat", "record": "30-25", "last10": "5-5", "streak": "L1",
             "power_rank": 14},
    "prediction": {"home_win_prob": 0.71, "away_win_prob": 0.29, "home_margin": 7.2, "total": 228.5,
                   "pick": "BOS", "pick_prob": 0.71, "confidence": "strong"},
    "market": {"home_spread": -4.5, "total": 226.0, "home_prob": 0.66},
    "edge": {"win_prob": 0.05, "margin": 2.7, "total": 2.5},
    "factors": [{"kind": "back_to_back", "team": "MIA", "side": "away", "value": True, "favors": "home",
                 "detail": "Miami is on the second night of a back-to-back"},
                {"kind": "injury", "team": "MIA", "side": "away",
                 "value": {"player": "Bam Adebayo", "status": "Out"}, "favors": "home",
                 "detail": "Bam Adebayo (Miami) is out"}],
    "injuries": {"home": [], "away": [{"team": "MIA", "player": "Bam Adebayo", "status": "Out",
                                       "reason": "knee", "mpg": 34.1}]},
    "upset_watch": False, "upset": None, "text": {"preview": None},
}


def test_normalize_accepts_brief_game_shape():
    n = writer.normalize_prediction(GAME_SHAPED)
    assert n["home"] == {"team": "BOS", "nickname": "Celtics", "record": "40-15", "last10": "8-2",
                         "streak": "W3", "power_rank": 2}
    assert n["home_win_prob"] == 0.71 and n["projected_home_margin"] == 7.2
    assert n["market_home_spread"] == -4.5 and n["market_total"] == 226.0
    assert n["our_number_minus_market_pts"] == 2.7
    assert n["reasons"][0].startswith("Miami is on the second night")
    assert n["injuries"] == ["Bam Adebayo (Heat): Out"]
    assert "upset_watch" not in n and "model_prices_injuries" not in n
    n2 = writer.normalize_prediction({**GAME_SHAPED, "model_uses_injuries": False})
    assert n2["model_prices_injuries"] is False


def synthetic_brief():
    fl = games()
    lf = league()
    tr = lf["teams"]
    pl = lf["players"]

    def slot(e):
        return {**e, "text": None}
    return {
        "meta": {"schema_version": "1.0", "day": "2026-02-16", "text_filled": False},
        "headline": {"text": None},
        "last_night": {"day": "2026-02-15", "text": None, "games": [
            {"game_id": fl[4]["game_id"], "final": True, "facts": fl[4], "facts_available": True,
             "home": {"team": "IND", "name": "Indiana Pacers", "score": 113},
             "away": {"team": "LAC", "name": "LA Clippers", "score": 114},
             "result": {"home_score": 113, "away_score": 114, "winner": "LAC", "margin": 1,
                        "total": 227, "overtimes": 0},
             "prediction": {"pick": "IND", "pick_prob": 0.62, "confidence": "lean"}, "correct": False,
             "market": None, "text": {"headline": None, "story": None}},
            {"game_id": fl[3]["game_id"], "final": True, "facts": fl[3], "facts_available": True,
             "home": {"team": "NYK", "name": "New York Knicks", "score": 120},
             "away": {"team": "BKN", "name": "Brooklyn Nets", "score": 66},
             "result": {"home_score": 120, "away_score": 66, "winner": "NYK", "margin": 54,
                        "total": 186, "overtimes": 0},
             "prediction": None, "correct": None, "market": None,
             "text": {"headline": "Already written", "story": None}},
            {"game_id": "0022500999", "final": True, "facts": None, "facts_available": False,
             "home": {"team": "UTA", "name": "Utah Jazz", "score": 100},
             "away": {"team": "MEM", "name": "Memphis Grizzlies", "score": 99},
             "result": None, "prediction": None, "correct": None, "market": None,
             "text": {"headline": None, "story": None}},
        ]},
        "team_pulse": {"available": True, "as_of": "2026-02-16", "method": lf["method"],
                       "games_per_team_median": 55, "early_season": False, "notes": [],
                       "rising": [slot(tr["rising"][0])], "sliding": [slot(tr["sliding"][0])],
                       "power_top10": tr["power_top10"], "power_movers_7d": tr["power_movers_7d"],
                       "streaks": tr["streaks"], "text": None},
        "player_pulse": {"available": True, "as_of": "2026-02-16", "method": lf["method"],
                         "hot": [slot(pl["hot"][0])], "cold": [slot(pl["cold"][0])],
                         "role_changes": pl["role_changes"][:2], "leaders_last10": pl["leaders_last10"],
                         "streaks": [], "career_highs_last7d": [], "text": None},
        "injury_watch": {"available": False, "text": None},
        "tonight": {"day": "2026-02-16", "text": None, "upset_watch": [],
                    "games": [GAME_SHAPED, {**GAME_SHAPED, "game_id": "0022500951",
                                            "home": {"team": "DEN", "name": "Denver Nuggets"},
                                            "away": {"team": "PHX", "name": "Phoenix Suns"},
                                            "prediction": {"home_win_prob": 0.38, "home_margin": -1.5,
                                                           "pick": "PHX", "pick_prob": 0.62},
                                            "market": None, "edge": None, "factors": [],
                                            "injuries": {"home": [], "away": []}}]},
        "scorecard": {"text": None},
    }


def scripted_for_fill(brief):
    fl = [c["facts"] for c in brief["last_night"]["games"] if c["facts"]]
    tp, pp = brief["team_pulse"], brief["player_pulse"]

    def pulse(prompt):
        teams = [writer.compact_team(tp["rising"][0], "rising"), writer.compact_team(tp["sliding"][0], "sliding")]
        pi = writer._player_entities({"players": {"hot": pp["hot"], "cold": pp["cold"]}})
        return json.dumps({
            "headline": templates.pulse_headline(teams, pi),
            "teams": {t["team"]: templates.team_blurb(t) for t in teams},
            "players": {p["key"]: templates.player_blurb(p) for p in pi}})

    def tonight(prompt):
        return json.dumps({"games": {g["game_id"]: templates.tonight(
            writer._tonight_template_input(writer.normalize_prediction(g)))
            for g in brief["tonight"]["games"]}})

    return [lambda p: good_stories(fl), pulse, tonight,
            lambda p: json.dumps({"headline": "Clippers erase a 24-point hole in Indiana"})]


def test_fill_synthetic_brief_fills_every_slot_it_owns_with_four_calls():
    brief = synthetic_brief()
    original = json.dumps(brief, sort_keys=True)
    fake = Fake(*scripted_for_fill(brief))
    out = writer.fill(brief, complete=fake)
    assert json.dumps(brief, sort_keys=True) == original                # input untouched
    json.dumps(out, allow_nan=False)
    assert len(fake.prompts) == 4
    c0, c1, c2 = out["last_night"]["games"]
    assert c0["text"]["headline"] and c0["text"]["story"]
    assert c1["text"]["headline"] == "Already written" and c1["text"]["story"]   # not overwritten
    assert c2["text"] == {"headline": None, "story": None}                      # no facts: stays null
    assert out["team_pulse"]["rising"][0]["text"] and out["team_pulse"]["sliding"][0]["text"]
    assert out["team_pulse"]["text"]                                             # pulse headline
    assert out["player_pulse"]["hot"][0]["text"] and out["player_pulse"]["cold"][0]["text"]
    assert out["player_pulse"]["role_changes"][0].get("text") is None            # no slot -> none made
    assert all(g["text"]["preview"] for g in out["tonight"]["games"])
    assert out["headline"]["text"] == "Clippers erase a 24-point hole in Indiana"
    assert out["last_night"]["text"] is None and out["scorecard"]["text"] is None
    meta = out["meta"]["writer"]
    assert meta["counts"] == {"llm_ok": len(meta["outcomes"])}
    assert out["meta"]["schema_version"] == "1.0" and out["meta"]["text_filled"] is False
    # context reaches the stories prompt; the headline prompt sees the written game headline
    assert "our_pregame_model" in fake.prompts[0]
    assert c0["text"]["headline"] in fake.prompts[3]


def test_fill_with_dead_llm_uses_templates_everywhere():
    brief = synthetic_brief()
    dead = llm.AllBackendsFailed({"claude": "Not logged in", "codex": "not installed"})
    out = writer.fill(brief, complete=Fake(dead, dead, dead, dead))
    assert out["last_night"]["games"][0]["text"]["story"]
    assert all(g["text"]["preview"] for g in out["tonight"]["games"])
    assert out["headline"]["text"] == templates.brief_headline(writer.brief_summary(out))
    counts = out["meta"]["writer"]["counts"]
    assert set(counts) == {"template"} and counts["template"] >= 8


def test_fill_is_a_noop_when_nothing_is_empty_and_ignores_unknown_shapes():
    fl = games()
    brief = {"headline": {"text": "H"}, "last_night": {"games": [
        {"game_id": fl[0]["game_id"], "facts": fl[0], "text": {"headline": "a", "story": "b"}}]},
        "mystery": [1, 2, 3]}
    fake = Fake()
    assert writer.fill(brief, complete=fake) == brief and fake.prompts == []
    assert writer.fill({}, complete=fake) == {}


def test_fill_early_season_with_no_pulse_entities_leaves_pulse_null():
    brief = synthetic_brief()
    brief["team_pulse"].update({"rising": [], "sliding": [], "streaks": [], "power_movers_7d": [],
                                "power_top10": [], "early_season": True})
    brief["player_pulse"].update({"hot": [], "cold": [], "role_changes": []})
    fake = Fake(*[scripted_for_fill(synthetic_brief())[i] for i in (0, 2, 3)])
    out = writer.fill(brief, complete=fake)
    assert len(fake.prompts) == 3
    assert out["team_pulse"]["text"] is None
