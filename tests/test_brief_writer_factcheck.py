"""brief/factcheck.py and brief/templates.py against the real fixture facts."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from courtside.brief import templates, writer
from courtside.brief.factcheck import FactIndex, check, count_sentences

FIX = Path(__file__).parent / "fixtures" / "brief_writer"
GAME_FILES = sorted(glob.glob(str(FIX / "game_facts_*.json")))


def game(gid="0022501066"):
    return json.loads((FIX / f"game_facts_{gid}.json").read_text())


def league():
    return json.loads((FIX / "league_facts_2026-02-15.json").read_text())


def msgs(text, facts, kind="game_story"):
    return check(text, facts, kind=kind).messages()


# ---- numbers -----------------------------------------------------------------------------

def test_true_statement_passes():
    t = ("Indiana led by 24 in the second quarter and still lost 114-113. A 13-4 Clippers run "
         "levelled it at 101 with 6:13 left, and Kawhi Leonard scored 28 points on 13-of-26 shooting.")
    assert msgs(t, game()) == []


def test_invented_number_is_caught():
    bad = msgs("Darius Garland scored 31 points and the Pacers shot 64.9% in the fourth.", game())
    assert any("'31'" in m for m in bad)
    assert any("64.9%" in m for m in bad)


def test_invented_run_score_and_lead_are_caught():
    f = game()
    assert any("14-2" in m for m in msgs("A 14-2 Clippers run changed everything.", f))
    assert any("115-113" in m for m in msgs("The Clippers won 115-113.", f))
    # 25 exists in the facts (a quarter score) but never as a lead: contextual check
    assert any("'25'" in m for m in msgs("Indiana led by 25 in the second quarter.", f))
    assert msgs("Indiana led by 24 in the second quarter.", f) == []


def test_score_in_either_order_and_margin_derivations():
    f = game()
    assert msgs("The Clippers won 114-113.", f) == []
    assert msgs("The Clippers won 113-114.", f) == []
    assert msgs("They won by 1 on a combined 227 points.", f) == []   # 114+113


def test_percent_and_rounding_tolerance():
    f = {"ts_pct": 0.649, "pts_avg": 43.5, "line": "30 pts (10-19 FG, 6-11 3P)"}
    assert msgs("He shot 64.9% true.", f) == []
    assert msgs("He shot 65% true.", f) == []
    assert msgs("He averaged 44 points.", f) == []
    assert msgs("He averaged 43.5 points.", f) == []
    assert msgs("He shot 64.5% true.", f) != []           # finer than the fact, and wrong
    assert msgs("He went 6-of-11 from deep and 10-for-19 overall.", f) == []
    assert msgs("He went 7-of-11 from deep.", f) != []


def test_times_and_ordinals():
    f = game()
    assert msgs("Level at 101 with 6:13 left.", f) == []
    assert any("6:14" in m for m in msgs("Level with 6:14 left.", f))
    assert msgs("It happened in the 4th quarter.", f) == []
    assert any("47th" in m for m in msgs("He was the league's 47th scorer.", f))


def test_spelled_numbers_are_checked_but_small_ones_ignored():
    f = game()
    assert any("twenty-five" in m.lower() or "twenty" in m.lower()
               for m in msgs("Indiana held a twenty-seven point edge.", f))
    assert msgs("It was a two-way game, and one of the best.", f) == []


def test_shot_type_numerals_and_embedded_digits_ignored():
    f = {"x": "no numbers here"}
    assert msgs("He hit a 3-pointer and a 2-point jumper.", f) == []


def test_pairs_from_line_shorthand():
    f = game()
    assert msgs("Obi Toppin had a 20-8 night.", f) == []   # 20 pts, 8 reb
    assert any("20-9" in m for m in msgs("Obi Toppin had a 20-9 night.", f))


# ---- names -------------------------------------------------------------------------------

def test_invented_name_is_caught():
    bad = msgs("Stephen Curry hit a late three for the Clippers.", game())
    assert any("Stephen" in m for m in bad) and any("Curry" in m for m in bad)
    assert any("Haliburton" in m for m in msgs("Indiana missed Haliburton badly.", game()))


def test_known_names_pass_accents_possessives_and_nicknames():
    f = league()
    t = "Nikola Jokic and the Lakers' Luka Doncic lead the way, while Shai Gilgeous-Alexander's streak continues."
    assert [m for m in msgs(t, f) if "name" in m] == []
    assert msgs("Kawhi Leonard's late basket sealed it for the Clippers over the Pacers.", game()) == []


def test_team_from_another_game_is_a_name_violation():
    assert any("Knicks" in m for m in msgs("It was nothing like what the Knicks did to Brooklyn.", game()))


def test_sentence_initial_common_words_are_not_names():
    assert msgs("Meanwhile, Kawhi Leonard finished it. Tonight he rests. Then the Clippers left.", game()) == []


def test_tricode_in_prose_is_only_a_warning():
    r = check("LAC won 114-113.", game())
    assert r.ok and any("tricode" in w for w in r.warnings)


# ---- style -------------------------------------------------------------------------------

def test_betting_language_is_a_violation_cliches_are_warnings():
    f = game()
    r = check("Take the Clippers; it's a lock.", f)
    assert any(v.kind == "betting" for v in r.violations)
    r = check("Kawhi Leonard took over when it mattered most!", f)
    assert r.ok
    assert any("took over" in w for w in r.warnings) and any("exclamation" in w for w in r.warnings)


def test_length_only_gross_overruns_are_violations():
    f = game()
    ok = "Kawhi Leonard won it for the Clippers. Indiana led by 24. The Pacers lost 114-113. Obi Toppin had 20 points."
    assert check(ok, f, kind="game_story").ok
    five = " ".join(["Kawhi Leonard scored."] * 6)
    assert any(v.kind == "length" for v in check(five, f, kind="game_story").violations)
    long_head = "Clippers erase a very large deficit against the Pacers in a late night game that went right down to the wire"
    assert any(v.kind == "length" for v in check(long_head, f, kind="game_headline").violations)
    slightly = "Clippers erase 24-point hole against the Pacers in a game on Sunday"      # 12 words: warning only
    r = check(slightly, f, kind="game_headline")
    assert r.ok and any("long" in w for w in r.warnings)


def test_sentence_counter_ignores_initials_and_decimals():
    assert count_sentences("P.J. Washington scored 20.5 points. Jr. was fine! Really?") == 3


def test_empty_text_fails():
    assert not check("", game()).ok and not check(None, game()).ok


def test_index_accepts_prebuilt_index():
    ix = FactIndex.build(game())
    assert check("Kawhi Leonard scored 28 points.", ix).ok


# ---- templates must themselves pass the checker -------------------------------------------

@pytest.mark.parametrize("path", GAME_FILES)
def test_game_templates_pass_factcheck_and_length(path):
    f = json.loads(Path(path).read_text())
    t = templates.game(f)
    assert msgs(t["headline"], f, "game_headline") == [], t["headline"]
    assert msgs(t["story"], f, "game_story") == [], t["story"]
    assert len(t["headline"].split()) <= 10
    assert 2 <= count_sentences(t["story"]) <= 4


def test_pulse_templates_pass_factcheck_for_every_entity():
    lf = league()
    method = writer._method_block(lf)
    teams, players = writer._team_entities(lf), writer._player_entities(lf)
    assert teams and players
    for e in teams:
        txt = templates.team_blurb(e)
        assert msgs(txt, {"entity": e, "method": method}, "blurb") == [], txt
    for e in players:
        txt = templates.player_blurb(e)
        assert msgs(txt, {"entity": e, "method": method}, "blurb") == [], txt
    head = templates.pulse_headline(teams, players)
    assert msgs(head, {"teams": teams, "players": players}, "pulse_headline") == [], head


def test_tonight_and_headline_templates_pass_factcheck():
    preds = [
        {"game_id": "1", "home": "BOS", "away": "MIA", "home_win_prob": 0.71, "home_margin": 7.2,
         "total": 228.5, "market_home_spread": -4.5, "reasons": ["Miami is on a back-to-back"]},
        {"game_id": "2", "home": "Denver Nuggets", "away": "Phoenix Suns", "home_win_prob": 38,
         "home_margin": -1.5, "reasons": ["Denver is without Nikola Jokic"]},
    ]
    for p in preds:
        n = writer.normalize_prediction(p)
        txt = templates.tonight(writer._tonight_template_input(n))
        assert msgs(txt, n, "tonight") == [], txt
        assert count_sentences(txt) <= 2
    summary = {"games": [{"game_id": "1", "headline": "Clippers erase 24-point deficit to beat Pacers",
                          "home": "IND", "away": "LAC", "margin": 1, "comeback_deficit": 24}],
               "pulse": {"headline": "Knicks surge while Bulls slide"}}
    h = templates.brief_headline(summary)
    assert msgs(h, summary, "brief_headline") == []


def test_outside_knowledge_words_need_support_in_the_facts():
    f = game()
    bad = msgs("Pacers forward Jarace Walker left early with an ankle injury.", f)
    assert any("forward" in m for m in bad) and any("ankle" in m for m in bad)
    assert any("injury" in m for m in bad)
    # words the facts themselves use are fine
    g = {"note": "Walker left the game and did not return", "x": "forward progress", "y": "injury report"}
    assert [m for m in msgs("Walker left; the injury report lists forward progress.", g) if "claim" in m] == []
    assert any("quotation" in m for m in msgs('Leonard called it "the biggest win of the year".', f))
    assert any("said" in m for m in msgs("Leonard said afterward that it felt great.", f))


def test_title_case_headline_is_rejected_but_names_still_checked_in_sentence_case():
    f = game()
    r = check("Clippers Erase 24-Point Deficit To Beat Pacers", f, kind="game_headline")
    assert any(v.kind == "format" and "sentence case" in v.detail for v in r.violations)
    assert check("Clippers erase 24-point deficit to beat Pacers", f, kind="game_headline").ok
    bad = check("Stephen Curry rescues Clippers", f, kind="game_headline")
    assert any(v.kind == "name" for v in bad.violations)
