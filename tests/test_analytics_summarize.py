from __future__ import annotations

import copy
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

from courtside.analytics.summarize import game_facts, notable_flags, unsung_score

FIX = Path(__file__).parent / "fixtures" / "analytics"
GIDS = ["0022500001", "0022500083", "0022500095", "0022500622", "0022501066"]
GOLDEN = FIX / "facts_0022501066.json"


@lru_cache(maxsize=None)
def load(gid):
    with gzip.open(FIX / f"boxscore_{gid}.json.gz", "rt") as f:
        box = json.load(f)
    with gzip.open(FIX / f"playbyplay_{gid}.json.gz", "rt") as f:
        pbp = json.load(f)
    return box, pbp


@lru_cache(maxsize=None)
def facts(gid):
    return game_facts(*load(gid))


@pytest.mark.parametrize("gid", GIDS)
def test_facts_are_json_serializable_compact_and_deterministic(gid):
    f = facts(gid)
    s = json.dumps(f, sort_keys=True, allow_nan=False)
    assert len(s) < 16_000
    box, pbp = load(gid)
    again = game_facts(copy.deepcopy(box), copy.deepcopy(pbp))
    assert json.dumps(again, sort_keys=True) == s
    assert json.loads(s) == f  # no tuples / non-JSON types survive


def test_facts_golden_snapshot():
    """Stable contract for the Brief writer. Regenerate deliberately with
    COURTSIDE_UPDATE_GOLDEN=1 uv run pytest tests/test_analytics_summarize.py"""
    f = facts("0022501066")
    if os.environ.get("COURTSIDE_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(f, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    assert GOLDEN.exists(), "golden facts missing; regenerate with COURTSIDE_UPDATE_GOLDEN=1"
    assert json.loads(GOLDEN.read_text()) == json.loads(json.dumps(f))


@pytest.mark.parametrize("gid", GIDS)
def test_core_facts_match_boxscore(gid):
    box, _ = load(gid)
    f = facts(gid)
    g = box["game"]
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        assert f[side]["team"] == g[key]["teamTricode"]
        assert f[side]["score"] == g[key]["score"]
        assert sum(f[side]["periods"]) == g[key]["score"]
        assert f["largest_lead"][g[key]["teamTricode"]]["lead"] == g[key]["statistics"]["biggestLead"]
    assert f["lead_changes"] == g["homeTeam"]["statistics"]["leadChanges"]
    assert f["times_tied"] == g["homeTeam"]["statistics"]["timesTied"]
    assert f["margin"] == abs(g["homeTeam"]["score"] - g["awayTeam"]["score"])
    assert f["overtimes"] == len(g["homeTeam"]["periods"]) - 4
    assert f["validation"]["ok"] is True
    assert f["validation"]["plus_minus_mismatches"] == 0
    # foul-outs agree with boxscore personal fouls
    six = {p["personId"] for k in ("homeTeam", "awayTeam") for p in g[k]["players"]
           if p["statistics"]["foulsPersonal"] >= 6}
    assert {x["id"] for x in f["incidents"]["foul_outs"]} == six


def test_specific_game_facts():
    f = facts("0022501066")
    assert f["winner"] == "LAC" and f["margin"] == 1
    assert f["comeback"] == {"team": "LAC", "deficit": 24,
                             "at": {"label": "Q2", "clock": "9:55", "score": {"IND": 45, "LAC": 21}}}
    # win-probability turning point: LAC's last 31 seconds (WP 8% -> win); the old run-based
    # placeholder picked the Q4 13-4 run (8:54-6:13) that levelled the game (still tested in test_analytics_story)
    assert f["turning_point"]["team"] == "LAC" and f["turning_point"]["method"] == "win_probability"
    assert f["turning_point"]["run"] == "4-0" and f["turning_point"]["reason"] == "flipped_lead"
    decisive = [k for k in f["clutch"]["key_plays"] if k.get("decisive")]
    assert decisive and decisive[0]["player"] == "Kawhi Leonard"
    assert any(n["name"] == "Darius Garland" and "30_point_game" in n["flags"] for n in f["notable_lines"])

    blow = facts("0022500622")
    assert blow["garbage_time"]["seconds"] == 720.0
    assert blow["garbage_time"]["from"] == {"label": "Q4", "clock": "12:00"}
    assert blow["clutch"] == {"seconds": 0.0, "had_clutch": False}
    assert all(r.get("garbage_time") for r in blow["runs"] if r["to"]["label"] == "Q4")

    ot2 = facts("0022500001")
    assert ot2["overtimes"] == 2
    assert len(ot2["clutch"]["key_plays"]) <= 8


@pytest.mark.parametrize("gid", GIDS)
def test_unsung_hero_rule(gid):
    f = facts(gid)
    winner = f["winner"]
    cands = f["unsung_hero"]
    assert cands, gid
    # winners first, then by score
    flags = [c["winner"] for c in cands]
    assert flags == sorted(flags, reverse=True)
    for c in cands:
        assert c["score"] >= 2.0
        assert c["minutes"] >= 12.0
        pts = int(c["line"].split(" ")[0])
        assert pts <= 20
        assert c["team"] == winner or not c["winner"]
    # not one of his team's top-2 scorers
    box, _ = load(gid)
    for c in cands:
        team_key = "homeTeam" if box["game"]["homeTeam"]["teamTricode"] == c["team"] else "awayTeam"
        pts = sorted(((p["statistics"]["points"], -p["personId"]) for p in box["game"][team_key]["players"]), reverse=True)
        top2 = {-pid for _, pid in pts[:2]}
        assert c["id"] not in top2


def _player(**kw):
    base = dict(pts=0, reb=0, oreb=0, ast=0, stl=0, blk=0, tov=0, fgm=0, fga=0, fg3m=0, fg3a=0,
                ftm=0, fta=0, stocks=0, ts_pct=None, starter=True, plus_minus_non_garbage=0,
                on_off=None, on_poss=0.0)
    base.update(kw)
    base["stocks"] = base["stl"] + base["blk"]
    return base


def test_notable_flag_thresholds():
    assert notable_flags(_player(pts=30, fga=20, fgm=11)) == ["30_point_game"]
    assert notable_flags(_player(pts=29, fga=20, fgm=11)) == []
    assert "50_point_game" in notable_flags(_player(pts=51, fga=30, fgm=18))
    td = notable_flags(_player(pts=12, reb=10, ast=10, fga=10, fgm=5))
    assert "triple_double" in td and "20_10" not in td
    assert "20_10" in notable_flags(_player(pts=20, reb=10, fga=15, fgm=8))
    assert "5_threes" in notable_flags(_player(pts=15, fg3m=5, fg3a=8, fga=8, fgm=5))
    assert "5_stocks" in notable_flags(_player(stl=2, blk=3, fga=2))
    assert "perfect_fg" in notable_flags(_player(pts=12, fga=5, fgm=5))
    assert "perfect_fg" not in notable_flags(_player(pts=8, fga=4, fgm=4))
    assert "cold_shooting" in notable_flags(_player(pts=10, fga=15, fgm=4))
    assert "bench_scoring" in notable_flags(_player(pts=20, fga=14, fgm=8, starter=False))


def test_unsung_score_formula():
    p = _player(pts=8, fga=6, fta=2, stl=2, blk=1, ast=4, oreb=2, reb=6, tov=1,
                plus_minus_non_garbage=12, on_off=20.0, on_poss=40.0, starter=False)
    hustle = 2 + 0.7 + 0.7 * 4 + 0.7 * 2 + 0.3 * 4 - 1
    over_par = 8 - 2 * 0.57 * (6 + 0.44 * 2)
    expect = 0.5 * 12 + 0.1 * 20.0 * 40 / 80 + hustle + 0.5 * over_par + 1.0
    assert unsung_score(p) == round(expect, 2)


def test_ejections_detected():
    box, pbp = load("0022500083")
    pbp = copy.deepcopy(pbp)
    acts = pbp["game"]["actions"]
    home = box["game"]["homeTeam"]
    pid = int(home["players"][0]["personId"])
    other = int(home["players"][1]["personId"])
    template = next(a for a in acts if a["actionType"] == "timeout")
    for k, (atype, sub) in enumerate((("foul", "technical"), ("foul", "technical"), ("ejection", "other"))):
        a = copy.deepcopy(template)
        a.update(actionType=atype, subType=sub, personId=pid if k < 2 else other,
                 teamId=home["teamId"], teamTricode=home["teamTricode"], descriptor=None,
                 description=f"synthetic {atype}", orderNumber=template["orderNumber"] + 1 + k,
                 actionNumber=90000 + k)
        acts.append(a)
    f = game_facts(box, pbp)
    ej = {(e["id"], e["reason"]) for e in f["incidents"]["ejections"]}
    assert (pid, "two technical fouls") in ej
    assert (other, "other") in ej
    assert f["validation"]["ok"]   # bench technicals / ejections are not on-floor evidence
