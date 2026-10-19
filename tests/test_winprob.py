"""Win-probability model: sanity cases, series properties, turning point, facts keys. No network."""

from __future__ import annotations

import copy
import gzip
import json
from functools import lru_cache
from pathlib import Path

import pytest

from courtside.analytics import game_story as gs
from courtside.analytics import winprob as wp
from courtside.analytics.pbp import normalize
from courtside.analytics.summarize import game_facts
from test_analytics_story import synth

FIX = Path(__file__).parent / "fixtures" / "analytics"
GIDS = ["0022500001", "0022500083", "0022500095", "0022500622", "0022501066"]


@lru_cache(maxsize=None)
def load(gid):
    with gzip.open(FIX / f"boxscore_{gid}.json.gz", "rt") as f:
        box = json.load(f)
    with gzip.open(FIX / f"playbyplay_{gid}.json.gz", "rt") as f:
        pbp = json.load(f)
    return box, pbp, normalize(pbp, box)


def P(**kw):
    kw.setdefault("exp", 0.0)
    kw.setdefault("poss", 0)
    kw.setdefault("period", 4)
    return wp.win_prob(**kw)


# ------------------------------------------------------------------------- the artifact

def test_coefficient_file_matches_feature_set():
    c = wp.load_coef()
    assert tuple(c["features"]) == wp.FEATURES and len(c["coef"]) == len(wp.FEATURES)
    assert c["test"]["season"] == "2025-26" and c["test"]["brier"] < 0.18
    assert wp.feature_matrix([3], [5], [0], [1], [4]).shape == (1, len(wp.FEATURES))


def test_scalar_and_vector_features_agree():
    import numpy as np
    for args in [(3, 5.0, 2.5, 1, 4), (-12, 1500.0, -6.0, -1, 2), (0, 300.0, 0.0, 0, 5)]:
        v = wp.feature_matrix(*[[a] for a in args])[0]
        assert np.allclose(v, wp.features(*args))


# ------------------------------------------------------------------------- end-of-game sanity

def test_tipoff_and_overtime_start_are_a_coin_flip():
    assert P(margin=0, t=2880, period=1) == pytest.approx(0.5, abs=0.01)
    assert P(margin=0, t=300, period=5) == pytest.approx(0.5, abs=0.03)
    assert P(margin=0, t=720, period=4) == pytest.approx(0.5, abs=0.01)


def test_up_three_with_five_seconds_and_the_ball_is_a_lock():
    assert P(margin=3, t=5, poss=1) >= 0.95
    assert P(margin=-3, t=5, poss=-1) <= 0.05
    assert P(margin=3, t=5, poss=-1) >= 0.90          # even with the opponent holding the ball
    assert P(margin=-3, t=5, poss=1) <= 0.10          # down three, the ball, 5 s: still a long shot


def test_pregame_spread_matters_early_and_vanishes_late():
    early = P(margin=0, t=2880, period=1, exp=7.0)
    assert 0.70 < early < 0.90
    assert early > P(margin=0, t=2880, period=1, exp=3.0) > 0.5
    assert abs(P(margin=2, t=10, exp=7.0, poss=1) - P(margin=2, t=10, exp=-7.0, poss=1)) < 0.03


@pytest.mark.parametrize("kw", [
    dict(t=2000, period=2), dict(t=600, period=4), dict(t=90, period=4, poss=1),
    dict(t=20, period=4, poss=-1, exp=3.0), dict(t=200, period=5),
])
def test_monotone_in_margin_and_symmetric(kw):
    ps = [P(margin=m, **kw) for m in range(-20, 21)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))
    other = dict(kw, poss=-kw.get("poss", 0), exp=-kw.get("exp", 0.0))
    for m in (1, 5, 12):
        assert P(margin=m, **kw) + P(margin=-m, **other) == pytest.approx(1.0, abs=1e-9)


def test_a_lead_is_worth_more_as_time_runs_out():
    ps = [P(margin=5, t=t, period=4) for t in (720, 400, 200, 90, 30, 10)]
    assert all(b > a for a, b in zip(ps, ps[1:])) and ps[-1] > 0.95
    assert P(margin=10, t=1440, period=3) < P(margin=10, t=300) < P(margin=10, t=60)


def test_possession_is_worth_more_in_close_late_games():
    gain_late = P(margin=0, t=15, poss=1) - P(margin=0, t=15, poss=-1)
    gain_early = P(margin=0, t=2000, period=2, poss=1) - P(margin=0, t=2000, period=2, poss=-1)
    assert gain_late > 0.12 and 0 < gain_early < 0.06 and gain_late > 3 * gain_early


def test_sanity_table_is_published():
    s = wp.sanity_cases(wp.load_coef())
    assert s["up3_5s_home_ball"] >= 0.95 and abs(s["tipoff_tied_spread0"] - 0.5) < 0.01


# ------------------------------------------------------------------------- series

@pytest.mark.parametrize("gid", GIDS)
def test_series_shape_and_end_state(gid):
    box, _, events = load(gid)
    pts = wp.series(events, pregame_spread=-3.0)
    assert len(pts) == len(events)
    assert [p["idx"] for p in pts] == [e.idx for e in events]
    assert all(0.0 <= p["home_wp"] <= 1.0 for p in pts)
    assert all(b["elapsed"] >= a["elapsed"] for a, b in zip(pts, pts[1:]))
    # tip-off: a mild home favourite
    assert 0.5 < pts[0]["home_wp"] < 0.7
    # the buzzer: the game is over, so the series ends at exactly 0 or 1 for the real winner
    h, a = events[-1].score_home, events[-1].score_away
    assert pts[-1]["home_wp"] == (1.0 if h > a else 0.0)
    # margin in the series is the feed's running score difference
    assert all(p["margin"] == e.margin_home for p, e in zip(pts, events))


def test_series_follows_the_score_in_a_synthetic_game():
    events = synth([(1, 600, "home", 2), (1, 500, "home", 3), (2, 600, "away", 3),
                    (3, 600, "home", 2), (4, 120, "away", 3), (4, 60, "away", 3), (4, 5, "away", 2)])
    pts = wp.series(events, None)
    first = pts[0]["home_wp"]
    assert first == pytest.approx(0.5, abs=0.01)
    scoring = [p for p, e in zip(pts, events) if e.points]
    assert scoring[0]["home_wp"] > first and scoring[1]["home_wp"] > scoring[0]["home_wp"]
    assert scoring[2]["home_wp"] < scoring[1]["home_wp"]       # away scores
    h, a = events[-1].score_home, events[-1].score_away
    assert (h, a) == (7, 11) and pts[-1]["home_wp"] == 0.0         # away won; the game is over


def test_timeline_downsample():
    _, _, events = load("0022501066")
    pts = wp.series(events, 0.0)
    tl = wp.timeline(pts, 100)
    assert 60 <= len(tl) <= 100
    assert tl[0]["elapsed"] == 0.0 and tl[-1]["elapsed"] == pts[-1]["elapsed"]
    assert all(b["elapsed"] > a["elapsed"] for a, b in zip(tl, tl[1:]))
    assert set(tl[0]) == {"elapsed", "margin", "home_wp"}


# ------------------------------------------------------------------------- turning point

def test_turning_point_prefers_the_decisive_swing_over_the_biggest_run():
    """The final 37 seconds (+0.37 for the winner) beat the slow 3-quarter climb (+0.07 a step)."""
    plays = [(1, 700, "home", 2), (1, 600, "away", 3), (2, 600, "away", 3), (3, 600, "away", 3),
             (4, 100, "away", 2), (4, 40, "home", 3), (4, 30, "away", 2), (4, 3, "away", 3)]
    events = synth(plays)
    scoring = [e.idx for e in events if e.points]
    wpv = [(scoring[0], 0.60), (scoring[1], 0.52), (scoring[2], 0.45), (scoring[3], 0.40),
           (scoring[4], 0.33), (scoring[5], 0.37), (scoring[6], 0.10), (scoring[7], 0.0)]
    tp = gs.turning_point(events, win_prob=wpv)
    assert tp["method"] == "win_probability" and tp["side"] == "away"
    assert tp["wp_swing"] == pytest.approx(0.37)
    assert (tp["wp_before"], tp["wp_after"]) == (0.63, 1.0)
    assert tp["key_play"]["wp_swing"] == pytest.approx(0.27)        # the 0:30 basket
    assert tp["key_play"]["clock"] == "0:30"
    assert tp["at"]["clock"] == "0:03" and tp["end_idx"] == scoring[7]
    assert tp["run"]["label"] == "5-0"


def test_turning_point_stretch_spans_a_burst_within_the_window():
    # three quick away baskets with the WP rising each time: the stretch is the burst after the
    # first one (0.45 total), not any single step (0.30 at most)
    plays = [(1, 700, "home", 2), (3, 500, "away", 3), (3, 470, "away", 3), (3, 440, "away", 3),
             (3, 100, "home", 2), (4, 400, "away", 3)]
    events = synth(plays)
    s = [e.idx for e in events if e.points]
    wpv = [(s[0], 0.60), (s[1], 0.55), (s[2], 0.40), (s[3], 0.10), (s[4], 0.12), (s[5], 0.0)]
    tp = gs.turning_point(events, win_prob=wpv)
    assert tp["wp_swing"] == pytest.approx(0.45)
    assert tp["run"]["label"] == "6-0" and tp["run"]["unanswered"]
    assert tp["reason"] in ("flipped_lead", "created_winning_margin")
    # stretches are capped: with a 20 s window only adjacent points qualify
    best = gs.wp_stretch(events, wpv, "away", window=20.0)
    assert best[0] == pytest.approx(0.30) and best[2] - best[1] == 1


def test_real_game_turning_point_is_the_winners_biggest_stretch():
    box, pbp, events = load("0022501066")        # LAC came back from 24 down to win 114-113
    pts = wp.series(events, 0.0)
    tp = gs.turning_point(events, win_prob=wp.wp_pairs(pts))
    assert tp["method"] == "win_probability"
    assert tp["side"] == gs.winner(events) == "away"
    assert tp["wp_swing"] >= 0.15 and 0 < tp["wp_before"] < tp["wp_after"] <= 1
    # no other stretch of at most WP_WINDOW seconds does better
    pw = [(p["elapsed"], 1.0 - p["home_wp"]) for p in pts]
    for i, (ti, pi) in enumerate(pw):
        for tj, pj in pw[i + 1:]:
            if tj - ti > gs.WP_WINDOW:
                break
            assert pj - pi <= tp["wp_swing"] + 1e-3 + 1e-9 or tj == pw[i + 1][0]


# ------------------------------------------------------------------------- facts

WP_KEYS = {"wp_model", "wp_timeline", "biggest_wp_plays", "excitement", "comeback_wp"}


@pytest.mark.parametrize("gid", GIDS)
def test_game_facts_has_wp_keys_and_serialises(gid):
    box, pbp, _ = load(gid)
    f = game_facts(box, pbp, pregame_spread=-4.5)
    assert WP_KEYS <= set(f)
    assert json.loads(json.dumps(f, allow_nan=False)) == f
    assert f["wp_model"]["pregame_spread"] == -4.5
    assert 60 <= len(f["wp_timeline"]) <= 100
    assert all(set(r) == {"elapsed", "margin", "home_wp"} for r in f["wp_timeline"])
    plays = f["biggest_wp_plays"]
    assert 1 <= len(plays) <= 3
    assert [p["wp_swing"] for p in plays] == sorted((p["wp_swing"] for p in plays), reverse=True)
    for p in plays:
        assert p["description"] and p["toward"] in (f["home"]["team"], f["away"]["team"])
        assert {"label", "clock", "score", "player"} <= set(p)
    assert f["excitement"] >= plays[0]["wp_swing"]
    cb = f["comeback_wp"]
    assert cb["team"] == f["winner"] and 0.0 <= cb["min_wp"] <= 1.0
    tp = f["turning_point"]
    assert tp["method"] == "win_probability" and tp["wp_swing"] > 0 and tp["run"] and tp["reason"]


def test_game_facts_keeps_its_previous_keys_and_is_deterministic():
    box, pbp, _ = load("0022501066")
    f = game_facts(box, pbp)
    old = {"schema", "game_id", "date", "arena", "home", "away", "winner", "margin", "overtimes",
           "top_performers", "unsung_hero", "runs", "turning_point", "clutch", "lead_changes",
           "times_tied", "largest_lead", "comeback", "garbage_time", "notable_lines", "incidents",
           "validation"}
    assert old <= set(f)
    assert game_facts(copy.deepcopy(box), copy.deepcopy(pbp)) == f


def test_comeback_game_wp_story():
    box, pbp, _ = load("0022501066")
    f = game_facts(box, pbp)
    assert f["winner"] == "LAC"
    assert f["comeback_wp"]["min_wp"] < 0.10       # LAC were nearly dead at the bottom
    assert f["excitement"] > 2.0


def test_explicit_win_prob_overrides_the_turning_point_series():
    box, pbp, events = load("0022500083")
    n = len(events)
    flat = [(0, 0.5), (n // 2, 0.5), (n - 1, 0.0 if events[-1].score_home < events[-1].score_away else 1.0)]
    f = game_facts(box, pbp, win_prob=flat)
    assert f["turning_point"]["wp_swing"] == pytest.approx(0.5)
    assert f["turning_point"]["at"]["clock"] is not None
