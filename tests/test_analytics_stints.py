"""Lineup reconstruction: the correctness gate.

For every fixture game, every player's reconstructed on-court seconds must equal the boxscore
minutes within 1 s and the on-court plus-minus must equal the boxscore plusMinusPoints exactly.
"""

from __future__ import annotations

import copy
import gzip
import json
from functools import lru_cache
from pathlib import Path

import pytest

from courtside.analytics.pbp import game_seconds, normalize, period_end, period_start
from courtside.analytics.stints import (
    box_players, box_starters, build_rotation, credit_index, lineup_at, on_court_plus_minus,
    on_court_seconds, validate_stints,
)

FIX = Path(__file__).parent / "fixtures" / "analytics"
GAMES = {
    "0022500001": "double overtime, 1-point finish",
    "0022500095": "single overtime",
    "0022500083": "1-point finish",
    "0022500622": "54-point blowout with garbage time",
    "0022501066": "24-point comeback, 1-point finish",
}


@lru_cache(maxsize=None)
def load(gid: str) -> tuple[dict, dict]:
    with gzip.open(FIX / f"boxscore_{gid}.json.gz", "rt") as f:
        box = json.load(f)
    with gzip.open(FIX / f"playbyplay_{gid}.json.gz", "rt") as f:
        pbp = json.load(f)
    return box, pbp


def rotation(gid: str):
    box, pbp = load(gid)
    events = normalize(pbp, box)
    return box, events, build_rotation(events, box)


def test_fixtures_are_complete_untrimmed_and_small():
    total = sum(p.stat().st_size for p in FIX.glob("*.json.gz"))
    assert total < 3_000_000
    for gid in GAMES:
        box, pbp = load(gid)
        assert box["game"]["gameStatus"] == 3
        actions = pbp["game"]["actions"]
        assert actions[-1]["actionType"] == "game"
        # untrimmed: the PBP accounts for the whole final score
        last = actions[-1]
        assert int(last["scoreHome"]) == box["game"]["homeTeam"]["score"]
        assert int(last["scoreAway"]) == box["game"]["awayTeam"]["score"]


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_minutes_and_plus_minus_reconcile_with_boxscore(gid):
    box, events, rot = rotation(gid)
    report = validate_stints(box, rot.stints, rot.anomalies)
    bad_min = [(c.name, c.box_seconds, c.seconds) for c in report.minutes_failures]
    bad_pm = [(c.name, c.box_plus_minus, c.plus_minus) for c in report.plus_minus_failures]
    assert not bad_min, f"{gid} minutes mismatch: {bad_min}"
    assert not bad_pm, f"{gid} plus-minus mismatch: {bad_pm}"
    assert report.max_seconds_diff <= 1.0
    assert rot.anomalies == []
    # every player who played is covered, and nobody unknown appears
    played = {pid for pid, p in box_players(box).items() if p["played"]}
    assert {c.player_id for c in report.checks} == played


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_stints_are_five_on_five_and_tile_the_game(gid):
    box, events, rot = rotation(gid)
    stints = rot.stints
    for s in stints:
        assert len(s.home) == 5 and len(s.away) == 5, s
        assert not set(s.home) & set(s.away)
        assert s.end >= s.start
    assert stints[0].start == 0.0
    assert stints[-1].end == game_seconds(events)
    for a, b in zip(stints, stints[1:]):
        assert b.start == a.end  # contiguous in time
        assert b.first_idx >= a.last_idx
    # team points and total seconds
    assert sum(s.home_pts for s in stints) == box["game"]["homeTeam"]["score"]
    assert sum(s.away_pts for s in stints) == box["game"]["awayTeam"]["score"]
    secs = on_court_seconds(stints)
    assert sum(secs.values()) == pytest.approx(10 * game_seconds(events), abs=0.5)


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_period_one_starters_are_box_starters(gid):
    box, events, rot = rotation(gid)
    bs = box_starters(box)
    assert rot.starters[(1, "home")] == bs["home"]
    assert rot.starters[(1, "away")] == bs["away"]


@pytest.mark.parametrize("gid", sorted(GAMES))
def test_possession_counts_are_balanced_and_near_estimate(gid):
    box, events, rot = rotation(gid)
    home = sum(s.home_poss for s in rot.stints)
    away = sum(s.away_poss for s in rot.stints)
    assert abs(home - away) <= 3
    for key, n in (("homeTeam", home), ("awayTeam", away)):
        st = box["game"][key]["statistics"]
        est = (st["fieldGoalsAttempted"] - st["reboundsOffensive"] + st["turnoversTotal"]
               + 0.44 * st["freeThrowsAttempted"])
        assert abs(n - est) / est < 0.08, (key, n, est)


def test_free_throws_are_credited_to_the_lineup_at_the_foul():
    # NYK-BKN: Brunson's 2nd FT (Q2 0:02.8) came after Anunoby was subbed out; the boxscore
    # credits Anunoby (+28), not Shamet.
    box, events, rot = rotation("0022500622")
    credit = credit_index(events)
    ft = next(e for e in events if e.is_ft and e.period == 2 and e.clock == 2.8 and e.ft_num == 2)
    foul = events[credit[ft.idx]]
    assert foul.is_foul and foul.clock == ft.clock
    anunoby = next(pid for pid, p in box_players(box).items() if p["name"] == "OG Anunoby")
    assert anunoby in lineup_at(rot.stints, foul.idx).home
    assert anunoby not in lineup_at(rot.stints, ft.idx).home
    assert on_court_plus_minus(rot.stints)[anunoby] == 28


def _strip_player_evidence(pbp: dict, pid: int, period: int) -> dict:
    """Remove every trace of ``pid`` from non-substitution actions of ``period`` (scores intact)."""
    out = copy.deepcopy(pbp)
    for a in out["game"]["actions"]:
        if a.get("period") != period or a.get("actionType") == "substitution":
            continue
        if a.get("personId") == pid:
            a["personId"] = 0
        for k in ("assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId"):
            if a.get(k) == pid:
                a.pop(k)
    return out


@pytest.mark.parametrize("gid", ["0022500001", "0022501066"])
def test_silent_full_period_player_is_recovered_from_boxscore(gid):
    """A player who plays an entire period without one recorded action never shows up in that
    period's events. The reconstruction must still find him (via the boxscore residual)."""
    box, pbp = load(gid)
    events = normalize(pbp, box)
    rot = build_rotation(events, box)
    target = None
    for p in range(2, max(e.period for e in events) + 1):
        ps = [s for s in rot.stints if s.period == p]
        for side in ("home", "away"):
            full = set(ps[0].lineup(side)).intersection(*(s.lineup(side) for s in ps))
            if full:
                target = (p, side, min(full))
                break
        if target:
            break
    assert target, "fixture has no full-period player"
    p, side, pid = target
    pbp2 = _strip_player_evidence(pbp, pid, p)
    ev2 = normalize(pbp2, box)
    assert not any(pid in (e.player_id, e.assist_id, e.steal_id, e.block_id, e.foul_drawn_id)
                   for e in ev2 if e.period == p)
    rot2 = build_rotation(ev2, box)
    assert pid in rot2.starters[(p, side)]
    report = validate_stints(box, rot2.stints, rot2.anomalies)
    assert report.ok, (report.minutes_failures, report.plus_minus_failures)


def test_ot_periods_are_five_minutes():
    assert period_start(5) == 2880.0 and period_end(5) == 3180.0
    assert period_end(6) == 3480.0
    box, events, rot = rotation("0022500001")
    assert game_seconds(events) == 3480.0
    assert {s.period for s in rot.stints} == {1, 2, 3, 4, 5, 6}
