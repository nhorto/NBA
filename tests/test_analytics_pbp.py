from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import pytest

from courtside.analytics.pbp import (
    elapsed_at, fmt_clock, normalize, parse_clock, period_label, period_start,
)

FIX = Path(__file__).parent / "fixtures" / "analytics"
GIDS = ["0022500001", "0022500083", "0022500095", "0022500622", "0022501066"]


@lru_cache(maxsize=None)
def load(gid):
    with gzip.open(FIX / f"boxscore_{gid}.json.gz", "rt") as f:
        box = json.load(f)
    with gzip.open(FIX / f"playbyplay_{gid}.json.gz", "rt") as f:
        pbp = json.load(f)
    return box, pbp


def test_clock_parsing_and_elapsed():
    assert parse_clock("PT12M00.00S") == 720.0
    assert parse_clock("PT11M36.00S") == 696.0
    assert parse_clock("PT00M05.30S") == 5.3
    assert parse_clock("PT45M15.10S") == 2715.1
    assert parse_clock("") == 0.0 and parse_clock(None) == 0.0 and parse_clock("garbage") == 0.0
    assert elapsed_at(1, 720.0) == 0.0
    assert elapsed_at(4, 0.0) == 2880.0
    assert elapsed_at(5, 300.0) == 2880.0
    assert elapsed_at(6, 60.0) == 3420.0
    assert period_start(7) == 3480.0
    assert fmt_clock(696.0) == "11:36" and fmt_clock(5.3) == "0:05.3" and fmt_clock(0.0) == "0:00"
    assert [period_label(p) for p in (1, 4, 5, 6)] == ["Q1", "Q4", "OT", "2OT"]


@pytest.mark.parametrize("gid", GIDS)
def test_event_counts_match_boxscore_team_totals(gid):
    box, pbp = load(gid)
    events = normalize(pbp, box)
    for key, side in (("homeTeam", "home"), ("awayTeam", "away")):
        st = box["game"][key]["statistics"]
        mine = [e for e in events if e.side == side]
        assert sum(e.is_fga for e in mine) == st["fieldGoalsAttempted"]
        assert sum(bool(e.is_fga and e.made) for e in mine) == st["fieldGoalsMade"]
        assert sum(e.is_three for e in mine) == st["threePointersAttempted"]
        assert sum(e.is_ft for e in mine) == st["freeThrowsAttempted"]
        assert sum(bool(e.is_ft and e.made) for e in mine) == st["freeThrowsMade"]
        assert sum(e.is_turnover for e in mine) == st["turnoversTotal"]
        assert sum(e.is_steal for e in mine) == st["steals"]
        assert sum(e.is_block for e in mine) == st["blocks"]
        assert sum(bool(e.is_fga and e.made and e.assist_id) for e in mine) == st["assists"]
        assert sum(e.points for e in mine) == st["points"]


@pytest.mark.parametrize("gid", GIDS)
def test_scores_are_consistent_with_points(gid):
    box, pbp = load(gid)
    events = normalize(pbp, box)
    h = a = 0
    for e in events:
        if e.side == "home":
            h += e.points
        elif e.side == "away":
            a += e.points
        assert (e.score_home, e.score_away) == (h, a), e.description
    orders = [e.order for e in events]
    assert orders == sorted(orders)
    assert [e.idx for e in events] == list(range(len(events)))


def test_event_fields():
    box, pbp = load("0022500001")
    events = normalize(pbp, box)
    three = next(e for e in events if e.is_three and e.made)
    assert three.points == 3 and three.x is not None and three.distance and three.distance > 22
    assert three.ends_possession
    sub = next(e for e in events if e.is_sub)
    assert sub.sub_dir in ("in", "out") and sub.player_id
    dreb = next(e for e in events if e.is_rebound and e.rebound_offensive is False)
    assert dreb.ends_possession and dreb.poss_side != dreb.side
    oreb = next(e for e in events if e.is_rebound and e.rebound_offensive)
    assert not oreb.ends_possession
    tov = next(e for e in events if e.is_turnover)
    assert tov.ends_possession and tov.poss_side == tov.side
    # OT event elapsed
    ot = next(e for e in events if e.period == 5)
    assert 2880.0 <= ot.elapsed <= 3180.0
    # to_dict is JSON-serializable
    json.dumps(three.to_dict())


def test_and_one_ends_possession_on_the_free_throw():
    box, pbp = load("0022500001")
    events = normalize(pbp, box)
    found = 0
    for i, e in enumerate(events):
        if not (e.is_ft and e.ft_of == 1 and not e.ft_retains_possession):
            continue
        fg = next((x for x in reversed(events[:i]) if x.is_fga and x.made and x.side == e.side
                   and x.clock == e.clock and x.period == e.period), None)
        if fg is None:
            continue
        found += 1
        assert not fg.ends_possession
        assert e.ends_possession == bool(e.made)
    assert found >= 1


def test_technical_free_throw_does_not_end_possession():
    box, pbp = load("0022500095")
    events = normalize(pbp, box)
    techs = [e for e in events if e.is_ft and e.ft_retains_possession]
    assert techs
    assert not any(e.ends_possession for e in techs)
