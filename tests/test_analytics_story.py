from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import pytest

from courtside.analytics import game_story as gs
from courtside.analytics.pbp import normalize, period_end, period_start
from courtside.analytics.stints import Stint, box_players, build_stints

FIX = Path(__file__).parent / "fixtures" / "analytics"
GIDS = ["0022500001", "0022500083", "0022500095", "0022500622", "0022501066"]

HOME_ID, AWAY_ID = 1, 2
HOME_P = tuple(range(101, 106))
AWAY_P = tuple(range(201, 206))
HOME_BENCH = tuple(range(106, 111))
AWAY_BENCH = tuple(range(206, 211))


@lru_cache(maxsize=None)
def fixture(gid):
    with gzip.open(FIX / f"boxscore_{gid}.json.gz", "rt") as f:
        box = json.load(f)
    with gzip.open(FIX / f"playbyplay_{gid}.json.gz", "rt") as f:
        pbp = json.load(f)
    events = normalize(pbp, box)
    return box, pbp, events, build_stints(events, box)


# ------------------------------------------------------------------------- synthetic games

def synth(plays, start=(0, 0)):
    """plays: (period, clock_seconds, 'home'|'away', points) -> normalized events.

    Period start/end markers are added for every period touched (and all earlier ones)."""
    actions = []
    h, a = start
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        clock = kw.pop("clock")
        m, s = divmod(clock, 60)
        actions.append({"actionNumber": n, "orderNumber": n * 10, "clock": f"PT{int(m):02d}M{s:05.2f}S",
                        "scoreHome": str(h), "scoreAway": str(a), "qualifiers": [], "personId": 0, **kw})

    last_p = max(p for p, *_ in plays)
    by_p = {}
    for pl in plays:
        by_p.setdefault(pl[0], []).append(pl)
    for p in range(1, last_p + 1):
        plen = 720.0 if p <= 4 else 300.0
        add(period=p, clock=plen, actionType="period", subType="start")
        for _, clock, side, pts in by_p.get(p, []):
            tid = HOME_ID if side == "home" else AWAY_ID
            pid = HOME_P[0] if side == "home" else AWAY_P[0]
            if side == "home":
                h += pts
            else:
                a += pts
            kind = {1: "freethrow", 2: "2pt", 3: "3pt"}[pts]
            add(period=p, clock=clock, teamId=tid, personId=pid, actionType=kind,
                subType="1 of 1" if pts == 1 else "Jump Shot", shotResult="Made",
                description=f"{side} {pts}")
        add(period=p, clock=0.0, actionType="period", subType="end")
    box = {"game": {"homeTeam": {"teamId": HOME_ID, "teamTricode": "HOM"},
                    "awayTeam": {"teamId": AWAY_ID, "teamTricode": "AWY"}}}
    return normalize({"game": {"actions": actions}}, box)


def stints_for(events, lineups=None):
    """Stints covering all events. Default: starters for every period.

    lineups: {period: [(clock_from, home_ids, away_ids), ...]}, clock_from descending; a segment
    runs from its clock_from to the next segment's clock_from."""
    out = []
    for p in sorted({e.period for e in events}):
        pev = [e for e in events if e.period == p]
        plen = period_end(p) - period_start(p)
        segs = (lineups or {}).get(p) or [(plen, HOME_P, AWAY_P)]
        members = [[] for _ in segs]
        for e in pev:
            k = max(i for i, sg in enumerate(segs) if e.clock <= sg[0] or i == 0)
            members[k].append(e.idx)
        nxt = pev[0].idx
        for k, (clock_from, hp, ap) in enumerate(segs):
            clock_to = segs[k + 1][0] if k + 1 < len(segs) else 0.0
            first = members[k][0] if members[k] else nxt
            last = members[k][-1] + 1 if members[k] else first
            nxt = last
            out.append(Stint(p, period_end(p) - clock_from, period_end(p) - clock_to, first, last,
                             tuple(sorted(hp)), tuple(sorted(ap)), 0, 0, 0, 0))
    return out


# ------------------------------------------------------------------------- runs

def test_unanswered_and_stretch_runs_on_synthetic_sequence():
    plays = [
        (1, 700, "home", 2), (1, 680, "away", 2),
        # home 10-0 unanswered
        (1, 650, "home", 3), (1, 620, "home", 2), (1, 600, "home", 2), (1, 560, "home", 3),
        (1, 540, "away", 2),                          # answered
        (1, 520, "home", 2), (1, 500, "home", 2),     # -> 14-2 stretch from 650 to 500
        (1, 300, "away", 2), (1, 280, "home", 2), (1, 250, "away", 3),
    ]
    events = synth(plays)
    un = gs.unanswered_runs(events, min_run=8)
    assert [(r.side, r.label) for r in un] == [("home", "10-0")]
    r = un[0]
    assert r.score_before == (2, 2) and r.score_after == (12, 2)
    assert r.minutes == round((650 - 560) / 60, 1)
    st = gs.stretch_runs(events, min_run=8)
    assert [(s.side, s.label, s.unanswered) for s in st] == [("home", "14-2", False)]
    assert st[0].score_before == (2, 2) and st[0].score_after == (16, 4)
    assert st[0].minutes == 2.5
    allr = gs.scoring_runs(events, min_run=8)
    assert {x.label for x in allr} == {"10-0", "14-2"}
    # threshold
    assert gs.unanswered_runs(events, min_run=11) == []


def test_stretch_runs_respect_opponent_share_and_duration():
    # 12-6: opponent share 50% > 40% -> not a run
    plays = [(1, 700, "home", 3), (1, 690, "away", 2), (1, 680, "home", 3), (1, 670, "away", 2),
             (1, 660, "home", 3), (1, 650, "away", 2), (1, 640, "home", 3)]
    assert gs.stretch_runs(synth(plays), min_run=8) == []
    # 10-0 spread over 8 minutes is unanswered but too slow to be a stretch run (max 6 min)
    slow = [(1, 700, "home", 2), (1, 580, "home", 2), (1, 460, "home", 2), (1, 340, "home", 2), (1, 220, "home", 2)]
    ev = synth(slow)
    assert [r.label for r in gs.unanswered_runs(ev)] == ["10-0"]
    assert all(r.minutes <= 6.0 for r in gs.stretch_runs(ev))


@pytest.mark.parametrize("gid", GIDS)
def test_biggest_unanswered_run_matches_boxscore(gid):
    # The feed's team-level biggestScoringRun is sometimes the game-wide maximum copied to both
    # teams, so compare per team as an upper bound and the game-wide maximum exactly.
    box, _, events, _ = fixture(gid)
    runs = gs.unanswered_runs(events, min_run=1)
    boxed = {}
    for key, side in (("homeTeam", "home"), ("awayTeam", "away")):
        st = box["game"][key]["statistics"]
        boxed[side] = st["biggestScoringRun"]
        best = max((r.pts for r in runs if r.side == side), default=0)
        assert best <= st["biggestScoringRun"]
    top = max(runs, key=lambda r: r.pts)
    assert top.pts == max(boxed.values())
    # ... and the score at the end of it matches "biggestScoringRunScore" ("away-home")
    h, a = top.score_after
    assert f"{a}-{h}" in {box["game"][k]["statistics"]["biggestScoringRunScore"] for k in ("homeTeam", "awayTeam")}


# ------------------------------------------------------------------------- leads

@pytest.mark.parametrize("gid", GIDS)
def test_lead_changes_ties_and_largest_lead_match_boxscore(gid):
    box, _, events, _ = fixture(gid)
    hs, as_ = box["game"]["homeTeam"]["statistics"], box["game"]["awayTeam"]["statistics"]
    lc = gs.lead_changes_and_ties(events)
    assert lc["lead_changes"] == hs["leadChanges"]
    assert lc["times_tied"] == hs["timesTied"]
    ll = gs.largest_lead(events)
    assert (ll["home"] or {"lead": 0})["lead"] == hs["biggestLead"]
    assert (ll["away"] or {"lead": 0})["lead"] == as_["biggestLead"]


def test_comeback():
    _, _, events, _ = fixture("0022501066")  # LAC won at IND after trailing 45-21
    cb = gs.comeback(events)
    assert cb["side"] == "away" and cb["deficit"] == 24
    assert cb["at"]["score"] == [45, 21] and cb["at"]["label"] == "Q2"


# ------------------------------------------------------------------------- clutch

def test_clutch_window_boundaries():
    # Enter Q4 at 80-76 (+4 home).
    plays = [(3, 100, "home", 2), (3, 50, "away", 2)]                 # Q3: never clutch
    plays += [(4, 400, "home", 2),   # 5:00+ left -> not clutch; margin becomes +6
              (4, 290, "away", 3),   # pre-margin 6 -> not a clutch event; window opens (+3)
              (4, 200, "home", 3),   # pre-margin 3 -> clutch event; margin +6, window closes
              (4, 100, "away", 2)]   # pre-margin 6 -> not clutch; window reopens (+4)
    events = synth(plays, start=(78, 74))
    st = stints_for(events)
    span = gs.clutch_span(events, st)
    q4 = period_start(4)
    assert span.intervals == ((q4 + 430.0, q4 + 520.0), (q4 + 620.0, q4 + 720.0))
    assert span.seconds == 190.0
    clutch_pts = sorted(events[i].description for i in span.events if events[i].points)
    assert clutch_pts == ["home 3"]
    c = gs.clutch(events, st)
    assert c["seconds"] == 190.0
    assert c["points"] == {"home": 3, "away": 0}
    assert [k["kind"] for k in c["key_plays"]] == []   # 3-pointer extended a lead


def test_clutch_inclusive_at_five_minutes_and_in_overtime():
    plays = [(4, 300.1, "home", 2),   # 5:00.1 left: not clutch
             (4, 300.0, "away", 2),   # exactly 5:00: clutch (pre-margin +2)
             (4, 10.0, "away", 2),    # go-ahead at 0:10? score was tied -> go-ahead
             (4, 5.0, "home", 2),     # tying
             (5, 250.0, "home", 3)]   # OT: clutch, go-ahead
    events = synth(plays, start=(90, 90))
    st = stints_for(events)
    span = gs.clutch_span(events, st)
    descs = [(events[i].period, events[i].clock) for i in sorted(span.events) if events[i].points]
    assert (4, 300.1) not in descs
    assert (4, 300.0) in descs and (5, 250.0) in descs
    q4 = period_start(4)
    assert span.intervals[0] == (q4 + 420.0, q4 + 720.0 + 300.0)   # 5:00 of Q4 through end of OT
    c = gs.clutch(events, st)
    kinds = [(k["kind"], k["side"], k["period"]) for k in c["key_plays"]]
    assert ("go_ahead", "away", 4) in kinds and ("tying", "home", 4) in kinds
    last = c["key_plays"][-1]
    assert last["kind"] == "go_ahead" and last["side"] == "home" and last.get("decisive") is True


def test_no_clutch_in_a_blowout():
    _, _, events, stints = fixture("0022500622")
    assert gs.clutch(events, stints)["seconds"] == 0.0


@pytest.mark.parametrize("gid", ["0022500001", "0022500083", "0022500095", "0022501066"])
def test_clutch_player_points_add_up(gid):
    _, _, events, stints = fixture(gid)
    c = gs.clutch(events, stints)
    assert c["seconds"] > 0
    for side in ("home", "away"):
        assert sum(p.get("pts", 0) for p in c["players"] if p["player_id"] in
                   {pid for s in stints for pid in s.lineup(side)}) == c["points"][side]


# ------------------------------------------------------------------------- garbage time

def test_garbage_time_thresholds_and_starters_rule():
    starters = HOME_P + AWAY_P
    # Home leads by 12 entering Q4; bench units in from 10:00.
    plays = [(4, 200.0, "away", 3),   # 3:20 -> margin 9 -> garbage ends
             (4, 100.0, "home", 2)]   # 1:40 -> margin 11 -> garbage resumes
    events = synth(plays, start=(100, 88))
    bench = {4: [(720.0, HOME_P, AWAY_P), (600.0, HOME_BENCH[:4] + (HOME_P[0],), AWAY_BENCH[:4] + (AWAY_P[0],))]}
    st = stints_for(events, bench)
    span = gs.garbage_time(events, st, starters)
    q4 = period_start(4)
    # margin 12: not >= 25 (12-9 min) nor >= 20 (9-6 min); from 6:00 the threshold is 10.
    assert span.intervals == ((q4 + 360.0, q4 + 520.0), (q4 + 620.0, q4 + 720.0))
    assert span.seconds == 260.0
    # Same game with three starters on the floor: never garbage time.
    three = {4: [(720.0, HOME_P, AWAY_P), (600.0, HOME_BENCH[:3] + HOME_P[:2], AWAY_BENCH[:4] + (AWAY_P[0],))]}
    assert gs.garbage_time(events, stints_for(events, three), starters).seconds == 0.0
    # 25+ in the first three minutes of the 4th is garbage immediately (with benches in).
    big = synth([(4, 500.0, "home", 2)], start=(110, 85))
    st_big = stints_for(big, {4: [(720.0, HOME_BENCH, AWAY_BENCH)]})
    assert gs.garbage_time(big, st_big, starters).intervals[0][0] == q4
    # ... but not in the 3rd quarter.
    q3 = synth([(3, 500.0, "home", 2)], start=(110, 60))
    assert gs.garbage_time(q3, stints_for(q3, {3: [(720.0, HOME_BENCH, AWAY_BENCH)]}), starters).seconds == 0.0


def test_garbage_time_excluded_from_impact_in_blowout():
    box, _, events, stints = fixture("0022500622")
    players = box_players(box)
    starters = [pid for pid, p in players.items() if p["starter"]]
    garbage = gs.garbage_time(events, stints, starters)
    assert garbage.seconds == 720.0 and garbage.intervals[0][0] == period_start(4)
    for g in GIDS:
        if g != "0022500622":
            b, _, ev, st = fixture(g)
            sts = [pid for pid, p in box_players(b).items() if p["starter"]]
            assert gs.garbage_time(ev, st, sts).seconds == 0.0
    impact = {p["player_id"]: p for p in gs.player_game_impact(box, stints, events, garbage=garbage)}
    q4 = [s for s in stints if s.period == 4]
    for pid, p in impact.items():
        side = p["side"]
        q4_pm = sum(s.pts_for(side) - s.pts_against(side) for s in q4 if pid in s.lineup(side))
        q4_sec = sum(s.duration for s in q4 if pid in s.lineup(side))
        assert p["plus_minus"] == players[pid]["plus_minus"]
        assert p["plus_minus_non_garbage"] == p["plus_minus"] - q4_pm
        assert p["minutes_non_garbage"] == round((players[pid]["seconds"] - q4_sec) / 60, 1)
    # runs inside garbage time never become the turning point
    tp = gs.turning_point(events, exclude=garbage)
    assert tp["run"]["end"]["period"] < 4


# ------------------------------------------------------------------------- impact / turning point

@pytest.mark.parametrize("gid", GIDS)
def test_player_game_impact(gid):
    box, _, events, stints = fixture(gid)
    players = box_players(box)
    imp = gs.player_game_impact(box, stints, events)
    assert {p["player_id"] for p in imp} == {pid for pid, p in players.items() if p["played"]}
    for p in imp:
        bp = players[p["player_id"]]
        assert p["plus_minus"] == bp["plus_minus"]
        assert p["minutes"] == round(bp["seconds"] / 60, 1)
        assert p["stocks"] == p["stl"] + p["blk"]
        if p["fga"] or p["fta"]:
            assert p["ts_pct"] == round(p["pts"] / (2 * (p["fga"] + 0.44 * p["fta"])), 3)
        if p["on_off"] is not None:
            assert p["on_off"] == round(p["on_net"] - p["off_net"], 1)
        assert p["winning_time"]["seconds"] >= 0
    # someone on the winning side was on the floor for the decisive run
    w = gs.winner(events)
    assert any(p["winning_time"]["stints"] > 0 and p["side"] == w for p in imp)


def test_on_off_is_consistent_with_team_totals():
    box, _, events, stints = fixture("0022500083")
    on, team = gs.on_court_tally(events, stints)
    for side in ("home", "away"):
        lineup_ids = {pid for s in stints for pid in s.lineup(side)}
        # five players on the floor at all times
        assert sum(on[pid].seconds for pid in lineup_ids) == pytest.approx(5 * team[side].seconds, abs=0.5)
        assert sum(on[pid].pts_for for pid in lineup_ids) == 5 * team[side].pts_for


@pytest.mark.parametrize("gid,expect", [
    ("0022500001", ("home", "flipped_lead", "19-7")),
    ("0022500083", ("away", "flipped_lead", "15-2")),
    ("0022501066", ("away", "levelled", "13-4")),
])
def test_turning_point_placeholder(gid, expect):
    _, _, events, _ = fixture(gid)
    tp = gs.turning_point(events)
    assert (tp["side"], tp["reason"], tp["run"]["label"]) == expect
    assert tp["method"] == "run" and tp["wp_swing"] is None


def test_turning_point_accepts_win_probability_series():
    plays = [(1, 700, "home", 2), (2, 600, "away", 3), (3, 500, "away", 3), (4, 100, "away", 2)]
    events = synth(plays)
    scoring = [e.idx for e in events if e.points]
    wp = [(scoring[0], 0.55), (scoring[1], 0.48), (scoring[2], 0.30), (scoring[3], 0.10)]
    tp = gs.turning_point(events, win_prob=wp)
    assert tp["method"] == "win_probability" and tp["side"] == "away"
    assert tp["wp_swing"] == pytest.approx(0.20)
    assert tp["at"]["period"] in (3, 4)
