"""Game story: runs, lead changes, largest leads, comebacks, clutch, garbage time, impact.

Everything here is a pure function of normalized events (``pbp.normalize``) and stints
(``stints.build_stints``). Sides are "home"/"away"; scores are ``(home, away)`` tuples. The facts
document (``summarize.game_facts``) maps sides to tricodes and names.

Definitions and thresholds are documented in docs/ANALYTICS.md.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from courtside.analytics.pbp import (
    AWAY, HOME, REG_PERIODS, Event, final_score, fmt_clock, period_end, period_label,
    period_start,
)
from courtside.analytics.stints import SIDES, Stint, box_players, credit_index

# ------------------------------------------------------------------------- thresholds

CLUTCH_CLOCK = 300.0          # last 5:00 of the 4th quarter or any overtime
CLUTCH_MARGIN = 5             # score within 5 (NBA.com definition)

# Cleaning-the-Glass-style garbage time (4th quarter only):
#   margin >= 25 with 12:00-9:01 left, >= 20 with 9:00-6:01 left, >= 10 with 6:00 or less,
#   AND two or fewer of the game's starters (both teams combined) on the floor.
GARBAGE_RULES: tuple[tuple[float, int], ...] = ((540.0, 25), (360.0, 20), (0.0, 10))
GARBAGE_MAX_STARTERS = 2

RUN_MAX_OPP_SHARE = 0.4       # an "X-Y run": opponent scored at most 40% of X
RUN_MAX_MINUTES = 6.0         # ... within at most 6 minutes of game clock
MIN_POSS_FOR_RATING = 10      # on/off ratings need at least 10 possessions each side


# ------------------------------------------------------------------------- moments

def moment(e: Event) -> dict[str, Any]:
    """Compact, JSON-friendly description of when an event happened."""
    return {"period": e.period, "label": period_label(e.period), "clock": fmt_clock(e.clock),
            "elapsed": e.elapsed, "score": [e.score_home, e.score_away]}


def _scoring(events: Sequence[Event]) -> list[Event]:
    return [e for e in events if e.points]


def _pre_score(events: Sequence[Event], idx: int) -> tuple[int, int]:
    return (events[idx - 1].score_home, events[idx - 1].score_away) if idx > 0 else (0, 0)


def winner(events: Sequence[Event]) -> str | None:
    h, a = final_score(events)
    return HOME if h > a else AWAY if a > h else None


def _lead(score: tuple[int, int], side: str) -> int:
    d = score[0] - score[1]
    return d if side == HOME else -d


# ------------------------------------------------------------------------- runs

@dataclass(frozen=True, slots=True)
class Run:
    side: str
    pts: int
    opp_pts: int
    unanswered: bool
    start_idx: int              # first scoring event of the run (side scored)
    end_idx: int                # last scoring event of the run (side scored)
    start_elapsed: float
    end_elapsed: float
    score_before: tuple[int, int]
    score_after: tuple[int, int]

    @property
    def net(self) -> int:
        return self.pts - self.opp_pts

    @property
    def minutes(self) -> float:
        return round((self.end_elapsed - self.start_elapsed) / 60.0, 1)

    @property
    def label(self) -> str:
        return f"{self.pts}-{self.opp_pts}"

    def to_dict(self, events: Sequence[Event]) -> dict[str, Any]:
        s, e = events[self.start_idx], events[self.end_idx]
        return {
            "side": self.side, "label": self.label, "pts": self.pts, "opp_pts": self.opp_pts,
            "unanswered": self.unanswered, "minutes": self.minutes,
            "start": {**moment(s), "score": list(self.score_before)},
            "end": moment(e),
        }


def unanswered_runs(events: Sequence[Event], min_run: int = 8) -> list[Run]:
    """Maximal stretches where one team scored every point, total >= ``min_run``."""
    out: list[Run] = []
    sc = _scoring(events)
    i = 0
    while i < len(sc):
        j = i
        pts = sc[i].points
        while j + 1 < len(sc) and sc[j + 1].side == sc[i].side:
            j += 1
            pts += sc[j].points
        if pts >= min_run and sc[i].side in SIDES:
            out.append(_make_run(events, sc[i], sc[j], sc[i].side, pts, 0, True))
        i = j + 1
    return out


def _make_run(events, first: Event, last: Event, side: str, pts: int, opp: int, unanswered: bool) -> Run:
    return Run(side=side, pts=pts, opp_pts=opp, unanswered=unanswered,
               start_idx=first.idx, end_idx=last.idx,
               start_elapsed=first.elapsed, end_elapsed=last.elapsed,
               score_before=_pre_score(events, first.idx),
               score_after=(last.score_home, last.score_away))


def stretch_runs(events: Sequence[Event], min_run: int = 8,
                 max_opp_share: float = RUN_MAX_OPP_SHARE,
                 max_minutes: float = RUN_MAX_MINUTES) -> list[Run]:
    """"X-Y runs" (e.g. 14-2): non-overlapping stretches that begin and end with the team scoring,
    with X >= ``min_run``, X - Y >= ``min_run``, Y <= ``max_opp_share`` * X, lasting at most
    ``max_minutes``. Chosen greedily by net margin (ties: shorter first)."""
    sc = _scoring(events)
    cands: list[tuple[int, float, int, int, int, int]] = []
    for side in SIDES:
        for i, a in enumerate(sc):
            if a.side != side:
                continue
            pf = pa = 0
            for j in range(i, len(sc)):
                b = sc[j]
                if b.elapsed - a.elapsed > max_minutes * 60:
                    break
                if b.side == side:
                    pf += b.points
                else:
                    pa += b.points
                    continue
                if pf >= min_run and pf - pa >= min_run and pa <= max_opp_share * pf:
                    cands.append((pf - pa, b.elapsed - a.elapsed, i, j, pf, pa))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    taken: list[tuple[int, int]] = []
    out: list[Run] = []
    for _net, _, i, j, pf, pa in cands:
        if any(not (j < ti or i > tj) for ti, tj in taken):
            continue
        taken.append((i, j))
        out.append(_make_run(events, sc[i], sc[j], sc[i].side, pf, pa, pa == 0))
    return sorted(out, key=lambda r: r.start_idx)


def scoring_runs(events: Sequence[Event], min_run: int = 8, **kw: Any) -> list[Run]:
    """All runs worth mentioning, chronological: unanswered runs (``unanswered=True``) plus "X-Y"
    stretch runs that are not identical to an unanswered run."""
    un = unanswered_runs(events, min_run)
    seen = {(r.start_idx, r.end_idx) for r in un}
    st = [r for r in stretch_runs(events, min_run, **kw) if (r.start_idx, r.end_idx) not in seen]
    return sorted(un + st, key=lambda r: (r.start_idx, -r.net))


# ------------------------------------------------------------------------- leads

def lead_changes_and_ties(events: Sequence[Event]) -> dict[str, Any]:
    """Lead changes (the leading team switches, possibly via a tie) and ties (score becomes level,
    0-0 excluded). Matches the boxscore ``leadChanges`` / ``timesTied`` counting."""
    leader: str | None = None
    lc = ties = 0
    last_lc: Event | None = None
    last_tie: Event | None = None
    for e in _scoring(events):
        d = e.score_home - e.score_away
        if d == 0:
            ties += 1
            last_tie = e
            continue
        now = HOME if d > 0 else AWAY
        if leader is not None and now != leader:
            lc += 1
            last_lc = e
        leader = now
    return {"lead_changes": lc, "times_tied": ties,
            "last_lead_change": moment(last_lc) if last_lc else None,
            "last_tie": moment(last_tie) if last_tie else None}


def largest_lead(events: Sequence[Event]) -> dict[str, dict[str, Any] | None]:
    """Per side: the largest lead and the first moment it was reached."""
    best: dict[str, tuple[int, Event | None]] = {HOME: (0, None), AWAY: (0, None)}
    for e in _scoring(events):
        d = e.score_home - e.score_away
        side = HOME if d > 0 else AWAY
        if abs(d) > best[side][0]:
            best[side] = (abs(d), e)
    return {s: ({"lead": v, **moment(e)} if e else None) for s, (v, e) in best.items()}


def comeback(events: Sequence[Event]) -> dict[str, Any] | None:
    """The largest deficit the eventual winner overcame (None for a tie / no events)."""
    w = winner(events)
    if w is None:
        return None
    worst, at = 0, None
    for e in _scoring(events):
        deficit = -_lead((e.score_home, e.score_away), w)
        if deficit > worst:
            worst, at = deficit, e
    out: dict[str, Any] = {"side": w, "deficit": worst}
    if at is not None:
        out.update(at=moment(at))
    return out


# ------------------------------------------------------------------------- windows

@dataclass(frozen=True, slots=True)
class Span:
    """A set of game-time intervals plus the events that occurred inside them."""
    intervals: tuple[tuple[float, float], ...]
    events: frozenset[int]

    @property
    def seconds(self) -> float:
        return round(float(sum(b - a for a, b in self.intervals)), 2)

    def contains_time(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.intervals)

    def overlap(self, start: float, end: float) -> float:
        return sum(max(0.0, min(b, end) - max(a, start)) for a, b in self.intervals)

    def to_list(self) -> list[list[float]]:
        return [[a, b] for a, b in self.intervals]


EMPTY = Span((), frozenset())

# predicate(period, clock, margin_home, lineup_stint) -> bool
Predicate = Callable[[int, float, int, Stint | None], bool]


def _stint_finder(stints: Sequence[Stint]) -> Callable[[int], Stint | None]:
    firsts = [s.first_idx for s in stints]

    def find(idx: int) -> Stint | None:
        k = bisect_right(firsts, idx) - 1
        return stints[k] if k >= 0 else (stints[0] if stints else None)
    return find


def build_span(events: Sequence[Event], stints: Sequence[Stint], pred: Predicate,
               thresholds: Callable[[int], Iterable[float]] = lambda p: ()) -> Span:
    """Evaluate ``pred`` on the piecewise-constant game state.

    The state between consecutive events is (period, margin after the previous event, lineup).
    Clock-based thresholds split segments so that each segment sits in a single clock bucket; a
    segment is tested at its midpoint clock. An event belongs to the span if ``pred`` holds for its
    *pre-event* state at its own clock (so the basket that pushes the margin from 5 to 7 in the
    last five minutes is still a clutch basket)."""
    find = _stint_finder(stints)
    intervals: list[list[float]] = []
    members: set[int] = set()

    def add(a: float, b: float) -> None:
        if b <= a:
            return
        if intervals and abs(intervals[-1][1] - a) < 1e-9:
            intervals[-1][1] = b
        else:
            intervals.append([a, b])

    by_period: dict[int, list[Event]] = {}
    for e in events:
        by_period.setdefault(e.period, []).append(e)
    prev_score = (0, 0)
    for p in sorted(by_period):
        pev = by_period[p]
        p0, p1 = period_start(p), period_end(p)
        cuts = sorted({p1 - c for c in thresholds(p) if 0 < c < p1 - p0})
        t = p0
        for e in pev:
            margin = prev_score[0] - prev_score[1]
            st = find(e.idx)
            for a, b in _split(t, e.elapsed, cuts):
                mid_clock = p1 - (a + b) / 2
                if pred(p, mid_clock, margin, st):
                    add(a, b)
            if pred(p, e.clock, margin, st):
                members.add(e.idx)
            t = max(t, e.elapsed)
            prev_score = (e.score_home, e.score_away)
        margin = prev_score[0] - prev_score[1]
        st = find(pev[-1].idx) if pev else None
        for a, b in _split(t, p1, cuts):
            if pred(p, p1 - (a + b) / 2, margin, st):
                add(a, b)
    return Span(tuple((round(a, 2), round(b, 2)) for a, b in intervals), frozenset(members))


def _split(a: float, b: float, cuts: list[float]) -> list[tuple[float, float]]:
    pts = [a] + [c for c in cuts if a < c < b] + [b]
    return [(x, y) for x, y in zip(pts, pts[1:]) if y > x]


def clutch_span(events: Sequence[Event], stints: Sequence[Stint]) -> Span:
    def pred(period, clock, margin, _st):
        return period >= REG_PERIODS and clock <= CLUTCH_CLOCK and abs(margin) <= CLUTCH_MARGIN
    return build_span(events, stints, pred, lambda p: (CLUTCH_CLOCK,) if p >= REG_PERIODS else ())


def garbage_threshold(clock: float) -> int:
    for floor, margin in GARBAGE_RULES:
        if clock > floor:
            return margin
    return GARBAGE_RULES[-1][1]


def garbage_time(events: Sequence[Event], stints: Sequence[Stint],
                 starters: Iterable[int]) -> Span:
    """Cleaning-the-Glass-style garbage time (see GARBAGE_RULES). ``starters`` are the game starters
    of both teams (boxscore ``starter == "1"``). Evaluated continuously, so garbage time ends if the
    margin drops back under the threshold or starters return."""
    starter_set = set(starters)

    def pred(period, clock, margin, st):
        if period != REG_PERIODS:
            return False
        if abs(margin) < garbage_threshold(clock):
            return False
        on = len(starter_set & set(st.home + st.away)) if st else 10
        return on <= GARBAGE_MAX_STARTERS
    return build_span(events, stints, pred,
                      lambda p: tuple(f for f, _ in GARBAGE_RULES if f > 0) if p == REG_PERIODS else ())


def run_span(events: Sequence[Event], run: Run) -> Span:
    idx = frozenset(range(run.start_idx, run.end_idx + 1))
    return Span(((run.start_elapsed, run.end_elapsed),), idx)


def complement(span: Span, events: Sequence[Event], game_end: float) -> Span:
    out, t = [], 0.0
    for a, b in span.intervals:
        if a > t:
            out.append((t, a))
        t = max(t, b)
    if t < game_end:
        out.append((t, game_end))
    return Span(tuple(out), frozenset(e.idx for e in events) - span.events)


# ------------------------------------------------------------------------- on-court tallies

@dataclass
class Tally:
    seconds: float = 0.0
    pts_for: int = 0
    pts_against: int = 0
    poss_for: int = 0          # team offensive possessions while on court
    poss_against: int = 0      # opponent offensive possessions while on court

    @property
    def plus_minus(self) -> int:
        return self.pts_for - self.pts_against

    @property
    def possessions(self) -> float:
        return (self.poss_for + self.poss_against) / 2

    @property
    def net_rating(self) -> float | None:
        p = self.possessions
        return round(100.0 * self.plus_minus / p, 1) if p >= MIN_POSS_FOR_RATING else None


def on_court_tally(events: Sequence[Event], stints: Sequence[Stint],
                   span: Span | None = None) -> tuple[dict[int, Tally], dict[str, Tally]]:
    """Per-player on-court tallies and per-team totals, restricted to ``span`` if given.

    Seconds come from stint/span interval overlap; points and possessions from events in the span,
    credited to the lineup on the floor (free throws: the lineup at the time of the foul)."""
    find = _stint_finder(stints)
    credit = credit_index(list(events))
    players: dict[int, Tally] = {}
    team = {s: Tally() for s in SIDES}
    for st in stints:
        secs = st.duration if span is None else span.overlap(st.start, st.end)
        if secs <= 0:
            continue
        for s in SIDES:
            team[s].seconds += secs
            for pid in st.lineup(s):
                players.setdefault(pid, Tally()).seconds += secs
    for e in events:
        if not (e.points or e.ends_possession):
            continue
        if span is not None and e.idx not in span.events:
            continue
        st = find(credit.get(e.idx, e.idx))
        if st is None:
            continue
        for s in SIDES:
            t_list = [team[s]] + [players.setdefault(pid, Tally()) for pid in st.lineup(s)]
            for t in t_list:
                if e.points:
                    if e.side == s:
                        t.pts_for += e.points
                    else:
                        t.pts_against += e.points
                if e.ends_possession and e.poss_side in SIDES:
                    if e.poss_side == s:
                        t.poss_for += 1
                    else:
                        t.poss_against += 1
    for t in players.values():
        t.seconds = round(t.seconds, 2)
    return players, team


# ------------------------------------------------------------------------- box stats from events

def event_box(events: Iterable[Event]) -> dict[int, dict[str, int]]:
    """Counting stats per player from events (used for clutch / span splits)."""
    out: dict[int, dict[str, int]] = {}

    def row(pid: int) -> dict[str, int]:
        return out.setdefault(pid, {"pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0,
                                    "fta": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0, "reb": 0})
    for e in events:
        pid = e.player_id
        if e.is_fga and pid:
            r = row(pid)
            r["fga"] += 1
            r["fg3a"] += e.is_three
            if e.made:
                r["fgm"] += 1
                r["fg3m"] += e.is_three
        if e.is_ft and pid:
            r = row(pid)
            r["fta"] += 1
            r["ftm"] += bool(e.made)
        if e.points and pid:
            row(pid)["pts"] += e.points
        if e.assist_id and e.is_fga and e.made:
            row(e.assist_id)["ast"] += 1
        if e.is_steal and pid:
            row(pid)["stl"] += 1
        if e.is_block and pid:
            row(pid)["blk"] += 1
        if e.is_turnover and pid:
            row(pid)["tov"] += 1
        if e.is_rebound and pid:
            row(pid)["reb"] += 1
    return out


# ------------------------------------------------------------------------- clutch

def clutch(events: Sequence[Event], stints: Sequence[Stint], span: Span | None = None) -> dict[str, Any]:
    """Clutch time = last 5:00 of the 4th quarter or any OT with the margin within 5 (NBA def).

    Returns total clutch seconds, the windows, team points in clutch, per-player clutch lines
    (points, FG, 3P, FT, AST, STL, BLK, TOV, on-court +/-, seconds) and key plays: go-ahead and
    tying baskets, steals and blocks inside the window. The last go-ahead basket by the eventual
    winner after which they never trailed or were tied is flagged ``decisive``."""
    span = span if span is not None else clutch_span(events, stints)
    if not span.intervals and not span.events:
        return {"seconds": 0.0, "windows": [], "points": {HOME: 0, AWAY: 0}, "players": [], "key_plays": []}
    evs = [e for e in events if e.idx in span.events]
    tallies, team = on_court_tally(events, stints, span)
    box = event_box(evs)
    players = []
    for pid, t in tallies.items():
        if t.seconds <= 0 and pid not in box:
            continue
        b = box.get(pid, {})
        players.append({"player_id": pid, "seconds": t.seconds, "plus_minus": t.plus_minus, **b})
    players.sort(key=lambda r: (-r.get("pts", 0), -r["plus_minus"], r["player_id"]))

    w = winner(events)
    key: list[dict[str, Any]] = []
    for e in evs:
        kind = None
        if e.points:
            before = _pre_score(events, e.idx)
            lb, la = _lead(before, e.side), _lead((e.score_home, e.score_away), e.side)
            if lb <= 0 < la:
                kind = "go_ahead"
            elif lb < 0 == la:
                kind = "tying"
        elif e.is_steal:
            kind = "steal"
        elif e.is_block:
            kind = "block"
        if kind:
            key.append({"kind": kind, "player_id": e.player_id, "side": e.side, "idx": e.idx,
                        "clock_s": e.clock, "description": e.description, **moment(e)})
    # decisive go-ahead: winner's last go-ahead after which they always led
    if w:
        for k in reversed(key):
            if k["kind"] == "go_ahead" and k["side"] == w:
                if all(_lead((x.score_home, x.score_away), w) > 0 for x in events[k["idx"]:] if x.points):
                    k["decisive"] = True
                break
    return {
        "seconds": span.seconds,
        "windows": span.to_list(),
        "points": {s: team[s].pts_for for s in SIDES},
        "players": players,
        "key_plays": key,
    }


# ------------------------------------------------------------------------- turning point

def turning_point(events: Sequence[Event], runs: Sequence[Run] | None = None,
                  win_prob: Sequence[tuple[int, float]] | None = None,
                  exclude: Span | None = None) -> dict[str, Any] | None:
    """The decisive moment.

    * With ``win_prob`` (a series of ``(event_idx, home_win_probability)``): the largest swing in the
      eventual winner's favour between consecutive points (method ``"win_probability"``), plus the
      run containing it if any.
    * Placeholder (no model yet, method ``"run"``): among the eventual winner's runs (excluding
      garbage time), the one with the largest net margin that either *flipped* the lead (winner not
      ahead before, ahead after), *created the winning margin* (winner ahead after and never
      trailed or tied again) or *levelled* the game from behind. Falls back to the winner's
      largest run.
    """
    w = winner(events)
    if w is None:
        return None
    runs = list(runs) if runs is not None else scoring_runs(events)
    if exclude is not None:
        runs = [r for r in runs if r.end_idx not in exclude.events]
    mine = [r for r in runs if r.side == w]

    if win_prob:
        pts = sorted(win_prob)
        best = None
        for (i0, p0), (i1, p1) in zip(pts, pts[1:]):
            swing = (p1 - p0) if w == HOME else (p0 - p1)
            if best is None or swing > best[0]:
                best = (swing, i0, i1)
        if best is None:
            return None
        swing, i0, i1 = best
        e = events[i1]
        run = next((r for r in mine if r.start_idx <= i1 <= r.end_idx), None)
        return {"method": "win_probability", "side": w, "wp_swing": round(swing, 3),
                "at": moment(e), "description": e.description,
                "run": run.to_dict(events) if run else None}

    scores = [(e.idx, _lead((e.score_home, e.score_away), w)) for e in _scoring(events)]

    def always_ahead_after(idx: int) -> bool:
        return all(lead > 0 for i, lead in scores if i > idx)

    best_run, reason = None, None
    for r in sorted(mine, key=lambda r: (-r.net, r.start_idx)):
        before, after = _lead(r.score_before, w), _lead(r.score_after, w)
        if before <= 0 < after:
            best_run, reason = r, "flipped_lead"
            break
        if after > 0 and always_ahead_after(r.end_idx):
            best_run, reason = r, "created_winning_margin"
            break
        if before < 0 == after:
            best_run, reason = r, "levelled"
            break
    if best_run is None and mine:
        best_run, reason = max(mine, key=lambda r: (r.net, -r.start_idx)), "largest_run"
    if best_run is None:
        return None
    return {"method": "run", "side": w, "reason": reason, "run": best_run.to_dict(events),
            "wp_swing": None}


# ------------------------------------------------------------------------- player impact

def ts_pct(pts: int, fga: int, fta: int) -> float | None:
    tsa = fga + 0.44 * fta
    return round(pts / (2 * tsa), 3) if tsa > 0 else None


def game_score(s: dict[str, Any]) -> float:
    """Hollinger Game Score from boxscore statistics."""
    g = (s.get("points", 0) + 0.4 * s.get("fieldGoalsMade", 0) - 0.7 * s.get("fieldGoalsAttempted", 0)
         - 0.4 * (s.get("freeThrowsAttempted", 0) - s.get("freeThrowsMade", 0))
         + 0.7 * s.get("reboundsOffensive", 0) + 0.3 * s.get("reboundsDefensive", 0)
         + s.get("steals", 0) + 0.7 * s.get("assists", 0) + 0.7 * s.get("blocks", 0)
         - 0.4 * s.get("foulsPersonal", 0) - s.get("turnovers", 0))
    return round(g, 1)


def decisive_runs(events: Sequence[Event], runs: Sequence[Run], tp: dict[str, Any] | None,
                  exclude: Span | None = None) -> dict[str, Run | None]:
    """Each side's "winning time" run: the turning-point run for the winner, otherwise the side's
    largest non-garbage run."""
    out: dict[str, Run | None] = {}
    for s in SIDES:
        cand = [r for r in runs if r.side == s and (exclude is None or r.end_idx not in exclude.events)]
        pick = None
        if tp and tp.get("method") == "run" and tp["side"] == s:
            st, lab = tp["run"]["start"]["elapsed"], tp["run"]["label"]
            pick = next((r for r in cand if r.start_elapsed == st and r.label == lab), None)
        out[s] = pick or (max(cand, key=lambda r: (r.net, -r.start_idx)) if cand else None)
    return out


def player_game_impact(box_payload: dict, stints: Sequence[Stint], events: Sequence[Event],
                       garbage: Span | None = None,
                       runs: Sequence[Run] | None = None,
                       tp: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per-player impact for everyone who played.

    Fields: minutes, points, TS%, game score, stocks (stl+blk), on-court +/- (all game and excluding
    garbage time), on/off net rating differential (per 100 possessions, team net with the player on
    minus team net with him off; garbage time excluded; None if either side has < 10 possessions),
    and "winning time" (stints, seconds and +/- while on the floor during his team's decisive run).
    """
    players = box_players(box_payload)
    events = list(events)
    game_end = max((s.end for s in stints), default=0.0)
    starters = [pid for pid, p in players.items() if p["starter"]]
    if garbage is None:
        garbage = garbage_time(events, stints, starters)
    ng = complement(garbage, events, game_end)
    on_all, _ = on_court_tally(events, stints)
    on_ng, team_ng = on_court_tally(events, stints, ng)
    if runs is None:
        runs = scoring_runs(events)
    if tp is None:
        tp = turning_point(events, runs, exclude=garbage)
    dec = decisive_runs(events, runs, tp, garbage)
    dec_tally = {s: (on_court_tally(events, stints, run_span(events, r))[0] if r else {}) for s, r in dec.items()}

    out = []
    for pid, p in players.items():
        if not p["played"] and pid not in on_all:
            continue
        s = p["stats"]
        side = p["side"]
        a, n, tm = on_all.get(pid, Tally()), on_ng.get(pid, Tally()), team_ng[side]
        off = Tally(seconds=tm.seconds - n.seconds, pts_for=tm.pts_for - n.pts_for,
                    pts_against=tm.pts_against - n.pts_against,
                    poss_for=tm.poss_for - n.poss_for, poss_against=tm.poss_against - n.poss_against)
        on_net, off_net = n.net_rating, off.net_rating
        r = dec[side]
        wt = dec_tally[side].get(pid)
        n_wt_stints = 0
        if r:
            n_wt_stints = sum(1 for st in stints if pid in st.lineup(side)
                              and st.first_idx <= r.end_idx and st.last_idx > r.start_idx)
        out.append({
            "player_id": pid, "name": p["name"], "side": side, "team": p["team"],
            "starter": p["starter"], "position": p["position"],
            "minutes": round(p["seconds"] / 60.0, 1),
            "minutes_non_garbage": round(n.seconds / 60.0, 1),
            "pts": s.get("points", 0),
            "reb": s.get("reboundsTotal", 0), "oreb": s.get("reboundsOffensive", 0),
            "ast": s.get("assists", 0), "stl": s.get("steals", 0), "blk": s.get("blocks", 0),
            "tov": s.get("turnovers", 0), "pf": s.get("foulsPersonal", 0),
            "fgm": s.get("fieldGoalsMade", 0), "fga": s.get("fieldGoalsAttempted", 0),
            "fg3m": s.get("threePointersMade", 0), "fg3a": s.get("threePointersAttempted", 0),
            "ftm": s.get("freeThrowsMade", 0), "fta": s.get("freeThrowsAttempted", 0),
            "stocks": s.get("steals", 0) + s.get("blocks", 0),
            "ts_pct": ts_pct(s.get("points", 0), s.get("fieldGoalsAttempted", 0), s.get("freeThrowsAttempted", 0)),
            "game_score": game_score(s),
            "plus_minus": a.plus_minus,
            "plus_minus_non_garbage": n.plus_minus,
            "on_net": on_net, "off_net": off_net,
            "on_off": round(on_net - off_net, 1) if on_net is not None and off_net is not None else None,
            "on_poss": n.possessions, "off_poss": off.possessions,
            "winning_time": {
                "stints": n_wt_stints,
                "seconds": wt.seconds if wt else 0.0,
                "plus_minus": wt.plus_minus if wt else 0,
            },
        })
    out.sort(key=lambda r: (r["side"] != HOME, -r["minutes"], r["player_id"]))
    return out
