"""Reconstruct who is on the floor: lineups ("stints") for both teams at every moment.

A stint is a maximal stretch with no substitution: both five-man units are fixed. Stints are
split at every substitution block (consecutive substitution events) and at period boundaries.

Period starters (the hard part). Substitutions alone are not enough because a player who plays a
whole period never appears in a substitution event. For each period and team:

1. **Period 1**: the boxscore ``starter == "1"`` players (authoritative).
2. **Later periods**: walk the period's events and record each player's *first evidence*:
   - a ``SUB in`` first  -> he started the period on the bench;
   - a ``SUB out`` first -> he started on the floor (covers players subbed out without ever
     being subbed in, i.e. players who started the period);
   - any on-court action first (shot, rebound, foul, turnover, assist, steal, block, foul drawn,
     jump ball) -> he started on the floor. Technical fouls and ejections are ignored as
     evidence because bench players can draw those.
3. If that yields fewer than five, the missing starters played the whole period without
   registering a single action. They are resolved with the boxscore: a provisional pass computes
   every player's on-court seconds, and the silent players with the largest unexplained residual
   (boxscore seconds minus reconstructed seconds) are added (ties broken by "was on the floor at
   the end of the previous period").
4. If evidence yields more than five (a missing substitution in the feed) the five with the
   earliest evidence are kept and an anomaly is recorded.

``validate_stints`` reconciles the reconstruction against the boxscore: per-player on-court
seconds vs ``minutes`` (tolerance 1 s) and on-court plus-minus vs ``plusMinusPoints`` (exact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from courtside.analytics.pbp import AWAY, HOME, Event, parse_clock, period_end, period_start

SIDES = (HOME, AWAY)


@dataclass(frozen=True, slots=True)
class Stint:
    period: int
    start: float                  # elapsed seconds
    end: float
    first_idx: int                # events[first_idx:last_idx] belong to this stint
    last_idx: int
    home: tuple[int, ...]         # sorted player ids
    away: tuple[int, ...]
    home_pts: int
    away_pts: int
    home_poss: int                # offensive possessions ended (counting method)
    away_poss: int

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def lineup(self, side: str) -> tuple[int, ...]:
        return self.home if side == HOME else self.away

    def pts_for(self, side: str) -> int:
        return self.home_pts if side == HOME else self.away_pts

    def pts_against(self, side: str) -> int:
        return self.away_pts if side == HOME else self.home_pts

    def poss(self, side: str) -> int:
        """Offensive possessions of ``side``."""
        return self.home_poss if side == HOME else self.away_poss

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period, "start": self.start, "end": self.end,
            "home": list(self.home), "away": list(self.away),
            "home_pts": self.home_pts, "away_pts": self.away_pts,
            "home_poss": self.home_poss, "away_poss": self.away_poss,
        }


@dataclass
class Rotation:
    stints: list[Stint]
    starters: dict[tuple[int, str], tuple[int, ...]]   # (period, side) -> five ids
    anomalies: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------- boxscore helpers

def box_players(box_payload: dict) -> dict[int, dict]:
    """person_id -> {side, team, name, starter, played, seconds, plus_minus, stats}."""
    g = (box_payload or {}).get("game") or {}
    out: dict[int, dict] = {}
    for key, side in (("homeTeam", HOME), ("awayTeam", AWAY)):
        t = g.get(key) or {}
        for p in t.get("players") or []:
            s = p.get("statistics") or {}
            pid = int(p["personId"])
            out[pid] = {
                "id": pid,
                "side": side,
                "team": t.get("teamTricode"),
                "name": p.get("name") or p.get("nameI"),
                "name_i": p.get("nameI") or p.get("name"),
                "position": p.get("position") or None,
                "starter": str(p.get("starter")) == "1",
                "played": str(p.get("played")) == "1",
                "status": p.get("status"),
                "not_playing_reason": p.get("notPlayingReason"),
                "seconds": parse_clock(s.get("minutes")),
                "plus_minus": int(round(float(s.get("plusMinusPoints") or 0))),
                "stats": s,
            }
    return out


def box_starters(box_payload: dict) -> dict[str, tuple[int, ...]]:
    players = box_players(box_payload)
    return {side: tuple(sorted(pid for pid, p in players.items() if p["side"] == side and p["starter"]))
            for side in SIDES}


# ------------------------------------------------------------------------- evidence

def _evidence_ids(e: Event) -> list[int]:
    """Players this event proves were on the floor (excluding substitutions)."""
    if e.is_sub or e.is_ejection or e.action_type in ("timeout", "period", "game", "instantreplay"):
        return []
    if e.is_foul and e.foul_kind == "technical":
        return []
    ids = [e.player_id, e.assist_id, e.steal_id, e.block_id]
    if not (e.is_foul and e.foul_kind == "technical"):
        ids.append(e.foul_drawn_id)
    return [i for i in ids if i]


def _period_evidence(events: list[Event], period: int, pside: dict[int, str]) -> dict[int, tuple[str, int]]:
    """player -> ('on'|'bench', event idx) from first evidence in the period."""
    ev: dict[int, tuple[str, int]] = {}
    for e in events:
        if e.period != period:
            continue
        if e.is_sub and e.player_id:
            if e.player_id not in ev:
                ev[e.player_id] = ("bench" if e.sub_dir == "in" else "on", e.idx)
            continue
        for pid in _evidence_ids(e):
            if pid in pside and pid not in ev:
                ev[pid] = ("on", e.idx)
    return ev


# ------------------------------------------------------------------------- simulation

def credit_index(events: list[Event]) -> dict[int, int]:
    """Free throw idx -> idx of the foul that produced it.

    The NBA credits free-throw points (for plus-minus) to the players on the floor when the foul
    was committed, even if substitutions are made between the foul and the free throws. We find the
    most recent foul at the same period and game clock before the free throw.
    """
    out: dict[int, int] = {}
    for e in events:
        if not e.is_ft:
            continue
        for j in range(e.idx - 1, max(-1, e.idx - 40), -1):
            f = events[j]
            if f.period != e.period or f.clock != e.clock:
                break
            if f.is_foul:
                out[e.idx] = j
                break
    return out


def _bound(p: int, t0: float, t1: float, first: int, last: int, on: dict[str, set[int]]) -> list:
    return [p, t0, t1, first, last, tuple(sorted(on[HOME])), tuple(sorted(on[AWAY]))]


def _simulate(events: list[Event], starters: dict[tuple[int, str], set[int]],
              periods: list[int], anomalies: list[str]) -> list[Stint]:
    """Lineup boundaries per period, then a tally pass for points and possessions."""
    bounds: list[list] = []   # [period, start, end, first_idx, last_idx, home, away]
    by_period: dict[int, list[Event]] = {}
    for e in events:
        by_period.setdefault(e.period, []).append(e)

    for p in periods:
        pev = by_period.get(p, [])
        on = {s: set(starters.get((p, s), set())) for s in SIDES}
        t0 = period_start(p)
        first = pev[0].idx if pev else (bounds[-1][4] if bounds else 0)

        in_block = False
        for e in pev:
            if e.is_sub:
                if not in_block:
                    bounds.append(_bound(p, t0, e.elapsed, first, e.idx, on))
                    in_block = True
                pid, side = e.player_id, e.side
                if not pid or side not in SIDES:
                    continue
                if e.sub_dir == "out":
                    if pid in on[side]:
                        on[side].discard(pid)
                    else:
                        anomalies.append(f"P{p} {e.clock}: sub out of {pid} who was not on the floor")
                else:
                    if pid in on[side]:
                        anomalies.append(f"P{p} {e.clock}: sub in of {pid} who was already on the floor")
                    on[side].add(pid)
                continue
            if in_block:
                in_block = False
                t0, first = bounds[-1][2], e.idx
                for s in SIDES:
                    if len(on[s]) != 5:
                        anomalies.append(f"P{p} {e.clock}: {s} has {len(on[s])} players on the floor")
        if in_block:  # period ended on a substitution block
            t0, first = bounds[-1][2], pev[-1].idx + 1
        bounds.append(_bound(p, t0, period_end(p), first, (pev[-1].idx + 1) if pev else first, on))

    # Tally pass. Free throws are credited to the stint of their foul.
    firsts = [b[3] for b in bounds]
    lasts = [b[4] for b in bounds]
    credit = credit_index(events)
    tally = [[0, 0, 0, 0] for _ in bounds]   # home_pts, away_pts, home_poss, away_poss

    def stint_of(idx: int) -> int | None:
        lo, hi = 0, len(bounds) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < firsts[mid]:
                hi = mid - 1
            elif idx >= lasts[mid]:
                lo = mid + 1
            else:
                return mid
        return None

    for e in events:
        if not e.points and not e.ends_possession:
            continue
        k = stint_of(credit.get(e.idx, e.idx))
        if k is None:
            k = stint_of(e.idx)
        if k is None:
            anomalies.append(f"event {e.idx} not inside any stint")
            continue
        if e.points:
            tally[k][0 if e.side == HOME else 1] += e.points
        if e.ends_possession and e.poss_side in SIDES:
            tally[k][2 if e.poss_side == HOME else 3] += 1

    stints = [Stint(period=b[0], start=b[1], end=b[2], first_idx=b[3], last_idx=b[4],
                    home=b[5], away=b[6], home_pts=t[0], away_pts=t[1], home_poss=t[2], away_poss=t[3])
              for b, t in zip(bounds, tally)]
    # Drop empty zero-length stints (pure substitution blocks); keep zero-length ones that carry
    # points or possessions.
    return [s for s in stints
            if s.duration > 0 or s.home_pts or s.away_pts or s.home_poss or s.away_poss]


def build_rotation(events: list[Event], box_payload: dict) -> Rotation:
    players = box_players(box_payload)
    pside = {pid: p["side"] for pid, p in players.items()}
    periods = sorted({e.period for e in events}) or [1, 2, 3, 4]
    anomalies: list[str] = []
    starters: dict[tuple[int, str], set[int]] = {}
    ambiguous: dict[tuple[int, str], tuple[int, set[int]]] = {}   # need k from candidates

    bstart = box_starters(box_payload)
    for p in periods:
        ev = _period_evidence(events, p, pside)
        for s in SIDES:
            if p == 1 and len(bstart[s]) == 5:
                starters[(p, s)] = set(bstart[s])
                for pid, (kind, _) in ev.items():
                    if pside.get(pid) == s and kind == "on" and pid not in bstart[s]:
                        anomalies.append(f"P1 {s}: {pid} acted before entering but is not a box starter")
                continue
            on = sorted(((idx, pid) for pid, (kind, idx) in ev.items()
                         if kind == "on" and pside.get(pid) == s))
            chosen = {pid for _, pid in on[:5]}
            if len(on) > 5:
                anomalies.append(f"P{p} {s}: {len(on)} players with on-floor evidence; kept earliest 5")
            starters[(p, s)] = chosen
            if len(chosen) < 5:
                silent = {pid for pid, pl in players.items()
                          if pl["side"] == s and pl["played"] and pid not in ev}
                ambiguous[(p, s)] = (5 - len(chosen), silent)

    if ambiguous:
        # Provisional pass, then fill silent full-period players by boxscore residual.
        prov = _simulate(events, starters, periods, [])
        secs = _seconds(prov)
        for (p, s), (k, silent) in sorted(ambiguous.items()):
            prev_on = set(starters_at_end(prov, p - 1, s)) if p > 1 else set()
            plen = period_end(p) - period_start(p)
            ranked = sorted(silent, key=lambda pid: (-(players[pid]["seconds"] - secs.get(pid, 0.0)),
                                                     pid not in prev_on, pid))
            pick = ranked[:k]
            for pid in pick:
                resid = players[pid]["seconds"] - secs.get(pid, 0.0)
                if resid < plen - 1.5:
                    anomalies.append(f"P{p} {s}: filled silent starter {pid} with residual {resid:.1f}s")
                secs[pid] = secs.get(pid, 0.0) + plen
            starters[(p, s)] |= set(pick)

    stints = _simulate(events, starters, periods, anomalies)
    return Rotation(
        stints=stints,
        starters={k: tuple(sorted(v)) for k, v in starters.items()},
        anomalies=anomalies,
    )


def build_stints(events: list[Event], box_payload: dict) -> list[Stint]:
    return build_rotation(events, box_payload).stints


def starters_at_end(stints: list[Stint], period: int, side: str) -> tuple[int, ...]:
    last = [s for s in stints if s.period == period]
    return last[-1].lineup(side) if last else ()


def _seconds(stints: list[Stint]) -> dict[int, float]:
    out: dict[int, float] = {}
    for st in stints:
        for pid in st.home + st.away:
            out[pid] = out.get(pid, 0.0) + st.duration
    return out


def on_court_seconds(stints: list[Stint]) -> dict[int, float]:
    return {k: round(v, 2) for k, v in _seconds(stints).items()}


def on_court_plus_minus(stints: list[Stint]) -> dict[int, int]:
    out: dict[int, int] = {}
    for st in stints:
        d = st.home_pts - st.away_pts
        for pid in st.home:
            out[pid] = out.get(pid, 0) + d
        for pid in st.away:
            out[pid] = out.get(pid, 0) - d
    return out


def lineup_at(stints: list[Stint], event_idx: int) -> Stint | None:
    """The stint containing event index ``event_idx`` (binary search)."""
    lo, hi = 0, len(stints) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = stints[mid]
        if event_idx < s.first_idx:
            hi = mid - 1
        elif event_idx >= s.last_idx:
            lo = mid + 1
        else:
            return s
    return None


# ------------------------------------------------------------------------- validation

@dataclass(frozen=True, slots=True)
class PlayerCheck:
    player_id: int
    name: str
    side: str
    box_seconds: float
    seconds: float
    box_plus_minus: int
    plus_minus: int

    @property
    def seconds_diff(self) -> float:
        return round(self.seconds - self.box_seconds, 2)

    @property
    def minutes_ok(self) -> bool:
        return abs(self.seconds_diff) <= 1.0

    @property
    def plus_minus_ok(self) -> bool:
        return self.plus_minus == self.box_plus_minus


@dataclass
class ValidationReport:
    checks: list[PlayerCheck]
    anomalies: list[str]

    @property
    def ok(self) -> bool:
        return all(c.minutes_ok and c.plus_minus_ok for c in self.checks)

    @property
    def max_seconds_diff(self) -> float:
        return max((abs(c.seconds_diff) for c in self.checks), default=0.0)

    @property
    def minutes_failures(self) -> list[PlayerCheck]:
        return [c for c in self.checks if not c.minutes_ok]

    @property
    def plus_minus_failures(self) -> list[PlayerCheck]:
        return [c for c in self.checks if not c.plus_minus_ok]

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "players": len(self.checks),
            "max_minutes_diff_s": round(self.max_seconds_diff, 2),
            "minutes_mismatches": len(self.minutes_failures),
            "plus_minus_mismatches": len(self.plus_minus_failures),
            "anomalies": len(self.anomalies),
        }


def validate_stints(box_payload: dict, stints: list[Stint], anomalies: list[str] | None = None) -> ValidationReport:
    players = box_players(box_payload)
    secs = on_court_seconds(stints)
    pm = on_court_plus_minus(stints)
    checks = []
    for pid, p in players.items():
        if not p["played"] and pid not in secs:
            continue
        checks.append(PlayerCheck(
            player_id=pid, name=p["name"], side=p["side"],
            box_seconds=p["seconds"], seconds=secs.get(pid, 0.0),
            box_plus_minus=p["plus_minus"], plus_minus=pm.get(pid, 0)))
    for pid in secs:
        if pid not in players:
            checks.append(PlayerCheck(pid, "?", "?", 0.0, secs[pid], 0, pm.get(pid, 0)))
    return ValidationReport(checks=checks, anomalies=list(anomalies or []))
