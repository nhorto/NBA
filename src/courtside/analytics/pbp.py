"""Normalize a cdn.nba.com play-by-play payload into typed, ordered events.

Pure functions only: input is the raw ``playbyplay_{game_id}.json`` payload (and optionally the
boxscore payload, used to map team ids to home/away). No DB, no network.

Conventions
-----------
* Events are ordered by ``orderNumber`` (the feed's canonical order; ``actionNumber`` is not
  monotonic once the scorer edits plays).
* ``clock`` is seconds remaining in the period; ``elapsed`` is game seconds elapsed at the
  event. Regulation periods are 720 s, overtime periods 300 s.
* ``score_home`` / ``score_away`` are the score *after* the event. ``points`` is what this event
  scored (made 2pt = 2, made 3pt = 3, made FT = 1).
* ``side`` is "home" / "away" for the team that performed the action.
* Possession accounting: ``ends_possession`` marks the event that ends an offensive possession and
  ``poss_side`` the offense that owned it. See ``_mark_possessions`` and docs/ANALYTICS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

REG_PERIODS = 4
REG_LEN = 720.0
OT_LEN = 300.0

HOME, AWAY = "home", "away"


def other(side: str | None) -> str | None:
    return {HOME: AWAY, AWAY: HOME}.get(side or "")


def period_length(period: int) -> float:
    return REG_LEN if period <= REG_PERIODS else OT_LEN


def period_start(period: int) -> float:
    """Elapsed game seconds at the start of ``period`` (1-based)."""
    if period <= REG_PERIODS:
        return REG_LEN * (period - 1)
    return REG_LEN * REG_PERIODS + OT_LEN * (period - REG_PERIODS - 1)


def period_end(period: int) -> float:
    return period_start(period) + period_length(period)


def elapsed_at(period: int, clock: float) -> float:
    return round(period_start(period) + period_length(period) - clock, 2)


_CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")


def parse_clock(s: str | None) -> float:
    """ISO-8601-ish duration ``PT11M36.00S`` -> 696.0 seconds. Unparseable -> 0.0."""
    if not s:
        return 0.0
    m = _CLOCK_RE.match(s.strip())
    if not m:
        return 0.0
    mins = int(m.group(1) or 0)
    secs = float(m.group(2) or 0.0)
    return round(mins * 60 + secs, 2)


def fmt_clock(seconds: float) -> str:
    """696.0 -> '11:36'; under a minute keeps tenths: 5.3 -> '0:05.3'."""
    if seconds < 60 and abs(seconds - round(seconds)) > 1e-6:
        return f"0:{seconds:04.1f}"
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def period_label(period: int) -> str:
    if period <= REG_PERIODS:
        return f"Q{period}"
    n = period - REG_PERIODS
    return "OT" if n == 1 else f"{n}OT"


# ------------------------------------------------------------------------- the event type

@dataclass(frozen=True, slots=True)
class Event:
    idx: int                      # position in the normalized list
    action_number: int
    order: int                    # orderNumber
    period: int
    clock: float                  # seconds remaining in period
    elapsed: float                # game seconds elapsed
    team_id: int | None
    team: str | None              # tricode
    side: str | None              # "home" | "away" | None
    player_id: int | None         # primary actor (None for team / admin events)
    action_type: str
    sub_type: str
    descriptor: str | None
    qualifiers: tuple[str, ...]
    description: str
    points: int
    score_home: int
    score_away: int
    # shots (field goals and free throws)
    is_fga: bool = False
    is_three: bool = False
    made: bool | None = None
    x: float | None = None
    y: float | None = None
    distance: float | None = None
    area: str | None = None
    # linked players
    assist_id: int | None = None
    steal_id: int | None = None
    block_id: int | None = None
    foul_drawn_id: int | None = None
    # categories
    is_ft: bool = False
    ft_num: int | None = None
    ft_of: int | None = None
    ft_retains_possession: bool = False   # technical / flagrant / take / away-from-play FT
    is_turnover: bool = False
    is_steal: bool = False                # the separate "steal" credit event
    is_block: bool = False                # the separate "block" credit event
    is_rebound: bool = False
    rebound_offensive: bool | None = None
    team_rebound: bool = False
    is_foul: bool = False
    foul_kind: str | None = None          # personal | offensive | technical | flagrant
    is_sub: bool = False
    sub_dir: str | None = None            # "in" | "out"
    is_jumpball: bool = False
    is_timeout: bool = False
    is_period_start: bool = False
    is_period_end: bool = False
    is_ejection: bool = False
    shot_action_number: int | None = None
    possession_raw: int | None = None     # feed's "possession" team id after the event
    # possession accounting (filled by _mark_possessions)
    ends_possession: bool = False
    poss_side: str | None = None          # offense whose possession ended here

    @property
    def margin_home(self) -> int:
        return self.score_home - self.score_away

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# ------------------------------------------------------------------------- parsing helpers

def _int(v: Any) -> int | None:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i


def _pid(v: Any) -> int | None:
    i = _int(v)
    return i if i else None


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_FT_RE = re.compile(r"(\d+)\s*of\s*(\d+)")
_RETAIN_WORDS = ("technical", "flagrant", "take", "away-from-play", "clear path")


def team_sides(box_payload: dict | None, pbp_payload: dict | None = None) -> dict[int, str]:
    """team_id -> 'home'/'away'. Uses the boxscore; returns {} if unavailable."""
    g = (box_payload or {}).get("game") or {}
    out: dict[int, str] = {}
    for key, side in (("homeTeam", HOME), ("awayTeam", AWAY)):
        tid = _int((g.get(key) or {}).get("teamId"))
        if tid:
            out[tid] = side
    return out


def _infer_sides(actions: list[dict]) -> dict[int, str]:
    """Without a boxscore: the team whose made shots move scoreHome is home."""
    prev_h = prev_a = 0
    votes: dict[int, list[int]] = {}
    for a in actions:
        h, w = _int(a.get("scoreHome")) or 0, _int(a.get("scoreAway")) or 0
        tid = _int(a.get("teamId"))
        if tid and (h != prev_h or w != prev_a):
            v = votes.setdefault(tid, [0, 0])
            v[0] += h != prev_h
            v[1] += w != prev_a
        prev_h, prev_a = h, w
    out: dict[int, str] = {}
    for tid, (hv, av) in votes.items():
        out[tid] = HOME if hv >= av else AWAY
    return out


def _actions(pbp_payload: dict) -> list[dict]:
    acts = ((pbp_payload or {}).get("game") or {}).get("actions") or []
    return sorted(acts, key=lambda a: (_int(a.get("orderNumber")) or 0, _int(a.get("actionNumber")) or 0))


# ------------------------------------------------------------------------- normalize

def normalize(pbp_payload: dict, box_payload: dict | None = None) -> list[Event]:
    """Raw cdn.nba.com play-by-play -> ordered list of ``Event``."""
    actions = _actions(pbp_payload)
    sides = team_sides(box_payload) or _infer_sides(actions)
    events: list[Event] = []
    prev_h = prev_a = 0
    for a in actions:
        atype = str(a.get("actionType") or "").lower()
        sub = str(a.get("subType") or "")
        sub_l = sub.lower()
        desc_l = str(a.get("descriptor") or "").lower()
        quals = tuple(str(q) for q in (a.get("qualifiers") or []))
        period = _int(a.get("period")) or 1
        clock = parse_clock(a.get("clock"))
        tid = _int(a.get("teamId"))
        side = sides.get(tid) if tid else None
        pid = _pid(a.get("personId"))
        result = str(a.get("shotResult") or "")
        made = None if not result else result.lower() == "made"

        is_fga = atype in ("2pt", "3pt")
        is_three = atype == "3pt"
        is_ft = atype == "freethrow"
        points = 0
        if made and is_fga:
            points = 3 if is_three else 2
        elif made and is_ft:
            points = 1

        # Score after event. The feed repeats the running score on every action; trust it but
        # never let it go backwards relative to our own accumulation of made shots.
        sh = _int(a.get("scoreHome"))
        sa = _int(a.get("scoreAway"))
        if sh is None or sa is None:
            sh, sa = prev_h, prev_a
        prev_h, prev_a = sh, sa

        ft_num = ft_of = None
        retains = False
        if is_ft:
            m = _FT_RE.search(sub_l)
            if m:
                ft_num, ft_of = int(m.group(1)), int(m.group(2))
            retains = any(w in sub_l or w in desc_l for w in _RETAIN_WORDS)

        foul_kind = None
        if atype == "foul":
            if "flagrant" in sub_l:
                foul_kind = "flagrant"
            elif "technical" in sub_l:
                foul_kind = "technical"
            elif "offensive" in sub_l:
                foul_kind = "offensive"
            else:
                foul_kind = "personal"

        is_reb = atype == "rebound"
        ev = Event(
            idx=len(events),
            action_number=_int(a.get("actionNumber")) or 0,
            order=_int(a.get("orderNumber")) or 0,
            period=period,
            clock=clock,
            elapsed=elapsed_at(period, clock),
            team_id=tid,
            team=a.get("teamTricode") or None,
            side=side,
            player_id=pid,
            action_type=atype,
            sub_type=sub,
            descriptor=a.get("descriptor") or None,
            qualifiers=quals,
            description=str(a.get("description") or ""),
            points=points,
            score_home=sh,
            score_away=sa,
            is_fga=is_fga,
            is_three=is_three,
            made=made if (is_fga or is_ft) else None,
            x=_float(a.get("x")) if is_fga else None,
            y=_float(a.get("y")) if is_fga else None,
            distance=_float(a.get("shotDistance")) if is_fga else None,
            area=a.get("area") if is_fga else None,
            assist_id=_pid(a.get("assistPersonId")),
            steal_id=_pid(a.get("stealPersonId")),
            block_id=_pid(a.get("blockPersonId")),
            foul_drawn_id=_pid(a.get("foulDrawnPersonId")),
            is_ft=is_ft,
            ft_num=ft_num,
            ft_of=ft_of,
            ft_retains_possession=retains,
            is_turnover=atype == "turnover",
            is_steal=atype == "steal",
            is_block=atype == "block",
            is_rebound=is_reb,
            rebound_offensive=(sub_l == "offensive") if is_reb else None,
            team_rebound=is_reb and pid is None,
            is_foul=atype == "foul",
            foul_kind=foul_kind,
            is_sub=atype == "substitution",
            sub_dir=sub_l if atype == "substitution" else None,
            is_jumpball=atype == "jumpball",
            is_timeout=atype == "timeout",
            is_period_start=atype == "period" and sub_l == "start",
            is_period_end=atype == "period" and sub_l == "end",
            is_ejection=atype == "ejection",
            shot_action_number=_int(a.get("shotActionNumber")),
            possession_raw=_pid(a.get("possession")),
        )
        events.append(ev)
    return _mark_possessions(events, sides)


# ------------------------------------------------------------------------- possessions

def _mark_possessions(events: list[Event], sides: dict[int, str]) -> list[Event]:
    """Flag possession-ending events (counting method, not the box-score estimate).

    A possession ends on:
      * a made field goal, unless it is an and-one (a non-technical "1 of 1" FT by the same team
        follows at the same clock before any other possession end) -- then the FT/rebound ends it;
      * the last free throw of a trip if made, unless the trip retains possession (technical,
        flagrant, transition take, away-from-play, clear path);
      * a defensive rebound of a missed field goal or of a missed final free throw
        (team defensive rebounds count, rebounds of non-final FTs do not);
      * a turnover (player or team);
      * the end of a period, if the offense did something since the last possession end
        (a shot, foul drawn, offensive rebound, heave...). Who owned it comes from the feed's
        ``possession`` field.
    """
    by_action = {e.action_number: e for e in events}
    out = list(events)

    def set_end(i: int, side: str | None) -> None:
        out[i] = replace(out[i], ends_possession=True, poss_side=side)

    n = len(events)
    for i, e in enumerate(events):
        if e.is_fga and e.made:
            if not _is_and_one(events, i):
                set_end(i, e.side)
        elif e.is_ft and e.made and not e.ft_retains_possession and e.ft_num == e.ft_of:
            set_end(i, e.side)
        elif e.is_rebound and e.rebound_offensive is False:
            shot = by_action.get(e.shot_action_number) if e.shot_action_number else None
            if shot is not None and shot.is_ft and (shot.ft_retains_possession or shot.ft_num != shot.ft_of):
                continue
            set_end(i, other(e.side))
        elif e.is_turnover:
            set_end(i, e.side)

    # Period ends.
    live = False
    last_poss_raw: int | None = None
    for i in range(n):
        e = out[i]
        if e.possession_raw:
            last_poss_raw = e.possession_raw
        if e.ends_possession:
            live = False
            continue
        if e.is_period_start:
            live = False
            continue
        if e.is_fga or e.is_ft or e.action_type == "heave" or (e.is_rebound and e.rebound_offensive) \
                or (e.is_foul and e.foul_kind in ("personal", "flagrant")) or e.is_jumpball:
            live = True
        if e.is_period_end and live:
            set_end(i, sides.get(last_poss_raw) if last_poss_raw else None)
            live = False
    return out


def _is_and_one(events: list[Event], i: int) -> bool:
    e = events[i]
    for j in range(i + 1, min(i + 25, len(events))):
        f = events[j]
        if f.period != e.period or f.clock != e.clock:
            return False
        if f.is_ft and f.side == e.side:
            return f.ft_of == 1 and not f.ft_retains_possession
        if f.is_fga or f.is_turnover or (f.is_rebound and not f.team_rebound):
            return False
    return False


# ------------------------------------------------------------------------- small utilities

def final_score(events: Iterable[Event]) -> tuple[int, int]:
    h = a = 0
    for e in events:
        h, a = e.score_home, e.score_away
    return h, a


def n_periods(events: list[Event]) -> int:
    return max((e.period for e in events), default=REG_PERIODS)


def game_seconds(events: list[Event]) -> float:
    return period_end(max(n_periods(events), REG_PERIODS))
