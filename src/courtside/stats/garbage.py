"""Garbage time on play-by-play stints (Cleaning the Glass definition) and the stint-based
numbers that can exclude it: team ratings, on/off, 5-man lineups, a player's on-court net.

Definition (cleaningtheglass.com/stats/guide/garbage_time), applied to stints:
  * 4th quarter, and the margin is >= 25 with 12:00-9:01 left, >= 20 with 9:00-6:01 left,
    >= 10 with 6:00 or less left;
  * two or fewer starters on the floor, both teams combined (starters = the two lineups of the
    game's first stint);
  * the game never leaves those conditions again: garbage time is the final run of stints
    that all start in garbage-time conditions. A comeback (or overtime) resets the clock.
A stint is judged at its start: the margin is the score after all earlier stints of the game
(home_pts / away_pts accumulated in stint order); time left = 2880 - start_sec.

Box scores cannot be split by time without play-by-play, so box-score numbers (points per
game, shooting, ...) always include garbage time; only stint-based numbers can exclude it.

  margins(stints)                       margin (home - away) at the start of each stint
  threshold(period, start_sec)          margin needed at that moment (None outside the window)
  is_garbage_start(...)                 one stint's start conditions
  mark_game(stints)                     {stint_no} in garbage time for one game
  season_pass(conn, season, type)       cached season pass (see its docstring)
  team_ratings(conn, season, type, exclude)   ORtg / DRtg / Net / Pace with ranks + percentiles
  team_stints(conn, team, season, type, exclude)  on/off + lineups (stats.teams.stints shape)
  player_onoff(conn, pid, season, type, exclude)  a player's on-court / off-court ratings
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Iterable

from . import leaders as L
from . import seasons as S
from .percentiles import cached, pct_values

REG_END = 2880.0                    # elapsed seconds at the end of regulation
WINDOWS = ((360.0, 10), (540.0, 20), (720.0, 25))     # (time left <= sec, margin >= pts)
MAX_STARTERS = 2
DEFINITION = ("Garbage time (Cleaning the Glass): 4th quarter, margin ≥ 25 with 12–9 "
              "min left, ≥ 20 with 9–6 min, ≥ 10 in the last 6 min, two or fewer "
              "starters on the floor, and the game never gets back inside those lines.")
RATING_KEYS = {"ortg": True, "drtg": False, "net": True, "pace": True}   # higher is better


def _f(x) -> float:
    return 0.0 if x is None else float(x)


def threshold(period: int, start_sec: float) -> int | None:
    """Margin needed for garbage time at this moment; None outside the 4th quarter window."""
    if period != 4:
        return None
    left = REG_END - _f(start_sec)
    for sec, pts in WINDOWS:
        if left <= sec + 1e-9:
            return pts
    return None


def is_garbage_start(period: int, start_sec: float, margin: float,
                     starters_on: int | None = None) -> bool:
    """True when a stint starts in garbage-time conditions. `starters_on` = starters of both
    teams on the floor (None = unknown: the margin rule alone)."""
    need = threshold(period, start_sec)
    if need is None or abs(margin) < need:
        return False
    return starters_on is None or starters_on <= MAX_STARTERS


def margins(stints: Iterable[dict]) -> list[float]:
    """Home-minus-away margin at the start of each stint (stints in stint_no order)."""
    out, m = [], 0.0
    for st in stints:
        out.append(m)
        m += _f(st["home_pts"]) - _f(st["away_pts"])
    return out


def _five(x) -> list[int]:
    if isinstance(x, (list, tuple)):
        return [int(p) for p in x]
    return [int(p) for p in json.loads(x or "[]")]


def mark_game(stints: list[dict]) -> set[int]:
    """{stint_no} of one game's stints that are in garbage time (see the module docstring)."""
    rows = sorted(stints, key=lambda r: r["stint_no"])
    if not rows:
        return set()
    first = rows[0]
    starters = None
    if int(first["period"]) == 1 and _f(first["start_sec"]) <= 0.0:
        starters = set(_five(first["home5"])) | set(_five(first["away5"]))
    flags = []
    for r, m in zip(rows, margins(rows)):
        on = None
        if starters is not None:
            on = len(starters & (set(_five(r["home5"])) | set(_five(r["away5"]))))
        flags.append(is_garbage_start(int(r["period"]), r["start_sec"], m, on))
    out = set()
    for r, g in zip(reversed(rows), reversed(flags)):
        if not g:
            break
        out.add(int(r["stint_no"]))
    return out


# ------------------------------------------------------------------ the season pass
_SQL = """
SELECT s.game_id, s.stint_no, s.period, s.start_sec, s.end_sec, s.home5, s.away5,
       s.home_pts, s.away_pts, s.home_poss, s.away_poss, t.team AS home, t.opp AS away
FROM team_games t JOIN stints s ON s.game_id = t.game_id
WHERE t.season=? AND t.season_type=? AND t.is_home=1
ORDER BY s.game_id, s.stint_no
"""


def _acc() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0]        # pts for, against, poss for, against, seconds


def compute_season(conn: sqlite3.Connection, season: str, season_type: str) -> dict:
    teams: dict[str, dict[str, list]] = defaultdict(lambda: {"all": _acc(), "xgt": _acc()})
    players: dict[int, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"all": _acc(), "xgt": _acc()}))
    garbage: dict[str, frozenset] = {}
    try:
        rows = [dict(r) for r in conn.execute(_SQL, (season, season_type))]
    except sqlite3.Error:
        rows = []
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_game[r["game_id"]].append(r)
    for gid, sts in by_game.items():
        for r in sts:
            r["home5"], r["away5"] = _five(r["home5"]), _five(r["away5"])
        g = mark_game(sts)
        if g:
            garbage[gid] = frozenset(g)
        for r in sts:
            sec = max(0.0, _f(r["end_sec"]) - _f(r["start_sec"]))
            hp, ap, hq, aq = (_f(r["home_pts"]), _f(r["away_pts"]), _f(r["home_poss"]),
                              _f(r["away_poss"]))
            keys = ("all",) if r["stint_no"] in g else ("all", "xgt")
            for team, five, v in ((r["home"], r["home5"], (hp, ap, hq, aq, sec)),
                                  (r["away"], r["away5"], (ap, hp, aq, hq, sec))):
                for key in keys:
                    a = teams[team][key]
                    for k in range(5):
                        a[k] += v[k]
                    for pid in five:
                        b = players[pid][team][key]
                        for k in range(5):
                            b[k] += v[k]
    return {"games": len(by_game), "garbage": garbage, "teams": dict(teams),
            "players": {pid: dict(t) for pid, t in players.items()}}


def season_pass(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """{games (with stints), garbage: {game_id: frozenset(stint_no)}, teams: {team: {"all" |
    "xgt": [pts for, pts against, poss for, poss against, seconds]}}, players: {player_id:
    {team: {"all" | "xgt": [...]}}}}; "xgt" excludes garbage time. Cached per season."""
    return cached(("stats.gt.season", season, season_type),
                  lambda c: compute_season(c, season, season_type), conn)


def _rtg(a: list[float]) -> dict:
    o = (100.0 * a[0] / a[2]) if a[2] else None
    d = (100.0 * a[1] / a[3]) if a[3] else None
    poss = (a[2] + a[3]) / 2.0
    return {"ortg": o, "drtg": d, "net": (o - d) if o is not None and d is not None else None,
            "pace": (REG_END * poss / a[4]) if a[4] else None, "poss": poss,
            "min": a[4] / 60.0}


# ------------------------------------------------------------------ team ratings
def team_ratings(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR,
                 exclude: bool = True) -> dict:
    """Stint-based ratings of every team: {games, by_team: {team: {ortg, drtg, net, pace, poss,
    gt_share (share of his possessions in garbage time), rk: {key: rank}, pct: {key: pct}}},
    n}. Possession-weighted (100 x points / possessions); pace = possessions per 48 minutes.
    exclude=True drops garbage-time stints."""
    sp = season_pass(conn, season, season_type)
    key = "xgt" if exclude else "all"
    by = {}
    for team, acc in sp["teams"].items():
        r = _rtg(acc[key])
        tot = (acc["all"][2] + acc["all"][3]) / 2.0
        r["gt_share"] = (1.0 - (acc["xgt"][2] + acc["xgt"][3]) / 2.0 / tot) if tot else None
        by[team] = r
    for k, hb in RATING_KEYS.items():
        vals = {t: r[k] for t, r in by.items()}
        rk, pc = L.rank_values(vals, hb), pct_values(vals, hb)
        for t, r in by.items():
            r.setdefault("rk", {})[k] = rk.get(t)
            r.setdefault("pct", {})[k] = pc.get(t)
    return {"games": sp["games"], "by_team": by, "n": len(by)}


def team_stints(conn: sqlite3.Connection, team: str, season: str,
                season_type: str = S.REGULAR, exclude: bool = False) -> dict:
    """On/off and 5-man lineups (the `stats.teams.stints` dict); exclude=True drops the
    garbage-time stints. Cached per team-season."""
    from . import teams as ST

    if not exclude:
        return ST.stints(conn, team, season, season_type)

    def build(c):
        skip = season_pass(c, season, season_type)["garbage"]
        return ST.compute_stints(c, team, season, season_type, skip=skip)
    return cached(("stats.stints.xgt", team, season, season_type), build, conn)


# ------------------------------------------------------------------ player on/off
ONOFF_MIN_POSS = 100.0               # on-court possessions to enter the percentile pool


def _pool(conn: sqlite3.Connection, season: str, season_type: str, exclude: bool) -> dict:
    """{player_id: on-court net} of the qualified players (players.league_table) with at least
    ONOFF_MIN_POSS on-court possessions, plus ranks and percentiles. Cached."""
    from . import players as P

    def build(c):
        sp = season_pass(c, season, season_type)
        lt = P.league_table(c, season, season_type)["players"]
        key = "xgt" if exclude else "all"
        vals = {}
        for pid, by_team in sp["players"].items():
            if not (lt.get(pid) or {}).get("qualified"):
                continue
            a = _acc()
            for acc in by_team.values():
                for k in range(5):
                    a[k] += acc[key][k]
            if (a[2] + a[3]) / 2.0 >= ONOFF_MIN_POSS:
                vals[pid] = _rtg(a)["net"]
        return {"net": vals, "rk": L.rank_values(vals, True), "pct": pct_values(vals, True),
                "n": sum(1 for v in vals.values() if v is not None)}
    return cached(("stats.gt.pool", season, season_type, bool(exclude)), build, conn)


def player_onoff(conn: sqlite3.Connection, pid: int, season: str,
                 season_type: str = S.REGULAR, exclude: bool = False) -> dict:
    """{games (season stints), teams: [{team, on: ratings, off: ratings, diff}], on (all his
    teams combined), rk / pct / n (on-court net among qualified players with 100+ on-court
    possessions)}. Off = his team's stints without him (all its games, as on the team page)."""
    sp = season_pass(conn, season, season_type)
    key = "xgt" if exclude else "all"
    mine = sp["players"].get(int(pid)) or {}
    rows, tot = [], _acc()
    for team, acc in sorted(mine.items(), key=lambda kv: -kv[1]["all"][4]):
        a = acc[key]
        t = sp["teams"].get(team, {}).get(key) or _acc()
        off = [t[k] - a[k] for k in range(5)]
        on_r = _rtg(a)
        off_r = _rtg(off) if (off[2] + off[3]) / 2.0 >= 1 else None
        diff = (on_r["net"] - off_r["net"]) if (off_r and on_r["net"] is not None
                                                 and off_r["net"] is not None) else None
        rows.append({"team": team, "on": on_r, "off": off_r, "diff": diff})
        for k in range(5):
            tot[k] += a[k]
    pool = _pool(conn, season, season_type, exclude)
    return {"games": sp["games"], "teams": rows, "on": _rtg(tot) if rows else None,
            "rk": pool["rk"].get(int(pid)), "pct": pool["pct"].get(int(pid)), "n": pool["n"],
            "min_poss": ONOFF_MIN_POSS}
