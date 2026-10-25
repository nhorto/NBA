"""Lineup explorer (docs/STATS_SITE.md, /lineups): on/off for any set of players, WOWY and the
best / worst 2-, 3- and 5-man combos of one team-season, from `stints`.

  load(conn, team, season, season_type)   the team-season's stints folded into 5-man units
                                          (+ every 2- and 3-man combo), cached once in memory
  select(data, on, off)                   totals with every `on` player on the floor and every
                                          `off` player off it, vs the team overall
  wowy(data, pid)                         his on / off, and per teammate: both on, he without
                                          the teammate, the teammate without him
  combos(data, size, min_poss, top)       best and worst units of 2 / 3 / 5 by net rating

The team's side of a stint comes from `team_games.is_home` (home5 or away5). Possessions of a
stint = (team poss + opp poss) / 2; ORtg / DRtg are per 100 of each side's own possessions
(as stats.teams.stints). Minutes = (end_sec - start_sec) / 60. After `load`, every function is
pure Python over a few hundred units, so a warm request costs about a millisecond.
"""

from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from typing import Iterable, Sequence

from . import db as sdb
from . import seasons as S

SIZES = (2, 3, 5)
TOP = 10
WOWY_MIN_POSS = 1.0           # a split needs at least this many possessions to get ratings


def _f(x) -> float:
    return 0.0 if x is None else float(x)


def _new() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0]          # pts for, pts against, poss for, poss against, sec


def _add(acc: list[float], v: Sequence[float]) -> None:
    for i in range(5):
        acc[i] += v[i]


def rtg(acc: Sequence[float] | None) -> dict:
    """{poss, min, pf, pa, ortg, drtg, net} of an accumulator (ratings None without poss)."""
    a = acc or _new()
    pf, pa, possf, possa, sec = a
    o = 100.0 * pf / possf if possf else None
    d = 100.0 * pa / possa if possa else None
    return {"poss": (possf + possa) / 2.0, "min": sec / 60.0, "pf": pf, "pa": pa,
            "ortg": o, "drtg": d, "net": (o - d) if o is not None and d is not None else None}


def compute(conn: sqlite3.Connection, team: str, season: str, season_type: str) -> dict:
    side: dict[str, bool] = {}
    try:
        for r in conn.execute("SELECT game_id, is_home FROM team_games WHERE team=? AND season=?"
                              " AND season_type=?", (team, season, season_type)):
            side[r[0]] = bool(r[1])
    except sqlite3.Error:
        pass
    units: dict[tuple[int, ...], list[float]] = {}
    total = _new()
    covered: set[str] = set()
    ids = list(side)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        try:
            rows = conn.execute(
                "SELECT game_id, start_sec, end_sec, home5, away5, home_pts, away_pts, "
                f"home_poss, away_poss FROM stints WHERE game_id IN ({','.join('?' * len(chunk))})",
                chunk).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            gid, s0, s1, h5, a5, hp, ap, hposs, aposs = tuple(r)
            home = side[gid]
            try:
                five = tuple(sorted(int(p) for p in json.loads(h5 if home else a5)))
            except (TypeError, ValueError):
                continue
            covered.add(gid)
            if home:
                v = (_f(hp), _f(ap), _f(hposs), _f(aposs), max(0.0, _f(s1) - _f(s0)))
            else:
                v = (_f(ap), _f(hp), _f(aposs), _f(hposs), max(0.0, _f(s1) - _f(s0)))
            _add(total, v)
            _add(units.setdefault(five, _new()), v)
    # every 2- and 3-man combo inside the units (5-man = the units themselves)
    combos: dict[int, dict[tuple[int, ...], list[float]]] = {2: {}, 3: {}, 5: {}}
    on_sec: dict[int, float] = {}
    for five, acc in units.items():
        for p in five:
            on_sec[p] = on_sec.get(p, 0.0) + acc[4]
        for size in (2, 3):
            for c in combinations(five, size):
                _add(combos[size].setdefault(c, _new()), acc)
        if len(five) == 5:
            combos[5][five] = acc
    from .players import names as _names
    nm = _names(conn, on_sec) if on_sec else {}
    players = [{"player_id": p, "name": nm.get(p) or f"#{p}", "min": s / 60.0}
               for p, s in sorted(on_sec.items(), key=lambda kv: -kv[1])]
    return {"team": team, "season": season, "season_type": season_type,
            "games": len(covered), "team_gp": len(side),
            "units": [(frozenset(k), k, v) for k, v in units.items()],
            "total": total, "combos": combos, "players": players,
            "names": {p["player_id"]: p["name"] for p in players}}


def load(conn: sqlite3.Connection, team: str, season: str, season_type: str = S.REGULAR
         ) -> dict:
    """The team-season's units (cached in memory until courtside.db changes)."""
    return sdb.cached(("lineups.units", team, season, season_type),
                      lambda c: compute(c, team, season, season_type), conn, domain="box")


def _acc(data: dict, on: Iterable[int], off: Iterable[int]) -> list[float]:
    on_s, off_s = frozenset(on), frozenset(off)
    acc = _new()
    for fs, _, v in data["units"]:
        if on_s <= fs and not (off_s & fs):
            _add(acc, v)
    return acc


def select(data: dict, on: Sequence[int], off: Sequence[int] = ()) -> dict:
    """{sel, team, rest}: the selection's totals, the team overall and the team without the
    selection (team minus sel). Overlapping on/off ids are the caller's job to reject."""
    acc = _acc(data, on, off)
    rest = [t - a for t, a in zip(data["total"], acc)]
    t = rtg(data["total"])
    sel = rtg(acc)
    sel["share"] = (sel["poss"] / t["poss"]) if t["poss"] else None
    return {"sel": sel, "team": t, "rest": rtg(rest),
            "diff": (sel["net"] - t["net"]) if sel["net"] is not None and t["net"] is not None
            else None}


def _r(acc) -> dict | None:
    r = rtg(acc)
    return r if r["poss"] >= WOWY_MIN_POSS else {**r, "ortg": None, "drtg": None, "net": None}


def wowy(data: dict, pid: int) -> dict:
    """{on, off, diff, mates: [{player_id, name, min_together, both, a_only, b_only}]}:
    a_only = he on without the teammate, b_only = the teammate on without him. Mates sorted by
    possessions together."""
    on = _r(_acc(data, (pid,), ()))
    off = _r(_acc(data, (), (pid,)))
    mates = []
    for p in data["players"]:
        m = p["player_id"]
        if m == pid:
            continue
        both = _acc(data, (pid, m), ())
        if both[2] + both[3] <= 0:
            continue
        mates.append({"player_id": m, "name": p["name"], "both": _r(both),
                      "a_only": _r(_acc(data, (pid,), (m,))),
                      "b_only": _r(_acc(data, (m,), (pid,)))})
    mates.sort(key=lambda r: -r["both"]["poss"])
    diff = (on["net"] - off["net"]) if on["net"] is not None and off["net"] is not None else None
    return {"on": on, "off": off, "diff": diff, "mates": mates}


def combos(data: dict, size: int, min_poss: float, top: int = TOP) -> dict:
    """{best, worst, n}: units of `size` with >= min_poss possessions by net rating (worst =
    the bottom `top`, worst first). With fewer than 2 * top units they are split in half, so
    no unit appears in both lists."""
    if size not in SIZES:
        raise ValueError("size must be 2, 3 or 5")
    names = data["names"]
    rows = []
    for ids, acc in data["combos"][size].items():
        r = rtg(acc)
        if r["poss"] < min_poss or r["net"] is None:
            continue
        r["ids"] = list(ids)
        r["names"] = [names.get(p) or f"#{p}" for p in ids]
        rows.append(r)
    rows.sort(key=lambda r: (-r["net"], -r["poss"]))
    k = min(top, (len(rows) + 1) // 2)          # few units: split them between the lists
    best = rows[:k]
    worst = sorted(rows[k:][-top:], key=lambda r: (r["net"], -r["poss"]))
    return {"best": best, "worst": worst, "n": len(rows)}
