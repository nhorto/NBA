"""League percentiles for the stats pages (Cleaning the Glass / Basketball Index style).

A percentile complements the rank chip: same qualified pool, same direction (rank 1 = best;
lower is better for TOV%, DRtg and opponent points).

    pct = 100 * (players he beats + half of the others he ties) / (pool size - 1)

so the best value is 100, the worst 0, and tied values share one percentile (a pool of equal
values sits at 50). A pool of fewer than 2 gives no percentile (None).

  pct_values(values, higher_better)        {key: pct} from {key: value}
  player_pcts(conn, season, season_type)   {player_id: {stat: pct}} over the qualified pool of
                                           `stats.players.league_table` (its RANKED stats)
  team_pcts(conn, season, season_type)     {team: {key: pct}} over the league table
                                           (`stats.teams.RANKED` directions)
Both are cached per season with `stats.db.cached` (domain "box" where supported).
"""

from __future__ import annotations

import bisect
import inspect
import sqlite3
from typing import Any, Callable

from . import db as sdb
from . import seasons as S


def cached(key: Any, fn: Callable[[sqlite3.Connection], Any], conn) -> Any:
    """`stats.db.cached`, passing domain="box" when that parameter exists."""
    try:
        has_domain = "domain" in inspect.signature(sdb.cached).parameters
    except (TypeError, ValueError):
        has_domain = False
    if has_domain:
        return sdb.cached(key, fn, conn, domain="box")
    return sdb.cached(key, fn, conn)


def pct_values(values: dict, higher_better: bool = True) -> dict:
    """{key: percentile 0..100} for the non-None values (see the module docstring)."""
    vals = sorted(v for v in values.values() if v is not None)
    n = len(vals)
    if n < 2:
        return {k: None for k, v in values.items() if v is not None}
    out = {}
    for k, v in values.items():
        if v is None:
            continue
        lo, hi = bisect.bisect_left(vals, v), bisect.bisect_right(vals, v)
        below = lo if higher_better else n - hi              # values he beats
        out[k] = 100.0 * (below + (hi - lo - 1) / 2.0) / (n - 1)
    return out


def player_pcts(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR
                ) -> dict[int, dict[str, float | None]]:
    """{player_id: {stat: pct}} for the qualified players of the season (others: absent)."""
    from . import players as P

    def build(c):
        lt = P.league_table(c, season, season_type)
        qual = {pid: p for pid, p in lt["players"].items() if p.get("qualified")}
        out: dict[int, dict] = {pid: {} for pid in qual}
        for k in P.RANKED:
            for pid, v in pct_values({pid: p.get(k) for pid, p in qual.items()}).items():
                out[pid][k] = v
        return out
    return cached(("stats.pct.players", season, season_type), build, conn) or {}


def team_pcts(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR
              ) -> dict[str, dict[str, float | None]]:
    """{team: {key: pct}} over the league table, directions from `stats.teams.RANKED`."""
    from . import teams as ST

    def build(c):
        rows = ST.league(c, season, season_type)["rows"]
        out: dict[str, dict] = {r["team"]: {} for r in rows}
        for k, hb in ST.RANKED.items():
            for t, v in pct_values({r["team"]: r.get(k) for r in rows}, hb).items():
                out[t][k] = v
        return out
    return cached(("stats.pct.teams", season, season_type), build, conn) or {}
