"""Research panels for the Tonight page (`web/players_view.py`): "Teammate out" with/without
splits and the game's market context (spread, total, implied team totals, pace ranks).

Pure functions on a read-only `sqlite3.Connection`; heavy parts are cached per database file
with `stats.db.cached`. Information only: nothing here says what to do with the numbers.

**Teammate out** (`with_without`). For a team, an absent player A (listed Out or Doubtful
tonight) and a teammate T, over this season and last (Regular Season and Playoffs, games
before the day, games T played for the team, `player_games.min > 0`):
  * "with" = games in which A also played for the team (min > 0);
  * "without" = games in which A had no row or 0 minutes for the team.
Only seasons in which A played at least one game for the team count, so a new arrival's (or a
rookie's) "without" sample is not every game his team played before he got there. Per game
means of minutes, usage % (`player_games_adv.usg_pct`), points, rebounds, assists and DK points
(`proj.dfs.rules.dk_points`), the game counts and the change (without - with). A "without"
sample under ``SMALL_N`` games is flagged ``small``.

**Market context** (`market_context`). The Brief's line: `predict.market_line` (newest
pre-tip `odds_snapshots` batch, else the closing line). Implied team totals are
total / 2 -/+ home spread / 2 (the favourite's is the bigger one). Pace rank from the season's
league table (`stats.teams.league`; 1 = fastest), the previous season when this one has no games.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Iterable, Mapping

from . import db as sdb
from . import seasons as S

SMALL_N = 3                      # "without" samples under this many games are a small sample
SEASON_TYPES = (S.REGULAR, S.PLAYOFFS)
STATS = ("min", "usg", "pts", "reb", "ast", "dk")
LABELS = {"min": "Min", "usg": "Usg%", "pts": "Pts", "reb": "Reb", "ast": "Ast", "dk": "DK"}


def window(day: date) -> tuple[str, str]:
    """(this season, last season) for a day: ('2026-27', '2025-26')."""
    cur = S.season_for(day)
    return cur, S.from_start_year(S.start_year(cur) - 1)


# ------------------------------------------------------------------ teammate out

_TEAM_SQL = """
SELECT pg.game_id, pg.player_id, pg.season, pg.min, pg.pts, pg.fg3m, pg.reb, pg.ast, pg.stl,
       pg.blk, pg.tov, a.usg_pct
FROM player_games pg
LEFT JOIN player_games_adv a ON a.game_id=pg.game_id AND a.player_id=pg.player_id
WHERE pg.team=? AND pg.season IN ({s}) AND pg.season_type IN ({t}) AND pg.game_date<?
"""


def compute_team_games(conn: sqlite3.Connection, team: str, seasons: Iterable[str],
                       before: str) -> dict:
    """{player_id: {game_id: line}} and {game_id: season} for a team's games before `before`.
    A line has min usg pts reb ast dk (usg None without an advanced row)."""
    from ..proj.dfs.rules import dk_points
    seasons = tuple(seasons)
    sql = _TEAM_SQL.format(s=",".join("?" * len(seasons)), t=",".join("?" * len(SEASON_TYPES)))
    by_player: dict[int, dict[str, dict]] = {}
    season_of: dict[str, str] = {}
    try:
        rows = conn.execute(sql, (team, *seasons, *SEASON_TYPES, before)).fetchall()
    except sqlite3.Error:
        rows = []
    for r in rows:
        gid, pid = r[0], int(r[1])
        season_of[gid] = r[2]
        line = {"pts": r[4], "fg3m": r[5], "reb": r[6], "ast": r[7], "stl": r[8], "blk": r[9],
                "tov": r[10]}
        by_player.setdefault(pid, {})[gid] = {
            "min": float(r[3] or 0.0), "usg": None if r[11] is None else float(r[11]),
            "pts": float(r[4] or 0), "reb": float(r[6] or 0), "ast": float(r[7] or 0),
            "dk": dk_points(line)}
    return {"by_player": by_player, "season_of": season_of}


def team_games(conn: sqlite3.Connection, team: str, seasons: Iterable[str], before: str) -> dict:
    seasons = tuple(seasons)
    return sdb.cached(("tonight.team_games", team, seasons, before),
                      lambda c: compute_team_games(c, team, seasons, before), conn, domain="box")


def _mean(lines: list[dict], key: str) -> float | None:
    vals = [x[key] for x in lines if x.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _arm(lines: list[dict]) -> dict:
    out = {"n": len(lines)}
    for k in STATS:
        out[k] = _mean(lines, k) if lines else None
    return out


def split(data: Mapping, absent: int, mate: int) -> dict | None:
    """With/without arms for one (absent, teammate) pair from `team_games` data; None when
    the absent player has no game for the team in the window or the teammate never played."""
    bp, season_of = data["by_player"], data["season_of"]
    a_played = {g for g, x in (bp.get(absent) or {}).items() if x["min"] > 0}
    if not a_played:
        return None
    seasons = {season_of[g] for g in a_played}
    lines = [(g, x) for g, x in (bp.get(mate) or {}).items()
             if x["min"] > 0 and season_of.get(g) in seasons]
    if not lines:
        return None
    w = _arm([x for g, x in lines if g in a_played])
    wo = _arm([x for g, x in lines if g not in a_played])
    delta = {k: (wo[k] - w[k]) if w[k] is not None and wo[k] is not None else None
             for k in STATS}
    return {"with": w, "without": wo, "delta": delta, "small": wo["n"] < SMALL_N,
            "seasons": sorted(seasons, reverse=True)}


def with_without(conn: sqlite3.Connection, team: str, absent: int, mates: Iterable[int],
                 day: date) -> dict:
    """{absent, seasons, a_games, rows: [{player_id, with, without, delta, small}]} for one
    absent player and the teammates (in the order given). Cached per (team, absent player,
    season set, day) for the database file."""
    seasons = window(day)
    mates = tuple(int(m) for m in mates if int(m) != int(absent))
    key = ("tonight.with_without", team, int(absent), seasons, day.isoformat(), mates)

    def build(c: sqlite3.Connection) -> dict:
        data = team_games(c, team, seasons, day.isoformat())
        a_lines = data["by_player"].get(int(absent)) or {}
        a_games = sum(1 for x in a_lines.values() if x["min"] > 0)
        rows = []
        used: set[str] = set()
        for m in mates:
            sp = split(data, int(absent), m)
            if sp is None:
                continue
            used.update(sp["seasons"])
            rows.append({"player_id": m, **sp})
        return {"absent": int(absent), "team": team, "seasons": sorted(used, reverse=True),
                "window": seasons, "a_games": a_games, "rows": rows}
    return sdb.cached(key, build, conn, domain="box")


# ------------------------------------------------------------------ market context

def implied_totals(total: float | None, home_spread: float | None
                   ) -> tuple[float | None, float | None]:
    """(home, away) implied points: total/2 - spread/2 and total/2 + spread/2, where the home
    spread is negative when the home team is favoured. (None, None) without a total; the
    spread defaults to 0 (a pick'em) when only the total is known."""
    if total is None:
        return None, None
    sp = float(home_spread or 0.0)
    return float(total) / 2.0 - sp / 2.0, float(total) / 2.0 + sp / 2.0


def pace_ranks(conn: sqlite3.Connection, day: date) -> dict:
    """{season, by_team: {team: {rank, pace, gp}}, n}: this season's league pace ranks, or the
    previous season's when this one has no games yet."""
    from . import teams as T
    cur, prev = window(day)
    for season in (cur, prev):
        try:
            lg = T.league(conn, season)
        except sqlite3.Error:
            continue
        if lg["rows"]:
            return {"season": season, "n": lg["n"],
                    "by_team": {r["team"]: {"rank": r["rk"].get("pace"), "pace": r["pace"],
                                            "gp": r["gp"]} for r in lg["rows"]}}
    return {"season": None, "n": 0, "by_team": {}}


def market_context(conn: sqlite3.Connection, game_id: str, home: str, away: str,
                   tipoff_utc: str | None, pace: Mapping | None = None) -> dict | None:
    """The game's market line with implied team totals and both teams' pace ranks; None when
    there is no line and no pace."""
    from .. import predict
    try:
        line = predict.market_line(conn, game_id, tipoff_utc)
    except (sqlite3.Error, KeyError, IndexError):
        line = None
    pace = pace or {}
    pt = pace.get("by_team") or {}
    if line is None and not pt:
        return None
    sp = (line or {}).get("home_spread")
    tot = (line or {}).get("total")
    h_imp, a_imp = implied_totals(tot, sp)
    fav = None
    if sp is not None and sp != 0:
        fav = home if sp < 0 else away
    return {
        "has_line": line is not None and (sp is not None or tot is not None),
        "home_spread": sp, "away_spread": None if sp is None else -sp, "total": tot,
        "favourite": fav, "fav_spread": None if sp is None else -abs(sp),
        "home_implied": h_imp, "away_implied": a_imp,
        "provider": (line or {}).get("provider"), "source": (line or {}).get("source"),
        "captured_at": (line or {}).get("captured_at"), "kind": (line or {}).get("kind"),
        "home_pace": pt.get(home), "away_pace": pt.get(away),
        "pace_season": pace.get("season"), "pace_n": pace.get("n") or 30,
    }
