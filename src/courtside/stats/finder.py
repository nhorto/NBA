"""Stat finder (docs/STATS_SITE.md, /finder): game-level search and streak / frequency screens
over the 3-season window, like Stathead or the FantasyLabs trends tool. Information only.

  game_search(conn, conds, ctx, window, sort=, group=)
      player games (min > 0) matching every condition: [{row}], count, capped at CAP rows.
      group=True: one row per player (games, last date, averages) instead.
  streaks(conn, screen, ctx, window)
      players reaching `stat op thr` in at least k of their last n games (min games given).

Safety. SQL is assembled ONLY from the whitelists below (STATS: key -> SQL expression, OPS: key
-> operator, CONTEXT: fixed fragments); every user value is a bound parameter, never text in the
SQL. Unknown stat / operator / value -> FinderError (the page drops it with a notice). Results
are capped at CAP rows (LIMIT CAP + 1, so the page can say "first 500"), and a progress handler
interrupts any query running longer than TIMEOUT_S.

DK points: `proj.dfs.rules.dk_points` registered as the SQL function cs_dk (not reimplemented).
Rest days, DK salary and real ownership come from cached Python maps registered as cs_rest /
cs_sal / cs_own (the same definitions as the player page: rest = days off since his team's
previous game of the same season, salary = latest classic capture of the day, own = mean real
%drafted over that day's classic contests).
"""

from __future__ import annotations

import math
import sqlite3
import time
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from .. import teams as _teams
from . import db as sdb
from . import seasons as S

CAP = 500
TIMEOUT_S = 4.0
MAX_CONDS = 6
MAX_ABS = 100_000.0
CODES = frozenset(t.tricode for t in _teams.TEAMS)
CLASSIC = 70

# key -> (label, SQL expression on player_games pg, kind, percent input)
# kind: "int" / "num" (1 decimal) / "pct" / "dollars"
STATS: dict[str, tuple[str, str, str, bool]] = {
    "pts": ("PTS", "pg.pts", "int", False),
    "reb": ("REB", "pg.reb", "int", False),
    "ast": ("AST", "pg.ast", "int", False),
    "fg3m": ("3PM", "pg.fg3m", "int", False),
    "stl": ("STL", "pg.stl", "int", False),
    "blk": ("BLK", "pg.blk", "int", False),
    "tov": ("TOV", "pg.tov", "int", False),
    "min": ("MIN", "pg.min", "num", False),
    "dk": ("DK pts", "cs_dk(pg.pts, pg.fg3m, pg.reb, pg.ast, pg.stl, pg.blk, pg.tov)", "num",
           False),
    "pra": ("PTS+REB+AST", "(IFNULL(pg.pts,0) + IFNULL(pg.reb,0) + IFNULL(pg.ast,0))", "int",
            False),
    "stocks": ("STL+BLK", "(IFNULL(pg.stl,0) + IFNULL(pg.blk,0))", "int", False),
    "dd": ("10+ categories", "cs_ddc(pg.pts, pg.reb, pg.ast, pg.stl, pg.blk)", "int", False),
    "fg_pct": ("FG%", "(CAST(pg.fgm AS REAL) / NULLIF(pg.fga, 0))", "pct", True),
    "fg3_pct": ("3P%", "(CAST(pg.fg3m AS REAL) / NULLIF(pg.fg3a, 0))", "pct", True),
    "fga": ("FGA", "pg.fga", "int", False),
    "fg3a": ("3PA", "pg.fg3a", "int", False),
    "fta": ("FTA", "pg.fta", "int", False),
    "oreb": ("OREB", "pg.oreb", "int", False),
    "pm": ("+/-", "pg.plus_minus", "int", False),
    "salary": ("DK salary", "cs_sal(pg.player_id, pg.game_date)", "dollars", False),
    "own": ("Real own%", "cs_own(pg.player_id, pg.game_date)", "num", False),
}
# the stat dropdown (salary / own have their own range inputs)
COND_STATS = tuple(k for k in STATS if k not in ("salary", "own"))
STREAK_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "stocks", "pra", "dk", "dd", "min",
                "tov", "fg_pct")
STAT_TIPS = {"dd": "How many of PTS / REB / AST / STL / BLK reached 10: 2 = a double-double, "
                   "3 = a triple-double",
             "fg_pct": "Enter a percentage (50 = 50%); games without a shot never match",
             "fg3_pct": "Enter a percentage (40 = 40%); games without a 3PA never match"}
OPS: dict[str, tuple[str, str]] = {"ge": (">=", "≥"), "le": ("<=", "≤"), "gt": (">", ">"),
                                   "lt": ("<", "<"), "eq": ("=", "=")}
SORTS = ("date",) + COND_STATS + ("salary", "own")
TYPES = {"all": None, "regular": S.REGULAR, "playoffs": S.PLAYOFFS, "playin": S.PLAYIN}
REST = {"0": ("= 0", "Back-to-back"), "1": ("= 1", "1 day rest"), "2": (">= 2", "2+ days rest")}

# context filter -> {accepted value: (SQL fragment, bound params)}
_STARTED = ("EXISTS (SELECT 1 FROM player_starts ps WHERE ps.game_id = pg.game_id "
            "AND ps.player_id = pg.player_id)")
_HAS_STARTS = "EXISTS (SELECT 1 FROM player_starts ps2 WHERE ps2.game_id = pg.game_id)"
CONTEXT: dict[str, dict[str, tuple[str, tuple]]] = {
    "ha": {"home": ("tg.is_home = 1", ()), "away": ("tg.is_home = 0", ())},
    "wl": {"w": ("tg.wl = 'W'", ()), "l": ("tg.wl = 'L'", ())},
    "role": {"starter": (_STARTED, ()), "bench": (f"NOT {_STARTED} AND {_HAS_STARTS}", ())},
    "rest": {k: (f"cs_rest(pg.team, pg.season, pg.game_date) {v[0]}", ())
             for k, v in REST.items()},
}


class FinderError(ValueError):
    """A stat / operator / value outside the whitelist."""


@dataclass(frozen=True)
class Cond:
    stat: str
    op: str
    value: float

    @classmethod
    def make(cls, stat: str, op: str, value: Any) -> "Cond":
        if stat not in STATS:
            raise FinderError(f"unknown stat {str(stat)[:20]!r}")
        if op not in OPS:
            raise FinderError(f"unknown operator {str(op)[:10]!r}")
        v = number(value)
        if v is None:
            raise FinderError(f"not a number: {str(value)[:20]!r}")
        return cls(stat, op, v)

    def sql(self) -> tuple[str, tuple]:
        expr = STATS[self.stat][1]
        v = self.value / 100.0 if STATS[self.stat][3] else self.value
        return f"{expr} {OPS[self.op][0]} ?", (v,)

    def label(self) -> str:
        lab, _, kind, pct_in = STATS[self.stat]
        if kind == "dollars":
            val = f"${self.value:,.0f}"
        elif pct_in:
            val = f"{self.value:g}%"
        else:
            val = f"{self.value:g}"
        return f"{lab} {OPS[self.op][1]} {val}"


def number(x: Any) -> float | None:
    """A finite number within ±MAX_ABS, or None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(str(x).strip().replace(",", "").replace("$", ""))
    except ValueError:
        return None
    return v if math.isfinite(v) and abs(v) <= MAX_ABS else None


@dataclass
class Context:
    """Validated context filters (every field is a whitelisted value or None)."""
    seasons: tuple[str, ...] = ()
    stype: str | None = None             # table value or None = every type
    ha: str | None = None
    wl: str | None = None
    role: str | None = None
    rest: str | None = None
    team: str | None = None
    opp: str | None = None
    player: int | None = None

    def clauses(self) -> tuple[list[str], list]:
        where, args = [], []
        if self.seasons:
            where.append(f"pg.season IN ({','.join('?' * len(self.seasons))})")
            args += list(self.seasons)
        if self.stype:
            where.append("pg.season_type = ?")
            args.append(self.stype)
        for key in ("ha", "wl", "role", "rest"):
            val = getattr(self, key)
            if val is not None:
                if val not in CONTEXT[key]:
                    raise FinderError(f"unknown {key} value")
                frag, a = CONTEXT[key][val]
                where.append(frag)
                args += list(a)
        if self.team:
            if self.team not in CODES:
                raise FinderError("unknown team")
            where.append("pg.team = ?")
            args.append(self.team)
        if self.player is not None:
            if not isinstance(self.player, int) or isinstance(self.player, bool):
                raise FinderError("bad player id")
            where.append("pg.player_id = ?")
            args.append(self.player)
        if self.opp:
            if self.opp not in CODES:
                raise FinderError("unknown opponent")
            where.append("tg.opp = ?")
            args.append(self.opp)
        return where, args


# ------------------------------------------------------------------ SQL functions + maps
def _dk(pts, fg3m, reb, ast, stl, blk, tov):
    from ..proj.dfs.rules import dk_points
    return float(dk_points({"pts": pts, "fg3m": fg3m, "reb": reb, "ast": ast, "stl": stl,
                            "blk": blk, "tov": tov}))


def _ddc(pts, reb, ast, stl, blk):
    from ..proj.dfs.rules import double_double_cats
    return double_double_cats({"pts": pts, "reb": reb, "ast": ast, "stl": stl, "blk": blk})


def rest_map(conn, window: Sequence[str]) -> dict[tuple[str, str], dict[str, int | None]]:
    """{(team, season): {game_date: rest days}} (days off since the previous game of that
    team-season, any season type; None for the first game). Cached."""
    window = tuple(window)

    def build(c):
        out: dict[tuple[str, str], list[str]] = {}
        for r in c.execute(f"SELECT team, season, game_date FROM team_games WHERE season IN "
                           f"({','.join('?' * len(window))})", window):
            out.setdefault((r[0], r[1]), []).append((r[2] or "")[:10])
        res: dict[tuple[str, str], dict[str, int | None]] = {}
        for k, days in out.items():
            days = sorted(set(days))
            m: dict[str, int | None] = {}
            for i, d in enumerate(days):
                if i == 0:
                    m[d] = None
                    continue
                try:
                    m[d] = max(0, (date.fromisoformat(d) - date.fromisoformat(days[i - 1])).days
                               - 1)
                except ValueError:
                    m[d] = None
            res[k] = m
        return res
    try:
        return sdb.cached(("finder.rest", window), build, conn, domain="box") or {}
    except sqlite3.Error:
        return {}


def salary_map(conn) -> dict[tuple[int, str], int]:
    """{(player_id, day): DK classic salary}: latest capture of the day (dfs_salaries via the
    DK id map), RotoGuru history underneath. Cached."""
    def build(c):
        out: dict[tuple[int, str], tuple[str, int]] = {}
        try:
            for r in c.execute("SELECT player_id, game_date, salary FROM dfs_hist_salaries "
                               "WHERE player_id IS NOT NULL AND salary IS NOT NULL"):
                out[(int(r[0]), (r[1] or "")[:10])] = ("", int(r[2]))
        except sqlite3.Error:
            pass
        try:
            ids = {str(r[0]): int(r[1]) for r in c.execute(
                "SELECT source_id, player_id FROM player_ids WHERE source='dk' AND "
                "player_id IS NOT NULL")}
            for r in c.execute("SELECT d.site_player_id, s.nba_day, d.salary, d.captured_at "
                               "FROM dfs_salaries d JOIN dfs_slates s ON s.slate_id=d.slate_id "
                               "WHERE d.game_type_id=? AND d.salary IS NOT NULL", (CLASSIC,)):
                pid = ids.get(str(r[0]))
                if pid is None:
                    continue
                k, cap = (pid, r[1]), r[3] or ""
                if k not in out or cap >= out[k][0]:
                    out[k] = (cap, int(r[2]))
        except sqlite3.Error:
            pass
        return {k: v[1] for k, v in out.items()}
    return sdb.cached(("finder.salary",), build, conn, domain="dfs") or {}


def own_map(conn) -> dict[tuple[int, str], float]:
    """{(player_id, day): mean real %drafted (0-100) over that day's classic contests}. Cached."""
    def build(c):
        try:
            return {(int(r[0]), r[1]): float(r[2]) for r in c.execute(
                "SELECT o.player_id, s.nba_day, AVG(o.pct_drafted) FROM dfs_contest_ownership o "
                "JOIN dfs_contests c ON c.contest_id=o.contest_id JOIN dfs_slates s ON "
                "s.slate_id=c.slate_id WHERE o.pct_drafted IS NOT NULL AND s.game_type_id=? "
                "GROUP BY o.player_id, s.nba_day", (CLASSIC,))}
        except sqlite3.Error:
            return {}
    return sdb.cached(("finder.own",), build, conn, domain="dfs") or {}


def _install(conn, window: Sequence[str], need: str) -> None:
    """Register the SQL functions the query text uses (only the maps it needs are built)."""
    conn.create_function("cs_dk", 7, _dk, deterministic=True)
    conn.create_function("cs_ddc", 5, _ddc, deterministic=True)
    if "cs_rest(" in need:
        rm = rest_map(conn, window)
        conn.create_function("cs_rest", 3, lambda t, s, d: (rm.get((t, s)) or {}).get(
            (d or "")[:10]), deterministic=True)
    if "cs_sal(" in need:
        sm = salary_map(conn)
        conn.create_function("cs_sal", 2, lambda p, d: sm.get((p, (d or "")[:10])),
                             deterministic=True)
    if "cs_own(" in need:
        om = own_map(conn)
        conn.create_function("cs_own", 2, lambda p, d: om.get((p, (d or "")[:10])),
                             deterministic=True)


class QueryTimeout(RuntimeError):
    pass


def _run(conn, sql: str, args: Sequence, timeout: float = TIMEOUT_S) -> list[dict]:
    """Execute with a deadline (progress handler); QueryTimeout when it runs out."""
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20_000)
    try:
        cur = conn.execute(sql, tuple(args))
        cols = [d[0] for d in cur.description or ()]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            raise QueryTimeout("query took too long") from e
        raise
    finally:
        conn.set_progress_handler(None, 0)


# ------------------------------------------------------------------ game search
_FROM = ("FROM player_games pg LEFT JOIN team_games tg ON tg.game_id = pg.game_id "
         "AND tg.team = pg.team")
_COLS = ("pg.game_id, pg.player_id, pg.player_name, pg.team, pg.season, pg.season_type, "
         "pg.game_date, pg.min, pg.pts, pg.reb, pg.ast, pg.stl, pg.blk, pg.tov, pg.fg3m, "
         "pg.fg3a, pg.fgm, pg.fga, pg.ftm, pg.fta, pg.plus_minus, tg.opp, tg.is_home, tg.wl")


def _where(conds: Sequence[Cond], ctx: Context) -> tuple[str, list]:
    where, args = ["pg.min > 0"], []
    w2, a2 = ctx.clauses()
    where += w2
    args += a2
    for c in conds:
        if not isinstance(c, Cond):
            raise FinderError("bad condition")
        frag, a = c.sql()
        where.append(frag)
        args += list(a)
    return " AND ".join(where), args


def _sort_expr(sort: str) -> str:
    if sort not in SORTS:
        raise FinderError("unknown sort")
    return "pg.game_date" if sort == "date" else STATS[sort][1]


def game_search(conn, conds: Sequence[Cond], ctx: Context, window: Sequence[str], *,
                sort: str = "date", group: bool = False, cap: int = CAP,
                timeout: float = TIMEOUT_S) -> dict:
    """{rows, count, capped, group}. Rows newest first (or by `sort`, high first), at most
    `cap`. group=True: {player_id, name, team, games, last, avg_pts, avg_dk} by games."""
    where, args = _where(conds, ctx)
    order = _sort_expr(sort)
    if group:
        sql = (f"SELECT pg.player_id, pg.player_name AS name, pg.team, COUNT(*) AS games, "
               f"MAX(pg.game_date) AS last, AVG(pg.pts) AS avg_pts, "
               f"AVG({STATS['dk'][1]}) AS avg_dk, COUNT(*) OVER () AS n_groups, "
               f"SUM(COUNT(*)) OVER () AS n_games {_FROM} WHERE {where} "
               f"GROUP BY pg.player_id ORDER BY games DESC, last DESC LIMIT ?")
    else:
        sql = (f"SELECT {_COLS}, COUNT(*) OVER () AS n_games {_FROM} WHERE {where} "
               f"ORDER BY {order} DESC, pg.game_date DESC, pg.player_id LIMIT ?")
    _install(conn, window, sql)
    rows = _run(conn, sql, [*args, cap + 1], timeout)
    capped = len(rows) > cap
    rows = rows[:cap]
    if group:
        count = int(rows[0]["n_games"]) if rows else 0
        groups = int(rows[0]["n_groups"]) if rows else 0
        return {"rows": rows, "count": count, "groups": groups, "capped": capped, "group": True}
    count = int(rows[0]["n_games"]) if rows else 0
    sal, own = salary_map(conn), own_map(conn)
    for r in rows:
        r["dk"] = _dk(r["pts"], r["fg3m"], r["reb"], r["ast"], r["stl"], r["blk"], r["tov"])
        k = (r["player_id"], (r["game_date"] or "")[:10])
        r["salary"] = sal.get(k)
        r["own"] = own.get(k)
        r["value"] = r["dk"] / r["salary"] * 1000 if r["salary"] else None
    return {"rows": rows, "count": count, "groups": None, "capped": capped, "group": False}


# ------------------------------------------------------------------ streaks
@dataclass(frozen=True)
class Screen:
    stat: str
    op: str
    thr: float
    k: int
    n: int
    min_gp: int

    @classmethod
    def make(cls, stat, op, thr, k, n, min_gp=None) -> "Screen":
        if stat not in STREAK_STATS:
            raise FinderError(f"unknown stat {str(stat)[:20]!r}")
        if op not in ("ge", "le"):
            raise FinderError(f"unknown operator {str(op)[:10]!r}")
        t = number(thr)
        if t is None:
            raise FinderError("threshold is not a number")
        try:
            n_, k_ = int(n), int(k)
            m_ = int(min_gp) if min_gp not in (None, "") else n_
        except (TypeError, ValueError) as e:
            raise FinderError("k, n and minimum games must be whole numbers") from e
        if not 1 <= n_ <= 82:
            raise FinderError("n must be 1 to 82")
        if not 1 <= k_ <= n_:
            raise FinderError("k must be 1 to n")
        return cls(stat, op, t, k_, n_, max(1, min(m_, 300)))

    def label(self) -> str:
        c = Cond(self.stat, self.op, self.thr).label()
        if self.k == self.n:
            return f"{c} in each of the last {self.n}"
        return f"{c} in at least {self.k} of the last {self.n}"


def streaks(conn, sc: Screen, ctx: Context, window: Sequence[str], *, cap: int = CAP,
            timeout: float = TIMEOUT_S) -> dict:
    """Players reaching the screen in their last n games played (within ctx's seasons / type /
    team filters). Rows: player_id, name, team, gp, n, hits, avg, last, seq (oldest -> newest
    values), run (current run of games reaching it). Most hits first, then the average."""
    where, args = _where((), ctx)
    expr = STATS[sc.stat][1]
    thr = sc.thr / 100.0 if STATS[sc.stat][3] else sc.thr
    cmp = OPS[sc.op][0]
    sql = (f"WITH g AS (SELECT pg.player_id, pg.player_name, pg.team, pg.game_date, "
           f"{expr} AS v, ROW_NUMBER() OVER (PARTITION BY pg.player_id ORDER BY pg.game_date "
           f"DESC, pg.game_id DESC) AS rn, COUNT(*) OVER (PARTITION BY pg.player_id) AS gp "
           f"{_FROM} WHERE {where}) "
           f"SELECT player_id, MAX(CASE WHEN rn = 1 THEN player_name END) AS name, "
           f"MAX(CASE WHEN rn = 1 THEN team END) AS team, MAX(gp) AS gp, COUNT(*) AS n, "
           f"SUM(CASE WHEN v {cmp} ? THEN 1 ELSE 0 END) AS hits, AVG(v) AS avg, "
           f"MAX(game_date) AS last, GROUP_CONCAT(IFNULL(v, ''), '|' ORDER BY rn DESC) AS seq, "
           f"COUNT(*) OVER () AS n_players "
           f"FROM g WHERE rn <= ? GROUP BY player_id "
           f"HAVING hits >= ? AND MAX(gp) >= ? "
           f"ORDER BY hits DESC, avg {'DESC' if sc.op == 'ge' else 'ASC'}, last DESC LIMIT ?")
    _install(conn, window, sql)
    rows = _run(conn, sql, [*args, thr, sc.n, sc.k, sc.min_gp, cap + 1], timeout)
    capped = len(rows) > cap
    rows = rows[:cap]
    count = int(rows[0]["n_players"]) if rows else 0
    for r in rows:
        seq = []
        for x in (r.pop("seq") or "").split("|"):
            try:
                seq.append(float(x) if x != "" else None)
            except ValueError:
                seq.append(None)
        r["seq"] = [{"v": v, "hit": v is not None and _cmp(v, cmp, thr)} for v in seq]
        run = 0
        for s in reversed(r["seq"]):
            if not s["hit"]:
                break
            run += 1
        r["run"] = run
    return {"rows": rows, "count": count, "capped": capped}


def _cmp(v: float, op: str, t: float) -> bool:
    return {">=": v >= t, "<=": v <= t, ">": v > t, "<": v < t, "=": v == t}[op]


# ------------------------------------------------------------------ parsing helpers
def parse_conds(params: Mapping[str, Any]) -> tuple[list[Cond], list[str]]:
    """Rows s1/o1/v1 .. s6/o6/v6 plus the DFS ranges (sal_min, sal_max, own_min, own_max).
    Returns (conditions, error notices). A row with an empty value is ignored."""
    conds, errs = [], []
    for i in range(1, MAX_CONDS + 1):
        s, o, v = params.get(f"s{i}"), params.get(f"o{i}") or "ge", params.get(f"v{i}")
        if not s and (v is None or str(v).strip() == ""):
            continue
        if v is None or str(v).strip() == "":
            continue
        try:
            c = Cond.make(s, o, v)
            if c.stat in ("salary", "own"):
                raise FinderError(f"unknown stat {s!r}")
            conds.append(c)
        except FinderError as e:
            errs.append(f"Condition {i} ignored: {e}.")
    for key, stat, op in (("sal_min", "salary", "ge"), ("sal_max", "salary", "le"),
                          ("own_min", "own", "ge"), ("own_max", "own", "le")):
        v = params.get(key)
        if v is None or str(v).strip() == "":
            continue
        try:
            conds.append(Cond.make(stat, op, v))
        except FinderError as e:
            errs.append(f"{key} ignored: {e}.")
    return conds, errs
