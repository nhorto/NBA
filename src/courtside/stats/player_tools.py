"""Player research tools (docs/STATS_SITE.md, player page): the recent-games bar chart with a
reference line, filters and splits, the minutes trend and the clutch line.

Pure functions on a read-only `sqlite3.Connection`; plain dicts/lists out. Information only:
everything is history ("reached 25+ in 7 of his last 10"), never over/under wording.

  games(conn, pid, window)             his played games (min > 0), newest first, each with
                                       opp / home / W-L, started, rest days and the opponent's
                                       defense tier vs his position (cached per player)
  teammates(conn, pid, window)         teammates he shared the floor with (games together)
  parse_filters(params, ...)           query params -> a clean filter dict
  apply_filters(conn, pid, games, f)   the filtered games (with/without a teammate is one
                                       indexed query on the teammate's rows)
  splits(conn, pid, games, ...)        per-game averages per split + sample size
  bar_chart(games, stat, win, line, ...)   server-side SVG geometry + the summary text
  minutes_trend(games, season, stype)  sparkline geometry, 30+/36+ shares, started vs bench
  clutch(conn, pid, season, stype)     last 5 min of Q4/OT within 5 points, from the archived
                                       play-by-play (cached per player-season)

Rest days: days off between his team's previous game (any season type, same season) and this
one; 0 = the second night of a back-to-back, None = first game of the season.
Opponent defense tier: the DK-points-allowed rank of the opponent against his position
(stats.leaders DvP, the same numbers as the team pages). Rank 1 allows the most, so the top-10
defenses are ranks n-9..n. A season whose teams average under 10 games uses the previous
season's ranks. Playoff games use the same season's regular-season ranks.
"""

from __future__ import annotations

import gzip
import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import db as sdb
from . import seasons as S

# key -> (label, integer stat)
STATS: dict[str, tuple[str, bool]] = {
    "pts": ("PTS", True), "reb": ("REB", True), "ast": ("AST", True), "fg3m": ("3PM", True),
    "pra": ("PRA", True), "dk": ("DK pts", False), "min": ("MIN", False),
    "stocks": ("STL+BLK", True)}
STAT_NAMES = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes",
              "pra": "pts+reb+ast", "dk": "DK points", "min": "minutes", "stocks": "steals+blocks"}
WINDOWS = (("5", "Last 5"), ("10", "Last 10"), ("20", "Last 20"), ("season", "Season"))
DEFAULT_WIN = "10"
SMALL = 3                       # splits with fewer games are greyed out
TIER_GAMES = 10                 # a season's DvP needs teams averaging this many games
TIERS = {"top": "Top-10 D", "mid": "Middle D", "bottom": "Bottom-10 D"}
REST = {"0": "Back-to-back", "1": "1 day rest", "2": "2+ days rest"}
CLUTCH_SEC, CLUTCH_MARGIN = 300.0, 5
SPLIT_KEYS = ("min", "pts", "reb", "ast", "fg3m", "pra", "stocks", "dk")


# ------------------------------------------------------------------ helpers
def _rows(conn, sql: str, args: Sequence = ()) -> list[dict]:
    if conn is None:
        return []
    try:
        cur = conn.execute(sql, tuple(args))
    except sqlite3.Error:
        return []
    cols = [d[0] for d in cur.description or ()]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _marks(xs: Sequence) -> str:
    return ",".join("?" * len(xs))


def _chunks(xs: Sequence, n: int = 400) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def stat_value(g: Mapping, key: str) -> float:
    """One game's value of a chart stat (missing parts count as 0)."""
    def v(k):
        return float(g.get(k) or 0)
    if key == "pra":
        return v("pts") + v("reb") + v("ast")
    if key == "stocks":
        return v("stl") + v("blk")
    return v(key)


def default_line(avg: float | None) -> float:
    """His average rounded to the nearest x.5 (26.3 -> 26.5, 26.8 -> 26.5, 0 -> 0.5)."""
    if avg is None or avg != avg:
        return 0.5
    return math.floor(avg) + 0.5


def threshold_label(line: float, integer: bool) -> str:
    """The 'reached N+' number: integer stats round the line up (24.5 -> '25')."""
    if integer:
        return str(int(math.ceil(line - 1e-9)))
    return f"{line:.1f}".rstrip("0").rstrip(".")


def count_hits(values: Sequence[float], line: float) -> int:
    return sum(1 for v in values if v >= line - 1e-9)


def _pct(k: int, n: int) -> str:
    return f"{math.floor(100.0 * k / n + 0.5)}%" if n else "–"     # same rounding as the JS


# ------------------------------------------------------------------ his games, enriched
def _dvp_season(conn, season: str) -> tuple[str, dict, dict] | None:
    """(source season, dvp rows by team, positions) used for games of `season`."""
    from . import leaders as L

    def build(c):
        for s in (season, S.prev_season(season)):
            pk = L.pack(c, s, S.REGULAR) or {}
            dv = pk.get("dvp") or {}
            if not dv:
                continue
            avg = sum(r.get("gp", 0) for r in dv.values()) / len(dv)
            if avg >= TIER_GAMES or s != season:
                pos = L.positions(c, s)
                return (s, dv, pos)
        return None
    return sdb.cached(("pl2-dvp", season), build, conn, domain="box")


def tier_of(rank: int | None, n: int) -> str | None:
    """DvP rank (1 = allows the most) -> 'bottom' (ranks 1-10), 'top' (last 10), else 'mid'."""
    if rank is None or not n:
        return None
    if rank <= 10:
        return "bottom"
    if rank > n - 10:
        return "top"
    return "mid"


def games(conn, pid: int, window: Sequence[str]) -> list[dict]:
    """His games with minutes in the window, newest first (cached). Each row: game_id, team,
    season, season_type, game_date, the box score, dk, opp, is_home, wl, started (True / False
    / None = unknown), rest (int or None), dvp_rank, dvp_n, tier, pos."""
    from .players import _box_rows                       # same cached box rows as the log
    window = tuple(window)

    def build(c):
        rows = [dict(r) for r in _box_rows(c, pid, window) if r["played"]]
        if not rows:
            return []
        gids = [r["game_id"] for r in rows]
        # who started: his starts + the games that have start data at all
        mine = {r["game_id"] for r in _rows(c, "SELECT game_id FROM player_starts WHERE "
                                               "player_id=?", (pid,))}
        covered: set[str] = set()
        for ch in _chunks(gids):
            covered |= {r["game_id"] for r in _rows(
                c, f"SELECT DISTINCT game_id FROM player_starts WHERE game_id IN ({_marks(ch)})",
                ch)}
        # rest: his teams' schedules in these seasons
        teams = sorted({r["team"] for r in rows if r["team"]})
        sched: dict[tuple[str, str], list[str]] = {}
        if teams:
            for r in _rows(c, f"SELECT team, season, game_date FROM team_games WHERE team IN "
                              f"({_marks(teams)}) AND season IN ({_marks(window)})",
                           (*teams, *window)):
                sched.setdefault((r["team"], r["season"]), []).append(r["game_date"] or "")
        for v in sched.values():
            v.sort()
        dvps = {s: _dvp_season(c, s) for s in {r["season"] for r in rows}}
        import bisect
        for r in rows:
            gid = r["game_id"]
            r["started"] = (gid in mine) if gid in covered else None
            days = sched.get((r["team"], r["season"]), [])
            i = bisect.bisect_left(days, r["game_date"] or "")
            r["rest"] = None
            if i > 0 and r["game_date"]:
                try:
                    d0 = date.fromisoformat(days[i - 1][:10])
                    d1 = date.fromisoformat(r["game_date"][:10])
                    r["rest"] = max(0, (d1 - d0).days - 1)
                except ValueError:
                    pass
            r["dvp_rank"] = r["dvp_n"] = r["tier"] = r["pos"] = r["dvp_season"] = None
            dv = dvps.get(r["season"])
            if dv and r["opp"]:
                src, rows_by_team, posmap = dv
                b = (posmap.get("pos") or {}).get(int(pid))
                row = rows_by_team.get(r["opp"])
                if b and row:
                    rk = (row.get("rk") or {}).get(b)
                    r.update(dvp_rank=rk, dvp_n=len(rows_by_team), pos=b, dvp_season=src,
                             tier=tier_of(rk, len(rows_by_team)))
        return rows
    return sdb.cached(("pl2-games", pid, window), build, conn, domain="box") or []


def teammates(conn, pid: int, window: Sequence[str]) -> list[dict]:
    """[{player_id, name, games, min, last, team, season, current, by_season: {season: games},
    min_by_season: {season: his minutes}}]:
    players who played (min > 0) for his team in his games; teammates from his latest team
    first, then most games together (cached). Scoped to his game ids."""
    window = tuple(window)

    def build(c):
        mine = games(c, pid, window)
        team_of = {g["game_id"]: g["team"] for g in mine}
        season_of = {g["game_id"]: (g["season"], g["game_date"]) for g in mine}
        gids = list(team_of)
        acc: dict[int, dict] = {}
        for ch in _chunks(gids):
            for r in _rows(c, f"SELECT game_id, player_id, team, min FROM player_games WHERE "
                              f"game_id IN ({_marks(ch)}) AND min > 0", ch):
                tid = int(r["player_id"])
                if tid == pid or r["team"] != team_of.get(r["game_id"]):
                    continue
                a = acc.setdefault(tid, {"player_id": tid, "games": 0, "min": 0.0, "last": "",
                                         "team": r["team"], "season": None, "by_season": {},
                                         "min_by_season": {}})
                a["games"] += 1
                sk = season_of[r["game_id"]][0]
                a["by_season"][sk] = a["by_season"].get(sk, 0) + 1
                a["min_by_season"][sk] = a["min_by_season"].get(sk, 0.0) + (r["min"] or 0)
                a["min"] += r["min"] or 0
                s, d = season_of[r["game_id"]]
                if (d or "") >= a["last"]:
                    a.update(last=d or "", team=r["team"], season=s)
        from .players import names
        nm = names(c, acc)
        out = list(acc.values())
        newest = max(mine, key=lambda g: g["game_date"] or "") if mine else {}
        for a in out:
            a["name"] = nm.get(a["player_id"], f"#{a['player_id']}")
            a["current"] = (a["season"], a["team"]) == (newest.get("season"), newest.get("team"))
        out.sort(key=lambda a: (not a["current"], -a["games"], a["name"]))
        return out
    return sdb.cached(("pl2-mates", pid, window), build, conn, domain="box") or []


def teammate_sets(conn, tid: int, window: Sequence[str]) -> tuple[set[tuple[str, str]],
                                                                   set[tuple[str, str]]]:
    """({(game_id, team) he played in}, {(season, team) he played for}) for one teammate:
    one query on his own rows (player_id index), never a table scan."""
    played, spells = set(), set()
    for r in _rows(conn, f"SELECT game_id, team, season FROM player_games WHERE player_id=? AND "
                         f"season IN ({_marks(window)}) AND min > 0", (tid, *window)):
        played.add((r["game_id"], r["team"]))
        spells.add((r["season"], r["team"]))
    return played, spells


# ------------------------------------------------------------------ filters
FILTER_KEYS = ("ha", "opp", "role", "rest", "wl", "last", "tm", "tmode", "dt")


def parse_filters(params: Mapping[str, Any], *, opps: Iterable[str] = (),
                  mates: Iterable[int] = ()) -> dict:
    """Clean filter dict from query params (unknown values are dropped)."""
    f: dict[str, Any] = {}
    ha = (params.get("ha") or "").lower()
    if ha in ("home", "away"):
        f["ha"] = ha
    opp = (params.get("opp") or "").strip().upper()[:3]
    if opp and opp in set(opps):
        f["opp"] = opp
    role = (params.get("role") or "").lower()
    if role in ("start", "bench"):
        f["role"] = role
    rest = params.get("rest") or ""
    if rest in REST:
        f["rest"] = rest
    wl = (params.get("wl") or "").upper()
    if wl in ("W", "L"):
        f["wl"] = wl
    try:
        last = int(params.get("last") or 0)
    except ValueError:
        last = 0
    if 0 < last <= 200:
        f["last"] = last
    try:
        tm = int(params.get("tm") or 0)
    except ValueError:
        tm = 0
    if tm and tm in set(mates):
        f["tm"] = tm
        f["tmode"] = "without" if (params.get("tmode") or "").lower() == "without" else "with"
    dt = (params.get("dt") or "").lower()
    if dt in TIERS:
        f["dt"] = dt
    return f


def _match(g: dict, f: dict) -> bool:
    if "ha" in f and bool(g.get("is_home")) != (f["ha"] == "home"):
        return False
    if "opp" in f and g.get("opp") != f["opp"]:
        return False
    if "role" in f and g.get("started") is not (f["role"] == "start"):
        return False
    if "rest" in f:
        r = g.get("rest")
        if r is None or (f["rest"] == "2" and r < 2) or (f["rest"] != "2" and r != int(f["rest"])):
            return False
    if "wl" in f and g.get("wl") != f["wl"]:
        return False
    if "dt" in f and g.get("tier") != f["dt"]:
        return False
    return True


def teammate_filter(conn, games_: list[dict], tid: int, mode: str,
                    window: Sequence[str]) -> list[dict]:
    """With: games where the teammate also played (min > 0) for the same team. Without: games
    where he did not, in the seasons he played for that team (so a teammate who arrived this
    year does not count every older game as 'without')."""
    played, spells = teammate_sets(conn, tid, window)
    out = []
    for g in games_:
        together = (g["game_id"], g["team"]) in played
        if mode == "with" and together:
            out.append(g)
        elif mode == "without" and not together and (g["season"], g["team"]) in spells:
            out.append(g)
    return out


def apply_filters(conn, pid: int, games_: list[dict], f: dict,
                  window: Sequence[str]) -> list[dict]:
    """The games (newest first) that pass every filter; `last` applies after the others."""
    out = [g for g in games_ if _match(g, f)]
    if "tm" in f:
        out = teammate_filter(conn, out, f["tm"], f.get("tmode", "with"), window)
    if "last" in f:
        out = out[:f["last"]]
    return out


# ------------------------------------------------------------------ averages + splits
def averages(gs: Sequence[dict]) -> dict:
    """{gp, min, pts, ..., dk, ts} per game over these games (gp 0 -> values None)."""
    n = len(gs)
    out: dict[str, Any] = {"gp": n}
    for k in SPLIT_KEYS:
        out[k] = sum(stat_value(g, k) for g in gs) / n if n else None
    pts = sum(g.get("pts") or 0 for g in gs)
    tsa = 2.0 * sum((g.get("fga") or 0) + 0.44 * (g.get("fta") or 0) for g in gs)
    out["ts"] = pts / tsa if tsa else None
    out["small"] = n < SMALL
    return out


def splits(conn, pid: int, gs: list[dict], *, window: Sequence[str], mate: dict | None,
           opp: str | None = None) -> list[dict]:
    """[{group, label, key, value, avg}] for these games (newest first). key/value = the filter
    param that selects the split, so rows can link to it."""
    def row(group, label, sel, key=None, value=None, **extra):
        return {"group": group, "label": label, "key": key, "value": value,
                "avg": averages(sel), **extra}
    out = [row("All", "All games", gs)]
    out += [row("Venue", "Home", [g for g in gs if g.get("is_home")], "ha", "home"),
            row("Venue", "Away", [g for g in gs if not g.get("is_home")], "ha", "away")]
    if any(g.get("started") is not None for g in gs):
        out += [row("Role", "Starter", [g for g in gs if g.get("started") is True], "role",
                    "start"),
                row("Role", "Bench", [g for g in gs if g.get("started") is False], "role",
                    "bench")]
    for k, lab in REST.items():
        sel = [g for g in gs if g.get("rest") is not None
               and (g["rest"] >= 2 if k == "2" else g["rest"] == int(k))]
        out.append(row("Rest", lab, sel, "rest", k))
    out += [row("Result", "Wins", [g for g in gs if g.get("wl") == "W"], "wl", "W"),
            row("Result", "Losses", [g for g in gs if g.get("wl") == "L"], "wl", "L")]
    if any(g.get("tier") for g in gs):
        for k, lab in TIERS.items():
            out.append(row("Opp defense", lab + " vs his position",
                           [g for g in gs if g.get("tier") == k], "dt", k))
    if mate:
        for mode in ("with", "without"):
            sel = teammate_filter(conn, gs, mate["player_id"], mode, window)
            out.append(row("Teammate", f"{mode.capitalize()} {mate['name']}", sel, "tm",
                           mate["player_id"], tmode=mode))
    if opp:
        out.append(row("Opponent", f"vs {opp}", [g for g in gs if g.get("opp") == opp], "opp",
                       opp))
    for n in (5, 10):
        if len(gs) > n:
            out.append(row("Recent", f"Last {n}", gs[:n], "last", n))
    return out


# ------------------------------------------------------------------ bar chart
CH_H, CH_TOP, CH_BOT, CH_LEFT, CH_RIGHT = 250, 22, 206, 34, 40


def nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return m * mag
    return 10 * mag


def bar_path(x: float, yt: float, w: float, r: float = 4.0) -> str:
    """A bar standing on the baseline with rounded top corners (flat when too short)."""
    h = CH_BOT - yt
    if h <= 0.05:
        return f"M{x:.1f},{CH_BOT} h{w:.1f}"
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{CH_BOT}V{yt + r:.1f}Q{x:.1f},{yt:.1f} {x + r:.1f},{yt:.1f}"
            f"H{x + w - r:.1f}Q{x + w:.1f},{yt:.1f} {x + w:.1f},{yt + r:.1f}V{CH_BOT}Z")


def chart_window(gs: list[dict], win: str, season: str, stype: str) -> list[dict]:
    """The bars, oldest first: the last N filtered games, or the selected season's."""
    if win == "season":
        sel = [g for g in gs if g["season"] == season and g["season_type"] == stype]
    else:
        sel = gs[:int(win)]
    return list(reversed(sel))


def window_phrase(win: str, n: int, season: str) -> str:
    if win == "season":
        return f"{season}"
    return f"last {n}"                 # fewer games than asked (filters): say how many


def summary(line: float, integer: bool, parts: Sequence[tuple[str, Sequence[float]]]) -> str:
    """'Reached 25+ in 7 of last 10 (70%) · 2026-27 18 of 31 (58%) · vs DAL 2 of 3'.
    parts: [(label, values)], the first is the chart window."""
    th = threshold_label(line, integer)
    out = []
    for i, (lab, vals) in enumerate(parts):
        k, n = count_hits(vals, line), len(vals)
        if i == 0:
            out.append(f"Reached {th}+ in {k} of {lab} ({_pct(k, n)})" if n
                       else f"No games for {lab}")
        elif n:
            out.append(f"{lab} {k} of {n} ({_pct(k, n)})")
    return " · ".join(out)


def bar_chart(gs: list[dict], stat: str, win: str, line: float | None, *, season: str,
              stype: str, season_avg: float | None, vs_opp: str | None) -> dict:
    """Geometry and text of the recent-games chart (all server-side; pro-players.js only moves
    the line). gs: filtered games, newest first."""
    label, integer = STATS[stat]
    bars_g = chart_window(gs, win, season, stype)
    vals = [stat_value(g, stat) for g in bars_g]
    if season_avg is None:
        pool = [stat_value(g, stat) for g in gs]
        season_avg = sum(pool) / len(pool) if pool else None
    dflt = default_line(season_avg)
    ln = dflt if line is None else max(0.0, float(line))
    n = len(bars_g)
    top = max(vals + [ln, dflt, 1.0])
    ymax = nice_max(top * 1.08)
    wide = n > 20
    step_px = 13 if wide else 36
    width = max(720, CH_LEFT + CH_RIGHT + n * step_px)
    plot_w = width - CH_LEFT - CH_RIGHT
    step = plot_w / max(n, 1)
    bw = min(step * (0.62 if not wide else 0.7), 44)
    h = CH_BOT - CH_TOP

    def y(v: float) -> float:
        return round(CH_BOT - h * min(v, ymax) / ymax, 1)
    bars = []
    for i, (g, v) in enumerate(zip(bars_g, vals)):
        x = CH_LEFT + step * i + (step - bw) / 2
        yt = y(v)
        hit = v >= ln - 1e-9
        res = f" ({g['wl']})" if g.get("wl") else ""
        role = {True: " · started", False: " · bench"}.get(g.get("started"), "")
        fgs = f"{int(g.get('fgm') or 0)}-{int(g.get('fga') or 0)} FG"
        tip = (f"{g['game_date']} · {'vs' if g.get('is_home') else '@'} {g.get('opp') or '?'}"
               f"{res}\n{round(g.get('min') or 0)} min{role}\n"
               f"{int(g.get('pts') or 0)} pts · {int(g.get('reb') or 0)} reb · "
               f"{int(g.get('ast') or 0)} ast · {int(g.get('fg3m') or 0)} 3PM · "
               f"{int(g.get('stl') or 0)} stl · {int(g.get('blk') or 0)} blk\n"
               f"{fgs} · {g.get('dk', 0):.2f} DK pts")
        bars.append({"x": round(x, 1), "y": yt, "w": round(bw, 1), "d": bar_path(x, yt, bw),
                     "h": round(max(CH_BOT - yt, 0.0), 1), "v": v,
                     "text": f"{v:g}" if integer else f"{v:.1f}", "hit": hit,
                     "cx": round(x + bw / 2, 1), "opp": g.get("opp") or "?",
                     "home": bool(g.get("is_home")), "tip": tip, "date": g["game_date"]})
    ticks = [round(ymax * i / 4, 2) for i in range(5)]
    season_vals = [stat_value(g, stat) for g in gs
                   if g["season"] == season and g["season_type"] == stype]
    vs_vals = [stat_value(g, stat) for g in gs if vs_opp and g.get("opp") == vs_opp]
    parts = [(window_phrase(win, n, season) if win != "season"
              else f"{n} game{'' if n == 1 else 's'} in {season}", vals)]
    if win != "season":
        parts.append((season, season_vals))
    if vs_opp:
        parts.append((f"vs {vs_opp}", vs_vals))
    text = summary(ln, integer, parts)
    data = {"stat": stat, "label": label, "integer": integer, "line": ln, "default": dflt,
            "ymax": ymax, "top": CH_TOP, "bot": CH_BOT,
            "parts": [{"label": lab, "values": v} for lab, v in parts]}
    return {"stat": stat, "label": label, "name": STAT_NAMES[stat], "integer": integer,
            "win": win, "line": ln, "default": dflt, "season_avg": season_avg, "bars": bars,
            "n": n, "hits": count_hits(vals, ln), "width": width, "height": CH_H,
            "left": CH_LEFT, "right": width - CH_RIGHT, "top": CH_TOP, "bot": CH_BOT,
            "ymax": ymax, "ticks": [{"v": t, "y": y(t)} for t in ticks], "line_y": y(ln),
            "wide": wide, "summary": text, "data": data,
            "line_text": threshold_label(ln, integer) + "+"}


def json_script(obj: Any) -> str:
    """JSON safe inside <script type="application/json"> (no '</' or '<!--')."""
    return json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c")


# ------------------------------------------------------------------ minutes trend
SP_W, SP_H, SP_PAD, SP_LAB = 300, 84, 8, 16


def minutes_trend(gs: list[dict], season: str, stype: str, n: int = 20) -> dict:
    """Last n games' minutes (oldest first) as a sparkline + shares; started vs bench minutes
    in the selected season. gs: unfiltered games, newest first."""
    last = list(reversed(gs[:n]))
    mins = [float(g.get("min") or 0) for g in last]
    hi = max(mins + [40.0])
    lo = 0.0

    def y(v):
        return round(SP_H - SP_PAD - (SP_H - 2 * SP_PAD) * (v - lo) / (hi - lo), 1)
    k = len(last)
    dx = (SP_W - 2 * SP_PAD - SP_LAB) / max(k - 1, 1)
    pts = []
    for i, (g, m) in enumerate(zip(last, mins)):
        pts.append({"x": round(SP_PAD + dx * i, 1) if k > 1 else SP_W / 2, "y": y(m), "min": m,
                    "started": g.get("started"),
                    "tip": f"{g['game_date']} · {'vs' if g.get('is_home') else '@'} "
                           f"{g.get('opp') or '?'}\n{m:.0f} min"
                           + {True: " · started", False: " · bench"}.get(g.get("started"), "")})
    sea = [g for g in gs if g["season"] == season and g["season_type"] == stype]

    def share(sel, t):
        return (sum(1 for g in sel if (g.get("min") or 0) >= t) / len(sel)) if sel else None

    def avg(sel):
        return sum(g.get("min") or 0 for g in sel) / len(sel) if sel else None
    st = [g for g in sea if g.get("started") is True]
    bn = [g for g in sea if g.get("started") is False]
    return {"n": k, "points": pts, "path": " ".join(f"{p['x']},{p['y']}" for p in pts),
            "w": SP_W, "h": SP_H, "lab_x": SP_W - SP_LAB, "y30": y(30), "y36": y(36), "avg": avg(last),
            "l5": avg(list(reversed(last))[:5]),
            "s30": share(last, 30), "s36": share(last, 36),
            "season_n": len(sea), "season_s30": share(sea, 30), "season_s36": share(sea, 36),
            "start_n": len(st), "start_min": avg(st), "bench_n": len(bn), "bench_min": avg(bn),
            "has_role": any(g.get("started") is not None for g in sea)}


# ------------------------------------------------------------------ clutch
def pbp_path(game_date: str, game_id: str) -> Path:
    from .. import config
    return config.data_dir() / "raw" / "nba_cdn" / "playbyplay" / str(game_date)[:10] \
        / f"{game_id}.json.gz"


def clutch_game(payload: dict, pid: int, team: str) -> dict:
    """His clutch line in one game: {sec, pts, fgm, fga, fg3m, fg3a, ftm, fta, pm, any}.
    Clutch = 4th quarter or overtime, 5:00 or less left, score within 5 before the play.
    On-court status: his substitutions; at a period start he is on when his first sub of the
    period is 'out', or when he has no sub but appears in the period."""
    from ..analytics.pbp import parse_clock
    acts = (payload.get("game") or {}).get("actions") or []
    acts = sorted(acts, key=lambda a: (a.get("orderNumber") or 0, a.get("actionNumber") or 0))
    start_on: dict[int, bool] = {}
    for a in acts:                                  # first sub of his in each late period
        p = int(a.get("period") or 0)
        if p >= 4 and p not in start_on and a.get("actionType") == "substitution" \
                and a.get("personId") == pid:
            start_on[p] = (a.get("subType") or "").lower() == "out"
    for a in acts:                                  # no sub: on when he shows up at all
        p = int(a.get("period") or 0)
        if p >= 4 and p not in start_on and pid in (a.get("personIdsFilter") or ()):
            start_on[p] = True
    out = {"sec": 0.0, "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0,
           "pm": 0, "any": False}
    sh = sa = 0
    cur_p, prev_clock, on = 0, 720.0, False
    side = None                                     # 'home' / 'away' once known
    for a in acts:
        p = int(a.get("period") or 0)
        clock = parse_clock(a.get("clock"))
        if p != cur_p:
            cur_p, on = p, start_on.get(p, False)
            prev_clock = 720.0 if p <= 4 else 300.0
        try:
            nh, na = int(a.get("scoreHome") or sh), int(a.get("scoreAway") or sa)
        except (TypeError, ValueError):
            nh, na = sh, sa
        if side is None and a.get("teamTricode") and (nh != sh or na != sa):
            scored_home = nh != sh
            side = ("home" if scored_home else "away") if a.get("teamTricode") == team else \
                ("away" if scored_home else "home")
        margin = abs(sh - sa)
        late = p >= 4 and margin <= CLUTCH_MARGIN
        if late and on:
            seg = max(0.0, min(prev_clock, CLUTCH_SEC) - clock)
            if seg > 0:
                out["sec"] += seg
                out["any"] = True
        clutch = late and clock <= CLUTCH_SEC
        if clutch and a.get("personId") == pid:
            t, made = a.get("actionType"), a.get("shotResult") == "Made"
            if t in ("2pt", "3pt"):
                out["fga"] += 1
                out["fgm"] += made
                out["any"] = True
                if t == "3pt":
                    out["fg3a"] += 1
                    out["fg3m"] += made
                if made:
                    out["pts"] += 3 if t == "3pt" else 2
            elif t == "freethrow":
                out["fta"] += 1
                out["ftm"] += made
                out["pts"] += made
                out["any"] = True
        if clutch and on and side and (nh != sh or na != sa):
            d = (nh - sh) - (na - sa)
            out["pm"] += d if side == "home" else -d
        if a.get("personId") == pid and a.get("actionType") == "substitution":
            on = (a.get("subType") or "").lower() == "in"
        sh, sa, prev_clock = nh, na, clock
    return out


def clutch(conn, pid: int, season: str, season_type: str = S.REGULAR,
           window: Sequence[str] | None = None) -> dict:
    """{games: [{game_id, game_date, opp, is_home, wl, ...line}], tot: {...}, n_games, with_pbp,
    missing} for one player-season (cached; reads the archived play-by-play)."""
    def build(c):
        mine = [g for g in games(c, pid, tuple(window or (season,)))
                if g["season"] == season and g["season_type"] == season_type]
        rows, missing = [], 0
        for g in mine:
            path = pbp_path(g["game_date"], g["game_id"])
            try:
                payload = json.loads(gzip.decompress(path.read_bytes()))
            except (OSError, ValueError):
                missing += 1
                continue
            ln = clutch_game(payload, pid, g["team"])
            if ln["any"]:
                rows.append({"game_id": g["game_id"], "game_date": g["game_date"],
                             "opp": g["opp"], "is_home": g["is_home"], "wl": g["wl"], **ln})
        tot = {k: sum(r[k] for r in rows) for k in ("sec", "pts", "fgm", "fga", "fg3m", "fg3a",
                                                     "ftm", "fta", "pm")}
        tot["min"] = tot["sec"] / 60.0
        tot["fg_pct"] = tot["fgm"] / tot["fga"] if tot["fga"] else None
        tot["ts"] = (tot["pts"] / (2 * (tot["fga"] + 0.44 * tot["fta"]))
                     if (tot["fga"] or tot["fta"]) else None)
        tot["p36"] = 36 * 60 * tot["pts"] / tot["sec"] if tot["sec"] >= 60 else None
        rows.sort(key=lambda r: r["game_date"], reverse=True)
        return {"games": rows, "tot": tot, "n_games": len(rows), "season_games": len(mine),
                "with_pbp": len(mine) - missing, "missing": missing, "season": season,
                "season_type": season_type,
                "wins": sum(1 for r in rows if r["wl"] == "W")}
    return sdb.cached(("pl2-clutch", pid, season, season_type), build, conn, domain="box")
