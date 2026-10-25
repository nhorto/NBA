"""Player season aggregates for the stats pages: league leaders, team rosters and defense vs
position (docs/STATS_SITE.md). Pure functions on a read-only `sqlite3.Connection`.

One heavy pass per (season, season type) reads every `player_games` line (+ `player_games_adv`
usage, + the opponent from `team_games`) and keeps only aggregates, cached with
`stats.db.cached` until courtside.db changes:

  players   player_id -> {name, team (latest), windows {"all", "l10", "l5"} -> sums}
            "l10"/"l5" are the player's last 10 / 5 games played (min > 0) of the season
  roster    (team, player_id) -> sums of the games he played FOR that team (trades split)
  dvp       defense vs position (see `dvp_table`)

Sums: gp, min, pts, reb, ast, stl, blk, fg3m, fgm, fga, ftm, fta, tov, oreb, dreb, pm, dk, dd,
td, usg_min (usage x minutes, so usage is minute-weighted). DK points use
`courtside.proj.dfs.rules.dk_points` (bonuses included).

Leaders qualifier (`qualifier`):
  early season (no team has played 10 games yet): >= 2 games and >= 20 min per game;
  later: games played >= 40% of his team's games (team = the team of his latest game).
  Per 36 adds >= 15 min per game; TS% and usage add >= 20 min per game. Totals and
  double-doubles have no qualifier (volume is the point).
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import timedelta

from . import db as sdb
from . import seasons as S

SUM_KEYS = ("gp", "min", "pts", "reb", "ast", "stl", "blk", "fg3m", "fgm", "fga", "ftm", "fta",
            "tov", "oreb", "dreb", "pm", "dk", "dd", "td", "usg_min")
WINDOWS = {"all": None, "l10": 10, "l5": 5}
EARLY_TEAM_GP = 10            # before any team has 10 games: the early-season qualifier
EARLY_MIN_GP, EARLY_MIN_MPG = 2, 20.0
LATE_SHARE = 0.40
PER36_MIN_MPG = 15.0
RATE_MIN_MPG = 20.0
TOP_N = 25

# key: (label, short, kind, rate) ; kind: num|pct|int ; rate stats ignore the mode
STATS = {
    "pts": ("Points", "Pts", "num", False),
    "reb": ("Rebounds", "Reb", "num", False),
    "ast": ("Assists", "Ast", "num", False),
    "stl": ("Steals", "Stl", "num", False),
    "blk": ("Blocks", "Blk", "num", False),
    "fg3m": ("3-pointers made", "3PM", "num", False),
    "dk": ("DK points", "DK", "num", False),
    "ts": ("True shooting", "TS%", "pct", True),
    "usg": ("Usage", "USG%", "pct", True),
    "min": ("Minutes", "Min", "num", False),
    "dd": ("Double-doubles", "DD", "int", True),
}
MODES = {"pg": "Per game", "tot": "Totals", "p36": "Per 36"}


def _zero() -> dict:
    return dict.fromkeys(SUM_KEYS, 0.0)


def _add(acc: dict, g: dict) -> None:
    for k in SUM_KEYS:
        acc[k] += g[k]


# ------------------------------------------------------------------ positions (DvP buckets)
DK_BUCKETS = ("PG", "SG", "SF", "PF", "C")
INDEX_BUCKETS = ("G", "F", "C")


def positions(conn: sqlite3.Connection, season: str) -> dict:
    """{basis: 'dk'|'index', buckets, pos: {player_id: bucket}}.

    'dk': the season has classic DK salaries (`dfs_salaries`, game_type_id 70): each player's
    first listed position on his latest salary row, DK ids mapped to NBA ids with
    `courtside.proj.ids.nba_ids` (the `player_ids` table the DFS code uses).
    'index': older seasons, `player_index.position` bucketed G / F / C by its first part
    ('G-F' -> G, 'F-C' -> F, 'C-F' -> C). The index is today's snapshot, not that season's.
    """
    lo, hi = S.date_range(season)
    pos: dict[int, str] = {}
    try:
        rows = conn.execute(
            "SELECT site_player_id, positions, MAX(captured_at) FROM dfs_salaries "
            "WHERE game_type_id=70 AND captured_at>=? AND captured_at<? AND positions<>'' "
            "GROUP BY site_player_id", (lo.isoformat(), (hi + timedelta(days=1)).isoformat()),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    if rows:
        from ..proj import ids as IDS
        dk2nba = IDS.nba_ids(conn, IDS.DK)
        for sid, p, _ in rows:
            nba = dk2nba.get(str(sid))
            first = (p or "").split("/")[0].strip().upper()
            if nba is not None and first in DK_BUCKETS:
                pos[nba] = first
        if pos:
            return {"basis": "dk", "buckets": DK_BUCKETS, "pos": pos}
    try:
        for pid, p in conn.execute("SELECT player_id, position FROM player_index "
                                   "WHERE position IS NOT NULL AND position<>''"):
            b = p.strip().upper()[:1]
            if b in INDEX_BUCKETS:
                pos[int(pid)] = b
    except sqlite3.Error:
        pass
    return {"basis": "index", "buckets": INDEX_BUCKETS, "pos": pos}


# ------------------------------------------------------------------ the season pass
_SQL = """
SELECT p.game_id, p.player_id, p.player_name, p.team, p.game_date, p.min, p.pts, p.reb, p.ast,
       p.stl, p.blk, p.fg3m, p.fgm, p.fga, p.ftm, p.fta, p.tov, p.oreb, p.dreb, p.plus_minus,
       a.usg_pct, t.opp
FROM player_games p
LEFT JOIN player_games_adv a ON a.game_id=p.game_id AND a.player_id=p.player_id
LEFT JOIN team_games t ON t.game_id=p.game_id AND t.team=p.team
WHERE p.season=? AND p.season_type=?
ORDER BY p.game_date, p.game_id
"""


def _f(x) -> float:
    return 0.0 if x is None else float(x)


def compute_pack(conn: sqlite3.Connection, season: str, season_type: str) -> dict:
    from ..proj.dfs.rules import double_double_cats, dk_points

    posmap = positions(conn, season)
    pos, buckets = posmap["pos"], posmap["buckets"]
    games: dict[int, list[dict]] = defaultdict(list)
    meta: dict[int, dict] = {}
    roster: dict[tuple[str, int], dict] = {}
    # defending team -> game_id -> [date, {bucket: dk}]
    allowed: dict[str, dict[str, list]] = defaultdict(dict)
    cover = [0.0, 0.0]                  # opponent minutes with / without a known position
    for r in conn.execute(_SQL, (season, season_type)):
        mins = _f(r["min"])
        pid = int(r["player_id"])
        dk = dk_points(r)
        cats = double_double_cats(r)
        if r["opp"]:
            slot = allowed[r["opp"]].setdefault(r["game_id"], [r["game_date"], {}])
            b = pos.get(pid)
            if b is not None:
                slot[1][b] = slot[1].get(b, 0.0) + dk
                cover[0] += mins
            elif mins > 0:
                cover[1] += mins
        m = meta.setdefault(pid, {"name": r["player_name"], "team": r["team"], "last": ""})
        if (r["game_date"] or "") >= m["last"]:
            m.update(name=r["player_name"] or m["name"], team=r["team"], last=r["game_date"] or "")
        if mins <= 0:
            continue
        g = {"gp": 1.0, "min": mins, "pts": _f(r["pts"]), "reb": _f(r["reb"]),
             "ast": _f(r["ast"]), "stl": _f(r["stl"]), "blk": _f(r["blk"]),
             "fg3m": _f(r["fg3m"]), "fgm": _f(r["fgm"]), "fga": _f(r["fga"]),
             "ftm": _f(r["ftm"]), "fta": _f(r["fta"]), "tov": _f(r["tov"]),
             "oreb": _f(r["oreb"]), "dreb": _f(r["dreb"]), "pm": _f(r["plus_minus"]),
             "dk": dk, "dd": 1.0 if cats >= 2 else 0.0, "td": 1.0 if cats >= 3 else 0.0,
             "usg_min": _f(r["usg_pct"]) * mins}
        games[pid].append(g)
        acc = roster.setdefault((r["team"], pid), _zero())
        _add(acc, g)
    players = {}
    for pid, gl in games.items():
        wins = {}
        for w, n in WINDOWS.items():
            acc = _zero()
            for g in (gl if n is None else gl[-n:]):
                _add(acc, g)
            wins[w] = acc
        m = meta[pid]
        players[pid] = {"name": m["name"], "team": m["team"], "w": wins}
    names = {pid: m["name"] for pid, m in meta.items()}
    return {"players": players, "roster": roster, "names": names,
            "dvp": _dvp(allowed, buckets), "basis": posmap["basis"], "buckets": buckets,
            "coverage": (cover[0] / (cover[0] + cover[1])) if sum(cover) else None}


def _dvp(allowed: dict, buckets: tuple) -> dict:
    """{team: {gp, all: {bucket: dk/g}, l10: {...}, rk: {bucket: rank}, rk10: {...}}}.
    Rank 1 = allows the MOST DK points per game (the best DFS matchup)."""
    out: dict[str, dict] = {}
    for team, per_game in allowed.items():
        gl = sorted(per_game.items(), key=lambda kv: (kv[1][0] or "", kv[0]))
        row = {"gp": len(gl)}
        for key, sel in (("all", gl), ("l10", gl[-10:])):
            n = len(sel)
            row[key] = {b: (sum(g[1][1].get(b, 0.0) for g in sel) / n if n else None)
                        for b in buckets}
            row[key]["tot"] = (sum(sum(g[1][1].values()) for g in sel) / n) if n else None
        out[team] = row
    for key, rk in (("all", "rk"), ("l10", "rk10")):
        for b in buckets + ("tot",):
            ranks = rank_values({t: r[key][b] for t, r in out.items()}, higher_better=True)
            for t, r in out.items():
                r.setdefault(rk, {})[b] = ranks.get(t)
    return out


def rank_values(values: dict, higher_better: bool = True) -> dict:
    """{key: rank} with rank 1 the best; ties share the best rank ("1224"); None unranked."""
    vals = [(k, v) for k, v in values.items() if v is not None]
    vals.sort(key=lambda kv: kv[1], reverse=higher_better)
    out, prev, prev_rank = {}, None, 0
    for i, (k, v) in enumerate(vals, start=1):
        if prev is None or v != prev:
            prev_rank = i
        out[k] = prev_rank
        prev = v
    return out


def pack(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """The cached season pass (see the module docstring)."""
    return sdb.cached(("stats.season_pack", season, season_type),
                      lambda c: compute_pack(c, season, season_type), conn, domain="box")


# ------------------------------------------------------------------ derived lines
def line(s: dict, mode: str = "pg") -> dict:
    """Display values from sums: counting stats per game ('pg'), totals ('tot') or per 36
    ('p36'); TS%, usage, FG% and double-doubles (a count) ignore the mode."""
    gp, mins = s["gp"], s["min"]
    if mode == "tot":
        div = 1.0
    elif mode == "p36":
        div = (mins / 36.0) if mins else 0.0
    else:
        div = gp
    out = {"gp": int(gp), "mpg": (mins / gp) if gp else None, "dd": int(s["dd"]),
           "td": int(s["td"])}
    for k in ("min", "pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "dk", "pm", "fga"):
        out[k] = (s[k] / div) if div else None
    if mode == "p36":
        out["min"] = (mins / gp) if gp else None          # minutes stay per game
    tsa = 2.0 * (s["fga"] + 0.44 * s["fta"])
    out["ts"] = (s["pts"] / tsa) if tsa else None
    out["fg"] = (s["fgm"] / s["fga"]) if s["fga"] else None
    out["usg"] = (s["usg_min"] / mins) if mins else None
    return out


def qualifier(stat: str, mode: str, max_team_gp: int) -> dict:
    """{early, text, fn(sums_all, team_gp) -> bool}: see the module docstring."""
    early = max_team_gp < EARLY_TEAM_GP
    rate = stat in ("ts", "usg")
    if mode == "tot" or stat == "dd":
        return {"early": early, "text": "No minimum (totals)", "fn": lambda s, tg: s["gp"] >= 1}
    if early:
        need_mpg = max(EARLY_MIN_MPG, RATE_MIN_MPG if rate else 0)
        text = f"Min {EARLY_MIN_GP} games and {need_mpg:.0f} min per game (early season)"
        return {"early": True, "text": text,
                "fn": lambda s, tg: s["gp"] >= EARLY_MIN_GP and s["min"] >= need_mpg * s["gp"]}
    need_mpg = RATE_MIN_MPG if rate else (PER36_MIN_MPG if mode == "p36" else 0.0)
    text = f"Played {LATE_SHARE:.0%} of his team's games"
    if need_mpg:
        text += f" and {need_mpg:.0f}+ min per game"

    def fn(s, tg):
        return (s["gp"] >= math.ceil(LATE_SHARE * (tg or 0) - 1e-9) and s["gp"] >= 1
                and s["min"] >= need_mpg * s["gp"])
    return {"early": False, "text": text, "fn": fn}


def team_gp(conn: sqlite3.Connection, season: str, season_type: str) -> dict[str, int]:
    """{team: games played} (cheap; cached)."""
    def calc(c):
        return {r[0]: int(r[1]) for r in c.execute(
            "SELECT team, COUNT(*) FROM team_games WHERE season=? AND season_type=? "
            "GROUP BY team", (season, season_type))}
    return sdb.cached(("stats.team_gp", season, season_type), calc, conn, domain="box")


def leaders(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR,
            stat: str = "pts", mode: str = "pg", window: str = "all", n: int = TOP_N) -> dict:
    """{rows: [{rank, player_id, name, team, gp, value, ...line}], qualified, rule, max}.

    The qualifier is applied to the player's whole season (all games), the value comes from
    the chosen window (all / last 10 / last 5 games played)."""
    stat = stat if stat in STATS else "pts"
    mode = mode if mode in MODES else "pg"
    window = window if window in WINDOWS else "all"
    key = ("stats.leaders", season, season_type, stat, mode, window, n)

    def calc(c):
        pk = pack(c, season, season_type)
        tgp = team_gp(c, season, season_type)
        q = qualifier(stat, mode, max(tgp.values(), default=0))
        rows = []
        for pid, p in pk["players"].items():
            if not q["fn"](p["w"]["all"], tgp.get(p["team"], 0)):
                continue
            ln = line(p["w"][window], mode)
            v = ln.get(stat)
            if v is None:
                continue
            rows.append({"player_id": pid, "name": p["name"], "team": p["team"],
                         "value": v, **ln})
        qualified = len(rows)
        rows.sort(key=lambda r: (-r["value"], r["name"] or ""))
        rows = rows[:n]
        prev, prev_rank = None, 0
        for i, r in enumerate(rows, start=1):
            if prev is None or r["value"] != prev:
                prev_rank = i
            r["rank"] = prev_rank
            prev = r["value"]
        return {"rows": rows, "qualified": qualified, "rule": q["text"], "early": q["early"],
                "max": rows[0]["value"] if rows else None, "stat": stat, "mode": mode,
                "window": window, "players": len(pk["players"])}
    return sdb.cached(key, calc, conn, domain="box")
