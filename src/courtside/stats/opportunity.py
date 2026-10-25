"""Opportunity vs conversion: how many chances a player had and how many he turned into the stat.

"LeBron got 8 rebounds on 10 chances" is 80% conversion. If his own season-long rate is 60%, he
"should" have had 6, and the extra 2 probably won't hold up. Pure functions on a (read-only)
`sqlite3.Connection`; plain dicts out; a missing table / a season without tracking gives zeros
and None rates (the page's "not collected yet" state), never an error.

  rebound_opportunity(conn, pid, game_ids)   REB per rebound chance vs his season and the league
  assist_opportunity(conn, pid, game_ids)    AST per potential assist, per pass; touches, passes
  shot_quality(conn, pid=, game_ids=, team=) expected vs actual FG% and points (shot_xfg)
  defense_activity(conn, pid, game_ids)      deflections, contests, rim FG% allowed
  regression_flags(conn, season, last_n)     league "running hot / running cold" list
  league_conversions(conn, season)           league rates by position group (cache it)
  player_opportunity(conn, pid, season, type) the player page's "Chances" panel: every block
                                             for last 5 / last 10 / season + baselines + flags
                                             (cached; dict shape in its docstring)

`game_ids` is any iterable of game ids (e.g. the last 10, the games vs one team); None means
every game of `season`. The season of the call is `season` or, if not given, the season of the
latest of the games.

Sources (per player-game)
-------------------------
  player_games         REB / OREB / DREB, AST, STL, BLK, FGM, FGA, PTS (official box)
  player_tracking      reb_chances_off / _def / _total (the player was within ~3.5 ft of the
                       rebound), passes, touches, secondary_ast, ft_ast, def_rim_fgm / _fga
                       (shots at the rim with him defending), cont / uncont FG
  player_hustle        deflections, contested_shots (2pt / 3pt), charges_drawn, loose_total,
                       box_outs, screen_assists
  player_tracking_ext  pot_ast (a pass leading to a shot that would be an assist if made, incl.
                       a shooting foul), ast_pts_created, drives, paint_touches, ... (nba.com
                       tracking dashboards per day; see analytics/tracking_ext_store.py)
  shots + shot_xfg     per attempt: zone, distance, 2/3, shot family -> xfg (analytics/xfg_store)
A conversion only uses games where its denominator was collected (a game without tracking does
not count as 0 chances). Plausibility guard: a game's pot_ast above POT_AST_MAX (per game), or
above POT_AST_RATIO x his assists once it reaches POT_AST_RATIO_MIN, is treated as not collected
(the feed has shipped such rows early in a season); `pot_ast_excluded` counts them.

Formulas
--------
Conversion of a stat on its opportunities, over a set of games:
    conv = SUM(stat) / SUM(opportunities)
    reb: REB / reb_chances_total   oreb: OREB / reb_chances_off   dreb: DREB / reb_chances_def
    ast_pot: AST / pot_ast         ast_pass: AST / passes
    rim_def: def_rim_fgm / def_rim_fga (FG% allowed at the rim; lower is better)
    stl_defl: STL / deflections
League rate by position group g (G / F / C: first listed position, team_rosters of the season,
else player_index, as proj/state.load_positions): p_g = SUM over the group's players of the stat /
SUM of the opportunities, whole season (all season types). 'all' = every player.
His baseline (season-long, shrunk to his group so a thin season does not swing it):
    p0 = (SUM_season(stat) + K * p_g) / (SUM_season(opps) + K),  K = K_CONV[metric] opportunities
Expected stat on the selected games and the part that "probably won't hold up":
    expected = opps_selected * p0          over_expected = stat_selected - expected

Shot quality (shot_xfg, per season; league expected FG% = actual by construction):
    xFGM = SUM(xfg_i)                 xFG% = xFGM / FGA
    PTS  = SUM(made_i * v_i)          xPTS = SUM(xfg_i * v_i), v = 3 for a three, else 2
    (field goals only: free throws are not in the shots table)
    shot making (FG%) = FG% - xFG%
    shot making (pts) = PTS - xPTS;   per shot = (PTS - xPTS) / FGA;   per game = / games
A team's own shots use team = T; with allowed=True, the shots of T's opponents in those games.

Regression flags (regression_flags): per player and metric, the window is his last `last_n`
games of the season that have the metric's denominator; his baseline is the rest of his season
(the games before the window), shrunk to his group's league rate:
    p0 = (n_base + K * p_g) / (d_base + K)
    z  = (n_w - d_w * p0) / sqrt(d_w * p0 * (1 - p0))
    z* = z * d_w / (d_w + K)          (shrunk z: sample-size guard)
For field goals (metric 'fg') the window's expected makes use the xFG model plus his own shooting
skill from the baseline games, shrunk to 0:
    skill = (FGM_base - xFGM_base) / (FGA_base + K_SKILL)
    E = SUM_window(xfg_i + skill)    Var = SUM_window(p_i (1 - p_i)), p_i = clip(xfg_i + skill)
    z = (FGM_w - E) / sqrt(Var),     z* = z * FGA_w / (FGA_w + K_CONV['fg'])
A player is flagged when d_w >= MIN_WINDOW[metric] and |z*| >= z_min; 'hot' = above his
baseline, 'cold' = below (for rim_def, "hot" means opponents scoring more on him: it is listed
with the sign of the stat, and `good` tells the reader whether that is good for the player).
"""

from __future__ import annotations

import math
import sqlite3
from typing import Iterable

from ..analytics.xfg_store import XFGTable

GROUPS = ("G", "F", "C")
# metric: (stat column, opportunity column, source of the opportunity)
CONV = {
    "reb": ("reb", "reb_chances_total"),
    "oreb": ("oreb", "reb_chances_off"),
    "dreb": ("dreb", "reb_chances_def"),
    "ast_pot": ("ast", "pot_ast"),
    "ast_pass": ("ast", "passes"),
    "rim_def": ("def_rim_fgm", "def_rim_fga"),
    "stl_defl": ("stl", "deflections"),
}
K_CONV = {"reb": 60.0, "oreb": 30.0, "dreb": 60.0, "ast_pot": 40.0, "ast_pass": 300.0,
          "rim_def": 40.0, "stl_defl": 30.0, "fg": 60.0}
K_SKILL = 150.0
MIN_WINDOW = {"reb": 40, "oreb": 15, "dreb": 30, "ast_pot": 25, "ast_pass": 150, "rim_def": 20,
              "stl_defl": 15, "fg": 50}
FLAG_METRICS = ("reb", "ast_pot", "fg")
HIGHER_IS_GOOD = {"rim_def": False}
POT_AST_MAX, POT_AST_RATIO, POT_AST_RATIO_MIN = 30, 6.0, 10

_PG = ("reb", "oreb", "dreb", "ast", "stl", "blk", "fgm", "fga", "fg3m", "fg3a", "pts", "min")
_T = ("reb_chances_off", "reb_chances_def", "reb_chances_total", "touches", "passes",
      "secondary_ast", "ft_ast", "cont_fgm", "cont_fga", "uncont_fgm", "uncont_fga",
      "def_rim_fgm", "def_rim_fga")
_H = ("deflections", "contested_shots", "contested_2pt", "contested_3pt", "charges_drawn",
      "loose_total", "box_outs", "screen_assists")
_X = ("pot_ast", "ast_pts_created", "passes_made", "passes_received", "drives", "drive_pts",
      "paint_touches", "elbow_touches", "post_touches", "time_of_poss")


# ---------------------------------------------------------------------------------- helpers

def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _ids(game_ids: Iterable[str] | None) -> list[str] | None:
    return None if game_ids is None else sorted({str(g) for g in game_ids})


def implausible_pot_ast(pot_ast, ast) -> bool:
    """True for a potential-assist count the feed cannot have meant (see the module docstring)."""
    if pot_ast is None:
        return False
    return pot_ast > POT_AST_MAX or (pot_ast >= POT_AST_RATIO_MIN
                                     and pot_ast > POT_AST_RATIO * (ast or 0))


def _rows(conn: sqlite3.Connection, *, season: str | None = None, player_id: int | None = None,
          game_ids: list[str] | None = None, season_type: str | None = None) -> list[dict]:
    """One dict per player-game of player_games with the tracking / hustle / ext columns
    (None when that feed has no row for the game; an implausible pot_ast is None too, with
    `pot_ast_suspect` set)."""
    have = _tables(conn)
    if "player_games" not in have:
        return []
    sel = ["pg.game_id", "pg.player_id", "pg.player_name", "pg.team", "pg.season",
           "pg.game_date"] + [f"pg.{c}" for c in _PG]
    joins = []
    for alias, table, cols in (("t", "player_tracking", _T), ("h", "player_hustle", _H),
                               ("x", "player_tracking_ext", _X)):
        if table in have:
            sel += [f"{alias}.{c} AS {c}" for c in cols]
            joins.append(f"LEFT JOIN {table} {alias} ON {alias}.game_id = pg.game_id "
                         f"AND {alias}.player_id = pg.player_id")
        else:
            sel += [f"NULL AS {c}" for c in cols]
    where, args = [], []
    if season is not None:
        where.append("pg.season = ?")
        args.append(season)
    if player_id is not None:
        where.append("pg.player_id = ?")
        args.append(int(player_id))
    if season_type is not None:
        where.append("pg.season_type = ?")
        args.append(season_type)
    if game_ids is not None:
        if not game_ids:
            return []
        where.append(f"pg.game_id IN ({','.join('?' * len(game_ids))})")
        args += game_ids
    sql = (f"SELECT {', '.join(sel)} FROM player_games pg {' '.join(joins)}"
           + (f" WHERE {' AND '.join(where)}" if where else ""))
    cur = conn.execute(sql, args)
    names = [d[0] for d in cur.description]
    out = [dict(zip(names, r)) for r in cur.fetchall()]
    for r in out:
        if implausible_pot_ast(r.get("pot_ast"), r.get("ast")):
            r["pot_ast"], r["pot_ast_suspect"] = None, True
    return out


def _season_for(conn: sqlite3.Connection, player_id: int | None, game_ids: list[str] | None,
                season: str | None) -> str | None:
    if season or not game_ids:
        return season
    q = ",".join("?" * len(game_ids))
    try:
        r = conn.execute(f"SELECT season FROM player_games WHERE game_id IN ({q}) "
                         f"ORDER BY game_date DESC LIMIT 1", game_ids).fetchone()
        if r is None and "shots" in _tables(conn):
            r = conn.execute(f"SELECT season FROM shots WHERE game_id IN ({q}) "
                             f"ORDER BY game_date DESC LIMIT 1", game_ids).fetchone()
    except sqlite3.OperationalError:
        return None
    return r[0] if r else None


def _grp(pos: str | None) -> str | None:
    p = (pos or "").strip()[:1].upper()
    return p if p in GROUPS else None


def positions(conn: sqlite3.Connection, season: str | None,
              player_id: int | None = None) -> dict[int, str]:
    """{player_id: 'G' | 'F' | 'C'}: team_rosters of the season, else player_index (one player
    when `player_id` is given)."""
    have = _tables(conn)
    out: dict[int, str] = {}
    one = "" if player_id is None else " WHERE player_id = ?"
    arg = () if player_id is None else (int(player_id),)
    if "player_index" in have:
        for pid, pos in conn.execute(f"SELECT player_id, position FROM player_index{one}", arg):
            g = _grp(pos)
            if g:
                out[pid] = g
    if "team_rosters" in have and season:
        for pid, pos in conn.execute("SELECT player_id, position FROM team_rosters "
                                     "WHERE season = ?" + one.replace(" WHERE", " AND"),
                                     (season, *arg)):
            g = _grp(pos)
            if g:
                out[pid] = g
    return out


def _sum(rows: list[dict], num: str, den: str) -> tuple[float, float, int]:
    """(SUM stat, SUM opportunities, games) over the rows where the opportunity is known."""
    n = d = 0.0
    g = 0
    for r in rows:
        if r.get(den) is None:
            continue
        n += r.get(num) or 0
        d += r[den] or 0
        g += 1
    return n, d, g


def _rate(n: float, d: float) -> float | None:
    return n / d if d else None


def _r(v: float | None, k: int = 3) -> float | None:
    return None if v is None else round(v, k)


def league_conversions(conn: sqlite3.Connection, season: str) -> dict:
    """{metric: {'G' | 'F' | 'C' | 'all': {'n', 'd', 'conv'}}} for the season (all types).
    Heavy-ish (one pass over the season's player-games): cache it per season."""
    rows = _rows(conn, season=season)
    pos = positions(conn, season)
    out: dict = {}
    for m, (num, den) in CONV.items():
        acc = {g: [0.0, 0.0] for g in (*GROUPS, "all")}
        for r in rows:
            if r.get(den) is None:
                continue
            for g in (pos.get(r["player_id"]), "all"):
                if g:
                    acc[g][0] += r.get(num) or 0
                    acc[g][1] += r[den] or 0
        out[m] = {g: {"n": a[0], "d": a[1], "conv": _rate(a[0], a[1])} for g, a in acc.items()}
    return out


def _prior(league: dict, metric: str, grp: str | None) -> float | None:
    lm = league.get(metric) or {}
    v = (lm.get(grp) or {}).get("conv") if grp else None
    return v if v is not None else (lm.get("all") or {}).get("conv")


def _conversion_block(sel: list[dict], season_rows: list[dict], metric: str, league: dict,
                      grp: str | None) -> dict:
    num, den = CONV[metric]
    n, d, g = _sum(sel, num, den)
    ns, ds, gs = _sum(season_rows, num, den)
    prior = _prior(league, metric, grp)
    k = K_CONV[metric]
    base = ((ns + k * prior) / (ds + k)) if prior is not None else _rate(ns, ds)
    exp = d * base if base is not None and g else None
    return {"stat": n, "opps": d, "games": g, "conv": _r(_rate(n, d)),
            "season_stat": ns, "season_opps": ds, "season_games": gs,
            "season_conv": _r(_rate(ns, ds)), "league_conv": _r(prior),
            "baseline_conv": _r(base), "expected": _r(exp, 2),
            "over_expected": _r(n - exp, 2) if exp is not None else None}


def _player_ctx(conn, player_id, game_ids, season, league):
    gids = _ids(game_ids)
    season = _season_for(conn, player_id, gids, season)
    try:
        season_rows = _rows(conn, season=season, player_id=player_id) if season else []
        if gids is None:
            sel = season_rows
        else:
            sel = _rows(conn, player_id=player_id, game_ids=gids)
        league = league if league is not None else (
            league_conversions(conn, season) if season else {})
        grp = positions(conn, season, player_id).get(int(player_id))
    except sqlite3.OperationalError:
        return gids, season, [], [], {}, None
    return gids, season, sel, season_rows, league, grp


# ---------------------------------------------------------------------------- the helpers

def rebound_opportunity(conn: sqlite3.Connection, player_id: int,
                        game_ids: Iterable[str] | None = None, *, season: str | None = None,
                        league: dict | None = None) -> dict:
    """Rebounds vs rebound chances on the games, with his season-long conversion, the league's
    for his position group, and the rebounds expected at his (shrunk) baseline.
    `league` = a cached league_conversions(conn, season) to skip that pass."""
    gids, season, sel, srows, league, grp = _player_ctx(conn, player_id, game_ids, season, league)
    total = _conversion_block(sel, srows, "reb", league, grp)
    return {"player_id": int(player_id), "season": season, "position": grp,
            "games_selected": len(sel), "games": total["games"],
            "reb": total["stat"], "chances": total["opps"], "conv": total["conv"],
            "season_conv": total["season_conv"], "season_chances": total["season_opps"],
            "league_conv": total["league_conv"],
            "league_conv_by_pos": {g: _r((league.get("reb") or {}).get(g, {}).get("conv"))
                                   for g in (*GROUPS, "all")},
            "baseline_conv": total["baseline_conv"], "expected_reb": total["expected"],
            "reb_over_expected": total["over_expected"],
            "off": _conversion_block(sel, srows, "oreb", league, grp),
            "def": _conversion_block(sel, srows, "dreb", league, grp)}


def assist_opportunity(conn: sqlite3.Connection, player_id: int,
                       game_ids: Iterable[str] | None = None, *, season: str | None = None,
                       league: dict | None = None) -> dict:
    """Assists vs potential assists (when collected) and vs passes; touches, secondary and FT
    assists, assist points created."""
    gids, season, sel, srows, league, grp = _player_ctx(conn, player_id, game_ids, season, league)
    pot = _conversion_block(sel, srows, "ast_pot", league, grp)
    per_pass = _conversion_block(sel, srows, "ast_pass", league, grp)

    def tot(c):
        vals = [r.get(c) for r in sel if r.get(c) is not None]
        return sum(vals) if vals else None

    ast = sum(r.get("ast") or 0 for r in sel)
    return {"player_id": int(player_id), "season": season, "position": grp,
            "games": len(sel), "ast": ast,
            "pot_ast": pot["opps"] if pot["games"] else None, "pot_ast_games": pot["games"],
            "pot_ast_excluded": sum(1 for r in sel if r.get("pot_ast_suspect")),
            "ast_per_pot_ast": pot["conv"], "season_ast_per_pot_ast": pot["season_conv"],
            "league_ast_per_pot_ast": pot["league_conv"], "expected_ast": pot["expected"],
            "ast_over_expected": pot["over_expected"],
            "passes": tot("passes"), "touches": tot("touches"),
            "secondary_ast": tot("secondary_ast"), "ft_ast": tot("ft_ast"),
            "ast_pts_created": tot("ast_pts_created"), "drives": tot("drives"),
            "paint_touches": tot("paint_touches"),
            "ast_per_pass": per_pass["conv"], "season_ast_per_pass": per_pass["season_conv"],
            "league_ast_per_pass": per_pass["league_conv"],
            "potential": pot, "per_pass": per_pass}


def defense_activity(conn: sqlite3.Connection, player_id: int,
                     game_ids: Iterable[str] | None = None, *, season: str | None = None,
                     league: dict | None = None) -> dict:
    """Deflections, contested shots, charges, loose balls, box-outs, STL / BLK, and the FG% he
    allowed at the rim vs his season and the league's for his group."""
    gids, season, sel, srows, league, grp = _player_ctx(conn, player_id, game_ids, season, league)
    hustle = [r for r in sel if r.get("deflections") is not None]

    def tot(rows, c):
        return sum(r.get(c) or 0 for r in rows)

    rim = _conversion_block(sel, srows, "rim_def", league, grp)
    n_h = len(hustle)
    out = {"player_id": int(player_id), "season": season, "position": grp, "games": len(sel),
           "hustle_games": n_h, "stl": tot(sel, "stl"), "blk": tot(sel, "blk")}
    for c in ("deflections", "contested_shots", "contested_2pt", "contested_3pt",
              "charges_drawn", "loose_total", "box_outs"):
        out[c] = tot(hustle, c) if n_h else None
        out[f"{c}_pg"] = _r(out[c] / n_h, 2) if n_h else None
    out.update({"rim_fgm_allowed": rim["stat"], "rim_fga_defended": rim["opps"],
                "rim_games": rim["games"], "rim_fg_allowed": rim["conv"],
                "season_rim_fg_allowed": rim["season_conv"],
                "league_rim_fg_allowed": rim["league_conv"],
                "rim_fgm_vs_expected": rim["over_expected"],
                "stl_per_deflection": _conversion_block(sel, srows, "stl_defl", league,
                                                        grp)["conv"]})
    return out


def _shot_rows(conn, *, player_id=None, team=None, game_ids=None, season=None, allowed=False):
    have = _tables(conn)
    if "shots" not in have:
        return []
    where, args = ["made IS NOT NULL"], []
    if game_ids is not None:
        if not game_ids:
            return []
        where.append(f"game_id IN ({','.join('?' * len(game_ids))})")
        args += game_ids
    if season is not None:
        where.append("season = ?")
        args.append(season)
    if player_id is not None:
        where.append("player_id = ?")
        args.append(int(player_id))
    if team is not None:
        if allowed:
            where.append("team <> ? AND game_id IN (SELECT game_id FROM shots WHERE team = ?"
                         + (" AND season = ?" if season else "") + ")")
            args += [team, team] + ([season] if season else [])
        else:
            where.append("team = ?")
            args.append(team)
    return conn.execute(
        "SELECT game_id, season, zone, distance, shot_type, sub_type, made FROM shots WHERE "
        + " AND ".join(where), args).fetchall()


def _xfg_lookup(table: XFGTable | None):
    """table.xfg memoised per (season, zone, distance, shot type, sub type): a season has
    ~230k shots but only ~8k distinct cells, and XFGTable.xfg costs ~2 us a call."""
    if table is None:
        return lambda *k: None
    memo: dict[tuple, float | None] = {}
    get, xfg = memo.get, table.xfg

    def look(*k):
        x = get(k, memo)
        if x is memo:
            x = memo[k] = xfg(*k)
        return x
    return look


def _quality(rows, table: XFGTable | None) -> dict:
    agg = {"fga": 0, "fgm": 0, "xfgm": 0.0, "pts": 0, "xpts": 0.0, "unmodelled": 0}
    zones: dict[str, dict] = {}
    kinds: dict[str, dict] = {}
    games = set()
    xfg = _xfg_lookup(table)
    for gid, season, zone, dist, stype, sub, made in rows:
        x = xfg(season, zone, dist, stype, sub)
        if x is None:                    # no shot_xfg fit for the season yet
            agg["unmodelled"] += 1
            continue
        v = 3 if stype == "3pt" else 2
        games.add(gid)
        for a in (agg, zones.setdefault(zone or "Unknown", {"fga": 0, "fgm": 0, "xfgm": 0.0,
                                                             "pts": 0, "xpts": 0.0}),
                  kinds.setdefault(stype or "2pt", {"fga": 0, "fgm": 0, "xfgm": 0.0, "pts": 0,
                                                    "xpts": 0.0})):
            a["fga"] += 1
            a["fgm"] += made
            a["xfgm"] += x
            a["pts"] += made * v
            a["xpts"] += x * v

    def fin(a, n_games=None):
        fga = a["fga"]
        o = {"fga": fga, "fgm": a["fgm"], "fg_pct": _r(_rate(a["fgm"], fga)),
             "xfgm": _r(a["xfgm"], 2), "xfg_pct": _r(_rate(a["xfgm"], fga)),
             "pts": a["pts"], "xpts": _r(a["xpts"], 2),
             "making_fg_pct": _r(_rate(a["fgm"] - a["xfgm"], fga)),
             "making_pts": _r(a["pts"] - a["xpts"], 2),
             "making_per_shot": _r(_rate(a["pts"] - a["xpts"], fga)),
             "pps": _r(_rate(a["pts"], fga)), "xpps": _r(_rate(a["xpts"], fga))}
        if n_games is not None:
            o["games"] = n_games
            o["making_per_game"] = _r(_rate(a["pts"] - a["xpts"], n_games), 2)
        return o

    out = fin(agg, len(games))
    out["unmodelled"] = agg["unmodelled"]
    out["by_zone"] = [{"zone": z, **fin(a)} for z, a in sorted(zones.items(),
                                                               key=lambda kv: -kv[1]["fga"])]
    out["by_type"] = {k: fin(a) for k, a in sorted(kinds.items())}
    return out


def shot_quality(conn: sqlite3.Connection, player_id: int | None = None,
                 game_ids: Iterable[str] | None = None, *, team: str | None = None,
                 season: str | None = None, allowed: bool = False) -> dict:
    """Expected vs actual FG% and field-goal points on the games for a player (player_id) or a
    team (team; allowed=True: what its opponents shot). game_ids None = the whole `season`."""
    gids = _ids(game_ids)
    if gids is None and season is None:
        raise ValueError("give game_ids or season")
    try:
        rows = _shot_rows(conn, player_id=player_id, team=team, game_ids=gids,
                          season=season if gids is None else None, allowed=allowed)
        tab = XFGTable(conn, {r[1] for r in rows}) if "shot_xfg" in _tables(conn) else None
    except sqlite3.OperationalError:
        rows, tab = [], None
    out = _quality(rows, tab)
    out.update({"player_id": player_id, "team": team, "allowed": allowed,
                "season": season or (sorted({r[1] for r in rows})[-1] if rows else None),
                "model": "shot_xfg" if tab is not None and tab.seasons() else None})
    return out


# --------------------------------------------------------------------------- regression

def _flag(n_w, d_w, exp, var, k):
    if d_w <= 0 or var <= 0:
        return None
    z = (n_w - exp) / math.sqrt(var)
    return z, z * d_w / (d_w + k)


def _fg_by_game(conn, season, table: XFGTable) -> dict[tuple[int, str], list[float]]:
    """{(player_id, game_id): [FGA, FGM, xFGM]} for the season's shots."""
    out: dict[tuple[int, str], list[float]] = {}
    xfg = _xfg_lookup(table)
    for pid, gid, zone, dist, stype, sub, made in conn.execute(
            "SELECT player_id, game_id, zone, distance, shot_type, sub_type, made FROM shots "
            "WHERE season = ? AND made IS NOT NULL AND player_id IS NOT NULL", (season,)):
        x = xfg(season, zone, dist, stype, sub)
        if x is None:
            continue
        a = out.setdefault((pid, gid), [0, 0, 0.0, []])
        a[0] += 1
        a[1] += made
        a[2] += x
        a[3].append(x)
    return out


def regression_flags(conn: sqlite3.Connection, season: str, last_n: int = 10, *,
                     metrics: Iterable[str] = FLAG_METRICS, z_min: float = 1.5,
                     limit: int = 25, league: dict | None = None,
                     season_type: str | None = None) -> dict:
    """League "running hot / running cold": players whose last `last_n` games convert far above
    or below their own season baseline (shrunk z, see the module docstring). `season_type`
    keeps the window and the baseline games to one season type (None = every game).
    Returns {"season", "last_n", "hot": [...], "cold": [...]} sorted by |z*|, each item:
    player_id, name, team, position, metric, games, n, d, rate, baseline, expected, diff, z, z_shrunk."""
    metrics = tuple(metrics)
    try:
        rows = _rows(conn, season=season, season_type=season_type)
        league = league if league is not None else league_conversions(conn, season)
        pos = positions(conn, season)
        fg = {}
        if "fg" in metrics and {"shots", "shot_xfg"} <= _tables(conn):
            fg = _fg_by_game(conn, season, XFGTable(conn, [season]))
    except sqlite3.OperationalError:
        return {"season": season, "last_n": last_n, "hot": [], "cold": []}
    by_p: dict[int, list[dict]] = {}
    for r in rows:
        by_p.setdefault(r["player_id"], []).append(r)
    flags = []
    for pid, prs in by_p.items():
        prs.sort(key=lambda r: (r["game_date"] or "", r["game_id"]))
        grp = pos.get(pid)
        name, team = prs[-1]["player_name"], prs[-1]["team"]
        for m in metrics:
            k = K_CONV[m]
            if m == "fg":
                games = [(r, fg[(pid, r["game_id"])]) for r in prs if (pid, r["game_id"]) in fg]
                win, base = games[-last_n:], games[:-last_n]
                fga_w = sum(a[0] for _, a in win)
                if fga_w < MIN_WINDOW[m]:
                    continue
                fga_b = sum(a[0] for _, a in base)
                skill = (sum(a[1] for _, a in base) - sum(a[2] for _, a in base)) / (fga_b + K_SKILL)
                ps = [min(max(x + skill, 0.01), 0.99) for _, a in win for x in a[3]]
                exp, var = sum(ps), sum(p * (1 - p) for p in ps)
                n_w, d_w = sum(a[1] for _, a in win), fga_w
                baseline = exp / d_w
            else:
                num, den = CONV[m]
                games = [r for r in prs if r.get(den) is not None]
                win, base = games[-last_n:], games[:-last_n]
                n_w, d_w, _ = _sum(win, num, den)
                if d_w < MIN_WINDOW[m]:
                    continue
                n_b, d_b, _ = _sum(base, num, den)
                prior = _prior(league, m, grp)
                if prior is None:
                    continue
                baseline = (n_b + k * prior) / (d_b + k)
                exp, var = d_w * baseline, d_w * baseline * (1 - baseline)
            z = _flag(n_w, d_w, exp, var, k)
            if z is None or abs(z[1]) < z_min:
                continue
            flags.append({"player_id": pid, "name": name, "team": team, "position": grp,
                          "metric": m, "games": len(win), "n": n_w, "d": d_w,
                          "rate": _r(n_w / d_w), "baseline": _r(baseline),
                          "expected": _r(exp, 1), "diff": _r(n_w - exp, 1),
                          "z": _r(z[0], 2), "z_shrunk": _r(z[1], 2),
                          "good": (z[1] > 0) == HIGHER_IS_GOOD.get(m, True)})
    flags.sort(key=lambda f: -abs(f["z_shrunk"]))
    return {"season": season, "last_n": last_n,
            "hot": [f for f in flags if f["z_shrunk"] > 0][:limit],
            "cold": [f for f in flags if f["z_shrunk"] < 0][:limit]}


# ------------------------------------------------------------------------- player panel

WINDOWS = (("l5", 5), ("l10", 10), ("season", None))
_REB_W = ("games", "reb", "chances", "conv", "expected_reb", "reb_over_expected")
_AST_W = ("games", "ast", "pot_ast", "pot_ast_games", "pot_ast_excluded", "ast_per_pot_ast",
          "expected_ast",
          "ast_over_expected", "passes", "touches", "ast_per_pass", "secondary_ast", "ft_ast",
          "ast_pts_created", "drives", "paint_touches")
_SHOT_W = ("games", "fga", "fgm", "fg_pct", "xfgm", "xfg_pct", "pts", "xpts", "making_fg_pct",
           "making_pts", "making_per_shot", "making_per_game", "pps", "xpps", "by_type")
_DEF_W = ("games", "hustle_games", "deflections", "deflections_pg", "contested_shots",
          "contested_shots_pg", "contested_2pt", "contested_3pt", "charges_drawn", "loose_total",
          "box_outs", "stl", "blk", "rim_games", "rim_fga_defended", "rim_fgm_allowed",
          "rim_fg_allowed", "rim_fgm_vs_expected", "stl_per_deflection")


def _pg(v, games):
    return _r(v / games, 1) if v is not None and games else None


def player_opportunity(conn: sqlite3.Connection, player_id: int, season: str,
                       season_type: str = "Regular Season") -> dict | None:
    """Everything the player page's "Chances" panel shows for one player-season, cached per
    (player, season, season type) via stats.db.cached (league conversions and the league's
    regression flags are cached per season and shared by every player page).

    Windows are his last 5 / last 10 / all games of `season_type` (by date). His own baseline and
    the league's are whole-season (every season type) rates, as in the module docstring.
    None when he has no games of that season type, or none of them has tracking, hustle or shots.

    {
      "player_id", "season", "season_type", "position": "G" | "F" | "C" | None,
      "games": int,                              # his games of the season type
      "windows": ["l5", "l10", "season"],
      "window_games": {"l5": 5, "l10": 10, "season": 62},
      "has": {"tracking", "hustle", "pot_ast", "shots"}: bool,   # any row in the season window
      "rebounds": {
        "l5" | "l10" | "season": {games, reb, chances, chances_pg, conv, expected_reb,
                                  reb_over_expected, off: {stat, opps, conv},
                                  def: {stat, opps, conv}},
        "baseline": {"own": shrunk season conv, "own_raw": season conv,
                     "league": his group's conv, "league_all": every player's conv}},
      "assists": {
        "l5" | "l10" | "season": {games, ast, pot_ast, pot_ast_games, pot_ast_pg,
                                  ast_per_pot_ast, expected_ast, ast_over_expected, passes,
                                  passes_pg, touches, touches_pg, ast_per_pass, secondary_ast,
                                  ft_ast, ast_pts_created, drives, paint_touches},
        "baseline": {"own": AST / potential AST (season), "league": group,
                     "own_per_pass", "league_per_pass"}},
      "shooting": {
        "l5" | "l10" | "season": {games, fga, fgm, fg_pct, xfgm, xfg_pct, pts, xpts,
                                  making_fg_pct, making_pts, making_per_shot, making_per_game,
                                  pps, xpps, by_type: {"2pt" | "3pt": {fga, fgm, fg_pct,
                                  xfg_pct, pts, xpts, making_pts, ...}}},
        "baseline": {"own": his season FG% - xFG% (all types), "league_fg_pct": the league's
                     FG% (= its xFG% by construction), "league_pps"}},
      "defense": {
        "l5" | "l10" | "season": {games, hustle_games, deflections, deflections_pg,
                                  contested_shots, contested_shots_pg, contested_2pt,
                                  contested_3pt, charges_drawn, loose_total, box_outs, stl,
                                  blk, rim_games, rim_fga_defended, rim_fgm_allowed,
                                  rim_fg_allowed, rim_fgm_vs_expected, stl_per_deflection},
        "baseline": {"own": his season rim FG% allowed, "league": his group's}},
      "flags": [regression_flags items for him (last 10 games of `season_type`, metrics
                reb / ast_pot / fg)],
    }
    Rates are fractions (0.55 = 55%), None when the denominator was never collected."""
    from . import db as sdb

    def build(c: sqlite3.Connection):
        return _player_opportunity(c, int(player_id), season, season_type, sdb)

    return sdb.cached(("opp-player", int(player_id), season, season_type), build, conn,
                      domain="box")


# Shared league aggregates, cached once per season for every player page, the Chances leaders
# and the background warmer (stats/warm.py): one key and one domain each.
def league_cached(conn: sqlite3.Connection, season: str) -> dict:
    """league_conversions(season), cached (box-score domain)."""
    from . import db as sdb
    return sdb.cached(("opp-league", season), lambda c: league_conversions(c, season), conn,
                      domain="box")


def league_shot_summary(conn: sqlite3.Connection, season: str) -> dict:
    """{fg_pct, xfg_pct, pps, fga} of every modelled shot of the season: shot_quality(season=)
    restricted to those keys, aggregated per (zone, distance, type, sub type) cell in SQL
    (~8k groups instead of ~230k rows through Python)."""
    try:
        if not {"shots", "shot_xfg"} <= _tables(conn):
            raise sqlite3.OperationalError("no shots")
        rows = conn.execute(
            "SELECT zone, distance, shot_type, sub_type, COUNT(*), SUM(made) FROM shots "
            "WHERE made IS NOT NULL AND season = ? GROUP BY zone, distance, shot_type, sub_type",
            (season,)).fetchall()
        tab = XFGTable(conn, [season])
    except sqlite3.OperationalError:
        rows, tab = [], None
    fga = fgm = pts = 0
    xfgm = xpts = 0.0
    for zone, dist, stype, sub, n, made in rows:
        x = tab.xfg(season, zone, dist, stype, sub) if tab is not None else None
        if x is None:
            continue
        v = 3 if stype == "3pt" else 2
        fga += n
        fgm += made
        xfgm += n * x
        pts += made * v
        xpts += n * x * v
    return {"fga": fga, "fg_pct": _r(_rate(fgm, fga)), "xfg_pct": _r(_rate(xfgm, fga)),
            "pps": _r(_rate(pts, fga))}


def league_shots_cached(conn: sqlite3.Connection, season: str) -> dict | None:
    """The league's season FG% / xFG% / points per shot / FGA, cached (box-score domain)."""
    from . import db as sdb
    return sdb.cached(("opp-league-shots", season), lambda c: league_shot_summary(c, season),
                      conn, domain="box")


def flags_cached(conn: sqlite3.Connection, season: str, season_type: str,
                 league: dict | None = None) -> dict | None:
    """regression_flags(season, last 10, every player) for a season type, cached (box)."""
    from . import db as sdb

    def build(c):
        lg = league if league is not None else league_cached(c, season)
        return regression_flags(c, season, 10, limit=100000, league=lg, season_type=season_type)
    return sdb.cached(("opp-flags", season, 10, season_type), build, conn, domain="box")


def _player_opportunity(conn, pid, season, season_type, sdb) -> dict | None:
    try:
        gids = [r[0] for r in conn.execute(
            "SELECT game_id FROM player_games WHERE player_id=? AND season=? AND season_type=? "
            "ORDER BY game_date, game_id", (pid, season, season_type))]
    except sqlite3.OperationalError:
        return None
    if not gids:
        return None
    league = league_cached(conn, season)
    win = {name: (gids[-n:] if n else gids) for name, n in WINDOWS}
    full = win["season"]
    reb = {k: rebound_opportunity(conn, pid, g, season=season, league=league)
           for k, g in win.items()}
    ast = {k: assist_opportunity(conn, pid, g, season=season, league=league)
           for k, g in win.items()}
    dfn = {k: defense_activity(conn, pid, g, season=season, league=league)
           for k, g in win.items()}
    sht = {k: shot_quality(conn, pid, g) for k, g in win.items()}
    has = {"tracking": reb["season"]["games"] > 0, "hustle": dfn["season"]["hustle_games"] > 0,
           "pot_ast": ast["season"]["pot_ast_games"] > 0, "shots": sht["season"]["fga"] > 0}
    if not any(has.values()):
        return None
    own_shot = shot_quality(conn, pid, season=season)
    lg_shot = league_shots_cached(conn, season)
    flags = flags_cached(conn, season, season_type, league)
    r0 = reb["season"]
    a0 = ast["season"]

    def rebw(r):
        o = {k: r[k] for k in _REB_W}
        o["chances_pg"] = _pg(r["chances"], r["games"])
        o["off"] = {k: r["off"][k] for k in ("stat", "opps", "conv")}
        o["def"] = {k: r["def"][k] for k in ("stat", "opps", "conv")}
        return o

    def astw(a):
        o = {k: a[k] for k in _AST_W}
        o["pot_ast_pg"] = _pg(a["pot_ast"], a["pot_ast_games"])
        o["passes_pg"] = _pg(a["passes"], a["games"])
        o["touches_pg"] = _pg(a["touches"], a["games"])
        return o

    return {
        "player_id": pid, "season": season, "season_type": season_type,
        "position": r0["position"], "games": len(full),
        "windows": [w for w, _ in WINDOWS],
        "window_games": {k: len(g) for k, g in win.items()},
        "has": has,
        "rebounds": {**{k: rebw(v) for k, v in reb.items()},
                     "baseline": {"own": r0["baseline_conv"], "own_raw": r0["season_conv"],
                                  "league": r0["league_conv"],
                                  "league_all": r0["league_conv_by_pos"]["all"]}},
        "assists": {**{k: astw(v) for k, v in ast.items()},
                    "baseline": {"own": a0["season_ast_per_pot_ast"],
                                 "league": a0["league_ast_per_pot_ast"],
                                 "own_per_pass": a0["season_ast_per_pass"],
                                 "league_per_pass": a0["league_ast_per_pass"]}},
        "shooting": {**{k: {f: v[f] for f in _SHOT_W} for k, v in sht.items()},
                     "baseline": {"own": own_shot["making_fg_pct"],
                                  "league_fg_pct": (lg_shot or {}).get("fg_pct"),
                                  "league_pps": (lg_shot or {}).get("pps")}},
        "defense": {**{k: {f: v[f] for f in _DEF_W} for k, v in dfn.items()},
                    "baseline": {"own": dfn["season"]["season_rim_fg_allowed"],
                                 "league": dfn["season"]["league_rim_fg_allowed"]}},
        "flags": [f for f in (flags or {}).get("hot", []) + (flags or {}).get("cold", [])
                  if f["player_id"] == pid],
    }
