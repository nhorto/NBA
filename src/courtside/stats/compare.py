"""Compare tool (/compare): 2-4 players or 2 teams side by side (docs/STATS_SITE.md).

Pure functions on a read-only `sqlite3.Connection`; plain dicts out. The heavy parts reuse the
cached season passes of players.py (league table), leaders.py (season pack), similar.py
(profiles with AST% / REB%) and teams.py (league ratings).

  percentile_rank(sorted_values, x, higher_better)   0-100 mid-rank percentile
  pools(conn, season, type)                 sorted league values per stat, for percentiles
  player(conn, pid, season, type)           every number of one player's compare column
  head_to_head(conn, pids)                  games where 2+ of the players met (opposite teams)
  team_factors(conn, season, type)          four factors, offense + defense, with ranks
  team_h2h(conn, a, b, seasons)             games between two teams, by season

Percentile pools: per-game stats use the league table's qualified players (40% of the most
games played, at least 2); per-36 and rate stats also need 15+ minutes per game. A percentile
is the share of the pool below the value, ties counted half ("better than 87%"); for stats
where less is better (turnovers) it is flipped, so a high percentile is always good.
"""

from __future__ import annotations

import bisect
import sqlite3
from typing import Iterable, Sequence

from . import db as sdb
from . import leaders as L
from . import players as P
from . import seasons as S
from . import similar as SIM
from . import teams as ST

MAX_PLAYERS, MAX_TEAMS = 4, 2
RATE_MIN_MPG = L.PER36_MIN_MPG
H2H_SHOW = 15
FORM_N = 10

# Player rows: (section, [(key, label, kind, nd, better, tip)]); better: 1 high, -1 low, 0 none.
# key "pg.x" = per game, "p36.x" = per 36, "adv.x" = rate, "dfs.x" = DK.
PLAYER_SECTIONS = (
    ("Per game", (
        ("pg.gp", "Games", "int", 0, 1, None),
        ("pg.min", "Minutes", "num", 1, 1, None),
        ("pg.pts", "Points", "num", 1, 1, None),
        ("pg.reb", "Rebounds", "num", 1, 1, None),
        ("pg.ast", "Assists", "num", 1, 1, None),
        ("pg.stl", "Steals", "num", 1, 1, None),
        ("pg.blk", "Blocks", "num", 1, 1, None),
        ("pg.tov", "Turnovers", "num", 1, -1, "Fewer is better"),
        ("pg.fg3m", "3PM", "num", 1, 1, None),
        ("pg.fg_pct", "FG%", "pct", 1, 1, None),
        ("pg.fg3_pct", "3P%", "pct", 1, 1, None),
        ("pg.ft_pct", "FT%", "pct", 1, 1, None),
        ("pg.pm", "+/−", "signed", 1, 1, "Plus-minus per game"),
    )),
    ("Per 36 minutes", (
        ("p36.pts", "Points", "num", 1, 1, None),
        ("p36.reb", "Rebounds", "num", 1, 1, None),
        ("p36.ast", "Assists", "num", 1, 1, None),
        ("p36.stl", "Steals", "num", 1, 1, None),
        ("p36.blk", "Blocks", "num", 1, 1, None),
        ("p36.tov", "Turnovers", "num", 1, -1, "Fewer is better"),
        ("p36.fg3m", "3PM", "num", 1, 1, None),
    )),
    ("Advanced", (
        ("adv.ts", "TS%", "pct", 1, 1, "True shooting: PTS / (2 × (FGA + 0.44 FTA))"),
        ("adv.usg", "USG%", "pct", 1, 1, "Share of team plays he finished while on the floor "
                                         "(minute-weighted)"),
        ("adv.astp", "AST%", "pct", 1, 1, "Share of teammate field goals he assisted while on "
                                           "the floor"),
        ("adv.rebp", "REB%", "pct", 1, 1, "Share of available rebounds he grabbed"),
        ("adv.par3", "3PA rate", "pct", 1, 1, "3-point attempts per field-goal attempt"),
        ("adv.ftr", "FTA rate", "pct", 1, 1, "Free-throw attempts per field-goal attempt"),
    )),
    ("DraftKings", (
        ("dfs.dk", "DK pts / game", "num", 1, 1, "DraftKings classic points (bonuses included)"),
        ("dfs.dk36", "DK pts / 36", "num", 1, 1, None),
        ("dfs.salary", "Avg salary", "dollars", 0, 0, "Average DK classic salary over his games "
                                                      "with a salary"),
        ("dfs.value", "DK value", "num", 2, 1, "DK points per $1,000 of salary, averaged over "
                                               "games with a salary"),
    )),
)
FORM_STATS = (("pts", "Points", 1), ("reb", "Rebounds", 1), ("ast", "Assists", 1),
              ("fg3m", "3PM", 1), ("min", "Minutes", 1), ("dk", "DK pts", 1))


# ------------------------------------------------------------------ percentiles
def percentile_rank(values: Sequence[float], x: float | None,
                    higher_better: bool = True) -> float | None:
    """Mid-rank percentile of x in the sorted `values`: 100 * (below + ties / 2) / n.
    Flipped (100 - p) when lower is better. None for a missing x or an empty pool."""
    if x is None or not values:
        return None
    lo = bisect.bisect_left(values, x)
    hi = bisect.bisect_right(values, x)
    p = 100.0 * (lo + 0.5 * (hi - lo)) / len(values)
    return p if higher_better else 100.0 - p


def _pg_line(row: dict) -> dict:
    return {"gp": row.get("gp"), "min": row.get("min"), "pts": row.get("pts"),
            "reb": row.get("reb"), "ast": row.get("ast"), "stl": row.get("stl"),
            "blk": row.get("blk"), "tov": row.get("tov"), "fg3m": row.get("fg3m"),
            "fg_pct": row.get("fg_pct"), "fg3_pct": row.get("fg3_pct"),
            "ft_pct": row.get("ft_pct"), "pm": row.get("pm")}


def _values(lt_row: dict | None, pk_row: dict | None, prof: dict | None) -> dict[str, float]:
    """Flat {"pg.pts": v, "p36.reb": v, "adv.ts": v, "dfs.dk": v, ...} of one player."""
    out: dict[str, float | None] = {}
    if lt_row:
        for k, v in _pg_line(lt_row).items():
            out[f"pg.{k}"] = v
        out["dfs.dk"] = lt_row.get("dk")
        out["adv.ts"] = lt_row.get("ts")
    if pk_row:
        ln = L.line(pk_row["w"]["all"], "p36")
        for k in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m"):
            out[f"p36.{k}"] = ln.get(k)
        out["dfs.dk36"] = ln.get("dk")
    if prof:
        for k in ("usg", "astp", "rebp", "par3", "ftr"):
            out[f"adv.{k}"] = prof.get(k)
    return out


def pools(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """{key: sorted values over the qualified pool} for every percentile row (cached)."""
    def build(c):
        lt = P.league_table(c, season, season_type)
        pk = L.pack(c, season, season_type) or {"players": {}}
        prof = SIM.season_profiles(c, season, season_type)
        acc: dict[str, list[float]] = {}
        for pid, row in lt["players"].items():
            if not row.get("qualified"):
                continue
            rate_ok = (row.get("min") or 0) >= RATE_MIN_MPG
            for k, v in _values(row, pk["players"].get(pid), prof.get(pid)).items():
                if v is None or k in ("pg.gp", "pg.min"):
                    continue
                if k.split(".")[0] in ("p36", "adv") or k == "dfs.dk36":
                    if not rate_ok:
                        continue
                acc.setdefault(k, []).append(float(v))
            for k in ("pg.gp", "pg.min"):
                if row.get(k[3:]) is not None:
                    acc.setdefault(k, []).append(float(row[k[3:]]))
        return {k: sorted(v) for k, v in acc.items()} | {"_n": len(
            [1 for r in lt["players"].values() if r.get("qualified")])}
    return sdb.cached(("cmp-pools", season, season_type), build, conn, domain="box") or {"_n": 0}


# ------------------------------------------------------------------ one player column
def _dk_value(log: Iterable[dict]) -> dict:
    with_sal = [g for g in log if g.get("salary") and g.get("played")]
    if not with_sal:
        return {"salary": None, "value": None, "n_sal": 0}
    return {"salary": sum(g["salary"] for g in with_sal) / len(with_sal),
            "value": sum(g["value"] for g in with_sal) / len(with_sal), "n_sal": len(with_sal)}


def player(conn: sqlite3.Connection, pid: int, season: str,
           season_type: str = S.REGULAR) -> dict:
    """{player_id, name, team, position, played, qualified, values: {key: v},
    pctile: {key: 0-100}, form: {...}, zones: [...], shots_n}. `played` False = no games of his
    in that season / type (the column shows dashes)."""
    lt = P.league_table(conn, season, season_type)
    row = lt["players"].get(pid)
    pk = (L.pack(conn, season, season_type) or {"players": {}})["players"].get(pid)
    prof = SIM.season_profiles(conn, season, season_type).get(pid)
    vals = _values(row, pk, prof)
    log = [g for g in P.game_log(conn, pid, (season,), season=season, season_type=season_type)
           if g.get("played")]
    vals.update({f"dfs.{k}": v for k, v in _dk_value(log).items()})
    pl = pools(conn, season, season_type)
    better = {k: b for _, rows in PLAYER_SECTIONS for k, _, _, _, b, _ in rows}
    pct = {}
    for k, v in vals.items():
        if k in pl and k not in ("pg.gp",):
            pct[k] = percentile_rank(pl[k], v, higher_better=better.get(k, 1) >= 0)
    shots = P.shot_chart(conn, pid, season, season_type) or {"n": 0, "zones": []}
    return {"player_id": pid, "name": (row or {}).get("name"), "team": (row or {}).get("team"),
            "position": (row or {}).get("position"), "played": row is not None,
            "qualified": bool((row or {}).get("qualified")), "values": vals, "pctile": pct,
            "form": form(log, row), "zones": shots.get("zones") or [], "shots_n": shots["n"],
            "pool_n": pl.get("_n", 0)}


def form(log: list[dict], season_row: dict | None, n: int = FORM_N) -> dict:
    """Last `n` games played (newest first in `log`): averages, diff vs the season average and
    the per-game DK points (oldest first, for the spark bars)."""
    games = log[:n]
    if not games:
        return {"n": 0, "avg": {}, "diff": {}, "dk": [], "games": []}
    avg = {k: sum((g.get(k) or 0) for g in games) / len(games) for k, _, _ in FORM_STATS}
    diff = {}
    for k, _, _ in FORM_STATS:
        base = (season_row or {}).get(k)
        if base is not None:
            diff[k] = avg[k] - base
    return {"n": len(games), "avg": avg, "diff": diff,
            "dk": [{"v": g["dk"], "date": g["game_date"], "opp": g.get("opp"),
                    "home": g.get("is_home")} for g in reversed(games)],
            "first": games[-1]["game_date"], "last": games[0]["game_date"]}


# ------------------------------------------------------------------ head to head
H2H_COLS = ("min", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fgm", "fga", "ftm", "fta")


def head_to_head(conn: sqlite3.Connection, pids: Sequence[int]) -> dict:
    """Every game (any season, regular + playoffs) where at least two of the players played
    (min > 0) on opposite teams. {games: [{game_id, date, season, season_type, home, away,
    home_pts, away_pts, lines: {pid: {team, wl, pts, reb, ast, dk, ...}}}] newest first,
    summary: {pid: {games, w, l, pts, reb, ast, dk}}}."""
    ids = sorted({int(p) for p in pids})
    if len(ids) < 2:
        return {"games": [], "summary": {}}

    def build(c):
        rows = P._rows(c, "SELECT pg.game_id, pg.player_id, pg.team, pg.season, pg.season_type, "
                          "pg.game_date, " + ", ".join(f"pg.{k}" for k in H2H_COLS) +
                          ", tg.wl, tg.opp, tg.is_home, tg.pts AS team_pts, o.pts AS opp_pts "
                          "FROM player_games pg LEFT JOIN team_games tg ON tg.game_id=pg.game_id "
                          "AND tg.team=pg.team LEFT JOIN team_games o ON o.game_id=pg.game_id "
                          "AND o.team=tg.opp WHERE pg.player_id IN (" + ",".join("?" * len(ids))
                          + ") AND pg.min > 0", ids)
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["game_id"], []).append(r)
        games = []
        for gid, rs in by.items():
            if len({r["team"] for r in rs}) < 2:
                continue
            r0 = rs[0]
            home_r = next((r for r in rs if r["is_home"]), None)
            if home_r is not None:
                home, away = home_r["team"], home_r["opp"]
                hp, ap = home_r["team_pts"], home_r["opp_pts"]
            else:
                away, home = r0["team"], r0["opp"]
                ap, hp = r0["team_pts"], r0["opp_pts"]
            lines = {}
            for r in rs:
                ln = {k: r[k] for k in H2H_COLS}
                ln.update(team=r["team"], wl=r["wl"], dk=P._dk(r))
                lines[int(r["player_id"])] = ln
            games.append({"game_id": gid, "date": r0["game_date"], "season": r0["season"],
                          "season_type": r0["season_type"], "home": home, "away": away,
                          "home_pts": hp, "away_pts": ap, "lines": lines})
        games.sort(key=lambda g: (g["date"] or "", g["game_id"]), reverse=True)
        summary = {}
        for pid in ids:
            mine = [g["lines"][pid] for g in games if pid in g["lines"]]
            if not mine:
                continue
            n = len(mine)
            summary[pid] = {"games": n, "w": sum(1 for x in mine if x["wl"] == "W"),
                            "l": sum(1 for x in mine if x["wl"] == "L"),
                            **{k: sum((x[k] or 0) for x in mine) / n
                               for k in ("pts", "reb", "ast", "dk")}}
        return {"games": games, "summary": summary}
    return sdb.cached(("cmp-h2h", tuple(ids)), build, conn,
                      domain="box") or {"games": [], "summary": {}}


# ------------------------------------------------------------------ teams
# (key, label, better: True = higher is better)
FACTORS = (
    ("off", "Offense", (("efg", "eFG%", True), ("tov", "TOV%", False), ("orb", "OREB%", True),
                        ("ftr", "FT rate", True))),
    ("def", "Defense", (("o_efg", "Opp eFG%", False), ("o_tov", "Opp TOV%", True),
                        ("drb", "DREB%", True), ("o_ftr", "Opp FT rate", False))),
)
_FF_SQL = """
SELECT t.team, COUNT(*) AS gp, SUM(t.fgm), SUM(t.fga), SUM(t.fg3m), SUM(t.fta), SUM(t.tov),
       SUM(t.oreb), SUM(t.dreb), SUM(o.fgm), SUM(o.fga), SUM(o.fg3m), SUM(o.fta), SUM(o.tov),
       SUM(o.oreb), SUM(o.dreb)
FROM team_games t JOIN team_games o ON o.game_id=t.game_id AND o.team=t.opp
WHERE t.season=? AND t.season_type=?
GROUP BY t.team
"""


def _ff(fgm, fga, fg3m, fta, tov, oreb, opp_dreb) -> dict:
    def d(a, b):
        return a / b if b else None
    return {"efg": d(fgm + 0.5 * fg3m, fga), "tov": d(tov, fga + 0.44 * fta + tov),
            "orb": d(oreb, oreb + opp_dreb), "ftr": d(fta, fga)}


def compute_factors(conn: sqlite3.Connection, season: str, season_type: str) -> dict:
    out: dict[str, dict] = {}
    try:
        rows = conn.execute(_FF_SQL, (season, season_type)).fetchall()
    except sqlite3.Error:
        rows = []
    for r in rows:
        v = [float(x or 0) for x in tuple(r)[2:]]
        fgm, fga, fg3m, fta, tov, oreb, dreb, ofgm, ofga, ofg3m, ofta, otov, ooreb, odreb = v
        off = _ff(fgm, fga, fg3m, fta, tov, oreb, odreb)
        de = _ff(ofgm, ofga, ofg3m, ofta, otov, ooreb, dreb)
        out[r[0]] = {"gp": int(r[1]), **off, "o_efg": de["efg"], "o_tov": de["tov"],
                     "drb": (1.0 - de["orb"]) if de["orb"] is not None else None,
                     "o_ftr": de["ftr"]}
    rk: dict[str, dict] = {t: {} for t in out}
    for _, _, items in FACTORS:
        for key, _, hb in items:
            ranks = L.rank_values({t: r[key] for t, r in out.items()}, higher_better=hb)
            for t in out:
                rk[t][key] = ranks.get(t)
    for t, r in out.items():
        r["rk"] = rk[t]
    return {"by_team": out, "n": len(out)}


def team_factors(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """{by_team: {team: {efg tov orb ftr o_efg o_tov drb o_ftr, rk: {key: rank}}}, n} (cached).
    eFG% = (FGM + 0.5 3PM) / FGA; TOV% = TOV / (FGA + 0.44 FTA + TOV); OREB% = OREB / (OREB +
    opp DREB); FT rate = FTA / FGA; DREB% = 1 - opponent OREB%. Rank 1 = best."""
    return sdb.cached(("cmp-ff", season, season_type),
                      lambda c: compute_factors(c, season, season_type), conn, domain="box") or {
        "by_team": {}, "n": 0}


def team_h2h(conn: sqlite3.Connection, a: str, b: str, seasons: Sequence[str]) -> list[dict]:
    """[{season, games: [{date, season_type, home, a_pts, b_pts, winner, game_id}] newest first,
    a_w, b_w}] for each season (regular season + playoffs), from the cached league tables."""
    out = []
    for s in seasons:
        games = []
        for t in (S.REGULAR, S.PLAYOFFS):
            lg = ST.league(conn, s, t) if conn is not None else None
            for g in ((lg or {}).get("games") or {}).get(a, []):
                if g["opp"] != b:
                    continue
                games.append({"game_id": g["game_id"], "date": g["date"], "season_type": t,
                              "a_home": g["home"], "a_pts": g["pts"], "b_pts": g["opp_pts"],
                              "winner": a if g["wl"] == "W" else b})
        games.sort(key=lambda g: (g["date"] or "", g["game_id"]), reverse=True)
        out.append({"season": s, "games": games,
                    "a_w": sum(1 for g in games if g["winner"] == a),
                    "b_w": sum(1 for g in games if g["winner"] == b)})
    return out
