"""Team stats for the Teams pages (docs/STATS_SITE.md): the league table with ranks, a team's
record and results, roster, defense vs position, on/off and 5-man lineups.

Pure functions on a read-only `sqlite3.Connection`; heavy parts are cached per season with
`stats.db.cached` (dropped when courtside.db changes).

Ratings are possession-weighted over the season, not means of game ratings:
  ORtg = 100 * sum(pts) / sum(poss)          DRtg = 100 * sum(opp pts) / sum(opp poss)
  Pace = 48 * sum((poss + opp poss) / 2) / (sum(team minutes) / 5)
  eFG% = (FGM + 0.5 3PM) / FGA, TS% = PTS / (2 (FGA + 0.44 FTA)), TOV% = TOV / (FGA + 0.44 FTA
  + TOV), OREB% = OREB / (OREB + opp DREB), all from season sums.
`poss` comes from `team_games_adv`; a game without an advanced row uses the box estimate
FGA - OREB + TOV + 0.44 FTA. Opp pts = pts - plus_minus.

League rank: 1 is the best. Lower is better for DRtg, TOV% and opp pts; for pace, 1 = fastest.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from . import db as sdb
from . import leaders as L
from . import seasons as S

# key -> higher is better
RANKED = {"win_pct": True, "pts": True, "opp_pts": False, "ortg": True, "drtg": False,
          "net": True, "pace": True, "efg": True, "ts": True, "tov": False, "oreb": True,
          "fg3a": True, "ast": True, "reb": True}

_SQL = """
SELECT t.team, t.game_id, t.game_date, t.opp, t.is_home, t.wl, t.min, t.pts, t.plus_minus,
       t.fgm, t.fga, t.fg3m, t.fg3a, t.fta, t.oreb, t.dreb, t.reb, t.ast, t.tov,
       a.poss, o.pts AS o_pts, o.fga AS o_fga, o.oreb AS o_oreb, o.dreb AS o_dreb,
       o.tov AS o_tov, o.fta AS o_fta, oa.poss AS o_poss
FROM team_games t
LEFT JOIN team_games_adv a ON a.game_id=t.game_id AND a.team=t.team
LEFT JOIN team_games o ON o.game_id=t.game_id AND o.team=t.opp
LEFT JOIN team_games_adv oa ON oa.game_id=t.game_id AND oa.team=t.opp
WHERE t.season=? AND t.season_type=?
ORDER BY t.game_date, t.game_id
"""


def _f(x) -> float:
    return 0.0 if x is None else float(x)


def _est_poss(fga, oreb, tov, fta) -> float:
    return _f(fga) - _f(oreb) + _f(tov) + 0.44 * _f(fta)


def _div(a, b, scale=1.0):
    return (scale * a / b) if b else None


def _record(games: list[dict]) -> str:
    w = sum(1 for g in games if g["wl"] == "W")
    return f"{w}-{len(games) - w}"


def _streak(games: list[dict]) -> str | None:
    if not games:
        return None
    last, n = games[-1]["wl"], 0
    for g in reversed(games):
        if g["wl"] != last:
            break
        n += 1
    return f"{last}{n}" if last in ("W", "L") else None


def compute_league(conn: sqlite3.Connection, season: str, season_type: str) -> dict:
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    games: dict[str, list[dict]] = defaultdict(list)
    for r in conn.execute(_SQL, (season, season_type)):
        a = acc[r["team"]]
        poss = r["poss"] if r["poss"] is not None else _est_poss(r["fga"], r["oreb"], r["tov"],
                                                                 r["fta"])
        o_poss = r["o_poss"] if r["o_poss"] is not None else (
            _est_poss(r["o_fga"], r["o_oreb"], r["o_tov"], r["o_fta"]) if r["o_fga"] is not None
            else poss)
        pts = _f(r["pts"])
        opp_pts = pts - _f(r["plus_minus"]) if r["plus_minus"] is not None else _f(r["o_pts"])
        a["gp"] += 1
        a["w"] += 1 if r["wl"] == "W" else 0
        a["pts"] += pts
        a["opp_pts"] += opp_pts
        a["poss"] += _f(poss)
        a["o_poss"] += _f(o_poss)
        a["min"] += _f(r["min"]) or 240.0
        for k in ("fgm", "fga", "fg3m", "fg3a", "fta", "oreb", "dreb", "reb", "ast", "tov"):
            a[k] += _f(r[k])
        a["o_dreb"] += _f(r["o_dreb"])
        games[r["team"]].append({"game_id": r["game_id"], "date": r["game_date"],
                                 "opp": r["opp"], "home": bool(r["is_home"]), "wl": r["wl"],
                                 "pts": int(pts), "opp_pts": int(round(opp_pts))})
    rows = []
    for team, a in acc.items():
        gp = a["gp"]
        ortg, drtg = _div(a["pts"], a["poss"], 100), _div(a["opp_pts"], a["o_poss"], 100)
        gl = games[team]
        row = {
            "team": team, "gp": int(gp), "w": int(a["w"]), "l": int(gp - a["w"]),
            "win_pct": _div(a["w"], gp),
            "pts": _div(a["pts"], gp), "opp_pts": _div(a["opp_pts"], gp),
            "ortg": ortg, "drtg": drtg,
            "net": (ortg - drtg) if ortg is not None and drtg is not None else None,
            "pace": _div(48.0 * (a["poss"] + a["o_poss"]) / 2.0, a["min"] / 5.0),
            "efg": _div(a["fgm"] + 0.5 * a["fg3m"], a["fga"]),
            "ts": _div(a["pts"], 2.0 * (a["fga"] + 0.44 * a["fta"])),
            "tov": _div(a["tov"], a["fga"] + 0.44 * a["fta"] + a["tov"]),
            "oreb": _div(a["oreb"], a["oreb"] + a["o_dreb"]),
            "fg3a": _div(a["fg3a"], gp), "ast": _div(a["ast"], gp), "reb": _div(a["reb"], gp),
            "home": _record([g for g in gl if g["home"]]),
            "road": _record([g for g in gl if not g["home"]]),
            "l10": _record(gl[-10:]), "streak": _streak(gl),
        }
        rows.append(row)
    ranks = {k: L.rank_values({r["team"]: r[k] for r in rows}, hb) for k, hb in RANKED.items()}
    for r in rows:
        r["rk"] = {k: ranks[k].get(r["team"]) for k in RANKED}
    rows.sort(key=lambda r: (-(r["net"] if r["net"] is not None else -999), r["team"]))
    return {"rows": rows, "by_team": {r["team"]: r for r in rows}, "games": dict(games),
            "n": len(rows), "max_gp": max((r["gp"] for r in rows), default=0)}


def league(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """{rows (sorted by net rating), by_team, games: {team: [game dicts, oldest first]}, n,
    max_gp}. Row keys: team gp w l win_pct pts opp_pts ortg drtg net pace efg ts tov oreb fg3a
    ast reb home road l10 streak, and rk = {key: league rank}."""
    return sdb.cached(("stats.league", season, season_type),
                      lambda c: compute_league(c, season, season_type), conn, domain="box")


# ------------------------------------------------------------------ team leaders
TEAM_CATEGORIES = (("Net rating", "net", "signed"), ("Off rating", "ortg", "num"),
                   ("Def rating", "drtg", "num"), ("Pace", "pace", "num"),
                   ("eFG%", "efg", "pct"), ("True shooting", "ts", "pct"),
                   ("Fewest TOV%", "tov", "pct"), ("OREB%", "oreb", "pct"),
                   ("Points / game", "pts", "num"), ("Fewest opp pts", "opp_pts", "num"),
                   ("3PA / game", "fg3a", "num"), ("Assists / game", "ast", "num"),
                   ("Rebounds / game", "reb", "num"))


def team_leaders(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> list:
    """[{label, key, kind, best: row, next: row}] the best team (and runner-up) per category."""
    lg = league(conn, season, season_type)
    out = []
    for label, key, kind in TEAM_CATEGORIES:
        ranked = sorted((r for r in lg["rows"] if r["rk"].get(key)),
                        key=lambda r: r["rk"][key])
        if ranked:
            out.append({"label": label, "key": key, "kind": kind, "best": ranked[0],
                        "next": ranked[1] if len(ranked) > 1 else None})
    return out


# ------------------------------------------------------------------ team page pieces
def roster(conn: sqlite3.Connection, team: str, season: str, season_type: str = S.REGULAR
           ) -> list[dict]:
    """Per-game lines of every player who played for `team` (his games for this team only),
    most minutes first."""
    pk = L.pack(conn, season, season_type)
    rows = []
    for (t, pid), s in pk["roster"].items():
        if t != team:
            continue
        ln = L.line(s, "pg")
        rows.append({"player_id": pid, "name": pk["names"].get(pid), **ln,
                     "tot_min": s["min"]})
    rows.sort(key=lambda r: -r["tot_min"])
    return rows


def dvp(conn: sqlite3.Connection, team: str, season: str, season_type: str = S.REGULAR
        ) -> dict | None:
    """{basis, buckets, coverage, n, row} for one team: DK points allowed per game to each
    position, all games and the last 10, with league ranks (1 = allows the most)."""
    pk = L.pack(conn, season, season_type)
    row = pk["dvp"].get(team)
    if not row:
        return None
    return {"basis": pk["basis"], "buckets": pk["buckets"], "coverage": pk["coverage"],
            "n": len(pk["dvp"]), "row": row}


def results(conn: sqlite3.Connection, team: str, season: str, season_type: str = S.REGULAR,
            n: int = 10) -> list[dict]:
    """The team's last `n` games, newest first."""
    gl = league(conn, season, season_type)["games"].get(team, [])
    return list(reversed(gl[-n:])) if n else list(reversed(gl))


def upcoming(conn: sqlite3.Connection, team: str, from_day: str, n: int = 6) -> list[dict]:
    """Scheduled games (`games`, the 2026-27 schedule) on or after `from_day`."""
    try:
        rows = conn.execute(
            "SELECT game_id, nba_day, tipoff_utc, home, away, status FROM games "
            "WHERE (home=? OR away=?) AND nba_day>=? AND status IN ('scheduled','live') "
            "ORDER BY nba_day, tipoff_utc LIMIT ?", (team, team, from_day, n)).fetchall()
    except sqlite3.Error:
        return []
    return [{"game_id": r["game_id"], "day": r["nba_day"], "tipoff": r["tipoff_utc"],
             "home": r["home"] == team, "opp": r["away"] if r["home"] == team else r["home"],
             "status": r["status"]} for r in rows]


def coach(conn: sqlite3.Connection, team: str, season: str) -> str | None:
    """Head coach name from `team_coaches` (None when the season is not loaded)."""
    try:
        r = conn.execute("SELECT coach_name FROM team_coaches WHERE season=? AND team=? AND "
                         "coach_type='Head Coach' ORDER BY sort_sequence LIMIT 1",
                         (season, team)).fetchone()
    except sqlite3.Error:
        return None
    return r[0] if r else None


# ------------------------------------------------------------------ on/off + lineups (stints)
ONOFF_PLAYERS = 12
LINEUPS = 10


def lineup_min_poss(gp: int) -> int:
    """Minimum possessions for a 5-man unit: 50 once the season is 25+ games old, else
    2 per game played (at least 10)."""
    return 50 if gp >= 25 else max(10, 2 * gp)


def _rtg(pf, pa, possf, possa) -> tuple:
    o = _div(pf, possf, 100)
    d = _div(pa, possa, 100)
    return o, d, (o - d) if o is not None and d is not None else None


def compute_stints(conn: sqlite3.Connection, team: str, season: str, season_type: str,
                   skip: dict | None = None) -> dict:
    """`skip` = {game_id: {stint_no}} to leave out (garbage time, stats/garbage.py)."""
    gl = league(conn, season, season_type)["games"].get(team, [])
    side = {g["game_id"]: g["home"] for g in gl}
    tot = [0.0, 0.0, 0.0, 0.0]                     # pts for, against, poss for, against
    on: dict[int, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])   # + seconds
    units: dict[tuple, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    covered = set()
    ids = list(side)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ("SELECT game_id, stint_no, start_sec, end_sec, home5, away5, home_pts, away_pts, "
             f"home_poss, away_poss FROM stints WHERE game_id IN ({','.join('?' * len(chunk))})")
        try:
            rows = conn.execute(q, chunk).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            home = side[r["game_id"]]
            covered.add(r["game_id"])
            if skip and r["stint_no"] in skip.get(r["game_id"], ()):
                continue
            five = json.loads(r["home5"] if home else r["away5"])
            if home:
                v = (_f(r["home_pts"]), _f(r["away_pts"]), _f(r["home_poss"]), _f(r["away_poss"]))
            else:
                v = (_f(r["away_pts"]), _f(r["home_pts"]), _f(r["away_poss"]), _f(r["home_poss"]))
            sec = max(0.0, _f(r["end_sec"]) - _f(r["start_sec"]))
            for k in range(4):
                tot[k] += v[k]
            for pid in five:
                a = on[int(pid)]
                for k in range(4):
                    a[k] += v[k]
                a[4] += sec
            u = units[tuple(sorted(int(p) for p in five))]
            for k in range(4):
                u[k] += v[k]
            u[4] += sec
    names = L.pack(conn, season, season_type)["names"]
    onoff = []
    for pid, a in sorted(on.items(), key=lambda kv: -kv[1][4])[:ONOFF_PLAYERS]:
        o_on, d_on, n_on = _rtg(*a[:4])
        off = [tot[k] - a[k] for k in range(4)]
        if (off[2] + off[3]) / 2 >= 1:
            o_off, d_off, n_off = _rtg(*off)
        else:
            o_off = d_off = n_off = None
        onoff.append({"player_id": pid, "name": names.get(pid) or str(pid),
                      "min": a[4] / 60.0, "on_poss": (a[2] + a[3]) / 2,
                      "off_poss": (off[2] + off[3]) / 2,
                      "on_ortg": o_on, "on_drtg": d_on, "on_net": n_on,
                      "off_ortg": o_off, "off_drtg": d_off, "off_net": n_off,
                      "diff": (n_on - n_off) if n_on is not None and n_off is not None else None})
    gp_cov = len(covered)
    need = lineup_min_poss(gp_cov)
    lineups = []
    for five, u in units.items():
        poss = (u[2] + u[3]) / 2
        if poss < need:
            continue
        o, d, n = _rtg(*u[:4])
        lineups.append({"ids": list(five), "names": [names.get(p) or str(p) for p in five],
                        "poss": poss, "min": u[4] / 60.0, "ortg": o, "drtg": d, "net": n})
    lineups.sort(key=lambda r: -r["poss"])
    t_o, t_d, t_n = _rtg(*tot)
    return {"games": gp_cov, "team_gp": len(gl), "onoff": onoff, "lineups": lineups[:LINEUPS],
            "units": len(units), "min_poss": need,
            "team": {"ortg": t_o, "drtg": t_d, "net": t_n, "poss": (tot[2] + tot[3]) / 2}}


def stints(conn: sqlite3.Connection, team: str, season: str, season_type: str = S.REGULAR
           ) -> dict:
    """On/off and 5-man lineups of one team-season from `stints` (computed once, cached).

    {games (with stints), team_gp, onoff: [top-minute players: on/off ORtg DRtg Net, diff =
    on net - off net], lineups: [units with >= min_poss possessions, most possessions first],
    min_poss, team}. Possessions of a stint = (team poss + opp poss) / 2; ratings are per 100
    of each side's own possessions. `games == 0` means "not collected yet"."""
    return sdb.cached(("stats.stints", team, season, season_type),
                      lambda c: compute_stints(c, team, season, season_type), conn, domain="box")


# ------------------------------------------------------------------ standings
def _sort_standing(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (-(r["win_pct"] or 0.0), -r["w"], r["l"], r["team"]))


def _standing_rows(rows: list[dict]) -> list[dict]:
    """Ranked rows with GB from the group leader: ((W1 - W) + (L - L1)) / 2."""
    rows = _sort_standing(rows)
    if not rows:
        return []
    lw, ll = rows[0]["w"], rows[0]["l"]
    out = []
    for i, r in enumerate(rows, start=1):
        out.append({**r, "pos": i, "gb": ((lw - r["w"]) + (r["l"] - ll)) / 2.0})
    return out


def compute_standings(conn: sqlite3.Connection, season: str, season_type: str) -> dict:
    from .. import teams as TM

    by_team = league(conn, season, season_type)["by_team"]
    conf, div, rank = {}, {}, {}
    for c in TM.CONFERENCES:
        conf[c] = _standing_rows([by_team[t.tricode] for t in TM.teams_in(conference=c)
                                  if t.tricode in by_team])
        for r in conf[c]:
            rank.setdefault(r["team"], {})["conf"] = r["pos"]
    for d, (c, codes) in TM.DIVISIONS.items():
        div[d] = {"conference": c,
                  "rows": _standing_rows([by_team[t] for t in codes if t in by_team])}
        for r in div[d]["rows"]:
            rank.setdefault(r["team"], {})["div"] = r["pos"]
    for t, rk in rank.items():
        rk.update(conference=TM.conference(t), division=TM.division(t))
    return {"conferences": conf, "divisions": div, "rank": rank}


def standings(conn: sqlite3.Connection, season: str, season_type: str = S.REGULAR) -> dict:
    """Conference and division standings by win % (then wins, fewer losses, code; the NBA's
    tiebreakers are not applied). {conferences: {East|West: rows}, divisions: {name:
    {conference, rows}}, rank: {team: {conf, div, conference, division}}}; each row is the
    league-table row plus pos and gb. Alignment: today's conferences/divisions (teams.py)."""
    return sdb.cached(("stats.standings", season, season_type),
                      lambda c: compute_standings(c, season, season_type), conn, domain="box")
