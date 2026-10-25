"""Games queries for the stats dashboard: the scoreboard for a night and a game's box score.

Pure functions on a read-only `sqlite3.Connection`; they return plain dicts/lists and never
raise on a missing table or row (the pages show an empty state instead).

    scoreboard(conn, day)        -> [game, ...] for one NBA day (the `games` table; older days
                                    fall back to `team_games`)
    adjacent_days(conn, day)     -> (previous night with games, next night with games)
    latest_final_day(conn)       -> newest night that has a final
    game_box(conn, game_id)      -> the box-score view model, or None when it is not stored
    dfs_for_game(conn, game_id)  -> {player_id: {"salary", "proj", "own"}} for the DFS view

Game card keys: game_id, day, status, home, away, home_score, away_score, winner ("home" /
"away" / None), tipoff_utc, ot (number of overtimes), home_rec / away_rec ("3-1" or None),
top (up to two {id, name, team, pts, line} from the story facts).

DFS (all optional, None where missing):
  salary  DK Classic salary (`dfs_salaries.game_type_id = 70`, newest capture at or before tip;
          the player is matched through `player_ids(source='dk')`, then by normalised name)
  proj    our projection: `player_projection_stats(stat='dk', cond='played').mean` of the latest
          `dfs_runs.ok = 1` run created before `games.tipoff_utc`
  own     real ownership: average `pct_drafted` over the contests of the classic slates that
          contain the game (`dfs_contest_ownership`)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..proj.dfs.rules import dk_points
from . import db as sdb
from .search import norm

REGULAR_PREFIX = "002"            # regular-season game ids; 004 playoffs, 005 play-in
OT_LABELS = ("OT", "2OT", "3OT", "4OT", "5OT", "6OT")


# ------------------------------------------------------------------ small helpers
def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []


def _one(conn: sqlite3.Connection, sql: str, args: tuple = ()):
    r = _rows(conn, sql, args)
    return r[0] if r else None


def _json(s) -> dict:
    try:
        v = json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _div(a, b) -> float | None:
    try:
        return a / b if b else None
    except TypeError:
        return None


def _rec(w, l) -> str | None:
    return None if w is None else f"{int(w)}-{int(l or 0)}"


# ------------------------------------------------------------------ nights
def latest_final_day(conn: sqlite3.Connection | None) -> str | None:
    """The newest NBA day with a final game (None without a database or finals)."""
    if conn is None:
        return None
    r = _one(conn, "SELECT MAX(nba_day) FROM games WHERE status='final'")
    if r and r[0]:
        return r[0]
    r = _one(conn, "SELECT MAX(game_date) FROM team_games")
    return r[0] if r and r[0] else None


def adjacent_days(conn: sqlite3.Connection, day: str) -> tuple[str | None, str | None]:
    """(previous, next) nights that have games, so the arrows skip empty days."""
    prev = _one(conn, "SELECT MAX(d) FROM (SELECT MAX(nba_day) d FROM games WHERE nba_day < ? "
                      "UNION ALL SELECT MAX(game_date) FROM team_games WHERE game_date < ?)",
                (day, day))
    nxt = _one(conn, "SELECT MIN(d) FROM (SELECT MIN(nba_day) d FROM games WHERE nba_day > ? "
                     "UNION ALL SELECT MIN(game_date) FROM team_games WHERE game_date > ?)",
               (day, day))
    return (prev[0] if prev else None, nxt[0] if nxt else None)


# ------------------------------------------------------------------ records to date
def records(conn: sqlite3.Connection, season: str, day: str, inclusive: bool) -> dict[str, tuple[int, int]]:
    """{team: (wins, losses)} in the regular season through `day` (inclusive or the day before).

    The `games` table is used when it has finals for the season (it is current the minute a
    game ends); otherwise `team_games`.
    """
    op = "<=" if inclusive else "<"

    def build(c: sqlite3.Connection):
        out: dict[str, list[int]] = {}
        rows = _rows(c, f"""SELECT team, SUM(w), SUM(l) FROM (
              SELECT home team, home_score > away_score w, home_score < away_score l FROM games
               WHERE season=? AND status='final' AND game_id LIKE '{REGULAR_PREFIX}%' AND nba_day {op} ?
              UNION ALL
              SELECT away, away_score > home_score, away_score < home_score FROM games
               WHERE season=? AND status='final' AND game_id LIKE '{REGULAR_PREFIX}%' AND nba_day {op} ?)
            GROUP BY team""", (season, day, season, day))
        if not rows:
            rows = _rows(c, f"""SELECT team, SUM(wl='W'), SUM(wl='L') FROM team_games
                 WHERE season=? AND season_type='Regular Season' AND game_date {op} ? GROUP BY team""",
                         (season, day))
        for r in rows:
            out[r[0]] = (int(r[1] or 0), int(r[2] or 0))
        return out

    return sdb.cached(("games-records", season, day, inclusive), build, conn)


# ------------------------------------------------------------------ story facts
def _facts(conn: sqlite3.Connection, game_ids: list[str]) -> dict[str, dict]:
    if not game_ids:
        return {}
    q = ",".join("?" * len(game_ids))
    return {r[0]: _json(r[1]) for r in _rows(
        conn, f"SELECT game_id, facts_json FROM game_facts WHERE game_id IN ({q})", tuple(game_ids))}


def _top(facts: dict, n: int = 2) -> list[dict]:
    out = []
    best = [p for p in (facts.get("top_performers") or []) if isinstance(p, dict) and p.get("name")]
    best.sort(key=lambda p: -(p.get("game_score") or 0))
    for p in best[:n]:
        line = p.get("line") or ""
        out.append({"id": p.get("id"), "name": p["name"], "team": p.get("team"),
                    "pts": line.split(" pts")[0] if " pts" in line else None, "line": line})
    return out


def _ot(facts: dict) -> int:
    """Overtimes played. The quarter scores win over the `overtimes` key: a regulation game
    can carry an empty 0-0 fifth period (PHX-GSW 2026-10-24 does) and is not an OT game."""
    hp = (facts.get("home") or {}).get("periods") or []
    ap = (facts.get("away") or {}).get("periods") or []
    if len(hp) > 4 or len(ap) > 4:
        return sum(1 for i in range(4, max(len(hp), len(ap)))
                   if (hp[i] if i < len(hp) else 0) or (ap[i] if i < len(ap) else 0))
    v = facts.get("overtimes")
    return v if isinstance(v, int) and not hp else 0


# ------------------------------------------------------------------ the scoreboard
def scoreboard(conn: sqlite3.Connection | None, day: str) -> list[dict]:
    """Every game of an NBA day, by tip-off time (see the module doc for the card keys)."""
    if conn is None:
        return []
    rows = [dict(r) for r in _rows(
        conn, "SELECT game_id, season, nba_day, tipoff_utc, home, away, status, home_score, "
              "away_score FROM games WHERE nba_day=? ORDER BY tipoff_utc, game_id", (day,))]
    if not rows:
        rows = _from_team_games(conn, day)
    if not rows:
        return []
    facts = _facts(conn, [r["game_id"] for r in rows])
    seasons = {r["season"] for r in rows if r.get("season")}
    season = next(iter(seasons)) if len(seasons) == 1 else (rows[0].get("season") or "")
    done = records(conn, season, day, True) if season else {}
    before = records(conn, season, day, False) if season and any(
        r["status"] != "final" for r in rows) else {}
    out = []
    for r in rows:
        f = facts.get(r["game_id"], {})
        hs, as_ = r["home_score"], r["away_score"]
        final = r["status"] == "final"
        winner = None
        if final and hs is not None and as_ is not None and hs != as_:
            winner = "home" if hs > as_ else "away"
        rec = done if final else before
        d = {"game_id": r["game_id"], "day": day, "status": r["status"],
             "home": r["home"], "away": r["away"], "home_score": hs, "away_score": as_,
             "winner": winner, "tipoff_utc": r.get("tipoff_utc"),
             "ot": _ot(f) if final else 0, "top": _top(f) if final else []}
        for side in ("home", "away"):
            wl = rec.get(r[side])
            d[f"{side}_rec"] = _rec(*wl) if wl else None
        out.append(d)
    return out


def _from_team_games(conn: sqlite3.Connection, day: str) -> list[dict]:
    """Older nights (no `games` rows): rebuild the cards from the two `team_games` rows."""
    by_game: dict[str, dict] = {}
    for r in _rows(conn, "SELECT game_id, team, opp, is_home, pts, season, min FROM team_games "
                         "WHERE game_date=? ORDER BY game_id", (day,)):
        g = by_game.setdefault(r["game_id"], {
            "game_id": r["game_id"], "season": r["season"], "nba_day": day, "tipoff_utc": None,
            "status": "final", "home": None, "away": None, "home_score": None,
            "away_score": None})
        side = "home" if r["is_home"] else "away"
        g[side] = r["team"]
        g[f"{side}_score"] = r["pts"]
    return [g for g in by_game.values() if g["home"] and g["away"]]


def ot_label(n: int) -> str:
    return OT_LABELS[min(max(n, 1), len(OT_LABELS)) - 1] if n else ""


# ------------------------------------------------------------------ DFS layer
def _salary_map(conn: sqlite3.Connection, game_id: str, tipoff: str | None) -> dict[int, int]:
    """{nba player_id: DK Classic salary} for the game (newest capture at or before tip-off)."""
    rows = _rows(conn, "SELECT site_player_id, player_name, salary, captured_at FROM dfs_salaries "
                       "WHERE game_id=? AND game_type_id=70 AND salary IS NOT NULL "
                       "ORDER BY captured_at", (game_id,))
    if not rows:
        return {}
    if tipoff:
        pre = [r for r in rows if (r["captured_at"] or "") <= tipoff]
        rows = pre or rows
    ids = {str(r[0]): int(r[1]) for r in _rows(
        conn, "SELECT source_id, player_id FROM player_ids WHERE source='dk' AND player_id IS NOT NULL "
              "AND source_id IN (SELECT site_player_id FROM dfs_salaries WHERE game_id=? "
              "AND game_type_id=70)", (game_id,))}
    names = {norm(r["player_name"]): int(r["player_id"]) for r in _rows(
        conn, "SELECT player_id, player_name FROM player_games WHERE game_id=?", (game_id,))}
    out: dict[int, int] = {}
    for r in rows:                                   # oldest first: the newest capture wins
        pid = ids.get(str(r["site_player_id"])) or names.get(norm(r["player_name"]))
        if pid:
            out[pid] = int(r["salary"])
    return out


def _proj_map(conn: sqlite3.Connection, game_id: str, tipoff: str | None) -> dict[int, float]:
    """{player_id: projected DK points}: latest ok run created before tip-off."""
    sql = ("SELECT p.player_id, s.mean FROM player_projections p "
           "JOIN player_projection_stats s ON s.projection_id=p.id AND s.stat='dk' "
           "AND s.cond='played' JOIN dfs_runs r ON r.run_id=p.run_id AND r.ok=1 "
           "WHERE p.game_id=?")
    args: tuple = (game_id,)
    if tipoff:
        sql += " AND p.created_at < ?"
        args += (tipoff,)
    sql += " ORDER BY p.created_at, p.id"
    return {int(r[0]): float(r[1]) for r in _rows(conn, sql, args) if r[1] is not None}


def _own_map(conn: sqlite3.Connection, game_id: str) -> dict[int, float]:
    """{player_id: average % drafted} over the contests of classic slates holding the game."""
    slates = [r[0] for r in _rows(
        conn, "SELECT slate_id FROM dfs_slates WHERE game_type_id=70 AND game_ids_json LIKE ?",
        (f'%"{game_id}"%',))]
    if not slates:
        return {}
    q = ",".join("?" * len(slates))
    return {int(r[0]): float(r[1]) for r in _rows(
        conn, f"SELECT o.player_id, AVG(o.pct_drafted) FROM dfs_contest_ownership o "
              f"JOIN dfs_contests c ON c.contest_id=o.contest_id WHERE c.slate_id IN ({q}) "
              f"AND o.pct_drafted IS NOT NULL GROUP BY o.player_id", tuple(slates))}


def dfs_for_game(conn: sqlite3.Connection, game_id: str, tipoff: str | None = None
                 ) -> dict[int, dict[str, Any]]:
    """{player_id: {"salary", "proj", "own"}} (None for the parts we do not have)."""
    sal, proj, own = (_salary_map(conn, game_id, tipoff), _proj_map(conn, game_id, tipoff),
                      _own_map(conn, game_id))
    return {pid: {"salary": sal.get(pid), "proj": proj.get(pid), "own": own.get(pid)}
            for pid in set(sal) | set(proj) | set(own)}


# ------------------------------------------------------------------ the box score
_TEAM_COLS = ("fgm fga fg3m fg3a ftm fta oreb dreb reb ast stl blk tov pf pts plus_minus min").split()


def _four_factors(away: dict, home: dict) -> list[dict]:
    """Rows for the four-factors card: eFG%, TOV%, OREB%, FT rate (+ off rating and pace).

    `better` is "away" / "home" / None; bar widths are the value over the larger side
    (plus 15%), the same scale on both sides of a row.
    """
    spec = [("eFG%", "efg", "pct", True), ("TOV%", "tov", "pct", False),
            ("OREB%", "oreb", "pct", True), ("FT rate", "ftr", "rate", True),
            ("Off rating", "ortg", "num", True), ("Pace", "pace", "num", True)]
    rows = []
    for label, key, kind, high in spec:
        a, h = away["ff"].get(key), home["ff"].get(key)
        if a is None and h is None:
            continue
        top = max(v for v in (a, h) if v is not None) * 1.15 or 1
        better = None
        if a is not None and h is not None and a != h:
            better = "away" if (a > h) == high else "home"
        rows.append({"label": label, "kind": kind, "away": a, "home": h, "better": better,
                     "away_w": round(100 * (a or 0) / top, 1), "home_w": round(100 * (h or 0) / top, 1)})
    return rows


def _team_block(tg: sqlite3.Row | None, adv: sqlite3.Row | None, opp: sqlite3.Row | None,
                code: str, score: int | None) -> dict:
    t = {c: (tg[c] if tg is not None else None) for c in _TEAM_COLS}
    a = dict(adv) if adv is not None else {}
    fga, fta, tov = t["fga"], t["fta"], t["tov"]
    efg = a.get("efg_pct")
    if efg is None and fga:
        efg = ((t["fgm"] or 0) + 0.5 * (t["fg3m"] or 0)) / fga
    tovp = a.get("tov_pct")
    if tovp is None and fga is not None and tov is not None:
        tovp = _div(tov, fga + 0.44 * (fta or 0) + tov)
    orebp = a.get("oreb_pct")
    if orebp is None and opp is not None and t["oreb"] is not None:
        orebp = _div(t["oreb"], t["oreb"] + (opp["dreb"] or 0))
    return {"team": code, "score": score if score is not None else t["pts"],
            "wl": tg["wl"] if tg is not None else None, "tot": t, "adv": a,
            "ff": {"efg": efg, "tov": tovp, "oreb": orebp, "ftr": _div(fta, fga),
                   "ortg": a.get("off_rating"), "pace": a.get("pace")}}


def game_box(conn: sqlite3.Connection | None, game_id: str) -> dict | None:
    """Box-score view model for one game, or None when no player lines are stored.

    Keys: game_id, day, season, season_type, status, tipoff_utc, home, away, periods
    ({labels, home, away} or None), ot, teams ([away, home], each: team, score, wl, tot, adv,
    ff, players, dnp, dk_total), four_factors, has_starts, has_dfs.
    A player row has the player_games columns plus: name, starter, dk, the player_games_adv
    columns (adv), salary, proj, own, value (DK points per $1k), diff (DK minus proj).
    """
    if conn is None:
        return None
    pg = _rows(conn, "SELECT * FROM player_games WHERE game_id=?", (game_id,))
    if not pg:
        return None
    g = _one(conn, "SELECT * FROM games WHERE game_id=?", (game_id,))
    tgs = {r["team"]: r for r in _rows(conn, "SELECT * FROM team_games WHERE game_id=?", (game_id,))}
    advs = {r["team"]: r for r in _rows(conn, "SELECT * FROM team_games_adv WHERE game_id=?", (game_id,))}
    if g is not None:
        home, away = g["home"], g["away"]
    else:
        home = next((t for t, r in tgs.items() if r["is_home"]), None)
        away = next((t for t, r in tgs.items() if not r["is_home"]), None)
        if not (home and away):
            codes = sorted({r["team"] for r in pg})
            if len(codes) != 2:
                return None
            away, home = codes
    padv = {r["player_id"]: dict(r) for r in _rows(
        conn, "SELECT * FROM player_games_adv WHERE game_id=?", (game_id,))}
    starts = {r[0] for r in _rows(conn, "SELECT player_id FROM player_starts WHERE game_id=?", (game_id,))}
    facts = _facts(conn, [game_id]).get(game_id, {})
    tipoff = g["tipoff_utc"] if g is not None else None
    dfs = dfs_for_game(conn, game_id, tipoff)

    teams = []
    for code, opp_code in ((away, home), (home, away)):
        score = None
        if g is not None:
            score = g["home_score"] if code == home else g["away_score"]
        tb = _team_block(tgs.get(code), advs.get(code), tgs.get(opp_code), code, score)
        players, dnp = [], []
        for r in pg:
            if r["team"] != code:
                continue
            p = dict(r)
            p["name"] = p["player_name"]
            p["starter"] = p["player_id"] in starts
            p["adv"] = padv.get(p["player_id"], {})
            p["dk"] = dk_points(r)
            d = dfs.get(p["player_id"], {})
            p["salary"], p["proj"], p["own"] = d.get("salary"), d.get("proj"), d.get("own")
            p["value"] = _div(p["dk"], p["salary"] / 1000) if p["salary"] else None
            p["diff"] = p["dk"] - p["proj"] if p["proj"] is not None else None
            played = (p["min"] or 0) > 0 or any(p[k] for k in ("fga", "fta", "reb", "ast", "tov", "pf"))
            (players if played else dnp).append(p)
        players.sort(key=lambda p: (not p["starter"], -(p["min"] or 0), p["name"] or ""))
        tb["players"], tb["dnp"] = players, [p["name"] for p in dnp]
        tb["dk_total"] = sum(p["dk"] for p in players)
        tb["proj_total"] = (sum(p["proj"] for p in players if p["proj"] is not None)
                            if any(p["proj"] is not None for p in players) else None)
        teams.append(tb)

    periods = None
    hp, ap = (facts.get("home") or {}).get("periods"), (facts.get("away") or {}).get("periods")
    if hp and ap:
        n = 4 + _ot(facts)
        periods = {"labels": [f"Q{i + 1}" if i < 4 else ot_label(i - 3) for i in range(n)],
                   "home": list(hp)[:n], "away": list(ap)[:n]}
    st = tgs.get(home)
    return {"game_id": game_id, "day": (g["nba_day"] if g is not None else None) or (st["game_date"] if st else None),
            "season": (g["season"] if g is not None else None) or (st["season"] if st else None),
            "season_type": st["season_type"] if st else None,
            "status": g["status"] if g is not None else "final", "tipoff_utc": tipoff,
            "home": home, "away": away, "periods": periods, "ot": _ot(facts) if facts else 0,
            "teams": teams, "four_factors": _four_factors(*teams),
            "has_starts": bool(starts),
            "has_dfs": any(p["salary"] is not None or p["proj"] is not None or p["own"] is not None
                           for t in teams for p in t["players"])}
