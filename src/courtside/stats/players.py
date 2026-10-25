"""Players: the directory and the player page (docs/STATS_SITE.md, mockup `#s-player`).

Pure functions on a read-only `sqlite3.Connection`; plain dicts/lists out. Every query
tolerates a missing table (returns empty) so a page shows its "not collected yet" state.

  league_table(conn, season, season_type)   per-game averages of every player + league ranks
                                            (cached per season/type)
  directory(conn, season, q=, team=, sort=)  the /players list (from the cached league table)
  header(conn, pid, today=, now=)            name, bio, team, next game, status, DK salary
  season_lines(conn, pid, window)            per season x type averages with ranks
  game_log(conn, pid, window, ...)           box score lines + DK pts/salary/value/real own%
  pick_projection(conn, game_id, pid, tip)   latest ok run created before tip
  projection_card(conn, pid, next_game, now) tonight's projection or the last graded game
  opponents / versus(conn, pid, opp, window) games vs a team + who guarded him
  hexbin(points, radius) / shot_chart(...)    server-side hexbin + zone table vs league
  play_types(conn, pid, season, type)        synergy offensive play types
  ownership(conn, pid)                       real DFS ownership (+ our pre-lock estimate)

Ranks: among players with at least max(2, 40% of the season's most games played) games; rank 1
is best, ties share the better rank. DK points always come from `proj.dfs.rules.dk_points`.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence

from .. import teams as _teams
from . import db as sdb
from . import seasons as S
from .search import norm

CLASSIC = 70                                   # DK classic game_type_id
RANKED = ("pts", "reb", "ast", "stl", "blk", "fg3m", "ts", "dk")
QUAL_SHARE, QUAL_MIN = 0.4, 2
SORTS = ("pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "min", "ts", "dk", "gp", "pm",
         "name", "team")
PAGE = 100
ZONES = ("Restricted Area", "Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3",
         "Above the Break 3", "Backcourt")
HEX_R = 12.0
# Court drawing frame: x in [-250, 250], y from the baseline (-47.5) to 372.5 (42 ft of court).
COURT_W, COURT_H, X0, Y0 = 500, 420, 250.0, 47.5
FG_RAMP = ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb")   # low -> high FG%
FG_LO, FG_HI = 0.25, 0.75
SYNERGY_THIN = 300            # fewer Offensive rows league-wide than this: "not enough yet"
PLAY_TYPES = {"PRRollMan": "P&R roll man", "PRBallHandler": "P&R ball handler",
              "Postup": "Post-up", "Spotup": "Spot-up", "Isolation": "Isolation",
              "Transition": "Transition", "Cut": "Cut", "OffRebound": "Putbacks",
              "OffScreen": "Off screen", "Misc": "Misc", "Handoff": "Handoff"}
LOG_COLS = ("min", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fgm", "fga",
            "ftm", "fta", "oreb", "dreb", "pf", "plus_minus")


# ------------------------------------------------------------------ small helpers
def _rows(conn: sqlite3.Connection | None, sql: str, args: Sequence = ()) -> list[dict]:
    if conn is None:
        return []
    try:
        cur = conn.execute(sql, tuple(args))
    except sqlite3.Error:
        return []
    cols = [d[0] for d in cur.description or ()]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _one(conn, sql: str, args: Sequence = ()) -> dict | None:
    r = _rows(conn, sql, args)
    return r[0] if r else None


def _dk(row) -> float:
    from ..proj.dfs.rules import dk_points           # lazy: keeps the web import light
    return float(dk_points(row))


def ts_pct(pts, fga, fta) -> float | None:
    tsa = 2.0 * ((fga or 0) + 0.44 * (fta or 0))
    return (pts or 0) / tsa if tsa > 0 else None


def _div(a, b) -> float | None:
    return a / b if (a is not None and b) else None


def _now_iso(now: datetime | None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _marks(xs: Sequence) -> str:
    return ",".join("?" * len(xs))


def full_name(first, last) -> str:
    return " ".join(x for x in ((first or "").strip(), (last or "").strip()) if x)


def names(conn, ids: Iterable[int]) -> dict[int, str]:
    """{player_id: display name}: player_index first + last, else the latest box-score name."""
    ids = sorted({int(i) for i in ids if i is not None})
    out: dict[int, str] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in _rows(conn, f"SELECT player_id, first_name, last_name FROM player_index "
                             f"WHERE player_id IN ({_marks(chunk)})", chunk):
            n = full_name(r["first_name"], r["last_name"])
            if n:
                out[int(r["player_id"])] = n
        miss = [p for p in chunk if p not in out]
        if miss:
            for r in _rows(conn, f"SELECT player_id, player_name FROM player_games WHERE "
                                 f"player_id IN ({_marks(miss)}) GROUP BY player_id", miss):
                if r["player_name"]:
                    out[int(r["player_id"])] = r["player_name"]
    return out


def _index_names(conn) -> dict[int, tuple[str, str | None]]:
    """{player_id: (name, position)} of the whole player_index (cached)."""
    def build(c):
        return {int(r["player_id"]): (full_name(r["first_name"], r["last_name"]), r["position"])
                for r in _rows(c, "SELECT player_id, first_name, last_name, position "
                                  "FROM player_index")}
    return sdb.cached(("pl-index-names",), build, conn, domain="box") or {}


def dk_ids(conn, player_id: int) -> list[str]:
    """DK playerIds mapped to this NBA player (the `player_ids` map of `proj.ids`)."""
    def build(c):
        try:
            from ..proj import ids as IDS
            m = IDS.nba_ids(c, "dk")
        except (sqlite3.Error, ImportError):
            return {}
        inv: dict[int, list[str]] = {}
        for dk, nba in m.items():
            inv.setdefault(int(nba), []).append(str(dk))
        return inv
    return list((sdb.cached(("pl-dk-ids",), build, conn, domain="box") or {})
                .get(int(player_id), []))


# ------------------------------------------------------------------ league table + ranks
def qualifier(max_gp: int) -> float:
    """Games needed to be ranked: 40% of the season's most games played, at least 2."""
    return max(float(QUAL_MIN), QUAL_SHARE * (max_gp or 0))


def rank_values(values: dict[Any, float | None], higher_better: bool = True) -> dict[Any, int]:
    """{key: rank} (1 = best; ties share the better rank). None values get no rank."""
    vals = sorted((v for v in values.values() if v is not None), reverse=higher_better)
    out = {}
    import bisect
    if higher_better:
        neg = [-v for v in vals]
        for k, v in values.items():
            if v is not None:
                out[k] = bisect.bisect_left(neg, -v) + 1
    else:
        for k, v in values.items():
            if v is not None:
                out[k] = bisect.bisect_left(vals, v) + 1
    return out


def _averages(acc: dict) -> dict:
    gp = acc["gp"]
    row = {k: acc[k] / gp for k in LOG_COLS if k in acc}
    row["gp"] = gp
    row["dk"] = acc["dk"] / gp
    row["ts"] = ts_pct(acc["pts"], acc["fga"], acc["fta"])
    row["fg_pct"] = _div(acc["fgm"], acc["fga"])
    row["fg3_pct"] = _div(acc["fg3m"], acc["fg3a"])
    row["ft_pct"] = _div(acc["ftm"], acc["fta"])
    row["pm"] = row.get("plus_minus")
    return row


def _accumulate(rows: Iterable[dict]) -> dict:
    acc = {k: 0.0 for k in LOG_COLS}
    acc["gp"] = 0
    acc["dk"] = 0.0
    for r in rows:
        acc["gp"] += 1
        for k in LOG_COLS:
            acc[k] += r.get(k) or 0
        acc["dk"] += r["dk"] if "dk" in r else _dk(r)
    return acc


def league_table(conn, season: str, season_type: str = S.REGULAR) -> dict:
    """Per-game averages of every player with a game (min > 0) in season/type, with ranks.

    {"players": {pid: {name, team, gp, min, pts, ..., ts, dk, ranks: {stat: rank}}},
     "n": qualified players, "min_gp": games needed, "max_gp": most games played}"""
    def build(c):
        rows = _rows(c, "SELECT player_id, player_name, team, game_date, "
                        + ", ".join(LOG_COLS) + " FROM player_games "
                        "WHERE season=? AND season_type=? AND min > 0", (season, season_type))
        accs: dict[int, dict] = {}
        last: dict[int, tuple[str, str, str]] = {}
        for r in rows:
            pid = int(r["player_id"])
            r["dk"] = _dk(r)
            a = accs.get(pid)
            if a is None:
                a = accs[pid] = {k: 0.0 for k in LOG_COLS} | {"gp": 0, "dk": 0.0}
            a["gp"] += 1
            for k in LOG_COLS:
                a[k] += r[k] or 0
            a["dk"] += r["dk"]
            d = r["game_date"] or ""
            if pid not in last or d > last[pid][0]:
                last[pid] = (d, r["team"] or "", r["player_name"] or "")
        idx = _index_names(c)
        players = {}
        for pid, a in accs.items():
            row = _averages(a)
            nm, pos = idx.get(pid, (None, None))
            row.update(player_id=pid, name=nm or last[pid][2], team=last[pid][1],
                       position=pos, last_game=last[pid][0])
            players[pid] = row
        max_gp = max((p["gp"] for p in players.values()), default=0)
        need = qualifier(max_gp)
        qual = {pid for pid, p in players.items() if p["gp"] >= need}
        ranks = {k: rank_values({pid: players[pid].get(k) for pid in qual}) for k in RANKED}
        for pid, p in players.items():
            p["qualified"] = pid in qual
            p["ranks"] = {k: ranks[k].get(pid) for k in RANKED}
        return {"players": players, "n": len(qual), "min_gp": need, "max_gp": max_gp,
                "season": season, "season_type": season_type}
    return sdb.cached(("pl-league", season, season_type), build, conn, domain="box") or {
        "players": {}, "n": 0, "min_gp": QUAL_MIN, "max_gp": 0}


def directory(conn, season: str, *, q: str = "", team: str = "", sort: str = "pts",
              desc: bool = True, page: int = 1, per_page: int = PAGE,
              season_type: str = S.REGULAR) -> dict:
    """The /players list: {rows, total, page, pages, teams, sort, n_qualified, min_gp}."""
    lt = league_table(conn, season, season_type)
    rows = list(lt["players"].values())
    teams = sorted({r["team"] for r in rows if r["team"]})
    nq = norm(q)
    if nq:
        words = nq.split()
        rows = [r for r in rows if all(any(t.startswith(w) for t in norm(r["name"]).split())
                                       for w in words) or nq in norm(r["name"])]
    team = (team or "").upper()
    if team:
        rows = [r for r in rows if r["team"] == team]
    sort = sort if sort in SORTS else "pts"
    if sort in ("name", "team"):
        rows.sort(key=lambda r: (norm(r[sort]), norm(r["name"])), reverse=not desc)
    else:
        rows.sort(key=lambda r: (r.get(sort) is None, -(r.get(sort) or 0) if desc
                                 else (r.get(sort) or 0), norm(r["name"])))
    total = len(rows)
    pages = max(1, math.ceil(total / per_page))
    page = min(max(1, page), pages)
    return {"rows": rows[(page - 1) * per_page: page * per_page], "total": total,
            "page": page, "pages": pages, "teams": teams, "sort": sort, "desc": desc,
            "n_qualified": lt["n"], "min_gp": lt["min_gp"], "offset": (page - 1) * per_page}


# ------------------------------------------------------------------ header
def _bio(conn, pid: int) -> dict | None:
    return _one(conn, "SELECT * FROM player_index WHERE player_id=?", (pid,))


def _last_box(conn, pid: int) -> dict | None:
    return _one(conn, "SELECT player_name, team, season, game_date FROM player_games "
                      "WHERE player_id=? ORDER BY game_date DESC LIMIT 1", (pid,))


def current_team(bio: dict | None, last: dict | None, today: date) -> str | None:
    """The newest box-score team in the current season, else the player_index team."""
    cur = S.current_season(today)
    if last and last.get("season") == cur and last.get("team"):
        return last["team"]
    if bio and bio.get("team"):
        return bio["team"]
    return (last or {}).get("team") or None


def next_game(conn, team: str | None, today: date) -> dict | None:
    """His team's next scheduled (or live) game from `games`: {game_id, nba_day, tipoff_utc,
    opp, is_home, status}."""
    if not team:
        return None
    r = _one(conn, "SELECT game_id, nba_day, tipoff_utc, home, away, status FROM games "
                   "WHERE (home=? OR away=?) AND nba_day>=? AND status IN ('scheduled','live') "
                   "ORDER BY nba_day, tipoff_utc LIMIT 1", (team, team, today.isoformat()))
    if not r:
        return None
    home = r["home"] == team
    return {"game_id": r["game_id"], "nba_day": r["nba_day"], "tipoff_utc": r["tipoff_utc"],
            "status": r["status"], "is_home": home, "opp": r["away"] if home else r["home"]}


def injury_status(conn, pid: int, team: str | None, now: datetime | None = None,
                  season: str | None = None) -> dict | None:
    """{status, reason, source, captured_at} from the latest injury reports for him (Out /
    Doubtful / Questionable listings only), or None."""
    if not team:
        return None
    at = _now_iso(now)
    try:
        from .. import injuries
    except ImportError:
        return None

    def build(c):
        try:
            cur = injuries.current(c, at, {team})
        except (sqlite3.Error, ValueError, KeyError):
            return []
        entries = (cur.get("by_team") or {}).get(team) or []
        if not entries:
            return []
        roster = {int(r["player_id"]): r["name"] for r in _rows(
            c, "SELECT player_id, player_name AS name FROM player_games WHERE team=? AND "
               "season=? GROUP BY player_id", (team, season or S.current_season()))}
        for r in _rows(c, "SELECT player_id, first_name, last_name FROM player_index "
                          "WHERE team=?", (team,)):
            roster[int(r["player_id"])] = full_name(r["first_name"], r["last_name"])
        pairs = [(p, n) for p, n in roster.items() if n]
        out = []
        for e in entries:
            mp, _ = injuries.match_player(e.get("player"), pairs, e.get("player_id"))
            if mp is not None:
                out.append((int(mp), {"status": e.get("status"), "reason": e.get("reason"),
                                      "source": e.get("source"),
                                      "captured_at": e.get("captured_at")}))
        return out
    for mp, st in sdb.cached(("pl-inj", team, at[:13]), build, conn,
                             domain=("box", "injuries")) or ():
        if mp == pid:
            return st
    return None


def current_salary(conn, pid: int) -> dict | None:
    """Latest DK classic salary: {salary, positions, nba_day, captured_at}."""
    ids = dk_ids(conn, pid)
    if not ids:
        return None
    return _one(conn, f"SELECT d.salary, d.positions, d.captured_at, s.nba_day "
                      f"FROM dfs_salaries d LEFT JOIN dfs_slates s ON s.slate_id=d.slate_id "
                      f"WHERE d.site_player_id IN ({_marks(ids)}) AND d.game_type_id=? "
                      f"AND d.salary IS NOT NULL ORDER BY d.captured_at DESC LIMIT 1",
                (*ids, CLASSIC))


def header(conn, pid: int, *, today: date, now: datetime | None = None) -> dict | None:
    """Everything the hero shows; None when the id is unknown."""
    bio = _bio(conn, pid)
    last = _last_box(conn, pid)
    if not bio and not last:
        return None
    name = full_name((bio or {}).get("first_name"), (bio or {}).get("last_name")) \
        or (last or {}).get("player_name") or f"Player {pid}"
    team = current_team(bio, last, today)
    b = bio or {}
    draft = None
    if b.get("draft_year"):
        if b.get("draft_round") and b.get("draft_number"):
            draft = f"drafted {b['draft_year']} R{b['draft_round']} #{b['draft_number']}"
        else:
            draft = f"{b['draft_year']} undrafted"
    initials = "".join(w[0] for w in name.replace("-", " ").split() if w[:1].isalpha())[:3]
    return {"player_id": pid, "name": name, "initials": initials.upper() or "?", "team": team,
            "position": b.get("position"), "jersey": b.get("jersey"), "height": b.get("height"),
            "weight": b.get("weight"), "country": b.get("country"), "college": b.get("college"),
            "draft": draft, "last_game": (last or {}).get("game_date"),
            "next": next_game(conn, team, today),
            "status": injury_status(conn, pid, team, now, S.current_season(today)),
            "salary": current_salary(conn, pid)}


# ------------------------------------------------------------------ season lines
def season_lines(conn, pid: int, window: Sequence[str]) -> list[dict]:
    """[{season, season_type, gp, min, ..., dk, ranks, n}] newest first, Regular then Playoffs,
    only where he played."""
    out = []
    for s in window:
        for t in (S.REGULAR, S.PLAYOFFS):
            lt = league_table(conn, s, t)
            p = lt["players"].get(pid)
            if p:
                out.append({**p, "season": s, "season_type": t, "n": lt["n"],
                            "min_gp": lt["min_gp"]})
    return out


# ------------------------------------------------------------------ game log
def _box_rows(conn, pid: int, window: Sequence[str]) -> list[dict]:
    """His box scores in the window (newest first) with opp / home / W-L and DK points."""
    def build(c):
        rows = _rows(c, f"SELECT pg.game_id, pg.team, pg.season, pg.season_type, pg.game_date, "
                        + ", ".join(f"pg.{k}" for k in LOG_COLS) +
                        f", tg.opp, tg.is_home, tg.wl FROM player_games pg LEFT JOIN team_games tg"
                        f" ON tg.game_id=pg.game_id AND tg.team=pg.team WHERE pg.player_id=? "
                        f"AND pg.season IN ({_marks(window)}) ORDER BY pg.game_date DESC",
                     (pid, *window))
        for r in rows:
            r["dk"] = _dk(r)
            r["ts"] = ts_pct(r["pts"], r["fga"], r["fta"])
            r["played"] = (r["min"] or 0) > 0
        return rows
    return sdb.cached(("pl-box", pid, tuple(window)), build, conn, domain="box") or []


def salary_by_day(conn, pid: int) -> dict[str, int]:
    """{nba_day: DK classic salary} (latest capture of the day; RotoGuru for old seasons)."""
    out: dict[str, tuple[str, int]] = {}
    for r in _rows(conn, "SELECT game_date, salary FROM dfs_hist_salaries WHERE player_id=? "
                         "AND salary IS NOT NULL", (pid,)):
        out[r["game_date"]] = ("", int(r["salary"]))
    ids = dk_ids(conn, pid)
    if ids:
        for r in _rows(conn, f"SELECT s.nba_day, d.salary, d.captured_at FROM dfs_salaries d "
                             f"JOIN dfs_slates s ON s.slate_id=d.slate_id WHERE d.site_player_id"
                             f" IN ({_marks(ids)}) AND d.game_type_id=? AND d.salary IS NOT NULL",
                       (*ids, CLASSIC)):
            day, cap = r["nba_day"], r["captured_at"] or ""
            if day not in out or cap >= out[day][0]:
                out[day] = (cap, int(r["salary"]))
    return {d: v[1] for d, v in out.items()}


def own_by_day(conn, pid: int) -> dict[str, float]:
    """{nba_day: average real %drafted over the classic contests of that day's slates}."""
    rows = _rows(conn, "SELECT s.nba_day, AVG(o.pct_drafted) AS own FROM dfs_contest_ownership o "
                       "JOIN dfs_contests c ON c.contest_id=o.contest_id "
                       "JOIN dfs_slates s ON s.slate_id=c.slate_id WHERE o.player_id=? "
                       "AND o.pct_drafted IS NOT NULL AND s.game_type_id=? GROUP BY s.nba_day",
                 (pid, CLASSIC))
    return {r["nba_day"]: r["own"] for r in rows}


def game_log(conn, pid: int, window: Sequence[str], *, season: str | None = None,
             season_type: str | None = None) -> list[dict]:
    """Newest first. season / season_type (None = all) filter the window."""
    sal = salary_by_day(conn, pid)
    own = own_by_day(conn, pid)
    out = []
    for r in _box_rows(conn, pid, window):
        if season and r["season"] != season:
            continue
        if season_type and r["season_type"] != season_type:
            continue
        r = dict(r)
        r["salary"] = sal.get(r["game_date"])
        r["value"] = r["dk"] / r["salary"] * 1000 if r["salary"] else None
        r["own"] = own.get(r["game_date"])
        out.append(r)
    return out


# ------------------------------------------------------------------ projection
def _tip(conn, game_id: str) -> str | None:
    r = _one(conn, "SELECT tipoff_utc FROM games WHERE game_id=?", (game_id,))
    return r["tipoff_utc"] if r else None


def pick_projection(conn, game_id: str, pid: int, tip: str | None) -> dict | None:
    """His projection from the latest `dfs_runs.ok=1` run created before tip (any ok run when the
    tip time is unknown), with its stats (cond='played') and grade when graded."""
    p = _one(conn, "SELECT p.id, p.game_id, p.nba_day, p.slate_id, p.run_id, p.created_at, "
                   "p.model_version, p.role, p.p_play, r.kind, r.created_at AS run_at "
                   "FROM player_projections p JOIN dfs_runs r ON r.run_id=p.run_id "
                   "WHERE p.game_id=? AND p.player_id=? AND r.ok=1 "
                   "AND (? IS NULL OR r.created_at < ?) "
                   "ORDER BY r.created_at DESC, p.created_at DESC, p.id DESC LIMIT 1",
             (game_id, pid, tip, tip))
    if not p:
        return None
    stats = {}
    for r in _rows(conn, "SELECT stat, mean, q10, q90 FROM player_projection_stats "
                         "WHERE projection_id=? AND cond='played'", (p["id"],)):
        stats[r["stat"]] = {"mean": r["mean"], "q10": r["q10"], "q90": r["q90"]}
    p["stats"] = stats
    p["grade"] = _one(conn, "SELECT * FROM player_projection_grades WHERE projection_id=?",
                      (p["id"],))
    return p


def projection_card(conn, pid: int, nxt: dict | None, now: datetime | None = None,
                    window: Sequence[str] | None = None) -> dict | None:
    """{mode: 'tonight'|'graded', game: {...}, proj: {...}, actual: {...}|None} or None.

    Game day before tip: tonight's projection. Otherwise the most recent game of his that has a
    graded projection, projection vs actual."""
    at = _now_iso(now)
    if nxt and nxt.get("tipoff_utc") and at < nxt["tipoff_utc"] \
            and nxt.get("nba_day") and nxt["nba_day"] <= _today_of(now):
        pr = pick_projection(conn, nxt["game_id"], pid, nxt["tipoff_utc"])
        if pr:
            return {"mode": "tonight", "game": nxt, "proj": pr, "actual": None}
    recent = _rows(conn, "SELECT pg.game_id, pg.game_date, pg.team, pg.min, pg.pts, pg.reb, "
                         "pg.ast, pg.fg3m, pg.stl, pg.blk, pg.tov, tg.opp, tg.is_home "
                         "FROM player_games pg LEFT JOIN team_games tg ON tg.game_id=pg.game_id "
                         "AND tg.team=pg.team WHERE pg.player_id=? "
                         "ORDER BY pg.game_date DESC LIMIT 15", (pid,))
    if not recent:
        return None
    gids = [r["game_id"] for r in recent]
    graded = {r["game_id"] for r in _rows(
        conn, f"SELECT DISTINCT game_id FROM player_projection_grades WHERE game_id IN "
              f"({_marks(gids)}) AND player_id=?", (*gids, pid))}
    for g in recent:
        if g["game_id"] not in graded:
            continue
        pr = pick_projection(conn, g["game_id"], pid, _tip(conn, g["game_id"]))
        if not pr:
            continue
        actual = {k: g[k] for k in ("min", "pts", "reb", "ast", "fg3m")}
        actual["dk"] = _dk(g)
        game = {"game_id": g["game_id"], "nba_day": g["game_date"], "opp": g["opp"],
                "is_home": bool(g["is_home"]), "team": g["team"]}
        return {"mode": "graded", "game": game, "proj": pr, "actual": actual}
    return None


def _today_of(now: datetime | None) -> str:
    from .. import config
    return config.nba_day(now or datetime.now(timezone.utc)).isoformat()


# ------------------------------------------------------------------ versus a team
def opponents(conn, pid: int, window: Sequence[str]) -> list[dict]:
    """[{team, games}] he faced in the window, by team code."""
    cnt: dict[str, int] = {}
    for r in _box_rows(conn, pid, window):
        if r["opp"] and r["played"]:
            cnt[r["opp"]] = cnt.get(r["opp"], 0) + 1
    return [{"team": t, "games": n} for t, n in sorted(cnt.items())]


def guarded_by(conn, pid: int, game_ids: Sequence[str]) -> list[dict]:
    """Who guarded him in these games (player_matchups, aggregated by defender), most
    possessions first: {def_player_id, name, team, games, poss, min, pts, fgm, fga, fg3m, fg3a,
    ast, tov, p100}."""
    if not game_ids:
        return []
    rows = _rows(conn, f"SELECT def_player_id, MAX(def_team) AS team, COUNT(DISTINCT game_id) AS "
                       f"games, SUM(partial_poss) AS poss, SUM(matchup_min) AS min, SUM(pts) AS "
                       f"pts, SUM(fgm) AS fgm, SUM(fga) AS fga, SUM(fg3m) AS fg3m, SUM(fg3a) AS "
                       f"fg3a, SUM(ast) AS ast, SUM(tov) AS tov FROM player_matchups "
                       f"WHERE game_id IN ({_marks(game_ids)}) AND off_player_id=? "
                       f"GROUP BY def_player_id", (*game_ids, pid))
    nm = names(conn, [r["def_player_id"] for r in rows])
    for r in rows:
        r["name"] = nm.get(int(r["def_player_id"]), f"#{r['def_player_id']}")
        r["p100"] = 100.0 * (r["pts"] or 0) / r["poss"] if r["poss"] else None
    rows.sort(key=lambda r: -(r["poss"] or 0))
    return rows


def versus(conn, pid: int, opp: str, window: Sequence[str]) -> dict:
    """{opp, games: [...newest first], avg: {...}, base: {...}, diff: {...}, guarded: [...]}.
    avg/base are per-game over games he played (base: all his games in the window)."""
    box = _box_rows(conn, pid, window)
    played = [r for r in box if r["played"]]
    games = [r for r in played if r["opp"] == opp]
    keys = ("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "dk")
    avg = _averages(_accumulate(games)) if games else {}
    base = _averages(_accumulate(played)) if played else {}
    diff = {k: avg[k] - base[k] for k in keys + ("ts",)
            if avg.get(k) is not None and base.get(k) is not None}
    return {"opp": opp, "games": games, "avg": avg, "base": base, "diff": diff,
            "guarded": guarded_by(conn, pid, [g["game_id"] for g in games])}


# ------------------------------------------------------------------ shot chart
def hex_center(x: float, y: float, r: float = HEX_R) -> tuple[float, float]:
    """Centre of the pointy-top hexagon (circumradius r) containing (x, y) (d3-hexbin)."""
    dx, dy = r * 2 * math.sin(math.pi / 3), r * 1.5
    py = y / dy
    pj = round(py)
    px = x / dx - (pj & 1) / 2
    pi = round(px)
    py1 = py - pj
    if abs(py1) * 3 > 1:
        px1 = px - pi
        pi2 = pi + (-1 if px < pi else 1) / 2
        pj2 = pj + (-1 if py < pj else 1)
        px2 = px - pi2
        py2 = py - pj2
        if px1 * px1 + py1 * py1 > px2 * px2 + py2 * py2:
            pi = pi2 + (1 if pj & 1 else -1) / 2
            pj = pj2
    return (pi + (pj & 1) / 2) * dx, pj * dy


def hexbin(points: Iterable[tuple[float, float, int]], r: float = HEX_R) -> list[dict]:
    """[(x, y, made)] -> [{x, y, n, made}] per hexagon centre (shot-chart coordinates)."""
    bins: dict[tuple[float, float], list[int]] = {}
    for x, y, made in points:
        if x is None or y is None:
            continue
        c = hex_center(float(x), float(y), r)
        key = (round(c[0], 3), round(c[1], 3))
        b = bins.setdefault(key, [0, 0])
        b[0] += 1
        b[1] += 1 if made else 0
    return [{"x": k[0], "y": k[1], "n": v[0], "made": v[1]} for k, v in bins.items()]


def fg_color(p: float | None) -> str:
    if p is None:
        return FG_RAMP[0]
    t = max(0.0, min(0.999, (p - FG_LO) / (FG_HI - FG_LO)))
    return FG_RAMP[int(t * len(FG_RAMP))]


def hex_points(cx: float, cy: float, r: float) -> str:
    return " ".join(f"{cx + r * math.cos(math.radians(60 * i - 30)):.1f},"
                    f"{cy + r * math.sin(math.radians(60 * i - 30)):.1f}" for i in range(6))


def percentile(values: Sequence[float], q: float) -> float:
    vs = sorted(values)
    if not vs:
        return 0.0
    return vs[min(len(vs) - 1, int(math.floor(len(vs) * q)))]


def league_zones(conn, season: str, season_type: str) -> dict[str, dict]:
    """{zone: {fga, fgm, fg_pct}} for the whole league (cached; uses the (team, season) index)."""
    def build(c):
        codes = [t.tricode for t in _teams.TEAMS]
        out = {}
        for r in _rows(c, f"SELECT zone, COUNT(*) AS fga, SUM(made) AS fgm FROM shots "
                          f"WHERE team IN ({_marks(codes)}) AND season=? AND season_type=? "
                          f"GROUP BY zone", (*codes, season, season_type)):
            if r["zone"]:
                out[r["zone"]] = {"fga": r["fga"], "fgm": r["fgm"] or 0,
                                  "fg_pct": _div(r["fgm"] or 0, r["fga"])}
        return out
    return sdb.cached(("pl-league-zones", season, season_type), build, conn, domain="box") or {}


def shot_chart(conn, pid: int, season: str, season_type: str = S.REGULAR) -> dict:
    """{n, games, made, hexes: [{points, fill, tip, n}], zones: [...], seasons_with_shots}.
    n == 0 -> empty state (no table / no rows)."""
    def build(c):
        rows = _rows(c, "SELECT game_id, x, y, made, zone FROM shots WHERE player_id=? AND "
                        "season=? AND season_type=?", (pid, season, season_type))
        out: dict[str, Any] = {"n": len(rows), "season": season, "season_type": season_type,
                               "games": len({r["game_id"] for r in rows}),
                               "made": sum(1 for r in rows if r["made"]),
                               "hexes": [], "zones": []}
        if not rows:
            return out
        bins = hexbin(((r["x"], r["y"], r["made"]) for r in rows))
        bins = [b for b in bins if -X0 <= b["x"] <= X0 and -Y0 <= b["y"] <= COURT_H - Y0]
        n = len(rows)
        floor = 3 if n >= 600 else 2 if n >= 200 else 1
        shown = [b for b in bins if b["n"] >= floor] or bins
        cap = max(1, percentile([b["n"] for b in shown], 0.85))
        for b in sorted(shown, key=lambda b: b["n"]):
            p = b["made"] / b["n"]
            size = 3.5 + (HEX_R - 3.5) * math.sqrt(min(b["n"], cap) / cap)
            out["hexes"].append({
                "points": hex_points(b["x"] + X0, b["y"] + Y0, size), "fill": fg_color(p),
                "n": b["n"], "made": b["made"],
                "tip": f"{b['n']} shot{'s' if b['n'] != 1 else ''} · {b['made']} made · "
                       f"{round(100 * p)}%"})
        lz = league_zones(c, season, season_type)
        zc: dict[str, list[int]] = {}
        for r in rows:
            z = zc.setdefault(r["zone"] or "Unknown", [0, 0])
            z[0] += 1
            z[1] += 1 if r["made"] else 0
        order = list(ZONES) + sorted(k for k in zc if k not in ZONES)
        for zname in order:
            if zname not in zc:
                continue
            fga, fgm = zc[zname]
            lg = (lz.get(zname) or {}).get("fg_pct")
            fg = fgm / fga
            out["zones"].append({"zone": zname, "fga": fga, "fgm": fgm, "share": fga / n,
                                 "fg_pct": fg, "league": lg,
                                 "diff": fg - lg if lg is not None else None})
        return out
    return sdb.cached(("pl-shots", pid, season, season_type), build, conn, domain="box")


def court_svg_paths() -> list[str]:
    """SVG path data of the half court in the chart frame (500 x 420, rim at (250, 47.5))."""
    return [
        f"M0 0 H{COURT_W} V{COURT_H} H0 Z",                                 # boundary
        "M170 0 V190 H330 V0",                                              # lane (16 ft)
        "M190 0 V190 M310 0 V190",                                          # inner lane lines
        "M310 190 A60 60 0 0 1 190 190",                                    # FT circle (top half)
        "M30 0 V89.5 A237.5 237.5 0 0 0 470 89.5 V0",                       # 3-pt line
        "M210 47.5 A40 40 0 0 0 290 47.5",                                  # restricted area
        "M220 40 H280",                                                     # backboard
    ]


# ------------------------------------------------------------------ play types
def play_types(conn, pid: int, season: str, season_type: str = S.REGULAR) -> dict:
    """{rows: [{play_type, label, poss, poss_pct, ppp, percentile, pts}], thin: bool,
    league_rows: n}. thin: nba.com has few rows yet for that season (early season)."""
    lg = _one(conn, "SELECT COUNT(*) AS n FROM synergy_playtypes WHERE season=? AND "
                    "season_type=? AND entity='P' AND grouping='Offensive'",
              (season, season_type))
    n_league = (lg or {}).get("n") or 0
    rows = _rows(conn, "SELECT play_type, team_id, team, poss, poss_pct, ppp, pts, percentile, gp "
                       "FROM synergy_playtypes WHERE season=? AND season_type=? AND entity='P' "
                       "AND grouping='Offensive' AND player_id=?", (season, season_type, pid))
    by: dict[str, dict] = {}
    for r in rows:                          # traded players: one row per team, combine them
        b = by.get(r["play_type"])
        if b is None:
            by[r["play_type"]] = dict(r, _best=r["poss"] or 0)
            continue
        b["poss"] = (b["poss"] or 0) + (r["poss"] or 0)
        b["pts"] = (b["pts"] or 0) + (r["pts"] or 0)
        if (r["poss"] or 0) > b["_best"]:
            b["percentile"], b["_best"] = r["percentile"], r["poss"] or 0
    total = sum((b["poss"] or 0) for b in by.values())
    out = []
    multi = len({r["team_id"] for r in rows}) > 1
    for pt, b in by.items():
        poss = b["poss"] or 0
        out.append({"play_type": pt, "label": PLAY_TYPES.get(pt, pt), "poss": poss,
                    "poss_pct": (poss / total if total else None) if multi else b["poss_pct"],
                    "ppp": (b["pts"] / poss if poss else None) if multi else b["ppp"],
                    "pts": b["pts"], "percentile": b["percentile"],
                    "pctile": (round(100 * b["percentile"]) if b["percentile"] is not None
                               else None)})
    out.sort(key=lambda r: -(r["poss"] or 0))
    return {"rows": out, "thin": n_league < SYNERGY_THIN, "league_rows": n_league,
            "season": season, "season_type": season_type}


# ------------------------------------------------------------------ real DFS ownership
def ownership(conn, pid: int, limit: int = 60) -> list[dict]:
    """Uploaded-standings ownership rows, newest first: {nba_day, contest_id, contest, field,
    entry_fee, own, fpts, slate_id, game_type_id, est}. `est` is our pre-lock ownership estimate
    (`dfs_ownership_proxy` of the latest ok run on that slate created before lock) when we have
    one for him."""
    rows = _rows(conn, "SELECT o.contest_id, o.pct_drafted AS own, o.fpts, c.name AS contest, "
                       "c.field_size AS field, c.entry_fee, c.slate_id, s.nba_day, "
                       "s.game_type_id, s.lock_utc FROM dfs_contest_ownership o "
                       "LEFT JOIN dfs_contests c ON c.contest_id=o.contest_id "
                       "LEFT JOIN dfs_slates s ON s.slate_id=c.slate_id WHERE o.player_id=? "
                       "ORDER BY s.nba_day DESC, c.field_size DESC LIMIT ?", (pid, limit))
    if not rows:
        return []
    ids = dk_ids(conn, pid)
    est_by_slate: dict[str, float | None] = {}
    for r in rows:
        sid = r["slate_id"]
        if sid and ids and sid not in est_by_slate:
            e = _one(conn, f"SELECT o.own FROM dfs_runs r JOIN dfs_ownership_proxy o ON "
                           f"o.run_id=r.run_id WHERE r.slate_id=? AND r.ok=1 AND o.player_id IN "
                           f"({_marks(ids)}) AND (? IS NULL OR r.created_at <= ?) "
                           f"ORDER BY r.created_at DESC LIMIT 1",
                     (sid, *[int(i) if str(i).isdigit() else i for i in ids],
                      r["lock_utc"], r["lock_utc"]))
            est_by_slate[sid] = e["own"] if e else None
        r["est"] = est_by_slate.get(sid) if sid else None
        r["contest"] = (r["contest"] or f"Contest {r['contest_id']}").strip()
    return rows
