"""The `/players` page: Tonight player cards (docs/PLAYERS.md T2.9, §2.8).

Per game, per player: our projection if he plays (minutes, points, rebounds, assists, 3PM, DK
points, each with a 10th-90th percentile range), availability (the injury report status and our
P(play)), and the book's main props shown as "our number vs the market" with both chances.
When the run's sims carry availability scenarios (`proj/scenarios.py`, P10), a card also shows
how his numbers move with a teammate who may or may not play (0.05 < P(available) < 0.95, at
least 200 sims in each arm): minutes and DK points if he plays, and the big-night chance.
A "Teammate out" tag (`proj/dfs/late_news_tag.py`, D4) marks a teammate of a starter or rotation
player listed Doubtful or Out shortly before lock whose projection rose since the run before.

Information only (STYLE.md, owner rule): the page never says "over", "edge", "lean", "value",
"smash" or "lock"; a prop reads "Points 24.5: our model 58% / market 52%". Our chance comes
from proj/props.py (the run's minutes distribution x a count model); the market's is no-vig.

Read-only: runs on the app's read-only SQLite connection and never migrates. The view model is
cached per (day, newest run, newest prop capture, newest injury capture).

Routes
  /players                  redirect to today's NBA day
  /players/<day>            ?game=<game_id> (filter) &sort=dk|pts|reb|ast|min|name
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import injuries
from ..proj import props as P
from ..proj import scenarios as SCN
from ..proj.dfs import late_news_tag as LN
from ..proj.dfs import xfactor as XF
from . import fmt
from .teams import identity

SORTS = {"dk": "Projection", "pts": "Points", "reb": "Rebounds", "ast": "Assists",
         "min": "Minutes", "name": "Name"}
CARD_MIN_DK = 12.0          # rotation cut: projected DK (all) for a player without props
STAT_ROWS = (("min", "Min"), ("pts", "Pts"), ("reb", "Reb"), ("ast", "Ast"), ("fg3m", "3PM"),
             ("dk", "DK"))
STATUS_CLASS = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable",
                "Probable": "probable", "Available": "healthy"}
DISCLAIMER = ("Information, not betting advice. Our chance comes from our projection model; "
              "the market’s comes from the book’s prices with the bookmaker margin "
              "removed. Neither is a promise.")


def _num(x: Any) -> float | None:
    return P._num(x)


def _q(conn: sqlite3.Connection, sql: str, args=()) -> list:
    try:
        return conn.execute(sql, tuple(args)).fetchall()
    except sqlite3.Error:
        return []


def fingerprint(conn: sqlite3.Connection, day: str) -> tuple:
    """Cheap cache key: what the page depends on changes when any of these move."""
    g = _q(conn, "SELECT COUNT(*), MAX(updated_at) FROM games WHERE nba_day=?", (day,))
    r = _q(conn, "SELECT MAX(created_at), COUNT(*) FROM dfs_runs")
    p = _q(conn, "SELECT MAX(id) FROM prop_snapshots")        # append-only change logs
    i = _q(conn, "SELECT MAX(id) FROM injury_snapshots")
    return (day, *(tuple(x[0]) if x else () for x in (g, r, p, i)))


def _status_map(cur: dict, teams: set[str], names: dict[int, list[str]],
                team_of: dict[int, str]) -> dict[int, dict]:
    """{player_id: {status, reason}} for the teams from `injuries.current(...)`."""
    out: dict[int, dict] = {}
    for team, entries in (cur.get("by_team") or {}).items():
        if team not in teams:
            continue
        roster = [(pid, ns[0]) for pid, ns in names.items() if ns and team_of.get(pid) == team]
        for e in entries:
            pid = e.get("player_id")
            pid = int(pid) if pid is not None and str(pid).isdigit() else None
            if pid is None or pid not in names:
                pid, _ = injuries.match_player(e.get("player"), roster)
            if pid is not None and pid not in out:
                out[pid] = {"status": e.get("status"), "reason": fmt.injury_reason(e.get("reason"))}
    return out


def _prop_view(r: dict) -> dict:
    ours = P.comparable(r["our_p_above"], r["our_p_push"])
    line = f"{r['line']:g}"
    label = P.LABEL[r["stat"]]
    text = f"{label} {line}: our model {fmt.pct(ours)} / market {fmt.pct(r['mkt_p_above'])}"
    return {"stat": r["stat"], "label": label, "line": line, "ours": ours,
            "mkt": r["mkt_p_above"], "push": r["our_p_push"] or 0.0, "text": text,
            "gap": None if ours is None or r["mkt_p_above"] is None
            else ours - r["mkt_p_above"]}


def _split_view(splits: list[dict], names: Mapping[int, list[str]]) -> list[dict]:
    """The card's teammate splits (`scenarios.card_splits`): per teammate, the "sits" then
    the "plays" arm (minutes and DK if he plays, big-night chance, share of sims)."""
    out = []
    for s in SCN.card_splits(splits):
        nm = (names.get(s["teammate"]) or [None])[0]
        if not nm:
            continue
        tot = max(s["plays"]["n"] + s["sits"]["n"], 1)
        rows = [{"label": lab, "min": a["min_played"], "dk": a["dk_played"], "big": a["big"],
                 "n": a["n"], "share": a["n"] / tot}
                for lab, a in (("sits", s["sits"]), ("plays", s["plays"]))]
        out.append({"teammate": nm, "short": fmt.last_name(nm), "p_avail": s["p_avail"],
                    "rows": rows,
                    "has_big": any(r["big"] is not None for r in rows)})
    return out


def _card(pid: int, pr: dict, name: str, team_codes: tuple[str, str], prows: list[dict],
          status: dict | None, splits: list[dict] | None = None) -> dict:
    st = pr.get("stats") or {}
    dist = P.player_dist(st)
    stats = []
    for key, label in STAT_ROWS:
        if key in ("min", "dk"):
            row = st.get((key, "played")) or {}
            mean, lo, hi = _num(row.get("mean")), _num(row.get("q10")), _num(row.get("q90"))
        else:
            mean = dist.means.get(key)
            rng = dist.range(key) if mean is not None else None
            lo, hi = (rng if rng else (None, None))
        stats.append({"key": key, "label": label, "mean": mean, "lo": lo, "hi": hi})
    dk_all = _num((st.get(("dk", "all")) or {}).get("mean"))
    team = pr.get("team") or ""
    home, away = team_codes
    opp = (f"vs {away}" if team == home else f"@ {home}") if team in (home, away) else ""
    p_play = _num(pr.get("p_play"))
    s_label = (status or {}).get("status")
    order = {s: i for i, s in enumerate(P.STATS)}
    pv = sorted((_prop_view(r) for r in prows), key=lambda x: order.get(x["stat"], 9))
    return {"player_id": pid, "name": name, "team": team, "opp": opp, "p_play": p_play,
            "status": s_label, "status_class": STATUS_CLASS.get(s_label or "", "unknown"),
            "reason": (status or {}).get("reason") or "", "stats": stats, "dk_all": dk_all,
            "by": {s["key"]: s["mean"] for s in stats}, "props": pv, "splits": splits or []}


def _sort_cards(cards: list[dict], sort: str) -> list[dict]:
    if sort == "name":
        return sorted(cards, key=lambda c: fmt.last_name(c["name"]).lower())
    if sort == "dk":
        return sorted(cards, key=lambda c: -(c["dk_all"] or 0.0))
    return sorted(cards, key=lambda c: (-(c["by"].get(sort) or 0.0), -(c["dk_all"] or 0.0)))


def _game_view(conn: sqlite3.Connection, res: dict, at: str, cache: dict) -> dict:
    g = res["game"]
    gid, home, away = g["game_id"], g.get("home") or "", g.get("away") or ""
    proj = res.get("proj") or {}
    run = res.get("run")
    names = cache.get(("names", run["run_id"], gid)) if run else None
    if names is None:
        names = P._player_names(conn, list(proj))
    by_pid: dict[int, list[dict]] = {}
    for r in res["rows"]:
        by_pid.setdefault(r["player_id"], []).append(r)
    team_of = {pid: (pr.get("team") or "") for pid, pr in proj.items()}
    if "injuries" not in cache:          # one report resolution per page build
        try:
            cache["injuries"] = injuries.current(conn, at)
        except Exception:  # noqa: BLE001 - a broken report table never breaks the page
            cache["injuries"] = {}
    status = _status_map(cache["injuries"], {home, away}, names, team_of)
    try:
        splits = SCN.run_splits(conn, run["run_id"]) if run else {}
    except Exception:  # noqa: BLE001 - a broken sims file never breaks the page
        splits = {}
    try:
        late = LN.flags_by_player(LN.run_flags(conn, run["run_id"])) if run else {}
    except Exception:  # noqa: BLE001 - an optional tag never breaks the page
        late = {}
    cards, listed = [], []
    for pid, pr in proj.items():
        prow = by_pid.get(pid, [])
        nm = (prow[0]["player_name"] if prow else None) or (names.get(pid) or [str(pid)])[0]
        c = _card(pid, pr, nm, (home, away), prow, status.get(pid),
                  _split_view(splits.get(pid) or [], names) if splits else None)
        f = late.get(pid)
        c.update(ln=f, ln_label=LN.label(f) if f else "")
        if c["status"] in ("Out", "Doubtful") and not prow:
            listed.append(c)
            continue
        if prow or (c["dk_all"] or 0.0) >= CARD_MIN_DK:
            cards.append(c)
    caps = sorted({r["captured_at"] for r in res["rows"]})
    return {
        "game_id": gid, "home": identity(home), "away": identity(away),
        "label": f"{away} @ {home}", "tip": g.get("tipoff_utc"),
        "status": g.get("status"),
        "run": None if run is None else {
            "kind": run.get("kind"), "created_at": run.get("created_at"),
            "model_version": run.get("model_version")},
        "market_at": caps[-1] if caps else None, "n_props": len(res["rows"]),
        "cards": cards, "listed": sorted(listed, key=lambda c: c["name"]),
    }


def build_page(conn: sqlite3.Connection | None, day: str, *, now: datetime,
               params: Mapping[str, str] | None = None, cache: dict | None = None) -> dict:
    """View model for /players/<day>. `cache` (optional) holds the expensive per-day part."""
    params = dict(params or {})
    sort = params.get("sort") if params.get("sort") in SORTS else "dk"
    want = params.get("game") or ""
    d = fmt.parse_day(day)
    base = {"day": day, "day_label": fmt.long_date(day), "sort": sort, "game": want,
            "disclaimer": DISCLAIMER, "games": [], "filters": [], "sorts": [],
            "prev_day": (d - timedelta(days=1)).isoformat() if d else None,
            "next_day": (d + timedelta(days=1)).isoformat() if d else None}
    if conn is None:
        return {**base, "state": "no_db"}
    at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache = cache if cache is not None else {}
    games = cache.get("games")
    if games is None:
        pcache: dict = {}
        res = P.compare_day(conn, day, cache=pcache)
        tips = [g["game"].get("tipoff_utc") for g in res["games"] if g["game"].get("tipoff_utc")]
        if tips:                         # a past day: the reports as they stood at the last tip
            at = min(at, max(tips))
        games = [_game_view(conn, g, at, pcache) for g in res["games"]]
        cache["games"] = games
        cache["summary"] = res["summary"]
    if not games:
        return {**base, "state": "no_games"}

    def url(**over) -> str:
        q = {"game": want, "sort": sort, **over}
        q = {k: v for k, v in q.items() if v and not (k == "sort" and v == "dk")}
        return f"/players/{day}" + (f"?{urlencode(q)}" if q else "")
    filters = [{"label": "All games", "url": url(game=""), "active": not want}]
    filters += [{"label": g["label"], "sub": fmt.tipoff(g["tip"]), "url": url(game=g["game_id"]),
                 "active": want == g["game_id"]} for g in games]
    sorts = [{"label": lab, "url": url(sort=k), "active": k == sort} for k, lab in SORTS.items()]
    shown = [dict(g, cards=_sort_cards(g["cards"], sort)) for g in games
             if not want or g["game_id"] == want]
    if want and not shown:
        shown = [dict(g, cards=_sort_cards(g["cards"], sort)) for g in games]
    s = cache.get("summary") or {}
    return {**base, "state": "ok", "games": shown, "filters": filters, "sorts": sorts,
            "n_games": len(games), "n_props": s.get("joined", 0),
            "n_cards": sum(len(g["cards"]) for g in shown),
            "has_splits": any(c["splits"] for g in shown for c in g["cards"]),
            "has_late": any(c.get("ln") for g in shown for c in g["cards"]),
            "ln_window": int(LN.LATE_NEWS_WINDOW.total_seconds() // 60),
            "big_k": f"{XF.BIG_K:g}", "split_min_n": SCN.MIN_N}


# ===================================================================== routes

def register(app, *, store, render: Callable, not_found: Callable, today: Callable[[], date],
             health_banner: Callable | None = None) -> None:
    """Add /players routes. Tests can set ``app.state.players_now`` to a callable returning an
    aware datetime."""
    def now() -> datetime:
        fn = getattr(app.state, "players_now", None)
        return fn() if fn else datetime.now(timezone.utc)

    @app.get("/players")
    def players_today():
        return RedirectResponse(f"/players/{today().isoformat()}", status_code=307)

    @app.get("/players/{day}", response_class=HTMLResponse)
    def players_day(day: str, request: Request):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "Player pages look like /players/2026-10-20.")
        conn = store.health_conn()
        try:
            if conn is None:
                vm = build_page(None, day, now=now())
            else:
                key = ("players",) + fingerprint(conn, day)
                cache = store.derived(key, dict)
                vm = build_page(conn, day, now=now(), params=dict(request.query_params),
                                cache=cache)
        finally:
            if conn is not None:
                conn.close()
        return render("players.html", v=vm, page="players",
                      health_banner=health_banner() if health_banner else None)
