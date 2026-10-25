"""Stat finder: game search and streak screens (docs/STATS_SITE.md, queries stats/finder.py).

Routes
  /finder                         saved screens + the game search form
  /finder?mode=games&...          player games matching every condition
      s1..s6 / o1..o6 / v1..v6    condition rows: stat (finder.COND_STATS), operator
                                  (ge le gt lt eq), value
      sal_min sal_max own_min own_max     DK salary ($) and real own% (0-100) ranges
      season=<S>|all  type=all|regular|playoffs|playin  ha=home|away  wl=w|l
      team=<ABBR>  opp=<ABBR>  role=starter|bench  rest=0|1|2 (2 = 2+)  player=<id>
      sort=date|<stat>  view=list|players (group by player)
  /finder?mode=streak&stat=pts&op=ge&thr=20&k=8&n=10&gp=10   (+ season / type / team)
      players reaching it in at least k of their last n games, with gp+ games in the scope

Every filter is a query parameter, so any search can be bookmarked. Values outside the
whitelists are dropped with a notice; results are capped at finder.CAP rows. Information only.
"""

from __future__ import annotations

from datetime import date
from typing import Callable
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse

from ..stats import finder as F
from ..stats import seasons as S
from .stats_common import open_db
from .teams import identity

TYPE_LABELS = {"all": "All games", "regular": "Regular season", "playoffs": "Playoffs",
               "playin": "Play-in"}
CTX_KEYS = ("ha", "wl", "role", "rest", "team", "opp")
SHOW_ROWS = 4                       # condition rows on the form (filled rows + blanks)

PRESETS = (
    ("20+ pts in 8 of last 10", "mode=streak&stat=pts&op=ge&thr=20&k=8&n=10"),
    ("Double-double in 4 of last 5", "mode=streak&stat=dd&op=ge&thr=2&k=4&n=5"),
    ("3+ threes in 5 straight", "mode=streak&stat=fg3m&op=ge&thr=3&k=5&n=5"),
    ("50+ DK pts at under $8K", "mode=games&s1=dk&o1=ge&v1=50&sal_max=7900&sort=dk"),
    ("40-point games", "mode=games&s1=pts&o1=ge&v1=40&sort=pts"),
    ("Triple-doubles", "mode=games&s1=dd&o1=ge&v1=3"),
    ("30+ on a back-to-back", "mode=games&s1=pts&o1=ge&v1=30&rest=0"),
    ("25+ off the bench", "mode=games&s1=pts&o1=ge&v1=25&role=bench&sort=pts"),
    ("15+ reb and 5+ ast", "mode=games&s1=reb&o1=ge&v1=15&s2=ast&o2=ge&v2=5"),
)


def _pick(qp, key: str, allowed) -> str | None:
    v = qp.get(key)
    return v if v in allowed else None


def _code(qp, key: str) -> str | None:
    v = (qp.get(key) or "").strip().upper()
    return v if v in F.CODES else None


def parse_context(qp, window: tuple[str, ...], notices: list[str]) -> tuple[F.Context, dict]:
    """(Context, the clean values for the form and the links)."""
    season = qp.get("season") or "all"
    if season != "all" and season not in window:
        notices.append(f"Season “{season[:12]}” is outside the 3-season window: using all three.")
        season = "all"
    stype = qp.get("type") or "all"
    if stype not in F.TYPES:
        stype = "all"
    vals = {"season": season, "type": stype,
            "ha": _pick(qp, "ha", F.CONTEXT["ha"]), "wl": _pick(qp, "wl", F.CONTEXT["wl"]),
            "role": _pick(qp, "role", F.CONTEXT["role"]),
            "rest": _pick(qp, "rest", F.CONTEXT["rest"]),
            "team": _code(qp, "team"), "opp": _code(qp, "opp"), "player": None}
    for k in CTX_KEYS:
        raw = qp.get(k)
        if raw and vals[k] is None:
            notices.append(f"Filter {k}=“{raw[:12]}” is not one of the choices: ignored.")
    raw_pid = (qp.get("player") or "").strip()
    if raw_pid:
        if raw_pid.isdigit() and len(raw_pid) <= 10:
            vals["player"] = int(raw_pid)
        else:
            notices.append("Player filter ignored: not a player id.")
    ctx = F.Context(seasons=window if season == "all" else (season,),
                    stype=F.TYPES[stype], ha=vals["ha"], wl=vals["wl"], role=vals["role"],
                    rest=vals["rest"], team=vals["team"], opp=vals["opp"],
                    player=vals["player"])
    return ctx, vals


def _qs(params: dict) -> str:
    q = {k: v for k, v in params.items() if v not in (None, "")}
    return "/finder" + ("?" + urlencode(q) if q else "")


def _fmt_val(v: float) -> str:
    return f"{v:g}"


def games_view(conn, qp, window, notices) -> dict:
    ctx, vals = parse_context(qp, window, notices)
    conds, errs = F.parse_conds(qp)
    notices += errs
    sort = qp.get("sort") if qp.get("sort") in F.SORTS else "date"
    view = "players" if qp.get("view") == "players" else "list"
    # the clean parameter set (links, the form's state)
    stat_rows = [c for c in conds if c.stat not in ("salary", "own")]
    rng = {k: qp.get(k) for k in ("sal_min", "sal_max", "own_min", "own_max")}
    rng = {k: (_fmt_val(F.number(v)) if F.number(v) is not None else "") for k, v in rng.items()}
    base: dict = {"mode": "games"}
    for i, c in enumerate(stat_rows, 1):
        base.update({f"s{i}": c.stat, f"o{i}": c.op, f"v{i}": _fmt_val(c.value)})
    base.update({k: v for k, v in rng.items() if v})
    base.update({k: vals[k] for k in ("season", "type") if vals[k] != "all"})
    base.update({k: vals[k] for k in CTX_KEYS if vals[k]})
    if vals["player"] is not None:
        base["player"] = vals["player"]
    if sort != "date":
        base["sort"] = sort
    rows_form = [{"stat": c.stat, "op": c.op, "value": _fmt_val(c.value)} for c in stat_rows]
    want = min(F.MAX_CONDS, max(SHOW_ROWS, len(rows_form) + 1))
    rows_form += [{"stat": "", "op": "ge", "value": ""}] * (want - len(rows_form))
    active = bool(conds) or any(vals[k] for k in CTX_KEYS) or vals["player"] is not None
    res, timed_out, player_name = None, False, None
    if active and conn is not None:
        try:
            res = F.game_search(conn, conds, ctx, window, sort=sort, group=view == "players")
        except F.QueryTimeout:
            timed_out = True
        except F.FinderError as e:
            notices.append(f"Search not run: {e}.")
        if vals["player"] is not None:
            r = conn.execute("SELECT player_name FROM player_games WHERE player_id=? "
                             "ORDER BY game_date DESC LIMIT 1", (vals["player"],)).fetchone()
            player_name = r[0] if r else f"#{vals['player']}"
    summary = [c.label() for c in conds]
    for k, lab in (("ha", {"home": "Home", "away": "Away"}), ("wl", {"w": "Wins", "l": "Losses"}),
                   ("role", {"starter": "Started", "bench": "Off the bench"}),
                   ("rest", {k2: v[1] for k2, v in F.REST.items()})):
        if vals[k]:
            summary.append(lab[vals[k]])
    if vals["team"]:
        summary.append(f"For {vals['team']}")
    if vals["opp"]:
        summary.append(f"vs {vals['opp']}")
    if player_name:
        summary.append(player_name)
    return {"vals": vals, "rows_form": rows_form, "rng": rng, "sort": sort, "view": view,
            "res": res, "timed_out": timed_out, "active": active, "summary": summary,
            "view_opts": [{"value": "list", "label": "Games", "href": _qs(base)},
                          {"value": "players", "label": "Group by player",
                           "href": _qs({**base, "view": "players"})}],
            "player_href": lambda pid: _qs({**{k: v for k, v in base.items()
                                              if k != "view"}, "player": pid}),
            "clear_player": _qs({k: v for k, v in base.items() if k != "player"})
            if vals["player"] is not None else None,
            "base_qs": _qs({**base, **({"view": view} if view != "list" else {})})}


def streak_view(conn, qp, window, notices) -> dict:
    ctx, vals = parse_context(qp, window, notices)
    stat = qp.get("stat") or "pts"
    form = {"stat": stat, "op": qp.get("op") or "ge", "thr": qp.get("thr") or "20",
            "k": qp.get("k") or "8", "n": qp.get("n") or "10", "gp": qp.get("gp") or ""}
    sc, res, timed_out = None, None, False
    try:
        sc = F.Screen.make(form["stat"], form["op"], form["thr"], form["k"], form["n"],
                           form["gp"] or None)
    except F.FinderError as e:
        notices.append(f"Screen not run: {e}.")
    if sc is not None and conn is not None:
        try:
            res = F.streaks(conn, sc, ctx, window)
        except F.QueryTimeout:
            timed_out = True
    if sc is not None:
        form.update(stat=sc.stat, op=sc.op, thr=_fmt_val(sc.thr), k=str(sc.k), n=str(sc.n),
                    gp=str(sc.min_gp) if sc.min_gp != sc.n else "")
    return {"vals": vals, "form": form, "screen": sc, "res": res, "timed_out": timed_out,
            "kind": F.STATS[sc.stat][2] if sc else "int"}


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/finder", response_class=HTMLResponse)
    def finder(request: Request):
        window = S.seasons(today())
        qp = request.query_params
        mode = "streak" if qp.get("mode") == "streak" else "games"
        notices: list[str] = []
        with open_db(store) as conn:
            ctx = (streak_view if mode == "streak" else games_view)(conn, qp, window, notices)
        here = str(request.url.query)
        presets = [{"label": lab, "href": f"/finder?{q}", "on": here == q}
                   for lab, q in PRESETS]
        teams = sorted(F.CODES)
        return render(
            "stats_finder.html", page="tools", tool="finder", mode=mode, window=window,
            notices=notices, presets=presets, teams=teams, team_name=lambda c: identity(c)["name"],
            stats=F.STATS, cond_stats=F.COND_STATS, streak_stats=F.STREAK_STATS,
            stat_tips=F.STAT_TIPS, ops=F.OPS, sorts=F.SORTS, types=TYPE_LABELS, rest=F.REST,
            cap=F.CAP,
            mode_opts=[{"value": "games", "label": "Game finder", "href": "/finder?mode=games"},
                       {"value": "streak", "label": "Streaks", "href": "/finder?mode=streak"}],
            **ctx)
