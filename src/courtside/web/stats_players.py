"""Players: the player directory and the player page (docs/STATS_SITE.md, mockup #s-player).

Routes
  /players                   directory: ?q= (also the no-JS fallback of the top-bar search),
                             ?team=, ?season=, ?sort=, ?page=
  /player/<int:player_id>    player page; unknown ids -> 404 page (non-numeric ids never match)
                             ?season= & ?type=regular|playoffs   tiles, shot chart, play types
                             ?log=<season>|all & ?logtype=regular|playoffs|all   game log
                             ?vs=<TEAM>                          versus a team
                             research tools (stats/player_tools.py):
                             ?stat=pts|reb|ast|fg3m|pra|dk|min|stocks & ?win=5|10|20|season
                             & ?line=<number>                    recent-games chart
                             filters (chart + game log): ?ha=home|away ?opp=<TEAM>
                             ?role=start|bench ?rest=0|1|2 ?wl=W|L ?last=<n> ?dt=top|mid|bottom
                             ?tm=<teammate id>&tmode=with|without

Queries live in `courtside.stats.players`; everything here is view-model glue. All switches
are plain links / GET forms, so the page is correct without JS (stats.js adds sorting + tips).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse

from ..stats import garbage as G
from ..stats import percentiles as PC
from ..stats import player_tools as T
from ..stats import players as P
from ..stats import search as SR
from ..stats import seasons as S
from .stats_common import open_db

DIR_SORTS = [("pts", "Pts"), ("reb", "Reb"), ("ast", "Ast"), ("dk", "DK"), ("min", "Min"),
             ("fg3m", "3PM"), ("ts", "TS%"), ("stl", "Stl"), ("blk", "Blk")]
TILES = [("Points", "pts"), ("Rebounds", "reb"), ("Assists", "ast"), ("DK pts", "dk"),
         ("TS%", "ts"), ("Minutes", "min"), ("Steals", "stl"), ("Blocks", "blk"),
         ("3PM", "fg3m")]
PROJ_STATS = [("Minutes", "min"), ("Points", "pts"), ("Rebounds", "reb"), ("Assists", "ast"),
              ("3PM", "fg3m")]


def player_head(conn, player_id: int, today: date | None = None) -> dict | None:
    """Header of the player page (None = unknown id)."""
    if conn is None:
        return None
    return P.header(conn, player_id, today=today or date.today())


def _href(path: str, params: dict) -> str:
    q = {k: v for k, v in params.items() if v not in (None, "")}
    return path + ("?" + urlencode(q) if q else "")


# ------------------------------------------------------------------ /players
def directory_view(conn, request: Request, today: date) -> dict:
    window = S.seasons(today)
    qp = request.query_params
    season = S.pick(qp.get("season"), allowed=window)
    q = (qp.get("q") or "").strip()[:SR.MAX_Q]
    team = (qp.get("team") or "").strip().upper()[:3]
    sort = qp.get("sort") if qp.get("sort") in P.SORTS else "pts"
    try:
        page = int(qp.get("page") or 1)
    except ValueError:
        page = 1
    d = P.directory(conn, season, q=q, team=team, sort=sort, page=page) if conn else \
        {"rows": [], "total": 0, "page": 1, "pages": 1, "teams": [], "sort": sort,
         "n_qualified": 0, "min_gp": 2, "offset": 0}
    base = {"q": q, "team": team, "season": season if season != window[0] else None,
            "sort": sort if sort != "pts" else None}

    def url(**over) -> str:
        return _href("/players", {**base, **over})
    others = []
    if q and conn is not None:
        shown = {r["player_id"] for r in d["rows"]}
        others = [h for h in SR.search(conn, q, limit=12, today=today)
                  if not (h["type"] == "player" and h["id"] in shown)][:8]
    return {"season": season, "seasons": window, "q": q, "team_code": team, "sort": sort,
            "dir": d, "others": others,
            "season_opts": [{"value": s, "label": s, "href": url(season=s, page=None)}
                            for s in window],
            "sort_opts": [{"value": k, "label": lab, "href": url(sort=k, page=None)}
                          for k, lab in DIR_SORTS],
            "prev_href": url(page=d["page"] - 1) if d["page"] > 1 else None,
            "next_href": url(page=d["page"] + 1) if d["page"] < d["pages"] else None}


# ------------------------------------------------------------------ /player/<id>
def _season_default(lines: list[dict], window: tuple[str, ...], want: str | None) -> str:
    if want in window:
        return want
    have = {ln["season"] for ln in lines}
    return next((s for s in window if s in have), window[0])


def _proj_view(card: dict | None) -> dict | None:
    if not card:
        return None
    pr = card["proj"]
    st = pr.get("stats") or {}
    act = card.get("actual") or {}
    dk = st.get("dk") or {}
    rows = []
    for label, k in PROJ_STATS:
        m = (st.get(k) or {}).get("mean")
        a = act.get(k)
        rows.append({"label": label, "proj": m, "actual": a,
                     "diff": (a - m) if (a is not None and m is not None) else None,
                     "lo": (st.get(k) or {}).get("q10"), "hi": (st.get(k) or {}).get("q90")})
    grade = pr.get("grade") or {}
    salary = grade.get("salary")
    return {"mode": card["mode"], "game": card["game"], "run_kind": pr.get("kind"),
            "run_at": pr.get("run_at"), "dk": dk.get("mean"), "lo": dk.get("q10"),
            "hi": dk.get("q90"), "actual_dk": act.get("dk"), "rows": rows,
            "p_play": pr.get("p_play"), "salary": salary,
            "in80": grade.get("in80")}


def player_view(conn, pid: int, head: dict, request: Request, today: date,
                now: datetime) -> dict:
    window = S.seasons(today)
    qp = request.query_params
    lines = P.season_lines(conn, pid, window)
    season = _season_default(lines, window, qp.get("season"))
    stype = S.season_type(qp.get("type"))
    log_season = qp.get("log") if qp.get("log") in window else None
    lt_raw = qp.get("logtype")
    log_type = S.SEASON_TYPES.get(lt_raw) if lt_raw in S.SEASON_TYPES else None
    gt = qp.get("gt") == "0"                 # exclude garbage time (stint-based numbers only)
    params = {"gt": "0" if gt else None,
              "season": qp.get("season") if qp.get("season") in window else None,
              "type": "playoffs" if stype == S.PLAYOFFS else None,
              "log": log_season, "logtype": lt_raw if log_type else None,
              "vs": None}
    base = f"/player/{pid}"

    opps = P.opponents(conn, pid, window)
    opp_codes = {o["team"] for o in opps}
    games_all = T.games(conn, pid, window)
    mates = T.teammates(conn, pid, window)
    filt = T.parse_filters(qp, opps=opp_codes, mates={m["player_id"] for m in mates})
    stat = qp.get("stat") if qp.get("stat") in T.STATS else "pts"
    win = qp.get("win") if qp.get("win") in dict(T.WINDOWS) else T.DEFAULT_WIN
    line = _num(qp.get("line"))
    params.update({k: filt.get(k) for k in T.FILTER_KEYS})
    params.update(stat=stat if stat != "pts" else None,
                  win=win if win != T.DEFAULT_WIN else None,
                  line=f"{line:g}" if line is not None else None)
    vs = (qp.get("vs") or "").strip().upper()[:3]
    if not vs:
        nxt = head.get("next")
        if nxt and nxt.get("opp"):
            vs = nxt["opp"]
        else:
            last = next((r for r in P.game_log(conn, pid, window) if r.get("opp")), None)
            vs = last["opp"] if last else ""
    else:
        params["vs"] = vs

    def url(anchor: str = "", **over) -> str:
        return _href(base, {**params, **over}) + (f"#{anchor}" if anchor else "")

    cur = next((ln for ln in lines if ln["season"] == season and ln["season_type"] == stype),
               None)
    prev_s = S.prev_season(season)
    prev = next((ln for ln in lines if ln["season"] == prev_s
                 and ln["season_type"] == S.REGULAR), None)
    for ln in lines:
        ln["pcts"] = PC.player_pcts(conn, ln["season"], ln["season_type"]).get(pid, {})
    tiles = []
    for label, k in TILES:
        v = cur.get(k) if cur else None
        tiles.append({"label": label, "key": k, "value": v,
                      "rank": (cur or {}).get("ranks", {}).get(k),
                      "pct": (cur or {}).get("pcts", {}).get(k),
                      "n": (cur or {}).get("n") or 30,
                      "prev": prev.get(k) if prev else None})

    log = P.game_log(conn, pid, window, season=log_season, season_type=log_type)
    research = _research(conn, pid, head, games_all, mates, filt, stat, win, line, season,
                         stype, window, url)
    if filt:
        keep = {g["game_id"] for g in research["filtered"]}
        log = [r for r in log if r["game_id"] in keep]
    teams_vs = sorted(opp_codes | ({vs} if vs else set()))
    shots = P.shot_chart(conn, pid, season, stype)
    return {
        "season": season, "stype": stype, "type_label": S.type_label(stype), "window": window,
        "season_opts": [{"value": s, "label": s, "href": url(season=s)} for s in window],
        "type_opts": [{"value": S.REGULAR, "label": "Regular season",
                       "href": url(type=None)},
                      {"value": S.PLAYOFFS, "label": "Playoffs", "href": url(type="playoffs")}],
        "line": cur, "prev_line": prev, "prev_season": prev_s, "tiles": tiles, "lines": lines,
        "log": log, "log_season": log_season or "all", "log_type": log_type or "all",
        "log_opts": [{"value": "all", "label": "Last 3 seasons", "href": url("log", log=None)}]
                    + [{"value": s, "label": s, "href": url("log", log=s)} for s in window],
        "logtype_opts": [{"value": "all", "label": "All", "href": url("log", logtype=None)},
                         {"value": S.REGULAR, "label": "Regular",
                          "href": url("log", logtype="regular")},
                         {"value": S.PLAYOFFS, "label": "Playoffs",
                          "href": url("log", logtype="playoffs")}],
        "proj": _proj_view(P.projection_card(conn, pid, head.get("next"), now)),
        "vs": P.versus(conn, pid, vs, window) if vs else None, "vs_team": vs,
        "vs_teams": teams_vs, "vs_counts": {o["team"]: o["games"] for o in opps},
        "vs_hidden": {k: v for k, v in params.items() if k != "vs" and v},
        "shots": shots, "court": P.court_svg_paths(), "court_w": P.COURT_W,
        "court_h": P.COURT_H, "ramp": P.FG_RAMP,
        "ptypes": P.play_types(conn, pid, season, stype),
        "own": P.ownership(conn, pid),
        "onoff": G.player_onoff(conn, pid, season, stype, exclude=gt), "gt": gt,
        "gt_def": G.DEFINITION,
        "gt_opts": [{"value": "all", "label": "All minutes", "href": url("onoff", gt=None)},
                    {"value": "x", "label": "Excl. garbage time", "href": url("onoff", gt="0")}],
        "rs": research, "pv": {"opportunity": _opportunity(conn, pid, season, stype),
                               "similar": _similar(conn, pid, season, today)},
    }


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x * 2) / 2 if 0 <= x <= 300 else None


def _opportunity(conn, pid: int, season: str, stype: str):
    """The "Chances" panel (web/stats_chances.panel over stats/opportunity.player_opportunity,
    per season type); None = nothing collected (the panel's empty state)."""
    try:
        from ..stats import opportunity as OPP
        from . import stats_chances as CH
    except ImportError:
        return None
    fn = getattr(OPP, "player_opportunity", None)
    if fn is None:
        return None
    try:
        return CH.panel(conn, pid, season, stype, fn(conn, pid, season, stype))
    except Exception:                                   # noqa: BLE001 - never break the page
        return None


def _similar(conn, pid: int, season: str, today: date):
    """The "Similar players" card (stats/similar.py; regular season of the selected season).
    Early in the current season (too few minutes yet) it shows last season, flagged."""
    try:
        from ..stats import similar as SIM
        res = SIM.similar(conn, pid, season, today)
        if res["status"] != "ok" and season == S.current_season(today):
            prev = SIM.similar(conn, pid, S.prev_season(season), today)
            if prev["status"] == "ok":
                return {**prev, "fallback_from": season}
        return res
    except Exception:                                   # noqa: BLE001 - never break the page
        return None


def _research(conn, pid, head, games_all, mates, filt, stat, win, line, season, stype,
              window, url) -> dict:
    """Bar chart, filter chips, splits, minutes trend and clutch (stats/player_tools.py)."""
    filtered = T.apply_filters(conn, pid, games_all, filt, window) if filt else games_all
    season_games = [g for g in games_all if g["season"] == season and g["season_type"] == stype]
    season_avg = T.averages(season_games)[stat] if stat in T.SPLIT_KEYS and season_games \
        else None
    nxt = head.get("next") or {}
    vs_opp = filt.get("opp") or nxt.get("opp")
    chart = T.bar_chart(filtered, stat, win, line, season=season, stype=stype,
                        season_avg=season_avg, vs_opp=vs_opp)
    mate_by = {m["player_id"]: m for m in mates}
    in_season = [m for m in mates if m["by_season"].get(season)]
    mate = mate_by.get(filt.get("tm")) or (
        max(in_season, key=lambda m: m["min_by_season"][season]) if in_season else None)
    pos = next((g["pos"] for g in games_all if g.get("pos")), None)

    def chip(key, value, label, **extra):
        on = filt.get(key) == value and all(filt.get(k) == v for k, v in extra.items())
        over = {key: None} if on else {key: value, **extra}
        if key == "tm" and on:
            over["tmode"] = None
        return {"label": label, "on": on, "href": url("filters", line=None, **over)}
    groups = [{"label": "Venue", "chips": [chip("ha", "home", "Home"),
                                           chip("ha", "away", "Away")]}]
    if any(g.get("started") is not None for g in games_all):
        groups.append({"label": "Role", "chips": [chip("role", "start", "Starter"),
                                                  chip("role", "bench", "Bench")]})
    groups.append({"label": "Rest", "chips": [chip("rest", "0", "Back-to-back"),
                                              chip("rest", "1", "1 day"),
                                              chip("rest", "2", "2+ days")]})
    groups.append({"label": "Result", "chips": [chip("wl", "W", "Wins"),
                                                chip("wl", "L", "Losses")]})
    if any(g.get("tier") for g in games_all):
        groups.append({"label": f"Opp D vs {pos}" if pos else "Opp D",
                       "chips": [chip("dt", "top", "Top 10"), chip("dt", "mid", "Middle"),
                                 chip("dt", "bottom", "Bottom 10")]})
    groups.append({"label": "Last", "chips": [chip("last", n, str(n)) for n in (5, 10, 20)]})
    labels = {"ha": lambda v: v.capitalize(), "opp": lambda v: f"vs {v}",
              "role": lambda v: "Starter" if v == "start" else "Bench",
              "rest": lambda v: T.REST[v], "wl": lambda v: "Wins" if v == "W" else "Losses",
              "last": lambda v: f"Last {v}",
              "dt": lambda v: T.TIERS[v] + (f" vs {pos}" if pos else ""),
              "tm": lambda v: f"{filt.get('tmode', 'with').capitalize()} "
                              f"{mate_by[v]['name']}"}
    active = [{"label": labels[k](filt[k]),
               "href": url("filters", line=None, **({k: None, "tmode": None} if k == "tm"
                                                     else {k: None}))}
              for k in T.FILTER_KEYS if k in filt and k in labels]
    clear = url("filters", line=None, **{k: None for k in T.FILTER_KEYS})
    hidden = {k: v for k, v in _params_of(url).items()}
    split_rows = T.splits(conn, pid, season_games, window=window, mate=mate,
                          opp=filt.get("opp") or nxt.get("opp"))
    for r in split_rows:
        if r["key"]:
            over = {r["key"]: r["value"]}
            if r["key"] == "tm":
                over["tmode"] = r["tmode"]
            r["href"] = url("filters", line=None, **over)
    return {
        "filtered": filtered, "filters": filt, "n_filtered": len(filtered),
        "n_all": len(games_all), "chart": chart, "chart_json": T.json_script(chart["data"]),
        "stat_opts": [{"value": k, "label": lab,
                       "href": url("trend", stat=k if k != "pts" else None, line=None)}
                      for k, (lab, _) in T.STATS.items()],
        "win_opts": [{"value": k, "label": lab,
                      "href": url("trend", win=k if k != T.DEFAULT_WIN else None)}
                     for k, lab in T.WINDOWS],
        "groups": groups, "active": active, "clear": clear, "hidden": hidden,
        "mates": mates[:40], "mate": mate, "opps": sorted({g["opp"] for g in games_all
                                                            if g.get("opp")}),
        "splits": split_rows, "season_games": len(season_games), "pos": pos,
        "minutes": T.minutes_trend(games_all, season, stype),
        "clutch": T.clutch(conn, pid, season, stype, window),
        "vs_opp": vs_opp,
    }


def _params_of(url) -> dict:
    """The current query params (as the url() helper would write them), for GET forms."""
    from urllib.parse import parse_qsl, urlsplit
    return dict(parse_qsl(urlsplit(url()).query))


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/players", response_class=HTMLResponse)
    def players(request: Request):
        with open_db(store) as conn:
            v = directory_view(conn, request, today())
        return render("stats_players.html", page="players", **v)

    @app.get("/player/{player_id:int}", response_class=HTMLResponse)
    def player(player_id: int, request: Request):
        t = today()
        now = datetime.now(timezone.utc)
        with open_db(store) as conn:
            head = P.header(conn, player_id, today=t, now=now) if conn is not None else None
            if head is None:
                return not_found("Player not found", f"No player with id {player_id}.")
            v = player_view(conn, player_id, head, request, t, now)
        return render("stats_player.html", page="player", p=head, **v)
