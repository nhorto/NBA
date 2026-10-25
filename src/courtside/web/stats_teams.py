"""Teams: the league table and the team page (docs/STATS_SITE.md).

Routes
  /teams              league table, all 30 teams (?season= ?type=regular|playoffs
                      ?view=league|conf|div: league table / East-West standings / divisions)
  /team/<ABBR>        team page (?season= ?type=); lower-case codes redirect to upper case,
                      unknown codes -> 404 page

Queries: courtside.stats.teams (league table, roster, DvP, on/off, lineups) and
courtside.stats.leaders (the cached season pass behind roster and DvP).
"""

from __future__ import annotations

from datetime import date
from typing import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import teams as T
from ..stats import garbage as G
from ..stats import percentiles as PC
from ..stats import seasons as S
from ..stats import teams as ST
from .stats_common import open_db, season_args, season_type_arg
from .teams import identity

CODES = frozenset(t.tricode for t in T.TEAMS)

# league-table columns: (key, header, kind, nd, tooltip)
LEAGUE_COLS = (
    ("pts", "Pts", "num", 1, "Points per game"),
    ("opp_pts", "Opp", "num", 1, "Opponent points per game (lower is better)"),
    ("ortg", "ORtg", "num", 1, "Points per 100 possessions"),
    ("drtg", "DRtg", "num", 1, "Opponent points per 100 possessions (lower is better)"),
    ("net", "Net", "signed", 1, "ORtg minus DRtg"),
    ("pace", "Pace", "num", 1, "Possessions per 48 minutes (1st = fastest)"),
    ("efg", "eFG%", "pct", 1, "Effective field-goal %"),
    ("ts", "TS%", "pct", 1, "True shooting %"),
    ("tov", "TOV%", "pct", 1, "Turnovers per 100 plays (lower is better)"),
    ("oreb", "OREB%", "pct", 1, "Share of own misses rebounded"),
    ("fg3a", "3PA", "num", 1, "3-point attempts per game"),
    ("ast", "Ast", "num", 1, "Assists per game"),
    ("reb", "Reb", "num", 1, "Rebounds per game"),
)
TILES = (("Off rating", "ortg", "num"), ("Def rating", "drtg", "num"),
         ("Net rating", "net", "signed"), ("Pace", "pace", "num"), ("eFG%", "efg", "pct"),
         ("TOV%", "tov", "pct"), ("OREB%", "oreb", "pct"), ("Pts / game", "pts", "num"),
         ("Opp pts", "opp_pts", "num"))
BUCKET_LABELS = {"PG": "Point guards", "SG": "Shooting guards", "SF": "Small forwards",
                 "PF": "Power forwards", "C": "Centers", "G": "Guards", "F": "Forwards"}
TYPE_OPTS = (("regular", "Regular season"), ("playoffs", "Playoffs"))
VIEWS = (("league", "League table"), ("conf", "Standings"), ("div", "Divisions"))


def _type_value(season_type: str) -> str:
    return "playoffs" if season_type == S.PLAYOFFS else "regular"


def _extra(season_type: str) -> str:
    return "&type=playoffs" if season_type == S.PLAYOFFS else ""


def _type_seg(base: str, season: str, season_type: str) -> list[dict]:
    return [{"value": v, "label": lab,
             "href": f"{base}?season={season}" + ("&type=playoffs" if v == "playoffs" else "")}
            for v, lab in TYPE_OPTS]


def _dvp_view(d: dict | None) -> dict | None:
    if not d:
        return None
    row, n = d["row"], d["n"]
    items = []
    for b in d["buckets"]:
        rk = row["rk"].get(b)
        if rk is None:
            word = ""
        elif rk <= n / 3:
            word = "gives up a lot"
        elif rk > 2 * n / 3:
            word = "tough matchup"
        else:
            word = "average"
        items.append({"bucket": b, "label": BUCKET_LABELS.get(b, b), "v": row["all"].get(b),
                      "rk": rk, "v10": row["l10"].get(b), "rk10": row["rk10"].get(b),
                      "word": word})
    basis = ("DK roster positions (first listed, from this season's DK salaries)"
             if d["basis"] == "dk" else
             "NBA player-index positions G / F / C (no DK salaries for this season)")
    return {"rows": items, "n": n, "gp": row["gp"], "basis": basis, "kind": d["basis"],
            "coverage": d["coverage"], "tot": row["all"].get("tot"),
            "tot_rk": row["rk"].get("tot")}


def _gt_arg(request: Request) -> bool:
    """`?gt=0` excludes garbage time from the stint-based numbers (default: include)."""
    return request.query_params.get("gt") == "0"


def _tiles(row: dict | None, n: int, pcts: dict, gtr: dict | None, gt_n: int,
           gt: bool) -> list[dict]:
    """Rating tiles with rank + percentile. With gt, ORtg / DRtg / Net / Pace come from the
    stints without garbage time (stats/garbage.py), ranked among the teams with stints."""
    out = []
    for lab, key, kind in TILES:
        tl = {"label": lab, "key": key, "kind": kind, "n": n, "gt": False,
              "value": row[key] if row else None, "rank": row["rk"][key] if row else None,
              "pct": pcts.get(key)}
        if gt and key in G.RATING_KEYS:
            tl.update(gt=True, n=gt_n or n, value=gtr[key] if gtr else None,
                      rank=gtr["rk"][key] if gtr else None,
                      pct=gtr["pct"][key] if gtr else None)
        out.append(tl)
    return out


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/teams", response_class=HTMLResponse)
    def teams(request: Request):
        season, window = season_args(request, today())
        stype = season_type_arg(request)
        view = request.query_params.get("view")
        view = view if view in dict(VIEWS) else "league"
        vx = "" if view == "league" else f"&view={view}"
        lg = stand = None
        pcts: dict = {}
        with open_db(store) as conn:
            if conn is not None:
                lg = ST.league(conn, season, stype)
                pcts = PC.team_pcts(conn, season, stype) if view == "league" else {}
                if view != "league":
                    stand = ST.standings(conn, season, stype)
        rows = lg["rows"] if lg else []
        tx = _extra(stype)
        return render("stats_teams.html", page="teams", season=season, seasons=window,
                      stype=stype, type_value=_type_value(stype), extra=tx + vx,
                      type_opts=[{**o, "href": o["href"] + vx}
                                 for o in _type_seg("/teams", season, stype)],
                      view=view, standings=stand,
                      view_opts=[{"value": v, "label": lab,
                                  "href": f"/teams?season={season}{tx}"
                                          + ("" if v == "league" else f"&view={v}")}
                                 for v, lab in VIEWS],
                      type_label=S.type_label(stype), rows=rows, n=len(rows) or 30,
                      cols=LEAGUE_COLS, max_gp=lg["max_gp"] if lg else 0, pcts=pcts)

    @app.get("/team/{abbr}", response_class=HTMLResponse)
    def team(abbr: str, request: Request):
        code = abbr.upper()
        if code not in CODES:
            return not_found("Team not found", f"There is no NBA team with the code {abbr}.")
        if code != abbr:
            q = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(f"/team/{code}{q}", status_code=301)
        season, window = season_args(request, today())
        stype = season_type_arg(request)
        gt = _gt_arg(request)
        ctx = {"row": None, "n": 30, "roster": [], "dvp": None, "stints": None,
               "results": [], "upcoming": [], "coach": None, "place": None,
               "tile_vals": [], "roster_pcts": {}, "gt_info": None}
        with open_db(store) as conn:
            if conn is not None:
                lg = ST.league(conn, season, stype)
                ctx["n"] = lg["n"] or 30
                ctx["row"] = lg["by_team"].get(code)
                gtr = G.team_ratings(conn, season, stype, exclude=True) if gt else None
                ctx["tile_vals"] = _tiles(ctx["row"], ctx["n"],
                                          PC.team_pcts(conn, season, stype).get(code, {}),
                                          gtr["by_team"].get(code) if gtr else None,
                                          gtr["n"] if gtr else 0, gt)
                if gtr:
                    mine = gtr["by_team"].get(code) or {}
                    ctx["gt_info"] = {"games": gtr["games"], "share": mine.get("gt_share"),
                                      "n": gtr["n"]}
                ctx["coach"] = ST.coach(conn, code, season)
                ctx["place"] = ST.standings(conn, season, stype)["rank"].get(code)
                if ctx["row"]:
                    ctx["roster"] = ST.roster(conn, code, season, stype)
                    ctx["dvp"] = _dvp_view(ST.dvp(conn, code, season, stype))
                    ctx["stints"] = G.team_stints(conn, code, season, stype, exclude=gt)
                    ctx["roster_pcts"] = PC.player_pcts(conn, season, stype)
                    ctx["results"] = ST.results(conn, code, season, stype, n=10)
                if season == window[0]:
                    ctx["upcoming"] = ST.upcoming(conn, code, today().isoformat())
        gx = "&gt=0" if gt else ""
        base = f"/team/{code}?season={season}{_extra(stype)}"
        return render("stats_team.html", page="team", t=identity(code), season=season,
                      seasons=window, stype=stype, type_value=_type_value(stype),
                      extra=_extra(stype) + gx,
                      type_opts=[{**o, "href": o["href"] + gx}
                                 for o in _type_seg(f"/team/{code}", season, stype)],
                      type_label=S.type_label(stype), tiles=TILES, gt=gt,
                      gt_opts=[{"value": "all", "label": "All minutes", "href": base},
                               {"value": "x", "label": "Excl. garbage time",
                                "href": base + "&gt=0"}],
                      gt_def=G.DEFINITION, **ctx)
