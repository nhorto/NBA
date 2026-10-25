"""Leaders: league leaders (stat / season / per-game / last-N filters) and team leaders.

Routes
  /leaders     ?season= ?type=regular|playoffs ?stat=pts|reb|ast|stl|blk|fg3m|dk|ts|usg|min|dd
               ?mode=pg|tot|p36 ?last=all|10|5
Server-rendered, every filter is a link (works without JS). Queries: courtside.stats.leaders
(qualifier rules in its docstring) and courtside.stats.teams.team_leaders.
"""

from __future__ import annotations

from datetime import date
from typing import Callable
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse

from ..stats import leaders as L
from ..stats import seasons as S
from ..stats import teams as ST
from . import stats_chances as CH
from .stats_common import open_db, season_args, season_type_arg

LAST = {"all": ("all", "All games"), "10": ("l10", "Last 10"), "5": ("l5", "Last 5")}
STAT_ORDER = ("pts", "reb", "ast", "stl", "blk", "fg3m", "dk", "ts", "usg", "min", "dd")
CHIP_LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "stl": "Steals",
               "blk": "Blocks", "fg3m": "3PM", "dk": "DK pts", "ts": "TS%", "usg": "Usage",
               "min": "Minutes", "dd": "Double-doubles"}
SIDE_COLS = (("pts", "Pts"), ("reb", "Reb"), ("ast", "Ast"), ("dk", "DK"))


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/leaders", response_class=HTMLResponse)
    def leaders(request: Request):
        season, window = season_args(request, today())
        stype = season_type_arg(request)
        qp = request.query_params
        stat = qp.get("stat") if qp.get("stat") in L.STATS else "pts"
        mode = qp.get("mode") if qp.get("mode") in L.MODES else "pg"
        last = qp.get("last") if qp.get("last") in LAST else "all"
        cur = {"season": season, "type": "playoffs" if stype == S.PLAYOFFS else "regular",
               "stat": stat, "mode": mode, "last": last}
        defaults = {"season": window[0], "type": "regular", "stat": "pts", "mode": "pg",
                    "last": "all"}

        def href(**over) -> str:
            p = {**cur, **over}
            q = {k: v for k, v in p.items() if v != defaults[k] or (k == "season" and "season" in over)}
            return "/leaders" + (f"?{urlencode(q)}" if q else "")

        board, tleaders, chances = None, [], None
        with open_db(store) as conn:
            if conn is not None:
                board = L.leaders(conn, season, stype, stat, mode, LAST[last][0])
                tleaders = ST.team_leaders(conn, season, stype)
                chances = CH.league_chances(conn, season, stype)   # running hot / cold
        label, short, kind, rate = L.STATS[stat]
        mode_label = "" if rate else L.MODES[mode].lower()
        return render(
            "stats_leaders.html", page="leaders", season=season, seasons=window, stype=stype,
            type_label=S.type_label(stype), board=board, tleaders=tleaders,
            stat=stat, mode=mode, last=last, label=label, short=short, kind=kind, rate=rate,
            mode_label=mode_label, last_label=LAST[last][1],
            side_cols=[c for c in SIDE_COLS if c[0] != stat],
            stat_opts=[{"value": k, "label": CHIP_LABELS[k], "href": href(stat=k)}
                       for k in STAT_ORDER],
            season_opts=[{"value": x, "label": x, "href": href(season=x)} for x in window],
            type_opts=[{"value": v, "label": lab, "href": href(type=v)}
                       for v, lab in (("regular", "Regular season"), ("playoffs", "Playoffs"))],
            mode_opts=[{"value": k, "label": v, "href": href(mode=k)} for k, v in L.MODES.items()],
            last_opts=[{"value": k, "label": v[1], "href": href(last=k)} for k, v in LAST.items()],
            type_value=cur["type"], chances=chances)
