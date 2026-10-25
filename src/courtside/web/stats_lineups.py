"""Lineup explorer: on/off for any set of players, WOWY and the best / worst combos of one
team-season (docs/STATS_SITE.md, queries stats/lineups.py from `stints`).

Routes
  /lineups                                  pick a team
  /lineups?team=DEN&season=2025-26          ?type=regular|playoffs
      on=<id> (1-5, repeatable)             players who must be on the floor
      off=<id> (repeatable)                 players who must be off it
      wowy=<id>                             with / without one player, per teammate
      min=<possessions>                     the combos' minimum (default by games covered)

The team-season's stints are folded into units once and cached in memory (stats.db.cached), so
a warm request is pure Python over a few hundred units. Unknown ids are dropped with a notice.
"""

from __future__ import annotations

from datetime import date
from typing import Callable
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse

from .. import teams as TM
from ..stats import lineups as LU
from ..stats import seasons as S
from ..stats import teams as ST
from .stats_common import open_db, season_args, season_type_arg
from .teams import identity

CODES = frozenset(t.tricode for t in TM.TEAMS)
MAX_ON, MAX_OFF = 5, 10
MIN_POSS_MAX = 2000


def default_min(games: int) -> int:
    """Default minimum possessions for the combo tables: the team page's 5-man rule (50 once
    25+ games have lineup data, else 2 per game, at least 10)."""
    return ST.lineup_min_poss(games)


def _ids(qp, key: str, notices: list[str]) -> list[int]:
    out: list[int] = []
    for raw in qp.getlist(key):
        v = (raw or "").strip()
        if v.isdigit() and len(v) <= 10:
            if int(v) not in out:
                out.append(int(v))
        elif v:
            notices.append(f"“{v[:20]}” is not a player id.")
    return out


class Urls:
    def __init__(self, team, season, window, stype, on, off, wowy, min_poss, min_default):
        self.cur = {"team": team, "season": season, "stype": stype, "on": list(on),
                    "off": list(off), "wowy": wowy, "min": min_poss}
        self.window, self.min_default = window, min_default

    def __call__(self, **over) -> str:
        p = {**self.cur, **over}
        q: list[tuple[str, str]] = [("team", p["team"])] if p["team"] else []
        if p["season"]:
            q.append(("season", p["season"]))
        if p["stype"] == S.PLAYOFFS:
            q.append(("type", "playoffs"))
        q += [("on", str(i)) for i in p["on"]] + [("off", str(i)) for i in p["off"]]
        if p["wowy"]:
            q.append(("wowy", str(p["wowy"])))
        if p["min"] is not None and p["min"] != self.min_default:
            q.append(("min", str(p["min"])))
        return "/lineups" + ("?" + urlencode(q) if q else "")


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/lineups", response_class=HTMLResponse)
    def lineups(request: Request):
        season, window = season_args(request, today())
        stype = season_type_arg(request)
        if stype not in (S.REGULAR, S.PLAYOFFS):
            stype = S.REGULAR
        qp = request.query_params
        notices: list[str] = []
        raw_team = (qp.get("team") or "").strip().upper()
        ctx: dict = {"season": season, "window": window, "stype": stype,
                     "type_label": S.type_label(stype), "notices": notices,
                     "teams": sorted(CODES), "tool": "lineups"}
        if raw_team and raw_team not in CODES:
            return not_found("Unknown team", f"“{raw_team[:12]}” is not a team code. "
                             "Pick a team at /lineups.")
        if not raw_team:
            return render("stats_lineups.html", page="tools", tcode=None, data=None, **ctx)
        team = raw_team
        on = _ids(qp, "on", notices)
        off = _ids(qp, "off", notices)
        wowy_raw = _ids(qp, "wowy", notices)
        with open_db(store) as conn:
            data = LU.load(conn, team, season, stype) if conn is not None else None
        games = (data or {}).get("games") or 0
        names = (data or {}).get("names") or {}
        mdef = default_min(games)
        raw_min = (qp.get("min") or "").strip()
        min_poss = mdef
        if raw_min:
            if raw_min.isdigit():
                min_poss = max(0, min(MIN_POSS_MAX, int(raw_min)))
            else:
                notices.append("Minimum possessions must be a whole number: using the default.")
        if games:
            for lst, what in ((on, "on"), (off, "off"), (wowy_raw, "WOWY")):
                bad = [p for p in lst if p not in names]
                for p in bad:
                    notices.append(f"Player {p} never played for {team} in {season} "
                                   f"({what} ignored).")
                lst[:] = [p for p in lst if p in names]
            both = [p for p in off if p in on]
            if both:
                notices.append("A player can't be both on and off: kept him on.")
                off = [p for p in off if p not in on]
            if len(on) > MAX_ON:
                notices.append(f"Up to {MAX_ON} players on: kept the first {MAX_ON}.")
                on = on[:MAX_ON]
            if len(off) > MAX_OFF:
                notices.append(f"Up to {MAX_OFF} players off: kept the first {MAX_OFF}.")
                off = off[:MAX_OFF]
        wowy_pid = wowy_raw[0] if wowy_raw else None
        urls = Urls(team, season, window, stype, on, off, wowy_pid, min_poss, mdef)
        sel = LU.select(data, on, off) if games and (on or off) else None
        team_r = LU.rtg(data["total"]) if games else None
        wy = LU.wowy(data, wowy_pid) if games and wowy_pid else None
        combos = {k: LU.combos(data, k, min_poss) for k in LU.SIZES} if games else {}
        roster = []
        for p in (data or {}).get("players", []):
            pid = p["player_id"]
            roster.append({**p, "on": pid in on, "off": pid in off, "wowy": pid == wowy_pid,
                           "wowy_href": urls(wowy=None if pid == wowy_pid else pid)})
        sel_names = {"on": [names.get(p, f"#{p}") for p in on],
                     "off": [names.get(p, f"#{p}") for p in off]}
        return render(
            "stats_lineups.html", page="tools", tcode=team, t=identity(team), data=data,
            games=games, roster=roster, on=on, off=off, sel=sel, team_r=team_r, wy=wy,
            wowy_pid=wowy_pid, wowy_name=names.get(wowy_pid) if wowy_pid else None,
            combos=combos, min_poss=min_poss, min_default=mdef, min_max=MIN_POSS_MAX,
            sel_names=sel_names, clear_href=urls(on=[], off=[]),
            season_opts=[{"value": x, "label": x,
                          "href": Urls(team, x, window, stype, [], [], None, None, None)()}
                         for x in window],
            type_opts=[{"value": v, "label": lab,
                        "href": Urls(team, season, window, v, [], [], None, None, None)()}
                       for v, lab in ((S.REGULAR, "Regular season"), (S.PLAYOFFS, "Playoffs"))],
            **ctx)
