"""Compare: 2-4 players or 2 teams side by side (docs/STATS_SITE.md).

Routes
  /compare                       picker + empty state
  /compare?p=<id>&p=<id>...      players (up to 4): per game, per 36, advanced, DK, league
                                 percentiles, last-10 form, head-to-head, shot zones
  /compare?t=<ABBR>&t=<ABBR>     teams (2): ratings + ranks, four factors, head-to-head
  ?season= (3-season window)  ?type=regular|playoffs
  ?add=<text | id | p:<id> | t:<ABBR>>   resolves one more player / team with the search
                                 index and redirects (303) to the clean URL (the no-JS picker)

Unknown or malformed ids are dropped with a notice; when every id given is unknown the page is
a 404. Queries: courtside.stats.compare (+ players / teams / similar caches).
"""

from __future__ import annotations

from datetime import date
from typing import Callable
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import teams as TM
from ..stats import compare as C
from ..stats import players as P
from ..stats import search as SR
from ..stats import seasons as S
from ..stats import teams as ST
from .stats_common import open_db, season_args, season_type_arg
from .teams import identity

CODES = frozenset(t.tricode for t in TM.TEAMS)
SHOT_MIN_FGA = 10              # zone FG% needs this many attempts to be marked the best

# team rating rows: (key, label, kind, better) ; better 0 = no leader (pace, 3PA volume)
TEAM_ROWS = (
    ("win_pct", "Win %", "pct", 1), ("pts", "Points", "num", 1),
    ("opp_pts", "Opp points", "num", -1), ("ortg", "Off rating", "num", 1),
    ("drtg", "Def rating", "num", -1), ("net", "Net rating", "signed", 1),
    ("pace", "Pace", "num", 0), ("efg", "eFG%", "pct", 1), ("ts", "TS%", "pct", 1),
    ("tov", "TOV%", "pct", -1), ("oreb", "OREB%", "pct", 1), ("fg3a", "3PA", "num", 0),
    ("ast", "Assists", "num", 1), ("reb", "Rebounds", "num", 1),
)
TEAM_TIPS = {"opp_pts": "Lower is better", "drtg": "Opponent points per 100 possessions (lower "
             "is better)", "pace": "Possessions per 48 minutes (1st = fastest; no leader)",
             "tov": "Turnovers per 100 plays (lower is better)",
             "fg3a": "3-point attempts per game (volume, no leader)"}


# ------------------------------------------------------------------ helpers
def leaders(values: list, better: int) -> list[bool]:
    """Which entries lead the row: the best non-missing value (ties all lead). No leader when
    `better` is 0 or fewer than two values are present."""
    present = [v for v in values if v is not None]
    if better == 0 or len(present) < 2:
        return [False] * len(values)
    best = max(present) if better > 0 else min(present)
    if all(v == best for v in present):
        return [False] * len(values)
    return [v is not None and v == best for v in values]


def parse_ids(request: Request) -> tuple[list[int], list[str], list[str]]:
    """(player ids, team codes, rejected raw values) from ?p= / ?t= (deduped, capped)."""
    qp = request.query_params
    pids, teams, bad = [], [], []
    for raw in qp.getlist("p"):
        v = (raw or "").strip()
        if v.isdigit() and 0 < len(v) <= 10:
            if int(v) not in pids:
                pids.append(int(v))
        elif v:
            bad.append(v[:20])
    for raw in qp.getlist("t"):
        v = (raw or "").strip().upper()
        if v in CODES:
            if v not in teams:
                teams.append(v)
        elif v:
            bad.append(raw[:20])
    return pids, teams, bad


def player_ident(conn, pid: int) -> dict | None:
    """{player_id, name, team, position, initials} or None for an unknown id (cheap)."""
    last = P._last_box(conn, pid)
    bio = P._one(conn, "SELECT first_name, last_name, position, team FROM player_index "
                       "WHERE player_id=?", (pid,))
    if not last and not bio:
        return None
    name = P.full_name((bio or {}).get("first_name"), (bio or {}).get("last_name")) \
        or (last or {}).get("player_name") or f"Player {pid}"
    team = (last or {}).get("team") or (bio or {}).get("team")
    ini = "".join(w[0] for w in name.replace("-", " ").split() if w[:1].isalpha())[:3]
    return {"player_id": pid, "name": name, "team": team, "position": (bio or {}).get("position"),
            "initials": ini.upper() or "?", "last_game": (last or {}).get("game_date")}


class Urls:
    """Builds /compare links that keep the current selection, season and type."""

    def __init__(self, pids, teams, season, window, stype):
        self.pids, self.teams, self.season, self.window, self.stype = \
            list(pids), list(teams), season, window, stype

    def __call__(self, *, pids=None, teams=None, season=None, stype=None) -> str:
        pids = self.pids if pids is None else pids
        teams = self.teams if teams is None else teams
        season = season or self.season
        stype = stype or self.stype
        q: list[tuple[str, str]] = [("p", str(p)) for p in pids] + [("t", t) for t in teams]
        if season != self.window[0]:
            q.append(("season", season))
        if stype == S.PLAYOFFS:
            q.append(("type", "playoffs"))
        return "/compare" + ("?" + urlencode(q) if q else "")

    def hidden(self) -> list[tuple[str, str]]:
        """Hidden inputs of the add form (everything but the new entity)."""
        q = [("p", str(p)) for p in self.pids] + [("t", t) for t in self.teams]
        if self.season != self.window[0]:
            q.append(("season", self.season))
        if self.stype == S.PLAYOFFS:
            q.append(("type", "playoffs"))
        return q


def resolve_add(conn, text: str, mode: str | None, today: date) -> tuple[str, str] | None:
    """('p', '203999') / ('t', 'DEN') for the picker's free text, or None."""
    a = (text or "").strip()[:SR.MAX_Q]
    if not a:
        return None
    if ":" in a:
        kind, _, val = a.partition(":")
        if kind == "p" and val.isdigit():
            return ("p", val)
        if kind == "t" and val.upper() in CODES:
            return ("t", val.upper())
        return None
    if a.isdigit():
        return ("p", a)
    if mode != "p" and a.upper() in CODES:
        return ("t", a.upper())
    want = {"p": "player", "t": "team"}.get(mode or "")
    for h in SR.search(conn, a, limit=12, today=today) if conn is not None else []:
        if want is None or h["type"] == want:
            return ("p", str(h["id"])) if h["type"] == "player" else ("t", h["id"])
    return None


# ------------------------------------------------------------------ players
def players_view(conn, idents: list[dict], season: str, stype: str) -> dict:
    pids = [i["player_id"] for i in idents]
    cols = [C.player(conn, pid, season, stype) for pid in pids]
    for c, ident in zip(cols, idents):
        c["ident"] = ident
    sections = []
    for title, rows in C.PLAYER_SECTIONS:
        out_rows = []
        for key, label, kind, nd, better, tip in rows:
            vals = [c["values"].get(key) for c in cols]
            if all(v is None for v in vals):
                continue
            lead = leaders(vals, better)
            out_rows.append({"key": key, "label": label, "kind": kind, "nd": nd, "tip": tip,
                             "better": better,
                             "cells": [{"v": v, "lead": ld, "pct": c["pctile"].get(key)}
                                       for v, ld, c in zip(vals, lead, cols)]})
        if out_rows:
            sections.append({"title": title, "rows": out_rows})
    # last-10 form
    form_rows = []
    for k, label, better in C.FORM_STATS:
        vals = [c["form"]["avg"].get(k) if c["form"]["n"] else None for c in cols]
        lead = leaders(vals, better)
        form_rows.append({"key": k, "label": label,
                          "cells": [{"v": v, "lead": ld, "diff": c["form"]["diff"].get(k)}
                                    for v, ld, c in zip(vals, lead, cols)]})
    dk_max = max((g["v"] for c in cols for g in c["form"]["dk"]), default=0) or 1
    # head to head
    h2h = C.head_to_head(conn, pids) if len(pids) >= 2 else {"games": [], "summary": {}}
    h2h_rows = []
    if h2h["summary"]:
        for k, label, kind in (("games", "Games", "int"), ("rec", "Record", "rec"),
                               ("pts", "Points", "num"), ("reb", "Rebounds", "num"),
                               ("ast", "Assists", "num"), ("dk", "DK pts", "num")):
            sm = [h2h["summary"].get(p) for p in pids]
            if k == "rec":
                vals = [(s["w"] / s["games"]) if s and s["games"] else None for s in sm]
            else:
                vals = [s.get(k) if s else None for s in sm]
            lead = leaders(vals, 0 if k == "games" else 1)
            h2h_rows.append({"key": k, "label": label, "kind": kind,
                             "cells": [{"v": v, "lead": ld, "s": s}
                                       for v, ld, s in zip(vals, lead, sm)]})
    # shot zones
    zone_names = [z for z in P.ZONES if z != "Backcourt"]
    league = P.league_zones(conn, season, stype) if conn is not None else {}
    zrows = []
    for zname in zone_names:
        zs = [next((z for z in c["zones"] if z["zone"] == zname), None) for c in cols]
        if not any(zs):
            continue
        fg = [z["fg_pct"] if z and z["fga"] >= SHOT_MIN_FGA else None for z in zs]
        lead = leaders(fg, 1)
        zrows.append({"zone": zname, "league": (league.get(zname) or {}).get("fg_pct"),
                      "cells": [{"z": z, "lead": ld} for z, ld in zip(zs, lead)]})
    share_max = max((z["z"]["share"] for r in zrows for z in r["cells"] if z["z"]), default=0)
    return {"cols": cols, "sections": sections, "form_rows": form_rows, "dk_max": dk_max,
            "h2h": h2h, "h2h_rows": h2h_rows, "h2h_games": h2h["games"][:C.H2H_SHOW],
            "zones": zrows, "share_max": share_max or 1,
            "shots_any": any(c["shots_n"] for c in cols),
            "pool_n": cols[0]["pool_n"] if cols else 0}


# ------------------------------------------------------------------ teams
def teams_view(conn, codes: list[str], season: str, stype: str) -> dict:
    lg = ST.league(conn, season, stype) if conn is not None else None
    by = (lg or {}).get("by_team") or {}
    n = (lg or {}).get("n") or 30
    place = ST.standings(conn, season, stype)["rank"] if conn is not None and lg else {}
    cols = [{"code": c, "t": identity(c), "row": by.get(c), "place": place.get(c)}
            for c in codes]
    rows = []
    for key, label, kind, better in TEAM_ROWS:
        vals = [c["row"].get(key) if c["row"] else None for c in cols]
        rks = [c["row"]["rk"].get(key) if c["row"] else None for c in cols]
        # leader by league rank (rank 1 = best in every ranked direction)
        lead = leaders([-r if r is not None else None for r in rks], better and 1)
        rows.append({"key": key, "label": label, "kind": kind, "tip": TEAM_TIPS.get(key),
                     "cells": [{"v": v, "rk": rk, "lead": ld}
                               for v, rk, ld in zip(vals, rks, lead)]})
    ff = C.team_factors(conn, season, stype) if conn is not None else {"by_team": {}, "n": 0}
    ff_sections = []
    for _, title, items in C.FACTORS:
        frows = []
        for key, label, _ in items:
            vals = [(ff["by_team"].get(c) or {}).get(key) for c in codes]
            rks = [((ff["by_team"].get(c) or {}).get("rk") or {}).get(key) for c in codes]
            lead = leaders([-r if r is not None else None for r in rks], 1)
            frows.append({"key": key, "label": label,
                          "cells": [{"v": v, "rk": rk, "lead": ld}
                                    for v, rk, ld in zip(vals, rks, lead)]})
        ff_sections.append({"title": title, "rows": frows})
    h2h = (C.team_h2h(conn, codes[0], codes[1], (season, S.prev_season(season)))
           if len(codes) == 2 and conn is not None else [])
    return {"tcols": cols, "trows": rows, "league_n": n, "ff": ff_sections, "ff_n": ff["n"] or 30,
            "th2h": h2h, "any_row": any(c["row"] for c in cols)}


# ------------------------------------------------------------------ route
def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request):
        t = today()
        season, window = season_args(request, t)
        stype = season_type_arg(request)
        pids, codes, bad = parse_ids(request)
        given = len(request.query_params.getlist("p")) + len(request.query_params.getlist("t"))
        notices = [f"“{b}” is not a player id or team code." for b in bad]
        with open_db(store) as conn:
            idents = []
            for pid in pids:
                ident = player_ident(conn, pid) if conn is not None else None
                if ident:
                    idents.append(ident)
                else:
                    notices.append(f"No player with id {pid}.")
            mode = "p" if idents else ("t" if codes else None)
            add = request.query_params.get("add")
            if add is not None:
                hit = resolve_add(conn, add, mode, t)
                urls = Urls([i["player_id"] for i in idents], codes if mode != "p" else [],
                            season, window, stype)
                if hit and (mode is None or hit[0] == mode):
                    kind, val = hit
                    if kind == "p" and int(val) not in urls.pids:
                        urls.pids.append(int(val))
                    elif kind == "t" and val not in urls.teams:
                        urls.teams.append(val)
                    return RedirectResponse(urls(), status_code=303)
                if add.strip():
                    what = {"p": "player", "t": "team"}.get(mode or "", "player or team")
                    notices.append(f"No {what} matches “{add.strip()[:40]}”.")
            if given and not idents and not codes:
                return not_found("Nothing to compare",
                                 "None of those player ids or team codes exist. "
                                 "Start again at /compare.")
            if mode == "p" and codes:
                notices.append("Players and teams can’t be mixed: showing the players.")
            if len(idents) > C.MAX_PLAYERS:
                notices.append(f"Up to {C.MAX_PLAYERS} players: showing the first "
                               f"{C.MAX_PLAYERS}.")
                idents = idents[:C.MAX_PLAYERS]
            if mode == "t" and len(codes) > C.MAX_TEAMS:
                notices.append(f"Two teams at a time: showing {codes[0]} and {codes[1]}.")
                codes = codes[:C.MAX_TEAMS]
            urls = Urls([i["player_id"] for i in idents], codes if mode == "t" else [],
                        season, window, stype)
            ctx: dict = {"mode": mode, "notices": notices, "season": season, "window": window,
                         "stype": stype, "type_label": S.type_label(stype),
                         "prev_season": S.prev_season(season), "hidden": urls.hidden(),
                         "season_opts": [{"value": s, "label": s, "href": urls(season=s)}
                                         for s in window],
                         "type_opts": [{"value": S.REGULAR, "label": "Regular season",
                                        "href": urls(stype=S.REGULAR)},
                                       {"value": S.PLAYOFFS, "label": "Playoffs",
                                        "href": urls(stype=S.PLAYOFFS)}]}
            if mode == "p":
                ctx["chips"] = [{"label": i["name"], "team": i["team"],
                                 "href": f"/player/{i['player_id']}?season={season}",
                                 "remove": urls(pids=[x for x in urls.pids
                                                      if x != i["player_id"]])}
                                for i in idents]
                ctx["can_add"] = len(idents) < C.MAX_PLAYERS
                ctx.update(players_view(conn, idents, season, stype) if conn is not None
                           else {"cols": []})
            elif mode == "t":
                ctx["chips"] = [{"label": identity(c)["name"], "team": c,
                                 "href": f"/team/{c}?season={season}",
                                 "remove": urls(teams=[x for x in codes if x != c])}
                                for c in codes]
                ctx["can_add"] = len(codes) < C.MAX_TEAMS
                ctx.update(teams_view(conn, codes, season, stype))
            else:
                ctx["chips"], ctx["can_add"] = [], True
        return render("stats_compare.html", page="players", **ctx)
