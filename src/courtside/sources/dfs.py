"""DraftKings daily-fantasy salaries (public, unauthenticated endpoints). FanDuel: see below.

Endpoints (all verified 2026-10-19; plain GET with the shared chrome-impersonating client)::

    www.draftkings.com/lobby/getcontests?sport=NBA              ~1.4 MB: DraftGroups, GameTypes, GameSets
    api.draftkings.com/draftgroups/v1/draftgroups/{id}/draftables   salaries, one item per roster slot
    api.draftkings.com/lineups/v1/gametypes/{gameTypeId}/rules      roster slots, salary cap, minimums
    api.draftkings.com/draftgroups/v1/{id}                          group metadata (not needed)

A *draft group* is a slate (set of games) of one *game type*: 70 Classic, 81 Showdown Captain
Mode, 73 Tiers, 112 In-Game Showdown (2H), 343 Single Stat - Points, 170 Best Ball, 188 Snake,
193 Snake Showdown. Only Classic and Showdown carry salaries a lineup optimizer needs, so those
are what the collector fetches (`SALARY_GAME_TYPES`). Draft groups appear a day or more ahead
and stay "Upcoming" until the first tip.

Draftables: ``{draftableId, playerId, playerDkId, displayName, position ("PG/SG"), rosterSlotId,
salary, status ("None"|"O"|"Q"|"GTD"), teamAbbreviation, competition{competitionId, name},
draftStatAttributes[{id: 219 = FPPG}]}``. A player is listed once per roster slot he is eligible
for (Classic: PG/SG/SF/PF/C/G/F/UTIL, so a "PG/SG" is 4 rows with the same salary); in
Showdown the Captain slot (476) carries 1.5x the UTIL (475) salary. ``playerId`` is DraftKings'
player key, ``playerDkId`` the id in DK's image URLs; both are kept raw.

FANDUEL: not freely accessible. ``api.fanduel.com/fixture-lists?sport=NBA`` (and every other
api.fanduel.com path tried) returns HTTP 401 "client failed to provide authentication
credentials"; it needs a logged-in session token. Skipped.

Pure fetch + parse; callers archive the raw payloads.
"""

from __future__ import annotations

import logging
from datetime import date

from .. import http, teams

log = logging.getLogger(__name__)

SITE = "dk"
SOURCE = "draftkings"
LOBBY_URL = "https://www.draftkings.com/lobby/getcontests"
API = "https://api.draftkings.com"

GAME_TYPES: dict[int, str] = {
    70: "classic", 81: "showdown", 73: "tiers", 112: "in_game_showdown",
    343: "single_stat_points", 170: "best_ball", 188: "snake", 193: "snake_showdown",
}
SALARY_GAME_TYPES = (70, 81)
FPPG_STAT_ID = 219

# --- DraftKings NBA Classic scoring + roster construction -----------------------------------
# Source: https://www.draftkings.com/help/rules/nba (the rules page for gameType 70 is
# /help/rules/4/70, reported by the rules endpoint as ``rulesUrl``). The page is a client-side
# app, so the point values are NOT in any JSON endpoint and are recorded here. Roster slots, the
# salary cap and the game/team minimums ARE served by `fetch_rules` and archived on every run.
CLASSIC_SCORING: dict[str, float] = {
    "point": 1.0, "three_pointer_made": 0.5, "rebound": 1.25, "assist": 1.5,
    "steal": 2.0, "block": 2.0, "turnover": -0.5,
    "double_double": 1.5,    # 10+ in two of: points, rebounds, assists, steals, blocks
    "triple_double": 3.0,    # 10+ in three of the same (the double-double bonus also applies)
}
CLASSIC_ROSTER = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")    # 8 players, $50,000 cap
CLASSIC_SALARY_CAP = 50_000
SHOWDOWN_ROSTER = ("CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL")   # CPT: 1.5x salary and points
SHOWDOWN_CAPTAIN_MULTIPLIER = 1.5


def classic_points(pts=0, fg3m=0, reb=0, ast=0, stl=0, blk=0, tov=0) -> float:
    """DraftKings NBA fantasy points for one stat line (Classic / Showdown UTIL)."""
    cats = sum(1 for v in (pts, reb, ast, stl, blk) if v >= 10)
    total = (pts * CLASSIC_SCORING["point"] + fg3m * CLASSIC_SCORING["three_pointer_made"]
             + reb * CLASSIC_SCORING["rebound"] + ast * CLASSIC_SCORING["assist"]
             + stl * CLASSIC_SCORING["steal"] + blk * CLASSIC_SCORING["block"]
             + tov * CLASSIC_SCORING["turnover"])
    if cats >= 2:
        total += CLASSIC_SCORING["double_double"]
    if cats >= 3:
        total += CLASSIC_SCORING["triple_double"]
    return total


# --------------------------------------------------------------------------- fetch

def fetch_lobby() -> dict:
    return http.fetch_json(LOBBY_URL, params={"sport": "NBA"})


def fetch_draftables(draft_group_id: int | str) -> dict:
    return http.fetch_json(f"{API}/draftgroups/v1/draftgroups/{draft_group_id}/draftables")


def fetch_rules(game_type_id: int | str) -> dict:
    return http.fetch_json(f"{API}/lineups/v1/gametypes/{game_type_id}/rules")


# --------------------------------------------------------------------------- parse

def _tricode(abbr: str | None) -> str | None:
    if not abbr:
        return None
    try:
        return teams.franchise(abbr)
    except KeyError:
        return None


def _game_type_name(game_type_id: int, lobby: dict | None = None) -> str:
    if game_type_id in GAME_TYPES:
        return GAME_TYPES[game_type_id]
    for gt in (lobby or {}).get("GameTypes") or []:
        if gt.get("GameTypeId") == game_type_id and gt.get("Name"):
            return str(gt["Name"]).lower().replace(" ", "_")
    return f"type_{game_type_id}"


def lobby_draft_groups(lobby: dict, *, day: date | None = None,
                       game_types: tuple[int, ...] | None = None) -> list[dict]:
    """Draft groups from the lobby: ``{draft_group_id, game_type_id, game_type, start_utc,
    start_et, game_count, suffix, tag}``. `day` keeps groups whose Eastern start date is that
    day; `game_types` restricts the type (default: all)."""
    out = []
    for g in lobby.get("DraftGroups") or []:
        if g.get("Sport") not in (None, "NBA"):
            continue
        gt = g.get("GameTypeId") or g.get("ContestTypeId")
        if gt is None or (game_types is not None and gt not in game_types):
            continue
        start_et = (g.get("StartDateEst") or "")[:19]
        if day is not None and start_et[:10] != day.isoformat():
            continue
        out.append({
            "draft_group_id": g["DraftGroupId"], "game_type_id": gt,
            "game_type": _game_type_name(gt, lobby),
            "start_utc": (g.get("StartDate") or "")[:19] + "Z" if g.get("StartDate") else None,
            "start_et": start_et, "game_count": g.get("GameCount"),
            "suffix": (g.get("ContestStartTimeSuffix") or "").strip(), "tag": g.get("DraftGroupTag"),
        })
    return sorted(out, key=lambda r: (r["start_utc"] or "", r["draft_group_id"]))


def parse_rules(rules: dict) -> dict:
    """{game_type_id, name, salary_cap, min_games, min_teams, slots: [{id, name, order,
    note}], rules_url}. Slot order is the lineup order (UTIL last)."""
    slots = []
    for t in rules.get("lineupTemplate") or []:
        rs = t.get("rosterSlot") or {}
        slots.append({"id": rs.get("id"), "name": rs.get("name"), "order": t.get("order"),
                      "note": rs.get("positionTip")})
    cap = rules.get("salaryCap") or {}
    return {"game_type_id": rules.get("gameTypeId"), "name": rules.get("gameTypeName"),
            "salary_cap": cap.get("maxValue"),
            "min_games": (rules.get("gameCount") or {}).get("minValue"),
            "min_teams": (rules.get("teamCount") or {}).get("minValue"),
            "slots": sorted(slots, key=lambda s: s["order"] or 0),
            "rules_url": rules.get("rulesUrl")}


def slot_names(rules: dict | None) -> dict[int, str]:
    """{rosterSlotId: 'PG'|...|'UTIL'|'CPT'} from a parsed (or raw) rules payload."""
    if rules is None:
        return {}
    parsed = rules if "slots" in rules else parse_rules(rules)
    return {s["id"]: s["name"] for s in parsed["slots"] if s.get("id") is not None}


def _fppg(d: dict) -> float | None:
    for a in d.get("draftStatAttributes") or []:
        if a.get("id") == FPPG_STAT_ID:
            try:
                return float(a.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def competitions(payload: dict) -> dict[int, dict]:
    """{competitionId: {home, away, start_time, name}} from a draftables payload."""
    out = {}
    for c in payload.get("competitions") or []:
        out[c["competitionId"]] = {
            "home": _tricode((c.get("homeTeam") or {}).get("abbreviation")),
            "away": _tricode((c.get("awayTeam") or {}).get("abbreviation")),
            "start_time": c.get("startTime"), "name": c.get("name")}
    return out


def parse_draftables(payload: dict, *, draft_group_id: int | str, game_type_id: int,
                     rules: dict | None = None, lobby: dict | None = None) -> list[dict]:
    """One row per draftable (= player x roster slot). Unknown teams map to None; draftables
    that are disabled (``isDisabled``) are kept with ``disabled=1``."""
    comps = competitions(payload)
    slots = slot_names(rules)
    gt_name = _game_type_name(game_type_id, lobby)
    rows = []
    for d in payload.get("draftables") or []:
        comp = d.get("competition") or {}
        cinfo = comps.get(comp.get("competitionId")) or {}
        team = _tricode(d.get("teamAbbreviation"))
        slot_id = d.get("rosterSlotId")
        rows.append({
            "site": SITE, "slate_id": str(draft_group_id), "game_type": gt_name,
            "game_type_id": game_type_id, "draftable_id": str(d.get("draftableId")),
            "player_name": d.get("displayName")
                or " ".join(x for x in (d.get("firstName"), d.get("lastName")) if x),
            "site_player_id": str(d["playerId"]) if d.get("playerId") is not None else None,
            "site_player_dk_id": str(d["playerDkId"]) if d.get("playerDkId") is not None
                else None,
            "team": team, "positions": d.get("position"),
            "roster_slot_id": slot_id, "roster_slot": slots.get(slot_id),
            "salary": d.get("salary"), "status": d.get("status"),
            "news_status": d.get("newsStatus"), "fppg": _fppg(d),
            "disabled": 1 if d.get("isDisabled") else 0,
            "competition_id": str(comp["competitionId"]) if comp.get("competitionId") else None,
            "game_label": comp.get("name") or cinfo.get("name"),
            "home": cinfo.get("home"), "away": cinfo.get("away"),
            "start_time": comp.get("startTime") or cinfo.get("start_time"),
        })
    return rows


def slate_games(payload: dict) -> list[dict]:
    """[{competition_id, home, away, start_time}] of one draft group."""
    return [{"competition_id": str(k), **{x: v[x] for x in ("home", "away", "start_time")}}
            for k, v in competitions(payload).items()]
