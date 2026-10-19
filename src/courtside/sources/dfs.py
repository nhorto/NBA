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

Contest detail (light capture, T0.7): ``api.draftkings.com/contests/v1/contests/{id}?format=json``
-> ``contestDetail{entryFee, maximumEntries, maximumEntriesPerUser, isGuaranteed, totalPayouts,
entries, payoutSummary[{minPosition, maxPosition, tierPayoutDescriptions{Cash}, payoutDescriptions
[{value, payoutDescription}]}]}`` (verified 2026-10-19; the tiers sum to ``totalPayouts``).

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

import json
import logging
import re
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
STARTER_ATTR_ID = 100          # playerGameAttributes id 100 = "In Starting Lineup" (value "true")
MIN_STARTERS_PER_GAME = 5      # flagged players needed before an unflagged player counts as 0
CONTEST_URL = API + "/contests/v1/contests/{contest_id}"

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


def fetch_contest(contest_id: int | str, *, retries: int = 0) -> dict:
    """One contest's detail (payout table). A single attempt by default: the caller counts
    requests against the daily budget and decides about retries."""
    return http.fetch_json(CONTEST_URL.format(contest_id=contest_id),
                           params={"format": "json"}, retries=retries)


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
            "start_time": c.get("startTime"), "name": c.get("name"),
            "lineups_available": c.get("startingLineupsAvailable")}
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
            "swappable": None if d.get("isSwappable") is None else int(bool(d["isSwappable"])),
            "starter": 1 if _flagged_starter(d) else None,
            "competition_id": str(comp["competitionId"]) if comp.get("competitionId") else None,
            "game_label": comp.get("name") or cinfo.get("name"),
            "home": cinfo.get("home"), "away": cinfo.get("away"),
            "start_time": comp.get("startTime") or cinfo.get("start_time"),
        })
    _fill_non_starters(rows)
    return rows


def _flagged_starter(d: dict) -> bool:
    for a in d.get("playerGameAttributes") or []:
        if a.get("id") == STARTER_ATTR_ID:
            return str(a.get("value")).strip().lower() in ("true", "1", "yes")
    return False


def _fill_non_starters(rows: list[dict]) -> None:
    """``starter`` is 1 for DK's "In Starting Lineup" flag and NULL while DK has not published
    anything. Once a game has >= MIN_STARTERS_PER_GAME flagged players (lineups are out), its
    unflagged players are 0: "confirmed not starting", not "unknown"."""
    flagged: dict[str | None, set] = {}
    for r in rows:
        if r["starter"]:
            flagged.setdefault(r["competition_id"], set()).add(r["site_player_id"])
    for r in rows:
        if (not r["starter"]
                and len(flagged.get(r["competition_id"], ())) >= MIN_STARTERS_PER_GAME):
            r["starter"] = 0


def slate_games(payload: dict) -> list[dict]:
    """[{competition_id, home, away, start_time}] of one draft group."""
    return [{"competition_id": str(k), **{x: v[x] for x in ("home", "away", "start_time")}}
            for k, v in competitions(payload).items()]


# --------------------------------------------------------------------------- slate tags

SLATE_TAGS = ("main", "early", "late", "turbo", "showdown", "other")


def classify_slates(groups: list[dict]) -> dict[int, str]:
    """{draft_group_id: tag} for the groups of ONE nba day (`lobby_draft_groups` rows, any
    game types; include groups that have already started so a locked main stays "main").

    Showdown (81) -> ``showdown``. Classic (70) groups, in this order:

    1. A suffix keyword (``ContestStartTimeSuffix`` such as " (Early)", " (Late)", " (Night)",
       " (Turbo)") decides: turbo -> ``turbo``, early -> ``early``, late / night / nightcap /
       after dark -> ``late``. Games are the same pool of DK-side games, only the window differs.
    2. Of the groups with no keyword (all Classic groups if every one has one), the *main* slate
       is the one with the most games (DraftKings' own "Featured" tag, then the earliest start,
       break ties). DK gives the full slate no suffix.
    3. Any other keyword-less group: ``early`` if it starts before the main, ``late`` if after,
       ``other`` if it starts together with the main with fewer games (a subset slate).

    Everything else (tiers, single stat, best ball, snake, in-game) is ``other``.
    """
    tags: dict[int, str] = {}
    classic = []
    for g in groups:
        gid, gt = g["draft_group_id"], g["game_type_id"]
        if gt == 81:
            tags[gid] = "showdown"
        elif gt == 70:
            classic.append(g)
        else:
            tags[gid] = "other"

    def keyword(g) -> str | None:
        s = (g.get("suffix") or "").lower()
        if "turbo" in s:
            return "turbo"
        if "early" in s:
            return "early"
        if any(k in s for k in ("late", "night", "after dark")):
            return "late"
        return None

    plain = [g for g in classic if keyword(g) is None] or classic
    main = None
    if plain:
        main = min(plain, key=lambda g: (-(g.get("game_count") or 0),
                                         0 if g.get("tag") == "Featured" else 1,
                                         g.get("start_utc") or "", g["draft_group_id"]))
    for g in classic:
        kw = keyword(g)
        if g is main:
            tags[g["draft_group_id"]] = "main"
        elif kw is not None:
            tags[g["draft_group_id"]] = kw
        else:
            a, b = g.get("start_utc") or "", main.get("start_utc") or ""
            tags[g["draft_group_id"]] = "early" if a < b else "late" if a > b else "other"
    return tags


# --------------------------------------------------------------------------- contests

CONTEST_MIN_FEE = 3.0           # owner default: free and sub-$3 contests are not captured
CONTEST_MIN_FIELD = 100
MAIN_TOP_GPP = 10               # top GPPs by total prizes
MAIN_DOUBLE_UP_CAP = 14         # double-ups / 50-50s, a spread of entry fees when more exist
MAIN_SINGLE_ENTRY_CAP = 6
MAIN_THREE_MAX_CAP = 6
MAIN_CONTEST_CAP = 37
MAX_SHOWDOWN_CONTESTS = 3       # one per Showdown slate, the three richest slates
DAY_CONTEST_CAP = MAIN_CONTEST_CAP + MAX_SHOWDOWN_CONTESTS    # 40


def _flag(attr: dict | None, key: str) -> bool:
    return str((attr or {}).get(key)).lower() == "true"


def lobby_contests(lobby: dict, draft_group_id: int | str | None = None) -> list[dict]:
    """Normalised lobby ``Contests`` rows: ``{contest_id, name, entry_fee, field_size,
    max_per_user, prizes, draft_group_id, game_type_id, double_up, guaranteed, attrs}``.
    Lobby keys: id, n (name), a (fee, dollars), m (max entries), mec (max per user), po (total
    prizes), dg, gameTypeId, attr {IsDoubleUp, IsFiftyfifty, IsGuaranteed, IsQualifier, League,
    IsBeginner, IsWinnerTakeAll, Multiplier ...}."""
    out = []
    for c in lobby.get("Contests") or []:
        if draft_group_id is not None and str(c.get("dg")) != str(draft_group_id):
            continue
        attr = c.get("attr") or {}
        out.append({
            "contest_id": str(c["id"]), "name": (c.get("n") or "").strip(),
            "entry_fee": float(c.get("a") or 0), "field_size": int(c.get("m") or 0),
            "max_per_user": int(c.get("mec") or 0), "prizes": float(c.get("po") or 0),
            "draft_group_id": c.get("dg"), "game_type_id": c.get("gameTypeId"),
            "double_up": _flag(attr, "IsDoubleUp") or _flag(attr, "IsFiftyfifty"),
            "guaranteed": _flag(attr, "IsGuaranteed"), "attrs": dict(attr)})
    return out


def eligible_contest(c: dict) -> bool:
    """Public cash contests that carry an informative payout curve: fee >= $3, field >= 100,
    prizes > 0; no private/H2H leagues, satellites/qualifiers (ticket prizes), beginner-only,
    winner-take-all or multiplier formats."""
    a = c["attrs"]
    if c["entry_fee"] < CONTEST_MIN_FEE or c["field_size"] < CONTEST_MIN_FIELD or c["prizes"] <= 0:
        return False
    return not any(_flag(a, k) for k in ("League", "IsQualifier", "IsBeginner",
                                         "IsWinnerTakeAll", "Multiplier"))


def _spread_by_fee(cs: list[dict], cap: int) -> list[dict]:
    """Up to `cap` contests, taking the biggest field of each entry fee first (cheapest fee
    first), then the second biggest of each fee, and so on."""
    by_fee: dict[float, list[dict]] = {}
    for c in sorted(cs, key=lambda c: (-c["field_size"], c["contest_id"])):
        by_fee.setdefault(c["entry_fee"], []).append(c)
    out: list[dict] = []
    depth = 0
    while len(out) < cap and any(len(v) > depth for v in by_fee.values()):
        for fee in sorted(by_fee):
            if len(by_fee[fee]) > depth and len(out) < cap:
                out.append(by_fee[fee][depth])
        depth += 1
    return out


def select_main_contests(contests: list[dict], *, cap: int = MAIN_CONTEST_CAP) -> list[dict]:
    """The payout-archive picks for one Classic slate, each with a ``selected_as`` list:

    * ``top_gpp``      the MAIN_TOP_GPP non-double-up contests with the biggest total prizes
    * ``double_up``    every double-up / 50-50 (a fee spread when over MAIN_DOUBLE_UP_CAP)
    * ``single_entry`` the MAIN_SINGLE_ENTRY_CAP biggest single-entry (max 1) GPPs
    * ``three_max``    the MAIN_THREE_MAX_CAP biggest 3-entry-max GPPs
    * ``fill_gpp``     any other eligible GPP, biggest prizes first, while under `cap`
    Priority when over `cap`: top_gpp, double_up, single_entry, three_max, fill_gpp."""
    el = [c for c in contests if eligible_contest(c)]
    gpp = sorted((c for c in el if not c["double_up"]),
                 key=lambda c: (-c["prizes"], -c["field_size"], c["contest_id"]))
    picks: dict[str, dict] = {}

    def add(c, why):
        if c["contest_id"] in picks:
            picks[c["contest_id"]]["selected_as"].append(why)
        elif len(picks) < cap:
            picks[c["contest_id"]] = {**c, "selected_as": [why]}

    for c in gpp[:MAIN_TOP_GPP]:
        add(c, "top_gpp")
    for c in _spread_by_fee([c for c in el if c["double_up"]], MAIN_DOUBLE_UP_CAP):
        add(c, "double_up")
    for c in [c for c in gpp if c["max_per_user"] == 1][:MAIN_SINGLE_ENTRY_CAP]:
        add(c, "single_entry")
    for c in [c for c in gpp if c["max_per_user"] == 3][:MAIN_THREE_MAX_CAP]:
        add(c, "three_max")
    for c in gpp:                    # the lobby can be thin: use the cap on the next biggest GPPs
        if c["contest_id"] not in picks:
            add(c, "fill_gpp")
    return list(picks.values())


def select_showdown_contests(lobby: dict, showdown_slate_ids: list, *, limit: int) -> list[dict]:
    """One contest per Showdown slate (its biggest eligible non-double-up), for the `limit`
    slates whose pick has the most prizes."""
    best = []
    for sid in showdown_slate_ids:
        el = [c for c in lobby_contests(lobby, sid) if eligible_contest(c) and not c["double_up"]]
        if el:
            top = max(el, key=lambda c: (c["prizes"], c["field_size"], c["contest_id"]))
            best.append({**top, "selected_as": ["showdown"]})
    best.sort(key=lambda c: (-c["prizes"], c["contest_id"]))
    return best[:max(0, limit)]


# --- contest detail

def _money(text) -> float | None:
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", str(text or ""))
    return float(m.group(1).replace(",", "")) if m else None


def normalize_payouts(detail: dict) -> list[dict]:
    """``contestDetail.payoutSummary`` -> ``[{min_pos, max_pos, payout}]`` (cash per finisher in
    the tier, ascending positions). A tier paying no cash (a ticket) has payout 0.0 and a
    ``non_cash`` text."""
    out = []
    for t in detail.get("payoutSummary") or []:
        lo, hi = t.get("minPosition"), t.get("maxPosition")
        if lo is None or hi is None:
            continue
        pd = (t.get("payoutDescriptions") or [{}])[0]
        val = pd.get("value")
        payout = float(val) if isinstance(val, (int, float)) and val else None
        if payout is None:
            payout = _money((t.get("tierPayoutDescriptions") or {}).get("Cash"))
        if payout is None:
            payout = _money(pd.get("payoutDescription"))
        row = {"min_pos": int(lo), "max_pos": int(hi), "payout": float(payout or 0.0)}
        if not payout:
            row["non_cash"] = (pd.get("payoutDescription")
                               or json.dumps(t.get("tierPayoutDescriptions"), sort_keys=True))
        out.append(row)
    return sorted(out, key=lambda r: r["min_pos"])


def parse_contest(payload: dict, *, lobby_row: dict | None = None, slate_id: str | int | None = None,
                  captured_at: str, raw_path: str | None = None) -> dict:
    """A ``dfs_contests`` row from a contest-detail payload (lobby row optional extras)."""
    d = payload.get("contestDetail") or {}
    tiers = normalize_payouts(d)
    if not d or not tiers:
        raise ValueError("contest payload has no contestDetail.payoutSummary")
    detail_attrs = {k: v for k, v in (d.get("attributes") or {}).items()
                    if "logo" not in k.lower() and "image" not in k.lower()}
    paid = sum((t["max_pos"] - t["min_pos"] + 1) * t["payout"] for t in tiers)
    attrs = {
        "lobby": (lobby_row or {}).get("attrs"), "detail": detail_attrs,
        "selected_as": (lobby_row or {}).get("selected_as"),
        "game_type_id": d.get("gameTypeId"), "contest_state": d.get("contestState"),
        "start_time": d.get("contestStartTime"), "entries_at_capture": d.get("entries"),
        "tiers": len(tiers), "paid_places": tiers[-1]["max_pos"], "payout_sum": paid,
        "scoring_style_id": d.get("scoringStyleId"), "cash_only": d.get("IsCashPrizeOnly")}
    guaranteed = d.get("isGuaranteed")
    return {
        "contest_id": str(d.get("contestKey") or (lobby_row or {}).get("contest_id")),
        "slate_id": str(slate_id if slate_id is not None else d.get("draftGroupId")),
        "name": d.get("name") or (lobby_row or {}).get("name"),
        "entry_fee": d.get("entryFee"), "field_size": d.get("maximumEntries"),
        "max_per_user": d.get("maximumEntriesPerUser"),
        "guaranteed": None if guaranteed is None else int(bool(guaranteed)),
        "total_prizes": d.get("totalPayouts"),
        "payout_json": json.dumps(tiers, separators=(",", ":")),
        "attrs_json": json.dumps(attrs, separators=(",", ":"), default=str),
        "captured_at": captured_at, "raw_path": raw_path}
