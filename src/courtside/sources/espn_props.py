"""ESPN player / game props (sports.core.api.espn.com): DraftKings (provider 100) and any other
provider ESPN lists for the event. Pure fetch + parse; callers archive the raw pages.

Endpoints (event id == competition id == ``games.espn_id``)::

    {CORE}/events/{id}/competitions/{id}/odds                      -> providers (each has a propBets $ref)
    {CORE}/events/{id}/competitions/{id}/odds/{provider}/propBets  -> items, paginated

``propBets`` takes ``?limit=N&page=P``. The default page is 25 items (43 pages for ~1,060 items);
``limit=1000`` is honoured, so a game is 2 requests instead of 43. (``limit`` above 1000 is
silently ignored and falls back to 25, ``pageSize`` is ignored.) `fetch_propbets` follows
``pageCount`` from the response, so it keeps working if ESPN ever changes the cap.

Item shape (one item = one *price*, not one market)::

    {athlete|team: {$ref}, provider: {$ref}, type: {id, name},
     odds: {american: {value, open}, decimal, fraction, total: {value, open}},
     current: {target: {value}}, open: {target: {value}}, lastUpdated}

OVER / UNDER SIDE SEMANTICS (verified 2026-10-19 on DET-BOS, 1,060 items)
-------------------------------------------------------------------------
Nothing in an item says which side it is: there is no ``side``/``overUnder`` field, and
``odds.total`` repeats the line on both items. An over/under market ("Total Points", "Total
Rebounds", "Total Points and Assists", ...) is a PAIR of items with the same athlete, type and
``current.target``, adjacent in the payload. The side is the payload order: the FIRST item of
the pair is the OVER, the SECOND is the UNDER. Evidence (see tests/test_espn_props.py):

* Line-move test, 31 pairs where ``current.target`` != ``open.target``. If the line went up,
  the over gets cheaper (its implied probability falls). The first item matched "over" in
  28 of 31 pairs; the 3 misses are combo markets (Pts+Ast, PRA) whose juice was re-priced at
  the same time (0.02-0.04 implied-probability moves). Example: athlete
  3078576 Total Points, line 14.5 -> 16.5: item 1 went -123 -> -104 (cheaper = over), item 2
  went -104 -> -123 (dearer = under).
* Milestone cross-check: the "Points Milestones" item for N is "N or more" = over (N - 0.5).
  Of 83 Total {Points,Assists,Rebounds,...} pairs with a milestone at line + 0.5, the milestone
  price was closer to the FIRST item's price in 75; the other 8 are near-ties (both sides
  priced within 3 cents of each other, about -115 either way). Every pair with a meaningful
  price asymmetry agrees.
* The pair is always adjacent (95 of 95 pairs).

So ``over_odds`` = first item, ``under_odds`` = second. This is an inference from price
behaviour, not a documented field: the raw pages are archived, rows record
``side_basis='payload_order'``, and parsing can be re-run if ESPN ever adds a real field.
Two-item groups without a line ("Will There Be Overtime?") are stored as kind ``yesno`` with
the same convention (first = yes, UNVERIFIED). Everything else is a single price:

* ``milestone`` - "Points Milestones" etc.: one item per target, line = N meaning "N or more"
  (over N - 0.5); the price is stored in ``over_odds``.
* ``single`` - one-sided markets (first basket, double-double, team to score first, spreads,
  moneylines): the price is stored in ``over_odds`` ("yes" / the subject wins), line = the
  target if there is one.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable

from .. import http

log = logging.getLogger(__name__)

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SOURCE = "espn"
KIND = "props"
PAGE_LIMIT = 1000     # largest honoured value
MAX_PAGES = 100       # hard stop on a runaway pageCount
SIDE_BASIS = "payload_order"

# type ids whose name contains "Milestones" are N-or-more ladders
MILESTONE_WORD = "Milestones"


# --------------------------------------------------------------------------- fetch

def event_url(espn_id: str | int) -> str:
    return f"{CORE}/events/{espn_id}/competitions/{espn_id}"


def fetch_providers(espn_id: str | int) -> dict:
    """The odds list: one item per provider, each with a ``propBets`` $ref (if it has props)."""
    return http.fetch_json(f"{event_url(espn_id)}/odds")


def fetch_propbets_page(espn_id: str | int, provider_id: str | int, page: int = 1, *,
                        limit: int = PAGE_LIMIT) -> dict:
    return http.fetch_json(f"{event_url(espn_id)}/odds/{provider_id}/propBets",
                           params={"limit": limit, "page": page})


def fetch_propbets(espn_id: str | int, provider_id: str | int, *,
                   on_page: Callable[[int, dict], None] | None = None,
                   limit: int = PAGE_LIMIT) -> list[dict]:
    """All pages of one provider's propBets. `on_page(page_no, payload)` is called right after
    each page is fetched (before anything is parsed) so callers can archive it."""
    pages: list[dict] = []
    page = 1
    while True:
        payload = fetch_propbets_page(espn_id, provider_id, page, limit=limit)
        if on_page:
            on_page(page, payload)
        pages.append(payload)
        count = int(payload.get("pageCount") or 1)
        if page >= count or page >= MAX_PAGES:
            if count > MAX_PAGES:
                log.warning("espn props: %s/%s has %s pages, stopped at %s",
                            espn_id, provider_id, count, MAX_PAGES)
            break
        page += 1
    return pages


def fetch_roster(espn_abbr: str) -> dict:
    """site-API team roster (ESPN abbreviation, e.g. 'NY'): used to map athlete id -> name."""
    return http.fetch_json(f"{SITE}/teams/{espn_abbr}/roster")


def fetch_athlete(athlete_id: str | int, season: int | str | None = None) -> dict:
    """One athlete document (fallback for ids missing from both rosters)."""
    base = f"{CORE}/seasons/{season}/athletes" if season else f"{CORE}/athletes"
    return http.fetch_json(f"{base}/{athlete_id}")


# --------------------------------------------------------------------------- parse helpers

def ref_id(obj: Any) -> str | None:
    """Last path segment of a ``{"$ref": ".../casinos/100?lang=en"}`` object."""
    ref = obj.get("$ref") if isinstance(obj, dict) else None
    if not ref:
        return None
    seg = str(ref).split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return seg or None


def american(value: Any) -> int | None:
    """'+105' / '-120' / 'EVEN' / 105 -> int American odds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    try:
        return int(float(s.replace("+", "")))
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _target(block: Any) -> float | None:
    return _num(((block or {}).get("target") or {}).get("value"))


def providers(payload: dict) -> list[dict]:
    """[{id, name, has_props}] from the odds-list payload."""
    out = []
    for it in payload.get("items") or []:
        prov = it.get("provider") or {}
        pid = str(prov.get("id") or ref_id(it.get("provider")) or "")
        if not pid:
            continue
        out.append({"id": pid, "name": prov.get("name"), "has_props": bool(it.get("propBets"))})
    return out


def roster_names(payload: dict) -> dict[str, str]:
    """{espn athlete id: display name} from a site-API roster payload."""
    out: dict[str, str] = {}
    for a in payload.get("athletes") or []:
        # some roster payloads group athletes by position: {"items": [...]}
        group = a.get("items") if isinstance(a, dict) and "items" in a else [a]
        for p in group or []:
            if p.get("id") and (p.get("displayName") or p.get("fullName")):
                out[str(p["id"])] = p.get("displayName") or p.get("fullName")
    return out


def athlete_name(payload: dict) -> str | None:
    return payload.get("displayName") or payload.get("fullName")


# --------------------------------------------------------------------------- parse

def _items(pages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in pages:
        out.extend(p.get("items") or [])
    return out


def parse_propbets(pages: list[dict], *, provider_id: str | None = None,
                   provider_name: str | None = None) -> list[dict]:
    """Normalised prop rows from one provider's propBets pages (see module docstring).

    Row keys: provider, provider_name, espn_athlete_id, espn_team_id, subject
    ('player'|'team'|'game'), market_id, market, kind ('ou'|'yesno'|'milestone'|'single'),
    line, over_odds, under_odds, open_line, open_over_odds, open_under_odds, last_updated,
    side_basis. `player_name` is added by the caller (`attach_names`).
    """
    items = _items(pages)
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for it in items:
        typ = it.get("type") or {}
        athlete, team = ref_id(it.get("athlete")), ref_id(it.get("team"))
        name = typ.get("name") or ""
        line = _target(it.get("current"))
        milestone = MILESTONE_WORD in name
        # milestones: every target is its own market; other items group by (subject, type,
        # line) so the two prices of an over/under pair land together, in payload order.
        key = (str(typ.get("id")), athlete, team, line, milestone)
        groups.setdefault(key, []).append(it)

    rows: list[dict] = []
    for (type_id, athlete, team, line, milestone), grp in groups.items():
        first = grp[0]
        typ = first.get("type") or {}
        prov = provider_id or ref_id(first.get("provider"))
        base = {
            "provider": prov, "provider_name": provider_name,
            "espn_athlete_id": athlete, "espn_team_id": team,
            "subject": "player" if athlete else ("team" if team else "game"),
            "market_id": type_id, "market": typ.get("name"),
            "line": line, "open_line": _target(first.get("open")),
            "side_basis": SIDE_BASIS,
        }

        def price(it, which="value"):
            return american(((it.get("odds") or {}).get("american") or {}).get(which))

        def updated(items_):
            stamps = [i.get("lastUpdated") for i in items_ if i.get("lastUpdated")]
            return max(stamps) if stamps else None

        if milestone:
            for it in grp:
                rows.append({**base, "kind": "milestone", "over_odds": price(it),
                             "under_odds": None, "open_over_odds": price(it, "open"),
                             "open_under_odds": None, "last_updated": updated([it])})
        elif len(grp) == 2:
            a, b = grp   # first = over (yes), second = under (no): see module docstring
            rows.append({**base, "kind": "ou" if line is not None else "yesno",
                         "over_odds": price(a), "under_odds": price(b),
                         "open_over_odds": price(a, "open"), "open_under_odds": price(b, "open"),
                         "last_updated": updated(grp)})
        else:
            if len(grp) > 2:
                log.warning("espn props: %s items for %s/%s/%s/%s, stored unpaired",
                            len(grp), type_id, athlete, team, line)
            for it in grp:
                rows.append({**base, "kind": "single", "over_odds": price(it),
                             "under_odds": None, "open_over_odds": price(it, "open"),
                             "open_under_odds": None, "last_updated": updated([it])})
    return rows


def attach_names(rows: list[dict], names: dict[str, str]) -> list[dict]:
    for r in rows:
        r["player_name"] = names.get(r["espn_athlete_id"]) if r.get("espn_athlete_id") else None
    return rows


def athlete_ids(rows: list[dict]) -> set[str]:
    return {r["espn_athlete_id"] for r in rows if r.get("espn_athlete_id")}


def row_key(r: dict) -> tuple:
    """Identity of a market within a game/provider (milestones: one per target line)."""
    return (r["provider"], r.get("espn_athlete_id"), r.get("espn_team_id"), r["market_id"],
            r["kind"], r["line"] if r["kind"] == "milestone" else None)


def row_state(r: dict) -> tuple:
    return (r.get("line"), r.get("over_odds"), r.get("under_odds"))
