"""Action Network line snapshots (api.actionnetwork.com): per-book moneyline / spread / total.

UNDOCUMENTED API: personal use only, at most one request per pregame run, never in a loop.

    GET /web/v1/scoreboard/nba?period=game&date=YYYYMMDD&bookIds=15,30,68,...

* ``date`` is the US/Eastern day (the 9:30 pm ET game of the day is included).
* ``bookIds`` is REQUIRED to get more than the default two books; unknown ids are ignored.
  Without it you only get 15 (Consensus) and 30 (Open). ``GET /web/v1/books`` lists all 459.
* Each game has ``odds``: one object per book with ``spread_home`` (negative = home favoured,
  same sign convention as ``odds_snapshots.home_spread``), ``spread_home_line`` /
  ``spread_away_line`` (prices), ``total`` with ``over`` / ``under`` prices, ``ml_home`` /
  ``ml_away``, team totals (``home_total`` ...), betting-percentage fields (null until the
  regular season / until enough action) and ``inserted`` = when that book's line was last
  written. Book 30 ("Open") is the opening line; book 15 is Action Network's consensus.
* Games carry ``status`` (scheduled | inprogress | complete); only ``scheduled`` games are worth
  storing as pre-game lines (live games return in-play lines).

Pure fetch + parse; callers archive the raw payload.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from .. import http, teams

log = logging.getLogger(__name__)

URL = "https://api.actionnetwork.com/web/v1/scoreboard/nba"
SOURCE = "actionnetwork"

# book_id -> stable lowercase name stored in line_snapshots.book (New Jersey books; ids from
# /web/v1/books). Ids not requested/returned are simply absent.
BOOKS: dict[int, str] = {
    15: "consensus", 30: "open", 68: "draftkings", 69: "fanduel", 75: "betmgm",
    123: "caesars", 71: "betrivers", 79: "bet365", 2724: "hardrock", 2988: "fanatics",
    3: "pinnacle", 78: "circa",
}
BOOK_IDS = tuple(BOOKS)


# --------------------------------------------------------------------------- fetch

def fetch_scoreboard(day: date, *, book_ids: tuple[int, ...] = BOOK_IDS) -> dict:
    return http.fetch_json(URL, params={
        "period": "game", "date": day.strftime("%Y%m%d"),
        "bookIds": ",".join(str(b) for b in book_ids)})


# --------------------------------------------------------------------------- parse

def _tricode(team: dict) -> str | None:
    abbr = (team or {}).get("abbr")
    if abbr:
        try:
            return teams.franchise(abbr)
        except KeyError:
            pass
    name = (team or {}).get("full_name") or ""
    for t in teams.TEAMS:
        if name.lower() == f"{t.city} {t.name}".lower():
            return t.tricode
    return None


def _int(v: Any) -> int | None:
    try:
        return None if v is None else int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def scoreboard_games(payload: dict) -> list[dict]:
    """[{an_game_id, start_time, status, home, away, books: [line rows]}]; games whose teams
    do not map to NBA teams are skipped (warned)."""
    out = []
    for g in payload.get("games") or []:
        by_id = {t.get("id"): t for t in g.get("teams") or []}
        home = _tricode(by_id.get(g.get("home_team_id")))
        away = _tricode(by_id.get(g.get("away_team_id")))
        if home is None or away is None:
            log.warning("action network: skipping game %s: teams do not map to NBA teams",
                        g.get("id"))
            continue
        books = []
        for o in g.get("odds") or []:
            if o.get("type") not in (None, "game"):
                continue
            bid = _int(o.get("book_id"))
            books.append({
                "book_id": bid, "book": BOOKS.get(bid, f"book_{bid}"),
                "is_open": 1 if bid == 30 else 0,
                "home_spread": _float(o.get("spread_home")),
                "home_spread_price": _int(o.get("spread_home_line")),
                "away_spread_price": _int(o.get("spread_away_line")),
                "total": _float(o.get("total")),
                "over_price": _int(o.get("over")), "under_price": _int(o.get("under")),
                "home_ml": _int(o.get("ml_home")), "away_ml": _int(o.get("ml_away")),
                "home_team_total": _float(o.get("home_total")),
                "away_team_total": _float(o.get("away_total")),
                "line_updated_at": o.get("inserted"),
                "spread_home_bets_pct": _float(o.get("spread_home_public")),
                "spread_home_money_pct": _float(o.get("spread_home_money")),
                "total_over_bets_pct": _float(o.get("total_over_public")),
                "total_over_money_pct": _float(o.get("total_over_money")),
                "ml_home_bets_pct": _float(o.get("ml_home_public")),
                "ml_home_money_pct": _float(o.get("ml_home_money")),
                "num_bets": _int(o.get("num_bets")),
            })
        out.append({"an_game_id": g.get("id"), "start_time": g.get("start_time"),
                    "status": g.get("real_status") or g.get("status"), "home": home,
                    "away": away, "books": books})
    return out
