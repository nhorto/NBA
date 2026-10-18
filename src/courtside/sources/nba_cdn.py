"""cdn.nba.com source: schedule, scoreboard, box score, play-by-play, odds.

Pure fetch + parse. No archiving and no DB access; the collect job does both.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from courtside import config
from courtside.http import fetch_json

BASE = "https://cdn.nba.com/static/json"
SCHEDULE_URL = f"{BASE}/staticData/scheduleLeagueV2.json"
SCOREBOARD_URL = f"{BASE}/liveData/scoreboard/todaysScoreboard_00.json"
BOXSCORE_URL = BASE + "/liveData/boxscore/boxscore_{game_id}.json"
PLAYBYPLAY_URL = BASE + "/liveData/playbyplay/playbyplay_{game_id}.json"
ODDS_URL = f"{BASE}/liveData/odds/odds_todaysGames.json"

SOURCE = "nba_cdn"

# Game-id prefix rule: first 3 chars of the 10-digit id are the game type. Checked against the real
# 2026-27 schedule, which currently holds only 001, 002 and 006 ids:
#   001 preseason  -> excluded
#   002 regular season, including NBA Cup group play and Cup knockout rounds
#       (gameSubtype "in-season-knockout" still uses 002)  -> included
#   003 All-Star   -> excluded (not present yet; standard NBA convention)
#   004 playoffs   -> included
#   005 play-in    -> included
#   006 NBA Cup championship game (neutral site, tricodes are empty/TBD until
#       the semifinals are decided)  -> included
INCLUDED_PREFIXES = frozenset({"002", "004", "005", "006"})

_STATUS = {1: "scheduled", 2: "live", 3: "final"}


# --------------------------------------------------------------------- fetchers

def fetch_schedule() -> dict:
    return fetch_json(SCHEDULE_URL)


def fetch_scoreboard() -> dict:
    return fetch_json(SCOREBOARD_URL)


def fetch_boxscore(game_id: str) -> dict:
    return fetch_json(BOXSCORE_URL.format(game_id=game_id))


def fetch_playbyplay(game_id: str) -> dict:
    return fetch_json(PLAYBYPLAY_URL.format(game_id=game_id))


def fetch_odds() -> dict:
    return fetch_json(ODDS_URL)


# --------------------------------------------------------------------- helpers

def _parse_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _status(code: Any, text: str | None, postponed_flag: Any = None) -> str:
    t = (text or "")
    if "PPD" in t.upper() or "postponed" in t.lower() or str(postponed_flag or "").upper() == "Y":
        return "postponed"
    return _STATUS.get(code, "scheduled")


def _row(game_id, season, tipoff: datetime | None, home: dict, away: dict,
         status_code, status_text, postponed_flag=None) -> dict | None:
    """Build a db.upsert_game row, or None if a team is TBD / the time is unusable."""
    h, a = home.get("teamTricode"), away.get("teamTricode")
    if not h or not a or tipoff is None:
        return None
    status = _status(status_code, status_text, postponed_flag)
    started = status in ("live", "final")
    return {
        "game_id": game_id,
        "season": season,
        "nba_day": config.nba_day(tipoff).isoformat(),
        "tipoff_utc": tipoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home": h,
        "away": a,
        "status": status,
        "home_score": home.get("score") if started else None,
        "away_score": away.get("score") if started else None,
    }


def _season_from_game_id(game_id: str) -> str | None:
    """'0022600001' -> '2026-27' (digits 3-4 are the season start year)."""
    try:
        y = 2000 + int(game_id[3:5])
    except (ValueError, IndexError):
        return None
    return f"{y}-{(y + 1) % 100:02d}"


# --------------------------------------------------------------------- schedule

def schedule_games(payload: dict) -> list[dict]:
    """Rows for regular season, Cup, play-in and playoff games (see INCLUDED_PREFIXES)."""
    league = (payload or {}).get("leagueSchedule") or {}
    season = league.get("seasonYear")
    out: list[dict] = []
    for day in league.get("gameDates") or []:
        for g in day.get("games") or []:
            gid = str(g.get("gameId") or "")
            if gid[:3] not in INCLUDED_PREFIXES:
                continue
            row = _row(gid, season or _season_from_game_id(gid),
                       _parse_utc(g.get("gameDateTimeUTC")),
                       g.get("homeTeam") or {}, g.get("awayTeam") or {},
                       g.get("gameStatus"), g.get("gameStatusText"), g.get("postponedStatus"))
            if row:
                out.append(row)
    return out


# --------------------------------------------------------------------- scoreboard

def scoreboard_games(payload: dict) -> list[dict]:
    """Rows for today's scoreboard, same shape as schedule_games (same prefix filter)."""
    sb = (payload or {}).get("scoreboard") or {}
    out: list[dict] = []
    for g in sb.get("games") or []:
        gid = str(g.get("gameId") or "")
        if gid[:3] not in INCLUDED_PREFIXES:
            continue
        row = _row(gid, _season_from_game_id(gid), _parse_utc(g.get("gameTimeUTC")),
                   g.get("homeTeam") or {}, g.get("awayTeam") or {},
                   g.get("gameStatus"), g.get("gameStatusText"))
        if row:
            out.append(row)
    return out


# --------------------------------------------------------------------- box score

def is_final(boxscore_payload: dict) -> bool:
    game = (boxscore_payload or {}).get("game") or {}
    return game.get("gameStatus") == 3


# --------------------------------------------------------------------- odds
#
# Payload shape (verified live 2026-10-18):
#   {"games": [{"gameId", "homeTeamId", "awayTeamId", "markets": [
#       {"name": "2way"|"spread"|(totals market, name unverified), "books": [
#           {"name": "FanDuel", "countryCode": "US", "outcomes": [
#               {"type": "home"|"away"(|"over"|"under"), "odds": "1.893",
#                "spread": "-2.5" (spread market), "total": ... (totals market)}]}]}]}]}
# Odds are DECIMAL strings; we convert them to American (moneyline-style) integers.
# Books repeat across countries (same name, several countryCode) and the book list
# differs per market, so we choose one provider per game and read every market from it.

_PREFERRED_BOOKS = ("FanDuel", "DraftKings", "BetMGM", "Caesars", "ESPN BET", "BetRivers")


def decimal_to_american(odds: Any) -> int | None:
    """Decimal odds (1.893) -> American (-112). Values already American (|x|>=100) pass through."""
    try:
        d = float(odds)
    except (TypeError, ValueError):
        return None
    if abs(d) >= 100:
        return int(d)
    if d <= 1.0:
        return None
    return round((d - 1) * 100) if d >= 2.0 else round(-100 / (d - 1))


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _market_kind(name: str | None) -> str | None:
    n = (name or "").lower()
    if "spread" in n or "handicap" in n:
        return "spread"
    if "total" in n or "over" in n:
        return "total"
    if n in ("2way", "moneyline", "money line", "h2h") or "moneyline" in n:
        return "ml"
    return None


def _pick_provider(markets: list[dict]) -> str | None:
    names: list[str] = []
    for m in markets:
        for b in m.get("books") or []:
            if b.get("name") and b["name"] not in names:
                names.append(b["name"])
    for p in _PREFERRED_BOOKS:
        if p in names:
            return p
    # Otherwise the first book of the spread market, else the first book anywhere.
    for m in markets:
        if _market_kind(m.get("name")) == "spread" and m.get("books"):
            return m["books"][0].get("name")
    return names[0] if names else None


def _book(market: dict, provider: str) -> dict | None:
    books = [b for b in market.get("books") or [] if b.get("name") == provider]
    if not books:
        return None
    for b in books:  # same book listed per country: prefer US
        if b.get("countryCode") == "US":
            return b
    return books[0]


def _outcome(book: dict, typ: str) -> dict:
    for o in book.get("outcomes") or []:
        if str(o.get("type", "")).lower() == typ:
            return o
    return {}


def odds_rows(payload: dict) -> list[dict]:
    """One row per game: game_id, source, provider, home_spread, total, home_ml, away_ml.

    Returns [] if the payload has no games / unexpected structure (off days, preseason).
    Missing markets give None fields.
    """
    games = (payload or {}).get("games") if isinstance(payload, dict) else None
    if not isinstance(games, list):
        return []
    rows: list[dict] = []
    for g in games:
        markets = g.get("markets")
        if not g.get("gameId") or not isinstance(markets, list) or not markets:
            continue
        provider = _pick_provider(markets)
        if not provider:
            continue
        row = {"game_id": str(g["gameId"]), "source": SOURCE, "provider": provider,
               "home_spread": None, "total": None, "home_ml": None, "away_ml": None}
        for m in markets:
            kind = _market_kind(m.get("name"))
            book = _book(m, provider) if kind else None
            if not book:
                continue
            if kind == "spread" and row["home_spread"] is None:
                home, away = _outcome(book, "home"), _outcome(book, "away")
                sp = _to_float(home.get("spread"))
                if sp is None and (a := _to_float(away.get("spread"))) is not None:
                    sp = -a
                row["home_spread"] = sp
            elif kind == "total" and row["total"] is None:
                for typ in ("over", "under"):
                    o = _outcome(book, typ)
                    t = _to_float(o.get("total", o.get("spread")))
                    if t is not None:
                        row["total"] = abs(t)
                        break
            elif kind == "ml" and row["home_ml"] is None:
                row["home_ml"] = decimal_to_american(_outcome(book, "home").get("odds"))
                row["away_ml"] = decimal_to_american(_outcome(book, "away").get("odds"))
        rows.append(row)
    return rows
