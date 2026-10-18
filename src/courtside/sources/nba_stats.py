"""stats.nba.com source: historical team/player game logs (basic + advanced).

Pure fetch + parse. No archiving and no DB access; courtside.backfill does both.

Endpoints (all verified 2025-26, one call per season/type/entity):
  leaguegamelog  PlayerOrTeam=T|P   basic box score per team/player game
  teamgamelogs   MeasureType=Advanced   per-team-game advanced (ratings, pace, four factors)
  playergamelogs MeasureType=Advanced   per-player-game advanced (usage, ratings)
The advanced endpoints are optional: backfill can skip them (include_advanced=False).
"""

from __future__ import annotations

from typing import Any

from courtside.http import BlockedError, fetch_json

SOURCE = "nba_stats"
BASE = "https://stats.nba.com/stats"
LEAGUEGAMELOG_URL = f"{BASE}/leaguegamelog"
TEAMGAMELOGS_URL = f"{BASE}/teamgamelogs"
PLAYERGAMELOGS_URL = f"{BASE}/playergamelogs"

SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")

# Franchises that changed tricode: stored as returned by the API for that season
# (e.g. NJN/NOH/CHA in 2010-11); no remapping is done here.


# --------------------------------------------------------------------- fetchers

def fetch_league_gamelog(season: str, season_type: str, player_or_team: str) -> dict:
    if player_or_team not in ("T", "P"):
        raise ValueError("player_or_team must be 'T' or 'P'")
    return _checked(fetch_json(LEAGUEGAMELOG_URL, params={
        "Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": player_or_team,
        "Season": season, "SeasonType": season_type, "Sorter": "DATE"}, timeout=90))


def fetch_gamelogs_advanced(season: str, season_type: str, entity: str) -> dict:
    if entity not in ("team", "player"):
        raise ValueError("entity must be 'team' or 'player'")
    url = TEAMGAMELOGS_URL if entity == "team" else PLAYERGAMELOGS_URL
    return _checked(fetch_json(url, params={
        "MeasureType": "Advanced", "PerMode": "Totals", "LeagueID": "00",
        "Season": season, "SeasonType": season_type}, timeout=120))


def _checked(payload: Any) -> dict:
    """Reject payloads that are not a stats.nba.com result-set envelope."""
    if not isinstance(payload, dict) or not (payload.get("resultSets") or payload.get("resultSet")):
        raise BlockedError("unexpected stats.nba.com payload (no result sets)")
    return payload


# ---------------------------------------------------------------------- parsing

def result_rows(payload: dict, set_index: int = 0) -> list[dict]:
    """Zip headers with rowSet for one result set (handles resultSet and resultSets)."""
    sets = payload.get("resultSets")
    if sets is None:
        sets = payload.get("resultSet")
    if isinstance(sets, dict):  # single resultSet object
        sets = [sets]
    if not sets:
        return []
    rs = sets[set_index]
    headers = rs["headers"]
    return [dict(zip(headers, row)) for row in rs["rowSet"]]


def parse_minutes(v: Any) -> float | None:
    """MIN as float minutes; accepts numbers, 'mm:ss', or None/''."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    if ":" in s:
        m, _, sec = s.partition(":")
        return int(m) + int(sec or 0) / 60.0
    return float(s)


def parse_matchup(matchup: str | None) -> tuple[int | None, str | None]:
    """'OKC vs. HOU' -> (1, 'HOU'); 'GSW @ LAL' -> (0, 'LAL')."""
    if not matchup:
        return None, None
    if " vs. " in matchup:
        return 1, matchup.split(" vs. ", 1)[1].strip()
    if " @ " in matchup:
        return 0, matchup.split(" @ ", 1)[1].strip()
    return None, None


def _date(v: str | None) -> str | None:
    return v[:10] if v else None


_BOX = ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
        "blk", "tov", "pf", "pts", "plus_minus")


def _box(r: dict) -> dict:
    return {c: r.get(c.upper()) for c in _BOX}


def team_game_rows(payload: dict, season: str, season_type: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        is_home, opp = parse_matchup(r.get("MATCHUP"))
        out.append({
            "game_id": r["GAME_ID"], "team": r["TEAM_ABBREVIATION"], "season": season,
            "season_type": season_type, "game_date": _date(r.get("GAME_DATE")),
            "opp": opp, "is_home": is_home, "wl": r.get("WL"),
            "min": parse_minutes(r.get("MIN")), **_box(r)})
    return out


def player_game_rows(payload: dict, season: str, season_type: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        out.append({
            "game_id": r["GAME_ID"], "player_id": r["PLAYER_ID"],
            "player_name": r.get("PLAYER_NAME"), "team": r["TEAM_ABBREVIATION"],
            "season": season, "season_type": season_type,
            "game_date": _date(r.get("GAME_DATE")),
            "min": parse_minutes(r.get("MIN")), **_box(r)})
    return out


def _pick(r: dict, *names: str) -> Any:
    """First non-None value among candidate header names (non-estimated ones listed first)."""
    for n in names:
        if r.get(n) is not None:
            return r[n]
    return None


def team_adv_rows(payload: dict) -> list[dict]:
    out = []
    for r in result_rows(payload):
        out.append({
            "game_id": r["GAME_ID"], "team": r["TEAM_ABBREVIATION"],
            "off_rating": _pick(r, "OFF_RATING", "E_OFF_RATING"),
            "def_rating": _pick(r, "DEF_RATING", "E_DEF_RATING"),
            "net_rating": _pick(r, "NET_RATING", "E_NET_RATING"),
            "pace": _pick(r, "PACE", "E_PACE"), "poss": r.get("POSS"),
            "efg_pct": r.get("EFG_PCT"), "ts_pct": r.get("TS_PCT"),
            "tov_pct": _pick(r, "TM_TOV_PCT", "E_TOV_PCT"),
            "oreb_pct": r.get("OREB_PCT"), "dreb_pct": r.get("DREB_PCT"),
            "ast_pct": r.get("AST_PCT"), "pie": r.get("PIE")})
    return out


def player_adv_rows(payload: dict) -> list[dict]:
    out = []
    for r in result_rows(payload):
        out.append({
            "game_id": r["GAME_ID"], "player_id": r["PLAYER_ID"],
            "off_rating": _pick(r, "OFF_RATING", "E_OFF_RATING"),
            "def_rating": _pick(r, "DEF_RATING", "E_DEF_RATING"),
            "net_rating": _pick(r, "NET_RATING", "E_NET_RATING"),
            "usg_pct": _pick(r, "USG_PCT", "E_USG_PCT"),
            "ts_pct": r.get("TS_PCT"), "efg_pct": r.get("EFG_PCT"),
            "ast_pct": r.get("AST_PCT"), "reb_pct": r.get("REB_PCT"),
            "pace": _pick(r, "PACE", "E_PACE"), "poss": r.get("POSS"), "pie": r.get("PIE")})
    return out
