"""stats.nba.com player / matchup / tracking data. Pure fetch + parse (no archive, no DB).

courtside.backfill_players archives the raw payloads and writes the rows.

Season-level endpoints (one call per season [x season type] [x variant]):
  playerindex            all-time bio (Historical=1; no birthdate)
  commonteamroster       per team-season: roster (birthdate, age, experience) + COACHES
  matchupsrollup         defender x opponent position (G/F/C) with % of time
  leagueseasonmatchups   every offender x defender pair (no player filter; ~25 MB, 90 s timeout)
  leaguedashptdefend     defended FG% vs expected, 4 DefenseCategory values
  leaguedashplayerptshot shooting by closest-defender distance, 4 CloseDefDistRange buckets
  synergyplaytypes       play types, offense/defense x player/team (param SeasonYear, not Season)
  leaguehustlestatsplayer

Per-game endpoints (one call per game id, ~20-200 KB):
  boxscorematchupsv3     who guarded whom
  boxscoreplayertrackv3  touches, distance, contested/uncontested FG, defended at rim
  boxscorehustlev2       contests, deflections, charges, screen assists, box-outs

Per-game matchups orientation (verified against the box score): the top-level player in the
payload is the OFFENSIVE player and each entry of his ``matchups`` is a DEFENDER who guarded him;
the statistics are what the offensive player did against that defender (playerPoints = the
offensive player's points while guarded by him).

Wide result sets are mapped to typed columns; any header not in the map is kept in ``extra_json``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from courtside import teams
from courtside.http import BlockedError, fetch_json
from courtside.sources.nba_stats import result_rows

SOURCE = "nba_stats"
BASE = "https://stats.nba.com/stats"
SEASON_TYPES = ("Regular Season", "Playoffs")

DEF_CATEGORIES = ("Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft")
# CloseDefDistRange API value -> short bucket label stored in the table
PTSHOT_BUCKETS = {"0-2 Feet - Very Tight": "0-2", "2-4 Feet - Tight": "2-4",
                  "4-6 Feet - Open": "4-6", "6+ Feet - Wide Open": "6+"}
PLAY_TYPES = ("Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup",
              "Handoff", "Cut", "OffScreen", "OffRebound", "Misc")
GROUPINGS = ("Offensive", "Defensive")
# Combinations the API answers with an empty result set (verified 2025-26): don't waste calls.
SYNERGY_EMPTY = {("P", "Defensive", pt) for pt in ("Transition", "Cut", "OffRebound", "Misc")}

GAME_KINDS = {  # kind (= endpoint) -> envelope key
    "boxscorematchupsv3": "boxScoreMatchups",
    "boxscoreplayertrackv3": "boxScorePlayerTrack",
    "boxscorehustlev2": "boxScoreHustle",
}

LEAGUE_TEAM_IDS = tuple(range(1610612737, 1610612767))  # the 30 franchises


# --------------------------------------------------------------------- fetchers

def _get(endpoint: str, params: dict, *, timeout: float = 40.0, retries: int = 2) -> dict:
    payload = fetch_json(f"{BASE}/{endpoint}", params=params, timeout=timeout, retries=retries)
    if not isinstance(payload, dict) or not (payload.get("resultSets") or payload.get("resultSet")):
        raise BlockedError(f"unexpected {endpoint} payload (no result sets)")
    return payload


def fetch_player_index(season: str) -> dict:
    return _get("playerindex", {
        "LeagueID": "00", "Season": season, "Active": "", "AllStar": "", "College": "",
        "Country": "", "DraftPick": "", "DraftRound": "", "DraftYear": "", "Height": "",
        "Historical": "1", "Team": "", "Weight": ""})


def fetch_roster(team_id: int, season: str) -> dict:
    return _get("commonteamroster", {"TeamID": team_id, "Season": season, "LeagueID": "00"})


def _season_params(season: str, season_type: str) -> dict:
    return {"LeagueID": "00", "PerMode": "Totals", "SeasonType": season_type, "Season": season}


def fetch_matchups_rollup(season: str, season_type: str) -> dict:
    return _get("matchupsrollup", _season_params(season, season_type))


def fetch_league_season_matchups(season: str, season_type: str) -> dict:
    return _get("leagueseasonmatchups", _season_params(season, season_type), timeout=90.0)


def fetch_pt_defend(season: str, season_type: str, category: str) -> dict:
    return _get("leaguedashptdefend",
                {**_season_params(season, season_type), "DefenseCategory": category})


def fetch_pt_shot(season: str, season_type: str, bucket_range: str) -> dict:
    params = {**_season_params(season, season_type),
              "OpponentTeamID": "0", "TeamID": "0", "LastNGames": "0", "Month": "0",
              "Period": "0", "DateFrom": "", "DateTo": "", "Location": "", "Outcome": "",
              "SeasonSegment": "", "VsConference": "", "VsDivision": "", "GameSegment": "",
              "CloseDefDistRange": bucket_range, "DribbleRange": "", "GeneralRange": "",
              "ShotClockRange": "", "ShotDistRange": "", "TouchTimeRange": "",
              "PlayerExperience": "", "PlayerPosition": "", "StarterBench": "", "College": "",
              "Conference": "", "Country": "", "Division": "", "DraftPick": "", "DraftYear": "",
              "Height": "", "Weight": ""}
    return _get("leaguedashplayerptshot", params)


def fetch_synergy(season: str, season_type: str, entity: str, grouping: str,
                  play_type: str) -> dict:
    if entity not in ("P", "T"):
        raise ValueError("entity must be 'P' or 'T'")
    return _get("synergyplaytypes", {
        "LeagueID": "00", "PerMode": "Totals", "PlayerOrTeam": entity,
        "SeasonType": season_type, "SeasonYear": season, "PlayType": play_type,
        "TypeGrouping": grouping})


def fetch_hustle_season(season: str, season_type: str) -> dict:
    return _get("leaguehustlestatsplayer", _season_params(season, season_type))


def fetch_game(kind: str, game_id: str) -> dict:
    """One per-game boxscore payload. Raises BlockedError for HTML / a foreign payload."""
    payload = fetch_json(f"{BASE}/{kind}", params={"GameID": game_id}, timeout=40.0, retries=2)
    return check_game(payload, kind)


def check_game(payload: Any, kind: str) -> dict:
    root = GAME_KINDS[kind]
    if not isinstance(payload, dict) or not isinstance(payload.get(root), dict):
        raise BlockedError(f"unexpected {kind} payload (no {root})")
    bs = payload[root]
    if "homeTeam" not in bs or "awayTeam" not in bs:
        raise BlockedError(f"unexpected {kind} payload (no teams)")
    return payload


# ---------------------------------------------------------------------- helpers

def minutes(v: Any) -> float | None:
    """'34:21' -> 34.35; 'PT34M21.00S' -> 34.35; numbers pass through; ''/None/garbage -> None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", s)
    if m and any(m.groups()):
        h, mi, sec = m.groups()
        return int(h or 0) * 60 + int(mi or 0) + float(sec or 0) / 60.0
    try:
        if ":" in s:
            mm, _, ss = s.partition(":")
            return int(mm) + int(ss or 0) / 60.0
        return float(s)
    except ValueError:
        return None


def height_inches(h: Any) -> int | None:
    """'6-8' -> 80."""
    m = re.fullmatch(r"\s*(\d+)-(\d+)\s*", str(h or ""))
    return int(m[1]) * 12 + int(m[2]) if m else None


def _int(v: Any) -> int | None:
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _birth_date(v: Any) -> str | None:
    """'DEC 01, 1992' -> '1992-12-01'."""
    if not v:
        return None
    try:
        return datetime.strptime(str(v).strip().title(), "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


def _tricode(team_id: Any) -> str | None:
    try:
        return teams.by_team_id(team_id).tricode
    except (KeyError, TypeError, ValueError):
        return None


# Headers that duplicate a typed column / an id lookup / a ratio of typed columns: not kept.
_DERIVED = frozenset({"SEASON_ID", "MATCHUP_MIN", "MATCHUP_FG_PCT", "MATCHUP_FG3_PCT",
                      "HELP_FG_PERC", "OFF_PLAYER_NAME", "DEF_PLAYER_NAME", "TEAM_NAME"})


def map_row(r: dict, colmap: dict[str, str], *, extra: bool = True) -> dict:
    """Rename headers via colmap; unmapped headers go to a compact JSON ``extra_json``."""
    out = {dst: r.get(src) for src, dst in colmap.items()}
    if extra:
        rest = {k: v for k, v in r.items() if k not in colmap and k not in _DERIVED}
        out["extra_json"] = json.dumps(rest, separators=(",", ":")) if rest else None
    return out


# -------------------------------------------------------------- season-level parsing

def parse_player_index(payload: dict, snapshot_season: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        if r.get("PERSON_ID") is None:
            continue
        out.append({
            "player_id": r["PERSON_ID"], "first_name": r.get("PLAYER_FIRST_NAME"),
            "last_name": r.get("PLAYER_LAST_NAME"), "slug": r.get("PLAYER_SLUG"),
            "team_id": r.get("TEAM_ID") or None, "team": r.get("TEAM_ABBREVIATION") or None,
            "jersey": r.get("JERSEY_NUMBER") or None, "position": r.get("POSITION") or None,
            "height": r.get("HEIGHT") or None, "height_in": height_inches(r.get("HEIGHT")),
            "weight": _int(r.get("WEIGHT")), "college": r.get("COLLEGE") or None,
            "country": r.get("COUNTRY") or None, "draft_year": _int(r.get("DRAFT_YEAR")),
            "draft_round": _int(r.get("DRAFT_ROUND")), "draft_number": _int(r.get("DRAFT_NUMBER")),
            "from_year": _int(r.get("FROM_YEAR")), "to_year": _int(r.get("TO_YEAR")),
            "roster_status": r.get("ROSTER_STATUS"), "is_defunct": r.get("IS_DEFUNCT"),
            "snapshot_season": snapshot_season})
    return out


def parse_roster(payload: dict, season: str) -> tuple[list[dict], list[dict]]:
    """-> (team_rosters rows, team_coaches rows) from one commonteamroster payload."""
    by_name = {s.get("name"): s for s in payload.get("resultSets") or [] if isinstance(s, dict)}

    def rows_of(name: str) -> list[dict]:
        return result_rows({"resultSets": [by_name[name]]}) if name in by_name else []

    roster, coaches = [], []
    for r in rows_of("CommonTeamRoster"):
        exp = r.get("EXP")
        roster.append({
            "season": season, "team_id": r["TeamID"], "team": _tricode(r["TeamID"]),
            "player_id": r["PLAYER_ID"], "player": r.get("PLAYER"), "jersey": r.get("NUM"),
            "position": r.get("POSITION") or None, "height": r.get("HEIGHT") or None,
            "height_in": height_inches(r.get("HEIGHT")), "weight": _int(r.get("WEIGHT")),
            "birth_date": _birth_date(r.get("BIRTH_DATE")), "age": r.get("AGE"),
            "exp": 0 if str(exp).upper() == "R" else _int(exp), "school": r.get("SCHOOL") or None,
            "how_acquired": r.get("HOW_ACQUIRED") or None})
    for r in rows_of("Coaches"):
        coaches.append({
            "season": season, "team_id": r["TEAM_ID"], "team": _tricode(r["TEAM_ID"]),
            "coach_id": r["COACH_ID"], "coach_name": r.get("COACH_NAME"),
            "coach_type": r.get("COACH_TYPE"), "is_assistant": r.get("IS_ASSISTANT"),
            "sort_sequence": r.get("SUB_SORT_SEQUENCE")})
    return roster, coaches


_MATCHUP_COLS = {
    "GP": "gp", "PARTIAL_POSS": "partial_poss", "PLAYER_PTS": "pts", "TEAM_PTS": "team_pts",
    "MATCHUP_AST": "ast", "MATCHUP_TOV": "tov", "MATCHUP_BLK": "blk", "MATCHUP_FGM": "fgm",
    "MATCHUP_FGA": "fga", "MATCHUP_FG3M": "fg3m", "MATCHUP_FG3A": "fg3a",
    "MATCHUP_FTM": "ftm", "MATCHUP_FTA": "fta", "SFL": "sfl"}


def parse_matchups_rollup(payload: dict, season: str, season_type: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        row = map_row(r, {**_MATCHUP_COLS, "DEF_PLAYER_ID": "def_player_id",
                          "DEF_PLAYER_NAME": "def_player_name", "POSITION": "position",
                          "PERCENT_OF_TIME": "pct_of_time"})
        row.update(season=season, season_type=season_type, matchup_min=minutes(r.get("MATCHUP_MIN")))
        out.append(row)
    return out


def parse_season_matchups(payload: dict, season: str, season_type: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        row = map_row(r, {**_MATCHUP_COLS, "OFF_PLAYER_ID": "off_player_id",
                          "DEF_PLAYER_ID": "def_player_id", "MATCHUP_TIME_SEC": "matchup_sec",
                          "HELP_BLK": "help_blk", "HELP_FGM": "help_fgm", "HELP_FGA": "help_fga"})
        row.update(season=season, season_type=season_type, matchup_min=minutes(r.get("MATCHUP_MIN")))
        out.append(row)
    return out


_PTDEF_COLS = {
    "CLOSE_DEF_PERSON_ID": "player_id", "PLAYER_NAME": "player_name",
    "PLAYER_LAST_TEAM_ID": "team_id", "PLAYER_LAST_TEAM_ABBREVIATION": "team",
    "PLAYER_POSITION": "position", "AGE": "age", "GP": "gp", "G": "g", "FREQ": "freq",
    "D_FGM": "d_fgm", "D_FGA": "d_fga", "D_FG_PCT": "d_fg_pct", "NORMAL_FG_PCT": "normal_fg_pct",
    "PCT_PLUSMINUS": "pct_plusminus"}


def parse_pt_defend(payload: dict, season: str, season_type: str, category: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        row = map_row(r, _PTDEF_COLS)
        row.update(season=season, season_type=season_type, category=category)
        out.append(row)
    return out


_PTSHOT_COLS = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "PLAYER_LAST_TEAM_ID": "team_id",
    "PLAYER_LAST_TEAM_ABBREVIATION": "team", "AGE": "age", "GP": "gp", "G": "g",
    "FGA_FREQUENCY": "fga_freq", "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
    "EFG_PCT": "efg_pct", "FG2A_FREQUENCY": "fg2a_freq", "FG2M": "fg2m", "FG2A": "fg2a",
    "FG2_PCT": "fg2_pct", "FG3A_FREQUENCY": "fg3a_freq", "FG3M": "fg3m", "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct"}


def parse_pt_shot(payload: dict, season: str, season_type: str, bucket: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        row = map_row(r, _PTSHOT_COLS)
        row.update(season=season, season_type=season_type, def_dist=bucket)
        out.append(row)
    return out


_SYN_COLS = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team", "PLAY_TYPE": "play_type", "TYPE_GROUPING": "grouping",
    "PERCENTILE": "percentile", "GP": "gp", "POSS_PCT": "poss_pct", "PPP": "ppp",
    "FG_PCT": "fg_pct", "FT_POSS_PCT": "ft_poss_pct", "TOV_POSS_PCT": "tov_poss_pct",
    "SF_POSS_PCT": "sf_poss_pct", "PLUSONE_POSS_PCT": "plusone_poss_pct",
    "SCORE_POSS_PCT": "score_poss_pct", "EFG_PCT": "efg_pct", "POSS": "poss", "PTS": "pts",
    "FGM": "fgm", "FGA": "fga", "FGMX": "fgmx"}


def parse_synergy(payload: dict, season: str, season_type: str, entity: str,
                  grouping: str, play_type: str) -> list[dict]:
    """player_id is 0 for team rows (entity 'T') so it can sit in the primary key."""
    out = []
    for r in result_rows(payload):
        row = map_row(r, _SYN_COLS)
        row.update(season=season, season_type=season_type, entity=entity,
                   player_id=row["player_id"] or 0, team=row["team"],
                   play_type=row["play_type"] or play_type, grouping=row["grouping"] or grouping)
        out.append(row)
    return out


_HUSTLE_SEASON_COLS = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team", "AGE": "age", "G": "gp", "MIN": "min",
    "CONTESTED_SHOTS": "contested_shots", "CONTESTED_SHOTS_2PT": "contested_2pt",
    "CONTESTED_SHOTS_3PT": "contested_3pt", "DEFLECTIONS": "deflections",
    "CHARGES_DRAWN": "charges_drawn", "SCREEN_ASSISTS": "screen_assists",
    "SCREEN_AST_PTS": "screen_ast_pts", "OFF_LOOSE_BALLS_RECOVERED": "loose_off",
    "DEF_LOOSE_BALLS_RECOVERED": "loose_def", "LOOSE_BALLS_RECOVERED": "loose_total",
    "OFF_BOXOUTS": "boxouts_off", "DEF_BOXOUTS": "boxouts_def",
    "BOX_OUT_PLAYER_TEAM_REBS": "boxout_team_rebs", "BOX_OUT_PLAYER_REBS": "boxout_player_rebs",
    "BOX_OUTS": "box_outs"}


def parse_hustle_season(payload: dict, season: str, season_type: str) -> list[dict]:
    out = []
    for r in result_rows(payload):
        row = map_row(r, _HUSTLE_SEASON_COLS)
        row.update(season=season, season_type=season_type)
        out.append(row)
    return out


# ------------------------------------------------------------------ per-game parsing

def _sides(bs: dict):
    """Yield (team block, opponent block) for home then away."""
    yield bs["homeTeam"], bs["awayTeam"]
    yield bs["awayTeam"], bs["homeTeam"]


def parse_game_matchups(payload: dict) -> list[dict]:
    """player_matchups rows: off_player_id is the top-level player, def_player_id his defender."""
    bs = payload["boxScoreMatchups"]
    game_id = bs["gameId"]
    out = []
    for side, other in _sides(bs):
        off_team, def_team = side.get("teamTricode"), other.get("teamTricode")
        for p in side.get("players") or []:
            for m in p.get("matchups") or []:
                s = m.get("statistics") or {}
                mm = (s.get("matchupMinutesSort") / 60.0 if s.get("matchupMinutesSort") is not None
                      else minutes(s.get("matchupMinutes")))
                out.append({
                    "game_id": game_id, "off_player_id": p["personId"],
                    "def_player_id": m["personId"], "off_team": off_team, "def_team": def_team,
                    "matchup_min": mm, "partial_poss": s.get("partialPossessions"),
                    "pct_def_time": s.get("percentageDefenderTotalTime"),
                    "pct_off_time": s.get("percentageOffensiveTotalTime"),
                    "pct_both_time": s.get("percentageTotalTimeBothOn"),
                    "switches": s.get("switchesOn"), "pts": s.get("playerPoints"),
                    "team_pts": s.get("teamPoints"), "ast": s.get("matchupAssists"),
                    "pot_ast": s.get("matchupPotentialAssists"), "tov": s.get("matchupTurnovers"),
                    "blk": s.get("matchupBlocks"), "fgm": s.get("matchupFieldGoalsMade"),
                    "fga": s.get("matchupFieldGoalsAttempted"),
                    "fg3m": s.get("matchupThreePointersMade"),
                    "fg3a": s.get("matchupThreePointersAttempted"),
                    "help_blk": s.get("helpBlocks"), "help_fgm": s.get("helpFieldGoalsMade"),
                    "help_fga": s.get("helpFieldGoalsAttempted"),
                    "ftm": s.get("matchupFreeThrowsMade"), "fta": s.get("matchupFreeThrowsAttempted"),
                    "sfl": s.get("shootingFouls")})
    return out


def _player_rows(bs: dict, fields: dict[str, str]) -> list[dict]:
    """Shared per-player extractor: players that never took the floor (0:00) are skipped."""
    game_id = bs["gameId"]
    out = []
    for side, _ in _sides(bs):
        for p in side.get("players") or []:
            pid = p.get("personId")
            s = p.get("statistics") or {}
            mins = minutes(s.get("minutes"))
            if not pid or not mins:
                continue
            row = {"game_id": game_id, "player_id": pid, "team": side.get("teamTricode"),
                   "minutes": mins}
            row.update({dst: s.get(src) for src, dst in fields.items()})
            out.append(row)
    return out


_TRACK_FIELDS = {
    "speed": "speed", "distance": "distance", "reboundChancesOffensive": "reb_chances_off",
    "reboundChancesDefensive": "reb_chances_def", "reboundChancesTotal": "reb_chances_total",
    "touches": "touches", "secondaryAssists": "secondary_ast", "freeThrowAssists": "ft_ast",
    "passes": "passes", "assists": "ast", "contestedFieldGoalsMade": "cont_fgm",
    "contestedFieldGoalsAttempted": "cont_fga", "uncontestedFieldGoalsMade": "uncont_fgm",
    "uncontestedFieldGoalsAttempted": "uncont_fga", "defendedAtRimFieldGoalsMade": "def_rim_fgm",
    "defendedAtRimFieldGoalsAttempted": "def_rim_fga"}

_HUSTLE_FIELDS = {
    "points": "points", "contestedShots": "contested_shots", "contestedShots2pt": "contested_2pt",
    "contestedShots3pt": "contested_3pt", "deflections": "deflections",
    "chargesDrawn": "charges_drawn", "screenAssists": "screen_assists",
    "screenAssistPoints": "screen_ast_pts", "looseBallsRecoveredOffensive": "loose_off",
    "looseBallsRecoveredDefensive": "loose_def", "looseBallsRecoveredTotal": "loose_total",
    "offensiveBoxOuts": "boxouts_off", "defensiveBoxOuts": "boxouts_def",
    "boxOutPlayerTeamRebounds": "boxout_team_rebs", "boxOutPlayerRebounds": "boxout_player_rebs",
    "boxOuts": "box_outs"}


def parse_game_tracking(payload: dict) -> list[dict]:
    return _player_rows(payload["boxScorePlayerTrack"], _TRACK_FIELDS)


def parse_game_hustle(payload: dict) -> list[dict]:
    return _player_rows(payload["boxScoreHustle"], _HUSTLE_FIELDS)


GAME_PARSERS = {
    "boxscorematchupsv3": ("player_matchups", parse_game_matchups),
    "boxscoreplayertrackv3": ("player_tracking", parse_game_tracking),
    "boxscorehustlev2": ("player_hustle", parse_game_hustle),
}


def game_has_data(payload: dict, kind: str) -> bool:
    """False for the empty shell the API returns when it has no data for the game."""
    return bool(GAME_PARSERS[kind][1](payload))
