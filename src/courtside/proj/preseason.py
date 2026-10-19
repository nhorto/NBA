"""Preseason role evidence (T1.9): collect preseason boxscores and extract features.

Preseason starts, minutes in the last 2 preseason games, used for the first 10 regular-season
games.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from courtside import archive, config, db, http
from courtside.http import FetchError, fetch_json
from courtside.sources import nba_cdn


SOURCE = nba_cdn.SOURCE
PRESEASON_PREFIX = "001"


def _parse_duration(duration_str: str | None) -> float | None:
    """Parse ISO 8601 duration like 'PT24M30.00S' to minutes.

    Returns the duration in minutes, or None if parsing fails.
    """
    if not duration_str:
        return None

    try:
        # Pattern: PT[hours]H[minutes]M[seconds]S
        # Examples: PT24M30.00S, PT1H24M30.00S
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", duration_str)
        if not match:
            return None

        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = float(match.group(3)) if match.group(3) else 0

        return hours * 60 + minutes + seconds / 60
    except (ValueError, AttributeError):
        return None


def _load_raw_schedule(season: str = "2026-27") -> dict:
    """Load the schedule from the latest archived file, or None if not found."""
    # Find the most recent schedule file
    schedule_dir = config.data_dir() / "raw" / SOURCE / "schedule"
    if not schedule_dir.exists():
        return {}

    # Get all schedule files, sorted by date descending
    schedule_files = sorted(schedule_dir.glob("*/*.json.gz"), reverse=True)
    for path in schedule_files:
        try:
            with gzip.open(path) as f:
                payload = json.load(f)
                league = payload.get("leagueSchedule", {})
                # seasonYear can be an int or str like "2026-27"
                season_year = league.get("seasonYear")
                if str(season_year) == season:
                    return payload
        except (json.JSONDecodeError, OSError):
            continue

    return {}


def _get_preseason_games(payload: dict) -> list[dict]:
    """Extract preseason game info from schedule payload.

    Returns list of dicts with: game_id, game_date (nba_day), home, away, status.
    """
    games = []
    league = payload.get("leagueSchedule", {})

    for day in league.get("gameDates", []):
        for g in day.get("games", []):
            game_id = str(g.get("gameId", ""))
            if not game_id.startswith(PRESEASON_PREFIX):
                continue

            # Parse the game
            tipoff = nba_cdn._parse_utc(g.get("gameDateTimeUTC"))
            home = g.get("homeTeam", {}).get("teamTricode")
            away = g.get("awayTeam", {}).get("teamTricode")
            status = g.get("gameStatus")

            if tipoff and home and away:
                games.append({
                    "game_id": game_id,
                    "game_date": config.nba_day(tipoff).isoformat(),
                    "home": home,
                    "away": away,
                    "status": status,  # 1=scheduled, 2=live, 3=final
                })

    return games


def _is_final_boxscore(boxscore: dict) -> bool:
    """Check if a boxscore is final and valid."""
    game = boxscore.get("game", {})
    if game.get("gameStatus") != 3:
        return False

    # Validate both teams have players
    home_players = game.get("homeTeam", {}).get("players", [])
    away_players = game.get("awayTeam", {}).get("players", [])
    return bool(home_players) and bool(away_players)


def _parse_boxscore_players(boxscore: dict) -> list[dict]:
    """Extract player stats from a boxscore.

    Returns list of dicts: game_id, game_date, team, player_id, starter, minutes.
    """
    players = []
    game = boxscore.get("game", {})
    game_id = game.get("gameId")

    if not game_id:
        return []

    # Infer game_date and season from game_id
    try:
        year = 2000 + int(str(game_id)[3:5])
        season = f"{year}-{(year + 1) % 100:02d}"
        # We don't have the exact date from the boxscore, but it will come from the schedule
    except (ValueError, IndexError):
        return []

    for team_key in ["homeTeam", "awayTeam"]:
        team_data = game.get(team_key, {})
        team = team_data.get("teamTricode")
        if not team:
            continue

        for player in team_data.get("players", []):
            player_id = player.get("personId")
            if not player_id:
                continue

            starter = player.get("starter")  # "1" or "0" as string/int
            minutes_str = player.get("statistics", {}).get("minutes")
            minutes = _parse_duration(minutes_str)

            players.append({
                "game_id": str(game_id),
                "team": team,
                "player_id": player_id,
                "starter": int(starter) if starter in (0, 1, "0", "1") else 0,
                "minutes": minutes,
            })

    return players


def collect_preseason(conn=None, season: str = "2026-27") -> dict:
    """Collect preseason boxscores and return summary.

    Fetches each finished preseason game's boxscore (if not already archived),
    saves to nba_cdn archive, and returns a summary of games found/fetched.

    Returns: {
        "season": "2026-27",
        "games_found": 67,
        "games_fetched": 5,
        "games_cached": 62,
        "games_missing": 0,
        "games_failed": 0,
        "errors": [],
    }
    """
    own = conn is None
    conn = conn or db.connect()

    try:
        summary = {
            "season": season,
            "games_found": 0,
            "games_fetched": 0,
            "games_cached": 0,
            "games_missing": 0,
            "games_failed": 0,
            "errors": [],
        }

        # Load the schedule
        schedule = _load_raw_schedule(season)
        games = _get_preseason_games(schedule)
        summary["games_found"] = len(games)

        if not games:
            print(f"No preseason games found for {season}")
            return summary

        # Index archived boxscores
        archived = {}
        archive_dir = config.data_dir() / "raw" / SOURCE / "boxscore"
        if archive_dir.exists():
            for path in archive_dir.glob("*/*.json.gz"):
                game_id = path.name[: -len(".json.gz")]
                archived[game_id] = path

        # Fetch final games that aren't archived yet
        games_by_date = {}
        for g in games:
            if g["status"] == 3:  # final
                game_id = g["game_id"]
                game_date = g["game_date"]

                if game_id in archived:
                    summary["games_cached"] += 1
                else:
                    if game_date not in games_by_date:
                        games_by_date[game_date] = []
                    games_by_date[game_date].append(game_id)

        # Fetch missing boxscores with polite throttling
        http.MIN_INTERVAL["cdn.nba.com"] = 0.5
        consecutive_fails = 0

        for date in sorted(games_by_date.keys()):
            for game_id in games_by_date[date]:
                try:
                    boxscore = nba_cdn.fetch_boxscore(game_id)

                    if not _is_final_boxscore(boxscore):
                        summary["games_failed"] += 1
                        summary["errors"].append(f"{game_id}: not final")
                        consecutive_fails += 1
                        continue

                    # Archive it
                    archive.save_raw(SOURCE, "boxscore", date, game_id, boxscore)
                    summary["games_fetched"] += 1
                    consecutive_fails = 0
                    time.sleep(0.5)  # Polite throttling

                except FetchError as e:
                    if e.status in (403, 404):
                        summary["games_missing"] += 1
                    else:
                        summary["games_failed"] += 1
                        summary["errors"].append(f"{game_id}: {e}")
                        consecutive_fails += 1
                except Exception as e:
                    summary["games_failed"] += 1
                    summary["errors"].append(f"{game_id}: {type(e).__name__}: {e}")
                    consecutive_fails += 1

                if consecutive_fails >= 10:
                    summary["errors"].append("Aborting: too many consecutive failures")
                    break

        return summary

    finally:
        if own:
            conn.close()


def preseason_features(conn=None, season: str = "2026-27") -> pd.DataFrame:
    """Extract preseason features for all players.

    Returns DataFrame with columns:
        player_id, team, pre_games, pre_starts, pre_starts_last2,
        pre_min_last2, pre_min_avg, pre_flag

    pre_flag = 1 if player has >= 1 preseason start, 0 otherwise.
    """
    own = conn is None
    conn = conn or db.connect()

    try:
        # Load schedule and games
        schedule = _load_raw_schedule(season)
        games = _get_preseason_games(schedule)

        if not games:
            return pd.DataFrame(columns=[
                "player_id", "team", "pre_games", "pre_starts",
                "pre_starts_last2", "pre_min_last2", "pre_min_avg", "pre_flag"
            ])

        # Sort games by date to find the last 2
        games.sort(key=lambda g: g["game_date"])

        # Group by team to find last 2 games per team
        games_by_team = {}
        for g in games:
            if g["status"] == 3:  # final
                for team in [g["home"], g["away"]]:
                    if team not in games_by_team:
                        games_by_team[team] = []
                    games_by_team[team].append(g)

        # Get last 2 game ids per team
        last2_ids = set()
        for team, team_games in games_by_team.items():
            for g in team_games[-2:]:
                last2_ids.add(g["game_id"])

        # Load all player data from archived boxscores
        archive_dir = config.data_dir() / "raw" / SOURCE / "boxscore"
        all_players = []

        if archive_dir.exists():
            for path in archive_dir.glob("*/*.json.gz"):
                game_id = path.name[: -len(".json.gz")]
                if not game_id.startswith(PRESEASON_PREFIX):
                    continue

                try:
                    with gzip.open(path) as f:
                        boxscore = json.load(f)

                    players = _parse_boxscore_players(boxscore)
                    for p in players:
                        p["game_id"] = game_id
                        p["is_last2"] = game_id in last2_ids
                        all_players.append(p)
                except (json.JSONDecodeError, OSError):
                    continue

        if not all_players:
            return pd.DataFrame(columns=[
                "player_id", "team", "pre_games", "pre_starts",
                "pre_starts_last2", "pre_min_last2", "pre_min_avg", "pre_flag"
            ])

        # Build the features dataframe
        df = pd.DataFrame(all_players)

        # Group by player and team
        features = []
        for (player_id, team), group in df.groupby(["player_id", "team"]):
            # Count games and starts
            pre_games = len(group)
            pre_starts = group["starter"].sum()

            # Last 2 stats
            last2 = group[group["is_last2"]]
            pre_starts_last2 = last2["starter"].sum() if len(last2) > 0 else 0
            pre_min_last2 = last2["minutes"].sum() if len(last2) > 0 else 0

            # Averages
            pre_min_avg = group["minutes"].mean() if pre_games > 0 else 0

            # Flag: 1 if >= 1 start
            pre_flag = 1 if pre_starts >= 1 else 0

            features.append({
                "player_id": player_id,
                "team": team,
                "pre_games": pre_games,
                "pre_starts": int(pre_starts),
                "pre_starts_last2": int(pre_starts_last2),
                "pre_min_last2": pre_min_last2,
                "pre_min_avg": pre_min_avg,
                "pre_flag": pre_flag,
            })

        result = pd.DataFrame(features)
        return result.sort_values(["team", "player_id"]).reset_index(drop=True)

    finally:
        if own:
            conn.close()


def save_features(features: pd.DataFrame, season: str = "2026-27") -> Path:
    """Save features to a CSV file under var/proj/."""
    proj_dir = config.data_dir() / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    path = proj_dir / f"preseason_{season}.csv"
    features.to_csv(path, index=False)
    return path
