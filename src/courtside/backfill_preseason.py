"""Historical preseason game logs (stats.nba.com leaguegamelog, SeasonType='Pre Season').

For preseason minutes features (item #2). One request per season and entity (T, P), archived
under raw/nba_stats/{team,player}_gamelog/<start-year>-10-01/<season>_pre_season.json.gz
(skipped when archived unless force; archived payloads are always re-parsed and upserted).

Rows go to preseason_team_games / preseason_player_games (migration 15), NEVER to
team_games / player_games: many readers of those have no season_type filter.

Flags:
  scrimmage   game played on/after Jan 1 of the season's second year. In practice only the
              July 2020 bubble scrimmages (game ids 00119001xx, filed under 2019-20). Normal
              preseasons, including the Dec 2011 lockout and Dec 2020 ones, end before Jan 1.
  vs_non_nba  either side is not an NBA franchise (international exhibition teams).
The logs carry no starter flag; MIN is whole minutes.
"""

from __future__ import annotations

import time

from . import archive, db
from .backfill import _detail, _log, _slug, archive_day, seasons_range
from .sources import nba_stats

SOURCE = nba_stats.SOURCE
SEASON_TYPE = nba_stats.PRESEASON
DEFAULT_FROM, DEFAULT_TO = "2010-11", "2026-27"
TEAM_TABLE, PLAYER_TABLE = "preseason_team_games", "preseason_player_games"


def is_scrimmage(season: str, game_date: str | None) -> bool:
    """True for games after the season's preseason window (e.g. the 2020 bubble scrimmages)."""
    return bool(game_date) and game_date >= f"{int(season[:4]) + 1}-01-01"


def non_nba_games(payload: dict) -> set[str]:
    """Game ids in a leaguegamelog payload with a row whose TEAM_ID is not an NBA franchise."""
    return {r["GAME_ID"] for r in nba_stats.result_rows(payload)
            if not nba_stats.is_nba_team_id(r.get("TEAM_ID"))}


def _flag(rows: list[dict], season: str, non_nba: set[str]) -> list[dict]:
    for r in rows:
        r["scrimmage"] = int(is_scrimmage(season, r["game_date"]))
        r["vs_non_nba"] = int(r["game_id"] in non_nba)
    return rows


def team_rows(payload: dict, season: str, non_nba: set[str] | None = None) -> list[dict]:
    games = non_nba_games(payload) | (non_nba or set())
    return _flag(nba_stats.team_game_rows(payload, season, SEASON_TYPE), season, games)


def player_rows(payload: dict, season: str, non_nba: set[str] | None = None) -> list[dict]:
    """non_nba: extra game ids from the team payload (a foreign side may list no players)."""
    games = non_nba_games(payload) | (non_nba or set())
    return _flag(nba_stats.player_game_rows(payload, season, SEASON_TYPE), season, games)


def _payload(kind: str, pot: str, season: str, force: bool, summary: dict) -> tuple[dict, str]:
    key, day = f"{season}_{_slug(SEASON_TYPE)}", archive_day(season)
    path = archive.raw_path(SOURCE, kind, day, key)
    if path.exists() and not force:
        summary["cached"] += 1
        return archive.load_raw(path), "cached"
    payload = nba_stats.fetch_league_gamelog(season, SEASON_TYPE, pot)
    archive.save_raw(SOURCE, kind, day, key, payload)
    summary["fetched"] += 1
    return payload, "fetched"


def backfill_preseason(seasons: list[str], force: bool = False, conn=None) -> dict:
    """Fetch (or reuse archived) preseason team + player logs per season; upsert the new tables.

    Failures are logged and counted; the run continues. Returns {"fetched", "cached", "failed",
    "rows": {table: n}, "by_season": {season: {...}}, "errors", "run_id"}.
    """
    own = conn is None
    conn = conn or db.connect()
    run_id = db.start_run(conn, "backfill_preseason")
    summary: dict = {"fetched": 0, "cached": 0, "failed": 0, "rows": {}, "by_season": {},
                     "errors": [], "run_id": run_id}
    try:
        for season in seasons:
            t0 = time.monotonic()
            non_nba: set[str] = set()
            for kind, pot, table, parse in (("team_gamelog", "T", TEAM_TABLE, team_rows),
                                            ("player_gamelog", "P", PLAYER_TABLE, player_rows)):
                label = f"{season} {SEASON_TYPE} {kind}"
                try:
                    payload, how = _payload(kind, pot, season, force, summary)
                    rows = parse(payload, season, non_nba)
                    if pot == "T":
                        non_nba = {r["game_id"] for r in rows if r["vs_non_nba"]}
                    n = db.upsert_rows(conn, table, rows)
                    summary["rows"][table] = summary["rows"].get(table, 0) + n
                    summary["by_season"].setdefault(season, {})[table] = n
                    _log(f"  {label}: {how}, {n} rows")
                except Exception as e:  # keep going; a rerun resumes the failures
                    summary["failed"] += 1
                    summary["errors"].append(f"{label}: {type(e).__name__}: {e}")
                    _log(f"  {label}: FAILED {type(e).__name__}: {e}")
            _log(f"{season} done in {time.monotonic() - t0:.1f}s")
        db.finish_run(conn, run_id, summary["failed"] == 0, _detail(summary))
    except BaseException as e:
        db.finish_run(conn, run_id, False, f"aborted: {type(e).__name__}: {e}")
        raise
    finally:
        if own:
            conn.close()
    return summary


def run(start: str = DEFAULT_FROM, end: str = DEFAULT_TO, force: bool = False) -> dict:
    return backfill_preseason(seasons_range(start, end), force=force)
