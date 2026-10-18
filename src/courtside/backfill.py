"""Historical backfills. Resumable: every raw response is archived before parsing."""

from __future__ import annotations

import re
import time
from typing import Callable

from . import archive, db
from .sources import nba_stats

SOURCE = nba_stats.SOURCE
DEFAULT_SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")


def seasons_range(start: str, end: str) -> list[str]:
    """seasons_range('2010-11', '2012-13') -> ['2010-11', '2011-12', '2012-13']."""
    def year(s: str) -> int:
        m = re.fullmatch(r"(\d{4})-(\d{2})", s)
        if not m or (int(m[1]) + 1) % 100 != int(m[2]):
            raise ValueError(f"invalid season {s!r}, expected e.g. '2025-26'")
        return int(m[1])
    a, b = year(start), year(end)
    if a > b:
        raise ValueError(f"start season {start} is after end season {end}")
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(a, b + 1)]


def archive_day(season: str) -> str:
    """Fixed pseudo nba_day used for archive paths: Oct 1 of the season's start year.

    Backfill payloads cover a whole season, so they are filed under one stable day per season
    (raw/nba_stats/<kind>/<start-year>-10-01/<season>_<type>.json.gz).
    """
    return f"{season[:4]}-10-01"


def _slug(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def _log(msg: str) -> None:
    print(msg, flush=True)


# kind -> (fetch(season, type), parse(payload, season, type) -> rows, table, optional)
def _jobs() -> list[tuple[str, Callable, Callable, str, bool]]:
    ns = nba_stats
    return [
        ("team_gamelog", lambda s, t: ns.fetch_league_gamelog(s, t, "T"),
         ns.team_game_rows, "team_games", False),
        ("player_gamelog", lambda s, t: ns.fetch_league_gamelog(s, t, "P"),
         ns.player_game_rows, "player_games", False),
        ("team_adv", lambda s, t: ns.fetch_gamelogs_advanced(s, t, "team"),
         lambda p, s, t: ns.team_adv_rows(p), "team_games_adv", True),
        ("player_adv", lambda s, t: ns.fetch_gamelogs_advanced(s, t, "player"),
         lambda p, s, t: ns.player_adv_rows(p), "player_games_adv", True),
    ]


def backfill_gamelogs(seasons: list[str], season_types=DEFAULT_SEASON_TYPES,
                      include_advanced: bool = True, force: bool = False, conn=None) -> dict:
    """Fetch (or reuse archived) game logs per season/type/entity and upsert into SQLite.

    Already-archived responses are not re-fetched (unless force) but are re-parsed and upserted,
    so the DB can be rebuilt from var/raw. Failures are logged and counted; the run continues.
    Returns {"fetched", "cached", "failed", "rows": {table: n}, "errors": [...], "run_id"}.
    """
    own = conn is None
    conn = conn or db.connect()
    run_id = db.start_run(conn, "backfill_gamelogs")
    summary: dict = {"fetched": 0, "cached": 0, "failed": 0, "rows": {}, "errors": [],
                     "run_id": run_id}
    try:
        for season in seasons:
            t0 = time.monotonic()
            for stype in season_types:
                for kind, fetch, parse, table, optional in _jobs():
                    if optional and not include_advanced:
                        continue
                    key = f"{season}_{_slug(stype)}"
                    day = archive_day(season)
                    label = f"{season} {stype} {kind}"
                    try:
                        path = archive.raw_path(SOURCE, kind, day, key)
                        if path.exists() and not force:
                            payload = archive.load_raw(path)
                            summary["cached"] += 1
                            how = "cached"
                        else:
                            payload = fetch(season, stype)
                            archive.save_raw(SOURCE, kind, day, key, payload)
                            summary["fetched"] += 1
                            how = "fetched"
                        n = db.upsert_rows(conn, table, parse(payload, season, stype))
                        summary["rows"][table] = summary["rows"].get(table, 0) + n
                        _log(f"  {label}: {how}, {n} rows")
                    except Exception as e:  # keep going; a rerun resumes the failures
                        summary["failed"] += 1
                        summary["errors"].append(f"{label}: {type(e).__name__}: {e}")
                        _log(f"  {label}: FAILED {type(e).__name__}: {e}")
            _log(f"{season} done in {time.monotonic() - t0:.1f}s")
        ok = summary["failed"] == 0
        db.finish_run(conn, run_id, ok, _detail(summary))
    except BaseException as e:
        db.finish_run(conn, run_id, False, f"aborted: {type(e).__name__}: {e}")
        raise
    finally:
        if own:
            conn.close()
    return summary


def _detail(summary: dict) -> str:
    return (f"fetched={summary['fetched']} cached={summary['cached']} "
            f"failed={summary['failed']} rows={summary['rows']}"
            + (f" errors={summary['errors'][:5]}" if summary["errors"] else ""))
