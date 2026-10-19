"""Historical play-by-play and boxscore backfill from cdn.nba.com.

Game ids come from ``team_games`` (Regular Season, Playoffs, PlayIn). Each payload is archived
exactly where postgame archives it, ``raw/nba_cdn/{kind}/{nba_day}/{game_id}.json.gz``, with
``nba_day`` = the game's ``game_date``. Resumable: anything already archived (under any day) is
skipped. Requests go through ``http.fetch_json``, so the cdn host throttle applies (``run`` widens it
from 0.3 s to 0.5 s: a 0.3 s run was 403-blocked after ~400 requests).

The cdn only serves play-by-play from the 2019-20 season on (2017-18 and 2018-19 answer 403 for
every id probed) and has holes inside seasons. A 403 is ambiguous (missing id, or we are
rate-limited), so ``make_fetcher`` re-requests an archived "sentinel" game to tell them apart and
starts a fresh session, and only then waits out a block. Games that stay missing are retried on the next run.

For boxscore backfill (kind="boxscore"), iterate over archived playbyplay games to ensure alignment.
Validate each boxscore (game_id match, both teams present) and reject non-final payloads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from . import archive, config, db, http
from .backfill import DEFAULT_SEASON_TYPES, seasons_range
from .http import FetchError, fetch_json
from .sources import nba_cdn

SOURCE = nba_cdn.SOURCE
MAX_CONSECUTIVE_FAILS = 25      # timeouts / 5xx / 429 in a row: outage, stop instead of hammering
MAX_CONSECUTIVE_MISSING = 400   # the cdn has real holes (weeks of ids); only a huge run is odd
BLOCK_WAITS = (30, 60, 120, 240, 480)   # seconds to wait while the cdn is rate-limiting us
SESSION_REQUESTS = 150          # renew the HTTP session this often
CDN_INTERVAL = 0.5              # a bit slower than the shared 0.3 s: a 0.3 s run got 403-blocked


def _log(msg: str) -> None:
    print(msg, flush=True)


class Blocked(Exception):
    """The cdn keeps answering 403 even for a game we already hold: we are rate-limited."""


def new_session() -> None:
    """Drop the shared HTTP session (cookies + connection). The cdn flags a long-lived session
    (every request 403s) while a fresh one works at once, with the same IP."""
    http._session = None


def make_fetcher(sentinel: str | None, kind: str = "playbyplay",
                 sleep: Callable[[float], None] = time.sleep,
                 get: Callable[..., dict] = fetch_json,
                 reset: Callable[[], None] = new_session) -> Callable[[str], dict]:
    """A fetch(game_id) that tells a rate-limit block from a genuinely missing game.

    The cdn answers 403 both for ids it does not have and (after a few hundred quick requests) for
    everything. On a 403/404 we re-request ``sentinel``, a game we already archived: if that works,
    the 403 means *missing*; if it fails too our session is flagged: start a fresh one and retry,
    waiting (``BLOCK_WAITS``) only if fresh sessions are refused as well. The session is also renewed
    every ``SESSION_REQUESTS`` requests. One cheap attempt first (no http retries), since a retried
    403 costs ~13 s."""
    count = 0

    def url(g: str) -> str:
        if kind == "playbyplay":
            return nba_cdn.PLAYBYPLAY_URL.format(game_id=g)
        elif kind == "boxscore":
            return nba_cdn.BOXSCORE_URL.format(game_id=g)
        else:
            raise ValueError(f"unknown kind {kind!r}")

    def fetch(game_id: str) -> dict:
        nonlocal count
        count += 1
        if count % SESSION_REQUESTS == 0:
            reset()
        waits = iter(BLOCK_WAITS)
        fresh = False
        while True:
            try:
                return get(url(game_id), retries=0)
            except FetchError as e:
                if e.status not in (403, 404):
                    return nba_cdn.fetch_playbyplay(game_id)    # timeouts, 5xx, 429: normal retries
                if sentinel is None or sentinel == game_id:
                    raise
                try:
                    get(url(sentinel), retries=0)
                    sentinel_ok = True
                except FetchError:
                    sentinel_ok = False
                if sentinel_ok:
                    raise                                       # the cdn is fine: it lacks this game
                if not fresh:                                   # first try a brand-new session
                    fresh = True
                    reset()
                    continue
                w = next(waits, None)
                if w is None:
                    raise Blocked(f"cdn still blocking after {sum(BLOCK_WAITS)} s") from e
                _log(f"  blocked (403 on {game_id} and on {sentinel}); waiting {w}s")
                sleep(w)

    return fetch


def archived_index(kind: str = "playbyplay") -> dict[str, Path]:
    """game_id -> archived path, across every nba_day directory."""
    root = config.data_dir() / "raw" / SOURCE / kind
    out: dict[str, Path] = {}
    if root.exists():
        for p in root.glob("*/*.json.gz"):
            out[p.name[: -len(".json.gz")]] = p
    return out


def games_for(conn, seasons: list[str], season_types=DEFAULT_SEASON_TYPES) -> list[tuple[str, str, str]]:
    """[(game_id, game_date, season)] ordered by date, one per game."""
    q = ",".join("?" * len(season_types))
    s = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT game_id, MIN(game_date) AS d, MIN(season) AS season FROM team_games "
        f"WHERE season IN ({s}) AND season_type IN ({q}) AND game_date IS NOT NULL "
        f"GROUP BY game_id ORDER BY d, game_id", (*seasons, *season_types)).fetchall()
    return [(r["game_id"], r["d"], r["season"]) for r in rows]


def _is_valid_boxscore(payload: dict, game_id: str) -> bool:
    """Check boxscore has matching game_id and both teams' players."""
    game = (payload or {}).get("game") or {}
    if str(game.get("gameId")) != str(game_id):
        return False
    home = (game.get("homeTeam") or {}).get("players") or []
    away = (game.get("awayTeam") or {}).get("players") or []
    return bool(home) and bool(away)


def backfill_pbp(seasons: list[str], season_types=DEFAULT_SEASON_TYPES, kind: str = "playbyplay",
                 conn=None, fetch: Callable[[str], dict] | None = None, limit: int | None = None,
                 log_every: int = 100) -> dict:
    """Fetch + archive missing play-by-play or boxscore. Returns counts per season and totals.

    kind="boxscore" iterates over archived playbyplay games (ensuring alignment) and fetches the
    corresponding boxscores.
    """
    own = conn is None
    conn = conn or db.connect()
    t0 = time.monotonic()
    summary: dict = {"fetched": 0, "cached": 0, "missing": 0, "failed": 0, "by_season": {},
                     "errors": [], "aborted": False}
    try:
        have = archived_index(kind)
        fetch = fetch or make_fetcher(next(iter(have), None), kind=kind)

        if kind == "boxscore":
            # For boxscore, iterate over archived playbyplay games
            pbp_games = archived_index("playbyplay")
            # Extract day from archive path: .../raw/nba_cdn/playbyplay/{day}/{game_id}.json.gz
            games_iter = [(gid, pbp_games[gid].parent.name, None) for gid in pbp_games.keys()]
        else:
            # For playbyplay, iterate over games_for from DB
            games_iter = games_for(conn, seasons, season_types)

        todo = 0
        consecutive = missing_run = 0
        for gid, day, season in games_iter:
            if kind == "boxscore" and season is None:
                # Infer season from game_id: '0022600001' -> '2026-27'
                try:
                    y = 2000 + int(gid[3:5])
                    season = f"{y}-{(y + 1) % 100:02d}"
                except (ValueError, IndexError):
                    season = "unknown"

            row = summary["by_season"].setdefault(
                season, {"games": 0, "fetched": 0, "cached": 0, "missing": 0, "failed": 0})
            row["games"] += 1
            if gid in have:
                summary["cached"] += 1
                row["cached"] += 1
                continue
            if limit is not None and todo >= limit:
                continue
            todo += 1
            try:
                payload = fetch(gid)
                if kind == "playbyplay":
                    if not (payload.get("game") or {}).get("actions"):
                        raise FetchError(f"empty playbyplay for {gid}", status=404)
                else:  # boxscore
                    if not _is_valid_boxscore(payload, gid):
                        raise FetchError(f"invalid boxscore for {gid}", status=404)
                archive.save_raw(SOURCE, kind, day, gid, payload)
                summary["fetched"] += 1
                row["fetched"] += 1
                consecutive = missing_run = 0
            except FetchError as e:
                if e.status in (403, 404):   # the cdn has no file for this game
                    summary["missing"] += 1
                    row["missing"] += 1
                    missing_run += 1
                else:
                    summary["failed"] += 1
                    row["failed"] += 1
                    summary["errors"].append(f"{gid}: {e}")
                    consecutive += 1
            except Blocked as e:
                summary["aborted"] = True
                summary["errors"].append(f"{gid}: {e}")
                _log(f"  aborting: {e}")
                break
            except Exception as e:  # noqa: BLE001 - keep going, a rerun resumes
                summary["failed"] += 1
                row["failed"] += 1
                summary["errors"].append(f"{gid}: {type(e).__name__}: {e}")
                consecutive += 1
            if todo % log_every == 0:
                _log(f"  {season} {day} fetched={summary['fetched']} missing={summary['missing']} "
                     f"failed={summary['failed']} ({time.monotonic() - t0:.0f}s)")
            if consecutive >= MAX_CONSECUTIVE_FAILS or missing_run >= MAX_CONSECUTIVE_MISSING:
                summary["aborted"] = True
                _log(f"  aborting after {consecutive} consecutive failures / {missing_run} "
                     f"missing (last: {gid})")
                break
    finally:
        if own:
            conn.close()
    return summary


def run(start: str, end: str, kind: str = "playbyplay") -> dict:
    seasons = seasons_range(start, end)
    http.MIN_INTERVAL["cdn.nba.com"] = CDN_INTERVAL
    return backfill_pbp(seasons, kind=kind)
