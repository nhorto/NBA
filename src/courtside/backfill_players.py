"""Player-data backfill from stats.nba.com: season tables and per-game box scores.

Same pattern as backfill.py / backfill_pbp.py: every raw response is archived (gzipped JSON under
``raw/nba_stats/<kind>/<nba_day>/<key>.json.gz``) before it is parsed, an archived response is never
fetched again, and parsing always works from the archive, so the tables can be rebuilt from
``var/raw`` and a killed run resumes where it stopped. One request at a time; the shared
``http`` throttle (0.7 s for stats.nba.com) applies.

Season tables (``run_season_tables``): playerindex, commonteamroster (+ coaches), matchupsrollup,
leagueseasonmatchups, leaguedashptdefend, leaguedashplayerptshot, synergyplaytypes,
leaguehustlestatsplayer. Archived under the season's pseudo day (``backfill.archive_day``).

Per-game (``run_per_game``): boxscorematchupsv3 (2017-18+), boxscoreplayertrackv3 and
boxscorehustlev2 (2019-20+). Game ids come from ``team_games``; archived under the game's
``game_date`` as ``raw/nba_stats/<kind>/<game_date>/<game_id>.json.gz``.

Blocks: an HTML / non-JSON / 403 / 429 / timeout response makes the ``Guard`` wait (BLOCK_WAITS,
escalating, fresh HTTP session each time) and retry the same request. After the last wait it raises
``Aborted``: the run stops cleanly and the next run resumes. Ten consecutive failed games also
trigger one back-off step. Progress is logged every ~100 games to ``var/logs/backfill_players.log``.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Callable, Iterator

from . import archive, backfill, backfill_pbp, config, db
from .backfill import archive_day, seasons_range
from .http import BlockedError, FetchError
from .sources import nba_players as np_

SOURCE = np_.SOURCE
LOG_NAME = "backfill_players.log"
DEFAULT_END = "2025-26"

# Season at which each endpoint's data starts being worth fetching (per-game) / default start.
GAME_FLOOR = {"boxscorematchupsv3": "2017-18", "boxscoreplayertrackv3": "2019-20",
              "boxscorehustlev2": "2019-20"}
GAME_ALIASES = {"matchups": "boxscorematchupsv3", "tracking": "boxscoreplayertrackv3",
                "hustle": "boxscorehustlev2"}
GAME_ORDER = ("matchups", "tracking", "hustle")
TABLE_FLOOR = {"index": None, "roster": "2015-16", "rollup": "2017-18", "lsm": "2017-18",
               "ptdefend": "2017-18", "ptshot": "2017-18", "synergy": "2017-18",
               "hustle": "2017-18"}
TABLE_ORDER = tuple(TABLE_FLOOR)

BLOCK_WAITS = (120, 300, 600, 900, 900, 900, 900, 900, 900, 900)   # seconds; ~2.6 h in total
MAX_CONSECUTIVE_FAILS = 10
RECENT_DAYS = 2    # an empty payload for a game this recent may just be "not processed yet"


# ------------------------------------------------------------------------ logging

def log_path():
    return config.data_dir() / "logs" / LOG_NAME


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------- block handling

class Aborted(Exception):
    """Still blocked after every back-off step: stop; a rerun resumes from the archive."""


def is_block(e: FetchError) -> bool:
    """HTML / 403 (BlockedError), 429, or no HTTP status at all (timeout / connection reset)."""
    return isinstance(e, BlockedError) or e.status in (None, 403, 429)


class Guard:
    """Runs fetches; on a block waits (escalating) with a fresh session and retries the request.

    Any success resets the escalation. Non-block failures (404, 5xx) are raised to the caller; after
    MAX_CONSECUTIVE_FAILS of them in a row one back-off step is taken too, since a sick endpoint
    looks the same as a quiet block."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep,
                 reset: Callable[[], None] = backfill_pbp.new_session,
                 waits: tuple[int, ...] = BLOCK_WAITS, log: Callable[[str], None] = _log):
        self.sleep, self.reset, self.waits, self.log = sleep, reset, waits, log
        self.level = 0
        self.fails = 0
        self.blocks = 0       # total block events (for the summary)

    def _back_off(self, why: str) -> None:
        if self.level >= len(self.waits):
            raise Aborted(f"still blocked after {sum(self.waits)} s of back-off ({why})")
        w = self.waits[self.level]
        self.level += 1
        self.blocks += 1
        self.log(f"  BLOCKED ({why}); backing off {w}s (step {self.level}/{len(self.waits)})")
        self.sleep(w)
        self.reset()

    def call(self, fn: Callable, *args):
        while True:
            try:
                out = fn(*args)
            except FetchError as e:
                if is_block(e):
                    self._back_off(str(e)[:120])
                    continue
                self.fails += 1
                if self.fails >= MAX_CONSECUTIVE_FAILS:
                    self.fails = 0
                    self._back_off(f"{MAX_CONSECUTIVE_FAILS} consecutive failures, last: {e}"[:160])
                raise
            self.fails = 0
            self.level = 0
            return out


# ------------------------------------------------------------------ season tables

class Job:
    """One archived request: ``fetch()`` -> payload, ``parse(payload)`` -> {table: rows}."""
    __slots__ = ("table", "kind", "day", "key", "fetch", "parse")

    def __init__(self, table, kind, day, key, fetch, parse):
        self.table, self.kind, self.day, self.key = table, kind, day, key
        self.fetch, self.parse = fetch, parse

    def __repr__(self) -> str:
        return f"Job({self.kind}/{self.day}/{self.key})"


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def _season_floor(start: str | None, table: str) -> str | None:
    return start or TABLE_FLOOR[table]


def season_jobs(start: str | None, end: str, season_types=np_.SEASON_TYPES,
                tables=TABLE_ORDER) -> Iterator[Job]:
    """All season-table requests, ordered by table then season (cheap tables first)."""
    N = np_
    for table in TABLE_ORDER:
        if table not in tables:
            continue
        if table == "index":
            yield Job("index", "playerindex", archive_day(end), f"all_{end}",
                      lambda: N.fetch_player_index(end),
                      lambda p: {"player_index": N.parse_player_index(p, end)})
            continue
        floor = _season_floor(start, table)
        for season in seasons_range(floor, end):
            day = archive_day(season)
            if table == "roster":
                for tid in N.LEAGUE_TEAM_IDS:
                    yield Job("roster", "commonteamroster", day, f"{season}_{tid}",
                              lambda tid=tid, s=season: N.fetch_roster(tid, s),
                              lambda p, s=season: dict(zip(
                                  ("team_rosters", "team_coaches"), N.parse_roster(p, s))))
                continue
            for st in season_types:
                key = f"{season}_{_slug(st)}"
                if table == "rollup":
                    yield Job(table, "matchupsrollup", day, key,
                              lambda s=season, t=st: N.fetch_matchups_rollup(s, t),
                              lambda p, s=season, t=st: {
                                  "matchup_rollup": N.parse_matchups_rollup(p, s, t)})
                elif table == "lsm":
                    yield Job(table, "leagueseasonmatchups", day, key,
                              lambda s=season, t=st: N.fetch_league_season_matchups(s, t),
                              lambda p, s=season, t=st: {
                                  "season_matchups": N.parse_season_matchups(p, s, t)})
                elif table == "ptdefend":
                    for cat in N.DEF_CATEGORIES:
                        yield Job(table, "leaguedashptdefend", day, f"{key}_{_slug(cat)}",
                                  lambda s=season, t=st, c=cat: N.fetch_pt_defend(s, t, c),
                                  lambda p, s=season, t=st, c=cat: {
                                      "pt_defend": N.parse_pt_defend(p, s, t, c)})
                elif table == "ptshot":
                    for rng, label in N.PTSHOT_BUCKETS.items():
                        yield Job(table, "leaguedashplayerptshot", day, f"{key}_{_slug(label)}",
                                  lambda s=season, t=st, r=rng: N.fetch_pt_shot(s, t, r),
                                  lambda p, s=season, t=st, b=label: {
                                      "pt_shot_defdist": N.parse_pt_shot(p, s, t, b)})
                elif table == "synergy":
                    for grouping in N.GROUPINGS:
                        for entity in ("P", "T"):
                            for pt in N.PLAY_TYPES:
                                if (entity, grouping, pt) in N.SYNERGY_EMPTY:
                                    continue
                                yield Job(
                                    table, "synergyplaytypes", day,
                                    f"{key}_{entity}_{grouping.lower()}_{_slug(pt)}",
                                    lambda s=season, t=st, e=entity, g=grouping, p_=pt:
                                        N.fetch_synergy(s, t, e, g, p_),
                                    lambda p, s=season, t=st, e=entity, g=grouping, p_=pt: {
                                        "synergy_playtypes": N.parse_synergy(p, s, t, e, g, p_)})
                elif table == "hustle":
                    yield Job(table, "leaguehustlestatsplayer", day, key,
                              lambda s=season, t=st: N.fetch_hustle_season(s, t),
                              lambda p, s=season, t=st: {
                                  "hustle_season": N.parse_hustle_season(p, s, t)})


def run_season_tables(start: str | None = None, end: str = DEFAULT_END,
                      season_types=np_.SEASON_TYPES, tables=TABLE_ORDER, force: bool = False,
                      conn=None, guard: Guard | None = None) -> dict:
    """Fetch (or reuse archived) season-level responses and upsert them.

    Returns {"fetched", "cached", "failed", "rows": {table: n}, "errors", "aborted", "run_id"}."""
    own = conn is None
    conn = conn or db.connect()
    guard = guard or Guard()
    run_id = db.start_run(conn, "backfill_players_season")
    summary: dict = {"fetched": 0, "cached": 0, "failed": 0, "rows": {}, "errors": [],
                     "aborted": False, "run_id": run_id}
    t0 = time.monotonic()
    try:
        jobs = list(season_jobs(start, end, season_types, tables))
        _log(f"season tables: {len(jobs)} requests ({start or 'table defaults'}..{end})")
        for i, job in enumerate(jobs, 1):
            try:
                path = archive.raw_path(SOURCE, job.kind, job.day, job.key)
                if path.exists() and not force:
                    payload = archive.load_raw(path)
                    summary["cached"] += 1
                else:
                    payload = guard.call(job.fetch)
                    archive.save_raw(SOURCE, job.kind, job.day, job.key, payload)
                    summary["fetched"] += 1
                for table, rows in job.parse(payload).items():
                    n = db.upsert_rows(conn, table, rows)
                    summary["rows"][table] = summary["rows"].get(table, 0) + n
            except Aborted as e:
                summary["aborted"] = True
                summary["errors"].append(f"{job}: {e}")
                _log(f"  aborting: {e}")
                break
            except Exception as e:  # noqa: BLE001 - keep going; a rerun resumes the failures
                summary["failed"] += 1
                summary["errors"].append(f"{job.kind}/{job.key}: {type(e).__name__}: {e}")
                _log(f"  FAILED {job.kind}/{job.key}: {type(e).__name__}: {e}")
            if i % 100 == 0:
                _log(f"  season tables {i}/{len(jobs)} fetched={summary['fetched']} "
                     f"cached={summary['cached']} failed={summary['failed']} "
                     f"({time.monotonic() - t0:.0f}s)")
        ok = summary["failed"] == 0 and not summary["aborted"]
        db.finish_run(conn, run_id, ok, _detail(summary))
    except BaseException as e:
        db.finish_run(conn, run_id, False, f"aborted: {type(e).__name__}: {e}")
        raise
    finally:
        if own:
            conn.close()
    _log(f"season tables done: fetched={summary['fetched']} cached={summary['cached']} "
         f"failed={summary['failed']} rows={summary['rows']}")
    return summary


# ------------------------------------------------------------------------ per game

def archived_games(kind: str) -> dict[str, object]:
    """game_id -> archived path for one kind, across every nba_day directory."""
    root = config.data_dir() / "raw" / SOURCE / kind
    out: dict = {}
    if root.exists():
        for p in root.glob("*/*.json.gz"):
            out[p.name[: -len(".json.gz")]] = p
    return out


def resolve_kinds(which: str) -> list[str]:
    """'matchups'|'tracking'|'hustle'|'all' -> endpoint kinds in run order."""
    if which == "all":
        return [GAME_ALIASES[k] for k in GAME_ORDER]
    if which in GAME_ALIASES:
        return [GAME_ALIASES[which]]
    if which in np_.GAME_KINDS:
        return [which]
    raise ValueError(f"unknown per-game kind {which!r}; use matchups|tracking|hustle|all")


def _season_year(s: str) -> int:
    return int(s[:4])


def run_per_game(which: str = "all", start: str | None = None, end: str = DEFAULT_END,
                 conn=None, guard: Guard | None = None,
                 fetch: Callable[[str, str], dict] | None = None, limit: int | None = None,
                 reparse: bool = False, log_every: int = 100, today: date | None = None) -> dict:
    """Backfill per-game box scores for each kind in turn (matchups, then tracking, then hustle).

    ``limit`` caps network fetches per kind (testing). ``reparse`` re-parses archived games that
    already have rows. Returns {"by_kind": {kind: counts}, "aborted", "run_id"}."""
    kinds = resolve_kinds(which)
    own = conn is None
    conn = conn or db.connect()
    guard = guard or Guard()
    fetch = fetch or np_.fetch_game
    today = today or date.today()
    run_id = db.start_run(conn, "backfill_players_games")
    summary: dict = {"by_kind": {}, "aborted": False, "run_id": run_id, "errors": []}
    try:
        for kind in kinds:
            if _stop(summary):
                break
            floor = GAME_FLOOR[kind]
            first = start if start and _season_year(start) > _season_year(floor) else floor
            if _season_year(first) > _season_year(end):
                continue
            summary["by_kind"][kind] = _run_kind(
                conn, kind, seasons_range(first, end), guard, fetch, summary, limit, reparse,
                log_every, today)
        ok = not summary["aborted"] and not any(
            v["failed"] for v in summary["by_kind"].values())
        db.finish_run(conn, run_id, ok, str({k: v for k, v in summary["by_kind"].items()}))
    except BaseException as e:
        db.finish_run(conn, run_id, False, f"aborted: {type(e).__name__}: {e}")
        raise
    finally:
        if own:
            conn.close()
    return summary


def _stop(summary: dict) -> bool:
    return summary["aborted"]


def _run_kind(conn, kind, seasons, guard, fetch, summary, limit, reparse, log_every,
              today) -> dict:
    table, parse = np_.GAME_PARSERS[kind]
    have = archived_games(kind)
    done = {r[0] for r in conn.execute(f"SELECT DISTINCT game_id FROM {table}")}
    games = backfill_pbp.games_for(conn, seasons)
    todo = sum(1 for g, _, _ in games if g not in have)
    counts = {"games": len(games), "fetched": 0, "cached": 0, "reparsed": 0, "empty": 0,
              "failed": 0, "rows": 0}
    _log(f"{kind}: {len(games)} games {seasons[0]}..{seasons[-1]}, {len(games) - todo} archived, "
         f"{todo} to fetch")
    t0 = time.monotonic()
    fetched_here = 0
    recent_cut = (today - timedelta(days=RECENT_DAYS)).isoformat()
    for gid, day, season in games:
        path = have.get(gid)
        try:
            if path is not None:
                if gid in done and not reparse:
                    counts["cached"] += 1
                    continue
                payload = archive.load_raw(path)
                counts["reparsed"] += 1
            else:
                if limit is not None and fetched_here >= limit:
                    continue
                payload = guard.call(fetch, kind, gid)
                fetched_here += 1
                if parse(payload) or day < recent_cut:   # empty + fresh game: retry on a rerun
                    archive.save_raw(SOURCE, kind, day, gid, payload)
                counts["fetched"] += 1
            rows = parse(payload)
            if rows:
                counts["rows"] += db.upsert_rows(conn, table, rows)
            else:
                counts["empty"] += 1
        except Aborted as e:
            summary["aborted"] = True
            summary["errors"].append(f"{kind} {gid}: {e}")
            _log(f"  aborting: {e}")
            break
        except Exception as e:  # noqa: BLE001 - keep going; a rerun resumes the failures
            counts["failed"] += 1
            summary["errors"].append(f"{kind} {gid}: {type(e).__name__}: {e}")
            _log(f"  FAILED {kind} {gid}: {type(e).__name__}: {e}")
            continue
        if path is None and fetched_here % log_every == 0:
            el = time.monotonic() - t0
            rate = fetched_here / el * 60 if el else 0.0
            left = max(todo - fetched_here, 0)
            eta = f", eta {left / rate / 60:.1f} h" if rate else ""
            _log(f"  {kind} {season} {day} fetched={fetched_here}/{todo} empty={counts['empty']} "
                 f"failed={counts['failed']} blocks={guard.blocks} {rate:.1f} games/min{eta}")
    _log(f"{kind} done: {counts}")
    return counts


def _detail(summary: dict) -> str:
    return (f"fetched={summary['fetched']} cached={summary['cached']} "
            f"failed={summary['failed']} aborted={summary['aborted']} rows={summary['rows']}"
            + (f" errors={summary['errors'][:5]}" if summary["errors"] else ""))
