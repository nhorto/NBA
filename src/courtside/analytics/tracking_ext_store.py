"""``player_tracking_ext``: per player-game tracking-dashboard fields (potential assists, drives,
touches by court area) from stats.nba.com ``leaguedashptstats`` queried one day at a time.

Why a per-day dashboard: the per-game feeds do not carry these numbers. ``boxscoreplayertrackv3``
has no potential assists or drives (checked over 2019-20 .. 2025-26 raw files) and
``boxscorematchupsv3`` leaves ``matchupPotentialAssists`` at 0 (all 1.9 M ``player_matchups``
rows). ``leaguedashptstats`` with ``DateFrom = DateTo = day`` returns one row per player who
played that day; a player plays at most one game a day, so the row is his game. The row is
matched to the game by (day, the row's team): ``team_games`` (game_date, team) first, then
``games`` (nba_day, home / away).

Measures (``nba_players.PT_MEASURES``): Passing (passes made / received, AST, FT / secondary AST,
POTENTIAL_AST, AST_PTS_CREATED, AST_ADJ), Drives (drives and what came of them), Possessions
(touches, front-court / elbow / post / paint touches, time of possession). Each measure updates
only its own columns, so a game can have Passing before Drives.

Requests: one per day x measure x season type present that day (normally 3 a day).

Archive: ``raw/nba_stats/leaguedashptstats/<day>/<measure>_<season_type>.json.gz``. A response is
archived (= final, never fetched again) only when it covers every team that finished a game that
day, or with ``archive_incomplete`` (past seasons: the data will not change any more). An
incomplete response (nba.com processes tracking with a lag of days early in a season) is still
stored, and the day is retried by the next run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from courtside import archive
from courtside.sources import nba_players as np_

SOURCE = np_.SOURCE
KIND = np_.PT_DAY_KIND
MEASURES = tuple(np_.PT_MEASURES)
# A game "has" a measure when this column is set on one of its rows.
KEY_COL = {"Passing": "pot_ast", "Drives": "drives", "Possessions": "touches"}
STYPE_BY_PREFIX = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}
BASE_COLS = ("game_id", "player_id", "team", "game_date", "season", "season_type", "minutes")


@dataclass
class DayGame:
    game_id: str
    season: str
    season_type: str
    final: bool


def archive_key(measure: str, season_type: str) -> str:
    return f"{measure.lower()}_{season_type.lower().replace(' ', '_')}"


def _season_of_gid(gid: str) -> str:
    yy = int(gid[3:5])
    return f"{2000 + yy}-{(yy + 1) % 100:02d}"


def day_games(conn: sqlite3.Connection, day: str) -> dict[str, DayGame]:
    """{team tricode: DayGame} for the NBA games (regular season / play-in / playoffs) of `day`.
    A game in ``team_games`` is final; a ``games`` row is final when its status is."""
    out: dict[str, DayGame] = {}
    for gid, team in conn.execute(
            "SELECT game_id, team FROM team_games WHERE game_date=?", (day,)):
        st = STYPE_BY_PREFIX.get(str(gid)[:3])
        if st:
            out[team] = DayGame(gid, _season_of_gid(gid), st, True)
    for gid, home, away, status in conn.execute(
            "SELECT game_id, home, away, status FROM games WHERE nba_day=?", (day,)):
        st = STYPE_BY_PREFIX.get(str(gid)[:3])
        if not st:
            continue
        for t in (home, away):
            if t and t not in out:
                out[t] = DayGame(gid, _season_of_gid(gid), st, status == "final")
    return out


def _upsert_sql(measure: str) -> tuple[str, tuple[str, ...]]:
    cols = BASE_COLS + tuple(np_.PT_MEASURES[measure].values())
    sets = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("game_id", "player_id"))
    return (f"INSERT INTO player_tracking_ext ({', '.join(cols)}) "
            f"VALUES ({', '.join(':' + c for c in cols)}) "
            f"ON CONFLICT(game_id, player_id) DO UPDATE SET {sets}", cols)


def store_rows(conn: sqlite3.Connection, day: str, measure: str, rows: list[dict],
               games: dict[str, DayGame]) -> tuple[int, set[str], list[int]]:
    """Write one measure's parsed rows of `day`. Returns (rows stored, teams covered, player ids
    whose team had no game that day)."""
    sql, cols = _upsert_sql(measure)
    out, seen, unmatched = [], set(), []
    for r in rows:
        g = games.get(r.get("team"))
        if g is None:
            unmatched.append(r["player_id"])
            continue
        seen.add(r["team"])
        full = {**r, "game_id": g.game_id, "game_date": day, "season": g.season,
                "season_type": g.season_type}
        out.append({c: full.get(c) for c in cols})
    with conn:
        conn.executemany(sql, out)
    return len(out), seen, unmatched


def measure_stored(conn: sqlite3.Connection, game_ids: set[str], measure: str) -> bool:
    """Every game in `game_ids` has at least one row with this measure's key column set."""
    if not game_ids:
        return True
    q = ",".join("?" * len(game_ids))
    n = conn.execute(f"SELECT COUNT(DISTINCT game_id) FROM player_tracking_ext WHERE game_id IN "
                     f"({q}) AND {KEY_COL[measure]} IS NOT NULL", tuple(game_ids)).fetchone()[0]
    return n == len(game_ids)


class Budget:
    """Caps network requests and stops after `empty_stop` empty answers in a row (nba.com has
    not processed those days yet: later days will not be processed either)."""

    def __init__(self, max_fetch: int | None = None, empty_stop: int | None = None):
        self.max_fetch, self.empty_stop = max_fetch, empty_stop
        self.requests = 0
        self.empty_run = 0

    def can_fetch(self) -> bool:
        if self.max_fetch is not None and self.requests >= self.max_fetch:
            return False
        return self.empty_stop is None or self.empty_run < self.empty_stop

    def record(self, n_rows: int) -> None:
        self.requests += 1
        self.empty_run = 0 if n_rows else self.empty_run + 1


def process_day(conn: sqlite3.Connection, day: str,
                fetch: Callable[[str, str, str, str], dict] | None, *,
                archive_incomplete: bool = False, budget: Budget | None = None,
                reparse: bool = False) -> dict:
    """Fill ``player_tracking_ext`` for one day: every measure x season type, from the archive
    when there, else via ``fetch(day, season, season_type, measure)`` (None = offline).

    Returns {"status": "no_games" | "done" | "pending", "rows", "requests", "cached",
    "pending": ["<measure>/<season type>", ...], "unmatched"}."""
    budget = budget or Budget()
    games = day_games(conn, day)
    final = {t: g for t, g in games.items() if g.final}
    out = {"status": "done", "rows": 0, "requests": 0, "cached": 0, "pending": [],
           "unmatched": 0}
    if not final:
        out["status"] = "no_games"
        return out
    for stype in sorted({g.season_type for g in final.values()}):
        expected = {t for t, g in final.items() if g.season_type == stype}
        gids = {final[t].game_id for t in expected}
        season = next(final[t].season for t in expected)
        for measure in MEASURES:
            path = archive.raw_path(SOURCE, KIND, day, archive_key(measure, stype))
            if path.exists():
                if not reparse and measure_stored(conn, gids, measure):
                    out["cached"] += 1
                    continue
                payload = archive.load_raw(path)
                fetched = False
            else:
                if fetch is None or not budget.can_fetch():
                    out["pending"].append(f"{measure}/{stype}")
                    continue
                payload = fetch(day, season, stype, measure)
                fetched = True
            rows = np_.parse_pt_day(payload, measure)
            if fetched:
                budget.record(len(rows))
                out["requests"] += 1
            n, seen, unmatched = store_rows(conn, day, measure, rows, games)
            out["rows"] += n
            out["unmatched"] += len(unmatched)
            complete = expected <= seen
            if fetched and (complete or archive_incomplete):
                archive.save_raw(SOURCE, KIND, day, archive_key(measure, stype), payload)
            elif fetched:                    # incomplete and still changing: retry next run
                out["pending"].append(f"{measure}/{stype}")
    if out["pending"]:
        out["status"] = "pending"
    return out


def season_days(conn: sqlite3.Connection, season: str) -> list[str]:
    """Days of `season` with a finished NBA game (team_games, plus final rows of games)."""
    days = {r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM team_games WHERE season=? AND game_date IS NOT NULL "
        "AND season_type IN ('Regular Season', 'PlayIn', 'Playoffs')", (season,))}
    days |= {r[0] for r in conn.execute(
        "SELECT DISTINCT nba_day FROM games WHERE season=? AND status='final'", (season,))}
    return sorted(d for d in days if d)


def days_missing(conn: sqlite3.Connection, days: list[str]) -> list[str]:
    """The days among `days` whose final games lack any measure (by the key columns)."""
    out = []
    for d in days:
        final = {g.game_id for g in day_games(conn, d).values() if g.final}
        if final and not all(measure_stored(conn, final, m) for m in MEASURES):
            out.append(d)
    return out


def backfill(conn: sqlite3.Connection, season: str, *, fetch=None, offline: bool = False,
             limit: int | None = None, current_season: str | None = None,
             on_abort: tuple[type[BaseException], ...] = ()) -> dict:
    """Every day of `season`, oldest first. Past seasons archive what they get; the current
    season (``current_season == season``) archives only complete days. ``fetch`` is wrapped by
    the caller (the backfill_players Guard); an exception in ``on_abort`` stops the run cleanly."""
    days = season_days(conn, season)
    budget = Budget(max_fetch=limit)
    s = {"days": len(days), "done": 0, "requests": 0, "cached": 0, "rows": 0, "pending": [],
         "failed": {}, "aborted": False, "error": None}
    for d in days:
        try:
            r = process_day(conn, d, None if offline else fetch, budget=budget,
                            archive_incomplete=season != current_season)
        except on_abort as e:
            s["aborted"], s["error"] = True, f"{d}: {type(e).__name__}: {e}"
            break
        except Exception as e:  # noqa: BLE001 - one bad day (404 / 5xx / bad payload): go on
            s["failed"][d] = f"{type(e).__name__}: {e}"
            continue
        s["requests"] += r["requests"]
        s["cached"] += r["cached"]
        s["rows"] += r["rows"]
        if r["status"] == "pending":
            s["pending"].append(d)
        elif r["status"] == "done":
            s["done"] += 1
    return s


def catchup_days(conn: sqlite3.Connection, day: date, window: int) -> list[str]:
    """Final days in [day - window, day) still missing a measure (oldest first)."""
    lo = (day - timedelta(days=window)).isoformat()
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT nba_day FROM games WHERE status='final' AND nba_day>=? AND nba_day<? "
        "ORDER BY nba_day", (lo, day.isoformat()))]
    return days_missing(conn, days)
