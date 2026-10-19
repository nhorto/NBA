"""URGENT backfill of the official NBA injury-report PDFs (PLAYERS.md T0.1).

The reports are served only on a rolling window (roughly late Dec 2025 onwards for 2025-26) and
older ones keep disappearing, so this archives what is left for every 2025-26 game day.

Per game day (oldest day first):
  priority slots   13:30 ET (after the 11am-1pm resubmission), 17:30 ET, and the last report
                   before the day's first tip (``games.tipoff_utc`` when known, else 18:30 ET)
  extra slots      hourly 12:00..22:00 ET, only while the total stays under ``max_pdfs``
A missing slot (S3 answers 403) walks back on the 15-minute grid, up to 45 minutes.

Archive layout and ``injury_snapshots`` rows are exactly those of ``collect._official_report_step``
(key ``Injury-Report_YYYY-MM-DD_HH_MMPM``, ``source="nba_official"``, ``captured_at`` = report
slot in UTC, "NOT YET SUBMITTED" placeholder rows skipped), so backfilled rows are
indistinguishable from live-captured ones. Resumable: an archived key is never re-fetched and a
parsed one is never re-inserted; known-missing URLs are remembered in a small state file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from . import archive, config, db, http
from .sources import injury_report as ir

log = logging.getLogger("courtside.backfill_injuries")

SOURCE = "nba_official"          # == collect.OFFICIAL_SOURCE
KIND = "injury_report"
SEASON = "2025-26"
SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")
PRIORITY_HOURS = ((13, 30), (17, 30))
DEFAULT_PRETIP = (18, 30)
EXTRA_HOURS = tuple(range(12, 23))       # 12:00 .. 22:00 ET
WALK_BACK_STEPS = 3                      # 15, 30, 45 minutes
MAX_PDFS = 3000
SENTINEL_EVERY = 8                       # consecutive misses before checking a known-good URL
MAX_CONSECUTIVE_ERRORS = 25
BLOCK_WAITS = (30, 60, 120, 240)
SESSION_REQUESTS = 150
SEARCH_LO, SEARCH_HI = date(2025, 10, 21), date(2026, 1, 10)

_sleep = time.sleep   # monkeypatch in tests


class Aborted(RuntimeError):
    """Too many consecutive transport errors, or the server keeps blocking us."""


# ------------------------------------------------------------------ keys / slots

def report_key(slot: datetime) -> str:
    return "Injury-Report_" + slot.astimezone(config.NBA_TZ).strftime("%Y-%m-%d_%I_%M%p")


def report_path(slot: datetime) -> Path:
    et = slot.astimezone(config.NBA_TZ)
    return archive.raw_path(SOURCE, KIND, et.date(), report_key(et), ext=".pdf")


def _et(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=config.NBA_TZ)


def pretip_slot(day: date, tip_utc: str | datetime | None) -> datetime:
    """Last published report strictly before the first tip (15-min grid, ET); 18:30 if unknown."""
    if tip_utc is None:
        return _et(day, *DEFAULT_PRETIP)
    if isinstance(tip_utc, str):
        tip_utc = datetime.fromisoformat(tip_utc.replace("Z", "+00:00"))
    tip_et = tip_utc.astimezone(config.NBA_TZ)
    return ir._floor_to_grid(tip_et - timedelta(minutes=1))


def day_slots(day: date, tip_utc: str | datetime | None = None
              ) -> tuple[list[datetime], list[datetime]]:
    """(priority slots, extra hourly slots) for a game day, ET-aware, in priority order."""
    priority = [_et(day, *PRIORITY_HOURS[0]), _et(day, *PRIORITY_HOURS[1]),
                pretip_slot(day, tip_utc)]
    extras = [_et(day, h) for h in EXTRA_HOURS]
    return priority, extras


def walk_back(slot: datetime, steps: int = WALK_BACK_STEPS) -> list[datetime]:
    """The slot itself, then each earlier 15-minute grid point up to `steps` back."""
    return [slot - timedelta(minutes=ir.GRID_MINUTES * i) for i in range(steps + 1)]


# ------------------------------------------------------------------ game days

def game_days(conn: sqlite3.Connection, start: date | None = None,
              end: date | None = None) -> list[date]:
    """Distinct 2025-26 game dates (RS + PlayIn + Playoffs) in [start, end], ascending."""
    marks = ",".join("?" * len(SEASON_TYPES))
    rows = conn.execute(
        f"SELECT DISTINCT game_date FROM team_games WHERE season=? AND season_type IN ({marks}) "
        "AND game_date IS NOT NULL ORDER BY game_date", (SEASON, *SEASON_TYPES)).fetchall()
    days = [date.fromisoformat(r[0][:10]) for r in rows]
    return [d for d in days if (start is None or d >= start) and (end is None or d <= end)]


def first_tips(conn: sqlite3.Connection) -> dict[date, str]:
    """Earliest known tipoff_utc per NBA day (empty for seasons the games table lacks)."""
    try:
        rows = conn.execute("SELECT nba_day, MIN(tipoff_utc) FROM games "
                            "WHERE tipoff_utc IS NOT NULL GROUP BY nba_day").fetchall()
    except sqlite3.Error:
        return {}
    return {date.fromisoformat(r[0]): r[1] for r in rows}


# ------------------------------------------------------------------ fetching

class Fetcher:
    """fetch(url) -> bytes, or None when the report is not published (403/404).

    S3 answers 403 both for missing keys and (rarely) when a long session is flagged, so after a
    run of misses a known-good URL is re-requested; if that fails too the session is renewed and,
    failing that, we wait. Other errors (5xx, timeouts) are retried then raised."""

    def __init__(self, get: Callable[..., bytes] | None = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        self._get = get or (lambda url: http.fetch_bytes(url, retries=0, timeout=20.0))
        self._sleep = sleep or (lambda s: _sleep(s))
        self.requests = 0
        self.miss_streak = 0
        self.sentinel: str | None = None

    def _raw(self, url: str) -> bytes | None:
        self.requests += 1
        if self.requests % SESSION_REQUESTS == 0:
            http._session = None
        last: Exception | None = None
        for attempt in range(3):
            if attempt:
                self._sleep(2.0 * attempt)
            try:
                data = self._get(url)
            except http.FetchError as e:
                if e.status in (403, 404):
                    return None
                last = e
                continue
            if not data.startswith(b"%PDF"):
                last = http.FetchError(f"not a PDF: {url}", url=url)
                continue
            return data
        assert last is not None
        raise last

    def _renew_if_blocked(self) -> None:
        if not self.sentinel:
            return
        for wait in (0, *BLOCK_WAITS):
            if wait:
                log.warning("sentinel unreachable: waiting %ss", wait)
                self._sleep(wait)
            http._session = None
            if self._raw(self.sentinel) is not None:
                return
        raise Aborted("server keeps refusing a known-good report: rate-limited")

    def __call__(self, url: str) -> bytes | None:
        data = self._raw(url)
        if data is not None:
            self.miss_streak = 0
            self.sentinel = self.sentinel or url
            return data
        self.miss_streak += 1
        if self.miss_streak % SENTINEL_EVERY == 0:
            self._renew_if_blocked()
        return None


def find_earliest_date(fetch: Callable[[str], bytes | None] | None = None,
                       lo: date = SEARCH_LO, hi: date = SEARCH_HI) -> date | None:
    """Earliest date whose 5:00 PM ET report is still served (binary search, assumes the
    window is contiguous up to `hi`). None if even `hi` is unavailable."""
    fetch = fetch or Fetcher()

    def avail(d: date) -> bool:
        return fetch(ir.report_url(_et(d, 17, 0))) is not None

    if avail(lo):
        return lo
    if not avail(hi):
        return None
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        if avail(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------ state (resumability)

class State:
    """Remembers URLs known missing and reports that parsed to zero rows, across restarts."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.missing: set[str] = set()
        self.empty: set[str] = set()
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                (self.missing if d.get("s") == "missing" else self.empty).add(d["k"])

    def _append(self, s: str, k: str) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"s": s, "k": k}) + "\n")

    def mark_missing(self, url: str) -> None:
        if url not in self.missing:
            self.missing.add(url)
            self._append("missing", url)

    def mark_empty(self, key: str) -> None:
        if key not in self.empty:
            self.empty.add(key)
            self._append("empty", key)


# ------------------------------------------------------------------ ingest

def official_rows(parsed: list[dict], report_dt: datetime, path: Path | str) -> list[dict]:
    """injury_snapshots rows, built exactly as collect._official_report_step does."""
    captured = report_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [{"captured_at": captured, "source": SOURCE, "team": r["team"],
             "player_name": r.get("player_name_norm") or r["player_name"], "player_id": None,
             "status": r.get("status"), "reason": r.get("reason"), "raw_path": str(path)}
            for r in parsed if r.get("player_name_norm") or r.get("player_name")]


@dataclass
class Stats:
    pdfs_new: int = 0
    pdfs_existing: int = 0
    bytes_new: int = 0
    slots_missing: int = 0
    rows: int = 0
    unnamed_skipped: int = 0
    parse_failures: int = 0
    fetch_errors: int = 0
    failures: list[str] = field(default_factory=list)
    # per day: priority slots resolved (bool each), for coverage
    priority: dict[date, list[bool]] = field(default_factory=dict)

    @property
    def pdfs(self) -> int:
        return self.pdfs_new + self.pdfs_existing

    def summary(self, days: list[date]) -> dict:
        got = [d for d in days if any(self.priority.get(d, ()))]
        full = [d for d in days if len(self.priority.get(d, ())) and all(self.priority[d])]
        return {"game_days": len(days), "days_captured": len(got), "days_all_priority": len(full),
                "coverage_pct": round(100 * len(got) / len(days), 1) if days else 0.0,
                "days_none": [d.isoformat() for d in days if d not in got],
                "pdfs_new": self.pdfs_new, "pdfs_existing": self.pdfs_existing,
                "mb_new": round(self.bytes_new / 1e6, 1), "slots_missing": self.slots_missing,
                "rows": self.rows, "unnamed_skipped": self.unnamed_skipped,
                "parse_failures": self.parse_failures, "fetch_errors": self.fetch_errors}


def _ingest(conn: sqlite3.Connection, pdf: bytes, report_dt: datetime, path: Path,
            key: str, state: State, stats: Stats) -> None:
    try:
        parsed = ir.parse_report(pdf)
    except ir.ParserUnavailable:
        raise
    except Exception as e:   # corrupt / unexpected layout: keep the PDF, count it, move on
        stats.parse_failures += 1
        stats.failures.append(f"{key}: {type(e).__name__}: {e}")
        log.warning("parse failed %s: %s: %s", key, type(e).__name__, e)
        return
    rows = official_rows(parsed, report_dt, path)
    stats.unnamed_skipped += len(parsed) - len(rows)
    if rows:
        stats.rows += db.insert_injury_snapshots(conn, rows)
    else:
        state.mark_empty(key)


def process_slot(conn: sqlite3.Connection, slot: datetime, fetch: Callable[[str], bytes | None],
                 state: State, stats: Stats) -> datetime | None:
    """Archive + ingest the newest published report at/before `slot` (<=45 min back).
    Returns the slot actually used, or None if every candidate was missing."""
    for cand in walk_back(slot):
        key, path = report_key(cand), report_path(cand)
        url = ir.report_url(cand)
        if path.exists():
            stats.pdfs_existing += 1
            have = key in state.empty or conn.execute(
                "SELECT 1 FROM injury_snapshots WHERE raw_path=? LIMIT 1", (str(path),)).fetchone()
            if not have:
                _ingest(conn, path.read_bytes(), cand, path, key, state, stats)
            return cand
        if url in state.missing:
            stats.slots_missing += 1
            continue
        pdf = fetch(url)
        if pdf is None:
            state.mark_missing(url)
            stats.slots_missing += 1
            continue
        archive.save_raw_bytes(SOURCE, KIND, cand.astimezone(config.NBA_TZ).date(), key, pdf,
                               ext=".pdf")
        stats.pdfs_new += 1
        stats.bytes_new += len(pdf)
        _ingest(conn, pdf, cand, path, key, state, stats)
        return cand
    return None


# ------------------------------------------------------------------ driver

def run(*, start: date | None = None, end: date | None = None, extras: bool = True,
        max_pdfs: int = MAX_PDFS, fetch: Callable[[str], bytes | None] | None = None,
        conn: sqlite3.Connection | None = None, state_path: Path | None = None,
        earliest: date | None = None, days: list[date] | None = None) -> dict:
    """Run the backfill. `start` defaults to the earliest date still served (binary search)."""
    fetch = fetch or Fetcher()
    conn = conn or db.connect()
    if state_path is None:
        state_path = config.data_dir() / "logs" / "backfill_injuries_state.jsonl"
    state = State(state_path)
    stats = Stats()
    if earliest is None and start is None:
        earliest = find_earliest_date(fetch)
        if earliest is None:
            raise Aborted(f"no report available even on {SEARCH_HI}")
        log.info("earliest available date: %s", earliest)
    start = start or earliest
    if days is None:
        days = game_days(conn, start, end)
    tips = first_tips(conn)
    plans = {d: day_slots(d, tips.get(d)) for d in days}
    consecutive_errors = 0

    def do(slot: datetime) -> datetime | None:
        nonlocal consecutive_errors
        try:
            got = process_slot(conn, slot, fetch, state, stats)
            consecutive_errors = 0
            return got
        except (http.FetchError, OSError) as e:
            stats.fetch_errors += 1
            consecutive_errors += 1
            log.warning("fetch error at %s: %s", slot, e)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise Aborted(f"{consecutive_errors} consecutive fetch errors") from e
            return None

    log.info("phase 1: priority slots, %d game days %s..%s", len(days),
             days[0] if days else None, days[-1] if days else None)
    for d in days:
        stats.priority[d] = [do(s) is not None for s in plans[d][0]]
        if not any(stats.priority[d]):
            log.warning("%s: ALL priority slots missing", d)
        log.info("%s priority=%s pdfs=%d rows=%d", d, stats.priority[d], stats.pdfs, stats.rows)

    if extras:
        log.info("phase 2: hourly extras (cap %d PDFs)", max_pdfs)
        for d in days:
            for s in plans[d][1]:
                if stats.pdfs >= max_pdfs:
                    log.info("PDF cap %d reached", max_pdfs)
                    break
                do(s)
            else:
                log.info("%s extras done pdfs=%d rows=%d", d, stats.pdfs, stats.rows)
                continue
            break

    out = stats.summary(days)
    out["earliest"] = (earliest or start).isoformat() if (earliest or start) else None
    out["failures"] = stats.failures
    return out


def setup_logging() -> Path:
    path = config.data_dir() / "logs" / "backfill_injuries.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger("courtside.backfill_injuries")
    root.setLevel(logging.INFO)
    if not root.handlers:
        for h in (logging.FileHandler(path), logging.StreamHandler()):
            h.setFormatter(fmt)
            root.addHandler(h)
    return path
