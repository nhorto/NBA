"""Background warm-up of the stats pages' shared aggregates (stale-while-revalidate).

One daemon thread per database file (`courtside-stats-warm`, lowered CPU priority where the OS
allows it) that

* at start, and whenever a data domain version moves on (stats/db.py: "box" after a night of
  box scores, "dfs" after a DFS tick), precomputes the expensive aggregates every page shares:
  league tables and ranks, standings, season packs (leaders base tables) and DvP for the three
  seasons, the similar-players matrix, the opportunity league baselines and regression flags,
  the search index; plus one-time process costs (heavy imports, template compiles);
* runs the per-key refreshes that `stats.db.cached` queues when a request is served a stale
  value (it registers itself with `stats.db.set_refresher`).

Every job goes through `stats.db.cached` in sync mode, so a fresh entry is a hit and costs
nothing, and requests keep getting the old value while the new one is built. One INFO log line
per cycle with the timings.

Disabled with COURTSIDE_WARM=0 (tests set it in conftest.py). Started from the web app's
startup hook (web/app.py) with `start(db_path, today=..., extra=...)`; `stop()` on shutdown.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import date
from typing import Any, Callable, Iterable, Iterator

from . import db as sdb
from . import seasons as S

ENV = "COURTSIDE_WARM"
POLL_S = 10.0                    # how often the thread looks at the domain versions
_STOP = object()


def enabled() -> bool:
    """False when COURTSIDE_WARM is 0 / false / no / off."""
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off", "")


def _logger() -> logging.Logger:
    log = logging.getLogger("courtside.warm")
    if log.hasHandlers():
        return log
    return logging.getLogger("uvicorn.error")       # visible under `courtside serve`


def _lower_priority() -> None:
    """Nice this thread (Linux: per-thread nice; macOS: background QoS). Best effort."""
    try:
        if hasattr(os, "PRIO_DARWIN_THREAD"):
            os.setpriority(os.PRIO_DARWIN_THREAD, 0, os.PRIO_DARWIN_BG)
        else:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except (OSError, AttributeError, ValueError):
        pass


@contextmanager
def _responsive() -> Iterator[None]:
    """Shorter GIL switch interval while the warmer computes. A request thread releases the GIL
    on every SQLite call and then waits up to the switch interval (5 ms) to get it back from a
    CPU-bound thread; with ~50 small queries a page that is 25 ms alone took 0.2-0.6 s during a
    warm cycle. Process-wide, restored afterwards (nested use keeps the outer value)."""
    global _resp_depth, _resp_prev
    with _resp_lock:
        if _resp_depth == 0:
            _resp_prev = sys.getswitchinterval()
            sys.setswitchinterval(min(_resp_prev, SWITCH_S))
        _resp_depth += 1
    try:
        yield
    finally:
        with _resp_lock:
            _resp_depth -= 1
            if _resp_depth == 0:
                sys.setswitchinterval(_resp_prev)


SWITCH_S = 0.0005
_resp_lock = threading.Lock()
_resp_depth = 0
_resp_prev = 0.005


# ------------------------------------------------------------------ the jobs
SLATE_MAX = 300                  # per-player warm-up: at most this many slate players


def _optional(module: str, attr: str) -> Callable | None:
    """A function of a stats module another feature owns (None when it is not there)."""
    import importlib
    try:
        return getattr(importlib.import_module(f"{__package__}.{module}"), attr, None)
    except ImportError:
        return None


def slate_teams(conn: sqlite3.Connection, today: date) -> list[str]:
    """Teams playing today (the `games` schedule)."""
    try:
        return sorted({t for r in conn.execute(
            "SELECT home, away FROM games WHERE nba_day=?", (today.isoformat(),)) for t in r if t})
    except sqlite3.Error:
        return []


def slate_players(conn: sqlite3.Connection, today: date) -> list[int]:
    """Players of the teams playing today (this season's box scores, else last season's)."""
    try:
        teams = slate_teams(conn, today)
        if not teams:
            return []
        q = ",".join("?" * len(teams))
        for season in S.seasons(today)[:2]:
            ids = [int(r[0]) for r in conn.execute(
                f"SELECT player_id FROM player_games WHERE season=? AND team IN ({q}) "
                f"GROUP BY player_id ORDER BY SUM(min) DESC", (season, *teams))]
            if ids:
                return ids[:SLATE_MAX]
    except sqlite3.Error:
        pass
    return []


def warm_player(conn: sqlite3.Connection, pid: int, today: date) -> None:
    """The per-player aggregates of /player/<pid> at its default season (regular season)."""
    from . import opportunity as OPP
    from . import player_tools as PT
    from . import players as P
    from . import similar as SIM

    window = S.seasons(today)
    have = {ln["season"] for ln in P.season_lines(conn, pid, window)}
    season = next((s for s in window if s in have), window[0])
    PT.games(conn, pid, window)
    PT.teammates(conn, pid, window)
    PT.clutch(conn, pid, season, S.REGULAR, window)
    P.shot_chart(conn, pid, season, S.REGULAR)
    OPP.player_opportunity(conn, pid, season, S.REGULAR)
    if (SIM.similar(conn, pid, season, today) or {}).get("status") != "ok":
        SIM.similar(conn, pid, S.prev_season(season), today)


def jobs(conn: sqlite3.Connection, today: date
         ) -> list[tuple[str, Callable[[sqlite3.Connection], Any]]]:
    """(name, fn(conn)): the shared aggregates (most used first), then the per-player pages of
    tonight's slate. Each fn calls the same public function the pages call, so it fills the
    same cache key."""
    from . import compare as CMP
    from . import leaders as L
    from . import opportunity as OPP
    from . import player_tools as PT
    from . import players as P
    from . import search as SR
    from . import similar as SIM
    from . import teams as T

    window = S.seasons(today)
    gt_season = _optional("garbage", "season_pass")          # garbage time (stints)
    out: list[tuple[str, Callable]] = [
        ("names", lambda c: (P._index_names(c), P.dk_ids(c, 0), SIM.bio(c))),
        ("search", lambda c: SR.search(c, "a", today=today)),
    ]
    for s in window:
        out += [
            (f"teams {s}", lambda c, s=s: (T.league(c, s), T.standings(c, s),
                                           CMP.team_factors(c, s))),
            (f"players {s}", lambda c, s=s: (P.league_table(c, s, S.REGULAR),
                                             P.league_table(c, s, S.PLAYOFFS),
                                             P.league_zones(c, s, S.REGULAR))),
            (f"leaders {s}", lambda c, s=s: (L.pack(c, s, S.REGULAR),
                                             L.team_gp(c, s, S.REGULAR),
                                             L.leaders(c, s, S.REGULAR))),
            (f"dvp {s}", lambda c, s=s: PT._dvp_season(c, s)),
        ]
        if gt_season is not None:
            out.append((f"garbage {s}", lambda c, s=s: gt_season(c, s, S.REGULAR)))
    out.append(("similar", lambda c: SIM.season_matrix(c, today)))
    for s in window:
        out += [
            (f"chances {s}", lambda c, s=s: (OPP.league_cached(c, s),
                                             OPP.league_shots_cached(c, s),
                                             OPP.flags_cached(c, s, S.REGULAR))),
            (f"compare {s}", lambda c, s=s: CMP.pools(c, s)),
        ]
    out.append(("tonight imports", lambda c: __import__("courtside.predict")))  # 0.6 s once
    from . import tonight as TN
    teams = slate_teams(conn, today)
    out.append(("tonight teams", lambda c: [TN.team_games(c, t, TN.window(today),
                                                          today.isoformat()) for t in teams]))
    for pid in slate_players(conn, today):
        out.append((f"player {pid}", lambda c, pid=pid: warm_player(c, pid, today)))
    return out


# ------------------------------------------------------------------ the thread
class Warmer:
    def __init__(self, db_path: str | os.PathLike, *, today: Callable[[], date] = date.today,
                 extra: Iterable[tuple[str, Callable[[], Any]]] = (), poll: float = POLL_S,
                 job_list: Callable[[sqlite3.Connection, date], list] | None = None):
        self.db_path = os.path.realpath(db_path)
        self.today = today
        self.extra = list(extra)                 # one-time jobs (template compiles...)
        self.poll = poll
        self.job_list = job_list or jobs
        self.q: queue.Queue = queue.Queue()
        self._pending: set = set()
        self._plock = threading.Lock()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.versions: dict | None = None        # domain versions of the last warm cycle
        self.cycles: list[dict] = []             # {reason, seconds, jobs: {name: s}, errors}
        self.refreshed = 0
        self.ready = threading.Event()           # set after the first cycle

    # -- stale-while-revalidate hook (stats.db.set_refresher)
    def submit(self, bucket: tuple, key: Any, compute: Callable[[], None]) -> None:
        tag = (bucket, key)
        with self._plock:
            if tag in self._pending:
                return
            self._pending.add(tag)
        self.q.put((tag, compute))

    def start(self) -> Warmer:
        if self.thread is None:
            sdb.set_refresher(self.submit)
            self.thread = threading.Thread(target=self._run, name="courtside-stats-warm",
                                           daemon=True)
            self.thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.q.put(_STOP)
        if sdb._refresher == self.submit:
            sdb.set_refresher(None)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout)

    # -- work
    def warm(self, reason: str) -> dict:
        """One cycle over the jobs (sync mode: stale or missing entries are rebuilt here)."""
        t0 = time.perf_counter()
        self.versions = sdb.domain_versions(self.db_path, force=True)
        sdb.unpin_all()                          # re-pinned below by what is still warmed
        timings: dict[str, float] = {}
        errors = 0
        todo = list(self.extra) if not self.cycles else []
        with sdb.ro(self.db_path) as conn, sdb.sync_mode(), _responsive():
            if conn is None:
                return {}
            try:
                day = self.today()
            except Exception:  # noqa: BLE001 - never fail the warmer over the clock
                day = date.today()
            for name, fn in todo:
                t = time.perf_counter()
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    errors += 1
                    _logger().exception("stats warm-up: %s failed", name)
                timings[name] = time.perf_counter() - t
            for name, fn in self.job_list(conn, day):
                if self._stop.is_set():
                    break
                self._drain(block=False)          # requests' stale keys first
                t = time.perf_counter()
                try:
                    # shared aggregates are pinned: per-player keys cannot evict them
                    with nullcontext() if name.startswith("player ") else sdb.pinning():
                        fn(conn)
                except Exception:  # noqa: BLE001
                    errors += 1
                    _logger().exception("stats warm-up: %s failed", name)
                timings[name] = time.perf_counter() - t
        cyc = {"reason": reason, "seconds": time.perf_counter() - t0, "jobs": timings,
               "errors": errors}
        self.cycles = (self.cycles + [cyc])[-20:]
        per_player = [v for k, v in timings.items() if k.startswith("player ")]
        slow = sorted(((k, v) for k, v in timings.items() if not k.startswith("player ")),
                      key=lambda kv: -kv[1])[:6]
        _logger().info("stats warm-up (%s): %.2fs, %d jobs, %d errors; slate players %d in "
                       "%.2fs; slowest: %s", reason, cyc["seconds"], len(timings), errors,
                       len(per_player), sum(per_player),
                       ", ".join(f"{k} {v:.2f}s" for k, v in slow))
        return cyc

    def _drain(self, block: bool, timeout: float = 0.0) -> bool:
        """Run queued refreshes; returns False when asked to stop."""
        while True:
            try:
                item = self.q.get(timeout=timeout) if block else self.q.get_nowait()
            except queue.Empty:
                return True
            if item is _STOP:
                return False
            tag, compute = item
            with self._plock:
                self._pending.discard(tag)
            try:
                with _responsive():
                    compute()
                self.refreshed += 1
            except Exception:  # noqa: BLE001
                _logger().exception("stats refresh of %r failed", tag[1])
            block = False

    def check(self) -> bool:
        """Warm again when a domain version moved on since the last cycle."""
        vers = sdb.domain_versions(self.db_path)
        if vers is not None and vers != self.versions:
            changed = sorted(d for d in vers if (self.versions or {}).get(d) != vers[d])
            self.warm("data change: " + ",".join(changed))
            return True
        return False

    def _run(self) -> None:
        _lower_priority()
        try:
            self.warm("start")
        except Exception:  # noqa: BLE001
            _logger().exception("stats warm-up failed")
        self.ready.set()
        nxt = time.monotonic() + self.poll
        while not self._stop.is_set():
            if not self._drain(block=True, timeout=max(0.0, nxt - time.monotonic())):
                return
            if time.monotonic() >= nxt:
                try:
                    self.check()
                except Exception:  # noqa: BLE001
                    _logger().exception("stats warm-up failed")
                nxt = time.monotonic() + self.poll


_lock = threading.Lock()
_warmers: dict[str, Warmer] = {}


def start(db_path: str | os.PathLike, *, today: Callable[[], date] = date.today,
          extra: Iterable[tuple[str, Callable[[], Any]]] = (), force: bool = False
          ) -> Warmer | None:
    """Start (once per database file and process) the warmer; None when disabled
    (COURTSIDE_WARM=0, unless `force`) or the database is missing."""
    if not (force or enabled()) or not os.path.exists(db_path):
        return None
    key = os.path.realpath(db_path)
    with _lock:
        w = _warmers.get(key)
        if w is None or w._stop.is_set():
            w = _warmers[key] = Warmer(key, today=today, extra=extra).start()
        return w


def stop(db_path: str | os.PathLike | None = None) -> None:
    """Stop one warmer (or all)."""
    with _lock:
        keys = [os.path.realpath(db_path)] if db_path else list(_warmers)
        ws = [_warmers.pop(k) for k in keys if k in _warmers]
    for w in ws:
        w.stop()
