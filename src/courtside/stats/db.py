"""Read-only SQLite access and an in-process memo cache for the stats pages.

The web process never writes: connections are opened with `mode=ro` (as in web/store.py).

`cached(key, fn, conn_or_path, domain=None)` memoises heavy aggregates (league tables, ranks,
leaders, ...) per database file. Thread-safe; a key is computed once even when several requests
miss at the same time.

    from courtside.stats import db as sdb
    rows = sdb.cached(("league", season), lambda c: league_table(c, season), conn,
                      domain="box")

`fn` always receives a read-only connection: the one you passed, or (when you pass a path) one
opened lazily on a miss and closed afterwards, so a warm hit never touches SQLite. `fn` must
only use that argument (a background refresh calls it with its own connection).

**Invalidation.** Two flavours:

* `domain=None` (default, the original behaviour): every entry is dropped as soon as the file
  version changes, i.e. (mtime_ns, size) of `courtside.db` plus its `-wal` file. In production
  the collectors write all day (a DFS tick every 5 minutes, odds, injuries, props), so use it
  only for cheap results or ones that really follow those tables (live scores, odds).
* `domain="box"` / `"dfs"` / `"injuries"` (or a tuple): the entry follows a *data version* per
  domain (DOMAINS below): `(MIN(rowid), MAX(rowid))` of every table of the domain, one query
  (~0.1-2 ms on the 1.5 GB DB; COUNT(*) was rejected: 0.7-0.9 s on shots/player_matchups).
  The collectors only append (INSERT / INSERT OR REPLACE, which re-inserts at a new rowid) or
  delete-then-insert, so a new night of box scores moves MAX(rowid) and a DFS tick does not.
  Not detected: an in-place UPDATE, or deleting the newest rows and re-inserting the same
  number (re-importing the last game); those show up with the domain's next append. The probe
  runs only when the file version changed and at most every PROBE_INTERVAL seconds per process.

**Stale-while-revalidate** (domain entries only). When a domain version moves on, the old
entries are kept. If a background refresher is registered (`set_refresher`, done by
`stats/warm.py` in the server), a request that finds a stale entry gets the old value at once
and the key is queued for recomputation; without one (tests, scripts, COURTSIDE_WARM=0) the
key is recomputed in the request as before. A value computed from a stale nested value is
stored as stale too, so it is refreshed again once its inputs are fresh.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

MAX_ENTRIES = 512                  # per database file (file flavour); least recently used go
MAX_DOMAIN_ENTRIES = 2048          # per database file and domain; pinned entries never go
PROBE_INTERVAL = float(os.environ.get("COURTSIDE_STATS_PROBE_S", "30"))  # seconds

# The tables each data domain follows (prefixes end with '*'). The "box" domain also follows
# the reference tables written by the same nightly backfill (player_index, team_rosters,
# team_coaches) and the DK-to-NBA id map (player_ids) used for names and positions.
DOMAINS: dict[str, tuple[str, ...]] = {
    "box": ("player_games", "team_games", "*_adv", "stints", "shots", "shot_xfg",
            "player_tracking", "player_tracking_ext", "player_hustle", "player_matchups",
            "synergy_playtypes", "player_starts",
            "player_index", "team_rosters", "team_coaches", "player_ids"),
    "dfs": ("dfs_salaries", "dfs_runs", "dfs_lineups", "dfs_ownership_proxy",
            "dfs_contest_ownership", "player_projection*"),
    "injuries": ("injury_snapshots",),
}


def connect_ro(path: str | Path) -> sqlite3.Connection | None:
    """Read-only connection (rows as sqlite3.Row), or None when the file is missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


@contextmanager
def ro(path: str | Path) -> Iterator[sqlite3.Connection | None]:
    """`with ro(path) as conn:` (conn is None without a database); always closed."""
    conn = connect_ro(path)
    try:
        yield conn
    finally:
        if conn is not None:
            conn.close()


def db_file(conn: sqlite3.Connection) -> str | None:
    """Filesystem path of a connection's main database, or None for in-memory/temp DBs."""
    try:
        for row in conn.execute("PRAGMA database_list"):
            if row[1] == "main":
                return row[2] or None
    except sqlite3.Error:
        return None
    return None


def version(path: str | Path) -> tuple | None:
    """(mtime_ns, size) of the DB file and its -wal file; None when the file is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        w = os.stat(f"{path}-wal")
        wal = (w.st_mtime_ns, w.st_size)
    except OSError:
        wal = None
    return (st.st_mtime_ns, st.st_size, wal)


# ------------------------------------------------------------------ domain versions
def _match(table: str, pat: str) -> bool:
    if pat.startswith("*"):
        return table.endswith(pat[1:])
    if pat.endswith("*"):
        return table.startswith(pat[:-1])
    return table == pat


def domain_tables(domain: str, tables: Iterable[str]) -> list[str]:
    """The existing tables (from `tables`) that `domain` follows, sorted."""
    pats = DOMAINS[domain]
    return sorted(t for t in tables if any(_match(t, p) for p in pats))


def probe(conn: sqlite3.Connection) -> dict[str, tuple]:
    """{domain: version} read now: per table (name, MIN(rowid), MAX(rowid)) in one statement
    (two scalar subqueries per table, so each uses the rowid min/max shortcut)."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    have = {r[0]: (r[1] or "") for r in rows}
    out = {}
    for dom in DOMAINS:
        tabs = domain_tables(dom, have)
        if not tabs:
            out[dom] = ()
            continue
        parts = []
        for t in tabs:
            if "WITHOUT ROWID" in have[t].upper():         # no rowid: fall back to a count
                parts += [f'(SELECT COUNT(*) FROM "{t}")', "NULL"]
            else:
                parts += [f'(SELECT MIN(rowid) FROM "{t}")', f'(SELECT MAX(rowid) FROM "{t}")']
        vals = conn.execute("SELECT " + ", ".join(parts)).fetchone()
        out[dom] = tuple((t, vals[2 * i], vals[2 * i + 1]) for i, t in enumerate(tabs))
    return out


class _Probe:
    __slots__ = ("file_ver", "at", "versions")

    def __init__(self, file_ver, at, versions):
        self.file_ver, self.at, self.versions = file_ver, at, versions


_probe_lock = threading.Lock()
_probes: dict[str, _Probe] = {}


def domain_versions(path: str | Path, conn: sqlite3.Connection | None = None, *,
                    force: bool = False) -> dict[str, tuple] | None:
    """{domain: version} of a database file (None when it is missing or unreadable).

    Re-probed only when the file version changed and the last probe is at least
    PROBE_INTERVAL seconds old (or `force`); `conn` (read-only, same file) saves opening one."""
    rpath = os.path.realpath(path)
    fver = version(rpath)
    if fver is None:
        return None
    with _probe_lock:
        p = _probes.get(rpath)
        now = time.monotonic()
        if p is not None and not force and (p.file_ver == fver
                                            or now - p.at < PROBE_INTERVAL):
            return p.versions
        try:
            if conn is not None:
                vers = probe(conn)
            else:
                with ro(rpath) as c:
                    if c is None:
                        return None
                    vers = probe(c)
        except sqlite3.Error:
            return None
        _probes[rpath] = _Probe(fver, now, vers)
        return vers


# ------------------------------------------------------------------ the cache
class _Bucket:
    """One database file + one invalidation flavour: entries {key: (version, value)}; `pinned`
    keys (the warmer's shared aggregates) are skipped by the LRU eviction."""
    __slots__ = ("version", "entries", "pinned")

    def __init__(self, ver):
        self.version = ver
        self.entries: OrderedDict = OrderedDict()
        self.pinned: set = set()


_lock = threading.Lock()
_buckets: dict[tuple, _Bucket] = {}
_key_locks: dict[tuple, threading.Lock] = {}
_MISS = object()
_STALE = object()                      # stored as the version of a value built from stale input
_HIT, _OLD, _NONE = 0, 1, 2
_tls = threading.local()       # .sync (never serve stale), .pin, .frames (open computations)
_refresher: Callable[[tuple, Any, Callable], None] | None = None


def set_refresher(fn: Callable[[tuple, Any, Callable], None] | None) -> None:
    """Register (or remove, None) the background refresher: `fn(bucket, key, compute)` must
    queue `compute()` to run later in another thread. Enables stale-while-revalidate."""
    global _refresher
    _refresher = fn


@contextmanager
def sync_mode() -> Iterator[None]:
    """Inside, stale entries are recomputed in this thread instead of served (the warmer)."""
    prev = getattr(_tls, "sync", False)
    _tls.sync = True
    try:
        yield
    finally:
        _tls.sync = prev


@contextmanager
def pinning() -> Iterator[None]:
    """Inside, every domain entry looked up or stored is pinned (kept by the LRU eviction):
    the warmer pins the shared aggregates so a burst of per-player keys cannot push them out."""
    prev = getattr(_tls, "pin", False)
    _tls.pin = True
    try:
        yield
    finally:
        _tls.pin = prev


def _norm_domain(domain) -> tuple[str, ...] | None:
    if domain is None:
        return None
    doms = (domain,) if isinstance(domain, str) else tuple(domain)
    for d in doms:
        if d not in DOMAINS:
            raise ValueError(f"unknown cache domain {d!r}")
    return tuple(sorted(set(doms)))


def _lookup(bk: tuple, ver, key) -> tuple[int, Any]:
    with _lock:
        b = _buckets.get(bk)
        if b is None:
            return _NONE, None
        e = b.entries.get(key)
        if e is None:
            return _NONE, None
        b.entries.move_to_end(key)
        if bk[1] is not None and getattr(_tls, "pin", False):
            b.pinned.add(key)
        return (_HIT if e[0] == ver else _OLD), e[1]


def _store(bk: tuple, ver, key, val, *, stale: bool = False) -> None:
    with _lock:
        b = _buckets.get(bk)
        if b is None:
            b = _buckets[bk] = _Bucket(ver)
        elif b.version != ver and not stale:
            if bk[1] is None:                       # file flavour: a new version drops all
                b = _buckets[bk] = _Bucket(ver)
            else:
                b.version = ver
        b.entries[key] = (_STALE if stale else ver, val)
        b.entries.move_to_end(key)
        if bk[1] is None:
            while len(b.entries) > MAX_ENTRIES:
                b.entries.popitem(last=False)
            return
        if getattr(_tls, "pin", False):
            b.pinned.add(key)
        spins = len(b.entries)
        while len(b.entries) > MAX_DOMAIN_ENTRIES and spins > 0:
            spins -= 1
            k = next(iter(b.entries))
            if k in b.pinned:
                b.entries.move_to_end(k)
            else:
                del b.entries[k]


def unpin_all() -> None:
    """Forget every pin (the warmer re-pins what it still warms at the start of a cycle)."""
    with _lock:
        for b in _buckets.values():
            b.pinned.clear()


def _drop(bk: tuple, key) -> None:
    with _lock:
        b = _buckets.get(bk)
        if b is not None:
            b.entries.pop(key, None)
            b.pinned.discard(key)


def _mark_parent_stale() -> None:
    frames = getattr(_tls, "frames", None)
    if frames:
        frames[-1][0] = True


def _compute(bk: tuple, ver, key, fn, conn) -> Any:
    """fn(conn or a fresh ro connection) stored under `ver` (as stale when a nested cached()
    served a stale value meanwhile). Caller holds the key lock."""
    frames = getattr(_tls, "frames", None)
    if frames is None:
        frames = _tls.frames = []
    frame = [False]
    frames.append(frame)
    try:
        if conn is not None:
            val = fn(conn)
        else:
            with ro(bk[0]) as c:
                if c is None:
                    return None
                val = fn(c)
    finally:
        frames.pop()
    tainted = frame[0]
    if tainted:
        _mark_parent_stale()
        if bk[1] is None:                 # file flavour has no stale state: just don't keep it
            return val
    _store(bk, ver, key, val, stale=tainted)
    return val


def _key_lock(bk: tuple, key) -> threading.Lock:
    with _lock:
        if len(_key_locks) > 8 * MAX_ENTRIES:
            _key_locks.clear()
        return _key_locks.setdefault((bk, key), threading.Lock())


def cached(key: Any, fn: Callable[[sqlite3.Connection], Any],
           conn_or_path: sqlite3.Connection | str | Path, domain: str | Iterable[str] | None = None
           ) -> Any:
    """`fn(conn)` memoised under `key` until the database changes: the whole file (default)
    or the tables of `domain` ("box", "dfs" or a tuple), see the module docstring.

    `key` must be hashable and include every argument the result depends on (season, team...).
    With an in-memory connection or a missing file the result is computed and not cached.
    """
    conn = conn_or_path if isinstance(conn_or_path, sqlite3.Connection) else None
    path = db_file(conn) if conn is not None else str(conn_or_path)
    fver = version(path) if path else None
    if fver is None:
        if conn is not None:
            return fn(conn)
        with ro(path) as c:
            return fn(c) if c is not None else None
    rpath = os.path.realpath(path)
    dom = _norm_domain(domain)
    ver = fver
    if dom is not None:
        vers = domain_versions(rpath, conn)
        if vers is None:                       # unreadable schema: behave like the default
            dom = None
        else:
            ver = tuple(vers.get(d) for d in dom)
    bk = (rpath, dom)
    state, val = _lookup(bk, ver, key)
    if state == _HIT:
        return val
    if state == _OLD and dom is not None and _refresher is not None \
            and not getattr(_tls, "sync", False):
        _mark_parent_stale()
        _refresher(bk, key, lambda: _refresh(bk, key, fn))
        return val
    with _key_lock(bk, key):                   # one computation per key at a time
        state, val = _lookup(bk, ver, key)
        if state == _HIT:
            return val
        return _compute(bk, ver, key, fn, conn)


def _refresh(bk: tuple, key, fn) -> None:
    """Recompute one stale entry with its own connection (the background refresher's job).
    On failure the entry is dropped, so the next request computes it in the foreground."""
    rpath, dom = bk
    try:
        with sync_mode():
            vers = domain_versions(rpath)
            if vers is None:
                _drop(bk, key)
                return
            ver = tuple(vers.get(d) for d in dom)
            with _key_lock(bk, key):
                if _lookup(bk, ver, key)[0] == _HIT:
                    return
                _compute(bk, ver, key, fn, None)
    except Exception:  # noqa: BLE001 - a background refresh must never take the thread down
        _drop(bk, key)
        raise


def stats() -> dict:
    """{buckets: [{path, domain, entries, stale}], probes} (diagnostics and tests)."""
    with _lock:
        out = []
        for (p, d), b in _buckets.items():
            stale = sum(1 for v, _ in b.entries.values() if v != b.version)
            out.append({"path": p, "domain": d, "entries": len(b.entries), "stale": stale,
                        "pinned": len(b.pinned)})
    return {"buckets": out, "probes": len(_probes)}


def clear() -> None:
    """Drop every cached entry and probe (tests)."""
    with _lock:
        _buckets.clear()
        _key_locks.clear()
    with _probe_lock:
        _probes.clear()
