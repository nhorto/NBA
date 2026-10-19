"""Read stored Briefs for the web app: `$COURTSIDE_DATA/briefs/*.json` plus the `briefs` table.

Parsed documents are cached by file mtime/size (files) or `built_at` (SQLite), so a page render
never re-parses an unchanged Brief. The database is opened read-only per call and never
migrated, so the web process is safe to run while collectors write.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

from .. import config

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BriefStore:
    def __init__(self, data_dir: str | Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else None
        self._lock = threading.Lock()
        self._files: dict[str, tuple[tuple[float, int], dict]] = {}
        self._rows: dict[str, tuple[str, dict]] = {}
        self._derived: dict[tuple, object] = {}

    # ---------------------------------------------------------------- locations
    @property
    def data_dir(self) -> Path:
        return self._data_dir or config.data_dir()

    @property
    def briefs_dir(self) -> Path:
        return self.data_dir / "briefs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "courtside.db"

    def _db(self) -> sqlite3.Connection | None:
        p = self.db_path
        if not p.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
            conn.execute("SELECT 1 FROM briefs LIMIT 1")
            return conn
        except sqlite3.Error:
            return None

    def health_conn(self) -> sqlite3.Connection | None:
        """Read-only connection for the Model Health page (monitoring tables, no `briefs`
        needed). None when there is no database yet."""
        p = self.db_path
        if not p.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error:
            return None

    # ---------------------------------------------------------------- listing
    def _file_days(self) -> dict[str, Path]:
        d = self.briefs_dir
        if not d.is_dir():
            return {}
        return {p.stem: p for p in d.glob("*.json") if _DAY_RE.match(p.stem)}

    def _db_days(self) -> dict[str, str]:
        conn = self._db()
        if conn is None:
            return {}
        try:
            return {r[0]: r[1] for r in conn.execute("SELECT nba_day, built_at FROM briefs")
                    if r[0] and _DAY_RE.match(r[0])}
        except sqlite3.Error:
            return {}
        finally:
            conn.close()

    def days(self) -> list[str]:
        """All days with a stored Brief, ascending."""
        return sorted(set(self._file_days()) | set(self._db_days()))

    def latest_day(self) -> str | None:
        ds = self.days()
        return ds[-1] if ds else None

    def neighbours(self, day: str) -> tuple[str | None, str | None]:
        ds = self.days()
        prev = max((d for d in ds if d < day), default=None)
        nxt = min((d for d in ds if d > day), default=None)
        return prev, nxt

    # ---------------------------------------------------------------- loading
    def _load_file(self, day: str, path: Path) -> dict | None:
        try:
            st = path.stat()
        except OSError:
            return None
        key = (st.st_mtime, st.st_size)
        hit = self._files.get(day)
        if hit and hit[0] == key:
            return hit[1]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict):
            return None
        with self._lock:
            self._files[day] = (key, doc)
        return doc

    def _load_row(self, day: str, built_at: str) -> dict | None:
        hit = self._rows.get(day)
        if hit and hit[0] == built_at:
            return hit[1]
        conn = self._db()
        if conn is None:
            return None
        try:
            r = conn.execute("SELECT brief_json FROM briefs WHERE nba_day=?", (day,)).fetchone()
        except sqlite3.Error:
            r = None
        finally:
            conn.close()
        if not r:
            return None
        try:
            doc = json.loads(r[0])
        except ValueError:
            return None
        with self._lock:
            self._rows[day] = (built_at, doc)
        return doc

    def get(self, day: str) -> dict | None:
        """The freshest stored Brief for `day` (file or DB row, whichever was built later)."""
        if not day or not _DAY_RE.match(day):
            return None
        path = self.briefs_dir / f"{day}.json"
        doc = self._load_file(day, path) if path.exists() else None
        db_built = self._db_days().get(day)
        if db_built:
            file_built = ((doc or {}).get("meta") or {}).get("built_at") or ""
            if doc is None or db_built > file_built:
                row = self._load_row(day, db_built)
                if row is not None:
                    return row
        return doc

    def version(self, day: str) -> tuple:
        """Cache key for derived data of a day (changes whenever the Brief changes)."""
        doc = self.get(day)
        meta = (doc or {}).get("meta") or {}
        return (day, meta.get("built_at"), id(doc))

    # ---------------------------------------------------------------- derived cache
    def derived(self, key: tuple, build):
        """Memoise `build()` under `key` (callers include `version(day)` in the key)."""
        hit = self._derived.get(key)
        if hit is not None:
            return hit
        val = build()
        with self._lock:
            if len(self._derived) > 256:
                self._derived.clear()
            self._derived[key] = val
        return val

    def find_game(self, game_id: str) -> tuple[str, str, dict] | None:
        """(day, 'last_night'|'tonight', game) for a game id, newest Brief first.

        A finished game (with facts) wins over a preview of the same game.
        """
        days = self.days()
        key = ("game-index", tuple((d, self.version(d)[1]) for d in days))

        def build():
            idx: dict[str, tuple[str, str]] = {}
            for d in days:  # ascending; later Briefs overwrite, last_night beats tonight
                b = self.get(d) or {}
                for g in ((b.get("tonight") or {}).get("games") or []):
                    gid = g.get("game_id")
                    if gid and (gid not in idx or idx[gid][1] == "tonight"):
                        idx[gid] = (d, "tonight")
                for g in ((b.get("last_night") or {}).get("games") or []):
                    if g.get("game_id"):
                        idx[g["game_id"]] = (d, "last_night")
            return idx

        idx = self.derived(key, build)
        hit = idx.get(game_id)
        if not hit:
            return None
        day, where = hit
        b = self.get(day) or {}
        for g in ((b.get(where) or {}).get("games") or []):
            if g.get("game_id") == game_id:
                return day, where, g
        return None
