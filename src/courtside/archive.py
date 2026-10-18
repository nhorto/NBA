"""Raw response archive: $COURTSIDE_DATA/raw/{source}/{kind}/{nba_day}/{key}.json.gz"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from . import config


def _stamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("snapshot time must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def raw_path(source: str, kind: str, nba_day: date | str, key: str, *,
             snapshot_at: datetime | None = None, ext: str = ".json.gz") -> Path:
    name = f"{key}__{_stamp(snapshot_at)}" if snapshot_at is not None else key
    return config.data_dir() / "raw" / source / kind / str(nba_day) / f"{name}{ext}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def save_raw(source: str, kind: str, nba_day: date | str, key: str, payload: Any, *,
             snapshot: bool = False, now: datetime | None = None) -> Path:
    """Gzip-JSON `payload` atomically. snapshot=True appends a UTC timestamp to the name."""
    at = (now or datetime.now(timezone.utc)) if snapshot else None
    path = raw_path(source, kind, nba_day, key, snapshot_at=at)
    _atomic_write(path, gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    return path


def save_raw_bytes(source: str, kind: str, nba_day: date | str, key: str, data: bytes, *,
                   snapshot: bool = False, now: datetime | None = None,
                   ext: str = ".pdf") -> Path:
    """Write raw bytes (e.g. PDF) without gzip."""
    at = (now or datetime.now(timezone.utc)) if snapshot else None
    path = raw_path(source, kind, nba_day, key, snapshot_at=at, ext=ext)
    _atomic_write(path, data)
    return path


def load_raw(path: str | Path) -> Any:
    """Load a saved raw file: .gz -> parsed JSON, otherwise raw bytes."""
    path = Path(path)
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return path.read_bytes()


def exists_final(source: str, kind: str, nba_day: date | str, key: str) -> bool:
    return raw_path(source, kind, nba_day, key).exists()
