"""Starters from the archived cdn boxscore -> ``player_starts`` (box score page, stats site).

Input is ``raw/nba_cdn/boxscore/<nba_day>/<game_id>.json.gz``: every player of ``game.homeTeam`` /
``game.awayTeam`` carries ``starter`` ("1" / "0"). Rows are ``(game_id, player_id, team, 'cdn')``
with ``team`` the boxscore ``teamTricode``. ``source = 'cdn'`` sits next to the older ``sdv`` and
``proxy`` rows (``proj/starters.py``): the short tag of the provider.

A game is stored only when it is final and both teams have exactly 5 players flagged (a stub
box has none). ``store_game`` replaces the game's rows (whatever their
source) in one transaction, so reruns are idempotent. ``backfill`` skips games that already have
``cdn`` rows unless ``force``; it needs no network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from courtside import archive, config

SOURCE_TAG = "cdn"          # `player_starts.source`
ARCHIVE_SOURCE = "nba_cdn"
KIND = "boxscore"
NOT_FINAL = {"1", "2", "inprogress", "periodbreak", "pregame"}   # as in stints_store


def parse_starters(box: dict) -> list[tuple[int, str]]:
    """[(player_id, team_tricode)] for the flagged starters of a final boxscore. Empty for a game
    that is not final or where either team does not have exactly 5 flagged."""
    g = (box or {}).get("game") or {}
    if str(g.get("gameStatus")).lower() in NOT_FINAL:
        return []
    out: list[tuple[int, str]] = []
    for side in ("homeTeam", "awayTeam"):
        t = g.get(side) or {}
        tri = t.get("teamTricode")
        st = [int(p["personId"]) for p in t.get("players") or []
              if str(p.get("starter")) == "1" and p.get("personId")]
        if not tri or len(set(st)) != 5:
            return []
        out.extend((pid, tri) for pid in sorted(set(st)))
    return out


def store_game(conn: sqlite3.Connection, game_id: str, nba_day: str, box: dict | None = None
               ) -> int:
    """Replace one game's ``player_starts`` from its archived boxscore. Returns the rows stored
    (0 leaves existing rows alone: nothing trustworthy to replace them with)."""
    if box is None:
        box = archive.load_raw(archive.raw_path(ARCHIVE_SOURCE, KIND, nba_day, game_id))
    rows = parse_starters(box)
    if not rows:
        return 0
    with conn:
        conn.execute("DELETE FROM player_starts WHERE game_id=?", (game_id,))
        conn.executemany(
            "INSERT INTO player_starts(game_id, player_id, team, source) VALUES (?,?,?,?)",
            [(game_id, pid, tri, SOURCE_TAG) for pid, tri in rows])
    return len(rows)


def stored_games(conn: sqlite3.Connection, game_ids: Iterable[str] | None = None) -> set[str]:
    """Games that already have ``cdn`` rows."""
    q = "SELECT DISTINCT game_id FROM player_starts WHERE source=?"
    p: list = [SOURCE_TAG]
    if game_ids is not None:
        ids = list(game_ids)
        if not ids:
            return set()
        q += f" AND game_id IN ({','.join('?' * len(ids))})"
        p += ids
    return {r[0] for r in conn.execute(q, p)}


def list_archived(seasons: Iterable[str] | str | None = None) -> list[tuple[str, str, Path]]:
    """[(game_id, nba_day, path)] of every archived boxscore, sorted by (day, game_id)."""
    from .shots_store import season_of
    if isinstance(seasons, str):
        seasons = [seasons]
    want = set(seasons) if seasons else None
    root = config.data_dir() / "raw" / ARCHIVE_SOURCE / KIND
    out = []
    if not root.is_dir():
        return out
    for dday in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(dday.glob("*.json.gz")):
            gid = f.name[: -len(".json.gz")]
            if len(gid) != 10 or not gid.isdigit():
                continue
            if want is not None and season_of(gid) not in want:
                continue
            out.append((gid, dday.name, f))
    return out


def backfill(conn: sqlite3.Connection, seasons: Iterable[str] | str | None = None, *,
             force: bool = False) -> dict:
    """Fill ``player_starts`` from every archived boxscore of ``seasons`` (offline). Games that
    already have ``cdn`` rows are skipped unless ``force``. Games whose box is not final or has
    no 5-man starter set are listed in ``unusable``; unreadable files in ``failed``."""
    games = list_archived(seasons)
    have = set() if force else stored_games(conn, [g[0] for g in games])
    s: dict = {"archived": len(games), "skipped": 0, "stored_games": 0, "rows": 0,
               "unusable": [], "failed": {}}
    for gid, day, path in games:
        if gid in have:
            s["skipped"] += 1
            continue
        try:
            n = store_game(conn, gid, day, archive.load_raw(path))
        except Exception as e:  # noqa: BLE001 - unreadable file: report, keep going
            s["failed"][gid] = f"{type(e).__name__}: {e}"
            continue
        if not n:
            s["unusable"].append(gid)
            continue
        s["stored_games"] += 1
        s["rows"] += n
    return s
