"""Shots store: one row per field-goal attempt from the cdn play-by-play archive (shot charts).

Input is ``raw/nba_cdn/playbyplay/<nba_day>/<game_id>.json.gz`` (archived nightly by postgame,
historically by ``courtside backfill pbp``). ``parse_shots`` is pure; ``store_game`` /
``backfill`` write the ``shots`` table (migration 17).

Coordinates
-----------
``x`` / ``y`` are the feed's ``xLegacy`` / ``yLegacy``: tenths of a foot from the centre of the
rim, the stats.nba.com shot-chart frame. ``y`` grows towards half court (the baseline is at
y = -52.5, the half-court line at y = 417.5); ``x`` < 0 is the shooter's LEFT side as the NBA
labels it (every 'Left Corner 3' in the feed has x < 0, every 'Right Corner 3' x > 0; checked on
2,000+ corner threes of 2025-26). ``hypot(x, y) / 10`` equals ``shotDistance`` to 0.1 ft.

Zones (computed here, not taken from the feed's ``area``: it is missing before 2023-24 and wrong
for heaves, e.g. a 62-ft three labelled 'Mid-Range'):

  Backcourt        y > 417.5 (beyond half court), any shot type
  Left/Right Corner 3   a three with y <= 89.5 (where the 22-ft corner line meets the 23.75-ft
                   arc); side from the sign of x
  Above-break 3    every other three
  Restricted area  a two within RA_RADIUS of the rim. The painted arc is 4 ft, but the NBA's own
                   labels (the feed's ``area``) put the boundary at ~4.4 ft (measured on 25k
                   2024-26 paint twos: 4.4 ft agrees 99.6%, 4.0 ft 96.4%), so 4.4 ft is used and
                   zone splits match nba.com
  Paint non-RA     a two inside the lane: |x| < 80 (16-ft lane) and y < 137.5 (free-throw line)
  Mid-range        every other two

``shot_type`` ('2pt' | '3pt') is the official ruling (``actionType``), never the geometry.

Older dialects (handled): 2019-20 payloads have no ``isFieldGoal`` (``actionType`` 2pt/3pt is
used), and Feb-Mar 2020 payloads write ``shotResult`` as ': Made'.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from courtside import archive, config

SOURCE = "nba_cdn"
KIND = "playbyplay"
BATCH = 50   # games per commit in backfill

ZONES = ("Restricted Area", "Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3",
         "Above the Break 3", "Backcourt")
RA_RADIUS = 44.0          # tenths of a foot (see the module docstring)
LANE_HALF_WIDTH = 80.0
FT_LINE_Y = 137.5
CORNER_MAX_Y = 89.5
HALF_COURT_Y = 417.5

SEASON_TYPES = {"001": "Pre Season", "002": "Regular Season", "003": "All Star",
                "004": "Playoffs", "005": "PlayIn", "006": "NBA Cup"}

COLS = ("game_id", "action_number", "player_id", "team", "season", "season_type", "game_date",
        "period", "clock", "x", "y", "distance", "made", "shot_type", "action_type", "sub_type",
        "descriptor", "zone")

_CLOCK = re.compile(r"PT(?:(\d+)M)?(?:([\d.]+)S)?")


def season_of(game_id: str) -> str | None:
    """'0022600001' -> '2026-27'."""
    try:
        yy = int(str(game_id)[3:5])
    except ValueError:
        return None
    return f"{2000 + yy}-{(yy + 1) % 100:02d}"


def season_type_of(game_id: str) -> str | None:
    return SEASON_TYPES.get(str(game_id)[:3])


def fmt_clock(clock: str | None) -> str | None:
    """'PT11M39.00S' -> '11:39'; under a minute keeps tenths: 'PT00M05.40S' -> '0:05.4'."""
    if not clock:
        return None
    m = _CLOCK.fullmatch(str(clock).strip())
    if not m:
        return str(clock)
    mins = int(m[1] or 0)
    secs = float(m[2] or 0)
    if mins == 0 and secs < 60 and secs != int(secs):
        return f"0:{secs:04.1f}"
    return f"{mins}:{int(secs):02d}"


def zone_for(x: float | None, y: float | None, shot_type: str) -> str | None:
    """Zone of a shot at legacy (x, y) (tenths of a foot from the rim). None without coordinates."""
    if x is None or y is None:
        return None
    if y > HALF_COURT_Y:
        return "Backcourt"
    if shot_type == "3pt":
        if y <= CORNER_MAX_Y:
            return "Left Corner 3" if x < 0 else "Right Corner 3"
        return "Above the Break 3"
    if math.hypot(x, y) <= RA_RADIUS:
        return "Restricted Area"
    if abs(x) < LANE_HALF_WIDTH and y < FT_LINE_Y:
        return "Paint (Non-RA)"
    return "Mid-Range"


def _num(v) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _is_fga(a: dict) -> bool:
    if "isFieldGoal" in a:
        return a.get("isFieldGoal") == 1
    return a.get("actionType") in ("2pt", "3pt")      # 2019-20 dialect


def parse_shots(pbp: dict, game_id: str | None = None, game_date: str | None = None
                ) -> list[dict]:
    """``shots`` rows (dicts keyed by COLS) for every field-goal attempt of one game. Pure."""
    game = (pbp or {}).get("game") or {}
    gid = str(game_id or game.get("gameId") or "")
    season, stype = season_of(gid), season_type_of(gid)
    out: dict[int, dict] = {}
    for a in game.get("actions") or []:
        if not _is_fga(a) or a.get("actionNumber") is None:
            continue
        at = str(a.get("actionType") or "")
        shot_type = "3pt" if at == "3pt" else "2pt"
        x, y = _num(a.get("xLegacy")), _num(a.get("yLegacy"))
        dist = _num(a.get("shotDistance"))
        if dist is None and x is not None and y is not None:
            dist = round(math.hypot(x, y) / 10.0, 2)
        res = str(a.get("shotResult") or "").lstrip(": ").strip().lower()
        made = 1 if res == "made" else 0 if res == "missed" else None
        an = int(a["actionNumber"])
        out[an] = {
            "game_id": gid, "action_number": an, "player_id": a.get("personId") or None,
            "team": a.get("teamTricode") or None, "season": season, "season_type": stype,
            "game_date": game_date, "period": a.get("period"), "clock": fmt_clock(a.get("clock")),
            "x": x, "y": y, "distance": dist, "made": made, "shot_type": shot_type,
            "action_type": at or None, "sub_type": a.get("subType") or None,
            "descriptor": a.get("descriptor") or None, "zone": zone_for(x, y, shot_type)}
    return [out[k] for k in sorted(out)]


# ------------------------------------------------------------------------------------- DB

_INSERT = (f"INSERT OR REPLACE INTO shots ({','.join(COLS)}) "
           f"VALUES ({','.join(':' + c for c in COLS)})")


def write_game(conn: sqlite3.Connection, game_id: str, rows: list[dict]) -> int:
    """Replace one game's shots (no commit: the caller owns the transaction)."""
    conn.execute("DELETE FROM shots WHERE game_id=?", (game_id,))
    conn.executemany(_INSERT, rows)
    return len(rows)


def store_game(conn: sqlite3.Connection, game_id: str, nba_day: str, pbp: dict | None = None
               ) -> int:
    """Parse one archived game (loaded from the archive when ``pbp`` is None) and replace its
    rows in one transaction. Returns the number of shots stored."""
    if pbp is None:
        pbp = archive.load_raw(archive.raw_path(SOURCE, KIND, nba_day, game_id))
    rows = parse_shots(pbp, game_id, str(nba_day))
    with conn:
        write_game(conn, game_id, rows)
    return len(rows)


def stored_games(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM shots")}


def list_archived(seasons: Iterable[str] | str | None = None) -> list[tuple[str, str, Path]]:
    """[(game_id, nba_day, path)] of every archived play-by-play, sorted by (day, game_id)."""
    if isinstance(seasons, str):
        seasons = [seasons]
    want = set(seasons) if seasons else None
    root = config.data_dir() / "raw" / SOURCE / KIND
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
             force: bool = False, limit: int | None = None, progress: bool = False) -> dict:
    """Fill ``shots`` from every archived play-by-play (no network). Games that already have
    rows are skipped unless ``force``; commits every BATCH games, so a killed run resumes."""
    t0 = time.monotonic()
    games = list_archived(seasons)
    have = set() if force else stored_games(conn)
    todo = [g for g in games if g[0] not in have]
    if limit is not None:
        todo = todo[:limit]
    s: dict = {"archived": len(games), "skipped": sum(g[0] in have for g in games),
               "attempted": 0, "stored_games": 0, "shots": 0, "empty": [], "failed": {},
               "seconds": 0.0}
    pending = 0
    for gid, day, path in todo:
        s["attempted"] += 1
        try:
            rows = parse_shots(archive.load_raw(path), gid, day)
        except Exception as e:  # noqa: BLE001 - unreadable file: report, keep going
            s["failed"][gid] = f"{type(e).__name__}: {e}"
            continue
        if not rows:
            s["empty"].append(gid)
            continue
        write_game(conn, gid, rows)
        s["stored_games"] += 1
        s["shots"] += len(rows)
        pending += 1
        if pending >= BATCH:
            conn.commit()
            pending = 0
        if progress and s["attempted"] % 500 == 0:
            print(f"  shots: {s['attempted']}/{len(todo)} games, {s['shots']} shots, "
                  f"{time.monotonic() - t0:.0f}s", flush=True)
    conn.commit()
    s["seconds"] = round(time.monotonic() - t0, 1)
    return s
