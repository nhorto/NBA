"""Stats-site backfill for one season (docs/STATS_SITE.md "Data"): everything the nightly postgame
now fills per game / per day, for the games already played. Resumable and idempotent: each part
skips what is already stored (shots, stints) or archived (matchups, season tables).

  1. shots           from the archived cdn play-by-play (no network)
  2. stints          from the archived cdn play-by-play + boxscore (no network)
  3. starts          player_starts (source 'cdn') from the archived cdn boxscore (no network)
  4. shot_xfg        refit of the season's expected-FG% table from `shots` (no network)
  5. player_matchups, player_tracking, player_hustle
                     boxscorematchupsv3 / boxscoreplayertrackv3 / boxscorehustlev2 per game
                     (stats.nba.com, 1 request per game and kind not yet archived; the
                     backfill_players Guard backs off on blocks). Offline: archived games only.
  6. player_tracking_ext  leaguedashptstats per day (Passing / Drives / Possessions: 3 requests
                     a day not yet archived, ~600 for a full past season). Offline: archive only.
  7. synergy_playtypes + season_matchups (stats.nba.com, 41 requests per season type)

Usage: ``courtside backfill stats --season 2026-27``.
"""

from __future__ import annotations

from datetime import date

from . import backfill, backfill_players as bp, config, db
from .analytics import shots_store, starts_store, stints_store, tracking_ext_store, xfg_store
from .sources import nba_players as np_


def season_types_for(season: str, today: date | None = None) -> tuple[str, ...]:
    """Regular season, plus Playoffs once a season's playoffs can have started (or are over)."""
    today = today or config.today_nba_day()
    end_year = int(season[:4]) + 1
    if today.year > end_year or (today.year == end_year and today.month >= 4):
        return ("Regular Season", "Playoffs")
    return ("Regular Season",)


def run(season: str, *, refresh: bool = False, limit: int | None = None,
        offline: bool = False, conn=None) -> dict:
    backfill.seasons_range(season, season)          # validates the label
    own = conn is None
    conn = conn or db.connect()
    out: dict = {"season": season}
    try:
        out["shots"] = {k: v for k, v in shots_store.backfill(conn, season, progress=True).items()
                        if k != "empty"}
        st = stints_store.build(conn, season, progress=True)
        out["stints"] = {"stored": st["validated"], "already": st["skipped"],
                         "failed": [f["game_id"] for f in st["failures"]]}
        sr = starts_store.backfill(conn, season)
        out["starts"] = {"stored_games": sr["stored_games"], "already": sr["skipped"],
                         "unusable": sr["unusable"][:10], "failed": sr["failed"]}
        x = xfg_store.fit_season(conn, season)
        out["shot_xfg"] = {"fga": x["fga"], "cells": x["cells"]}
        guard = bp.Guard()
        aborted = False
        for name in ("matchups", "tracking", "hustle"):
            if aborted:
                break
            # offline: limit 0 = parse what is archived, never fetch
            g = bp.run_per_game(name, season, season, conn=conn, guard=guard,
                                limit=0 if offline else limit)
            out[name] = {**g["by_kind"], "aborted": g["aborted"], "errors": g["errors"][:10]}
            aborted = g["aborted"]
        if not aborted:
            from .trends.common import season_of
            tx = tracking_ext_store.backfill(
                conn, season, fetch=lambda *a: guard.call(np_.fetch_pt_day, *a),
                offline=offline, limit=limit,
                current_season=season_of(config.today_nba_day()), on_abort=(bp.Aborted,))
            out["tracking_ext"] = {**tx, "pending": tx["pending"][:10]}
            aborted = tx["aborted"]
        if not offline:
            if not aborted:
                s = bp.run_season_tables(season, season, season_types=season_types_for(season),
                                         tables=("synergy", "lsm"), force=refresh, conn=conn)
                out["season_tables"] = {k: s[k] for k in ("fetched", "cached", "failed",
                                                          "aborted", "rows")}
    finally:
        if own:
            conn.close()
    return out


def ok(summary: dict) -> bool:
    t = summary.get("season_tables") or {}
    if (summary["shots"]["failed"] or summary["starts"]["failed"] or t.get("failed")
            or t.get("aborted") or (summary.get("tracking_ext") or {}).get("aborted")
            or (summary.get("tracking_ext") or {}).get("failed")):
        return False
    for name in ("matchups", "tracking", "hustle"):
        m = summary.get(name) or {}
        if m.get("aborted") or any(isinstance(v, dict) and v.get("failed") for v in m.values()):
            return False
    return True
