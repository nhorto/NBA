"""News latency (IMPROVEMENTS.md P7 part B; display only): for each injury news event, the minutes
from the news to our next projection re-run of that game, and to lock.

Read-only on the tracking tables, tolerant of missing tables:

* **News events** (`injury_snapshots`, captured on or after `since`):
  - official report (`nba_official`): a player's status differs from his status in that team's
    previous report (or he is newly listed). News time = the report's time (`captured_at`, the
    report's nominal ET time; the PDF lands a few minutes later).
  - ESPN (`espn`): a player's status differs from the previous ESPN capture. News time = the first
    capture that shows it (so this is the time we *saw* it; ESPN's own publish time is in the raw
    comment `date` only, see `research/min_restrict_eval.py`).
  The first batch of each source in the window is a baseline, not news.
* **Game** of an event: the team's next game (`games`) with tip after the news and within
  MAX_AHEAD_H. Events with no such game are dropped (long-term injuries between games).
* **Re-run**: the first `dfs_runs` row (ok, with a model version, i.e. it re-projected) of a slate
  that holds the game, started after the news could be seen (a run that started earlier could not
  see it): official reports OFFICIAL_LAG after their nominal time (the PDF lands a few minutes
  later), ESPN captures at once (5 s slack for a capture made by the run's own tick); and
  finished (created_at + seconds) by tip; the latency is to its finish. None = missed.
* **Lock**: the game's tip (the late-swap lock). Games not tipped by `as_of` (default: the newest
  injury capture) are pending and left out of the summary.
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
import pandas as pd

SEASON_START = "2026-10-01T00:00:00Z"
MAX_AHEAD_H = 30.0
SAME_TICK = pd.Timedelta(seconds=5)    # an ESPN capture stamped by the run's own collect step
OFFICIAL_LAG = pd.Timedelta(minutes=5)  # the report's PDF lands a few minutes after its time
SOURCES = ("nba_official", "espn")
SOURCE_LABEL = {"nba_official": "Official report", "espn": "ESPN"}


def _ts(s) -> pd.Timestamp | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return None
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _read(conn, sql: str, args: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=args)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def _status_changes(snap: pd.DataFrame) -> pd.DataFrame:
    """Per source: rows whose status differs from the player's previous batch (team-scoped for
    the official report, whose batches list only the teams with a game)."""
    out = []
    for src, s in snap.groupby("source", sort=False):
        s = s.sort_values("captured_at")
        batches = s["captured_at"].unique()
        if len(batches) < 2:
            continue
        first = batches[0]
        last: dict = {}
        team_seen: dict = {}
        for cap, b in s.groupby("captured_at", sort=True):
            teams_now = set(b["team"])
            for r in b.itertuples(index=False):
                k = (r.team, r.player_name)
                prev = last.get(k)
                if cap != first and prev != r.status and (prev is not None or
                                                          r.team in team_seen):
                    out.append({"source": src, "news_at": cap, "team": r.team,
                                "player_name": r.player_name, "old": prev, "new": r.status})
                last[k] = r.status
            for t in teams_now:
                team_seen[t] = cap
    return pd.DataFrame(out, columns=["source", "news_at", "team", "player_name", "old", "new"])


def events(conn: sqlite3.Connection | None, since: str = SEASON_START,
           as_of: str | None = None) -> pd.DataFrame:
    """One row per news event with its game, tip, next re-run and the latencies (minutes);
    `pending` = the game had not tipped by `as_of`."""
    cols = ["source", "news_at", "team", "player_name", "old", "new", "game_id", "tip",
            "rerun_at", "rerun_kind", "to_rerun_min", "to_lock_min", "missed", "pending"]
    if conn is None:
        return pd.DataFrame(columns=cols)
    snap = _read(conn, "SELECT captured_at, source, team, player_name, status FROM "
                       "injury_snapshots WHERE captured_at>=? AND source IN (?, ?)",
                 (since, *SOURCES))
    games = _read(conn, "SELECT game_id, tipoff_utc, home, away FROM games WHERE tipoff_utc>=?",
                  (since,))
    if not len(snap) or not len(games):
        return pd.DataFrame(columns=cols)
    snap["status"] = snap["status"].fillna("")
    cut = _ts(as_of) if as_of else _ts(snap["captured_at"].max())
    ev = _status_changes(snap)
    if not len(ev):
        return pd.DataFrame(columns=cols)
    slates = _read(conn, "SELECT slate_id, game_ids_json FROM dfs_slates")
    runs = _read(conn, "SELECT slate_id, kind, created_at, seconds FROM dfs_runs WHERE ok=1 AND "
                       "model_version IS NOT NULL AND model_version!='' AND created_at>=?",
                 (since,))
    game_slates: dict[str, set] = {}
    for r in slates.itertuples(index=False):
        try:
            gids = json.loads(r.game_ids_json or "[]")
        except ValueError:
            gids = []
        for g in gids:
            game_slates.setdefault(str(g), set()).add(str(r.slate_id))
    if len(runs):
        runs["start"] = [_ts(c) for c in runs["created_at"]]
        runs["done"] = [(t + pd.Timedelta(seconds=float(s or 0.0)))
                        for t, s in zip(runs["start"], runs["seconds"])]
        runs["slate_id"] = runs["slate_id"].astype(str)
    games["tip"] = [_ts(t) for t in games["tipoff_utc"]]
    by_team: dict[str, list] = {}
    for g in games.itertuples(index=False):
        if g.tip is None:
            continue
        for t in (g.home, g.away):
            by_team.setdefault(t, []).append((g.tip, g.game_id))
    for t in by_team:
        by_team[t].sort()
    rows = []
    for e in ev.itertuples(index=False):
        news = _ts(e.news_at)
        nxt = next(((tip, gid) for tip, gid in by_team.get(e.team, []) if tip > news), None)
        if nxt is None or (nxt[0] - news).total_seconds() / 3600.0 > MAX_AHEAD_H:
            continue
        tip, gid = nxt
        rr = None
        if len(runs):
            sl = game_slates.get(str(gid), set())
            seen = news + OFFICIAL_LAG if e.source == "nba_official" else news - SAME_TICK
            cand = runs[runs["slate_id"].isin(sl) & (runs["start"] >= seen)
                        & (runs["done"] <= tip)]
            if len(cand):
                rr = cand.loc[cand["done"].idxmin()]
        rows.append({"source": e.source, "news_at": e.news_at, "team": e.team,
                     "player_name": e.player_name, "old": e.old if isinstance(e.old, str) else None,
                     "new": e.new, "game_id": gid,
                     "tip": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "rerun_at": rr["done"].strftime("%Y-%m-%dT%H:%M:%SZ") if rr is not None
                     else None, "rerun_kind": rr["kind"] if rr is not None else None,
                     "to_rerun_min": (rr["done"] - news).total_seconds() / 60.0
                     if rr is not None else np.nan,
                     "to_lock_min": (tip - news).total_seconds() / 60.0,
                     "missed": rr is None and tip <= cut, "pending": tip > cut})
    return pd.DataFrame(rows, columns=cols)


def _q(x: pd.Series, q: float) -> float | None:
    x = x.dropna()
    return float(np.quantile(x, q)) if len(x) else None


def summary(ev: pd.DataFrame) -> dict:
    """Per source and overall: n, share missed (no re-run before tip), quantiles (p25 / p50 /
    p75 / p90) of news -> re-run and news -> lock, and the share re-run within 30 / 60 min."""
    out = {"pending": int(ev["pending"].sum()) if "pending" in ev else 0}
    if "pending" in ev:
        ev = ev[~ev["pending"].astype(bool)]
    groups = [("all", ev)] + [(s, ev[ev["source"] == s]) for s in SOURCES]
    for lab, d in groups:
        if not len(d):
            out[lab] = {"n": 0}
            continue
        r, lk = d["to_rerun_min"], d["to_lock_min"]
        out[lab] = {"n": int(len(d)), "missed": int(d["missed"].sum()),
                    "missed_share": float(d["missed"].mean()),
                    "rerun_q": [_q(r, q) for q in (0.25, 0.5, 0.75, 0.9)],
                    "lock_q": [_q(lk, q) for q in (0.25, 0.5, 0.75, 0.9)],
                    "within_30": float((r <= 30).sum() / len(d)),
                    "within_60": float((r <= 60).sum() / len(d)),
                    "late_news_60": int((lk <= 60).sum())}
    return out

