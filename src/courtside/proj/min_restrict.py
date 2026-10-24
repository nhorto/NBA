"""Restricted-minutes prior (`COURTSIDE_MIN_RESTRICT`, `+rs`; IMPROVEMENTS.md P7).

A player whose latest ESPN injury comment (as known at the run time) puts him on a minutes
restriction (`sources.espn_restrict`) gets his planned minutes capped in the minutes engine:

    lim = parsed cap                                   when the comment gives a number
        = RS_FRAC x his normal minutes (0.5 ew_f + 0.5 ew_s)   otherwise

and the engine plans min(plan, lim) (`minutes.project_arrays` reads the panel's `rs_cap`
column through `base_and_pplay`). The freed minutes go to his teammates through the usual
reconcile. P(plays) is unchanged.

**No fit.** ESPN comments are archived only since 2026-10-18 and the official reports never
mention minutes, so seasons < 2026-27 hold no flagged rows. RS_FRAC is a documented rule, not a
fit; `docs/research/min-restrict.md` has the numbers.

Staleness: the flagged comment's news `date` must be within RS_MAX_AGE_H of the run time. The flag
is sticky over that window (ESPN replaces a comment with the next news, which often drops the
restriction remark): the newest flagged comment of any snapshot in the window counts, unless the
newest snapshot lists him Out or he has played a game (`player_games`, tip after the news, before
the run) since. Model version only (`+rs`, after `+tr`); same bundle and seeds. OFF
(default): no column is added and the engine is bit-identical.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

MIN_RESTRICT_ENV = "COURTSIDE_MIN_RESTRICT"
MIN_RESTRICT_DEFAULT = False
MIN_RESTRICT_TAG = "+rs"
RS_FRAC = 0.75          # flagged without a number: 75 % of his normal minutes
RS_MAX_AGE_H = 72.0     # the comment must be this fresh at the run time
RS_FLOOR = 6.0          # never plan a flagged player below this (the cap is a ceiling)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


MIN_RESTRICT = _env_flag(MIN_RESTRICT_ENV, MIN_RESTRICT_DEFAULT)


def set_min_restrict(on: bool) -> None:
    """Flip the switch (use `pipeline.set_min_restrict` to keep MODEL_VERSION in step)."""
    global MIN_RESTRICT
    MIN_RESTRICT = bool(on)


def _iso(at: datetime) -> str:
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def flagged_players(conn: sqlite3.Connection, at: datetime, candidates: pd.DataFrame
                    ) -> list[dict]:
    """ESPN-flagged candidates as known at `at`: [{key, player_id, cap, phrase, news_at,
    snapshot}] for the rows of `candidates` (columns key, player_id, team) whose team's ESPN
    entry matches by name (`injuries.match_player` on the team's candidates)."""
    from .. import injuries
    from ..sources import espn_restrict as ER
    at_u = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    since = at_u - timedelta(hours=RS_MAX_AGE_H)
    snap, rows = ER.flags_at(conn, _iso(at_u), _iso(since))
    if not rows or not len(candidates):
        return []
    pids = sorted({int(p) for p in candidates["player_id"]})
    ph = ",".join("?" * len(pids))
    names = {int(r[0]): f"{r[1] or ''} {r[2] or ''}".strip() for r in conn.execute(
        f"SELECT player_id, first_name, last_name FROM player_index WHERE player_id IN ({ph})",
        pids)}
    by_team: dict[str, list[tuple[int, str, str]]] = {}
    for k, p, t in zip(candidates["key"], candidates["player_id"], candidates["team"]):
        by_team.setdefault(str(t), []).append((int(p), names.get(int(p), ""), str(k)))
    out = []
    for r in rows:
        if (r.get("status") or "").strip().lower() == "out" or not r.get("team"):
            continue
        news = _ts(r.get("date"))
        if news is None or (at_u - news).total_seconds() / 3600.0 > RS_MAX_AGE_H:
            continue
        cand = by_team.get(r["team"]) or []
        pid, _how = injuries.match_player(r["player_name"], [(p, n) for p, n, _ in cand if n])
        if pid is None:
            continue
        if _played_since(conn, int(pid), news, at_u):
            continue
        key = next(k for p, _, k in cand if p == pid)
        out.append({"key": key, "player_id": int(pid), "cap": r.get("cap"),
                    "phrase": r.get("phrase", ""), "news_at": r.get("date"), "snapshot": snap,
                    "player_name": r["player_name"], "team": r["team"]})
    return out


def _played_since(conn: sqlite3.Connection, pid: int, news: datetime, at: datetime) -> bool:
    """Has the player logged minutes in a game that tipped after `news` and before `at`?"""
    try:
        r = conn.execute("SELECT 1 FROM player_games pg JOIN games g ON g.game_id=pg.game_id "
                         "WHERE pg.player_id=? AND pg.min>0 AND g.tipoff_utc>? AND "
                         "g.tipoff_utc<? LIMIT 1", (pid, _iso(news), _iso(at))).fetchone()
    except sqlite3.Error:
        return False
    return r is not None


def caps_for(panel: pd.DataFrame, flagged: list[dict]) -> dict:
    """{(key, player_id): lim}: the parsed cap, else RS_FRAC x 0.5 (ew_f + ew_s), floored."""
    if not flagged or not len(panel):
        return {}
    idx = {(str(k), int(p)): i for i, (k, p) in enumerate(zip(panel["key"], panel["player_id"]))}
    ew = (0.5 * panel["ew_f"].to_numpy(float) + 0.5 * panel["ew_s"].to_numpy(float)
          if "ew_f" in panel and "ew_s" in panel else np.full(len(panel), np.nan))
    out = {}
    for f in flagged:
        i = idx.get((str(f["key"]), int(f["player_id"])))
        if i is None:
            continue
        if f.get("cap") is not None:
            lim = float(f["cap"])
        elif np.isfinite(ew[i]) and ew[i] > 0:
            lim = RS_FRAC * float(ew[i])
        else:
            continue
        out[(str(f["key"]), int(f["player_id"]))] = max(lim, RS_FLOOR)
    return out


def restrict_caps(conn: sqlite3.Connection, panel: pd.DataFrame, at: datetime | None
                  ) -> dict:
    """{(key, player_id): planned-minutes cap} for the candidate panel as known at `at`
    ({} when `at` is None or nobody is flagged)."""
    if at is None or not len(panel):
        return {}
    return caps_for(panel, flagged_players(conn, at, panel))


def rs_column(panel: pd.DataFrame, caps: dict) -> np.ndarray:
    """The panel's `rs_cap` column (NaN = not restricted)."""
    return np.array([caps.get((str(k), int(p)), np.nan)
                     for k, p in zip(panel["key"], panel["player_id"])], float)
