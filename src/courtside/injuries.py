"""Injury snapshots -> "who is out right now" and "what changed since yesterday".

`injury_snapshots` is append-only: ESPN writes one batch per pregame run (every row of a batch
shares `captured_at`), the official NBA report writes one batch per published PDF (keyed by
the publication time). A *batch* is therefore (source, captured_at).

* `current(conn, at)`   union of the newest ESPN batch and the newest official batch at/before
  `at` (the official row wins for a player on both), keeping only availability-relevant
  statuses.
* `diff(conn, at)`      newest batch vs. the batch ~24 h earlier, per source (official wins),
  split into new absences / returns / status changes. Official reports only list the teams
  playing that day, so for them only teams present in *both* batches are compared.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone

OFFICIAL = "nba_official"
ESPN = "espn"

# Normalised status -> severity. Anything not listed (Probable, Available, "") is healthy.
SEVERITY = {"out": 3, "doubtful": 2, "questionable": 1, "day-to-day": 1}
LABEL = {3: "Out", 2: "Doubtful", 1: "Questionable"}
OFFICIAL_MAX_AGE_H = 48        # an older official batch is ignored for "current"
BASELINE_TARGET_H = 24
BASELINE_MIN_GAP_H = 12        # the comparison batch must be at least this much older

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str | None) -> str:
    """'Jaren Jackson Jr.' / 'Jackson Jr., Jaren' / 'Nikola Jokić' -> 'jaren jackson'."""
    if not name:
        return ""
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def severity(status: str | None) -> int:
    return SEVERITY.get((status or "").strip().lower(), 0)


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def batches(conn: sqlite3.Connection, source: str, at: str | None = None) -> list[str]:
    """captured_at of every batch of `source` (ascending), optionally only those <= `at`."""
    sql = "SELECT DISTINCT captured_at FROM injury_snapshots WHERE source=?"
    args: list = [source]
    if at is not None:
        sql += " AND captured_at<=?"
        args.append(at)
    return [r[0] for r in conn.execute(sql + " ORDER BY captured_at", args)]


def _batch_rows(conn: sqlite3.Connection, source: str, captured_at: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT team, player_name, player_id, status, reason FROM injury_snapshots "
        "WHERE source=? AND captured_at=?", (source, captured_at))]


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(team, normalised name) -> row; the most severe row wins on a duplicate."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not r.get("player_name"):
            continue
        k = (r["team"], norm_name(r["player_name"]))
        if k not in out or severity(r["status"]) > severity(out[k]["status"]):
            out[k] = r
    return out


def _entry(r: dict, source: str, captured_at: str) -> dict:
    sev = severity(r["status"])
    return {"team": r["team"], "player": r["player_name"], "player_id": r.get("player_id"),
            "status": LABEL.get(sev, r["status"]), "severity": sev,
            "reason": (r.get("reason") or "").split("|")[0].strip() or None,
            "source": source, "captured_at": captured_at}


def current(conn: sqlite3.Connection, at: str, teams: set[str] | None = None) -> dict:
    """Relevant (Out/Doubtful/Questionable) injuries known at or before `at`.

    Returns {"by_team": {team: [entry,...]}, "batches": {source: captured_at},
    "latest_captured_at": str|None}. Entries are ordered most severe first."""
    used: dict[str, str] = {}
    merged: dict[tuple[str, str], dict] = {}
    espn_b = batches(conn, ESPN, at)
    off_b = batches(conn, OFFICIAL, at)
    if espn_b:
        used[ESPN] = espn_b[-1]
        for k, r in _index(_batch_rows(conn, ESPN, espn_b[-1])).items():
            merged[k] = _entry(r, ESPN, espn_b[-1])
    if off_b:
        age = _ts(at) - _ts(off_b[-1]) if "T" in at else timedelta(0)
        if age <= timedelta(hours=OFFICIAL_MAX_AGE_H):
            used[OFFICIAL] = off_b[-1]
            off_rows = _batch_rows(conn, OFFICIAL, off_b[-1])
            listed_teams = {r["team"] for r in off_rows}
            # the official report speaks for every team it lists: drop ESPN rows for a team it
            # covers (a player cleared there must not linger from ESPN), then add its rows.
            for k in [k for k in merged if k[0] in listed_teams]:
                del merged[k]
            for k, r in _index(off_rows).items():
                merged[k] = _entry(r, OFFICIAL, off_b[-1])
    by_team: dict[str, list[dict]] = {}
    for (team, _), e in merged.items():
        if e["severity"] < 1 or (teams is not None and team not in teams):
            continue
        by_team.setdefault(team, []).append(e)
    for lst in by_team.values():
        lst.sort(key=lambda e: (-e["severity"], e["player"]))
    return {"by_team": by_team, "batches": used,
            "latest_captured_at": max(used.values()) if used else None}


def _pick_baseline(cands: list[str], latest: str) -> str | None:
    """The batch closest to latest-24h among those at least BASELINE_MIN_GAP_H older."""
    lt = _ts(latest)
    ok = [c for c in cands if lt - _ts(c) >= timedelta(hours=BASELINE_MIN_GAP_H)]
    if not ok:
        return None
    target = lt - timedelta(hours=BASELINE_TARGET_H)
    return min(ok, key=lambda c: abs(_ts(c) - target))


def _diff_states(old: dict, new: dict, comparable_teams: set[str] | None) -> list[dict]:
    changes = []
    for k in set(old) | set(new):
        team, _ = k
        if comparable_teams is not None and team not in comparable_teams:
            continue
        o, n = old.get(k), new.get(k)
        so = severity(o["status"]) if o else 0
        sn = severity(n["status"]) if n else 0
        if so == sn:
            continue
        ref = n or o
        if so == 0:
            kind = "new_absence"
        elif sn == 0:
            kind = "return"
        else:
            kind = "status_change"
        changes.append({
            "kind": kind, "team": team, "player": ref["player"], "player_id": ref.get("player_id"),
            "from_status": LABEL.get(so, "Healthy") if o else None,
            "to_status": LABEL.get(sn, "Healthy") if n else "Healthy",
            "direction": "worse" if sn > so else "better",
            "reason": (n or o).get("reason"),
            "source": ref["source"]})
    return changes


def diff(conn: sqlite3.Connection, at: str) -> dict:
    """Changes between the newest batch at/before `at` and the batch ~24 h earlier.

    {"available": bool, "latest": ts|None, "baseline": ts|None,
     "new_absences": [...], "returns": [...], "status_changes": [...], "sources": [...]}"""
    out: dict = {"available": False, "latest": None, "baseline": None,
                 "new_absences": [], "returns": [], "status_changes": [], "sources": []}
    per_source: dict[str, list[dict]] = {}
    spans = []
    for source in (OFFICIAL, ESPN):
        b = batches(conn, source, at)
        if not b:
            continue
        latest = b[-1]
        if source == OFFICIAL and _ts(at) - _ts(latest) > timedelta(hours=OFFICIAL_MAX_AGE_H):
            continue
        base = _pick_baseline(b[:-1], latest)
        if base is None:
            continue
        def state(cap):
            rows = _batch_rows(conn, source, cap)
            idx = _index(rows)
            return {k: _entry(r, source, cap) for k, r in idx.items()}, {r["team"] for r in rows}
        new_s, new_t = state(latest)
        old_s, old_t = state(base)
        comparable = (new_t & old_t) if source == OFFICIAL else None
        per_source[source] = _diff_states(old_s, new_s, comparable)
        spans.append((source, latest, base))
    if not per_source:
        return out
    # Official wins per team where it produced a diff; ESPN fills the remaining teams.
    chosen: list[dict] = []
    covered: set[str] = set()
    if OFFICIAL in per_source:
        chosen += per_source[OFFICIAL]
        # teams the official comparison covered (even with zero changes) are not re-reported
        sp = next(s for s in spans if s[0] == OFFICIAL)
        covered = {r["team"] for r in _batch_rows(conn, OFFICIAL, sp[1])} & \
                  {r["team"] for r in _batch_rows(conn, OFFICIAL, sp[2])}
    if ESPN in per_source:
        chosen += [c for c in per_source[ESPN] if c["team"] not in covered]
    out["available"] = True
    out["sources"] = [s[0] for s in spans]
    out["latest"] = max(s[1] for s in spans)
    out["baseline"] = min(s[2] for s in spans)
    order = {"new_absences": "new_absence", "returns": "return", "status_changes": "status_change"}
    for key, kind in order.items():
        out[key] = sorted((c for c in chosen if c["kind"] == kind),
                          key=lambda c: (c["team"], c["player"]))
    return out
