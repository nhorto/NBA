"""P(play) recalibration (docs/PLAYERS.md §4.1, review item M6, T1.3).

The hand-set `injuries.PLAY_PROB` map says "Questionable = 0.5" whatever the report time, the
reason or the player.  This module replaces it with Beta-binomial cells learned from the rescued
official NBA injury reports (151 game days, 2025-12-22 -> 2026-06-13) against who actually
played (`player_games.min > 0`; a player with no box row was inactive = did not play).

Model (one hierarchy per *status*; every level shrinks to its parent):

    L1  status                                 prior = injuries.play_probability(status), m1 pseudo-obs
    L2  status x hours-to-tip bucket           (of the report used)      parent = L1 posterior
    L3  status x bucket x reason class         (injury / management / illness / rest / ...)
    L4  status x bucket x reason x role        (starter / rotation / bench by recent minutes)

    posterior(cell) = (k + m * parent) / (n + m)        k = games played, n = games

so a sparse cell is its parent, a dense one is its own rate.  Statuses are the report's own
(out, doubtful, questionable, probable, available) plus two for a player the report does *not*
list: `unlisted` (his team has a report on file: P ~ 0.97) and `unlisted_noreport` (no report for
the team yet).  ESPN's "Day-To-Day" keeps the hand-set 0.6 (no ESPN history to fit).
Levels 1-2 are made monotone across the five report statuses (Out <= Doubtful <= Questionable
<= Probable <= Available) by weighted pool-adjacent-violators, so the status x bucket table
cannot invert; reason/role cells with enough data may still disagree (an 'Available' G-League
assignee plays about half the time).

Decision time: `p_play*` use the LATEST report published at or before `at` for the team's game
day (ET) -- never a later one -- and fall back to an older official report, then ESPN, then the
prior.  The hours-to-tip bucket is that of the report used, so a 12:00 report asked about at
lock-10 is still bucket "gt6h".

Public API (consumed by T1.4 / T1.6)
  p_play(conn, player_id, game_id, at, params=None) -> float
  p_play_slate(conn, nba_day, at, params=None)
        -> DataFrame[player_id, team, game_id, status, p_play, source, report_time, + player,
                     reason_class, role, hours_to_tip, bucket]
  fit(conn, lo=None, hi=None, ...) -> AvailabilityParams     (writes nothing; `.save()` does)
  evaluate(conn, ...) -> dict      (time split + blocked CV, vs the old map; see `report`)
  load_params(path=None) -> AvailabilityParams     (JSON shipped in this package)
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, injuries

PARAMS_PATH = Path(__file__).with_name("availability_params.json")

# --- vocabulary ------------------------------------------------------------------------------

LISTED = ("out", "doubtful", "questionable", "probable", "available")   # ascending P(play)
STATUSES = LISTED + ("daytoday", "removed", "unlisted", "unlisted_noreport")
BUCKETS = ("lt45m", "45m_2h", "2h_6h", "gt6h")          # hours from the report to the tip
BUCKET_EDGES = (0.75, 2.0, 6.0)
NOBUCKET = "none"                                       # no report on file for the team
REASONS = ("injury", "management", "illness", "rest", "personal", "gleague", "suspension",
           "other", "none")
ROLES = ("starter", "rotation", "bench")
ROLE_EDGES = (15.0, 26.0)                               # mean minutes of the last 10 appearances
CLAMP = (0.005, 0.995)
DEFAULT_TIP_ET = (19, 30)                               # when a game's tip time is unknown

# P(play) priors = the current hand-set map (injuries.PLAY_PROB), clamped so Beta stays proper.
# The two "unlisted" statuses have no entry there ("not listed = 1"); 0.99 is the §4.1 override.
WEAK_PRIOR_M = 5.0
_PRIOR_EXTRA = {"removed": 0.5, "unlisted": 0.99, "unlisted_noreport": 0.97}

_STATUS_WORDS = {
    "out": "out", "out for season": "out", "inactive": "out", "suspended": "out",
    "doubtful": "doubtful", "questionable": "questionable", "game time decision": "questionable",
    "gtd": "questionable", "probable": "probable", "available": "available",
    "healthy": "available", "active": "available",
    "day-to-day": "daytoday", "day to day": "daytoday",
}


def canon_status(status: str | None) -> str:
    """Report wording -> one of LISTED / 'daytoday'.  Unknown wording is treated as questionable."""
    s = (status or "").strip().lower()
    if not s:
        return "available"
    return _STATUS_WORDS.get(s, "questionable")


def reason_class(reason: str | None) -> str:
    """Parse the report's reason text into a class.

    'Injury/Illness - Left Ankle; Sprain' -> injury;  '...; Injury Management' -> management;
    '...; Illness' -> illness;  'Rest' / "Coach's Decision" -> rest;  'Personal Reasons' / 'Not
    With Team' -> personal;  'G League - ...' / 'Return to Competition Reconditioning' -> gleague;
    'League Suspension' -> suspension;  anything else -> other;  empty -> none."""
    s = (reason or "").split("|")[0].strip().lower()
    if not s:
        return "none"
    if s.startswith("g league") or s.startswith("return to competition") or "two-way" in s:
        return "gleague"
    if "suspension" in s:
        return "suspension"
    if s.startswith("rest") or "coach" in s:
        return "rest"
    if s.startswith("personal") or s.startswith("not with team"):
        return "personal"
    if s.startswith("injury/illness"):
        detail = s.partition("-")[2]                  # 'left ankle; sprain'
        body, _, what = detail.partition(";")
        if "management" in what or "reconditioning" in what or "load" in what:
            return "management"
        if "illness" in what or "illness" in body or "covid" in what or "flu" in what:
            return "illness"
        return "injury"
    if s.startswith("concussion"):
        return "injury"
    if "management" in s:
        return "management"
    if "illness" in s or "sick" in s:
        return "illness"
    if "injury" in s:
        return "injury"
    return "other"


def hours_bucket(hours: float | None) -> str:
    """Hours between the report and the tip -> bucket (None -> no report)."""
    if hours is None or (isinstance(hours, float) and np.isnan(hours)):
        return NOBUCKET
    if hours < BUCKET_EDGES[0]:
        return BUCKETS[0]
    if hours < BUCKET_EDGES[1]:
        return BUCKETS[1]
    if hours < BUCKET_EDGES[2]:
        return BUCKETS[2]
    return BUCKETS[3]


def role_of(avg_min: float | None) -> str:
    if avg_min is None or (isinstance(avg_min, float) and np.isnan(avg_min)):
        return "bench"
    if avg_min >= ROLE_EDGES[1]:
        return "starter"
    if avg_min >= ROLE_EDGES[0]:
        return "rotation"
    return "bench"


def prior_p(status: str) -> float:
    """The hand-set P(play) for a status, clamped into the open interval."""
    if status in _PRIOR_EXTRA:
        p = _PRIOR_EXTRA[status]
    else:
        p = injuries.play_probability({"daytoday": "Day-To-Day"}.get(status, status.capitalize()))
    return float(min(max(p, CLAMP[0] * 2), 1 - CLAMP[0] * 2))


def old_map_p(status: str | None, listed: bool = True) -> float:
    """The map being replaced: `injuries.play_probability` (unlisted = 1)."""
    return injuries.play_probability(status) if listed else 1.0


# --- the Beta-binomial hierarchy -------------------------------------------------------------

LEVELS = 4


def _keys(status: str, bucket: str, reason: str, role: str) -> list[str]:
    return [status, f"{status}|{bucket}", f"{status}|{bucket}|{reason}",
            f"{status}|{bucket}|{reason}|{role}"]


@dataclass
class AvailabilityParams:
    """Counts per cell + the shrinkage strengths.  JSON-serialisable; `p` is deterministic."""
    m1: float = 20.0                  # pseudo-observations of the hand-set prior (spec: 20)
    m: float = 20.0                   # pseudo-observations of a parent posterior, levels 2-4
    cells: dict[str, list[int]] = field(default_factory=dict)     # key -> [n, k]
    meta: dict = field(default_factory=dict)

    # -- posterior chain ---
    def _m1(self, status: str) -> float:
        """Pseudo-observations of the status prior: m1 for the hand-set statuses, a weak 5 for
        the ones the old map never priced (removed / unlisted*)."""
        return min(self.m1, WEAK_PRIOR_M) if status in _PRIOR_EXTRA else self.m1

    def _post(self, key: str, parent: float, m: float) -> tuple[float, int]:
        n, k = self.cells.get(key, (0, 0))
        return (k + m * parent) / (n + m), n

    def chain(self, status: str, bucket: str, reason: str, role: str) -> list[tuple[float, int]]:
        """[(posterior mean, n)] for levels 1..4 of one cell.

        For the five report statuses levels 1 and 2 are made monotone across statuses (Out <=
        Doubtful <= Questionable <= Probable <= Available) by weighted pool-adjacent-violators;
        levels 3-4 then shrink to that monotone parent, so a reason/role cell with enough data
        may still disagree with the ordering (e.g. 'Available' G-League assignees)."""
        keys = _keys(status, bucket, reason, role)
        if status in LISTED:
            l1 = [self._post(s, prior_p(s), self._m1(s)) for s in LISTED]
            v1 = _pava([p for p, _ in l1], [n + self.m1 for _, n in l1])
            l2 = [self._post(f"{s}|{bucket}", v1[i], self.m) for i, s in enumerate(LISTED)]
            v2 = _pava([p for p, _ in l2], [n + self.m for _, n in l2])
            i = LISTED.index(status)
            out = [(v1[i], l1[i][1]), (v2[i], l2[i][1])]
        else:
            p1, n1 = self._post(keys[0], prior_p(status), self._m1(status))
            p2, n2 = self._post(keys[1], p1, self.m)
            out = [(p1, n1), (p2, n2)]
        p3, n3 = self._post(keys[2], out[1][0], self.m)
        p4, n4 = self._post(keys[3], p3, self.m)
        return out + [(p3, n3), (p4, n4)]

    def p(self, status: str, bucket: str = NOBUCKET, reason: str = "none", role: str = "bench"
          ) -> float:
        """P(play) for a cell, clamped to CLAMP."""
        v = self.chain(status, bucket, reason, role)[-1][0]
        return float(min(max(v, CLAMP[0]), CLAMP[1]))

    def p_status_bucket(self, status: str, bucket: str) -> float:
        """Level-2 posterior (reasons and roles pooled), monotone across the report statuses."""
        v = self.chain(status, bucket, "none", "bench")[1][0]
        return float(min(max(v, CLAMP[0]), CLAMP[1]))

    def n_cell(self, status: str, bucket: str, reason: str, role: str) -> int:
        return self.chain(status, bucket, reason, role)[-1][1]

    def table(self, buckets=BUCKETS) -> pd.DataFrame:
        """status x bucket table of P(play) (level 2) with the training n of every cell."""
        rows = []
        for s in STATUSES:
            r = {"status": s, "n": self.cells.get(s, (0, 0))[0]}
            for b in buckets:
                r[b] = self.p_status_bucket(s, b)
                r[f"n_{b}"] = self.cells.get(f"{s}|{b}", (0, 0))[0]
            rows.append(r)
        return pd.DataFrame(rows)

    # -- (de)serialisation ---
    def to_json(self) -> dict:
        return {"m1": self.m1, "m": self.m, "meta": self.meta,
                "cells": {k: list(v) for k, v in sorted(self.cells.items())}}

    @classmethod
    def from_json(cls, d: dict) -> "AvailabilityParams":
        return cls(m1=float(d["m1"]), m=float(d["m"]), meta=d.get("meta", {}),
                   cells={k: [int(v[0]), int(v[1])] for k, v in d["cells"].items()})

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else PARAMS_PATH
        d = self.to_json()
        lines = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in d["cells"].items())
        head = json.dumps({k: d[k] for k in ("m", "m1", "meta")}, sort_keys=True)[:-1]
        path.write_text(head + ',\n "cells": {\n' + lines + "\n }}\n")
        return path


def _pava(y: list[float], w: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators: the non-decreasing fit closest to y."""
    blocks: list[list[float]] = []          # [value, weight, count]
    for v, wt in zip(y, w):
        blocks.append([v, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop()
            a = blocks[-1]
            tw = a[1] + b[1]
            a[0] = (a[0] * a[1] + b[0] * b[1]) / tw
            a[1] = tw
            a[2] += b[2]
    out: list[float] = []
    for v, _, c in blocks:
        out += [v] * int(c)
    return out


_PARAMS_CACHE: dict[str, AvailabilityParams] = {}


def load_params(path: Path | str | None = None) -> AvailabilityParams:
    """The fitted params shipped in the package (cached).  Missing file -> pure prior."""
    path = Path(path) if path else PARAMS_PATH
    key = str(path)
    if key not in _PARAMS_CACHE:
        if path.exists():
            _PARAMS_CACHE[key] = AvailabilityParams.from_json(json.loads(path.read_text()))
        else:
            _PARAMS_CACHE[key] = AvailabilityParams()
    return _PARAMS_CACHE[key]


# --- time helpers ----------------------------------------------------------------------------

def _aware(at: datetime | str) -> datetime:
    if isinstance(at, str):
        return injuries._ts(at) if at.endswith("Z") and "." not in at else \
            datetime.fromisoformat(at.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("`at` must be timezone-aware")
    return at.astimezone(timezone.utc)


def _parse(ts: str) -> datetime:
    return injuries._ts(ts)


def _day_start_utc(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=config.NBA_TZ).astimezone(timezone.utc)


def _default_tip(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, *DEFAULT_TIP_ET,
                    tzinfo=config.NBA_TZ).astimezone(timezone.utc)


# --- tip times -------------------------------------------------------------------------------

def espn_scoreboard_tips(day: date) -> list[dict]:
    """Archived ESPN scoreboard events for `day` ([{home, away, tipoff_utc}]); [] if none."""
    from ..sources import espn as espn_src
    d = config.data_dir() / "raw" / "espn_odds" / "scoreboard" / day.isoformat()
    out: list[dict] = []
    for f in sorted(d.glob("*.json.gz")) if d.exists() else []:
        try:
            with gzip.open(f, "rt") as fh:
                out += espn_src.scoreboard_events(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def day_games(conn: sqlite3.Connection, day: date) -> pd.DataFrame:
    """[game_id, home, away, tip (aware UTC datetime), tip_known] for an NBA day.

    Prefers the live `games` table; falls back to `team_games` (history) with tip times from
    the archived ESPN scoreboards, and to 19:30 ET when no tip is known."""
    d = day.isoformat()
    rows = conn.execute("SELECT game_id, home, away, tipoff_utc FROM games WHERE nba_day=? "
                        "AND status != 'postponed'", (d,)).fetchall()
    games = [dict(game_id=r[0], home=r[1], away=r[2], tip_s=r[3]) for r in rows]
    if not games:
        tg = pd.read_sql_query(
            "SELECT game_id, team, opp, is_home FROM team_games WHERE game_date=?", conn,
            params=(d,))
        sb = {(e["home"], e["away"]): e["tipoff_utc"] for e in espn_scoreboard_tips(day)}
        seen: set[str] = set()
        for r in tg.itertuples():
            if r.game_id in seen:
                continue
            seen.add(r.game_id)
            home, away = (r.team, r.opp) if r.is_home else (r.opp, r.team)
            tip = sb.get((home, away)) or sb.get((away, home))
            games.append(dict(game_id=r.game_id, home=home, away=away, tip_s=tip))
    out = []
    for g in games:
        known = bool(g["tip_s"])
        tip = _parse(g["tip_s"]) if known else _default_tip(day)
        out.append(dict(game_id=g["game_id"], home=g["home"], away=g["away"], tip=tip,
                        tip_known=known))
    return pd.DataFrame(out, columns=["game_id", "home", "away", "tip", "tip_known"])


# --- history: roles, universe ----------------------------------------------------------------

class History:
    """Box history for role and the unlisted universe.  Everything is 'strictly before day'."""

    def __init__(self, pg: pd.DataFrame, tg: pd.DataFrame):
        pg = pg[pg["min"] > 0].sort_values(["game_date", "game_id"], kind="mergesort")
        self.by_player: dict[int, tuple[list[str], list[float]]] = {}
        for pid, g in pg.groupby("player_id", sort=False):
            self.by_player[int(pid)] = (g["game_date"].tolist(), g["min"].astype(float).tolist())
        # team -> sorted [(date, game_id)]; (team, game_id) -> {pid}
        self.team_games: dict[str, list[tuple[str, str]]] = {}
        for r in tg.sort_values(["game_date", "game_id"], kind="mergesort").itertuples():
            self.team_games.setdefault(r.team, []).append((r.game_date, r.game_id))
        self._tg_dates = {t: [d for d, _ in v] for t, v in self.team_games.items()}
        self.box: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for r in pg.itertuples():
            self.box.setdefault((r.game_id, r.team), []).append((int(r.player_id), float(r.min)))
        self.played = {(r.game_id, int(r.player_id)) for r in pg.itertuples()}
        self.last_team: dict[int, tuple[list[str], list[str]]] = {}
        for pid, g in pg.groupby("player_id", sort=False):
            self.last_team[int(pid)] = (g["game_date"].tolist(), g["team"].tolist())

    def avg_min(self, pid: int, day: date, n: int = 10) -> float | None:
        """Mean minutes of the player's last `n` appearances strictly before `day`."""
        rec = self.by_player.get(pid)
        if not rec:
            return None
        i = bisect_right(rec[0], (day - timedelta(days=1)).isoformat())
        if i == 0:
            return None
        v = rec[1][max(0, i - n):i]
        return float(np.mean(v))

    def team_of(self, pid: int, day: date) -> str | None:
        rec = self.last_team.get(pid)
        if not rec:
            return None
        i = bisect_right(rec[0], (day - timedelta(days=1)).isoformat())
        return rec[1][i - 1] if i else None

    def universe(self, team: str, day: date, n_games: int = 10) -> dict[int, float]:
        """{player_id: avg minutes} of players who appeared for `team` in its last `n_games`
        games before `day` and whose latest team is still `team`."""
        dates = self._tg_dates.get(team, [])
        i = bisect_right(dates, (day - timedelta(days=1)).isoformat())
        out: dict[int, float] = {}
        for _, gid in self.team_games.get(team, [])[max(0, i - n_games):i]:
            for pid, _m in self.box.get((gid, team), ()):
                if pid not in out and self.team_of(pid, day) == team:
                    a = self.avg_min(pid, day)
                    out[pid] = a if a is not None else 0.0
        return out


def load_history(conn: sqlite3.Connection, since: date) -> History:
    pg = pd.read_sql_query(
        "SELECT game_id, player_id, team, game_date, min FROM player_games WHERE game_date >= ?",
        conn, params=((since - timedelta(days=400)).isoformat(),))
    tg = pd.read_sql_query(
        "SELECT game_id, team, game_date FROM team_games WHERE game_date >= ?", conn,
        params=((since - timedelta(days=400)).isoformat(),))
    return History(pg, tg)


# --- reports ---------------------------------------------------------------------------------

@dataclass
class Batch:
    t: datetime                       # captured_at (aware UTC)
    ts: str
    rows: dict[str, dict]             # normalised name -> {name, status, reason}


def load_reports(conn: sqlite3.Connection, lo_utc: datetime, hi_utc: datetime
                 ) -> dict[tuple[date, str], list[Batch]]:
    """Official report batches captured in [lo_utc, hi_utc], grouped (ET day, team), by time.

    A team's batch only exists when the report lists >= 1 row for it (a team that has not
    submitted is absent from that report).  Rows are cumulative within a day."""
    cur = conn.execute(
        "SELECT captured_at, team, player_name, status, reason FROM injury_snapshots "
        "WHERE source=? AND captured_at>=? AND captured_at<=? ORDER BY captured_at, id",
        (injuries.OFFICIAL, injuries.iso(lo_utc), injuries.iso(hi_utc)))
    out: dict[tuple[date, str], dict[str, Batch]] = {}
    for ts, team, name, status, reason in cur:
        if not name:
            continue
        t = _parse(ts)
        day = config.nba_day(t)
        b = out.setdefault((day, team), {}).setdefault(ts, Batch(t, ts, {}))
        k = injuries.norm_name(name)
        prev = b.rows.get(k)
        st = canon_status(status)
        if prev is None or _rank(st) < _rank(prev["status"]):     # most severe duplicate wins
            b.rows[k] = {"name": name, "status": st, "reason": reason or ""}
    return {k: sorted(v.values(), key=lambda b: b.t) for k, v in out.items()}


def _rank(status: str) -> int:
    return LISTED.index(status) if status in LISTED else 2


def _latest(batches: list[Batch], at: datetime) -> tuple[Batch | None, list[Batch]]:
    """(newest batch with t <= at, all batches with t <= at)."""
    upto = [b for b in batches if b.t <= at]
    return (upto[-1] if upto else None), upto


# --- one team-game ---------------------------------------------------------------------------

def team_game_rows(*, game_id: str, day: date, team: str, tip: datetime, at: datetime,
                   batches: list[Batch], hist: History, pool: list[tuple[int, str]],
                   league: list[tuple[int, str]], _match_cache: dict | None = None,
                   source: str = injuries.OFFICIAL) -> list[dict]:
    """Every candidate player of `team` for `game_id` as known at `at`.

    Candidates: players the report on file lists (matched to a player_id), players listed in an
    earlier report of the day and dropped from the latest ('removed'), and the team's regulars
    (appeared in its last 10 games) the report does not list ('unlisted').  Uses only batches
    with t <= at (and at is never after the tip)."""
    at = min(at, tip)
    latest, upto = _latest(batches, at)
    cache = _match_cache if _match_cache is not None else {}

    def pid_of(name: str) -> int | None:
        k = (team, name)
        if k not in cache:
            cache[k] = injuries.match_player(name, pool, None, league)[0]
        return cache[k]

    rows: dict[int, dict] = {}
    hours = (tip - latest.t).total_seconds() / 3600 if latest else None
    bucket = hours_bucket(hours)
    base = dict(game_id=game_id, day=day, team=team, at=at, report_time=latest.ts if latest else None,
                hours=hours, bucket=bucket, source=source if latest else "prior")
    if latest is not None:
        for nk, r in latest.rows.items():
            pid = pid_of(r["name"])
            if pid is None:
                continue
            row = dict(base, player_id=pid, status=r["status"], reason=reason_class(r["reason"]),
                       reason_raw=r["reason"], player=r["name"])
            if pid not in rows or _rank(r["status"]) < _rank(rows[pid]["status"]):
                rows[pid] = row
        earlier: dict[str, dict] = {}
        for b in upto[:-1]:
            earlier.update(b.rows)
        for nk, r in earlier.items():
            if nk in latest.rows:
                continue
            pid = pid_of(r["name"])
            if pid is not None and pid not in rows:
                rows[pid] = dict(base, player_id=pid, status="removed",
                                 reason=reason_class(r["reason"]), reason_raw=r["reason"],
                                 player=r["name"])
    un_status = "unlisted" if latest is not None else "unlisted_noreport"
    for pid, avg in hist.universe(team, day).items():
        if pid not in rows:
            rows[pid] = dict(base, player_id=pid, status=un_status, reason="none", reason_raw="",
                             player=None)
    for pid, r in rows.items():
        r["avg_min"] = hist.avg_min(pid, day)
        r["role"] = role_of(r["avg_min"])
        r["played"] = int((game_id, pid) in hist.played)
        r["listed"] = r["status"] in LISTED
    return list(rows.values())


# --- sample builders -------------------------------------------------------------------------

def _pools(conn, days):
    """day -> (rosters by team, league list), computed once per day."""
    cache = {}
    for d in days:
        act = injuries.active_players(conn, d)
        cache[d] = (injuries.recent_rosters(conn, d, None, act), [(p, n) for p, n, _ in act])
    return cache


def build_samples(conn: sqlite3.Connection, lo: date, hi: date, *,
                  mode: str = "bucket", offsets_min: tuple[int, ...] = (10, 45, 300),
                  hist: History | None = None) -> pd.DataFrame:
    """Candidate-player rows for every game with an NBA day in [lo, hi].

    mode='bucket'   training rows: per team-game and hours bucket, the LATEST report of that
                    bucket (plus a no-report row set when the team has no report before tip).
    mode='scenario' evaluation rows: for each offset (minutes before the game's own tip) the
                    latest report published at or before tip - offset.  Column `scenario`.
    """
    hist = hist or load_history(conn, lo)
    reports = load_reports(conn, _day_start_utc(lo), _day_start_utc(hi + timedelta(days=1)))
    days = sorted({d for d, _ in reports} | set())
    cur = conn.execute("SELECT DISTINCT game_date FROM team_games WHERE game_date>=? AND game_date<=? "
                       "ORDER BY game_date", (lo.isoformat(), hi.isoformat()))
    game_days = [date.fromisoformat(r[0]) for r in cur]
    all_rows: list[dict] = []
    report_days = {d for d, _ in reports}
    stype = dict(conn.execute("SELECT game_id, season_type FROM team_games WHERE game_date>=? "
                              "AND game_date<=?", (lo.isoformat(), hi.isoformat())).fetchall())
    for day in game_days:
        if day not in report_days:
            continue                       # no official report that day: nothing to evaluate
        games = day_games(conn, day)
        if games.empty:
            continue
        act = injuries.active_players(conn, day)
        rosters = injuries.recent_rosters(conn, day, None, act)
        league = [(p, n) for p, n, _ in act]
        mc: dict = {}
        for g in games.itertuples():
            for team in (g.home, g.away):
                batches = [b for b in reports.get((day, team), []) if b.t < g.tip]
                common = dict(game_id=g.game_id, day=day, team=team, tip=g.tip, hist=hist,
                              pool=rosters.get(team, []), league=league, _match_cache=mc,
                              batches=batches)
                if mode == "bucket":
                    chosen: dict[str, Batch] = {}
                    for b in batches:
                        bk = hours_bucket((g.tip - b.t).total_seconds() / 3600)
                        chosen[bk] = b            # batches ascend in time: last = latest
                    ats = [(bk, b.t) for bk, b in chosen.items()]
                    if not batches:
                        ats = [(NOBUCKET, g.tip - timedelta(minutes=1))]
                else:
                    ats = [(f"T-{o}m", g.tip - timedelta(minutes=o)) for o in offsets_min]
                for tag, at in ats:
                    for r in team_game_rows(at=at, **common):
                        r["scenario"] = tag
                        r["season_type"] = stype.get(g.game_id, "Regular Season")
                        r["tip"] = g.tip
                        r["tip_known"] = g.tip_known
                        all_rows.append(r)
    df = pd.DataFrame(all_rows)
    return df


# --- fitting ---------------------------------------------------------------------------------

def counts_from(df: pd.DataFrame) -> dict[str, list[int]]:
    """Cell counts [n, k] at all four levels for a sample frame."""
    cells: dict[str, list[int]] = {}
    if df.empty:
        return cells
    base = df[["status", "bucket", "reason", "role", "played"]].copy()
    for lvl, cols in enumerate((["status"], ["status", "bucket"], ["status", "bucket", "reason"],
                                ["status", "bucket", "reason", "role"])):
        g = base.groupby(cols)["played"].agg(["size", "sum"])
        for idx, (n, k) in g.iterrows():
            key = "|".join(idx if isinstance(idx, tuple) else (idx,))
            cells[key] = [int(n), int(k)]
    return cells


def fit_from_samples(df: pd.DataFrame, m1: float = 20.0, m: float = 20.0, meta: dict | None = None
                     ) -> AvailabilityParams:
    return AvailabilityParams(m1=m1, m=m, cells=counts_from(df), meta=meta or {})


DEFAULT_M = 20.0                 # blocked-CV optimum is flat over 5-40 (see evaluate)
TRAIN_LO = date(2025, 12, 22)
RS_END = date(2026, 4, 12)


def fit(conn: sqlite3.Connection, lo: date | None = None, hi: date | None = None, *,
        m1: float = 20.0, m: float | None = None, samples: pd.DataFrame | None = None,
        seed_note: str = "") -> AvailabilityParams:
    """Fit the cells on regular-season games with an official report in [lo, hi]
    (default: all rescued regular-season days).  `m` default: the CV-chosen value in
    DEFAULT_M.  Regenerate the shipped JSON with `fit(conn).save()`."""
    lo = lo or TRAIN_LO
    hi = hi or RS_END
    df = samples if samples is not None else build_samples(conn, lo, hi, mode="bucket")
    df = df[(df["day"] >= lo) & (df["day"] <= hi) & (df["season_type"] == "Regular Season")]
    meta = {"fit_lo": lo.isoformat(), "fit_hi": hi.isoformat(), "rows": int(len(df)),
            "games": int(df["game_id"].nunique()) if len(df) else 0,
            "sources": "nba_official injury reports vs player_games.min>0", "note": seed_note}
    return fit_from_samples(df, m1=m1, m=DEFAULT_M if m is None else m, meta=meta)



# --- serving ---------------------------------------------------------------------------------

def _fallback_batches(res: dict, team: str, day: date) -> tuple[list[Batch], str]:
    """No official report for (day, team) at/before `at`: the older official report or the ESPN
    list `injuries.resolve` picked for the team, as a one-batch list (+ its source)."""
    t = res["teams"].get(team)
    if not t:
        return [], "prior"
    rows: dict[str, dict] = {}
    for r in t["rows"]:
        if not r.get("player_name"):
            continue
        reason = r.get("reason") or ""
        if t["source"] == injuries.ESPN and reason and "-" not in reason[:20]:
            reason = "Injury/Illness - " + reason
        k = injuries.norm_name(r["player_name"])
        st = canon_status(r.get("status"))
        if k not in rows or _rank(st) < _rank(rows[k]["status"]):
            rows[k] = {"name": r["player_name"], "status": st, "reason": reason}
    return [Batch(_parse(t["captured_at"]), t["captured_at"], rows)], t["source"]


def _slate(conn: sqlite3.Connection, day: date, at: datetime, params: AvailabilityParams,
           game_ids: set[str] | None = None, hist: History | None = None,
           starters: set[int] | None = None) -> pd.DataFrame:
    games = day_games(conn, day)
    if game_ids is not None:
        games = games[games["game_id"].isin(game_ids)]
    cols = ["player_id", "team", "game_id", "status", "p_play", "source", "report_time",
            "player", "reason_class", "role", "hours_to_tip", "bucket", "n_cell", "override"]
    if games.empty:
        return pd.DataFrame(columns=cols)
    hist = hist or load_history(conn, day)
    reports = load_reports(conn, _day_start_utc(day), at)
    res = injuries.resolve(conn, injuries.iso(at), espn_max_age_h=injuries.ESPN_MAX_AGE_H)
    act = injuries.active_players(conn, day)
    rosters = injuries.recent_rosters(conn, day, None, act)
    league = [(p, n) for p, n, _ in act]
    mc: dict = {}
    out: list[dict] = []
    for g in games.itertuples():
        for team in (g.home, g.away):
            batches = [b for b in reports.get((day, team), []) if b.t <= at]
            source = injuries.OFFICIAL
            if not batches:
                batches, source = _fallback_batches(res, team, day)
                batches = [b for b in batches if b.t <= at]
            out += team_game_rows(game_id=g.game_id, day=day, team=team, tip=g.tip, at=at,
                                  batches=batches, hist=hist, pool=rosters.get(team, []),
                                  league=league, _match_cache=mc, source=source)
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = predict(df, params)
    df["override"] = False
    if starters:
        m = df["player_id"].isin(starters)
        df.loc[m, "p_play"] = 0.99
        df.loc[m, "override"] = True
    df = df.rename(columns={"reason": "reason_class", "hours": "hours_to_tip"})
    return df[cols].sort_values(["game_id", "team", "p_play", "player_id"],
                                ascending=[True, True, True, True]).reset_index(drop=True)


def predict(df: pd.DataFrame, params: AvailabilityParams) -> pd.DataFrame:
    """Adds `p_play` and `n_cell` to a sample frame (status/bucket/reason/role columns)."""
    cache: dict[tuple, tuple[float, int]] = {}
    ps, ns = [], []
    for key in zip(df["status"], df["bucket"], df["reason"], df["role"]):
        if key not in cache:
            cache[key] = (params.p(*key), params.n_cell(*key))
        ps.append(cache[key][0])
        ns.append(cache[key][1])
    out = df.copy()
    out["p_play"] = ps
    out["n_cell"] = ns
    return out


def p_play_slate(conn: sqlite3.Connection, nba_day: date | str, at: datetime | str,
                 params: AvailabilityParams | None = None, *, starters=None,
                 hist: History | None = None) -> pd.DataFrame:
    """P(play) for every candidate of every game on `nba_day`, as known at `at` (aware datetime
    or ISO 'Z' string).  Uses only reports published at or before `at`.

    Columns: player_id, team, game_id, status (report status; 'unlisted' when the team's report
    omits him, 'unlisted_noreport' when no report is on file), p_play, source ('nba_official',
    'espn', or 'prior'), report_time (the report used), plus player, reason_class, role,
    hours_to_tip (report -> tip), bucket, n_cell (training games in his cell), override.
    `starters`: iterable of player_ids with the DK starter flag = 1 -> P = 0.99 (§4.1)."""
    day = date.fromisoformat(nba_day) if isinstance(nba_day, str) else nba_day
    return _slate(conn, day, _aware(at), params or load_params(), hist=hist,
                  starters=set(starters) if starters is not None else None)


def p_play(conn: sqlite3.Connection, player_id: int, game_id: str, at: datetime | str,
           params: AvailabilityParams | None = None, *, hist: History | None = None) -> float:
    """P(player plays in game_id) as known at `at`.  A player the pipeline cannot place on either
    team gets the bench-regular 'unlisted' rate."""
    params = params or load_params()
    r = conn.execute("SELECT nba_day FROM games WHERE game_id=?", (game_id,)).fetchone()
    if r is None:
        r = conn.execute("SELECT game_date FROM team_games WHERE game_id=? LIMIT 1",
                         (game_id,)).fetchone()
    if r is None:
        raise KeyError(f"unknown game {game_id}")
    day = date.fromisoformat(str(r[0])[:10])
    df = _slate(conn, day, _aware(at), params, game_ids={game_id}, hist=hist)
    hit = df[df["player_id"] == int(player_id)]
    if len(hit):
        return float(hit["p_play"].iloc[0])
    return params.p("unlisted", "gt6h", "none", "bench")


# --- evaluation (honest: time split + blocked CV) --------------------------------------------

SCENARIOS = ("T-300m", "T-45m", "T-10m")           # ~T-5h, lock-45, lock-10 (the game's own tip)
M_GRID = (5.0, 10.0, 20.0, 40.0, 80.0)
SPLIT_DAY = date(2026, 3, 15)                       # fit <= this; test 03-16 .. 04-12 (RS)
UNCERTAIN = ("doubtful", "questionable", "probable")


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), CLAMP[0], CLAMP[1])


def old_map_column(df: pd.DataFrame) -> np.ndarray:
    """The map being replaced, row by row (unlisted / removed / no-report = 1)."""
    out = []
    for st in df["status"]:
        out.append(old_map_p({"daytoday": "Day-To-Day"}.get(st, st.capitalize()), True)
                   if st in LISTED or st == "daytoday" else 1.0)
    return _clip(out)


def scores(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return {"n": int(len(y)), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "base_rate": float(y.mean()) if len(y) else None,
            "mean_p": float(p.mean()) if len(y) else None}


def paired_brier(y, p_new, p_old, groups) -> dict:
    """mean(Brier_new - Brier_old) with a game-clustered standard error."""
    y = np.asarray(y, float)
    d = (_clip(p_new) - y) ** 2 - (_clip(p_old) - y) ** 2
    g = pd.Series(d).groupby(np.asarray(groups)).agg(["sum", "size"])
    N = len(d)
    mean = d.sum() / N
    se = float(np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / N)
    return {"diff": float(mean), "se": se, "games": int(len(g))}


def _subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all":
        return df
    if name == "uncertain":
        return df[df["status"].isin(UNCERTAIN)]
    if name == "listed":
        return df[df["status"].isin(LISTED)]
    if name == "unlisted":
        return df[df["status"].isin(("unlisted", "unlisted_noreport", "removed"))]
    return df[df["status"] == name]


def compare(df: pd.DataFrame, p_new, p_old) -> dict:
    """{scenario: {subset: {new, old, paired}}} for a frame with `scenario`, `played`."""
    df = df.assign(_pn=np.asarray(p_new), _po=np.asarray(p_old))
    out: dict = {}
    for sc in list(SCENARIOS) + ["pooled"]:
        d = df if sc == "pooled" else df[df["scenario"] == sc]
        if d.empty:
            continue
        out[sc] = {}
        for sub in ("all", "listed", "uncertain", "unlisted"):
            s = _subset(d, sub)
            if s.empty:
                continue
            out[sc][sub] = {"new": scores(s["played"], s["_pn"]), "old": scores(s["played"], s["_po"]),
                            "paired_brier": paired_brier(s["played"], s["_pn"], s["_po"], s["game_id"])}
    return out


def calibration(df: pd.DataFrame, p_new, p_old) -> pd.DataFrame:
    """Per status (pooled over scenarios): n, observed, mean predicted (new / old), gaps."""
    d = df.assign(_pn=_clip(p_new), _po=_clip(p_old))
    rows = []
    for st, g in d.groupby("status"):
        rows.append({"status": st, "n": len(g), "obs": g["played"].mean(), "new": g["_pn"].mean(),
                     "old": g["_po"].mean(), "gap_new": g["_pn"].mean() - g["played"].mean(),
                     "gap_old": g["_po"].mean() - g["played"].mean(),
                     "ok_new": bool(len(g) < 50 or abs(g["_pn"].mean() - g["played"].mean()) <= 0.08),
                     "ok_old": bool(len(g) < 50 or abs(g["_po"].mean() - g["played"].mean()) <= 0.08)})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, p_new, p_old, cols: list[str]) -> pd.DataFrame:
    """n / observed / new / old mean P by arbitrary columns (e.g. status x role)."""
    d = df.assign(_pn=_clip(p_new), _po=_clip(p_old))
    return (d.groupby(cols, observed=True).agg(n=("played", "size"), obs=("played", "mean"),
                                               new=("_pn", "mean"), old=("_po", "mean"))
            .reset_index())


def reliability_bins(df: pd.DataFrame, p_new, edges=(0, .02, .1, .3, .5, .7, .85, .95, .99, 1.01)):
    d = df.assign(_pn=_clip(p_new))
    d["bin"] = pd.cut(d["_pn"], list(edges), right=False)
    g = d.groupby("bin", observed=True).agg(n=("played", "size"), obs=("played", "mean"),
                                            pred=("_pn", "mean"))
    return g.reset_index()


def _week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _fit_counts(T: pd.DataFrame) -> dict[str, list[int]]:
    return counts_from(T)


def _predict_with(S: pd.DataFrame, cells, m1, m) -> np.ndarray:
    return predict(S, AvailabilityParams(m1=m1, m=m, cells=cells))["p_play"].to_numpy()


def blocked_cv(T: pd.DataFrame, S: pd.DataFrame, ms=M_GRID, m1: float = 20.0) -> dict:
    """Leave-one-week-out CV (all rows of a calendar week are held out together).  Returns the
    out-of-fold predictions per m, and the pooled log-loss per m."""
    T = T.assign(_wk=T["day"].map(_week))
    S = S.assign(_wk=S["day"].map(_week))
    weeks = sorted(S["_wk"].unique())
    oof = {m: np.full(len(S), np.nan) for m in ms}
    for wk in weeks:
        cells = _fit_counts(T[T["_wk"] != wk])
        idx = np.where(S["_wk"].to_numpy() == wk)[0]
        for m in ms:
            oof[m][idx] = _predict_with(S.iloc[idx], cells, m1, m)
    ll = {m: scores(S["played"], oof[m])["logloss"] for m in ms}
    return {"oof": oof, "logloss": ll, "weeks": len(weeks)}


def evaluate(conn: sqlite3.Connection, *, ms=M_GRID, lo: date = TRAIN_LO, hi: date | None = None,
             T: pd.DataFrame | None = None, S: pd.DataFrame | None = None) -> dict:
    """Everything §4.1 asks for.  Nothing is tuned on the test window: m is chosen by blocked CV
    inside the training window (<= SPLIT_DAY) for the time split."""
    hi = hi or date(2026, 6, 14)
    if T is None:
        T = build_samples(conn, lo, hi, mode="bucket")
    if S is None:
        S = build_samples(conn, lo, hi, mode="scenario")
    S = S[S["scenario"].isin(SCENARIOS)].reset_index(drop=True)
    rs_T = T[T["season_type"] == "Regular Season"]
    rs_S = S[S["season_type"] == "Regular Season"]
    res: dict = {"sizes": {"train_rows": int(len(rs_T)), "scenario_rows": int(len(rs_S)),
                           "days": int(rs_S["day"].nunique()), "games": int(rs_S["game_id"].nunique())}}
    # --- time split ---
    tr_T = rs_T[rs_T["day"] <= SPLIT_DAY]
    tr_S = rs_S[rs_S["day"] <= SPLIT_DAY]
    te = rs_S[(rs_S["day"] > SPLIT_DAY) & (rs_S["day"] <= RS_END)].reset_index(drop=True)
    cvt = blocked_cv(tr_T, tr_S, ms)
    m_best = min(cvt["logloss"], key=cvt["logloss"].get)
    cells = _fit_counts(tr_T)
    p_new = _predict_with(te, cells, 20.0, m_best)
    p_old = old_map_column(te)
    res["time_split"] = {"m_tuned_on_train_cv": m_best, "train_cv_logloss": cvt["logloss"],
                         "train_days": [lo.isoformat(), SPLIT_DAY.isoformat()],
                         "test_days": [(SPLIT_DAY + timedelta(days=1)).isoformat(), RS_END.isoformat()],
                         "n_train_rows": int(len(tr_T)), "n_test_rows": int(len(te)),
                         "test": compare(te, p_new, p_old),
                         "calibration": calibration(te, p_new, p_old),
                         "reliability": reliability_bins(te, p_new),
                         "by_status_role": by_group(te[te["status"].isin(("unlisted", "questionable", "probable", "available"))],
                                                    p_new[te["status"].isin(("unlisted", "questionable", "probable", "available")).to_numpy()],
                                                    old_map_column(te)[te["status"].isin(("unlisted", "questionable", "probable", "available")).to_numpy()],
                                                    ["status", "role"]),
                         "m_sensitivity": {m: scores(te["played"], _predict_with(te, cells, 20.0, m))["logloss"]
                                           for m in ms}}
    # --- playoffs / play-in: fit on the whole regular season (their own data never used) ---
    po = S[S["season_type"] != "Regular Season"].reset_index(drop=True)
    if len(po):
        cells_rs = _fit_counts(rs_T)
        cv_all = blocked_cv(rs_T, rs_S, ms)
        m_all = min(cv_all["logloss"], key=cv_all["logloss"].get)
        pn, pold = _predict_with(po, cells_rs, 20.0, m_all), old_map_column(po)
        res["playoffs"] = {"n_rows": int(len(po)), "games": int(po["game_id"].nunique()),
                           "test": compare(po, pn, pold), "calibration": calibration(po, pn, pold)}
    # --- blocked CV over the whole regular season ---
    cv = blocked_cv(rs_T, rs_S, ms)
    m_cv = min(cv["logloss"], key=cv["logloss"].get)
    po_cv = cv["oof"][m_cv]
    res["cv"] = {"weeks": cv["weeks"], "m_chosen": m_cv, "logloss_by_m": cv["logloss"],
                 "test": compare(rs_S, po_cv, old_map_column(rs_S)),
                 "calibration": calibration(rs_S, po_cv, old_map_column(rs_S)),
                 "reliability": reliability_bins(rs_S, po_cv)}
    # --- hours-to-tip effect ---
    res["hours_effect"] = hours_effect(rs_S, rs_T)
    res["m_final"] = m_cv
    return res


def hours_effect(S: pd.DataFrame, T: pd.DataFrame) -> dict:
    """How much the report's age matters.  (a) observed play rate by status x bucket (pooled over
    reasons/roles) with n; (b) the same players followed from the T-5h report to the T-10m one:
    where Questionable-at-T-5h players end up and how often each group plays."""
    tab = (T[T["status"].isin(LISTED)].groupby(["status", "bucket"])["played"]
           .agg(["size", "mean"]).reset_index())
    a = S[S["scenario"] == "T-300m"][["game_id", "player_id", "status"]].rename(columns={"status": "s300"})
    b = S[S["scenario"] == "T-10m"][["game_id", "player_id", "status", "played"]].rename(
        columns={"status": "s10"})
    j = a.merge(b, on=["game_id", "player_id"])
    q = j[j["s300"] == "questionable"]
    follow = q.groupby("s10")["played"].agg(["size", "mean"]).reset_index()
    return {"by_status_bucket": tab, "q_at_5h_followed_to_10m": follow,
            "q_at_5h_overall": {"n": int(len(q)), "played": float(q["played"].mean()) if len(q) else None}}


# --- report / CLI ----------------------------------------------------------------------------

def _fmt_cmp(block: dict, scen: str = "pooled", subs=("listed", "uncertain", "unlisted", "all")) -> list[str]:
    lines = [f"{'subset':10s} {'n':>6s} {'ll_new':>7s} {'ll_old':>7s} {'br_new':>7s} {'br_old':>7s} "
             f"{'dBrier':>8s} {'(se)':>7s}"]
    for sub in subs:
        c = block.get(scen, {}).get(sub)
        if not c:
            continue
        lines.append(f"{sub:10s} {c['new']['n']:6d} {c['new']['logloss']:7.4f} {c['old']['logloss']:7.4f} "
                     f"{c['new']['brier']:7.4f} {c['old']['brier']:7.4f} "
                     f"{c['paired_brier']['diff']:8.4f} {c['paired_brier']['se']:7.4f}")
    return lines


def report(res: dict, params: AvailabilityParams | None = None) -> str:
    """Plain-text rendering of `evaluate` output (+ the shipped table)."""
    out: list[str] = []
    ts, cv = res["time_split"], res["cv"]
    out.append(f"sizes: {res['sizes']}")
    out.append(f"\n== time split: fit {ts['train_days']} ({ts['n_train_rows']} rows), "
               f"test {ts['test_days']} ({ts['n_test_rows']} rows); m={ts['m_tuned_on_train_cv']} "
               f"(blocked CV inside the train window)")
    for sc in list(SCENARIOS) + ["pooled"]:
        out.append(f"-- {sc}")
        out += _fmt_cmp(ts["test"], sc)
    out.append("\ncalibration by status (time-split test):\n" + ts["calibration"].round(3).to_string(index=False))
    out.append("\nunlisted / uncertain by role:\n" + ts["by_status_role"].round(3).to_string(index=False))
    out.append("\nreliability:\n" + ts["reliability"].round(3).to_string(index=False))
    out.append(f"\n== blocked CV, leave-one-week-out over the regular season "
               f"({cv['weeks']} weeks), m={cv['m_chosen']}; log-loss by m: "
               f"{ {k: round(v, 5) for k, v in cv['logloss_by_m'].items()} }")
    for sc in list(SCENARIOS) + ["pooled"]:
        out.append(f"-- {sc}")
        out += _fmt_cmp(cv["test"], sc)
    out.append("\ncalibration by status (CV):\n" + cv["calibration"].round(3).to_string(index=False))
    out.append("\nreliability (CV):\n" + cv["reliability"].round(3).to_string(index=False))
    if "playoffs" in res:
        po = res["playoffs"]
        out.append(f"\n== play-in + playoffs, fit on the regular season ({po['n_rows']} rows, {po['games']} games)")
        out += _fmt_cmp(po["test"], "pooled")
        out.append(po["calibration"].round(3).to_string(index=False))
    he = res["hours_effect"]
    out.append("\n== hours-to-tip: play rate by status x bucket of the report used\n"
               + he["by_status_bucket"].pivot(index="status", columns="bucket", values="mean")
               .reindex(columns=list(BUCKETS)).round(3).to_string())
    out.append("n:\n" + he["by_status_bucket"].pivot(index="status", columns="bucket", values="size")
               .reindex(columns=list(BUCKETS)).fillna(0).astype(int).to_string())
    out.append(f"Questionable at T-5h: n={he['q_at_5h_overall']['n']}, played {he['q_at_5h_overall']['played']:.3f}; "
               "their status at T-10m:\n" + he["q_at_5h_followed_to_10m"].round(3).to_string(index=False))
    if params is not None:
        out.append("\n== fitted P(play), status x bucket (reason and role pooled)\n"
                   + params.table().round(3).to_string(index=False))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m courtside.proj.availability")
    ap.add_argument("cmd", choices=["fit", "eval", "slate"])
    ap.add_argument("--day", help="slate: NBA day (YYYY-MM-DD)")
    ap.add_argument("--at", help="slate: ISO UTC decision time (default: now)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    if a.cmd == "fit":
        params = fit(conn)
        print(f"fitted on {params.meta['rows']} rows / {params.meta['games']} team-games")
        if not a.no_write:
            print("wrote", params.save())
            _PARAMS_CACHE.clear()
        print(params.table().round(3).to_string(index=False))
    elif a.cmd == "eval":
        print(report(evaluate(conn), load_params()))
    else:
        at = a.at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        day = a.day or config.today_nba_day().isoformat()
        df = p_play_slate(conn, day, at)
        print(df[df["p_play"] < 0.97].to_string(index=False))


if __name__ == "__main__":
    main()
