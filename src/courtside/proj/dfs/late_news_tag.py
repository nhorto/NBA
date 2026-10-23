"""Late-news tag: projected beneficiaries of a starter or rotation player ruled out shortly
before lock (docs/IMPROVEMENTS.md D4; docs/research/xfactor.md §0: the opener's real X-factor
nights were minutes / P(play) news, not tail luck, and the public's ownership lags late news).

Version ``ln-v1`` (a tag and metadata only: no projection, sim, lineup or X factor number
changes; the X factor stays ``xf-v1``).

Rule, per DFS run (slate lock L, run time `now`):

* **News player.** A player of the slate's teams whose injury-report status (official NBA report
  and ESPN, each source on its own) moved from Available / Probable / Questionable / not listed
  (severity <= 1) to Doubtful or Out (severity >= 2), with the news time (the first
  ``captured_at`` of his current Doubtful-or-Out streak, earliest over the sources) inside
  [L - ``LATE_NEWS_WINDOW``, min(L, now)], still Doubtful or Out in the newest batch by `now`,
  and a starter or rotation player: projected minutes if he plays >= ``ROTATION_MIN`` (in the
  baseline run, else this run) or model tier star. A source that already had him Doubtful or
  Out in its first batch of the lookback (no earlier healthier status) makes the news old.
  Rows are tied to the slate's day through ``injury_snapshots.game_date`` (schema 16) when the
  column exists (official rows for another date are dropped; ESPN rows carry none).
  Doubtful -> Out is not new news (the field already had him unlikely to play).
* **Beneficiaries.** Teammates whose projection rose against the **baseline run**: the newest
  successful DFS run (any slate of the same NBA day that projected his team) created before the
  team's earliest news time, with the same model version. Gain = this run minus the baseline,
  in DK points (mean over all sims, so his own P(play) counts) and in minutes if he plays. A
  teammate qualifies with gain >= ``MIN_GAIN_DK`` DK points or >= ``MIN_GAIN_MIN`` minutes,
  P(play) >= ``P_PLAY_MIN`` and a DK id; at most ``PER_TEAM`` per team, largest DK gain first.
  Why the run-to-run diff and not the P10 scenario split: a healthy player ruled out has
  P(available) = 1 in the baseline (no "sits" arm) and 0 now (no "plays" arm), so the split
  exists only for players who were already uncertain, and older runs carry no scenarios; the
  diff measures what the model actually moved, for every stored run.
* **Per flag:** the news player(s) (status from / to, source, news time, minutes to lock), the
  gains, this run's projection, the proxy popularity (``dfs_ownership_proxy`` own and band) and
  ``own_uncertainty`` (``OWN_UNCERTAINTY`` for a flag, 1.0 otherwise): the hook for the D5
  ownership-uncertainty field, which does not read it yet.

Stored by the DFS runner in ``dfs_runs.detail["late_news_tag"]`` (no migration); `run_flags`
reads that, or computes it for older runs (cached per run). Information only (STYLE.md).

Public API
  compute(conn, slate_id, run_id, now, model_version=None, *, window=...) -> dict
  compute_safe(...) -> dict | None              never raises (a tag must not fail a lock run)
  run_flags(conn, run_id) -> dict               stored, else computed; swap runs -> their build
  flags_by_dk(res) / flags_by_player(res) -> {id: flag}
  own_uncertainty(res) -> {dk id: multiplier}   (default 1.0 for everybody else)
  label(flag) -> str                            the tag's plain-English title
  news_events(conn, teams, day, since, until) -> {(team, norm name): event}
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

log = logging.getLogger(__name__)

VERSION = "ln-v1"
DETAIL_KEY = "late_news_tag"
LATE_NEWS_WINDOW = timedelta(minutes=60)
LOOKBACK = timedelta(hours=18)        # injury batches read before the lock
CAPTURE_SLACK = timedelta(seconds=120)  # a run's own refresh is captured just after its start
ROTATION_MIN = 15.0                   # news player: projected minutes if he plays
MIN_GAIN_DK = 3.0
MIN_GAIN_MIN = 3.0
PER_TEAM = 3
P_PLAY_MIN = 0.5
NEWS_P_PLAY_MAX = 0.2                 # the run must have taken the news in (Doubtful 0.1, Out 0)
OWN_UNCERTAINTY = 2.0                 # D5 hook: ownership-uncertainty multiplier of a flag
SOURCES = ("nba_official", "espn")
SOURCE_LABEL = {"nba_official": "official injury report", "espn": "ESPN"}
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 64


# ============================================================================== small helpers

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(s) -> datetime | None:
    if not s:
        return None
    s = str(s)
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _utc(dt: datetime | str) -> datetime:
    if isinstance(dt, str):
        out = _ts(dt)
        if out is None:
            raise ValueError(f"bad timestamp {dt!r}")
        return out
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    try:
        return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


def _loads(s, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


# ============================================================================== injury news

def news_events(conn: sqlite3.Connection, teams: Iterable[str], day: str | None,
                since: datetime, until: datetime) -> dict[tuple[str, str], dict]:
    """{(team, normalised name): event} for every player of `teams` who is Doubtful or Out in
    the newest covering batch at or before `until` (module doc). Event: {name, team, at (news
    time, None = old news, before the lookback or in a source's first batch), source,
    status_from, status_to}. An official batch covers the teams it lists (for `day`); an ESPN
    batch covers every team."""
    from ... import injuries as INJ
    teams = sorted({t for t in teams if t})
    if not teams:
        return {}
    gd = _has_column(conn, "injury_snapshots", "game_date") and day
    q = ",".join("?" * len(teams))
    sel = "source, captured_at, team, player_name, status" + (", game_date" if gd else "")
    rows = conn.execute(
        f"SELECT {sel} FROM injury_snapshots WHERE captured_at>=? AND captured_at<=? AND "
        f"source IN ('nba_official','espn') AND team IN ({q}) ORDER BY captured_at",
        (_iso(since), _iso(until), *teams)).fetchall()
    espn_batches = [r[0] for r in conn.execute(
        "SELECT DISTINCT captured_at FROM injury_snapshots WHERE source='espn' AND "
        "captured_at>=? AND captured_at<=? ORDER BY captured_at", (_iso(since), _iso(until)))]
    # per source: ordered batches -> {team: {norm: (name, status)}} (covered teams only)
    batches: dict[str, dict[str, dict[str, dict]]] = {s: {} for s in SOURCES}
    for t in espn_batches:
        batches["espn"][t] = {}
    for r in rows:
        src, cap, team, name, status = r[0], r[1], r[2], r[3], r[4]
        if gd and src == "nba_official" and r[5] is not None and str(r[5]) != str(day):
            continue
        b = batches[src].setdefault(cap, {})
        tm = b.setdefault(team, {})
        k = INJ.norm_name(name)
        if not k:
            continue
        if k not in tm or INJ.severity(status) > INJ.severity(tm[k][1]):
            tm[k] = (name, status)
    # per (source, team, player): series of (captured_at, sev, status, name) over covering batches
    series: dict[tuple[str, str, str], list[tuple]] = {}
    covering: dict[tuple[str, str], list[str]] = {}
    for src, bs in batches.items():
        for cap in sorted(bs):
            b = bs[cap]
            cov = teams if src == "espn" else [t for t in teams if t in b]
            for team in cov:
                covering.setdefault((src, team), []).append(cap)
                for k, (name, status) in b.get(team, {}).items():
                    series.setdefault((src, team, k), [])
    for (src, team, k) in list(series):
        out = []
        for cap in covering.get((src, team), []):
            nm_st = batches[src][cap].get(team, {}).get(k)
            st = nm_st[1] if nm_st else None
            out.append((cap, INJ.severity(st), st, nm_st[0] if nm_st else None))
        series[(src, team, k)] = out
    events: dict[tuple[str, str], dict] = {}
    by_player: dict[tuple[str, str], list[tuple[str, list]]] = {}
    for (src, team, k), ser in series.items():
        by_player.setdefault((team, k), []).append((src, ser))
    for key, per in by_player.items():
        latest = max(((ser[-1][0], src, ser[-1]) for src, ser in per if ser), default=None)
        if latest is None or latest[2][1] < 2:
            continue                                  # newest word: likely to play
        best = None
        for src, ser in per:
            if not ser or ser[-1][1] < 2:
                continue
            i = len(ser) - 1
            while i > 0 and ser[i - 1][1] >= 2:
                i -= 1
            if i == 0:
                cand = (None, src, None, ser[0])       # Doubtful / Out from its first batch
            else:
                prev = ser[i - 1]
                if prev[1] >= 2:
                    continue
                cand = (ser[i][0], src, prev[2], ser[i])
            if best is None or (cand[0] is None) or (best[0] is not None and cand[0] < best[0]):
                best = cand
            if best[0] is None:
                break
        if best is None:
            continue
        at, src, frm, row = best
        name = next((r[3] for _, ser in per for r in reversed(ser) if r[3]), key[1])
        events[key] = {"team": key[0], "name": name, "at": at, "source": src,
                       "status_from": frm or ("Not listed" if at else None),
                       "status_to": latest[2][2]}
    return events


# ============================================================================== run data

def _run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    r = conn.execute("SELECT run_id, slate_id, kind, created_at, model_version, detail, ok "
                     "FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
    if r is None:
        return None
    return {"run_id": r[0], "slate_id": r[1], "kind": r[2], "created_at": r[3],
            "model_version": r[4], "detail": r[5], "ok": r[6]}


def _slate(conn: sqlite3.Connection, slate_id) -> dict:
    r = conn.execute("SELECT nba_day, lock_utc, tag FROM dfs_slates WHERE slate_id=?",
                     (str(slate_id),)).fetchone()
    return {"nba_day": r[0], "lock_utc": r[1], "tag": r[2]} if r else {}


def _lock(conn: sqlite3.Connection, slate_id, slate: dict) -> datetime | None:
    t = _ts(slate.get("lock_utc"))
    if t is not None:
        return t
    r = conn.execute("SELECT MIN(start_time) FROM dfs_salaries WHERE slate_id=?",
                     (str(slate_id),)).fetchone()
    return _ts(r[0]) if r else None


def run_players(conn: sqlite3.Connection, run_id: str) -> dict[int, dict]:
    """{NBA player id: {team, role, p_play, dk_id, game_id, min (if he plays), dk (all sims),
    dk_played}} from a run's ``player_projections`` (+ stats)."""
    out: dict[int, dict] = {}
    for pid, team, role, p_play, inputs, gid, stat, cond, mean in conn.execute(
            "SELECT pp.player_id, pp.team, pp.role, pp.p_play, pp.inputs_json, pp.game_id, "
            "s.stat, s.cond, s.mean FROM player_projections pp LEFT JOIN "
            "player_projection_stats s ON s.projection_id=pp.id AND s.stat IN ('min','dk') "
            "WHERE pp.run_id=?", (run_id,)):
        d = out.get(int(pid))
        if d is None:
            dk = (_loads(inputs, {}) or {}).get("dk_id")
            dk = None if dk is None or str(dk) in ("", "nan", "None") else str(dk)
            d = out[int(pid)] = {"team": team, "role": role, "game_id": gid, "dk_id": dk,
                                 "p_play": None if p_play is None else float(p_play),
                                 "min": None, "min_all": None, "dk": None, "dk_played": None}
        if mean is None:
            continue
        key = {("min", "played"): "min", ("min", "all"): "min_all", ("dk", "all"): "dk",
               ("dk", "played"): "dk_played"}.get((stat, cond))
        if key:
            d[key] = float(mean)
    return out


def _names(conn: sqlite3.Connection, slate_id, at: datetime, players: Mapping[int, dict]
           ) -> dict[int, str]:
    """NBA id -> name: the slate's DK names (latest capture by `at`), else player_games."""
    dk_name: dict[str, str] = {}
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (str(slate_id), _iso(at))).fetchone()[0] or \
        conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=?",
                     (str(slate_id),)).fetchone()[0]
    if cap:
        for sid, nm in conn.execute("SELECT site_player_id, player_name FROM dfs_salaries "
                                    "WHERE slate_id=? AND captured_at=?", (str(slate_id), cap)):
            if sid is not None and nm:
                dk_name.setdefault(str(sid), nm)
    out = {pid: dk_name[d["dk_id"]] for pid, d in players.items()
           if d.get("dk_id") and d["dk_id"] in dk_name}
    miss = [pid for pid in players if pid not in out]
    for i in range(0, len(miss), 500):
        part = miss[i:i + 500]
        q = ",".join("?" * len(part))
        for pid, nm in conn.execute(f"SELECT player_id, MAX(player_name) FROM player_games "
                                    f"WHERE player_id IN ({q}) GROUP BY player_id", part):
            if nm:
                out[int(pid)] = nm
    return out


def baseline_run(conn: sqlite3.Connection, day: str, team: str, before: datetime,
                 model_version: str | None, exclude: str | None = None) -> dict | None:
    """The newest ok DFS run of NBA day `day` that projected `team`, created before `before`,
    with the same model version (module doc)."""
    rows = conn.execute(
        "SELECT r.run_id, r.created_at, r.model_version FROM dfs_runs r WHERE r.ok=1 AND "
        "r.created_at<? AND r.run_id IN (SELECT DISTINCT run_id FROM player_projections WHERE "
        "nba_day=? AND team=?) ORDER BY r.created_at DESC",
        (_iso(before), day, team)).fetchall()
    for rid, at, mv in rows:
        if rid == exclude or (model_version is not None and mv != model_version):
            continue
        return {"run_id": rid, "created_at": at, "model_version": mv}
    return None


def _ownership(conn: sqlite3.Connection, run_id: str) -> dict[str, tuple]:
    try:
        rows = conn.execute("SELECT player_id, own, band FROM dfs_ownership_proxy WHERE "
                            "run_id=?", (run_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): (None if r[1] is None else float(r[1]), r[2]) for r in rows}


# ============================================================================== the tag

def compute(conn: sqlite3.Connection, slate_id, run_id: str, now: datetime | str,
            model_version: str | None = None, *, window: timedelta = LATE_NEWS_WINDOW,
            per_team: int = PER_TEAM, min_gain_dk: float = MIN_GAIN_DK,
            min_gain_min: float = MIN_GAIN_MIN) -> dict:
    """The late-news tag of one run (module doc): {version, run_id, slate_id, lock, now,
    window_min, news: [...], flags: [...], own_uncertainty: {dk id: x}}. Read-only."""
    from ... import injuries as INJ
    now = _utc(now)
    sid = str(slate_id)
    slate = _slate(conn, sid)
    lock = _lock(conn, sid, slate)
    day = slate.get("nba_day")
    out = {"version": VERSION, "run_id": run_id, "slate_id": sid,
           "lock": _iso(lock) if lock else None, "now": _iso(now),
           "window_min": int(window.total_seconds() // 60), "news": [], "flags": [],
           "own_uncertainty": {}}
    if lock is None or not day:
        return out
    cur = run_players(conn, run_id)
    if not cur:
        return out
    if model_version is None:
        r = _run(conn, run_id)
        model_version = r["model_version"] if r else None
    until = min(now + CAPTURE_SLACK, lock)
    teams = {d["team"] for d in cur.values() if d.get("team")}
    events = news_events(conn, teams, day, lock - LOOKBACK, until)
    lo = lock - window
    late = {k: e for k, e in events.items()
            if e["at"] is not None and lo <= _ts(e["at"]) <= until}
    if not late:
        return out
    names = _names(conn, sid, now, cur)
    roster: dict[str, list[tuple[int, str]]] = {}
    for pid, d in cur.items():
        if pid in names:
            roster.setdefault(d["team"], []).append((pid, names[pid]))
    # map news rows -> NBA ids of the run
    news_by_team: dict[str, list[dict]] = {}
    for (team, _), e in late.items():
        pid, _how = INJ.match_player(e["name"], roster.get(team, []))
        if pid is None or pid not in cur:
            continue
        if INJ.severity(e["status_from"]) >= 2:
            continue
        news_by_team.setdefault(team, []).append({**e, "player_id": pid})
    own = _ownership(conn, run_id)
    base_cache: dict[str, dict] = {}
    for team, items in sorted(news_by_team.items()):
        t0 = min(_ts(e["at"]) for e in items)
        base = baseline_run(conn, day, team, t0, model_version, exclude=run_id)
        bp = base_cache.get(base["run_id"]) if base else None
        if base and bp is None:
            bp = base_cache[base["run_id"]] = run_players(conn, base["run_id"])
        bp = bp or {}
        kept = []
        for e in items:
            c, b = cur[e["player_id"]], bp.get(e["player_id"], {})
            mins = [m for m in (b.get("min"), c.get("min")) if m is not None]
            rot = (max(mins) if mins else 0.0) >= ROTATION_MIN or \
                (b.get("role") or c.get("role")) == "star"
            took_in = c.get("p_play") is None or c["p_play"] <= NEWS_P_PLAY_MAX
            at = _ts(e["at"])
            rec = {"player_id": e["player_id"], "dk_id": c.get("dk_id"),
                   "name": names.get(e["player_id"], e["name"]), "team": team,
                   "status_from": e["status_from"], "status_to": e["status_to"],
                   "source": e["source"], "at": e["at"],
                   "min_to_lock": round((lock - at).total_seconds() / 60.0, 1),
                   "base_min": b.get("min"), "base_dk": b.get("dk"),
                   "p_play": c.get("p_play"), "rotation": bool(rot), "taken_in": bool(took_in),
                   "baseline_run": base["run_id"] if base else None,
                   "baseline_at": base["created_at"] if base else None, "beneficiaries": []}
            out["news"].append(rec)
            if rot and took_in:
                kept.append(rec)
        if not kept or not base:
            continue
        news_ids = {e["player_id"] for e in items}
        cands = []
        for pid, c in cur.items():
            if c.get("team") != team or pid in news_ids or not c.get("dk_id"):
                continue
            if c.get("p_play") is None or c["p_play"] < P_PLAY_MIN:
                continue
            b = bp.get(pid)
            if not b:
                continue
            g_dk = (c["dk"] - b["dk"]) if c.get("dk") is not None and b.get("dk") is not None \
                else None
            g_min = (c["min"] - b["min"]) if c.get("min") is not None and \
                b.get("min") is not None else None
            if not ((g_dk is not None and g_dk >= min_gain_dk) or
                    (g_min is not None and g_min >= min_gain_min)):
                continue
            cands.append((pid, c, b, g_dk, g_min))
        cands.sort(key=lambda x: (-(x[3] if x[3] is not None else -99.0),
                                  -(x[4] if x[4] is not None else -99.0), x[0]))
        first = min(kept, key=lambda e: e["at"])
        for pid, c, b, g_dk, g_min in cands[:per_team]:
            o, band = own.get(c["dk_id"], (None, None))
            flag = {"player_id": pid, "dk_id": c["dk_id"], "name": names.get(pid, str(pid)),
                    "team": team,
                    "news": [{k: e[k] for k in ("player_id", "dk_id", "name", "status_from",
                                                "status_to", "source", "at", "min_to_lock")}
                             for e in kept],
                    "news_at": first["at"], "min_to_lock": first["min_to_lock"],
                    "gain_dk": None if g_dk is None else round(g_dk, 2),
                    "gain_min": None if g_min is None else round(g_min, 2),
                    "proj_dk": c.get("dk"), "proj_min": c.get("min"), "p_play": c.get("p_play"),
                    "base_dk": b.get("dk"), "base_min": b.get("min"),
                    "baseline_run": base["run_id"], "baseline_at": base["created_at"],
                    "own": o, "band": band, "own_uncertainty": OWN_UNCERTAINTY}
            out["flags"].append(flag)
            out["own_uncertainty"][c["dk_id"]] = OWN_UNCERTAINTY
            for e in kept:
                e["beneficiaries"].append(pid)
    out["news"].sort(key=lambda e: (e["at"] or "", e["name"] or ""))
    return out


def compute_safe(conn: sqlite3.Connection, slate_id, run_id: str, now, model_version=None,
                 **kw) -> dict | None:
    """`compute` that never raises (the tag must never fail a lock run)."""
    try:
        return compute(conn, slate_id, run_id, now, model_version, **kw)
    except Exception:  # noqa: BLE001
        log.exception("late-news tag failed for run %s", run_id)
        return None


def _build_run(conn: sqlite3.Connection, r: dict) -> dict | None:
    """A swap run reuses the projections of the build it swaps: the newest ok non-swap run of
    the slate at or before it."""
    if r["kind"] != "swap":
        return r
    b = conn.execute("SELECT run_id FROM dfs_runs WHERE slate_id=? AND ok=1 AND kind!='swap' "
                     "AND created_at<=? ORDER BY created_at DESC LIMIT 1",
                     (r["slate_id"], r["created_at"])).fetchone()
    return _run(conn, b[0]) if b else None


def run_flags(conn: sqlite3.Connection, run_id: str | None) -> dict:
    """The run's stored tag (``dfs_runs.detail["late_news_tag"]``, same `VERSION`), else
    computed from the database as of the run's time and cached. {} on any problem."""
    if not run_id:
        return {}
    if run_id in _CACHE:
        return _CACHE[run_id]
    try:
        r = _run(conn, run_id)
        if r is None or not r["ok"]:
            return {}
        r = _build_run(conn, r)
        if r is None:
            return {}
        det = _loads(r["detail"], {}) or {}
        res = det.get(DETAIL_KEY) if isinstance(det, dict) else None
        if not (isinstance(res, dict) and res.get("version") == VERSION):
            res = compute(conn, r["slate_id"], r["run_id"], r["created_at"], r["model_version"])
    except (sqlite3.Error, ValueError, KeyError, TypeError):
        log.exception("late-news tag unavailable for run %s", run_id)
        return {}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[run_id] = res
    return res


def flags_by_dk(res: Mapping | None) -> dict[str, dict]:
    return {str(f["dk_id"]): f for f in (res or {}).get("flags") or [] if f.get("dk_id")}


def flags_by_player(res: Mapping | None) -> dict[int, dict]:
    return {int(f["player_id"]): f for f in (res or {}).get("flags") or []}


def own_uncertainty(res: Mapping | None) -> dict[str, float]:
    """{dk id: ownership-uncertainty multiplier} of the flagged players (D5 hook); everybody
    else is 1.0 (use ``.get(dk_id, 1.0)``)."""
    return {str(k): float(v) for k, v in ((res or {}).get("own_uncertainty") or {}).items()}


def _clock(at: str | None) -> str:
    t = _ts(at)
    if t is None:
        return ""
    from zoneinfo import ZoneInfo
    et = t.astimezone(ZoneInfo("America/New_York"))
    return et.strftime("%I:%M %p ET").lstrip("0")


def label(flag: Mapping) -> str:
    """Plain English for the tag's title (information only, STYLE.md)."""
    parts = []
    for n in flag.get("news") or []:
        src = SOURCE_LABEL.get(n.get("source"), n.get("source") or "")
        parts.append(f"{n['name']} listed {n['status_to']} at {_clock(n['at'])} on the {src}, "
                     f"{n['min_to_lock']:.0f} minutes before the slate's first game")
    gains = []
    if flag.get("gain_dk") is not None and flag["gain_dk"] > 0:
        gains.append(f"{flag['gain_dk']:+.1f} DK points")
    if flag.get("gain_min") is not None and abs(flag["gain_min"]) >= 0.5:
        gains.append(f"{flag['gain_min']:+.0f} expected minutes")
    s = "Late news: " + "; ".join(parts) + "."
    if gains:
        s += f" Our projection for him moved {' and '.join(gains)} since the " \
             f"{_clock(flag.get('baseline_at'))} run."
    if flag.get("band"):
        s += (f" Estimated popularity ({flag['band']}) is a proxy and may not reflect the "
              "news yet.")
    return s + " Estimates, not advice."
