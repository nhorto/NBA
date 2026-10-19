"""The `/dfs` page: slate tabs, cash lineup, Top N, player pool, "why this lineup", DK CSV.

Read-only: every query runs on the web app's read-only SQLite connection and never migrates.
The view model is built here (`build_page`); `register()` adds the routes to the FastAPI app.

Routes
  /dfs                                  redirect to today's NBA day
  /dfs/<day>                            the day's default slate (main first)
  /dfs/<day>/<slate_id>                 one slate
  /dfs/<day>/<slate_id>/export.csv      ?mode=topn|cash &n=10 &fmt=id|name_id   (DK upload CSV)
  /dfs/<day>/<slate_id>/lineup.txt      ?mode=topn|cash &rank=1                 (copyable text)

Pool query params: sort=proj|value|salary|pplay|name, dir=asc|desc, pos=PG|SG|SF|PF|C, q=<name>,
hide=out, all=1. Top N: n=1..50 (default 10).

Data contract (see docs/PLAYERS.md §6) for what the page reads and tolerates:
  dfs_slates / dfs_salaries   slate list, games, lock time, pool (latest capture per slate)
  dfs_runs                    newest ok run (with lineups if any); model_version drives the badge:
                              empty / 'fppg*' / '*fallback*' = DraftKings FPPG fallback, shown
                              with a "Beta" badge; a version containing 'beta' is also "Beta"
  dfs_lineups                 cash / topn rows; slots_json is the list written by
                              `optimizer.save_lineups` (slot, player_id, name, team, salary, mean,
                              draftable_id); p10/p50/p90/p_cash/why_json are optional
  player_projections(+stats)  per-player 'dk' mean (cond 'all') and range (cond 'played'), 'min',
                              p_play, inputs_json (keys documented in proj/dfs/why.py); matched to
                              DK ids through player_ids(source='dk')
  dfs_runs.late_news_json     a list (or {"players": [...]}) of {name | player, player_id?, team?,
                              reason | why | change}
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode

from fastapi import Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from markupsafe import Markup, escape

from ..proj.dfs import export as dfs_export
from ..proj.dfs import why as dfs_why
from . import fmt

TAG_ORDER = {"main": 0, "early": 1, "late": 2, "turbo": 3, "showdown": 9}
TAG_LABEL = {"main": "Main", "early": "Early", "late": "Late", "turbo": "Turbo",
             "showdown": "Showdown"}
RUN_LABEL = {"lock45": "45-minute run", "lock10": "10-minute run", "swap": "Late-swap run",
             "manual": "Manual run", "topn": "Manual run", "cash": "Manual run"}
POS_FILTERS = ("PG", "SG", "SF", "PF", "C")
SORTS = ("proj", "value", "salary", "pplay", "name")
POOL_DEFAULT_ROWS = 40
TOPN_DEFAULT, TOPN_MAX = 10, 50
STATUS_LABEL = {"OUT": "Out", "O": "Out", "Q": "Q", "GTD": "GTD", "D": "D"}
STATUS_CLASS = {"OUT": "out", "O": "out", "Q": "questionable", "GTD": "questionable",
                "D": "doubtful"}
CAP = 50_000


# ===================================================================== small helpers


def _runner_reason(it: Mapping) -> str:
    """Plain-English text for a `runner.late_news` entry (structured `reasons` + details)."""
    parts = []
    for r in it.get("reasons") or ():
        if r == "status":
            st = it.get("report_status") or it.get("status") or {}
            parts.append(f"status {st.get('from') or 'none'} → {st.get('to') or 'none'}")
        elif r == "unexpected_starter":
            parts.append("named a starter")
        elif r == "unexpected_non_starter":
            parts.append("not in the starting lineup")
        elif r == "minutes":
            m = it.get("minutes") or {}
            parts.append(f"projected minutes {m.get('from')} → {m.get('to')}")
        else:
            parts.append(str(r).replace("_", " "))
    return "; ".join(parts)

def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _dt(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _json(s: Any, default: Any = None) -> Any:
    if s is None or s == "":
        return default
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return default


def money(x: Any) -> str:
    v = _num(x)
    return "" if v is None else f"${int(round(v)):,}"


def _clock(dt: datetime | None) -> str:
    if dt is None:
        return ""
    t = dt.astimezone(fmt.ET)
    return f"{t:%I:%M}".lstrip("0") + f" {t:%p}"


def _et(dt: datetime | None) -> str:
    return f"{_clock(dt)} ET" if dt else ""


def _rows(conn: sqlite3.Connection, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql, tuple(args)).fetchall()
    except sqlite3.Error:
        return []


def _chunks(seq: list, n: int = 400):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _norm_name(s: Any) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def duration_text(delta: timedelta) -> str:
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "under 1m"
    h, m = divmod(mins, 60)
    if h and m:
        return f"{h}h {m}m"
    return f"{h}h" if h else f"{m}m"


def lock_text(lock: datetime | None, now: datetime) -> str:
    if lock is None:
        return ""
    delta = lock - now
    if delta.total_seconds() > 0:
        if delta >= timedelta(hours=24):
            t = lock.astimezone(fmt.ET)
            return f"Locks {t:%a} {_clock(lock)} ET"
        return f"Locks in {duration_text(delta)}"
    return f"Locked at {_et(lock)}"


# ===================================================================== svg

def range_svg(lo: float | None, mid: float | None, hi: float | None, vmax: float,
              label: str = "Range", width: int = 112) -> Markup:
    """p10-p90 bar with a dot at the projection, drawn on a 0..vmax track."""
    if mid is None or vmax <= 0:
        return Markup("")
    h, pad = 14, 5
    span = width - 2 * pad

    def x(v: float) -> float:
        return pad + span * min(max(v / vmax, 0.0), 1.0)
    parts = [f'<line class="rng-track" x1="{pad}" x2="{width - pad}" y1="7" y2="7"/>']
    desc = f"{label}: projection {mid:.1f}"
    if lo is not None and hi is not None and hi >= lo:
        parts.append(f'<line class="rng-span" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="7" y2="7"/>')
        desc = f"{label}: {lo:.0f} to {hi:.0f}, projection {mid:.1f}"
    parts.append(f'<circle class="rng-dot" cx="{x(mid):.1f}" cy="7" r="3.6"/>')
    return Markup(f'<svg class="rng" viewBox="0 0 {width} {h}" width="{width}" height="{h}" '
                  f'role="img" aria-label="{escape(desc)}"><title>{escape(desc)}</title>'
                  + "".join(parts) + "</svg>")


# ===================================================================== data access

def day_list(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in _rows(conn, "SELECT DISTINCT nba_day FROM dfs_slates "
                                      "WHERE nba_day IS NOT NULL ORDER BY nba_day")]


def day_slates(conn: sqlite3.Connection, day: str) -> list[dict]:
    rows = [dict(r) for r in _rows(conn, "SELECT * FROM dfs_slates WHERE nba_day=?", (day,))]
    for r in rows:
        r["slate_id"] = str(r["slate_id"])
        r["game_ids"] = _json(r.get("game_ids_json"), []) or []
        r["lock"] = _dt(r.get("lock_utc"))
    rows.sort(key=lambda r: (TAG_ORDER.get((r.get("tag") or "").lower(), 5),
                             r["lock"] or datetime.max.replace(tzinfo=timezone.utc),
                             r["slate_id"]))
    return rows


def _latest_capture(conn: sqlite3.Connection, slate_id: str) -> str | None:
    r = _rows(conn, "SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=?", (slate_id,))
    return r[0][0] if r and r[0][0] else None


def salary_pool(conn: sqlite3.Connection, slate_id: str) -> tuple[list[dict], str | None]:
    """One dict per DK player of the slate at its latest capture (Showdown CPT rows folded)."""
    cap = _latest_capture(conn, slate_id)
    if cap is None:
        return [], None
    rows = _rows(conn, "SELECT * FROM dfs_salaries WHERE slate_id=? AND captured_at=? "
                       "AND disabled=0 ORDER BY id", (slate_id, cap))
    seen: dict[str, dict] = {}
    for r in rows:
        key = str(r["site_player_id"] or f"name:{r['player_name']}")
        if key in seen:
            continue
        if r["roster_slot"] == "CPT" and any(
                (o["site_player_id"] == r["site_player_id"] and o["roster_slot"] != "CPT")
                for o in rows):
            continue
        status = (r["status"] or "").strip().upper()
        status = "" if status in ("", "NONE") else status
        st = r["starter"] if "starter" in r.keys() else None
        team, home, away = r["team"], r["home"], r["away"]
        opp = None
        if team and home and away:
            opp = f"vs {away}" if team == home else f"@ {home}"
        seen[key] = {
            "id": str(r["site_player_id"] or key), "name": r["player_name"], "team": team,
            "positions": [p for p in (r["positions"] or "").replace(",", "/").split("/") if p],
            "salary": int(r["salary"] or 0), "status": status,
            "starter": None if st is None else bool(st), "fppg": _num(r["fppg"]),
            "game_id": r["game_id"], "game": r["game_label"], "opp": opp,
            "start": r["start_time"]}
    return list(seen.values()), cap


def slate_games(pool: list[dict]) -> list[dict]:
    games: dict[str, dict] = {}
    for p in pool:
        key = p.get("game_id") or p.get("game") or ""
        if key and key not in games:
            games[key] = {"label": p.get("game") or key, "start": _dt(p.get("start"))}
    return sorted(games.values(), key=lambda g: (g["start"] or datetime.max.replace(
        tzinfo=timezone.utc), g["label"]))


def slate_runs(conn: sqlite3.Connection, slate_id: str) -> list[dict]:
    return [dict(r) for r in _rows(
        conn, "SELECT * FROM dfs_runs WHERE slate_id=? ORDER BY created_at DESC, rowid DESC",
        (slate_id,))]


def _run_has_lineups(conn: sqlite3.Connection, run_id: str) -> bool:
    return bool(_rows(conn, "SELECT 1 FROM dfs_lineups WHERE run_id=? AND mode IN "
                            "('cash','topn','swap','showdown') LIMIT 1", (run_id,)))


def pick_run(conn: sqlite3.Connection, runs: list[dict]) -> tuple[dict | None, dict | None]:
    """(run to show, newest failed run newer than it or None). Prefers the newest ok run that
    has lineups, else the newest ok run."""
    ok = [r for r in runs if r.get("ok") in (1, True, None)]
    shown = next((r for r in ok if _run_has_lineups(conn, r["run_id"])), None) \
        or (ok[0] if ok else None)
    failed = None
    if runs and runs[0].get("ok") == 0 and (shown is None or runs[0]["created_at"] > shown["created_at"]):
        failed = runs[0]
    return shown, failed


def mode_run(conn: sqlite3.Connection, runs: list[dict], mode: str) -> dict | None:
    """Newest ok run that has lineups of `mode` ('cash' / 'topn'). Manual CLI builds write one
    run per mode, so each section picks its own run."""
    for r in runs:
        if r.get("ok") in (1, True, None) and _rows(
                conn, "SELECT 1 FROM dfs_lineups WHERE run_id=? AND mode=? LIMIT 1",
                (r["run_id"], mode)):
            return r
    return None


def load_lineups(conn: sqlite3.Connection, run_id: str, mode: str, limit: int) -> list[dict]:
    out = []
    for r in _rows(conn, "SELECT * FROM dfs_lineups WHERE run_id=? AND mode=? "
                         "ORDER BY rank, id LIMIT ?", (run_id, mode, limit)):
        d = dict(r)
        slots = _json(d.get("slots_json"), [])
        d["slots"] = [s for s in slots if isinstance(s, dict)] if isinstance(slots, list) else []
        out.append(d)
    return out


def dk_to_nba(conn: sqlite3.Connection) -> dict[str, int]:
    return {str(r[0]): int(r[1]) for r in _rows(
        conn, "SELECT source_id, player_id FROM player_ids WHERE source='dk' "
              "AND player_id IS NOT NULL")}


def load_projections(conn: sqlite3.Connection, run: dict | None, game_ids: list[str],
                     ) -> dict:
    """The newest projection set for the slate: the shown run's rows if it has any, else the
    newest run_id with rows for the slate's games. Returns {model_version, created_at,
    by_nba_id: {id: {...}}} (empty by_nba_id when nothing is stored)."""
    run_id = None
    if run and _rows(conn, "SELECT 1 FROM player_projections WHERE run_id=? LIMIT 1",
                     (run["run_id"],)):
        run_id = run["run_id"]
    elif game_ids:
        q = ",".join("?" * len(game_ids))
        r = _rows(conn, f"SELECT run_id FROM player_projections WHERE run_id IS NOT NULL AND "
                        f"game_id IN ({q}) ORDER BY created_at DESC LIMIT 1", game_ids)
        run_id = r[0][0] if r else None
    out: dict = {"model_version": None, "created_at": None, "run_id": run_id, "by_nba_id": {}}
    if run_id is None:
        return out
    prows = _rows(conn, "SELECT * FROM player_projections WHERE run_id=? "
                        "ORDER BY created_at, id", (run_id,))
    if run and run.get("model_version"):
        mv = [p for p in prows if p["model_version"] == run["model_version"]]
        prows = mv or prows
    by_pid: dict[int, sqlite3.Row] = {}
    for p in prows:                     # later rows win (same player in several games: rare)
        by_pid[int(p["player_id"])] = p
    stats: dict[int, dict] = {}
    ids = [p["id"] for p in by_pid.values()]
    for chunk in _chunks(ids):
        q = ",".join("?" * len(chunk))
        for s in _rows(conn, "SELECT projection_id, stat, cond, mean, sd, q10, q90 FROM "
                             f"player_projection_stats WHERE projection_id IN ({q})", chunk):
            stats.setdefault(s["projection_id"], {})[(s["stat"], s["cond"])] = dict(s)
    for pid, p in by_pid.items():
        st = stats.get(p["id"], {})
        dk_all, dk_pl = st.get(("dk", "all")), st.get(("dk", "played"))
        mn = st.get(("min", "played")) or st.get(("min", "all"))
        mean = _num((dk_all or dk_pl or {}).get("mean"))
        out["by_nba_id"][pid] = {
            "mean": mean,
            "q10": _num((dk_pl or dk_all or {}).get("q10")),
            "q90": _num((dk_pl or dk_all or {}).get("q90")),
            "min": _num((mn or {}).get("mean")),
            "p_play": _num(p["p_play"]),
            "inputs": _json(p["inputs_json"], {}) or {}}
    first = prows[0] if prows else None
    if first is not None:
        out["model_version"] = first["model_version"]
        out["created_at"] = first["created_at"]
    return out


def is_fallback(model_version: str | None) -> bool:
    mv = (model_version or "").strip().lower()
    return not mv or "fppg" in mv or "fallback" in mv


def projection_badge(model_version: str | None) -> dict:
    """{label, beta, fallback}: what the header says about the projection source."""
    if is_fallback(model_version):
        return {"label": "Projections: DraftKings FPPG fallback", "beta": True, "fallback": True}
    return {"label": f"Projections: model {model_version}",
            "beta": "beta" in str(model_version).lower(), "fallback": False}


# ===================================================================== late news

def parse_late_news(raw: Any) -> list[dict]:
    data = _json(raw, None) if isinstance(raw, str) else raw
    if isinstance(data, Mapping):
        for k in ("players", "items", "late_news", "flagged"):
            if isinstance(data.get(k), (list, dict)):
                data = data[k]
                break
    if isinstance(data, Mapping):
        data = [{"name": k, **(v if isinstance(v, Mapping) else {"reason": v})}
                for k, v in data.items()]
    out = []
    for it in data or []:
        if isinstance(it, str):
            out.append({"name": it, "id": None, "team": None, "reason": ""})
            continue
        if not isinstance(it, Mapping):
            continue
        name = it.get("name") or it.get("player") or it.get("player_name")
        pid = next((it[k] for k in ("dk_player_id", "site_player_id", "player_id", "id")
                    if it.get(k) not in (None, "")), None)
        reason = _runner_reason(it) or next(
            (it[k] for k in ("reason", "why", "change", "detail", "note", "kind") if it.get(k)), "")
        if isinstance(reason, (list, tuple)):
            reason = "; ".join(str(x) for x in reason)
        if name or pid is not None:
            out.append({"name": str(name or pid), "id": None if pid is None else str(pid),
                        "team": it.get("team"), "reason": str(reason)})
    return out


def late_hits(slots: list[dict], news: list[dict]) -> list[str]:
    ids = {n["id"] for n in news if n["id"]}
    names = {_norm_name(n["name"]) for n in news}
    return [s.get("name") or str(s.get("player_id")) for s in slots
            if str(s.get("player_id")) in ids or _norm_name(s.get("name")) in names]


# ===================================================================== view model

def _pool_view(pool: list[dict], proj: dict, nba_ids: dict[str, int]) -> list[dict]:
    model = bool(proj["by_nba_id"])
    rows = []
    for p in pool:
        d = dict(p)
        pr = proj["by_nba_id"].get(nba_ids.get(p["id"], -1)) if model else None
        out = d["status"] in ("OUT", "O")
        if pr is not None:
            d.update(proj=pr["mean"], q10=pr["q10"], q90=pr["q90"], min_mean=pr["min"],
                     p_play=pr["p_play"], inputs=pr["inputs"], src="model")
        elif model:
            d.update(proj=None, q10=None, q90=None, min_mean=None, p_play=None, inputs={},
                     src="none")
        else:
            d.update(proj=d["fppg"], q10=None, q90=None, min_mean=None, p_play=None, inputs={},
                     src="fppg")
        if out and d["src"] != "model":
            d["proj"] = None
        d["value"] = dfs_why.per_k(d["proj"], d["salary"])
        d["salary_text"] = money(d["salary"])
        rows.append(d)
    return rows


def _sort_pool(rows: list[dict], sort: str, desc: bool) -> list[dict]:
    key = {"proj": lambda r: r["proj"], "value": lambda r: r["value"],
           "salary": lambda r: r["salary"], "pplay": lambda r: r["p_play"],
           "name": lambda r: _norm_name(r["name"])}[sort]
    have = [r for r in rows if key(r) is not None]
    miss = [r for r in rows if key(r) is None]
    have.sort(key=lambda r: (key(r), r["salary"], r["name"]), reverse=desc)
    return have + miss


def _eligible(positions: list[str], pos: str) -> bool:
    return pos in positions


def _qs(params: Mapping[str, Any], defaults: Mapping[str, Any], anchor: str = "") -> str:
    q = {k: v for k, v in params.items() if v not in (None, "") and defaults.get(k) != v}
    return ("?" + urlencode(q) if q else "") + anchor


def _enrich_lineup(row: dict, by_id: dict[str, dict], first_ids: set[str] | None,
                   news: list[dict], pool_for_why: list[dict], vmax: float) -> dict:
    slots = []
    for s in row["slots"]:
        p = by_id.get(str(s.get("player_id")), {})
        mean = _num(s.get("mean"))
        slots.append({
            "slot": s.get("slot"), "player_id": str(s.get("player_id")),
            "name": s.get("name") or p.get("name") or str(s.get("player_id")),
            "team": s.get("team") or p.get("team"), "salary": _num(s.get("salary")),
            "mean": mean, "salary_text": money(s.get("salary")), "status": p.get("status", ""), "p_play": p.get("p_play"),
            "q10": p.get("q10"), "q90": p.get("q90"), "draftable_id": s.get("draftable_id"),
            "positions": p.get("positions") or [],
            "bar": range_svg(p.get("q10"), mean, p.get("q90"), vmax) if p.get("q10") is not None
            else Markup("")})
        st = STATUS_LABEL.get(slots[-1]["status"])
        slots[-1]["status_label"] = st
        slots[-1]["status_class"] = STATUS_CLASS.get(slots[-1]["status"], "unknown")
    lu = {"id": row.get("id"), "rank": row.get("rank"), "mode": row.get("mode"),
          "salary": row.get("salary"), "mean": _num(row.get("mean")),
          "p10": _num(row.get("p10")), "p50": _num(row.get("p50")), "p90": _num(row.get("p90")),
          "p_cash": _num(row.get("p_cash")), "slots": slots, "raw_slots": row["slots"]}
    if lu["salary"] is None:
        lu["salary"] = int(sum(s["salary"] or 0 for s in slots))
    lu["salary_text"] = money(lu["salary"])
    lu["late"] = late_hits(row["slots"], news)
    ids = {s["player_id"] for s in slots}
    if first_ids is not None and first_ids:
        lu["chips_in"] = [s["name"] for s in slots if s["player_id"] not in first_ids]
        lu["chips_out"] = [by_id[i]["name"] for i in sorted(first_ids - ids) if i in by_id]
    else:
        lu["chips_in"], lu["chips_out"] = [], []
    # why: the stored one wins, else the template v0 from the pool
    why = dfs_why.loads(row.get("why_json")) or dfs_why.why_lineup(
        {"slots": row["slots"], "salary": lu["salary"], "mean": lu["mean"]}, pool_for_why,
        cap=CAP)
    lu["why"] = dfs_why.render_lines(why)
    lu["why_raw"] = why
    lu["text"] = dfs_export.copy_text(
        row["slots"], salary=lu["salary"], mean=lu["mean"],
        title=("Cash lineup" if row.get("mode") == "cash" else f"Top N #{row.get('rank')}"))
    return lu


def build_page(conn: sqlite3.Connection | None, day: str, slate_id: str | None, *,
               now: datetime, params: Mapping[str, str] | None = None) -> dict:
    params = dict(params or {})
    v: dict[str, Any] = {"day": day, "day_label": fmt.long_date(day) or day, "slate": None,
                         "slates": [], "state": "ok", "prev_day": None, "next_day": None,
                         "late": None, "cash": None, "topn": [], "pool": None, "contests": 0}
    days = day_list(conn) if conn is not None else []
    v["prev_day"] = next((d for d in reversed(days) if d < day), None)
    v["next_day"] = next((d for d in days if d > day), None)
    slates = day_slates(conn, day) if conn is not None else []
    if not slates:
        v["state"] = "no_slates"
        return v
    chosen = next((s for s in slates if s["slate_id"] == slate_id), None) if slate_id \
        else slates[0]
    if chosen is None:
        v["state"] = "no_such_slate"
        return v

    # ---- tabs
    pools: dict[str, tuple[list[dict], str | None]] = {}
    for s in slates:
        pl, cap = salary_pool(conn, s["slate_id"])
        pools[s["slate_id"]] = (pl, cap)
        games = slate_games(pl)
        tag = (s.get("tag") or "").lower()
        label = TAG_LABEL.get(tag, "Slate")
        if tag == "showdown" and len(games) == 1:
            label = f"Showdown {games[0]['label']}"
        n_games = len(games) or len(s["game_ids"])
        sub = f"{n_games} game{'s' if n_games != 1 else ''}"
        if s["lock"]:
            sub += f" · {_clock(s['lock'])}"
        v["slates"].append({"id": s["slate_id"], "label": label, "sub": sub,
                            "url": f"/dfs/{day}/{s['slate_id']}", "active": s is chosen})

    pool, captured = pools[chosen["slate_id"]]
    sid = chosen["slate_id"]
    games = slate_games(pool)
    is_showdown = (chosen.get("tag") or "").lower() == "showdown"
    runs = slate_runs(conn, sid)
    run, failed = pick_run(conn, runs)
    proj = load_projections(conn, run, [str(g) for g in chosen["game_ids"]])
    nba_ids = dk_to_nba(conn) if proj["by_nba_id"] else {}
    rows = _pool_view(pool, proj, nba_ids)
    by_id = {r["id"]: r for r in rows}

    mv = proj["model_version"] if proj["by_nba_id"] else None
    if mv is None and run and run.get("model_version") and not proj["by_nba_id"]:
        mv = run["model_version"] if not is_fallback(run["model_version"]) else None
    badge = projection_badge(mv)
    lock = chosen["lock"]
    captured_dt = _dt(captured)

    last_run = None
    if run:
        created = _dt(run["created_at"])
        label = RUN_LABEL.get(run["kind"], str(run["kind"]).capitalize())
        text = f"Last run: {label}, {_et(created)}"
        if created and lock and created < lock and lock - created <= timedelta(hours=3):
            text += f" ({duration_text(lock - created)} before lock)"
        last_run = {"text": text, "iso": run["created_at"], "kind": run["kind"]}
    v["slate"] = {
        "id": sid, "tag": chosen.get("tag"), "is_showdown": is_showdown,
        "games": [{"label": g["label"], "time": _et(g["start"])} for g in games],
        "lock_text": lock_text(lock, now), "lock_iso": chosen.get("lock_utc"),
        "locked": bool(lock and lock <= now), "last_run": last_run, "proj": badge,
        "captured_text": f"Salaries as of {fmt.stamp(captured)}" if captured else "",
        "failed_note": (f"The {_et(_dt(failed['created_at']))} run did not finish; showing the "
                        f"previous one." if failed else ""),
        "stale_note": ("These salaries are more than 18 hours old; they refresh before each run."
                       if captured_dt and now - captured_dt > timedelta(hours=18)
                       and not (lock and lock <= now) else ""),
        "run_id": run["run_id"] if run else None}

    # ---- lineups (each section reads its own run; a late-news list belongs to its run)
    vmax = max([r["q90"] for r in rows if r.get("q90")] + [40.0])
    vmax = math.ceil(vmax / 10.0) * 10
    pool_for_why = [{"id": r["id"], "name": r["name"], "team": r["team"], "salary": r["salary"],
                     "proj": r["proj"], "p_play": r["p_play"], "status": r["status"],
                     "game": r["game"], "min_mean": r.get("min_mean"),
                     "inputs": r.get("inputs")} for r in rows]
    try:
        n = max(1, min(TOPN_MAX, int(params.get("n", TOPN_DEFAULT))))
    except (TypeError, ValueError):
        n = TOPN_DEFAULT
    v["n"] = n
    topn: list[dict] = []
    cash_fallback = False
    topn_total = 0
    news_all: list[dict] = []
    cash_run, topn_run = mode_run(conn, runs, "cash"), mode_run(conn, runs, "topn")

    def news_of(r: dict | None) -> list[dict]:
        found = parse_late_news(r.get("late_news_json")) if r else []
        for it in found:
            if not any(o["name"] == it["name"] for o in news_all):
                news_all.append(it)
        return found

    def source_note(r: dict | None) -> str:
        if not r or (run and r["run_id"] == run["run_id"]):
            return ""
        return f"From the {_et(_dt(r['created_at']))} {RUN_LABEL.get(r['kind'], str(r['kind']).lower()).lower()}"

    topn_rows = load_lineups(conn, topn_run["run_id"], "topn", TOPN_MAX) if topn_run else []
    first_ids = {str(x.get("player_id")) for x in topn_rows[0]["slots"]} if topn_rows else set()
    cash_rows = load_lineups(conn, cash_run["run_id"], "cash", 1) if cash_run else []
    cash_src = cash_run
    if not cash_rows and topn_rows:
        cash_rows, cash_fallback, cash_src = topn_rows[:1], True, topn_run
    if cash_rows:
        v["cash"] = _enrich_lineup(cash_rows[0], by_id, None, news_of(cash_src), pool_for_why, vmax)
    v["cash_note"] = source_note(cash_src)
    topn_news = news_of(topn_run)
    topn_total = len(topn_rows)
    topn = [_enrich_lineup(r, by_id, first_ids if i else None, topn_news, pool_for_why, vmax)
            for i, r in enumerate(topn_rows[:n])]
    v["topn_note"] = source_note(topn_run)
    for i, lu in enumerate(topn):
        if i:
            best = topn[0]
            lu["d_mean"] = None if lu["mean"] is None or best["mean"] is None \
                else lu["mean"] - best["mean"]
            lu["d_salary"] = (lu["salary"] or 0) - (best["salary"] or 0)
        else:
            lu["d_mean"] = lu["d_salary"] = None
    shown = ([v["cash"]] if v["cash"] else []) + topn
    lows = [lu["p10"] for lu in shown if lu["p10"] is not None]
    highs = [lu["p90"] for lu in shown if lu["p90"] is not None]
    v["has_lineup_ranges"] = bool(lows and highs)
    if lows and highs:
        lo_s, hi_s = min(lows) * 0.97, max(highs) * 1.01
        for lu in shown:
            lu["bar"] = _lineup_bar(lu, lo_s, hi_s)
    v["topn"], v["topn_total"], v["cash_fallback"] = topn, topn_total, cash_fallback
    v["has_lineups"] = bool(v["cash"] or topn)

    # ---- late-news banner (lineups shown on the page, each counted once)
    if news_all:
        shown = {}
        for lu in ([v["cash"]] if v["cash"] else []) + topn:
            if lu["late"]:
                shown[lu.get("id") or id(lu)] = lu
        k = len(shown)
        if k:
            head = (f"{k} lineup contains a late-news player" if k == 1
                    else f"{k} lineups contain late-news players")
        else:
            head = (f"Late news on {len(news_all)} player"
                    f"{'s' if len(news_all) != 1 else ''}; none are in the lineups below")
        v["late"] = {"head": head, "players": news_all, "k": k}

    # ---- pool
    sort = params.get("sort") if params.get("sort") in SORTS else "proj"
    dflt_dir = "asc" if sort == "name" else "desc"
    direction = params.get("dir") if params.get("dir") in ("asc", "desc") else dflt_dir
    pos = (params.get("pos") or "").upper()
    pos = pos if pos in POS_FILTERS else ""
    q = (params.get("q") or "").strip()[:40]
    hide_out = params.get("hide") == "out"
    show_all = params.get("all") == "1"
    view_rows = rows
    if pos:
        view_rows = [r for r in view_rows if _eligible(r["positions"], pos)]
    if q:
        nq = _norm_name(q)
        view_rows = [r for r in view_rows if nq in _norm_name(r["name"])
                     or q.lower() == (r["team"] or "").lower()]
    if hide_out:
        view_rows = [r for r in view_rows if r["status"] not in ("OUT", "O")]
    view_rows = _sort_pool(view_rows, sort, direction == "desc")
    total = len(view_rows)
    shown_rows = view_rows if show_all else view_rows[:POOL_DEFAULT_ROWS]
    has_model_range = any(r.get("q10") is not None for r in rows)
    has_pplay = any(r.get("p_play") is not None for r in rows)
    has_starter = any(r.get("starter") is not None for r in rows)
    base = f"/dfs/{day}/{sid}"
    defaults = {"sort": "proj", "dir": "desc"}
    cur = {"sort": sort, "dir": direction, "pos": pos, "q": q,
           "hide": "out" if hide_out else "", "all": "1" if show_all else "", "n": params.get("n")}

    def url(anchor="#pool", **over):
        merged = {**cur, **over}
        if merged.get("sort") == "proj" and merged.get("dir") == "desc":
            merged.pop("sort", None)
            merged.pop("dir", None)
        elif merged.get("sort") == "name" and merged.get("dir") == "asc":
            merged.pop("dir", None)
        return base + _qs(merged, {}, anchor)

    heads = []
    for key, label, title in (("name", "Player", "Sort by name"),
                              ("salary", "Salary", "Sort by salary"),
                              ("proj", "Proj", "Sort by projected DK points"),
                              ("value", "Per $1k", "Sort by projected points per $1,000"),
                              ("pplay", "P(play)", "Sort by chance to play")):
        if key == "pplay" and not has_pplay:
            continue
        active = sort == key
        nxt = ("asc" if direction == "desc" else "desc") if active else \
            ("asc" if key == "name" else "desc")
        heads.append({"key": key, "label": label, "title": title, "url": url(sort=key, dir=nxt),
                      "aria": ("descending" if direction == "desc" else "ascending")
                      if active else "none",
                      "arrow": ("▼" if direction == "desc" else "▲") if active else ""})
    for r in shown_rows:
        r["bar"] = range_svg(r.get("q10"), r.get("proj"), r.get("q90"), vmax) \
            if r.get("q10") is not None else Markup("")
        r["status_label"] = STATUS_LABEL.get(r["status"])
        r["status_class"] = STATUS_CLASS.get(r["status"], "unknown")
    v["pool"] = {
        "rows": shown_rows, "total": total, "all_total": len(rows), "shown": len(shown_rows),
        "show_all": show_all, "more_url": url(all="1") if not show_all and total > len(shown_rows)
        else "", "less_url": url(all="") if show_all and total > POOL_DEFAULT_ROWS else "",
        "heads": heads, "sort": sort, "dir": direction, "pos": pos, "q": q, "hide_out": hide_out,
        "has_range": has_model_range, "has_pplay": has_pplay, "has_starter": has_starter,
        "source": "model" if proj["by_nba_id"] else "fppg",
        "pos_chips": [{"label": "All", "url": url(pos="", all=""), "active": not pos}] + [
            {"label": p, "url": url(pos=p, all=""), "active": pos == p} for p in POS_FILTERS],
        "hide_url": url(hide="" if hide_out else "out", all=""),
        "action": base, "hidden": {k: val for k, val in
                                   (("sort", sort if sort != "proj" else ""),
                                    ("dir", direction if (sort, direction) != ("proj", "desc")
                                     and not (sort == "name" and direction == "asc") else ""),
                                    ("pos", pos), ("hide", "out" if hide_out else ""))
                                   if val}}

    # ---- contests and export links
    cnt = _rows(conn, "SELECT COUNT(*) FROM dfs_contests WHERE slate_id=?", (sid,))
    v["contests"] = int(cnt[0][0]) if cnt else 0
    v["export"] = {"topn": f"{base}/export.csv?mode=topn&n={n}",
                   "cash": f"{base}/export.csv?mode=cash", "base": base}
    v["empty_msg"] = (
        "No lineups were built for this slate." if v["slate"]["locked"] else
        "Lineups appear 45 minutes before lock. Until then, the pool below uses DraftKings "
        "FPPG." if badge["fallback"] else
        "Lineups appear 45 minutes before lock.")
    return v


def _lineup_bar(lu: dict, lo_s: float, hi_s: float) -> Markup:
    if lu["mean"] is None or hi_s <= lo_s:
        return Markup("")
    width, h, pad = 120, 14, 5
    span = width - 2 * pad

    def x(val: float) -> float:
        return pad + span * min(max((val - lo_s) / (hi_s - lo_s), 0.0), 1.0)
    parts = [f'<line class="rng-track" x1="{pad}" x2="{width - pad}" y1="7" y2="7"/>']
    desc = f"Projection {lu['mean']:.1f}"
    if lu["p10"] is not None and lu["p90"] is not None:
        parts.append(f'<line class="rng-span" x1="{x(lu["p10"]):.1f}" x2="{x(lu["p90"]):.1f}" '
                     f'y1="7" y2="7"/>')
        desc = f"Simulated range, 10th to 90th percentile {lu['p10']:.0f} to {lu['p90']:.0f}, " \
               f"projection {lu['mean']:.1f}"
    parts.append(f'<circle class="rng-dot" cx="{x(lu["mean"]):.1f}" cy="7" r="3.6"/>')
    return Markup(f'<svg class="rng" viewBox="0 0 {width} {h}" width="{width}" height="{h}" '
                  f'role="img" aria-label="{escape(desc)}"><title>{escape(desc)}</title>'
                  + "".join(parts) + "</svg>")


# ===================================================================== routes

def register(app, *, store, render: Callable, not_found: Callable, today: Callable[[], date],
             health_banner: Callable | None = None) -> None:
    """Add the /dfs routes to `app`. `render` and `not_found` are the app's page helpers.
    Tests can set ``app.state.dfs_now`` to a callable returning an aware datetime."""
    def now() -> datetime:
        fn = getattr(app.state, "dfs_now", None)
        return fn() if fn else datetime.now(timezone.utc)

    def page(day: str, slate_id: str | None, request: Request):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "DFS URLs look like /dfs/2026-10-20.")
        conn = store.health_conn()
        try:
            vm = build_page(conn, day, slate_id, now=now(), params=dict(request.query_params))
        finally:
            if conn is not None:
                conn.close()
        if vm["state"] == "no_such_slate":
            return not_found("Slate not found", f"There is no slate {slate_id} on {day}.")
        return render("dfs.html", v=vm, page="dfs",
                      health_banner=health_banner() if health_banner else None)

    @app.get("/dfs")
    def dfs_today():
        return RedirectResponse(f"/dfs/{today().isoformat()}", status_code=307)

    @app.get("/dfs/{day}", response_class=HTMLResponse)
    def dfs_day(day: str, request: Request):
        return page(day, None, request)

    @app.get("/dfs/{day}/{slate_id}", response_class=HTMLResponse)
    def dfs_slate(day: str, slate_id: str, request: Request):
        return page(day, slate_id, request)

    def _selected(day: str, slate_id: str, mode: str, n: int | None, rank: int | None):
        """(lineup slot-lists, error Response | None) for the export routes."""
        conn = store.health_conn()
        if conn is None:
            return None, PlainTextResponse("No database.", status_code=404)
        try:
            if not any(s["slate_id"] == slate_id for s in day_slates(conn, day)):
                return None, PlainTextResponse("No such slate.", status_code=404)
            runs = slate_runs(conn, slate_id)
            rows = []
            if mode == "cash":
                r = mode_run(conn, runs, "cash")
                rows = load_lineups(conn, r["run_id"], "cash", 1) if r else []
            if not rows:
                r = mode_run(conn, runs, "topn")
                rows = load_lineups(conn, r["run_id"], "topn", TOPN_MAX if mode != "cash" else 1) \
                    if r else []
        finally:
            conn.close()
        if rank is not None:
            rows = [r for r in rows if r["rank"] == rank] or rows[rank - 1:rank]
        elif n is not None:
            rows = rows[:n]
        if not rows:
            return None, PlainTextResponse("No lineups yet for this slate.", status_code=404)
        return rows, None

    @app.get("/dfs/{day}/{slate_id}/export.csv")
    def dfs_export_csv(day: str, slate_id: str, mode: str = "topn", n: int = TOPN_DEFAULT,
                       style: str = Query("id", alias="fmt")):
        mode = "cash" if mode == "cash" else "topn"
        if style not in dfs_export.STYLES:
            return PlainTextResponse("fmt must be id or name_id.", status_code=400)
        rows, err = _selected(day, slate_id, mode, max(1, min(n, TOPN_MAX)), None)
        if err is not None:
            return err
        try:
            body = dfs_export.to_csv([r["slots"] for r in rows], style=style)
        except dfs_export.ExportError as e:
            return PlainTextResponse(f"Cannot export: {e}", status_code=409)
        name = f"courtside-dfs-{day}-{slate_id}-{mode}.csv"
        return Response(body, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"',
                                 "Cache-Control": "no-store"})

    @app.get("/dfs/{day}/{slate_id}/lineup.txt")
    def dfs_lineup_text(day: str, slate_id: str, mode: str = "cash", rank: int = 1):
        mode = "topn" if mode == "topn" else "cash"
        rows, err = _selected(day, slate_id, mode, None, rank if mode == "topn" else 1)
        if err is not None:
            return err
        r = rows[0]
        title = "Cash lineup" if mode == "cash" else f"Top N #{r['rank']}"
        text = dfs_export.copy_text(r["slots"], salary=r.get("salary"), mean=_num(r.get("mean")),
                                    title=title)
        return PlainTextResponse(text + "\n", headers={"Cache-Control": "no-store"})
