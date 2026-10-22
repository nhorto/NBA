"""The `/dfs/weekly` page: the weekly DFS report (T3.3, `proj/reports.py`) in the site's look.

Routes
  GET /dfs/weekly                 the newest graded week (the 7 days ending at the newest graded day)
  GET /dfs/weekly/<YYYY-MM-DD>    the 7 days ending at that NBA day

Read-only, like `/health/dfs`: the report is computed on the web app's read-only connection and
cached until the nightly grading (or an upload) changes something. Real contest numbers and
simulated ones are kept apart by the report itself; this module only lays it out.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from fastapi.responses import HTMLResponse

from ..proj import reports as RP
from . import fmt

_CACHE: dict = {}
_CACHE_MAX = 12
_NEG = re.compile(r"(?<![\w.])-(?=\d)")


def html_text(s) -> str:
    """Report text with typographic minus and plus-minus signs (the report itself is ASCII)."""
    s = str(s).replace("+-", "±")
    return _NEG.sub(fmt.MINUS, s)


def _state_token(conn) -> tuple:
    """Changes whenever grading, an upload or an entry could change the report."""
    def one(sql):
        try:
            return conn.execute(sql).fetchone()[0]
        except Exception:  # noqa: BLE001 - a missing table is just "nothing yet"
            return None
    return (one("SELECT MAX(graded_at) FROM player_projection_grades"),
            one("SELECT MAX(graded_at) FROM dfs_lineup_grades"),
            one("SELECT COUNT(*) FROM dfs_entries"),
            one("SELECT MAX(uploaded_at) FROM dfs_uploads"))


def prepare(rep: dict) -> dict:
    """Add display helpers: per-table numeric columns and HTML-ready cell text."""
    out = dict(rep)
    secs = []
    for sec in rep["sections"]:
        s = dict(sec)
        tabs = []
        for t in sec["tables"]:
            rows = [[html_text(c) for c in r] for r in t["rows"]]
            num = [i > 0 and bool(rows) and all(RP._numlike(str(r[i])) for r in t["rows"])
                   for i in range(len(t["columns"]))]
            tabs.append({**t, "rows": rows, "num": num,
                         "columns": [html_text(c) for c in t["columns"]]})
        s["tables"] = tabs
        s["lines"] = [html_text(x) for x in sec["lines"]]
        secs.append(s)
    out["sections"] = secs
    out["headline"] = [html_text(x) for x in rep["headline"]]
    return out


def build(conn, week_end: date | None, today: date, days: int = 7) -> dict:
    """The page's view model."""
    latest = RP.latest_graded_day(conn) if conn is not None else None
    end = week_end or latest or today - timedelta(days=1)
    vm = {"has_db": conn is not None, "week_end": end.isoformat(), "latest": latest,
          "is_latest": week_end is None or (latest is not None and end >= latest),
          "prev_end": (end - timedelta(days=days)).isoformat(),
          "next_end": (min(end + timedelta(days=days), latest).isoformat()
                       if latest is not None and end < latest else None),
          "days": days, "rep": None}
    if conn is None:
        return vm
    key = (end.isoformat(), days) + _state_token(conn)
    rep = _CACHE.get(key)
    if rep is None:
        rep = prepare(RP.weekly_dfs_report(conn, end, days=days))
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = rep
    vm["rep"] = rep
    return vm


def register(app, *, store, render, not_found, today, health_banner=None) -> None:
    """Add the /dfs/weekly routes. Register BEFORE `dfs_view.register` ('/dfs/{day}' and
    '/dfs/{day}/{slate}' would otherwise capture them)."""

    def banner():
        return health_banner() if health_banner else None

    def page(week_end: date | None):
        conn = store.health_conn()
        try:
            vm = build(conn, week_end, today())
        finally:
            if conn is not None:
                conn.close()
        return render("dfs_weekly.html", v=vm, page="dfs", health_banner=banner())

    @app.get("/dfs/weekly", response_class=HTMLResponse)
    def weekly_latest():
        return page(None)

    @app.get("/dfs/weekly/{day}", response_class=HTMLResponse)
    def weekly_for(day: str):
        d = fmt.parse_day(day)
        if d is None or len(day) != 10:
            return not_found("Not a date", "Weekly reports look like /dfs/weekly/2026-10-26.")
        return page(d)
