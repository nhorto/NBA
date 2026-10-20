"""The slate page's "Late swap" action: re-solve the unlocked slots, show what changes and why,
and hand back a DK edit CSV (docs/PLAYERS.md §2.5, task T2.14; the solver is proj/dfs/swap.py).

Routes
  GET  /dfs/<day>/<slate>/swap                our saved lineups, as of now (?scope=all|news,
                                              ?at=<ISO time> previews another moment)
  POST /dfs/<day>/<slate>/swap                multipart: file = DK entries CSV (+ scope)
  GET  /dfs/<day>/<slate>/swap.csv            ?fmt=name_id|id &scope=  (saved lineups)
                                              ?t=<token>               (an uploaded file's result)

Read-only: the plan is computed on the web app's read-only connection and an uploaded file is
never stored on disk or in the database (its result CSV is kept in memory for the download,
the last `MAX_RESULTS` only). Nothing here talks to DraftKings.
"""

from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone
from typing import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from ..proj.dfs import export as dfs_export
from ..proj.dfs import swap as SW
from . import dfs_view, fmt
from .dfs_results import parse_multipart, read_body, same_origin

MAX_UPLOAD = 5 * 1024 * 1024
MAX_RESULTS = 24
RESULT_TTL_S = 6 * 3600
STATUS = {"swapped": ("Swap proposed", "t-warn"), "unchanged": ("No change", "t-plain"),
          "all_locked": ("All locked", "t-plain"), "no_legal": ("No legal swap", "t-neg"),
          "unusable": ("Could not read", "t-neg")}
SCOPE_LABEL = {"all": "Every lineup", "news": "Only lineups with late news"}


def _money(x) -> str:
    return dfs_view.money(x) if x is not None else fmt.DASH


def _num(x, d=1) -> str:
    return fmt.DASH if x is None else f"{x:.{d}f}"


def _signed(x, d=1) -> str:
    if x is None:
        return fmt.DASH
    if abs(x) < 0.05:
        return "0.0"
    return f"+{x:.{d}f}" if x > 0 else f"{fmt.MINUS}{abs(x):.{d}f}"


def build(plan: SW.SwapPlan, *, day: str, slate_label: str, base: str, scope: str,
          token: str | None = None, at: str | None = None) -> dict:
    """The page's view model from a `SwapPlan`."""
    games = []
    for g in plan.games:
        games.append({"label": g["label"], "started": g["started"],
                      "time": dfs_view._et(g["start"]) if g["start"] else ""})
    n_started = sum(g["started"] for g in games)
    pj = plan.projection
    if pj.get("source") == "model":
        made = dfs_view._et(dfs_view._dt(pj.get("created_at")))
        proj = f"Projections: model {pj.get('model_version')}" + (f", run {made}" if made else "")
    elif pj.get("source") == "given":
        proj = "Projections: supplied"
    else:
        proj = "Projections: DraftKings FPPG fallback (no stored model run for this slate)"
    names = {pid: p.name for pid, p in plan.by_id.items()}
    entries = []
    for r in plan.results:
        e = r.entry
        label, cls = STATUS.get(r.status, (r.status, "t-plain"))
        title = e.label if plan.source == "saved" else (e.contest_name or "Entry")
        sub = None if plan.source == "saved" else f"Entry {e.entry_id}" + (
            f" · {e.entry_fee}" if e.entry_fee else "")
        rows = []
        for s in r.slots:
            b, a = s.before or {}, s.after or {}
            rows.append({
                "slot": s.slot, "fixed": s.fixed, "fixed_why": s.fixed_why,
                "changed": s.changed, "before": b.get("name") or fmt.DASH,
                "before_team": b.get("team"), "before_out": bool(b.get("out")),
                "before_proj": b.get("proj"), "after": a.get("name") if s.changed else None,
                "after_team": a.get("team") if s.changed else None,
                "after_proj": a.get("proj") if s.changed else None,
                "delta": (a.get("mean") or 0) - (b.get("mean") or 0)
                if s.changed and a.get("mean") is not None else None,
                "reason": s.reason})
        entries.append({
            "key": e.key, "title": title, "sub": sub, "status": r.status, "status_label": label,
            "status_class": cls, "rows": rows, "n_changed": r.n_changed,
            "n_fixed": sum(s.fixed for s in r.slots),
            "objective": "Cash: favours a safer floor" if r.objective == "cash"
            else "Highest projection",
            "before_mean": _num(r.before_mean), "after_mean": _num(r.after_mean),
            "delta": _signed(r.delta), "delta_raw": r.delta,
            "salary": _money(r.after_salary if r.status == "swapped" else r.before_salary),
            "notes": r.notes + ([f"{p}" for p in e.problems] if r.status == "unusable" else [])})
    out_in_entries = []
    used = {pid for r in plan.results for _, pid in r.entry.pairs if pid}
    for pid, why in plan.ruled_out.items():
        if pid in used:
            out_in_entries.append({"name": names.get(pid, pid), "why": why})
    exposure = []
    for d in plan.portfolio.get("exposure", [])[:10]:
        if d["before"] == d["after"] and len(plan.results) < 2:
            continue
        exposure.append({"name": names.get(d["player_id"], d["player_id"]),
                         "before": f"{d['before']:.0%}", "after": f"{d['after']:.0%}",
                         "fixed": d["fixed"], "moved": d["before"] != d["after"]})
    groups = []
    for g, info in plan.portfolio.get("groups", {}).items():
        groups.append({"label": "Our Top N" if g == "m:topn" else (
            "Our cash lineup" if g == "m:cash" else f"Contest {g[2:]}" if g.startswith("c:")
            else "Entries"), **info})
    fmts = plan.formats
    fmt_note = None
    if plan.source == "upload":
        seen = [k for k in ("name_id", "id", "name") if fmts.get(k)]
        fmt_note = "Read cells as " + ", ".join(
            {"name_id": "Name (ID)", "id": "bare IDs", "name": "names"}[k] for k in seen) + "."
    swapped = sum(r.status == "swapped" for r in plan.results)
    from urllib.parse import quote
    q = f"?t={token}" if token else f"?scope={scope}" + (f"&at={quote(at)}" if at else "")
    return {
        "day": day, "slate_label": slate_label, "base": base, "scope": scope,
        "scope_label": SCOPE_LABEL[scope], "source": plan.source,
        "as_of": dfs_view._et(plan.now), "as_of_iso": SW.iso(plan.now), "games": games,
        "n_started": n_started, "all_open": n_started == 0,
        "all_started": games and n_started == len(games), "proj": proj,
        "fallback": pj.get("source") == "fppg", "entries": entries,
        "n_entries": len(entries), "n_swapped": swapped, "ruled_out": out_in_entries,
        "exposure": exposure if len(entries) > 1 else [], "groups": groups,
        "warnings": plan.warnings, "fmt_note": fmt_note, "seconds": f"{plan.seconds:.1f}",
        "csv": f"{base}/swap.csv{q}", "csv_id": f"{base}/swap.csv{q}&fmt=id",
        "token": token}


def register(app, *, store, render: Callable, not_found: Callable, today=None,
             health_banner: Callable | None = None) -> None:
    """Add the late-swap routes. Tests can set ``app.state.dfs_now``."""
    results: dict[str, tuple[float, dict[str, str], str]] = {}

    def now(at: str | None = None) -> datetime:
        if at:
            try:
                dt = datetime.fromisoformat(at.strip().replace("Z", "+00:00").replace(" ", "+"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        fn = getattr(app.state, "dfs_now", None)
        return fn() if fn else datetime.now(timezone.utc)

    def banner():
        return health_banner() if health_banner else None

    def slate_of(conn, day: str, slate_id: str):
        s = next((s for s in dfs_view.day_slates(conn, day) if s["slate_id"] == slate_id), None)
        if s is None:
            return None, None
        tag = (s.get("tag") or "").lower()
        label = dfs_view.TAG_LABEL.get(tag, "Slate")
        return s, label

    def page(request: Request, day: str, slate_id: str, *, data: bytes | None = None,
             filename: str = "", scope: str = "all", error: str | None = None,
             at: str | None = None):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "DFS URLs look like /dfs/2026-10-20.")
        scope = scope if scope in SW.SCOPES else "all"
        conn = store.health_conn()
        if conn is None:
            return not_found("No database", "The DFS database has not been created yet.")
        vm = {"day": day, "base": f"/dfs/{day}/{slate_id}", "scope": scope, "error": error,
              "scope_label": SCOPE_LABEL[scope], "filename": filename, "at": at or ""}
        try:
            s, label = slate_of(conn, day, slate_id)
            if s is None:
                return not_found("Slate not found", f"There is no slate {slate_id} on {day}.")
            vm["slate_label"] = label
            vm["is_showdown"] = (s.get("tag") or "").lower() == "showdown"
            if not vm["is_showdown"] and error is None:
                try:
                    plan = SW.plan(conn, slate_id, now=now(at), csv_data=data, scope=scope)
                    token = None
                    if data is not None:
                        token = _keep(results, plan, filename)
                    vm.update(build(plan, day=day, slate_label=label, base=vm["base"],
                                    scope=scope, token=token, at=at))
                except SW.SwapError as e:
                    vm["error"] = str(e)
        finally:
            conn.close()
        return render("dfs_swap.html", v=vm, page="dfs", health_banner=banner())

    @app.get("/dfs/{day}/{slate_id}/swap", response_class=HTMLResponse)
    def swap_page(day: str, slate_id: str, request: Request, scope: str = "all",
                  at: str | None = None):
        return page(request, day, slate_id, scope=scope, at=at)

    @app.post("/dfs/{day}/{slate_id}/swap", response_class=HTMLResponse)
    async def swap_upload(day: str, slate_id: str, request: Request):
        if not same_origin(request):
            return PlainTextResponse("Cross-site request refused.", status_code=403)
        body = await read_body(request, MAX_UPLOAD + 256 * 1024)
        if body is None:
            return page(request, day, slate_id, error="That file is too large (the limit is "
                                                      f"{MAX_UPLOAD // (1024 * 1024)} MB).")
        ct = request.headers.get("content-type", "")
        if not ct.lower().startswith("multipart/form-data"):
            return page(request, day, slate_id, error="No file received.")
        fields, files = parse_multipart(ct, body)
        fn, data = files.get("file", ("", b""))
        scope = fields.get("scope", "all")
        at = fields.get("at") or None
        if not data:
            return page(request, day, slate_id, scope=scope, at=at,
                        error="Choose your DraftKings entries CSV first.")
        return page(request, day, slate_id, data=data, filename=fn or "entries.csv", scope=scope,
                    at=at)

    @app.get("/dfs/{day}/{slate_id}/swap.csv")
    def swap_csv(day: str, slate_id: str, request: Request, t: str | None = None,
                 scope: str = "all", at: str | None = None):
        style = request.query_params.get("fmt") or "name_id"
        if style not in dfs_export.STYLES:
            return PlainTextResponse("fmt must be name_id or id.", status_code=400)
        if t:
            hit = results.get(t)
            if hit is None or time.time() - hit[0] > RESULT_TTL_S:
                return PlainTextResponse("That result has expired. Upload the file again.",
                                         status_code=410)
            body, name = hit[1][style], hit[2]
        else:
            conn = store.health_conn()
            if conn is None:
                return PlainTextResponse("No database.", status_code=404)
            try:
                s, _ = slate_of(conn, day, slate_id)
                if s is None:
                    return PlainTextResponse("No such slate.", status_code=404)
                try:
                    plan = SW.plan(conn, slate_id, now=now(at),
                                   scope=scope if scope in SW.SCOPES else "all")
                except SW.SwapError as e:
                    return PlainTextResponse(str(e), status_code=409)
            finally:
                conn.close()
            if not plan.results:
                return PlainTextResponse("No saved lineups for this slate.", status_code=404)
            try:
                body = plan.to_csv(style=style)
            except dfs_export.ExportError as e:
                return PlainTextResponse(f"Cannot export: {e}", status_code=409)
            name = f"courtside-late-swap-{day}-{slate_id}.csv"
        return Response(body, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"',
                                 "Cache-Control": "no-store"})


def _keep(results: dict, plan: SW.SwapPlan, filename: str) -> str:
    """Keep an upload's result CSVs in memory for the download link."""
    now = time.time()
    for k in [k for k, v in results.items() if now - v[0] > RESULT_TTL_S]:
        results.pop(k, None)
    while len(results) >= MAX_RESULTS:
        results.pop(next(iter(results)))
    tok = secrets.token_urlsafe(9)
    stem = (filename.rsplit(".", 1)[0] or "entries")[:40]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    results[tok] = (now, {s: plan.to_csv(style=s) for s in dfs_export.STYLES},
                    f"{stem}-late-swap.csv")
    return tok
