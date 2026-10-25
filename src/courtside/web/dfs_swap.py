"""The slate page's "Late swap" action: re-solve the unlocked slots, show what changes and why,
and hand back a DK edit CSV (docs/PLAYERS.md §2.5, task T2.14; the solver is proj/dfs/swap.py).

Routes
  GET  /dfs/<day>/<slate>/swap                our saved lineups, as of now (?scope=all|news,
                                              ?at=<ISO time> previews another moment,
                                              ?job=<id> shows one background job)
  POST /dfs/<day>/<slate>/swap                multipart: file = DK entries CSV (+ scope)
  GET  /dfs/<day>/<slate>/swap.csv            ?fmt=name_id|id &scope= &j=<job>  (saved lineups)
                                              ?t=<token>               (an uploaded file's result)

Background: `SW.plan` can take most of a minute (the contest-aware swap, D8), so it runs in a
background job (`SwapJobs`, at most `WORKERS` at once) on its own read-only connection, keyed
by (slate, scope, at, sha256 of the upload or "saved", projection run). A request waits up to
`WAIT_S` for it, so a fast plan renders exactly as before; otherwise the page shows a "working"
state that refreshes itself. A finished job is reused for the same key (`LIVE_REUSE_S` when
the plan is "as of now", else `RESULT_TTL_S`) and the CSV downloads use its plan, never a
recompute. A job is given up after `JOB_TIMEOUT_S`.

Read-only: the plan is computed on a read-only connection and an uploaded file is never
stored on disk or in the database (its bytes live in memory only until its job finishes; the
plan is kept in memory for the download, the last `MAX_RESULTS` jobs only). Nothing here
talks to DraftKings.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from ..proj.dfs import export as dfs_export
from ..proj.dfs import swap as SW
from . import dfs_view, fmt
from .dfs_results import parse_multipart, read_body, same_origin

log = logging.getLogger(__name__)

MAX_UPLOAD = 5 * 1024 * 1024
MAX_RESULTS = 24
RESULT_TTL_S = 6 * 3600
LIVE_REUSE_S = 60                # an "as of now" plan is reused this long
WORKERS = 2                      # background plans at once
WAIT_S = 2.0                     # a request waits this long before showing "working"
JOB_TIMEOUT_S = 120.0
REFRESH_S = 3
TIMEOUT_MSG = ("Working out the swap took more than two minutes, so it was given up. "
               "Try again in a moment.")
EXPIRED_MSG = ("That late swap has expired (or the server restarted). Open the late swap "
               "again, or upload your entries file again.")
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
          token: str | None = None, at: str | None = None, job: str | None = None) -> dict:
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
                "before_id": b.get("player_id"), "after_id": a.get("player_id") if s.changed else None,
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
            out_in_entries.append({"name": names.get(pid, pid), "why": why, "id": pid})
    exposure = []
    for d in plan.portfolio.get("exposure", [])[:10]:
        if d["before"] == d["after"] and len(plan.results) < 2:
            continue
        exposure.append({"name": names.get(d["player_id"], d["player_id"]), "id": d["player_id"],
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
    q = f"?t={token}" if token else f"?scope={scope}" + (f"&at={quote(at)}" if at else "") + (
        f"&j={job}" if job else "")
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




# ===================================================================== background jobs

@dataclass
class Job:
    """One background `SW.plan` (memory only)."""
    id: str
    key: tuple
    slate_id: str
    source: str                      # saved | upload
    scope: str
    at: str | None
    filename: str
    created: float
    fn: Callable | None = None       # the plan call (holds the upload's bytes until it ends)
    done: threading.Event = field(default_factory=threading.Event)
    plan: SW.SwapPlan | None = None
    error: str | None = None
    timed_out: bool = False

    def state(self) -> str:
        """'ok', 'error' or 'pending' (a job past `JOB_TIMEOUT_S` turns into an error)."""
        if self.timed_out:
            return "error"
        if self.done.is_set():
            return "error" if self.error is not None or self.plan is None else "ok"
        if time.time() - self.created > JOB_TIMEOUT_S:
            self.timed_out, self.error = True, TIMEOUT_MSG
            return "error"
        return "pending"

    def wait(self, seconds: float) -> str:
        self.done.wait(max(0.0, min(seconds, JOB_TIMEOUT_S - (time.time() - self.created))))
        return self.state()

    def download_name(self) -> str:
        stem = (self.filename.rsplit(".", 1)[0] or "entries")[:40]
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        return f"{stem}-late-swap.csv"


class SwapJobs:
    """Background swap plans: at most `WORKERS` run at once, the last `MAX_RESULTS` are kept."""

    def __init__(self, connect: Callable[[], sqlite3.Connection | None], workers: int = WORKERS):
        self.connect = connect
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.slots = threading.BoundedSemaphore(workers)

    def get(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        with self.lock:
            j = self.jobs.get(job_id)
            if j is not None and time.time() - j.created > RESULT_TTL_S:
                self.jobs.pop(job_id, None)
                return None
            return j

    def _reusable(self, j: Job, now: float) -> bool:
        st = j.state()
        if st == "pending":
            return True
        ttl = RESULT_TTL_S if j.at else LIVE_REUSE_S
        return st == "ok" and now - j.created <= ttl

    def submit(self, key: tuple, fn: Callable[[sqlite3.Connection], SW.SwapPlan], *,
               slate_id: str, source: str, scope: str, at: str | None,
               filename: str = "") -> Job:
        """The running or reusable job for `key`, else a new one started on `fn`."""
        now = time.time()
        with self.lock:
            for j in reversed(list(self.jobs.values())):
                if j.key == key and self._reusable(j, now):
                    return j
            self._prune(now)
            j = Job(id=secrets.token_urlsafe(9), key=key, slate_id=slate_id, source=source,
                    scope=scope, at=at, filename=filename, created=now, fn=fn)
            self.jobs[j.id] = j
        threading.Thread(target=self._run, args=(j,), name=f"swap-{j.id}", daemon=True).start()
        return j

    def _prune(self, now: float) -> None:
        for k in [k for k, j in self.jobs.items()
                  if now - j.created > RESULT_TTL_S and j.state() != "pending"]:
            self.jobs.pop(k, None)
        while len(self.jobs) >= MAX_RESULTS:
            done = [j for j in self.jobs.values() if j.state() != "pending"] or list(
                self.jobs.values())
            # saved-lineup plans go first: an upload's result is the one a user cannot redo
            victim = min(done, key=lambda j: (j.source == "upload", j.created))
            self.jobs.pop(victim.id, None)

    def _run(self, j: Job) -> None:
        with self.slots:
            conn = None
            plan, error = None, None
            try:
                if j.state() != "pending":             # timed out while queued
                    return
                conn = self.connect()
                if conn is None:
                    raise SW.SwapError("The DFS database has not been created yet.")
                plan = j.fn(conn)
            except SW.SwapError as e:
                error = str(e)
            except Exception as e:  # noqa: BLE001 - the error is shown on the page
                log.exception("late swap job %s failed", j.id)
                error = f"The late swap failed ({type(e).__name__}: {e}). Try again."
            finally:
                if conn is not None:
                    conn.close()
                j.fn = None                             # drop the upload's bytes
                if not j.timed_out:
                    j.plan, j.error = plan, error
                j.done.set()


def run_id_of(conn: sqlite3.Connection, slate_id: str) -> str | None:
    """The slate's newest projection run (the one `SW.load_projections` uses first), for the
    job key; None when there is none or the query fails."""
    try:
        r = conn.execute(
            "SELECT r.run_id FROM dfs_runs r WHERE r.slate_id=? AND r.ok=1 AND EXISTS (SELECT "
            "1 FROM player_projections p WHERE p.run_id=r.run_id) ORDER BY r.created_at DESC, "
            "r.rowid DESC LIMIT 1", (str(slate_id),)).fetchone()
    except sqlite3.Error:
        return None
    return str(r[0]) if r and r[0] is not None else None


# ===================================================================== routes

def register(app, *, store, render: Callable, not_found: Callable, today=None,
             health_banner: Callable | None = None) -> None:
    """Add the late-swap routes. Tests can set ``app.state.dfs_now``; ``app.state.swap_jobs``
    is the job table."""
    jobs = SwapJobs(store.health_conn)
    app.state.swap_jobs = jobs

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

    def start(conn, slate_id: str, *, scope: str, at: str | None, data: bytes | None = None,
              filename: str = "") -> Job:
        """Start (or reuse) the plan of the saved lineups (`data` None) or of an upload."""
        when = now(at)
        digest = "saved" if data is None else hashlib.sha256(data).hexdigest()
        key = (slate_id, scope, at or "", digest, run_id_of(conn, slate_id))
        return jobs.submit(
            key, lambda c: SW.plan(c, slate_id, now=when, csv_data=data, scope=scope),
            slate_id=slate_id, source="saved" if data is None else "upload", scope=scope,
            at=at, filename=filename)

    def page(request: Request, day: str, slate_id: str, *, data: bytes | None = None,
             filename: str = "", scope: str = "all", error: str | None = None,
             at: str | None = None, job_id: str | None = None):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "DFS URLs look like /dfs/2026-10-20.")
        scope = scope if scope in SW.SCOPES else "all"
        conn = store.health_conn()
        if conn is None:
            return not_found("No database", "The DFS database has not been created yet.")
        job = None
        try:
            s, label = slate_of(conn, day, slate_id)
            if s is None:
                return not_found("Slate not found", f"There is no slate {slate_id} on {day}.")
            nba = dfs_view.dk_to_nba(conn)      # DK id -> NBA id, for /player/<id> links only
            showdown = (s.get("tag") or "").lower() == "showdown"
            if not showdown and error is None:
                if job_id:
                    job = jobs.get(job_id)
                    if job is None or job.slate_id != slate_id:
                        job, error = None, EXPIRED_MSG
                    else:
                        scope, at, filename = job.scope, job.at, job.filename
                else:
                    job = start(conn, slate_id, scope=scope, at=at, data=data, filename=filename)
        finally:
            conn.close()
        base = f"/dfs/{day}/{slate_id}"
        vm = {"day": day, "base": base, "scope": scope, "error": error,
              "scope_label": SCOPE_LABEL[scope], "filename": filename, "at": at or "",
              "slate_label": label, "is_showdown": showdown, "nba": nba}
        if job is not None:
            st = job.wait(WAIT_S)
            if st == "ok":
                vm.update(build(job.plan, day=day, slate_label=label, base=base, scope=scope,
                                token=job.id if job.source == "upload" else None, at=at,
                                job=job.id if job.source == "saved" else None))
            elif st == "error":
                vm["error"] = job.error
            else:
                vm["pending"] = True
                vm["refresh"] = REFRESH_S
                vm["elapsed"] = int(time.time() - job.created)
                vm["refresh_url"] = (f"{base}/swap?scope={scope}" + (
                    f"&at={quote(at)}" if at else "") + f"&job={job.id}")
        return render("dfs_swap.html", v=vm, page="dfs", health_banner=banner())

    @app.get("/dfs/{day}/{slate_id}/swap", response_class=HTMLResponse)
    def swap_page(day: str, slate_id: str, request: Request, scope: str = "all",
                  at: str | None = None, job: str | None = None):
        return page(request, day, slate_id, scope=scope, at=at, job_id=job)

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
                 j: str | None = None, scope: str = "all", at: str | None = None):
        style = request.query_params.get("fmt") or "name_id"
        if style not in dfs_export.STYLES:
            return PlainTextResponse("fmt must be name_id or id.", status_code=400)
        if t:
            job = jobs.get(t)
            if job is None or job.source != "upload" or job.state() != "ok":
                return PlainTextResponse("That result has expired. Upload the file again.",
                                         status_code=410)
            name = job.download_name()
        else:
            job = jobs.get(j)
            if job is None or job.source != "saved" or job.slate_id != slate_id:
                conn = store.health_conn()
                if conn is None:
                    return PlainTextResponse("No database.", status_code=404)
                try:
                    s, _ = slate_of(conn, day, slate_id)
                    if s is None:
                        return PlainTextResponse("No such slate.", status_code=404)
                    job = start(conn, slate_id, scope=scope if scope in SW.SCOPES else "all",
                                at=at)
                finally:
                    conn.close()
            st = job.wait(WAIT_S)
            if st == "pending":
                return PlainTextResponse("The late swap is still being worked out. Try the "
                                         "download again in a minute.", status_code=202)
            if st == "error":
                return PlainTextResponse(job.error or "The late swap failed.", status_code=409)
            if not job.plan.results:
                return PlainTextResponse("No saved lineups for this slate.", status_code=404)
            name = f"courtside-late-swap-{day}-{slate_id}.csv"
        try:
            body = job.plan.to_csv(style=style)
        except dfs_export.ExportError as e:
            return PlainTextResponse(f"Cannot export: {e}", status_code=409)
        return Response(body, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"',
                                 "Cache-Control": "no-store"})
