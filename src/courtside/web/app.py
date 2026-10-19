"""Courtside web front end: server-rendered Brief pages (FastAPI + Jinja2).

Routes: `/` (latest Brief), `/brief/<day>`, `/archive`, `/game/<game_id>`, `/health` (model health),
`/healthz`, `/api/health`, `/dfs` (DFS slates, see dfs_view.py),
`/api/brief/<day|latest>`. Run with `courtside serve`; see docs/WEB.md.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import config
from . import dfs_view, fmt, health, view
from .store import BriefStore
from .teams import identity

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"


def _asset_version() -> str:
    h = hashlib.sha1()
    for p in sorted(STATIC.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.name}{st.st_mtime_ns}{st.st_size}".encode())
    return h.hexdigest()[:10]


def make_env(dev: bool = False) -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      autoescape=select_autoescape(["html", "xml"]),
                      auto_reload=dev, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(fmt.FILTERS)
    env.globals.update(team=identity, DASH=fmt.DASH, MINUS=fmt.MINUS)
    return env


class _CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


def create_app(data_dir: str | Path | None = None, *, dev: bool | None = None) -> FastAPI:
    dev = bool(os.environ.get("COURTSIDE_WEB_DEV")) if dev is None else dev
    store = BriefStore(data_dir)
    env = make_env(dev)
    app = FastAPI(title="Courtside", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.mount("/static", _CachedStatic(directory=str(STATIC)), name="static")
    asset_v = {"v": _asset_version()}

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    def render(name: str, status: int = 200, **ctx) -> HTMLResponse:
        if dev:
            asset_v["v"] = _asset_version()
        ctx.setdefault("asset_v", asset_v["v"])
        ctx.setdefault("latest_day", store.latest_day())
        ctx.setdefault("health_banner", None)
        html = env.get_template(name).render(**ctx)
        return HTMLResponse(html, status_code=status)

    def not_found(msg: str, detail: str = "") -> HTMLResponse:
        return render("404.html", status=404, message=msg, detail=detail, page="error")

    def _today():
        try:
            return config.today_nba_day()
        except Exception:  # noqa: BLE001 - never fail a page over the clock
            return datetime.now(timezone.utc).date()

    def health_banner():
        """Site-wide notice on the Brief pages: only open *critical* alerts trigger it."""
        conn = store.health_conn()
        if conn is None:
            return None
        try:
            return health.banner(conn)
        finally:
            conn.close()

    def health_view():
        conn = store.health_conn()
        try:
            return health.build(conn, _today())
        finally:
            if conn is not None:
                conn.close()

    def brief_page(day: str, *, is_latest: bool) -> HTMLResponse:
        brief = store.get(day)
        if brief is None:
            return not_found("No Brief for that day",
                             f"There is no stored Brief for {fmt.long_date(day) or day}.")
        vm = store.derived(("brief-view",) + store.version(day), lambda: view.brief_view(brief))
        prev, nxt = store.neighbours(day)
        return render("brief.html", v=vm, day=day, prev_day=prev, next_day=nxt,
                      is_latest=is_latest, page="brief", health_banner=health_banner())

    @app.get("/", response_class=HTMLResponse)
    def index():
        day = store.latest_day()
        if day is None:
            return render("welcome.html", page="welcome")
        return brief_page(day, is_latest=True)

    @app.get("/brief/{day}", response_class=HTMLResponse)
    def brief_for_day(day: str):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "Brief URLs look like /brief/2026-02-20.")
        return brief_page(day, is_latest=day == store.latest_day())

    @app.get("/archive", response_class=HTMLResponse)
    def archive():
        days = list(reversed(store.days()))

        def build():
            rows = []
            for d in days:
                b = store.get(d) or {}
                vm = store.derived(("brief-view",) + store.version(d),
                                   lambda b=b: view.brief_view(b))
                rows.append({"day": d, "headline": vm["masthead"]["headline"],
                             "games": len(vm["last_night"]["cards"]),
                             "tonight": len(vm["tonight"]["games"]),
                             "picks": vm["last_night"]["picks"]})
            return rows
        key = ("archive",) + tuple(store.version(d)[1] or "" for d in days) + tuple(days)
        rows = store.derived(key, build)
        months: list[dict] = []
        for r in rows:
            d = fmt.parse_day(r["day"])
            label = f"{d:%B} {d.year}"
            if not months or months[-1]["label"] != label:
                months.append({"label": label, "rows": []})
            months[-1]["rows"].append(r)
        return render("archive.html", months=months, count=len(rows), page="archive")

    @app.get("/game/{game_id}", response_class=HTMLResponse)
    def game(game_id: str):
        hit = store.find_game(game_id)
        if hit is None:
            return not_found("Game not found",
                             f"No stored Brief mentions game {game_id}.")
        day, where, g = hit
        key = ("game", game_id, where) + store.version(day)
        if where == "last_night":
            vm = store.derived(key, lambda: view.game_view(g))
            return render("game.html", g=vm, day=day, page="game")
        vm = store.derived(key, lambda: view.tonight_view(g))
        return render("preview.html", t=vm, day=day, page="game")

    dfs_view.register(app, store=store, render=render, not_found=not_found, today=_today,
                      health_banner=health_banner)

    @app.get("/health", response_class=HTMLResponse)
    def model_health():
        return render("health.html", h=health_view(), page="health")

    @app.get("/api/health")
    def api_health():
        return JSONResponse(health.api_summary(health_view()))

    @app.get("/healthz")
    def healthz():
        days = store.days()
        latest = days[-1] if days else None
        built = ((store.get(latest) or {}).get("meta") or {}).get("built_at") if latest else None
        age_h = None
        if built:
            try:
                bt = datetime.fromisoformat(built.replace("Z", "+00:00"))
                age_h = round((datetime.now(timezone.utc) - bt).total_seconds() / 3600, 2)
            except ValueError:
                pass
        return {"ok": True, "status": "ok" if latest else "no_briefs", "briefs": len(days),
                "latest_day": latest, "latest_built_at": built, "latest_age_hours": age_h,
                "data_dir": str(store.data_dir)}

    @app.get("/api/brief/{day}")
    def api_brief(day: str):
        if day == "latest":
            day = store.latest_day() or ""
        brief = store.get(day) if day else None
        if brief is None:
            return JSONResponse({"error": "not_found", "day": day}, status_code=404)
        return JSONResponse(brief)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            if request.url.path.startswith(("/api/", "/static/")):
                return JSONResponse({"error": "not_found"}, status_code=404)
            return not_found("Page not found", "That page doesn’t exist.")
        return Response(str(exc.detail), status_code=exc.status_code)

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory courtside.web.app:app_factory` (used by --reload)."""
    return create_app()
