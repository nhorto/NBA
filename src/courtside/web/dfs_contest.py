"""The contest builder page (docs/PLAYERS.md §2.3; tasks T2.12 / T2.16).

Routes
  GET  /dfs/<day>/<slate_id>/contest            the builder (form + this slate's builds);
                                                ?p=<portfolio id> shows one build (or "building")
  POST /dfs/<day>/<slate_id>/contest            start a build -> 303 to ?p=<id>
  GET  /dfs/<day>/<slate_id>/contest/<id>.csv   DK upload CSV of a build (?fmt=id|name_id)

A build runs `proj/dfs/portfolio.build_for_slate` in a background thread (one at a time per
server) with its own write connection; the page polls with a meta refresh while the
`dfs_portfolios` row says "building". Everything else reads the app's read-only connection.

Every number on the page is simulated against a modelled field (ownership and field are
proxies). There is no dollar or Kelly sizing, only the light guidance of §2.3. Courtside never
signs in to DraftKings or enters contests: the CSV is uploaded by hand.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping

from fastapi import Query, Request
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse,
                               Response)

from .. import db
from ..proj.dfs import export as dfs_export
from ..proj.dfs import field as dfs_field
from ..proj.dfs import portfolio as PF
from . import dfs_view as V
from . import fmt
from .dfs_results import parse_form, read_body, same_origin

N_DEFAULT = 20
EXP_DEFAULT = 60                 # percent
GAMMA_DEFAULT = 5
GAMMA_CHOICES = (3, 4, 5, 6, 7)
POOL_ROWS = 90                   # players offered for lock / exclude (by projection)
REFRESH_S = 4
MAX_FLASH = 24
SIM_LABEL = "Simulated, not real results"

GUIDANCE = (
    "Cash contests (double-ups, 50/50s) suit projection-driven builds: the lineup only has to "
    "beat about half the field.",
    "Single-entry and 3-max tournaments keep out the 150-entry professionals; big multi-entry "
    "tournaments are where they play.",
    "Large fields carry the most variance: most entries win nothing, and the expected payout "
    "rests on rare top finishes.",
)
# Showdown (IMPROVEMENTS.md D10): one game, so the entries move together; the builder's joint
# simulations carry that, Top N by projection does not (docs/research/b6-d10.md).
GUIDANCE_SHOWDOWN = (
    "Showdown is a single game: every entry rides on the same game script and the same few "
    "stars. The top lineups by projection often differ by one player and win or lose together; "
    "a set picked here on our joint simulations of the game spreads across captains and game "
    "scripts. On past Showdown slates such sets had a higher simulated payout than the top "
    "lineups by projection, but cashed somewhat less often: the payout sits in the top places.",
)
STAKES = ("Sensible stakes: even a strong player loses money in about half of all 100-contest "
          "stretches. Play only with money you are happy to spend on entertainment.")


# ===================================================================== builds (background)

class Builds:
    """One build at a time per server; progress text per portfolio id (memory only)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running: str | None = None
        self.stage: dict[str, tuple[str, float]] = {}

    def busy(self) -> bool:
        return self.running is not None

    def start(self, db_path, portfolio_id: str, slate_id: str, contest_id: str,
              s: PF.Settings, now: datetime | None) -> bool:
        with self.lock:
            if self.running is not None:
                return False
            self.running = portfolio_id
        self.stage[portfolio_id] = ("starting", time.time())
        t = threading.Thread(target=self._run, name=f"portfolio-{portfolio_id}", daemon=True,
                             args=(db_path, portfolio_id, slate_id, contest_id, s, now))
        t.start()
        return True

    def _run(self, db_path, portfolio_id, slate_id, contest_id, s, now) -> None:
        t0 = time.time()
        conn = None
        try:
            conn = db.connect(db_path)
            pf = PF.build_for_slate(conn, slate_id, contest_id, s, now=now,
                                    progress=lambda m: self.stage.__setitem__(
                                        portfolio_id, (m, t0)))
            PF.save(conn, pf, portfolio_id)
        except Exception as e:  # noqa: BLE001 - the error is shown on the page
            msg = str(e) if isinstance(e, PF.PortfolioError) else f"{type(e).__name__}: {e}"
            try:
                if conn is None:
                    conn = db.connect(db_path)
                PF.fail_row(conn, portfolio_id, msg, round(time.time() - t0, 1))
            except Exception:  # noqa: BLE001
                pass
        finally:
            if conn is not None:
                conn.close()
            self.stage.pop(portfolio_id, None)
            with self.lock:
                self.running = None


# ===================================================================== view model

def _money(x: Any, cents: bool | None = None) -> str:
    v = V._num(x)
    if v is None:
        return fmt.EM
    if cents is None:
        cents = abs(v) < 100
    s = f"${abs(v):,.2f}" if cents else f"${abs(v):,.0f}"
    return (fmt.MINUS + s) if v < 0 else s


def _roi(x: Any) -> str:
    v = V._num(x)
    if v is None:
        return fmt.EM
    return fmt.signed(100 * v, 0) + "%"


def contest_label(c: Mapping) -> str:
    fee = _money(c["entry_fee"], cents=c["entry_fee"] < 10 and c["entry_fee"] != int(c["entry_fee"]))
    return (f"{c['name']} · {fee} · {c['field_size']:,} entries · max {c['max_per_user']} · "
            f"top prize {_money(c['top_prize'], cents=False)}")


def _contest_view(c: Mapping) -> dict:
    kind = {"cash": "Cash (double-up)", "small_gpp": "Small tournament",
            "large_gpp": "Large tournament"}.get(c.get("archetype"), "Tournament")
    return {**c, "label": contest_label(c), "kind": kind,
            "fee_text": _money(c["entry_fee"]), "top_text": _money(c["top_prize"], cents=False),
            "field_text": f"{c['field_size']:,}", "max_n": max(1, min(int(c["max_per_user"]),
                                                                    PF.MAX_ENTRIES)),
            "paid_text": (f"{c['paid']:,} paid ({c['paid'] / c['field_size']:.0%})"
                          if c.get("paid") else "")}


def _players(conn, slate: dict, run: dict | None) -> list[dict]:
    pool, _ = V.salary_pool(conn, slate["slate_id"])
    proj = V.load_projections(conn, run, [str(g) for g in slate["game_ids"]])
    if not proj["by_nba_id"]:
        return []
    rows = V._pool_view(pool, proj, V.dk_to_nba(conn))
    rows = [r for r in rows if r.get("proj") is not None and r["proj"] >= PF.MIN_PROJ
            and r["status"] not in ("OUT", "O")]
    rows.sort(key=lambda r: (-(r["proj"] or 0), r["name"]))
    return [{"id": r["id"], "name": r["name"], "team": r["team"],
             "positions": "/".join(r["positions"]), "salary_text": r["salary_text"],
             "proj": r["proj"], "status": r["status"]} for r in rows[:POOL_ROWS]]


def _relative_text(r: float | None) -> str:
    if r is None:
        return fmt.EM
    pct = (r - 1) * 100
    return "about the same" if abs(pct) < 0.5 else fmt.signed(pct, 0) + "%"


def curve_view(cv: Mapping | None, calibrated: bool, cal_slates: int = 0) -> dict | None:
    """The "chances by number of entries" table: one row per entry count, chances as ranges
    (rare ones as "~1 in X") until payouts are calibrated, and the fees. No expected payout."""
    if not cv or not cv.get("points"):
        return None
    rows = []
    for r in cv["points"]:
        n_sims = r.get("n_sims")
        ch = lambda k: PF.chance_text(r.get(k), calibrated, n_sims)  # noqa: E731
        se = r.get("se") or PF.curve_se(r)          # D9; older builds: binomial from p and n
        sx = lambda k: PF.se_text(r.get(k), se.get(k))  # noqa: E731
        rows.append({"entries": r["entries"], "requested": r.get("requested", r["entries"]),
                     "short": r["entries"] < r.get("requested", r["entries"]),
                     "this_set": r.get("source") == "build",
                     "extended": r.get("source") == "extended",
                     "cost": _money(r.get("cost"), cents=float(r.get("cost") or 0) % 1 != 0),
                     "cash": ch("p_any_cash"), "top1": ch("p_any_top1"),
                     "top01": ch("p_any_top01"), "first": ch("p_first"),
                     "cash_se": sx("p_any_cash"), "top1_se": sx("p_any_top1"),
                     "top01_se": sx("p_any_top01"), "first_se": sx("p_first")})
    return {"rows": rows, "calibrated": calibrated, "cal_slates": cal_slates,
            "cal_needed": PF.MIN_CAL_SLATES, "n_build": cv.get("n_build"),
            "has_extended": any(r["extended"] for r in rows),
            "has_subset": any(not r["this_set"] and not r["extended"] for r in rows),
            "short": next((r for r in rows if r["short"]), None),
            "n_sims": next((r.get("n_sims") for r in cv["points"] if r.get("n_sims")), None),
            "incomplete": not cv.get("complete", True)}


def _portfolio_view(d: dict, contests: dict[str, dict], now: datetime, builds: Builds,
                    calibrated: bool = False, cal_slates: int = 0) -> dict:
    data = d.get("data") or {}
    sm = data.get("summary") or {}
    info = data.get("info") or {}
    c = data.get("contest") or contests.get(str(d.get("contest_id"))) or {}
    created = V._dt(d.get("created_at"))
    status = d["status"]
    stage = builds.stage.get(d["portfolio_id"])
    if status == "building" and stage is None and created and \
            (now - created).total_seconds() > PF.STALE_BUILD_S:
        status = "stale"
    out = {"id": d["portfolio_id"], "status": status, "error": d.get("error"),
           "created_text": f"{fmt.stamp(d.get('created_at'))}" if d.get("created_at") else "",
           "n_requested": d.get("n_requested"), "n_built": d.get("n_built"),
           "seconds": d.get("seconds"), "contest": c,
           "contest_name": c.get("name") or f"Contest {d.get('contest_id')}",
           "settings": d.get("settings") or {}}
    if status == "building":
        st, t0 = stage or ("queued", time.time())
        names = {"starting": "Starting", "field": "Building the field proxy",
                 "candidates": "Solving candidate lineups", "scoring": "Scoring against the field",
                 "selection": "Choosing the portfolio", "strengths": "Checking soft and sharp fields",
                 "curve": "Working out chances by number of entries"}
        out["stage"] = names.get(st, st[:1].upper() + st[1:])
        out["elapsed"] = int(time.time() - t0)
        return out
    if status != "ok":
        return out
    fee = float(c.get("entry_fee") or 0)
    st = sm.get("strengths") or {}
    base = sm.get("baseline") or {}
    out["calibrated"] = calibrated
    out["headline"] = {
        "relative": _relative_text(sm.get("relative")),
        "relative_se": (fmt.num(100 * sm["relative_se"], 0) + "%"
                        if sm.get("relative_se") is not None else None),
        "baseline_n": sm.get("requested"),
        "any_cash": PF.prob_band(sm.get("p_any_cash")),
        "any_top1": PF.prob_band(sm.get("p_any_top1")),
        "base_any_cash": PF.prob_band(base.get("p_any_cash")) if base else None}
    out["summary"] = {
        "entries": sm.get("entries"), "requested": sm.get("requested"),
        "stopped": sm.get("stopped"),
        "e_payout": _money(sm.get("e_payout")), "e_payout_se": _money(sm.get("e_payout_se")),
        "cost": _money(sm.get("cost")), "roi": _roi(sm.get("roi")),
        "p_any_cash": fmt.pct(sm.get("p_any_cash")), "p_profit": fmt.pct(sm.get("p_profit")),
        "p_any_top1": fmt.pct(sm.get("p_any_top1"), 1),
        "e_cashing": fmt.num(sm.get("e_cashing"), 1), "n_sims": f"{sm.get('n_sims', 0):,}",
        "range": ({"soft": _roi(st["soft"]["roi"]), "medium": _roi(sm.get("roi")),
                   "sharp": _roi(st["sharp"]["roi"]),
                   "soft_e": _money(st["soft"]["e_payout"]), "sharp_e": _money(st["sharp"]["e_payout"]),
                   "ordered": st["soft"]["e_payout"] >= sm.get("e_payout", 0) >= st["sharp"]["e_payout"]}
                  if st.get("soft") and st.get("sharp") else None),
        "base_e": _money(base.get("e_payout")) if base else None}
    out["curve"] = curve_view(sm.get("curve"), calibrated, cal_slates)
    out["exposure"] = [{
        "name": r["name"], "team": r.get("team"), "positions": r.get("positions"),
        "count": r["count"], "share": fmt.pct(r["share"]), "share_v": r["share"],
        "cap": r.get("cap"), "band": r.get("own_band"), "locked": r.get("locked"),
        "salary_text": V.money(r.get("salary"))} for r in data.get("exposure") or []]
    lus = []
    for lu in d.get("lineups") or []:
        why = lu.get("why") or {}
        lus.append({
            "rank": lu["rank"], "slots": [{**s, "salary_text": V.money(s.get("salary"))}
                                          for s in lu["slots"]],
            "salary_text": V.money(lu.get("salary")), "mean": lu.get("mean"),
            "p10": lu.get("p10"), "p90": lu.get("p90"),
            "p_cash": fmt.pct(lu.get("p_cash")), "p_top1": fmt.pct(lu.get("p_top1"), 1),
            "cash_band": PF.prob_band(lu.get("p_cash")), "top1_band": PF.prob_band(lu.get("p_top1")),
            "e_payout": _money(lu.get("e_payout")), "dupes": lu.get("dupes"),
            "dupe_band": why.get("dupe_band") or dfs_field.dupe_band(lu.get("dupes") or 0),
            "names": ", ".join(fmt.last_name(s.get("name") or "") for s in lu["slots"])})
    out["lineups"] = lus
    fld = info.get("field") or {}
    out["contest_line"] = " · ".join(x for x in (
        f"Entry fee {_money(fee)}",
        f"{int(c['field_size']):,} entries" if c.get("field_size") else "",
        f"{int(c['paid']):,} paid" if c.get("paid") else "") if x)
    out["basis"] = {
        "candidates": info.get("candidates_total"), "n_sims": f"{int(info.get('n_sims') or 0):,}",
        "field_sample": f"{int(fld.get('sampled') or 0):,}", "ownership_calibrated": info.get("ownership_calibrated"),
        "archetype": c.get("archetype"), "seconds": info.get("seconds"),
        "gamma": info.get("gamma"), "fee": fee}
    return out


def build_page(conn: sqlite3.Connection | None, day: str, slate_id: str, *, now: datetime,
               portfolio_id: str | None, builds: Builds, flash: dict | None = None,
               form: Mapping[str, str] | None = None) -> dict:
    v: dict[str, Any] = {"day": day, "day_label": fmt.long_date(day) or day, "state": "ok",
                         "flash": flash, "sim_label": SIM_LABEL, "guidance": GUIDANCE,
                         "stakes": STAKES, "busy": builds.busy(), "refresh": None,
                         "portfolio": None, "recent": []}
    slates = V.day_slates(conn, day) if conn is not None else []
    slate = next((s for s in slates if s["slate_id"] == slate_id), None)
    if slate is None:
        v["state"] = "no_such_slate"
        return v
    pool, _ = V.salary_pool(conn, slate_id)
    games = V.slate_games(pool)
    tag = (slate.get("tag") or "").lower()
    label = V.TAG_LABEL.get(tag, "Slate")
    if tag == "showdown" and len(games) == 1:
        label = f"Showdown {games[0]['label']}"
    lock = slate["lock"]
    run = dfs_field.latest_run(conn, slate_id)
    is_sd = tag == "showdown"
    if is_sd:
        v["guidance"] = GUIDANCE_SHOWDOWN + GUIDANCE
    v["slate"] = {"id": slate_id, "label": label, "url": f"/dfs/{day}/{slate_id}",
                  "is_showdown": is_sd, "n_slots": 6 if is_sd else 8,
                  "games": len(games), "lock_text": V.lock_text(lock, now),
                  "locked": bool(lock and lock <= now), "has_sims": run is not None,
                  "run_text": (f"Projection run {fmt.stamp(run['created_at'])}" if run else "")}
    contests = [_contest_view(c) for c in PF.contest_options(conn, slate_id)]
    v["contests"] = contests
    by_cid = {c["contest_id"]: c for c in contests}
    form = dict(form or {})
    sel = form.get("contest") or ""
    if sel not in by_cid:
        gpp = [c for c in contests if c["archetype"] != "cash" and c["max_per_user"] >= N_DEFAULT]
        sel = (max(gpp, key=lambda c: c["field_size"])["contest_id"] if gpp
               else (contests[0]["contest_id"] if contests else ""))
    v["form"] = {"contest": sel, "n": form.get("n") or str(min(N_DEFAULT, by_cid[sel]["max_n"])
                                                            if sel else N_DEFAULT),
                 "max_exp": form.get("max_exp") or str(EXP_DEFAULT),
                 "gamma": form.get("gamma") or str(GAMMA_DEFAULT),
                 "acts": {k[2:]: val for k, val in form.items() if k.startswith("p_")}}
    v["gamma_choices"] = tuple(g for g in GAMMA_CHOICES if g < 6) if is_sd else GAMMA_CHOICES
    v["players"] = _players(conn, slate, run) if run else []
    v["can_build"] = bool(contests) and run is not None and not v["slate"]["locked"]
    v["why_not"] = ("" if v["can_build"] else
                    "This slate has locked: DraftKings no longer takes new entries."
                    if v["slate"]["locked"] else
                    "No projection run with simulations for this slate yet; the builder needs one."
                    if run is None else "No contests captured for this slate.")
    v["action"] = f"/dfs/{day}/{slate_id}/contest"
    for r in PF.recent(conn, slate_id, 8):
        c = by_cid.get(str(r["contest_id"])) or {}
        st = r["status"]
        created = V._dt(r["created_at"])
        if st == "building" and r["portfolio_id"] not in builds.stage and created and \
                (now - created).total_seconds() > PF.STALE_BUILD_S:
            st = "stale"
        v["recent"].append({"id": r["portfolio_id"], "status": st,
                            "url": f"{v['action']}?p={r['portfolio_id']}",
                            "contest": c.get("name") or f"Contest {r['contest_id']}",
                            "n": r["n_built"] if r["n_built"] is not None else r["n_requested"],
                            "when": fmt.stamp(r["created_at"]),
                            "active": r["portfolio_id"] == portfolio_id})
    if portfolio_id:
        d = PF.load(conn, portfolio_id)
        if d is None or str(d.get("slate_id")) != slate_id:
            v["state"] = "no_such_portfolio"
            return v
        cal_n = PF.calibration_slates(conn)
        pv = _portfolio_view(d, by_cid, now, builds,
                             calibrated=PF.payout_calibrated(conn), cal_slates=cal_n)
        pv["csv"] = f"{v['action']}/{portfolio_id}.csv"
        pv["json"] = f"{v['action']}/{portfolio_id}.json"
        v["portfolio"] = pv
        if pv["status"] == "building":
            v["refresh"] = REFRESH_S
    return v


# ===================================================================== form parsing

def parse_build(form: Mapping[str, str], contests: Mapping[str, dict],
                player_ids: set[str]) -> tuple[str, PF.Settings]:
    """(contest id, settings) from the POSTed form, or PortfolioError with a plain message."""
    cid = (form.get("contest") or "").strip()
    c = contests.get(cid)
    if c is None:
        raise PF.PortfolioError("Pick one of this slate's contests.")
    try:
        n = int(form.get("n") or N_DEFAULT)
        mx = float(form.get("max_exp") or EXP_DEFAULT)
        gamma = int(form.get("gamma") or GAMMA_DEFAULT)
    except ValueError:
        raise PF.PortfolioError("Entries, exposure and overlap must be numbers.") from None
    if n < 1:
        raise PF.PortfolioError("Choose at least 1 entry.")
    if n > c["max_n"]:
        raise PF.PortfolioError(f"This contest allows at most {c['max_n']} entries per user.")
    if not 10 <= mx <= 100:
        raise PF.PortfolioError("Max exposure must be between 10% and 100%.")
    if gamma not in GAMMA_CHOICES:
        raise PF.PortfolioError("Pick a shared-player limit from the list.")
    locks, excl = [], []
    for k, val in form.items():
        if not k.startswith("p_") or val not in ("lock", "exclude"):
            continue
        pid = k[2:]
        if pid not in player_ids:
            raise PF.PortfolioError("A selected player is not in this slate's pool.")
        (locks if val == "lock" else excl).append(pid)
    if len(locks) > 4:
        raise PF.PortfolioError("Lock at most 4 players: more leaves little to diversify.")
    s = PF.Settings(n=n, gamma=gamma, max_exposure=mx / 100.0, locks=tuple(sorted(locks)),
                    excludes=tuple(sorted(excl)), strengths=True)
    return cid, s


# ===================================================================== routes

def register(app, *, store, render: Callable, not_found: Callable, today: Callable[[], date],
             health_banner: Callable | None = None) -> None:
    builds = Builds()
    app.state.contest_builds = builds
    flashes: dict[str, dict] = {}

    def now() -> datetime:
        fn = getattr(app.state, "dfs_now", None)
        return fn() if fn else datetime.now(timezone.utc)

    def stash(msg: dict) -> str:
        tok = secrets.token_urlsafe(8)
        if len(flashes) >= MAX_FLASH:
            flashes.pop(next(iter(flashes)))
        flashes[tok] = msg
        return tok

    def page(day: str, slate_id: str, p: str | None, r: str | None,
             form: Mapping[str, str] | None = None, status: int = 200):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "DFS URLs look like /dfs/2026-10-20.")
        conn = store.health_conn()
        try:
            vm = build_page(conn, day, slate_id, now=now(), portfolio_id=p, builds=builds,
                            flash=flashes.get(r or ""), form=form)
        finally:
            if conn is not None:
                conn.close()
        if vm["state"] == "no_such_slate":
            return not_found("Slate not found", f"There is no slate {slate_id} on {day}.")
        if vm["state"] == "no_such_portfolio":
            return not_found("Build not found", "That contest build does not exist on this slate.")
        resp = render("dfs_contest.html", v=vm, page="dfs", status=status,
                      health_banner=health_banner() if health_banner else None)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/dfs/{day}/{slate_id}/contest", response_class=HTMLResponse)
    def contest_page(day: str, slate_id: str, p: str | None = None, r: str | None = None):
        return page(day, slate_id, p, r)

    @app.post("/dfs/{day}/{slate_id}/contest")
    async def contest_build(day: str, slate_id: str, request: Request):
        if not same_origin(request):
            return PlainTextResponse("Cross-site request refused.", status_code=403)
        body = await read_body(request, 256 * 1024)
        if body is None:
            return PlainTextResponse("Request too large.", status_code=413)
        form = parse_form(body)
        base = f"/dfs/{day}/{slate_id}/contest"
        conn = store.health_conn()
        try:
            vm = build_page(conn, day, slate_id, now=now(), portfolio_id=None, builds=builds)
        finally:
            if conn is not None:
                conn.close()
        if vm["state"] != "ok":
            return not_found("Slate not found", f"There is no slate {slate_id} on {day}.")

        def err(msg: str):
            return page(day, slate_id, None, stash({"kind": "error", "text": msg}), form=form,
                        status=400)

        if not vm["can_build"]:
            return err(vm["why_not"])
        if builds.busy():
            return err("Another build is running. Try again when it finishes (usually under "
                       "two minutes).")
        try:
            cid, s = parse_build(form, {c["contest_id"]: c for c in vm["contests"]},
                                 {p["id"] for p in vm["players"]})
        except PF.PortfolioError as e:
            return err(str(e))
        pid = PF.new_id()
        conn = db.connect(store.db_path)
        try:
            PF.start_row(conn, pid, slate_id, cid, s)
        finally:
            conn.close()
        if not builds.start(store.db_path, pid, slate_id, cid, s, now()):
            conn = db.connect(store.db_path)
            try:
                PF.fail_row(conn, pid, "Another build was running.")
            finally:
                conn.close()
            return err("Another build is running. Try again when it finishes.")
        return RedirectResponse(f"{base}?p={pid}", status_code=303)

    @app.get("/dfs/{day}/{slate_id}/contest/{portfolio_id}.json")
    def contest_json(day: str, slate_id: str, portfolio_id: str):
        """Every raw number of a build (simulated, uncalibrated), for inspection."""
        conn = store.health_conn()
        if conn is None:
            return JSONResponse({"error": "no database"}, status_code=404)
        try:
            d = PF.load(conn, portfolio_id)
        finally:
            conn.close()
        if d is None or str(d.get("slate_id")) != slate_id:
            return JSONResponse({"error": "not_found"}, status_code=404)
        data = d.get("data") or {}
        body = {"portfolio_id": portfolio_id, "slate_id": slate_id, "status": d["status"],
                "label": "simulated against a modelled field (proxy), uncalibrated: not real "
                         "results", "contest": data.get("contest"), "settings": d["settings"],
                "summary": data.get("summary"), "exposure": data.get("exposure"),
                "truth": (data.get("info") or {}).get("truth"),
                "curve": (data.get("summary") or {}).get("curve"),
                "lineups": [{k: lu.get(k) for k in ("rank", "salary", "mean", "p10", "p50",
                                                    "p90", "p_cash", "p_top1", "e_payout",
                                                    "dupes")} | {"slots": lu["slots"]}
                            for lu in d["lineups"]]}
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/dfs/{day}/{slate_id}/contest/{portfolio_id}.csv")
    def contest_csv(day: str, slate_id: str, portfolio_id: str,
                    style: str = Query("id", alias="fmt")):
        if style not in dfs_export.STYLES:
            return PlainTextResponse("fmt must be id or name_id.", status_code=400)
        conn = store.health_conn()
        if conn is None:
            return PlainTextResponse("No database.", status_code=404)
        try:
            d = PF.load(conn, portfolio_id)
        finally:
            conn.close()
        if d is None or str(d.get("slate_id")) != slate_id or d["status"] != "ok" \
                or not d["lineups"]:
            return PlainTextResponse("No finished build with that id on this slate.",
                                     status_code=404)
        try:
            body = dfs_export.to_csv([lu["slots"] for lu in d["lineups"]], style=style)
        except dfs_export.ExportError as e:
            return PlainTextResponse(f"Cannot export: {e}", status_code=409)
        name = f"courtside-contest-{day}-{d.get('contest_id') or slate_id}-{portfolio_id}.csv"
        return Response(body, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"',
                                 "Cache-Control": "no-store"})
