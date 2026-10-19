"""The `/dfs/results` page: upload DraftKings results, see the owner's REAL entries and P&L.

Routes (all on the same tailnet-only server as the rest of the site)
  GET  /dfs/results                       the page
  POST /dfs/results/upload                multipart: file (+ contest_id, slate_id, username, remember)
  POST /dfs/results/entry                 add an entry by hand
  POST /dfs/results/entry/<id>            set the fee / winnings of an entry
  POST /dfs/results/link                  attach an uploaded contest to a slate
  POST /dfs/results/username              remember the owner's DK username

Everything shown here is real data (uploads and hand entries). Simulated builds live on /dfs and
the DFS tab of /health. This is the only place the site writes to the database, so each POST
opens its own short write connection (the rest of the web app is read-only), checks that the
request comes from this site (Origin / Sec-Fetch-Site) and caps the body size. No python-multipart
dependency: the multipart body is parsed with the standard library.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs, urlparse

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from .. import db
from ..proj.dfs import standings as ST
from . import fmt

BODY_SLACK = 2 * 1024 * 1024        # multipart overhead on top of the file limit
MAX_FLASH = 24


# ===================================================================== request helpers

def same_origin(request: Request) -> bool:
    """Reject cross-site form posts: Origin (when sent) must be this host; browsers that send
    Sec-Fetch-Site must say same-origin / none."""
    h = request.headers
    sfs = h.get("sec-fetch-site")
    if sfs and sfs not in ("same-origin", "none"):
        return False
    origin = h.get("origin")
    if origin is None:
        return True
    return origin != "null" and urlparse(origin).netloc == h.get("host", "")


async def read_body(request: Request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than `limit`."""
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:
        return None
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit:
            return None
    return bytes(buf)


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """({field: text}, {field: (filename, bytes)}) of a multipart/form-data body."""
    msg = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\n\r\n" + body)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    if not msg.is_multipart():
        return fields, files
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fn = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if fn is not None:
            files[name] = (fn, payload)
        else:
            fields[name] = payload.decode("utf-8", "replace").strip()
    return fields, files


def parse_form(body: bytes) -> dict[str, str]:
    return {k: v[-1].strip() for k, v in parse_qs(body.decode("utf-8", "replace"),
                                                  keep_blank_values=True).items()}


def _float(s: str | None) -> float | None:
    s = (s or "").strip().replace(",", "").replace("$", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ST.StandingsError(f"'{s}' is not a number.") from None


def _int(s: str | None) -> int | None:
    v = _float(s)
    return None if v is None else int(v)


# ===================================================================== view model

def _money(v, signed: bool = False) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return fmt.EM
    if x != x:
        return fmt.EM
    sign = ("+" if x > 0 else fmt.MINUS if x < 0 else "") if signed else (fmt.MINUS if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def _rows(conn, sql, args=()):
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.Error:
        return []


def slate_options(conn) -> list[dict]:
    out = []
    for s in _rows(conn, "SELECT slate_id, nba_day, tag FROM dfs_slates ORDER BY nba_day DESC, "
                         "lock_utc DESC LIMIT 40"):
        out.append({"id": str(s["slate_id"]),
                    "label": f"{fmt.short_date(s['nba_day'])} {(s['tag'] or 'slate').capitalize()} "
                             f"({s['slate_id']})"})
    return out


def lineup_options(conn) -> list[dict]:
    """Our stored lineups of the last ~10 slates: the run in effect at lock (lock10, else lock45)."""
    out = []
    slates = _rows(conn, "SELECT slate_id, nba_day, tag FROM dfs_slates ORDER BY nba_day DESC, "
                         "lock_utc DESC LIMIT 10")
    for s in slates:
        runs = _rows(conn, "SELECT run_id, kind FROM dfs_runs WHERE slate_id=? AND ok=1 AND "
                           "kind IN ('lock10','lock45') ORDER BY kind='lock10' DESC, created_at DESC "
                           "LIMIT 1", (s["slate_id"],))
        if not runs:
            continue
        for lu in _rows(conn, "SELECT id, mode, rank, mean FROM dfs_lineups WHERE run_id=? AND "
                              "mode IN ('cash','topn','gpp') ORDER BY mode='cash' DESC, mode, rank "
                              "LIMIT 12", (runs[0]["run_id"],)):
            mode = {"cash": "Cash", "topn": "Top N", "gpp": "GPP"}.get(lu["mode"], lu["mode"])
            out.append({"id": lu["id"], "slate_id": str(s["slate_id"]),
                        "label": f"{fmt.short_date(s['nba_day'])} {(s['tag'] or '').capitalize()} "
                                 f"{mode} #{lu['rank'] or 1} (proj {fmt.num(lu['mean'], 1)})"})
    return out


def _lineup_label(conn, lineup_id) -> str:
    if not lineup_id:
        return ""
    r = _rows(conn, "SELECT l.mode, l.rank, r.kind FROM dfs_lineups l LEFT JOIN dfs_runs r "
                    "ON r.run_id=l.run_id WHERE l.id=?", (lineup_id,))
    if not r:
        return "our lineup"
    mode = {"cash": "cash", "topn": "Top N", "gpp": "GPP", "swap": "swap"}.get(r[0]["mode"], r[0]["mode"])
    return f"our {mode} #{r[0]['rank'] or 1}"


def build_page(conn: sqlite3.Connection | None, *, flash: dict | None = None,
               data_dir=None) -> dict:
    users = ST.owner_usernames(data_dir)
    vm = {"flash": flash, "usernames": users, "needs_username": not users,
          "max_mb": ST.MAX_UPLOAD_BYTES // (1024 * 1024), "slates": [], "lineups": [],
          "contests": [], "entries": [], "totals": None, "has_db": conn is not None}
    if conn is None:
        return vm
    vm["slates"] = slate_options(conn)
    vm["lineups"] = lineup_options(conn)
    res = ST.my_results(conn)
    t = res["totals"]
    vm["totals"] = {
        "entries": t["entries"], "settled": t["settled"], "pending": t["pending"],
        "fees": _money(t["fees"]), "winnings": _money(t["winnings"]) if t["settled"] else fmt.EM,
        "net": _money(t["net"], True) if t["settled"] else fmt.EM,
        "net_sign": ("pos" if t["net"] > 0 else "neg" if t["net"] < 0 else "") if t["settled"] else "",
        "roi": fmt.pct(t["roi"], 1) if t["roi"] is not None else fmt.EM,
        "cashed": f"{t['cashed']} of {t['settled']}" if t["settled"] else fmt.EM,
        "no_fee": t["no_fee"]}
    for e in res["entries"]:
        vm["entries"].append({
            "id": e["entry_id"], "day": e.get("nba_day") or (e.get("created_at") or "")[:10],
            "contest": e.get("contest_name") or f"Contest {e['contest_id'] or '?'}",
            "contest_id": e["contest_id"], "source": e["source"],
            "fee": _money(e.get("entry_fee")), "fee_missing": e.get("entry_fee") is None,
            "points": fmt.num(e.get("points"), 1),
            "place": (f"{e['rank']:,} of {e['field_size']:,}" if e.get("rank") and e.get("field_size")
                      else (f"{e['rank']:,}" if e.get("rank") else fmt.EM)),
            "pct": (f"top {e['percentile']:.1f}%" if e.get("percentile") is not None else ""),
            "payout": _money(e.get("payout")), "pending": e.get("payout") is None,
            "net": _money(e.get("net"), True),
            "net_sign": ("pos" if (e.get("net") or 0) > 0 else "neg" if (e.get("net") or 0) < 0 else ""),
            "lineup": _lineup_label(conn, e.get("lineup_id")),
            "players": ", ".join(s.get("name") or "" for s in e["slots"][:8]) if e["slots"] else ""})
    for c in ST.list_contests(conn):
        top = c["ownership"]["top"][:3]
        vm["contests"].append({
            "contest_id": c["contest_id"], "name": c["name"] or f"Contest {c['contest_id']}",
            "day": fmt.medium_date(c["nba_day"]) if c["nba_day"] else "", "slate_id": c["slate_id"],
            "game_type": (c.get("game_type") or "classic").capitalize(),
            "field": f"{c['field_size']:,}", "win": fmt.num(c["winning_score"], 1),
            "top1": fmt.num(c["top1_score"], 1), "median": fmt.num(c["median_score"], 1),
            "line": fmt.num(c["cash_line"], 1) if c["cash_line"] is not None else fmt.EM,
            "line_note": ("from the payout table" if c["cash_line_source"] == "payout table"
                          else "from paid rows" if c["cash_line_source"] else
                          "unknown: no payout table for this contest"),
            "paid": f"{c['paid_places']:,}" if c["paid_places"] else fmt.EM,
            "own_top": [f"{o['name']} {o['pct']:.1f}%" for o in top if o.get("pct") is not None],
            "own_sum": fmt.num(c["ownership"]["sum_pct"], 0) if c["ownership"]["sum_pct"] else "",
            "unmatched": [f"{u['name']}" for u in c["unmatched"][:12]],
            "n_unmatched": len(c["unmatched"]), "linked": bool(c["slate_id"]),
            "slate_how": c.get("slate_how") or "", "uploaded": fmt.stamp(c["uploaded_at"]),
            "filename": c["filename"] or "",
            "fee": _money(c["entry_fee"]) if c["entry_fee"] is not None else ""})
    return vm


# ===================================================================== routes

def register(app, *, store, render, not_found, today, health_banner=None) -> None:
    """Add the /dfs/results routes. Register BEFORE `dfs_view.register` ('/dfs/{day}' would
    otherwise capture '/dfs/results')."""
    flashes: dict[str, dict] = {}

    def banner():
        return health_banner() if health_banner else None

    def stash(msg: dict) -> str:
        tok = secrets.token_urlsafe(8)
        if len(flashes) >= MAX_FLASH:
            flashes.pop(next(iter(flashes)))
        flashes[tok] = msg
        return tok

    def back(msg: dict):
        return RedirectResponse(f"/dfs/results?r={stash(msg)}", status_code=303)

    def wconn() -> sqlite3.Connection:
        return db.connect(store.db_path)

    def forbidden():
        return PlainTextResponse("Cross-site request refused.", status_code=403)

    @app.get("/dfs/results", response_class=HTMLResponse)
    def results_page(request: Request, r: str | None = None):
        conn = store.health_conn()
        try:
            vm = build_page(conn, flash=flashes.get(r or ""), data_dir=store.data_dir)
        finally:
            if conn is not None:
                conn.close()
        return render("dfs_results.html", v=vm, page="dfs", health_banner=banner())

    @app.post("/dfs/results/upload")
    async def upload(request: Request):
        if not same_origin(request):
            return forbidden()
        body = await read_body(request, ST.MAX_UPLOAD_BYTES + BODY_SLACK)
        if body is None:
            return back({"kind": "error", "title": "That file is too large",
                         "lines": [f"The limit is {ST.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."]})
        ct = request.headers.get("content-type", "")
        if not ct.lower().startswith("multipart/form-data"):
            return back({"kind": "error", "title": "No file received", "lines": []})
        fields, files = parse_multipart(ct, body)
        fn, data = files.get("file", ("", b""))
        if not data:
            return back({"kind": "error", "title": "Choose a file first",
                         "lines": ["Pick the standings CSV (or zip) you downloaded from DraftKings."]})
        username = fields.get("username", "").strip()[:ST.MAX_NAME_LEN]
        conn = wconn()
        try:
            if username and fields.get("remember"):
                ST.set_owner_username(username, store.data_dir)
            owners = [username] if username else None
            res = ST.import_upload(conn, data, fn or "upload.csv",
                                   contest_id=re.sub(r"\D", "", fields.get("contest_id", "")) or None,
                                   slate_id=fields.get("slate_id") or None,
                                   owner_names=owners, data_dir=store.data_dir)
        except ST.StandingsError as e:
            return back({"kind": "error", "title": "Could not import that file", "lines": [str(e)]})
        finally:
            conn.close()
        return back(_flash_for(res, fn))

    @app.post("/dfs/results/entry")
    async def add_entry(request: Request):
        if not same_origin(request):
            return forbidden()
        body = await read_body(request, 256 * 1024)
        if body is None:
            return PlainTextResponse("Too large.", status_code=413)
        f = parse_form(body)
        conn = wconn()
        try:
            eid = ST.add_manual_entry(
                conn, contest_id=f.get("contest_id", ""), entry_fee=_float(f.get("fee")),
                points=_float(f.get("points")), rank=_int(f.get("rank")),
                payout=_float(f.get("payout")), field_size=_int(f.get("field_size")),
                lineup_id=_int(f.get("lineup_id")) or None, lineup_text=f.get("lineup_text"),
                slate_id=f.get("slate_id") or None, note=f.get("note"))
        except ST.StandingsError as e:
            return back({"kind": "error", "title": "Could not add that entry", "lines": [str(e)]})
        finally:
            conn.close()
        return back({"kind": "ok", "title": "Entry added",
                     "lines": [f"Saved as entry {eid}. Winnings can be added later."
                               if not f.get("payout") else f"Saved as entry {eid}."]})

    @app.post("/dfs/results/entry/{entry_id}")
    async def edit_entry(entry_id: int, request: Request):
        if not same_origin(request):
            return forbidden()
        body = await read_body(request, 64 * 1024)
        if body is None:
            return PlainTextResponse("Too large.", status_code=413)
        f = parse_form(body)
        conn = wconn()
        try:
            ok = ST.update_entry(conn, entry_id, entry_fee=_float(f.get("fee")),
                                 payout=_float(f.get("payout")))
        except ST.StandingsError as e:
            return back({"kind": "error", "title": "Could not save", "lines": [str(e)]})
        finally:
            conn.close()
        return back({"kind": "ok" if ok else "error",
                     "title": "Saved" if ok else "No such entry", "lines": []})

    @app.post("/dfs/results/link")
    async def link(request: Request):
        if not same_origin(request):
            return forbidden()
        body = await read_body(request, 64 * 1024)
        if body is None:
            return PlainTextResponse("Too large.", status_code=413)
        f = parse_form(body)
        conn = wconn()
        try:
            ST.link_contest_slate(conn, f.get("contest_id", ""), f.get("slate_id", ""))
        except ST.StandingsError as e:
            return back({"kind": "error", "title": "Could not link", "lines": [str(e)]})
        finally:
            conn.close()
        return back({"kind": "ok", "title": "Contest linked to the slate",
                     "lines": ["Its real cash line is used the next time that day is graded."]})

    @app.post("/dfs/results/username")
    async def username(request: Request):
        if not same_origin(request):
            return forbidden()
        body = await read_body(request, 16 * 1024)
        f = parse_form(body or b"")
        try:
            ST.set_owner_username(f.get("username", ""), store.data_dir)
        except ST.StandingsError as e:
            return back({"kind": "error", "title": "Could not save", "lines": [str(e)]})
        return back({"kind": "ok", "title": "Username saved",
                     "lines": ["Upload a standings file again to pick up your entries "
                               "(a file you already uploaded is re-read, nothing is duplicated)."]})


def _flash_for(res: dict, filename: str) -> dict:
    """The message shown after an upload."""
    if res.get("kind") == "zip":
        lines = []
        for m in res.get("members", []):
            if m.get("status") == "error":
                lines.append(f"{m['filename']}: {m['error']}")
            else:
                lines.append(f"{m['filename']}: {m['status']}"
                             + (f", contest {m['contest_id']}" if m.get("contest_id") else ""))
        return {"kind": "ok", "title": "Zip imported", "lines": lines}
    if res.get("kind") == "history":
        if res["status"] == "duplicate":
            return {"kind": "info", "title": "Already imported", "lines": [filename]}
        return {"kind": "ok", "title": "Contest history imported",
                "lines": [f"{res['entries']} NBA entries ({res['new_entries']} new)."
                          + (f" {res['skipped']} rows skipped (other sports)." if res["skipped"] else "")]}
    cid = res.get("contest_id")
    if res["status"] == "duplicate":
        lines = [f"Contest {cid} was already imported from this exact file; nothing changed."]
        if res.get("owner_entries"):
            lines.append(f"Your username matched {res['owner_entries']} entr"
                         f"{'y' if res['owner_entries'] == 1 else 'ies'}; they were added.")
        return {"kind": "info", "title": "Already imported", "lines": lines}
    lines = [f"Contest {cid}: {res['entries']:,} entries, "
             f"{res['ownership_rows']} players' ownership."]
    if res.get("slate_id"):
        lines.append(f"Linked to slate {res['slate_id']} ({res['slate_how']}).")
    else:
        lines.append("Could not tell which slate this contest was. Choose it in the list below "
                     "so its cash line can be used for grading.")
    if res.get("owner_entries"):
        n = res["owner_entries"]
        lines.append(f"Found {n} of your entr{'y' if n == 1 else 'ies'}.")
    elif res.get("owner_configured"):
        lines.append("Your DK username did not match any entry in this contest.")
    else:
        lines.append("No DK username is set, so none of your entries were picked up. Enter it "
                     "above and upload this file again.")
    if not res.get("payout_table"):
        lines.append("No payout table is stored for this contest, so its cash line and your "
                     "winnings are unknown for now.")
    if res.get("unmatched"):
        lines.append(f"{len(res['unmatched'])} player name(s) could not be matched to an NBA "
                     "player; they are listed under the contest.")
    if res.get("locked"):
        lines.append(f"{res['locked']} lineups were hidden (LOCKED) in the file.")
    return {"kind": "ok", "title": "Standings imported", "lines": lines}
