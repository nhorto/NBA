"""Official NBA injury report (PDF on ak-static.cms.nba.com). Pure: fetch + parse only.

URL scheme (verified 2026-10-18):
    https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM|PM.pdf
with a zero-padded 12-hour clock in US/Eastern, e.g. ``Injury-Report_2026-04-10_05_15PM.pdf``.
Snapshots exist every 15 minutes (:00/:15/:30/:45) around the clock in 2026; reports from
before late December 2025 are no longer on the server. A missing snapshot answers 403
(S3 AccessDenied XML), not 404.

Parsing uses pdfplumber (pure Python, optional extra ``pdf``). The PDF is a bordered table
(Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason) in which
the first four cells are blank when they repeat the row above, long reasons wrap onto a
second line (vertically centred in the cell), and a row can even straddle a page break.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import http, teams
from ..config import NBA_TZ

SOURCE = "nba_official"
BASE_URL = "https://ak-static.cms.nba.com/referee/injury/"
GRID_MINUTES = 15
MAX_PROBES = 40
FINE_GRID_HOURS = 2  # probe the 15-min grid this far back, then hourly

STATUSES = ("Out", "Questionable", "Doubtful", "Probable", "Available")
NOT_SUBMITTED = "NOT YET SUBMITTED"

# Fallback column left edges (x, points) for a landscape-letter report; real edges are read
# from the header rects when present.
_DEFAULT_EDGES = (19.0, 116.0, 196.0, 260.0, 421.0, 582.0, 662.0, 823.0)


class ParserUnavailable(RuntimeError):
    """pdfplumber is not installed (install the ``pdf`` extra: ``uv sync --extra pdf``)."""


class ReportParseError(ValueError):
    """The PDF did not look like an NBA injury report."""


# --------------------------------------------------------------------------- URLs

def _floor_to_grid(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("report time must be timezone-aware")
    et = dt.astimezone(NBA_TZ)
    return et.replace(minute=et.minute - et.minute % GRID_MINUTES, second=0, microsecond=0)


def report_url(report_time: datetime) -> str:
    """URL of the snapshot at `report_time` (aware), floored to the 15-minute grid, in ET."""
    et = _floor_to_grid(report_time)
    hour12 = et.hour % 12 or 12
    ampm = "AM" if et.hour < 12 else "PM"
    return f"{BASE_URL}Injury-Report_{et:%Y-%m-%d}_{hour12:02d}_{et.minute:02d}{ampm}.pdf"


def _probe_times(now: datetime, max_back_hours: int, max_probes: int) -> list[datetime]:
    """Newest-first slots: 15-min grid for the first FINE_GRID_HOURS, then hourly."""
    # Work in UTC so DST transitions never produce skipped/duplicated wall-clock slots.
    start = _floor_to_grid(now).astimezone(timezone.utc)
    limit = start - timedelta(hours=max_back_hours)
    out: list[datetime] = []
    seen: set[str] = set()
    t = start
    while t >= limit and len(out) < max_probes:
        et = t.astimezone(NBA_TZ)
        fine = (start - t) < timedelta(hours=FINE_GRID_HOURS)
        if fine or et.minute == 0:
            url = report_url(et)
            if url not in seen:
                seen.add(url)
                out.append(et)
        t -= timedelta(minutes=GRID_MINUTES)
    return out


def _exists(url: str) -> bool:
    try:
        http.fetch_bytes(url, retries=0, timeout=15.0)
    except http.FetchError as e:
        if e.status in (403, 404):
            return False
        raise
    return True


def find_latest_report(now: datetime | None = None, max_back_hours: int = 24,
                       max_probes: int = MAX_PROBES) -> tuple[str, datetime] | None:
    """Newest published report at or before `now`: (url, ET snapshot time), or None.

    Probes the 15-minute grid for the last two hours, then hourly slots, newest first.
    403/404 mean "not published"; any other error propagates.
    """
    now = now or datetime.now(timezone.utc)
    for slot in _probe_times(now, max_back_hours, max_probes):
        url = report_url(slot)
        if _exists(url):
            return url, slot
    return None


def fetch_report(url: str) -> bytes:
    data = http.fetch_bytes(url, retries=2)
    if not data.startswith(b"%PDF"):
        raise http.FetchError(f"not a PDF: {url}", url=url)
    return data


# --------------------------------------------------------------------------- parsing

def parser_available() -> bool:
    return importlib.util.find_spec("pdfplumber") is not None


def _team_names() -> dict[str, str]:
    names = {f"{t.city} {t.name}".lower(): t.tricode for t in teams.TEAMS}
    names["la clippers"] = "LAC"
    names["los angeles clippers"] = "LAC"
    return names


def team_tricode(name: str | None) -> str | None:
    """'Boston Celtics' -> 'BOS'; unknown names -> None."""
    if not name:
        return None
    return _team_names().get(re.sub(r"\s+", " ", name).strip().lower())


def normalize_player_name(name: str | None) -> str | None:
    """'Jackson Jr., Jaren' -> 'Jaren Jackson Jr.'"""
    if not name:
        return None
    if "," in name:
        last, _, first = name.partition(",")
        return f"{first.strip()} {last.strip()}".strip()
    return name.strip()


def _clean(cell: Any) -> str:
    return re.sub(r"\s+", " ", cell).strip() if cell else ""


def _status(raw: str) -> str:
    for s in STATUSES:
        if raw.lower() == s.lower():
            return s
    return raw


def _iso_date(raw: str) -> str:
    try:
        return datetime.strptime(raw, "%m/%d/%Y").date().isoformat()
    except ValueError:
        return raw


def _time_24h(raw: str) -> str | None:
    """'01:00 (ET)' -> '13:00'. The PDF omits AM/PM; assume games start 11:xx or noon-10:xx PM."""
    m = re.match(r"(\d{1,2}):(\d{2})", raw)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h != 11 and h != 12:
        h += 12
    return f"{h:02d}:{mi:02d}"


def rows_from_table(table: Iterable[Sequence[Any]]) -> list[dict]:
    """Turn raw table rows into records, carrying blank date/time/matchup/team cells forward.

    Each input row is 7 cells: Game Date, Game Time, Matchup, Team, Player Name, Current
    Status, Reason (None or "" for blank). Header rows are skipped. A row with only a
    reason (a cell that wrapped across a page break) extends the previous record's reason.
    A "NOT YET SUBMITTED" team becomes a record with that status and player_name None.
    """
    out: list[dict] = []
    carry = {"date": "", "time": "", "matchup": "", "team": ""}
    for raw in table:
        cells = [_clean(c) for c in list(raw)[:7]]
        cells += [""] * (7 - len(cells))
        date_, time_, matchup, team, player, status, reason = cells
        if not any(cells) or player.lower() == "player name" or date_.lower() == "game date":
            continue
        if not (date_ or time_ or matchup or team or player or status) and reason:
            if out:
                prev = out[-1]
                prev["reason"] = f"{prev['reason']} {reason}".strip() if prev["reason"] else reason
            continue
        for key, val in (("date", date_), ("time", time_), ("matchup", matchup), ("team", team)):
            if val:
                carry[key] = val
        if not (player or status or reason):
            continue  # a bare team/game label with no player: carried forward, nothing to emit
        base = {
            "game_date": _iso_date(carry["date"]) if carry["date"] else None,
            "game_time": carry["time"] or None,
            "game_time_et": _time_24h(carry["time"]),
            "matchup": carry["matchup"] or None,
            "team": team_tricode(carry["team"]),
            "team_name": carry["team"] or None,
            "source": SOURCE,
        }
        if NOT_SUBMITTED in " ".join((player, status, reason)).upper():
            out.append({**base, "player_name": None, "player_name_norm": None,
                        "status": NOT_SUBMITTED, "reason": None})
            continue
        out.append({**base, "player_name": player or None,
                    "player_name_norm": normalize_player_name(player),
                    "status": _status(status),
                    "reason": None if reason in ("", "-") else reason})
    return out


def _import_pdfplumber() -> Any:
    try:
        import pdfplumber
    except ImportError as e:
        raise ParserUnavailable(
            "pdfplumber is not installed; run `uv sync --extra pdf` "
            "(the raw PDF can still be archived)") from e
    return pdfplumber


def _page_table(page: Any) -> list[list[str]]:
    """Cell texts for one page, using the player-column horizontal rules as row borders."""
    words = page.extract_words(x_tolerance=1)
    anchor = next((w for w in words if w["text"] == "Matchup"), None)
    if anchor is not None:
        body_top = anchor["bottom"]
    else:
        title = [w for w in words if w["text"].startswith("Report")]
        if not title:
            raise ReportParseError("page has no injury report title")
        body_top = title[0]["bottom"]

    edges = list(_DEFAULT_EDGES)
    header = sorted(r["x0"] for r in page.rects if abs(r["bottom"] - r["top"] - 22) < 3
                    and r["x0"] > 5)
    if len(header) == 7:
        edges = header + [max(r["x1"] for r in page.rects if r["x0"] > 5)]
    p_lo, p_hi = edges[4], edges[5]

    ys = sorted({round(l["top"], 1) for l in page.lines
                 if abs(l["bottom"] - l["top"]) < 1 and l["x0"] <= p_lo + 5
                 and l["x1"] >= p_hi - 5 and l["top"] > body_top})
    if not ys:
        if any(w["top"] > body_top + 2 and not w["text"].startswith(("Page", "of"))
               for w in words):
            raise ReportParseError("no table rules found on page")
        return []
    bounds = [body_top] + ys
    rows: list[list[list[Any]]] = [[[] for _ in range(7)] for _ in ys]
    for w in words:
        mid = (w["top"] + w["bottom"]) / 2
        if mid <= body_top or mid > bounds[-1]:
            continue
        r = next(i for i in range(len(ys)) if bounds[i] <= mid <= bounds[i + 1])
        c = max(i for i in range(7) if w["x0"] + 1 >= edges[i] or i == 0)
        rows[r][c].append(w)
    table = []
    for cells in rows:
        table.append([" ".join(w["text"] for w in sorted(ws, key=lambda w: (round(w["top"]), w["x0"])))
                      for ws in cells])
    return table


def parse_report(pdf: bytes) -> list[dict]:
    """Parse an injury report PDF into rows (see module docstring / rows_from_table)."""
    import io

    pdfplumber = _import_pdfplumber()
    table: list[list[str]] = []
    report_time: str | None = None
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        if not doc.pages:
            raise ReportParseError("empty PDF")
        for page in doc.pages:
            if report_time is None:
                m = re.search(r"Injury Report:\s*(\d\d/\d\d/\d\d)\s+(\d\d:\d\d [AP]M)",
                              page.extract_text(x_tolerance=1) or "")
                if m:
                    dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%m/%d/%y %I:%M %p")
                    report_time = dt.replace(tzinfo=NBA_TZ).isoformat()
            table.extend(_page_table(page))
    if report_time is None:
        raise ReportParseError("no 'Injury Report:' title found")
    rows = rows_from_table(table)
    for r in rows:
        r["report_time"] = report_time
    return rows
