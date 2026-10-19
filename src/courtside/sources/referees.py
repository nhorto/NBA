"""NBA referee assignments (https://official.nba.com/referee-assignments/).

The page is server-rendered HTML, refreshed around 9:00 am ET on game days. Structure (verified
against Wayback captures from 2025-11 to 2026-04 plus the live empty page of 2026-10-19)::

    <article class="... referee-assignments ...">
      <h1 class="entry-title">NBA Referee Assignments</h1>
      <div class="entry-meta">January 18, 2026</div>          <- the NBA day of the table
      <div class="dayContent"><div class="nba-refs-content"><table class="table">
        <thead> Game | Crew Chief | Referee | Umpire | Alternate </thead>
        <tbody> <tr><td>Orlando @ Memphis</td><td>[<a>]Courtney Kirkland (#61)[</a>]</td> ... </tr>
      </table></div></div>
      <div class="replay-center-container"> ... Replay Center: A, B, C ... </div>
    </article>
    <!--G League--> <article> ... <div class="gl-refs-content"> ... (G League, ignored)
    <!--WNBA-->     <article> ... <div class="wnba-refs-content"> (ignored)

Cells hold the official as plain text or wrapped in a link to a bio PDF, with the jersey number
in parentheses; the Alternate cell is often empty. Game labels are "Away @ Home" with city names
("L.A. Lakers", "LA Clippers", "New York", ...). On an off day (or before posting) the tbody is
empty: `parse_assignments` then returns no games, never raises.

Pure fetch + parse; callers archive the HTML (``save_raw_bytes(..., ext='.html')``).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from html import unescape

from .. import http, teams

log = logging.getLogger(__name__)

URL = "https://official.nba.com/referee-assignments/"
SOURCE = "nba_official"
KIND = "referee_assignments"
ROLES = ("crew_chief", "referee", "umpire", "alternate")
_COLUMNS = {"crew chief": "crew_chief", "referee": "referee", "umpire": "umpire",
            "alternate": "alternate"}


# --------------------------------------------------------------------------- fetch

def fetch_page(url: str = URL, *, retries: int = 2, timeout: float = 25.0) -> bytes:
    """GET the page (HTML). `http.fetch_bytes` refuses HTML by design, so this uses the shared
    impersonating session and per-host throttle directly."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            http._sleep(1.5 ** attempt)
        http._throttle(url)
        try:
            resp = http._get_session().get(url, headers=http._build_headers(url, None),
                                           timeout=timeout)
        except Exception as e:  # noqa: BLE001 - timeouts / connection errors
            last = http.FetchError(f"{type(e).__name__}: {e}", url=url)
            continue
        if resp.status_code == 200 and resp.content:
            return resp.content
        last = http.FetchError(f"HTTP {resp.status_code}: {url}", status=resp.status_code,
                               url=url)
        if resp.status_code in (403, 404):
            break
    assert last is not None
    raise last


# --------------------------------------------------------------------------- team names

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", unescape(s).lower().replace(".", "")).strip()


def _build_team_names() -> dict[str, str]:
    out: dict[str, str] = {}
    cities: dict[str, list[str]] = {}
    for t in teams.TEAMS:
        cities.setdefault(_norm(t.city), []).append(t.tricode)
        out[_norm(f"{t.city} {t.name}")] = t.tricode
        out[_norm(t.name)] = t.tricode
    for city, codes in cities.items():
        if len(codes) == 1:          # "Los Angeles" alone is ambiguous
            out[city] = codes[0]
    out.update({"la lakers": "LAL", "la clippers": "LAC", "los angeles lakers": "LAL",
                "los angeles clippers": "LAC", "lakers": "LAL", "clippers": "LAC",
                "ny knicks": "NYK", "gs warriors": "GSW", "golden state": "GSW",
                "okc thunder": "OKC", "sa spurs": "SAS", "no pelicans": "NOP",
                "philadelphia 76ers": "PHI", "portland trail blazers": "POR",
                "trail blazers": "POR", "blazers": "POR", "sixers": "PHI"})
    return out


_TEAM_NAMES = _build_team_names()


def team_tricode(label: str) -> str | None:
    """'L.A. Lakers' / 'New York' / 'Knicks' -> NBA tricode (None if unknown or ambiguous)."""
    key = _norm(label)
    if key in _TEAM_NAMES:
        return _TEAM_NAMES[key]
    try:
        return teams.franchise(key.upper())
    except KeyError:
        return None


# --------------------------------------------------------------------------- parse

def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


_PERSON = re.compile(r"\s*([^()#,]+?)\s*\(\s*#?\s*(\d+)\s*\)")


def parse_officials(cell_html: str) -> list[dict]:
    """[{name, number}] from one table cell ('Zach Zarba (#15)', link-wrapped, or empty)."""
    text = _text(cell_html)
    if not text:
        return []
    found = [{"name": m.group(1).strip(), "number": int(m.group(2))}
             for m in _PERSON.finditer(text)]
    if found:
        return found
    return [{"name": n.strip(), "number": None} for n in text.split(",") if n.strip()]


def _nba_section(html: str) -> str:
    """The NBA <article> only (G League / WNBA tables share the markup)."""
    i = html.find("nba-refs-content")
    if i < 0:
        return ""
    # the article that owns the table ends at the next </article>
    j = html.find("</article>", i)
    return html[i: j if j > 0 else len(html)]


def _page_day(html: str) -> date | None:
    m = re.search(r"(?<!W)NBA Referee Assignments</h1>\s*<div class=\"entry-meta\">([^<]+)</div>", html)
    if not m:
        return None
    raw = _text(m.group(1))
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_assignments(html: bytes | str) -> dict:
    """Parse the page into ``{"nba_day": date|None, "games": [...], "replay_center": [...]}``.

    Each game: ``{"label", "away_name", "home_name", "away", "home", "crew_chief": [..],
    "referee": [..], "umpire": [..], "alternate": [..]}`` where each role is a list of
    ``{"name", "number"}``. Empty table -> ``games == []``.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    out: dict = {"nba_day": _page_day(html), "games": [], "replay_center": []}
    section = _nba_section(html)
    if not section:
        return out

    # header order -> roles (tolerates reordered / renamed columns)
    heads = [_COLUMNS.get(_text(h).lower()) for h in
             re.findall(r"<th[^>]*>(.*?)</th>", section, flags=re.S)]
    roles = [h for h in heads[1:]] if len(heads) > 1 else list(ROLES)
    if not any(roles):
        roles = list(ROLES)

    tbody = re.search(r"<tbody>(.*?)</tbody>", section, flags=re.S)
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody.group(1) if tbody else "", flags=re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
        if not cells:
            continue
        label = _text(cells[0])
        if not label:
            continue
        away_name, _, home_name = (p.strip() for p in label.partition("@"))
        game = {"label": label, "away_name": away_name, "home_name": home_name,
                "away": team_tricode(away_name) if away_name else None,
                "home": team_tricode(home_name) if home_name else None,
                **{r: [] for r in ROLES}}
        for role, cell in zip(roles, cells[1:]):
            if role:
                game[role] = parse_officials(cell)
        out["games"].append(game)

    rc = re.search(r"Replay Center:?\s*</strong>(.*?)</p>", section, flags=re.S)
    if rc:
        names = [n.strip() for n in _text(rc.group(1)).split(",") if n.strip()]
        out["replay_center"] = [{"name": n, "number": None} for n in names]
    return out


def assignment_rows(parsed: dict) -> list[dict]:
    """Flat rows: nba_day, game_label, home, away, official_name, official_number, role.
    Replay-center officials come with ``game_label`` None and role 'replay_center'."""
    day = parsed.get("nba_day")
    rows: list[dict] = []
    for g in parsed.get("games") or []:
        for role in ROLES:
            for o in g.get(role) or []:
                rows.append({"nba_day": day.isoformat() if day else None,
                             "game_label": g["label"], "home": g["home"], "away": g["away"],
                             "official_name": o["name"], "official_number": o["number"],
                             "role": role})
    for o in parsed.get("replay_center") or []:
        rows.append({"nba_day": day.isoformat() if day else None, "game_label": None,
                     "home": None, "away": None, "official_name": o["name"],
                     "official_number": o["number"], "role": "replay_center"})
    return rows
