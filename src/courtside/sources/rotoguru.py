"""RotoGuru daily DraftKings NBA salaries and points (history 2014-10-29 .. 2021-05).

One HTML page per NBA day, no login, no impersonation needed (a plain GET with a descriptive
User-Agent works; robots.txt is 404)::

    http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon=M&day=D&year=Y

The ``&scsv=1`` export stops after 2017, so the HTML view is the source. The page title says
"Fantasy Points for 15-Nov-2016". Rows sit in one table grouped Guards / Forwards / Centers /
Unlisted, each prefixed by a ``<!--RGID-->`` comment::

    Pos | Name (link playrh.cgi?<RGID>x, '^' after it = starter) | DK Pts | Salary | Team | Opp | Score | Min | Stats

Quirks (checked on 2014-2021 samples):

* Name is ``Last, First``; teams are lower-case RotoGuru codes (``nor`` = NOP, ``pho`` = PHX).
* ``Opp`` is ``@ cle`` (away) or ``v tor`` (home).
* Pos is one position through 2014-15, later ``SG/SF`` style; ``NA`` for players DK did not list.
* Salary ``$7,600``; ``N/A`` for a player of a game that was not on DraftKings' (main) slate.
  Those games still carry points and stats, so a missing salary is the off-slate signal.
* Min is ``39:43`` (2016+) or whole minutes (2014-15); ``DNP`` = active but did not play (0),
  ``NA`` = inactive (NULL). DNP/NA players have no box-score row in ``player_games``.
* Stats: ``28pt 5rb 9as 4st 1to 4trey 11-23fg 2-2ft`` plus ``bl``; a missing token is zero.
  ``trey`` is threes made. DK points are NOT in the stat string: ``recompute_points`` rebuilds
  them (with the double/triple-double bonuses) as a cross-check.
* The page is cp1252 (``player\\x92s``); player names are ASCII.
* A day with no games returns the page shell with no rows.

``fetch_page`` is the only network call (1 request/s, escalating back-off); ``parse_page`` is pure.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
from datetime import date

from .. import teams
from . import dfs

SOURCE = "rotoguru"
KIND = "dk"
URL = "http://rotoguru1.com/cgi-bin/hyday.pl"
USER_AGENT = "courtside-research/0.1 (personal NBA analytics; sequential, 1 request/s)"
MIN_INTERVAL = 1.0
RETRY_WAITS = (5, 20, 60, 180)     # seconds before each retry of one page
FIRST_DAY = "2014-10-29"           # first day RotoGuru has (the 2014-10-28 opener is missing)

# RotoGuru codes that differ from our tricodes once upper-cased (teams.franchise covers PHO).
TEAM_ALIASES = {"NOR": "NOP", "PHO": "PHX"}

_sleep = time.sleep   # monkeypatch in tests
_monotonic = time.monotonic
_last_request: list[float] = []


class PageError(Exception):
    """The response is not a complete RotoGuru page for the requested day."""


# ------------------------------------------------------------------------------ fetch

def day_url(day: date | str) -> str:
    d = date.fromisoformat(str(day))
    return f"{URL}?game=dk&mon={d.month}&day={d.day}&year={d.year}"


def _throttle() -> None:
    if _last_request:
        wait = MIN_INTERVAL - (_monotonic() - _last_request[0])
        if wait > 0:
            _sleep(wait)
    _last_request[:] = [_monotonic()]


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_page(day: date | str, *, timeout: float = 30.0,
               retries: tuple[int, ...] = RETRY_WAITS) -> bytes:
    """Raw HTML bytes of one day's page, checked complete. Retries (waiting ``retries[i]`` s) on
    network errors, 5xx/429 and truncated pages; raises ``PageError`` / ``OSError`` at the end."""
    last: Exception | None = None
    for attempt in range(len(retries) + 1):
        if attempt:
            _sleep(retries[attempt - 1])
        _throttle()
        try:
            data = _get(day_url(day), timeout)
            validate_page(data, day)
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = e
        except (OSError, PageError) as e:   # URLError / timeouts are OSErrors
            last = e
    assert last is not None
    raise last


# ------------------------------------------------------------------------------ parse

def _text(data: bytes | str) -> str:
    return data if isinstance(data, str) else data.decode("cp1252", errors="replace")


_TITLE = re.compile(r"Fantasy Points for (\d{1,2})-([A-Za-z]{3})-(\d{4})")
_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def page_date(data: bytes | str) -> str | None:
    """The day the page says it is for ('15-Nov-2016' -> '2016-11-15'), or None."""
    m = _TITLE.search(_text(data))
    if not m or m[2] not in _MONTHS:
        return None
    return date(int(m[3]), _MONTHS[m[2]], int(m[1])).isoformat()


def validate_page(data: bytes | str, day: date | str) -> None:
    """Raise PageError unless this is the complete page for ``day`` (title date + footer)."""
    s = _text(data)
    got = page_date(s)
    if got != str(day):
        raise PageError(f"page is for {got!r}, wanted {str(day)!r}")
    if "RotoGuru is produced by" not in s or "</BODY>" not in s.upper():
        raise PageError(f"truncated page for {day}")


_ROW = re.compile(r"<!--(\d+)-->\s*<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_NAME = re.compile(r"<a [^>]*>(.*?)</a>\s*(\^?)", re.S)
_TOKEN = re.compile(r"(\d+)(pt|rb|as|st|bl|to|trey)\b")
_STAT_KEYS = {"pt": "pts", "rb": "reb", "as": "ast", "st": "stl", "bl": "blk", "to": "tov",
              "trey": "fg3m"}


def team_code(code: str | None) -> str | None:
    """RotoGuru's lower-case team code -> our tricode (None if blank or unknown)."""
    if not code or not code.strip():
        return None
    c = code.strip().upper()
    try:
        return teams.franchise(TEAM_ALIASES.get(c, c))
    except KeyError:
        return None


def display_name(raw: str) -> str:
    """'Jackson Jr., Jaren' -> 'Jaren Jackson Jr.' (also unescapes HTML entities)."""
    s = html.unescape(raw).strip()
    last, sep, first = s.partition(", ")
    return f"{first.strip()} {last.strip()}" if sep else s


def parse_minutes(s: str) -> float | None:
    """'39:43' -> 39.717, '37' -> 37.0, 'DNP' -> 0.0 (active, did not play), 'NA' -> None."""
    s = s.strip()
    if s == "DNP":
        return 0.0
    m = re.fullmatch(r"(\d+)(?::(\d{2}))?", s)
    if not m:
        return None
    return int(m[1]) + (int(m[2]) / 60 if m[2] else 0.0)


def parse_stats(stats: str | None) -> dict[str, int]:
    """'28pt 5rb 9as 4st 1to 4trey 11-23fg 2-2ft' -> {pts, reb, ast, stl, tov, fg3m, ...}
    (blk included; makes/attempts are left out; absent categories are 0)."""
    out = dict.fromkeys(_STAT_KEYS.values(), 0)
    for n, tag in _TOKEN.findall(stats or ""):
        out[_STAT_KEYS[tag]] = int(n)
    return out


def recompute_points(stats: str | None) -> float:
    """DraftKings Classic points rebuilt from the stat string (DD +1.5, TD +3 on top)."""
    return dfs.classic_points(**parse_stats(stats))


def is_triple_double(stats: str | None) -> bool:
    s = parse_stats(stats)
    return sum(1 for k in ("pts", "reb", "ast", "stl", "blk") if s[k] >= 10) >= 3


def parse_page(data: bytes | str, day: date | str | None = None) -> list[dict]:
    """Player rows of one page, as ``dfs_hist_salaries`` dicts (without game_id / on_slate /
    raw_path / player_id). ``game_date`` is the page's own date unless ``day`` is given.
    A row with an unexpected shape is skipped; an unknown team code leaves team/opp None."""
    s = _text(data)
    game_date = str(day) if day is not None else page_date(s)
    rows: list[dict] = []
    for rg_id, body in _ROW.findall(s):
        cells = _CELL.findall(body)
        if len(cells) != 9:
            continue
        pos, name_c, pts, sal, team, opp, _score, mins, stats = cells
        nm = _NAME.search(name_c)
        if not nm:
            continue
        sal_digits = re.sub(r"\D", "", sal)
        opp = opp.replace("&nbsp;", " ").strip()
        stats = html.unescape(stats).replace("\xa0", " ").strip()
        try:
            dk = float(pts)
        except ValueError:
            dk = None
        rows.append({
            "game_date": game_date, "rg_id": rg_id, "name": display_name(nm[1]),
            "pos": None if pos.strip() in ("", "NA") else pos.strip(),
            "team": team_code(team), "opp": team_code(opp[1:] if opp[:1] in "@v" else opp),
            "home": None if opp[:1] not in ("@", "v") else int(opp[0] == "v"),
            "starter": int(nm[2] == "^"),
            "salary": int(sal_digits) if sal_digits else None,
            "dk_pts": dk, "minutes": parse_minutes(mins), "raw_stats": stats or None,
        })
    return rows
