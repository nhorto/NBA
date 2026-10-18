"""ESPN site API (site.api.espn.com): scoreboard (with odds), summary, injuries.

Pure fetch + parse. No archiving and no DB access; callers archive the raw payloads.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from .. import http, teams
from ..config import nba_day

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

SOURCE = "espn"


# --------------------------------------------------------------------------- fetch

def fetch_scoreboard(day: date) -> dict:
    return http.fetch_json(f"{BASE}/scoreboard", params={"dates": day.strftime("%Y%m%d")})


def fetch_summary(espn_id: str | int) -> dict:
    return http.fetch_json(f"{BASE}/summary", params={"event": str(espn_id)})


def fetch_injuries() -> dict:
    return http.fetch_json(f"{BASE}/injuries")


# --------------------------------------------------------------------------- helpers

def _tricode(abbr: str | None) -> str | None:
    if not abbr:
        return None
    try:
        return teams.espn_to_tricode(abbr)
    except KeyError:
        return None


def _parse_dt(s: str) -> datetime:
    """ESPN dates look like '2026-10-20T19:00Z' (no seconds)."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _status(status: dict | None) -> str:
    t = (status or {}).get("type") or {}
    name = (t.get("name") or "").upper()
    if name in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"):
        return "postponed"
    state = t.get("state")
    if state == "pre":
        return "scheduled"
    if state == "in":
        return "live"
    if state == "post" or t.get("completed"):
        return "final"
    return "scheduled"


def _sides(comp: dict) -> tuple[dict, dict] | None:
    home = away = None
    for c in comp.get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    if home is None or away is None:
        return None
    return home, away


def _event_teams(event: dict) -> tuple[dict, dict, dict, str, str] | None:
    """(competition, home competitor, away competitor, home tricode, away tricode)."""
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    sides = _sides(comp)
    if sides is None:
        return None
    home_c, away_c = sides
    home = _tricode((home_c.get("team") or {}).get("abbreviation"))
    away = _tricode((away_c.get("team") or {}).get("abbreviation"))
    if home is None or away is None:
        return None
    return comp, home_c, away_c, home, away


def _score(c: dict) -> int | None:
    s = c.get("score")
    if s in (None, ""):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- scoreboard

def scoreboard_events(payload: dict) -> list[dict]:
    """Normalized games. Events whose teams do not map to NBA teams are skipped (warned)."""
    out: list[dict] = []
    for ev in payload.get("events") or []:
        parsed = _event_teams(ev)
        if parsed is None:
            log.warning("espn: skipping event %s (%s): teams do not map to NBA teams",
                        ev.get("id"), ev.get("shortName"))
            continue
        comp, home_c, away_c, home, away = parsed
        tip = _parse_dt(ev.get("date") or comp["date"])
        status = _status(ev.get("status") or comp.get("status"))
        hs, as_ = _score(home_c), _score(away_c)
        if status in ("scheduled", "postponed"):
            hs = as_ = None
        out.append({
            "espn_id": str(ev["id"]),
            "nba_day": nba_day(tip).isoformat(),
            "tipoff_utc": _iso_z(tip),
            "home": home,
            "away": away,
            "status": status,
            "home_score": hs,
            "away_score": as_,
        })
    return out


def _num(v: Any) -> float | None:
    """'+1.5' / '-108' / 222.5 / 'o222.5' -> float; 'EVEN'/'PK' -> 100/0; else None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("even", "ev"):
        return 100.0
    if s in ("pk", "pick", "pickem"):
        return 0.0
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _int(v: Any) -> int | None:
    n = _num(v)
    return None if n is None else int(round(n))


def _dig(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _home_spread(o: dict, home: str, away: str) -> float | None:
    """Spread from the HOME team's perspective (negative = home favored).

    Sources in order of trust:
      1. pointSpread.home.close.line  ("+1.5" for a home underdog)   [newer shape]
      2. magnitude from `spread` + sign from the favorite flags      [`spread` is
         usually already home-signed, but flags make the sign unambiguous]
      3. raw `spread` as home-signed
      4. `details` text ("BOS -1.5": team code + number from that team's side)
    """
    v = _num(_dig(o, "pointSpread", "home", "close", "line"))
    if v is not None:
        return v
    sp = _num(o.get("spread"))
    if sp is not None:
        home_fav = _dig(o, "homeTeamOdds", "favorite")
        away_fav = _dig(o, "awayTeamOdds", "favorite")
        if home_fav is True and away_fav is not True:
            return -abs(sp)
        if away_fav is True and home_fav is not True:
            return abs(sp)
        return sp
    details = o.get("details") or ""
    m = re.match(r"\s*([A-Za-z]+)\s+([-+]?\d+(?:\.\d+)?)\s*$", details)
    if m:
        team_abbr, n = m.group(1).upper(), float(m.group(2))
        tri = _tricode(team_abbr) or team_abbr
        if tri == home:
            return n
        if tri == away:
            return -n
    if details.strip().upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    return None


def _total(o: dict) -> float | None:
    v = _num(o.get("overUnder"))
    if v is None:
        v = _num(_dig(o, "total", "over", "close", "line"))
    return v


def _moneyline(o: dict, side: str) -> int | None:
    v = _int(_dig(o, "moneyline", side, "close", "odds"))
    if v is None:
        v = _int(_dig(o, f"{side}TeamOdds", "moneyLine"))
    return v


def scoreboard_odds(payload: dict) -> list[dict]:
    """One row per event that has odds, from competitions[0].odds[0]."""
    out: list[dict] = []
    for ev in payload.get("events") or []:
        parsed = _event_teams(ev)
        if parsed is None:
            continue  # already warned about in scoreboard_events
        comp, _h, _a, home, away = parsed
        odds_list = comp.get("odds") or []
        if not odds_list or not isinstance(odds_list[0], dict):
            continue
        o = odds_list[0]
        row = {
            "espn_id": str(ev["id"]),
            "home": home,
            "away": away,
            "source": SOURCE,
            "provider": _dig(o, "provider", "name"),
            "home_spread": _home_spread(o, home, away),
            "total": _total(o),
            "home_ml": _moneyline(o, "home"),
            "away_ml": _moneyline(o, "away"),
        }
        if all(row[k] is None for k in ("home_spread", "total", "home_ml", "away_ml")):
            continue
        out.append(row)
    return out


# --------------------------------------------------------------------------- summary

_SUMMARY_KEYS = ("boxscore", "plays", "winprobability", "injuries", "pickcenter", "odds")


def summary_parts(payload: dict) -> dict:
    """Which parts of an ESPN summary payload are present and non-empty (booleans)."""
    out = {k: bool(payload.get(k)) for k in _SUMMARY_KEYS}
    box = payload.get("boxscore") or {}
    out["boxscore"] = bool(box.get("players") or box.get("teams"))
    return out


# --------------------------------------------------------------------------- injuries

_ATHLETE_ID = re.compile(r"/id/(\d+)")


def _athlete_id(ath: dict) -> str | None:
    if ath.get("id"):
        return str(ath["id"])
    for link in ath.get("links") or []:
        m = _ATHLETE_ID.search(link.get("href") or "")
        if m:
            return m.group(1)
    return None


def _reason(inj: dict) -> str:
    """Compact: 'Knee, Right: Dislocated; ret 2026-11-02 | <short comment>'."""
    d = inj.get("details") or {}
    head = d.get("type") or ""
    if d.get("side") and head:
        head = f"{head}, {d['side']}"
    if d.get("detail"):
        head = f"{head}: {d['detail']}" if head else d["detail"]
    if d.get("returnDate"):
        head = f"{head}; ret {d['returnDate']}" if head else f"ret {d['returnDate']}"
    comment = (inj.get("shortComment") or "").strip()
    if len(comment) > 160:
        comment = comment[:157].rstrip() + "..."
    if head and comment:
        return f"{head} | {comment}"
    return head or comment


def injury_rows(payload: dict) -> list[dict]:
    out: list[dict] = []
    for team_block in payload.get("injuries") or []:
        for inj in team_block.get("injuries") or []:
            ath = inj.get("athlete") or {}
            abbr = (ath.get("team") or {}).get("abbreviation") or team_block.get("abbreviation")
            tri = _tricode(abbr)
            if tri is None:
                log.warning("espn injuries: skipping %s (team %r does not map)",
                            ath.get("displayName"), abbr)
                continue
            aid = _athlete_id(ath)
            out.append({
                "team": tri,
                "player_name": ath.get("displayName")
                or " ".join(filter(None, [ath.get("firstName"), ath.get("lastName")])),
                "player_id": f"espn:{aid}" if aid else None,
                "status": inj.get("status") or "",
                "reason": _reason(inj),
                "source": SOURCE,
            })
    return out
