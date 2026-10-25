"""The global search box: players of the last three seasons plus the 30 teams.

`search(conn, q)` -> up to `limit` dicts, best first:
    {"type": "player", "id": 203999, "name": "Nikola Jokić", "team": "DEN",
     "season": "2026-27", "url": "/player/203999", "sub": "DEN"}
    {"type": "team", "id": "DEN", "name": "Denver Nuggets", "team": "DEN",
     "url": "/team/DEN", "sub": "Team"}

Players: anyone in `player_games` in the 3-season window; the name comes from `player_index`
(first + last) when present, else the box-score name; the team is the one of his most recent
game. Matching is accent- and case-insensitive ("jokic" finds Jokić) and ignores dots and
apostrophes ("pj" finds P.J.); initials work too ("sga"). Ranking: an exact team code, then names that start with the
query, then names with a word that starts with each query word, then substring matches; ties
go to the more recent player, then to the name.

The index is built once per database version (`stats.db.cached`).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from datetime import date

from .. import teams as _teams
from . import db as sdb
from . import seasons as S

LIMIT = 10
MAX_Q = 60
_FOLD = str.maketrans({"đ": "d", "ð": "d", "ł": "l", "ø": "o", "æ": "ae", "œ": "oe",
                       "ß": "ss", "ı": "i", "þ": "th"})
_DROP = re.compile(r"[.'’`]")
_NONWORD = re.compile(r"[^a-z0-9]+")


def norm(s: str | None) -> str:
    """'Nikola Jokić' -> 'nikola jokic'; "D'Angelo" -> 'dangelo'; 'P.J.' -> 'pj'."""
    s = unicodedata.normalize("NFKD", (s or "").lower().translate(_FOLD))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _DROP.sub("", s)
    return _NONWORD.sub(" ", s).strip()


def _player_rows(conn: sqlite3.Connection, window: tuple[str, ...]) -> list[dict]:
    marks = ",".join("?" * len(window))
    sql = f"""
        SELECT pg.player_id, pg.player_name, pg.team, pg.season, pg.game_date,
               pi.first_name, pi.last_name
        FROM player_games pg
        JOIN (SELECT player_id, MAX(game_date) AS d FROM player_games
              WHERE season IN ({marks}) GROUP BY player_id) m
          ON m.player_id = pg.player_id AND m.d = pg.game_date
        LEFT JOIN player_index pi ON pi.player_id = pg.player_id
    """
    try:
        rows = conn.execute(sql, window).fetchall()
    except sqlite3.Error:
        try:                                   # no player_index table: box-score names only
            rows = conn.execute(sql.replace(
                "LEFT JOIN player_index pi ON pi.player_id = pg.player_id", "").replace(
                "pi.first_name, pi.last_name", "NULL, NULL"), window).fetchall()
        except sqlite3.Error:
            return []
    out: dict[int, dict] = {}
    for pid, bname, team, season, gdate, first, last in rows:
        full = " ".join(x for x in ((first or "").strip(), (last or "").strip()) if x)
        name = full or (bname or "").strip()
        if not name or pid in out:
            continue
        out[pid] = {"type": "player", "id": int(pid), "name": name, "team": team or "",
                    "season": season, "last_game": gdate, "url": f"/player/{int(pid)}",
                    "sub": team or "", "_keys": {norm(name), norm(bname)} - {""}}
    return list(out.values())


def _team_rows() -> list[dict]:
    out = []
    for t in _teams.TEAMS:
        name = f"{t.city} {t.name}"
        out.append({"type": "team", "id": t.tricode, "name": name, "team": t.tricode,
                    "url": f"/team/{t.tricode}", "sub": "Team",
                    "_keys": {norm(name), norm(t.name), norm(t.tricode)}})
    return out


def build_index(conn: sqlite3.Connection, today: date | None = None) -> list[dict]:
    window = S.seasons(today)
    players = _player_rows(conn, window)
    cur = window[0]
    for p in players:
        if p["season"] != cur:
            p["sub"] = f"{p['team']} · {p['season']}" if p["team"] else p["season"]
    return _team_rows() + players


def _score(entry: dict, q: str, words: list[str]) -> int | None:
    """Lower is better; None = no match."""
    best = None
    for k in entry["_keys"]:
        if entry["type"] == "team" and k == q and len(q) <= 3:
            s = 0                                       # exact tricode: 'den', 'gsw'
        elif k.startswith(q):
            s = 1
        elif all(any(t.startswith(w) for t in k.split()) for w in words):
            s = 2
        elif len(words) == 1 and len(q) >= 2 and "".join(t[0] for t in k.split()) == q:
            s = 2                                       # initials: 'sga', 'kat'
        elif q in k:
            s = 3
        else:
            continue
        best = s if best is None else min(best, s)
    return best


def search(conn: sqlite3.Connection, q: str, *, limit: int = LIMIT,
           today: date | None = None) -> list[dict]:
    """Ranked players and teams for the query `q` (empty list for an empty query)."""
    nq = norm((q or "")[:MAX_Q])
    if not nq:
        return []
    window = S.seasons(today)
    index = sdb.cached(("search-index", window), lambda c: build_index(c, today), conn,
                       domain="box")
    words = nq.split()
    hits = []
    for e in index or ():
        s = _score(e, nq, words)
        if s is not None:
            recent = e.get("last_game") or "9999"            # teams sort with current players
            hits.append((s, _neg(recent), e["name"], e))
    hits.sort(key=lambda h: h[:3])
    return [{k: v for k, v in h[3].items() if not k.startswith("_")} for h in hits[:limit]]


def _neg(day: str) -> tuple:
    """Sort key that puts later ISO dates first."""
    return tuple(-ord(c) for c in day)
