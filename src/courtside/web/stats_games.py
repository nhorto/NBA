"""Games: the scoreboard for a night, plus the box-score hook for /game/<game_id>.

Routes
  /games            the latest night with finals (falls back to today)
  /games/<day>      a given night (YYYY-MM-DD); bad dates -> 404 page

The scoreboard shows one card per game: both teams with their record to date and score (the
winner in bold), FINAL / FINAL/OT or the tip-off time, and the night's top performers. Each
card links to /game/<game_id>. A night with no finals yet (tonight) lists the scheduled games
with tip times; a night without games shows an empty state with a link to the nearest night.

Box-score hook: app.py's `/game/<game_id>` route calls `game_box(store, game_id)` and passes the
result to game.html / preview.html as `box` (None when the box score is not stored yet, e.g.
a preview, in which case the page is unchanged). game.html renders it with _game_box.html.
Queries live in courtside.stats.games.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

from fastapi.responses import HTMLResponse

from ..stats import db as sdb
from ..stats import games as G
from . import fmt
from .stats_common import open_db

latest_final_day = G.latest_final_day


def game_box(store, game_id: str) -> dict | None:
    """Box score view model for /game/<game_id>; None when no player lines are stored."""
    with open_db(store) as conn:
        if conn is None:
            return None
        return sdb.cached(("game-box", game_id), lambda c: G.game_box(c, game_id), conn)


def night_view(conn, day: str) -> dict:
    """View model for one night of the scoreboard."""
    games = G.scoreboard(conn, day)
    prev_day, next_day = G.adjacent_days(conn, day) if conn is not None else (None, None)
    n_final = sum(1 for g in games if g["status"] == "final")
    n_live = sum(1 for g in games if g["status"] == "live")
    for g in games:
        g["ot_label"] = G.ot_label(g["ot"])
    if not games:
        note = "No games on this night."
    elif n_final == len(games):
        note = f"{len(games)} game{'s' if len(games) != 1 else ''} · all final"
    else:
        bits = [f"{len(games)} games"]
        if n_final:
            bits.append(f"{n_final} final")
        if n_live:
            bits.append(f"{n_live} live")
        if len(games) - n_final - n_live:
            bits.append(f"{len(games) - n_final - n_live} to play")
        note = " · ".join(bits)
    return {"games": games, "prev_day": prev_day, "next_day": next_day, "note": note,
            "all_final": bool(games) and n_final == len(games)}


def register(app, *, store, render: Callable, not_found: Callable,
             today: Callable[[], date]) -> None:
    def page(day: str) -> HTMLResponse:
        with open_db(store) as conn:
            vm = night_view(conn, day)
        return render("stats_games.html", page="games", day=day, **vm)

    @app.get("/games", response_class=HTMLResponse)
    def games_latest():
        with open_db(store) as conn:
            day = G.latest_final_day(conn)
        return page(day or today().isoformat())

    @app.get("/games/{day}", response_class=HTMLResponse)
    def games_day(day: str):
        if fmt.parse_day(day) is None or len(day) != 10:
            return not_found("Not a date", "Scoreboard URLs look like /games/2026-10-24.")
        return page(day)
