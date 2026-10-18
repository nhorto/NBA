"""Tiny synthetic leagues in the real schema, for tests (no network, no real data).

``make_league(conn, ...)`` creates the migration-2 tables (``team_games``, ``player_games``, and
the two ``_adv`` tables) and fills them with a round-robin league: one game per team per day,
9 players per team whose minutes sum to 240 and whose points sum to the team score.

Hooks let a test script specific stories:
* ``margin_fn(season, day, home, away, margin) -> margin`` to force results;
* ``player_fn(season, day, team, slot, row) -> row | None`` to edit a line or drop it (DNP).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date, timedelta

import numpy as np

MINUTES = [36, 34, 32, 30, 28, 26, 22, 18, 14]            # sums to 240
SHARES = np.array([0.24, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03])
POSS = 100.0


def schema(conn: sqlite3.Connection) -> None:
    from courtside import db
    conn.executescript(db.MIGRATIONS[1])


def _split(total: int, shares: np.ndarray) -> list[int]:
    raw = shares / shares.sum() * total
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[: total - out.sum()]:
        out[i] += 1
    return out.tolist()


def player_id(team: str, slot: int) -> int:
    return (sum(ord(c) for c in team) * 7 + ord(team[0])) * 100 + slot


def make_league(conn: sqlite3.Connection, seasons: dict[str, int] | None = None,
                codes: tuple[str, ...] = ("BOS", "NYK", "LAL", "GSW", "DEN", "MIA"),
                strengths: dict[str, float] | None = None, seed: int = 0,
                margin_fn: Callable | None = None, player_fn: Callable | None = None,
                noise: float = 10.0) -> None:
    """``seasons`` maps season label -> number of game days (= games per team)."""
    seasons = seasons or {"2023-24": 30, "2024-25": 30}
    strengths = strengths or {c: s for c, s in zip(codes, np.linspace(-5, 5, len(codes)))}
    rng = np.random.default_rng(seed)
    schema(conn)
    gid = 0
    for season, days in seasons.items():
        y = int(season[:4])
        d0 = date(y, 10, 22)
        teams = list(codes)
        n = len(teams)
        for day in range(days):
            gday = d0 + timedelta(days=day)
            pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]
            for k, (a, b) in enumerate(pairs):
                h, aw = (a, b) if (day + k) % 2 == 0 else (b, a)
                m = strengths[h] - strengths[aw] + 2.0 + rng.normal(0, noise)
                if margin_fn is not None:
                    m = margin_fn(season, day, h, aw, m)
                m = int(round(m)) or 1
                gid += 1
                game_id = f"002{y % 100:02d}{gid:05d}"
                hp = 110 + (m + 1) // 2
                ap = hp - m
                for team, opp, is_home, pts, opts in ((h, aw, 1, hp, ap), (aw, h, 0, ap, hp)):
                    _insert_team(conn, game_id, season, gday, team, opp, is_home, pts, opts)
                    _insert_players(conn, game_id, season, gday, day, team, pts, pts - opts,
                                    player_fn)
    conn.commit()


def _insert_team(conn, game_id, season, gday, team, opp, is_home, pts, opts):
    fga = 88
    conn.execute(
        "INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, is_home, wl,"
        " min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, pts,"
        " plus_minus) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (game_id, team, season, "Regular Season", gday.isoformat(), opp, is_home,
         "W" if pts > opts else "L", 240.0, 40, fga, 12, 34, 18, 23, 10, 34, 44, 25, 8, 5, 14,
         20, pts, float(pts - opts)))
    conn.execute(
        "INSERT INTO team_games_adv (game_id, team, off_rating, def_rating, net_rating, pace,"
        " poss) VALUES (?,?,?,?,?,?,?)",
        (game_id, team, 100 * pts / POSS, 100 * opts / POSS, 100 * (pts - opts) / POSS, POSS,
         POSS))


def _insert_players(conn, game_id, season, gday, day, team, pts, margin, player_fn):
    pts_split = _split(pts, SHARES)
    for slot, (mins, p) in enumerate(zip(MINUTES, pts_split)):
        fta = round(p * 0.25)
        ftm = round(fta * 0.8)
        fg3m = round(p * 0.1)
        fgm = max((p - ftm - fg3m) // 2, 0)
        row = {"game_id": game_id, "player_id": player_id(team, slot),
               "player_name": f"{team} Player {slot}", "team": team, "season": season,
               "season_type": "Regular Season", "game_date": gday.isoformat(), "min": float(mins),
               "fgm": fgm, "fga": max(round(fgm / 0.47), fgm), "fg3m": fg3m, "fg3a": fg3m * 3,
               "ftm": ftm, "fta": fta, "oreb": 1, "dreb": 4 if slot < 5 else 2,
               "reb": 5 if slot < 5 else 3, "ast": 5 if slot == 0 else 2, "stl": 1, "blk": 0,
               "tov": 2, "pf": 2, "pts": p, "plus_minus": float(round(margin * mins / 48)),
               "usg_pct": round(0.32 - 0.02 * slot, 3)}
        if player_fn is not None:
            row = player_fn(season, day, team, slot, row)
            if row is None:
                continue
        usg = row.pop("usg_pct")
        cols = list(row)
        conn.execute(f"INSERT INTO player_games ({','.join(cols)}) VALUES "
                     f"({','.join('?' * len(cols))})", [row[c] for c in cols])
        conn.execute("INSERT INTO player_games_adv (game_id, player_id, usg_pct, poss) "
                     "VALUES (?,?,?,?)", (game_id, row["player_id"], usg, POSS * row["min"] / 48))


def day_date(season: str, day: int) -> date:
    return date(int(season[:4]), 10, 22) + timedelta(days=day)
