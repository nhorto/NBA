"""Synthetic day for the props tests (tests/test_proj_props.py, tests/test_web_players.py).

Two games on 2026-10-20; invented players; four projection runs (one that must win per game,
one created after tip-off, a failed one and a late-swap one, both ignored); main props, a
milestone (ignored), a post-tip snapshot (ignored), a nickname, a rostered player with no
projection and a name nobody knows. Nothing touches the network or var/.
"""

from __future__ import annotations

DAY = "2026-10-20"
G1, G2 = "0022600901", "0022600902"
TIP1, TIP2 = "2026-10-20T23:00:00Z", "2026-10-21T01:30:00Z"

# player_id: (team, name, game, {stat: played mean}, min quantiles)
MINQ = (18.0, 21.0, 26.0, 30.0, 34.0, 37.0, 39.0)
PLAYERS = {
    101: ("AAA", "Ann Alder", G1, {"pts": 20.0, "reb": 5.2, "ast": 4.0, "fg3m": 2.1}, MINQ),
    102: ("AAA", "Nicolas Claxton", G1, {"pts": 9.0, "reb": 7.0, "ast": 2.0, "fg3m": 0.1}, MINQ),
    201: ("BBB", "Bo Birch", G1, {"pts": 18.0, "reb": 6.0, "ast": 7.0, "fg3m": 1.5}, MINQ),
    202: ("BBB", "Cy Cedar", G1, {"pts": 4.0, "reb": 2.0, "ast": 1.0, "fg3m": 0.4},
          (2.0, 4.0, 8.0, 12.0, 15.0, 18.0, 20.0)),
    301: ("CCC", "Di Dogwood", G2, {"pts": 15.0, "reb": 4.0, "ast": 3.0, "fg3m": 2.6}, MINQ),
    401: ("DDD", "Ed Elm", G2, {"pts": 12.0, "reb": 9.0, "ast": 2.0, "fg3m": 0.3}, MINQ),
}

# (game, captured_at, athlete id, name, market, kind, line, over, under)
PROPS = [
    (G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Total Points", "ou", 19.5, -115, -105),
    (G1, "2026-10-20T20:00:00Z", "a101", "Ann Alder", "Total Points", "ou", 20.5, -110, -110),
    (G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Total Rebounds", "ou", 5.0, -105, -115),
    (G1, "2026-10-20T14:00:00Z", "a101", "Ann Alder", "Points Milestones", "milestone", 25.0,
     150, None),
    (G1, "2026-10-20T14:00:00Z", "a102", "Nic Claxton", "Total Assists", "ou", 2.5, 120, -150),
    (G1, "2026-10-20T14:00:00Z", "a201", "Bo Birch", "Total Points, Rebounds, and Assists",
     "ou", 30.5, -110, -120),
    (G1, "2026-10-20T14:00:00Z", "a999", "Zed Zeta", "Total Points", "ou", 8.5, -110, -110),
    (G1, "2026-10-20T14:00:00Z", "a998", "Qq Nobody", "Total Points", "ou", 5.5, -110, -110),
    (G1, "2026-10-20T23:10:00Z", "a101", "Ann Alder", "Total Points", "ou", 30.5, -110, -110),
    (G2, "2026-10-20T14:00:00Z", "a301", "Di Dogwood", "Total 3-Point Field Goals", "ou", 2.5,
     130, -160),
    (G2, "2026-10-20T14:00:00Z", "a401", "Ed Elm", "Total Points", "ou", 12.5, -110, -110),
]
N_MAIN = 8          # main props at the newest pre-tip state (G1: 6, G2: 2)
N_JOINED = 6


def _projection(conn, run_id: str, created: str, pid: int, mv: str = "pathA-test",
                scale: float = 1.0) -> None:
    team, _, game, means, mq = PLAYERS[pid]
    cur = conn.execute(
        "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, team, run_id, "
        "created_at, model_version, role, p_play) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (game, DAY, "S1", pid, team, run_id, created, mv, "role", 0.95))
    pj = cur.lastrowid
    for stat, mu in means.items():
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?,?,?,?)", (pj, stat, "played", mu * scale))
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?,?,?,?)", (pj, stat, "all", mu * scale * 0.95))
    conn.execute("INSERT INTO player_projection_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (pj, "min", "played", mq[3], 6.0, *mq))
    dk = means["pts"] + 1.25 * means["reb"] + 1.5 * means["ast"] + 0.5 * means["fg3m"]
    for cond in ("played", "all"):
        conn.execute("INSERT INTO player_projection_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (pj, "dk", cond, dk, 9.0, dk - 15, dk - 11, dk - 6, dk, dk + 6, dk + 11,
                      dk + 15))


def _run(conn, run_id: str, kind: str, created: str, ok: int = 1) -> None:
    conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
                 "VALUES (?,?,?,?,?,?)", (run_id, "S1", kind, created, "pathA-test", ok))


def populate(conn) -> None:
    for gid, tip, home, away in ((G1, TIP1, "BBB", "AAA"), (G2, TIP2, "DDD", "CCC")):
        conn.execute("INSERT INTO games (game_id, nba_day, tipoff_utc, home, away, status) "
                     "VALUES (?,?,?,?,?, 'scheduled')", (gid, DAY, tip, home, away))
    # the run that counts for G1 (and an older one for G2)
    _run(conn, "r_main", "manual", "2026-10-20T15:00:00Z")
    for pid in PLAYERS:
        _projection(conn, "r_main", "2026-10-20T15:00:00Z", pid)
    # ignored for G1: a failed lock45, a swap run, and (for G1) a run after tip-off
    _run(conn, "r_fail", "lock45", "2026-10-20T22:30:00Z", ok=0)
    _projection(conn, "r_fail", "2026-10-20T22:30:00Z", 101, scale=3.0)
    _run(conn, "r_swap", "swap", "2026-10-20T22:40:00Z")
    _projection(conn, "r_swap", "2026-10-20T22:40:00Z", 101, scale=3.0)
    # after G1's tip, before G2's: wins for G2 only
    _run(conn, "r_late", "lock10", "2026-10-21T01:20:00Z")
    for pid in (101, 301, 401):
        _projection(conn, "r_late", "2026-10-21T01:20:00Z", pid, scale=1.1)
    for gid, cap, aid, name, market, kind, line, o, u in PROPS:
        conn.execute(
            "INSERT INTO prop_snapshots (game_id, captured_at, provider, provider_name, "
            "espn_athlete_id, player_name, subject, market_id, market, kind, line, over_odds, "
            "under_odds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (gid, cap, "100", "DraftKings", aid, name, "player", market, market, kind, line, o,
             u))
    team_ids = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
    roster = [(999, "BBB", "Zed Zeta")] + [(pid, v[0], v[1]) for pid, v in PLAYERS.items()]
    for pid, team, name in roster:          # 102's only name is "Nicolas": the prop says "Nic"
        conn.execute("INSERT INTO team_rosters (season, team_id, team, player_id, player) "
                     "VALUES ('2026-27', ?, ?, ?, ?)", (team_ids[team], team, pid, name))
    conn.execute("INSERT INTO injury_snapshots (captured_at, source, team, player_name, "
                 "player_id, status, reason) VALUES ('2026-10-20T17:00:00Z', 'nba_official', "
                 "'BBB', 'Bo Birch', NULL, 'Questionable', 'Injury/Illness - Left Ankle; Sprain')")
    conn.commit()


def finish_g1(conn) -> None:
    """G1 final with a box score: Alder 21 pts / 5 reb (5.0 line: push), Claxton 2 ast, Birch
    DNP (no row), Cedar played."""
    conn.execute("UPDATE games SET status='final' WHERE game_id=?", (G1,))
    for pid, team, mn, pts, reb, ast, fg3m in ((101, "AAA", 34, 21, 5, 3, 2),
                                               (102, "AAA", 28, 8, 9, 2, 0),
                                               (202, "BBB", 12, 4, 1, 0, 0)):
        conn.execute("INSERT INTO player_games (game_id, player_id, team, season, game_date, "
                     "min, pts, reb, ast, fg3m) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (G1, pid, team, "2026-27", DAY, mn, pts, reb, ast, fg3m))
    conn.commit()
