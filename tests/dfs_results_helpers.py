"""Synthetic DraftKings standings + slate builders for the T3.1 / T3.2 tests (no network)."""

from __future__ import annotations

import csv
import io
import json

from courtside.proj.dfs import standings as ST

DAY = "2026-10-20"
SLATE = "9001"
SD_SLATE = "9002"
CONTEST = "196456028"
FNAME_FMT = "contest-standings-{}.csv"
LOCK = "2026-10-20T23:00:00Z"
NAMES = ["Joel Embiid", "LeBron James", "Tyrese Maxey", "Nikola Jokić", "Jaren Jackson Jr.",
         "P.J. Washington", "Luka Dončić", "Karl-Anthony Towns", "De'Aaron Fox", "Nic Claxton",
         "Kelly Oubre Jr.", "Anthony Edwards", "Jalen Brunson", "Bam Adebayo",
         "Cade Cunningham", "Jrue Holiday", "Alperen Şengün", "Derrick White"]
OTHER_SLATE_NAMES = ["Zed Zeta", "Yan Yarrow", "Xan Xerxes", "Wes Willow", "Vic Vale",
                     "Uma Umber", "Tom Thorn", "Sal Sage", "Rex Reed", "Quin Quill",
                     "Pat Pine", "Opal Oak"]
TABLE = [{"min_pos": 1, "max_pos": 1, "payout": 100.0},
         {"min_pos": 2, "max_pos": 3, "payout": 20.0},
         {"min_pos": 4, "max_pos": 5, "payout": 6.0}]
HEADER = ["Rank", "EntryId", "EntryName", "TimeRemaining", "Points", "Lineup", "",
          "Player", "Roster Position", "%Drafted", "FPTS"]


def nba_id(i: int) -> int:
    return 2000 + i


def dk_id(i: int) -> str:
    return str(5000 + i)


def lineup_text(idx: list[int], slots=ST.CLASSIC_SLOTS) -> str:
    return " ".join(f"{s} {NAMES[i]}" for s, i in zip(slots, idx))


def populate(conn, *, contest: bool = True, second_slate: bool = True, runs: bool = True,
             payout: bool = True) -> None:
    """Main slate 9001 (all NAMES), a smaller Early slate of OTHER names, a Showdown slate,
    DK ids mapped to NBA ids, one captured contest with a payout table, one run with a
    stored lineup that equals NAMES[0:8]."""
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
                 "lock_utc) VALUES (?,?,?,?,?,?)", (SLATE, DAY, 70, "main", '["G0","G1"]', LOCK))
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
                 "lock_utc) VALUES (?,?,?,?,?,?)", (SD_SLATE, DAY, 81, "showdown", '["G1"]', LOCK))

    def sal(slate, i, name, slot, salary):
        conn.execute(
            "INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, game_type_id, "
            "draftable_id, player_name, site_player_id, team, positions, roster_slot, salary, "
            "status, fppg, disabled, game_id, game_label, home, away, start_time, starter) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?)",
            ("2026-10-20T20:00:00Z", "dk", slate, "classic", 70 if slate == SLATE else 81,
             f"D{slate}{i}{slot}", name, dk_id(i), "AAA", "PG", slot, salary, "None",
             20.0 + i, "G0", "AAA @ BBB", "BBB", "AAA", "2026-10-20T23:00:00Z", 1))
    for i, name in enumerate(NAMES):
        sal(SLATE, i, name, "UTIL", 3000 + 300 * i)
        conn.execute("INSERT INTO player_ids (source, source_id, player_id, name, team, how) "
                     "VALUES ('dk', ?, ?, ?, 'AAA', 'exact')", (dk_id(i), nba_id(i), name))
    for j, name in enumerate(OTHER_SLATE_NAMES):
        i = 100 + j
        conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                     "game_type_id, draftable_id, player_name, site_player_id, team, positions, "
                     "roster_slot, salary, status, fppg, disabled, game_id, start_time) VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
                     ("2026-10-20T20:00:00Z", "dk", "9003", "classic", 70, f"D3{i}", name, dk_id(i),
                      "CCC", "PG", "UTIL", 3500, "None", 15.0, "G9", "2026-10-20T22:00:00Z"))
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json, "
                 "lock_utc) VALUES ('9003', ?, 70, 'early', '[\"G9\"]', ?)", (DAY, LOCK))
    if second_slate:     # Showdown pool: 6 of the names, each as CPT and UTIL
        for i in range(6):
            sal(SD_SLATE, i, NAMES[i], "CPT", 6000)
            sal(SD_SLATE, i, NAMES[i], "UTIL", 4000)
    if contest:
        conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, entry_fee, field_size, "
                     "payout_json, captured_at) VALUES (?,?,?,?,?,?,?)",
                     (CONTEST, SLATE, "NBA $20K Test", 3.0, 10,
                      json.dumps(TABLE) if payout else None, "2026-10-20T12:00:00Z"))
    if runs:
        conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
                     "VALUES ('run1', ?, 'lock10', '2026-10-20T22:50:00Z', 'pathA-v0.1', 1)", (SLATE,))
        slots = [{"slot": s, "player_id": dk_id(i), "name": NAMES[i], "team": "AAA",
                  "salary": 5000} for s, i in zip(ST.CLASSIC_SLOTS, range(8))]
        conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean) "
                     "VALUES ('run1','cash',1,?,49000,300)", (json.dumps(slots),))
    conn.commit()


def standings_csv(entries, players=(), *, header: list[str] | None = None, eol="\r\n") -> bytes:
    """entries: (rank, entry_id, name, time_remaining, points, lineup); players: (name, position,
    pct, fpts). The two blocks share rows like DK's file."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol)
    w.writerow(header or HEADER)
    for i in range(max(len(entries), len(players))):
        e = list(entries[i]) if i < len(entries) else [""] * 6
        p = list(players[i]) if i < len(players) else [""] * 4
        w.writerow(e + [""] + p)
    return buf.getvalue().encode("utf-8")


def ten_entries(owner: str = "nhorto", owner_rank: int = 4):
    """10 entries, ranks 1..10, scores 300..210; the owner holds `owner_rank` with the stored
    lineup (NAMES[0:8])."""
    others = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10],
              [4, 5, 6, 7, 8, 9, 10, 11], [5, 6, 7, 8, 9, 10, 11, 12], [6, 7, 8, 9, 10, 11, 12, 13]]
    out = []
    for r in range(1, 11):
        idx = list(range(8)) if r == owner_rank else others[(r - 1) % len(others)]
        name = f"{owner} (2/20)" if r == owner_rank else f"user{r:02d}"
        out.append((r, 4600000000 + r, name, 0, 300 - 10 * (r - 1), lineup_text(idx)))
    return out


def player_block(pcts=None):
    pcts = pcts or {i: 60 - 3 * i for i in range(len(NAMES))}
    return [(NAMES[i], "PG/SG", f"{pcts[i]:.2f}%", f"{30 + i:.1f}") for i in range(len(NAMES))]
