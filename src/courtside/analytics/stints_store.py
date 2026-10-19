"""Stints store: persist reconstructed lineups and per-player on-court tallies (docs/PLAYERS.md T0.4).

Input is the raw cdn.nba.com archive (``raw/nba_cdn/playbyplay`` + ``raw/nba_cdn/boxscore``, one
file per game under an NBA-day directory). Each game runs through the existing reconstruction
(``pbp.normalize`` -> ``stints.build_rotation``) and is only stored when it reconciles with the
boxscore (``stints.validate_stints``). Two tiers, both reported:

  strict     every player's on-court seconds within 1 s of the boxscore minutes and on-court
             plus-minus exactly equal to ``plusMinusPoints`` (the spec; the 2025-26 fixtures pass
             it at 0.0 s). Only ~80% of 2024-25 games are strict: the boxscore minutes of older
             seasons carry a few seconds of scorer noise (and sometimes a 10-14 s swap between two
             players at a sub), and about 1.5% of games have a 1-2 point plus-minus credit that
             sits on a neighbouring lineup at a sub / free-throw boundary.
  accepted   every player within ``MINUTES_TOL_S`` (30 s) of the boxscore minutes and within
             ``PM_TOL`` (2) points of plusMinusPoints, and five players per side in every stint.
             Stored. The play-by-play clock, not the boxscore minutes, is the better clock, and a
             one- or two-point misallocation does not matter for with/without factors.

Games that are not accepted are not stored (Phase 2 with/without factors must not eat bad lineups); they are listed in the summary and
in the ``stints_store`` data check, and are retried on the next build.

Tables (migration 9)
--------------------
``stints``  one row per stint (a stretch with no substitution, cut at period ends)
    stint_no 0-based, game order; period; start_sec/end_sec elapsed game seconds (regulation
    periods 720 s, overtimes 300 s); home5/away5 = JSON list of the five player ids, sorted
    (``"[1,2,3,4,5]"``); home_pts/away_pts = points credited to the stint (free throws go to the
    lineup on the floor at the foul, as the NBA's own plus-minus does); home_poss/away_poss =
    offensive possessions *ended* in the stint (counting method of ``pbp._mark_possessions``,
    docs/ANALYTICS.md section 1: made FG, final made FT, defensive rebound, turnover, period end).

``player_game_oncourt``  one row per player who was on the floor at any time
    started            boxscore starter flag
    sec_q13/sec_q4/sec_ot   on-court seconds in periods 1-3 / period 4 / overtimes. Their sum is
                       the boxscore minutes (+-1 s, validated)
    sec_garbage        on-court seconds in garbage time, the existing Cleaning-the-Glass notion of
                       ``game_story.garbage_time``: 4th quarter only, score margin >= 25 with
                       12:00-9:01 left, >= 20 with 9:00-6:01 left, >= 10 with 6:00 or less, and at
                       most 2 of the game's 10 starters on the floor. Evaluated continuously, so
                       it stops if the margin shrinks or starters return
    on_poss/opp_on_poss     possessions ended by his team's offense / by the opponent's offense
                       while he was on the floor
    team_fga_on/team_fgm_on/team_misses_on   his team's field-goal attempts / makes / misses on
                       the floor (free throws excluded; misses = FGA - FGM; heaves are not
                       attempts, as in the boxscore)
    opp_misses_on      opponent field-goal misses while he was on the floor
    tm_fgm_on          teammates' made field goals while he was on the floor (team FGM minus his
                       own; the denominator of assist rate)
    opp_tov_on         opponent turnovers while he was on the floor (steal-rate denominator)
    opp_2pa_on         opponent two-point attempts while he was on the floor (block-rate
                       denominator)
    Field-goal events are assigned to the stint that contains them (``Stint.first_idx`` ..
    ``last_idx``); a non-scoring event that falls into a dropped zero-length stint goes to the
    preceding stint.

Public API
  build(conn, seasons=None, *, game_ids=None, force=False, workers=1, limit=None,
        minutes_tol=30.0, pm_tol=2) -> summary dict
  compute_game(box, pbp) -> GameResult                         (pure; no DB)
  list_archived_games(seasons=None, game_ids=None) -> [(game_id, day, pbp_path, box_path)]
  stored_games(conn) -> set[str]
  load_stints(conn, game_id) -> list[dict]                     (home5/away5 decoded to lists)
  add_commands(sub) -> {"stints": handler}                     (``courtside proj stints``)
"""

from __future__ import annotations

import json
import sqlite3
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from courtside import archive, config
from courtside.analytics.game_story import garbage_time
from courtside.analytics.pbp import AWAY, HOME, REG_PERIODS, Event, normalize
from courtside.analytics.stints import (
    SIDES, Stint, ValidationReport, box_players, build_rotation, validate_stints,
)

SOURCE = "nba_cdn"
CHECK_NAME = "stints_store"
MINUTES_TOL_S = 30.0                   # accepted tier: per-player seconds vs boxscore minutes
PM_TOL = 2                             # accepted tier: per-player plus-minus vs boxscore
NOT_FINAL = {"1", "2", "inprogress", "periodbreak", "pregame"}   # boxscore gameStatus values
CHECK_WARN, CHECK_FAIL = 0.99, 0.95    # accepted fraction of the games attempted in a run
BATCH = 50                             # games per commit
FAILURES_LISTED = 25                   # game ids put in the data-check detail

STINT_COLS = ("game_id", "stint_no", "period", "start_sec", "end_sec", "home5", "away5",
              "home_pts", "away_pts", "home_poss", "away_poss")
PGO_COLS = ("game_id", "player_id", "team", "started", "sec_q13", "sec_q4", "sec_ot",
            "sec_garbage", "on_poss", "opp_on_poss", "team_fga_on", "team_fgm_on",
            "team_misses_on", "opp_misses_on", "tm_fgm_on", "opp_tov_on", "opp_2pa_on")


# ------------------------------------------------------------------------- pure computation

@dataclass
class GameResult:
    game_id: str
    ok: bool
    causes: list[str] = field(default_factory=list)   # empty when ok
    detail: str = ""
    stints: list[tuple] = field(default_factory=list)
    players: list[tuple] = field(default_factory=list)
    anomalies: int = 0
    strict: bool = False              # also passes the strict tier


def season_of(game_id: str) -> str:
    """'0022400001' -> '2024-25' (digits 3-4 of an NBA game id are the season's start year)."""
    yy = int(game_id[3:5])
    return f"{2000 + yy}-{(yy + 1) % 100:02d}"


def _place(stints: Sequence[Stint], firsts: list[int], idx: int) -> int | None:
    """Index of the stint whose event range holds ``idx``; falls back to the preceding stint."""
    k = bisect_right(firsts, idx) - 1
    if k < 0:
        return 0 if stints else None
    return k


def garbage_seconds(events: Sequence[Event], stints: Sequence[Stint],
                    starters: Iterable[int]) -> list[float]:
    """Garbage-time seconds of every stint (definition in the module docstring)."""
    span = garbage_time(events, stints, starters)
    return [span.overlap(st.start, st.end) if st.duration > 0 else 0.0 for st in stints]


def tally_game(events: Sequence[Event], stints: Sequence[Stint], box: dict) -> list[tuple]:
    """``player_game_oncourt`` rows (without game_id) for one game."""
    players = box_players(box)
    starters = [pid for pid, p in players.items() if p["starter"]]
    firsts = [s.first_idx for s in stints]
    # per stint: [fga, fgm, 2pa, tov] per side
    t = [{HOME: [0, 0, 0, 0], AWAY: [0, 0, 0, 0]} for _ in stints]
    own_fgm: dict[int, int] = {}
    for e in events:
        if e.side not in SIDES:
            continue
        if e.is_fga:
            k = _place(stints, firsts, e.idx)
            if k is None:
                continue
            c = t[k][e.side]
            c[0] += 1
            c[1] += bool(e.made)
            c[2] += not e.is_three
            if e.made and e.player_id:
                own_fgm[e.player_id] = own_fgm.get(e.player_id, 0) + 1
        elif e.is_turnover:
            k = _place(stints, firsts, e.idx)
            if k is not None:
                t[k][e.side][3] += 1
    garb = garbage_seconds(events, stints, starters)

    acc: dict[int, dict] = {}
    for k, st in enumerate(stints):
        dur = st.duration
        for side in SIDES:
            opp = AWAY if side == HOME else HOME
            mine, theirs = t[k][side], t[k][opp]
            for pid in st.lineup(side):
                a = acc.setdefault(pid, {
                    "side": side, "q13": 0.0, "q4": 0.0, "ot": 0.0, "garb": 0.0, "poss": 0,
                    "opp_poss": 0, "fga": 0, "fgm": 0, "opp_miss": 0, "opp_tov": 0, "opp_2pa": 0})
                if st.period < REG_PERIODS:
                    a["q13"] += dur
                elif st.period == REG_PERIODS:
                    a["q4"] += dur
                else:
                    a["ot"] += dur
                a["garb"] += garb[k]
                a["poss"] += st.poss(side)
                a["opp_poss"] += st.poss(opp)
                a["fga"] += mine[0]
                a["fgm"] += mine[1]
                a["opp_miss"] += theirs[0] - theirs[1]
                a["opp_tov"] += theirs[3]
                a["opp_2pa"] += theirs[2]
    rows = []
    for pid in sorted(acc):
        a = acc[pid]
        info = players.get(pid, {})
        rows.append((
            pid, info.get("team"), int(bool(info.get("starter"))),
            round(a["q13"], 2), round(a["q4"], 2), round(a["ot"], 2), round(a["garb"], 2),
            float(a["poss"]), float(a["opp_poss"]), a["fga"], a["fgm"], a["fga"] - a["fgm"],
            a["opp_miss"], max(0, a["fgm"] - own_fgm.get(pid, 0)), a["opp_tov"], a["opp_2pa"]))
    return rows


def _classify(report: ValidationReport, minutes_tol: float, pm_tol: int) -> tuple[list[str], str]:
    """Causes (empty = accepted) and a detail string for the accepted tier."""
    causes, parts = [], []
    bad_min = [c for c in report.checks if abs(c.seconds_diff) > minutes_tol]
    # Some 2019-20 boxscores ('finalbox', Feb-Mar 2020) carry plusMinusPoints = 0 for everybody:
    # no reference, so only minutes (and the team score, checked in compute_game) are validated.
    pm_known = any(c.box_plus_minus for c in report.checks)
    bad_pm = [c for c in report.checks
              if pm_known and abs(c.plus_minus - c.box_plus_minus) > pm_tol]
    if bad_min:
        causes.append("minutes")
        c = max(bad_min, key=lambda c: abs(c.seconds_diff))
        parts.append(f"{len(bad_min)} minutes mismatches > {minutes_tol:.0f}s (worst {c.name} "
                     f"{c.seconds_diff:+.1f}s)")
    if bad_pm:
        causes.append("plus_minus")
        c = max(bad_pm, key=lambda c: abs(c.plus_minus - c.box_plus_minus))
        parts.append(f"{len(bad_pm)} plus-minus mismatches > {pm_tol} (worst {c.name} "
                     f"{c.box_plus_minus} vs {c.plus_minus})")
    return causes, "; ".join(parts)


def _norm(name: str | None) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _in_player_by_name(a: dict, box: dict | None) -> int | None:
    """Incoming player of a 'SUB: <in> FOR <out>' action, matched by family name on the team."""
    desc = str(a.get("description") or "")
    if not desc.startswith("SUB:") or " FOR " not in desc or not box:
        return None
    name = _norm(desc[4:].split(" FOR ", 1)[0])
    g = box.get("game") or {}
    for key in ("homeTeam", "awayTeam"):
        t = g.get(key) or {}
        if t.get("teamId") != a.get("teamId"):
            continue
        hits = [int(p["personId"]) for p in t.get("players") or []
                if int(p["personId"]) != a.get("personId")
                and name in (_norm(p.get("familyName")), _norm(p.get("nameI")), _norm(p.get("name")))]
        return hits[0] if len(hits) == 1 else None
    return None


def repair_legacy_pbp(pbp: dict, box: dict | None = None) -> dict:
    """Make two older play-by-play dialects readable by ``pbp.normalize`` (returns a shallow copy;
    modern payloads are returned unchanged). Proper home for both fixes: ``pbp.normalize``.

    * One-sided substitutions (about 250 games, Oct 2019 - Mar 2020): a substitution is ONE ``out``
      action naming the incoming player in ``subsInPersonId``, with no ``in`` action. The missing
      ``in`` action is added (``orderNumber`` + 1, same clock, ``personId`` = incoming player).
    * 'SUB: <in> FOR <out>' substitutions (about 10 bubble games, Aug 2020): one ``out`` action
      with only the outgoing ``personId``; the incoming player is matched by family name among the
      team's boxscore players (ambiguous names stay unresolved and the game fails validation).
    * ``shotResult`` written as ``": Made"`` / ``": Missed"`` (about 70 games, Feb - Mar 2020,
      boxscore status 'finalbox'): the leading ``": "`` is stripped, otherwise no shot scores.
    """
    game = (pbp or {}).get("game") or {}
    actions = game.get("actions") or []
    ins = {(a.get("period"), a.get("clock"), a.get("personId")) for a in actions
           if a.get("actionType") == "substitution" and a.get("subType") == "in"}
    out, changed = [], False
    for a in actions:
        r = a.get("shotResult")
        if isinstance(r, str) and r.startswith(":"):
            a = {**a, "shotResult": r.lstrip(": ")}
            changed = True
        out.append(a)
        pid = a.get("subsInPersonId")
        if not pid and a.get("actionType") == "substitution" and a.get("subType") == "out":
            pid = _in_player_by_name(a, box)
        if (a.get("actionType") == "substitution" and a.get("subType") == "out" and pid
                and (a.get("period"), a.get("clock"), pid) not in ins):
            out.append({**a, "subType": "in", "personId": pid,
                        "orderNumber": int(a.get("orderNumber") or 0) + 1,
                        "description": f"SUB in: {a.get('subsInPlayerName') or pid}",
                        "subsInPersonId": pid,
                        "personIdsFilter": [pid]})
            changed = True
    if not changed:
        return pbp
    return {**pbp, "game": {**game, "actions": out}}


def compute_game(box: dict, pbp: dict, game_id: str = "", *, minutes_tol: float = MINUTES_TOL_S,
                 pm_tol: int = PM_TOL) -> GameResult:
    """Reconstruct, validate and tally one game. Never raises: failures become causes."""
    gid = game_id or str(((box or {}).get("game") or {}).get("gameId") or "")
    try:
        g = (box or {}).get("game") or {}
        status = str(g.get("gameStatus") or "").lower()
        if status in NOT_FINAL:   # older archives carry 'finalbox' / None for finished games
            return GameResult(gid, False, ["not_final"], f"gameStatus {g.get('gameStatus')}")
        if not ((pbp or {}).get("game") or {}).get("actions"):
            return GameResult(gid, False, ["no_actions"], "empty play-by-play")
        if not any((q.get("statistics") or {}).get("minutes")
                   for t in ("homeTeam", "awayTeam") for q in (g.get(t) or {}).get("players") or []):
            return GameResult(gid, False, ["stub_boxscore"], "boxscore has no player minutes")
        events = normalize(repair_legacy_pbp(pbp, box), box)
        rot = build_rotation(events, box)
        stints = rot.stints
        if not stints:
            return GameResult(gid, False, ["no_stints"], "no stints")
        if any(len(s.home) != 5 or len(s.away) != 5 for s in stints):
            bad = sum(len(s.home) != 5 or len(s.away) != 5 for s in stints)
            return GameResult(gid, False, ["lineup_size"], f"{bad} stints without 5 per side",
                              anomalies=len(rot.anomalies))
        report = validate_stints(box, stints, rot.anomalies)
        causes, detail = _classify(report, minutes_tol, pm_tol)
        g = box["game"]
        if (sum(s.home_pts for s in stints), sum(s.away_pts for s in stints)) != (
                g["homeTeam"].get("score"), g["awayTeam"].get("score")) \
                and g["homeTeam"].get("score") is not None:
            causes.append("score")
            detail += "; stint points differ from the final score"
        if causes:
            if rot.anomalies:
                detail += f"; {len(rot.anomalies)} anomalies, first: {rot.anomalies[0]}"
            return GameResult(gid, False, causes, detail, anomalies=len(rot.anomalies))
        stint_rows = [
            (n, s.period, s.start, s.end, json.dumps(list(s.home), separators=(",", ":")),
             json.dumps(list(s.away), separators=(",", ":")), s.home_pts, s.away_pts,
             float(s.home_poss), float(s.away_poss))
            for n, s in enumerate(stints)]
        return GameResult(gid, True, [], "", stint_rows, tally_game(events, stints, box),
                          anomalies=len(rot.anomalies),
                          strict=report.ok and any(p.box_plus_minus for p in report.checks))
    except Exception as e:   # noqa: BLE001 - a bad payload must not stop a 8,000-game run
        return GameResult(gid, False, ["exception"], f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------------- archive scan

def list_archived_games(seasons: Iterable[str] | str | None = None,
                        game_ids: Iterable[str] | None = None
                        ) -> list[tuple[str, str, Path, Path]]:
    """Games with both a play-by-play and a boxscore archived, sorted by (day, game_id)."""
    if isinstance(seasons, str):
        seasons = [seasons]
    want_seasons = set(seasons) if seasons else None
    want_ids = set(game_ids) if game_ids else None
    root = config.data_dir() / "raw" / SOURCE
    out = []
    pbp_root, box_root = root / "playbyplay", root / "boxscore"
    if not pbp_root.is_dir() or not box_root.is_dir():
        return out
    for dday in sorted(pbp_root.iterdir()):
        bday = box_root / dday.name
        if not dday.is_dir() or not bday.is_dir():
            continue
        for f in sorted(dday.glob("*.json.gz")):
            gid = f.name[:-len(".json.gz")]
            if len(gid) != 10 or not gid.isdigit():
                continue
            if want_ids is not None and gid not in want_ids:
                continue
            if want_seasons is not None and season_of(gid) not in want_seasons:
                continue
            b = bday / f.name
            if b.exists():
                out.append((gid, dday.name, f, b))
    return out


def stored_games(conn: sqlite3.Connection) -> set[str]:
    """Games present in both tables."""
    a = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM stints")}
    b = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM player_game_oncourt")}
    return a & b


def load_stints(conn: sqlite3.Connection, game_id: str) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM stints WHERE game_id=? ORDER BY stint_no", (game_id,)):
        d = dict(r)
        d["home5"], d["away5"] = json.loads(d["home5"]), json.loads(d["away5"])
        out.append(d)
    return out


# ------------------------------------------------------------------------- build

def _work(task: tuple[str, str, str, str, float, int]) -> GameResult:
    gid, _day, pbp_path, box_path, minutes_tol, pm_tol = task
    try:
        pbp = archive.load_raw(pbp_path)
        box = archive.load_raw(box_path)
    except Exception as e:   # noqa: BLE001 - truncated / unreadable archive file
        return GameResult(gid, False, ["unreadable"], f"{type(e).__name__}: {e}")
    return compute_game(box, pbp, gid, minutes_tol=minutes_tol, pm_tol=pm_tol)


def _store(conn: sqlite3.Connection, r: GameResult) -> None:
    conn.execute("DELETE FROM stints WHERE game_id=?", (r.game_id,))
    conn.execute("DELETE FROM player_game_oncourt WHERE game_id=?", (r.game_id,))
    conn.executemany(
        f"INSERT INTO stints ({','.join(STINT_COLS)}) VALUES ({','.join('?' * len(STINT_COLS))})",
        [(r.game_id, *s) for s in r.stints])
    conn.executemany(
        f"INSERT INTO player_game_oncourt ({','.join(PGO_COLS)}) "
        f"VALUES ({','.join('?' * len(PGO_COLS))})",
        [(r.game_id, *p) for p in r.players])


def write_check(conn: sqlite3.Connection, summary: dict) -> dict:
    """Store the ``stints_store`` data check for a build run."""
    from courtside.mlops import checks
    n = summary["attempted"]
    fails = summary["failures"]
    if n == 0:
        res = checks._res(CHECK_NAME, "skip", None, CHECK_WARN, "no new games to validate")
    else:
        rate = summary["validated"] / n
        causes = summary["failure_causes"]
        head = (f"{summary['validated']}/{n} games validated ({rate:.2%}; strict "
                f"{summary['strict']}/{n}); {summary['stored_total']} stored in total")
        listed = ", ".join(f"{f['game_id']} [{'/'.join(f['causes'])}]" for f in fails[:FAILURES_LISTED])
        more = f" (+{len(fails) - FAILURES_LISTED} more)" if len(fails) > FAILURES_LISTED else ""
        bad = f"{head}; causes {causes}; failed: {listed}{more}"
        res = checks._ladder(CHECK_NAME, rate, CHECK_WARN, CHECK_FAIL, head, bad)
    res["run_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO data_checks (run_at, check_name, status, value, threshold, "
            "detail) VALUES (:run_at, :check_name, :status, :value, :threshold, :detail)", res)
    return res


def build(conn: sqlite3.Connection, seasons: Iterable[str] | str | None = None, *,
          game_ids: Iterable[str] | None = None, force: bool = False, workers: int = 1,
          limit: int | None = None, progress: bool = False,
          minutes_tol: float = MINUTES_TOL_S, pm_tol: int = PM_TOL) -> dict:
    """Reconstruct and store stints for every archived game of ``seasons`` / ``game_ids``.

    Games already stored are skipped unless ``force``. Commits every ``BATCH`` games, so a killed
    run resumes where it stopped. ``workers`` > 1 reconstructs games in a process pool (the DB is
    only written by this process). Returns a summary dict (counts, failures, seconds)."""
    t0 = time.monotonic()
    games = list_archived_games(seasons, game_ids)
    have = set() if force else stored_games(conn)
    todo = [g for g in games if g[0] not in have]
    if limit is not None:
        todo = todo[:limit]
    summary: dict = {
        "archived": len(games), "skipped": sum(g[0] in have for g in games),
        "attempted": 0, "validated": 0, "strict": 0, "failures": [], "failure_causes": {},
        "stints": 0, "player_rows": 0, "anomaly_games": 0, "seconds": 0.0,
    }
    tasks = [(gid, day, str(p), str(b), minutes_tol, pm_tol) for gid, day, p, b in todo]

    def results():
        if workers <= 1 or len(tasks) < 2 * BATCH:
            for t in tasks:
                yield _work(t)
            return
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, maxtasksperchild=200) as pool:
            yield from pool.imap(_work, tasks, chunksize=4)

    pending = 0
    for r in results():
        summary["attempted"] += 1
        if r.anomalies:
            summary["anomaly_games"] += 1
        if r.ok:
            _store(conn, r)
            summary["validated"] += 1
            summary["strict"] += r.strict
            summary["stints"] += len(r.stints)
            summary["player_rows"] += len(r.players)
            pending += 1
        else:
            summary["failures"].append({"game_id": r.game_id, "causes": r.causes,
                                        "detail": r.detail})
            for c in r.causes:
                summary["failure_causes"][c] = summary["failure_causes"].get(c, 0) + 1
        if pending >= BATCH:
            conn.commit()
            pending = 0
        if progress and summary["attempted"] % 100 == 0:
            print(f"  {summary['attempted']}/{len(tasks)} games, {summary['validated']} stored, "
                  f"{time.monotonic() - t0:.0f}s", flush=True)
    conn.commit()
    summary["stored_total"] = len(stored_games(conn))
    summary["seconds"] = round(time.monotonic() - t0, 1)
    summary["check"] = write_check(conn, summary) if summary["attempted"] else None
    return summary


# ------------------------------------------------------------------------- CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("stints", help="reconstruct and store stints / on-court tallies from "
                                      "the cdn play-by-play + boxscore archive")
    p.add_argument("--season", action="append",
                   help="season like 2024-25 (repeatable); default: every archived season")
    p.add_argument("--force", action="store_true", help="recompute games already stored")
    p.add_argument("--workers", type=int, default=1, help="reconstruction processes (default 1)")
    p.add_argument("--limit", type=int, help="stop after this many games (testing)")
    return {"stints": _stints}


def _stints(args) -> int:
    from courtside import db
    conn = db.connect()
    s = build(conn, args.season, force=args.force, workers=args.workers, limit=args.limit,
              progress=True)
    print(f"== stints: {s['validated']}/{s['attempted']} validated (strict {s['strict']}) "
          f"({s['skipped']} already stored, {s['archived']} archived with both files); "
          f"{s['stints']} stints, {s['player_rows']} player rows, {s['seconds']}s; "
          f"{s['anomaly_games']} games with feed anomalies")
    if s["failures"]:
        print(f"   failure causes {s['failure_causes']}")
        for f in s["failures"][:40]:
            print(f"   {f['game_id']} {'/'.join(f['causes'])}: {f['detail']}")
    if s["check"]:
        print(f"   check {s['check']['check_name']}: {s['check']['status']}")
    return 0
