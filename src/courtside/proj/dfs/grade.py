"""Nightly DFS grading (docs/PLAYERS.md §7.4, §8 Phase 3, task T3.0).

Every DFS build is graded once the games of its slate are final and the box scores are in
``player_games``. `grade_day(conn, day)` is what the postgame job calls (step ``dfs_grade``);
it is idempotent and safe to re-run (``force=True`` re-grades everything).

What is written (migration 12 tables, plus the two snapshot tables of migration 9)

* ``player_projection_grades``: one row per ``player_projections`` row of the day (every run,
  every slate) whose game is final and boxed. ``played`` = box minutes > 0; ``actual_dk`` is
  `rules.dk_points` of the box line (DNP, i.e. no box row, = 0); ``proj_min`` / ``proj_dk`` are
  the stored means of the UNCONDITIONAL ("all") distribution; ``in80`` / ``in50`` test the stored
  q10-q90 / q25-q75 of that distribution; ``dk_fppg`` / ``salary`` come from the latest
  ``dfs_salaries`` capture at or before the run (the baselines).
  PIT and CRPS: from the run's simulation file (``dfs_runs.sim_path``) when it is there: the
  randomised PIT u = F(y-) + V (F(y) - F(y-)) of the simulated DK points of that player
  (`metrics.pit_randomized`; the DNP point mass at 0 is spread by V ~ U(0,1), V a fixed hash of
  the projection id so a re-grade gives the same number) and `metrics.crps_samples_matrix`.
  Without sims, PIT is read off the stored quantiles: the "played" quantiles q05..q95 are
  joined linearly (tails extended by one inter-quantile step, floor 0) into F_played and mixed
  with the DNP atom, F(y) = (1-p_play) + p_play F_played(y), y = 0 -> V (1-p_play); CRPS is the
  7-level pinball approximation `metrics.crps_quantiles` (coarse and slightly low).
* ``dfs_lineup_grades``: one row per ``dfs_lineups`` row (cash / topn / swap / ...) of every ok
  run of a slate whose games are ALL final. ``actual`` = sum of the players' actual DK points
  (Showdown CPT x1.5); ``hindsight`` = the optimal lineup of the slate pool solved with the actual
  points (`optimizer.optimize`, same rules); ``ratio_hindsight`` = actual / hindsight;
  ``pct_random`` = mid-rank percentile of ``actual`` among N=5,000 random salary-valid lineups of
  the pool (seeded by the slate; slots filled in turn by uniform draws, rejected when over the
  cap or under min games/teams). The cash line comes only from an uploaded standings file
  (``dfs_contest_results``): ``cash_line_source`` 'real' with ``cash_line`` = the lowest score
  that was paid, ``would_cash``, ``est_rank`` (1 + entries scoring more) and ``est_payout`` (the
  contest payout table at that rank). Without one the source is 'none' and those columns stay
  NULL: nothing is invented. ``detail_json`` carries the day, mode, rank, run kind, whether the
  lineup was the one in effect at lock ("effective": the reference run's lineup, replaced by its
  swap version), per-player actuals, unresolved players and the per-contest numbers.
  ``dfs_lineups.actual`` and ``ratio_hindsight`` are filled too.
* ``dfs_metric_snapshots``, ``player_metric_snapshots``: as_of = the graded day, windows 1d / 7d /
  30d / season (see `write_snapshots`). The per-slate baselines live in dfs_metric_snapshots as
  model ``slate:<slate_id>``, metric ``baselines``, window ``1d`` (value = hindsight points,
  details_json = the FPPG-optimizer and salary-implied-optimizer lineups built from the same pool
  at the time of the reference run and scored with the actual points, the random-lineup summary
  and our effective lineups), so every night shows ours vs FPPG vs salary vs hindsight.

Reference run of a slate: its latest ``lock10`` run, else ``lock45``, else the latest non-swap
build. Baselines drop DK "O" players (a human using FPPG would not roster a known Out).
Salary-implied = the RotoGuru OLS line (seasons before the day's season) of DK points on salary.

Simulated and real results are always separate: lineup metrics are about our stored builds;
``model='real'`` rows come only from ``dfs_entries`` (entries the owner actually played).

Ids: DK ``playerId`` (slot lists, pool) -> NBA id through ``player_ids`` (source 'dk'); a pool
player without a match is looked up by name in the day's box scores, else left out of the
hindsight/baseline pools and listed in ``detail_json.unresolved``. Slates called ``rg:<date>``
are RotoGuru history (validation): pool = `optimizer.hist_pool`, ids from ``dfs_hist_salaries``.

CLI (register ``courtside.proj.dfs.grade`` in ``DFS_MODULES`` of ``proj/cli.py``)::

    courtside dfs grade --date 2026-10-20 [--force]
    courtside dfs report [--window 7d] [--as-of 2026-10-21]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config
from .. import metrics as MT
from .. import store as SO
from . import optimizer as O
from . import rules as R

log = logging.getLogger("courtside.proj.dfs.grade")

GRADER_VERSION = "grade-v1"
N_RANDOM = 5_000
WINDOWS = ("1d", "7d", "30d", "season")
WINDOW_DAYS = {"1d": 1, "7d": 7, "30d": 30, "season": None}
MIN_DAYS_SE = 5                         # clustered SEs need at least this many days / games
SOLVE_LIMIT_S = 30.0
KIND_PRIORITY = {"lock10": 0, "lock45": 1, "manual": 2, "swap": 3}
SIMULATED_SOURCES = frozenset({"sim", "simulated", "proxy", "synthetic", "backtest"})
QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
QLEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CATCHUP_DAYS = 2                        # postgame step also retries the 2 days before


# ============================================================================== small helpers

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _loads(s, default=None):
    if s is None or s == "":
        return default
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return default


def _f(x):
    """JSON/SQL-safe float (None for NaN / inf / None)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _dumps(o) -> str:
    return json.dumps(o, default=str, sort_keys=True)


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _u01(key: int) -> float:
    """Deterministic uniform(0,1) from an integer (splitmix64); the PIT randomisation."""
    m = (1 << 64) - 1
    x = (int(key) * 0x9E3779B97F4A7C15 + 0xD1B54A32D192ED03) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    x ^= x >> 31
    return ((x >> 11) + 0.5) / float(1 << 53)


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _is_hist(slate_id: str) -> bool:
    return str(slate_id).startswith("rg:")


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


# ============================================================================== game state

def game_states(conn: sqlite3.Connection, day: date, game_ids) -> dict[str, str]:
    """{game_id: 'final' | 'pending' | 'nobox'}. 'final' needs the box score rows in
    ``player_games``; a ``games`` row must say final. Past games with no ``games`` row (history
    loaded from the game logs) count as final once boxed."""
    ids = sorted({str(g) for g in game_ids if g})
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    status = {r[0]: r[1] for r in conn.execute(
        f"SELECT game_id, status FROM games WHERE game_id IN ({ph})", ids)}
    boxed = {r[0] for r in conn.execute(
        f"SELECT DISTINCT game_id FROM player_games WHERE game_id IN ({ph})", ids)}
    past = day < config.today_nba_day()
    out = {}
    for g in ids:
        st = status.get(g)
        if st is None:
            out[g] = "final" if (g in boxed and past) else ("nobox" if past else "pending")
        elif st != "final":
            out[g] = "pending"
        else:
            out[g] = "final" if g in boxed else "nobox"
    return out


def load_box(conn: sqlite3.Connection, game_ids) -> dict[int, dict]:
    """{nba player_id: {min, dk, name, team, game_id}} of the given games' box scores."""
    ids = sorted({str(g) for g in game_ids if g})
    out: dict[int, dict] = {}
    if not ids:
        return out
    ph = ",".join("?" * len(ids))
    for r in conn.execute(
            f"SELECT game_id, player_id, player_name, team, min, pts, fg3m, reb, ast, stl, blk, "
            f"tov FROM player_games WHERE game_id IN ({ph})", ids):
        out[int(r["player_id"])] = {"min": float(r["min"] or 0.0), "dk": R.dk_points(r),
                                    "name": r["player_name"], "team": r["team"],
                                    "game_id": r["game_id"]}
    return out


# ============================================================================== slates, runs

def day_slates(conn: sqlite3.Connection, day: date) -> list[dict]:
    """Slates of an NBA day that have a stored run or projections (dfs_slates, plus slates that
    only appear in player_projections such as historical 'rg:<date>')."""
    d = day.isoformat()
    out: dict[str, dict] = {}
    for r in conn.execute("SELECT * FROM dfs_slates WHERE nba_day=? ORDER BY slate_id", (d,)):
        out[str(r["slate_id"])] = {"slate_id": str(r["slate_id"]), "tag": r["tag"],
                                   "game_type_id": r["game_type_id"], "lock_utc": r["lock_utc"],
                                   "game_ids": [str(g) for g in _loads(r["game_ids_json"], [])
                                                or []]}
    proj_games: dict[str, set] = {}
    for sid, gid in conn.execute("SELECT DISTINCT slate_id, game_id FROM player_projections "
                                 "WHERE nba_day=? AND slate_id IS NOT NULL", (d,)):
        proj_games.setdefault(str(sid), set()).add(str(gid))
    for sid in proj_games:
        out.setdefault(sid, {"slate_id": sid, "tag": None, "game_type_id": None,
                             "lock_utc": None, "game_ids": []})
    res = []
    for sid, s in out.items():
        has_run = conn.execute("SELECT 1 FROM dfs_runs WHERE slate_id=? AND ok=1 LIMIT 1",
                               (sid,)).fetchone()
        if not has_run and sid not in proj_games:
            continue
        if not s["game_ids"]:
            s["game_ids"] = sorted(proj_games.get(sid, ()))
        if not s["game_ids"] and _is_hist(sid):
            s["game_ids"] = [r[0] for r in conn.execute(
                "SELECT DISTINCT game_id FROM dfs_hist_salaries WHERE game_date=? AND on_slate=1 "
                "AND game_id IS NOT NULL", (d,))]
        res.append(s)
    return res


def slate_runs(conn: sqlite3.Connection, slate_id: str) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM dfs_runs WHERE slate_id=? AND ok=1 "
                          "ORDER BY created_at, rowid", (str(slate_id),)):
        d = dict(r)
        d["detail_obj"] = _loads(d.get("detail"), {}) or {}
        if not isinstance(d["detail_obj"], dict):
            d["detail_obj"] = {}
        out.append(d)
    return out


def reference_run(runs: list[dict]) -> dict | None:
    """Latest lock10 run, else lock45, else the latest non-swap build."""
    builds = [r for r in runs if r["kind"] != "swap" and not r["detail_obj"].get("skipped")]
    if not builds:
        return None
    best = min(KIND_PRIORITY.get(r["kind"], 2) for r in builds)
    cand = [r for r in builds if KIND_PRIORITY.get(r["kind"], 2) == best]
    return max(cand, key=lambda r: r["created_at"])


def lineup_rows(conn: sqlite3.Connection, run_ids: list[str]) -> list[dict]:
    out = []
    for rid in run_ids:
        for r in conn.execute("SELECT * FROM dfs_lineups WHERE run_id=? ORDER BY mode, rank, id",
                              (rid,)):
            d = dict(r)
            d["slots"] = _loads(d["slots_json"], []) or []
            d["why"] = _loads(d.get("why_json"), {}) or {}
            out.append(d)
    return out


def effective_lineups(runs: list[dict], by_run: dict[str, list[dict]]) -> dict[int, dict]:
    """{lineup id: {'role': mode, 'rank', 'via'}} for the lineups in effect at lock: the
    reference run's lineups, each replaced by its newest later swap version."""
    ref = reference_run(runs)
    if ref is None:
        return {}
    cur = {(d["mode"], d["rank"]): {"id": d["id"], "via": "build"} for d in by_run.get(ref["run_id"], [])}
    for r in runs:
        if r["kind"] != "swap" or r["created_at"] <= ref["created_at"]:
            continue
        for d in by_run.get(r["run_id"], []):
            frm = d["why"].get("from") or {}
            key = (frm.get("mode"), frm.get("rank"))
            if key in cur:
                cur[key] = {"id": d["id"], "via": "swap"}
    return {v["id"]: {"role": k[0], "rank": k[1], "via": v["via"]} for k, v in cur.items()}


# ============================================================================== pools and ids

def _pool_at(conn, slate_id: str, ts: str | None, rules, *, latest: bool = False):
    """Pool of a slate at the latest capture at or before `ts` (else the earliest capture;
    `latest` = the newest capture). None when the slate has no captures or no rules."""
    caps = [r[0] for r in conn.execute("SELECT DISTINCT captured_at FROM dfs_salaries WHERE "
                                       "slate_id=? ORDER BY captured_at", (str(slate_id),))]
    if not caps or rules is None:
        return None
    if latest or not ts:
        pick = caps[-1]
    else:
        pick = max((c for c in caps if c <= ts), default=caps[0])
    return R.build_pool(conn, slate_id, rules=rules, captured_at=pick)


def slate_rules(conn, slate_id: str):
    return O.classic_rules() if _is_hist(slate_id) else R.load_rules(conn, slate_id)


def slate_pool(conn, slate: dict, rules, ts: str | None, day: date):
    sid = slate["slate_id"]
    if rules is None:
        return None
    if _is_hist(sid):
        return O.hist_pool(conn, day.isoformat(), rules)
    return _pool_at(conn, sid, ts, rules)


def id_map(conn, slate_id: str, day: date) -> dict[str, int]:
    """{pool player id: NBA player_id}."""
    if _is_hist(slate_id):
        return {str(r[0]): int(r[1]) for r in conn.execute(
            "SELECT rg_id, player_id FROM dfs_hist_salaries WHERE game_date=? AND "
            "player_id IS NOT NULL", (day.isoformat(),))}
    from .. import ids as IDS
    return IDS.nba_ids(conn, "dk")


class Actuals:
    """Actual DK points of the pool players of one slate (box scores of its final games)."""

    def __init__(self, conn, slate: dict, day: date, box: dict[int, dict]):
        self.box = box
        self.hist = _is_hist(slate["slate_id"])
        self.nba = id_map(conn, slate["slate_id"], day)
        self.by_name: dict[str, list[int]] = {}
        from ... import injuries
        for nid, b in box.items():
            self.by_name.setdefault(injuries.norm_name(b["name"]), []).append(nid)
        self._norm = injuries.norm_name
        self.hist_pts = ({str(r[0]): float(r[1]) for r in conn.execute(
            "SELECT rg_id, dk_pts FROM dfs_hist_salaries WHERE game_date=? AND dk_pts IS NOT NULL",
            (day.isoformat(),))} if self.hist else {})
        self.minutes: dict[str, float] = {}
        self.unresolved: dict[str, str] = {}

    def of(self, p) -> float | None:
        """Actual DK points (mapped + no box row = DNP = 0); None when unresolvable."""
        pid = p.dk_player_id
        nid = self.nba.get(pid)
        if nid is None:
            hits = self.by_name.get(self._norm(p.name), [])
            if len(hits) == 1:
                nid = hits[0]
        if nid is None:
            if pid in self.hist_pts:
                return self.hist_pts[pid]
            self.unresolved[pid] = p.name
            return None
        b = self.box.get(nid)
        self.minutes[pid] = b["min"] if b else 0.0
        return b["dk"] if b else 0.0

    def for_players(self, players) -> dict[str, float]:
        out = {}
        for p in players:
            if p.dk_player_id in out:
                continue
            v = self.of(p)
            if v is not None:
                out[p.dk_player_id] = float(v)
        return out


# ============================================================================== lineups

def lineup_actual(slots: list[dict], actual: dict[str, float], rules) -> tuple[float, list[dict], list[str]]:
    """(points, per-player detail, unresolved ids) of a stored lineup (slots_json entries)."""
    total, det, miss = 0.0, [], []
    for s in slots:
        pid = str(s["player_id"])
        try:
            mult = rules.slot_multiplier(s["slot"])
        except KeyError:
            mult = 1.0
        a = actual.get(pid)
        if a is None:
            miss.append(pid)
            a = 0.0
        total += a * mult
        det.append({"slot": s["slot"], "player_id": pid, "name": s.get("name"),
                    "actual": round(a, 2), "mult": mult})
    return total, det, miss


def score_lineup(lu, actual: dict[str, float], rules) -> float:
    return float(sum(actual.get(e.player.dk_player_id, 0.0) * rules.slot_multiplier(e.slot)
                     for e in lu.entries))


def random_lineup_scores(players, rules, actual: dict[str, float], n: int = N_RANDOM,
                         seed: int = 1) -> np.ndarray:
    """Actual scores of up to `n` random salary-valid lineups of the pool: slots (fewest
    eligible players first) are filled in turn by uniform draws without replacement; a lineup
    over the cap, or under min games / min teams, is rejected. Seeded. Players without a known
    actual are not drawn."""
    pl = [p for p in players if p.dk_player_id in actual]
    if not pl:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(rules.slot_names))
    count = {nm: rules.slot_names.count(nm) for nm in names}
    elig = {nm: np.array([i for i, p in enumerate(pl) if nm in p.slots], dtype=int)
            for nm in names}
    if any(len(elig[nm]) < count[nm] for nm in names):
        return np.zeros(0)
    order = sorted(names, key=lambda nm: len(elig[nm]) / count[nm])
    sal = {nm: np.array([pl[i].salary_for(nm, rules) for i in range(len(pl))]) for nm in names}
    act = np.array([actual[p.dk_player_id] for p in pl])
    teams = np.array([hash(p.team or f"?{i}") for i, p in enumerate(pl)])
    games = np.array([hash(O.game_key(p)) for p in pl])
    mult = {nm: rules.slot_multiplier(nm) for nm in names}
    out = []
    tries, max_tries = 0, n * 40
    while len(out) < n and tries < max_tries:
        tries += 1
        used: list[int] = []
        total_sal = 0
        score = 0.0
        ok = True
        for nm in order:
            for _ in range(count[nm]):
                cand = elig[nm]
                for _try in range(30):
                    i = int(cand[rng.integers(len(cand))])
                    if i not in used:
                        break
                else:
                    ok = False
                    break
                used.append(i)
                total_sal += int(sal[nm][i])
                score += act[i] * mult[nm]
            if not ok:
                break
        if not ok or total_sal > rules.salary_cap:
            continue
        if rules.min_teams and len(set(teams[used])) < rules.min_teams:
            continue
        if rules.min_games and len(set(games[used])) < rules.min_games:
            continue
        out.append(score)
    return np.array(out)


# ------------------------------------------------------------------ cash lines (real only)

def _payout_at(table: list[dict], rank: int) -> float:
    for t in table:
        if t["min_pos"] <= rank <= t["max_pos"]:
            return float(t["payout"])
    return 0.0


def contest_results(conn, slate_id: str) -> list[dict]:
    """Contests of the slate with an uploaded standings file: id, name, fee, field, the sorted
    scores and the real cash line. Empty when nothing was uploaded."""
    if not _has_table(conn, "dfs_contest_results"):
        return []
    cids = {r[0] for r in conn.execute("SELECT contest_id FROM dfs_contests WHERE slate_id=?",
                                       (str(slate_id),))}
    if _has_table(conn, "dfs_entries"):
        cids |= {r[0] for r in conn.execute("SELECT DISTINCT contest_id FROM dfs_entries WHERE "
                                            "slate_id=? AND contest_id IS NOT NULL", (str(slate_id),))}
    out = []
    for cid in sorted(c for c in cids if c):
        rows = conn.execute("SELECT points, payout FROM dfs_contest_results WHERE contest_id=? "
                            "AND points IS NOT NULL", (cid,)).fetchall()
        if not rows:
            continue
        pts = np.array([r[0] for r in rows], float)
        pay = np.array([r[1] if r[1] is not None else np.nan for r in rows], float)
        meta = conn.execute("SELECT name, entry_fee, field_size, payout_json FROM dfs_contests "
                            "WHERE contest_id=?", (cid,)).fetchone()
        table = _loads(meta["payout_json"], []) if meta else []
        table = [t for t in (table or []) if isinstance(t, dict) and "min_pos" in t]
        line = None
        paid = pts[np.nan_to_num(pay, nan=0.0) > 0]
        if len(paid):
            line = float(paid.min())
        elif table:
            places = max((t["max_pos"] for t in table if (t.get("payout") or 0) > 0), default=0)
            srt = np.sort(pts)[::-1]
            if 0 < places <= len(srt):
                line = float(srt[places - 1])
        out.append({"contest_id": cid, "name": meta["name"] if meta else None,
                    "entry_fee": meta["entry_fee"] if meta else None,
                    "field_size": int(len(pts)), "points": np.sort(pts)[::-1],
                    "payouts": pay[np.argsort(-pts, kind="stable")], "table": table,
                    "cash_line": line})
    return out


def _cash_fields(actual: float, contests: list[dict], lineup: dict) -> tuple[dict, list[dict]]:
    """(dfs_lineup_grades columns, per-contest detail) for a lineup against the real contests."""
    per = []
    for c in contests:
        srt = c["points"]                                  # descending
        rank = int((srt > actual + 1e-9).sum()) + 1
        if c["table"]:
            payout = _payout_at(c["table"], rank)
        elif rank <= len(srt):
            payout = float(np.nan_to_num(c["payouts"][rank - 1], nan=0.0))
        else:
            payout = 0.0
        per.append({"contest_id": c["contest_id"], "name": c["name"], "field": c["field_size"],
                    "cash_line": c["cash_line"], "est_rank": rank, "est_payout": payout,
                    "would_cash": (None if c["cash_line"] is None
                                   else int(actual >= c["cash_line"] - 1e-9))})
    if not per:
        return {"cash_line": None, "cash_line_source": "none", "would_cash": None,
                "est_rank": None, "est_payout": None}, per
    pick = None
    if lineup.get("contest_id"):
        pick = next((p for p in per if p["contest_id"] == lineup["contest_id"]), None)
    if pick is None and lineup["mode"] == "cash":
        pick = next((p for p in per if p["name"] and any(
            k in p["name"].lower() for k in ("double up", "50/50", "h2h", "head"))), None)
    if pick is None:
        pick = max(per, key=lambda p: p["field"])
    return {"cash_line": pick["cash_line"], "cash_line_source": "real",
            "would_cash": pick["would_cash"], "est_rank": pick["est_rank"],
            "est_payout": pick["est_payout"]}, per


# ------------------------------------------------------------------ baselines

def salary_line(conn, day: date, cache: dict | None = None) -> tuple[float, float, str]:
    """(a, b, how): DK points = a + b * salary, OLS on RotoGuru rows of seasons before the day's
    season (evallite.fit_salary_line, all_seasons); the 5-points-per-$1k rule if it can't fit.
    `cache` (a dict kept by the caller) avoids re-reading the 250k RotoGuru rows."""
    from .. import evallite as EV
    season = EV.season_label(day.isoformat())
    if cache is not None and season in cache:
        return cache[season]
    try:
        line = (*EV.fit_salary_line(EV.load_rotoguru(conn), season, "all_seasons"),
                f"rotoguru<{season}")
    except Exception as e:  # noqa: BLE001
        log.warning("salary line fit failed (%s); using %.1f FP per $1k", e, O.SALARY_FP_PER_K)
        line = (0.0, O.SALARY_FP_PER_K / 1000.0, "fixed")
    if cache is not None:
        cache[season] = line
    return line


def hist_fppg(conn, day: date, nba_ids: list[int]) -> dict[int, float]:
    """Season-to-date DK mean before `day` (previous season's if fewer than 5 games), the DK
    FPPG convention, from box scores. Used for historical slates only (no DK capture)."""
    cur = R.season_of(day)
    prev = R.prev_season(cur)
    if not nba_ids:
        return {}
    ph = ",".join("?" * len(nba_ids))
    rows: dict[int, dict[str, list[float]]] = {}
    for r in conn.execute(
            f"SELECT player_id, season, pts, fg3m, reb, ast, stl, blk, tov FROM player_games "
            f"WHERE player_id IN ({ph}) AND season IN (?, ?) AND game_date < ?",
            (*nba_ids, cur, prev, day.isoformat())):
        rows.setdefault(int(r["player_id"]), {}).setdefault(r["season"], []).append(R.dk_points(r))
    out = {}
    for pid, by in rows.items():
        v = by.get(cur, [])
        if len(v) < R.MIN_CURRENT_GAMES:
            v = by.get(prev, []) or v
        if v:
            out[pid] = float(np.mean(v))
    return out


def baseline_lineup(pool, projections: dict, rules, actual: dict[str, float], bans) -> dict | None:
    lu = O.optimize(pool, projections, rules, mode="mean", bans=bans, time_limit_s=SOLVE_LIMIT_S)
    if lu is None:
        return None
    return {"actual": round(score_lineup(lu, actual, rules), 3),
            "projected": round(float(lu.projected), 3), "salary": lu.salary,
            "optimal": bool(lu.optimal),
            "players": [{"slot": e.slot, "player_id": e.player.dk_player_id, "name": e.player.name,
                         "actual": round(actual.get(e.player.dk_player_id, 0.0), 2)}
                        for e in lu.entries]}


# ============================================================================== slate grading

def grade_slate(conn, day: date, slate: dict, *, force: bool = False, n_random: int = N_RANDOM,
                now: str | None = None, cache: dict | None = None) -> dict:
    """Grade every stored lineup of one slate. Returns a status dict (never raises for the
    normal not-ready cases)."""
    t0 = time.time()
    sid = slate["slate_id"]
    now = now or _now()
    runs = slate_runs(conn, sid)
    by_run = {r["run_id"]: lineup_rows(conn, [r["run_id"]]) for r in runs}
    rows = [d for r in runs for d in by_run[r["run_id"]]]
    if not rows:
        return {"slate_id": sid, "status": "skipped", "reason": "no lineups"}
    states = game_states(conn, day, slate["game_ids"])
    not_final = {g: s for g, s in states.items() if s != "final"}
    if not slate["game_ids"] or not_final:
        return {"slate_id": sid, "status": "pending", "reason": "games not final",
                "games": not_final or {"?": "no game list"}}
    graded = {r[0]: (r[1],) for r in conn.execute(
        f"SELECT lineup_id, cash_line_source FROM dfs_lineup_grades WHERE lineup_id IN "
        f"({','.join('?' * len(rows))})", [d["id"] for d in rows])}
    contests = contest_results(conn, sid)
    if not force and len(graded) == len(rows):
        stale = bool(contests) and any(g[0] == "none" for g in graded.values())
        if not stale:
            return {"slate_id": sid, "status": "cached", "lineups": len(rows)}
    rules = slate_rules(conn, sid)
    if rules is None:
        return {"slate_id": sid, "status": "skipped", "reason": "no rules for slate"}
    ref = reference_run(runs)
    ref_ts = ref["created_at"] if ref else None
    lock = slate.get("lock_utc")
    hind_pool = slate_pool(conn, slate, rules, lock or ref_ts, day)
    ref_pool = slate_pool(conn, slate, rules, ref_ts, day)
    if hind_pool is None or ref_pool is None:
        return {"slate_id": sid, "status": "skipped", "reason": "no salary capture for slate"}
    box = load_box(conn, slate["game_ids"])
    A = Actuals(conn, slate, day, box)
    actual = A.for_players(list(hind_pool.players) + list(ref_pool.players))
    stored_ids = {str(s["player_id"]) for d in rows for s in d["slots"]}
    missing = stored_ids - set(actual)
    if missing:                                    # lineup players that left the pool
        by_name = {s["player_id"]: s for d in rows for s in d["slots"]}
        for pid in missing:
            nm = by_name.get(pid, {}).get("name")
            if nm:
                q = type("P", (), {"dk_player_id": pid, "name": nm})()
                v = A.of(q)
                if v is not None:
                    actual[pid] = float(v)
    # hindsight
    hs = O.optimize(hind_pool, {k: O.Projection(v) for k, v in actual.items()
                                if k in hind_pool.by_id()}, rules, mode="mean",
                    time_limit_s=SOLVE_LIMIT_S)
    hindsight = score_lineup(hs, actual, rules) if hs is not None else None
    # random lineups
    rnd = random_lineup_scores(hind_pool.players, rules, actual, n_random, _seed(sid, day))
    # baselines from the reference pool
    a, b, how = salary_line(conn, day, cache)
    bans = {p.dk_player_id for p in ref_pool.players if p.status == "O"}
    if _is_hist(sid):          # no DK status in history: the baselines get the "known availability"
        bans |= {k for k, m in A.minutes.items() if m <= 0}     # oracle that our backtest run has
    if _is_hist(sid):
        fp = hist_fppg(conn, day, sorted({n for k in (p.dk_player_id for p in ref_pool.players)
                                          if (n := A.nba.get(k)) is not None}))
        fppg_proj = {p.dk_player_id: O.Projection(fp[A.nba[p.dk_player_id]])
                     for p in ref_pool.players if A.nba.get(p.dk_player_id) in fp}
    else:
        fppg_proj = {p.dk_player_id: O.Projection(p.fppg) for p in ref_pool.players
                     if p.fppg is not None}
    sal_proj = {p.dk_player_id: O.Projection(a + b * p.salary) for p in ref_pool.players}
    base = {"fppg": baseline_lineup(ref_pool, fppg_proj, rules, actual, bans),
            "salary": baseline_lineup(ref_pool, sal_proj, rules, actual, bans)}
    for v in base.values():
        if v:
            v["pct_random"] = _pct(v["actual"], rnd)
            v["ratio_hindsight"] = _ratio(v["actual"], hindsight)
    eff = effective_lineups(runs, by_run)
    kind_of = {r["run_id"]: r["kind"] for r in runs}
    out_rows, ours = [], []
    for d in rows:
        pts, det, miss = lineup_actual(d["slots"], actual, rules)
        cash, per = _cash_fields(pts, contests, d)
        e = eff.get(d["id"])
        detail = {"day": day.isoformat(), "mode": d["mode"], "rank": d["rank"],
                  "kind": kind_of.get(d["run_id"]), "via": (e or {}).get("via"),
                  "effective": bool(e), "role": (e or {}).get("role"),
                  "game_type": "showdown" if rules.is_showdown else "classic",
                  "reference_run": ref["run_id"] if ref else None, "players": det,
                  "unresolved": miss, "contests": per, "n_random": int(len(rnd)),
                  "version": GRADER_VERSION}
        ratio = _ratio(pts, hindsight)
        pr = _pct(pts, rnd)
        out_rows.append((d["id"], d["run_id"], sid, now, pts, d["mean"], hindsight, ratio, pr,
                         cash["cash_line"], cash["cash_line_source"], cash["would_cash"],
                         cash["est_rank"], cash["est_payout"], _dumps(detail)))
        if e:
            ours.append({"role": e["role"], "rank": e["rank"], "via": e["via"],
                         "lineup_id": d["id"], "actual": round(pts, 3),
                         "ratio_hindsight": ratio, "pct_random": pr})
    summary = {"slate_id": sid, "day": day.isoformat(), "tag": slate.get("tag"),
               "game_type": "showdown" if rules.is_showdown else "classic",
               "reference_run": ({"run_id": ref["run_id"], "kind": ref["kind"],
                                  "created_at": ref["created_at"]} if ref else None),
               "pool": len(hind_pool.players), "pool_resolved": len(actual),
               "unresolved_pool": sorted(A.unresolved.values())[:25],
               "hindsight": {"actual": _f(hindsight), "players": [
                   {"slot": e.slot, "player_id": e.player.dk_player_id, "name": e.player.name,
                    "actual": round(actual.get(e.player.dk_player_id, 0.0), 2)}
                   for e in hs.entries] if hs else []},
               "fppg": base["fppg"], "salary": base["salary"], "salary_line": [a, b, how],
               "baseline_bans": ("DNPs (history: known-availability oracle, like the projections)"
                                 if _is_hist(sid) else "DK status O"),
               "random": {"n": int(len(rnd)), "mean": _f(rnd.mean()) if len(rnd) else None,
                          "p50": _f(np.percentile(rnd, 50)) if len(rnd) else None,
                          "p90": _f(np.percentile(rnd, 90)) if len(rnd) else None,
                          "p99": _f(np.percentile(rnd, 99)) if len(rnd) else None},
               "ours": ours, "real_contests": [c["contest_id"] for c in contests],
               "graded_at": now, "version": GRADER_VERSION}
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO dfs_lineup_grades (lineup_id, run_id, slate_id, graded_at, "
            "actual, projected, hindsight, ratio_hindsight, pct_random, cash_line, "
            "cash_line_source, would_cash, est_rank, est_payout, detail_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out_rows)
        conn.executemany("UPDATE dfs_lineups SET actual=?, ratio_hindsight=? WHERE id=?",
                         [(r[4], r[7], r[0]) for r in out_rows])
        conn.execute(
            "INSERT OR REPLACE INTO dfs_metric_snapshots (as_of, model, metric, \"window\", n, "
            "value, se, details_json) VALUES (?,?,?,?,?,?,?,?)",
            (day.isoformat(), f"slate:{sid}", "baselines", "1d", len(out_rows),
             _f(hindsight), None, _dumps(summary)))
    return {"slate_id": sid, "status": "graded", "lineups": len(out_rows),
            "hindsight": _f(hindsight), "fppg": (base["fppg"] or {}).get("actual"),
            "salary": (base["salary"] or {}).get("actual"),
            "unresolved": len(A.unresolved), "real_contests": len(contests),
            "seconds": round(time.time() - t0, 2)}


def _ratio(actual, hindsight):
    if hindsight is None or hindsight <= 0:
        return None
    return _f(actual / hindsight)


def _pct(value, ref: np.ndarray):
    if not len(ref):
        return None
    return _f(MT.percentile_rank(float(value), ref))


# ============================================================================== player grading

def pit_from_quantiles(y: float, p_play: float | None, qp: np.ndarray | None,
                       qa: np.ndarray | None, v: float) -> float | None:
    """Randomised PIT from stored quantiles (see the module docstring). `qp`: q05..q95 given the
    player plays, `qa`: q05..q95 of the unconditional distribution (used when no `qp`)."""
    def cdf(q, x):
        q = np.maximum.accumulate(np.asarray(q, float))
        lo = max(0.0, q[0] - (q[1] - q[0]))
        hi = q[-1] + (q[-1] - q[-2])
        xs = np.concatenate([[lo], q, [hi]]) + np.arange(len(q) + 2) * 1e-9
        ts = np.concatenate([[0.0], QLEVELS, [1.0]])
        return float(np.interp(x, xs, ts))
    if qp is not None and p_play is not None and np.all(np.isfinite(qp)):
        p0 = min(max(1.0 - p_play, 0.0), 1.0)
        if y <= 0.0:
            return v * p0
        return p0 + (1.0 - p0) * cdf(qp, y)
    if qa is not None and np.all(np.isfinite(qa)):
        return cdf(qa, y) if y > 0 else v * cdf(qa, 0.0)
    return None


def _sim_cols(path):
    if not path or not Path(path).exists():
        return None
    try:
        z = SO.load_sims(path)
        return {int(p): j for j, p in enumerate(z["player_id"])}, z["fp"].astype(np.float32)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read sims %s: %s", path, e)
        return None


def grade_players(conn, day: date, *, force: bool = False, now: str | None = None) -> dict:
    """Grade the day's projection rows whose game is final. Returns counts."""
    now = now or _now()
    d = day.isoformat()
    proj = [dict(r) for r in conn.execute(
        "SELECT pp.id, pp.game_id, pp.player_id, pp.run_id, pp.slate_id, pp.p_play, "
        "pp.inputs_json, (SELECT 1 FROM player_projection_grades g WHERE g.projection_id=pp.id) "
        "AS done FROM player_projections pp WHERE pp.nba_day=?", (d,))]
    if not proj:
        return {"projections": 0, "graded": 0, "pending": 0}
    states = game_states(conn, day, {p["game_id"] for p in proj})
    todo = [p for p in proj if states.get(str(p["game_id"])) == "final" and (force or not p["done"])]
    pending = sum(1 for p in proj if states.get(str(p["game_id"])) != "final")
    if not todo:
        return {"projections": len(proj), "graded": 0, "pending": pending}
    stats: dict[tuple, dict] = {}
    for r in conn.execute("SELECT s.* FROM player_projection_stats s JOIN player_projections pp "
                          "ON pp.id=s.projection_id WHERE pp.nba_day=? AND s.stat IN ('dk','min')",
                          (d,)):
        stats[(r["projection_id"], r["stat"], r["cond"])] = r
    box = load_box(conn, {p["game_id"] for p in todo})
    runs = {r["run_id"]: r for r in conn.execute(
        "SELECT run_id, slate_id, kind, created_at, sim_path FROM dfs_runs")}
    sal_cache: dict = {}
    out = []
    by_run: dict = {}
    for p in todo:
        by_run.setdefault(p["run_id"], []).append(p)
    for run_id, ps in by_run.items():
        run = runs.get(run_id)
        sims = _sim_cols(run["sim_path"]) if run else None
        for p in ps:
            b = box.get(int(p["player_id"]))
            played = 1 if (b and b["min"] > 0) else 0
            y = b["dk"] if b else 0.0
            amin = b["min"] if b else 0.0
            inputs = _loads(p["inputs_json"], {}) or {}
            sa = stats.get((p["id"], "dk", "all"))
            sp = stats.get((p["id"], "dk", "played"))
            ma = stats.get((p["id"], "min", "all"))
            qa = (np.array([sa[c] if sa[c] is not None else np.nan for c in QCOLS], float)
                  if sa is not None else None)
            qp = (np.array([sp[c] if sp[c] is not None else np.nan for c in QCOLS], float)
                  if sp is not None else None)
            v = _u01(p["id"])
            pit = crps = None
            col = sims[0].get(int(p["player_id"])) if sims else None
            if col is not None:
                X = sims[1][:, [col]]
                pit = _f(MT.pit_randomized(X, np.array([y]), np.array([v]))[0])
                crps = _f(MT.crps_samples_matrix(X, np.array([y]))[0])
            else:
                pit = _f(pit_from_quantiles(y, p["p_play"], qp, qa, v))
                if qa is not None and np.all(np.isfinite(qa)):
                    crps = _f(MT.crps_quantiles(qa[None, :], QLEVELS, np.array([y]))[0])
            in80 = in50 = None
            if sa is not None and sa["q10"] is not None and sa["q90"] is not None:
                in80 = int(sa["q10"] - 1e-9 <= y <= sa["q90"] + 1e-9)
            if sa is not None and sa["q25"] is not None and sa["q75"] is not None:
                in50 = int(sa["q25"] - 1e-9 <= y <= sa["q75"] + 1e-9)
            fppg, salary = _baseline_for(conn, sal_cache, p, run, inputs, day)
            out.append((p["id"], p["game_id"], int(p["player_id"]), run_id, now, played, amin, y,
                        _f(ma["mean"]) if ma is not None else None,
                        _f(sa["mean"]) if sa is not None else None, pit, in80, in50, crps,
                        fppg, salary))
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO player_projection_grades (projection_id, game_id, player_id, "
            "run_id, graded_at, played, actual_min, actual_dk, proj_min, proj_dk, pit, in80, "
            "in50, crps, dk_fppg, salary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out)
    return {"projections": len(proj), "graded": len(out), "pending": pending}


def _baseline_for(conn, cache: dict, p: dict, run, inputs: dict, day: date):
    """(DK FPPG, salary) of a projection's player from the latest capture at or before the run."""
    dk = inputs.get("dk_id")
    if dk is None:
        return None, None
    dk = str(dk)
    sid = p["slate_id"] or (run["slate_id"] if run else None)
    ts = run["created_at"] if run else None
    key = (sid, ts)
    if key not in cache:
        tab: dict = {}
        if sid and _is_hist(str(sid)):
            fp_by_nba: dict = {}
            rows = conn.execute("SELECT rg_id, player_id, salary FROM dfs_hist_salaries WHERE "
                                "game_date=? AND on_slate=1", (day.isoformat(),)).fetchall()
            fp_by_nba = hist_fppg(conn, day, [r[1] for r in rows if r[1] is not None])
            tab = {str(r[0]): (fp_by_nba.get(r[1]), r[2]) for r in rows}
        elif sid:
            caps = [r[0] for r in conn.execute(
                "SELECT DISTINCT captured_at FROM dfs_salaries WHERE slate_id=? ORDER BY "
                "captured_at", (str(sid),))]
            if caps:
                pick = max((c for c in caps if ts and c <= ts), default=caps[0])
                for r in conn.execute(
                        "SELECT site_player_id, fppg, salary, roster_slot FROM dfs_salaries WHERE "
                        "slate_id=? AND captured_at=? AND disabled=0", (str(sid), pick)):
                    if r[0] is None:
                        continue
                    if r[0] not in tab or r[3] != "CPT":
                        tab[str(r[0])] = (r[1], r[2])
        cache[key] = tab
    hit = cache[key].get(dk)
    if hit is None:
        return None, None
    return _f(hit[0]), (None if hit[1] is None else int(hit[1]))


# ============================================================================== snapshots

def _season_first_day(day: date) -> date:
    y = day.year if day.month >= 8 else day.year - 1
    return date(y, 8, 1)


def window_days(day: date, win: str) -> tuple[date, date]:
    n = WINDOW_DAYS[win]
    return (_season_first_day(day) if n is None else day - timedelta(days=n - 1)), day


def _cl_mean(vals, clusters, min_clusters=MIN_DAYS_SE) -> tuple[float | None, float | None, int]:
    """(mean, clustered SE or None, n clusters)."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    v = v[ok]
    cl = np.asarray(clusters)[ok]
    if not len(v):
        return None, None, 0
    pc = MT.paired_cluster(v, cl)
    se = pc["se"] if pc["n_clusters"] >= min_clusters and np.isfinite(pc["se"]) else None
    return _f(pc["mean"]), _f(se), int(pc["n_clusters"])


def _lineup_frame(conn, lo: date, hi: date) -> pd.DataFrame:
    rows = []
    for r in conn.execute("SELECT lineup_id, slate_id, actual, projected, hindsight, "
                          "ratio_hindsight, pct_random, would_cash, cash_line_source, "
                          "detail_json FROM dfs_lineup_grades"):
        det = _loads(r["detail_json"], {}) or {}
        dy = det.get("day")
        if not dy or not (lo.isoformat() <= dy <= hi.isoformat()):
            continue
        rows.append({"lineup_id": r["lineup_id"], "slate_id": r["slate_id"], "day": dy,
                     "actual": r["actual"], "projected": r["projected"],
                     "hindsight": r["hindsight"], "ratio": r["ratio_hindsight"],
                     "pct_random": r["pct_random"], "would_cash": r["would_cash"],
                     "src": r["cash_line_source"], "effective": bool(det.get("effective")),
                     "role": det.get("role"), "gt": det.get("game_type", "classic")})
    return pd.DataFrame(rows)


def _slate_frame(conn, lo: date, hi: date) -> pd.DataFrame:
    rows = []
    for r in conn.execute("SELECT as_of, model, details_json FROM dfs_metric_snapshots WHERE "
                          "model LIKE 'slate:%' AND metric='baselines' AND as_of BETWEEN ? AND ?",
                          (lo.isoformat(), hi.isoformat())):
        s = _loads(r["details_json"], {}) or {}
        f, sa = s.get("fppg") or {}, s.get("salary") or {}
        rows.append({"slate_id": s.get("slate_id"), "day": r["as_of"],
                     "gt": s.get("game_type", "classic"),
                     "hindsight": (s.get("hindsight") or {}).get("actual"),
                     "fppg": f.get("actual"), "fppg_ratio": f.get("ratio_hindsight"),
                     "fppg_pct": f.get("pct_random"), "salary": sa.get("actual"),
                     "salary_ratio": sa.get("ratio_hindsight"), "salary_pct": sa.get("pct_random"),
                     "random": (s.get("random") or {}).get("mean")})
    return pd.DataFrame(rows)


def _entries(conn, lo: date, hi: date) -> pd.DataFrame:
    if not _has_table(conn, "dfs_entries"):
        return pd.DataFrame()
    rows = []
    for r in conn.execute(
            "SELECT e.entry_id, e.entry_fee, e.payout, e.source, e.created_at, e.contest_id, "
            "s.nba_day FROM dfs_entries e LEFT JOIN dfs_slates s ON s.slate_id=e.slate_id"):
        dy = r["nba_day"] or (r["created_at"] or "")[:10]
        if not dy or not (lo.isoformat() <= dy <= hi.isoformat()):
            continue
        rows.append({"fee": r["entry_fee"] or 0.0, "payout": r["payout"], "day": dy,
                     "source": (r["source"] or "").lower()})
    return pd.DataFrame(rows)


def lineup_snapshot_rows(conn, day: date, win: str) -> list[tuple]:
    """(model, metric, n, value, se, details) rows of dfs_metric_snapshots for one window."""
    lo, hi = window_days(day, win)
    L = _lineup_frame(conn, lo, hi)
    S = _slate_frame(conn, lo, hi)
    out: list[tuple] = []
    base_info = {"window": win, "from": lo.isoformat(), "to": hi.isoformat()}
    if len(L):
        L = L[L["effective"] & L["actual"].notna()]
    for gt, prefix in (("classic", ""), ("showdown", "sd_")):
        Lg = L[L["gt"] == gt] if len(L) else L
        Sg = S[S["gt"] == gt] if len(S) else S
        if len(Lg) and len(Sg):
            Lg = Lg.merge(Sg[["slate_id", "day", "fppg", "salary"]], on=["slate_id", "day"],
                          how="left", suffixes=("", "_b"))
        for role, g in (Lg.groupby("role") if len(Lg) else []):
            model = f"{prefix}{role}"
            n_days = int(g["day"].nunique())
            for metric, col in (("actual_mean", "actual"), ("ratio_hindsight", "ratio"),
                                ("pct_random", "pct_random")):
                v, se, ncl = _cl_mean(g[col].to_numpy(float), g["day"].to_numpy())
                if v is not None:
                    out.append((model, metric, int(g[col].notna().sum()), v, se,
                                {**base_info, "n_days": n_days, "n_slates": int(g["slate_id"].nunique())}))
            bias = (g["actual"] - g["projected"]).to_numpy(float)
            v, se, _ = _cl_mean(bias, g["day"].to_numpy())
            if v is not None:
                out.append((model, "actual_minus_projected", int(np.isfinite(bias).sum()), v, se,
                            base_info))
            for other in ("fppg", "salary"):
                if other not in g:
                    continue
                m = g[other].notna()
                if not m.any():
                    continue
                d = (g.loc[m, "actual"] - g.loc[m, other]).to_numpy(float)
                v, se, ncl = _cl_mean(d, g.loc[m, "day"].to_numpy())
                z = v / se if v is not None and se else None
                out.append((model, f"diff_vs_{other}", int(m.sum()), v, se,
                            {**base_info, "n_days": int(g.loc[m, "day"].nunique()),
                             "win_rate": _f((d > 0).mean()), "tie_rate": _f((d == 0).mean()),
                             "z": _f(z), "se_note": f"SE clustered by day, needs >= {MIN_DAYS_SE} days"}))
            rc = g[g["src"] == "real"]
            if len(rc) and rc["would_cash"].notna().any():
                rcm = rc["would_cash"].dropna().astype(float)
                out.append((model, "would_cash_rate", int(len(rcm)), _f(rcm.mean()), None,
                            {**base_info, "source": "real"}))
        if len(Sg):
            for name, col, rcol, pcol in (("fppg", "fppg", "fppg_ratio", "fppg_pct"),
                                          ("salary", "salary", "salary_ratio", "salary_pct"),
                                          ("hindsight", "hindsight", None, None),
                                          ("random", "random", None, None)):
                model = f"{prefix}{name}"
                for metric, c in (("actual_mean", col), ("ratio_hindsight", rcol),
                                  ("pct_random", pcol)):
                    if c is None or c not in Sg:
                        continue
                    v, se, _ = _cl_mean(Sg[c].to_numpy(float), Sg["day"].to_numpy())
                    if v is not None:
                        out.append((model, metric, int(Sg[c].notna().sum()), v, se,
                                    {**base_info, "n_days": int(Sg["day"].nunique())}))
    E = _entries(conn, lo, hi)
    if len(E):
        real = E[~E["source"].isin(SIMULATED_SOURCES)]
        if len(real):
            fees = float(real["fee"].sum())
            paid = real["payout"].dropna()
            wins = float(paid.sum())
            net = wins - float(real.loc[paid.index, "fee"].sum())
            det = {**base_info, "source": "real", "pending_entries": int(real["payout"].isna().sum()),
                   "cashed": int((paid > 0).sum())}
            out += [("real", "entries", int(len(real)), float(len(real)), None, det),
                    ("real", "fees", int(len(real)), fees, None, det),
                    ("real", "winnings", int(len(paid)), wins, None, det),
                    ("real", "net", int(len(paid)), net, None, det)]
            if float(real.loc[paid.index, "fee"].sum()) > 0:
                out.append(("real", "roi", int(len(paid)),
                            net / float(real.loc[paid.index, "fee"].sum()), None, det))
    return out


def _player_frame(conn, lo: date, hi: date) -> pd.DataFrame:
    q = ("SELECT g.projection_id, g.game_id, g.player_id, g.played, g.actual_min, g.actual_dk, "
         "g.proj_min, g.proj_dk, g.pit, g.in80, g.in50, g.crps, g.dk_fppg, g.salary, "
         "pp.nba_day AS day, r.kind, r.created_at, s.mean AS proj_min_played "
         "FROM player_projection_grades g JOIN player_projections pp ON pp.id=g.projection_id "
         "LEFT JOIN dfs_runs r ON r.run_id=g.run_id "
         "LEFT JOIN player_projection_stats s ON s.projection_id=g.projection_id AND "
         "s.stat='min' AND s.cond='played' WHERE pp.nba_day BETWEEN ? AND ?")
    df = pd.read_sql_query(q, conn, params=(lo.isoformat(), hi.isoformat()))
    if df.empty:
        return df
    return _one_per_player(df)


def _one_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, player): lock10 > lock45 > manual > swap, newest first. Manual builds
    (often earlier, an older model, players the lock runs later dropped) count only on nights
    with no scheduled run (the weekly report's rule, proj/reports.dedupe)."""
    manual = ~df["kind"].isin(("lock10", "lock45", "swap"))
    sched_days = set(df.loc[~manual, "day"])
    df = df[~manual | ~df["day"].isin(sched_days)].copy()
    df["prio"] = df["kind"].map(KIND_PRIORITY).fillna(2)
    df = df.sort_values(["prio", "created_at"], ascending=[True, False])
    return df.drop_duplicates(["game_id", "player_id"]).reset_index(drop=True)


def player_snapshot_rows(conn, day: date, win: str, cache: dict | None = None) -> list[tuple]:
    """(model, metric, n, value, se, details) rows of player_metric_snapshots for one window.
    Population: one row per (game, player) (lock10 run preferred; manual runs only on nights
    with no scheduled run), projected minutes >= 10,
    DNP scored 0, never filtered on actual minutes (PLAYERS.md §7.2). 'fppg' and 'salary' are
    scored on their own available rows; the paired gains use the rows all three have. Positive
    bias = over-projection. SEs are clustered by game (>= 5 games)."""
    lo, hi = window_days(day, win)
    df = _player_frame(conn, lo, hi)
    out: list[tuple] = []
    if df.empty:
        return out
    info = {"window": win, "from": lo.isoformat(), "to": hi.isoformat(),
            "population": "projected minutes >= 10, DNP = 0"}
    df = df[(df["proj_min"] >= 10) & df["proj_dk"].notna()].copy()
    if df.empty:
        return out
    a, b, how = salary_line(conn, day, cache)
    df["sal_dk"] = a + b * df["salary"]
    y = df["actual_dk"].to_numpy(float)
    cl = df["game_id"].to_numpy()

    def mae_row(model, pred, mask=None):
        m = np.ones(len(df), bool) if mask is None else mask
        e = pred[m] - y[m]
        if not m.any():
            return
        v, se, ncl = _cl_mean(np.abs(e), cl[m], MIN_DAYS_SE)
        out.append((model, "dk_mae", int(m.sum()), v, se, {**info, "n_games": ncl}))
        v, se, _ = _cl_mean(e, cl[m], MIN_DAYS_SE)
        out.append((model, "dk_bias", int(m.sum()), v, se, info))

    mae_row("ours", df["proj_dk"].to_numpy(float))
    has_f = df["dk_fppg"].notna().to_numpy()
    has_s = df["sal_dk"].notna().to_numpy()
    mae_row("fppg", df["dk_fppg"].to_numpy(float), has_f)
    mae_row("salary", df["sal_dk"].to_numpy(float), has_s)
    ours = df["proj_dk"].to_numpy(float)
    for name, col, mask in (("fppg", "dk_fppg", has_f), ("salary", "sal_dk", has_s)):
        if mask.sum() >= 2:
            g = MT.gain_pct(np.abs(df.loc[mask, col].to_numpy(float) - y[mask]),
                            np.abs(ours[mask] - y[mask]), cl[mask])
            ncl = g["n_clusters"]
            out.append(("ours", f"dk_mae_gain_vs_{name}", int(g["n"]), _f(g["gain_pct"]),
                        _f(g["se_pct"]) if ncl >= MIN_DAYS_SE else None,
                        {**info, "unit": "% MAE gain over baseline, paired rows", "n_games": ncl,
                         "mae_ours": _f(g["mae_model"]), "mae_base": _f(g["mae_base"])}))
    cov = df[df["in80"].notna()]
    if len(cov):
        v, se, ncl = _cl_mean(cov["in80"].to_numpy(float), cov["game_id"].to_numpy())
        out.append(("ours", "dk_cov80", len(cov), v, se, {**info, "target": 0.8}))
    cov = df[df["in50"].notna()]
    if len(cov):
        v, se, _ = _cl_mean(cov["in50"].to_numpy(float), cov["game_id"].to_numpy())
        out.append(("ours", "dk_cov50", len(cov), v, se, {**info, "target": 0.5}))
    c = df[df["crps"].notna()]
    if len(c):
        v, se, _ = _cl_mean(c["crps"].to_numpy(float), c["game_id"].to_numpy())
        out.append(("ours", "dk_crps", len(c), v, se, info))
    p = df[df["pit"].notna()]
    if len(p):
        u = p["pit"].to_numpy(float)
        hist = MT.pit_histogram(u, 10)
        out.append(("ours", "dk_pit_cov80", len(p), _f(((u >= 0.1) & (u <= 0.9)).mean()), None,
                    {**info, "target": 0.8, "pit_deciles": [round(float(h), 4) for h in hist]}))
    m = df[df["actual_min"].notna() & df["proj_min"].notna()]
    if len(m):
        e = (m["proj_min"] - m["actual_min"]).to_numpy(float)
        v, se, _ = _cl_mean(np.abs(e), m["game_id"].to_numpy())
        out.append(("ours", "min_mae", len(m), v, se, {**info, "note": "unconditional minutes"}))
        v, se, _ = _cl_mean(e, m["game_id"].to_numpy())
        out.append(("ours", "min_bias", len(m), v, se, info))
    mp = df[(df["played"] == 1) & (df["proj_min_played"] >= 15)]
    if len(mp):
        e = np.abs(mp["proj_min_played"] - mp["actual_min"]).to_numpy(float)
        v, se, _ = _cl_mean(e, mp["game_id"].to_numpy())
        out.append(("ours", "min_mae_played15", len(mp), v, se,
                    {**info, "note": "played, projected minutes when playing >= 15"}))
    return out


def write_snapshots(conn, day: date, cache: dict | None = None) -> dict:
    """Rewrite the as_of = `day` rows of both snapshot tables for all four windows."""
    d = day.isoformat()
    counts = {"dfs": 0, "player": 0}
    with conn:
        conn.execute("DELETE FROM dfs_metric_snapshots WHERE as_of=? AND model NOT LIKE 'slate:%'",
                     (d,))
        conn.execute("DELETE FROM player_metric_snapshots WHERE as_of=?", (d,))
        for win in WINDOWS:
            for model, metric, n, v, se, det in lineup_snapshot_rows(conn, day, win):
                conn.execute("INSERT OR REPLACE INTO dfs_metric_snapshots (as_of, model, metric, "
                             "\"window\", n, value, se, details_json) VALUES (?,?,?,?,?,?,?,?)",
                             (d, model, metric, win, n, _f(v), _f(se), _dumps(det)))
                counts["dfs"] += 1
            for model, metric, n, v, se, det in player_snapshot_rows(conn, day, win, cache):
                conn.execute("INSERT OR REPLACE INTO player_metric_snapshots (as_of, model, "
                             "metric, \"window\", n, value, se, details_json) "
                             "VALUES (?,?,?,?,?,?,?,?)",
                             (d, model, metric, win, n, _f(v), _f(se), _dumps(det)))
                counts["player"] += 1
    return counts


# ============================================================================== entry points

def grade_day(conn: sqlite3.Connection, day, *, force: bool = False, n_random: int = N_RANDOM
              ) -> dict:
    """Grade an NBA day: players, lineups (per slate, once all its games are final), baselines
    and snapshots. No-op (status 'skipped', nothing written) without runs or projections; a
    day whose games are not final yet writes nothing either. Idempotent; `force` re-grades.
    Raises RuntimeError after finishing everything else if a slate failed."""
    t0 = time.time()
    day = _as_date(day)
    slates = day_slates(conn, day)
    if not slates:
        return {"status": "skipped", "reason": "no dfs runs or projections", "day": day.isoformat()}
    now = _now()
    cache: dict = {}
    res: dict = {"day": day.isoformat(), "slates": [], "errors": []}
    try:
        res["players"] = grade_players(conn, day, force=force, now=now)
    except Exception as e:  # noqa: BLE001
        log.exception("player grading failed")
        res["players"] = {"error": str(e)}
        res["errors"].append(f"players: {type(e).__name__}: {e}")
    for s in slates:
        try:
            res["slates"].append(grade_slate(conn, day, s, force=force, n_random=n_random,
                                             now=now, cache=cache))
        except Exception as e:  # noqa: BLE001
            log.exception("grading slate %s failed", s["slate_id"])
            res["slates"].append({"slate_id": s["slate_id"], "status": "error", "error": str(e)})
            res["errors"].append(f"slate {s['slate_id']}: {type(e).__name__}: {e}")
    new = (res["players"].get("graded", 0) > 0
           or any(s["status"] == "graded" for s in res["slates"]))
    graded = res["players"].get("graded", 0) + sum(s.get("lineups", 0) for s in res["slates"]
                                                    if s["status"] == "graded")
    pend = (res["players"].get("pending", 0) > 0
            or any(s["status"] == "pending" for s in res["slates"]))
    res["status"] = "ok" if graded else ("pending" if pend else "cached")
    missing = res["status"] == "cached" and not _has_snapshots(conn, day) and (
        any(s["status"] == "cached" for s in res["slates"])
        or res["players"].get("projections", 0) > res["players"].get("pending", 0))
    if new or force or missing:
        try:
            res["snapshots"] = write_snapshots(conn, day, cache)
        except Exception as e:  # noqa: BLE001
            log.exception("snapshots failed")
            res["errors"].append(f"snapshots: {type(e).__name__}: {e}")
    res["seconds"] = round(time.time() - t0, 2)
    if res["errors"]:
        raise RuntimeError("; ".join(res["errors"]))
    return res


def _has_snapshots(conn, day: date) -> bool:
    d = day.isoformat()
    return bool(conn.execute("SELECT 1 FROM player_metric_snapshots WHERE as_of=? LIMIT 1",
                             (d,)).fetchone() or conn.execute(
        "SELECT 1 FROM dfs_metric_snapshots WHERE as_of=? AND model NOT LIKE 'slate:%' LIMIT 1",
        (d,)).fetchone())


def grade_recent(conn: sqlite3.Connection, day, *, lookback: int = CATCHUP_DAYS,
                 force: bool = False) -> dict:
    """`grade_day(day)` plus a retry of the `lookback` days before it (games that were not final
    or boxed at the previous run). Quiet days cost a few queries. Raises RuntimeError at the end
    if any day had a grading error."""
    day = _as_date(day)
    errors = []
    try:
        res = grade_day(conn, day, force=force)
    except RuntimeError as e:
        res, errors = {"day": day.isoformat(), "status": "error"}, [str(e)]
    prior = {}
    for k in range(1, lookback + 1):
        d = day - timedelta(days=k)
        try:
            r = grade_day(conn, d)
        except RuntimeError as e:
            prior[d.isoformat()] = {"status": "error", "error": str(e)}
            errors.append(f"{d}: {e}")
            continue
        if r["status"] == "ok":
            prior[d.isoformat()] = {"status": r["status"], "players": r.get("players"),
                                    "slates": [s["status"] for s in r.get("slates", [])]}
    if prior:
        res["catch_up"] = prior
    if errors:
        raise RuntimeError("; ".join(errors))
    return res


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("grade", help="grade a day's stored DFS builds against the box scores")
    p.add_argument("--date", default=None, help="NBA day (default: yesterday)")
    p.add_argument("--force", action="store_true", help="re-grade although already graded")
    p = sub.add_parser("report", help="print the latest DFS tracking snapshot table")
    p.add_argument("--window", default="7d", choices=WINDOWS)
    p.add_argument("--as-of", default=None, help="snapshot day (default: the latest)")
    return {"grade": _grade_cmd, "report": _report_cmd}


def _conn():
    from ... import db
    return db.connect()


def _grade_cmd(args) -> int:
    day = _as_date(args.date) if args.date else config.yesterday_nba_day()
    conn = _conn()
    try:
        res = grade_day(conn, day, force=args.force)
    except RuntimeError as e:
        print(f"grade {day}: ERROR {e}")
        return 1
    print(json.dumps(res, indent=2, default=str))
    return 0


def _fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _report_cmd(args) -> int:
    conn = _conn()
    as_of = args.as_of or (conn.execute(
        "SELECT MAX(as_of) FROM dfs_metric_snapshots").fetchone()[0])
    if not as_of:
        print("no DFS snapshots yet (nothing graded)")
        return 0
    print(f"DFS tracking as of {as_of}, window {args.window}  "
          "(simulated builds vs actual results; 'real' = the owner's own entries)")
    rows = conn.execute("SELECT model, metric, n, value, se FROM dfs_metric_snapshots WHERE "
                        "as_of=? AND \"window\"=? AND model NOT LIKE 'slate:%' "
                        "ORDER BY model, metric", (as_of, args.window)).fetchall()
    print("\nLineups")
    print(f"{'model':<14}{'metric':<24}{'n':>6}{'value':>12}{'se':>10}")
    for r in rows:
        print(f"{r['model']:<14}{r['metric']:<24}{r['n'] or 0:>6}{_fmt(r['value']):>12}"
              f"{_fmt(r['se']):>10}")
    prow = conn.execute("SELECT model, metric, n, value, se FROM player_metric_snapshots WHERE "
                        "as_of=? AND \"window\"=? ORDER BY model, metric",
                        (as_of, args.window)).fetchall()
    print("\nPlayers (DK points, projected minutes >= 10, DNP = 0)")
    print(f"{'model':<14}{'metric':<24}{'n':>6}{'value':>12}{'se':>10}")
    for r in prow:
        print(f"{r['model']:<14}{r['metric']:<24}{r['n'] or 0:>6}{_fmt(r['value']):>12}"
              f"{_fmt(r['se']):>10}")
    print(f"\nSlates on {as_of}: ours (in effect at lock) vs FPPG vs salary-implied vs hindsight")
    print(f"{'slate':<16}{'type':<9}{'cash':>8}{'top1':>8}{'fppg':>8}{'salary':>8}{'hindsight':>10}"
          f"{'random':>8}")
    for r in conn.execute("SELECT details_json FROM dfs_metric_snapshots WHERE as_of=? AND "
                          "model LIKE 'slate:%' AND metric='baselines' ORDER BY model", (as_of,)):
        s = _loads(r["details_json"], {})
        o = {(x["role"], x["rank"]): x["actual"] for x in s.get("ours", [])}
        print(f"{s['slate_id']:<16}{s.get('game_type', ''):<9}"
              f"{_fmt(o.get(('cash', 1)) or o.get(('cash', None)), 1):>8}"
              f"{_fmt(o.get(('topn', 1)), 1):>8}"
              f"{_fmt((s.get('fppg') or {}).get('actual'), 1):>8}"
              f"{_fmt((s.get('salary') or {}).get('actual'), 1):>8}"
              f"{_fmt((s.get('hindsight') or {}).get('actual'), 1):>10}"
              f"{_fmt((s.get('random') or {}).get('mean'), 1):>8}")
    return 0
