"""Research columns for the DFS slate page's player pool (`web/dfs_view.py`): boom % and
bust %, leverage against the estimated popularity, and DK points against salary.

Read-only and display-only: everything here reads what a run already produced (its sims, its
stored quantiles, its lineups and its popularity proxy) or past box scores. No model logic.
Information only: these numbers describe our simulations and past games; none says what to do.

**Boom % / bust %** (per DK player, *if he plays*):
  boom = P(DK points >= ``BOOM_K`` x salary / 1000), bust = P(DK points < ``BUST_K`` x salary /
  1000). The boom mark is the X factor's big-night mark (`proj.dfs.xfactor.BIG_K`).
  * From the run's sims when the file is readable (`dfs_runs.sim_path`, npz with ``fp`` S x n,
    ``dk_id`` and ``min_t`` n x S): the share of sims in which he played (min_t > 0) at or above
    / below the mark. Raw sim shares (the big-night chance column is the recalibrated,
    all-games version of the boom number).
  * Otherwise from the stored DK quantiles if he plays (`player_projection_stats`, stat 'dk',
    cond 'played', q05..q95): linear interpolation of the CDF between the quantiles, and an
    exponential tail beyond q05 / q95 fitted to the last two quantiles (the chance halves every
    q95 - q90 points above q95, every q10 - q05 below q05). An approximation; the page says so.

**Leverage** = his share of the run's Top N lineups minus his estimated popularity
(`dfs_ownership_proxy.own`, percent of entries), in percentage points. Signed: positive means
he is in more of our Top N lineups than the proxy expects the field to play him.

**DK points vs salary** ("value +/-"): his mean DK points in his last ``LAST_N`` games played
(this season and last, same as the box scores) minus the salary-implied DK points at tonight's
salary. The salary-implied line is an OLS fit of actual DK points on salary per season over
every graded Classic player-game with a salary (`player_projection_grades`, played, before the
day; one row per player-game), the same line as `proj.metrics.salary_implied` but fit on the
season instead of one slate. The season figure (this season's games only) rides in the tooltip.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import db as sdb
from . import seasons as S

BOOM_K = 6.0                 # boom: DK >= 6 x salary/1000 (xfactor.BIG_K, the big-night mark)
BUST_K = 3.0                 # bust: DK < 3 x salary/1000
LAST_N = 10
FIT_MIN_ROWS = 50            # player-games needed for a season's salary line
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
Q_COLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
_LN2 = math.log(2.0)


# ------------------------------------------------------------------ boom / bust

def tail_above(qv: Sequence[float | None], t: float) -> float | None:
    """P(X >= t) from quantiles at QS (piecewise-linear CDF, exponential tails)."""
    p_below = cdf(qv, t)
    return None if p_below is None else 1.0 - p_below


def cdf(qv: Sequence[float | None], t: float) -> float | None:
    """P(X < t) from the values at QS: linear between quantiles, exponential beyond the ends
    (P halves every q10 - q05 points below q05 and every q95 - q90 points above q95). None
    when any quantile is missing."""
    if qv is None or len(qv) != len(QS) or any(v is None for v in qv):
        return None
    q = np.maximum.accumulate(np.asarray(qv, float))
    if not np.all(np.isfinite(q)):
        return None
    lo, hi = q[0], q[-1]
    if t <= lo:
        s = (q[1] - q[0]) / _LN2
        if s <= 0:
            return 0.0 if t < lo else QS[0]
        return QS[0] * math.exp(-(lo - t) / s)
    if t >= hi:
        s = (q[-1] - q[-2]) / _LN2
        if s <= 0:
            return 1.0 if t > hi else QS[-1]
        return 1.0 - (1.0 - QS[-1]) * math.exp(-(t - hi) / s)
    # inside: q is non-decreasing; for flat runs take the right-most matching level
    j = int(np.searchsorted(q, t, side="right"))     # q[j-1] <= t < q[j]
    x0, x1 = q[j - 1], q[j]
    p0, p1 = QS[j - 1], QS[j]
    return float(p0 + (p1 - p0) * (t - x0) / (x1 - x0)) if x1 > x0 else float(p1)


def marks(salary: float) -> tuple[float, float]:
    """(boom mark, bust mark) in DK points for a salary."""
    return BOOM_K * float(salary) / 1000.0, BUST_K * float(salary) / 1000.0


def boom_bust_quantiles(qv: Sequence[float | None], salary: float | None
                        ) -> tuple[float | None, float | None]:
    if not salary or salary <= 0:
        return None, None
    b_hi, b_lo = marks(salary)
    up = tail_above(qv, b_hi)
    dn = cdf(qv, b_lo)
    return up, dn


def boom_bust_sims(fp: np.ndarray, played: np.ndarray | None, salary: float | None
                   ) -> tuple[float | None, float | None, int]:
    """(boom, bust, sims in which he played) from one player's simulated DK points."""
    if not salary or salary <= 0:
        return None, None, 0
    x = np.asarray(fp, np.float32)
    if played is not None:
        x = x[np.asarray(played, bool)]
    n = int(x.size)
    if n == 0:
        return None, None, 0
    b_hi, b_lo = marks(salary)
    return (float((x >= np.float32(b_hi)).mean()), float((x < np.float32(b_lo)).mean()), n)


_SIM_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_SIM_LOCK = threading.Lock()
_SIM_MAX = 8


def resolve_sim_path(path: str | None, slate_id: str | None = None) -> Path | None:
    """The run's sim file: the stored path, else the same file name under this machine's
    sims dir (a database copied from another machine keeps the other machine's paths)."""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    try:
        from ..proj.store import sims_dir
        alt = sims_dir() / str(slate_id or p.parent.name) / p.name
    except Exception:  # noqa: BLE001
        return None
    return alt if alt.exists() else None


def sim_boom_bust(conn: sqlite3.Connection, run_id: str | None,
                  salaries: Mapping[str, float]) -> dict[str, tuple]:
    """{dk id: (boom, bust, n played sims)} from the run's sims; {} when the run has no
    readable sims. Cached per (run, file, mtime, salaries)."""
    if not run_id:
        return {}
    try:
        r = conn.execute("SELECT sim_path, slate_id FROM dfs_runs WHERE run_id=?",
                         (run_id,)).fetchone()
    except sqlite3.Error:
        return {}
    if r is None:
        return {}
    p = resolve_sim_path(r[0], r[1])
    if p is None:
        return {}
    try:
        key = (run_id, str(p), p.stat().st_mtime_ns,
               tuple(sorted((str(k), float(v)) for k, v in salaries.items() if v)))
    except OSError:
        return {}
    with _SIM_LOCK:
        hit = _SIM_CACHE.get(key)
    if hit is not None:
        return hit
    out: dict[str, tuple] = {}
    try:
        with np.load(p, allow_pickle=False) as z:
            if "fp" not in z.files or "dk_id" not in z.files:
                return {}
            dk = [str(x) for x in z["dk_id"]]
            cols = {}
            for j, d in enumerate(dk):
                if d in salaries and d not in cols:
                    cols[d] = j
            if cols:
                idx = list(cols.values())
                fp = np.asarray(z["fp"][:, idx], np.float32)
                mt = np.asarray(z["min_t"][idx, :]) if "min_t" in z.files else None
                for k, (d, _j) in enumerate(cols.items()):
                    played = (mt[k] > 0) if mt is not None else None
                    out[d] = boom_bust_sims(fp[:, k], played, salaries.get(d))
    except (OSError, ValueError, KeyError):
        return {}
    with _SIM_LOCK:
        _SIM_CACHE[key] = out
        while len(_SIM_CACHE) > _SIM_MAX:
            _SIM_CACHE.popitem(last=False)
    return out


# ------------------------------------------------------------------ leverage

def lineup_shares(lineups: Iterable[Sequence[Mapping]]) -> tuple[dict[str, float], int]:
    """({dk id: share of lineups holding him}, number of lineups)."""
    n, cnt = 0, {}
    for slots in lineups:
        n += 1
        for pid in {str(s.get("player_id")) for s in slots if isinstance(s, Mapping)}:
            cnt[pid] = cnt.get(pid, 0) + 1
    return ({k: v / n for k, v in cnt.items()} if n else {}), n


def ownership(conn: sqlite3.Connection, run_ids: Sequence[str | None]) -> tuple[dict, str | None]:
    """({dk id: est. popularity in percent}, run id used): the first run with proxy rows."""
    for rid in run_ids:
        if not rid:
            continue
        try:
            rows = conn.execute("SELECT player_id, own FROM dfs_ownership_proxy WHERE run_id=?",
                                (rid,)).fetchall()
        except sqlite3.Error:
            return {}, None
        if rows:
            return {str(r[0]): float(r[1]) for r in rows if r[1] is not None}, rid
    return {}, None


def leverage(share: float | None, own_pct: float | None) -> float | None:
    """Top N share (0..1) minus estimated popularity (percent), in percentage points."""
    if own_pct is None:
        return None
    return 100.0 * (share or 0.0) - float(own_pct)


# ------------------------------------------------------------------ DK points vs salary

def _season_of_game(game_id: str) -> str | None:
    """'0022600104' -> '2026-27' (NBA game ids carry the season's start year)."""
    g = str(game_id or "")
    if len(g) < 5 or not g[3:5].isdigit():
        return None
    return S.from_start_year(2000 + int(g[3:5]))


_FIT_SQL = """
SELECT g.game_id, g.salary, g.actual_dk
FROM player_projection_grades g
JOIN (SELECT MAX(gr.projection_id) AS pid FROM player_projection_grades gr
      JOIN player_projections p ON p.id = gr.projection_id
      WHERE gr.salary > 0 AND gr.actual_dk IS NOT NULL AND gr.played = 1 AND p.nba_day < ?
      GROUP BY gr.game_id, gr.player_id) k ON k.pid = g.projection_id
"""


def compute_salary_lines(conn: sqlite3.Connection, before: str) -> dict[str, dict]:
    """{season: {a, b, n}} OLS of DK points on salary per season (rows before `before`)."""
    try:
        rows = conn.execute(_FIT_SQL, (before,)).fetchall()
    except sqlite3.Error:
        return {}
    by: dict[str, list] = {}
    for gid, sal, y in rows:
        s = _season_of_game(gid)
        if s:
            by.setdefault(s, []).append((float(sal), float(y)))
    out = {}
    for s, xy in by.items():
        if len(xy) < FIT_MIN_ROWS:
            continue
        x = np.array([a for a, _ in xy])
        y = np.array([b for _, b in xy])
        if np.ptp(x) <= 0:
            continue
        b, a = np.polyfit(x, y, 1)
        out[s] = {"a": float(a), "b": float(b), "n": len(xy)}
    return out


def salary_lines(conn: sqlite3.Connection, before: str) -> dict[str, dict]:
    return sdb.cached(("dfs.salary_lines", before),
                      lambda c: compute_salary_lines(c, before), conn, domain="dfs")


def expected_dk(line: Mapping | None, salary: float | None) -> float | None:
    if not line or not salary:
        return None
    return line["a"] + line["b"] * float(salary)


def compute_recent_dk(conn: sqlite3.Connection, ids: Sequence[int], day: str,
                      seasons: Sequence[str]) -> dict[int, dict]:
    """{player_id: {last: [dk, ...] newest first (<= LAST_N), last_seasons: [...], cur,
    season_mean, season_n}} from games played (min > 0) before `day`; `seasons[0]` is the
    current season."""
    from ..proj.dfs.rules import dk_points
    out: dict[int, dict] = {}
    if not ids:
        return out
    cur = seasons[0]
    for i in range(0, len(ids), 400):
        chunk = list(ids[i:i + 400])
        sql = ("SELECT player_id, season, game_date, pts, fg3m, reb, ast, stl, blk, tov "
               "FROM player_games WHERE player_id IN ({}) AND season IN ({}) "
               "AND season_type IN (?, ?) AND game_date < ? AND min > 0 "
               "ORDER BY player_id, game_date DESC").format(
                   ",".join("?" * len(chunk)), ",".join("?" * len(seasons)))
        try:
            rows = conn.execute(sql, (*chunk, *seasons, S.REGULAR, S.PLAYOFFS, day)).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            d = out.setdefault(int(r[0]), {"last": [], "last_seasons": [], "season": [],
                                           "cur": cur})
            dk = dk_points({"pts": r[3], "fg3m": r[4], "reb": r[5], "ast": r[6], "stl": r[7],
                            "blk": r[8], "tov": r[9]})
            if len(d["last"]) < LAST_N:
                d["last"].append(dk)
                d["last_seasons"].append(r[1])
            if r[1] == cur:
                d["season"].append(dk)
    for d in out.values():
        d["season_n"] = len(d["season"])
        d["season_mean"] = (sum(d["season"]) / len(d["season"])) if d["season"] else None
        del d["season"]
    return out


def recent_dk(conn: sqlite3.Connection, ids: Iterable[int], day: date) -> dict[int, dict]:
    cur = S.season_for(day)
    seasons = (cur, S.from_start_year(S.start_year(cur) - 1))
    ids = tuple(sorted({int(i) for i in ids}))
    return sdb.cached(("dfs.recent_dk", ids, day.isoformat(), seasons),
                      lambda c: compute_recent_dk(c, ids, day.isoformat(), seasons), conn,
                      domain="box")


def value_vs_salary(recent: Mapping | None, line: Mapping | None, salary: float | None
                    ) -> dict | None:
    """{pm, last_mean, last_n, last_prev (games from last season), expected, season_pm,
    season_mean, season_n} or None without a salary line or recent games."""
    exp = expected_dk(line, salary)
    if exp is None or not recent or not recent.get("last"):
        return None
    last = recent["last"]
    lm = sum(last) / len(last)
    cur = recent.get("cur")
    prev_n = sum(1 for s in recent.get("last_seasons") or [] if s != cur)
    sm = recent.get("season_mean")
    return {"pm": lm - exp, "last_mean": lm, "last_n": len(last), "last_prev": prev_n,
            "expected": exp, "season_mean": sm, "season_n": recent.get("season_n") or 0,
            "season_pm": None if sm is None else sm - exp}
