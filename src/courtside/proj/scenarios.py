"""Teammate-availability splits from a run's sims (docs/IMPROVEMENTS.md P10).

Every sim of a game draws who is available (`copula.draw_scenarios`); the sims npz keeps those
draws (`store.scenario_arrays` / `store.read_scenarios`). For a player and each **high-leverage
teammate** (LO < P(teammate available) < HI in the sims) this splits the sims into the two arms
"teammate plays" (available) and "teammate sits" (drawn out) and gives per arm:

* `n`         sims in the arm;
* `dk`        mean DK points over all sims of the arm (his own DNP mass included, like `dk_all`);
* `dk_played` mean DK points when he plays; `min_played` mean minutes when he plays; `min` mean
  minutes over all sims; `p_play` share of the arm's sims he plays;
* `big_raw`   share of the arm's sims with DK >= `xfactor.BIG_K` x salary/1000 (exactly the X
  factor's big night, `xfactor.raw_chances`; it contains his own DNP mass), only with a salary;
* `big`       `big_raw` through the X factor's per-tier recalibration (`xfactor.calibrate`,
  fit on unconditional chances: a calibrated estimate for the arm, not a separately fitted one).

"Plays" for the player himself = minutes > 0 in the sim (as the pipeline's `played`); the arms
are cut on the teammate's drawn availability (not his minutes: an available teammate can still
get no minutes). An arm with fewer than `MIN_N` sims is reported but `show` is False.

Public API
  player_splits(fp, sc, j, salary=None, tier=None) -> list[dict]   (one per lever teammate)
  run_splits(conn, run_id) -> {NBA player_id: [split, ...]}   (cached per sims file)
  card_splits(splits, k=CARD_MAX) -> the shown subset (both arms >= MIN_N, sorted by |dDK|)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np

from . import store as SO
from .dfs import xfactor as XF

log = logging.getLogger(__name__)

LO, HI = 0.05, 0.95          # high-leverage teammate: LO < P(available) < HI
MIN_N = 200                  # sims per arm to show a split
CARD_MAX = 2                 # teammates shown per player card
CARD_MIN_SHIFT = 1.0         # ... whose arms differ by >= 1 DK point or 1 minute (if he plays)

_CACHE: dict[tuple, dict] = {}
_CACHE_MAX = 16


def lever_teammates(sc: SO.Scenarios, j: int, lo: float = LO, hi: float = HI
                    ) -> list[tuple[int, float]]:
    """[(column, P(available) in the sims)] of j's teammates with lo < P < hi."""
    cols = np.flatnonzero((sc.team == sc.team[j]) & (np.arange(len(sc.team)) != j))
    if not len(cols):
        return []
    pa = 1.0 - sc.absent[:, cols].mean(0)
    return [(int(c), float(p)) for c, p in zip(cols, pa) if lo < p < hi]


def _arm(fp: np.ndarray, mins: np.ndarray, m: np.ndarray, mark: float | None,
         tier: str | None) -> dict:
    n = int(m.sum())
    out = {"n": n, "dk": None, "dk_played": None, "min": None, "min_played": None,
           "p_play": None, "big_raw": None, "big": None}
    if not n:
        return out
    f = fp[m].astype(np.float32)
    x = mins[m]
    pl = x > 0
    npl = int(pl.sum())
    out.update(dk=float(f.mean()), min=float(x.mean()), p_play=npl / n,
               dk_played=float(f[pl].mean()) if npl else None,
               min_played=float(x[pl].mean()) if npl else None)
    if mark is not None:
        raw = float((f >= np.float32(mark)).mean())
        out.update(big_raw=raw, big=float(XF.calibrate([raw], [tier])[0]))
    return out


def player_splits(fp: np.ndarray, sc: SO.Scenarios, j: int, salary: float | None = None,
                  tier: str | None = None, lo: float = LO, hi: float = HI,
                  min_n: int = MIN_N) -> list[dict]:
    """Per lever teammate k of column j: {"col": k, "p_avail", "plays": arm, "sits": arm,
    "show"} (module doc). `fp` (S, n) is the run's DK sims (same columns as `sc`)."""
    mark = (XF.big_mark(salary) if salary is not None and np.isfinite(salary) and salary > 0
            else None)
    fj, mj = np.asarray(fp[:, j]), sc.minutes[:, j]
    out = []
    for k, pa in lever_teammates(sc, j, lo, hi):
        sits = sc.absent[:, k]
        a, b = _arm(fj, mj, ~sits, mark, tier), _arm(fj, mj, sits, mark, tier)
        out.append({"col": k, "p_avail": pa, "plays": a, "sits": b,
                    "show": a["n"] >= min_n and b["n"] >= min_n})
    return out


def _shift(s: dict) -> float:
    a, b = s["plays"], s["sits"]
    d = [abs((b[key] or 0.0) - (a[key] or 0.0)) for key in ("dk_played", "min_played")]
    return max(d)


def card_splits(splits: list[dict], k: int = CARD_MAX,
                min_shift: float = CARD_MIN_SHIFT) -> list[dict]:
    """The splits a player card shows: both arms >= MIN_N sims and a shift of at least
    `min_shift` (DK points or minutes, if he plays), largest DK shift first, at most k."""
    ok = [s for s in splits if s["show"] and _shift(s) >= min_shift]
    ok.sort(key=lambda s: -abs((s["sits"]["dk"] or 0.0) - (s["plays"]["dk"] or 0.0)))
    return ok[:k]


# ============================================================================ live runs

def _run_meta(conn: sqlite3.Connection, run_id: str) -> tuple[str | None, bool]:
    r = conn.execute("SELECT r.sim_path, s.tag FROM dfs_runs r LEFT JOIN dfs_slates s "
                     "USING (slate_id) WHERE r.run_id=?", (run_id,)).fetchone()
    if r is None:
        return None, False
    return (r[0] or None), (r[1] or "").lower() == "showdown"


def run_splits(conn: sqlite3.Connection, run_id: str | None) -> dict[int, list[dict]]:
    """{NBA player_id: player_splits(...) with "teammate" (NBA id) per split} for every column
    of a run's sims; {} when the run has no sims or its sims carry no scenarios (runs stored
    before P10). Big-night chances need the player's DK Classic salary (none on Showdown)."""
    if not run_id:
        return {}
    try:
        path, showdown = _run_meta(conn, run_id)
    except sqlite3.Error:
        return {}
    if not path or not Path(path).exists():
        return {}
    try:
        key = (run_id, path, Path(path).stat().st_mtime_ns)
    except OSError:
        return {}
    if key in _CACHE:
        return _CACHE[key]
    try:
        z = SO.load_sims(path)
    except (OSError, ValueError):
        log.warning("scenarios: sims for run %s unreadable at %s", run_id, path)
        return {}
    sc = SO.read_scenarios(z)
    out: dict[int, list[dict]] = {}
    if sc is not None:
        pid = [int(x) for x in z["player_id"]]
        dk = [str(x) for x in z["dk_id"]] if "dk_id" in z else [""] * len(pid)
        sal = {} if showdown else XF._salaries(conn, run_id)
        tier = {d: t for d, (t, _) in XF._projection_info(conn, run_id).items()}
        fp = z["fp"]
        for j, p in enumerate(pid):
            if p in out:
                continue
            sp = player_splits(fp, sc, j, sal.get(dk[j]), tier.get(dk[j]))
            for s in sp:
                s["teammate"] = pid[s["col"]]
            out[p] = sp
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out
