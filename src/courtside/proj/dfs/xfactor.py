"""X factor: low-popularity bench/role players with a real chance of a big night (owner request
2026-10-21; research and every number below: docs/research/xfactor.md,
research/xfactor_calibration.py).

Definitions (per DK Classic player of a live projection run):

* **big night** = DK points >= ``BIG_K`` x salary/1000 (a fixed multiple of salary, not the
  slate-relative mark of PLAYERS.md §2.1: the fixed multiple is what the walk-forward backtest
  could calibrate; the slate-relative mark moves it by only about ±0.5x, see the doc).
* **big-night chance** = share of the run's sims (``dfs_runs.sim_path``, 25k per run) with the
  player at or above that mark, then **recalibrated**: the raw sims overstate the far upper tail
  of bench and role players (PIT > 0.99 in about 0.3-0.8 % of player-games vs 1 %), so the raw
  share goes through a per-tier logistic map logit p' = a + b logit p (``STRETCH``, fit on the
  RotoGuru seasons in the doc; walk-forward checked). The sims include the DNP mass, so the
  chance is unconditional (it already contains P(play)).
* **X factor** = model tier bench or role, chance >= ``P_MIN``, popularity proxy band "Low"
  (``dfs_ownership_proxy``, large-GPP archetype, < 10 % estimated), P(play) >= ``P_PLAY_MIN``,
  and not among the slate's ``VALUE_TOP_EXCLUDE`` best mean DK per $1k (orchestrator, 2026-10-21:
  the proxy is uncalibrated and runs low exactly for those players).

Computed at read time from the run's npz (no migration, works for every stored run) and cached
per (run_id, sim file mtime). Everything here is an estimate from our sims and a popularity
PROXY: information, not advice (STYLE.md).

Public API
  big_mark(salary, k=BIG_K) -> float
  raw_chances(fp, salaries, k=BIG_K) -> np.ndarray
  calibrate(p, tiers) -> np.ndarray
  chance_text(p) -> "1 in 5" (banded)
  is_xfactor(chance, band, p_play, tier) -> bool
  run_table(conn, run_id) -> {dk id: {"chance", "raw", "text", "band", "own", "tier", "p_play",
                                     "salary", "x"}}   (empty when the run has no sims)
  xfactor_ids(table) -> set[str]
  lineup_count(slots_lists, ids) -> int   (lineups with at least one X factor player)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

VERSION = "xf-v1"
BIG_K = 6.0                  # big night = DK >= 6 x salary/1000
P_MIN = 0.33                 # calibrated big-night chance needed for the tag (1 in 3)
P_PLAY_MIN = 0.85            # chance he plays
CAP = 5                      # at most this many tags per slate (highest chances)
OWN_BANDS = ("Low",)         # popularity proxy bands allowed (field.BANDS: Low < 10 %)
TIERS = ("bench", "role")    # model tiers (player_projections.role) allowed
# The slate's top value plays (mean DK per $1k, among players with P(play) >= 0.5) are left out:
# other projection sites find them too, so they tend to be popular whatever the (uncalibrated)
# proxy says. On 2026-10-21 the first version tagged value ranks 1/3/5/7/11 (proxy 3-6 %).
VALUE_TOP_EXCLUDE = 10
# logit p' = a + b * logit p per model tier, fit on RotoGuru 2015-16 .. 2020-21 (harness avail
# protocol with --tails, model pathA-v0.3+p2 code, P(play) >= 0.85 rows, n = 117k, k = 6 ONLY:
# refit if BIG_K changes); `research/xfactor_calibration.py bignight`, docs/research/xfactor.md.
# Walk-forward (fit on seasons < S) the same map leaves the tagged sets calibrated in 2016-21.
# Unknown tier: no change.
STRETCH: dict[str, tuple[float, float]] = {"bench": (-0.2902, 0.8275), "role": (-0.2890, 0.7789),
                                           "star": (-0.3186, 0.7724)}
LABEL = ("Chance of a big night from our simulations plus a low estimated popularity "
         "(proxy), leaving out the slate's 10 best projections per $1k (those tend to be popular). "
         "Estimates, not advice.")
_EPS = 1e-4
_ONE_IN = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)


def big_mark(salary: float, k: float = BIG_K) -> float:
    return float(k) * float(salary) / 1000.0


def raw_chances(fp: np.ndarray, salaries: Sequence[float], k: float = BIG_K) -> np.ndarray:
    """Share of sims (rows of `fp`, S x n) at or above k x salary/1000 per column; NaN where the
    salary is missing or not positive."""
    fp = np.asarray(fp)
    sal = np.asarray(salaries, float)
    out = np.full(sal.shape, np.nan)
    ok = np.isfinite(sal) & (sal > 0)
    if fp.size and ok.any():
        cols = np.flatnonzero(ok)
        marks = k * sal[cols] / 1000.0
        out[cols] = (fp[:, cols].astype(np.float32) >= marks[None, :].astype(np.float32)
                     ).mean(0)
    return out


def calibrate(p: Sequence[float], tiers: Sequence[str | None]) -> np.ndarray:
    """The per-tier logistic recalibration (``STRETCH``) of raw sim chances."""
    p = np.asarray(p, float)
    out = p.copy()
    ok = np.isfinite(p)
    z = np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
    for i, t in enumerate(tiers):
        if not ok[i] or t not in STRETCH:
            continue
        a, b = STRETCH[t]
        out[i] = 1.0 / (1.0 + math.exp(-(a + b * z[i])))
    return out


def chance_text(p: float | None) -> str:
    """Banded "1 in N" (N from a short ladder, nearest on the log scale)."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1 / 75:
        return "under 1 in 50"
    if p >= 0.6:
        return "better than 1 in 2"
    n = min(_ONE_IN, key=lambda x: abs(math.log(x) - math.log(1.0 / p)))
    return f"1 in {n}"


def is_xfactor(chance: float | None, band: str | None, p_play: float | None,
               tier: str | None) -> bool:
    return (chance is not None and np.isfinite(chance) and chance >= P_MIN
            and band in OWN_BANDS and p_play is not None and p_play >= P_PLAY_MIN
            and tier in TIERS)


# ============================================================================ live runs

_CACHE: dict[tuple, dict] = {}
_CACHE_MAX = 32


def _sim_path(conn: sqlite3.Connection, run_id: str) -> str | None:
    r = conn.execute("SELECT sim_path FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
    return r[0] if r and r[0] else None


def _salaries(conn: sqlite3.Connection, run_id: str) -> dict[str, float]:
    """DK id -> Classic salary from the run's slate, latest capture at or before the run (else
    the latest capture)."""
    r = conn.execute("SELECT slate_id, created_at FROM dfs_runs WHERE run_id=?",
                     (run_id,)).fetchone()
    if r is None:
        return {}
    sid, at = r[0], r[1]
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (sid, at)).fetchone()[0] \
        or conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=?",
                        (sid,)).fetchone()[0]
    out: dict[str, float] = {}
    for pid, sal, slot in conn.execute(
            "SELECT site_player_id, salary, roster_slot FROM dfs_salaries WHERE slate_id=? AND "
            "captured_at=?", (sid, cap)):
        if pid is None or sal is None or slot == "CPT":
            continue
        out.setdefault(str(pid), float(sal))
    return out


def _projection_info(conn: sqlite3.Connection, run_id: str) -> dict[str, tuple]:
    """DK id -> (tier, p_play) from the run's player_projections rows."""
    out: dict[str, tuple] = {}
    for role, p_play, inputs in conn.execute(
            "SELECT role, p_play, inputs_json FROM player_projections WHERE run_id=?",
            (run_id,)):
        try:
            dk = json.loads(inputs or "{}").get("dk_id")
        except (TypeError, ValueError):
            dk = None
        if dk is None or str(dk) in ("", "nan", "None"):
            continue
        out[str(dk)] = (role, None if p_play is None else float(p_play))
    return out


def _ownership(conn: sqlite3.Connection, run_id: str) -> dict[str, tuple]:
    try:
        rows = conn.execute("SELECT player_id, own, band FROM dfs_ownership_proxy WHERE run_id=?",
                            (run_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): (None if r[1] is None else float(r[1]), r[2]) for r in rows}


def run_table(conn: sqlite3.Connection, run_id: str | None) -> dict[str, dict]:
    """Per DK id of a run: big-night chance (raw and calibrated), its text band, popularity
    band, tier, P(play), salary and the X factor flag. {} when the run has no readable sims or
    is a Showdown slate (the tag is for Classic)."""
    if not run_id:
        return {}
    path = _sim_path(conn, run_id)
    if not path or not Path(path).exists():
        return {}
    try:
        key = (run_id, path, Path(path).stat().st_mtime_ns)
    except OSError:
        return {}
    if key in _CACHE:
        return _CACHE[key]
    r = conn.execute("SELECT s.tag FROM dfs_runs r JOIN dfs_slates s USING (slate_id) WHERE "
                     "r.run_id=?", (run_id,)).fetchone()
    if r is not None and (r[0] or "").lower() == "showdown":
        return {}
    from ..store import load_sims
    try:
        z = load_sims(path)
    except (OSError, ValueError):
        log.warning("xfactor: sims for run %s unreadable at %s", run_id, path)
        return {}
    if "fp" not in z or "dk_id" not in z:
        return {}
    dk = [str(x) for x in z["dk_id"]]
    sal = _salaries(conn, run_id)
    info = _projection_info(conn, run_id)
    own = _ownership(conn, run_id)
    keep = [j for j, d in enumerate(dk) if d in sal and d not in ("", "None", "nan")]
    seen: set[str] = set()
    keep = [j for j in keep if not (dk[j] in seen or seen.add(dk[j]))]
    out: dict[str, dict] = {}
    if keep:
        ids = [dk[j] for j in keep]
        raw = raw_chances(z["fp"][:, keep], [sal[d] for d in ids])
        tiers = [info.get(d, (None, None))[0] for d in ids]
        cal = calibrate(raw, tiers)
        means = z["fp"][:, keep].astype(np.float32).mean(0)
        vrank = value_ranks(ids, means, [sal[d] for d in ids],
                            [info.get(d, (None, None))[1] for d in ids])
        for d, p0, p1, t, m in zip(ids, raw, cal, tiers, means):
            o, b = own.get(d, (None, None))
            pp = info.get(d, (None, None))[1]
            vr = vrank.get(d)
            out[d] = {"chance": float(p1), "raw": float(p0), "text": chance_text(p1),
                      "own": o, "band": b, "tier": t, "p_play": pp, "salary": sal[d],
                      "mark": big_mark(sal[d]), "mean": float(m), "value_rank": vr,
                      "x": is_xfactor(p1, b, pp, t)
                      and not (vr is not None and vr <= VALUE_TOP_EXCLUDE)}
        apply_cap(out)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def value_ranks(ids: Sequence[str], means: Sequence[float], salaries: Sequence[float],
                p_play: Sequence[float | None], min_play: float = 0.5) -> dict[str, int]:
    """Rank (1 = best) of mean DK per $1k among players with P(play) >= `min_play`."""
    rows = [(float(m) / (float(s) / 1000.0), d) for d, m, s, pp in zip(ids, means, salaries, p_play)
            if pp is not None and pp >= min_play and s and s > 0 and np.isfinite(m)]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return {d: i + 1 for i, (_, d) in enumerate(rows)}


def apply_cap(table: dict[str, dict], cap: int = CAP) -> None:
    """Keep the tag on the `cap` players with the highest chance (ties: lower popularity)."""
    tagged = sorted((d for d, r in table.items() if r.get("x")),
                    key=lambda d: (-table[d]["chance"], table[d]["own"] if table[d]["own"]
                                   is not None else 99.0, d))
    for d in tagged[cap:]:
        table[d]["x"] = False


def xfactor_ids(table: Mapping[str, Mapping]) -> set[str]:
    return {d for d, r in table.items() if r.get("x")}


def lineup_count(slots_lists: Iterable[Sequence[Mapping]], ids: set[str]) -> int:
    """Number of lineups (each a list of slot dicts with ``player_id``) with at least one
    player in `ids`."""
    return sum(1 for slots in slots_lists
               if any(str(s.get("player_id")) in ids for s in slots))
