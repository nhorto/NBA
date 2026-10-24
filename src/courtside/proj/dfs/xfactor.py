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
                                     "salary", "x"[, "splits"]}}   (empty when the run has no
                                     sims; "splits" only when the sims carry availability
                                     scenarios: `proj/scenarios.py` teammate plays/sits arms,
                                     each with "teammate_dk" / "teammate" ids, P10)
  xfactor_ids(table) -> set[str]
  lineup_count(slots_lists, ids) -> int   (lineups with at least one X factor player)

**xf-v2 (switch ``COURTSIDE_XF_V2``, default OFF; IMPROVEMENTS.md D3, docs/research/xfactor-v2.md).**
OFF nothing below runs and the table is exactly xf-v1. ON, the X factor is redefined as
leverage against the modelled GPP field:

* **Fields.** ``LEV_K`` (4) noisy fields of the slate's reference GPP (`field.default_contests`):
  the D5 K-field machinery (`field.perturb_ownership`, late-news ``own_unc``) is used whatever
  ``COURTSIDE_FIELD_K`` says, because top-1 % share over ownership is only honest with ownership
  noise. Sim row s is scored against field s mod K (as `field.evaluate` on a FieldSet), on the
  builder's calibrated scoring truth (`portfolio.truth_sims`), ``LEV_SIMS`` rows.
* **Top-1 % share.** Per sim, the field sample's top ``LEV_TOP`` (1 %) of entries by score (count
  weighted, ties at the line included; `field.evaluate`'s top-1 % line on the sample); a player's
  share = mean over sims of the share of those entries holding him.
* **Leverage** = share / ownership, pooled over the K fields (ownership = his realised share of
  the fields' entries), shrunk toward 1: ``lev = (share + a) / (own + a)`` with ``a =
  LEV_PRIOR`` (0.25 %). That is the posterior mean of the per-entry top-1 % rate under a
  Beta(a F q, a F (1 - q)) prior centred on the field's base rate q (a F = 100 pseudo-entries
  at the default 4 x 10,000 sample): his C = own F entries are the trials (binomial-bound
  variance), so a 0.1 %-owned player (40 entries) keeps 29 % of his raw leverage, a 1 % player 80 %,
  a 5 % player 95 %.
* **Tag** (`is_xfactor_v2`): P(play) >= P_PLAY_MIN, leverage >= ``LEV_MIN``, top-1 % share >=
  ``SHARE_MIN`` (no 0.1 %-owned scrubs), modelled-field popularity below ``OWN_MAX``, not among the
  slate's ``VALUE_TOP_EXCLUDE`` best PUBLIC projections per $1k (the proxy's public number,
  `public_value_ranks`); at most ``CAP`` per slate, highest leverage first. Rows gain "lev",
  "lev_raw", "share", "lev_own", "pub_value_rank", "x_v1" (the v1 flag) and, on tagged rows,
  "split_text" (P10, when the sims carry scenarios) and "late_news" (the D4 Teammate-out label).
  Cached on disk per run (``data_dir()/cache/xfactor``): a cold run costs one K-field build.
* **Builder** (`builder_leverage`): `portfolio.build_portfolio` adds leverage-tilted candidate
  solves next to the ownership-penalised ones when the switch is ON.
* **Precomputed** (`store_safe`, ON only): the DFS runner (`runner._execute`, after the late-news
  tag) stores the run's leverage table in ``dfs_runs.detail[DETAIL_KEY]``; `apply_v2` reads it
  first (then the disk cache, then computes). OFF the detail is unchanged.

**Tail remap** (`tail_remap.TAIL_REMAP`, `+tr`): the big-night chance uses ``STRETCH_TR`` (the
same map refit on remapped sims) instead of ``STRETCH`` while that switch is on (`stretch_table`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
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
# The same map refit on tail-remapped sims (`+tr`, `proj/tail_remap.py`, IMPROVEMENTS.md D6): the
# remap narrows the upper tails, so STRETCH (fit on raw sims) runs about 1 pt low on them. Same
# method, seasons and population (RotoGuru 2015-16 .. 2020-21, P(play) >= 0.85, k = 6), arm
# pathA-v0.3+mf+p2+rx+rb+tr; `research/xf_ship_prep.py stretch`, docs/research/tail-remap.md
# (addendum; n = 120k; seasons without remap params use 2019-20's, which are fit on 2018-19).
# Held-out 2019-21 (walk-forward): mean 24.1 % vs 24.2 % observed (STRETCH: 23.3 %).
# Used only while `tail_remap.TAIL_REMAP` is on (`stretch_table`); OFF -> STRETCH.
STRETCH_TR: dict[str, tuple[float, float]] = {"bench": (-0.2715, 0.8181), "role": (-0.1860, 0.8142),
                                              "star": (-0.2532, 0.8272)}
LABEL = ("Chance of a big night from our simulations plus a low estimated popularity "
         "(proxy), leaving out the slate's 10 best projections per $1k (those tend to be popular). "
         "Estimates, not advice.")
_EPS = 1e-4
_ONE_IN = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)

# ---- xf-v2 (leverage; module doc). Read once at import; `set_xf_v2` changes it at run time.
XF_V2_ENV = "COURTSIDE_XF_V2"
XF_V2_DEFAULT = True          # shipped 2026-10-24 (owner OK: replaces xf-v1)
VERSION_V2 = "xf-v2"
LEV_K = 4                    # noisy fields (D5 machinery, explicitly, whatever COURTSIDE_FIELD_K)
LEV_SIMS = 4000              # truth-sim rows scored (1,000 per field)
LEV_FIELD_N = 10_000         # sampled entries per field (the slate page's reference size)
LEV_TOP = 0.01               # top 1 % of the field's entries
LEV_PRIOR = 0.0025           # shrinkage: lev = (share + a) / (own + a), a = 0.25 %
LEV_MIN = 1.5                # tag: shrunk leverage at least this
SHARE_MIN = 0.02             # tag: in at least 2 % of the top-1 % entries
OWN_MAX = 0.10               # tag: his share of the modelled fields' entries under 10 %
LEV_LOG_CLIP = 1.5           # builder: |log leverage| used in the tilt
LEV_TILT_MIN = 0.25          # builder: lambda ~ U(LEV_TILT_MIN, LEV_TILT_MAX) * sd(mu) on a
LEV_TILT_MAX = 1.0           # scenario objective (portfolio.candidates)
LEV_N_SOLVES = 100           # builder: leverage-tilted solves (x Settings.scale), as n_ownpen
LEV_SEED = 31                # builder: own rng stream (seed + 31), the other candidates unchanged
LABEL_V2 = ("Share of the best 1 % of entries in our simulated contests well above his estimated "
            "popularity (a proxy), leaving out the slate's 10 best public projections per $1k. "
            "Estimates, not advice.")


def _on(v) -> bool:
    return str(v).strip().lower() in ("1", "on", "true", "yes", "v2")


XF_V2 = _on(os.environ.get(XF_V2_ENV)) if os.environ.get(XF_V2_ENV) is not None \
    else XF_V2_DEFAULT


def set_xf_v2(on: bool) -> None:
    global XF_V2
    XF_V2 = bool(on) if isinstance(on, bool) else _on(on)


def xf_v2() -> bool:
    return bool(XF_V2)


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


def _tail_remap_on() -> bool:
    from .. import tail_remap as TRM
    return bool(TRM.TAIL_REMAP)


def stretch_table() -> dict[str, tuple[float, float]]:
    """``STRETCH_TR`` while the tail remap (`tail_remap.TAIL_REMAP`, `+tr`) is on, else
    ``STRETCH`` (the process switch, as the sims of the runs it produces)."""
    return STRETCH_TR if _tail_remap_on() else STRETCH


def calibrate(p: Sequence[float], tiers: Sequence[str | None]) -> np.ndarray:
    """The per-tier logistic recalibration (`stretch_table`: ``STRETCH``, or ``STRETCH_TR``
    with the tail remap on) of raw sim chances."""
    p = np.asarray(p, float)
    out = p.copy()
    ok = np.isfinite(p)
    z = np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
    tab = stretch_table()
    for i, t in enumerate(tiers):
        if not ok[i] or t not in tab:
            continue
        a, b = tab[t]
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
    if xf_v2():
        key = key + (VERSION_V2,)
    if _tail_remap_on():
        key = key + ("+tr",)
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
        _attach_splits(out, z, keep, ids, sal, tiers)
        if xf_v2():
            try:
                apply_v2(conn, run_id, out, key[2])
            except Exception:  # noqa: BLE001 - the tag must not break a page or a build
                log.exception("xfactor v2 unavailable for run %s; showing xf-v1", run_id)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def _attach_splits(out: dict, z: dict, keep: list, ids: list, sal: dict, tiers: list) -> None:
    """Teammate-availability splits (`proj/scenarios.py`) per row, when the sims carry the
    scenarios (P10); the same big-night mark and per-tier recalibration as "chance"."""
    from .. import scenarios as SCN
    from ..store import read_scenarios
    sc = read_scenarios(z)
    if sc is None:
        return
    dk_all = [str(x) for x in z["dk_id"]]
    pid = [int(x) for x in z["player_id"]]
    for j, d, t in zip(keep, ids, tiers):
        sp = SCN.player_splits(z["fp"], sc, j, sal[d], t)
        for s in sp:
            s["teammate"] = pid[s["col"]]
            s["teammate_dk"] = dk_all[s["col"]]
        out[d]["splits"] = sp


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


# ============================================================================ xf-v2: leverage

def _lineup_matrices(f) -> tuple:
    """(A, B): A (U x n) DK-score weights of the field's unique lineups (x2 per slot, CPT x3, as
    `field.evaluate`), B (U x n) 0/1 'lineup holds player'."""
    from scipy import sparse
    U, k = f.lineups.shape
    n = len(f.pids)
    w = np.array([3.0 if s == "CPT" else 2.0 for s in f.slots], np.float32)
    rows = np.repeat(np.arange(U), k)
    cols = f.lineups.ravel()
    A = sparse.csr_matrix((np.tile(w, U), (rows, cols)), shape=(U, n), dtype=np.float32)
    B = sparse.csr_matrix((np.ones(U * k, np.float32), (rows, cols)), shape=(U, n))
    B.data[:] = 1.0
    return A, B


def top1_counts(f, fp: np.ndarray, top: float = LEV_TOP, chunk: int = 500
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """One field against sim rows `fp` (r x n, columns = f.pids): (share_sum (n,), own (n,),
    rows). share_sum = Σ over rows of the share of the row's top-`top` entries holding each
    player (entries count-weighted by the sample multiplicity, the line = the score of the
    ceil(top x F)-th best entry, ties at the line included); own = his share of the field's
    entries. Scores in `field.evaluate`'s exact integer units (1/32 point x slot weight)."""
    A, B = _lineup_matrices(f)
    c = np.asarray(f.counts, np.float64)
    Fn = float(c.sum())
    n1 = max(1, int(math.ceil(top * Fn)))
    U = len(c)
    m = min(U, n1)
    share = np.zeros(len(f.pids))
    fq = np.rint(np.asarray(fp, np.float32) * 32).astype(np.float32)
    for a in range(0, fq.shape[0], chunk):
        X = fq[a:a + chunk]
        S = np.asarray((A @ X.T), np.float64)              # (U, r)
        r = S.shape[1]
        if m < U:
            part = np.argpartition(-S, m - 1, axis=0)[:m]
        else:
            part = np.broadcast_to(np.arange(U)[:, None], (U, r))
        sv = np.take_along_axis(S, part, 0)
        o = np.argsort(-sv, axis=0, kind="stable")
        idx = np.take_along_axis(part, o, 0)
        svs = np.take_along_axis(sv, o, 0)
        cum = np.cumsum(c[idx], axis=0)
        pos = np.minimum((cum < n1).sum(0), m - 1)
        line = svs[pos, np.arange(r)]
        W = (S >= line[None, :]) * c[:, None]              # (U, r) entries in the top 1 %
        tot = W.sum(0)
        X_i = np.asarray(B.T @ W)                          # (n, r)
        share += (X_i / np.maximum(tot, 1.0)[None, :]).sum(1)
    own = np.asarray(B.T @ c).ravel() / max(Fn, 1.0)
    return share, own, float(fq.shape[0])


def leverage(fields, fp: np.ndarray, *, top: float = LEV_TOP, prior: float = LEV_PRIOR
             ) -> dict[str, np.ndarray]:
    """Top-1 % share and leverage per player (columns of `fp` = the fields' common `pids`)
    against K fields (a `field.FieldSet`, a list of Fields or one Field): row s is scored against
    field s mod K. Returns arrays over pids: "share" (mean over all rows), "own" (realised
    ownership, mean over the fields, row-weighted), "lev_raw" = share / own, "lev" = (share +
    prior) / (own + prior) (module doc), "share_k" / "own_k" (K x n) per field."""
    fl = list(getattr(fields, "fields", None) or (fields if isinstance(fields, (list, tuple))
                                                     else [fields]))
    S = fp.shape[0]
    K = len(fl)
    parts = ([(np.arange(k, S, K), f) for k, f in enumerate(fl)] if S >= K
             else [(np.arange(S), f) for f in fl])
    sh_k, own_k, w = [], [], []
    for rows, f in parts:
        sh, ow, r = top1_counts(f, fp[rows], top)
        sh_k.append(sh / max(r, 1.0))
        own_k.append(ow)
        w.append(r)
    w = np.array(w) / max(sum(w), 1.0)
    share = (np.array(sh_k) * w[:, None]).sum(0)
    own = (np.array(own_k) * w[:, None]).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(own > 0, share / own, np.nan)
    lev = (share + prior) / (own + prior)
    return {"share": share, "own": own, "lev_raw": raw, "lev": lev,
            "share_k": np.array(sh_k), "own_k": np.array(own_k)}


def noisy_fields(players, inputs, contest, rules, *, base=None, ownership=None,
                 k: int = LEV_K, own_unc: Mapping[str, float] | None = None,
                 n: int | None = None, mix=None, seed: int = 0, threads: int = 2):
    """K fields with ownership noise (`field.build_fields`, called with k explicitly, so it does
    not depend on COURTSIDE_FIELD_K). `base`: a FieldSet with K fields is returned as it is; a
    Field lends its optimizer-like solves (the K fields are then sampled around its ownership,
    no new CP-SAT solves)."""
    from . import field as F
    if isinstance(base, F.FieldSet) and base.k == k:
        return base
    if base is None:
        return F.build_fields(players, inputs, contest, k=k, own_unc=own_unc, rules=rules,
                              n=n, mix=mix, ownership=ownership, seed=seed, threads=threads)
    if isinstance(base, F.FieldSet):
        base = base.fields[0]
        op = ownership
    else:
        op = ownership or base.ownership
    t0 = time.perf_counter()
    unc = {str(a): float(b) for a, b in (own_unc or {}).items()}
    sdv = F.own_sd_for(op, unc)
    rng = np.random.default_rng([int(seed), 0xD3])
    fields = []
    for j in range(k):
        opj = F.perturb_ownership(op, sdv, rng)
        f = F.build_field(players, inputs, contest, rules=rules, n=n or int(base.n),
                          mix=base.mix, archetype=base.archetype, ownership=opj,
                          seed=seed + 100 + j, threads=threads, opt_pairs=base.opt_solves)
        f.version = F.FIELD_VERSION + F.FIELD_K_TAG
        fields.append(f)
    return F.FieldSet(fields, op, sdv, unc, seconds=time.perf_counter() - t0)


def public_value_ranks(op) -> dict[str, int]:
    """Rank (1 = best) of the ownership proxy's PUBLIC projection per $1k (`pub_proj` / salary
    in thousands) among the players the proxy drafts (own > 0)."""
    inp = op.inputs
    pub = inp["pub_proj"].to_numpy(float)
    sal = inp["salary"].to_numpy(float)
    rows = [(pub[i] / (sal[i] / 1000.0), str(p)) for i, p in enumerate(op.pids)
            if op.own[i] > 0 and sal[i] > 0 and np.isfinite(pub[i])]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return {d: i + 1 for i, (_, d) in enumerate(rows)}


def is_xfactor_v2(lev: float | None, share: float | None, own: float | None,
                  p_play: float | None, pub_rank: int | None) -> bool:
    """The xf-v2 rule before the cap (module doc). `own` = his realised share of the modelled
    fields' entries (the leverage's denominator; it includes the optimizer-like part, which
    drafts the public's value plays more than the stored proxy says)."""
    ok = lambda v: v is not None and np.isfinite(v)                      # noqa: E731
    return (ok(lev) and lev >= LEV_MIN and ok(share) and share >= SHARE_MIN
            and p_play is not None and p_play >= P_PLAY_MIN
            and ok(own) and own < OWN_MAX
            and not (pub_rank is not None and pub_rank <= VALUE_TOP_EXCLUDE))


def apply_cap_v2(table: dict[str, dict], cap: int = CAP) -> None:
    """Keep the v2 tag on the `cap` players with the highest leverage (ties: lower popularity)."""
    tagged = sorted((d for d, r in table.items() if r.get("x")),
                    key=lambda d: (-table[d]["lev"], table[d].get("own") or 99.0, d))
    for d in tagged[cap:]:
        table[d]["x"] = False


def _params_key() -> str:
    p = (VERSION_V2, LEV_K, LEV_SIMS, LEV_FIELD_N, LEV_TOP, LEV_PRIOR)
    return hashlib.sha1(repr(p).encode()).hexdigest()[:10]


def _cache_path(run_id: str) -> Path:
    from ... import config
    return config.data_dir() / "cache" / "xfactor" / f"{run_id}.{_params_key()}.json"


def run_leverage(conn: sqlite3.Connection, run_id: str, *, seed: int = 0,
                 n_sims: int = LEV_SIMS, n_field: int = LEV_FIELD_N,
                 contest=None, as_of: str | None = None,
                 own_unc: Mapping[str, float] | None = None) -> dict:
    """{"players": {dk id: {share, own (realised), lev, lev_raw, pub_value_rank}}, "contest",
    "k", "seconds"} for a stored run: K noisy fields of the slate's reference GPP (or
    `contest`) on the run's truth sims (module doc). No caching here (see `apply_v2`).
    `as_of` (default: the run row's created_at) and `own_unc` (default: the run's late-news
    tag, `field.run_own_uncertainty`) let the DFS runner compute it before it saves the run
    (`store_safe`) with exactly the inputs a later page view sees."""
    from . import field as F
    from . import portfolio as PF
    from .. import role_blend as RB
    t0 = time.perf_counter()
    r = conn.execute("SELECT slate_id, created_at FROM dfs_runs WHERE run_id=?",
                     (run_id,)).fetchone()
    if r is None:
        return {}
    sid, at = str(r[0]), (as_of or r[1])
    ctx = F.prepare(conn, sid, run_id, n_sims=n_sims, seed=seed, as_of=at)
    if ctx["sims"] is None:
        return {}
    pool = ctx["pool"]
    if contest is None:
        contest = F.default_contests(conn, sid)[1]
    op = F.ownership_proxy(ctx["inputs"], pool, archetype=contest.archetype)
    unc = (dict(own_unc) if own_unc is not None
           else F.run_own_uncertainty(conn, run_id, slate_id=sid))
    fs = noisy_fields(pool.players, ctx["inputs"], contest, pool.rules, ownership=op,
                      own_unc=unc, n=n_field, seed=seed)
    fp, _ = F.align_sims(ctx["sims"], fs.pids, n_sims, seed)
    fp, _ = PF.truth_sims(fp, fs.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                          np.random.default_rng(seed + 7), blended=RB.run_blended(conn, run_id))
    lv = leverage(fs, fp)
    pvr = public_value_ranks(op)
    players = {}
    for i, p in enumerate(fs.pids):
        if op.own[i] <= 0:
            continue
        players[str(p)] = {"share": float(lv["share"][i]), "lev_own": float(lv["own"][i]),
                           "lev": float(lv["lev"][i]),
                           "lev_raw": (None if not np.isfinite(lv["lev_raw"][i])
                                       else float(lv["lev_raw"][i])),
                           "proxy_own": float(op.own[i]), "pub_value_rank": pvr.get(str(p))}
    return {"version": VERSION_V2, "players": players, "k": fs.k,
            "contest": {"contest_id": contest.contest_id, "name": contest.name,
                        "field_size": contest.field_size, "archetype": contest.archetype},
            "own_unc": {a: b for a, b in unc.items() if b != 1.0},
            "n_sims": int(fp.shape[0]), "seconds": round(time.perf_counter() - t0, 2)}


def _split_text(row: dict, names: Mapping[str, str]) -> str | None:
    """The P10 split shown on a tagged row: the teammate whose absence moves his DK points most
    (`scenarios.card_splits`), as plain text; None without scenarios or a big enough shift."""
    from .. import scenarios as SCN
    sp = SCN.card_splits(row.get("splits") or [], k=1)
    if not sp:
        return None
    s = sp[0]
    nm = names.get(str(s.get("teammate_dk")), "a teammate")
    a, b = s["plays"], s["sits"]
    if a.get("big") is None or b.get("big") is None:
        return (f"With {nm} out: {b['dk']:.1f} DK points in our simulations vs {a['dk']:.1f} "
                f"with him available.")
    return (f"With {nm} out: big-night chance {chance_text(b['big'])} vs "
            f"{chance_text(a['big'])} with him available.")


# ---- precompute in the DFS run path (switch ON; `runner._execute`, after the late-news tag)

DETAIL_KEY = "xfactor_v2"        # dfs_runs.detail key of the stored leverage table


def _sims_mtime(conn: sqlite3.Connection, run_id: str) -> int | None:
    path = _sim_path(conn, run_id)
    try:
        return Path(path).stat().st_mtime_ns if path else None
    except OSError:
        return None


def precompute_run(conn: sqlite3.Connection, run_id: str, *, as_of: str | None = None,
                   late_news: Mapping | None = None) -> dict | None:
    """The run's `run_leverage` result with "mtime" (sims file) and "params" (`_params_key`),
    ready for ``dfs_runs.detail[DETAIL_KEY]``; None for Showdown, a run without sims or an empty
    result. `late_news`: the run's late-news tag (`late_news_tag.compute`), whose D5 ownership
    uncertainty the page would read back from the saved run."""
    mtime = _sims_mtime(conn, run_id)
    if mtime is None:
        return None
    r = conn.execute("SELECT s.tag FROM dfs_runs r JOIN dfs_slates s USING (slate_id) WHERE "
                     "r.run_id=?", (run_id,)).fetchone()
    if r is not None and (r[0] or "").lower() == "showdown":
        return None
    unc = None
    if late_news is not None:
        from . import late_news_tag as LNT
        unc = LNT.own_uncertainty(late_news)
    res = run_leverage(conn, run_id, as_of=as_of, own_unc=unc)
    if not res:
        return None
    res["mtime"] = mtime
    res["params"] = _params_key()
    return res


def store_safe(detail: dict, conn: sqlite3.Connection, slate_id, run_id: str,
               as_of: str | None = None, late_news: Mapping | None = None) -> None:
    """Runner hook: with the switch ON, ``detail[DETAIL_KEY]`` = `precompute_run` (the first
    page view then reads it instead of building the fields). OFF: nothing (the detail is
    unchanged). Never raises."""
    if not xf_v2():
        return
    try:
        res = precompute_run(conn, run_id, as_of=as_of, late_news=late_news)
    except Exception:  # noqa: BLE001 - the tag must never fail a lock run
        log.exception("xfactor v2 precompute failed for run %s (slate %s)", run_id, slate_id)
        return
    if res:
        detail[DETAIL_KEY] = res


def stored_leverage(conn: sqlite3.Connection, run_id: str, mtime: int | None) -> dict | None:
    """The run's stored table (``dfs_runs.detail[DETAIL_KEY]``) when its version, parameters
    and sims mtime match, else None."""
    try:
        r = conn.execute("SELECT detail FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
        det = json.loads(r[0]) if r and r[0] else None
    except (sqlite3.Error, ValueError, TypeError):
        return None
    res = det.get(DETAIL_KEY) if isinstance(det, dict) else None
    if (isinstance(res, dict) and res.get("version") == VERSION_V2
            and res.get("params") == _params_key() and res.get("mtime") == mtime
            and isinstance(res.get("players"), dict)):
        return res
    return None


def apply_v2(conn: sqlite3.Connection, run_id: str, out: dict[str, dict], mtime: int) -> None:
    """Switch ON: replace the tag ("x") in a v1 table by the xf-v2 rule (v1's flag kept as
    "x_v1"), from the run's leverage: the table stored with the run (`store_safe`), else the
    disk cache per run and sims mtime, else computed (and cached)."""
    path = _cache_path(run_id)
    res = stored_leverage(conn, run_id, mtime)
    try:
        if res is None and path.exists():
            d = json.loads(path.read_text())
            if d.get("mtime") == mtime and d.get("version") == VERSION_V2:
                res = d
    except (OSError, ValueError):
        res = None
    if res is None:
        res = run_leverage(conn, run_id)
        if not res:
            return
        res["mtime"] = mtime
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res))
            tmp.replace(path)
        except OSError:
            log.warning("xfactor v2: cache not written at %s", path)
    lev = res.get("players") or {}
    names = {}
    try:
        names = {str(a): b for a, b in conn.execute(
            "SELECT DISTINCT site_player_id, player_name FROM dfs_salaries s JOIN dfs_runs r ON "
            "r.slate_id = s.slate_id WHERE r.run_id=?", (run_id,))}
    except sqlite3.Error:
        pass
    try:
        from . import late_news_tag as LNT
        ln = LNT.flags_by_dk(LNT.run_flags(conn, run_id))
    except Exception:  # noqa: BLE001
        ln = {}
    for d, row in out.items():
        row["x_v1"] = bool(row.get("x"))
        L = lev.get(d) or {}
        row.update(lev=L.get("lev"), lev_raw=L.get("lev_raw"), share=L.get("share"),
                   lev_own=L.get("lev_own"), pub_value_rank=L.get("pub_value_rank"),
                   version=VERSION_V2)
        row["x"] = bool(L) and is_xfactor_v2(L.get("lev"), L.get("share"), L.get("lev_own"),
                                             row.get("p_play"), L.get("pub_value_rank"))
    apply_cap_v2(out)
    for d, row in out.items():
        if not row["x"]:
            continue
        row["split_text"] = _split_text(row, names)
        if d in ln:
            from . import late_news_tag as LNT
            row["late_news"] = LNT.label(ln[d])
        row["title"] = v2_title(row)


def v2_title(row: Mapping) -> str:
    """Plain-English hover text of a v2 tag (information only, STYLE.md)."""
    s = (f"In our simulated contests he is in {row['share'] * 100:.1f} % of the best 1 % of "
         f"entries and in {row['lev_own'] * 100:.1f} % of all entries of the modelled field "
         f"(estimated popularity, a proxy): about {row['lev']:.1f} times as often.")
    if row.get("split_text"):
        s += " " + row["split_text"]
    if row.get("late_news"):
        s += " Teammate out: see that tag."
    return s + " Estimates, not advice."


def builder_leverage(fld, players, inputs, contest, rules, fp: np.ndarray, *,
                     own_unc: Mapping[str, float] | None = None, seed: int = 0,
                     threads: int = 2, n_sims: int = LEV_SIMS) -> tuple[dict, dict]:
    """({dk id: shrunk leverage}, info) for the contest builder (switch ON): the build's own
    field (a FieldSet with LEV_K fields as it is; a single Field lends its solves to LEV_K
    noisy fields) on at most `n_sims` of the generation rows `fp` (columns = fld.pids)."""
    t0 = time.perf_counter()
    fs = noisy_fields(players, inputs, contest, rules, base=fld,
                      ownership=getattr(fld, "ownership", None), own_unc=own_unc,
                      seed=seed, threads=threads)
    S = fp.shape[0]
    rows = (np.arange(S) if S <= n_sims
            else np.sort(np.random.default_rng(seed + LEV_SEED).choice(S, n_sims, replace=False)))
    lv = leverage(fs, fp[rows])
    act = fs.ownership.own > 0
    out = {str(p): float(lv["lev"][i]) for i, p in enumerate(fs.pids) if act[i]}
    hi = sorted(out.items(), key=lambda kv: -kv[1])[:5]
    return out, {"version": VERSION_V2, "k": fs.k, "rows": int(len(rows)),
                 "reused_fields": fs is fld, "top": [[p, round(v, 3)] for p, v in hi],
                 "seconds": round(time.perf_counter() - t0, 2)}


def tilt_scores(ids: Sequence[str], base: np.ndarray, lev: Mapping[str, float], lam: float
                ) -> dict[str, float]:
    """A leverage-tilted objective: mu_i + lam * clip(log lev_i, +-LEV_LOG_CLIP) (lev 1 for
    players without one)."""
    lg = np.array([math.log(max(float(lev.get(i, 1.0)), 1e-6)) for i in ids])
    return dict(zip(ids, base + lam * np.clip(lg, -LEV_LOG_CLIP, LEV_LOG_CLIP)))
