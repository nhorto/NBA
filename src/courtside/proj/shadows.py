"""Player-side live shadows and their pre-registered grading (docs/IMPROVEMENTS.md E1,
docs/MLOPS.md section 6).

The 2026-27 live season is the confirmatory set for player-side switches. At each scheduled
pregame run (10:00, 13:30, 17:00, 18:30, 20:00, 22:00 ET) `run` projects the day's main DK
Classic slate with the live champion and then once per candidate arm (the champion with ONE
switch on, flipped with its `pipeline.set_*` setter), all with the same seed and sim count
(common random numbers). Nothing is served: the projections go to files under
``<data>/proj/shadows/<nba_day>/``, never to ``player_projections`` / ``dfs_runs`` (which feed
the site, the DFS monitor's champion registry and the props), and the database connection is
read-only.

Isolation (the champion is never affected):

* the step runs in its own low-priority process after the pregame job (`deploy/run-job.sh`),
  never in the DFS lock tick, and skips itself when a lock-45 / lock-10 / swap build window of
  any slate falls within [now - 10 min, now + budget + 2 min] (`lock_conflict`);
* every arm runs in try/except; the switch flags, the bundle suffix and MODEL_VERSION are
  restored (and checked) after every arm (`arm_on`);
* a time budget (default 240 s for the whole run, `COURTSIDE_SHADOW_BUDGET_S`) is enforced: an
  arm that would not fit is skipped, and a running arm is interrupted by SIGALRM at the deadline;
* an arm whose bundle file is not on disk is skipped: a shadow never fits or writes a bundle;
* `COURTSIDE_SHADOWS=0` (or SHADOWS_DEFAULT = False) turns the step off; the arm list comes from
  the registry (status "active") or `COURTSIDE_SHADOW_ARMS` ("+tr,+ep").

Stored per arm and run (`.npz`): player_id, game_id, team, tier (the run's own), p_play,
E[min] (all), E[min | plays], E[DK] (all) and K = 100 mid-point quantiles of the unconditional
DK distribution (DNP = 0), from which the grader computes CRPS (`metrics.crps_quantiles`; the
same approximation for both arms, exactly 0 difference on unchanged rows). Only games that have
not tipped at the run time are kept. A JSON manifest per run records statuses and seconds.

Grading (`grade`, postgame step ``shadow_grade``): per night and arm, rows paired with the
champion run of the same time, actuals from ``player_games`` (DNP = 0); per arm the registry's
primary metric at its run slot over all accumulated nights of the season, mean difference
(arm - champion) with the two-way night x player clustered SE (`metrics.paired_cluster2`), a
tier breakdown (champion tiers) and the pass / fail / pending decision (`decide`). The report is
``<data>/reports/shadows/shadow_grades.json`` (shown on /health/dfs).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import os
import signal
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger("courtside.proj.shadows")

SHADOWS_DEFAULT = True                 # accumulate from the day it is merged; env overrides
SHADOWS_ENV = "COURTSIDE_SHADOWS"      # "0" / "off" disables the pregame step
ARMS_ENV = "COURTSIDE_SHADOW_ARMS"     # "+tr,+ep" (or "tr,ep"): overrides the registry's active list
BUDGET_ENV = "COURTSIDE_SHADOW_BUDGET_S"
BUDGET_S = 240.0                       # whole run (90 s timed out +lf and skipped +tm on 2026-10-24)
SIMS_ENV = "COURTSIDE_SHADOW_SIMS"
SHADOW_SIMS = 10_000                   # champion and arms alike (25,000 x 8 does not fit the budget)
SEED_KIND = "pregame"                  # stable_seed(slate, SEED_KIND, SEED_VERSION), shared
K_LEVELS = 100
LEVELS = (np.arange(K_LEVELS) + 0.5) / K_LEVELS
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "mlops" / "shadow_registry.json"
PREGAME_TIMES = ("10:00", "13:30", "17:00", "18:30", "20:00", "22:00")   # deploy/schedule.sh
SLOT_LATE_MIN = 50                     # a run up to 50 min after a scheduled time takes its slot
GUARD_BACK_S = 10 * 60                 # a lock build started up to 10 min ago may still run
GUARD_PAD_S = 120
GUARD_STEP_S = 60
CLASSIC = 70
CHAMPION = "champion"
EMIN = 10.0
TIERS = ("star", "role", "bench")
P_CLIP = 1e-4
REPORT_NAME = "shadow_grades.json"

# arm tag -> (pipeline setter, module, flag attribute). The setters keep MODEL_VERSION and the
# bundle suffix in step; the flags are read back to restore and to check.
SWITCHES: dict[str, tuple[str, str, str]] = {
    "+tr": ("set_tail_remap", "courtside.proj.tail_remap", "TAIL_REMAP"),
    "+ep": ("set_early_prior", "courtside.proj.early_prior", "EARLY_PRIOR"),
    "+gd": ("set_inj_gameday", "courtside.proj.availability", "INJ_GAMEDAY"),
    "+lf": ("set_line_fresh", "courtside.proj.pipeline", "LINE_FRESH"),
    "+tm": ("set_tier_match", "courtside.proj.copula", "TIER_MATCH"),
    "+rs": ("set_min_restrict", "courtside.proj.min_restrict", "MIN_RESTRICT"),
    "+pq": ("set_pplayq", "courtside.proj.availability", "PPLAYQ"),
}


class ShadowTimeout(Exception):
    """The run's time budget ran out."""


class BundleMissing(Exception):
    """The arm's bundle file is not on disk (a shadow never fits one)."""


# ============================================================================== config

def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def enabled() -> bool:
    return _flag(SHADOWS_ENV, SHADOWS_DEFAULT)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def budget_s() -> float:
    return _float_env(BUDGET_ENV, BUDGET_S)


def shadow_sims() -> int:
    return int(_float_env(SIMS_ENV, SHADOW_SIMS))


def root() -> Path:
    return config.data_dir() / "proj" / "shadows"


def report_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or config.data_dir()) / "reports" / "shadows"


def _norm_arm(a: str) -> str:
    a = a.strip()
    return a if a.startswith("+") else f"+{a}"


def load_registry(path: Path | str | None = None) -> dict:
    """The registry with the defaults filled into every arm entry."""
    reg = json.loads(Path(path or REGISTRY_PATH).read_text())
    dflt = reg.get("defaults", {})
    arms = []
    for a in reg.get("arms", []):
        e = {**dflt, **a}
        e["bar"] = {**dflt.get("bar", {}), **a.get("bar", {})}
        arms.append(e)
    reg["arms"] = arms
    return reg


def select_arms(reg: dict, names=None) -> list[dict]:
    """Registry entries to run: `names` (or COURTSIDE_SHADOW_ARMS), else every 'active' arm."""
    env = os.environ.get(ARMS_ENV, "").strip()
    if names is None and env:
        names = [x for x in env.split(",") if x.strip()]
    if names is None:
        return [a for a in reg["arms"] if a.get("status") == "active"]
    want = [_norm_arm(n) for n in names]
    by = {a["arm"]: a for a in reg["arms"]}
    return [by.get(n, {"arm": n, "status": "unregistered"}) for n in want]


# ============================================================================== switches

def switch_state() -> dict:
    """Every shadowable switch flag plus the pipeline's bundle suffix and MODEL_VERSION."""
    from . import pipeline as P
    st = {tag: bool(getattr(importlib.import_module(m), a)) for tag, (_, m, a) in SWITCHES.items()}
    st["_VS"], st["MODEL_VERSION"] = P._VS, P.MODEL_VERSION
    return st


def restore(snap: dict) -> None:
    """Put every switch back to `snap` (through the setters) and check the result."""
    from . import pipeline as P
    for tag, (setter, m, a) in SWITCHES.items():
        if bool(getattr(importlib.import_module(m), a)) != snap[tag]:
            getattr(P, setter)(snap[tag])
    now = switch_state()
    if now != snap:
        raise RuntimeError(f"switch state not restored: {now} != {snap}")


@contextlib.contextmanager
def arm_on(tag: str):
    """The champion with switch `tag` on, restored on exit (also on an error or a timeout)."""
    from . import pipeline as P
    snap = switch_state()
    try:
        getattr(P, SWITCHES[tag][0])(True)
        yield P.MODEL_VERSION
    finally:
        restore(snap)


@contextlib.contextmanager
def _deadline(seconds: float):
    """Raise ShadowTimeout in the block after `seconds` (SIGALRM; main thread on Unix only,
    elsewhere only the between-arm budget check applies)."""
    if seconds <= 0:
        raise ShadowTimeout("no time left")
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):
        raise ShadowTimeout("over the time budget")
    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ============================================================================== schedule

def _utc(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slot_of(at: datetime) -> str:
    """The scheduled pregame time ("13:30") a run at `at` belongs to, else "adhoc"."""
    et = _utc(at).astimezone(config.NBA_TZ)
    mins = et.hour * 60 + et.minute
    best = "adhoc"
    for s in PREGAME_TIMES:
        h, m = (int(x) for x in s.split(":"))
        if 0 <= mins - (h * 60 + m) <= SLOT_LATE_MIN:
            best = s
    return best


def run_key(at: datetime) -> str:
    return _utc(at).astimezone(config.NBA_TZ).strftime("%H%M")


def lock_conflict(conn: sqlite3.Connection, now: datetime, budget: float) -> str | None:
    """A reason to skip when any DFS build window (lock45 / lock10 / swap, the tick's own
    `runner.due_runs`) touches [now - 10 min, now + budget + 2 min]; a failing check also
    skips (fail safe)."""
    try:
        from .dfs import runner as RN
        t = now - timedelta(seconds=GUARD_BACK_S)
        end = now + timedelta(seconds=budget + GUARD_PAD_S)
        while t <= end:
            for d in RN.due_runs(conn, t):
                if d.get("note"):          # a swap with no saved lineups never runs
                    continue
                return f"{d['kind']} window of slate {d['slate_id']} at {_iso(t)}"
            t += timedelta(seconds=GUARD_STEP_S)
    except Exception as e:  # noqa: BLE001
        return f"lock guard failed ({type(e).__name__}: {e})"
    return None


def pick_slate(conn: sqlite3.Connection, day: date, at: datetime) -> dict | None:
    """The day's Classic slate with the most games not tipped at `at` (ties: more games, then
    the lower slate id), with salaries captured. None when there is none."""
    from . import availability as AV
    tips = {r.game_id: r.tip for r in AV.day_games(conn, day).itertuples()}
    best = None
    for sid, gj in conn.execute("SELECT slate_id, game_ids_json FROM dfs_slates WHERE nba_day=? "
                                "AND game_type_id=?", (day.isoformat(), CLASSIC)).fetchall():
        gids = json.loads(gj) if gj else []
        if not gids or conn.execute("SELECT 1 FROM dfs_salaries WHERE slate_id=? LIMIT 1",
                                    (str(sid),)).fetchone() is None:
            continue
        open_ = sorted(g for g in gids if g in tips and tips[g] > at)
        key = (len(open_), len(gids), -int(sid) if str(sid).isdigit() else 0)
        if open_ and (best is None or key > best[0]):
            best = (key, {"slate_id": str(sid), "game_ids": gids, "open": open_,
                          "tips": {g: _iso(tips[g]) for g in open_}})
    return best[1] if best else None


def _ids_mapped(conn: sqlite3.Connection, slate_id: str) -> float:
    from . import ids as IDS
    from .dfs import rules as R
    pool = R.build_pool(conn, slate_id)
    dk = [p.dk_player_id for p in pool.players]
    return len(IDS.nba_ids(conn, "dk", dk)) / len(dk) if dk else 0.0


# ============================================================================== projecting

def _project(conn, slate_id: str, at: datetime, *, n_sims: int, seed: int, ctx, bundle):
    from . import pipeline as P
    return P.project_slate(conn, slate_id, at, run_id="shadow", n_sims=n_sims, seed=seed,
                           kind=SEED_KIND, ctx=ctx, bundle=bundle, store=False)


def _ref_bundle(conn, season: str, ctx):
    from . import pipeline as P
    path = P.cache_dir() / f"bundle_{season}{P._VS}.json"
    if not path.exists():
        raise BundleMissing(path.name)
    return P.load_bundle(conn, season, ctx)


def _arm_bundle(season: str, ref_vs: str, ref_bundle, cache: dict):
    """The champion's bundle when the arm keeps its suffix, else the arm's file from disk."""
    from . import early_prior as EP
    from . import pipeline as P
    if P._VS == ref_vs:
        return ref_bundle
    if P._VS not in cache:
        path = P.cache_dir() / f"bundle_{season}{P._VS}.json"
        if not path.exists():
            raise BundleMissing(path.name)
        b = P.Bundle.from_json(path.read_text())
        if EP.EARLY_PRIOR and not EP.has_ep(b.state):
            raise BundleMissing(f"{path.name} has no early prior")
        cache[P._VS] = b
    return cache[P._VS]


def result_arrays(res, open_games) -> dict:
    """The stored arrays of one projection (DayResult): rows of `open_games` only."""
    pl = res.players.reset_index(drop=True)
    pid, fp = res.fp_matrix()
    if len(pid) != len(pl) or not np.array_equal(np.asarray(pid, np.int64),
                                                 pl["player_id"].to_numpy(np.int64)):
        raise RuntimeError("sims and players are not aligned")
    keep = pl["game_id"].astype(str).isin(set(map(str, open_games))).to_numpy()
    q = (np.quantile(fp[:, keep], LEVELS, axis=0, method="inverted_cdf").T
         if keep.any() else np.zeros((0, K_LEVELS)))
    s = pl[keep]
    return {"player_id": s["player_id"].to_numpy(np.int64),
            "game_id": s["game_id"].astype(str).to_numpy(dtype=str),
            "team": s["team"].astype(str).to_numpy(dtype=str),
            "tier": s["tier"].astype(str).to_numpy(dtype=str),
            "p_play": s["p_play"].to_numpy(float), "min_mean": s["min_mean"].to_numpy(float),
            "min_played": s["min_played"].to_numpy(float), "dk_mean": s["dk_mean"].to_numpy(float),
            "q": np.asarray(q, np.float32)}


def _save(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays, levels=LEVELS, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_arm(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame({k: z[k] for k in ("player_id", "game_id", "team", "tier", "p_play",
                                         "min_mean", "min_played", "dk_mean")})
    return df, z["q"].astype(float), json.loads(str(z["meta"]))


def _file_tag(tag: str) -> str:
    return CHAMPION if tag == CHAMPION else tag.lstrip("+")


def run(conn: sqlite3.Connection, now: datetime | None = None, *, arms=None,
        budget: float | None = None, n_sims: int | None = None, ctx=None, bundle=None,
        guard: bool = True, force: bool = False, project=None, registry: dict | None = None,
        out_root: Path | None = None) -> dict:
    """One shadow run at `now` (see the module doc). Never raises: returns a summary with
    ``status`` ok | skipped | disabled | error. `project` replaces `_project` (tests)."""
    if not enabled():
        return {"status": "disabled", "reason": f"{SHADOWS_ENV}=0"}
    t0 = time.monotonic()
    try:
        return _run(conn, _utc(now), t0=t0, arms=arms,
                    budget=budget_s() if budget is None else float(budget),
                    n_sims=n_sims or shadow_sims(), ctx=ctx, bundle=bundle, guard=guard,
                    force=force, project=project or _project,
                    registry=registry or load_registry(), out_root=out_root or root())
    except Exception as e:  # noqa: BLE001 - a shadow must never break anything
        log.exception("shadow run failed")
        return {"status": "error", "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.monotonic() - t0, 2)}


def _run(conn, now: datetime, *, t0: float, arms, budget: float, n_sims: int, ctx, bundle,
         guard: bool, force: bool, project, registry: dict, out_root: Path) -> dict:
    from . import pipeline as P
    day = config.nba_day(now)
    left = lambda: budget - (time.monotonic() - t0)  # noqa: E731
    slot, key = slot_of(now), run_key(now)
    out = {"status": "ok", "day": day.isoformat(), "at": _iso(now), "slot": slot, "run": key,
           "budget_s": budget, "n_sims": n_sims}
    d = out_root / day.isoformat()
    mpath = d / f"{key}.json"
    if mpath.exists() and not force:
        return {**out, "status": "skipped", "reason": f"already ran ({mpath.name})"}
    # a skip leaves a manifest too, so the run log shows it (it holds no arm files)
    if guard:
        why = lock_conflict(conn, now, budget)
        if why:
            return _finish({**out, "status": "skipped", "reason": why}, mpath, t0)
    slate = pick_slate(conn, day, now)
    if slate is None:
        return _finish({**out, "status": "skipped",
                        "reason": "no Classic slate with games still to tip"}, mpath, t0)
    if _ids_mapped(conn, slate["slate_id"]) < 0.9:
        return _finish({**out, "status": "skipped", "reason": "slate's DK ids not mapped yet"},
                       mpath, t0)
    snap = switch_state()
    season = P.season_of(day)
    seed = P.stable_seed(slate["slate_id"], SEED_KIND, P.SEED_VERSION)
    out.update(slate_id=slate["slate_id"], games=slate["open"], seed=seed,
               champion_version=snap["MODEL_VERSION"], arms={})
    entries = select_arms(registry, arms)
    # rotate the arm order by run so a tight budget does not always cut the same arms
    if entries:
        k = (day.toordinal() * len(PREGAME_TIMES)
             + (PREGAME_TIMES.index(slot) if slot in PREGAME_TIMES else 0)) % len(entries)
        entries = entries[k:] + entries[:k]

    def store(tag: str, res, version: str, secs: float) -> dict:
        arr = result_arrays(res.result, slate["open"])
        name = f"{key}_{_file_tag(tag)}.npz"
        _save(d / name, arr, {"arm": tag, "model_version": version, "seconds": secs,
                              "slate_id": slate["slate_id"], "at": _iso(now), "seed": seed,
                              "n_sims": n_sims, "run": key, "slot": slot})
        return {"status": "ok", "model_version": version, "seconds": secs, "rows": len(arr["q"]),
                "file": name}

    # ---- the champion (same code path as the arms, no switch touched)
    try:
        t = time.monotonic()
        with _deadline(left()):
            ctx = ctx or P.Context(conn, season)
            ref_vs = P._VS
            ref_b = bundle or _ref_bundle(conn, season, ctx)
            tp = time.monotonic()
            res = project(conn, slate["slate_id"], now, n_sims=n_sims, seed=seed, ctx=ctx,
                          bundle=ref_b)
        ref_s = round(time.monotonic() - t, 2)
        est = time.monotonic() - tp         # one projection, without the context load
        out["arms"][CHAMPION] = store(CHAMPION, res, snap["MODEL_VERSION"], ref_s)
    except Exception as e:  # noqa: BLE001
        log.warning("shadow champion failed: %s", e)
        out["arms"][CHAMPION] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        out["status"] = "error"
        restore(snap)
        return _finish(out, mpath, t0)
    if switch_state() != snap:            # the champion's own run must not move a switch
        restore(snap)
    cache: dict = {}
    for e in entries:
        tag = e["arm"]
        if tag not in SWITCHES:
            out["arms"][tag] = {"status": "skipped", "reason": "unknown arm (no setter)"}
            continue
        if snap[tag]:
            out["arms"][tag] = {"status": "skipped", "reason": "already on in the champion"}
            continue
        if left() < 1.1 * est:
            out["arms"][tag] = {"status": "skipped", "reason": f"budget ({left():.0f} s left, "
                                                              f"needs ~{est:.0f} s)"}
            continue
        t = time.monotonic()
        try:
            with arm_on(tag) as version:
                b = _arm_bundle(season, ref_vs, ref_b, cache)
                with _deadline(left()):
                    res = project(conn, slate["slate_id"], now, n_sims=n_sims, seed=seed,
                                  ctx=ctx, bundle=b)
            secs = round(time.monotonic() - t, 2)
            out["arms"][tag] = store(tag, res, version, secs)
            if b is ref_b:                   # arms with their own bundle load cost more
                est = max(est, secs)
        except ShadowTimeout as x:
            out["arms"][tag] = {"status": "timeout", "error": str(x),
                                "seconds": round(time.monotonic() - t, 2)}
        except BundleMissing as x:
            out["arms"][tag] = {"status": "skipped", "reason": f"bundle missing: {x}"}
        except Exception as x:  # noqa: BLE001 - one broken arm must not stop the others
            log.warning("shadow %s failed: %s", tag, x)
            out["arms"][tag] = {"status": "error", "error": f"{type(x).__name__}: {x}",
                                "seconds": round(time.monotonic() - t, 2)}
        if switch_state() != snap:        # arm_on restores; belt and braces
            restore(snap)
    return _finish(out, mpath, t0)


def _finish(out: dict, mpath: Path, t0: float) -> dict:
    out["seconds_total"] = round(time.monotonic() - t0, 2)
    try:
        mpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = mpath.with_name(mpath.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=str))
        os.replace(tmp, mpath)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot write shadow manifest %s: %s", mpath, e)
    return out


# ============================================================================== grading

def _season_label(d: date) -> str:
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def night_rows(conn: sqlite3.Connection, day: date, out_root: Path | None = None
               ) -> tuple[pd.DataFrame | None, str]:
    """Paired rows (arm vs the champion of the same run) of one night with actuals:
    (frame, 'final' | 'pending' | 'none'). Games not final are left out."""
    from . import metrics as MT
    from .dfs import grade as G
    d = (out_root or root()) / day.isoformat()
    mans = sorted(p for p in d.glob("*.json") if not p.name.startswith("graded"))
    pairs = []
    for mp in mans:
        try:
            man = json.loads(mp.read_text())
        except ValueError:
            continue
        arms = man.get("arms") or {}
        ref = arms.get(CHAMPION) or {}
        if ref.get("status") != "ok":
            continue
        try:
            rdf, rq, _ = load_arm(d / ref["file"])
        except Exception as e:  # noqa: BLE001
            log.warning("cannot read %s: %s", ref.get("file"), e)
            continue
        for tag, a in arms.items():
            if tag == CHAMPION or a.get("status") != "ok":
                continue
            try:
                adf, aq, _ = load_arm(d / a["file"])
            except Exception as e:  # noqa: BLE001
                log.warning("cannot read %s: %s", a.get("file"), e)
                continue
            pairs.append((man, tag, rdf, rq, adf, aq))
    if not pairs:
        return None, "none"
    games = sorted({g for p in pairs for g in p[2]["game_id"]})
    states = G.game_states(conn, day, games)
    final = {g for g, s in states.items() if s == "final"}
    pending = any(s == "pending" for s in states.values())
    box = G.load_box(conn, final)
    frames = []
    sal_cache: dict = {}
    for man, tag, rdf, rq, adf, aq in pairs:
        r = rdf.reset_index().rename(columns={"index": "ri"})
        a = adf.reset_index().rename(columns={"index": "ai"})
        m = r.merge(a, on=["game_id", "player_id"], suffixes=("_c", "_a"))
        m = m[m["game_id"].isin(final)]
        if m.empty:
            continue
        qc, qa = rq[m["ri"].to_numpy()], aq[m["ai"].to_numpy()]
        pid = m["player_id"].to_numpy(np.int64)
        y = np.array([box[p]["dk"] if p in box else 0.0 for p in pid])
        amin = np.array([box[p]["min"] if p in box else 0.0 for p in pid])
        played = (amin > 0).astype(int)
        changed = ((np.abs(m["dk_mean_a"] - m["dk_mean_c"]) > 1e-9)
                   | (np.abs(m["min_mean_a"] - m["min_mean_c"]) > 1e-9)
                   | (np.abs(m["p_play_a"] - m["p_play_c"]) > 1e-9)).to_numpy() \
            | (np.abs(qa - qc).max(axis=1) > 1e-6)
        mp_c, mp_a = m["min_played_c"].to_numpy(float), m["min_played_a"].to_numpy(float)
        f = pd.DataFrame({
            "day": day.isoformat(), "run": man["run"], "slot": man.get("slot", "adhoc"),
            "arm": tag, "game_id": m["game_id"].to_numpy(), "player_id": pid,
            "tier": m["tier_c"].to_numpy(), "played": played, "y": y, "amin": amin,
            "emin_c": m["min_mean_c"].to_numpy(float), "emin_a": m["min_mean_a"].to_numpy(float),
            "crps_c": MT.crps_quantiles(qc, LEVELS, y), "crps_a": MT.crps_quantiles(qa, LEVELS, y),
            "ae_c": np.abs(m["dk_mean_c"].to_numpy(float) - y),
            "ae_a": np.abs(m["dk_mean_a"].to_numpy(float) - y),
            "mae_c": np.where(played == 1, np.abs(mp_c - amin), np.nan),
            "mae_a": np.where(played == 1, np.abs(mp_a - amin), np.nan),
            "ll_c": _ll(m["p_play_c"], played), "ll_a": _ll(m["p_play_a"], played),
            "changed": changed.astype(int),
            # E4 report-only columns (EXTRA_COLS): signed E[DK] of both arms and the salary
            "dk_c": m["dk_mean_c"].to_numpy(float), "dk_a": m["dk_mean_a"].to_numpy(float),
            "salary": _salaries_for(conn, man, pid, sal_cache)})
        frames.append(f)
    if not frames:
        return None, "pending" if pending else "none"
    return pd.concat(frames, ignore_index=True), "pending" if pending else "final"


def _rows_cache(day_dir: Path) -> Path:
    return day_dir / "graded.csv.gz"


EXTRA_COLS = ("dk_c", "dk_a", "salary")     # E4 report-only columns (added 2026-10-24)
_MERGE_KEYS = ("day", "run", "arm", "game_id", "player_id")


def _salaries_for(conn: sqlite3.Connection, man: dict, pids: np.ndarray, cache: dict
                  ) -> np.ndarray:
    """DK Classic salary of each NBA player id on the manifest's slate (latest capture at or
    before the run; the first one if none is earlier). NaN when unknown. Report-only."""
    sid, at = man.get("slate_id"), man.get("at")
    out = np.full(len(pids), np.nan)
    if not sid:
        return out
    key = (str(sid), str(at or ""))
    if key not in cache:
        tab: dict = {}
        try:
            caps = [r[0] for r in conn.execute(
                "SELECT DISTINCT captured_at FROM dfs_salaries WHERE slate_id=? ORDER BY "
                "captured_at", (str(sid),))]
            if caps:
                pick = max((c for c in caps if at and c <= str(at)), default=caps[0])
                from . import ids as IDS
                dk2nba = IDS.nba_ids(conn, IDS.DK)
                for dk, sal, slot in conn.execute(
                        "SELECT site_player_id, salary, roster_slot FROM dfs_salaries WHERE "
                        "slate_id=? AND captured_at=?", (str(sid), pick)):
                    nba = dk2nba.get(str(dk))
                    if nba is None or sal is None or slot == "CPT":
                        continue
                    tab.setdefault(nba, float(sal))
        except sqlite3.Error as e:
            log.warning("shadow salaries for slate %s: %s", sid, e)
        cache[key] = tab
    tab = cache[key]
    for i, p in enumerate(pids):
        v = tab.get(int(p))
        if v is not None:
            out[i] = v
    return out


def _with_extra(conn: sqlite3.Connection, day: date, base: Path, df: pd.DataFrame,
                cache: Path) -> pd.DataFrame:
    """A cached night written before EXTRA_COLS existed: recompute the night and merge ONLY the
    new columns onto the cached rows (every cached number is kept as it was). The cache is
    rewritten when the night is final."""
    try:
        fresh, state = night_rows(conn, day, base)
    except Exception as e:  # noqa: BLE001 - report-only columns must never break grading
        log.warning("shadow extra columns for %s: %s", day, e)
        return df
    if fresh is None:
        return df
    keys = list(_MERGE_KEYS)
    f = fresh[keys + list(EXTRA_COLS)].copy()
    for k in ("day", "run", "arm", "game_id"):
        f[k] = f[k].astype(str)
    f["player_id"] = f["player_id"].astype(np.int64)
    d = df.drop(columns=[c for c in EXTRA_COLS if c in df.columns])
    left = d.assign(**{k: d[k].astype(str) for k in ("day", "run", "arm", "game_id")},
                    player_id=d["player_id"].astype(np.int64))
    out = left.merge(f.drop_duplicates(keys), on=keys, how="left")
    out = out.assign(**{k: df[k].to_numpy() for k in ("day", "run", "arm", "game_id",
                                                       "player_id")})
    if state == "final":
        tmp = cache.with_name(cache.name + ".tmp")
        out.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, cache)
    return out


def season_rows(conn: sqlite3.Connection, as_of: date, *, force: bool = False,
                out_root: Path | None = None) -> pd.DataFrame:
    """Every graded shadow row of `as_of`'s season up to `as_of` (complete nights cached as
    ``<day>/graded.csv.gz``; a night with games still pending is recomputed next time)."""
    base = out_root or root()
    season = _season_label(as_of)
    frames = []
    for dd in sorted(p for p in base.glob("????-??-??") if p.is_dir()):
        try:
            day = date.fromisoformat(dd.name)
        except ValueError:
            continue
        if day > as_of or _season_label(day) != season:
            continue
        cache = _rows_cache(dd)
        newest = max((p.stat().st_mtime for p in dd.glob("*.json")), default=0.0)
        if cache.exists() and not force and cache.stat().st_mtime >= newest:
            cached = pd.read_csv(cache, dtype={"game_id": str, "run": str})
            if not set(EXTRA_COLS) <= set(cached.columns):
                cached = _with_extra(conn, day, base, cached, cache)
            frames.append(cached)
            continue
        df, state = night_rows(conn, day, base)
        if df is None:
            continue
        if state == "final":
            tmp = cache.with_name(cache.name + ".tmp")
            df.to_csv(tmp, index=False, compression="gzip")
            os.replace(tmp, cache)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _select_run(df: pd.DataFrame, run: str) -> pd.DataFrame:
    """Rows of the registered run: a slot ("13:30", the earliest run of that slot per night) or
    "last" (per night and game, the latest run)."""
    if df.empty:
        return df
    df = df.copy()
    df["run"] = df["run"].astype(str).str.zfill(4)
    if run == "last":
        last = df.groupby(["day", "game_id"])["run"].transform("max")
        return df[df["run"] == last]
    df = df[df["slot"] == run]
    if df.empty:
        return df
    first = df.groupby("day")["run"].transform("min")
    return df[df["run"] == first]


def _population(df: pd.DataFrame, pop: str) -> pd.DataFrame:
    if pop == "emin10":
        return df[(df["emin_c"] >= EMIN) | (df["emin_a"] >= EMIN)]
    if pop == "changed":
        return df[df["changed"] == 1]
    return df


def _diff(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, np.ndarray]:
    if metric == "dk_crps":
        return df, (df["crps_a"] - df["crps_c"]).to_numpy(float)
    if metric == "dk_mae":
        return df, (df["ae_a"] - df["ae_c"]).to_numpy(float)
    if metric == "min_mae":
        df = df[(df["played"] == 1) & df["mae_a"].notna() & df["mae_c"].notna()]
        return df, (df["mae_a"] - df["mae_c"]).to_numpy(float)
    if metric == "pplay_ll":
        return df, (df["ll_a"] - df["ll_c"]).to_numpy(float)
    raise ValueError(f"unknown metric {metric}")


def paired(df: pd.DataFrame, metric: str) -> dict:
    """Mean arm - champion of `metric` with the two-way night x player clustered SE and z."""
    from . import metrics as MT
    df, d = _diff(df, metric)
    if not len(d):
        return {"n": 0, "nights": 0, "mean": None, "se": None, "z": None}
    pc = MT.paired_cluster2(d, df["day"].to_numpy(), df["player_id"].to_numpy())
    mean, se = pc["mean"], pc["se"]
    if np.isfinite(se) and se > 0:
        z = mean / se
    elif np.isfinite(se) and se == 0 and mean == 0:
        z = 0.0                      # identical on every row
    else:
        z = None
    f = lambda v: None if v is None or not np.isfinite(v) else float(v)  # noqa: E731
    return {"n": int(len(d)), "nights": int(df["day"].nunique()),
            "players": int(df["player_id"].nunique()), "mean": f(mean), "se": f(se),
            "z": f(z) if z is not None else None,
            "base": f(float(_base(df, metric))) if len(df) else None}


def _base(df: pd.DataFrame, metric: str) -> float:
    col = {"dk_crps": "crps_c", "dk_mae": "ae_c", "min_mae": "mae_c", "pplay_ll": "ll_c"}[metric]
    return float(df[col].mean())


def decide(entry: dict, prim: dict, tiers: dict, guards: list[dict]) -> tuple[str, str]:
    """(pass | fail | pending, reason) for one arm under its pre-registered bar.
    lower:     pass when z <= bar.z (e.g. -2), no tier z > bar.tier_z and every guard holds,
               from min_nights on; fail when z >= +2 (clearly worse) or max_nights pass
               without a pass; else pending.
    not_worse: from min_nights on, pass when z <= bar.z (e.g. +2) with the same tier and guard
               rules, else fail."""
    bar = entry.get("bar") or {}
    zthr, tz = float(bar.get("z", -2.0)), float(bar.get("tier_z", 2.0))
    minn, maxn = int(entry.get("min_nights", 14)), int(entry.get("max_nights", 60))
    nights, z = prim.get("nights", 0), prim.get("z")
    if nights < minn:
        return "pending", f"{nights} of {minn} nights"
    if prim.get("n", 0) < int(entry.get("min_rows", 0)):
        return "pending", f"{prim.get('n', 0)} of {entry['min_rows']} rows"
    if z is None:
        return "pending", "no standard error yet"
    bad_t = [t for t, v in tiers.items() if v.get("z") is not None and v["z"] > tz]
    bad_g = [f"{g['metric']} z {g['z']:+.2f}" for g in guards
             if g.get("z") is not None and g["z"] > float(g.get("bar_z", 2.0))]
    ok = z <= zthr and not bad_t and not bad_g
    if ok:
        return "pass", f"z {z:+.2f} <= {zthr:+.1f} over {nights} nights; no tier above {tz:+.1f}"
    why = ([f"z {z:+.2f} > {zthr:+.1f}"] if z > zthr else []) \
        + [f"tier {t} z {tiers[t]['z']:+.2f}" for t in bad_t] + bad_g
    if bar.get("direction", "lower") == "not_worse":
        return "fail", "; ".join(why)
    if z >= 2.0:
        return "fail", f"clearly worse (z {z:+.2f})"
    if nights >= maxn:
        return "fail", f"no pass by {maxn} nights ({'; '.join(why)})"
    return "pending", "bar not met yet: " + "; ".join(why)


def arm_report(rows: pd.DataFrame, entry: dict) -> dict:
    """One arm's numbers and decision (rows = every graded row of the season, all arms)."""
    keys = ("arm", "switch", "hypothesis", "metric", "population", "run", "bar", "min_nights",
            "max_nights", "min_rows", "window", "status")
    out = {k: entry.get(k) for k in keys if entry.get(k) is not None}
    sub = rows[rows["arm"] == entry["arm"]] if len(rows) else rows
    win = entry.get("window") or {}
    if len(sub) and win.get("until"):
        sub = sub[sub["day"] <= win["until"]]
    if len(sub) and win.get("from"):
        sub = sub[sub["day"] >= win["from"]]
    at_run = _select_run(sub, entry.get("run", "13:30")) if len(sub) else sub
    pop = _population(at_run, entry.get("population", "emin10")) if len(at_run) else at_run
    metric = entry.get("metric", "dk_crps")
    prim = paired(pop, metric) if len(pop) else {"n": 0, "nights": 0, "mean": None, "se": None,
                                                  "z": None}
    tiers = {}
    for t in TIERS:
        x = pop[pop["tier"] == t] if len(pop) else pop
        if len(x):
            tiers[t] = paired(x, metric)
    guards = []
    for g in entry.get("guard") or []:
        gp = _population(at_run, g.get("population", "emin10")) if len(at_run) else at_run
        r = paired(gp, g["metric"]) if len(gp) else {"n": 0, "z": None}
        guards.append({**g, "bar_z": g.get("z", 2.0), **{k: r.get(k) for k in
                                                          ("n", "mean", "se", "z")}})
    status, reason = decide(entry, prim, tiers, guards)
    sec = {}
    if len(at_run):
        e10 = _population(at_run, "emin10")
        for mname in ("dk_crps", "dk_mae"):
            if (mname, "emin10") != (metric, entry.get("population")) and len(e10):
                sec[mname] = paired(e10, mname)
    by_night = []
    if len(pop):
        dd, d = _diff(pop, metric)
        for day, x in dd.assign(_d=d).groupby("day"):
            by_night.append({"day": day, "n": int(len(x)), "mean": float(x["_d"].mean())})
    return {**out, "decision": status, "reason": reason, "primary": prim, "tiers": tiers,
            "guards": guards, "secondary": sec, "by_night": by_night,
            "reported_only": reported_only(at_run),
            "changed_share": (float(at_run["changed"].mean()) if len(at_run) else None),
            "rows_all_runs": int(len(sub))}


def reported_only(at_run: pd.DataFrame) -> dict:
    """E4 metrics shown next to an arm, never part of its decision (no arm's bar changes):
    on the DFS-relevant rows (salary >= $4k or E[DK] >= 15 in either arm,
    `metrics.dfs_population`) the paired DK CRPS (arm - champion, two-way SE) and the
    value-ranking metric corr(E[DK] - salary-implied, actual - salary-implied) of each, the
    salary-implied line fit per night on the slate's rows (`metrics.value_ranking`, SE
    clustered by night), arm minus champion. {} until the rows carry EXTRA_COLS."""
    from . import metrics as MT
    if not len(at_run) or not set(EXTRA_COLS) <= set(at_run.columns):
        return {}
    sal = at_run["salary"].to_numpy(float)
    m = MT.dfs_population(sal, at_run["dk_c"].to_numpy(float), at_run["dk_a"].to_numpy(float))
    d = at_run[m]
    out: dict = {"population": f"salary >= {MT.DFS_SAL_MIN:.0f} or E[DK] >= "
                               f"{MT.DFS_PROJ_MIN:.0f} (either arm)", "n": int(len(d))}
    if not len(d):
        return out
    out["dk_crps_dfs"] = paired(d, "dk_crps")
    v = d[np.isfinite(d["salary"].to_numpy(float))]
    if len(v) >= 3:
        day = v["day"].astype(str).to_numpy()
        vr = MT.value_ranking({"arm": v["dk_a"].to_numpy(float),
                               "champion": v["dk_c"].to_numpy(float)}, v["y"].to_numpy(float),
                              v["salary"].to_numpy(float), day, day, base="champion")
        f = lambda x: None if x is None or not np.isfinite(x) else float(x)  # noqa: E731
        vb = vr["vs_base"].get("arm") or {}
        out["value"] = {"n": vr["n"], "nights": vr["n_clusters"],
                        "r_arm": f(vb.get("r_a")), "r_champion": f(vb.get("r_b")),
                        "diff": f(vb.get("diff")), "se": f(vb.get("se")),
                        "z": f(vb.get("z")) if vb.get("z") is not None else None}
    return out


def run_log(out_root: Path, as_of: date, days: int = 7) -> list[dict]:
    """The shadow runs of the last `days` nights (status and seconds per arm)."""
    out = []
    lo = as_of - timedelta(days=days - 1)
    for dd in sorted(p for p in out_root.glob("????-??-??") if p.is_dir()):
        try:
            day = date.fromisoformat(dd.name)
        except ValueError:
            continue
        if not lo <= day <= as_of:
            continue
        for mp in sorted(dd.glob("[0-9]*.json")):
            try:
                m = json.loads(mp.read_text())
            except ValueError:
                continue
            out.append({"day": m.get("day"), "run": m.get("run"), "slot": m.get("slot"),
                        "status": m.get("status"), "reason": m.get("reason"),
                        "seconds_total": m.get("seconds_total"),
                        "slate_id": m.get("slate_id"), "n_games": len(m.get("games") or []),
                        "arms": {k: {"status": v.get("status"), "seconds": v.get("seconds"),
                                     "reason": v.get("reason") or v.get("error")}
                                 for k, v in (m.get("arms") or {}).items()}})
    return out


def grade(conn: sqlite3.Connection, as_of: date, *, force: bool = False,
          registry: dict | None = None, out_root: Path | None = None,
          data_dir: Path | None = None) -> dict:
    """Grade every shadow night of the season up to `as_of` and write the report."""
    reg = registry or load_registry()
    base = out_root or root()
    rows = season_rows(conn, as_of, force=force, out_root=base)
    rep = {"as_of": as_of.isoformat(), "generated_at": _iso(datetime.now(timezone.utc)),
           "season": _season_label(as_of), "registered": reg.get("registered"),
           "nights": sorted(rows["day"].unique().tolist()) if len(rows) else [],
           "n_rows": int(len(rows)),
           "arms": [arm_report(rows, e) for e in reg["arms"]],
           "runs": run_log(base, as_of)}
    d = report_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(rep, indent=1, default=str)
    for name in (REPORT_NAME, f"shadow_grades_{as_of.isoformat()}.json"):
        tmp = d / (name + ".tmp")
        tmp.write_text(txt)
        os.replace(tmp, d / name)
    return rep


def postgame_step(conn: sqlite3.Connection, day: date) -> dict:
    """The postgame job's ``shadow_grade`` step: never raises (status 'error' instead)."""
    try:
        if not root().exists():
            return {"status": "skipped", "reason": "no shadow runs yet"}
        rep = grade(conn, day)
        return {"nights": len(rep["nights"]), "rows": rep["n_rows"],
                "decisions": {a["arm"]: a["decision"] for a in rep["arms"]}}
    except Exception as e:  # noqa: BLE001 - grading shadows must never fail the job
        log.exception("shadow grade failed")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def load_report(conn: sqlite3.Connection | None = None, data_dir: Path | None = None) -> dict:
    """The latest report ({} if none). With `conn`, the data dir is the database's folder."""
    if data_dir is None and conn is not None:
        for row in conn.execute("PRAGMA database_list"):
            if row[1] == "main" and row[2]:
                data_dir = Path(row[2]).parent
    p = report_dir(data_dir) / REPORT_NAME
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ============================================================================== CLI

def readonly_connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or config.db_path())
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def add_commands(sub) -> dict:
    p = sub.add_parser("shadows", help="player-side live shadows: run (pregame) / grade / status")
    p.add_argument("action", choices=("run", "grade", "status"))
    p.add_argument("--at", default=None, help="run time (ISO, default now)")
    p.add_argument("--date", default=None, help="grade: NBA day (default: yesterday)")
    p.add_argument("--arms", default=None, help="comma list, e.g. +tr,+ep (default: registry)")
    p.add_argument("--budget", type=float, default=None, help="seconds (default 90)")
    p.add_argument("--sims", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-run / re-grade")
    p.add_argument("--no-guard", action="store_true", help="skip the lock-window check")
    return {"shadows": _cmd}


def _cmd(args) -> int:
    if args.action == "status":
        print(json.dumps(load_report(data_dir=config.data_dir()), indent=1)[:20000])
        return 0
    if not config.db_path().exists():
        print("no database")
        return 1
    conn = readonly_connect()
    try:
        if args.action == "run":
            at = datetime.fromisoformat(args.at.replace("Z", "+00:00")) if args.at else None
            arms = [a for a in args.arms.split(",") if a.strip()] if args.arms else None
            res = run(conn, at, arms=arms, budget=args.budget, n_sims=args.sims,
                      guard=not args.no_guard, force=args.force)
            print(json.dumps(res, default=str))
            return 0 if res.get("status") in ("ok", "skipped", "disabled") else 1
        day = date.fromisoformat(args.date) if args.date else config.yesterday_nba_day()
        rep = grade(conn, day, force=args.force)
        for a in rep["arms"]:
            p = a["primary"]
            z = "" if p.get("z") is None else f"z {p['z']:+.2f}"
            print(f"{a['arm']:<5} {a['metric']:<9} {a.get('run', ''):<6} nights {p['nights']:>3} "
                  f"n {p['n']:>6} {z:<10} {a['decision']:<8} {a['reason']}")
        return 0
    finally:
        conn.close()
