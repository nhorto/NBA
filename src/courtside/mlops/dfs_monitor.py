"""DFS monitoring: registry specs and alerts (docs/PLAYERS.md §8 Phase 3, task T3.1).

Nightly, after ``dfs_grade`` wrote its snapshots (`monitor.run_after_job` calls `run`):

* **Registry** (`sync_registry`): the four DFS components ``player_minutes``, ``player_rates``,
  ``player_dist`` and ``dfs_engine`` are registered in ``model_versions`` like the game models,
  one spec per distinct ``dfs_runs.model_version`` (Classic / Showdown / lock runs; the DK-FPPG
  fallback and historical ``rg:`` validation runs are not our model and are skipped). The newest
  version is the champion, earlier ones are retired, every change is logged in
  ``registry_events``. Today one pipeline version drives all three projection components, so they
  move together; the engine spec also carries the runner's constants (cash lambda, P(play) cut).
* **Alerts** (`desired_alerts`, called by ``alerts.desired_alerts`` so the usual one-open-row-per-key
  dedupe and auto-resolve apply). All have kind ``dfs``:

  ============================  =====================================================================
  key                           fires when
  ============================  =====================================================================
  ``dfs:min_bias``              mean (projected - actual) minutes of projected starters (E[min | plays]
                                >= 26, played) over the last 300 such rows is beyond +-1.0, n >= 100
  ``dfs:cov80``                 DK-points 80% interval coverage over the last 300 projection rows
                                (projected minutes >= 10) is outside 0.80 +- 3 SE, n >= 200 (SE: the
                                larger of the per-game-clustered and the binomial one)
  ``dfs:missed_lock:<slate>``   a slate locked >= 15 minutes ago (last 4 days) has no ok lock45 / lock10
                                run: critical when neither exists, warn when only lock10 is missing
  ``dfs:recon_gap``             a ``proj_recon_gap:*`` data check of the last 2 days is not ok (team
                                total reconciliation residual > 0.5, or pre-reconciliation gap > 6)
  ``dfs:live_min_mae``          live minutes MAE (played, projected >= 15, last 300 rows, n >= 150)
                                exceeds the live-like baseline + 0.5. The baseline is set by T2.4:
                                ``$COURTSIDE_LIVE_LIKE_MIN_MAE`` or ``<data>/proj/live_like_baseline.json``
                                (``{"min_mae": 4.7}``); until then the rule is dormant
  ``dfs:fallback``              a lock run of the last 3 days used DK's FPPG instead of our projections
  ============================  =====================================================================

  Covered by the existing data-check alerts (``data:<check>``, warn at the first failure, critical
  at the third): the FPPG scoring reconciliation (``dfs_scoring_fppg``) and the DK id match
  (``ids_dk``, which fails on any unmatched likely-rotation player and warns below 98%). The per-run
  ``proj_recon_gap:<run>`` checks are handled here instead (`handles_check`), as one alert.

Each rule is guarded: if one raises, its open alerts are carried over unchanged (never
spuriously resolved) and the error is logged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .. import config

log = logging.getLogger("courtside.mlops.dfs_monitor")

COMPONENTS = ("player_minutes", "player_rates", "player_dist", "dfs_engine")
COMPONENT_LABEL = {"player_minutes": "Player minutes", "player_rates": "Player rates",
                   "player_dist": "Player distribution", "dfs_engine": "DFS engine"}
FALLBACK_VERSION = "fppg-fallback"

LAST_N = 300
MIN_BIAS_LIMIT = 1.0             # minutes
MIN_BIAS_MIN_N = 100
STARTER_MIN = 26.0               # E[min | plays]: the top two minute tiers
COV_TARGET, COV_SE_K, COV_MIN_N = 0.80, 3.0, 200
LIVE_MAE_MARGIN, LIVE_MAE_MIN_N = 0.5, 150
LOCK_GRACE_MIN = 15
LOCK_LOOKBACK_DAYS = 4
RECON_LOOKBACK_DAYS = 2
FALLBACK_LOOKBACK_DAYS = 3


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _a(key, severity, message, details=None) -> dict:
    return {"key": key, "severity": severity, "kind": "dfs", "message": message,
            "details": details or {}}


def is_fallback(model_version: str | None) -> bool:
    mv = (model_version or "").lower()
    return not mv or mv.startswith("fppg") or "fallback" in mv


def handles_check(name: str) -> bool:
    """Data checks that this module turns into its own alert (the generic data alert skips them)."""
    return str(name).startswith("proj_recon_gap:")


# ============================================================================== registry

def run_versions(conn: sqlite3.Connection) -> list[dict]:
    """[{model_version, first, last, runs}] of our own ok DFS runs, oldest first."""
    rows = conn.execute(
        "SELECT model_version, MIN(created_at) AS first, MAX(created_at) AS last, COUNT(*) AS runs "
        "FROM dfs_runs WHERE ok=1 AND model_version IS NOT NULL AND model_version != '' "
        "AND COALESCE(slate_id, '') NOT LIKE 'rg:%' GROUP BY model_version ORDER BY first").fetchall()
    return [dict(r) for r in rows if not is_fallback(r["model_version"])]


def _engine_constants() -> dict:
    try:
        from ..proj.dfs import runner
        return {"cash_lambda": runner.CASH_LAMBDA, "p_play_min": runner.P_PLAY_MIN,
                "topn_default": 10}
    except Exception:  # noqa: BLE001 - the spec still registers without the constants
        return {}


def component_params(component: str, model_version: str) -> dict:
    p = {"component": component, "model_version": model_version, "source": "dfs_runs.model_version"}
    if component == "dfs_engine":
        p.update(_engine_constants())
    return p


def sync_registry(conn: sqlite3.Connection) -> dict:
    """Register every seen pipeline version for the four components; newest = champion.
    Idempotent. Returns {component: {'champion': id, 'versions': [ids]}}."""
    from . import registry
    versions = run_versions(conn)
    out: dict = {}
    if not versions:
        return out
    for comp in COMPONENTS:
        ids = []
        for v in versions:
            ids.append(registry.register(
                conn, comp, component_params(comp, v["model_version"]), None,
                code=v["model_version"], status="candidate",
                notes=f"DFS pipeline version {v['model_version']} (first live run {v['first']})",
                reason=f"first {comp} run with pipeline {v['model_version']}"))
        newest = ids[-1]
        cur = registry.get_version(conn, newest)
        if cur and cur["status"] != "champion":
            registry.promote(conn, newest, f"newest DFS pipeline version "
                                           f"{versions[-1]['model_version']}")
        for vid in ids[:-1]:
            row = registry.get_version(conn, vid)
            if row and row["status"] in ("candidate", "shadow"):
                registry.set_status(conn, vid, "retired", "superseded by a newer DFS pipeline version")
        out[comp] = {"champion": newest, "versions": ids}
    return out


def component_versions(conn) -> list[dict]:
    """model_versions rows of the DFS components (newest first), for the health page."""
    ph = ",".join("?" * len(COMPONENTS))
    try:
        return [dict(r) for r in conn.execute(
            f"SELECT * FROM model_versions WHERE model_name IN ({ph}) ORDER BY created_at DESC",
            COMPONENTS)]
    except sqlite3.Error:
        return []


# ============================================================================== data

def _player_rows(conn, as_of: date):
    """Graded projection rows (one per game x player, lock10 preferred), newest day first."""
    from ..proj.dfs import grade
    lo = grade._season_first_day(as_of) - timedelta(days=60)
    df = grade._player_frame(conn, lo, as_of)
    if df.empty:
        return df
    return df.sort_values(["day", "game_id", "player_id"], ascending=[False, True, True]
                          ).reset_index(drop=True)


def live_like_baseline() -> float | None:
    """The live-like minutes MAE T2.4 measured (env or var/proj/live_like_baseline.json)."""
    env = os.environ.get("COURTSIDE_LIVE_LIKE_MIN_MAE")
    if env:
        try:
            return float(env)
        except ValueError:
            return None
    p = config.data_dir() / "proj" / "live_like_baseline.json"
    try:
        v = json.loads(p.read_text())
        x = v.get("min_mae") if isinstance(v, dict) else v
        return float(x) if x is not None else None
    except (OSError, ValueError, TypeError):
        return None


def _clustered_se(values: np.ndarray, clusters: np.ndarray) -> float | None:
    from ..proj import metrics as MT
    if len(values) < 10 or len(set(clusters.tolist())) < 5:
        return None
    pc = MT.paired_cluster(values, clusters)
    se = pc["se"]
    return float(se) if se is not None and np.isfinite(se) else None


# ============================================================================== rules

def _rule_min_bias(conn, as_of: date, df) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df[(df["played"] == 1) & (df["proj_min_played"] >= STARTER_MIN)
           & df["actual_min"].notna()].head(LAST_N)
    n = len(d)
    if n < MIN_BIAS_MIN_N:
        return []
    e = (d["proj_min_played"] - d["actual_min"]).to_numpy(float)
    bias = float(e.mean())
    se = _clustered_se(e, d["game_id"].to_numpy()) or float(e.std(ddof=1) / math.sqrt(n))
    se = se if se and se > 1e-9 else 1e-9
    if abs(bias) <= MIN_BIAS_LIMIT:
        return []
    what = "over-projecting" if bias > 0 else "under-projecting"
    return [_a("dfs:min_bias", "warn",
               f"Minutes engine is {what} projected starters by {abs(bias):.1f} min when they "
               f"play (last {n} starter-games, {abs(bias) / se:.1f} SE; alert beyond "
               f"{MIN_BIAS_LIMIT:.1f})",
               {"bias": bias, "se": se, "n": n, "limit": MIN_BIAS_LIMIT,
                "population": f"E[min | plays] >= {STARTER_MIN:g}, played"})]


def _rule_cov80(conn, as_of: date, df) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df[(df["proj_min"] >= 10) & df["proj_dk"].notna() & df["in80"].notna()].head(LAST_N)
    n = len(d)
    if n < COV_MIN_N:
        return []
    x = d["in80"].to_numpy(float)
    cov = float(x.mean())
    se_b = math.sqrt(COV_TARGET * (1 - COV_TARGET) / n)
    se_c = _clustered_se(x, d["game_id"].to_numpy())
    se = max(se_b, se_c or 0.0)
    if abs(cov - COV_TARGET) <= COV_SE_K * se:
        return []
    side = "too narrow (too many results outside the range)" if cov < COV_TARGET else \
        "too wide"
    return [_a("dfs:cov80", "warn",
               f"The 80% range for DK points covered {cov:.1%} of the last {n} player results, "
               f"not 80%: the ranges are {side} ({abs(cov - COV_TARGET) / se:.1f} SE; alert "
               f"beyond {COV_SE_K:g} SE)",
               {"coverage": cov, "se": se, "n": n, "target": COV_TARGET})]


def _rule_live_min_mae(conn, as_of: date, df) -> list[dict]:
    base = live_like_baseline()
    if base is None or df is None or df.empty:
        return []
    d = df[(df["played"] == 1) & (df["proj_min_played"] >= 15)
           & df["actual_min"].notna()].head(LAST_N)
    n = len(d)
    if n < LIVE_MAE_MIN_N:
        return []
    mae = float(np.abs(d["proj_min_played"] - d["actual_min"]).mean())
    if mae <= base + LIVE_MAE_MARGIN:
        return []
    return [_a("dfs:live_min_mae", "warn",
               f"Live minutes error is {mae:.2f} min (players projected 15+ min, last {n} "
               f"games), more than {LIVE_MAE_MARGIN:g} above the live-like backtest's "
               f"{base:.2f}",
               {"mae": mae, "baseline": base, "margin": LIVE_MAE_MARGIN, "n": n})]


def _et(ts: str) -> str:
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return ts
    t = dt.astimezone(config.NBA_TZ)
    return f"{t:%b} {t.day}, {t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'} ET"


def slate_run_status(conn, slate_id: str) -> dict:
    """{'lock45': n ok runs, 'lock10': n ok runs, 'failed': n failed lock runs, 'fallback': n}."""
    out = {"lock45": 0, "lock10": 0, "failed": 0, "fallback": 0}
    for r in conn.execute("SELECT kind, ok, model_version FROM dfs_runs WHERE slate_id=? "
                          "AND kind IN ('lock45','lock10')", (str(slate_id),)):
        if r["ok"]:
            out[r["kind"]] += 1
            out["fallback"] += int(is_fallback(r["model_version"]))
        else:
            out["failed"] += 1
    return out


def _rule_missed_lock(conn, as_of: date, now: datetime) -> list[dict]:
    lo = _iso(datetime.combine(as_of - timedelta(days=LOCK_LOOKBACK_DAYS), datetime.min.time(),
                               tzinfo=timezone.utc))
    cut = _iso(now - timedelta(minutes=LOCK_GRACE_MIN))
    out = []
    for s in conn.execute(
            "SELECT slate_id, tag, game_type_id, lock_utc, nba_day FROM dfs_slates "
            "WHERE lock_utc IS NOT NULL AND lock_utc>=? AND lock_utc<=? AND nba_day<=? "
            "AND EXISTS (SELECT 1 FROM dfs_salaries d WHERE d.slate_id=dfs_slates.slate_id) "
            "ORDER BY lock_utc", (lo, cut, as_of.isoformat())):
        st = slate_run_status(conn, s["slate_id"])
        if st["lock10"]:
            continue
        label = (s["tag"] or "").capitalize() or "Slate"
        both = not st["lock45"]
        miss = "no lock-45 or lock-10 run" if both else "no lock-10 run (the lock-45 build stands)"
        tail = f"; {st['failed']} failed attempt(s)" if st["failed"] else ""
        out.append(_a(f"dfs:missed_lock:{s['slate_id']}", "critical" if both else "warn",
                      f"{label} slate {s['slate_id']} locked {_et(s['lock_utc'])} with {miss}"
                      f"{tail}",
                      {"slate_id": s["slate_id"], "lock_utc": s["lock_utc"], **st}))
    return out


def _rule_recon_gap(conn, as_of: date, now: datetime) -> list[dict]:
    since = _iso(datetime.combine(as_of - timedelta(days=RECON_LOOKBACK_DAYS),
                                  datetime.min.time(), tzinfo=timezone.utc))
    bad = []
    for r in conn.execute("SELECT run_at, check_name, status, value, threshold, detail FROM "
                          "data_checks WHERE check_name LIKE 'proj_recon_gap:%' AND run_at>=? "
                          "AND status NOT IN ('ok','skip') ORDER BY run_at DESC", (since,)):
        try:
            det = json.loads(r["detail"]) if r["detail"] else {}
        except ValueError:
            det = {}
        bad.append({"run": r["check_name"].split(":", 1)[1], "gap": r["value"],
                    "resid": det.get("max_abs_resid") if isinstance(det, dict) else None,
                    "teams": (det.get("teams") if isinstance(det, dict) else None) or [],
                    "run_at": r["run_at"]})
    if not bad:
        return []
    worst = max(bad, key=lambda b: (b["resid"] or 0.0, b["gap"] or 0.0))
    teams = ", ".join(str(t.get("team")) for t in worst["teams"][:4] if t.get("team"))
    return [_a("dfs:recon_gap", "warn",
               f"Team-total reconciliation is off in {len(bad)} recent projection run"
               f"{'s' if len(bad) != 1 else ''}: pre-reconciliation gap up to "
               f"{max((b['gap'] or 0.0) for b in bad):.1f} points"
               + (f", residual {worst['resid']:.2f} (limit 0.5)" if worst["resid"] is not None else "")
               + (f", teams {teams}" if teams else "")
               + ". A stale injury report or an unmatched player name is the usual cause.",
               {"runs": bad[:8]})]


def _rule_fallback(conn, as_of: date, now: datetime) -> list[dict]:
    since = _iso(now - timedelta(days=FALLBACK_LOOKBACK_DAYS))
    rows = conn.execute(
        "SELECT run_id, slate_id, kind, created_at, model_version FROM dfs_runs WHERE ok=1 AND "
        "kind IN ('lock45','lock10') AND created_at>=? AND COALESCE(slate_id,'') NOT LIKE 'rg:%'",
        (since,)).fetchall()
    fb = [r for r in rows if is_fallback(r["model_version"])]
    if not fb:
        return []
    return [_a("dfs:fallback", "warn",
               f"{len(fb)} of {len(rows)} recent lock runs used DraftKings' own FPPG instead of "
               f"our projections (the projection pipeline failed or was missing). Those lineups "
               f"are not ours to grade.",
               {"runs": [{"run_id": r["run_id"], "slate_id": r["slate_id"], "kind": r["kind"],
                          "created_at": r["created_at"]} for r in fb[:10]]})]


RULES = (("dfs:min_bias", "rows", _rule_min_bias), ("dfs:cov80", "rows", _rule_cov80),
         ("dfs:live_min_mae", "rows", _rule_live_min_mae),
         ("dfs:missed_lock", "now", _rule_missed_lock), ("dfs:recon_gap", "now", _rule_recon_gap),
         ("dfs:fallback", "now", _rule_fallback))


def _carry_over(conn, prefix: str) -> list[dict]:
    out = []
    try:
        for r in conn.execute("SELECT key, severity, kind, message, details_json FROM alerts "
                              "WHERE status='open' AND key LIKE ?", (prefix + "%",)):
            out.append(_a(r["key"], r["severity"], r["message"],
                          json.loads(r["details_json"]) if r["details_json"] else {}))
    except (sqlite3.Error, ValueError):
        pass
    return out


def desired_alerts(conn: sqlite3.Connection, as_of, *, now: datetime | None = None) -> list[dict]:
    """The DFS alerts that should be open (see the module docstring)."""
    as_of = _as_date(as_of)
    now = now or _now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    df = None
    df_err = False
    try:
        df = _player_rows(conn, as_of)
    except Exception:  # noqa: BLE001
        log.exception("dfs monitor: cannot read graded rows")
        df_err = True
    out: list[dict] = []
    for prefix, needs, fn in RULES:
        try:
            if needs == "rows":
                if df_err:
                    raise RuntimeError("graded rows unavailable")
                out += fn(conn, as_of, df)
            else:
                out += fn(conn, as_of, now)
        except Exception:  # noqa: BLE001 - a broken rule must not resolve its open alerts
            log.exception("dfs monitor rule %s failed; keeping its open alerts", prefix)
            out += _carry_over(conn, prefix)
    return out


# ============================================================================== job hook

def run(conn: sqlite3.Connection, as_of) -> dict:
    """Postgame hook (after ``dfs_grade``): sync the registry. Alerts are evaluated by
    ``alerts.evaluate`` right after, through `desired_alerts`."""
    res = sync_registry(conn)
    return {"registry": {k: v["champion"] for k, v in res.items()}}


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("monitor", help="sync the DFS registry specs and show the DFS alerts "
                                       "(--evaluate also writes them to the alerts table)")
    p.add_argument("--as-of", default=None, help="day (default: today)")
    p.add_argument("--evaluate", action="store_true", help="reconcile the alerts table")
    return {"monitor": _monitor_cmd}


def _monitor_cmd(args) -> int:
    from .. import db
    from . import alerts
    conn = db.connect()
    as_of = _as_date(args.as_of) if args.as_of else config.today_nba_day()
    res = sync_registry(conn)
    for comp, v in res.items():
        print(f"{comp}: champion {v['champion']} ({len(v['versions'])} version(s))")
    if not res:
        print("no DFS runs yet: nothing to register")
    if args.evaluate:
        alerts.evaluate(conn, as_of)
    want = desired_alerts(conn, as_of)
    print(f"\n{len(want)} DFS alert(s) for {as_of}")
    for a in want:
        print(f"  [{a['severity']}] {a['key']}: {a['message']}")
    return 0
