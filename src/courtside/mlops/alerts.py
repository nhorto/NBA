"""Alert rules -> `alerts` (one open row per key; re-firing updates, clearing resolves).

`evaluate(conn, as_of)` recomputes every alert condition from the stored snapshots/checks and
reconciles the table. It is idempotent: running it twice changes nothing the second time.

Rules (windows are from `metric_snapshots`; "n" is the window's game count)
---------------------------------------------------------------------------
performance   key perf:<model>    last300 ll_diff_vs_mkt > 2 * ll_diff_se, with n >= 100
                                   (market-using models only; gbm_opinion / elo_sched are
                                   market-free by design and are not held to the market)
perf_band     key band:<model>    last300 log_loss > benchmark + 2 * sqrt(ll_se^2 + 0.015^2),
                                   n >= 100. Benchmark = `model_versions.benchmark_json` of the
                                   model's champion/shadow version when present, else the
                                   docs/MODEL.md last-3-season walk-forward constants. The 0.015
                                   is the season-to-season spread of log-loss, so the tolerance
                                   widens for a short window and is never tighter than a season.
calibration   key calib:<model>   Spiegelhalter z-test, |z| > 2.5 with n >= 200 on last300 OR on
                                   season (critical if either |z| > 3.5); the message reports
                                   both z values plus the logistic calibration slope. z > 0 =
                                   overconfident. ECE is displayed but is NOT an alert: at
                                   n=300 the market itself scores ~0.065, so any ECE threshold
                                   is noise; the z-test has a known null distribution.
bias          key bias:<model>:home|margin|total
                                   last300 |bias| > 2 SE, n >= 100 (home: predicted win rate vs
                                   actual; margin / total: predicted - actual, in points)
data          key data:<check>    consecutive non-ok results of a check (skips ignored):
                                   fail -> warn at 1, critical at 3; warn -> warn at 2, critical at 3
Metric rules read the newest snapshot <= as_of and are ignored if it is > 10 days old.
drift         key drift:<feature> latest PSI > 0.25 warn, > 0.5 critical (snapshot <= 7 days old)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import date, datetime, timezone

from . import drift as drift_mod
from . import monitor

log = logging.getLogger("courtside.mlops.alerts")

MIN_N = 100
CALIB_MIN_N = 200
CALIB_Z_WARN, CALIB_Z_CRIT = 2.5, 3.5
SE_K = 2.0
PSI_WARN, PSI_CRIT = 0.25, 0.5
DATA_WARN_RUNS, DATA_CRIT_RUNS = 2, 3
SEASON_LL_SD = 0.015
DRIFT_MAX_AGE_DAYS = 7
METRICS_MAX_AGE_DAYS = 10     # older metric snapshots (offseason, stalled monitor) raise no alerts
MARKET_FREE = {"gbm_opinion", "elo_sched"}
# docs/MODEL.md "Rung 3": regular-season walk-forward log-loss over the last 3 test seasons
DOC_BENCHMARKS = {"gbm_market": 0.5790, "gbm_opinion": 0.5876, "elo_sched": 0.6044}
KINDS = ("performance", "calibration", "bias", "data", "drift")
_SEV = {"info": 0, "warn": 1, "critical": 2}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# ------------------------------------------------------------------ benchmarks

_SUBKEYS = ("recent", "last3", "recent_season", "walk_forward", "metrics", "benchmark", "overall")


def _dig(d, key: str):
    if not isinstance(d, dict):
        return None
    if isinstance(d.get(key), (int, float)):
        return float(d[key])
    for sub in _SUBKEYS:
        v = _dig(d.get(sub), key)
        if v is not None:
            return v
    for v in d.values():
        if isinstance(v, dict):
            r = _dig(v, key)
            if r is not None:
                return r
    return None


def benchmark_ll(conn: sqlite3.Connection, label: str, version: str | None
                 ) -> tuple[float | None, str]:
    """(expected log-loss, where it came from) for a model label."""
    model = label.split("@", 1)[0]
    try:
        if "@" in label:
            row = conn.execute("SELECT benchmark_json FROM model_versions WHERE version_id=?",
                               (label.split("@", 1)[1],)).fetchone()
        else:
            row = conn.execute("SELECT benchmark_json FROM model_versions WHERE model_name=? "
                               "AND status='champion' ORDER BY created_at DESC LIMIT 1",
                               (model,)).fetchone()
        if row and row[0]:
            v = _dig(json.loads(row[0]), "log_loss")
            if v is not None:
                return v, "the champion's registered backtest benchmark"
    except (sqlite3.Error, ValueError):
        pass
    v = DOC_BENCHMARKS.get(model)
    return v, "the last-3-season walk-forward result in docs/MODEL.md"


# ------------------------------------------------------------------ rule families

def _a(key, severity, kind, message, details=None) -> dict:
    return {"key": key, "severity": severity, "kind": kind, "message": message,
            "details": details or {}}


def _metric_alerts(conn: sqlite3.Connection, as_of: date) -> list[dict]:
    out: list[dict] = []
    snaps = monitor.snapshots(conn, as_of)
    if snaps and (as_of - date.fromisoformat(snaps[0]["as_of"])).days > METRICS_MAX_AGE_DAYS:
        return []
    season = {r["model"]: r for r in snaps if r["window"] == "season"}
    for r in snaps:
        if r["window"] != "last300" or r["model"] == monitor.MARKET:
            continue
        model, n, det = r["model"], r["n"] or 0, r["details"]
        model_base = model.split("@", 1)[0]
        games = f"the last {n} games" if n < 300 else "the last 300 games"
        # performance vs the market
        d, se = r["ll_diff_vs_mkt"], r["ll_diff_se"]
        mkt_n = r["mkt_n"] or 0
        if (model_base not in MARKET_FREE and d is not None and se and mkt_n >= MIN_N
                and d > SE_K * se):
            out.append(_a(f"perf:{model}", "warn", "performance",
                          f"{model} is scoring worse than the betting market: log-loss "
                          f"{d:+.4f} versus the market over {games} ({d / se:.1f} standard "
                          f"errors, same games)",
                          {"model": model, "ll_diff": d, "se": se, "n": mkt_n, "window": "last300"}))
        # performance vs the backtest expectation
        bench, src = benchmark_ll(conn, model, r["model_version"])
        ll_se = det.get("ll_se")
        if bench is not None and r["log_loss"] is not None and ll_se and n >= MIN_N:
            tol = SE_K * math.sqrt(ll_se ** 2 + SEASON_LL_SD ** 2)
            if r["log_loss"] > bench + tol:
                out.append(_a(f"band:{model}", "warn", "performance",
                              f"{model} log-loss is {r['log_loss']:.4f} over {games}, worse than "
                              f"the backtest expectation of {bench:.4f} by more than the "
                              f"allowed {tol:.4f} (benchmark: {src})",
                              {"model": model, "log_loss": r["log_loss"], "benchmark": bench,
                               "tolerance": tol, "n": n}))
        # calibration: Spiegelhalter z on last300 and on the season
        z300 = det.get("spiegelhalter_z") if n >= CALIB_MIN_N else None
        sr = season.get(model)
        zs = (sr["details"].get("spiegelhalter_z") if sr and (sr["n"] or 0) >= CALIB_MIN_N
              else None)
        sig = [(w, z, row) for w, z, row in (("last 300 games", z300, r), ("the season", zs, sr))
               if z is not None and abs(z) > CALIB_Z_WARN]
        if sig:
            w, z, row = sig[0]
            slope = row["details"].get("cal_slope")
            over = (slope < 1) if slope is not None else (z > 0)
            what = ("overconfident: its favorites win less often than predicted" if over else
                    "underconfident: its favorites win more often than predicted")
            parts = []
            if z300 is not None:
                parts.append(f"z={z300:+.1f} over the last 300 games (n={n})")
            if zs is not None:
                parts.append(f"z={zs:+.1f} over the season (n={sr['n']})")
            sl = f"calibration slope {slope:.2f} over {w}; " if slope is not None else ""
            sev = "critical" if any(abs(x[1]) > CALIB_Z_CRIT for x in sig) else "warn"
            out.append(_a(f"calib:{model}", sev, "calibration",
                          f"{model} is {what} ({sl}{'; '.join(parts)}; alert when |z| > "
                          f"{CALIB_Z_WARN})",
                          {"model": model, "z_last300": z300, "z_season": zs, "slope": slope,
                           "intercept": row["details"].get("cal_intercept"), "n": n,
                           "ece": r["ece"]}))
        # biases
        if n >= MIN_N:
            hb = (r["home_pred"] - r["home_actual"]) if r["home_pred"] is not None else None
            hse = det.get("home_bias_se")
            if hb is not None and hse and abs(hb) > SE_K * hse:
                what = "Over" if hb > 0 else "Under"
                out.append(_a(f"bias:{model}:home", "warn", "bias",
                              f"{what}-predicting home teams for {model}: it gives them "
                              f"{r['home_pred']:.1%} to win, but they won {r['home_actual']:.1%} "
                              f"over {games} ({hb * 100:+.1f} points, {abs(hb) / hse:.1f} SE)",
                              {"model": model, "bias": hb, "se": hse, "n": n}))
            for field, nkey, se_key, noun, unit in (
                    ("margin_bias", "n_margin", "margin_bias_se", "home margins", "pts"),
                    ("total_bias", "n_total", "total_bias_se", "totals", "pts")):
                b, bse, bn = r[field], det.get(se_key), det.get(nkey) or 0
                if b is None or not bse or bn < MIN_N or abs(b) <= SE_K * bse:
                    continue
                what = "Over" if b > 0 else "Under"
                kind = field.split("_")[0]
                out.append(_a(f"bias:{model}:{kind}", "warn", "bias",
                              f"{what}-predicting {noun} by {abs(b):.1f} {unit} for {model} "
                              f"over {games} ({abs(b) / bse:.1f} SE)",
                              {"model": model, "bias": b, "se": bse, "n": bn}))
    return out


def _data_alerts(conn: sqlite3.Connection) -> list[dict]:
    out = []
    names = [r[0] for r in conn.execute("SELECT DISTINCT check_name FROM data_checks")]
    for name in names:
        rows = conn.execute("SELECT status, detail, run_at FROM data_checks WHERE check_name=? "
                            "AND status != 'skip' ORDER BY run_at DESC LIMIT 12", (name,)).fetchall()
        if not rows or rows[0]["status"] == "ok":
            continue
        streak = 0
        for r in rows:
            if r["status"] == "ok":
                break
            streak += 1
        latest = rows[0]
        if streak >= DATA_CRIT_RUNS:
            sev = "critical"
        elif streak >= DATA_WARN_RUNS or latest["status"] == "fail":
            sev = "warn"
        else:
            continue
        label = "failed" if latest["status"] == "fail" else "warned"
        out.append(_a(f"data:{name}", sev, "data",
                      f"Data check {name} has {label} {streak} run{'s' if streak != 1 else ''} in "
                      f"a row: {latest['detail']}",
                      {"check": name, "status": latest["status"], "streak": streak,
                       "run_at": latest["run_at"]}))
    return out


def _drift_alerts(conn: sqlite3.Connection, as_of: date) -> list[dict]:
    out = []
    for r in drift_mod.latest(conn, as_of):
        age = (as_of - date.fromisoformat(r["as_of"])).days
        if age > DRIFT_MAX_AGE_DAYS or r["psi"] is None or r["psi"] <= PSI_WARN:
            continue
        sev = "critical" if r["psi"] > PSI_CRIT else "warn"
        out.append(_a(f"drift:{r['feature']}", sev, "drift",
                      f"Live input {r['feature']} has shifted from its reference "
                      f"({r['ref_label']}): PSI {r['psi']:.2f} over {r['cur_n']} recent games "
                      f"(warn above {PSI_WARN}, critical above {PSI_CRIT})",
                      {"feature": r["feature"], "psi": r["psi"], "ref": r["ref_label"],
                       "n": r["cur_n"]}))
    return out


def desired_alerts(conn: sqlite3.Connection, as_of) -> list[dict]:
    as_of = _as_date(as_of)
    return _metric_alerts(conn, as_of) + _data_alerts(conn) + _drift_alerts(conn, as_of)


# ------------------------------------------------------------------ reconcile

def evaluate(conn: sqlite3.Connection, as_of) -> list[dict]:
    """Upsert the alerts that fire, resolve the ones that cleared. Returns the alert rows that
    were opened, updated, escalated or resolved this call plus every other open alert, each with
    an `event` of opened | updated | resolved | open."""
    now = _now()
    want = {a["key"]: a for a in desired_alerts(conn, as_of)}
    existing = {r["key"]: dict(r) for r in conn.execute(
        "SELECT * FROM alerts WHERE status='open' ORDER BY id")}
    out: list[dict] = []
    with conn:
        for key, a in want.items():
            details = json.dumps(a["details"], allow_nan=False, default=str)
            ex = existing.get(key)
            if ex is None:
                cur = conn.execute(
                    "INSERT INTO alerts (key, created_at, updated_at, severity, kind, message, "
                    "details_json, status) VALUES (?,?,?,?,?,?,?, 'open')",
                    (key, now, now, a["severity"], a["kind"], a["message"], details))
                row = {**a, "id": cur.lastrowid, "created_at": now, "updated_at": now,
                       "status": "open", "event": "opened"}
                log.warning("ALERT %s [%s]: %s", key, a["severity"], a["message"])
            else:
                changed = (ex["severity"], ex["message"]) != (a["severity"], a["message"])
                esc = _SEV[a["severity"]] > _SEV.get(ex["severity"], 0)
                conn.execute("UPDATE alerts SET updated_at=?, severity=?, kind=?, message=?, "
                             "details_json=? WHERE id=?",
                             (now, a["severity"], a["kind"], a["message"], details, ex["id"]))
                row = {**a, "id": ex["id"], "created_at": ex["created_at"], "updated_at": now,
                       "status": "open", "event": "updated" if changed else "open"}
                if esc:
                    log.warning("ALERT escalated %s [%s]: %s", key, a["severity"], a["message"])
            out.append(row)
        for key, ex in existing.items():
            if key in want:
                continue
            conn.execute("UPDATE alerts SET status='resolved', resolved_at=?, updated_at=? "
                         "WHERE id=?", (now, now, ex["id"]))
            log.info("alert resolved %s", key)
            out.append({**{k: ex[k] for k in ("id", "key", "severity", "kind", "message",
                                              "created_at")},
                        "details": json.loads(ex["details_json"] or "{}"), "updated_at": now,
                        "status": "resolved", "resolved_at": now, "event": "resolved"})
    return out


def list_alerts(conn: sqlite3.Connection, include_resolved: bool = False, limit: int = 50
                ) -> list[dict]:
    sql = "SELECT * FROM alerts"
    if not include_resolved:
        sql += " WHERE status='open'"
    sql += " ORDER BY status='open' DESC, CASE severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 " \
           "ELSE 2 END, updated_at DESC LIMIT ?"
    return [dict(r) for r in conn.execute(sql, (limit,))]
