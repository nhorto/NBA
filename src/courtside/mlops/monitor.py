"""Model monitoring: resolved-prediction metrics per model and window -> `metric_snapshots`.

What is scored
--------------
* Covered models: every `predictions.model` whose rows have `role` champion or NULL (legacy),
  labelled by model name; shadow rows (`role = 'shadow'`) are labelled `<model>@<version>` so
  they never mix with the champion of the same name.
* The market is scored as the pseudo-model `market`: the de-vigged moneyline probability of the
  latest pre-tip `odds_snapshots` batch, else the consensus closing line (`predict.market_line`),
  else the line stored on the prediction row. Its margin is `-home_spread`, its total the line.
* Per (game, label) only the LATEST PRE-TIP prediction is scored (`created_at <= tipoff_utc`;
  any row when the tip-off is unknown), and only for games that are final with
  `nba_day < as_of`. Results come from `games` (final) else `team_games`.

Windows (all end at the last scored game before `as_of`): `7d` / `30d` (nba_day within the last
7 / 30 days), `last100` / `last300` (most recent N games of that model) and `season` (since
July 1 of the season containing `as_of`).

`replay()` loads once and slices per day, so backfilling a season is one query plus numpy.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .. import config, predict, teams
from ..model import metrics
from ..trends.common import season_of

log = logging.getLogger("courtside.mlops.monitor")

WINDOWS = ("7d", "30d", "last100", "last300", "season")
MARKET = "market"
LOAD_DAYS = 540            # history loaded before the first as_of (covers last300 across a summer)
ECE_BINS = 10

_COLS = ("model", "model_version", "window", "n", "log_loss", "brier", "accuracy", "ece",
         "home_pred", "home_actual", "margin_bias", "margin_mae", "total_bias", "total_mae",
         "mkt_n", "mkt_log_loss", "mkt_brier", "ll_diff_vs_mkt", "ll_diff_se", "details_json")


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# ------------------------------------------------------------------ loading

@dataclass
class Series:
    """Resolved predictions of one label, sorted by (nba_day, game_id)."""
    label: str
    model: str
    days: np.ndarray                     # '<U10' nba_day
    game_ids: np.ndarray
    p: np.ndarray                        # P(home win)
    y: np.ndarray                        # 1 = home won
    margin_pred: np.ndarray              # nan when absent
    margin: np.ndarray                   # actual home margin
    total_pred: np.ndarray
    total: np.ndarray                    # actual combined points
    mkt_p: np.ndarray = field(default=None)   # market P(home win) on the same game, nan if none
    versions: np.ndarray = field(default=None)  # model_version per row ('' when absent)
    shadow: bool = False

    def __len__(self) -> int:
        return len(self.days)


def _franchise(code: str | None, season: str | None) -> str | None:
    if not code:
        return code
    try:
        return teams.franchise(code, season)
    except KeyError:
        return code


def _results(conn: sqlite3.Connection, start: str, end_excl: str) -> dict[str, tuple[int, int]]:
    """game_id -> (home_pts, away_pts) for games that have a prediction in [start, end_excl)."""
    res: dict[str, tuple[int, int]] = {}
    for g in conn.execute(
            "SELECT game_id, home_score, away_score FROM games WHERE status='final' "
            "AND home_score IS NOT NULL AND away_score IS NOT NULL AND game_id IN "
            "(SELECT game_id FROM predictions WHERE nba_day>=? AND nba_day<?)", (start, end_excl)):
        res[g["game_id"]] = (g["home_score"], g["away_score"])
    pts: dict[str, dict[str, int]] = {}
    for t in conn.execute(
            "SELECT game_id, team, season, pts, min FROM team_games WHERE game_id IN "
            "(SELECT game_id FROM predictions WHERE nba_day>=? AND nba_day<?)", (start, end_excl)):
        if t["min"] and t["pts"] is not None and t["game_id"] not in res:
            pts.setdefault(t["game_id"], {})[_franchise(t["team"], t["season"])] = t["pts"]
    for p in conn.execute("SELECT DISTINCT game_id, home, away FROM predictions "
                          "WHERE nba_day>=? AND nba_day<?", (start, end_excl)):
        gid = p["game_id"]
        if gid not in res and p["home"] in pts.get(gid, {}) and p["away"] in pts.get(gid, {}):
            res[gid] = (pts[gid][p["home"]], pts[gid][p["away"]])
    return res


def _label(model: str, role: str | None, version: str | None) -> str:
    return f"{model}@{version or 'shadow'}" if role == "shadow" else model


def load_series(conn: sqlite3.Connection, start: str, end_excl: str) -> dict[str, Series]:
    """All covered series (models + the market) for games with nba_day in [start, end_excl)."""
    rows = conn.execute(
        "SELECT p.game_id, p.model, p.created_at, p.role, p.model_version, "
        "COALESCE(p.nba_day, g.nba_day) AS day, p.home_win_prob, p.home_margin, p.total, "
        "p.market_home_spread, p.market_total, p.market_home_prob, g.tipoff_utc "
        "FROM predictions p LEFT JOIN games g ON g.game_id = p.game_id "
        "WHERE COALESCE(p.nba_day, g.nba_day) >= ? AND COALESCE(p.nba_day, g.nba_day) < ? "
        "AND (p.role IS NULL OR p.role IN ('champion', 'shadow')) "
        "AND p.home_win_prob IS NOT NULL "
        "AND (g.tipoff_utc IS NULL OR p.created_at <= g.tipoff_utc) "
        "ORDER BY p.created_at", (start, end_excl)).fetchall()
    if not rows:
        return {}
    results = _results(conn, start, end_excl)
    latest: dict[tuple[str, str], sqlite3.Row] = {}
    for r in rows:                      # ascending created_at: the last write wins
        if r["game_id"] in results:
            latest[(r["game_id"], _label(r["model"], r["role"], r["model_version"]))] = r
    # market per game: line at/before tip-off, else closing line, else the line on the row
    market: dict[str, dict] = {}
    for (gid, _), r in sorted(latest.items()):
        if gid in market:
            continue
        ml = predict.market_line(conn, gid, r["tipoff_utc"])
        p = ml["home_prob"] if ml else r["market_home_prob"]
        spread = ml["home_spread"] if ml and ml.get("home_spread") is not None \
            else r["market_home_spread"]
        total = ml["total"] if ml and ml.get("total") is not None else r["market_total"]
        market[gid] = {"p": p, "spread": spread, "total": total, "day": r["day"]}

    by_label: dict[str, list] = {}
    for (gid, label), r in latest.items():
        by_label.setdefault(label, []).append(r)

    def build(label: str, model: str, items: list[dict], shadow: bool) -> Series:
        items.sort(key=lambda d: (d["day"], d["game_id"]))
        nan = np.nan
        return Series(
            label=label, model=model, shadow=shadow,
            days=np.array([d["day"] for d in items], dtype="<U10"),
            game_ids=np.array([d["game_id"] for d in items], dtype=object),
            p=np.array([d["p"] for d in items], float),
            y=np.array([d["y"] for d in items], float),
            margin_pred=np.array([nan if d["mp"] is None else d["mp"] for d in items], float),
            margin=np.array([d["margin"] for d in items], float),
            total_pred=np.array([nan if d["tp"] is None else d["tp"] for d in items], float),
            total=np.array([d["total"] for d in items], float),
            mkt_p=np.array([nan if d["mkt"] is None else d["mkt"] for d in items], float),
            versions=np.array([d["version"] or "" for d in items], dtype=object))

    out: dict[str, Series] = {}
    for label, rs in by_label.items():
        items = []
        for r in rs:
            hs, as_ = results[r["game_id"]]
            if hs == as_:
                continue
            m = market.get(r["game_id"], {})
            items.append({"day": r["day"], "game_id": r["game_id"], "p": r["home_win_prob"],
                          "y": int(hs > as_), "mp": r["home_margin"], "margin": hs - as_,
                          "tp": r["total"], "total": hs + as_, "mkt": m.get("p"),
                          "version": r["model_version"]})
        if items:
            out[label] = build(label, rs[0]["model"], items, rs[0]["role"] == "shadow")
    # the market pseudo-model: every game any covered series scored
    seen: dict[str, dict] = {}
    for label, rs in by_label.items():
        for r in rs:
            gid = r["game_id"]
            if gid in seen or results[gid][0] == results[gid][1]:
                continue
            m = market[gid]
            if m["p"] is None:
                continue
            hs, as_ = results[gid]
            seen[gid] = {"day": r["day"], "game_id": gid, "p": m["p"], "y": int(hs > as_),
                         "mp": None if m["spread"] is None else -m["spread"],
                         "margin": hs - as_, "tp": m["total"], "total": hs + as_,
                         "mkt": m["p"], "version": None}
    if seen:
        out[MARKET] = build(MARKET, MARKET, list(seen.values()), False)
    return out


# ------------------------------------------------------------------ metric math

def _se(x: np.ndarray) -> float | None:
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) >= 2 else None


def _round(v, nd=5):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


def _loglik(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, metrics.EPS, 1 - metrics.EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def window_metrics(s: Series, lo: int, hi: int) -> dict | None:
    """Metrics for rows [lo, hi) of a series, or None when empty."""
    n = hi - lo
    if n <= 0:
        return None
    p, y = s.p[lo:hi], s.y[lo:hi]
    ll = _loglik(y, p)
    cal = metrics.calibration_table(y, p, ECE_BINS)
    mp, mg = s.margin_pred[lo:hi], s.margin[lo:hi]
    tp, tt = s.total_pred[lo:hi], s.total[lo:hi]
    mdiff, tdiff = mp - mg, tp - tt
    mk = s.mkt_p[lo:hi]
    have = ~np.isnan(mk)
    out = {
        "n": n, "log_loss": float(ll.mean()), "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))), "ece": metrics.ece(y, p, ECE_BINS),
        "home_pred": float(p.mean()), "home_actual": float(y.mean()),
        "margin_bias": float(np.nanmean(mdiff)) if (~np.isnan(mdiff)).any() else None,
        "margin_mae": float(np.nanmean(np.abs(mdiff))) if (~np.isnan(mdiff)).any() else None,
        "total_bias": float(np.nanmean(tdiff)) if (~np.isnan(tdiff)).any() else None,
        "total_mae": float(np.nanmean(np.abs(tdiff))) if (~np.isnan(tdiff)).any() else None,
        "mkt_n": int(have.sum()), "mkt_log_loss": None, "mkt_brier": None,
        "ll_diff_vs_mkt": None, "ll_diff_se": None,
    }
    if have.any():
        ym, pm, mm = y[have], p[have], mk[have]
        out["mkt_log_loss"] = float(_loglik(ym, mm).mean())
        out["mkt_brier"] = float(np.mean((mm - ym) ** 2))
        if s.label != MARKET:
            d = _loglik(ym, pm) - _loglik(ym, mm)
            out["ll_diff_vs_mkt"] = float(d.mean())
            out["ll_diff_se"] = _se(d)
    out["details"] = {
        "calibration": [{"lo": round(c["lo"], 2), "hi": round(c["hi"], 2), "n": c["n"],
                         "pred": _round(c["mean_p"], 4), "actual": _round(c["frac_pos"], 4)}
                        for c in cal],
        "ll_se": _round(_se(ll)),
        "home_bias_se": _round(_se(p - y)),
        "margin_bias_se": _round(_se(mdiff)), "n_margin": int((~np.isnan(mdiff)).sum()),
        "total_bias_se": _round(_se(tdiff)), "n_total": int((~np.isnan(tdiff)).sum()),
        "first_day": str(s.days[lo]), "last_day": str(s.days[hi - 1]),
    }
    versions = sorted({v for v in s.versions[lo:hi] if v})
    if versions:
        out["details"]["versions"] = versions
    return out


def window_bounds(s: Series, window: str, as_of: date) -> tuple[int, int]:
    """Row range [lo, hi) of `window` for games with nba_day < as_of."""
    hi = int(np.searchsorted(s.days, as_of.isoformat(), side="left"))
    if window == "7d":
        lo = int(np.searchsorted(s.days, (as_of - timedelta(days=7)).isoformat(), side="left"))
    elif window == "30d":
        lo = int(np.searchsorted(s.days, (as_of - timedelta(days=30)).isoformat(), side="left"))
    elif window == "last100":
        lo = max(0, hi - 100)
    elif window == "last300":
        lo = max(0, hi - 300)
    elif window == "season":
        lo = int(np.searchsorted(s.days, f"{season_of(as_of)[:4]}-07-01", side="left"))
    else:
        raise ValueError(f"unknown window {window!r}")
    return min(lo, hi), hi


def snapshot_rows(series: dict[str, Series], as_of: date) -> list[dict]:
    """metric_snapshots rows (dicts) for `as_of` from already-loaded series."""
    rows = []
    for label in sorted(series, key=lambda k: (k == MARKET, k)):
        s = series[label]
        for w in WINDOWS:
            lo, hi = window_bounds(s, w, as_of)
            m = window_metrics(s, lo, hi)
            if m is None:
                continue
            det = m.pop("details")
            vers = det.get("versions")
            version = None
            if s.shadow and "@" in label:
                version = label.split("@", 1)[1]
            elif vers:
                version = vers[-1]
            rows.append({"as_of": as_of.isoformat(), "model": label, "model_version": version,
                         "window": w, **{k: _round(v) if isinstance(v, float) else v
                                         for k, v in m.items()},
                         "details_json": json.dumps(det, allow_nan=False)})
    return rows


def _write(conn: sqlite3.Connection, rows: list[dict], as_ofs: list[str]) -> None:
    cols = ("as_of", *_COLS)
    with conn:
        conn.executemany("DELETE FROM metric_snapshots WHERE as_of=?", [(a,) for a in as_ofs])
        conn.executemany(
            f"INSERT OR REPLACE INTO metric_snapshots ({', '.join(cols)}) "
            f"VALUES ({', '.join(':' + c for c in cols)})", rows)


# ------------------------------------------------------------------ public API

def compute_metrics(conn: sqlite3.Connection, as_of) -> list[dict]:
    """Compute and store metric_snapshots for `as_of` (games with nba_day < as_of)."""
    as_of = _as_date(as_of)
    series = load_series(conn, (as_of - timedelta(days=LOAD_DAYS)).isoformat(),
                         as_of.isoformat())
    rows = snapshot_rows(series, as_of)
    _write(conn, rows, [as_of.isoformat()])
    return rows


def replay(conn: sqlite3.Connection, start, end) -> dict:
    """Snapshots for every day in [start, end] (inclusive). Loads once, slices per day."""
    start, end = _as_date(start), _as_date(end)
    series = load_series(conn, (start - timedelta(days=LOAD_DAYS)).isoformat(),
                         end.isoformat())
    rows, days = [], []
    d = start
    while d <= end:
        rows += snapshot_rows(series, d)
        days.append(d.isoformat())
        d += timedelta(days=1)
    _write(conn, rows, days)
    return {"start": start.isoformat(), "end": end.isoformat(), "days": len(days),
            "rows": len(rows), "models": sorted(series), "games": {k: len(v) for k, v in
                                                                   series.items()}}


def latest_snapshot_day(conn: sqlite3.Connection, as_of) -> str | None:
    r = conn.execute("SELECT MAX(as_of) FROM metric_snapshots WHERE as_of<=?",
                     (_as_date(as_of).isoformat(),)).fetchone()
    return r[0] if r else None


def snapshots(conn: sqlite3.Connection, as_of) -> list[dict]:
    """The stored snapshot rows of the latest as_of <= `as_of`, details_json parsed."""
    day = latest_snapshot_day(conn, as_of)
    if day is None:
        return []
    out = []
    for r in conn.execute("SELECT * FROM metric_snapshots WHERE as_of=?", (day,)):
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json")) if r["details_json"] else {}
        out.append(d)
    return out


# ------------------------------------------------------------------ post-job hook

def run_after_job(conn: sqlite3.Connection, job: str, day, summary: dict | None = None) -> dict:
    """What collect.py runs at the end of a job. Every stage is guarded; never raises.

    pregame:  checks + alerts.
    postgame: checks + metrics(day+1) + drift + alerts.
    Returns {"checks": {...counts}, "alerts": {...}, "line": one-line summary}."""
    from . import alerts, checks, drift
    day = _as_date(day)
    out: dict = {}
    as_of = day + timedelta(days=1) if job == "postgame" else day

    def guarded(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - monitoring must never break collection
            log.exception("monitoring stage %s failed", name)
            out.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            return None

    res = guarded("checks", lambda: checks.run_checks(conn, job, day, summary))
    if res is not None:
        counts = {s: sum(1 for r in res if r["status"] == s) for s in ("ok", "warn", "fail", "skip")}
        out["checks"] = counts
    if job == "postgame":
        m = guarded("metrics", lambda: compute_metrics(conn, as_of))
        if m is not None:
            out["metric_rows"] = len(m)
        dr = guarded("drift", lambda: drift.compute_drift(conn, as_of))
        if dr is not None:
            out["drift_features"] = len(dr)
            out["drift_max_psi"] = max((r["psi"] for r in dr), default=None)
    al = guarded("alerts", lambda: alerts.evaluate(conn, as_of))
    if al is not None:
        open_ = [a for a in al if a["status"] == "open"]
        out["alerts"] = {"open": len(open_),
                         "warn": sum(a["severity"] == "warn" for a in open_),
                         "critical": sum(a["severity"] == "critical" for a in open_),
                         "new": sum(a.get("event") == "opened" for a in al),
                         "resolved": sum(a.get("event") == "resolved" for a in al)}
    c = out.get("checks", {})
    a = out.get("alerts", {})
    out["line"] = (f"health[{job} {day}]: checks ok={c.get('ok', '?')} warn={c.get('warn', '?')} "
                   f"fail={c.get('fail', '?')} skip={c.get('skip', '?')}; alerts open="
                   f"{a.get('open', '?')} (critical {a.get('critical', '?')}, "
                   f"new {a.get('new', '?')}, resolved {a.get('resolved', '?')})"
                   + (f"; errors in {sorted(out['errors'])}" if out.get("errors") else ""))
    log.info(out["line"])
    return out
