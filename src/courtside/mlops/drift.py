"""Feature drift: population stability index (PSI) of live inputs vs a reference -> `drift_snapshots`.

Current window: champion (`PRODUCTION_MODEL`, role champion/NULL) predictions for games with
`nba_day` in [as_of - 30 d, as_of), the latest pre-tip row per game, features parsed from
`predictions.inputs_json` (plus the `home_win_prob` / `total` / `home_margin` columns).

Reference (documented, in order of preference):
  1. the previous full season's predictions of the same model (`ref_label` = "<season> preds"),
     when it has at least MIN_N games;
  2. otherwise the same season's predictions made BEFORE the current window (`ref_label` =
     "<season> before window"), so a season that has no predecessor in the DB still gets a
     meaningful comparison (this is what the 2025-26 backfill replay uses).
The reference bins are the reference's own deciles (quantile bins; ties collapsed), so PSI is
the usual sum((cur% - ref%) * ln(cur% / ref%)) with a 1e-4 floor on the proportions.

A feature is skipped (no row written) when either side has fewer than MIN_N values.

Features (what is stored in `inputs_json` today):
  elo_diff              inputs.elo.home - away
  strength_diff         sum of `explain` rows factor=player_strength (home-margin points)
  absences              sum of `explain` rows factor=absences (home-margin points)
  rest_diff             inputs.schedule.rest_diff
  b2b_total             inputs.schedule home.b2b + away.b2b
  market_spread         inputs.market.home_spread
  market_total          inputs.market.total
  pred_home_prob        predictions.home_win_prob
  pred_total            predictions.total
  lineup_strength_diff  inputs.availability home.lineup_strength - away.lineup_strength
                        (live rows only: backfilled rows have basis=regulars_played, no value)
`strength_diff` and `absences` come from `explain`, which is truncated to the top few factors per
game, so a missing factor counts as 0 (the same on both sides, so the PSI stays comparable).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np

from ..trends.common import prior_season, season_of

log = logging.getLogger("courtside.mlops.drift")

WINDOW_DAYS = 30
MIN_N = 100
BINS = 10
PSI_EPS = 1e-4
FEATURES = ("elo_diff", "strength_diff", "absences", "rest_diff", "b2b_total", "market_spread",
            "market_total", "pred_home_prob", "pred_total", "lineup_strength_diff")


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _explain_sum(inp: dict, factor: str) -> float | None:
    ex = inp.get("explain")
    if ex is None:
        return None
    return sum(_num(e.get("home_margin_pts")) or 0.0 for e in ex if e.get("factor") == factor)


def extract(row: sqlite3.Row | dict) -> dict[str, float | None]:
    """Feature dict for one stored prediction row (needs `inputs_json`)."""
    try:
        inp = json.loads(row["inputs_json"]) if row["inputs_json"] else {}
    except ValueError:
        inp = {}
    sched = inp.get("schedule") or {}
    elo = inp.get("elo") or {}
    mkt = inp.get("market") or {}
    avail = inp.get("availability") or {}
    f: dict[str, float | None] = dict.fromkeys(FEATURES)
    if _num(elo.get("home")) is not None and _num(elo.get("away")) is not None:
        f["elo_diff"] = _num(elo["home"]) - _num(elo["away"])
    f["strength_diff"] = _explain_sum(inp, "player_strength")
    f["absences"] = _explain_sum(inp, "absences")
    f["rest_diff"] = _num(sched.get("rest_diff"))
    hb, ab = _num((sched.get("home") or {}).get("b2b")), _num((sched.get("away") or {}).get("b2b"))
    if hb is not None and ab is not None:
        f["b2b_total"] = hb + ab
    f["market_spread"] = _num(mkt.get("home_spread")) if mkt else _num(row["market_home_spread"])
    f["market_total"] = _num(mkt.get("total")) if mkt else _num(row["market_total"])
    f["pred_home_prob"] = _num(row["home_win_prob"])
    f["pred_total"] = _num(row["total"])
    hl = _num((avail.get("home") or {}).get("lineup_strength"))
    al = _num((avail.get("away") or {}).get("lineup_strength"))
    if hl is not None and al is not None:
        f["lineup_strength_diff"] = hl - al
    return f


def psi(ref, cur, bins: int = BINS) -> float | None:
    """PSI of `cur` against `ref` using quantile bins of `ref` (ties collapsed). None if either
    side is empty."""
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    if len(edges) == 0:                 # constant reference: split at its value
        edges = np.unique(ref)[:1]
    r = np.bincount(np.digitize(ref, edges, right=True), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.digitize(cur, edges, right=True), minlength=len(edges) + 1) / len(cur)
    r, c = np.maximum(r, PSI_EPS), np.maximum(c, PSI_EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def _load(conn: sqlite3.Connection, model: str, start: str, end_excl: str
          ) -> tuple[list[str], list[dict]]:
    """(days, features) of the latest pre-tip champion/legacy prediction per game."""
    rows = conn.execute(
        "SELECT p.game_id, p.created_at, COALESCE(p.nba_day, g.nba_day) AS day, "
        "p.home_win_prob, p.total, p.market_home_spread, p.market_total, p.inputs_json "
        "FROM predictions p LEFT JOIN games g ON g.game_id=p.game_id "
        "WHERE p.model=? AND COALESCE(p.nba_day, g.nba_day)>=? AND COALESCE(p.nba_day, g.nba_day)<? "
        "AND (p.role IS NULL OR p.role='champion') "
        "AND (g.tipoff_utc IS NULL OR p.created_at<=g.tipoff_utc) ORDER BY p.created_at",
        (model, start, end_excl)).fetchall()
    last = {r["game_id"]: r for r in rows}
    days, feats = [], []
    for r in last.values():
        days.append(r["day"])
        feats.append(extract(r))
    return days, feats


def _column(feats: list[dict], name: str) -> np.ndarray:
    return np.array([f[name] for f in feats if f[name] is not None], float)


def compute_drift(conn: sqlite3.Connection, as_of, *, model: str | None = None,
                  min_n: int = MIN_N, window_days: int = WINDOW_DAYS) -> list[dict]:
    """PSI per feature for `as_of`; stores and returns the rows
    {as_of, feature, psi, ref_label, cur_n}. Features with < min_n values on either side are
    skipped (not written)."""
    if model is None:
        from .. import predict
        model = getattr(predict, "PRODUCTION_MODEL", "gbm_market")
    as_of = _as_date(as_of)
    win_start = (as_of - timedelta(days=window_days)).isoformat()
    _, cur = _load(conn, model, win_start, as_of.isoformat())
    out: list[dict] = []
    ref, ref_label = [], None
    if len(cur) >= min_n:
        season = season_of(as_of - timedelta(days=1))
        prev = prior_season(season)
        _, ref = _load(conn, model, f"{prev[:4]}-07-01", f"{season[:4]}-07-01")
        ref_label = f"{prev} preds"
        if len(ref) < min_n:
            _, ref = _load(conn, model, f"{season[:4]}-07-01", win_start)
            ref_label = f"{season} before window"
    for feat in FEATURES:
        c, r = _column(cur, feat), _column(ref, feat)
        if len(c) < min_n or len(r) < min_n:
            continue
        v = psi(r, c)
        if v is None:
            continue
        out.append({"as_of": as_of.isoformat(), "feature": feat, "psi": round(v, 4),
                    "ref_label": ref_label, "cur_n": int(len(c))})
    with conn:
        conn.execute("DELETE FROM drift_snapshots WHERE as_of=?", (as_of.isoformat(),))
        conn.executemany("INSERT INTO drift_snapshots (as_of, feature, psi, ref_label, cur_n) "
                         "VALUES (:as_of, :feature, :psi, :ref_label, :cur_n)", out)
    return out


def latest(conn: sqlite3.Connection, as_of) -> list[dict]:
    """Stored drift rows of the newest as_of <= `as_of`."""
    r = conn.execute("SELECT MAX(as_of) FROM drift_snapshots WHERE as_of<=?",
                     (_as_date(as_of).isoformat(),)).fetchone()
    if not r or r[0] is None:
        return []
    return [dict(x) for x in conn.execute(
        "SELECT * FROM drift_snapshots WHERE as_of=? ORDER BY psi DESC", (r[0],))]
