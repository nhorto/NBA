"""Model Health page: monitoring tables -> a template-ready view model (and the JSON summary).

Reads the MLOps tables (`model_versions`, `registry_events`, `metric_snapshots`, `data_checks`,
`drift_snapshots`, `alerts`; docs/MLOPS.md) through a read-only connection. Every query is
tolerant of a missing table or an empty one, so the page renders on a fresh database. All display
strings are built here, so templates never print None or NaN.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime, timezone

from . import charts, fmt
from . import health_charts as hc

MARKET = "market"
SEV_ORDER = {"critical": 0, "warn": 1, "info": 2}
MODEL_ORDER = ("gbm_market", "gbm_opinion", "elo_sched")
CHART_MODELS = ("gbm_market", "gbm_opinion", MARKET)
MODEL_INFO = {
    "gbm_market": ("GBM, market-aware", "Gradient boosting that sees the betting line"),
    "gbm_opinion": ("GBM, independent", "Gradient boosting that ignores the line by design"),
    "elo_sched": ("Elo + schedule", "Ratings and rest only; a simple baseline"),
    MARKET: ("The market", "De-vigged betting line; the bar to clear"),
}
STALE_SNAPSHOT_DAYS = 7
MODEL_RECENT_DAYS = 14        # a model not scored in this long before the newest snapshot is hidden
MIN_PLOT_N = 30               # rolling windows with fewer games are too noisy to draw
PSI_WARN, PSI_CRIT = 0.25, 0.5
CAL_MIN_N = 100

PSI_LABELS = {
    "elo_diff": "Elo gap", "strength_diff": "Team strength gap", "absences": "Absences (impact)",
    "rest_diff": "Rest-days gap", "b2b_total": "Back-to-backs", "market_spread": "Market spread",
    "market_total": "Market total", "pred_total": "Predicted total",
    "pred_home_prob": "Predicted home win %",
}
ACTIONS = {
    "register": ("Registered", "plain"), "shadow": ("Entered shadow", "info"),
    "promote": ("Promoted to champion", "pos"), "rollback": ("Rolled back", "warn"),
    "retire": ("Retired", "plain"), "reject": ("Rejected", "neg"),
}
STATUS_PILL = {"ok": ("OK", "healthy"), "warn": ("Warn", "questionable"),
               "fail": ("Fail", "out"), "skip": ("Skipped", "unknown")}
SEV_PILL = {"critical": ("Critical", "out"), "warn": ("Warn", "questionable"),
            "info": ("Info", "unknown")}


# ------------------------------------------------------------------ reading

def _q(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.Error:
        return []


def _json(s) -> dict:
    try:
        v = json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _f(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def critical_alerts(conn: sqlite3.Connection) -> list[dict]:
    """Open critical alerts (the only thing that triggers the site-wide banner)."""
    return _q(conn, "SELECT key, severity, kind, message, created_at FROM alerts "
                    "WHERE status='open' AND severity='critical' ORDER BY updated_at DESC")


def banner(conn: sqlite3.Connection | None) -> dict | None:
    if conn is None:
        return None
    rows = critical_alerts(conn)
    if not rows:
        return None
    n = len(rows)
    return {"count": n,
            "text": f"Model health: {n} critical alert{'s' if n != 1 else ''}.",
            "detail": _short(rows[0]["message"], 90)}


def _vshort(version_id: str | None) -> str:
    """'gbm_market@v1-7edd944e' -> 'v1-7edd944e' (the model name is shown beside it)."""
    return (version_id or "").split("@")[-1]


def _short(msg: str, n: int = 110) -> str:
    msg = (msg or "").strip()
    cut = msg.split(". ")[0] if ". " in msg else msg
    if len(cut) > n:
        cut = cut[:n].rsplit(" ", 1)[0] + "…"
    return cut


# ------------------------------------------------------------------ small text helpers

def _label(model: str) -> tuple[str, str, str | None]:
    """(display name, one-line description, shadow version) for a snapshot label."""
    base, _, ver = model.partition("@")
    name, desc = MODEL_INFO.get(base, (base.replace("_", " ").capitalize(), ""))
    if ver:
        return f"{name} (shadow)", desc or "Logged for evaluation only", ver
    return name, desc, None


def _trim(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return s[:-2] if nd and s.endswith(".0") else s


def vs_market(diff, se, n) -> dict:
    """Plain-English paired log-loss comparison with the market (negative diff = we are better)."""
    d, s = _f(diff), _f(se)
    if d is None:
        return {"text": "No market line on these games", "tone": "none", "z": None}
    if n is not None and n < 30:
        return {"text": f"Too early to compare ({int(n)} games with a line)", "tone": "none",
                "z": None}
    side = "better" if d < 0 else "worse"
    if s is None or s <= 0:
        return {"text": f"{abs(d):.3f} {side} than the market", "tone": "level", "z": None}
    z = abs(d) / s
    if z < 1:
        return {"text": f"Level with the market: {abs(d):.3f} {side}, within noise "
                        f"(≈{_trim(z)} SE)", "tone": "level", "z": z}
    return {"text": f"{abs(d):.3f} {side} than the market (≈{_trim(z)} SE)",
            "tone": "ahead" if d < 0 else "behind", "z": z}


def calib_verdict(z, n) -> dict:
    z = _f(z)
    if z is None or n is None or n < CAL_MIN_N:
        return {"label": "Too early to judge", "tone": "none",
                "note": f"Needs about {CAL_MIN_N} scored games; {int(n or 0)} so far."}
    az = abs(z)
    over = z > 0
    word = "overconfident" if over else "underconfident"
    note = ("Its favorites win less often than it says." if over else
            "Its favorites win more often than it says.")
    if az < 1.5:
        return {"label": "Well calibrated", "tone": "ok",
                "note": "When it says 70%, it happens about 70% of the time."}
    if az < 2.5:
        return {"label": f"Slightly {word}", "tone": "mild", "note": note}
    if az < 3.5:
        return {"label": word.capitalize(), "tone": "warn", "note": note}
    return {"label": f"Badly {word}", "tone": "bad", "note": note}


def _bias_phrase(z: float | None, bias: float | None, noun: str, title: str) -> tuple[str, str]:
    """Plain-English verdict on a bias (predicted minus actual) and its tone."""
    if bias is None:
        return "Not enough data", "none"
    if z is None:
        return "No error band yet", "none"
    az = abs(z)
    over = bias > 0
    if az < 1:
        return f"{title} on target", "ok"
    if az < 2:
        return f"Slight {'over' if over else 'under'}-prediction, within noise", "mild"
    verb = "Over-predicts" if over else "Under-predicts"
    return f"{verb} {noun} by {abs(bias):.1f} pts", "warn" if az < 3.5 else "bad"


def _nice_scale(x: float, floor: float) -> float:
    for s in (2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50):
        if s >= max(x, floor):
            return float(s)
    return float(math.ceil(max(x, floor)))


# ------------------------------------------------------------------ the view model

def build(conn: sqlite3.Connection | None, today: date | None = None) -> dict:
    today = today or date.today()
    c = conn
    snaps = _latest_snapshots(c) if c else []
    as_of = max((r["as_of"] for r in snaps), default=None)
    versions = _q(c, "SELECT * FROM model_versions ORDER BY model_name, created_at DESC") if c else []
    open_alerts = _q(c, "SELECT * FROM alerts WHERE status='open'") if c else []
    open_alerts.sort(key=lambda a: (SEV_ORDER.get(a["severity"], 9), -_ts_key(a["updated_at"])))
    resolved = _q(c, "SELECT * FROM alerts WHERE status='resolved' "
                     "ORDER BY resolved_at DESC, id DESC LIMIT 12") if c else []
    checks = _checks(c) if c else {"groups": [], "counts": {}, "n": 0}
    drift = _drift(c, today) if c else None
    events = _events(c) if c else []

    by_label: dict[str, dict[str, dict]] = {}
    for r in snaps:
        by_label.setdefault(r["model"], {})[r["window"]] = r
    labels = sorted(by_label, key=_model_sort)

    champions = [v for v in versions if v["status"] == "champion"]
    shadows = [v for v in versions if v["status"] == "shadow"]
    roles = {v["model_name"]: "Champion" for v in champions}
    for v in shadows:
        roles.setdefault(f'{v["model_name"]}@{v["version_id"]}', "Shadow")
    ver_of = {v["model_name"]: v["version_id"] for v in champions}

    models = [_model_card(lab, by_label[lab], roles, ver_of) for lab in labels]
    stale = None
    if as_of:
        d = fmt.parse_day(as_of)
        age = (today - d).days if d else None
        stale = age if age is not None and age > STALE_SNAPSHOT_DAYS else None
        snapshot_age = age
    else:
        snapshot_age = None

    vm = {
        "as_of": as_of, "snapshot_age": snapshot_age, "stale_days": stale,
        "has_metrics": bool(snaps), "models": models,
        "champions": [_version(v, "champion") for v in champions],
        "shadows": [_version(v, "shadow") for v in shadows],
        "registry_empty": not versions,
        "alerts": _alerts_view(open_alerts, resolved),
        "checks": checks, "drift": drift, "history": _history_view(events),
        "calibration": _calibration(by_label, labels),
        "bias": _bias(by_label, labels),
    }
    vm.update(_trends(c, labels) if c and snaps else {"perf": None, "acc": None, "legend": [],
                                                        "bias_charts": [], "min_plot_n": MIN_PLOT_N})
    vm["status"] = _overall(vm)
    vm["games_scored"] = max((m["season"]["n"] or 0 for m in models if m["id"] != MARKET
                              and m["season"]), default=0)
    return vm


def _ts_key(s) -> float:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _model_sort(label: str):
    base = label.partition("@")[0]
    if label == MARKET:
        return (3, 0, label)
    if "@" in label:
        return (2, MODEL_ORDER.index(base) if base in MODEL_ORDER else 9, label)
    return (0 if base in MODEL_ORDER else 1, MODEL_ORDER.index(base) if base in MODEL_ORDER else 9,
            label)


def _latest_snapshots(c) -> list[dict]:
    rows = _q(c, "SELECT m.* FROM metric_snapshots m JOIN (SELECT model, window, MAX(as_of) a "
                 "FROM metric_snapshots GROUP BY model, window) x ON x.model=m.model AND "
                 "x.window=m.window AND x.a=m.as_of")
    if not rows:
        return []
    newest = max(r["as_of"] for r in rows)
    nd = fmt.parse_day(newest)
    keep = []
    for r in rows:
        d = fmt.parse_day(r["as_of"])
        if nd and d and (nd - d).days <= MODEL_RECENT_DAYS:
            keep.append(r)
    return keep


def _version(v: dict, status: str) -> dict:
    when = v.get("promoted_at") if status == "champion" else None
    when = when or v.get("created_at")
    label, _, _ = _label(v["model_name"])
    return {"model": v["model_name"], "label": label, "version_id": v["version_id"],
            "when": fmt.stamp(when) or fmt.medium_date(when), "when_raw": when,
            "when_word": "Promoted" if status == "champion" and v.get("promoted_at") else
                         ("Registered" if status == "champion" else "In shadow since"),
            "sha": (v.get("code_sha") or "")[:7], "dirty": "dirty" in (v.get("code_sha") or ""),
            "vshort": _vshort(v["version_id"]), "status": status}


# ---- scoreboard

def _stats(r: dict | None) -> dict | None:
    if not r or not r.get("n"):
        return None
    d = _json(r.get("details_json"))
    vm = vs_market(r.get("ll_diff_vs_mkt"), r.get("ll_diff_se"), r.get("mkt_n"))
    diff, se = _f(r.get("ll_diff_vs_mkt")), _f(r.get("ll_diff_se"))
    return {
        "n": int(r["n"]), "log_loss": _f(r.get("log_loss")), "brier": _f(r.get("brier")),
        "accuracy": _f(r.get("accuracy")), "ece": _f(r.get("ece")),
        "ll": fmt.num(r.get("log_loss"), 3), "br": fmt.num(r.get("brier"), 3),
        "acc": fmt.pct(r.get("accuracy"), 1),
        "vs": (f'{fmt.signed(diff, 3)} <small>±{fmt.num(se, 3)}</small>'
               if diff is not None and se is not None else
               (fmt.signed(diff, 3) if diff is not None else fmt.EM)),
        "ll_diff_vs_mkt": diff, "ll_diff_se": se, "vs_market": vm,
        "first_day": d.get("first_day"), "last_day": d.get("last_day"),
        "details": d, "row": r,
    }


def _model_card(label: str, wins: dict[str, dict], roles: dict, ver_of: dict) -> dict:
    name, desc, shadow_ver = _label(label)
    season, l100 = _stats(wins.get("season")), _stats(wins.get("last100"))
    ver = (shadow_ver or ver_of.get(label)
           or next((w.get("model_version") for w in wins.values() if w.get("model_version")), None))
    role = roles.get(label)
    if label == MARKET:
        role = "Benchmark"
    elif role is None and label.partition("@")[0] == "elo_sched":
        role = "Baseline"
    pick = season or l100
    return {"id": label, "name": name, "desc": desc, "role": role, "version": _vshort(ver) or None,
            "season": season, "last100": l100,
            "headline": pick["vs_market"] if pick and label != MARKET else None,
            "headline_window": "season" if season else "last 100"}


# ---- calibration

def _calibration(by_label: dict, labels: list[str]) -> list[dict]:
    out = []
    for lab in labels:
        r = by_label[lab].get("season")
        if not r:
            continue
        d = _json(r.get("details_json"))
        bins = [{"mean_p": b.get("pred"), "frac_home_won": b.get("actual"), "n": b.get("n"),
                 "lo": b.get("lo"), "hi": b.get("hi")}
                for b in (d.get("calibration") or [])
                if isinstance(b, dict) and b.get("n") and b.get("pred") is not None
                and b.get("actual") is not None]
        n = int(r["n"] or 0)
        z, slope = _f(d.get("spiegelhalter_z")), _f(d.get("cal_slope"))
        icpt, sse = _f(d.get("cal_intercept")), _f(d.get("cal_slope_se"))
        name, _, _ = _label(lab)
        verdict = calib_verdict(z, n)
        svg = charts.calibration_chart(bins) if n >= 20 else None
        out.append({
            "id": lab, "name": name, "n": n, "svg": svg, "bins": bins, "verdict": verdict,
            "z": z, "z_text": fmt.signed(z, 1), "slope": slope, "slope_text": fmt.num(slope, 2),
            "slope_se": fmt.num(sse, 2) if sse is not None else "", "intercept": icpt,
            "icpt_text": fmt.signed(icpt, 2), "ece": fmt.pct(r.get("ece"), 1),
            "market": lab == MARKET,
        })
    return out


# ---- over/under-prediction

_BIAS_METRICS = (
    # key, title, noun for the sentence, scale floor, multiplier to display points
    ("home", "Home win rate", "home wins", 10.0),
    ("margin", "Home margin", "the home margin", 4.0),
    ("total", "Game total", "totals", 4.0),
)


def _bias_value(r: dict, key: str) -> tuple[float | None, float | None]:
    """(bias, SE) in display points; bias = predicted minus actual (positive = over-predicts)."""
    d = _json(r.get("details_json"))
    if key == "home":
        hp, ha = _f(r.get("home_pred")), _f(r.get("home_actual"))
        se = _f(d.get("home_bias_se"))
        return (None if hp is None or ha is None else 100 * (hp - ha),
                None if se is None else 100 * se)
    if key == "margin":
        return _f(r.get("margin_bias")), _f(d.get("margin_bias_se"))
    return _f(r.get("total_bias")), _f(d.get("total_bias_se"))


def _bias(by_label: dict, labels: list[str]) -> dict:
    scales = {}
    for key, _t, _n, floor in _BIAS_METRICS:
        peak = 0.0
        for lab in labels:
            for w in ("season", "last100"):
                r = by_label[lab].get(w)
                if not r:
                    continue
                b, se = _bias_value(r, key)
                if b is not None:
                    peak = max(peak, abs(b) + 2 * (se or 0))
        scales[key] = _nice_scale(peak * 1.05, floor)
    cards = []
    for lab in labels:
        s, l = by_label[lab].get("season"), by_label[lab].get("last100")
        base = s or l
        if not base:
            continue
        name, _, _ = _label(lab)
        rows = []
        for key, title, noun, _floor in _BIAS_METRICS:
            row = _bias_row(key, title, noun, s, l, scales[key])
            if row:
                rows.append(row)
        if rows:
            cards.append({"id": lab, "name": name, "rows": rows,
                          "n": int((s or l)["n"] or 0), "market": lab == MARKET})
    return {"cards": cards, "scales": scales}


def _bias_row(key, title, noun, s, l, scale) -> dict | None:
    sb, sse = _bias_value(s, key) if s else (None, None)
    lb, lse = _bias_value(l, key) if l else (None, None)
    primary, pse, pwin, pn = (sb, sse, "season", s and s["n"]) if sb is not None else \
        (lb, lse, "last 100", l and l["n"])
    if primary is None:
        return None
    z = primary / pse if pse else None
    phrase, tone = _bias_phrase(z, primary, noun, title)

    def pos(v):
        return 50 + 50 * max(-1.0, min(1.0, v / scale))

    band = None
    if pse:
        lo, hi = pos(primary - 2 * pse), pos(primary + 2 * pse)
        band = {"left": round(lo, 1), "width": round(max(0.8, hi - lo), 1)}
    l_tone = "ok"
    l_z = lb / lse if lb is not None and lse else None
    if l_z is not None:
        l_tone = "ok" if abs(l_z) < 2 else ("warn" if abs(l_z) < 3.5 else "bad")
    detail = ""
    if key == "home" and s and _f(s.get("home_pred")) is not None:
        detail = f'{fmt.pct(s["home_pred"], 1)} predicted, {fmt.pct(s["home_actual"], 1)} actual'
    out_of_scale = abs(primary) > scale
    return {
        "key": key, "title": title, "phrase": phrase, "tone": tone, "detail": detail,
        "value": fmt.signed(primary, 1), "band_text": f"±{fmt.num(2 * pse, 1)}" if pse else "",
        "window": pwin, "n": int(pn or 0), "z": z, "raw": primary, "raw_se": pse,
        "season": sb, "last100": lb,
        "season_pos": round(pos(sb), 1) if sb is not None else None,
        "last_pos": round(pos(lb), 1) if lb is not None else None,
        "band": band, "last_text": fmt.signed(lb, 1) if lb is not None else "",
        "last_tone": l_tone, "last_flag": _last_flag(lb, l_z) if l_tone != "ok" else "",
        "scale": fmt.num(scale, 0), "clipped": out_of_scale,
    }


def _last_flag(bias: float, z: float | None) -> str:
    return f"{'over' if bias > 0 else 'under'}-predicting, {abs(z):.1f} SE" if z else ""


# ---- charts over time

def _trends(c, labels: list[str]) -> dict:
    rows = _q(c, "SELECT as_of, model, n, log_loss, accuracy, home_pred, home_actual, "
                 "margin_bias, total_bias FROM metric_snapshots WHERE window='last100' "
                 "ORDER BY as_of")
    by: dict[str, list[dict]] = {}
    for r in rows:
        if (r["n"] or 0) >= MIN_PLOT_N:
            by.setdefault(r["model"], []).append(r)
    chart_labels = [m for m in CHART_MODELS if m in by]

    def series(field, scale=1.0, models=chart_labels, fn=None):
        out = []
        for m in models:
            pts = []
            for r in by.get(m, []):
                v = fn(r) if fn else _f(r.get(field))
                if v is not None:
                    pts.append((r["as_of"], v * scale, r["n"]))
            name = _label(m)[0]
            out.append({"key": m, "label": name, "pts": pts,
                        "cls": "mk" if m == MARKET else ("s1" if m == "gbm_market" else "s2")})
        return out

    legend = [{"label": _label(m)[0], "cls": "mk" if m == MARKET else
               ("s1" if m == "gbm_market" else "s2")} for m in chart_labels]
    perf = hc.chart_pair(series("log_loss"), ytick=lambda v: f"{v:.2f}",
                         aria="Rolling last-100-game log-loss by date, lower is better",
                         min_span=0.06)
    acc = hc.chart_pair(series("accuracy", 100.0), ytick=lambda v: f"{v:.0f}%",
                        aria="Rolling last-100-game accuracy by date", min_span=8)
    bias_charts = []
    specs = (
        ("home", "Home win rate", "pts", lambda r: _gap(r)),
        ("margin", "Home margin", "pts", lambda r: _f(r.get("margin_bias"))),
        ("total", "Game total", "pts", lambda r: _f(r.get("total_bias"))),
    )
    for key, title, unit, fn in specs:
        svg = hc.small_chart(series(None, fn=fn), ytick=lambda v: fmt.signed(v, 0) if abs(v) >= 1
                             or v == 0 else fmt.signed(v, 1), zero=True,
                             aria=f"{title} bias by date, predicted minus actual, last 100 games",
                             min_span=4 if key == "home" else 2)
        if svg:
            bias_charts.append({"key": key, "title": title, "unit": unit, "svg": svg})
    return {"perf": perf, "acc": acc, "legend": legend, "bias_charts": bias_charts,
            "min_plot_n": MIN_PLOT_N}


def _gap(r: dict) -> float | None:
    hp, ha = _f(r.get("home_pred")), _f(r.get("home_actual"))
    return None if hp is None or ha is None else 100 * (hp - ha)


# ---- drift

def _drift(c, today: date) -> dict | None:
    r = _q(c, "SELECT MAX(as_of) AS a FROM drift_snapshots")
    day = r[0]["a"] if r else None
    if not day:
        return None
    rows = [x for x in _q(c, "SELECT * FROM drift_snapshots WHERE as_of=? ORDER BY psi DESC",
                          (day,)) if _f(x.get("psi")) is not None]
    if not rows:
        return None
    top = max(x["psi"] for x in rows)
    scale = max(0.6, math.ceil(top * 10 * 1.05) / 10)
    feats = []
    for x in rows:
        psi = float(x["psi"])
        tone = "crit" if psi > PSI_CRIT else ("warn" if psi > PSI_WARN else "ok")
        feats.append({"feature": x["feature"], "label": PSI_LABELS.get(
            x["feature"], x["feature"].replace("_", " ").capitalize()),
            "psi": psi, "text": f"{psi:.2f}", "tone": tone, "w": round(100 * psi / scale, 1),
            "n": x.get("cur_n")})
    d = fmt.parse_day(day)
    return {"as_of": day, "ref": rows[0].get("ref_label") or "", "features": feats,
            "x_warn": round(100 * PSI_WARN / scale, 1), "x_crit": round(100 * PSI_CRIT / scale, 1),
            "scale": scale, "max_psi": top, "n": max((x.get("cur_n") or 0) for x in rows),
            "age": (today - d).days if d else None,
            "flagged": sum(1 for f in feats if f["tone"] != "ok")}


# ---- alerts

def _alerts_view(open_: list[dict], resolved: list[dict]) -> dict:
    def one(a):
        lab, cls = SEV_PILL.get(a["severity"], (a["severity"].capitalize(), "unknown"))
        return {"key": a["key"], "severity": a["severity"], "sev_label": lab, "sev_cls": cls,
                "kind": (a.get("kind") or "").capitalize(), "message": a["message"],
                "since": fmt.stamp(a.get("created_at")) or fmt.medium_date(a.get("created_at")),
                "updated": fmt.stamp(a.get("updated_at")),
                "resolved": fmt.stamp(a.get("resolved_at")) or fmt.medium_date(a.get("resolved_at")),
                "created_at": a.get("created_at")}
    o = [one(a) for a in open_]
    return {"open": o, "resolved": [one(a) for a in resolved],
            "critical": sum(1 for a in o if a["severity"] == "critical"),
            "warn": sum(1 for a in o if a["severity"] == "warn"),
            "info": sum(1 for a in o if a["severity"] == "info")}


# ---- data checks

_JOBS = (("pregame", "Pregame"), ("postgame", "Postgame"), ("stale", "Freshness"))
_STALE_NAMES = {"stale.pregame": "Pregame run is recent", "stale.postgame": "Postgame run is recent"}


def _check_title(name: str) -> str:
    if name in _STALE_NAMES:
        return _STALE_NAMES[name]
    tail = name.split(".", 1)[-1].replace("_", " ")
    return tail[:1].upper() + tail[1:]


def _check_value(name: str, v: float | None, t: float | None) -> tuple[str, str]:
    """Measured value and limit as short text, or blanks when the pair says nothing."""
    if v is None:
        return "", ""
    if name.startswith("stale."):
        return f"{v:.1f} h", (f"{t:.0f} h" if t else "")
    if t is not None and 0 < t <= 1 and 0 <= v <= 1:
        return fmt.pct(v, 0), fmt.pct(t, 0)
    if not t:
        return (fmt.num(v, 2) if v else ""), ""
    return fmt.num(v, 2), fmt.num(t, 2)


def _checks(c) -> dict:
    rows = _q(c, "SELECT * FROM data_checks ORDER BY run_at DESC, check_name LIMIT 1500")
    if not rows:
        return {"groups": [], "counts": {}, "n": 0, "latest": []}
    hist: dict[str, list[dict]] = {}
    for r in rows:
        hist.setdefault(r["check_name"], []).append(r)
    groups: dict[str, list[dict]] = {}
    counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
    latest = []
    for name, rs in sorted(hist.items()):
        r = rs[0]
        job = name.split(".", 1)[0]
        job = job if job in dict(_JOBS) else "stale"
        st = r["status"] if r["status"] in STATUS_PILL else "skip"
        counts[st] += 1
        lab, cls = STATUS_PILL[st]
        v, t = _f(r.get("value")), _f(r.get("threshold"))
        vtxt, ttxt = _check_value(name, v, t)
        item = {
            "name": name, "title": _check_title(name), "status": st, "pill": lab, "pill_cls": cls,
            "value": vtxt, "threshold": ttxt,
            "detail": (r.get("detail") or "").strip(), "run_at": r["run_at"],
            "history": [{"status": h["status"] if h["status"] in STATUS_PILL else "skip",
                         "when": fmt.stamp(h["run_at"]), "detail": (h.get("detail") or "").strip()}
                        for h in reversed(rs[:14])],
            "runs": len(rs),
        }
        groups.setdefault(job, []).append(item)
        latest.append({"name": name, "status": st, "value": v, "threshold": t,
                       "detail": item["detail"], "run_at": r["run_at"]})
    out = []
    for job, label in _JOBS:
        items = groups.get(job)
        if items:
            out.append({"job": job, "label": label, "rows": items,
                        "ran": fmt.stamp(max(i["run_at"] for i in items)),
                        "bad": sum(1 for i in items if i["status"] in ("fail", "warn"))})
    return {"groups": out, "counts": counts, "n": len(latest), "latest": latest}


# ---- registry history

def _events(c) -> list[dict]:
    return _q(c, "SELECT e.id, e.ts, e.version_id, e.action, e.reason, v.model_name, v.code_sha "
                 "FROM registry_events e LEFT JOIN model_versions v ON v.version_id=e.version_id "
                 "ORDER BY e.ts DESC, e.id DESC LIMIT 60")


def _history_view(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        lab, tone = ACTIONS.get(e["action"], ((e["action"] or "Event").capitalize(), "plain"))
        model = e.get("model_name")
        out.append({"when": fmt.stamp(e["ts"]) or fmt.medium_date(e["ts"]), "action": lab,
                    "tone": tone, "version_id": _vshort(e["version_id"]),
                    "model": _label(model)[0] if model else "",
                    "sha": (e.get("code_sha") or "")[:7], "dirty": "dirty" in (e.get("code_sha") or ""),
                    "reason": (e.get("reason") or "").strip(),
                    "raw_action": e["action"], "ts": e["ts"]})
    return out


# ---- overall status

def _overall(vm: dict) -> dict:
    al = vm["alerts"]
    counts = vm["checks"].get("counts") or {}
    covered = {a["key"][5:] for a in al["open"] if a["key"].startswith("data:")}
    uncovered = 0
    for g in vm["checks"].get("groups", []):
        for i in g["rows"]:
            if i["status"] in ("fail", "warn") and i["name"] not in covered:
                uncovered += 1
    watch = al["warn"] + uncovered
    has_any = vm["has_metrics"] or vm["checks"].get("n") or al["open"] or vm["drift"]
    if al["critical"]:
        state = "critical"
        n = al["critical"]
        head = f"{n} critical alert{'s' if n != 1 else ''} need{'s' if n == 1 else ''} attention."
        sub = ("Something has moved far enough from its expected range that the Brief's numbers "
               "should be read with care until it is looked at.")
    elif watch:
        state = "warn"
        head = f"Healthy, with {watch} thing{'s' if watch != 1 else ''} to watch."
        sub = "Nothing is broken badly enough to page anyone. The items below are worth a look."
    elif has_any:
        state = "ok"
        head = "Everything checks out."
        sub = "No open alerts, and the latest data checks passed."
    else:
        state = "unknown"
        head = "Monitoring hasn't started yet."
        sub = ("Model health fills in once games are scored. Early in a season there are too "
               "few results to say anything yet.")
    word = {"ok": "Healthy", "warn": "Watch", "critical": "Critical", "unknown": "Pending"}[state]
    return {"state": state, "headline": head, "sub": sub, "word": word, "watch": watch,
            "failing": counts.get("fail", 0) + counts.get("warn", 0)}


# ------------------------------------------------------------------ JSON summary

def api_summary(vm: dict) -> dict:
    def win(s):
        if not s:
            return None
        return {"n": s["n"], "log_loss": s["log_loss"], "brier": s["brier"],
                "accuracy": s["accuracy"], "ece": s["ece"],
                "ll_diff_vs_mkt": s["ll_diff_vs_mkt"], "ll_diff_se": s["ll_diff_se"]}

    return {
        "status": vm["status"]["state"], "headline": vm["status"]["headline"],
        "as_of": vm["as_of"], "snapshot_age_days": vm["snapshot_age"],
        "champions": [{"model": v["model"], "version_id": v["version_id"],
                       "promoted_at": v["when_raw"], "code_sha": v["sha"] or None}
                      for v in vm["champions"]],
        "alerts": {"open": len(vm["alerts"]["open"]), "critical": vm["alerts"]["critical"],
                   "warn": vm["alerts"]["warn"], "info": vm["alerts"]["info"],
                   "items": [{"key": a["key"], "severity": a["severity"], "kind": a["kind"].lower(),
                              "message": a["message"], "since": a["created_at"]}
                             for a in vm["alerts"]["open"]]},
        "models": [{"model": m["id"], "label": m["name"], "role": m["role"],
                    "version": m["version"], "season": win(m["season"]),
                    "last100": win(m["last100"]),
                    "vs_market": m["headline"]["text"] if m["headline"] else None}
                   for m in vm["models"]],
        "calibration": [{"model": cb["id"], "n": cb["n"], "slope": cb["slope"],
                         "spiegelhalter_z": cb["z"], "verdict": cb["verdict"]["label"]}
                        for cb in vm["calibration"]],
        "bias": [{"model": cd["id"], **{r["key"]: {"season": r["season"], "last100": r["last100"],
                                                  "verdict": r["phrase"]} for r in cd["rows"]}}
                 for cd in vm["bias"]["cards"]],
        "drift": ({"as_of": vm["drift"]["as_of"], "max_psi": vm["drift"]["max_psi"],
                   "flagged": vm["drift"]["flagged"],
                   "features": [{"feature": f["feature"], "psi": f["psi"], "level": f["tone"]}
                                for f in vm["drift"]["features"]]} if vm["drift"] else None),
        "checks": {**{k: vm["checks"].get("counts", {}).get(k, 0)
                      for k in ("ok", "warn", "fail", "skip")},
                   "latest": vm["checks"].get("latest", [])},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
