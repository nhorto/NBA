"""The DFS tab of Model Health (`/health/dfs`): view model from the DFS tracking tables.

Reads (read-only connection, every query tolerant of a missing or empty table): the snapshot
tables ``dfs_metric_snapshots`` / ``player_metric_snapshots`` (written nightly by
``proj/dfs/grade.py``), ``dfs_slates`` / ``dfs_runs`` / ``dfs_lineup_grades``, ``alerts``,
``model_versions`` and, for the REAL block, ``dfs_entries`` and the uploaded standings
(``proj/dfs/standings.py``).

Three kinds of numbers, never mixed on the page:

* **Paper results**: our stored builds scored with the real box scores of games that happened.
  They were not entered in contests, so they carry no money. (`lineups`, `players`, `slates`.)
* **Real contests**: the owner's own entries and DK standings he uploaded: fees, winnings, net,
  ROI, real cash lines. (`real`.)
* **Backtest / simulation**: not shown here; it lives in the research reports.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone

from . import fmt
from . import health_charts as hc

WINDOWS = ("7d", "30d", "season")
WINDOW_LABEL = {"7d": "7 days", "30d": "30 days", "season": "Season"}
OPENING_NIGHT = "2026-10-20"
FIRST_GRADES = "2026-10-21"
ROLE_LABEL = {"cash": "Our cash lineup", "topn": "Our Top N lineups", "gpp": "Our GPP lineups",
              "swap": "Our late swaps"}
BASE_LABEL = {"fppg": "DK FPPG optimizer", "salary": "Salary-implied optimizer",
              "hindsight": "Best possible (hindsight)", "random": "Random lineups"}
BASE_ORDER = ("fppg", "salary", "hindsight", "random")
PLAYER_LABEL = {"ours": "Our projections", "fppg": "DK FPPG", "salary": "Salary-implied"}
SERIES_CLS = {"ours": "s1", "fppg": "s2", "salary": "mk"}
ALERT_PREFIXES = ("dfs:", "data:dfs_", "data:ids_")
SEV_PILL = {"critical": ("Critical", "out"), "warn": ("Warn", "questionable"),
            "info": ("Info", "unknown")}
SEV_ORDER = {"critical": 0, "warn": 1, "info": 2}


def _q(conn, sql: str, args: tuple = ()) -> list[dict]:
    if conn is None:
        return []
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


def _money(v, signed: bool = False) -> str:
    x = _f(v)
    if x is None:
        return fmt.EM
    sign = ("+" if x > 0 else fmt.MINUS if x < 0 else "") if signed else (fmt.MINUS if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def _cell(row: dict | None, kind: str, nd: int = 1) -> dict:
    """One table cell: {'text', 'se', 'n'}; kind: num | pct | signed."""
    if not row or _f(row.get("value")) is None:
        return {"text": fmt.EM, "se": "", "n": ""}
    v, se = _f(row["value"]), _f(row.get("se"))
    if kind == "pct":
        text = f"{100 * v:.{nd}f}%"
        sep = f"±{100 * se:.{nd}f}" if se is not None else ""
    elif kind == "signed":
        text = fmt.signed(v, nd)
        sep = f"±{se:.{nd}f}" if se is not None else ""
    else:
        text = fmt.num(v, nd)
        sep = f"±{se:.{nd}f}" if se is not None else ""
    n = row.get("n")
    return {"text": text, "se": sep, "n": f"n={int(n)}" if n else ""}


# ------------------------------------------------------------------ snapshots

def _latest(rows: list[dict]) -> dict[tuple, dict]:
    """{(model, metric, window): newest row}."""
    out: dict[tuple, dict] = {}
    for r in rows:
        k = (r["model"], r["metric"], r["window"])
        if k not in out or r["as_of"] > out[k]["as_of"]:
            out[k] = r
    return out


def _load(conn, table: str, today: date) -> list[dict]:
    since = (today - timedelta(days=430)).isoformat()
    return _q(conn, f"SELECT as_of, model, metric, \"window\", n, value, se, details_json "
                    f"FROM {table} WHERE as_of>=? AND model NOT LIKE 'slate:%'", (since,))


def _lineup_tables(dfs_latest: dict) -> list[dict]:
    models = {m for (m, _, _) in dfs_latest}
    roles = [m for m in ("cash", "topn", "gpp", "swap") if m in models]
    roles += sorted(m for m in models if m not in roles and m not in BASE_LABEL
                    and m != "real" and not m.startswith("sd_"))
    bases = [b for b in BASE_ORDER if b in models]
    tables = []
    for metric, title, note, kind, nd in (
            ("actual_mean", "Average score", "DK points scored by the lineup (classic slates).",
             "num", 1),
            ("ratio_hindsight", "Share of the best possible score",
             "Score divided by the best lineup anyone could have built in hindsight.", "pct", 0),
            ("pct_random", "Percentile among random lineups",
             "Where the score would rank among 5,000 random salary-valid lineups.", "pct", 0)):
        rows = []
        for m in roles + bases:
            if metric == "pct_random" and m in ("random", "hindsight"):
                continue
            cells = [_cell(dfs_latest.get((m, metric, w)), kind, nd) for w in WINDOWS]
            if all(c["text"] == fmt.EM for c in cells):
                continue
            rows.append({"label": ROLE_LABEL.get(m, f"Our {m} lineups") if m in roles
                         else BASE_LABEL[m], "ours": m in roles, "cells": cells})
        if rows:
            tables.append({"title": title, "note": note, "rows": rows})
    return tables


def _vs_rows(dfs_latest: dict) -> list[dict]:
    rows = []
    for m in ("cash", "topn", "gpp"):
        for other, lab in (("fppg", "DK FPPG optimizer"), ("salary", "Salary-implied optimizer")):
            cells = [_cell(dfs_latest.get((m, f"diff_vs_{other}", w)), "signed", 1) for w in WINDOWS]
            if any(c["text"] != fmt.EM for c in cells):
                rows.append({"label": f"{ROLE_LABEL.get(m, m)} minus {lab}", "cells": cells})
    return rows


def _would_cash(dfs_latest: dict) -> list[dict]:
    rows = []
    for m in ("cash", "topn", "gpp"):
        cells = [_cell(dfs_latest.get((m, "would_cash_rate", w)), "pct", 0) for w in WINDOWS]
        if any(c["text"] != fmt.EM for c in cells):
            rows.append({"label": ROLE_LABEL.get(m, m), "cells": cells})
    return rows


def _player_tables(pl_latest: dict) -> dict:
    mae_rows = []
    for m in ("ours", "fppg", "salary"):
        cells = [_cell(pl_latest.get((m, "dk_mae", w)), "num", 2) for w in WINDOWS]
        if any(c["text"] != fmt.EM for c in cells):
            mae_rows.append({"label": PLAYER_LABEL[m], "ours": m == "ours", "cells": cells})
    gain_rows = []
    for other in ("fppg", "salary"):
        cells = [_cell(pl_latest.get(("ours", f"dk_mae_gain_vs_{other}", w)), "signed", 1)
                 for w in WINDOWS]
        if any(c["text"] != fmt.EM for c in cells):
            gain_rows.append({"label": f"Error reduction vs {PLAYER_LABEL[other]} (%)",
                              "cells": cells})
    calib = []
    for metric, label, kind, nd, tip in (
            ("dk_cov80", "80% range coverage", "pct", 1, "target 80%"),
            ("dk_cov50", "50% range coverage", "pct", 1, "target 50%"),
            ("dk_crps", "CRPS (DK points)", "num", 2, "lower is better"),
            ("min_mae_played15", "Minutes error (players 15+ min)", "num", 2, "minutes, when they play"),
            ("min_mae", "Minutes error (all, incl. DNP)", "num", 2, "minutes"),
            ("min_bias", "Minutes bias (projected minus actual)", "signed", 2, "minutes"),
            ("dk_bias", "DK points bias (projected minus actual)", "signed", 2, "positive = over")):
        cells = [_cell(pl_latest.get(("ours", metric, w)), kind, nd) for w in WINDOWS]
        if any(c["text"] != fmt.EM for c in cells):
            calib.append({"label": label, "tip": tip, "cells": cells})
    return {"mae": mae_rows, "gain": gain_rows, "calib": calib}


# ------------------------------------------------------------------ trends

def _series(conn, table: str, models: tuple, metric: str, window: str = "7d") -> dict[str, list]:
    out = {}
    for m in models:
        rows = _q(conn, f"SELECT as_of, n, value FROM {table} WHERE model=? AND metric=? AND "
                        f"\"window\"=? ORDER BY as_of", (m, metric, window))
        pts = [(r["as_of"], _f(r["value"]), r["n"]) for r in rows if _f(r["value"]) is not None]
        if pts:
            out[m] = pts
    return out


def _trend_charts(conn) -> list[dict]:
    charts = []
    ser = _series(conn, "dfs_metric_snapshots", ("cash", "fppg", "salary"), "ratio_hindsight")
    series = [{"key": k, "label": {"cash": "Our cash lineup", "fppg": "DK FPPG optimizer",
                                   "salary": "Salary-implied"}[k],
               "cls": {"cash": "s1", "fppg": "s2", "salary": "mk"}[k], "pts": v}
              for k, v in ser.items()]
    pair = hc.chart_pair(series, ytick=lambda v: f"{100 * v:.0f}%",
                         aria="Share of the best possible score, rolling 7 days") if series else None
    if pair:
        charts.append({"title": "Share of the best possible score", "note": "rolling 7 days, higher is better",
                       "legend": [{"cls": s["cls"], "label": s["label"]} for s in series],
                       "wide": pair["wide"], "narrow": pair["narrow"]})
    ser = _series(conn, "player_metric_snapshots", ("ours", "fppg", "salary"), "dk_mae")
    series = [{"key": k, "label": PLAYER_LABEL[k], "cls": SERIES_CLS[k], "pts": v}
              for k, v in ser.items()]
    pair = hc.chart_pair(series, ytick=lambda v: f"{v:.1f}",
                         aria="DK points error per player, rolling 7 days") if series else None
    if pair:
        charts.append({"title": "DK points error per player", "note": "rolling 7 days, lower is better",
                       "legend": [{"cls": s["cls"], "label": s["label"]} for s in series],
                       "wide": pair["wide"], "narrow": pair["narrow"]})
    return charts


# ------------------------------------------------------------------ slates, alerts, versions

def _slate_rows(conn, today: date) -> list[dict]:
    since = (today - timedelta(days=7)).isoformat()
    slates = _q(conn, "SELECT slate_id, nba_day, tag, game_type_id, lock_utc FROM dfs_slates "
                      "WHERE nba_day>=? AND nba_day<=? ORDER BY lock_utc DESC, slate_id DESC LIMIT 24",
                (since, (today + timedelta(days=1)).isoformat()))
    if not slates:
        return []
    base = {}
    for r in _q(conn, "SELECT as_of, details_json FROM dfs_metric_snapshots WHERE "
                      "model LIKE 'slate:%' AND metric='baselines' AND as_of>=?", (since,)):
        s = _json(r["details_json"])
        if s.get("slate_id") is not None:
            base[str(s["slate_id"])] = s
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for s in slates:
        sid = str(s["slate_id"])
        runs = _q(conn, "SELECT kind, ok, model_version FROM dfs_runs WHERE slate_id=? AND "
                        "kind IN ('lock45','lock10')", (sid,))
        graded = _q(conn, "SELECT COUNT(*) AS n FROM dfs_lineup_grades WHERE slate_id=?", (sid,))
        locked = bool(s["lock_utc"] and s["lock_utc"] <= now)

        def state(kind):
            ok = [r for r in runs if r["kind"] == kind and r["ok"]]
            if ok:
                return ("OK", "healthy")
            if any(r["kind"] == kind for r in runs):
                return ("Failed", "out")
            return ("Missed", "out") if locked else ("Pending", "unknown")
        b = base.get(sid) or {}
        ours = {(o.get("role"), o.get("rank")): o.get("actual") for o in b.get("ours", [])}
        score = ours.get(("cash", 1)) or ours.get(("cash", None))
        mv = next((r["model_version"] for r in runs if r["ok"] and r["model_version"]), None)
        out.append({
            "slate_id": sid, "day": s["nba_day"], "label": (s["tag"] or "slate").capitalize(),
            "lock": fmt.tipoff(s["lock_utc"]) or fmt.EM, "l45": state("lock45"),
            "l10": state("lock10"), "version": mv or "",
            "graded": bool(graded and graded[0]["n"]),
            "ours": fmt.num(score, 1), "fppg": fmt.num((b.get("fppg") or {}).get("actual"), 1),
            "salary": fmt.num((b.get("salary") or {}).get("actual"), 1),
            "hindsight": fmt.num((b.get("hindsight") or {}).get("actual"), 1)})
    return out


def _mins(v) -> str:
    """Minutes as '9 min' / '1.4 h' / '25 h' (dash for None / NaN)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return fmt.EM
    if v < 90:
        return f"{v:.0f} min"
    return f"{v / 60:.1f} h" if v < 600 else f"{v / 60:.0f} h"


def _latency(conn) -> dict:
    """News latency (P7 part B, display only): injury news -> our next re-run of the game, and
    -> lock (tip), this season (`proj/news_latency.py`)."""
    from ..proj import news_latency as NL
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        ev = NL.events(conn, NL.SEASON_START, as_of=now)
    except Exception:  # noqa: BLE001 - a display row must never break the page
        return {}
    if not len(ev):
        return {}
    s = NL.summary(ev)
    rows = []
    for key, label in (("nba_official", "Official report"), ("espn", "ESPN status"),
                       ("all", "All news")):
        v = s.get(key) or {}
        if not v.get("n"):
            continue
        q, lk = v["rerun_q"], v["lock_q"]
        rows.append({"label": label, "n": v["n"], "missed": v["missed"],
                     "p50": _mins(q[1]), "p75": _mins(q[2]), "p90": _mins(q[3]),
                     "lock": _mins(lk[1]), "w60": fmt.pct(v["within_60"], 0)})
    done = ev[~ev["pending"].astype(bool)]
    late = done[done["to_lock_min"] <= 60].sort_values("news_at", ascending=False).head(6)
    recent = [{"player": r.player_name, "team": r.team,
               "change": f"{r.old if isinstance(r.old, str) and r.old else 'unlisted'} to {r.new}",
               "src": NL.SOURCE_LABEL.get(r.source, r.source),
               "news": fmt.tipoff(r.news_at) or r.news_at, "to_lock": _mins(r.to_lock_min),
               "rerun": (f"{r.rerun_kind} after {_mins(r.to_rerun_min)}" if not r.missed
                         else "none before tip")}
              for r in late.itertuples(index=False)]
    return {"rows": rows, "recent": recent, "pending": s.get("pending", 0)}


_METRIC_LABEL = {"dk_crps": "DK CRPS", "dk_mae": "DK error", "min_mae": "Minutes error",
                 "pplay_ll": "P(play) log loss"}
_POP_LABEL = {"emin10": "E[min] 10+", "changed": "changed rows", "all": "all rows"}
_SHADOW_SE_NIGHTS = 5          # fewer nights: a two-way SE is not worth showing
_DECISION_PILL = {"pass": ("Pass", "healthy"), "fail": ("Fail", "out"),
                  "pending": ("Pending", "unknown")}


def _shadows(conn) -> dict:
    """Player-side live shadows (IMPROVEMENTS E1, `proj/shadows.py`): each pre-registered arm vs
    the champion on the same runs, from the nightly report. Display only."""
    from ..proj import shadows as SH
    try:
        rep = SH.load_report(conn)
    except Exception:  # noqa: BLE001 - a display row must never break the page
        return {}
    if not rep or not rep.get("arms"):
        return {}
    rows = []
    for a in rep["arms"]:
        p = a.get("primary") or {}
        se_ok = (p.get("nights") or 0) >= _SHADOW_SE_NIGHTS
        tiers = {t: v.get("z") for t, v in (a.get("tiers") or {}).items()
                 if v.get("z") is not None and se_ok}
        worst = max(tiers.items(), key=lambda kv: kv[1]) if tiers else None
        label, cls = _DECISION_PILL.get(a.get("decision"), ("Pending", "unknown"))
        run = a.get("run") or ""
        rows.append({
            "arm": a.get("arm"), "switch": a.get("switch") or "",
            "metric": _METRIC_LABEL.get(a.get("metric"), a.get("metric") or ""),
            "where": f"{_POP_LABEL.get(a.get('population'), a.get('population') or '')}, "
                     f"{'last run before tip' if run == 'last' else run + ' run'}",
            "nights": f"{p.get('nights') or 0} of {a.get('min_nights') or 0}",
            "diff": fmt.signed(p.get("mean"), 4),
            "se": fmt.num(p.get("se"), 4) if se_ok else fmt.EM,
            "z": fmt.signed(p.get("z"), 1) if se_ok else fmt.EM,
            "worst": f"{worst[0]} {fmt.signed(worst[1], 1)}" if worst else fmt.EM,
            "bar": f"z \u2264 {fmt.signed((a.get('bar') or {}).get('z'), 0)}",
            "status": label, "cls": cls, "reason": a.get("reason") or "",
            "paused": a.get("status") != "active"})
    runs = rep.get("runs") or []
    last = runs[-1] if runs else None
    last_txt = ""
    if last:
        ok = sum(1 for v in (last.get("arms") or {}).values() if v.get("status") == "ok")
        when = f"last run {fmt.short_date(last.get('day'))} {last.get('slot') or ''}"
        last_txt = (f"{when}: skipped ({last.get('reason') or 'no reason'})"
                    if last.get("status") == "skipped" else
                    f"{when}: {last.get('status')}, {max(ok - 1, 0)} arms in "
                    f"{fmt.num(last.get('seconds_total'), 0)} s")
    return {"rows": rows, "nights": len(rep.get("nights") or []), "as_of": rep.get("as_of"),
            "last": last_txt}


def _alerts(conn) -> list[dict]:
    rows = _q(conn, "SELECT key, severity, kind, message, created_at, updated_at FROM alerts "
                    "WHERE status='open'")
    rows = [r for r in rows if r["key"].startswith(ALERT_PREFIXES)]
    rows.sort(key=lambda r: (SEV_ORDER.get(r["severity"], 9), r["updated_at"] or ""))
    out = []
    for a in rows:
        lab, cls = SEV_PILL.get(a["severity"], (a["severity"].capitalize(), "unknown"))
        out.append({"key": a["key"], "sev_label": lab, "sev_cls": cls, "message": a["message"],
                    "kind": (a["kind"] or "").capitalize(),
                    "since": fmt.stamp(a["created_at"]) or fmt.medium_date(a["created_at"])})
    return out


def _versions(conn) -> list[dict]:
    from ..mlops import dfs_monitor as DM
    rows = _q(conn, "SELECT version_id, model_name, status, created_at, promoted_at, notes "
                    "FROM model_versions WHERE model_name IN (%s) ORDER BY created_at DESC"
              % ",".join("?" * len(DM.COMPONENTS)), DM.COMPONENTS)
    out = []
    for r in rows:
        if r["status"] != "champion":
            continue
        pv = ""
        pr = _q(conn, "SELECT params_json FROM model_versions WHERE version_id=?", (r["version_id"],))
        if pr:
            pv = _json(pr[0]["params_json"]).get("model_version") or ""
        out.append({"component": DM.COMPONENT_LABEL.get(r["model_name"], r["model_name"]),
                    "version": r["version_id"].split("@")[-1], "pipeline": pv,
                    "since": fmt.medium_date(r["promoted_at"] or r["created_at"])})
    order = {n: i for i, n in enumerate(DM.COMPONENT_LABEL.values())}
    out.sort(key=lambda r: order.get(r["component"], 9))
    return out


# ------------------------------------------------------------------ real results

def _entry_day(e: dict) -> str:
    return (e.get("nba_day") or (e.get("created_at") or "")[:10] or "")


def _real_block(conn, today: date) -> dict:
    from ..proj.dfs import standings as ST
    res = ST.my_results(conn) if conn is not None else {"entries": [], "totals": ST._totals([])}
    entries = res["entries"]
    wins = {"7d": today - timedelta(days=6), "30d": today - timedelta(days=29),
            "season": date(today.year if today.month >= 8 else today.year - 1, 8, 1)}
    cols = []
    for w in WINDOWS:
        sel = [e for e in entries if _entry_day(e) >= wins[w].isoformat()]
        t = ST._totals(sel)
        cols.append({"window": w, "label": WINDOW_LABEL[w], "t": t})

    def row(label, fn):
        return {"label": label, "cells": [{"text": fn(c["t"]), "se": "", "n": ""} for c in cols]}
    table = [
        row("Entries", lambda t: str(t["entries"])),
        row("Entry fees", lambda t: _money(t["fees"]) if t["entries"] else fmt.EM),
        row("Winnings", lambda t: _money(t["winnings"]) if t["settled"] else fmt.EM),
        row("Net", lambda t: _money(t["net"], True) if t["settled"] else fmt.EM),
        row("ROI", lambda t: fmt.pct(t["roi"], 1) if t["roi"] is not None else fmt.EM),
        row("Cashed", lambda t: f"{t['cashed']} of {t['settled']}" if t["settled"] else fmt.EM),
        row("Winnings not entered yet", lambda t: str(t["pending"])),
    ]
    recent = []
    for e in entries[:6]:
        recent.append({
            "day": e.get("nba_day") or (e.get("created_at") or "")[:10],
            "contest": e.get("contest_name") or f"Contest {e['contest_id']}",
            "fee": _money(e.get("entry_fee")), "points": fmt.num(e.get("points"), 1),
            "place": (f"{e['rank']:,} of {e['field_size']:,}" if e.get("rank") and e.get("field_size")
                      else fmt.EM),
            "payout": _money(e.get("payout")), "net": _money(e.get("net"), True)})
    contests = []
    for c in ST.list_contests(conn)[:4] if conn is not None else []:
        contests.append({"contest_id": c["contest_id"], "name": c["name"] or f"Contest {c['contest_id']}",
                         "day": c["nba_day"] or "", "field": f"{c['field_size']:,}",
                         "win": fmt.num(c["winning_score"], 1), "top1": fmt.num(c["top1_score"], 1),
                         "line": fmt.num(c["cash_line"], 1) if c["cash_line"] is not None else fmt.EM})
    return {"table": table, "columns": cols, "has_entries": bool(entries), "recent": recent,
            "contests": contests, "totals": res["totals"]}


# ------------------------------------------------------------------ build

def build(conn: sqlite3.Connection | None, today: date | None = None) -> dict:
    today = today or date.today()
    dfs_rows = _load(conn, "dfs_metric_snapshots", today) if conn else []
    pl_rows = _load(conn, "player_metric_snapshots", today) if conn else []
    dfs_latest, pl_latest = _latest(dfs_rows), _latest(pl_rows)
    as_of = max([r["as_of"] for r in dfs_rows + pl_rows], default=None)
    vm = {
        "today": today.isoformat(), "as_of": as_of, "has_data": bool(dfs_rows or pl_rows),
        "opening_night": OPENING_NIGHT, "first_grades": FIRST_GRADES,
        "windows": [{"id": w, "label": WINDOW_LABEL[w]} for w in WINDOWS],
        "lineup_tables": _lineup_tables(dfs_latest), "vs_rows": _vs_rows(dfs_latest),
        "cash_rows": _would_cash(dfs_latest),
        "players": _player_tables(pl_latest),
        "charts": _trend_charts(conn) if conn else [],
        "slates": _slate_rows(conn, today) if conn else [],
        "latency": _latency(conn) if conn else {},
        "shadows": _shadows(conn) if conn else {},
        "alerts": _alerts(conn) if conn else [],
        "versions": _versions(conn) if conn else [],
        "real": _real_block(conn, today),
    }
    vm["alert_counts"] = {"critical": sum(a["sev_cls"] == "out" for a in vm["alerts"]),
                          "n": len(vm["alerts"])}
    return vm
