"""Weekly DFS report (docs/PLAYERS.md §8 Phase 3, task T3.3).

``weekly_dfs_report(conn, week_end, days=7)`` reads what the nightly grading wrote
(``player_projection_grades``, ``dfs_lineup_grades``, the per-slate baselines in
``dfs_metric_snapshots``, plus uploaded standings and the owner's entries when they exist) and
answers, in plain words, how the projections and lineups did over the window and the season so
far. Read-only: it never writes to the database.

What it counts

* **Projections.** One row per (game, player): the lock10 run if there is one, else lock45, else a
  manual run, else a swap run (the monitor's rule), projected minutes >= 10, a player who did not
  play scores 0. Cuts: all slates, by slate kind (main / showdown), by run kind (each kind
  deduplicated on its own), by tier and by model version. Tiers use the minutes model's tier edges
  (12 / 20 / 26 / 32 minutes of E[min | plays], `proj/minutes.py` TIER_EDGES; the monitor's
  "starter" is the top two): star >= 32, starter 26-32, rotation 20-26, bench < 20.
* **Lineups.** Per slate, our cash lineup and our Top N (mean over the N) against the two baseline
  lineups the grader built from the same pool (DraftKings FPPG, salary alone), on actual points,
  share of the hindsight-best lineup and percentile among 5,000 random lineups. The slate is the
  unit; differences are paired by slate.
* **Standard errors** cluster by NBA date (``MT.paired_cluster``), because players and slates of
  one night share games, injuries and noise. With one slate per date this is the same as
  clustering by slate. An SE is shown only from ``MIN_NIGHTS`` (5) nights; below that the report
  says "too few nights to tell", and a difference is only called a result at >= 2 SE.
* **Real and simulated are kept apart.** Real = uploaded standings (``dfs_contest_results``) and the
  owner's own entries (``dfs_entries``). Without an uploaded standings file there is no real cash
  line; the report then shows only simulated numbers (our modelled field, which reads high) and
  the random-lineup percentile, both labelled as proxies.
* **Live-like check.** The nightly numbers and the window are compared with the T2.4 bands in
  ``mlops/live_like_baseline.json`` (a Dec-Apr slice; October nights may run worse).

Entry points: `weekly_dfs_report`, `render_text`, `write_report`; CLI ``courtside proj
weekly-report`` (`proj/cli_proj.py`); page ``/dfs/weekly`` (`web/dfs_weekly.py`).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import metrics as MT

log = logging.getLogger("courtside.proj.reports")

VERSION = "weekly-v1"
MIN_NIGHTS = 5                      # an SE needs this many nights (same as the nightly grader)
Z_CALL = 2.0                        # a difference is a result only beyond this many SE
ROI_MIN_CONTESTS = 10               # no ROI until this many uploaded standings (HANDOFF §4)
KIND_PRIORITY = {"lock10": 0, "lock45": 1, "swap": 2, "manual": 3}
RUN_KINDS = ("lock45", "lock10", "swap", "manual")
SIMULATED_SOURCES = frozenset({"sim", "simulated", "proxy", "synthetic", "backtest"})
TIER_EDGES = (12.0, 20.0, 26.0, 32.0)          # proj/minutes.py TIER_EDGES
TIER_OF_INDEX = {0: "bench", 1: "bench", 2: "rotation", 3: "starter", 4: "star"}
TIER_ORDER = ("star", "starter", "rotation", "bench")
TIER_RANGE = {"star": "32+ min", "starter": "26-32 min", "rotation": "20-26 min",
              "bench": "under 20 min"}
SLATE_KINDS = ("main", "showdown")
RANGE_METRICS = ("dk_mae", "min_mae", "min_mae_played15", "cov80", "pplay_cal_err")
RANGE_LABEL = {"dk_mae": "DK points error", "min_mae": "Minutes error (all)",
               "min_mae_played15": "Minutes error (played, 15+)", "cov80": "80% range holds",
               "pplay_cal_err": "P(play) miss (uncertain players)"}
RANGE_MIN_N = {"dk_mae": 20, "min_mae": 20, "min_mae_played15": 15, "cov80": 20,
               "pplay_cal_err": 8}
HIGHER_IS_WORSE = frozenset({"dk_mae", "min_mae", "min_mae_played15"})
BANNED_WORDS = ("edge", "lean", "sleeper", "smash", "boom", "bust", "leverage", "fade", "chalk",
                "punt", "sharp", "pivot", "value play", "best bet")     # checked by the tests


# ============================================================================== small helpers

def _f(x):
    """JSON-safe float (None for NaN / inf / None)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _clean(o):
    """Recursively make `o` JSON-safe (numpy scalars, NaN, dates)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return _f(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return o


def season_first_day(day: date) -> date:
    """Start of the season window used by the grader's 'season' snapshots (1 Aug)."""
    y = day.year if day.month >= 8 else day.year - 1
    return date(y, 8, 1)


def tier_label(mean_played) -> pd.Series:
    """Tier name from E[min | plays] with the minutes model's tier edges (NaN -> None)."""
    m = pd.Series(mean_played, dtype=float)
    idx = np.searchsorted(np.asarray(TIER_EDGES), m.fillna(-1).to_numpy(), side="right")
    out = pd.Series([TIER_OF_INDEX[int(i)] for i in idx], index=m.index, dtype=object)
    out[m.isna()] = None
    return out


# ============================================================================== statistics

def cluster_stat(values, clusters) -> dict:
    """Mean of `values` with the cluster-robust SE (clusters = NBA dates).

    {mean, se, n, n_nights, se_ok}: ``se`` is the raw clustered SE (None below 2 clusters);
    ``se_ok`` says whether there are enough nights (>= MIN_NIGHTS) to show or use it."""
    v = np.asarray(values, float)
    cl = np.asarray(clusters)
    ok = np.isfinite(v)
    v, cl = v[ok], cl[ok]
    if not len(v):
        return {"mean": None, "se": None, "n": 0, "n_nights": 0, "se_ok": False}
    pc = MT.paired_cluster(v, cl)
    se = _f(pc["se"])
    g = int(pc["n_clusters"])
    return {"mean": _f(pc["mean"]), "se": se, "n": int(pc["n"]), "n_nights": g,
            "se_ok": bool(se is not None and g >= MIN_NIGHTS)}


def reading(adv: float | None, se: float | None, n_nights: int) -> tuple[str, float | None, str]:
    """(call, z, words) for an advantage of ours over a baseline (positive = we are ahead).

    call: 'too_few' (< MIN_NIGHTS nights), 'none' (under Z_CALL SE), 'ahead', 'behind'."""
    if adv is None or se is None or n_nights < MIN_NIGHTS:
        return "too_few", None, "too few nights to tell"
    if se <= 1e-12:
        return "none", None, "no clear difference"
    z = adv / se
    if abs(z) < Z_CALL:
        return "none", _f(z), "no clear difference"
    return ("ahead", _f(z), f"ahead ({z:+.1f} SE)") if z > 0 else \
        ("behind", _f(z), f"behind ({z:+.1f} SE)")


def compare(adv_values, clusters) -> dict:
    """Paired advantage of ours over a baseline: per-row `adv_values` (positive = ours better),
    clustered by date. Adds z, a call and the share of rows where ours won."""
    st = cluster_stat(adv_values, clusters)
    v = np.asarray(adv_values, float)
    v = v[np.isfinite(v)]
    call, z, words = reading(st["mean"], st["se"], st["n_nights"])
    st.update({"z": z, "call": call, "reading": words,
               "win_rate": _f((v > 0).mean()) if len(v) else None,
               "tie_rate": _f((v == 0).mean()) if len(v) else None})
    return st


# ============================================================================== data: players

def load_player_frame(conn, lo: date, hi: date) -> pd.DataFrame:
    """Graded projection rows of [lo, hi]: every run, every slate (deduplicated later per cut).
    Historical ``rg:`` validation slates and DK-FPPG fallback runs are left out."""
    if not _has_table(conn, "player_projection_grades"):
        return pd.DataFrame()
    q = (
        "SELECT g.projection_id, g.game_id, g.player_id, g.run_id, g.played, g.actual_min, "
        "g.actual_dk, g.proj_min, g.proj_dk, g.in80, g.in50, g.dk_fppg, g.salary, "
        "pp.nba_day AS day, pp.slate_id, pp.role, pp.model_version, pp.p_play, "
        "r.kind, r.created_at, s.tag AS slate_tag, sm.mean AS proj_min_played "
        "FROM player_projection_grades g JOIN player_projections pp ON pp.id=g.projection_id "
        "LEFT JOIN dfs_runs r ON r.run_id=g.run_id "
        "LEFT JOIN dfs_slates s ON s.slate_id=pp.slate_id "
        "LEFT JOIN player_projection_stats sm ON sm.projection_id=g.projection_id AND "
        "sm.stat='min' AND sm.cond='played' "
        "WHERE pp.nba_day BETWEEN ? AND ? AND COALESCE(pp.slate_id, '') NOT LIKE 'rg:%'")
    df = pd.read_sql_query(q, conn, params=(lo.isoformat(), hi.isoformat()))
    if df.empty:
        return df
    mv = df["model_version"].fillna("").str.lower()
    df = df[~(mv.str.startswith("fppg") | mv.str.contains("fallback"))].copy()
    df["kind"] = df["kind"].fillna("manual")
    df["prio"] = df["kind"].map(KIND_PRIORITY).fillna(2)
    df["slate_kind"] = df["slate_tag"].fillna("untagged")
    df["tier"] = tier_label(df["proj_min_played"])
    return df.reset_index(drop=True)


def dedupe(df: pd.DataFrame, manual: str = "fallback") -> pd.DataFrame:
    """One row per (game, player): best run kind (lock10 > lock45 > swap), newest run.

    Manual runs (hand-started builds, often earlier in the day and covering players the lock
    runs later dropped) are used only on nights that have no scheduled run at all
    (``manual='fallback'``), or always (``'keep'``)."""
    if df.empty:
        return df
    if manual == "fallback":
        sched_days = set(df.loc[df["kind"].isin(("lock10", "lock45", "swap")), "day"])
        df = df[(df["kind"] != "manual") | ~df["day"].isin(sched_days)]
    d = df.sort_values(["prio", "created_at"], ascending=[True, False], na_position="last")
    return d.drop_duplicates(["game_id", "player_id"]).reset_index(drop=True)


def add_salary_line(conn, df: pd.DataFrame) -> pd.DataFrame:
    """``sal_dk``: the salary-implied DK points (RotoGuru OLS line fit on earlier seasons)."""
    df = df.copy()
    df["sal_dk"] = np.nan
    if df.empty:
        return df
    from .dfs import grade as G
    cache: dict = {}
    for day in sorted(df["day"].unique()):
        a, b, _how = G.salary_line(conn, _as_date(day), cache)
        m = df["day"] == day
        df.loc[m, "sal_dk"] = a + b * df.loc[m, "salary"].astype(float)
    return df


# ============================================================================== projections

def projection_cut(df: pd.DataFrame) -> dict:
    """Projection numbers of one (already deduplicated) population."""
    u = df[(df["proj_min"] >= 10) & df["proj_dk"].notna() & df["actual_dk"].notna()] \
        if len(df) else df
    out: dict = {"n_players": int(len(u)),
                 "n_slates": int(u["slate_id"].nunique()) if len(u) else 0,
                 "n_nights": int(u["day"].nunique()) if len(u) else 0}
    if not len(u):
        return out
    cl = u["day"].to_numpy()
    y = u["actual_dk"].to_numpy(float)
    err = u["proj_dk"].to_numpy(float) - y
    out["dk_mae"] = cluster_stat(np.abs(err), cl)
    out["dk_bias"] = cluster_stat(err, cl)
    for name, col in (("fppg", "dk_fppg"), ("salary", "sal_dk")):
        if col not in u:
            continue
        m = u[col].notna().to_numpy()
        if not m.any():
            continue
        base_e = np.abs(u.loc[m, col].to_numpy(float) - y[m])
        ours_e = np.abs(err[m])
        cmp_ = compare(base_e - ours_e, cl[m])
        cmp_.update({"mae_ours": _f(ours_e.mean()), "mae_base": _f(base_e.mean()),
                     "gain_pct": _f(100.0 * cmp_["mean"] / base_e.mean())
                     if cmp_["mean"] is not None and base_e.mean() > 0 else None,
                     "n_slates": int(u.loc[m, "slate_id"].nunique())})
        out[f"vs_{name}"] = cmp_
        out[f"dk_mae_{name}"] = cluster_stat(base_e, cl[m])
    m = u["actual_min"].notna().to_numpy()
    if m.any():
        e = u.loc[m, "proj_min"].to_numpy(float) - u.loc[m, "actual_min"].to_numpy(float)
        out["min_mae"] = cluster_stat(np.abs(e), cl[m])
        out["min_bias"] = cluster_stat(e, cl[m])
    mp = u[(u["played"] == 1) & (u["proj_min_played"] >= 15) & u["actual_min"].notna()]
    if len(mp):
        out["min_mae_played15"] = cluster_stat(
            np.abs(mp["proj_min_played"].to_numpy(float) - mp["actual_min"].to_numpy(float)),
            mp["day"].to_numpy())
    c = u[u["in80"].notna()]
    if len(c):
        cv = cluster_stat(c["in80"].to_numpy(float), c["day"].to_numpy())
        if cv["se_ok"] and cv["se"] and cv["se"] > 0:
            cv["status"] = "inside" if abs(cv["mean"] - 0.8) <= Z_CALL * cv["se"] else (
                "too wide" if cv["mean"] > 0.8 else "too narrow")
        else:
            cv["status"] = "too few nights to tell"
        cv["target"] = 0.8
        out["cov80"] = cv
    return out


def projections_block(df: pd.DataFrame) -> dict:
    """All projection cuts for one window frame (with ``sal_dk``)."""
    head = dedupe(df)
    out: dict = {"headline": projection_cut(head), "by_slate_kind": {}, "by_run_kind": {},
                 "by_tier": {}, "by_version": {}}
    if df.empty:
        return out
    for k in sorted(df["slate_kind"].unique(), key=lambda s: (s not in SLATE_KINDS, s)):
        out["by_slate_kind"][k] = projection_cut(dedupe(df[df["slate_kind"] == k]))
    for k in RUN_KINDS:
        sub = df[df["kind"] == k]
        if len(sub):
            out["by_run_kind"][k] = projection_cut(dedupe(sub, "keep"))
    for t in TIER_ORDER:
        sub = head[head["tier"] == t]
        if len(sub):
            out["by_tier"][t] = projection_cut(sub)
    for v in sorted(head["model_version"].fillna("unknown").unique()):
        out["by_version"][v] = projection_cut(head[head["model_version"].fillna("unknown") == v])
    return out


# ============================================================================== data: lineups

def load_lineup_frame(conn, lo: date, hi: date) -> pd.DataFrame:
    if not (_has_table(conn, "dfs_lineup_grades") and _has_table(conn, "dfs_slates")):
        return pd.DataFrame()
    q = (
        "SELECT gl.lineup_id, gl.run_id, gl.slate_id, s.nba_day AS day, s.tag AS tag, "
        "gl.actual, gl.projected, gl.hindsight, gl.ratio_hindsight AS ratio, gl.pct_random, "
        "gl.would_cash, gl.cash_line_source AS src, "
        "json_extract(gl.detail_json, '$.kind') AS kind, "
        "json_extract(gl.detail_json, '$.mode') AS mode, "
        "json_extract(gl.detail_json, '$.effective') AS effective, "
        "json_extract(gl.detail_json, '$.via') AS via, "
        "json_extract(gl.detail_json, '$.role') AS role, "
        "json_extract(gl.detail_json, '$.game_type') AS gt, "
        "l.p_cash, l.p_top1, r.model_version "
        "FROM dfs_lineup_grades gl JOIN dfs_slates s ON s.slate_id=gl.slate_id "
        "LEFT JOIN dfs_lineups l ON l.id=gl.lineup_id LEFT JOIN dfs_runs r ON r.run_id=gl.run_id "
        "WHERE s.nba_day BETWEEN ? AND ? AND gl.slate_id NOT LIKE 'rg:%' "
        "AND gl.actual IS NOT NULL")
    df = pd.read_sql_query(q, conn, params=(lo.isoformat(), hi.isoformat()))
    if df.empty:
        return df
    df["effective"] = df["effective"].fillna(0).astype(int)
    df["slate_kind"] = df["tag"].fillna(df["gt"]).fillna("untagged")
    df.loc[df["slate_kind"] == "classic", "slate_kind"] = "main"
    return df


def load_baselines(conn, lo: date, hi: date) -> pd.DataFrame:
    """Per-slate baseline lineups (DK FPPG, salary alone), hindsight and the random summary."""
    if not _has_table(conn, "dfs_metric_snapshots"):
        return pd.DataFrame()
    q = ("SELECT substr(model, 7) AS slate_id, as_of AS day, "
         "json_extract(details_json, '$.hindsight.actual') AS hindsight, "
         "json_extract(details_json, '$.fppg.actual') AS fppg, "
         "json_extract(details_json, '$.fppg.ratio_hindsight') AS fppg_ratio, "
         "json_extract(details_json, '$.fppg.pct_random') AS fppg_pct, "
         "json_extract(details_json, '$.salary.actual') AS salary, "
         "json_extract(details_json, '$.salary.ratio_hindsight') AS salary_ratio, "
         "json_extract(details_json, '$.salary.pct_random') AS salary_pct, "
         "json_extract(details_json, '$.random.mean') AS random_mean "
         "FROM dfs_metric_snapshots WHERE model LIKE 'slate:%' AND metric='baselines' "
         "AND as_of BETWEEN ? AND ?")
    return pd.read_sql_query(q, conn, params=(lo.isoformat(), hi.isoformat()))


POPULATIONS = (("in_effect", "As built at lock (after any swap)"),
               ("lock45", "lock45 run"), ("lock10", "lock10 run"),
               ("swap", "Swapped lineups"))
GROUPS = (("cash", "cash lineup"), ("topn", "Top N lineups"))


def lineup_populations(L: pd.DataFrame):
    """Yield (population, group, frame) of our lineups for the cuts above."""
    if L.empty:
        return
    eff = L[L["effective"] == 1]
    for pop, _ in POPULATIONS:
        for grp, _g in GROUPS:
            if pop == "in_effect":
                sub = eff[eff["role"] == grp]
            elif pop == "swap":
                s = eff[eff["via"] == "swap"]
                sub = s[s["role"] == grp]
            else:
                k = L[L["kind"] == pop]
                sub = k[k["mode"] == grp]
            yield pop, grp, sub


def slate_units(sub: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """One row per slate: the mean of our lineups of this group, joined to the slate baselines."""
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["slate_id", "day", "slate_kind"], as_index=False).agg(
        actual=("actual", "mean"), ratio=("ratio", "mean"), pct=("pct_random", "mean"),
        n_lineups=("lineup_id", "size"), p_cash=("p_cash", "mean"), p_top1=("p_top1", "mean"))
    if len(base):
        g = g.merge(base.drop(columns=["day"]), on="slate_id", how="left")
    else:
        for c in ("hindsight", "fppg", "fppg_ratio", "fppg_pct", "salary", "salary_ratio",
                  "salary_pct", "random_mean"):
            g[c] = np.nan
    return g


def lineup_cut(units: pd.DataFrame) -> dict:
    """Ours vs the two baselines on points, share of hindsight and random percentile."""
    out: dict = {"n_slates": int(len(units)),
                 "n_nights": int(units["day"].nunique()) if len(units) else 0,
                 "n_lineups": int(units["n_lineups"].sum()) if len(units) else 0}
    if not len(units):
        return out
    cl = units["day"].to_numpy()
    for metric, ours, fcol, scol in (("points", "actual", "fppg", "salary"),
                                     ("ratio", "ratio", "fppg_ratio", "salary_ratio"),
                                     ("pct", "pct", "fppg_pct", "salary_pct")):
        m: dict = {"ours": cluster_stat(units[ours].to_numpy(float), cl)}
        for name, col in (("fppg", fcol), ("salary", scol)):
            ok = (units[ours].notna() & units[col].notna()).to_numpy()
            m[name] = cluster_stat(units.loc[ok, col].to_numpy(float), cl[ok])
            if ok.any():
                cmp_ = compare(units.loc[ok, ours].to_numpy(float)
                               - units.loc[ok, col].to_numpy(float), cl[ok])
                cmp_["ours_mean"] = _f(units.loc[ok, ours].mean())
                m["vs_" + name] = cmp_
        out[metric] = m
    out["hindsight"] = _f(units["hindsight"].mean())
    out["random_mean"] = _f(units["random_mean"].mean())
    out["sim_p_cash"] = _f(units["p_cash"].mean()) if units["p_cash"].notna().any() else None
    out["sim_p_top1"] = _f(units["p_top1"].mean()) if units["p_top1"].notna().any() else None
    return out


def lineups_block(conn, lo: date, hi: date) -> dict:
    L = load_lineup_frame(conn, lo, hi)
    base = load_baselines(conn, lo, hi)
    out: dict = {"cuts": {}, "n_lineups": int(len(L)),
                 "n_slates": int(L["slate_id"].nunique()) if len(L) else 0}
    if L.empty:
        return out
    kinds = sorted(L["slate_kind"].unique(), key=lambda s: (s not in SLATE_KINDS, s))
    for sk in kinds:
        Lk = L[L["slate_kind"] == sk]
        for pop, grp, sub in lineup_populations(Lk):
            if sub.empty:
                continue
            out["cuts"].setdefault(sk, {}).setdefault(pop, {})[grp] = \
                lineup_cut(slate_units(sub, base))
    return out


# ============================================================================== real vs simulated

def real_block(conn, lo: date, hi: date) -> dict:
    """Uploaded standings and the owner's own entries (real money). Nothing here is simulated."""
    out: dict = {"uploaded_contests_total": 0, "uploaded_contests_window": 0, "entries": None,
                 "cash_check": None, "roi_allowed": False}
    if _has_table(conn, "dfs_contest_results"):
        out["uploaded_contests_total"] = int(conn.execute(
            "SELECT COUNT(DISTINCT contest_id) FROM dfs_contest_results").fetchone()[0])
        if _has_table(conn, "dfs_contests"):
            out["uploaded_contests_window"] = int(conn.execute(
                "SELECT COUNT(DISTINCT cr.contest_id) FROM dfs_contest_results cr "
                "JOIN dfs_contests c ON c.contest_id=cr.contest_id "
                "JOIN dfs_slates s ON s.slate_id=c.slate_id WHERE s.nba_day BETWEEN ? AND ?",
                (lo.isoformat(), hi.isoformat())).fetchone()[0])
    out["roi_allowed"] = out["uploaded_contests_total"] >= ROI_MIN_CONTESTS
    if _has_table(conn, "dfs_entries"):
        rows = conn.execute(
            "SELECT e.entry_fee, e.payout, e.source, e.created_at, s.nba_day "
            "FROM dfs_entries e LEFT JOIN dfs_slates s ON s.slate_id=e.slate_id").fetchall()
        mine = []
        for r in rows:
            dy = r[4] or (r[3] or "")[:10]
            if dy and lo.isoformat() <= dy <= hi.isoformat() \
                    and (r[2] or "").lower() not in SIMULATED_SOURCES:
                mine.append(r)
        if mine:
            fees = sum(float(r[0] or 0.0) for r in mine)
            settled = [r for r in mine if r[1] is not None]
            paid = sum(float(r[1]) for r in settled)
            settled_fees = sum(float(r[0] or 0.0) for r in settled)
            out["entries"] = {"n": len(mine), "settled": len(settled),
                              "cashed": sum(1 for r in settled if float(r[1]) > 0),
                              "fees": fees, "winnings": paid,
                              "net": paid - settled_fees,
                              "roi": (paid - settled_fees) / settled_fees
                              if (out["roi_allowed"] and settled_fees > 0) else None}
    L = load_lineup_frame(conn, lo, hi)
    if len(L):
        r = L[(L["effective"] == 1) & (L["src"] == "real") & L["would_cash"].notna()]
        if len(r):
            out["cash_check"] = {"n_lineups": int(len(r)),
                                 "n_slates": int(r["slate_id"].nunique()),
                                 "would_cash": cluster_stat(r["would_cash"].to_numpy(float),
                                                            r["day"].to_numpy())}
    return out


# ============================================================================== live-like bands

def live_like_file() -> dict | None:
    try:
        from ..mlops import dfs_monitor as DM
        return DM.live_like_file()
    except Exception as e:  # noqa: BLE001
        log.warning("live-like file unavailable: %s", e)
        return None


def _spec_for(fl: dict, kind: str | None, version: str | None) -> dict:
    """The nightly p-ranges of the slice arm that matches a night's run kind and model."""
    if kind == "lock10" and isinstance((fl.get("lock10_reference") or {}).get("daily"), dict):
        return fl["lock10_reference"]["daily"]
    pm = fl.get("per_model") or {}
    if version:
        for key in (version, version.split("+")[0]):
            if key in pm and isinstance(pm[key].get("daily"), dict):
                return pm[key]["daily"]
    return fl.get("daily") or {}


def _values(conn, rows: pd.DataFrame) -> tuple[dict, int]:
    """({metric: (value, n)}, rows) of a set of graded rows, the grader's definitions."""
    from ..mlops import dfs_monitor as DM
    vals, _mv = DM.nightly_values(conn, rows)
    u = rows[(rows["proj_min"] >= 10) & rows["proj_dk"].notna() & rows["actual_dk"].notna()
             & rows["actual_min"].notna()]
    if len(u):
        vals["min_mae"] = (float((u["proj_min"] - u["actual_min"]).abs().mean()), len(u))
    return vals, int(len(rows))


def _status(metric: str, v: float, lo: float, hi: float) -> str:
    if v > hi:
        return "above"
    if v < lo:
        return "below"
    return "inside"


def _meaning(metric: str, status: str) -> str:
    if status == "inside":
        return "inside the live-like range"
    if metric in HIGHER_IS_WORSE:
        return "above the range (worse)" if status == "above" else "below the range (better)"
    if metric == "cov80":
        return "above the range (ranges too wide)" if status == "above" \
            else "below the range (ranges too narrow)"
    return f"{status} the range"


def _band(s: dict, n: int, k_nights: int) -> dict | None:
    """Expected range of a k-night average with `n` rows in total: the slice's mean plus or minus
    two SD of one night, widened by sqrt(rows per slice night / rows per night here) when the
    nights here are smaller than a slice night, and divided by sqrt(k)."""
    try:
        mu, sd = float(s["mean"]), float(s["sd"])
        ref = float(s.get("rows_median") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    per_night = n / max(k_nights, 1)
    widen = math.sqrt(max(1.0, ref / per_night)) if ref and per_night > 0 else 1.0
    sd_w = sd * widen / math.sqrt(max(k_nights, 1))
    return {"slice_mean": mu, "lo": mu - Z_CALL * sd_w, "hi": mu + Z_CALL * sd_w,
            "widen": widen}


def _judge(metric: str, v: float, n: int, spec: dict, k_nights: int) -> dict:
    if n < RANGE_MIN_N[metric]:
        return {"value": _f(v), "n": n, "status": "too few rows"}
    b = _band(spec, n, k_nights)
    if b is None:
        return {"value": _f(v), "n": n, "status": "no band"}
    st = _status(metric, v, b["lo"], b["hi"])
    return {"value": _f(v), "n": n, "slice_mean": _f(b["slice_mean"]), "lo": _f(b["lo"]),
            "hi": _f(b["hi"]), "status": st, "meaning": _meaning(metric, st),
            "nights_averaged": k_nights}


def live_like_block(conn, head: pd.DataFrame) -> dict:
    """Nightly values and the window against the T2.4 live-like bands.

    One rule for a night and for the window: inside when the value is within the slice mean plus
    or minus two SD of an average of the same number of nights (see `_band`)."""
    fl = live_like_file()
    out: dict = {"available": bool(fl), "nights": [], "window": {}, "note": None}
    if not fl or head.empty:
        return out
    out["note"] = fl.get("note")
    out["slice"] = {"dates": fl.get("slice"), "n_dates": fl.get("n_dates"),
                    "model": fl.get("model")}
    spec_win: dict = {}
    k_nights = head["day"].nunique()
    for day, rows in head.groupby("day"):
        kind = rows["kind"].mode().iloc[0] if len(rows) else None
        version = rows["model_version"].mode().iloc[0] if rows["model_version"].notna().any() \
            else None
        spec = _spec_for(fl, kind, version)
        spec_win = spec_win or spec
        vals, nrows = _values(conn, rows)
        night = {"day": str(day), "rows": vals["dk_mae"][1] if "dk_mae" in vals else nrows,
                 "kind": kind, "model_version": version,
                 "metrics": {m: _judge(m, *vals[m], spec[m], 1)
                             for m in RANGE_METRICS if m in vals and m in spec}}
        out["nights"].append(night)
    vals, _n_rows = _values(conn, head)
    for m in RANGE_METRICS:
        if m in vals and m in spec_win:
            out["window"][m] = _judge(m, *vals[m], spec_win[m], k_nights)
    return out


# ============================================================================== the report

def _window_block(conn, lo: date, hi: date, label: str, with_live: bool = True) -> dict:
    P = load_player_frame(conn, lo, hi)
    if len(P):
        P = add_salary_line(conn, P)
    head = dedupe(P) if len(P) else P
    proj = projections_block(P) if len(P) else projections_block(pd.DataFrame())
    lin = lineups_block(conn, lo, hi)
    nights = sorted({str(d) for d in head["day"].unique()}) if len(head) else []
    lin_nights = set()
    if lin["cuts"]:
        L = load_lineup_frame(conn, lo, hi)
        lin_nights = {str(d) for d in L["day"].unique()}
    versions_seen = {}
    if len(P):
        for v, g in P.groupby(P["model_version"].fillna("unknown")):
            versions_seen[str(v)] = {"rows": int(len(g)),
                                     "kinds": sorted(g["kind"].unique().tolist())}
    return {"label": label, "from": lo.isoformat(), "to": hi.isoformat(),
            "nights": nights, "n_nights": len(nights),
            "lineup_nights": sorted(lin_nights),
            "projections": proj, "lineups": lin, "real": real_block(conn, lo, hi),
            "live_like": live_like_block(conn, head) if with_live else {"available": False},
            "versions_seen": versions_seen,
            "versions_scored": sorted({str(v) for v in head["model_version"].fillna("unknown")})
            if len(head) else []}


def weekly_dfs_report(conn: sqlite3.Connection, week_end, days: int = 7) -> dict:
    """The weekly DFS report for the `days` NBA days ending at `week_end` (inclusive), plus the
    season so far. Returns a JSON-safe dict: raw numbers under ``window`` and ``season``, and
    the text/HTML-ready ``sections`` built from them."""
    end = _as_date(week_end)
    lo = end - timedelta(days=max(int(days), 1) - 1)
    s0 = season_first_day(end)
    win = _window_block(conn, lo, end, f"Last {days} days")
    season = win if lo <= s0 else _window_block(conn, s0, end, "Season so far")
    same = season is win or (season["nights"] == win["nights"]
                             and season["lineup_nights"] == win["lineup_nights"])
    rep = {"version": VERSION, "week_end": end.isoformat(), "days": int(days),
           "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "thresholds": {"min_nights": MIN_NIGHTS, "z_call": Z_CALL,
                          "roi_min_contests": ROI_MIN_CONTESTS},
           "window": win, "season": season, "season_same_as_window": same}
    rep["sections"] = build_sections(rep)
    rep["headline"] = headline_lines(rep)
    return _clean(rep)


# ============================================================================== presentation

def _n(x, d=2) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _sg(x, d=2) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}"


def _pct(x, d=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _se_txt(st: dict | None, d=2, pct=False) -> str:
    """'mean +- se' when the SE is usable, else the mean with the number of nights."""
    if not st or st.get("mean") is None:
        return "n/a"
    f = (lambda v: _pct(v)) if pct else (lambda v: _n(v, d))
    if st.get("se_ok"):
        return f"{f(st['mean'])} +-{(100 * st['se']):.1f}" if pct else \
            f"{f(st['mean'])} +-{st['se']:.{d}f}"
    return f"{f(st['mean'])} (SE n/a)"


def _adv_txt(cmp_: dict | None, d=2) -> str:
    if not cmp_ or cmp_.get("mean") is None:
        return "n/a"
    if cmp_.get("se_ok"):
        return f"{_sg(cmp_['mean'], d)} +-{cmp_['se']:.{d}f}"
    return f"{_sg(cmp_['mean'], d)} (SE n/a)"


def _sample_words(n_nights: int) -> str:
    return f"{n_nights} night{'s' if n_nights != 1 else ''}"


def _proj_row(label: str, c: dict) -> list[str]:
    if not c.get("n_players"):
        return [label, "0", "0", "0"] + ["n/a"] * 8
    base = lambda k: c.get(k, {}).get("mae_base")          # noqa: E731
    vf, vs = c.get("vs_fppg"), c.get("vs_salary")
    return [label, f"{c['n_players']:,}", str(c["n_slates"]), str(c["n_nights"]),
            _n(c["dk_mae"]["mean"]),
            _n(base("vs_fppg")) if vf else "n/a",
            _n(base("vs_salary")) if vs else "n/a",
            _adv_txt(vf), _adv_txt(vs),
            _n(c.get("min_mae", {}).get("mean")), _sg(c.get("min_bias", {}).get("mean")),
            _pct(c.get("cov80", {}).get("mean"))]


PROJ_COLS = ["Group", "Players", "Slates", "Nights", "DK points error (ours)", "DK FPPG",
             "Salary only", "FPPG error minus ours", "Salary error minus ours",
             "Minutes error", "Minutes bias", "80% range holds"]


def _proj_lines(c: dict, name: str, ref: dict | None = None) -> list[str]:
    """Reading lines of a cut; a cut whose readings equal the reference cut's adds nothing."""
    out = []
    if not c.get("n_players"):
        return out
    for key, who in (("vs_fppg", "DraftKings FPPG"), ("vs_salary", "salary alone")):
        if key in c and (ref is None or key not in ref or ref[key]["reading"] != c[key]["reading"]):
            out.append(f"{name}: against {who}: {c[key]['reading']}.")
    return out


def _fmt_val(metric: str, v) -> str:
    if v is None:
        return "n/a"
    if metric == "cov80":
        return f"{100 * v:.1f}%"
    if metric == "pplay_cal_err":
        return f"{v:+.3f}"
    return f"{v:.2f}"


def _fmt_rng(metric: str, m: dict) -> str:
    if m.get("lo") is None:
        return "n/a"
    if metric == "cov80":
        return f"{100 * m['lo']:.0f}% to {100 * m['hi']:.0f}%"
    if metric == "pplay_cal_err":
        return f"{m['lo']:+.2f} to {m['hi']:+.2f}"
    return f"{m['lo']:.2f} to {m['hi']:.2f}"


def _section_projections(blk: dict, title: str, sid: str, full: bool) -> dict:
    p = blk["projections"]
    h = p["headline"]
    sec = {"id": sid, "title": title, "lede": "", "tables": [], "lines": []}
    if not h.get("n_players"):
        sec["lede"] = "No graded projections in this window."
        return sec
    sec["lede"] = ("Every projected player of 10+ minutes counts and a player who did not play "
                   "scores zero. Error is the average miss in DraftKings points (lower is "
                   "better). 'FPPG error minus ours' is how much closer we are than the "
                   "DraftKings season average (positive = we are closer). Plus-minus figures "
                   "are standard errors across nights and appear from 5 nights. The DK FPPG and "
                   "salary columns are scored on the players that have that baseline, and the "
                   "difference columns use exactly those players. Manual runs are left out "
                   "of the headline numbers on nights that have scheduled runs (the /health "
                   "DFS tab keeps them, so its figures can differ).")
    rows = [_proj_row("All slates (lock10, else lock45, else swap)", h)]
    for k, c in p["by_slate_kind"].items():
        rows.append(_proj_row(f"{k} slates", c))
    for k, c in p["by_run_kind"].items():
        lab = {"lock45": "lock45 runs", "lock10": "lock10 runs", "swap": "swap runs",
               "manual": "manual runs (not scheduled)"}[k]
        rows.append(_proj_row(lab, c))
    sec["tables"].append({"id": sid + "-cuts", "title": "By slate kind and run kind",
                          "columns": PROJ_COLS, "rows": rows})
    sec["lines"] += _proj_lines(h, "All slates")
    for k, c in p["by_slate_kind"].items():
        sec["lines"] += _proj_lines(c, f"{k} slates", h)
    if p["by_tier"]:
        trows = []
        for t, c in p["by_tier"].items():
            r = _proj_row(f"{t} ({TIER_RANGE[t]})", c)
            trows.append([r[0], r[1], r[4], r[5], r[7], r[9], r[10], r[11]])
        sec["tables"].append({
            "id": sid + "-tiers", "title": "By tier (projected minutes when playing)",
            "columns": ["Tier", "Players", "DK points error (ours)", "DK FPPG",
                        "FPPG error minus ours", "Minutes error", "Minutes bias",
                        "80% range holds"], "rows": trows})
    if len(p["by_version"]) >= 1 and full:
        vrows = []
        for v, c in p["by_version"].items():
            r = _proj_row(v, c)
            vrows.append([r[0], r[1], r[3], r[4], r[5], r[7], r[11]])
        sec["tables"].append({
            "id": sid + "-versions", "title": "By model version",
            "columns": ["Version", "Players", "Nights", "DK points error (ours)", "DK FPPG",
                        "FPPG error minus ours", "80% range holds"], "rows": vrows})
    cv = h.get("cov80")
    if cv:
        sec["lines"].append(f"80% ranges held {_pct(cv['mean'])} of the time "
                            f"(target 80%): {cv['status']}.")
    return sec


def _section_live(blk: dict, sid: str = "live") -> dict:
    ll = blk["live_like"]
    sec = {"id": sid, "title": "Against the live-like bands (T2.4)", "lede": "", "tables": [],
           "lines": []}
    if not ll.get("available"):
        sec["lede"] = "The live-like baseline file was not found, so there is nothing to compare."
        return sec
    if not ll["nights"]:
        sec["lede"] = "No graded night in this window."
        return sec
    sl = ll.get("slice") or {}
    d = sl.get("dates") or ["?", "?"]
    sec["lede"] = (f"The bands come from replaying {d[0]} to {d[1]} ({sl.get('n_dates')} nights) "
                   "the way the live job runs. That is December to April; "
                   "October nights run with less data on every player, so being a little "
                   "worse than the band early in the season is expected. A night, and the "
                   "window as a whole, is 'inside' when it is within two standard "
                   "deviations of the slice average for an average of that many nights "
                   "(wider when there are fewer players than a typical slice night). About "
                   "1 time in 20 lands outside by chance.")
    w = ll["window"]
    wrows = []
    for m in RANGE_METRICS:
        x = w.get(m)
        if not x:
            continue
        wrows.append([RANGE_LABEL[m], str(x["n"]), _fmt_val(m, x["value"]),
                      _fmt_val(m, x.get("slice_mean")), _fmt_rng(m, x) if x.get("lo") is not None
                      else "n/a", x.get("meaning", x["status"])])
        if m in ("dk_mae", "min_mae_played15", "cov80") and x.get("meaning"):
            sec["lines"].append(f"{RANGE_LABEL[m]} {_fmt_val(m, x['value'])}: "
                                f"{x['meaning']}.")
    sec["tables"].append({"id": sid + "-window", "title": "The window as a whole",
                          "columns": ["Measure", "Rows", "Ours", "Slice average",
                                      "Expected range for this many nights", "Reading"],
                          "rows": wrows})
    nrows = []
    for n in ll["nights"]:
        r = [n["day"], str(n["rows"])]
        for m in ("dk_mae", "min_mae_played15", "cov80", "pplay_cal_err"):
            x = n["metrics"].get(m)
            if not x:
                r.append("n/a")
            elif x["status"] == "too few rows":
                r.append(f"{_fmt_val(m, x['value'])} (too few rows)")
            else:
                r.append(f"{_fmt_val(m, x['value'])}: {x['status']}")
        nrows.append(r)
    sec["tables"].append({"id": sid + "-nights", "title": "Night by night",
                          "columns": ["Night", "Rows", "DK points error",
                                      "Minutes error (played, 15+)", "80% range holds",
                                      "P(play) miss"], "rows": nrows,
                          "note": "each night against the same rule"})
    return sec


def _pp(v: dict | None) -> str:
    """Percentile-point advantage (a difference of two fractions) as 'x.x pp'."""
    if not v or v.get("mean") is None:
        return "n/a"
    x = 100 * v["mean"]
    t = "0.0" if abs(x) < 0.05 else f"{x:+.1f}"
    return f"{t} pp" + (f" +-{100 * v['se']:.1f}" if v.get("se_ok") else " (SE n/a)")


def _lineup_rows(blk: dict, metric: str, d: int, pct=False,
                 pops_only: tuple = ("in_effect",)) -> list[list[str]]:
    rows = []
    f = (lambda x: "n/a" if x is None else f"{100 * x:.1f}%") if pct else \
        (lambda x: _n(x, d))
    for sk, pops in blk["lineups"]["cuts"].items():
        for pop, _plabel in POPULATIONS:
            if pop not in pops_only:
                continue
            for grp, glabel in GROUPS:
                c = pops.get(pop, {}).get(grp)
                if not c or metric not in c:
                    continue
                m = c[metric]
                vf, vs = m.get("vs_fppg"), m.get("vs_salary")
                advf, advs = (_pp(vf), _pp(vs)) if pct else (_adv_txt(vf, d), _adv_txt(vs, d))
                rd = vf["reading"] if vf else "n/a"
                rs = vs["reading"] if vs else "n/a"
                lab = f"{sk}: {glabel}" if pops_only == ("in_effect",) else \
                    f"{sk}: {glabel}, {dict(POPULATIONS)[pop]}"
                rows.append([lab, str(c["n_slates"]), str(c["n_nights"]),
                             f(m["ours"]["mean"]), f(m["fppg"]["mean"]), f(m["salary"]["mean"]),
                             advf, advs, rd if rd == rs else f"FPPG: {rd}; salary: {rs}"])
    return rows


LINEUP_COLS = ["Lineups", "Slates", "Nights", "Ours", "DK FPPG lineup", "Salary-only lineup",
               "Ours minus FPPG", "Ours minus salary", "Reading"]


def _by_run_rows(blk: dict) -> list[list[str]]:
    rows = []
    for sk, pops in blk["lineups"]["cuts"].items():
        for pop, plabel in POPULATIONS:
            if pop == "in_effect":
                continue
            for grp, glabel in GROUPS:
                c = pops.get(pop, {}).get(grp)
                if not c or "points" not in c:
                    continue
                pt, ra, pc = c["points"], c["ratio"], c["pct"]
                vf, vs = pt.get("vs_fppg"), pt.get("vs_salary")
                rd = vf["reading"] if vf else "n/a"
                rs = vs["reading"] if vs else "n/a"
                rows.append([f"{sk}: {glabel}, {plabel}", str(c["n_slates"]),
                             _n(pt["ours"]["mean"], 1), _n(pt["fppg"]["mean"], 1),
                             _n(pt["salary"]["mean"], 1), _adv_txt(vf, 1), _adv_txt(vs, 1),
                             _n(ra["ours"]["mean"], 3),
                             "n/a" if pc["ours"]["mean"] is None
                             else f"{100 * pc['ours']['mean']:.1f}%",
                             rd if rd == rs else f"FPPG: {rd}; salary: {rs}"])
    return rows


def _section_lineups(blk: dict, title: str, sid: str) -> dict:
    sec = {"id": sid, "title": title, "lede": "", "tables": [], "lines": []}
    lin = blk["lineups"]
    if not lin["cuts"]:
        sec["lede"] = "No graded lineups in this window."
        return sec
    sec["lede"] = ("PAPER RESULTS: lineups we built, scored on the real box scores after the "
                   "games. Nobody entered them. Each slate is one data point (for Top N, the "
                   "mean of the N lineups) against the lineups DraftKings' own FPPG and salary "
                   "alone would have built from the same pool at the same time. 'Share of "
                   "best' is our score over the best possible lineup of the slate. Random "
                   "percentile is where the score sits among 5,000 random legal lineups: "
                   "it is a proxy, because a random field is far weaker than a real DK field. "
                   "The tables show the lineups in effect at lock (the lock10 build, with any "
                   "swap applied); the last table splits them by run kind.")
    for metric, ttl, d, pct in (("points", "Actual DK points", 1, False),
                                ("ratio", "Share of the best possible lineup", 3, False),
                                ("pct", "Random-lineup percentile (proxy for the field)", 1,
                                 True)):
        rows = _lineup_rows(blk, metric, d, pct)
        if rows:
            sec["tables"].append({"id": f"{sid}-{metric}", "title": ttl,
                                  "columns": LINEUP_COLS, "rows": rows})
    rr = _by_run_rows(blk)
    if rr:
        sec["tables"].append({
            "id": sid + "-runs", "title": "By run kind (points, share of best, random percentile)",
            "columns": ["Lineups", "Slates", "Ours (points)", "DK FPPG lineup",
                        "Salary-only lineup", "Ours minus FPPG", "Ours minus salary",
                        "Ours (share of best)", "Ours (random pct)", "Reading (points)"],
            "rows": rr})
    for sk, pops in lin["cuts"].items():
        c = pops.get("in_effect", {}).get("cash") or pops.get("in_effect", {}).get("topn")
        if c and c.get("hindsight") is not None:
            sec["lines"].append(
                f"{sk}: the best possible lineup averaged {c['hindsight']:.1f} points and a "
                f"random lineup {_n(c.get('random_mean'), 1)}.")
    return sec


def _section_real(blk: dict, sid: str = "real") -> dict:
    r = blk["real"]
    sec = {"id": sid, "title": "Real contests and simulated numbers, kept apart", "lede": "",
           "tables": [], "lines": []}
    sec["lede"] = ("REAL means DraftKings standings you uploaded and entries you actually "
                   "played. SIMULATED means our own model's field. The two are never "
                   "combined.")
    rows = []
    cc = r.get("cash_check")
    if cc:
        w = cc["would_cash"]
        rows.append(["Real: our lineups against an uploaded cash line",
                     f"{cc['n_lineups']} lineups, {cc['n_slates']} slates",
                     f"{_pct(w['mean'])} would have cashed" + (f" +-{100 * w['se']:.1f}"
                                                              if w.get("se_ok") else "")])
    else:
        rows.append(["Real: our lineups against an uploaded cash line", "none",
                     "No standings uploaded for these slates, so there is no real cash line. "
                     "Nothing about cashing is claimed."])
    e = r.get("entries")
    if e:
        roi = ("ROI hidden until %d uploaded standings (%d so far)"
               % (ROI_MIN_CONTESTS, r["uploaded_contests_total"])) if e["roi"] is None \
            else f"ROI {100 * e['roi']:+.0f}%"
        rows.append(["Real: your own entries",
                     f"{e['n']} entries, {e['settled']} settled, {e['cashed']} cashed",
                     f"fees ${e['fees']:.2f}, winnings ${e['winnings']:.2f}, "
                     f"net ${e['net']:+.2f}; {roi}"])
    else:
        rows.append(["Real: your own entries", "none",
                     "No entries recorded in this window."])
    rows.append(["Real: uploaded standings",
                 f"{r['uploaded_contests_total']} in total, "
                 f"{r['uploaded_contests_window']} for this window", ""])
    for sk, pops in blk["lineups"]["cuts"].items():
        for grp, glabel in GROUPS:
            c = (pops.get("lock10", {}).get(grp) or pops.get("lock45", {}).get(grp))
            if c and c.get("sim_p_cash") is not None:
                rows.append([f"SIMULATED: {sk} {glabel} (as first built)",
                             f"{c['n_slates']} slates",
                             f"{100 * c['sim_p_cash']:.0f}% to cash and "
                             f"{100 * (c.get('sim_p_top1') or 0):.1f}% to finish top 1%, in our "
                             "own modelled field (simulated proxy; reads high)"])
    sec["tables"].append({"id": sid + "-table", "title": "Real vs simulated",
                          "columns": ["Source", "Sample", "What it says"], "rows": rows})
    return sec


def _section_versions(rep: dict) -> dict:
    win = rep["window"]
    sec = {"id": "versions", "title": "Model versions in this window", "lede": "", "tables": [],
           "lines": []}
    seen = win["versions_seen"]
    scored = win["versions_scored"]
    if not seen:
        sec["lede"] = "No graded runs in this window."
        return sec
    rows = [[v, str(x["rows"]), ", ".join(x["kinds"]), "yes" if v in scored else "no"]
            for v, x in sorted(seen.items())]
    sec["tables"].append({"id": "versions-table", "title": "Versions that produced graded rows",
                          "columns": ["Version", "Graded rows (all runs)", "Run kinds",
                                      "Counted in the headline numbers"], "rows": rows})
    if len(scored) > 1:
        sec["lines"].append(f"This window mixes {len(scored)} model versions "
                            f"({', '.join(scored)}). The headline numbers blend them; the "
                            "per-version table in the projections section shows each one.")
    elif scored:
        sec["lines"].append(f"One model version is scored in this window: {scored[0]}.")
    others = [v for v in seen if v not in scored]
    if others:
        sec["lines"].append(f"Only in extra runs (manual or swap builds), not in the headline "
                            f"numbers: {', '.join(others)}.")
    return sec


def build_sections(rep: dict) -> list[dict]:
    win, sea = rep["window"], rep["season"]
    secs = [(_section_projections(win, "Projections: this window", "proj", True), "Projections"),
            (_section_lineups(win, "Lineups: this window (paper results)", "lineups"),
             "Lineups"),
            (_section_live(win), "Live-like bands"),
            (_section_real(win), "Real vs simulated"),
            (_section_versions(rep), "Versions")]
    if not rep["season_same_as_window"]:
        secs.append((_section_projections(sea, "Projections: season so far", "proj-season",
                                          True), "Projections, season"))
        secs.append((_section_lineups(sea, "Lineups: season so far (paper results)",
                                      "lineups-season"), "Lineups, season"))
    out = []
    for sec, nav in secs:
        sec["nav"] = nav
        out.append(sec)
    return out


def headline_lines(rep: dict) -> list[str]:
    win = rep["window"]
    out: list[str] = []
    n = win["n_nights"]
    lo, hi = win["from"], win["to"]
    if n == 0:
        return [f"No graded slates between {lo} and {hi}. Nothing to report yet."]
    nl = win["nights"]
    out.append(f"{_sample_words(n)} graded ({nl[0]}" + (f" to {nl[-1]}" if n > 1 else "")
               + f"; window {lo} to {hi}).")
    if n < MIN_NIGHTS:
        out.append(f"Too few nights to tell: a standard error needs at least {MIN_NIGHTS}. "
                   "Read every number below as a description of those nights, not a result.")
    h = win["projections"]["headline"]
    if h.get("n_players"):
        vf, vs = h.get("vs_fppg"), h.get("vs_salary")
        s = (f"Projections ({h['n_players']:,} players, {h['n_slates']} slates): DK points "
             f"error {h['dk_mae']['mean']:.2f}")
        for c, who in ((vf, "DK FPPG"), (vs, "salary alone")):
            if c:
                s += (f"; on the {c['n']:,} players with a {who} number, ours "
                      f"{c['mae_ours']:.2f} vs {c['mae_base']:.2f}")
        out.append(s + ".")
    results = []
    for key, who in (("vs_fppg", "DraftKings FPPG"), ("vs_salary", "salary alone")):
        c = h.get(key)
        if c and c["call"] in ("ahead", "behind"):
            results.append(f"Projections are {c['call']} of {who} by {abs(c['mean']):.2f} "
                           f"DK points of error ({c['z']:+.1f} SE, {c['n_nights']} nights).")
    for sk, pops in win["lineups"]["cuts"].items():
        for grp, glabel in GROUPS:
            c = pops.get("in_effect", {}).get(grp)
            if not c:
                continue
            for name, who in (("vs_fppg", "FPPG"), ("vs_salary", "salary-only")):
                v = c["points"].get(name)
                if v and v["call"] in ("ahead", "behind"):
                    results.append(f"{sk} {glabel.lower()}: {v['call']} of the {who} lineup by "
                                   f"{abs(v['mean']):.1f} points ({v['z']:+.1f} SE).")
    if results:
        out += results
    else:
        out.append("No difference reaches 2 standard errors, so there is no result to report "
                   "yet.")
    for sk, pops in win["lineups"]["cuts"].items():
        c = pops.get("in_effect", {}).get("cash")
        if c:
            pts = c["points"]
            out.append(f"Cash lineup, {sk} ({c['n_slates']} slates): {_n(pts['ours']['mean'], 1)} "
                       f"points vs {_n(pts['fppg']['mean'], 1)} for the FPPG lineup and "
                       f"{_n(pts['salary']['mean'], 1)} for salary alone (paper results).")
    ll = win["live_like"]
    if ll.get("available") and ll["window"]:
        bits = []
        for m in ("dk_mae", "min_mae_played15", "cov80"):
            x = ll["window"].get(m)
            if x and x.get("meaning"):
                bits.append(f"{RANGE_LABEL[m]} {_fmt_val(m, x['value'])} "
                            f"({x['meaning']})")
        if bits:
            out.append("Against the live-like bands: " + "; ".join(bits) + ".")
    if len(win["versions_scored"]) > 1:
        out.append(f"The window mixes model versions ({', '.join(win['versions_scored'])}).")
    r = win["real"]
    if r.get("cash_check"):
        out.append(f"Real cash lines exist for {r['cash_check']['n_slates']} slates "
                   "(uploaded standings).")
    else:
        out.append("No standings uploaded for these slates: nothing about real cash lines yet; "
                   "lineup numbers are paper results and proxies.")
    return out


# ============================================================================== text and files

def _table_text(t: dict) -> list[str]:
    cols = [str(c) for c in t["columns"]]
    rows = [[str(c) for c in r] for r in t["rows"]]
    w = [min(120, max(len(cols[i]), *(len(r[i]) for r in rows))) if rows else len(cols[i])
         for i in range(len(cols))]
    right = [i > 0 and bool(rows) and all(_numlike(r[i]) for r in rows) for i in range(len(cols))]

    def fmt(r):
        cells = []
        for i, c in enumerate(r):
            c = c if len(c) <= w[i] else c[: w[i] - 1] + "~"
            cells.append(c.rjust(w[i]) if right[i] else c.ljust(w[i]))
        return "  " + "  ".join(cells).rstrip()
    out = [t["title"] + (f" ({t['note']})" if t.get("note") else "")]
    out.append(fmt(cols))
    out.append("  " + "  ".join("-" * x for x in w))
    out += [fmt(r) for r in rows]
    return out


def _numlike(s: str) -> bool:
    return bool(s) and (s[0].isdigit() or s[0] in "+-") and len(s) < 26 and " " not in s[:1]


def render_text(rep: dict) -> str:
    """The report as readable plain text."""
    L: list[str] = []
    L.append(f"COURTSIDE WEEKLY DFS REPORT   week ending {rep['week_end']} "
             f"({rep['window']['from']} to {rep['window']['to']}); generated {rep['generated']}")
    L.append("")
    for ln in rep["headline"]:
        L += textwrap.wrap("* " + ln, 100, subsequent_indent="  ")
    for sec in rep["sections"]:
        L += ["", sec["title"].upper(), "=" * len(sec["title"])]
        if sec["lede"]:
            L += textwrap.wrap(sec["lede"], 100)
        for t in sec["tables"]:
            L += [""] + _table_text(t)
        if sec["lines"]:
            L.append("")
            for ln in sec["lines"]:
                L += textwrap.wrap("- " + ln, 100, subsequent_indent="  ")
    return "\n".join(L) + "\n"


def write_report(rep: dict, out_dir: Path | None = None) -> Path:
    """Write ``weekly_dfs_<week_end>.json`` (default ``<data>/reports``). Returns the path."""
    out_dir = Path(out_dir) if out_dir else config.data_dir() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"weekly_dfs_{rep['week_end']}.json"
    p.write_text(json.dumps(rep, indent=1, default=str))
    return p


def latest_graded_day(conn) -> date | None:
    """Newest NBA day with graded player rows (None when nothing is graded)."""
    if not _has_table(conn, "player_projection_grades"):
        return None
    r = conn.execute("SELECT MAX(pp.nba_day) FROM player_projection_grades g "
                     "JOIN player_projections pp ON pp.id=g.projection_id "
                     "WHERE COALESCE(pp.slate_id, '') NOT LIKE 'rg:%'").fetchone()
    return date.fromisoformat(r[0][:10]) if r and r[0] else None
