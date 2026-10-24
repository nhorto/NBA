"""E5: the per-slate real-standings calibration report (docs/IMPROVEMENTS.md E5).

Report-only: it reads the owner's real entries (``dfs_entries``), uploaded standings
(``dfs_contest_results`` / ``dfs_contest_ownership``), the contest-history uploads
(``dfs_uploads``) and the nightly grades, and writes one JSON file
(``<data>/reports/real_calib/real_calib.json``). Nothing it computes feeds a lineup, a
projection or a stored number.

The unit is the SLATE: every lineup of a night shares one draw of the games, so lineups and
entries are averaged within a slate first and slates are the independent units of every SE.

Per Classic slate with real entries or uploaded standings:

* **Contests.** Type (cash = double-up / 50-50, small or large GPP), field, places paid (payout
  table, standings, or the contest history's Places_Paid), and the real cash line: exact from an
  uploaded standings file, else BRACKETED by the owner's own entries (the best entry that did
  not cash and the worst that did: the line lies in (lo, hi]); the top-1 % line likewise.
* **Entries.** Points, place, the share of the field beaten (1 - (place - 0.5) / field), in the
  money or not, and the lineup it was (``lineup_id``, else the player set, else the one stored
  lineup of the slate that scored exactly those points).
* **Simulated vs real (``sims`` > 0).** Each entered lineup against a modelled field of that
  contest, built and scored as the contest builder does (`field.field_for`, calibrated truth
  `portfolio.truth_sims`): simulated P(cash), the PIT of (score - real cash line) within the
  lineup's simulated distribution of (score - simulated cash line) (an interval when the line is
  bracketed) and the PIT of the real finishing place among the simulated places. Calibrated
  chances put both PITs uniform on [0, 1] (mean 0.5) over many slates; our chances reading high
  shows up as PITs below 0.5. DU P(cash) by the line PIT detects a 0.10 shift in about 33 slates
  (a plain cash rate needs about 45).
* **Ownership** (uploaded standings only): proxy vs real %Drafted per contest
  (`field.calibration_report`: MAE, rank correlation, top-10 bias), grouped by contest type and
  slate size (2-4 games vs 5+).
* **Nightly, no standings needed:** our in-effect cash lineup and Top N (mean) on actual points,
  their share of the hindsight-best lineup, and the DK-FPPG optimizer baseline of the same pool
  (the grader's slate baselines), paired by slate.

CLI: ``courtside dfs real-calib [--sims 2000] [--slate S] [--no-write] [--json] [--db PATH]``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config

log = logging.getLogger("courtside.proj.dfs.real_calib")

VERSION = "real-calib-v1"
REPORT_DIR = "real_calib"
REPORT_NAME = "real_calib.json"
SEASON_START = "2026-10-20"
CLASSIC = 70
SMALL_SLATE_MAX_GAMES = 4
FIELD_N = 5_000               # modelled-field sample per contest (sims > 0)
SLATES_FOR_DU_PIT = 33        # E5: a 0.10 shift in DU P(cash) via the line PIT
SLATES_FOR_TOP1 = 100


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _loads(s, default=None):
    try:
        return json.loads(s) if s else default
    except (TypeError, ValueError):
        return default


def _q(conn, sql, args=()) -> list[dict]:
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.Error:
        return []


def _key(slots_json) -> tuple | None:
    """Player-set key of a slots_json (DK ids, sorted)."""
    s = _loads(slots_json, None) if isinstance(slots_json, str) else slots_json
    if not s:
        return None
    try:
        return tuple(sorted(str(d.get("player_id")) for d in s))
    except AttributeError:
        return None


def mean_se(values) -> dict:
    """Mean over slates and its SE (sd / sqrt(n)); SE None below 2 slates."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0, "mean": None, "se": None}
    return {"n": int(len(v)), "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None}


# ============================================================================== inputs

def slate_rows(conn, since: str = SEASON_START) -> dict[str, dict]:
    """Classic slates of the season: id -> {day, tag, lock, n_games}."""
    out = {}
    for r in _q(conn, "SELECT slate_id, nba_day, tag, lock_utc, game_ids_json FROM dfs_slates "
                      "WHERE game_type_id=? AND nba_day>=? ORDER BY nba_day, lock_utc",
                (CLASSIC, since)):
        g = _loads(r["game_ids_json"], []) or []
        out[str(r["slate_id"])] = {"slate_id": str(r["slate_id"]), "day": r["nba_day"],
                                   "tag": r["tag"], "lock": r["lock_utc"], "n_games": len(g)}
    return out


def slate_lineups(conn, sid: str) -> list[dict]:
    """Our stored lineups of a slate (ok runs) with their actual points (when graded)."""
    rows = _q(conn, "SELECT l.id, l.run_id, l.mode, l.rank, l.slots_json, l.p_cash, "
                    "COALESCE(g.actual, l.actual) AS actual, r.kind, r.created_at "
                    "FROM dfs_lineups l JOIN dfs_runs r ON r.run_id=l.run_id "
                    "LEFT JOIN dfs_lineup_grades g ON g.lineup_id=l.id "
                    "WHERE r.slate_id=? AND r.ok=1 ORDER BY r.created_at, l.mode, l.rank",
              (sid,))
    for r in rows:
        r["key"] = _key(r["slots_json"])
    return rows


def match_entries(entries: list[dict], lineups: list[dict]) -> None:
    """Fill each entry's ``lineup`` (one of `lineups`) and ``matched_by`` in place: its
    lineup_id, else its player set, else the points (only when exactly one player set of the
    slate scored exactly those points; ties between runs go to the run explaining most
    entries)."""
    by_id = {r["id"]: r for r in lineups}
    by_key: dict = {}
    for r in lineups:
        if r["key"]:
            by_key.setdefault(r["key"], r)
    cover: dict = {}
    for e in entries:
        if e.get("points") is None:
            continue
        for r in lineups:
            if r["actual"] is not None and abs(float(r["actual"]) - float(e["points"])) < 0.01:
                cover.setdefault(r["run_id"], set()).add(e["entry_id"])
    best_runs = sorted(cover, key=lambda k: -len(cover[k]))
    for e in entries:
        e["lineup"], e["matched_by"] = None, None
        if e.get("lineup_id") in by_id:
            e["lineup"], e["matched_by"] = by_id[e["lineup_id"]], "lineup id"
            continue
        k = _key(e.get("slots_json"))
        if k and k in by_key:
            e["lineup"], e["matched_by"] = by_key[k], "player set"
            continue
        if e.get("points") is None:
            continue
        hits = [r for r in lineups if r["actual"] is not None and r["key"]
                and abs(float(r["actual"]) - float(e["points"])) < 0.01]
        keys = {r["key"] for r in hits}
        order = {k_: i for i, k_ in enumerate(best_runs)}
        hits.sort(key=lambda r: (order.get(r["run_id"], 99), r["mode"] != "topn",
                                 r["rank"] or 0))
        if len(keys) == 1:
            e["lineup"], e["matched_by"] = hits[0], "points"
        elif len(keys) > 1:
            lead = (best_runs and (len(best_runs) == 1 or len(cover[best_runs[0]])
                                   > len(cover[best_runs[1]])))
            top = [r for r in hits if lead and r["run_id"] == best_runs[0]]
            if top and len({r["key"] for r in top}) == 1:
                e["lineup"] = top[0]
                e["matched_by"] = "points (the run that explains most entries)"
            else:
                e["matched_by"] = f"ambiguous ({len(keys)} lineups scored {e['points']})"


def contest_info(conn, cid: str, hist: dict, entries: list[dict] | None = None) -> dict:
    """Name, fee, field, places paid, payout table and type of a contest from what is known
    (captured contest, contest history, the owner's entries, uploaded standings)."""
    from . import field as F
    c = F.load_contest(conn, cid)
    h = hist.get(str(cid), {})
    ent = entries or []
    name = (c.name if c and c.name else None) or h.get("name") or f"Contest {cid}"
    fee = (c.entry_fee if c and c.entry_fee else None) or h.get("entry_fee") or next(
        (e["entry_fee"] for e in ent if e.get("entry_fee")), None)
    m = (c.field_size if c and c.field_size > 1 else None) or h.get("entries") or next(
        (e["field_size"] for e in ent if e.get("field_size")), None)
    pays = list(c.payouts) if c and c.payouts else []
    places, how = None, None
    if pays:
        places, how = max(int(t["max_pos"]) for t in pays), (
            "inferred double-up" if c.source == "inferred" else "payout table")
    elif h.get("places_paid"):
        places, how = int(h["places_paid"]), "contest history"
    if m is None:
        r = conn.execute("SELECT COUNT(*) FROM dfs_contest_results WHERE contest_id=?",
                         (str(cid),)).fetchone()
        m = int(r[0]) if r and r[0] else None
    if F.is_double_up(name):
        arch = "cash"
        if places is None and m:
            places, how = F.du_paid_places(m), "inferred double-up"
    elif c is not None and c.payouts:
        arch = c.archetype
    else:
        arch = "small_gpp" if (m or 0) <= F.SMALL_GPP_MAX else "large_gpp"
    return {"contest_id": str(cid), "name": name, "fee": _f(fee), "field": m, "type": arch,
            "places_paid": places, "places_how": how, "payouts": pays}


def _bracket(entries: list[dict], ok_fn) -> dict | None:
    """(lo, hi] of a line from entries on either side: hi = the lowest score that made it, lo =
    the highest that did not. One side only gives a half-open bound."""
    made = [e["points"] for e in entries if e.get("points") is not None and ok_fn(e)]
    miss = [e["points"] for e in entries if e.get("points") is not None and not ok_fn(e)]
    if not made and not miss:
        return None
    return {"lo": max(miss) if miss else None, "hi": min(made) if made else None,
            "source": "bracketed by the owner's entries"}


def contest_block(conn, cid: str, entries: list[dict], hist: dict) -> dict:
    """One contest: info, real lines (exact or bracketed) and the owner's entries."""
    info = contest_info(conn, cid, hist, entries)
    pts = sorted((float(r[0]) for r in conn.execute(
        "SELECT points FROM dfs_contest_results WHERE contest_id=? AND points IS NOT NULL",
        (str(cid),))), reverse=True)
    m = info["field"]
    places = info["places_paid"]
    for e in entries:
        rk, fs = e.get("rank"), e.get("field_size") or m
        e["beat"] = (1.0 - (rk - 0.5) / fs) if rk and fs else None
        if places and rk:
            e["cashed"] = bool(rk <= places)
        else:
            e["cashed"] = (e.get("payout") or 0) > 0 if e.get("payout") is not None else None
        e["top1"] = bool(rk <= max(1, math.ceil(0.01 * fs))) if rk and fs else None
    out = {**{k: v for k, v in info.items() if k != "payouts"}, "standings": bool(pts),
           "entries": []}
    if pts and places:
        out["cash_line"] = {"value": pts[min(places, len(pts)) - 1],
                            "source": f"standings, place {places} ({info['places_how']})"}
        out["top1_line"] = {"value": pts[max(1, math.ceil(0.01 * len(pts))) - 1],
                            "source": "standings"}
        out["standings_field"] = len(pts)
    else:
        b = _bracket([e for e in entries if e["cashed"] is not None], lambda e: e["cashed"])
        out["cash_line"] = b
        t = _bracket([e for e in entries if e["top1"] is not None], lambda e: e["top1"])
        out["top1_line"] = t
    for e in entries:
        lu = e.get("lineup")
        out["entries"].append({
            "entry_id": e["entry_id"], "points": _f(e.get("points")), "rank": e.get("rank"),
            "field": e.get("field_size") or m, "beat": _f(e.get("beat")), "cashed": e["cashed"],
            "payout": _f(e.get("payout")), "lineup_id": lu["id"] if lu else None,
            "run_id": lu["run_id"] if lu else None,
            "lineup": (f"{lu['kind']} {lu['mode']} #{lu['rank']}" if lu else None),
            "stored_p_cash": _f(lu["p_cash"]) if lu else None, "matched_by": e.get("matched_by")})
    ok = [e for e in out["entries"] if e["cashed"] is not None]
    out["cash_rate"] = (sum(e["cashed"] for e in ok) / len(ok)) if ok else None
    bt = [e["beat"] for e in out["entries"] if e["beat"] is not None]
    out["mean_beat"] = float(np.mean(bt)) if bt else None
    return out


# ============================================================================== simulated vs real

def _pit(sim: np.ndarray, real: float) -> float:
    return float((sim < real).mean() + 0.5 * (sim == real).mean())


def contest_sims(conn, sid: str, cblock: dict, entries: list[dict], payouts: list[dict], *,
                 n_sims: int, seed: int = 0, n_field: int = FIELD_N, ctx_cache: dict) -> dict:
    """Each entered lineup of one contest against a modelled field of that contest (see the
    module doc). Returns {lineup_id: {p_cash, pit_line (or pit_line_lo / hi), pit_rank}} plus
    the simulated cash-line summary."""
    from . import field as F
    from . import portfolio as PF
    lus = {}
    for e in entries:
        lu = e.get("lineup")
        if lu is not None:
            lus.setdefault(lu["id"], lu)
    if not lus or not cblock.get("field") or not cblock.get("places_paid"):
        return {"status": "skipped", "reason": "no matched lineup, field size or places paid"}
    runs = pd.Series([lu["run_id"] for lu in lus.values()]).value_counts()
    run_id = str(runs.index[0])
    if run_id not in ctx_cache:
        ctx_cache[run_id] = F.prepare(conn, sid, run_id, n_sims=n_sims, seed=seed)
    ctx = ctx_cache[run_id]
    if ctx["sims"] is None:
        return {"status": "skipped", "reason": f"run {run_id} has no stored sims"}
    M = int(cblock["field"])
    pays = payouts or [{"min_pos": 1, "max_pos": int(cblock["places_paid"]),
                        "payout": 2.0 * float(cblock.get("fee") or 1.0)}]
    contest = F.Contest(cblock["contest_id"], cblock["name"], float(cblock.get("fee") or 1.0), M,
                        pays, cblock["type"], "real-calib")
    f = F.field_for(ctx, contest, n=n_field, seed=seed, k=0)
    fp, ainfo = F.align_sims(ctx["sims"], f.pids, n_sims, seed)
    fp, _ = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                          np.random.default_rng(seed + 7))
    ids = list(lus)
    ev = F.evaluate(f, fp, contest, [lus[i]["slots_json"] for i in ids], seed=seed,
                    tiers=False, per_sim=True)
    sc = (ev.per_sim["score"].astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
    dsim = sc - ev.cash_line[:, None]
    rank_sim = ev.per_sim["above"] + 1.0 + 0.5 * ev.per_sim["tied"]
    beat_sim = 1.0 - (rank_sim - 0.5) / M
    line = cblock.get("cash_line") or {}
    out = {"status": "ok", "run_id": run_id, "n_sims": ainfo["n_sims"], "field_n": n_field,
           "sim_cash_line": {k: _f(v) for k, v in ev.cash_line_summary().items()},
           "missing_sim_players": len(ainfo["missing"]), "lineups": {}}
    for j, lid in enumerate(ids):
        pts = lus[lid]["actual"]
        d: dict = {"p_cash": _f(ev.lineups[j]["p_cash"]), "sim_mean": _f(sc[:, j].mean())}
        if pts is not None:
            if line.get("value") is not None:
                d["pit_line"] = _pit(dsim[:, j], float(pts) - float(line["value"]))
            elif line.get("lo") is not None or line.get("hi") is not None:
                # line in (lo, hi]: score - line in [score - hi, score - lo)
                lo = _pit(dsim[:, j], float(pts) - float(line["hi"])) if line.get("hi") is not None else 0.0
                hi = _pit(dsim[:, j], float(pts) - float(line["lo"])) if line.get("lo") is not None else 1.0
                d["pit_line_lo"], d["pit_line_hi"] = lo, hi
        ent = [e for e in entries if e.get("lineup") is not None and e["lineup"]["id"] == lid
               and e.get("beat") is not None]
        if ent:
            d["pit_rank"] = float(np.mean([_pit(beat_sim[:, j], e["beat"]) for e in ent]))
        out["lineups"][str(lid)] = d
    return out


# ============================================================================== nightly

def nightly(conn, since: str, until: str) -> dict:
    """Per Classic slate: our in-effect cash lineup and Top N mean vs the DK-FPPG optimizer
    baseline on actual points, and each one's share of the hindsight-best lineup."""
    from .. import reports as R
    lo, hi = date.fromisoformat(since), date.fromisoformat(until)
    try:
        L = R.load_lineup_frame(conn, lo, hi)
        B = R.load_baselines(conn, lo, hi)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return {}
    if L.empty:
        return {}
    eff = L[(L["effective"] == 1) & (L["gt"].fillna("classic") != "showdown")
            & (L["slate_kind"] != "showdown")]
    out = {}
    base = B.set_index("slate_id") if len(B) else pd.DataFrame()
    for sid, g in eff.groupby("slate_id"):
        row: dict = {"day": g["day"].iloc[0], "tag": g["slate_kind"].iloc[0]}
        for role in ("cash", "topn"):
            x = g[g["role"] == role]
            if len(x):
                row[role] = {"actual": _f(x["actual"].mean()), "ratio": _f(x["ratio"].mean()),
                             "n": int(len(x))}
        if len(base) and str(sid) in base.index:
            b = base.loc[str(sid)]
            row["fppg"] = {"actual": _f(b["fppg"]), "ratio": _f(b["fppg_ratio"])}
            row["hindsight"] = _f(b["hindsight"])
        out[str(sid)] = row
    return out


def nightly_summary(rows: dict) -> dict:
    out: dict = {"n_slates": len(rows)}
    for role in ("cash", "topn"):
        d = [r[role]["actual"] - r["fppg"]["actual"] for r in rows.values()
             if role in r and "fppg" in r and r["fppg"]["actual"] is not None
             and r[role]["actual"] is not None]
        out[f"{role}_vs_fppg"] = mean_se(d)
        out[f"{role}_ratio"] = mean_se([r[role]["ratio"] for r in rows.values() if role in r])
    out["fppg_ratio"] = mean_se([r["fppg"]["ratio"] for r in rows.values() if "fppg" in r])
    return out


# ============================================================================== report

def _slate_size(n_games: int | None) -> str:
    return "2-4 games" if (n_games or 0) <= SMALL_SLATE_MAX_GAMES else "5+ games"


def ownership_rows(conn) -> list[dict]:
    """`field.calibration_report` contest rows that have a proxy comparison."""
    from . import field as F
    try:
        rep = F.calibration_report(conn)
    except Exception as e:  # noqa: BLE001 - a report section must never fail the report
        log.warning("ownership section unavailable: %s", e)
        return []
    return [r for r in rep.get("contests", []) if r.get("proxy")]


def build_report(conn, *, since: str = SEASON_START, until: str | None = None,
                 sims: int = 0, seed: int = 0, slate: str | None = None,
                 n_field: int = FIELD_N, ownership: bool = True) -> dict:
    """The E5 report (see the module doc). `sims` = 0 skips the modelled-field part."""
    from . import standings as ST
    until = until or config.yesterday_nba_day().isoformat()
    slates = {k: v for k, v in slate_rows(conn, since).items() if v["day"] <= until}
    hist = ST.history_contests(conn)
    ent = _q(conn, "SELECT entry_id, contest_id, slate_id, lineup_id, slots_json, entry_fee, "
                   "points, rank, field_size, payout, source FROM dfs_entries "
                   "WHERE slate_id IS NOT NULL")
    std = {str(r["contest_id"]): str(r["slate_id"]) for r in _q(
        conn, "SELECT DISTINCT r.contest_id, c.slate_id FROM dfs_contest_results r JOIN "
              "dfs_contests c ON c.contest_id=r.contest_id WHERE c.slate_id IS NOT NULL")}
    by_slate: dict[str, dict[str, list]] = {}
    for e in ent:
        by_slate.setdefault(str(e["slate_id"]), {}).setdefault(str(e["contest_id"]), []).append(e)
    for cid, sid in std.items():
        by_slate.setdefault(sid, {}).setdefault(cid, [])
    own = ownership_rows(conn) if ownership else []
    night = nightly(conn, since, until)
    rep: dict = {"version": VERSION, "generated_at": datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "from": since, "to": until, "sims": int(sims), "slates": []}
    for sid in sorted(set(by_slate) & set(slates), key=lambda s: (slates[s]["day"], s)):
        if slate and sid != str(slate):
            continue
        sl = slates[sid]
        lus = slate_lineups(conn, sid)
        all_e = [e for es in by_slate[sid].values() for e in es]
        match_entries(all_e, lus)
        blk = {**sl, "size": _slate_size(sl["n_games"]), "contests": [], "ownership": [],
               "nightly": night.get(sid)}
        ctx_cache: dict = {}
        for cid, es in sorted(by_slate[sid].items()):
            # one row per DK entry (a history file may list an entry twice)
            seen, uniq = set(), []
            for e in es:
                k = e.get("entry_id")
                if k not in seen:
                    seen.add(k)
                    uniq.append(e)
            cb = contest_block(conn, cid, uniq, hist)
            if sims and uniq:
                try:
                    info = contest_info(conn, cid, hist, uniq)
                    cb["sim"] = contest_sims(conn, sid, cb, uniq, info["payouts"], n_sims=sims,
                                             seed=seed, n_field=n_field, ctx_cache=ctx_cache)
                except Exception as x:  # noqa: BLE001
                    log.exception("real-calib sims failed for %s", cid)
                    cb["sim"] = {"status": "error", "reason": f"{type(x).__name__}: {x}"}
            blk["contests"].append(cb)
        for r in own:
            if str(r.get("slate_id")) == sid:
                blk["ownership"].append({"contest_id": r["contest_id"], "name": r.get("name"),
                                         "type": r.get("archetype"), **r["proxy"]})
        rep["slates"].append(blk)
    for sid, row in sorted(night.items(), key=lambda kv: kv[1]["day"]):
        if sid not in {b["slate_id"] for b in rep["slates"]} and sid in slates and \
                (not slate or sid == str(slate)):
            rep["slates"].append({**slates[sid], "size": _slate_size(slates[sid]["n_games"]),
                                  "contests": [], "ownership": [], "nightly": row})
    rep["slates"].sort(key=lambda b: (b["day"], b["slate_id"]))
    rep["summary"] = summarize(rep)
    rep["needs"] = needs(rep)
    return rep


def _slate_pits(blk: dict, kind: str) -> list:
    """Per contest of type `kind` ('cash' / 'gpp'): the slate-level PIT values."""
    out = []
    for c in blk["contests"]:
        is_cash = c["type"] == "cash"
        if (kind == "cash") != is_cash:
            continue
        sim = c.get("sim") or {}
        for d in (sim.get("lineups") or {}).values():
            out.append(d)
    return out


def summarize(rep: dict) -> dict:
    """Slate-level aggregates (one value per slate, then mean and SE over slates)."""
    du_line, du_mid, gpp_rank, beat, du_rate, du_p = [], [], [], [], [], []
    gpp_rate, gpp_p = [], []
    own: dict = {}
    for b in rep["slates"]:
        c_line = [d["pit_line"] for d in _slate_pits(b, "cash") if "pit_line" in d]
        c_mid = [0.5 * (d["pit_line_lo"] + d["pit_line_hi"]) for d in _slate_pits(b, "cash")
                 if "pit_line_lo" in d]
        if c_line:
            du_line.append(float(np.mean(c_line)))
        if c_line or c_mid:
            du_mid.append(float(np.mean(c_line + c_mid)))
        g = [d["pit_rank"] for d in _slate_pits(b, "gpp") if "pit_rank" in d]
        if g:
            gpp_rank.append(float(np.mean(g)))
        bt = [e["beat"] for c in b["contests"] if c["type"] != "cash" for e in c["entries"]
              if e["beat"] is not None]
        if bt:
            beat.append(float(np.mean(bt)))
        du = [e["cashed"] for c in b["contests"] if c["type"] == "cash" for e in c["entries"]
              if e["cashed"] is not None]
        if du:
            du_rate.append(float(np.mean(du)))
        ps = [d["p_cash"] for d in _slate_pits(b, "cash") if d.get("p_cash") is not None]
        if ps:
            du_p.append(float(np.mean(ps)))
        gr = [e["cashed"] for c in b["contests"] if c["type"] != "cash" for e in c["entries"]
              if e["cashed"] is not None]
        if gr:
            gpp_rate.append(float(np.mean(gr)))
        gp = [d["p_cash"] for d in _slate_pits(b, "gpp") if d.get("p_cash") is not None]
        if gp:
            gpp_p.append(float(np.mean(gp)))
        for o in b["ownership"]:
            k = f"{o['type']} | {b['size']}"
            a = own.setdefault(k, {"type": o["type"], "size": b["size"], "rows": []})
            a["rows"].append((b["slate_id"], o))
    own_out = []
    for k, a in sorted(own.items()):
        per_slate: dict = {}
        for sid, o in a["rows"]:
            per_slate.setdefault(sid, []).append(o)
        agg = {}
        for m in ("mae_pp", "spearman", "top10_bias_pp", "top10_mae_pp"):
            agg[m] = mean_se([float(np.mean([o[m] for o in os_ if o.get(m) is not None]))
                              for os_ in per_slate.values()])
        own_out.append({"type": a["type"], "size": a["size"], "contests": len(a["rows"]),
                        "slates": len(per_slate), **agg})
    nights = {b["slate_id"]: b["nightly"] for b in rep["slates"] if b.get("nightly")}
    return {"du_pit_line": {**mean_se(du_line), "needed": SLATES_FOR_DU_PIT},
            "du_pit_line_incl_bracketed": mean_se(du_mid),
            "du_cash_rate": mean_se(du_rate), "du_sim_p_cash": mean_se(du_p),
            "gpp_pit_rank": mean_se(gpp_rank), "gpp_mean_beat": mean_se(beat),
            "gpp_cash_rate": mean_se(gpp_rate), "gpp_sim_p_cash": mean_se(gpp_p),
            "ownership": own_out, "nightly": nightly_summary(nights),
            "slates_with_entries": sum(1 for b in rep["slates"] if b["contests"]),
            "slates_with_standings": sum(1 for b in rep["slates"]
                                         if any(c["standings"] for c in b["contests"]))}


def needs(rep: dict) -> list[str]:
    """What the owner could upload to tighten the report (plain sentences)."""
    out = []
    for b in rep["slates"]:
        miss = [c for c in b["contests"] if not c["standings"]]
        if not miss:
            continue
        big = max(miss, key=lambda c: c.get("field") or 0)
        out.append(f"{b['day']} slate {b['slate_id']}: standings for {big['name']} "
                   f"({big['contest_id']}, {big.get('field') or '?'} entries) would give the exact "
                   f"cash and top-1 % lines and the field's ownership; "
                   f"{len(miss)} of {len(b['contests'])} contests have no standings.")
    if not any(c["type"] == "cash" for b in rep["slates"] for c in b["contests"]):
        out.append("No double-up yet: DU P(cash) calibration needs double-up entries.")
    return out


def report_path(data_dir: Path | None = None) -> Path:
    return (data_dir or config.data_dir()) / "reports" / REPORT_DIR / REPORT_NAME


def write_report(rep: dict, data_dir: Path | None = None) -> Path:
    p = report_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(rep, indent=1, default=str)
    for name in (REPORT_NAME, f"real_calib_{rep['to']}.json"):
        tmp = p.parent / (name + ".tmp")
        tmp.write_text(txt)
        os.replace(tmp, p.parent / name)
    return p


def load_report(conn: sqlite3.Connection | None = None, data_dir: Path | None = None) -> dict:
    """The latest report ({} if none). With `conn`, the data dir is the database's folder."""
    if data_dir is None and conn is not None:
        for row in conn.execute("PRAGMA database_list"):
            if row[1] == "main" and row[2]:
                data_dir = Path(row[2]).parent
    p = report_path(data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


# ============================================================================== text

def _n(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def _line(c: dict | None) -> str:
    if not c:
        return "-"
    if c.get("value") is not None:
        return f"{c['value']:.2f}"
    lo, hi = c.get("lo"), c.get("hi")
    return f"({_n(lo, 2) if lo is not None else '?'}, {_n(hi, 2) if hi is not None else '?'}]"


def _ms(s: dict, d=3) -> str:
    if not s or s.get("mean") is None:
        return "-"
    return f"{s['mean']:.{d}f}" + (f" (SE {s['se']:.{d}f})" if s.get("se") is not None else "") \
        + f", {s['n']} slate{'s' if s['n'] != 1 else ''}"


def format_report(rep: dict) -> str:
    L = [f"E5 real-standings calibration ({rep['version']}), {rep['from']} .. {rep['to']}; "
         f"field sims {rep['sims'] or 'off'}"]
    for b in rep["slates"]:
        L.append(f"  {b['day']} slate {b['slate_id']} ({b.get('tag')}, {b['n_games']} games)")
        n = b.get("nightly") or {}
        if n:
            parts = []
            for role in ("cash", "topn"):
                if role in n:
                    parts.append(f"{role} {_n(n[role]['actual'])} ({_n(n[role]['ratio'], 2)} of "
                                 "hindsight)")
            if "fppg" in n:
                parts.append(f"FPPG optimizer {_n(n['fppg']['actual'])} "
                             f"({_n(n['fppg']['ratio'], 2)})")
            L.append("    nightly: " + "; ".join(parts))
        for c in b["contests"]:
            L.append(f"    {c['contest_id']} {c['name'][:40]:<40} {c['type']:<9} field "
                     f"{c.get('field') or '?'} paid {c.get('places_paid') or '?'} "
                     f"({c.get('places_how') or '-'}); cash line {_line(c.get('cash_line'))}; "
                     f"top 1% {_line(c.get('top1_line'))}; "
                     f"{'standings' if c['standings'] else 'no standings'}")
            for e in c["entries"]:
                sim = ((c.get("sim") or {}).get("lineups") or {}).get(str(e["lineup_id"])) or {}
                pit = (f" PIT line {sim['pit_line']:.2f}" if "pit_line" in sim else
                       (f" PIT line [{sim['pit_line_lo']:.2f}, {sim['pit_line_hi']:.2f}]"
                        if "pit_line_lo" in sim else ""))
                pr = f" PIT place {sim['pit_rank']:.2f}" if "pit_rank" in sim else ""
                pc = f" sim P(cash) {sim['p_cash']:.2f}" if sim.get("p_cash") is not None else ""
                L.append(f"      {_n(e['points'], 2):>7} place {e['rank'] or '?'} beat "
                         f"{_n(e['beat'], 3)} {'cash' if e['cashed'] else 'out ' if e['cashed'] is False else '?   '}"
                         f" {e.get('lineup') or e.get('matched_by') or 'lineup unknown'}{pc}{pit}{pr}")
            s = c.get("sim")
            if s and s.get("status") != "ok":
                L.append(f"      sims: {s.get('status')} ({s.get('reason')})")
        for o in b["ownership"]:
            L.append(f"    ownership {o['contest_id']} {o['type']}: MAE {o['mae_pp']:.1f} pp, rank "
                     f"corr {o['spearman']:.2f}, top-10 bias {o['top10_bias_pp']:+.1f} pp")
    s = rep.get("summary") or {}
    L.append("  summary (unit = slate; mean (SE), slates)")
    L.append(f"    DU PIT of (score - real line): {_ms(s.get('du_pit_line'))} (0.5 = calibrated; "
             f"~{SLATES_FOR_DU_PIT} slates to see a 0.10 shift)")
    L.append(f"    DU realised cash rate {_ms(s.get('du_cash_rate'))}; simulated "
             f"{_ms(s.get('du_sim_p_cash'))}")
    L.append(f"    GPP PIT of the real place: {_ms(s.get('gpp_pit_rank'))}; share of field "
             f"beaten {_ms(s.get('gpp_mean_beat'))}")
    L.append(f"    GPP in the money {_ms(s.get('gpp_cash_rate'))}; simulated "
             f"{_ms(s.get('gpp_sim_p_cash'))}")
    for o in s.get("ownership") or []:
        L.append(f"    ownership {o['type']} {o['size']}: {o['contests']} contests / {o['slates']} "
                 f"slates, MAE {_ms(o['mae_pp'], 1)}, rank corr {_ms(o['spearman'], 2)}, top-10 "
                 f"bias {_ms(o['top10_bias_pp'], 1)}")
    nt = s.get("nightly") or {}
    if nt:
        L.append(f"    nightly ({nt['n_slates']} slates): cash - FPPG optimizer "
                 f"{_ms(nt.get('cash_vs_fppg'), 1)}; Top N - FPPG {_ms(nt.get('topn_vs_fppg'), 1)};"
                 f" share of hindsight cash {_ms(nt.get('cash_ratio'), 3)}, Top N "
                 f"{_ms(nt.get('topn_ratio'), 3)}, FPPG {_ms(nt.get('fppg_ratio'), 3)}")
    for t in rep.get("needs") or []:
        L.append(f"  needs: {t}")
    return "\n".join(L)


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("real-calib", help="E5: per-slate real-standings calibration report "
                                          "(report only)")
    p.add_argument("--from", dest="since", default=SEASON_START)
    p.add_argument("--to", dest="until", default=None)
    p.add_argument("--slate", default=None)
    p.add_argument("--sims", type=int, default=0,
                   help="sims per lineup for the modelled-field part (0 = off; 2000 is enough)")
    p.add_argument("--field-n", type=int, default=FIELD_N)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--db", default=None, help="database path (opened read-only)")
    p.add_argument("--no-write", action="store_true", help="do not write the JSON report")
    p.add_argument("--json", action="store_true")
    return {"real-calib": _cmd}


def _cmd(args) -> int:
    path = Path(args.db) if args.db else config.db_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rep = build_report(conn, since=args.since, until=args.until, sims=args.sims,
                           seed=args.seed, slate=args.slate, n_field=args.field_n)
    finally:
        conn.close()
    if not args.no_write:
        p = write_report(rep)
        print(f"wrote {p}")
    print(json.dumps(rep, indent=1, default=str) if args.json else format_report(rep))
    return 0
