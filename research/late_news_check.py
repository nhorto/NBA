"""Late-news tag check (docs/IMPROVEMENTS.md D4; product code `proj/dfs/late_news_tag.py`).

For every slate with stored DFS runs (ok, not swap) this recomputes the tag of each run as of the
run's time (or reads the stored one, `dfs_runs.detail["late_news_tag"]`) and reports:

  1. per slate: the late news players found (rotation / taken in by the run / baseline run) and
     the number of flags per run;
  2. per flagged player (one row per slate and player, the latest run that flagged him): the
     news, minutes to lock, gain, our projection vs his realised DK points (`player_games`) and
     the proxy popularity vs realised ownership (`dfs_contest_ownership`, uploaded standings of
     the slate's contests, one value per contest);
  3. near misses: for each rotation news player, the three largest teammate DK gains even when
     below the tag's minimum, with the same realised columns (the thresholds are a first guess).

Read-only. Point COURTSIDE_DATA at a data root (or pass --db). Sims files are not read.

    COURTSIDE_DATA=var/tmp/d4 uv run python research/late_news_check.py [--min-gain 3]
        [--window 60] [--rotation-min 15] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

from courtside import config
from courtside.proj.dfs import late_news_tag as L
from courtside.proj.dfs import rules as R


def _realised(conn: sqlite3.Connection, day: str, pids: set[int]) -> dict[int, dict]:
    if not pids:
        return {}
    q = ",".join("?" * len(pids))
    out = {}
    for r in conn.execute(f"SELECT * FROM player_games WHERE game_date=? AND player_id IN ({q})",
                          (day, *pids)):
        out[int(r["player_id"])] = {"dk": round(R.dk_points(r), 2), "min": r["min"]}
    return out


def _own(conn: sqlite3.Connection, slate_id: str) -> dict[int, list[tuple[str, float]]]:
    """NBA id -> [(contest name or id, %drafted)] over the slate's uploaded contests."""
    out: dict[int, list] = {}
    try:
        rows = conn.execute(
            "SELECT o.contest_id, c.name, c.field_size, o.player_id, o.pct_drafted FROM "
            "dfs_contest_ownership o JOIN dfs_contests c USING (contest_id) WHERE c.slate_id=?",
            (slate_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    for cid, name, fs, pid, pct in rows:
        out.setdefault(int(pid), []).append((f"{cid} (n={fs})", pct))
    return out


def _contests(conn: sqlite3.Connection, slate_id: str) -> int:
    try:
        return conn.execute("SELECT COUNT(DISTINCT o.contest_id) FROM dfs_contest_ownership o "
                            "JOIN dfs_contests c USING (contest_id) WHERE c.slate_id=?",
                            (slate_id,)).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def _fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def run(conn: sqlite3.Connection, *, window: int, min_gain: float, use_stored: bool) -> dict:
    slates = conn.execute(
        "SELECT DISTINCT r.slate_id, s.nba_day, s.tag, s.lock_utc FROM dfs_runs r JOIN "
        "dfs_slates s USING (slate_id) WHERE r.ok=1 AND r.kind!='swap' ORDER BY s.lock_utc, "
        "r.slate_id").fetchall()
    report = {"window_min": window, "min_gain": min_gain, "version": L.VERSION, "slates": []}
    for sid, day, tag, lock in slates:
        runs = conn.execute("SELECT run_id, kind, created_at, model_version, detail FROM dfs_runs "
                            "WHERE slate_id=? AND ok=1 AND kind!='swap' ORDER BY created_at",
                            (sid,)).fetchall()
        srep = {"slate_id": sid, "day": day, "tag": tag, "lock": lock, "runs": [],
                "flags": {}, "near": {}, "contests": _contests(conn, sid)}
        for rid, kind, at, mv, detail in runs:
            det = L._loads(detail, {}) or {}
            res = det.get(L.DETAIL_KEY) if use_stored and isinstance(det, dict) else None
            stored = isinstance(res, dict) and res.get("version") == L.VERSION
            if not stored:
                res = L.compute(conn, sid, rid, at, mv, window=timedelta(minutes=window),
                                min_gain_dk=min_gain, min_gain_min=min_gain)
            news = [n for n in res["news"]]
            srep["runs"].append({"run_id": rid, "kind": kind, "at": at, "stored": stored,
                                 "news": [{k: n[k] for k in ("name", "team", "status_from",
                                                             "status_to", "source", "at",
                                                             "min_to_lock", "rotation",
                                                             "taken_in", "baseline_run")}
                                          for n in news],
                                 "flags": len(res["flags"])})
            for f in res["flags"]:
                srep["flags"][f["player_id"]] = {**f, "run_id": rid, "kind": kind}
            # near misses: best teammate gains for rotation news players, no minimum
            rot = [n for n in news if n["rotation"] and n["taken_in"] and n["baseline_run"]]
            if rot:
                loose = L.compute(conn, sid, rid, at, mv, window=timedelta(minutes=window),
                                  min_gain_dk=-1e9, min_gain_min=-1e9)
                for f in loose["flags"]:
                    srep["near"][(f["player_id"])] = {**f, "run_id": rid, "kind": kind}
        pids = set(srep["flags"]) | set(srep["near"])
        real = _realised(conn, day, pids)
        own = _own(conn, sid)
        for d in (srep["flags"], srep["near"]):
            for pid, f in d.items():
                f["real_dk"] = (real.get(pid) or {}).get("dk")
                f["real_min"] = (real.get(pid) or {}).get("min")
                f["real_own"] = own.get(pid, [])
        report["slates"].append(srep)
    return report


def _row_text(f: dict) -> str:
    news = ", ".join(f"{n['name']} {n['status_to']} @{n['min_to_lock']:.0f}m"
                     for n in f.get("news") or [])
    ro = "; ".join(f"{c.split(' ')[0]}:{p:.1f}%" for c, p in f["real_own"]) or "-"
    return (f"  {f['name'][:22]:<22} {f['team']:<4} {f['kind']:<6} news [{news}] "
            f"gain {_fmt(f['gain_dk'])} DK / {_fmt(f['gain_min'])} min | proj {_fmt(f['proj_dk'])}"
            f" real {_fmt(f['real_dk'])} DK ({_fmt(f['real_min'], 0)} min) | proxy "
            f"{_fmt(f.get('own'))}% {f.get('band') or '-'} real {ro}")


def print_report(rep: dict) -> None:
    print(f"late-news tag {rep['version']}: window {rep['window_min']} min, min gain "
          f"{rep['min_gain']:g} DK or min, rotation >= {L.ROTATION_MIN:g} min, <= {L.PER_TEAM} "
          f"per team")
    print("\n1. flags per slate")
    print(f"  {'slate':<7} {'day':<10} {'tag':<8} {'lock':<20} {'news (rot)':<11} "
          f"{'flags per run':<28} uploads")
    tot_runs = tot_flags = 0
    for s in rep["slates"]:
        names = {}
        for r in s["runs"]:
            for n in r["news"]:
                names[n["name"]] = n["rotation"] and n["taken_in"]
        per = " ".join(f"{r['kind']}:{r['flags']}" for r in s["runs"]
                       if r["kind"] in ("lock45", "lock10")) or \
            " ".join(f"{r['kind']}:{r['flags']}" for r in s["runs"][-2:])
        tot_runs += len(s["runs"])
        tot_flags += sum(r["flags"] for r in s["runs"])
        print(f"  {s['slate_id']:<7} {s['day']:<10} {s['tag'] or '-':<8} {s['lock'] or '-':<20} "
              f"{len(names):>2} ({sum(names.values())})     {per:<28} {s['contests']}")
    print(f"  runs {tot_runs}, flags {tot_flags}, flagged players "
          f"{sum(len(s['flags']) for s in rep['slates'])}")
    print("\n   news players (status move inside the window, as of each run):")
    seen = set()
    for s in rep["slates"]:
        for r in s["runs"]:
            for n in r["news"]:
                k = (s["slate_id"], n["name"])
                if k in seen:
                    continue
                seen.add(k)
                print(f"  {s['slate_id']:<7} {n['name'][:22]:<22} {n['team']:<4} "
                      f"{n['status_from']} -> {n['status_to']} ({n['source']}) {n['at']} "
                      f"{n['min_to_lock']:.0f} min to lock; rotation {n['rotation']}, "
                      f"taken in {n['taken_in']}, baseline {n['baseline_run'] or '-'}")
    print("\n2. flagged players: projection vs realised DK, proxy vs realised ownership")
    any_flag = False
    for s in rep["slates"]:
        for f in s["flags"].values():
            any_flag = True
            print(f"  [{s['slate_id']}]" + _row_text(f))
    if not any_flag:
        print("  (none)")
    print("\n3. near misses: top teammate gains of rotation news players (no minimum)")
    any_near = False
    for s in rep["slates"]:
        for f in sorted(s["near"].values(), key=lambda f: -(f["gain_dk"] or 0))[:6]:
            any_near = True
            print(f"  [{s['slate_id']}]" + _row_text(f))
    if not any_near:
        print("  (none)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=None, help="courtside.db (default: config)")
    ap.add_argument("--window", type=int, default=int(L.LATE_NEWS_WINDOW.total_seconds() // 60))
    ap.add_argument("--min-gain", type=float, default=L.MIN_GAIN_DK)
    ap.add_argument("--rotation-min", type=float, default=L.ROTATION_MIN)
    ap.add_argument("--recompute", action="store_true",
                    help="ignore tags stored on the runs (always recompute)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    L.ROTATION_MIN = a.rotation_min
    path = a.db or config.db_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rep = run(conn, window=a.window, min_gain=a.min_gain, use_stored=not a.recompute)
    print_report(rep)
    if a.json:
        def enc(o):
            return {str(k): v for k, v in o.items()} if isinstance(o, dict) else str(o)
        a.json.write_text(json.dumps(rep, default=enc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
