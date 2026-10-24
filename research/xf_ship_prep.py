"""X factor ship prep (D3 / D6): the big-night STRETCH map refit on `+tr` sims (`xfactor.STRETCH_TR`)
and the cost of precomputing the xf-v2 leverage table in the DFS run path.

Docs: docs/research/xfactor-v2.md, docs/research/tail-remap.md (addenda "ship prep").

    python research/xf_ship_prep.py harness S [--tr-key K]
        avail harness rows with --tails (4,000 sims) of the arm in the environment, one RotoGuru
        season -> OUT/rows_avail_<S>_<MV>.parquet. --tr-key: seasons that have no tail-remap
        params (2015-16 .. 2018-19: the walk-forward fit starts with 2018-19 data) use season
        K's params (K = 2019-20 is fit on 2018-19 only, so no held-out fold (2019-20, 2020-21)
        sees its own season through the remap).
    python research/xf_ship_prep.py stretch OUT_JSON ON_FILE... [--off OFF_FILE...]
        the per-tier logistic map of xfactor.md §3 (`xfactor_calibration.stretch_fit`, k = 6,
        P(play) >= 0.85, RotoGuru rows) fit on ALL given ON seasons (the STRETCH_TR candidate),
        plus walk-forward (fit on seasons < S) reliability per held-out season, tier and
        predicted bin: shipped STRETCH vs the refit, on the ON rows (and OFF for reference).
    python research/xf_ship_prep.py build SLATE KIND NOW OUT_JSON [--on]
        (scratch DB copy, one per arm) one DFS runner build (`runner.run_slate`, refresh=False,
        force) with COURTSIDE_XF_V2 OFF or ON; writes a fingerprint (lineups, projections, sims
        hash, detail without the stored table), the build seconds and, ON, the hook's seconds and
        whether the stored table equals a fresh `run_leverage` after the save (the page path).
    python research/xf_ship_prep.py compare OFF_JSON ON_JSON
    python research/xf_ship_prep.py precompute RUN_ID [RUN_ID ...]
        (scratch DB copy) time `xfactor.precompute_run` for stored runs: seconds, and the stored
        table equals a fresh `run_leverage`.

Environment: COURTSIDE_DATA (bundles + DB), XFP_OUT (default var/tmp/xfprep/out).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
OUT = Path(os.environ.get("XFP_OUT", ROOT / "var" / "tmp" / "xfprep" / "out"))
K = 6
COL = "p_ge_6x"
BINS = (0, .05, .10, .15, .20, .30, .40, 1.0001)


def say(m: str) -> None:
    print(f"[xfp] {time.strftime('%H:%M:%S')} {m}", flush=True)


def cmd_harness(season: str, tr_key: str | None) -> None:
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    from courtside.proj import tail_remap as TR
    M.GBM_PARAMS["num_threads"] = 1
    if tr_key and TR.TAIL_REMAP:
        orig = TR.cells

        def cells(s: str) -> dict:                  # seasons without params -> season tr_key's
            t = orig(s)
            return t if t else orig(tr_key)
        TR.cells = cells
        say(f"tail remap: seasons without params use {tr_key}'s "
            f"({len(cells(season))} cells for {season})")
    conn = __import__("courtside.db", fromlist=["connect"]).connect()
    t0 = time.time()
    st = B.Settings(protocol="avail", n_sims=4000, modes=(), baselines=B.BASELINES, pplay="",
                    tails=True)
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_avail_{season}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    say(f"harness {season} {P.MODEL_VERSION}: {len(pl)} rows, skipped {len(meta['skipped'])}, "
        f"{time.time() - t0:.0f}s -> {f}")


def _rows(files: list[str]) -> pd.DataFrame:
    from xfactor_calibration import prep
    d = prep(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    d = d[d["salary"].notna() & d["pplay_hi"] & d[COL].notna()].copy()
    d["hit"] = (d["y"] >= K * d["salary"] / 1000.0).astype(float)
    return d


def _rel(x: pd.DataFrame, pc: str) -> list[dict]:
    from xfactor_calibration import rate_se
    out = []
    p = x[pc].to_numpy(float)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        o, se = rate_se(x["hit"].to_numpy()[m], x["date"].to_numpy()[m])
        out.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                    "pred": float(p[m].mean()), "obs": o, "se": se})
    return out


def _summary(x: pd.DataFrame, pc: str) -> dict:
    from xfactor_calibration import rate_se
    o, se = rate_se(x["hit"].to_numpy(), x["date"].to_numpy())
    r = {"n": int(len(x)), "pred": float(x[pc].mean()), "obs": o, "se": se,
         "brier": float(((x[pc] - x["hit"]) ** 2).mean())}
    br = x[x["tier"].isin(["bench", "role"]) & (x[pc] >= 1 / 3 - 0.0033)]
    if len(br):
        o, se = rate_se(br["hit"].to_numpy(), br["date"].to_numpy())
        r["tag"] = {"n": int(len(br)), "pred": float(br[pc].mean()), "obs": o, "se": se}
    return r


def cmd_stretch(out_json: str, on: list[str], off: list[str]) -> None:
    from xfactor_calibration import stretch_apply, stretch_fit

    from courtside.proj.dfs import xfactor as XF
    arms = {"on": _rows(on)}
    if off:
        arms["off"] = _rows(off)
    rep: dict = {"shipped": XF.STRETCH}
    for arm, d in arms.items():
        allfit = stretch_fit(d, K, COL)
        rep[f"{arm}_all"] = allfit
        say(f"{arm}: n {len(d)} seasons {sorted(d['season'].unique())}; fit on all: "
            + json.dumps({t: [round(a, 4), round(b, 4)] for t, (a, b) in allfit.items()}))
        d["p_ship"] = stretch_apply(d, COL, XF.STRETCH)
        wf = {}
        for s in sorted(d["season"].unique()):
            tr = d[d["season"] < s]
            if tr.empty:
                continue
            ab = stretch_fit(tr, K, COL)
            te = d[d["season"] == s].copy()
            te["p_wf"] = stretch_apply(te, COL, ab)
            blk = {"params": ab, "ship": _summary(te, "p_ship"), "refit": _summary(te, "p_wf"),
                   "raw": _summary(te, COL)}
            for t, x in te.groupby("tier"):
                blk[f"tier_{t}"] = {"ship": _summary(x, "p_ship"), "refit": _summary(x, "p_wf"),
                                    "rel_ship": _rel(x, "p_ship"), "rel_refit": _rel(x, "p_wf")}
            wf[s] = blk
            say(f"  {arm} {s}: obs {100 * blk['ship']['obs']:.2f}%  shipped "
                f"{100 * blk['ship']['pred']:.2f}% (Brier {blk['ship']['brier']:.5f})  refit<S "
                f"{100 * blk['refit']['pred']:.2f}% (Brier {blk['refit']['brier']:.5f})")
        rep[f"{arm}_wf"] = wf
        # pooled over the honest held-out seasons (test sims with walk-forward tr params)
        held = [s for s in wf if s >= "2019-20"]
        if held:
            te = pd.concat([d[d["season"] == s].assign(
                p_wf=stretch_apply(d[d["season"] == s], COL, wf[s]["params"])) for s in held])
            pool = {"seasons": held, "ship": _summary(te, "p_ship"), "refit": _summary(te, "p_wf")}
            for t, x in te.groupby("tier"):
                pool[f"tier_{t}"] = {"ship": _summary(x, "p_ship"), "refit": _summary(x, "p_wf"),
                                     "rel_ship": _rel(x, "p_ship"), "rel_refit": _rel(x, "p_wf")}
            rep[f"{arm}_heldout"] = pool
            say(f"  {arm} held-out {held}: obs {100 * pool['ship']['obs']:.2f}% shipped "
                f"{100 * pool['ship']['pred']:.2f}% refit {100 * pool['refit']['pred']:.2f}%")
            for t in ("bench", "role", "star"):
                b = pool.get(f"tier_{t}")
                if not b:
                    continue
                print(f"    {t:<5} obs {100 * b['ship']['obs']:.2f}±{100 * b['ship']['se']:.2f} "
                      f"ship {100 * b['ship']['pred']:.2f} refit {100 * b['refit']['pred']:.2f}")
                for nm in ("rel_ship", "rel_refit"):
                    print(f"      {nm:<9} " + "  ".join(
                        f"[{r['bin']}] n{r['n']} {100 * r['pred']:.1f}->{100 * r['obs']:.1f}"
                        f"±{100 * r['se']:.1f}" for r in b[nm]))
    Path(out_json).write_text(json.dumps(rep, indent=1, default=float))
    say(f"wrote {out_json}")


def cmd_precompute(run_ids: list[str]) -> None:
    from courtside import db
    from courtside.proj.dfs import xfactor as XF
    conn = db.connect()
    for rid in run_ids:
        t0 = time.perf_counter()
        res = XF.precompute_run(conn, rid)
        t1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        fresh = XF.run_leverage(conn, rid)
        t2 = time.perf_counter() - t0
        same = (res or {}).get("players") == fresh.get("players")
        say(f"{rid}: precompute {t1:.1f}s (stored {bool(res)}), fresh run_leverage {t2:.1f}s, "
            f"players equal {same}, n {len(fresh.get('players') or {})}")


def _fingerprint(conn, run_id: str) -> dict:
    import hashlib
    row = dict(zip(("detail", "late", "mv", "created", "sim_path"), conn.execute(
        "SELECT detail, late_news_json, model_version, created_at, sim_path FROM dfs_runs WHERE "
        "run_id=?", (run_id,)).fetchone()))
    det = json.loads(row["detail"].replace(run_id, "RUN"))
    lus = [list(r) for r in conn.execute(
        "SELECT mode, rank, slots_json, salary, mean, p10, p50, p90, p_cash, p_top1, e_payout, "
        "dupes, why_json, pct_field_proxy FROM dfs_lineups "
        "WHERE run_id=? ORDER BY mode, rank", (run_id,))]
    pj = [list(r) for r in conn.execute(
        "SELECT * FROM player_projections WHERE run_id=? ORDER BY player_id", (run_id,))]
    cols = [d[0] for d in conn.execute("SELECT * FROM player_projections LIMIT 0").description]
    drop = {i for i, c in enumerate(cols) if c in ("id", "run_id", "created_at")}
    pj = [[v for i, v in enumerate(r) if i not in drop] for r in pj]
    h = None
    if row["sim_path"] and Path(row["sim_path"]).exists():
        import numpy as _np
        z = _np.load(row["sim_path"], allow_pickle=False)
        hh = hashlib.sha1()
        for k in sorted(z.files):
            hh.update(k.encode())
            hh.update(_np.ascontiguousarray(z[k]).tobytes())
        h = hh.hexdigest()
    return {"detail": det, "lineups": lus, "projections": pj, "sims_sha1": h,
            "late": row["late"], "mv": row["mv"], "created": row["created"]}


def cmd_build(slate: str, kind: str, now: str, out: str, on: bool) -> None:
    from datetime import datetime

    from courtside import db
    from courtside.proj.dfs import runner
    from courtside.proj.dfs import xfactor as XF
    XF.set_xf_v2(on)
    conn = db.connect()
    t_hook = {}
    orig = XF.store_safe

    def timed(*a, **k):
        t0 = time.perf_counter()
        orig(*a, **k)
        t_hook["s"] = time.perf_counter() - t0
    XF.store_safe = timed
    t0 = time.perf_counter()
    res = runner.run_slate(conn, slate, kind, now=datetime.fromisoformat(now.replace("Z", "+00:00")),
                           refresh=False, force=True)
    wall = time.perf_counter() - t0
    say(f"build {slate} {kind} xf_v2={on}: {res.get('status')} run {res.get('run_id')} "
        f"{wall:.1f}s, hook {t_hook.get('s')}")
    rid = res["run_id"]
    fp = _fingerprint(conn, rid)
    rep = {"run_id": rid, "status": res.get("status"), "wall": wall, "hook_s": t_hook.get("s"),
           "on": on}
    stored = fp["detail"].pop(XF.DETAIL_KEY, None)
    if on:
        t1 = time.perf_counter()
        fresh = XF.run_leverage(conn, rid)                 # the page path after the save
        rep["fresh_s"] = time.perf_counter() - t1
        rep["stored"] = bool(stored)
        rep["stored_players"] = len((stored or {}).get("players") or {})
        rep["stored_equals_fresh"] = bool(stored) and stored["players"] == fresh.get("players")
        rep["stored_read"] = XF.stored_leverage(conn, rid, XF._sims_mtime(conn, rid)) is not None
        tab = XF.run_table(conn, rid)
        rep["tagged"] = sorted(d for d, r in tab.items() if r.get("x"))
        say(f"  stored {rep['stored']} ({rep['stored_players']} players), equals fresh "
            f"{rep['stored_equals_fresh']}, read back {rep['stored_read']}, fresh "
            f"{rep['fresh_s']:.1f}s, tags {rep['tagged']}")
    rep["fingerprint"] = fp
    Path(out).write_text(json.dumps(rep, default=str))


def cmd_compare(a: str, b: str) -> None:
    A, B = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    fa, fb = A["fingerprint"], B["fingerprint"]
    for k in fa:
        if k == "detail":
            ks = sorted(set(fa[k]) | set(fb[k]))
            bad = [x for x in ks if fa[k].get(x) != fb[k].get(x)]
            say(f"detail: {len(ks)} keys, differing {bad}")
        else:
            say(f"{k}: {'identical' if fa[k] == fb[k] else 'DIFFERENT'}"
                + (f" ({len(fa[k])} rows)" if isinstance(fa[k], list) else ""))
    say(f"wall OFF {A['wall']:.1f}s ON {B['wall']:.1f}s, hook ON {B.get('hook_s')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("--tr-key")
    s = sub.add_parser("stretch")
    s.add_argument("out_json")
    s.add_argument("on", nargs="+")
    s.add_argument("--off", nargs="*", default=[])
    b = sub.add_parser("build")
    b.add_argument("slate")
    b.add_argument("kind")
    b.add_argument("now")
    b.add_argument("out")
    b.add_argument("--on", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    p = sub.add_parser("precompute")
    p.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "harness":
        cmd_harness(a.season, a.tr_key)
    elif a.cmd == "stretch":
        cmd_stretch(a.out_json, a.on, a.off)
    elif a.cmd == "build":
        cmd_build(a.slate, a.kind, a.now, a.out, a.on)
    elif a.cmd == "compare":
        cmd_compare(a.a, a.b)
    else:
        cmd_precompute(a.runs)


if __name__ == "__main__":
    main()
