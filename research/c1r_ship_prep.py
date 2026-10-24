"""Ship prep for the copula v1 retune (`+c1r`, `COURTSIDE_COPULA=v1r`): the `+tr` tail-remap
params and the X factor `STRETCH_TR` map refit on c1r sims, and the props check.

Docs: docs/research/tail-remap.md (addendum 2026-10-24, c1r), docs/research/p8-e3.md (the `+c1r`
confirmation). Code under test: `proj/tail_remap.py` (`params_path`: the c1r params file is read
only while `pipeline.COPULA == "v1r"`), `proj/dfs/xfactor.py` (`stretch_table`: `STRETCH_TR_C1R`
only with `+tr` on and copula v1r).

    python research/c1r_ship_prep.py harness S PROTO [--tr-key K] [--tag T] [--pq-lomo PATH]
        p8_e3_eval's harness (absent + minutes capture, 4,000 sims) with --tails (the PIT and
        q95 / q99 / p_ge_6x columns) -> OUT/rows_<P>_<S>_<MV><T>.parquet (+ absent_...).
        --tr-key: seasons without tail-remap params (2015-16 .. 2018-19) use season K's.
        The arm comes from the environment (Mac scratch `run.sh`: COP, TR, PQ).
    python research/c1r_ship_prep.py fit ROWS_DIR MV
        `tail_remap_eval.py fit` (same datasets, population, shrinkage, walk-forward) on the
        `TR = 0` rows of arm MV -> src/courtside/proj/tail_remap_params_c1r.json
    python research/c1r_ship_prep.py stretch OUT_JSON ON_FILE... [--off OFF_FILE...]
        `xf_ship_prep.py stretch` with the shipped reference = STRETCH_TR (the live `+tr` map)
    python research/c1r_ship_prep.py propsrun OUT.parquet
        (scratch DB copy) the 2026-27 graded props at the G3 lock-45 state, re-projected in the
        environment's arm (`cmd_propsrun`)
    python research/c1r_ship_prep.py props V0.parquet V1R.parquet [--json OUT]
        props log loss, v1r minus v0 (and vs the market), two-way SEs (report only)

Environment: COURTSIDE_DATA (bundles + DB), C1R_OUT (default var/tmp/c1rship/out). Heavy: Mac
scratch `courtside-compute/c1rship`.
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
OUT = Path(os.environ.get("C1R_OUT", ROOT / "var" / "tmp" / "c1rship" / "out"))


def say(m: str) -> None:
    print(f"[c1r] {time.strftime('%H:%M:%S')} {m}", flush=True)


# ============================================================================== harness

def _patch_tr_key(tr_key: str, season: str) -> None:
    from courtside.proj import tail_remap as TR
    orig = TR.cells

    def cells(s: str, copula: str | None = None) -> dict:   # no params for s -> tr_key's
        t = orig(s, copula)
        return t if t else orig(tr_key, copula)
    TR.cells = cells
    say(f"tail remap: seasons without params use {tr_key}'s ({len(cells(season))} cells for "
        f"{season})")


def cmd_harness(season: str, protocol: str, tag: str, tr_key: str | None,
                pq_lomo: str | None) -> None:
    import minutes_fix_eval as MFE
    import p8_e3_eval as P8

    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    from courtside.proj import tail_remap as TR
    M.GBM_PARAMS["num_threads"] = 1
    MFE._patch_capture()
    P8._patch_absent_capture()
    if pq_lomo:
        P8._patch_pq_lomo(pq_lomo)
    if tr_key and TR.TAIL_REMAP:
        _patch_tr_key(tr_key, season)
    say(f"arm {P.MODEL_VERSION}; copula {P.COPULA}; tail remap {TR.TAIL_REMAP} params "
        f"{TR.params_path().name}")
    conn = P8._conn()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "", tails=True)
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{protocol}_{season}_{P.MODEL_VERSION}{tag}"
    f = OUT / f"rows_{stem}.parquet"
    pl.to_parquet(f)
    if P8._ABS:
        pd.concat(P8._ABS, ignore_index=True).to_parquet(OUT / f"absent_{stem}.parquet")
    say(f"harness {season} {protocol} {P.MODEL_VERSION}{tag}: {len(pl)} rows, skipped "
        f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}")


# ============================================================================== fit

def cmd_fit(rows_dir: str, mv: str) -> None:
    """tail_remap_eval.cmd_fit on the c1r rows, written to the c1r params file."""
    import tail_remap_eval as TE

    from courtside.proj import tail_remap as TR
    frames = {}
    for proto, s in TE.DATASETS:
        f = Path(rows_dir) / f"rows_{proto}_{s}_{mv}.parquet"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        frames[(proto, s)] = TE.played_frame(pd.read_parquet(f))
        say(f"{proto} {s}: {len(frames[(proto, s)])} played rows ({f.name})")
    out = {"version": "tr-v1-c1r", "copula": "v1r", "levels": list(TR.LEVELS),
           "median": TR.MEDIAN, "min_bands": [list(b) for b in TR.MIN_BANDS],
           "shrink": TR.SHRINK, "arm": mv, "fit_pplay_min": TE.FIT_PPLAY_MIN,
           "method": "as tail_remap_params.json (tail_remap_eval.py fit), on copula v1r (+c1r) "
                     "sims with the tail remap OFF: q(v) = v-quantile of the played-conditional "
                     "PIT over played rows, per tier x E[min | plays] band, shrunk to the tier / "
                     "pooled; harness rows of seasons < S (walk-forward)",
           "seasons": {}}
    for S in TE.TARGETS:
        tr = [k for k in frames if k[1] < S]
        if not tr:
            continue
        fr = pd.concat([frames[k] for k in tr], ignore_index=True)
        fit = TR.fit_cells(fr)
        out["seasons"][S] = {"train": [f"{p}:{s}" for p, s in tr], **fit}
        say(f"{S}: {len(fr)} rows on {[f'{p}:{s}' for p, s in tr]}; tiers "
            + " ".join(f"{t} {[round(x, 4) for x in h]}" for t, h in fit["tiers"].items()))
    path = TR.TR_PATHS_BY_COPULA["v1r"]
    path.write_text(json.dumps(out, indent=1) + "\n")
    TR.cells.cache_clear()
    say(f"wrote {path}")


# ============================================================================== X factor

def cmd_stretch(out_json: str, on: list[str], off: list[str]) -> None:
    """xf_ship_prep stretch; the 'shipped' reference is STRETCH_TR (live with +tr on v0 sims)."""
    import xf_ship_prep as XP

    from courtside.proj.dfs import xfactor as XF
    XF.STRETCH = dict(XF.STRETCH_TR)
    XP.cmd_stretch(out_json, on, off)


# ============================================================================== props

def _ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)      # props_blend PREREG clip
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


PROP_DAYS = ("2026-10-20", "2026-10-23")


def cmd_propsrun(out_file: str) -> None:
    """Every final 2026-27 game with props (PROP_DAYS): its lock-45 run (`props_blend.lock45_run`,
    the G3 row rule) is re-projected in THIS process's arm (copula from the environment) on the
    same slate, at the run's created_at, with the same seed (`project_slate` default: slate x
    kind) and the live sim count, stored in the scratch DB under a new run id; then the props
    are scored at the same market state (`props.compare_game`, market_at = the original run's
    created_at) and graded as `props_blend.day_rows` -> OUT_FILE (parquet). The DB must be a
    scratch copy (COURTSIDE_DATA)."""
    from datetime import datetime

    from courtside import db
    from courtside.proj import pipeline as P
    from courtside.proj import props as PR
    from courtside.proj import props_blend as PB
    from courtside.proj import tail_remap as TR
    conn = db.connect()
    say(f"arm {P.MODEL_VERSION} copula {P.COPULA} tr {TR.TAIL_REMAP}; db {db.__name__}")
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT nba_day FROM games WHERE nba_day BETWEEN ? AND ? ORDER BY 1", PROP_DAYS)]
    rows, new_runs, cache = [], {}, {}
    stamp = time.strftime("%H%M%S")
    for day in days:
        games = PR.day_games(conn, day)
        states = PB._game_states(conn, day, [g["game_id"] for g in games]) if games else {}
        final = [g["game_id"] for g in games if states.get(g["game_id"]) == "final"]
        box = {}
        if final:
            ph = ",".join("?" * len(final))
            box = {(r[0], int(r[1])): r for r in conn.execute(
                f"SELECT game_id, player_id, min, pts, reb, ast, fg3m FROM player_games "
                f"WHERE game_id IN ({ph})", final)}
        for g in games:
            gid = g["game_id"]
            if gid not in final:
                continue
            run = PB.lock45_run(conn, gid, g.get("tipoff_utc"))
            if run is None or not PR.market_state(conn, gid, run["created_at"]):
                continue
            if run["run_id"] not in new_runs:
                at = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                rid = f"c1rp-{P.COPULA}-{stamp}-{run['run_id']}"
                t0 = time.time()
                P.project_slate(conn, run["slate_id"], at, run_id=rid, kind="lock45")
                nr = PR._rows(conn, "SELECT * FROM dfs_runs WHERE run_id=?", (rid,))[0]
                nr["created_at"] = run["created_at"]
                new_runs[run["run_id"]] = nr
                say(f"{day} slate {run['slate_id']} ({run['run_id']} {run['model_version']}) at "
                    f"{run['created_at']} -> {rid} {time.time() - t0:.0f}s")
            nr = new_runs[run["run_id"]]
            res = PR.compare_game(conn, g, run=nr, market_at=run["created_at"], cache=cache)
            dists = cache.get(("dists", nr["run_id"], gid), {})
            for r in res["rows"]:
                pid = int(r["player_id"])
                b = box.get((gid, pid))
                y = None
                if b is not None and (b[2] or 0) > 0:
                    vals = {"pts": b[3], "reb": b[4], "ast": b[5], "fg3m": b[6]}
                    vals["pra"] = (b[3] or 0) + (b[4] or 0) + (b[5] or 0)
                    act = float(vals[r["stat"]] or 0)
                    line = float(r["line"])
                    y = None if act == line else int(act > line)
                d = dists.get(pid)
                rows.append({"day": day, "game_id": gid, "player_id": pid, "stat": r["stat"],
                             "line": float(r["line"]), "captured_at": r["captured_at"],
                             "provider": r.get("provider"), "orig_run": run["run_id"],
                             "run_id": nr["run_id"], "model_version": nr.get("model_version"),
                             "dist_source": getattr(d, "source", None),
                             "our_mean": r.get("our_mean"),
                             "p_ours": PR.comparable(r["our_p_above"], r["our_p_push"]),
                             "p_mkt": r["mkt_p_above"], "hold": r["hold"], "y": y})
    df = pd.DataFrame(rows)
    df.to_parquet(out_file)
    say(f"{len(df)} prop rows ({df['y'].notna().sum() if len(df) else 0} graded), "
        f"{len(new_runs)} runs -> {out_file}")


def cmd_props(v0_file: str, v1r_file: str, out_json: str | None) -> None:
    """Paired props log loss on identical rows: v1r minus v0 (and each minus the market's no-vig
    chance), SEs two-way by game and player (`metrics.paired_cluster2`); by stat."""
    from courtside.proj import metrics as MT
    k = ["game_id", "player_id", "stat", "line", "captured_at"]
    a, b = pd.read_parquet(v0_file), pd.read_parquet(v1r_file)
    a = a.drop_duplicates(k)
    b = b.drop_duplicates(k)
    d = a.merge(b, on=k, suffixes=("_v0", "_v1r"))
    d = d[d["y_v0"].notna() & d["p_ours_v0"].notna() & d["p_ours_v1r"].notna()
          & d["p_mkt_v0"].notna()].copy()
    y = d["y_v0"].to_numpy(float)
    d["ll_v0"], d["ll_v1r"], d["ll_mkt"] = (_ll(d["p_ours_v0"], y), _ll(d["p_ours_v1r"], y),
                                            _ll(d["p_mkt_v0"], y))

    def diff(x: pd.DataFrame, c1: str, c0: str) -> dict:
        r = MT.paired_cluster2((x[c1] - x[c0]).to_numpy(float), x["game_id"].to_numpy(),
                               x["player_id"].to_numpy())
        return {"n": int(r["n"]), "mean": float(r["mean"]), "se": float(r["se"]),
                "z": float(r["mean"] / r["se"]) if r["se"] and r["se"] > 0 else None}

    rep = {"n_v0": int(len(a)), "n_v1r": int(len(b)), "n_paired_graded": int(len(d)),
           "games": int(d["game_id"].nunique()), "players": int(d["player_id"].nunique()),
           "dist_source": {"v0": d["dist_source_v0"].value_counts().to_dict(),
                           "v1r": d["dist_source_v1r"].value_counts().to_dict()},
           "ll": {"v0": float(d["ll_v0"].mean()), "v1r": float(d["ll_v1r"].mean()),
                  "market": float(d["ll_mkt"].mean())},
           "v1r_minus_v0": diff(d, "ll_v1r", "ll_v0"),
           "v0_minus_mkt": diff(d, "ll_v0", "ll_mkt"),
           "v1r_minus_mkt": diff(d, "ll_v1r", "ll_mkt"),
           "mean_p": {"v0": float(d["p_ours_v0"].mean()), "v1r": float(d["p_ours_v1r"].mean()),
                      "y": float(y.mean())},
           "by_stat": {}}
    for st, x in d.groupby("stat"):
        rep["by_stat"][st] = {"n": int(len(x)), "v1r_minus_v0": diff(x, "ll_v1r", "ll_v0"),
                              "ll_v0": float(x["ll_v0"].mean()),
                              "ll_v1r": float(x["ll_v1r"].mean()),
                              "ll_mkt": float(x["ll_mkt"].mean())}
    print(json.dumps(rep, indent=1, default=float))
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol")
    h.add_argument("--tag", default="")
    h.add_argument("--tr-key")
    h.add_argument("--pq-lomo")
    f = sub.add_parser("fit")
    f.add_argument("rows_dir")
    f.add_argument("mv")
    s = sub.add_parser("stretch")
    s.add_argument("out_json")
    s.add_argument("on", nargs="+")
    s.add_argument("--off", nargs="*", default=[])
    r = sub.add_parser("propsrun")
    r.add_argument("out_file")
    q = sub.add_parser("props")
    q.add_argument("v0")
    q.add_argument("v1r")
    q.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "propsrun":
        cmd_propsrun(a.out_file)
    elif a.cmd == "props":
        cmd_props(a.v0, a.v1r, a.json)
    elif a.cmd == "harness":
        cmd_harness(a.season, a.protocol, a.tag, a.tr_key, a.pq_lomo)
    elif a.cmd == "fit":
        cmd_fit(a.rows_dir, a.mv)
    elif a.cmd == "stretch":
        cmd_stretch(a.out_json, a.on, a.off)


if __name__ == "__main__":
    main()
