"""Minutes restriction (`COURTSIDE_MIN_RESTRICT`, `+rs`; IMPROVEMENTS.md P7): corpus, harness,
evaluation, OFF identity and news latency.

Docs: docs/research/min-restrict.md. Code under test: `sources/espn_restrict.py`,
`proj/min_restrict.py`, the `rs_cap` lines of `proj/minutes.py`, the `+rs` lines of
`proj/pipeline.py`, `proj/news_latency.py`.

    python research/min_restrict_eval.py flags RAW_DIR
        parse every archived ESPN injuries snapshot -> OUT/flags.json (unique comments, flags)
    python research/min_restrict_eval.py harness LO HI
        live-like lock-45 rows (`backtest.live_like_day`, P(play) "shipped", 4,000 sims) of the
        2026-27 dates in [LO, HI], the arm in the environment -> OUT/rows_<MV>.parquet and
        OUT/flagged_<MV>.json (every flagged candidate, ON or OFF: the flags are computed in both)
    python research/min_restrict_eval.py compare OFF ON
        minutes MAE / DK CRPS on flagged vs unflagged rows, ON - OFF (two-way date x player SE)
    python research/min_restrict_eval.py same A B
        OFF identity: two rows parquets (e.g. HEAD tree vs working tree) equal bit for bit
    python research/min_restrict_eval.py latency [SINCE]
        news -> next re-run / lock distribution (`proj/news_latency.py`) on COURTSIDE_DATA's DB

Environment: COURTSIDE_DATA (bundle + DB + raw/espn), RS_OUT (default var/tmp/p7/out). Heavy: Mac
scratch (`courtside-compute/rs`).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RS_OUT", ROOT / "var" / "tmp" / "p7" / "out"))
SEASON = "2026-27"


def say(m: str) -> None:
    print(f"[rs] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import db
    return db.connect()


# ============================================================================== corpus

def cmd_flags(raw_dir: str) -> None:
    from courtside.sources import espn_restrict as ER
    files = sorted(glob.glob(str(Path(raw_dir) / "*" / "injuries__*.json.gz")))
    uniq: dict[tuple, dict] = {}
    per_snap = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            payload = json.load(fh)
        n_flag = 0
        for tb in payload.get("injuries") or []:
            for inj in tb.get("injuries") or []:
                name = (inj.get("athlete") or {}).get("displayName")
                k = (name, inj.get("shortComment") or "", inj.get("longComment") or "")
                if k not in uniq:
                    r = ER.parse_comment(k[1], k[2], name)
                    uniq[k] = {"name": name, "date": inj.get("date"), "status": inj.get("status"),
                               "flag": r.flag, "cap": r.cap, "phrase": r.phrase}
                n_flag += int(uniq[k]["flag"])
        per_snap.append({"file": Path(f).name, "n_flagged": n_flag})
    fl = [v for v in uniq.values() if v["flag"]]
    out = {"snapshots": len(files), "first": Path(files[0]).name if files else None,
           "last": Path(files[-1]).name if files else None, "unique_comments": len(uniq),
           "flagged_comments": len(fl), "flagged_players": sorted({v["name"] for v in fl}),
           "with_cap": sum(v["cap"] is not None for v in fl), "flagged": fl,
           "per_snapshot_flagged": per_snap}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flags.json").write_text(json.dumps(out, indent=1))
    say(f"{len(files)} snapshots, {len(uniq)} unique comments, {len(fl)} flagged "
        f"({out['with_cap']} with a cap): {out['flagged_players']}")


# ============================================================================== harness

def _install_hooks(conn, flagged: list) -> None:
    """`backtest.live_like_day` calls `pipeline.build_games` directly (not `project_games`), so
    the run's `at` is taken from its `availability.p_play_slate` call and the +rs caps are added
    here, exactly as `project_games` does on the live path. Flags are recorded ON and OFF."""
    from courtside.proj import availability as AV
    from courtside.proj import min_restrict as MRS
    from courtside.proj import pipeline as P
    state = {"at": None}
    slate0, build0 = AV.p_play_slate, P.build_games

    def slate(conn_, day, at, *a, **k):
        state["at"] = at
        return slate0(conn_, day, at, *a, **k)

    def build(ctx, b, conn_, day, games, panel, p_avail, env, starters=None, pp_fix=None,
              rs_caps=None):
        caps = MRS.restrict_caps(conn_, panel, state["at"])
        if caps:
            fl = {(f["key"], f["player_id"]): f for f in
                  MRS.flagged_players(conn_, state["at"], panel)}
            for (k, pid), lim in caps.items():
                f = fl.get((k, pid), {})
                flagged.append({"date": day.isoformat(), "key": k, "player_id": int(pid),
                                "lim": lim, "cap": f.get("cap"), "name": f.get("player_name"),
                                "news_at": f.get("news_at"), "at": P._iso(state["at"]),
                                "phrase": f.get("phrase")})
        return build0(ctx, b, conn_, day, games, panel, p_avail, env, starters, pp_fix=pp_fix,
                      rs_caps=(caps if MRS.MIN_RESTRICT else None))
    AV.p_play_slate = slate
    P.build_games = build


def cmd_harness(lo: str, hi: str) -> None:
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    flagged: list = []
    if "rs_caps" in P.build_games.__code__.co_varnames:     # HEAD trees have no +rs hook
        _install_hooks(conn, flagged)
    t0 = time.time()
    st = B.Settings(protocol="live", n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="shipped")
    pl, _, meta = B.run_season(conn, SEASON, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("RS_TAG", "")
    f = OUT / f"rows_{P.MODEL_VERSION}{tag}.parquet"
    pl.to_parquet(f)
    (OUT / f"flagged_{P.MODEL_VERSION}{tag}.json").write_text(json.dumps(flagged, indent=1))
    say(f"harness {lo}..{hi} {P.MODEL_VERSION}: {len(pl)} rows, {len(flagged)} flagged, "
        f"skipped {meta['skipped']}, {time.time() - t0:.0f}s -> {f}")


# ============================================================================== evaluation

def _played(d: pd.DataFrame) -> pd.Series:
    return (d["minutes"] > 0) & (d["p_play"] >= 0.05)


def cmd_compare(off_path: str, on_path: str) -> None:
    from courtside.proj.metrics import paired_cluster2
    a, b = pd.read_parquet(off_path), pd.read_parquet(on_path)
    fl = json.loads(Path(on_path.replace("rows_", "flagged_").replace(".parquet", ".json"))
                    .read_text())
    flag = {(f["date"], int(f["player_id"])) for f in fl}
    k = ["date", "player_id"]
    m = a.merge(b, on=k, suffixes=("_off", "_on"))
    m["flag"] = [(d, int(p)) in flag for d, p in zip(m["date"], m["player_id"])]
    m["played"] = (m["minutes_off"] > 0) & (m["p_play_off"] >= 0.05)
    m["ae_min_off"] = (m["min_played_off"] - m["minutes_off"]).abs()
    m["ae_min_on"] = (m["min_played_on"] - m["minutes_on"]).abs()
    m["ae_exp_off"] = (m["exp_min_off"] - m["minutes_off"]).abs()
    m["ae_exp_on"] = (m["exp_min_on"] - m["minutes_on"]).abs()
    out = {"n_rows": int(len(m)), "n_flagged_rows": int(m["flag"].sum()),
           "dates": sorted(m["date"].unique().tolist()), "cells": {}}
    lines = ["| cell | n | min MAE OFF | ON | ON-OFF (SE) | DK CRPS OFF | ON | ON-OFF (SE) |",
             "|---|---|---|---|---|---|---|---|"]
    for lab, sel in (("flagged, played", m["flag"] & m["played"]),
                     ("flagged, all", m["flag"]),
                     ("unflagged, played", ~m["flag"] & m["played"]),
                     ("all rows", np.ones(len(m), bool))):
        s = m[sel]
        if not len(s):
            continue
        c1, c2 = s["date"].to_numpy(), s["player_id"].to_numpy()
        dm = paired_cluster2((s["ae_min_on"] - s["ae_min_off"]).to_numpy(), c1, c2)
        cr = s.dropna(subset=["crps_off", "crps_on"])
        dc = paired_cluster2((cr["crps_on"] - cr["crps_off"]).to_numpy(),
                             cr["date"].to_numpy(), cr["player_id"].to_numpy())
        cell = {"n": int(len(s)), "min_mae_off": float(s["ae_min_off"].mean()),
                "min_mae_on": float(s["ae_min_on"].mean()), "d_min": dm,
                "crps_off": float(cr["crps_off"].mean()) if len(cr) else None,
                "crps_on": float(cr["crps_on"].mean()) if len(cr) else None, "d_crps": dc}
        out["cells"][lab] = cell

        def f(x, nd=3):
            return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
        lines.append(f"| {lab} | {cell['n']} | {f(cell['min_mae_off'])} | {f(cell['min_mae_on'])} "
                     f"| {f(dm['mean'])} ({f(dm['se'])}) | {f(cell['crps_off'])} | "
                     f"{f(cell['crps_on'])} | {f(dc['mean'])} ({f(dc['se'])}) |")
    rows = m[m["flag"]][["date", "player_id", "team_off", "status_off", "minutes_off",
                         "min_played_off", "min_played_on", "exp_min_off", "exp_min_on",
                         "mean_off", "mean_on", "y_off", "crps_off", "crps_on"]]
    names = {(f["date"], int(f["player_id"])): (f["name"], f["lim"], f["cap"]) for f in fl}
    rows = rows.assign(name=[names.get((d, int(p)), ("", None, None))[0]
                             for d, p in zip(rows["date"], rows["player_id"])],
                       lim=[names.get((d, int(p)), ("", None, None))[1]
                            for d, p in zip(rows["date"], rows["player_id"])])
    out["flagged_rows"] = json.loads(rows.to_json(orient="records"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "compare.md").write_text("\n".join(lines) + "\n\n" + rows.to_string(index=False)
                                    + "\n")
    print("\n".join(lines))
    print(rows.to_string(index=False))
    # identity outside the flagged teams' games
    keys = {(f["date"], f["key"].split("|")[0]) for f in fl}
    other = m[[(d, g) not in keys for d, g in zip(m["date"], m["game_id_off"])]]
    same = bool(np.array_equal(other["mean_off"].to_numpy(), other["mean_on"].to_numpy())
                and np.array_equal(other["crps_off"].to_numpy(), other["crps_on"].to_numpy(),
                                   equal_nan=True))
    say(f"rows outside flagged games identical ON vs OFF: {same} (n {len(other)})")


def cmd_same(a_path: str, b_path: str) -> None:
    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    k = ["date", "game_id", "player_id"]
    a, b = a.sort_values(k).reset_index(drop=True), b.sort_values(k).reset_index(drop=True)
    ok = list(a.columns) == list(b.columns) and len(a) == len(b)
    diffs = []
    if ok:
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            if x.dtype.kind == "f":
                eq = np.array_equal(x, y, equal_nan=True)
            else:
                sx, sy = pd.Series(x), pd.Series(y)
                eq = bool(((sx.isna() & sy.isna()) | (sx.astype(str) == sy.astype(str))).all())
            if not eq:
                diffs.append(c)
    say(f"same rows/cols: {ok}, n {len(a)} vs {len(b)}, differing columns: {diffs}")
    sys.exit(0 if ok and not diffs else 1)


def cmd_latency(since: str | None) -> None:
    from courtside.proj import news_latency as NL
    conn = _conn()
    ev = NL.events(conn, since or NL.SEASON_START)
    s = NL.summary(ev)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "latency_events.csv", index=False)
    (OUT / "latency.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"flags": lambda: cmd_flags(args[0]),
     "harness": lambda: cmd_harness(args[0], args[1]),
     "compare": lambda: cmd_compare(args[0], args[1]),
     "same": lambda: cmd_same(args[0], args[1]),
     "latency": lambda: cmd_latency(args[0] if args else None)}[cmd]()
