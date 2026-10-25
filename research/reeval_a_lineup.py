"""Re-evaluation workstream A: the lineup-known minutes weights (`+lu`, minutes.LINEUP).

Docs: docs/research/reeval-A.md.  Code under test: `minutes.project_arrays` with confirmed
starters (lock-10): an expected starter left out of the lineup is demoted to
min(adj, (1 - w_demote) adj + w_demote bench_target); an unexpected starter is lifted toward
w_target x team starter target + (1 - w_target) x his EWMA.  OFF = 0.5 / 0.5.

    python research/reeval_a_lineup.py fit S [S ...] [--seasons 3]   # grid fit on <= 3 seasons < S
    python research/reeval_a_lineup.py eval S [S ...]          # OFF vs ON on season S (fit < S)
    python research/reeval_a_lineup.py write                   # OUT/lineup_fit_*.json ->
                                                               #   src/courtside/proj/lineup_params.json
    python research/reeval_a_lineup.py all S [S ...] --eval S1,S2   # fit + write + eval, one process

Minutes level only (no DK sims), the T2.2 avail-oracle panels (`minutes_eval.prepare_season`)
with the actual starters as the lock-10 lineup (PLAYERS.md §7.1 L5), each season's own minutes
params (the live stack's, fit on seasons < that season): $COURTSIDE_DATA/proj/
bundle_<S>_v1_cx_mf_ep_ms_c1r.json (else _mf_ep_ms, _mf, plain; see `params_for`).  Objective:
MAE of E[min | plays] (`mean0`) on played rows of team-games with exactly 5 confirmed starters,
pooled over the fit seasons (up to --seasons most recent seasons < S that have a bundle; the
seasons' frames are built once per process); the grid includes the OFF point (0.5, 0.5).  Output: $REEVAL_A_OUT (var/tmp/reeval-a).
Heavy (one oracle panel per season, ~5-10 min each on the Mac): run on the Mac scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("REEVAL_A_OUT", ROOT / "var" / "tmp" / "reeval-a"))
GRID_D = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_T = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SUFFIXES = ("_v1_cx_mf_ep_ms_c1r", "_v1_cx_mf_ep_ms", "_v1_cx_mf_ms", "_v1_cx_mf", "_v1_cx",
            "_v1")


def say(m: str) -> None:
    print(f"[lu] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import config
    return sqlite3.connect(f"file:{config.data_dir() / 'courtside.db'}?mode=ro", uri=True)


PARAMS_OVERRIDE: str | None = None     # --params (smoke tests only: one file for every season)
MAX_DATES = 0                          # --max-dates (smoke tests only)
LEAN = False                           # --lean (smoke tests only, see season_frame)


def params_for(season: str):
    from courtside import config
    from courtside.proj import minutes as M
    if PARAMS_OVERRIDE:
        d = json.loads(Path(PARAMS_OVERRIDE).read_text())
        return M.MinutesParams(**d.get("minutes", d)), PARAMS_OVERRIDE
    pdir = config.data_dir() / "proj"
    for suf in SUFFIXES:
        f = pdir / f"bundle_{season}{suf}.json"
        if f.exists():
            d = json.loads(f.read_text())
            return M.MinutesParams(**d["minutes"]), str(f)
    raise FileNotFoundError(f"no bundle for {season} in {pdir}")


def prev_season(s: str, k: int = 1) -> str:
    y = int(s[:4]) - k
    return f"{y}-{(y + 1) % 100:02d}"


_FRAMES: dict = {}
_ERR: dict = {}            # (season, w_demote, w_target) -> (sum |error|, n)


def has_bundle(season: str) -> bool:
    try:
        params_for(season)
        return True
    except FileNotFoundError:
        return False


def fit_seasons_of(season: str, k: int) -> list[str]:
    """Up to `k` most recent seasons < S that have a bundle (oldest first; looks 8 back)."""
    got = [prev_season(season, j) for j in range(1, 9) if has_bundle(prev_season(season, j))]
    return sorted(got[:k])


def season_frame(conn, season: str):
    """(panel, ctx, params, src) of one season (cached), with the 5-starter team-game mask."""
    if season not in _FRAMES:
        _FRAMES[season] = _season_frame(conn, season)
    return _FRAMES[season]


def _season_frame(conn, season: str):
    from courtside.proj import minutes as M
    from courtside.proj import minutes_eval as ME
    prm, src = params_for(season)
    s0 = int(season[:4])
    h = M.load_history(conn, since=f"{s0 - 7}-{(s0 - 6) % 100:02d}")
    if M.uses_preseason(prm):
        M.attach_preseason(h, conn)
    if LEAN:                # smoke test: no 5-season training panel; shares from S-1 and S
        keep = h.shares
        h.shares = None
        panel = M.oracle_panel(h, M.eval_targets(h, [season]))
        prev = M.oracle_panel(h, M.eval_targets(h, [prev_season(season)]))
        h.shares = keep
        ctx = {"panel": panel, "shares": M.PairShares(pd.concat(
            [M.pair_events(prev), M.pair_events(panel)], ignore_index=True))}
    else:
        full = ME.prepare_season(conn, h, season)
        ctx = {"panel": full["panel"], "shares": full["shares"]}     # drop the training panel
        del full
    panel = ctx["panel"]
    if MAX_DATES:
        keep = sorted(panel["date"].unique())[:MAX_DATES]
        panel = panel[panel["date"].isin(keep)].reset_index(drop=True)
    n5 = panel.groupby("key")["act_started"].transform(lambda s: s.fillna(0).astype(bool).sum())
    panel = panel.assign(lineup5=(n5 == 5).to_numpy())
    say(f"{season}: params {src}; rows {len(panel)}, team-games with 5 starters "
        f"{panel.loc[panel['lineup5'], 'key'].nunique()} / {panel['key'].nunique()}")
    return panel, ctx, prm, src


_BP_CACHE: dict = {}


def _memo_base_and_pplay():
    """The GBM plan and P(plays) do not depend on the +lu weights: compute them once per
    (panel, params) instead of once per grid point."""
    from courtside.proj import minutes as M
    if getattr(M.base_and_pplay, "_lu_memo", False):
        return
    orig = M.base_and_pplay

    def memo(p, params, absent=None, *args, **kw):
        k = (id(p), id(params), args, tuple(sorted(kw.items())))
        if k not in _BP_CACHE:
            _BP_CACHE[k] = orig(p, params, absent, *args, **kw)
        return _BP_CACHE[k]

    memo._lu_memo = True
    M.base_and_pplay = memo


def project(panel, ctx, prm, w: tuple[float, float] | None):
    """run_projection with confirmed starters; w None = OFF code path, else +lu with weights."""
    from courtside.proj import minutes as M
    _memo_base_and_pplay()
    if w is None:
        M.set_lineup(False)
        M._LU_OVERRIDE = None
    else:
        M.set_lineup(True)
        M._LU_OVERRIDE = (float(w[0]), float(w[1]))
    try:
        r = M.run_projection(panel, prm, starters=True, shares=ctx["shares"], detail=True)
    finally:
        M.set_lineup(False)
        M._LU_OVERRIDE = None
    return r


def cells(r: pd.DataFrame) -> np.ndarray:
    exp = r["exp_start"].to_numpy(bool) if "exp_start" in r else np.zeros(len(r), bool)
    st = r["act_started"].fillna(0).to_numpy(bool)
    return np.where(exp & st, "expected_starter", np.where(exp & ~st, "demoted",
                    np.where(~exp & st, "unexpected_starter", "bench")))


def pop(r: pd.DataFrame) -> np.ndarray:
    return (r["played"].to_numpy(bool) & r["lineup5"].to_numpy(bool)
            & ~r["absent"].to_numpy(bool))


def cmd_fit(a) -> None:
    for season in a.season:
        fit_one(season, a.seasons)


def fit_one(season: str, k: int) -> None:
    conn = _conn()
    seasons = fit_seasons_of(season, k)
    if not seasons:
        say(f"{season}: no earlier season with a bundle, nothing to fit")
        return
    frames = {}
    for s in seasons:
        frames[s] = season_frame(conn, s)
    res = []
    base = {}
    for wd in GRID_D:
        for wt in GRID_T:
            err, n = 0.0, 0
            per = {}
            for s, (panel, ctx, prm, _) in frames.items():
                key = (s, wd, wt)
                if key not in _ERR:                 # each (season, weights) projected once
                    r = project(panel, ctx, prm, (wd, wt))
                    m = pop(r)
                    e = np.abs(r.loc[m, "mean0"].to_numpy(float)
                               - r.loc[m, "act_min"].to_numpy(float))
                    _ERR[key] = (float(e.sum()), int(len(e)))
                es, en = _ERR[key]
                per[s] = es / max(en, 1)
                err += es
                n += en
            res.append({"w_demote": wd, "w_target": wt, "mae": err / n, "n": n, "per": per})
            say(f"w_demote {wd} w_target {wt}: MAE {err / n:.4f} {per}")
    for s, (panel, ctx, prm, _) in frames.items():
        r = project(panel, ctx, prm, None)
        m = pop(r)
        base[s] = float(np.abs(r.loc[m, "mean0"] - r.loc[m, "act_min"]).mean())
    g = pd.DataFrame(res).sort_values(["mae", "w_demote", "w_target"])
    best = g.iloc[0]
    off = g[(g["w_demote"] == 0.5) & (g["w_target"] == 0.5)].iloc[0]
    out = {"season": season, "fit_seasons": seasons, "w_demote": float(best["w_demote"]),
           "w_target": float(best["w_target"]), "mae_fit": float(best["mae"]),
           "mae_grid_off": float(off["mae"]), "mae_off_code": base, "n": int(best["n"]),
           "params": {s: v[3] for s, v in frames.items()},
           "grid": g.drop(columns=["per"]).to_dict("records")}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"lineup_fit_{season}.json"
    f.write_text(json.dumps(out, indent=1))
    say(f"best ({out['w_demote']}, {out['w_target']}) MAE {out['mae_fit']:.4f} vs grid OFF "
        f"{out['mae_grid_off']:.4f} (OFF code {base}) -> {f}")


def cmd_eval(a) -> None:
    for season in a.season:
        eval_one(season)


def eval_one(season: str) -> None:
    from courtside.proj.metrics import paired_cluster
    fit = json.loads((OUT / f"lineup_fit_{season}.json").read_text())
    w = (fit["w_demote"], fit["w_target"])
    conn = _conn()
    panel, ctx, prm, src = season_frame(conn, season)
    r0 = project(panel, ctx, prm, None)
    r1 = project(panel, ctx, prm, w)
    assert (r0["key"].to_numpy() == r1["key"].to_numpy()).all()
    m = pop(r0)
    c = cells(r0)
    y = r0["act_min"].to_numpy(float)
    e0 = np.abs(r0["mean0"].to_numpy(float) - y)
    e1 = np.abs(r1["mean0"].to_numpy(float) - y)
    gid = r0["game_id"].to_numpy()
    lines = [f"## +lu minutes eval {season} (weights {w}, fit on {fit['fit_seasons']}; params "
             f"{src})", "", "| cell | n | MAE off | MAE on | diff | se | z | bias off | bias on |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ("all", "expected_starter", "demoted", "unexpected_starter", "bench"):
        s = m & ((c == name) if name != "all" else True)
        if not s.any():
            continue
        d = paired_cluster((e1 - e0)[s], gid[s])
        b0 = float((r0["mean0"].to_numpy(float) - y)[s].mean())
        b1 = float((r1["mean0"].to_numpy(float) - y)[s].mean())
        lines.append(f"| {name} | {int(s.sum())} | {e0[s].mean():.3f} | {e1[s].mean():.3f} | "
                     f"{d['mean']:+.4f} | {d['se']:.4f} | {d['mean'] / d['se']:+.1f} | "
                     f"{b0:+.2f} | {b1:+.2f} |")
    text = "\n".join(lines)
    print(text)
    (OUT / f"lineup_eval_{season}.md").write_text(text + "\n")


def cmd_write(a) -> None:
    from courtside.proj import minutes as M
    seasons = {}
    for f in sorted(OUT.glob("lineup_fit_*.json")):
        d = json.loads(f.read_text())
        seasons[d["season"]] = {"w_demote": d["w_demote"], "w_target": d["w_target"],
                                "fit_seasons": d["fit_seasons"], "mae_fit": d["mae_fit"],
                                "mae_off": d["mae_grid_off"], "n": d["n"]}
    doc = {"note": "minutes.LINEUP (+lu) weights per season S, fit on seasons < S by "
                   "research/reeval_a_lineup.py fit S (avail-oracle panels, actual starters)",
           "seasons": seasons}
    M.LU_PATH.write_text(json.dumps(doc, indent=1) + "\n")
    say(f"wrote {M.LU_PATH}: {sorted(seasons)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("fit", "eval", "write", "all"))
    ap.add_argument("season", nargs="*", help="fit / eval: seasons S; all: fit seasons")
    ap.add_argument("--eval", default="", help="all: comma-separated seasons to evaluate")
    ap.add_argument("--seasons", type=int, default=3, help="fit on up to N seasons < S")
    ap.add_argument("--params", default=None, help="smoke test: one params file for all seasons")
    ap.add_argument("--max-dates", type=int, default=0, help="smoke test: first N dates only")
    ap.add_argument("--lean", action="store_true", help="smoke test: lean panels (see code)")
    a = ap.parse_args(argv)
    global PARAMS_OVERRIDE, MAX_DATES, LEAN
    PARAMS_OVERRIDE, MAX_DATES, LEAN = a.params, a.max_dates, a.lean
    if a.cmd == "all":          # one process, season frames built once and shared
        cmd_fit(a)
        cmd_write(a)
        for season in [x for x in a.eval.split(",") if x]:
            eval_one(season)
        return
    {"fit": cmd_fit, "eval": cmd_eval, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
