"""Re-evaluation workstream B (docs/research/reeval-B.md): per-minute ablation of the context
hypotheses H1 (F8 / F8r / F8c player vs opponent / coach), H2 (F1 opponent availability, retest),
H3 (F7 referee crews) and H4 (F3x tracking-dashboard opportunity), staged like
research/context_ablation2.py, plus the H1 / H3 signal diagnostics.

Stages (per test season S; every layer fit walk-forward on the 6 regular seasons < S; one
LightGBM thread; per-minute NB log-lik and DK CRPS / MAE given actual minutes, paired, SE = the
larger of the two-way (game x player-season) and date-clustered SEs, as every earlier ablation):

  vc     H1: placebo-corrected variance component of the player x opponent effect (and x coach)
         per stat on the regular-season rows of seasons < S, the implied pseudo-count, split-half
         reliability (real vs placebo pairs) on the same rows; H3: the official effect variance
         per target (fta, pf, pace). No model fit.
  live   on the live layer P+F2: P+F2, +F8, +F8r, +F8c, +F7, +F1
  ext    on the opt-in P+F2+F3+F6: P+F2+F3+F6, +F8, +F8r, +F7, +F1 (+F3x with --f3x)
  all    vc + live + ext

    COURTSIDE_DATA=<scratch>/data python research/reeval_b_ablation.py 2024-25 --stage all \
        --db <scratch>/data/courtside.db --out <scratch>/out

--sample (geekom1 validation only): the last 2 training seasons + S, a 1/K player subsample
(hash of player_id) for the fits, a faster learning rate and small leaves. Its numbers are a
pipeline check, not evidence.

Writes OUT/reeval_b_<stage>_<S>.json and .txt. Refuses to run with the repo's var/ as the data
dir (feature caches are written under <data>/proj/features).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from courtside import config
from courtside.proj import context as CX
from courtside.proj import features as FX
from courtside.proj.features import opp_history as F8
from courtside.proj.features import referees as F7
from courtside.proj.features import tracking_ext as F3X
from courtside.proj.features.common import season_label

LIVE = ("P", "F2")
EXT = ("P", "F2", "F3", "F6")


def variants(stage: str, f3x: bool) -> dict:
    if stage == "live":
        base = LIVE
        add = ("F8", "F8r", "F8c", "F7", "F1")
    else:
        base = EXT
        add = ("F8", "F8r", "F7", "F1") + (("F3x",) if f3x else ())
    name = "+".join(base)
    v = {name: base}
    for g in add:
        v[f"{name}+{g}"] = base + (g,)
    return v


def pairs_for(v: dict) -> list:
    names = list(v)
    return [(n, names[0]) for n in names[1:]]


def _say(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load_frame(conn, season: str, sample: int, log) -> pd.DataFrame:
    sp = CX.state_params_for(conn, season)
    fr = FX.build_frame(conn, season, sp, n_train=6, log=log)
    if sample:
        keep = fr["season"] >= season_label(int(season[:4]) - 2)
        fr = fr[keep].reset_index(drop=True)
    return fr


def subsample(fr: pd.DataFrame, k: int) -> pd.DataFrame:
    """1/k of the players (all their rows), deterministic."""
    if k <= 1:
        return fr
    h = pd.util.hash_pandas_object(fr["player_id"], index=False).to_numpy() % k
    return fr[h == 0].reset_index(drop=True)


def stage_vc(conn, fr: pd.DataFrame, season: str) -> dict:
    t0 = time.time()
    rs = (fr["season_type"] == "Regular Season").to_numpy() & (fr["season"] < season).to_numpy()
    d = fr[rs].reset_index(drop=True)
    out: dict = {"season": season, "n_rows": int(len(d)),
                 "seasons": sorted(d["season"].unique().tolist())}
    key = F8._franchise(d["opp"], d["season"])
    out["h1_opponent"] = F8.variance_components(d, key)
    out["h1_split_half"] = F8.split_half(d, key, min_games=6)
    ck = F8.coach_keys(conn, d)
    out["h1_coach"] = F8.variance_components(d, ck)
    # how much of a pair's raw residual the shrinkage keeps, at typical evidence
    sh = {}
    for s, v in out["h1_opponent"].items():
        t2, c, eb = max(v["tau2"], 0.0), v["c"], v["ebar"]
        sh[s] = {f"{g}g": (t2 / (t2 + c / (g * eb)) if t2 > 0 else 0.0) for g in (4, 8, 12)}
    out["h1_shrink_weight"] = sh
    # H3: official effect variance per target (games of seasons < S)
    try:
        seasons = sorted(fr["season"].unique())
        offs = F7.load_officials([season_label(int(seasons[0][:4]) - 1)] + list(seasons))
        gt = F7.game_targets(conn, fr)
        gt["season"] = [season_label(int(s)) for s in gt["sidx"]]
        out["h3_officials"] = F7.fit_params(gt, offs, (gt["season"] < season).to_numpy())
        out["h3_coverage"] = {"games": int(gt["game_id"].nunique()),
                              "with_crew": int(gt["game_id"].isin(set(offs["game_id"])).sum())}
    except Exception as e:  # noqa: BLE001 - report, keep the H1 numbers
        out["h3_error"] = f"{type(e).__name__}: {e}"
    out["h4_coverage"] = F3X.coverage(conn).to_dict("records")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def format_vc(v: dict) -> str:
    L = [f"== vc {v['season']} (rows of {v['seasons'][0]}..{v['seasons'][-1]}, "
         f"n={v['n_rows']}; {v['seconds']}s)",
         "  H1 player x opponent: tau2 (SE, z), sd of the effect, NB calibration c, "
         "pseudo-count k (expected counts / games); split-half r real vs placebo (pairs >= 6 g)"]
    for s, x in v["h1_opponent"].items():
        sh = v["h1_split_half"][s]
        L.append(f"    {s:5s} tau2 {x['tau2']:+.5f} ({x['se']:.5f}, z {x['z']:+.1f})  "
                 f"sd {x['sd_effect']:.3f}  c {x['c']:.2f}  k {x['k']:.0f} / "
                 f"{x['k_games']:.0f} g   split-half r {sh['real']['r']:+.3f} "
                 f"(SB {sh['real']['sb']:+.3f}, n {sh['real']['n_pairs']}) vs placebo "
                 f"{sh['placebo']['r']:+.3f}   weight at 4/8/12 g "
                 + "/".join(f"{w:.2f}" for w in v["h1_shrink_weight"][s].values()))
    L.append("  H1 player x opponent head coach:")
    for s, x in v["h1_coach"].items():
        L.append(f"    {s:5s} tau2 {x['tau2']:+.5f} ({x['se']:.5f}, z {x['z']:+.1f})  "
                 f"sd {x['sd_effect']:.3f}")
    if "h3_officials" in v:
        L.append(f"  H3 officials ({v['h3_coverage']['with_crew']}/{v['h3_coverage']['games']} "
                 "games with a crew):")
        for q, x in v["h3_officials"].items():
            z = x["raw_tau2"] / x["se"] if x["se"] and x["se"] > 0 else float("nan")
            L.append(f"    {q:5s} tau2 {x['raw_tau2']:+.6f} (SE {x['se']:.6f}, z {z:+.1f}) "
                     f"sd {x.get('sd_effect', 0):.4f}  c {x['c']:.2f}  blocks "
                     f"{x.get('n_blocks', 0)}")
    else:
        L.append(f"  H3: {v.get('h3_error')}")
    L.append(f"  H4 player_tracking_ext coverage: {v['h4_coverage'] or 'no table / no rows'}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("season")
    ap.add_argument("--stage", default="all", choices=("vc", "live", "ext", "all"))
    ap.add_argument("--f3x", action="store_true", help="add +F3x to the ext stage")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=0,
                    help="validation: keep 1/K of players, 2 training seasons, fast LightGBM")
    a = ap.parse_args()
    if config.data_dir().resolve() == (config.REPO_ROOT / "var").resolve():
        print("refusing to run with the repo's var/ as the data dir: set COURTSIDE_DATA to a "
              "scratch dir (feature caches are written there)", file=sys.stderr)
        return 2
    FX.set_context_b(True)
    CX.LGB_PARAMS["num_threads"] = a.threads
    if a.sample:
        CX.LGB_PARAMS.update(learning_rate=0.15, min_data_in_leaf=300,
                             min_sum_hessian_in_leaf=10.0)
    p = a.db or str(config.db_path())
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    out = Path(a.out) if a.out else config.data_dir() / "reports" / "reeval_b"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_sample" if a.sample else ""
    t0 = time.time()
    fr = load_frame(conn, a.season, a.sample, _say)
    _say(f"frame {a.season}: {len(fr)} rows x {fr.shape[1]} cols")
    stages = ("vc", "live", "ext") if a.stage == "all" else (a.stage,)
    for st in stages:
        t1 = time.time()
        if st == "vc":
            v = stage_vc(conn, fr, a.season)
            txt = format_vc(v)
        else:
            vs = variants(st, a.f3x)
            groups = sorted({g for gs in vs.values() for g in gs
                             if g in FX.EXTENSION_GROUPS or g in FX.REEVAL_GROUPS})
            ff = FX.extend_frame(conn, fr, groups, season=a.season, log=_say) if groups else fr
            ff = subsample(ff, a.sample)
            v = CX.run_ablation(conn, a.season, frame=ff, variants=vs, pairs=pairs_for(vs),
                                save_models=False, log=_say)
            v["stage"], v["sample"] = st, a.sample
            txt = CX.format_ablation(v)
            del ff
        (out / f"reeval_b_{st}_{a.season}{tag}.json").write_text(
            json.dumps(v, indent=1, default=float))
        (out / f"reeval_b_{st}_{a.season}{tag}.txt").write_text(txt + "\n")
        print(txt, flush=True)
        _say(f"stage {st} {a.season} done in {time.time() - t1:.0f}s")
    _say(f"all done {a.season} in {time.time() - t0:.0f}s (pid {os.getpid()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
