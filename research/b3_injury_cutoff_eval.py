"""IMPROVEMENTS B3 (+ game-day rows): compare day-by-day 2025-26 replays of `gbm_market`.

Inputs are `research/game_calibration_replay.py` CSVs, `label=path`, the first one the
reference ("before"). For each arm: log loss, Brier, recalibration slope (date-clustered SE)
and the same for the de-vigged close on the same games; paired arm-minus-reference log-loss /
Brier (date-clustered SE) and slope difference (date-cluster bootstrap). Subsets: all
Regular Season games, and the games where both teams were priced from the injury report in
the reference arm. Leak check: games whose newest injury batch read is after the tip.

    uv run python research/b3_injury_cutoff_eval.py --db scratch.db \\
        --arm legacy=out/legacy_off.csv.gz --arm tip=out/tip_off.csv.gz \\
        --arm tip_gd=out/tip_on.csv.gz --out out/b3_eval.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game_calibration import calib, game_table, paired, slope_diff_boot  # noqa: E402


def load(path: str, games: pd.DataFrame, model: str) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"game_id": str})
    d["game_id"] = d["game_id"].str.zfill(10)
    d = d[d["model"] == model]
    d = d.merge(games, on="game_id", how="inner")
    d["y"] = d["home_win"].astype(float)
    d["date"] = pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d")
    d["post_tip"] = (d["max_captured_at"].notna() & d["tip"].notna()
                     & (d["max_captured_at"].astype(str) > d["tip"].astype(str)))
    return d.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--arm", action="append", required=True, help="label=path; first = ref")
    ap.add_argument("--model", default="gbm_market")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    games = game_table(conn)
    arms = {}
    for s in a.arm:
        lab, _, path = s.partition("=")
        arms[lab] = load(path, games, a.model)
    ref_lab = next(iter(arms))
    ref = arms[ref_lab]
    rep = (ref["basis_home"] == "injury_report") & (ref["basis_away"] == "injury_report")
    keys = {"all": ref["game_id"][ref["season_type"] == "Regular Season"],
            "report": ref["game_id"][(ref["season_type"] == "Regular Season") & rep],
            "report_all_types": ref["game_id"][rep]}
    res: dict = {"model": a.model, "ref": ref_lab, "arms": {}}
    for lab, d in arms.items():
        res["arms"][lab] = {"rows": int(len(d)), "post_tip_games": int(d["post_tip"].sum()),
                            "post_tip_games_rs": int(d["post_tip"][d["season_type"]
                                                                  == "Regular Season"].sum()),
                            "report_games_rs": int(((d["basis_home"] == "injury_report")
                                                    & (d["basis_away"] == "injury_report")
                                                    & (d["season_type"] == "Regular Season"))
                                                   .sum()),
                            "subsets": {}}
    for sub, ids in keys.items():
        r0 = ref.set_index("game_id").loc[ids.to_numpy()].reset_index()
        for lab, d in arms.items():
            di = d.set_index("game_id").loc[ids.to_numpy()].reset_index()
            assert (di["y"].to_numpy() == r0["y"].to_numpy()).all()
            o = {"model": calib(di["y"], di["p"], di["date"])}
            m = di["p_mkt"].notna()
            o["market"] = calib(di["y"][m], di["p_mkt"][m], di["date"][m])
            o["model_on_mkt"] = calib(di["y"][m], di["p"][m], di["date"][m])
            o["mean_abs_dp_vs_ref"] = float(np.mean(np.abs(di["p"] - r0["p"])))
            if lab != ref_lab:
                o["vs_ref"] = paired(di["y"].to_numpy(), di["p"].to_numpy(),
                                     r0["p"].to_numpy(), di["date"].to_numpy())
                o["slope_vs_ref"] = slope_diff_boot(di["y"], di["p"], r0["p"], di["date"],
                                                    reps=a.reps)
            res["arms"][lab]["subsets"][sub] = o
            c = o["model"]
            line = (f"{sub:>16} {lab:>10} n={c['n']:5d} LL {c['log_loss']:.4f} "
                    f"Brier {c['brier']:.4f} slope {c['slope']:.3f} ({c['slope_se']:.3f}) "
                    f"| close slope {o['market']['slope']:.3f} ({o['market']['slope_se']:.3f})"
                    f" LL {o['market']['log_loss']:.4f} n={o['market']['n']}")
            if "vs_ref" in o:
                v, sd = o["vs_ref"], o["slope_vs_ref"]
                line += (f" | dLL {v['d_ll']:+.5f} ({v['d_ll_se']:.5f}) dBrier "
                         f"{v['d_brier']:+.5f} ({v['d_brier_se']:.5f}) dslope "
                         f"{sd['diff']:+.3f} ({sd['se']:.3f}) |dp| {o['mean_abs_dp_vs_ref']:.4f}")
            print(line)
    for lab, o in res["arms"].items():
        print(f"{lab}: post-tip batch read in {o['post_tip_games']} games "
              f"({o['post_tip_games_rs']} RS); RS report games {o['report_games_rs']}")
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
