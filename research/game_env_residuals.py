"""T0.8: game-environment error structure for the DFS simulator (docs/PLAYERS.md §4.4-4.5).

Per season: sd of margin and total residuals for the walk-forward game models (rung-3 OOF
predictions, a Mac scratch artifact) and for the closing line (DB), and the residual
correlations the joint simulation must reproduce. Usage:

    uv run python research/game_env_residuals.py [--preds PATH/rung3_merged_preds.csv.gz]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from courtside import db


def _stats(x: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(x), "sd_margin": x.rm.std(), "sd_total": x.rt.std(), "bias_total": x.rt.mean(),
        "corr_m_t": np.corrcoef(x.rm, x.rt)[0, 1],
        "corr_absm_t": np.corrcoef(np.abs(x.margin), x.rt)[0, 1]})


def closing(conn) -> pd.DataFrame:
    d = pd.read_sql("""
        SELECT h.game_id, h.season, h.pts - a.pts AS margin, h.pts + a.pts AS total,
               c.home_spread_close, c.total_close
        FROM team_games h JOIN team_games a ON a.game_id = h.game_id AND a.is_home = 0
        JOIN closing_lines c ON c.game_id = h.game_id
        WHERE h.is_home = 1 AND h.season_type = 'Regular Season'
          AND c.home_spread_close IS NOT NULL AND c.total_close IS NOT NULL""", conn)
    d = d.drop_duplicates("game_id")
    d["rm"] = d.margin + d.home_spread_close
    d["rt"] = d.total - d.total_close
    return d.groupby("season").apply(_stats, include_groups=False)


def oof(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.model.isin(["gbm_market", "gbm_opinion"]) & d.total_pred.notna()]
    d["rm"] = d.margin - d.margin_pred
    d["rt"] = d.total - d.total_pred
    return d.groupby(["model", "season"]).apply(_stats, include_groups=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None, help="rung3_merged_preds.csv.gz (OOF predictions)")
    a = ap.parse_args()
    pd.set_option("display.width", 140)
    print("closing line (regular season):")
    print(closing(db.connect()).round(3).to_string())
    if a.preds:
        print("\nwalk-forward OOF (rung 3):")
        print(oof(a.preds).round(3).to_string())


if __name__ == "__main__":
    main()
