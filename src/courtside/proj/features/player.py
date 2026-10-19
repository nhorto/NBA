"""P: player descriptors from the pre-game EB state (docs/PLAYERS.md Phase 2F).

Not a context group of its own: it is the minimum a learned layer needs to know *who* a context
acts on (a centre vs a guard facing a team missing its rim protector), so every non-base
variant of the ablation carries it, and F1 / F2 are measured on top of it. On its own it
measures how much the EB state is mis-calibrated by role (T2.7's tier bias).

  p_grp        position group (0 G, 1 F, 2 C, 3 unknown)
  p_mpg        minutes-per-game state
  p_geff       log(1 + decayed games)
  p_lrel_<s>   log(rate / its cell prior): how far the player's rate sits from his cell
  p_ln_n_<s>   log(1 + effective minutes of evidence for the rate)
  p_lrel_dk    the same for DK per minute
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REL_STATS = ("pts", "fg3m", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fta")
EXTRA = {"pts": ("fga", "fta", "fg3m"), "reb": ("oreb", "dreb")}
COLS = (("p_grp", "p_mpg", "p_geff", "p_lrel_dk")
        + tuple(f"p_lrel_{s}" for s in REL_STATS) + tuple(f"p_ln_n_{s}" for s in REL_STATS))


def row_features(rates: pd.DataFrame) -> pd.DataFrame:
    """From any frame with the state columns (walk_forward / rate_frame): r_/p_/n_<s>, r_dk,
    p_dk, mpg, g_eff, grp."""
    out = {"p_grp": rates["grp"].to_numpy(int).astype(float),
           "p_mpg": rates["mpg"].to_numpy(float),
           "p_geff": np.log1p(np.maximum(rates["g_eff"].to_numpy(float), 0.0))}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["p_lrel_dk"] = np.log(rates["r_dk"].to_numpy(float) / rates["p_dk"].to_numpy(float))
        for s in REL_STATS:
            out[f"p_lrel_{s}"] = np.log(np.maximum(rates[f"r_{s}"].to_numpy(float), 1e-6)
                                        / np.maximum(rates[f"p_{s}"].to_numpy(float), 1e-6))
            out[f"p_ln_n_{s}"] = np.log1p(np.maximum(rates[f"n_{s}"].to_numpy(float), 0.0))
    return pd.DataFrame(out, index=rates.index)


def columns(stat: str) -> list[str]:
    own = (stat,) + EXTRA.get(stat, ())
    return (["p_grp", "p_mpg", "p_geff", "p_lrel_dk"] + [f"p_lrel_{s}" for s in own]
            + [f"p_ln_n_{stat}"])


def monotone(stat: str) -> dict:
    return {}


CATEGORICAL = ("p_grp",)
