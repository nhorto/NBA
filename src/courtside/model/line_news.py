"""News since the line (IMPROVEMENTS G1): COURTSIDE_NEWS_SINCE_LINE, default OFF.

`gbm_market` is fit on closing lines, which already price the day's injury news; live, its
market input is the newest captured line, which has not seen news published after it was
captured. ON, the headline model's line is moved by what that news is worth before the model
sees it:

    shift = beta * (opinion margin with the lineups priced now
                    - opinion margin with the lineups as the reports stood at line capture)

and the same for the total with the opinion total. "Opinion" is `gbm_opinion`, which has no
market input, so the difference is purely the absence value of the change in the reports.
`beta` is the slope of ESPN open->close line moves on that difference
(`research/g1_line_news_eval.py`, docs/research/line-timing.md).

Rules (`predict.price_day` applies them):

* only snapshot lines with a capture time move; a closing line (replays, backfills) already
  prices the news, so its shift is 0;
* the capture-time state is `injuries.availability` at min(capture time, the injury cutoff
  the game is priced with): the same function as the priced state, so with no news since the
  line the two lineups and the shift are exactly 0;
* guard: a side counts only if the newest batch lists its team at both times; otherwise its
  current lineup is used at capture too (a team that had not filed yet would otherwise look
  healthy at capture and turn every standing absence into "news");
* the opinion model and every non-market scorer see the unshifted features.

ON tags the market rows' model_version with `TAG` and stores `inputs.news_since_line`. OFF
(default) `price_day` never calls into this module: bit-identical to before.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ENV = "COURTSIDE_NEWS_SINCE_LINE"
DEFAULT = False
TAG = "+nsl"

# Fitted on 2025-26 (879 games, ESPN open -> close moves on the opinion model's news value,
# official injury archive 22 Dec 2025 -> 13 Jun 2026, open proxy = the reports at 03:00Z the
# evening before, guarded lineups): margin slope 0.447 (date-clustered SE 0.046). The total
# slope is -0.79 (SE 0.94), indistinguishable from 0 (the opinion total barely moves with
# absences: mean |delta| 0.07 pts), so totals are not shifted. docs/research/line-timing.md.
BETA_MARGIN = 0.45
BETA_TOTAL = 0.0


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


ON = _env_flag(ENV, DEFAULT)


def set_news_since_line(on: bool) -> None:
    """Flip the switch at run time (evaluation scripts, tests)."""
    global ON
    ON = bool(on)


def tagged(version_id: str | None) -> str | None:
    """`version_id` + TAG (None stays None: an unregistered model has no version to tag)."""
    return f"{version_id}{TAG}" if version_id else None


def capture_time(line: dict | None) -> str | None:
    """When the line the model is fed was captured: snapshot lines only (a closing line has no
    capture time and already prices the news)."""
    if not line or line.get("kind") != "snapshot":
        return None
    return line.get("captured_at") or None


def guarded(now: np.ndarray, cap: np.ndarray, listed_now: np.ndarray,
            listed_cap: np.ndarray) -> np.ndarray:
    """Capture-time lineup strengths where the team was listed at both times, else the current
    ones (no news for that side). NaN capture strengths also fall back to the current ones."""
    now, cap = np.asarray(now, float), np.asarray(cap, float)
    ok = np.asarray(listed_now, bool) & np.asarray(listed_cap, bool) & np.isfinite(cap)
    return np.where(ok, cap, now)


def opinion_values(scorer, G: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(margin, total) of a fitted no-market scorer on feature rows G (no calibration)."""
    from . import gbm
    m = scorer.margin_at(G)
    if "total" in scorer.heads_:
        fit, k, _ = scorer.heads_["total"]
        t = gbm._head_predict(fit, G, k, scorer.market)
    else:
        t = np.zeros(len(G))
    return m, t


def shift(scorer, G_now: pd.DataFrame, G_cap: pd.DataFrame, beta_margin: float = None,
          beta_total: float = None) -> dict:
    """{"delta_margin", "delta_total"}: opinion value now minus at capture (home margin points,
    game total points), and {"shift_margin", "shift_total"}: beta times those."""
    bm = BETA_MARGIN if beta_margin is None else beta_margin
    bt = BETA_TOTAL if beta_total is None else beta_total
    m1, t1 = opinion_values(scorer, G_now)
    m0, t0 = opinion_values(scorer, G_cap)
    dm, dt = np.nan_to_num(m1 - m0), np.nan_to_num(t1 - t0)
    return {"delta_margin": dm, "delta_total": dt, "shift_margin": bm * dm,
            "shift_total": bt * dt, "beta_margin": bm, "beta_total": bt}


def apply_shift(G: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Copy of G with the market line moved (rows without a line stay NaN)."""
    out = G.copy()
    out["mkt_margin"] = G["mkt_margin"].to_numpy(float) + s["shift_margin"]
    out["mkt_total"] = G["mkt_total"].to_numpy(float) + s["shift_total"]
    return out
