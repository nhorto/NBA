"""Benchmark backtest predictions against closing lines (read-only on `closing_lines`).

Market P(home win) is the de-vigged closing moneyline pair (multiplicative normalisation);
the market margin is -home_spread_close. When a game has lines from several sources the first
in SOURCE_PRIORITY wins. All comparisons are on the games that have a usable line.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import metrics

SOURCE_PRIORITY = ("espn", "kaggle")   # same preference as odds_history.consensus_line


def american_to_prob(ml) -> np.ndarray:
    ml = np.asarray(ml, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def load_closing(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT game_id, source, home_spread_close, home_ml_close, away_ml_close "
        "FROM closing_lines", conn)
    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    df["rank"] = df["source"].map(rank).fillna(len(rank))
    ok_ml = (df["home_ml_close"].abs() >= 100) & (df["away_ml_close"].abs() >= 100)
    ph = american_to_prob(df["home_ml_close"].where(ok_ml))
    pa = american_to_prob(df["away_ml_close"].where(ok_ml))
    df["p_mkt"] = np.where(ok_ml, ph / (ph + pa), np.nan)
    df["overround"] = np.where(ok_ml, ph + pa - 1, np.nan)
    df = df.sort_values(["game_id", "rank"]).drop_duplicates("game_id")
    return df[["game_id", "source", "home_spread_close", "p_mkt", "overround"]]


def add_market_columns(frame: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """frame + `mkt_margin` (= -closing home spread; NaN without a line) and `mkt_p`
    (de-vigged closing moneyline). Closing lines are pre-game information."""
    cl = load_closing(conn)
    m = frame[["game_id"]].merge(cl, on="game_id", how="left")
    out = frame.copy()
    out["mkt_margin"] = -m["home_spread_close"].to_numpy(float)
    out["mkt_p"] = m["p_mkt"].to_numpy(float)
    return out


def has_closing_lines(conn: sqlite3.Connection) -> bool:
    r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='closing_lines'"
                     ).fetchone()
    return bool(r) and conn.execute("SELECT COUNT(*) FROM closing_lines").fetchone()[0] > 0


def compare(preds: pd.DataFrame, closing: pd.DataFrame, models: list[str] | None = None
            ) -> dict:
    """Per model, on Regular Season games with a market probability (and, separately, a
    spread): model vs market log-loss / Brier / accuracy, margin MAE, ATS record."""
    out = {}
    d0 = preds.merge(closing, on="game_id", how="inner")
    d0 = d0[d0["season_type"] == "Regular Season"]
    for name, d in d0.groupby("model", sort=False):
        if models and name not in models:
            continue
        r = {}
        dm = d[d["p_mkt"].notna()]
        if len(dm):
            y = dm["y"].to_numpy()
            r["n_ml"] = int(len(dm))
            r["seasons"] = [dm["season"].min(), dm["season"].max()]
            for k, p in (("model", dm["p"].to_numpy()), ("market", dm["p_mkt"].to_numpy())):
                r[k] = {"log_loss": metrics.log_loss(y, p), "brier": metrics.brier(y, p),
                        "accuracy": metrics.accuracy(y, p)}
            r["corr_p"] = float(np.corrcoef(dm["p"], dm["p_mkt"])[0, 1])
        ds = d[d["home_spread_close"].notna() & d["margin_pred"].notna()]
        if len(ds):
            mk = -ds["home_spread_close"].to_numpy()
            mp = ds["margin_pred"].to_numpy()
            act = ds["margin"].to_numpy(float)
            r["n_spread"] = int(len(ds))
            r["margin_mae_model"] = metrics.mae(act, mp)
            r["margin_mae_market"] = metrics.mae(act, mk)
            ats = act - mk                       # >0: home covered
            pick_home = mp > mk
            live = (ats != 0) & (mp != mk)
            win = np.where(pick_home, ats > 0, ats < 0)[live]
            r["ats"] = {"n": int(live.sum()), "win_rate": float(win.mean()) if live.any() else None}
            big = live & (np.abs(mp - mk) >= 3)
            wb = np.where(pick_home, ats > 0, ats < 0)[big]
            r["ats_edge3"] = {"n": int(big.sum()),
                              "win_rate": float(wb.mean()) if big.any() else None}
        out[name] = r
    return out


def format_compare(res: dict) -> str:
    lines = [f"{'model':<22}{'n':>6}{'LL mod':>8}{'LL mkt':>8}{'Br mod':>8}{'Br mkt':>8}"
             f"{'acc mod':>8}{'acc mkt':>8}{'MAE mod':>8}{'MAE mkt':>8}{'ATS':>7}{'n ATS':>7}"]
    for n, r in res.items():
        if "model" not in r:
            continue
        a, m = r["model"], r["market"]
        lines.append(f"{n:<22}{r['n_ml']:>6}{a['log_loss']:>8.4f}{m['log_loss']:>8.4f}"
                     f"{a['brier']:>8.4f}{m['brier']:>8.4f}{a['accuracy']:>8.3f}"
                     f"{m['accuracy']:>8.3f}{r.get('margin_mae_model', np.nan):>8.2f}"
                     f"{r.get('margin_mae_market', np.nan):>8.2f}"
                     f"{(r.get('ats') or {}).get('win_rate') or np.nan:>7.3f}"
                     f"{(r.get('ats') or {}).get('n', 0):>7}")
    return "\n".join(lines)
