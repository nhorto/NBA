"""Walk-forward-by-season backtest harness.

Model interface (duck-typed; see model/baselines.py):

    model.fit(train_df)               train_df = every game in seasons < S (with results)
    model.predict_proba(test_df)      -> np.ndarray, P(home win) for each row of test_df
    model.predict_margin(test_df)     optional -> np.ndarray, expected home margin

`test_df` is season S in chronological order and DOES include result columns, so online models
(Elo, later player ratings) can update after each game, exactly as they would in production
the next morning. The contract is that the prediction for a game on day d may depend only on
train_df and on test games played before day d. `check_no_lookahead` enforces this
empirically by scrambling the results of every game on or after a cut date and asserting that
no prediction on or before that date changes; tests run it for every registered model.

Learned parameters are fit only on train_df each fold. Fixed hyperparameters (Elo) were tuned
on 2000-01..2010-11, before the first test season.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from . import dataset, metrics, schedule

SEGMENTS = {
    "Regular Season": lambda d: d["season_type"] == "Regular Season",
    "Playoffs": lambda d: d["season_type"] == "Playoffs",
    "All": lambda d: pd.Series(True, index=d.index),
}


class Model(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray: ...


def build_frame(conn) -> pd.DataFrame:
    """Games + schedule features, chronological. Schedule features use no results."""
    games = dataset.load_games(conn)
    frame = schedule.add_schedule_features(games)
    frame.attrs["anomalies"] = games.attrs.get("anomalies", {})
    return frame


def seasons_in(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["season"].unique())


def walk_forward_splits(frame: pd.DataFrame, test_seasons: Iterable[str]):
    """Yield (season, train_df, test_df): train = all seasons strictly before S, test = S."""
    for s in test_seasons:
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"empty train or test for season {s}")
        yield s, train, test


def run_backtest(frame: pd.DataFrame, factories: dict[str, Callable[[], Model]],
                 test_seasons: Iterable[str], log: Callable[[str], None] | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Returns (predictions, fitted-model descriptions). predictions has one row per
    (model, test game): model, game_id, season, season_type, game_date, y, margin, p, margin_pred.
    A fresh model instance is built for every fold."""
    rows, desc = [], {}
    for s, train, test in walk_forward_splits(frame, list(test_seasons)):
        for name, make in factories.items():
            m = make()
            m.fit(train)
            p = np.asarray(m.predict_proba(test), dtype=float)
            if p.shape != (len(test),) or not np.all((p >= 0) & (p <= 1)):
                raise ValueError(f"{name}: bad predict_proba output for {s}")
            mp = (np.asarray(m.predict_margin(test), dtype=float)
                  if hasattr(m, "predict_margin") else np.full(len(test), np.nan))
            rows.append(pd.DataFrame({
                "model": name, "game_id": test["game_id"].to_numpy(), "season": s,
                "season_type": test["season_type"].to_numpy(),
                "game_date": test["game_date"].to_numpy(),
                "y": test["home_win"].to_numpy(), "margin": test["margin"].to_numpy(),
                "p": p, "margin_pred": mp}))
            if hasattr(m, "describe"):
                desc.setdefault(name, {})[s] = m.describe()
        if log:
            log(f"  {s}: {len(test)} games")
    return pd.concat(rows, ignore_index=True), desc


def _summ(d: pd.DataFrame, calib: bool = False) -> dict:
    has_m = d["margin_pred"].notna().all()
    out = metrics.summarize(d["y"].to_numpy(), d["p"].to_numpy(),
                            d["margin"].to_numpy() if has_m else None,
                            d["margin_pred"].to_numpy() if has_m else None)
    if calib:
        out["calibration"] = metrics.calibration_table(d["y"].to_numpy(), d["p"].to_numpy())
    return out


def evaluate(preds: pd.DataFrame, last_n: int = 3) -> dict:
    """{model: {"per_season": {season: {segment: metrics}}, "aggregate": {segment: metrics},
    "last_n": {...}}}. Aggregates pool all games (not an average of season metrics)."""
    out = {}
    seasons = sorted(preds["season"].unique())
    last = seasons[-last_n:]
    for name, d in preds.groupby("model", sort=False):
        r = {"per_season": {}, "aggregate": {}, f"last{last_n}": {},
             "seasons": [seasons[0], seasons[-1]], f"last{last_n}_seasons": last}
        for seg, f in SEGMENTS.items():
            ds = d[f(d)]
            if len(ds):
                r["aggregate"][seg] = _summ(ds, calib=True)
            dl = ds[ds["season"].isin(last)]
            if len(dl):
                r[f"last{last_n}"][seg] = _summ(dl, calib=True)
        for s, dd in d.groupby("season"):
            r["per_season"][s] = {seg: _summ(dd[f(dd)]) for seg, f in SEGMENTS.items()
                                  if f(dd).any()}
        out[name] = r
    return out


def check_no_lookahead(make: Callable[[], Model], train: pd.DataFrame, test: pd.DataFrame,
                       cut_date, seed: int = 0) -> float:
    """Scramble results of all test games on/after cut_date; return the max abs change in
    predictions for test games on/before cut_date (0.0 means no lookahead)."""
    cut = pd.Timestamp(cut_date)
    rng = np.random.default_rng(seed)
    alt = test.copy()
    fut = (alt["game_date"] >= cut).to_numpy()
    k = int(fut.sum())
    hp = rng.integers(70, 140, k)
    ap = rng.integers(70, 140, k)
    ap = np.where(ap == hp, ap + 1, ap)
    alt.loc[fut, "home_pts"] = hp
    alt.loc[fut, "away_pts"] = ap
    alt.loc[fut, "margin"] = hp - ap
    alt.loc[fut, "home_win"] = (hp > ap).astype(int)
    m1, m2 = make(), make()
    m1.fit(train)
    m2.fit(train)
    p1 = np.asarray(m1.predict_proba(test))
    p2 = np.asarray(m2.predict_proba(alt))
    upto = (test["game_date"] <= cut).to_numpy()
    diff = float(np.max(np.abs(p1[upto] - p2[upto]))) if upto.any() else 0.0
    if hasattr(m1, "predict_margin"):
        q1, q2 = np.asarray(m1.predict_margin(test)), np.asarray(m2.predict_margin(alt))
        diff = max(diff, float(np.max(np.abs(q1[upto] - q2[upto]))) if upto.any() else 0.0)
    return diff


# --- reporting -----------------------------------------------------------------------------

def _fmt(v, nd=4):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def format_tables(ev: dict, last_n: int = 3) -> str:
    lines = []
    names = list(ev)
    any_r = next(iter(ev.values()))
    span = f"{any_r['seasons'][0]}..{any_r['seasons'][1]}"
    lastk = f"last{last_n}"
    lspan = ", ".join(any_r[f"{lastk}_seasons"])
    hdr = f"{'model':<11}{'n':>7}{'logloss':>9}{'brier':>8}{'acc':>7}{'ece':>7}{'MAE':>7}{'RMSE':>7}"
    for seg in ("Regular Season", "Playoffs"):
        for scope, label in (("aggregate", span), (lastk, lspan)):
            lines.append(f"\n{seg} | {label}")
            lines.append(hdr)
            for n in names:
                m = ev[n][scope].get(seg)
                if not m:
                    continue
                lines.append(f"{n:<11}{m['n']:>7}{_fmt(m['log_loss']):>9}{_fmt(m['brier']):>8}"
                             f"{_fmt(m['accuracy'], 3):>7}{_fmt(m['ece'], 3):>7}"
                             f"{_fmt(m.get('margin_mae'), 2):>7}{_fmt(m.get('margin_rmse'), 2):>7}")
    lines.append("\nRegular Season log-loss / accuracy by season")
    lines.append(f"{'season':<9}" + "".join(f"{n:>18}" for n in names))
    for s in sorted(any_r["per_season"]):
        cells = []
        for n in names:
            m = ev[n]["per_season"].get(s, {}).get("Regular Season")
            cells.append(f"{_fmt(m['log_loss']) + ' / ' + _fmt(m['accuracy'], 3):>18}"
                         if m else f"{'-':>18}")
        lines.append(f"{s:<9}" + "".join(cells))
    return "\n".join(lines)


def write_report(ev: dict, desc: dict, frame: pd.DataFrame, out_dir: Path,
                 extra: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"backtest_{ts}.json"
    an = dict(frame.attrs.get("anomalies", {}))
    an.pop("games_by_season_type", None)
    doc = {"created_utc": ts, "n_games": int(len(frame)),
           "seasons": [frame["season"].min(), frame["season"].max()],
           "results": ev, "models": desc, "anomalies": an, **(extra or {})}
    path.write_text(json.dumps(doc, indent=1, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(type(o))
