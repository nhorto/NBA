"""Production scoring for the stacked models: fit once a day, re-price many times.

`predict.py` calls this. The expensive part of a GBM prediction is the player-engine pass
over all history (~30 s) plus the fits (~1 s each with frozen configs). None of that changes
within an NBA day: the training set is every game before the day. What does change within the
day is cheap: the market line (`mkt_margin`, `mkt_total`) and the injury report (expected-lineup
strength). So a day's `Bundle` holds

* the fitted heads and calibrators of each model (`Scorer`, picklable),
* the day's feature rows `F` built without any live line or injury override,
* the player-engine state (ratings, minutes projections, rosters) for `expected_lineup`,

and `price()` overwrites the market and lineup columns of `F` and scores it. Overwriting columns
of `F` is exactly what `features.build_features` does with `mkt_*` and `lineup_*` columns
(tests pin the equivalence), without rebuilding 100k rows of features.

The cache lives in `data_dir()/models/<day>/bundle.pkl` and is keyed by the newest game date
(and count) in `team_games` before the day: new results invalidate it, a new line or injury
report does not. Leakage: nothing here sees a result on/after the day; for past days the engine
state used for lineups is rebuilt from games before the day only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, injuries
from .model import gbm, players as pl

log = logging.getLogger("courtside.production")

CACHE_VERSION = 2
KEEP_DAYS = 8                    # model dirs older than this are pruned

# Factors worth showing a reader. `market` is the line itself, `season_phase` and
# `pace_scoring` are bookkeeping that does not explain a margin.
SHOWN_FACTORS = ("absences", "player_strength", "team_rating", "rest", "travel",
                 "recent_form", "continuity")
# `home_court` is deliberately not shown: explain() centres every factor on the training mean,
# so a normal home-court edge reads as ~0 and a below-average one as a negative number.
FACTORS_MIN_PTS = 0.25           # below this a factor is noise, unless fewer than 3 survive
FACTORS_PER_GAME = 4


# ------------------------------------------------------------------ scoring on supplied features

class Scorer(gbm.GBMModel):
    """A fitted `GBMModel` reduced to what scoring needs (heads + calibrator), restored from the
    cache. Features are supplied by the caller (`set_features`); `predict_*`, `contributions`
    and `explain` work as on the original model, for the frame aligned with those features."""

    @classmethod
    def from_model(cls, m: gbm.GBMModel) -> "Scorer":
        s = cls.__new__(cls)
        s.name, s.market = m.name, m.market
        s.heads_, s.cal_ = m.heads_, m.cal_
        s.search_ = getattr(m, "search_", {})
        s.cal_scores_ = getattr(m, "cal_scores_", None)
        s.oof_resid_sd_ = getattr(m, "oof_resid_sd_", None)
        s.train_ = None
        s._out = None
        return s

    def set_features(self, F: pd.DataFrame, post: np.ndarray) -> None:
        out = {"F": F}
        for target, (fit, k, _) in self.heads_.items():
            out[target] = gbm._head_predict(fit, F, k, self.market)
        out["p"] = np.clip(self.cal_.predict(out["margin"], post), 1e-4, 1 - 1e-4)
        self._out = out

    def _predict_all(self, df=None) -> dict:
        if self._out is None:
            raise RuntimeError("Scorer.set_features() first")
        return self._out

    def margin_at(self, F: pd.DataFrame) -> np.ndarray:
        fit, k, _ = self.heads_["margin"]
        return gbm._head_predict(fit, F, k, self.market)


@dataclass
class Bundle:
    key: dict
    day: str
    scorers: dict[str, Scorer]
    F: pd.DataFrame                       # one row per game of `game_ids`, no live line / lineup
    game_ids: list[str]
    state: object | None                  # players.PlayerState before the day (or None: lazy)
    train_games: int
    train_through: str
    fitted_at: str
    fit_seconds: float
    timings: dict = field(default_factory=dict)
    cache: str = "miss"                   # "hit" | "miss" | "refit" (set by load_or_fit)


# ------------------------------------------------------------------ cache

def models_dir(day) -> Path:
    return config.data_dir() / "models" / str(day)


def cache_key(conn, day, game_ids: list[str], names: tuple[str, ...]) -> dict:
    """What a fitted day depends on: the training set (newest game date and row count before
    `day`), the day's slate, the frozen hyperparameters and the code version."""
    r = conn.execute("SELECT MAX(game_date), COUNT(*) FROM team_games WHERE game_date < ?",
                     (str(day),)).fetchone()
    frozen = hashlib.sha1(repr({n: gbm.FROZEN.get(n) for n in names}).encode()).hexdigest()[:12]
    return {"v": CACHE_VERSION, "day": str(day), "through": r[0], "n_team_games": r[1],
            "slate": hashlib.sha1("|".join(sorted(game_ids)).encode()).hexdigest()[:12],
            "frozen": frozen, "models": list(names)}


def load_bundle(day, key: dict) -> Bundle | None:
    path = models_dir(day) / "bundle.pkl"
    if not path.exists():
        return None
    try:
        b = pickle.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - a stale/corrupt pickle is just a miss
        log.warning("model cache %s unreadable; refitting", path)
        return None
    return b if isinstance(b, Bundle) and b.key == key else None


def save_bundle(b: Bundle) -> Path:
    d = models_dir(b.day)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"bundle.pkl.{os.getpid()}.tmp"        # two runs of the day cannot clobber each other
    tmp.write_bytes(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(d / "bundle.pkl")
    (d / "meta.json").write_text(json.dumps({
        "key": b.key, "fitted_at": b.fitted_at, "fit_seconds": round(b.fit_seconds, 1),
        "train_games": b.train_games, "train_through": b.train_through,
        "timings": b.timings, "has_state": b.state is not None}, indent=1))
    prune(b.day)
    return d / "bundle.pkl"


def prune(day, keep_days: int = KEEP_DAYS) -> None:
    root = config.data_dir() / "models"
    if not root.exists():
        return
    cutoff = (date.fromisoformat(str(day)) - pd.Timedelta(days=keep_days)).isoformat()
    for d in root.iterdir():
        if d.is_dir() and d.name < cutoff:
            shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ fitting

def registry(conn):
    """The production model factories (win-prob + opinion) sharing one engine/feature cache."""
    from .model import player_model
    return player_model.make_registry(lambda: pl.load_player_games(conn), conn=conn)


def has_player_rows(conn, day) -> bool:
    """Do box scores for `day` already exist (a past/demo day)? Then the engine state must be
    rebuilt from games before the day only."""
    return conn.execute("SELECT 1 FROM player_games WHERE game_date=? AND min>0 LIMIT 1",
                        (str(day),)).fetchone() is not None


def player_state(conn, train: pd.DataFrame, players: pd.DataFrame | None = None):
    """PlayerState after the last game of `train`: ratings, minutes projections, rosters."""
    players = players if players is not None else pl.load_player_games(conn)
    _, st = pl.run_player_engine(train, players, pl.TUNED, gbm.ENGINE_MODES, return_state=True)
    return st


def fit_bundle(conn, day, train: pd.DataFrame, test: pd.DataFrame,
               names: tuple[str, ...], key: dict, reg=None) -> Bundle:
    """Cold path: engine pass, fit every model on `train`, features for `test`'s games."""
    t0 = time.time()
    reg = reg or registry(conn)[0]
    models = {n: reg[n]() for n in names}
    m0 = next(iter(models.values()))
    fc = m0.fc
    live = not has_player_rows(conn, day)
    full = pd.concat([train, test])
    tm = {}
    t = time.time()
    if live:
        m0.prepare(train, test)        # one engine pass: features and player state
        state = fc.engine.state(full)
    else:
        fc.prime(full)
        state = None                   # built lazily from `train` (no peeking at the day's box)
    tm["engine_features"] = round(time.time() - t, 1)
    t = time.time()
    for m in models.values():
        m.fit(train)
    tm["fit"] = round(time.time() - t, 1)
    F = m0._test_features(test).reset_index(drop=True)
    scorers = {n: Scorer.from_model(m) for n, m in models.items()}
    return Bundle(key=key, day=str(day), scorers=scorers, F=F,
                  game_ids=list(test["game_id"]), state=state, train_games=int(len(train)),
                  train_through=train["game_date"].max().strftime("%Y-%m-%d"),
                  fitted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  fit_seconds=time.time() - t0, timings=tm)


def load_or_fit(conn, day, train: pd.DataFrame, test: pd.DataFrame,
                names: tuple[str, ...], *, use_cache: bool = True, refit: bool = False
                ) -> Bundle:
    key = cache_key(conn, day, list(test["game_id"]), names)
    if use_cache and not refit:
        b = load_bundle(day, key)
        if b is not None:
            b.cache = "hit"
            return b
    b = fit_bundle(conn, day, train, test, names, key)
    b.cache = "refit" if refit else "miss"
    if use_cache:
        save_bundle(b)
    return b


def ensure_state(conn, b: Bundle, train: pd.DataFrame) -> object:
    """The engine state for lineups; built (and cached in the bundle) on first need."""
    if b.state is None:
        t = time.time()
        b.state = player_state(conn, train)
        b.timings["state"] = round(time.time() - t, 1)
        if b.cache in ("hit", "miss", "refit"):       # not for a backfill's throwaway bundle
            save_bundle(b)
    return b.state


# ------------------------------------------------------------------ pricing

def lineup_strengths(state, rows: pd.DataFrame, avail_by_team: dict[str, dict]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Expected-lineup strength of the home/away side of each row, NaN where the team has no
    entry in `avail_by_team` ({raw team code: {player_id: P(plays)}}; the engine itself is
    keyed by franchise, the injury report by the raw tricode)."""
    out = []
    for side in ("home", "away"):
        v = np.full(len(rows), np.nan)
        for i, (team, raw, st) in enumerate(zip(rows[side], rows[f"{side}_raw"],
                                                rows["season_type"])):
            av = avail_by_team.get(raw)
            if av is None:
                continue
            lu = state.expected_lineup(team, None, av, playoff=st != "Regular Season")
            v[i] = float(lu.attrs["strength"])
        out.append(v)
    return out[0], out[1]


def apply_overrides(F: pd.DataFrame, mkt_margin=None, mkt_total=None, lineup_home=None,
                    lineup_away=None) -> pd.DataFrame:
    """Copy of `F` with the live line and expected-lineup strengths applied, the same way
    `features.build_features` does for `mkt_margin`/`mkt_total`/`lineup_*` frame columns:
    finite values replace; ps_diff and the absence losses are recomputed from the roster
    strength (roster = available lineup + absence loss, which the override leaves unchanged)."""
    G = F.copy()
    for col, v in (("mkt_margin", mkt_margin), ("mkt_total", mkt_total)):
        if v is not None:
            v = np.asarray(v, float)
            cur = G[col].to_numpy(float)
            G[col] = np.where(np.isfinite(v), v, cur)
    ph, pa = G["ps_home"].to_numpy(float).copy(), G["ps_away"].to_numpy(float).copy()
    rh, ra = ph + G["abs_loss_home"].to_numpy(float), pa + G["abs_loss_away"].to_numpy(float)
    for arr, v in ((ph, lineup_home), (pa, lineup_away)):
        if v is not None:
            v = np.asarray(v, float)
            arr[np.isfinite(v)] = v[np.isfinite(v)]
    if lineup_home is not None or lineup_away is not None:
        G["ps_home"], G["ps_away"] = ph, pa
        G["ps_diff"] = ph - pa
        G["abs_loss_home"], G["abs_loss_away"] = rh - ph, ra - pa
    return G


def absence_effects(scorer: Scorer, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """What each side's absences cost it, in points of its own expected margin: the margin
    with the lineups as priced minus the margin with that side at full recent-roster strength
    (<= 0 when someone is out). An exact re-score of the same heads, so unlike `explain`'s
    mean-centred `absences` factor it reads as "LAL's absences cost it 2.2 points"."""
    base = scorer.margin_at(F)
    rh = F["ps_home"].to_numpy(float) + F["abs_loss_home"].to_numpy(float)
    ra = F["ps_away"].to_numpy(float) + F["abs_loss_away"].to_numpy(float)
    eff_h = base - scorer.margin_at(apply_overrides(F, lineup_home=rh))
    eff_a = -(base - scorer.margin_at(apply_overrides(F, lineup_away=ra)))
    return np.nan_to_num(eff_h), np.nan_to_num(eff_a)


def injury_effects(scorer: Scorer, F: pd.DataFrame, state, rows: pd.DataFrame,
                   cover: dict[str, dict]) -> list[dict]:
    """What each listed player's status is worth to the model, in points of his team's expected
    margin: margin(actual report) - margin(that player healthy), from his team's side (so
    negative when his absence hurts his team). One counterfactual lineup per player, an exact
    re-score of the same heads. Only for matched players with P(plays) < 1."""
    out = []
    base = scorer.margin_at(F)
    for i, (_, r) in enumerate(rows.iterrows()):
        for side in ("home", "away"):
            team = r[side]
            c = cover.get(r[f"{side}_raw"])
            if not c:
                continue
            for rec in c["players"]:
                pid = rec["player_id"]
                if pid is None or rec["p_play"] >= 1.0:
                    continue
                av = {k: v for k, v in c["avail"].items() if k != pid}
                lu = state.expected_lineup(team, None, av,
                                           playoff=r["season_type"] != "Regular Season")
                s_cf = float(lu.attrs["strength"])
                lh = F["ps_home"].to_numpy(float)[i] if side == "away" else s_cf
                la = F["ps_away"].to_numpy(float)[i] if side == "home" else s_cf
                G = apply_overrides(F.iloc[[i]].reset_index(drop=True),
                                    lineup_home=[lh], lineup_away=[la])
                d = float(base[i] - scorer.margin_at(G)[0])      # home margin, actual - healthy
                eff = d if side == "home" else -d
                out.append({"game_id": r["game_id"], "team": team, "side": side,
                            "player": rec["player"], "player_id": pid, "status": rec["status"],
                            "p_play": rec["p_play"], "effect_margin": round(eff, 2)})
    return out


# ------------------------------------------------------------------ factors

_LABEL = {
    "player_strength": "roster strength", "team_rating": "team rating",
    "home_court": "home court", "rest": "rest and schedule", "travel": "travel",
    "recent_form": "recent form", "continuity": "roster continuity",
}


def _names(players: list[dict], limit: int = 3) -> str:
    shown = [f"{p['player']} ({p['status']})" for p in players[:limit]]
    more = len(players) - limit
    return ", ".join(shown) + (f" +{more} more" if more > 0 else "")


def key_factors(explained: list[dict], home: str, away: str,
                absentees: dict[str, list[dict]] | None = None,
                absence_pts: dict[str, float] | None = None,
                top: int = FACTORS_PER_GAME) -> list[dict]:
    """Brief-ready model factors from `GBMModel.explain` rows (see docs/BRIEF.md, kind
    "model_factor"): {kind, factor, team, side, value (= impact_pts), impact_pts, favors,
    detail}. `impact_pts` is from `team`'s side; `absentees` ({team: [{player, status}, ...]},
    most important first) names who is missing for the absences factor. `absence_pts`
    ({team: points}, see `absence_effects`) replaces explain's mean-centred absences number by
    the cost against a healthy roster."""
    sides = {home: "home", away: "away"}
    explained = [e for e in explained if e["factor"] != "absences" or absence_pts is None]
    for team, pts in (absence_pts or {}).items():
        explained.append({"factor": "absences", "team": team, "impact_pts": round(pts, 2)})
    cand = [e for e in explained if e["factor"] in SHOWN_FACTORS and abs(e["impact_pts"]) >= 0.05]
    cand.sort(key=lambda e: -abs(e["impact_pts"]))
    keep = [e for e in cand if abs(e["impact_pts"]) >= FACTORS_MIN_PTS]
    if len(keep) < 3:
        keep = cand[:3]
    out = []
    for e in keep[:top]:
        team, pts = e["team"], e["impact_pts"]
        side = sides.get(team)
        other = away if team == home else home
        if e["factor"] == "absences":
            who = (absentees or {}).get(team) or []
            detail = (f"{team} absences: {_names(who)}" if who
                      else f"{team} absences cost {abs(pts):.1f} pts")
        else:
            detail = f"{team} {_LABEL[e['factor']]} {pts:+.1f} pts"
        favors = sides.get(team if pts >= 0 else other)
        out.append({"kind": "model_factor", "factor": e["factor"], "team": team, "side": side,
                    "value": round(pts, 2), "impact_pts": round(pts, 2), "favors": favors,
                    "detail": detail})
    return out
