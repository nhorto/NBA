"""Learned context layer (docs/PLAYERS.md Phase 2F, F0).

Per DK stat s the per-minute rate becomes

    rate_s = EB rate_s x m_s(x),    m_s = clip(exp(f_s(x) - f_s(x_ref)), CLIP)

with f_s a LightGBM fit on seasons < S (walk-forward) of the log-ratio of the actual count to
the EB expectation given the ACTUAL minutes: Poisson deviance with offset log E_s, where
E_s = r_s x min x f_s(min/mpg) x tier bump (`features.common.base_rows`). Because the offset
already carries the minutes, f learns per-minute context only, never minutes. The Poisson
gradient weights every row by its expected count (minutes x rate): that is the minutes weight.
Features are strictly pre-tip (`features`); groups are toggled for the ablation.

Overfit control: shallow trees (depth 4, 15 leaves), >= 1500 rows and a hessian floor per leaf,
L2 50, row / column subsampling, learning rate 0.03, rounds by early stopping on the last
training season (then refit on all training seasons with that count), monotone constraints on
a stat's own DvP index, and the multiplier clipped to [0.8, 1.25]. The intercept is learned
(Poisson score equation: the E-weighted mean multiplier on the training rows is 1, i.e. the
layer is level-neutral where the EB state is calibrated); with `center=True` the model is
shifted so that it returns exactly 1 at `x_ref`, the league-average context (no opposing
absence, opponent indices 0, a median player).

Public API
  ContextModel (JSON round trip: to_json / from_json; boosters embedded as model strings)
  fit(frame, train_mask, groups, ...) -> ContextModel
  ContextModel.predict(X) -> {stat: multipliers}
  save(model, path) / load(path)
  apply_rates(model, rates, team_feats, game_id, team) -> rates with r_<s> scaled (the
      pipeline hook, see INTEGRATION below)
  evaluate(test, mults, ...) / two_way_se(...)        ablation metrics
  run_ablation(conn, season, ...)                       the F0-F2 ablation for one season

CLI: python -m courtside.proj.context {dvp-params, ablation, fit, bundle} ...
  bundle --season S   the pipeline's ON bundle (`bundle_<S>_v1_cx.json`: the OFF bundle of S +
                      this layer fit on seasons < S), whatever the switch is set to

INTEGRATION (wired in `pipeline`, behind `pipeline.CONTEXT_ON`, default OFF; env
COURTSIDE_CONTEXT=1 turns it on; the extended layer P+F2+F3+F6 additionally needs
`pipeline.CONTEXT_EXT`, env COURTSIDE_CONTEXT_EXT=1: bundle `bundle_<S>_v1_cx2.json`,
`pathA-v0.4`, F3 per player and F6 per team-game from `features.LiveExtras`):
  1. Bundle: `Bundle.context` (JSON key "context", only present when set, so an OFF bundle is
     byte-identical to the pre-context format), fit in `load_bundle` on seasons < S by
     `fit_season(conn, S, bundle.state)` = `fit(FX.build_frame(conn, S, sp), season < S,
     ("P", "F2"))`. An ON bundle is the OFF bundle of the same season plus the layer, cached as
     `bundle_<S>_v1_cx.json`; ON runs are versioned `pathA-v0.3` (OFF: `pathA-v0.2`).
  2. Context (once per season): `cx_t = FXC.load_team_games(conn, S-2, S)` at init and the
     `FXC.base_rows(conn, [S-2, S-1, S], state)` history on first use with the bundle's state
     params (`pipeline.Context.context_inputs`; every later use is strictly before the day).
  3. pipeline.build_games, right after `panel["_star"] = _stars(panel, rates)` (stars stay on
     the EB rates): F2 day features with the environment's market total in the pace term
     (`FX.day_features(..., totals=)`; the closing total in backtests, the live books total on
     the live path), then `apply_rates`: r_<s> *= m_s for the 7 DK stats and r_dk recomputed,
     before the bump/usage grids and the reconciliation (which still scales team points to the
     implied total).
  F1 (not shipped) would need the opponent's availability scenario: compute
  `features.opp_avail.scenario_features(opp_players, kept_masks)` for the opponent's kept
  scenarios (<= 16), predict m_s per (opponent scenario, player), and in copula.simulate_game
  multiply the per-sim means by m_s[opp_assign] (where usage.sim_means applies F).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import state as ST
from . import features as FX
from .features.common import DK_STATS, DK_W

CONTEXT_VERSION = 1
CLIP = (0.8, 1.25)
LGB_PARAMS = {
    "objective": "poisson", "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
    "min_data_in_leaf": 1500, "min_sum_hessian_in_leaf": 50.0, "lambda_l2": 50.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "poisson_max_delta_step": 0.7, "seed": 7, "deterministic": True, "force_row_wise": True,
    "num_threads": 3, "verbose": -1, "max_bin": 63,
}
MAX_ROUNDS, EARLY = 1500, 60
VARIANTS = {"base": (), "P": ("P",), "P+F1": ("P", "F1"), "P+F2": ("P", "F2"),
            "P+F1+F2": ("P", "F1", "F2"),
            # second ablation (F3 / F5 / F6 on top of the shipped P+F2)
            "P+F2+F3": ("P", "F2", "F3"), "P+F2+F5": ("P", "F2", "F5"),
            "P+F2+F6": ("P", "F2", "F6"), "P+F2+F3+F5+F6": ("P", "F2", "F3", "F5", "F6"),
            "P+F2+F3+F5": ("P", "F2", "F3", "F5"), "P+F2+F3+F6": ("P", "F2", "F3", "F6")}
VARIANTS_2 = ("P+F2", "P+F2+F3", "P+F2+F5", "P+F2+F6", "P+F2+F3+F5+F6")
# staged: F5 / F6 on top of P+F2+F3 (F3 passes alone by a wide margin)
VARIANTS_3 = ("P+F2", "P+F2+F3", "P+F2+F3+F5", "P+F2+F3+F6", "P+F2+F3+F5+F6")


# ============================================================================== model

@dataclass
class ContextModel:
    season: str
    groups: list
    stats: list
    columns: dict                       # stat -> feature list
    boosters: dict                      # stat -> LightGBM model string
    n_rounds: dict
    offset: dict                        # stat -> f(x_ref) (subtracted when centered)
    reference: dict                     # stat -> {column: neutral value}
    clip: list = field(default_factory=lambda: list(CLIP))
    train_seasons: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    version: int = CONTEXT_VERSION

    def __post_init__(self):
        self._b: dict = {}

    def booster(self, stat: str):
        import lightgbm as lgb
        if stat not in self._b:
            self._b[stat] = lgb.Booster(model_str=self.boosters[stat])
        return self._b[stat]

    def raw(self, X: pd.DataFrame, stat: str) -> np.ndarray:
        if stat not in self.boosters:
            return np.zeros(len(X))
        cols = self.columns[stat]
        A = X[cols].to_numpy(float) if len(X) else np.zeros((0, len(cols)))
        return (self.booster(stat).predict(A, raw_score=True, num_threads=1)
                - self.offset.get(stat, 0.0))

    def predict_stat(self, X: pd.DataFrame, stat: str) -> np.ndarray:
        return np.clip(np.exp(self.raw(X, stat)), *self.clip)

    def predict(self, X: pd.DataFrame) -> dict:
        """{stat: multiplier (n,)} (1.0 for stats without a booster)."""
        return {s: self.predict_stat(X, s) for s in DK_STATS}

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items()}
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ContextModel":
        return cls(**json.loads(s))


def save(model: ContextModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.to_json())


def load(path: Path) -> ContextModel:
    return ContextModel.from_json(Path(path).read_text())


def identity(season: str = "") -> ContextModel:
    return ContextModel(season=season, groups=[], stats=list(DK_STATS), columns={}, boosters={},
                        n_rounds={}, offset={}, reference={})


def _reference(train: pd.DataFrame, cols: list, groups: Sequence[str]) -> dict:
    neutral = {}
    for g in groups:
        neutral.update(FX.GROUPS[g].neutral)
    ref = {}
    for c in cols:
        if c in neutral:
            ref[c] = float(neutral[c])
        else:
            v = train[c].to_numpy(float)
            v = v[np.isfinite(v)]
            ref[c] = float(np.median(v)) if len(v) else 0.0
    if "p_grp" in ref:
        ref["p_grp"] = 1.0                   # forwards: the middle group
    return ref


def fit(frame: pd.DataFrame, train_mask: np.ndarray, groups: Sequence[str], *,
        season: str = "", stats: Sequence[str] = DK_STATS, params: dict | None = None,
        max_rounds: int = MAX_ROUNDS, early: int = EARLY, center: bool = False,
        log: Callable[[str], None] | None = None) -> ContextModel:
    """Per-stat LightGBM (Poisson, offset log E_s) on the regular-season rows of `train_mask`;
    rounds by early stopping on the last training season, then a refit on all of them."""
    import lightgbm as lgb
    say = log or (lambda m: None)
    groups = list(groups)
    if not groups:
        return identity(season)
    prm = dict(LGB_PARAMS, **(params or {}))
    tr = frame[np.asarray(train_mask, bool) & (frame["season_type"] == "Regular Season").to_numpy()]
    seasons = sorted(tr["season"].unique())
    inner = (tr["season"] < seasons[-1]).to_numpy() if len(seasons) > 1 else None
    cats = FX.categorical_for(groups)
    m = ContextModel(season=season, groups=groups, stats=list(stats), columns={}, boosters={},
                     n_rounds={}, offset={}, reference={}, train_seasons=seasons,
                     meta={"n_train": int(len(tr)), "params": prm, "center": center})
    for s in stats:
        cols = FX.model_columns(s, groups)
        E = tr[f"E_{s}"].to_numpy(float)
        ok = E > 1e-6
        X = tr.loc[ok, cols].to_numpy(float)
        y = tr.loc[ok, s].to_numpy(float)
        off = np.log(E[ok])
        mono = FX.monotone_for(s, groups)
        p = dict(prm, monotone_constraints=[int(mono.get(c, 0)) for c in cols])
        cat_idx = [cols.index(c) for c in cats if c in cols]
        t0 = time.time()
        n_best = max_rounds
        if inner is not None:
            ii = inner[ok]
            dtr = lgb.Dataset(X[ii], y[ii], init_score=off[ii], feature_name=cols,
                              categorical_feature=cat_idx, free_raw_data=False)
            dva = lgb.Dataset(X[~ii], y[~ii], init_score=off[~ii], reference=dtr,
                              categorical_feature=cat_idx, free_raw_data=False)
            b = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(early, verbose=False)])
            n_best = max(int(b.best_iteration or 0), 1)
        dall = lgb.Dataset(X, y, init_score=off, feature_name=cols, categorical_feature=cat_idx,
                           free_raw_data=False)
        b = lgb.train(p, dall, num_boost_round=n_best)
        m.columns[s] = cols
        m.boosters[s] = b.model_to_string()
        m.n_rounds[s] = n_best
        ref = _reference(tr, cols, groups)
        m.reference[s] = ref
        r = np.array([[ref[c] for c in cols]], float)
        m.offset[s] = float(b.predict(r, raw_score=True)[0]) if center else 0.0
        say(f"context {season} {'+'.join(groups)} {s}: {n_best} rounds ({time.time() - t0:.0f}s)")
    return m


LIVE_GROUPS = ("P", "F2")
# the extended layer behind pipeline.CONTEXT_EXT (Phase 2F second ablation; see PLAYERS.md)
LIVE_GROUPS_EXT = ("P", "F2", "F3", "F6")
LIVE_N_TRAIN = 6


def fit_season(conn: sqlite3.Connection, season: str, sp: dict, *,
               groups: Sequence[str] = LIVE_GROUPS, n_train: int = LIVE_N_TRAIN,
               params: dict | None = None, frame: pd.DataFrame | None = None,
               log: Callable[[str], None] | None = None) -> ContextModel:
    """The layer the pipeline uses for season S: `fit` on the regular-season rows of the
    n_train seasons < S of `FX.build_frame(conn, S, sp)` (sp = the bundle's rate-state params,
    so the P features and the E_ offsets match the live rates). Walk-forward: rows of S (and
    later) never enter the fit."""
    say = log or (lambda m: None)
    t0 = time.time()
    fr = frame if frame is not None else FX.build_frame(conn, season, sp, n_train=n_train,
                                                         log=say)
    ext = [g for g in groups if g in FX.EXTENSION_GROUPS]
    if ext:
        fr = FX.extend_frame(conn, fr, ext, season=season, cache=frame is None, log=say)
    train = (fr["season"] < season).to_numpy()
    m = fit(fr, train, groups, season=season, params=params, log=say)
    m.meta["n_train_seasons"] = n_train
    say(f"context {season} {'+'.join(groups)} fitted on {m.train_seasons} "
        f"({time.time() - t0:.0f}s)")
    return m


def importances(model: ContextModel, kind: str = "gain") -> pd.DataFrame:
    """Share of total split gain per (stat, feature)."""
    rows = []
    for s in model.boosters:
        b = model.booster(s)
        g = b.feature_importance(importance_type=kind)
        tot = g.sum() or 1.0
        rows += [{"stat": s, "feature": f, "share": float(v / tot)}
                 for f, v in zip(b.feature_name(), g)]
    return pd.DataFrame(rows)


# ============================================================================== pipeline hook

def apply_rates(model: ContextModel, rates: pd.DataFrame, team_feats: pd.DataFrame,
                game_id: Sequence, team: Sequence,
                player_feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scale the EB rates of a `state.rate_frame` result (one row per panel row) by the
    context multipliers: r_<s> *= m_s for the DK stats, r_dk recomputed. `team_feats` =
    `features.team_game_features` rows (game_id, team, F1/F2 (+ F6) columns) for the day;
    `player_feats` = per (game_id, player_id) F3 / F5 columns (`features.LiveExtras.day`)."""
    if not model.boosters:
        return rates
    X = rates.copy()
    X["game_id"] = np.asarray(game_id)
    X["team"] = np.asarray(team)
    if player_feats is not None and "player_id" not in X:
        raise ValueError("player features need rates.player_id")
    X = FX.attach(X, team_feats, player_feats)
    mult = model.predict(X)
    out = rates.copy()
    for s in DK_STATS:
        out[f"r_{s}"] = rates[f"r_{s}"].to_numpy(float) * mult[s]
    out["r_dk"] = ST.dk_rate({s: out[f"r_{s}"].to_numpy(float) for s in ST.DK_WEIGHTS})
    return out


# ============================================================================== evaluation

def two_way_se(d: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float:
    """Cameron-Gelbach-Miller two-way clustered SE of mean(d): V1 + V2 - V12 (V12 = the
    intersection, here the row), floored at max(V1, V2)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return float("nan")
    e = d - d.mean()

    def v(c):
        s = pd.Series(e).groupby(np.asarray(c)).sum().to_numpy()
        G = len(s)
        return G / max(G - 1, 1) * float(np.sum(s ** 2))
    v1, v2 = v(g1), v(g2)
    v12 = n / (n - 1) * float(np.sum(e ** 2))
    return float(np.sqrt(max(v1 + v2 - v12, v1, v2)) / n)


def _dk_actual(rows: pd.DataFrame) -> np.ndarray:
    from .state_eval import DD_CATS
    cats = sum((rows[s].to_numpy(float) >= 10).astype(int) for s in DD_CATS)
    return (sum(DK_W[s] * rows[s].to_numpy(float) for s in DK_STATS)
            + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))


def dk_mean(rows: pd.DataFrame, mult: dict | None) -> np.ndarray:
    """E[DK | actual minutes] = sum_s w_s E_s m_s + E[DD/TD bonus] (independent NB)."""
    from .state_eval import DD_CATS, expected_bonus
    mu = {s: rows[f"E_{s}"].to_numpy(float) * (1.0 if mult is None else mult[s])
          for s in DK_STATS}
    phi = {s: rows[f"d_{s}"].to_numpy(float) for s in DD_CATS}
    bon = expected_bonus({s: np.maximum(mu[s], 1e-9) for s in DD_CATS}, phi)
    return sum(DK_W[s] * mu[s] for s in DK_STATS) + bon


def cv_table(rows: pd.DataFrame) -> tuple[list, list]:
    """CV-by-level of DK given actual minutes (copula.fit_cv) on base residuals."""
    from .copula import fit_cv
    mu = dk_mean(rows, None)
    return fit_cv(mu, _dk_actual(rows) - mu)


def _date_se(d: np.ndarray, dates: np.ndarray) -> float:
    from .metrics import paired_cluster
    return float(paired_cluster(d, dates)["se"])


def evaluate(test: pd.DataFrame, mults: dict, cv: tuple, *, pairs: Sequence[tuple],
             masks: dict | None = None) -> dict:
    """NB log-lik (sum of the 7 DK stats, and per stat) and DK MAE / CRPS given actual minutes,
    per variant, and paired differences with two-way clustered SEs (game x player-season) and
    date-clustered SEs (`se_date`). `mults` {variant: {stat: (n,)} or None (= base)};
    pairs [(a, b)] -> a minus b."""
    from .metrics import crps_normal
    y_dk = _dk_actual(test)
    g1 = test["game_id"].to_numpy()
    g2 = (test["player_id"].astype(str) + "|" + test["season"]).to_numpy()
    gd = (test["game_date"].astype(str).to_numpy() if "game_date" in test
          else test["game_id"].to_numpy())
    masks = masks or {"all": np.ones(len(test), bool)}
    per = {}
    for v, m in mults.items():
        ll = {s: ST.nb_logpmf(test[s].to_numpy(float),
                              np.maximum(test[f"E_{s}"].to_numpy(float)
                                         * (1.0 if m is None else m[s]), 1e-9),
                              test[f"d_{s}"].to_numpy(float)) for s in DK_STATS}
        mu = dk_mean(test, m)
        sd = np.interp(mu, cv[0], cv[1]) * np.maximum(mu, 0.5)
        per[v] = {"ll": ll, "ll_tot": sum(ll.values()), "ae": np.abs(y_dk - mu),
                  "crps": crps_normal(mu, sd, y_dk), "mu": mu}
    out = {"n": {k: int(mk.sum()) for k, mk in masks.items()}, "variants": {}, "pairs": {}}
    for k, mk in masks.items():
        for v, r in per.items():
            out["variants"].setdefault(v, {})[k] = {
                "ll": float(r["ll_tot"][mk].mean()), "mae": float(r["ae"][mk].mean()),
                "crps": float(r["crps"][mk].mean()), "bias": float((y_dk - r["mu"])[mk].mean()),
                "ll_by_stat": {s: float(r["ll"][s][mk].mean()) for s in DK_STATS}}
        for a, b in pairs:
            ra, rb = per[a], per[b]
            res = {}
            for met, sign in (("ll", 1.0), ("mae", -1.0), ("crps", -1.0)):
                key = "ll_tot" if met == "ll" else ("ae" if met == "mae" else "crps")
                d = sign * (ra[key] - rb[key])[mk]           # positive = a better
                res[met] = {"diff": float(d.mean()),
                            "se": two_way_se(d, g1[mk], g2[mk]),
                            "se_date": _date_se(d, gd[mk])}
            res["ll_by_stat"] = {}
            for s in DK_STATS:
                d = (ra["ll"][s] - rb["ll"][s])[mk]
                res["ll_by_stat"][s] = {"diff": float(d.mean()),
                                        "se": two_way_se(d, g1[mk], g2[mk]),
                                        "se_date": _date_se(d, gd[mk])}
            out["pairs"].setdefault(f"{a} vs {b}", {})[k] = res
    return out


PAIRS = (("P", "base"), ("P+F1", "P"), ("P+F2", "P"), ("P+F1+F2", "P+F2"),
         ("P+F1+F2", "P+F1"), ("P+F1+F2", "base"),
         ("P+F2+F3", "P+F2"), ("P+F2+F5", "P+F2"), ("P+F2+F6", "P+F2"),
         ("P+F2+F3+F5+F6", "P+F2"),
         ("P+F2+F3+F5", "P+F2+F3"), ("P+F2+F3+F6", "P+F2+F3"),
         ("P+F2+F3+F5+F6", "P+F2+F3"), ("P+F2+F3+F5+F6", "P+F2+F3+F6"),
         ("P+F2+F3+F5+F6", "P+F2+F3+F5"))


def state_params_for(conn: sqlite3.Connection, season: str) -> dict:
    """Rate-state params fit on seasons < S: <data>/proj/features/state_params_<S>.json, then
    <data>/proj/state_params_<S>.json / the bundle's, else `pipeline.fit_state_params`."""
    from .. import config
    for p in (config.data_dir() / "proj" / "features" / f"state_params_{season}.json",
              config.data_dir() / "proj" / f"state_params_{season}.json"):
        if p.exists():
            return json.loads(p.read_text())
    from . import pipeline as P
    return P.fit_state_params(conn, season)


def model_path(season: str, variant: str) -> Path:
    from .. import config
    return config.data_dir() / "proj" / "features" / f"context_{season}_{variant}.json"


def sanity(test: pd.DataFrame, mults: dict) -> dict:
    """Big-man rebounding and guard assists vs opponents missing the role (test rows)."""
    out = {}
    m = mults.get("P+F1+F2") or mults.get("P+F1")
    if m is None:
        return out
    mpg = test["mpg"].to_numpy(float) >= 15
    cases = {
        "C reb, opp starting C out (vac_C>=24)": ("reb", (test["grp"] == 2).to_numpy()
                                                  & (test["f1_vac_C"] >= 24).to_numpy()),
        "C reb, opp healthy": ("reb", (test["grp"] == 2).to_numpy()
                               & (test["f1_n_abs"] == 0).to_numpy()),
        "C reb, opp top rebounder out": ("reb", (test["grp"] == 2).to_numpy()
                                         & (test["f1_reb_top"] == 1).to_numpy()),
        "C blk, opp rim protector out": ("blk", (test["grp"] == 2).to_numpy()
                                         & (test["f1_rim_top"] == 1).to_numpy()),
        "all pts, opp rim protector out": ("pts", (test["f1_rim_top"] == 1).to_numpy()),
        "all pts, opp healthy": ("pts", (test["f1_n_abs"] == 0).to_numpy()),
        "G stl, opp ball-handler out": ("stl", (test["grp"] == 0).to_numpy()
                                        & (test["f1_bh_top"] == 1).to_numpy()),
        "G ast, opp perimeter defender out": ("ast", (test["grp"] == 0).to_numpy()
                                              & (test["f1_perim_top"] == 1).to_numpy()),
    }
    for lab, (s, mk) in cases.items():
        mk = mk & mpg
        E = test[f"E_{s}"].to_numpy(float)[mk]
        out[lab] = {"n": int(mk.sum()),
                    "actual/EB (sum ratio)": float(test[s].to_numpy(float)[mk].sum()
                                                   / max(E.sum(), 1e-9)),
                    "model mult (E-weighted)": float((m[s][mk] * E).sum() / max(E.sum(), 1e-9))}
    return out


def extract_proxy(test: pd.DataFrame, mults: dict, extract: Path, *,
                  alpha: float = 0.5) -> dict:
    """Fast end-to-end proxy on the T2.7 projection extract (`corr_extract_<S>.pkl`: the
    pipeline's pregame DK mean `mu` and minutes `min_proj` per player-game, avail protocol).
    mu_new = mu x sum_s a_s m'_s / sum_s a_s with a_s = w_s r_s min_proj, where the points
    multiplier is renormalised per team-game (the pipeline reconciles team points to the
    implied total; the other stats follow with c^alpha). Rows: players who played."""
    ex = pd.read_pickle(extract)["rows"]
    t = test[["game_id", "player_id", "season", "team"] + [f"r_{s}" for s in DK_STATS]
             + list(DK_STATS)].copy()
    t["_i"] = np.arange(len(t))
    t["dk"] = _dk_actual(test)
    d = ex[["game_id", "player_id", "mu", "min_proj"]].merge(t, on=["game_id", "player_id"])
    idx = d["_i"].to_numpy()
    a = {s: DK_W[s] * d[f"r_{s}"].to_numpy(float) * d["min_proj"].to_numpy(float)
         for s in DK_STATS}
    tot = sum(a.values())
    key = (d["game_id"] + "|" + d["team"]).to_numpy()
    g1 = d["game_id"].to_numpy()
    g2 = (d["player_id"].astype(str) + "|" + d["season"]).to_numpy()
    y = d["dk"].to_numpy(float)
    out = {"n": int(len(d)), "variants": {}, "pairs": {}}
    mus = {}
    for v, m in mults.items():
        if m is None:
            mus[v] = d["mu"].to_numpy(float)
            continue
        mm = {s: m[s][idx] for s in DK_STATS}
        ap = pd.Series(a["pts"]).groupby(key)
        k = (ap.transform("sum") / pd.Series(a["pts"] * mm["pts"]).groupby(key)
             .transform("sum")).to_numpy()
        new = a["pts"] * mm["pts"] * k + sum(a[s] * mm[s] * k ** alpha
                                             for s in DK_STATS if s != "pts")
        mus[v] = d["mu"].to_numpy(float) * new / np.maximum(tot, 1e-9)
    big = d["min_proj"].to_numpy(float) >= 15
    for v, mu in mus.items():
        out["variants"][v] = {"mae": float(np.abs(y - mu).mean()),
                              "mae_min15": float(np.abs(y - mu)[big].mean()),
                              "bias": float((y - mu).mean())}
    for a_, b_ in PAIRS:
        if a_ in mus and b_ in mus:
            dd = np.abs(y - mus[b_]) - np.abs(y - mus[a_])
            out["pairs"][f"{a_} vs {b_}"] = {"mae": float(dd.mean()),
                                             "se": two_way_se(dd, g1, g2)}
    return out


def run_ablation(conn: sqlite3.Connection, season: str, *, n_train: int = 6,
                 variants: dict | None = None, frame: pd.DataFrame | None = None,
                 base: str = "E", center: bool = False, save_models: bool = True,
                 log: Callable[[str], None] | None = None) -> dict:
    """Walk-forward ablation for test season S (models fit on the n_train seasons < S).
    base 'E' = EB x actual minutes x the market reconciliation (the pipeline's mean, default);
    'E0' = without the market (what a layer would learn if the pipeline had no total)."""
    say = log or (lambda m: None)
    t0 = time.time()
    variants = variants or VARIANTS
    sp = state_params_for(conn, season)
    fr = frame if frame is not None else FX.build_frame(conn, season, sp, n_train=n_train,
                                                         log=say)
    ext = sorted({g for v in variants.values() for g in v if g in FX.EXTENSION_GROUPS})
    if ext:
        fr = FX.extend_frame(conn, fr, ext, season=season, log=say)
    if base != "E":
        fr = fr.copy()
        for s in DK_STATS:
            fr[f"E_{s}"] = fr[f"{base}_{s}"]
    rs = (fr["season_type"] == "Regular Season").to_numpy()
    train = rs & (fr["season"] < season).to_numpy()
    test_m = rs & (fr["season"] == season).to_numpy()
    test = fr[test_m].reset_index(drop=True)
    cv = cv_table(fr[train])
    mults, models, imp = {}, {}, {}
    fit_s = {}
    for v, groups in variants.items():
        t1 = time.time()
        mdl = fit(fr, train, groups, season=season, center=center, log=say)
        fit_s[v] = round(time.time() - t1, 1)
        models[v] = mdl
        mults[v] = None if not groups else mdl.predict(test)
        if groups:
            imp[v] = importances(mdl)
        if save_models and groups and base == "E" and not center:
            save(mdl, model_path(season, v))
    masks = {"all": np.ones(len(test), bool), "mpg15": (test["mpg"] >= 15).to_numpy()}
    pairs = [p for p in PAIRS if p[0] in mults and p[1] in mults]
    ev = evaluate(test, mults, cv, pairs=pairs, masks=masks)
    ev["season"] = season
    ev["base"], ev["center"] = base, center
    ev["train_seasons"] = sorted(fr.loc[train, "season"].unique().tolist())
    ev["fit_seconds"] = fit_s
    ev["n_rounds"] = {v: m.n_rounds for v, m in models.items()}
    ev["level"] = {s: float(test[s].sum() / test[f"E_{s}"].sum()) for s in DK_STATS}
    ev["mult_summary"] = {v: {s: {"sd_log": float(np.std(np.log(mm[s]))),
                                  "wmean": float((mm[s] * test[f"E_{s}"]).sum()
                                                 / test[f"E_{s}"].sum()),
                                  "p05": float(np.quantile(mm[s], 0.05)),
                                  "p95": float(np.quantile(mm[s], 0.95)),
                                  "clipped": float(np.mean((mm[s] <= CLIP[0] + 1e-9)
                                                           | (mm[s] >= CLIP[1] - 1e-9)))}
                              for s in DK_STATS}
                          for v, mm in mults.items() if mm is not None}
    ev["importance"] = {v: d.sort_values("share", ascending=False).groupby("stat").head(8)
                        .to_dict("records") for v, d in imp.items()}
    ev["sanity"] = sanity(test, mults)
    from .. import config
    for ex in (config.data_dir() / "proj" / f"corr_extract_{season}.pkl",
               config.REPO_ROOT / "var" / "proj" / f"corr_extract_{season}.pkl"):
        if ex.exists() and base == "E":
            ev["extract_proxy"] = extract_proxy(test, mults, ex)
            break
    ev["seconds"] = round(time.time() - t0, 1)
    return ev


def format_ablation(ev: dict) -> str:
    L = [f"== {ev['season']} (fit on {ev['train_seasons'][0]}..{ev['train_seasons'][-1]}; "
         f"base {ev.get('base', 'E')}{', centered' if ev.get('center') else ''}; "
         f"{ev['seconds']}s)"]
    for k, n in ev["n"].items():
        L.append(f"  population {k}: n={n}")
        L.append(f"    {'variant':10s} {'NB LL/row':>10s} {'DK MAE':>8s} {'DK CRPS':>8s} "
                 f"{'bias':>7s}")
        for v, r in ev["variants"].items():
            x = r[k]
            L.append(f"    {v:10s} {x['ll']:10.4f} {x['mae']:8.4f} {x['crps']:8.4f} "
                     f"{x['bias']:+7.3f}")
        L.append("    paired (positive = first better; SE two-way game x player-season):")
        for p, d in ev["pairs"].items():
            x = d[k]
            sd = {m: x[m].get("se_date", float("nan")) for m in ("ll", "mae", "crps")}
            L.append(f"    {p:20s} LL {x['ll']['diff']:+.5f} (SE {x['ll']['se']:.5f} / date "
                     f"{sd['ll']:.5f}, z {x['ll']['diff'] / max(x['ll']['se'], sd['ll']):+.1f})"
                     f"  MAE {x['mae']['diff']:+.4f} (SE {x['mae']['se']:.4f} / "
                     f"{sd['mae']:.4f})  CRPS {x['crps']['diff']:+.4f} "
                     f"(SE {x['crps']['se']:.4f} / {sd['crps']:.4f}, z "
                     f"{x['crps']['diff'] / max(x['crps']['se'], sd['crps']):+.1f})")
            L.append("      LL by stat: " + "  ".join(
                f"{s} {v['diff']:+.4f}/{max(v['se'], v.get('se_date', 0.0)):.4f}"
                for s, v in x["ll_by_stat"].items()))
    if "extract_proxy" in ev:
        x = ev["extract_proxy"]
        L.append(f"  end-to-end proxy (T2.7 extract, pipeline mu x context, played, n={x['n']}):")
        for v, r in x["variants"].items():
            L.append(f"    {v:10s} DK MAE {r['mae']:.4f} (min_proj>=15 {r['mae_min15']:.4f}) "
                     f"bias {r['bias']:+.3f}")
        for p, r in x["pairs"].items():
            L.append(f"    {p:20s} MAE {r['mae']:+.4f} (SE {r['se']:.4f})")
    return "\n".join(L)


# ============================================================================== CLI

def _conn(path: str | None):
    from .. import config
    p = path or str(config.db_path())
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from .. import config
    ap = argparse.ArgumentParser(description="context layer (Phase 2F)")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("dvp-params", help="freeze the F2 DvP shrinkage on 2010-11..2013-14")
    a1.add_argument("--sp-season", default="2019-20")
    a2 = sub.add_parser("ablation")
    a2.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25"])
    a2.add_argument("--n-train", type=int, default=6)
    a2.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a2.add_argument("--out", default=None)
    a2.add_argument("--base", default="E", choices=("E", "E0"))
    a2.add_argument("--center", action="store_true")
    a3 = sub.add_parser("fit", help="fit and save the layer for season S (seasons < S)")
    a3.add_argument("--season", required=True)
    a3.add_argument("--groups", nargs="+", default=["P", "F2"])
    a3.add_argument("--n-train", type=int, default=6)
    a4 = sub.add_parser("bundle", help="fit / load the pipeline's context-ON bundle of season S")
    a4.add_argument("--season", required=True)
    a4.add_argument("--refit", action="store_true", help="refit everything (minutes included)")
    a = ap.parse_args(argv)
    conn = _conn(a.db)
    if a.cmd == "dvp-params":
        from .features.common import base_rows
        from .features.opp_defense import fit_dvp_params
        sp = state_params_for(conn, a.sp_season)
        wf = base_rows(conn, ["2009-10", "2010-11", "2011-12", "2012-13", "2013-14"], sp)
        p = fit_dvp_params(wf, ["2010-11", "2011-12", "2012-13", "2013-14"], log=print)
        print(json.dumps(p, indent=1, sort_keys=True))
        return 0
    if a.cmd == "bundle":
        from . import pipeline as P
        was = P.CONTEXT_ON
        P.set_context(True)
        try:
            t0 = time.time()
            b = P.load_bundle(conn, a.season, refit=a.refit, log=print)
            path = P.cache_dir() / f"bundle_{a.season}{P._VS}.json"
            print(f"{P.MODEL_VERSION} bundle {path} ({time.time() - t0:.0f}s): context "
                  f"{'+'.join(b.context.groups)} fit on {b.context.train_seasons}, rounds "
                  f"{b.context.n_rounds}")
        finally:
            P.set_context(was)
        return 0
    if a.cmd == "fit":
        sp = state_params_for(conn, a.season)
        fr = FX.build_frame(conn, a.season, sp, n_train=a.n_train, log=print)
        train = (fr["season"] < a.season).to_numpy()
        m = fit(fr, train, a.groups, season=a.season, log=print)
        path = model_path(a.season, "+".join(a.groups))
        save(m, path)
        print(f"wrote {path}")
        return 0
    if a.cmd == "ablation":
        res = []
        for s in a.seasons:
            ev = run_ablation(conn, s, n_train=a.n_train, base=a.base, center=a.center,
                              variants={v: VARIANTS[v] for v in a.variants}, log=print)
            print(format_ablation(ev), flush=True)
            res.append(ev)
        out = Path(a.out) if a.out else config.data_dir() / "reports" / "context_ablation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
