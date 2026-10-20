"""F4: shot quality and shooting luck from the cdn play-by-play (docs/PLAYERS.md Phase 2F).

Three layers:

1. **Shot store** (`parse_game`, `update_store`, `load_shots`): every field-goal attempt and free
   throw of the archived cdn play-by-play (`<data>/raw/nba_cdn/playbyplay/<day>/<gid>.json.gz`,
   2019-20 on; an archived file is a final game) with the shooter, made, legacy x / y (feet from
   the basket), distance, shot type (jump / layup / dunk / hook), descriptor flags (driving,
   pull-up, step-back, fadeaway, turnaround, running, floating, cutting, finger roll, tip, bank,
   putback, reverse, alley-oop), context qualifiers (second chance, fast break, off a turnover),
   period / seconds left and the score before the shot. Assisted (made shots only) and blocked
   (misses only) are recorded but are outcome-dependent: they never enter the xFG model, only
   the player's shot-diet history. Cached per season as parquet under
   `<data>/proj/features/shots/` and extended incrementally (new archive files only).

2. **xFG model** (`XFGModel`, `fit_xfg`): P(make) per field-goal attempt, LightGBM binary on the
   pre-shot features only (`SHOT_FEATURES`), fit on the shots of seasons < S (rounds by early
   stopping on the last of them, then refit on all; one thread, deterministic). `fit_baseline`
   is the distance-only reference (empirical make rate per 2 / 3 x 1-ft distance bin).

3. **Player / opponent features** (`player_features`, `build`): strictly before the game day
   (`common.ew_asof`), from per player-game sums of the shots (games without archived PBP carry
   no minutes, so they neither dilute nor inform the rates):
   * shot quality: xFG% on 2s and 3s and expected points per FGA vs the league (shrunk);
   * shot diet: rim (< 4 ft) / short 2 (4-14 ft) / mid 2 (>= 14 ft) / corner 3 / above-the-break 3
     shares of FGA, 3PA rate, free-throw rate, FT%, assisted share of made 2s / 3s (shrunk to the
     player's position group);
   * shooting over expected (long memory: half-life 365 d): (made - xFG - league drift) per
     attempt, 2s and 3s, heavily shrunk ("skill");
   * **shooting luck inside the EB rates** (row level, `row_features`): the made-minus-expected
     points (and 3PM) accumulated with the EB state's own decay (pts: half-life 90 d; fg3m: 120 d)
     divided by the EB evidence (n_<s> + kappa_s) and the EB rate: the share of the player's EB
     rate that is shooting luck, f4_lk_pts / f4_lk_fg3m. A layer that learns m < 1 for positive
     luck regresses hot shooters to their shot quality. Monotone -1.
   * the opponent's shot quality and luck allowed (per defending team, half-life 120 d).
   Rows before the shot era are NaN (no source), a player without shots in the era sits at the
   prior with f4_ln_fga 0.
"""

from __future__ import annotations

import gzip
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ... import config
from .. import state as ST
from .common import ew_asof, season_label

SHOT_START = "2019-20"
STORE_VERSION = 1
SUBTYPES = ("jumpshot", "layup", "dunk", "hook")          # code 0..3, other 4
DESCS = ("driving", "pullup", "stepback", "fadeaway", "turnaround", "running", "floating",
         "cutting", "fingerroll", "tip", "bank", "putback", "reverse", "alleyoop")
QUALS = ("2ndchance", "fastbreak", "fromturnover")
SHOT_FEATURES = (("is3", "dist", "ax", "ay", "sub") + tuple(f"d_{d}" for d in DESCS)
                 + tuple(f"q_{q}" for q in QUALS) + ("secs", "ot", "margin"))
STORE_COLS = (["game_id", "order", "team", "player_id", "kind", "made", "dist", "ax", "ay",
               "xl", "sub"] + [f"d_{d}" for d in DESCS] + [f"q_{q}" for q in QUALS]
              + ["period", "secs", "sh", "sa", "assisted", "blocked"])


# ============================================================================== parsing

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _i(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _clock(s) -> float:
    from ...analytics.pbp import parse_clock
    return parse_clock(s)


def parse_game(payload: dict, game_id: str) -> pd.DataFrame:
    """Field-goal attempts (kind 2 / 3) and free throws (kind 1) of one cdn play-by-play
    payload, in feed order. Score columns sh / sa are BEFORE the event."""
    acts = ((payload or {}).get("game") or {}).get("actions") or []
    acts = sorted(acts, key=lambda a: (_i(a.get("orderNumber")), _i(a.get("actionNumber"))))
    rows = []
    sh = sa = 0
    for a in acts:
        t = str(a.get("actionType") or "").lower()
        nh, na = a.get("scoreHome"), a.get("scoreAway")
        if t in ("2pt", "3pt", "freethrow") and a.get("personId"):
            res = str(a.get("shotResult") or "").lstrip(": ").lower()
            if res in ("made", "missed"):
                made = int(res == "made")
                if t == "freethrow":
                    kind = 1
                    dist = ax = ay = xl = np.nan
                    sub = 4
                    dflags = [0] * len(DESCS)
                    qflags = [0] * len(QUALS)
                else:
                    kind = 3 if t == "3pt" else 2
                    xl, yl = _f(a.get("xLegacy")), _f(a.get("yLegacy"))
                    dist = _f(a.get("shotDistance"))
                    ax, ay = abs(xl) / 10.0, yl / 10.0
                    if not np.isfinite(dist) and np.isfinite(ax):
                        dist = math.hypot(ax, ay)
                    st = str(a.get("subType") or "").lower().replace(" ", "")
                    sub = SUBTYPES.index(st) if st in SUBTYPES else 4
                    ds = str(a.get("descriptor") or "").lower().replace(" ", "").replace("-", "")
                    dflags = [int(d in ds) for d in DESCS]
                    q = {str(x).lower() for x in (a.get("qualifiers") or [])}
                    qflags = [int(x in q) for x in QUALS]
                    xl = xl / 10.0
                rows.append((game_id, _i(a.get("orderNumber")), str(a.get("teamTricode") or ""),
                             _i(a.get("personId")), kind, made, dist, ax, ay, xl, sub,
                             *dflags, *qflags, _i(a.get("period"), 1), _clock(a.get("clock")),
                             sh, sa,
                             int(bool(a.get("assistPersonId"))) if made and kind > 1 else 0,
                             int(bool(a.get("blockPersonId"))) if not made and kind > 1 else 0))
        if nh is not None and na is not None:
            sh, sa = _i(nh, sh), _i(na, sa)
    df = pd.DataFrame(rows, columns=STORE_COLS)
    return _types(df)


def _types(df: pd.DataFrame) -> pd.DataFrame:
    i8 = ["kind", "made", "sub", "period", "assisted", "blocked"] + \
        [f"d_{d}" for d in DESCS] + [f"q_{q}" for q in QUALS]
    for c in i8:
        df[c] = df[c].astype(np.int8)
    for c in ("dist", "ax", "ay", "xl", "secs"):
        df[c] = df[c].astype(np.float32)
    for c in ("sh", "sa", "order"):
        df[c] = df[c].astype(np.int32)
    df["player_id"] = df["player_id"].astype(np.int64)
    return df


def season_of_gid(gid: str) -> str:
    return season_label(2000 + int(gid[3:5]))


def pbp_root() -> Path:
    return config.data_dir() / "raw" / "nba_cdn" / "playbyplay"


def store_dir() -> Path:
    return config.data_dir() / "proj" / "features" / "shots"


def archived(seasons: Iterable[str] | None = None, root: Path | None = None
             ) -> dict[str, list[tuple[str, Path]]]:
    """{season: [(game_id, path)]} of the archived regular-season / play-in / playoff PBP."""
    root = root or pbp_root()
    want = set(seasons) if seasons else None
    out: dict[str, list] = {}
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json.gz")):
            gid = f.name[:-len(".json.gz")]
            if len(gid) != 10 or not gid.isdigit() or gid[2] not in "245":
                continue
            s = season_of_gid(gid)
            if s < SHOT_START or (want is not None and s not in want):
                continue
            out.setdefault(s, []).append((gid, f))
    return out


def update_store(seasons: Iterable[str] | None = None, *, root: Path | None = None,
                 write: bool = True, log: Callable[[str], None] | None = None
                 ) -> dict[str, pd.DataFrame]:
    """Parse the archive files not in the season parquet yet and rewrite it (incremental).
    Returns {season: shots}. With write=False (read-only data dir) nothing is written."""
    say = log or (lambda m: None)
    out = {}
    arch = archived(seasons, root)
    for s in sorted(set(arch) | set(seasons or [])):
        path = store_dir() / f"shots_{s}_v{STORE_VERSION}.parquet"
        have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=STORE_COLS)
        done = set(have["game_id"].unique()) if len(have) else set()
        todo = [(g, f) for g, f in arch.get(s, []) if g not in done]
        if todo:
            t0 = time.time()
            parts = [have] if len(have) else []
            for g, f in todo:
                try:
                    parts.append(parse_game(json.loads(gzip.decompress(f.read_bytes())), g))
                except (OSError, ValueError) as e:            # a broken file: skip, log
                    say(f"shots: {g} unreadable ({type(e).__name__})")
            have = _types(pd.concat(parts, ignore_index=True)).sort_values(
                ["game_id", "order"], kind="mergesort").reset_index(drop=True)
            say(f"shots {s}: +{len(todo)} games -> {have['game_id'].nunique()} games, "
                f"{len(have)} shots ({time.time() - t0:.0f}s)")
            if write:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    have.to_parquet(tmp)
                    tmp.replace(path)
                except OSError as e:
                    say(f"shots: store not written ({e})")
        out[s] = have
    return out


def load_shots(conn: sqlite3.Connection, first: str | None = None, last: str | None = None, *,
               shots: pd.DataFrame | None = None, update: bool = True,
               log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Shots of seasons [first, last] joined to team_games (game_date, day, sidx, season_type,
    the shooter's side, the defending team): only games present in team_games are kept, so a
    truncated database truncates the shots. `shots` overrides the store (tests)."""
    first = max(first or SHOT_START, SHOT_START)
    if not last:
        last = conn.execute("SELECT MAX(season) FROM team_games").fetchone()[0] or first
    if shots is None:
        seasons = [season_label(y) for y in range(int(first[:4]), int(last[:4]) + 1)]
        st = update_store(seasons, log=log) if update else {
            s: pd.read_parquet(p) for s in seasons
            for p in [store_dir() / f"shots_{s}_v{STORE_VERSION}.parquet"] if p.exists()}
        parts = [v for k, v in st.items() if first <= k <= last and len(v)]
        shots = (pd.concat(parts, ignore_index=True) if parts
                 else _types(pd.DataFrame(columns=STORE_COLS)))
    tg = pd.read_sql_query(
        "SELECT game_id, team, opp, season, season_type, game_date, is_home FROM team_games "
        "WHERE season >= ? AND season <= ?", conn, params=(first, last))
    s = shots.merge(tg, on=["game_id", "team"], how="inner")
    s["day"] = ST._to_day(s["game_date"]) if len(s) else np.zeros(0, np.int64)
    s["sidx"] = s["season"].str[:4].astype(int) if len(s) else np.zeros(0, int)
    home = s["is_home"].astype(int).to_numpy() == 1
    m = np.where(home, s["sh"] - s["sa"], s["sa"] - s["sh"]).astype(float)
    s["margin"] = np.clip(m, -30, 30)
    s["is3"] = (s["kind"] == 3).astype(np.int8)
    s["ot"] = (s["period"] > 4).astype(np.int8)
    return s.sort_values(["day", "game_id", "order"], kind="mergesort").reset_index(drop=True)


# ============================================================================== xFG model

XFG_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_data_in_leaf": 400, "lambda_l2": 10.0, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "seed": 11, "deterministic": True, "force_row_wise": True,
    "num_threads": 1, "verbose": -1, "max_bin": 127,
}
XFG_MAX_ROUNDS, XFG_EARLY = 1500, 50


@dataclass
class XFGModel:
    booster: str
    features: list
    train_seasons: list
    n_rounds: int
    n_train: int
    platt: dict = field(default_factory=dict)    # kind "2" / "3" -> [intercept, slope]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self._b = None

    def raw(self, shots: pd.DataFrame) -> np.ndarray:
        import lightgbm as lgb
        if self._b is None:
            self._b = lgb.Booster(model_str=self.booster)
        if not len(shots):
            return np.zeros(0)
        return self._b.predict(shots[self.features].to_numpy(float), raw_score=True,
                               num_threads=1)

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        """P(make) per FGA, Platt-recalibrated per 2 / 3."""
        z = self.raw(shots)
        if self.platt and len(z):
            k = shots["kind"].to_numpy(int)
            for kk, (a, b) in self.platt.items():
                mk = k == int(kk)
                z[mk] = a + b * z[mk]
        return 1.0 / (1.0 + np.exp(-z))

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items()}, sort_keys=True)

    @classmethod
    def from_json(cls, s: str | dict) -> "XFGModel":
        return cls(**(json.loads(s) if isinstance(s, str) else s))


def _fga(shots: pd.DataFrame) -> pd.DataFrame:
    return shots[(shots["kind"] > 1) & np.isfinite(shots["dist"].to_numpy(float))]


def fit_xfg(shots: pd.DataFrame, *, params: dict | None = None,
            log: Callable[[str], None] | None = None) -> XFGModel:
    """xFG on the regular-season + playoff FGAs of `shots` (the caller passes seasons < S only):
    rounds by early stopping on the last season, then a refit on all."""
    import lightgbm as lgb
    say = log or (lambda m: None)
    t0 = time.time()
    f = _fga(shots)
    seasons = sorted(f["season"].unique())
    cols = list(SHOT_FEATURES)
    X, y = f[cols].to_numpy(float), f["made"].to_numpy(float)
    prm = dict(XFG_PARAMS, **(params or {}))
    cat = [cols.index("sub")]
    n_best = 400
    platt = {}
    if len(seasons) > 1:
        ii = (f["season"] < seasons[-1]).to_numpy()
        dtr = lgb.Dataset(X[ii], y[ii], feature_name=cols, categorical_feature=cat,
                          free_raw_data=False)
        dva = lgb.Dataset(X[~ii], y[~ii], reference=dtr, categorical_feature=cat,
                          free_raw_data=False)
        b = lgb.train(prm, dtr, num_boost_round=XFG_MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(XFG_EARLY, verbose=False)])
        n_best = max(int(b.best_iteration or 0), 1)
        # per-kind Platt scaling on the held-out last season (the trees over-spread 3s)
        zv = b.predict(X[~ii], num_iteration=n_best, raw_score=True)
        kv = f["kind"].to_numpy(int)[~ii]
        for kk in (2, 3):
            mk = kv == kk
            if mk.sum() > 1000:
                platt[str(kk)] = list(_logit_fit(y[~ii][mk], zv[mk]))
    b = lgb.train(prm, lgb.Dataset(X, y, feature_name=cols, categorical_feature=cat),
                  num_boost_round=n_best)
    say(f"xFG fit on {seasons} ({len(f)} FGA, {n_best} rounds, {time.time() - t0:.0f}s)")
    return XFGModel(booster=b.model_to_string(), features=cols, train_seasons=seasons,
                    n_rounds=n_best, n_train=int(len(f)), platt=platt)


def _logit_fit(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """(intercept, slope) of the logistic regression of y on the logit z (Newton)."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    a, b = 0.0, 1.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + b * z)))
        w = mu * (1 - mu)
        g = np.array([np.sum(y - mu), np.sum((y - mu) * z)])
        Hm = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z)]])
        step = np.linalg.solve(Hm, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def dist_bin(shots: pd.DataFrame) -> np.ndarray:
    d = np.clip(np.nan_to_num(shots["dist"].to_numpy(float), nan=0.0), 0, 47)
    b = np.where(d < 30, np.floor(d), np.where(d < 36, 30, np.where(d < 47, 31, 32)))
    return (shots["is3"].to_numpy(int) * 100 + b).astype(int)


def fit_baseline(shots: pd.DataFrame) -> dict:
    """Distance-only reference: make rate per (2 / 3, 1-ft distance bin), +1/+2 smoothed."""
    f = _fga(shots)
    g = pd.DataFrame({"b": dist_bin(f), "m": f["made"].to_numpy(float)}).groupby("b")["m"]
    return {int(k): float((v.sum() + 1) / (len(v) + 2)) for k, v in g}


def predict_baseline(table: dict, shots: pd.DataFrame) -> np.ndarray:
    b = dist_bin(shots)
    glob = float(np.mean(list(table.values()))) if table else 0.45
    return np.array([table.get(int(k), glob) for k in b])


def xfg_quality(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> dict:
    """Log-loss, Brier, calibration (ECE over probability deciles-ish bins, logistic slope /
    intercept of y on logit p) and the mean predicted vs actual."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        mk = idx == k
        if mk.any():
            ece += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    a, b = _logit_fit(y, np.log(p / (1 - p)))
    return {"n": int(len(y)), "logloss": ll, "brier": float(np.mean((y - p) ** 2)),
            "ece": float(ece), "cal_intercept": float(a), "cal_slope": float(b),
            "mean_p": float(p.mean()), "mean_y": float(y.mean())}


def xfg_cache_path(season: str) -> Path:
    return config.data_dir() / "proj" / "features" / f"xfg_{season}.json"


def xfg_for_season(conn: sqlite3.Connection, season: str, *, shots: pd.DataFrame | None = None,
                   cache: bool = True, log: Callable[[str], None] | None = None) -> XFGModel:
    """The xFG model of season S: fit on the shots of seasons SHOT_START .. S-1 (walk-forward),
    cached as xfg_<S>.json keyed by the training-shot count."""
    prev = season_label(int(season[:4]) - 1)
    sh = shots if shots is not None else load_shots(conn, SHOT_START, prev, log=log)
    sh = sh[sh["season"] < season]
    n = int(len(_fga(sh)))
    path = xfg_cache_path(season)
    if cache and path.exists():
        m = XFGModel.from_json(path.read_text())
        if m.n_train == n:
            return m
    m = fit_xfg(sh, log=log)
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(m.to_json())
        except OSError:
            pass
    return m


# ============================================================================== features

H, RHO = 180.0, 0.8                      # shot quality / diet memory
H_SK, RHO_SK = 365.0, 0.9                # shooting over expected ("skill"): long memory
H_OPP, RHO_OPP = 120.0, 0.7              # opponent shot quality / luck allowed
# luck inside the EB rates: the EB state's own decay and kappa (state params of 2019-20 ..
# 2026-27: pts H 90 rho 1 kappa 65, fg3m H 120 rho 1 (0.85 in some years) kappa 25)
EB_LUCK = {"pts": (90.0, 1.0, 65.0), "fg3m": (120.0, 1.0, 25.0)}
K_X, K_SH, K_FT, K_AST = 40.0, 60.0, 40.0, 25.0
K_SK2, K_SK3 = 150.0, 250.0
K_OPP = 600.0

SUMS = ("_smin", "n2", "m2", "x2", "n3", "m3", "x3", "rim", "sht", "mid", "c3", "atb",
        "fta", "ftm", "a2", "a3")
QUAL_COLS = ("f4_xfg2", "f4_xfg3", "f4_xpps")
DIET_COLS = ("f4_rim", "f4_sht", "f4_mid", "f4_c3", "f4_atb", "f4_3par", "f4_ftr", "f4_ft",
             "f4_ast2", "f4_ast3")
SK_COLS = ("f4_sk2", "f4_sk3")
OPP_COLS = ("f4_o_xfg", "f4_o_lk", "f4_o_lk3")
LUCK_SUMS = ("f4_lpts_s", "f4_l3_s")                 # raw EB-decay luck sums (per player-day)
TEAM_COLS = QUAL_COLS + DIET_COLS + SK_COLS + OPP_COLS + LUCK_SUMS + ("f4_ln_fga",)
ROW_COLS = ("f4_lk_pts", "f4_lk_fg3m")
NEUTRAL = {c: 0.0 for c in QUAL_COLS + SK_COLS + OPP_COLS + ROW_COLS + ("f4_ftr", "f4_ft")}

PER_STAT = {
    "pts": QUAL_COLS + ("f4_rim", "f4_mid", "f4_c3", "f4_atb", "f4_3par", "f4_ftr", "f4_ft",
                        "f4_ast2", "f4_ast3") + SK_COLS + ("f4_o_xfg", "f4_o_lk", "f4_lk_pts",
                                                            "f4_ln_fga"),
    "fg3m": ("f4_xfg3", "f4_c3", "f4_atb", "f4_3par", "f4_ft", "f4_ast3", "f4_sk3", "f4_o_lk3",
             "f4_lk_fg3m", "f4_ln_fga"),
}


def game_sums(shots: pd.DataFrame, xfg: np.ndarray) -> pd.DataFrame:
    """Per (game_id, player_id) sums of the shot columns (x = xFG per FGA)."""
    k = shots["kind"].to_numpy(int)
    made = shots["made"].to_numpy(float)
    d = np.nan_to_num(shots["dist"].to_numpy(float))
    ay = np.nan_to_num(shots["ay"].to_numpy(float), nan=99.0)
    is2, is3, ft = k == 2, k == 3, k == 1
    x = np.zeros(len(shots))
    fg = k > 1
    x[fg] = xfg
    v = pd.DataFrame({
        "game_id": shots["game_id"].to_numpy(), "player_id": shots["player_id"].to_numpy(),
        "n2": is2 * 1.0, "m2": is2 * made, "x2": is2 * x, "n3": is3 * 1.0, "m3": is3 * made,
        "x3": is3 * x, "rim": is2 & (d < 4), "sht": is2 & (d >= 4) & (d < 14),
        "mid": is2 & (d >= 14), "c3": is3 & (ay <= 9.25), "atb": is3 & (ay > 9.25),
        "fta": ft * 1.0, "ftm": ft * made,
        "a2": is2 * shots["assisted"].to_numpy(float), "a3": is3 * shots["assisted"].to_numpy(float)})
    return v.groupby(["game_id", "player_id"], sort=False).sum().astype(float).reset_index()


def player_history(conn: sqlite3.Connection, shots: pd.DataFrame, xfg: np.ndarray, *,
                   pos=None) -> pd.DataFrame:
    """One row per player-game (min > 0) of the seasons in `shots` whose game has PBP shots:
    minutes (_smin) + the shot sums, day, sidx, grp."""
    gs = game_sums(shots, xfg)
    first, last = shots["season"].min(), shots["season"].max()
    pg = pd.read_sql_query(
        "SELECT game_id, player_id, season, game_date, min FROM player_games WHERE min > 0 "
        "AND season >= ? AND season <= ?", conn, params=(first, last))
    pg = pg[pg["game_id"].isin(set(shots["game_id"].unique()))]
    h = pg.merge(gs, on=["game_id", "player_id"], how="left")
    for c in SUMS[1:]:
        h[c] = h[c].astype(float).fillna(0.0)
    h["_smin"] = h["min"].astype(float)
    h["day"] = ST._to_day(h["game_date"]) if len(h) else np.zeros(0, np.int64)
    h["sidx"] = h["season"].str[:4].astype(int) if len(h) else np.zeros(0, int)
    pos = pos if pos is not None else ST.load_positions(conn)
    h["grp"] = (ST._attach_grp(h["player_id"].to_numpy(), h["sidx"].to_numpy(), pos)
                if len(h) else np.zeros(0, int))
    return h


def team_history(shots: pd.DataFrame, xfg: np.ndarray) -> pd.DataFrame:
    """Per (game, defending team): FGA, 3PA, makes and xFG allowed."""
    fg = (shots["kind"] > 1).to_numpy()
    s = shots[fg]
    d = pd.DataFrame({"team": s["opp"].to_numpy(), "game_id": s["game_id"].to_numpy(),
                      "day": s["day"].to_numpy(), "sidx": s["sidx"].to_numpy(),
                      "n": 1.0, "m": s["made"].to_numpy(float), "x": xfg,
                      "n3": (s["kind"] == 3).to_numpy(float)})
    d["m3"] = d["m"] * d["n3"]
    d["x3"] = d["x"] * d["n3"]
    return d.groupby(["team", "game_id", "day", "sidx"], sort=False)[
        ["n", "m", "x", "n3", "m3", "x3"]].sum().reset_index()


@dataclass
class ShotHistory:
    """Everything `player_features` needs, built once (frame build or the live path)."""
    h: pd.DataFrame          # player-game sums
    t: pd.DataFrame          # defending team-game sums
    xfg: XFGModel


def load_history(conn: sqlite3.Connection, xfg: XFGModel, *, first: str | None = None,
                 last: str | None = None, shots: pd.DataFrame | None = None, pos=None,
                 log: Callable[[str], None] | None = None) -> ShotHistory:
    sh = load_shots(conn, first, last, shots=shots, log=log)
    fga = (sh["kind"] > 1).to_numpy() & np.isfinite(sh["dist"].to_numpy(float))
    sh = sh[(sh["kind"] == 1).to_numpy() | fga].reset_index(drop=True)
    x = xfg.predict(sh[(sh["kind"] > 1).to_numpy()])
    return ShotHistory(h=player_history(conn, sh, x, pos=pos), t=team_history(sh, x), xfg=xfg)


def _lr(num, den, prior, K):
    """log of the shrunk ratio (num + K prior) / (den + K) over prior."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (num + K * prior) / (den + K)
        return np.log(np.maximum(r, 1e-9) / np.maximum(prior, 1e-9))


def player_features(sh: ShotHistory, q: pd.DataFrame) -> pd.DataFrame:
    """F4 per query (player_id, day, sidx, grp[, opp]) strictly before each day, in query
    order: TEAM_COLS (the luck sums are raw; `row_features` turns them into EB shares)."""
    nq = len(q)
    out = pd.DataFrame(index=range(nq), columns=list(TEAM_COLS), dtype=float)
    if nq == 0:
        return out
    h = sh.h
    names = list(SUMS)
    V = h[names].to_numpy(float)
    qk, qd, qs = q["player_id"].to_numpy(), q["day"].to_numpy(), q["sidx"].to_numpy()
    hk, hd, hs = h["player_id"].to_numpy(), h["day"].to_numpy(), h["sidx"].to_numpy()
    S = ew_asof(hk, hd, hs, V, qk, qd, qs, half_life=H, rho=RHO)
    ix = {n: i for i, n in enumerate(names)}
    # league and position-group sums (same weights)
    hg = pd.DataFrame(V, columns=names)
    hg["grp"], hg["day"], hg["sidx"] = h["grp"].to_numpy(), hd, hs
    by_g = hg.groupby(["grp", "day", "sidx"], sort=False)[names].sum().reset_index()
    by_all = hg.groupby(["day", "sidx"], sort=False)[names].sum().reset_index()
    z = np.zeros(nq, int)
    L = ew_asof(np.zeros(len(by_all), int), by_all["day"].to_numpy(), by_all["sidx"].to_numpy(),
                by_all[names].to_numpy(float), z, qd, qs, half_life=H, rho=RHO)
    qg = np.asarray(q["grp"].to_numpy(), int)
    G = ew_asof(by_g["grp"].to_numpy(), by_g["day"].to_numpy(), by_g["sidx"].to_numpy(),
                by_g[names].to_numpy(float), qg, qd, qs, half_life=H, rho=RHO)
    known = (qg < 3) & (G[:, ix["n2"]] + G[:, ix["n3"]] > 0)
    P = np.where(known[:, None], G, L)
    era = L[:, ix["_smin"]] > 0

    def c(A, n):
        return A[:, ix[n]]
    with np.errstate(divide="ignore", invalid="ignore"):
        lx2, lx3 = c(L, "x2") / c(L, "n2"), c(L, "x3") / c(L, "n3")
        lfga = c(L, "n2") + c(L, "n3")
        lpps = (2 * c(L, "x2") + 3 * c(L, "x3")) / lfga
        out["f4_xfg2"] = _lr(c(S, "x2"), c(S, "n2"), lx2, K_X)
        out["f4_xfg3"] = _lr(c(S, "x3"), c(S, "n3"), lx3, K_X)
        fga = c(S, "n2") + c(S, "n3")
        out["f4_xpps"] = _lr(2 * c(S, "x2") + 3 * c(S, "x3"), fga, lpps, K_X)
        pfga = c(P, "n2") + c(P, "n3")
        for n in ("rim", "sht", "mid", "c3", "atb"):
            p = c(P, n) / pfga
            out[f"f4_{n}"] = (c(S, n) + K_SH * p) / (fga + K_SH)
        p3 = c(P, "n3") / pfga
        out["f4_3par"] = (c(S, "n3") + K_SH * p3) / (fga + K_SH)
        out["f4_ftr"] = _lr(c(S, "fta"), fga, c(P, "fta") / pfga, K_SH)
        out["f4_ft"] = _lr(c(S, "ftm"), c(S, "fta"), c(L, "ftm") / c(L, "fta"), K_FT)
        out["f4_ast2"] = (c(S, "a2") + K_AST * c(P, "a2") / c(P, "m2")) / (c(S, "m2") + K_AST)
        out["f4_ast3"] = (c(S, "a3") + K_AST * c(P, "a3") / c(P, "m3")) / (c(S, "m3") + K_AST)
        out["f4_ln_fga"] = np.log1p(np.maximum(fga, 0.0))
    # shooting over expected, long memory, net of the league's own drift (the xFG model's
    # level vs this season's make rate)
    sk = ["n2", "m2", "x2", "n3", "m3", "x3"]
    Vs = h[sk].to_numpy(float)
    Ss = ew_asof(hk, hd, hs, Vs, qk, qd, qs, half_life=H_SK, rho=RHO_SK)
    da = hg.groupby(["day", "sidx"], sort=False)[sk].sum().reset_index()
    Ls = ew_asof(np.zeros(len(da), int), da["day"].to_numpy(), da["sidx"].to_numpy(),
                 da[sk].to_numpy(float), z, qd, qs, half_life=H_SK, rho=RHO_SK)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, (n, m, x, K) in enumerate((("n2", "m2", "x2", K_SK2), ("n3", "m3", "x3", K_SK3))):
            i0 = sk.index(n)
            drift = (Ls[:, i0 + 1] - Ls[:, i0 + 2]) / Ls[:, i0]
            out[SK_COLS[j]] = (Ss[:, i0 + 1] - Ss[:, i0 + 2] - drift * Ss[:, i0]) / (Ss[:, i0] + K)
    # luck inside the EB rates: made - expected (net of the league drift at the same decay)
    for col, stat in (("f4_lpts_s", "pts"), ("f4_l3_s", "fg3m")):
        hl, rl, _ = EB_LUCK[stat]
        El = ew_asof(hk, hd, hs, Vs, qk, qd, qs, half_life=hl, rho=rl)
        Ll = ew_asof(np.zeros(len(da), int), da["day"].to_numpy(), da["sidx"].to_numpy(),
                     da[sk].to_numpy(float), z, qd, qs, half_life=hl, rho=rl)
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = (Ll[:, 1] - Ll[:, 2]) / Ll[:, 0]
            d3 = (Ll[:, 4] - Ll[:, 5]) / Ll[:, 3]
        l3 = El[:, 4] - El[:, 5] - d3 * El[:, 3]
        if stat == "pts":
            out[col] = 2 * (El[:, 1] - El[:, 2] - d2 * El[:, 0]) + 3 * l3
        else:
            out[col] = l3
    out.loc[~era, :] = np.nan
    # opponent (defending team) shot quality and luck allowed
    if "opp" in q:
        t = sh.t
        tv = t[["n", "m", "x", "n3", "m3", "x3"]].to_numpy(float)
        To = ew_asof(t["team"].to_numpy(), t["day"].to_numpy(), t["sidx"].to_numpy(), tv,
                     q["opp"].to_numpy(), qd, qs, half_life=H_OPP, rho=RHO_OPP)
        ta = t.groupby(["day", "sidx"], sort=False)[["n", "m", "x", "n3", "m3", "x3"]].sum() \
            .reset_index()
        Lt = ew_asof(np.zeros(len(ta), int), ta["day"].to_numpy(), ta["sidx"].to_numpy(),
                     ta[["n", "m", "x", "n3", "m3", "x3"]].to_numpy(float), z, qd, qs,
                     half_life=H_OPP, rho=RHO_OPP)
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = Lt[:, 2] / Lt[:, 0]
            out["f4_o_xfg"] = _lr(To[:, 2], To[:, 0], lx, K_OPP)
            dr = (Lt[:, 1] - Lt[:, 2]) / Lt[:, 0]
            dr3 = (Lt[:, 4] - Lt[:, 5]) / Lt[:, 3]
            out["f4_o_lk"] = (To[:, 1] - To[:, 2] - dr * To[:, 0]) / (To[:, 0] + K_OPP)
            out["f4_o_lk3"] = (To[:, 4] - To[:, 5] - dr3 * To[:, 3]) / (To[:, 3] + K_OPP / 3)
        tera = Lt[:, 0] > 0
        for cc in OPP_COLS:
            out.loc[~tera, cc] = np.nan
    return out


def build(conn: sqlite3.Connection, rows: pd.DataFrame, *, sh: ShotHistory | None = None,
          xfg: XFGModel | None = None, season: str | None = None, pos=None,
          log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """F4 for frame rows (game_id, player_id, opp, day, sidx, grp): game_id, player_id +
    TEAM_COLS. The xFG model is the one of `season` (fit on seasons < S) unless given."""
    if sh is None:
        last = ST.season_label(int(rows["sidx"].max())) if len(rows) else None
        if xfg is None:
            xfg = xfg_for_season(conn, season or last, log=log)
        sh = load_history(conn, xfg, last=last, pos=pos, log=log)
    q = rows[["game_id", "player_id", "opp", "day", "sidx", "grp"]].drop_duplicates(
        ["game_id", "player_id"]).reset_index(drop=True)
    f = player_features(sh, q)
    f.insert(0, "game_id", q["game_id"].to_numpy())
    f.insert(1, "player_id", q["player_id"].to_numpy())
    return f


def row_features(rows: pd.DataFrame) -> pd.DataFrame:
    """f4_lk_<s> = the luck sum at the EB decay / (n_<s> + kappa_s) / r_<s>: the share of the
    EB per-minute rate that is made-minus-expected shooting."""
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for col, stat, src in (("f4_lk_pts", "pts", "f4_lpts_s"),
                               ("f4_lk_fg3m", "fg3m", "f4_l3_s")):
            kappa = EB_LUCK[stat][2]
            v = (rows[src].to_numpy(float) / (rows[f"n_{stat}"].to_numpy(float) + kappa)
                 / np.maximum(rows[f"r_{stat}"].to_numpy(float), 1e-4))
            out[col] = np.clip(v, -1.0, 1.0)
    return pd.DataFrame(out, index=rows.index)


def columns(stat: str) -> list[str]:
    return list(PER_STAT.get(stat, ()))


def monotone(stat: str) -> dict:
    """More luck inside the EB rate can only lower the multiplier."""
    return {"pts": {"f4_lk_pts": -1}, "fg3m": {"f4_lk_fg3m": -1}}.get(stat, {})
