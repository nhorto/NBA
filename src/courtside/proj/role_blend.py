"""Role-aware blend of our projection with the last-10 DK average (rb; IMPROVEMENTS.md P1 + D1,
docs/research/truth-beta-roles.md for the measurement, docs/research/role-blend.md for the test).

    mu* = mu_played + beta(r) * (L10 - mu_played),   r = planned minutes / last-10 minutes

* **mu_played**: the sims' mean DK points over the sims where the player plays; **planned**: the
  sims' mean minutes where he plays (E[min | plays]); **L10**: mean DK points of his last up-to-10
  games played (seasons S-1 and S, the `backtest.BoxBaselines` convention); **last-10 minutes**:
  mean minutes of his last up-to-10 appearances (any season loaded).  All strictly before the day.
* **beta(r)** by ratio bucket (<0.8, 0.8-0.95, 0.95-1.05, 1.05-1.25, >1.25), fit per season S by
  minimising a CRPS proxy on the harness rows of seasons < S (walk-forward;
  `research/role_blend_eval.py fit` -> `role_blend_params.json`). **Returning** players (>= 3
  team games missed right before the game, after an appearance this season) and players without
  history get beta = 0.
* Applied in `pipeline.run_games` right after the sims are drawn: every played sim of the player
  moves by delta = mu* - mu_played (non-played sims stay 0), so the stored projections, the sims
  npz, the display, cash / Top N, builder candidates and the contest builder all carry mu*.
  `portfolio.truth_sims` then skips its own flat `TRUTH_BETA` shift (TRUTH_TAU stays).
* Switch `COURTSIDE_ROLE_BLEND` (default OFF; `pipeline.set_role_blend`): model version
  `<arm>+rb`, same bundle, same seeds. OFF: nothing of it runs (bit-identical).
* Weekly refit (`refit_weekly`, not scheduled): the season's graded live rows are added to the
  stored training loss curves of that season and beta re-minimised per bucket.

Public API
  ROLE_BLEND, set_role_blend(on), betas(season) -> {bucket: beta}
  RoleHistory(conn, season).lookup(player_ids, teams, day) -> (l10, l10min, missed)
  bucket_of(planned, l10min) -> array of bucket labels; delta(...) -> per-player shift
  apply(sims, specs, day) -> {shifted, mean_abs_shift}   (in place on GameSims.fp / .info)
  crps_proxy(q, y, shift) / loss_curves(frame) / fit_betas(curves) / refit_weekly(conn, season)
  run_blended(conn, run_id) -> bool | None
"""

from __future__ import annotations

import bisect
import json
import os
import sqlite3
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROLE_BLEND_DEFAULT = True              # shipped 2026-10-23 (owner OK; env still overrides)
ROLE_BLEND_ENV = "COURTSIDE_ROLE_BLEND"
ROLE_BLEND_TAG = "+rb"
RB_PATH = Path(__file__).with_name("role_blend_params.json")

BUCKETS = ("<0.8", "0.8-0.95", "0.95-1.05", "1.05-1.25", ">1.25")
EDGES = (0.8, 0.95, 1.05, 1.25)
# the B10 starting values (docs/research/truth-beta-roles.md): used only when the params file
# has no season <= S
DEFAULT_BETA = {"<0.8": 0.0, "0.8-0.95": 0.16, "0.95-1.05": 0.12, "1.05-1.25": 0.09,
                ">1.25": 0.07}
GRID = tuple(round(0.02 * k, 2) for k in range(16))      # 0 .. 0.30
MIN_ROWS = 150                         # rows a bucket needs before its beta leaves 0 / the prior
WINDOW = 10
RETURNING_MISSED = 3
SEASON_TYPES = ("Regular Season", "Playoffs", "PlayIn", "Play-In")
QCOLS = ("q10", "q25", "q50", "q75", "q90")


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


ROLE_BLEND = _flag(ROLE_BLEND_ENV, ROLE_BLEND_DEFAULT)


def set_role_blend(on: bool) -> None:
    """Flip the role-blend switch at run time (`pipeline.set_role_blend` also updates the model
    version)."""
    global ROLE_BLEND
    ROLE_BLEND = bool(on)


# ============================================================================== parameters

def _read(path: Path = RB_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=None)
def betas(season: str) -> dict:
    """{bucket: beta} for season S (fit on seasons < S); a season missing from the file uses the
    latest earlier one (still walk-forward), none at all -> DEFAULT_BETA."""
    seasons = _read().get("seasons", {})
    if season in seasons:
        return {k: float(seasons[season]["beta"].get(k, 0.0)) for k in BUCKETS}
    prev = [s for s in seasons if s < season]
    if prev:
        return {k: float(seasons[max(prev)]["beta"].get(k, 0.0)) for k in BUCKETS}
    return dict(DEFAULT_BETA)


def run_blended(conn: sqlite3.Connection, run_id: str | None) -> bool | None:
    """Whether a stored projection run's sims carry the blend (model version has `+rb`); None
    when the run is unknown (callers then fall back to the process switch)."""
    if not run_id:
        return None
    try:
        r = conn.execute("SELECT model_version FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
    except sqlite3.Error:
        return None
    if r is None or r[0] is None:
        return None
    return ROLE_BLEND_TAG in str(r[0])


# ============================================================================== history

def _season_label(d: str) -> str:
    from .evallite import season_label
    return season_label(d)


def _season_minus(season: str, k: int) -> str:
    y = int(season[:4]) - k
    return f"{y}-{(y + 1) % 100:02d}"


class RoleHistory:
    """Box history for the blend inputs of season S (box rows with min > 0, seasons S-3..S,
    regular season / play-in / playoffs; every lookup strictly before the day)."""

    def __init__(self, conn: sqlite3.Connection, season: str):
        from .copula import DK_STATS, DK_W
        q = ("SELECT player_id, team, game_date, min, pts, fg3m, reb, ast, stl, blk, tov "
             "FROM player_games WHERE season >= ? AND season <= ? AND season_type IN ({})"
             .format(",".join("?" * len(SEASON_TYPES))))
        pg = pd.read_sql_query(q, conn, params=(_season_minus(season, 3), season, *SEASON_TYPES))
        pg["min"] = pd.to_numeric(pg["min"], errors="coerce").fillna(0.0)
        cats = sum((pg[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
        pg["dk"] = (sum(DK_W[s] * pg[s].astype(float) for s in DK_STATS)
                    + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
        labels = {d: _season_label(d) for d in pg["game_date"].unique()}
        pg["slab"] = pg["game_date"].map(labels)
        self.season = season
        self.dk_from = _season_minus(season, 1)
        p = pg[pg["min"] > 0].sort_values(["player_id", "game_date"], kind="mergesort")
        self.by: dict[int, tuple] = {}
        for pid, g in p.groupby("player_id", sort=False):
            dts = list(g["game_date"])
            cmin = np.concatenate([[0.0], np.cumsum(g["min"].to_numpy(float))])
            k = (g["slab"] >= self.dk_from).to_numpy()
            ddk = list(g["game_date"][k])
            cdk = np.concatenate([[0.0], np.cumsum(g["dk"].to_numpy(float)[k])])
            self.by[int(pid)] = (dts, cmin, ddk, cdk)
        self.sched = {k: sorted(set(g["game_date"])) for k, g in pg.groupby(["team", "slab"])}

    def one(self, pid, team, day: str) -> tuple[float, float, int]:
        """(L10 DK, last-10 minutes, team games missed right before `day` this season)."""
        nan = float("nan")
        if pid is None or (isinstance(pid, float) and not np.isfinite(pid)) or int(pid) not in self.by:
            return nan, nan, 0
        dts, cmin, ddk, cdk = self.by[int(pid)]
        i = bisect.bisect_left(dts, day)
        n = min(i, WINDOW)
        l10min = (cmin[i] - cmin[i - n]) / n if n else nan
        j = bisect.bisect_left(ddk, day)
        m = min(j, WINDOW)
        l10 = (cdk[j] - cdk[j - m]) / m if m else nan
        missed = 0
        if i and isinstance(team, str):
            last = dts[i - 1]
            s = _season_label(day)
            if _season_label(last) == s:
                ds = self.sched.get((team, s))
                if ds:
                    missed = bisect.bisect_left(ds, day) - bisect.bisect_right(ds, last)
        return float(l10), float(l10min), int(missed)

    def lookup(self, player_ids: Sequence, teams: Sequence, day) -> tuple[np.ndarray, ...]:
        d = day.isoformat() if isinstance(day, (date, datetime)) else str(day)[:10]
        out = [self.one(p, t, d) for p, t in zip(player_ids, teams)]
        if not out:
            return np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
        a = np.array(out, float)
        return a[:, 0], a[:, 1], a[:, 2].astype(np.int64)


# ============================================================================== the shift

def bucket_of(planned: np.ndarray, l10min: np.ndarray) -> np.ndarray:
    """Ratio bucket label per player ('nohist' without last-10 minutes)."""
    planned = np.asarray(planned, float)
    l10min = np.asarray(l10min, float)
    out = np.full(len(planned), "nohist", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = planned / l10min
    ok = np.isfinite(r) & (l10min > 0)
    out[ok] = np.array(BUCKETS, dtype=object)[np.digitize(r[ok], EDGES)]
    return out


def delta(mu_p: np.ndarray, planned: np.ndarray, l10: np.ndarray, l10min: np.ndarray,
          missed: np.ndarray, beta: dict) -> np.ndarray:
    """beta(r) * (L10 - mu_played); 0 for returning / no history / no L10 / mu_played unknown."""
    mu_p = np.asarray(mu_p, float)
    l10 = np.asarray(l10, float)
    b = bucket_of(planned, l10min)
    bv = np.array([float(beta.get(k, 0.0)) for k in b], float)
    ok = (np.isfinite(l10) & (l10 > 0) & np.isfinite(mu_p) & (b != "nohist")
          & (np.asarray(missed) < RETURNING_MISSED))
    return np.where(ok, bv * (l10 - np.where(ok, mu_p, 0.0)), 0.0)


def attach(players: pd.DataFrame, hist: RoleHistory, team: str, day) -> pd.DataFrame:
    """The blend inputs as `rb_*` columns of a TeamSpec players frame (ON only)."""
    l10, l10min, missed = hist.lookup(players["player_id"].to_numpy(), [team] * len(players), day)
    players["rb_l10"] = l10
    players["rb_l10min"] = l10min
    players["rb_missed"] = missed
    return players


def apply(sims: list, specs: list, day) -> dict:
    """Shift each game's played sims by delta (in place: GameSims.fp; info gets `rb_shift` and
    `mu_played` + delta). The specs' players frames carry the `rb_*` inputs (`attach`)."""
    from .pipeline import season_of
    d = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    beta = betas(season_of(d))
    n_sh, tot = 0, 0.0
    for g, gs in zip(sims, specs):
        pl = pd.concat([gs.home.players, gs.away.players], ignore_index=True)
        if "rb_l10" not in pl:
            continue
        ix = {int(p): i for i, p in enumerate(pl["player_id"].to_numpy(np.int64))}
        rows = np.array([ix.get(int(p), -1) for p in g.player_id])
        take = lambda c: np.where(rows >= 0, pl[c].to_numpy(float)[np.maximum(rows, 0)], np.nan)
        played = g.minutes > 0
        npl = played.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_p = np.where(npl > 0, np.where(played, g.fp, 0.0).astype(np.float64).sum(0) / npl,
                            np.nan)
            planned = np.where(npl > 0, np.where(played, g.minutes, 0.0).astype(np.float64)
                               .sum(0) / npl, np.nan)
        dl = delta(mu_p, planned, take("rb_l10"), take("rb_l10min"),
                   np.nan_to_num(take("rb_missed"), nan=0.0), beta)
        dl = np.where(npl > 0, dl, 0.0)
        g.fp = (g.fp + np.where(played, dl[None, :], 0.0)).astype(np.float32)
        g.info = g.info.copy()
        g.info["mu_played"] = g.info["mu_played"].to_numpy(float) + dl
        g.info["rb_shift"] = dl
        n_sh += int((dl != 0).sum())
        tot += float(np.abs(dl).sum())
    return {"shifted": n_sh, "mean_abs_shift": tot / n_sh if n_sh else 0.0, "beta": beta}


# ============================================================================== fitting

TAU_K = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
TAU_G = (np.arange(100) + 0.5) / 100.0


def crps_proxy(q: np.ndarray, y: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """CRPS of the quantile function rebuilt from q10..q90 (linear tails; B10's validated proxy,
    r = 0.99 with the sim CRPS), its positive part shifted by `shift` (the played sims move,
    DNP mass stays at 0): 2 x mean over 100 levels of the pinball loss."""
    Q = np.asarray(q, float)
    lo = np.maximum(0.0, Q[:, 0] - (Q[:, 1] - Q[:, 0]) * (0.10 / 0.15))
    hi = Q[:, 4] + (Q[:, 4] - Q[:, 3]) * (0.10 / 0.15) * 1.5
    K = np.column_stack([lo, Q, hi])
    G = np.empty((len(Q), len(TAU_G)))
    for i in range(len(TAU_K) - 1):
        a, b = TAU_K[i], TAU_K[i + 1]
        m = (TAU_G >= a) & (TAU_G < b)
        w = (TAU_G[m] - a) / (b - a)
        G[:, m] = K[:, [i]] + (K[:, [i + 1]] - K[:, [i]]) * w[None, :]
    G = np.where(G > 1e-9, G + np.asarray(shift, float)[:, None], G)
    yy = np.asarray(y, float)[:, None]
    loss = ((yy < G).astype(float) - TAU_G[None, :]) * (G - yy)
    return 2.0 * loss.mean(1)


def loss_curves(fr: pd.DataFrame, grid: Sequence[float] = GRID) -> dict:
    """Per bucket: n and the summed CRPS proxy at each grid beta. `fr` columns: mu_p, planned,
    l10, l10min, missed, y, q10..q90 (projected rows; DNP rows have y = 0)."""
    fr = fr[np.isfinite(fr["q50"].to_numpy(float)) & np.isfinite(fr["y"].to_numpy(float))]
    b = bucket_of(fr["planned"].to_numpy(float), fr["l10min"].to_numpy(float))
    l10 = fr["l10"].to_numpy(float)
    mu = fr["mu_p"].to_numpy(float)
    ok = (np.isfinite(l10) & (l10 > 0) & np.isfinite(mu)
          & (fr["missed"].to_numpy(float) < RETURNING_MISSED))
    Q = fr[list(QCOLS)].to_numpy(float)
    y = fr["y"].to_numpy(float)
    out = {}
    for k in BUCKETS:
        m = ok & (b == k)
        if not m.any():
            out[k] = {"n": 0, "loss": [0.0] * len(grid)}
            continue
        gap = l10[m] - mu[m]
        out[k] = {"n": int(m.sum()),
                  "loss": [float(crps_proxy(Q[m], y[m], bb * gap).sum()) for bb in grid]}
    return out


def add_curves(a: dict, b: dict, w: float = 1.0) -> dict:
    out = {}
    for k in BUCKETS:
        x, y = a.get(k, {"n": 0, "loss": None}), b.get(k, {"n": 0, "loss": None})
        if x["loss"] is None:
            out[k] = {"n": y["n"], "loss": [w * v for v in y["loss"]]} if y["loss"] else x
        elif y["loss"] is None:
            out[k] = x
        else:
            out[k] = {"n": x["n"] + y["n"],
                      "loss": [u + w * v for u, v in zip(x["loss"], y["loss"])]}
    return out


def fit_betas(curves: dict, grid: Sequence[float] = GRID, min_rows: int = MIN_ROWS,
              prior: dict | None = None) -> dict:
    """argmin of each bucket's loss curve; buckets with < min_rows rows keep `prior` (else 0)."""
    out = {}
    for k in BUCKETS:
        c = curves.get(k)
        if not c or c["n"] < min_rows or not c["loss"]:
            out[k] = float((prior or {}).get(k, 0.0))
            continue
        out[k] = float(grid[int(np.argmin(c["loss"]))])
    return out


# ============================================================================== weekly refit

def live_rows(conn: sqlite3.Connection, season: str, until: str | None = None) -> pd.DataFrame:
    """Graded live rows of season S (latest run per player-game, nights before `until`): the
    loss_curves frame. A run made with the blend ON is un-shifted (inputs_json rb_shift) so the
    refit sees the base projection."""
    from .evallite import season_start
    lo = season_start(season)
    hi = until or "9999-12-31"
    q = """SELECT p.nba_day AS date, p.game_id, p.player_id, p.team, p.created_at, p.p_play,
                  p.minutes_json, p.inputs_json, g.actual_dk, g.played,
                  s.q10, s.q25, s.q50, s.q75, s.q90, sp.mean AS mu_p
           FROM player_projections p
           JOIN player_projection_grades g ON g.projection_id = p.id
           JOIN player_projection_stats s ON s.projection_id = p.id AND s.stat = 'dk'
                AND s.cond = 'all'
           LEFT JOIN player_projection_stats sp ON sp.projection_id = p.id AND sp.stat = 'dk'
                AND sp.cond = 'played'
           WHERE p.nba_day >= ? AND p.nba_day < ?"""
    a = pd.read_sql_query(q, conn, params=(lo, hi))
    if a.empty:
        return a
    a = a.sort_values("created_at").groupby(["game_id", "player_id"]).tail(1).copy()
    inp = [json.loads((s or "{}").replace("NaN", "null")) for s in a["inputs_json"]]
    mj = [json.loads((s or "{}").replace("NaN", "null")) for s in a["minutes_json"]]
    sh = np.array([float(i.get("rb_shift") or 0.0) for i in inp])
    a["planned"] = [m.get("mean_played", np.nan) for m in mj]
    a["planned"] = a["planned"].astype(float)
    a["mu_p"] = a["mu_p"].astype(float) - sh
    for c in QCOLS:
        v = a[c].to_numpy(float)
        a[c] = np.where(v > 1e-9, v - sh, v)
    a["y"] = a["actual_dk"].astype(float)
    a = a[a["p_play"].astype(float) > 0.02]
    h = RoleHistory(conn, season)
    l10, l10min, missed = [], [], []
    for pid, team, d in zip(a["player_id"], a["team"], a["date"]):
        x = h.one(pid, team, d)
        l10.append(x[0]); l10min.append(x[1]); missed.append(x[2])
    a["l10"], a["l10min"], a["missed"] = l10, l10min, missed
    return a.reset_index(drop=True)


def refit_weekly(conn: sqlite3.Connection, season: str | None = None, *, path: Path = RB_PATH,
                 until: str | None = None, weight: float = 1.0, write: bool = True) -> dict:
    """The weekly refit hook (not scheduled): season S's beta re-minimised on its stored training
    curves (seasons < S) + `weight` x the season's graded live rows so far (nights < `until`).
    Writes `seasons[S]` (beta, refit info; the training curves stay) when `write`."""
    from .pipeline import season_of
    season = season or season_of(date.today())
    par = _read(path)
    seasons = par.setdefault("seasons", {})
    prev = [s for s in seasons if s <= season]
    base = seasons.get(season) or (seasons[max(prev)] if prev else {})
    train = base.get("curves", {})
    fr = live_rows(conn, season, until)
    live = loss_curves(fr) if len(fr) else {}
    comb = add_curves(train, live, weight) if live else train
    prior = base.get("beta", DEFAULT_BETA)
    new = fit_betas(comb, prior=prior) if comb else dict(prior)
    entry = dict(base)
    entry.update({"beta": new, "curves": train, "train": base.get("train", []),
                  "refit": {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "until": until, "weight": weight, "n_live": int(len(fr)),
                            "nights": int(fr["date"].nunique()) if len(fr) else 0,
                            "n_live_bucket": {k: v["n"] for k, v in live.items()},
                            "beta_before": prior}})
    seasons[season] = entry
    if write:
        Path(path).write_text(json.dumps(par, indent=1) + "\n")
        betas.cache_clear()
    return entry
