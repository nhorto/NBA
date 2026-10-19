"""Walk-forward evaluation of the rate state (docs/PLAYERS.md §7.2, T1.1): given the ACTUAL minutes
(this isolates the rate model), predict each stat and DK points for every player-game of season S
with params fit on seasons < S, against three baselines:

  (a) season-to-date per-minute average, unshrunk (DK FPPG convention: before the 5th game of
      the season it is last season's average; first-ever games fall back to the prior);
  (b) last-10-games per-minute average (any season, fewer games when there are fewer);
  (c) the prior only (position group x tier league rate, season to date).

Rates are floored at 5% of the prior for the log-likelihood (a rate of exactly 0 would score
-inf on one block), the same for every method. The log-likelihood is the NB2 log pmf of each
count given mu = rate x actual minutes, with the dispersion phi of the fitted params (same phi
for all methods), averaged per player-game.

DK points: prediction = sum(weight x rate) x minutes + E[DD/TD bonus]; the bonus expectation
treats pts/reb/ast/stl/blk as independent NB variables (the real ones are positively
correlated, so it is approximate) and the same approximation is applied to every method. The
actual is the full DK score including the real bonuses. 'DK (no bonus)' removes both.

Population: regular season, actual minutes >= 10 (DNPs do not exist in the box rows).

    python -m courtside.proj.state_eval                       # seasons 2023-24 and 2024-25
    python -m courtside.proj.state_eval --fit-default --write # fit DEFAULT_PARAMS (seasons < 2026-27)

Every method's mean is rate x actual minutes x f_stat(minutes / mpg): the fitted minutes table of
the params (same for all, so the comparison is about the rates). `eb_raw` is the EB rate without it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import state
from .state import DK_WEIGHTS, STATS, nb_logpmf, phi_for

POP_MIN = 10.0
FLOOR = 0.05
TRAIN_SEASONS = 8
EARLY_DAYS = 35
DD_CATS = ("pts", "reb", "ast", "stl", "blk")
METHODS = ("eb", "std", "l10", "prior", "eb_raw")   # eb_raw: eb without the minutes table


# ------------------------------------------------------------------------------- baselines

def baseline_rates(games: pd.DataFrame, wf: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-minute rates of the baselines for every row of `wf` (same order). `games` must hold
    the history (all games, min > 0) up to the end of the season."""
    g = games.sort_values(["player_id", "day", "game_id"]).reset_index(drop=True)
    cols = ["min", *STATS]
    x = g[cols].to_numpy(float)
    pid = g["player_id"].to_numpy()
    sidx = g["sidx"].to_numpy()
    # exclusive cumulative sums within (player, season)
    grp = g.groupby(["player_id", "sidx"], sort=False)
    c = grp[cols].cumsum().to_numpy()
    e_std = c - x
    n_std = grp.cumcount().to_numpy()
    # previous season totals
    tot = g.groupby(["player_id", "sidx"])[cols].sum().reset_index()
    tot["sidx"] += 1
    prev = g[["player_id", "sidx"]].merge(tot, on=["player_id", "sidx"], how="left")[cols].to_numpy()
    # last 10 games: exclusive cumulative sums over the player's whole history
    cp = g.groupby("player_id", sort=False)[cols].cumsum().to_numpy()
    ep = cp - x
    k = g.groupby("player_id", sort=False).cumcount().to_numpy()
    idx = np.arange(len(g))
    lag = np.where(k >= 10, idx - 10, idx)
    l10 = np.where((k >= 10)[:, None], ep - ep[lag], ep)          # rows i-10..i-1
    l10_n = l10[:, 0]
    ok_std = (n_std >= 5) & (e_std[:, 0] > 0)
    ok_prev = np.isfinite(prev[:, 0]) & (prev[:, 0] > 0)
    use_cur_small = (~ok_std) & (~ok_prev) & (e_std[:, 0] > 0)

    key = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "player_id": pid})
    pos = wf[["game_id", "player_id"]].merge(key.assign(_i=np.arange(len(key))),
                                             on=["game_id", "player_id"], how="left")["_i"].to_numpy()
    out = {m: {} for m in METHODS}
    for j, s in enumerate(STATS):
        prior = wf[f"p_{s}"].to_numpy()
        cur = e_std[pos, 1 + j] / np.maximum(e_std[pos, 0], 1e-9)
        pr = prev[pos, 1 + j] / np.maximum(prev[pos, 0], 1e-9)
        std = np.where(ok_std[pos], cur, np.where(ok_prev[pos], pr,
                       np.where(use_cur_small[pos], cur, prior)))
        last10 = np.where(l10_n[pos] > 0, l10[pos, 1 + j] / np.maximum(l10_n[pos], 1e-9), prior)
        out["eb"][s] = wf[f"r_{s}"].to_numpy()
        out["eb_raw"][s] = out["eb"][s]
        out["std"][s] = std
        out["l10"][s] = last10
        out["prior"][s] = prior
    for m in METHODS:                   # floor relative to the prior
        for s in STATS:
            out[m][s] = np.maximum(out[m][s], FLOOR * wf[f"p_{s}"].to_numpy())
    return out


# ------------------------------------------------------------------------------- DK

def dk_actual(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(DK points with DD/TD bonuses, without) of the actual stat lines."""
    base = sum(DK_WEIGHTS[s] * df[s].to_numpy(float) for s in DK_WEIGHTS)
    cats = sum((df[s].to_numpy() >= 10).astype(int) for s in DD_CATS)
    return base + 1.5 * (cats >= 2) + 3.0 * (cats >= 3), base


def expected_bonus(mu: dict[str, np.ndarray], phi: dict[str, np.ndarray]) -> np.ndarray:
    """E[1.5 * 1(>=2 cats >= 10) + 3 * 1(>=3 cats >= 10)] with independent NB categories."""
    p = []
    for s in DD_CATS:
        n = 1.0 / phi[s]
        p.append(sps.nbinom.sf(9, n, 1.0 / (1.0 + phi[s] * mu[s])))
    n_rows = len(p[0])
    dist = np.zeros((6, n_rows))
    dist[0] = 1.0
    for pi in p:
        new = dist * (1 - pi)
        new[1:] += dist[:-1] * pi
        dist = new
    return 1.5 * dist[2:].sum(0) + 3.0 * dist[3:].sum(0)


# ------------------------------------------------------------------------------- evaluation

def _cluster_se(delta: np.ndarray, game: np.ndarray) -> float:
    d = delta - delta.mean()
    s = pd.Series(d).groupby(game).sum().to_numpy()
    return float(np.sqrt(np.sum(s ** 2)) / len(delta))


def evaluate(conn: sqlite3.Connection, season: str, params: dict | None = None, *,
             train_seasons: int = TRAIN_SEASONS, relative: bool = False, log=print) -> dict:
    """Fit on the `train_seasons` seasons before `season` (unless `params` given), evaluate."""
    t0 = time.time()
    s0 = state.sidx_of(season)
    if params is None:
        params = state.fit_params(conn, [state.season_label(i) for i in range(s0 - train_seasons, s0)],
                                  relative=relative, log=lambda m: log(f"  [{time.time() - t0:5.0f}s] {m}"))
    t_fit = time.time() - t0
    first = state.season_label(s0 - params["history_seasons"])
    games = state.load_games(conn, since_season=first, until_season=season)
    t1 = time.time()
    wf = state.walk_forward(conn, season, params, games=games)
    t_wf = time.time() - t1
    base = baseline_rates(games, wf)
    keep = ((wf["season_type"] == "Regular Season") & (wf["min"] >= POP_MIN)).to_numpy()
    first_day = pd.Timestamp(wf.loc[wf["season_type"] == "Regular Season", "game_date"].min())
    early = (pd.to_datetime(wf["game_date"]) < first_day + pd.Timedelta(days=EARLY_DAYS)).to_numpy()
    minutes = wf["min"].to_numpy()
    tier = wf["tier"].to_numpy()
    act_dk, act_base = dk_actual(wf)
    phi = {s: phi_for(params, s, tier) for s in STATS}
    fmult = {s: state.minutes_multiplier(params, s, minutes, wf["mpg"].to_numpy()) for s in STATS}
    res: dict = {"season": season, "n": int(keep.sum()), "n_early": int((keep & early).sum()),
                 "seconds_fit": t_fit, "seconds_walk_forward": t_wf, "stats": {}, "dk": {}}
    preds_dk, preds_dk_nb = {}, {}
    for m in METHODS:
        mu = {s: base[m][s] * minutes * (1.0 if m == "eb_raw" else fmult[s]) for s in STATS}
        ll_tot = 0.0
        for s in STATS:
            y = wf[s].to_numpy(float)
            mae = float(np.mean(np.abs(y - mu[s])[keep]))
            ll = float(np.mean(nb_logpmf(y, mu[s], phi[s])[keep]))
            res["stats"].setdefault(s, {})[m] = {"mae": mae, "ll": ll}
            ll_tot += ll
        res["stats"].setdefault("_sum_ll", {})[m] = {"ll": ll_tot}
        dk_nb = sum(DK_WEIGHTS[s] * mu[s] for s in DK_WEIGHTS)
        bonus = expected_bonus({s: mu[s] for s in DD_CATS}, {s: phi[s] for s in DD_CATS})
        preds_dk[m] = dk_nb + bonus
        preds_dk_nb[m] = dk_nb
    gid = wf["game_id"].to_numpy()
    for label, pred, act in (("dk", preds_dk, act_dk), ("dk_nobonus", preds_dk_nb, act_base)):
        res["dk"][label] = {}
        for m in METHODS:
            err = np.abs(act - pred[m])
            e = {"mae": float(err[keep].mean()), "mae_early": float(err[keep & early].mean()),
                 "bias": float((pred[m] - act)[keep].mean())}
            if m != "eb":
                d = (np.abs(act - pred["eb"]) - err)[keep]
                e["eb_gain_pct"] = float(-d.mean() / err[keep].mean() * 100)
                e["eb_gain_se_pct"] = _cluster_se(d, gid[keep]) / err[keep].mean() * 100
                de = (np.abs(act - pred["eb"]) - err)[keep & early]
                e["eb_gain_early_pct"] = float(-de.mean() / err[keep & early].mean() * 100)
            res["dk"][label][m] = e
    res["wf"] = wf
    res["params"] = params
    return res


def format_report(res: dict) -> str:
    out = [f"== {res['season']}: n={res['n']} player-games (min>={POP_MIN:g}, regular season); "
           f"early-season slice n={res['n_early']}; fit {res['seconds_fit']:.0f}s, "
           f"walk-forward {res['seconds_walk_forward']:.1f}s"]
    out.append("stat   | MAE eb std l10 prior eb_raw                       | NB loglik/game eb std l10 prior eb_raw")
    for s in STATS:
        r = res["stats"][s]
        out.append(f"{s:6s} | " + " ".join(f"{r[m]['mae']:7.3f}" for m in METHODS) + "  | "
                   + " ".join(f"{r[m]['ll']:7.4f}" for m in METHODS))
    r = res["stats"]["_sum_ll"]
    out.append(f"{'sum':6s} | {'':37s}| " + " ".join(f"{r[m]['ll']:7.3f}" for m in METHODS))
    for label in ("dk", "dk_nobonus"):
        out.append(f"DK FP ({'with E[DD/TD]' if label == 'dk' else 'no bonus'}): MAE (bias; early-season MAE)")
        for m in METHODS:
            e = res["dk"][label][m]
            extra = ""
            if m != "eb":
                extra = (f"  eb better by {e['eb_gain_pct']:.2f}% (SE {e['eb_gain_se_pct']:.2f}), "
                         f"early {e['eb_gain_early_pct']:.2f}%")
            out.append(f"  {m:5s} {e['mae']:.3f} ({e['bias']:+.3f}; {e['mae_early']:.3f}){extra}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from .. import db

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25"])
    ap.add_argument("--fit-default", action="store_true",
                    help="fit params on the 8 seasons before --as-of-season (default 2026-27)")
    ap.add_argument("--as-of-season", default="2026-27")
    ap.add_argument("--write", action="store_true", help="with --fit-default: print JSON")
    ap.add_argument("--bump", action="store_true", help="also fit/report the usage bump")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--relative", action="store_true", help="era-relative shrinkage (tested: worse)")
    a = ap.parse_args(argv)
    conn = db.connect()
    if a.fit_default:
        s0 = state.sidx_of(a.as_of_season)
        seasons = [state.season_label(i) for i in range(s0 - TRAIN_SEASONS, s0)]
        p = state.fit_params(conn, seasons, relative=a.relative, log=print)
        if a.bump:
            p["usage_bump"] = state.fit_usage_bump(conn, p, seasons, log=print)
        print(state.params_to_json(p))
        if a.json_out:
            open(a.json_out, "w").write(state.params_to_json(p))
        return 0
    for season in a.seasons:
        res = evaluate(conn, season, relative=a.relative)
        print(format_report(res), flush=True)
        if a.bump:
            s0 = state.sidx_of(season)
            prior_seasons = [state.season_label(i) for i in range(s0 - TRAIN_SEASONS, s0)]
            bump = state.fit_usage_bump(conn, res["params"], prior_seasons)
            held = state.fit_usage_bump(conn, res["params"], [season], wf=res["wf"])
            print("usage bump fit on seasons <", season, json.dumps(
                {r: {k: round(v, 3) for k, v in d.items() if k in ("n", "games", "pts", "ast", "dk")}
                 for r, d in bump.items()}))
            print("usage bump realised in", season, json.dumps(
                {r: {k: round(v, 3) for k, v in d.items() if k in ("n", "games", "pts", "ast", "dk")}
                 for r, d in held.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
