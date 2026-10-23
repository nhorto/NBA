"""B10: the truth-shift beta by role-change bucket and anchor (measurement only, no sims).

`portfolio.truth_sims` shifts every playing player's simulated DK points by
beta * (FPPG - mu_played), beta = TRUTH_BETA = 0.18. This script re-measures that slope on the
live arm's existing per-row projection outputs and asks whether it should depend on the role
change (planned minutes / last-10 minutes) and which public anchor to shrink toward.

  beta   slope through the origin of (actual - mu_played) on (anchor - mu_played), played rows
         (minutes > 0), SE two-way clustered by date and player (Cameron-Gelbach-Miller, the
         `metrics.paired_cluster2` math applied to the slope's influence function).
  anchors  fppg   DK FPPG. Historical seasons: our box-score proxy of it (this season's mean
                  once >= 5 games, else last season's; `backtest.BoxBaselines`), DK's own FPPG
                  is not stored before 2026-27. Live: DK's FPPG from the slate (grades.dk_fppg).
           l10    mean DK of the last up-to-10 games played (across the season boundary).
           sal    salary-implied: OLS dk = a + b * salary on RotoGuru played rows of seasons < S
                  (`evallite.fit_salary_line(.., 'played_only')`, the harness `sal_played`).
           mix    joint through-origin regression on the available anchors.
  buckets  r = E[min | plays] / mean minutes of the last up-to-10 appearances (strictly before
           the date): <0.8, 0.8-0.95, 0.95-1.05, 1.05-1.25, >1.25, 'nohist' (no prior game);
           plus the overlapping 'returning' row: >= 3 team games missed (this season) right
           before the game, after at least one appearance this season.
  gains  walk-forward (parameters fit on datasets of seasons < S only), applied as truth_sims
         does (only where the player plays): MAE of mu_played + delta on played rows, and a
         CRPS proxy on all projected rows (quantile function rebuilt from q10..q90, the
         positive part shifted by delta), both paired vs OFF, SE clustered by date x player.

Data (read-only):
  ROWS_DIR (default var/tmp/b10/rows): backtest player rows of the live arm pathA-v0.3+mf+p2
      (copied from the Mac scratch courtside-compute/mf/out_dk): rows_avail_<S>_... for the
      RotoGuru seasons 2019-20 / 2020-21 (salaries) and the avail harness 2023-24 / 2024-25,
      rows_live_2025-26_... (the live-path protocol, an extra hold-out).
  DB (default $COURTSIDE_DATA/courtside.db): box scores, RotoGuru salaries, and the live
      2026-27 nights (player_projections + player_projection_grades), opened read-only.

  uv run python research/truth_beta_roles.py [--out var/tmp/b10/out.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from courtside.proj import evallite as E
from courtside.proj.metrics import paired_cluster2, _cluster_var

ROOT = Path(__file__).resolve().parents[1]
ARM = "pathA-v0.3+mf+p2"
TRUTH_BETA = 0.18                       # portfolio.TRUTH_BETA (not imported: read-only copy)
BUCKETS = ["<0.8", "0.8-0.95", "0.95-1.05", "1.05-1.25", ">1.25", "nohist"]
EDGES = [0.8, 0.95, 1.05, 1.25]
DATASETS = [("rg", "2019-20", "avail"), ("rg", "2020-21", "avail"), ("hz", "2023-24", "avail"),
            ("hz", "2024-25", "avail"), ("lp", "2025-26", "live")]
LIVE_MODELS = {"2026-10-20": "pathA-v0.2", "2026-10-21": "pathA-v0.3+p2",
               "2026-10-22": "pathA-v0.3+p2"}
TAU_K = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
TAU_G = (np.arange(100) + 0.5) / 100.0


# ------------------------------------------------------------------------------ box history

class Hist:
    """Per-player played dates / cumulative minutes, team schedules (dates per team+season)."""

    def __init__(self, conn):
        pg = pd.read_sql_query("SELECT player_id, team, season, game_date, min FROM player_games "
                               "WHERE season >= '2018-19' AND season_type IN ('Regular Season', "
                               "'Playoffs', 'PlayIn', 'Play-In') ", conn)
        if pg.empty:
            pg = pd.read_sql_query("SELECT player_id, team, season, game_date, min FROM "
                                   "player_games WHERE season >= '2018-19'", conn)
        pg["min"] = pd.to_numeric(pg["min"], errors="coerce").fillna(0.0)
        pg["season"] = [E.season_label(d) for d in pg["game_date"]]
        p = pg[pg["min"] > 0].sort_values(["player_id", "game_date"])
        self.by = {}
        for pid, g in p.groupby("player_id"):
            self.by[int(pid)] = (list(g["game_date"]), np.concatenate(
                [[0.0], np.cumsum(g["min"].to_numpy(float))]))
        self.sched = {k: sorted(set(g["game_date"])) for k, g in pg.groupby(["team", "season"])}

    def l10min(self, pid, day):
        if pid is None or not np.isfinite(pid) or int(pid) not in self.by:
            return np.nan, None
        dts, cs = self.by[int(pid)]
        i = bisect.bisect_left(dts, day)
        n = min(i, 10)
        return ((cs[i] - cs[i - n]) / n if n else np.nan), (dts[i - 1] if i else None)

    def missed(self, team, day, last):
        """Team games strictly between the player's last appearance and `day` (this season)."""
        if last is None or team is None or not isinstance(team, str):
            return 0
        s = E.season_label(day)
        if E.season_label(last) != s:
            return 0
        ds = self.sched.get((team, s))
        if not ds:
            return 0
        return bisect.bisect_left(ds, day) - bisect.bisect_right(ds, last)


def add_roles(d: pd.DataFrame, h: Hist) -> pd.DataFrame:
    l10, miss = [], []
    for pid, team, day in zip(d["player_id"], d["team"], d["date"]):
        m, last = h.l10min(pid, day)
        l10.append(m)
        miss.append(h.missed(team, day, last))
    d["l10min"] = l10
    d["missed"] = miss
    d["ratio"] = d["planned"] / d["l10min"]
    b = np.full(len(d), "nohist", dtype=object)
    ok = np.isfinite(d["ratio"].to_numpy()) & (d["l10min"].to_numpy() > 0)
    b[ok] = np.array(BUCKETS[:5], dtype=object)[np.digitize(d["ratio"].to_numpy()[ok], EDGES)]
    d["bucket"] = b
    d["returning"] = d["missed"] >= 3
    return d


# ------------------------------------------------------------------------------ data

def load_rows(rows_dir: Path, kind, season, proto, h: Hist, conn) -> pd.DataFrame:
    d = pd.read_parquet(rows_dir / f"rows_{proto}_{season}_{ARM}.parquet")
    d = d[d["has_proj"] & (d["p_play"] > 0.02) & d["y"].notna()].copy()
    d["mu_p"] = d["mean"] / d["p_play"]
    d["planned"] = d["exp_min"] / d["p_play"]
    d["played"] = d["minutes"].fillna(0) > 0
    d["fppg"] = d["b_fppg"]
    d["l10"] = d["b_last10"]
    d["sal"] = d["sal_played"] if kind == "rg" else np.nan
    d["ds"] = f"{kind}:{season}"
    d["c1"] = d["date"]
    d["season"] = season
    if d["team"].isna().any():          # RotoGuru rows without a projection carry no team
        d = d[d["team"].notna()]
    return add_roles(d, h)


def load_live(conn, h: Hist) -> pd.DataFrame:
    q = """SELECT p.nba_day AS date, p.game_id, p.player_id, p.team, p.model_version, p.role AS tier,
                  p.p_play, p.created_at, p.minutes_json, p.inputs_json, g.played, g.actual_min,
                  g.actual_dk, g.proj_dk, g.crps, g.dk_fppg, g.salary
           FROM player_projections p JOIN player_projection_grades g ON g.projection_id = p.id
           WHERE p.nba_day BETWEEN '2026-10-20' AND '2026-10-22'"""
    a = pd.read_sql_query(q, conn)
    a = a[[LIVE_MODELS.get(dd) == mv for dd, mv in zip(a["date"], a["model_version"])]]
    # the latest run per player-game; DK FPPG / salary from the latest run that has them
    a = a.sort_values("created_at")
    fp = a.dropna(subset=["dk_fppg"]).groupby(["game_id", "player_id"])[["dk_fppg", "salary"]].last()
    a = a.groupby(["game_id", "player_id"]).tail(1).set_index(["game_id", "player_id"])
    a[["dk_fppg", "salary"]] = fp.reindex(a.index)
    a = a.reset_index()
    inp = [json.loads(s.replace("NaN", "null")) for s in a["inputs_json"]]
    mj = [json.loads(s.replace("NaN", "null")) for s in a["minutes_json"]]
    a["mu_p"] = [i.get("mu_played", np.nan) for i in inp]
    a["planned"] = [m.get("mean_played", np.nan) for m in mj]
    a["y"] = a["actual_dk"].astype(float)
    a["minutes"] = a["actual_min"].astype(float)
    a["played"] = a["played"].astype(bool)
    a["mean"] = a["proj_dk"].astype(float)
    a = a[a["p_play"] > 0.02].copy()
    from courtside.proj.backtest import BoxBaselines
    bx = BoxBaselines(conn, "2026-27")
    trip = [bx.get(int(p), dd) for p, dd in zip(a["player_id"], a["date"])]
    a["fppg_proxy"] = [t[0] for t in trip]
    a["l10"] = [t[2] for t in trip]
    a["fppg"] = a["dk_fppg"].where(a["dk_fppg"] > 0)
    ab, bb = E.fit_salary_line(E.load_rotoguru(conn), "2026-27", "played_only")
    a["sal"] = ab + bb * a["salary"].astype(float)
    a["ds"] = "live:2026-27"
    a["c1"] = a["game_id"]          # 3 nights: cluster by game (and player), not date
    a["season"] = "2026-27"
    for c in ("q10", "q25", "q50", "q75", "q90"):
        a[c] = np.nan
    return add_roles(a, h)


# ------------------------------------------------------------------------------ estimation

def _twoway_V(psi: np.ndarray, c1, c2) -> np.ndarray:
    """Two-way cluster covariance of sum_i psi_i (psi: n x k), V1 + V2 - V12 (CGM)."""
    both = pd.Series(np.asarray(c1)).astype(str) + "|" + pd.Series(np.asarray(c2)).astype(str)

    def cv(lab):
        s = pd.DataFrame(psi).groupby(np.asarray(lab)).sum().to_numpy()
        g = len(s)
        return g / (g - 1) * s.T @ s if g > 1 else np.full((psi.shape[1],) * 2, np.nan)
    V = cv(c1) + cv(c2) - cv(both.to_numpy())
    if np.any(np.diag(V) <= 0):
        V = np.maximum(cv(c1), cv(c2))
    return V


def slope(d: pd.DataFrame, cols: list[str]) -> dict:
    """Through-origin OLS of (y - mu_p) on (anchor - mu_p) for `cols`; two-way clustered SE."""
    s = d[d["played"]]
    X = np.column_stack([s[c].to_numpy(float) - s["mu_p"].to_numpy(float) for c in cols])
    yv = s["y"].to_numpy(float) - s["mu_p"].to_numpy(float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    if len(cols) == 1:
        ok &= s[cols[0]].to_numpy(float) > 0
    X, yv = X[ok], yv[ok]
    n = len(yv)
    if n < 15:
        return {"b": [np.nan] * len(cols), "se": [np.nan] * len(cols), "n": n}
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ yv)
    e = yv - X @ b
    inv = np.linalg.inv(XtX)
    psi = (X * e[:, None]) @ inv
    dates, pids = s["c1"].to_numpy()[ok], s["player_id"].to_numpy()[ok]
    if len(cols) == 1:      # identical to paired_cluster2 on the slope's influence d_i
        pc = paired_cluster2(b[0] + n * psi[:, 0], dates, pids)
        se = [pc["se"]]
    else:
        V = _twoway_V(psi, dates, pids)
        se = list(np.sqrt(np.diag(V)))
    return {"b": [float(x) for x in b], "se": [float(x) for x in se], "n": int(n),
            "gap_sd": float(np.sqrt((X[:, 0] ** 2).mean()))}


# ------------------------------------------------------------------------------ designs

def delta(d: pd.DataFrame, design: dict) -> np.ndarray:
    """Per-row shift of the played sims under a design {anchor cols, beta per bucket}."""
    out = np.zeros(len(d))
    cols, betas = design["cols"], design["beta"]          # betas: bucket -> list per col
    for k, bl in betas.items():
        m = (d["bucket"] == k).to_numpy() if k != "*" else np.ones(len(d), bool)
        if design.get("returning_sep") and k != "returning":
            m = m & ~d["returning"].to_numpy()
        if k == "returning":
            m = d["returning"].to_numpy()
        for c, b in zip(cols, bl):
            g = d[c].to_numpy(float) - d["mu_p"].to_numpy(float)
            ok = m & np.isfinite(g) & (d[c].to_numpy(float) > 0)
            if design.get("sign") == "below":
                ok &= g < 0
            out[ok] += b * g[ok]
    return out


def fit_design(train: pd.DataFrame, cols: list[str], by_bucket: bool, returning_sep=False,
               min_n=150) -> dict:
    flat = slope(train, cols)["b"]
    flat = [0.0 if not np.isfinite(x) else x for x in flat]
    if not by_bucket:
        return {"cols": cols, "beta": {"*": flat}}
    betas = {}
    for k in BUCKETS:
        sub = train[(train["bucket"] == k) & (~train["returning"] if returning_sep else True)]
        r = slope(sub, cols)
        betas[k] = r["b"] if r["n"] >= min_n and np.all(np.isfinite(r["b"])) else flat
    if returning_sep:
        r = slope(train[train["returning"]], cols)
        betas["returning"] = r["b"] if r["n"] >= min_n else flat
    return {"cols": cols, "beta": betas, "returning_sep": returning_sep}


def crps_proxy(d: pd.DataFrame, dl: np.ndarray) -> np.ndarray:
    """CRPS of the quantile function rebuilt from q10..q90 (linear tails), the positive part
    shifted by dl: 2 * mean over 100 levels of the pinball loss."""
    Q = d[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float)
    lo = np.maximum(0.0, Q[:, 0] - (Q[:, 1] - Q[:, 0]) * (0.10 / 0.15))
    hi = Q[:, 4] + (Q[:, 4] - Q[:, 3]) * (0.10 / 0.15) * 1.5
    K = np.column_stack([lo, Q, hi])
    G = np.empty((len(d), len(TAU_G)))
    for i in range(len(TAU_K) - 1):
        a, b = TAU_K[i], TAU_K[i + 1]
        m = (TAU_G >= a) & (TAU_G < b)
        w = (TAU_G[m] - a) / (b - a)
        G[:, m] = K[:, [i]] + (K[:, [i + 1]] - K[:, [i]]) * w[None, :]
    G = np.where(G > 1e-9, G + dl[:, None], G)
    y = d["y"].to_numpy(float)[:, None]
    loss = ((y < G).astype(float) - TAU_G[None, :]) * (G - y)
    return 2.0 * loss.mean(1)


def evaluate(d: pd.DataFrame, designs: dict, has_q=True) -> dict:
    """Paired (design - OFF) MAE on played rows and CRPS proxy on all rows, by slice."""
    slices = {"all": np.ones(len(d), bool)}
    for t in ("star", "role", "bench"):
        slices[t] = (d["tier"] == t).to_numpy()
    for k in BUCKETS[:5]:
        slices[k] = (d["bucket"] == k).to_numpy()
    slices["promoted(>1.05)"] = d["bucket"].isin(["1.05-1.25", ">1.25"]).to_numpy()
    slices["returning"] = d["returning"].to_numpy()
    played = d["played"].to_numpy()
    y, mu = d["y"].to_numpy(float), d["mu_p"].to_numpy(float)
    base_ae = np.abs(y - mu)
    base_c = crps_proxy(d, np.zeros(len(d))) if has_q else None
    out = {"_proxy_check": None}
    if has_q and "crps" in d:
        c = d["crps"].to_numpy(float)
        ok = np.isfinite(c)
        out["_proxy_check"] = {"sim_crps": float(c[ok].mean()), "proxy": float(base_c[ok].mean()),
                               "corr": float(np.corrcoef(c[ok], base_c[ok])[0, 1])}
    dates, pids = d["c1"].to_numpy(), d["player_id"].to_numpy()
    for name, des in designs.items():
        dl = delta(d, des)
        ae = np.abs(y - (mu + dl))
        res = {}
        for sn, m in slices.items():
            mp = m & played & np.isfinite(base_ae)
            r = {"n_played": int(mp.sum()), "base_mae": float(base_ae[mp].mean()) if mp.any() else np.nan}
            if mp.sum() > 10:
                pc = paired_cluster2(ae[mp] - base_ae[mp], dates[mp], pids[mp])
                r["d_mae"], r["se_mae"] = pc["mean"], pc["se"]
                r["mean_shift"] = float(dl[mp].mean())
            if has_q and m.sum() > 10:
                cc = crps_proxy(d[m], dl[m])
                pc = paired_cluster2(cc - base_c[m], dates[m], pids[m])
                r["n"], r["base_crps"] = int(m.sum()), float(base_c[m].mean())
                r["d_crps"], r["se_crps"] = pc["mean"], pc["se"]
                pm = played[m]
                pc = paired_cluster2((cc - base_c[m])[pm], dates[m][pm], pids[m][pm])
                r["d_crps_p"], r["se_crps_p"] = pc["mean"], pc["se"]
            res[sn] = r
        out[name] = res
    return out


# ------------------------------------------------------------------------------ main

def beta_tables(d: pd.DataFrame, anchors: list[str]) -> dict:
    t = {}
    groups = [("all", np.ones(len(d), bool))] + [(k, (d["bucket"] == k).to_numpy()) for k in BUCKETS]
    groups += [("returning", d["returning"].to_numpy())]
    for t_ in ("star", "role", "bench"):
        groups.append((t_, (d["tier"] == t_).to_numpy()))
    for gname, m in groups:
        sub = d[m]
        row = {a: slope(sub, [a]) for a in anchors}
        mix = [a for a in ("fppg", "l10", "sal") if a in anchors]
        row["mix"] = slope(sub, mix) | {"cols": mix}
        t[gname] = row
    return t


def sign_table(d: pd.DataFrame, a: str = "fppg") -> dict:
    """beta separately where the anchor is below / above mu_played, per bucket."""
    g = d[a] - d["mu_p"]
    out = {}
    for k in BUCKETS[:5] + ["returning", "all"]:
        m = d["returning"] if k == "returning" else (d["bucket"] == k if k != "all" else g == g)
        out[k] = {"below": slope(d[m & (g < 0)], [a]), "above": slope(d[m & (g > 0)], [a])}
    return out


def fit_crps(train: pd.DataFrame, col: str, by_bucket: bool, grid=np.arange(0, 0.31, 0.02),
             sign: str | None = None) -> dict:
    """beta per bucket (and 'returning') minimising the train CRPS proxy on a grid."""
    t = train[train["q50"].notna()]
    groups = {"*": np.ones(len(t), bool)} if not by_bucket else (
        {k: ((t["bucket"] == k) & ~t["returning"]).to_numpy() for k in BUCKETS}
        | {"returning": t["returning"].to_numpy()})
    g = t[col].to_numpy(float) - t["mu_p"].to_numpy(float)
    ok = np.isfinite(g) & (t[col].to_numpy(float) > 0)
    if sign == "below":
        ok &= g < 0
    betas = {}
    for k, m in groups.items():
        mm = m & ok
        if mm.sum() < 150:
            betas[k] = [0.0]
            continue
        sub = t[mm]
        best = min(grid, key=lambda b: crps_proxy(sub, b * g[mm]).mean())
        betas[k] = [float(best)]
    return {"cols": [col], "beta": betas, "returning_sep": by_bucket, "sign": sign}


def gap_table(d: pd.DataFrame) -> dict:
    g = (d["fppg"] - d["mu_p"]).abs()
    out = {}
    for lo, hi in [(0, 2), (2, 4), (4, 7), (7, 1e9)]:
        out[f"|fppg-mu| {lo}-{hi if hi < 1e9 else 'inf'}"] = slope(d[(g >= lo) & (g < hi)], ["fppg"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROOT / "var/tmp/b10/rows"))
    ap.add_argument("--db", default=str(Path(os.environ.get("COURTSIDE_DATA", ROOT / "var"))
                                        / "courtside.db"))
    ap.add_argument("--out", default=str(ROOT / "var/tmp/b10/out.json"))
    a = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    h = Hist(conn)
    data = {}
    for kind, s, proto in DATASETS:
        data[f"{kind}:{s}"] = load_rows(Path(a.rows), kind, s, proto, h, conn)
        print(f"loaded {kind}:{s} {len(data[f'{kind}:{s}'])} rows", file=sys.stderr)
    data["live:2026-27"] = load_live(conn, h)
    print(f"loaded live {len(data['live:2026-27'])} rows", file=sys.stderr)

    res = {"beta": {}, "gap": {}, "buckets": {}, "eval": {}, "designs": {}}
    for k, d in data.items():
        anchors = ["fppg", "l10"] + (["sal"] if d["sal"].notna().any() else [])
        res["beta"][k] = beta_tables(d, anchors)
        res["gap"][k] = gap_table(d)
        res.setdefault("sign", {})[k] = sign_table(d)
        res["buckets"][k] = {b: int(((d["bucket"] == b) & d["played"]).sum()) for b in BUCKETS} | {
            "returning": int((d["returning"] & d["played"]).sum()), "played": int(d["played"].sum())}
    pooled = {"rg": pd.concat([data["rg:2019-20"], data["rg:2020-21"]], ignore_index=True),
              "hz": pd.concat([data["hz:2023-24"], data["hz:2024-25"]], ignore_index=True)}
    for k, d in pooled.items():
        anchors = ["fppg", "l10"] + (["sal"] if k == "rg" else [])
        res["beta"][f"{k}:pooled"] = beta_tables(d, anchors)
        res["gap"][f"{k}:pooled"] = gap_table(d)
        res["sign"][f"{k}:pooled"] = sign_table(d)

    # walk-forward gains: fit on datasets of seasons < S
    order = list(data)
    for i, k in enumerate(order):
        if i == 0:
            continue
        train = pd.concat([data[j] for j in order[:i]], ignore_index=True)
        test = data[k]
        has_sal = test["sal"].notna().any() and train["sal"].notna().any()
        trs = train[train["sal"].notna()] if has_sal else train
        designs = {
            "flat0.18_fppg": {"cols": ["fppg"], "beta": {"*": [TRUTH_BETA]}},
            "flat_fppg": fit_design(train, ["fppg"], False),
            "bucket_fppg": fit_design(train, ["fppg"], True, returning_sep=True),
            "flat_l10": fit_design(train, ["l10"], False),
            "bucket_l10": fit_design(train, ["l10"], True, returning_sep=True),
            "flat_mix2": fit_design(train, ["fppg", "l10"], False),
            "bucket_mix2": fit_design(train, ["fppg", "l10"], True, returning_sep=True),
            "flat_fppg_crps": fit_crps(train, "fppg", False),
            "bucket_fppg_crps": fit_crps(train, "fppg", True),
            "bucket_l10_crps": fit_crps(train, "l10", True),
            "below_fppg_crps": fit_crps(train, "fppg", False, sign="below"),
        }
        if has_sal:
            designs["flat_sal"] = fit_design(trs, ["sal"], False)
            designs["bucket_mix3"] = fit_design(trs, ["fppg", "l10", "sal"], True, returning_sep=True)
            designs["flat_mix3"] = fit_design(trs, ["fppg", "l10", "sal"], False)
        res["designs"][k] = {n: dd["beta"] for n, dd in designs.items()}
        res["eval"][k] = evaluate(test, designs, has_q=not k.startswith("live"))
        print(f"evaluated {k}", file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(a.out)


if __name__ == "__main__":
    main()
