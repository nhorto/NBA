"""Blowout minutes (`COURTSIDE_BLOWOUT=1`, `+bo`): diagnosis and evaluation.

Known fact (research/dfs_correlations*.py, 2015-26): in blowouts stars lose about -5.5 DK at
|margin| >= 26, almost all of it lost minutes. The Path A sims draw the game margin and then the
minutes at that margin (`minutes.sample_minutes`: plan + g_tier(|m|, won) - c_tier, noise, early
exits, DNP-CD draws, then the per-sim water-filling to 240 + 25 n_OT). This script checks whether
the sims' minutes / DK *conditional on the margin* match what really happens at that margin, by
|margin| bucket (0-9, 10-19, 20-25, 26+) x tier x won / lost, for the OFF arm and +mf.

  mdiag S [--params FILE] [--tag T] [--sims N]
      minutes engine alone on the avail oracle panel of S (params fit < S): `sample_minutes` at
      each team-game's ACTUAL margin and overtime (N sims) -> per row sim mean (all / given
      play), P(play), 10 / 90 % interval and randomised PIT of the box minutes, plus the team's
      minutes outside the candidate set -> OUT/mdiag_<S>_<T>.parquet and a printed table
  mtable OUT/mdiag_<S>_<T>.parquet [...]
      the bucket x tier tables of one or more mdiag files (gap = actual - sim, SE by date)
  harness S PROTOCOL
      T2.1 harness (avail) / T2.4 slice (live, live10) for the arm of the environment switches,
      4,000 sims, with extra per-player columns from the same sims: the actual team margin, the
      DK / minutes mean, 10 / 90 % quantiles and randomised PIT CONDITIONAL on the sim's team
      margin being near the actual one (+-3, widened until >= 100 sims), and lineup rows (top-8,
      30 random 8-player sets per date, one team's top-4) -> OUT/rows_<protocol>_<S>_<version>
      .parquet, OUT/lineups_<protocol>_<S>_<version>.parquet
  dk OFF.parquet ON.parquet --label L
      paired ON - OFF on identical harness rows (E[min] >= 10 in either arm): DK CRPS / MAE /
      coverage / PIT overall and by tier, the blowout-bucket calibration (conditional and
      unconditional PIT / coverage / bias by |margin| bucket x tier), lineup spread; SE by date
      -> OUT/dk_<L>.json
  summary L [L ...]
      markdown table of several dk_<L>.json (the docs/PLAYERS.md results block)
  fit S --tag T [--minfix] [--blowout] / same A B
      minutes params of S as the pipeline fits them (+ the switches) / identical apart from the
      blowout stage?

Tiers (as research/minutes_fix_eval.py): by E[min | plays] (the OFF arm's): star >= 30, starter
24-30, rotation 15-24, bench < 15. Buckets by the team's actual |final margin|.

Environment: COURTSIDE_DATA (bundles), BO_OUT (default var/tmp/bo/out). Heavy: Mac scratch
(`courtside-compute/bo`).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("BO_OUT", ROOT / "var" / "tmp" / "bo" / "out"))
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
TIER_NAMES = tuple(t[0] for t in TIERS)
BUCKETS = ((0, 10, "0-9"), (10, 20, "10-19"), (20, 26, "20-25"), (26, 999, "26+"))
BUCKET_NAMES = tuple(b[2] for b in BUCKETS)
SD_MARGIN = 13.5


def _conn():
    from courtside import config
    db = config.data_dir() / "courtside.db"
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def tier4(min_c) -> np.ndarray:
    x = np.asarray(min_c, float)
    out = np.full(len(x), "bench", dtype=object)
    for name, lo, hi in TIERS:
        out[(x >= lo) & (x < hi)] = name
    return out


def bucket(margin) -> np.ndarray:
    a = np.abs(np.asarray(margin, float))
    out = np.full(len(a), "", dtype=object)
    for lo, hi, name in BUCKETS:
        out[(a >= lo) & (a < hi)] = name
    return out


def cl_mean(z, cl) -> tuple[float, float]:
    """Mean and its SE clustered by `cl`."""
    z = np.asarray(z, float)
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    m = float(z.mean())
    s = pd.Series(z - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    return m, float(np.sqrt(g / max(g - 1, 1) * (s ** 2).sum()) / n)


def load_params(season: str, path: str | None = None):
    from courtside import config
    from courtside.proj import minutes as M
    pdir = config.data_dir() / "proj"
    cands = [Path(path)] if path else [pdir / f"bundle_{season}_v1_cx.json",
                                       pdir / f"bundle_{season}_v1.json"]
    for f in cands:
        if f.exists():
            d = json.loads(f.read_text())
            sd = d.get("copula", {}).get("sd_margin", SD_MARGIN)
            return M.MinutesParams(**d.get("minutes", d)), float(sd), str(f)
    raise FileNotFoundError(cands)


# ============================================================================== minutes level

def sim_at_margin(r: pd.DataFrame, prm, sims: int, seed: int = 23) -> pd.DataFrame:
    """Per row of a projected oracle panel: `sample_minutes` at the team-game's actual margin and
    overtime -> mean (all sims), P(play), mean given play, 10 / 90 % (all sims and given play),
    randomised PIT (all sims)."""
    from courtside.proj import minutes as M
    from courtside.proj.metrics import pit_randomized
    rng = np.random.default_rng(seed)
    cols = ("s_mean", "s_p", "s_meanc", "s_q10", "s_q90", "s_q10c", "s_q90c", "s_pit")
    out = {k: np.full(len(r), np.nan) for k in cols}
    y_all = r["act_min"].fillna(0.0).to_numpy(float)
    ms = r["margin"].to_numpy(float)
    ot = r["n_ot"].fillna(0).to_numpy(float)
    for idx in r.groupby("key", sort=False).indices.values():
        d = r.iloc[idx]
        proj = d[["p_play", "pf36", "reserve"]].assign(mean=d["mean0"].to_numpy())
        x = M.sample_minutes(proj, prm, np.full(sims, ms[idx[0]]), rng,
                             n_ot=np.full(sims, ot[idx[0]]))
        y = y_all[idx]
        out["s_mean"][idx] = x.mean(0)
        pl = x > 0
        out["s_p"][idx] = pl.mean(0)
        with np.errstate(all="ignore"):
            out["s_meanc"][idx] = np.where(pl.any(0), (x * pl).sum(0) / np.maximum(pl.sum(0), 1),
                                           np.nan)
            xc = np.where(pl, x, np.nan)
            ok = pl.sum(0) >= 20
            if ok.any():
                q = np.nanquantile(xc[:, ok], [0.1, 0.9], axis=0)
                out["s_q10c"][idx[ok]], out["s_q90c"][idx[ok]] = q[0], q[1]
        out["s_q10"][idx], out["s_q90"][idx] = np.quantile(x, [0.1, 0.9], axis=0)
        out["s_pit"][idx] = pit_randomized(x, y, rng.random(len(idx)))
    return pd.DataFrame(out, index=r.index)


def cmd_mdiag(season: str, params: str | None, tag: str, sims: int) -> None:
    from courtside.proj import minutes as M
    from courtside.proj import minutes_eval as ME
    t0 = time.time()
    conn = _conn()
    prm, _sd, src = load_params(season, params)
    print(f"[bo] mdiag {season} {tag}: params {src} (minfix {bool(prm.minfix)}, blowout "
          f"{bool(getattr(prm, 'blowout', None))})", flush=True)
    s0 = int(season[:4])
    h = M.load_history(conn, since=f"{s0 - 7}-{(s0 - 6) % 100:02d}")
    ctx = ME.prepare_season(conn, h, season)
    panel = ctx["panel"]
    r = M.run_projection(panel, prm, shares=ctx["shares"])
    sm = sim_at_margin(r, prm, sims)
    keep = ["key", "game_id", "team", "date", "season", "player_id", "roster", "absent",
            "played", "act_min", "act_min_r", "act_started", "base", "p_play", "mean0",
            "reserve", "margin", "n_ot", "ew_f", "ew_s", "st10", "app10"]
    d = pd.concat([r[[c for c in keep if c in r]].reset_index(drop=True),
                   sm.reset_index(drop=True)], axis=1)
    # minutes of the team outside the candidate set (call-ups, deep bench not in the panel)
    d["team_total"] = 240.0 + 25.0 * d["n_ot"].fillna(0)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"mdiag_{season}_{tag}.parquet"
    d.to_parquet(f)
    print(f"[bo] wrote {f} ({len(d)} rows, {time.time() - t0:.0f}s)", flush=True)
    print(mtable([d], [tag]), flush=True)


def _mprep(d: pd.DataFrame, tier_from: pd.Series | None = None) -> pd.DataFrame:
    d = d[~d["absent"].astype(bool)].copy()
    d["y"] = d["act_min"].fillna(0.0)
    d["pl"] = (d["y"] > 0).astype(float)
    d["tier"] = tier4(d["mean0"] if tier_from is None else tier_from.loc[d.index])
    d["bucket"] = bucket(d["margin"])
    d["won"] = np.where(d["margin"] > 0, "won", "lost")
    return d


def mtable(ds: list, tags: list) -> str:
    """Per arm: actual - sim minutes (all rows, given play), P(play) gap, coverage, PIT mean by
    bucket x tier (tier from the first arm), and by bucket x tier x won/lost for stars /
    starters; plus the team's minutes outside the candidates (actual vs the reserve)."""
    base = ds[0]
    L = []
    for d0, tag in zip(ds, tags):
        d = _mprep(d0, base["mean0"] if len(d0) == len(base) else None)
        L.append(f"\n== {tag}: actual - sim at the actual margin (minutes), all available "
                 f"candidates; [given play]; P gap = sim - actual played; cov80 / mean PIT")
        rows = []
        for (b, t), q in d.groupby(["bucket", "tier"]):
            g, se = cl_mean(q["y"] - q["s_mean"], q["date"])
            qp = q[q["pl"] > 0]
            gc, sec = cl_mean(qp["y"] - qp["s_meanc"], qp["date"]) if len(qp) > 1 else (np.nan,) * 2
            cov = ((q["s_q10"] <= q["y"]) & (q["y"] <= q["s_q90"])).mean()
            rows.append({"bucket": b, "tier": t, "n": len(q), "act": q["y"].mean(),
                         "sim": q["s_mean"].mean(), "gap": f"{g:+.2f} ({se:.2f})",
                         "gap|pl": f"{gc:+.2f} ({sec:.2f})",
                         "Pgap": f"{(q['s_p'] - q['pl']).mean():+.3f}",
                         "cov80": f"{cov:.3f}", "pit": f"{q['s_pit'].mean():.3f}"})
        L.append(pd.DataFrame(rows).to_string(index=False))
        rows = []
        for (b, w, t), q in d[d["tier"].isin(["star", "starter", "bench"])].groupby(
                ["bucket", "won", "tier"]):
            g, se = cl_mean(q["y"] - q["s_mean"], q["date"])
            rows.append({"bucket": b, "side": w, "tier": t, "n": len(q), "act": q["y"].mean(),
                         "sim": q["s_mean"].mean(), "gap": f"{g:+.2f} ({se:.2f})"})
        L.append(pd.DataFrame(rows).to_string(index=False))
        # team level: minutes of the candidates (actual vs sims) and outside them
        k = d0.assign(y=d0["act_min"].fillna(0.0)).groupby("key").agg(
            margin=("margin", "first"), date=("date", "first"), tt=("team_total", "first"),
            act=("y", "sum"), sim=("s_mean", "sum"), res=("reserve", "first"))
        k["bucket"] = bucket(k["margin"])
        k["outside"] = k["tt"] - k["act"]
        t = k.groupby("bucket").agg(n=("act", "size"), outside_act=("outside", "mean"),
                                    reserve=("res", "mean"), cand_act=("act", "mean"),
                                    cand_sim=("sim", "mean"))
        L.append("team-games: minutes outside the candidates (actual) vs the reserve; "
                 "candidates' minutes actual vs sims\n" + t.round(2).to_string())
    return "\n".join(L)


def cmd_mtable(files: list) -> None:
    ds = [pd.read_parquet(f) for f in files]
    print(mtable(ds, [Path(f).stem for f in files]))


# ============================================================================== harness

_LAST: dict = {}
WIN0, WIN_STEP, WIN_MAX, WIN_MIN_SIMS = 3.0, 2.0, 15.0, 100
N_RANDOM, LINEUP, STACK = 30, 8, 4
C_COLS = ("act_margin", "c_n", "c_mean", "c_q10", "c_q90", "c_pit", "c_crps", "c_min_mean",
          "c_min_q10", "c_min_q90", "c_min_pit", "min_q10", "min_q90", "min_pit")


def home_margins(conn) -> dict:
    """game_id -> final home margin (neutral site: the alphabetical first team as home, as the
    copula v1 eval)."""
    tg = pd.read_sql_query("SELECT game_id, team, is_home, pts FROM team_games", conn)
    out = {}
    for gid, g in tg.groupby("game_id"):
        if len(g) != 2:
            continue
        h = g[g["is_home"] == 1]
        if len(h) == 1:
            out[gid] = float(h["pts"].iloc[0] - g[g["is_home"] != 1]["pts"].iloc[0])
        else:
            g = g.sort_values("team")
            out[gid] = float(g["pts"].iloc[0] - g["pts"].iloc[1])
    return out


def _window(mt: np.ndarray, a: float) -> np.ndarray:
    w = WIN0
    while True:
        sel = np.abs(mt - a) <= w
        if sel.sum() >= WIN_MIN_SIMS or w >= WIN_MAX:
            return sel
        w += WIN_STEP


def conditional_stats(players: pd.DataFrame, res, day: str, hm: dict) -> pd.DataFrame:
    """C_COLS per projected player of a harness date (module doc: `harness`)."""
    from courtside.proj import pipeline as P
    from courtside.proj.metrics import crps_samples_matrix, pit_randomized
    rng = np.random.default_rng(P.stable_seed(day, "bo-cond", 1))
    yk = {}
    for r in players.itertuples(index=False):
        if r.player_id is not None and np.isfinite(r.player_id) and r.has_proj:
            yk[int(r.player_id)] = (float(r.y), float(r.minutes))
    st = {}
    for g in res.sims:
        a_h = hm.get(g.game_id)
        teams = list(g.teams)
        if a_h is None or not teams:
            continue
        home = teams[0]
        for side_team in np.unique(g.team):
            sign = 1.0 if side_team == home else -1.0
            mt = sign * g.margin
            a = sign * a_h
            sel = _window(mt, a)
            cols = np.flatnonzero((g.team == side_team)
                                  & np.isin(g.player_id, np.fromiter(yk, np.int64)))
            if not len(cols):
                continue
            ys = np.array([yk[int(g.player_id[j])][0] for j in cols])
            ms = np.array([yk[int(g.player_id[j])][1] for j in cols])
            X = g.fp[:, cols].astype(float)
            Xm = g.minutes[:, cols].astype(float)
            Xc, Xmc = X[sel], Xm[sel]
            q = np.quantile(Xc, [0.1, 0.9], axis=0)
            qm = np.quantile(Xmc, [0.1, 0.9], axis=0)
            qu = np.quantile(Xm, [0.1, 0.9], axis=0)
            pit = pit_randomized(Xc, ys, rng.random(len(cols)))
            pitm = pit_randomized(Xmc, ms, rng.random(len(cols)))
            pitu = pit_randomized(Xm, ms, rng.random(len(cols)))
            crps = crps_samples_matrix(Xc, ys)
            for k, j in enumerate(cols):
                st[int(g.player_id[j])] = (a, int(sel.sum()), float(Xc[:, k].mean()),
                                           float(q[0, k]), float(q[1, k]), float(pit[k]),
                                           float(crps[k]), float(Xmc[:, k].mean()),
                                           float(qm[0, k]), float(qm[1, k]), float(pitm[k]),
                                           float(qu[0, k]), float(qu[1, k]), float(pitu[k]))
    out = players.copy()
    vals = [st.get(int(p), (np.nan,) * len(C_COLS)) if p is not None and np.isfinite(p)
            else (np.nan,) * len(C_COLS) for p in out["player_id"]]
    for k, c in enumerate(C_COLS):
        out[c] = [v[k] for v in vals]
    return out


def lineup_rows(players: pd.DataFrame, res, day: str, lineups: pd.DataFrame, hm: dict
                ) -> list:
    """Lineup sims (sum of per-sim DK over the lineup's players; games share the sim index) vs
    the actual score: the projected top-8, 30 random 8-player sets (players >= 15 projected
    minutes, P(play) >= 0.9; chosen from the date and ids only, identical across arms), each
    team's projected top-4 (a one-team stack, with its actual margin) and, on RotoGuru seasons,
    the harness's cash / mean lineups."""
    from courtside.proj import pipeline as P
    col = {}
    for gi, g in enumerate(res.sims):
        for j, pid in enumerate(g.player_id):
            col[int(pid)] = (gi, j)
    S = res.sims[0].fp.shape[0] if res.sims else 0
    act = {int(r.player_id): float(r.y) for r in players.itertuples(index=False)
           if r.player_id is not None and np.isfinite(r.player_id)}
    out = []

    def add(kind, pids, rank=1, actual=None, margin=np.nan):
        x = np.zeros(S)
        for p in pids:
            gi, j = col[int(p)]
            x += res.sims[gi].fp[:, j]
        y = float(sum(act.get(int(p), 0.0) for p in pids)) if actual is None else actual
        q10, q90 = np.quantile(x, (0.1, 0.9))
        out.append({"date": day, "kind": kind, "rank": rank, "n": len(pids), "actual": y,
                    "mean": float(x.mean()), "sd": float(x.std()), "q10": float(q10),
                    "q90": float(q90), "margin": margin,
                    "pit": float((x < y).mean() + 0.5 * (x == y).mean())})

    if len(lineups) and "method" in lineups:
        rg2pid = {str(r.rg_id): int(r.player_id) for r in players.itertuples(index=False)
                  if r.has_proj and r.player_id is not None and np.isfinite(r.player_id)}
        for r in lineups.itertuples(index=False):
            if r.method not in ("ours_cash", "ours_mean") or not r.ids:
                continue
            ids = r.ids.split("|")
            if all(i in rg2pid and rg2pid[i] in col for i in ids):
                add(r.method, [rg2pid[i] for i in ids], int(r.rank), float(r.actual))
    pl = res.players.drop_duplicates("player_id")
    pl = pl[(pl["p_play"] >= 0.9) & (pl["min_played"] >= 15)
            & pl["player_id"].map(lambda p: int(p) in col)].sort_values("player_id")
    if len(pl) >= LINEUP and pl["game_id"].nunique() >= 2:
        top = pl.sort_values(["dk_mean", "player_id"], ascending=[False, True])
        pick = list(top["player_id"].iloc[:LINEUP])
        if top["game_id"].iloc[:LINEUP].nunique() < 2:
            pick = pick[:LINEUP - 1] + [int(top[top["game_id"] != top["game_id"].iloc[0]]
                                            ["player_id"].iloc[0])]
        add("top8", pick)
        rng = np.random.default_rng(P.stable_seed(day, "bo-eval-random"))
        ids = pl["player_id"].to_numpy(np.int64)
        for k in range(N_RANDOM):
            add("random8", list(rng.choice(ids, LINEUP, replace=False)), k + 1)
    for (gid, team), q in pl.groupby(["game_id", "team"]):
        if len(q) < STACK or gid not in hm:
            continue
        g = res.sims[col[int(q["player_id"].iloc[0])][0]]
        sign = 1.0 if team == next(iter(g.teams)) else -1.0
        top = q.sort_values(["dk_mean", "player_id"], ascending=[False, True])
        add("team4", list(top["player_id"].iloc[:STACK]), margin=sign * hm[gid])
    return out


def _patch_capture() -> None:
    from courtside.proj import backtest as B
    from courtside.proj import pipeline as P
    if getattr(B.SeasonRun.run_date, "_bo", False):
        return
    ph, ll, orig = P.project_historical, B.live_like_day, B.SeasonRun.run_date

    def ph2(*a, **k):
        r = ph(*a, **k)
        _LAST["res"] = r
        return r

    def ll2(*a, **k):
        r = ll(*a, **k)
        _LAST["res"] = r
        return r

    def run_date(self, day):
        players, lineups, meta = orig(self, day)
        res = _LAST.pop("res", None)
        if res is not None:
            hm = _LAST["hm"]
            players = conditional_stats(players, res, day, hm)
            _LAST.setdefault("lineups", []).extend(lineup_rows(players, res, day, lineups, hm))
        _LAST["tip_unknown"] = _LAST.get("tip_unknown", 0) + int(meta.get("tip_unknown") or 0)
        return players, lineups, meta

    run_date._bo = True
    P.project_historical, B.live_like_day, B.SeasonRun.run_date = ph2, ll2, run_date


def cmd_harness(season: str, protocol: str, lineups: bool) -> None:
    """T2.1 harness (avail) or the T2.4 slice (live / live10, P(play) cells leave-one-month-out)
    for the arm of the environment switches, 4,000 sims, with C_COLS and lineup rows."""
    from courtside import db
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    _patch_capture()
    conn = db.connect()
    _LAST["hm"] = home_margins(conn)
    _LAST["lineups"] = []
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    rg = B.is_rotoguru(season)
    st = B.Settings(protocol=protocol, n_sims=4000, modes=("cash",) if (rg and lineups) else (),
                    baselines=B.BASELINES, pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    pd.DataFrame(_LAST["lineups"]).to_parquet(
        OUT / f"lineups_{protocol}_{season}_{P.MODEL_VERSION}.parquet")
    tu = _LAST.get("tip_unknown", 0)
    print(f"[bo] harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
          f"{len(meta['skipped'])}, tip_unknown {tu}, {time.time() - t0:.0f}s -> {f}",
          flush=True)


# ============================================================================== report

def _pitdev(u) -> float:
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    if not len(u):
        return float("nan")
    h = np.histogram(np.clip(u, 0, 1 - 1e-12), bins=10, range=(0, 1))[0]
    return float(np.abs(h / h.sum() - 0.1).max())


def pair_rows(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    """Identical harness rows of two arms (projected, E[min] >= 10 in either arm), the OFF
    arm's tier (E[min | plays]) and the actual |margin| bucket."""
    a = off.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    b = on.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    cols = ["mean", "q10", "q90", "crps", "pit", "exp_min", "p_play"] + \
        [c for c in C_COLS if c in b and c != "act_margin"]
    d = a.merge(b[["date", "player_id"] + cols], on=["date", "player_id"],
                suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool)].reset_index(drop=True)
    d = d[(d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)].reset_index(drop=True)
    d["tier4"] = tier4(d["exp_min_off"] / d["p_play_off"].clip(lower=0.05))
    d["bucket"] = bucket(d["act_margin"]) if "act_margin" in d else ""
    d["side"] = np.where(d.get("act_margin", pd.Series(0, index=d.index)) > 0, "won", "lost")
    return d


def paired(d: pd.DataFrame, f) -> dict:
    z0, z1 = np.asarray(f("off"), float), np.asarray(f("on"), float)
    ok = np.isfinite(z0) & np.isfinite(z1)
    cl = d["date"].to_numpy()[ok]
    m0, _ = cl_mean(z0[ok], cl)
    dd, se = cl_mean(z1[ok] - z0[ok], cl)
    return {"off": m0, "on": float(z1[ok].mean()), "diff": dd, "se": se,
            "z": dd / se if se > 0 else float("nan"), "n": int(ok.sum())}


def metrics(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(float)
    m = d["minutes"].fillna(0).to_numpy(float)
    out = {"n": len(d)}
    defs = {"dk_crps": lambda a: d[f"crps_{a}"],
            "dk_mae": lambda a: np.abs(y - d[f"mean_{a}"]),
            "dk_bias": lambda a: d[f"mean_{a}"] - y,
            "dk_cov80": lambda a: ((d[f"q10_{a}"] <= y) & (y <= d[f"q90_{a}"])).astype(float),
            "min_bias": lambda a: d[f"exp_min_{a}"] - m}
    if "c_crps_off" in d:
        defs.update({
            "c_crps": lambda a: d[f"c_crps_{a}"],
            "c_bias": lambda a: d[f"c_mean_{a}"] - y,
            "c_cov80": lambda a: ((d[f"c_q10_{a}"] <= y) & (y <= d[f"c_q90_{a}"])).astype(float),
            "c_pit": lambda a: d[f"c_pit_{a}"],
            "c_min_bias": lambda a: d[f"c_min_mean_{a}"] - m,
            "c_min_cov80": lambda a: ((d[f"c_min_q10_{a}"] <= m) & (m <= d[f"c_min_q90_{a}"]))
            .astype(float),
            "pit": lambda a: d[f"pit_{a}"],
            "min_cov80": lambda a: ((d[f"min_q10_{a}"] <= m) & (m <= d[f"min_q90_{a}"]))
            .astype(float)})
    for k, f in defs.items():
        if len(d) >= 2:
            out[k] = paired(d, f)
    for a in ("off", "on"):
        out[f"pitdev_{a}"] = _pitdev(d[f"pit_{a}"])
        if f"c_pit_{a}" in d:
            out[f"c_pitdev_{a}"] = _pitdev(d[f"c_pit_{a}"])
    return out


def lineup_table(l0: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
    if l0 is None or l1 is None or not len(l0) or not len(l1):
        return pd.DataFrame()
    k = ["date", "kind", "rank"]
    d = l0.merge(l1[k + ["mean", "sd", "q10", "q90", "pit", "actual"]], on=k,
                 suffixes=("_off", "_on"))
    # the same players in both arms, except our own optimised lineups (each arm's own lineup
    # of the date, scored against its own actual score)
    own = d["kind"].isin(["ours_cash", "ours_mean"])
    d = d[own | np.isclose(d["actual_off"], d["actual_on"])]
    rows = []
    groups = [(kd, q) for kd, q in d.groupby("kind")]
    t4 = d[d["kind"] == "team4"]
    if len(t4):
        b = bucket(t4["margin"])
        groups += [(f"team4 {bb}", t4[b == bb]) for bb in BUCKET_NAMES]
    for kd, q in groups:
        if len(q) < 5:
            continue
        sd = paired(q, lambda a, q=q: q[f"sd_{a}"])
        cov = paired(q, lambda a, q=q: ((q[f"q10_{a}"] <= q[f"actual_{a}"])
                                        & (q[f"actual_{a}"] <= q[f"q90_{a}"])).astype(float))
        rows.append({"lineups": kd, "n": len(q), "actual": round(q["actual_off"].mean(), 1),
                     "sim mean off/on": f"{q['mean_off'].mean():.1f} / {q['mean_on'].mean():.1f}",
                     "sd off": round(sd["off"], 2), "sd diff (z)": f"{sd['diff']:+.3f} ({sd['z']:+.1f})",
                     "cov80 off -> on": f"{cov['off']:.3f} -> {cov['on']:.3f}",
                     "pitdev off -> on": f"{_pitdev(q['pit_off']):.3f} -> {_pitdev(q['pit_on']):.3f}"})
    return pd.DataFrame(rows)


def cmd_dk(off: str, on: str, label: str) -> None:
    """Paired ON - OFF tables (module doc: `dk`) -> printed + OUT/dk_<label>.json."""
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    d = pair_rows(a, b)
    res = {"all": metrics(d)}
    for t in TIER_NAMES:
        res[t] = metrics(d[d["tier4"] == t])
    bk = {}
    for bb in BUCKET_NAMES:
        for t in TIER_NAMES:
            q = d[(d["bucket"] == bb) & (d["tier4"] == t)]
            if len(q) >= 20:
                bk[f"{bb}|{t}"] = metrics(q)
        for side in ("won", "lost"):
            q = d[(d["bucket"] == bb) & (d["side"] == side) & (d["tier4"] == "star")]
            if len(q) >= 20:
                bk[f"{bb}|{side}|star"] = metrics(q)
    res["buckets"] = bk
    lo = [Path(off).with_name(Path(off).name.replace("rows_", "lineups_")),
          Path(on).with_name(Path(on).name.replace("rows_", "lineups_"))]
    lt = lineup_table(*(pd.read_parquet(f) if f.exists() else None for f in lo))
    res["lineups"] = lt.to_dict("records")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label}: {Path(off).name} -> {Path(on).name} (rows {len(d)})")
    keys = ["dk_crps", "dk_mae", "dk_bias", "dk_cov80", "min_bias", "c_crps", "c_bias",
            "c_cov80", "c_min_bias", "min_cov80"]

    def line(name, r):
        row = {"rows": name, "n": r["n"]}
        for k in keys:
            if k in r:
                row[k] = f"{r[k]['off']:.3f} {r[k]['diff']:+.4f} ({r[k]['z']:+.1f})"
        row["pitdev"] = f"{r.get('pitdev_off', np.nan):.4f}>{r.get('pitdev_on', np.nan):.4f}"
        if "c_pitdev_off" in r:
            row["c_pitdev"] = f"{r['c_pitdev_off']:.4f}>{r['c_pitdev_on']:.4f}"
        return row
    print(pd.DataFrame([line(n, r) for n, r in res.items() if n not in ("buckets", "lineups")])
          .set_index("rows").to_string())
    print("\nblowout buckets (actual |margin|) x tier: c_ = conditional on the sim margin near "
          "the actual one; bias = sim - actual; c_pit mean")
    rows = []
    for n, r in bk.items():
        row = {"cell": n, "n": r["n"]}
        for k in ("c_bias", "c_min_bias", "c_cov80", "c_pit", "c_crps", "dk_bias", "dk_cov80",
                  "pit"):
            if k in r:
                row[k] = f"{r[k]['off']:+.2f}>{r[k]['on']:+.2f}" if "bias" in k else \
                    f"{r[k]['off']:.3f}>{r[k]['on']:.3f}"
        row["c_crps z"] = f"{r['c_crps']['z']:+.1f}" if "c_crps" in r else ""
        row["crps z"] = f"{r['dk_crps']['z']:+.1f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("cell").to_string())
    if len(lt):
        print("\nlineups: sum of per-sim DK; sd of the lineup sims, 10-90 coverage, PIT max dev")
        print(lt.to_string(index=False))


def cmd_summary(labels: list) -> None:
    """Markdown table (PLAYERS.md results block) from OUT/dk_<label>.json files, one column per
    label."""
    rs = [json.loads((OUT / f"dk_{lb}.json").read_text()) for lb in labels]

    def cell(r, k, fmt="{off:.3f} / {diff:+.4f} ({z:+.1f})"):
        return fmt.format(**r[k]) if k in r else ""

    def cov(r, k="dk_cov80"):
        return f"{r[k]['off']:.3f} → {r[k]['on']:.3f}" if k in r else ""

    def bk(r, key, k):
        b = r["buckets"].get(key)
        return f"{b[k]['off']:+.2f} → {b[k]['on']:+.2f}" if b and k in b else ""

    rows = [("rows", [str(r["all"]["n"]) for r in rs]),
            ("DK CRPS OFF / Δ (z)", [cell(r["all"], "dk_crps") for r in rs]),
            ("DK MAE OFF / Δ (z)", [cell(r["all"], "dk_mae") for r in rs]),
            ("80% cov OFF → ON", [cov(r["all"]) for r in rs]),
            ("DK PIT max dev", [f"{r['all']['pitdev_off']:.4f} → {r['all']['pitdev_on']:.4f}"
                                for r in rs]),
            ("CRPS z star/starter/rotation/bench",
             [" / ".join(f"{r[t]['dk_crps']['z']:+.1f}" for t in TIER_NAMES) for r in rs]),
            ("80% cov star/starter/rotation/bench OFF→ON",
             [" / ".join(f"{r[t]['dk_cov80']['off']:.3f}→{r[t]['dk_cov80']['on']:.3f}"
                         for t in TIER_NAMES) for r in rs]),
            ("minutes 80% cov OFF → ON", [cov(r["all"], "min_cov80") for r in rs]),
            ("DK CRPS at the actual margin Δ (z)",
             [cell(r["all"], "c_crps", "{diff:+.4f} ({z:+.1f})") for r in rs]),
            ("… z star/starter/rotation/bench",
             [" / ".join(f"{r[t]['c_crps']['z']:+.1f}" for t in TIER_NAMES) for r in rs])]
    for bb in BUCKET_NAMES:
        rows.append((f"star DK bias at the margin, {bb}", [bk(r, f"{bb}|star", "c_bias")
                                                           for r in rs]))
    for bb in ("20-25", "26+"):
        for side in ("won", "lost"):
            rows.append((f"star DK bias at the margin, {bb} {side}",
                         [bk(r, f"{bb}|{side}|star", "c_bias") for r in rs]))
    for bb in BUCKET_NAMES:
        rows.append((f"star minutes bias at the margin, {bb}",
                     [bk(r, f"{bb}|star", "c_min_bias") for r in rs]))
    for bb in ("0-9", "26+"):
        rows.append((f"bench DK bias at the margin, {bb}", [bk(r, f"{bb}|bench", "c_bias")
                                                            for r in rs]))
    for kd in ("ours_cash", "top8", "random8", "team4", "team4 26+"):
        vals = []
        for r in rs:
            m = [x for x in r.get("lineups", []) if x["lineups"] == kd]
            vals.append(f"{m[0]['sd off']:.1f} {m[0]['sd diff (z)']}; cov {m[0]['cov80 off -> on']}"
                        if m else "")
        if any(vals):
            rows.append((f"lineup sim sd OFF Δ (z); cov80, {kd}", vals))
    print("| | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for name, vals in rows:
        print(f"| {name} | " + " | ".join(vals) + " |")


def cmd_fit(season: str, tag: str, minfix: bool, blowout: bool) -> None:
    """Minutes params of S (5 seasons < S, the pipeline's recipe: oracle training panel, pair
    shares of it, v1, 1 LightGBM thread) [+ minfix] [+ blowout] -> OUT/mparams_<S>_<T>.json."""
    from courtside.proj import minutes as M
    M.GBM_PARAMS["num_threads"] = 1
    t0 = time.time()
    conn = _conn()
    s0 = int(season[:4])
    h = M.load_history(conn, since=f"{s0 - 7}-{(s0 - 6) % 100:02d}")
    train = sorted(x for x in h.tg["season"].unique() if x < season)[-5:]
    panel = M.oracle_panel(h, M.eval_targets(h, train))
    kw: dict = {}
    if minfix:
        kw["minfix"] = True
    if blowout:
        kw["blowout"] = True
    mp = M.fit(h, season, window=5, log=lambda m: print(m, flush=True), panel=panel,
               shares=M.PairShares(M.pair_events(panel)), version="v1", v1_kw=kw)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"mparams_{season}_{tag}.json"
    f.write_text(mp.to_json())
    print(f"[bo] fit {season} {tag}: {f} ({time.time() - t0:.0f}s)", flush=True)


def cmd_same(a: str, b: str) -> None:
    """Are two minutes params identical apart from `blowout` and the timing diagnostics?"""
    da, db = (json.loads(Path(f).read_text()) for f in (a, b))
    da, db = da.get("minutes", da), db.get("minutes", db)
    for d in (da, db):
        d.pop("blowout", None)
        d.pop("diagnostics", None)
    diff = [k for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
    print(f"[bo] same {a} {b}: {'IDENTICAL' if not diff else 'DIFFER in ' + str(diff)}")


# ============================================================================== main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mdiag")
    a.add_argument("season")
    a.add_argument("--params")
    a.add_argument("--tag", default="off")
    a.add_argument("--sims", type=int, default=400)
    t = sub.add_parser("mtable")
    t.add_argument("files", nargs="+")
    ft = sub.add_parser("fit")
    ft.add_argument("season")
    ft.add_argument("--tag", required=True)
    ft.add_argument("--minfix", action="store_true")
    ft.add_argument("--blowout", action="store_true")
    hh = sub.add_parser("harness")
    hh.add_argument("season")
    hh.add_argument("protocol", choices=("avail", "live", "live10"))
    hh.add_argument("--lineups", action="store_true")
    dk = sub.add_parser("dk")
    dk.add_argument("off")
    dk.add_argument("on")
    dk.add_argument("--label", default="cmp")
    su = sub.add_parser("summary")
    su.add_argument("labels", nargs="+")
    sm = sub.add_parser("same")
    sm.add_argument("a")
    sm.add_argument("b")
    x = ap.parse_args(argv)
    if x.cmd == "harness":
        cmd_harness(x.season, x.protocol, x.lineups)
    elif x.cmd == "summary":
        cmd_summary(x.labels)
    elif x.cmd == "dk":
        cmd_dk(x.off, x.on, x.label)
    elif x.cmd == "fit":
        cmd_fit(x.season, x.tag, x.minfix, x.blowout)
    elif x.cmd == "same":
        cmd_same(x.a, x.b)
    elif x.cmd == "mdiag":
        cmd_mdiag(x.season, x.params, x.tag, x.sims)
    elif x.cmd == "mtable":
        cmd_mtable(x.files)


if __name__ == "__main__":
    main()
