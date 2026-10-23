"""T2.10 evaluation: Path A copula v1 vs v0 on top of a context arm (docs/PLAYERS.md T2.10, §7.3).

Runs the T2.1 harness's per-date work (`backtest.SeasonRun.run_date`: walk-forward bundle fit on
seasons < S, avail protocol, closing lines, the harness's CRPS / randomised PIT) and, from the
same sims, the extra T2.10 metrics the harness does not keep:

  * per-stat calibration (pts / reb / ast / fg3m, conditional on playing): randomised PIT of the
    v1 component sims and of the props.py approximation (NB given the run's minutes, the
    fallback), and P(stat > line) vs realised at two typical lines: the player's season-to-date
    average of the stat (floor + 0.5, a market-like line) and his projected median;
  * the blowout curve: per tier, E[DK | margin bin] - E[DK] of the sims (played sims, weighted
    by each game's simulated bin probability) vs the actual DK residual by the actual bin, for
    |margin| and for the signed team margin (winner / loser);
  * lineup-level calibration: the sum of the lineup's per-sim DK points (games are independent,
    same sim index) -> 10-90 coverage and PIT of the actual lineup score. RotoGuru seasons: the
    harness's cash lineup (`ours_cash`). Other seasons: the projected top-8 (>= 2 games) and 30
    random 8-player sets per date among players projected >= 15 minutes (identical in both arms:
    chosen from the date and player ids only).

The arm comes from the environment (COURTSIDE_CONTEXT / _EXT / COURTSIDE_COPULA), so the two
copula arms share bundles (v1 = the arm's bundle + copula_v1) and seeds (common random numbers).

  python research/copula_v1_eval.py run --season 2024-25 --out DIR [--dates N] [--sims 4000]
  python research/copula_v1_eval.py report --v0 DIR0 --v1 DIR1 [--seasons ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj import backtest as B
from courtside.proj import copula as C
from courtside.proj import metrics as MT
from courtside.proj import pipeline as P
from courtside.proj import props as PR

PSTATS = ("pts", "reb", "ast", "fg3m")
ABS_EDGES = (-0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 1e9)
ABS_LABELS = ("0-5", "6-10", "11-15", "16-20", "21-25", "26+")
SGN_EDGES = (-1e9, -20.5, -10.5, -3.5, 3.5, 10.5, 20.5, 1e9)
SGN_LABELS = ("<=-21", "-20..-11", "-10..-4", "-3..3", "4..10", "11..20", ">=21")
N_RANDOM = 30
LINEUP = 8


# ============================================================================== per date

def _actual_stats(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    a = pd.read_sql_query(
        "SELECT game_id, player_id, game_date, min, pts, reb, ast, fg3m, stl, blk, tov FROM "
        "player_games WHERE season IN (?, ?)", conn, params=(season, P._prev_season(season)))
    return a


class StdLines:
    """Season-to-date average per stat (the player's games of S strictly before the date; the
    previous season's average before his 5th game): a market-like prop line floor(avg) + 0.5."""

    def __init__(self, acts: pd.DataFrame, season: str):
        a = acts[acts["min"] > 0].sort_values(["player_id", "game_date"])
        a = a.assign(season=[P.season_of(pd.Timestamp(d).date()) for d in a["game_date"]])
        self.by = {pid: g for pid, g in a.groupby("player_id")}
        self.season = season

    def get(self, pid: int, day: str) -> dict:
        g = self.by.get(int(pid))
        if g is None:
            return {}
        prev = g[g["game_date"] < day]
        cur = prev[prev["season"] == self.season]
        src = cur if len(cur) >= 5 else prev[prev["season"] != self.season]
        if not len(src):
            src = cur
        if not len(src):
            return {}
        return {s: float(src[s].mean()) for s in PSTATS}


def _pit_pmf(pmf: np.ndarray, y: int, v: float) -> float:
    cdf = np.cumsum(pmf)
    lo = float(cdf[y - 1]) if 0 < y <= len(cdf) else (0.0 if y <= 0 else 1.0)
    hi = float(cdf[y]) if 0 <= y < len(cdf) else 1.0
    return lo + v * (hi - lo)


def _p_above(pmf: np.ndarray, line: float) -> float:
    return PR.line_probs(pmf, line)[0]


def date_extras(res, day: str, season: str, acts_key: dict, lines: StdLines,
                rng: np.random.Generator, phi: dict | None = None) -> tuple[list, list, list]:
    """(stat rows, blowout rows, per-game lineup inputs) for one date."""
    st = res.stats
    stat_rows, blow_rows = [], []
    keyed = {}
    for k, g in st.groupby("key"):
        keyed[int(k)] = {(r.stat, r.cond): r._asdict() for r in g.itertuples(index=False)}
    players = res.players.reset_index(drop=True)
    pos = {}
    for gi, g in enumerate(res.sims):
        for j, pid in enumerate(g.player_id):
            pos[(g.game_id, int(pid))] = (gi, j)
    for k, r in players.iterrows():
        gid, pid = r["game_id"], int(r["player_id"])
        a = acts_key.get((gid, pid))
        gi, j = pos[(gid, pid)]
        g = res.sims[gi]
        played_sims = g.minutes[:, j] > 0
        if a is None or not (a["min"] or 0) > 0 or played_sims.sum() < 200:
            continue
        approx = PR.player_dist(keyed.get(k, {}), phi=phi)
        sims = (PR.sims_dist({s: g.stats[s][:, j] for s in PSTATS}, g.minutes[:, j])
                if g.stats is not None else None)
        std = lines.get(pid, day)
        for s in PSTATS:
            y = int(a[s])
            v = float(rng.random())
            row = {"date": day, "season": season, "game_id": gid, "player_id": pid,
                   "tier": r["tier"], "stat": s, "y": y}
            for name, d in (("approx", approx), ("sims", sims)):
                pmf = d.pmf(s) if d is not None else None
                if pmf is None:
                    continue
                row[f"pit_{name}"] = _pit_pmf(pmf, y, v)
                row[f"mean_{name}"] = float((np.arange(len(pmf)) * pmf).sum())
                med = PR.pmf_quantile(pmf, 0.5)
                row[f"pmed_{name}"] = _p_above(pmf, med + 0.5)
                row[f"line_med_{name}"] = med + 0.5
                if s in std:
                    L = math.floor(std[s]) + 0.5
                    row["line_std"] = L
                    row[f"pstd_{name}"] = _p_above(pmf, L)
            stat_rows.append(row)
    # blowout curve, per player-game that played (actual) with enough played sims
    for g in res.sims:
        teams = list(g.teams)
        home = teams[0]
        S = g.fp.shape[0]
        mh = g.margin
        for j, pid in enumerate(g.player_id):
            a = acts_key.get((g.game_id, int(pid)))
            pl = g.minutes[:, j] > 0
            if a is None or not (a["min"] or 0) > 0 or pl.sum() < 200:
                continue
            sign = 1.0 if g.team[j] == home else -1.0
            mt = sign * mh
            x = g.fp[pl, j].astype(float)
            mean = x.mean()
            row = {"date": day, "game_id": g.game_id, "player_id": int(pid),
                   "tier": g.info.at[j, "tier"], "pmean": mean,
                   "y": float(a["dk"]), "act_margin": float(sign * a["home_margin"])}
            for kind, edges, labels, m in (("a", ABS_EDGES, ABS_LABELS, np.abs(mt)),
                                           ("s", SGN_EDGES, SGN_LABELS, mt)):
                b = np.clip(np.searchsorted(edges, m[pl], side="right") - 1, 0, len(labels) - 1)
                for i, lab in enumerate(labels):
                    sel = b == i
                    row[f"{kind}p_{lab}"] = float(sel.mean())
                    row[f"{kind}d_{lab}"] = float(x[sel].mean() - mean) if sel.sum() >= 20 else 0.0
            blow_rows.append(row)
    return stat_rows, blow_rows


def lineup_rows(res, day: str, players_df: pd.DataFrame, lineups: pd.DataFrame,
                acts_key: dict, rg: bool, n_random: int = N_RANDOM) -> list:
    """Lineup sims (sum of per-sim DK over the lineup's players) vs the actual score."""
    col = {}
    for gi, g in enumerate(res.sims):
        for j, pid in enumerate(g.player_id):
            col[int(pid)] = (gi, j)
    S = res.sims[0].fp.shape[0] if res.sims else 0

    def sims_of(pids):
        tot = np.zeros(S)
        for p in pids:
            gi, j = col[int(p)]
            tot += res.sims[gi].fp[:, j]
        return tot

    out = []

    def add(kind, pids, actual, rank=1):
        x = sims_of(pids)
        q10, q50, q90 = np.quantile(x, (0.1, 0.5, 0.9))
        out.append({"date": day, "kind": kind, "rank": rank, "n": len(pids), "actual": actual,
                    "mean": float(x.mean()), "q10": float(q10), "q50": float(q50),
                    "q90": float(q90), "sd": float(x.std()),
                    "pit": float((x < actual).mean() + 0.5 * (x == actual).mean())})

    if rg and len(lineups):
        rg2pid = {str(r.rg_id): int(r.player_id) for r in players_df.itertuples(index=False)
                  if r.has_proj and not pd.isna(r.player_id)}
        for r in lineups.itertuples(index=False):
            if r.method not in ("ours_cash", "ours_mean", "ours_topn") or not r.ids:
                continue
            ids = r.ids.split("|")
            if not all(i in rg2pid and rg2pid[i] in col for i in ids):
                continue
            add(r.method, [rg2pid[i] for i in ids], float(r.actual), int(r.rank))
        return out

    def actual_of(pids):
        tot = 0.0
        for p in pids:
            gi, j = col[int(p)]
            a = acts_key.get((res.sims[gi].game_id, int(p)))
            tot += float(a["dk"]) if a is not None and (a["min"] or 0) > 0 else 0.0
        return tot

    pl = res.players.copy()
    pl = pl[(pl["p_play"] >= 0.9) & (pl["min_played"] >= 15)].sort_values("player_id")
    if len(pl) < LINEUP or pl["game_id"].nunique() < 2:
        return out
    top = pl.sort_values(["dk_mean", "player_id"], ascending=[False, True])
    pick = list(top["player_id"].iloc[:LINEUP])
    if top["game_id"].iloc[:LINEUP].nunique() < 2:
        pick = pick[:LINEUP - 1] + [int(top[top["game_id"] != top["game_id"].iloc[0]]
                                        ["player_id"].iloc[0])]
    add("top8", pick, actual_of(pick))
    rng = np.random.default_rng(P.stable_seed(day, "c1-eval-random"))
    ids = pl["player_id"].to_numpy(np.int64)
    for k in range(n_random):
        pk = rng.choice(ids, LINEUP, replace=False)
        add("random8", list(pk), actual_of(pk), k + 1)
    return out


# ============================================================================== run

def _wait_windows(spec: str | None, log) -> None:
    """Sleep while the Eastern time is inside one of the 'HH:MM-HH:MM' windows (live lock
    runs on the shared Mac)."""
    if not spec:
        return
    from datetime import datetime
    from zoneinfo import ZoneInfo
    wins = []
    for w in spec.split(","):
        a, b = w.split("-")
        wins.append((a.strip(), b.strip()))
    said = False
    while True:
        now = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")
        if not any(a <= now < b for a, b in wins):
            return
        if not said:
            log(f"  pausing in lock window at {now} ET")
            said = True
        time.sleep(30)


def cmd_run(a) -> int:
    conn = sqlite3.connect(f"file:{a.db or config.db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out = Path(a.out) / a.season
    out.mkdir(parents=True, exist_ok=True)
    rg = B.is_rotoguru(a.season)
    st = B.Settings(protocol="avail", n_sims=a.sims, modes=("cash",) if rg else (),
                    baselines=(), topn_baselines=())
    log = lambda m: print(m, flush=True)                                   # noqa: E731
    sr = B.SeasonRun(conn, a.season, st, log=log)
    days = sr.dates(a.dates)
    acts = _actual_stats(conn, a.season)
    tg = pd.read_sql_query("SELECT h.game_id, h.pts - w.pts AS home_margin FROM team_games h "
                           "JOIN team_games w ON w.game_id = h.game_id AND w.is_home = 0 "
                           "WHERE h.is_home = 1 AND h.season = ?", conn, params=(a.season,))
    neutral = pd.read_sql_query(
        "SELECT game_id, team, pts FROM team_games WHERE season = ? AND game_id NOT IN "
        "(SELECT game_id FROM team_games WHERE is_home = 1 AND season = ?)", conn,
        params=(a.season, a.season))
    hm = dict(zip(tg["game_id"], tg["home_margin"]))
    for gid, g in neutral.groupby("game_id"):             # neutral site: alphabetical "home"
        g = g.sort_values("team")
        if len(g) == 2:
            hm[gid] = float(g["pts"].iloc[0] - g["pts"].iloc[1])
    cats = sum((acts[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    acts["dk"] = (sum(C.DK_W[s] * acts[s].astype(float) for s in C.DK_STATS)
                  + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
    acts["home_margin"] = acts["game_id"].map(hm)
    acts_key = {(r.game_id, int(r.player_id)): r._asdict() for r in acts.itertuples(index=False)}
    lines = StdLines(acts, a.season)
    phi = PR.phi_for(conn, a.season)          # props dispersion fit on seasons < S only (B7)
    captured = {}
    orig = P.project_historical

    def cap(*args, **kw):
        r = orig(*args, **kw)
        captured["res"] = r
        return r
    P.project_historical = cap
    sr.load()
    t0 = time.time()
    log(f"[c1eval] {a.season} {len(days)} dates, {P.MODEL_VERSION}, sims {a.sims}")
    for k, day in enumerate(days):
        f = out / f"{day}.players.parquet"
        if f.exists():
            continue
        _wait_windows(a.pause, log)
        t1 = time.time()
        players, lineups, meta = sr.run_date(day)
        res = captured.pop("res")
        rng = np.random.default_rng(P.stable_seed(day, "c1-eval-pit"))
        srows, brows = date_extras(res, day, a.season, acts_key, lines, rng, phi)
        lrows = lineup_rows(res, day, players, lineups, acts_key, rg)
        pd.DataFrame(srows).to_parquet(out / f"{day}.stats.parquet")
        pd.DataFrame(brows).to_parquet(out / f"{day}.blow.parquet")
        pd.DataFrame(lrows).to_parquet(out / f"{day}.lineups.parquet")
        meta.update({"model_version": P.MODEL_VERSION, "seconds_sim": res.meta["seconds_sim"],
                     "seconds_extras": round(time.time() - t1 - meta["seconds"], 2)})
        (out / f"{day}.meta.json").write_text(json.dumps(meta, default=str))
        players.to_parquet(f)                                # written last: the commit marker
        if k % 10 == 0:
            log(f"  {day} ({k + 1}/{len(days)}) {meta['seconds']}s + extras "
                f"{meta['seconds_extras']}s, elapsed {time.time() - t0:.0f}s")
    log(f"[c1eval] done in {time.time() - t0:.0f}s")
    return 0


# ============================================================================== report

def _load(d: Path, kind: str) -> pd.DataFrame:
    fs = sorted(d.glob(f"*.{kind}.parquet"))
    fr = [pd.read_parquet(f) for f in fs]
    fr = [x for x in fr if len(x)]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def _pc(d, cl) -> tuple[float, float]:
    r = MT.paired_cluster(np.asarray(d, float), np.asarray(cl))
    return r["mean"], r["se"]


def _pit_dev(u: np.ndarray, bins: int = 10) -> tuple[list, float]:
    h = MT.pit_histogram(u, bins)
    return [float(x) for x in h], float(np.max(np.abs(np.asarray(h) - 1.0 / bins)))


def spiegelhalter(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    num = ((y - p) * (1 - 2 * p)).sum()
    den = np.sqrt((((1 - 2 * p) ** 2) * p * (1 - p)).sum())
    return float(num / den)


def report_season(d0: Path, d1: Path, season: str) -> dict:
    out: dict = {"season": season}
    p0, p1 = _load(d0 / season, "players"), _load(d1 / season, "players")
    k = ["date", "player_id", "rg_id"]
    p0 = p0[p0["has_proj"].astype(bool)].copy()
    p1 = p1[p1["has_proj"].astype(bool)].copy()
    for p in (p0, p1):
        p["rg_id"] = p["rg_id"].fillna("")
    j = p0.merge(p1, on=k, suffixes=("_0", "_1"))
    j = j[j["crps_0"].notna() & j["crps_1"].notna()]
    assert np.allclose(j["y_0"], j["y_1"])
    out["n_rows"], out["n_dates"] = int(len(j)), int(j["date"].nunique())
    out["exp_min_identical"] = bool(np.allclose(j["exp_min_0"], j["exp_min_1"]))
    cl = j["date"].to_numpy()
    y = j["y_0"].to_numpy()
    for name, col in (("crps", "crps"), ("mae", "mean")):
        a0 = j[f"{col}_0"].to_numpy() if name == "crps" else np.abs(j["mean_0"] - y)
        a1 = j[f"{col}_1"].to_numpy() if name == "crps" else np.abs(j["mean_1"] - y)
        m, se = _pc(np.asarray(a0) - np.asarray(a1), cl)
        out[name] = {"v0": float(np.mean(a0)), "v1": float(np.mean(a1)), "gain": m, "se": se,
                     "z": m / se if se else None}
    # unconditional population of the harness report (exp min >= 10)
    u = j[j["exp_min_0"] >= 10]
    clu = u["date"].to_numpy()
    yu = u["y_0"].to_numpy()
    for name in ("crps", "mae"):
        a0 = u["crps_0"].to_numpy() if name == "crps" else np.abs(u["mean_0"] - yu)
        a1 = u["crps_1"].to_numpy() if name == "crps" else np.abs(u["mean_1"] - yu)
        m, se = _pc(a0 - a1, clu)
        out[f"{name}_u10"] = {"v0": float(a0.mean()), "v1": float(a1.mean()), "gain": m,
                              "se": se, "z": m / se if se else None, "n": int(len(u))}
    for arm in ("0", "1"):
        c80 = ((u[f"q10_{arm}"] <= yu) & (yu <= u[f"q90_{arm}"])).astype(float)
        c50 = ((u[f"q25_{arm}"] <= yu) & (yu <= u[f"q75_{arm}"])).astype(float)
        r80 = MT.rate_se(c80.to_numpy(), clu)
        out[f"cov80_{arm}"] = (r80["rate"], r80["se"])
        out[f"cov50_{arm}"] = float(c50.mean())
        h, dev = _pit_dev(u[f"pit_{arm}"].to_numpy())
        out[f"pit_{arm}"] = h
        out[f"pit_maxdev_{arm}"] = dev
        out[f"bias_{arm}"] = float((u[f"mean_{arm}"] - yu).mean())
        tiers = {}
        for t, g in u.groupby(f"tier_{arm}"):
            yy = g["y_0"].to_numpy()
            tiers[str(t)] = {"n": int(len(g)),
                             "cov80": float(((g[f"q10_{arm}"] <= yy) & (yy <= g[f"q90_{arm}"])
                                             ).mean()),
                             "crps": float(g[f"crps_{arm}"].mean()),
                             "bias": float((g[f"mean_{arm}"] - yy).mean())}
        out[f"tiers_{arm}"] = tiers
    d80 = (((u["q10_1"] <= yu) & (yu <= u["q90_1"])).astype(float)
           - ((u["q10_0"] <= yu) & (yu <= u["q90_0"])).astype(float))
    m, se = _pc(d80.to_numpy(), clu)
    out["cov80_diff"] = (m, se)
    # per-stat calibration (v1 sims vs the props approximation, each arm's own run)
    s0, s1 = _load(d0 / season, "stats"), _load(d1 / season, "stats")
    out["stats"] = stat_report(s0, s1)
    # blowout curve
    out["blowout"] = {"v0": blow_report(_load(d0 / season, "blow")),
                      "v1": blow_report(_load(d1 / season, "blow"))}
    # lineups
    out["lineups"] = {"v0": lineup_report(_load(d0 / season, "lineups")),
                      "v1": lineup_report(_load(d1 / season, "lineups"))}
    secs = []
    for arm, d in (("v0", d0), ("v1", d1)):
        ms = [json.loads(f.read_text()) for f in sorted((d / season).glob("*.meta.json"))]
        secs.append({"arm": arm, "dates": len(ms),
                     "seconds_sim_per_date": float(np.mean([m["seconds_sim"] for m in ms])),
                     "games_per_date": float(np.mean([m["n_games"] for m in ms]))})
    out["timing"] = secs
    return out


def stat_report(s0: pd.DataFrame, s1: pd.DataFrame) -> dict:
    out = {}
    for s in PSTATS:
        a = s1[s1["stat"] == s]
        z = s0[s0["stat"] == s]
        r = {"n": int(len(a))}
        for name, df, col in (("v1_sims", a, "sims"), ("v1_approx", a, "approx"),
                              ("v0_approx", z, "approx")):
            if f"pit_{col}" not in df or df[f"pit_{col}"].isna().all():
                continue
            h, dev = _pit_dev(df[f"pit_{col}"].dropna().to_numpy())
            yy = df["y"].to_numpy()
            ok = df[f"pstd_{col}"].notna() & df["line_std"].notna()
            p_std = df.loc[ok, f"pstd_{col}"].to_numpy()
            y_std = (df.loc[ok, "y"].to_numpy() > df.loc[ok, "line_std"].to_numpy()).astype(float)
            p_med = df[f"pmed_{col}"].to_numpy()
            y_med = (yy > df[f"line_med_{col}"].to_numpy()).astype(float)
            ll = -np.mean(y_std * np.log(np.clip(p_std, 1e-6, 1)) + (1 - y_std)
                          * np.log(np.clip(1 - p_std, 1e-6, 1)))
            r[name] = {"pit": h, "pit_maxdev": dev,
                       "mean": float(df[f"mean_{col}"].mean()), "actual": float(yy.mean()),
                       "std_line": {"n": int(ok.sum()), "p": float(p_std.mean()),
                                    "freq": float(y_std.mean()),
                                    "z": spiegelhalter(p_std, y_std), "logloss": float(ll),
                                    "brier": float(np.mean((p_std - y_std) ** 2)),
                                    "bins": _reliab(p_std, y_std)},
                       "med_line": {"p": float(p_med.mean()), "freq": float(y_med.mean()),
                                    "z": spiegelhalter(p_med, y_med)}}
        out[s] = r
    return out


def _reliab(p, y, edges=(0, .2, .35, .45, .55, .65, .8, 1.0001)) -> list:
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [{"bin": f"{edges[i]:.2f}-{min(edges[i + 1], 1):.2f}", "n": int((b == i).sum()),
             "p": float(p[b == i].mean()) if (b == i).any() else None,
             "freq": float(y[b == i].mean()) if (b == i).any() else None}
            for i in range(len(edges) - 1)]


def blow_report(b: pd.DataFrame) -> dict:
    out = {}
    if not len(b):
        return out
    b = b[b["act_margin"].notna()].copy()
    b["res"] = b["y"] - b["pmean"]
    for kind, edges, labels, m in (("abs", ABS_EDGES, ABS_LABELS, b["act_margin"].abs()),
                                   ("signed", SGN_EDGES, SGN_LABELS, b["act_margin"])):
        b["bin"] = np.clip(np.searchsorted(edges, m.to_numpy(), side="right") - 1, 0,
                           len(labels) - 1)
        k = "a" if kind == "abs" else "s"
        tab = {}
        for t in ("star", "role", "bench"):
            g = b[b["tier"] == t]
            if not len(g):
                continue
            base = float(g["res"].mean())
            rows = []
            for i, lab in enumerate(labels):
                gi = g[g["bin"] == i]
                w = g[f"{k}p_{lab}"].to_numpy()
                sim = float((w * g[f"{k}d_{lab}"].to_numpy()).sum() / max(w.sum(), 1e-9))
                act = float(gi["res"].mean()) if len(gi) else float("nan")
                se = float(gi["res"].std() / np.sqrt(len(gi))) if len(gi) > 1 else float("nan")
                rows.append({"bin": lab, "n": int(len(gi)), "actual": act,
                             "actual_c": act - base, "se": se, "sim": sim,
                             "diff_c": sim - (act - base)})
            tab[t] = rows
        out[kind] = tab
    return out


def lineup_report(lu: pd.DataFrame) -> dict:
    out = {}
    if not len(lu):
        return out
    for kind, g in lu.groupby("kind"):
        if kind == "ours_topn":
            g = g[g["rank"] == 1]
        y = g["actual"].to_numpy()
        cov = ((g["q10"] <= y) & (y <= g["q90"])).astype(float).to_numpy()
        r = MT.rate_se(cov, g["date"].to_numpy())
        h, dev = _pit_dev(g["pit"].to_numpy())
        out[kind] = {"n": int(len(g)), "cov80": r["rate"], "cov80_se": r["se"],
                     "pit": h, "pit_maxdev": dev,
                     "below_q10": float((y < g["q10"]).mean()),
                     "above_q90": float((y > g["q90"]).mean()),
                     "mean_minus_actual": float((g["mean"] - y).mean()),
                     "sd": float(g["sd"].mean())}
    return out


def cmd_report(a) -> int:
    reps = [report_season(Path(a.v0), Path(a.v1), s) for s in a.seasons]
    txt = json.dumps(reps, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--season", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--dates", type=int, default=0)
    r.add_argument("--sims", type=int, default=4000)
    r.add_argument("--db", default=None)
    r.add_argument("--pause", default=None, help="ET windows to pause in, '14:05-14:55,...'")
    p = sub.add_parser("report")
    p.add_argument("--v0", required=True)
    p.add_argument("--v1", required=True)
    p.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
