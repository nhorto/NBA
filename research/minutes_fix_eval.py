"""Minutes fix (`COURTSIDE_MINFIX=1`, `+mf`): minutes-level diagnosis and evaluation.

The minutes-ratio (+mr) and preseason (+pre) switches both failed their bars on the minutes side
(docs/PLAYERS.md §0.4): bench players planned below their recent minutes (E[min | plays] < 0.9 x
the mean of their last 10 appearances) had P(play) +0.04..0.06 and E[min] +0.4..0.8 too high on
the avail harness, rotation / bench minute draws looked too dispersed, and the season-start debut
reserve over-reserved. This script works at the minutes level only (no DK sims): the avail oracle
panel of a season (`minutes_eval.prepare_season`: candidates, regulars who did not play absent),
the minutes engine of that season (params fit on the 5 seasons < S), and minute draws exactly as
the harness sims make them for one availability scenario (`minutes.sample_minutes`: margin ~
N(closing spread, sd), OT when |margin| < 1, early exits, DNP draws, the per-sim 240).

  diag S [--params FILE] [--tag T] [--sims N]
      per-row decomposition of season S (GBM base, stage-2 lift, reconciled mean, P(play),
      margin-integrated E[min], sim minutes PIT / 80 % interval / CRPS) and the reason a player is
      planned below his last-10 minutes -> OUT/diag_<S>_<T>.parquet + a printed summary
  report OUT/diag_<S>_<A>.parquet OUT/diag_<S>_<B>.parquet
      paired B - A on identical rows (minutes MAE / CRPS / bias, P(play) Brier / bias, 80 %
      coverage, PIT) overall, by tier and for the planned-demotion rows; SE clustered by date
  debut DB
      the season-start debut reserve checked on the 2026-27 games played so far plus history

Population for the paired tables (as the harness): available candidates, E[min] (A) >= 10 for
the DK-like population, all available candidates for the P(play) tables. Tiers by E[min | plays]
(A): star >= 30, starter 24-30, rotation 15-24, bench < 15. Demoted / promoted: E[min | plays] <
0.9 x / > 1.1 x the mean minutes of the last 10 appearances (strictly before the date).

Environment: COURTSIDE_DATA (bundles bundle_<S>_v1_cx.json or minutes_params_<S>_v1*.json),
MF_OUT (default var/tmp/mf/out). Heavy: run on the Mac scratch (`courtside-compute/mf`).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MF_OUT", ROOT / "var" / "tmp" / "mf" / "out"))
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
SD_MARGIN = 13.5


def _conn():
    from courtside import config
    db = config.data_dir() / "courtside.db"
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def load_params(season: str, path: str | None = None):
    from courtside import config
    from courtside.proj import minutes as M
    pdir = config.data_dir() / "proj"
    cands = [Path(path)] if path else [pdir / f"bundle_{season}_v1_cx.json",
                                       pdir / f"bundle_{season}_v1.json",
                                       pdir / f"minutes_params_{season}_v1.json"]
    for f in cands:
        if f.exists():
            d = json.loads(f.read_text())
            sd = d.get("copula", {}).get("sd_margin", SD_MARGIN)
            return M.MinutesParams(**d.get("minutes", d)), float(sd), str(f)
    raise FileNotFoundError(cands)


def last10_minutes(h, panel: pd.DataFrame) -> np.ndarray:
    """Mean box minutes of each row's last 10 appearances strictly before the row's date (any
    team; NaN without an appearance)."""
    hh = h.hist[["player_id", "date", "min"]].sort_values(["player_id", "date"], kind="mergesort")
    g = hh.groupby("player_id")["min"]
    hh = hh.assign(l10=g.transform(lambda s: s.rolling(10, min_periods=1).mean()))
    q = pd.DataFrame({"player_id": panel["player_id"].to_numpy(np.int64),
                      "date_q": panel["date"].to_numpy(np.int64) - 1,
                      "i": np.arange(len(panel))}).sort_values("date_q", kind="mergesort")
    hh = hh.rename(columns={"date": "date_q"}).astype({"player_id": np.int64, "date_q": np.int64})
    m = pd.merge_asof(q, hh[["player_id", "date_q", "l10"]].sort_values("date_q"),
                      on="date_q", by="player_id", direction="backward")
    return m.sort_values("i")["l10"].to_numpy(float)


def reasons(panel: pd.DataFrame, absent: np.ndarray) -> pd.DataFrame:
    """Per row, the situation behind a plan below recent minutes: own return / ramp (missed the
    team's last game or <= 2 games back from >= 3 missed), a teammate in his groups back today
    (missed >= 1 team game, available now) or back within 2 games, new team / season start, and
    the returning minutes (EWMA blend) in total / in his groups / of expected starters."""
    from courtside.proj import minutes as M
    ab = np.asarray(absent, bool)
    n = len(panel)
    ew = 0.5 * panel["ew_f"].to_numpy(float) + 0.5 * panel["ew_s"].to_numpy(float)
    codes, _ = pd.factorize(panel["key"].to_numpy())
    g1 = panel["g1"].to_numpy(int)
    g2 = panel["g2"].to_numpy(int)
    mn = panel["missed_now"].to_numpy(float)
    gsr = panel["gsr_now"].to_numpy(float)
    back = (~ab) & (mn >= 1)
    recent = (~ab) & (mn == 0) & (gsr <= 2)
    order = panel["st10"].to_numpy(float) * 100.0 + ew
    r_all = pd.Series(order).groupby(codes).rank(ascending=False, method="first").to_numpy()
    st = r_all <= 5
    nk = codes.max() + 1 if n else 1

    def tot(x):
        return np.bincount(codes, x, nk)[codes]

    def in_grp(x):
        return sum(tot(np.where(g1 == g, x, 0.0)) * ((g1 == g) | (g2 == g)) for g in range(3))

    out = pd.DataFrame(index=panel.index)
    out["ret_tot"] = tot(np.where(back, ew, 0.0)) - np.where(back, ew, 0.0)
    out["ret_same"] = in_grp(np.where(back, ew, 0.0)) - np.where(back, ew, 0.0)
    out["ret_st"] = tot(np.where(back & st, ew, 0.0)) - np.where(back & st, ew, 0.0)
    out["rec_same"] = in_grp(np.where(recent, ew, 0.0)) - np.where(recent, ew, 0.0)
    own = (mn >= 1) | (gsr <= 2)
    newt = (panel["new_team"].to_numpy(float) > 0) if "new_team" in panel else np.zeros(n, bool)
    news = panel["new_season"].to_numpy(float) > 0
    why = np.full(n, "other", dtype=object)
    why[(out["rec_same"].to_numpy() >= 10)] = "teammate back <= 2 games"
    why[(out["ret_same"].to_numpy() >= 10)] = "teammate back today"
    why[(out["ret_st"].to_numpy() >= 20)] = "starter back today"
    why[newt | news] = "new season / team"
    why[own] = "own return / ramp"
    out["why"] = why
    return out


def sim_minutes(r: pd.DataFrame, prm, spread: pd.Series, sd: float, sims: int, seed: int = 11
                ) -> pd.DataFrame:
    """Per row: E[min] (unconditional, sims), P(play) in the sims, 10 / 90 % quantiles, the
    randomised PIT and the CRPS of the actual box minutes, from `sample_minutes` draws."""
    from courtside.proj import minutes as M
    from courtside.proj.metrics import crps_samples_matrix, pit_randomized
    rng = np.random.default_rng(seed)
    out = {k: np.full(len(r), np.nan) for k in ("e_sim", "p_sim", "q10", "q90", "pit", "crps",
                                                "q10c", "q90c", "pitc")}
    y_all = r["act_min"].fillna(0.0).to_numpy(float)
    for key, idx in r.groupby("key", sort=False).indices.items():
        s = spread.get(key, 0.0)
        s = 0.0 if not np.isfinite(s) else float(s)
        margins = s + sd * rng.standard_normal(sims)
        n_ot = (np.abs(margins) < 1.0).astype(float)
        d = r.iloc[idx]
        proj = d[["p_play", "pf36", "reserve"]].assign(mean=d["mean0"].to_numpy())
        x = M.sample_minutes(proj, prm, margins, rng, n_ot=n_ot)
        y = y_all[idx]
        out["e_sim"][idx] = x.mean(0)
        out["p_sim"][idx] = (x > 0).mean(0)
        out["q10"][idx], out["q90"][idx] = np.quantile(x, [0.1, 0.9], axis=0)
        out["pit"][idx] = pit_randomized(x, y, rng.random(len(idx)))
        out["crps"][idx] = crps_samples_matrix(x[:1000], y)
        for j, i in enumerate(idx):
            xi = x[:, j][x[:, j] > 0]
            if y[j] > 0 and len(xi) >= 50:
                out["q10c"][i], out["q90c"][i] = np.quantile(xi, [0.1, 0.9])
                out["pitc"][i] = float((xi < y[j]).mean() + rng.random() * (xi == y[j]).mean())
    return pd.DataFrame(out, index=r.index)


def cmd_diag(season: str, params: str | None, tag: str, sims: int) -> None:
    from courtside.proj import minutes as M
    from courtside.proj import minutes_eval as ME
    t0 = time.time()
    conn = _conn()
    prm, sd, src = load_params(season, params)
    print(f"[mf] diag {season}: params {src} (minfix {bool(getattr(prm, 'minfix', None))}), "
          f"sd_margin {sd:.2f}", flush=True)
    s0 = int(season[:4])
    h = M.load_history(conn, since=f"{s0 - 7}-{(s0 - 6) % 100:02d}")
    if M.uses_preseason(prm):
        M.attach_preseason(h, conn)
    ctx = ME.prepare_season(conn, h, season)
    panel = ctx["panel"]
    print(f"[mf] panels ready ({time.time() - t0:.0f}s): test rows {len(panel)}", flush=True)
    r = M.run_projection(panel, prm, shares=ctx["shares"])
    r["exp_min"] = ME.integrate(r, prm, ctx["spread"])
    ab = r["absent"].to_numpy(bool)
    sm = sim_minutes(r, prm, ctx["spread"], sd, sims)
    rs = reasons(panel, ab)
    keep = ["key", "game_id", "team", "date", "season", "player_id", "roster", "absent",
            "played", "act_min", "act_min_r", "act_started", "base", "adj", "lift", "p_play",
            "mean0", "exp_min", "reserve", "promoted", "exp_start", "missed_now", "gsr_now",
            "new_season", "ew_f", "ew_s", "last", "st10", "app10", "n_gp", "tgs", "g1", "g2"]
    keep = [c for c in keep if c in r]
    d = pd.concat([r[keep].reset_index(drop=True), sm.reset_index(drop=True),
                   rs.reset_index(drop=True)], axis=1)
    d["l10"] = last10_minutes(h, panel)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"diag_{season}_{tag}.parquet"
    d.drop(columns=[c for c in ("promoted", "exp_start") if c in d]).to_parquet(f)
    print(f"[mf] wrote {f} ({len(d)} rows, {time.time() - t0:.0f}s)", flush=True)
    print(summary(d), flush=True)


def cmd_fit(season: str, tag: str, minfix: bool, pre: bool, opts: dict | None = None) -> None:
    """Minutes params of S (5 seasons < S, the pipeline's recipe: oracle training panel, pair
    shares of it, v1, 1 LightGBM thread) with or without the minutes fix -> OUT/mparams_<S>_<T>."""
    from courtside.proj import minutes as M
    M.GBM_PARAMS["num_threads"] = 1
    t0 = time.time()
    conn = _conn()
    s0 = int(season[:4])
    h = M.load_history(conn, since=f"{s0 - 7}-{(s0 - 6) % 100:02d}")
    if pre:
        M.attach_preseason(h, conn)
    train = sorted(x for x in h.tg["season"].unique() if x < season)[-5:]
    panel = M.oracle_panel(h, M.eval_targets(h, train))
    kw = {"minfix": (opts or True)} if minfix else {}
    if pre:
        kw["features"] = M.preseason_features()
    mp = M.fit(h, season, window=5, log=lambda m: print(m, flush=True), panel=panel,
               shares=M.PairShares(M.pair_events(panel)), version="v1", v1_kw=kw,
               debut_reserve=pre)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"mparams_{season}_{tag}.json"
    f.write_text(mp.to_json())
    print(f"[mf] fit {season} {tag}: {f} ({time.time() - t0:.0f}s)", flush=True)


# ============================================================================== tables

def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d[~d["absent"].astype(bool)].copy()
    d["y"] = d["act_min"].fillna(0.0)
    d["pl"] = (d["y"] > 0).astype(float)
    d["P"] = d["mean0"]
    d["tier4"] = "bench"
    for name, lo, hi in TIERS:
        d.loc[(d["P"] >= lo) & (d["P"] < hi), "tier4"] = name
    v = d["P"] / d["l10"].where(d["l10"] >= 1)
    d["dem"] = np.where(v < 0.9, "demoted", np.where(v > 1.1, "promoted", "steady"))
    return d


def cl_mean(z, cl) -> tuple[float, float]:
    z = np.asarray(z, float)
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    m = float(z.mean())
    s = pd.Series(z - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    return m, float(np.sqrt(g / max(g - 1, 1) * (s ** 2).sum()) / n)


def summary(d: pd.DataFrame) -> str:
    d = prep(d)
    L = []
    g = d.groupby(["tier4", "dem"])
    t = pd.DataFrame({"n": g.size(), "pp": g["p_play"].mean(), "p_sim": g["p_sim"].mean(),
                      "played": g["pl"].mean(), "E[min]": g["exp_min"].mean(),
                      "e_sim": g["e_sim"].mean(), "min": g["y"].mean(),
                      "cov80": g.apply(lambda q: ((q.q10 <= q.y) & (q.y <= q.q90)).mean()),
                      "cov80|pl": g.apply(lambda q: ((q.q10c <= q.y) & (q.y <= q.q90c))[q.pl > 0].mean()),
                      "crps": g["crps"].mean()})
    t["dP"] = t["p_sim"] - t["played"]
    t["dE"] = t["e_sim"] - t["min"]
    L.append("by tier x plan vs last-10 (all available candidates)\n" + t.round(3).to_string())
    b = d[(d["tier4"] == "bench") & (d["dem"] == "demoted")]
    g = b.groupby("why")
    t = pd.DataFrame({"n": g.size(), "p_sim": g["p_sim"].mean(), "played": g["pl"].mean(),
                      "e_sim": g["e_sim"].mean(), "min": g["y"].mean(),
                      "P": g["P"].mean(), "min|pl": g.apply(lambda q: q.y[q.pl > 0].mean()),
                      "base": g["base"].mean(), "adj": g["adj"].mean(), "l10": g["l10"].mean()})
    t["dP"] = t["p_sim"] - t["played"]
    t["dE"] = t["e_sim"] - t["min"]
    L.append("bench demoted by reason\n" + t.round(3).to_string())
    pit = d[d["exp_min"] >= 10]
    for name, q in pit.groupby("tier4"):
        h = np.histogram(q["pit"], bins=10, range=(0, 1))[0] / max(len(q), 1)
        hc = np.histogram(q["pitc"].dropna(), bins=10, range=(0, 1))[0] / max(q["pitc"].notna().sum(), 1)
        L.append(f"PIT {name} (E[min]>=10, n {len(q)}): uncond {np.round(h, 3).tolist()} | "
                 f"given played {np.round(hc, 3).tolist()}")
    return "\n\n".join(L)


def cmd_report(fa: str, fb: str) -> None:
    a, b = pd.read_parquet(fa), pd.read_parquet(fb)
    k = ["key", "player_id"]
    a, b = prep(a), prep(b)
    m = a.merge(b[k + ["p_sim", "e_sim", "q10", "q90", "pit", "crps", "mean0", "q10c", "q90c"]],
                on=k, suffixes=("_a", "_b"))
    print(f"rows paired {len(m)} (A {len(a)}, B {len(b)})")
    pops = {"all available": m, "E[min] A >= 10": m[m["e_sim_a"] >= 10]}
    for name, q in pops.items():
        print(f"\n## {name}")
        rows = []
        for lab, sub in [("all", q)] + [(t, q[q["tier4"] == t]) for t, _, _ in TIERS] + \
                [("bench demoted", q[(q["tier4"] == "bench") & (q["dem"] == "demoted")]),
                 ("bench promoted", q[(q["tier4"] == "bench") & (q["dem"] == "promoted")]),
                 ("rotation demoted", q[(q["tier4"] == "rotation") & (q["dem"] == "demoted")])]:
            if len(sub) < 20:
                continue
            cl = sub["date"].to_numpy()
            y, pl = sub["y"].to_numpy(float), sub["pl"].to_numpy(float)
            r = {"rows": lab, "n": len(sub)}
            for met, fa_, fb_ in (
                    ("MAE", np.abs(sub["e_sim_a"] - y), np.abs(sub["e_sim_b"] - y)),
                    ("CRPS", sub["crps_a"], sub["crps_b"]),
                    ("Brier", (sub["p_sim_a"] - pl) ** 2, (sub["p_sim_b"] - pl) ** 2),
                    ("cov80", ((sub.q10_a <= y) & (y <= sub.q90_a)).astype(float),
                     ((sub.q10_b <= y) & (y <= sub.q90_b)).astype(float))):
                da, _ = cl_mean(np.asarray(fa_, float), cl)
                dd, se = cl_mean(np.asarray(fb_, float) - np.asarray(fa_, float), cl)
                r[f"{met} A"] = da
                r[f"{met} d"] = dd
                r[f"{met} z"] = dd / se if se > 0 else np.nan
            r["E bias A"] = float((sub["e_sim_a"] - y).mean())
            r["E bias B"] = float((sub["e_sim_b"] - y).mean())
            r["P bias A"] = float((sub["p_sim_a"] - pl).mean())
            r["P bias B"] = float((sub["p_sim_b"] - pl).mean())
            rows.append(r)
        print(pd.DataFrame(rows).set_index("rows").round(4).to_string())


# ============================================================================== debut reserve

def new_player_stats(h, rows: pd.DataFrame) -> pd.DataFrame:
    """Per (team, date) row: the camp's new players (no box row strictly before the date):
    d_sum = their preseason minutes per team preseason game (the debut reserve's input,
    `minutes._new_player_minutes`), d_top1 / d_top2 = the largest one / two players' share of it,
    n15 = how many averaged >= 15 per team game."""
    pre = h.pre
    first = h.hist.groupby("player_id")["date"].min()
    out = []
    for (season, team), g in rows.groupby(["season", "team"]):
        tm = pre.teams[(pre.teams["season"] == season) & (pre.teams["team"] == team)]
        pl = pre.players[(pre.players["season"] == season) & (pre.players["team"] == team)
                         & (pre.players["min"] > 0)]
        for d in g["date"].unique():
            t_ = tm[tm["date"] < d]
            q = pl[pl["date"] < d]
            r = {"season": season, "team": team, "date": d, "d_sum": np.nan, "d_top1": np.nan,
                 "d_top2": np.nan, "n15": np.nan}
            if len(t_):
                m = q.groupby("player_id")["min"].sum() / len(t_)
                fd = first.reindex(m.index).to_numpy(float)
                m = m[~(fd < d)].sort_values(ascending=False)
                r.update(d_sum=float(m.sum()), d_top1=float(m.iloc[:1].sum()),
                         d_top2=float(m.iloc[:2].sum()), n15=float((m >= 15).sum()))
            out.append(r)
    return pd.DataFrame(out)


def cmd_debut(*seasons) -> None:
    """Debut minutes per team-game (players with no box row before the date) vs the camp's new
    players' preseason minutes, by team games into the season, every season (2026-27: the games
    played so far), and leave-one-season-out fits of candidate debut-reserve designs."""
    from courtside.proj import minutes as M
    conn = _conn()
    seasons = seasons or ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                          "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
    s0 = int(min(seasons)[:4]) - 2
    h = M.load_history(conn, since=f"{s0}-{(s0 + 1) % 100:02d}")
    M.attach_preseason(h, conn)
    rows = []
    for s in seasons:
        t = M.eval_targets(h, [s])
        t = t[t["date"] <= t.groupby("team")["date"].transform(lambda x: np.sort(x)[min(14, len(x) - 1)])]
        if not len(t):
            continue
        p = M.pregame_panel(h, t)
        ew = 0.5 * p["ew_f"] + 0.5 * p["ew_s"]
        ro = p["roster"] == 1
        p = p.assign(_ew=np.where(ro, ew, 0.0), _r15=(ro & (ew >= 15)).astype(float))
        agg = p.groupby("key").agg(s_ew=("_ew", "sum"), n_rot=("_r15", "sum"))
        kk = p.drop_duplicates("key")[["key", "team", "date", "season", "tgs", "pre_new_tg"]]
        kk = kk.merge(agg, left_on="key", right_index=True, how="left")
        deb = M.debut_minutes(h, p)
        rows.append(kk.assign(debut=deb.reindex(kk["key"]).to_numpy(float)))
    d = pd.concat(rows, ignore_index=True)
    d = d.merge(new_player_stats(h, d), on=["season", "team", "date"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "debut_rows.parquet")
    print(debut_report(d))


def debut_report(d: pd.DataFrame) -> str:
    from courtside.proj import minutes as M
    d = d.copy()
    d["stage"] = pd.cut(d["tgs"], [-1, 0, 4, 14], labels=["0", "1-4", "5-14"])
    L = [d.groupby(["season", "stage"], observed=True).agg(
        n=("debut", "size"), debut=("debut", "mean"), d_sum=("d_sum", "mean"),
        d_max=("d_sum", "max"), top2=("d_top2", "mean"), n15=("n15", "mean"),
        corr=("debut", lambda x: np.corrcoef(x, d.loc[x.index, "d_sum"].fillna(0))[0, 1]
              if len(x) > 3 else np.nan)).round(2).to_string()]
    seasons = sorted(d["season"].unique())
    designs = {
        "constant by stage": lambda tr, te: _stage_fit(tr, te, None),
        "current (stage x D, cap p95)": lambda tr, te: _stage_fit(tr, te, "d_sum", q=0.95),
        "stage x D, cap p50": lambda tr, te: _stage_fit(tr, te, "d_sum", q=0.50),
        "stage x top-2 new players": lambda tr, te: _stage_fit(tr, te, "d_top2", q=1.0),
        "stage x top-1 new player": lambda tr, te: _stage_fit(tr, te, "d_top1", q=1.0),
        "stage x n15": lambda tr, te: _stage_fit(tr, te, "n15", q=1.0),
        "stage x D (p95) + stage x slack": lambda tr, te: _stage_fit(tr, te, "d_sum", q=0.95,
                                                                        extra="slack"),
        "stage x top-2 + stage x slack": lambda tr, te: _stage_fit(tr, te, "d_top2", q=1.0,
                                                                   extra="slack"),
        "stage x min(D, p95) x slack share": lambda tr, te: _stage_fit(tr, te, "d_x_slack",
                                                                       q=0.95),
    }
    d["slack"] = np.clip(240.0 - d["s_ew"].fillna(240.0), 0.0, 120.0)
    d["d_x_slack"] = d["d_sum"].fillna(0) * d["slack"] / 60.0
    res = []
    for s in seasons[3:]:
        tr = d[(d["season"] < s) & (d["season"] >= seasons[max(0, seasons.index(s) - 5)])]
        te = d[d["season"] == s]
        for name, fn in designs.items():
            pr = fn(tr, te)
            for st in ("0", "1-4", "5-14"):
                m = (te["stage"] == st).to_numpy()
                if m.any():
                    e = pr[m] - te["debut"].to_numpy(float)[m]
                    res.append({"season": s, "design": name, "stage": st, "n": int(m.sum()),
                                "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                                "pred": float(pr[m].mean())})
    r = pd.DataFrame(res)
    L.append(r.pivot_table(index=["design", "stage"], columns="season", values="mae").round(2)
             .to_string())
    L.append(r.pivot_table(index=["design", "stage"], columns="season", values="bias").round(2)
             .to_string())
    return "\n\n".join(L)


def _stage_fit(tr: pd.DataFrame, te: pd.DataFrame, col: str | None, q: float = 0.95,
               extra: str | None = None) -> np.ndarray:
    """OLS of debut minutes on stage one-hots (+ stage x min(col, cap)), cap = the training
    game-0 quantile q; prediction clipped to [0, DEBUT_MAX]."""
    from courtside.proj import minutes as M
    def X(df, cap):
        b = np.clip(np.searchsorted(np.asarray(M.DEBUT_BINS), df["tgs"].to_numpy(float),
                                    side="right") - 1, 0, len(M.DEBUT_BINS) - 1)
        oh = np.eye(len(M.DEBUT_BINS))[b]
        if col is None:
            return oh
        v = np.minimum(np.nan_to_num(df[col].to_numpy(float)), cap)
        cols = [oh, oh * v[:, None]]
        if extra:
            cols.append(oh * np.nan_to_num(df[extra].to_numpy(float))[:, None])
        return np.column_stack(cols)
    cap = 1e9
    if col is not None:
        v0 = tr.loc[(tr["tgs"] == 0), col].dropna().to_numpy(float)
        cap = float(np.quantile(v0, q)) if len(v0) >= 20 else 1e9
    A = X(tr, cap)
    beta = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ tr["debut"].to_numpy(float))
    return np.clip(X(te, cap) @ beta, 0.0, M.DEBUT_MAX)


# ============================================================================== DK harness

_LAST: dict = {}
MIN_COLS = ("min_mean", "min_p", "min_q10", "min_q90", "min_pit", "min_crps")


def add_minutes_stats(players: pd.DataFrame, res, day: str) -> pd.DataFrame:
    """Per projected player of a harness date: the sims' minutes mean, P(minutes > 0), 10 / 90 %
    quantiles, randomised PIT and CRPS of the actual box minutes (the harness rows' `minutes`)."""
    from courtside.proj import pipeline as P
    from courtside.proj.metrics import crps_samples_matrix, pit_randomized
    rng = np.random.default_rng(P.stable_seed(day, "mf-minutes-pit", 1))
    y_of = {int(p): float(m) for p, m in zip(players["player_id"], players["minutes"])
            if p is not None and np.isfinite(p)}
    st = {}
    for g in res.sims:
        pids = np.asarray(g.player_id, np.int64)
        ys = np.array([y_of.get(int(p), np.nan) for p in pids], float)
        ok = np.isfinite(ys)
        if not ok.any():
            continue
        X = g.minutes[:, ok].astype(float)
        yy = ys[ok]
        q = np.quantile(X, [0.1, 0.9], axis=0)
        pit = pit_randomized(X, yy, rng.random(len(yy)))
        crps = crps_samples_matrix(X[:2000], yy)
        for j, pid in enumerate(pids[ok]):
            st[int(pid)] = (float(X[:, j].mean()), float((X[:, j] > 0).mean()), float(q[0, j]),
                            float(q[1, j]), float(pit[j]), float(crps[j]))
    out = players.copy()
    vals = [st.get(int(p), (np.nan,) * 6) if p is not None and np.isfinite(p) else (np.nan,) * 6
            for p in out["player_id"]]
    for k, c in enumerate(MIN_COLS):
        out[c] = [v[k] for v in vals]
    return out


def _patch_capture() -> None:
    """Keep each harness date's DayResult so the minutes columns can be added to its rows
    (proj/backtest.py is used unchanged; this only wraps it in the driver process)."""
    from courtside.proj import backtest as B
    from courtside.proj import pipeline as P
    if getattr(B.SeasonRun.run_date, "_mf", False):
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
            players = add_minutes_stats(players, res, day)
        return players, lineups, meta

    run_date._mf = True
    P.project_historical, B.live_like_day, B.SeasonRun.run_date = ph2, ll2, run_date


def cmd_harness(season: str, protocol: str) -> None:
    """T2.1 harness (avail) or the T2.4 slice (live / live10, P(play) cells leave-one-month-out)
    for the arm of the environment switches, 4,000 sims, with the minutes columns ->
    OUT/rows_<protocol>_<S>_<MODEL_VERSION>.parquet."""
    from courtside import db
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    _patch_capture()
    conn = db.connect()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    print(f"[mf] harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
          f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}", flush=True)


def _last10(conn, rows: pd.DataFrame) -> np.ndarray:
    """Mean box minutes of each row's last 10 appearances strictly before its date."""
    g = pd.read_sql_query("SELECT player_id, game_date, min FROM player_games WHERE min > 0",
                          conn)
    g = g.sort_values(["player_id", "game_date"], kind="mergesort")
    g["l10"] = g.groupby("player_id")["min"].transform(lambda x: x.rolling(10, 1).mean())
    g["d"] = pd.to_datetime(g["game_date"]).values.astype("datetime64[D]").astype(np.int64) + 1
    q = pd.DataFrame({"player_id": pd.to_numeric(rows["player_id"]).fillna(-1).astype(np.int64),
                      "d": pd.to_datetime(rows["date"]).values.astype("datetime64[D]")
                      .astype(np.int64), "i": np.arange(len(rows))}).sort_values("d")
    g = g[["player_id", "d", "l10"]].astype({"player_id": np.int64}).sort_values("d")
    m = pd.merge_asof(q, g, on="d", by="player_id", direction="backward")
    return m.sort_values("i")["l10"].to_numpy(float)


DK_KEY = ["date", "player_id", "game_id"]


def dk_pair(off: pd.DataFrame, on: pd.DataFrame, l10: np.ndarray | None = None) -> pd.DataFrame:
    a = off.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"]).copy()
    a["l10"] = l10 if l10 is not None and len(l10) == len(a) else np.nan
    cols = ["mean", "q10", "q90", "crps", "pit", "exp_min", "p_play"] + \
        [c for c in MIN_COLS if c in on]
    b = on.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    d = a.merge(b[["date", "player_id"] + cols], on=["date", "player_id"], suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool)].reset_index(drop=True)
    d["pop"] = (d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)
    d["min_c"] = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    d["tier4"] = "bench"
    for name, lo, hi in TIERS:
        d.loc[(d["min_c"] >= lo) & (d["min_c"] < hi), "tier4"] = name
    v = d["min_c"] / d["l10"].where(d["l10"] >= 1)
    d["dem"] = np.where(v < 0.9, "demoted", np.where(v > 1.1, "promoted", "steady"))
    return d


def dk_metrics(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(float)
    m = d["minutes"].fillna(0).to_numpy(float)
    pl = (m > 0).astype(float)
    cl = d["date"].to_numpy()
    out = {"n": int(len(d))}
    defs = {"dk_mae": lambda a: np.abs(y - d[f"mean_{a}"]),
            "dk_crps": lambda a: d[f"crps_{a}"],
            "dk_bias": lambda a: d[f"mean_{a}"] - y,
            "dk_cov80": lambda a: ((d[f"q10_{a}"] <= y) & (y <= d[f"q90_{a}"])).astype(float),
            "min_mae": lambda a: np.abs(d[f"exp_min_{a}"] - m),
            "min_bias": lambda a: d[f"exp_min_{a}"] - m,
            "p_bias": lambda a: d[f"p_play_{a}"] - pl}
    if "min_crps_off" in d and "min_crps_on" in d:
        defs.update({"min_crps": lambda a: d[f"min_crps_{a}"],
                     "min_cov80": lambda a: ((d[f"min_q10_{a}"] <= m) & (m <= d[f"min_q90_{a}"]))
                     .astype(float)})
    for k, f in defs.items():
        z0, z1 = np.asarray(f("off"), float), np.asarray(f("on"), float)
        ok = np.isfinite(z0) & np.isfinite(z1)
        if ok.sum() < 2:
            continue
        m0, _ = cl_mean(z0[ok], cl[ok])
        dd, se = cl_mean(z1[ok] - z0[ok], cl[ok])
        out[k] = {"off": m0, "on": float(z1[ok].mean()), "diff": dd, "se": se,
                  "z": dd / se if se > 0 else float("nan")}
    for a in ("off", "on"):
        h = np.histogram(np.clip(d[f"pit_{a}"].dropna(), 0, 1 - 1e-12), bins=10, range=(0, 1))[0]
        out[f"dk_pitdev_{a}"] = float(np.abs(h / max(h.sum(), 1) - 0.1).max())
        if f"min_pit_{a}" in d:
            h = np.histogram(np.clip(d[f"min_pit_{a}"].dropna(), 0, 1 - 1e-12), bins=10,
                             range=(0, 1))[0]
            out[f"min_pitdev_{a}"] = float(np.abs(h / max(h.sum(), 1) - 0.1).max())
    return out


def cmd_dk(off: str, on: str, label: str) -> None:
    """Paired ON - OFF on identical harness rows (E[min] >= 10 in either arm, DNP = 0): DK MAE /
    CRPS / bias / 80 % coverage / PIT, minutes MAE / CRPS / coverage / PIT, P(play) bias,
    overall, by tier (E[min | plays] OFF) and for the planned-demotion rows; SE by date."""
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    a = a.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"]).reset_index(drop=True)
    l10 = _last10(_conn(), a)
    d = dk_pair(a, b, l10)
    tg = pd.read_sql_query("SELECT game_id, team, game_date FROM team_games WHERE "
                           "season_type = 'Regular Season'", _conn())
    tg = tg.sort_values("game_date")
    tg["tgn"] = tg.groupby([tg["team"], tg["game_id"].str[3:5]]).cumcount()
    d = d.merge(tg[["game_id", "team", "tgn"]], on=["game_id", "team"], how="left")
    q = d[d["pop"]]
    res = {"all": dk_metrics(q)}
    for t, _, _ in TIERS:
        res[t] = dk_metrics(q[q["tier4"] == t])
    for t in ("bench", "rotation"):
        for dm in ("demoted", "promoted"):
            x = q[(q["tier4"] == t) & (q["dem"] == dm)]
            if len(x) > 20:
                res[f"{t} {dm}"] = dk_metrics(x)
    x = q[q["tgn"] < 15]
    if len(x) > 50:
        res["first 15 team games"] = dk_metrics(x)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label}: {off} -> {on} (rows {len(q)})")
    keys = ["dk_mae", "dk_crps", "dk_bias", "dk_cov80", "min_mae", "min_crps", "min_cov80",
            "min_bias", "p_bias"]
    rows = []
    for name, r in res.items():
        row = {"rows": name, "n": r["n"]}
        for k in keys:
            if k in r:
                row[k] = f"{r[k]['off']:.3f} {r[k]['diff']:+.4f} ({r[k]['z']:+.1f})"
        row["pit dk"] = f"{r.get('dk_pitdev_off', np.nan):.4f}>{r.get('dk_pitdev_on', np.nan):.4f}"
        if "min_pitdev_off" in r:
            row["pit min"] = f"{r['min_pitdev_off']:.4f}>{r['min_pitdev_on']:.4f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("rows").to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("diag")
    a.add_argument("season")
    a.add_argument("--params")
    a.add_argument("--tag", default="off")
    a.add_argument("--sims", type=int, default=2000)
    ft = sub.add_parser("fit")
    ft.add_argument("season")
    ft.add_argument("--tag", required=True)
    ft.add_argument("--minfix", action="store_true")
    ft.add_argument("--pre", action="store_true")
    ft.add_argument("--pplay")
    ft.add_argument("--noise")
    r = sub.add_parser("report")
    r.add_argument("a")
    r.add_argument("b")
    hh = sub.add_parser("harness")
    hh.add_argument("season")
    hh.add_argument("protocol", choices=("avail", "live", "live10"))
    dk = sub.add_parser("dk")
    dk.add_argument("off")
    dk.add_argument("on")
    dk.add_argument("--label", default="cmp")
    dd = sub.add_parser("debut")
    dd.add_argument("seasons", nargs="*")
    x = ap.parse_args(argv)
    if x.cmd == "diag":
        cmd_diag(x.season, x.params, x.tag, x.sims)
    elif x.cmd == "fit":
        opts = {k: v for k, v in (("pplay", x.pplay), ("noise", x.noise)) if v}
        cmd_fit(x.season, x.tag, x.minfix, x.pre, opts or None)
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    elif x.cmd == "dk":
        cmd_dk(x.off, x.on, x.label)
    elif x.cmd == "report":
        cmd_report(x.a, x.b)
    else:
        cmd_debut(*x.seasons)


if __name__ == "__main__":
    main()
