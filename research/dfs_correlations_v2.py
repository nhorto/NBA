"""NBA DFS correlation study v2 (T2.7): causal residuals vs OUR pregame projection.

v1 (`dfs_correlations.py`) took residuals against each player's own in-season mean, which uses
the whole season (hindsight). Here the residual is actual DK - the pregame projection mean from
`pipeline.project_historical(protocol="avail")` (bundle fit on seasons < S, closing-line
environment, availability oracle), and the tier of a player is the copula's own role tier on the
projection (`copula.role_tiers`), so nothing at or after the game date enters either.

Steps (research/ is read-only on the DB; extracts are cached under $COURTSIDE_DATA/proj/):

  extract <season ...> [--every N] [--sims N]   project_historical over the regular-season dates
  analyze [--out PATH] [--write-params]         all tables below; writes var/proj/corr_<S>.json
  (default: analyze)

Env: COURTSIDE_DB (read-only snapshot, default <repo>/var/courtside.db), COURTSIDE_DATA.
Residual modes (see proj/corr.py): U unconditional, P given actual team points, PM given team
points and actual minutes (the T1.4 `conditional_frame` construction, the copula's input).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj import corr as K

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(11)
NBOOT = 200
TIER_NAMES = K.TIERS


def db_conn() -> sqlite3.Connection:
    path = os.environ.get("COURTSIDE_DB") or str(config.REPO_ROOT / "var" / "courtside.db")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.astype(object).iterrows():
        vals = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                             for v in r.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def rci(t: pd.DataFrame, nd=3) -> pd.DataFrame:
    """Compact: r [lo, hi], pairs."""
    out = pd.DataFrame(index=t.index)
    out["r"] = t["r"].map(lambda v: f"{v:+.{nd}f}")
    if "lo" in t:
        out["95% CI"] = [f"[{lo:+.{nd}f},{hi:+.{nd}f}]" for lo, hi in zip(t["lo"], t["hi"])]
    out["pairs"] = t["n"].map(lambda v: f"{int(v):,}")
    return out


# V1 numbers (research/dfs_correlations_output.txt; residual vs in-season own mean, 2015-16..)
V1_TEAM_TIER = {"star|star": 0.048, "star|role": 0.029, "star|bench": -0.034, "role|role": 0.034,
                "role|bench": 0.022, "bench|bench": 0.089}
V1_OPP_TIER = {"star|star": 0.053, "star|role": 0.021, "star|bench": -0.006, "role|role": 0.003,
               "role|bench": 0.002, "bench|bench": 0.024}
V1_TEAM_POS = {"C|C": -0.088, "F|C": 0.012, "G|C": 0.043, "F|F": 0.026, "G|F": 0.047, "G|G": 0.026}
V1_CV = {5: 0.79, 15: 0.58, 25: 0.41, 35: 0.31, 45: 0.26, 55: 0.24}
V1_SEASON = {"2016-17": 0.020, "2017-18": 0.018, "2018-19": 0.024, "2019-20": 0.020,
             "2020-21": 0.045, "2021-22": 0.046, "2022-23": 0.037, "2023-24": 0.040,
             "2024-25": 0.042, "2025-26": 0.059}


# ============================================================================== helpers

def prep(f: pd.DataFrame, mode: str, b=None, cvinfo=None):
    r, b = K.residuals(f, mode, b)
    z, cv = K.standardise(r, cvinfo)
    return z, b, cv


def pstats(z: pd.DataFrame, col="z") -> K.PairStats:
    return K.PairStats(z["game_id"], z["team"], z["grp"], z[col])


def cochran_q(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    m = (w * est).sum() / w.sum()
    return float((w * (est - m) ** 2).sum()), len(est) - 1, float(m)


def chi2_sf(x, df):
    from scipy import stats
    return float(stats.chi2.sf(x, df))


# ============================================================================== sections

def sec_data(conn, f, seasons):
    print("\n## 0. Data: causal residual frame")
    q = ",".join("?" * len(seasons))
    n_all = pd.read_sql_query(
        f"select season, count(*) n from player_games where season_type='Regular Season' and "
        f"min>0 and season in ({q}) group by season", conn, params=seasons).set_index("season")["n"]
    ex_meta = []
    for s in seasons:
        m = pd.read_pickle(K.extract_path(s))
        ex_meta.append((s, len(m["done"]), m.get("every"), m.get("n_sims")))
    t = pd.DataFrame(ex_meta, columns=["season", "dates", "every", "n_sims"]).set_index("season")
    t["player-games in frame"] = f.groupby("season").size()
    t["played (all)"] = n_all
    t["coverage"] = t["player-games in frame"] / t["played (all)"]
    print(md(t))
    print(f"\nrows {len(f):,}; games {f.game_id.nunique():,}; mean projected DK {f.mu.mean():.2f}, "
          f"mean actual {f.dk.mean():.2f}. Rows: players with p_play > 0 in the projection who "
          "played and have mu > 0.5 (coverage is lower than 100% by the dates skipped, "
          "if any, and by deep-bench players the projection does not carry).")
    print("\nPosition coverage (G/F/C/unknown):", f["pos"].value_counts(normalize=True)
          .sort_index().round(3).to_dict())
    t2 = f.groupby("tier").agg(n=("dk", "size"), mu=("mu", "mean"), dk=("dk", "mean"),
                               min_proj=("min_proj", "mean"), min_act=("min", "mean"))
    t2.index = [TIER_NAMES[i] for i in t2.index]
    t2["bias (act-proj)"] = t2["dk"] - t2["mu"]
    print("\nProjection tier summary (copula role tiers):"); print(md(t2, "{:.2f}"))
    print("\nBias (mean actual DK - projection) by season x tier; RMSE of residual by tier:")
    b = f.assign(res=f.dk - f.mu).groupby(["season", "tier"]).res.mean().unstack()
    b.columns = [TIER_NAMES[i] for i in b.columns]
    print(md(b, "{:+.2f}"))
    rm = f.assign(res=f.dk - f.mu).groupby("tier").res.apply(lambda x: np.sqrt((x ** 2).mean()))
    print("RMSE:", {TIER_NAMES[i]: round(float(v), 2) for i, v in rm.items()})


def sec_team_opp(f):
    print("\n## 1. Teammate / opponent residual correlation, residual vs OUR projection")
    print("Residual centred by (season, tier). 'raw' = DK points; 'z' = divided by CV(level) x "
          "level (what the copula sees). All seasons pooled; 95% CI = game-cluster bootstrap.")
    tabs = {}
    for mode in K.MODES:
        z, b, cv = prep(f, mode)
        tabs[mode] = (z, b, cv)
    print(f"\nElasticity of the relative residual to (team pts / implied pts - 1), by tier "
          f"(star, role, bench): {np.round(tabs['PM'][1], 3).tolist()}")
    for kind, v1 in (("team", V1_TEAM_TIER), ("opp", V1_OPP_TIER)):
        nm = "Teammates" if kind == "team" else "Opponents"
        print(f"\n### {nm}: all pairs")
        rows = []
        for mode in K.MODES:
            z = tabs[mode][0]
            for col in ("rc", "z"):
                t = pstats(z, col).table(kind, "all", nboot=NBOOT)
                rows.append((f"{mode} {'raw' if col == 'rc' else 'z'}", t.r.iloc[0],
                             f"[{t.lo.iloc[0]:+.3f},{t.hi.iloc[0]:+.3f}]", int(t.n.iloc[0])))
        print(md(pd.DataFrame(rows, columns=["residual", "r", "95% CI", "pairs"]).set_index("residual")))
        print("v1 (vs in-season mean): teammates 0.034, opponents 0.011")
        print(f"\n### {nm}: by role pair (tier = projection tier at the date)")
        cols = {}
        for mode in K.MODES:
            z = tabs[mode][0]
            ps_ = pstats(z, "z").table(kind, "tier", nboot=NBOOT)
            cols[f"{mode} z"] = ps_["r"]
            cols[f"{mode} z CI"] = [f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(ps_.lo, ps_.hi)]
            if mode == "U":
                cols["U raw"] = pstats(z, "rc").table(kind, "tier")["r"]
                cols["pairs"] = ps_["n"].map(lambda v: f"{int(v):,}")
        t = pd.DataFrame(cols)
        t["v1"] = pd.Series(v1)
        print(md(t[["v1", "U raw", "U z", "U z CI", "P z", "P z CI", "PM z", "PM z CI", "pairs"]]))
    # where the sign comes from: minutes and per-minute channels
    print("\n### Teammate / opponent correlation of the minutes and per-minute channels (U)")
    m = f.copy()
    m["x_min"] = m["min"] - m["min_proj"]
    m["x_pm"] = m["dk"] / m["min"] - m["mu"] / m["min_proj"]
    rows = []
    for lab, col in (("minutes (actual - projected)", "x_min"), ("DK per minute residual", "x_pm")):
        m["x"] = m[col] - m.groupby(["season", "tier"])[col].transform("mean")
        sd = m.groupby("tier")["x"].transform("std")
        m["xz"] = (m["x"] / sd).clip(-6, 6)
        ps_ = K.PairStats(m["game_id"], m["team"], m["grp"], m["xz"])
        a, o = ps_.table("team", "all", nboot=100), ps_.table("opp", "all", nboot=100)
        rows.append((lab, a.r.iloc[0], f"[{a.lo.iloc[0]:+.3f},{a.hi.iloc[0]:+.3f}]", o.r.iloc[0],
                     f"[{o.lo.iloc[0]:+.3f},{o.hi.iloc[0]:+.3f}]"))
    print(md(pd.DataFrame(rows, columns=["channel", "teammates r", "95% CI", "opponents r",
                                         "95% CI "]).set_index("channel")))
    print("v1 minutes residual (vs own season mean): teammates +0.08. The avail protocol already "
          "conditions on who sits, so the absence-driven co-movement that v1 counted is gone "
          "from the residual.")
    # position pairs
    print("\n### Teammates by position pair (G/F/C, season roster position)")
    cols = {}
    for mode in ("U", "PM"):
        z = tabs[mode][0]
        t = pstats(z, "z").table("team", "pos", nboot=NBOOT)
        cols[f"{mode} z"] = t["r"]
        cols[f"{mode} CI"] = [f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(t.lo, t.hi)]
        cols["pairs"] = t["n"].map(lambda v: f"{int(v):,}")
    t = pd.DataFrame(cols)
    t["v1 (raw)"] = pd.Series(V1_TEAM_POS)
    print(md(t[["v1 (raw)", "U z", "U CI", "PM z", "PM CI", "pairs"]]))
    print("\n### Opponents by position pair")
    cols = {}
    for mode in ("U", "PM"):
        t = pstats(tabs[mode][0], "z").table("opp", "pos", nboot=NBOOT)
        cols[f"{mode} z"] = t["r"]
        cols[f"{mode} CI"] = [f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(t.lo, t.hi)]
    print(md(pd.DataFrame(cols)))
    # cells and heterogeneity of position within tier pair
    print("\n### Role pair x position pair (teammates; cells with >= 20,000 pairs; "
          "z, mode U and PM; z-score vs the tier-pair value)")
    for mode in ("U", "PM"):
        z = tabs[mode][0]
        ps_ = pstats(z, "z")
        tc = ps_.table("team", "cell", nboot=NBOOT)
        tt = ps_.table("team", "tier", nboot=0)
        tc["parent"] = [tt.loc[K.pair_label(a.split(":")[0], b.split(":")[0]), "r"]
                        for a, b in (l.split("|") for l in tc.index)]
        tc["dev"] = tc["r"] - tc["parent"]
        tc["z"] = tc["dev"] / tc["se"].clip(lower=1e-6)
        Q = float((tc["z"] ** 2).sum())
        dfree = len(tc)
        print(f"\n{mode}: heterogeneity of the 45 cells around their tier-pair value: "
              f"Q = {Q:.1f} on ~{dfree} df (p = {chi2_sf(Q, dfree):.3f}; tier-pair value is "
              "estimated from the same data, so df is slightly over-stated)")
        big = tc[tc["n"] >= 20000].copy()
        big["abs"] = big["z"].abs()
        big = big.sort_values("abs", ascending=False).head(12)
        out = pd.DataFrame({"r": big["r"], "tier-pair r": big["parent"], "dev": big["dev"],
                            "se": big["se"], "z-score": big["z"],
                            "pairs": big["n"].map(lambda v: f"{int(v):,}")})
        print("largest deviations:"); print(md(out))
    return tabs


def sec_drift(f, seasons):
    print("\n## 2. Drift by season (residual vs projection; z; per-season CV and elasticity)")
    for mode in ("U", "PM"):
        rows = []
        for s in seasons:
            z, _, _ = prep(f[f["season"] == s], mode)
            ps_ = pstats(z)
            a = ps_.table("team", "all", nboot=100)
            o = ps_.table("opp", "all", nboot=100)
            t = ps_.table("team", "tier", nboot=0)
            rows.append((s, a.r.iloc[0], a.se.iloc[0], o.r.iloc[0], o.se.iloc[0],
                         *[t.r.get(k, np.nan) for k in ("star|star", "star|role", "role|role",
                                                       "role|bench", "bench|bench", "star|bench")]))
        d = pd.DataFrame(rows, columns=["season", "team r", "se", "opp r", "se_o", "SS", "SR", "RR",
                                        "RB", "BB", "SB"]).set_index("season")
        if mode == "U":
            d["v1 team r"] = pd.Series(V1_SEASON)
        print(f"\nMode {mode}:"); print(md(d))
        q, dfree, m = cochran_q(d["team r"], d["se"])
        yrs = np.arange(len(d))
        w = 1 / d["se"] ** 2
        slope = np.polyfit(yrs, d["team r"], 1, w=np.sqrt(w))[0]
        print(f"teammate r: weighted mean {m:.3f}, SD across seasons {d['team r'].std():.3f}, "
              f"mean SE {d['se'].mean():.3f}, Cochran Q {q:.1f} on {dfree} df "
              f"(p = {chi2_sf(q, dfree):.3g}), trend {slope:+.4f}/season")
        q, dfree, m = cochran_q(d["opp r"], d["se_o"])
        print(f"opponent r: weighted mean {m:.3f}, SD across seasons {d['opp r'].std():.4f}, "
              f"Cochran Q {q:.1f} on {dfree} df (p = {chi2_sf(q, dfree):.3g})")


def sec_folds(f, seasons, windows=(1, 2, 3), ks=(0, 500, 2000, 8000, 32000, 128000)):
    print("\n## 3. Folds: seasons < S, window length and shrinkage chosen out of sample")
    print("For each test season S and window w (last w seasons < S) the prediction is the fold's "
          "shrunk teammate table; the target is S's own raw cell r (same elasticity and CV as the "
          "fold). Metric: pair-weighted MSE over the 45 tier x position cells and, separately, "
          "6 opponent tier pairs (lower is better; sampling noise is common to all rows).")
    res = []
    for S in seasons[1:]:
        test = f[f["season"] == S]
        for w in windows:
            tr_seasons = [s for s in seasons if s < S][-w:]
            train = f[f["season"].isin(tr_seasons)]
            for mode in ("U", "PM"):
                ztr, b, cv = prep(train, mode)
                zte, _, _ = prep(test, mode, b=b, cvinfo={"cv_mu": cv["cv_mu"], "cv": cv["cv"]})
                ps_tr, ps_te = pstats(ztr), pstats(zte)
                obs = {}
                for kind in ("team", "opp"):
                    lev = "cell" if kind == "team" else "tier"
                    r, n = ps_te.estimate(kind, lev, ps_te.totals())
                    obs[kind] = (r[0], n[0])
                for ktier in ks:
                    for kcell in (ktier, 1e12):
                        t = K._table_from_stats(ps_tr, ktier, kcell, mode, b, cv)
                        pc = np.array([t.rho_team_cell[l] for l in K.CATS["cell"][0]])
                        po = np.array([t.rho_opp[l] for l in K.CATS["tier"][0]])
                        for kind, pred in (("team", pc), ("opp", po)):
                            r, n = obs[kind]
                            mse = float((n * (pred - r) ** 2).sum() / n.sum())
                            res.append((S, w, mode, ktier, "tier-only" if kcell > 1e11 else "cell",
                                        kind, mse))
    d = pd.DataFrame(res, columns=["S", "w", "mode", "k", "level", "kind", "mse"])
    d["mse"] *= 1e4
    best = {}
    for kind in ("team", "opp"):
        for mode in ("U", "PM"):
            x = d[(d["kind"] == kind) & (d["mode"] == mode)]
            if kind == "opp":
                x = x[x["level"] == "cell"]
            g = x.groupby(["w", "k", "level"]).mse.mean().reset_index()
            print(f"\n{kind} {mode}: MSE x 1e4 averaged over test seasons "
                  f"(rows: window, columns: k; 'cell' = cells shrunk to tier pair with k, "
                  f"'tier-only' = cells replaced by the shrunk tier pair)")
            pv = g.pivot_table(index=["level", "w"], columns="k", values="mse")
            print(md(pv, "{:.2f}"))
            bi = g.loc[g["mse"].idxmin()]
            best[(kind, mode)] = bi
            print(f"best: w={int(bi['w'])}, k={bi['k']:.0f}, {bi['level']} (MSE {bi['mse']:.2f})")
    return d, best


def sec_t14(conn, f, seasons):
    print("\n## 4. Consistency with T1.4 (copula v0 `fit_copula`, 2 prior seasons)")
    print("T1.4 rho_team come from the bundle (conditional on actual minutes and team points, "
          "residual vs rate x minutes x environment, z = resid / (cv mu)); PM z below is the "
          "analogous v2 estimate from the 2 prior seasons.")
    from courtside.proj import pipeline as P
    rows = []
    for S in seasons[2:]:
        path = P.cache_dir() / f"bundle_{S}.json"
        if not path.exists():
            continue
        cp = json.loads(path.read_text())["copula"]
        tr = [s for s in seasons if s < S][-2:]
        z, b, cv = prep(f[f["season"].isin(tr)], "PM")
        t = pstats(z).table("team", "tier")
        o = pstats(z).table("opp", "all")
        for k in ("star|star", "star|role", "role|role", "bench|bench"):
            rows.append((S, k, cp["rho_team"].get(k, np.nan), t.r.get(k, np.nan)))
        rows.append((S, "opp", cp["rho_opp"], o.r.iloc[0]))
    if not rows:
        print("(no bundles found)")
        return
    d = pd.DataFrame(rows, columns=["S", "pair", "T1.4 bundle", "v2 PM z"])
    print(md(d.pivot(index="S", columns="pair", values=["T1.4 bundle", "v2 PM z"]).round(3)))
    print(f"mean |difference| {np.mean(np.abs(d['T1.4 bundle'] - d['v2 PM z'])):.3f}; "
          f"correlation across rows {np.corrcoef(d['T1.4 bundle'], d['v2 PM z'])[0, 1]:.2f}")


def sec_blowout(f):
    print("\n## 5. Blowout curve: residual vs OUR projection by final margin")
    d = f.copy()
    d["res"] = d.dk - d.mu
    d["res_min"] = d["min"] - d.min_proj
    d["amargin"] = d.margin.abs()
    d["bin"] = pd.cut(d.amargin, [-1, 5, 10, 15, 20, 25, 100],
                      labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26+"])

    def tab(x, label):
        g = x.groupby("bin")
        se = g.res.std() / np.sqrt(g.size())
        t = pd.DataFrame({"n": g.size(), "mean DK resid": g.res.mean(), "95% +/-": 1.96 * se,
                          "mean min resid": g.res_min.mean(),
                          "DK/min resid": g.apply(lambda y: (y.dk / y["min"] - y.mu / y.min_proj).mean())})
        print(f"\n{label}"); print(md(t))
    tab(d[d.tier == 0], "Stars (tier at date)")
    tab(d[d.tier == 1], "Role players")
    tab(d[d.tier <= 1], "Stars + role (starters)")
    tab(d[d.tier == 2], "Bench")
    print("\nStarters (stars + role) by game winner / loser, |margin| >= 16 and baseline:")
    x = d[d.tier <= 1]
    for lab, dd in (("winning team |m|>=16", x[(x.margin > 0) & (x.amargin >= 16)]),
                    ("losing team |m|>=16", x[(x.margin < 0) & (x.amargin >= 16)]),
                    ("close |m|<=5", x[x.amargin <= 5])):
        print(f"  {lab}: n={len(dd):,}, DK resid {dd.res.mean():+.2f} +/- {1.96 * dd.res.std() / np.sqrt(len(dd)):.2f}, "
              f"minutes resid {dd.res_min.mean():+.2f}")
    print("\nSigned final margin (team perspective) x tier, mean DK residual [minutes residual]:")
    sb = pd.cut(d.margin, [-100, -21, -11, -4, 3, 10, 20, 100],
                labels=["<=-21", "-20..-11", "-10..-4", "-3..3", "4..10", "11..20", ">=21"])
    d["sb"] = sb
    out = {}
    for t in range(3):
        g = d[d.tier == t].groupby("sb")
        out[TIER_NAMES[t]] = [f"{a:+.2f} [{b:+.2f}]" for a, b in zip(g.res.mean(), g.res_min.mean())]
    out["n"] = [f"{int(v):,}" for v in d.groupby("sb").size()]
    print(md(pd.DataFrame(out, index=d.groupby("sb").size().index)))
    print("\nPM-mode check: residual given actual minutes by |margin| bin (DK per projected minute "
          "channel only), stars+role:")
    z, _, _ = prep(f, "PM")
    z["amargin"] = z.margin.abs()
    z["bin"] = pd.cut(z.amargin, [-1, 5, 10, 15, 20, 25, 100],
                      labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26+"])
    print(md(z[z.tier <= 1].groupby("bin").res.agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "PM resid"})))


def star_out_frame(conn, seasons):
    """Per (game_id, team): number of to-date stars out. Stars = top 2 by prior DK mean among
    candidates (appeared in >= 7 of last 10 team games or one of the last 3) with prior minutes
    >= 28, all from games before the date (rolling 20 played games, forward-filled to absent
    games)."""
    s0 = int(seasons[0][:4])
    first = K.season_label(s0 - 1)
    q = ",".join("?" * (len(seasons) + 1))
    pg = pd.read_sql_query(
        f"""SELECT game_id, player_id, team, season, game_date AS date, min, pts, fg3m, reb, ast,
                   stl, blk, tov FROM player_games WHERE season_type='Regular Season'
            AND min>0 AND season IN ({q})""", conn, params=[first] + list(seasons))
    pg["dk"] = K.dk_points(pg)
    pg = pg.sort_values(["player_id", "date", "game_id"])
    g = pg.groupby("player_id", sort=False)
    pg["rm"] = g["min"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    pg["rd"] = g["dk"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out = []
    for team, tp in pg.groupby("team"):
        games = tp[["game_id", "date"]].drop_duplicates().sort_values(["date", "game_id"])
        gid = games["game_id"].to_numpy()
        gidx = {g_: i for i, g_ in enumerate(gid)}
        pids = np.sort(tp["player_id"].unique())
        pidx = {p: i for i, p in enumerate(pids)}
        G, P_ = len(gid), len(pids)
        played = np.zeros((G, P_), bool)
        rm = np.full((G, P_), np.nan)
        rd = np.full((G, P_), np.nan)
        gi = tp["game_id"].map(gidx).to_numpy()
        pi = tp["player_id"].map(pidx).to_numpy()
        played[gi, pi] = True
        rm[gi, pi] = tp["rm"].to_numpy()
        rd[gi, pi] = tp["rd"].to_numpy()
        rm = pd.DataFrame(rm).ffill().shift(1).to_numpy()      # as of the previous game
        rd = pd.DataFrame(rd).ffill().shift(1).to_numpy()
        pl = pd.DataFrame(played.astype(float))
        l10 = pl.rolling(10, min_periods=1).sum().shift(1).to_numpy()
        l3 = pl.rolling(3, min_periods=1).sum().shift(1).to_numpy()
        elig = ((l10 >= 7) | (l3 >= 1)) & (rm >= K.STAR_MIN)
        score = np.where(elig, rd, -np.inf)
        score = np.where(np.isnan(score), -np.inf, score)
        top = np.argsort(-score, axis=1)[:, :2]
        n_out = np.zeros(G, int)
        for j in range(2):
            ok = np.isfinite(score[np.arange(G), top[:, j]])
            n_out += (ok & ~played[np.arange(G), top[:, j]]).astype(int)
        out.append(pd.DataFrame({"game_id": gid, "team": team, "n_star_out": n_out}))
    return pd.concat(out, ignore_index=True)


def sec_absence(conn, f, seasons):
    print("\n## 6. Star-absence residual check (the bump is already in the projection)")
    so = star_out_frame(conn, seasons)
    d = f.merge(so, on=["game_id", "team"], how="left")
    d = d[d["n_star_out"].notna()].copy()
    d["res"] = d.dk - d.mu
    d["res_min"] = d["min"] - d.min_proj
    d["so"] = np.where(d.n_star_out >= 1, "star out", "no star out")
    d["top6"] = d.groupby(["game_id", "team"]).min_proj.rank(ascending=False, method="first") <= 6
    print("A 'star' is one of the team's two highest prior-DK players (prior minutes >= 28) who "
          "appeared in >= 7 of the last 10 team games or one of the last 3, using only earlier "
          "games. Residual = actual DK - projection (projection already knows who is out).")
    rows = []
    for lab, mask in (("top-6 by projected minutes", d.top6),
                      ("stars (proj tier 0)", d.tier == 0),
                      ("role (proj tier 1)", d.tier == 1),
                      ("bench (proj tier 2)", d.tier == 2)):
        for so_lab, x in d[mask].groupby("so"):
            rows.append((lab, so_lab, len(x), x.res.mean(), 1.96 * x.res.std() / np.sqrt(len(x)),
                         x.res_min.mean(), x.mu.mean()))
    t = pd.DataFrame(rows, columns=["group", "games", "n", "mean DK resid", "95% +/-",
                                    "mean min resid", "mean projection"]).set_index("group")
    print(md(t, "{:.2f}"))
    for lab, mask in (("top-6", d.top6), ("stars", d.tier == 0)):
        a = d[mask & (d.so == "star out")].res
        b = d[mask & (d.so == "no star out")].res
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f"{lab}: residual difference (star out - not) {diff:+.2f} +/- {1.96 * se:.2f} DK "
              f"(target: within +/-0.6; v1 raw bump was +3.1 for top-6)")
    x = d[d.top6]
    print("By number of stars out (top-6 mean DK resid):",
          {int(k): (len(v), round(float(v.res.mean()), 2)) for k, v in x.groupby("n_star_out")})


def sec_cv(f, seasons):
    print("\n## 7. CV by level of the residual")
    from scipy import stats
    for mode in ("U", "PM"):
        r, b = K.residuals(f, mode)
        r["rc"] = r.res - r.groupby(["season", "tier"]).res.transform("mean")
        r["band"] = pd.cut(r.ref, [0, 10, 20, 30, 40, 50, 80])
        g = r.groupby("band")
        t = pd.DataFrame({"n": g.size(), "mean level": g.ref.mean(), "RMS resid": g.rc.apply(lambda x: np.sqrt((x ** 2).mean())),
                          "CV": g.apply(lambda x: np.sqrt((x.rc ** 2).mean()) / x.ref.mean()),
                          "skew": g.rc.apply(stats.skew), "resid p10 / p90": g.rc.apply(
                              lambda x: f"{np.percentile(x, 10):+.1f} / {np.percentile(x, 90):+.1f}")})
        print(f"\nMode {mode} (level = {'projection mean' if mode == 'U' else 'conditional mean'}):")
        print(md(t, "{:.2f}"))
        if mode == "U":
            print("v1 (resid vs in-season mean): CV 0.79 (<=10) / 0.58 / 0.41 / 0.31 / 0.26 / 0.24 "
                  "(50-80)")
    print("\nCV at selected levels from the fold fit (previous 3 seasons): v2 U (projection "
          "residual), v2 PM (given actual minutes and team points), T1.4 bundle (given actual "
          "minutes and team points, its own rate model):")
    from courtside.proj import pipeline as P
    levels = (5, 12, 22, 35, 50)
    rows = []
    for S in seasons[1:]:
        tr = [x for x in seasons if x < S][-3:]
        row = [S]
        for mode in ("U", "PM"):
            r, _ = K.residuals(f[f.season.isin(tr)], mode)
            r["rc"] = r.res - r.groupby(["season", "tier"]).res.transform("mean")
            mu, cv = K.fit_cv(r.ref.to_numpy(), r.rc.to_numpy())
            row += [float(np.interp(x, mu, cv)) for x in levels]
        path = P.cache_dir() / f"bundle_{S}.json"
        if path.exists():
            cp = json.loads(path.read_text())["copula"]
            row += [float(np.interp(x, cp["cv_mu"], cp["cv"])) for x in levels]
        else:
            row += [np.nan] * len(levels)
        rows.append(row)
    cols = ["S"] + [f"{m} @{x}" for m in ("U", "PM", "T1.4") for x in levels]
    print(md(pd.DataFrame(rows, columns=cols).set_index("S"), "{:.2f}"))


def sec_team_specific(f, seasons):
    print("\n## 8. Team-specific effects")
    for mode in ("U", "PM"):
        z, _, _ = prep(f, mode)
        z["z2"] = z.z ** 2
        tg = z.groupby(["season", "team", "game_id"]).agg(S=("z", "sum"), Q=("z2", "sum"),
                                                         n=("z", "size")).reset_index()
        tg = tg.sort_values(["season", "team", "game_id"])
        tg["half"] = tg.groupby(["season", "team"]).cumcount() % 2
        v = float(z.z2.mean())

        def r_of(x):
            num = ((x.S ** 2 - x.Q) / 2).sum()
            cnt = (x.n * (x.n - 1) / 2).sum()
            return num / cnt / v if cnt else np.nan
        full = tg.groupby(["season", "team"]).apply(r_of).rename("r")
        h = tg.groupby(["season", "team", "half"]).apply(r_of).unstack()
        h.columns = ["A", "B"]
        h = h.dropna()
        rho = np.corrcoef(h.A, h.B)[0, 1]
        cov = np.cov(h.A, h.B)[0, 1]
        n = len(h)
        se = (1 - rho ** 2) / np.sqrt(n - 1)
        print(f"\nMode {mode}: team-season teammate r (all pairs, z): mean {full.mean():.3f}, SD "
              f"{full.std():.3f} over {len(full)} team-seasons")
        print(f"  split-half (odd/even games) correlation {rho:+.3f} +/- {1.96 * se:.3f}; "
              f"covariance of halves {cov:.5f} -> implied SD of true team-level r "
              f"{np.sqrt(max(cov, 0)):.3f} (half-sample sampling SD {h.A.std():.3f})")
        # persistence across consecutive seasons
        fr = full.reset_index()
        fr["sn"] = fr.season.str[:4].astype(int)
        nxt = fr.assign(sn=fr.sn - 1)[["team", "sn", "r"]].rename(columns={"r": "r_next"})
        pr = fr.merge(nxt, on=["team", "sn"]).dropna()
        if len(pr) > 10:
            rr = np.corrcoef(pr.r, pr.r_next)[0, 1]
            print(f"  same team, adjacent seasons: corr {rr:+.3f} +/- "
                  f"{1.96 * (1 - rr ** 2) / np.sqrt(len(pr) - 1):.3f} (n={len(pr)})")
        # team effect on residual level: does a team's mean residual persist (bias, not corr)?
        tb = z.assign(res0=z.res).groupby(["season", "team"]).res.mean()
        print(f"  team-season mean residual SD {tb.std():.2f} DK (mean n/season "
              f"{z.groupby(['season', 'team']).size().mean():.0f} rows)")


def sec_tiers_td(conn, f):
    """Robustness: tiers_to_date (prior played games) instead of the projection tier."""
    print("\n## 9. Robustness: tiers from prior played games (`tiers_to_date`) vs projection tiers")
    s0 = f.season.min()
    q = ",".join("?" * 2)
    pg = pd.read_sql_query(
        """SELECT game_id, player_id, team, season, game_date AS date, min, pts, fg3m, reb, ast,
                  stl, blk, tov FROM player_games WHERE season_type='Regular Season' AND min>0
           AND season >= ?""", conn, params=[K.season_label(int(s0[:4]) - 1)])
    pg["dk"] = K.dk_points(pg)
    pg = pg.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    pg["tier_td"] = K.tiers_to_date(pg)
    g = f.merge(pg[["game_id", "player_id", "tier_td"]], on=["game_id", "player_id"], how="left")
    g = g[g.tier_td.notna()].copy()
    agree = (g.tier == g.tier_td).mean()
    print(f"agreement of tier labels: {agree:.1%}; confusion (rows = projection tier, cols = to-date tier):")
    print(pd.crosstab(g.tier.map(dict(enumerate(TIER_NAMES))), g.tier_td.map(dict(enumerate(TIER_NAMES)))))
    cols = {}
    for name, tcol in (("projection tier", "tier"), ("to-date tier", "tier_td")):
        x = g.copy()
        x["tier"] = x[tcol].astype(int)
        x["grp"] = x["tier"] * K.NPOS + x["pos"]
        z, _, _ = prep(x, "U")
        t = pstats(z).table("team", "tier", nboot=100)
        cols[name] = t["r"]
        cols[name + " CI"] = [f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(t.lo, t.hi)]
    print("\nTeammates, U z:"); print(md(pd.DataFrame(cols)))


def sec_regulars(f):
    print("\n## 9b. Robustness: regulars only (P(play) >= 0.9, projected minutes >= 15)")
    x = f[(f.p_play >= 0.9) & (f.min_proj >= 15)]
    print(f"{len(x):,} of {len(f):,} player-games ({len(x) / len(f):.0%})")
    for mode in ("U", "PM"):
        cols = {}
        for lab, d in (("all rows", f), ("regulars", x)):
            z, _, _ = prep(d, mode)
            ps_ = pstats(z)
            for kind in ("team", "opp"):
                t = ps_.table(kind, "tier", nboot=100)
                a = ps_.table(kind, "all", nboot=100)
                t.loc["all"] = a.iloc[0]
                cols[f"{kind} {lab}"] = t["r"]
        print(f"\nMode {mode} (z):"); print(md(pd.DataFrame(cols)))


# ============================================================================== params

def write_params(f, seasons, window, k_tier, k_cell, out_seasons):
    print(f"\n## 10. Persisted parameters (window {window} seasons, k_tier {k_tier:.0f}, "
          f"k_cell {k_cell:.0f})")
    rows = []
    for S in out_seasons:
        tr = [s for s in seasons if s < S][-window:]
        if not tr:
            continue
        p = K.fit(f, tr, season=S, k_shrink=k_tier, k_cell=k_cell)
        path = K.save(p)
        t = p.table("PM")
        u = p.table("U")
        rows.append((S, ",".join(tr), u.rho_team_all, u.rho_opp_all, t.rho_team_all, t.rho_opp_all,
                     len(p.to_json())))
    print(md(pd.DataFrame(rows, columns=["S", "train", "U team", "U opp", "PM team", "PM opp",
                                         "bytes"]).set_index("S")))
    print("written to $COURTSIDE_DATA/proj (var/proj) as corr_<S>.json")


def sec_matrices(f, seasons, window=3):
    last = seasons[-1]
    tr = seasons[-window:]
    print(f"\n## 11. Fold matrices for the next season (train {tr[0]}..{tr[-1]})")
    p = K.fit(f, tr)
    for mode in ("U", "PM"):
        t = p.table(mode)
        print(f"\nMode {mode}: teammates all {t.rho_team_all:+.3f}, opponents all {t.rho_opp_all:+.3f}")
        rows = []
        for lab in K.CATS["tier"][0]:
            rows.append((lab, t.rho_team[lab], t.rho_opp[lab], int(t.n[f'team:{lab}'])))
        print(md(pd.DataFrame(rows, columns=["tier pair", "teammate", "opponent", "teammate pairs"]).set_index("tier pair")))
    t = p.table("PM")
    print("\nPM teammate cells (shrunk), 3x3 by position within each tier pair (rows = first-tier position):")
    for tp in K.CATS["tier"][0]:
        a, b = tp.split("|")
        m = pd.DataFrame(index=K.POS[:3], columns=K.POS[:3], dtype=float)
        for pa in K.POS[:3]:
            for pb in K.POS[:3]:
                key = K.pair_label(f"{a}:{pa}", f"{b}:{pb}")
                m.loc[pa, pb] = t.rho_team_cell.get(key, np.nan)
        print(f"\n{tp}:"); print(md(m))
    return p


# ============================================================================== main

def analyze(args):
    t0 = time.time()
    conn = db_conn()
    seasons = sorted(s for s in (args.seasons or [K.season_label(y) for y in range(2016, 2026)])
                     if K.extract_path(s).exists())
    print(f"# Courtside DFS correlation study v2 (T2.7): causal residuals vs pregame projection")
    print(f"seasons: {seasons[0]}..{seasons[-1]} ({len(seasons)}); generated "
          f"{time.strftime('%Y-%m-%d')}")
    f = K.load_frame(conn, seasons)
    sec_data(conn, f, seasons)
    sec_team_opp(f)
    sec_drift(f, seasons)
    d, best = sec_folds(f, seasons)
    sec_t14(conn, f, seasons)
    sec_blowout(f)
    sec_absence(conn, f, seasons)
    sec_cv(f, seasons)
    sec_team_specific(f, seasons)
    sec_tiers_td(conn, f)
    sec_regulars(f)
    bt = best[("team", "PM")]
    bu = best[("team", "U")]
    # persisted: window/k chosen on the PM teammate objective (the copula's input)
    window = int(bt["w"])
    k_tier = float(bt["k"])
    k_cell = 1e12 if bt["level"] == "tier-only" else float(bt["k"])
    if args.write_params:
        nxt = K.next_season(seasons[-1])
        write_params(f, seasons, window, k_tier, k_cell,
                     [s for s in seasons[1:]] + [nxt])
    sec_matrices(f, seasons, window)
    print(f"\n(U-objective best: w={int(bu['w'])}, k={bu['k']:.0f}, {bu['level']})")
    print(f"\nTotal analysis time {time.time() - t0:.0f}s")


def extract(args):
    conn = db_conn()
    for s in args.seasons:
        t0 = time.time()
        K.extract_season(conn, s, every=args.every, n_sims=args.sims, limit=args.limit,
                         log=lambda m: print(m, flush=True))
        print(f"{s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="analyze", choices=["extract", "analyze"])
    ap.add_argument("seasons", nargs="*")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write-params", action="store_true")
    a = ap.parse_args()
    (extract if a.cmd == "extract" else analyze)(a)
