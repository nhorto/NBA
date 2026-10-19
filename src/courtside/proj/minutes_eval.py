"""Walk-forward "avail" backtest of the minutes engine (docs/PLAYERS.md §4.2, §7; T1.2).

Protocol (per test season S): `minutes.fit` on the 5 seasons before S; every regular-season
team-game of S is projected from state strictly before its date. Scenario = the avail oracle of
the rung-2 engine (players.REGULAR_MIN on the rung-2 EWMA): regulars who did not play are absent,
everyone else on the recent roster is available (DNP-CDs are the model's problem). Two modes:
lock-45 (no starters) and lock-10 (actual starters as a proxy for confirmed ones, §7.1 L5).

Projection = `expected_minutes` (g integrated over margin ~ N(closing spread, 12.4^2)).
Population for the conditional metrics: projected >= 15 and played (never filtered on actual).
Baseline: rung-2 EWMA (`players` avail mode: 0.5/0.5 blend of alpha 0.2/0.02 EWMAs from 12,
floor 6, >= 8 players, water-fill to 240 with cap 42) on the same rows.

Tests reported:
  two_starters_out: team-games with >= 2 expected starters absent; the top bench player (highest
    healthy projection among available non-starters, played) -> projected rise vs healthy
    (scenario with nobody absent) and realised rise (actual - healthy projection). Variant
    max_proj_rise: the available non-starter with P(play) >= 0.8 and the largest projected rise
    (selected on the projection only, never on the outcome).
  star_absence: research/dfs_correlations.py definitions on the test season (stars = top-2 USG
    with >= 28 mpg and >= 20 GP; remaining top-6 by mpg; single-team players; other star
    available) -> projected delta minutes (scenario vs same scenario with the star available),
    implied delta DK at his EWMA DK/min, and the realised with/without difference.

Public API
  run_season(conn, history, season, log=print) -> dict
  run(conn, seasons=("2023-24", "2024-25"), log=print) -> dict;  format_report(dict) -> str
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable

import numpy as np
import pandas as pd

from courtside.model.players import REGULAR_MIN, allocate_minutes, select_lineup
from courtside.proj import minutes as M

TIER_LABELS = ("15-20", "20-26", "26-32", "32+")


def spreads(conn: sqlite3.Connection) -> pd.DataFrame:
    """game_id, team, spread (expected margin, + = team favoured) from closing lines."""
    cl = pd.read_sql_query("SELECT game_id, AVG(home_spread_close) hs FROM closing_lines "
                           "WHERE home_spread_close IS NOT NULL GROUP BY game_id", conn)
    tg = pd.read_sql_query("SELECT game_id, team, is_home FROM team_games", conn)
    m = tg.merge(cl, on="game_id")
    m["spread"] = np.where(m["is_home"] == 1, -m["hs"], m["hs"])
    return m[["game_id", "team", "spread"]]


def baseline(panel: pd.DataFrame) -> np.ndarray:
    """Rung-2 EWMA minutes in the avail mode, per team-game."""
    out = np.zeros(len(panel))
    proj = panel["base_ew"].to_numpy(float)
    reg = proj >= REGULAR_MIN
    pl = panel["played"].to_numpy(bool)
    ro = panel["roster"].to_numpy(int) == 1
    pa = np.where(reg, pl, ro).astype(float)
    for _, idx in panel.groupby("key", sort=False).indices.items():
        keep = select_lineup(proj[idx], pa[idx], 6.0, 8)
        m = np.zeros(len(idx))
        m[keep] = allocate_minutes(proj[idx][keep], pa[idx][keep], cap=42.0)
        out[idx] = m
    return out


def integrate(r: pd.DataFrame, prm: M.MinutesParams, spread: pd.Series, col="mean0"
              ) -> np.ndarray:
    """expected_minutes per team-game (spread indexed by key; NaN -> 0)."""
    out = np.zeros(len(r))
    sub = r[["p_play", "pf36", "reserve"]].copy()
    sub["mean"] = r[col].to_numpy()
    for key, idx in r.groupby("key", sort=False).indices.items():
        s = spread.get(key, 0.0)
        out[idx] = M.expected_minutes(sub.iloc[idx], prm, spread=s)
    return out


def _cl_se(diff: np.ndarray, cluster: np.ndarray) -> float:
    """SE of the mean of `diff`, clustered by `cluster` (game)."""
    n = len(diff)
    u = pd.Series(diff - diff.mean()).groupby(cluster).sum().to_numpy()
    G = len(u)
    if G < 2:
        return float("nan")
    return float(np.sqrt(G / (G - 1) * (u ** 2).sum()) / n)


def metrics(r: pd.DataFrame, proj: str, base: str = "base_rung2") -> dict:
    pop = r[r["played"] & (r[proj] >= 15)]
    y = pop["act_min"].to_numpy(float)
    e = pop[proj].to_numpy(float) - y
    eb = pop[base].to_numpy(float) - y
    gid = pop["game_id"].to_numpy()
    d = np.abs(eb) - np.abs(e)
    out = {"n": int(len(pop)), "mae": float(np.abs(e).mean()),
           "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
           "baseline_mae_same_rows": float(np.abs(eb).mean()),
           "improvement_pct": float(100 * d.mean() / np.abs(eb).mean()),
           "improvement_se_pct": float(100 * _cl_se(d, gid) / np.abs(eb).mean())}
    t = np.clip(np.searchsorted([20.0, 26.0, 32.0], pop[proj].to_numpy(float), side="right"),
                0, 3)
    out["by_tier"] = {TIER_LABELS[k]: {"n": int((t == k).sum()),
                                       "mae": float(np.abs(e[t == k]).mean()),
                                       "bias": float(e[t == k].mean()),
                                       "baseline_mae": float(np.abs(eb[t == k]).mean()),
                                       "baseline_bias": float(eb[t == k].mean())}
                      for k in range(4) if (t == k).any()}
    allp = r[r["played"] & (r[proj] > 0)]
    out["mae_all_played"] = float(np.abs(allp[proj] - allp["act_min"]).mean())
    bp = r[r["played"] & (r[base] >= 15)]
    out["baseline_own_population"] = {"n": int(len(bp)), "mae": float(
        np.abs(bp[base] - bp["act_min"]).mean())}
    return out


def _counterfactual(panel: pd.DataFrame, prm, keys, absent: np.ndarray, spread,
                    shares=None) -> pd.DataFrame:
    sub = panel[panel["key"].isin(keys)].copy()
    sub["cf_absent"] = absent[sub.index.to_numpy()]
    r = M.run_projection(sub, prm, absent_col="cf_absent", shares=shares)
    r["proj"] = integrate(r, prm, spread)
    return r


def two_starters_test(panel, r, prm, spread, shares=None) -> dict:
    """Projected and realised rise of the top bench player when >= 2 expected starters sit."""
    nab = r.groupby("key").apply(lambda d: int((d["exp_start"] & d["absent"]).sum()),
                                 include_groups=False)
    keys = nab[nab >= 2].index
    healthy = _counterfactual(panel, prm, keys, np.zeros(len(panel), bool), spread, shares)
    hk = healthy.set_index(["key", "player_id"])["proj"]
    rows = []
    rk = r[r["key"].isin(keys)]
    for key, d in rk.groupby("key", sort=False):
        bench = d[~d["exp_start"] & ~d["absent"]]
        if bench.empty:
            continue
        hp = hk.loc[key].reindex(bench["player_id"]).to_numpy()
        # value play: largest projected rise among players likely to play (p >= 0.8)
        rise = np.where(bench["p_play"].to_numpy() >= 0.8, bench["proj"].to_numpy() - hp,
                        -np.inf)
        for kind, i in (("top_bench", int(np.argmax(hp))), ("max_proj_rise", int(np.argmax(rise)))):
            b = bench.iloc[i]
            rows.append({"kind": kind, "key": key, "n_starters_out": int(nab[key]),
                         "player_id": b["player_id"], "healthy": hp[i], "proj": b["proj"],
                         "played": bool(b["played"]), "actual": b["act_min"],
                         "promoted": bool(b["promoted"])})
    t = pd.DataFrame(rows)
    res = {}
    for kind in ("top_bench", "max_proj_rise"):
        tk = t[t["kind"] == kind]
        pl = tk[tk["played"]]
        for lab, s_ in (("all >=2 out", pl), ("exactly 2 out", pl[pl["n_starters_out"] == 2])):
            if s_.empty:
                continue
            pr = (s_["proj"] - s_["healthy"]).to_numpy()
            re = (s_["actual"] - s_["healthy"]).to_numpy()
            res[f"{kind} | {lab}"] = {
                "team_games": int(len(s_)), "proj_rise": float(pr.mean()),
                "realised_rise": float(re.mean()), "realised_minus_proj": float(re.mean() - pr.mean()),
                "realised_se": float(re.std(ddof=1) / np.sqrt(len(re))),
                "share_promoted": float(s_["promoted"].mean())}
        res[f"{kind} | dnp_share"] = float(1 - tk["played"].mean()) if len(tk) else None
    return res


def _research_stars(conn, season):
    """Stars / top-6 per team-season on the test season with research definitions."""
    pg = pd.read_sql_query(
        "SELECT p.game_id, p.player_id, p.team, p.min, a.usg_pct FROM player_games p "
        "LEFT JOIN player_games_adv a USING (game_id, player_id) "
        "WHERE p.season = ? AND p.season_type = 'Regular Season' AND p.min > 0",
        conn, params=(season,))
    pg["usg_pct"] = pg["usg_pct"].astype(float)
    nt = pg.groupby("player_id")["team"].nunique()
    pg = pg[pg["player_id"].isin(nt[nt == 1].index)]
    st = pg.groupby(["team", "player_id"]).agg(gp=("min", "size"), mmin=("min", "mean"),
                                               usg=("usg_pct", "mean")).reset_index()
    out = {}
    for team, s in st.groupby("team"):
        cand = s[(s["mmin"] >= 28) & (s["gp"] >= 20)].sort_values("usg", ascending=False).head(2)
        if len(cand) < 2:
            continue
        top6 = s.sort_values("mmin", ascending=False).head(6)["player_id"].tolist()
        played = pg[pg["team"] == team].groupby("player_id")["game_id"].apply(set)
        out[team] = {"stars": cand["player_id"].tolist(), "top6": top6, "played": played}
    return pg, out


def star_absence_test(conn, season, panel, r, prm, spread, shares=None) -> dict:
    pg, teams = _research_stars(conn, season)
    rows, realised = [], []
    games = r.groupby("team")["game_id"].apply(set)
    for team, info in teams.items():
        tgames = games.get(team, set())
        s1, s2 = info["stars"]
        for star, other in ((s1, s2), (s2, s1)):
            ps, po = info["played"].get(star, set()), info["played"].get(other, set())
            out = (tgames - ps) & po
            inn = ps & po
            if len(out) < 3:
                continue
            for pid in info["top6"]:
                if pid == star:
                    continue
                d = pg[(pg["team"] == team) & (pg["player_id"] == pid)]
                do, di = d[d["game_id"].isin(out)], d[d["game_id"].isin(inn)]
                if len(do) < 3 or len(di) < 10:
                    continue
                realised.append({"n_out": len(do), "d_min": do["min"].mean() - di["min"].mean()})
                for g in do["game_id"]:
                    rows.append({"key": f"{g}|{team}", "player_id": pid, "star": star,
                                 "other_star": pid == other})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"pairs": 0}
    # counterfactual: same scenario, star available
    cf = panel["absent"].to_numpy().copy()
    star_by_key = t.groupby("key")["star"].first()
    m = panel["key"].map(star_by_key)
    cf[(panel["player_id"] == m).to_numpy()] = False
    alt = _counterfactual(panel, prm, star_by_key.index, cf, spread, shares)
    a = alt.set_index(["key", "player_id"])["proj"]
    cur = r.set_index(["key", "player_id"])
    idx = pd.MultiIndex.from_frame(t[["key", "player_id"]])
    t["proj"] = cur["proj"].reindex(idx).to_numpy()
    t["proj_with_star"] = a.reindex(idx).to_numpy()
    t["actual"] = cur["act_min"].reindex(idx).to_numpy()
    t["dkpm"] = cur["dkpm"].reindex(idx).to_numpy()
    t["base_rung2"] = cur["base_rung2"].reindex(idx).to_numpy()
    t = t.dropna(subset=["proj", "proj_with_star"])
    t["d_proj"] = t["proj"] - t["proj_with_star"]
    rz = pd.DataFrame(realised)
    res = {"player_games": int(len(t)), "star_out_team_games": int(t["key"].nunique()),
           "proj_d_min": float(t["d_proj"].mean()),
           "proj_d_dk_at_ewma_dkpm": float((t["d_proj"] * t["dkpm"]).mean()),
           "realised_d_min_research_method": float(np.average(rz["d_min"], weights=rz["n_out"]))
           if len(rz) else None,
           "actual_minus_proj_with_star": float((t["actual"] - t["proj_with_star"]).mean()),
           "actual_minus_proj": float((t["actual"] - t["proj"]).mean()),
           "non_star_proj_d_min": float(t.loc[~t["other_star"], "d_proj"].mean()),
           "other_star_proj_d_min": float(t.loc[t["other_star"], "d_proj"].mean())}
    return res


def coverage(r: pd.DataFrame, prm, spread, n_games: int = 400, sims: int = 400,
             seed: int = 7) -> dict:
    """80%/50% coverage of sampled minutes (conditional on playing) on a subsample."""
    rng = np.random.default_rng(seed)
    keys = r["key"].unique()
    keys = rng.choice(keys, size=min(n_games, len(keys)), replace=False)
    hit80 = hit50 = n = 0
    pit = []
    for key in keys:
        d = r[r["key"] == key]
        proj = d[["p_play", "pf36", "reserve"]].assign(mean=d["mean0"].to_numpy())
        s = spread.get(key, 0.0)
        s = 0.0 if not np.isfinite(s) else s
        margins = rng.normal(s, M.MARGIN_SD, sims)
        x = M.sample_minutes(proj, prm, margins, rng)
        pl = d["played"].to_numpy() & (d["proj"].to_numpy() >= 15)
        for i in np.flatnonzero(pl):
            xi = x[:, i][x[:, i] > 0]
            if len(xi) < 50:
                continue
            y = d["act_min"].to_numpy()[i]
            lo, hi = np.quantile(xi, [0.1, 0.9])
            l5, h5 = np.quantile(xi, [0.25, 0.75])
            hit80 += lo <= y <= hi
            hit50 += l5 <= y <= h5
            pit.append((xi < y).mean())
            n += 1
    pit = np.asarray(pit)
    dec = np.histogram(pit, bins=np.linspace(0, 1, 11))[0] / max(len(pit), 1)
    return {"n": n, "cov80": hit80 / max(n, 1), "cov50": hit50 / max(n, 1),
            "pit_deciles": [round(float(v), 3) for v in dec]}


def run_season(conn, h: M.History, season: str, log: Callable[[str], None] = print,
               prm: M.MinutesParams | None = None, window: int = 5) -> dict:
    t0 = time.time()
    tr_seasons = sorted(x for x in h.tg["season"].unique() if x < season)[-window:]
    ptr = M.oracle_panel(h, M.eval_targets(h, tr_seasons))
    ev_tr = M.pair_events(ptr)
    if prm is None:
        prm = M.fit(h, season, window=window, log=log, panel=ptr, shares=M.PairShares(ev_tr))
    t_fit = time.time() - t0
    del ptr
    tg = M.eval_targets(h, [season])
    panel = M.oracle_panel(h, tg)
    # pair shares: events of the training seasons + the test season, looked up strictly
    # before each game's date
    shares = M.PairShares(pd.concat([ev_tr, M.pair_events(panel)], ignore_index=True))
    sp = spreads(conn)
    sp["key"] = sp["game_id"] + "|" + sp["team"]
    spread = sp.set_index("key")["spread"]
    r = M.run_projection(panel, prm, shares=shares)
    r["proj"] = integrate(r, prm, spread)
    r["base_rung2"] = baseline(panel)
    r10 = M.run_projection(panel, prm, starters=True, shares=shares)
    r["proj_lock10"] = integrate(r10, prm, spread)
    r["proj_nomargin"] = r["mean0"]
    out = {"season": season, "params": json.loads(prm.to_json()),
           "team_games": int(panel["key"].nunique()),
           "lock45": metrics(r, "proj"), "lock10": metrics(r, "proj_lock10"),
           "lock45_without_margin_integration": metrics(r, "proj_nomargin")}
    # variants (same fitted params): literal deterministic next man up; proportional 240
    for name, kw in (("lock45_nmu_top", {"nmu_mode": "top"}),
                     ("lock45_proportional_waterfill", {"recon": "proportional"})):
        q = M.MinutesParams.from_json(prm.to_json())
        for k_, v_ in kw.items():
            setattr(q, k_, v_)
        rv = M.run_projection(panel, q, shares=shares)
        rv["proj"] = integrate(rv, q, spread)
        rv["base_rung2"] = r["base_rung2"].to_numpy()
        out[name] = metrics(rv, "proj")
        if name == "lock45_nmu_top":
            out["two_starters_out_nmu_top"] = two_starters_test(panel, rv, q, spread, shares)
    pop = r[r["played"] & (r["proj"] >= 15)]
    out["lock10_on_lock45_rows_mae"] = float(np.abs(pop["proj_lock10"] - pop["act_min"]).mean())
    out["two_starters_out"] = two_starters_test(panel, r, prm, spread, shares)
    out["star_absence"] = star_absence_test(conn, season, panel, r, prm, spread, shares)
    out["sample_coverage"] = coverage(r, prm, spread)
    out["seconds"] = round(time.time() - t0, 1)
    out["fit_seconds"] = round(t_fit, 1)
    log(f"{season}: done in {out['seconds']} s")
    return out


def run(conn, seasons=("2023-24", "2024-25"), log: Callable[[str], None] = print,
        history: M.History | None = None) -> dict:
    t0 = time.time()
    h = history if history is not None else M.load_history(conn)
    t_load = time.time() - t0
    res = {"load_seconds": round(t_load, 1),
           "seasons": {s: run_season(conn, h, s, log=log) for s in seasons}}
    res["total_seconds"] = round(time.time() - t0, 1)
    return res


def format_report(res: dict) -> str:
    L = []
    for s, r in res["seasons"].items():
        L.append(f"## {s}  (fit on {', '.join(r['params']['train_seasons'])}; "
                 f"{r['team_games']} team-games; {r['seconds']} s)")
        for mode in ("lock45", "lock10", "lock45_without_margin_integration", "lock45_nmu_top",
                     "lock45_proportional_waterfill"):
            m = r[mode]
            L.append(f"- {mode}: n={m['n']} MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} bias "
                     f"{m['bias']:+.3f} | rung-2 EWMA same rows {m['baseline_mae_same_rows']:.3f}"
                     f" -> {m['improvement_pct']:.1f}% better (SE {m['improvement_se_pct']:.1f})"
                     f" | all played MAE {m['mae_all_played']:.3f}")
            if mode == "lock45":
                for k, v in m["by_tier"].items():
                    L.append(f"    tier {k}: n={v['n']} MAE {v['mae']:.3f} bias {v['bias']:+.3f}"
                             f" | EWMA MAE {v['baseline_mae']:.3f} bias {v['baseline_bias']:+.3f}")
                bp = m["baseline_own_population"]
                L.append(f"    rung-2 EWMA own population (>=15): n={bp['n']} MAE {bp['mae']:.3f}")
        L.append(f"- two starters out: {json.dumps(r['two_starters_out'], default=float)}")
        L.append(f"- two starters out, literal top next-man-up: "
                 f"{json.dumps(r['two_starters_out_nmu_top'], default=float)}")
        L.append(f"- star absence: {json.dumps(r['star_absence'], default=float)}")
        L.append(f"- sampled coverage: {json.dumps(r['sample_coverage'])}")
    return "\n".join(L)


if __name__ == "__main__":  # pragma: no cover - manual run
    import sys
    c = sqlite3.connect("file:var/courtside.db?mode=ro", uri=True)
    seasons = tuple(sys.argv[1:]) or ("2023-24", "2024-25")
    out = run(c, seasons)
    print(format_report(out))
    print(json.dumps({k: v for k, v in out.items() if k != "seasons"}))
