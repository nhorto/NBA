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

T2.2 (v1 vs v0.5): `run_compare` fits both on the same training panel and scores them on the
same test rows (paired gain clustered by game); `live_like` re-projects the 2025-12-22 ->
2026-04-12 slice from the official reports at the first tip - 45 min (enumerated report
scenarios, E[min | plays] = scenario mixture) and, for the gap, with the oracle; `run_v1` +
`format_v1_report` write var/reports/minutes_v1_eval.{json,txt}
(`python -m courtside.proj.minutes_eval v1 [seasons]`).

Public API
  run_season(conn, history, season, log=print) -> dict
  prepare_season / evaluate_params / run_compare / live_like / run_v1 / format_v1_report
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
    """v0.5 walk-forward evaluation of one season (the T1.2 report)."""
    t0 = time.time()
    ctx = prepare_season(conn, h, season, window=window)
    if prm is None:
        prm = M.fit(h, season, window=window, log=log, panel=ctx["ptr"],
                    shares=M.PairShares(ctx["ev_tr"]))
    t_fit = time.time() - t0
    out = evaluate_params(conn, season, ctx, prm, variants=True)
    out.pop("_frame")
    out["seconds"] = round(time.time() - t0, 1)
    out["fit_seconds"] = round(t_fit, 1)
    log(f"{season}: done in {out['seconds']} s")
    return out


def prepare_season(conn, h: M.History, season: str, window: int = 5,
                   use_share: bool = False) -> dict:
    """Training panel (oracle, `window` seasons < S), test panel (oracle, S), pair shares
    (training + test events, every lookup strictly before the game date), closing spreads.
    use_share: attach `share_map` (v1 share features) to both panels."""
    tr_seasons = sorted(x for x in h.tg["season"].unique() if x < season)[-window:]
    keep = h.shares
    h.shares = None
    ptr = M.oracle_panel(h, M.eval_targets(h, tr_seasons))
    ev_tr = M.pair_events(ptr)
    panel = M.oracle_panel(h, M.eval_targets(h, [season]))
    h.shares = keep
    shares = M.PairShares(pd.concat([ev_tr, M.pair_events(panel)], ignore_index=True))
    if use_share:
        ptr["share_map"] = M.share_maps(shares, ptr)
        panel["share_map"] = M.share_maps(shares, panel)
    sp = spreads(conn)
    sp["key"] = sp["game_id"] + "|" + sp["team"]
    return {"season": season, "train_seasons": tr_seasons, "ptr": ptr, "ev_tr": ev_tr,
            "panel": panel, "shares": shares, "spread": sp.set_index("key")["spread"],
            "base_rung2": baseline(panel)}


def evaluate_params(conn, season: str, ctx: dict, prm: M.MinutesParams,
                    variants: bool = True) -> dict:
    """Every T1.2 metric of `prm` (v0.5 or v1) on the prepared test panel."""
    panel, shares, spread = ctx["panel"], ctx["shares"], ctx["spread"]
    M.V1_STATS.update(rows=0, fallback_rows=0, panel_fallback_calls=0)
    r = M.run_projection(panel, prm, shares=shares)
    fb = dict(M.V1_STATS)
    r["proj"] = integrate(r, prm, spread)
    r["base_rung2"] = ctx["base_rung2"]
    r10 = M.run_projection(panel, prm, starters=True, shares=shares)
    r["proj_lock10"] = integrate(r10, prm, spread)
    r["proj_nomargin"] = r["mean0"]
    pj = json.loads(prm.to_json())
    pj["gbm_model"] = f"<{len(prm.gbm_model)} chars>" if prm.gbm_model else ""
    out = {"season": season, "version": prm.version, "params": pj,
           "team_games": int(panel["key"].nunique()),
           "lock45": metrics(r, "proj"), "lock10": metrics(r, "proj_lock10"),
           "lock45_without_margin_integration": metrics(r, "proj_nomargin")}
    if prm.version == "v1":
        ok = M.v1_ok(panel)
        av = ~panel["absent"].to_numpy(bool)
        pop = (r["played"] & (r["proj"] >= 15)).to_numpy()
        out["fallback"] = {"rows_available": int(av.sum()),
                           "share_available_rows": float((~ok & av).sum() / max(av.sum(), 1)),
                           "share_eval_population": float((~ok & pop).sum() / max(pop.sum(), 1)),
                           "v1_stats_lock45": fb}
        # stage-2 off: the GBM's learned promotion alone (+ the 240 reconciliation)
        q = M.MinutesParams.from_json(prm.to_json())
        q.nmu_scale, q.share_coef = 0.0, 0.0
        rv = M.run_projection(panel, q, shares=shares)
        rv["proj"] = integrate(rv, q, spread)
        rv["base_rung2"] = r["base_rung2"].to_numpy()
        out["lock45_v1_no_stage2"] = metrics(rv, "proj")
    elif variants:
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
    out["_frame"] = r[["key", "game_id", "date", "player_id", "played", "act_min", "proj",
                       "proj_lock10", "base_rung2"]]
    return out


def compare_frames(fr: dict) -> dict:
    """v1 vs v0.5 on identical rows: the v0.5 population (played, v0.5 projected >= 15), the
    union of both populations, with the paired difference clustered by game."""
    a, b = fr["v0.5"], fr["v1"]
    assert (a["key"].to_numpy() == b["key"].to_numpy()).all()
    out = {}
    pl = a["played"].to_numpy(bool)
    for lab, m in (("v05_population", pl & (a["proj"].to_numpy() >= 15)),
                   ("union_population", pl & ((a["proj"].to_numpy() >= 15)
                                              | (b["proj"].to_numpy() >= 15))),
                   ("v1_population", pl & (b["proj"].to_numpy() >= 15))):
        y = a["act_min"].to_numpy(float)[m]
        ea = np.abs(a["proj"].to_numpy(float)[m] - y)
        eb = np.abs(b["proj"].to_numpy(float)[m] - y)
        d = ea - eb
        out[lab] = {"n": int(m.sum()), "mae_v05": float(ea.mean()), "mae_v1": float(eb.mean()),
                    "gain": float(d.mean()),
                    "gain_se": _cl_se(d, a["game_id"].to_numpy()[m]),
                    "bias_v05": float((a["proj"].to_numpy(float)[m] - y).mean()),
                    "bias_v1": float((b["proj"].to_numpy(float)[m] - y).mean())}
    return out


def run_compare(conn, h: M.History, season: str, log: Callable[[str], None] = print,
                window: int = 5, v1_kw: dict | None = None, prms: dict | None = None) -> dict:
    """Walk-forward v0.5 and v1 (fit < S) on the same test rows (T2.2). prms: optional
    already-fitted {"v0.5": .., "v1": ..} (fit < S) to skip the fits."""
    t0 = time.time()
    v1_kw = dict(v1_kw or {})
    ctx = prepare_season(conn, h, season, window=window,
                         use_share=bool(v1_kw.get("use_share")))
    sh_tr = M.PairShares(ctx["ev_tr"])
    if prms is not None:
        prm05, prm1 = prms["v0.5"], prms["v1"]
        t05 = 0.0
    else:
        prm05 = M.fit(h, season, window=window, log=log, panel=ctx["ptr"], shares=sh_tr)
        t05 = time.time() - t0
        prm1 = M.fit_v1(h, prm05, ctx["ptr"], sh_tr, log=log, **v1_kw)
    t1 = time.time() - t0 - t05
    out = {"season": season, "fit_seconds": {"v0.5": round(t05, 1), "v1_extra": round(t1, 1)},
           "versions": {}}
    frames = {}
    for name, prm in (("v0.5", prm05), ("v1", prm1)):
        te = time.time()
        res = evaluate_params(conn, season, ctx, prm, variants=False)
        frames[name] = res.pop("_frame")
        res["eval_seconds"] = round(time.time() - te, 1)
        out["versions"][name] = res
        log(f"{season} {name}: lock45 MAE {res['lock45']['mae']:.3f} "
            f"bias {res['lock45']['bias']:+.3f} ({res['eval_seconds']} s)")
    out["paired"] = compare_frames(frames)
    out["seconds"] = round(time.time() - t0, 1)
    out["_prm"] = {"v0.5": prm05, "v1": prm1}
    return out


LIVE_LO, LIVE_HI = "2025-12-22", "2026-04-12"
MAX_ENUM = 4            # uncertain players enumerated per team-game (2^4 scenarios)


def report_scenarios(av: pd.DataFrame, key_of: dict, lo: float = 0.02, hi: float = 0.98
                     ) -> dict:
    """Per team-game key: list of (weight, frozenset absent ids) from the report P(play) of
    listed players (pipeline.LISTED semantics: unlisted players are left to the minutes engine).
    p <= lo -> out; p >= hi -> available; the MAX_ENUM most uncertain are enumerated, any
    further uncertain player is out when p < 0.5."""
    from itertools import product
    from . import availability as AV
    listed = set(AV.LISTED) | {"removed", "daytoday"}      # = pipeline.LISTED
    out = {}
    av = av[av["status"].isin(listed)]
    for (gid, team), d in av.groupby(["game_id", "team"]):
        key = key_of.get((gid, team))
        if key is None:
            continue
        pid = d["player_id"].astype(int).to_numpy()
        p = d["p_play"].to_numpy(float)
        sure_out = set(pid[p <= lo].tolist())
        unc = np.flatnonzero((p > lo) & (p < hi))
        unc = unc[np.argsort(-np.abs(p[unc] - 0.5) * -1)][:MAX_ENUM] if len(unc) else unc
        rest = set(np.flatnonzero((p > lo) & (p < hi)).tolist()) - set(unc.tolist())
        sure_out |= {int(pid[i]) for i in rest if p[i] < 0.5}
        sc = []
        for bits in product((0, 1), repeat=len(unc)):
            w = 1.0
            ab = set(sure_out)
            for b, i in zip(bits, unc):
                w *= p[i] if b else 1 - p[i]
                if not b:
                    ab.add(int(pid[i]))
            sc.append((w, frozenset(ab)))
        out[key] = sc
    return out


def live_like(conn, h: M.History, prms: dict, lo: str = LIVE_LO, hi: str = LIVE_HI,
              log: Callable[[str], None] = print, shares: "M.PairShares | None" = None) -> dict:
    """Live-like slice (§4.2 M6, §7.1 L4): lock-45 minutes from the official reports at the
    first tip - 45 min (availability.p_play_slate, as pipeline.report_avail) instead of the
    oracle. Candidates = recent roster + report players not 'out' (box history required).
    E[min | plays] mixes the enumerated report scenarios (weight x P(plays | available)).
    The same rows are also projected with the oracle for the gap. prms: {name: params}."""
    from datetime import date as _date, timedelta
    from . import availability as AV
    t0 = time.time()
    season = "2025-26"
    tg = M.eval_targets(h, [season])
    d_lo = M._days(pd.Series([lo]))[0]
    d_hi = M._days(pd.Series([hi]))[0]
    tg = tg[(tg["date"] >= d_lo) & (tg["date"] <= d_hi)].reset_index(drop=True)
    days = sorted(tg["date"].unique())
    avh = AV.load_history(conn, _date.fromisoformat(lo))
    sp = spreads(conn)
    sp["key"] = sp["game_id"] + "|" + sp["team"]
    spread = sp.set_index("key")["spread"]
    rows, panels, scen = [], [], {}
    for dd in days:
        day = (pd.Timestamp(int(dd), unit="D")).date()
        g = AV.day_games(conn, day)
        at = (g["tip"].min() if len(g) else AV._default_tip(day)) - timedelta(minutes=45)
        rf = conn.row_factory
        conn.row_factory = sqlite3.Row         # injuries.* reads rows as mappings
        try:
            av = AV.p_play_slate(conn, day, at, hist=avh)
        finally:
            conn.row_factory = rf
        t = tg[tg["date"] == dd]
        key_of = {(a, b): k for a, b, k in zip(t["game_id"], t["team"], t["key"])}
        ex = av[(av["status"] != "out") & av["game_id"].isin(set(t["game_id"]))]
        ex = pd.DataFrame({"key": [key_of.get((a, b)) for a, b in zip(ex["game_id"], ex["team"])],
                           "player_id": ex["player_id"].astype(np.int64)}).dropna()
        p = M.pregame_panel(h, t, extra=ex if len(ex) else None)
        p = p[(p["roster"] == 1) | p["last_date"].notna()]
        panels.append(p)
        scen.update(report_scenarios(av, key_of))
        rows.append({"day": str(day), "at": str(at), "n_report_rows": int(len(av))})
    base = pd.concat(panels, ignore_index=True)
    if shares is not None and "share_map" not in base and h.shares is not None:
        base["share_map"] = M.share_maps(h.shares, base)
    # stacked scenario panel
    parts = []
    for key, d in base.groupby("key", sort=False):
        sc = scen.get(key, [(1.0, frozenset())])
        for s_i, (w, ab) in enumerate(sc):
            q = d.copy()
            q["absent"] = q["player_id"].isin(ab)
            q["skey"] = f"{key}#{s_i}"
            q["w"] = w
            parts.append(q)
    st = pd.concat(parts, ignore_index=True)
    st = st.rename(columns={"key": "tkey"}).rename(columns={"skey": "key"})
    act = h.hist[["game_id", "team", "player_id", "min"]].copy()
    act["tkey"] = act["game_id"] + "|" + act["team"]
    act = act.set_index(["tkey", "player_id"])["min"]
    out = {"slice": [lo, hi], "team_games": int(base["key"].nunique()),
           "days": len(days), "scenarios": int(st["key"].nunique()), "versions": {}}
    sp_s = pd.Series(spread.reindex(st["tkey"].unique()).to_numpy(),
                     index=st["tkey"].unique())
    for name, prm in prms.items():
        r = M.run_projection(st, prm, shares=shares)
        spk = pd.Series(sp_s.reindex(r["tkey"]).to_numpy(), index=r["key"]).groupby(
            level=0).first()
        r["proj"] = integrate(r, prm, spk)
        r["wp"] = r["w"] * r["p_play"]
        r["wpm"] = r["wp"] * r["proj"]
        agg = r.groupby(["tkey", "player_id"]).agg(wp=("wp", "sum"), wpm=("wpm", "sum"),
                                                   w=("w", "sum"))
        agg["proj"] = np.where(agg["wp"] > 0, agg["wpm"] / agg["wp"].clip(lower=1e-12), 0.0)
        agg["act_min"] = act.reindex(agg.index).to_numpy()
        agg["played"] = agg["act_min"].notna() & (agg["act_min"] > 0)
        agg = agg.reset_index()
        agg["game_id"] = agg["tkey"].str.split("|").str[0]
        pop = agg[agg["played"] & (agg["proj"] >= 15)]
        e = (pop["proj"] - pop["act_min"]).to_numpy(float)
        t_ = np.clip(np.searchsorted([20.0, 26.0, 32.0], pop["proj"].to_numpy(float),
                                     side="right"), 0, 3)
        out["versions"][name] = {
            "n": int(len(pop)), "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "mae_se": _cl_se(np.abs(e), pop["game_id"].to_numpy()),
            "by_tier": {TIER_LABELS[k]: {"n": int((t_ == k).sum()),
                                         "mae": float(np.abs(e[t_ == k]).mean()),
                                         "bias": float(e[t_ == k].mean())}
                        for k in range(4) if (t_ == k).any()},
            "played_not_projected": int((~agg["proj"].gt(0) & agg["played"]).sum())}
        out["versions"][name]["_frame"] = agg
        log(f"live-like {name}: n={len(pop)} MAE {np.abs(e).mean():.3f} bias {e.mean():+.3f}")
    # oracle on the same slice (for the gap)
    t = tg.copy()
    po = M.oracle_panel(h, t)
    if "share_map" in base and "share_map" not in po and h.shares is not None:
        po["share_map"] = M.share_maps(h.shares, po)
    for name, prm in prms.items():
        r = M.run_projection(po, prm, shares=shares)
        r["proj"] = integrate(r, prm, spread)
        pop = r[r["played"] & (r["proj"] >= 15)]
        e = (pop["proj"] - pop["act_min"]).to_numpy(float)
        out["versions"][name]["oracle_same_slice"] = {"n": int(len(pop)),
                                                      "mae": float(np.abs(e).mean()),
                                                      "bias": float(e.mean())}
    # identical rows across versions (played & any version >= 15)
    names = list(prms)
    fr = {k: out["versions"][k].pop("_frame").set_index(["tkey", "player_id"]) for k in names}
    if len(names) == 2:
        a, b = fr[names[0]], fr[names[1]].reindex(fr[names[0]].index)
        m = a["played"] & ((a["proj"] >= 15) | (b["proj"] >= 15))
        ea = np.abs(a.loc[m, "proj"] - a.loc[m, "act_min"]).to_numpy()
        eb = np.abs(b.loc[m, "proj"] - b.loc[m, "act_min"]).to_numpy()
        out["paired_union"] = {"n": int(m.sum()), f"mae_{names[0]}": float(ea.mean()),
                               f"mae_{names[1]}": float(eb.mean()),
                               "gain": float((ea - eb).mean()),
                               "gain_se": _cl_se(ea - eb, a.loc[m, "game_id"].to_numpy())}
    out["report_days_sample"] = rows[:3]
    out["seconds"] = round(time.time() - t0, 1)
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


def coach_coverage(conn, h: M.History, seasons) -> dict:
    """Share of regular-season team-games with a known coach (N4) and the curated table's
    agreement with the team_coaches snapshots."""
    out = {}
    if h.coach is None:
        return {"available": False}
    g = h.coach.games
    tgx = h.tg[h.tg["season_type"] == "Regular Season"][["game_id", "team", "season"]]
    g = g.merge(tgx, on=["game_id", "team"])
    for s in seasons:
        x = g[g["season"] == s]
        out[s] = {"team_games": int(len(x)), "known": float((x["coach_id"] != 0).mean())
                  if len(x) else None, "coaches": int(x["coach_id"].nunique())}
    snap = M.load_coach_snapshots(conn)
    chk = M.coach_snapshot_check(M.coach_stints(snap), snap)
    out["snapshot_check"] = {"n": int(len(chk)), "agree": float(chk["ok"].mean()),
                             "disagree": chk[~chk["ok"]].to_dict("records")}
    st = M.coach_stints(snap)
    out["stints"] = {"curated": int((st["source"] == "curated").sum()),
                     "snapshot": int((st["source"] == "snapshot").sum()),
                     "in_season_changes": int(sum(
                         1 for t, v in M.COACH_STINTS.items() for d, _ in v
                         if not d.endswith("-10-01")))}
    return out


def latency(h: M.History, prms: dict, season: str, n: int = 60, seed: int = 3) -> dict:
    """Seconds per copula.ScenarioMinutes.project call (fresh absent sets) per version."""
    from . import copula as C
    tg = M.eval_targets(h, [season])
    rng = np.random.default_rng(seed)
    t = tg.iloc[rng.choice(len(tg), size=min(n, len(tg)), replace=False)].reset_index(drop=True)
    panel = M.pregame_panel(h, t)
    out = {}
    for name, prm in prms.items():
        t0 = time.time()
        calls = 0
        for _, d in panel.groupby("key", sort=False):
            sm = C.ScenarioMinutes(d, prm)
            for k in range(5):
                ab = np.zeros(len(sm), bool)
                ab[rng.choice(len(sm), size=min(k, len(sm)), replace=False)] = True
                sm.project(ab)
                calls += 1
        out[name] = round((time.time() - t0) / max(calls, 1) * 1000, 2)
    return {"ms_per_project_call": out, "calls_per_version": calls}


def run_v1(conn, seasons=("2023-24", "2024-25"), live: bool = True,
           log: Callable[[str], None] = print, history: M.History | None = None,
           v1_kw: dict | None = None, params_dir=None) -> dict:
    """T2.2: v0.5 vs v1 walk-forward on `seasons` + the live-like slice (fit < 2025-26)."""
    from pathlib import Path
    t0 = time.time()
    h = history if history is not None else M.load_history(conn)
    res = {"load_seconds": round(time.time() - t0, 1), "seasons": {}}
    prms = {}
    pdir = Path(params_dir) if params_dir is not None else None

    def saved(s):
        if pdir is None:
            return None
        f = {v: pdir / f"minutes_params_{s}_{v}.json" for v in ("v0.5", "v1")}
        if all(x.exists() for x in f.values()):
            return {v: M.MinutesParams.from_json(x.read_text()) for v, x in f.items()}
        return None

    def save(s, d, partial=None):
        if pdir is None:
            return
        pdir.mkdir(parents=True, exist_ok=True)
        for v, prm in d.items():
            (pdir / f"minutes_params_{s}_{v}.json").write_text(prm.to_json())
        if partial is not None:
            (pdir / f"minutes_v1_eval_partial_{s}.json").write_text(
                json.dumps(partial, default=float))

    for s in seasons:
        r = run_compare(conn, h, s, log=log, v1_kw=v1_kw, prms=saved(s))
        prms[s] = r.pop("_prm")
        res["seasons"][s] = r
        save(s, prms[s], r)
    if live:
        tl = time.time()
        ctx = prepare_season(conn, h, "2025-26")
        sh = M.PairShares(ctx["ev_tr"])
        got = saved("2025-26")
        if got is None:
            p05 = M.fit(h, "2025-26", log=log, panel=ctx["ptr"], shares=sh)
            p1 = M.fit_v1(h, p05, ctx["ptr"], sh, log=log, **(v1_kw or {}))
            save("2025-26", {"v0.5": p05, "v1": p1})
        else:
            p05, p1 = got["v0.5"], got["v1"]
        prms["2025-26"] = {"v0.5": p05, "v1": p1}
        ll = live_like(conn, h, {"v0.5": p05, "v1": p1}, log=log, shares=ctx["shares"])
        ll["fit_v1_diagnostics"] = p1.diagnostics.get("v1")
        ll["seconds_incl_fit"] = round(time.time() - tl, 1)
        res["live_like"] = ll
    all_s = sorted({x for s in prms for x in prms[s]["v1"].train_seasons} | set(seasons))
    res["coach_coverage"] = coach_coverage(conn, h, all_s)
    last = list(prms)[-1]
    res["latency"] = latency(h, prms[last], seasons[-1])
    res["total_seconds"] = round(time.time() - t0, 1)
    return res


def _m(x, k="mae"):
    return f"{x[k]:.3f}" if x and x.get(k) is not None else "-"


def format_v1_report(res: dict) -> str:
    L = ["# minutes v1 (T2.2) vs v0.5: walk-forward, fit on the 5 seasons < S, lock-45 unless "
         "noted; MAE on played rows projected >= 15 (each version's own population) and paired "
         "on identical rows", ""]
    L.append("| season | protocol | v0.5 MAE | v1 MAE | paired gain (v0.5 rows) | v0.5 bias | "
             "v1 bias |")
    L.append("|---|---|---|---|---|---|---|")
    for s, r in res["seasons"].items():
        a, b = r["versions"]["v0.5"]["lock45"], r["versions"]["v1"]["lock45"]
        pg = r["paired"]["v05_population"]
        L.append(f"| {s} | avail oracle | {a['mae']:.3f} | {b['mae']:.3f} | "
                 f"{pg['gain']:+.3f} (SE {pg['gain_se']:.3f}, n {pg['n']}) | {a['bias']:+.3f} | "
                 f"{b['bias']:+.3f} |")
    ll = res.get("live_like")
    if ll:
        a, b = ll["versions"]["v0.5"], ll["versions"]["v1"]
        pu = ll.get("paired_union", {})
        L.append(f"| 2025-26 slice {ll['slice'][0]}..{ll['slice'][1]} | official reports "
                 f"(tip-45) | {a['mae']:.3f} | {b['mae']:.3f} | "
                 f"{pu.get('gain', float('nan')):+.3f} (SE {pu.get('gain_se', float('nan')):.3f},"
                 f" union n {pu.get('n')}) | {a['bias']:+.3f} | {b['bias']:+.3f} |")
        L.append(f"| same slice | avail oracle | {_m(a['oracle_same_slice'])} | "
                 f"{_m(b['oracle_same_slice'])} | | "
                 f"{a['oracle_same_slice']['bias']:+.3f} | {b['oracle_same_slice']['bias']:+.3f} |")
    L.append("")
    for s, r in res["seasons"].items():
        L.append(f"## {s}")
        for v in ("v0.5", "v1"):
            x = r["versions"][v]
            m = x["lock45"]
            L.append(f"- {v} lock45: n={m['n']} MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} bias "
                     f"{m['bias']:+.3f} | rung-2 EWMA same rows {m['baseline_mae_same_rows']:.3f} "
                     f"-> {m['improvement_pct']:.1f}% better (SE {m['improvement_se_pct']:.1f}) "
                     f"| all played {m['mae_all_played']:.3f} | lock10 MAE "
                     f"{x['lock10']['mae']:.3f}")
            L.append("    tiers: " + "; ".join(f"{k} n={t['n']} MAE {t['mae']:.3f} bias "
                                               f"{t['bias']:+.3f}" for k, t in m["by_tier"].items()))
            ts = x["two_starters_out"]
            vp = ts.get("max_proj_rise | all >=2 out", {})
            vp2 = ts.get("max_proj_rise | exactly 2 out", {})
            L.append(f"    S14 value play, >=2 starters out: n={vp.get('team_games')} projected "
                     f"+{vp.get('proj_rise', float('nan')):.2f} realised "
                     f"+{vp.get('realised_rise', float('nan')):.2f}; exactly 2: projected "
                     f"+{vp2.get('proj_rise', float('nan')):.2f} realised "
                     f"+{vp2.get('realised_rise', float('nan')):.2f}")
            sa = x["star_absence"]
            L.append(f"    star absence: remaining top-6 projected d_min "
                     f"{sa.get('proj_d_min', float('nan')):+.2f} (realised, research method "
                     f"{sa.get('realised_d_min_research_method') or float('nan'):+.2f}); actual "
                     f"- projection {sa.get('actual_minus_proj', float('nan')):+.2f}")
            c = x["sample_coverage"]
            L.append(f"    sampled minutes coverage: 80% {c['cov80']:.3f}, 50% {c['cov50']:.3f}"
                     f" (n {c['n']})")
            if v == "v1":
                f = x["fallback"]
                L.append(f"    S15 fallback to v0.5: {100 * f['share_available_rows']:.2f}% of "
                         f"available candidate rows, {100 * f['share_eval_population']:.2f}% of "
                         f"the evaluated population")
                ns = x["lock45_v1_no_stage2"]
                L.append(f"    v1 without the stage-2 terms (nmu_scale=0, share_coef=0): MAE "
                         f"{ns['mae']:.3f} bias {ns['bias']:+.3f}")
                d = x["params"]["diagnostics"].get("v1", {})
                L.append(f"    fit: {d.get('n_rounds')} rounds, recon_cv "
                         f"{x['params']['recon_cv']}, nmu_scale {x['params']['nmu_scale']}, "
                         f"k_add {x['params']['k_add']}, share_coef {x['params']['share_coef']};"
                         f" OOF train MAE (fixed pop) {d.get('oof_train_mae_fixed_pop')}; "
                         f"top features {[n for n, _ in d.get('importance_gain_top20', [])[:8]]}")
        for k, v in r["paired"].items():
            L.append(f"- paired {k}: n={v['n']} v0.5 {v['mae_v05']:.3f} v1 {v['mae_v1']:.3f} "
                     f"gain {v['gain']:+.3f} (SE {v['gain_se']:.3f}); bias v0.5 "
                     f"{v['bias_v05']:+.3f} v1 {v['bias_v1']:+.3f}")
        L.append(f"- seconds: fit {r['fit_seconds']}, total {r['seconds']}")
    if ll:
        L.append(f"## live-like slice ({ll['team_games']} team-games, {ll['days']} days, "
                 f"{ll['scenarios']} report scenarios; {ll['seconds_incl_fit']} s incl. fit)")
        for v in ("v0.5", "v1"):
            x = ll["versions"][v]
            L.append(f"- {v}: n={x['n']} MAE {x['mae']:.3f} (SE {x['mae_se']:.3f}) bias "
                     f"{x['bias']:+.3f}; oracle same slice {_m(x['oracle_same_slice'])}; tiers "
                     + "; ".join(f"{k} {t['mae']:.2f}/{t['bias']:+.2f}"
                                 for k, t in x["by_tier"].items()))
    cc = res.get("coach_coverage", {})
    if cc.get("snapshot_check"):
        L.append("## coach by date (N4)")
        L.append(f"- stints: {cc['stints']}; snapshot agreement "
                 f"{100 * cc['snapshot_check']['agree']:.1f}% of {cc['snapshot_check']['n']}"
                 f" (season, team) snapshots; disagreements: "
                 f"{[(d['season'], d['team'], d['snapshot']) for d in cc['snapshot_check']['disagree']]}")
        L.append("- known coach share of regular-season team-games: " + ", ".join(
            f"{s} {v['known']:.3f}" for s, v in cc.items()
            if isinstance(v, dict) and "known" in v and v["known"] is not None))
    L.append(f"## runtime: {res.get('latency')}; load {res['load_seconds']} s, total "
             f"{res['total_seconds']} s")
    return "\n".join(L)


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
    if sys.argv[1:2] == ["v1"]:          # python -m courtside.proj.minutes_eval v1 [seasons]
        from pathlib import Path
        from .. import config
        seasons = tuple(sys.argv[2:]) or ("2023-24", "2024-25")
        out = run_v1(c, seasons, params_dir=config.data_dir() / "proj")
        rep = config.data_dir() / "reports"
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "minutes_v1_eval.json").write_text(json.dumps(out, default=float, indent=1))
        txt = format_v1_report(out)
        (rep / "minutes_v1_eval.txt").write_text(txt + "\n")
        print(txt)
        sys.exit(0)
    seasons = tuple(sys.argv[1:]) or ("2023-24", "2024-25")
    out = run(c, seasons)
    print(format_report(out))
    print(json.dumps({k: v for k, v in out.items() if k != "seasons"}))
