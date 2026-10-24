"""G1 (IMPROVEMENTS §4) analysis: does the opinion model's absence value explain line moves, and
does a "news since the line" shift help a market model that is fed a stale line?

Inputs: the per-day dumps of `research/g1_line_news_replay.py` (fitted pair, feature rows and
injury lineups at eight cutoffs per game) and a database with ESPN open/close lines
(`closing_lines`, source 'espn') and results. Light: no fitting, only re-scoring.

Phase 1 (slope). For each game with an ESPN open and close spread:
    move  = close home margin line - open home margin line      (= open spread - close spread)
    delta = opinion margin(lineups at the tip) - opinion margin(lineups at the open proxy)
and `move ~ b * delta` (through the origin, and with an intercept), date-clustered SEs. b = 1:
the market moved by what our opinion model says the news between open and close is worth.
Open proxies are the replay cutoffs (the ESPN open has no timestamp). Two lineup rules:
`raw` = production `injuries.availability` semantics at both times (a team the newest batch
does not list is healthy), `guard` = a side counts only if the newest batch listed it at both
times (else its tip lineup is used at the open too, i.e. no news for that side); `guard` is what
the live switch does. The same for totals with the opinion total.

Phase 2 (walk-forward). The market model (`gbm_market`) is fed the OPEN line (the stale line a
morning run sees; no intraday snapshots exist for 2025-26) and the injury lineups at the run
time T (tip, or 21:00Z = 4-5 PM ET). OFF = as production prices it today. ON = the open line
shifted by b_d * delta(T vs open proxy) (guarded), b_d fitted on games of earlier days only
(no-intercept OLS, expanding window, b_d = 0 until `--min-fit` games). References: the same
model fed the close (what replays assume), and the de-vigged open / close moneylines.
Scores: log loss, Brier, and CRPS of margin and total under Normal(mu, sigma): sigma_margin =
the day's `oof_resid_sd_`, sigma_total = sd of OFF total residuals on earlier days (same sigma
for both arms). Paired ON - OFF differences with date-clustered and two-way (date x home team,
`proj.metrics.paired_cluster2`) SEs.

    uv run python research/g1_line_news_eval.py --days var/tmp/g1/days --db var/courtside.db \\
        --out var/tmp/g1/eval.json --md var/tmp/g1/tables.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import production
from courtside.model import dataset, gbm, market as market_mod
from courtside.proj import metrics as pm

CUTOFFS = ("d1_17", "d1_20", "d1_23", "d0_03", "d0_17", "d0_21", "tip90", "tip")
OPEN_PROXIES = CUTOFFS[:-2]
PRIMARY_OPEN = "d0_03"          # pre-registered: the evening before (and the morning-run state)


# --- loading -----------------------------------------------------------------------------------

def _score(sc, G, post, total=True):
    sc.set_features(G, post)
    p = sc.predict_proba(None)
    m = sc.predict_margin(None)
    t = sc.predict_total(None) if total else np.full(len(G), np.nan)
    return p, m, t


def _with_line(F, mm, mt):
    G = F.copy()
    G["mkt_margin"] = np.asarray(mm, float)
    G["mkt_total"] = np.asarray(mt, float)
    return G


def load_days(days_dir: Path, lines: pd.DataFrame) -> pd.DataFrame:
    """One row per game: opinion margin/total at every cutoff (raw and guarded vs the tip and
    vs d0_21), listed flags, the market model's outputs for every (line, T) combination needed
    except the ON arm (which needs b_d: those are scored in `walk_forward` from the stored
    per-day scorers, kept in `_SCORERS`)."""
    recs = []
    for path in sorted(days_dir.glob("day_*.pkl")):
        r = pickle.loads(path.read_bytes())
        F, test = r["F"], r["test"]
        op, mk = r["scorers"]["gbm_opinion"], r["scorers"]["gbm_market"]
        post = (test["season_type"].to_numpy() != "Regular Season").astype(float)
        rh = (F["ps_home"] + F["abs_loss_home"]).to_numpy(float)
        ra = (F["ps_away"] + F["abs_loss_away"]).to_numpy(float)
        lu = {}
        for c in CUTOFFS:
            lh, la = r["lineups"][c]
            lu[c] = (np.where(np.isfinite(lh), lh, rh), np.where(np.isfinite(la), la, ra))
        L = test[["game_id", "home", "away", "season_type", "tip"]].copy()
        L["day"] = r["day"]
        L["sigma_m"] = mk.oof_resid_sd_
        ln = lines.reindex(L["game_id"]).reset_index(drop=True)
        for c in ln.columns:
            L[c] = ln[c].to_numpy()
        for c in CUTOFFS:
            G = production.apply_overrides(F, lineup_home=lu[c][0], lineup_away=lu[c][1])
            _, m, t = _score(op, G, post)
            L[f"op_m_{c}"], L[f"op_t_{c}"] = m, t
            L[f"lst_h_{c}"] = r["listed"][c][:, 0]
            L[f"lst_a_{c}"] = r["listed"][c][:, 1]
            L[f"fresh_{c}"] = r["fresh"][c]
        # guarded opinion values at each open proxy, relative to run time T in {tip, d0_21}
        for T in ("tip", "d0_21"):
            for c in OPEN_PROXIES:
                if c == T:
                    continue
                gh = r["listed"][c][:, 0] & r["listed"][T][:, 0]
                ga = r["listed"][c][:, 1] & r["listed"][T][:, 1]
                lh = np.where(gh, lu[c][0], lu[T][0])
                la = np.where(ga, lu[c][1], lu[T][1])
                G = production.apply_overrides(F, lineup_home=lh, lineup_away=la)
                _, m, t = _score(op, G, post)
                L[f"opg_m_{c}_{T}"], L[f"opg_t_{c}_{T}"] = m, t
        # market model: open or close line, lineups at T
        for T in ("tip", "d0_21"):
            G = production.apply_overrides(F, lineup_home=lu[T][0], lineup_away=lu[T][1])
            for lab, mm, mt in (("open", L["open_m"], L["open_t"]),
                                ("close", L["close_m"], L["close_t"])):
                p, m, t = _score(mk, _with_line(G, mm, mt), post)
                L[f"mk_p_{lab}_{T}"], L[f"mk_m_{lab}_{T}"], L[f"mk_t_{lab}_{T}"] = p, m, t
        _SCORERS[r["day"]] = (mk, F, lu, post)
        recs.append(L)
    return pd.concat(recs, ignore_index=True)


_SCORERS: dict[str, tuple] = {}


def load_lines(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT game_id, home_spread_open, home_spread_close, total_open, total_close, "
        "home_ml_open, away_ml_open, home_ml_close, away_ml_close FROM closing_lines "
        "WHERE source='espn'", conn).set_index("game_id")

    def devig(h, a):
        ok = (h.abs() >= 100) & (a.abs() >= 100)
        ph = market_mod.american_to_prob(h.where(ok))
        pa = market_mod.american_to_prob(a.where(ok))
        return np.where(ok, ph / (ph + pa), np.nan)
    return pd.DataFrame({
        "open_m": -df["home_spread_open"], "close_m": -df["home_spread_close"],
        "open_t": df["total_open"], "close_t": df["total_close"],
        "open_p": devig(df["home_ml_open"], df["away_ml_open"]),
        "close_p": devig(df["home_ml_close"], df["away_ml_close"])}, index=df.index)


# --- statistics ----------------------------------------------------------------------------------

def ols_cluster(y, x, cl, intercept=False):
    """OLS of y on x (optionally with an intercept), date-clustered sandwich SE of the slope."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in pd.unique(cl):
        s = (X[cl == g] * e[cl == g][:, None]).sum(axis=0)
        meat += np.outer(s, s)
    G = len(pd.unique(cl))
    V = XtX_inv @ meat @ XtX_inv * G / max(G - 1, 1)
    j = 1 if intercept else 0
    r2 = 1 - (e ** 2).sum() / ((y - (y.mean() if intercept else 0)) ** 2).sum()
    return {"b": float(b[j]), "se": float(np.sqrt(V[j, j])), "a": float(b[0]) if intercept
            else 0.0, "r2": float(r2), "n": int(len(y)), "n_dates": int(G)}


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- phase 1 -------------------------------------------------------------------------------------

def phase1(D: pd.DataFrame) -> list[dict]:
    out = []
    ok = D["open_m"].notna() & D["close_m"].notna()
    for target, mcol, lcol in (("margin", "op_m", "m"), ("total", "op_t", "t")):
        for rule in ("raw", "guard"):
            for c in OPEN_PROXIES:
                if rule == "raw":
                    delta = D[f"{mcol}_tip"] - D[f"{mcol}_{c}"]
                    sel = ok & D[f"fresh_{c}"] & D["fresh_tip"]
                else:
                    delta = D[f"{mcol}_tip"] - D[f"opg_{lcol}_{c}_tip"]
                    sel = ok & D[f"fresh_{c}"] & D["fresh_tip"]
                if target == "total":
                    sel = sel & D["open_t"].notna() & D["close_t"].notna()
                move = (D[f"close_{lcol}"] - D[f"open_{lcol}"])[sel].to_numpy(float)
                x = delta[sel].to_numpy(float)
                cl = D.loc[sel, "day"].to_numpy()
                r0 = ols_cluster(move, x, cl)
                r1 = ols_cluster(move, x, cl, intercept=True)
                big = np.abs(x) >= 0.5
                rb = ols_cluster(move[big], x[big], cl[big]) if big.sum() > 30 else {}
                out.append({"target": target, "rule": rule, "open_proxy": c, "n": r0["n"],
                            "b": r0["b"], "se": r0["se"], "b_int": r1["b"], "se_int": r1["se"],
                            "a_int": r1["a"], "r2": r1["r2"],
                            "corr": float(np.corrcoef(move, x)[0, 1]) if x.std() > 0 else np.nan,
                            "mean_abs_move": float(np.abs(move).mean()),
                            "mean_abs_delta": float(np.abs(x).mean()),
                            "share_delta_ge_0.5": float(big.mean()),
                            "b_big": rb.get("b"), "se_big": rb.get("se"),
                            "n_big": int(big.sum())})
    return out


# --- phase 2 -------------------------------------------------------------------------------------

def walk_forward(D: pd.DataFrame, open_proxy: str, T: str, min_fit: int,
                 fixed_b: tuple[float, float] | None = None) -> pd.DataFrame:
    """Per game: OFF / ON market-model outputs fed the open line with lineups at T, b_d fitted
    on earlier days (or `fixed_b`)."""
    D = D[D["open_m"].notna() & D["close_m"].notna() & D[f"fresh_{T}"]
          & D[f"fresh_{open_proxy}"]].copy()
    D = D.sort_values(["day", "game_id"]).reset_index(drop=True)
    dm = (D[f"op_m_{T}"] - D[f"opg_m_{open_proxy}_{T}"]).to_numpy(float)
    dt = (D[f"op_t_{T}"] - D[f"opg_t_{open_proxy}_{T}"]).to_numpy(float)
    # the slope is fit on the realised open->close move against the news open->tip (Phase 1)
    fm = (D["op_m_tip"] - D[f"opg_m_{open_proxy}_tip"]).to_numpy(float)
    ft = (D["op_t_tip"] - D[f"opg_t_{open_proxy}_tip"]).to_numpy(float)
    mv_m = (D["close_m"] - D["open_m"]).to_numpy(float)
    mv_t = (D["close_t"] - D["open_t"]).to_numpy(float)
    days = D["day"].to_numpy()
    bm = np.zeros(len(D))
    bt = np.zeros(len(D))
    for d in np.unique(days):
        cur = days == d
        if fixed_b is not None:
            bm[cur], bt[cur] = fixed_b
            continue
        prev = days < d
        okm = prev & np.isfinite(mv_m) & np.isfinite(fm)
        okt = prev & np.isfinite(mv_t) & np.isfinite(ft)
        if okm.sum() >= min_fit and (fm[okm] ** 2).sum() > 0:
            bm[cur] = np.clip((fm[okm] @ mv_m[okm]) / (fm[okm] @ fm[okm]), 0.0, 1.5)
        if okt.sum() >= min_fit and (ft[okt] ** 2).sum() > 0:
            bt[cur] = np.clip((ft[okt] @ mv_t[okt]) / (ft[okt] @ ft[okt]), 0.0, 1.5)
    D["b_m"], D["b_t"], D["delta_m"], D["delta_t"] = bm, bt, dm, dt
    on_m = D["open_m"].to_numpy(float) + bm * np.nan_to_num(dm)
    on_t = D["open_t"].to_numpy(float) + bt * np.nan_to_num(dt)
    pon, mon, ton = (np.empty(len(D)) for _ in range(3))
    for d in np.unique(days):
        idx = np.flatnonzero(days == d)
        mk, F, lu, post = _SCORERS[d]
        g_of = {g: i for i, g in enumerate(_GAME_IDS[d])}
        rows = [g_of[g] for g in D.loc[idx, "game_id"]]
        Fd = F.iloc[rows].reset_index(drop=True)
        G = production.apply_overrides(Fd, lineup_home=lu[T][0][rows],
                                       lineup_away=lu[T][1][rows])
        p, m, t = _score(mk, _with_line(G, on_m[idx], on_t[idx]), post[rows])
        pon[idx], mon[idx], ton[idx] = p, m, t
    D["on_p"], D["on_m"], D["on_t"] = pon, mon, ton
    D["off_p"], D["off_m"], D["off_t"] = (D[f"mk_p_open_{T}"], D[f"mk_m_open_{T}"],
                                          D[f"mk_t_open_{T}"])
    D["cl_p"], D["cl_m"], D["cl_t"] = (D[f"mk_p_close_{T}"], D[f"mk_m_close_{T}"],
                                       D[f"mk_t_close_{T}"])
    return D


_GAME_IDS: dict[str, list[str]] = {}


def score(D: pd.DataFrame, res: pd.DataFrame) -> dict:
    D = D.merge(res, on="game_id", how="left")
    y = D["home_win"].to_numpy(float)
    mg = D["margin"].to_numpy(float)
    tot = D["total_pts"].to_numpy(float)
    # sigma_total: sd of OFF total residuals on earlier days (expanding; 18 until 100 games)
    days = D["day"].to_numpy()
    r_off = tot - D["off_t"].to_numpy(float)
    st = np.full(len(D), 18.0)
    for d in np.unique(days):
        prev = (days < d) & np.isfinite(r_off)
        if prev.sum() >= 100:
            st[days == d] = np.sqrt(np.mean(r_off[prev] ** 2))
    sm = D["sigma_m"].to_numpy(float)
    arms = {}
    for a in ("off", "on", "cl"):
        arms[a] = {"ll": ll(D[f"{a}_p"], y), "brier": (D[f"{a}_p"].to_numpy() - y) ** 2,
                   "crps_m": pm.crps_normal(D[f"{a}_m"].to_numpy(float), sm, mg),
                   "crps_t": pm.crps_normal(D[f"{a}_t"].to_numpy(float), st, tot)}
    for a, col in (("mkt_open", "open_p"), ("mkt_close", "close_p")):
        p = D[col].to_numpy(float)
        arms[a] = {"ll": ll(p, y), "brier": (p - y) ** 2}
    out = {"n": int(len(D)), "n_dates": int(D["day"].nunique()),
           "share_shifted": float((np.abs(D["b_m"] * np.nan_to_num(D["delta_m"])) > 0.05).mean()),
           "mean_abs_shift": float(np.abs(D["b_m"] * np.nan_to_num(D["delta_m"])).mean()),
           "b_m_last": float(D["b_m"].iloc[-1]), "b_t_last": float(D["b_t"].iloc[-1])}
    pml = np.isfinite(D["open_p"]).to_numpy() & np.isfinite(D["close_p"]).to_numpy()
    for a, v in arms.items():
        for k, x in v.items():
            keep = pml if a.startswith("mkt") else np.isfinite(x)
            out[f"{a}_{k}"] = float(np.nanmean(x[keep]))
    for k in ("ll", "brier", "crps_m", "crps_t"):
        for a, b in (("on", "off"), ("cl", "off")):
            d = arms[a][k] - arms[b][k]
            ok = np.isfinite(d)
            r = pm.paired_cluster2(d[ok], D["day"].to_numpy()[ok], D["home"].to_numpy()[ok])
            out[f"{a}_minus_{b}_{k}"] = {"mean": r["mean"], "se_date": r["se_c1"],
                                         "se_2way": r["se"]}
    for a in ("off", "on", "cl"):
        d = arms[a]["ll"] - arms["mkt_open"]["ll"]
        r = pm.paired_cluster2(d[pml], D["day"].to_numpy()[pml], D["home"].to_numpy()[pml])
        out[f"{a}_minus_mktopen_ll"] = {"mean": r["mean"], "se_date": r["se_c1"],
                                        "se_2way": r["se"], "n": r["n"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-fit", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    lines = load_lines(conn)
    D = load_days(Path(a.days), lines)
    for d, grp in D.groupby("day"):
        _GAME_IDS[d] = list(grp["game_id"])
    g = dataset.load_games(conn)
    res = pd.DataFrame({"game_id": g["game_id"], "home_win": g["home_win"].astype(float),
                        "margin": g["margin"].astype(float),
                        "total_pts": (g["home_pts"] + g["away_pts"]).astype(float)})
    print(f"{len(D)} games, {D['day'].nunique()} days, {D['open_m'].notna().sum()} with an "
          f"ESPN open")
    p1 = phase1(D)
    t1 = pd.DataFrame(p1)
    print(t1.round(3).to_string())
    p2 = {}
    for T in ("tip", "d0_21"):
        for c in OPEN_PROXIES:
            if c == T:
                continue
            W = walk_forward(D, c, T, a.min_fit)
            p2[f"{c}->{T}"] = score(W, res)
            if c == PRIMARY_OPEN:
                for st in ("Regular Season", "Playoffs"):
                    sub = W[W["season_type"] == st]
                    if len(sub) > 50:
                        p2[f"{c}->{T} [{st}]"] = score(sub, res)
                W.to_csv(Path(a.out).with_name(f"wf_{c}_{T}.csv.gz"), index=False)
    t2 = phase2_table(p2)
    print(t2)
    Path(a.out).write_text(json.dumps({"phase1": p1, "phase2": p2}, indent=1, default=float))
    if a.md:
        hdr = "| " + " | ".join(t1.columns) + " |\n|" + "---|" * len(t1.columns) + "\n"
        body = "\n".join("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                             for v in row) + " |"
                          for row in t1.itertuples(index=False))
        Path(a.md).write_text(hdr + body + "\n\n" + t2 + "\n")
    return 0


def _d(v, k, f="{:+.5f}"):
    x = v[k]
    return f"{f.format(x['mean'])} ({x['se_date']:.5f} / {x['se_2way']:.5f})"


def phase2_table(p2: dict) -> str:
    head = ("| scenario | n | dates | b_m (last) | mean abs shift | LL OFF | LL ON | ON-OFF LL "
            "(SE date / 2-way) | ON-OFF Brier | ON-OFF CRPS margin | ON-OFF CRPS total | "
            "LL close-fed | LL open ML | LL close ML | OFF-open ML LL | ON-open ML LL |")
    lines = [head, "|" + "---|" * 17]
    for k, v in p2.items():
        lines.append(
            f"| {k} | {v['n']} | {v['n_dates']} | {v['b_m_last']:.2f} | "
            f"{v['mean_abs_shift']:.3f} | {v['off_ll']:.4f} | {v['on_ll']:.4f} | "
            f"{_d(v, 'on_minus_off_ll')} | {_d(v, 'on_minus_off_brier')} | "
            f"{_d(v, 'on_minus_off_crps_m', '{:+.4f}')} | "
            f"{_d(v, 'on_minus_off_crps_t', '{:+.4f}')} | {v['cl_ll']:.4f} | "
            f"{v['mkt_open_ll']:.4f} | {v['mkt_close_ll']:.4f} | "
            f"{_d(v, 'off_minus_mktopen_ll')} | {_d(v, 'on_minus_mktopen_ll')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
