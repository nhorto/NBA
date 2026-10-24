"""G4 (IMPROVEMENTS): are the opinion model's injury/absence values too big?

Inputs: the G1 per-day dumps (`research/g1_line_news_replay.py`: the day's fitted pair, feature
rows and production injury lineups at eight cutoffs) and, optionally, the G4 dumps
(`research/as_replay.py`: per-player report state + a compact engine state at the same
cutoffs), from which the lineups are rebuilt under other P(play) maps. A database with ESPN
open/close lines and results. Light: no fitting, only re-scoring.

Checks (all 2025-26, official report archive; date-clustered SEs):

1. Line move (G1 Phase 1, reproduced): close - open on delta = opinion margin(tip lineups) -
   opinion margin(guarded lineups at the open proxy d0_03).
2. (a) Outcomes: margin - open and margin - close on the same delta. If our values are right the
   slopes are ~1 and ~0.55 (1 - 0.45); if the market is right ~0.45 and ~0.
   Level form (more power: standing absences, not just news): A = opinion margin(tip lineups)
   - opinion margin(full recent roster) ("absence component"); margin - opinion margin(full)
   on A gives k (the truth's scale of our values), close - opinion(full) on A gives m (the
   market's), margin - close on A gives k - m.
3. (b) Placebos: delta permuted within date; delta of the same home team's game on another
   date (randomised timing); and the pre-open news delta (d0_03 vs d1_17), which the open
   should already price.

Walk-forward (--wf): absence scale s on A: lineup_s = roster - s * (roster - lineup), s chosen
on games of earlier days (grid 0..1.5, opinion margin MSE vs outcomes; s = 1 until --min-fit
games). OFF = s 1. Scored: opinion and gbm_market (fed the close, and the open) log loss,
Brier, margin CRPS (Normal, the scorer's oof sd), paired ON - OFF, date / two-way SEs; and the
de-vigged close moneyline as the reference.

    uv run python research/as_eval.py --g1 var/tmp/g1/days --db var/courtside.db \\
        --out var/tmp/as/eval.json [--as-days var/tmp/as/days --maps base,rebuild,pp15,cells,cells_ship]
`rebuild` is the base map rebuilt from the G4 dumps (equal to the G1 lineups to ~1e-15, but
LightGBM splits can flip on the last bit, so the other maps are compared with `rebuild`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import production
from courtside.model import dataset
from courtside.proj import metrics as pm

_spec = importlib.util.spec_from_file_location(
    "g1e", Path(__file__).with_name("g1_line_news_eval.py"))
g1e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1e)
ols_cluster, load_lines, ll = g1e.ols_cluster, g1e.load_lines, g1e.ll

CUTS = ("d1_17", "d0_03", "d0_17", "tip")
OPEN = "d0_03"
S_GRID = np.round(np.arange(0.0, 1.501, 0.1), 3)


# --- lineups under alternative P(play) maps ------------------------------------------------

def remap_avail(avail: dict, players: list[dict], mapping, cells: dict | None, team: str):
    """{pid: P} for one team's cover under `mapping`: 'base' (as priced), 'pp15' (players.py's
    STATUS_P_PLAY: Doubtful 0.15), 'cells' (the DFS side's calibrated cell P for the listed
    players; players the slate does not have keep the base P)."""
    if mapping in ("base", "rebuild"):
        return dict(avail)
    # the status behind each priced P: the record whose P is the one priced (a batch can hold
    # two rows for a player, e.g. both nights of a back-to-back; `avail` keeps the min)
    status = {}
    for r in players:
        pid = r.get("player_id")
        if pid is not None and pid in avail and r.get("p_play") == avail[pid]:
            status.setdefault(pid, (r.get("status") or "").strip().lower())
    out = {}
    for pid, p in avail.items():
        st = status.get(pid, "")
        if mapping == "pp15":
            from courtside.model.players import STATUS_P_PLAY
            out[pid] = STATUS_P_PLAY.get(st, p) if p > 0 else p
            continue
        c = (cells or {}).get((team, int(pid)))
        # cells: the calibrated P for a listed, not-Out player, only when the slate read the same
        # status (otherwise it used another row/report: keep the priced P)
        same = c is not None and c[1] == {"game time decision": "questionable",
                                          "gtd": "questionable"}.get(st, st)
        out[pid] = c[0] if (same and p > 0 and st != "out") else p
    return out


def as_lineups(rec: dict, mapping: str) -> dict:
    """Rebuild {cut: (home, away)} lineup strengths from a G4 dump under `mapping`."""
    st, test = rec["state"], rec["test"]
    out = {}
    for c in CUTS:
        lh = np.full(len(test), np.nan)
        la = np.full(len(test), np.nan)
        for i, r in test.iterrows():
            cv = rec["cover"][c].get(r["game_id"])
            if not cv or not cv["teams"]:
                continue
            cells = rec["cells" if mapping == "cells" else "cells_ship"][c].get(r["game_id"])
            for side, arr in (("home", lh), ("away", la)):
                t = cv["teams"].get(r[f"{side}_raw"])
                if t is None:
                    continue
                av = remap_avail(t["avail"], t["players"], mapping, cells, r[f"{side}_raw"])
                lu = st.expected_lineup(r[side], None, av,
                                        playoff=r["season_type"] != "Regular Season")
                arr[i] = float(lu.attrs["strength"])
        out[c] = (lh, la)
    return out


def news_classes(as_dir: Path, a_: str = OPEN, b_: str = "tip") -> pd.DataFrame:
    """Per game, what kind of report news happened between cutoffs a_ and b_ (guarded sides:
    the team listed at both): 'none'; 'q_resolved' (every player whose P changed was listed
    Questionable / Doubtful / Probable / GTD at a_); 'new_out' (someone unlisted or Available
    at a_ is Out / Doubtful at b_, nothing else); 'mixed'. Also the count of Q-type players
    at a_ and whether any of them was ruled out by b_."""
    qtypes = {"questionable", "doubtful", "probable", "game time decision", "gtd",
              "day-to-day", "day to day"}
    rows = []
    for path in sorted(as_dir.glob("day_*.pkl")):
        r = pickle.loads(path.read_bytes())
        for gid in r["test"]["game_id"]:
            ca, cb = r["cover"][a_].get(gid), r["cover"][b_].get(gid)
            kinds, n_q, q_out, q_in = set(), 0, 0, 0
            for t in ((cb or {}).get("teams") or {}):
                ta = (ca or {}).get("teams", {}).get(t)
                tb = cb["teams"][t]
                if ta is None or ta["source"] is None or tb["source"] is None:
                    continue
                sa = {x["player_id"]: (x["status"] or "").strip().lower()
                      for x in ta["players"] if x["player_id"] is not None}
                for pid in set(ta["avail"]) | set(tb["avail"]):
                    pa, pb = ta["avail"].get(pid, 1.0), tb["avail"].get(pid, 1.0)
                    st = sa.get(pid, "")
                    if st in qtypes:
                        n_q += 1
                        q_out += pb == 0.0
                        q_in += pb == 1.0
                    if pa == pb:
                        continue
                    if st in qtypes:
                        kinds.add("q")
                    elif pa == 1.0 and pb <= 0.15:
                        kinds.add("new_out")
                    else:
                        kinds.add("other")
            lab = ("none" if not kinds else "q_resolved" if kinds == {"q"} else
                   "new_out" if kinds == {"new_out"} else "mixed")
            rows.append({"game_id": gid, "news": lab, "n_q_open": n_q, "q_ruled_out": q_out,
                         "q_cleared": q_in})
    return pd.DataFrame(rows)


def class_checks(D: pd.DataFrame, res: pd.DataFrame, N: pd.DataFrame) -> list[dict]:
    D = D.merge(res, on="game_id", how="left").merge(N, on="game_id", how="left")
    ok = (D["open_m"].notna() & D["close_m"].notna() & D[f"fresh_{OPEN}"] & D["fresh_tip"]
          & D["margin"].notna())
    X = D[ok]
    out = []
    for lab in ("q_resolved", "new_out", "mixed", "none"):
        Z = X[X["news"] == lab]
        if len(Z) < 30:
            continue
        dl = (Z["op_tip"] - Z[f"opg_{OPEN}_tip"]).to_numpy(float)
        if (dl ** 2).sum() == 0:
            out.append({"check": f"class {lab}", "n": len(Z), "b": np.nan, "se": np.nan,
                        "b_int": np.nan, "se_int": np.nan, "r2": np.nan,
                        "expect": f"mean |move| {np.abs(Z['close_m'] - Z['open_m']).mean():.2f}"})
            continue
        cl = Z["day"].to_numpy()
        out.append(_row(f"class {lab}: move ~ delta", (Z["close_m"] - Z["open_m"])
                        .to_numpy(float), dl, cl, f"mean |delta| {np.abs(dl).mean():.2f}"))
        out.append(_row(f"class {lab}: margin-open ~ delta", (Z["margin"] - Z["open_m"])
                        .to_numpy(float), dl, cl, ""))
    return out


# --- per-game frame ------------------------------------------------------------------------

def _om(sc, G, post):
    sc.set_features(G, post)
    return sc.predict_proba(None), sc.predict_margin(None)


def load(g1_dir: Path, lines: pd.DataFrame, as_dir: Path | None, mapping: str,
         scales=S_GRID) -> pd.DataFrame:
    recs = []
    for path in sorted(g1_dir.glob("day_*.pkl")):
        r = pickle.loads(path.read_bytes())
        F, test = r["F"], r["test"]
        op, mk = r["scorers"]["gbm_opinion"], r["scorers"]["gbm_market"]
        post = (test["season_type"].to_numpy() != "Regular Season").astype(float)
        rh = (F["ps_home"] + F["abs_loss_home"]).to_numpy(float)
        ra = (F["ps_away"] + F["abs_loss_away"]).to_numpy(float)
        if mapping == "base":
            raw = {c: r["lineups"][c] for c in CUTS}
        else:
            a = pickle.loads((as_dir / path.name).read_bytes())
            assert list(a["test"]["game_id"]) == list(test["game_id"])
            raw = as_lineups(a, "base" if mapping == "rebuild" else mapping)
        lu = {c: (np.where(np.isfinite(raw[c][0]), raw[c][0], rh),
                  np.where(np.isfinite(raw[c][1]), raw[c][1], ra)) for c in CUTS}
        L = test[["game_id", "home", "away", "season_type"]].copy()
        L["day"] = r["day"]
        L["sd_op"], L["sd_mk"] = op.oof_resid_sd_, mk.oof_resid_sd_
        ln = lines.reindex(L["game_id"]).reset_index(drop=True)
        for c in ln.columns:
            L[c] = ln[c].to_numpy()
        for c in CUTS:
            L[f"fresh_{c}"] = r["fresh"][c]
            L[f"lst_h_{c}"], L[f"lst_a_{c}"] = r["listed"][c][:, 0], r["listed"][c][:, 1]
        _, L["op_full"] = _om(op, production.apply_overrides(F, lineup_home=rh,
                                                             lineup_away=ra), post)
        for c in CUTS:
            _, L[f"op_{c}"] = _om(op, production.apply_overrides(
                F, lineup_home=lu[c][0], lineup_away=lu[c][1]), post)
        # guarded lineups at the earlier cutoffs relative to the later one (G1's rule)
        for a_, b_ in (("d0_03", "tip"), ("d1_17", "d0_03"), ("d0_17", "tip")):
            gh = r["listed"][a_][:, 0] & r["listed"][b_][:, 0]
            ga = r["listed"][a_][:, 1] & r["listed"][b_][:, 1]
            lh = np.where(gh, lu[a_][0], lu[b_][0])
            la = np.where(ga, lu[a_][1], lu[b_][1])
            _, m = _om(op, production.apply_overrides(F, lineup_home=lh, lineup_away=la), post)
            L[f"opg_{a_}_{b_}"] = m
            L[f"opb_{a_}_{b_}"] = L[f"op_{b_}"]
        # absence-scaled scores at run times T
        for T in ("tip", "d0_17"):
            for s in scales:
                sh = rh - s * (rh - lu[T][0])
                sa = ra - s * (ra - lu[T][1])
                G = production.apply_overrides(F, lineup_home=sh, lineup_away=sa)
                p, m = _om(op, G, post)
                L[f"op_p_{T}_{s}"], L[f"op_m_{T}_{s}"] = p, m
                for lab, col in (("close", "close_m"), ("open", "open_m")):
                    Gm = G.copy()
                    Gm["mkt_margin"] = L[col].to_numpy(float)
                    Gm["mkt_total"] = L[f"{lab}_t"].to_numpy(float)
                    p, m = _om(mk, Gm, post)
                    L[f"mk{lab}_p_{T}_{s}"], L[f"mk{lab}_m_{T}_{s}"] = p, m
        recs.append(L)
    return pd.concat(recs, ignore_index=True)


# --- checks ----------------------------------------------------------------------------------

def _row(name, y, x, cl, expect=""):
    r = ols_cluster(y, x, cl)
    r1 = ols_cluster(y, x, cl, intercept=True)
    return {"check": name, "n": r["n"], "b": r["b"], "se": r["se"], "b_int": r1["b"],
            "se_int": r1["se"], "r2": r1["r2"], "expect": expect}


def checks(D: pd.DataFrame, res: pd.DataFrame, seed: int = 7, n_perm: int = 2000) -> list[dict]:
    D = D.merge(res, on="game_id", how="left")
    ok = (D["open_m"].notna() & D["close_m"].notna() & D[f"fresh_{OPEN}"] & D["fresh_tip"]
          & D["margin"].notna())
    X = D[ok].reset_index(drop=True)
    cl = X["day"].to_numpy()
    delta = (X["op_tip"] - X[f"opg_{OPEN}_tip"]).to_numpy(float)
    move = (X["close_m"] - X["open_m"]).to_numpy(float)
    r_open = (X["margin"] - X["open_m"]).to_numpy(float)
    r_close = (X["margin"] - X["close_m"]).to_numpy(float)
    out = [_row("1 move ~ delta (G1)", move, delta, cl, "0.45"),
           _row("2a margin-open ~ delta", r_open, delta, cl, "ours right: 1; market: ~0.45"),
           _row("2a margin-close ~ delta", r_close, delta, cl, "ours right: ~0.55; market: 0")]
    # joint: does delta add to the move?  margin - open = c1 move + c2 delta
    Z = np.column_stack([move, delta])
    b, *_ = np.linalg.lstsq(Z, r_open, rcond=None)
    e = r_open - Z @ b
    XtX = np.linalg.inv(Z.T @ Z)
    meat = sum(np.outer(z, z) for z in (
        (Z[cl == g] * e[cl == g][:, None]).sum(0) for g in pd.unique(cl)))
    V = XtX @ meat @ XtX
    out.append({"check": "2a margin-open ~ move + delta: delta coef", "n": len(e),
                "b": float(b[1]), "se": float(np.sqrt(V[1, 1])), "b_int": float(b[0]),
                "se_int": float(np.sqrt(V[0, 0])), "r2": np.nan,
                "expect": "b_int column = move coef (~1)"})
    # levels: absence component at the tip (standing absences, more spread than the news)
    sel = D["fresh_tip"] & D["margin"].notna()
    Y = D[sel].reset_index(drop=True)
    A = (Y["op_tip"] - Y["op_full"]).to_numpy(float)
    cy = Y["day"].to_numpy()
    out.append(_row("2a level: margin-op(full) ~ A (k)", (Y["margin"] - Y["op_full"])
                    .to_numpy(float), A, cy, "ours right: 1"))
    yc = Y["close_m"].notna().to_numpy()
    out.append(_row("2a level: close-op(full) ~ A (m)", (Y["close_m"] - Y["op_full"])
                    .to_numpy(float)[yc], A[yc], cy[yc], "market scale of our A"))
    out.append(_row("2a level: margin-close ~ A (k-m)", (Y["margin"] - Y["close_m"])
                    .to_numpy(float)[yc], A[yc], cy[yc], "0 if market right"))
    yo = Y["open_m"].notna().to_numpy()
    out.append(_row("2a level: open-op(full) ~ A", (Y["open_m"] - Y["op_full"])
                    .to_numpy(float)[yo], A[yo], cy[yo], ""))
    # placebos
    rng = np.random.default_rng(seed)
    days = cl
    perm_m, perm_o = [], []
    by_day = [np.flatnonzero(days == d) for d in pd.unique(days)]
    for _ in range(n_perm):
        x = delta.copy()
        for idx in by_day:
            x[idx] = delta[rng.permutation(idx)]
        perm_m.append((x @ move) / (x @ x))
        perm_o.append((x @ r_open) / (x @ x))
    for lab, arr, y in (("move", perm_m, move), ("margin-open", perm_o, r_open)):
        arr = np.asarray(arr)
        obs = (delta @ y) / (delta @ delta)
        out.append({"check": f"3b placebo: {lab} ~ delta permuted within date", "n": len(y),
                    "b": float(arr.mean()), "se": float(arr.std()), "b_int": np.nan,
                    "se_int": np.nan, "r2": np.nan,
                    "expect": f"~0; p(|b|>=obs {obs:.2f}) = "
                              f"{float((np.abs(arr) >= abs(obs)).mean()):.4f}"})
    # randomised timing: the same home team's delta from another date
    home = X["home"].to_numpy()
    sm, so = [], []
    for _ in range(200):
        x = np.empty_like(delta)
        for t in pd.unique(home):
            idx = np.flatnonzero(home == t)
            if len(idx) == 1:
                x[idx] = 0.0
                continue
            sh = rng.permutation(len(idx))
            while len(idx) > 1 and np.any(sh == np.arange(len(idx))):
                sh = rng.permutation(len(idx))
            x[idx] = delta[idx[sh]]
        sm.append((x @ move) / (x @ x))
        so.append((x @ r_open) / (x @ x))
    for lab, arr in (("move", sm), ("margin-open", so)):
        arr = np.asarray(arr)
        out.append({"check": f"3b placebo: {lab} ~ same home team's delta, other date",
                    "n": len(delta), "b": float(arr.mean()), "se": float(arr.std()),
                    "b_int": np.nan, "se_int": np.nan, "r2": np.nan, "expect": "~0"})
    # pre-open news (d1_17 -> d0_03): the open should already have it
    pre_ok = ok & D["fresh_d1_17"]
    P = D[pre_ok].reset_index(drop=True)
    dpre = (P["op_d0_03"] - P["opg_d1_17_d0_03"]).to_numpy(float)
    cp = P["day"].to_numpy()
    out.append(_row("3b pre-open news: move ~ delta(d1_17->d0_03)",
                    (P["close_m"] - P["open_m"]).to_numpy(float), dpre, cp, "~0 if open has it"))
    out.append(_row("3b pre-open news: margin-open ~ delta(d1_17->d0_03)",
                    (P["margin"] - P["open_m"]).to_numpy(float), dpre, cp, "~0 if open has it"))
    return out


# --- walk-forward ----------------------------------------------------------------------------

def walk_forward(D: pd.DataFrame, res: pd.DataFrame, T: str, min_fit: int,
                 fixed: float | None = None, scales=S_GRID) -> dict:
    D = D.merge(res, on="game_id", how="left")
    D = D[D[f"fresh_{T}"] & D["margin"].notna()].sort_values(["day", "game_id"])
    D = D.reset_index(drop=True)
    y, mg = D["home_win"].to_numpy(float), D["margin"].to_numpy(float)
    days = D["day"].to_numpy()
    sse = np.column_stack([(mg - D[f"op_m_{T}_{s}"].to_numpy(float)) ** 2 for s in scales])
    s_d = np.ones(len(D))
    for d in np.unique(days):
        prev = days < d
        if fixed is not None:
            s_d[days == d] = fixed
        elif prev.sum() >= min_fit:
            s_d[days == d] = scales[int(np.argmin(sse[prev].sum(0)))]
    pick = {s: (s_d == s) for s in scales}

    def arm(prefix, kind):
        out = np.full(len(D), np.nan)
        for s, m in pick.items():
            if m.any():
                out[m] = D[f"{prefix}_{kind}_{T}_{s}"].to_numpy(float)[m]
        return out
    res_ = {"T": T, "n": int(len(D)), "n_dates": int(len(np.unique(days))),
            "s_last": float(s_d[-1]), "s_mean": float(s_d.mean()),
            "s_insample": float(scales[int(np.argmin(sse.sum(0)))])}
    for prefix, sdcol in (("op", "sd_op"), ("mkclose", "sd_mk"), ("mkopen", "sd_mk")):
        need = D["close_m"].notna() if prefix != "op" else pd.Series(True, index=D.index)
        if prefix == "mkopen":
            need = D["open_m"].notna()
        k = need.to_numpy()
        p_on, m_on = arm(prefix, "p"), arm(prefix, "m")
        p_off = D[f"{prefix}_p_{T}_1.0"].to_numpy(float)
        m_off = D[f"{prefix}_m_{T}_1.0"].to_numpy(float)
        sd = D[sdcol].to_numpy(float)
        l_on, l_off = ll(p_on, y), ll(p_off, y)
        c_on, c_off = pm.crps_normal(m_on, sd, mg), pm.crps_normal(m_off, sd, mg)
        b_on, b_off = (p_on - y) ** 2, (p_off - y) ** 2
        res_[prefix] = {"n": int(k.sum()), "ll_off": float(l_off[k].mean()),
                        "ll_on": float(l_on[k].mean())}
        for lab, a_, b_ in (("ll", l_on, l_off), ("brier", b_on, b_off),
                            ("crps_m", c_on, c_off)):
            r = pm.paired_cluster2((a_ - b_)[k], days[k], D["home"].to_numpy()[k])
            res_[prefix][f"d_{lab}"] = {"mean": r["mean"], "se_date": r["se_c1"],
                                        "se_2way": r["se"]}
        pc = D["close_p"].to_numpy(float)
        kk = k & np.isfinite(pc)
        res_[prefix]["ll_close_ml"] = float(ll(pc[kk], y[kk]).mean())
        res_[prefix]["ll_off_same"] = float(l_off[kk].mean())
    return res_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-days")
    ap.add_argument("--maps", default="base")
    ap.add_argument("--min-fit", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--md")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    lines = load_lines(conn)
    g = dataset.load_games(conn)
    res = pd.DataFrame({"game_id": g["game_id"], "home_win": g["home_win"].astype(float),
                        "margin": g["margin"].astype(float)})
    out = {}
    for mp in a.maps.split(","):
        cache = Path(a.cache) / f"D_{mp}.pkl" if a.cache else None
        if cache and cache.exists():
            D = pd.read_pickle(cache)
        else:
            D = load(Path(a.g1), lines, Path(a.as_days) if a.as_days else None, mp)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                D.to_pickle(cache)
        C = checks(D, res)
        if a.as_days:
            C += class_checks(D, res, news_classes(Path(a.as_days)))
        print(f"== {mp}: {len(D)} games")
        print(pd.DataFrame(C).round(3).to_string())
        W = {}
        for T in ("tip", "d0_17"):
            W[T] = walk_forward(D, res, T, a.min_fit)
            W[T + "_s0.5"] = walk_forward(D, res, T, a.min_fit, fixed=0.5)
            W[T + "_s0.0"] = walk_forward(D, res, T, a.min_fit, fixed=0.0)
        print(json.dumps(W, indent=1, default=float)[:3000])
        out[mp] = {"checks": C, "wf": W}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    if a.md:
        Path(a.md).write_text("\n\n".join(f"## {mp}\n\n{md_checks(v['checks'])}\n\n"
                                          f"{md_wf(v['wf'])}" for mp, v in out.items()) + "\n")
    return 0



# --- markdown (docs/research/absence-scale.md) ---------------------------------------------------

def md_checks(C: list[dict]) -> str:
    lines = ["| # | check | n | b (SE) | b with intercept (SE) | R² | expectation |",
             "|---|---|---|---|---|---|---|"]
    for k, c in enumerate(C, 1):
        bi = "" if not np.isfinite(c["b_int"]) else f"{c['b_int']:.2f} ({c['se_int']:.2f})"
        r2 = "" if not np.isfinite(c["r2"]) else f"{c['r2']:.3f}"
        lines.append(f"| {k} | {c['check']} | {c['n']} | {c['b']:.2f} ({c['se']:.2f}) | {bi} | "
                     f"{r2} | {c['expect']} |")
    return "\n".join(lines)


def md_wf(W: dict) -> str:
    lines = ["| arm (T, s) | model | n | s last | LL OFF | LL ON | ON − OFF LL (date / 2-way) | "
             "ON − OFF Brier (2-way) | ON − OFF CRPS margin (date / 2-way) | close ML LL |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lab = {"op": "gbm_opinion", "mkclose": "gbm_market, close", "mkopen": "gbm_market, open"}
    for k, v in W.items():
        arm = k.replace("_s", ", s = ") if "_s" in k else f"{k}, s walk-forward"
        for p in ("op", "mkclose", "mkopen"):
            x = v[p]
            d, b, c = x["d_ll"], x["d_brier"], x["d_crps_m"]
            lines.append(
                f"| {arm} | {lab[p]} | {x['n']} | {v['s_last']:.1f} | {x['ll_off']:.4f} | "
                f"{x['ll_on']:.4f} | {d['mean']:+.5f} ({d['se_date']:.5f} / {d['se_2way']:.5f}) | "
                f"{b['mean']:+.5f} ({b['se_2way']:.5f}) | {c['mean']:+.4f} ({c['se_date']:.4f} / "
                f"{c['se_2way']:.4f}) | {x['ll_close_ml']:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
