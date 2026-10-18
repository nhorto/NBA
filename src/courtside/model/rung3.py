"""Rung-3 walk-forward evaluation: GBM variants vs rung 2 and the closing line.

`run(conn, ...)` fits every model on seasons < S and predicts season S (backtest contract),
then reports probability metrics, margin and total MAE, calibration, ATS / over-under records
with Wilson intervals at edge thresholds, playoff results, feature contributions and timings.
Read-only on the database; writes only to `out_dir`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest, features as ft, gbm, metrics, player_model, players as pl

EDGES = (0.0, 2.0, 3.0, 4.0)
BREAK_EVEN = 110 / 210          # -110 both sides


def connect_ro(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_frame(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Game frame with schedule, box (h_*/a_*) and market columns; and the player rows."""
    frame = backtest.build_frame(conn)
    frame = ft.attach_box(frame, ft.load_team_box(conn))
    frame = ft.attach_market(frame, ft.load_closing_full(conn))
    return frame, pl.load_player_games(conn)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def record(pred: np.ndarray, line: np.ndarray, actual: np.ndarray, edges=EDGES) -> list[dict]:
    """Against-the-line record: pick the side our number favours vs the line (home/over if
    pred > line). Pushes and pred == line are excluded. Wilson 95% CI; p-value vs 50%."""
    from scipy.stats import binomtest
    ok = np.isfinite(pred) & np.isfinite(line) & np.isfinite(actual)
    out = []
    for e in edges:
        diff = pred - line
        res = actual - line
        live = ok & (np.abs(diff) > e) & (res != 0) & (diff != 0)
        win = (np.sign(diff) == np.sign(res))[live]
        n, k = int(live.sum()), int(win.sum())
        lo, hi = wilson(k, n)
        out.append({"edge_gt": e, "n": n, "wins": k, "rate": k / n if n else None,
                    "ci95": [lo, hi],
                    "p_vs_50": float(binomtest(k, n, 0.5).pvalue) if n else None,
                    "beats_break_even": bool(n and lo > BREAK_EVEN)})
    return out


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def _ll_vec(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run_folds(frame: pd.DataFrame, factories: dict[str, Callable], seasons: list[str],
              log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict, dict]:
    rows, desc, contrib = [], {}, {}
    timing: dict = {}
    for s, train, test in backtest.walk_forward_splits(frame, seasons):
        for name, make in factories.items():
            t0 = time.time()
            m = make()
            m.fit(train)
            t1 = time.time()
            p = np.asarray(m.predict_proba(test), float)
            mp = np.asarray(m.predict_margin(test), float)
            tp = (np.asarray(m.predict_total(test), float) if hasattr(m, "predict_total")
                  else np.full(len(test), np.nan))
            t2 = time.time()
            timing.setdefault(name, []).append({"season": s, "fit_s": t1 - t0,
                                                "predict_s": t2 - t1})
            rows.append(pd.DataFrame({
                "model": name, "game_id": test["game_id"].to_numpy(), "season": s,
                "season_type": test["season_type"].to_numpy(),
                "game_date": test["game_date"].to_numpy(), "y": test["home_win"].to_numpy(),
                "margin": test["margin"].to_numpy(),
                "total": (test["home_pts"] + test["away_pts"]).to_numpy(),
                "p": p, "margin_pred": mp, "total_pred": tp}))
            if hasattr(m, "describe"):
                desc.setdefault(name, {})[s] = m.describe()
            if hasattr(m, "contributions"):
                C = m.contributions(test)
                rs = (test["season_type"] == "Regular Season").to_numpy()
                a = C[rs].abs()
                c = contrib.setdefault(name, {"sum": pd.Series(dtype=float), "n": 0})
                c["sum"] = c["sum"].add(a.sum(), fill_value=0.0)
                c["n"] += int(rs.sum())
        log(f"  {s}: {len(test)} games ({time.strftime('%H:%M:%S')})")
    preds = pd.concat(rows, ignore_index=True)
    imp = {n: (c["sum"] / max(c["n"], 1)).drop("_bias", errors="ignore")
           .sort_values(ascending=False).to_dict() for n, c in contrib.items()}
    return preds, desc, {"importance_mean_abs_pts": imp, "timing": timing}


def evaluate(preds: pd.DataFrame, frame: pd.DataFrame, last_n: int = 3) -> dict:
    mk = frame.set_index("game_id")[["mkt_margin", "mkt_total", "mkt_p"]]
    d = preds.join(mk, on="game_id")
    ev = backtest.evaluate(d[["model", "game_id", "season", "season_type", "game_date", "y",
                              "margin", "p", "margin_pred"]], last_n)
    seasons = sorted(d["season"].unique())
    last = seasons[-last_n:]
    out = {"probability": ev, "market": {}, "ats": {}, "ou": {}, "totals": {}, "paired": {}}
    names = list(dict.fromkeys(d["model"]))
    for seg in ("Regular Season", "Playoffs"):
        for scope, ss in (("all", seasons), (f"last{last_n}", last)):
            dd = d[(d["season_type"] == seg) & d["season"].isin(ss) & d["mkt_p"].notna()]
            if dd.empty:
                continue
            ref = dd[dd["model"] == names[0]]
            y, pm = ref["y"].to_numpy(), ref["mkt_p"].to_numpy()
            tab = {"n": int(len(ref)), "market": {
                "log_loss": metrics.log_loss(y, pm), "brier": metrics.brier(y, pm),
                "accuracy": metrics.accuracy(y, pm), "ece": metrics.ece(y, pm),
                "margin_mae": metrics.mae(ref["margin"][ref["mkt_margin"].notna()],
                                          ref["mkt_margin"].dropna())
                if ref["mkt_margin"].notna().any() else None}}
            for n in names:
                x = dd[dd["model"] == n]
                tab[n] = {"log_loss": metrics.log_loss(x["y"], x["p"]),
                          "brier": metrics.brier(x["y"], x["p"]),
                          "accuracy": metrics.accuracy(x["y"], x["p"]),
                          "ece": metrics.ece(x["y"], x["p"]),
                          "margin_mae": metrics.mae(x["margin"], x["margin_pred"]),
                          "ll_minus_market": _paired(_ll_vec(x["y"].to_numpy(), x["p"].to_numpy()),
                                                     _ll_vec(x["y"].to_numpy(), x["mkt_p"].to_numpy()))}
            out["market"][f"{seg}|{scope}"] = tab
    rs = d[d["season_type"] == "Regular Season"]
    # paired log-loss differences vs the rung-2 champion on all RS games
    if "ps_elo_sched_mh" in names:
        b = rs[rs["model"] == "ps_elo_sched_mh"].set_index("game_id")
        for n in names:
            x = rs[rs["model"] == n].set_index("game_id").loc[b.index]
            out["paired"][n] = _paired(_ll_vec(x["y"].to_numpy(), x["p"].to_numpy()),
                                       _ll_vec(b["y"].to_numpy(), b["p"].to_numpy()))
    for n in names:
        x = rs[rs["model"] == n]
        if x["margin_pred"].notna().any():
            out["ats"][n] = {
                "all": record(x["margin_pred"].to_numpy(), x["mkt_margin"].to_numpy(),
                              x["margin"].to_numpy(float)),
                f"last{last_n}": record(*(x[x["season"].isin(last)][c].to_numpy(float)
                                          for c in ("margin_pred", "mkt_margin", "margin")))}
        if x["total_pred"].notna().any():
            xt = x[x["total_pred"].notna()]
            out["ou"][n] = {
                "all": record(xt["total_pred"].to_numpy(), xt["mkt_total"].to_numpy(),
                              xt["total"].to_numpy(float)),
                f"last{last_n}": record(*(xt[xt["season"].isin(last)][c].to_numpy(float)
                                          for c in ("total_pred", "mkt_total", "total")))}
            tot = {}
            for scope, ss in (("all", seasons), (f"last{last_n}", last)):
                a = xt[xt["season"].isin(ss)]
                w = a[a["mkt_total"].notna()]
                tot[scope] = {"n": int(len(a)), "mae": metrics.mae(a["total"], a["total_pred"]),
                              "rmse": metrics.rmse(a["total"], a["total_pred"]),
                              "n_line": int(len(w)),
                              "mae_on_line_games": metrics.mae(w["total"], w["total_pred"])
                              if len(w) else None,
                              "market_mae": metrics.mae(w["total"], w["mkt_total"])
                              if len(w) else None,
                              "mae_vs_line": metrics.mae(w["mkt_total"], w["total_pred"])
                              if len(w) else None}
            out["totals"][n] = tot
    return out


def config_summary(desc: dict) -> dict:
    out = {}
    for n, by in desc.items():
        rows = []
        for s, d in by.items():
            if "margin" not in d:
                continue
            r = {"season": s, "calibrator": d["calibrator"]["kind"]}
            for t in ("margin", "total"):
                if t in d:
                    r[f"{t}_n_iter"] = d[t]["n_iter"]
                    r[f"{t}_leaves"] = d[t]["cfg"]["num_leaves"]
                    r[f"{t}_half_life"] = d[t]["cfg"]["half_life"]
            rows.append(r)
        if rows:
            out[n] = rows
    return out


def format_report(res: dict, last_n: int = 3) -> str:
    L = []
    ev = res["probability"]
    names = list(ev)
    w = max(len(n) for n in names) + 2
    for seg in ("Regular Season", "Playoffs"):
        for scope in ("aggregate", f"last{last_n}"):
            L.append(f"\n{seg} | {scope} (all games)")
            L.append(f"{'model':<{w}}{'n':>6}{'LL':>8}{'Brier':>8}{'acc':>7}{'ECE':>7}{'MAE':>7}")
            for n in names:
                m = ev[n][scope].get(seg)
                if m:
                    L.append(f"{n:<{w}}{m['n']:>6}{m['log_loss']:>8.4f}{m['brier']:>8.4f}"
                             f"{m['accuracy']:>7.3f}{m['ece']:>7.3f}"
                             f"{m.get('margin_mae', float('nan')):>7.2f}")
    for key, tab in res["market"].items():
        L.append(f"\n{key} | games with a closing moneyline (n={tab['n']})")
        L.append(f"{'model':<{w}}{'LL':>8}{'Brier':>8}{'acc':>7}{'ECE':>7}{'MAE':>7}"
                 f"{'LL-mkt':>9}{'se':>7}")
        m = tab["market"]
        L.append(f"{'market':<{w}}{m['log_loss']:>8.4f}{m['brier']:>8.4f}{m['accuracy']:>7.3f}"
                 f"{m['ece']:>7.3f}{(m['margin_mae'] or float('nan')):>7.2f}")
        for n in names:
            x = tab[n]
            L.append(f"{n:<{w}}{x['log_loss']:>8.4f}{x['brier']:>8.4f}{x['accuracy']:>7.3f}"
                     f"{x['ece']:>7.3f}{x['margin_mae']:>7.2f}"
                     f"{x['ll_minus_market']['diff']:>+9.4f}{x['ll_minus_market']['se']:>7.4f}")
    if res["paired"]:
        L.append("\nRS log-loss minus ps_elo_sched_mh (paired, all test games)")
        for n, v in res["paired"].items():
            L.append(f"  {n:<{w}}{v['diff']:+.4f} +- {v['se']:.4f}")
    for kind in ("ats", "ou"):
        for n, r in res[kind].items():
            for scope, rows in r.items():
                L.append(f"\n{kind.upper()} {n} ({scope}): edge, n, rate, 95% CI")
                for x in rows:
                    if x["n"]:
                        L.append(f"  >{x['edge_gt']:.0f}  {x['n']:>6}  {x['rate']:.3f}  "
                                 f"[{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]  p={x['p_vs_50']:.3f}")
    for n, t in res["totals"].items():
        for scope, v in t.items():
            L.append(f"\nTOTAL {n} ({scope}): MAE {v['mae']:.2f} (n={v['n']}); on line games "
                     f"{v['mae_on_line_games'] or float('nan'):.2f} vs market "
                     f"{v['market_mae'] or float('nan'):.2f} (n={v['n_line']})")
    return "\n".join(L)


def run(conn, seasons_from: str = "2012-13", seasons_to: str | None = None,
        models: list[str] | None = None, ablations: bool = True, out_dir: Path | None = None,
        threads: int = 4, log: Callable[[str], None] = print) -> dict:
    t_start = time.time()
    frame, players = load_frame(conn)
    seasons = [s for s in backtest.seasons_in(frame)
               if s >= seasons_from and (seasons_to is None or s <= seasons_to)]
    out_dir = Path(out_dir or gbm.cache_dir() or ".")
    eng, fc = gbm.make_caches(players, disk_dir=out_dir / "engine")
    t0 = time.time()
    fc.prime(frame[frame["season"] <= seasons[-1]])
    t_feat = time.time() - t0
    log(f"engine + features: {t_feat:.1f}s (engine runs this process: {eng.n_runs})")
    memo = gbm.FitMemo()
    reg, _ = gbm.make_registry(players, ablations=ablations, fcache=fc, memo=memo,
                               frozen=False, threads=threads)
    reg["ps_elo_sched_mh"] = lambda: player_model.PlayerStrengthModel(
        eng, "ps_elo_sched_mh", "ps_elo_sched", "played", "margin")
    reg["ps_elo_sched"] = lambda: player_model.PlayerStrengthModel(
        eng, "ps_elo_sched", "ps_elo_sched", "played", "logit")
    # rung 2 on the leak-safer availability (apples-to-apples with the GBMs)
    reg["ps_elo_sched_mh_avail"] = lambda: player_model.PlayerStrengthModel(
        eng, "ps_elo_sched_mh_avail", "ps_elo_sched", "avail", "margin")
    reg["ps_elo_sched_avail_tight"] = lambda: player_model.PlayerStrengthModel(
        eng, "ps_elo_sched_avail_tight", "ps_elo_sched", "avail_tight", "logit")
    order = ["ps_elo_sched_mh", "ps_elo_sched", "ps_elo_sched_mh_avail",
             "ps_elo_sched_avail_tight", *[n for n in reg if n.startswith("gbm")]]
    names = [n for n in order if models is None or n in models]
    log(f"testing {seasons[0]}..{seasons[-1]} with {names}")
    preds, desc, extra = run_folds(frame, {n: reg[n] for n in names}, seasons, log)
    res = evaluate(preds, frame)
    res["configs"] = config_summary(desc)
    res.update(extra)
    res["memo"] = {"hits": memo.hits, "misses": memo.misses}
    res["wall_time_s"] = time.time() - t_start
    res["feature_build_s"] = t_feat
    print(format_report(res))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rung3_{ts}.json"
    doc = {"created_utc": ts, "seasons": [seasons[0], seasons[-1]], "results": res,
           "models": desc}
    path.write_text(json.dumps(doc, indent=1, default=backtest._json_default))
    preds.to_csv(out_dir / f"rung3_preds_{ts}.csv.gz", index=False)
    log(f"report: {path}")
    return res


def merge(paths: list, frame: pd.DataFrame, out_dir: Path) -> dict:
    """Combine reports of parallel runs over disjoint model sets (same seasons), re-evaluate
    on the pooled predictions and write one combined report."""
    preds, desc, imp, timing = [], {}, {}, {}
    for p in map(Path, paths):      # later reports override earlier ones model by model
        doc = json.loads(p.read_text())
        pr = pd.read_csv(p.with_name(p.name.replace("rung3_", "rung3_preds_")
                                     .replace(".json", ".csv.gz")),
                         dtype={"game_id": str}, parse_dates=["game_date"])
        new = set(pr["model"])
        preds = [x[~x["model"].isin(new)] for x in preds]
        preds.append(pr)
        desc.update(doc["models"])
        imp.update(doc["results"].get("importance_mean_abs_pts", {}))
        timing.update(doc["results"].get("timing", {}))
    preds = pd.concat(preds, ignore_index=True)
    order = ["ps_elo_sched_mh", "ps_elo_sched", "ps_elo_sched_mh_avail",
             "ps_elo_sched_avail_tight"]
    names = [n for n in order if n in set(preds["model"])] + sorted(
        n for n in set(preds["model"]) if n not in order)
    preds = pd.concat([preds[preds["model"] == n] for n in names], ignore_index=True)
    res = evaluate(preds, frame)
    res["configs"] = config_summary(desc)
    res["importance_mean_abs_pts"] = imp
    res["timing"] = timing
    print(format_report(res))
    out_dir = Path(out_dir)
    path = out_dir / "rung3_merged.json"
    path.write_text(json.dumps({"results": res, "models": desc}, indent=1,
                               default=backtest._json_default))
    preds.to_csv(out_dir / "rung3_merged_preds.csv.gz", index=False)
    print(f"report: {path}")
    return res


def production_timing(conn, threads: int = 4, log: Callable[[str], None] = print) -> dict:
    """Simulate the morning job on the last game day in the data: fresh caches (no disk), one
    engine pass over history + that day, then (a) a full fit with the inner search, which
    also yields the configs to freeze, and (b) the daily fit with those configs frozen, then
    predict + explain."""
    t0 = time.time()
    frame, players = load_frame(conn)
    out = {"load_s": time.time() - t0}
    day = frame["game_date"].max()
    train, test = frame[frame["game_date"] < day], frame[frame["game_date"] == day]
    out.update(n_train=int(len(train)), n_test=int(len(test)))
    eng = gbm.DiskEngineCache(players, pl.TUNED, gbm.ENGINE_MODES, None)
    fc = gbm.FeatureCache(eng)
    for name, spec in gbm.SPECS.items():
        m = gbm.GBMModel(fc, name, threads=threads, **spec)
        t1 = time.time()
        m.prepare(train, test)
        t2 = time.time()
        m.fit(train)
        t3 = time.time()
        chosen = m.chosen()
        f = gbm.GBMModel(fc, name, threads=threads, fixed=chosen, **spec)
        f.fit(train)
        t4 = time.time()
        f.predict_proba(test), f.predict_margin(test), f.predict_total(test)
        ex = f.explain(test.iloc[0], test)
        t5 = time.time()
        out[name] = {"engine_and_features_s": t2 - t1, "fit_with_search_s": t3 - t2,
                     "fit_frozen_s": t4 - t3, "predict_explain_s": t5 - t4,
                     "chosen": {t: [c.to_dict(), k] for t, (c, k) in chosen.items()},
                     "example_game": test["game_id"].iloc[0], "example": ex}
        log(f"{name}: engine+features {t2 - t1:.1f}s, fit+search {t3 - t2:.1f}s, frozen fit "
            f"{t4 - t3:.1f}s, predict+explain {t5 - t4:.2f}s; chosen {out[name]['chosen']}")
    out["total_s"] = time.time() - t0
    return out
