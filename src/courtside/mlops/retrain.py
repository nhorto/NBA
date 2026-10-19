"""Weekly challenger job and the offline/online promotion gates (docs/MLOPS.md section 4).

What a challenger is
--------------------
A production model is a GBM *spec*: fixed hyperparameters (the `gbm.FROZEN` format, which the
daily fit uses) plus the search grid those were chosen from. The walk-forward benchmark never
uses the fixed values (they were picked on all data, test seasons included); it evaluates the
*procedure*: for each test season S the inner time-ordered search runs on seasons < S only.
That is the "inner validation that never touches the benchmark test seasons".

* `search="small"|"medium"`: a wider grid than `gbm.GRID` over the key LightGBM knobs
  (leaves, min leaf size, L2, recency half-life; trees by early stopping). The challenger's
  production params are then re-tuned by the same search on every completed season, exactly how
  `gbm.FROZEN` was produced (`gbm-timing`).
* `params={...}`: a hand-made challenger. FROZEN format (`{"margin": [cfg, n_trees], ...}`, or
  `{"fixed": {...}}`) is evaluated with those values fixed in every fold, against the champion
  with *its* fixed values in every fold (like for like). `{"search_grid": [cfg, ...]}` is
  evaluated as a search procedure.

Only the margin head is benchmarked: the win probability (and so log-loss) is the calibrated
margin. The total head is carried over from the champion unless explicitly given.

Offline gate (all test games, regular season + playoffs, identical game ids)
---------------------------------------------------------------------------
d_i = LL_challenger_i - LL_champion_i, paired SE = sd(d) / sqrt(n).
1. overall: mean(d) < -2 SE (better by more than 2 paired standard errors);
2. most recent test season: mean(d_recent) <= 2 SE_recent (not worse beyond noise);
3. calibration: ECE_challenger - ECE_champion <= 0.005 (10 equal-width bins, all games).
Pass -> registered `candidate`, then `shadow`. Fail -> registered `candidate`, then `retired`
with a `reject` event carrying the reasons. Everything is logged to MLflow.

Online gate (`evaluate_shadows`)
--------------------------------
For each shadow: resolved live games where both the shadow and the champion of its model have a
latest pre-tip prediction, d_i = LL_shadow_i - LL_champion_i.
* n < 100: wait.
* n >= 100, mean(d) <= 0 and no open `critical` alert: promote (old champion retired).
* n >= 300 and mean(d) > 0: reject (retired, `reject` event).
* otherwise (better but a critical alert is open, or worse with n < 300): wait.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..model import metrics
from . import registry, tracking

log = logging.getLogger("courtside.mlops.retrain")

SEASONS_FROM = "2012-13"         # docs/MODEL.md benchmark: test 2012-13 -> last completed season
GATE_SE = 2.0
GATE_ECE_TOL = 0.005
SHADOW_MIN_GAMES = 100
SHADOW_MAX_GAMES = 300


# ------------------------------------------------------------------ search spaces

def search_grid(kind: str = "small") -> list[dict]:
    """Search spaces (GBMConfig dicts). Both contain gbm.GRID, so the champion's own search
    is a subset and its inner fits are shared (memoised) with the challenger's."""
    from ..model import gbm
    if kind == "small":
        combos = [dict(num_leaves=nl, half_life=hl, min_data_in_leaf=md)
                  for nl in (7, 15, 31) for hl in (None, 4.0, 2.0) for md in (300, 100)]
    elif kind == "medium":
        combos = [dict(num_leaves=nl, half_life=hl, min_data_in_leaf=md, lambda_l2=l2)
                  for nl in (7, 15, 31) for hl in (None, 4.0, 2.0) for md in (300, 100)
                  for l2 in (10.0, 50.0)]
        combos += [dict(num_leaves=nl, feature_fraction=0.5, learning_rate=0.03)
                   for nl in (7, 15)]
    else:
        raise ValueError(f"unknown search {kind!r} (small | medium)")
    grid = [gbm.GBMConfig(**c).to_dict() for c in combos]
    for c in gbm.GRID:
        if c.to_dict() not in grid:
            grid.append(c.to_dict())
    return grid


# ------------------------------------------------------------------ metrics

def ll_vec(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """mean(a - b) and its paired standard error."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": int(n), "diff": float(d.mean()) if n else float("nan"), "se": se}


def _summ(y, p, margin=None, margin_pred=None) -> dict:
    if len(y) == 0:
        return {"n": 0}
    out = {"n": int(len(y)), "log_loss": metrics.log_loss(y, p), "brier": metrics.brier(y, p),
           "accuracy": metrics.accuracy(y, p), "ece": metrics.ece(y, p)}
    if margin is not None and margin_pred is not None:
        ok = np.isfinite(np.asarray(margin_pred, float))
        if ok.any():
            out["margin_mae"] = metrics.mae(np.asarray(margin, float)[ok],
                                            np.asarray(margin_pred, float)[ok])
    return out


def summarize(preds: pd.DataFrame, mkt_p: pd.Series | None = None) -> dict:
    """Benchmark summary of one model's per-game walk-forward predictions (columns game_id,
    season, season_type, y, p, margin, margin_pred). `mkt_p`: de-vigged closing P(home) by
    game_id, for the vs-market rows."""
    d = preds.copy()
    seasons = sorted(d["season"].unique())
    rs = d["season_type"] == "Regular Season"
    last = seasons[-1]
    out = {
        "seasons": [seasons[0], last],
        "all": _summ(d["y"], d["p"], d["margin"], d["margin_pred"]),
        "regular_season": _summ(d.loc[rs, "y"], d.loc[rs, "p"], d.loc[rs, "margin"],
                                d.loc[rs, "margin_pred"]),
        "playoffs": _summ(d.loc[~rs, "y"], d.loc[~rs, "p"]),
        "last_season": {"season": last, **_summ(d.loc[d["season"] == last, "y"],
                                                 d.loc[d["season"] == last, "p"])},
        "by_season_log_loss": {s: metrics.log_loss(g["y"], g["p"])
                               for s, g in d.groupby("season")},
        "calibration": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                        for r in metrics.calibration_table(d["y"], d["p"])],
    }
    if mkt_p is not None:
        d["mkt_p"] = d["game_id"].map(mkt_p)
        vm = {}
        last3 = seasons[-3:]
        for key, m in (("regular_season", rs),
                       ("regular_season_last3", rs & d["season"].isin(last3)),
                       ("playoffs", ~rs)):
            x = d[m & d["mkt_p"].notna()]
            if len(x) < 2:
                continue
            a, b = ll_vec(x["y"], x["p"]), ll_vec(x["y"], x["mkt_p"])
            vm[key] = {"n": int(len(x)), "log_loss": float(a.mean()),
                       "market_log_loss": float(b.mean()), "ll_minus_market": paired(a, b)}
        out["vs_market"] = vm
    return out


# ------------------------------------------------------------------ offline gate

def offline_gate(champ: pd.DataFrame, chal: pd.DataFrame, *, se_mult: float = GATE_SE,
                 ece_tol: float = GATE_ECE_TOL) -> dict:
    """The offline gate on per-game predictions (columns game_id, season, y, p) of the champion
    and the challenger. Only games present in both count. Returns {"passed", "reasons",
    "overall", "recent", "ece", "n"}; see the module docstring for the rules."""
    a = champ[["game_id", "season", "y", "p"]].rename(columns={"p": "p_champ"})
    b = chal[["game_id", "p"]].rename(columns={"p": "p_chal"})
    m = a.merge(b, on="game_id", how="inner")
    if len(m) < 2:
        return {"passed": False, "reasons": ["fewer than 2 common benchmark games"], "n": len(m)}
    l_champ, l_chal = ll_vec(m["y"], m["p_champ"]), ll_vec(m["y"], m["p_chal"])
    overall = paired(l_chal, l_champ)
    last = sorted(m["season"].unique())[-1]
    r = (m["season"] == last).to_numpy()
    recent = {"season": last, **paired(l_chal[r], l_champ[r])}
    ece_c, ece_n = metrics.ece(m["y"], m["p_champ"]), metrics.ece(m["y"], m["p_chal"])
    ece = {"champion": ece_c, "challenger": ece_n, "diff": ece_n - ece_c, "tol": ece_tol}
    reasons = []
    if not overall["diff"] < -se_mult * overall["se"]:
        reasons.append(f"overall log-loss diff {overall['diff']:+.5f} is not below "
                       f"-{se_mult:g} SE ({-se_mult * overall['se']:+.5f}, n={overall['n']})")
    if recent["n"] > 1 and recent["diff"] > se_mult * recent["se"]:
        reasons.append(f"worse on {last}: {recent['diff']:+.5f} > {se_mult:g} SE "
                       f"({se_mult * recent['se']:.5f}, n={recent['n']})")
    if ece["diff"] > ece_tol:
        reasons.append(f"ECE worse by {ece['diff']:.4f} (> {ece_tol})")
    return {"passed": not reasons, "reasons": reasons, "n": int(len(m)), "overall": overall,
            "recent": recent, "ece": ece,
            "log_loss": {"champion": float(l_champ.mean()), "challenger": float(l_chal.mean())},
            "rules": {"overall": f"diff < -{se_mult:g} SE", "recent": f"diff <= {se_mult:g} SE",
                      "ece": f"ECE diff <= {ece_tol}"}}


# ------------------------------------------------------------------ params helpers

def _cfg(d: dict):
    from ..model import gbm
    return gbm.GBMConfig(**d)


def challenger_params(champion_params: dict, explicit: dict) -> tuple[dict, str]:
    """Merge an explicit params dict over the champion's -> (params, mode "fixed"|"search").
    Accepts {"fixed": {target: [cfg, n]}}, the bare FROZEN format {target: [cfg, n]} (cfg keys
    override the champion's cfg for that target), and/or {"search_grid": [cfg, ...]}."""
    explicit = dict(explicit or {})
    out = json.loads(json.dumps(champion_params))
    fixed = explicit.pop("fixed", None) or {}
    for t in ("margin", "total"):
        if t in explicit:
            fixed[t] = explicit.pop(t)
    grid = explicit.pop("search_grid", None)
    if explicit:
        raise ValueError(f"unknown params keys {sorted(explicit)} "
                         "(use fixed / margin / total / search_grid)")
    for t, v in fixed.items():
        cfg, n = (v["cfg"], v["n_iter"]) if isinstance(v, dict) and "cfg" in v else v
        base = out["fixed"].get(t, [{}, 0])[0]
        out["fixed"][t] = [_cfg({**base, **cfg}).to_dict(), int(n)]
    if grid is not None:
        out["search_grid"] = [_cfg(c).to_dict() for c in grid]
        return out, "search"
    if not fixed:
        raise ValueError("explicit params need fixed values or a search_grid")
    return out, "fixed"


# ------------------------------------------------------------------ benchmark

def last_completed_season(frame: pd.DataFrame, today=None) -> str:
    today = today or config.today_nba_day()
    start = today.year if today.month >= 7 else today.year - 1
    current = f"{start}-{(start + 1) % 100:02d}"
    done = [s for s in sorted(frame["season"].unique()) if s < current]
    return done[-1]


def walk_forward(frame: pd.DataFrame, factories: dict[str, Callable], seasons: list[str],
                 log_fn: Callable[[str], None] = print) -> pd.DataFrame:
    """Per-game walk-forward predictions: fit on seasons < S, predict S (lean rung3.run_folds:
    no contributions/describe)."""
    from ..model import backtest
    rows = []
    for s, train, test in backtest.walk_forward_splits(frame, seasons):
        t0 = time.time()
        for name, make in factories.items():
            m = make()
            m.fit(train)
            rows.append(pd.DataFrame({
                "model": name, "game_id": test["game_id"].to_numpy(), "season": s,
                "season_type": test["season_type"].to_numpy(),
                "game_date": test["game_date"].to_numpy(), "y": test["home_win"].to_numpy(),
                "margin": test["margin"].to_numpy(),
                "p": np.asarray(m.predict_proba(test), float),
                "margin_pred": np.asarray(m.predict_margin(test), float)}))
        log_fn(f"  {s}: {len(test)} games, {time.time() - t0:.0f}s")
    return pd.concat(rows, ignore_index=True)


def _mkt_p(frame: pd.DataFrame) -> pd.Series | None:
    if "mkt_p" not in frame.columns:
        return None
    return frame.set_index("game_id")["mkt_p"].dropna()


def run_challenger(conn, model_name: str, *, search: str = "small", params: dict | None = None,
                   seasons_from: str = SEASONS_FROM, seasons_to: str | None = None,
                   threads: int = 4, register_result: bool = True, data=None,
                   model_kw: dict | None = None,
                   log_fn: Callable[[str], None] = print) -> dict:
    """Build a challenger for `model_name`, run the offline gate against the champion on the
    frozen walk-forward benchmark, register the outcome and log it to MLflow. `data`:
    optional (frame, players) to skip loading; `model_kw`: extra GBMModel kwargs (tests)."""
    from ..model import gbm, rung3
    t_start = time.time()
    champ = registry.get_champion(conn, model_name)
    if champ is None:
        registry.bootstrap(conn, log_fn=log_fn)
        champ = registry.get_champion(conn, model_name)
    if champ is None:
        raise ValueError(f"no champion for {model_name!r}")
    cparams = champ["params"]
    if params:
        nparams, mode = challenger_params(cparams, params)
    else:
        nparams, mode = json.loads(json.dumps(cparams)), "search"
        nparams["search_grid"] = search_grid(search)
    market = bool(cparams.get("market", gbm.SPECS[model_name]["market"]))

    frame, players = data if data is not None else rung3.load_frame(conn)
    seasons_to = seasons_to or last_completed_season(frame)
    seasons = [s for s in sorted(frame["season"].unique()) if seasons_from <= s <= seasons_to]
    disk = gbm.cache_dir() or (config.data_dir() / "cache")
    _, fc = gbm.make_caches(players, disk_dir=(disk / "engine") if data is None else None)
    t0 = time.time()
    fc.prime(frame[frame["season"] <= seasons_to])
    t_feat = time.time() - t0
    log_fn(f"{model_name}: engine + features {t_feat:.0f}s; mode={mode}; test "
           f"{seasons[0]}..{seasons[-1]}")
    memo = gbm.FitMemo()
    kw = {"market": market, "memo": memo, "targets": ("margin",), "threads": threads,
          **(model_kw or {})}

    def factory(p: dict, how: str):
        if how == "fixed":
            fx = {"margin": (_cfg(p["fixed"]["margin"][0]), int(p["fixed"]["margin"][1]))}
            return lambda: gbm.GBMModel(fc, model_name, fixed=fx, **kw)
        grid = tuple(_cfg(c) for c in p["search_grid"])
        return lambda: gbm.GBMModel(fc, model_name, grid=grid, **kw)

    t0 = time.time()
    preds = walk_forward(frame, {"champion": factory(cparams, mode),
                                 "challenger": factory(nparams, mode)}, seasons, log_fn)
    t_bench = time.time() - t0
    pc, pn = preds[preds["model"] == "champion"], preds[preds["model"] == "challenger"]
    gate = offline_gate(pc, pn)

    t0 = time.time()
    if mode == "search":     # production params: the same search on every completed season
        train = frame[frame["season"] <= seasons_to]
        m = factory(nparams, "search")()
        m.fit(train)
        cfg, k = m.chosen()["margin"]
        nparams["fixed"]["margin"] = [cfg.to_dict(), int(k)]
    t_final = time.time() - t0

    mkt = _mkt_p(frame)
    bench = {"procedure": f"walk-forward {seasons[0]}..{seasons[-1]}, margin head, {mode}"
                          f" ({'search on seasons < S' if mode == 'search' else 'fixed params'})",
             "mode": mode, "summary": summarize(pn, mkt), "champion_summary": summarize(pc, mkt),
             "gate": gate}
    timing = {"engine_features_s": round(t_feat, 1), "benchmark_s": round(t_bench, 1),
              "final_fit_s": round(t_final, 1), "total_s": round(time.time() - t_start, 1)}
    features = registry.feature_list(model_name)
    spec = registry.spec_hash(model_name, nparams, features, registry.code_sha())
    result = {"model_name": model_name, "champion": champ["version_id"], "mode": mode,
              "search": search if not params else None, "params": nparams, "spec_hash": spec,
              "gate": gate, "timing": timing, "seasons": [seasons[0], seasons[-1]],
              "memo": {"hits": memo.hits, "misses": memo.misses}}
    log_fn(f"gate: {'PASS' if gate['passed'] else 'FAIL'} "
           f"overall {gate['overall']['diff']:+.5f} +- {gate['overall']['se']:.5f} "
           f"(n={gate['n']}); {gate['recent']['season']} {gate['recent']['diff']:+.5f} +- "
           f"{gate['recent']['se']:.5f}; ECE {gate['ece']['champion']:.4f} -> "
           f"{gate['ece']['challenger']:.4f}")
    for r in gate["reasons"]:
        log_fn(f"  reject: {r}")

    if spec == champ["spec_hash"]:
        result["status"] = "no_change"
        log_fn("challenger spec equals the champion's; nothing registered")
        return result

    run_id = tracking.log_benchmark(
        f"challenger/{model_name}/{mode}",
        {"model_name": model_name, "mode": mode, "champion": champ["version_id"],
         "params": nparams, "spec_hash": spec, "seasons": f"{seasons[0]}..{seasons[-1]}"},
        {"gate": {k: gate[k] for k in ("overall", "recent", "ece", "log_loss")},
         "passed": float(gate["passed"]), "challenger": bench["summary"],
         "champion": bench["champion_summary"], "timing": timing},
        {"gate": gate, "params": nparams, "preds": preds},
        tags={"kind": "challenger", "model_name": model_name, "passed": gate["passed"]})
    result["mlflow_run_id"] = run_id
    if not register_result:
        result["status"] = "not_registered"
        return result
    vid = registry.register(conn, model_name, nparams, features, status="candidate",
                            benchmark=bench, mlflow_run_id=run_id,
                            notes=f"challenger ({mode}{', ' + search if not params else ''}) "
                                  f"vs {champ['version_id']}")
    result["version_id"] = vid
    cur = registry.get_version(conn, vid)
    if cur["status"] != "candidate":            # an identical spec registered before
        result["status"] = f"already_{cur['status']}"
        return result
    if gate["passed"]:
        registry.set_status(conn, vid, "shadow", reason="passed offline gate",
                            details={"gate": _brief_gate(gate)})
        result["status"] = "shadow"
    else:
        registry.set_status(conn, vid, "retired", reason="; ".join(gate["reasons"]),
                            action="reject", details={"gate": _brief_gate(gate)})
        result["status"] = "rejected"
    log_fn(f"registered {vid}: {result['status']} ({timing['total_s']:.0f}s)")
    return result


def _brief_gate(g: dict) -> dict:
    return {k: g.get(k) for k in ("passed", "reasons", "n", "overall", "recent", "ece")}


# ------------------------------------------------------------------ online gate

def _open_critical_alerts(conn) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM alerts WHERE status='open' "
                            "AND severity='critical'").fetchone()[0]
    except Exception:  # noqa: BLE001 - no alerts table yet
        return 0


def shadow_pairs(conn, version_id: str, model_name: str) -> pd.DataFrame:
    """Resolved live games with a latest pre-tip prediction from both the shadow and the
    champion of `model_name`: game_id, nba_day, y, p_shadow, p_champion."""
    rows = conn.execute(
        "SELECT s.game_id, s.nba_day, s.home_win_prob AS p_shadow, "
        "c.home_win_prob AS p_champion, g.home_score, g.away_score "
        "FROM latest_pretip_predictions_all s "
        "JOIN latest_pretip_predictions c ON c.game_id = s.game_id AND c.model = ? "
        "JOIN games g ON g.game_id = s.game_id AND g.status = 'final' "
        "  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL "
        "WHERE s.model_version = ? AND s.role = 'shadow' "
        "AND g.home_score != g.away_score "
        "AND s.home_win_prob IS NOT NULL AND c.home_win_prob IS NOT NULL "
        "ORDER BY s.nba_day, s.game_id", (model_name, version_id)).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows],
                      columns=["game_id", "nba_day", "p_shadow", "p_champion", "home_score",
                               "away_score"])
    df["y"] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def shadow_decision(n: int, diff: float, critical_alerts: int, *,
                    min_games: int = SHADOW_MIN_GAMES, max_games: int = SHADOW_MAX_GAMES
                    ) -> tuple[str, str]:
    """(action, why) with action in promote | reject | wait. See the module docstring."""
    if n < min_games:
        return "wait", f"{n} of {min_games} resolved games"
    if diff <= 0 and critical_alerts == 0:
        return "promote", f"paired log-loss diff {diff:+.5f} <= 0 over {n} games"
    if diff <= 0:
        return "wait", f"better ({diff:+.5f}) but {critical_alerts} critical alert(s) open"
    if n >= max_games:
        return "reject", f"worse by {diff:+.5f} after {n} games (>= {max_games})"
    return "wait", f"worse by {diff:+.5f} over {n} games; deciding at {max_games}"


def evaluate_shadows(conn, *, dry_run: bool = False, min_games: int = SHADOW_MIN_GAMES,
                     max_games: int = SHADOW_MAX_GAMES,
                     log_fn: Callable[[str], None] = print) -> list[dict]:
    """Apply the online gate to every shadow version; promote/reject unless `dry_run`."""
    out = []
    crit = _open_critical_alerts(conn)
    for v in registry.list_versions(conn, status="shadow"):
        vid, name = v["version_id"], v["model_name"]
        pairs = shadow_pairs(conn, vid, name)
        n = len(pairs)
        st = (paired(ll_vec(pairs["y"], pairs["p_shadow"]), ll_vec(pairs["y"], pairs["p_champion"]))
              if n else {"n": 0, "diff": float("nan"), "se": float("nan")})
        action, why = shadow_decision(n, st["diff"], crit, min_games=min_games,
                                      max_games=max_games)
        rec = {"version_id": vid, "model_name": name, "n": n, "diff": st["diff"],
               "se": st["se"], "critical_alerts": crit, "action": action, "why": why,
               "applied": False}
        details = {k: rec[k] for k in ("n", "diff", "se", "critical_alerts")}
        if not dry_run and action == "promote":
            registry.promote(conn, vid, reason=f"online gate: {why}", details=details)
            rec["applied"] = True
        elif not dry_run and action == "reject":
            registry.set_status(conn, vid, "retired", reason=f"online gate: {why}",
                                action="reject", details=details)
            rec["applied"] = True
        if action != "wait":
            tracking.log_benchmark(f"shadow/{vid}/{action}", {"version_id": vid,
                                                               "model_name": name},
                                   {"n": n, "diff": st["diff"], "se": st["se"]},
                                   tags={"kind": "online_gate", "action": action,
                                         "applied": rec["applied"]})
        log_fn(f"{vid}: {action} ({why})")
        out.append(rec)
    return out

