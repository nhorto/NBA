"""Upper-tail quantile remap (`COURTSIDE_TAIL_REMAP`, `+tr`; IMPROVEMENTS.md D6): fit and evaluation.

Docs: docs/research/tail-remap.md.  Code under test: `proj/tail_remap.py`, its hook in
`pipeline.run_games` (after the role blend).

    python research/tail_remap_eval.py fit ROWS_DIR [--arm ARM]
        per-season tier x minutes-band quantiles q(v) of U_played, walk-forward on the OFF
        harness rows of seasons < S -> src/courtside/proj/tail_remap_params.json
    python research/tail_remap_eval.py harness S PROTO
        backtest rows (4,000 sims, --tails) of the arm in the environment
        -> OUT/rows_<P>_<S>_<MV>.parquet
    python research/tail_remap_eval.py pit LABEL=FILE[,FILE...] ...
        upper-tail PIT shares (P(play) >= 0.85, unconditional randomised PIT, as xfactor.md §2) by
        tier and tier x minutes band, plus the played-conditional shares; SE clustered by date
    python research/tail_remap_eval.py dk OFF ON LABEL
        paired ON - OFF DK CRPS / MAE / bias (two-way date x player SEs), tiers, minutes bands
    python research/tail_remap_eval.py bignight OFF ON [OFF ON ...]
        X factor big-night chance (P(DK >= 6x salary/1000), RotoGuru rows): the live STRETCH map
        and a walk-forward refit on each arm; reliability of the tag set
    python research/tail_remap_eval.py tau S [--dates N]
        per date OFF / ON sims: PIT shares of the builder's scoring truth with TRUTH_TAU 0.18 vs 0
        and the residual-sd criterion TRUTH_TAU came from
    python research/tail_remap_eval.py builder S [--dates N]
        RotoGuru slates: the contest builder (portfolio_backtest setup) per arm (tr 0/1) x tau
        (0.18 / 0): predicted per-lineup P(top 1 %) and P(any top 1 %) vs realised at the actual
        points against the field proxy -> OUT/builder_<S>.csv

Environment: COURTSIDE_DATA (bundles + DB), TR_OUT (default var/tmp/tr/out). Heavy: Mac scratch
(`courtside-compute/tr`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
OUT = Path(os.environ.get("TR_OUT", ROOT / "var" / "tmp" / "tr" / "out"))
ARM = "pathA-v0.3+mf+p2"
# training datasets (OFF rows) -> their season; target S gets every dataset of seasons < S
DATASETS = (("avail", "2018-19"), ("avail", "2019-20"), ("avail", "2020-21"),
            ("avail", "2023-24"), ("avail", "2024-25"), ("live", "2025-26"))
TARGETS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
           "2026-27")
FIT_PPLAY_MIN = 0.85         # played rows with sim P(play) >= this enter the fit (the population
                             # xfactor.md §2 grades; 0.2 / 0.5 left bench ~0.3 pt short, the doc)
P_HI = 0.85
THRESH = (0.90, 0.95, 0.99)


def say(m: str) -> None:
    print(f"[tr] {time.strftime('%H:%M:%S')} {m}", flush=True)


def blackout() -> bool:
    """Scratch-box guard (TR_GUARD=1): True from 17:30 to 19:30 ET (nothing may run on the Mac
    17:45-19:30), checked before each date of the long loops."""
    if os.environ.get("TR_GUARD") != "1":
        return False
    from datetime import datetime
    from zoneinfo import ZoneInfo
    hm = datetime.now(ZoneInfo("America/New_York")).strftime("%H%M")
    return "1730" <= hm < "1930"


def _conn():
    from courtside import db
    return db.connect()


def played_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Harness rows -> played rows with the played-conditional PIT `uc` and E[min | plays]."""
    from courtside.proj import tail_remap as TR
    d = rows[rows["has_proj"].astype(bool) & rows["pit"].notna() & (rows["minutes"] > 0)
             & (rows["p_play"] >= FIT_PPLAY_MIN)].copy()
    d["min_played"] = d["exp_min"] / d["p_play"]
    d["uc"] = TR.cond_pit(d["pit"], d["p_play"])
    return d.reset_index(drop=True)


# ============================================================================== fit

def cmd_fit(rows_dir: str, arm: str) -> None:
    from courtside.proj import tail_remap as TR
    frames = {}
    for proto, s in DATASETS:
        f = Path(rows_dir) / f"rows_{proto}_{s}_{arm}{'+rx' if proto == 'live' else ''}.parquet"
        if not f.exists():
            f = Path(rows_dir) / f"rows_{proto}_{s}_{arm}.parquet"
        if not f.exists():
            say(f"missing {f}; skipped")
            continue
        frames[(proto, s)] = played_frame(pd.read_parquet(f))
        say(f"{proto} {s}: {len(frames[(proto, s)])} played rows ({f.name})")
    out = {"version": "tr-v1", "levels": list(TR.LEVELS), "median": TR.MEDIAN,
           "min_bands": [list(b) for b in TR.MIN_BANDS], "shrink": TR.SHRINK, "arm": arm,
           "fit_pplay_min": FIT_PPLAY_MIN,
           "method": "q(v) = v-quantile of the played-conditional PIT over played rows, per tier x "
                     "E[min | plays] band, shrunk to the tier / pooled; harness rows of seasons < S "
                     "(walk-forward); g(v) = q(v) above the median (piecewise linear), identity "
                     "below",
           "seasons": {}}
    for S in TARGETS:
        tr = [k for k in frames if k[1] < S]
        if not tr:
            say(f"{S}: no training rows; not written (no remap)")
            continue
        fr = pd.concat([frames[k] for k in tr], ignore_index=True)
        fit = TR.fit_cells(fr)
        out["seasons"][S] = {"train": [f"{p}:{s}" for p, s in tr], **fit}
        say(f"{S}: {len(fr)} rows on {[f'{p}:{s}' for p, s in tr]}; tiers "
            + " ".join(f"{t} {[round(x, 4) for x in h]}" for t, h in fit["tiers"].items()))
    TR.TR_PATH.write_text(json.dumps(out, indent=1) + "\n")
    TR.cells.cache_clear()
    say(f"wrote {TR.TR_PATH}")


# ============================================================================== harness

def cmd_harness(season: str, protocol: str) -> None:
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "", tails=True)
    lo, hi = (B.SLICE if live else (None, None))
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"rows_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    say(f"harness {season} {protocol} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
        f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}")


# ============================================================================== PIT

def _rate(ind, cl) -> tuple[float, float]:
    from xfactor_calibration import rate_se
    return rate_se(np.asarray(ind, float), np.asarray(cl))


def pit_block(d: pd.DataFrame, col: str = "pit") -> dict:
    out = {"n": int(len(d))}
    for t in THRESH:
        m, se = _rate(d[col].to_numpy(float) > t, d["date"].astype(str).to_numpy())
        out[f"gt{t:.2f}"] = [100 * m, 100 * se]
    return out


def cmd_pit(specs: list[str], out_json: str | None) -> None:
    from courtside.proj import tail_remap as TR
    res = {}
    for spec in specs:
        lab, f = spec.split("=", 1)            # FILE or FILE,FILE,... (pooled)
        d = pd.concat([pd.read_parquet(x) for x in f.split(",")], ignore_index=True)
        d = d[d["has_proj"].astype(bool) & d["pit"].notna()].copy()
        d["min_played"] = d["exp_min"] / d["p_play"].clip(lower=1e-6)
        d["band"] = d["min_played"].map(TR.band_of)
        hi = d[d["p_play"] >= P_HI]
        r = {"all": pit_block(hi)}
        for t, x in hi.groupby("tier"):
            r[f"tier:{t}"] = pit_block(x)
        for (t, b), x in hi.groupby(["tier", "band"]):
            if len(x) >= 500:
                r[f"cell:{t}|{b}"] = pit_block(x)
        pf = played_frame(d)
        r["played:all"] = pit_block(pf, "uc")
        for t, x in pf.groupby("tier"):
            r[f"played:{t}"] = pit_block(x, "uc")
        res[lab] = r
        print(f"\n## {lab}  (share % above level, ± SE by date; nominal 10 / 5 / 1)")
        for k, v in r.items():
            print(f"  {k:22s} n={v['n']:6d}  " + "  ".join(
                f">{t:.2f} {v[f'gt{t:.2f}'][0]:5.2f}±{v[f'gt{t:.2f}'][1]:.2f}" for t in THRESH))
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))


# ============================================================================== paired DK

def cmd_dk(off: str, on: str, label: str) -> None:
    from role_blend_eval import TIERS, _metrics, _pair

    from courtside.proj import tail_remap as TR
    d = _pair(pd.read_parquet(off), pd.read_parquet(on))
    d["tier4"] = "bench"
    mc = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    for name, lo, hi in TIERS:
        d.loc[(mc >= lo) & (mc < hi), "tier4"] = name
    d["band"] = (d["exp_min_off"] / d["p_play_off"].clip(lower=1e-6)).map(TR.band_of)
    d["pop"] = (d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)
    d["shift"] = d["mean_on"] - d["mean_off"]
    res = {"label": label}
    for popn, q in (("pop10", d[d["pop"]]), ("all", d)):
        r = {"all": _metrics(q)}
        for t, _, _ in TIERS:
            r[f"tier:{t}"] = _metrics(q[q["tier4"] == t])
        for t in TR.TIERS:
            r[f"ctier:{t}"] = _metrics(q[q["tier"] == t])
        for (t, b), x in q.groupby(["tier", "band"]):
            if len(x) >= 500:
                r[f"cell:{t}|{b}"] = _metrics(x)
        r["_shift"] = {"mean": float(q["shift"].mean()),
                       "mean_abs": float(q["shift"].abs().mean()),
                       "q90_move": float((q["q90_on"] - q["q90_off"]).mean())}
        res[popn] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label} (ON - OFF; z two-way date x player)")
    for popn in ("pop10", "all"):
        print(f"### {popn}  shift {res[popn]['_shift']}")
        for k, v in res[popn].items():
            if k.startswith("_") or "crps" not in v:
                continue
            c, m = v["crps"], v["mae"]
            print(f"  {k:18s} n={v['n']:6d}  CRPS {c['off']:.4f} {c['diff']:+.4f} ({c['se']:.4f}) "
                  f"z {c['z']:+.2f}   MAE {m['off']:.4f} {m['diff']:+.4f} z {m['z']:+.2f}   "
                  f"bias {v['bias']['off']:+.3f}->{v['bias']['off'] + v['bias']['diff']:+.3f}")


# ============================================================================== X factor

def cmd_bignight(files: list[str], out_json: str | None) -> None:
    """files: OFF ON pairs (RotoGuru seasons, harness --tails). Per arm: the live per-tier stretch
    (`xfactor.STRETCH`, fit on raw OFF sims) and a walk-forward refit on the arm's own rows."""
    from xfactor_calibration import prep, rate_se, stretch_apply, stretch_fit

    from courtside.proj.dfs import xfactor as XF
    col, k = "p_ge_6x", 6
    arms = {"off": [], "on": []}
    for i, f in enumerate(files):
        arms["off" if i % 2 == 0 else "on"].append(pd.read_parquet(f))
    res = {}
    for arm, fs in arms.items():
        d = prep(pd.concat(fs, ignore_index=True))
        d = d[d["salary"].notna() & d["pplay_hi"] & d[col].notna()].copy()
        d["hit"] = (d["y"] >= k * d["salary"] / 1000.0).astype(float)
        live = XF.STRETCH
        d["p_live"] = stretch_apply(d, col, live)
        r = {"n": int(len(d)), "base": float(d["hit"].mean()), "raw": float(d[col].mean()),
             "live_map": float(d["p_live"].mean())}
        for s in sorted(d["season"].unique()):
            te = d[d["season"] == s].copy()
            tr = d[d["season"] != s]          # the other RotoGuru season (2 seasons only)
            ab = stretch_fit(tr, k, col)
            te["p_wf"] = stretch_apply(te, col, ab)
            br = te[te["tier"].isin(["bench", "role"])]
            blk = {"n": int(len(te)), "base": float(te["hit"].mean()),
                   "raw": float(te[col].mean()), "live_map": float(te["p_live"].mean()),
                   "refit": float(te["p_wf"].mean()), "refit_params": ab,
                   "brier_live": float(((te["p_live"] - te["hit"]) ** 2).mean()),
                   "brier_refit": float(((te["p_wf"] - te["hit"]) ** 2).mean())}
            for nm, pc in (("live_map", "p_live"), ("refit", "p_wf"), ("raw", col)):
                x = br[br[pc] >= XF.P_MIN]
                if len(x):
                    o, se = rate_se(x["hit"].to_numpy(), x["date"].to_numpy())
                    blk[f"tag_{nm}"] = {"n": int(len(x)), "per_day": len(x) / te["date"].nunique(),
                                        "pred": float(x[pc].mean()), "obs": o, "se": se}
            r[s] = blk
        res[arm] = r
        print(f"\n## arm {arm}: n {r['n']}, base {100 * r['base']:.2f}%, raw {100 * r['raw']:.2f}%, "
              f"live map {100 * r['live_map']:.2f}%")
        for s, b in r.items():
            if not isinstance(b, dict):
                continue
            print(f"  {s}: base {100 * b['base']:.2f}% raw {100 * b['raw']:.2f}% live-map "
                  f"{100 * b['live_map']:.2f}% refit {100 * b['refit']:.2f}%  Brier live "
                  f"{b['brier_live']:.5f} refit {b['brier_refit']:.5f}  params {b['refit_params']}")
            for nm in ("raw", "live_map", "refit"):
                t = b.get(f"tag_{nm}")
                if t:
                    print(f"     tag (bench/role, chance >= {XF.P_MIN}) by {nm:8s}: {t['n']:5d} rows "
                          f"({t['per_day']:.1f}/day) pred {100 * t['pred']:.1f}% obs "
                          f"{100 * t['obs']:.1f} ± {100 * t['se']:.1f}%")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))


# ============================================================================== TRUTH_TAU

def _pit_rand(X: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    from courtside.proj.metrics import pit_randomized
    return pit_randomized(X, y, rng.random(len(y)))


def cmd_tau(season: str, n_dates: int) -> None:
    """Per date, OFF and ON sims (same seeds): PIT of the actual in the builder's scoring truth
    (truth_sims, beta 0, tau 0 / TRUTH_TAU) for P(play) >= 0.85 players; and per played row the
    residual (y - mu_played) next to the sims' played sd (TRUTH_TAU's criterion)."""
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    from courtside.proj.dfs import portfolio as PF
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    sr = B.SeasonRun(conn, season, B.Settings(modes=(), n_sims=4000), log=lambda m: None)
    act = sr.box.actual
    rows = []
    for day in sr.dates(n_dates):
        if blackout():
            say("blackout: stopping")
            break
        for arm in (0, 1):
            P.set_tail_remap(bool(arm))
            res = P.project_historical(conn, day, protocol="avail", n_sims=4000, ctx=sr.ctx,
                                       bundle=sr.bundle)
            rng = np.random.default_rng(P.stable_seed(day, "tr-tau"))
            for g in res.sims:
                X = np.asarray(g.fp, np.float32)
                pids = np.asarray(g.player_id)
                y = np.array([(act.get((g.game_id, int(p))) or (0.0, 0.0))[0] for p in pids], float)
                mins = np.array([(act.get((g.game_id, int(p))) or (0.0, 0.0))[1] for p in pids],
                                float)
                pp = (X != 0).mean(0)
                tier = g.info["tier"].to_numpy()
                inp = pd.DataFrame({"pid": [str(p) for p in pids], "fppg": np.nan})
                for tau in (0.0, PF.TRUTH_TAU):
                    Xt, _ = PF.truth_sims(X, [str(p) for p in pids], inp, 0.0, tau,
                                          np.random.default_rng(rng.integers(1 << 31)),
                                          blended=False)
                    u = _pit_rand(Xt.astype(float), y, rng)
                    for j in range(len(pids)):
                        rows.append({"date": day, "arm": arm, "tau": tau, "pid": int(pids[j]),
                                     "tier": tier[j], "p_play": float(pp[j]), "pit": float(u[j]),
                                     "y": y[j], "played": mins[j] > 0,
                                     "mu_p": float(X[X[:, j] != 0, j].mean()) if pp[j] > 0 else np.nan,
                                     "sd_p": float(X[X[:, j] != 0, j].std()) if pp[j] > 0 else np.nan})
        say(f"tau {season} {day}: {len(rows)} rows")
    P.set_tail_remap(False)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"tau_{season}.parquet")
    tau_report(d, season)


def tau_report(d: pd.DataFrame, label: str) -> dict:
    out = {}
    print(f"\n## TRUTH_TAU check {label} (P(play) >= 0.85; share % above level ± SE by date)")
    for (arm, tau), x in d.groupby(["arm", "tau"]):
        hi = x[x["p_play"] >= P_HI]
        r = {"all": pit_block(hi)}
        for t, z in hi.groupby("tier"):
            r[t] = pit_block(z)
        pl = x[x["played"] & (x["p_play"] >= P_HI) & x["mu_p"].notna()]
        res = pl["y"] - pl["mu_p"]
        r["resid_sd"] = float(res.std())
        r["sim_sd_rms"] = float(np.sqrt((pl["sd_p"] ** 2).mean()))
        r["implied_tau"] = float(math.sqrt(max(0.0, (r["resid_sd"] / r["sim_sd_rms"]) ** 2 - 1)))
        out[f"tr{arm}_tau{tau:g}"] = r
        print(f"  tr={arm} tau={tau:.2f}: " + "  ".join(
            f"{k} " + "/".join(f"{v[f'gt{t:.2f}'][0]:.2f}" for t in THRESH)
            for k, v in r.items() if isinstance(v, dict))
            + f"   resid sd {r['resid_sd']:.3f} vs sims {r['sim_sd_rms']:.3f} -> implied tau "
              f"{r['implied_tau']:.3f}")
    return out


# ============================================================================== builder

def cmd_builder(season: str, n_dates: int, n: int, field: int, sims: int) -> None:
    """The contest builder on RotoGuru slates (research/portfolio_backtest.py setup: avail
    projections, live-like public field, preset GPP ladder of `field` entries, N = n) per arm
    (tr 0 / 1) x tau (TRUTH_TAU / 0). Predicted per-lineup P(top 1 %) and P(any top 1 %) next to
    the realised ranks at the actual points against the same field."""
    from courtside.proj import backtest as BT
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    from courtside.proj.dfs import field as F
    from courtside.proj.dfs import optimizer as O
    from courtside.proj.dfs import portfolio as PF
    from portfolio_backtest import slate_sims
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"builder_{season}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    have = set(zip(done["date"], done["arm"], done["tau"])) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    sr = BT.SeasonRun(conn, season, BT.Settings(modes=("cash",), n_sims=sims),
                      log=lambda m: print(m, flush=True))
    for day in sr.dates(n_dates):
        if blackout():
            say("blackout: stopping (resumable)")
            break
        pool = O.hist_pool(conn, day, sr.rules)
        inp0 = F.hist_inputs(conn, pool)
        if inp0["fppg"].notna().sum() < 40:
            continue
        prow = sr.rg[sr.rg["game_date"] == day]
        rg_of = {int(r.player_id): str(r.rg_id) for r in prow.itertuples(index=False)
                 if not pd.isna(r.player_id)}
        actual = O.hist_projections(conn, pool, "actual")
        in_pool = {p.dk_player_id for p in pool.players}
        for arm in (0, 1):
            if all((day, arm, t) in have for t in (PF.TRUTH_TAU, 0.0)):
                continue
            P.set_tail_remap(bool(arm))
            res = P.project_historical(conn, day, protocol="avail", n_sims=sims, ctx=sr.ctx,
                                       bundle=sr.bundle)
            pl = res.players.drop_duplicates("player_id").set_index("player_id")
            mu = {rg_of[int(i)]: float(q["dk_mean"]) for i, q in pl.iterrows()
                  if int(i) in rg_of and rg_of[int(i)] in in_pool}
            if len(mu) < 3 * sr.rules.n_slots:
                break
            ss = slate_sims(res, rg_of)
            inp = inp0.copy()
            inp["proj"] = inp["pid"].map(mu)
            contest = F.preset_contest("gpp", field, 1.0)
            for tau in (PF.TRUTH_TAU, 0.0):
                if (day, arm, tau) in have:
                    continue
                t0 = time.time()
                s = PF.Settings(n=n, n_sims=None, n_field=field - n, threads=1,
                                seed=P.stable_seed(day, "portfolio-bt"), tau=tau, curve=False)
                pf = PF.build_portfolio(pool, inp, ss, contest, s, mu=mu, rules=sr.rules)
                pairs = [[(e["slot"], e["player_id"]) for e in json.loads(lu["slots_json"])]
                         for lu in pf.lineups]
                act = np.array([actual[p].mean if p in actual else 0.0 for p in pf.field.pids],
                               np.float32)
                lad = PF.Ladder(contest)
                ev = F.evaluate(pf.field, act.reshape(1, -1), contest, list(pairs), seed=1,
                                tiers=False, own_entries=len(pairs), per_sim=True)
                ps = ev.per_sim
                pay, rank = PF.joint_payouts(lad, ps["score"], np.rint(ps["above"]).astype(np.int64),
                                             np.rint(ps["tied"]).astype(np.int64))
                lus = pf.lineups
                r = {"date": day, "season": season, "arm": arm, "tau": tau, "n": len(lus),
                     "field": field, "top1": lad.top1,
                     "pred_p_top1_mean": float(np.mean([lu["p_top1"] for lu in lus])),
                     "pred_p_any_top1": float(pf.summary["p_any_top1"]),
                     "pred_p_any_cash": float(pf.summary["p_any_cash"]),
                     "pred_e_payout": float(pf.summary["e_payout"]),
                     "real_top1_share": float((rank[0] <= lad.top1).mean()),
                     "real_any_top1": float((rank[0] <= lad.top1).any()),
                     "real_any_cash": float((rank[0] <= lad.paid).any()),
                     "real_payout": float(pay.sum()), "best_rank": int(rank.min()),
                     "seconds": round(time.time() - t0, 1)}
                rows.append(r)
                pd.DataFrame(rows).to_csv(out, index=False)
                say(f"builder {day} tr={arm} tau={tau}: pred P(top1) {r['pred_p_top1_mean']:.4f} "
                    f"any {r['pred_p_any_top1']:.3f}; real share {r['real_top1_share']:.3f} any "
                    f"{r['real_any_top1']:.0f}; best rank {r['best_rank']} ({r['seconds']}s)")
    P.set_tail_remap(False)
    builder_report(pd.DataFrame(rows))


def builder_report(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n## builder P(top 1 %): predicted vs realised (SE over slates)")
    for (arm, tau), x in df.groupby(["arm", "tau"]):
        k = len(x)
        se = lambda c: x[c].std(ddof=1) / math.sqrt(k) if k > 1 else float("nan")  # noqa: E731
        print(f"  tr={arm} tau={tau:.2f} slates {k}: per-lineup P(top1) pred "
              f"{100 * x.pred_p_top1_mean.mean():.2f}% real {100 * x.real_top1_share.mean():.2f} ± "
              f"{100 * se('real_top1_share'):.2f}%;  P(any top1) pred {100 * x.pred_p_any_top1.mean():.1f}% "
              f"real {100 * x.real_any_top1.mean():.1f} ± {100 * se('real_any_top1'):.1f}%;  "
              f"P(any cash) pred {100 * x.pred_p_any_cash.mean():.1f}% real "
              f"{100 * x.real_any_cash.mean():.1f}%;  E[pay] pred {x.pred_e_payout.mean():.2f} real "
              f"{x.real_payout.mean():.2f} ± {se('real_payout'):.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("rows_dir")
    f.add_argument("--arm", default=ARM)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    p = sub.add_parser("pit")
    p.add_argument("specs", nargs="+")
    p.add_argument("--json")
    d = sub.add_parser("dk")
    d.add_argument("off")
    d.add_argument("on")
    d.add_argument("label")
    b = sub.add_parser("bignight")
    b.add_argument("files", nargs="+")
    b.add_argument("--json")
    t = sub.add_parser("tau")
    t.add_argument("season")
    t.add_argument("--dates", type=int, default=30)
    u = sub.add_parser("builder")
    u.add_argument("season")
    u.add_argument("--dates", type=int, default=30)
    u.add_argument("--n", type=int, default=20)
    u.add_argument("--field", type=int, default=5000)
    u.add_argument("--sims", type=int, default=4000)
    r = sub.add_parser("report")
    r.add_argument("what", choices=("builder", "tau"))
    r.add_argument("files", nargs="+")
    x = ap.parse_args(argv)
    if x.cmd == "fit":
        cmd_fit(x.rows_dir, x.arm)
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    elif x.cmd == "pit":
        cmd_pit(x.specs, x.json)
    elif x.cmd == "dk":
        cmd_dk(x.off, x.on, x.label)
    elif x.cmd == "bignight":
        cmd_bignight(x.files, x.json)
    elif x.cmd == "tau":
        cmd_tau(x.season, x.dates)
    elif x.cmd == "builder":
        cmd_builder(x.season, x.dates, x.n, x.field, x.sims)
    elif x.what == "builder":
        builder_report(pd.concat([pd.read_csv(f) for f in x.files], ignore_index=True))
    else:
        d = pd.concat([pd.read_parquet(f) for f in x.files], ignore_index=True)
        tau_report(d, "pooled")


if __name__ == "__main__":
    main()
