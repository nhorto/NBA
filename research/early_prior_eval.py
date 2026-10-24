"""Early-season priors (`COURTSIDE_EARLY_PRIOR`, `+ep`; IMPROVEMENTS.md P3): fit and evaluation.

Docs: docs/research/early-prior.md.  Code under test: `proj/early_prior.py`, its hooks in
`state.rate_frame` / `walk_forward` (`ep=`) and `pipeline` (`_load_bundle_ep`, `build_games`).

    python research/early_prior_eval.py state S [--parts dwmg]
        fit `+ep` on seasons < S (base state params of S) and compare the walk-forward state of S
        base vs +ep given the actual minutes: NB log-lik per stat, DK points (no bonus) error;
        first 30 days and full season; two-way date x player SEs -> OUT/state_<S>_<parts>.json
    python research/early_prior_eval.py fitparams S [--parts dwmg]
        the `+ep` params of S alone -> OUT/ep_params_<S>.json
    python research/early_prior_eval.py harness S PROTO
        backtest rows (4,000 sims) of the arm in the environment -> OUT/rows_<P>_<S>_<MV>.parquet
    python research/early_prior_eval.py dk OFF ON LABEL
        paired ON - OFF DK CRPS / MAE (two-way date x player SEs): first 30 days and full season,
        tiers, rookies / age groups

Environment: COURTSIDE_DATA (bundles + DB), EP_OUT (default var/tmp/ep/out), EP_STATE_DIR (base
state params `ep_state_<S>.json` when the data dir has no bundle). Heavy: Mac scratch
(`courtside-compute/ep`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("EP_OUT", ROOT / "var" / "tmp" / "ep" / "out"))
STATE_DIR = Path(os.environ.get("EP_STATE_DIR", ROOT / "var" / "tmp" / "ep" / "state"))
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
FIRST_DAYS = 30


def say(m: str) -> None:
    print(f"[ep] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _conn():
    from courtside import db
    return db.connect()


def base_state(season: str) -> dict:
    """The base rate-state params of S: the live arm's bundle in the data dir, else
    EP_STATE_DIR/ep_state_<S>.json, else DEFAULT_PARAMS when it was fit on seasons < S."""
    from courtside.proj import pipeline as P
    from courtside.proj import state as ST
    for name in (f"bundle_{season}_v1_cx_mf.json", f"bundle_{season}_v1_cx.json"):
        f = P.cache_dir() / name
        if f.exists():
            return json.loads(f.read_text())["state"]
    f = STATE_DIR / f"ep_state_{season}.json"
    if f.exists():
        return json.loads(f.read_text())
    if ST.DEFAULT_PARAMS["fit_seasons"][-1] == P._prev_season(season):
        return ST.DEFAULT_PARAMS
    raise FileNotFoundError(f"no base state params for {season}")


def rs_start(conn, season: str) -> int:
    from courtside.proj import state as ST
    d0 = conn.execute("SELECT MIN(game_date) FROM team_games WHERE season = ? AND "
                      "season_type = 'Regular Season'", (season,)).fetchone()[0]
    return ST._day_of(d0)


# ============================================================================== state level

def _dk_hat(w: pd.DataFrame, sp: dict, pref: str = "r_") -> np.ndarray:
    from courtside.proj import state as ST
    m, mpg = w["min"].to_numpy(float), w["mpg"].to_numpy(float)
    return sum(ST.DK_WEIGHTS[s] * w[f"{pref}{s}"].to_numpy(float) * m
               * ST.minutes_multiplier(sp, s, m, mpg) for s in ST.DK_WEIGHTS)


def _nb_ll(w: pd.DataFrame, sp: dict) -> np.ndarray:
    """Per-row NB log-likelihood summed over the DK stats given actual minutes."""
    from courtside.proj import state as ST
    m, mpg, tier = w["min"].to_numpy(float), w["mpg"].to_numpy(float), w["tier"].to_numpy(int)
    tot = np.zeros(len(w))
    for s in ST.DK_WEIGHTS:
        mu = w[f"r_{s}"].to_numpy(float) * m * ST.minutes_multiplier(sp, s, m, mpg)
        tot += ST.nb_logpmf(w[s].to_numpy(float), mu, ST.phi_for(sp, s, tier))
    return tot


def cmd_state(season: str, parts: str) -> dict:
    from courtside.proj import early_prior as EP
    from courtside.proj import state as ST
    from courtside.proj.metrics import paired_cluster2
    conn = _conn()
    sp = base_state(season)
    data = EP.load_data(conn)
    pos = ST.load_positions(conn)
    s0 = ST.sidx_of(season)
    first = ST.season_label(min(s0 - EP.FIT_SEASONS, ST.sidx_of(EP.FIRST_BIO_SEASON))
                            - sp.get("history_seasons", 6))
    games = ST.load_games(conn, since_season=first, until_season=season)
    t0 = time.time()
    epp = EP.fit(conn, season, sp, data=data, games=games[games["season"] < season], pos=pos,
                 parts=tuple(parts), log=say)
    say(f"fit {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ep_params_{season}_{parts}.json").write_text(json.dumps(epp, indent=1))
    a = ST.walk_forward(conn, season, sp, games=games, pos=pos)
    b = ST.walk_forward(conn, season, sp, games=games, pos=pos, ep=EP.EPInput(epp, data))
    y = sum(ST.DK_WEIGHTS[s] * a[s].to_numpy(float) for s in ST.DK_WEIGHTS)
    ha, hb = _dk_hat(a, sp), _dk_hat(b, sp)
    la, lb = _nb_ll(a, sp), _nb_ll(b, sp)
    day = ST._to_day(a["game_date"])
    rs = (a["season_type"] == "Regular Season").to_numpy()
    d0 = rs_start(conn, season)
    bio = data.bio.set_index("player_id")
    fs = bio["first_sidx"].reindex(a["player_id"]).to_numpy(float)
    rookie = ~np.isfinite(fs) | (fs >= s0)
    mpg = a["mpg"].to_numpy(float)
    res = {"season": season, "parts": parts, "ep": {k: epp[k] for k in ("w", "v", "d", "m", "g")},
           "diag": epp["diagnostics"]}
    dates, pids = a["game_date"].astype(str).to_numpy(), a["player_id"].to_numpy()
    for name, sel in (("first30", rs & (day < d0 + FIRST_DAYS)), ("full", rs),
                      ("first30_rookie", rs & (day < d0 + FIRST_DAYS) & rookie),
                      ("first30_mpg20", rs & (day < d0 + FIRST_DAYS) & (mpg >= 20)),
                      ("full_mpg20", rs & (mpg >= 20))):
        r = {"n": int(sel.sum())}
        for k, z0, z1 in (("abs", np.abs(ha - y), np.abs(hb - y)),
                          ("sq", (ha - y) ** 2, (hb - y) ** 2),
                          ("nll", -la, -lb)):
            pc = paired_cluster2((z1 - z0)[sel], dates[sel], pids[sel])
            r[k] = {"off": float(z0[sel].mean()), "diff": pc["mean"], "se": pc["se"],
                    "z": pc["mean"] / pc["se"] if pc["se"] > 0 else float("nan")}
        r["bias_off"] = float((ha - y)[sel].mean())
        r["bias_on"] = float((hb - y)[sel].mean())
        res[name] = r
        say(f"{season} {name}: n {r['n']}  |err| {r['abs']['diff']:+.4f} (z {r['abs']['z']:+.2f})"
            f"  sq {r['sq']['diff']:+.3f} (z {r['sq']['z']:+.2f})  nll {r['nll']['diff']:+.4f}"
            f" (z {r['nll']['z']:+.2f})  bias {r['bias_off']:+.3f} -> {r['bias_on']:+.3f}")
    (OUT / f"state_{season}_{parts}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def cmd_fitparams(season: str, parts: str) -> None:
    from courtside.proj import early_prior as EP
    conn = _conn()
    epp = EP.fit(conn, season, base_state(season), parts=tuple(parts), log=say)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ep_params_{season}.json").write_text(json.dumps(epp, indent=1))


# ============================================================================== harness

def cmd_harness(season: str, protocol: str, first_days: int) -> None:
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    M.GBM_PARAMS["num_threads"] = 1
    conn = _conn()
    t0 = time.time()
    live = protocol in B.LIVE_OFFSETS
    st = B.Settings(protocol=protocol, n_sims=4000, modes=(), baselines=B.BASELINES,
                    pplay="loo_month" if live else "")
    lo, hi = (B.SLICE if live else (None, None))
    if first_days:
        from courtside.proj import state as ST
        d0 = rs_start(conn, season)
        hi = str(ST._day_to_date([d0 + first_days - 1]).iloc[0])
    pl, _, meta = B.run_season(conn, season, st, log=lambda m: print(m, flush=True),
                               date_lo=lo, date_hi=hi)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_f{first_days}" if first_days else ""
    f = OUT / f"rows_{protocol}_{season}{tag}_{P.MODEL_VERSION}.parquet"
    pl.to_parquet(f)
    say(f"harness {season} {protocol}{tag} {P.MODEL_VERSION}: {len(pl)} rows, skipped "
        f"{len(meta['skipped'])}, {time.time() - t0:.0f}s -> {f}")


# ============================================================================== paired DK

def _pair(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    k = ["date", "player_id"]
    a = off.dropna(subset=["player_id"]).drop_duplicates(k)
    b = on.dropna(subset=["player_id"]).drop_duplicates(k)
    cols = ["mean", "q10", "q90", "crps", "exp_min", "p_play"]
    d = a.merge(b[k + cols], on=k, suffixes=("_off", "_on"))
    d = d[d["has_proj"].astype(bool) & d["crps_off"].notna() & d["crps_on"].notna()]
    return d.reset_index(drop=True)


def _metrics(d: pd.DataFrame) -> dict:
    from courtside.proj.metrics import paired_cluster2
    y = d["y"].to_numpy(float)
    dates, pids = d["date"].astype(str).to_numpy(), d["player_id"].to_numpy()
    out = {"n": int(len(d))}
    if len(d) < 20:
        return out
    for k, f in (("crps", lambda a: d[f"crps_{a}"].to_numpy(float)),
                 ("mae", lambda a: np.abs(d[f"mean_{a}"].to_numpy(float) - y)),
                 ("bias", lambda a: d[f"mean_{a}"].to_numpy(float) - y)):
        z0, z1 = f("off"), f("on")
        pc = paired_cluster2(z1 - z0, dates, pids)
        out[k] = {"off": float(z0.mean()), "diff": pc["mean"], "se": pc["se"],
                  "se_date": pc["se_c1"], "z": pc["mean"] / pc["se"] if pc["se"] > 0 else np.nan}
    return out


def cmd_dk(off: str, on: str, label: str) -> None:
    from datetime import date
    from courtside.proj import early_prior as EP
    from courtside.proj import pipeline as P
    from courtside.proj import state as ST
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    d = _pair(a, b)
    conn = _conn()
    season = P.season_of(date.fromisoformat(str(d["date"].iloc[0])[:10]))
    s0 = ST.sidx_of(season)
    d0 = rs_start(conn, season)
    d["day"] = ST._to_day(d["date"].astype(str).str[:10])
    data = EP.load_data(conn)
    bio = data.bio.set_index("player_id")
    pid = d["player_id"].astype(np.int64)
    fs = bio["first_sidx"].reindex(pid).to_numpy(float)
    d["rookie"] = ~np.isfinite(fs) | (fs >= s0)
    feb1 = ST._day_of(f"{s0 + 1}-02-01")
    d["age"] = (feb1 - bio["birth_day"].reindex(pid).to_numpy(float)) / 365.25
    d["tier4"] = "bench"
    mc = d["exp_min_off"] / d["p_play_off"].clip(lower=0.05)
    for name, lo, hi in TIERS:
        d.loc[(mc >= lo) & (mc < hi), "tier4"] = name
    d["pop"] = (d["exp_min_off"] >= 10) | (d["exp_min_on"] >= 10)
    d["shift"] = d["mean_on"] - d["mean_off"]
    res = {"label": label, "season": season}
    for span, q0 in (("first30", d[d["day"] < d0 + FIRST_DAYS]), ("full", d)):
        for popn, q in (("pop10", q0[q0["pop"]]), ("all", q0)):
            r = {"all": _metrics(q)}
            for t, _, _ in TIERS:
                r[f"tier:{t}"] = _metrics(q[q["tier4"] == t])
            r["rookie"] = _metrics(q[q["rookie"]])
            r["age<=23"] = _metrics(q[q["age"] < 24])
            r["age24-29"] = _metrics(q[(q["age"] >= 24) & (q["age"] < 30)])
            r["age30+"] = _metrics(q[q["age"] >= 30])
            r["_shift"] = {"mean": float(q["shift"].mean()),
                           "mean_abs": float(q["shift"].abs().mean())}
            res[f"{span}:{popn}"] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dk_{label}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n## {label} (ON - OFF; z two-way date x player)")
    for key in ("first30:pop10", "full:pop10", "first30:all", "full:all"):
        print(f"### {key}  shift {res[key]['_shift']}")
        for k, v in res[key].items():
            if k.startswith("_") or "crps" not in v:
                continue
            c, m = v["crps"], v["mae"]
            print(f"  {k:<14} n {v['n']:>6}  CRPS {c['off']:.3f} {c['diff']:+.4f} "
                  f"({c['se']:.4f}, z {c['z']:+.2f})  MAE {m['diff']:+.4f} (z {m['z']:+.2f})  "
                  f"bias {v['bias']['off']:+.3f} -> {v['bias']['off'] + v['bias']['diff']:+.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("state")
    a.add_argument("season")
    a.add_argument("--parts", default="dwmg")
    a = sub.add_parser("fitparams")
    a.add_argument("season")
    a.add_argument("--parts", default="dwmg")
    a = sub.add_parser("harness")
    a.add_argument("season")
    a.add_argument("protocol")
    a.add_argument("--first-days", type=int, default=0)
    a = sub.add_parser("dk")
    a.add_argument("off")
    a.add_argument("on")
    a.add_argument("label")
    x = ap.parse_args(argv)
    if x.cmd == "state":
        cmd_state(x.season, x.parts)
    elif x.cmd == "fitparams":
        cmd_fitparams(x.season, x.parts)
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol, x.first_days)
    else:
        cmd_dk(x.off, x.on, x.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
