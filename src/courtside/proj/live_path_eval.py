"""Opening-week LIVE-path minutes check (permanent regression eval).

Why it exists: the walk-forward eval in `minutes_eval` scores the minutes engine on the ORACLE
panel (candidates = whoever appears in the box score of the target game). The live path builds
candidates differently (`pipeline.candidate_panel` + `drop_moved_players` + DK-pool extras forced
in + report-based P(play)), and a bug that only that path exercises (departed players kept,
forced extras scored as deep-bench DNPs, v1's reserve dropping ~25 minutes per team) slipped
through at the 2026-27 opener. This module re-projects the FIRST days of a season the way the
live path would see them, then scores the result against actual minutes.

Construction (per season S, first `ndays` game days, regular season):
  * candidates: `minutes.pregame_panel` (the team's last 10 games, strictly before the date) plus
    the season's roster forced in as DK-pool extras (each player on the team he first plays for
    in S, the stand-in for the DK pool / roster snapshot, `pipeline.merge_extras`);
  * a forced player is removed from every other team's candidates (`candidate_panel`);
  * departed players dropped (`pipeline.drop_moved_players`): non-forced candidates whose
    current team (first team of S) differs from the candidate team, or who never appear in S;
  * availability: actual (regulars REGULAR_MIN that did not play are absent), the rest available
    with the model's P(plays | available);
  * projection = `run_projection` + `integrate` (margin integration at the closing spread).

Reported per version (v0.5, v1): minutes MAE and bias over the population (played, projected
>= 15), by tier, and team minutes: the model's sum of p x E[min | plays] against the target
240 - reserve (`team_dev`, must be ~0), and the pre-reconciliation slack
(240 - reserve) - sum(p x base) the reconciliation had to absorb.

    python -m courtside.proj.live_path_eval [--seasons 2024-25 2025-26] [--days 5]
        [--params-v05 'var/proj/bundle_{season}.json'] [--params-v1 'path/{season}.json']

Register `courtside.proj.live_path_eval` in `PROJ_MODULES` (proj/cli.py) for
`courtside proj live-path-eval`. `check` returns the violations (empty = pass).

Public API
  opening_panel(h, season, ndays=5) -> DataFrame     live-style candidates + actuals
  score(panel, prm, shares, spread) -> (frame, dict)
  evaluate_season(h, season, prms, shares, spread, ndays=5) -> dict
  run(conn, seasons, ndays=5, params=None, params_dir=None, log=print) -> dict
  check(res, max_team_dev=2.0, max_bias=1.5, gate=('v1',)) -> [str];  format_report(res) -> str
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from courtside.model.players import REGULAR_MIN
from courtside.proj import minutes as M
from courtside.proj import minutes_eval as E

SEASONS = ("2024-25", "2025-26")
NDAYS = 5
MAX_TEAM_DEV = 2.0          # |sum p x mean - (240 - reserve)| per team-game, minutes
MAX_BIAS = 1.5              # opening-week bias of the projected >= 15 population, minutes
VERSIONS = ("v0.5", "v1")
GATE_VERSIONS = ("v1",)    # the version the live path ships; v0.5 is the reference


# ============================================================================== panel

def opening_panel(h: M.History, season: str, ndays: int = NDAYS) -> pd.DataFrame:
    """Live-style candidates of the first `ndays` game days of `season` with the actual minutes
    (`act_min`, `played`) and the oracle availability (`absent`). Uses `h` only (no DB)."""
    tg = h.tg[(h.tg["season"] == season) & (h.tg["season_type"] == "Regular Season")]
    days = sorted(tg["date"].unique())[:ndays]
    t = tg[tg["date"].isin(days)][["game_id", "team", "date", "season", "margin", "n_ot"]].copy()
    t["key"] = t["game_id"] + "|" + t["team"]
    hs = h.hist[(h.hist["season"] == season) & (h.hist["season_type"] == "Regular Season")]
    # current team = first team of the season; the DK pool / roster snapshot stand-in
    first_team = hs.sort_values("date").drop_duplicates("player_id").set_index("player_id")["team"]
    roster = first_team.rename("team").reset_index()
    ex = t[["key", "team"]].merge(roster, on="team")[["key", "player_id"]]
    p = M.pregame_panel(h, t, extra=ex)
    p = p[(p["roster"] == 1) | p["last_date"].notna()].copy()
    ft = p["player_id"].map(first_team)
    forced_here = p.merge(ex.assign(f=1), on=["key", "player_id"], how="left")["f"] \
        .fillna(0).to_numpy() > 0
    # candidate_panel + drop_moved_players: a player forced onto a team leaves the other teams'
    # candidates, and a non-forced candidate whose current team differs (or who is gone) is dropped
    moved = ~forced_here & (ft.isna() | (ft != p["team"])).to_numpy()
    p = p[~moved].copy()
    act = h.hist[["game_id", "team", "player_id", "min"]].copy()
    act["key"] = act["game_id"] + "|" + act["team"]
    p = p.merge(act[["key", "player_id", "min"]].rename(columns={"min": "act_min"}),
                on=["key", "player_id"], how="left")
    p["played"] = p["act_min"].notna()
    p = p.merge(t[["key", "game_id", "margin", "n_ot"]], on="key", how="left",
                suffixes=("", "_t"))
    p["absent"] = (p["base_ew"].to_numpy() >= REGULAR_MIN) & ~p["played"].to_numpy()
    return p.reset_index(drop=True)


def default_shares(h: M.History, season: str, window: int = 5) -> M.PairShares:
    """Pair shares of the `window` seasons before `season` plus the season itself, as
    `pipeline.Context.shares` builds them."""
    seasons = sorted(x for x in h.tg["season"].unique() if x < season)[-window:] + [season]
    return M.PairShares(M.pair_events(M.oracle_panel(h, M.eval_targets(h, seasons))))


# ============================================================================== scoring

def tier_bucket(proj: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted([20.0, 26.0, 32.0], proj, side="right"), 0, 3)


def team_minutes(r: pd.DataFrame, prm: M.MinutesParams) -> pd.DataFrame:
    """Per team-game: target = 240 - reserve, `team_dev` = target - sum(p x mean0) (zero up to
    the 42-minute cap), `slack` = target - sum(p x base) (what the reconciliation absorbed)."""
    base = np.asarray(M.base_and_pplay(r, prm, r["absent"].to_numpy(bool))[0], float)
    g = pd.DataFrame({"key": r["key"].to_numpy(),
                      "pm": (r["p_play"] * r["mean0"]).to_numpy(),
                      "pb": (r["p_play"].to_numpy() * base),
                      "reserve": r["reserve"].to_numpy()}).groupby("key")
    out = pd.DataFrame({"reserve": g["reserve"].first(), "sum_pm": g["pm"].sum(),
                        "sum_pb": g["pb"].sum()})
    out["target"] = 240.0 - out["reserve"]
    out["team_dev"] = out["target"] - out["sum_pm"]
    out["slack"] = out["target"] - out["sum_pb"]
    return out


def score(panel: pd.DataFrame, prm: M.MinutesParams, shares, spread: pd.Series
          ) -> tuple[pd.DataFrame, dict]:
    """Project the panel with `prm`; returns (frame with `proj`, metrics dict)."""
    r = M.run_projection(panel, prm, shares=shares)
    r["proj"] = E.integrate(r, prm, spread)
    pop = r[r["played"] & (r["proj"] >= 15)]
    e = (pop["proj"] - pop["act_min"]).to_numpy(float)
    tier = tier_bucket(pop["proj"].to_numpy(float))
    out = {"version": prm.version, "n": int(len(pop)),
           "mae": float(np.abs(e).mean()) if len(e) else float("nan"),
           "bias": float(e.mean()) if len(e) else float("nan"),
           "by_tier": {E.TIER_LABELS[k]: {"n": int((tier == k).sum()),
                                          "mae": float(np.abs(e[tier == k]).mean())
                                          if (tier == k).any() else None,
                                          "bias": float(e[tier == k].mean())
                                          if (tier == k).any() else None}
                       for k in range(4)}}
    tm = team_minutes(r, prm)
    out["team"] = {"n": int(len(tm)), "reserve_mean": float(tm["reserve"].mean()),
                   "target_mean": float(tm["target"].mean()),
                   "sum_mean": float(tm["sum_pm"].mean()),
                   "dev_mean": float(tm["team_dev"].mean()),
                   "dev_max_abs": float(tm["team_dev"].abs().max()),
                   "slack_mean": float(tm["slack"].mean()),
                   "slack_max_abs": float(tm["slack"].abs().max())}
    return r, out


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """v0.5 vs v1 on the union of both >= 15 populations (same rows), gain clustered by game."""
    m = (a["played"] & ((a["proj"] >= 15) | (b["proj"] >= 15))).to_numpy()
    if not m.any():
        return {}
    y = a["act_min"].to_numpy(float)[m]
    ea, eb = np.abs(a["proj"].to_numpy(float)[m] - y), np.abs(b["proj"].to_numpy(float)[m] - y)
    d = ea - eb
    return {"n": int(m.sum()), "mae_v05": float(ea.mean()), "mae_v1": float(eb.mean()),
            "gain": float(d.mean()), "gain_se": E._cl_se(d, a["game_id"].to_numpy()[m]),
            "bias_v05": float((a["proj"].to_numpy(float)[m] - y).mean()),
            "bias_v1": float((b["proj"].to_numpy(float)[m] - y).mean())}


def evaluate_season(h: M.History, season: str, prms: dict[str, M.MinutesParams], shares,
                    spread: pd.Series, ndays: int = NDAYS) -> dict:
    """Opening-week live-path metrics of each version in `prms` on the same panel."""
    panel = opening_panel(h, season, ndays)
    out = {"season": season, "days": ndays, "team_games": int(panel["key"].nunique()),
           "candidates": int(len(panel)), "versions": {}}
    frames = {}
    for name, prm in prms.items():
        frames[name], out["versions"][name] = score(panel, prm, shares, spread)
    if "v0.5" in frames and "v1" in frames:
        out["paired"] = paired(frames["v0.5"], frames["v1"])
    return out


# ============================================================================== run / report

def _load_params(spec: str, season: str) -> M.MinutesParams:
    txt = Path(spec.format(season=season)).read_text()
    d = json.loads(txt)
    return M.MinutesParams(**d["minutes"]) if "minutes" in d else M.MinutesParams.from_json(txt)


def run(conn: sqlite3.Connection, seasons=SEASONS, ndays: int = NDAYS,
        params: dict[str, str] | None = None, params_dir=None,
        log: Callable[[str], None] = print) -> dict:
    """Live-path opening-week eval of `seasons`. Params for version v of season S: the file
    `params[v]` (a bundle or MinutesParams JSON; `{season}` is substituted), else
    `<params_dir>/minutes_params_<S>_<v>.json`, else fitted (fit < S, saved to params_dir)."""
    from courtside.proj import pipeline as P
    t0 = time.time()
    res: dict = {"seasons": {}, "ndays": ndays}
    sp = E.spreads(conn)
    sp["key"] = sp["game_id"] + "|" + sp["team"]
    spread = sp.set_index("key")["spread"]
    pdir = Path(params_dir) if params_dir is not None else None
    for s in seasons:
        ctx = P.Context(conn, s)
        prms = {}
        for v in VERSIONS:
            f = pdir / f"minutes_params_{s}_{v}.json" if pdir is not None else None
            if params and params.get(v):
                prms[v] = _load_params(params[v], s)
            elif f is not None and f.exists():
                prms[v] = M.MinutesParams.from_json(f.read_text())
            else:
                log(f"{s} {v}: fitting")
                panel = ctx.train_panel()
                prms[v] = M.fit(ctx.h, s, window=P.MINUTES_WINDOW, log=log, panel=panel,
                                shares=M.PairShares(M.pair_events(panel)),
                                version="v1" if v == "v1" else "v0.5")
                if f is not None:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(prms[v].to_json())
        res["seasons"][s] = evaluate_season(ctx.h, s, prms, ctx.shares, spread, ndays)
        log(f"{s}: " + ", ".join(
            f"{v} MAE {x['mae']:.3f} bias {x['bias']:+.2f}"
            for v, x in res["seasons"][s]["versions"].items()))
    res["seconds"] = round(time.time() - t0, 1)
    res["violations"] = check(res)
    return res


def check(res: dict, max_team_dev: float = MAX_TEAM_DEV, max_bias: float = MAX_BIAS,
          gate: tuple = GATE_VERSIONS) -> list[str]:
    """Threshold violations (empty list = pass) for the gated versions (the shipped v1; v0.5 is
    reported for reference and is known to be biased here): team minutes within `max_team_dev`
    of 240 - reserve for every team-game, opening-week bias within +-`max_bias`."""
    bad = []
    for s, r in res["seasons"].items():
        for v, x in r["versions"].items():
            if v not in gate:
                continue
            if not x["team"]["dev_max_abs"] <= max_team_dev:
                bad.append(f"{s} {v}: team minutes off 240 - reserve by "
                           f"{x['team']['dev_max_abs']:.2f} > {max_team_dev}")
            if not abs(x["bias"]) <= max_bias:
                bad.append(f"{s} {v}: opening-week bias {x['bias']:+.2f} outside +-{max_bias}")
    return bad


def format_report(res: dict) -> str:
    L = [f"Opening-week live-path minutes check (first {res.get('ndays', NDAYS)} days, "
         f"departed dropped, season roster forced in, actual availability)"]
    for s, r in res["seasons"].items():
        L.append(f"\n{s}: {r['team_games']} team-games, {r['candidates']} candidates")
        for v, x in r["versions"].items():
            L.append(f"  {v:5s} n={x['n']} MAE {x['mae']:.3f} bias {x['bias']:+.2f}")
            L.append("        tiers " + "  ".join(
                f"{k}: n={t['n']} MAE {t['mae']:.2f} bias {t['bias']:+.2f}"
                for k, t in x["by_tier"].items() if t["n"]))
            t = x["team"]
            L.append(f"        team minutes: sum p x mean {t['sum_mean']:.1f} vs 240 - reserve "
                     f"{t['target_mean']:.1f} (reserve {t['reserve_mean']:.1f}); dev mean "
                     f"{t['dev_mean']:+.2f} max {t['dev_max_abs']:.2f}; pre-reconcile slack "
                     f"mean {t['slack_mean']:+.1f} max |{t['slack_max_abs']:.1f}|")
        pr = r.get("paired")
        if pr:
            L.append(f"  paired (union n={pr['n']}): v0.5 {pr['mae_v05']:.3f} v1 {pr['mae_v1']:.3f} "
                     f"gain {pr['gain']:+.3f} (SE {pr['gain_se']:.3f}); bias v0.5 "
                     f"{pr['bias_v05']:+.2f} v1 {pr['bias_v1']:+.2f}")
    bad = res.get("violations")
    L.append("\n" + ("PASS" if not bad else "FAIL: " + "; ".join(bad)))
    return "\n".join(L)


# ============================================================================== CLI

def add_commands(sub) -> dict:
    """`courtside proj live-path-eval` (register this module in PROJ_MODULES)."""
    p = sub.add_parser("live-path-eval",
                       help="opening-week minutes check built like the live path: departed "
                            "dropped, season roster forced in; MAE/bias/tier, team minutes "
                            "vs 240 - reserve (exit 1 on a threshold violation)")
    p.add_argument("--seasons", nargs="+", default=list(SEASONS))
    p.add_argument("--days", type=int, default=NDAYS, help="first N game days (default 5)")
    p.add_argument("--params-v05", help="MinutesParams or bundle JSON; {season} is substituted")
    p.add_argument("--params-v1", help="same, for v1")
    p.add_argument("--params-dir", default=None,
                   help="cache of fitted params (default <data_dir>/proj)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    return {"live-path-eval": _cmd}


def _cmd(args) -> int:
    from .. import config, db
    conn = db.connect()
    pdir = args.params_dir or (config.data_dir() / "proj")
    res = run(conn, tuple(args.seasons), args.days,
              params={"v0.5": args.params_v05, "v1": args.params_v1}, params_dir=pdir)
    print(json.dumps(res, default=float, indent=1) if args.json else format_report(res))
    return 1 if res["violations"] else 0


if __name__ == "__main__":  # pragma: no cover - manual run
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    add_commands(ap.add_subparsers(dest="cmd", required=True))
    raise SystemExit(_cmd(ap.parse_args(["live-path-eval", *sys.argv[1:]])))
