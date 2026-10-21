"""X factor (big-night) calibration: are the upper tails of our per-player DK distributions honest?

Owner request 2026-10-21 ("X factor": low-popularity bench/role players with a real chance of a
big night). Write-up: docs/research/xfactor.md. Subcommands:

  pit DIR...        upper-tail PIT reliability from backtest player caches (each DIR = one
                    protocol/season cache of `courtside.proj.backtest`, i.e. *.players.parquet):
                    observed share with PIT > t (t = .80 .90 .95 .98 .99) vs 1 - t, by season,
                    model tier, expected-minutes band, salary band (RotoGuru seasons), P(play);
                    SEs clustered by date.
  bignight DIR...   big-night reliability from caches written with the harness's optional tail
                    output (backtest-dfs --tails: columns p_ge_5x / p_ge_6x / p_ge_7x, q95, q99):
                    binned predicted P(DK >= k * salary/1000) vs observed, Brier vs a salary-band
                    baseline fit on seasons < S (walk-forward), by tier / minutes band, plus a
                    per-tier tail-stretch recalibration fit on seasons < S.
  harness SEASON    (scratch box) run the avail harness with --tails, no lineups, one season.
  opener            read-only check of the opening-night (2026-10-20) live runs on the Mac prod
                    DB / npz: P(DK >= k * salary/1000) for named players, plus their popularity
                    band (run ON the Mac with COURTSIDE_DATA pointing at the prod var/).

    python research/xfactor_calibration.py pit var/tmp/xf/bt/avail_2019-20 ... [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESH = (0.80, 0.90, 0.95, 0.98, 0.99)
KS = (5, 6, 7)
MIN_BANDS = [(0, 8, "<8"), (8, 16, "8-16"), (16, 24, "16-24"), (24, 32, "24-32"), (32, 99, "32+")]
SAL_BANDS = [(0, 4000, "<4.0k"), (4000, 5000, "4.0-4.9k"), (5000, 6500, "5.0-6.4k"),
             (6500, 8500, "6.5-8.4k"), (8500, 99999, "8.5k+")]
P_PLAY_HIGH = 0.85


# ---------------------------------------------------------------------------------- helpers

def load_dir(d: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(Path(d) / "*.players.parquet")))
    if not fs:
        raise SystemExit(f"no *.players.parquet under {d}")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    name = Path(d).name                    # <protocol>_<season>, or <season>/<config hash>
    df["src"] = f"tails_{Path(d).parent.name}" if name.startswith("v1-") else name
    return df


def band(x: pd.Series, bands) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype=object)
    for lo, hi, lab in bands:
        out[(x >= lo) & (x < hi)] = lab
    return out


def rate_se(ind: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    """Mean of an indicator with the SE clustered by `cl` (dates)."""
    ind = np.asarray(ind, float)
    n = len(ind)
    if n == 0:
        return float("nan"), float("nan")
    m = float(ind.mean())
    s = pd.Series(ind - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    se = float(np.sqrt(g / (g - 1) * (s ** 2).sum()) / n) if g > 1 else float("nan")
    return m, se


def mean_diff_se(d: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    return rate_se(d, cl)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["has_proj"].astype(bool) & df["pit"].notna()].copy()
    d["mins_band"] = band(d["exp_min"] / d["p_play"].clip(lower=1e-6), MIN_BANDS)
    d["sal_band"] = band(d["salary"], SAL_BANDS)
    d["pplay_hi"] = d["p_play"] >= P_PLAY_HIGH
    return d


# ---------------------------------------------------------------------------------- PIT tails

def tail_row(x: pd.DataFrame) -> dict:
    cl = x["date"].to_numpy()
    u = x["pit"].to_numpy(float)
    row = {"n": int(len(x)), "days": int(x["date"].nunique())}
    for t in THRESH:
        m, se = rate_se(u > t, cl)
        row[f"gt{t:.2f}"] = (m, se)
    m, se = rate_se(u < 0.10, cl)
    row["lt0.10"] = (m, se)
    return row


def tail_table(d: pd.DataFrame, by: list[str]) -> list[dict]:
    rows = []
    for key, x in d.groupby(by, dropna=True):
        key = key if isinstance(key, tuple) else (key,)
        r = tail_row(x)
        r["key"] = dict(zip(by, key))
        rows.append(r)
    return rows


def fmt_tail(rows: list[dict], title: str) -> str:
    L = [title, f"{'group':<34} {'n':>7} " + " ".join(f"{'>' + format(t, '.2f'):>13}"
                                                      for t in THRESH) + f" {'<0.10':>13}",
         f"{'expected':<34} {'':>7} " + " ".join(f"{100 * (1 - t):>13.1f}" for t in THRESH)
         + f" {10.0:>13.1f}"]
    for r in rows:
        g = " ".join(str(v) for v in r["key"].values())
        cells = []
        for t in THRESH:
            m, se = r[f"gt{t:.2f}"]
            cells.append(f"{100 * m:6.2f}±{100 * se:4.2f}".rjust(13))
        m, se = r["lt0.10"]
        cells.append(f"{100 * m:6.2f}±{100 * se:4.2f}".rjust(13))
        L.append(f"{g:<34} {r['n']:>7} " + " ".join(cells))
    return "\n".join(L)


def cmd_pit(dirs: list[str], out_json: str | None) -> None:
    d = prep(pd.concat([load_dir(x) for x in dirs], ignore_index=True))
    rep = {}
    blocks = [
        ("ALL projected rows (DNP included, P(play) from the model), by source", ["src"]),
        ("P(play) >= 0.85, by source", None),
        ("P(play) >= 0.85, by source x tier", ["src", "tier"]),
        ("P(play) >= 0.85, by source x expected-minutes band (minutes if he plays)",
         ["src", "mins_band"]),
        ("P(play) >= 0.85, RotoGuru seasons, by salary band", ["sal_band"]),
        ("P(play) >= 0.85, RotoGuru seasons, by season x salary band", ["season", "sal_band"]),
        ("P(play) >= 0.85, all seasons, by tier x minutes band", ["tier", "mins_band"]),
        ("P(play) >= 0.85, bench+role with minutes 8-24, by source", ["src"]),
    ]
    for title, by in blocks:
        x = d
        if "P(play) >= 0.85" in title:
            x = x[x["pplay_hi"]]
        if "RotoGuru" in title:
            x = x[x["salary"].notna()]
        if "minutes 8-24" in title:
            x = x[x["tier"].isin(["bench", "role"]) & x["mins_band"].isin(["8-16", "16-24"])]
        rows = tail_table(x, by or ["src"])
        rep[title] = rows
        print(fmt_tail(rows, "\n== " + title + "  (share %, ± SE clustered by date)"))
    if out_json:
        Path(out_json).write_text(json.dumps(rep, default=str, indent=1))


# ---------------------------------------------------------------------------------- big nights

def salary_baseline(train: pd.DataFrame, test: pd.DataFrame, k: int) -> np.ndarray:
    """P(DK >= k * salary/1000) from the historical frequency in $500 salary bins (seasons < S,
    the same P(play) >= 0.85 population; bins with < 200 rows pool with the neighbours)."""
    edges = np.arange(3000, 15001, 500)
    hit = (train["y"] >= k * train["salary"] / 1000.0).to_numpy(float)
    b_tr = np.clip(np.searchsorted(edges, train["salary"].to_numpy(), "right") - 1, 0,
                   len(edges) - 1)
    b_te = np.clip(np.searchsorted(edges, test["salary"].to_numpy(), "right") - 1, 0,
                   len(edges) - 1)
    num = np.bincount(b_tr, hit, len(edges))
    den = np.bincount(b_tr, None, len(edges)).astype(float)
    rate = np.full(len(edges), np.nan)
    for i in range(len(edges)):
        lo, hi = i, i
        while den[lo:hi + 1].sum() < 200 and (lo > 0 or hi < len(edges) - 1):
            lo, hi = max(lo - 1, 0), min(hi + 1, len(edges) - 1)
        rate[i] = num[lo:hi + 1].sum() / max(den[lo:hi + 1].sum(), 1.0)
    return rate[b_te]


def reliability(p: np.ndarray, y: np.ndarray, cl: np.ndarray,
                edges=(0, .02, .05, .10, .15, .20, .30, .40, 1.0001)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        obs, se = rate_se(y[m], cl[m])
        out.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                    "pred": float(p[m].mean()), "obs": obs, "se": se})
    return out


def brier_block(x: pd.DataFrame, pcol: str, bcol: str, k: int) -> dict:
    y = (x["y"] >= k * x["salary"] / 1000.0).to_numpy(float)
    p = x[pcol].to_numpy(float)
    b = x[bcol].to_numpy(float)
    cl = x["date"].to_numpy()
    bs, bb = (p - y) ** 2, (b - y) ** 2
    dm, dse = mean_diff_se(bb - bs, cl)
    return {"n": int(len(x)), "base_rate": float(y.mean()), "pred_mean": float(p.mean()),
            "brier": float(bs.mean()), "brier_salary": float(bb.mean()),
            "skill_vs_salary": float(1 - bs.mean() / bb.mean()) if bb.mean() > 0 else None,
            "diff": dm, "diff_se": dse}


def stretch_fit(train: pd.DataFrame, k: int, col: str) -> dict:
    """Per-tier logistic recalibration logit p' = a + b logit p, fit on seasons < S."""
    out = {}
    for tier, x in train.groupby("tier"):
        p = np.clip(x[col].to_numpy(float), 1e-4, 1 - 1e-4)
        y = (x["y"] >= k * x["salary"] / 1000.0).to_numpy(float)
        z = np.log(p / (1 - p))
        a, b = 0.0, 1.0
        for _ in range(50):                                   # Newton on (a, b)
            q = 1 / (1 + np.exp(-(a + b * z)))
            w = q * (1 - q)
            g = np.array([(y - q).sum(), ((y - q) * z).sum()])
            H = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]])
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
            a, b = a + step[0], b + step[1]
            if np.abs(step).max() < 1e-8:
                break
        out[tier] = (float(a), float(b))
    return out


def stretch_apply(x: pd.DataFrame, col: str, ab: dict) -> np.ndarray:
    p = np.clip(x[col].to_numpy(float), 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p))
    out = p.copy()
    for tier, (a, b) in ab.items():
        m = (x["tier"] == tier).to_numpy()
        out[m] = 1 / (1 + np.exp(-(a + b * z[m])))
    return out


def cmd_bignight(dirs: list[str], out_json: str | None) -> None:
    d = prep(pd.concat([load_dir(x) for x in dirs], ignore_index=True))
    d = d[d["salary"].notna() & d["pplay_hi"]].copy()
    need = [f"p_ge_{k}x" for k in KS]
    if any(c not in d for c in need):
        raise SystemExit(f"caches lack {need}: rerun the harness with --tails (subcommand harness)")
    seasons = sorted(d["season"].unique())
    rep: dict = {}
    for k in KS:
        col = f"p_ge_{k}x"
        print(f"\n######## big night = DK >= {k} x salary/1000  (P(play) >= {P_PLAY_HIGH})")
        for s in seasons:
            te = d[d["season"] == s].copy()
            tr = d[d["season"] < s]
            if tr.empty:
                print(f"  {s}: no earlier season in these caches: salary baseline uses "
                      f"the other season (NOT walk-forward; reported for shape only)")
                tr = d[d["season"] != s]
            te["p_sal"] = salary_baseline(tr, te, k)
            y = (te["y"] >= k * te["salary"] / 1000.0).to_numpy(float)
            cl = te["date"].to_numpy()
            blk = {"overall": brier_block(te, col, "p_sal", k),
                   "reliability": reliability(te[col].to_numpy(float), y, cl),
                   "reliability_salary": reliability(te["p_sal"].to_numpy(float), y, cl)}
            for by in ("tier", "mins_band", "sal_band"):
                blk[by] = {str(g): brier_block(x, col, "p_sal", k) for g, x in te.groupby(by)}
            sub = te[te["tier"].isin(["bench", "role"]) & te["mins_band"].isin(["8-16", "16-24"])]
            blk["bench_role_8_24"] = brier_block(sub, col, "p_sal", k)
            blk["reliability_bench_role_8_24"] = reliability(
                sub[col].to_numpy(float), (sub["y"] >= k * sub["salary"] / 1000).to_numpy(float),
                sub["date"].to_numpy())
            if (d["season"] < s).any():
                ab = stretch_fit(d[d["season"] < s], k, col)
                te["p_str"] = stretch_apply(te, col, ab)
                blk["stretch_params"] = ab
                blk["stretched"] = brier_block(te, "p_str", "p_sal", k)
                blk["reliability_stretched"] = reliability(te["p_str"].to_numpy(float), y, cl)
            rep[f"{k}x/{s}"] = blk
            o = blk["overall"]
            print(f"\n  {s}: n {o['n']}, base rate {100 * o['base_rate']:.2f}%, mean pred "
                  f"{100 * o['pred_mean']:.2f}%, Brier {o['brier']:.5f} vs salary "
                  f"{o['brier_salary']:.5f} (skill {100 * o['skill_vs_salary']:.1f}%, "
                  f"diff {o['diff']:.5f} ± {o['diff_se']:.5f})")
            print("    reliability (ours):    " + "  ".join(
                f"[{r['bin']}] n{r['n']} {100 * r['pred']:.1f}->{100 * r['obs']:.1f}±"
                f"{100 * r['se']:.1f}" for r in blk["reliability"]))
            print("    reliability (salary):  " + "  ".join(
                f"[{r['bin']}] n{r['n']} {100 * r['pred']:.1f}->{100 * r['obs']:.1f}±"
                f"{100 * r['se']:.1f}" for r in blk["reliability_salary"]))
            for by in ("tier", "mins_band"):
                for g, b in blk[by].items():
                    print(f"    {by:<9} {g:<7} n{b['n']:>6} base {100 * b['base_rate']:5.2f}% "
                          f"pred {100 * b['pred_mean']:5.2f}%  Brier {b['brier']:.5f} vs "
                          f"{b['brier_salary']:.5f}  skill {100 * (b['skill_vs_salary'] or 0):5.1f}%"
                          f" (diff SE {b['diff_se']:.5f})")
            b = blk["bench_role_8_24"]
            print(f"    bench/role 8-24 min: n{b['n']} base {100 * b['base_rate']:.2f}% pred "
                  f"{100 * b['pred_mean']:.2f}% Brier {b['brier']:.5f} vs {b['brier_salary']:.5f}"
                  f" skill {100 * b['skill_vs_salary']:.1f}%")
            print("      reliability: " + "  ".join(
                f"[{r['bin']}] n{r['n']} {100 * r['pred']:.1f}->{100 * r['obs']:.1f}±"
                f"{100 * r['se']:.1f}" for r in blk["reliability_bench_role_8_24"]))
            if "stretched" in blk:
                b = blk["stretched"]
                print(f"    stretched (per-tier logit fit on < {s}): Brier {b['brier']:.5f}, "
                      f"mean pred {100 * b['pred_mean']:.2f}%  params {blk['stretch_params']}")
    rep["selection"] = {**selection(d, seasons, 6), **selection(d, seasons, 7)}
    rep["stretch_all"] = {f"{k}x": stretch_fit(d, k, f"p_ge_{k}x") for k in KS}
    print("\nper-tier stretch fit on ALL seasons given (the live parameters): "
          + json.dumps(rep["stretch_all"]))
    if out_json:
        Path(out_json).write_text(json.dumps(rep, default=str, indent=1))


SEL_K = 6
SEL_P = (0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40)


def selection(d: pd.DataFrame, seasons: list[str], k: int = SEL_K) -> dict:
    """The tag rule without the popularity filter (no ownership history): bench/role,
    P(play) >= 0.85, calibrated chance (per-tier stretch fit on seasons < S) >= p. Per season:
    tagged per day, mean predicted vs observed share of big nights (k = SEL_K), date-clustered
    SE; the salary-band baseline's predicted share on the same rows for contrast."""
    col = f"p_ge_{k}x"
    out = {}
    print(f"\n######## selection (k = {k}; bench/role, P(play) >= {P_PLAY_HIGH}; no "
          f"popularity filter)")
    for s in seasons:
        tr = d[d["season"] < s]
        if tr.empty:
            continue
        te = d[(d["season"] == s) & d["tier"].isin(["bench", "role"])].copy()
        ab = stretch_fit(tr, k, col)
        te["p_str"] = stretch_apply(te, col, ab)
        te["p_sal"] = salary_baseline(tr, te, k)
        te["hit"] = (te["y"] >= k * te["salary"] / 1000.0).astype(float)
        days = te["date"].nunique()
        for p in SEL_P:
            x = te[te["p_str"] >= p]
            if x.empty:
                continue
            obs, se = rate_se(x["hit"].to_numpy(), x["date"].to_numpy())
            r = {"n": int(len(x)), "per_day": len(x) / days, "pred": float(x["p_str"].mean()),
                 "pred_raw": float(x[col].mean()), "pred_salary": float(x["p_sal"].mean()),
                 "obs": obs, "se": se}
            out[f"{k}x/{s}/{p}"] = r
            print(f"  {s} chance >= {p:.2f}: {r['n']:>5} rows ({r['per_day']:.1f}/day of "
                  f"{days}); predicted {100 * r['pred']:.1f}% (raw {100 * r['pred_raw']:.1f}%, "
                  f"salary band {100 * r['pred_salary']:.1f}%) observed {100 * obs:.1f} ± "
                  f"{100 * se:.1f}%")
        for p in (0.25, 0.33):
            x = te[te["p_str"] >= p].sort_values("p_str", ascending=False)
            x = x.groupby("date", group_keys=False).head(SEL_CAP)
            if x.empty:
                continue
            obs, se = rate_se(x["hit"].to_numpy(), x["date"].to_numpy())
            r = {"n": int(len(x)), "per_day": len(x) / days, "pred": float(x["p_str"].mean()),
                 "pred_salary": float(x["p_sal"].mean()), "obs": obs, "se": se}
            out[f"{k}x/{s}/{p}/top{SEL_CAP}"] = r
            print(f"  {s} chance >= {p:.2f}, top {SEL_CAP}/day: {r['n']:>5} rows "
                  f"({r['per_day']:.1f}/day); predicted {100 * r['pred']:.1f}% (salary band "
                  f"{100 * r['pred_salary']:.1f}%) observed {100 * obs:.1f} ± {100 * se:.1f}%")
    return out


SEL_CAP = 5


# ---------------------------------------------------------------------------------- opener

OPENER_NAMES = ("Andre Drummond", "Javonte Green", "Jared McCain", "Duncan Robinson",
                "Paul George")


def opener_frame(conn, day: str, kinds=("lock10", "lock45")) -> pd.DataFrame:
    """Every player of the day's main-slate runs (`kinds`): salary, actual DK, sim mean, q90,
    q99, P(DK >= k x salary/1000) from the run's npz, the popularity proxy (own %, band)."""
    from courtside.proj.store import load_sims
    runs = conn.execute(
        "SELECT r.run_id, r.slate_id, r.kind, r.model_version, r.sim_path FROM dfs_runs r JOIN "
        "dfs_slates s USING (slate_id) WHERE s.nba_day=? AND s.tag='main' AND r.ok=1 AND "
        "r.sim_path IS NOT NULL ORDER BY r.created_at", (day,)).fetchall()
    rows = []
    for run_id, sid, kind, mv, path in runs:
        if kind not in kinds or not Path(path).exists():
            continue
        z = load_sims(path)
        fp = z["fp"].astype(np.float32)
        dk = [str(x) for x in z["dk_id"]]
        own = {str(r[0]): (r[1], r[2]) for r in conn.execute(
            "SELECT player_id, own, band FROM dfs_ownership_proxy WHERE run_id=?", (run_id,))}
        sal = {}
        for r in conn.execute("SELECT site_player_id, salary, player_name FROM dfs_salaries WHERE "
                              "slate_id=? ORDER BY captured_at", (sid,)):
            sal[str(r[0])] = (r[1], r[2])
        grade = {}
        for r in conn.execute("SELECT p.inputs_json, g.actual_dk, p.role, p.p_play, g.actual_min "
                              "FROM player_projections p LEFT JOIN player_projection_grades g ON "
                              "g.projection_id = p.id WHERE p.run_id=?", (run_id,)):
            d = json.loads(r[0] or "{}").get("dk_id")
            if d:
                grade[str(d)] = (r[1], r[2], r[3], r[4])
        for j, d in enumerate(dk):
            if d not in sal:
                continue
            s_, name = sal[d]
            x = fp[:, j]
            g = grade.get(d, (None, None, None, None))
            row = {"run": run_id, "slate": sid, "kind": kind, "model": mv, "name": name,
                   "salary": s_, "actual": g[0], "tier": g[1], "p_play": g[2],
                   "actual_min": g[3], "mean": float(x.mean()),
                   "q90": float(np.quantile(x, .9)), "q99": float(np.quantile(x, .99)),
                   "own": own.get(d, (None, None))[0], "band": own.get(d, (None, None))[1]}
            for k in KS:
                row[f"p_ge_{k}x"] = float((x >= k * s_ / 1000.0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_opener(day: str, names: tuple[str, ...], out: str | None) -> None:
    """Read-only: the opener's main-slate lock runs; P(DK >= k x salary/1000) for `names`."""
    import sqlite3

    from courtside import config
    db = config.data_dir() / "courtside.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = opener_frame(conn, day)
    if out:
        df.to_parquet(out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df[df["name"].isin(names)].round(3).to_string(index=False))


def cmd_harness(season: str, sims: int, n_dates: int = 0) -> None:
    """Scratch-box driver: the avail-protocol harness with the optional tail output, player level
    only (no lineups), for one season (cache keyed apart from every existing cache)."""
    from courtside import db
    from courtside.proj import backtest as B
    conn = db.connect()
    rep = B.run(conn, [season], modes=(), sims=sims, tails=True, workers=1, n_dates=n_dates,
                log=lambda m: print(m, flush=True))
    print(json.dumps({k: rep.get(k) for k in ("seconds", "peak_rss_mb")}), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pit")
    a.add_argument("dirs", nargs="+")
    a.add_argument("--json")
    b = sub.add_parser("bignight")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--json")
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("--sims", type=int, default=4000)
    h.add_argument("--dates", type=int, default=0)
    c = sub.add_parser("opener")
    c.add_argument("--day", default="2026-10-20")
    c.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "pit":
        cmd_pit(args.dirs, args.json)
    elif args.cmd == "harness":
        cmd_harness(args.season, args.sims, args.dates)
    elif args.cmd == "bignight":
        cmd_bignight(args.dirs, args.json)
    else:
        cmd_opener(args.day, OPENER_NAMES, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
