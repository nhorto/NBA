"""Tier match (IMPROVEMENTS.md B11, switch COURTSIDE_TIER_MATCH, tag `+tm`).

Docs: docs/research/tier-match.md.  Code under test: `copula.TIER_MATCH`, `copula.draw_tiers`,
`copula.fit_style_tiers` (the draw's rho / CV tiers from the fit's inputs, r_dk x mpg and mpg).

    python research/tier_match.py harness S PROTO   # minutes_fix_eval's harness (rows parquet) +
                                                    # every draw's tier under both definitions
                                                    # -> OUT/tm_<PROTO>_<S>_<MODEL_VERSION>.parquet
    python research/tier_match.py mismatch FILE...  # tier-mismatch counts of those parquets
    python research/tier_match.py dk OFF ON LABEL   # paired ON - OFF DK CRPS, two-way SEs
                                                    # (metrics.paired_cluster2, date x player),
                                                    # overall / by tier / on the changed-tier rows

Env: TM_OUT (default var/tmp/tm/out).  Both arms run the same harness days and seeds; the tier
switch changes the correlation matrix and CV tiers of the draw only (same minutes, same means).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import minutes_fix_eval as MF  # noqa: E402

from courtside import config  # noqa: E402

OUT = Path(os.environ.get("TM_OUT", config.REPO_ROOT / "var" / "tmp" / "tm" / "out"))
TIER_NAMES = ("star", "role", "bench")


# ============================================================================== capture

def _patch_tiers() -> list:
    """Wrap `copula.draw_tiers`: at every game draw record each player's tier under the
    projected-mean definition (OFF) and the fit's (ON) plus the one the draw used."""
    from courtside.proj import backtest as B
    from courtside.proj import copula as C
    rows: list = []
    cur = {"day": None}
    orig, run_date = C.draw_tiers, B.SeasonRun.run_date

    def draw_tiers(team, mu_played, min_played, parts, played):
        used = orig(team, mu_played, min_played, parts, played)
        pl = pd.concat([p[1] for p in parts], ignore_index=True)
        old = C.role_tiers(team, mu_played, min_played)
        new = C.fit_style_tiers(team, pl, played.mean(0))
        rows.append(pd.DataFrame({
            "date": str(cur["day"]), "team": np.asarray(team).astype(str),
            "player_id": pl["player_id"].to_numpy(np.int64), "t_old": old, "t_new": new,
            "t_used": used, "p_play": played.mean(0), "mu_played": mu_played,
            "min_played": min_played, "mpg": pl["mpg"].to_numpy(float),
            "star_score": (pl["r_dk"].to_numpy(float) * pl["mpg"].to_numpy(float)
                           if "r_dk" in pl else np.nan)}))
        return used

    def rd(self, day):
        cur["day"] = day
        return run_date(self, day)

    C.draw_tiers = draw_tiers
    B.SeasonRun.run_date = rd
    return rows


def cmd_harness(season: str, protocol: str) -> None:
    from courtside.proj import pipeline as P
    rows = _patch_tiers()
    MF.OUT = OUT
    MF.cmd_harness(season, protocol)
    f = OUT / f"tm_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    d.to_parquet(f)
    print(f"[tm] {len(d)} player draws -> {f}", flush=True)


# ============================================================================== mismatch

def mismatch_table(d: pd.DataFrame) -> dict:
    """Counts of player-game rows whose tier differs between the two definitions.  `present` =
    P(play) >= 0.5; `min10` = E[min | plays] >= 10
    (the DK-metric population).  vs_used = rows where the draw's tier differs from the fit's
    definition (0 ON)."""
    d = d.drop_duplicates(["date", "player_id"]).copy()
    out = {"rows": int(len(d)), "dates": int(d["date"].nunique())}
    for nm, sub in (("all", d), ("present", d[d["p_play"] >= 0.5]),
                    ("min10", d[d["min_played"] >= 10])):
        ch = sub["t_old"] != sub["t_new"]
        out[nm] = {"n": int(len(sub)), "changed": int(ch.sum()),
                   "pct": float(100 * ch.mean()) if len(sub) else float("nan"),
                   "vs_used": int((sub["t_used"] != sub["t_new"]).sum()),
                   "old_vs_used": int((sub["t_used"] != sub["t_old"]).sum())}
    p = d[d["p_play"] >= 0.5]
    out["moves_present"] = {f"{TIER_NAMES[a]}->{TIER_NAMES[b]}": int(n) for (a, b), n in
                            p[p["t_old"] != p["t_new"]].groupby(["t_old", "t_new"]).size().items()}
    tg = p.assign(ch=lambda x: x["t_old"] != x["t_new"]).groupby(["date", "team"])["ch"].any()
    out["team_games"] = {"n": int(len(tg)), "with_change": int(tg.sum())}
    return out


def cmd_mismatch(*files: str) -> None:
    for f in files:
        print(f"\n## {Path(f).name}")
        print(json.dumps(mismatch_table(pd.read_parquet(f)), indent=1))


# ============================================================================== DK CRPS

def _row(name: str, d: pd.DataFrame) -> dict:
    from courtside.proj import metrics as MT
    x = (d["crps_on"] - d["crps_off"]).to_numpy(float)
    c1, c2 = d["date"].to_numpy(), d["player_id"].to_numpy()
    r = MT.paired_cluster2(x, c1, c2)
    one = MT.paired_cluster(x, c1)
    se = r["se"]
    return {"set": name, "n": r["n"], "dates": r["n_c1"], "players": r["n_c2"],
            "crps_off": float(d["crps_off"].mean()), "diff": r["mean"], "se_1way": one["se"],
            "se_2way": se, "z_2way": r["mean"] / se if se and se > 0 else float("nan")}


def cmd_dk(off: str, on: str, label: str, mism: str | None = None) -> None:
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    a = a.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"]).reset_index(drop=True)
    d = MF.dk_pair(a, b)
    d = d[d["pop"] & np.isfinite(d["crps_off"]) & np.isfinite(d["crps_on"])].reset_index(drop=True)
    d["player_id"] = d["player_id"].astype(np.int64)
    rows = [_row("all", d)]
    for t, _, _ in MF.TIERS:
        x = d[d["tier4"] == t]
        if len(x) > 50:
            rows.append(_row(f"tier {t}", x))
    mpath = Path(mism) if mism else Path(off.replace("rows_", "tm_"))
    if mpath.exists():
        m = pd.read_parquet(mpath).drop_duplicates(["date", "player_id"])
        m["chg"] = m["t_old"] != m["t_new"]
        nm = dict(enumerate(TIER_NAMES))
        m["tr"] = m["t_old"].map(nm) + ">" + m["t_new"].map(nm)
        d = d.merge(m[["date", "player_id", "chg", "tr"]], on=["date", "player_id"], how="left")
        d["chg"] = d["chg"].fillna(False).astype(bool)
        for name, x in (("tier unchanged", d[~d["chg"]]), ("tier changed", d[d["chg"]])):
            if len(x) > 30:
                rows.append(_row(name, x))
        for tr, x in d[d["chg"]].groupby("tr"):
            if len(x) > 50:
                rows.append(_row(f"  changed {tr}", x))
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tmdk_{label}.json").write_text(json.dumps(rows, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(f"\n## {label}: {Path(off).name} -> {Path(on).name}")
    print(res.round(5).to_string(index=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    m = sub.add_parser("mismatch")
    m.add_argument("files", nargs="+")
    d = sub.add_parser("dk")
    d.add_argument("off")
    d.add_argument("on")
    d.add_argument("label")
    d.add_argument("--mism")
    x = ap.parse_args(argv)
    if x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    elif x.cmd == "mismatch":
        cmd_mismatch(*x.files)
    else:
        cmd_dk(x.off, x.on, x.label, x.mism)


if __name__ == "__main__":
    main()
