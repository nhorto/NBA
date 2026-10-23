"""E2: re-compute the +mf DK CRPS delta vs OFF with one-way (date) and two-way (date x player)
SEs from saved per-row harness outputs (no simulation). Usage:
  python research/e2_twoway_mf.py <dir with rows_<proto>_<season>_pathA-v0.3[+mf].p2 parquets>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minutes_fix_eval as M  # noqa: E402
from courtside.proj import metrics as MT  # noqa: E402

SETS = [("2023-24", "avail", "2023-24"), ("2024-25", "avail", "2024-25"),
        ("live", "live", "2025-26"), ("live10", "live10", "2025-26"),
        ("RG 2019-20", "avail", "2019-20"), ("RG 2020-21", "avail", "2020-21")]


def load(dirp: Path, proto: str, season: str) -> pd.DataFrame:
    f = lambda tag: pd.read_parquet(dirp / f"rows_{proto}_{season}_pathA-v0.3{tag}+p2.parquet")
    a = f("").dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    d = M.dk_pair(a.reset_index(drop=True), f("+mf"))
    d = d[d["pop"]]
    ok = np.isfinite(d["crps_off"]) & np.isfinite(d["crps_on"])
    return d[ok].reset_index(drop=True)


def row(name, d):
    x = (d["crps_on"] - d["crps_off"]).to_numpy(float)
    r = MT.paired_cluster2(x, d["date"].to_numpy(), d["player_id"].to_numpy())
    one = MT.paired_cluster(x, d["date"].to_numpy())
    return {"set": name, "n": r["n"], "n_dates": r["n_c1"], "n_players": r["n_c2"],
            "diff": r["mean"], "se_1way": one["se"], "se_player": r["se_c2"], "se_2way": r["se"],
            "ratio": r["se"] / one["se"], "z_1way": r["mean"] / one["se"],
            "z_2way": r["mean"] / r["se"]}


if __name__ == "__main__":
    dirp = Path(sys.argv[1])
    out, frames = [], []
    for name, proto, season in SETS:
        try:
            d = load(dirp, proto, season)
        except FileNotFoundError:
            continue
        out.append(row(name, d))
        if name in ("2023-24", "2024-25"):
            d["season"] = season
            frames.append(d)
    if frames:
        out.append(row("2023-25 pooled", pd.concat(frames, ignore_index=True)))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).round(5).to_string(index=False))
