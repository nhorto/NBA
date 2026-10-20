"""Phase 2F second ablation: F3 tracking, F5 matchups, F6 schedule on top of the shipped P+F2.

Walk-forward per test season S (every layer fit on the 6 regular seasons < S, early stopping on
the last of them), per-minute NB log-lik and DK MAE / CRPS given actual minutes on the season-S
regular-season rows, paired against P+F2 with two-way (game x player-season) and date-clustered
SEs. LightGBM on one thread (deterministic; the Mac is shared).

    python research/context_ablation2.py 2023-24 [--variants ...] [--out DIR]

Writes DIR/ablation2_<S>.json and .txt (default <data>/reports/context_ablation2/).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from courtside import config
from courtside.proj import context as CX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("season")
    ap.add_argument("--variants", nargs="+", default=list(CX.VARIANTS_2))
    ap.add_argument("--staged", action="store_true", help="F5 / F6 on top of P+F2+F3")
    ap.add_argument("--tag", default="ablation2")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    CX.LGB_PARAMS["num_threads"] = a.threads
    if a.staged:
        a.variants = list(CX.VARIANTS_3)
    p = a.db or str(config.db_path())
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    t0 = time.time()
    ev = CX.run_ablation(conn, a.season, variants={v: CX.VARIANTS[v] for v in a.variants},
                         save_models=True, log=lambda m: print(m, flush=True))
    out = Path(a.out) if a.out else config.data_dir() / "reports" / "context_ablation2"
    out.mkdir(parents=True, exist_ok=True)
    txt = CX.format_ablation(ev)
    (out / f"{a.tag}_{a.season}.json").write_text(json.dumps(ev, indent=1, default=float))
    (out / f"{a.tag}_{a.season}.txt").write_text(txt + "\n")
    print(txt, flush=True)
    print(f"done {a.season} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
