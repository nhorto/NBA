"""`+ep` OFF identity at the frame level: `git show HEAD:src/courtside/proj/state.py` vs the
working-tree state.py (switch OFF = no `ep`), on the real DB.

    python research/ep_state_identity.py [SEASON ...] [--day YYYY-MM-DD ...]

walk_forward of each season and rate_frame / build_state on each day (every player with a game in
the last 400 days) are compared with `assert_frame_equal(check_exact=True)`. The pipeline-level
check (HEAD state.py + pipeline.py tree vs the code tree, projections and sims) runs on the Mac
scratch (`courtside-compute/ep/lane_id.sh`).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def head_state():
    src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:src/courtside/proj/state.py"],
                         check=True, capture_output=True, text=True).stdout
    d = Path(tempfile.mkdtemp(prefix="ep_head_"))
    f = d / "state_head.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("courtside.proj._state_head", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="*", default=["2023-24", "2025-26"])
    ap.add_argument("--day", action="append", default=None)
    a = ap.parse_args(argv)
    from courtside import db
    from courtside.proj import state as ST
    H = head_state()
    conn = db.connect()
    pos = ST.load_positions(conn)
    ok = True
    for s in a.seasons:
        first = ST.season_label(ST.sidx_of(s) - 6)
        g = ST.load_games(conn, since_season=first, until_season=s)
        x = H.walk_forward(conn, s, H.DEFAULT_PARAMS, games=g, pos=pos)
        y = ST.walk_forward(conn, s, ST.DEFAULT_PARAMS, games=g, pos=pos)
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            print(f"walk_forward {s}: identical ({x.shape[0]} x {x.shape[1]})")
        except AssertionError as e:
            ok = False
            print(f"walk_forward {s}: DIFFERENT {str(e)[:300]}")
    for d in a.day or ["2023-10-26", "2024-01-10", "2026-06-01"]:
        dd = date.fromisoformat(d)
        x, y = H.build_state(conn, dd), ST.build_state(conn, dd)
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            print(f"build_state {d}: identical ({x.shape[0]} rows)")
        except AssertionError as e:
            ok = False
            print(f"build_state {d}: DIFFERENT {str(e)[:300]}")
    print("ALL IDENTICAL" if ok else "NOT IDENTICAL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
