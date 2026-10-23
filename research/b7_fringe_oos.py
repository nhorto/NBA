"""B7 (docs/IMPROVEMENTS.md): the fringe P(play) "mid" model, out of sample.

The shipped `fringe_params.json` "mid" model (`availability.FRINGE_FEATURES["mid"]`) was fit on
the T2.4 slice's own fringe rows (research/p2_fit.py), and the p2-vs-v1 slice comparison then
scored the slice with it. This script:

    python research/b7_fringe_oos.py fit ROWS_PARQUET OUT_DIR
        Fringe log loss, same rows, three ways: SHIPPED params, REFIT on all rows (in-sample), and
        OUT OF SAMPLE: leave-one-month-out (lomo: the month's own rows never in its fit) and
        forward (fwd: months before only; the first month has no fit and uses lomo's). Writes
        OUT_DIR/fringe_lomo.json ({month: FringeParams json}, the params the slice run uses for
        each day) and OUT_DIR/b7_fringe.json (metrics). The packaged fringe_params.json is only
        READ, never written.
    B7_LOMO=OUT_DIR/fringe_lomo.json python research/b7_fringe_oos.py slice PROTOCOL [--sims N]
        research/t24_slice.py run with `availability.load_fringe` returning the params of the
        day's month (fringe_features sees the panel's date first), nothing else changed. With
        B7_LOMO unset it runs the shipped params (the control: must equal the stored p2 frames).

Why leave-month-out: the fringe population needs real report status ("not on any report"), and
the official reports start 2025-12-22, so there are no earlier-season rows to fit on (seasons < S
would leave nothing for 2025-26; RotoGuru-era rows have no reports and are used only for the
separate "early" regime). The slice has five calendar months, which is also how the report cells
and the pq offsets are held out (`pq_lomo.json`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from courtside.proj import availability as AV

LISTED = {"out", "doubtful", "questionable", "probable", "available", "removed", "daytoday"}


def ll_vec(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mid_rows(rows_file: Path) -> pd.DataFrame:
    """The slice's "mid" fringe rows (p2_fit.fringe + regime filter)."""
    S = pd.read_parquet(rows_file)
    F = S[(S["roster"] == 0) & ~S["status"].isin(LISTED)]
    F = F[F["regime"] == "mid"].copy()
    F["month"] = F["date"].str[:7]
    return F.reset_index(drop=True)


def _predict(F: pd.DataFrame, params: AV.FringeParams) -> np.ndarray:
    return params.p(F.assign(regime="mid"))


def cluster_se(d: np.ndarray, groups: np.ndarray) -> float:
    """SE of mean(d) clustering rows by `groups` (dates)."""
    s = pd.Series(d - d.mean()).groupby(groups).sum()
    return float(np.sqrt((s ** 2).sum()) / len(d))


def cmd_fit(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F = mid_rows(Path(a.rows))
    y = F["played"].to_numpy(float)
    months = sorted(F["month"].unique())
    shipped = AV.FringeParams.from_json(json.loads(AV.FRINGE_PATH.read_text()))
    full = AV.fit_fringe_model(F, "mid")
    p_ship = _predict(F, shipped)
    p_full = _predict(F, AV.FringeParams(models={"mid": full}))
    p_lomo = np.zeros(len(F))
    p_fwd = np.full(len(F), np.nan)
    lomo = {}
    for m in months:
        te = (F["month"] == m).to_numpy()
        mod = AV.fit_fringe_model(F[~te], "mid")
        fp = AV.FringeParams(models={"mid": mod}, meta={"held_out_month": m,
                                                         "n_train": int((~te).sum())})
        lomo[m] = fp.to_json()
        p_lomo[te] = _predict(F[te], fp)
        tr = (F["month"] < m).to_numpy()
        if tr.sum() >= 30:
            p_fwd[te] = _predict(F[te], AV.FringeParams(models={"mid": AV.fit_fringe_model(
                F[tr], "mid")}))
    (out / "fringe_lomo.json").write_text(json.dumps(lomo, indent=1) + "\n")

    ybar = float(y.mean())
    res = {"rows": a.rows, "n": int(len(F)), "played": ybar, "months": months,
           "n_by_month": {m: int((F["month"] == m).sum()) for m in months},
           "coef_full": dict(zip(full["features"], map(float, full["coef"]))),
           "coef_shipped": dict(zip(shipped.models["mid"]["features"],
                                    shipped.models["mid"]["coef"])),
           "coef_lomo": {m: dict(zip(AV.FRINGE_FEATURES["mid"],
                                     lomo[m]["models"]["mid"]["coef"])) for m in months}}
    dates = F["date"].to_numpy()
    p_cl = np.zeros(len(F))             # constant = the training months' base rate (honest const)
    for m in months:
        te = (F["month"] == m).to_numpy()
        p_cl[te] = float(y[~te].mean())
    arms = {"const": np.full(len(F), ybar), "const_lomo": p_cl, "shipped": p_ship, "refit_in_sample": p_full,
            "lomo": p_lomo, "fwd": p_fwd, "engine": F["pp_engine"].to_numpy(float)}
    L = {k: ll_vec(y, p) for k, p in arms.items()}
    for k, p in arms.items():
        ok = np.isfinite(p)
        res[k] = {"n": int(ok.sum()), "p": float(p[ok].mean()), "played": float(y[ok].mean()),
                  "ll": float(L[k][ok].mean()), "brier": float(((p[ok] - y[ok]) ** 2).mean())}
    for a_, b_ in (("lomo", "refit_in_sample"), ("lomo", "shipped"), ("shipped", "refit_in_sample"),
                   ("lomo", "const"), ("refit_in_sample", "const"), ("lomo", "const_lomo")):
        ok = np.isfinite(arms[a_]) & np.isfinite(arms[b_])
        d = L[a_][ok] - L[b_][ok]
        res[f"d_ll[{a_}-{b_}]"] = {"diff": float(d.mean()), "se": cluster_se(d, dates[ok])}
    ok = np.isfinite(p_fwd)
    res["fwd_vs_lomo_same_rows"] = {"n": int(ok.sum()),
                                    "fwd": float(L["fwd"][ok].mean()),
                                    "lomo": float(L["lomo"][ok].mean())}
    res["by_month"] = {m: {"n": int((F["month"] == m).sum()),
                           **{k: float(L[k][(F["month"] == m).to_numpy()
                                            & np.isfinite(arms[k])].mean())
                              for k in ("const", "shipped", "refit_in_sample", "lomo")}}
                       for m in months}
    # largest per-row shift in P between the shipped params and the month-held-out ones
    res["p_shift_lomo_minus_shipped"] = {
        "mean": float((p_lomo - p_ship).mean()), "mean_abs": float(np.abs(p_lomo - p_ship).mean()),
        "max_abs": float(np.abs(p_lomo - p_ship).max())}
    (out / "b7_fringe.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def cmd_slice(a) -> None:
    lomo = {}
    if os.environ.get("B7_LOMO"):
        lomo = {m: AV.FringeParams.from_json(d)
                for m, d in json.loads(Path(os.environ["B7_LOMO"]).read_text()).items()}
    state = {"month": None}
    orig_feats, orig_load = AV.fringe_features, AV.load_fringe

    def feats(panel, h, salary=None):
        if len(panel):
            d = np.datetime64(int(panel["date"].iloc[0]), "D")
            state["month"] = str(d)[:7]
            assert (panel["date"].astype(int).to_numpy() == int(panel["date"].iloc[0])).all()
        return orig_feats(panel, h, salary)

    def load(path=None):
        if not lomo or path is not None:
            return orig_load(path)
        return lomo[state["month"]]            # KeyError = a date outside the fitted months

    AV.fringe_features, AV.load_fringe = feats, load
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import t24_slice
    # the scratch box's no-heavy-work windows (ET): nothing 17:45-19:30, on top of the lock windows
    t24_slice.BLACKOUTS_ET = tuple(t24_slice.BLACKOUTS_ET) + (((17, 45), (19, 30)),
                                                             ((21, 15), (22, 45)))
    print(f"[b7] fringe {'LOMO months ' + str(sorted(lomo)) if lomo else 'SHIPPED (control)'}",
          flush=True)
    t24_slice.main(["run", a.protocol, "--sims", str(a.sims), "--chunk", "4"]
                   + (["--dates", str(a.dates)] if a.dates else []))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("rows")
    f.add_argument("out")
    s = sub.add_parser("slice")
    s.add_argument("protocol", choices=["live", "live10"])
    s.add_argument("--sims", type=int, default=4000)
    s.add_argument("--dates", type=int, default=0)
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "slice": cmd_slice}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
