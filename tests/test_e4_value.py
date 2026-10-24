"""E4: DFS-relevant population and the value-ranking metric (proj/metrics.py, proj/backtest.py
dfs_pop_report / slice rows, proj/value_eval.py on live grades, proj/shadows.py reported-only
block). Report-only additions: existing numbers and shadow decisions must not move."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import backtest as B
from courtside.proj import metrics as MT
from courtside.proj import shadows as SH
from courtside.proj import value_eval as V


# ============================================================================== metrics

def test_dfs_population():
    sal = np.array([3000, 4000, np.nan, 3500, np.nan])
    ours = np.array([10, 5, 16, 14, 3])
    fp = np.array([np.nan, 2, 1, 15, 2])
    assert MT.dfs_population(sal, ours).tolist() == [False, True, True, False, False]
    assert MT.dfs_population(sal, ours, fp).tolist() == [False, True, True, True, False]
    assert MT.dfs_population(None, ours).tolist() == [False, False, True, False, False]
    with pytest.raises(ValueError):
        MT.dfs_population(None)


def test_salary_implied_is_a_per_slate_ols_line():
    rng = np.random.default_rng(0)
    sal = rng.uniform(3000, 11000, 200)
    slate = np.repeat(["a", "b"], 100)
    y = np.where(slate == "a", 2.0 + 0.005 * sal, -1.0 + 0.004 * sal)
    s = MT.salary_implied(y, sal, slate)
    assert np.allclose(s, y)
    y2 = y + rng.normal(0, 3, 200)
    s2 = MT.salary_implied(y2, sal, slate)
    for g in ("a", "b"):
        m = slate == g
        b, a = np.polyfit(sal[m], y2[m], 1)
        assert np.allclose(s2[m], a + b * sal[m])
    tiny = MT.salary_implied(y[:3], sal[:3], np.array(["c"] * 3))
    assert np.isnan(tiny).all()


def test_corr_cluster_matches_pearson_and_iid_se():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    cl = np.arange(n)                         # one row per cluster = the iid case
    c = MT.corr_cluster(x, y, cl)
    r = np.corrcoef(x, y)[0, 1]
    assert c["r"] == pytest.approx(r)
    # normal-theory SE of r: (1 - r^2) / sqrt(n)
    assert c["se"] == pytest.approx((1 - r * r) / np.sqrt(n), rel=0.08)
    # clustered SE widens when rows share a cluster effect
    g = np.repeat(np.arange(40), n // 40)
    xs = x + rng.normal(size=40)[g]
    ys = y + rng.normal(size=40)[g]
    assert MT.corr_cluster(xs, ys, g)["se"] > MT.corr_cluster(xs, ys, np.arange(n))["se"]


def test_corr_paired_cluster():
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.normal(size=n)
    a = y + rng.normal(0, 1.0, n)
    b = y + rng.normal(0, 2.0, n)
    d = rng.integers(0, 30, n)
    p = MT.corr_paired_cluster(a, b, y, d)
    assert p["r_a"] == pytest.approx(np.corrcoef(a, y)[0, 1])
    assert p["r_b"] == pytest.approx(np.corrcoef(b, y)[0, 1])
    assert p["diff"] == pytest.approx(p["r_a"] - p["r_b"]) and p["z"] > 5
    same = MT.corr_paired_cluster(a, a, y, d)
    assert same["diff"] == 0.0 and same["se"] == 0.0 and same["z"] is None


def test_value_ranking():
    rng = np.random.default_rng(3)
    n_slates, per = 12, 60
    slate = np.repeat(np.arange(n_slates), per)
    sal = rng.uniform(3000, 11000, n_slates * per)
    skill = rng.normal(0, 6, n_slates * per)            # value beyond salary
    y = 5 * sal / 1000 + skill + rng.normal(0, 8, n_slates * per)
    good = 5 * sal / 1000 + skill
    blind = 5 * sal / 1000 + rng.normal(0, 1, n_slates * per)   # salary + noise only
    v = MT.value_ranking({"good": good, "blind": blind, "actual": y}, y, sal, slate, slate,
                         base="blind")
    f = v["forecasts"]
    assert f["actual"]["r"] == pytest.approx(1.0)
    assert f["good"]["r"] > 0.4 and abs(f["blind"]["r"]) < 0.15
    assert v["vs_base"]["good"]["diff"] > 0.3 and v["vs_base"]["good"]["z"] > 3
    assert v["n"] == n_slates * per and v["n_slates"] == n_slates
    # rows with a missing forecast are dropped for every forecast (identical rows)
    g2 = good.copy()
    g2[:10] = np.nan
    v2 = MT.value_ranking({"good": g2, "blind": blind}, y, sal, slate, slate)
    assert v2["n"] == n_slates * per - 10


# ============================================================================== harness

def _players(n_dates=8, per=40, salaried=True, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for p in range(per):
            sal = float(rng.choice(np.arange(3000, 11001, 100)))
            mean = 5 * sal / 1000 + rng.normal(0, 4)
            y = max(0.0, mean + rng.normal(0, 9))
            rows.append({"date": f"2020-01-{d + 1:02d}", "season": "2019-20", "player_id": p,
                         "salary": sal if salaried else np.nan, "y": y,
                         "minutes": 20.0 if y > 0 else 0.0, "has_proj": True, "mean": mean,
                         "q10": mean - 10, "q25": mean - 5, "q50": mean, "q75": mean + 5,
                         "q90": mean + 10, "exp_min": 25.0 if mean > 12 else 8.0, "p_play": 0.9,
                         "tier": "role", "crps": abs(y - mean) * 0.7, "pit": rng.uniform(),
                         "b_fppg": 5 * sal / 1000 + rng.normal(0, 6), "b_season_avg": mean + 1,
                         "b_last10": mean, "sal_all": 5 * sal / 1000, "sal_prev": 5 * sal / 1000,
                         "sal_played": 5 * sal / 1000, "sal_fixed": 5 * sal / 1000,
                         "min_played": 25.0, "p_avail": 0.99, "status": "", "game_id": f"g{p % 4}"})
    return pd.DataFrame(rows)


def test_player_report_gains_the_dfs_block_and_keeps_everything_else():
    df = _players()
    rep = B.player_report(df)
    d = rep["dfs_pop"]
    m = MT.dfs_population(df["salary"], df["mean"], df["b_fppg"])
    assert d["n"] == int(m.sum())
    assert d["mae"] == pytest.approx(np.abs(df["mean"][m] - df["y"][m]).mean())
    v = d["value"]
    assert set(v["forecasts"]) == {"ours", "fppg_proxy", "season_avg"}
    assert v["vs_base"]["ours"]["base"] == "fppg_proxy"
    # the existing keys are computed exactly as before (on E[min] >= 10)
    u = df[df["exp_min"] >= 10]
    assert rep["n"] == len(u) and rep["mae"] == pytest.approx(np.abs(u["mean"] - u["y"]).mean())
    json.dumps(rep, default=float)
    txt = "\n".join(B._fmt_player(rep))
    assert "E4 DFS rows" in txt and "value corr" in txt


def test_player_report_without_salaries_has_no_value_block():
    rep = B.player_report(_players(salaried=False))
    assert rep["dfs_pop"]["n"] > 0 and "value" not in rep["dfs_pop"]


def test_slice_report_adds_dfs_cells_only():
    a = _players(salaried=False)
    b = a.assign(mean=a["mean"] + 1.0)
    rep = B.slice_report({("x", "live"): a, ("y", "live"): b}, reference="avail", base_arm="x")
    rows = B.slice_rows(a)
    assert rows["V"].tolist() == (rows["mean"] >= 15).tolist()      # no salary: projected >= 15
    for m in B.SLICE_METRICS:                                         # unchanged definitions
        assert rep["cells"]["x|live"][m] == B.metric_se(rows, m)
    c = rep["dfs_pop"]["x|live"]["dk_mae_dfs"]
    v = rows[rows["V"]]
    assert c["value"] == pytest.approx(np.abs(v["mean"] - v["y"]).mean()) and c["n"] == len(v)
    assert "y|live" in rep["dfs_pop_vs_base_arm"]
    assert "E4 DFS-relevant rows" in B.format_slice_report(rep)


# ============================================================================== live grades

def _grades_db(tmp_path):
    c = db.connect(tmp_path / "g.db")
    rng = np.random.default_rng(5)
    with c:
        for sid, day, tag, lock in (("100", "2026-10-21", "main", "2026-10-21T23:00:00Z"),
                                    ("101", "2026-10-22", "main", "2026-10-22T23:00:00Z"),
                                    ("102", "2026-10-22", "turbo", "2026-10-23T00:00:00Z"),
                                    ("103", "2026-10-22", "showdown", "2026-10-22T23:00:00Z")):
            c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, lock_utc) "
                      "VALUES (?,?,?,?,?)", (sid, day, 81 if tag == "showdown" else 70, tag, lock))
        runs = [("r100a", "100", "lock45", "2026-10-21T22:15:00Z"),
                ("r100b", "100", "lock10", "2026-10-21T22:50:00Z"),
                ("r101", "101", "lock10", "2026-10-22T22:50:00Z"),
                ("r102", "102", "lock10", "2026-10-22T23:50:00Z"),
                ("r103", "103", "lock10", "2026-10-22T22:50:00Z")]
        for rid, sid, kind, ts in runs:
            c.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, ok) VALUES "
                      "(?,?,?,?,1)", (rid, sid, kind, ts))
        pid = 0
        for rid, sid, kind, ts in runs:
            day = "2026-10-21" if sid == "100" else "2026-10-22"
            for p in range(30):
                player = p + (0 if sid == "100" else 1000)        # 102 repeats 101's rows
                game = f"g{sid if sid != '102' else '101'}_{p % 3}"
                sal = 3000 + 250 * p
                fppg = 5 * sal / 1000 + rng.normal(0, 6)
                y = 5 * sal / 1000 + rng.normal(0, 8)
                proj = y + rng.normal(0, 6)
                status = "OUT" if p == 29 else "None"
                pid += 1
                c.execute("INSERT INTO player_projections (id, game_id, nba_day, slate_id, "
                          "player_id, run_id, created_at, model_version, inputs_json) VALUES "
                          "(?,?,?,?,?,?,?,?,?)", (pid, game, day, sid, player, rid, ts,
                                                  "pathA-v0.3", json.dumps({"dk_id": f"dk{player}"})))
                c.execute("INSERT INTO player_projection_grades (projection_id, game_id, "
                          "player_id, run_id, graded_at, played, actual_min, actual_dk, proj_min, "
                          "proj_dk, crps, dk_fppg, salary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (pid, game, player, rid, "x", 1, 30.0, y, 25.0 if p % 5 else 6.0,
                           proj + (100 if rid == "r100a" else 0), abs(y - proj) * 0.7, fppg, sal))
                c.execute("INSERT OR IGNORE INTO dfs_salaries (captured_at, site, slate_id, "
                          "game_type, player_name, site_player_id, salary, status, disabled) "
                          "VALUES (?, 'dk', ?, 'classic', ?, ?, ?, ?, 0)",
                          ("2026-10-21T12:00:00Z", sid, f"P{player}", f"dk{player}", sal, status))
    return c


def test_live_frame_dedupes_and_flags_out(tmp_path):
    c = _grades_db(tmp_path)
    df = V.live_frame(c, "2026-10-20", "2026-10-23")
    assert not df.duplicated(["game_id", "player_id"]).any()
    assert set(df["slate_id"]) == {"100", "101"}         # showdown dropped, main beats turbo
    assert (df.loc[df["slate_id"] == "100", "run_id"] == "r100b").all()   # lock10 over lock45
    assert (df["dk_status"] == "OUT").sum() == 2


def test_live_report_numbers(tmp_path):
    c = _grades_db(tmp_path)
    rep = V.live_report(c, "2026-10-20", "2026-10-23")
    df = V.live_frame(c, "2026-10-20", "2026-10-23")
    pops = V.populations(df)
    d = df[pops["dfs"]]
    assert (d["dk_status"] != "OUT").all()
    cut = rep["cuts"]["dfs"]
    assert cut["n"] == len(d)
    ref = MT.value_ranking({"ours": d["proj_dk"].to_numpy(float),
                            "fppg": d["dk_fppg"].to_numpy(float)}, d["actual_dk"].to_numpy(float),
                           d["salary"].to_numpy(float), d["slate_id"].astype(str).to_numpy(),
                           d["day"].to_numpy(), base="fppg")
    got = cut["value"]["vs_base"]["ours"]
    assert got["r_a"] == pytest.approx(ref["vs_base"]["ours"]["r_a"])
    assert got["diff"] == pytest.approx(ref["vs_base"]["ours"]["diff"])
    assert got["se_game"] is not None
    assert cut["mae_vs_fppg"]["diff"] == pytest.approx(
        (np.abs(d["proj_dk"] - d["actual_dk"]) - np.abs(d["dk_fppg"] - d["actual_dk"])).mean())
    assert set(rep["by_night"]) == {"2026-10-21", "2026-10-22"}
    json.dumps(rep)
    assert "value corr ours" in V.format_report(rep)


# ============================================================================== shadows

def _shadow_rows(seed=6, n_nights=6, n_players=40, arm_better=True):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_nights):
        for p in range(n_players):
            sal = 3000 + 200 * p
            y = 5 * sal / 1000 + rng.normal(0, 8)
            c = y + rng.normal(0, 8)
            a = y + rng.normal(0, 5 if arm_better else 11)
            out.append({"day": f"2026-11-{k + 1:02d}", "run": "1330", "slot": "13:30", "arm": "+x",
                        "game_id": f"g{k}_{p % 3}", "player_id": p, "tier": ("star", "role",
                                                                             "bench")[p % 3],
                        "played": 1, "y": y, "amin": 20.0, "emin_c": 20.0, "emin_a": 20.0,
                        "crps_c": abs(y - c) * 0.6, "crps_a": abs(y - a) * 0.6,
                        "ae_c": abs(y - c), "ae_a": abs(y - a), "mae_c": 1.0, "mae_a": 1.0,
                        "ll_c": 0.3, "ll_a": 0.3, "changed": 1, "dk_c": c, "dk_a": a,
                        "salary": float(sal)})
    return pd.DataFrame(out)


def _entry(**kw):
    e = {"arm": "+x", "metric": "dk_crps", "population": "emin10", "run": "13:30",
         "bar": {"direction": "lower", "z": -2.0, "tier_z": 2.0}, "min_nights": 3,
         "max_nights": 60, "status": "active"}
    e.update(kw)
    return e


def test_shadow_value_is_reported_only():
    rows = _shadow_rows()
    with_extra = SH.arm_report(rows, _entry())
    without = SH.arm_report(rows.drop(columns=list(SH.EXTRA_COLS)), _entry())
    ro = with_extra.pop("reported_only")
    assert without.pop("reported_only") == {}
    assert json.dumps(with_extra, sort_keys=True, default=str) == \
        json.dumps(without, sort_keys=True, default=str)        # decision and numbers unchanged
    assert ro["n"] > 0 and ro["value"]["r_arm"] > ro["value"]["r_champion"]
    assert ro["value"]["diff"] > 0 and ro["value"]["nights"] == 6
    assert ro["dk_crps_dfs"]["mean"] < 0
