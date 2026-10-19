"""eval-lite (proj/evallite.py): synthetic checks of the statistics, the walk-forward salary line,
the gate logic and the lineup scoring (always with ACTUAL points). No pipeline run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import evallite as E
from courtside.proj.dfs import rules as R

FIX = Path(__file__).parent / "fixtures" / "dfs"
CLASSIC = R.Rules.from_payload(json.loads((FIX / "rules_70.json").read_text()))


# --------------------------------------------------------------------------- statistics

def test_paired_cluster_matches_formula():
    d = np.array([1.0, 3.0, -2.0, 4.0, 0.0, 2.0])
    cl = np.array(["a", "a", "b", "b", "c", "c"])
    r = E.paired_cluster(d, cl)
    dbar = d.mean()
    s = np.array([(1 - dbar) + (3 - dbar), (-2 - dbar) + (4 - dbar), (0 - dbar) + (2 - dbar)])
    want = np.sqrt(3 / 2 * (s ** 2).sum()) / 6
    assert r["mean"] == pytest.approx(dbar) and r["se"] == pytest.approx(want)
    assert r["n"] == 6 and r["n_clusters"] == 3


def test_paired_cluster_singletons_is_plain_se():
    rng = np.random.default_rng(1)
    d = rng.normal(0.5, 2.0, 200)
    r = E.paired_cluster(d, np.arange(200))
    assert r["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(200))


def test_paired_cluster_correlated_cluster_widens_se():
    rng = np.random.default_rng(2)
    shared = np.repeat(rng.normal(0, 3, 40), 10)          # a date effect shared by 10 rows
    d = shared + rng.normal(0, 1, 400)
    naive = E.paired_cluster(d, np.arange(400))["se"]
    clustered = E.paired_cluster(d, np.repeat(np.arange(40), 10))["se"]
    assert clustered > 2 * naive


def test_gain_pct_and_sign():
    base = np.array([10.0, 10.0, 10.0, 10.0])
    model = np.array([9.0, 9.0, 9.0, 9.0])
    g = E.gain_pct(base, model, np.array([1, 1, 2, 2]))
    assert g["gain_pct"] == pytest.approx(10.0) and g["mae_base"] == 10.0 and g["mae_model"] == 9.0
    assert E.gain_pct(model, base, np.array([1, 1, 2, 2]))["gain_pct"] < 0


# --------------------------------------------------------------------------- salary line

def _salary_rows():
    """2018-19: pts = 0.004 * salary (slope 4/1k). 2019-20 and later: pts = 0.010 * salary."""
    rows = []
    for day, slope in (("2018-11-01", 0.004), ("2019-02-01", 0.004), ("2019-11-01", 0.010),
                       ("2020-08-05", 0.010), ("2020-12-30", 0.010)):
        for sal in range(3000, 11000, 200):
            rows.append({"game_date": day, "salary": sal, "pts": slope * sal, "minutes": 20.0})
    return pd.DataFrame(rows)


def test_salary_line_uses_only_earlier_seasons():
    rows = _salary_rows()
    a, b = E.fit_salary_line(rows, "2019-20")
    assert b == pytest.approx(0.004) and a == pytest.approx(0.0, abs=1e-9)
    # rows of S and later are ignored: scrambling them changes nothing
    scr = rows.copy()
    late = scr["game_date"] >= "2019-09-01"
    scr.loc[late, "pts"] = 1234.5
    assert E.fit_salary_line(scr, "2019-20") == (a, b)
    # for 2020-21 the bubble (Aug 2020) is 2019-20: it is training data, Dec 2020 is not
    a2, b2 = E.fit_salary_line(rows, "2020-21")
    assert 0.004 < b2 < 0.010
    scr2 = rows.copy()
    scr2.loc[scr2["game_date"] >= "2020-11-01", "pts"] = 999.0
    assert E.fit_salary_line(scr2, "2020-21") == (a2, b2)
    # the bubble rows do matter for 2020-21
    nob = rows[rows["game_date"] != "2020-08-05"]
    assert E.fit_salary_line(nob, "2020-21") != (a2, b2)


def test_salary_line_modes():
    rows = _salary_rows()
    assert E.fit_salary_line(rows, "2020-21", "prev_season")[1] == pytest.approx(0.010)
    played = rows.copy()
    played["minutes"] = 0.0
    played.loc[played["salary"] >= 6000, "minutes"] = 25.0
    played.loc[played["minutes"] == 0, "pts"] = 0.0
    a_all, b_all = E.fit_salary_line(played, "2019-20", "all_seasons")
    a_pl, b_pl = E.fit_salary_line(played, "2019-20", "played_only")
    assert (a_all, b_all) != (a_pl, b_pl)
    with pytest.raises(ValueError):
        E.fit_salary_line(rows, "2019-20", "nope")
    with pytest.raises(ValueError):
        E.fit_salary_line(rows.iloc[:10], "2019-20")


def test_season_boundaries_put_the_bubble_in_2019_20():
    assert E.season_label("2020-08-10") == "2019-20"
    assert E.season_label("2020-10-05") == "2019-20"
    assert E.season_label("2020-12-25") == "2020-21"
    assert E.season_label("2019-10-22") == "2019-20"
    assert E.season_label("2021-05-01") == "2020-21"
    assert E.is_rotoguru_season("2019-20") and E.is_rotoguru_season("2020-21")
    assert not E.is_rotoguru_season("2024-25")


# --------------------------------------------------------------------------- gate logic

def test_judge_thresholds_are_inclusive():
    assert E.judge(3.0, "ge", 3.0) and not E.judge(2.99, "ge", 3.0)
    assert E.judge(4.8, "le", 4.8) and not E.judge(4.81, "le", 4.8)
    assert E.judge(0.76, "between", (0.76, 0.84)) and E.judge(0.84, "between", (0.76, 0.84))
    assert not E.judge(0.7599, "between", (0.76, 0.84)) and not E.judge(0.8401, "between",
                                                                         (0.76, 0.84))
    assert E.judge(None, "ge", 1) is None and E.judge(float("nan"), "le", 1) is None


def _results(gain=5.0, se=0.6, cov=0.82, mae=4.6, sal_gain=(2.0, 1.0)):
    fp = {"gain_vs_std_pct": gain, "gain_vs_std_se_pct": se, "n": 8000, "cov80": cov,
          "cov80_se": 0.01}
    proj = {"2024-25": {"fp": fp, "minutes": {"mae": mae, "mae_se": 0.05, "n": 7000}}}
    rg = {}
    for s, g in zip(("2019-20", "2020-21"), sal_gain):
        rg[s] = {"player": {"exp_min_ge_10": {"vs_salary_best": {"gain_pct": g, "se_pct": 0.8,
                                                                 "n": 9000}}},
                 "lineup": {"paired": {"ours_mean_minus_salary": {"mean_diff": 5.0, "se": 2.0,
                                                                  "z": 2.5, "n": 140},
                                       "ours_cash_minus_salary": {"mean_diff": -1.0, "se": 2.0,
                                                                  "z": -0.5, "n": 140}}}}
    return {"projection": proj, "rotoguru": rg}


def test_gate_table_all_pass():
    rows = E.gate_table(_results())
    core = [r for r in rows if r["core"]]
    assert [r["id"] for r in core] == ["G1", "G3", "G4", "G2", "G2"]
    assert all(r["verdict"] == "PASS" for r in core)
    assert E.overall(rows) == "PASS"
    sup = {r["desc"].split()[1]: r for r in rows if not r["core"]}
    assert len([r for r in rows if not r["core"]]) == 4
    g1 = next(r for r in rows if r["id"] == "G1")
    assert g1["sig"] is True and g1["value"] == 5.0 and g1["n"] == 8000
    assert sup  # supplementary lines exist and never decide the verdict


@pytest.mark.parametrize("kw,failing", [
    ({"gain": 2.9}, "G1"), ({"cov": 0.75}, "G3"), ({"cov": 0.85}, "G3"),
    ({"mae": 4.9}, "G4"), ({"sal_gain": (2.0, -0.1)}, "G2"),
])
def test_gate_table_fails_where_expected(kw, failing):
    rows = E.gate_table(_results(**kw))
    bad = [r["id"] for r in rows if r["core"] and r["verdict"] == "FAIL"]
    assert bad == [failing]
    assert E.overall(rows) == "FAIL"


def test_gate_significance_flag_and_supplementary_do_not_decide():
    rows = E.gate_table(_results(gain=3.2, se=2.0))
    g1 = next(r for r in rows if r["id"] == "G1")
    assert g1["verdict"] == "PASS" and g1["sig"] is False        # passes the bar, under 2 SE
    res = _results()
    res["rotoguru"]["2019-20"]["lineup"]["paired"]["ours_mean_minus_salary"]["mean_diff"] = -9.0
    rows = E.gate_table(res)
    assert any(r["id"] == "G2b" and r["verdict"] == "FAIL" for r in rows)
    assert E.overall(rows) == "PASS"


def test_gate_table_missing_parts_are_skipped_not_passed():
    assert E.overall(E.gate_table({"projection": {}, "rotoguru": {}})) == "NO RESULTS"
    res = _results()
    res["projection"]["2024-25"]["minutes"]["mae"] = None
    assert E.overall(E.gate_table(res)) == "INCOMPLETE"


# --------------------------------------------------------------------------- lineup scoring

def test_lineup_actual_uses_given_points_and_zero_for_missing():
    assert E.lineup_actual(["a", "b", "z"], {"a": 10.0, "b": 2.5, "c": 99.0}) == 12.5
    assert E.lineup_actual(["a"], {"a": None}) == 0.0


def _slate(day="2019-11-05"):
    """Four teams in two games; 10 players per team. 'hero' players are cheap, projected ~0 but
    score 50 actual points; everybody else scores 1.0 actual and is projected at 5/$1k."""
    rows, proj = [], {}
    poss = ["PG", "SG", "SF", "PF", "C"] * 2
    games = [("AAA", "BBB"), ("CCC", "DDD")]
    k = 0
    for home, away in games:
        for team, opp, is_home in ((home, away, 1), (away, home, 0)):
            for i, pos in enumerate(poss):
                k += 1
                hero = i < 5
                sal = 3000 if hero else 4000 + 250 * ((k * 7) % 8)
                rows.append({"game_date": day, "rg_id": f"r{k}", "player_id": 1000 + k,
                             "name": f"P{k}", "pos": pos, "team": team, "opp": opp,
                             "home": is_home, "starter": 1, "salary": sal,
                             "dk_pts": 50.0 if hero else 1.0, "minutes": 30.0, "on_slate": 1})
                proj[f"r{k}"] = {"dk_mean": 0.1 if hero else 5.0 * sal / 1000,
                                 "dk_q25": 0.0 if hero else 4.0 * sal / 1000, "min_mean": 30.0}
    return rows, proj


def test_lineups_are_scored_with_actual_points_not_projections(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rows, proj = _slate()
    db.insert_rows(conn, "dfs_hist_salaries", rows)
    pool_rows = pd.DataFrame(rows).rename(columns={"dk_pts": "pts"})
    lines = {"all_seasons": (0.0, 0.005)}
    std_map = {}
    out = E._lineups_for_day(conn, "2019-11-05", pool_rows, proj, lines, std_map, CLASSIC)
    # our lineup is built from the (wrong) projections: eight non-hero players, 1.0 actual each
    assert out["ours_mean"] == pytest.approx(8.0)
    assert out["ours_mean_proj"] > 100          # what the projection thought it was worth
    assert out["ours_cash"] == pytest.approx(8.0)
    assert out["salary"] == pytest.approx(8.0)
    assert out["salary_avail"] == pytest.approx(8.0)
    # hindsight is the best possible actual score: eight heroes at 50
    assert out["hindsight"] == pytest.approx(400.0)
    assert out["hindsight"] > out["ours_mean"]
    # scoring against the same pool: equals the sum of the actuals of the chosen ids
    assert np.isnan(out["std_avail"])           # no season-to-date numbers were supplied


def test_evaluate_lineups_paired_and_ratio():
    ld = pd.DataFrame({
        "date": ["d1", "d2", "d3", "d4"], "n_pool": [100] * 4,
        "ours_mean": [300.0, 320.0, 310.0, 330.0], "ours_cash": [290.0, 300.0, 305.0, 310.0],
        "salary": [280.0, 300.0, 330.0, 300.0], "salary_avail": [285.0, 300.0, 300.0, 305.0],
        "std_avail": [290.0, 310.0, 300.0, 320.0], "hindsight": [400.0, 400.0, 400.0, np.nan]})
    r = E.evaluate_lineups(ld)
    assert r["n_dates"] == 3 and r["n_dates_dropped"] == 1        # failed date dropped for all
    assert r["methods"]["ours_mean"]["mean_actual"] == pytest.approx(310.0)
    assert r["methods"]["ours_mean"]["ratio_to_hindsight"] == pytest.approx(930 / 1200)
    p = r["paired"]["ours_mean_minus_salary"]
    diff = np.array([20.0, 20.0, -20.0])
    assert p["mean_diff"] == pytest.approx(diff.mean())
    assert p["se"] == pytest.approx(diff.std(ddof=1) / np.sqrt(3))
    assert p["win_rate"] == pytest.approx(2 / 3)


def test_evaluate_frame_clusters_by_date_and_applies_populations():
    rng = np.random.default_rng(3)
    n = 600
    day = np.repeat([f"2025-01-{d:02d}" for d in range(1, 21)], n // 20)
    y = rng.gamma(4, 5, n)
    played = y > 2
    mean = y * 0.0 + 20.0 + rng.normal(0, 6, n)
    df = pd.DataFrame({
        "game_id": [f"g{i // 10}" for i in range(n)], "player_id": np.arange(n),
        "game_date": day, "season": "2024-25", "team": "X", "tier": rng.integers(0, 3, n),
        "p_play": 1.0, "y": y, "played": played, "act_min": np.where(played, 25.0, 0.0),
        "mean": mean, "q10": mean - 12, "q25": mean - 6, "q50": mean, "q75": mean + 6,
        "q90": mean + 12, "crps": 4.0, "exp_min": 25.0, "min_played": 25.0,
        "pmean": mean, "pq10": mean - 12, "pq25": mean - 6, "pq75": mean + 6, "pq90": mean + 12,
        "pcrps": 4.0, "base_std": 20.0 + rng.normal(0, 9, n), "base_ewma": 20.0, "base_min": 25.0})
    df.loc[:20, "exp_min"] = 5.0                 # excluded from both populations
    df.loc[:20, "base_std"] = np.nan
    out = E.evaluate_frame(df)
    assert out["fp"]["n"] == int(((df["exp_min"] >= 10) & df["base_std"].notna()).sum())
    assert out["fp"]["n_dates"] == 20
    u = df[(df["exp_min"] >= 10) & df["base_std"].notna()]
    want = 100 * (1 - np.abs(u["mean"] - u["y"]).mean() / np.abs(u["base_std"] - u["y"]).mean())
    assert out["fp"]["gain_vs_std_pct"] == pytest.approx(want)
    m = df[df["played"] & (df["exp_min"] >= 15)]
    assert out["minutes"]["n"] == len(m)
    assert out["minutes"]["mae"] == pytest.approx(np.abs(m["exp_min"] - m["act_min"]).mean())
    assert 0 <= out["fp"]["cov80"] <= 1


def test_usable_dates_drops_neutral_site_dates(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    rows = []
    for gid, day, a, b, ha, hb in (("g1", "2024-12-13", "AAA", "BBB", 1, 0),
                                   ("g2", "2024-12-14", "CCC", "DDD", 0, 0),      # Cup semi
                                   ("g3", "2024-12-15", "AAA", "CCC", 1, 0),
                                   ("g4", "2024-12-15", "BBB", "DDD", 0, 0)):     # neutral
        for team, h in ((a, ha), (b, hb)):
            rows.append({"game_id": gid, "team": team, "season": "2024-25",
                         "season_type": "Regular Season", "game_date": day, "is_home": h})
    db.insert_rows(conn, "team_games", rows)
    days, excluded = E.usable_dates(conn, "2024-25")
    assert [d.isoformat() for d in days] == ["2024-12-13", "2024-12-15"]
    assert {e["date"] for e in excluded} == {"2024-12-14", "2024-12-15"}
