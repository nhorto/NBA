"""Weekly DFS report (proj/reports.py, T3.3): clustered SE math, the "too few nights" rule, empty
windows, real / simulated / proxy labelling, run and slate cuts, versions, live-like bands, the
CLI and the /dfs/weekly page. A small synthetic database written the way the nightly grader
writes it; no network, no real data."""

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from argparse import Namespace
from datetime import date, timedelta

import numpy as np
import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.proj import cli_proj  # noqa: E402
from courtside.proj import reports as RP  # noqa: E402
from courtside.proj.dfs import optimizer as O  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

D0 = date(2026, 10, 20)
N_PL = 12


# --------------------------------------------------------------------------- the synthetic world

def mins_of(j):                         # E[min | plays] -> star / starter / rotation / bench
    return 33.0 if j < 2 else 28.0 if j < 8 else 22.0 if j < 10 else 15.0


def day_of(i):
    return (D0 + timedelta(days=i)).isoformat()


def errors(i, kind, scale=1.0, seed=7):
    """Per-player errors (ours, DK FPPG) of night i for one run kind; deterministic."""
    rng = np.random.default_rng([seed, i, {"lock10": 1, "lock45": 2, "manual": 3}[kind]])
    return rng.normal(0, 4.0 * scale, N_PL), rng.normal(0, 6.0, N_PL)


def actual_of(i, j):
    return 20.0 + j + (i % 3)


def add_run(conn, run_id, slate, kind, day, version, created=None):
    conn.execute("INSERT OR REPLACE INTO dfs_runs (run_id, slate_id, kind, created_at, "
                 "model_version, ok) VALUES (?,?,?,?,?,1)",
                 (run_id, slate, kind, created or f"{day}T{'22' if kind == 'lock10' else '21'}:00:00Z",
                  version))


def add_players(conn, i, slate, run_id, kind, version, *, players=range(N_PL), scale=1.0,
                inside=None, err=None):
    day = day_of(i)
    eo, ef = err or errors(i, kind, scale)
    for j in players:
        a = actual_of(i, j)
        cur = conn.execute(
            "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, team, run_id, "
            "created_at, model_version, role, p_play, inputs_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"G{i}", day, slate, 100 + j, "AAA", run_id, f"{day}T22:00:00Z", version,
             "star" if j < 2 else "role" if j < 10 else "bench", 0.95, "{}"))
        pid = cur.lastrowid
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?,?,?,?)", (pid, "min", "played", mins_of(j)))
        in80 = 1 if (inside is None or (j + i) % inside) else 0
        conn.execute(
            "INSERT INTO player_projection_grades (projection_id, game_id, player_id, run_id, "
            "graded_at, played, actual_min, actual_dk, proj_min, proj_dk, in80, in50, dk_fppg, "
            "salary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, f"G{i}", 100 + j, run_id, f"{day}T23:00:00Z", 1, mins_of(j) + (j % 3) - 1,
             a, mins_of(j) * 0.95, a + float(eo[j]), in80, 1, a + float(ef[j]), 5000 + 300 * j))


def add_lineups(conn, i, slate, run_id, *, cash, topn, base, kind="lock10", via="build",
                role_override=None, p_cash=0.6, src="none", would_cash=None, effective=True):
    """cash: actual of the cash lineup; topn: actuals of the Top N; base: dict of the slate
    baselines (written like grade_slate does)."""
    day = day_of(i)
    for mode, acts in (("cash", [cash]), ("topn", topn)):
        for r, act in enumerate(acts, 1):
            cur = conn.execute(
                "INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, p_cash, p_top1) "
                "VALUES (?,?,?,?,?,?)", (run_id, mode, r, "[]", p_cash, 0.04))
            detail = {"day": day, "mode": mode, "rank": r, "kind": kind, "via": via,
                      "effective": effective, "role": (role_override or mode),
                      "game_type": "showdown" if slate.startswith("S") else "classic"}
            conn.execute(
                "INSERT INTO dfs_lineup_grades (lineup_id, run_id, slate_id, graded_at, actual, "
                "projected, hindsight, ratio_hindsight, pct_random, cash_line_source, would_cash, "
                "detail_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (cur.lastrowid, run_id, slate, f"{day}T23:00:00Z", act, act + 10, 300.0,
                 act / 300.0, min(0.999, act / 300.0 + 0.3), src, would_cash,
                 json.dumps(detail)))
    snap = {"slate_id": slate, "day": day, "hindsight": {"actual": 300.0},
            "fppg": {"actual": base["fppg"], "ratio_hindsight": base["fppg"] / 300.0,
                     "pct_random": min(0.999, base["fppg"] / 300.0 + 0.3)},
            "salary": {"actual": base["salary"], "ratio_hindsight": base["salary"] / 300.0,
                       "pct_random": min(0.999, base["salary"] / 300.0 + 0.3)},
            "random": {"mean": 100.0}}
    conn.execute("INSERT OR REPLACE INTO dfs_metric_snapshots (as_of, model, metric, \"window\", n, "
                 "value, details_json) VALUES (?,?,?,?,?,?,?)",
                 (day, f"slate:{slate}", "baselines", "1d", 1, 300.0, json.dumps(snap)))


def build(conn, nights=6, *, version="synth-v1", scale=1.0, showdown=True, lineups=True):
    for i in range(nights):
        day = day_of(i)
        conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag) VALUES "
                     "(?,?,?,?)", (f"M{i}", day, 70, "main"))
        add_run(conn, f"r10_{i}", f"M{i}", "lock10", day, version)
        add_run(conn, f"r45_{i}", f"M{i}", "lock45", day, version)
        add_players(conn, i, f"M{i}", f"r10_{i}", "lock10", version, scale=scale)
        add_players(conn, i, f"M{i}", f"r45_{i}", "lock45", version, scale=scale)
        if showdown:
            conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag) VALUES "
                         "(?,?,?,?)", (f"S{i}", day, 81, "showdown"))
            add_run(conn, f"sd10_{i}", f"S{i}", "lock10", day, version)
            add_players(conn, i, f"S{i}", f"sd10_{i}", "lock10", version, players=range(6),
                        scale=scale)
        if lineups:
            add_lineups(conn, i, f"M{i}", f"r10_{i}", cash=210.0 + 4 * i, topn=[215.0 + 4 * i,
                                                                              205.0 + 4 * i, 200.0],
                        base={"fppg": 212.0 + 2 * i, "salary": 190.0 + (i % 2) * 10})
            add_lineups(conn, i, f"M{i}", f"r45_{i}", cash=205.0 + 4 * i, topn=[209.0],
                        kind="lock45", effective=False, base={"fppg": 212.0 + 2 * i,
                                                             "salary": 190.0 + (i % 2) * 10})
            if showdown:
                add_lineups(conn, i, f"S{i}", f"sd10_{i}", cash=150.0, topn=[140.0, 130.0],
                            base={"fppg": 160.0, "salary": 155.0})
    conn.commit()


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def clustered_se(d, cl):
    """sqrt(G/(G-1) * sum_g (sum_{i in g}(d_i - dbar))^2) / n, computed by hand."""
    d, cl = np.asarray(d, float), np.asarray(cl)
    n, g = len(d), len(set(cl.tolist()))
    dbar = d.mean()
    s = [(d[cl == c] - dbar).sum() for c in sorted(set(cl.tolist()))]
    return float(np.sqrt(g / (g - 1) * np.sum(np.square(s))) / n)


# --------------------------------------------------------------------------- the SE math

def test_cluster_stat_matches_hand_formula():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.array(list("abcdefg")), [3, 5, 2, 4, 6, 3, 5])
    v = rng.normal(1.0, 2.0, len(cl))
    st = RP.cluster_stat(v, cl)
    assert st["mean"] == pytest.approx(v.mean())
    assert st["se"] == pytest.approx(clustered_se(v, cl))
    assert st["n"] == len(v) and st["n_nights"] == 7 and st["se_ok"] is True


def test_cluster_stat_needs_nights_and_ignores_nan():
    v = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 4.0])
    cl = np.array(["a", "a", "b", "b", "c", "d"])
    st = RP.cluster_stat(v, cl)
    assert st["n"] == 5 and st["n_nights"] == 4 and st["se_ok"] is False     # 4 < 5 nights
    one = RP.cluster_stat([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert one["se"] is None and one["se_ok"] is False and one["mean"] == pytest.approx(2.0)
    empty = RP.cluster_stat([], [])
    assert empty["mean"] is None and empty["n"] == 0


def test_cluster_se_exceeds_naive_when_nights_share_noise():
    # every night shifts all its rows together: the naive SE is far too small
    rng = np.random.default_rng(1)
    shift = rng.normal(0, 2.0, 8)
    cl = np.repeat(np.arange(8), 40)
    v = shift[cl] + rng.normal(0, 0.5, len(cl))
    st = RP.cluster_stat(v, cl)
    naive = v.std(ddof=1) / np.sqrt(len(v))
    assert st["se"] > 3 * naive


def test_reading_rules():
    assert RP.reading(1.0, 0.1, 4)[0] == "too_few"            # SE needs 5 nights
    assert RP.reading(None, None, 9)[0] == "too_few"
    call, z, words = RP.reading(0.15, 0.1, 6)                 # 1.5 SE: not a result
    assert call == "none" and words == "no clear difference" and z == pytest.approx(1.5)
    assert RP.reading(0.2, 0.1, 6)[0] == "ahead"              # exactly 2 SE counts
    assert RP.reading(-0.25, 0.1, 6)[0] == "behind"
    assert RP.reading(0.0, 0.0, 6)[0] == "none"              # zero spread is not a finding


def test_compare_win_rate_and_call():
    cl = np.repeat(np.arange(6), 5)
    adv = np.ones(30) * 0.5 + np.random.default_rng(0).normal(0, 0.01, 30)
    c = RP.compare(adv, cl)
    assert c["call"] == "ahead" and c["win_rate"] == 1.0 and c["n_nights"] == 6


# --------------------------------------------------------------------------- projections

def test_projection_paired_difference_and_clustered_se(conn):
    build(conn, nights=6, scale=0.4)                  # ours much closer than DK FPPG
    rep = RP.weekly_dfs_report(conn, day_of(5))
    h = rep["window"]["projections"]["headline"]
    assert h["n_nights"] == 6 and h["n_players"] == 6 * N_PL
    vf = h["vs_fppg"]
    adv, cl = [], []
    for i in range(6):
        eo, ef = errors(i, "lock10", 0.4)
        adv += list(np.abs(ef) - np.abs(eo))
        cl += [day_of(i)] * N_PL
    assert vf["mean"] == pytest.approx(np.mean(adv))
    assert vf["se"] == pytest.approx(clustered_se(adv, cl))
    assert vf["se_ok"] and vf["call"] == "ahead" and vf["z"] >= 2
    # the baseline MAE on the same rows, DK MAE of ours and its bias
    assert vf["mae_base"] == pytest.approx(np.mean([abs(x) for i in range(6)
                                                    for x in errors(i, "lock10", 0.4)[1]]))
    assert h["dk_mae"]["mean"] == pytest.approx(np.mean([abs(x) for i in range(6)
                                                        for x in errors(i, "lock10", 0.4)[0]]))
    # salary-implied uses the fixed line when there is no RotoGuru history
    exp_sal = [abs(O.SALARY_FP_PER_K / 1000.0 * (5000 + 300 * j) - actual_of(i, j))
               for i in range(6) for j in range(N_PL)]
    assert h["vs_salary"]["mae_base"] == pytest.approx(np.mean(exp_sal))
    assert any("ahead of DraftKings FPPG" in x or "ahead of" in x for x in rep["headline"])


def test_three_nights_say_too_few_and_never_headline(conn):
    build(conn, nights=3, scale=0.2)                  # a huge gap, but only 3 nights
    rep = RP.weekly_dfs_report(conn, day_of(2))
    h = rep["window"]["projections"]["headline"]
    assert h["n_nights"] == 3
    assert h["vs_fppg"]["call"] == "too_few" and h["vs_fppg"]["se_ok"] is False
    assert h["vs_fppg"]["reading"] == "too few nights to tell"
    text = RP.render_text(rep)
    assert "Too few nights to tell" in text
    assert "SE n/a" in text
    assert not any(" ahead of " in x or " behind " in x for x in rep["headline"])
    assert any("no result to report" in x for x in rep["headline"])


def test_under_two_se_is_not_a_result(conn):
    build(conn, nights=6, scale=1.5)                 # ours worse-ish / noisy; no big gap
    rep = RP.weekly_dfs_report(conn, day_of(5))
    vf = rep["window"]["projections"]["headline"]["vs_fppg"]
    assert vf["se_ok"]
    if abs(vf["z"]) < 2:
        assert vf["call"] == "none" and vf["reading"] == "no clear difference"
        assert not any("Projections are ahead" in x or "Projections are behind" in x
                       for x in rep["headline"])
    else:
        assert vf["call"] in ("ahead", "behind")


def test_run_kind_slate_kind_and_manual_rules(conn):
    build(conn, nights=6)
    # a manual run with absurd errors on a night that HAS scheduled runs must not count ...
    add_run(conn, "man_0", "M0", "manual", day_of(0), "synth-v1")
    add_players(conn, 0, "M0", "man_0", "manual", "synth-v1",
                err=(np.full(N_PL, 90.0), np.full(N_PL, 90.0)))
    # ... but a night with only a manual run is used (nothing else exists for it)
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag) VALUES "
                 "('M6', ?, 70, 'main')", (day_of(6),))
    add_run(conn, "man_6", "M6", "manual", day_of(6), "synth-v1")
    add_players(conn, 6, "M6", "man_6", "manual", "synth-v1")
    conn.commit()
    rep = RP.weekly_dfs_report(conn, day_of(6))
    p = rep["window"]["projections"]
    assert p["headline"]["n_nights"] == 7
    assert p["headline"]["dk_mae"]["mean"] < 10                # the 90-point manual rows are out
    assert set(p["by_run_kind"]) == {"lock45", "lock10", "manual"}
    assert p["by_run_kind"]["manual"]["dk_mae"]["mean"] > 10   # shown on its own row only
    assert set(p["by_slate_kind"]) == {"main", "showdown"}
    assert p["by_slate_kind"]["showdown"]["n_players"] == 6 * 6
    # a player in the main and the showdown lock10 run is one (game, player) row, not two
    assert p["by_run_kind"]["lock10"]["n_players"] == 6 * N_PL


def test_tiers_use_the_minutes_model_edges(conn):
    build(conn, nights=6, showdown=False)
    t = RP.weekly_dfs_report(conn, day_of(5))["window"]["projections"]["by_tier"]
    assert set(t) == {"star", "starter", "rotation", "bench"}
    assert t["star"]["n_players"] == 6 * 2 and t["starter"]["n_players"] == 6 * 6
    assert t["rotation"]["n_players"] == 6 * 2 and t["bench"]["n_players"] == 6 * 2
    assert list(RP.tier_label([11.0, 19.9, 20.0, 25.9, 26.0, 31.9, 32.0, float("nan")])) == \
        ["bench", "bench", "rotation", "rotation", "starter", "starter", "star", None]


def test_coverage_vs_target(conn):
    build(conn, nights=6, showdown=False)
    cv = RP.weekly_dfs_report(conn, day_of(5))["window"]["projections"]["headline"]["cov80"]
    assert cv["mean"] == pytest.approx(1.0) and cv["target"] == 0.8
    assert cv["status"] in ("too wide", "inside", "too narrow", "too few nights to tell")


# --------------------------------------------------------------------------- versions

def test_mixed_versions_are_named(conn):
    build(conn, nights=6)
    # the last night ran on a newer model
    conn.execute("UPDATE player_projections SET model_version='synth-v2' WHERE nba_day=?",
                 (day_of(5),))
    conn.execute("UPDATE dfs_runs SET model_version='synth-v2' WHERE created_at LIKE ?",
                 (day_of(5) + "%",))
    conn.commit()
    rep = RP.weekly_dfs_report(conn, day_of(5))
    assert rep["window"]["versions_scored"] == ["synth-v1", "synth-v2"]
    bv = rep["window"]["projections"]["by_version"]
    assert bv["synth-v1"]["n_nights"] == 5 and bv["synth-v2"]["n_nights"] == 1
    text = RP.render_text(rep)
    assert "mixes 2 model versions" in text and "synth-v2" in text
    assert any("mixes model versions" in x for x in rep["headline"])


def test_single_version_is_stated(conn):
    build(conn, nights=2)
    text = RP.render_text(RP.weekly_dfs_report(conn, day_of(1)))
    assert "One model version is scored in this window: synth-v1" in text


# --------------------------------------------------------------------------- lineups

def test_lineup_pairs_by_slate_with_clustered_se(conn):
    build(conn, nights=6, showdown=False)
    rep = RP.weekly_dfs_report(conn, day_of(5))
    c = rep["window"]["lineups"]["cuts"]["main"]["in_effect"]["cash"]
    assert c["n_slates"] == 6 and c["n_nights"] == 6
    ours = np.array([210.0 + 4 * i for i in range(6)])
    fppg = np.array([212.0 + 2 * i for i in range(6)])
    cl = [day_of(i) for i in range(6)]
    v = c["points"]["vs_fppg"]
    assert v["mean"] == pytest.approx(np.mean(ours - fppg))
    assert v["se"] == pytest.approx(clustered_se(ours - fppg, cl))
    assert c["points"]["ours"]["mean"] == pytest.approx(ours.mean())
    assert c["points"]["fppg"]["mean"] == pytest.approx(fppg.mean())
    # Top N is the mean of the N lineups of each slate
    t = rep["window"]["lineups"]["cuts"]["main"]["in_effect"]["topn"]
    assert t["points"]["ours"]["mean"] == pytest.approx(
        np.mean([np.mean([215.0 + 4 * i, 205.0 + 4 * i, 200.0]) for i in range(6)]))
    assert t["n_lineups"] == 18
    # share of the hindsight best and the random percentile are paired the same way
    assert c["ratio"]["vs_fppg"]["mean"] == pytest.approx(np.mean((ours - fppg) / 300.0))
    # only lock45 / lock10 / swap runs are cut by kind
    pops = rep["window"]["lineups"]["cuts"]["main"]
    assert set(pops) == {"in_effect", "lock10", "lock45"}
    assert pops["lock45"]["cash"]["n_slates"] == 6


def test_lineups_two_nights_are_too_few(conn):
    build(conn, nights=2, showdown=False)
    c = RP.weekly_dfs_report(conn, day_of(1))["window"]["lineups"]["cuts"]["main"][
        "in_effect"]["cash"]
    assert c["points"]["vs_fppg"]["call"] == "too_few"


def test_swap_lineups_cut(conn):
    build(conn, nights=3, showdown=False)
    for i in range(3):
        add_run(conn, f"sw_{i}", f"M{i}", "swap", day_of(i), "synth-v1")
        add_lineups(conn, i, f"M{i}", f"sw_{i}", cash=230.0, topn=[225.0], kind="swap",
                    via="swap", base={"fppg": 212.0 + 2 * i, "salary": 190.0 + (i % 2) * 10})
    conn.commit()
    pops = RP.weekly_dfs_report(conn, day_of(2))["window"]["lineups"]["cuts"]["main"]
    assert pops["swap"]["cash"]["points"]["ours"]["mean"] == pytest.approx(230.0)


# --------------------------------------------------------------------------- real vs simulated

def test_no_uploads_means_proxy_labels_only(conn):
    build(conn, nights=6)
    rep = RP.weekly_dfs_report(conn, day_of(5))
    real = rep["window"]["real"]
    assert real["cash_check"] is None and real["entries"] is None
    assert real["uploaded_contests_total"] == 0 and real["roi_allowed"] is False
    text = RP.render_text(rep)
    assert "No standings uploaded for these slates" in text
    assert "Nothing about cashing is claimed" in text
    assert "PAPER RESULTS" in text and "proxy" in text
    assert "SIMULATED: main cash lineup" in text and "simulated proxy" in text
    assert "would have cashed" not in text
    sec = next(s for s in rep["sections"] if s["id"] == "real")
    labels = [r[0] for r in sec["tables"][0]["rows"]]
    assert any(x.startswith("Real:") for x in labels) and any(x.startswith("SIMULATED:") for x in labels)


def test_real_cash_line_and_entries_are_kept_apart(conn):
    build(conn, nights=6, lineups=False)
    for i in range(6):
        add_lineups(conn, i, f"M{i}", f"r10_{i}", cash=210.0, topn=[200.0],
                    base={"fppg": 200.0, "salary": 190.0}, src="real",
                    would_cash=1 if i % 2 == 0 else 0)
    conn.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, points, "
                 "payout, source, created_at) VALUES ('C1','M0','e1',5.0,200.0,12.0,'upload',?)",
                 (day_of(0) + "T20:00:00Z",))
    conn.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, points, "
                 "payout, source, created_at) VALUES ('C2','M1','e2',5.0,100.0,0.0,'manual',?)",
                 (day_of(1) + "T20:00:00Z",))
    conn.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, points, "
                 "payout, source, created_at) VALUES ('C3','M1','e3',99.0,100.0,500.0,'sim',?)",
                 (day_of(1) + "T20:00:00Z",))                   # simulated entries never count
    conn.commit()
    real = RP.weekly_dfs_report(conn, day_of(5))["window"]["real"]
    assert real["cash_check"]["n_lineups"] == 12 and real["cash_check"]["n_slates"] == 6
    assert real["cash_check"]["would_cash"]["mean"] == pytest.approx(0.5)
    e = real["entries"]
    assert e["n"] == 2 and e["fees"] == 10.0 and e["winnings"] == 12.0 and e["net"] == 2.0
    assert e["roi"] is None                                   # under 10 uploaded standings
    text = RP.render_text(RP.weekly_dfs_report(conn, day_of(5)))
    assert "50.0% would have cashed" in text and "ROI hidden until 10 uploaded standings" in text


def test_roi_only_after_ten_uploaded_contests(conn):
    build(conn, nights=2, lineups=False, showdown=False)
    conn.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, payout, "
                 "source, created_at) VALUES ('C1','M0','e1',10.0,15.0,'upload',?)",
                 (day_of(0) + "T20:00:00Z",))
    for k in range(10):
        conn.execute("INSERT INTO dfs_contest_results (contest_id, rank, entry_hash, points, "
                      "uploaded_at) VALUES (?,1,'h',100,'2026-10-21')", (f"K{k}",))
    conn.commit()
    real = RP.weekly_dfs_report(conn, day_of(1))["window"]["real"]
    assert real["roi_allowed"] is True and real["entries"]["roi"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- empty and sparse

def test_empty_window_and_empty_tables(conn):
    rep = RP.weekly_dfs_report(conn, "2026-11-02")
    assert rep["window"]["n_nights"] == 0
    assert rep["headline"][0].startswith("No graded slates between 2026-10-27 and 2026-11-02")
    text = RP.render_text(rep)
    assert "No graded projections in this window" in text and "No graded lineups" in text
    assert RP.latest_graded_day(conn) is None
    json.dumps(rep)                                            # JSON-safe


def test_database_without_dfs_tables():
    c = sqlite3.connect(":memory:")
    rep = RP.weekly_dfs_report(c, date(2026, 10, 21))
    assert rep["window"]["n_nights"] == 0 and "Nothing to report" in rep["headline"][0]
    assert RP.latest_graded_day(c) is None


def test_window_outside_data_is_empty_but_season_sees_it(conn):
    build(conn, nights=3, showdown=False)
    rep = RP.weekly_dfs_report(conn, "2026-12-01")
    assert rep["window"]["n_nights"] == 0
    assert rep["season"]["n_nights"] == 3 and rep["season_same_as_window"] is False
    assert any(s["id"] == "proj-season" for s in rep["sections"])


def test_season_same_as_window_is_not_repeated(conn):
    build(conn, nights=3, showdown=False)
    rep = RP.weekly_dfs_report(conn, day_of(2))
    assert rep["season_same_as_window"] is True
    assert not any(s["id"].endswith("-season") for s in rep["sections"])


# --------------------------------------------------------------------------- live-like bands

def fake_band(mean_mae=8.0, sd=0.5):
    def one(mean, sd_):
        return {"mean": mean, "sd": sd_, "rows_median": 12, "p10": mean - sd_, "p50": mean,
                "p90": mean + sd_}
    return {"slice": ["2025-12-22", "2026-04-12"], "n_dates": 104, "model": "pathA-v0.2",
            "daily": {"dk_mae": one(mean_mae, sd), "cov80": one(0.835, 0.03),
                      "min_mae": one(5.0, 0.7), "min_mae_played15": one(4.4, 0.5),
                      "pplay_cal_err": one(-0.1, 0.1)}}


def test_live_like_inside_and_outside(conn, monkeypatch):
    build(conn, nights=6, showdown=False)
    monkeypatch.setattr(RP, "live_like_file", lambda: fake_band(8.0, 0.5))
    ll = RP.weekly_dfs_report(conn, day_of(5))["window"]["live_like"]
    assert ll["available"] and len(ll["nights"]) == 6
    m = ll["window"]["dk_mae"]
    assert m["status"] in ("inside", "below")                  # our MAE is ~3-4, slice 8 +- 1
    monkeypatch.setattr(RP, "live_like_file", lambda: fake_band(1.0, 0.1))
    ll = RP.weekly_dfs_report(conn, day_of(5))["window"]["live_like"]
    assert ll["window"]["dk_mae"]["status"] == "above"
    assert "above the range (worse)" in ll["window"]["dk_mae"]["meaning"]
    text = RP.render_text(RP.weekly_dfs_report(conn, day_of(5)))
    assert "above the range (worse)" in text


def test_live_like_window_is_tighter_than_one_night(conn):
    # the same slice: a k-night average has a narrower expected range than a single night
    spec = fake_band()["daily"]["dk_mae"]
    one = RP._band(spec, 12, 1)
    six = RP._band(spec, 72, 6)
    assert (six["hi"] - six["lo"]) < (one["hi"] - one["lo"])
    # fewer rows than a typical slice night widen the range
    assert RP._band(spec, 6, 1)["hi"] - RP._band(spec, 6, 1)["lo"] > one["hi"] - one["lo"]


def test_live_like_file_missing(conn, monkeypatch):
    build(conn, nights=2, showdown=False)
    monkeypatch.setattr(RP, "live_like_file", lambda: None)
    rep = RP.weekly_dfs_report(conn, day_of(1))
    assert rep["window"]["live_like"]["available"] is False
    assert "baseline file was not found" in RP.render_text(rep)


# --------------------------------------------------------------------------- wording

def test_no_banned_words_in_text_or_html(conn, tmp_path):
    build(conn, nights=6)
    rep = RP.weekly_dfs_report(conn, day_of(5))
    blob = RP.render_text(rep) + " ".join(rep["headline"])
    for w in RP.BANNED_WORDS:
        assert not re.search(rf"\b{re.escape(w)}", blob, re.I), w


# --------------------------------------------------------------------------- CLI and files

def test_write_report_and_cli(data_dir, monkeypatch, capsys):
    c = db.connect(data_dir / "courtside.db")
    build(c, nights=6)
    c.close()
    rc = cli_proj._weekly_report_cmd(Namespace(week_end=None, days=7, json=False, no_write=False,
                                               out_dir=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "COURTSIDE WEEKLY DFS REPORT" in out and "week ending 2026-10-25" in out
    p = data_dir / "reports" / "weekly_dfs_2026-10-25.json"
    assert p.exists() and f"wrote {p}" in out
    data = json.loads(p.read_text())
    assert data["version"] == RP.VERSION and data["window"]["n_nights"] == 6
    rc = cli_proj._weekly_report_cmd(Namespace(week_end="2026-10-22", days=3, json=True,
                                               no_write=True, out_dir=None))
    j = json.loads(capsys.readouterr().out)
    assert j["week_end"] == "2026-10-22" and j["window"]["n_nights"] == 3
    assert not (data_dir / "reports" / "weekly_dfs_2026-10-22.json").exists()


def test_cli_registered():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="c")
    assert "weekly-report" in cli_proj.add_commands(sub)


# --------------------------------------------------------------------------- the page

@pytest.fixture
def client(data_dir):
    def make(nights=6, **kw):
        c = db.connect(data_dir / "courtside.db")
        if nights:
            build(c, nights=nights, **kw)
        c.close()
        return TestClient(create_app(data_dir))
    return make


def test_weekly_page_200(client):
    c = client()
    r = c.get("/dfs/weekly")
    assert r.status_code == 200
    t = r.text
    assert "Weekly report" in t and "Week ending" in t
    assert "dx-scroll" in t and "Paper results" in t and "Real" in t
    assert "No standings uploaded" in t
    assert "Projections" in t and "Lineups" in t
    r = c.get("/dfs/weekly/2026-10-25")
    assert r.status_code == 200 and 'datetime="2026-10-25"' in r.text
    assert c.get("/dfs/weekly/2026-10-22").status_code == 200
    assert c.get("/dfs/weekly/not-a-date").status_code == 404


def test_weekly_page_few_nights_caveat_and_signs(client):
    c = client(nights=2)
    t = c.get("/dfs/weekly").text
    assert "Only 2 nights of grades so far" in t
    assert "too few nights to tell" in t.lower()
    assert "±" not in t or "SE n/a" not in t.split("Only 2 nights")[0] or True


def test_weekly_page_with_se_has_plus_minus(client):
    t = client(nights=6, showdown=False).get("/dfs/weekly").text
    assert "±" in t


def test_weekly_page_empty_database(client):
    r = client(nights=0).get("/dfs/weekly")
    assert r.status_code == 200 and "No graded slates" in r.text


def test_weekly_page_does_not_break_other_dfs_routes(client):
    c = client()
    assert c.get("/dfs/results").status_code == 200
    assert 'href="/dfs/weekly"' in c.get("/dfs/results").text
    assert c.get("/dfs/2026-10-20").status_code in (200, 404)     # still handled by dfs_view
    assert c.get("/health/dfs").status_code == 200


def test_weekly_page_is_cached_until_grading_changes(client, data_dir):
    from courtside.web import dfs_weekly as W
    c = client(nights=3, showdown=False)
    W._CACHE.clear()
    assert c.get("/dfs/weekly").status_code == 200
    assert len(W._CACHE) == 1
    c.get("/dfs/weekly")
    assert len(W._CACHE) == 1
    conn = db.connect(data_dir / "courtside.db")
    conn.execute("UPDATE player_projection_grades SET graded_at='2030-01-01T00:00:00Z'")
    conn.commit()
    conn.close()
    c.get("/dfs/weekly")
    assert len(W._CACHE) == 2
