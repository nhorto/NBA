"""DFS backtest harness (proj/backtest.py) on a synthetic RotoGuru league: lineup methods, random
reference, caching, parity with eval-lite, the leakage rules L1-L6 of docs/PLAYERS.md §7.1 (each
as a test that fails if the rule is broken), and the leak probe.

The league is the one of the pipeline tests (4 teams, 4 seasons 2019-20..2022-23) with RotoGuru
rows added for 2019-20 and 2020-21; the walk-forward season is 2020-21 (a RotoGuru season)."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import config, db
from courtside.proj import backtest as B
from courtside.proj import copula as C
from courtside.proj import evallite as E
from courtside.proj import metrics as MT
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import state as ST
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R

from test_proj_minutes import ROLES
from test_proj_pipeline import _fill

FIX = Path(__file__).parent / "fixtures" / "dfs"
CLASSIC = R.Rules.from_payload(json.loads((FIX / "rules_70.json").read_text()))
SEASON = "2020-21"
REAL_LOAD_BUNDLE = P.load_bundle
SET = B.Settings(n_sims=300, n_topn=3, n_random=300, crps_sub=300)


# --------------------------------------------------------------------------- synthetic league

def _dk_points(pg: pd.DataFrame) -> pd.Series:
    cats = sum((pg[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    return (sum(C.DK_W[s] * pg[s].astype(float) for s in C.DK_STATS)
            + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))


def _add_rotoguru(conn: sqlite3.Connection) -> None:
    pg = pd.read_sql_query(
        "SELECT p.*, t.opp, t.is_home FROM player_games p JOIN team_games t "
        "ON p.game_id = t.game_id AND p.team = t.team", conn)
    st = {(g, p) for g, p in conn.execute("SELECT game_id, player_id FROM player_starts")}
    pg["dk"] = _dk_points(pg)
    rows = []
    for r in pg.itertuples(index=False):
        k = int(r.player_id) % 1000
        role, mins, _ = ROLES[k]
        pos = {"G": ("PG", "SG"), "F": ("SF", "PF"), "C": ("C", "C")}[role][k % 2]
        sal = int(round((2500 + 135 * mins) / 100) * 100)
        rows.append((r.game_date, f"rg{int(r.player_id)}", int(r.player_id), f"P{int(r.player_id)}",
                     pos, r.team, r.opp, int(r.is_home), int((r.game_id, r.player_id) in st),
                     sal, float(r.dk), float(r.min), None, r.game_id, 1, None))
    conn.executemany(
        "INSERT INTO dfs_hist_salaries (game_date, rg_id, player_id, name, pos, team, opp, home, "
        "starter, salary, dk_pts, minutes, raw_stats, game_id, on_slate, raw_path) "
        "VALUES (" + ",".join("?" * 16) + ")", [x for x in rows if x[0] < "2021-09-01"])
    conn.commit()


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    _add_rotoguru(c)
    return c


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    orig = O.hist_dates
    monkeypatch.setattr(O, "hist_dates", lambda conn, min_players=10: orig(conn, min_players=10))
    monkeypatch.setattr(O, "classic_rules", lambda game_type_id=70: CLASSIC)
    monkeypatch.setattr(
        P, "load_bundle",
        lambda conn, season, ctx=None, **k: P.Bundle(
            season=season, minutes=M.default_params(season), state=ST.default_params(),
            copula=C.default_copula_params(season)))


def _clone(conn) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    conn.backup(c)
    c.row_factory = sqlite3.Row
    return c


def _season_run(conn, settings=SET) -> B.SeasonRun:
    return B.SeasonRun(conn, SEASON, settings, log=lambda m: None)


@pytest.fixture(scope="module")
def days(conn):
    return sorted({r[0] for r in conn.execute(
        "SELECT game_date FROM dfs_hist_salaries WHERE game_date >= ? AND game_date < ?",
        (E.season_start(SEASON), "2021-09-01"))})


def _run_day(conn, day, settings=SET):
    sr = _season_run(conn, settings)
    return sr, sr.run_date(day)


def _sql_scramble(c: sqlite3.Connection, cut: str, *, strictly_after: bool = False) -> None:
    """Replace every outcome and input row on (or strictly after) `cut` with noise: box scores,
    team scores, closing lines of those games, RotoGuru salaries / points, injuries."""
    op = ">" if strictly_after else ">="
    with c:
        c.execute(f"UPDATE player_games SET min = 1 + abs(random() % 40), pts = abs(random() % 45),"
                  f" reb = abs(random() % 15), ast = abs(random() % 12), fg3m = abs(random() % 6),"
                  f" stl = abs(random() % 4), blk = abs(random() % 4), tov = abs(random() % 6)"
                  f" WHERE game_date {op} ?", (cut,))
        c.execute(f"UPDATE team_games SET pts = 80 + abs(random() % 50) WHERE game_date {op} ?",
                  (cut,))
        c.execute(f"UPDATE closing_lines SET home_spread_close = (abs(random() % 20) - 10), "
                  f"total_close = 190 + abs(random() % 50) WHERE game_id IN "
                  f"(SELECT game_id FROM team_games WHERE game_date {op} ?)", (cut,))
        c.execute(f"UPDATE dfs_hist_salaries SET salary = 3000 + (abs(random() % 90) * 100), "
                  f"dk_pts = abs(random() % 60), minutes = abs(random() % 40) "
                  f"WHERE game_date {op} ?", (cut,))
        c.execute(f"DELETE FROM player_starts WHERE game_id IN "
                  f"(SELECT game_id FROM team_games WHERE game_date {op} ?)", (cut,))
        c.execute(f"DELETE FROM injury_snapshots WHERE captured_at {op} ?", (cut,))


def _tables_read(conn, fn):
    seen: set = set()

    def auth(action, a1, a2, dbn, src):
        if action == sqlite3.SQLITE_READ and a1:
            seen.add(a1)
        return sqlite3.SQLITE_OK

    conn.set_authorizer(auth)
    try:
        out = fn()
    finally:
        conn.set_authorizer(None)
    return seen, out


def _same_projection(a: pd.DataFrame, b: pd.DataFrame, *, outcomes: bool = True) -> bool:
    """Projection columns equal (CRPS / PIT too unless the outcomes were changed on purpose)."""
    cols = ["rg_id", "mean", "q10", "q25", "q50", "q75", "q90", "exp_min"]
    cols += ["crps", "pit"] if outcomes else []
    return a[cols].equals(b[cols])


# --------------------------------------------------------------------------- random reference

def test_random_lineups_are_legal_seeded_and_spend_constrained(conn, days):
    pool = O.hist_pool(conn, days[8], CLASSIC)
    pl = pool.players
    rng = np.random.default_rng(3)
    idx = B.random_lineups(pl, CLASSIC, 400, rng)
    assert idx.shape == (400, CLASSIC.n_slots)
    names = CLASSIC.slot_names
    for row in idx[:100]:
        assert len(set(row)) == CLASSIC.n_slots                              # distinct players
        assert all(names[k] in pl[i].slots for k, i in enumerate(row))        # slot eligibility
        assert sum(pl[i].salary for i in row) <= CLASSIC.salary_cap
        assert len({pl[i].team for i in row}) >= CLASSIC.min_teams
        assert len({pl[i].game_id for i in row}) >= CLASSIC.min_games
    again = B.random_lineups(pl, CLASSIC, 400, np.random.default_rng(3))
    assert np.array_equal(idx, again)
    sp = B.random_lineups(pl, CLASSIC, 200, rng, weight_pow=2.0,
                          min_salary=B.SPEND_FRAC * CLASSIC.salary_cap)
    sal = np.array([[pl[i].salary for i in row] for row in sp]).sum(1)
    assert len(sp) > 50 and sal.min() >= B.SPEND_FRAC * CLASSIC.salary_cap
    assert sal.max() <= CLASSIC.salary_cap


def test_random_lineup_percentile_sanity(conn, days):
    day = days[8]
    pool = O.hist_pool(conn, day, CLASSIC)
    rows = conn.execute("SELECT rg_id, dk_pts FROM dfs_hist_salaries WHERE game_date=?", (day,))
    actual = {str(r[0]): float(r[1]) for r in rows}
    ref = B.RandomReference(pool, CLASSIC, actual, np.random.default_rng(1), 1500)
    best = O.optimize(pool, O.hist_projections(conn, pool, "actual"), CLASSIC)
    worst_proj = {k: -v for k, v in actual.items()}
    worst = O.optimize(pool, worst_proj, CLASSIC)
    a_best, a_worst = E.lineup_actual(best.player_ids, actual), E.lineup_actual(worst.player_ids,
                                                                              actual)
    assert ref.pct(a_best) == (1.0, 1.0)                    # hindsight beats every random lineup
    assert ref.pct(a_worst)[0] < 0.02                       # the anti-hindsight lineup loses
    # fresh random lineups from the same law sit around the median of the reference
    act = np.array([actual.get(p.dk_player_id, 0.0) for p in pool.players])
    fresh = act[B.random_lineups(pool.players, CLASSIC, 600, np.random.default_rng(99))].sum(1)
    pct = MT.percentile_ranks(fresh, ref.any)
    assert pct.mean() == pytest.approx(0.5, abs=0.05)
    # the spend-constrained reference is a harder bar than uniform random picks
    assert ref.spend.mean() > ref.any.mean()


# --------------------------------------------------------------------------- baselines

def test_box_baselines_match_evallite_and_hand_computation(conn, days):
    box = B.BoxBaselines(conn, SEASON)
    rg = E.load_rotoguru(conn)
    rows = rg[(rg["game_date"] >= E.season_start(SEASON))].iloc[::7]
    ref = E.std_baseline(conn, SEASON, rows)
    got = np.array([box.get(p, d)[0] for p, d in zip(rows["player_id"], rows["game_date"])])
    assert np.allclose(got, ref, equal_nan=True) and np.isfinite(got).sum() > 20
    # last10 / season_avg by hand for one player-date
    pid, day = int(rows["player_id"].iloc[-1]), str(rows["game_date"].iloc[-1])
    a = P._actuals(conn, SEASON)
    a = a[(a["player_id"] == pid) & (a["min"] > 0) & (a["game_date"] < day)].sort_values("game_date")
    fppg, savg, l10 = box.get(pid, day)
    assert l10 == pytest.approx(a["dk"].tail(10).mean())
    cur = a[a["game_date"] >= E.season_start(SEASON)]
    assert savg == pytest.approx(cur["dk"].mean())
    assert all(np.isnan(box.get(None, day)))
    assert all(np.isnan(box.get(-5, day)))


# --------------------------------------------------------------------------- one date

def test_run_date_methods_hindsight_and_topn(conn, days):
    day = days[10]
    sr, (players, lineups, meta) = _run_day(conn, day)
    assert meta["error"] is None and meta["n_pool"] > 20
    singles = lineups[lineups["rank"] == 1]
    methods = set(singles["method"])
    want = {"ours_mean", "ours_cash", "hindsight", "ours_topn", "fppg_proxy_avail_topn"}
    for b in B.BASELINES:
        want |= {b, f"{b}_avail"}
    assert methods == want
    one = lineups[lineups["kind"].isin(["single", "hindsight"])].set_index("method")["actual"]
    assert (one.drop("hindsight") <= one["hindsight"] + 1e-9).all()
    tn = lineups[lineups["method"] == "ours_topn"].sort_values("rank")
    assert list(tn["rank"]) == [1, 2, 3]
    assert (np.diff(tn["proj"]) <= 1e-9).all()                        # Top-N is non-increasing
    assert tn["ids"].nunique() == 3
    assert lineups["pct_rand"].between(0, 1).all() and lineups["pct_spend"].between(0, 1).all()
    assert one["hindsight"] == pytest.approx(lineups[lineups["method"] == "hindsight"]
                                             ["actual"].iloc[0])
    assert float(lineups[lineups["method"] == "hindsight"]["pct_rand"].iloc[0]) == 1.0
    # player level: the actual outcome is the RotoGuru DK points, PIT is a probability
    rg = conn.execute("SELECT rg_id, dk_pts FROM dfs_hist_salaries WHERE game_date=?", (day,))
    ydb = {str(r[0]): float(r[1]) for r in rg}
    assert all(players["y"].iloc[i] == pytest.approx(ydb[players["rg_id"].iloc[i]])
               for i in range(len(players)))
    pr = players[players["has_proj"]]
    assert pr["pit"].between(0, 1).all() and (pr["crps"] > 0).all()
    assert (pr["q10"] <= pr["q25"]).all() and (pr["q25"] <= pr["q75"]).all()
    assert players["sal_all"].notna().all() and players["b_fppg"].notna().any()


def test_matches_evallite_lineups_for_the_same_date(conn, days):
    """The harness reproduces eval-lite's per-date lineup scores (same projections, same pools,
    same optimiser calls) for the shared methods."""
    day = days[12]
    sr, (players, lineups, _) = _run_day(conn, day)
    rg = E.load_rotoguru(conn)
    lines = {m: E.fit_salary_line(rg, SEASON, m) for m in E.SALARY_MODES}
    ctx = P.Context(conn, SEASON)
    res = P.project_historical(conn, day, protocol="avail", n_sims=SET.n_sims, ctx=ctx,
                               bundle=P.load_bundle(conn, SEASON))
    pl = res.players.drop_duplicates("player_id").set_index("player_id")
    pool_rows = rg[rg["game_date"] == day]
    proj = {str(r.rg_id): pl.loc[int(r.player_id)] for r in pool_rows.itertuples(index=False)
            if int(r.player_id) in pl.index}
    season_rows = rg[(rg["game_date"] >= E.season_start(SEASON))]
    std = E.std_baseline(conn, SEASON, season_rows)
    std_map = {(d, int(p)): v for d, p, v in zip(season_rows["game_date"], season_rows["player_id"],
                                                 std) if np.isfinite(v)}
    old = E._lineups_for_day(conn, day, pool_rows, proj, lines, std_map, CLASSIC)
    new = lineups[lineups["kind"].isin(["single", "hindsight"])].set_index("method")["actual"]
    for ours, theirs in (("ours_mean", "ours_mean"), ("ours_cash", "ours_cash"),
                         ("salary", "salary"), ("salary_avail", "salary_avail"),
                         ("fppg_proxy_avail", "std_avail"), ("hindsight", "hindsight")):
        assert new[ours] == pytest.approx(old[theirs]), ours


# --------------------------------------------------------------------------- caching

def test_cache_is_incremental_idempotent_and_keyed_by_config(conn, monkeypatch):
    calls = []
    orig = B.SeasonRun.run_date

    def spy(self, day):
        calls.append(day)
        return orig(self, day)

    monkeypatch.setattr(B.SeasonRun, "run_date", spy)
    p1, l1, m1 = B.run_season(conn, SEASON, SET, n_dates=4, log=lambda m: None)
    first = list(calls)
    assert len(first) == 4 and m1["n_run"] == 4 and not m1["skipped"]
    p2, l2, m2 = B.run_season(conn, SEASON, SET, n_dates=4, log=lambda m: None)
    assert calls == first                                       # nothing recomputed
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(l1, l2)
    # a superset of dates only computes the new ones
    B.run_season(conn, SEASON, SET, n_dates=6, log=lambda m: None)
    assert len(calls) > 4 and len(set(calls)) == len(calls)
    n = len(calls)
    # a different setting uses a different cache directory
    other = B.Settings(**{**B.asdict(SET), "n_topn": 2, "modes": ("cash", "topn"),
                          "baselines": B.BASELINES, "topn_baselines": ("fppg_proxy",)})
    B.run_season(conn, SEASON, other, n_dates=2, log=lambda m: None)
    assert len(calls) == n + 2
    assert B.SeasonRun(conn, SEASON, SET, need_context=False).cache().root != \
        B.SeasonRun(conn, SEASON, other, need_context=False).cache().root
    # force recomputes
    B.run_season(conn, SEASON, SET, n_dates=4, force=True, log=lambda m: None)
    assert len(calls) == n + 2 + 4


def test_skipped_date_is_recorded_not_fatal(conn, monkeypatch):
    orig = B.SeasonRun.run_date

    def boom(self, day):
        if day.endswith("-10") or day.endswith("-11"):
            raise RuntimeError("boom")
        return orig(self, day)

    monkeypatch.setattr(B.SeasonRun, "run_date", boom)
    p, l, m = B.run_season(conn, SEASON, SET, n_dates=0, log=lambda m: None)
    assert m["skipped"] and all("boom" in s["error"] for s in m["skipped"])
    assert m["n_run"] == m["n_dates"] - len(m["skipped"]) > 0
    assert not set(p["date"]) & {s["date"] for s in m["skipped"]}


# --------------------------------------------------------------------------- L1 no lookahead

def test_l1_scrambling_rows_on_or_after_a_cut_leaves_earlier_dates_bit_identical(conn, days):
    cut = days[13]
    d_before = days[12]
    sr, (p0, l0, _) = _run_day(conn, d_before)
    c2 = _clone(conn)
    _sql_scramble(c2, cut)                       # box, PBP-like, salary, injury, closing >= cut
    sr2, (p1, l1, _) = _run_day(c2, d_before)
    pd.testing.assert_frame_equal(p0, p1)        # every projection, baseline and metric column
    pd.testing.assert_frame_equal(l0, l1)        # every lineup incl. hindsight (day < cut)
    # the control: scrambling a date BEFORE the projected one changes the projection
    c3 = _clone(conn)
    _sql_scramble(c3, days[3], strictly_after=False)
    _, (p2, _, _) = _run_day(c3, d_before)
    assert not _same_projection(p0, p2)
    # the lineups depend on the cut-free frame only: the salary of a later date does not matter
    assert (p0["salary"] == p1["salary"]).all()


def test_l1_outcomes_of_the_day_itself_never_reach_projections_or_baseline_lineups(conn, days):
    day = days[14]
    sr, (p0, l0, _) = _run_day(conn, day)
    c2 = _clone(conn)
    with c2:       # RotoGuru outcomes of the day: points and minutes (not salary, position, team)
        c2.execute("UPDATE dfs_hist_salaries SET dk_pts = abs(random() % 70), "
                   "minutes = abs(random() % 38) WHERE game_date = ?", (day,))
    _, (p1, l1, _) = _run_day(c2, day)
    assert _same_projection(p0, p1, outcomes=False)
    methods = [m for m in l0["method"].unique() if m != "hindsight"]
    for m in methods:
        a = l0[l0["method"] == m].sort_values("rank")
        b = l1[l1["method"] == m].sort_values("rank")
        assert list(a["ids"]) == list(b["ids"]), m               # same lineups ...
        assert list(a["proj"]) == list(b["proj"]), m
    hs0 = l0[l0["method"] == "hindsight"]["ids"].iloc[0]
    hs1 = l1[l1["method"] == "hindsight"]["ids"].iloc[0]
    assert hs0 != hs1                                            # ... but hindsight moved


# --------------------------------------------------------------------------- L2 fit < S

def test_l2_bundle_and_salary_line_ignore_season_S_and_later(conn, days, tmp_path):
    S = "2021-22"
    ctx = P.Context(conn, S)
    b1 = REAL_LOAD_BUNDLE(conn, S, ctx, refit=True).to_json()
    c2 = _clone(conn)
    _sql_scramble(c2, "2021-09-01")              # every row of 2021-22 and 2022-23 (box, closing)
    shutil.rmtree(config.data_dir() / "proj", ignore_errors=True)    # drop the cached fits
    ctx2 = P.Context(c2, S)
    b2 = REAL_LOAD_BUNDLE(c2, S, ctx2, refit=True).to_json()
    assert b1 == b2
    # the control: scrambling a TRAINING season changes the fit
    c3 = _clone(conn)
    _sql_scramble(c3, "2020-09-01")
    shutil.rmtree(config.data_dir() / "proj", ignore_errors=True)
    b3 = REAL_LOAD_BUNDLE(c3, S, P.Context(c3, S), refit=True).to_json()
    assert b3 != b1
    # salary lines of the harness: rows of the season (and its Dec-2020 start) are not used
    sr = _season_run(conn)
    c4 = _clone(conn)
    _sql_scramble(c4, E.season_start(SEASON))
    sr4 = _season_run(c4)
    assert sr.lines == sr4.lines and sr.salary_fit == "ols_seasons_before_S"
    assert sr.lines["all_seasons"][1] > 0


def test_first_rotoguru_season_falls_back_to_the_fixed_salary_line():
    # no RotoGuru rows before 2014-15: nothing in-sample is ever fit
    with pytest.raises(ValueError):
        E.fit_salary_line(pd.DataFrame({"game_date": ["2014-11-01"] * 60, "salary": [5000] * 60,
                                        "pts": [20.0] * 60, "minutes": [30.0] * 60}), "2014-15")


# --------------------------------------------------------------------------- L3 season aggregates

FORBIDDEN = {"matchup_rollup", "season_matchups", "pt_defend", "pt_shot_defdist",
             "synergy_playtypes", "hustle_season", "player_hustle", "player_matchups", "stints",
             "player_game_oncourt", "prop_snapshots", "prop_compare", "line_snapshots",
             "dfs_salaries", "dfs_ownership_proxy", "dfs_contest_results",
             "dfs_contest_ownership", "dfs_contests", "injury_snapshots", "dfs_hist_salaries"}


def test_l3_projection_stage_reads_no_lagged_aggregate_tables(conn, days):
    day = days[10]

    def project():
        ctx = P.Context(conn, SEASON)
        return P.project_historical(conn, day, protocol="avail", n_sims=200, ctx=ctx,
                                    bundle=P.load_bundle(conn, SEASON))

    seen, res = _tables_read(conn, project)
    assert res.sims and seen
    assert not (seen & FORBIDDEN), sorted(seen & FORBIDDEN)
    assert {"player_games", "team_games"} <= seen
    # fitting the bundle is a box-score fit too
    S = "2021-22"
    ctx = P.Context(conn, S)
    seen_fit, _ = _tables_read(conn, lambda: REAL_LOAD_BUNDLE(conn, S, ctx, refit=True))
    assert not (seen_fit & FORBIDDEN), sorted(seen_fit & FORBIDDEN)


# --------------------------------------------------------------------------- L4 / L5 availability, starters

def _cloned_day_variant(conn, day, sql):
    c2 = _clone(conn)
    with c2:
        for s, args in sql:
            c2.execute(s, args)
    return c2


def test_l4_l5_the_avail_oracle_exposes_only_who_played_not_minutes_starters_or_margin(conn, days):
    """'Known availability' (L4): the only information taken from the projected day is which
    regulars did not play. Actual minutes, the starting five (L5: lock-45 mode never sees
    starters), points, margin and overtime of that day are not inputs."""
    day = days[12]
    base = P.project_historical(conn, day, protocol="avail", n_sims=300,
                                ctx=P.Context(conn, SEASON), bundle=P.load_bundle(conn, SEASON))
    gids = "(SELECT game_id FROM team_games WHERE game_date = ?)"
    c2 = _cloned_day_variant(conn, day, [
        (f"UPDATE player_games SET min = 3 + min * 0.5, pts = pts + 11, reb = reb + 5 "
         f"WHERE game_date = ?", (day,)),                          # still played (min > 0)
        (f"UPDATE team_games SET pts = pts + 25 WHERE game_date = ? AND is_home = 1", (day,)),
        (f"DELETE FROM player_starts WHERE game_id IN {gids}", (day,)),
        (f"INSERT INTO player_starts (game_id, player_id, team, source) "
         f"SELECT game_id, player_id, team, 'flip' FROM player_games WHERE game_date = ? AND "
         f"player_id % 1000 >= 5", (day,)),                        # the bench "starts"
    ])
    alt = P.project_historical(c2, day, protocol="avail", n_sims=300,
                               ctx=P.Context(c2, SEASON), bundle=P.load_bundle(c2, SEASON))
    assert len(base.sims) == len(alt.sims) > 0
    for a, b in zip(base.sims, alt.sims):
        assert np.array_equal(a.player_id, b.player_id)
        assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    # the control: who played IS information (a regular sitting out changes the projection)
    c3 = _clone(conn)
    victim = c3.execute("SELECT player_id, game_id FROM player_games WHERE game_date = ? AND "
                        "player_id % 1000 < 5 LIMIT 1", (day,)).fetchone()
    with c3:
        c3.execute("DELETE FROM player_games WHERE game_id = ? AND player_id = ?", tuple(victim[::-1]))
    alt2 = P.project_historical(c3, day, protocol="avail", n_sims=300,
                                ctx=P.Context(c3, SEASON), bundle=P.load_bundle(c3, SEASON))
    assert not all(np.array_equal(a.fp, b.fp) for a, b in zip(base.sims, alt2.sims)
                   if a.game_id == b.game_id)


def test_l4_report_labels_the_protocol_as_known_availability(conn):
    p, l, m = B.run_season(conn, SEASON, SET, n_dates=3, log=lambda m: None)
    rep = B.build_report(p, l.assign(season=SEASON), {SEASON: m}, SET, seasons=[SEASON])
    rep["seconds"] = 1.0
    txt = B.format_report(rep)
    assert "protocol avail" in txt and "known availability" in txt
    assert rep["settings"]["protocol"] == "avail"
    with pytest.raises(ValueError):
        B.run(conn, [SEASON], protocol="reports")        # real reports exist only for 2025-26


# --------------------------------------------------------------------------- L6 never features

def test_l6_salaries_and_props_are_never_projection_inputs(conn, days):
    day = days[11]
    sr, (p0, l0, _) = _run_day(conn, day)

    def project():
        return P.project_historical(conn, day, protocol="avail", n_sims=200, ctx=sr.ctx,
                                    bundle=sr.bundle)

    seen, _ = _tables_read(conn, project)
    assert not (seen & {"dfs_hist_salaries", "dfs_salaries", "prop_snapshots", "prop_compare",
                        "line_snapshots", "dfs_ownership_proxy"})
    c2 = _clone(conn)
    with c2:
        c2.execute("UPDATE dfs_hist_salaries SET salary = 3000 + (abs(random() % 90) * 100) "
                   "WHERE game_date >= ?", (E.season_start(SEASON),))
    _, (p1, l1, _) = _run_day(c2, day)
    assert _same_projection(p0, p1)                              # projections ignore salary
    # salary-implied lineups, by contrast, do depend on salary (the control)
    sal0 = l0[l0["method"] == "salary"]["ids"].iloc[0]
    sal1 = l1[l1["method"] == "salary"]["ids"].iloc[0]
    assert sal0 != sal1
    # no projection column is a function of salary: constant-salary pool, same means
    c3 = _clone(conn)
    with c3:
        c3.execute("UPDATE dfs_hist_salaries SET salary = 6000 WHERE game_date >= ?",
                   (E.season_start(SEASON),))
    _, (p3, _, _) = _run_day(c3, day)
    assert _same_projection(p0, p3)


# --------------------------------------------------------------------------- leak probe

def test_leak_probe_passes_on_the_clean_pipeline(conn, days):
    out = B.leak_probe(conn, SEASON, [days[9], days[12]], n_sims=300, log=lambda m: None)
    assert out["pass"] and out["n_dates"] == 2
    assert out["mean_abs_rel_diff"] < B.LEAK_TOL
    assert all(d["n_players"] > 10 and d["n_players_full"] == d["n_players_cut"]
               for d in out["dates"])


def test_leak_probe_cut_context_physically_removes_the_day_and_later(conn, days):
    day = days[10]
    ctx = P.Context(conn, SEASON)
    cut = B.cut_context(ctx, day)
    d = P._day_int(P._as_date(day))
    assert (ctx.h.hist["date"] >= d).any() and (ctx.games["day"] >= d).any()
    assert not (cut.h.hist["date"] >= d).any() and not (cut.h.tg["date"] >= d).any()
    assert not (cut.games["day"] >= d).any()
    assert ctx.h.hist["date"].max() >= d                         # the original is untouched


def test_leak_probe_detects_a_deliberate_lookahead(conn, days, monkeypatch):
    """Negative control: a pipeline that peeks at rows on/after the date must fail the probe."""
    orig = ST.rate_frame

    def peeking(conn_, queries, params=None, games=None, pos=None):
        out = orig(conn_, queries, params, games=games, pos=pos)
        qd = pd.to_datetime(queries["game_date"]).max()
        future = int((games["day"] >= int(qd.value // 86_400_000_000_000)).sum()) \
            if games is not None else 0
        for c in [c for c in out.columns if c.startswith("r_")]:
            out[c] = out[c] * (1.0 + 0.2 * (future > 0))
        return out

    monkeypatch.setattr(ST, "rate_frame", peeking)
    out = B.leak_probe(conn, SEASON, [days[10]], n_sims=300, log=lambda m: None)
    assert not out["pass"] and out["mean_abs_rel_diff"] > B.LEAK_TOL


# --------------------------------------------------------------------------- reports

def _fake_lineups(n_dates=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dates):
        day = f"2020-12-{k + 1:02d}" if k < 28 else f"2021-01-{k - 27:02d}"
        hs = 380 + rng.normal(0, 10)
        base = {"ours_mean": hs * 0.68, "ours_cash": hs * 0.69, "salary": hs * 0.5,
                "salary_avail": hs * 0.6, "fppg_proxy_avail": hs * 0.64, "hindsight": hs}
        shared = rng.normal(0, 15)                                 # slate effect
        for m, v in base.items():
            a = v + shared + rng.normal(0, 8) if m != "hindsight" else hs
            rows.append({"date": day, "method": m, "kind": "hindsight" if m == "hindsight"
                         else "single", "rank": 1, "actual": a, "proj": a, "salary": 49000,
                         "pct_rand": min(a / hs, 1.0), "pct_spend": min(a / hs, 1.0) * 0.9,
                         "ids": "x", "season": "2020-21"})
        for r in range(1, 4):
            rows.append({"date": day, "method": "ours_topn", "kind": "topn", "rank": r,
                         "actual": base["ours_mean"] - 3 * r + shared + rng.normal(0, 8),
                         "proj": 0.0, "salary": 49000, "pct_rand": 0.9, "pct_spend": 0.8,
                         "ids": f"t{r}", "season": "2020-21"})
    return pd.DataFrame(rows)


def test_lineup_report_ratios_pairs_and_topn():
    ld = _fake_lineups()
    rep = B.lineup_report(ld)
    m = rep["methods"]
    assert rep["n_dates"] == 40 and m["hindsight"]["ratio_to_hindsight"] == 1.0
    assert m["ours_mean"]["ratio_to_hindsight"] == pytest.approx(0.68, abs=0.05)
    assert m["ours_mean"]["mean_actual"] > m["salary"]["mean_actual"]
    p = rep["paired"]["ours_cash_minus_salary"]
    assert p["mean_diff"] > 0 and p["z"] > 3 and p["win_rate"] > 0.9 and p["n"] == 40
    assert "ours_mean_minus_hindsight" not in rep["paired"]
    # slate effects are shared across methods, so the paired SE is far below the unpaired one
    assert p["se"] < m["ours_cash"]["se"]
    t = rep["topn"]["ours_topn"]
    assert t["mean_lineups"] == 3 and t["best_of_n"] >= t["rank1"] - 30
    assert t["mean_actual"] < m["ours_mean"]["mean_actual"]
    # a date where a method failed is dropped for everybody
    bad = ld.copy()
    bad.loc[(bad["date"] == "2020-12-05") & (bad["method"] == "salary"), "actual"] = np.nan
    r2 = B.lineup_report(bad)
    assert r2["n_dates"] == 39 and r2["n_dates_dropped"] == 1
    assert B.lineup_report(pd.DataFrame()) == {}


def _fake_players(seed=0, n=3000, seasons=("2019-20", "2020-21")):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        y0 = int(s[:4])
        tier = rng.choice(["bench", "role", "star"], n, p=[.3, .5, .2])
        mu = np.where(tier == "star", 38, np.where(tier == "role", 24, 12)) + rng.normal(0, 3, n)
        sd = 0.35 * mu
        y = np.clip(mu + rng.normal(0, 1, n) * sd, 0, None)
        sim = mu[None, :] + rng.normal(0, 1, (400, n)) * sd[None, :]
        pit = MT.pit_randomized(sim, y, rng.random(n))
        crps = MT.crps_samples_matrix(sim[:200], y)
        day = [f"{y0}-12-{(i % 28) + 1:02d}" for i in range(n)]
        rows.append(pd.DataFrame({
            "date": day, "season": s, "rg_id": [f"r{i}" for i in range(n)],
            "player_id": np.arange(n), "game_id": "g", "team": "T", "pos": "PG",
            "salary": 5000.0, "y": y, "minutes": 20.0, "has_proj": True, "mean": mu,
            "q10": mu - 1.28 * sd, "q25": mu - 0.674 * sd, "q50": mu, "q75": mu + 0.674 * sd,
            "q90": mu + 1.28 * sd, "exp_min": 25.0, "p_play": 1.0, "tier": tier, "crps": crps,
            "pit": pit, "sal_all": mu + rng.normal(0, 8, n), "sal_prev": mu + rng.normal(0, 9, n),
            "sal_played": mu + rng.normal(0, 7, n), "sal_fixed": mu + rng.normal(0, 10, n),
            "b_season_avg": mu + rng.normal(0, 4, n), "b_fppg": mu + rng.normal(0, 4, n),
            "b_last10": mu + rng.normal(0, 6, n)}))
    return pd.concat(rows, ignore_index=True)


def test_player_report_gains_calibration_and_naive_crps():
    df = B.with_naive_crps(_fake_players())
    first = df[df["season"] == "2019-20"]
    second = df[df["season"] == "2020-21"]
    assert first["crps_naive"].isna().all()                  # nothing earlier: never in-sample
    assert second["crps_naive"].notna().all()
    rep = B.player_report(second)
    assert rep["n"] == len(second) and rep["share_with_projection"] == 1.0
    assert rep["vs"]["fppg_proxy"]["gain_pct"] > 5 and rep["vs"]["last10"]["gain_pct"] > 10
    assert rep["best_salary"].startswith("salary[") and rep["vs_best_salary"]["gain_pct"] > 0
    assert rep["cov80"] == pytest.approx(0.8, abs=0.03) and rep["cov50"] == pytest.approx(0.5, abs=0.03)
    assert all(abs(x - 0.1) < 0.025 for x in rep["pit_deciles"])
    assert rep["crps_vs_naive_normal"]["gain_pct"] > -5        # calibrated sims ~ the naive CV normal
    assert {r["tier"] for r in rep["by_tier"]} == {"bench", "role", "star"}
    assert abs(rep["bias"]) < 0.5
    # the naive reference of a season is fit on earlier seasons only
    three = _fake_players(seasons=("2019-20", "2020-21", "2021-22"))
    ref = B.with_naive_crps(three)
    scr = three.copy()
    late = scr["season"] == "2021-22"
    scr.loc[late, ["y", "b_fppg", "tier"]] = [123.0, 1.0, "star"]
    out = B.with_naive_crps(scr)
    mid = ref["season"] == "2020-21"
    assert np.array_equal(ref.loc[mid, "crps_naive"], out.loc[mid, "crps_naive"])
    early = scr.copy()
    early.loc[scr["season"] == "2019-20", "y"] = 5.0           # ... but the training rows matter
    assert not np.allclose(B.with_naive_crps(early).loc[mid, "crps_naive"],
                           ref.loc[mid, "crps_naive"])
    empty = B.player_report(df.iloc[:0])
    assert empty["pool_rows"] == 0


def test_acceptance_rows_and_text_report_from_synthetic_frames():
    players = B.with_naive_crps(_fake_players())
    ld = _fake_lineups()
    ld["season"] = "2020-21"
    rep = {"generated": "now", "harness_version": 1, "settings": B.asdict(SET),
           "seasons": ["2019-20", "2020-21"], "seconds": 1.0, "meta": {}, "per_season": {}}
    for s in rep["seasons"]:
        rep["per_season"][s] = {"player": B.player_report(players[players["season"] == s]),
                                "lineup": B.lineup_report(ld) if s == "2020-21" else {}}
    rep["pooled"] = {"player": B.player_report(players), "lineup": B.lineup_report(ld)}
    rep["pooled"]["acceptance"] = B.acceptance(rep["pooled"]["player"], rep["pooled"]["lineup"])
    acc = {a["metric"]: a for a in rep["pooled"]["acceptance"]}
    assert acc["80% coverage"]["ok"] and acc["50% coverage"]["ok"]
    assert acc["PIT deciles (randomised), largest share"]["ok"]
    assert acc["DK FP MAE gain vs best of salary-implied / fppg / last10"]["ok"] is True
    assert acc["hindsight ratio (cash)"]["ok"] is True
    txt = B.format_report(rep)
    assert "section 7.3" in txt and "POOLED" in txt and "ours_cash_minus_salary" in txt
    json.dumps(E._clean(rep))                                   # JSON-safe


def test_cache_key_changes_with_model_code_and_bundle(conn, monkeypatch):
    sr = _season_run(conn)
    k0 = sr.cache().root
    monkeypatch.setenv("COURTSIDE_BT_CODE_TAG", "edit-1")
    k1 = _season_run(conn).cache().root
    monkeypatch.setenv("COURTSIDE_BT_CODE_TAG", "edit-2")
    k2 = _season_run(conn).cache().root
    assert len({k0, k1, k2}) == 3
    monkeypatch.delenv("COURTSIDE_BT_CODE_TAG")
    assert B.code_fingerprint() == B.code_fingerprint() and len(B.code_fingerprint()) == 12
    assert _season_run(conn).cache().root == k0
    # a different bundle (e.g. refit parameters) is a different cache
    b2 = P.Bundle(season=SEASON, minutes=M.default_params(SEASON), state=ST.default_params(),
                  copula=C.default_copula_params(SEASON))
    b2.state = {**b2.state, "m0": 777.0}
    assert B.config_key(SET, b2.to_json(), CLASSIC) != B.config_key(SET, sr.bundle_json, CLASSIC)


def test_cli_registration_and_arguments():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="sub_cmd")
    handlers = B.add_commands(sub)
    assert set(handlers) == {"backtest-dfs"}
    a = p.parse_args(["backtest-dfs", "--seasons", "2019-20", "2020-21", "--dates", "20",
                      "--protocol", "avail", "--sims", "500", "--topn", "5", "--workers", "3",
                      "--leak-probe", "2", "--modes", "cash"])
    assert a.seasons == ["2019-20", "2020-21"] and a.dates == 20 and a.workers == 3
    assert a.topn == 5 and a.leak_probe == 2 and a.modes == ["cash"] and not a.force
