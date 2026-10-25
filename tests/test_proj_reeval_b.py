"""Re-evaluation workstream B (docs/research/reeval-B.md): F8 / F8c player vs opponent (H1),
F7 referee crews (H3), F3x tracking-dashboard opportunity (H4).

The switch (OFF = the registry and every existing layer exactly as before), L1 scramble and
L5 truncation leak tests per group, planted signals, empirical-Bayes behaviour (no history ->
exactly 0, placebo-corrected variance components), and the per-day hooks."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from courtside.proj import context as CX
from courtside.proj import features as FX
from courtside.proj.features import common as CM
from courtside.proj.features import opp_history as F8
from courtside.proj.features import referees as F7
from courtside.proj.features import tracking_ext as F3X

from test_proj_features import SEASONS, TEAMS, make_league, params

SEASON_LIST = list(SEASONS)
LAST = SEASON_LIST[-1]
PAIR = (100, "DDD")              # planted: player 100 scores 1.6x vs DDD
WHISTLE = 7                      # planted: official 7's games have 1.5x FTA
SMALL = {"min_data_in_leaf": 50, "min_sum_hessian_in_leaf": 1.0, "num_threads": 1}
OLD_GROUPS = {"P", "F1", "F2", "F3", "F5", "F6", "F4", "F9"}


@pytest.fixture
def b_on():
    was = FX.context_b_on()
    FX.set_context_b(True)
    yield
    FX.set_context_b(was)


def _officials(conn) -> pd.DataFrame:
    """3 of 12 synthetic officials per game (deterministic)."""
    gids = [r[0] for r in conn.execute("SELECT DISTINCT game_id FROM team_games ORDER BY 1")]
    rng = np.random.default_rng(3)
    rows = []
    for g in gids:
        for o in rng.choice(12, 3, replace=False):
            rows.append((g, int(o) + 1, f"F{o + 1}", f"L{o + 1}", str(o + 1)))
    return pd.DataFrame(rows, columns=["game_id", "official_id", "first_name", "last_name",
                                       "jersey_num"])


def _plant(conn) -> sqlite3.Connection:
    """Copy with the planted pair and whistle effects, team-game FTA kept consistent, and a
    coach table (AAA's coach moves to BBB in the last season)."""
    c = sqlite3.connect(":memory:")
    conn.backup(c)
    pid, opp = PAIR
    c.execute("UPDATE player_games SET pts = CAST(pts * 1.6 + 0.5 AS INTEGER) WHERE player_id = ?"
              " AND game_id IN (SELECT game_id FROM team_games WHERE team = ? )", (pid, opp))
    off = _officials(conn)
    g7 = sorted(set(off.loc[off["official_id"] == WHISTLE, "game_id"]))
    q = ",".join("?" * len(g7))
    c.execute(f"UPDATE player_games SET fta = CAST(fta * 1.5 + 0.5 AS INTEGER) "
              f"WHERE game_id IN ({q})", g7)
    c.execute("UPDATE team_games SET fta = (SELECT SUM(p.fta) FROM player_games p WHERE "
              "p.game_id = team_games.game_id AND p.team = team_games.team)")
    rows = []
    for s in SEASON_LIST:
        for i, t in enumerate(TEAMS):
            cid = 900 + i
            if s == LAST and t == "BBB":
                cid = 900                  # AAA's coach
            if s == LAST and t == "AAA":
                cid = 950
            rows.append((s, i + 1, t, cid, f"Coach {cid}", "Head Coach", 0, 1))
    c.executemany("INSERT INTO team_coaches VALUES (?,?,?,?,?,?,?,?)", rows)
    c.commit()
    return c


def _ext_rows(conn) -> None:
    """player_tracking_ext rows for every player-game: potential assists ~ 2 x AST, drives ~
    minutes / 4, touches ~ 2 x minutes."""
    rng = np.random.default_rng(4)
    pg = pd.read_sql_query("SELECT game_id, player_id, team, game_date, season, min, ast, pts "
                           "FROM player_games", conn)
    rows = []
    for r in pg.itertuples():
        drv = int(rng.poisson(r.min / 4))
        rows.append((r.game_id, r.player_id, r.team, r.game_date, r.season, "Regular Season",
                     float(r.min), int(r.min * 1.2), int(r.min * 1.2), int(r.ast), 0, 0,
                     int(2 * r.ast + rng.poisson(1)), int(2.3 * r.ast), int(r.ast), drv,
                     drv // 3, drv // 2, 0, 0, int(drv * 0.6), drv // 4, drv // 8,
                     int(rng.poisson(drv / 10)), 0, int(2 * r.min), int(r.min), r.min / 10,
                     3.0, 2.0, int(rng.poisson(1)), int(rng.poisson(1)), int(rng.poisson(2))))
    conn.executemany("INSERT INTO player_tracking_ext VALUES (" + ",".join("?" * 33) + ")",
                     rows)
    conn.commit()


@pytest.fixture(scope="module")
def league():
    c = _plant(make_league())
    _ext_rows(c)
    return c


@pytest.fixture(scope="module")
def officials(league):
    return _officials(league)


def frame_rows(conn, officials, monkeypatch=None, groups=("F7", "F8", "F8r", "F8c", "F3x")):
    sp = params()
    wf = CM.base_rows(conn, SEASON_LIST, sp)
    tg = FX.team_game_features(conn, wf, sp)
    fr = FX.attach(wf, tg)
    was = FX.context_b_on()
    FX.set_context_b(True)
    orig = F7.load_officials
    F7.load_officials = lambda seasons: officials
    try:
        return FX.extend_frame(conn, fr, list(groups), cache=False)
    finally:
        F7.load_officials = orig
        FX.set_context_b(was)


@pytest.fixture(scope="module")
def frame(league, officials):
    return frame_rows(league, officials)


def _copy(conn):
    c2 = sqlite3.connect(":memory:")
    conn.backup(c2)
    return c2


def _scramble(conn, cut: str, seed: int = 2):
    """Every counting stat of player_games / team_games / player_tracking_ext on or after `cut`
    re-randomised (who played and minutes kept)."""
    c2 = _copy(conn)
    rng = np.random.default_rng(seed)
    cols = ["fgm", "fga", "fg3m", "fg3a", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk",
            "tov", "pf", "pts"]
    for (rid,) in c2.execute("SELECT rowid FROM player_games WHERE game_date >= ?",
                             (cut,)).fetchall():
        vals = [int(v) for v in rng.integers(0, 12, len(cols))]
        c2.execute("UPDATE player_games SET " + ", ".join(f"{k}=?" for k in cols)
                   + " WHERE rowid=?", (*vals, rid))
    c2.execute("UPDATE team_games SET pf = pf + 9, fga = fga + 6, tov = tov + 3, fta = fta + 5 "
               "WHERE game_date >= ?", (cut,))
    c2.execute("UPDATE player_tracking_ext SET pot_ast = abs(random() % 13), "
               "drives = abs(random() % 17), drive_pts = abs(random() % 11), "
               "paint_touches = abs(random() % 9) WHERE game_date >= ?", (cut,))
    c2.commit()
    return c2


def _cols(fr, prefixes):
    return [c for c in fr.columns if c.startswith(prefixes)]


# ------------------------------------------------------------------------- switch / OFF

def test_switch_default_off_registry_unchanged():
    assert FX.CONTEXT_B_DEFAULT is False
    was = FX.context_b_on()
    FX.set_context_b(False)
    try:
        assert set(FX.GROUPS) == OLD_GROUPS
        assert FX.REEVAL_GROUPS == ()
        live = ("P", "F2", "F3", "F5", "F6", "F4")
        before = {s: (FX.model_columns(s, live), FX.monotone_for(s, live)) for s in CM.DK_STATS}
        FX.set_context_b(True)
        assert set(FX.GROUPS) == OLD_GROUPS | {"F7", "F8", "F8r", "F8c", "F3x"}
        assert set(FX.REEVAL_GROUPS) == {"F7", "F8", "F8r", "F8c", "F3x"}
        after = {s: (FX.model_columns(s, live), FX.monotone_for(s, live)) for s in CM.DK_STATS}
        assert before == after
        FX.set_context_b(False)
        assert set(FX.GROUPS) == OLD_GROUPS and FX.REEVAL_GROUPS == ()
    finally:
        FX.set_context_b(was)


def test_off_ignores_groups(league):
    """Switch OFF: extend_frame ignores the B groups (frame unchanged), as any unknown group."""
    was = FX.context_b_on()
    FX.set_context_b(False)
    try:
        sp = params()
        wf = CM.base_rows(league, SEASON_LIST, sp)
        fr = FX.attach(wf, FX.team_game_features(league, wf, sp))
        out = FX.extend_frame(league, fr, ["F7", "F8", "F8r", "F8c", "F3x"], cache=False)
        assert out is fr
    finally:
        FX.set_context_b(was)


def test_existing_layer_bit_identical_with_b_columns(frame):
    """A P + F2 layer fit on a frame carrying the B columns is byte-identical to the same layer
    fit without them (the B groups only act when a variant names them)."""
    train = (frame["season"] < LAST).to_numpy()
    b = _cols(frame, ("f7_", "f8_", "f8r_", "f8c_", "t3x_"))
    assert len(b) > 20
    m1 = CX.fit(frame, train, ("P", "F2"), season=LAST, params=SMALL, max_rounds=40)
    m2 = CX.fit(frame.drop(columns=b), train, ("P", "F2"), season=LAST, params=SMALL,
                max_rounds=40)
    assert m1.to_json() == m2.to_json()


# ------------------------------------------------------------------------- leak tests

@pytest.mark.parametrize("prefixes", [("f8_",), ("f8r_",), ("f8c_",), ("f7_",), ("t3x_",)])
def test_no_lookahead_scramble(league, officials, frame, prefixes):
    """L1 per group: every stat on or after the cut re-randomised -> every feature of the games
    up to the cut day identical, later ones changed."""
    cut = (SEASONS[LAST] + timedelta(days=30)).isoformat()
    other = frame_rows(_scramble(league, cut), officials)
    cols = _cols(frame, prefixes)
    assert len(cols) >= 4
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other[other["game_date"] <= cut].sort_values(key)
    assert len(a) == len(b) and len(a) > 1000
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())
    a2 = frame[frame["game_date"] > cut].sort_values(key)
    b2 = other[other["game_date"] > cut].sort_values(key)
    assert not np.allclose(a2[cols].to_numpy(float), b2[cols].to_numpy(float), equal_nan=True)


def test_no_lookahead_truncation(league, officials, frame):
    """L5: every row after the cut deleted -> every B feature up to the cut identical."""
    cut = (SEASONS[LAST] + timedelta(days=40)).isoformat()
    c2 = _copy(league)
    for t in ("player_games", "team_games", "player_tracking_ext"):
        c2.execute(f"DELETE FROM {t} WHERE game_date > ?", (cut,))
    c2.commit()
    other = frame_rows(c2, officials)
    cols = _cols(frame, ("f7_", "f8_", "f8r_", "f8c_", "t3x_"))
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other.sort_values(key)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


# ------------------------------------------------------------------------- F8 (H1)

def test_f8_no_history_is_zero(frame):
    """A player's first game vs an opponent: exactly 0 on every F8 column (theta = 1)."""
    first = frame.sort_values("day").drop_duplicates(["player_id", "opp"])
    cols = list(F8.COLS) + list(F8.COLS_R)
    assert np.all(np.abs(first[cols].to_numpy()) < 1e-12)


def test_f8_planted_pair(frame):
    """The planted pair (1.6x points) gets a clearly positive shrunk effect late; the same
    player vs other opponents stays near 0; shrinkage keeps it below the raw log 1.6."""
    late = frame[(frame["season"] == LAST) & (frame["day"] > frame["day"].max() - 30)]
    pid, opp = PAIR
    a = late[(late["player_id"] == pid) & (late["opp"] == opp)]["f8_pts"]
    o = late[(late["player_id"] == pid) & (late["opp"] != opp)]["f8_pts"]
    assert len(a) and len(o)
    assert a.mean() > 0.12
    # (with 3 opponents the planted games lift his level vs the others: slightly negative)
    assert a.mean() > o.mean() + 0.15 and o.mean() > -0.2
    assert a.max() < np.log(1.6)
    rest = late[late["player_id"] != pid]["f8_pts"]
    assert rest.abs().mean() < 0.05


def test_f8_variance_components_detect_planted(league, frame):
    """Placebo-corrected tau2: real > 0 with a strong planted per-pair effect; ~0 without."""
    fr = frame[frame["season_type"] == "Regular Season"].reset_index(drop=True)
    key = F8._franchise(fr["opp"], fr["season"])
    vc0 = F8.variance_components(fr, key)
    assert abs(vc0["reb"]["z"]) < 3.5
    rng = np.random.default_rng(9)
    eff = {k: float(np.exp(rng.normal(0, 0.25))) for k in
           set(zip(fr["player_id"], key))}
    m = np.array([eff[k] for k in zip(fr["player_id"], key)])
    fr2 = fr.copy()
    fr2["reb"] = rng.poisson(fr2["E_reb"].to_numpy() * np.exp(
        np.nan_to_num(fr2["f2_dvp_reb_grp"].to_numpy())) * m)
    vc = F8.variance_components(fr2, key)
    assert 0.02 < vc["reb"]["tau2"] < 0.15, vc["reb"]
    assert vc["reb"]["z"] > 3
    sh = F8.split_half(fr2, key, min_games=6)
    assert sh["reb"]["real"]["r"] > sh["reb"]["placebo"]["r"] + 0.15


def test_f8c_follows_the_coach(frame):
    """F8c is keyed on the opponent's head coach: AAA's new coach in the last season has no
    history, so a CCC / DDD player's first game vs AAA that season has F8c evidence 0 while F8
    (keyed on the franchise) has some; the planted DDD pair (same coach) persists."""
    last = frame[(frame["season"] == LAST) & frame["team"].isin(["CCC", "DDD"])]
    first = last[last["opp"] == "AAA"].sort_values("day").drop_duplicates("player_id")
    assert len(first) > 5
    assert (first["f8c_n"] == 0).all() and (first["f8_n"] > 0).mean() > 0.8
    late = frame[(frame["season"] == LAST) & (frame["day"] > frame["day"].max() - 30)]
    a = late[(late["player_id"] == PAIR[0]) & (late["opp"] == PAIR[1])]["f8c_pts"]
    assert a.mean() > 0.1


# ------------------------------------------------------------------------- F7 (H3)

def test_f7_planted_whistle(frame, officials):
    """Games officiated by the planted whistle-heavy official get a positive crew FTA index
    late in the data; the index is the same for both teams of a game."""
    late = frame[(frame["season"] == LAST)]
    g7 = set(officials.loc[officials["official_id"] == WHISTLE, "game_id"])
    w = late[late["game_id"].isin(g7)].drop_duplicates("game_id")["f7_fta"]
    o = late[~late["game_id"].isin(g7)].drop_duplicates("game_id")["f7_fta"]
    assert w.mean() > o.mean() + 0.03
    per_game = late.groupby("game_id")["f7_fta"].nunique()
    assert (per_game == 1).all()


def test_f7_day_hook_matches_table_and_needs_no_result(league, officials, frame):
    """F7 for one day from the per-day hook (posted crews, history strictly before the day)
    equals the frame's columns, also when the day's and later results are not in the DB."""
    day = (SEASONS[LAST] + timedelta(days=50)).isoformat()
    d = frame[frame["game_date"] == day].drop_duplicates(["game_id", "team"])
    assert len(d)
    tg = pd.read_sql_query("SELECT game_id, team, opp FROM team_games WHERE is_home = 1 AND "
                           "game_date = ?", league, params=(day,))
    games = pd.DataFrame({"game_id": tg["game_id"], "home": tg["team"], "away": tg["opp"]})
    crews = officials[officials["game_id"].isin(games["game_id"])][["game_id", "official_id"]]
    hist_frame = frame[frame["game_date"] < day]
    hk = F7.History(league, frame, LAST, officials=officials)
    got = hk.day_features(day, games, crews).set_index(["game_id", "team"])
    ref = d.set_index(["game_id", "team"])[list(F7.TEAM_COLS)]
    np.testing.assert_allclose(got.loc[ref.index, list(F7.TEAM_COLS)].to_numpy(float),
                               ref.to_numpy(float), rtol=1e-9, atol=1e-12)
    # live: the day's games are not completed yet (only earlier history; params of < S)
    hk2 = F7.History(league, hist_frame, LAST, officials=officials)
    got2 = hk2.day_features(day, games, crews).set_index(["game_id", "team"])
    np.testing.assert_allclose(got2.loc[ref.index, list(F7.TEAM_COLS)].to_numpy(float),
                               ref.to_numpy(float), rtol=1e-9, atol=1e-12)


def test_f7_crews_from_assignments(officials):
    c = sqlite3.connect(":memory:")
    from courtside import db
    db.migrate(c)
    rows = [("2024-01-05", "0022300999", "F3 L3", 3, "crew_chief"),
            ("2024-01-05", "0022300999", "f5  l5".replace("  ", " "), None, "referee"),
            ("2024-01-05", "0022300999", "Unknown Person", 4, "umpire"),
            ("2024-01-05", None, "F9 L9", 9, "replay_center")]
    c.executemany("INSERT INTO referee_assignments(nba_day, game_id, official_name, "
                  "official_number, role, captured_at) VALUES (?,?,?,?,?, 'x')", rows)
    got = F7.crews_from_assignments(c, "2024-01-05", officials)
    assert sorted(got["official_id"].tolist()) == [3, 4, 5]     # #4 mapped by jersey number
    assert set(got["game_id"]) == {"0022300999"}


# ------------------------------------------------------------------------- F3x (H4)

def test_f3x_era_and_gap(league, frame):
    """Before the dashboard era: NaN; with history, the assist gap tracks potential assists."""
    late = frame[frame["season"] == LAST].dropna(subset=["t3x_gap_ast"])
    assert len(late) > 500
    assert np.corrcoef(np.log(late["t3x_impl_ast"]), np.log(late["r_ast"]))[0, 1] > 0.5
    c2 = _copy(league)
    c2.execute("DELETE FROM player_tracking_ext")
    c2.commit()
    h = F3X.load_history(c2)
    q = frame[["game_id", "player_id", "day", "sidx", "grp"]].head(50)
    assert F3X.player_features(h, q).isna().all().all()
    cov = F3X.coverage(league)
    assert set(cov["season"]) == set(SEASON_LIST) and (cov["pot_ast"] > 0).all()


def test_registry_columns_exist(frame, b_on):
    for s in CM.DK_STATS:
        for g in ("F7", "F8", "F8r", "F8c", "F3x"):
            for c in FX.model_columns(s, [g]):
                assert c in frame.columns, (s, g, c)
        mono = FX.monotone_for(s, ["F8", "F8r", "F8c", "F3x"])
        assert all(c in FX.model_columns(s, ["F8", "F8r", "F8c", "F3x"]) for c in mono)


def test_b_layer_fits_and_clips(frame, b_on):
    """A P + F2 + F8 + F7 layer fits on seasons < S and predicts finite, clipped multipliers."""
    train = (frame["season"] < LAST).to_numpy()
    m = CX.fit(frame, train, ("P", "F2", "F8", "F7"), season=LAST, params=SMALL, max_rounds=40)
    test = frame[frame["season"] == LAST]
    mult = m.predict(test)
    for s in CM.DK_STATS:
        assert np.isfinite(mult[s]).all()
        assert mult[s].min() >= CX.CLIP[0] - 1e-12 and mult[s].max() <= CX.CLIP[1] + 1e-12
