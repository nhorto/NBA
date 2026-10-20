"""Phase 2F F4: shot quality and shooting luck (xFG). Parser dialects, the incremental store, the
xFG model (walk-forward, calibrated, JSON round trip), leak tests in the L1 / L5 style
(scrambling or deleting every shot / game on or after a cut leaves the features of every game up
to the cut identical), planted luck, the registry, and the per-day hook reproducing the frame."""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from courtside.proj import features as FX
from courtside.proj import state as ST
from courtside.proj.features import common as CM
from courtside.proj.features import shots as F4

from test_proj_features import SEASONS, make_league, params

SEASON_LIST = list(SEASONS)
HOT = 101                 # AAA's second guard: makes 3s far above expectation in 2023-24


# ------------------------------------------------------------------------- synthetic shots

def synth_shots(conn: sqlite3.Connection, seed: int = 0, hot: int | None = HOT) -> pd.DataFrame:
    """Shot rows (store format) for every player-game of a test league, consistent with its box
    counts where they exist: 2s at 1-20 ft (make prob falls with distance), 3s at 22-27 ft,
    free throws. `hot` makes 85% of his 3s in the last season."""
    rng = np.random.default_rng(seed)
    pg = pd.read_sql_query("SELECT * FROM player_games ORDER BY game_id, player_id", conn)
    tg = pd.read_sql_query("SELECT game_id, season FROM team_games", conn).drop_duplicates(
        "game_id").set_index("game_id")["season"]
    last = max(tg)
    rows = []
    order = {}
    for r in pg.itertuples():
        m = float(r.min or 0)
        if m <= 0:
            continue
        fga = int(r.fga) if r.fga is not None and not pd.isna(r.fga) else int(rng.poisson(0.38 * m))
        n3 = int(r.fg3a) if getattr(r, "fg3a", None) not in (None,) and not pd.isna(r.fg3a) \
            else int(rng.poisson(0.12 * m))
        n3 = min(n3, fga)
        n2 = fga - n3
        fta = int(r.fta) if r.fta is not None and not pd.isna(r.fta) else 0
        for kind, n in ((2, n2), (3, n3), (1, fta)):
            for _ in range(n):
                o = order.get(r.game_id, 0) + 1
                order[r.game_id] = o
                if kind == 1:
                    dist = ax = ay = xl = np.nan
                    made = int(rng.random() < 0.78)
                    sub = 4
                else:
                    dist = rng.uniform(1, 20) if kind == 2 else rng.uniform(22, 27)
                    ang = rng.uniform(0, np.pi)
                    ax, ay = abs(dist * np.cos(ang)), dist * np.sin(ang)
                    xl = ax
                    p = (0.68 - 0.015 * dist) if kind == 2 else 0.36
                    if hot is not None and kind == 3 and r.player_id == hot and \
                            tg[r.game_id] == last:
                        p = 0.85
                    made = int(rng.random() < p)
                    sub = 0 if dist > 6 else int(rng.integers(1, 4))
                rows.append((r.game_id, o, r.team, int(r.player_id), kind, made, dist, ax, ay,
                             xl, sub, *[int(rng.random() < 0.1) if kind > 1 else 0
                                        for _ in F4.DESCS],
                             *[int(rng.random() < 0.1) if kind > 1 else 0 for _ in F4.QUALS],
                             int(rng.integers(1, 5)), float(rng.uniform(0, 720)),
                             int(rng.integers(0, 120)), int(rng.integers(0, 120)),
                             int(made and kind > 1 and rng.random() < 0.6),
                             int((not made) and kind > 1 and rng.random() < 0.05)))
    return F4._types(pd.DataFrame(rows, columns=F4.STORE_COLS))


def write_store(shots: pd.DataFrame, season_of: dict) -> None:
    d = F4.store_dir()
    d.mkdir(parents=True, exist_ok=True)
    s = shots["game_id"].map(season_of)
    for season, part in shots.groupby(s):
        part.reset_index(drop=True).to_parquet(d / f"shots_{season}_v{F4.STORE_VERSION}.parquet")


def install_shots(conn: sqlite3.Connection, **kw) -> pd.DataFrame:
    """Synthetic store for `conn` under the current COURTSIDE_DATA."""
    sh = synth_shots(conn, **kw)
    so = dict(conn.execute("SELECT game_id, season FROM team_games").fetchall())
    write_store(sh, so)
    return sh


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("f4data")
    mp.setenv("COURTSIDE_DATA", str(d))
    yield d
    mp.undo()


@pytest.fixture(scope="module")
def league(data_dir):
    c = make_league()
    install_shots(c)
    return c


def frame_rows(conn) -> pd.DataFrame:
    sp = params()
    wf = CM.base_rows(conn, SEASON_LIST, sp)
    tg = FX.team_game_features(conn, wf, sp)
    fr = FX.attach(wf, tg)
    return FX.extend_frame(conn, fr, ["F4"], season="2023-24", cache=False)


@pytest.fixture(scope="module")
def frame(league):
    return frame_rows(league)


def _copy(conn):
    c2 = sqlite3.connect(":memory:")
    conn.backup(c2)
    return c2


# ------------------------------------------------------------------------- parser / store

def _payload(acts):
    return {"meta": {}, "game": {"gameId": "0022300001", "actions": acts}}


def _act(n, **kw):
    a = {"orderNumber": n * 10000, "actionNumber": n, "period": 1, "clock": "PT10M00.00S",
         "teamTricode": "AAA", "personId": 7, "scoreHome": "0", "scoreAway": "0"}
    a.update(kw)
    return a


def test_parse_dialects():
    acts = [
        _act(1, actionType="3pt", subType="Jump Shot", descriptor="step back", shotResult="Made",
             xLegacy=-227, yLegacy=60, shotDistance=23.5, qualifiers=["fastbreak"],
             assistPersonId=9, scoreHome="3"),
        _act(2, actionType="2pt", subType="layup", descriptor="drivingfingerroll",
             shotResult=": Missed", xLegacy=5, yLegacy=10, shotDistance=1.1, blockPersonId=4,
             teamTricode="BBB", personId=8, scoreHome="3"),
        _act(3, actionType="freethrow", subType="1 of 2", shotResult="Made", scoreHome="4"),
        _act(4, actionType="2pt", subType="DUNK", descriptor="alley-oop", shotResult="Made",
             xLegacy=0, yLegacy=0, shotDistance=0.0, scoreHome="6"),
        _act(5, actionType="rebound", subType="defensive"),
        _act(6, actionType="2pt", shotResult="", xLegacy=0, yLegacy=0),         # no result: skip
    ]
    d = F4.parse_game(_payload(acts), "0022300001")
    assert list(d["kind"]) == [3, 2, 1, 2]
    assert list(d["made"]) == [1, 0, 1, 1]
    a, b, ft, dk = (d.iloc[i] for i in range(4))
    assert a["ax"] == pytest.approx(22.7) and a["ay"] == pytest.approx(6.0)
    assert a["xl"] == pytest.approx(-22.7)
    assert a["sub"] == 0 and a["d_stepback"] == 1 and a["q_fastbreak"] == 1
    assert a["assisted"] == 1 and a["blocked"] == 0 and a["sh"] == 0         # score BEFORE
    assert b["sub"] == 1 and b["d_driving"] == 1 and b["d_fingerroll"] == 1
    assert b["blocked"] == 1 and b["assisted"] == 0 and b["team"] == "BBB" and b["sh"] == 3
    assert ft["sub"] == 4 and np.isnan(ft["dist"]) and ft["sh"] == 3
    assert dk["sub"] == 2 and dk["d_alleyoop"] == 1 and dk["sh"] == 4


def test_store_incremental(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    root = tmp_path / "raw" / "nba_cdn" / "playbyplay"

    def put(day, gid, n):
        p = root / day / f"{gid}.json.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        acts = [_act(i + 1, actionType="2pt", subType="Layup", shotResult="Made", xLegacy=0,
                     yLegacy=5, shotDistance=0.5) for i in range(n)]
        p.write_bytes(gzip.compress(json.dumps(_payload(acts)).encode()))
    put("2023-10-24", "0022300001", 3)
    put("2023-10-24", "0012300001", 3)                 # preseason: ignored
    put("2018-10-24", "0021800001", 3)                 # before the shot era: ignored
    st = F4.update_store()
    assert set(st) == {"2023-24"} and st["2023-24"]["game_id"].nunique() == 1
    put("2023-10-25", "0022300002", 2)
    st = F4.update_store()
    assert st["2023-24"]["game_id"].nunique() == 2 and len(st["2023-24"]) == 5
    assert (F4.store_dir() / f"shots_2023-24_v{F4.STORE_VERSION}.parquet").exists()


# ------------------------------------------------------------------------- xFG model

def test_xfg_beats_distance_free_and_round_trips():
    rng = np.random.default_rng(3)
    n = 40000
    dist = rng.uniform(0, 27, n)
    is3 = (dist > 22).astype(int)
    dunk = (dist < 3) & (rng.random(n) < 0.4)
    p = np.where(dunk, 0.9, np.where(is3 == 1, 0.36, 0.65 - 0.015 * dist))
    s = pd.DataFrame({"kind": np.where(is3 == 1, 3, 2), "made": (rng.random(n) < p).astype(int),
                      "dist": dist, "ax": dist / 2, "ay": dist / 2,
                      "sub": np.where(dunk, 2, 0), "is3": is3, "secs": 300.0, "ot": 0,
                      "margin": 0.0,
                      "season": np.where(np.arange(n) < n // 2, "2021-22", "2022-23")})
    for c in F4.SHOT_FEATURES:
        if c not in s:
            s[c] = 0
    m = F4.fit_xfg(s)
    assert m.train_seasons == ["2021-22", "2022-23"] and set(m.platt) == {"2", "3"}
    px = m.predict(s)
    pb = F4.predict_baseline(F4.fit_baseline(s), s)
    y = s["made"].to_numpy()
    qx, qb = F4.xfg_quality(y, px), F4.xfg_quality(y, pb)
    assert qx["logloss"] < qb["logloss"] - 0.005           # the shot type is worth something
    assert 0.8 < qx["cal_slope"] < 1.2 and abs(qx["mean_p"] - qx["mean_y"]) < 0.01
    m2 = F4.XFGModel.from_json(m.to_json())
    np.testing.assert_array_equal(m2.predict(s), px)
    m3 = F4.XFGModel.from_json(json.loads(m.to_json()))   # dict form (ContextModel.meta)
    np.testing.assert_array_equal(m3.predict(s), px)


def test_xfg_walk_forward(league):
    m = F4.xfg_for_season(league, "2023-24", cache=False)
    assert m.train_seasons == ["2021-22", "2022-23"]


# ------------------------------------------------------------------------- no lookahead

def _scramble(conn, cut: str):
    """Every shot of a game on or after `cut` re-randomised in the store; box stats on or after
    the cut changed too."""
    c2 = _copy(conn)
    ids = {r[0] for r in c2.execute("SELECT DISTINCT game_id FROM team_games WHERE game_date "
                                    ">= ?", (cut,))}
    sh = synth_shots(conn, seed=99)
    keep = synth_shots(conn, seed=0)
    late = keep["game_id"].isin(ids)
    out = pd.concat([keep[~late], sh[sh["game_id"].isin(ids)]], ignore_index=True)
    rng = np.random.default_rng(5)
    m = out["game_id"].isin(ids).to_numpy()
    out.loc[m, "made"] = (rng.random(m.sum()) < 0.5).astype(np.int8)
    so = dict(c2.execute("SELECT game_id, season FROM team_games").fetchall())
    write_store(out, so)
    q = ",".join("?" * len(ids))
    c2.execute(f"UPDATE player_games SET pts = pts + 7, fg3m = fg3m + 2 WHERE game_id IN ({q})",
               list(ids))
    c2.commit()
    return c2


@pytest.fixture
def restore_store(league):
    yield
    install_shots(league)


def test_no_lookahead_scramble(league, frame, restore_store):
    """L1 for F4: re-randomising every shot (locations, types, results) and box stat on or after
    the cut leaves the F4 features of every game up to the cut identical and changes later ones
    (the xFG model is the one of seasons < S either way)."""
    cut = (SEASONS["2023-24"] + timedelta(days=30)).isoformat()
    other = frame_rows(_scramble(league, cut))
    cols = [c for c in frame.columns if c.startswith("f4_")]
    assert len(cols) >= len(F4.TEAM_COLS) + len(F4.ROW_COLS)
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other[other["game_date"] <= cut].sort_values(key)
    assert len(a) == len(b) and len(a) > 1000
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())
    a2 = frame[frame["game_date"] > cut].sort_values(key)
    b2 = other[other["game_date"] > cut].sort_values(key)
    assert not np.allclose(a2[cols].to_numpy(), b2[cols].to_numpy(), equal_nan=True)


def test_no_lookahead_truncation(league, frame):
    """L5 for F4: physically deleting every game after the cut (team_games / player_games; the
    store's later shots drop out with them) leaves features up to it identical."""
    cut = (SEASONS["2023-24"] + timedelta(days=40)).isoformat()
    c2 = _copy(league)
    for t in ("player_games", "team_games"):
        c2.execute(f"DELETE FROM {t} WHERE game_date > ?", (cut,))
    c2.commit()
    other = frame_rows(c2)
    cols = [c for c in frame.columns if c.startswith("f4_")]
    key = ["game_id", "player_id"]
    a = frame[frame["game_date"] <= cut].sort_values(key)
    b = other.sort_values(key)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


# ------------------------------------------------------------------------- values

def test_era_prior_and_planted_luck(league, frame):
    """Before any shot: NaN; a player without shots sits at the prior with no evidence; the
    planted hot 3-point shooter carries positive luck inside his EB fg3m rate (and the layer's
    monotone constraint can only lower his multiplier for it)."""
    first = frame[frame["day"] == frame["day"].min()]
    assert first[list(F4.QUAL_COLS)].isna().all().all()
    late = frame[(frame["season"] == "2023-24") & (frame["day"] > frame["day"].max() - 30)]
    hot = late[late["player_id"] == HOT]
    rest = late[(late["player_id"] != HOT) & (late["grp"] == hot["grp"].iloc[0])]
    assert hot["f4_lk_fg3m"].mean() > rest["f4_lk_fg3m"].mean() + 0.05
    assert hot["f4_sk3"].mean() > rest["f4_sk3"].mean() + 0.02
    assert hot["f4_l3_s"].mean() > 2.0
    assert FX.monotone_for("fg3m", ["F4"]) == {"f4_lk_fg3m": -1}
    sh = F4.load_history(league, F4.xfg_for_season(league, "2023-24"))
    q = pd.DataFrame({"player_id": [123456789], "day": [int(late["day"].max())],
                      "sidx": [2023], "grp": [0], "opp": ["BBB"]})
    f = F4.player_features(sh, q)
    assert f.loc[0, "f4_ln_fga"] == 0.0
    assert np.allclose(f.loc[0, ["f4_xfg2", "f4_xfg3", "f4_ft", "f4_sk2", "f4_sk3"]].to_numpy(
        float), 0.0)
    assert np.isfinite(f.loc[0, list(F4.OPP_COLS)].to_numpy(float)).all()


def test_registry_columns_exist(frame):
    for s in CM.DK_STATS:
        cols = FX.model_columns(s, ["F4"])
        assert bool(cols) == (s in ("pts", "fg3m"))
        for c in cols:
            assert c in frame.columns, (s, c)
        mono = FX.monotone_for(s, ["P", "F2", "F3", "F4", "F6"])
        assert all(c in FX.model_columns(s, ["P", "F2", "F3", "F4", "F6"]) for c in mono)


def test_live_extras_match_frame(league, frame):
    """The pipeline hook (`LiveExtras.day` with the layer's xFG model as JSON: one day's games,
    opponents from home / away) reproduces the frame's F4 columns."""
    m = F4.xfg_for_season(league, "2023-24")
    ex = FX.LiveExtras(league, "2023-24", ("F4",), xfg=json.loads(m.to_json()))
    day = (SEASONS["2023-24"] + timedelta(days=50)).isoformat()
    d = frame[frame["game_date"] == day]
    assert len(d) > 20
    rows = d[["game_id", "team", "player_id", "grp"]].copy()
    rows["w"] = 1.0
    tg = pd.read_sql_query("SELECT game_id, team, opp FROM team_games WHERE is_home = 1 AND "
                           "game_date = ?", league, params=(day,))
    games = pd.DataFrame({"game_id": tg["game_id"], "home": tg["team"], "away": tg["opp"]})
    tf, pf = ex.day(league, day, games, rows)
    cols = list(F4.TEAM_COLS)
    got = pf.set_index(["game_id", "player_id"]).loc[
        pd.MultiIndex.from_frame(d[["game_id", "player_id"]]), cols]
    np.testing.assert_allclose(got.to_numpy(float), d[cols].to_numpy(float), rtol=1e-9,
                               atol=1e-12)
    # row level (EB shares) from the same columns and the EB state
    X = d[["game_id", "player_id", "n_pts", "r_pts", "n_fg3m", "r_fg3m"]].merge(
        pf, on=["game_id", "player_id"], how="left")
    np.testing.assert_allclose(F4.row_features(X)[list(F4.ROW_COLS)].to_numpy(float),
                               d[list(F4.ROW_COLS)].to_numpy(float), rtol=1e-9, atol=1e-12)
