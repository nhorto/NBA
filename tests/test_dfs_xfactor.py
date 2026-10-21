"""X factor (big-night chance + low popularity) and the harness's optional tail output."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from courtside import db
from courtside.proj import backtest as B
from courtside.proj.dfs import portfolio as PF
from courtside.proj.dfs import xfactor as XF
from courtside.web.app import create_app

from test_web_dfs import DAY, NOW, SLATE, players, populate, text_of  # noqa: E402

S = 4000


# ----------------------------------------------------------------------------- pure helpers

def test_raw_chances_and_mark():
    fp = np.zeros((10, 3), np.float16)
    fp[:3, 0] = 30.0            # 3 of 10 sims at 30 >= 6 x 4.5
    fp[:, 1] = 100.0
    p = XF.raw_chances(fp, [4500, 9000, np.nan])
    assert p[0] == pytest.approx(0.3) and p[1] == pytest.approx(1.0) and np.isnan(p[2])
    assert XF.big_mark(4500) == pytest.approx(27.0)


def test_calibrate_maps_per_tier(monkeypatch):
    monkeypatch.setattr(XF, "STRETCH", {"bench": (-0.5, 1.0), "role": (0.0, 1.0)})
    out = XF.calibrate([0.2, 0.2, 0.2, np.nan], ["bench", "role", None, "bench"])
    assert out[0] < 0.2 and out[1] == pytest.approx(0.2) and out[2] == 0.2 and np.isnan(out[3])


def test_chance_text_bands():
    assert XF.chance_text(0.2) == "1 in 5"
    assert XF.chance_text(0.13) == "1 in 8"
    assert XF.chance_text(0.001) == "under 1 in 50"
    assert XF.chance_text(0.7) == "better than 1 in 2"
    assert XF.chance_text(None) == "" and XF.chance_text(float("nan")) == ""


def test_is_xfactor_rules():
    ok = dict(chance=XF.P_MIN + 0.01, band="Low", p_play=0.95, tier="bench")
    assert XF.is_xfactor(**ok)
    assert not XF.is_xfactor(**{**ok, "chance": XF.P_MIN - 0.01})
    assert not XF.is_xfactor(**{**ok, "band": "Med"})          # Duncan Robinson's case
    assert not XF.is_xfactor(**{**ok, "p_play": 0.5})
    assert not XF.is_xfactor(**{**ok, "tier": "star"})
    assert not XF.is_xfactor(**{**ok, "band": None})


def test_lineup_count():
    ls = [[{"player_id": "1"}, {"player_id": "2"}], [{"player_id": "3"}], [{"player_id": 2}]]
    assert XF.lineup_count(ls, {"2"}) == 2
    assert XF.lineup_count(ls, set()) == 0


# ----------------------------------------------------------------------------- harness option

def test_tails_option_keeps_the_old_cache_key(monkeypatch):
    seen = []
    monkeypatch.setattr(B, "_hash", lambda obj: seen.append(obj) or "h")
    monkeypatch.setattr(B, "code_fingerprint", lambda: "c")
    B.config_key(B.Settings(), "{}", None)
    assert "tails" not in seen[-1]["s"] and "pplay" not in seen[-1]["s"]
    B.config_key(B.Settings(tails=True), "{}", None)
    assert seen[-1]["s"]["tails"] is True


def test_sim_tails():
    fp = np.zeros((100, 2))
    fp[:, 0] = np.arange(100)               # 0..99
    fp[:, 1] = 50.0
    res = SimpleNamespace(sims=[SimpleNamespace(fp=fp, player_id=np.array([7, 8]))])
    t = B.sim_tails(res, {7: 10_000.0})
    assert t[7]["p_ge_5x"] == pytest.approx(0.50)      # >= 50
    assert t[7]["p_ge_7x"] == pytest.approx(0.30)      # >= 70
    assert t[7]["q99"] == pytest.approx(np.quantile(np.arange(100), 0.99))
    assert "p_ge_5x" not in t[8] and t[8]["q95"] == pytest.approx(50.0)


# ----------------------------------------------------------------------------- live run + page

HOT, MED, LOW_P = 1, 2, 5        # player indexes: X factor; Medium popularity; low chance


def _sims(n_players: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    fp = rng.normal(5.0, 2.0, (S, n_players)).clip(0)
    ps = players()
    for i in (HOT, MED):                         # half the sims at a big night (>= 6 x salary)
        fp[: S // 2, i] = 6.0 * ps[i]["salary"] / 1000.0 + 1.0
    fp[: S // 100, LOW_P] = 6.0 * ps[LOW_P]["salary"] / 1000.0 + 1.0     # 1 %
    return fp.astype(np.float16)


def _add_xf_inputs(d, conn) -> None:
    ps = players()
    path = d / "sims.npz"
    np.savez_compressed(path, fp=_sims(len(ps)), player_id=np.array([500 + p["i"] for p in ps]),
                        dk_id=np.array([p["id"] for p in ps]))
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id='run1'", (str(path),))
    for p in ps:
        conn.execute("UPDATE player_projections SET role=?, inputs_json=? WHERE run_id='run1' "
                     "AND player_id=?", ("star" if p["i"] >= 25 else "bench",
                                         json.dumps({"dk_id": p["id"]}), 500 + p["i"]))
        band = "Med" if p["i"] == MED else "Low"
        conn.execute("INSERT INTO dfs_ownership_proxy (run_id, player_id, own, band, version) "
                     "VALUES ('run1', ?, ?, ?, 'test')", (int(p["id"]), 15.0 if band == "Med"
                                                           else 2.0, band))
    conn.commit()


@pytest.fixture
def xf_db(data_dir, monkeypatch):
    monkeypatch.setattr(XF, "STRETCH", {})          # raw chances in these tests
    monkeypatch.setattr(XF, "VALUE_TOP_EXCLUDE", 0)  # tested on its own below
    XF._CACHE.clear()
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    populate(conn, model=True)
    _add_xf_inputs(data_dir, conn)
    yield data_dir, conn
    conn.close()


def test_run_table(xf_db):
    _, conn = xf_db
    t = XF.run_table(conn, "run1")
    ps = players()
    hot = t[ps[HOT]["id"]]
    assert hot["chance"] == pytest.approx(0.5) and hot["text"] == "1 in 2" and hot["x"]
    assert hot["mark"] == pytest.approx(6.0 * ps[HOT]["salary"] / 1000.0)
    assert not t[ps[MED]["id"]]["x"]                  # Medium popularity: no tag
    assert t[ps[LOW_P]["id"]]["chance"] == pytest.approx(0.01)
    assert not t[ps[LOW_P]["id"]]["x"]
    assert not t[ps[3]["id"]]["x"]                    # P(play) 0.8 < 0.85
    assert XF.xfactor_ids(t) == {ps[HOT]["id"]}
    assert XF.run_table(conn, None) == {} and XF.run_table(conn, "nope") == {}


def test_value_ranks():
    r = XF.value_ranks(["a", "b", "c", "d"], [30.0, 20.0, 40.0, 50.0],
                       [5000, 2000, 8000, 5000], [0.9, 0.9, 0.9, 0.3])
    assert r == {"b": 1, "a": 2, "c": 3}             # d: P(play) < 0.5, not ranked


def test_top_value_plays_are_not_tagged(xf_db, monkeypatch):
    _, conn = xf_db
    ps = players()
    t = XF.run_table(conn, "run1")
    hot = t[ps[HOT]["id"]]
    assert hot["x"] and hot["value_rank"] is not None
    monkeypatch.setattr(XF, "VALUE_TOP_EXCLUDE", hot["value_rank"])
    XF._CACHE.clear()
    assert not XF.run_table(conn, "run1")[ps[HOT]["id"]]["x"]


def test_run_table_without_sims(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    populate(conn, model=True)
    assert XF.run_table(conn, "run1") == {}
    conn.close()


def test_pool_shows_big_night_and_tag(xf_db):
    d, _ = xf_db
    app = create_app(d)
    app.state.dfs_now = lambda: NOW
    from fastapi.testclient import TestClient
    c = TestClient(app)
    html = c.get(f"/dfs/{DAY}/{SLATE}?all=1").text
    t = text_of(html)
    assert "Big-night chance" in t and "1 in 2" in t
    assert t.count("X factor") >= 2                   # the tag (+ the footnote)
    assert "not advice" in t and "proxy" in t
    assert "1 on this slate" in t
    lower = t.lower()
    for w in ("edge", "lean", "sleeper", "smash", "boom", "leverage", "fade", "chalk", "lock ",
              "punt", "value play"):
        assert w not in lower, w
    # sortable by the new column, X factor player before the 1 % one
    html = c.get(f"/dfs/{DAY}/{SLATE}?sort=big&all=1").text
    assert html.index(players()[HOT]["name"]) < html.index(players()[LOW_P]["name"])


def test_pool_without_sims_has_no_column(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    populate(conn, model=True)
    conn.close()
    app = create_app(data_dir)
    app.state.dfs_now = lambda: NOW
    from fastapi.testclient import TestClient
    t = text_of(TestClient(app).get(f"/dfs/{DAY}/{SLATE}").text)
    assert "Big-night chance" not in t and "X factor" not in t


def test_portfolio_xfactor_summary(xf_db):
    _, conn = xf_db
    ps = players()
    hot = ps[HOT]
    slots_a = [{"player_id": hot["id"], "name": hot["name"]}, {"player_id": ps[9]["id"]}]
    slots_b = [{"player_id": ps[9]["id"]}, {"player_id": ps[10]["id"]}]
    pf = SimpleNamespace(run_id="run1",
                         slots=lambda: [slots_a, slots_b])
    out = PF.xfactor_summary(conn, pf)
    assert out["lineups_with"] == 1 and out["entries"] == 2 and out["n_players"] == 1
    assert out["players_used"] == [hot["name"]]
    assert PF.xfactor_summary(conn, SimpleNamespace(run_id=None, slots=lambda: [])) is None


def test_cap_keeps_the_highest_chances():
    t = {str(i): {"chance": 0.3 + 0.01 * i, "own": 1.0, "x": True} for i in range(8)}
    t["9"] = {"chance": 0.9, "own": 1.0, "x": False}
    XF.apply_cap(t, cap=3)
    assert XF.xfactor_ids(t) == {"7", "6", "5"}
