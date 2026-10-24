"""G4 absence scale (COURTSIDE_ABSENCE_SCALE): OFF is bit-identical and never enters the new
code; ON pulls the injury-report lineup strengths toward the full roster by s, for the priced
lineups and the per-player injury effects alike, and tags both models' rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synth import day_date, synth_conn

from courtside import predict
from courtside.model import absence_scale as asc, line_news

DAY = day_date("2024-25", 22)
INS = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
       "VALUES (?,?,?,?,?,?)")


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)
    monkeypatch.setattr(line_news, "ON", False)
    monkeypatch.setattr(asc, "ON", False)
    monkeypatch.setattr(asc, "S", None)


def _world():
    """Past day with one official report: the home star of the first game Out, the away side
    of that game listed with nobody missing."""
    conn = synth_conn()
    base = predict.predict_day(conn, DAY, store=False)
    g = base[0]
    at = f"{DAY}T15:00:00Z"
    conn.execute(INS, (at, "nba_official", g["home"], f"{g['home']} Player 0", "Out", "knee"))
    conn.execute(INS, (at, "nba_official", g["away"], f"{g['away']} Player 0", "Available", "x"))
    conn.commit()
    return conn, g["game_id"]


def _by_game(rows):
    return {r["game_id"]: r for r in rows}


def _numbers(rows):
    keys = ("home_win_prob", "home_margin", "total")
    return [(r["game_id"], *(r[k] for k in keys), *(r["opinion"][k] for k in keys))
            for r in rows]


def _strip(rows):
    def one(r):
        out = {k: v for k, v in r.items() if k not in ("created_at", "opinion")}
        out["inputs"] = {k: v for k, v in r["inputs"].items() if k != "fit"}
        if r.get("opinion"):
            out["opinion"] = one(r["opinion"])
        return out
    return [one(r) for r in rows]


def _on(monkeypatch, s):
    monkeypatch.setattr(asc, "ON", True)
    monkeypatch.setattr(asc, "S", s)


def _effect(row, gid):
    eff = [e for e in row["inputs"]["model_effects"] if e["game_id"] == gid
           and e["side"] == "home"]
    assert len(eff) == 1
    return eff[0]["effect_margin"]


# --- plumbing ----------------------------------------------------------------------------------

def test_default_is_off_and_env_parsing():
    assert asc.DEFAULT is None
    for v, want in (("", None), ("off", None), ("0ff", None), ("false", None), (None, None),
                    ("on", asc.SCALE), ("true", asc.SCALE), ("0.5", 0.5), ("0", 0.0),
                    ("1", 1.0), ("9", asc.S_RANGE[1]), ("-1", asc.S_RANGE[0]), ("nan", None)):
        assert asc.parse(v) == want, v
    assert asc.tagged("gbm_opinion@v1-abc") == "gbm_opinion@v1-abc+as"
    assert asc.tagged(None) is None


def test_scale_strength():
    assert asc.scale_strength(5.0, 3.0, 1.0) == 3.0
    assert asc.scale_strength(5.0, 3.0, 0.0) == 5.0
    assert asc.scale_strength(5.0, 3.0, 0.5) == 4.0
    assert asc.scale_strength(np.nan, 3.0, 0.5) == 3.0


def test_set_absence_scale():
    asc.set_absence_scale(0.4)
    assert asc.ON and asc.S == 0.4
    asc.set_absence_scale(None)
    assert not asc.ON and asc.S is None


# --- OFF: bit-identical -------------------------------------------------------------------------

def test_off_never_enters_the_new_code_and_is_deterministic(monkeypatch):
    conn, _ = _world()
    ref = predict.predict_day(conn, DAY, store=False)

    def boom(*a, **k):
        raise AssertionError("absence-scale code ran with the switch OFF")
    for name in ("wrap", "tagged", "scale_strength", "roster_strengths"):
        monkeypatch.setattr(asc, name, boom)
    off = predict.predict_day(conn, DAY, store=False)
    assert _strip(off) == _strip(ref)
    for r in off:
        for x in (r, r["opinion"]):
            assert "absence_scale" not in x["inputs"]
            assert x["model_version"] is None or not x["model_version"].endswith(asc.TAG)


def test_on_scale_one_prices_exactly_like_off(monkeypatch):
    conn, _ = _world()
    off = predict.predict_day(conn, DAY, store=False)
    _on(monkeypatch, 1.0)
    on = predict.predict_day(conn, DAY, store=False)
    assert _numbers(on) == _numbers(off)
    for a, b in zip(on, off):
        assert a["inputs"]["model_effects"] == b["inputs"]["model_effects"]
        assert a["inputs"]["absence_scale"] == 1.0
        assert a["opinion"]["inputs"]["absence_scale"] == 1.0


# --- ON ----------------------------------------------------------------------------------------

def test_on_shrinks_the_absence_and_its_effect(monkeypatch):
    conn, gid = _world()
    off = _by_game(predict.predict_day(conn, DAY, store=False))
    res = {}
    for s in (0.0, 0.5, 1.5):
        _on(monkeypatch, s)
        res[s] = _by_game(predict.predict_day(conn, DAY, store=False))
    lu = {s: res[s][gid]["inputs"]["availability"]["home"]["lineup_strength"] for s in res}
    lu[1.0] = off[gid]["inputs"]["availability"]["home"]["lineup_strength"]
    assert lu[0.0] != lu[1.0]                                  # the Out changes the lineup
    assert lu[0.5] == pytest.approx((lu[0.0] + lu[1.0]) / 2, abs=1e-9)
    assert lu[1.5] == pytest.approx(lu[1.0] - (lu[0.0] - lu[1.0]) / 2, abs=1e-9)
    # opinion margin monotone in s (direction = the sign of the absence); the per-player
    # effect shrinks with s and vanishes at s = 0
    sgn = np.sign(lu[1.0] - lu[0.0])
    m = {s: sgn * res[s][gid]["opinion"]["home_margin"] for s in res}
    m[1.0] = sgn * off[gid]["opinion"]["home_margin"]
    assert m[0.0] <= m[0.5] <= m[1.0] <= m[1.5]
    assert m[0.0] < m[1.5]
    e1 = _effect(off[gid], gid)
    assert e1 != 0
    assert _effect(res[0.0][gid], gid) == 0.0
    assert abs(_effect(res[0.5][gid], gid)) <= abs(e1) <= abs(_effect(res[1.5][gid], gid))
    # the away side was listed with nobody missing: unchanged lineup
    for s in res:
        assert res[s][gid]["inputs"]["availability"]["away"]["lineup_strength"] == \
            off[gid]["inputs"]["availability"]["away"]["lineup_strength"]
    # rows are tagged, the market and opinion alike
    for r in res[0.5].values():
        for x in (r, r["opinion"]):
            assert x["inputs"]["absence_scale"] == 0.5
            v = x["model_version"]
            assert v is None or v.endswith(asc.TAG)


def test_on_days_without_a_report_are_unchanged(monkeypatch):
    conn = synth_conn()
    off = predict.predict_day(conn, DAY, store=False)
    _on(monkeypatch, 0.3)
    on = predict.predict_day(conn, DAY, store=False)
    assert _numbers(on) == _numbers(off)


def test_scaled_state_proxy_forwards_and_scales():
    class St:
        names = ["a"]

        def expected_lineup(self, team, game=None, availability=None, playoff=False):
            df = pd.DataFrame({"x": [1]})
            df.attrs["strength"] = 2.0 if availability else 6.0
            return df
    F = pd.DataFrame({"ps_home": [2.0], "abs_loss_home": [4.0], "ps_away": [1.0],
                      "abs_loss_away": [0.0]})
    rows = pd.DataFrame({"home": ["H"], "away": ["A"]})
    p = asc.wrap(St(), F, rows, 0.25)
    assert p.names == ["a"]
    assert p.expected_lineup("H", None, {1: 0.0}).attrs["strength"] == pytest.approx(5.0)
    assert p.expected_lineup("Z", None, {1: 0.0}).attrs["strength"] == 2.0   # unknown team
