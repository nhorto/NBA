"""+mf+mr: the minutes-ratio table keyed on the arm's own minutes engine (`minutes_ratio.plans_tag`).

With the minutes fix on, the table's training plans P are the +mf engine's walk-forward plans
(the sims read P from that engine); every other arm keeps the OFF engine's plans, writes no
`plans` key and uses the original cache names, so the OFF-engine +mr table is untouched."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from courtside import config
from courtside.proj import minutes as M
from courtside.proj import minutes_ratio as MR
from courtside.proj import pipeline as P
from courtside.proj import state as ST

from test_proj_minutes import V1_KW, _league, _stints


def _mp(minfix=False, blowout=False, pre=False, season="2022-23"):
    mp = M.default_params(season)
    mp.version = "v1"
    if minfix:
        mp.minfix = {"version": 1, "pplay": "logit"}
    if blowout:
        mp.blowout = {"version": 1}
    if pre:
        mp.gbm_features = list(M.V1_PRESEASON[:1])
    return mp


@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX,
             P.BLOWOUT)
    yield
    P.set_copula(state[2])
    P.set_pplay(state[3])
    P.set_minratio(state[4])
    P.set_preseason(state[5])
    P.set_minfix(state[6])
    P.set_blowout(state[7])
    P.set_context(state[0], state[1])


def test_plans_tag():
    assert MR.plans_tag(None) == ""
    assert MR.plans_tag(_mp()) == ""
    assert MR.plans_tag(_mp(minfix=True)) == "mf"
    assert MR.plans_tag(_mp(minfix=True, blowout=True)) == ""      # +mf+bo keeps OFF plans
    assert MR.plans_tag(_mp(minfix=True, pre=True)) == ""          # +pre+mf keeps OFF plans
    with pytest.raises(ValueError):
        MR._sfx("bo")


def test_table_matches_plans():
    t = {"plan_factor": False, "mult_p": {}}
    assert MR.table_matches(t) and MR.table_matches(t, "")
    assert not MR.table_matches(t, "mf")                           # OFF-engine table under +mf
    tm = dict(t, plans="mf")
    assert MR.table_matches(tm, "mf") and not MR.table_matches(tm)


def test_minutes_params_for_reads_the_arms_engine(data_dir):
    pdir = config.data_dir() / "proj"
    pdir.mkdir(parents=True)
    off, mf = _mp(), _mp(minfix=True)
    off.q0, mf.q0 = 0.011, 0.022
    (pdir / "bundle_2022-23_v1.json").write_text(json.dumps({"minutes": json.loads(off.to_json())}))
    # an _mf-named file without minfix params is skipped; the cache fallback is used
    (pdir / "minutes_params_2022-23_v1_mf.json").write_text(off.to_json())
    MR.cache_dir().mkdir(parents=True)
    (MR.cache_dir() / "minutes_params_2022-23_mf.json").write_text(mf.to_json())
    assert MR.minutes_params_for(None, "2022-23").q0 == 0.011
    assert MR.minutes_params_for(None, "2022-23", plans="mf").q0 == 0.022
    (pdir / "minutes_params_2022-23_v1_mf.json").write_text(mf.to_json())
    assert MR.minutes_params_for(None, "2022-23", plans="mf").minfix
    with pytest.raises(ValueError):
        MR.minutes_params_for(None, "2021-22", plans="mf")         # nothing to read or fit with


def test_planned_rows_cache_names(monkeypatch, data_dir):
    seen = []

    def fake(conn, season, log=None, plans=""):
        seen.append((season, plans))
        return pd.DataFrame({"game_id": ["G"], "team": ["T"], "player_id": [1], "season": season,
                             "planned": [20.0 if not plans else 25.0], "p_play": [0.9]})
    monkeypatch.setattr(MR, "planned_season", fake)
    a = MR.planned_rows(None, ["2021-22"])
    b = MR.planned_rows(None, ["2021-22"], plans="mf")
    assert (MR.cache_dir() / "planned_2021-22.parquet").exists()
    assert (MR.cache_dir() / "planned_2021-22_mf.parquet").exists()
    assert a["planned"].iloc[0] == 20.0 and b["planned"].iloc[0] == 25.0
    MR.planned_rows(None, ["2021-22"], plans="mf")                 # cached: no third call
    assert seen == [("2021-22", ""), ("2021-22", "mf")]
    with pytest.raises(FileNotFoundError):
        MR.planned_rows(None, ["2020-21"], plans="mf", compute=False)


def _rows(n=40_000, seed=5):
    rng = np.random.default_rng(seed)
    mpg = rng.uniform(8, 36, n)
    planned = np.clip(mpg * np.exp(rng.normal(0, 0.2, n)), 2, 42)
    m = np.clip(planned * np.exp(rng.normal(0, 0.2, n)), 0.5, 50)
    d = {"season": rng.choice(["2019-20", "2020-21", "2021-22"], n), "min": m,
         "planned": planned, "mpg": mpg}
    for s in ST.STATS:
        r = rng.uniform(0.02, 0.6, n)
        d[f"r_{s}"] = r
        d[s] = rng.poisson(r * m)
    return pd.DataFrame(d)


def test_fit_state_records_plans_only_for_mf(monkeypatch):
    rows = _rows()
    monkeypatch.setattr(MR, "training_frame", lambda *a, **k: rows)
    monkeypatch.setattr(ST, "fit_usage_bump", lambda *a, **k: {"b": 1})
    sp = {"x": 1}
    off = MR.fit_state(None, "2022-23", sp, None)
    mf = MR.fit_state(None, "2022-23", sp, None, plans="mf")
    assert "plans" not in off["minutes_adj_mr"]                     # OFF-engine JSON unchanged
    assert mf["minutes_adj_mr"].pop("plans") == "mf"
    assert json.dumps(off, sort_keys=True) == json.dumps(mf, sort_keys=True)
    assert sp == {"x": 1}


def test_fit_bundle_uses_the_arms_plans(monkeypatch):
    from courtside.proj import copula as C
    got = {}

    def fake_rows(conn, seasons, log=None, compute=True, plans=""):
        got["rows"] = plans
        return pd.DataFrame()

    def fake_state(conn, season, sp, planned, log=None, plans=""):
        got["state"] = plans
        return {"minutes_adj_mr": {"plans": plans}}
    monkeypatch.setattr(MR, "planned_rows", fake_rows)
    monkeypatch.setattr(MR, "fit_state", fake_state)
    monkeypatch.setattr(C, "fit_copula", lambda *a, **k: "cp")
    for mp, want in ((_mp(), ""), (_mp(minfix=True), "mf"),
                     (_mp(minfix=True, blowout=True), "")):
        base = SimpleNamespace(minutes=mp, state={}, copula=None, context=None)
        b = MR.fit_bundle(None, "2022-23", base, fit_context=None)
        assert got == {"rows": want, "state": want} and b.minutes is mp and b.copula == "cp"


def test_load_bundle_mr_refits_a_table_of_the_other_engine(monkeypatch, restore_switches):
    """A cached `_mf_mr` bundle whose table was keyed on the OFF engine is refit under +mf; a
    matching cache is returned as is; the +mr-alone cache (no `plans` key) is still valid."""
    calls = []
    cached = {}

    class FakeB:
        def __init__(self, minutes, table):
            self.minutes, self.state = minutes, {"minutes_adj_mr": table}

        def to_json(self):
            return "fit"

    def from_json(s):
        return cached["b"]

    def fake_fit(conn, season, base, fit_context=None, log=None):
        calls.append(MR.plans_tag(base.minutes))
        return FakeB(base.minutes, {"plan_factor": False, "plans": "mf"})
    monkeypatch.setattr(P.Bundle, "from_json", staticmethod(from_json))
    monkeypatch.setattr(MR, "fit_bundle", fake_fit)
    P.set_context(False)
    for minfix in (True, False):
        P.set_minfix(minfix)
        P.set_minratio(True)
        mp = _mp(minfix=minfix)
        monkeypatch.setattr(P, "load_bundle", lambda *a, mp=mp, **k: SimpleNamespace(
            minutes=mp, state={}, copula=None, context=None))
        path = P.cache_dir() / f"bundle_2022-23{P._VS}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached")
        old_table = {"plan_factor": False, "mult_p": {}}             # OFF-engine table
        cached["b"] = FakeB(mp, old_table)
        calls.clear()
        b = P._load_bundle_mr(None, "2022-23", None, refit=False, log=None)
        if minfix:
            assert calls == ["mf"] and path.read_text() == "fit"
            cached["b"] = FakeB(mp, dict(old_table, plans="mf"))
            calls.clear()
            assert P._load_bundle_mr(None, "2022-23", None, refit=False, log=None) is cached["b"]
            assert calls == []
        else:
            assert calls == [] and b is cached["b"] and path.read_text() == "cached"


def test_planned_season_mf_engine_on_a_league(monkeypatch, data_dir):
    """planned_season with plans "mf" fits and caches the minutes-fix engine of the season (fit on
    the seasons before it) and its plans differ from the OFF engine's on the same panel."""
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    real_fit = M.fit
    kws = []

    def fake_fit(h_, season, **kw):
        kws.append(kw.get("v1_kw"))
        return real_fit(h_, season, window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                        v1_kw=dict(V1_KW, **(kw.get("v1_kw") or {})))
    monkeypatch.setattr(M, "load_history", lambda conn, since=None: h)
    monkeypatch.setattr(M, "fit", fake_fit)
    off = MR.planned_season(None, "2022-23")
    mf = MR.planned_season(None, "2022-23", plans="mf")
    assert kws == [None, {"minfix": True}]
    assert (MR.cache_dir() / "minutes_params_2022-23.json").exists()
    mpf = M.MinutesParams.from_json(
        (MR.cache_dir() / "minutes_params_2022-23_mf.json").read_text())
    assert MR.plans_tag(mpf) == "mf"
    k = ["game_id", "team", "player_id"]
    assert off[k].equals(mf[k])                                    # same rows, other engine
    assert not np.allclose(off["p_play"], mf["p_play"])
    assert len(off) > 100
