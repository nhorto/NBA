"""Retraining: offline-gate math on synthetic per-game losses, shadow promotion rules, params
merging, and a tiny end-to-end challenger run on a synthetic league. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_model_gbm import SMALL, TINY_GRID, league  # noqa: F401  (league is a fixture)

from courtside import db
from courtside.mlops import registry, retrain
from courtside.model import gbm


@pytest.fixture(autouse=True)
def no_mlflow(monkeypatch):
    monkeypatch.setenv("COURTSIDE_MLFLOW", "0")


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "r.db")


# --- offline gate -----------------------------------------------------------------------------

def _preds(p_champ, p_chal, y, seasons):
    ids = [f"g{i}" for i in range(len(y))]
    a = pd.DataFrame({"game_id": ids, "season": seasons, "y": y, "p": p_champ})
    b = pd.DataFrame({"game_id": ids, "season": seasons, "y": y, "p": p_chal})
    return a, b


def _synthetic(n=4000, shift=0.0, recent_shift=None, seed=0):
    """Champion p = truth + noise; challenger = less noisy (shift > 0 better)."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < truth).astype(int)
    seasons = np.where(np.arange(n) < n * 0.8, "2023-24", "2024-25")
    noise = rng.normal(0, 0.12, n)
    champ = np.clip(truth + noise, 0.02, 0.98)
    k = np.full(n, shift)
    if recent_shift is not None:
        k[seasons == "2024-25"] = recent_shift
    chal = np.clip(truth + noise * (1 - k), 0.02, 0.98)
    return _preds(champ, chal, y, seasons)


def test_gate_math_matches_a_hand_computed_paired_se():
    y = np.array([1, 0, 1, 1, 0, 1])
    pc = np.array([0.6, 0.4, 0.7, 0.5, 0.5, 0.6])
    pn = np.array([0.7, 0.3, 0.8, 0.6, 0.4, 0.7])
    a, b = _preds(pc, pn, y, ["2023-24"] * 3 + ["2024-25"] * 3)
    g = retrain.offline_gate(a, b)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    d = ll(pn) - ll(pc)
    assert g["overall"]["diff"] == pytest.approx(d.mean())
    assert g["overall"]["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(6))
    assert g["recent"]["season"] == "2024-25" and g["recent"]["n"] == 3
    assert g["recent"]["diff"] == pytest.approx(d[3:].mean())
    assert g["n"] == 6


def test_gate_passes_a_clearly_better_challenger():
    a, b = _synthetic(shift=0.6)
    g = retrain.offline_gate(a, b)
    assert g["passed"] and not g["reasons"]
    assert g["overall"]["diff"] < -2 * g["overall"]["se"]


def test_gate_rejects_noise_level_and_worse_challengers():
    a, b = _synthetic(shift=0.0)                       # identical predictions: diff 0, not < -2SE
    g = retrain.offline_gate(a, b)
    assert not g["passed"] and "overall" in g["reasons"][0]
    a, b = _synthetic(shift=-0.5)
    assert not retrain.offline_gate(a, b)["passed"]


def test_gate_rejects_a_challenger_worse_on_the_recent_season():
    a, b = _synthetic(shift=0.6, recent_shift=-1.5)
    g = retrain.offline_gate(a, b)
    assert any("worse on 2024-25" in r for r in g["reasons"])
    assert not g["passed"]


def test_gate_rejects_worse_calibration():
    a, b = _synthetic(shift=0.6)
    b = b.copy()
    b["p"] = np.clip(b["p"] + 0.1, 0.01, 0.99)            # systematically too high
    g = retrain.offline_gate(a, b)
    assert g["ece"]["diff"] > 0.005 and any("ECE" in r for r in g["reasons"])


def test_gate_only_uses_common_games():
    a, b = _synthetic(n=200, shift=0.6)
    g = retrain.offline_gate(a, b.iloc[:150])
    assert g["n"] == 150


# --- online gate ------------------------------------------------------------------------------

@pytest.mark.parametrize("n, diff, crit, action", [
    (50, -0.1, 0, "wait"),          # too few games
    (100, -0.001, 0, "promote"),
    (100, 0.0, 0, "promote"),       # tie: <= 0 promotes
    (150, -0.01, 1, "wait"),        # better but a critical alert is open
    (299, 0.002, 0, "wait"),        # worse, still collecting
    (300, 0.002, 0, "reject"),
    (400, -0.002, 2, "wait"),
])
def test_shadow_decision_rules(n, diff, crit, action):
    assert retrain.shadow_decision(n, diff, crit)[0] == action


def _shadow_world(conn, n_games: int, shadow_better: bool, seed=0):
    """Champion + shadow predictions for `n_games` final games."""
    registry.bootstrap(conn, log_mlflow=False, log_fn=lambda s: None)
    p = registry.production_params("gbm_opinion")
    p["fixed"]["margin"][1] = 7
    sid = registry.register(conn, "gbm_opinion", p, registry.feature_list("gbm_opinion"),
                            status="shadow")
    champ = registry.get_champion(conn, "gbm_opinion")["version_id"]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_games):
        gid, day = f"00224{i:05d}", f"2024-11-{1 + i % 28:02d}"
        truth = rng.uniform(0.25, 0.75)
        y = rng.random() < truth
        conn.execute("INSERT INTO games (game_id, nba_day, status, home_score, away_score, "
                     "tipoff_utc) VALUES (?,?,?,?,?,?)",
                     (gid, day, "final", 110 if y else 100, 100 if y else 110, f"{day}T23:00:00Z"))
        good, bad = truth, float(np.clip(truth + rng.normal(0, 0.2), 0.05, 0.95))
        ps, pc = (good, bad) if shadow_better else (bad, good)
        for model, ver, role, pr in (("gbm_opinion", champ, "champion", pc),
                                     (sid, sid, "shadow", ps)):
            rows.append((gid, model, f"{day}T12:00:00Z", pr, "{}", day, "BOS", "NYK", ver, role))
    conn.executemany("INSERT INTO predictions (game_id, model, created_at, home_win_prob, "
                     "inputs_json, nba_day, home, away, model_version, role) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return sid, champ


def test_evaluate_shadows_promotes_a_better_shadow(conn):
    sid, champ = _shadow_world(conn, 120, shadow_better=True)
    (r,) = retrain.evaluate_shadows(conn, log_fn=lambda s: None)
    assert r["n"] == 120 and r["diff"] < 0 and r["action"] == "promote" and r["applied"]
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == sid
    assert registry.get_version(conn, champ)["status"] == "retired"


def test_evaluate_shadows_waits_while_a_critical_alert_is_open(conn):
    sid, _ = _shadow_world(conn, 120, shadow_better=True)
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message) "
                 "VALUES ('perf', 't', 't', 'critical', 'performance', 'bad')")
    conn.commit()
    (r,) = retrain.evaluate_shadows(conn, log_fn=lambda s: None)
    assert r["action"] == "wait" and registry.get_version(conn, sid)["status"] == "shadow"


def test_evaluate_shadows_rejects_a_worse_shadow_after_300(conn):
    sid, champ = _shadow_world(conn, 300, shadow_better=False)
    (r,) = retrain.evaluate_shadows(conn, dry_run=True, log_fn=lambda s: None)
    assert r["action"] == "reject" and not r["applied"]
    assert registry.get_version(conn, sid)["status"] == "shadow"          # dry run
    (r,) = retrain.evaluate_shadows(conn, log_fn=lambda s: None)
    assert r["applied"] and registry.get_version(conn, sid)["status"] == "retired"
    ev = registry.events(conn, version_id=sid, limit=1)[0]
    assert ev["action"] == "reject" and "online gate" in ev["reason"]
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == champ


# --- params -----------------------------------------------------------------------------------

def test_challenger_params_merge_over_the_champion():
    champ = registry.production_params("gbm_opinion")
    p, mode = retrain.challenger_params(champ, {"margin": [{"num_leaves": 31}, 60]})
    assert mode == "fixed" and p["fixed"]["margin"][1] == 60
    assert p["fixed"]["margin"][0]["num_leaves"] == 31
    md = champ["fixed"]["margin"][0]["min_data_in_leaf"]
    assert p["fixed"]["margin"][0]["min_data_in_leaf"] == md
    assert p["fixed"]["total"] == champ["fixed"]["total"]
    p, mode = retrain.challenger_params(champ, {"search_grid": [{"num_leaves": 3}]})
    assert mode == "search" and p["search_grid"][0]["num_leaves"] == 3
    with pytest.raises(ValueError):
        retrain.challenger_params(champ, {"bogus": 1})
    grid = retrain.search_grid("small")
    assert all(c.to_dict() in grid for c in gbm.GRID) and len(grid) == 18


# --- end to end (tiny synthetic league) ------------------------------------------------------

@pytest.fixture
def tiny(monkeypatch, conn, league):
    real = gbm.make_caches
    monkeypatch.setattr(gbm, "make_caches",
                        lambda players, *a, **k: real(players, params=SMALL, disk_dir=None))
    champ = registry.production_params("gbm_opinion")
    champ["search_grid"] = [TINY_GRID[0].to_dict()]
    champ["fixed"]["margin"] = [TINY_GRID[0].to_dict(), 10]
    vid = registry.register(conn, "gbm_opinion", champ, registry.feature_list("gbm_opinion"),
                            status="champion")
    kw = dict(data=league, seasons_from="2003-04", seasons_to="2004-05",
              model_kw={"burn_in": 1, "n_inner": 2}, threads=1, log_fn=lambda s: None)
    return vid, kw


def test_run_challenger_search_mode_end_to_end(conn, tiny):
    champ_id, kw = tiny
    res = retrain.run_challenger(conn, "gbm_opinion",
                                 params={"search_grid": [c.to_dict() for c in TINY_GRID]}, **kw)
    g = res["gate"]
    assert res["mode"] == "search" and g["n"] > 0 and g["recent"]["season"] == "2004-05"
    v = registry.get_version(conn, res["version_id"])
    assert v["status"] == ("shadow" if g["passed"] else "retired")
    assert res["status"] == ("shadow" if g["passed"] else "rejected")
    actions = [e["action"] for e in registry.events(conn, version_id=res["version_id"])]
    assert actions[-1] == "register" and actions[0] == ("shadow" if g["passed"] else "reject")
    assert v["benchmark"]["gate"]["n"] == g["n"] and "summary" in v["benchmark"]
    # production params were re-tuned by the search (margin), total carried over
    cfg, n = v["params"]["fixed"]["margin"]
    assert gbm.GBMConfig(**cfg) in TINY_GRID and n >= 0
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == champ_id
    assert res["timing"]["total_s"] >= 0


def test_run_challenger_fixed_mode_and_no_change(conn, tiny):
    champ_id, kw = tiny
    same = retrain.run_challenger(conn, "gbm_opinion",
                                  params={"margin": [TINY_GRID[0].to_dict(), 10]}, **kw)
    assert same["status"] == "no_change" and same["gate"]["overall"]["diff"] == 0.0
    assert not same["gate"]["passed"]
    res = retrain.run_challenger(conn, "gbm_opinion",
                                 params={"margin": [TINY_GRID[1].to_dict(), 25]}, **kw)
    assert res["mode"] == "fixed" and res["status"] in ("shadow", "rejected")
    assert registry.get_version(conn, res["version_id"])["params"]["fixed"]["margin"][1] == 25
