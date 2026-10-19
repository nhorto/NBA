"""Prediction lineage: champion/shadow tagging, shadow params flow into the fit, shadows never
reach the Brief or scorecard, migration 6 never clobbers across versions. Synthetic league."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import pytest
from synth import day_date, synth_conn

from courtside import db, predict, production
from courtside.brief import assemble
from courtside.mlops import registry, retrain
from courtside.model import gbm

DAY = day_date("2024-25", 22)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)
    monkeypatch.setenv("COURTSIDE_MLFLOW", "0")


LONG = {"2020-21": 40, "2021-22": 40, "2022-23": 40, "2023-24": 40, "2024-25": 40}


def lineage_conn(seasons=None) -> sqlite3.Connection:
    """synth_conn (migrations 1-4) brought up to the current schema. `LONG` seasons leave
    enough history for inner folds (non-zero tree counts)."""
    conn = synth_conn(seasons)
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
    conn.executemany("INSERT INTO schema_version VALUES (?, 'x')", [(1,), (2,), (3,), (4,)])
    conn.commit()
    db.migrate(conn)
    return conn


def _shadow_params(n_trees: int = 3) -> dict:
    p = registry.production_params(gbm.PRODUCTION_OPINION)
    p["fixed"]["margin"] = [gbm.GBMConfig(num_leaves=4, min_data_in_leaf=20).to_dict(), n_trees]
    return p


def _setup_registry(conn, shadow: bool = True) -> tuple[dict, str | None]:
    registry.bootstrap(conn, log_mlflow=False, log_fn=lambda s: None)
    champs = {n: registry.get_champion(conn, n)["version_id"] for n in predict.PRODUCTION_MODELS}
    sid = None
    if shadow:
        sid = registry.register(conn, gbm.PRODUCTION_OPINION, _shadow_params(),
                                registry.feature_list(gbm.PRODUCTION_OPINION), status="shadow")
    return champs, sid


def test_predictions_are_tagged_with_version_and_role(monkeypatch):
    conn = lineage_conn(LONG)
    champs, sid = _setup_registry(conn)
    seen = []
    real = production.fit_bundle
    monkeypatch.setattr(production, "fit_bundle",
                        lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    rows = predict.predict_day(conn, DAY)
    assert len(rows) == 3
    for r in rows:
        assert r["model_version"] == champs["gbm_market"] and r["role"] == "champion"
        assert r["opinion"]["model_version"] == champs["gbm_opinion"]
        (s,) = r["shadows"]
        assert s["model"] == s["model_version"] == sid and s["role"] == "shadow"
        assert s["inputs"]["model_version"] == sid
    stored = conn.execute("SELECT model, model_version, role, COUNT(*) FROM predictions "
                          "GROUP BY 1, 2, 3 ORDER BY 1").fetchall()
    assert [tuple(x) for x in stored] == [
        ("gbm_market", champs["gbm_market"], "champion", 3),
        ("gbm_opinion", champs["gbm_opinion"], "champion", 3),
        (sid, sid, "shadow", 3)]
    # the shadow's params really were used for its fit (4 leaves / 3 trees, not FROZEN's)
    b = seen[-1]
    assert b.scorers[sid].heads_["margin"][2].num_leaves == 4
    assert b.scorers[sid].heads_["margin"][1] == 3
    cfg, n = gbm.FROZEN["gbm_opinion"]["margin"]
    assert b.scorers["gbm_opinion"].heads_["margin"][2].num_leaves == cfg.num_leaves
    assert b.scorers["gbm_opinion"].heads_["margin"][1] == n
    assert b.versions[sid] == {"base": "gbm_opinion", "version_id": sid, "role": "shadow"}
    by = {(r["opinion"]["game_id"]): (r["opinion"]["home_win_prob"],
                                       r["shadows"][0]["home_win_prob"]) for r in rows}
    assert any(a != c for a, c in by.values())


def test_full_feature_vector_is_stored_for_drift():
    conn = lineage_conn()
    rows = predict.predict_day(conn, DAY)
    f = json.loads(conn.execute("SELECT inputs_json FROM predictions WHERE model='gbm_market' "
                                "LIMIT 1").fetchone()[0])["features"]
    for k in ("strength_diff", "lineup_strength_home", "lineup_strength_away",
              "lineup_strength_diff", "absences", "abs_loss_home", "abs_loss_away",
              "out_home", "questionable_away", "elo_diff", "rest_diff", "b2b_home", "b2b_total",
              "market_spread", "market_total", "pred_home_prob", "pred_total"):
        assert k in f, k
    assert f["pred_home_prob"] in {r["home_win_prob"] for r in rows}
    assert abs(f["strength_diff"] - f["lineup_strength_diff"]) < 1e-3


def test_registry_champion_params_flow_into_production(monkeypatch):
    conn = lineage_conn()
    champs, _ = _setup_registry(conn, shadow=False)
    p = _shadow_params(5)
    new = registry.register(conn, "gbm_opinion", p, registry.feature_list("gbm_opinion"))
    registry.promote(conn, new, "test")
    seen = []
    real = production.fit_bundle
    monkeypatch.setattr(production, "fit_bundle",
                        lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    rows = predict.predict_day(conn, DAY, store=False)
    assert rows[0]["opinion"]["model_version"] == new
    assert seen[-1].scorers["gbm_opinion"].heads_["margin"][2].num_leaves == 4


def test_cache_key_includes_versions(monkeypatch):
    conn = lineage_conn()
    k0 = production.cache_key(conn, DAY, ["a"], predict.PRODUCTION_MODELS,
                              production.resolve_specs(conn, predict.PRODUCTION_MODELS))
    champs, sid = _setup_registry(conn)
    specs = production.resolve_specs(conn, predict.PRODUCTION_MODELS)
    assert set(specs) == {"gbm_market", "gbm_opinion", sid}
    k1 = production.cache_key(conn, DAY, ["a"], tuple(specs), specs)
    assert k1 != k0 and k1["versions"]["gbm_opinion"] == champs["gbm_opinion"]
    assert k1["versions"][sid] == sid
    registry.set_status(conn, sid, "retired", "done")
    specs2 = production.resolve_specs(conn, predict.PRODUCTION_MODELS)
    k2 = production.cache_key(conn, DAY, ["a"], tuple(specs2), specs2)
    assert k2 != k1 and sid not in k2["versions"] and k2["frozen"] != k0["frozen"]


def test_broken_registry_falls_back_to_frozen(monkeypatch):
    conn = synth_conn()                                       # no registry tables at all
    specs = production.resolve_specs(conn, predict.PRODUCTION_MODELS)
    assert all(s.version_id is None and s.params() == gbm.FROZEN[s.base] for s in specs.values())
    conn2 = lineage_conn()
    _setup_registry(conn2, shadow=False)
    conn2.execute("UPDATE model_versions SET params_json='{\"oops\": 1}'")
    conn2.commit()
    specs = production.resolve_specs(conn2, predict.PRODUCTION_MODELS)
    assert all(s.version_id is None for s in specs.values())


def test_shadow_rows_never_reach_the_brief_or_scorecard():
    conn = lineage_conn()
    _, sid = _setup_registry(conn)
    rows = predict.predict_day(conn, DAY)
    gid = rows[0]["game_id"]
    # adversarial: a shadow row under the champion's own model name, newer than the champion's
    conn.execute("INSERT INTO predictions (game_id, model, created_at, home_win_prob, nba_day, "
                 "home, away, model_version, role) VALUES (?,?,?,?,?,?,?,?,?)",
                 (gid, "gbm_market", f"{DAY}T12:30:00Z", 0.999, DAY.isoformat(), rows[0]["home"],
                  rows[0]["away"], sid, "shadow"))
    conn.commit()
    latest = predict.latest_pretip(conn, day=DAY.isoformat())
    assert latest[gid]["home_win_prob"] == rows[0]["home_win_prob"] != 0.999
    assert latest[gid]["role"] == "champion"
    assert predict.latest_pretip(conn, day=DAY.isoformat(), model=sid) == {}
    assert set(predict.latest_pretip(conn, day=DAY.isoformat(), model=sid,
                                     include_shadow=True)) == {r["game_id"] for r in rows}
    view = conn.execute("SELECT COUNT(*) FROM latest_pretip_predictions WHERE role='shadow'"
                        ).fetchone()[0]
    assert view == 0
    # scorecard: resolved predictions of DAY (results exist in team_games)
    res = assemble.resolved_predictions(conn, DAY.isoformat(),
                                        (DAY + timedelta(days=1)).isoformat(), "gbm_market")
    assert len(res) == 3 and all(r["p"] != 0.999 for r in res)
    assert assemble.resolved_predictions(conn, DAY.isoformat(), "9999-01-01", sid) == []


def test_migration6_never_clobbers_across_versions():
    conn = lineage_conn()
    base = {"game_id": "g1", "model": "gbm_opinion", "created_at": "2024-01-01T12:00:00Z",
            "home_win_prob": 0.6, "home_margin": 1.0, "total": 220.0, "market_home_spread": None,
            "market_total": None, "market_home_prob": None, "inputs": {}, "nba_day": "2024-01-01",
            "home": "BOS", "away": "NYK"}
    predict.store_predictions(conn, [{**base, "model_version": "v1", "role": "champion"},
                                     {**base, "home_win_prob": 0.4, "model_version": "v2",
                                      "role": "shadow"}])
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2
    # a rerun of the same version at the same virtual time still replaces (idempotent)
    predict.store_predictions(conn, [{**base, "home_win_prob": 0.65, "model_version": "v1",
                                      "role": "champion"}])
    rows = conn.execute("SELECT model_version, home_win_prob FROM predictions ORDER BY 1"
                        ).fetchall()
    assert [tuple(r) for r in rows] == [("v1", 0.65), ("v2", 0.4)]
    # readers see the champion only; the evaluation view sees both, one row each
    v = conn.execute("SELECT model_version FROM latest_pretip_predictions").fetchall()
    assert [r[0] for r in v] == ["v1"]
    a = conn.execute("SELECT model_version FROM latest_pretip_predictions_all ORDER BY 1"
                     ).fetchall()
    assert [r[0] for r in a] == ["v1", "v2"]
    # ties on created_at between a legacy (NULL) row and a tagged one: one row, the newest id
    predict.store_predictions(conn, [{**base, "home_win_prob": 0.7}])
    v = conn.execute("SELECT model_version, home_win_prob FROM latest_pretip_predictions"
                     ).fetchall()
    assert len(v) == 1 and v[0][1] == 0.7
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO predictions (game_id, model, created_at, role) "
                     "VALUES ('g', 'm', 't', 'bogus')")


def test_migration6_keeps_existing_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
    for i, sql in enumerate(db.MIGRATIONS[:5], start=1):
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version VALUES (?, 'x')", (i,))
    conn.execute("INSERT INTO predictions (game_id, model, created_at, home_win_prob, "
                 "inputs_json) VALUES ('g1', 'gbm_market', 't1', 0.55, '{}')")
    conn.commit()
    conn.close()
    c = db.connect(path)
    r = c.execute("SELECT id, game_id, home_win_prob, role FROM latest_pretip_predictions"
                  ).fetchone()
    assert tuple(r) == (1, "g1", 0.55, None)


def test_shadow_pairs_use_live_rows_of_both_versions():
    conn = lineage_conn(LONG)
    _, sid = _setup_registry(conn)
    predict.predict_day(conn, DAY)
    # mark the day's games final in `games`
    for g in conn.execute("SELECT h.game_id, h.pts, a.pts FROM team_games h JOIN team_games a "
                          "ON a.game_id=h.game_id AND a.is_home=0 WHERE h.is_home=1 AND "
                          "h.game_date=?", (DAY.isoformat(),)).fetchall():
        conn.execute("INSERT OR REPLACE INTO games (game_id, nba_day, status, home_score, "
                     "away_score) VALUES (?,?,?,?,?)", (g[0], DAY.isoformat(), "final", g[1], g[2]))
    conn.commit()
    pairs = retrain.shadow_pairs(conn, sid, "gbm_opinion")
    assert len(pairs) == 3 and (pairs["p_shadow"] != pairs["p_champion"]).any()
    out = retrain.evaluate_shadows(conn, dry_run=True, log_fn=lambda s: None)
    assert out[0]["n"] == 3 and out[0]["action"] == "wait"
    out = retrain.evaluate_shadows(conn, min_games=3, max_games=3, log_fn=lambda s: None)
    assert out[0]["action"] in ("promote", "reject") and out[0]["applied"]
    st = registry.get_version(conn, sid)["status"]
    assert st == ("champion" if out[0]["action"] == "promote" else "retired")
