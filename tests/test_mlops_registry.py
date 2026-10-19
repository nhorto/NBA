"""Model registry: spec hashing, register/promote/rollback atomicity, bootstrap, card, MLflow
optional. No network."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from courtside import db
from courtside.mlops import modelcard, registry, tracking


@pytest.fixture(autouse=True)
def no_mlflow(monkeypatch):
    monkeypatch.setenv("COURTSIDE_MLFLOW", "0")


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "reg.db")


P1 = {"market": False, "fixed": {"margin": [{"num_leaves": 15}, 89]}, "search_grid": []}
P2 = {"market": False, "fixed": {"margin": [{"num_leaves": 31}, 60]}, "search_grid": []}
FEATS = ["ps_diff", "elo_diff"]


def _champions(conn, name):
    return [r[0] for r in conn.execute("SELECT version_id FROM model_versions WHERE "
                                       "model_name=? AND status='champion'", (name,))]


def _events(conn):
    return [(r[0], r[1]) for r in conn.execute("SELECT version_id, action FROM registry_events "
                                               "ORDER BY id")]


# --- identity ---------------------------------------------------------------------------------

def test_spec_hash_is_deterministic_and_sensitive():
    a = registry.spec_hash("gbm_opinion", {"b": 1, "a": [1, 2]}, ("x", "y"), "abc")
    b = registry.spec_hash("gbm_opinion", {"a": [1, 2], "b": 1}, ["x", "y"], "abc")
    assert a == b and len(a) == 64                  # key order / tuple vs list don't matter
    assert a == registry.spec_hash("gbm_opinion", {"b": 1, "a": [1, 2]}, ["x", "y"], "abc")
    for other in (registry.spec_hash("gbm_market", {"b": 1, "a": [1, 2]}, ["x", "y"], "abc"),
                  registry.spec_hash("gbm_opinion", {"b": 2, "a": [1, 2]}, ["x", "y"], "abc"),
                  registry.spec_hash("gbm_opinion", {"b": 1, "a": [1, 2]}, ["x"], "abc"),
                  registry.spec_hash("gbm_opinion", {"b": 1, "a": [1, 2]}, ["x", "y"], "abd")):
        assert other != a


def test_production_spec_hash_is_stable_across_processes():
    code = ("from courtside.mlops import registry as r; n='gbm_market'; "
            "print(r.spec_hash(n, r.production_params(n), r.feature_list(n), 'sha'))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip() for _ in range(2)}
    assert len(outs) == 1 and len(outs.pop()) == 64


def test_code_sha_tracks_model_files_and_dirty_state(tmp_path):
    root = tmp_path / "repo"
    (root / "src/courtside/model").mkdir(parents=True)
    (root / "src/courtside/model/gbm.py").write_text("X = 1\n")
    (root / "src/courtside/production.py").write_text("Y = 1\n")
    (root / "README.md").write_text("hi\n")
    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([*g, "add", "."], cwd=root, check=True)
    subprocess.run([*g, "commit", "-qm", "m"], cwd=root, check=True)
    clean = registry.code_sha(root)
    assert len(clean) == 12 and "dirty" not in clean
    (root / "README.md").write_text("docs only\n")              # unrelated change: same sha
    subprocess.run([*g, "commit", "-qam", "docs"], cwd=root, check=True)
    assert registry.code_sha(root) == clean
    (root / "src/courtside/model/gbm.py").write_text("X = 2\n")
    dirty = registry.code_sha(root)
    assert dirty.startswith(clean + "-dirty-") and dirty == registry.code_sha(root)
    assert registry.code_sha(tmp_path / "nogit").startswith("nogit-")


# --- register / promote / rollback ------------------------------------------------------------

def test_register_ids_are_readable_and_identical_specs_dedupe(conn):
    v1 = registry.register(conn, "gbm_opinion", P1, FEATS, code="c1")
    assert v1.startswith("gbm_opinion@v1-") and len(v1.split("-")[-1]) == 8
    assert registry.register(conn, "gbm_opinion", P1, FEATS, code="c1") == v1
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c1", status="shadow")
    assert v2.startswith("gbm_opinion@v2-")
    assert registry.get_version(conn, v2)["status"] == "shadow"
    assert registry.get_version(conn, v1)["params"] == P1
    assert [e[1] for e in _events(conn)] == ["register", "register", "shadow"]


def test_promote_swaps_champions_atomically_and_logs(conn):
    v1 = registry.register(conn, "gbm_opinion", P1, FEATS, code="c", status="champion")
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    m1 = registry.register(conn, "gbm_market", P1, FEATS, code="c", status="champion")
    assert _champions(conn, "gbm_opinion") == [v1]
    assert registry.promote(conn, v2, "better") == v1
    assert _champions(conn, "gbm_opinion") == [v2] and _champions(conn, "gbm_market") == [m1]
    old = registry.get_version(conn, v1)
    assert old["status"] == "retired" and old["retired_at"] and old["promoted_at"]
    assert registry.get_version(conn, v2)["promoted_at"]
    assert _events(conn)[-2:] == [(v1, "retire"), (v2, "promote")]
    assert registry.promote(conn, v2, "again") is None                   # no-op
    assert _champions(conn, "gbm_opinion") == [v2]


def test_failed_promotion_rolls_back_everything(conn, monkeypatch):
    v1 = registry.register(conn, "gbm_opinion", P1, FEATS, code="c", status="champion")
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    n_events = len(_events(conn))
    real = registry._event

    def boom(c, vid, action, *a, **k):
        if action == "promote":
            raise RuntimeError("disk full")
        return real(c, vid, action, *a, **k)
    monkeypatch.setattr(registry, "_event", boom)
    with pytest.raises(RuntimeError):
        registry.promote(conn, v2, "x")
    assert _champions(conn, "gbm_opinion") == [v1]                       # old one kept
    assert registry.get_version(conn, v1)["retired_at"] is None
    assert registry.get_version(conn, v2)["status"] == "candidate"
    assert len(_events(conn)) == n_events and not conn.in_transaction


def test_one_champion_invariant_is_enforced(conn):
    registry.register(conn, "gbm_opinion", P1, FEATS, code="c", status="champion")
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    with pytest.raises(registry.RegistryError):
        registry.set_status(conn, v2, "champion")                        # only via promote
    # a corrupted second champion makes the next promotion fail and change nothing
    conn.execute("UPDATE model_versions SET status='champion' WHERE version_id=?", (v2,))
    conn.commit()
    v3 = registry.register(conn, "gbm_opinion", {**P2, "x": 1}, FEATS, code="c")
    with pytest.raises(registry.RegistryError):
        registry.promote(conn, v3, "x")
    assert registry.get_version(conn, v3)["status"] == "candidate"


def test_promote_inside_a_callers_transaction_uses_a_savepoint(conn):
    v1 = registry.register(conn, "gbm_opinion", P1, FEATS, code="c", status="champion")
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message) "
                 "VALUES ('k', 't', 't', 'info', 'x', 'm')")
    assert conn.in_transaction
    registry.promote(conn, v2, "inside")
    conn.rollback()                                    # the caller abandons its transaction
    assert _champions(conn, "gbm_opinion") == [v1]


def test_rollback_restores_the_last_retired_champion_not_rejects(conn):
    v1 = registry.register(conn, "gbm_opinion", P1, FEATS, code="c", status="champion")
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    rej = registry.register(conn, "gbm_opinion", {**P2, "r": 1}, FEATS, code="c")
    registry.set_status(conn, rej, "retired", "gate failed", action="reject")
    registry.promote(conn, v2, "shadow won")
    assert registry.rollback(conn, "gbm_opinion", "live regression") == v1
    assert _champions(conn, "gbm_opinion") == [v1]
    assert registry.get_version(conn, v2)["status"] == "retired"
    assert _events(conn)[-1] == (v1, "rollback")
    assert registry.rollback(conn, "gbm_opinion", "undo") == v2         # and back again
    with pytest.raises(registry.RegistryError):
        registry.rollback(conn, "gbm_market", "nothing there")


# --- bootstrap / card / cli ------------------------------------------------------------------

def test_bootstrap_registers_production_champions_once(conn):
    from courtside.model import gbm
    out = registry.bootstrap(conn, log_fn=lambda s: None)
    assert [o["created"] for o in out] == [True, True]
    for name in (gbm.PRODUCTION_WINPROB, gbm.PRODUCTION_OPINION):
        c = registry.get_champion(conn, name)
        assert c["version_id"].startswith(f"{name}@v1-")
        assert registry.fixed_from_params(c["params"]) == gbm.FROZEN[name]
        assert c["benchmark"]["regular_season"]["n"] == 16898
        assert c["benchmark"]["all"]["log_loss"] > 0.5
    again = registry.bootstrap(conn, log_fn=lambda s: None)
    assert [o["created"] for o in again] == [False, False]
    assert conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] == 2


def test_model_card_renders_from_registry(conn, tmp_path):
    registry.bootstrap(conn, log_fn=lambda s: None)
    v = registry.register(conn, "gbm_opinion", P2 | {"fixed": {"margin": [
        registry.production_params("gbm_opinion")["fixed"]["margin"][0], 7]}}, FEATS,
        code="c", status="shadow", notes="test shadow")
    path = modelcard.write(conn, tmp_path / "MODEL_CARD.md")
    text = path.read_text()
    for s in ("## Intended use", "not betting advice", "gbm_market@v1-", "gbm_opinion@v1-",
              "Versus the closing line", "Calibration", "Injury data only from 2026",
              "Availability proxy", "No edge against the spread", "Early-season priors", v):
        assert s in text, s


def test_cli_commands(conn, tmp_path, monkeypatch, capsys):
    from courtside.mlops import cli
    monkeypatch.setattr(db, "connect", lambda *a, **k: conn)
    assert cli.main(["bootstrap"]) == 0
    assert cli.main(["versions"]) == 0
    v2 = registry.register(conn, "gbm_opinion", P2, FEATS, code="c")
    assert cli.main(["promote", v2, "--reason", "manual"]) == 0
    assert registry.get_champion(conn, "gbm_opinion")["version_id"] == v2
    assert cli.main(["rollback", "gbm_opinion", "--reason", "oops"]) == 0
    assert registry.get_champion(conn, "gbm_opinion")["version_id"].startswith("gbm_opinion@v1-")
    assert cli.main(["shadows", "--dry-run"]) == 0
    assert cli.main(["card", "--out", str(tmp_path / "card.md")]) == 0
    assert (tmp_path / "card.md").exists()
    assert "gbm_opinion@v1-" in capsys.readouterr().out


# --- MLflow is optional -----------------------------------------------------------------------

def test_mlflow_missing_never_crashes(conn, monkeypatch):
    monkeypatch.setenv("COURTSIDE_MLFLOW", "1")
    monkeypatch.setitem(sys.modules, "mlflow", None)            # import mlflow -> ImportError
    assert tracking._import_mlflow() is None and not tracking.available()
    assert tracking.log_benchmark("x", {"a": 1}, {"m": 0.5}) is None
    out = registry.bootstrap(conn, log_fn=lambda s: None)       # log_mlflow=True by default
    assert all(o["created"] and o["mlflow_run_id"] is None for o in out)


def test_mlflow_misconfigured_never_crashes(monkeypatch, tmp_path):
    monkeypatch.setenv("COURTSIDE_MLFLOW", "1")
    monkeypatch.setattr(tracking, "tracking_uri", lambda: "notascheme://broken")
    assert tracking.log_benchmark("x", {"a": 1}, {"m": 0.5}) is None


def test_mlflow_logs_runs_locally(monkeypatch, data_dir):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("COURTSIDE_MLFLOW", "1")
    rid = tracking.log_benchmark("bench", {"p": {"leaves": 7}},
                                 {"all": {"log_loss": 0.6, "n": 10}, "bad": float("nan"),
                                  "txt": "skip"}, {"gate": {"passed": True}})
    assert rid and (data_dir / "mlflow.db").exists()
    import mlflow
    run = mlflow.get_run(rid)
    assert run.data.metrics["all.log_loss"] == 0.6 and run.data.params["p.leaves"] == "7"


def test_production_import_does_not_import_mlflow():
    code = "import sys, courtside.predict, courtside.production; print('mlflow' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_params_round_trip_to_frozen_format():
    from courtside.model import gbm
    for n in (gbm.PRODUCTION_WINPROB, gbm.PRODUCTION_OPINION):
        p = json.loads(json.dumps(registry.production_params(n)))
        assert registry.fixed_from_params(p) == gbm.FROZEN[n]
