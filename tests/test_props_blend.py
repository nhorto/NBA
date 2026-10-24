"""G3 props blend evaluator (proj/props_blend.py): maths, walk-forward without look-ahead, the
lock-45 rows, eligibility, the pre-registered status logic, read-only behaviour, the ledger."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import config, db
from courtside.proj import props as P
from courtside.proj import props_blend as B

import props_helpers as H

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the pre-registration

def test_prereg_doc_is_pinned_and_constants_fixed():
    assert B.PREREG_DOC.exists()
    assert hashlib.sha256(B.PREREG_DOC.read_bytes()).hexdigest() == B.PREREG_SHA256
    assert B.PREREG["decision_slates"] == 60 and B.PREREG["first_scored_day"] == "2026-10-25"
    assert B.PREREG["z_pass"] == -2.0 and B.PREREG["z_guard"] == 2.0
    assert B.PREREG["run_kind"] == "lock45" and B.PREREG["train_min_rows"] == 1000
    assert B.PREREG["devig_min_rows"] == 10000 and B.PREREG["clip"] == 0.001
    assert B.PREREG["stats"] == P.STATS


# ------------------------------------------------------------------ maths

def test_devig_methods():
    qo, qu = B.implied_arr([-110, 120, -150]), B.implied_arr([-110, -150, 125])
    m = B.devig(qo, qu, "mult")
    np.testing.assert_allclose(m, [P.no_vig(o, u)[0] for o, u in
                                   ((-110, -110), (120, -150), (-150, 125))])
    a = B.devig(qo, qu, "add")
    np.testing.assert_allclose(a, qo - (qo + qu - 1) / 2)
    pw = B.devig(qo, qu, "power")
    k = np.log(pw) / np.log(qo)
    np.testing.assert_allclose(qo ** k + qu ** k, 1.0, atol=1e-9)
    for meth in ("mult", "add", "power"):        # a fair two-way price: 0.5 whatever the method
        assert B.devig(qo[:1], qu[:1], meth)[0] == pytest.approx(0.5)
    # the favourite gets a little more under power than under proportional
    assert pw[2] > m[2]
    assert np.isnan(B.implied_arr([50, None])).all()


def _sim(n_days=20, per_day=300, a=0.7, b=0.4, seed=0, start="2026-10-20"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days).strftime("%Y-%m-%d")
    rows = []
    for di, d in enumerate(days):
        lm = rng.normal(0, 0.6, per_day)
        lo = lm + rng.normal(0, 0.5, per_day)
        p = 1 / (1 + np.exp(-(a * lm + b * lo)))
        y = (rng.random(per_day) < p).astype(int)
        for i in range(per_day):
            rows.append({"nba_day": d, "game_id": f"g{di}_{i % 10}", "player_id": 1000 * di + i
                         % 120, "stat": P.STATS[(i // 120) % 5], "line": 10.5 + i, "p_mkt":
                         1 / (1 + np.exp(-lm[i])), "p_ours": 1 / (1 + np.exp(-lo[i])),
                         "y": y[i]})
    return pd.DataFrame(rows)


def test_fit_logit_recovers_coefficients_and_sandwich_se():
    df = _sim(n_days=30, per_day=1000, seed=1)
    X = B.blend_X(df["p_mkt"], df["p_ours"])
    b = B.fit_logit(X, df["y"].to_numpy())
    assert b == pytest.approx([0.7, 0.4], abs=0.08)
    V = B.sandwich2(X, df["y"].to_numpy(), b, df["game_id"], df["player_id"])
    se = np.sqrt(np.diag(V))
    assert np.all(se > 0) and np.all(se < 0.1)
    # with an offset only (the shade term): recovers a known intercept
    rng = np.random.default_rng(2)
    off = rng.normal(0, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(off + 0.3)))).astype(int)
    c = B.fit_logit(np.ones((20000, 1)), y, offset=off)
    assert c[0] == pytest.approx(0.3, abs=0.05)


def test_walk_forward_never_looks_ahead():
    df = _sim(n_days=8, per_day=400, seed=3)
    X = B.blend_X(df["p_mkt"], df["p_ours"])
    pred, coefs = B.walk_forward(df, X)
    days = sorted(df["nba_day"].unique())
    # the first days have < 1,000 training rows (400, 800): not scored
    assert np.isnan(pred[df["nba_day"].isin(days[:3])]).all()
    assert np.isfinite(pred[df["nba_day"].isin(days[3:])]).all()
    assert sorted(coefs) == days[3:]
    # flipping the outcomes of day k and later leaves every prediction up to day k unchanged
    k = days[5]
    df2 = df.copy()
    late = df2["nba_day"] >= k
    df2.loc[late, "y"] = 1 - df2.loc[late, "y"]
    pred2, _ = B.walk_forward(df2, X)
    upto = (df["nba_day"] <= k).to_numpy()
    np.testing.assert_array_equal(pred[upto], pred2[upto])
    assert not np.allclose(pred[~upto], pred2[~upto])


# ------------------------------------------------------------------ rows at the lock-45 state

@pytest.fixture
def conn():
    c = db.connect()
    H.populate(c)
    # a lock-45 run for G1 only (G2 has props but no lock-45 run)
    c.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, ok) "
              "VALUES ('r45', 'S1', 'lock45', '2026-10-20T22:15:00Z', 'pathA-test', 1)")
    for pid in (101, 102, 201, 202):
        H._projection(c, "r45", "2026-10-20T22:15:00Z", pid, scale=1.05)
    c.commit()
    return c


def test_lock45_run_rule(conn):
    assert B.lock45_run(conn, H.G1, H.TIP1)["run_id"] == "r45"     # r_fail (ok=0) ignored
    assert B.lock45_run(conn, H.G2, H.TIP2) is None
    assert B.lock45_run(conn, H.G1, "2026-10-20T22:00:00Z") is None  # created after the cap


def test_day_rows_and_grading(conn):
    res = B.day_rows(conn, H.DAY)
    assert res["games"] == 1 and res["no_run"] == 1
    rows = {(r["player_id"], r["stat"], r["line"]): r for r in res["rows"]}
    # market state at the run's created_at (22:15): Alder's points line is the 20:00 one, 20.5
    assert set(rows) == {(101, "pts", 20.5), (101, "reb", 5.0), (102, "ast", 2.5),
                         (201, "pra", 30.5)}
    r = rows[(101, "pts", 20.5)]
    assert r["run_id"] == "r45" and r["captured_at"] == "2026-10-20T20:00:00Z"
    assert r["p_mkt"] == pytest.approx(0.5) and r["dist_source"] == "approx"
    assert r["game_state"] == "pending" and r["y"] is None
    # same numbers as props.compare_game at that state
    g = [x for x in P.day_games(conn, H.DAY) if x["game_id"] == H.G1][0]
    run = B.lock45_run(conn, H.G1, H.TIP1)
    ref = {(x["player_id"], x["stat"], x["line"]): x for x in
           P.compare_game(conn, g, run=run, market_at=run["created_at"])["rows"]}
    for k, v in rows.items():
        assert v["our_p_above"] == ref[k]["our_p_above"]
        assert v["p_ours"] == pytest.approx(P.comparable(ref[k]["our_p_above"],
                                                         ref[k]["our_p_push"]))
    H.finish_g1(conn)
    rows = {(r["player_id"], r["stat"], r["line"]): r for r in B.day_rows(conn, H.DAY)["rows"]}
    assert (rows[(101, "pts", 20.5)]["result"], rows[(101, "pts", 20.5)]["y"]) == ("above", 1)
    assert rows[(101, "reb", 5.0)]["result"] == "push" and rows[(101, "reb", 5.0)]["y"] is None
    assert (rows[(102, "ast", 2.5)]["result"], rows[(102, "ast", 2.5)]["y"]) == ("below", 0)
    assert rows[(201, "pra", 30.5)]["result"] == "void"
    # the grades agree with props.grade_day on the same keys
    P.write_day(conn, H.DAY, at="2026-10-21T03:00:00Z")
    P.grade_day(conn, H.DAY)
    for (pid, stat, line), r in rows.items():
        g = conn.execute("SELECT result FROM prop_compare WHERE player_id=? AND market=? AND "
                         "line=?", (pid, stat, line)).fetchone()
        if g is not None:
            assert g[0] == r["result"]


def test_eligibility_rules():
    base = {"game_id": "g", "player_id": 1, "stat": "pts", "game_state": "final",
            "result": "above", "p_ours": 0.5, "hold": 0.05, "captured_at": "T1", "nba_day": "d"}
    df = pd.DataFrame([
        {**base, "line": 20.5, "p_mkt": 0.55},
        {**base, "line": 21.5, "p_mkt": 0.48},                      # closest to 0.5: kept
        {**base, "player_id": 2, "line": 9.5, "p_mkt": None},
        {**base, "player_id": 3, "line": 9.5, "p_mkt": 0.5, "hold": 0.4},
        {**base, "player_id": 4, "line": 9.0, "p_mkt": 0.5, "result": "push"},
        {**base, "player_id": 5, "line": 9.5, "p_mkt": 0.5, "result": "void"},
        {**base, "player_id": 6, "line": 9.5, "p_mkt": 0.5, "game_state": "pending",
         "result": None},
    ])
    assert list(B.eligibility(df)) == ["dup", "ok", "price", "hold", "push", "void",
                                       "not_final"]


# ------------------------------------------------------------------ status logic

def _frame_for(n_days, signal: bool, seed=0):
    df = _sim(n_days=n_days, per_day=250, a=0.7 if signal else 1.0, b=0.6 if signal else 0.0,
              seed=seed, start="2026-10-20")      # no signal: a calibrated market, ours noise
    if not signal:                                   # ours = pure noise around the market
        rng = np.random.default_rng(seed + 9)
        df["p_ours"] = 1 / (1 + np.exp(-rng.normal(0, 1, len(df))))
    df["game_state"], df["result"] = "final", np.where(df["y"] == 1, "above", "below")
    df["hold"], df["captured_at"] = 0.05, "2026-10-20T21:00:00Z"
    df["tipoff_utc"], df["run_created_at"] = "2026-10-20T23:00:00Z", "2026-10-20T22:15:00Z"
    df["model_version"], df["dist_source"] = "m", "approx"
    df["over_odds"], df["under_odds"] = -110, -110
    for c in B.LEDGER_COLS:
        if c not in df:
            df[c] = None
    return df[list(B.LEDGER_COLS)]


def test_collecting_before_the_minimum_is_interim():
    rep = B.evaluate(_frame_for(20, signal=True))
    assert rep["status"] == "collecting" and rep["decision"] is None
    assert rep["slates"]["n_pre_registration"] == 5      # 10-20 .. 10-24: training only
    assert rep["n_scored_slates"] == 15 and rep["scored_days"][0] == "2026-10-25"
    assert rep["interim"]["flag"].startswith("INTERIM: 15 of 60")
    assert rep["interim"]["primary"]["mean"] < 0
    assert "dry_run_pre_registration" in rep
    assert rep["devig"]["status"].startswith("waiting: 5000 of 10000")


def test_decision_pass_and_fail_on_60_slates():
    rep = B.evaluate(_frame_for(70, signal=True, seed=4))
    assert rep["status"] == "decided" and rep["decision"]["result"] == "PASS"
    assert rep["decision"]["decision_days"] == ["2026-10-25", "2026-12-23"]
    assert rep["decision"]["n"] == 60 * 250 and "follow_up" in rep
    rep = B.evaluate(_frame_for(70, signal=False, seed=5))
    assert rep["decision"]["result"] == "FAIL"
    # 17,500 rows: the de-vig comparison runs (reported only); fair prices: methods agree
    dv = rep["devig"]
    assert dv["flag"] == "reported only" and dv["n"] > 0
    assert abs(dv["vs_mult_raw"]["power"]["mean"]) < 1e-9


def test_devig_forced_is_flagged_interim():
    rep = B.evaluate(_frame_for(12, signal=True), devig=True)
    assert rep["devig"]["flag"].startswith("INTERIM") and rep["devig"]["n"] > 0


# ------------------------------------------------------------------ read-only, ledger, CLI

def _file_db(conn) -> Path:
    path = config.db_path()
    conn.commit()
    conn.close()
    return path


def test_run_is_read_only_and_ledger_freezes(conn, tmp_path):
    H.finish_g1(conn)
    path = _file_db(conn)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rep = B.run(ro, "2026-10-20", "2026-10-20", today="2026-10-22")
    assert rep["rows"] == 4 and rep["eligible"] == 2       # Alder pts, Claxton ast
    assert rep["status"] == "collecting" and rep["build"]["frozen_new"] == []  # G2 pending
    assert "ledger_written" not in rep["build"]
    ro.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # G2 final too (no rows of its own: no lock-45 run) -> the day freezes into the ledger
    rw = sqlite3.connect(path)
    rw.execute("UPDATE games SET status='final' WHERE game_id=?", (H.G2,))
    rw.commit()
    rw.close()
    led = tmp_path / "ledger.csv"
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rep1 = B.run(ro, "2026-10-20", "2026-10-20", ledger=led, today="2026-10-22")
    assert rep1["build"]["frozen_new"] == ["2026-10-20"] and led.exists()
    rep2 = B.run(ro, "2026-10-20", "2026-10-20", ledger=led, today="2026-10-22")
    ro.close()
    assert rep2["build"]["from_ledger"] == ["2026-10-20"] and rep2["build"]["computed"] == []
    for k in ("rows", "eligible", "excluded", "status"):
        assert rep1[k] == rep2[k]
    back = B.read_ledger(led)
    assert back["game_id"].tolist() == [H.G1] * 4                   # leading zeros kept


def test_nothing_user_facing_imports_it():
    hits = []
    for p in (ROOT / "src" / "courtside").rglob("*.py"):
        if p.name == "props_blend.py":
            continue
        if "props_blend" in p.read_text():
            hits.append(p.relative_to(ROOT).as_posix())
    assert hits == ["src/courtside/proj/cli.py"]


def test_props_outputs_unchanged_by_the_evaluator(conn):
    before = P.compare_day(conn, H.DAY)
    B.day_rows(conn, H.DAY)
    after = P.compare_day(conn, H.DAY)
    strip = lambda res: [(g["run"] or {}).get("run_id") for g in res["games"]] + \
        [sorted((r["player_id"], r["stat"], r["line"], r["our_p_above"], r["mkt_p_above"])
                for r in g["rows"]) for g in res["games"]]
    assert strip(before) == strip(after)
    assert conn.execute("SELECT COUNT(*) FROM prop_compare").fetchone()[0] == 0


def test_cli(conn, capsys):
    from courtside import cli
    H.finish_g1(conn)
    path = _file_db(conn)
    assert cli.main(["proj", "props-blend-eval", "--db", str(path), "--from", "2026-10-20",
                     "--to", "2026-10-20"]) == 0
    out = capsys.readouterr().out
    assert "G3 props blend test" in out and "status: collecting" in out
