"""Player-side live shadows (proj/shadows.py, IMPROVEMENTS E1): the champion is bit-identical
with shadows on or off, a failing shadow cannot touch it, the time budget holds, the lock guard,
the pre-registered grading rules, the registry, the /health/dfs card and the job wiring."""

from __future__ import annotations

import json
import sqlite3
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import config, db
from courtside.proj import metrics as MT
from courtside.proj import pipeline as P
from courtside.proj import shadows as SH
from courtside.proj import store as SO
from courtside.proj import tail_remap as TR

from test_proj_pipeline import SEASON, _copy, _fill, _late_day

ALL_ARMS = ["+tr", "+ep", "+gd", "+lf", "+tm", "+rs", "+pq"]
SID = "777"
N_SIMS = 1500


# ============================================================================== world

@pytest.fixture(scope="module")
def world():
    """A migrated in-memory DB of the synthetic league with one Classic slate of the late day
    (both games, every player mapped to a DK id), the Context and a default bundle."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    day = _late_day(c)
    games = P._home_away(c, day)
    gids = list(games["game_id"])
    with c:
        c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json) "
                  "VALUES (?, ?, 70, 'main', ?)", (SID, day.isoformat(), json.dumps(gids)))
        k = 0
        for g in games.itertuples():
            for pid, team in c.execute("SELECT player_id, team FROM player_games WHERE game_id=?",
                                       (g.game_id,)).fetchall():
                c.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                          "game_type_id, draftable_id, player_name, site_player_id, team, "
                          "positions, roster_slot, salary, game_id, home, away, start_time, "
                          "disabled) VALUES ('2023-01-01T00:00:00Z', 'dk', ?, 'Classic', 70, ?, "
                          "?, ?, ?, 'SF', 'UTIL', 5000, ?, ?, ?, '2030-01-01T00:00:00Z', 0)",
                          (SID, f"d{k}", f"P{pid}", f"dk{pid}", team, g.game_id, g.home, g.away))
                c.execute("INSERT OR IGNORE INTO player_ids (source, source_id, player_id) VALUES "
                          "('dk', ?, ?)", (f"dk{pid}", int(pid)))
                k += 1
    bundle = P.Bundle(season=SEASON, minutes=P.M.default_params(SEASON),
                      state=P.ST.default_params(), copula=P.C.default_copula_params(SEASON))
    ctx = P.Context(c, SEASON)
    # 16:00 UTC: before the default 19:30 ET tip of a day without known tips
    now = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=16)
    return {"conn": c, "day": day, "bundle": bundle, "ctx": ctx, "now": now, "games": gids}


def _champion(w, seed=12345):
    """The champion's projection exactly as a DFS lock run makes it (minus the store)."""
    out = P.project_slate(w["conn"], SID, w["now"], run_id="lock", n_sims=N_SIMS, seed=seed,
                          kind="lock45", ctx=w["ctx"], bundle=w["bundle"], store=False)
    r = out.result
    pid, fp = r.fp_matrix()
    return {"proj": {k: (v.mean, v.p10, v.p25, v.p50, v.p90) for k, v in out.items()},
            "players": r.players.copy(), "stats": r.stats.copy(), "fp": fp.copy(),
            "pid": pid.copy(), "version": out.meta["model_version"]}


def _same(a, b):
    assert a["proj"] == b["proj"]
    assert a["version"] == b["version"]
    pd.testing.assert_frame_equal(a["players"], b["players"], check_exact=True)
    pd.testing.assert_frame_equal(a["stats"], b["stats"], check_exact=True)
    assert np.array_equal(a["pid"], b["pid"])
    assert a["fp"].dtype == b["fp"].dtype and np.array_equal(a["fp"], b["fp"])


def _run(w, tmp_path, **kw):
    kw.setdefault("arms", ALL_ARMS)
    kw.setdefault("budget", 600)
    return SH.run(w["conn"], w["now"], n_sims=N_SIMS, ctx=w["ctx"], bundle=w["bundle"],
                  guard=False, out_root=tmp_path / "shadows", **kw)


@pytest.fixture(scope="module")
def shadow_run(world, tmp_path_factory):
    """One full shadow run (every arm) on the world, plus the champion before and after it."""
    out = tmp_path_factory.mktemp("sh")
    before = _champion(world)
    snap = SH.switch_state()
    res = _run(world, out)
    after = _champion(world)
    return {"res": res, "before": before, "after": after, "snap": snap, "root": out / "shadows"}


# ============================================================================== identity

def test_champion_bit_identical_with_shadows_on_or_off(world, shadow_run):
    res = shadow_run["res"]
    assert res["status"] == "ok", res
    ok = [a for a, v in res["arms"].items() if v["status"] == "ok" and a != SH.CHAMPION]
    assert len(ok) >= 4, res["arms"]
    # OFF = the champion before any shadow ran; ON = after a full shadow run in the same process
    _same(shadow_run["before"], shadow_run["after"])
    assert SH.switch_state() == shadow_run["snap"]
    # and nothing was written where the site, the props or the DFS registry read
    c = world["conn"]
    for t in ("player_projections", "player_projection_stats", "dfs_runs", "dfs_lineups"):
        assert c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0, t
    assert not (config.data_dir() / "sims").exists()


def test_shadow_champion_is_the_champion_code_path(world, shadow_run):
    """The stored shadow champion = `project_slate` at the shadow seed, row for row."""
    res = shadow_run["res"]
    ch = _champion(world, seed=res["seed"])
    df, q, meta = SH.load_arm(shadow_run["root"] / res["day"] / res["arms"][SH.CHAMPION]["file"])
    assert meta["model_version"] == ch["version"] == res["champion_version"]
    pl = ch["players"].reset_index(drop=True)
    assert np.array_equal(df["player_id"].to_numpy(), pl["player_id"].to_numpy(np.int64))
    assert np.array_equal(df["dk_mean"].to_numpy(), pl["dk_mean"].to_numpy(float))
    exp = np.quantile(ch["fp"], SH.LEVELS, axis=0, method="inverted_cdf").T.astype(np.float32)
    assert np.array_equal(q, exp.astype(float))


def test_arms_are_tagged_and_share_seeds(world, shadow_run):
    res = shadow_run["res"]
    d = shadow_run["root"] / res["day"]
    for tag, v in res["arms"].items():
        if v["status"] != "ok":
            continue
        _, _, meta = SH.load_arm(d / v["file"])
        assert meta["seed"] == res["seed"] and meta["n_sims"] == N_SIMS
        if tag != SH.CHAMPION:
            assert meta["model_version"] != res["champion_version"]
    # +ep changes the bundle suffix: no bundle file in the test data dir, so it is skipped and
    # no bundle is fitted or written
    assert res["arms"]["+ep"]["status"] == "skipped"
    assert "bundle missing" in res["arms"]["+ep"]["reason"]
    assert not list(config.data_dir().rglob("bundle_*"))
    man = json.loads((d / f"{res['run']}.json").read_text())
    assert man["arms"].keys() == res["arms"].keys()
    # the arms really switch something: some stored projections differ from the champion's
    _, q0, _ = SH.load_arm(d / res["arms"][SH.CHAMPION]["file"])
    moved = [t for t, v in res["arms"].items() if t != SH.CHAMPION and v["status"] == "ok"
             and not np.array_equal(SH.load_arm(d / v["file"])[1], q0)]
    assert moved, res["arms"]


def test_disabled_flag(world, tmp_path, monkeypatch):
    monkeypatch.setenv(SH.SHADOWS_ENV, "0")
    res = _run(world, tmp_path)
    assert res["status"] == "disabled"
    assert not (tmp_path / "shadows").exists()
    monkeypatch.setenv(SH.SHADOWS_ENV, "1")
    assert SH.enabled()
    monkeypatch.delenv(SH.SHADOWS_ENV)
    assert SH.enabled() is SH.SHADOWS_DEFAULT is True


def test_already_ran_is_idempotent(world, shadow_run, tmp_path):
    res = SH.run(world["conn"], world["now"], arms=["+tm"], n_sims=N_SIMS, ctx=world["ctx"],
                 bundle=world["bundle"], guard=False, out_root=shadow_run["root"])
    assert res["status"] == "skipped" and "already ran" in res["reason"]


# ============================================================================== isolation

def test_failing_shadow_cannot_touch_the_champion(world, tmp_path, monkeypatch):
    before = _champion(world)
    snap = SH.switch_state()

    def project(conn, sid, at, **kw):          # +tr blows up inside the projection
        if TR.TAIL_REMAP:
            raise RuntimeError("boom in the tail remap")
        return SH._project(conn, sid, at, **kw)

    real = P.set_tier_match

    def bad_setter(on):                         # +tm's setter half-applies, then raises
        real(on)
        if on:
            raise ValueError("setter broke")
    monkeypatch.setattr(P, "set_tier_match", bad_setter)
    res = _run(world, tmp_path, arms=["+tr", "+tm", "+lf", "+nope"], project=project)
    assert res["status"] == "ok"
    assert res["arms"][SH.CHAMPION]["status"] == "ok"
    assert res["arms"]["+tr"]["status"] == "error" and "boom" in res["arms"]["+tr"]["error"]
    assert res["arms"]["+tm"]["status"] == "error" and "setter" in res["arms"]["+tm"]["error"]
    assert res["arms"]["+lf"]["status"] == "ok"
    assert res["arms"]["+nope"]["status"] == "skipped"
    assert SH.switch_state() == snap
    monkeypatch.setattr(P, "set_tier_match", real)
    _same(before, _champion(world))


def test_failing_champion_or_registry_never_raises(world, tmp_path, monkeypatch):
    snap = SH.switch_state()

    def project(conn, sid, at, **kw):
        raise MemoryError("out of memory")
    res = _run(world, tmp_path, project=project)
    assert res["status"] == "error" and res["arms"][SH.CHAMPION]["status"] == "error"
    assert set(res["arms"]) == {SH.CHAMPION}
    assert SH.switch_state() == snap
    monkeypatch.setattr(SH, "REGISTRY_PATH", tmp_path / "missing.json")
    res = SH.run(world["conn"], world["now"], guard=False, out_root=tmp_path / "x")
    assert res["status"] == "error" and "missing.json" in res["error"]


def test_lock_guard_skips_near_a_build(world, tmp_path, monkeypatch):
    from courtside.proj.dfs import runner as RN
    hit = world["now"] + timedelta(seconds=90)

    def due(conn, t):
        return [{"kind": "lock10", "slate_id": SID}] if abs((t - hit).total_seconds()) < 31 else []
    monkeypatch.setattr(RN, "due_runs", due)
    res = SH.run(world["conn"], world["now"], arms=["+tm"], n_sims=N_SIMS, ctx=world["ctx"],
                 bundle=world["bundle"], budget=60, out_root=tmp_path / "shadows")
    assert res["status"] == "skipped" and "lock10" in res["reason"]
    d = tmp_path / "shadows" / res["day"]
    assert [p.name for p in d.iterdir()] == [f"{res['run']}.json"]      # a manifest, no arms
    assert SH.run_log(tmp_path / "shadows", world["day"])[0]["status"] == "skipped"
    # a window 10 minutes past the budget does not block
    hit = world["now"] + timedelta(seconds=60 + SH.GUARD_PAD_S + 600)
    assert SH.lock_conflict(world["conn"], world["now"], 60) is None

    def broken(conn, t):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(RN, "due_runs", broken)
    assert "lock guard failed" in SH.lock_conflict(world["conn"], world["now"], 60)


def test_lock_guard_on_the_real_tick_logic(world):
    """With the real `runner.due_runs`: the slate's lock-45 window blocks, far away does not."""
    c = _copy(world["conn"])
    lock = world["now"] + timedelta(minutes=45)
    ts = lock.strftime("%Y-%m-%dT%H:%M:%SZ")
    with c:
        c.execute("UPDATE dfs_salaries SET start_time=?", (ts,))
        c.execute("UPDATE dfs_slates SET lock_utc=? WHERE slate_id=?", (ts, SID))
    assert SH.lock_conflict(c, world["now"], 90) is not None
    assert SH.lock_conflict(c, world["now"] - timedelta(hours=3), 90) is None


# ============================================================================== budget

@pytest.fixture(scope="module")
def cached_projection(world):
    return P.project_slate(world["conn"], SID, world["now"], run_id="c", n_sims=N_SIMS,
                           kind=SH.SEED_KIND, ctx=world["ctx"], bundle=world["bundle"],
                           store=False)


def test_budget_skips_arms_that_do_not_fit(world, tmp_path, cached_projection):
    def project(conn, sid, at, **kw):
        time.sleep(0.4)
        return cached_projection
    t = time.monotonic()
    res = _run(world, tmp_path, arms=["+tm", "+lf", "+rs", "+pq"], project=project, budget=1.0)
    took = time.monotonic() - t
    st = {a: v["status"] for a, v in res["arms"].items()}
    assert st[SH.CHAMPION] == "ok"
    assert sum(s == "ok" for s in st.values()) == 2           # the champion + one arm
    assert sum(s == "skipped" for s in st.values()) == 3
    assert all("budget" in v["reason"] for v in res["arms"].values() if v["status"] == "skipped")
    assert took < 1.3


def test_budget_interrupts_a_running_arm(world, tmp_path, cached_projection):
    snap = SH.switch_state()

    def project(conn, sid, at, **kw):
        if P.MODEL_VERSION != snap["MODEL_VERSION"]:
            time.sleep(10)                                  # a hung arm
        return cached_projection
    t = time.monotonic()
    res = _run(world, tmp_path, arms=["+lf"], project=project, budget=0.8)
    took = time.monotonic() - t
    assert res["arms"]["+lf"]["status"] == "timeout"
    assert took < 1.5
    assert SH.switch_state() == snap
    assert res["seconds_total"] < 1.5


def test_budget_and_sims_defaults(monkeypatch):
    assert SH.budget_s() == SH.BUDGET_S == 240.0
    monkeypatch.setenv(SH.BUDGET_ENV, "45")
    assert SH.budget_s() == 45.0
    monkeypatch.setenv(SH.SIMS_ENV, "5000")
    assert SH.shadow_sims() == 5000


# ============================================================================== grading

def test_grade_end_to_end(world, shadow_run, tmp_path):
    root = shadow_run["root"]
    rep = SH.grade(world["conn"], world["day"], out_root=root, data_dir=tmp_path)
    assert {a["arm"] for a in rep["arms"]} == set(ALL_ARMS)
    assert rep["nights"] == [world["day"].isoformat()]
    by = {a["arm"]: a for a in rep["arms"]}
    # +tm at 13:30 has no rows (the world's run is at 12:00 ET, "adhoc"); +lf runs "last"
    assert by["+lf"]["primary"]["n"] > 0 and by["+lf"]["primary"]["nights"] == 1
    assert by["+lf"]["decision"] == "pending" and "1 of 21 nights" in by["+lf"]["reason"]
    assert by["+tm"]["primary"]["n"] == 0
    assert (root / world["day"].isoformat() / "graded.csv.gz").exists()
    assert json.loads((tmp_path / "reports" / "shadows" / SH.REPORT_NAME).read_text())["as_of"] \
        == world["day"].isoformat()
    # the quantile CRPS of the champion matches the sample CRPS of the full sims
    rows = SH.season_rows(world["conn"], world["day"], out_root=root)
    ch = _champion(world, seed=shadow_run["res"]["seed"])
    from courtside.proj.dfs import grade as G
    box = G.load_box(world["conn"], world["games"])
    pid = ch["pid"]
    y = np.array([box[int(p)]["dk"] if int(p) in box else 0.0 for p in pid])
    full = MT.crps_samples_matrix(ch["fp"], y)
    r = rows[rows["arm"] == "+lf"].set_index("player_id")["crps_c"]
    exact = pd.Series(full, index=pid.astype(np.int64))[r.index]
    assert np.allclose(r.to_numpy(), exact.to_numpy(), rtol=0.03, atol=0.02)


def test_old_cache_gains_the_e4_columns_only(world, shadow_run, tmp_path):
    """A night cached before EXTRA_COLS existed: the new columns are merged in, every cached
    number stays as it was, and the reported-only block appears."""
    import os
    root = shadow_run["root"]
    SH.grade(world["conn"], world["day"], out_root=root, data_dir=tmp_path)
    cache = root / world["day"].isoformat() / "graded.csv.gz"
    new = pd.read_csv(cache, dtype={"game_id": str, "run": str})
    assert set(SH.EXTRA_COLS) <= set(new.columns)
    sal = new["salary"].dropna()
    assert len(sal) > 0.5 * len(new) and (sal == 5000).all()
    old = new.drop(columns=list(SH.EXTRA_COLS))
    old.to_csv(cache, index=False, compression="gzip")
    t = time.time() + 5
    os.utime(cache, (t, t))
    rows = SH.season_rows(world["conn"], world["day"], out_root=root)
    pd.testing.assert_frame_equal(rows[old.columns].reset_index(drop=True),
                                  old.reset_index(drop=True))
    assert np.allclose(rows["dk_c"], new["dk_c"]) and np.allclose(rows["dk_a"], new["dk_a"])
    again = pd.read_csv(cache, dtype={"game_id": str, "run": str})
    assert set(SH.EXTRA_COLS) <= set(again.columns)        # the cache was rewritten
    rep = SH.grade(world["conn"], world["day"], out_root=root, data_dir=tmp_path)
    lf = {a["arm"]: a for a in rep["arms"]}["+lf"]
    assert lf["reported_only"]["n"] > 0 and "dk_crps_dfs" in lf["reported_only"]


def _rows(n_nights, n_players, d, *, tier_shift=None, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_nights):
        for p in range(n_players):
            base = rng.gamma(3, 2)
            tier = ("star", "role", "bench")[p % 3]
            dd = d + rng.normal(0, 0.02) + (tier_shift or {}).get(tier, 0.0)
            out.append({"day": f"2026-11-{k + 1:02d}", "run": "1330", "slot": "13:30",
                        "arm": "+x", "game_id": f"g{k}", "player_id": p, "tier": tier,
                        "played": 1, "y": 10.0, "amin": 20.0, "emin_c": 20.0, "emin_a": 20.0,
                        "crps_c": base, "crps_a": base + dd, "ae_c": 1.0, "ae_a": 1.0,
                        "mae_c": 1.0, "mae_a": 1.0, "ll_c": 0.3, "ll_a": 0.3, "changed": 1})
    return pd.DataFrame(out)


def _entry(**kw):
    e = {"arm": "+x", "metric": "dk_crps", "population": "emin10", "run": "13:30",
         "bar": {"direction": "lower", "z": -2.0, "tier_z": 2.0}, "min_nights": 14,
         "max_nights": 60, "status": "active"}
    e.update(kw)
    return e


def test_pass_fail_pending_rules():
    better = _rows(20, 30, -0.05)
    assert SH.arm_report(better, _entry())["decision"] == "pass"
    assert SH.arm_report(better[better["day"] < "2026-11-10"], _entry())["decision"] == "pending"
    worse = _rows(20, 30, +0.05)
    a = SH.arm_report(worse, _entry())
    assert a["decision"] == "fail" and "clearly worse" in a["reason"]
    flat = _rows(20, 30, 0.0)
    assert SH.arm_report(flat, _entry())["decision"] == "pending"
    assert SH.arm_report(flat, _entry(max_nights=20))["decision"] == "fail"
    # one tier clearly worse blocks a pass
    tier_bad = _rows(20, 30, -0.05, tier_shift={"bench": +0.07})
    a = SH.arm_report(tier_bad, _entry())
    assert a["primary"]["z"] < -2 and a["decision"] == "pending" and "tier bench" in a["reason"]
    # not_worse: pass when not clearly worse, fail otherwise
    nw = _entry(bar={"direction": "not_worse", "z": 2.0, "tier_z": 2.0})
    assert SH.arm_report(flat, nw)["decision"] == "pass"
    assert SH.arm_report(worse, nw)["decision"] == "fail"
    # min_rows and a failing guard
    assert "rows" in SH.arm_report(better, _entry(min_rows=10_000))["reason"]
    g = _entry(guard=[{"metric": "dk_crps", "population": "emin10", "z": 2.0}], metric="dk_mae")
    noise = np.random.default_rng(1).normal(0, 0.05, len(worse))
    worse_g = worse.assign(ae_a=worse["ae_a"] - 0.5 + noise)
    a = SH.arm_report(worse_g, g)
    assert a["primary"]["mean"] < 0 and a["decision"] != "pass" and "dk_crps" in a["reason"]
    # the window cuts nights
    a = SH.arm_report(better, _entry(window={"until": "2026-11-05"}))
    assert a["primary"]["nights"] == 5 and a["decision"] == "pending"


def test_paired_is_two_way_clustered():
    df = _rows(6, 12, -0.03, seed=3)
    p = SH.paired(df, "dk_crps")
    ref = MT.paired_cluster2((df["crps_a"] - df["crps_c"]).to_numpy(), df["day"].to_numpy(),
                             df["player_id"].to_numpy())
    assert p["mean"] == pytest.approx(ref["mean"]) and p["se"] == pytest.approx(ref["se"])
    assert p["z"] == pytest.approx(ref["mean"] / ref["se"])
    same = df.assign(crps_a=df["crps_c"])
    assert SH.paired(same, "dk_crps")["z"] == 0.0


def test_run_selection():
    df = pd.concat([_rows(2, 3, 0.0).assign(run="1000", slot="10:00"),
                    _rows(2, 3, 0.0).assign(run="1330", slot="13:30"),
                    _rows(2, 3, 0.0).assign(run="1345", slot="13:30"),
                    _rows(2, 3, 0.0).assign(run="2200", slot="22:00")])
    assert set(SH._select_run(df, "13:30")["run"]) == {"1330"}
    assert set(SH._select_run(df, "last")["run"]) == {"2200"}
    assert set(SH._select_run(df, "10:00")["run"]) == {"1000"}


def test_postgame_step_never_raises(world, monkeypatch):
    from courtside import collect
    (config.data_dir() / "proj" / "shadows").mkdir(parents=True)

    def boom(*a, **k):
        raise RuntimeError("grader broke")
    monkeypatch.setattr(SH, "grade", boom)
    out = collect._shadow_grade_step(world["conn"], world["day"])
    assert out["status"] == "error" and "grader broke" in out["error"]


# ============================================================================== registry, slots

def test_registry_is_complete_and_preregistered():
    reg = SH.load_registry()
    arms = {a["arm"]: a for a in reg["arms"]}
    assert set(arms) == set(ALL_ARMS) == set(SH.SWITCHES)
    for a in arms.values():
        assert a["hypothesis"] and a["switch"].startswith("COURTSIDE_")
        assert a["metric"] in ("dk_crps", "dk_mae", "min_mae", "pplay_ll")
        assert a["population"] in ("emin10", "changed", "all")
        assert a["run"] == "last" or a["run"] in SH.PREGAME_TIMES
        assert a["bar"]["direction"] in ("lower", "not_worse")
        assert {"z", "tier_z"} <= set(a["bar"]) and a["min_nights"] >= 1
        assert a["status"] in ("active", "paused", "shipped", "retired")
    assert arms["+tr"]["run"] == "13:30" and arms["+tr"]["metric"] == "dk_crps"
    assert arms["+tr"]["bar"] == {"direction": "lower", "z": -2.0, "tier_z": 2.0}
    assert arms["+tr"]["min_nights"] == 14
    # +tr and +ep shipped 2026-10-24 (now in the champion): registered, no longer run
    assert arms["+tr"]["status"] == arms["+ep"]["status"] == "shipped"
    assert [a["arm"] for a in SH.select_arms(reg)] == [a for a in ALL_ARMS if a not in ("+tr", "+ep")]


def test_arm_list_override(monkeypatch):
    reg = SH.load_registry()
    monkeypatch.setenv(SH.ARMS_ENV, "tr, +ep")
    assert [a["arm"] for a in SH.select_arms(reg)] == ["+tr", "+ep"]
    assert [a["arm"] for a in SH.select_arms(reg, ["pq"])] == ["+pq"]


def test_every_setter_flips_and_restores():
    snap = SH.switch_state()
    for tag in SH.SWITCHES:
        with SH.arm_on(tag) as v:
            assert SH.switch_state()[tag] is True and tag in v
        assert SH.switch_state() == snap


def test_slots():
    et = config.NBA_TZ
    assert SH.slot_of(datetime(2026, 10, 23, 13, 33, tzinfo=et)) == "13:30"
    assert SH.slot_of(datetime(2026, 10, 23, 10, 0, tzinfo=et)) == "10:00"
    assert SH.slot_of(datetime(2026, 12, 1, 10, 49, tzinfo=et)) == "10:00"
    assert SH.slot_of(datetime(2026, 10, 23, 12, 0, tzinfo=et)) == "adhoc"
    assert SH.slot_of(datetime(2026, 10, 23, 18, 35, tzinfo=et)) == "18:30"
    assert SH.run_key(datetime(2026, 10, 23, 17, 33, tzinfo=timezone.utc)) == "1333"


# ============================================================================== surfaces

def test_health_dfs_shadows_card(data_dir):
    warnings.filterwarnings("ignore", message="Using `httpx`")
    from fastapi.testclient import TestClient
    from courtside.web import health_dfs
    from courtside.web.app import create_app
    d = data_dir.parent / "web"
    d.mkdir(parents=True)
    conn = db.connect(d / "courtside.db")
    assert health_dfs.build(conn, date(2026, 10, 25))["shadows"] == {}
    better = _rows(20, 30, -0.05)
    rep = {"as_of": "2026-11-20", "nights": sorted(better["day"].unique()),
           "arms": [SH.arm_report(better.assign(arm="+tr"),
                                  _entry(arm="+tr", switch="COURTSIDE_TAIL_REMAP")),
                    SH.arm_report(better.iloc[:0], _entry(arm="+ep", min_nights=14))],
           "runs": [{"day": "2026-11-20", "slot": "13:30", "status": "ok", "seconds_total": 41.0,
                     "arms": {"champion": {"status": "ok"}, "+tr": {"status": "ok"}}}]}
    (d / "reports" / "shadows").mkdir(parents=True)
    (d / "reports" / "shadows" / SH.REPORT_NAME).write_text(json.dumps(rep, default=str))
    vm = health_dfs.build(conn, date(2026, 11, 21))["shadows"]
    assert [r["status"] for r in vm["rows"]] == ["Pass", "Pending"] and vm["nights"] == 20
    conn.close()
    r = TestClient(create_app(d)).get("/health/dfs")
    assert r.status_code == 200
    assert 'id="shadows"' in r.text and "+tr" in r.text and "pill-healthy" in r.text
    assert "None" not in r.text.split('id="shadows"')[1].split("</div>")[0]


def test_job_wiring():
    root = Path(__file__).resolve().parents[1]
    sh = (root / "deploy" / "run-job.sh").read_text()
    i = sh.index('if [ "$job" = "pregame" ]; then')
    block = sh[i:sh.index("fi", sh.index("not fatal", i))]
    assert "nice -n 10" in block and "courtside proj shadows run" in block
    assert "dfs tick" not in block
    from courtside import collect
    from courtside.proj import cli as PC
    assert "courtside.proj.shadows" in PC.PROJ_MODULES
    assert "shadow" not in " ".join(collect.PREGAME_STEPS)       # never inside the pregame job
    src = (root / "src" / "courtside" / "collect.py").read_text()
    assert 'run.step("shadow_grade", _shadow_grade_step, conn, day)' in src
