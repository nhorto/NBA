"""Role-aware blend (COURTSIDE_ROLE_BLEND, `+rb`; `proj/role_blend.py`, IMPROVEMENTS.md P1 + D1).

OFF by default and nothing of it runs when off (same sims, same frames, same versions, same
truth_sims); ON every played sim of a player moves by beta(r) x (L10 - mu_played), the draws are
unchanged (common random numbers), the version gets `+rb` and the contest builder's flat
TRUTH_BETA shift is skipped (tau kept)."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC
from courtside.proj import role_blend as RB
from courtside.proj.dfs import portfolio as PF

DAY = "2024-01-10"


@pytest.fixture
def restore():
    state = (RB.ROLE_BLEND, RC.RECON_RESERVE)
    yield
    P.set_recon_reserve(state[1])
    P.set_role_blend(state[0])


def _bundle(mp, sp):
    return SimpleNamespace(copula=C.default_copula_params("2023-24"), minutes=mp, state=sp,
                           season="2023-24", copula_v1=None)


def _with_inputs(spec, l10_of, l10min_of, missed_of=lambda i: 0):
    for side in (spec.home, spec.away):
        pl = side.players.copy()
        n = len(pl)
        pl["rb_l10"] = [l10_of(i) for i in range(n)]
        pl["rb_l10min"] = [l10min_of(i) for i in range(n)]
        pl["rb_missed"] = [missed_of(i) for i in range(n)]
        side.players = pl
    return spec


# ------------------------------------------------------------------------------ switch

def test_default_off_and_env_flag(monkeypatch):
    assert RB.ROLE_BLEND_DEFAULT is True                              # shipped 2026-10-23
    monkeypatch.setenv(RB.ROLE_BLEND_ENV, "1")
    assert RB._flag(RB.ROLE_BLEND_ENV, False) is True
    monkeypatch.setenv(RB.ROLE_BLEND_ENV, "off")
    assert RB._flag(RB.ROLE_BLEND_ENV, True) is False
    monkeypatch.delenv(RB.ROLE_BLEND_ENV)
    assert RB._flag(RB.ROLE_BLEND_ENV, RB.ROLE_BLEND_DEFAULT) is True


def test_version_tag_composes(restore):
    P.set_recon_reserve(False)
    P.set_role_blend(False)
    off_vs, off_mv = P._VS, P.MODEL_VERSION
    assert "+rb" not in off_mv
    P.set_role_blend(True)
    assert P._VS == off_vs and P.MODEL_VERSION == off_mv + "+rb"     # model version only
    P.set_recon_reserve(True)
    assert P.MODEL_VERSION == off_mv + "+rr+rb"
    P.set_role_blend(False)
    assert P.MODEL_VERSION == off_mv + "+rr"
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                  # seeds unchanged


# ------------------------------------------------------------------------------ the shift

def test_bucket_and_delta():
    planned = np.array([20.0, 23.0, 25.0, 28.0, 40.0, 10.0, 10.0, 25.0])
    l10min = np.array([30.0, 25.0, 25.0, 25.0, 25.0, np.nan, 0.0, 25.0])
    b = RB.bucket_of(planned, l10min)
    assert list(b) == ["<0.8", "0.8-0.95", "0.95-1.05", "1.05-1.25", ">1.25", "nohist",
                       "nohist", "0.95-1.05"]
    beta = {"<0.8": 0.0, "0.8-0.95": 0.16, "0.95-1.05": 0.12, "1.05-1.25": 0.09, ">1.25": 0.07}
    mu = np.full(8, 30.0)
    l10 = np.array([20.0, 20.0, 40.0, 20.0, 20.0, 20.0, 20.0, np.nan])
    missed = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    d = RB.delta(mu, planned, l10, l10min, missed, beta)
    assert d == pytest.approx([0.0, -1.6, 1.2, -0.9, -0.7, 0.0, 0.0, 0.0])
    d2 = RB.delta(mu, planned, l10, l10min, np.array([3] * 8), beta)    # returning -> 0
    assert (d2 == 0).all()
    assert RB.bucket_of(np.array([0.8 * 25, 1.25 * 25]), np.array([25.0, 25.0])).tolist() == \
        ["0.8-0.95", ">1.25"]                                          # edges: right-open


def test_run_games_off_never_runs(game, restore, monkeypatch):  # noqa: F811
    spec, mp, sp = _spec(game, p_half=[1, 3])
    b = _bundle(mp, sp)
    P.set_role_blend(False)
    a = P.run_games([spec], b, 2000, 7, P._as_date(DAY))
    monkeypatch.setattr(RB, "apply", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    spec2 = _with_inputs(_spec(game, p_half=[1, 3])[0], lambda i: 50.0, lambda i: 10.0)
    c = P.run_games([spec2], b, 2000, 7, P._as_date(DAY))   # rb_* columns present, still OFF
    assert np.array_equal(a.sims[0].fp, c.sims[0].fp)
    pd.testing.assert_frame_equal(a.players, c.players, check_exact=True)
    pd.testing.assert_frame_equal(a.stats, c.stats, check_exact=True)
    assert "role_blend" not in a.meta and "rb_shift" not in a.players


def test_run_games_on_shifts_played_sims_only(game, restore, monkeypatch):  # noqa: F811
    beta = {"<0.8": 0.0, "0.8-0.95": 0.16, "0.95-1.05": 0.12, "1.05-1.25": 0.09, ">1.25": 0.07}
    monkeypatch.setattr(RB, "betas", lambda season: beta)
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    b = _bundle(mp, sp)
    P.set_role_blend(False)
    off = P.run_games([spec], b, 3000, 11, P._as_date(DAY))
    g0 = off.sims[0]
    played = g0.minutes > 0
    npl = played.sum(0)
    mu_p = np.where(npl > 0, g0.fp.astype(float).sum(0) / np.maximum(npl, 1), np.nan)
    plan = np.where(npl > 0, g0.minutes.astype(float).sum(0) / np.maximum(npl, 1), np.nan)
    # inputs: L10 = mu + 10, last-10 minutes chosen to put players in every bucket; player 2 of
    # each side is returning, player 4 has no history
    ratios = [0.7, 0.9, 1.0, 1.1, 1.4]
    pid = g0.player_id
    idx = {int(p): j for j, p in enumerate(pid)}

    def col(side, f):
        return [f(idx[int(p)], i) for i, p in enumerate(side.players["player_id"])]

    for side in (spec.home, spec.away):
        side.players = side.players.copy()
        side.players["rb_l10"] = col(side, lambda j, i: (mu_p[j] + 10.0) if np.isfinite(mu_p[j])
                                     else 20.0)
        side.players["rb_l10min"] = col(side, lambda j, i: np.nan if i == 4 else
                                        (plan[j] / ratios[i % 5] if np.isfinite(plan[j]) else 20.0))
        side.players["rb_missed"] = col(side, lambda j, i: 3 if i == 2 else 0)
    P.set_role_blend(True)
    on = P.run_games([spec], b, 3000, 11, P._as_date(DAY))
    g1 = on.sims[0]
    assert np.array_equal(g0.minutes, g1.minutes)                    # same draws
    assert (g1.fp[~played] == 0).all()                               # DNP sims stay 0
    dl = (g1.fp.astype(float) - g0.fp.astype(float))
    sh = g1.info["rb_shift"].to_numpy(float)
    for side in (spec.home, spec.away):
        for i, p in enumerate(side.players["player_id"]):
            j = idx[int(p)]
            if npl[j] == 0:
                assert sh[j] == 0
                continue
            exp = 0.0 if i in (2, 4) else beta[RB.BUCKETS[i % 5]] * 10.0
            assert sh[j] == pytest.approx(exp, abs=1e-4)
            assert np.allclose(dl[played[:, j], j], exp, atol=2e-3)
    # the summary carries mu*: mean over all sims moves by delta x P(play)
    m0 = off.players.set_index("player_id")["dk_mean"]
    m1 = on.players.set_index("player_id")["dk_mean"]
    pp = off.players.set_index("player_id")["p_play"]
    shs = pd.Series(sh, index=pid)
    assert np.allclose((m1 - m0).to_numpy(), (shs.reindex(m0.index) * pp).to_numpy(), atol=2e-3)
    assert on.meta["role_blend"]["shifted"] == int((sh != 0).sum()) > 0


def test_on_with_zero_beta_draws_exactly_off(game, restore, monkeypatch):  # noqa: F811
    monkeypatch.setattr(RB, "betas", lambda season: {k: 0.0 for k in RB.BUCKETS})
    spec, mp, sp = _spec(game, p_half=[2])
    spec = _with_inputs(spec, lambda i: 40.0, lambda i: 20.0)
    b = _bundle(mp, sp)
    P.set_role_blend(False)
    a = P.run_games([spec], b, 2000, 5, P._as_date(DAY))
    P.set_role_blend(True)
    c = P.run_games([spec], b, 2000, 5, P._as_date(DAY))
    assert np.array_equal(a.sims[0].fp, c.sims[0].fp)
    assert np.array_equal(a.sims[0].minutes, c.sims[0].minutes)
    cols = [x for x in a.players.columns]
    pd.testing.assert_frame_equal(a.players[cols], c.players[cols], check_exact=True)


# ------------------------------------------------------------------------------ history

def _box_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE player_games (player_id INT, team TEXT, season TEXT, season_type "
                 "TEXT, game_date TEXT, min REAL, pts INT, fg3m INT, reb INT, ast INT, stl INT, "
                 "blk INT, tov INT)")
    rows = []
    # team AAA plays every day 2023-11-01..2023-11-20; player 1 plays 11-01..11-12, then sits
    for d in range(1, 21):
        ds = f"2023-11-{d:02d}"
        rows.append((9, "AAA", "2023-24", "Regular Season", ds, 30, 20, 0, 5, 5, 0, 0, 0))
        if d <= 12:
            rows.append((1, "AAA", "2023-24", "Regular Season", ds, 10 + d, d, 0, 0, 0, 0, 0, 0))
    # player 2: 3 games two seasons ago (minutes count, DK does not), 2 games last season
    for ds, s in (("2021-12-01", "2021-22"), ("2021-12-02", "2021-22"), ("2021-12-03", "2021-22"),
                  ("2022-12-01", "2022-23"), ("2022-12-02", "2022-23")):
        rows.append((2, "AAA", s, "Regular Season", ds, 20, 10, 0, 0, 0, 0, 0, 0))
    conn.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return conn


def test_role_history_strictly_before_and_returning():
    h = RB.RoleHistory(_box_db(), "2023-24")
    l10, l10m, miss = h.one(1, "AAA", "2023-11-13")
    assert l10m == pytest.approx(np.mean([10 + d for d in range(3, 13)]))
    assert l10 == pytest.approx(np.mean(range(3, 13)))               # pts only
    assert miss == 0
    l10b, _, miss = h.one(1, "AAA", "2023-11-16")                    # missed 13, 14, 15
    assert miss == 3 and l10b == pytest.approx(l10)
    assert h.one(1, "AAA", "2023-11-05")[1] == pytest.approx(np.mean([11, 12, 13, 14]))
    l10, l10m, miss = h.one(2, "AAA", "2023-11-05")                  # last game last season
    assert l10m == pytest.approx(20.0) and l10 == pytest.approx(10.0)
    assert miss == 0                                                 # not returning (new season)
    assert np.isnan(h.one(77, "AAA", "2023-11-05")[0])
    a, b, c = h.lookup([1, 77], ["AAA", "AAA"], P._as_date("2023-11-13"))
    assert a[0] == pytest.approx(l10b) and np.isnan(a[1]) and c.tolist() == [0, 0]


# ------------------------------------------------------------------------------ params / fit

def test_params_walk_forward_and_fallback():
    par = json.loads(RB.RB_PATH.read_text())
    assert par["seasons"]
    for s, e in par["seasons"].items():
        assert e["train"] and all(t.split(":")[1] < s for t in e["train"])
        assert set(e["beta"]) == set(RB.BUCKETS)
        assert all(0.0 <= v <= 0.30 for v in e["beta"].values())
        assert RB.betas(s) == pytest.approx(e["beta"])
    last = max(par["seasons"])
    assert RB.betas("2099-00") == pytest.approx(par["seasons"][last]["beta"])
    assert RB.betas("1990-91") == RB.DEFAULT_BETA


def _fit_frame(n, beta_true, rng):
    """Rows whose outcome is mu + beta_true * (L10 - mu) + noise; quantiles of N(mu, 6)."""
    mu = rng.uniform(15, 40, n)
    gap = rng.normal(0, 6, n)
    y = mu + beta_true * gap + rng.normal(0, 6, n)
    z = np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816])
    q = mu[:, None] + 6.0 * z[None, :]
    fr = pd.DataFrame({"mu_p": mu, "planned": 30.0, "l10": mu + gap, "l10min": 30.0,
                       "missed": 0, "y": y})
    for k, c in enumerate(RB.QCOLS):
        fr[c] = q[:, k]
    return fr


def test_fit_recovers_shift_and_min_rows():
    rng = np.random.default_rng(3)
    fr = _fit_frame(6000, 0.2, rng)
    cur = RB.loss_curves(fr)
    assert cur["0.95-1.05"]["n"] == int((fr["l10"] > 0).sum()) > 5900   # L10 <= 0: no shift
    assert cur["<0.8"]["n"] == 0
    b = RB.fit_betas(cur)
    assert b["0.95-1.05"] == pytest.approx(0.2, abs=0.06)
    assert b["<0.8"] == 0.0                                          # < MIN_ROWS -> 0 / prior
    assert RB.fit_betas(cur, prior={"<0.8": 0.05})["<0.8"] == 0.05
    both = RB.add_curves(cur, cur)
    assert both["0.95-1.05"]["n"] == 2 * cur["0.95-1.05"]["n"]
    assert np.allclose(both["0.95-1.05"]["loss"], 2 * np.array(cur["0.95-1.05"]["loss"]))


def test_refit_weekly_adds_live_rows(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    train = RB.loss_curves(_fit_frame(3000, 0.0, rng))
    path = tmp_path / "rb.json"
    path.write_text(json.dumps({"seasons": {"2026-27": {
        "beta": RB.fit_betas(train), "train": ["avail:2024-25"], "curves": train}}}))
    live = _fit_frame(30000, 0.3, rng).assign(date="2026-11-01")
    monkeypatch.setattr(RB, "live_rows", lambda conn, season, until=None: live)
    e = RB.refit_weekly(None, "2026-27", path=path, until="2026-11-02")
    assert e["beta"]["0.95-1.05"] > e["refit"]["beta_before"]["0.95-1.05"]
    assert e["refit"]["n_live"] == 30000 and e["curves"] == train  # training curves kept
    saved = json.loads(path.read_text())["seasons"]["2026-27"]
    assert saved["beta"] == e["beta"]


# ------------------------------------------------------------------------------ truth_sims

def test_truth_sims_skips_beta_when_blended(restore):
    rng = np.random.default_rng(0)
    fp = rng.uniform(0, 40, (500, 3)).astype(np.float32)
    fp[:, 2] = 0.0
    inp = pd.DataFrame({"pid": ["a", "b", "c"], "fppg": [10.0, 50.0, 20.0]})
    P.set_role_blend(False)
    x0, i0 = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.18, 0.18, np.random.default_rng(1))
    x1, i1 = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.18, 0.18, np.random.default_rng(1),
                           blended=False)
    assert np.array_equal(x0, x1) and i0 == i1 and "role_blend" not in i0   # OFF unchanged
    xb, ib = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.18, 0.18, np.random.default_rng(1),
                           blended=True)
    xz, iz = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.0, 0.18, np.random.default_rng(1))
    assert np.array_equal(xb, xz) and ib["beta"] == 0.0 and ib["tau"] == 0.18
    assert ib["role_blend"] is True
    P.set_role_blend(True)                                            # None -> the switch
    xs, _ = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.18, 0.18, np.random.default_rng(1))
    assert np.array_equal(xs, xz)
    same, _ = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.18, 0.0, np.random.default_rng(1))
    assert same is fp                                                 # nothing left to do


def test_run_blended_reads_model_version():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE dfs_runs (run_id TEXT, model_version TEXT)")
    conn.executemany("INSERT INTO dfs_runs VALUES (?,?)", [("r1", "pathA-v0.3+mf+p2+rb"),
                                                           ("r2", "pathA-v0.3+mf+p2")])
    assert RB.run_blended(conn, "r1") is True
    assert RB.run_blended(conn, "r2") is False
    assert RB.run_blended(conn, "r3") is None and RB.run_blended(conn, None) is None
