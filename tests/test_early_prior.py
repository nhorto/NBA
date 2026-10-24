"""Early-season prior (COURTSIDE_EARLY_PRIOR, `+ep`; `proj/early_prior.py`, IMPROVEMENTS P3).

OFF (the default) never reaches the `+ep` code: `state.rate_frame` / `walk_forward` without `ep`
are the base path, the bundle JSON and version strings are the ones before the switch. ON with
neutral params reproduces the base rates; ON only changes r_<stat> (n_, p_, mpg, tier as base);
preseason games count only within their own season and strictly before the day; scrimmages and
games against non-NBA teams are dropped; the fit recovers informative preseason games."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import early_prior as EP
from courtside.proj import pipeline as P
from courtside.proj import state as ST

SEASONS = {"2022-23": date(2022, 10, 20), "2023-24": date(2023, 10, 20),
           "2024-25": date(2024, 10, 20)}
N_GAMES = 50
N_PLAYERS = 30
NEW = 99            # no regular-season game ever, preseason games of 2024-25 only
PRE_LEVEL = 0.9     # preseason per-minute production relative to the season's true rate


def _league(c: sqlite3.Connection, seed: int = 0) -> None:
    """Two teams, 30 players, per-(player, season) true rates (so last season misleads and the
    preseason is informative); 5 preseason games per season at PRE_LEVEL x the true rate."""
    rng = np.random.default_rng(seed)
    pids = list(range(1, N_PLAYERS + 1))
    mins = {p: float(rng.integers(14, 34)) for p in pids}
    base = {"pts": 0.45, "fg3m": 0.06, "oreb": 0.05, "dreb": 0.13, "ast": 0.1, "stl": 0.03,
            "blk": 0.02, "tov": 0.06, "fga": 0.38, "fta": 0.1, "pf": 0.09}
    rows, pre, tg, gid = [], [], [], 0

    def line(p, m, true, lvl):
        x = {k: int(rng.poisson(true[k] * lvl * m)) for k in base}
        return [m, 0, x["fga"], x["fg3m"], 0, 0, x["fta"], x["oreb"], x["dreb"],
                x["oreb"] + x["dreb"], x["ast"], x["stl"], x["blk"], x["tov"], x["pf"], x["pts"]]

    for season, start in SEASONS.items():
        true = {p: {k: v * float(np.exp(rng.normal(0, 0.3))) for k, v in base.items()}
                for p in pids}
        true[NEW] = {k: v * 1.5 for k, v in base.items()}
        for k in range(5):                                      # preseason, before the start
            day = start - timedelta(days=12 - 2 * k)
            g = f"PRE{season[:4]}{k}"
            for p in pids + ([NEW] if season == "2024-25" else []):
                pre.append([g, p, f"P{p}", "AAA" if p <= 15 or p == NEW else "BBB", season,
                            "Pre Season", day.isoformat()]
                           + line(p, float(np.clip(rng.normal(mins.get(p, 20) * 0.8, 3), 4, 36)),
                                  true[p], PRE_LEVEL) + [0.0, 0, 0])
        for k in range(N_GAMES):
            day = start + timedelta(days=2 * k)
            gid += 1
            g = f"G{gid:05d}"
            for t, o, h in (("AAA", "BBB", 1), ("BBB", "AAA", 0)):
                tg.append((g, t, season, "Regular Season", day.isoformat(), o, h))
            for p in pids:
                m = float(np.clip(rng.normal(mins[p], 3), 6, 40))
                rows.append([g, p, f"P{p}", "AAA" if p <= 15 else "BBB", season,
                             "Regular Season", day.isoformat()] + line(p, m, true[p], 1.0)
                            + [0.0])
    c.executemany(
        "INSERT INTO player_games(game_id, player_id, player_name, team, season, season_type, "
        "game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, "
        "pts, plus_minus) VALUES (" + ",".join("?" * 24) + ")", rows)
    c.executemany(
        "INSERT INTO preseason_player_games(game_id, player_id, player_name, team, season, "
        "season_type, game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, "
        "blk, tov, pf, pts, plus_minus, scrimmage, vs_non_nba) VALUES ("
        + ",".join("?" * 26) + ")", pre)
    c.executemany("INSERT INTO team_games(game_id, team, season, season_type, game_date, opp, "
                  "is_home) VALUES (?,?,?,?,?,?,?)", tg)
    pos = ["G", "F", "C"]
    for season in SEASONS:
        c.executemany("INSERT INTO team_rosters(season, team_id, team, player_id, position, "
                      "birth_date) VALUES (?,?,?,?,?,?)",
                      [(season, 1 if p <= 15 else 2, "AAA" if p <= 15 else "BBB", p, pos[p % 3],
                        f"{1990 + p % 12}-03-01") for p in pids])
    c.executemany("INSERT INTO player_index(player_id, position, draft_number) VALUES (?,?,?)",
                  [(p, pos[p % 3], p if p < 25 else None) for p in pids + [NEW]])
    c.commit()


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _league(c)
    return c


def _q(pids, day):
    return pd.DataFrame({"player_id": list(pids), "game_date": day})


def _ep(params: dict | None = None, conn=None, data=None) -> EP.EPInput:
    return EP.EPInput(params or EP.neutral_params(), data or EP.load_data(conn))


def _hand_params(w=0.6, v=0.9) -> dict:
    p = EP.neutral_params()
    p["w"] = {s: w for s in ST.STATS}
    p["v"] = {s: v for s in ST.STATS}
    return p


@pytest.fixture
def restore_switches():
    keep = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX,
            P.BLOWOUT, EP.EARLY_PRIOR)
    yield
    P.set_copula(keep[2])
    P.set_pplay(keep[3])
    P.set_minratio(keep[4])
    P.set_preseason(keep[5])
    P.set_minfix(keep[6])
    P.set_blowout(keep[7])
    P.set_early_prior(keep[8])
    P.set_context(keep[0], keep[1])


# ============================================================================== switch

def test_default_on_and_env_flag(monkeypatch):
    # shipped 2026-10-24; tests/conftest.py pins it off for the suite
    assert EP.EARLY_PRIOR_DEFAULT is True and EP.EARLY_PRIOR is False
    for v, exp in (("1", True), ("on", True), ("0", False), ("off", False), ("", True)):
        monkeypatch.setenv(EP.EARLY_PRIOR_ENV, v)
        assert EP._flag(EP.EARLY_PRIOR_ENV, EP.EARLY_PRIOR_DEFAULT) is exp


def test_versions_compose(restore_switches):
    P.set_early_prior(False)
    for on in (False, True):
        for mf in (False, True):
            for mr in (False, True):
                for cop in ("v0", "v1"):
                    vs, mv = P._versions(on, False, cop, "v2", mr, False, mf, False)
                    assert "_ep" not in vs and "+ep" not in mv
    P.set_early_prior(True)
    vs, mv = P._versions(True, False, "v1", "v2", True, False, True, True)
    assert vs == "_v1_cx_mf_bo_ep_mr_c1"
    assert mv == "pathA-v0.3+mf+bo+ep+mr+c1+p2"
    vs, mv = P._versions(True, False, "v0", "v2", False, False, True, False)
    assert vs == "_v1_cx_mf_ep" and mv.startswith("pathA-v0.3+mf+ep+p2")
    P.set_context(True)
    P.set_minfix(True)
    assert P._VS.endswith("_mf_ep") and "+mf+ep" in P.MODEL_VERSION
    P.set_early_prior(False)
    assert "_ep" not in P._VS and "+ep" not in P.MODEL_VERSION


def test_bundle_json_off_unchanged():
    b = P.Bundle(season="2024-25", minutes=P.M.default_params("2024-25"),
                 state=ST.default_params(), copula=P.C.default_copula_params("2024-25"))
    assert "early_prior" not in b.to_json()
    assert not EP.has_ep(b.state)


# ============================================================================== OFF = base path

def test_off_never_reaches_ep(conn, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("early_prior.rates called with the switch off")
    monkeypatch.setattr(EP, "rates", boom)
    q = _q(range(1, N_PLAYERS + 1), "2024-11-15")
    a = ST.rate_frame(conn, q, ST.default_params())
    b = ST.rate_frame(conn, q, ST.default_params(), ep=None)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    w0 = ST.walk_forward(conn, "2024-25", ST.default_params())
    w1 = ST.walk_forward(conn, "2024-25", ST.default_params(), ep=None)
    pd.testing.assert_frame_equal(w0, w1, check_exact=True)


def test_ep_input_none_unless_on_and_fitted(conn, restore_switches):
    class Ctx:
        pass
    sp = dict(ST.default_params())
    b0 = P.Bundle(season="2024-25", minutes=P.M.default_params("2024-25"), state=sp,
                  copula=P.C.default_copula_params("2024-25"))
    b1 = P.Bundle(season="2024-25", minutes=b0.minutes, state=dict(sp, early_prior=EP.neutral_params()),
                  copula=b0.copula)
    P.set_early_prior(False)
    assert P._ep_input(Ctx(), b0, conn) is None and P._ep_input(Ctx(), b1, conn) is None
    P.set_early_prior(True)
    assert P._ep_input(Ctx(), b0, conn) is None
    ctx = Ctx()
    e = P._ep_input(ctx, b1, conn)
    assert isinstance(e, EP.EPInput) and e.data is ctx._ep_data
    assert P._ep_input(ctx, b1, conn).data is e.data            # loaded once per Context


# ============================================================================== ON math

def test_neutral_params_reproduce_base(conn):
    sp = ST.default_params()
    q = _q(list(range(1, N_PLAYERS + 1)) + [NEW], "2024-11-15")
    a = ST.rate_frame(conn, q, sp)
    b = ST.rate_frame(conn, q, sp, ep=_ep(conn=conn))
    for s in (*ST.STATS, "dk"):
        assert np.allclose(b[f"r_{s}"], a[f"r_{s}"], rtol=1e-11, atol=1e-13)
    wa = ST.walk_forward(conn, "2024-25", sp)
    wb = ST.walk_forward(conn, "2024-25", sp, ep=_ep(conn=conn))
    for s in ST.STATS:
        assert np.allclose(wb[f"r_{s}"], wa[f"r_{s}"], rtol=1e-11, atol=1e-13)


def test_on_changes_only_rates(conn):
    sp = ST.default_params()
    q = _q(range(1, N_PLAYERS + 1), "2024-10-24")
    a = ST.rate_frame(conn, q, sp)
    b = ST.rate_frame(conn, q, sp, ep=_ep(_hand_params(), conn))
    rcols = [c for c in a.columns if c.startswith("r_")]
    other = [c for c in a.columns if c not in rcols]
    pd.testing.assert_frame_equal(a[other], b[other], check_exact=True)
    assert not np.allclose(a["r_pts"], b["r_pts"])


def test_preseason_math_for_a_new_player(conn):
    """No regular-season game: r = (w v Xpre + kappa mu) / (w Npre + kappa), decayed by H."""
    sp = ST.default_params()
    day = "2024-10-21"
    a = ST.rate_frame(conn, _q([NEW], day), sp)
    b = ST.rate_frame(conn, _q([NEW], day), sp, ep=_ep(_hand_params(0.6, 0.9), conn))
    pre = pd.read_sql_query("SELECT * FROM preseason_player_games WHERE player_id = ?", conn,
                            params=(NEW,))
    lag = (pd.Timestamp(day) - pd.to_datetime(pre["game_date"])).dt.days.to_numpy(float)
    for s in ("pts", "ast", "reb"):
        h, kappa = sp["stats"][s]["H"], sp["stats"][s]["kappa"]
        wt = np.exp(-lag * np.log(2) / h)
        x, n = float(np.sum(wt * pre[s])), float(np.sum(wt * pre["min"]))
        mu = float(a[f"p_{s}"].iloc[0])
        assert a[f"r_{s}"].iloc[0] == pytest.approx(mu)
        assert b[f"r_{s}"].iloc[0] == pytest.approx((0.6 * 0.9 * x + kappa * mu)
                                                     / (0.6 * n + kappa), rel=1e-10)


def test_preseason_strictly_before_and_own_season(conn):
    sp = ST.default_params()
    epi = _ep(_hand_params(), conn)
    first_pre = date(2024, 10, 8)
    # on the first preseason day nothing of it is known yet: neutral = base
    a = ST.rate_frame(conn, _q([NEW, 3], first_pre.isoformat()), sp)
    b = ST.rate_frame(conn, _q([NEW, 3], first_pre.isoformat()), sp, ep=epi)
    assert np.allclose(a["r_pts"], b["r_pts"], rtol=1e-11)
    # the day after it: only that game
    nxt = (first_pre + timedelta(days=1)).isoformat()
    b1 = ST.rate_frame(conn, _q([NEW], nxt), sp, ep=epi)
    a1 = ST.rate_frame(conn, _q([NEW], nxt), sp)
    assert b1["r_pts"].iloc[0] != pytest.approx(a1["r_pts"].iloc[0])
    # a season's preseason never reaches the next season
    nxt_season = "2025-10-25"
    a2 = ST.rate_frame(conn, _q([NEW], nxt_season), sp)
    b2 = ST.rate_frame(conn, _q([NEW], nxt_season), sp, ep=epi)
    assert np.allclose(a2["r_pts"], b2["r_pts"], rtol=1e-11)


def test_load_preseason_drops_scrimmage_and_non_nba(conn):
    c = sqlite3.connect(":memory:")
    db.migrate(c)
    c.executemany("INSERT INTO team_games(game_id, team, season, season_type, game_date) "
                  "VALUES (?,?,?,?,?)", [("G1", "AAA", "2024-25", "Regular Season", "2024-10-20")])
    base = ["2024-25", "Pre Season", "2024-10-10", 20.0] + [1] * 16
    rows = [["P1", 1, "a", "AAA"] + base + [0, 0], ["P2", 2, "b", "AAA"] + base + [1, 0],
            ["P3", 3, "c", "AAA"] + base + [0, 1], ["P4", 4, "d", "XXX"] + base + [0, 0],
            ["P5", 5, "e", "AAA", "2024-25", "Pre Season", "2024-10-21", 20.0] + [1] * 16 + [0, 0]]
    c.executemany(
        "INSERT INTO preseason_player_games(game_id, player_id, player_name, team, season, "
        "season_type, game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, "
        "blk, tov, pf, pts, plus_minus, scrimmage, vs_non_nba) VALUES ("
        + ",".join("?" * 26) + ")", rows)
    pre = EP.load_preseason(c)
    assert pre["pid"].tolist() == [1]          # scrimmage, non-NBA, non-NBA team, post-start out


# ============================================================================== fit

def test_fit_recovers_informative_preseason(conn):
    sp = ST.default_params()
    epp = EP.fit(conn, "2025-26", sp)
    assert epp["parts"] == list(EP.PARTS) and epp["fit_seasons"] == ["2020-21", "2024-25"]
    assert epp["w"]["pts"] > 0.0 and epp["w"]["fga"] > 0.0
    assert all(epp["v"][s] in EP.V_GRID and epp["w"][s] in EP.W_GRID for s in ST.STATS)
    # pieces not asked for stay neutral
    for s in ST.STATS:
        assert epp["d"][s] == [1.0] * (EP.N_AGE + 1) and epp["m"][s] == [1.0] * EP.N_DRAFT
        assert epp["g"][s] == [1.0] * EP.N_ROLE
    assert epp["diagnostics"]["ll_gain"]["pts"] > 0
    # out of sample (the fixture's 2024-25 is a training season of 2025-26; refit on < 2024-25)
    ep2 = EP.fit(conn, "2024-25", sp)
    wa = ST.walk_forward(conn, "2024-25", sp)
    wb = ST.walk_forward(conn, "2024-25", sp, ep=_ep(ep2, conn))
    early = pd.to_datetime(wa["game_date"]) < pd.Timestamp("2024-11-20")
    ma = wa["min"].to_numpy(float)
    ll = [ST.nb_logpmf(w["pts"][early], (w["r_pts"] * ma)[early], 0.05).sum() for w in (wa, wb)]
    assert ll[1] > ll[0]


def test_fit_parts_other_pieces_run(conn):
    epp = EP.fit(conn, "2025-26", ST.default_params(), parts=("d", "c", "m", "g"), passes=1)
    assert epp["w"]["pts"] == 0.0 and epp["v"]["pts"] == 1.0
    assert all(len(epp["d"][s]) == EP.N_AGE + 1 for s in ST.STATS)


# ============================================================================== bundle

def test_load_bundle_ep_adds_state_only(monkeypatch, restore_switches):
    base = P.Bundle(season="2024-25", minutes=P.M.default_params("2024-25"),
                    state=ST.default_params(), copula=P.C.default_copula_params("2024-25"))
    calls = []

    def fake_fit(conn, season, sp, **kw):
        calls.append(season)
        assert "early_prior" not in sp
        return EP.neutral_params()
    monkeypatch.setattr(EP, "fit", fake_fit)
    real = P.load_bundle
    monkeypatch.setattr(P, "load_bundle", lambda *a, **k: base if not EP.EARLY_PRIOR
                        else real(*a, **k))
    P.set_context(False)
    P.set_minfix(False)
    P.set_early_prior(True)
    b = P._load_bundle_ep(None, "2024-25", None, refit=False, log=None)
    assert calls == ["2024-25"] and EP.has_ep(b.state)
    assert {k: v for k, v in b.state.items() if k != "early_prior"} == base.state
    assert b.minutes == base.minutes and b.copula == base.copula and b.context is None
    assert "early_prior" not in base.state
    assert (P.cache_dir() / "bundle_2024-25_v1_ep.json").exists()
    b2 = P._load_bundle_ep(None, "2024-25", None, refit=False, log=None)   # cached
    assert calls == ["2024-25"] and b2.state == b.state
