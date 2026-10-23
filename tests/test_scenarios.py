"""P10: availability scenarios kept from the sims (copula -> sims npz -> conditional splits ->
X factor table and the Tonight player card)."""

from __future__ import annotations

import json
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from courtside import db
from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import scenarios as SCN
from courtside.proj import store as SO
from courtside.proj.dfs import xfactor as XF

from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

warnings.filterwarnings("ignore", message="Using `httpx`")


def _half_nonstars(game, k=2):  # noqa: F811
    _, teams, _ = game
    pl = teams["home"][2]
    order = np.argsort(-pl["mpg"].to_numpy())
    return [int(i) for i in order if not pl["star"].iloc[i]][:k]


@pytest.fixture(scope="module")
def sim(game):  # noqa: F811
    """A game with two P(available) = 0.5 non-star home players (4 scenarios, none collapsed);
    40k sims; plus the run's scenario arrays read back through the npz format."""
    spec, mp, sp = _spec(game, p_out=None, p_half=_half_nonstars(game))
    g = C.simulate_game(spec, C.default_copula_params(), mp, sp, 40_000, 5)
    z = {"player_id": g.player_id, "fp": g.fp.astype(np.float16), **SO.scenario_arrays([g])}
    return spec, mp, g, z, SO.read_scenarios(z)


# ------------------------------------------------------------------------------ copula carry

def test_gamesims_carry_scenarios(sim):
    spec, _, g, _, _ = sim
    S, n = g.fp.shape
    assert g.absent.shape == (S, n) and g.absent.dtype == bool
    assert set(g.scenario) == {spec.home.team, spec.away.team}
    assert (g.minutes[g.absent] == 0).all()
    for t, idx in g.scenario.items():
        cols = np.flatnonzero(np.asarray(g.team) == t)
        assert idx.dtype == np.uint8 and idx.shape == (S,)
        kept = g.kept[t]
        assert kept.shape[1] == len(cols) and idx.max() < len(kept)
        # the kept set of each sim is a subset of its drawn absent set
        assert not (kept[idx] & ~g.absent[:, cols]).any()
    assert g.teams[spec.home.team]["collapsed_sims"] == 0
    assert len(g.kept[spec.home.team]) == 4 and len(g.kept[spec.away.team]) == 1


def test_v1_carries_the_same_scenarios(game):  # noqa: F811
    spec, mp, sp = _spec(game, p_half=_half_nonstars(game))
    cp = C.default_copula_params()
    a = C.simulate_game(spec, cp, mp, sp, 3000, 9)
    spec, mp, sp = _spec(game, p_half=_half_nonstars(game))
    b = C.simulate_game_v1(spec, cp, C.default_copula_v1_params(), mp, sp, 3000, 9)
    assert np.array_equal(a.absent, b.absent) and np.array_equal(a.minutes, b.minutes)
    for t in a.scenario:
        assert np.array_equal(a.scenario[t], b.scenario[t])
        assert np.array_equal(a.kept[t], b.kept[t])


# ------------------------------------------------------------------------------ npz format

def test_npz_round_trip(sim, tmp_path):
    _, _, g, _, _ = sim
    path = SO.save_sims("S", "r1", g.player_id, g.fp, dk_id=[f"d{p}" for p in g.player_id],
                        game_id=[g.game_id] * len(g.player_id), root=tmp_path,
                        extra=SO.scenario_arrays([g]))
    z = SO.load_sims(path)
    sc = SO.read_scenarios(z)
    assert np.array_equal(sc.absent, g.absent)
    assert sc.idx.dtype == np.uint8
    played = g.minutes > 0
    assert ((sc.minutes > 0) == played).all()                 # whole minutes keep DNP = 0
    assert np.abs(sc.minutes - g.minutes)[played].max() <= 0.5 + 1e-6
    for t in g.scenario:
        cols = np.flatnonzero(np.asarray(g.team) == t)
        assert np.array_equal(sc.team_idx(t), g.scenario[t])
        assert np.array_equal(sc.team_kept(t)[:, cols], g.kept[t])
        assert not np.delete(sc.team_kept(t), cols, axis=1).any()
    assert list(sc.team) == [str(t) for t in g.team]


def test_old_npz_loads_without_scenarios(tmp_path):
    fp = np.ones((10, 3), np.float32)
    path = SO.save_sims("S", "old", [1, 2, 3], fp, root=tmp_path)
    z = SO.load_sims(path)
    assert set(z) == {"fp", "player_id"} and SO.read_scenarios(z) is None
    assert SO.scenario_arrays([SimpleNamespace(absent=None, scenario=None)]) == {}
    assert SO.scenario_arrays([]) == {}


# ------------------------------------------------------------------------------ splits

def _ref_minutes(spec, mp, g, j, arm_mask, seed=0):
    """E[min | plays] over an arm, re-drawn independently per scenario from the minutes engine
    (`ScenarioMinutes.project(kept)` through `minutes.sample_minutes`, same margins)."""
    team = spec.home
    idx, kept = g.scenario[team.team], g.kept[team.team]
    rng = np.random.default_rng(seed)
    n_ot = (np.abs(g.margin) < C.OT_BAND).astype(float)
    tot = cnt = 0.0
    for b in range(len(kept)):
        s = np.flatnonzero((idx == b) & arm_mask)
        if not len(s):
            continue
        x = M.sample_minutes(team.minutes.project(kept[b]), mp, g.margin[s], rng, n_ot=n_ot[s])
        tot += x[:, j][x[:, j] > 0].sum()
        cnt += (x[:, j] > 0).sum()
    return tot / cnt


def test_conditional_means_match_the_scenario_projection(sim):
    """Arm means = ScenarioMinutes.project(kept) (through the engine's sampler) x rates, within
    MC error: minutes vs an independent re-draw per scenario; DK vs those minutes x his DK per
    minute (no star bump: the uncertain players are not stars, so only minutes move)."""
    spec, mp, g, z, sc = sim
    home = np.flatnonzero(np.asarray(g.team) == spec.home.team)
    checked = 0
    for j in home:
        played = g.minutes[:, j] > 0
        rate = g.fp[played, j].mean() / g.minutes[played, j].mean()
        for s in SCN.player_splits(z["fp"], sc, j):
            for arm, m in (("plays", ~g.absent[:, s["col"]]), ("sits", g.absent[:, s["col"]])):
                a = s[arm]
                assert a["n"] == int(m.sum()) and s["show"]
                ref = _ref_minutes(spec, mp, g, j, m, seed=checked)
                se = g.minutes[m & played, j].std() / np.sqrt((m & played).sum())
                assert a["min_played"] == pytest.approx(ref, abs=5 * se * np.sqrt(2) + 0.1)
                assert a["dk_played"] == pytest.approx(rate * ref, rel=0.03)
                checked += 1
    assert checked >= 2 * 2 * (len(home) - 2)
    # the split is real: the main backup's minutes rise when the starter ahead of him sits
    shifts = [s["sits"]["min_played"] - s["plays"]["min_played"]
              for j in home for s in SCN.player_splits(z["fp"], sc, j)]
    assert max(shifts) > 2.0


def test_lever_band_and_big_night(sim):
    spec, _, g, z, sc = sim
    j = int(np.flatnonzero(np.asarray(g.team) == spec.home.team)[0])
    lev = SCN.lever_teammates(sc, j)
    assert len(lev) == 2 and all(0.4 < p < 0.6 for _, p in lev)
    assert SCN.lever_teammates(sc, j, lo=0.6) == []
    sal = 5000.0
    s = SCN.player_splits(z["fp"], sc, j, salary=sal, tier="role")[0]
    m = sc.absent[:, s["col"]]
    raw = XF.raw_chances(z["fp"][m][:, [j]], [sal])[0]          # the X factor definition
    assert s["sits"]["big_raw"] == pytest.approx(raw)
    assert s["sits"]["big"] == pytest.approx(XF.calibrate([raw], ["role"])[0])
    assert s["plays"]["n"] + s["sits"]["n"] == len(m)
    assert SCN.player_splits(z["fp"], sc, j)[0]["sits"]["big"] is None     # no salary


def test_card_splits_rules():
    def arm(n, dk, mn):
        return {"n": n, "dk": dk, "dk_played": dk, "min": mn, "min_played": mn, "big": None}
    a = {"show": True, "plays": arm(900, 10, 15), "sits": arm(300, 14, 20)}
    small = {"show": False, "plays": arm(990, 10, 15), "sits": arm(10, 20, 30)}
    flat = {"show": True, "plays": arm(600, 10, 15), "sits": arm(600, 10.4, 15.5)}
    big = {"show": True, "plays": arm(600, 10, 15), "sits": arm(600, 18, 24)}
    assert SCN.card_splits([a, small, flat, big]) == [big, a]
    assert SCN.card_splits([a, big], k=1) == [big]


# ------------------------------------------------------------------------------ live run, card

S_WEB = 4000


def _web_sims(path):
    """Sims for props_helpers' G1 (AAA: 101, 102; BBB: 201, 202): 102 sits in a third of the
    sims, and 101 then plays 34 instead of 26 minutes."""
    rng = np.random.default_rng(1)
    pid = np.array([101, 102, 201, 202])
    sits = rng.random(S_WEB) < 1 / 3
    absent = np.zeros((S_WEB, 4), bool)
    absent[:, 1] = sits
    mins = np.column_stack([np.where(sits, 34.0, 26.0), np.where(sits, 0.0, 24.0),
                            np.full(S_WEB, 30.0), np.full(S_WEB, 12.0)])
    mins += rng.normal(0, 0.5, mins.shape) * (mins > 0)
    fp = mins.copy()                         # DK = minutes: big night (>= 30) iff he gets 34
    kept = {"AAA": np.array([[False, False], [False, True]]),
            "BBB": np.array([[False, False]])}
    g = SimpleNamespace(fp=fp.astype(np.float32), minutes=mins.astype(np.float32),
                        player_id=pid, team=np.array(["AAA", "AAA", "BBB", "BBB"]),
                        absent=absent, scenario={"AAA": sits.astype(np.uint8),
                                                 "BBB": np.zeros(S_WEB, np.uint8)},
                        kept=kept, game_id="0022600901")
    return SO.save_sims("S1", "r_main", pid, g.fp, dk_id=[f"d{p}" for p in pid],
                        game_id=[g.game_id] * 4, root=path, extra=SO.scenario_arrays([g]))


@pytest.fixture
def web(data_dir):
    import props_helpers as H
    from fastapi.testclient import TestClient
    from courtside.web.app import create_app
    from test_web_players import NOW
    conn = db.connect()
    H.populate(conn)
    path = _web_sims(data_dir / "sims")
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id='r_main'", (str(path),))
    for pid in (101, 102, 201, 202):
        conn.execute("UPDATE player_projections SET inputs_json=? WHERE run_id='r_main' AND "
                     "player_id=?", (json.dumps({"dk_id": f"d{pid}"}), pid))
        conn.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                     "player_name, site_player_id, salary, roster_slot) VALUES "
                     "('2026-10-20T12:00:00Z','dk','S1','classic',?,?,5000,'UTIL')",
                     (str(pid), f"d{pid}"))
    conn.commit()
    SCN._CACHE.clear()
    XF._CACHE.clear()
    app = create_app(data_dir)
    app.state.players_now = lambda: NOW
    yield conn, TestClient(app), path
    conn.close()


def test_run_splits_and_xfactor_columns(web):
    conn, _, _ = web
    sp = SCN.run_splits(conn, "r_main")
    s = sp[101][0]
    assert s["teammate"] == 102 and s["p_avail"] == pytest.approx(2 / 3, abs=0.03)
    assert s["sits"]["min_played"] == pytest.approx(34.0, abs=0.2)
    assert s["plays"]["min_played"] == pytest.approx(26.0, abs=0.2)
    assert s["sits"]["big_raw"] == pytest.approx(1.0) and s["plays"]["big_raw"] == 0.0   # 30
    assert sp[201] == [] and SCN.run_splits(conn, None) == {}
    t = XF.run_table(conn, "r_main")
    assert t["d101"]["splits"][0]["teammate_dk"] == "d102"
    assert t["d101"]["splits"][0]["sits"]["big"] == pytest.approx(
        XF.calibrate([1.0], ["role"])[0])


def test_card_shows_the_split(web):
    from test_web_players import ADVICE_RE, _main_text
    import props_helpers as H
    _, client, _ = web
    r = client.get(f"/players/{H.DAY}")
    assert r.status_code == 200
    t = _main_text(r.text)
    assert r.text.count('class="pl-split"') == 1
    assert re.search(r"With and without Nicolas Claxton .*? Claxton sits 34 34\.0 \d+% 3\d% "
                     r"Claxton plays 26 26\.0 \d+% 6\d%", t)
    assert "With and without" in t and not ADVICE_RE.search(t)


def test_card_without_scenarios(web, tmp_path):
    import props_helpers as H
    conn, client, path = web
    old = SO.save_sims("S1", "old", [101, 102], np.ones((10, 2)), root=tmp_path)
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id='r_main'", (str(old),))
    conn.commit()
    SCN._CACHE.clear()
    r = client.get(f"/players/{H.DAY}?sort=name")
    assert r.status_code == 200 and 'class="pl-split"' not in r.text
    assert "With and without" not in r.text
    assert SCN.run_splits(conn, "r_main") == {}
