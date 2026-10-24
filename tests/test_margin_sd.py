"""B7: the minutes eval integrates margins at the fitted per-season sd (seasons < S), not 12.4.
No live path uses the hard-coded value, so there is no switch (and no `+ms` tag)."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import minutes_eval as ME


def _db(n_per_season=120, sds=(12.0, 14.0, 13.0, 20.0)):
    """4 seasons, margins = -spread + noise(sd per season); the last (2023-24) must be excluded
    when fitting for it."""
    rng = np.random.default_rng(5)
    c = sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE team_games (game_id TEXT, team TEXT, season TEXT, season_type TEXT,
                             is_home INTEGER, pts INTEGER);
    CREATE TABLE closing_lines (game_id TEXT, source TEXT, home_spread_close REAL,
                                total_close REAL);""")
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    for s, sd in zip(seasons, sds):
        for i in range(n_per_season):
            g, hs = f"{s}-{i}", float(rng.integers(-10, 10))
            mar = -hs + rng.normal(0, sd)
            tot = 220 + rng.normal(0, 18)
            ph, pa = (tot + mar) / 2, (tot - mar) / 2
            c.executemany("INSERT INTO team_games VALUES (?,?,?,?,?,?)",
                          [(g, "H", s, "Regular Season", 1, ph), (g, "A", s, "Regular Season", 0, pa)])
            c.execute("INSERT INTO closing_lines VALUES (?,?,?,?)", (g, "x", hs, 220.0))
    return c


@pytest.fixture(autouse=True)
def _reset():
    yield
    ME.set_margin_sd(None)


def _g_params():
    prm = M.default_params()
    nt, nb = len(M.TIER_EDGES) + 1, len(M.MARGIN_BINS) - 1
    tab = np.zeros((nt, 2, nb))
    ramp = np.clip((np.arange(nb) - 2) * 1.2, 0, None)
    tab[4], tab[0] = -ramp, +ramp
    prm.g_table = tab.tolist()
    return prm


def _proj():
    return pd.DataFrame({"mean": [36.0, 34, 33, 30, 30, 22, 20, 17, 11, 7],
                         "p_play": 1.0, "pf36": 3.0})


def test_margin_sd_is_the_copula_fit_on_prior_seasons():
    c = _db()
    sd = ME.margin_sd(c, "2023-24")
    assert sd == C.fit_env(c, "2023-24")["sd_margin"]
    assert 12.5 < sd < 14.5                  # the 20-sd 2023-24 season is not in the fit
    assert ME.margin_sd(c, "2024-25") > sd + 1.0   # but it is in the next season's


def test_margin_sd_falls_back_like_the_copula_when_data_is_thin():
    c = _db(n_per_season=5)
    assert ME.margin_sd(c, "2023-24") == C.fit_env(c, "2023-24")["sd_margin"] == 13.7


def test_default_is_unchanged():
    assert M.MARGIN_SD == 12.4
    prm, proj = _g_params(), _proj()
    assert np.array_equal(M.expected_minutes(proj, prm, spread=3.0),
                          M.expected_minutes(proj, prm, spread=3.0, sd=12.4))
    ME.set_margin_sd(None)
    assert ME._sd() == 12.4


def test_integrate_uses_the_set_sd():
    prm = _g_params()
    r = _proj().assign(key="g", reserve=0.0, mean0=_proj()["mean"])
    sp = pd.Series({"g": 12.0})
    base = ME.integrate(r, prm, sp)
    assert np.array_equal(base, M.expected_minutes(r, prm, spread=12.0, sd=12.4))
    ME.set_margin_sd(14.0)
    wide = ME.integrate(r, prm, sp)
    assert np.array_equal(wide, M.expected_minutes(r, prm, spread=12.0, sd=14.0))
    assert not np.array_equal(wide, base)


def test_no_live_path_uses_the_hardcoded_sd():
    """project_team / expected_minutes have no caller outside minutes, minutes_eval and tests:
    the pipeline and copula draw margins from the fitted env, so there is no `+ms` switch."""
    import pathlib
    import courtside
    root = pathlib.Path(courtside.__file__).parent
    hits = {p.relative_to(root).as_posix() for p in root.rglob("*.py")
            if any(k in p.read_text() for k in ("MARGIN_SD", "expected_minutes(", "project_team("))}
    assert hits <= {"proj/minutes.py", "proj/minutes_eval.py"}
    from courtside.proj import pipeline as P
    assert "+ms" not in P.MODEL_VERSION
