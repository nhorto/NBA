"""B7: the props dispersion PHI is fit on seasons < S only (proj/props.py).

Live (S = 2026-27) keeps the packaged PHI, trained on 2023-24..2025-26, all in the past; any
earlier evaluation season gets `phi_for` / `fit_phi(before=S)`, which never reads S or later."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from courtside.proj import props as P

LIVE_PHI_LITERAL = {"pts": 2.40, "reb": 1.11, "ast": 1.16, "fg3m": 1.09, "pra": 2.05}
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
           "2026-27"]


def make_db(path=":memory:", boost: dict | None = None) -> sqlite3.Connection:
    """30 players x 82 games x each season. `boost[season]` multiplies the noise of that
    season's counts, so the seasons have clearly different dispersion."""
    rng = np.random.default_rng(7)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE player_games (player_id INT, game_date TEXT, season TEXT, "
                 "min REAL, pts INT, reb INT, ast INT, fg3m INT)")
    rows = []
    for si, season in enumerate(SEASONS):
        k = (boost or {}).get(season, 1.0)
        y0 = int(season[:4])
        for pid in range(1, 31):
            base = 8 + pid * 0.5
            for g in range(82):
                day = f"{y0}-{11 + g // 30:02d}-{1 + g % 28:02d}"
                gamma = rng.gamma(1.0 / (0.15 * k), 0.15 * k)      # per-game rate noise
                rows.append((pid, day, season, 30.0, *(int(rng.poisson(base * f * gamma))
                                                       for f in (1.0, 0.4, 0.3, 0.1))))
    conn.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?)", rows)
    return conn


@pytest.fixture(scope="module")
def conn():
    return make_db()


def test_live_constants_pinned():
    assert P.PHI == LIVE_PHI_LITERAL
    assert P.LIVE_PHI_SEASON == "2026-27" and P.PHI_N_SEASONS == 3


def test_phi_seasons_window():
    assert P.phi_seasons("2026-27") == ("2023-24", "2024-25", "2025-26")     # the live window
    assert P.phi_seasons("2025-26") == ("2022-23", "2023-24", "2024-25")
    assert P.phi_seasons("2023-24") == ("2020-21", "2021-22", "2022-23")
    for s in SEASONS:
        assert all(t < s for t in P.phi_seasons(s))


def test_live_fit_unchanged(conn):
    """fit_phi's old default (2023-24..2025-26) == before='2026-27' == the explicit tuple."""
    old = P.fit_phi(conn, ("2023-24", "2024-25", "2025-26"))
    assert P.fit_phi(conn) == old
    assert P.fit_phi(conn, before="2026-27") == old
    assert set(old) == set(P.STATS)


def test_live_season_serves_the_packaged_phi_without_a_db():
    assert P.phi_for(None, "2026-27") is P.PHI
    assert P.phi_for(None, "2027-28") is P.PHI
    with pytest.raises(ValueError):
        P.phi_for(None, "2025-26")


def test_fit_never_reads_season_S_or_later(conn):
    """Making S and later seasons wildly noisier must not move phi for S; making an earlier
    season noisier must (so the fit really uses the seasons before S)."""
    base = P.fit_phi(conn, before="2025-26")
    boost_late = make_db(boost={"2025-26": 6.0, "2026-27": 6.0})
    assert P.fit_phi(boost_late, before="2025-26") == base            # S and later: ignored
    boost_early = make_db(boost={"2023-24": 6.0})
    assert P.fit_phi(boost_early, before="2025-26")["pts"] > base["pts"] + 0.5
    # the live fit does read 2025-26 (it is before 2026-27)
    assert P.fit_phi(boost_late, before="2026-27")["pts"] > P.fit_phi(conn)["pts"] + 0.5


def test_evaluation_seasons_differ_from_the_live_fit(conn):
    """2023-24 / 2024-25 / 2025-26 each get their own pre-S window, not the live 2023-26 one."""
    live = P.fit_phi(conn)
    for s in ("2023-24", "2024-25", "2025-26"):
        assert P.fit_phi(conn, before=s) != live


def test_fit_rejects_overlapping_seasons(conn):
    with pytest.raises(ValueError):
        P.fit_phi(conn, ("2023-24", "2024-25"), before="2024-25")


def test_phi_for_matches_fit_and_is_memoised(tmp_path):
    c = make_db(str(tmp_path / "t.db"))
    a = P.phi_for(c, "2024-25")
    assert a == P.fit_phi(c, before="2024-25")
    assert P.phi_for(c, "2024-25") is a


def test_player_dist_phi_override_and_live_identity():
    stats = {("pts", "played"): {"mean": 18.0}, ("reb", "played"): {"mean": 5.0},
             ("ast", "played"): {"mean": 4.0}, ("fg3m", "played"): {"mean": 2.0},
             ("min", "played"): {"mean": 30.0, "q05": 20, "q10": 23, "q25": 27, "q50": 31,
                                 "q75": 34, "q90": 37, "q95": 39}}
    live = P.player_dist(stats)
    explicit = P.player_dist(stats, phi=P.PHI)
    wide = P.player_dist(stats, phi={**P.PHI, "pts": 3.5})
    for s in P.STATS:
        assert np.array_equal(live.pmf(s), explicit.pmf(s))               # live: bit-identical
    assert live.phi is None
    assert not np.array_equal(live.pmf("pts"), wide.pmf("pts"))
    assert np.array_equal(live.pmf("reb"), wide.pmf("reb"))
