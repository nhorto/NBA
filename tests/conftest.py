import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("COURTSIDE_DATA", str(d))
    monkeypatch.setenv("COURTSIDE_NO_WRITER", "1")   # tests never call the LLM writer
    return d


def _synthetic_team_rows(seasons=("2000-01", "2001-02", "2002-03", "2003-04"),
                         codes=("BOS", "NYK", "LAL", "GSW", "DEN", "MIA"), seed=0,
                         playoff_games=4):
    """team_games-shaped rows for a small round-robin league (no network, no DB)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    strength = {c: s for c, s in zip(codes, np.linspace(-6, 6, len(codes)))}
    rows, gid = [], 0
    for si, season in enumerate(seasons):
        y = int(season[:4])
        day = pd.Timestamp(f"{y}-11-01")
        teams_ = list(codes)
        n = len(teams_)
        for rep in range(4):
            for r in range(n - 1):  # circle method round robin
                pairs = [(teams_[i], teams_[n - 1 - i]) for i in range(n // 2)]
                for a, b in pairs:
                    h, aw = (a, b) if (r + rep) % 2 == 0 else (b, a)
                    gid += 1
                    _add_game(rows, f"002{y % 100:02d}{gid:05d}", season, "Regular Season",
                              day, h, aw, strength, rng)
                teams_ = [teams_[0]] + [teams_[-1]] + teams_[1:-1]
                day += pd.Timedelta(days=1 + (r % 3 == 0))  # mix of b2b and rest
        for k in range(playoff_games):
            gid += 1
            h, aw = (codes[-1], codes[-2]) if k % 2 == 0 else (codes[-2], codes[-1])
            day += pd.Timedelta(days=2)
            _add_game(rows, f"004{y % 100:02d}{gid:05d}", season, "Playoffs", day, h, aw,
                      strength, rng)
    return pd.DataFrame(rows)


def _add_game(rows, game_id, season, stype, day, h, a, strength, rng):
    import numpy as np
    m = int(round(strength[h] - strength[a] + 3 + rng.normal(0, 11)))
    if m == 0:
        m = 1
    hp = int(100 + max(m, 0) + rng.integers(0, 5))
    ap = hp - m
    for team, opp, is_home, pts, opts in ((h, a, 1, hp, ap), (a, h, 0, ap, hp)):
        rows.append({"game_id": game_id, "team": team, "season": season, "season_type": stype,
                     "game_date": day.strftime("%Y-%m-%d"), "opp": opp, "is_home": is_home,
                     "wl": "W" if pts > opts else "L", "min": 240.0, "pts": pts})


@pytest.fixture
def synthetic_rows():
    return _synthetic_team_rows


@pytest.fixture
def synthetic_frame():
    def make(**kw):
        from courtside.model import dataset, schedule
        return schedule.add_schedule_features(dataset.build_games(_synthetic_team_rows(**kw)))
    return make
