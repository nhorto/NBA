"""predict_day: no leakage, schedule-only upcoming games, market and injuries attached."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest
from synth import day_date, synth_conn

from courtside import predict, production
from courtside.model import baselines


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)


DAY = day_date("2024-25", 22)          # a day whose games are already in team_games


def _strip(rows):
    """Rows minus what legitimately differs between runs (clock, fit bookkeeping)."""
    def one(r):
        out = {k: v for k, v in r.items() if k not in ("created_at", "opinion")}
        out["inputs"] = {k: v for k, v in r["inputs"].items() if k != "fit"}
        if r.get("opinion"):
            out["opinion"] = one(r["opinion"])
        return out
    return [one(r) for r in rows]


def test_predict_past_day_from_team_games_and_stores_history():
    conn = synth_conn()
    rows = predict.predict_day(conn, DAY)
    n_games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM team_games WHERE game_date=?",
                           (DAY.isoformat(),)).fetchone()[0]
    assert len(rows) == n_games == 3
    for r in rows:
        assert 0 < r["home_win_prob"] < 1 and r["total"] > 150
        assert r["model"] == predict.PRODUCTION_MODEL == predict.PRODUCTION_WINPROB
        assert r["opinion"]["model"] == predict.PRODUCTION_OPINION
        assert r["created_at"] == f"{DAY.isoformat()}T12:00:00Z"       # past day: virtual time
        assert set(r["injuries"]) == {"home", "away"}
    stored = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert stored == 6                                  # two model rows per game
    predict.predict_day(conn, DAY)                                   # idempotent for past days
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 6
    latest = predict.latest_pretip(conn, day=DAY.isoformat())
    assert set(latest) == {r["game_id"] for r in rows}
    assert latest[rows[0]["game_id"]]["inputs"]["schedule"]["home"]["rest"] is not None


class SpyModel:
    """Delegates to the real model but records what it was fit and asked to predict on."""
    seen: list = []

    def __init__(self):
        self.inner = baselines.MODELS["elo_sched"]()
        self.name = "spy"

    def fit(self, train):
        SpyModel.seen.append(("fit", train["game_date"].max(), len(train),
                              int(train["game_id"].nunique())))
        self.inner.fit(train)

    def predict_proba(self, test):
        SpyModel.seen.append(("test", test["game_date"].min(), test["game_date"].max(),
                              float(test["margin"].abs().sum()),
                              float(test["home_pts"].abs().sum())))
        return self.inner.predict_proba(test)

    def predict_margin(self, test):
        return self.inner.predict_margin(test)


def test_fit_uses_only_games_before_day_and_test_results_are_masked(monkeypatch):
    conn = synth_conn()
    SpyModel.seen = []
    monkeypatch.setattr(predict, "model_factory", lambda name, conn=None: SpyModel)
    predict.predict_day(conn, DAY, model="spy", store=False)
    fit = next(s for s in SpyModel.seen if s[0] == "fit")
    test = next(s for s in SpyModel.seen if s[0] == "test")
    assert fit[1] < np.datetime64(DAY)                       # newest training game before DAY
    expected_train = conn.execute("SELECT COUNT(DISTINCT game_id) FROM team_games "
                                  "WHERE game_date < ?", (DAY.isoformat(),)).fetchone()[0]
    assert fit[3] == expected_train
    assert test[1] == test[2] == np.datetime64(DAY)
    assert test[3] == 0.0 and test[4] == 0.0                 # results are placeholders


def test_results_on_or_after_day_never_change_predictions():
    conn = synth_conn()
    before = predict.predict_day(conn, DAY, store=False)
    cut = DAY.isoformat()
    # scramble every result from DAY onward (the games being predicted and everything later)
    conn.execute("UPDATE team_games SET pts = pts + ((abs(random()) % 40) - 20), "
                 "wl = CASE WHEN abs(random()) % 2 = 0 THEN 'W' ELSE 'L' END "
                 "WHERE game_date >= ?", (cut,))
    # keep the pairs consistent for the loader: make it decisive and non-tied
    conn.execute("UPDATE team_games SET pts = pts + 1 WHERE is_home = 1 AND game_date >= ? "
                 "AND pts = (SELECT pts FROM team_games t2 WHERE t2.game_id = team_games.game_id "
                 "AND t2.is_home = 0)", (cut,))
    after = predict.predict_day(conn, DAY, store=False)
    assert _strip(before) == _strip(after)
    # and changing the result of an EARLIER game does change them (the test can fail)
    conn.execute("UPDATE team_games SET pts = pts + 30 WHERE is_home = 1 AND game_date = ?",
                 ((DAY - timedelta(days=1)).isoformat(),))
    changed = predict.predict_day(conn, DAY, store=False)
    assert _strip(changed) != _strip(before)


def test_upcoming_games_come_from_games_table_without_leakage():
    conn = synth_conn()
    last = day_date("2024-25", 39)
    nxt = last + timedelta(days=1)
    conn.executemany(
        "INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
        "VALUES (?,?,?,?,?,?,?)",
        [("0022400901", "2024-25", nxt.isoformat(), f"{nxt}T23:00:00Z", "BOS", "NYK", "scheduled"),
         ("0022400902", "2024-25", nxt.isoformat(), f"{nxt}T23:30:00Z", "LAL", "GSW", "scheduled"),
         ("0012400903", "2024-25", nxt.isoformat(), f"{nxt}T23:30:00Z", "DEN", "MIA", "scheduled"),
         ("0022400904", "2024-25", nxt.isoformat(), None, "DEN", "MIA", "postponed")])
    rows = predict.predict_day(conn, nxt, created_at="2024-12-01T15:00:00Z")
    assert [r["game_id"] for r in rows] == ["0022400901", "0022400902"]   # preseason + postponed skipped
    assert all(r["tipoff_utc"] for r in rows)
    # schedule features exist for games that are not in team_games (b2b etc. from history)
    assert rows[0]["inputs"]["schedule"]["home"]["rest"] in (0, 1, 2, 3, 4)
    # same prediction no matter what happens afterwards: add a result for the game, rerun
    again = predict.predict_day(conn, nxt, created_at="2024-12-01T15:00:00Z", store=False)
    assert _strip(rows) == _strip(again)
    # a later prediction row never replaces the earlier one (append-only)
    predict.predict_day(conn, nxt, created_at="2024-12-01T16:00:00Z")
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE game_id='0022400901'").fetchone()[0]
    assert n == 4                                   # two models x two runs


def test_market_line_is_pre_tip_devigged_and_falls_back_to_closing():
    conn = synth_conn()
    nxt = day_date("2024-25", 39) + timedelta(days=1)
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES ('0022400901','2024-25',?,?,'BOS','NYK','scheduled')",
                 (nxt.isoformat(), f"{nxt}T23:00:00Z"))
    ins = ("INSERT INTO odds_snapshots (game_id, captured_at, source, provider, home_spread, "
           "total, home_ml, away_ml) VALUES ('0022400901',?,?,?,?,?,?,?)")
    conn.execute(ins, (f"{nxt}T14:00:00Z", "espn", "DK", -3.0, 220.0, -150, 130))
    conn.execute(ins, (f"{nxt}T20:00:00Z", "espn", "DK", -4.0, 221.0, -200, 170))
    conn.execute(ins, (f"{nxt}T23:30:00Z", "espn", "DK", -9.0, 230.0, -900, 600))   # after tip
    conn.commit()
    (r,) = predict.predict_day(conn, nxt, store=False, created_at=f"{nxt}T21:00:00Z")
    assert r["market_home_spread"] == -4.0 and r["market_total"] == 221.0
    ph, pa = 200 / 300, 100 / 270
    assert r["market_home_prob"] == pytest.approx(ph / (ph + pa), abs=1e-4)
    # no snapshots at all: closing_lines is used
    conn.execute("DELETE FROM odds_snapshots")
    gid = conn.execute("SELECT game_id FROM team_games WHERE game_date=? LIMIT 1",
                       (DAY.isoformat(),)).fetchone()[0]
    conn.execute("INSERT INTO closing_lines (game_id, source, provider, home_spread_close, "
                 "total_close, home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', 2.5, 215.5, "
                 "120, -140)", (gid,))
    rows = {x["game_id"]: x for x in predict.predict_day(conn, DAY, store=False)}
    assert rows[gid]["market_home_spread"] == 2.5 and rows[gid]["market_total"] == 215.5
    assert rows[gid]["market_home_prob"] < 0.5
    other = next(x for k, x in rows.items() if k != gid)
    assert other["market_home_prob"] is None and other["market_home_spread"] is None


def test_injuries_attached_from_latest_snapshot():
    conn = synth_conn()
    (g, *_) = predict.predict_day(conn, DAY, store=False)
    home = g["home"]
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
           "VALUES (?,?,?,?,?,?)")
    conn.execute(ins, (f"{DAY - timedelta(days=1)}T15:00:00Z", "espn", home, "Old Guy", "Out", "x"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", home, "Star Man", "Out", "knee"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", home, "Role Guy", "Probable", "x"))
    conn.execute(ins, (f"{DAY}T15:00:00Z", "espn", g["away"], "Wing Man", "Questionable", "x"))
    conn.commit()
    rows = {r["game_id"]: r for r in predict.predict_day(conn, DAY, store=False)}
    inj = rows[g["game_id"]]["injuries"]
    assert [x["player"] for x in inj["home"]] == ["Star Man"]          # latest batch, relevant only
    assert [x["status"] for x in inj["away"]] == ["Questionable"]


def test_too_little_history_returns_nothing():
    conn = synth_conn({"2024-25": 3})
    assert predict.predict_day(conn, day_date("2024-25", 2), store=False) == []


def test_backfill_equals_single_day_and_is_walk_forward():
    conn = synth_conn()
    d0, d1 = day_date("2024-25", 10), day_date("2024-25", 14)
    res = predict.predict_days(conn, d0, d1)
    assert res["days"] == 5 and res["games"] == 15
    stored = {r["game_id"]: r["home_win_prob"] for r in conn.execute(
        "SELECT game_id, home_win_prob FROM predictions WHERE nba_day=?", (d1.isoformat(),))}
    single = {r["game_id"]: r["home_win_prob"]
              for r in predict.predict_day(conn, d1, store=False)}
    assert stored == pytest.approx(single, abs=1e-9)
    json.dumps([dict(r) for r in conn.execute("SELECT * FROM predictions")])


def test_registry_is_generic():
    conn = synth_conn()
    assert predict.model_factory("elo")().name == "elo"
    with pytest.raises(KeyError):
        predict.model_factory("nope")
    rows = predict.predict_day(conn, DAY, model="elo", store=False)
    assert rows and rows[0]["model"] == "elo"


# --- the production pair: lines, injuries, factors, cache -------------------------------------

def _closing_lines(conn, before: date, noise: float = 4.0):
    """A closing spread/total for every game before `before` that tracks the real margin."""
    rng = np.random.default_rng(3)
    rows = conn.execute("SELECT h.game_id, h.pts - a.pts AS m, h.pts + a.pts AS t FROM team_games h "
                        "JOIN team_games a ON a.game_id=h.game_id AND a.is_home=0 "
                        "WHERE h.is_home=1 AND h.game_date < ?", (before.isoformat(),)).fetchall()
    conn.executemany(
        "INSERT INTO closing_lines (game_id, source, provider, home_spread_close, total_close, "
        "home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', ?, ?, -150, 130)",
        [(r["game_id"], -0.7 * r["m"] + rng.normal(0, noise), r["t"] + rng.normal(0, noise))
         for r in rows])
    conn.commit()


def _by_game(rows, which="headline"):
    return {r["game_id"]: (r if which == "headline" else r["opinion"]) for r in rows}


def _snapshot_line(conn, day, spread, total=220.0, at=None):
    at = at or f"{day}T11:00:00Z"
    for r in conn.execute("SELECT DISTINCT game_id FROM team_games WHERE game_date=?",
                          (day.isoformat(),)).fetchall():
        conn.execute("INSERT INTO odds_snapshots (game_id, captured_at, source, provider, "
                     "home_spread, total, home_ml, away_ml) VALUES (?,?,?,?,?,?,?,?)",
                     (r[0], at, "espn", "DK", spread, total, -200, 170))
    conn.commit()


def test_headline_model_takes_the_live_line_and_the_opinion_does_not():
    conn = synth_conn()
    _closing_lines(conn, DAY)
    base = predict.predict_day(conn, DAY, store=False)
    assert all(r["market_home_spread"] is None for r in base)           # no line for DAY yet
    _snapshot_line(conn, DAY, spread=-14.0, total=250.0)                 # home -14, total 250
    new = predict.predict_day(conn, DAY, store=False)
    for b, n in zip(base, new):
        assert n["market_home_spread"] == -14.0 and n["market_total"] == 250.0
        assert n["home_margin"] > b["home_margin"] + 1.0                 # moved toward the line
        assert n["total"] > b["total"]
        assert n["opinion"]["home_margin"] == b["opinion"]["home_margin"]    # no market input
        assert n["opinion"]["home_win_prob"] == b["opinion"]["home_win_prob"]
        # the stored market columns are the line both models were compared with
        assert n["opinion"]["market_home_spread"] == -14.0
    # the closing line is the fallback when no snapshot exists
    conn.execute("DELETE FROM odds_snapshots")
    gid = base[0]["game_id"]
    conn.execute("INSERT INTO closing_lines (game_id, source, provider, home_spread_close, "
                 "total_close, home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', -3.5, 221, "
                 "-160, 140)", (gid,))
    r = _by_game(predict.predict_day(conn, DAY, store=False))[gid]
    assert r["market_home_spread"] == -3.5 and r["inputs"]["market"]["kind"] == "closing"


def test_injury_report_is_priced_by_both_models_and_explained():
    conn = synth_conn()
    base = predict.predict_day(conn, DAY, store=False)
    g = base[0]
    home = g["home"]
    # no report for a past day with box scores: the engine's leak-safe "regulars who played"
    av = g["inputs"]["availability"]
    assert av["home"]["basis"] == av["away"]["basis"] == "regulars_played"
    assert av["home"]["source"] is None
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
           "reason) VALUES (?,?,?,?,?,?)")
    # a report that lists the star as healthy: the lineup is the full roster
    conn.execute(ins, (f"{DAY}T14:00:00Z", "nba_official", home, f"{home} Player 0", "Available", "x"))
    conn.commit()
    healthy = _by_game(predict.predict_day(conn, DAY, store=False))[g["game_id"]]
    assert healthy["inputs"]["availability"]["home"]["basis"] == "injury_report"
    # a newer report: he is out
    conn.execute(ins, (f"{DAY}T15:00:00Z", "nba_official", home, f"{home} Player 0", "Out", "knee"))
    conn.commit()
    out = _by_game(predict.predict_day(conn, DAY, store=False))[g["game_id"]]
    for key in ("headline", "opinion"):
        a, b = (healthy, out) if key == "headline" else (healthy["opinion"], out["opinion"])
        assert b["home_margin"] != a["home_margin"], key               # the absence is priced
        assert b["home_win_prob"] != a["home_win_prob"], key
    nav = out["inputs"]["availability"]
    assert nav["home"]["source"] == "nba_official" and nav["away"]["source"] is None
    (p,) = nav["home"]["players"]
    assert p["player"] == f"{home} Player 0" and p["p_play"] == 0.0 and p["how"] == "exact"
    # what the absence is worth = the difference between the two runs, from the home side
    (eff,) = [e for e in out["opinion"]["inputs"]["model_effects"]
              if e["player_id"] == p["player_id"]]
    assert eff["team"] == home and eff["side"] == "home"
    assert eff["effect_margin"] == pytest.approx(
        out["opinion"]["home_margin"] - healthy["opinion"]["home_margin"], abs=0.02)
    fac = {f["factor"]: f for f in out["opinion"]["inputs"]["factors"]}
    if "absences" in fac:                                # shown when it clears the noise floor
        assert fac["absences"]["team"] == home
        assert fac["absences"]["detail"] == f"{home} absences: {home} Player 0 (Out)"
        assert fac["absences"]["impact_pts"] == pytest.approx(eff["effect_margin"], abs=0.3)
    # the injuries the Brief lists agree with the ones priced
    assert [x["player"] for x in out["injuries"]["home"]] == [f"{home} Player 0"]


def test_factor_shape_in_the_stored_inputs():
    conn = synth_conn()
    rows = predict.predict_day(conn, DAY)
    for r in rows:
        for x in (r, r["opinion"]):
            fac = x["inputs"]["factors"]
            assert len(fac) <= 4
            for f in fac:
                assert set(f) == {"kind", "factor", "team", "side", "value", "impact_pts",
                                  "favors", "detail"}
                assert f["kind"] == "model_factor" and f["team"] in (x["home"], x["away"])
            assert [abs(f["impact_pts"]) for f in fac] == sorted(
                [abs(f["impact_pts"]) for f in fac], reverse=True)
            json.dumps(x["inputs"], allow_nan=False)
    stored = conn.execute("SELECT inputs_json FROM predictions WHERE model=?",
                          (predict.PRODUCTION_OPINION,)).fetchone()[0]
    assert json.loads(stored)["factors"]


class Counter:
    """Counts cold fits by wrapping production.fit_bundle."""
    def __init__(self, monkeypatch):
        self.n = 0
        real = production.fit_bundle

        def counting(*a, **k):
            self.n += 1
            return real(*a, **k)
        monkeypatch.setattr(production, "fit_bundle", counting)


def test_fitted_models_are_cached_per_day_and_reused_with_fresh_lines(monkeypatch, data_dir):
    conn = synth_conn()
    _closing_lines(conn, DAY)
    cnt = Counter(monkeypatch)
    first = predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 1 and first[0]["inputs"]["fit"]["cache"] == "miss"
    assert (data_dir / "models" / DAY.isoformat() / "bundle.pkl").exists()
    # later runs of the same day: no refit, but new lines and injuries are picked up
    _snapshot_line(conn, DAY, spread=-12.0)
    home = first[0]["home"]
    conn.execute("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
                 "reason) VALUES (?,?,?,?,?,?)",
                 (f"{DAY}T15:00:00Z", "nba_official", home, f"{home} Player 1", "Out", "x"))
    conn.commit()
    second = predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 1 and second[0]["inputs"]["fit"]["cache"] == "hit"
    assert second[0]["inputs"]["fit"]["fitted_at"] == first[0]["inputs"]["fit"]["fitted_at"]
    assert second[0]["market_home_spread"] == -12.0 and first[0]["market_home_spread"] != -12.0
    assert second[0]["home_margin"] != first[0]["home_margin"]
    assert second[0]["inputs"]["availability"]["home"]["basis"] == "injury_report"
    # a cached run equals a cold run with the same inputs
    cold = predict.predict_day(conn, DAY, store=False, use_cache=False)
    assert cnt.n == 2 and cold[0]["inputs"]["fit"]["cache"] == "miss"
    assert _strip(second)[0]["home_margin"] == _strip(cold)[0]["home_margin"]
    assert _strip(second)[0]["opinion"]["home_win_prob"] == _strip(cold)[0]["opinion"]["home_win_prob"]
    # explicit refit
    forced = predict.predict_day(conn, DAY, store=False, refit=True)
    assert cnt.n == 3 and forced[0]["inputs"]["fit"]["cache"] == "refit"


def test_cache_is_invalidated_when_new_results_arrive(monkeypatch, data_dir):
    conn = synth_conn()
    cnt = Counter(monkeypatch)
    predict.predict_day(conn, DAY, store=False)
    predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 1
    key0 = production.cache_key(conn, DAY, ["a"], predict.PRODUCTION_MODELS)
    # another result (even a late one that leaves the newest date alone) invalidates it ...
    cols = [r[1] for r in conn.execute("PRAGMA table_info(team_games)")]
    sel = ", ".join("'9' || substr(game_id, 2)" if c == "game_id" else
                    "'2023-09-15'" if c == "game_date" else c for c in cols)
    conn.execute(f"INSERT INTO team_games SELECT {sel} FROM team_games "
                 f"WHERE game_id = (SELECT MIN(game_id) FROM team_games)")
    conn.commit()
    key1 = production.cache_key(conn, DAY, ["a"], predict.PRODUCTION_MODELS)
    assert key1["n_team_games"] > key0["n_team_games"] and key1["through"] == key0["through"]
    predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 2
    # ... and so does a newer game date; results ON the day (which the model must not see) do not
    conn.execute("UPDATE team_games SET pts = pts + 3 WHERE game_date = ?", (DAY.isoformat(),))
    conn.commit()
    predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 2
    other = DAY + timedelta(days=1)
    k_later = production.cache_key(conn, other, ["a"], predict.PRODUCTION_MODELS)
    assert k_later["through"] == DAY.isoformat() != key1["through"]
    # a different slate or different frozen hyperparameters are different fits too
    assert production.cache_key(conn, DAY, ["b"], predict.PRODUCTION_MODELS) != key1


def test_old_model_dirs_are_pruned_and_a_corrupt_cache_is_just_a_miss(monkeypatch, data_dir):
    conn = synth_conn()
    old = data_dir / "models" / "2000-01-01"
    old.mkdir(parents=True)
    (old / "bundle.pkl").write_bytes(b"x")
    cnt = Counter(monkeypatch)
    predict.predict_day(conn, DAY, store=False)
    assert not old.exists() and cnt.n == 1
    (data_dir / "models" / DAY.isoformat() / "bundle.pkl").write_bytes(b"not a pickle")
    predict.predict_day(conn, DAY, store=False)
    assert cnt.n == 2


def test_backfill_with_weekly_refit_reuses_heads_but_not_features():
    conn = synth_conn()
    d0, d1 = day_date("2024-25", 10), day_date("2024-25", 16)
    res = predict.predict_days(conn, d0, d1, refit_days=3)
    assert res["days"] == 7 and res["games"] == 21 and res["fits"] == 3
    daily = predict.predict_days(synth_conn(), d0, d1)
    assert daily["fits"] == 7
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE model IN (?, ?)",
                     predict.PRODUCTION_MODELS).fetchone()[0]
    assert n == 42
