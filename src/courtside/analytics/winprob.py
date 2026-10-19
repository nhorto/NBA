"""In-game win probability.

A logistic model on a Brownian-motion style feature set, trained on cdn.nba.com play-by-play from
2019-20 on (``courtside backfill pbp``, ``courtside winprob train``). Production inference is
pure Python on the coefficients committed in ``winprob_coef.json``; scikit-learn is only needed to
retrain.

State (all from the HOME team's point of view, evaluated *after* each event):

* ``margin``  home score - away score
* ``t``       seconds remaining. Regulation: to the end of the 4th. Overtime: to the end of that
              OT period. A tie at 0:00 of regulation is treated as the start of an OT (``t = 300``)
* ``exp``     pregame expected home margin = ``-home_spread`` (``home_spread`` uses the betting
              convention of ``closing_lines``: -6.5 means home favoured by 6.5). Missing -> 0
* ``poss``    +1 home has the ball, -1 away, 0 unknown. Derived by counting possessions
              (``Event.ends_possession``), not from the feed's ``possession`` field, which names the
              team that last acted, not who has the ball next

Every feature is odd under swapping the teams (margin, spread and possession all flip sign) and
there is no intercept, so the model is exactly symmetric: a tied pick'em game is 50.0%.

Features: see ``FEATURES``. ``u = sqrt(t + EPS)``. The pregame prior is ``exp * t / 2880``, i.e. the
spread decays linearly with the time left, and is then scaled by ``1/u`` like the margin, so at
tip-off the spread dominates and by the final buzzer only the score matters.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from courtside.analytics.pbp import (
    AWAY, HOME, OT_LEN, REG_PERIODS, Event, other, period_end,
)

COEF_PATH = Path(__file__).with_name("winprob_coef.json")
REG_SECONDS = 2880.0
EPS = 4.0                     # seconds; keeps m/sqrt(t) finite at the buzzer
FEATURES = (
    "margin_over_u",          # m / u
    "prior_over_u",           # exp * (t/2880) / u
    "poss_over_u",            # poss / u
    "poss_late",              # poss * exp(-t/20)       ball in the last ~half minute
    "poss_mid",               # poss * exp(-t/120)
    "margin_over_t",          # m / (t + 20)            discrete score effects near the end
    "poss_late_close",        # poss * exp(-t/30) / (1 + |m|)   the ball matters in close endgames
    "margin_over_u_ot",       # m / u * ot
    "lead_1s", "lead_2s", "lead_4s", "lead_8s",             # sign(m) * exp(-t/k)    last-seconds locks
    "close_ball_1s", "close_ball_2s", "close_ball_4s",      # poss * exp(-t/k) / (1 + |m|)
)
_LEAD_K = (1.0, 2.0, 4.0, 8.0)
_BALL_K = (1.0, 2.0, 4.0)

SCHEMA = 1


# ------------------------------------------------------------------------- state

def seconds_left(period: int, elapsed: float, margin: int) -> float:
    """Seconds remaining in the game (regulation) / in the current OT period."""
    if period <= REG_PERIODS:
        t = period_end(REG_PERIODS) - elapsed
        if period == REG_PERIODS and t <= 0.05 and margin == 0:
            return OT_LEN               # tied at the buzzer -> overtime follows
        return max(t, 0.0)
    return max(period_end(period) - elapsed, 0.0)


def event_states(events: Sequence[Event]) -> list[tuple[int, float, int, int]]:
    """Per event: ``(margin, t, period, poss)`` of the state *after* the event."""
    out = []
    owner: str | None = None
    for e in events:
        if e.is_period_start:
            owner = None                  # the arrow/tip decides; the first action reveals it
        if e.is_jumpball and e.side:
            owner = e.side
        elif e.ends_possession and e.poss_side:
            owner = other(e.poss_side)
        elif owner is None and e.side and (e.is_fga or e.is_ft or e.is_turnover):
            owner = e.side
        if e.is_period_end:
            owner = None                  # the next period starts with a dead ball / jump ball
        margin = e.score_home - e.score_away
        poss = 1 if owner == HOME else -1 if owner == AWAY else 0
        out.append((margin, seconds_left(e.period, e.elapsed, margin), e.period, poss))
    return out


# ------------------------------------------------------------------------- model

def features(margin: float, t: float, exp: float, poss: int, period: int) -> list[float]:
    u = math.sqrt(t + EPS)
    ot = 1.0 if period > REG_PERIODS else 0.0
    sgn = float((margin > 0) - (margin < 0))
    return [
        margin / u,
        exp * (t / REG_SECONDS) / u,
        poss / u,
        poss * math.exp(-t / 20.0),
        poss * math.exp(-t / 120.0),
        margin / (t + 20.0),
        poss * math.exp(-t / 30.0) / (1.0 + abs(margin)),
        margin / u * ot,
        *[sgn * math.exp(-t / k) for k in _LEAD_K],
        *[poss * math.exp(-t / k) / (1.0 + abs(margin)) for k in _BALL_K],
    ]


def feature_matrix(margin, t, exp, poss, period):
    """Vectorised ``features`` over numpy arrays -> (n, len(FEATURES))."""
    import numpy as np
    margin, t, exp, poss = (np.asarray(a, dtype=np.float64) for a in (margin, t, exp, poss))
    ot = (np.asarray(period) > REG_PERIODS).astype(np.float64)
    u = np.sqrt(t + EPS)
    cols = [
        margin / u,
        exp * (t / REG_SECONDS) / u,
        poss / u,
        poss * np.exp(-t / 20.0),
        poss * np.exp(-t / 120.0),
        margin / (t + 20.0),
        poss * np.exp(-t / 30.0) / (1.0 + np.abs(margin)),
        margin / u * ot,
        *[np.sign(margin) * np.exp(-t / k) for k in _LEAD_K],
        *[poss * np.exp(-t / k) / (1.0 + np.abs(margin)) for k in _BALL_K],
    ]
    return np.stack(cols, axis=1)


@lru_cache(maxsize=1)
def load_coef(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else COEF_PATH
    d = json.loads(p.read_text())
    if tuple(d["features"]) != FEATURES:
        raise ValueError(f"{p}: coefficient features {d['features']} do not match {FEATURES}")
    return d


def win_prob(margin: float, t: float, exp: float = 0.0, poss: int = 0, period: int = 1,
             coef: dict[str, Any] | None = None) -> float:
    """P(home wins) for one state."""
    w = (coef or load_coef())["coef"]
    z = sum(a * b for a, b in zip(w, features(margin, t, exp, poss, period)))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def spread_to_exp(home_spread: float | None) -> float:
    """Betting-convention home spread (-6.5 = home favoured) -> expected home margin."""
    return 0.0 if home_spread is None else -float(home_spread)


# ------------------------------------------------------------------------- series

def series(events: Sequence[Event], pregame_spread: float | None = None,
           coef: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One point per event: ``{idx, period, elapsed, margin, home_wp}`` (state after the event).

    ``pregame_spread`` is the home spread (negative = home favoured); ``None`` means no prior.
    Every game starts from its tip-off probability, so the first point is the pregame WP.
    """
    exp = spread_to_exp(pregame_spread)
    c = coef or load_coef()
    out = []
    for e, (margin, t, period, poss) in zip(events, event_states(events)):
        if period >= REG_PERIODS and t <= 0.05 and margin != 0:
            wp = 1.0 if margin > 0 else 0.0     # the clock is at 0:00 and the game is not tied
        else:
            wp = round(win_prob(margin, t, exp, poss, period, c), 4)
        out.append({"idx": e.idx, "period": e.period, "elapsed": e.elapsed, "margin": margin,
                    "home_wp": wp})
    return out


def wp_pairs(points: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    """``series`` -> the ``[(event_idx, p_home)]`` shape ``game_story.turning_point`` takes."""
    return [(p["idx"], p["home_wp"]) for p in points]


def timeline(points: Sequence[dict[str, Any]], n: int = 100) -> list[dict[str, Any]]:
    """About ``n`` evenly spaced (by elapsed time) points for a chart: the last point at or before
    each grid time, always including tip-off and the final state."""
    if not points:
        return []
    end = points[-1]["elapsed"]
    out: list[dict[str, Any]] = []
    j = 0
    for k in range(n):
        target = end * k / max(n - 1, 1)
        while j + 1 < len(points) and points[j + 1]["elapsed"] <= target:
            j += 1
        p = points[j]
        row = {"elapsed": round(target, 1), "margin": p["margin"], "home_wp": p["home_wp"]}
        if not out or out[-1] != row:
            out.append(row)
    return out


def swings(points: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    """``[(idx, delta_home_wp)]`` for every event whose WP changed relative to the previous one."""
    return [(b["idx"], b["home_wp"] - a["home_wp"]) for a, b in zip(points, points[1:])
            if b["home_wp"] != a["home_wp"]]


# ------------------------------------------------------------------------- training

def _sample_rows(pbp: dict, home_tricode: str, away_tricode: str, exp: float):
    """Sampled training rows for one game from the raw payload, or None if unusable."""
    from courtside.analytics.pbp import normalize
    ids: dict[str, int] = {}
    for a in (pbp.get("game") or {}).get("actions") or []:
        tc, tid = a.get("teamTricode"), a.get("teamId")
        if tc and tid:
            ids[tc] = int(tid)
        if home_tricode in ids and away_tricode in ids:
            break
    if home_tricode not in ids or away_tricode not in ids:
        return None
    box = {"game": {"homeTeam": {"teamId": ids[home_tricode]},
                    "awayTeam": {"teamId": ids[away_tricode]}}}
    events = normalize(pbp, box)
    if len(events) < 100:
        return None
    last = events[-1]
    if last.score_home == last.score_away:
        return None
    won = 1 if last.score_home > last.score_away else 0
    rows = []
    for e, (margin, t, period, poss) in zip(events, event_states(events)):
        if e.points or e.ends_possession or e.is_period_start or e.is_period_end:
            rows.append((margin, t, exp, poss, period, e.elapsed, won))
    return rows


def _load_game(args):
    path, home, away, exp = args
    from courtside import archive
    try:
        return _sample_rows(archive.load_raw(path), home, away, exp)
    except Exception:  # noqa: BLE001 - a corrupt payload must not stop a training run
        return None


def build_samples(conn=None, workers: int = 4, seasons: Sequence[str] | None = None):
    """Walk the archived play-by-play and build the training table.

    Returns a pandas DataFrame with ``game_id, season, margin, t, exp, poss, period, elapsed,
    won`` where ``exp`` is the expected home margin from the consensus closing line (0 if none).
    """
    import concurrent.futures as cf

    import numpy as np
    import pandas as pd

    from courtside import backfill_pbp, db, odds_history
    own = conn is None
    conn = conn or db.connect()
    try:
        index = backfill_pbp.archived_index()
        q = ("SELECT game_id, season, MAX(CASE WHEN is_home=1 THEN team END) AS home, "
             "MAX(CASE WHEN is_home=0 THEN team END) AS away FROM team_games "
             "WHERE season_type IN ('Regular Season','Playoffs','PlayIn') GROUP BY game_id, season")
        games = [r for r in conn.execute(q) if r["game_id"] in index
                 and (seasons is None or r["season"] in seasons) and r["home"] and r["away"]]
        jobs, meta = [], []
        n_line = 0
        for r in games:
            line = odds_history.consensus_line(conn, r["game_id"])
            hs = line["home_spread"] if line else None
            n_line += hs is not None
            jobs.append((index[r["game_id"]], r["home"], r["away"], spread_to_exp(hs)))
            meta.append((r["game_id"], r["season"]))
    finally:
        if own:
            conn.close()
    frames = []
    with cf.ProcessPoolExecutor(workers) as ex:
        for (gid, season), rows in zip(meta, ex.map(_load_game, jobs, chunksize=16)):
            if not rows:
                continue
            a = np.asarray(rows, dtype=np.float64)
            df = pd.DataFrame(a, columns=["margin", "t", "exp", "poss", "period", "elapsed", "won"])
            df.insert(0, "season", season)
            df.insert(0, "game_id", gid)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["games_with_line"] = n_line
    out.attrs["games"] = len(games)
    return out


def metrics(y, p) -> dict[str, float]:
    import numpy as np
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


TIME_BUCKETS = (("Q1", 2160, 2880.01), ("Q2", 1440, 2160), ("Q3", 720, 1440),
                ("Q4 >5:00", 300, 720), ("Q4 5:00-1:00", 60, 300), ("last 1:00", 0, 60))


def calibration_by_time(df, p) -> list[dict[str, Any]]:
    """Per time bucket (regulation only, by seconds remaining): n, mean predicted, actual rate,
    Brier, and the largest decile calibration gap."""
    import numpy as np
    out = []
    reg = (df["period"].to_numpy() <= REG_PERIODS)
    t = df["t"].to_numpy()
    y = df["won"].to_numpy()
    for name, lo, hi in TIME_BUCKETS:
        m = reg & (t >= lo) & (t < hi)
        if not m.any():
            continue
        pm, ym = p[m], y[m]
        bins = np.clip((pm * 10).astype(int), 0, 9)
        gaps = [abs(pm[bins == b].mean() - ym[bins == b].mean())
                for b in range(10) if (bins == b).sum() >= 200]
        out.append({"bucket": name, "n": int(m.sum()), "pred": float(pm.mean()),
                    "actual": float(ym.mean()), "brier": float(np.mean((pm - ym) ** 2)),
                    "max_decile_gap": float(max(gaps)) if gaps else None})
    m = ~reg
    if m.any():
        out.append({"bucket": "overtime", "n": int(m.sum()), "pred": float(p[m].mean()),
                    "actual": float(y[m].mean()), "brier": float(np.mean((p[m] - y[m]) ** 2)),
                    "max_decile_gap": None})
    return out


def reliability(y, p, bins: int = 10) -> list[dict[str, float]]:
    import numpy as np
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return [{"bin": f"{k / bins:.1f}-{(k + 1) / bins:.1f}", "n": int((b == k).sum()),
             "pred": float(p[b == k].mean()), "actual": float(y[b == k].mean())}
            for k in range(bins) if (b == k).any()]


def fit_logistic(df, C: float = 1e3):
    """Fit on a samples frame; returns the coefficient list aligned with ``FEATURES``.

    Rows are unweighted, matching the by-event test metric.
    """
    from sklearn.linear_model import LogisticRegression
    X = feature_matrix(df["margin"], df["t"], df["exp"], df["poss"], df["period"])
    y = df["won"].to_numpy()
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
    clf.fit(X, y)
    return clf.coef_[0].tolist()


def predict_frame(df, coef: Sequence[float]):
    import numpy as np
    X = feature_matrix(df["margin"], df["t"], df["exp"], df["poss"], df["period"])
    return 1.0 / (1.0 + np.exp(-(X @ np.asarray(coef))))


def sanity_cases(coef: dict[str, Any]) -> dict[str, float]:
    """Values quoted in docs/ANALYTICS.md; also asserted in the tests."""
    wp = lambda **k: round(win_prob(coef=coef, **k), 4)  # noqa: E731
    return {
        "tipoff_tied_spread0": wp(margin=0, t=2880, exp=0, poss=0, period=1),
        "up3_5s_home_ball": wp(margin=3, t=5, exp=0, poss=1, period=4),
        "up3_5s_away_ball": wp(margin=3, t=5, exp=0, poss=-1, period=4),
        "down3_5s_home_ball": wp(margin=-3, t=5, exp=0, poss=1, period=4),
        "up1_30s_home_ball": wp(margin=1, t=30, exp=0, poss=1, period=4),
        "up10_2min": wp(margin=10, t=120, exp=0, poss=0, period=4),
        "up10_5min": wp(margin=10, t=300, exp=0, poss=0, period=4),
        "up20_halftime": wp(margin=20, t=1440, exp=0, poss=0, period=3),
        "tied_ot_start": wp(margin=0, t=300, exp=0, poss=0, period=5),
        "tied_q4_start": wp(margin=0, t=720, exp=0, poss=0, period=4),
        "tipoff_home_fav_7": wp(margin=0, t=2880, exp=7, poss=0, period=1),
    }


def train_cli(args) -> int:
    """``courtside winprob train``: build samples, fit on seasons before ``--test-season``, report
    on the test season, then write the coefficients (fit on the training seasons, or on every
    season with ``--final``)."""
    import numpy as np

    from courtside import config
    df = build_samples()
    games = df["game_id"].nunique()
    print(f"{len(df)} sampled states from {games} games "
          f"({df.attrs.get('games_with_line', '?')}/{df.attrs.get('games', '?')} with a closing line)")
    test = df["season"] == args.test_season
    train = df["season"] < args.test_season
    if not test.any() or not train.any():
        print("error: need games both before and in the test season")
        return 2
    coef = fit_logistic(df[train])
    p = predict_frame(df[test], coef)
    m = metrics(df.loc[test, "won"], p)
    print(f"train seasons {df.loc[train, 'season'].min()}..{df.loc[train, 'season'].max()} "
          f"({df[train]['game_id'].nunique()} games), test {args.test_season} "
          f"({df[test]['game_id'].nunique()} games)")
    print(f"test log-loss {m['logloss']:.4f}  Brier {m['brier']:.4f}  (n={m['n']})")
    print("calibration by time bucket (regulation):")
    for r in calibration_by_time(df[test], p):
        gap = "" if r["max_decile_gap"] is None else f"  max decile gap {r['max_decile_gap']:.3f}"
        print(f"  {r['bucket']:<14} n={r['n']:>7}  pred {r['pred']:.3f}  actual {r['actual']:.3f}  "
              f"brier {r['brier']:.4f}{gap}")
    print("reliability (all states):")
    for r in reliability(df.loc[test, "won"], p):
        print(f"  {r['bin']}  n={r['n']:>7}  pred {r['pred']:.3f}  actual {r['actual']:.3f}")
    final = coef
    if getattr(args, "final", False):
        final = fit_logistic(df)
    doc = {"schema": SCHEMA, "features": list(FEATURES), "coef": final, "eps": EPS,
           "trained_on": {"seasons": sorted(df["season"].unique()) if args.final
                          else sorted(df.loc[train, "season"].unique()),
                          "games": int(games if args.final else df[train]["game_id"].nunique()),
                          "states": int(len(df) if args.final else train.sum())},
           "test": {"season": args.test_season, **m,
                    "by_time": calibration_by_time(df[test], p)}}
    sane = sanity_cases(doc)
    print("sanity:", json.dumps(sane))
    if not args.no_write:
        COEF_PATH.write_text(json.dumps(doc, indent=1) + "\n")
        out = config.data_dir() / "models"
        out.mkdir(parents=True, exist_ok=True)
        (out / "winprob_v1.json").write_text(json.dumps(doc, indent=1) + "\n")
        np.savez(out / "winprob_v1.npz", coef=np.asarray(final), features=np.asarray(FEATURES))
        load_coef.cache_clear()
        print(f"wrote {COEF_PATH} and {out}/winprob_v1.json|npz")
    return 0
