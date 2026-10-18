"""Daily predictions: fit on everything strictly before `day`, predict that day's games.

`predict_day(conn, day)` is what the pregame job and the Brief call. The production model is a
name in the model registry (`PRODUCTION_MODEL`), so swapping in the player-strength model is a
one-line change. Rows are appended to `predictions` (history is never rewritten) together with
the market line that was available and the injuries known at that moment.

No leakage, by construction
---------------------------
* Training data is `games` rows with `game_date < day`, nothing else.
* The games being predicted come from the schedule (`games` table; or, for past days that were
  never in `games`, from `team_games`), and their result columns are overwritten with
  placeholders before any model sees them. Schedule features (rest, back-to-backs, travel)
  use only dates and sites, so they are computed on the combined history + upcoming frame.
* `predict_days` (used by `--backfill-from`) builds the combined frame once and then, for each
  day, hands the model `train = date < d` and `test = date == d`, so every day is a true
  walk-forward fit.

Created-at convention: today/future days use the wall clock; past days (demo and backfill) use
`<day>T12:00:00Z` (07:00 ET, before any tip-off) so reruns are idempotent and the "latest
pre-tip prediction" view picks them up.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from . import config, injuries, odds_history, teams
from .model import dataset, elo as elo_mod, schedule, sites

log = logging.getLogger("courtside.predict")

PRODUCTION_MODEL = "elo_sched"
MIN_TRAIN_GAMES = 500
TOTAL_WINDOW = 20           # games per team in the total projection
TOTAL_SHRINK = 8.0          # pseudo-games of league average
_ID_TYPE = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}
_RESULT_COLS = ("home_pts", "away_pts", "margin", "home_win", "ot")


# ------------------------------------------------------------------ model registry

def model_factory(name: str, conn: sqlite3.Connection | None = None) -> Callable:
    """Factory for a registered model. Baselines first; the player-strength models (which need
    the player logs) are resolved lazily so importing this module stays cheap."""
    from .model import baselines
    if name in baselines.MODELS:
        return baselines.MODELS[name]
    if conn is None:
        raise KeyError(f"unknown model {name!r}")
    from .model import player_model, players
    reg, _cache = player_model.make_registry(lambda: players.load_player_games(conn))
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; known: "
                       f"{sorted(baselines.MODELS) + sorted(reg)}")
    return reg[name]


# ------------------------------------------------------------------ small helpers

def _as_date(d) -> date:
    return d if isinstance(d, date) and not isinstance(d, datetime) else \
        date.fromisoformat(str(d)[:10])


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_created_at(day: date) -> str:
    """Wall clock for today/future; a fixed pre-tip virtual time for past days."""
    if day >= config.today_nba_day():
        return _iso_now()
    return f"{day.isoformat()}T12:00:00Z"


def injury_cutoff(day: date, created_at: str) -> str:
    """Latest injury snapshot time usable for `day`: now for live days, else the end of the
    NBA day (05:00 UTC next day) for past days."""
    if day >= config.today_nba_day():
        return created_at
    return (datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            + timedelta(days=1, hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ------------------------------------------------------------------ frame assembly

def _upcoming_from_games(conn: sqlite3.Connection, day: date, skip_ids: set[str]) -> pd.DataFrame:
    """Frame rows (dataset.build_games columns) for scheduled games of `day` in `games`, with
    placeholder results. Preseason/All-Star/Cup-final ids and postponed games are skipped."""
    rows = []
    for g in conn.execute("SELECT game_id, season, home, away, status FROM games "
                          "WHERE nba_day=? ORDER BY game_id", (day.isoformat(),)):
        gid = g["game_id"]
        if gid in skip_ids or g["status"] == "postponed" or not g["home"] or not g["away"]:
            continue
        stype = _ID_TYPE.get(gid[:3])
        if stype is None:
            continue
        season = g["season"] or _season_of(day)
        try:
            h, a = teams.franchise(g["home"], season), teams.franchise(g["away"], season)
            site = sites.home_site(g["home"], season)
        except KeyError:
            log.warning("predict: unknown team in %s (%s@%s); skipped", gid, g["away"], g["home"])
            continue
        rows.append({
            "game_id": gid, "season": season, "season_type": stype,
            "game_date": pd.Timestamp(day), "home_raw": g["home"], "away_raw": g["away"],
            "home_pts": 0, "away_pts": 0, "margin": 0, "home_win": 0, "ot": 0,
            "home": h, "away": a, "is_neutral": False, "neutral_reason": "",
            "site": site.name, "site_lat": site.lat, "site_lon": site.lon, "site_tz": site.tz})
    return pd.DataFrame(rows)


def _season_of(d: date) -> str:
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def build_prediction_frame(conn: sqlite3.Connection, days: list[date]) -> pd.DataFrame:
    """History up to max(days) plus, for days whose games are not in `team_games` yet, the
    scheduled games from `games` (placeholder results). Schedule features added on the whole.

    Callers must mask result columns of any row on/after the day they predict."""
    hist = dataset.load_games(conn)
    last = pd.Timestamp(max(days))
    hist = hist[hist["game_date"] <= last]
    have = set(hist["game_id"])
    extra = [u for u in (_upcoming_from_games(conn, d, have) for d in sorted(set(days)))
             if not u.empty]
    full = pd.concat([hist, *extra], ignore_index=True) if extra else hist.copy()
    full = full.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    return schedule.add_schedule_features(full)


def _enough_history(conn: sqlite3.Connection) -> bool:
    n = conn.execute("SELECT COUNT(*) FROM team_games").fetchone()[0]
    return n >= 2 * MIN_TRAIN_GAMES


def _mask_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _RESULT_COLS:
        out[c] = 0
    return out


# ------------------------------------------------------------------ totals

def project_totals(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Expected game total: mean of the two teams' recent game totals (last TOTAL_WINDOW games,
    shrunk toward the league mean of the last ~400 games). A simple, leak-free baseline; the
    market total is reported next to it."""
    if len(train) == 0:
        return np.full(len(test), np.nan)
    league = float((train["home_pts"] + train["away_pts"]).tail(400).mean())
    tot = (train["home_pts"] + train["away_pts"]).to_numpy(float)
    long = pd.concat([
        pd.DataFrame({"team": train["home"].to_numpy(), "t": tot}),
        pd.DataFrame({"team": train["away"].to_numpy(), "t": tot})], ignore_index=True)
    # chronological order within a team is preserved by the stable concat/sort below
    order = np.argsort(np.concatenate([train.index.to_numpy(), train.index.to_numpy()]),
                       kind="stable")
    long = long.iloc[order]
    est: dict[str, float] = {}
    for team, g in long.groupby("team", sort=False):
        v = g["t"].to_numpy()[-TOTAL_WINDOW:]
        est[team] = (v.sum() + TOTAL_SHRINK * league) / (len(v) + TOTAL_SHRINK)
    return np.array([(est.get(h, league) + est.get(a, league)) / 2
                     for h, a in zip(test["home"], test["away"])])


# ------------------------------------------------------------------ market lines

def market_line(conn: sqlite3.Connection, game_id: str, tipoff_utc: str | None = None
                ) -> dict | None:
    """Latest pre-tip line: newest `odds_snapshots` batch (<= tip-off), else the closing line.

    {source, provider, captured_at, home_spread, total, home_ml, away_ml, home_prob}.
    Spread is the home spread (negative = home favoured); home_prob is the de-vigged moneyline."""
    sql = ("SELECT captured_at, source, provider, home_spread, total, home_ml, away_ml "
           "FROM odds_snapshots WHERE game_id=?")
    args: list = [game_id]
    if tipoff_utc:
        sql += " AND captured_at<=?"
        args.append(tipoff_utc)
    rows = conn.execute(sql + " ORDER BY captured_at DESC, id DESC", args).fetchall()
    line = None
    if rows:
        cap = rows[0]["captured_at"]
        grp = sorted((r for r in rows if r["captured_at"] == cap),
                     key=lambda r: 0 if r["source"] == "espn" else 1)
        line = {"source": grp[0]["source"], "provider": grp[0]["provider"], "captured_at": cap,
                "kind": "snapshot"}
        for k in ("home_spread", "total", "home_ml", "away_ml"):
            line[k] = next((r[k] for r in grp if r[k] is not None), None)
    else:
        cl = odds_history.consensus_line(conn, game_id)
        if cl:
            line = {"source": cl["source"], "provider": cl["provider"], "captured_at": None,
                    "kind": "closing", "home_spread": cl["home_spread"], "total": cl["total"],
                    "home_ml": cl["home_ml"], "away_ml": cl["away_ml"]}
    if line is None:
        return None
    line["home_prob"] = odds_history.devig_home_prob(line["home_ml"], line["away_ml"])
    if all(line[k] is None for k in ("home_spread", "total", "home_prob")):
        return None
    return line


# ------------------------------------------------------------------ prediction

def _factors(row: pd.Series) -> dict:
    keys = ("rest", "b2b", "three_in_four", "four_in_six", "games_last7", "travel_km",
            "tz_change")
    out = {}
    for side in ("home", "away"):
        out[side] = {k: _py(row[f"{side}_{k}"]) for k in keys}
    out["rest_diff"] = _py(row["rest_diff"])
    out["is_neutral"] = bool(row["is_neutral"])
    out["season_type"] = row["season_type"]
    return out


def _predict_one_day(conn: sqlite3.Connection, full: pd.DataFrame, day: date, model_name: str,
                     created_at: str, factory: Callable, *, with_injuries: bool = True,
                     with_market: bool = True) -> list[dict]:
    ts = pd.Timestamp(day)
    train = full[full["game_date"] < ts]
    test = full[full["game_date"] == ts]
    if test.empty:
        return []
    if len(train) < MIN_TRAIN_GAMES:
        log.warning("predict %s: only %d training games; skipped", day, len(train))
        return []
    test = _mask_results(test)
    model = factory()
    model.fit(train)
    p = np.asarray(model.predict_proba(test), dtype=float)
    mg = (np.asarray(model.predict_margin(test), dtype=float)
          if hasattr(model, "predict_margin") else np.full(len(test), np.nan))
    tot = project_totals(train, test)
    e = elo_mod.run_elo(pd.concat([train, test]), elo_mod.TUNED).iloc[len(train):]

    tips = {r["game_id"]: (r["tipoff_utc"], r["status"]) for r in conn.execute(
        "SELECT game_id, tipoff_utc, status FROM games WHERE nba_day=?", (day.isoformat(),))}
    inj_at = injury_cutoff(day, created_at)
    inj = injuries.current(conn, inj_at, set(test["home"]) | set(test["away"])) \
        if with_injuries else {"by_team": {}, "batches": {}}

    out = []
    for i, (_, r) in enumerate(test.iterrows()):
        gid = r["game_id"]
        tip, status = tips.get(gid, (None, None))
        mk = market_line(conn, gid, tip) if with_market else None
        mkt_prob = mk["home_prob"] if mk else None
        team_inj = {"home": inj["by_team"].get(r["home"], []),
                    "away": inj["by_team"].get(r["away"], [])}
        inputs = {
            "model": model_name,
            "train_games": int(len(train)), "train_through": train["game_date"].max()
            .strftime("%Y-%m-%d"),
            "schedule": _factors(r),
            "elo": {"home": _py(e["elo_home_pre"].iat[i]), "away": _py(e["elo_away_pre"].iat[i])},
            "market": ({k: _py(mk[k]) for k in ("source", "provider", "captured_at", "kind",
                                                "home_spread", "total", "home_ml", "away_ml",
                                                "home_prob")} if mk else None),
            "injuries": {s: [{k: x[k] for k in ("player", "status", "reason", "source")}
                             for x in team_inj[s]] for s in team_inj},
            "injuries_batches": inj.get("batches", {}),
        }
        out.append({
            "game_id": gid, "nba_day": day.isoformat(), "model": model_name,
            "created_at": created_at, "home": r["home"], "away": r["away"],
            "season": r["season"], "season_type": r["season_type"],
            "tipoff_utc": tip, "status": status,
            "home_win_prob": _py(p[i]), "home_margin": _py(mg[i]), "total": _py(tot[i]),
            "market_home_spread": _py(mk["home_spread"]) if mk else None,
            "market_total": _py(mk["total"]) if mk else None,
            "market_home_prob": _py(mkt_prob),
            "inputs": inputs, "injuries": inputs["injuries"],
        })
    return out


def store_predictions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Append rows to `predictions` (INSERT OR REPLACE only collides on an identical
    (game_id, model, created_at) key, i.e. a rerun of the same virtual time)."""
    if not rows:
        return 0
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO predictions (game_id, model, created_at, home_win_prob, "
            "home_margin, total, market_home_spread, market_total, market_home_prob, "
            "inputs_json, nba_day, home, away) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r["game_id"], r["model"], r["created_at"], r["home_win_prob"], r["home_margin"],
              r["total"], r["market_home_spread"], r["market_total"], r["market_home_prob"],
              json.dumps(r["inputs"], allow_nan=False), r["nba_day"], r["home"], r["away"])
             for r in rows])
    return len(rows)


def predict_day(conn: sqlite3.Connection, day, *, model: str | None = None, store: bool = True,
                created_at: str | None = None) -> list[dict]:
    """Predict every game of `day` with a model fit on games strictly before `day`.

    Returns one dict per game: game_id, nba_day, home, away, model, created_at, tipoff_utc,
    home_win_prob, home_margin, total, market_home_spread/total/prob, `inputs` (schedule
    factors, Elo, market, injuries) and `injuries` ({"home": [...], "away": [...]}, latest
    Out/Doubtful/Questionable per team). Stores the rows in `predictions` unless store=False."""
    day = _as_date(day)
    name = model or PRODUCTION_MODEL
    created_at = created_at or default_created_at(day)
    if not _enough_history(conn):
        log.warning("predict %s: team_games is (nearly) empty; no predictions", day)
        return []
    full = build_prediction_frame(conn, [day])
    rows = _predict_one_day(conn, full, day, name, created_at, model_factory(name, conn))
    if store:
        store_predictions(conn, rows)
    return rows


def predict_days(conn: sqlite3.Connection, start, end, *, model: str | None = None,
                 log_fn: Callable[[str], None] | None = None) -> dict:
    """Walk-forward predictions for every game day in [start, end] (e.g. a demo backfill).

    The combined frame is built once; each day is fit on `date < day` only. Market lines are the
    closing lines in `closing_lines`; injuries use whatever snapshots exist (usually none for
    old days). Returns {"days": n, "games": n, "skipped_days": [...]}."""
    start, end = _as_date(start), _as_date(end)
    name = model or PRODUCTION_MODEL
    if not _enough_history(conn):
        return {"days": 0, "games": 0, "skipped_days": [], "model": name}
    full = build_prediction_frame(conn, [start, end])
    played = sorted({d.date() for d in full["game_date"]
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
    factory = model_factory(name, conn)
    n_games, skipped = 0, []
    for k, d in enumerate(played):
        rows = _predict_one_day(conn, full, d, name, default_created_at(d), factory)
        if not rows:
            skipped.append(d.isoformat())
        n_games += store_predictions(conn, rows)
        if log_fn and (k % 10 == 0 or k == len(played) - 1):
            log_fn(f"  {d} ({k + 1}/{len(played)} days, {n_games} predictions)")
    return {"days": len(played), "games": n_games, "skipped_days": skipped, "model": name}


# ------------------------------------------------------------------ reading back

def latest_pretip(conn: sqlite3.Connection, game_ids: list[str] | None = None, *,
                  day: str | None = None, model: str | None = None) -> dict[str, dict]:
    """game_id -> stored prediction row (dict, `inputs` parsed), via the
    `latest_pretip_predictions` view. Filter by game ids or by nba_day."""
    model = model or PRODUCTION_MODEL
    sql, args = "SELECT * FROM latest_pretip_predictions WHERE model=?", [model]
    if day is not None:
        sql += " AND nba_day=?"
        args.append(day)
    if game_ids is not None:
        if not game_ids:
            return {}
        sql += f" AND game_id IN ({','.join('?' * len(game_ids))})"
        args += list(game_ids)
    out = {}
    for r in conn.execute(sql, args):
        d = dict(r)
        d["inputs"] = json.loads(d.pop("inputs_json")) if r["inputs_json"] else {}
        out[d["game_id"]] = d
    return out
