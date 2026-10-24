"""Daily predictions: fit on everything strictly before `day`, predict that day's games.

`predict_day(conn, day)` is what the pregame job and the Brief call. Production is a *pair* of
stacked LightGBM models (docs/MODEL.md, rung 3), both stored in `predictions` (two `model` rows
per game, appended, never rewritten):

* `PRODUCTION_WINPROB = "gbm_market"`: the headline win probability, margin and total. It sees
  the live market line (`mkt_margin`, `mkt_total`: the newest pre-tip `odds_snapshots` batch,
  else `closing_lines`), so it is about as good as the market and is what the Brief shows.
* `PRODUCTION_OPINION = "gbm_opinion"`: "our independent number" with no market input at all.
  The Brief's "our number vs market" edge is this model minus the market, so the comparison
  means something.

News since the line (IMPROVEMENTS G1, `model.line_news`, COURTSIDE_NEWS_SINCE_LINE, default
OFF): `gbm_market` was fit on closing lines, which already price the day's injury news. ON, a
snapshot line is moved by beta x (gbm_opinion's margin with the lineups priced now minus with
the reports as they stood at the line's `captured_at`) before the headline model sees it; the
market rows' model_version gets `+nsl` and `inputs.news_since_line` records the line capture
time, the deltas and the shift. Closing lines are never moved. OFF never enters that code.

Absence scale (IMPROVEMENTS G4, `model.absence_scale`, COURTSIDE_ABSENCE_SCALE, default OFF):
ON, the injury-report lineup strengths are pulled toward the full roster by a scale s (both
models' rows get `+as`). OFF never enters that code.

Injuries are priced: the newest usable report (official over ESPN when it is the newer one;
`injuries.availability`) becomes an expected lineup per team, and the lineup strength replaces
the engine's availability (`production.apply_overrides`). Days without a report keep the
engine's own availability: the rotation regulars who played (leak-safe, past/demo days only) or
the full recent roster (live days). Each row's `inputs.availability.<side>.basis` says which.

Fits are cached per NBA day (`production.py`): the first run of the day fits (~1 min), later
runs only re-price with fresh lines and injuries. Any other `model=` name (the Elo baselines)
still goes through the one-model path below.

Lineage (docs/MLOPS.md): each production row carries `model_version` (the registry version that
produced it, `mlops.registry`; NULL when the registry is empty) and `role`:

* `champion`: `model` = gbm_market / gbm_opinion, shown to readers.
* `shadow`: `model` = the shadow's version id (e.g. "gbm_opinion@v2-1a2b3c4d"), logged for the
  online gate only. Readers (`latest_pretip`, the Brief, the scorecard) only see
  `role IS NULL OR role = 'champion'` (NULL = rows from before the registry).

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

from . import config, injuries, odds_history, production, teams
from .model import absence_scale, dataset, elo as elo_mod, gbm, line_news, schedule, sites

log = logging.getLogger("courtside.predict")

PRODUCTION_WINPROB = "gbm_market"      # headline probability / margin / total (mirrors gbm.*)
PRODUCTION_OPINION = "gbm_opinion"     # "our number" for the edge vs the market
PRODUCTION_MODELS = (PRODUCTION_WINPROB, PRODUCTION_OPINION)
PRODUCTION_MODEL = PRODUCTION_WINPROB  # the model name the Brief and scorecard read by default
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
    reg, _cache = player_model.make_registry(lambda: players.load_player_games(conn), conn=conn)
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


def injury_cutoff(day: date, created_at: str, tip: str | None = None) -> str:
    """Latest injury snapshot time usable for `day`.

    Live days (today/future): `created_at`, i.e. now. Past days (replays, backfills): what a
    live pre-tip run could have seen (IMPROVEMENTS B3), the game's tip-off `tip` (ISO UTC) when
    given, else `created_at` (the 12:00Z virtual time of `default_created_at`), and never later
    than the end of the NBA day (05:00 UTC next day). Before B3 every past day used the end of
    the day, so replays could price news published after the tip. Slate-level callers without
    a tip and with a wall-clock `created_at` (the Brief rebuilding a past day) still get the
    end of the day."""
    if day >= config.today_nba_day():
        return created_at
    eod = (datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
           + timedelta(days=1, hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t = _iso_utc(tip) if tip else created_at
    return min(t, eod)


def _iso_utc(ts: str) -> str:
    """Any ISO timestamp ('...Z', '+00:00', no seconds) -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _past_tips(conn: sqlite3.Connection, day: date, tips: dict) -> dict[str, str]:
    """game_id -> tip-off (ISO UTC) for a past day: the `games` table, else
    `proj.availability.day_games` (history seasons have no `games` rows: the archived ESPN
    scoreboard, else its 19:30 ET default, the same tips the DFS availability model uses)."""
    out = {g: t for g, (t, _) in tips.items() if t}
    from .proj import availability as pav          # past days only: live runs never import it
    for r in pav.day_games(conn, day).itertuples():
        if r.game_id not in out:
            out[r.game_id] = injuries.iso(r.tip)
    return out


def _injury_groups(conn: sqlite3.Connection, day: date, created_at: str, test: pd.DataFrame,
                   tips: dict) -> list[tuple[str, list[int] | None]]:
    """[(injury cutoff, row positions of `test`)]. Live days: one cutoff for the whole slate,
    [(created_at, None)], exactly as before B3. Past days: one group per distinct per-game
    cutoff (`injury_cutoff` with the game's tip)."""
    if day >= config.today_nba_day():
        return [(injury_cutoff(day, created_at), None)]
    pt = _past_tips(conn, day, tips)
    groups: dict[str, list[int]] = {}
    for i, gid in enumerate(test["game_id"]):
        groups.setdefault(injury_cutoff(day, created_at, pt.get(gid)), []).append(i)
    return sorted(groups.items())


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
    inj_of: dict[int, dict] = {}
    inj_none = {"by_team": {}, "batches": {}}
    if with_injuries:
        for inj_at, idx in _injury_groups(conn, day, created_at, test, tips):
            sub = test if idx is None else test.iloc[idx]
            g_inj = injuries.current(conn, inj_at, set(sub["home"]) | set(sub["away"]),
                                     game_day=day)
            for k in (range(len(test)) if idx is None else idx):
                inj_of[k] = g_inj

    out = []
    for i, (_, r) in enumerate(test.iterrows()):
        inj = inj_of.get(i, inj_none)
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


def has_lineage(conn: sqlite3.Connection) -> bool:
    """Does `predictions` have the lineage columns (migration 5)?"""
    return any(r[1] == "role" for r in conn.execute("PRAGMA table_info(predictions)"))


def shown_roles_sql(conn: sqlite3.Connection, alias: str = "") -> str:
    """SQL condition (prefixed with AND) restricting prediction rows to what readers may see:
    champion rows and legacy rows without a role. Empty on a pre-lineage schema."""
    if not has_lineage(conn):
        return ""
    a = f"{alias}." if alias else ""
    return f" AND ({a}role IS NULL OR {a}role = 'champion')"


def store_predictions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Append rows to `predictions` (INSERT OR REPLACE only collides on an identical
    (game_id, model, created_at) key, i.e. a rerun of the same virtual time). Lineage columns
    (`model_version`, `role`) are written when the schema has them."""
    if not rows:
        return 0
    cols = ["game_id", "model", "created_at", "home_win_prob", "home_margin", "total",
            "market_home_spread", "market_total", "market_home_prob", "inputs_json", "nba_day",
            "home", "away"]
    lineage = has_lineage(conn)
    if lineage:
        cols += ["model_version", "role"]
    vals = []
    for r in rows:
        v = [r["game_id"], r["model"], r["created_at"], r["home_win_prob"], r["home_margin"],
             r["total"], r["market_home_spread"], r["market_total"], r["market_home_prob"],
             json.dumps(r["inputs"], allow_nan=False), r["nba_day"], r["home"], r["away"]]
        if lineage:
            v += [r.get("model_version"), r.get("role")]
        vals.append(v)
    with conn:
        conn.executemany(f"INSERT OR REPLACE INTO predictions ({', '.join(cols)}) "
                         f"VALUES ({','.join('?' * len(cols))})", vals)
    return len(rows)


def _is_production(model: str | None) -> bool:
    return model is None or model in PRODUCTION_MODELS


def _flat(rows: list[dict]) -> list[dict]:
    """Headline rows with their attached `opinion` and `shadows` rows -> every row to store."""
    return [x for r in rows for x in (r, r.get("opinion"), *r.get("shadows", ())) if x]


def predict_day(conn: sqlite3.Connection, day, *, model: str | None = None, store: bool = True,
                created_at: str | None = None, refit: bool = False,
                use_cache: bool = True) -> list[dict]:
    """Predict every game of `day` with models fit on games strictly before `day`.

    Default (`model=None`): the production pair, fit with the registry champions' params.
    Returns one dict per game (the headline `gbm_market` row) with the `gbm_opinion` row under
    `"opinion"` and any shadow versions' rows under `"shadows"`; all are stored (shadows with
    role='shadow', `model` = version id). Row keys (+ `model_version`, `role`):
    game_id, nba_day, home, away, model, created_at, tipoff_utc, home_win_prob, home_margin,
    total, market_home_spread/total/prob, `inputs` (schedule factors, Elo, market, availability,
    model factors, injury effects, fit info) and `injuries` ({"home": [...], "away": [...]},
    latest Out/Doubtful/Questionable per team). The fitted models are cached per NBA day and
    reused (`refit=True` or `use_cache=False` to force a fit); new results invalidate them.

    Any other `model` name is a registry model (an Elo baseline), one row per game."""
    day = _as_date(day)
    created_at = created_at or default_created_at(day)
    if not _enough_history(conn):
        log.warning("predict %s: team_games is (nearly) empty; no predictions", day)
        return []
    full = build_prediction_frame(conn, [day])
    if _is_production(model):
        rows = _production_day(conn, full, day, created_at, refit=refit, use_cache=use_cache)
        if store:
            store_predictions(conn, _flat(rows))
        return rows
    name = model or PRODUCTION_MODEL
    rows = _predict_one_day(conn, full, day, name, created_at, model_factory(name, conn))
    if store:
        store_predictions(conn, rows)
    return rows


def predict_days(conn: sqlite3.Connection, start, end, *, model: str | None = None,
                 log_fn: Callable[[str], None] | None = None, refit_days: int = 1) -> dict:
    """Walk-forward predictions for every game day in [start, end] (e.g. a demo backfill).

    Each day uses models fit on `date < day` only (refit every `refit_days` days; between refits
    the heads are reused, which is still walk-forward because the heads only ever saw earlier
    games while the features are rebuilt as of each day). Market lines are the closing lines in
    `closing_lines` (or an archived snapshot); injuries use whatever snapshots exist, usually
    none for old days, which then get the engine's leak-safe "regulars who played"
    availability. Returns {"days": n, "games": n, "skipped_days": [...], "model": ...}."""
    start, end = _as_date(start), _as_date(end)
    if not _enough_history(conn):
        return {"days": 0, "games": 0, "skipped_days": [], "model": model or PRODUCTION_MODEL}
    full = build_prediction_frame(conn, [start, end])
    played = sorted({d.date() for d in full["game_date"]
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
    if _is_production(model):
        return _production_days(conn, full, played, log_fn, refit_days)
    name = model
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


# ------------------------------------------------------------------ production pair

def _production_day(conn, full: pd.DataFrame, day: date, created_at: str, *,
                    refit: bool, use_cache: bool) -> list[dict]:
    ts = pd.Timestamp(day)
    train = full[full["game_date"] < ts]
    test = full[full["game_date"] == ts]
    if test.empty:
        return []
    if len(train) < MIN_TRAIN_GAMES:
        log.warning("predict %s: only %d training games; skipped", day, len(train))
        return []
    test = _mask_results(test).reset_index(drop=True)
    specs = production.resolve_specs(conn, PRODUCTION_MODELS)
    bundle = production.load_or_fit(conn, day, train, test, PRODUCTION_MODELS,
                                    use_cache=use_cache, refit=refit, specs=specs)
    e = elo_mod.run_elo(pd.concat([train, test]), elo_mod.TUNED).iloc[len(train):]
    return price_day(conn, day, train, test, bundle, created_at, e.reset_index(drop=True))


def _production_days(conn, full: pd.DataFrame, played: list[date], log_fn,
                     refit_days: int) -> dict:
    """Backfill: one engine pass + feature build over the whole frame (every later day is a
    prefix slice of it), then per day: fit (every `refit_days` days) and price."""
    import time
    t0 = time.time()
    specs = production.resolve_specs(conn, PRODUCTION_MODELS, shadows=False)
    reg, _ = production.registry(conn, specs)
    names = tuple(specs)
    fc = reg[names[0]]().fc
    fc.prime(full)
    elo_full = elo_mod.run_elo(full, elo_mod.TUNED)
    if log_fn:
        log_fn(f"  engine + features over {len(full)} games: {time.time() - t0:.0f}s")
    scorers, fit_day, n_games, skipped, n_fits = None, None, 0, [], 0
    for k, d in enumerate(played):
        ts = pd.Timestamp(d)
        upto = full[full["game_date"] <= ts]
        train, test = upto[upto["game_date"] < ts], upto[upto["game_date"] == ts]
        if len(train) < MIN_TRAIN_GAMES or test.empty:
            skipped.append(d.isoformat())
            continue
        t1 = time.time()
        fit_s = 0.0
        if scorers is None or (d - fit_day).days >= refit_days:
            models = {n: reg[n]() for n in names}
            for m in models.values():
                m.fit(train)
            scorers = {n: production.Scorer.from_model(m) for n, m in models.items()}
            fit_day, fit_s, n_fits = d, time.time() - t1, n_fits + 1
        F = fc.get(upto, "avail_tight").iloc[len(train):].reset_index(drop=True)
        test = _mask_results(test).reset_index(drop=True)
        key = production.cache_key(conn, d, list(test["game_id"]), names, specs)
        bundle = production.Bundle(
            key=key, day=d.isoformat(), scorers=scorers, F=F, game_ids=list(test["game_id"]),
            state=None, train_games=int(len(train)),
            train_through=train["game_date"].max().strftime("%Y-%m-%d"),
            fitted_at=_iso_now(), fit_seconds=fit_s, cache="backfill",
            timings={"fit_day": fit_day.isoformat(), "refit_days": refit_days},
            versions=production.versions_of(specs))
        rows = price_day(conn, d, train, test, bundle, default_created_at(d),
                         elo_full.iloc[len(train):len(train) + len(test)].reset_index(drop=True))
        n_games += store_predictions(conn, _flat(rows))
        if log_fn and (k % 10 == 0 or k == len(played) - 1):
            log_fn(f"  {d} ({k + 1}/{len(played)} days, {n_games} predictions, "
                   f"{n_fits} fits, {time.time() - t0:.0f}s)")
    return {"days": len(played), "games": n_games // 2, "rows": n_games,
            "skipped_days": skipped, "model": PRODUCTION_WINPROB, "fits": n_fits,
            "refit_days": refit_days, "seconds": round(time.time() - t0, 1)}


def _line_arrays(lines: list[dict | None]) -> tuple[np.ndarray, np.ndarray]:
    mm = np.array([-l["home_spread"] if l and l.get("home_spread") is not None else np.nan
                   for l in lines], float)
    mt = np.array([l["total"] if l and l.get("total") is not None else np.nan
                   for l in lines], float)
    return mm, mt


def _played_keys(conn, day: date) -> set[tuple[str, str]]:
    return {(r[0], r[1]) for r in conn.execute(
        "SELECT DISTINCT game_id, team FROM player_games WHERE game_date=? AND min>0",
        (day.isoformat(),))}


_STATUS_KEYS = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable"}


def feature_vector(G: pd.DataFrame, i: int, row: pd.Series, mk: dict | None,
                   team_inj: dict, pred: tuple[float, float, float]) -> dict:
    """The full priced input vector of one game (stored as `inputs.features`, for drift
    monitoring; not truncated like `explain`). Strengths are engine points after the injury
    override; `absences` = away loss - home loss (> 0 helps the home side, the sign of the
    home margin), `absences_total` = both sides' loss."""
    def g(col):
        return _py(G[col].iat[i]) if col in G.columns else None
    ah, aa = g("abs_loss_home"), g("abs_loss_away")
    ph, pa = g("ps_home"), g("ps_away")
    b2b_h, b2b_a = _py(row.get("home_b2b")), _py(row.get("away_b2b"))
    f = {"strength_diff": g("ps_diff"), "lineup_strength_home": ph, "lineup_strength_away": pa,
         "lineup_strength_diff": _py(ph - pa) if ph is not None and pa is not None else None,
         "abs_loss_home": ah, "abs_loss_away": aa,
         "absences": _py(aa - ah) if ah is not None and aa is not None else None,
         "absences_total": _py(aa + ah) if ah is not None and aa is not None else None,
         "elo_diff": g("elo_diff"), "rest_diff": _py(row.get("rest_diff")),
         "b2b_home": b2b_h, "b2b_away": b2b_a,
         "b2b_total": (int(b2b_h) + int(b2b_a)) if b2b_h is not None and b2b_a is not None
         else None,
         "market_spread": _py(mk["home_spread"]) if mk and mk.get("home_spread") is not None
         else None,
         "market_total": _py(mk["total"]) if mk and mk.get("total") is not None else None,
         "market_home_prob": _py(mk["home_prob"]) if mk and mk.get("home_prob") is not None
         else None,
         "pred_home_prob": _py(pred[0]), "pred_margin": _py(pred[1]), "pred_total": _py(pred[2])}
    for side in ("home", "away"):
        counts = dict.fromkeys(_STATUS_KEYS.values(), 0)
        for x in team_inj.get(side) or []:
            k = _STATUS_KEYS.get(x.get("status"))
            if k:
                counts[k] += 1
        for k, v in counts.items():
            f[f"{k}_{side}"] = v
    return f


def _news_since_line(conn, day: date, test: pd.DataFrame, F: pd.DataFrame, mm, mt,
                     lines: list[dict | None], cover: dict, inj_of: dict, state, lh, la,
                     opinion) -> dict:
    """IMPROVEMENTS G1 (`model.line_news`, ON only): per game, the opinion model's value of the
    injury news published since the market line was captured, and the line shift beta * that.
    Lines without a capture time (closing lines) and games without a usable report now get 0;
    a side counts only if the newest batch lists its team both now and at capture."""
    n = len(test)
    line_at = [line_news.capture_time(l) for l in lines]
    inj_at: list = [None] * n
    lh_c = None if lh is None else lh.copy()
    la_c = None if la is None else la.copy()
    side_ok = [[False, False] for _ in range(n)]
    if cover and lh is not None:
        groups: dict[str, list[int]] = {}
        for i in range(n):
            cap, at = line_at[i], inj_of[i]["at"]
            if cap is not None and _iso_utc(cap) < _iso_utc(at):
                groups.setdefault(_iso_utc(cap), []).append(i)
        for cap, idx in sorted(groups.items()):
            sub = test.iloc[idx]
            cov0 = injuries.availability(conn, cap, day,
                                         set(sub["home_raw"]) | set(sub["away_raw"]))
            if not cov0:
                continue
            h0, a0 = production.lineup_strengths(state, sub, {t: c["avail"]
                                                              for t, c in cov0.items()})
            for j, (k, (_, r)) in enumerate(zip(idx, sub.iterrows())):
                inj_at[k] = cap
                for s, (now, arr, v0) in enumerate((("home_raw", lh_c, h0),
                                                    ("away_raw", la_c, a0))):
                    lst_now = (cover.get(r[now]) or {}).get("source") is not None
                    lst_cap = (cov0.get(r[now]) or {}).get("source") is not None
                    arr[k] = line_news.guarded(arr[k], v0[j], lst_now, lst_cap)
                    side_ok[k][s] = bool(lst_now and lst_cap)
    G_now = production.apply_overrides(F, mm, mt, lh, la)
    G_cap = G_now if lh_c is None else production.apply_overrides(F, mm, mt, lh_c, la_c)
    out = line_news.shift(opinion, G_now, G_cap)
    out["line_at"] = line_at
    out["injuries_at_line"] = inj_at
    out["sides"] = [{"home": a, "away": b} for a, b in side_ok]
    return out


def _nsl_record(nsl: dict, i: int) -> dict:
    """The stored `inputs.news_since_line` of game i (market rows, switch ON only)."""
    return {"line_at": nsl["line_at"][i], "injuries_at_line": nsl["injuries_at_line"][i],
            "sides": nsl["sides"][i],
            **{k: _py(nsl[k][i]) for k in ("delta_margin", "delta_total", "shift_margin",
                                            "shift_total")},
            "beta_margin": nsl["beta_margin"], "beta_total": nsl["beta_total"]}


def price_day(conn, day: date, train: pd.DataFrame, test: pd.DataFrame,
              bundle: "production.Bundle", created_at: str, elo_pre: pd.DataFrame) -> list[dict]:
    """Score the day's games with the fitted pair: live lines and injury lineups applied to the
    bundle's features. `test` rows are in `bundle.game_ids` order or any subset/order of it.
    Returns headline rows with `opinion` attached (see predict_day)."""
    pos = {g: i for i, g in enumerate(bundle.game_ids)}
    F = bundle.F.iloc[[pos[g] for g in test["game_id"]]].reset_index(drop=True)
    test = test.reset_index(drop=True)
    tips = {r["game_id"]: (r["tipoff_utc"], r["status"]) for r in conn.execute(
        "SELECT game_id, tipoff_utc, status FROM games WHERE nba_day=?", (day.isoformat(),))}
    lines = [market_line(conn, g, tips.get(g, (None, None))[0]) for g in test["game_id"]]
    mm, mt = _line_arrays(lines)

    # injuries -> expected lineups (only when a usable report exists). Live days: one cutoff
    # (now) for the slate; past days: each game's own pre-tip cutoff (B3, `_injury_groups`).
    cover: dict[str, dict] = {}
    inj_of: dict[int, dict] = {}
    groups = _injury_groups(conn, day, created_at, test, tips)
    for at, idx in groups:
        sub = test if idx is None else test.iloc[idx]
        cover.update(injuries.availability(conn, at, day,
                                           set(sub["home_raw"]) | set(sub["away_raw"])))
        g_inj = injuries.current(conn, at, set(sub["home"]) | set(sub["away"]), game_day=day)
        for k in (range(len(test)) if idx is None else idx):
            inj_of[k] = {**g_inj, "at": at}
    past = groups[0][1] is not None
    lh = la = None
    state = None
    if cover:
        state = production.ensure_state(conn, bundle, train)
        if absence_scale.ON:                   # G4, COURTSIDE_ABSENCE_SCALE (default OFF)
            state = absence_scale.wrap(state, F, test)
        lh, la = production.lineup_strengths(state, test, {t: c["avail"]
                                                           for t, c in cover.items()})
    G = production.apply_overrides(F, mm, mt, lh, la)
    post = (test["season_type"].to_numpy() != "Regular Season").astype(float)
    scorers = bundle.scorers
    G_of = {n: G for n in scorers}             # the features each scorer prices
    nsl = None
    if line_news.ON:                           # G1, COURTSIDE_NEWS_SINCE_LINE (default OFF)
        nsl = _news_since_line(conn, day, test, F, mm, mt, lines, cover, inj_of, state,
                               lh, la, scorers[PRODUCTION_OPINION])
        G_m = line_news.apply_shift(G, nsl)
        G_of = {n: (G_m if sc.market else G) for n, sc in scorers.items()}
    for n, sc in scorers.items():
        sc.set_features(G_of[n], post)
    effects = (production.injury_effects(scorers[PRODUCTION_OPINION], G, state, test, cover)
               if cover else [])
    played = _played_keys(conn, day) if day < config.today_nba_day() else set()
    contrib = {n: sc.contributions(test) for n, sc in scorers.items()}
    pred = {n: (sc.predict_proba(test), sc.predict_margin(test), sc.predict_total(test))
            for n, sc in scorers.items()}
    absent = {n: production.absence_effects(sc, G_of[n]) for n, sc in scorers.items()}

    fit_info = {"fitted_at": bundle.fitted_at, "cache": bundle.cache,
                "fit_seconds": round(bundle.fit_seconds, 1), "timings": bundle.timings}
    versions = getattr(bundle, "versions", None) or {}
    out = []
    for i, r in test.iterrows():
        gid = r["game_id"]
        tip, status = tips.get(gid, (None, None))
        mk = lines[i]
        inj = inj_of[i]
        team_inj = {"home": inj["by_team"].get(r["home"], []),
                    "away": inj["by_team"].get(r["away"], [])}
        avail, absentees = {}, {}
        for side in ("home", "away"):
            raw = r[f"{side}_raw"]
            c = cover.get(raw)
            if c:
                basis = "injury_report"
            elif (gid, raw) in played:
                basis = "regulars_played"
            else:
                basis = "full_roster"
            listed = [p for p in (c["players"] if c else []) if p["p_play"] < 1.0]
            eff = {e["player_id"]: e["effect_margin"] for e in effects
                   if e["game_id"] == gid and e["team"] == r[side]}
            listed.sort(key=lambda p: (eff.get(p["player_id"], 0.0), p["p_play"]))
            absentees[r[side]] = [p for p in listed if p["player_id"] is not None]
            avail[side] = {
                "basis": basis, "source": c["source"] if c else None,
                "captured_at": c["captured_at"] if c else None,
                "lineup_strength": _py((lh if side == "home" else la)[i])
                if lh is not None else None,
                "players": [{k: _py(p[k]) for k in ("player", "status", "p_play", "player_id",
                                                    "how")} for p in listed][:20]}
        base_inputs = {
            "train_games": int(bundle.train_games), "train_through": bundle.train_through,
            "schedule": _factors(r),
            "elo": {"home": _py(elo_pre["elo_home_pre"].iat[i]),
                    "away": _py(elo_pre["elo_away_pre"].iat[i])},
            "market": ({k: _py(mk[k]) for k in ("source", "provider", "captured_at", "kind",
                                                "home_spread", "total", "home_ml", "away_ml",
                                                "home_prob")} if mk else None),
            "injuries": {s: [{k: x[k] for k in ("player", "status", "reason", "source")}
                             for x in team_inj[s]] for s in team_inj},
            "injuries_batches": inj.get("batches", {}),
            "availability": avail, "fit": fit_info,
            "model_effects": [e for e in effects if e["game_id"] == gid],
        }
        if past:                                   # B3: the pre-tip cutoff this game used
            base_inputs["injuries_at"] = inj["at"]
        if injuries.INJ_GAMEDAY_GAME:              # lineage of the opt-in game-day rows
            base_inputs["injuries_gameday"] = True
        rows = {}
        for name, sc in scorers.items():
            expl = gbm.reasons_from_contrib(contrib[name].iloc[i], r["home"], r["away"], top=12)
            factors = production.key_factors(
                expl, r["home"], r["away"], absentees,
                {r["home"]: float(absent[name][0][i]), r["away"]: float(absent[name][1][i])})
            ver = versions.get(name, {})
            vid = ver.get("version_id")
            p, mg, tot = pred[name]
            if nsl is not None and sc.market:
                vid = line_news.tagged(vid)
            if absence_scale.ON:
                vid = absence_scale.tagged(vid)
            inputs = {"model": name, "model_version": vid,
                      **base_inputs, "factors": factors,
                      "explain": [{k: _py(v) for k, v in e.items()} for e in expl[:8]],
                      "features": feature_vector(G_of[name], i, r, mk, team_inj,
                                                 (p[i], mg[i], tot[i]))}
            if nsl is not None and sc.market:
                inputs["news_since_line"] = _nsl_record(nsl, i)
            if absence_scale.ON:
                inputs["absence_scale"] = absence_scale.S
            rows[name] = {
                "game_id": gid, "nba_day": day.isoformat(), "model": name,
                "model_version": vid, "role": ver.get("role", "champion"),
                "created_at": created_at, "home": r["home"], "away": r["away"],
                "season": r["season"], "season_type": r["season_type"],
                "tipoff_utc": tip, "status": status,
                "home_win_prob": _py(p[i]), "home_margin": _py(mg[i]), "total": _py(tot[i]),
                "market_home_spread": _py(mk["home_spread"]) if mk else None,
                "market_total": _py(mk["total"]) if mk else None,
                "market_home_prob": _py(mk["home_prob"]) if mk else None,
                "inputs": inputs, "injuries": inputs["injuries"],
            }
        head = rows[PRODUCTION_WINPROB]
        head["opinion"] = rows[PRODUCTION_OPINION]
        shadows = [rows[k] for k in rows if rows[k]["role"] == "shadow"]
        if shadows:
            head["shadows"] = shadows
        out.append(head)
    return out


# ------------------------------------------------------------------ reading back

def latest_pretip(conn: sqlite3.Connection, game_ids: list[str] | None = None, *,
                  day: str | None = None, model: str | None = None,
                  include_shadow: bool = False) -> dict[str, dict]:
    """game_id -> stored prediction row (dict, `inputs` parsed), via the
    `latest_pretip_predictions` view. Filter by game ids or by nba_day. Only reader-visible
    rows (champion / legacy NULL role) unless `include_shadow`."""
    model = model or PRODUCTION_MODEL
    view = "latest_pretip_predictions"
    if include_shadow and conn.execute("SELECT 1 FROM sqlite_master WHERE type='view' AND "
                                       "name='latest_pretip_predictions_all'").fetchone():
        view = "latest_pretip_predictions_all"
    sql, args = f"SELECT * FROM {view} WHERE model=?", [model]
    if not include_shadow:
        sql += shown_roles_sql(conn)
    if day is not None:
        sql += " AND nba_day=?"
        args.append(day)
    if game_ids is not None:
        if not game_ids:
            return {}
        sql += f" AND game_id IN ({','.join('?' * len(game_ids))})"
        args += list(game_ids)
    out = {}
    for r in conn.execute(sql + " ORDER BY created_at", args):
        d = dict(r)
        d["inputs"] = json.loads(d.pop("inputs_json")) if r["inputs_json"] else {}
        out[d["game_id"]] = d
    return out
