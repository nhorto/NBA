"""Slate projection pipeline (docs/PLAYERS.md §4, §5.3, §6, §7; T1.4).

Inputs per game, all as known at the run time (`at`) or strictly before the game date:

* **Environment** (`environment`): live = median over books of each book's latest
  `line_snapshots` row at or before `at` (spread, total, team totals; the 'open' book is
  skipped), else the latest `gbm_market` row of `predictions` at or before `at`, else the
  closing line, else a league default (data check). Backtests use the closing line.
* **Availability:** live = `availability.p_play_slate` at `at` (DK starter flags = P 0.99).
  Report statuses (out ... available, removed, day-to-day) give P(available) for the scenario
  draw; players the report does not list get 1.0 because their DNP-CD probability is already
  the minutes engine's P(plays | available). Backtest protocol "avail" (§7.1 L4) = the oracle:
  regulars who did not play are out, everybody else available; protocol "reports" = the
  official reports at the first tip - 45 min (2025-12-22+).
* **Minutes** (`minutes.pregame_panel`, scenario re-runs in `copula.ScenarioMinutes`), **rates**
  (`state.rate_frame`: EB per-minute rates before the day), **usage bump** (`state.bump_for`)
  for the team's top-2 regular stars (by DK/min x mpg; appeared in >= 7 of the last 10 team
  games or in one of the last 3).
* **Parameters** (`Bundle`): minutes params, rate-state params and copula params, all fit on
  seasons < S (S = the season of the day), cached as `<data_dir>/proj/bundle_<S>.json`.

Seeds (common random numbers): `stable_seed(slate_or_day, kind, MODEL_VERSION)`; every game
draws from SeedSequence([seed, crc32(game_id)]), so a game's sims do not depend on which other
games are on the slate.

Public API
  project_slate(conn, slate_id, at, *, run_id, n_sims=25_000, seed=None, kind=None, ctx=None,
                bundle=None, store=True) -> ProjectionSet {DK playerId (str): Projection}
  project_day(conn, nba_day, at, *, n_sims=25_000, seed=None, ...) -> DayResult (NBA ids)
  project_historical(conn, game_date, *, protocol="avail" | "reports", ...) -> DayResult
  load_bundle(conn, season, ctx) / Context(conn, season)
  backtest(conn, season, n_dates=60, ...) -> dict  (quick walk-forward evaluation)
"""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
import time
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from . import availability as AV
from . import copula as C
from . import minutes as M
from . import reconcile as RC
from . import state as ST
from . import store as SO
from .dfs.optimizer import Projection

MODEL_VERSION = C.MODEL_VERSION
HISTORY_SEASONS = 7            # box seasons loaded for the minutes engine (window 5 + warm-up)
MINUTES_WINDOW = 5
LISTED = set(AV.LISTED) | {"removed", "daytoday"}
DEFAULT_TOTAL = 225.0
DEFAULT_HOME_EDGE = 2.0
STATS_OUT = ("min", "dk")


# ============================================================================== small helpers

def season_of(day: date) -> str:
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _slate_day(pool) -> date:
    """The NBA (Eastern) day of a slate without a ``dfs_slates`` row: its earliest tip in ET.
    A 9:30 PM ET tip is the next day in UTC, so the UTC date of any one player is wrong."""
    # DK start times look like "2026-10-20T19:00:00.0000000Z" (UTC, 7-digit fraction).
    starts = [datetime.fromisoformat(p.start_time[:19]).replace(tzinfo=timezone.utc)
              for p in pool.players if p.start_time]
    return min(starts).astimezone(config.NBA_TZ).date()


def _aware(at) -> datetime:
    if isinstance(at, str):
        at = datetime.fromisoformat(at.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("`at` must be timezone-aware")
    return at.astimezone(timezone.utc)


def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_seed(*parts) -> int:
    """Process-independent 63-bit seed from the parts (slate/day, run kind, model version)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def game_seed(seed: int, game_id: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed) & (2 ** 63 - 1), zlib.crc32(str(game_id).encode())])


def cache_dir() -> Path:
    return config.data_dir() / "proj"


# ============================================================================== context, bundle

class Context:
    """Data loaded once per process for one season: the minutes History (box seasons S-7..),
    the rate-state game frame and positions, and pair shares (events of the training seasons
    plus season S; every lookup is strictly before the projected date)."""

    def __init__(self, conn: sqlite3.Connection, season: str, *, history: M.History | None = None,
                 log: Callable[[str], None] | None = None):
        self.season = season
        self.log = log or (lambda m: None)
        s0 = int(season[:4])
        since = f"{s0 - HISTORY_SEASONS}-{(s0 - HISTORY_SEASONS + 1) % 100:02d}"
        t0 = time.time()
        self.h = history if history is not None else M.load_history(conn, since=since)
        st_first = f"{s0 - 1 - 6}-{(s0 - 6) % 100:02d}"
        self.games = ST.load_games(conn, since_season=st_first)
        self.pos = ST.load_positions(conn)
        self._shares: M.PairShares | None = None
        self.train_seasons = sorted(x for x in self.h.tg["season"].unique()
                                    if x < season)[-MINUTES_WINDOW:]
        self.seconds_load = time.time() - t0

    def train_panel(self) -> pd.DataFrame:
        return M.oracle_panel(self.h, M.eval_targets(self.h, self.train_seasons))

    @property
    def shares(self) -> M.PairShares:
        if self._shares is None:
            t0 = time.time()
            ev = [M.pair_events(self.train_panel())]
            cur = M.eval_targets(self.h, [self.season])
            if len(cur):
                ev.append(M.pair_events(M.oracle_panel(self.h, cur)))
            self._shares = M.PairShares(pd.concat(ev, ignore_index=True))
            self.log(f"pair shares built in {time.time() - t0:.1f}s")
        return self._shares


@dataclass
class Bundle:
    season: str
    minutes: M.MinutesParams
    state: dict
    copula: C.CopulaParams

    def to_json(self) -> str:
        return json.dumps({"season": self.season, "minutes": json.loads(self.minutes.to_json()),
                           "state": self.state, "copula": json.loads(self.copula.to_json()),
                           "model_version": MODEL_VERSION})

    @classmethod
    def from_json(cls, s: str) -> "Bundle":
        d = json.loads(s)
        return cls(season=d["season"], minutes=M.MinutesParams(**d["minutes"]),
                   state=d["state"], copula=C.CopulaParams(**d["copula"]))


def fit_state_params(conn: sqlite3.Connection, season: str, log=None) -> dict:
    """Rate-state params for seasons < S: DEFAULT_PARAMS when it was fit exactly on those, a
    cached `state_params_<S>.json`, else fit_params on 8 seasons + the usage bump on 3."""
    path = cache_dir() / f"state_params_{season}.json"
    if path.exists():
        return json.loads(path.read_text())
    if ST.DEFAULT_PARAMS.get("fit_seasons", [None, None])[-1] == _prev_season(season):
        return ST.DEFAULT_PARAMS
    s0 = int(season[:4])
    seasons = [f"{y}-{(y + 1) % 100:02d}" for y in range(s0 - 8, s0)]
    p = ST.fit_params(conn, seasons, log=log)
    p["usage_bump"] = ST.fit_usage_bump(conn, p, seasons[-3:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ST.params_to_json(p))
    return p


def load_bundle(conn: sqlite3.Connection, season: str, ctx: Context | None = None, *,
                refit: bool = False, log=None) -> Bundle:
    """The fitted Bundle for season S (everything fit on seasons < S), cached on disk."""
    say = log or (lambda m: None)
    path = cache_dir() / f"bundle_{season}.json"
    if path.exists() and not refit:
        return Bundle.from_json(path.read_text())
    t0 = time.time()
    sp = fit_state_params(conn, season, log=say)
    mpath = cache_dir() / f"minutes_params_{season}.json"
    if mpath.exists() and not refit:
        mp = M.MinutesParams.from_json(mpath.read_text())
    else:
        ctx = ctx or Context(conn, season, log=say)
        panel = ctx.train_panel()
        mp = M.fit(ctx.h, season, window=MINUTES_WINDOW, log=say, panel=panel,
                   shares=M.PairShares(M.pair_events(panel)))
    cp = C.fit_copula(conn, season, sp, log=say)
    b = Bundle(season=season, minutes=mp, state=sp, copula=cp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.to_json())
    say(f"bundle {season} fitted in {time.time() - t0:.0f}s -> {path}")
    return b


# ============================================================================== environment

def environment(conn: sqlite3.Connection, game_id: str, at: datetime | None = None, *,
                mode: str = "live") -> dict:
    """Expected home margin / total and implied team points for a game.

    mode 'live': books at `at` -> gbm_market at `at` -> closing line -> default.
    mode 'closing': closing line -> default (backtests)."""
    if mode == "live" and at is not None:
        ts = _iso(at)
        rows = conn.execute(
            "SELECT book, home_spread, total, home_team_total, away_team_total, captured_at "
            "FROM line_snapshots WHERE game_id=? AND captured_at<=? AND book!='open' "
            "ORDER BY captured_at", (game_id, ts)).fetchall()
        if rows:
            latest = {}
            for r in rows:
                latest[r[0]] = r
            v = list(latest.values())

            def med(i):
                x = [r[i] for r in v if r[i] is not None]
                return float(np.median(x)) if x else None
            hs, tot, htt, att = med(1), med(2), med(3), med(4)
            if hs is not None and tot is not None:
                h, a = RC.implied_points(-hs, tot, htt, att)
                return {"mu_margin": -hs, "mu_total": tot, "home_pts": h, "away_pts": a,
                        "source": f"books:{len(v)}",
                        "captured_at": max(r[5] for r in v)}
        r = conn.execute(
            "SELECT home_margin, total, created_at FROM predictions WHERE game_id=? AND "
            "model='gbm_market' AND created_at<=? AND (role IS NULL OR role='champion') "
            "ORDER BY created_at DESC LIMIT 1", (game_id, ts)).fetchone()
        if r is not None and r[0] is not None and r[1] is not None:
            h, a = RC.implied_points(r[0], r[1])
            return {"mu_margin": float(r[0]), "mu_total": float(r[1]), "home_pts": h,
                    "away_pts": a, "source": "gbm_market", "captured_at": r[2]}
    r = conn.execute("SELECT AVG(home_spread_close), AVG(total_close) FROM closing_lines "
                     "WHERE game_id=? AND home_spread_close IS NOT NULL AND total_close IS NOT "
                     "NULL", (game_id,)).fetchone()
    if r is not None and r[0] is not None:
        h, a = RC.implied_points(-r[0], r[1])
        return {"mu_margin": -float(r[0]), "mu_total": float(r[1]), "home_pts": h,
                "away_pts": a, "source": "closing"}
    h, a = RC.implied_points(DEFAULT_HOME_EDGE, DEFAULT_TOTAL)
    return {"mu_margin": DEFAULT_HOME_EDGE, "mu_total": DEFAULT_TOTAL, "home_pts": h,
            "away_pts": a, "source": "default"}


# ============================================================================== inputs

def _day_int(day: date) -> int:
    return int(np.datetime64(day, "D").astype(np.int64))


def _targets(games: pd.DataFrame, day: date) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        for team, opp in ((g.home, g.away), (g.away, g.home)):
            rows.append({"key": f"{g.game_id}|{team}", "game_id": g.game_id, "team": team,
                         "opp": opp, "date": _day_int(day), "season": season_of(day)})
    return pd.DataFrame(rows)


def _stars(panel: pd.DataFrame, rates: pd.DataFrame) -> np.ndarray:
    """Top-2 regular stars of each team-game (key) by DK/min x mpg (fit_usage_bump's rule)."""
    score = (rates["r_dk"] * rates["mpg"]).to_numpy(float)
    regular = ((panel["app10"].to_numpy(float) >= 7)
               | (panel["missed_now"].to_numpy(float) < 3)) & panel["last_date"].notna().to_numpy()
    out = np.zeros(len(panel), bool)
    for _, idx in panel.groupby("key", sort=False).indices.items():
        cand = idx[regular[idx]]
        top = cand[np.argsort(-score[cand], kind="stable")[:2]]
        out[top] = True
    return out


def build_games(ctx: Context, b: Bundle, conn: sqlite3.Connection, day: date,
                games: pd.DataFrame, panel: pd.DataFrame, p_avail: dict, env: dict,
                starters: set | None = None) -> list[C.GameSpec]:
    """GameSpecs from a candidate panel (pregame_panel rows keyed game_id|team, optional
    `absent` column) and P(available) {(key, player_id): p} (default 1.0; absent -> 0)."""
    panel = panel.reset_index(drop=True)
    q = pd.DataFrame({"player_id": panel["player_id"].astype(np.int64),
                      "game_date": day.isoformat()})
    rates = ST.rate_frame(conn, q, b.state, games=ctx.games, pos=ctx.pos)
    panel["_star"] = _stars(panel, rates)
    shares = ctx.shares
    out = []
    by_key = panel.groupby("key", sort=False).indices
    for g in games.itertuples(index=False):
        teams = {}
        e = env[g.game_id]
        for side, team in (("home", g.home), ("away", g.away)):
            key = f"{g.game_id}|{team}"
            idx = by_key.get(key, np.zeros(0, int))
            p = panel.iloc[idx].reset_index(drop=True)
            r = rates.iloc[idx].reset_index(drop=True)
            if "absent" in p:
                pa = np.where(p["absent"].to_numpy(bool), 0.0, 1.0)
            else:
                pa = np.ones(len(p))
            pa = np.array([p_avail.get((key, int(pid)), v)
                           for pid, v in zip(p["player_id"], pa)], float)
            pl = pd.DataFrame({"player_id": p["player_id"].astype(np.int64), "p_avail": pa})
            for s in C.DK_STATS:
                pl[f"r_{s}"] = r[f"r_{s}"].to_numpy(float)
            pl["r_dk"] = r["r_dk"].to_numpy(float)
            pl["mpg"] = r["mpg"].to_numpy(float)
            pl["tier_state"] = r["tier"].to_numpy(int)
            pl["star"] = p["_star"].to_numpy(bool)
            pl["rotation"] = pl["mpg"].to_numpy() >= 15.0
            sm = C.ScenarioMinutes(p.drop(columns=["_star"]), b.minutes, shares,
                                   starters=_team_starters(starters, p))
            teams[side] = C.TeamSpec(team=team, players=pl, minutes=sm,
                                     implied=e["home_pts"] if side == "home" else e["away_pts"])
        out.append(C.GameSpec(game_id=g.game_id, home=teams["home"], away=teams["away"],
                              mu_margin=e["mu_margin"], mu_total=e["mu_total"],
                              env_source=e.get("source", "")))
    return out


def _team_starters(starters: set | None, p: pd.DataFrame):
    """Confirmed starters of a team (only when exactly five are flagged for it)."""
    if not starters:
        return None
    s = [int(x) for x in p["player_id"] if int(x) in starters]
    return s if len(s) == 5 else None


# ============================================================================== results

@dataclass
class DayResult:
    day: date
    sims: list                      # GameSims per game
    players: pd.DataFrame           # one row per player (summary)
    stats: pd.DataFrame             # long: key (players index), stat, cond, mean, sd, q..
    teams: list                     # reconciliation info per team-game
    meta: dict = field(default_factory=dict)

    def projections(self, key: str = "player_id") -> dict:
        out = {}
        for r in self.players.itertuples(index=False):
            k = getattr(r, key)
            if k is None or (isinstance(k, float) and not np.isfinite(k)):
                continue
            out[k] = Projection(mean=round(float(r.dk_mean), 3), p25=round(float(r.dk_q25), 3),
                                p10=round(float(r.dk_q10), 3), p50=round(float(r.dk_q50), 3),
                                p90=round(float(r.dk_q90), 3))
        return out

    def fp_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(player_id (n,), fp (S, n) float32) over all games of the day."""
        if not self.sims:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        return (np.concatenate([g.player_id for g in self.sims]),
                np.concatenate([g.fp for g in self.sims], axis=1))


class ProjectionSet(dict):
    """{DK playerId (str): Projection} plus `.meta` (run_id, sim_path, seconds, checks) and
    `.result` (the DayResult, NBA ids)."""
    meta: dict
    result: DayResult | None


def summarize_games(sims: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    prow, srows = [], []
    for g in sims:
        played = g.minutes > 0
        sd = C.summarize(g.fp, played)
        sm = C.summarize(g.minutes, played)
        info = g.info.reset_index(drop=True)
        for j in range(len(info)):
            r = info.iloc[j].to_dict()
            r["game_id"] = g.game_id
            a = sd[(sd["col"] == j) & (sd["cond"] == "all")].iloc[0]
            pd_ = sd[(sd["col"] == j) & (sd["cond"] == "played")].iloc[0]
            ma = sm[(sm["col"] == j) & (sm["cond"] == "all")].iloc[0]
            mpl = sm[(sm["col"] == j) & (sm["cond"] == "played")].iloc[0]
            for c in ("mean", "sd", "q05", "q10", "q25", "q50", "q75", "q90", "q95"):
                r[f"dk_{c}"] = a[c]
                r[f"dkp_{c}"] = pd_[c]
            r["min_mean"] = ma["mean"]
            r["min_played"] = mpl["mean"]
            key = len(prow)
            prow.append(r)
            for stat, frame in (("dk", sd), ("min", sm)):
                for cond in ("played", "all"):
                    row = frame[(frame["col"] == j) & (frame["cond"] == cond)].iloc[0].to_dict()
                    row.pop("col")
                    row.update({"key": key, "stat": stat})
                    srows.append(row)
            for s in C.PROP_STATS:
                srows.append({"key": key, "stat": s, "cond": "all", "mean": r[f"{s}_all"]})
                srows.append({"key": key, "stat": s, "cond": "played",
                              "mean": r[f"{s}_played"] if r["p_play"] > 0 else None})
    return pd.DataFrame(prow), pd.DataFrame(srows)


def run_games(specs: list, b: Bundle, n_sims: int, seed: int, day: date,
              log: Callable[[str], None] | None = None) -> DayResult:
    t0 = time.time()
    sims = [C.simulate_game(gs, b.copula, b.minutes, b.state, n_sims, game_seed(seed, gs.game_id))
            for gs in specs]
    t_sim = time.time() - t0
    players, stats = summarize_games(sims)
    teams = [dict(v, game_id=g.game_id) for g in sims for v in g.teams.values()]
    meta = {"seconds_sim": round(t_sim, 2), "seconds_summary": round(time.time() - t0 - t_sim, 2),
            "n_sims": n_sims, "seed": seed, "games": len(specs),
            "recon": RC.acceptance(teams),
            "env": {g.game_id: {"mu_margin": g.mu_margin, "mu_total": g.mu_total,
                                "source": g.env_source} for g in specs}}
    return DayResult(day=day, sims=sims, players=players, stats=stats, teams=teams, meta=meta)


# ============================================================================== live

def report_avail(conn: sqlite3.Connection, day: date, at: datetime, games: pd.DataFrame,
                 starters: set | None = None) -> tuple[dict, dict, dict]:
    """From the official reports at `at`: P(available) {player_id: p} for report statuses
    (unlisted players are left to the minutes engine), report players to force into the
    candidates {(game_id, team): [player_id]} (everybody not 'out') and a summary."""
    av = AV.p_play_slate(conn, day, at, starters=starters)
    av = av[av["game_id"].isin(set(games["game_id"]))]
    p_avail, extras = {}, {}
    for r in av.itertuples(index=False):
        if r.status in LISTED:
            p_avail[int(r.player_id)] = float(r.p_play)
        if r.status != "out":
            extras.setdefault((r.game_id, r.team), []).append(int(r.player_id))
    n_listed = int(av["status"].isin(list(LISTED)).sum()) if len(av) else 0
    return p_avail, extras, {"n_report_rows": int(len(av)), "n_listed": n_listed}


def candidate_panel(ctx: Context, day: date, games: pd.DataFrame,
                    extras: dict | None = None) -> pd.DataFrame:
    """Pregame minutes candidates of each team-game (players of the team's last 10 games whose
    last team it is) + `extras` {(game_id, team): [player_id]} (report / DK pool players with
    box history; a player forced onto a team is removed from any other team's candidates).
    Players without any box history (rookies before their debut) are not projected."""
    tg = _targets(games, day)
    ex = [{"key": f"{gid}|{team}", "player_id": int(p)}
          for (gid, team), pids in (extras or {}).items() for p in pids]
    exdf = pd.DataFrame(ex, columns=["key", "player_id"]).drop_duplicates("player_id")
    exdf = exdf[exdf["key"].isin(set(tg["key"]))]
    panel = M.pregame_panel(ctx.h, tg, extra=exdf if len(exdf) else None)
    panel = panel[(panel["roster"] == 1) | panel["last_date"].notna()]
    if len(exdf):
        fteam = dict(zip(exdf["player_id"], exdf["key"].str.split("|").str[1]))
        keep = [fteam.get(int(p), t) == t for p, t in zip(panel["player_id"], panel["team"])]
        panel = panel[np.asarray(keep, bool)]
    panel = panel.merge(tg[["key", "game_id"]], on="key", how="left")
    return panel.reset_index(drop=True)


def project_games(conn: sqlite3.Connection, day, games: pd.DataFrame, *, ctx: Context,
                  bundle: Bundle, n_sims: int, seed: int, p_avail: dict | None = None,
                  at: datetime | None = None, env_mode: str = "live",
                  extras: dict | None = None, starters: set | None = None,
                  log: Callable[[str], None] | None = None) -> DayResult:
    """Core of the live and report paths: candidates (+ extras), P(available) {player_id: p}
    (default 1; 0 = out), environment per `env_mode` at `at`, simulate, summarise."""
    day = _as_date(day)
    panel = candidate_panel(ctx, day, games, extras)
    pa = {(k, int(pid)): float(p_avail[int(pid)])
          for k, pid in zip(panel["key"], panel["player_id"]) if int(pid) in (p_avail or {})}
    env = {g: environment(conn, g, at, mode=env_mode) for g in games["game_id"]}
    specs = build_games(ctx, bundle, conn, day, games, panel, pa, env, starters)
    return run_games(specs, bundle, n_sims, seed, day, log)


def _day_games(conn: sqlite3.Connection, day: date, game_ids=None) -> pd.DataFrame:
    g = AV.day_games(conn, day)
    if game_ids is not None:
        g = g[g["game_id"].isin(set(game_ids))]
    return g.reset_index(drop=True)


def project_day(conn: sqlite3.Connection, nba_day, at, *, n_sims: int = 25_000,
                seed: int | None = None, kind: str = "manual", game_ids=None,
                ctx: Context | None = None, bundle: Bundle | None = None,
                extras: dict | None = None, starters: set | None = None,
                log: Callable[[str], None] | None = None) -> DayResult:
    """All games of an NBA day (or `game_ids`) as known at `at`, keyed by NBA player_id."""
    t0 = time.time()
    day = _as_date(nba_day)
    at = _aware(at)
    season = season_of(day)
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    games = _day_games(conn, day, game_ids)
    if seed is None:
        seed = stable_seed(day.isoformat(), kind, MODEL_VERSION)
    p_avail, rep_extras, avinfo = report_avail(conn, day, at, games, starters)
    ex = {k: list(v) for k, v in rep_extras.items()}
    for k, v in (extras or {}).items():
        ex.setdefault(k, []).extend(v)
    res = project_games(conn, day, games, ctx=ctx, bundle=b, n_sims=n_sims, seed=seed,
                        p_avail=p_avail, at=at, env_mode="live", extras=ex, starters=starters,
                        log=log)
    res.meta.update({"seconds_total": round(time.time() - t0, 2), "at": _iso(at),
                     "kind": kind, **avinfo})
    return res


def project_slate(conn: sqlite3.Connection, slate_id, at, *, run_id: str,
                  n_sims: int = 25_000, seed: int | None = None, kind: str | None = None,
                  ctx: Context | None = None, bundle: Bundle | None = None,
                  store: bool = True, log: Callable[[str], None] | None = None
                  ) -> ProjectionSet:
    """Projections for a DK slate as known at `at`: {DK playerId (str, = PoolPlayer
    .dk_player_id): Projection(mean, p25, p10, p50, p90)} with cond all = the P(play)-weighted
    mixture with 0. Writes player_projections / player_projection_stats, the sims npz and
    dfs_runs(run_id).sim_path when `store`. Seed default: stable_seed(slate, kind, version)."""
    from . import ids as IDS
    from .dfs import rules as R
    t0 = time.time()
    at = _aware(at)
    sid = str(slate_id)
    row = conn.execute("SELECT nba_day, game_ids_json FROM dfs_slates WHERE slate_id=?",
                       (sid,)).fetchone()
    pool = R.build_pool(conn, sid)
    if kind is None:
        k = conn.execute("SELECT kind FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
        kind = k[0] if k else "manual"
    game_ids = json.loads(row[1]) if row and row[1] else sorted(
        {p.game_id for p in pool.players if p.game_id})
    day = _as_date(row[0]) if row else _slate_day(pool)
    dk2nba = IDS.nba_ids(conn, "dk", [p.dk_player_id for p in pool.players])
    extras: dict = {}
    starters: set = set()
    for p in pool.players:
        nid = dk2nba.get(str(p.dk_player_id))
        if nid is None or not p.game_id or not p.team:
            continue
        extras.setdefault((p.game_id, p.team), []).append(nid)
        if p.starter:
            starters.add(nid)
    if seed is None:
        seed = stable_seed(sid, kind, MODEL_VERSION)
    res = project_day(conn, day, at, n_sims=n_sims, seed=seed, kind=kind, game_ids=game_ids,
                      ctx=ctx, bundle=bundle, extras=extras, starters=starters or None, log=log)
    nba2dk = {}
    for dk, nid in dk2nba.items():
        nba2dk.setdefault(nid, dk)
    res.players["dk_id"] = [nba2dk.get(int(p)) for p in res.players["player_id"]]
    res.players["slate_id"] = sid
    out = ProjectionSet(res.projections(key="dk_id"))
    out.result = res
    meta = {"run_id": run_id, "slate_id": sid, "kind": kind, "seed": seed,
            "model_version": MODEL_VERSION, "n_players": len(out),
            "dk_pool": len(pool.players),
            "dk_unprojected": sorted(set(p.dk_player_id for p in pool.players) - set(out))}
    if store:
        meta.update(save_run(conn, res, run_id=run_id, slate_id=sid, kind=kind,
                             seconds=time.time() - t0))
    meta["seconds"] = round(time.time() - t0, 2)
    meta.update(res.meta)
    out.meta = meta
    return out


def save_run(conn: sqlite3.Connection, res: DayResult, *, run_id: str, slate_id: str | None,
             kind: str, seconds: float | None = None) -> dict:
    """Store a DayResult: projections + stats, the sims npz, dfs_runs, data checks."""
    pl = res.players.copy()
    proj = pd.DataFrame({
        "game_id": pl["game_id"], "nba_day": res.day.isoformat(), "slate_id": slate_id,
        "player_id": pl["player_id"].astype(np.int64), "team": pl["team"], "run_id": run_id,
        "model_version": MODEL_VERSION, "role": pl["tier"], "p_play": pl["p_play"]}, index=pl.index)
    proj["minutes_json"] = [{"mean_all": r.min_mean, "mean_played": r.min_played}
                            for r in pl.itertuples(index=False)]
    proj["inputs_json"] = [
        {"p_avail": r.p_avail, "normal": bool(r.normal), "mu_played": r.mu_played,
         "dk_id": getattr(r, "dk_id", None), "env": res.meta.get("env", {}).get(r.game_id)}
        for r in pl.itertuples(index=False)]
    SO.write_projections(conn, proj, res.stats)
    pid, fp = res.fp_matrix()
    dk = list(pl["dk_id"]) if "dk_id" in pl else None
    gids = np.concatenate([np.full(len(g.player_id), g.game_id, dtype=object) for g in res.sims]) \
        if res.sims else []
    path = SO.save_sims(slate_id or res.day.isoformat(), run_id, pid, fp, dk_id=dk, game_id=gids)
    SO.record_run(conn, run_id, slate_id=slate_id, kind=kind, model_version=MODEL_VERSION,
                  sim_path=str(path), seconds=seconds,
                  detail={"n_sims": res.meta.get("n_sims"), "seed": res.meta.get("seed")})
    checks = RC.check_rows(res.teams, run_id)
    if checks:
        from .dfs.rules import store_checks
        store_checks(conn, checks)
    return {"sim_path": str(path), "checks": checks}


# ============================================================================== historical

def _hist_games(ctx: Context, day: date) -> pd.DataFrame:
    tg = ctx.h.tg
    d = tg[(tg["date"] == _day_int(day)) & (tg["season_type"].isin(
        ["Regular Season", "PlayIn", "Playoffs"]))]
    return d


def _home_away(conn: sqlite3.Connection, day: date) -> pd.DataFrame:
    rows = conn.execute("SELECT game_id, team, is_home FROM team_games WHERE game_date=?",
                        (day.isoformat(),)).fetchall()
    teams: dict = {}
    for gid, team, is_home in rows:
        teams.setdefault(gid, {})[team] = int(bool(is_home))
    out = []
    for gid, t in sorted(teams.items()):
        if len(t) != 2:
            continue
        home = [k for k, v in t.items() if v]
        if len(home) == 1:
            out.append({"game_id": gid, "home": home[0],
                        "away": next(k for k in t if k != home[0])})
        else:
            # Neutral site (Cup knockouts, international games): team_games marks both teams
            # away and the nominal home team isn't stored, so the order is alphabetical and the
            # sign of a closing spread for these few games is uncertain.
            a, b = sorted(t)
            out.append({"game_id": gid, "home": a, "away": b})
    return pd.DataFrame(out, columns=["game_id", "home", "away"])


def historical_inputs(conn: sqlite3.Connection, ctx: Context, day: date, *,
                      protocol: str = "avail", at: datetime | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(games, panel, p_avail) for a past day using only data before it, except the
    availability oracle (protocol 'avail': regulars who did not play are out, §7.1 L4)."""
    games = _home_away(conn, day)
    if protocol == "avail":
        tg = ctx.h.tg[ctx.h.tg["date"] == _day_int(day)]
        t = tg[["game_id", "team", "date", "season", "margin", "n_ot"]].copy()
        t["key"] = t["game_id"] + "|" + t["team"]
        panel = M.oracle_panel(ctx.h, t)
        return games, panel.reset_index(drop=True), {}
    if protocol != "reports":
        raise ValueError(f"unknown protocol {protocol!r}")
    raise RuntimeError("use project_historical(protocol='reports')")


def project_historical(conn: sqlite3.Connection, game_date, *, protocol: str = "avail",
                       n_sims: int = 5_000, seed: int | None = None, ctx: Context | None = None,
                       bundle: Bundle | None = None, at: datetime | None = None,
                       log: Callable[[str], None] | None = None) -> DayResult:
    """Backtest projection of every game on `game_date` from data before it (closing lines as
    the market; availability per `protocol`: 'avail' oracle or 'reports' at first tip - 45)."""
    t0 = time.time()
    day = _as_date(game_date)
    season = season_of(day)
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    if seed is None:
        seed = stable_seed(day.isoformat(), f"hist-{protocol}", MODEL_VERSION)
    if protocol == "reports":
        games = _home_away(conn, day)
        if at is None:
            g = AV.day_games(conn, day)
            at = (g["tip"].min() if len(g) else AV._default_tip(day)) - timedelta(minutes=45)
        at = _aware(at)
        p_avail, extras, avinfo = report_avail(conn, day, at, games)
        res = project_games(conn, day, games, ctx=ctx, bundle=b, n_sims=n_sims, seed=seed,
                            p_avail=p_avail, at=None, env_mode="closing", extras=extras,
                            log=log)
        res.meta.update({"at": _iso(at), **avinfo})
    else:
        games, panel, p_avail = historical_inputs(conn, ctx, day, protocol=protocol)
        env = {g: environment(conn, g, None, mode="closing") for g in games["game_id"]}
        specs = build_games(ctx, b, conn, day, games, panel, p_avail, env)
        res = run_games(specs, b, n_sims, seed, day, log)
    res.meta.update({"protocol": protocol, "seconds_total": round(time.time() - t0, 2)})
    return res


# ============================================================================== evaluation

def crps_samples(x: np.ndarray, y: float) -> float:
    """CRPS of an empirical sample x for outcome y: E|X - y| - E|X - X'|/2 (sorted formula)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    e1 = np.abs(x - y).mean()
    i = np.arange(1, n + 1)
    e2 = 2.0 * ((2 * i - n - 1) * x).sum() / (n * n)
    return float(e1 - 0.5 * e2)


def _actuals(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    a = pd.read_sql_query(
        "SELECT game_id, player_id, game_date, min, pts, fg3m, reb, ast, stl, blk, tov FROM "
        "player_games WHERE season <= ? AND season >= ?", conn,
        params=(season, _prev_season(season)))
    cats = sum((a[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    a["dk"] = (sum(C.DK_W[s] * a[s].astype(float) for s in C.DK_STATS)
               + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
    a["season"] = [season_of(date.fromisoformat(d)) for d in a["game_date"]]
    return a


def _baselines(acts: pd.DataFrame, rows: pd.DataFrame, panel_base: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date DK average (FPPG convention: last season's average before the 5th game)
    and EWMA minutes (rung-2, avail water-fill) x season-to-date DK per minute, per row
    (player_id, game_date)."""
    a = acts[acts["min"] > 0].sort_values(["player_id", "game_date"])
    out = []
    by = {pid: g for pid, g in a.groupby("player_id")}
    for r in rows.itertuples(index=False):
        g = by.get(int(r.player_id))
        std = dpm = np.nan
        if g is not None:
            prev = g[g["game_date"] < r.game_date]
            cur = prev[prev["season"] == r.season]
            last = prev[prev["season"] == _prev_season(r.season)]
            src = cur if len(cur) >= 5 else (last if len(last) else cur)
            if len(src):
                std = float(src["dk"].mean())
                dpm = float(src["dk"].sum() / max(src["min"].sum(), 1.0))
        out.append((std, dpm))
    o = np.array(out, float).reshape(-1, 2)
    res = rows.copy()
    res["base_std"] = o[:, 0]
    res["base_dpm"] = o[:, 1]
    res = res.merge(panel_base.drop_duplicates(["game_id", "player_id"]),
                    on=["game_id", "player_id"], how="left")
    res["base_ewma"] = res["base_min"].fillna(0.0) * res["base_dpm"]
    return res


def backtest(conn: sqlite3.Connection, season: str = "2024-25", *, n_dates: int = 60,
             n_sims: int = 4_000, protocol: str = "avail", ctx: Context | None = None,
             bundle: Bundle | None = None, log: Callable[[str], None] = print,
             dates: list | None = None) -> dict:
    """Quick walk-forward evaluation (fit < S) on `n_dates` regular-season dates spread evenly
    over S: DK FP MAE vs season-to-date and EWMA-minutes x season rate, CRPS, 50/80% coverage,
    calibration by tier, teammate/opponent correlation (sim vs actual)."""
    from .minutes_eval import baseline as rung2_baseline
    t0 = time.time()
    ctx = ctx or Context(conn, season, log=log)
    b = bundle or load_bundle(conn, season, ctx, log=log)
    tg = ctx.h.tg
    rs = tg[(tg["season"] == season) & (tg["season_type"] == "Regular Season")]
    all_days = sorted(rs["date"].unique())
    if dates is None:
        pick = np.unique(np.linspace(0, len(all_days) - 1, n_dates).round().astype(int))
        dates = [date(1970, 1, 1) + timedelta(days=int(all_days[i])) for i in pick]
    acts = _actuals(conn, season)
    act_key = acts.set_index(["game_id", "player_id"])
    frames, times, corr_rows = [], [], []
    for k, day in enumerate(dates):
        t1 = time.time()
        res = project_historical(conn, day, protocol=protocol, n_sims=n_sims, ctx=ctx, bundle=b)
        times.append((time.time() - t1, len(res.sims)))
        rows = []
        for g in res.sims:
            rows += _eval_rows(g, act_key, day, season)
            corr_rows.append(_sim_pair_corr(g))
        if not rows:
            continue
        d = pd.DataFrame(rows)
        # rung-2 EWMA minutes baseline (avail water-fill) on the same day's oracle panel
        _, panel, _ = historical_inputs(conn, ctx, day, protocol="avail")
        panel = panel.copy()
        panel["base_min"] = rung2_baseline(panel)
        bl = _baselines(acts, d[["game_id", "player_id", "game_date", "season"]],
                        panel[["game_id", "player_id", "base_min"]])
        for c in ("base_std", "base_ewma", "base_min"):
            d[c] = bl[c].to_numpy()
        frames.append(d)
        if log and (k % 10 == 0 or k == len(dates) - 1):
            log(f"  {day} ({k + 1}/{len(dates)}): {len(res.sims)} games in {times[-1][0]:.1f}s; "
                f"rss {peak_rss_mb():.0f} MB")
    df = pd.concat(frames, ignore_index=True)
    out = {"season": season, "protocol": protocol, "n_dates": len(dates), "n_sims": n_sims,
           "seconds": round(time.time() - t0, 1),
           "seconds_per_game": round(sum(t for t, _ in times) / max(sum(n for _, n in times), 1),
                                     3),
           "copula": json.loads(b.copula.to_json())}
    out.update(evaluate_rows(df))
    out["sim_corr"] = _pool_corr(corr_rows)
    out["actual_corr"] = _actual_pair_corr(df)
    out["frame"] = df
    return out


def _eval_rows(g, act_key: pd.DataFrame, day: date, season: str, sub: int = 2000) -> list:
    """One evaluation row per simulated player of a game (actual DK, sim quantiles, CRPS)."""
    info = g.info.reset_index(drop=True)
    sub = min(sub, g.fp.shape[0])
    rows = []
    for j in range(len(info)):
        pid = int(info.at[j, "player_id"])
        try:
            a = act_key.loc[(g.game_id, pid)]
            y, ymin = float(a["dk"]), float(a["min"] or 0.0)
        except KeyError:
            y, ymin = 0.0, 0.0
        x = g.fp[:, j].astype(float)
        mins = g.minutes[:, j]
        xp = x[mins > 0]
        q = np.quantile(x, (.1, .25, .5, .75, .9))
        r = {"game_id": g.game_id, "player_id": pid, "game_date": day.isoformat(),
             "season": season, "team": info.at[j, "team"], "tier": info.at[j, "tier"],
             "p_play": float(info.at[j, "p_play"]), "y": y, "played": ymin > 0,
             "act_min": ymin, "mean": float(x.mean()), "q10": q[0], "q25": q[1], "q50": q[2],
             "q75": q[3], "q90": q[4], "crps": crps_samples(x[:sub], y),
             "exp_min": float(mins.mean()),
             "min_played": float(mins[mins > 0].mean()) if (mins > 0).any() else 0.0,
             "pmean": np.nan}
        if len(xp) >= 50:
            qp = np.quantile(xp, (.1, .25, .75, .9))
            r.update({"pmean": float(xp.mean()), "pq10": qp[0], "pq25": qp[1], "pq75": qp[2],
                      "pq90": qp[3], "pcrps": crps_samples(xp[:sub], y)})
        rows.append(r)
    return rows


def _sim_pair_corr(g) -> dict:
    """Mean simulated DK correlation of teammate / opponent pairs (both p_play >= .9, played
    minutes >= 15) for one game."""
    info = g.info.reset_index(drop=True)
    ok = np.flatnonzero((info["p_play"] >= 0.9) & (info["min_played"] >= 15))
    if len(ok) < 4:
        return {"team": [], "opp": []}
    c = np.corrcoef(g.fp[:, ok].astype(float).T)
    team = info["team"].to_numpy()[ok]
    tm, op = [], []
    for a in range(len(ok)):
        for b_ in range(a + 1, len(ok)):
            (tm if team[a] == team[b_] else op).append(c[a, b_])
    return {"team": tm, "opp": op}


def _pool_corr(rows: list) -> dict:
    t = [x for r in rows for x in r["team"]]
    o = [x for r in rows for x in r["opp"]]
    return {"team": float(np.mean(t)) if t else None, "opp": float(np.mean(o)) if o else None,
            "n_team": len(t), "n_opp": len(o)}


def _actual_pair_corr(df: pd.DataFrame) -> dict:
    """Pooled correlation of standardised actual residuals (y - mean)/sd(sim) over the same
    kind of pairs (played, projected played minutes >= 15)."""
    d = df[(df["played"]) & (df["min_played"] >= 15) & (df["p_play"] >= 0.9)].copy()
    sd = (d["q90"] - d["q10"]) / 2.563
    d["z"] = (d["y"] - d["mean"]) / sd.clip(lower=1.0)
    d["z"] = d["z"] - d["z"].mean()
    v = float((d["z"] ** 2).mean())
    tm = op = 0.0
    nt = no = 0
    for gid, g in d.groupby("game_id"):
        z = g["z"].to_numpy()
        t = g["team"].to_numpy()
        for a in range(len(z)):
            for b_ in range(a + 1, len(z)):
                if t[a] == t[b_]:
                    tm += z[a] * z[b_]
                    nt += 1
                else:
                    op += z[a] * z[b_]
                    no += 1
    return {"team": tm / max(nt, 1) / v, "opp": op / max(no, 1) / v, "n_team": nt, "n_opp": no}


def _cl_se(d: np.ndarray, cl: np.ndarray) -> float:
    d = d - d.mean()
    s = pd.Series(d).groupby(cl).sum().to_numpy()
    return float(np.sqrt((s ** 2).sum()) / len(d))


def evaluate_rows(df: pd.DataFrame) -> dict:
    """Metrics on backtest rows: unconditional (exp minutes >= 10, DNP = 0) and conditional
    (played, projected played minutes >= 15)."""
    out: dict = {}
    u = df[(df["exp_min"] >= 10) & df["base_std"].notna()].copy()
    u["base_ewma"] = u["base_ewma"].fillna(u["base_std"])
    y = u["y"].to_numpy()
    err = {"model_mean": np.abs(u["mean"] - y), "model_p50": np.abs(u["q50"] - y),
           "season_to_date": np.abs(u["base_std"] - y),
           "ewma_min_x_season_rate": np.abs(u["base_ewma"] - y)}
    gid = u["game_id"].to_numpy()
    out["uncond"] = {"n": int(len(u)), "dnp_rate": float((~u["played"]).mean()),
                     "mae": {k: float(v.mean()) for k, v in err.items()},
                     "crps_model": float(u["crps"].mean()),
                     "gain_vs_std_pct": float((1 - err["model_mean"].mean()
                                               / err["season_to_date"].mean()) * 100),
                     "gain_vs_std_se_pct": _cl_se((err["model_mean"] - err["season_to_date"])
                                                  .to_numpy(), gid)
                     / err["season_to_date"].mean() * 100,
                     "gain_vs_ewma_pct": float((1 - err["model_mean"].mean()
                                                / err["ewma_min_x_season_rate"].mean()) * 100),
                     "cov80": float(((u["q10"] <= y) & (y <= u["q90"])).mean()),
                     "cov50": float(((u["q25"] <= y) & (y <= u["q75"])).mean()),
                     "bias": float((u["mean"] - y).mean())}
    c = df[df["played"] & (df["min_played"] >= 15) & df["pmean"].notna()]
    yc = c["y"].to_numpy()
    out["cond_played"] = {"n": int(len(c)), "mae": float(np.abs(c["pmean"] - yc).mean()),
                          "crps": float(c["pcrps"].mean()),
                          "cov80": float(((c["pq10"] <= yc) & (yc <= c["pq90"])).mean()),
                          "cov50": float(((c["pq25"] <= yc) & (yc <= c["pq75"])).mean()),
                          "bias": float((c["pmean"] - yc).mean())}
    tiers = {}
    u2 = u.copy()
    u2["lvl"] = pd.cut(u2["mean"], [-1, 10, 20, 30, 40, 200],
                       labels=["<10", "10-20", "20-30", "30-40", "40+"])
    for name, g in list(u2.groupby("tier")) + list(u2.groupby("lvl", observed=True)):
        yy = g["y"].to_numpy()
        tiers[str(name)] = {"n": int(len(g)), "pred": float(g["mean"].mean()),
                            "actual": float(yy.mean()),
                            "mae": float(np.abs(g["mean"] - yy).mean()),
                            "mae_std": float(np.abs(g["base_std"] - yy).mean()),
                            "cov80": float(((g["q10"] <= yy) & (yy <= g["q90"])).mean()),
                            "cov50": float(((g["q25"] <= yy) & (yy <= g["q75"])).mean())}
    out["calibration"] = tiers
    return out


def format_backtest(res: dict) -> str:
    u, c = res["uncond"], res["cond_played"]
    lines = [f"walk-forward {res['season']} ({res['protocol']}), {res['n_dates']} dates, "
             f"{res['n_sims']} sims; {res['seconds']}s ({res['seconds_per_game']}s/game)",
             f"unconditional (exp min >= 10, DNP = 0): n={u['n']} dnp={u['dnp_rate']:.3f}",
             "  MAE " + ", ".join(f"{k} {v:.3f}" for k, v in u["mae"].items()),
             f"  gain vs season-to-date {u['gain_vs_std_pct']:+.2f}% "
             f"(SE {u['gain_vs_std_se_pct']:.2f}), vs EWMA x rate {u['gain_vs_ewma_pct']:+.2f}%",
             f"  CRPS {u['crps_model']:.3f}  cov80 {u['cov80']:.3f}  cov50 {u['cov50']:.3f}  "
             f"bias {u['bias']:+.3f}",
             f"conditional (played, proj min >= 15): n={c['n']} MAE {c['mae']:.3f} CRPS "
             f"{c['crps']:.3f} cov80 {c['cov80']:.3f} cov50 {c['cov50']:.3f} bias {c['bias']:+.3f}",
             "calibration (unconditional):"]
    for k, v in res["calibration"].items():
        lines.append(f"  {k:<6} n={v['n']:<5} pred {v['pred']:6.2f} actual {v['actual']:6.2f} "
                     f"MAE {v['mae']:.2f} (std {v['mae_std']:.2f}) cov80 {v['cov80']:.3f} "
                     f"cov50 {v['cov50']:.3f}")
    sc, ac = res["sim_corr"], res["actual_corr"]
    lines.append(f"pair corr teammates sim {sc['team']:.3f} vs actual {ac['team']:.3f}; "
                 f"opponents sim {sc['opp']:.3f} vs actual {ac['opp']:.3f}")
    return "\n".join(lines)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
