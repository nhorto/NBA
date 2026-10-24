"""RotoGuru DFS backtest harness (docs/PLAYERS.md §7, §8 rows T2.1 / T2.4 / T2.17).

    courtside proj backtest-dfs [--seasons 2019-20 2020-21 ...] [--dates N] [--protocol avail]
                                [--sims 4000] [--topn 20] [--workers 1] [--leak-probe N]
    courtside proj backtest-dfs --seasons 2025-26 --protocol live|live10 --slice --modes
                                                                  (T2.4 live-like slice)

Protocols: ``avail`` (known availability, §7.1 L4), ``reports`` (official report at the day's
first tip - 45), ``live`` / ``live10`` (T2.4: the live lock path per game at its own tip - 45 /
- 10: `live_like_day`, `LiveInputs`; out-of-month P(play) params; lock-10 with actual starters).
The slice tables and the monitor's baseline file come from `slice_report` /
`live_baseline_json` (driver: ``research/t24_slice.py``).

Register by adding ``"courtside.proj.backtest"`` to ``PROJ_MODULES`` in ``proj/cli.py``.

Honest walk-forward only. Each season S is projected with a bundle fit on seasons < S
(`pipeline.load_bundle`, cached under ``var/proj``); salary-implied lines use RotoGuru rows of
seasons < S; every baseline (season average, DK FPPG convention, last 10) is rebuilt from box
scores strictly before the date. RotoGuru seasons (2014-15 .. 2020-21) get player- and
lineup-level results; other seasons (2024-25, or 2025-26 with ``--protocol reports``) are
player-level only (no salaries).

Per date, for RotoGuru seasons:
  * projections for every game (`pipeline.project_historical`), per-player: DK points actual,
    mean, 10/25/50/75/90% quantiles, CRPS and randomised PIT from the sims, and the baselines;
  * lineups scored with the ACTUAL RotoGuru DK points: ours (mean objective = ``ours_mean``,
    cash objective mu - 0.3 (mu - p25) = ``ours_cash``), Top-N (``ours_topn``, ranks 1..N), each
    baseline's optimiser lineup (``<b>`` from the whole pool, ``<b>_avail`` restricted to
    players our availability oracle keeps: the like-for-like comparator), and the
    hindsight-optimal lineup (same eligibility);
  * the percentile of every lineup among 2,000 random valid lineups of the slate pool
    (``pct_rand``: uniform picks, ``pct_spend``: salary-weighted picks that use >= 96% of the cap).

Per-date results are cached as parquet under ``<data>/reports/backtest/cache/<protocol>/<season>
/<config>/`` (the config hash covers settings, harness version, model version, bundle, rules and the source of the projection / optimiser modules),
so reruns are incremental and several processes can share a season (``--workers``).

Public API
  run(conn, seasons, *, protocol, modes, n_topn, baselines, sims, ...) -> report dict
  run_season(conn, season, settings, ...) -> (players, lineups, meta)
  leak_probe(conn, season, days, ...) -> dict  (the physical-removal probe)
  cut_context(ctx, day) -> Context with every frame truncated strictly before `day`
  player_report / lineup_report / format_report / write_reports
  random_lineups(pool, rules, n, rng, ...) -> (m, n_slots) player indices
  BoxBaselines(conn, season).get(player_id, day) -> (fppg, season_avg, last10)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .. import config
from . import evallite as E
from . import metrics as MT

HARNESS_VERSION = 1
BASELINES = ("salary", "season_avg", "fppg_proxy", "last10")
ROTOGURU_SEASONS = ("2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21")
DEFAULT_SEASONS = ROTOGURU_SEASONS + ("2024-25",)
REPORTS_FIRST = "2025-12-22"            # first day with rescued official injury reports
SLICE = ("2025-12-22", "2026-04-12")    # T2.4 live-like slice: rescued reports, regular season
# Live-like protocols (T2.4): minutes before EACH GAME'S OWN TIP at which the official report is
# read (DK runs a Showdown slate per game, so the graded live run of a game is its own lock run).
# 'live' = lock-45 (no starters, §7.1 L5); 'live10' = lock-10 with the actual starters as the
# proxy for DK's announced starters (L5).
LIVE_OFFSETS = {"live": 45, "live10": 10}
PROTOCOLS = ("avail", "reports", *LIVE_OFFSETS)
PPLAY_MODES = ("loo_month", "shipped")  # P(play) params for live protocols (see LiveInputs)
LIVE_INPUTS_VERSION = 2                 # DK-pool stand-in rules (LiveInputs); keys the live caches
NEW_PLAYER_DAYS = 2                     # a player's first game of S within this many days: in pool
CASH_LAMBDA = E.CASH_LAMBDA
LEAK_TOL = 0.005                        # §8 T2.1: leak probe < 0.5% (mean abs relative)
SPEND_FRAC = 0.96
SOLVE_LIMIT_S = 20.0
PLAYER_COLS = ("date", "season", "rg_id", "player_id", "game_id", "team", "pos", "salary", "y",
               "minutes", "has_proj", "mean", "q10", "q25", "q50", "q75", "q90", "exp_min",
               "p_play", "tier", "crps", "pit", "sal_all", "sal_prev", "sal_played", "sal_fixed",
               "b_season_avg", "b_fppg", "b_last10",
               # player-level (box) rows only; additive, so older caches load with NaN here
               "min_played", "p_avail", "status", "env_source")
# optional upper-tail output (Settings.tails; docs/research/xfactor.md): per-player sim q95/q99
# and, on RotoGuru rows, P(DK >= k * salary/1000) for k in TAIL_KS. Off by default; when off the
# frame, the cache and the config hash are exactly as before.
TAIL_KS = (5, 6, 7)
TAIL_COLS = ("q95", "q99", *(f"p_ge_{k}x" for k in TAIL_KS))


# ====================================================================== settings and cache

@dataclass(frozen=True)
class Settings:
    protocol: str = "avail"
    n_sims: int = 4_000
    modes: tuple = ("cash", "topn")
    n_topn: int = 20
    topn_min_diff: int = 1
    lam: float = CASH_LAMBDA
    baselines: tuple = BASELINES
    topn_baselines: tuple = ("fppg_proxy",)
    n_random: int = 2_000
    crps_sub: int = 2_000
    pplay: str = ""          # live protocols: PPLAY_MODES ("" = not used; kept out of the hash)
    tails: bool = False      # add TAIL_COLS to the player rows (False = kept out of the hash)

    def lineups(self) -> bool:
        return bool(self.modes)


def is_rotoguru(season: str) -> bool:
    return E.is_rotoguru_season(season)


def cache_root() -> Path:
    return config.data_dir() / "reports" / "backtest"


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _rules_sig(rules) -> str:
    return _hash({"cap": rules.salary_cap, "games": rules.min_games, "teams": rules.min_teams,
                  "slots": rules.slot_names}) if rules is not None else "none"


CODE_FILES = ("pipeline.py", "minutes.py", "state.py", "copula.py", "reconcile.py",
              "availability.py", "evallite.py", "dfs/optimizer.py", "dfs/topn.py", "dfs/rules.py",
              "context.py", *sorted(f"features/{q.name}" for q in
                                    (Path(__file__).parent / "features").glob("*.py")))


def code_fingerprint() -> str:
    """Hash of the source of every module that can change a projection or a lineup, so a model
    edit invalidates the cache even when MODEL_VERSION was not bumped (the harness is the judge
    of model changes: a stale cache must never be reused). ``COURTSIDE_BT_CODE_TAG`` overrides it
    (pin a tag to keep a cache across unrelated edits)."""
    tag = os.environ.get("COURTSIDE_BT_CODE_TAG")
    if tag:
        return f"tag:{tag}"
    h = hashlib.sha1()
    root = Path(__file__).parent
    for f in CODE_FILES:
        p = root / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:12]


def config_key(settings: Settings, bundle_json: str, rules) -> str:
    """Hash of everything that changes a per-date result (cache directory name)."""
    from . import pipeline as P
    s = asdict(settings)
    if not s.get("pplay"):          # field added with T2.4: absent = the pre-T2.4 key, unchanged
        s.pop("pplay", None)
    if not s.get("tails"):          # optional tail output: off = the earlier key, unchanged
        s.pop("tails", None)
    if settings.protocol in LIVE_OFFSETS:
        s["live_inputs"] = LIVE_INPUTS_VERSION
    return f"v{HARNESS_VERSION}-" + _hash({"s": s, "model": P.MODEL_VERSION,
                                           "bundle": hashlib.sha1(bundle_json.encode()).hexdigest(),
                                           "rules": _rules_sig(rules), "code": code_fingerprint()})


def _write_atomic(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    writer(tmp)
    os.replace(tmp, path)


class DateCache:
    """Per-date results of one (protocol, season, config): ``<date>.players.parquet``,
    ``<date>.lineups.parquet`` and ``<date>.json`` (meta, written last = the commit marker)."""

    def __init__(self, root: Path):
        self.root = root

    def meta_path(self, day: str) -> Path:
        return self.root / f"{day}.json"

    def has(self, day: str) -> bool:
        return self.meta_path(day).exists()

    def save(self, day: str, players: pd.DataFrame, lineups: pd.DataFrame, meta: dict) -> None:
        _write_atomic(self.root / f"{day}.players.parquet", lambda p: players.to_parquet(p))
        _write_atomic(self.root / f"{day}.lineups.parquet", lambda p: lineups.to_parquet(p))
        _write_atomic(self.meta_path(day), lambda p: p.write_text(json.dumps(E._clean(meta))))

    def load_meta(self, day: str) -> dict:
        return json.loads(self.meta_path(day).read_text())

    def load(self, day: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        meta = json.loads(self.meta_path(day).read_text())
        if meta.get("error"):
            return pd.DataFrame(), pd.DataFrame(), meta
        return (pd.read_parquet(self.root / f"{day}.players.parquet"),
                pd.read_parquet(self.root / f"{day}.lineups.parquet"), meta)


# ====================================================================== box baselines

class BoxBaselines:
    """Per-player causal DK-points baselines rebuilt from box scores (games with min > 0) of the
    season and the one before; every value uses games strictly before `day`.

      fppg        DK FPPG convention: this season's mean once >= 5 games, else last season's
                  (identical to `evallite.std_baseline`; bubble = 2019-20)
      season_avg  this season's mean to date (last season's when no game yet)
      last10      mean of the last up-to-10 games played (across the season boundary)"""

    def __init__(self, conn: sqlite3.Connection, season: str):
        from . import pipeline as P
        full = P._actuals(conn, season)
        a = full[full["min"] > 0].sort_values(["player_id", "game_date"])
        self.cs0 = E.season_start(season)
        self.cs1 = E.season_start(E.prev_season(season))
        self.by: dict[int, tuple[list, np.ndarray]] = {}
        for pid, g in a.groupby("player_id"):
            self.by[int(pid)] = (list(g["game_date"]),
                                 np.concatenate([[0.0], np.cumsum(g["dk"].to_numpy(float))]))
        self.actual = {(gid, int(pid)): (float(dk), float(mn or 0.0)) for gid, pid, dk, mn in zip(
            *[full[c] for c in ("game_id", "player_id", "dk", "min")])}

    def get(self, pid, day: str) -> tuple[float, float, float]:
        import bisect
        nan = float("nan")
        if pid is None or (isinstance(pid, float) and np.isnan(pid)) or int(pid) not in self.by:
            return nan, nan, nan
        dts, cs = self.by[int(pid)]
        i = bisect.bisect_left(dts, day)                     # games strictly before the date
        j = min(bisect.bisect_left(dts, self.cs0), i)
        h = min(bisect.bisect_left(dts, self.cs1), j)
        n_cur, n_last = i - j, j - h
        cur = (cs[i] - cs[j]) / n_cur if n_cur else nan
        last = (cs[j] - cs[h]) / n_last if n_last else nan
        fppg = cur if n_cur >= 5 else (last if n_last > 0 else cur)
        season_avg = cur if n_cur else last
        n10 = min(i, 10)
        last10 = (cs[i] - cs[i - n10]) / n10 if n10 else nan
        return float(fppg), float(season_avg), float(last10)


# ====================================================================== random lineups

def random_lineups(players: Sequence, rules, n: int, rng: np.random.Generator, *,
                   weight_pow: float = 0.0, min_salary: float = 0.0, batch: int = 4_000,
                   max_batches: int = 60) -> np.ndarray:
    """Up to `n` random lineups, each row the pool indices of the roster slots in
    ``rules.slot_names`` order. Every slot is filled in turn with a random eligible player not
    yet used (probability proportional to salary**weight_pow), and a row is kept only when it is
    a legal DK lineup: salary in [min_salary, cap], >= min_games games, >= min_teams teams.
    Restrictive slots (PG, SG, SF, PF, C) are filled before G, F, UTIL, so the draw is not
    uniform over all legal lineups, only a fixed, documented random reference."""
    names = rules.slot_names
    npl = len(players)
    elig = np.array([[s in p.slots for s in names] for p in players], bool)
    sal = np.array([p.salary for p in players], float)
    teams = np.unique([p.team or "?" for p in players], return_inverse=True)[1]
    games = np.unique([p.game_id or "?" for p in players], return_inverse=True)[1]
    w = np.maximum(sal, 1.0) ** weight_pow
    kept: list[np.ndarray] = []
    have = 0
    for _ in range(max_batches):
        used = np.zeros((batch, npl), bool)
        sel = np.zeros((batch, len(names)), np.int64)
        ok = np.ones(batch, bool)
        for k in range(len(names)):
            wk = w[None, :] * elig[:, k][None, :] * ~used
            cs = np.cumsum(wk, axis=1)
            ok &= cs[:, -1] > 0
            r = rng.random(batch) * cs[:, -1]
            idx = np.minimum((cs < r[:, None]).sum(1), npl - 1)
            sel[:, k] = idx
            used[np.arange(batch), idx] = True
        tot = sal[sel].sum(1)
        nt = 1 + (np.diff(np.sort(teams[sel], axis=1), axis=1) != 0).sum(1)
        ng = 1 + (np.diff(np.sort(games[sel], axis=1), axis=1) != 0).sum(1)
        v = ok & (tot <= rules.salary_cap) & (tot >= min_salary)
        v &= nt >= (rules.min_teams or 0)
        v &= ng >= (rules.min_games or 0)
        kept.append(sel[v])
        have += int(v.sum())
        if have >= n:
            break
    out = np.concatenate(kept) if kept else np.zeros((0, len(names)), np.int64)
    return out[:n]


class RandomReference:
    """Actual-point distribution of random legal lineups of one slate pool."""

    def __init__(self, pool, rules, actual: dict, rng: np.random.Generator, n: int):
        pl = pool.players
        act = np.array([float(actual.get(p.dk_player_id, 0.0) or 0.0) for p in pl])
        any_ = random_lineups(pl, rules, n, rng)
        spend = random_lineups(pl, rules, n, rng, weight_pow=2.0,
                               min_salary=SPEND_FRAC * rules.salary_cap)
        self.any = np.sort(act[any_].sum(1)) if len(any_) else np.zeros(0)
        self.spend = np.sort(act[spend].sum(1)) if len(spend) else np.zeros(0)

    def pct(self, value: float) -> tuple[float, float]:
        return (MT.percentile_rank(value, self.any), MT.percentile_rank(value, self.spend))


# ====================================================================== live-like inputs (T2.4)

class LiveInputs:
    """What a live lock run knows about a past day that the box-score history alone does not.

    * **DK pool stand-in** (`team_at`, `pool_extras`): live runs force every DK-pool player onto
      his CURRENT team (`pipeline.merge_extras`) and drop candidates whose current roster team
      differs (`drop_moved_players`). There is no DK pool for 2025-26, so a player's team on day
      d is the team of his next game of season S on or after d (a trade is known to the pool the
      day it happens). In the stand-in: players with a game of S before d AND a game of S on or
      after d (a player who never plays again is either on the report as Out, handled there, or
      off a roster), plus players without an earlier game of S whose first one is within
      NEW_PLAYER_DAYS of d (a fresh signing or a debut: the pool lists him once he is signed).
      Without that guard a player signed in March sat in December's pool on his old history
      (v1 of the stand-in: ~11 such rows per night, P(play) 0.52 vs 4% played). This is the
      mid-season generalisation of `live_path_eval.opening_panel` (first team of S) and uses
      the future only for roster membership, which the real pool knows at lock.
    * **P(play) params** (`params`): the shipped `availability_params.json` was fit on exactly
      the slice (2025-12-22 .. 2026-04-12), so the slice uses out-of-block params by default
      ('loo_month': fit on the other regular-season months of the rescued reports); 'shipped'
      reproduces the live params (in-sample here).
    * **Starters** (`starters`, lock-10 only): the actual starters (`player_starts`), the §7.1
      L5 proxy for DK's announced starters.
    * **Environment** (`environment`): the books' `line_snapshots` at the run time when any is
      archived for the game, else the closing line (never the gbm_market fallback: the slice's
      prediction rows are a replay, not what was known). `source` records which one."""

    def __init__(self, conn: sqlite3.Connection, season: str, pplay: str = "loo_month",
                 log: Callable[[str], None] = print):
        from . import availability as AV
        if pplay not in PPLAY_MODES:
            raise ValueError(f"unknown pplay mode {pplay!r}")
        self.conn, self.season, self.pplay, self.log = conn, season, pplay, log
        lo, hi = E.season_start(season), E.season_start(E.next_season(season))
        pg = pd.read_sql_query(
            "SELECT player_id, team, game_date FROM player_games WHERE game_date>=? AND "
            "game_date<? AND min>0 ORDER BY game_date", conn, params=(lo, hi))
        self.by: dict[int, tuple[list[str], list[str]]] = {
            int(p): (list(g["game_date"]), list(g["team"])) for p, g in pg.groupby("player_id")}
        self.hist = AV.load_history(conn, date.fromisoformat(lo))
        self._params: dict[str, object] = {}
        self._samples: pd.DataFrame | None = None

    # ---- roster stand-in -----------------------------------------------------------------
    def team_at(self, day) -> dict[int, str]:
        """{player_id: team} of the DK-pool stand-in on `day` (see the class docstring)."""
        import bisect
        d = str(day)[:10]
        soon = (P_date(d) + timedelta(days=NEW_PLAYER_DAYS)).isoformat()
        out = {}
        for pid, (dts, teams) in self.by.items():
            i = bisect.bisect_left(dts, d)
            if i < len(dts) and (i > 0 or dts[i] <= soon):
                out[pid] = teams[i]
        return out

    @staticmethod
    def pool_extras(current: dict[int, str], games: pd.DataFrame) -> dict:
        """{(game_id, team): [player_id]} for the teams of `games` (the DK pool as extras)."""
        by_team: dict[str, list[int]] = {}
        for pid, t in current.items():
            by_team.setdefault(t, []).append(int(pid))
        out = {}
        for g in games.itertuples(index=False):
            for t in (g.home, g.away):
                if by_team.get(t):
                    out[(g.game_id, t)] = sorted(by_team[t])
        return out

    # ---- P(play) params --------------------------------------------------------------------
    def params(self, day):
        """AvailabilityParams for `day`: shipped, or fit on the regular-season report days of
        every OTHER calendar month (cached per month)."""
        from . import availability as AV
        if self.pplay == "shipped":
            if "shipped" not in self._params:
                self._params["shipped"] = AV.load_params()
            return self._params["shipped"]
        d = P_date(day)
        month = d.strftime("%Y-%m")
        if month not in self._params:
            if self._samples is None:
                t0 = time.time()
                s = AV.build_samples(self.conn, AV.TRAIN_LO, AV.RS_END, mode="bucket",
                                     hist=self.hist)
                s = s[s["season_type"] == "Regular Season"] if len(s) else s
                self._samples = s
                self.log(f"P(play) training rows {len(s)} built in {time.time() - t0:.0f}s")
            s = self._samples
            keep = np.array([x.strftime("%Y-%m") != month for x in s["day"]], bool) \
                if len(s) else np.zeros(0, bool)
            train = s[keep]
            meta = {"fit": "leave-one-month-out", "held_out_month": month,
                    "rows": int(len(train)), "games": int(train["game_id"].nunique())
                    if len(train) else 0}
            self._params[month] = AV.fit_from_samples(train, m1=20.0, m=AV.DEFAULT_M, meta=meta)
        return self._params[month]

    def params_tag(self) -> str:
        return self.pplay

    # ---- starters / environment ------------------------------------------------------------
    def starters(self, game_ids) -> set[int]:
        ids = list(game_ids)
        if not ids:
            return set()
        ph = ",".join("?" * len(ids))
        return {int(r[0]) for r in self.conn.execute(
            f"SELECT player_id FROM player_starts WHERE game_id IN ({ph})", ids)}

    def environment(self, game_id: str, at: datetime) -> dict:
        from . import pipeline as P
        n = self.conn.execute(
            "SELECT COUNT(*) FROM line_snapshots WHERE game_id=? AND captured_at<=? AND "
            "book!='open'", (game_id, P._iso(at))).fetchone()[0]
        if n:
            return P.environment(self.conn, game_id, at, mode="live")
        return P.environment(self.conn, game_id, None, mode="closing")


def P_date(day) -> date:
    return day if isinstance(day, date) and not isinstance(day, datetime) else \
        date.fromisoformat(str(day)[:10])


def report_inputs(av: pd.DataFrame) -> tuple[dict, dict, dict]:
    """`pipeline.report_avail` on an already computed `availability.p_play_slate` frame:
    (P(available) {player_id: p} for listed statuses, report players to force (listed, not
    out) {(game_id, team): [player_id]}, {player_id: report status} for every row)."""
    from . import pipeline as P
    p_avail, extras, status = {}, {}, {}
    for r in av.itertuples(index=False):
        status[int(r.player_id)] = r.status
        if r.status in P.LISTED:
            p_avail[int(r.player_id)] = float(r.p_play)
        if r.status in P.LISTED and r.status != "out":
            extras.setdefault((r.game_id, r.team), []).append(int(r.player_id))
    return p_avail, extras, status


def live_like_day(conn: sqlite3.Connection, ctx, bundle, live: LiveInputs, day, *,
                  offset_min: int = 45, n_sims: int = 4_000, seed: int | None = None,
                  use_starters: bool = False):
    """Project a past day the way the live lock runs did (T2.4): for each tip time, the
    official report at tip - `offset_min` (P(play) cells from `live.params(day)`), the DK-pool
    stand-in forced in (`merge_extras`, pool first), report-listed players forced in,
    `candidate_panel` + `drop_moved_players`, the environment of `live.environment`, then
    `build_games` / `run_games` exactly as `pipeline.project_games`. Every game draws from
    SeedSequence([seed, crc32(game_id)]), so grouping by tip does not change the draws.
    Returns a `pipeline.DayResult` whose players frame also carries `status` (the report status
    at the run, '' when the player was not among the report's candidates) and `env_source`."""
    from . import availability as AV
    from . import pipeline as P
    d = P_date(day)
    games = P._home_away(conn, d)
    if seed is None:
        seed = P.stable_seed(d.isoformat(), f"hist-live{offset_min}", P.SEED_VERSION)
    tips = AV.day_games(conn, d)
    tip_of = dict(zip(tips["game_id"], tips["tip"]))
    games = games.assign(tip=[tip_of.get(g, AV._default_tip(d)) for g in games["game_id"]])
    current = live.team_at(d)
    params = live.params(d)
    sims, players, teams = [], [], []
    env_all: dict = {}
    info = {"groups": 0, "n_report_rows": 0, "n_listed": 0, "n_forced_pool": 0,
            "n_forced_report": 0, "n_dropped_moved": 0, "tip_unknown": int((~tips["tip_known"])
                                                                          .sum()) if len(tips) else 0}
    for tip, gg in games.groupby("tip", sort=True):
        gg = gg[["game_id", "home", "away"]].reset_index(drop=True)
        at = tip - timedelta(minutes=offset_min)
        starters = live.starters(gg["game_id"]) if use_starters else None
        av = AV.p_play_slate(conn, d, at, params, starters=starters, hist=live.hist)
        av = av[av["game_id"].isin(set(gg["game_id"]))]
        p_avail, rep_extras, status = report_inputs(av)
        pool = live.pool_extras(current, gg)
        ex = P.merge_extras(pool, rep_extras)
        panel = P.candidate_panel(ctx, d, gg, ex)
        n0 = len(panel)
        panel = P.drop_moved_players(panel, current, ex)
        pa = {(k, int(pid)): float(p_avail[int(pid)])
              for k, pid in zip(panel["key"], panel["player_id"]) if int(pid) in p_avail}
        env = {g: live.environment(g, at) for g in gg["game_id"]}
        if P.PPLAY == "v2":
            fix = P.pplay_v2_fix(ctx, panel, p_avail, P.listed_of(rep_extras), starters)
            specs = P.build_games(ctx, bundle, conn, d, gg, panel, pa, env, starters,
                                  pp_fix=fix)
        else:
            specs = P.build_games(ctx, bundle, conn, d, gg, panel, pa, env, starters)
        r = P.run_games(specs, bundle, n_sims, seed, d)
        pl = r.players.copy()
        pl["status"] = [status.get(int(p), "") for p in pl["player_id"]]
        pl["env_source"] = [env[g].get("source", "") for g in pl["game_id"]]
        pl["at"] = P._iso(at)
        sims += r.sims
        players.append(pl)
        teams += r.teams
        env_all.update(r.meta.get("env", {}))
        info["groups"] += 1
        info["n_report_rows"] += int(len(av))
        info["n_listed"] += int(av["status"].isin(list(P.LISTED)).sum()) if len(av) else 0
        info["n_forced_pool"] += sum(len(v) for v in pool.values())
        info["n_forced_report"] += sum(len(v) for v in rep_extras.values())
        info["n_dropped_moved"] += n0 - len(panel)
    pl = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    src = pd.Series([e.get("source", "") for e in env_all.values()]).str.split(":").str[0]
    meta = {"protocol": f"live{offset_min}", "seed": seed, "n_sims": n_sims, "env": env_all,
            "env_sources": {k: int(v) for k, v in src.value_counts().items()},
            "pplay": live.params_tag(), "starters": bool(use_starters), **info}
    return P.DayResult(day=d, sims=sims, players=pl, stats=pd.DataFrame(), teams=teams, meta=meta)


# ====================================================================== one season

class SeasonRun:
    """Everything one season's per-date work needs: bundle (fit < S), projection context, the
    RotoGuru frame and salary lines, causal box baselines, the Classic rules."""

    def __init__(self, conn: sqlite3.Connection, season: str, settings: Settings,
                 log: Callable[[str], None] = print, *, need_context: bool = True):
        from . import pipeline as P
        from .dfs import optimizer as O
        self.conn, self.season, self.settings, self.log = conn, season, settings, log
        self.rg_season = is_rotoguru(season)
        self.rules = O.classic_rules() if self.rg_season and settings.lineups() else None
        if self.rg_season and settings.lineups() and self.rules is None:
            raise LookupError("no archived Classic rules payload (needed for the lineup level)")
        self.bundle = P.load_bundle(conn, season, None, log=log)       # fit on seasons < S
        self.bundle_json = self.bundle.to_json()
        self.ctx = None
        self.box = None
        self.rg = None
        self.lines: dict = {}
        self.salary_fit = None
        self.live: LiveInputs | None = None
        if need_context:
            self.load()

    def load(self) -> "SeasonRun":
        from . import pipeline as P
        if self.ctx is not None:
            return self
        self.ctx = P.Context(self.conn, self.season, log=self.log)
        self.box = BoxBaselines(self.conn, self.season)
        if self.settings.protocol in LIVE_OFFSETS:
            self.live = LiveInputs(self.conn, self.season, self.settings.pplay or "loo_month",
                                   log=self.log)
        if self.rg_season:
            self.rg = E.load_rotoguru(self.conn)
            try:
                self.lines = {m: E.fit_salary_line(self.rg, self.season, m)
                              for m in E.SALARY_MODES}
                self.salary_fit = "ols_seasons_before_S"
            except ValueError:                       # 2014-15: no earlier RotoGuru season
                fixed = (0.0, 5.0 / 1000.0)
                self.lines = {m: fixed for m in E.SALARY_MODES}
                self.salary_fit = "fixed_5_per_1k (no earlier RotoGuru season)"
        return self

    def dates(self, n: int = 0) -> list[str]:
        from .dfs import optimizer as O
        if self.rg_season:
            lo, hi = E.season_start(self.season), E.season_start(E.next_season(self.season))
            days = [d for d in O.hist_dates(self.conn) if lo <= d < hi]
        else:
            days = [d.isoformat() for d in E.usable_dates(self.conn, self.season)[0]]
            if self.settings.protocol == "reports" or self.settings.protocol in LIVE_OFFSETS:
                days = [d for d in days if d >= REPORTS_FIRST]
        return E._pick(days, n)

    def cache(self) -> DateCache:
        key = config_key(self.settings, self.bundle_json, self.rules)
        return DateCache(cache_root() / "cache" / self.settings.protocol / self.season / key)

    # ---- one date -------------------------------------------------------------------------

    def run_date(self, day: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        from . import pipeline as P
        t0 = time.time()
        s = self.settings
        if s.protocol in LIVE_OFFSETS:
            off = LIVE_OFFSETS[s.protocol]
            res = live_like_day(self.conn, self.ctx, self.bundle, self.live, day, offset_min=off,
                                n_sims=s.n_sims, use_starters=off <= 10)
        else:
            res = P.project_historical(self.conn, day, protocol=s.protocol, n_sims=s.n_sims,
                                       ctx=self.ctx, bundle=self.bundle)
        pl = res.players.drop_duplicates("player_id").set_index("player_id")
        rng = np.random.default_rng(P.stable_seed(day, "backtest-pit", HARNESS_VERSION))
        if self.rg_season:
            players, proj, pool_rows = self._players_rotoguru(day, res, pl, rng)
        else:
            players, proj, pool_rows = self._players_box(day, res, pl, rng), {}, None
        lineups = pd.DataFrame()
        meta = {"date": day, "n_games": len(res.sims), "n_proj": int(players["has_proj"].sum()),
                "n_rows": int(len(players))}
        for k in ("env_sources", "groups", "n_report_rows", "n_listed", "n_forced_pool",
                  "n_forced_report", "n_dropped_moved", "tip_unknown", "pplay", "starters"):
            if k in res.meta:
                meta[k] = res.meta[k]
        if self.rg_season and s.lineups():
            lineups, extra = self._lineups(day, pool_rows, proj)
            meta.update(extra)
        meta["seconds"] = round(time.time() - t0, 2)
        meta["error"] = None
        return players, lineups, meta

    def _sim_scores(self, res, y_of: Callable[[int], float], rng) -> dict:
        """{player_id: (crps, randomised pit)} from the sims, for players with an outcome."""
        from .metrics import crps_samples_matrix, pit_randomized
        out = {}
        sub = self.settings.crps_sub
        for g in res.sims:
            pids = np.asarray(g.player_id)
            ys = np.array([y_of(int(p)) for p in pids], float)
            ok = np.isfinite(ys)
            if not ok.any():
                continue
            X = g.fp[:, ok].astype(float)
            yy = ys[ok]
            crps = crps_samples_matrix(X[:sub], yy)
            pit = pit_randomized(X, yy, rng.random(len(yy)))
            for pid, c, u in zip(pids[ok], crps, pit):
                out[int(pid)] = (float(c), float(u))
        return out

    def _players_rotoguru(self, day, res, pl, rng):
        rg = self.rg
        pool_rows = rg[rg["game_date"] == day]
        y_rg = {int(r.player_id): float(r.pts) for r in pool_rows.itertuples(index=False)
                if not pd.isna(r.player_id)}
        sc = self._sim_scores(res, lambda p: y_rg.get(p, np.nan), rng)
        a_all, b_all = self.lines["all_seasons"]
        a_pr, b_pr = self.lines["prev_season"]
        a_pl, b_pl = self.lines["played_only"]
        rows, proj = [], {}
        for r in pool_rows.itertuples(index=False):
            pid = None if pd.isna(r.player_id) else int(r.player_id)
            q = pl.loc[pid] if pid is not None and pid in pl.index else None
            sal = float(r.salary)
            fppg, savg, l10 = self.box.get(pid, day)
            row = {"date": day, "season": self.season, "rg_id": str(r.rg_id), "player_id": pid,
                   "game_id": None, "team": None, "pos": r.pos, "salary": sal,
                   "y": float(r.pts), "minutes": float(r.minutes or 0.0), "has_proj": q is not None,
                   "sal_all": a_all + b_all * sal, "sal_prev": a_pr + b_pr * sal,
                   "sal_played": a_pl + b_pl * sal, "sal_fixed": 5.0 * sal / 1000.0,
                   "b_season_avg": savg, "b_fppg": fppg, "b_last10": l10}
            if q is not None:
                proj[str(r.rg_id)] = q
                row.update({"game_id": q["game_id"], "team": q["team"], "mean": float(q["dk_mean"]),
                            "q10": float(q["dk_q10"]), "q25": float(q["dk_q25"]),
                            "q50": float(q["dk_q50"]), "q75": float(q["dk_q75"]),
                            "q90": float(q["dk_q90"]), "exp_min": float(q["min_mean"]),
                            "p_play": float(q["p_play"]), "tier": q["tier"]})
                c = sc.get(pid)
                if c is not None:
                    row["crps"], row["pit"] = c
            rows.append(row)
        frame = _frame(rows)
        if self.settings.tails:
            sal = {int(r.player_id): float(r.salary) for r in pool_rows.itertuples(index=False)
                   if not pd.isna(r.player_id)}
            frame = _with_tails(frame, sim_tails(res, sal))
        return frame, proj, pool_rows

    def _players_box(self, day, res, pl, rng):
        """Player-level rows for a season without salaries: every simulated player."""
        act = self.box.actual
        y_of = {}
        for g in res.sims:
            for pid in g.player_id:
                v = act.get((g.game_id, int(pid)))
                y_of[int(pid)] = v[0] if v else 0.0
        sc = self._sim_scores(res, lambda p: y_of.get(p, np.nan), rng)
        env = {g: (e or {}).get("source", "") for g, e in res.meta.get("env", {}).items()}
        rows = []
        for pid, q in pl.iterrows():
            pid = int(pid)
            a = act.get((q["game_id"], pid))
            y, mins = (a[0], a[1]) if a else (0.0, 0.0)
            fppg, savg, l10 = self.box.get(pid, day)
            row = {"date": day, "season": self.season, "rg_id": None, "player_id": pid,
                   "game_id": q["game_id"], "team": q["team"], "pos": None, "salary": np.nan,
                   "y": y, "minutes": mins, "has_proj": True, "mean": float(q["dk_mean"]),
                   "q10": float(q["dk_q10"]), "q25": float(q["dk_q25"]),
                   "q50": float(q["dk_q50"]), "q75": float(q["dk_q75"]),
                   "q90": float(q["dk_q90"]), "exp_min": float(q["min_mean"]),
                   "p_play": float(q["p_play"]), "tier": q["tier"],
                   "b_season_avg": savg, "b_fppg": fppg, "b_last10": l10,
                   "min_played": float(q["min_played"]) if "min_played" in q else np.nan,
                   "p_avail": float(q["p_avail"]) if "p_avail" in q else np.nan,
                   "status": q["status"] if "status" in q else None,
                   "env_source": q["env_source"] if "env_source" in q
                   else env.get(q["game_id"], "")}
            c = sc.get(pid)
            if c is not None:
                row["crps"], row["pit"] = c
            rows.append(row)
        frame = _frame(rows)
        return _with_tails(frame, sim_tails(res, {})) if self.settings.tails else frame

    def _lineups(self, day, pool_rows, proj):
        from . import pipeline as P
        from .dfs import optimizer as O
        from .dfs import topn as T
        s, rules = self.settings, self.rules
        pool = O.hist_pool(self.conn, day, rules)
        actual = {str(r.rg_id): float(r.pts) for r in pool_rows.itertuples(index=False)}
        in_pool = {p.dk_player_id: p for p in pool.players}
        ref = RandomReference(pool, rules, actual,
                              np.random.default_rng(P.stable_seed(day, "backtest-random",
                                                                  HARNESS_VERSION)), s.n_random)
        ours = {k: O.Projection(float(q["dk_mean"]), p25=float(q["dk_q25"]))
                for k, q in proj.items() if k in in_pool}
        avail = {k for k, q in proj.items() if k in in_pool and float(q["min_mean"]) >= 10}
        a, b = self.lines["all_seasons"]
        base: dict[str, dict] = {}
        if "salary" in s.baselines:
            base["salary"] = {k: O.Projection(a + b * p.salary) for k, p in in_pool.items()}
        col = {"season_avg": 1, "fppg_proxy": 0, "last10": 2}
        for name in s.baselines:
            if name == "salary":
                continue
            d = {}
            for r in pool_rows.itertuples(index=False):
                k = str(r.rg_id)
                if k not in in_pool or pd.isna(r.player_id):
                    continue
                v = self.box.get(int(r.player_id), day)[col[name]]
                if np.isfinite(v):
                    d[k] = O.Projection(float(v))
            base[name] = d
        jobs: dict[str, tuple] = {}
        if "cash" in s.modes:
            jobs["ours_mean"] = (ours, "mean")
            jobs["ours_cash"] = (ours, "cash")
            for name, d in base.items():
                jobs[name] = (d, "mean")
                jobs[f"{name}_avail"] = ({k: v for k, v in d.items() if k in avail}, "mean")
        jobs["hindsight"] = (O.hist_projections(self.conn, pool, "actual"), "mean")
        rows: list[dict] = []
        t_solve = time.time()

        def add(name, kind, lu, rank=1):
            val = E.lineup_actual(lu.player_ids, actual) if lu is not None else np.nan
            p1, p2 = ref.pct(val) if lu is not None else (np.nan, np.nan)
            rows.append({"date": day, "method": name, "kind": kind, "rank": rank, "actual": val,
                         "proj": float(lu.projected) if lu is not None else np.nan,
                         "salary": int(lu.salary) if lu is not None else 0,
                         "pct_rand": p1, "pct_spend": p2,
                         "ids": "|".join(sorted(lu.player_ids)) if lu is not None else ""})

        for name, (pj, mode) in jobs.items():
            lu = O.optimize(pool, pj, rules, mode=mode, lam=s.lam, time_limit_s=SOLVE_LIMIT_S)
            add(name, "hindsight" if name == "hindsight" else "single", lu)
        if "topn" in s.modes:
            tops = {"ours_topn": (ours, "mean")}
            for name in s.topn_baselines:
                if name in base:
                    tops[f"{name}_avail_topn"] = ({k: v for k, v in base[name].items()
                                                   if k in avail}, "mean")
            for name, (pj, mode) in tops.items():
                r = T.top_n(pool, pj, rules, s.n_topn, min_diff=s.topn_min_diff, mode=mode,
                            lam=s.lam, time_limit_s=SOLVE_LIMIT_S,
                            total_time_limit_s=SOLVE_LIMIT_S * 6)
                for rl in r.lineups:
                    add(name, "topn", rl.lineup, rl.rank)
        extra = {"n_pool": len(in_pool), "n_ours": len(ours), "n_avail": len(avail),
                 "n_random": [int(len(ref.any)), int(len(ref.spend))],
                 "seconds_solve": round(time.time() - t_solve, 2)}
        return pd.DataFrame(rows), extra


def sim_tails(res, salary: dict) -> dict:
    """{player_id: {q95, q99, p_ge_<k>x}} from the per-game sims (`res.sims`: fp (S, n) and
    player_id); P(FP >= k * salary/1000) only for players in `salary` (RotoGuru rows)."""
    out = {}
    for g in res.sims:
        X = np.asarray(g.fp, float)
        if not X.size:
            continue
        q = np.quantile(X, [0.95, 0.99], axis=0)
        for j, pid in enumerate(np.asarray(g.player_id)):
            pid = int(pid)
            d = {"q95": float(q[0, j]), "q99": float(q[1, j])}
            sal = salary.get(pid)
            if sal is not None and np.isfinite(sal):
                col = X[:, j]
                for k in TAIL_KS:
                    d[f"p_ge_{k}x"] = float((col >= k * sal / 1000.0).mean())
            out[pid] = d
    return out


def _with_tails(frame: pd.DataFrame, tails: dict) -> pd.DataFrame:
    for c in TAIL_COLS:
        frame[c] = [tails.get(int(p), {}).get(c, np.nan) if pd.notna(p) else np.nan
                    for p in frame["player_id"]]
    return frame


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in PLAYER_COLS:
        if c not in df:
            df[c] = np.nan
    return df[list(PLAYER_COLS)]


# ====================================================================== running

def _season_worker(args) -> list[str]:
    """Process-pool worker: runs its slice of dates for one season, writing the date cache."""
    season, days, settings_json, data_dir, tag = args
    os.environ["COURTSIDE_DATA"] = data_dir
    from .. import db
    conn = db.connect()
    st = Settings(**{k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in json.loads(settings_json).items()})
    sr = SeasonRun(conn, season, st, log=lambda m: print(f"[{tag}] {m}", flush=True))
    return _run_days(sr, days, sr.cache(), lambda m: print(f"[{tag}] {m}", flush=True))


def _run_days(sr: SeasonRun, days: list[str], cache: DateCache, log) -> list[str]:
    done = []
    t0 = time.time()
    for k, day in enumerate(days):
        if cache.has(day):
            continue
        try:
            players, lineups, meta = sr.run_date(day)
            cache.save(day, players, lineups, meta)
        except Exception as e:                       # keep running; the date is recorded as skipped
            _write_atomic(cache.meta_path(day), lambda p, e=e: p.write_text(json.dumps(
                {"date": day, "error": f"{type(e).__name__}: {e}"})))
            log(f"  {day}: SKIPPED {type(e).__name__}: {e}")
            done.append(day)
            continue
        done.append(day)
        if len(done) % 10 == 1:
            log(f"  {sr.season} {day} ({k + 1}/{len(days)}): {meta['seconds']}s, elapsed "
                f"{time.time() - t0:.0f}s")
    return done


def run_season(conn: sqlite3.Connection, season: str, settings: Settings, *, n_dates: int = 0,
               workers: int = 1, force: bool = False, log: Callable[[str], None] = print,
               date_lo: str | None = None, date_hi: str | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run (or load from the cache) every date of `season` (within [date_lo, date_hi] when
    given; `n_dates` evenly spaced among those); returns the concatenated player and lineup
    frames and a meta dict (seconds, skipped dates, cache directory, salary fit)."""
    t0 = time.time()
    sr = SeasonRun(conn, season, settings, log=log, need_context=False)
    cache = sr.cache()
    days = sr.dates(0)
    days = [d for d in days if (date_lo is None or d >= date_lo) and (date_hi is None or d <= date_hi)]
    days = E._pick(days, n_dates)
    if force:
        for d in days:
            for suffix in (".json", ".players.parquet", ".lineups.parquet"):
                (cache.root / f"{d}{suffix}").unlink(missing_ok=True)
    pending = [d for d in days if not cache.has(d)]
    log(f"{season}: {len(days)} dates, {len(pending)} to run (cache {cache.root})")
    if pending:
        if workers > 1 and len(pending) >= 2 * workers:
            _run_parallel(season, pending, settings, workers, log)
        else:
            sr.load()
            _run_days(sr, pending, cache, log)
    pls, lus, skipped, secs = [], [], [], 0.0
    for d in days:
        if not cache.has(d):
            skipped.append({"date": d, "error": "missing"})
            continue
        p, ln, m = cache.load(d)
        if m.get("error"):
            skipped.append({"date": d, "error": m["error"]})
            continue
        pls.append(p)
        lus.append(ln)
        secs += float(m.get("seconds", 0.0))
    players = pd.concat(pls, ignore_index=True) if pls else pd.DataFrame(columns=PLAYER_COLS)
    lineups = pd.concat(lus, ignore_index=True) if lus and any(len(x) for x in lus) \
        else pd.DataFrame()
    meta = {"season": season, "n_dates": len(days), "n_run": len(days) - len(skipped),
            "skipped": skipped, "date_seconds": round(secs, 1),
            "wall_seconds": round(time.time() - t0, 1), "cache": str(cache.root),
            "dates_first_last": [days[0], days[-1]] if days else None,
            "salary_fit": _salary_fit_label(conn, season)}
    if settings.protocol in LIVE_OFFSETS:
        envs: dict[str, int] = {}
        for d in days:
            if cache.has(d):
                for k, v in (cache.load_meta(d).get("env_sources") or {}).items():
                    envs[k] = envs.get(k, 0) + int(v)
        meta["env_sources"] = envs
        meta["pplay"] = settings.pplay or "loo_month"
    return players, lineups, meta


def _salary_fit_label(conn, season) -> str | None:
    if not is_rotoguru(season):
        return None
    try:
        E.fit_salary_line(E.load_rotoguru(conn), season, "all_seasons")
        return "ols_seasons_before_S"
    except ValueError:
        return "fixed_5_per_1k (no earlier RotoGuru season)"


def _run_parallel(season, pending, settings, workers, log) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    chunks = [pending[i::workers] for i in range(workers)]
    sj = json.dumps(asdict(settings))
    args = [(season, c, sj, str(config.data_dir()), f"{season}#{i}")
            for i, c in enumerate(chunks) if c]
    log(f"{season}: {len(pending)} dates on {len(args)} worker processes")
    with ProcessPoolExecutor(max_workers=len(args), mp_context=mp.get_context("spawn")) as ex:
        for _ in ex.map(_season_worker, args):
            pass


# ====================================================================== reports

PIT_BINS = 10
WORK_LINEUP_SINGLES = ("ours_mean", "ours_cash")


def _naive_crps(df: pd.DataFrame, prior: pd.DataFrame | None) -> np.ndarray:
    """CRPS of the naive normal N(fppg, (CV_tier * fppg)^2) with the CV per model tier fit on the
    prior seasons' rows (NaN when there are none: nothing in-sample is ever used)."""
    if prior is None or prior.empty:
        return np.full(len(df), np.nan)
    p = prior[prior["has_proj"] & (prior["exp_min"] >= 10) & prior["b_fppg"].notna()]
    cv = MT.fit_group_cv(p["tier"].to_numpy(), p["b_fppg"].to_numpy(float), p["y"].to_numpy(float))
    return MT.naive_normal_crps(df["tier"].to_numpy(), df["b_fppg"].to_numpy(float),
                                df["y"].to_numpy(float), cv)


def with_naive_crps(players: pd.DataFrame) -> pd.DataFrame:
    """Add ``crps_naive`` (see `_naive_crps`), fit season by season on the seasons before it."""
    out = []
    for s in sorted(players["season"].unique()):
        d = players[players["season"] == s].copy()
        d["crps_naive"] = _naive_crps(d, players[players["season"] < s])
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else players


BASELINE_COLS = {"sal_all": "salary[all_seasons]", "sal_prev": "salary[prev_season]",
                 "sal_played": "salary[played_only]", "sal_fixed": "fixed 5/$1k",
                 "b_season_avg": "season_avg", "b_fppg": "fppg_proxy", "b_last10": "last10"}


def player_report(df: pd.DataFrame) -> dict:
    """Player-level metrics on the unconditional population (every projected, salaried-or-played
    player with expected minutes >= 10, DNP scored 0): MAE vs each baseline on identical rows,
    CRPS (vs the naive normal when present), coverage, PIT deciles, tier tables."""
    out: dict = {"pool_rows": int(len(df)),
                 "share_with_projection": float(df["has_proj"].mean()) if len(df) else None}
    u = df[df["has_proj"].astype(bool) & (df["exp_min"] >= 10)]
    if u.empty:
        return out
    y = u["y"].to_numpy(float)
    day = u["date"].to_numpy()
    e_m = np.abs(u["mean"].to_numpy(float) - y)
    out.update({"n": int(len(u)), "n_dates": int(u["date"].nunique()),
                "dnp_rate": float((u["minutes"] <= 0).mean()), "mae": float(e_m.mean()),
                "bias": float((u["mean"] - u["y"]).mean())})
    vs = {}
    for col, label in BASELINE_COLS.items():
        if col not in u or not u[col].notna().any():
            continue
        m = u[col].notna().to_numpy()
        vs[label] = MT.gain_pct(np.abs(u[col].to_numpy(float)[m] - y[m]), e_m[m], day[m])
    out["vs"] = vs
    sal = {k: v for k, v in vs.items() if k.startswith("salary[")}
    if sal:
        best = min(sal, key=lambda k: sal[k]["mae_base"])
        out["best_salary"] = best
        out["vs_best_salary"] = sal[best]
    out["crps"] = float(u["crps"].mean())
    if "crps_naive" in u and u["crps_naive"].notna().any():
        m = u["crps_naive"].notna().to_numpy()
        g = MT.gain_pct(u["crps_naive"].to_numpy(float)[m], u["crps"].to_numpy(float)[m], day[m])
        out["crps_vs_naive_normal"] = {"gain_pct": g["gain_pct"], "se_pct": g["se_pct"],
                                       "crps_naive": g["mae_base"], "n": g["n"]}
    c80 = MT.rate_se(MT.interval_hits(u["q10"], u["q90"], y), day)
    c50 = MT.rate_se(MT.interval_hits(u["q25"], u["q75"], y), day)
    out.update({"cov80": c80["rate"], "cov80_se": c80["se"], "cov50": c50["rate"],
                "cov50_se": c50["se"]})
    pit = u["pit"].to_numpy(float)
    out["pit_deciles"] = [float(x) for x in MT.pit_histogram(pit, PIT_BINS)]
    out["by_tier"] = MT.tier_table(u, "tier", base="b_fppg").to_dict("records")
    lv = u.assign(level=pd.cut(u["mean"], [-1, 10, 20, 30, 40, 200],
                               labels=["<10", "10-20", "20-30", "30-40", "40+"]))
    out["by_level"] = MT.tier_table(lv, "level", base="b_fppg").to_dict("records")
    out["dfs_pop"] = dfs_pop_report(df)
    return out


VALUE_FORECASTS = {"ours": "mean", "fppg_proxy": "b_fppg", "season_avg": "b_season_avg"}


def dfs_pop_report(df: pd.DataFrame) -> dict:
    """E4 (report only): the DFS-relevant population (salary >= $4k where known, or projected
    >= 15 DK by us or by the FPPG proxy; `metrics.dfs_population`) of the projected rows: MAE,
    gain over the FPPG proxy, CRPS (vs the naive normal when present), and on rows with a
    salary the value-ranking metric corr(forecast - salary-implied, actual - salary-implied)
    per date (`metrics.value_ranking`, SE clustered by date), each forecast vs the FPPG proxy."""
    d = df[df["has_proj"].astype(bool)] if len(df) else df
    if d.empty:
        return {}
    fp = d["b_fppg"].to_numpy(float) if "b_fppg" in d else np.full(len(d), np.nan)
    m = MT.dfs_population(d["salary"].to_numpy(float), d["mean"].to_numpy(float), fp)
    v = d[m]
    if v.empty:
        return {"n": 0}
    y = v["y"].to_numpy(float)
    day = v["date"].to_numpy()
    e_m = np.abs(v["mean"].to_numpy(float) - y)
    out: dict = {"definition": f"salary >= {MT.DFS_SAL_MIN:.0f} or projected (ours or FPPG "
                               f"proxy) >= {MT.DFS_PROJ_MIN:.0f} DK",
                 "n": int(len(v)), "n_dates": int(v["date"].nunique()), "mae": float(e_m.mean()),
                 "bias": float((v["mean"] - v["y"]).mean())}
    if "b_fppg" in v and v["b_fppg"].notna().any():
        k = v["b_fppg"].notna().to_numpy()
        out["vs_fppg_proxy"] = MT.gain_pct(np.abs(v["b_fppg"].to_numpy(float)[k] - y[k]), e_m[k],
                                           day[k])
    c = v["crps"].to_numpy(float)
    if np.isfinite(c).any():
        out["crps"] = float(np.nanmean(c))
        if "crps_naive" in v and v["crps_naive"].notna().any():
            k = (v["crps_naive"].notna() & v["crps"].notna()).to_numpy()
            g = MT.gain_pct(v["crps_naive"].to_numpy(float)[k], c[k], day[k])
            out["crps_vs_naive_normal"] = {"gain_pct": g["gain_pct"], "se_pct": g["se_pct"],
                                           "n": g["n"]}
    if v["salary"].notna().any():
        preds = {k: v[c_].to_numpy(float) for k, c_ in VALUE_FORECASTS.items()
                 if c_ in v and v[c_].notna().any()}
        if "ours" in preds and "fppg_proxy" in preds:
            out["value"] = MT.value_ranking(preds, y, v["salary"].to_numpy(float), day, day,
                                            base="fppg_proxy")
    return out


def lineup_report(ld: pd.DataFrame) -> dict:
    """Lineup-level tables from the long lineup frame (one row per date, method, rank)."""
    if ld is None or ld.empty:
        return {}
    singles = ld[ld["rank"] == 1]
    singles = singles[singles["kind"].isin(["single", "hindsight"])]
    w = singles.pivot(index="date", columns="method", values="actual")
    methods = list(w.columns)
    ok = w.notna().all(axis=1)
    w = w[ok]
    day = w.index.to_numpy()
    hs = w["hindsight"].to_numpy(float)
    pr = singles.pivot(index="date", columns="method", values="pct_rand").loc[w.index]
    ps = singles.pivot(index="date", columns="method", values="pct_spend").loc[w.index]
    out: dict = {"n_dates": int(len(w)), "n_dates_dropped": int((~ok).sum()), "methods": {},
                 "paired": {}}
    for m in methods:
        v = w[m].to_numpy(float)
        pc = MT.paired_cluster(v, day)
        out["methods"][m] = {"mean_actual": pc["mean"], "se": pc["se"],
                             "ratio_to_hindsight": float(v.sum() / hs.sum()),
                             "mean_per_date_ratio": float((v / hs).mean()),
                             "cash_rate_proxy_ge_300": float((v >= 300).mean()),
                             "pct_rand": float(pr[m].mean()), "pct_spend": float(ps[m].mean())}
    bases = [m for m in methods if m not in ("hindsight",) and not m.startswith("ours")]
    for ours in [m for m in methods if m.startswith("ours")]:
        for other in bases:
            out["paired"][f"{ours}_minus_{other}"] = MT.paired_summary(
                w[ours].to_numpy(float), w[other].to_numpy(float), day)
    tn = ld[ld["kind"] == "topn"]
    if not tn.empty:
        out["topn"] = {}
        g = tn.groupby(["method", "date"])
        per = pd.DataFrame({"mean": g["actual"].mean(), "best": g["actual"].max(),
                            "rank1": tn[tn["rank"] == 1].groupby(["method", "date"])["actual"]
                            .first(), "pct_rand": g["pct_rand"].mean(),
                            "pct_spend": g["pct_spend"].mean(), "n": g["actual"].size()}
                           ).reset_index()
        for m, d in per.groupby("method"):
            pc = MT.paired_cluster(d["mean"].to_numpy(float), d["date"].to_numpy())
            out["topn"][m] = {"n_dates": int(len(d)), "mean_lineups": float(d["n"].mean()),
                              "mean_actual": pc["mean"], "se": pc["se"],
                              "best_of_n": float(d["best"].mean()),
                              "rank1": float(d["rank1"].mean()),
                              "pct_rand": float(d["pct_rand"].mean()),
                              "pct_spend": float(d["pct_spend"].mean())}
        out["topn_paired"] = {}
        if "ours_topn" in set(per["method"]):
            a = per[per["method"] == "ours_topn"].set_index("date")
            for m in set(per["method"]) - {"ours_topn"}:
                b = per[per["method"] == m].set_index("date")
                idx = a.index.intersection(b.index)
                out["topn_paired"][f"ours_topn_minus_{m}"] = MT.paired_summary(
                    a.loc[idx, "mean"].to_numpy(float), b.loc[idx, "mean"].to_numpy(float),
                    idx.to_numpy())
    return out


ACCEPT_COV80 = (0.77, 0.83)
ACCEPT_COV50 = (0.47, 0.53)
ACCEPT_TIER_COV80 = (0.74, 0.86)
ACCEPT_PIT = (0.08, 0.12)


def acceptance(player: dict, lineup: dict) -> list[dict]:
    """The DFS-backtest rows of PLAYERS.md §7.3 that this harness can fill (T2.17 adds the
    field-proxy rows). Significant = gain >= 2 SE. Informational: pass/fail from the point
    estimate, nothing here gates anything."""
    rows: list[dict] = []

    def add(name, value, target, ok, extra=""):
        rows.append({"metric": name, "value": value, "target": target, "ok": ok, "note": extra})

    vs = player.get("vs", {}) if player else {}
    cand = {k: v for k, v in vs.items() if k == player.get("best_salary") or k in
            ("fppg_proxy", "last10")}
    if cand:
        k = min(cand, key=lambda k: cand[k]["mae_base"])
        g = cand[k]
        add("DK FP MAE gain vs best of salary-implied / fppg / last10", g["gain_pct"],
            ">= 5% (>= 2 SE)", bool(g["gain_pct"] >= 5 and g["gain_pct"] >= 2 * g["se_pct"]),
            f"best baseline {k}, SE {g['se_pct']:.2f}")
    cn = (player or {}).get("crps_vs_naive_normal")
    if cn:
        add("DK FP CRPS gain vs naive mean + tier-CV normal", cn["gain_pct"], ">= 5% (>= 2 SE)",
            bool(cn["gain_pct"] >= 5 and cn["gain_pct"] >= 2 * cn["se_pct"]),
            f"SE {cn['se_pct']:.2f}")
    if player and "cov80" in player:
        add("80% coverage", player["cov80"], "0.77-0.83",
            ACCEPT_COV80[0] <= player["cov80"] <= ACCEPT_COV80[1])
        add("50% coverage", player["cov50"], "0.47-0.53",
            ACCEPT_COV50[0] <= player["cov50"] <= ACCEPT_COV50[1])
        tiers = {str(r["tier"]): r.get("cov80") for r in player.get("by_tier", [])}
        add("80% coverage by role tier (worst)", min(tiers.values()) if tiers else None,
            "0.74-0.86 each", bool(tiers) and all(
                ACCEPT_TIER_COV80[0] <= v <= ACCEPT_TIER_COV80[1] for v in tiers.values()),
            ", ".join(f"{k} {v:.3f}" for k, v in tiers.items()))
        pit = player["pit_deciles"]
        add("PIT deciles (randomised), largest share", max(pit), "each 0.08-0.12",
            all(ACCEPT_PIT[0] <= x <= ACCEPT_PIT[1] for x in pit),
            " ".join(f"{100 * x:.0f}" for x in pit))
    pr = (lineup or {}).get("paired", {}).get("ours_cash_minus_fppg_proxy_avail")
    if pr:
        add("cash lineup minus fppg-optimizer lineup (paired)", pr["mean_diff"],
            "> 0 at 2 SE, wins >= 55%", bool(pr["z"] is not None and pr["z"] >= 2
                                            and pr["win_rate"] >= 0.55),
            f"SE {pr['se']:.2f}, wins {pr['win_rate']:.3f}")
    me = (lineup or {}).get("methods", {})
    if "ours_cash" in me:
        better = all(me["ours_cash"]["mean_actual"] > me[m]["mean_actual"]
                     for m in me if m in ("fppg_proxy_avail", "salary_avail", "fppg_proxy",
                                          "salary"))
        add("hindsight ratio (cash)", me["ours_cash"]["ratio_to_hindsight"],
            "reported; beats both baseline optimisers", better,
            ", ".join(f"{m} {me[m]['ratio_to_hindsight']:.3f}" for m in
                      ("fppg_proxy_avail", "salary_avail") if m in me))
    return rows


def build_report(players: pd.DataFrame, lineups: pd.DataFrame, metas: dict, settings: Settings,
                 *, seasons: Sequence[str]) -> dict:
    """Per-season and pooled tables from the per-season frames."""
    players = with_naive_crps(players) if not players.empty else players
    rep: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "harness_version": HARNESS_VERSION, "settings": asdict(settings),
                 "seasons": list(seasons), "per_season": {}, "meta": metas}
    for s in seasons:
        ps = players[players["season"] == s] if not players.empty else players
        ls = lineups[lineups["season"] == s] if not lineups.empty else lineups
        rep["per_season"][s] = {"player": player_report(ps) if len(ps) else {},
                                "lineup": lineup_report(ls)}
    rep["pooled"] = {"player": player_report(players) if len(players) else {},
                     "lineup": lineup_report(lineups)}
    rep["pooled"]["acceptance"] = acceptance(rep["pooled"]["player"], rep["pooled"]["lineup"])
    rg = [s for s in seasons if is_rotoguru(s)]
    if rg and not lineups.empty:
        rep["pooled_rotoguru"] = {"lineup": lineup_report(lineups[lineups["season"].isin(rg)]),
                                  "player": player_report(players[players["season"].isin(rg)])}
        rep["pooled_rotoguru"]["acceptance"] = acceptance(rep["pooled_rotoguru"]["player"],
                                                          rep["pooled_rotoguru"]["lineup"])
    return rep


def run(conn: sqlite3.Connection, seasons: Sequence[str] | None = None, *,
        protocol: str = "avail", modes: Sequence[str] = ("cash", "topn"), n_topn: int = 20,
        baselines: Sequence[str] = BASELINES, sims: int = 4_000, n_dates: int = 0,
        n_random: int = 2_000, topn_baselines: Sequence[str] = ("fppg_proxy",),
        workers: int = 1, force: bool = False, leak_probe_dates: int = 0,
        date_lo: str | None = None, date_hi: str | None = None, pplay: str = "",
        tails: bool = False, log: Callable[[str], None] = print) -> dict:
    """The backtest. `seasons` default 2014-15 .. 2020-21 plus 2024-25 (player level only).
    Walk-forward: season S uses a bundle fit on seasons < S and salary lines from RotoGuru
    seasons < S. Cached per date, so a rerun only does what is missing."""
    t0 = time.time()
    seasons = list(seasons) if seasons else list(DEFAULT_SEASONS)
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    for s in seasons:
        if protocol != "avail" and is_rotoguru(s):
            raise ValueError(f"protocol {protocol!r} needs the rescued official reports (2025-26)")
    bad = [m for m in modes if m not in ("cash", "topn")]
    if bad:
        raise ValueError(f"unknown mode(s) {bad}")
    if pplay and pplay not in PPLAY_MODES:
        raise ValueError(f"unknown pplay mode {pplay!r}")
    if protocol in LIVE_OFFSETS:
        pplay = pplay or "loo_month"
    else:
        pplay = ""                   # only the live protocols read P(play) params
    st = Settings(protocol=protocol, n_sims=sims, modes=tuple(modes), n_topn=n_topn,
                  baselines=tuple(baselines), topn_baselines=tuple(topn_baselines),
                  n_random=n_random, pplay=pplay, tails=bool(tails))
    pls, lus, metas = [], [], {}
    for s in seasons:
        p, ln, m = run_season(conn, s, st, n_dates=n_dates, workers=workers, force=force, log=log,
                              date_lo=date_lo, date_hi=date_hi)
        metas[s] = m
        if len(p):
            pls.append(p)
        if len(ln):
            lus.append(ln.assign(season=s))
    players = pd.concat(pls, ignore_index=True) if pls else pd.DataFrame(columns=PLAYER_COLS)
    lineups = pd.concat(lus, ignore_index=True) if lus else pd.DataFrame()
    rep = build_report(players, lineups, metas, st, seasons=seasons)
    if leak_probe_dates:
        rep["leak_probe"] = {}
        for s in seasons:
            days = SeasonRun(conn, s, st, log=log, need_context=False).dates(leak_probe_dates)
            rep["leak_probe"][s] = leak_probe(conn, s, days, n_sims=min(sims, 2_000), log=log)
    rep["seconds"] = round(time.time() - t0, 1)
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # KB on Linux, bytes on macOS
    rep["peak_rss_mb"] = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return rep


# ====================================================================== live-like slice (T2.4)
#
# Metric definitions follow the live grader (`dfs/grade.player_snapshot_rows`) and the monitor
# (`mlops/dfs_monitor`), so the slice is the expected range of exactly what is graded at night:
#   U  (DK points, unconditional): every projected player with E[min] >= 10, DNP scored 0
#      dk_mae |mean - y|, dk_bias, dk_crps (sims), cov80 / cov50 (q10-q90 / q25-q75),
#      min_mae |E[min] - minutes| (the grader's 'min_mae')
#   M  (minutes when playing): played and E[min | plays] >= 15: min_mae_played15, min_bias_played15
#      (the monitor's dfs:live_min_mae population)
#   S  starters: played and E[min | plays] >= 26 (the monitor's dfs:min_bias population)
#   Q  report-uncertain: 0.015 < P(available from the report) < 0.985 (D/Q/P and most Available
#      rows; no status needed, so the same rule can run on live rows): pplay_cal_err =
#      mean P(play) - share played
# SEs are two-way clustered by day and player (Cameron-Gelbach-Miller), day clusters nest games.

UNCERTAIN_PAVAIL = (0.015, 0.985)
STATUS_ORDER = ("out", "doubtful", "questionable", "probable", "available", "removed",
                "daytoday", "unlisted", "unlisted_noreport", "")
STATUS_LABEL = {"out": "Out", "doubtful": "Doubtful (D)", "questionable": "Questionable (Q)",
                "probable": "Probable (P)", "available": "Available", "removed": "removed from report",
                "daytoday": "Day-to-day / GTD (ESPN)", "unlisted": "not on report (team filed)",
                "unlisted_noreport": "no report on file", "": "not a report candidate"}
SLICE_METRICS = ("dk_mae", "dk_crps", "cov80", "cov50", "dk_bias", "min_mae",
                 "min_mae_played15", "min_bias_played15", "min_bias_starters", "pplay_cal_err")
DAILY_METRICS = ("dk_mae", "dk_crps", "cov80", "min_mae", "min_mae_played15", "pplay_cal_err")
PCTS = (2.5, 5, 10, 25, 50, 75, 90, 95, 97.5)
WINDOW_N = 300                     # the monitor's "last 300 rows"


def slice_rows(players: pd.DataFrame) -> pd.DataFrame:
    """Per-row metric contributions of a box-path players frame: the populations U / M / S / Q
    as boolean columns and the per-row error columns (see the section comment)."""
    d = players[players["has_proj"].astype(bool)].copy().reset_index(drop=True)
    y = d["y"].to_numpy(float)
    mins = d["minutes"].to_numpy(float)
    played = mins > 0
    mp = d["min_played"].to_numpy(float) if "min_played" in d else np.full(len(d), np.nan)
    pa = d["p_avail"].to_numpy(float) if "p_avail" in d else np.full(len(d), np.nan)
    d["played"] = played.astype(float)
    d["U"] = d["exp_min"].to_numpy(float) >= 10
    # E4: DFS-relevant rows (salary >= $4k where known, else projected >= 15 DK); report only
    d["V"] = MT.dfs_population(d["salary"].to_numpy(float) if "salary" in d
                               else np.full(len(d), np.nan), d["mean"].to_numpy(float))
    d["M"] = played & (mp >= 15)
    d["S"] = played & (mp >= 26)
    d["Q"] = (pa > UNCERTAIN_PAVAIL[0]) & (pa < UNCERTAIN_PAVAIL[1])
    d["dk_mae"] = np.abs(d["mean"].to_numpy(float) - y)
    d["dk_bias"] = d["mean"].to_numpy(float) - y
    d["dk_crps"] = d["crps"].to_numpy(float)
    d["cov80"] = MT.interval_hits(d["q10"], d["q90"], y).astype(float)
    d["cov50"] = MT.interval_hits(d["q25"], d["q75"], y).astype(float)
    d["min_mae"] = np.abs(d["exp_min"].to_numpy(float) - mins)
    d["min_mae_played15"] = np.abs(mp - mins)
    d["min_bias_played15"] = mp - mins
    d["min_bias_starters"] = mp - mins
    d["pplay_cal_err"] = d["p_play"].to_numpy(float) - played
    d["dk_mae_dfs"] = d["dk_mae"]
    d["dk_crps_dfs"] = d["dk_crps"]
    d["cov80_dfs"] = d["cov80"]
    if "status" in d:
        d["status"] = d["status"].fillna("").astype(str)
    else:
        d["status"] = ""
    return d


METRIC_POP = {"dk_mae": "U", "dk_crps": "U", "cov80": "U", "cov50": "U", "dk_bias": "U",
              "min_mae": "U", "min_mae_played15": "M", "min_bias_played15": "M",
              "min_bias_starters": "S", "pplay_cal_err": "Q",
              "dk_mae_dfs": "V", "dk_crps_dfs": "V", "cov80_dfs": "V"}
DFS_SLICE_METRICS = ("dk_mae_dfs", "dk_crps_dfs", "cov80_dfs")   # E4, reported in rep["dfs_pop"]


def _cl_var(z: np.ndarray, cl: np.ndarray) -> float:
    s = pd.Series(z).groupby(cl).sum().to_numpy()
    g = len(s)
    return float(g / (g - 1) * (s ** 2).sum()) if g > 1 else float("nan")


def _two_way_var(z: np.ndarray, day: np.ndarray, player: np.ndarray) -> float:
    """Var of sum(z) two-way clustered by day and player (CGM: V_day + V_player - V_both)."""
    both = pd.Series(day).astype(str) + "|" + pd.Series(player).astype(str)
    v = _cl_var(z, day) + _cl_var(z, player) - _cl_var(z, both.to_numpy())
    return v if v > 0 else _cl_var(z, day)


def _pop(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = d[d[METRIC_POP[metric]]]
    return x[np.isfinite(x[metric].to_numpy(float))]


def metric_se(d: pd.DataFrame, metric: str) -> dict:
    """Mean of `metric` over its population with the two-way clustered SE."""
    x = _pop(d, metric)
    if x.empty:
        return {"value": None, "se": None, "n": 0}
    v = x[metric].to_numpy(float)
    m = float(v.mean())
    var = _two_way_var((v - m) / len(v), x["date"].to_numpy(), x["player_id"].to_numpy())
    return {"value": m, "se": float(np.sqrt(var)) if np.isfinite(var) else None, "n": int(len(v))}


def metric_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """mean_a - mean_b of `metric` (each over its own population; the populations may differ,
    e.g. live vs avail) with the delta-method SE two-way clustered by day and player over the
    stacked rows of both frames (same days and players -> the covariance is counted)."""
    xa, xb = _pop(a, metric), _pop(b, metric)
    if xa.empty or xb.empty:
        return {"diff": None, "se": None, "z": None}
    va, vb = xa[metric].to_numpy(float), xb[metric].to_numpy(float)
    ma, mb = float(va.mean()), float(vb.mean())
    z = np.concatenate([(va - ma) / len(va), -(vb - mb) / len(vb)])
    day = np.concatenate([xa["date"].to_numpy(), xb["date"].to_numpy()])
    pl = np.concatenate([xa["player_id"].to_numpy(), xb["player_id"].to_numpy()])
    se = float(np.sqrt(_two_way_var(z, day, pl)))
    return {"diff": ma - mb, "se": se, "z": (ma - mb) / se if se > 0 else None,
            "n_a": int(len(va)), "n_b": int(len(vb))}


def pplay_by_status(d: pd.DataFrame) -> list[dict]:
    """P(play) calibration per report status at the run (every projected candidate, no minutes
    filter): mean projected P(play), the report's P(available), the share that played, the gap
    with its SE (clustered by day and player) and the Brier score."""
    out = []
    for st in STATUS_ORDER:
        x = d[d["status"] == st]
        if x.empty:
            continue
        p = x["p_play"].to_numpy(float)
        y = x["played"].to_numpy(float)
        e = p - y
        var = _two_way_var((e - e.mean()) / len(e), x["date"].to_numpy(), x["player_id"].to_numpy())
        pa = x["p_avail"].to_numpy(float) if "p_avail" in x else np.full(len(x), np.nan)
        out.append({"status": st, "label": STATUS_LABEL.get(st, st), "n": int(len(x)),
                    "p_play": float(p.mean()), "p_avail": float(np.nanmean(pa))
                    if np.isfinite(pa).any() else None,
                    "played": float(y.mean()), "gap": float(e.mean()),
                    "gap_se": float(np.sqrt(var)) if np.isfinite(var) else None,
                    "brier": float((e ** 2).mean())})
    return out


def tier_bias(d: pd.DataFrame) -> list[dict]:
    """DK bias / MAE (population U) and minutes bias (population M) per projection tier."""
    out = []
    for t in sorted(d["tier"].dropna().unique()):
        x = d[d["tier"] == t]
        row = {"tier": str(t)}
        for m in ("dk_bias", "dk_mae", "cov80", "min_bias_played15", "min_mae_played15"):
            row[m] = metric_se(x, m)
        out.append(row)
    return out


def daily_values(d: pd.DataFrame, metrics=DAILY_METRICS) -> pd.DataFrame:
    """One row per date: each metric over that night's population (+ n per metric)."""
    rows = []
    for day, x in d.groupby("date"):
        r = {"date": day}
        for m in metrics:
            p = _pop(x, m)
            r[m] = float(p[m].mean()) if len(p) else np.nan
            r[f"n_{m}"] = int(len(p))
        rows.append(r)
    return pd.DataFrame(rows)


def window_values(d: pd.DataFrame, n: int = WINDOW_N) -> pd.DataFrame:
    """The monitor's rolling 'last n rows' metrics as of the end of every date: cov80 (U),
    minutes MAE (M) and starter minutes bias (S), newest rows first."""
    d = d.sort_values(["date", "game_id", "player_id"])
    days = sorted(d["date"].unique())
    rows = []
    for day in days:
        upto = d[d["date"] <= day]
        r = {"date": day}
        for m, pop in (("cov80", "U"), ("min_mae_played15", "M"), ("min_bias_starters", "S")):
            x = upto[upto[pop]]
            x = x[np.isfinite(x[m].to_numpy(float))].iloc[::-1].head(n)
            r[m] = float(x[m].mean()) if len(x) >= n else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def ranges(values: pd.DataFrame, metrics, n_cols: bool = True) -> dict:
    """Percentiles (PCTS) of each metric's per-date values + mean, sd, number of dates and the
    median rows per date (the n the band refers to)."""
    out = {}
    for m in metrics:
        if m not in values:
            continue
        v = values[m].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        r = {f"p{str(p).replace('.', '_')}": float(np.percentile(v, p)) for p in PCTS}
        r.update({"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                  "n_days": int(len(v))})
        if n_cols and f"n_{m}" in values:
            nn = values[f"n_{m}"].to_numpy(float)
            r["rows_median"] = float(np.median(nn[nn > 0])) if (nn > 0).any() else None
        out[m] = r
    return out


def slice_report(frames: dict, *, reference: str = "avail", primary: str = "live",
                 base_arm: str | None = None) -> dict:
    """The T2.4 tables from {(arm, protocol): players frame}: per arm and protocol every
    SLICE_METRICS value with its SE, the loss vs `reference` (same arm, same dates), each arm
    vs `base_arm` within a protocol, tier bias, P(play) by status, and the daily / rolling-window
    ranges of every (arm, protocol)."""
    rows = {k: slice_rows(v) for k, v in frames.items()}
    arms = sorted({a for a, _ in rows})
    base_arm = base_arm or arms[0]
    days = {k: set(v["date"]) for k, v in rows.items()}
    common = set.intersection(*days.values()) if days else set()
    rows = {k: v[v["date"].isin(common)].reset_index(drop=True) for k, v in rows.items()}
    rep: dict = {"dates": sorted(common), "n_dates": len(common), "arms": arms,
                 "protocols": sorted({p for _, p in rows}), "base_arm": base_arm,
                 "reference": reference, "primary": primary, "cells": {}, "vs_reference": {},
                 "vs_base_arm": {}, "tiers": {}, "pplay_status": {}, "daily": {}, "window": {},
                 "dfs_pop": {}, "dfs_pop_vs_base_arm": {}}
    for (arm, proto), d in rows.items():
        k = f"{arm}|{proto}"
        rep["cells"][k] = {m: metric_se(d, m) for m in SLICE_METRICS}
        rep["tiers"][k] = tier_bias(d)
        rep["dfs_pop"][k] = {m: metric_se(d, m) for m in DFS_SLICE_METRICS}
        if arm != base_arm and (base_arm, proto) in rows:
            rep["dfs_pop_vs_base_arm"][k] = {m: metric_diff(d, rows[(base_arm, proto)], m)
                                             for m in DFS_SLICE_METRICS}
        if d["status"].ne("").any():
            rep["pplay_status"][k] = pplay_by_status(d)
        dv = daily_values(d)
        rep["daily"][k] = ranges(dv, DAILY_METRICS)
        rep["window"][k] = ranges(window_values(d), ("cov80", "min_mae_played15",
                                                    "min_bias_starters"), n_cols=False)
        if proto != reference and (arm, reference) in rows:
            rep["vs_reference"][k] = {m: metric_diff(d, rows[(arm, reference)], m)
                                      for m in SLICE_METRICS if m != "pplay_cal_err"}
        if arm != base_arm and (base_arm, proto) in rows:
            rep["vs_base_arm"][k] = {m: metric_diff(d, rows[(base_arm, proto)], m)
                                     for m in SLICE_METRICS}
    return rep


def live_baseline_json(rep: dict, *, arm: str, protocol: str = "live", note: str = "") -> dict:
    """The monitor's thresholds file (`<data>/proj/live_like_baseline.json`, shipped default
    `mlops/live_like_baseline.json`) from a slice report: `min_mae` (dfs:live_min_mae) of the
    live arm, plus the nightly and last-300-row ranges of every arm in `per_model`."""
    from . import pipeline as P

    def block(a):
        k = f"{a}|{protocol}"
        c = rep["cells"][k]
        return {"min_mae": c["min_mae_played15"]["value"],
                "means": {m: c[m]["value"] for m in SLICE_METRICS},
                "daily": rep["daily"][k], "window300": rep["window"][k]}

    main = block(arm)
    out = {"min_mae": main["min_mae"], "model": arm, "protocol": protocol,
           "slice": [rep["dates"][0], rep["dates"][-1]] if rep["dates"] else None,
           "n_dates": rep["n_dates"], "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "source": "T2.4 live-like slice (docs/PLAYERS.md); proj/backtest.py "
                     "slice_report / live_baseline_json",
           "definitions": {
               "dk_mae": "|proj_dk - actual_dk|, proj_min >= 10, DNP = 0 (grader 'dk_mae')",
               "cov80": "actual DK inside the stored q10..q90, same rows (grader 'dk_cov80')",
               "min_mae_played15": "|E[min | plays] - actual min|, played, E[min | plays] >= 15",
               "min_mae": "|E[min] - actual min|, proj_min >= 10 (grader 'min_mae')",
               "pplay_cal_err": "mean P(play) - share played over rows with "
                                f"{UNCERTAIN_PAVAIL[0]} < p_avail < {UNCERTAIN_PAVAIL[1]}",
               "daily": "percentiles of the per-night values over the slice; rows_median = the "
                        "median rows per night the percentiles refer to",
               "window300": "percentiles of the monitor's rolling last-300-row values"},
           "daily": main["daily"], "window300": main["window300"], "means": main["means"],
           "per_model": {a: block(a) for a in rep["arms"] if f"{a}|{protocol}" in rep["cells"]},
           "seed_version": P.SEED_VERSION}
    if note:
        out["note"] = note
    return out


def format_slice_report(rep: dict) -> str:
    def f(c, nd=3):
        if not c or c.get("value") is None:
            return "-"
        se = c.get("se")
        return f"{c['value']:.{nd}f}" + (f" ({se:.{nd}f})" if se is not None else "")

    def g(c, nd=3):
        if not c or c.get("diff") is None:
            return "-"
        return f"{c['diff']:+.{nd}f} ({c['se']:.{nd}f})"

    L = [f"T2.4 live-like slice: {rep['n_dates']} dates "
         f"{rep['dates'][0] if rep['dates'] else '-'} .. {rep['dates'][-1] if rep['dates'] else '-'}"
         f"; SEs two-way clustered by day and player", ""]
    hdr = f"  {'arm|protocol':<26}" + "".join(f"{m:>18}" for m in
                                              ("min_mae_played15", "dk_mae", "dk_crps", "cov80",
                                               "dk_bias", "min_mae"))
    L.append(hdr)
    for k, c in rep["cells"].items():
        L.append(f"  {k:<26}" + "".join(f"{f(c[m]):>18}" for m in
                                        ("min_mae_played15", "dk_mae", "dk_crps", "cov80",
                                         "dk_bias", "min_mae")))
    L.append("")
    L.append(f"  loss vs {rep['reference']} (same arm, same dates): value - reference (SE)")
    for k, c in rep["vs_reference"].items():
        L.append(f"  {k:<26}" + "".join(f"{g(c[m]):>18}" for m in
                                        ("min_mae_played15", "dk_mae", "dk_crps", "cov80",
                                         "dk_bias", "min_mae")))
    if rep["vs_base_arm"]:
        L.append("")
        L.append(f"  arm vs {rep['base_arm']} (same protocol): value - base (SE)")
        for k, c in rep["vs_base_arm"].items():
            L.append(f"  {k:<26}" + "".join(f"{g(c[m]):>18}" for m in
                                            ("min_mae_played15", "dk_mae", "dk_crps", "cov80",
                                             "dk_bias", "min_mae")))
    if rep.get("dfs_pop"):
        L.append("")
        L.append("  E4 DFS-relevant rows (salary >= $4k where known, else projected >= 15 DK):"
                 " DK MAE / CRPS / cov80 (SE)")
        for k, c in rep["dfs_pop"].items():
            L.append(f"  {k:<26}" + "".join(f"{f(c[m]):>18}" for m in DFS_SLICE_METRICS))
        for k, c in (rep.get("dfs_pop_vs_base_arm") or {}).items():
            L.append(f"  {k + ' - base':<26}" + "".join(f"{g(c[m]):>18}"
                                                         for m in DFS_SLICE_METRICS))
    for k, t in rep["tiers"].items():
        L.append("")
        L.append(f"  bias by tier, {k}: DK bias / DK MAE / cov80 (U); minutes bias / MAE (M)")
        for r in t:
            L.append(f"    {r['tier']:<6} n={r['dk_bias']['n']:<6} " + "  ".join(
                f"{m} {f(r[m], 2)}" for m in ("dk_bias", "dk_mae", "cov80",
                                              "min_bias_played15", "min_mae_played15")))
    for k, t in rep["pplay_status"].items():
        L.append("")
        L.append(f"  P(play) by report status at the run, {k}")
        for r in t:
            L.append(f"    {r['label']:<30} n={r['n']:<6} P(play) {r['p_play']:.3f}  "
                     f"report P {'-' if r['p_avail'] is None else format(r['p_avail'], '.3f')}  "
                     f"played {r['played']:.3f}  gap {r['gap']:+.3f} "
                     f"({'-' if r['gap_se'] is None else format(r['gap_se'], '.3f')})  "
                     f"Brier {r['brier']:.3f}")
    for k, dd in rep["daily"].items():
        L.append("")
        L.append(f"  nightly ranges, {k}: p10 / p50 / p90 (rows per night, median)")
        for m, r in dd.items():
            L.append(f"    {m:<18} {r['p10']:.3f} / {r['p50']:.3f} / {r['p90']:.3f}  "
                     f"(n_days {r['n_days']}, rows {r.get('rows_median')})")
    return "\n".join(L)


# ====================================================================== leak probe

def cut_context(ctx, day):
    """A copy of the projection context with every frame physically truncated to games strictly
    before `day` (history, rate-state games; pair shares rebuilt from the truncated history)."""
    from . import minutes as M
    from . import pipeline as P
    d = P._as_date(day)
    c = copy.copy(ctx)
    c.h = M.truncate(ctx.h, d)
    c.games = ctx.games[ctx.games["day"] < P._day_int(d)].reset_index(drop=True)
    c._shares = None
    return c


def leak_probe_day(conn: sqlite3.Connection, ctx, bundle, day, *, n_sims: int = 1_500) -> dict:
    """Project `day` twice with the same seed:

    full  the normal avail path (oracle panel built from the day's own box score);
    cut   every history frame truncated before `day`; the ONLY thing carried over from the day
          is the availability oracle (which candidates exist and who is absent: L4 'known
          availability'). The cut panel has no actual minutes, starters, margin or overtime.

    Returns the mean absolute relative difference of the projected DK mean over players with a
    full-path mean >= 1 (0 when nothing about the day itself leaks beyond the oracle)."""
    from . import minutes as M
    from . import pipeline as P
    d = P._as_date(day)
    seed = P.stable_seed(d.isoformat(), "hist-avail", P.MODEL_VERSION)
    games, panel, p_avail = P.historical_inputs(conn, ctx, d, protocol="avail")
    env = {g: P.environment(conn, g, None, mode="closing") for g in games["game_id"]}
    full = P.run_games(P.build_games(ctx, bundle, conn, d, games, panel, p_avail, env),
                       bundle, n_sims, seed, d)
    cctx = cut_context(ctx, d)
    tg = P._targets(games, d)
    # the season label is a schedule fact; `season_of` mislabels the Aug-Oct 2020 bubble
    # (2019-20 games dated after Aug 1), so take the label the history carries for the game
    lab = ctx.h.tg.set_index(["game_id", "team"])["season"]
    tg["season"] = [lab.get((g, t), s) for g, t, s in zip(tg["game_id"], tg["team"], tg["season"])]
    oracle = panel[["key", "player_id", "absent"]]
    p0 = M.pregame_panel(cctx.h, tg, extra=oracle[["key", "player_id"]])
    p0 = oracle.merge(p0.drop(columns=[c for c in ("absent",) if c in p0]),
                      on=["key", "player_id"], how="left")          # oracle candidate order
    p0 = p0.merge(tg[["key", "game_id"]], on="key", how="left")
    cut = P.run_games(P.build_games(cctx, bundle, conn, d, games, p0, {}, env),
                      bundle, n_sims, seed, d)
    a = full.players.set_index(["game_id", "player_id"])["dk_mean"]
    b = cut.players.set_index(["game_id", "player_id"])["dk_mean"]
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)
    m = a >= 1.0
    rel = np.abs(a[m] - b[m]) / a[m]
    return {"date": d.isoformat(), "n_players": int(m.sum()),
            "n_players_full": int(len(full.players)), "n_players_cut": int(len(cut.players)),
            "mean_abs_rel_diff": float(rel.mean()) if len(rel) else float("nan"),
            "max_abs_rel_diff": float(rel.max()) if len(rel) else float("nan"),
            "max_abs_diff": float(np.abs(a - b).max()) if len(a) else float("nan")}


def leak_probe(conn: sqlite3.Connection, season: str, days: Sequence, *, n_sims: int = 1_500,
               log: Callable[[str], None] = print) -> dict:
    """`leak_probe_day` on each date (one Context and bundle per season); passes when the mean
    absolute relative change over dates is below 0.5%."""
    from . import pipeline as P
    ctx = P.Context(conn, season, log=log)
    bundle = P.load_bundle(conn, season, ctx, log=log)
    rows = []
    for day in days:
        r = leak_probe_day(conn, ctx, bundle, day, n_sims=n_sims)
        rows.append(r)
        log(f"  leak probe {season} {day}: mean abs rel diff {r['mean_abs_rel_diff']:.5f} "
            f"(max {r['max_abs_rel_diff']:.5f})")
    mean = float(np.nanmean([r["mean_abs_rel_diff"] for r in rows])) if rows else float("nan")
    return {"season": season, "n_dates": len(rows), "mean_abs_rel_diff": mean,
            "max_abs_rel_diff": max((r["max_abs_rel_diff"] for r in rows), default=float("nan")),
            "tolerance": LEAK_TOL, "pass": bool(rows and mean < LEAK_TOL), "dates": rows}


# ====================================================================== text report

def _pct(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _fmt_player(p: dict) -> list[str]:
    if not p or "n" not in p:
        return ["  (no projected players)"]
    L = [f"  players (exp min >= 10, DNP = 0): n={p['n']} ({p['n_dates']} dates) dnp "
         f"{p['dnp_rate']:.3f}  MAE {p['mae']:.3f}  bias {p['bias']:+.3f}  pool rows "
         f"{p['pool_rows']} (projected {_pct(p['share_with_projection'])})"]
    for k, g in p["vs"].items():
        star = " <- best salary" if k == p.get("best_salary") else ""
        L.append(f"    vs {k:<20} MAE {g['mae_base']:.3f} -> ours {g['gain_pct']:+.2f}% "
                 f"(SE {g['se_pct']:.2f}, n={g['n']}){star}")
    cn = p.get("crps_vs_naive_normal")
    L.append(f"    CRPS {p['crps']:.3f}" + (f"  vs naive normal {cn['crps_naive']:.3f}: "
             f"{cn['gain_pct']:+.2f}% (SE {cn['se_pct']:.2f})" if cn else
             "  (naive-normal reference needs an earlier season in the run)"))
    L.append(f"    cov80 {p['cov80']:.3f} (SE {p['cov80_se']:.3f})  cov50 {p['cov50']:.3f} "
             f"(SE {p['cov50_se']:.3f})  PIT deciles " + " ".join(f"{100 * x:.0f}" for x in
                                                                  p["pit_deciles"]))
    for r in p["by_tier"]:
        L.append(f"    tier {str(r['tier']):<6} n={r['n']:<6} pred {r['pred']:6.2f} actual "
                 f"{r['actual']:6.2f} bias {r['bias']:+.2f} MAE {r['mae']:.2f}"
                 + (f" (fppg {r['mae_base']:.2f})" if "mae_base" in r else "")
                 + f" cov80 {r.get('cov80', float('nan')):.3f} cov50 {r.get('cov50', float('nan')):.3f}")
    L += _fmt_dfs_pop(p.get("dfs_pop") or {})
    return L


def _fmt_dfs_pop(d: dict) -> list[str]:
    """E4 lines of `dfs_pop_report` (empty when absent)."""
    if not d.get("n"):
        return []
    L = [f"    E4 DFS rows ({d['definition']}): n={d['n']}  MAE {d['mae']:.3f}  bias "
         f"{d['bias']:+.3f}" + (f"  CRPS {d['crps']:.3f}" if "crps" in d else "")]
    g = d.get("vs_fppg_proxy")
    if g:
        L.append(f"      vs fppg_proxy MAE {g['mae_base']:.3f} -> ours {g['gain_pct']:+.2f}% "
                 f"(SE {g['se_pct']:.2f})")
    v = d.get("value")
    if v:
        fc = v["forecasts"]
        L.append("      value corr (forecast - salary-implied vs actual - salary-implied), SE by "
                 "date: " + "  ".join(f"{k} {x['r']:.3f} ({x['se']:.3f})" for k, x in fc.items()))
        for k, x in v["vs_base"].items():
            L.append(f"      {k} - {x['base']}: {x['diff']:+.3f} (SE {x['se']:.3f}), n={x['n']}")
    return L


def _fmt_lineup(lu: dict) -> list[str]:
    if not lu:
        return []
    L = [f"  lineups: {lu['n_dates']} dates (dropped {lu['n_dates_dropped']}); mean ACTUAL DK points",
         f"    {'method':<22} {'mean':>7} {'SE':>5} {'/hind':>6} {'date-ratio':>10} "
         f"{'>=300':>6} {'pct_rand':>8} {'pct_spend':>9}"]
    for m, v in lu["methods"].items():
        L.append(f"    {m:<22} {v['mean_actual']:7.2f} {v['se']:5.2f} {v['ratio_to_hindsight']:6.3f} "
                 f"{v['mean_per_date_ratio']:10.3f} {v['cash_rate_proxy_ge_300']:6.3f} "
                 f"{v['pct_rand']:8.3f} {v['pct_spend']:9.3f}")
    shown = [k for k in lu["paired"] if k.startswith("ours_cash_minus_")
             or k.rsplit("_minus_", 1)[1] in ("salary", "salary_avail", "fppg_proxy_avail")]
    for k in shown:
        v = lu["paired"][k]
        z = "-" if v["z"] is None else f"{v['z']:+.1f}"
        L.append(f"    {k:<42} {v['mean_diff']:+7.2f} (SE {v['se']:.2f}, z {z}) wins "
                 f"{v['win_rate']:.3f} ties {v['tie_rate']:.3f}")
    for m, v in lu.get("topn", {}).items():
        L.append(f"    top-N {m:<22} mean {v['mean_actual']:7.2f} (SE {v['se']:.2f}) best-of-N "
                 f"{v['best_of_n']:7.2f} rank1 {v['rank1']:7.2f} pct_rand {v['pct_rand']:.3f} "
                 f"pct_spend {v['pct_spend']:.3f} ({v['mean_lineups']:.1f} lineups/date)")
    for k, v in lu.get("topn_paired", {}).items():
        z = "-" if v["z"] is None else f"{v['z']:+.1f}"
        L.append(f"    {k:<42} {v['mean_diff']:+7.2f} (SE {v['se']:.2f}, z {z}) wins "
                 f"{v['win_rate']:.3f}")
    return L


def format_report(rep: dict) -> str:
    st = rep["settings"]
    L = [f"DFS backtest {rep['generated']}  (harness v{rep['harness_version']}, protocol "
         f"{st['protocol']}, {st['n_sims']} sims, Top-{st['n_topn']}, {rep['seconds']:.0f}s, peak "
         f"RSS {rep.get('peak_rss_mb', 0):.0f} MB)", ""]
    for s in rep["seasons"]:
        m = rep["meta"].get(s, {})
        r = rep["per_season"][s]
        L.append(f"== {s} (fit < {s}, {st['protocol']}; {m.get('n_run', 0)}/{m.get('n_dates', 0)} "
                 f"dates {m.get('dates_first_last') or '-'}; compute {m.get('date_seconds', 0):.0f}s)")
        if m.get("salary_fit"):
            L.append(f"  salary lines: {m['salary_fit']}")
        if m.get("skipped"):
            L.append(f"  skipped: {len(m['skipped'])} e.g. {m['skipped'][:2]}")
        L += _fmt_player(r["player"]) + _fmt_lineup(r["lineup"]) + [""]
    L.append("== POOLED (all seasons run)")
    L += _fmt_player(rep["pooled"]["player"]) + _fmt_lineup(rep["pooled"]["lineup"]) + [""]
    if "pooled_rotoguru" in rep and any(not is_rotoguru(s) for s in rep["seasons"]):
        L.append("== POOLED RotoGuru seasons only")
        L += _fmt_lineup(rep["pooled_rotoguru"]["lineup"]) + [""]
    acc = rep.get("pooled_rotoguru", rep["pooled"]).get("acceptance") or []
    if acc:
        L.append("== PLAYERS.md section 7.3 rows this harness fills (informational; pooled RotoGuru)")
        for a in acc:
            v = a["value"]
            L.append(f"  [{'ok' if a['ok'] else '--'}] {a['metric']:<58} "
                     f"{'-' if v is None else f'{v:.3f}':>8}  target {a['target']}"
                     + (f"  ({a['note']})" if a["note"] else ""))
        L.append("")
    for s, lp in rep.get("leak_probe", {}).items():
        L.append(f"leak probe {s}: {'PASS' if lp['pass'] else 'FAIL'} mean abs rel diff "
                 f"{lp['mean_abs_rel_diff']:.5f} (max {lp['max_abs_rel_diff']:.5f}) over "
                 f"{lp['n_dates']} dates, tolerance {lp['tolerance']}")
    L += ["", "NOTES",
          "- avail protocol = known availability (a perfect lock-10): minutes and DK MAE are "
          "optimistic (PLAYERS.md 7.6); eligibility is single-position pools (pessimistic).",
          "- baselines use only data before the date: salary lines are OLS on RotoGuru rows of "
          "seasons < S (2014-15 has none: fixed 5/$1k); season_avg / fppg_proxy / last10 are "
          "rebuilt from box scores. '<b>_avail' lineups are restricted to players our "
          "availability oracle keeps (the like-for-like comparison); '<b>' lineups can pick DNPs.",
          "- hindsight = optimal lineup on the actual RotoGuru DK points (same pool, same rules).",
          "- pct_rand: percentile of the lineup among 2,000 random legal lineups of the pool "
          "(uniform picks); pct_spend: salary-weighted picks using >= 96% of the cap. Random "
          "lineups are a floor reference, not the contest field (T2.17).",
          "- closing lines are the market input in every backtest (as in eval-lite)."]
    return "\n".join(L)


def write_reports(rep: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (config.data_dir() / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    jp, tp = out_dir / f"dfsbt_{stamp}.json", out_dir / f"dfsbt_{stamp}.txt"
    jp.write_text(json.dumps(E._clean(rep), indent=1))
    tp.write_text(format_report(rep) + "\n")
    return jp, tp


# ====================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("backtest-dfs", help="RotoGuru DFS backtest harness (walk-forward, "
                       "baselines, hindsight, random-lineup percentile, leak probe)")
    p.add_argument("--seasons", nargs="+", default=None,
                   help=f"default: {' '.join(DEFAULT_SEASONS)} (<= 2020-21 = RotoGuru lineups)")
    p.add_argument("--dates", type=int, default=0, help="evenly spaced dates per season (0 = all)")
    p.add_argument("--protocol", choices=PROTOCOLS, default="avail",
                   help="avail = known availability; reports = official report at the first "
                        "tip - 45; live / live10 = the live lock-45 / lock-10 path per game "
                        "(T2.4, 2025-26 only)")
    p.add_argument("--from", dest="date_lo", default=None, help="first date (ISO)")
    p.add_argument("--to", dest="date_hi", default=None, help="last date (ISO)")
    p.add_argument("--slice", action="store_true",
                   help=f"the T2.4 live-like slice {SLICE[0]} .. {SLICE[1]} (sets --from/--to)")
    p.add_argument("--pplay", choices=PPLAY_MODES, default="",
                   help="live protocols: P(play) params (default loo_month = fit without the "
                        "date's month; shipped = in-sample on the slice)")
    p.add_argument("--sims", type=int, default=4_000)
    p.add_argument("--topn", type=int, default=20, help="N for the Top-N lineups")
    p.add_argument("--modes", nargs="+", default=["cash", "topn"], choices=("cash", "topn"))
    p.add_argument("--random", type=int, default=2_000, help="random lineups per date")
    p.add_argument("--workers", type=int, default=1, help="processes per season (keep <= 4)")
    p.add_argument("--leak-probe", type=int, default=0, metavar="N",
                   help="also run the leak probe on N dates per season")
    p.add_argument("--force", action="store_true", help="ignore and overwrite the date cache")
    p.add_argument("--tails", action="store_true",
                   help="also store per-player sim q95/q99 and P(DK >= k x salary/1000), k = "
                        f"{'/'.join(map(str, TAIL_KS))} (own cache key; off = unchanged)")
    p.add_argument("--out", default=None, help="report directory (default var/reports)")
    return {"backtest-dfs": _cmd}


def _cmd(args) -> int:
    from .. import db
    conn = db.connect()
    lo, hi = (SLICE if args.slice else (None, None))
    rep = run(conn, args.seasons, protocol=args.protocol, modes=args.modes, n_topn=args.topn,
              sims=args.sims, n_dates=args.dates, n_random=args.random, workers=args.workers,
              force=args.force, leak_probe_dates=args.leak_probe,
              date_lo=args.date_lo or lo, date_hi=args.date_hi or hi, pplay=args.pplay,
              tails=args.tails, log=lambda m: print(m, flush=True))
    jp, tp = write_reports(rep, Path(args.out) if args.out else None)
    print(format_report(rep))
    print(f"\nwrote {jp}\nwrote {tp}")
    return 0
