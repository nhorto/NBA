"""RotoGuru DFS backtest harness (docs/PLAYERS.md §7, §8 rows T2.1 / T2.4 / T2.17).

    courtside proj backtest-dfs [--seasons 2019-20 2020-21 ...] [--dates N] [--protocol avail]
                                [--sims 4000] [--topn 20] [--workers 1] [--leak-probe N]

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
from datetime import date, datetime
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
CASH_LAMBDA = E.CASH_LAMBDA
LEAK_TOL = 0.005                        # §8 T2.1: leak probe < 0.5% (mean abs relative)
SPEND_FRAC = 0.96
SOLVE_LIMIT_S = 20.0
PLAYER_COLS = ("date", "season", "rg_id", "player_id", "game_id", "team", "pos", "salary", "y",
               "minutes", "has_proj", "mean", "q10", "q25", "q50", "q75", "q90", "exp_min",
               "p_play", "tier", "crps", "pit", "sal_all", "sal_prev", "sal_played", "sal_fixed",
               "b_season_avg", "b_fppg", "b_last10")


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
              "availability.py", "evallite.py", "dfs/optimizer.py", "dfs/topn.py", "dfs/rules.py")


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
    return f"v{HARNESS_VERSION}-" + _hash({"s": asdict(settings), "model": P.MODEL_VERSION,
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
        if need_context:
            self.load()

    def load(self) -> "SeasonRun":
        from . import pipeline as P
        if self.ctx is not None:
            return self
        self.ctx = P.Context(self.conn, self.season, log=self.log)
        self.box = BoxBaselines(self.conn, self.season)
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
            if self.settings.protocol == "reports":
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
        return _frame(rows), proj, pool_rows

    def _players_box(self, day, res, pl, rng):
        """Player-level rows for a season without salaries: every simulated player."""
        act = self.box.actual
        y_of = {}
        for g in res.sims:
            for pid in g.player_id:
                v = act.get((g.game_id, int(pid)))
                y_of[int(pid)] = v[0] if v else 0.0
        sc = self._sim_scores(res, lambda p: y_of.get(p, np.nan), rng)
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
                   "b_season_avg": savg, "b_fppg": fppg, "b_last10": l10}
            c = sc.get(pid)
            if c is not None:
                row["crps"], row["pit"] = c
            rows.append(row)
        return _frame(rows)

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
               workers: int = 1, force: bool = False, log: Callable[[str], None] = print
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run (or load from the cache) every date of `season`; returns the concatenated player and
    lineup frames and a meta dict (seconds, skipped dates, cache directory, salary fit)."""
    t0 = time.time()
    sr = SeasonRun(conn, season, settings, log=log, need_context=False)
    cache = sr.cache()
    days = sr.dates(n_dates)
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
        log: Callable[[str], None] = print) -> dict:
    """The backtest. `seasons` default 2014-15 .. 2020-21 plus 2024-25 (player level only).
    Walk-forward: season S uses a bundle fit on seasons < S and salary lines from RotoGuru
    seasons < S. Cached per date, so a rerun only does what is missing."""
    t0 = time.time()
    seasons = list(seasons) if seasons else list(DEFAULT_SEASONS)
    if protocol not in ("avail", "reports"):
        raise ValueError(f"unknown protocol {protocol!r}")
    for s in seasons:
        if protocol == "reports" and is_rotoguru(s):
            raise ValueError("protocol 'reports' needs the rescued official reports (2025-26)")
    bad = [m for m in modes if m not in ("cash", "topn")]
    if bad:
        raise ValueError(f"unknown mode(s) {bad}")
    st = Settings(protocol=protocol, n_sims=sims, modes=tuple(modes), n_topn=n_topn,
                  baselines=tuple(baselines), topn_baselines=tuple(topn_baselines),
                  n_random=n_random)
    pls, lus, metas = [], [], {}
    for s in seasons:
        p, ln, m = run_season(conn, s, st, n_dates=n_dates, workers=workers, force=force, log=log)
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
    p.add_argument("--protocol", choices=("avail", "reports"), default="avail")
    p.add_argument("--sims", type=int, default=4_000)
    p.add_argument("--topn", type=int, default=20, help="N for the Top-N lineups")
    p.add_argument("--modes", nargs="+", default=["cash", "topn"], choices=("cash", "topn"))
    p.add_argument("--random", type=int, default=2_000, help="random lineups per date")
    p.add_argument("--workers", type=int, default=1, help="processes per season (keep <= 4)")
    p.add_argument("--leak-probe", type=int, default=0, metavar="N",
                   help="also run the leak probe on N dates per season")
    p.add_argument("--force", action="store_true", help="ignore and overwrite the date cache")
    p.add_argument("--out", default=None, help="report directory (default var/reports)")
    return {"backtest-dfs": _cmd}


def _cmd(args) -> int:
    from .. import db
    conn = db.connect()
    rep = run(conn, args.seasons, protocol=args.protocol, modes=args.modes, n_topn=args.topn,
              sims=args.sims, n_dates=args.dates, n_random=args.random, workers=args.workers,
              force=args.force, leak_probe_dates=args.leak_probe,
              log=lambda m: print(m, flush=True))
    jp, tp = write_reports(rep, Path(args.out) if args.out else None)
    print(format_report(rep))
    print(f"\nwrote {jp}\nwrote {tp}")
    return 0
