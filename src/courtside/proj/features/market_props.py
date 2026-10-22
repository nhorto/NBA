"""F9: market player-prop lines as model inputs (docs/PLAYERS.md Phase 2F, F9). NOT LIVE.

Status: the PIPE only. Prop lines exist from 2026-10-19 (ESPN / DraftKings, `prop_snapshots`),
so nothing can be fit or validated yet; `readiness` says when there is enough. The group is
registered in `features.GROUPS` as "F9" but is in no live group set (`context.LIVE_GROUPS*`)
and no shipped bundle, so it changes nothing until a model is fit with it. PLAYERS.md §7.1 L6
("never features: ... props") has to be amended by the owner before an F9 model ships.

Source and timing
    `prop_snapshots` is a change log (db.py): a row is written when a market's line or either
    price differs from the last stored state, at the pregame collector's polls (6 a day:
    10:00, 13:30, 17:00, 18:30, 20:00, 22:00 ET; games already tipped are not polled). The state
    at T is the newest row per (game, provider, athlete, market) with **captured_at <= T**
    (our clock: when we saw it). ESPN's `last_updated` is never used for timing (it is
    sometimes later than our capture). Rows captured after the cutoff are dropped BEFORE
    anything else is computed, including the first-seen line used for `move`.

Markets (player over/under, kind 'ou')
    pts, reb, ast, fg3m, pra (the props.py set) + stl, blk and the combos pr, pa, ra, sb.
    Milestone ladders, quarter markets and one-sided prices are ignored.

De-vig and the implied mean (method)
    1. No-vig chance at the line, proportional (props.no_vig): p = q_o / (q_o + q_u) with q the
       American prices' implied chances; hold = q_o + q_u - 1. On an integer line the prices
       refund the push, so p is a chance given no push.
    2. Implied distribution: a count Y | plays (props are void on a DNP) with mean mu, drawn as
       a mixture over a minutes multiplier m ~ Gamma(mean 1, CV MIN_CV) of NB1(mu * m,
       var = PHI_MKT[stat] * mean) (props.stat_pmf; PHI_MKT = props.PHI's var/mean given
       minutes, measured on 70k player-games). mu is solved so that the model's comparable
       chance P(Y > L) / (1 - P(Y = L)) equals p (monotone in mu; interpolated on a fixed
       log-spaced grid per stat, so it is fast and deterministic). That mu is `mean`.
    3. `fair` = the continuous median of that distribution: the half-line at which it would be
       50/50 (linear between half-lines). At p = 0.5 on a half line, fair == the line.
    Assumptions: the shape (dispersion, minutes CV) is ours, not the market's. Sensitivity is
    small near p = 0.5 (the mean moves ~ skew x sd) and grows with |p - 0.5|. `pover` keeps the
    raw no-vig chance so a model can use it directly.

Multiple books, moves, missing lines
    * Per book: the newest state <= cutoff. Across books (only DraftKings today): `line` and
      `mean` are medians, `pover` the mean, `nbooks` the count.
    * `move` = line now minus the first line we captured for that market (<= cutoff);
      `mmove` = implied mean now minus the implied mean at that first capture (catches juice
      moves on an unchanged line). ESPN's own `open_line` is kept as `open`.
    * `age` = minutes from the capture to the cutoff (stale states are possible: the change log
      cannot say that a market was pulled; see the gaps note in the readiness report).
    * Missing lines: most bench players have none. `has_<s>` = 0 and the numeric columns are
      NaN (LightGBM routes NaN; a blend falls back to our number). `has_any` / `nmkts` say
      whether the player had any main market at all ("the book bothered" is information).

Public API
    devig(over, under) -> (p_over, hold)
    implied_means(stat, lines, ps) / implied_mean(stat, line, p) / fair_line(stat, mu)
    load_snapshots(conn, game_ids=None)        player o/u rows of the markets above
    asof(snaps, cutoffs)                       newest row per book/athlete/stat <= cutoff[game]
    cutoffs_for(conn, game_ids, mode, anchor)  lock45 / lock10 cutoffs (tip- or slate-anchored)
    lines_asof(conn, game_ids, cutoffs, candidates=None) -> wide (game_id, player_id) frame
    live_lines(conn, game_ids, at, candidates=None)      the live-path loader (cutoff min(at, tip))
    columns / monotone / NEUTRAL / build(conn, frame) / row_features(frame)   the F9 group
    readiness(conn, mode) / diagnostic(conn, mode)                            reports

Live-path hook (NOT wired; pipeline.py is owned elsewhere). When a bundle's layer carries F9,
in `pipeline.context_rates` (after `ext` is computed):
    if "F9" in b.context.groups:
        from .features import market_props as MP
        f9 = MP.live_lines(conn, games["game_id"].tolist(), at=<run created_at, ISO Z>,
                           candidates={g: ids of that game's panel rows})
        pf = f9 if pf is None else pf.merge(f9, on=["game_id", "player_id"], how="left")
    and pass `player_feats=pf` to `CX.apply_rates` (attach() then adds the F9 row features).
    The run's own time is the cutoff, so a lock-45 run sees the market as of its build.

CLI (read-only):
    python -m courtside.proj.features.market_props readiness [--db PATH] [--mode lock45]
    python -m courtside.proj.features.market_props diagnostic [--db PATH] [--mode lock45]
    python -m courtside.proj.features.market_props lines --day 2026-10-21 [--db PATH]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import timedelta
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .. import props as PR

# ESPN market name (lower case) -> stat key
MARKETS = {
    **PR.MARKETS,
    "total steals": "stl",
    "total blocks": "blk",
    "total points and rebounds": "pr",
    "total points and assists": "pa",
    "total assists and rebounds": "ra",
    "total steals and blocks": "sb",
}
MAIN = ("pts", "reb", "ast", "fg3m", "pra")             # the props.py / §7.5 markets
STATS = MAIN + ("stl", "blk", "pr", "pa", "ra", "sb")
FEATURE_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "pra")
# var / mean given minutes: props.PHI (measured); stl / blk Poisson-like; combos by analogy
PHI_MKT = {**PR.PHI, "stl": 1.05, "blk": 1.10, "pr": 2.10, "pa": 2.10, "ra": 1.20, "sb": 1.05}
MIN_CV = 0.20            # played-minutes CV of a player with a line (starters / rotation)
N_MIX = 16               # minutes-multiplier mixture components
MU_GRID = np.geomspace(0.02, 90.0, 700)
MODES = {"lock45": 45, "lock10": 10}
DEFAULT_MODE = "lock45"

# wide columns per stat (prefix f9_<col>_<stat>)
PER_STAT = ("has", "line", "pover", "hold", "mean", "fair", "move", "mmove", "open", "age",
            "nbooks")


def _iso(ts) -> str:
    """ISO UTC 'YYYY-MM-DDTHH:MM:SSZ' (the format of captured_at / tipoff_utc; naive = UTC)."""
    t = pd.Timestamp(ts)
    t = t.tz_convert("UTC") if t.tzinfo else t
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================ de-vig maths

def devig(over_odds, under_odds) -> tuple[float | None, float | None]:
    """(no-vig chance of the over, hold): proportional, as props.no_vig."""
    return PR.no_vig(over_odds, under_odds)


def _multipliers(cv: float = MIN_CV, n: int = N_MIX) -> np.ndarray:
    from scipy.stats import gamma
    k = 1.0 / (cv * cv)
    m = gamma.ppf((np.arange(n) + 0.5) / n, a=k, scale=1.0 / k)
    return m / m.mean()


@lru_cache(maxsize=None)
def _table(stat: str) -> tuple[np.ndarray, np.ndarray]:
    """(pmf, cdf) matrices [len(MU_GRID), K+1] of the implied distribution of `stat`."""
    mult = _multipliers()
    pmfs = [PR.stat_pmf(float(mu), mult, PHI_MKT[stat]) for mu in MU_GRID]
    K = max(len(p) for p in pmfs)
    pmf = np.zeros((len(MU_GRID), K))
    for i, p in enumerate(pmfs):
        pmf[i, :len(p)] = p
    return pmf, np.cumsum(pmf, axis=1)


def _comparable_curve(stat: str, line: float) -> np.ndarray:
    """P(Y > L) / (1 - P(Y = L)) at every grid mean (non-decreasing in mu)."""
    pmf, cdf = _table(stat)
    K = pmf.shape[1] - 1
    L = float(line)
    i_below = math.ceil(L) - 1
    below = cdf[:, min(i_below, K)] if i_below >= 0 else np.zeros(len(MU_GRID))
    push = pmf[:, int(L)] if (L == int(L) and 0 <= int(L) <= K) else np.zeros(len(MU_GRID))
    above = np.clip(1.0 - below - push, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp = np.where(push < 1.0, above / (1.0 - push), 0.0)
    return np.maximum.accumulate(comp)


def implied_mean(stat: str, line, p_over) -> float | None:
    """Mean of the implied played-conditional distribution whose comparable chance above
    `line` is `p_over` (module doc). None when the inputs are missing or out of range."""
    if stat not in PHI_MKT or line is None or p_over is None:
        return None
    L, p = float(line), float(p_over)
    if not (math.isfinite(L) and math.isfinite(p)) or L < 0 or not 0.0 < p < 1.0:
        return None
    comp = _comparable_curve(stat, L)
    if p <= comp[0] or p >= comp[-1]:
        return None
    # first index with comp >= p; interpolate on log mu between it and the one before
    j = int(np.searchsorted(comp, p))
    c0, c1 = comp[j - 1], comp[j]
    w = 0.0 if c1 <= c0 else (p - c0) / (c1 - c0)
    return float(math.exp(math.log(MU_GRID[j - 1]) * (1 - w) + math.log(MU_GRID[j]) * w))


def implied_means(stat: str, lines: Iterable, ps: Iterable) -> np.ndarray:
    return np.array([np.nan if (v := implied_mean(stat, L, p)) is None else v
                     for L, p in zip(lines, ps)], float)


def _cdf_at_mu(stat: str, mu: float) -> np.ndarray:
    _, cdf = _table(stat)
    j = int(np.clip(np.searchsorted(MU_GRID, mu), 1, len(MU_GRID) - 1))
    a, b = math.log(MU_GRID[j - 1]), math.log(MU_GRID[j])
    w = float(np.clip((math.log(mu) - a) / (b - a), 0.0, 1.0))
    return cdf[j - 1] * (1 - w) + cdf[j] * w


def fair_line(stat: str, mu) -> float | None:
    """The half-line at which the implied distribution with mean `mu` is 50/50 (continuous
    median, linear between half-lines)."""
    if mu is None or not np.isfinite(mu) or mu <= 0:
        return None
    cdf = _cdf_at_mu(stat, float(mu))         # P(Y <= k) = P(Y < k + 0.5)
    k = int(np.searchsorted(cdf, 0.5))
    lo = cdf[k - 1] if k > 0 else 0.0         # P(Y < k - 0.5)
    hi = cdf[min(k, len(cdf) - 1)]
    w = 0.0 if hi <= lo else (0.5 - lo) / (hi - lo)
    return float((k - 0.5) + w)


# ============================================================================ snapshots

def _rows(conn, sql: str, args: Sequence = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=list(args))


def load_snapshots(conn: sqlite3.Connection, game_ids: Iterable[str] | None = None
                   ) -> pd.DataFrame:
    """Player over/under rows of the F9 markets (all captures; filter with `asof`)."""
    sql = ("SELECT id, game_id, captured_at, provider, espn_athlete_id, player_name, market, "
           "line, over_odds, under_odds, open_line FROM prop_snapshots WHERE kind='ou' "
           "AND espn_athlete_id IS NOT NULL AND line IS NOT NULL")
    args: list = []
    if game_ids is not None:
        ids = sorted({str(g) for g in game_ids})
        if not ids:
            return _empty_snaps()
        sql += f" AND game_id IN ({','.join('?' * len(ids))})"
        args = ids
    try:
        df = _rows(conn, sql, args)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return _empty_snaps()
    df["stat"] = df["market"].str.strip().str.lower().map(MARKETS)
    return df[df["stat"].notna()].reset_index(drop=True)


def _empty_snaps() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "game_id", "captured_at", "provider", "espn_athlete_id",
                                 "player_name", "market", "line", "over_odds", "under_odds",
                                 "open_line", "stat"])


def asof(snaps: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """Newest row per (game_id, provider, espn_athlete_id, stat) with captured_at <= the
    game's cutoff (games without a cutoff are dropped). Adds `first_line` / `first_over` /
    `first_under` (the earliest capture <= cutoff), `n_changes`, `cutoff`, `age` (minutes)."""
    if not len(snaps) or not cutoffs:
        return snaps.iloc[0:0].assign(first_line=[], first_over=[], first_under=[],
                                      n_changes=[], cutoff=[], age=[])
    s = snaps.copy()
    s["cutoff"] = s["game_id"].astype(str).map({str(k): _iso(v) for k, v in cutoffs.items()})
    s = s[s["cutoff"].notna() & (s["captured_at"] <= s["cutoff"])]       # strictly pre-cutoff
    if not len(s):
        return s.assign(first_line=[], first_over=[], first_under=[], n_changes=[], age=[])
    s = s.sort_values(["captured_at", "id"], kind="mergesort")
    key = ["game_id", "provider", "espn_athlete_id", "stat"]
    g = s.groupby(key, sort=False)
    first = g.head(1).set_index(key)[["line", "over_odds", "under_odds"]]
    first.columns = ["first_line", "first_over", "first_under"]
    last = g.tail(1).set_index(key)
    last["n_changes"] = g.size()
    out = last.join(first).reset_index()
    out["age"] = ((pd.to_datetime(out["cutoff"], utc=True)
                   - pd.to_datetime(out["captured_at"], utc=True)).dt.total_seconds() / 60.0)
    return out.reset_index(drop=True)


def cutoffs_for(conn: sqlite3.Connection, game_ids: Iterable[str], mode: str = DEFAULT_MODE,
                anchor: str = "tip") -> dict[str, str]:
    """{game_id: cutoff ISO} = lock - 45 / - 10 minutes. anchor 'tip': the game's own tip-off
    (one consistent rule per game, the training default); 'slate': the earliest lock of the
    DK slates that contain the game (what a slate's lock-45 run actually sees), else the tip."""
    off = timedelta(minutes=MODES[mode])
    ids = sorted({str(g) for g in game_ids})
    if not ids:
        return {}
    tips = dict(conn.execute(
        f"SELECT game_id, tipoff_utc FROM games WHERE game_id IN ({','.join('?' * len(ids))}) "
        "AND tipoff_utc IS NOT NULL", ids).fetchall())
    lock = dict(tips)
    if anchor == "slate":
        import json
        try:
            for lk, gj in conn.execute("SELECT lock_utc, game_ids_json FROM dfs_slates "
                                       "WHERE lock_utc IS NOT NULL"):
                for g in json.loads(gj or "[]"):
                    g = str(g)
                    if g in lock and lk < lock[g]:
                        lock[g] = lk
        except sqlite3.OperationalError:
            pass
    return {g: _iso(pd.Timestamp(t) - off) for g, t in lock.items()}


# ============================================================================ players

def _game_candidates(conn: sqlite3.Connection, game_ids: Sequence[str]) -> dict[str, list[int]]:
    """Default candidates per game: the two teams' latest rosters + anyone with a box row or a
    projection for the game."""
    out: dict[str, set] = {str(g): set() for g in game_ids}
    if not game_ids:
        return {}
    ph = ",".join("?" * len(game_ids))
    games = conn.execute(f"SELECT game_id, home, away FROM games WHERE game_id IN ({ph})",
                         list(game_ids)).fetchall()
    try:
        season = conn.execute("SELECT MAX(season) FROM team_rosters").fetchone()[0]
        for gid, home, away in games:
            for (pid,) in conn.execute("SELECT player_id FROM team_rosters WHERE season=? AND "
                                       "team IN (?, ?)", (season, home, away)):
                out[str(gid)].add(int(pid))
    except sqlite3.OperationalError:
        pass
    for table in ("player_games", "player_projections"):
        try:
            for gid, pid in conn.execute(f"SELECT DISTINCT game_id, player_id FROM {table} "
                                         f"WHERE game_id IN ({ph})", list(game_ids)):
                out[str(gid)].add(int(pid))
        except sqlite3.OperationalError:
            continue
    return {g: sorted(v) for g, v in out.items()}


def map_players(conn: sqlite3.Connection, rows: pd.DataFrame,
                candidates: dict | None = None) -> pd.DataFrame:
    """Adds `player_id` (props.match_name on the game's candidates; NaN when unmatched or
    ambiguous). `candidates` {game_id: [player_id]} (default `_game_candidates`)."""
    out = rows.copy()
    if not len(out):
        out["player_id"] = pd.Series(dtype=float)
        return out
    games = sorted(out["game_id"].astype(str).unique())
    cands = {str(k): list(v) for k, v in (candidates or {}).items()}
    missing = [g for g in games if g not in cands]
    if missing:
        cands.update(_game_candidates(conn, missing))
    names = PR._player_names(conn, sorted({int(p) for v in cands.values() for p in v}))
    aliases = PR._aliases(conn)
    memo: dict[tuple, float] = {}
    pids = []
    for gid, aid, nm in zip(out["game_id"].astype(str), out["espn_athlete_id"],
                            out["player_name"]):
        k = (gid, aid)
        if k not in memo:
            c = {int(p): names.get(int(p), []) for p in cands.get(gid, [])}
            pid, _ = PR.match_name(nm, c, aliases) if c else (None, "no_match")
            memo[k] = float(pid) if pid is not None else np.nan
        pids.append(memo[k])
    out["player_id"] = pids
    return out


# ============================================================================ the pipe

def _per_book(a: pd.DataFrame) -> pd.DataFrame:
    """De-vig + implied mean / fair line per book row of `asof`."""
    a = a.copy()
    dv = [devig(o, u) for o, u in zip(a["over_odds"], a["under_odds"])]
    a["pover"] = [d[0] for d in dv]
    a["hold"] = [d[1] for d in dv]
    p0 = [devig(o, u)[0] for o, u in zip(a["first_over"], a["first_under"])]
    a["mean"] = [implied_mean(s, L, p) for s, L, p in zip(a["stat"], a["line"], a["pover"])]
    a["mean0"] = [implied_mean(s, L, p) for s, L, p in zip(a["stat"], a["first_line"], p0)]
    a["fair"] = [fair_line(s, m) for s, m in zip(a["stat"], a["mean"])]
    for c in ("pover", "hold", "mean", "mean0", "fair"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["move"] = a["line"].astype(float) - a["first_line"].astype(float)
    a["mmove"] = a["mean"] - a["mean0"]
    return a


def consensus(a: pd.DataFrame) -> pd.DataFrame:
    """Across books, per (game_id, player_id, stat): medians of line / mean / fair / move /
    mmove, mean of pover / hold, max age, nbooks."""
    a = a[a["player_id"].notna() & a["pover"].notna()]
    if not len(a):
        return pd.DataFrame(columns=["game_id", "player_id", "stat", *PER_STAT])
    g = a.groupby(["game_id", "player_id", "stat"], sort=True)
    out = g.agg(line=("line", "median"), mean=("mean", "median"), fair=("fair", "median"),
                move=("move", "median"), mmove=("mmove", "median"), open=("open_line", "median"),
                pover=("pover", "mean"), hold=("hold", "mean"), age=("age", "max"),
                nbooks=("provider", "nunique")).reset_index()
    out["has"] = 1.0
    return out


def wide_columns() -> list[str]:
    return ([f"f9_{c}_{s}" for s in STATS for c in PER_STAT] + ["f9_has_any", "f9_nmkts"])


def _wide(cons: pd.DataFrame, keys: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = wide_columns()
    if len(cons):
        w = cons.pivot_table(index=["game_id", "player_id"], columns="stat",
                             values=list(PER_STAT), aggfunc="first")
        w.columns = [f"f9_{c}_{s}" for c, s in w.columns]
        w = w.reset_index()
    else:
        w = pd.DataFrame(columns=["game_id", "player_id"])
    if keys is not None:
        k = keys[["game_id", "player_id"]].drop_duplicates().copy()
        k["game_id"] = k["game_id"].astype(str)
        k["player_id"] = k["player_id"].astype(float)
        w["game_id"] = w["game_id"].astype(str)
        w["player_id"] = w["player_id"].astype(float)
        w = k.merge(w, on=["game_id", "player_id"], how="left")
    miss = [c for c in cols if c not in w]
    if miss:
        w = pd.concat([w, pd.DataFrame(np.nan, index=w.index, columns=miss)], axis=1)
    for s in STATS:
        w[f"f9_has_{s}"] = w[f"f9_has_{s}"].fillna(0.0)
    main = [f"f9_has_{s}" for s in MAIN]
    w["f9_nmkts"] = w[main].sum(axis=1).astype(float)
    w["f9_has_any"] = (w["f9_nmkts"] > 0).astype(float)
    w["player_id"] = w["player_id"].astype("int64") if len(w) else w["player_id"]
    return w[["game_id", "player_id", *cols]].reset_index(drop=True)


def lines_asof(conn: sqlite3.Connection, game_ids: Iterable[str], cutoffs: dict,
               candidates: dict | None = None, keys: pd.DataFrame | None = None,
               snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """The F9 pipe: one row per (game_id, player_id) with the market state of every F9 stat
    as of the game's cutoff (module doc). `keys` (game_id, player_id) = the rows to return
    (players without a line get has = 0 and NaN); default: only players with a line."""
    ids = sorted({str(g) for g in game_ids})
    s = snaps if snaps is not None else load_snapshots(conn, ids)
    a = asof(s, {g: cutoffs[g] for g in ids if g in cutoffs})
    if len(a):
        a = map_players(conn, a, candidates)
        a = _per_book(a)
    else:
        a = a.assign(player_id=[], pover=[])
    return _wide(consensus(a), keys)


def live_lines(conn: sqlite3.Connection, game_ids: Iterable[str], at,
               candidates: dict | None = None, keys: pd.DataFrame | None = None
               ) -> pd.DataFrame:
    """The live-path loader: cutoff = min(at, tip-off) per game (never a post-tip state)."""
    ids = sorted({str(g) for g in game_ids})
    at_s = _iso(at)
    tips = cutoffs_for(conn, ids, "lock10")           # tip - 10; recover the tip
    cut = {}
    for g in ids:
        tip = _iso(pd.Timestamp(tips[g]) + timedelta(minutes=10)) if g in tips else None
        cut[g] = min(at_s, tip) if tip else at_s
    return lines_asof(conn, ids, cut, candidates=candidates, keys=keys)


# ============================================================================ the F9 group

MODEL_STATS = ("pts", "fg3m", "reb", "ast", "stl", "blk")     # DK stats with their own market


def columns(stat: str) -> list[str]:
    cols = ["f9_has_any", "f9_nmkts"]
    if stat in MODEL_STATS:
        cols += [f"f9_has_{stat}", f"f9_lr_{stat}", f"f9_move_{stat}", f"f9_pover_{stat}"]
    if stat in ("pts", "reb", "ast", "tov"):
        cols += ["f9_lr_pra"]
    return cols


def monotone(stat: str) -> dict:
    return {f"f9_lr_{stat}": 1} if stat in MODEL_STATS else {}


ROW_COLS = tuple(f"f9_lr_{s}" for s in MODEL_STATS) + ("f9_lr_pra",)
TEAM_COLS = tuple(wide_columns())          # the merged table's columns (store.extend_frame)
NEUTRAL = {**{c: 0.0 for c in ROW_COLS}, **{f"f9_move_{s}": 0.0 for s in MODEL_STATS},
           **{f"f9_pover_{s}": 0.5 for s in MODEL_STATS}}


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """f9_lr_<s> = log(market implied mean / (EB rate x mpg)): how far the market's number
    sits from our pre-game per-minute rate times the minutes state. NaN without a line."""
    out = {}
    mpg = np.maximum(frame["mpg"].to_numpy(float), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in MODEL_STATS:
            mu = frame[f"f9_mean_{s}"].to_numpy(float)
            r = np.maximum(frame[f"r_{s}"].to_numpy(float), 1e-6)
            out[f"f9_lr_{s}"] = np.log(np.maximum(mu, 1e-3) / (r * mpg))
            out[f"f9_lr_{s}"][~np.isfinite(mu)] = np.nan
        mu = frame["f9_mean_pra"].to_numpy(float)
        r = sum(np.maximum(frame[f"r_{s}"].to_numpy(float), 1e-6) for s in ("pts", "reb", "ast"))
        lr = np.log(np.maximum(mu, 1e-3) / (r * mpg))
        lr[~np.isfinite(mu)] = np.nan
        out["f9_lr_pra"] = lr
    return pd.DataFrame(out, index=frame.index)


def build(conn: sqlite3.Connection, frame: pd.DataFrame, mode: str = DEFAULT_MODE,
          anchor: str = "tip") -> pd.DataFrame:
    """F9 table for the (game_id, player_id) rows of `frame` (store.extend_frame): only games
    with snapshots are queried, so historical frames cost one query and come back all-NaN."""
    keys = frame[["game_id", "player_id"]].drop_duplicates()
    try:
        with_props = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM prop_snapshots")}
    except sqlite3.OperationalError:
        with_props = set()
    games = sorted(set(keys["game_id"].astype(str)) & with_props)
    cands = {g: sorted(keys.loc[keys["game_id"].astype(str) == g, "player_id"].astype(int))
             for g in games}
    cut = cutoffs_for(conn, games, mode, anchor) if games else {}
    return lines_asof(conn, games, cut, candidates=cands, keys=keys)


# ============================================================================ readiness

# Target for the first walk-forward test (module doc of `readiness`)
TARGET_PER_MARKET = 2000
TARGET_NIGHTS = 15


def _played(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return _rows(conn, "SELECT pg.game_id, pg.player_id, pg.min, pg.pts, pg.reb, pg.ast, "
                           "pg.fg3m, pg.stl, pg.blk FROM player_games pg JOIN games g "
                           "ON g.game_id = pg.game_id WHERE g.status = 'final' AND pg.min > 0")
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame(columns=["game_id", "player_id", "min"])


def readiness(conn: sqlite3.Connection, mode: str = DEFAULT_MODE) -> dict:
    """How many graded player-games with a pre-cutoff line we hold, per market.

    Target: >= TARGET_PER_MARKET played player-games with a line per main market over
    >= TARGET_NIGHTS nights before the first walk-forward test. Why: the §7.5 "adds
    information" bar is the blend beating the market by > 2 game-clustered SE. Assuming a
    per-row log-score gain of ~0.004 nats with a per-row sd of ~0.04 and a cluster design
    effect ~1.5 (players of one game share its environment), 2 SE needs ~900 test rows and
    80% power ~1,800; the §7.5 calibration check needs >= 500 per market. A 1-2 parameter blend
    fit per fold needs a few hundred rows, so expanding weekly folds from ~2,000 rows give a
    first test with ~1,500 out-of-sample rows. Re-estimate the per-row sd from the first folds.
    """
    games = _rows(conn, "SELECT game_id, nba_day, status, tipoff_utc FROM games")
    snaps = load_snapshots(conn)
    if not len(snaps):
        return {"mode": mode, "nights": 0, "markets": {}, "per_night": [], "target": {
            "per_market": TARGET_PER_MARKET, "nights": TARGET_NIGHTS}}
    gids = sorted(snaps["game_id"].astype(str).unique())
    cut = cutoffs_for(conn, gids, mode)
    a = asof(snaps, cut)
    a = map_players(conn, a)
    a = a[a["player_id"].notna()]
    a["player_id"] = a["player_id"].astype(int)
    a = a.drop_duplicates(["game_id", "player_id", "stat"])
    day = dict(zip(games["game_id"].astype(str), games["nba_day"]))
    a["nba_day"] = a["game_id"].astype(str).map(day)
    pl = _played(conn)
    pl_keys = set(zip(pl["game_id"].astype(str), pl["player_id"].astype(int)))
    boxed = set(pl["game_id"].astype(str))     # final AND its box score is in (else not graded)
    final = set(games.loc[games["status"] == "final", "game_id"].astype(str)) & boxed
    a["final"] = a["game_id"].astype(str).isin(final)
    a["played"] = [(g, p) in pl_keys for g, p in zip(a["game_id"].astype(str), a["player_id"])]
    markets = {}
    for s in STATS:
        x = a[a["stat"] == s]
        markets[s] = {"lined": int(len(x)), "graded": int(x["final"].sum()),
                      "played": int((x["final"] & x["played"]).sum()),
                      "void_dnp": int((x["final"] & ~x["played"]).sum()),
                      "per_game": round(len(x) / max(len(gids), 1), 1)}
    per_night = (a.groupby("nba_day").agg(games=("game_id", "nunique"),
                                          lined_pts=("stat", lambda v: int((v == "pts").sum())))
                 .reset_index().to_dict("records"))
    # projection: future scheduled games x observed per-game yield of played lined rows
    nights_done = a.loc[a["final"], "nba_day"].nunique()
    sched = games[(games["status"] == "scheduled")
                  & (~games["game_id"].astype(str).isin(set(gids)))].sort_values("nba_day")
    eta = {}
    played_games = max(int(a.loc[a["final"], "game_id"].nunique()), 1)
    for s in MAIN:
        rate = markets[s]["played"] / played_games
        need = TARGET_PER_MARKET - markets[s]["played"]
        if need <= 0:
            eta[s] = "reached"
            continue
        if rate <= 0 or not len(sched):
            eta[s] = None
            continue
        cum = np.cumsum(np.full(len(sched), rate))
        hit = int(np.searchsorted(cum, need))
        eta[s] = (sched["nba_day"].iloc[hit] if hit < len(sched) else None)
    return {"mode": mode, "nights": int(a["nba_day"].nunique()), "nights_graded": int(nights_done),
            "games_with_props": len(gids), "markets": markets, "per_night": per_night,
            "eta": eta, "providers": sorted(snaps["provider"].astype(str).unique()),
            "captures_per_day": snaps.groupby(snaps["captured_at"].str[:10])["captured_at"]
            .nunique().to_dict(),
            "target": {"per_market": TARGET_PER_MARKET, "nights": TARGET_NIGHTS}}


EVAL_PLAN = """Evaluation plan (when the target is met; nothing ships before it):
  * Rows: played player-games with a line at the cutoff (lock-45, and lock-10 separately),
    our run at the same cutoff (same-snapshot rule of §7.5), per main market.
  * Arms: market alone (implied NB from the de-vigged line), ours alone, and the blend
    log mu = a + b log ours + c log mkt (or the logit-average of P(above) for the binary test),
    fit walk-forward on expanding weekly folds (never on the test week).
  * Metrics: NB log-lik of the actual count and CRPS of the count distribution; binary
    log-loss of P(above line). SE clustered by game (two-way game x player as a check).
  * Bar (HANDOFF §4 / PLAYERS §7.5): the blend beats the market by > 2 clustered SE, is
    calibrated (Spiegelhalter |z| < 2.5, n >= 500) and beats the naive baselines. Only then
    is F9 a candidate input to the DFS projection (and L6 must be amended by the owner);
    end to end it must also not worsen DK CRPS (T2.1 live slice)."""


def format_readiness(r: dict) -> str:
    lines = [f"F9 readiness ({r['mode']} cutoff; providers {', '.join(r.get('providers', []))})",
             f"nights with props {r['nights']} (graded {r.get('nights_graded', 0)}), games "
             f"{r.get('games_with_props', 0)}; polls with a change per UTC day "
             f"{r.get('captures_per_day', {})}",
             f"target per main market: >= {r['target']['per_market']} played player-games over "
             f">= {r['target']['nights']} nights",
             f"{'market':6} {'lined':>6} {'/game':>6} {'graded':>7} {'played':>7} {'void':>5}  eta"]
    for s, m in r["markets"].items():
        lines.append(f"{s:6} {m['lined']:6d} {m['per_game']:6.1f} {m['graded']:7d} "
                     f"{m['played']:7d} {m['void_dnp']:5d}  {r.get('eta', {}).get(s, '') or ''}")
    lines.append(EVAL_PLAN)
    lines.append("Gaps: one provider (DraftKings via ESPN); 6 polls a day, so lock-45 of a 19:00 "
                 "ET slate sees the 17:00 poll; the change log cannot record a pulled market "
                 "(stale state possible; check `age`); no history before 2026-10-19.")
    return "\n".join(lines)


# ============================================================================ diagnostic

def _tier(role: str | None) -> str:
    return role or "?"


def diagnostic(conn: sqlite3.Connection, mode: str = DEFAULT_MODE) -> pd.DataFrame:
    """Our played-conditional means vs the de-vigged market mean, per game at the newest run
    of kind `mode` created before tip-off, the market taken at that run's created_at (same
    snapshot). Descriptive only. One row per (game, player, stat) joined."""
    snaps = load_snapshots(conn)
    gids = sorted(snaps["game_id"].astype(str).unique())
    rows = []
    for gid in gids:
        g = conn.execute("SELECT game_id, nba_day, tipoff_utc, status FROM games WHERE "
                         "game_id=?", (gid,)).fetchone()
        if not g or not g[2]:
            continue
        run = conn.execute(
            "SELECT r.run_id, r.created_at, r.model_version FROM dfs_runs r WHERE r.kind=? "
            "AND COALESCE(r.ok,1)=1 AND r.created_at <= ? AND EXISTS (SELECT 1 FROM "
            "player_projections p WHERE p.run_id=r.run_id AND p.game_id=?) "
            "ORDER BY r.created_at DESC LIMIT 1", (mode, g[2], gid)).fetchone()
        if not run:
            continue
        proj = PR.run_projections(conn, {"run_id": run[0], "model_version": run[2]}, gid)
        if not proj:
            continue
        cands = {gid: sorted(proj)}
        w = lines_asof(conn, [gid], {gid: run[1]}, candidates=cands, snaps=snaps[
            snaps["game_id"].astype(str) == gid])
        for _, r in w.iterrows():
            pr = proj.get(int(r["player_id"]))
            if pr is None:
                continue
            st = pr["stats"]
            means = {s: PR._num((st.get((s, "played")) or {}).get("mean"))
                     for s in ("pts", "reb", "ast", "fg3m")}
            if all(means[s] is not None for s in ("pts", "reb", "ast")):
                means["pra"] = means["pts"] + means["reb"] + means["ast"]
            for s in MAIN:
                if r.get(f"f9_has_{s}", 0) != 1 or means.get(s) is None:
                    continue
                rows.append({"game_id": gid, "nba_day": g[1], "status": g[3],
                             "run_id": run[0], "model_version": run[2],
                             "player_id": int(r["player_id"]), "tier": _tier(pr.get("role")),
                             "stat": s, "ours": means[s], "mkt_mean": r[f"f9_mean_{s}"],
                             "line": r[f"f9_line_{s}"], "pover": r[f"f9_pover_{s}"],
                             "fair": r[f"f9_fair_{s}"], "p_play": pr.get("p_play")})
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    pl = _played(conn)
    if len(pl):
        pl = pl.melt(id_vars=["game_id", "player_id"], value_vars=["pts", "reb", "ast", "fg3m"],
                     var_name="stat", value_name="actual")
        pra = (_played(conn).assign(stat="pra", actual=lambda x: x["pts"] + x["reb"] + x["ast"])
               [["game_id", "player_id", "stat", "actual"]])
        pl = pd.concat([pl, pra], ignore_index=True)
        pl["game_id"] = pl["game_id"].astype(str)
        d = d.merge(pl, on=["game_id", "player_id", "stat"], how="left")
    else:
        d["actual"] = np.nan
    return d


def format_diagnostic(d: pd.DataFrame) -> str:
    if not len(d):
        return "diagnostic: no joined rows"
    nights = sorted(d["nba_day"].unique())
    out = [f"F9 early diagnostic: n = {len(nights)} night(s) {', '.join(nights)}; DESCRIPTIVE "
           f"ONLY, no inference. Models: {', '.join(sorted(d['model_version'].dropna().unique()))}",
           "diff = ours - market implied mean (played-conditional); vs fair = ours - the "
           "market's 50/50 line (median; shape-free sensitivity); corr = Pearson",
           f"{'stat':5} {'tier':6} {'n':>4} {'corr':>6} {'diff':>6} {'|diff|':>6} "
           f"{'vsfair':>6} {'n_act':>5} {'MAE ours':>8} {'MAE mkt':>8}"]
    for s in MAIN:
        for tier in ("all", "star", "role", "bench"):
            x = d[d["stat"] == s] if tier == "all" else d[(d["stat"] == s) & (d["tier"] == tier)]
            if len(x) < 3:
                continue
            c = np.corrcoef(x["ours"], x["mkt_mean"])[0, 1]
            diff = x["ours"] - x["mkt_mean"]
            a = x[x["actual"].notna()]
            mae_o = (a["ours"] - a["actual"]).abs().mean() if len(a) else np.nan
            mae_m = (a["mkt_mean"] - a["actual"]).abs().mean() if len(a) else np.nan
            vf = (x["ours"] - x["fair"]).mean()
            out.append(f"{s:5} {tier:6} {len(x):4d} {c:6.3f} {diff.mean():+6.2f} "
                       f"{diff.abs().mean():6.2f} {vf:+6.2f} {len(a):5d} {mae_o:8.2f} "
                       f"{mae_m:8.2f}")
    return "\n".join(out)


# ============================================================================ CLI

def _conn(path: str | None) -> sqlite3.Connection:
    from ... import config
    p = path or str(config.data_dir() / "courtside.db")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="courtside.proj.features.market_props")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("readiness", "diagnostic", "lines"):
        p = sub.add_parser(name)
        p.add_argument("--db")
        p.add_argument("--mode", choices=sorted(MODES), default=DEFAULT_MODE)
        if name == "lines":
            p.add_argument("--day", required=True)
    a = ap.parse_args(argv)
    conn = _conn(a.db)
    if a.cmd == "readiness":
        print(format_readiness(readiness(conn, a.mode)))
    elif a.cmd == "diagnostic":
        print(format_diagnostic(diagnostic(conn, a.mode)))
    else:
        gids = [r[0] for r in conn.execute("SELECT game_id FROM games WHERE nba_day=?",
                                           (a.day,))]
        w = lines_asof(conn, gids, cutoffs_for(conn, gids, a.mode))
        show = ["game_id", "player_id"] + [f"f9_{c}_{s}" for s in MAIN
                                           for c in ("line", "pover", "mean", "fair")]
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(w[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
