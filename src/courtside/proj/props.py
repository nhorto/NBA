"""Player props vs our projection (docs/PLAYERS.md T2.9, §2.8, §7.5): `prop_compare`.

For each main player prop (points, rebounds, assists, PRA, 3PM: the ESPN/DraftKings "Total ..."
over/under markets in `prop_snapshots`) of a game, this joins the book line to the newest
pre-tip projection run for that game and writes one `prop_compare` row with both chances of
finishing above the line. Information only: STYLE.md bans "over" / "edge" / "lean" as advice.

Which run
    The newest ok `dfs_runs` row of kind lock10 / lock45 / manual created before tip-off (and
    before `at`) that has `player_projections` for the game: the same newest-run rule the DFS
    page uses, across every slate that contains the game. Rows of the run's model_version only.

Our probability: a per-stat distribution built from the run
    **Copula v1 runs** (T2.10, `COURTSIDE_COPULA=v1`) store per-stat count sims beside the DK
    sims (`store.stat_sims_path`): then a stat's distribution **conditional on playing** (§7.5:
    props on a player who does not play are void) is the empirical pmf of the player's played
    sims (PRA = pts + reb + ast per sim), when he has >= MIN_PLAYED_SIMS of them.
    Otherwise (v0 runs, a missing file) the fallback: Path A v0 simulates DK points and minutes;
    per-stat values (pts, reb, ast, fg3m) are stored as means only (rates x minutes,
    environment-scaled, `player_projection_stats` with NULL sd/quantiles) and the sims npz holds
    DK points only, so the distribution of a stat given playing is approximated:

        minutes  m ~ the run's simulated played-minutes distribution (its stored q05..q95,
                     interpolated; `minutes_draws`)
        stat | m ~ NB1 with mean mu * m / E[m] and variance PHI[stat] * mean (Poisson if PHI <= 1)

    mu is the run's played-conditional stat mean (PRA = pts + reb + ast). PHI is the
    var/mean ratio of the stat around a 30-game per-minute rate x actual minutes, measured on
    2023-24..2025-26 (70k player-games, `fit_phi`; the live value only: an evaluation of season S
    uses `phi_for(conn, S)`, fit on the 3 seasons before S): it is flat in the mean (pts 2.2-2.5,
    reb 1.09-1.13, ast 1.13-1.21, 3PM 1.03-1.10, PRA 1.85-2.13). It includes a naive rate's
    estimation error, so it is a little wide for our (shrunk) rates: conservative, pulled
    towards 50%. Calibration (§7.5 bar) is measured from the graded rows.

Lines and pushes
    A half line L has no push: P(above) = P(Y >= ceil(L)). An integer line L pushes on Y == L:
    `our_p_above` = P(Y > L), `our_p_push` = P(Y = L), `our_p_below` = P(Y < L); the three sum
    to 1. Book prices on an integer line refund the push, so their no-vig chance is a chance
    given no push; `comparable(...)` gives ours on the same basis, P(Y > L) / (1 - P(Y = L)).

The market's chance
    No-vig, proportional: p_over = q_o / (q_o + q_u), q = the American price's implied chance;
    `hold` = q_o + q_u - 1. Stored in `mkt_p_above`.

Snapshots
    `prop_snapshots` is a change log; the market state at T is the newest row per (provider,
    athlete, market) with captured_at <= T. One provider per game (DraftKings, else the first).
    `write_day` stores the state at the run's created_at (same-snapshot comparison for the §7.5
    blend test) and the newest pre-tip state, each a row keyed (projection_id, market, line,
    captured_at); the page shows the newest.

Grading (postgame)
    `grade_day` fills actual / result ('above' | 'below' | 'push' | 'void': did not play) /
    went_above (1, 0, NULL on push or void) / graded_at once the game is final with a box score
    (proj.dfs.grade.game_states).

CLI (no external fetches; read-only unless --write / grade):
    python -m courtside.proj.props compare --day 2026-10-20 [--db PATH] [--write] [--unjoined]
    python -m courtside.proj.props grade --day 2026-10-20 [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
from scipy.special import gammaln

from .. import config
from ..injuries import match_player, norm_name

# ESPN market name (lower case) -> our stat. Main markets only (not milestones / quarters).
MARKETS = {
    "total points": "pts",
    "total rebounds": "reb",
    "total assists": "ast",
    "total points, rebounds, and assists": "pra",
    "total 3-point field goals": "fg3m",
}
STATS = ("pts", "reb", "ast", "pra", "fg3m")
LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "pra": "Pts + Reb + Ast",
         "fg3m": "3-pointers"}
# var / mean of a stat given minutes (module doc; `fit_phi` re-measures it). This is the LIVE
# value: it serves season LIVE_PHI_SEASON, whose training seasons (the 3 before it, 2023-24 ..
# 2025-26) are all in the past. Scoring any earlier season must use `phi_for` (seasons < S only).
PHI = {"pts": 2.40, "reb": 1.11, "ast": 1.16, "fg3m": 1.09, "pra": 2.05}
LIVE_PHI_SEASON = "2026-27"
PHI_N_SEASONS = 3                   # trailing seasons before S that the fit uses
RUN_KINDS = ("lock10", "lock45", "manual")
PREFERRED_PROVIDER = "100"          # DraftKings via ESPN
QUANTS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
N_MINUTES = 64                       # minute draws per player (inverse-CDF midpoints)
MIN_PLAYED_SIMS = 200                # stat sims: fewer played sims -> the approximation
MAX_MINUTES = 58.0

# unjoined reasons
R_NO_RUN = "no pre-tip projection run for the game"
R_NOT_PROJECTED = "player is on a roster but not in the run's projections"
R_NO_MATCH = "name not matched to an NBA player"
R_AMBIGUOUS = "name matches more than one player"
R_NO_MEAN = "projection has no mean for this stat"
R_BAD_PRICE = "missing over or under price"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ============================================================================ market maths

def implied(odds: Any) -> float | None:
    """Implied chance of an American price (-120 -> 0.5455, +150 -> 0.4)."""
    a = _num(odds)
    if a is None or a == 0 or -100 < a < 100:
        return None
    return 100.0 / (a + 100.0) if a > 0 else -a / (-a + 100.0)


def no_vig(over_odds: Any, under_odds: Any) -> tuple[float | None, float | None]:
    """(no-vig chance of the over side, hold). Proportional: q_o / (q_o + q_u)."""
    qo, qu = implied(over_odds), implied(under_odds)
    if qo is None or qu is None or qo + qu <= 0:
        return None, None
    return qo / (qo + qu), qo + qu - 1.0


# ============================================================================ our distribution

def minutes_draws(q: dict | None, mean: float | None = None, n: int = N_MINUTES) -> np.ndarray:
    """Played-minutes draws from stored quantiles q05..q95 (inverse CDF at the n midpoints,
    linear inside, the edge slopes extended into the tails, clipped to [0.5, 58]). Without
    quantiles: the mean as a point mass. Empty array when nothing is known."""
    vals = [_num((q or {}).get(c)) for c in QCOLS]
    if any(v is None for v in vals):
        m = _num(mean)
        return np.array([m], float) if m and m > 0 else np.zeros(0)
    xs, ys = np.array(QUANTS), np.maximum.accumulate(np.array(vals, float))
    u = (np.arange(n) + 0.5) / n
    m = np.interp(u, xs, ys)
    lo_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    hi_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    m = np.where(u < xs[0], ys[0] - (xs[0] - u) * lo_slope, m)
    m = np.where(u > xs[-1], ys[-1] + (u - xs[-1]) * hi_slope, m)
    return np.clip(m, 0.5, MAX_MINUTES)


def stat_pmf(mu: float, minutes: np.ndarray | None, phi: float) -> np.ndarray:
    """P(Y = k), k = 0..K, of a stat with played-conditional mean `mu`: a mixture over the
    minute draws of NB1(mean mu * m / mean(m), var = phi * mean); Poisson when phi <= 1.
    Sums to 1 (the tail beyond K, < 1e-9, is folded into the last cell)."""
    mu = max(float(mu), 0.0)
    if mu <= 0:
        return np.array([1.0])
    if minutes is None or len(minutes) == 0 or float(np.mean(minutes)) <= 0:
        means = np.array([mu])
    else:
        means = mu * np.asarray(minutes, float) / float(np.mean(minutes))
    means = np.maximum(means, 1e-6)
    top = float(means.max())
    K = int(math.ceil(top + 12.0 * math.sqrt(max(phi, 1.0) * top) + 10))
    k = np.arange(K + 1, dtype=float)[:, None]
    lgk = gammaln(k + 1.0)
    if phi <= 1.0 + 1e-6:                   # Poisson
        lam = means[None, :]
        logp = k * np.log(lam) - lam - lgk
    else:                                   # NB1: size r = mean / (phi - 1), success p = 1 / phi
        r = (means / (phi - 1.0))[None, :]
        logp = (gammaln(k + r) - gammaln(r) - lgk + r * math.log(1.0 / phi)
                + k * math.log(1.0 - 1.0 / phi))
    out = np.exp(logp).mean(axis=1)
    out[-1] += max(0.0, 1.0 - out.sum())
    return out / out.sum()


def line_probs(pmf: np.ndarray, line: float) -> tuple[float, float, float]:
    """(P(Y > line), P(Y == line), P(Y < line)) for a pmf on 0..K. Half lines: push 0."""
    L = float(line)
    cdf = np.cumsum(pmf)
    K = len(pmf) - 1

    def cdf_at(x: float) -> float:            # P(Y <= x) for real x
        i = math.floor(x)
        if i < 0:
            return 0.0
        return float(cdf[min(i, K)])
    below = cdf_at(math.ceil(L) - 1)
    push = float(pmf[int(L)]) if L == int(L) and 0 <= int(L) <= K else 0.0
    above = max(0.0, 1.0 - below - push)
    return above, push, below


def comparable(p_above: float | None, p_push: float | None) -> float | None:
    """Our chance above the line on the market's basis (a push is refunded):
    P(above) / (1 - P(push))."""
    if p_above is None:
        return None
    pp = p_push or 0.0
    return p_above / (1.0 - pp) if pp < 1.0 else None


def pmf_quantile(pmf: np.ndarray, q: float) -> int:
    """Smallest k with P(Y <= k) >= q."""
    return int(np.searchsorted(np.cumsum(pmf), q - 1e-12))


@dataclass
class PlayerDist:
    """Per-stat played-conditional distributions of one projection (pmfs memoised). `source`
    'sims' = empirical pmfs of the run's per-stat sims, 'approx' = the NB-given-minutes
    approximation."""
    means: dict
    minutes: np.ndarray
    _pmfs: dict = field(default_factory=dict, repr=False)
    source: str = "approx"
    phi: dict | None = None             # None = the live PHI; evaluations pass phi_for(S)

    def pmf(self, stat: str) -> np.ndarray | None:
        mu = self.means.get(stat)
        if mu is None:
            return None
        if stat not in self._pmfs:
            self._pmfs[stat] = stat_pmf(mu, self.minutes, (self.phi or PHI)[stat])
        return self._pmfs[stat]

    def range(self, stat: str, lo: float = 0.10, hi: float = 0.90) -> tuple[int, int] | None:
        p = self.pmf(stat)
        if p is None:
            return None
        return pmf_quantile(p, lo), pmf_quantile(p, hi)


def sims_dist(counts: dict, minutes: np.ndarray | None = None) -> PlayerDist | None:
    """PlayerDist from one player's per-stat sim counts {stat: (S,)} and minutes (S,) (0 = did
    not play): empirical pmfs over the played sims, PRA per sim. None when he has fewer than
    MIN_PLAYED_SIMS played sims or a stat is missing."""
    if any(s not in counts for s in ("pts", "reb", "ast", "fg3m")):
        return None
    played = (np.asarray(minutes) > 0) if minutes is not None else np.ones(
        len(counts["pts"]), bool)
    if int(played.sum()) < MIN_PLAYED_SIMS:
        return None
    vals = {s: np.asarray(counts[s], np.int64)[played] for s in ("pts", "reb", "ast", "fg3m")}
    vals["pra"] = vals["pts"] + vals["reb"] + vals["ast"]
    pmfs = {s: np.bincount(v) / len(v) for s, v in vals.items()}
    means = {s: float(v.mean()) for s, v in vals.items()}
    return PlayerDist(means=means, minutes=np.asarray(minutes, float)[played]
                      if minutes is not None else np.zeros(0), _pmfs=pmfs, source="sims")


def run_stat_sims(run: dict | None, game_id: str | None = None,
                  cache: dict | None = None) -> dict[int, PlayerDist]:
    """{player_id: PlayerDist} from a run's per-stat sims file (copula v1 runs), restricted to
    `game_id` when the file carries game ids. {} when the run has none (v0 runs)."""
    from .store import load_sims, stat_sims_path
    sp = (run or {}).get("sim_path")
    if not sp:
        return {}
    key = ("statsims", sp)
    z = cache.get(key) if cache is not None else None
    if z is None:
        path = stat_sims_path(sp)
        try:
            z = load_sims(path) if path.exists() else {}
        except (OSError, ValueError):
            z = {}
        if cache is not None:
            cache[key] = z
    if not z or "player_id" not in z:
        return {}
    cols = np.arange(len(z["player_id"]))
    if game_id is not None and "game_id" in z:
        cols = cols[z["game_id"] == str(game_id)]
    out = {}
    for j in cols:
        d = sims_dist({s: z[s][:, j] for s in ("pts", "reb", "ast", "fg3m") if s in z},
                      z["min"][:, j] if "min" in z else None)
        if d is not None:
            out.setdefault(int(z["player_id"][j]), d)
    return out


def player_dist(stats: dict, phi: dict | None = None) -> PlayerDist:
    """From {(stat, cond): row} of player_projection_stats: played-conditional means of
    pts/reb/ast/fg3m (+ PRA) and the played-minutes draws. `phi` overrides the live PHI (an
    evaluation of season S passes `phi_for(conn, S)`)."""
    means = {}
    for s in ("pts", "reb", "ast", "fg3m"):
        row = stats.get((s, "played")) or {}
        means[s] = _num(row.get("mean"))
    if all(means.get(s) is not None for s in ("pts", "reb", "ast")):
        means["pra"] = means["pts"] + means["reb"] + means["ast"]
    else:
        means["pra"] = None
    mrow = stats.get(("min", "played")) or {}
    return PlayerDist(means=means, minutes=minutes_draws(mrow, mrow.get("mean")), phi=phi)


# ============================================================================ data access

def _rows(conn: sqlite3.Connection, sql: str, args: Iterable = ()) -> list[dict]:
    cur = conn.execute(sql, tuple(args))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def day_games(conn: sqlite3.Connection, day: str) -> list[dict]:
    return _rows(conn, "SELECT game_id, nba_day, home, away, tipoff_utc, status, espn_id "
                       "FROM games WHERE nba_day=? ORDER BY tipoff_utc, game_id", (day,))


def pretip_run(conn: sqlite3.Connection, game_id: str, cutoff: str | None) -> dict | None:
    """The newest ok lock10/lock45/manual run created before `cutoff` with projections for
    the game (None if there is none)."""
    q = ",".join("?" * len(RUN_KINDS))
    sql = (f"SELECT r.* FROM dfs_runs r WHERE r.kind IN ({q}) AND COALESCE(r.ok, 1) = 1 "
           "AND EXISTS (SELECT 1 FROM player_projections p WHERE p.run_id = r.run_id "
           "AND p.game_id = ?)")
    args: list = [*RUN_KINDS, game_id]
    if cutoff:
        sql += " AND r.created_at <= ?"
        args.append(cutoff)
    sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1"
    rows = _rows(conn, sql, args)
    return rows[0] if rows else None


def run_projections(conn: sqlite3.Connection, run: dict, game_id: str) -> dict[int, dict]:
    """{player_id: projection row + 'stats': {(stat, cond): row}} of one run and game."""
    sql = "SELECT * FROM player_projections WHERE run_id=? AND game_id=?"
    args: list = [run["run_id"], game_id]
    if run.get("model_version"):
        rows = _rows(conn, sql + " AND model_version=? ORDER BY id", args + [run["model_version"]])
        if not rows:
            rows = _rows(conn, sql + " ORDER BY id", args)
    else:
        rows = _rows(conn, sql + " ORDER BY id", args)
    out = {int(r["player_id"]): dict(r, stats={}) for r in rows}
    by_id = {r["id"]: r for r in out.values()}
    ids = list(by_id)
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        for s in _rows(conn, "SELECT * FROM player_projection_stats WHERE projection_id IN "
                             f"({','.join('?' * len(chunk))})", chunk):
            by_id[s["projection_id"]]["stats"][(s["stat"], s["cond"])] = s
    return out


def market_state(conn: sqlite3.Connection, game_id: str, at: str | None) -> list[dict]:
    """Main over/under player props of a game as of `at` (newest row per provider, athlete and
    market with captured_at <= at), one provider (DraftKings first)."""
    sql = ("SELECT * FROM prop_snapshots WHERE game_id=? AND kind='ou' "
           "AND espn_athlete_id IS NOT NULL")
    args: list = [game_id]
    if at:
        sql += " AND captured_at <= ?"
        args.append(at)
    sql += " ORDER BY captured_at, id"
    latest: dict[tuple, dict] = {}
    for r in _rows(conn, sql, args):
        stat = MARKETS.get((r.get("market") or "").strip().lower())
        if stat is None:
            continue
        r["stat"] = stat
        latest[(r["provider"], r["espn_athlete_id"], r["market_id"])] = r
    provs = {r["provider"] for r in latest.values()}
    if not provs:
        return []
    prov = PREFERRED_PROVIDER if PREFERRED_PROVIDER in provs else sorted(provs)[0]
    return [r for r in latest.values() if r["provider"] == prov]


def _player_names(conn: sqlite3.Connection, pids: list[int]) -> dict[int, list[str]]:
    """Every name variant we hold for the players (rosters, player_index, box scores, DK)."""
    names: dict[int, list[str]] = {p: [] for p in pids}
    if not pids:
        return names
    ph = ",".join("?" * len(pids))

    def add(pid, name):
        if pid is None or not name:
            return
        lst = names.setdefault(int(pid), [])
        if name not in lst:
            lst.append(name)
    queries = (
        f"SELECT player_id, player FROM team_rosters WHERE player_id IN ({ph}) "
        "ORDER BY season DESC",
        f"SELECT player_id, TRIM(COALESCE(first_name,'') || ' ' || COALESCE(last_name,'')) "
        f"FROM player_index WHERE player_id IN ({ph})",
        f"SELECT DISTINCT i.player_id, s.player_name FROM player_ids i JOIN dfs_salaries s "
        f"ON s.site_player_id = i.source_id WHERE i.source='dk' AND i.player_id IN ({ph})",
    )
    for sql in queries:
        try:
            for pid, name in conn.execute(sql, pids):
                add(pid, name)
        except sqlite3.OperationalError:        # a table missing in a small/test DB
            continue
    try:                                        # box-score name of his latest game (index seek)
        for pid in pids:
            r = conn.execute("SELECT player_name FROM player_games WHERE player_id=? "
                             "ORDER BY game_date DESC LIMIT 1", (pid,)).fetchone()
            if r:
                add(pid, r[0])
    except sqlite3.OperationalError:
        pass
    return names


def _aliases(conn: sqlite3.Connection) -> dict[str, int]:
    try:
        return {r[0]: int(r[1]) for r in conn.execute(
            "SELECT name_norm, player_id FROM player_aliases WHERE source IN ('espn', 'dk')")}
    except sqlite3.OperationalError:
        return {}


def _roster_ids(conn: sqlite3.Connection, teams: Iterable[str]) -> list[tuple[int, str]]:
    """(player_id, name) of the teams' current rosters (for the 'not projected' reason)."""
    teams = [t for t in teams if t]
    if not teams:
        return []
    try:
        season = conn.execute("SELECT MAX(season) FROM team_rosters").fetchone()[0]
        return [(int(r[0]), r[1]) for r in conn.execute(
            f"SELECT player_id, player FROM team_rosters WHERE season=? AND team IN "
            f"({','.join('?' * len(teams))})", (season, *teams))]
    except sqlite3.OperationalError:
        return []


def match_name(name: str | None, cands: dict[int, list[str]],
               aliases: dict[str, int] | None = None) -> tuple[int | None, str]:
    """ESPN name -> one of `cands` {player_id: [names]}: alias, exact/compact on any name
    variant, then injuries.match_player (nicknames, typos) on the primary names.
    -> (player_id, how) or (None, 'no_match' | 'ambiguous')."""
    key = norm_name(name)
    if not key:
        return None, "no_match"
    if aliases and key in aliases and aliases[key] in cands:
        return aliases[key], "alias"
    comp = key.replace(" ", "")
    for how, test in (("exact", lambda n: norm_name(n) == key),
                      ("compact", lambda n: norm_name(n).replace(" ", "") == comp)):
        hit = {p for p, ns in cands.items() if any(test(n) for n in ns)}
        if len(hit) == 1:
            return hit.pop(), how
        if len(hit) > 1:
            return None, "ambiguous"
    roster = [(p, ns[0]) for p, ns in cands.items() if ns]
    pid, how = match_player(name, roster)
    if pid is not None:
        return pid, how or "fuzzy"
    return None, "ambiguous" if how == "ambiguous" else "no_match"


# ============================================================================ the comparison

def _cutoff(tip: str | None, at: str | None) -> str | None:
    vals = [v for v in (tip, at) if v]
    return min(vals) if vals else None


def compare_game(conn: sqlite3.Connection, game: dict, *, at: str | None = None,
                 market_at: str | None = None, run: dict | None = None,
                 proj: dict[int, dict] | None = None,
                 cache: dict | None = None) -> dict:
    """One game: {game, run, rows, unjoined, props}. `at` caps the run and the market (default:
    tip-off); `market_at` takes the market at another time (<= cutoff), e.g. the run's
    created_at. `run` / `proj` / `cache` let callers reuse what they already loaded."""
    gid = game["game_id"]
    cutoff = _cutoff(game.get("tipoff_utc"), at)
    if run is None:
        run = pretip_run(conn, gid, cutoff)
    props = market_state(conn, gid, _cutoff(cutoff, market_at) if market_at else cutoff)
    cache = cache if cache is not None else {}
    if run is not None and proj is None:
        proj = cache.get(("proj", run["run_id"], gid))
        if proj is None:
            proj = cache[("proj", run["run_id"], gid)] = run_projections(conn, run, gid)
    out = {"game": game, "run": run, "proj": proj or {}, "rows": [], "unjoined": [],
           "props": len(props)}
    if not props:
        return out
    if run is None:
        out["unjoined"] = [{**_prop_brief(p), "reason": R_NO_RUN} for p in props]
        return out
    names = cache.get(("names", run["run_id"], gid))
    if names is None:
        names = _player_names(conn, list(proj))
        cache[("names", run["run_id"], gid)] = names
    aliases = cache.setdefault("aliases", _aliases(conn))
    dists: dict[int, PlayerDist] = cache.setdefault(("dists", run["run_id"], gid), {})
    sim_dists = cache.get(("simdists", run["run_id"], gid))
    if sim_dists is None:                           # copula v1 runs: real per-stat sims
        sim_dists = cache[("simdists", run["run_id"], gid)] = run_stat_sims(run, gid, cache)
    roster = None
    for p in props:
        pid, how = match_name(p.get("player_name"), names, aliases)
        if pid is None:
            if roster is None:
                roster = _roster_ids(conn, (game.get("home"), game.get("away")))
            rp, _ = match_player(p.get("player_name"), roster) if roster else (None, None)
            reason = (R_NOT_PROJECTED if rp is not None
                      else R_AMBIGUOUS if how == "ambiguous" else R_NO_MATCH)
            out["unjoined"].append({**_prop_brief(p), "reason": reason})
            continue
        pr = proj[pid]
        d = dists.get(pid)
        if d is None:
            d = dists[pid] = sim_dists.get(pid) or player_dist(pr["stats"])
        stat = p["stat"]
        pmf = d.pmf(stat)
        if pmf is None:
            out["unjoined"].append({**_prop_brief(p), "player_id": pid, "reason": R_NO_MEAN})
            continue
        line = float(p["line"])
        above, push, below = line_probs(pmf, line)
        mkt, hold = no_vig(p.get("over_odds"), p.get("under_odds"))
        out["rows"].append({
            "projection_id": pr["id"], "game_id": gid, "player_id": pid, "team": pr.get("team"),
            "player_name": p.get("player_name"), "espn_athlete_id": p.get("espn_athlete_id"),
            "stat": stat, "market": LABEL[stat], "line": line,
            "over_odds": p.get("over_odds"), "under_odds": p.get("under_odds"),
            "captured_at": p["captured_at"], "provider": p.get("provider"),
            "mkt_p_above": mkt, "hold": hold,
            "our_p_above": above, "our_p_push": push, "our_p_below": below,
            "our_mean": d.means.get(stat), "run_id": run["run_id"],
            "run_created_at": run.get("created_at"), "model_version": run.get("model_version"),
            "match": how})
        if mkt is None:
            out["unjoined"].append({**_prop_brief(p), "player_id": pid, "reason": R_BAD_PRICE,
                                    "joined": True})
    return out


def _prop_brief(p: dict) -> dict:
    return {"game_id": p.get("game_id"), "player_name": p.get("player_name"),
            "espn_athlete_id": p.get("espn_athlete_id"), "stat": p.get("stat"),
            "market": LABEL.get(p.get("stat"), p.get("market")), "line": p.get("line"),
            "captured_at": p.get("captured_at")}


def compare_day(conn: sqlite3.Connection, day: str, *, at: str | None = None,
                cache: dict | None = None) -> dict:
    """Every game of an NBA day at the newest pre-tip market state: {day, games: [compare_game
    results], summary: {props, joined, join_rate, unjoined}}. Read-only."""
    cache = cache if cache is not None else {}
    games = [compare_game(conn, g, at=at, cache=cache) for g in day_games(conn, day)]
    return {"day": day, "games": games, "summary": summary(games)}


def summary(games: list[dict]) -> dict:
    n = sum(g["props"] for g in games)
    joined = sum(len(g["rows"]) for g in games)
    unj = [u for g in games for u in g["unjoined"] if not u.get("joined")]
    by_reason: dict[str, int] = {}
    for u in unj:
        by_reason[u["reason"]] = by_reason.get(u["reason"], 0) + 1
    return {"props": n, "joined": joined, "join_rate": joined / n if n else None,
            "unjoined": len(unj), "by_reason": by_reason}


# ============================================================================ storage

COMPARE_COLS = ("projection_id", "market", "line", "captured_at", "mkt_p_above", "our_p_above",
                "game_id", "player_id", "run_id", "run_created_at", "provider",
                "espn_athlete_id", "player_name", "over_odds", "under_odds", "hold",
                "our_p_push", "our_p_below", "our_mean", "built_at")
PK = ("projection_id", "market", "line", "captured_at")


def _store_row(r: dict, built_at: str) -> tuple:
    v = dict(r, market=r["stat"], built_at=built_at)
    return tuple(v.get(c) for c in COMPARE_COLS)


def write_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Upsert compare rows (market column = our stat key). Grading columns are kept."""
    if not rows:
        return 0
    built = _now()
    sets = ", ".join(f"{c}=excluded.{c}" for c in COMPARE_COLS if c not in PK)
    sql = (f"INSERT INTO prop_compare ({', '.join(COMPARE_COLS)}) VALUES "
           f"({', '.join('?' * len(COMPARE_COLS))}) ON CONFLICT({', '.join(PK)}) DO UPDATE SET "
           f"{sets}")
    with conn:
        conn.executemany(sql, [_store_row(r, built) for r in rows])
    return len(rows)


def write_day(conn: sqlite3.Connection, day: str, *, at: str | None = None) -> dict:
    """Compute and store a day: per game the newest pre-tip market state and the state at the
    run's created_at (when props had been captured by then). Returns the summary."""
    at = at or _now()
    cache: dict = {}
    games_latest, n_rows = [], 0
    for g in day_games(conn, day):
        res = compare_game(conn, g, at=at, cache=cache)
        games_latest.append(res)
        rows = list(res["rows"])
        if res["run"] is not None and res["run"].get("created_at"):
            at_run = compare_game(conn, g, at=at, market_at=res["run"]["created_at"],
                                  run=res["run"], cache=cache)
            rows += at_run["rows"]
        seen, uniq = set(), []
        for r in rows:
            k = (r["projection_id"], r["stat"], r["line"], r["captured_at"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        n_rows += write_rows(conn, uniq)
    s = summary(games_latest)
    s["rows_written"] = n_rows
    s["day"] = day
    return s


def grade_day(conn: sqlite3.Connection, day: str) -> dict:
    """Fill actual / result / went_above / graded_at for the day's rows of final games."""
    from .dfs.grade import game_states
    ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_id FROM prop_compare WHERE game_id IN "
        "(SELECT game_id FROM games WHERE nba_day=?)", (day,))]
    if not ids:
        return {"status": "skipped", "reason": "no prop_compare rows", "day": day}
    states = game_states(conn, date.fromisoformat(day), ids)
    final = [g for g in ids if states.get(g) == "final"]
    if not final:
        return {"status": "pending", "reason": "games not final", "day": day}
    ph = ",".join("?" * len(final))
    box = {(r[0], int(r[1])): r for r in conn.execute(
        f"SELECT game_id, player_id, min, pts, reb, ast, fg3m FROM player_games "
        f"WHERE game_id IN ({ph})", final)}
    now = _now()
    upd, counts = [], {"above": 0, "below": 0, "push": 0, "void": 0}
    for r in _rows(conn, f"SELECT rowid AS rid, game_id, player_id, market, line FROM "
                         f"prop_compare WHERE game_id IN ({ph})", final):
        b = box.get((r["game_id"], int(r["player_id"] or -1)))
        if b is None or not (b[2] or 0) > 0:
            actual, res, above = None, "void", None
        else:
            vals = {"pts": b[3], "reb": b[4], "ast": b[5], "fg3m": b[6]}
            vals["pra"] = (b[3] or 0) + (b[4] or 0) + (b[5] or 0)
            actual = float(vals[r["market"]] or 0)
            line = float(r["line"])
            res = "above" if actual > line else "below" if actual < line else "push"
            above = None if res == "push" else int(res == "above")
        counts[res] += 1
        upd.append((actual, res, above, now, r["rid"]))
    with conn:
        conn.executemany("UPDATE prop_compare SET actual=?, result=?, went_above=?, graded_at=? "
                         "WHERE rowid=?", upd)
    return {"status": "ok", "day": day, "games": len(final), "rows": len(upd), **counts}


# ============================================================================ collect hooks

def pregame_step(conn: sqlite3.Connection, day: date) -> dict:
    """collect pregame: store today's comparison (newest run and props so far)."""
    s = write_day(conn, day.isoformat())
    if not s["props"]:
        return {"status": "skipped", "reason": "no main props captured"}
    return {k: s[k] for k in ("props", "joined", "join_rate", "rows_written", "by_reason")}


def postgame_step(conn: sqlite3.Connection, day: date) -> dict:
    """collect postgame: final pre-tip comparison for the day, then the grades."""
    s = write_day(conn, day.isoformat())
    g = grade_day(conn, day.isoformat())
    if not s["props"] and g.get("status") == "skipped":
        return {"status": "skipped", "reason": "no main props captured"}
    return {"props": s["props"], "joined": s["joined"], "rows_written": s["rows_written"],
            "grade": g}


# ============================================================================ dispersion fit

def phi_seasons(season: str, n: int = PHI_N_SEASONS) -> tuple[str, ...]:
    """The `n` seasons immediately before `season`, oldest first (S = '2026-27' -> 2023-24,
    2024-25, 2025-26: the live fit window)."""
    y = int(season[:4])
    return tuple(f"{y - k}-{str(y - k + 1)[2:]}" for k in range(n, 0, -1))


_PHI_CACHE: dict = {}


def phi_for(conn: sqlite3.Connection | None, season: str, window: int = 30) -> dict[str, float]:
    """PHI for scoring `season`, fit on seasons < `season` only (never the season itself or
    anything later). The live season returns the packaged PHI unchanged (same training seasons,
    no DB needed); an earlier season is re-fit on its own trailing window and memoised."""
    if season >= LIVE_PHI_SEASON:
        return PHI
    if conn is None:
        raise ValueError(f"phi_for({season!r}) needs a connection: only the live season "
                         f"{LIVE_PHI_SEASON} has a packaged PHI")
    db_file = next((r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main"), "")
    if not db_file:                         # in-memory DB: no stable identity, do not memoise
        return fit_phi(conn, before=season, window=window)
    key = (db_file, season, window)
    if key not in _PHI_CACHE:
        _PHI_CACHE[key] = fit_phi(conn, before=season, window=window)
    return _PHI_CACHE[key]


def fit_phi(conn: sqlite3.Connection, seasons: tuple[str, ...] | None = None,
            window: int = 30, *, before: str | None = None) -> dict[str, float]:
    """Re-measure PHI: mean of (y - mu)^2 / mean of mu per stat, mu = the player's previous
    `window` games' per-minute rate x actual minutes (>= 8 min, >= 200 prior minutes).

    Seasons fitted: `seasons` if given, else the PHI_N_SEASONS before `before` (default
    LIVE_PHI_SEASON = 2023-24..2025-26). To score season S pass before=S so the fit never
    sees S."""
    import pandas as pd
    if seasons is None:
        seasons = phi_seasons(before or LIVE_PHI_SEASON)
    elif before is not None and any(s >= before for s in seasons):
        raise ValueError(f"fit_phi: seasons {seasons} are not all < {before}")
    ph = ",".join("?" * (len(seasons) + 1))
    prev = f"{int(seasons[0][:4]) - 1}-{seasons[0][2:4]}"
    df = pd.read_sql_query(
        f"SELECT player_id, game_date, season, min, pts, reb, ast, fg3m FROM player_games "
        f"WHERE season IN ({ph}) AND min > 0 ORDER BY player_id, game_date", conn,
        params=[prev, *seasons])
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    g = df.groupby("player_id")
    den = g["min"].transform(lambda x: x.shift(1).rolling(window, min_periods=15).sum())
    out = {}
    for s in STATS:
        num = g[s].transform(lambda x: x.shift(1).rolling(window, min_periods=15).sum())
        mu = num / den * df["min"]
        m = df["season"].isin(seasons) & mu.notna() & (df["min"] >= 8) & (den >= 200)
        y, mu_ = df.loc[m, s].to_numpy(float), np.maximum(mu[m].to_numpy(float), 0.05)
        out[s] = round(float(((y - mu_) ** 2).mean() / mu_.mean()), 3)
    return out


# ============================================================================ CLI

def _fmt_row(r: dict) -> str:
    ours = comparable(r["our_p_above"], r["our_p_push"])
    mkt = r["mkt_p_above"]
    line = f"{r['line']:g}"
    push = f" (push {r['our_p_push']:.0%})" if r["our_p_push"] else ""
    return (f"  {r['player_name']:<24} {r['team'] or '':<4} {LABEL[r['stat']]} {line}: "
            f"our model {ours:.0%}{push} / market "
            f"{'n/a' if mkt is None else f'{mkt:.0%}'}   [mean {r['our_mean']:.1f}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m courtside.proj.props")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "grade", "fit-phi"):
        p = sub.add_parser(name)
        p.add_argument("--day", default=None)
        p.add_argument("--db", default=None, help="SQLite path (default: the configured DB)")
        if name == "fit-phi":
            p.add_argument("--before", default=None,
                           help="score this season: fit on the 3 seasons before it "
                                "(default: the live window)")
        if name == "compare":
            p.add_argument("--write", action="store_true", help="store prop_compare rows")
            p.add_argument("--at", default=None, help="ISO UTC cap on runs and props")
            p.add_argument("--rows", type=int, default=12, help="sample rows to print")
    a = ap.parse_args(argv)
    day = a.day or config.today_nba_day().isoformat()
    if a.cmd == "compare" and not a.write:
        path = a.db or str(config.db_path())
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        from .. import db
        conn = db.connect(a.db)
    if a.cmd == "fit-phi":
        print(json.dumps(fit_phi(conn, before=a.before)))
        return 0
    if a.cmd == "grade":
        print(json.dumps(grade_day(conn, day), indent=1))
        return 0
    if a.write:
        print(json.dumps(write_day(conn, day, at=a.at), indent=1, default=str))
    res = compare_day(conn, day, at=a.at)
    s = res["summary"]
    rate = "n/a" if s["join_rate"] is None else f"{s['join_rate']:.1%}"
    print(f"{day}: {s['props']} main props, {s['joined']} joined ({rate})")
    for g in res["games"]:
        run = g["run"] or {}
        print(f"{g['game']['away']} @ {g['game']['home']} {g['game']['tipoff_utc']}: "
              f"{g['props']} props, {len(g['rows'])} joined, run {run.get('run_id')} "
              f"{run.get('kind')} {run.get('created_at')} {run.get('model_version')}")
        for r in g["rows"][:a.rows]:
            print(_fmt_row(r))
        for u in g["unjoined"]:
            print(f"  UNJOINED {u.get('player_name')} {u.get('market')} {u.get('line')}: "
                  f"{u['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
