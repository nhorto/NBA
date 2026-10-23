"""Top N lineups by no-good cuts (docs/PLAYERS.md §2.2, §5.4).

After lineup L_k the model gets Σ_{i∈L_k} x_i ≤ roster size − d (d = "min different players",
default 1), and the next lineup is solved: N sequential solves of one growing model, so the
objectives are non-increasing and lineup k is optimal among those at least d players away from
every earlier one. Per-player exposure caps (max fraction of the N lineups) ban a player once he
has been used `floor(cap·N)` times. When fewer than N legal lineups remain the list is short and
`TopNResult.stopped` says why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import floor
from typing import Mapping, Sequence

from .optimizer import (DEFAULT_LAMBDA, Lineup, LineupModel, Projections, model_for)
from .rules import PoolPlayer, Rules, SlatePool


@dataclass(frozen=True)
class LineupDiff:
    """A lineup against the #1 lineup."""
    n_different: int
    players_in: tuple[str, ...]      # ids in this lineup, not in #1
    players_out: tuple[str, ...]     # ids in #1, not in this lineup
    d_projected: float
    d_objective: float
    d_salary: int


@dataclass(frozen=True)
class RankedLineup:
    rank: int
    lineup: Lineup
    diff: LineupDiff

    def to_dk_row(self) -> list[str]:
        return self.lineup.to_dk_row()


@dataclass(frozen=True)
class TopNResult:
    lineups: tuple[RankedLineup, ...]
    requested: int
    stopped: str | None              # None | "exhausted" | "timeout"
    seconds: float

    def __len__(self) -> int:
        return len(self.lineups)

    def __iter__(self):
        return iter(self.lineups)

    @property
    def complete(self) -> bool:
        return len(self.lineups) == self.requested

    def exposure(self) -> dict[str, float]:
        """Fraction of the returned lineups each player is in."""
        out: dict[str, int] = {}
        for r in self.lineups:
            for pid in r.lineup.player_ids:
                out[pid] = out.get(pid, 0) + 1
        n = max(len(self.lineups), 1)
        return {k: v / n for k, v in out.items()}


def diff_vs(first: Lineup, other: Lineup) -> LineupDiff:
    a, b = first.player_ids, other.player_ids
    return LineupDiff(len(b - a), tuple(sorted(b - a)), tuple(sorted(a - b)),
                      other.projected - first.projected, other.objective - first.objective,
                      other.salary - first.salary)


def _caps(max_exposure, n: int) -> tuple[float | None, dict[str, float]]:
    if max_exposure is None:
        return None, {}
    if isinstance(max_exposure, Mapping):
        return None, {k: float(v) for k, v in max_exposure.items()}
    return float(max_exposure), {}


def top_n(pool: SlatePool | Sequence[PoolPlayer], projections: Projections,
          rules: Rules | None = None, n: int = 10, *, min_diff: int = 1,
          mode: str = "mean", lam: float = DEFAULT_LAMBDA,
          max_exposure: float | Mapping[str, float] | None = None,
          total_time_limit_s: float | None = None, model: LineupModel | None = None,
          **options) -> TopNResult:
    """The N best lineups, each at least `min_diff` players away from every earlier one.

    `max_exposure`   one fraction for every player, or {id: fraction}: a player appears in at
                     most floor(fraction·N) lineups (0 keeps him out).
    `total_time_limit_s` stops the loop after that many seconds (`stopped == "timeout"`).
    `options`        `LineupModel` options (locks, bans, now, existing, min_salary, max_salary,
                     time_limit_s per solve, seed). Exposure caps never override a lock.
    `model`          reuse an already built (and possibly pre-cut) `LineupModel`.
    """
    t0 = time.perf_counter()
    if model is None:
        model = model_for(pool, projections, rules, mode=mode, lam=lam, **options)
    uniform, per_player = _caps(max_exposure, n)
    limits: dict[str, int] = {}
    if uniform is not None:
        limits = {pid: floor(uniform * n + 1e-9) for pid in model.x}
    limits.update({pid: floor(c * n + 1e-9) for pid, c in per_player.items() if pid in model.x})
    used: dict[str, int] = {}
    for pid, lim in limits.items():
        if lim <= 0 and pid not in model.forced:
            model.ban(pid)
    out: list[RankedLineup] = []
    stopped = None
    while len(out) < n:
        left = None
        if total_time_limit_s is not None:
            left = total_time_limit_s - (time.perf_counter() - t0)
            if left <= 0:
                stopped = "timeout"
                break
        lu = model.solve(time_limit_s=_min_limit(model.time_limit_s, left))
        if lu is None:
            stopped = "timeout" if model.status == "unknown" else "exhausted"
            break
        out.append(RankedLineup(len(out) + 1, lu, diff_vs(out[0].lineup, lu) if out
                                else diff_vs(lu, lu)))
        model.add_no_good(lu, min_diff)
        for pid in lu.player_ids:
            used[pid] = used.get(pid, 0) + 1
            lim = limits.get(pid)
            if lim is not None and used[pid] >= lim and pid not in model.forced:
                model.ban(pid)
    return TopNResult(tuple(out), n, stopped, time.perf_counter() - t0)


NEAR_TIE_DK = 0.5               # rank_by_mean: means this close count as tied (PLAYERS.md §5.4)


def rank_by_mean(res: TopNResult, p_cash: Sequence[float | None], *,
                 tol: float = NEAR_TIE_DK) -> TopNResult:
    """PLAYERS.md §5.4: the same lineups ranked by mean (`Lineup.projected`), ties and near-ties
    broken by P(cash) (`p_cash[i]` of `res.lineups[i]`, None = unknown, last). Walking down the
    mean order, a group is the leader and every next lineup within `tol` DK of the leader; a group
    is ordered by P(cash) desc, then the old order. Ranks and the diffs vs #1 are recomputed;
    the set never changes (IMPROVEMENTS.md D2, `cash_pcash`)."""
    items = list(res.lineups)
    if len(p_cash) != len(items):
        raise ValueError(f"{len(p_cash)} P(cash) values for {len(items)} lineups")
    order = sorted(range(len(items)), key=lambda i: (-items[i].lineup.projected, i))
    out_idx: list[int] = []
    k = 0
    while k < len(order):
        lead = items[order[k]].lineup.projected
        j = k
        while j < len(order) and lead - items[order[j]].lineup.projected <= tol + 1e-9:
            j += 1
        grp = order[k:j]
        grp.sort(key=lambda i: (-(p_cash[i] if p_cash[i] is not None else float("-inf")), i))
        out_idx.extend(grp)
        k = j
    first = items[out_idx[0]].lineup if out_idx else None
    ranked = tuple(RankedLineup(r + 1, items[i].lineup, diff_vs(first, items[i].lineup))
                   for r, i in enumerate(out_idx))
    return TopNResult(ranked, res.requested, res.stopped, res.seconds)


def _min_limit(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None
