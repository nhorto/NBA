"""CP-SAT DraftKings lineup engine (docs/PLAYERS.md §5.2, §5.3, §5.6, §5.7; task T1.5).

The solver is independent of where the numbers come from: it takes a `SlatePool` (or any list
of `PoolPlayer`), the `Rules` and a mapping {DK player id -> `Projection` | float}. Rules decide
everything about the roster (slots, cap, ``min_games``, ``min_teams``, Showdown multiplier);
nothing is hard-coded here.

Model (`LineupModel`):

* binary ``x[player, slot name]`` over the slots the player is eligible for (identical slots,
  Showdown UTIL x5, are one group with a count, so there is no symmetry to break);
* every slot group is filled exactly its count, each player at most once (so a Showdown player
  cannot be both CPT and UTIL), salary ≤ cap (the CPT slot carries its own, 1.5x, salary);
* ≥ ``min_games`` distinct games and ≥ ``min_teams`` distinct teams, by indicator variables;
* locks (force in, optionally in a slot), bans, and late swap: a player whose game started is
  only allowed if he is locked in (an `existing` lineup pins its started players to their slots);
* objective Σ score·multiplier·x with scores scaled x100 to integers.

`num_workers=1` and a fixed seed make a solve deterministic. Top N / exposure live in `topn.py`.
Persistence helpers (`save_run`, `save_lineups`) write migration 9's ``dfs_runs`` and
``dfs_lineups``; ``Lineup.to_dk_row()`` is the DraftKings upload row.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ortools.sat.python import cp_model

from .rules import PoolPlayer, Rules, SlatePool, _aware

SCALE = 100                    # points -> integer objective coefficients
DEFAULT_LAMBDA = 0.3           # cash fallback: mu - lam * (mu - p25)
DEFAULT_SEED = 1


# ============================================================================== projections

@dataclass(frozen=True)
class Projection:
    """What the solver needs from the projection pipeline for one player: the mean DK points
    and (optionally) the 25th percentile for the cash objective. Percentiles are carried for the
    callers' display; only `mean` and `p25` enter an objective."""
    mean: float
    p25: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


Projections = Mapping[str, "Projection | float | Mapping[str, float]"]


def as_projection(v) -> Projection:
    """A float (the mean), a mapping with ``mean`` (+ optional ``p25``...) or a `Projection`."""
    if isinstance(v, Projection):
        return v
    if isinstance(v, Mapping):
        return Projection(**{k: (None if v.get(k) is None else float(v[k]))
                             for k in ("mean", "p25", "p10", "p50", "p90") if k in v})
    return Projection(float(v))


def objective_score(proj: Projection, mode: str = "mean", lam: float = DEFAULT_LAMBDA) -> float:
    """``mean``: mu. ``cash``: mu - lam * (mu - p25), mu if p25 is missing (PLAYERS.md §5.3)."""
    if mode == "mean":
        return proj.mean
    if mode == "cash":
        if proj.p25 is None:
            return proj.mean
        return proj.mean - lam * (proj.mean - proj.p25)
    raise ValueError(f"unknown objective mode {mode!r} (mean|cash)")


def player_scores(players: Iterable[PoolPlayer], projections: Projections, *,
                  mode: str = "mean", lam: float = DEFAULT_LAMBDA
                  ) -> tuple[dict[str, float], dict[str, float]]:
    """(scores, means) keyed by DK player id for the players that have a projection.
    Players without one are left out (and therefore cannot be picked)."""
    scores: dict[str, float] = {}
    means: dict[str, float] = {}
    for p in players:
        raw = projections.get(p.dk_player_id)
        if raw is None:
            continue
        pr = as_projection(raw)
        scores[p.dk_player_id] = objective_score(pr, mode, lam)
        means[p.dk_player_id] = pr.mean
    return scores, means


# ============================================================================== lineups

@dataclass(frozen=True)
class LineupEntry:
    slot: str
    player: PoolPlayer
    salary: int          # salary in this slot (CPT carries the multiplier)
    mean: float          # projected points in this slot (multiplier applied)
    score: float         # objective value in this slot (multiplier applied)

    @property
    def draftable_id(self) -> str | None:
        return self.player.draftable_ids.get(self.slot)


@dataclass(frozen=True)
class Lineup:
    """A solved lineup: one entry per roster slot in DK slot order."""
    entries: tuple[LineupEntry, ...]
    salary: int
    projected: float                 # Σ mean (multiplier applied)
    objective: float                 # Σ score (what the solver maximised)
    optimal: bool = True             # False when a time limit stopped the search

    @property
    def player_ids(self) -> frozenset[str]:
        return frozenset(e.player.dk_player_id for e in self.entries)

    @property
    def players(self) -> tuple[PoolPlayer, ...]:
        return tuple(e.player for e in self.entries)

    def slot_pairs(self) -> list[tuple[str, str]]:
        return [(e.slot, e.player.dk_player_id) for e in self.entries]

    def slot_map(self) -> dict[str, str | list[str]]:
        """slot name -> player id (a list when the name repeats, e.g. Showdown UTIL)."""
        out: dict[str, Any] = {}
        for e in self.entries:
            k, v = e.slot, e.player.dk_player_id
            if k in out:
                out[k] = (out[k] if isinstance(out[k], list) else [out[k]]) + [v]
            else:
                out[k] = v
        return out

    def to_dk_row(self) -> list[str]:
        """DraftKings upload row: the draftable id of each slot in the template's column order
        (PG,SG,SF,PF,C,G,F,UTIL; Showdown CPT,UTIL x5)."""
        row = []
        for e in self.entries:
            if e.draftable_id is None:
                raise KeyError(f"{e.player.name} has no draftable id for slot {e.slot}")
            row.append(str(e.draftable_id))
        return row

    def dk_header(self) -> list[str]:
        return [e.slot for e in self.entries]

    def slots_json(self) -> str:
        return json.dumps([{"slot": e.slot, "player_id": e.player.dk_player_id,
                            "name": e.player.name, "team": e.player.team,
                            "salary": e.salary, "mean": round(e.mean, 3),
                            "draftable_id": e.draftable_id} for e in self.entries])


def build_lineup(rules: Rules, by_id: Mapping[str, PoolPlayer], pairs: Iterable[tuple[str, str]],
                 scores: Mapping[str, float], means: Mapping[str, float] | None = None,
                 *, optimal: bool = True) -> Lineup:
    """A `Lineup` from (slot name, player id) pairs, ordered into the rules' slot order.
    Within a repeated slot name the more expensive player goes first (then id)."""
    means = scores if means is None else means
    groups: dict[str, list[PoolPlayer]] = defaultdict(list)
    for slot, pid in pairs:
        groups[slot].append(by_id[pid])
    queues = {name: iter(sorted(ps, key=lambda p, n=name: (-p.salary_for(n, rules),
                                                           p.dk_player_id)))
              for name, ps in groups.items()}
    ents: list[LineupEntry] = []
    for slot in rules.slots:                       # the template's column order
        p = next(queues.get(slot.name, iter(())), None)
        if p is None:
            raise ValueError(f"no player for slot {slot.name}")
        ents.append(LineupEntry(slot.name, p, p.salary_for(slot.name, rules),
                                means.get(p.dk_player_id, 0.0) * slot.multiplier,
                                scores.get(p.dk_player_id, 0.0) * slot.multiplier))
    return Lineup(tuple(ents), sum(e.salary for e in ents), sum(e.mean for e in ents),
                  sum(e.score for e in ents), optimal)


# ============================================================================== the model

def game_key(p: PoolPlayer) -> str:
    return p.competition_id or p.game_id or p.game_label or f"team:{p.team}"


def _norm_locks(locks) -> dict[str, str | None]:
    if not locks:
        return {}
    if isinstance(locks, Mapping):
        return dict(locks)
    return dict.fromkeys(locks)


class LineupModel:
    """One CP-SAT roster model that can be solved, cut (`add_no_good`) and solved again.

    `players`   candidate `PoolPlayer`s; only those with a score can be picked.
    `scores`    {dk_player_id: objective value per point-of-multiplier-1}.
    `means`     {dk_player_id: mean} used for `Lineup.projected` (default: the scores).
    `locks`     ids to force in ({id: slot name or None} or an iterable of ids).
    `bans`      ids to keep out.
    `now`       enables late swap: players whose game has started are banned unless locked
                (explicitly, or by being a started player of `existing`).
    `existing`  a `Lineup` or (slot, id) pairs: its started players are pinned to their slots.
    `min_salary`/`max_salary` bound the salary used (max defaults to the rules' cap).
    """

    def __init__(self, players: Sequence[PoolPlayer], rules: Rules,
                 scores: Mapping[str, float], means: Mapping[str, float] | None = None, *,
                 locks=None, bans: Iterable[str] = (), now: datetime | None = None,
                 existing: Lineup | Iterable[tuple[str, str]] | None = None,
                 min_salary: int | None = None, max_salary: int | None = None,
                 time_limit_s: float | None = None, seed: int = DEFAULT_SEED):
        self.rules = rules
        self.scores = dict(scores)
        self.means = dict(scores if means is None else means)
        self.time_limit_s = time_limit_s
        self.seed = seed
        self.status = "unsolved"
        self.players = {p.dk_player_id: p for p in players}
        self.by_id = self.players
        self.slot_counts = Counter(rules.slot_names)
        self._cuts = 0
        self._impossible = False       # a forced player cannot be placed

        locks = _norm_locks(locks)
        banned = set(bans)
        pinned: dict[str, str] = {}
        started: set[str] = set()
        if now is not None:
            now = _aware(now)
            started = {pid for pid, p in self.players.items() if p.is_locked(now)}
        if existing is not None:
            pairs = existing.slot_pairs() if isinstance(existing, Lineup) else list(existing)
            for slot, pid in pairs:
                if pid not in self.players:
                    raise ValueError(f"existing lineup player {pid} is not in the pool")
                if pid in started:
                    pinned[pid] = slot
        for pid, slot in locks.items():
            if pid not in self.players:
                raise ValueError(f"locked player {pid} is not in the pool")
            if pid not in self.scores:
                raise ValueError(f"locked player {self.players[pid].name} has no projection")
            if slot is not None:
                self._check_slot(pid, slot)
        for pid, slot in pinned.items():
            if pid not in self.scores:
                raise ValueError(f"fixed player {self.players[pid].name} has no projection")
            self._check_slot(pid, slot)
        forced = {**locks, **pinned}
        for pid in banned:
            if pid in forced:
                raise ValueError(f"player {pid} is both locked and banned")
        for pid in started:
            if pid not in forced:
                banned.add(pid)
        self.forced = forced
        self.banned = banned

        m = self.model = cp_model.CpModel()
        self.x: dict[str, dict[str, cp_model.IntVar]] = {}
        self.y: dict[str, Any] = {}
        names = set(self.slot_counts)
        for pid, p in self.players.items():
            if pid not in self.scores or (pid in banned and pid not in forced):
                continue
            slots = [s for s in p.slots if s in names]
            if not slots:
                continue
            self.x[pid] = {s: m.NewBoolVar(f"x_{pid}_{s}") for s in slots}
        for pid, xs in self.x.items():
            y = m.NewBoolVar(f"y_{pid}")
            m.Add(sum(xs.values()) == y)
            self.y[pid] = y
        for pid, slot in forced.items():
            if pid not in self.x or (slot is not None and slot not in self.x[pid]):
                self._impossible = True
                continue
            if slot is None:
                m.Add(self.y[pid] == 1)
            else:
                m.Add(self.x[pid][slot] == 1)
        for name, n in self.slot_counts.items():
            m.Add(sum(xs[name] for xs in self.x.values() if name in xs) == n)
        sal = [self.players[pid].salary_for(s, rules) * v
               for pid, xs in self.x.items() for s, v in xs.items()]
        used = sum(sal)
        cap = rules.salary_cap if max_salary is None else min(rules.salary_cap, max_salary)
        m.Add(used <= cap)
        if min_salary:
            m.Add(used >= min_salary)
        self._indicator_min(rules.min_games, game_key, "game")
        self._indicator_min(rules.min_teams, lambda p: p.team or f"?{p.dk_player_id}", "team")
        self._objective = sum(int(round(self.scores[pid] * rules.slot_multiplier(s) * SCALE)) * v
                              for pid, xs in self.x.items() for s, v in xs.items())
        m.Maximize(self._objective)

    # ---- construction helpers

    def _check_slot(self, pid: str, slot: str) -> None:
        p = self.players[pid]
        if slot not in self.slot_counts or slot not in p.slots:
            raise ValueError(f"{p.name} is not eligible for slot {slot}")

    def _indicator_min(self, minimum: int | None, key, label: str) -> None:
        if not minimum:
            return
        groups: dict[str, list] = defaultdict(list)
        for pid in self.x:
            groups[key(self.players[pid])].append(self.y[pid])
        inds = []
        for k, ys in groups.items():
            g = self.model.NewBoolVar(f"{label}_{k}")
            self.model.Add(g <= sum(ys))
            inds.append(g)
        self.model.Add(sum(inds) >= minimum)

    # ---- editing between solves

    def ban(self, pid: str) -> None:
        """Keep a player out of every later solve."""
        if pid in self.y:
            self.model.Add(self.y[pid] == 0)

    def _identity_terms(self, lineup: Lineup) -> list:
        """Variables that identify `lineup`: a player counts once, except that a multiplied slot
        (Showdown CPT) is part of the identity, so a captain swap is a different lineup."""
        terms = []
        for e in lineup.entries:
            pid = e.player.dk_player_id
            xs = self.x.get(pid)
            if not xs:
                continue
            mult = self.rules.slot_multiplier(e.slot)
            terms.append(sum(v for s, v in xs.items() if self.rules.slot_multiplier(s) == mult))
        return terms

    def add_no_good(self, lineup: Lineup, min_diff: int = 1) -> None:
        """Later lineups must differ from `lineup` by at least `min_diff` players:
        Σ_{i∈L} x_i ≤ roster size − d (PLAYERS.md §5.4)."""
        d = max(1, int(min_diff))
        terms = self._identity_terms(lineup)
        self.model.Add(sum(terms) <= self.rules.n_slots - d)
        self._cuts += 1

    # ---- solving

    def solve(self, time_limit_s: float | None = None) -> Lineup | None:
        """The best lineup under everything added so far, or None (see `self.status`:
        ``infeasible`` or ``unknown`` when a time limit hit before any solution)."""
        if self._impossible:
            self.status = "infeasible"
            return None
        s = cp_model.CpSolver()
        s.parameters.num_workers = 1
        s.parameters.random_seed = self.seed
        P = s.parameters
        P.linearization_level = 2      # the LP bound proves salary-knapsack optimality
        P.add_lp_constraints_lazily = False   # tuned on 125-140 player slates: Top-20 in
        P.cut_level = 0                       # about 2 s instead of 4-5 s with the defaults
        tl = time_limit_s if time_limit_s is not None else self.time_limit_s
        if tl:
            s.parameters.max_time_in_seconds = float(tl)
        st = s.Solve(self.model)
        if st == cp_model.OPTIMAL:
            self.status = "optimal"
        elif st == cp_model.FEASIBLE:
            self.status = "feasible"
        elif st == cp_model.INFEASIBLE:
            self.status = "infeasible"
            return None
        else:
            self.status = "unknown"
            return None
        pairs = [(slot, pid) for pid, xs in self.x.items() for slot, v in xs.items()
                 if s.Value(v)]
        return build_lineup(self.rules, self.players, pairs, self.scores, self.means,
                            optimal=(st == cp_model.OPTIMAL))


# ============================================================================== entry point

def optimize(pool: SlatePool | Sequence[PoolPlayer], projections: Projections,
             rules: Rules | None = None, *, mode: str = "mean", lam: float = DEFAULT_LAMBDA,
             **options) -> Lineup | None:
    """The single best lineup. `mode` is ``mean`` (maximise Σ mu) or ``cash`` (Σ mu − λ(mu − p25),
    the PLAYERS.md §5.3 fallback). `options` go to `LineupModel` (locks, bans, now, existing,
    min_salary, max_salary, time_limit_s, seed). Returns None when no legal lineup exists."""
    model = model_for(pool, projections, rules, mode=mode, lam=lam, **options)
    return model.solve()


def model_for(pool: SlatePool | Sequence[PoolPlayer], projections: Projections,
              rules: Rules | None = None, *, mode: str = "mean", lam: float = DEFAULT_LAMBDA,
              **options) -> LineupModel:
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    if rules is None:
        rules = pool.rules if isinstance(pool, SlatePool) else None
    if rules is None:
        raise ValueError("no rules: pass `rules` or a pool that carries them")
    scores, means = player_scores(players, projections, mode=mode, lam=lam)
    return LineupModel(players, rules, scores, means, **options)


# ============================================================================== persistence

def save_run(conn: sqlite3.Connection, *, slate_id: str | None, kind: str,
             model_version: str | None = None, seconds: float | None = None, ok: bool = True,
             detail: Any = None, path: str | None = None, run_id: str | None = None,
             created_at: str | None = None) -> str:
    """Insert a ``dfs_runs`` row and return its run_id. `detail` may be any JSON-able value."""
    run_id = run_id or uuid.uuid4().hex[:12]
    created_at = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if detail is not None and not isinstance(detail, str):
        detail = json.dumps(detail, default=str, sort_keys=True)
    conn.execute(
        "INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, path, "
        "seconds, ok, detail) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, None if slate_id is None else str(slate_id), kind, created_at, model_version,
         path, seconds, 1 if ok else 0, detail))
    conn.commit()
    return run_id


def save_lineups(conn: sqlite3.Connection, run_id: str, mode: str, lineups: Iterable,
                 *, contest_id: str | None = None) -> int:
    """Insert ``dfs_lineups`` rows (slots_json, salary, mean; the simulation columns stay NULL).
    `lineups` are `Lineup`s or `topn.RankedLineup`s; plain lineups are ranked by position."""
    n = 0
    for i, item in enumerate(lineups, 1):
        lu = getattr(item, "lineup", item)
        rank = getattr(item, "rank", None) or i
        conn.execute(
            "INSERT INTO dfs_lineups (run_id, mode, contest_id, rank, slots_json, salary, mean) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, mode, contest_id, rank, lu.slots_json(), lu.salary, lu.projected))
        n += 1
    conn.commit()
    return n


# ============================================================================== RotoGuru history

SALARY_FP_PER_K = 5.0          # the salary-implied baseline: 5 DK points per $1,000


def classic_rules(game_type_id: int = 70) -> Rules | None:
    """The newest archived Classic rules payload (``var/raw/draftkings/rules/*/gametype_70``).
    RotoGuru-era (2014-2021) Classic was the same 8-slot, $50,000 format with 2 games / 2 teams
    minimums, so it prices historical slates too. None when nothing is archived."""
    from ... import archive, config
    root = config.data_dir() / "raw" / "draftkings" / "rules"
    for p in sorted(root.glob(f"*/gametype_{game_type_id}.json.gz"), reverse=True):
        return Rules.from_payload(archive.load_raw(p))
    return None


def hist_dates(conn: sqlite3.Connection, *, min_players: int = 60) -> list[str]:
    """RotoGuru dates that have an on-slate pool of at least `min_players` priced players."""
    return [r[0] for r in conn.execute(
        "SELECT game_date FROM dfs_hist_salaries WHERE on_slate=1 AND salary>0 AND pos IS NOT NULL "
        "GROUP BY game_date HAVING COUNT(*) >= ? ORDER BY game_date", (min_players,))]


def hist_pool(conn: sqlite3.Connection, day: str, rules: Rules | None = None) -> SlatePool:
    """The player pool of a RotoGuru date: ``dfs_hist_salaries`` rows with ``on_slate=1`` and a
    salary. Eligibility comes from the row's position (PG -> PG/G/UTIL; a dual position such as
    PG/SG unions both) through `Rules.eligible_slots`. `dk_player_id` is the RotoGuru id.
    There are no start times, so nothing is locked."""
    rules = rules or classic_rules()
    if rules is None:
        raise LookupError("no archived Classic rules payload; pass `rules`")
    rows = conn.execute(
        "SELECT rg_id, name, pos, team, opp, home, starter, salary FROM dfs_hist_salaries "
        "WHERE game_date=? AND on_slate=1 AND salary>0 AND pos IS NOT NULL ORDER BY rg_id",
        (day,)).fetchall()
    if not rows:
        raise LookupError(f"no on-slate dfs_hist_salaries rows for {day}")
    players = []
    for rg_id, name, pos, team, opp, home, starter, salary in rows:
        positions = tuple(p for p in pos.split("/") if p)
        slots = rules.eligible_slots(positions)
        if not slots:
            continue
        h, a = (team, opp) if home else (opp, team)
        gk = f"{day}:{'-'.join(sorted([team or '?', opp or '?']))}"
        players.append(PoolPlayer(
            dk_player_id=str(rg_id), dk_dk_id=None, name=name, team=team, positions=positions,
            salary=int(salary), slots=slots, salary_by_slot={s: int(salary) for s in slots},
            draftable_ids={}, game_id=gk, competition_id=None, game_label=f"{a} @ {h}",
            home=h, away=a, start_time=None, status=None, news_status=None,
            starter=None if starter is None else bool(starter), swappable=None, fppg=None))
    return SlatePool(slate_id=f"rg:{day}", game_type_id=rules.game_type_id,
                     game_type="classic", captured_at=day, rules=rules, players=tuple(players))


def hist_projections(conn: sqlite3.Connection, pool: SlatePool, kind: str = "actual"
                     ) -> dict[str, Projection]:
    """``actual``: the real DK points (hindsight-optimal). ``salary``: the salary-implied
    baseline, 5 points per $1,000. Keyed by RotoGuru id."""
    day = pool.captured_at
    if kind == "salary":
        return {p.dk_player_id: Projection(round(SALARY_FP_PER_K * p.salary / 1000, 2))
                for p in pool.players}
    if kind in ("actual", "rotoguru-actual"):
        have = {str(r[0]): r[1] for r in conn.execute(
            "SELECT rg_id, dk_pts FROM dfs_hist_salaries WHERE game_date=? AND dk_pts IS NOT NULL",
            (day,))}
        return {p.dk_player_id: Projection(have[p.dk_player_id])
                for p in pool.players if p.dk_player_id in have}
    raise ValueError(f"unknown historical projection {kind!r} (actual|salary)")
