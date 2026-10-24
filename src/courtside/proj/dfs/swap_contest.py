"""Contest-aware late swap (docs/IMPROVEMENTS.md D8; docs/research/d8-d9.md).

Switch ``COURTSIDE_SWAP_CONTEST`` (default OFF; `set_contest_swap`, or ``swap.plan(contest_aware=
...)``). OFF, `swap.plan` re-solves every entry's open slots by its mean (cash: mu - 0.3 (mu -
p25)), which strips the leverage and diversity a contest-builder entry was built with. ON, each
entry's open slots are chosen by the contest it is in:

* **sims**: the projection run's stored sims, scored as the contest builder scores them
  (`portfolio.truth_sims`), conditioned on what is known at swap time:
  - a now-known Out (DK status O, or a late-news change to Out): only the sim rows where the run
    drew him out (P10 scenario arrays, `store.read_scenarios`), so his teammates' minutes and
    points are those of that scenario. Skipped for a player when it would leave fewer than
    MIN_ROWS rows, or the run has no scenario arrays (runs before 2026-10-23); his column is
    zeroed either way;
  - locked players (game started): their realised points when given (`locked_points`, the same
    value in every row). The live swap has no in-game feed, so live they keep their sims;
* **field**: the contest's medium field proxy (`portfolio.strength_field`), on the same rows;
* **candidates** per entry: the mean solve (what OFF would pick), plus N_SCEN scenario solves
  w x score + (1 - w) x FP_s (as the builder's, w = SCEN_W, s a random conditioned row) under the
  same locks, bans, cuts and caps, plus the entry as it stands;
* **objective**: GPP, the joint E[payout] of all our entries in the contest (the others as they
  stand or as already swapped; the builder's greedy marginal, `portfolio.greedy`); cash,
  P(rank <= paid places) (a ladder paying 1 to every paid place);
* an entry changes only when a ruled-out player has to go, or when the best candidate beats the
  entry as it stands by more than GAIN_Z paired SEs on the rows;
* **contest** per entry: its Contest ID when captured; else (our saved lineups) the slate's
  largest captured double-up (cash) or GPP (Top N), `field.default_contests` (presets if none).

Version ``swap-contest-1`` in the plan's summary (``contest``). Nothing here talks to DraftKings.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import field as F
from . import portfolio as PF

log = logging.getLogger("courtside.proj.dfs.swap_contest")

ENV = "COURTSIDE_SWAP_CONTEST"
DEFAULT = True                 # shipped 2026-10-24 (owner OK; the web swap runs in the background)
VERSION = "swap-contest-1"
TAG = "+cs"
N_SCEN = 16                    # scenario solves per entry
SCEN_W = 0.5                   # portfolio.Settings.scen_w
N_SIMS = 10_000                # sim rows scored (the first N_SIMS of the conditioned rows)
MIN_ROWS = 1_000               # conditioned rows needed to keep conditioning on an Out
GAIN_Z = 1.0                   # change an entry only for a gain > GAIN_Z paired SEs
N_PASSES = 2                   # coordinate passes over a contest's entries
FIELD_SOLVES = 80              # the field proxy's optimizer-like solves (builder: 200)
TIME_LIMIT_S = 120.0           # the whole contest-aware plan (OFF: swap.TOTAL_LIMIT_S)


def _on(v: Any, default: bool) -> bool:
    if v is None:
        return default
    x = str(v).strip().lower()
    if x in ("1", "true", "yes", "on"):
        return True
    if x in ("0", "false", "no", "off", ""):
        return False
    return default


CONTEST_SWAP = _on(os.environ.get(ENV), DEFAULT)


def contest_swap() -> bool:
    return CONTEST_SWAP


def set_contest_swap(on: Any) -> None:
    """Set the switch for later plans (the env is read once at import)."""
    global CONTEST_SWAP
    CONTEST_SWAP = _on(on, False) if not isinstance(on, bool) else on


def cash_ladder(lad: PF.Ladder) -> PF.Ladder:
    """A ladder paying 1 to every paid place: its E[payout] is P(cash) (ties split)."""
    pay = np.zeros(lad.M + 1)
    pay[:lad.paid] = 1.0
    out = PF.Ladder.__new__(PF.Ladder)
    out.__dict__.update(lad.__dict__)
    out.cum = np.concatenate([[0.0], np.cumsum(pay)])
    return out


# ============================================================================== sims

def known_outs(pool_players: Iterable, news: Iterable[Mapping]) -> set[str]:
    """DK ids known Out at swap time: DK status O / OUT, or a late-news change to Out."""
    from .swap import _news_is_out
    out = {p.dk_player_id for p in pool_players
           if str(p.status or "").upper() in F.OUT_STATUSES}
    out |= {str(n["player_id"]) for n in news if n.get("player_id") is not None
            and _news_is_out(n)}
    return out


def condition_rows(sims: Mapping[str, np.ndarray], outs: Iterable[str],
                   rows: np.ndarray | None = None, min_rows: int = MIN_ROWS
                   ) -> tuple[np.ndarray, dict]:
    """(row indices, info): the stored sim rows (of `rows`, default all) consistent with every
    known Out whose drawn-out rows keep at least `min_rows` (players in the order given, sorted
    by id; one that would leave fewer is skipped). No scenario arrays: `rows` unchanged."""
    from .. import store as SO
    S = sims["fp"].shape[0]
    mask = np.zeros(S, bool)
    mask[np.arange(S) if rows is None else np.asarray(rows)] = True
    info: dict = {"scenarios": False, "conditioned": [], "skipped": [], "rows_in": int(mask.sum())}
    sc = SO.read_scenarios(sims)
    if sc is None:
        info["rows"] = int(mask.sum())
        return np.flatnonzero(mask), info
    info["scenarios"] = True
    col = {}
    for j, d in enumerate(str(x) for x in sims["dk_id"]):
        col.setdefault(d, j)
    for pid in sorted(set(outs)):
        j = col.get(str(pid))
        if j is None:
            continue
        m = mask & sc.absent[:, j]
        if m.sum() == mask.sum():
            continue                                    # already out in every kept row
        if m.sum() >= min_rows:
            mask = m
            info["conditioned"].append({"player_id": pid, "rows": int(m.sum())})
        else:
            info["skipped"].append({"player_id": pid, "rows": int(m.sum())})
    info["rows"] = int(mask.sum())
    return np.flatnonzero(mask), info


@dataclass
class ContestSims:
    """One contest's field and scoring truth on the conditioned rows."""
    contest: F.Contest
    field: Any
    fp: np.ndarray                    # (S x field.pids) float32
    col: dict[str, int]
    lad: PF.Ladder


@dataclass
class Context:
    """Everything the contest-aware choice needs for one plan (built once, `prepare`)."""
    run_id: str
    pool: Any                         # the run's slate pool (field + ownership inputs)
    inputs: Any
    sims: dict                        # stored sims restricted to the conditioned rows
    rows_info: dict
    outs: set[str]
    locked_points: dict[str, float]
    blended: bool | None
    seed: int = 0
    contests: dict = field(default_factory=dict)        # key -> ContestSims
    defaults: tuple | None = None
    conn: Any = None
    info: dict = field(default_factory=dict)
    deadline: float = math.inf

    def contest_for(self, contest_id: str | None, mode: str) -> F.Contest:
        """The entry's captured contest, else the slate's default for its objective."""
        if contest_id:
            c = F.load_contest(self.conn, str(contest_id))
            if c is not None and c.field_size > 1:
                return c
        if self.defaults is None:
            slate = self.info.get("slate_id")
            self.defaults = F.default_contests(self.conn, slate)
        return self.defaults[0] if mode == "cash" else self.defaults[1]

    def get(self, contest: F.Contest) -> ContestSims:
        key = (contest.contest_id, contest.name, contest.field_size)
        cs = self.contests.get(key)
        if cs is not None:
            return cs
        t = time.perf_counter()
        players, rules = self.pool.players, self.pool.rules
        s = PF.Settings(n=1, seed=self.seed, field_solves=FIELD_SOLVES, threads=2)
        fld, _ = PF.strength_field(players, self.inputs, contest, rules, "medium", s,
                                   seed=self.seed)
        fp, _ = F.align_sims(self.sims, fld.pids)
        fp, _ = PF.truth_sims(fp, fld.pids, self.inputs, s.beta, s.tau,
                              np.random.default_rng(self.seed + 7), blended=self.blended)
        col = {p: j for j, p in enumerate(fld.pids)}
        fp = np.array(fp, np.float32, copy=True)
        for pid, v in self.locked_points.items():
            if pid in col:
                fp[:, col[pid]] = np.float32(v)
        for pid in self.outs:
            if pid in col and pid not in self.locked_points:
                fp[:, col[pid]] = 0.0
        cs = ContestSims(contest, fld, fp, col, PF.Ladder(contest))
        self.contests[key] = cs
        self.info.setdefault("fields", []).append(
            {"contest": contest.contest_id, "name": contest.name, "archetype": contest.archetype,
             "seconds": round(time.perf_counter() - t, 1)})
        return cs


def prepare(conn: sqlite3.Connection, slate_id, run_id: str | None, *, news: Sequence[Mapping],
            pool_players: Iterable, locked_points: Mapping[str, float] | None = None,
            rows: np.ndarray | None = None, n_sims: int | None = N_SIMS, seed: int = 0,
            time_limit_s: float = TIME_LIMIT_S) -> tuple[Context | None, str | None]:
    """(Context, None) or (None, why) when the run has no stored sims."""
    from .. import role_blend as RB
    from .. import store as SO
    if not run_id:
        return None, "no stored projection run"
    pool, inputs, info = F.slate_inputs(conn, slate_id, run_id)
    path = info.get("sim_path")
    if path and not os.path.exists(str(path)):       # a copied data dir: same layout under it
        alt = SO.sims_dir() / str(slate_id) / os.path.basename(str(path))
        path = str(alt) if alt.exists() else None
    if not path:
        return None, "the projection run has no stored sims"
    z = SO.load_sims(path)
    outs = known_outs(pool_players, news)
    idx, rinfo = condition_rows(z, outs, rows)
    if n_sims and len(idx) > n_sims:
        idx = idx[:n_sims]
    rinfo["used"] = int(len(idx))
    sims = {"fp": z["fp"][idx], "dk_id": z["dk_id"]}
    ctx = Context(run_id=run_id, pool=pool, inputs=inputs, sims=sims, rows_info=rinfo,
                  outs=outs, locked_points={str(k): float(v) for k, v in
                                            (locked_points or {}).items()},
                  blended=RB.run_blended(conn, run_id), seed=seed, conn=conn,
                  info={"slate_id": str(slate_id)},
                  deadline=time.perf_counter() + time_limit_s)
    return ctx, None


# ============================================================================== choice

def scenario_scores(ctx: Context, cs: ContestSims, base: Mapping[str, float], k: int,
                    rng: np.random.Generator) -> list[dict[str, float]]:
    """`k` scenario objectives w x base + (1 - w) x FP_s on random conditioned rows (players
    without a sims column keep their base score)."""
    ids = list(base)
    b = np.array([base[i] for i in ids], float)
    cols = np.array([cs.col.get(i, -1) for i in ids])
    out = []
    for _ in range(k):
        r = int(rng.integers(0, cs.fp.shape[0]))
        x = np.where(cols >= 0, cs.fp[r, np.maximum(cols, 0)].astype(float), b)
        out.append(dict(zip(ids, SCEN_W * b + (1 - SCEN_W) * x)))
    return out


def _pairs(lu) -> list[tuple[str, str]]:
    return lu.slot_pairs() if hasattr(lu, "slot_pairs") else [(s, str(p)) for s, p in lu]


@dataclass
class Pending:
    """One target (a set of `copies` identical entries) waiting for its contest-aware choice."""
    gk: tuple
    mode: str                          # 'cash' | 'mean'
    contest: F.Contest
    copies: int
    before: list                       # [(slot, id)] as it stands
    keep_ok: bool                      # it may stay as it is (legal, nobody ruled out)
    cands: list                        # [Lineup]: the mean solve first, then scenario solves
    group: str = ""


def _ok(a: Sequence, others: Sequence[Sequence], n_slots: int, min_diff: int,
        max_overlap: int | None) -> bool:
    ids = {p for _, p in a}
    for o in others:
        sh = len(ids & {p for _, p in o})
        if sh > n_slots - max(1, int(min_diff)) or (max_overlap is not None and sh > max_overlap):
            return False
    return True


def _marginal_pick(lad: PF.Ladder, P: PF._Pool, others: list[int], cols: list[int]) -> int:
    """The column of `cols` with the largest joint marginal E[payout] next to `others` (the
    builder's greedy step, `portfolio.greedy`)."""
    idx = list(others) + list(cols)
    sub = PF._Pool(P.score[:, idx], P.above[:, idx], P.tied[:, idx])
    k = len(others)
    inc = np.zeros((len(idx), 1), np.int16)
    got, _ = PF.greedy(lad, PF.CandSet(sub, inc), k + 1, gamma=8, cap=np.array([10 ** 6]),
                       low=np.array([0]), start=list(range(k)))
    return cols[got[-1] - k] if len(got) > k else cols[0]


def _joint(lad: PF.Ladder, P: PF._Pool, cols: list[int]) -> np.ndarray:
    pay, _ = PF.joint_payouts(lad, P.score[:, cols], P.above[:, cols], P.tied[:, cols])
    return pay


def decide(ctx: Context, items: Sequence[Pending], fixed: Mapping[str, Sequence], *,
           n_slots: int, min_diff: int = 1, max_overlap: int | None = None) -> dict:
    """{gk: result} for the pending targets, contest by contest: our entries in the contest that
    are not re-solved (`fixed` {group: [(pairs, copies)]}) stay as they are; the targets are
    chosen one after another (plan order), each given the others' current lineups, by the
    builder's greedy marginal (GPP: joint E[payout]; cash: P(cash) via `cash_ladder`) among its
    candidates that keep `min_diff` / `max_overlap` against the others (N_PASSES passes; the
    start is each entry as it stands, or its mean solve when it cannot stay). A target that may
    stay as it is changes only for a gain > GAIN_Z paired SEs. result: {"lineup": Lineup |
    None (keep), "objective", "value", "keep_value", "gain", "se", "contest", ...}."""
    by_c: dict[tuple, list[Pending]] = {}
    for it in items:
        by_c.setdefault((it.contest.contest_id, it.contest.name, it.contest.field_size, it.mode,
                         it.group), []).append(it)
    out: dict = {}
    for key, its in by_c.items():
        contest, mode = its[0].contest, its[0].mode
        cs = ctx.get(contest)
        lad = cs.lad if mode != "cash" else cash_ladder(cs.lad)
        lus: list = []
        at: dict = {}

        def col(pairs) -> int:
            """The column of a lineup; identical lineups share one (same sims, same dupes)."""
            key = tuple(sorted((str(sl), str(p)) for sl, p in pairs))
            if key not in at:
                at[key] = len(lus)
                lus.append(list(pairs))
            return at[key]
        fx_cols: list[int] = []
        for pairs, copies in fixed.get(its[0].group, ()):
            fx_cols += [col(pairs)] * int(copies)
        bcol, ccols = {}, {}
        for it in its:
            if all(p for _, p in it.before):
                bcol[it.gk] = col(it.before)
            ccols[it.gk] = [col(_pairs(c)) for c in it.cands]
        n_ours = len(fx_cols) + sum(it.copies for it in its)
        ev = F.evaluate(cs.field, cs.fp, contest, lus, seed=ctx.seed, tiers=False,
                        own_entries=n_ours, per_sim=True)
        P = PF.dupe_pool(cs.field, ev, lus, contest, max(1, contest.field_size - n_ours),
                         np.random.default_rng(ctx.seed + 3))
        cur = {it.gk: (bcol[it.gk] if it.keep_ok and it.gk in bcol else ccols[it.gk][0])
               for it in its}
        for it in [x for _ in range(N_PASSES) for x in its]:
            others = fx_cols + [cur[o.gk] for o in its if o.gk != it.gk for _ in range(o.copies)]
            other_lus = [lus[c] for c in set(others)]
            cols = [c for c in ccols[it.gk]
                    if _ok(lus[c], other_lus, n_slots, min_diff, max_overlap)] or ccols[it.gk][:1]
            best = _marginal_pick(lad, P, others, cols)
            pb = _joint(lad, P, others + [best] * it.copies)
            tb = pb.sum(1)
            # shown: GPP the joint E[payout] of our entries in the contest; cash the entry's own
            # chance to cash. The gain test uses the joint totals either way.
            div = it.copies if mode == "cash" else 1
            own = (lambda pay: pay[:, len(others):].sum(1) / div) if mode == "cash" else \
                (lambda pay: pay.sum(1))
            res = {"objective": "p_cash" if mode == "cash" else "e_payout",
                   "contest": contest.contest_id, "contest_name": contest.name,
                   "value": float(own(pb).mean()), "n_sims": int(len(tb)),
                   "candidates": len(it.cands), "entries": n_ours}
            choice = best
            if it.keep_ok and it.gk in bcol:
                pk = _joint(lad, P, others + [bcol[it.gk]] * it.copies)
                d = (tb - pk.sum(1)) / div
                res.update(keep_value=float(own(pk).mean()), gain=float(d.mean()),
                           se=float(d.std() / math.sqrt(len(d))) if len(d) > 1 else 0.0)
                if not (res["gain"] > 0 and res["gain"] > GAIN_Z * res["se"]):
                    choice = bcol[it.gk]
            cur[it.gk] = choice
            res["lineup"] = None if choice == bcol.get(it.gk) else \
                it.cands[ccols[it.gk].index(choice)]
            res["candidates_distinct"] = len(set(ccols[it.gk]))
            out[it.gk] = res
    return out


def note(res: Mapping) -> str:
    """One plain line for the entry's notes."""
    if res.get("objective") == "p_cash":
        txt = f"chance to cash {res['value']:.0%}"
        if res.get("gain") is not None:
            txt += f" ({res['gain'] * 100:+.1f} ± {res['se'] * 100:.1f} pt vs keeping it)"
    else:
        txt = f"simulated E[payout] of our entries ${res['value']:,.2f}"
        if res.get("gain") is not None:
            txt += f" ({res['gain']:+,.2f} ± {res['se']:,.2f} vs keeping it)"
    kept = res.get("lineup") is None
    return (f"Contest-aware ({res.get('contest_name') or 'contest'}): "
            + ("kept, the change does not clear the noise; " if kept and res.get("gain") is not None
               else "") + txt + ".")
