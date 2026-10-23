"""Cash lineup by P(beat the cash line); Top N by mean, then P(cash) (IMPROVEMENTS.md D2).

Switch ``COURTSIDE_CASH_PCASH`` (`CASH_PCASH_ENV`; default OFF, `CASH_PCASH_DEFAULT`). OFF, the
runner is unchanged: the cash lineup is mu - 0.3 (mu - p25) (`runner.CASH_LAMBDA`, the
PLAYERS.md §5.3 fallback) and Top N is in solve order. ON (version `VERSION`, written to
``dfs_runs.detail["cash_pcash"]``), after the usual solves (`choose`):

1. **Candidates**: the Top N plus solves of mu - λ (mu - p25) on `LAMBDA_GRID` (λ = 0 is the
   Top N #1 and λ = 0.3 the OFF cash lineup, so only 0.15 and 0.5 are new solves), deduplicated.
2. **Field**: the slate's double-up reference field, exactly as `field.score_run` builds it
   (`default_contests`, `field_for(n=10_000, seed=0)`) on the calibrated truth
   (`portfolio.truth_sims`, same seed). The field is kept and handed to the run's `score_run`
   (`reuse_fields`), so it is built once per run.
3. **Split** (`portfolio.split_rows`, mode "gs", the contest builder's report split): the
   truth's rows are permuted (seed `SPLIT_SEED`); the pick is made on S (55 %), its P(cash) is
   reported on the disjoint R (45 %), so the reported number is not the max of ~14 noisy ones.
4. **Pick**: the candidate with the highest P(score >= cash line) on S (ties: the OFF cash lineup,
   then candidate order).
5. **Top N**: same lineups, ranked by mean with near-ties (<= `topn.NEAR_TIE_DK` = 0.5 DK of the
   group's leader) broken by P(cash) on S (`topn.rank_by_mean`, PLAYERS.md §5.4).

It never fails a run: any error (or no sims: the FPPG fallback) keeps the OFF lineups and records
the status. Late swap and the swap candidates still re-solve the cash lineup under λ = 0.3.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any

import numpy as np

log = logging.getLogger("courtside.proj.dfs.cash_pcash")

VERSION = "cash-pcash-1"
CASH_PCASH_ENV = "COURTSIDE_CASH_PCASH"
CASH_PCASH_DEFAULT = "off"            # "off" | "on" (env overrides, read at import)
DETAIL_KEY = "cash_pcash"
LAMBDA_GRID = (0.0, 0.15, 0.3, 0.5)
RULE_LAMBDA = 0.3                     # runner.CASH_LAMBDA: the OFF cash lineup
SPLIT_SEED = 0                        # split_rows(S, SPLIT_SEED, "gs") (adds portfolio.SPLIT_SEED)
FIELD_N = 10_000                      # field.score_run's n_field
FIELD_SEED = 0                        # field.score_run's seed


def pcash_mode(v: Any, default: str = "off") -> str:
    """'on' | 'off' from an env string / bool: 1 true yes on -> 'on', 0 false no off -> 'off',
    anything else (or None) -> `default`."""
    if v is None:
        return default
    if isinstance(v, bool):
        return "on" if v else "off"
    x = str(v).strip().lower()
    if x in ("1", "true", "yes", "on"):
        return "on"
    if x in ("0", "false", "no", "off"):
        return "off"
    return default


CASH_PCASH = pcash_mode(os.environ.get(CASH_PCASH_ENV), CASH_PCASH_DEFAULT) == "on"


def set_cash_pcash(on: bool | str) -> None:
    """Switch for later runs (the env is read once at import)."""
    global CASH_PCASH
    CASH_PCASH = pcash_mode(on) == "on"


# ============================================================================== field reuse

_PENDING: dict[str, tuple[tuple, Any]] = {}     # run_id -> (field_for key, Field)


def _field_key(run_id, contest, n, mix, n_solves, seed, kw=None) -> tuple:
    extra = tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
                         for k, v in (kw or {}).items()))
    return (str(run_id), contest.contest_id, contest.name, int(contest.field_size),
            contest.archetype, n, mix, n_solves, seed, extra)


@contextlib.contextmanager
def reuse_fields(run_id: str):
    """While open, `field.field_for` returns the cash field `choose` built for `run_id` when
    called with the same arguments (the run's `score_run`), else builds as usual. A no-op when
    nothing is pending (switch OFF, or `choose` did not get as far as the field)."""
    hit = _PENDING.get(str(run_id))
    if hit is None:
        yield
        return
    from . import field as F
    orig = F.field_for

    def field_for(ctx, contest, *, n=None, mix=None, n_solves=F.N_SOLVES, seed=0, **kw):
        key = _field_key((ctx.get("info") or {}).get("run_id"), contest, n, mix, n_solves, seed,
                         kw)
        if key == hit[0]:
            return hit[1]
        return orig(ctx, contest, n=n, mix=mix, n_solves=n_solves, seed=seed, **kw)

    F.field_for = field_for
    try:
        yield
    finally:
        F.field_for = orig
        _PENDING.pop(str(run_id), None)


# ============================================================================== candidates

def _ident(rules, lu) -> frozenset:
    """A lineup's identity: its players, a multiplied slot (Showdown CPT) kept apart."""
    return frozenset((rules.slot_multiplier(s), p) for s, p in lu.slot_pairs())


def candidates(O, pool, projs, rules, bans, now, cash, res, *, solve_limit_s: float | None = None,
               grid=LAMBDA_GRID) -> list[tuple[str, Any]]:
    """[(source, Lineup)]: the OFF cash lineup ('lam0.3'), the Top N ('topn<k>') and the other
    λ-grid solves, first occurrence kept. λ = 0 is the mean objective, i.e. Top N #1 (solved only
    when the Top N is empty)."""
    out: list[tuple[str, Any]] = []
    seen: set = set()

    def add(src, lu):
        if lu is None:
            return
        k = _ident(rules, lu)
        if k not in seen:
            seen.add(k)
            out.append((src, lu))

    add(f"lam{RULE_LAMBDA:g}", cash)
    for r in res:
        add(f"topn{r.rank}", r.lineup)
    for lam in grid:
        if abs(lam - RULE_LAMBDA) < 1e-12 or (lam == 0.0 and len(res)):
            continue
        try:
            lu = O.optimize(pool, projs, rules, mode="cash", lam=float(lam), bans=bans, now=now,
                            time_limit_s=solve_limit_s)
        except Exception:  # noqa: BLE001
            log.exception("cash grid solve lam=%s failed", lam)
            lu = None
        add(f"lam{lam:g}", lu)
    return out


# ============================================================================== scoring

def cash_field(conn, slate_id, run_id: str, *, contest=None):
    """(ctx, contest, Field, truth fp) for a run, as `field.score_run` builds its cash part.
    None when the run has no sims (FPPG fallback)."""
    from . import field as F
    from . import portfolio as PF
    ctx = F.prepare(conn, slate_id, run_id, n_sims=None, seed=FIELD_SEED)
    if ctx["sims"] is None:
        return None
    contest = contest or F.default_contests(conn, slate_id)[0]
    f = F.field_for(ctx, contest, n=FIELD_N, mix=None, n_solves=F.N_SOLVES, seed=FIELD_SEED,
                    **field_kw(conn, run_id))
    fp, _ = F.align_sims(ctx["sims"], f.pids, None, FIELD_SEED)
    fp, _ = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                          np.random.default_rng(FIELD_SEED + 7))
    return ctx, contest, f, fp


def field_kw(conn, run_id: str) -> dict:
    """The extra `field_for` arguments `score_run` passes (D5 ownership uncertainty, when its
    COURTSIDE_FIELD_K switch is on), so the reused field is the one score_run would build."""
    from . import field as F
    fk = getattr(F, "field_k", None)
    unc = getattr(F, "run_own_uncertainty", None)
    if fk is None or unc is None or not fk():
        return {}
    return {"own_unc": unc(conn, run_id)}


def split(n_rows: int, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(select rows S, report rows R): the contest builder's "gs" split of the truth rows."""
    from . import portfolio as PF
    rows = PF.split_rows(n_rows, seed, "gs")
    return rows["S"], rows["R"]


def score(f, fp: np.ndarray, contest, lineups, rows_s, rows_r, *, seed: int = FIELD_SEED
          ) -> tuple[list[float], list[float]]:
    """P(cash) of each lineup on the select rows and on the report rows."""
    from . import field as F
    pairs = [lu.slot_pairs() for lu in lineups]
    es = F.evaluate(f, fp[rows_s], contest, pairs, seed=seed, tiers=False)
    er = F.evaluate(f, fp[rows_r], contest, pairs, seed=seed + 1, tiers=False)
    return ([float(x["p_cash"]) for x in es.lineups], [float(x["p_cash"]) for x in er.lineups])


def pick(cands: list[tuple[str, Any]], p_sel: list[float]) -> int:
    """Index of the max P(cash) on S; ties go to the earliest candidate (the OFF cash lineup is
    first)."""
    best = 0
    for i, p in enumerate(p_sel):
        if p > p_sel[best] + 1e-12:
            best = i
    return best


def choose(conn, slate_id, run_id: str, O, T, pool, projs, rules, bans, now, cash, res, *,
           solve_limit_s: float | None = None) -> tuple[Any, Any, dict]:
    """(cash lineup, Top N result, detail) with the switch ON (module docstring)."""
    t0 = time.perf_counter()
    info: dict = {"version": VERSION, "status": "ok"}
    built = cash_field(conn, slate_id, run_id)
    if built is None:
        return cash, res, {**info, "status": "no_sims"}
    _, contest, f, fp = built
    _PENDING[str(run_id)] = (_field_key(run_id, contest, FIELD_N, None, _n_solves(), FIELD_SEED,
                                        field_kw(conn, run_id)), f)
    t1 = time.perf_counter()
    cands = candidates(O, pool, projs, rules, bans, now, cash, res, solve_limit_s=solve_limit_s)
    if not cands:
        return cash, res, {**info, "status": "no_candidates"}
    t2 = time.perf_counter()
    rs, rr = split(fp.shape[0])
    p_s, p_r = score(f, fp, contest, [lu for _, lu in cands], rs, rr)
    k = pick(cands, p_s)
    src = {i: s for i, (s, _) in enumerate(cands)}
    rule_i = 0 if cash is not None else None
    by_ident = {_ident(rules, lu): i for i, (_, lu) in enumerate(cands)}
    p_top = [p_s[by_ident[_ident(rules, r.lineup)]] for r in res]
    new_res = T.rank_by_mean(res, p_top) if len(res) else res
    old_rank = {_ident(rules, r.lineup): r.rank for r in res}
    info.update({
        "contest": {"contest_id": contest.contest_id, "name": contest.name,
                    "field_size": contest.field_size, "paid": contest.paid,
                    "source": contest.source},
        "n_select": len(rs), "n_report": len(rr), "lambda_grid": list(LAMBDA_GRID),
        "candidates": [{"src": s, "mean": round(lu.projected, 2), "p_cash_select": round(p_s[i], 4)}
                       for i, (s, lu) in enumerate(cands)],
        "pick": src[k], "changed": k != rule_i,
        "p_cash_report": round(p_r[k], 4),
        "p_cash_report_rule": round(p_r[rule_i], 4) if rule_i is not None else None,
        "topn_order": [old_rank[_ident(rules, r.lineup)] for r in new_res],
        "seconds": {"field": round(t1 - t0, 2), "solves": round(t2 - t1, 2),
                    "score": round(time.perf_counter() - t2, 2),
                    "total": round(time.perf_counter() - t0, 2)}})
    return cands[k][1], new_res, info


def _n_solves() -> int:
    from . import field as F
    return F.N_SOLVES


def choose_safe(conn, slate_id, run_id: str, O, T, pool, projs, rules, bans, now, cash, res,
                **kw) -> tuple[Any, Any, dict]:
    """`choose` that never raises: on error the OFF lineups are kept."""
    try:
        return choose(conn, slate_id, run_id, O, T, pool, projs, rules, bans, now, cash, res,
                      **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("cash P(cash) pick failed for run %s", run_id)
        return cash, res, {"version": VERSION, "status": f"error: {type(e).__name__}: {e}"}
