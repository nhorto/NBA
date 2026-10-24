"""Ownership proxy and field proxy (docs/PLAYERS.md §5.3, §5.5, S12; task T2.11).

**Everything here is a PROXY.** No real ownership exists until the owner uploads DK standings
CSVs (T3.2, `standings.ownership_dataset`). Until >= 10 slates are uploaded, the ownership
numbers are hand-set from public DFS knowledge and shown as bands; the field and every number
computed against it (cash line, P(cash), top-1 %, expected payout, dupes) is "simulated against
a modelled field, not real results".

Four parts:

1. **Ownership proxy** (`ownership_proxy`). A public-information utility per player
       u_i = conc * ( b_value*z(value_i) + (b_proj + b_small*small)*z(proj_i) + b_total*z(total_i)
                      + b_repl*replacement_i + b_q*questionable_i + b_bench*non_starter_i )
   from the inputs the public uses: DK FPPG blended with our projection (the "public
   projection"), its value per $1k, the team implied total (line_snapshots), DK status, DK
   starter flag, an injury-replacement flag (our projected minutes >= 6 above his recent
   minutes) and slate size (`small` grows as the slate shrinks: stars get rostered more on small
   slates). The allocation is a softmax per roster slot, balanced Sinkhorn-style so that every
   slot column sums to its count (Σ own = 800 % Classic, 600 % Showdown with the CPT column as
   the captain split) and no player exceeds `row_cap`. A salary "shadow price" λ (bisection)
   makes the expected lineup spend Σ own_i*salary_i equal the field's spend (SPEND, 99.7 % of
   the cap), so the proxy is affordable on average. `conc` depends on the contest archetype
   (cash is much more concentrated). `fit_from_uploads` re-estimates the b's from real %Drafted.

2. **Field proxy** (`build_field`). F lineups (F up to the field size, or a 10-20k sample for big
   GPPs; each sampled lineup is a weight of M/F entries), a labelled mixture (S12):
   * `mix` (default: the archetype's, 0.40 / 0.20 large GPP) **optimizer-like**: `n_solves`
     CP-SAT solves on the public
     projection x exp(N(0, σ²)), resampled with replacement (reproduces optimizer dupes);
   * 1 - mix **ownership-sampled**: slots filled in scarcity order from the per-slot ownership
     allocation, with salary feasibility at every step and a per-lineup spend floor drawn from the
     salary-use behaviour (most of the field spends >= 99 % of the cap); invalid lineups (games /
     teams minimums) are dropped. The per-slot weights get a few iterative-proportional-fitting
     steps so that the *whole field's* ownership reproduces the proxy where the optimizer part
     allows it.
   Archetypes (`ARCHETYPES`): cash (more concentrated ownership, low optimizer noise so entries
   duplicate as in real double-ups), small GPP, large GPP (a smaller optimizer share: real big
   GPPs are mostly unique lineups); all spend ~99.7 % of the cap on average. Calibrated in
   direction and scale on the 2026-10-20 standings (field-proxy-1.1), not fitted.
   The default mix is the archetype's (`mix=None` / "medium"); "soft" 20/80 and "sharp" 70/30
   (`STRENGTHS`) remain for explicit use. The contest builder varies the field's information
   (`portfolio.STRENGTH_W`), not the mix.

3. **Scoring against the sims** (`evaluate`). Player DK points of the projection run (npz, S x n)
   are quantised to 1/32 point so lineup sums are exact (dupes tie exactly); field scores are one
   BLAS matmul per chunk of sims, sorted once per sim. Per sim: the cash line c_s (score of the
   last paid rank among the M-1 other entries), the top-1 % line and the line at every payout-tier
   boundary. Per candidate lineup: P(cash) = mean_s[score_s >= c_s], P(top 1 %), E[payout] (rank
   within the field per sim -> payout ladder, ties split; small fields draw M-1 opponents per sim,
   big fields scale the sample with a Beta-binomial rank draw) and E[dupes] (expected identical
   lineups among the other entries: exact for the optimizer part, a calibrated product of
   ownerships for the sampled part, §5.5.6).

4. **Store** (`score_run`): `dfs_ownership_proxy` for the run and P(cash) / P(top 1 %) /
   E[payout] / dupes on the run's cash + Top N rows of `dfs_lineups`.

CLI: `courtside dfs field --slate ID [--contest ID] [--n 5000]` (`add_commands`);
`courtside dfs field-calib [--fit [--save]]`: what the uploaded standings say (partial calibration
report: real lineup structure and concentration vs the proxy, the cash / GPP concentration
exponent per slate) and, from MIN_FIT_SLATES slates, the ownership refit. The field-at-actual
comparison (cash line, top-1 % line vs the real standings) is research/real_standings_calib.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .rules import PoolPlayer, Rules, SlatePool

log = logging.getLogger(__name__)

LABEL = "proxy"
MODEL = "own-proxy-1.2"              # 1.2: cash row cap 0.97; 1.1: per-archetype conc + spend
HANDSET_VERSION = f"{MODEL}-handset"
FIELD_VERSION = "field-proxy-1.2"    # 1.2: cash row cap 0.97; 1.1: cash noise, salary use
MIN_FIT_SLATES = 10                 # §9: show bands until >= 10 calibrated slates
PARAMS_FILE = "dfs_ownership_params.json"

# Hand-set utility weights (logit scale, per SD of the slate). Public DFS knowledge, not fitted:
# value per $1k is the main driver of ownership (Haugh & Singal; every ownership article), raw
# projection pulls stars, small slates concentrate on stars, injury replacements and high team
# totals draw chalk, Q / GTD players and confirmed bench players are faded.
DEFAULT_PARAMS: dict[str, float] = {
    "b_value": 1.10,        # z(public value per $1k)
    "b_proj": 0.70,         # z(public projection): star pull
    "b_small": 0.70,        # extra star pull x small (1 on a 2-game slate, 0 from 8 games)
    "b_total": 0.20,        # z(team implied total)
    "b_repl": 0.60,         # injury replacement: projected minutes >= 6 over his recent minutes
    "b_q": -0.80,           # DK status Q / GTD
    "b_bench": -0.60,       # confirmed non-starter when DK starters are out
    "w_ours": 0.50,         # public projection = w*ours + (1-w)*DK FPPG
}
ROW_CAP_GPP = 0.92          # no player above 92 % in a GPP (one slot per lineup)
ROW_CAP_CASH = 0.97         # cash / double-up: real ownership reached 93 % (2026-10-20), so 0.92
                            # saturated; 0.97 leaves headroom without letting the proxy hit 100 %
ROW_CAP = ROW_CAP_GPP       # the GPP cap (kept for callers / tests that import the old name)
SPEND = 0.997               # field's expected spend / cap: real fields average $49,820-49,920
                            # of $50,000 (4 contests, 31k lineups, 2026-10-20); 1.0 used 0.99
HARD_FLOOR = 0.90           # sampled lineups below 90 % of the cap are rejected
MIN_PUB_PROJ = 2.0          # the field never drafts players projected under 2 DK points
BANDS = ((0.25, "High"), (0.10, "Med"), (0.0, "Low"))

# Contest archetypes: concentration of the ownership utility (cash is chalkier), projection noise
# of the optimizer-like part (cash fields are sharper and duplicate more), salary use (share of
# sampled lineups spending >= 99 % of the cap; the rest >= 98 % / 96 %) and expected spend.
# Calibrated in direction and scale on the first real standings (2026-10-20, slate 154575, three
# double-ups + one 29.7k GPP; docs/research/real-standings-2026-10-20.md), NOT fitted (1 slate):
# * conc: double-up ownership ~ (GPP ownership) ** k on the same slate, k = 2.05 / 2.20 / 3.00
#   (MAE-best power fit of real %Drafted, `sharpen_exponent`); 1.0 had cash / large GPP = 1.39.
#   Cash 1.60 (= 1.78 x) is a compromise: higher matches the real concentration but piles onto
#   the players our proxy wrongly favours (MAE up). Large GPP keeps 0.9 (best MAE vs the real
#   GPP). small_gpp: no data, unchanged.
# * noise: real double-ups put 18-35 % of entries on ONE lineup and 62-89 % on duplicated ones;
#   at 0.10 our cash field had ~1 % / 23-50 % and a top-1 % line 10 points above the real one
#   (293 vs 283.5 at the actual box scores). 0.03 gives ~8 % / 40-54 % and a 286 top-1 % line;
#   the cash line at the actual box scores stays inside the real spread (248 vs 237-248).
# * full_spend: 95-97 % of real lineups used >= $49,500 (99-100 % >= $49,000).
# * mix (the default = "medium" optimizer-like share): the real 29.7k GPP had 82 % unique
#   lineups and 28 % of entries in duplicated ones (3,000-entry draws); at 0.40 our large-GPP
#   field had 62 % / 48 % (the optimizer part resamples ~200 solves), at 0.20 80 % / 30 %, with
#   the same cash line at the actual box scores (253.4 vs 253.25 real) and the real field's mean
#   projection (ours 248.8 / public 256.9 vs 247.6 / 256.0 real). Cash and small GPP keep 0.40.
ARCHETYPES: dict[str, dict[str, float]] = {
    "cash": {"conc": 1.60, "noise": 0.03, "full_spend": 0.97, "mix": 0.40},
    "small_gpp": {"conc": 1.00, "noise": 0.15, "full_spend": 0.95, "mix": 0.40},
    "large_gpp": {"conc": 0.90, "noise": 0.18, "full_spend": 0.95, "mix": 0.20},
}

# Ownership uncertainty (IMPROVEMENTS.md D5, docs/research/field-k.md; switch COURTSIDE_FIELD_K,
# OFF by default). ON, the field is K fields instead of one: each draws per-player log-odds noise
# on the proxy ownership, logit(own'_i) = logit(own_i) + c_i + OWN_SD[arch] * u_i * z_i (u_i =
# late_news_tag's own_uncertainty, 2.0 for late-news beneficiaries, else 1.0), re-balanced to
# the slot totals and the row cap by `sinkhorn`, and is sampled from it (the optimizer-like
# solves are shared). Sims are assigned to the fields in turn (row s -> field s mod K), so every
# chance is an average over the K fields at no extra scoring cost, and the per-field values give
# the across-field spread (display only). The noise is centred (`mean_shift`): each player's
# expected ownership stays at the proxy, the field is more uncertain, not weaker. OWN_SD
# reproduces the proxy's measured errors on the 2026-10-20 uploads by simulation
# (research/field_k.py fit: 60 draws per sd on a 0.05 grid, proxy vs draw scored exactly like
# proxy vs real): large GPP 0.86 from the overall MAE vs the real 29.7k GPP (4.57 pp; implies a
# top-10 MAE of 13.9 vs 12.6 real, rank corr 0.92 vs 0.87); cash 2.65 from the double-ups'
# top-10 MAE (27.4 pp, mean of three; implies an overall MAE of 10.7 vs 10.0 real, top-10 bias
# -23 vs -25, rank corr 0.79 vs 0.77). Direct log-odds residual sd: 1.04 GPP, 1.5-2.0
# double-ups. small_gpp has no data: the large-GPP value. docs/research/field-k.md.
FIELD_K_ENV = "COURTSIDE_FIELD_K"
FIELD_K_DEFAULT = 0          # OFF; "1" / "on" = FIELD_K_ON fields, an integer >= 2 = that many
FIELD_K_ON = 4
FIELD_K_TAG = "+k"
OWN_SD: dict[str, float] = {"cash": 2.65, "small_gpp": 0.86, "large_gpp": 0.86}
OWN_P_CLIP = 1e-4            # log-odds of ownership clipped to [1e-4, 1 - 1e-4]
FLOOR_FULL, FLOOR_MOST, FLOOR_REST = 0.99, 0.98, 0.96   # per-lineup spend floors (x cap) ...
FLOOR_SPLIT = 0.8           # ... of the lineups below FLOOR_FULL, this share gets FLOOR_MOST
STRENGTHS = {"soft": 0.20, "medium": 0.40, "sharp": 0.70}    # optimizer-like share (S12)
DEFAULT_MIX = STRENGTHS["medium"]   # mix=None / "medium" use the archetype's "mix" instead
SMALL_GPP_MAX = 1000        # field size up to which a GPP is "small"
FIELD_MAX = 20_000          # sampled lineups for big GPPs (§5.5.3)
FIELD_MIN = 5_000           # population sample for small contests (opponents drawn per sim)
N_SOLVES = 200              # optimizer-like CP-SAT solves (§5.5.3)
SOLVE_THREADS = 2
SOLVE_LIMIT_S = 1.0
FILL_ORDER = ("CPT", "C", "PG", "SG", "SF", "PF", "G", "F", "UTIL", "FLEX")   # scarcity order
Q_STATUSES = {"Q", "GTD", "D"}
OUT_STATUSES = {"O", "OUT"}
UNIT = 32                   # sims quantised to 1/32 DK point: exact integer lineup sums
WUNIT = 2                   # slot multiplier x2 (CPT 1.5 -> 3): integer weights
BIAS = 1 << 16              # keeps (negative) scores positive inside a row (scores > -1024 pts)
ROW_OFF = 1 << 17           # per-row offset for one flat searchsorted (scores < 1024 pts)
MAX_ROWS = (1 << 31) // ROW_OFF - 1


def field_k_mode(v: Any, default: int = FIELD_K_DEFAULT) -> int:
    """Number of fields from an env string / bool / int: '' / None -> `default`; 0 off false no
    -> 0 (one field, the classic path); 1 on true yes -> FIELD_K_ON; an integer >= 2 -> it."""
    if v is None:
        return int(default)
    if isinstance(v, bool):
        return FIELD_K_ON if v else 0
    x = str(v).strip().lower()
    if x == "":
        return int(default)
    if x in ("0", "off", "false", "no"):
        return 0
    if x in ("1", "on", "true", "yes"):
        return FIELD_K_ON
    try:
        k = int(x.lstrip("k"))
    except ValueError:
        log.warning("%s=%r not understood; ownership uncertainty off", FIELD_K_ENV, v)
        return 0
    return FIELD_K_ON if k == 1 else max(0, k)


FIELD_K = field_k_mode(os.environ.get(FIELD_K_ENV), FIELD_K_DEFAULT)


def set_field_k(k: Any) -> None:
    """Set the number of fields for later builds (the env is read once at import)."""
    global FIELD_K
    FIELD_K = field_k_mode(k, 0)


def field_k() -> int:
    """Fields per contest: 0 = OFF (one field, bit-identical to field-proxy-1.2)."""
    return FIELD_K


# Showdown captain term (IMPROVEMENTS.md B6, PLAYERS.md §5.7, docs/research/b6-d10.md; switch
# COURTSIDE_SD_CPT, OFF by default). OFF, the CPT column of a Showdown allocation uses the UTIL
# utility, so captains differ from UTIL only through the salary shadow price at 1.5x salary. ON,
# the CPT column gets its own star-weighted utility (`cpt_utility`):
#     u_cpt_i = conc * ( c_value*z(value_i) + c_proj*z(pub_i) + c_star*log(pub_i / pub_max)
#                        + b_total*z(total_i) + b_repl*repl_i + b_q*q_i + b_bench*bench_i )
# pub_max = the slate's best public projection, so the CPT weight carries a (pub_i / pub_max)
# ^ (conc * c_star) star factor: a smooth version of the public-projection rank (a hard rank put
# 28 % vs 11 % CPT on two players projected 41.1 and 41.0 on slate 154826). Value per $1k matters
# much less than at UTIL. Measured on the 15 stored Showdown slates (research/b6_showdown_own.py):
# OFF, the three best value plays outside the top-6 projections get 14 % of their ownership at CPT
# vs 17 % for the top-3 projections (1/6 = proportional), and the top CPT is 16 %.
# HAND-SET PRIOR, not fitted: no Showdown standings had been uploaded by 2026-10-24 and no public
# Showdown ownership exists in our data. Version tag "+cpt" on the ownership proxy (and the field
# built from it) of Showdown slates only; Classic is untouched either way.
SD_CPT_ENV = "COURTSIDE_SD_CPT"
SD_CPT_DEFAULT = False
SD_CPT_TAG = "+cpt"
CPT_PARAMS: dict[str, float] = {
    "c_value": 0.40,        # z(public value per $1k) at CPT (UTIL: b_value 1.10)
    "c_proj": 0.70,         # z(public projection) at CPT (UTIL on one game: b_proj + b_small)
    "c_star": 3.00,         # log(public projection / the slate's best): star factor
}


def sd_cpt_mode(v: Any, default: bool = SD_CPT_DEFAULT) -> bool:
    """'' / None -> `default`; 1 on true yes -> True; 0 off false no -> False."""
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    x = str(v).strip().lower()
    if x == "":
        return bool(default)
    if x in ("1", "on", "true", "yes"):
        return True
    if x in ("0", "off", "false", "no"):
        return False
    log.warning("%s=%r not understood; Showdown captain term off", SD_CPT_ENV, v)
    return False


SD_CPT = sd_cpt_mode(os.environ.get(SD_CPT_ENV), SD_CPT_DEFAULT)


def set_sd_cpt(on: Any) -> None:
    """Turn the Showdown captain term on / off for later proxies (the env is read at import)."""
    global SD_CPT
    SD_CPT = sd_cpt_mode(on, False)


def sd_cpt() -> bool:
    return SD_CPT


# ============================================================================== inputs

def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _game(p: PoolPlayer) -> str:
    return p.competition_id or p.game_id or p.game_label or f"team:{p.team}"


def latest_run(conn: sqlite3.Connection, slate_id) -> dict | None:
    """The newest ok run of the slate that has stored sims."""
    r = conn.execute("SELECT run_id, kind, created_at, sim_path FROM dfs_runs WHERE slate_id=? "
                     "AND ok=1 AND sim_path IS NOT NULL ORDER BY created_at DESC LIMIT 1",
                     (str(slate_id),)).fetchone()
    return dict(r) if r else None


def _run_projections(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """DK id -> proj (dk mean, cond all), p25, proj_min, p_play, nba id for a projection run."""
    rows = conn.execute(
        "SELECT pp.player_id, pp.p_play, pp.inputs_json, "
        "(SELECT mean FROM player_projection_stats s WHERE s.projection_id=pp.id AND s.stat='dk' "
        " AND s.cond='all') AS proj, "
        "(SELECT q25 FROM player_projection_stats s WHERE s.projection_id=pp.id AND s.stat='dk' "
        " AND s.cond='all') AS p25, "
        "(SELECT mean FROM player_projection_stats s WHERE s.projection_id=pp.id AND s.stat='min' "
        " AND s.cond='all') AS proj_min FROM player_projections pp WHERE pp.run_id=?",
        (run_id,)).fetchall()
    out = []
    for r in rows:
        try:
            dk = json.loads(r["inputs_json"] or "{}").get("dk_id")
        except (TypeError, ValueError):
            dk = None
        if dk is None or (isinstance(dk, float) and not math.isfinite(dk)) or str(dk) == "nan":
            continue
        out.append({"pid": str(dk), "nba_id": int(r["player_id"]), "proj": _f(r["proj"]),
                    "p25": _f(r["p25"]), "proj_min": _f(r["proj_min"]),
                    "p_play": _f(r["p_play"])})
    return pd.DataFrame(out, columns=["pid", "nba_id", "proj", "p25", "proj_min", "p_play"])


def team_totals(conn: sqlite3.Connection, game_ids: Iterable[str],
                at: str | None = None) -> dict[tuple[str, str], float]:
    """{(game_id, 'home'|'away'): implied team total}: the latest line snapshot at or before `at`
    (consensus book when present, else the median over books)."""
    out: dict[tuple[str, str], float] = {}
    for gid in {g for g in game_ids if g}:
        sql = ("SELECT captured_at, book, home_team_total, away_team_total, total, home_spread "
               "FROM line_snapshots WHERE game_id=?")
        args: list = [gid]
        if at:
            sql += " AND captured_at<=?"
            args.append(at)
        try:
            rows = conn.execute(sql + " ORDER BY captured_at DESC", args).fetchall()
        except sqlite3.OperationalError:
            return out
        if not rows:
            continue
        last = rows[0]["captured_at"]
        cur = [r for r in rows if r["captured_at"] == last]
        pick = [r for r in cur if r["book"] == "consensus"] or cur
        hs, aws = [], []
        for r in pick:
            h, a = _f(r["home_team_total"]), _f(r["away_team_total"])
            if (h is None or a is None) and _f(r["total"]) is not None \
                    and _f(r["home_spread"]) is not None:
                t, s = float(r["total"]), float(r["home_spread"])
                h, a = (t - s) / 2, (t + s) / 2
            if h is not None and a is not None:
                hs.append(h)
                aws.append(a)
        if hs:
            out[(gid, "home")] = float(np.median(hs))
            out[(gid, "away")] = float(np.median(aws))
    return out


def recent_minutes(conn: sqlite3.Connection, nba_ids: Iterable[int], before: str,
                   n_games: int = 15) -> dict[int, float]:
    """Mean minutes over each player's last `n_games` games before `before` (any season type)."""
    ids = sorted({int(i) for i in nba_ids if i is not None})
    out: dict[int, float] = {}
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        q = ",".join("?" * len(chunk))
        try:
            df = pd.read_sql_query(
                f"SELECT player_id, game_date, min FROM player_games WHERE player_id IN ({q}) "
                f"AND game_date < ? ORDER BY game_date DESC", conn, params=[*chunk, before])
        except Exception:  # noqa: BLE001 - optional input
            return out
        for pid, g in df.groupby("player_id"):
            m = pd.to_numeric(g["min"], errors="coerce").dropna().head(n_games)
            if len(m):
                out[int(pid)] = float(m.mean())
    return out


def pool_inputs(pool: SlatePool | Sequence[PoolPlayer], *, proj: Mapping[str, float] | None = None,
                fppg: Mapping[str, float] | None = None,
                proj_min: Mapping[str, float] | None = None,
                recent_min: Mapping[str, float] | None = None,
                totals: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The ownership-model input table, one row per pool player (keyed by DK player id `pid`).
    Missing inputs stay NaN and drop out of the utility (contribute 0)."""
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    proj, fppg = proj or {}, fppg or {}
    proj_min, recent_min, totals = proj_min or {}, recent_min or {}, totals or {}
    rows = []
    for p in players:
        f = fppg.get(p.dk_player_id, p.fppg)
        rows.append({
            "pid": p.dk_player_id, "name": p.name, "team": p.team, "game": _game(p),
            "positions": "/".join(p.positions), "salary": int(p.salary),
            "status": p.status, "starter": p.starter,
            "fppg": _f(f), "proj": _f(proj.get(p.dk_player_id)),
            "proj_min": _f(proj_min.get(p.dk_player_id)),
            "recent_min": _f(recent_min.get(p.dk_player_id)),
            "team_total": _f(totals.get(p.dk_player_id))})
    return pd.DataFrame(rows)


def capture_at(conn: sqlite3.Connection, slate_id, ts: str | None) -> str | None:
    """The slate's latest `dfs_salaries` capture at or before `ts` (None: no capture that early,
    or no `ts`)."""
    if not ts:
        return None
    r = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                     "captured_at<=?", (str(slate_id), ts)).fetchone()
    return r[0] if r else None


def slate_inputs(conn: sqlite3.Connection, slate_id, run_id: str | None = None, *,
                 as_of: str | None = None) -> tuple[SlatePool, pd.DataFrame, dict]:
    """(pool, inputs, info) for a live DK slate from its latest salary capture, the projection run
    (`run_id`, default the newest run with sims), line snapshots and recent box-score minutes.
    `as_of` (ISO UTC) uses the latest capture at or before it instead (statuses, starter flags
    and FPPG as the public saw them then: the ownership refit and the real-standings research
    rebuild a past run's inputs this way; live runs leave it None = the newest capture)."""
    from . import rules as R
    from .. import ids as IDS
    sid = str(slate_id)
    pool = R.build_pool(conn, sid, captured_at=capture_at(conn, sid, as_of))
    run = None
    if run_id is None:
        run = latest_run(conn, sid)
        run_id = run["run_id"] if run else None
    else:
        r = conn.execute("SELECT run_id, kind, created_at, sim_path FROM dfs_runs WHERE run_id=?",
                         (run_id,)).fetchone()
        run = dict(r) if r else {"run_id": run_id, "sim_path": None, "created_at": None}
    pr = _run_projections(conn, run_id) if run_id else pd.DataFrame(
        columns=["pid", "nba_id", "proj", "p25", "proj_min", "p_play"])
    proj = dict(zip(pr["pid"], pr["proj"]))
    pmin = dict(zip(pr["pid"], pr["proj_min"]))
    slate = conn.execute("SELECT nba_day, lock_utc FROM dfs_slates WHERE slate_id=?",
                         (sid,)).fetchone()
    day = slate["nba_day"] if slate else (pool.captured_at or "")[:10]
    lock = slate["lock_utc"] if slate else None
    dk2nba = IDS.nba_ids(conn, IDS.DK, [p.dk_player_id for p in pool.players])
    rec = recent_minutes(conn, dk2nba.values(), day) if day else {}
    recent = {dk: rec[n] for dk, n in dk2nba.items() if n in rec}
    tt = team_totals(conn, {p.game_id for p in pool.players}, lock)
    totals = {}
    for p in pool.players:
        side = "home" if p.team == p.home else "away"
        if (p.game_id, side) in tt:
            totals[p.dk_player_id] = tt[(p.game_id, side)]
    inp = pool_inputs(pool, proj=proj, proj_min=pmin, recent_min=recent, totals=totals)
    info = {"run_id": run_id, "sim_path": run.get("sim_path") if run else None,
            "n_projected": int(inp["proj"].notna().sum()), "n_totals": len(totals),
            "n_recent_min": len(recent), "nba_day": day, "lock_utc": lock}
    return pool, inp, info


# ============================================================================== ownership

@dataclass
class OwnershipProxy:
    """Ownership proxy of one slate and contest archetype. `own` are fractions (Σ = n slots);
    `by_slot` is the per-slot allocation (players x distinct slot names; CPT = captain split)."""
    pids: list[str]
    names: list[str]
    own: np.ndarray
    by_slot: np.ndarray
    slot_names: tuple[str, ...]
    slot_counts: np.ndarray
    archetype: str
    version: str
    params: dict
    inputs: pd.DataFrame
    lam: float = 0.0
    spend: float = 0.0
    label: str = LABEL
    row_cap: float | None = None        # the cap the allocation used (None: the archetype's)

    @property
    def cap(self) -> float:
        return row_cap_for(self.archetype) if self.row_cap is None else self.row_cap

    @property
    def bands(self) -> list[str]:
        return [band(o) for o in self.own]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.pids, (float(x) for x in self.own)))

    def captain(self) -> dict[str, float]:
        if "CPT" not in self.slot_names:
            return {}
        k = self.slot_names.index("CPT")
        return dict(zip(self.pids, (float(x) for x in self.by_slot[:, k])))

    def table(self, n: int | None = None) -> pd.DataFrame:
        df = self.inputs.copy()
        df["own"] = self.own
        df["band"] = self.bands
        if "CPT" in self.slot_names:
            df["own_cpt"] = self.by_slot[:, self.slot_names.index("CPT")]
        df = df.sort_values("own", ascending=False)
        return df.head(n) if n else df


def band(own: float) -> str:
    """Low / Med / High (PLAYERS.md §2.1, §9: bands until calibrated)."""
    for lo, name in BANDS:
        if own >= lo:
            return name
    return "Low"


def classify_contest(contest: Mapping[str, Any]) -> str:
    """cash (double-up / 50-50 / one tier paying >= 30 %), small_gpp (field <= 1000), large_gpp."""
    attrs = contest.get("attrs") or {}
    flags = json.dumps(attrs).lower()
    tiers = contest.get("payouts") or []
    m = int(contest.get("field_size") or 0)
    paid = max((int(t["max_pos"]) for t in tiers), default=0)
    if "isdoubleup" in flags or "isfiftyfifty" in flags or "double_up" in flags or (
            len({t["payout"] for t in tiers}) == 1 and m and paid / m >= 0.30):
        return "cash"
    return "small_gpp" if m <= SMALL_GPP_MAX else "large_gpp"


def row_cap_for(archetype: str) -> float:
    """The most ownership one player can have: 0.97 for cash (double-up / 50-50), 0.92 for GPPs."""
    return ROW_CAP_CASH if archetype == "cash" else ROW_CAP_GPP


def _z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & ref
    if ok.sum() < 3:
        return np.zeros_like(x)
    mu, sd = float(np.mean(x[ok])), float(np.std(x[ok]))
    z = (x - mu) / (sd if sd > 1e-9 else 1.0)
    return np.where(np.isfinite(z), z, 0.0)


def _eligibility(inputs: pd.DataFrame, pool_by_id: Mapping[str, PoolPlayer],
                 rules: Rules) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """(slot names, counts, eligibility (n x k) bool, salary per slot (n x k))."""
    names = rules.distinct_slot_names
    counts = np.array([rules.slot_names.count(s) for s in names], float)
    n = len(inputs)
    E = np.zeros((n, len(names)), bool)
    S = np.zeros((n, len(names)), float)
    for i, pid in enumerate(inputs["pid"]):
        p = pool_by_id[pid]
        for k, s in enumerate(names):
            if s in p.slots:
                E[i, k] = True
                S[i, k] = p.salary_for(s, rules)
    return names, counts, E, S


def public_projection(inputs: pd.DataFrame, w_ours: float) -> np.ndarray:
    ours = inputs["proj"].to_numpy(float)
    fp = inputs["fppg"].to_numpy(float)
    both = np.isfinite(ours) & np.isfinite(fp)
    return np.where(both, w_ours * ours + (1 - w_ours) * fp,
                    np.where(np.isfinite(ours), ours, np.where(np.isfinite(fp), fp, 0.0)))


def utility(inputs: pd.DataFrame, params: Mapping[str, float], archetype: str = "large_gpp"
            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(utility u (n,), active mask, features) of the public-information ownership model."""
    P = {**DEFAULT_PARAMS, **(params or {})}
    pub = public_projection(inputs, P["w_ours"])
    sal = inputs["salary"].to_numpy(float)
    status = inputs["status"].fillna("").astype(str).str.upper().to_numpy()
    active = (pub >= MIN_PUB_PROJ) & ~np.isin(status, list(OUT_STATUSES)) & (sal > 0)
    value = pub / np.maximum(sal / 1000.0, 0.1)
    ref = active & (pub >= 8)
    games = inputs.loc[active, "game"].nunique() if active.any() else 0
    small = float(np.clip((8 - games) / 6.0, 0.0, 1.0))
    zt = _z(inputs["team_total"].to_numpy(float), ref)
    pmin, rmin = inputs["proj_min"].to_numpy(float), inputs["recent_min"].to_numpy(float)
    repl = (np.isfinite(pmin) & np.isfinite(rmin) & (pmin - np.nan_to_num(rmin) >= 6)).astype(float)
    q = np.isin(status, list(Q_STATUSES)).astype(float)
    st = inputs["starter"]
    known = st.notna().any() and bool(st.fillna(False).astype(bool).any())
    bench = (known & (~st.fillna(False).astype(bool)).to_numpy() & (repl == 0)).astype(float)
    conc = ARCHETYPES[archetype]["conc"]
    u = conc * (P["b_value"] * _z(value, ref) + (P["b_proj"] + P["b_small"] * small) * _z(pub, ref)
                + P["b_total"] * zt + P["b_repl"] * repl + P["b_q"] * q + P["b_bench"] * bench)
    feats = {"pub": pub, "value": value, "small": small, "games": games, "repl": repl, "q": q,
             "bench": bench}
    return np.where(active, u, -np.inf), active, feats


def cpt_utility(inputs: pd.DataFrame, params: Mapping[str, float], archetype: str,
                feats: Mapping[str, Any], active: np.ndarray) -> np.ndarray:
    """The Showdown captain utility (B6, `CPT_PARAMS`): star-weighted by the public projection
    relative to the slate's best, with the UTIL utility's team-total, replacement, Q and bench
    terms. -inf where inactive."""
    P = {**DEFAULT_PARAMS, **CPT_PARAMS, **(params or {})}
    pub = np.asarray(feats["pub"], float)
    ref = active & (pub >= 8)
    top = float(np.max(pub[active])) if active.any() else 1.0
    star = np.log(np.clip(pub, MIN_PUB_PROJ, None) / max(top, MIN_PUB_PROJ))
    zt = _z(inputs["team_total"].to_numpy(float), ref)
    conc = ARCHETYPES[archetype]["conc"]
    u = conc * (P["c_value"] * _z(np.asarray(feats["value"], float), ref)
                + P["c_proj"] * _z(pub, ref) + P["c_star"] * star
                + P["b_total"] * zt + P["b_repl"] * feats["repl"] + P["b_q"] * feats["q"]
                + P["b_bench"] * feats["bench"])
    return np.where(active, u, -np.inf)


def sinkhorn(weights: np.ndarray, counts: np.ndarray, row_cap: float = ROW_CAP,
             iters: int = 300, tol: float = 1e-9) -> np.ndarray:
    """Balance a non-negative (players x slots) matrix so each column sums to its slot count and
    no row (player) exceeds `row_cap`. Columns of zeros stay zero."""
    X = weights.astype(float).copy()
    for _ in range(iters):
        cs = X.sum(0)
        X *= np.where(cs > 0, counts / np.where(cs > 0, cs, 1), 0.0)
        rs = X.sum(1)
        over = rs > row_cap
        if not over.any():
            if np.all(np.abs(X.sum(0) - counts) < tol * max(1.0, counts.sum())):
                break
            continue
        X[over] *= (row_cap / rs[over])[:, None]
    cs = X.sum(0)
    X *= np.where(cs > 0, counts / np.where(cs > 0, cs, 1), 0.0)
    return X


def ownership_proxy(inputs: pd.DataFrame, pool: SlatePool | Sequence[PoolPlayer],
                    rules: Rules | None = None, *, archetype: str = "large_gpp",
                    params: Mapping[str, float] | None = None, version: str | None = None,
                    spend: float | None = None, row_cap: float | None = None,
                    cpt: bool | None = None) -> OwnershipProxy:
    """The ownership proxy (Σ own = number of slots). See the module docstring. `spend`
    (fraction of the cap the field spends on average) defaults to SPEND; `row_cap` (most
    ownership any one player gets) to the archetype's (`row_cap_for`). `cpt` (None: the
    COURTSIDE_SD_CPT switch, `sd_cpt()`): the Showdown captain term (B6, `cpt_utility`) for
    the CPT column; ignored on rosters without a CPT slot."""
    spend = SPEND if spend is None else float(spend)
    row_cap = row_cap_for(archetype) if row_cap is None else float(row_cap)
    rules = rules or (pool.rules if isinstance(pool, SlatePool) else None)
    if rules is None:
        raise ValueError("no rules")
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    by_id = {p.dk_player_id: p for p in players}
    inputs = inputs[inputs["pid"].isin(by_id)].reset_index(drop=True)
    if params is None or version is None:
        lp, lv = load_params()
        params = params if params is not None else lp
        version = version or lv
    P = {**DEFAULT_PARAMS, **params}
    u, active, feats = utility(inputs, P, archetype)
    names, counts, E, S = _eligibility(inputs, by_id, rules)
    E &= active[:, None]
    target = spend * rules.salary_cap
    s_k = S / 1000.0
    umax = np.max(u[active]) if active.any() else 0.0
    base = np.where(active, np.exp(np.clip(u - umax, -50, 0)), 0.0)
    use_cpt = (SD_CPT if cpt is None else bool(cpt)) and "CPT" in names
    if use_cpt:                         # B6: the CPT column gets its own utility
        uc = cpt_utility(inputs, params, archetype, feats, active)
        ucmax = np.max(uc[active]) if active.any() else 0.0
        base2 = np.repeat(base[:, None], len(names), axis=1)
        base2[:, names.index("CPT")] = np.where(active, np.exp(np.clip(uc - ucmax, -50, 0)), 0.0)
        P = {**P, **CPT_PARAMS, **{k: v for k, v in params.items() if k in CPT_PARAMS}}

    def alloc(lam: float) -> np.ndarray:
        if use_cpt:
            W = base2 * np.exp(-lam * (s_k - s_k[E].mean() if E.any() else 0)) * E
            return sinkhorn(W, counts, row_cap)
        W = base[:, None] * np.exp(-lam * (s_k - s_k[E].mean() if E.any() else 0)) * E
        return sinkhorn(W, counts, row_cap)

    def spent(X):
        return float((X * S).sum())

    lo, hi = -3.0, 3.0
    X_lo, X_hi = alloc(lo), alloc(hi)
    if spent(X_lo) < target:            # even spending up cannot reach the target
        lam, X = lo, X_lo
    elif spent(X_hi) > target:
        lam, X = hi, X_hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            Xm = alloc(mid)
            if spent(Xm) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 2e-3:
                break
        lam = (lo + hi) / 2
        X = alloc(lam)
    inputs = inputs.copy()
    inputs["pub_proj"] = feats["pub"]
    inputs["pub_value"] = feats["value"]
    inputs["replacement"] = feats["repl"].astype(bool)
    return OwnershipProxy(pids=list(inputs["pid"]), names=list(inputs["name"]), own=X.sum(1),
                          by_slot=X, slot_names=names, slot_counts=counts, archetype=archetype,
                          version=f"{version}/{archetype}" + (SD_CPT_TAG if use_cpt else ""),
                          params=P, inputs=inputs, lam=lam,
                          spend=spent(X), row_cap=row_cap)


# ============================================================================== params (fit)

def params_path() -> Path:
    from ... import config
    return config.data_dir() / "models" / PARAMS_FILE


def load_params() -> tuple[dict, str]:
    """(params, version): the fitted parameters once `fit_from_uploads` saved them (>= 10 real
    slates), else the hand-set defaults."""
    p = params_path()
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if d.get("n_slates", 0) >= MIN_FIT_SLATES:
                return {**DEFAULT_PARAMS, **d["params"]}, d["version"]
        except (OSError, ValueError, KeyError):
            pass
    return dict(DEFAULT_PARAMS), HANDSET_VERSION


def calibrated() -> bool:
    return load_params()[1] != HANDSET_VERSION


FIT_KEYS = ("b_value", "b_proj", "b_total", "b_repl", "b_q", "w_ours")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _slate_lock(conn, slate_id) -> str | None:
    try:
        r = conn.execute("SELECT lock_utc FROM dfs_slates WHERE slate_id=?",
                         (str(slate_id),)).fetchone()
    except (AttributeError, sqlite3.Error):
        return None
    return r[0] if r else None


def fit_from_uploads(conn: sqlite3.Connection, *, save: bool = True,
                     min_slates: int = MIN_FIT_SLATES, maxiter: int = 400,
                     loso: bool | int = True) -> dict:
    """Re-estimate the ownership-proxy weights from real %Drafted (`standings.ownership_dataset`).

    Needs >= `min_slates` distinct slates of uploads; otherwise returns {"status":
    "insufficient", ...} and leaves the hand-set proxy in place. The model is refit as it runs
    live (same inputs, same Sinkhorn allocation, the contest's archetype) by Nelder-Mead on the
    mean absolute error in percentage points over (contest, player) rows. Reports MAE and rank
    correlation (pooled and mean per contest) for the hand-set and the fitted parameters, plus
    leave-one-slate-out MAE of the fit (`loso`: True = every slate, an int = that many held-out
    slates, False = skip). Saves `var/models/dfs_ownership_params.json` when `save`."""
    from scipy.optimize import minimize
    from . import standings as ST
    ds = ST.ownership_dataset(conn)
    slates = sorted(ds["slate_id"].dropna().astype(str).unique()) if len(ds) else []
    base = {"status": "insufficient", "n_slates": len(slates), "n_rows": int(len(ds)),
            "needed": min_slates, "version": HANDSET_VERSION}
    if len(slates) < min_slates:
        return base
    groups = []                              # (slate, pool, inputs, archetype, actual (n,))
    cache: dict[str, tuple] = {}
    for cid, g in ds.groupby("contest_id"):
        sid = str(g["slate_id"].iloc[0])
        if sid not in cache:
            try:                                 # inputs as the public saw them at lock
                cache[sid] = slate_inputs(conn, sid, g["run_id"].dropna().iloc[0]
                                          if g["run_id"].notna().any() else None,
                                          as_of=_slate_lock(conn, sid))[:2]
            except Exception as e:  # noqa: BLE001
                log.warning("ownership fit: slate %s skipped (%s)", sid, e)
                cache[sid] = None
        if cache[sid] is None:
            continue
        pool, inp = cache[sid]
        c = load_contest(conn, str(cid))
        arch = c.archetype if c else "large_gpp"
        act = dict(zip(g["dk_id"].astype(str), g["pct_drafted"].astype(float) / 100.0))
        y = np.array([act.get(p, 0.0) for p in inp["pid"]])
        groups.append((sid, pool, inp, arch, y))
    if not groups:
        return {**base, "status": "no_usable_slates"}

    def predict(params, gs):
        out = []
        for sid, pool, inp, arch, y in gs:
            op = ownership_proxy(inp, pool, archetype=arch, params=params, version="fit")
            # Showdown uploads store the UTIL %Drafted (standings._ownership_rows; CPT in the
            # upload's detail), so a Showdown contest is compared on the UTIL column (B6)
            pred = dict(zip(op.pids, op.by_slot[:, op.slot_names.index("UTIL")]
                            if "CPT" in op.slot_names and "UTIL" in op.slot_names else op.own))
            out.append((np.array([pred.get(p, 0.0) for p in inp["pid"]]), y))
        return out

    def metrics(pairs):
        pp = np.concatenate([p for p, _ in pairs])
        yy = np.concatenate([y for _, y in pairs])
        keep = (pp > 0.001) | (yy > 0.001)
        per = [_spearman(p[(p > 0.001) | (y > 0.001)], y[(p > 0.001) | (y > 0.001)])
               for p, y in pairs]
        return {"mae_pp": float(np.mean(np.abs(pp[keep] - yy[keep])) * 100),
                "spearman": _spearman(pp[keep], yy[keep]),
                "spearman_per_contest": float(np.nanmean(per)) if per else float("nan"),
                "n": int(keep.sum())}

    def loss(theta, gs):
        params = {**DEFAULT_PARAMS, **dict(zip(FIT_KEYS, theta))}
        params["w_ours"] = float(np.clip(params["w_ours"], 0.0, 1.0))
        pairs = predict(params, gs)
        pp = np.concatenate([p for p, _ in pairs])
        yy = np.concatenate([y for _, y in pairs])
        return float(np.mean(np.abs(pp - yy)))

    x0 = np.array([DEFAULT_PARAMS[k] for k in FIT_KEYS])

    def fit(gs):
        r = minimize(loss, x0, args=(gs,), method="Nelder-Mead",
                     options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
        p = {**DEFAULT_PARAMS, **dict(zip(FIT_KEYS, r.x))}
        p["w_ours"] = float(np.clip(p["w_ours"], 0.0, 1.0))
        return p

    fitted = fit(groups)
    held = sorted({g[0] for g in groups}) if loso else []
    if loso is not True and loso:
        held = held[: int(loso)]
    loso_pairs = []
    for s in held:
        tr = [g for g in groups if g[0] != s]
        te = [g for g in groups if g[0] == s]
        if tr and te:
            loso_pairs.extend(predict(fit(tr), te))
    n_sl = len({g[0] for g in groups})
    version = f"{MODEL}-fit-{datetime.now(timezone.utc):%Y%m%d}-{n_sl}s"
    rep = {"status": "fitted", "version": version, "n_slates": n_sl,
           "n_contests": len(groups), "params": fitted,
           "handset": metrics(predict(DEFAULT_PARAMS, groups)),
           "fitted_in_sample": metrics(predict(fitted, groups)),
           "fitted_loso": metrics(loso_pairs) if loso_pairs else None,
           "loso_slates": len(held)}
    if save:
        path = params_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rep, indent=2, default=float))
    return rep


# ============================================================================== calibration report

def sharpen(p: np.ndarray, k: float, total: float, cap: float = 0.97) -> np.ndarray:
    """p ** k rescaled so the column sums to `total` (each value capped): the power map from one
    contest type's ownership to a more (k > 1) or less concentrated one."""
    x = np.power(np.maximum(np.asarray(p, float), 1e-5), k)
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        c = math.sqrt(lo * hi)
        if np.minimum(cap, c * x).sum() > total:
            hi = c
        else:
            lo = c
    return np.minimum(cap, math.sqrt(lo * hi) * x)


def sharpen_exponent(gpp: Mapping[str, float], cash: Mapping[str, float],
                     grid: Sequence[float] = tuple(np.arange(1.0, 3.01, 0.05))) -> float:
    """k minimising the MAE of cash ownership ~ sharpen(GPP ownership, k) (same slate). The
    proxy's cash / large-GPP `conc` ratio should be about this k."""
    keys = sorted(set(gpp) | set(cash))
    g = np.array([gpp.get(x, 0.0) for x in keys])
    c = np.array([cash.get(x, 0.0) for x in keys])
    keep = (g > 0.001) | (c > 0.001)
    errs = [float(np.mean(np.abs(sharpen(g, k, c.sum())[keep] - c[keep]))) for k in grid]
    return float(grid[int(np.argmin(errs))])


def calibration_report(conn: sqlite3.Connection) -> dict:
    """What uploaded standings say about the field proxy, per contest: real lineup structure
    (unique lineups, entries in duplicated lineups, the most duplicated lineup's share),
    concentration (top-5 / top-10 %Drafted sums) and the ownership proxy's errors at the
    contest's archetype (inputs at lock), plus per slate the cash vs large-GPP concentration
    exponent (`sharpen_exponent`). A PARTIAL calibration: it reports the parameters one slate
    can inform (ARCHETYPES conc ratio, noise / dupes, spend) and changes nothing;
    `fit_from_uploads` refits the ownership weights from MIN_FIT_SLATES slates."""
    from . import standings as ST
    rows = conn.execute(
        "SELECT DISTINCT r.contest_id, c.slate_id FROM dfs_contest_results r JOIN dfs_contests c "
        "ON c.contest_id=r.contest_id WHERE c.slate_id IS NOT NULL ORDER BY c.slate_id").fetchall()
    nba2dk = {}
    try:
        from .. import ids as IDS
        nba2dk = {v: k for k, v in IDS.nba_ids(conn, IDS.DK).items()}
    except Exception:  # noqa: BLE001 - optional mapping
        pass
    ds = ST.ownership_dataset(conn)
    run_of = dict(zip(ds["contest_id"].astype(str), ds["run_id"])) if len(ds) else {}
    out: dict = {"label": LABEL, "version": f"{MODEL}/{FIELD_VERSION}", "contests": [],
                 "slates": {}, "n_slates": len({r[1] for r in rows}), "needed": MIN_FIT_SLATES}
    cache: dict = {}
    by_slate: dict = {}
    for cid, sid in rows:
        c = load_contest(conn, cid)
        res = conn.execute("SELECT lineup_text FROM dfs_contest_results WHERE contest_id=?",
                           (cid,)).fetchall()
        keys = []
        for (txt,) in res:
            pr = ST.parse_lineup(txt)
            if pr:
                keys.append(tuple(sorted(n for _, n in pr)))
        cnt = pd.Series(keys).value_counts() if keys else pd.Series(dtype=int)
        m = max(1, len(keys))
        own = {}
        for pid, pct in conn.execute("SELECT player_id, pct_drafted FROM dfs_contest_ownership "
                                     "WHERE contest_id=?", (cid,)):
            k = nba2dk.get(int(pid))
            if k is not None and pct is not None:
                own[str(k)] = own.get(str(k), 0.0) + float(pct) / 100.0
        top = np.sort(np.array(list(own.values()) or [0.0]))[::-1]
        row = {"contest_id": cid, "slate_id": sid, "name": c.name if c else None,
               "archetype": c.archetype if c else None, "payouts": c.source if c else None,
               "entries": len(keys), "unique_share": len(cnt) / m,
               "dupe_entry_share": float(cnt[cnt > 1].sum()) / m if len(cnt) else 0.0,
               "max_dupe_share": float(cnt.max()) / m if len(cnt) else 0.0,
               "own_top5": float(top[:5].sum()), "own_top10": float(top[:10].sum())}
        if c is not None and own:
            if sid not in cache:
                try:
                    run = run_of.get(str(cid))
                    cache[sid] = slate_inputs(conn, sid, run if isinstance(run, str) else None,
                                              as_of=_slate_lock(conn, sid))[:2]
                except Exception as e:  # noqa: BLE001
                    log.warning("calibration report: slate %s inputs failed (%s)", sid, e)
                    cache[sid] = None
            pred = None
            if cache[sid] is not None:
                try:
                    pool_, inp_ = cache[sid]          # slate_inputs order: (pool, inputs)
                    pred = ownership_proxy(inp_, pool_, archetype=c.archetype).as_dict()
                except (ValueError, KeyError) as e:
                    log.warning("calibration report: no proxy for slate %s (%s)", sid, e)
            if pred is not None:
                ks = sorted(set(pred) | set(own))
                pp = np.array([pred.get(x, 0.0) for x in ks])
                yy = np.array([own.get(x, 0.0) for x in ks])
                keep = (pp > 0.001) | (yy > 0.001)
                o = np.argsort(-yy)[:10]
                row["proxy"] = {"mae_pp": float(np.mean(np.abs(pp - yy)[keep]) * 100),
                                "spearman": _spearman(pp[keep], yy[keep]),
                                "top10_mae_pp": float(np.mean(np.abs(pp - yy)[o]) * 100),
                                "top10_bias_pp": float(np.mean((pp - yy)[o]) * 100),
                                "own_top10": float(np.sort(pp)[::-1][:10].sum())}
        by_slate.setdefault(sid, []).append((row, own))
        out["contests"].append(row)
    for sid, items in by_slate.items():
        gpp = [o for r, o in items if r["archetype"] == "large_gpp" and o]
        cash = [(r, o) for r, o in items if r["archetype"] == "cash" and o]
        if gpp and cash:
            g = max(gpp, key=len)
            ks = [sharpen_exponent(g, o) for _, o in cash]
            out["slates"][sid] = {"cash_vs_gpp_exponent": ks,
                                  "proxy_conc_ratio": ARCHETYPES["cash"]["conc"]
                                  / ARCHETYPES["large_gpp"]["conc"]}
    return out


# ============================================================================== contests

@dataclass
class Contest:
    contest_id: str | None
    name: str
    entry_fee: float
    field_size: int
    payouts: list[dict]                 # [{min_pos, max_pos, payout}]
    archetype: str
    source: str = "captured"            # captured | preset

    @property
    def paid(self) -> int:
        return max((int(t["max_pos"]) for t in self.payouts), default=0)

    def ladder(self) -> np.ndarray:
        """payout by rank, index 0 unused, length field_size + 1."""
        pay = np.zeros(self.field_size + 2)
        for t in self.payouts:
            lo, hi = int(t["min_pos"]), min(int(t["max_pos"]), self.field_size)
            if lo <= hi:
                pay[lo:hi + 1] = float(t["payout"])
        return pay[: self.field_size + 1]


DU_PAID_FRAC = 0.435        # DK double-ups pay 2x the fee to round(0.435 x entries) places


def is_double_up(name: str | None) -> bool:
    n = (name or "").lower()
    return "double up" in n or "double-up" in n or "50/50" in n


def du_paid_places(entries: int) -> int:
    """Paid places of a DK double-up with `entries` entries (exact on the 2026-10-20 uploads:
    125 / 287, 500 / 1,149, 200 / 459)."""
    return max(1, min(int(entries), int(round(DU_PAID_FRAC * int(entries)))))


def _history_entry(conn: sqlite3.Connection, contest_id: str) -> tuple[str | None, float | None]:
    """(contest name, entry fee) from the owner's contest-history entries ("history: NBA $2
    Double Up (1/3)"), for contests known only from an upload."""
    try:
        r = conn.execute("SELECT note, entry_fee FROM dfs_entries WHERE contest_id=? AND note "
                         "LIKE 'history:%' LIMIT 1", (str(contest_id),)).fetchone()
    except sqlite3.OperationalError:
        return None, None
    if r is None:
        return None, None
    name = (r[0] or "").replace("history:", "").split(" (")[0].strip() or None
    return name, _f(r[1])


def load_contest(conn: sqlite3.Connection, contest_id: str) -> Contest | None:
    """A captured contest. A contest known only from an uploaded standings file (no name, fee or
    payout table) takes its name and fee from the owner's contest history; a double-up among
    those gets DK's double-up table (source 'inferred'), so it is classified as cash."""
    r = conn.execute("SELECT * FROM dfs_contests WHERE contest_id=?", (str(contest_id),)).fetchone()
    if r is None:
        return None
    d = dict(r)
    pays = json.loads(d["payout_json"] or "[]")
    try:
        attrs = json.loads(d["attrs_json"] or "{}")
    except ValueError:
        attrs = {}
    m = int(d["field_size"] or 0)
    name, fee, source = (d["name"] or "").strip(), float(d["entry_fee"] or 0), "captured"
    if not pays and (not name or not fee):
        h_name, h_fee = _history_entry(conn, str(d["contest_id"]))
        name, fee = name or (h_name or ""), fee or (h_fee or 0.0)
        if is_double_up(name) and m >= 2 and fee:
            pays = [{"min_pos": 1, "max_pos": du_paid_places(m), "payout": 2.0 * fee}]
            source = "inferred"
    arch = classify_contest({"attrs": {k: attrs.get(k) for k in ("lobby", "selected_as")},
                             "payouts": pays, "field_size": m})
    return Contest(str(d["contest_id"]), name, fee, m, pays, arch, source)


def slate_contests(conn: sqlite3.Connection, slate_id) -> list[Contest]:
    ids = [r[0] for r in conn.execute("SELECT contest_id FROM dfs_contests WHERE slate_id=? "
                                      "ORDER BY field_size", (str(slate_id),))]
    return [c for c in (load_contest(conn, i) for i in ids) if c and c.field_size > 1]


def preset_contest(kind: str = "double_up", field_size: int = 1000, entry_fee: float = 1.0
                   ) -> Contest:
    """Described contests (§5.5.4): a double-up paying 2x the fee to 44.4 % of the field (the
    captured DK double-ups pay 43.5-46.5 %), or the fallback GPP ladder (top 0.1 % / 1 % / 20 %
    sharing an 86 % prize pool 30/25/45). Labelled source='preset'."""
    m = int(field_size)
    if kind == "double_up":
        paid = max(1, int(round(m * 0.444)))
        return Contest(None, f"preset double-up ({m})", entry_fee, m,
                       [{"min_pos": 1, "max_pos": paid, "payout": 2 * entry_fee}], "cash",
                       "preset")
    pool = 0.86 * m * entry_fee
    t01, t1, t20 = max(1, m // 1000), max(2, m // 100), max(3, m // 5)
    tiers = [(1, t01, 0.30 * pool / t01), (t01 + 1, t1, 0.25 * pool / max(1, t1 - t01)),
             (t1 + 1, t20, 0.45 * pool / max(1, t20 - t1))]
    return Contest(None, f"preset GPP ({m})", entry_fee, m,
                   [{"min_pos": a, "max_pos": b, "payout": round(p, 2)} for a, b, p in tiers],
                   "small_gpp" if m <= SMALL_GPP_MAX else "large_gpp", "preset")


def default_contests(conn: sqlite3.Connection, slate_id) -> tuple[Contest, Contest]:
    """(cash, GPP) reference contests of a slate: the largest captured double-up and the largest
    captured GPP; presets when none is captured."""
    cs = slate_contests(conn, slate_id)
    cash = [c for c in cs if c.archetype == "cash"]
    gpp = [c for c in cs if c.archetype != "cash"]
    return (max(cash, key=lambda c: c.field_size) if cash else preset_contest("double_up", 1000),
            max(gpp, key=lambda c: c.field_size) if gpp else preset_contest("gpp", 10_000))


# ============================================================================== field

@dataclass
class Field:
    """A sampled field proxy. `lineups` (U x slots) are player indices into `pids` for the unique
    lineups (slot order = `slots`), `counts` their multiplicity in the F-lineup sample, of which
    `opt_counts` came from the optimizer-like part. Every sampled lineup stands for
    field_size / F entries."""
    pids: list[str]
    slots: tuple[str, ...]
    lineups: np.ndarray
    counts: np.ndarray
    opt_counts: np.ndarray
    salary: np.ndarray
    field_size: int
    mix: float
    archetype: str
    ownership: OwnershipProxy
    n_solves: int
    opt_keys: dict              # lineup key -> frequency among the optimizer-like solves
    dupe_c: float               # product-form constant of the sampled part (§5.5.6)
    cap: int
    slot_salary: np.ndarray     # (players x distinct slot names) salary in that slot
    seconds: float = 0.0
    info: dict = dc_field(default_factory=dict)
    label: str = LABEL
    version: str = FIELD_VERSION
    opt_solves: list = dc_field(default_factory=list, repr=False)   # the optimizer-like solves

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def realized_ownership(self) -> np.ndarray:
        own = np.zeros(len(self.pids))
        np.add.at(own, self.lineups.ravel(), np.repeat(self.counts, self.lineups.shape[1]))
        return own / self.n

    def salary_use(self) -> dict:
        s = np.repeat(self.salary, self.counts)
        return {"mean_left": float(self.cap - s.mean()),
                "share_ge_98": float(np.mean(s >= 0.98 * self.cap)),
                "share_ge_99": float(np.mean(s >= 0.99 * self.cap)),
                "share_ge_96": float(np.mean(s >= 0.96 * self.cap)),
                "min": int(s.min()), "max": int(s.max())}

    def summary(self) -> dict:
        real = self.realized_ownership()
        tgt = self.ownership.own
        return {"label": self.label, "version": self.version, "archetype": self.archetype,
                "mix_optimizer_like": self.mix,
                "mix_realized": float(self.opt_counts.sum()) / max(1, self.n),
                "field_size": self.field_size, "sampled": self.n,
                "unique": int(len(self.counts)), "n_solves": self.n_solves,
                "own_target_mae_pp": float(np.mean(np.abs(real - tgt)[(real > 0) | (tgt > 0)])
                                           * 100),
                "salary_use": self.salary_use(), "seconds": round(self.seconds, 2), **self.info}


def field_version(op: "OwnershipProxy | None" = None, k: bool = False) -> str:
    """The field proxy's version: FIELD_VERSION, + FIELD_K_TAG for K fields (D5), + SD_CPT_TAG
    when the ownership proxy carries the Showdown captain term (B6)."""
    return (FIELD_VERSION + (FIELD_K_TAG if k else "")
            + (SD_CPT_TAG if op is not None and SD_CPT_TAG in op.version else ""))


def _key(row: np.ndarray, cpt_first: bool) -> tuple:
    return (int(row[0]),) + tuple(sorted(int(x) for x in row[1:])) if cpt_first \
        else tuple(sorted(int(x) for x in row))


def _canon(L: np.ndarray, cpt_first: bool) -> np.ndarray:
    """Canonical rows for identity (sorted players; Showdown keeps the captain first)."""
    if cpt_first:
        return np.concatenate([L[:, :1], np.sort(L[:, 1:], axis=1)], axis=1)
    return np.sort(L, axis=1)


def undominated(proj: np.ndarray, salary: np.ndarray, masks: np.ndarray, k: int) -> np.ndarray:
    """Players that can be in an optimal lineup: drop i when >= k other players cost no more,
    project at least as much (strictly better in one, ties broken by index) and are eligible for
    every slot i is. Exact for rosters without multipliers when team / game minimums don't bind
    (CP-SAT still enforces them on the survivors)."""
    n = len(proj)
    i = np.arange(n)
    le_sal = salary[None, :] <= salary[:, None]
    ge_proj = proj[None, :] >= proj[:, None]
    strict = (salary[None, :] < salary[:, None]) | (proj[None, :] > proj[:, None]) | \
        (i[None, :] < i[:, None])
    sup = (masks[None, :] & masks[:, None]) == masks[:, None]
    dom = le_sal & ge_proj & strict & sup
    np.fill_diagonal(dom, False)
    return dom.sum(1) < k


def _opt_lineups(pool_players: Sequence[PoolPlayer], rules: Rules, pub: Mapping[str, float],
                 n_solves: int, noise: float, rng: np.random.Generator,
                 threads: int = SOLVE_THREADS) -> list[list[tuple[str, str]]]:
    """`n_solves` CP-SAT optimal lineups on public projection x exp(N(0, noise²)) (dominated
    players pruned first on Classic rosters, which makes big slates ~5x faster)."""
    from . import optimizer as O
    ids = list(pub)
    base = np.array([pub[i] for i in ids])
    by_id = {p.dk_player_id: p for p in pool_players}
    names = rules.distinct_slot_names
    masks = np.array([sum(1 << names.index(s) for s in by_id[i].slots if s in names)
                      for i in ids], np.int64)
    sal = np.array([by_id[i].salary for i in ids], float)
    prune = len(set(rules.slot_multiplier(s) for s in names)) == 1
    draws = []
    for _ in range(n_solves):
        v = base * np.exp(rng.normal(0, noise, len(ids)))
        keep = undominated(v, sal, masks, rules.n_slots) if prune else np.ones(len(ids), bool)
        draws.append({ids[j]: round(float(v[j]), 3) for j in np.flatnonzero(keep)})

    def solve(pr):
        players = [by_id[i] for i in pr]
        lu = O.optimize(players, pr, rules, time_limit_s=SOLVE_LIMIT_S)
        return None if lu is None else lu.slot_pairs()

    if threads > 1 and n_solves > 1:
        with ThreadPoolExecutor(threads) as ex:
            out = list(ex.map(solve, draws))
    else:
        out = [solve(d) for d in draws]
    return [o for o in out if o is not None]


class _Sampler:
    """Vectorised ownership sampler: slots filled in scarcity order from per-slot weights with
    salary feasibility and a per-lineup spend floor."""

    def __init__(self, rules: Rules, E: np.ndarray, S: np.ndarray, names: tuple[str, ...],
                 games: np.ndarray, teams: np.ndarray, full_spend: float):
        self.rules = rules
        self.E, self.S, self.names = E, S, names
        seq = sorted(rules.slot_names, key=lambda s: (FILL_ORDER.index(s) if s in FILL_ORDER
                                                       else 99))
        self.seq = [names.index(s) for s in seq]
        self.games, self.teams = games, teams
        self.full_spend = full_spend
        self.cap = rules.salary_cap
        n_slots = len(self.seq)
        # cheapest / dearest remaining completion per step (eligible salaries per slot)
        mins = [S[E[:, k], k].min() if E[:, k].any() else 0 for k in self.seq]
        maxs = [S[E[:, k], k].max() if E[:, k].any() else 0 for k in self.seq]
        self.min_rest = np.array([sum(mins[t + 1:]) for t in range(n_slots)])
        self.max_rest = np.array([sum(sorted(maxs[t + 1:], reverse=True)) for t in range(n_slots)])

    def floors(self, B: int, rng) -> np.ndarray:
        u = rng.random(B)
        f = np.where(u < self.full_spend, FLOOR_FULL,
                     np.where(u < self.full_spend + (1 - self.full_spend) * FLOOR_SPLIT,
                              FLOOR_MOST, FLOOR_REST))
        return f * self.cap

    def sample(self, W: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
        """(<= B, slots) player indices in `rules.slots` order; failed / invalid rows dropped."""
        n = W.shape[0]
        used = np.zeros(B)
        picked = np.zeros((B, n), bool)
        out = np.full((B, len(self.seq)), -1, np.int64)
        ok = np.ones(B, bool)
        floor = self.floors(B, rng)
        rows = np.arange(B)
        for t, k in enumerate(self.seq):
            room = self.cap - used - self.min_rest[t]
            need = floor - used - self.max_rest[t]
            sal = self.S[:, k][None, :]
            feas = (sal <= room[:, None]) & ~picked & self.E[:, k][None, :]
            soft = feas & (sal >= need[:, None])
            has_soft = soft.any(1)
            mask = np.where(has_soft[:, None], soft, feas)
            p = W[:, k][None, :] * mask
            cdf = np.cumsum(p, axis=1)
            tot = cdf[:, -1]
            ok &= tot > 0
            u = rng.random(B) * tot
            idx = np.minimum((cdf <= u[:, None]).sum(1), n - 1)
            out[:, t] = idx
            picked[rows, idx] = True
            used += self.S[idx, k]
        ok &= (used <= self.cap) & (used >= HARD_FLOOR * self.cap)
        out, used = out[ok], used[ok]
        # games / teams minimums
        keep = np.ones(len(out), bool)
        for codes, minimum in ((self.games, self.rules.min_games),
                               (self.teams, self.rules.min_teams)):
            if minimum and len(out):
                g = np.sort(codes[out], axis=1)
                keep &= 1 + (np.diff(g, axis=1) != 0).sum(1) >= minimum
        out = out[keep]
        # back to rules slot order (the sequence is a permutation of rules.slot_names)
        seq_names = [self.names[k] for k in self.seq]
        order = []
        taken = set()
        for s in self.rules.slot_names:
            j = next(j for j, nm in enumerate(seq_names) if nm == s and j not in taken)
            taken.add(j)
            order.append(j)
        return out[:, order]


def _strength(mix, archetype: str | None = None) -> float:
    """Optimizer-like share: None / "medium" = the archetype's default (field-proxy-1.1),
    "soft" / "sharp" or a number as given."""
    if mix is None or mix == "medium":
        return float(ARCHETYPES.get(archetype or "", {}).get("mix", DEFAULT_MIX))
    if isinstance(mix, str):
        return STRENGTHS[mix]
    return float(mix)


def build_field(pool: SlatePool | Sequence[PoolPlayer], inputs: pd.DataFrame,
                contest: Contest | None = None, *, rules: Rules | None = None,
                n: int | None = None, mix: float | str | None = None,
                archetype: str | None = None, ownership: OwnershipProxy | None = None,
                n_solves: int = N_SOLVES, seed: int = 0, ipf_iters: int = 3,
                threads: int = SOLVE_THREADS, opt_pairs: list | None = None) -> Field:
    """Sample a field proxy for `contest` (or a preset 10k GPP). `n` sampled lineups (default:
    min(field size, 20k), at least 5k so small contests can draw opponents per sim).
    `opt_pairs`: reuse these optimizer-like solves (another field's `opt_solves`; the K fields of
    `build_fields` share them) instead of solving."""
    t0 = time.perf_counter()
    rules = rules or (pool.rules if isinstance(pool, SlatePool) else None)
    if rules is None:
        raise ValueError("no rules")
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    contest = contest or preset_contest("gpp", 10_000)
    archetype = archetype or contest.archetype
    mix = _strength(mix, archetype)
    rng = np.random.default_rng(seed)
    if n is None:
        n = int(min(max(contest.field_size, FIELD_MIN), FIELD_MAX))
    op = ownership or ownership_proxy(inputs, players, rules, archetype=archetype)
    by_id = {p.dk_player_id: p for p in players}
    pids = op.pids
    idx_of = {p: i for i, p in enumerate(pids)}
    names, counts, E, S = _eligibility(op.inputs, by_id, rules)
    active = op.own > 0
    E &= active[:, None]
    games = pd.factorize(np.array([_game(by_id[p]) for p in pids], dtype=object))[0]
    teams = pd.factorize(np.array([by_id[p].team or f"?{p}" for p in pids], dtype=object))[0]
    cpt_first = rules.is_showdown and rules.slot_names[0] == "CPT"
    A = ARCHETYPES[archetype]

    # ---- optimizer-like part
    n_opt = int(round(mix * n))
    pub = op.inputs["pub_proj"].to_numpy(float)
    st = op.inputs["status"].fillna("").astype(str).str.upper().to_numpy()
    pubd = {p: float(pub[i] * (0.6 if st[i] in Q_STATUSES else 1.0))
            for i, p in enumerate(pids) if active[i]}
    if opt_pairs is not None:
        opt_pairs = list(opt_pairs) if n_opt > 0 else []
    else:
        opt_pairs = []
        if n_opt > 0 and n_solves > 0:
            opt_players = [by_id[p] for p in pubd]
            opt_pairs = _opt_lineups(opt_players, rules, pubd, n_solves, A["noise"], rng,
                                     threads)
    if opt_pairs:
        opt_L = np.zeros((len(opt_pairs), len(rules.slot_names)), np.int64)
        for r, pairs in enumerate(opt_pairs):
            used = set()
            for slot, pid in pairs:
                j = next(j for j, s in enumerate(rules.slot_names) if s == slot and j not in used)
                used.add(j)
                opt_L[r, j] = idx_of[pid]
        pick = rng.integers(0, len(opt_L), n_opt)
        opt_sample = opt_L[pick]
        okeys: dict = {}
        for row in opt_L:
            k = _key(row, cpt_first)
            okeys[k] = okeys.get(k, 0) + 1 / len(opt_L)
    else:
        n_opt = 0
        opt_sample = np.zeros((0, len(rules.slot_names)), np.int64)
        okeys = {}
    n_own = n - n_opt

    # ---- ownership-sampled part (IPF so the whole field reproduces the proxy ownership)
    sampler = _Sampler(rules, E, S, names, games, teams, A["full_spend"])
    W = op.by_slot.copy()
    W[~E] = 0
    opt_own = np.zeros(len(pids))
    if n_opt:
        np.add.at(opt_own, opt_sample.ravel(), 1.0)
        opt_own /= n_opt
    frac = n_opt / n if n else 0.0
    tgt = np.clip((op.own - frac * opt_own) / max(1e-9, 1 - frac), 0.0, op.cap)
    tgt = np.where(active, np.maximum(tgt, 1e-4), 0.0)
    pilot = max(4000, min(20000, n_own))
    for _ in range(ipf_iters if n_own else 0):
        L = sampler.sample(W, pilot, rng)
        real = np.bincount(L.ravel(), minlength=len(pids)) / max(1, len(L))
        adj = np.where(real > 0, (tgt / np.maximum(real, 1e-6)) ** 0.8, 1.0)
        W *= np.clip(adj, 0.2, 5.0)[:, None]
    own_rows = []
    have = 0
    tries = 0
    while have < n_own and tries < 20:
        B = int((n_own - have) * 1.3) + 500
        L = sampler.sample(W, B, rng)
        own_rows.append(L)
        have += len(L)
        tries += 1
    own_sample = (np.concatenate(own_rows)[:n_own] if own_rows
                  else np.zeros((0, len(rules.slot_names)), np.int64))
    if len(own_sample) < n_own:
        log.warning("field sampler produced %d of %d lineups", len(own_sample), n_own)

    # ---- unique lineups, counts, product-form dupe constant
    allL = np.concatenate([opt_sample, own_sample])
    src = np.concatenate([np.ones(len(opt_sample), bool), np.zeros(len(own_sample), bool)])
    canon = _canon(allL, cpt_first)
    uniq, first, inv, cnt = np.unique(canon, axis=0, return_index=True, return_inverse=True,
                                      return_counts=True)
    inv = inv.ravel()
    lineups = allL[first]
    opt_cnt = np.bincount(inv[src], minlength=len(uniq))
    slot_k = [names.index(s) for s in rules.slot_names]
    sal_u = np.array([sum(S[r[j], slot_k[j]] for j in range(len(r))) for r in lineups])
    dupe_c = _dupe_constant(own_sample, cpt_first, op, S, slot_k, rules.salary_cap)
    f = Field(pids=pids, slots=rules.slot_names, lineups=lineups, counts=cnt, opt_counts=opt_cnt,
              salary=sal_u, field_size=contest.field_size, mix=mix, archetype=archetype,
              ownership=op, n_solves=len(opt_pairs), opt_keys=okeys, dupe_c=dupe_c,
              cap=rules.salary_cap, slot_salary=S, opt_solves=opt_pairs)
    if SD_CPT_TAG in op.version:        # B6 captain term ON (Showdown only)
        f.version = field_version(op)
    f.info["contest"] = {"contest_id": contest.contest_id, "name": contest.name,
                         "source": contest.source}
    f.info["opt_unique"] = len(okeys)
    f.seconds = time.perf_counter() - t0
    return f


def _log_g(L: np.ndarray, op: OwnershipProxy, S: np.ndarray, slot_k: list[int], cap: int,
           cpt_first: bool) -> np.ndarray:
    """log g(L) = Σ log own_i (captain: CPT share) + log h(salary left) for lineups (rows)."""
    own = np.maximum(op.own, 1e-6)
    lg = np.log(own)[L]
    if cpt_first:
        k = op.slot_names.index("CPT")
        flex = [j for j, s in enumerate(op.slot_names) if s != "CPT"]
        lg[:, 0] = np.log(np.maximum(op.by_slot[L[:, 0], k], 1e-6))
        lg[:, 1:] = np.log(np.maximum(op.by_slot[:, flex].sum(1), 1e-6))[L[:, 1:]]
    sal = sum(S[L[:, j], slot_k[j]] for j in range(L.shape[1]))
    h = np.where(cap - sal <= 1000, 0.0, np.log(0.25))
    return lg.sum(1) + h


def _dupe_constant(own_sample: np.ndarray, cpt_first: bool, op: OwnershipProxy, S: np.ndarray,
                   slot_k: list[int], cap: int) -> float:
    """c of q(L) = c * g(L) for the ownership-sampled part, by the collision estimator
    Σ_L q_L² = E_{L~q}[q_L] = c * E_{L~q}[g(L)]: c = (pairs of identical draws / all pairs) /
    mean g over the draws. NaN when the sample has fewer than 10 collisions."""
    F = len(own_sample)
    if F < 2:
        return float("nan")
    _, cnt = np.unique(_canon(own_sample, cpt_first), axis=0, return_counts=True)
    coll = float((cnt * (cnt - 1)).sum())
    if coll < 10 * 2:
        return float("nan")
    c2 = coll / (F * (F - 1))
    lg = _log_g(own_sample, op, S, slot_k, cap, cpt_first)
    m = lg.max()
    mean_g = float(np.exp(lg - m).mean()) * math.exp(m)
    return c2 / mean_g if mean_g > 0 else float("nan")


# ============================================================================== K fields (D5)

def own_sd_for(op: OwnershipProxy, own_unc: Mapping[str, float] | None = None,
               sd: float | None = None) -> np.ndarray:
    """Per-player log-odds sd of the ownership noise: the archetype's OWN_SD (or `sd`) x the
    player's `own_unc` multiplier (late_news_tag.own_uncertainty; default 1.0)."""
    base = OWN_SD.get(op.archetype.split("/")[0], OWN_SD["large_gpp"]) if sd is None else sd
    u = own_unc or {}
    return np.array([base * float(u.get(str(p), 1.0)) for p in op.pids], float)


def mean_shift(p: np.ndarray, sd: np.ndarray, nodes: int = 40) -> np.ndarray:
    """c_i with E[sigmoid(logit p_i + c_i + sd_i z)] = p_i, z ~ N(0, 1) (Gauss-Hermite,
    bisection): the log-odds noise then leaves each player's expected ownership at the proxy
    (uncentred noise would move ownership from the chalk to the long tail on average, i.e. a
    weaker field, not just a more uncertain one)."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    lp = np.log(p / (1 - p))
    span = sd * sd + 4.0
    lo, hi = -span, span
    for _ in range(60):
        mid = (lo + hi) / 2
        m = (w[None, :] / (1.0 + np.exp(-(lp + mid)[:, None] - sd[:, None] * x[None, :]))).sum(1)
        below = m < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(sd > 0, (lo + hi) / 2, 0.0)


def perturb_ownership(op: OwnershipProxy, sd: np.ndarray, rng: np.random.Generator
                      ) -> OwnershipProxy:
    """One ownership draw: logit(own'_i) = logit(own_i) + c_i + sd_i z_i (own clipped to
    [OWN_P_CLIP, 1 - OWN_P_CLIP]; c_i = `mean_shift`, so E[own'_i] = own_i; players with no
    ownership stay at 0), each player's per-slot row scaled by own'_i / own_i, then re-balanced
    by `sinkhorn` to the slot totals and the row cap. Σ own stays the number of slots."""
    own = op.own
    act = own > 0
    p = np.clip(own, OWN_P_CLIP, 1 - OWN_P_CLIP)
    z = rng.standard_normal(len(own))
    lo = np.log(p / (1 - p)) + mean_shift(p, sd) + sd * z
    new = 1.0 / (1.0 + np.exp(-lo))
    r = np.where(act, new / p, 0.0)
    X = sinkhorn(op.by_slot * r[:, None], op.slot_counts, op.cap)
    return dc_replace(op, own=X.sum(1), by_slot=X, version=op.version + FIELD_K_TAG)


@dataclass
class FieldSet:
    """K field proxies of one contest (ownership uncertainty, D5): field k is scored on the sim
    rows s with s mod K == k (`evaluate`), so chances average over the fields. `ownership` is
    the un-perturbed proxy (exposure tables, ownership penalties and display use it)."""
    fields: list
    ownership: OwnershipProxy
    sd: np.ndarray
    own_unc: dict
    seconds: float = 0.0
    label: str = LABEL
    version: str = FIELD_VERSION + FIELD_K_TAG

    @property
    def k(self) -> int:
        return len(self.fields)

    def __getattr__(self, name):        # pids, slots, field_size, archetype, mix, cap, ...
        if name in ("fields", "__setstate__", "__getstate__"):
            raise AttributeError(name)
        return getattr(self.fields[0], name)

    def rows_of(self, S: int, k: int) -> np.ndarray:
        """The sim rows (of S) scored against field k."""
        return np.arange(k, S, self.k)

    def summary(self) -> dict:
        out = self.fields[0].summary()
        base = self.ownership.own
        act = base > 0
        out["version"] = self.version
        out["field_k"] = {
            "k": self.k, "sd": float(OWN_SD.get(self.ownership.archetype, np.nan)),
            "own_unc_players": int(sum(1 for v in self.own_unc.values() if float(v) != 1.0)),
            "own_draw_mae_pp": [round(float(np.mean(np.abs(f.ownership.own - base)[act]) * 100),
                                      2) for f in self.fields],
            "realized_mae_pp": [round(float(np.mean(np.abs(f.realized_ownership() - base)[act])
                                            * 100), 2) for f in self.fields],
            "unique": [int(len(f.counts)) for f in self.fields],
            "seconds": round(self.seconds, 2)}
        return out


def build_fields(pool: SlatePool | Sequence[PoolPlayer], inputs: pd.DataFrame,
                 contest: Contest | None = None, *, k: int | None = None,
                 own_unc: Mapping[str, float] | None = None, rules: Rules | None = None,
                 n: int | None = None, mix: float | str | None = None,
                 archetype: str | None = None, ownership: OwnershipProxy | None = None,
                 n_solves: int = N_SOLVES, seed: int = 0, ipf_iters: int = 3,
                 threads: int = SOLVE_THREADS, sd: float | None = None) -> FieldSet:
    """K (default `field_k()`, at least 2) fields, each sampled from its own ownership draw
    (`perturb_ownership`), sharing one set of optimizer-like solves (they depend on the public
    projection, not on ownership). Field j uses seed + j; the ownership draws their own stream."""
    t0 = time.perf_counter()
    K = max(2, int(k if k is not None else (field_k() or FIELD_K_ON)))
    rules = rules or (pool.rules if isinstance(pool, SlatePool) else None)
    if rules is None:
        raise ValueError("no rules")
    players = pool.players if isinstance(pool, SlatePool) else tuple(pool)
    contest = contest or preset_contest("gpp", 10_000)
    archetype = archetype or contest.archetype
    op = ownership or ownership_proxy(inputs, players, rules, archetype=archetype)
    unc = {str(a): float(b) for a, b in (own_unc or {}).items()}
    sdv = own_sd_for(op, unc, sd)
    rng = np.random.default_rng([int(seed), 0xD5])
    fields: list[Field] = []
    shared = None
    for j in range(K):
        opj = perturb_ownership(op, sdv, rng)
        f = build_field(players, inputs, contest, rules=rules, n=n, mix=mix,
                        archetype=archetype, ownership=opj, n_solves=n_solves, seed=seed + j,
                        ipf_iters=ipf_iters, threads=threads, opt_pairs=shared)
        f.version = field_version(op, k=True)
        shared = f.opt_solves
        fields.append(f)
    fs = FieldSet(fields, op, sdv, unc, seconds=time.perf_counter() - t0)
    if SD_CPT_TAG in op.version:        # B6 captain term ON (Showdown only)
        fs.version = field_version(op, k=True)
    return fs


def _evaluate_k(fs: FieldSet, fp: np.ndarray, contest: Contest, lineups: Sequence, *,
                seed: int, chunk, tiers: bool, own_entries: int, per_sim: bool) -> FieldEval:
    """`evaluate` against a FieldSet: rows s mod K -> field k, then reassembled in row order;
    per-lineup chances / payouts are the row-weighted means over the fields (= the mean over
    all rows), with the per-field values (`*_k`) and their range (`*_lo` / `*_hi`). Fewer rows
    than fields (e.g. one row of actual points): every field scores every row and per-row
    outputs are averaged over the fields."""
    t0 = time.perf_counter()
    S = fp.shape[0]
    K = fs.k
    lus = list(lineups)
    if S >= K:
        parts = [(fs.rows_of(S, j), fs.fields[j]) for j in range(K)]
    else:
        parts = [(np.arange(S), f) for f in fs.fields]
    evs = [evaluate(f, fp[r], contest, lus, seed=seed + j, chunk=chunk, tiers=tiers,
                    own_entries=own_entries, per_sim=per_sim)
           for j, (r, f) in enumerate(parts)]
    if S >= K:
        w = np.array([len(r) for r, _ in parts], float) / S
        cash = np.empty(S)
        top1 = np.empty(S)
        tl = np.empty((evs[0].tier_lines.shape[0], S))
        for (r, _), ev in zip(parts, evs):
            cash[r], top1[r], tl[:, r] = ev.cash_line, ev.top1_line, ev.tier_lines
        keep = None
        if per_sim and evs[0].per_sim is not None:
            keep = {key: np.empty((S,) + v.shape[1:], v.dtype)
                    for key, v in evs[0].per_sim.items()}
            for (r, _), ev in zip(parts, evs):
                for key in keep:
                    keep[key][r] = ev.per_sim[key]
    else:
        w = np.full(K, 1.0 / K)
        cash = np.mean([ev.cash_line for ev in evs], 0)
        top1 = np.mean([ev.top1_line for ev in evs], 0)
        tl = np.mean([ev.tier_lines for ev in evs], 0)
        keep = None
        if per_sim and evs[0].per_sim is not None:
            keep = {"score": evs[0].per_sim["score"],
                    "above": np.mean([ev.per_sim["above"] for ev in evs], 0).astype(np.float32),
                    "tied": np.mean([ev.per_sim["tied"] for ev in evs], 0).astype(np.float32)}
    res = []
    for i in range(len(lus)):
        per = [ev.lineups[i] for ev in evs]
        d = {}
        for key in ("p_cash", "p_top1", "e_payout", "mean"):
            d[key] = float(sum(wk * x[key] for wk, x in zip(w, per)))
        d["e_payout_se"] = float(math.sqrt(sum((wk * x["e_payout_se"]) ** 2
                                               for wk, x in zip(w, per))))
        d["e_roi"] = (d["e_payout"] / contest.entry_fee - 1) if contest.entry_fee else None
        d["dupes"] = float(np.mean([x["dupes"] for x in per]))
        d["dupe_band"] = dupe_band(d["dupes"])
        d["unknown_players"] = per[0]["unknown_players"]
        for key in ("p_cash", "p_top1", "e_payout", "dupes"):
            vals = [float(x[key]) for x in per]
            d[key + "_k"] = vals
            if key != "dupes":
                d[key + "_lo"], d[key + "_hi"] = min(vals), max(vals)
        res.append({k2: d[k2] for k2 in ("p_cash", "p_top1", "e_payout", "e_payout_se", "e_roi",
                                          "dupes", "dupe_band", "mean", "unknown_players")}
                   | {k2: v for k2, v in d.items() if k2.endswith(("_k", "_lo", "_hi"))})
    info = dict(evs[0].info)
    info.update({"n_sims": S, "field_k": K, "k_mode": "rows" if S >= K else "all",
                 "seconds": round(time.perf_counter() - t0, 2)})
    return FieldEval(cash, top1, tl, res, contest, info, per_sim=keep)


# ============================================================================== sims

def align_sims(sims: Mapping[str, np.ndarray], pids: Sequence[str], n_sims: int | None = None,
               seed: int = 0) -> tuple[np.ndarray, dict]:
    """(S x n) float32 DK points for `pids` (DK ids) from a stored sims npz dict (`fp`, `dk_id`).
    Players without a column score 0 (reported). `n_sims` subsamples sims deterministically."""
    fp = sims["fp"]
    dk = [str(x) for x in sims["dk_id"]]
    col = {}
    for j, d in enumerate(dk):
        if d and d != "nan" and d not in col:
            col[d] = j
    S = fp.shape[0]
    rows = np.arange(S)
    if n_sims and n_sims < S:
        rows = np.sort(np.random.default_rng(seed).choice(S, n_sims, replace=False))
    out = np.zeros((len(rows), len(pids)), np.float32)
    miss = []
    for i, p in enumerate(pids):
        j = col.get(str(p))
        if j is None:
            miss.append(p)
        else:
            out[:, i] = fp[rows, j]
    return out, {"n_sims": len(rows), "of": S, "missing": miss}


# ============================================================================== evaluate

@dataclass
class FieldEval:
    cash_line: np.ndarray           # (S,) score at the last paid rank
    top1_line: np.ndarray           # (S,) score at the top-1 % rank
    tier_lines: np.ndarray          # (tiers, S) score at each payout tier's last rank
    lineups: list[dict]             # per candidate: p_cash, p_top1, e_payout, e_roi, dupes...
    contest: Contest
    info: dict
    label: str = LABEL
    per_sim: dict | None = None     # evaluate(per_sim=True): (S x L) "score" (1/64-point units
                                    # + BIAS, int32), "above" / "tied" (other entries, float32)

    def cash_line_summary(self) -> dict:
        q = np.percentile(self.cash_line, [10, 25, 50, 75, 90])
        t = np.percentile(self.top1_line, [10, 50, 90])
        return {"cash_p10": q[0], "cash_p25": q[1], "cash_p50": q[2], "cash_p75": q[3],
                "cash_p90": q[4], "cash_mean": float(self.cash_line.mean()),
                "top1_p10": t[0], "top1_p50": t[1], "top1_p90": t[2]}


def _as_pairs(lu) -> list[tuple[str, str]]:
    if hasattr(lu, "slot_pairs"):
        return lu.slot_pairs()
    if isinstance(lu, str):
        lu = json.loads(lu)
    if lu and isinstance(lu[0], Mapping):
        return [(d["slot"], str(d["player_id"])) for d in lu]
    return [(s, str(p)) for s, p in lu]


def _cand_matrix(field: Field, lineups: Sequence, rules_slots: tuple[str, ...]
                 ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(weights (L x n) int, index rows (L x slots) or -1, unknown counts)."""
    idx = {p: i for i, p in enumerate(field.pids)}
    mult = {"CPT": 3}
    Wc = np.zeros((len(lineups), len(field.pids)), np.float32)
    rows = np.full((len(lineups), len(rules_slots)), -1, np.int64)
    unknown = []
    for r, lu in enumerate(lineups):
        pairs = _as_pairs(lu)
        u = 0
        used = set()
        for slot, pid in pairs:
            i = idx.get(str(pid))
            if i is None:
                u += 1
                continue
            Wc[r, i] += mult.get(slot, WUNIT)
            j = next((j for j, s in enumerate(rules_slots) if s == slot and j not in used), None)
            if j is not None:
                used.add(j)
                rows[r, j] = i
        unknown.append(u)
    return Wc, rows, unknown


def dupes(field: Field, rows: np.ndarray) -> np.ndarray:
    """Expected identical lineups among the field_size - 1 other entries, per candidate row
    (player indices in slot order; rows with an unknown player get 0). Optimizer-like part:
    exact frequency among its solves. Ownership-sampled part: the empirical frequency when the
    lineup appears >= 3 times in the sample, else c * Π own * h(salary left) (§5.5.6)."""
    cpt_first = field.slots[0] == "CPT"
    n_own = int(field.counts.sum() - field.opt_counts.sum())
    w_opt = float(field.opt_counts.sum()) / max(1, int(field.counts.sum()))   # realized share
    own_cnt = {}
    canon = _canon(field.lineups, cpt_first)
    for r, c, oc in zip(canon, field.counts, field.opt_counts):
        own_cnt[tuple(int(x) for x in r)] = int(c - oc)
    op = field.ownership
    S = field.slot_salary
    slot_k = [op.slot_names.index(s) for s in field.slots]
    out = np.zeros(len(rows))
    M = max(0, field.field_size - 1)
    for i, row in enumerate(rows):
        if (row < 0).any():
            continue
        k = _key(row, cpt_first)
        q_opt = field.opt_keys.get(k, 0.0)
        cnt = own_cnt.get(k, 0)
        if n_own and cnt >= 3:
            q_own = cnt / n_own
        elif n_own and math.isfinite(field.dupe_c):
            lg = _log_g(row[None, :], op, S, slot_k, field.cap, cpt_first)[0]
            q_own = min(1.0, field.dupe_c * math.exp(lg))
        else:
            q_own = cnt / n_own if n_own else 0.0
        out[i] = M * (w_opt * q_opt + (1 - w_opt) * q_own)
    return out


def evaluate(field: Field, fp: np.ndarray, contest: Contest, lineups: Sequence = (), *,
             seed: int = 0, chunk: int | None = None, tiers: bool = True,
             own_entries: int = 1, per_sim: bool = False) -> FieldEval:
    """Score the field and candidate lineups on per-sim player DK points `fp` (S x n, columns =
    `field.pids`). See the module docstring for the definitions.

    `own_entries` is how many of the contest's M entries are ours (the other entries are
    M - own_entries; default 1, a single entry). `per_sim=True` also returns, per sim and
    candidate, the score and the number of other entries above / tied with it (the inputs of a
    joint multi-entry payout, proj/dfs/portfolio.py). A `FieldSet` (K fields, D5) is scored
    row-wise per field and reassembled (`_evaluate_k`)."""
    if isinstance(field, FieldSet):
        return _evaluate_k(field, fp, contest, lineups, seed=seed, chunk=chunk, tiers=tiers,
                           own_entries=own_entries, per_sim=per_sim)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    S_all, n = fp.shape
    M = int(contest.field_size)
    if M < 2:
        raise ValueError("contest field size must be >= 2")
    m_o = max(1, M - max(1, int(own_entries)))
    R = min(max(1, contest.paid), M)
    R1 = max(1, int(math.ceil(0.01 * M)))
    tier_ranks = sorted({min(int(t["max_pos"]), M) for t in contest.payouts})
    pay = contest.ladder()
    cum = np.concatenate([[0.0], np.cumsum(pay[1:])])       # cum[r] = Σ_{1..r}

    # integer units: fp in 1/32, slot weights x2 -> scores in 1/64 point
    fq = np.rint(fp.astype(np.float32) * UNIT).astype(np.float32)
    A = np.zeros((len(field.lineups), n), np.float32)
    for j, s in enumerate(field.slots):
        np.add.at(A, (np.arange(len(field.lineups)), field.lineups[:, j]),
                  3 if s == "CPT" else WUNIT)
    As = sparse.csr_matrix(A)
    inv = np.repeat(np.arange(len(field.counts)), field.counts)
    F = len(inv)
    Wc, rows, unknown = _cand_matrix(field, list(lineups), field.slots)
    L = len(rows)
    exact = m_o == F                 # the sample IS the other M-1 entries: used as is
    explicit = m_o < F               # opponents drawn per sim from the sample
    n_o = m_o if (explicit or exact) else F
    scale = 1.0 if (explicit or exact) else m_o / F
    if chunk is None:
        chunk = int(max(32, min(4096, 6_000_000 // max(n_o, 1))))
    chunk = min(chunk, MAX_ROWS)
    cash = np.empty(S_all, np.float64)
    top1 = np.empty(S_all, np.float64)
    tl = np.empty((len(tier_ranks), S_all), np.float64) if tiers else np.zeros((0, S_all))
    p_cash = np.zeros(L)
    p_top1 = np.zeros(L)
    e_pay = np.zeros(L)
    e_pay2 = np.zeros(L)
    score_sum = np.zeros(L)
    denom = UNIT * WUNIT
    keep = {"score": np.zeros((S_all, L), np.int32), "above": np.zeros((S_all, L), np.float32),
            "tied": np.zeros((S_all, L), np.float32)} if per_sim and L else None

    def pos_of(rank: int) -> int:     # ascending index of the rank-th largest of n_o values
        k = rank if scale == 1.0 else int(math.ceil(rank / scale))
        return int(np.clip(n_o - k, 0, n_o - 1))

    p_cash_pos, p_top_pos = pos_of(R), pos_of(R1)
    tier_pos = [pos_of(r) for r in tier_ranks]
    for a in range(0, S_all, chunk):
        b = min(S_all, a + chunk)
        X = fq[a:b]
        fs = np.ascontiguousarray((As @ X.T).T, dtype=np.int32)  # (c, U) exact integers
        fs += np.int32(BIAS)
        if explicit:
            pick = rng.integers(0, F, (b - a, n_o))
            others = np.take_along_axis(fs, inv[pick], axis=1)
        else:
            others = fs[:, inv]
        others.sort(axis=1)
        cash[a:b] = others[:, p_cash_pos]
        top1[a:b] = others[:, p_top_pos]
        if tiers:
            tl[:, a:b] = others[:, tier_pos].T
        if L:
            cs = (X @ Wc.T).astype(np.int64) + BIAS                # (c, L)
            score_sum += cs.sum(0)
            p_cash += (cs >= others[:, [p_cash_pos]]).sum(0)
            p_top1 += (cs >= others[:, [p_top_pos]]).sum(0)
            ridx = np.arange(b - a, dtype=np.int64)[:, None]
            others += (ridx * ROW_OFF).astype(np.int32)        # in place: rows stay sorted
            flat = others.ravel()
            q = (cs + ridx * ROW_OFF).astype(np.int32).ravel()   # same dtype: no copy of flat
            start = ridx * n_o
            le = np.searchsorted(flat, q, side="right").reshape(cs.shape) - start
            lt = np.searchsorted(flat, q, side="left").reshape(cs.shape) - start
            gt = n_o - le
            eq = le - lt
            if explicit or exact:
                above = gt.astype(np.float64)
                tied = eq.astype(np.float64)
            else:
                p = rng.beta(gt + 1.0, F - gt + 1.0)
                above = rng.binomial(m_o, p).astype(np.float64)
                tied = eq * scale
            if keep is not None:
                keep["score"][a:b] = cs
                keep["above"][a:b] = above
                keep["tied"][a:b] = tied
            lo = np.clip(np.rint(above) + 1, 1, M).astype(np.int64)
            hi = np.clip(np.rint(above + tied) + 1, 1, M).astype(np.int64)
            won = (cum[hi] - cum[lo - 1]) / (hi - lo + 1)
            e_pay += won.sum(0)
            e_pay2 += (won * won).sum(0)
    to_pts = lambda v: (v - BIAS) / denom                      # noqa: E731
    cash_pts, top1_pts = to_pts(cash), to_pts(top1)
    tl_pts = to_pts(tl) if tiers else tl
    dp = dupes(field, rows) if L else np.zeros(0)
    res = []
    for i in range(L):
        ep = e_pay[i] / S_all
        se = math.sqrt(max(0.0, e_pay2[i] / S_all - ep * ep) / S_all)
        res.append({"p_cash": p_cash[i] / S_all, "p_top1": p_top1[i] / S_all, "e_payout": ep,
                    "e_payout_se": se,
                    "e_roi": (ep / contest.entry_fee - 1) if contest.entry_fee else None,
                    "dupes": float(dp[i]), "dupe_band": dupe_band(dp[i]),
                    "mean": (score_sum[i] / S_all - BIAS) / denom, "unknown_players": unknown[i]})
    info = {"n_sims": S_all, "field_sample": F, "opponents": n_o,
            "mode": "exact" if exact else ("drawn" if explicit else "scaled"),
            "scale": scale, "paid": R, "top1_rank": R1, "tier_ranks": tier_ranks,
            "seconds": round(time.perf_counter() - t0, 2), "label": LABEL}
    return FieldEval(cash_pts, top1_pts, tl_pts, res, contest, info, per_sim=keep)


def dupe_band(d: float) -> str:
    """§5.5.6: Low < 0.3, Med 0.3-2, High > 2."""
    return "Low" if d < 0.3 else ("Med" if d <= 2 else "High")


# ============================================================================== store / runner

def store_ownership(conn: sqlite3.Connection, run_id: str, op: OwnershipProxy, *,
                    commit: bool = True) -> int:
    """Replace the run's `dfs_ownership_proxy` rows (player_id = DK player id, own in percent,
    band Low/Med/High, version incl. archetype)."""
    conn.execute("DELETE FROM dfs_ownership_proxy WHERE run_id=?", (run_id,))
    rows = [(run_id, pid, round(float(o) * 100, 2), band(o), op.version)
            for pid, o in zip(op.pids, op.own) if o > 0]
    conn.executemany("INSERT INTO dfs_ownership_proxy (run_id, player_id, own, band, version) "
                     "VALUES (?,?,?,?,?)", rows)
    if commit:
        conn.commit()
    return len(rows)


def prepare(conn: sqlite3.Connection, slate_id, run_id: str | None = None, *,
            n_sims: int | None = None, seed: int = 0, as_of: str | None = None) -> dict:
    """Pool, inputs, rules and aligned sims of a slate run (shared by `score_run` and the CLI).
    `as_of`: the salary capture at or before it (re-scoring a past run as it was)."""
    from .. import store as SO
    pool, inp, info = slate_inputs(conn, slate_id, run_id, as_of=as_of)
    if pool.rules is None:
        raise LookupError(f"no rules for slate {slate_id}")
    sims = None
    if info.get("sim_path") and Path(info["sim_path"]).exists():
        sims = SO.load_sims(info["sim_path"])
    return {"pool": pool, "inputs": inp, "info": info, "sims": sims, "n_sims": n_sims,
            "seed": seed}


def field_for(ctx: dict, contest: Contest, *, n: int | None = None, mix=None,
              n_solves: int = N_SOLVES, seed: int = 0, k: int | None = None,
              own_unc: Mapping[str, float] | None = None) -> Field | FieldSet:
    """The contest's field proxy: one Field, or K fields (`build_fields`) when `k` (default
    `field_k()`, the COURTSIDE_FIELD_K switch) is on."""
    pool = ctx["pool"]
    op = ownership_proxy(ctx["inputs"], pool, archetype=contest.archetype)
    K = field_k() if k is None else int(k)
    if K:
        return build_fields(pool, ctx["inputs"], contest, k=K, own_unc=own_unc, n=n, mix=mix,
                            ownership=op, n_solves=n_solves, seed=seed)
    return build_field(pool, ctx["inputs"], contest, n=n, mix=mix, ownership=op,
                       n_solves=n_solves, seed=seed)


def run_own_uncertainty(conn: sqlite3.Connection, run_id: str | None, *, slate_id=None,
                        now: datetime | None = None) -> dict[str, float]:
    """{DK id: ownership-uncertainty multiplier} of a run (late_news_tag; {} = 1.0 for everybody,
    also on any problem): the stored run's tag; a run not saved yet (the runner scores the field
    before it writes the run row) is tagged as of `now` (default: the clock) when `slate_id` is
    given."""
    if not run_id:
        return {}
    try:
        from . import late_news_tag as LNT
        saved = conn.execute("SELECT 1 FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
        if saved is None and slate_id is not None:
            res = LNT.compute_safe(conn, slate_id, run_id, now or datetime.now(timezone.utc))
        else:
            res = LNT.run_flags(conn, run_id)
        return LNT.own_uncertainty(res)
    except Exception:  # noqa: BLE001
        log.exception("own_uncertainty unavailable for run %s", run_id)
        return {}


def score_run(conn: sqlite3.Connection, run_id: str, *, slate_id=None,
              cash_contest: str | None = None, gpp_contest: str | None = None,
              n_field: int | None = 10_000, n_sims: int | None = None, mix=None,
              n_solves: int = N_SOLVES, seed: int = 0, write: bool = True,
              commit: bool = True, modes: Sequence[str] = ("cash", "topn"),
              truth: bool = True, as_of: str | None = None,
              own_unc: Mapping[str, float] | None = None) -> dict:
    """Ownership proxy + field proxies + lineup metrics for one projection run (runner hook).

    Writes `dfs_ownership_proxy` (large-GPP archetype, the slate-level "popularity (proxy)") and,
    on the run's `cash` and `topn` rows of `dfs_lineups`:
      p_cash, e_payout  vs the cash reference contest (default: the slate's largest captured
                        double-up; e_payout in dollars per entry, ties split);
      p_top1, dupes     vs the GPP reference contest (default: the slate's largest captured GPP).
    Returns the basis (contest ids, field summaries, cash-line quantiles, timings), suitable for
    `dfs_runs.detail["field"]`. Every number is against a modelled field (label "proxy").
    `commit=False` leaves the transaction to the caller (the runner commits with its run row).
    `truth=True` scores on the calibrated truth of the contest builder (`portfolio.truth_sims`:
    our sims shifted toward the public number and widened by model error), not on our
    projection as if it were exact (which put good cash lineups at ~80-90 %).
    A run without stored sims (the FPPG fallback) still gets the ownership proxy.
    COURTSIDE_FIELD_K on (`field_k()`): K fields per contest (D5, `build_fields`) with the
    late-news `own_unc` multipliers (default: the stored run's tag, `run_own_uncertainty`); the
    stored numbers are the means over the fields and `per_lineup` adds their range."""
    t0 = time.perf_counter()
    K = field_k()
    sid = slate_id
    if sid is None:
        r = conn.execute("SELECT slate_id FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone()
        if r is None:
            raise LookupError(f"unknown run {run_id}")
        sid = r[0]
    ctx = prepare(conn, sid, run_id, n_sims=n_sims, seed=seed,
                  **({"as_of": as_of} if as_of else {}))
    pool = ctx["pool"]
    own_gpp = ownership_proxy(ctx["inputs"], pool, archetype="large_gpp")
    if write:
        store_ownership(conn, run_id, own_gpp, commit=commit)
    out = {"status": "ok", "label": LABEL, "run_id": run_id, "version": own_gpp.version,
           "field_version": field_version(own_gpp, k=bool(K))}
    if ctx["sims"] is None:
        return {**out, "status": "no_sims", "lineups": 0,
                "seconds": round(time.perf_counter() - t0, 2)}
    dc, dg = default_contests(conn, sid)
    cc = load_contest(conn, cash_contest) if cash_contest else dc
    gc = load_contest(conn, gpp_contest) if gpp_contest else dg
    q = ",".join("?" * len(modes))
    rows = [dict(x) for x in conn.execute(
        f"SELECT id, mode, rank, slots_json FROM dfs_lineups WHERE run_id=? AND mode IN ({q}) "
        "ORDER BY mode, rank", (run_id, *modes))]
    out["lineups"] = len(rows)
    evals = {}
    if K and own_unc is None:
        own_unc = run_own_uncertainty(conn, run_id, slate_id=sid)
    for tag, contest in (("cash", cc), ("gpp", gc)):
        f = field_for(ctx, contest, n=n_field, mix=mix, n_solves=n_solves, seed=seed,
                      **({"own_unc": own_unc} if K else {}))
        fp, ainfo = align_sims(ctx["sims"], f.pids, n_sims, seed)
        tinfo = None
        if truth:
            from . import portfolio as PF
            fp, tinfo = PF.truth_sims(fp, f.pids, ctx["inputs"], PF.TRUTH_BETA, PF.TRUTH_TAU,
                                      np.random.default_rng(seed + 7))
        ev = evaluate(f, fp, contest, [x["slots_json"] for x in rows], seed=seed, tiers=False)
        evals[tag] = ev
        out[tag] = {"contest_id": contest.contest_id, "name": contest.name,
                    "field_size": contest.field_size, "entry_fee": contest.entry_fee,
                    "archetype": contest.archetype, "source": contest.source,
                    "field": f.summary(), "cash_line": ev.cash_line_summary(),
                    "n_sims": ainfo["n_sims"], "missing_sim_players": len(ainfo["missing"]),
                    "seconds_eval": ev.info["seconds"], "truth": tinfo}
    if write and rows:
        for i, x in enumerate(rows):
            c, g = evals["cash"].lineups[i], evals["gpp"].lineups[i]
            conn.execute("UPDATE dfs_lineups SET p_cash=?, e_payout=?, p_top1=?, dupes=? "
                         "WHERE id=?", (round(c["p_cash"], 4), round(c["e_payout"], 4),
                                        round(g["p_top1"], 4), round(g["dupes"], 3), x["id"]))
        if commit:
            conn.commit()
    out["per_lineup"] = [{"mode": x["mode"], "rank": x["rank"],
                          "p_cash": evals["cash"].lineups[i]["p_cash"],
                          "e_payout": evals["cash"].lineups[i]["e_payout"],
                          "p_top1": evals["gpp"].lineups[i]["p_top1"],
                          "gpp_e_payout": evals["gpp"].lineups[i]["e_payout"],
                          "dupes": evals["gpp"].lineups[i]["dupes"]}
                         for i, x in enumerate(rows)]
    if K:                               # across-field spread (display only)
        for i, d in enumerate(out["per_lineup"]):
            c, g = evals["cash"].lineups[i], evals["gpp"].lineups[i]
            d.update({"p_cash_lo": c["p_cash_lo"], "p_cash_hi": c["p_cash_hi"],
                      "p_top1_lo": g["p_top1_lo"], "p_top1_hi": g["p_top1_hi"]})
        out["field_k"] = {"k": K, "own_unc": {str(a): float(b) for a, b in
                                              (own_unc or {}).items() if float(b) != 1.0}}
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def score_run_safe(conn: sqlite3.Connection, run_id: str, **kw) -> dict:
    """`score_run` that never raises (a proxy statistic must not fail a lock run)."""
    try:
        return score_run(conn, run_id, **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("field proxy scoring failed for run %s", run_id)
        return {"status": f"error: {type(e).__name__}: {e}", "run_id": run_id, "label": LABEL}


# ============================================================================== history (RotoGuru)

def hist_inputs(conn: sqlite3.Connection, pool: SlatePool, min_games: int = 3) -> pd.DataFrame:
    """Ownership inputs for a RotoGuru date: the public projection is the season-to-date mean DK
    points (an FPPG analogue, games with minutes > 0, >= `min_games`), starters from RotoGuru's
    flag (known by lock). RotoGuru has no status, so availability is *known* (the T2.1 protocol,
    optimistic): a player who did not play that day is marked OUT, as if ruled out by lock.
    No team totals or our projection."""
    from .rules import season_of
    day = pool.captured_at
    d = date.fromisoformat(day)
    season = season_of(d)
    start = f"{season[:4]}-08-01"
    df = pd.read_sql_query(
        "SELECT rg_id, AVG(dk_pts) AS fppg, COUNT(*) AS n FROM dfs_hist_salaries WHERE "
        "game_date>=? AND game_date<? AND minutes>0 AND dk_pts IS NOT NULL GROUP BY rg_id",
        conn, params=[start, day])
    f = {str(r.rg_id): float(r.fppg) for r in df.itertuples() if r.n >= min_games}
    inp = pool_inputs(pool, fppg=f)
    played = {str(r[0]) for r in conn.execute(
        "SELECT rg_id FROM dfs_hist_salaries WHERE game_date=? AND minutes>0", (day,))}
    inp["status"] = np.where(inp["pid"].isin(played), inp["status"], "OUT")
    return inp


def backtest_cash_lines(conn: sqlite3.Connection, dates: Sequence[str], *, n: int = 3000,
                        paid_frac: float = 0.444, field_size: int = 1000, mix=None,
                        n_solves: int = 100, seed: int = 0, log_fn=None) -> pd.DataFrame:
    """Implied double-up cash lines on RotoGuru history: a cash-archetype field proxy built from
    public information (salary, season-to-date DK average, starters) and scored with the ACTUAL
    DK points (one 'sim' = what happened). Also the hindsight-optimal score for scale."""
    from . import optimizer as O
    rules = O.classic_rules()
    rows = []
    for day in dates:
        t0 = time.perf_counter()
        try:
            pool = O.hist_pool(conn, day, rules)
        except LookupError:
            continue
        inp = hist_inputs(conn, pool)
        if inp["fppg"].notna().sum() < 40:
            continue
        actual = O.hist_projections(conn, pool, "actual")
        contest = preset_contest("double_up", field_size)
        contest.payouts[0]["max_pos"] = int(round(field_size * paid_frac))
        op = ownership_proxy(inp, pool, rules, archetype="cash", version=HANDSET_VERSION,
                             params=DEFAULT_PARAMS)
        f = build_field(pool, inp, contest, rules=rules, n=n, mix=mix, ownership=op,
                        n_solves=n_solves, seed=seed)
        fp = np.array([[actual[p].mean if p in actual else 0.0 for p in f.pids]], np.float32)
        ev = evaluate(f, fp, contest, [], seed=seed, tiers=False)
        best = O.optimize(pool, actual, rules)
        games = len({_game(p) for p in pool.players})
        rows.append({"date": day, "games": games, "players": len(pool.players),
                     "cash_line": float(ev.cash_line[0]), "top1_line": float(ev.top1_line[0]),
                     "optimal": best.projected if best else None,
                     "field_mean": float((np.rint(fp * UNIT)[0] @ _field_matrix(f).T
                                          ).mean() / (UNIT * WUNIT)),
                     "share_ge_98": f.salary_use()["share_ge_98"],
                     "seconds": round(time.perf_counter() - t0, 2)})
        if log_fn:
            log_fn(f"{day}: {rows[-1]}")
    return pd.DataFrame(rows)


def _field_matrix(f: Field) -> np.ndarray:
    A = np.zeros((len(f.lineups), len(f.pids)), np.float32)
    for j, s in enumerate(f.slots):
        np.add.at(A, (np.arange(len(f.lineups)), f.lineups[:, j]), 3 if s == "CPT" else WUNIT)
    return np.repeat(A, f.counts, axis=0)


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("field", help="ownership proxy + field proxy for a slate (PROXY: not real "
                                     "ownership or results)")
    p.add_argument("--slate", required=True, help="DK draft group id")
    p.add_argument("--contest", default=None, help="contest id (default: the slate's largest "
                                                   "double-up)")
    p.add_argument("--run", default=None, help="projection run id (default: newest with sims)")
    p.add_argument("--n", type=int, default=5000, help="sampled field lineups (default 5000)")
    p.add_argument("--sims", type=int, default=None, help="subsample sims (default: all)")
    p.add_argument("--mix", default=None, help="optimizer-like share or soft|medium|sharp")
    p.add_argument("--solves", type=int, default=N_SOLVES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--write", action="store_true",
                   help="store dfs_ownership_proxy and lineup metrics for the run (score_run)")
    q = sub.add_parser("field-calib", help="what uploaded DK standings say about the ownership "
                                           "and field proxy (partial calibration report; "
                                           "--fit refits the ownership weights from 10 slates)")
    q.add_argument("--fit", action="store_true", help="also run fit_from_uploads (needs "
                                                      f"{MIN_FIT_SLATES} slates)")
    q.add_argument("--save", action="store_true", help="with --fit: save the fitted weights")
    q.add_argument("--json", action="store_true")
    return {"field": _field_cmd, "field-calib": _calib_cmd}


def _calib_cmd(args) -> int:
    from ... import db
    conn = db.connect()
    rep = calibration_report(conn)
    if args.fit:
        rep["fit"] = fit_from_uploads(conn, save=args.save)
    if args.json:
        print(json.dumps(rep, indent=1, default=float))
        return 0
    print(f"FIELD CALIBRATION (proxy {rep['version']}): {rep['n_slates']} slate(s) of uploaded "
          f"standings; the ownership refit needs {rep['needed']}")
    for r in rep["contests"]:
        px = r.get("proxy") or {}
        print(f"  {r['contest_id']} {str(r['name'] or '')[:34]:<34} {r['archetype'] or '?':<9} "
              f"{r['entries']:>6,} entries  unique {r['unique_share']:.0%}  in dupes "
              f"{r['dupe_entry_share']:.0%}  top lineup {r['max_dupe_share']:.1%}  top-10 own "
              f"{r['own_top10'] * 100:.0f}%"
              + (f" (proxy {px['own_top10'] * 100:.0f}%, MAE {px['mae_pp']:.1f} pp, top-10 MAE "
                 f"{px['top10_mae_pp']:.1f} pp)" if px else ""))
    for sid, v in rep["slates"].items():
        print(f"  slate {sid}: cash ~ GPP ownership ** k, k = "
              + ", ".join(f"{k:.2f}" for k in v["cash_vs_gpp_exponent"])
              + f"; proxy conc ratio cash / large GPP = {v['proxy_conc_ratio']:.2f}")
    if args.fit:
        f = rep["fit"]
        print(f"  ownership refit: {f['status']} ({f.get('n_slates')} of {f.get('needed', '')} "
              "slates)" if f["status"] != "fitted" else
              f"  ownership refit: {f['version']} (in-sample MAE "
              f"{f['fitted_in_sample']['mae_pp']:.2f} pp vs hand-set {f['handset']['mae_pp']:.2f})")
    return 0


def _field_cmd(args) -> int:
    from ... import db
    conn = db.connect()
    t0 = time.perf_counter()
    ctx = prepare(conn, args.slate, args.run, n_sims=args.sims, seed=args.seed)
    pool = ctx["pool"]
    if args.contest:
        contest = load_contest(conn, args.contest)
        if contest is None:
            print(f"unknown contest {args.contest}")
            return 1
    else:
        contest = default_contests(conn, args.slate)[0]
    mix = args.mix
    if mix is not None and mix not in STRENGTHS:
        mix = float(mix)
    op = ownership_proxy(ctx["inputs"], pool, archetype=contest.archetype)
    cal = "calibrated on real uploads" if calibrated() else \
        "hand-set, NOT calibrated (no real ownership yet); bands only"
    print(f"OWNERSHIP PROXY  slate {args.slate}  {op.version}  ({cal})")
    print(f"  archetype {contest.archetype}; Σ own = {op.own.sum() * 100:.1f}%; expected spend "
          f"${op.spend:,.0f}; run {ctx['info']['run_id']}")
    t = op.table(args.top)
    cpt = "own_cpt" in t
    print(f"  {'player':<26}{'team':<5}{'pos':<7}{'sal':>6}{'pub':>7}{'val':>6}  "
          f"{'own%':>6}  band{'  cpt%' if cpt else ''}")
    for r in t.itertuples():
        flag = " repl" if r.replacement else ""
        st = f" {r.status}" if isinstance(r.status, str) and r.status else ""
        print(f"  {r.name[:25]:<26}{(r.team or ''):<5}{r.positions:<7}{r.salary:>6}"
              f"{r.pub_proj:>7.1f}{r.pub_value:>6.2f}  {r.own * 100:>5.1f}%  {r.band:<4}"
              f"{f'  {r.own_cpt * 100:4.1f}' if cpt else ''}{st}{flag}")
    f = build_field(pool, ctx["inputs"], contest, n=args.n, mix=mix, ownership=op,
                    n_solves=args.solves, seed=args.seed)
    s = f.summary()
    su = s["salary_use"]
    print(f"\nFIELD PROXY  {contest.name} ({contest.source}): field {contest.field_size:,}, "
          f"fee ${contest.entry_fee:g}, paid {contest.paid}")
    print(f"  {s['archetype']}; optimizer-like {s['mix_optimizer_like']:.0%} "
          f"({s['n_solves']} solves, {s['opt_unique']} unique) / ownership-sampled "
          f"{1 - s['mix_optimizer_like']:.0%}; {s['sampled']:,} sampled, {s['unique']:,} unique")
    print(f"  salary use: >=98% {su['share_ge_98']:.0%}, >=99% {su['share_ge_99']:.0%}, mean "
          f"left ${su['mean_left']:,.0f}; field vs proxy ownership MAE "
          f"{s['own_target_mae_pp']:.1f} pp; built in {s['seconds']:.1f}s")
    if ctx["sims"] is None:
        print("\n(no stored sims for this slate run: cash line not available)")
        return 0
    rows = [dict(x) for x in conn.execute(
        "SELECT mode, rank, slots_json FROM dfs_lineups WHERE run_id=? AND mode IN ('cash','topn')"
        " ORDER BY mode, rank", (ctx["info"]["run_id"],))]
    fp, ainfo = align_sims(ctx["sims"], f.pids, args.sims, args.seed)
    ev = evaluate(f, fp, contest, [r["slots_json"] for r in rows], seed=args.seed)
    c = ev.cash_line_summary()
    print(f"\nCASH LINE (proxy, {ainfo['n_sims']:,} sims; rank {contest.paid} of "
          f"{contest.field_size:,}): p10 {c['cash_p10']:.1f}  p25 {c['cash_p25']:.1f}  "
          f"p50 {c['cash_p50']:.1f}  p75 {c['cash_p75']:.1f}  p90 {c['cash_p90']:.1f}")
    print(f"TOP-1% LINE (proxy): p10 {c['top1_p10']:.1f}  p50 {c['top1_p50']:.1f}  "
          f"p90 {c['top1_p90']:.1f}   (scored in {ev.info['seconds']:.1f}s)")
    if rows:
        print("\nRUN LINEUPS vs this field (simulated against a modelled field, not real "
              "results)")
        for r, m in zip(rows, ev.lineups):
            print(f"  {r['mode']:<5}#{r['rank']:<3} mean {m['mean']:6.1f}  P(cash) "
                  f"{m['p_cash']:.1%}  P(top1%) {m['p_top1']:.1%}  E[payout] "
                  f"${m['e_payout']:.2f} ±{m['e_payout_se']:.2f}  dupes {m['dupes']:.2f} "
                  f"({m['dupe_band']})")
    if args.write and ctx["info"]["run_id"]:
        res = score_run(conn, ctx["info"]["run_id"], cash_contest=args.contest
                        if contest.archetype == "cash" else None,
                        gpp_contest=args.contest if contest.archetype != "cash" else None,
                        n_field=args.n, n_sims=args.sims, mix=mix, n_solves=args.solves,
                        seed=args.seed)
        print(f"\nwrote run {res['run_id']}: {res['lineups']} lineups ({res['seconds']}s)")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0
