"""Usage v1: per-player with/without factors (docs/PLAYERS.md §4.3 Phase 2, T2.3).

When regular teammates are absent, player i's per-minute rate of DK stat s is multiplied by

    phi_i,s = exp( clip( sum_{j in A} l_ij,s , -CAP, +CAP ) )

over the set A of absent *regulars* j (appeared in >= 7 of the team's last 10 games or in one of
the last 3: the pipeline's `_stars` rule, so a long absence that the rate state has already
absorbed stops triggering), with one additive log term per absent teammate:

    l_ij,s = beta_s[cell(i, j)] * share_j,s                       (1) tier / position-pair prior
           + lam_s * log(1 + w_st (R_st - 1)),  w_st = n_off / (n_off + K_st,s)   (2) stints, 2019+
           + log(1 + w_bx (R_bx - 1)),          w_bx = n_bx  / (n_bx  + K_bx,s)   (3) box with/without

(1) share_j,s = j's share of the team's production of s among its regulars (pre-game rate x mpg);
    cell(i, j) = i's role (star = top 2 of the team's regulars by DK/min x mpg, starter >= 24 mpg,
    rotation >= 15, bench) x (same position group as j or not): 8 cells per stat.
(2) R_st = i's per-possession rate with j OFF the floor / with j ON, from the play-by-play
    stints of earlier games (both played, garbage time excluded, last season x RHO), each side
    shrunk by one pseudo-count to i's pooled rate; n_off = i's minutes with j off. This is the
    spec's phi_emp n_off / (n_off + K_phi) on the multiplicative scale; lam_s (fit) maps the
    in-game on/off contrast to the whole-game absence effect.
(3) R_bx = sum(actual) / sum(model mean with (1)+(2)) over i's earlier games in which j was an
    absent regular (all seasons, last season x RHO), n_bx = i's minutes in those games.

**Combination.** Additive on the log scale: vacated usage adds up (the prior is linear in the
absentee's share, so two absent stars give about twice the log lift of one), each pair's own
evidence is shrunk separately, and the total is clipped to +-CAP (0.4 = x0.67..x1.49), which
bounds any multi-absence.

**Consistency with minutes.** Every fit is on the per-minute rate *given the actual minutes*:
the expected count is the T1.1 EB rate x actual minutes x f_stat(min/mpg) (state's minutes
table), so the minutes engine's promotion is never re-learned here; phi only carries the
per-minute change. beta / lam are a negative-binomial GLM (state dispersion per tier) on
training player-games with >= 1 absent regular; K_st and K_bx per stat are chosen on the same
rows by the NB log-likelihood (every row's pair evidence uses only earlier games, so this is a
walk-forward choice). Everything is fit on seasons < S.

Public API
  UsageParams (JSON) / default_params()
  build_stint_tallies(conn, seasons, path=None, workers=1) -> {"pairs", "totals"} (cached)
  UsageData(conn, season, sp, ...)       per-season frames (rows, absences, pair evidence)
  fit(conn, seasons, sp=None, ...) -> UsageParams
  pair_matrix(params, team_players, evidence) -> {stat: (n, n) log-factor matrix L[j, i]}
  factor_matrix(L, absent) -> {stat: (S, n) multipliers}   (absent (S, n) bool; the sim)
  factors(params, team_players, absent, evidence=None) -> {player_id: {stat: multiplier}}
  UsageData.evidence(day, player_ids) -> pair evidence for the sim (stint + box layers)
  load_usage(conn, season, sp) -> (params fit on seasons < S, UsageData ready for evidence)
  sim_means(players, sp, x, played, F) -> raw_pts, raw_oth, stats (drop-in for the bump grids)
  evaluate(conn, season, ...) -> acceptance (NB LL vs the tier bump), residual check
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import state as ST

DK_STATS = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")
DK_W = dict(ST.DK_WEIGHTS)
OTHER_STATS = tuple(s for s in DK_STATS if s != "pts")
ROLES = ("star", "starter", "rotation", "bench")
N_CELL = 2 * len(ROLES)               # role x (same position group as the absentee)
STARTER_MPG, ROTATION_MPG = 24.0, 15.0
MIN_ABSENT_MPG = 10.0                 # absent regulars below this are ignored
CAP = 0.4                             # |sum of log terms| <= CAP
RHO = 0.6                             # last season's pair evidence weight (per season back)
K_GRID = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0, float("inf"))
RIDGE = 1.0                           # L2 on beta (share units), keeps empty cells at 0
STINT_FIRST = "2019-20"               # play-by-play stints start here
SAT_MPG = 25.0                        # "teammate sat": a top-2 star or a >= 25 mpg regular out


# ============================================================================== params

@dataclass
class UsageParams:
    version: int = 1
    fit_seasons: list = field(default_factory=list)
    beta: dict = field(default_factory=dict)      # stat -> [N_CELL] log lift per unit share
    lam: dict = field(default_factory=dict)       # stat -> stint-layer coefficient
    k_stint: dict = field(default_factory=dict)   # stat -> K (minutes), inf = layer off
    k_box: dict = field(default_factory=dict)     # stat -> K (minutes), inf = layer off
    alpha: dict = field(default_factory=dict)     # stat -> [4] log lift per absentee by i's role
    ridge: dict = field(default_factory=dict)     # stat -> chosen L2 (diagnostic)
    cap: float = CAP
    rho: float = RHO
    min_absent_mpg: float = MIN_ABSENT_MPG
    diagnostics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("k_stint", "k_box"):
            d[k] = {s: (None if not np.isfinite(v) else float(v)) for s, v in d[k].items()}
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "UsageParams":
        d = json.loads(s)
        for k in ("k_stint", "k_box"):
            d[k] = {st: (float("inf") if v is None else float(v)) for st, v in d[k].items()}
        return cls(**d)

    def k_of(self, layer: str, stat: str) -> float:
        return float((self.k_stint if layer == "stint" else self.k_box).get(stat, float("inf")))


def default_params() -> UsageParams:
    """All-zero params: every factor is exactly 1."""
    return UsageParams(beta={s: [0.0] * N_CELL for s in DK_STATS},
                       alpha={s: [0.0] * len(ROLES) for s in DK_STATS},
                       lam={s: 0.0 for s in DK_STATS},
                       k_stint={s: float("inf") for s in DK_STATS},
                       k_box={s: float("inf") for s in DK_STATS})


def cell_of(role: np.ndarray, same_grp: np.ndarray) -> np.ndarray:
    """Cell index: role (0 star .. 3 bench) x same position group."""
    return np.asarray(role, int) * 2 + np.asarray(same_grp, bool).astype(int)


def role_of(mpg: np.ndarray, is_star: np.ndarray) -> np.ndarray:
    mpg = np.asarray(mpg, float)
    return np.where(np.asarray(is_star, bool), 0,
                    np.where(mpg >= STARTER_MPG, 1, np.where(mpg >= ROTATION_MPG, 2, 3)))


def shrink_term(ratio, n, k) -> np.ndarray:
    """log(1 + w (ratio - 1)), w = n / (n + k); 0 when k is inf or n is 0."""
    ratio = np.asarray(ratio, float)
    n = np.asarray(n, float)
    if not np.isfinite(k):
        return np.zeros(np.broadcast(ratio, n).shape)
    w = n / (n + k)
    return np.log1p(w * (np.nan_to_num(ratio, nan=1.0) - 1.0))


# ============================================================================== stint tallies

def _game_pair_tally(box: dict, pbp: dict):
    """Per (i, j) teammates who both played: i's seconds / possessions / DK stats with j OFF the
    floor (non-garbage stints), and per player the same totals over all non-garbage stints.
    Returns (pairs list, totals list) or None when the game does not reconstruct."""
    from courtside.analytics import stints_store as SS
    from courtside.analytics.pbp import normalize
    from courtside.analytics.stints import box_players, build_rotation
    events = normalize(SS.repair_legacy_pbp(pbp, box), box)
    rot = build_rotation(events, box)
    stints = rot.stints
    if not stints or any(len(s.home) != 5 or len(s.away) != 5 for s in stints):
        return None
    players = box_players(box)
    starters = [pid for pid, p in players.items() if p["starter"]]
    garb = SS.garbage_seconds(events, stints, starters)
    firsts = [s.first_idx for s in stints]
    K = len(stints)
    si = {s: k for k, s in enumerate(DK_STATS)}
    y: dict[int, np.ndarray] = {}               # player -> (K, 7) stats per stint

    def add(pid, idx, stat, v=1.0):
        if not pid:
            return
        k = SS._place(stints, firsts, idx)
        if k is None:
            return
        a = y.get(pid)
        if a is None:
            a = y[pid] = np.zeros((K, len(DK_STATS)))
        a[k, si[stat]] += v

    for e in events:
        if (e.is_fga or e.is_ft) and e.made and e.player_id:
            add(e.player_id, e.idx, "pts", float(e.points or 0))
            if e.is_fga and e.is_three:
                add(e.player_id, e.idx, "fg3m")
        if e.is_fga and e.made and e.assist_id:
            add(e.assist_id, e.idx, "ast")
        if e.is_rebound and e.player_id and not e.team_rebound:
            add(e.player_id, e.idx, "reb")
        if e.is_turnover:
            if e.player_id:
                add(e.player_id, e.idx, "tov")
            if e.steal_id:
                add(e.steal_id, e.idx, "stl")
        if e.is_fga and e.block_id:
            add(e.block_id, e.idx, "blk")
    pairs, totals = [], []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        keep = [k for k, s in enumerate(stints) if garb[k] <= 0 and s.duration > 0]
        if not keep:
            continue
        pids = sorted({p for k in keep for p in stints[k].lineup(side)})
        pidx = {p: n for n, p in enumerate(pids)}
        P = len(pids)
        on = np.zeros((len(keep), P))
        dur = np.array([stints[k].duration for k in keep], float)
        poss = np.array([(stints[k].poss(side) + stints[k].poss(opp)) / 2.0 for k in keep])
        Y = np.zeros((len(keep), P, len(DK_STATS)))
        for r, k in enumerate(keep):
            for p in stints[k].lineup(side):
                on[r, pidx[p]] = 1.0
                a = y.get(p)
                if a is not None:
                    Y[r, pidx[p]] = a[k]
        team = (players.get(pids[0]) or {}).get("team")
        sec_t = on.T @ dur
        pos_t = on.T @ poss
        y_t = np.einsum("ki,kis->is", on, Y)
        both = on[:, :, None] * on[:, None, :]                # (k, i, j)
        sec_on = np.einsum("kij,k->ij", both, dur)
        pos_on = np.einsum("kij,k->ij", both, poss)
        y_on = np.einsum("kij,kis->ijs", both, Y)
        for a_ in range(P):
            totals.append((team, pids[a_], sec_t[a_], pos_t[a_], *y_t[a_]))
            for b_ in range(P):
                if a_ == b_:
                    continue
                pairs.append((team, pids[a_], pids[b_], sec_t[a_] - sec_on[a_, b_],
                              pos_t[a_] - pos_on[a_, b_], *(y_t[a_] - y_on[a_, b_])))
    return pairs, totals


def _tally_work(task):
    gid, pbp_path, box_path = task
    from courtside import archive
    try:
        r = _game_pair_tally(archive.load_raw(box_path), archive.load_raw(pbp_path))
    except Exception:   # noqa: BLE001 - one bad archive must not stop the build
        r = None
    return gid, r


def stint_cache_path() -> Path:
    from .. import config
    return config.data_dir() / "proj" / "usage_stint_tallies.pkl"


def build_stint_tallies(conn: sqlite3.Connection, seasons: Iterable[str] | None = None, *,
                        path: Path | None = None, workers: int = 1,
                        log: Callable[[str], None] | None = None) -> dict:
    """Pair on/off tallies for every game in the `stints` table (accepted reconstructions) of
    `seasons`, from the raw cdn archive. Cached (incrementally) at `path`. Returns
    {"pairs": DataFrame(game_id, team, i, j, sec_off, poss_off, <stat>_off...),
     "totals": DataFrame(game_id, team, i, sec, poss, <stat>...)}."""
    from courtside.analytics import stints_store as SS
    say = log or (lambda m: None)
    path = path or stint_cache_path()
    have = {"pairs": [], "totals": []}
    done: set = set()
    if path.exists():
        saved = pd.read_pickle(path)
        have = {"pairs": [saved["pairs"]], "totals": [saved["totals"]]}
        done = set(saved["totals"]["game_id"].unique())
    ok = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM stints")}
    todo = [(gid, p, b) for gid, _d, p, b in SS.list_archived_games(seasons)
            if gid in ok and gid not in done]
    say(f"stint tallies: {len(done)} cached, {len(todo)} to build")
    if todo:
        t0 = time.time()
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_tally_work, todo, chunksize=32))
        else:
            res = [_tally_work(t) for t in todo]
        pc = ["team", "i", "j", "sec_off", "poss_off"] + [f"{s}_off" for s in DK_STATS]
        tc = ["team", "i", "sec", "poss"] + list(DK_STATS)
        pr, tr = [], []
        for gid, r in res:
            if r is None:
                continue
            if r[0]:
                pr.append(pd.DataFrame(r[0], columns=pc).assign(game_id=gid))
            if r[1]:
                tr.append(pd.DataFrame(r[1], columns=tc).assign(game_id=gid))
        if pr:
            have["pairs"].append(pd.concat(pr, ignore_index=True))
            have["totals"].append(pd.concat(tr, ignore_index=True))
        say(f"stint tallies: built {len(res)} games in {time.time() - t0:.0f}s")
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in have.items()}
    for k in out:
        if len(out[k]):
            f = out[k].select_dtypes("float64").columns
            out[k][f] = out[k][f].astype(np.float32)
            for c in ("i", "j"):
                if c in out[k]:
                    out[k][c] = out[k][c].astype(np.int64)
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        pd.to_pickle(out, tmp)
        tmp.replace(path)
    if seasons is not None:
        want = {s[2:4] for s in seasons}
        for k in out:
            if len(out[k]):
                out[k] = out[k][out[k]["game_id"].str[3:5].isin(want)].reset_index(drop=True)
    return out


# ============================================================================== frames

def _rel_season(sidx) -> np.ndarray:
    return np.asarray(sidx, float) - 2000.0


def _regulars(wf: pd.DataFrame) -> pd.DataFrame:
    """Regulars of every regular-season team-game: (game_id, team, day, sidx, player_id,
    played). A regular appeared in >= 7 of the team's previous 10 games of the season or in one
    of the previous 3 (the pipeline's `_stars` rule). Traded players are dropped from their old
    team (their next appearance is for another team)."""
    w = wf[["game_id", "team", "player_id", "day", "sidx"]]
    tg = w[["team", "game_id", "day", "sidx"]].drop_duplicates().sort_values(
        ["team", "sidx", "day", "game_id"]).reset_index(drop=True)
    tg["k"] = tg.groupby(["team", "sidx"]).cumcount()
    w = w.merge(tg[["team", "game_id", "k"]], on=["team", "game_id"])
    out = []
    for (team, sidx), d in w.groupby(["team", "sidx"], sort=False):
        g = tg[(tg["team"] == team) & (tg["sidx"] == sidx)]
        G = len(g)
        pids = np.sort(d["player_id"].unique())
        pi = np.searchsorted(pids, d["player_id"].to_numpy())
        played = np.zeros((G, len(pids)), bool)
        played[d["k"].to_numpy(), pi] = True
        c = np.vstack([np.zeros((1, len(pids))), np.cumsum(played, 0)])   # c[k] = sum before k
        app10 = c[np.arange(G)] - c[np.maximum(np.arange(G) - 10, 0)]
        app3 = c[np.arange(G)] - c[np.maximum(np.arange(G) - 3, 0)]
        reg = (app10 >= 7) | (app3 >= 1)
        kk, pp = np.nonzero(reg | played)
        out.append(pd.DataFrame({"game_id": g["game_id"].to_numpy()[kk], "team": team,
                                 "day": g["day"].to_numpy()[kk], "sidx": sidx,
                                 "player_id": pids[pp], "played": played[kk, pp],
                                 "regular": reg[kk, pp]}))
    r = pd.concat(out, ignore_index=True)
    # drop absent regulars whose next appearance (any team) is for another team
    ab = r[~r["played"]].copy()
    app = w[["player_id", "day", "team"]].rename(columns={"team": "next_team", "day": "nday"})
    app = app.sort_values("nday")
    ab = ab.sort_values("day")
    ab = pd.merge_asof(ab, app, left_on="day", right_on="nday", by="player_id",
                       direction="forward", allow_exact_matches=False)
    traded = ab["next_team"].notna() & (ab["next_team"] != ab["team"])
    drop = ab.loc[traded, ["game_id", "player_id"]]
    r = r.merge(drop.assign(_d=1), on=["game_id", "player_id"], how="left")
    r = r[r["_d"].isna()].drop(columns="_d")
    return r.reset_index(drop=True)


class UsageData:
    """Walk-forward frames for seasons [first, last]: every regular-season player-game row
    with its pre-game EB expectation given actual minutes, the absent regulars of its
    team-game (pairs), and the causal pair evidence (stint layer; the box layer is added by
    `box_evidence` once the model mean is known).

    rows:  one per player-game (i played): ids, day, sidx, min, y_<s>, E_<s> (rate x min x
           f(min/mpg)), d_<s> (NB dispersion), mpg, star, role, grp, sat (teammate sat),
           bump role (tier-bump baseline: 0 none, 1 star, 2 rotation).
    pairs: one per (row, absent regular j): r (row index), i, j, cell, share_<s>, and the stint
           evidence ratio rs_<s> with its minutes n_st."""

    def __init__(self, conn: sqlite3.Connection, seasons: Sequence[str], sp: dict, *,
                 games: pd.DataFrame | None = None, pos=None, tallies: dict | None = None,
                 log: Callable[[str], None] | None = None):
        say = log or (lambda m: None)
        t0 = time.time()
        self.seasons = sorted(seasons)
        self.sp = sp
        if games is None:
            first = ST.season_label(ST.sidx_of(self.seasons[0]) - sp["history_seasons"])
            games = ST.load_games(conn, since_season=first, until_season=self.seasons[-1])
        pos = pos if pos is not None else ST.load_positions(conn)
        wf = pd.concat([ST.walk_forward(conn, s, sp, games=games, pos=pos)
                        for s in self.seasons], ignore_index=True)
        wf = wf[wf["season_type"] == "Regular Season"].reset_index(drop=True)
        wf["day"] = ST._to_day(wf["game_date"])
        wf["sidx"] = wf["season"].str[:4].astype(int)
        wf["star_score"] = wf["r_dk"] * wf["mpg"]
        self.wf = wf
        self.games = games
        say(f"usage data: walk-forward {len(wf)} rows in {time.time() - t0:.0f}s")
        self._build(say)
        self._stints(conn, tallies, say)
        say(f"usage data: {len(self.rows)} rows, {len(self.pairs)} absent pairs "
            f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- rows and absences
    def _build(self, say):
        wf, sp = self.wf, self.sp
        reg = _regulars(wf)
        # state of every regular at the team-game: own row when playing, else last state
        st_cols = ["mpg", "star_score", "grp"] + [f"r_{s}" for s in DK_STATS]
        cur = reg[reg["played"]].merge(wf[["game_id", "player_id"] + st_cols],
                                       on=["game_id", "player_id"], how="left")
        ab = reg[~reg["played"]].sort_values("day")
        last = wf[["player_id", "day"] + st_cols].sort_values("day")
        ab = pd.merge_asof(ab, last, on="day", by="player_id", allow_exact_matches=False)
        ab = ab[ab["mpg"].notna()]
        R = pd.concat([cur, ab], ignore_index=True)
        R = R[R["regular"]]
        # stars: top 2 regulars by DK/min x mpg (pre-game state)
        R["rk"] = R.groupby(["game_id", "team"])["star_score"].rank(ascending=False,
                                                                    method="first")
        R["star"] = R["rk"] <= 2
        for s in DK_STATS:
            R[f"tot_{s}"] = R[f"r_{s}"] * R["mpg"]
        tot = R.groupby(["game_id", "team"])[[f"tot_{s}" for s in DK_STATS]].sum()
        R = R.drop(columns=[f"tot_{s}" for s in DK_STATS])
        A = R[~R["played"] & (R["mpg"] >= MIN_ABSENT_MPG)].merge(
            tot.reset_index(), on=["game_id", "team"])
        for s in DK_STATS:
            A[f"share_{s}"] = (A[f"r_{s}"] * A["mpg"] / A[f"tot_{s}"].clip(lower=1e-9)).clip(0, 1)
        A["sat"] = A["star"] | (A["mpg"] >= SAT_MPG)
        # rows
        rows = wf[["game_id", "team", "player_id", "day", "sidx", "season", "min", "mpg",
                   "tier", "grp", "star_score"] + list(DK_STATS)].copy()
        mins = wf["min"].to_numpy(float)
        mpg = wf["mpg"].to_numpy(float)
        tier = wf["tier"].to_numpy(int)
        for s in DK_STATS:
            rows[f"E_{s}"] = (wf[f"r_{s}"].to_numpy(float) * mins
                              * ST.minutes_multiplier(sp, s, mins, mpg))
            rows[f"d_{s}"] = ST.phi_for(sp, s, tier)
        rows = rows.merge(R[["game_id", "player_id", "star"]], on=["game_id", "player_id"],
                          how="left")
        rows["star"] = rows["star"].fillna(False).astype(bool)
        rows["role"] = role_of(rows["mpg"], rows["star"])
        sat = A[A["sat"]][["game_id", "team"]].drop_duplicates().assign(sat=True)
        rows = rows.merge(sat, on=["game_id", "team"], how="left")
        rows["sat"] = rows["sat"].fillna(False).astype(bool)
        rows = rows.reset_index(drop=True)
        rows["r"] = np.arange(len(rows))
        P = rows[["r", "game_id", "team", "player_id", "day", "sidx", "role", "grp"]].merge(
            A[["game_id", "team", "player_id", "grp", "star", "mpg"]
              + [f"share_{s}" for s in DK_STATS]].rename(
                columns={"player_id": "j", "grp": "grp_j", "star": "star_j", "mpg": "mpg_j"}),
            on=["game_id", "team"])
        P = P.rename(columns={"player_id": "i"})
        P["cell"] = cell_of(P["role"], (P["grp"] == P["grp_j"]) & (P["grp"] < 3))
        self.regs = R[["game_id", "team", "day", "player_id", "played", "regular", "mpg", "star",
                       "grp"] + [f"r_{s}" for s in DK_STATS]].reset_index(drop=True)
        self.rows = rows
        self.pairs = P.sort_values(["i", "j", "day"]).reset_index(drop=True)
        self.absent = A
        # tier-bump baseline roles (pipeline rule via state.star_absence_flags)
        fl = ST.star_absence_flags(wf)
        rows = rows.merge(fl, on=["game_id", "team", "player_id"], how="left")
        role = np.where(rows["is_star"].fillna(False).astype(bool), 1,
                        np.where(rows["is_star"].notna() & (rows["mpg"] >= 15.0), 2, 0))
        self.rows = rows.drop(columns="is_star").assign(bump_role=role)

    # ---------------------------------------------------------------- stint layer
    def _stints(self, conn, tallies, say):
        P = self.pairs
        for s in DK_STATS:
            P[f"rs_{s}"] = 1.0
        P["n_st"] = 0.0
        if tallies is None:
            path = stint_cache_path()
            if not path.exists():
                say("usage data: no stint tallies cached; stint layer empty")
                return
            tallies = pd.read_pickle(path)
        pr, tt = tallies["pairs"], tallies["totals"]
        if not len(pr):
            return
        gd = self.games[["game_id", "day", "sidx"]].drop_duplicates("game_id")
        want = P[["i", "j"]].drop_duplicates()
        pg = pr.merge(want, on=["i", "j"]).merge(gd, on="game_id")
        pg = pg.merge(tt.drop(columns="team"), on=["game_id", "i"], suffixes=("", "_t"))
        disc = RHO ** (-_rel_season(pg["sidx"]))
        cols = {"n": pg["sec_off"] / 60.0, "po": pg["poss_off"], "pt": pg["poss"]}
        for s in DK_STATS:
            cols[f"yo_{s}"] = pg[f"{s}_off"]
            cols[f"yt_{s}"] = pg[s]
        C = pd.DataFrame({k: np.asarray(v, float) * disc for k, v in cols.items()})
        C[["i", "j", "day"]] = pg[["i", "j", "day"]].to_numpy()
        C = C.sort_values(["i", "j", "day"])
        val = [c for c in C.columns if c not in ("i", "j", "day")]
        C[val] = C.groupby(["i", "j"])[val].cumsum()
        self.stint_cum = C.sort_values("day").reset_index(drop=True)
        self._stint_vals = val
        q = P[["i", "j", "day", "sidx"]].reset_index().sort_values("day")
        q = pd.merge_asof(q, self.stint_cum, on="day", by=["i", "j"],
                          allow_exact_matches=False).sort_values("index")
        ok = q["n"].notna().to_numpy()
        f = np.where(ok, RHO ** _rel_season(q["sidx"]), 0.0)
        st = stint_ratios(q, f)
        for s in DK_STATS:
            P[f"rs_{s}"] = st[s]
        P["n_st"] = np.nan_to_num(q["n"].to_numpy(float) * f)
        say(f"usage data: stint evidence on {ok.mean():.0%} of absent pairs")

    # ---------------------------------------------------------------- box layer
    def box_evidence(self, mu: dict) -> None:
        """Pair box evidence from model means `mu` {stat: (len(rows),)} (prior + stint
        layers): for each absent pair row, sums over the pair's EARLIER rows of actual,
        model mean and minutes, discounted RHO per season -> columns rb_<s>, n_bx."""
        P = self.pairs
        rows = self.rows
        r = P["r"].to_numpy()
        disc = RHO ** (-_rel_season(P["sidx"]))
        key = P["i"].to_numpy() * 10_000_000 + P["j"].to_numpy()
        # pairs are sorted by (i, j, day); exclusive cumsum within a pair
        start = np.r_[True, key[1:] != key[:-1]]
        grp_id = np.cumsum(start) - 1

        def excl(v):                                  # per-pair cumsum of earlier rows
            v = np.asarray(v, float)
            return pd.Series(v).groupby(grp_id).cumsum().to_numpy() - v
        f = RHO ** _rel_season(P["sidx"])
        mins = rows["min"].to_numpy(float)[r]
        P["n_bx"] = excl(mins * disc) * f
        cum = {"i": P["i"].to_numpy(), "j": P["j"].to_numpy(), "day": P["day"].to_numpy(),
               "n": excl(mins * disc) + mins * disc}
        for s in DK_STATS:
            ov = rows[s].to_numpy(float)[r] * disc
            ev = np.asarray(mu[s], float)[r] * disc
            o = excl(ov) * f
            e = excl(ev) * f
            P[f"rb_{s}"] = np.where(e > 0, (o + 0.5) / (e + 0.5), 1.0)
            cum[f"o_{s}"] = excl(ov) + ov            # inclusive (for later queries)
            cum[f"e_{s}"] = excl(ev) + ev
        self.box_cum = pd.DataFrame(cum).sort_values("day").reset_index(drop=True)

    # ---------------------------------------------------------------- sim-time evidence
    def evidence(self, day, player_ids: Sequence[int]) -> dict:
        """Pair evidence strictly before `day` (date or int day) for a team's players, as
        (n, n) matrices indexed [j, i] (j absent, i affected): stint ratio rs_<s> and minutes
        n_st; box ratio rb_<s> and minutes n_bx (box needs `box_evidence` / `predict_log`
        to have run with the params in use)."""
        d = int(day) if isinstance(day, (int, np.integer)) else ST._day_of(day)
        sidx = ST.sidx_of(ST.season_of(pd.Timestamp(ST.EPOCH + np.timedelta64(d, "D")).date()))
        pids = np.asarray(player_ids, np.int64)
        n = len(pids)
        J, I = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        q = pd.DataFrame({"i": pids[I.ravel()], "j": pids[J.ravel()], "day": d,
                          "k": np.arange(n * n)})
        f = RHO ** _rel_season(sidx)
        out = {}
        if getattr(self, "stint_cum", None) is not None:
            a = pd.merge_asof(q, self.stint_cum, on="day", by=["i", "j"],
                              allow_exact_matches=False).sort_values("k")
            ok = a["n"].notna().to_numpy()
            st = stint_ratios(a, np.where(ok, f, 0.0))
            for s in DK_STATS:
                out[f"rs_{s}"] = st[s].reshape(n, n)
            out["n_st"] = np.nan_to_num(a["n"].to_numpy(float) * f).reshape(n, n)
        if getattr(self, "box_cum", None) is not None:
            b = pd.merge_asof(q, self.box_cum, on="day", by=["i", "j"],
                              allow_exact_matches=False).sort_values("k")
            out["n_bx"] = np.nan_to_num(b["n"].to_numpy(float) * f).reshape(n, n)
            for s in DK_STATS:
                o = np.nan_to_num(b[f"o_{s}"].to_numpy(float)) * f
                e = np.nan_to_num(b[f"e_{s}"].to_numpy(float)) * f
                out[f"rb_{s}"] = np.where(e > 0, (o + 0.5) / (e + 0.5), 1.0).reshape(n, n)
        return out


def stint_ratios(q: pd.DataFrame, f: np.ndarray, c: float = 1.0) -> dict:
    """Per-possession rate with j OFF / with j ON from cumulative stint sums (columns po, pt,
    yo_<s>, yt_<s>, times the season discount factor f); each side shrunk by `c` pseudo-counts
    to the pooled rate. 1.0 without evidence."""
    po = np.nan_to_num(q["po"].to_numpy(float)) * f
    pt = np.nan_to_num(q["pt"].to_numpy(float)) * f
    pn = np.maximum(pt - po, 0.0)
    out = {}
    for s in DK_STATS:
        yo = np.nan_to_num(q[f"yo_{s}"].to_numpy(float)) * f
        yt = np.nan_to_num(q[f"yt_{s}"].to_numpy(float)) * f
        yn = np.maximum(yt - yo, 0.0)
        rbar = np.where(pt > 0, yt / np.maximum(pt, 1e-9), 0.0)
        ok = (rbar > 0) & (po > 0) & (pn > 0)
        pseudo = np.where(ok, c / np.maximum(rbar, 1e-9), 0.0)
        r_off = (yo + c) / np.maximum(po + pseudo, 1e-9)
        r_on = (yn + c) / np.maximum(pn + pseudo, 1e-9)
        out[s] = np.where(ok, r_off / r_on, 1.0)
    return out


# ============================================================================== fitting

N_FEAT = N_CELL + len(ROLES)          # share by cell + one intercept per absentee by i's role
RIDGE_GRID = (3.0, 30.0, 300.0)       # L2 on the prior coefficients (chosen walk-forward)
INNER_FOLDS = 2                       # last training seasons used as walk-forward folds


def _nb_ll(y, mu, d) -> np.ndarray:
    return ST.nb_logpmf(y, mu, d)


def _pair_sum(D: UsageData, v) -> np.ndarray:
    return np.bincount(D.pairs["r"].to_numpy(), weights=np.asarray(v, float),
                       minlength=len(D.rows))


def _prior_features(D: UsageData, stat: str) -> np.ndarray:
    """(len(rows), N_FEAT): per row, sum over absent j of share_j,s in each cell, then the
    number of absentees in each role of i (the per-absentee intercept)."""
    P = D.pairs
    r = P["r"].to_numpy()
    X = np.zeros((len(D.rows), N_FEAT))
    np.add.at(X, (r, P["cell"].to_numpy()), P[f"share_{stat}"].to_numpy(float))
    np.add.at(X, (r, N_CELL + P["role"].to_numpy()), 1.0)
    return X


def _stint_feature(D: UsageData, stat: str, k: float) -> np.ndarray:
    P = D.pairs
    return _pair_sum(D, shrink_term(P[f"rs_{stat}"], P["n_st"], k))


def _box_feature(D: UsageData, stat: str, k: float) -> np.ndarray:
    P = D.pairs
    if f"rb_{stat}" not in P:
        return np.zeros(len(D.rows))
    return _pair_sum(D, shrink_term(P[f"rb_{stat}"], P["n_bx"], k))


def _glm(y, E, d, X, ridge, iters: int = 40) -> np.ndarray:
    """NB2 (fixed dispersion d) log-link regression with offset log E and an L2 penalty."""
    ridge = np.broadcast_to(np.asarray(ridge, float), (X.shape[1],))
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = E * np.exp(np.clip(X @ theta, -3, 3))
        den = 1.0 + d * mu
        g = X.T @ ((y - mu) / den) - ridge * theta
        H = (X * (mu / den)[:, None]).T @ X + np.diag(ridge)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(theta)), g)
        theta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return theta


def _stat_arrays(D: UsageData, s: str):
    rows = D.rows
    return (rows[s].to_numpy(float), np.maximum(rows[f"E_{s}"].to_numpy(float), 1e-6),
            rows[f"d_{s}"].to_numpy(float))


def fit_frame(D: UsageData, train: np.ndarray, *, k_grid=K_GRID, ridge_grid=RIDGE_GRID,
              layers=("stint", "box"), cap: float = CAP, folds: int = INNER_FOLDS,
              log: Callable[[str], None] | None = None) -> UsageParams:
    """Fit the prior (NB GLM) and choose K_stint, ridge and K_box per stat on the rows `train`
    (bool over D.rows) with >= 1 absent regular.

    Selection is walk-forward inside the training seasons: for each of the last `folds`
    training seasons t, the GLM is fit on training seasons < t and scored (NB log-likelihood,
    capped factor) on t; the configuration with the best summed score is refit on all training
    rows. K_box is then chosen the same way on the box layer (its pair evidence is causal)."""
    say = log or (lambda m: None)
    rows = D.rows
    has = _pair_sum(D, np.ones(len(D.pairs))) > 0
    tr = np.asarray(train, bool) & has
    seas = rows["season"].to_numpy()
    tseasons = sorted(set(seas[tr]))
    fold_s = tseasons[-folds:] if len(tseasons) > folds else tseasons[1:]
    p = default_params()
    p.cap = cap
    p.fit_seasons = tseasons
    p.alpha = {}
    p.ridge = {}
    diag = {"n_train": int(tr.sum()), "folds": fold_s, "stats": {}}
    base_logs = {}
    ks = k_grid if "stint" in layers else (float("inf"),)
    for s in DK_STATS:
        y, E, d = _stat_arrays(D, s)
        Xp = _prior_features(D, s)
        xs_k = {k: _stint_feature(D, s, k) for k in ks}
        scores = {}
        for k in ks:
            X = np.column_stack([Xp, xs_k[k]]) if np.isfinite(k) else Xp
            for lam in ridge_grid:
                rid = np.r_[np.full(N_FEAT, lam), np.full(X.shape[1] - N_FEAT, 1.0)]
                sc = 0.0
                for t in fold_s:
                    fi = tr & (seas < t)
                    te = tr & (seas == t)
                    if not fi.any() or not te.any():
                        continue
                    th = _glm(y[fi], E[fi], d[fi], X[fi], rid)
                    lg = np.clip(X[te] @ th, -cap, cap)
                    sc += float(_nb_ll(y[te], E[te] * np.exp(lg), d[te]).sum()
                                - _nb_ll(y[te], E[te], d[te]).sum())
                scores[(k, lam)] = sc
        (k, lam), sc = max(scores.items(), key=lambda kv: kv[1])
        X = np.column_stack([Xp, xs_k[k]]) if np.isfinite(k) else Xp
        rid = np.r_[np.full(N_FEAT, lam), np.full(X.shape[1] - N_FEAT, 1.0)]
        th = _glm(y[tr], E[tr], d[tr], X[tr], rid)
        if sc <= 0.0:                        # no walk-forward gain over no adjustment: off
            k, th = float("inf"), np.zeros(N_FEAT)
            X = Xp
        p.beta[s] = [float(x) for x in th[:N_CELL]]
        p.alpha[s] = [float(x) for x in th[N_CELL:N_FEAT]]
        p.lam[s] = float(th[N_FEAT]) if np.isfinite(k) else 0.0
        p.k_stint[s] = float(k)
        p.ridge[s] = float(lam)
        base_logs[s] = X @ th
        diag["stats"][s] = {"cv_gain_prior_stint": sc,
                            "cv_gain_prior_only": max(v for (kk, _), v in scores.items()
                                                      if not np.isfinite(kk))
                            if any(not np.isfinite(kk) for kk, _ in scores) else None}
        say(f"  {s}: K_stint {k:g} ridge {lam:g} lam {p.lam[s]:+.3f} cv gain {sc:+.1f} "
            f"beta {np.round(th[:N_CELL], 2).tolist()} alpha {np.round(th[N_CELL:N_FEAT], 3).tolist()}")
    if "box" in layers:
        D.box_evidence({s: _stat_arrays(D, s)[1] * np.exp(np.clip(base_logs[s], -cap, cap))
                        for s in DK_STATS})
        for s in DK_STATS:
            if not np.any(base_logs[s]):
                p.k_box[s] = float("inf")
                diag["stats"][s]["box_gain"] = 0.0
                continue
            y, E, d = _stat_arrays(D, s)
            sel = tr & np.isin(seas, fold_s)
            best = None
            for k in k_grid:
                lg = np.clip(base_logs[s] + _box_feature(D, s, k), -cap, cap)
                ll = float(_nb_ll(y[sel], E[sel] * np.exp(lg[sel]), d[sel]).sum())
                if best is None or ll > best[0] + 1e-9:
                    best = (ll, k)
            ll_off = float(_nb_ll(y[sel], E[sel] * np.exp(np.clip(base_logs[s], -cap, cap)[sel]),
                                  d[sel]).sum())
            p.k_box[s] = float(best[1])
            diag["stats"][s]["box_gain"] = best[0] - ll_off
            say(f"  {s}: K_box {best[1]:g} gain over prior+stint {best[0] - ll_off:+.1f}")
    p.diagnostics = diag
    return p


def _base_log(D: UsageData, p: UsageParams, s: str) -> np.ndarray:
    """Unclipped prior + stint log term per row."""
    th = np.r_[np.asarray(p.beta.get(s, [0.0] * N_CELL), float),
               np.asarray(p.alpha.get(s, [0.0] * len(ROLES)), float)]
    out = _prior_features(D, s) @ th
    if np.isfinite(p.k_of("stint", s)) and p.lam.get(s, 0.0):
        out = out + p.lam[s] * _stint_feature(D, s, p.k_of("stint", s))
    return out


def predict_log(D: UsageData, p: UsageParams, *, refresh_box: bool = True) -> dict:
    """{stat: (len(rows),)} total (clipped) log factor of every row under params p (all pair
    evidence causal). The box layer is recomputed from p's own prior + stint means."""
    base = {s: _base_log(D, p, s) for s in DK_STATS}
    if refresh_box and any(np.isfinite(p.k_of("box", s)) for s in DK_STATS):
        D.box_evidence({s: _stat_arrays(D, s)[1] * np.exp(np.clip(base[s], -p.cap, p.cap))
                        for s in DK_STATS})
    out = {}
    for s in DK_STATS:
        lg = base[s]
        if np.isfinite(p.k_of("box", s)):
            lg = lg + _box_feature(D, s, p.k_of("box", s))
        out[s] = np.clip(lg, -p.cap, p.cap)
    return out


def fit(conn: sqlite3.Connection, seasons: Sequence[str], sp: dict | None = None, *,
        data: UsageData | None = None, log: Callable[[str], None] | None = None,
        **kw) -> UsageParams:
    """UsageParams fit on `seasons` (all < S for a walk-forward season S). `sp` = the rate-state
    params of S (default state.DEFAULT_PARAMS)."""
    sp = sp or ST.DEFAULT_PARAMS
    D = data or UsageData(conn, seasons, sp, log=log)
    train = D.rows["season"].isin(list(seasons)).to_numpy()
    return fit_frame(D, train, log=log, **kw)


# ============================================================================== evaluation

def bump_multipliers(rows: pd.DataFrame, bump: dict) -> dict:
    """Tier-bump baseline per row (state.fit_usage_bump roles: 1 the other star, 2 rotation)."""
    role = rows["bump_role"].to_numpy()
    return {s: np.where(role == 1, bump.get("star", {}).get(s, 1.0),
                        np.where(role == 2, bump.get("rotation", {}).get(s, 1.0), 1.0))
            for s in DK_STATS}


def row_ll(rows: pd.DataFrame, mult: dict | None) -> dict:
    """NB log-likelihood per row and stat of the actual count given actual minutes."""
    out = {}
    for s in DK_STATS:
        E = np.maximum(rows[f"E_{s}"].to_numpy(float), 1e-6)
        m = 1.0 if mult is None else np.asarray(mult[s], float)
        out[s] = _nb_ll(rows[s].to_numpy(float), E * m, rows[f"d_{s}"].to_numpy(float))
    return out


def compare_ll(rows: pd.DataFrame, a: dict, b: dict, mask: np.ndarray) -> dict:
    """Paired LL difference a - b (sum over DK stats per row, and per stat), SE clustered by
    game."""
    from .metrics import paired_cluster
    mask = np.asarray(mask, bool)
    g = rows["game_id"].to_numpy()[mask]
    tot = sum(a[s] - b[s] for s in DK_STATS)[mask]
    pc = paired_cluster(tot, g)
    out = {"mean": pc["mean"], "se": pc["se"], "z": pc["mean"] / pc["se"] if pc["se"] else 0.0,
           "n": pc["n"], "games": pc["n_clusters"], "total": float(tot.sum()), "by_stat": {}}
    for s in DK_STATS:
        q = paired_cluster((a[s] - b[s])[mask], g)
        out["by_stat"][s] = {"mean": q["mean"], "se": q["se"]}
    return out


def dk_mean_shift(rows: pd.DataFrame, mult: dict) -> np.ndarray:
    """Change in expected DK points given actual minutes (no bonus) from multipliers."""
    return sum(DK_W[s] * rows[f"E_{s}"].to_numpy(float) * (np.asarray(mult[s]) - 1.0)
               for s in DK_STATS)


def evaluate_frame(D: UsageData, p: UsageParams, season: str, bump: dict) -> dict:
    """Acceptance on season `season` rows of D: NB LL of the 7 DK stats given actual minutes,
    usage v1 vs the tier bump (and vs no adjustment), on 'teammate sat' games (a top-2 star or
    a >= 25 mpg regular absent), on every game with an absent regular, and on healthy games;
    plus the DK residual given actual minutes by group (the per-minute part of the T2.7
    star-absence check)."""
    rows = D.rows
    lg = predict_log(D, p)
    mm = {s: np.exp(lg[s]) for s in DK_STATS}
    mb = bump_multipliers(rows, bump)
    ll_m, ll_b, ll_0 = row_ll(rows, mm), row_ll(rows, mb), row_ll(rows, None)
    te = (rows["season"] == season).to_numpy()
    has = _pair_sum(D, np.ones(len(D.pairs))) > 0
    star_ab = _pair_sum(D, D.pairs["star_j"].to_numpy(float)) > 0
    sat = rows["sat"].to_numpy(bool)
    out = {"season": season, "n_rows": int(te.sum())}
    for lab, m in (("sat", te & sat), ("star_out", te & star_ab), ("any_absent", te & has),
                   ("healthy", te & ~has), ("sat_mpg15", te & sat & (rows["mpg"] >= 15).to_numpy())):
        out[lab] = {"model_vs_bump": compare_ll(rows, ll_m, ll_b, m),
                    "bump_vs_none": compare_ll(rows, ll_b, ll_0, m),
                    "model_vs_none": compare_ll(rows, ll_m, ll_0, m)}
    # DK residual given actual minutes (per-minute part), top-6 by mpg, star out vs not
    dk_act = sum(DK_W[s] * rows[s].to_numpy(float) for s in DK_STATS)
    base = sum(DK_W[s] * rows[f"E_{s}"].to_numpy(float) for s in DK_STATS)
    top6 = (rows.groupby(["game_id", "team"])["mpg"].rank(ascending=False, method="first")
            <= 6).to_numpy()
    res = {}
    for lab, mult in (("none", None), ("bump", mb), ("model", mm)):
        mu = base if mult is None else base + dk_mean_shift(rows, mult)
        r_ = dk_act - mu
        a = r_[te & top6 & star_ab]
        b = r_[te & top6 & ~star_ab]
        res[lab] = {"top6_star_out": float(a.mean()), "top6_no_star_out": float(b.mean()),
                    "diff": float(a.mean() - b.mean()),
                    "se": float(np.sqrt(a.var() / len(a) + b.var() / len(b)))}
    out["pm_residual"] = res
    return out


# ============================================================================== sim API

def team_shares(team_players: pd.DataFrame) -> dict:
    """{stat: (n,)} each regular's share of the team's production (rate x mpg over regulars);
    0 for non-regulars."""
    reg = _regular_mask(team_players)
    mpg = team_players["mpg"].to_numpy(float)
    out = {}
    for s in DK_STATS:
        v = np.where(reg, team_players[f"r_{s}"].to_numpy(float) * mpg, 0.0)
        out[s] = np.clip(v / max(float(v.sum()), 1e-9), 0.0, 1.0)
    return out


def _regular_mask(team_players: pd.DataFrame) -> np.ndarray:
    if "regular" in team_players:
        return team_players["regular"].to_numpy(bool)
    return np.ones(len(team_players), bool)


def pair_matrix(p: UsageParams, team_players: pd.DataFrame, evidence: dict | None = None
                ) -> dict:
    """{stat: L (n, n)} with L[j, i] = log-factor on player i's per-minute rate of `stat` when
    teammate j is absent (rows of non-regulars and of regulars under p.min_absent_mpg are 0,
    the diagonal is 0). team_players columns: player_id, mpg, r_<stat>, grp (0 G, 1 F, 2 C,
    3 unknown), star (top-2 regulars by DK/min x mpg), regular."""
    tp = team_players.reset_index(drop=True)
    n = len(tp)
    mpg = tp["mpg"].to_numpy(float)
    grp = tp["grp"].to_numpy(int) if "grp" in tp else np.full(n, 3)
    star = tp["star"].to_numpy(bool) if "star" in tp else np.zeros(n, bool)
    role = role_of(mpg, star)
    trig = _regular_mask(tp) & (mpg >= p.min_absent_mpg)
    same = (grp[:, None] == grp[None, :]) & (grp[None, :] < 3)          # [j, i]
    cell = cell_of(np.broadcast_to(role[None, :], (n, n)), same)
    sh = team_shares(tp)
    ev = evidence or {}
    out = {}
    for s in DK_STATS:
        beta = np.asarray(p.beta.get(s, [0.0] * N_CELL), float)
        alpha = np.asarray(p.alpha.get(s, [0.0] * len(ROLES)), float)
        L = beta[cell] * sh[s][:, None] + alpha[role][None, :]
        k = p.k_of("stint", s)
        if np.isfinite(k) and p.lam.get(s, 0.0) and f"rs_{s}" in ev:
            L = L + p.lam[s] * shrink_term(ev[f"rs_{s}"], ev["n_st"], k)
        k = p.k_of("box", s)
        if np.isfinite(k) and f"rb_{s}" in ev:
            L = L + shrink_term(ev[f"rb_{s}"], ev["n_bx"], k)
        L = np.where(trig[:, None], L, 0.0)
        np.fill_diagonal(L, 0.0)
        out[s] = L
    return out


def factor_matrix(L: dict, absent: np.ndarray, cap: float = CAP) -> dict:
    """{stat: (S, n)} multipliers for absent masks (S, n) (or (n,)): exp(clip(sum over absent
    j of L[j, i], +-cap)). Exactly 1 for every player when nobody is absent."""
    a = np.atleast_2d(np.asarray(absent, bool)).astype(float)
    return {s: np.exp(np.clip(a @ Ls, -cap, cap)) for s, Ls in L.items()}


def factors(p: UsageParams, team_players: pd.DataFrame, absent, evidence: dict | None = None
            ) -> dict:
    """{player_id: {stat: multiplier}} for one absent set (bool mask over team_players or a
    collection of absent player ids)."""
    tp = team_players.reset_index(drop=True)
    pids = tp["player_id"].to_numpy(np.int64)
    a = np.asarray(absent)
    if a.dtype != bool or a.shape != (len(tp),):
        a = np.isin(pids, np.asarray(list(absent), np.int64))
    F = factor_matrix(pair_matrix(p, tp, evidence), a, p.cap)
    return {int(pid): {s: float(F[s][0, k]) for s in DK_STATS} for k, pid in enumerate(pids)}


def sim_means(players: pd.DataFrame, sp: dict, x: np.ndarray, played: np.ndarray,
              F: dict | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop-in for copula's bump grids: per-sim (S, n) raw points, raw other DK and
    reb/ast/fg3m given minutes x: rate x m x f(m/mpg) x usage factor."""
    x = np.asarray(x, float)
    mpg = np.broadcast_to(players["mpg"].to_numpy(float)[None, :], x.shape)
    mu = {}
    for s in DK_STATS:
        r = players[f"r_{s}"].to_numpy(float)[None, :]
        m = r * x * ST.minutes_multiplier(sp, s, x, mpg) * played
        mu[s] = m * F[s] if F is not None else m
    raw_oth = sum(DK_W[s] * mu[s] for s in OTHER_STATS)
    return mu["pts"], raw_oth, {s: mu[s] for s in ("reb", "ast", "fg3m")}


def params_path(season: str) -> Path:
    from .. import config
    return config.data_dir() / "proj" / f"usage_params_{season}.json"


N_TRAIN_SEASONS = 5


def load_usage(conn: sqlite3.Connection, season: str, sp: dict, *, refit: bool = False,
               log: Callable[[str], None] | None = None, **kw) -> tuple[UsageParams, UsageData]:
    """(params fit on the N_TRAIN_SEASONS seasons < S, UsageData over those + S with the box
    layer prepared) for the sim. Params cached at `params_path(S)`."""
    s0 = ST.sidx_of(season)
    seasons = [ST.season_label(y) for y in range(s0 - N_TRAIN_SEASONS, s0 + 1)]
    D = UsageData(conn, seasons, sp, log=log, **kw)
    path = params_path(season)
    if path.exists() and not refit:
        p = UsageParams.from_json(path.read_text())
    else:
        p = fit_frame(D, (D.rows["season"] < season).to_numpy(), log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p.to_json())
    predict_log(D, p)                      # prepares the box-layer evidence for `evidence`
    return p, D


def evaluate(conn: sqlite3.Connection, season: str, sp: dict, *,
             n_train: int = N_TRAIN_SEASONS, log: Callable[[str], None] | None = None,
             **kw) -> dict:
    """Walk-forward acceptance for season S: usage v1 fit on the n_train seasons < S (rate
    state `sp` fit on seasons < S), the tier bump = sp['usage_bump'] (the pipeline's, fit on the
    3 seasons < S) or refit the same way. Returns `evaluate_frame` plus params and bump."""
    s0 = ST.sidx_of(season)
    seasons = [ST.season_label(y) for y in range(s0 - n_train, s0 + 1)]
    D = UsageData(conn, seasons, sp, log=log, **kw)
    p = fit_frame(D, (D.rows["season"] < season).to_numpy(), log=log)
    bump = sp.get("usage_bump") or {}
    if not bump:
        tb = seasons[-4:-1]
        bump = ST.fit_usage_bump(conn, sp, tb, wf=D.wf[D.wf["season"].isin(tb)])
    ev = evaluate_frame(D, p, season, bump)
    ev["params"] = json.loads(p.to_json())
    ev["bump"] = {k: {s: v.get(s) for s in DK_STATS} for k, v in bump.items()}
    return ev


# ============================================================================== diagnosis

def diagnose(conn: sqlite3.Connection, seasons: Sequence[str]) -> dict:
    """Decompose the T2.7 star-absence residual (top-6 by projected minutes, actual DK - our
    pregame projection, star out vs not; corr extracts) into a minutes part
    (min - min_proj) x proj DK/min and a per-minute part dk - min x proj DK/min, raw and
    relative to the same player's own healthy-game residual (composition-adjusted)."""
    import importlib.util
    from .. import config
    from . import corr as K
    spec = importlib.util.spec_from_file_location(
        "dfs_correlations_v2", config.REPO_ROOT / "research" / "dfs_correlations_v2.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    f = K.load_frame(conn, list(seasons))
    d = f.merge(V.star_out_frame(conn, list(seasons)), on=["game_id", "team"], how="left")
    d = d[d["n_star_out"].notna()].copy()
    d["so"] = d["n_star_out"] >= 1
    d["top6"] = d.groupby(["game_id", "team"])["min_proj"].rank(ascending=False,
                                                                method="first") <= 6
    rate = d["mu"] / d["min_proj"].clip(lower=1.0)
    d["res"] = d["dk"] - d["mu"]
    d["c_min"] = (d["min"] - d["min_proj"]) * rate
    d["c_pm"] = d["dk"] - d["min"] * rate
    comp = ("res", "c_min", "c_pm")
    x = d[d["top6"]].copy()
    out = {"n_star_out": int(x["so"].sum()), "n_healthy": int((~x["so"]).sum())}

    def cl_se(v, g):
        from .metrics import paired_cluster
        return paired_cluster(v, g)["se"]
    out["raw"] = {c: {"diff": float(x.loc[x.so, c].mean() - x.loc[~x.so, c].mean())}
                  for c in comp}
    h = x[~x["so"]].groupby(["player_id", "season"])[list(comp)].mean().add_suffix("_h")
    x = x.join(h, on=["player_id", "season"])
    s = x[x["so"] & x["res_h"].notna()]
    out["composition_adjusted"] = {
        c: {"excess": float((s[c] - s[c + "_h"]).mean()),
            "se": cl_se((s[c] - s[c + "_h"]).to_numpy(), s["game_id"].to_numpy()),
            "own_healthy_mean": float(s[c + "_h"].mean()), "star_out_mean": float(s[c].mean())}
        for c in comp}
    out["n_matched"] = int(len(s))
    return out


# ============================================================================== CLI

def main(argv: list[str] | None = None) -> int:
    import argparse
    from .. import config
    ap = argparse.ArgumentParser(description="usage v1 with/without factors (T2.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("tallies", help="build / extend the stint pair tally cache")
    a1.add_argument("--workers", type=int, default=1)
    a2 = sub.add_parser("eval", help="walk-forward acceptance for a season")
    a2.add_argument("--season", required=True)
    a2.add_argument("--state-params", default=None,
                    help="rate-state params JSON fit on seasons < S (default "
                         "<data>/proj/state_params_<S>.json)")
    a2.add_argument("--out", default=None)
    a3 = sub.add_parser("diagnose", help="decompose the T2.7 star-absence residual")
    a3.add_argument("--seasons", nargs="+",
                    default=[ST.season_label(y) for y in range(2016, 2026)])
    a = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{config.db_path()}?mode=ro", uri=True)
    if a.cmd == "tallies":
        r = build_stint_tallies(conn, None, workers=a.workers, log=print)
        print({k: len(v) for k, v in r.items()})
    elif a.cmd == "eval":
        path = Path(a.state_params or (config.data_dir() / "proj"
                                       / f"state_params_{a.season}.json"))
        sp = json.loads(path.read_text())
        ev = evaluate(conn, a.season, sp, log=print)
        txt = json.dumps(ev, indent=1, default=float)
        if a.out:
            Path(a.out).write_text(txt)
        print(txt)
    else:
        print(json.dumps(diagnose(conn, a.seasons), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
