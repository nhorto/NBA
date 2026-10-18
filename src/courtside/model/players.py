"""Player ratings, expected minutes and player-based team strength, strictly sequential.

One chronological pass over the games (`run_player_engine`). Each game day:

1. **Morning solve.** Player ratings are the posterior mean of a ridge (Gaussian-prior)
   adjusted plus-minus regression over every earlier game, with exponentially decayed evidence
   and a per-player prior mean from a box-score composite:

       r = argmin  sum_obs w_o (pm_o - x_o . r)^2  +  lam * sum_i (r_i - mu_i)^2

   One observation per player-game: the player's on-court plus-minus `pm_i`. Its expectation
   under random lineup mixing within the game (G = game length in minutes) is

       E[pm_i] = (m_i/48) * ( r_i + sum_{k teammate} r_k * 4 m_k / (5G - m_i)
                                  - sum_{j opponent} r_j * m_j / G  + h * home_sign )

   with weight w = 48 / m_i (on-court margin variance grows with minutes). `r` is in points per
   48 minutes of team margin, so team strength = sum r_i * minutes_i / 48 is the team's expected
   margin against an average team. `h` is a global home-court term (0 at neutral sites).
   The sufficient statistics A = X'WX and b = X'Wy are accumulated incrementally (time-scaled
   so the exponential decay costs nothing) and the solve runs over players active in the last
   `active_days` only.

2. **Box prior.** mu_i = beta . (z_i - z_bar). z_i are the player's decayed per-100-possession
   box rates (shrunk toward league rates with `box_n0` pseudo-possessions), minutes per game
   (shrunk toward `m0`), log evidence minutes and rookie/sophomore flags. beta is refit at every
   season start by a *predictive* regression: the margin of each game in the previous
   `beta_window` seasons on sum(minutes/48 * z_pre) home minus away, where z_pre is the
   pre-game feature vector. Nothing from the current season is used. A never-seen player gets
   mu(z_default): league-average rates, m0 minutes, zero evidence, rookie flag, i.e. a
   replacement-level prior learned from real debuts.

3. **Season boundary.** Evidence carried into a new season is multiplied by `apm_offseason`
   (APM) and `box_offseason` (box stats) on top of the time decay, so ratings regress toward
   the (box) prior over the summer.

4. **Expected minutes and lineups.** `expected_lineup` (shared by backtest and production):
   project each candidate's minutes from a recency-weighted EWMA of his minutes when playing,
   weight by P(plays), drop deep-bench players (projection < `floor`, keeping at least
   `min_players`), and renormalise to 240 with a per-player cap (water-filling).

The pre-game outputs of day d depend only on games before d, plus (in the backtest) the set
of players who logged minutes in the game itself, the availability oracle; see MODEL.md.
"""

from __future__ import annotations

import sqlite3
import zlib
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

RATE_STATS = ("pts", "fga", "fta", "fg3a", "ast", "tov", "oreb", "dreb", "stl", "blk", "pf")
FEATURES = RATE_STATS + ("mpg", "evid", "rookie", "soph")
N_RATE = len(RATE_STATS)
N_FEAT = len(FEATURES)
PLAYER_COLUMNS = ("game_id", "player_id", "player_name", "team", "min", *RATE_STATS,
                  "plus_minus", "poss")

# Availability modes computed in one pass (strength columns "<mode>_home", "<mode>_away").
#   played          players who logged minutes (oracle availability), floor applied
#   played_nofloor  same, every player who logged minutes (incl. garbage time)
#   roster          ignore absences: everyone in the team's recent rotation history
#   soft            "played", but status-change players marked Questionable (p=0.5) w.p. q_soft
#   flip            "played", but status-change players' availability flipped w.p. q_flip
#   box             "played" lineup, rated by the box prior mu only (no plus-minus)
ALL_MODES = ("played", "played_nofloor", "roster", "soft", "flip", "box")


@dataclass(frozen=True)
class PlayerParams:
    lam: float = 30.0              # prior precision of each rating (per-48 units)
    lam_hca: float = 1.0
    apm_half_life: float = 365.0   # days
    apm_offseason: float = 0.7     # extra evidence carry over each summer
    box_half_life: float = 365.0
    box_offseason: float = 0.8
    box_n0: float = 300.0          # pseudo-possessions toward league-average rates
    m0: float = 12.0               # minutes prior for a player with no history
    mpg_k: float = 3.0             # pseudo-games toward m0 in the mpg box feature
    min_alpha_short: float = 0.35  # EWMA weights of minutes when playing
    min_alpha_long: float = 0.08
    min_w_short: float = 0.5
    floor: float = 6.0             # projected minutes below this: deep bench, not in lineup
    cap: float = 42.0
    min_players: int = 8
    roster_games: int = 10         # "roster" = played for the team in its last N games
    active_days: int = 1100
    beta_window: int = 6
    beta_ridge: float = 1.0
    q_soft: float = 0.5
    q_flip: float = 0.25
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Frozen after `courtside model tune-players` (2000-01..2010-11 only, scored 2003-04..2010-11,
# 102 configs, tuning log-loss 0.58293; the top 10 are within 0.0003). Two a-priori overrides,
# not chosen on results: floor=6 (identical tuning score to floor=0 to 1e-5, and it removes
# the outcome-linked garbage-time appearances from the lineup) and active_days=1460 (match the
# 4-year evidence half-life).
TUNED = PlayerParams(lam=300.0, apm_half_life=1460.0, apm_offseason=0.8, box_half_life=365.0,
                     box_offseason=0.6, box_n0=600.0, min_alpha_short=0.2, min_alpha_long=0.02,
                     min_w_short=0.5, floor=6.0, active_days=1460)


# --- data ----------------------------------------------------------------------------------

def load_player_games(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-player box rows (+ possessions from player_games_adv)."""
    q = ("SELECT p.game_id, p.player_id, p.player_name, p.team, p.min, "
         + ", ".join(f"p.{c}" for c in RATE_STATS)
         + ", p.plus_minus, a.poss FROM player_games p "
           "LEFT JOIN player_games_adv a USING (game_id, player_id)")
    df = pd.read_sql_query(q, conn)
    for c in ("min", *RATE_STATS, "plus_minus", "poss"):
        df[c] = df[c].astype("float64")
    return df


def _prepare(games: pd.DataFrame, players: pd.DataFrame):
    """Align player rows to games. Returns sorted arrays and per-game slices."""
    gi = pd.Series(np.arange(len(games)), index=games["game_id"].to_numpy())
    p = players[players["game_id"].isin(gi.index)]
    p = p[p["min"].fillna(0) > 0].copy()
    p["gi"] = gi.loc[p["game_id"].to_numpy()].to_numpy()
    hr = games["home_raw"].to_numpy()[p["gi"].to_numpy()]
    ar = games["away_raw"].to_numpy()[p["gi"].to_numpy()]
    team = p["team"].to_numpy()
    side = np.where(team == hr, 0, np.where(team == ar, 1, -1))
    p["side"] = side
    p = p[p["side"] >= 0]
    p = p.sort_values(["gi", "side", "min"], ascending=[True, True, False], kind="mergesort")
    # player index in order of first appearance (causal)
    codes, uniq = pd.factorize(p["player_id"], sort=False)
    p["pi"] = codes
    poss = p["poss"].to_numpy(float)
    # possessions missing: estimate from minutes at ~2.08 poss/min
    poss = np.where(np.isfinite(poss) & (poss > 0), poss, p["min"].to_numpy() * 2.08)
    gis = p["gi"].to_numpy()
    starts = np.searchsorted(gis, np.arange(len(games) + 1))
    names = p.groupby("pi")["player_name"].last().reindex(range(len(uniq))).to_numpy()
    return {
        "gi": gis, "side": p["side"].to_numpy(), "pi": p["pi"].to_numpy(),
        "min": p["min"].to_numpy(float), "pm": p["plus_minus"].fillna(0).to_numpy(float),
        "poss": poss, "stats": p[list(RATE_STATS)].fillna(0).to_numpy(float),
        "starts": starts, "player_ids": np.asarray(uniq), "names": names,
    }


# --- lineups -------------------------------------------------------------------------------

def allocate_minutes(proj: np.ndarray, p_play: np.ndarray, total: float = 240.0,
                     cap: float = 42.0) -> np.ndarray:
    """Expected minutes: weights p*proj renormalised to `total`, no player above `cap`
    (excess redistributed proportionally among the uncapped)."""
    w = np.asarray(p_play, float) * np.maximum(np.asarray(proj, float), 0.0)
    out = np.zeros_like(w)
    if w.sum() <= 0:
        return out
    free = w > 0
    remaining = total
    for _ in range(len(w) + 1):
        s = w[free].sum()
        if s <= 0:
            break
        trial = np.where(free, w * remaining / s, out)
        over = free & (trial > cap)
        if not over.any():
            out = trial
            break
        out = np.where(over, cap, out)
        free = free & ~over
        remaining = total - out[~free].sum()
    return out


def select_lineup(proj: np.ndarray, p_play: np.ndarray, floor: float, min_players: int
                  ) -> np.ndarray:
    """Mask of players who get minutes: P(play) > 0 and projection >= floor; if fewer than
    min_players qualify, add the next-highest projections."""
    avail = p_play > 0
    keep = avail & (proj >= floor)
    if keep.sum() < min_players:
        rest = np.flatnonzero(avail & ~keep)
        need = min_players - int(keep.sum())
        keep[rest[np.argsort(-proj[rest], kind="stable")[:need]]] = True
    return keep


def lineup_strength(ratings, proj, p_play, prm: PlayerParams, floor: float | None = None):
    """(strength, minutes): strength = sum rating * minutes / 48 over the expected lineup."""
    f = prm.floor if floor is None else floor
    keep = select_lineup(proj, p_play, f, prm.min_players)
    mins = np.zeros(len(proj))
    mins[keep] = allocate_minutes(proj[keep], p_play[keep], cap=prm.cap)
    return float(np.dot(ratings, mins) / 48.0), mins


# --- engine --------------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Engine state at the end of a pass: what production needs to build tomorrow's lineups."""
    params: PlayerParams
    player_ids: np.ndarray
    names: np.ndarray
    rating: np.ndarray          # last solved posterior mean (nan: never solved)
    prior: np.ndarray           # box prior mu at the last solve
    proj_min: np.ndarray        # minutes projection when playing
    last_team: np.ndarray       # franchise of the last game (object)
    last_day: np.ndarray        # day number of last game
    season_minutes: np.ndarray  # minutes logged in the last season of the pass
    team_history: dict          # team -> deque of player-index arrays (recent games)
    mu_new: float               # prior of a never-seen player
    hca: float
    as_of: pd.Timestamp | None = None
    _pos: dict = field(default_factory=dict)

    def index_of(self, player_id) -> int | None:
        if not self._pos:
            self._pos.update({int(p): i for i, p in enumerate(self.player_ids)})
        return self._pos.get(int(player_id))

    def roster(self, team: str) -> np.ndarray:
        """Players who appeared for `team` in its recent games and whose last team it is."""
        hist = self.team_history.get(team)
        if not hist:
            return np.array([], dtype=int)
        cand = np.unique(np.concatenate(list(hist)))
        return cand[self.last_team[cand] == team]

    def expected_lineup(self, team: str, game: dict | None = None,
                        availability: dict | None = None) -> pd.DataFrame:
        """Expected lineup for `team`'s next game.

        availability: {player_id: P(plays)}; players not listed default to 1.0 if on the
        recent roster. Unknown player_ids in `availability` with P > 0 (new signings) are
        added with the new-player prior. None = full roster (ignore injuries).
        `game` is informational (date, opponent) and unused by the minutes logic.
        """
        idx = list(self.roster(team))
        extra = []
        for pid, pr in (availability or {}).items():
            i = self.index_of(pid)
            if i is None:
                if pr > 0:
                    extra.append((pid, pr))
            elif i not in idx and pr > 0:
                idx.append(i)
        idx = np.asarray(idx, dtype=int)
        p_play = np.array([float((availability or {}).get(int(self.player_ids[i]), 1.0))
                           for i in idx])
        proj = self.proj_min[idx].astype(float)
        rat = np.where(np.isfinite(self.rating[idx]), self.rating[idx], self.mu_new)
        pids = [int(self.player_ids[i]) for i in idx]
        names = [self.names[i] for i in idx]
        for pid, pr in extra:
            pids.append(int(pid)); names.append(None)
            proj = np.append(proj, self.params.m0); rat = np.append(rat, self.mu_new)
            p_play = np.append(p_play, pr)
        strength, mins = lineup_strength(rat, proj, p_play, self.params)
        df = pd.DataFrame({"player_id": pids, "name": names, "p_play": p_play,
                           "proj_min": proj, "minutes": mins, "rating": rat,
                           "contribution": rat * mins / 48.0})
        df = df.sort_values("minutes", ascending=False, kind="stable").reset_index(drop=True)
        df.attrs["strength"] = strength
        return df

    def ratings_table(self, min_season_minutes: float = 0.0) -> pd.DataFrame:
        df = pd.DataFrame({"player_id": self.player_ids, "name": self.names,
                           "team": self.last_team, "rating": self.rating, "prior": self.prior,
                           "season_min": self.season_minutes, "proj_min": self.proj_min})
        df = df[np.isfinite(df["rating"]) & (df["season_min"] >= min_season_minutes)]
        return df.sort_values("rating", ascending=False).reset_index(drop=True)


def _season_index(games: pd.DataFrame) -> np.ndarray:
    s = games["season"].to_numpy()
    uniq = sorted(set(s))
    pos = {v: i for i, v in enumerate(uniq)}
    return np.array([pos[v] for v in s]), uniq


def run_player_engine(games: pd.DataFrame, players: pd.DataFrame,
                      params: PlayerParams = TUNED, modes: tuple[str, ...] = ALL_MODES,
                      return_state: bool = False):
    """Sequential pass. Returns a DataFrame aligned with `games` with columns
    `<mode>_home`, `<mode>_away` (pre-game team strength, points vs average) for each mode,
    plus `apm_hca`; optionally also the final PlayerState."""
    prm = params
    n_games = len(games)
    d = _prepare(games, players)
    N = len(d["player_ids"])
    H = N  # index of the home-court parameter in the APM system
    day = (games["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64))
    if n_games > 1 and (np.diff(day) < 0).any():
        raise ValueError("games must be sorted by game_date")
    sidx, _ = _season_index(games)
    home_team = games["home"].to_numpy()
    away_team = games["away"].to_numpy()
    neutral = games["is_neutral"].to_numpy(bool)
    glen = 48.0 + 5.0 * games["ot"].to_numpy(float) if "ot" in games else np.full(n_games, 48.0)
    margin = games["margin"].to_numpy(float)
    gid = games["game_id"].to_numpy()

    # player state
    box_S = np.zeros((N, N_RATE)); box_poss = np.zeros(N); box_min = np.zeros(N)
    box_gp = np.zeros(N); box_t = np.zeros(N); box_season = np.zeros(N, dtype=int)
    ew_s = np.full(N, prm.m0); ew_l = np.full(N, prm.m0)
    first_season = np.full(N, -1)
    last_day = np.full(N, -10**9, dtype=np.int64)
    last_team = np.empty(N, dtype=object)
    season_min = np.zeros(N)
    rating = np.full(N, np.nan); prior = np.full(N, np.nan)
    team_hist: dict[str, deque] = {}
    team_last: dict[str, np.ndarray] = {}

    # league rates (per possession): previous season totals, else running current season
    lg_cur = np.zeros(N_RATE + 1); lg_prev = None

    # APM sufficient statistics (time-scaled)
    A = np.zeros((N + 1, N + 1)); b = np.zeros(N + 1); t_ref = int(day[0]) if n_games else 0
    hca = 0.0

    # box-prior regression samples
    F = np.zeros((n_games, N_FEAT)); F_ok = np.zeros(n_games, bool)
    Zm = np.zeros((n_games, N_FEAT)); Mm = np.zeros(n_games)
    beta = np.zeros(N_FEAT); zbar = np.zeros(N_FEAT)
    ln_half_b = np.log(0.5) / prm.box_half_life
    ln2_apm = np.log(2.0) / prm.apm_half_life

    out = {f"{m}_{s}": np.full(n_games, np.nan) for m in modes for s in ("home", "away")}
    out["apm_hca"] = np.full(n_games, np.nan)

    def league_rates():
        src = lg_prev if lg_prev is not None else lg_cur
        if src[-1] <= 0:
            return np.zeros(N_RATE)
        return src[:-1] / src[-1]

    def z_of(idx, t, s, L):
        f = np.exp(ln_half_b * (t - box_t[idx])) * prm.box_offseason ** (s - box_season[idx])
        f = np.where(box_gp[idx] > 0, f, 0.0)
        S = box_S[idx] * f[:, None]; P = box_poss[idx] * f
        M = box_min[idx] * f; G = box_gp[idx] * f
        z = np.empty((len(idx), N_FEAT))
        z[:, :N_RATE] = 100.0 * (S + prm.box_n0 * L) / (P + prm.box_n0)[:, None]
        z[:, N_RATE] = (M + prm.mpg_k * prm.m0) / (G + prm.mpg_k)
        z[:, N_RATE + 1] = np.log1p(M / 100.0)
        fs = first_season[idx]
        z[:, N_RATE + 2] = ((fs == s) | (fs < 0)) & (s > 0)
        z[:, N_RATE + 3] = (fs == s - 1) & (fs > 0)
        return z

    def z_new(s, L):
        z = np.empty(N_FEAT)
        z[:N_RATE] = 100.0 * L
        z[N_RATE] = prm.m0; z[N_RATE + 1] = 0.0
        z[N_RATE + 2] = float(s > 0); z[N_RATE + 3] = 0.0
        return z

    def fit_beta(s):
        """Predictive regression on seasons [max(1, s-W), s-1]; z_bar from season s-1."""
        lo = max(1, s - prm.beta_window)
        m = F_ok & (sidx >= lo) & (sidx <= s - 1)
        if m.sum() < 50:
            return np.zeros(N_FEAT), np.zeros(N_FEAT)
        X = F[m]
        hc = (~neutral[m]).astype(float)
        sd = X.std(axis=0); sd[sd == 0] = 1.0
        Xs = np.column_stack([X / sd, hc])
        P = np.eye(Xs.shape[1]) * prm.beta_ridge; P[-1, -1] = 1e-6
        coef = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ margin[m])
        bta = coef[:-1] / sd
        last = (sidx == s - 1) & F_ok
        zb = Zm[last].sum(axis=0) / max(Mm[last].sum(), 1e-9)
        return bta, zb

    cur_season = sidx[0] if n_games else 0
    starts, P_pi, P_side, P_min = d["starts"], d["pi"], d["side"], d["min"]
    P_pm, P_poss, P_stats = d["pm"], d["poss"], d["stats"]
    mu_new = 0.0
    dirty = False
    # minutes-projection diagnostic per season: [sum |err|, sum err^2, n] over played players
    min_err = np.zeros((int(sidx.max()) + 1 if n_games else 1, 3))

    gi = 0
    while gi < n_games:
        t = int(day[gi])
        gj = gi
        while gj < n_games and day[gj] == t:
            gj += 1
        s = int(sidx[gi])
        # -- season boundary
        if s != cur_season:
            sc = np.exp(-ln2_apm * (t - t_ref)) * prm.apm_offseason
            A *= sc; b *= sc; t_ref = t
            lg_prev = lg_cur.copy(); lg_cur[:] = 0.0
            beta, zbar = fit_beta(s)
            season_min[:] = 0.0
            cur_season = s
            dirty = True
        L = league_rates()
        mu_new = float(beta @ (z_new(s, L) - zbar))

        # -- morning solve over active players
        if dirty:
            act = np.flatnonzero(last_day >= t - prm.active_days)
            if len(act):
                zact = z_of(act, t, s, L)
                mu = (zact - zbar) @ beta
                sc = np.exp(-ln2_apm * (t - t_ref))
                ix = np.append(act, H)
                M = A[np.ix_(ix, ix)] * sc
                lam = np.full(len(ix), prm.lam); lam[-1] = prm.lam_hca
                M[np.diag_indices_from(M)] += lam
                rhs = b[ix] * sc + lam * np.append(mu, 0.0)
                x = cho_solve(cho_factor(M, lower=True, check_finite=False), rhs,
                              check_finite=False)
                rating[act] = x[:-1]; prior[act] = mu; hca = float(x[-1])
            dirty = False

        # -- predict today's games (pre-game)
        for g in range(gi, gj):
            a0, a1 = starts[g], starts[g + 1]
            rows = np.arange(a0, a1)
            out["apm_hca"][g] = hca
            for side, team in ((0, home_team[g]), (1, away_team[g])):
                rs = rows[P_side[rows] == side]
                played = P_pi[rs]
                hist = team_hist.get(team)
                roster = np.array([], dtype=int)
                if hist:
                    cand = np.unique(np.concatenate(list(hist)))
                    roster = cand[last_team[cand] == team]
                cands = np.unique(np.concatenate([played, roster])).astype(int)
                if len(cands) == 0:
                    for m in modes:
                        out[f"{m}_{'home' if side == 0 else 'away'}"][g] = 0.0
                    continue
                proj = prm.min_w_short * ew_s[cands] + (1 - prm.min_w_short) * ew_l[cands]
                rat = np.where(np.isfinite(rating[cands]), rating[cands], mu_new)
                is_played = np.isin(cands, played)
                in_roster = np.isin(cands, roster)
                key = "home" if side == 0 else "away"
                p_pl = is_played.astype(float)
                for m in modes:
                    if m == "played":
                        v, _ = lineup_strength(rat, proj, p_pl, prm)
                    elif m == "played_nofloor":
                        v, _ = lineup_strength(rat, proj, p_pl, prm, floor=0.0)
                    elif m == "roster":
                        v, _ = lineup_strength(rat, proj, in_roster.astype(float), prm)
                    elif m == "box":
                        pr_ = np.where(np.isfinite(prior[cands]), prior[cands], mu_new)
                        v, _ = lineup_strength(pr_, proj, p_pl, prm)
                    else:  # soft / flip
                        last = team_last.get(team, np.array([], dtype=int))
                        in_last = np.isin(cands, last)
                        border = (is_played & in_roster & ~in_last) | (~is_played & in_last)
                        rng = np.random.default_rng(
                            [prm.seed, zlib.crc32(f"{gid[g]}{side}{m}".encode())])
                        u = rng.random(len(cands))
                        pp = p_pl.copy()
                        if m == "soft":
                            pp[border & (u < prm.q_soft)] = 0.5
                        else:
                            hit = border & (u < prm.q_flip)
                            pp[hit] = 1.0 - pp[hit]
                        v, _ = lineup_strength(rat, proj, pp, prm)
                    out[f"{m}_{key}"][g] = v

            # box-prior regression sample (pre-game z, actual minutes)
            if a1 > a0 and s >= 1:
                pis = P_pi[rows]
                z = z_of(pis, t, s, L)
                wmin = P_min[rows] / 48.0
                sign = np.where(P_side[rows] == 0, 1.0, -1.0)
                F[g] = (z * (wmin * sign)[:, None]).sum(axis=0)
                Zm[g] = (z * P_min[rows][:, None]).sum(axis=0)
                Mm[g] = P_min[rows].sum()
                F_ok[g] = True

        # -- ingest today's results
        for g in range(gi, gj):
            a0, a1 = starts[g], starts[g + 1]
            if a1 == a0:
                continue
            rows = np.arange(a0, a1)
            pis = P_pi[rows]; mins = P_min[rows]; sides = P_side[rows]
            err = prm.min_w_short * ew_s[pis] + (1 - prm.min_w_short) * ew_l[pis] - mins
            min_err[s] += (np.abs(err).sum(), (err ** 2).sum(), len(err))
            # box state
            f = np.exp(ln_half_b * (t - box_t[pis])) * prm.box_offseason ** (s - box_season[pis])
            box_S[pis] = box_S[pis] * f[:, None] + P_stats[rows]
            box_poss[pis] = box_poss[pis] * f + P_poss[rows]
            box_min[pis] = box_min[pis] * f + mins
            box_gp[pis] = box_gp[pis] * f + 1.0
            box_t[pis] = t; box_season[pis] = s
            lg_cur[:-1] += P_stats[rows].sum(axis=0); lg_cur[-1] += P_poss[rows].sum()
            ew_s[pis] += prm.min_alpha_short * (mins - ew_s[pis])
            ew_l[pis] += prm.min_alpha_long * (mins - ew_l[pis])
            fs = first_season[pis]
            first_season[pis] = np.where(fs < 0, s, fs)
            last_day[pis] = t
            season_min[pis] += mins
            for side, team in ((0, home_team[g]), (1, away_team[g])):
                ps_ = pis[sides == side]
                last_team[ps_] = team
                dq = team_hist.setdefault(team, deque(maxlen=prm.roster_games))
                dq.append(ps_)
                team_last[team] = ps_
            # APM rows
            Gm = glen[g]
            hsign = 0.0 if neutral[g] else 1.0
            n = len(rows)
            X = np.zeros((n, n + 1))
            w48 = mins / 48.0
            for side in (0, 1):
                me = sides == side
                op = ~me
                mi = mins[me]
                own = (w48[me] * 4.0 / (5.0 * Gm - mi))[:, None] * mins[me][None, :]
                X[np.ix_(me, me)] = own
                X[np.flatnonzero(me), np.flatnonzero(me)] = w48[me]
                X[np.ix_(me, op)] = -(w48[me] / Gm)[:, None] * mins[op][None, :]
                X[me, n] = w48[me] * hsign * (1.0 if side == 0 else -1.0)
            w = 1.0 / np.maximum(w48, 1.0 / 48.0)
            e = np.exp(ln2_apm * (t - t_ref))
            XtW = X.T * w
            ix = np.append(pis, H)
            A[np.ix_(ix, ix)] += e * (XtW @ X)
            b[ix] += e * (XtW @ P_pm[rows])
            dirty = True
        gi = gj

    res = pd.DataFrame(out, index=games.index)
    if not return_state:
        return res
    t_end = int(day[-1]) + 1 if n_games else 0
    proj = prm.min_w_short * ew_s + (1 - prm.min_w_short) * ew_l
    state = PlayerState(params=prm, player_ids=d["player_ids"], names=d["names"],
                        rating=rating.copy(), prior=prior.copy(), proj_min=proj,
                        last_team=last_team, last_day=last_day, season_minutes=season_min,
                        team_history=team_hist, mu_new=mu_new, hca=hca,
                        as_of=games["game_date"].iloc[-1] if n_games else None)
    # final solve including the last day's games, so production ratings are current
    act = np.flatnonzero(last_day >= t_end - prm.active_days)
    if len(act):
        L = league_rates()
        mu = (z_of(act, t_end, cur_season, L) - zbar) @ beta
        sc = np.exp(-ln2_apm * (t_end - t_ref))
        ix = np.append(act, H)
        M = A[np.ix_(ix, ix)] * sc
        lam = np.full(len(ix), prm.lam); lam[-1] = prm.lam_hca
        M[np.diag_indices_from(M)] += lam
        x = cho_solve(cho_factor(M, lower=True), b[ix] * sc + lam * np.append(mu, 0.0))
        state.rating[act] = x[:-1]; state.prior[act] = mu; state.hca = float(x[-1])
    state.beta = dict(zip(FEATURES, map(float, beta)))
    _, uniq = _season_index(games)
    state.minutes_error = {uniq[i]: {"mae": float(e[0] / e[2]), "rmse": float(np.sqrt(e[1] / e[2]))}
                           for i, e in enumerate(min_err) if e[2] > 0}
    return res, state


# --- production availability ---------------------------------------------------------------

STATUS_P_PLAY = {"out": 0.0, "out for season": 0.0, "suspended": 0.0, "doubtful": 0.15,
                 "questionable": 0.5, "day-to-day": 0.6, "game time decision": 0.5,
                 "probable": 0.9, "available": 1.0}


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    for suf in (" jr.", " jr", " sr.", " iii", " ii", " iv"):
        s = s.removesuffix(suf)
    return "".join(ch for ch in s if ch.isalnum())


def availability_from_injuries(conn: sqlite3.Connection, state: PlayerState, team: str,
                               as_of: str, sources=("official", "espn")) -> dict:
    """{player_id: P(plays)} for `team` from the latest injury snapshot at or before `as_of`
    (ISO UTC). Names are matched to the rated players; the first listed source that has a
    snapshot wins. Unmatched names are ignored (reported in the attrs of nothing: callers can
    diff against the snapshot)."""
    by_name = {_norm_name(n): int(p) for p, n in zip(state.player_ids, state.names)
               if isinstance(n, str)}
    for src in sources:
        row = conn.execute("SELECT MAX(captured_at) FROM injury_snapshots WHERE source=? AND "
                           "team=? AND captured_at<=?", (src, team, as_of)).fetchone()
        if not row or not row[0]:
            continue
        av = {}
        for name, status in conn.execute(
                "SELECT player_name, status FROM injury_snapshots WHERE source=? AND team=? "
                "AND captured_at=?", (src, team, row[0])):
            pid = by_name.get(_norm_name(name))
            if pid is not None:
                av[pid] = STATUS_P_PLAY.get((status or "").strip().lower(), 0.5)
        return av
    return {}
