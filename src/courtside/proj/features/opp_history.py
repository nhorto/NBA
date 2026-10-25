"""F8 / F8c: player vs this opponent (re-evaluation H1, docs/research/reeval-B.md).

The owner's idea: "how well a player does against certain teams... might help determine how good
or bad they will play against that team". Built as a **residual** on what the model already
knows, so it can only add the player x opponent *interaction*:

* Per past game g of the player, stat s: the base expectation Eb = E_s x exp(f2_dvp_<s>_grp), i.e.
  the EB rate x actual minutes x market reconciliation (`common.base_rows`) x the opponent's
  as-of allowed-vs-expected index for the player's position group (F2, live). So neither
  the player's level (EB), the game environment (market total) nor the opponent's defence by
  position (F2) is counted again.
* Player level from his other opponents: R_all = (A_y - P_y + K0 ebar) / (A_E - P_E + K0 ebar)
  over the player's earlier games (recency weight w = 2^(-days / H) x RHO^(seasons); A = all
  opponents, P = this one); removes any EB lag (form) without absorbing the pair itself.
* Pair (player, opponent franchise; F8c: player, opponent head coach), earlier games only:
  X = R_all x P_E (expected count of the pair given his current level), r = P_y / X, sampling
  noise var(r) = c_s x sum(w^2 Var_NB) / X^2 with Var_NB = Eb + phi Eb^2 (NB2, the state's phi)
  and c_s the placebo calibration of that variance.
* Empirical-Bayes shrinkage toward 0 (the BLUP of a random pair effect of variance tau2_s):
      theta = 1 + tau2 / (tau2 + var(r)) x (r - 1),   f8_<s> = log(max(theta, 0.25)).
  tau2_s and c_s are fit **walk-forward** (`fit_params` on the regular-season rows of seasons
  < S) by a placebo-corrected method of moments over (player, opponent, season) blocks: the
  observed excess of (Y - X)^2 over the NB variance, net of the same excess when the opponent
  labels are permuted within player-season (so a mis-specified phi cancels).
  The implied pseudo-count is k_s = c_s / tau2_s expected counts (`params["k"]`).

Columns (per player-game; strictly pre-tip: `common.ew_asof` excludes the game's own day)
  f8_<s>   for the 7 DK stats, f8_dk (DK-weighted composite)   log shrunk pair effect
  f8_n     log(1 + weighted games vs this opponent)              evidence
  F8r: f8r_<s>, f8r_dk, f8r_n  the same pair ratio with a fixed light pseudo-count
       (K_RAW_GAMES = 3 games of expected count): no EB fit, the layer decides how much to trust
       it (a check that the EB shrinkage did not throw away a usable signal)
  F8c: f8c_<s>, f8c_dk, f8c_n keyed on the opponent's head coach of that season
  (`team_coaches`; that table is a season snapshot, so a mid-season hire is known for the whole
  season: an identity lookahead, not an outcome one. Treat F8c as exploratory.)

Stats see their own f8_<s>, f8_dk and f8_n, with a +1 monotone constraint on their own column.
Neutral (league-average context) is 0 for every column. Players with no earlier game vs the
opponent get exactly 0 (theta = 1) and n = 0.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd

from .common import DK_STATS, DK_W, ew_asof

H, RHO = 540.0, 0.8              # pair memory: ~1.5-season half-life, 0.8 per season boundary
K0_GAMES = 10.0                  # player-level ratio prior, in games of expected count
TAU2_FLOOR = 1e-6
STATS8 = DK_STATS + ("dk",)
K_RAW_GAMES = 3.0               # F8r: fixed light pseudo-count (games of expected count)
COLS = tuple(f"f8_{s}" for s in STATS8) + ("f8_n",)
COLS_R = tuple(f"f8r_{s}" for s in STATS8) + ("f8r_n",)
COLS_C = tuple(f"f8c_{s}" for s in STATS8) + ("f8c_n",)
NEUTRAL = {c: 0.0 for c in COLS}
NEUTRAL_R = {c: 0.0 for c in COLS_R}
NEUTRAL_C = {c: 0.0 for c in COLS_C}


# ============================================================================== inputs

def _franchise(codes: pd.Series, seasons: pd.Series) -> np.ndarray:
    from ... import teams
    cache: dict = {}
    out = []
    for c, s in zip(codes.astype(str).to_numpy(), seasons.astype(str).to_numpy()):
        k = (c, s)
        if k not in cache:
            try:
                cache[k] = teams.franchise(c, s)
            except (KeyError, ValueError):
                cache[k] = c
        out.append(cache[k])
    return np.asarray(out, dtype=object)


def base_values(fr: pd.DataFrame) -> dict:
    """Per stat (7 DK + dk): y, Eb (E x F2 group index) and the NB variance, (n,) arrays."""
    out = {}
    ydk = np.zeros(len(fr))
    edk = np.zeros(len(fr))
    vdk = np.zeros(len(fr))
    for s in DK_STATS:
        E = np.maximum(fr[f"E_{s}"].to_numpy(float), 0.0)
        c = f"f2_dvp_{s}_grp"
        idx = np.exp(np.nan_to_num(fr[c].to_numpy(float))) if c in fr else 1.0
        Eb = E * idx
        y = fr[s].to_numpy(float)
        phi = fr[f"d_{s}"].to_numpy(float) if f"d_{s}" in fr else 0.0
        V = Eb + phi * Eb ** 2
        out[s] = (y, Eb, V)
        w = DK_W[s]
        ydk += w * y
        edk += w * Eb
        vdk += w * w * V
    out["dk"] = (ydk, edk, vdk)
    return out


def coach_keys(conn: sqlite3.Connection, fr: pd.DataFrame) -> np.ndarray:
    """Opponent head coach id per row ('' when unknown): team_coaches 'Head Coach' of the
    opponent's season (lowest sort_sequence)."""
    tc = pd.read_sql_query(
        "SELECT season, team, coach_id, sort_sequence FROM team_coaches "
        "WHERE coach_type = 'Head Coach'", conn)
    if not len(tc):
        return np.full(len(fr), "", dtype=object)
    tc = (tc.sort_values(["season", "team", "sort_sequence", "coach_id"])
          .drop_duplicates(["season", "team"]))
    m = fr[["season", "opp"]].merge(tc.rename(columns={"team": "opp"})[["season", "opp",
                                                                        "coach_id"]],
                                    on=["season", "opp"], how="left")
    cid = m["coach_id"]
    return np.where(cid.notna(), "c" + cid.fillna(0).astype(np.int64).astype(str), "")


def pair_keys(fr: pd.DataFrame, opp_key: np.ndarray) -> np.ndarray:
    pid = fr["player_id"].astype(np.int64).astype(str).to_numpy()
    ok = np.asarray([bool(k) for k in opp_key])
    return np.where(ok, pid.astype(object) + "|" + opp_key.astype(object), "")


# ============================================================================== walk-forward fit

def _blocks(fr: pd.DataFrame, opp_key: np.ndarray, vals: dict, rng=None) -> pd.DataFrame:
    """(player, opponent key, season) sums of y, Eb, V for every stat; with `rng` the opponent
    labels are permuted within player-season first (the placebo)."""
    d = pd.DataFrame({"pid": fr["player_id"].to_numpy(), "season": fr["season"].to_numpy(),
                      "ok": opp_key})
    if rng is not None:
        g = d.groupby(["pid", "season"], sort=False).indices
        ok = d["ok"].to_numpy().copy()
        for ix in g.values():
            ok[ix] = ok[rng.permutation(ix)]
        d["ok"] = ok
    for s, (y, Eb, V) in vals.items():
        d[f"y_{s}"], d[f"e_{s}"], d[f"v_{s}"] = y, Eb, V
    d = d[d["ok"] != ""]
    cols = [c for c in d.columns if c[:2] in ("y_", "e_", "v_")]
    b = d.groupby(["pid", "season", "ok"], sort=False)[cols].sum().reset_index()
    ps = d.groupby(["pid", "season"], sort=False)[cols].sum()
    tot = ps.reindex(pd.MultiIndex.from_frame(b[["pid", "season"]])).to_numpy()
    rest = pd.DataFrame(tot - b[cols].to_numpy(), columns=cols)
    for s in vals:
        # the player's level that season from his OTHER opponents (leave-pair-out)
        R = ((rest[f"y_{s}"] + 1.0) / (rest[f"e_{s}"] + 1.0)).to_numpy()
        b[f"R_{s}"] = np.clip(R, 0.2, 5.0)
    return b


def _moments(b: pd.DataFrame, s: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per block: (Y - X)^2, V (NB variance at the player's level), X^2."""
    R = b[f"R_{s}"].to_numpy()
    X = R * b[f"e_{s}"].to_numpy()
    E = b[f"e_{s}"].to_numpy()
    phiE2 = b[f"v_{s}"].to_numpy() - E            # sum phi Eb^2
    V = R * E + R * R * phiE2
    return (b[f"y_{s}"].to_numpy() - X) ** 2, V, X ** 2


def variance_components(fr: pd.DataFrame, opp_key: np.ndarray, *, seed: int = 11) -> dict:
    """Placebo-corrected method-of-moments variance of the player x opponent effect per stat on
    the rows of `fr` (regular season): {s: {tau2, se, c, k, k_games, sd_effect, n_blocks}}.
    c = sum (Y-X)^2 / sum V on the placebo blocks (NB variance calibration);
    tau2 = (sum (Y-X)^2 - c sum V) / sum X^2 on the real blocks (SE clustered by player)."""
    vals = base_values(fr)
    real = _blocks(fr, opp_key, vals)
    plac = _blocks(fr, opp_key, vals, rng=np.random.default_rng(seed))
    out = {}
    for s in vals:
        d2p, Vp, _ = _moments(plac, s)
        c = float(d2p.sum() / max(Vp.sum(), 1e-12))
        d2, V, X2 = _moments(real, s)
        num = d2 - c * V
        tau2 = float(num.sum() / max(X2.sum(), 1e-12))
        per = pd.Series(num).groupby(real["pid"].to_numpy()).sum().to_numpy()
        G = len(per)
        se = float(math.sqrt(G / max(G - 1, 1) * np.sum((per - per.mean()) ** 2))
                   / max(X2.sum(), 1e-12))
        ebar = float(vals[s][1].mean()) if len(fr) else 1.0
        t = max(tau2, TAU2_FLOOR)
        out[s] = {"tau2": tau2, "se": se, "z": tau2 / se if se > 0 else float("nan"),
                  "c": c, "k": c / t, "k_games": c / t / max(ebar, 1e-9),
                  "sd_effect": math.sqrt(max(tau2, 0.0)), "n_blocks": int(len(real)),
                  "ebar": ebar}
    return out


def fit_params(fr: pd.DataFrame, train_mask: np.ndarray, *, coach: np.ndarray | None = None,
               opp_key: np.ndarray | None = None) -> dict:
    """tau2_s, c_s, ebar_s on the regular-season rows of `train_mask` (seasons < S)."""
    m = np.asarray(train_mask, bool) & (fr["season_type"] == "Regular Season").to_numpy()
    if opp_key is None:
        opp_key = _franchise(fr["opp"], fr["season"]) if coach is None else coach
    vc = variance_components(fr[m].reset_index(drop=True), np.asarray(opp_key)[m])
    return {s: {"tau2": max(v["tau2"], TAU2_FLOOR), "c": v["c"], "ebar": v["ebar"],
                "k": v["k"], "raw_tau2": v["tau2"], "se": v["se"]} for s, v in vc.items()}


# ============================================================================== features

def features(fr: pd.DataFrame, opp_key: np.ndarray, params: dict, *, prefix: str = "f8"
             ) -> pd.DataFrame:
    """F8 columns for every row of `fr` from the earlier rows of `fr` (history = query set)."""
    vals = base_values(fr)
    n = len(fr)
    pid = fr["player_id"].to_numpy()
    day, sidx = fr["day"].to_numpy(), fr["sidx"].to_numpy()
    pk = pair_keys(fr, np.asarray(opp_key))
    has = pk != ""
    out = {}
    stats = list(vals)
    Yh = np.column_stack([vals[s][0] for s in stats] + [vals[s][1] for s in stats])
    A = ew_asof(pid, day, sidx, Yh, pid, day, sidx, half_life=H, rho=RHO)
    hv = np.column_stack([vals[s][0] for s in stats] + [vals[s][1] for s in stats]
                         + [np.ones(n)])
    pkh = np.where(has, pk, "__none__")
    P = np.zeros((n, hv.shape[1]))
    P2 = np.zeros((n, len(stats)))
    if has.any():
        P[has] = ew_asof(pkh[has], day[has], sidx[has], hv[has], pkh[has], day[has], sidx[has],
                         half_life=H, rho=RHO)
        vv = np.column_stack([vals[s][2] for s in stats])
        P2[has] = ew_asof(pkh[has], day[has], sidx[has], vv[has], pkh[has], day[has],
                          sidx[has], half_life=H / 2.0, rho=RHO ** 2)
    k = len(stats)
    for j, s in enumerate(stats):
        p = params.get(s, {})
        tau2 = float(p.get("tau2", TAU2_FLOOR))
        c = float(p.get("c", 1.0))
        k0 = K0_GAMES * float(p.get("ebar", 1.0))
        # his level from his OTHER opponents (leave-pair-out, as in `variance_components`)
        R = np.clip((A[:, j] - P[:, j] + k0) / (np.maximum(A[:, k + j] - P[:, k + j], 0.0) + k0),
                    0.2, 5.0)
        X = R * P[:, k + j]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(X > 1e-9, P[:, j] / np.maximum(X, 1e-9), 1.0)
            noise = np.where(X > 1e-9, c * R * P2[:, j] / np.maximum(X, 1e-9) ** 2, np.inf)
        w = np.where(np.isfinite(noise), tau2 / (tau2 + noise), 0.0)
        theta = 1.0 + w * (r - 1.0)
        out[f"{prefix}_{s}"] = np.log(np.maximum(theta, 0.25))
        # light fixed shrinkage (the GBM decides how much to trust it): K_RAW_GAMES games
        kr = K_RAW_GAMES * float(p.get("ebar", 1.0)) * R
        out[f"{prefix}r_{s}"] = np.log(np.maximum((P[:, j] + kr) / (X + kr), 0.25))
    out[f"{prefix}_n"] = np.log1p(np.maximum(P[:, -1], 0.0))
    out[f"{prefix}r_n"] = out[f"{prefix}_n"]
    return pd.DataFrame(out, index=fr.index)


def build(conn: sqlite3.Connection, frame: pd.DataFrame, *, season: str = "",
          coach: bool = False) -> pd.DataFrame:
    """F8 (or F8c) table for the rows of `frame` (game_id, player_id + columns). Params are fit
    on the frame's regular-season rows of seasons < `season` (default: the frame's last)."""
    fr = frame.reset_index(drop=True)
    season = season or str(fr["season"].max())
    key = coach_keys(conn, fr) if coach else _franchise(fr["opp"], fr["season"])
    train = (fr["season"] < season).to_numpy()
    prm = fit_params(fr, train, opp_key=key) if train.any() else {}
    f = features(fr, key, prm, prefix="f8c" if coach else "f8")
    f.insert(0, "game_id", fr["game_id"].to_numpy())
    f.insert(1, "player_id", fr["player_id"].to_numpy())
    f.attrs["params"] = prm
    return f


# ============================================================================== reliability

def split_half(fr: pd.DataFrame, opp_key: np.ndarray, *, min_games: int = 6,
               seed: int = 5) -> dict:
    """Split-half reliability of the pair residual on the rows of `fr` (regular season, one
    block = (player, opponent key) over every season of `fr`): games of each pair alternate into
    halves A / B by date; r_h = Y_h / X_h - 1 with X_h = the player's leave-pair-out level x Eb_h.
    Returns per stat the precision-weighted (1 / (1/X_A + 1/X_B)) correlation of r_A and r_B
    across pairs with X_A, X_B >= 1 (and Spearman-Brown 2r / (1 + r)), with the same numbers for
    placebo pairs (labels permuted within player: the shared-level artefact)."""
    vals = base_values(fr)

    def one(key):
        d = pd.DataFrame({"pid": fr["player_id"].to_numpy(), "ok": key,
                          "day": fr["day"].to_numpy()})
        for s, (y, Eb, _) in vals.items():
            d[f"y_{s}"], d[f"e_{s}"] = y, Eb
        d = d[d["ok"] != ""].sort_values(["pid", "ok", "day"], kind="mergesort")
        d["k"] = d.groupby(["pid", "ok"]).cumcount()
        d["n"] = d.groupby(["pid", "ok"])["k"].transform("size")
        cols = [c for c in d.columns if c[:2] in ("y_", "e_")]
        tot = d.groupby("pid")[cols].sum()
        d = d[d["n"] >= min_games]
        pair = d.groupby(["pid", "ok"])[cols].sum()
        rest = tot.reindex(pair.index.get_level_values(0)).to_numpy() - pair.to_numpy()
        rest = pd.DataFrame(rest, index=pair.index, columns=cols)
        d["h"] = d["k"] % 2
        hs = d.groupby(["pid", "ok", "h"])[cols].sum().unstack("h")
        res = {}
        for s in vals:
            R = ((rest[f"y_{s}"] + 1.0) / (rest[f"e_{s}"] + 1.0)).clip(0.2, 5.0)
            XA = (R * hs[(f"e_{s}", 0)]).to_numpy(float)
            XB = (R * hs[(f"e_{s}", 1)]).to_numpy(float)
            rA = hs[(f"y_{s}", 0)].to_numpy(float) / np.maximum(XA, 1e-9) - 1.0
            rB = hs[(f"y_{s}", 1)].to_numpy(float) / np.maximum(XB, 1e-9) - 1.0
            # precision weights (Poisson: var r_h ~ 1 / X_h); halves with < 1 expected count out
            ok = np.isfinite(rA) & np.isfinite(rB) & (XA >= 1.0) & (XB >= 1.0)
            w = 1.0 / (1.0 / np.maximum(XA, 1e-9) + 1.0 / np.maximum(XB, 1e-9))
            r = _wcorr(rA[ok], rB[ok], w[ok]) if ok.sum() > 10 else float("nan")
            res[s] = {"r": r, "sb": 2 * r / (1 + r) if r > -1 else float("nan"),
                      "n_pairs": int(ok.sum())}
        return res

    key = np.asarray(opp_key)
    real = one(key)
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"pid": fr["player_id"].to_numpy(), "ok": key})
    pk = key.copy()
    for ix in d.groupby("pid", sort=False).indices.values():
        pk[ix] = pk[rng.permutation(ix)]
    plac = one(pk)
    return {s: {"real": real[s], "placebo": plac[s]} for s in real}


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    ma, mb = float((w * a).sum()), float((w * b).sum())
    cov = float((w * (a - ma) * (b - mb)).sum())
    va, vb = float((w * (a - ma) ** 2).sum()), float((w * (b - mb) ** 2).sum())
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


# ============================================================================== registry

def columns(stat: str) -> list[str]:
    return [f"f8_{stat}", "f8_dk", "f8_n"]


def monotone(stat: str) -> dict:
    return {f"f8_{stat}": 1}


def columns_r(stat: str) -> list[str]:
    return [f"f8r_{stat}", "f8r_dk", "f8r_n"]


def monotone_r(stat: str) -> dict:
    return {f"f8r_{stat}": 1}


def columns_c(stat: str) -> list[str]:
    return [f"f8c_{stat}", "f8c_dk", "f8c_n"]


def monotone_c(stat: str) -> dict:
    return {f"f8c_{stat}": 1}
