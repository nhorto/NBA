"""Similar players: the player-seasons since 2000-01 whose statistical profile is closest to a
player's season (the "Similar players" card on /player/<id>; docs/STATS_SITE.md).

The method is deliberately transparent:

1. Profile of every regular-season player-season (`season_profiles`, one cached pass per
   season): per-36 points, rebounds, assists, steals, blocks and turnovers; 3PA rate
   (3PA / FGA); FTA rate (FTA / FGA); TS%; minute-weighted USG%, AST% and REB%
   (`player_games_adv`); height (`player_index.height_in`) and age on February 1 of the season
   (`team_rosters.birth_date`; players without one get 22.5 + seasons since their first).
2. The comparison pool: player-seasons with at least `POOL_MIN` minutes. Each feature is
   z-scored within its own season (mean and spread of that season's pool), so eras with a
   different pace or 3-point volume compare fairly. A missing value counts as average (z = 0).
3. Distance: weighted Euclidean distance between z-vectors, normalised by the weight total
   (`WEIGHTS`), i.e. the weighted root-mean-square z gap.
4. Score: 100 * (1 - distance / `SCALE`), floored at 0. 100 = identical profile; two random
   pool players typically score about 35 (median random-pair distance ~1.3 on the 2026 data).
5. The player himself is excluded, and each other player appears once (his closest season).

`season_matrix(conn)` (all seasons, numpy arrays) and every `similar(conn, pid, season)` result
are cached with `stats.db.cached`, so a warm card costs a dict lookup.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Sequence

import numpy as np

from . import db as sdb
from . import seasons as S

FIRST_SEASON = "2000-01"
POOL_MIN = 1000.0              # minutes for a player-season to enter the comparison pool
TARGET_MIN = 150.0             # minutes before a season gets a similar-players list at all
THIN_POOL = 120                # a season pool smaller than this borrows the previous season's z
TOP_N = 8
SCALE = 2.0                    # distance that maps to a score of 0 (~90th pct of random pairs)
ROOKIE_AGE = 22.5              # mean age in a player's first season (measured on team_rosters)

# feature key -> (label, weight). Scoring volume, playmaking and size carry the most weight.
WEIGHTS: dict[str, tuple[str, float]] = {
    "pts36": ("Pts/36", 1.0), "reb36": ("Reb/36", 0.8), "ast36": ("Ast/36", 1.0),
    "stl36": ("Stl/36", 0.5), "blk36": ("Blk/36", 0.6), "tov36": ("TOV/36", 0.5),
    "par3": ("3PA rate", 0.8), "ftr": ("FTA rate", 0.5), "ts": ("TS%", 0.7),
    "usg": ("USG%", 1.0), "astp": ("AST%", 0.8), "rebp": ("REB%", 0.8),
    "height": ("Height", 0.9), "age": ("Age", 0.6),
}
FEATURES = tuple(WEIGHTS)
_W = np.array([w for _, w in WEIGHTS.values()], dtype=float)

_SQL = """
SELECT pg.player_id, pg.team, COUNT(*) AS gp, SUM(pg.min) AS min, SUM(pg.pts) AS pts,
       SUM(pg.reb) AS reb, SUM(pg.ast) AS ast, SUM(pg.stl) AS stl, SUM(pg.blk) AS blk,
       SUM(pg.tov) AS tov, SUM(pg.fgm) AS fgm, SUM(pg.fga) AS fga, SUM(pg.fg3m) AS fg3m,
       SUM(pg.fg3a) AS fg3a, SUM(pg.ftm) AS ftm, SUM(pg.fta) AS fta,
       SUM(a.usg_pct * pg.min) AS usg_m, SUM(a.ast_pct * pg.min) AS astp_m,
       SUM(a.reb_pct * pg.min) AS rebp_m,
       SUM(CASE WHEN a.usg_pct IS NOT NULL THEN pg.min END) AS adv_min,
       MAX(pg.game_date) AS last, MAX(pg.player_name) AS bname
FROM player_games pg
LEFT JOIN player_games_adv a ON a.game_id=pg.game_id AND a.player_id=pg.player_id
WHERE pg.season=? AND pg.season_type=? AND pg.min > 0
GROUP BY pg.player_id, pg.team
"""
_SUMS = ("gp", "min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a",
         "ftm", "fta", "usg_m", "astp_m", "rebp_m", "adv_min")


def _div(a, b):
    return (a / b) if (a is not None and b) else None


# ------------------------------------------------------------------ bio
def bio(conn: sqlite3.Connection) -> dict[int, dict]:
    """{player_id: {name, height_in, from_year, birth}} (cached)."""
    def build(c):
        out: dict[int, dict] = {}
        try:
            for r in c.execute("SELECT player_id, first_name, last_name, height_in, from_year "
                               "FROM player_index"):
                nm = " ".join(x for x in ((r[1] or "").strip(), (r[2] or "").strip()) if x)
                out[int(r[0])] = {"name": nm or None, "height_in": r[3], "from_year": r[4],
                                  "birth": None}
        except sqlite3.Error:
            pass
        try:
            for pid, bd in c.execute("SELECT player_id, MIN(birth_date) FROM team_rosters "
                                     "WHERE birth_date IS NOT NULL GROUP BY player_id"):
                out.setdefault(int(pid), {"name": None, "height_in": None, "from_year": None,
                                          "birth": None})["birth"] = bd
        except sqlite3.Error:
            pass
        return out
    return sdb.cached(("sim-bio",), build, conn, domain="box") or {}


def age_in(season: str, b: dict | None) -> tuple[float | None, bool]:
    """(age on Feb 1 of the season, estimated?) from a bio dict."""
    y = S.start_year(season)
    if b and b.get("birth"):
        try:
            bd = date.fromisoformat(str(b["birth"])[:10])
            return (date(y + 1, 2, 1) - bd).days / 365.25, False
        except ValueError:
            pass
    if b and b.get("from_year"):
        return ROOKIE_AGE + (y - int(b["from_year"])), True
    return None, True


# ------------------------------------------------------------------ per-season profiles
def compute_profiles(conn: sqlite3.Connection, season: str, season_type: str) -> dict[int, dict]:
    acc: dict[int, dict] = {}
    try:
        rows = conn.execute(_SQL, (season, season_type)).fetchall()
    except sqlite3.Error:
        return {}
    for r in rows:
        r = dict(zip(("player_id", "team") + _SUMS + ("last", "bname"), tuple(r)))
        pid = int(r["player_id"])
        a = acc.get(pid)
        if a is None:
            a = acc[pid] = {k: 0.0 for k in _SUMS} | {"teams": {}, "last": "", "bname": None}
        for k in _SUMS:
            a[k] += float(r[k] or 0)
        a["teams"][r["team"]] = a["teams"].get(r["team"], 0.0) + float(r["min"] or 0)
        if (r["last"] or "") >= a["last"]:
            a["last"], a["bname"] = r["last"] or "", r["bname"] or a["bname"]
    bios = bio(conn)
    out = {}
    for pid, a in acc.items():
        m = a["min"]
        p36 = (36.0 / m) if m else 0.0
        b = bios.get(pid)
        age, est = age_in(season, b)
        teams = sorted(a["teams"], key=lambda t: -a["teams"][t])
        tsa = 2.0 * (a["fga"] + 0.44 * a["fta"])
        out[pid] = {
            "player_id": pid, "season": season, "name": (b or {}).get("name") or a["bname"],
            "team": teams[0] if teams else None, "teams": teams, "gp": int(a["gp"]), "min": m,
            "mpg": _div(m, a["gp"]),
            "pts": _div(a["pts"], a["gp"]), "reb": _div(a["reb"], a["gp"]),
            "ast": _div(a["ast"], a["gp"]),
            "pts36": a["pts"] * p36, "reb36": a["reb"] * p36, "ast36": a["ast"] * p36,
            "stl36": a["stl"] * p36, "blk36": a["blk"] * p36, "tov36": a["tov"] * p36,
            "par3": _div(a["fg3a"], a["fga"]), "ftr": _div(a["fta"], a["fga"]),
            "ts": (a["pts"] / tsa) if tsa else None,
            "usg": _div(a["usg_m"], a["adv_min"]), "astp": _div(a["astp_m"], a["adv_min"]),
            "rebp": _div(a["rebp_m"], a["adv_min"]),
            "height": (b or {}).get("height_in"), "age": age, "age_est": est,
        }
    return out


def season_profiles(conn: sqlite3.Connection, season: str,
                    season_type: str = S.REGULAR) -> dict[int, dict]:
    """{player_id: profile} of every player with a game (min > 0) in season / type (cached).
    Profile keys: name team teams gp min mpg pts reb ast (per game), the FEATURES, age_est."""
    return sdb.cached(("sim-prof", season, season_type),
                      lambda c: compute_profiles(c, season, season_type), conn, domain="box") or {}


# ------------------------------------------------------------------ the matrix
def all_seasons(today: date | None = None) -> list[str]:
    first, last = S.start_year(FIRST_SEASON), S.start_year(S.current_season(today))
    return [S.from_start_year(y) for y in range(first, last + 1)]


def _vec(p: dict) -> list[float]:
    return [float("nan") if p.get(k) is None else float(p[k]) for k in FEATURES]


def compute_matrix(conn: sqlite3.Connection, seasons: Sequence[str]) -> dict:
    zs, meta, stats = [], [], {}
    prev = None
    for s in seasons:
        prof = season_profiles(conn, s)
        pool = [p for p in prof.values() if p["min"] >= POOL_MIN]
        if len(pool) >= THIN_POOL or prev is None:
            if pool:
                x = np.array([_vec(p) for p in pool], dtype=float)
                with np.errstate(invalid="ignore"):
                    mu = np.nanmean(x, axis=0)
                    sd = np.nanstd(x, axis=0)
                sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
                mu = np.where(np.isfinite(mu), mu, 0.0)
                stats[s] = (mu, sd, len(pool), s)
                prev = stats[s]
        else:
            stats[s] = (prev[0], prev[1], len(pool), prev[3])   # thin season: borrow z-scale
        if s not in stats:
            continue
        mu, sd = stats[s][0], stats[s][1]
        for p in pool:
            zs.append(np.nan_to_num((np.array(_vec(p)) - mu) / sd, nan=0.0))
            meta.append({k: p[k] for k in ("player_id", "season", "name", "team", "gp", "min",
                                           "mpg", "pts", "reb", "ast", "pts36", "reb36", "ast36",
                                           "par3", "ts", "usg", "age", "age_est")})
    z = np.vstack(zs) if zs else np.zeros((0, len(FEATURES)))
    return {"z": z, "meta": meta, "pids": np.array([m["player_id"] for m in meta], dtype=np.int64),
            "stats": stats, "seasons": list(seasons)}


def season_matrix(conn: sqlite3.Connection, today: date | None = None) -> dict:
    """{z: N x F z-scores of the pool, meta: [row dicts], pids, stats: {season: (mean, sd,
    pool size, scale season)}} over every season since 2000-01 (cached)."""
    seasons = tuple(all_seasons(today))
    return sdb.cached(("sim-matrix", seasons), lambda c: compute_matrix(c, seasons), conn,
                      domain="box")


def distances(z: np.ndarray, target: np.ndarray, weights: np.ndarray = _W) -> np.ndarray:
    """Weighted RMS z-gap of each row of `z` to `target`."""
    if not len(z):
        return np.zeros(0)
    return np.sqrt(((z - target) ** 2 * weights).sum(axis=1) / weights.sum())


def score(d: float) -> float:
    return max(0.0, 100.0 * (1.0 - d / SCALE))


def rank_similar(z: np.ndarray, meta: list[dict], target: np.ndarray, exclude: int,
                 n: int = TOP_N, one_per_player: bool = True) -> list[dict]:
    """The `n` closest rows (excluding player `exclude`), closest first, with distance + score."""
    d = distances(z, target)
    out, seen = [], set()
    for i in np.argsort(d, kind="stable"):
        m = meta[int(i)]
        pid = m["player_id"]
        if pid == exclude or (one_per_player and pid in seen):
            continue
        seen.add(pid)
        out.append({**m, "distance": float(d[i]), "score": score(float(d[i]))})
        if len(out) >= n:
            break
    return out


def _gaps(target_z: np.ndarray, row_z: np.ndarray) -> list[str]:
    """Labels of the three features where the two profiles differ the most (weighted)."""
    g = np.abs(target_z - row_z) * np.sqrt(_W)
    return [WEIGHTS[FEATURES[i]][0] for i in np.argsort(-g)[:3] if g[i] > 0.35]


def similar(conn: sqlite3.Connection, pid: int, season: str, today: date | None = None,
            n: int = TOP_N) -> dict:
    """{status: ok|thin|none, season, target: profile, rows: [...], method, min_pool}.

    status 'none': no regular-season minutes that season; 'thin': fewer than TARGET_MIN."""
    def build(c):
        prof = season_profiles(c, season).get(pid)
        base = {"season": season, "rows": [], "target": prof, "pool_min": POOL_MIN,
                "target_min": TARGET_MIN, "method": METHOD}
        if prof is None:
            return {**base, "status": "none"}
        if prof["min"] < TARGET_MIN:
            return {**base, "status": "thin"}
        mx = season_matrix(c, today)
        st = mx["stats"].get(season)
        if st is None and mx["stats"]:
            st = mx["stats"][max(mx["stats"])]
        if st is None or not len(mx["z"]):
            return {**base, "status": "none"}
        tz = np.nan_to_num((np.array(_vec(prof)) - st[0]) / st[1], nan=0.0)
        rows = rank_similar(mx["z"], mx["meta"], tz, pid, n)
        idx = {(m["player_id"], m["season"]): i for i, m in enumerate(mx["meta"])}
        for r in rows:
            i = idx.get((r["player_id"], r["season"]))
            r["gaps"] = _gaps(tz, mx["z"][i]) if i is not None else []
        return {**base, "status": "ok", "rows": rows, "pool": len(mx["meta"]),
                "seasons": f"{mx['seasons'][0]} to {mx['seasons'][-1]}"}
    return sdb.cached(("sim", int(pid), season), build, conn, domain="box")


METHOD = (
    "How similarity works:\n"
    "1. Each player-season since 2000-01 with " + f"{POOL_MIN:,.0f}" + "+ regular-season minutes "
    "is described by " + str(len(FEATURES)) + " numbers: points, rebounds, assists, steals, "
    "blocks and turnovers per 36 minutes; 3PA rate; FTA rate; TS%; usage, assist and rebound "
    "percentage; height; age.\n"
    "2. Each number becomes a z-score within its own season, so eras compare fairly.\n"
    "3. Distance = weighted Euclidean distance (scoring, assists and usage weigh 1.0; height 0.9; "
    "rebounds, AST%, REB% and 3PA rate 0.8; TS% 0.7; age and blocks 0.6; the rest 0.5).\n"
    "4. Score = 100 × (1 − distance / " + f"{SCALE:g}" + "): 100 is an identical profile; two "
    "random rotation players typically score about 35.\n"
    "Each player appears once (his closest season); he himself is left out."
)


def feature_rows() -> list[dict]:
    return [{"key": k, "label": lab, "weight": w} for k, (lab, w) in WEIGHTS.items()]
