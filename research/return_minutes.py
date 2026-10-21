"""Return-from-absence minutes / DK error study (task #3, research only: measure before building).

How wrong are the live model's minutes and DK projections in a player's first games back from an
absence (injury ramp-up, "minutes restriction"), and is a fix worth building?

No new projections are run: it reads the per-date player rows the walk-forward harness already
wrote on the Mac scratch box (`proj/backtest.py` DateCache, one parquet per date), copied to
var/tmp/ret3 by `pull` (a few tens of MB). Which cache is which arm (from the drivers' logs):

  2025-26 slice (104 dates 2025-12-22 .. 2026-04-12, T2.4 setup, 4,000 sims, closing lines):
    live+p2   p2/out/players_pathA-v0.3+p2_live.parquet    = cache p2/.../live/2025-26/v1-1888940222
    live10+p2 p2/out/players_pathA-v0.3+p2_live10.parquet  = cache p2/.../live10/2025-26/v1-bab3326db9
    live      p2/out/players_pathA-v0.3_live.parquet        (v0.3, p2 off; v1-3cd421b06b)
    avail     t24/out/players_pathA-v0.3_avail.parquet      (v0.3 oracle availability; v1-47d4067b9f)
  earlier seasons, avail protocol, context ON (= pathA-v0.3; bundle fit on seasons < S):
    cx/data/reports/backtest/cache/avail/2019-20/v1-16c46b8db0   (RotoGuru DK pool rows)
    cx/.../avail/2020-21/v1-c19b1a7e03                          (RotoGuru DK pool rows)
    cx/.../avail/2023-24/v1-df4e9671e4                          (every simulated player)
    cx/.../avail/2024-25/v1-b2065d96b8                          (every simulated player)
  (the "off" twins are the other hash in each cx dir; var-t21 = the T2.1 full run on pathA-v0.1,
  minutes v0, not used.)

Return state (from box scores strictly before the game, regular season, local DB read-only):
  missed_now   team games of his current team since his last appearance (season start counts
               when he played the season before; first-ever seasons are excluded)
  gsr          games played since the end of the latest absence of >= 3 team games
               (0 = first game back), absence length = that run of missed team games
  pre_min      mean minutes of his last 10 appearances before that absence (normal rows: before
               the game); tiers star >= 30, starter 24-30, rotation 15-24, bench < 15
  normal       no >= 3-game absence earlier this season, or >= 16 games since it
2025-26: the last official-report row before the return game's tip (since the absence began),
classified into soft tissue / knee-ankle-foot / illness / rest-management / reconditioning /
G League / other.

Errors: minutes = actual - E[min | plays] (= exp_min / p_play) on played rows; DK = actual -
E[DK | plays] (= mean / p_play) on played rows; SEs clustered by date. Fix: a post-fit
multiplier on E[min | plays] by games-since-return x absence length, fit walk-forward on the avail
rows of earlier seasons and applied to S (DK quantiles scaled by the DK ratio of the same cell).

    python research/return_minutes.py pull            # rsync the caches from the Mac (small)
    python research/return_minutes.py run [--out var/tmp/ret3/out]
    python research/return_minutes.py news            # grep stored text for restriction mentions
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sqlite3
import subprocess
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import injuries as INJ

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "var" / "tmp" / "ret3"
DB = ROOT / "var" / "courtside.db"
MAC = "nicholashorton@mac-mini"
SCR = "Developer/courtside-compute"

RAW = {   # arm -> (local file, Mac source)
    "live+p2": ("raw/players_pathA-v0.3+p2_live.parquet", "p2/out/players_pathA-v0.3+p2_live.parquet"),
    "live10+p2": ("raw/players_pathA-v0.3+p2_live10.parquet",
                  "p2/out/players_pathA-v0.3+p2_live10.parquet"),
    "live": ("raw/players_pathA-v0.3_live.parquet", "p2/out/players_pathA-v0.3_live.parquet"),
    "avail-2526": ("raw/players_pathA-v0.3_avail.parquet", "t24/out/players_pathA-v0.3_avail.parquet"),
}
CX = {    # season -> cache dir of the context-ON avail arm
    "2019-20": "v1-16c46b8db0", "2020-21": "v1-c19b1a7e03",
    "2023-24": "v1-df4e9671e4", "2024-25": "v1-b2065d96b8",
}
AVAIL_SEASONS = ("2019-20", "2020-21", "2023-24", "2024-25", "2025-26")

GSR_BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 15, "6-15"))
ABS_BUCKETS = ((3, 5, "3-5"), (6, 15, "6-15"), (16, 10_000, "16+"))
TIERS = ((30, 99, "star"), (24, 30, "starter"), (15, 24, "rotation"), (-1, 15, "bench"))
NORMAL_GSR = 16
P_MIN = 0.05            # E[.|plays] = mean / p_play only where p_play >= this


# ====================================================================== pull

def pull() -> None:
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    for _arm, (loc, src) in RAW.items():
        subprocess.run(["scp", "-q", f"{MAC}:{SCR}/{src}", str(WORK / loc)], check=True)
    for s, key in CX.items():
        d = WORK / "cx" / s
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--include=*.players.parquet", "--exclude=*",
                        f"{MAC}:{SCR}/cx/data/reports/backtest/cache/avail/{s}/{key}/", f"{d}/"],
                       check=True)
    print("pulled into", WORK)


# ====================================================================== load

def load_arms() -> dict[str, pd.DataFrame]:
    out = {}
    for arm, (loc, _src) in RAW.items():
        out[arm] = pd.read_parquet(WORK / loc)
    for s in CX:
        fs = sorted(glob.glob(str(WORK / "cx" / s / "*.players.parquet")))
        out[f"avail-{s[2:4]}{s[5:7]}"] = pd.concat([pd.read_parquet(f) for f in fs],
                                                   ignore_index=True)
    for arm, d in list(out.items()):
        d = out[arm] = d[d["player_id"].notna()].reset_index(drop=True)   # unmapped RotoGuru ids
        d["arm"] = arm
        d["player_id"] = d["player_id"].astype("int64")
        d["game_id"] = d["game_id"].astype(str)
    return out


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def load_box(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    pg = pd.read_sql_query(
        "SELECT game_id, player_id, player_name, team, season, game_date, min FROM player_games "
        "WHERE season_type='Regular Season' AND season >= '2018-19' AND min > 0", conn)
    tg = pd.read_sql_query(
        "SELECT game_id, team, season, game_date FROM team_games "
        "WHERE season_type='Regular Season' AND season >= '2018-19'", conn)
    return pg, tg


def prev_season(s: str) -> str:
    a = int(s[:4]) - 1
    return f"{a}-{str(a + 1)[2:]}"


# ====================================================================== return state

class BoxState:
    """Return state of (player, team, date) from regular-season box appearances strictly before."""

    def __init__(self, pg: pd.DataFrame, tg: pd.DataFrame):
        tg = tg.sort_values(["game_date", "game_id"])
        self.tdates = {(s, t): g["game_date"].to_numpy(str)
                       for (s, t), g in tg.groupby(["season", "team"])}
        pg = pg.sort_values(["game_date", "game_id"])
        self.app = {}
        for pid, g in pg.groupby("player_id", sort=False):
            self.app[int(pid)] = (g["game_date"].to_numpy(str), g["team"].to_numpy(str),
                                  g["season"].to_numpy(str), g["min"].to_numpy(float))
        self._missed: dict[int, np.ndarray] = {}

    def _n_team(self, season, team, lo, hi) -> int:
        """Team games of `team` in `season` with lo < date < hi (lo None = season start)."""
        d = self.tdates.get((season, team))
        if d is None:
            return 0
        a = 0 if lo is None else bisect_right(d, lo)
        return max(0, bisect_left(d, hi) - a)

    def missed_before(self, pid) -> np.ndarray:
        """Per appearance: team games missed since his previous appearance in the same season
        (first appearance of a season: team games before it, NaN if he has no appearance in the
        season before)."""
        if pid in self._missed:
            return self._missed[pid]
        dates, teams, seas, _m = self.app[pid]
        out = np.empty(len(dates))
        for i in range(len(dates)):
            if i > 0 and seas[i - 1] == seas[i]:
                out[i] = self._n_team(seas[i], teams[i], dates[i - 1], dates[i])
            elif i > 0 and seas[i - 1] == prev_season(seas[i]):
                out[i] = self._n_team(seas[i], teams[i], None, dates[i])
            elif i > 0:                                      # missed whole season(s): long
                out[i] = self._n_team(seas[i], teams[i], None, dates[i]) + 82
            else:
                out[i] = np.nan
        self._missed[pid] = out
        return out

    def state(self, pid: int, team: str, season: str, day: str) -> dict:
        rec = self.app.get(pid)
        nan = dict(missed_now=np.nan, gsr=np.nan, abs_len=np.nan, pre_min=np.nan,
                   team_change=np.nan, first_season=1.0)
        if rec is None:
            return nan
        dates, teams, seas, mins = rec
        j = bisect_left(dates, day)                          # appearances strictly before
        jj = j
        while jj > 0 and seas[jj - 1] == season:
            jj -= 1                                          # jj = first appearance of season
        if j > jj:                                           # appeared this season already
            missed_now = self._n_team(season, team, dates[j - 1], day)
            team_change = float(teams[j - 1] != team)
        elif jj > 0 and seas[jj - 1] == prev_season(season):
            missed_now = self._n_team(season, team, None, day)
            team_change = float(teams[jj - 1] != team)
        elif jj > 0:                                         # back after a whole lost season
            missed_now = self._n_team(season, team, None, day) + 82
            team_change = float(teams[jj - 1] != team)
        else:
            return nan
        mb = self.missed_before(pid)
        if missed_now >= 3:
            gsr, abs_len, start = 0, missed_now, j
        else:
            idx = [i for i in range(jj, j) if mb[i] >= 3]
            if idx:
                i = idx[-1]
                gsr, abs_len, start = j - i, mb[i], i
            else:
                gsr, abs_len, start = np.nan, 0.0, j
        lo = max(0, start - 10)
        pre = float(mins[lo:start].mean()) if start > 0 else np.nan
        return dict(missed_now=float(missed_now), gsr=float(gsr), abs_len=float(abs_len),
                    pre_min=pre, team_change=team_change, first_season=0.0,
                    ret_date=(dates[start] if (gsr == gsr and gsr > 0) else day))


def attach_state(df: pd.DataFrame, bs: BoxState) -> pd.DataFrame:
    rows = [bs.state(int(p), t, s, d) for p, t, s, d in
            zip(df["player_id"], df["team"], df["season"], df["date"])]
    st = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, st], axis=1)
    out["gsr_b"] = bucket(out["gsr"], GSR_BUCKETS, normal=True)
    out["abs_b"] = bucket(out["abs_len"], ABS_BUCKETS)
    out["tier_pre"] = bucket(out["pre_min"], TIERS, closed_hi=False)
    out.loc[out["missed_now"].between(1, 2) & out["gsr"].isna(), "gsr_b"] = "short"
    return out


def bucket(x: pd.Series, spec, normal=False, closed_hi=True) -> pd.Series:
    v = x.to_numpy(float)
    out = np.full(len(v), "", dtype=object)
    for lo, hi, name in spec:
        m = (v >= lo) & ((v <= hi) if closed_hi else (v < hi))
        out[m] = name
    if normal:
        out[np.isnan(v) | (v >= NORMAL_GSR)] = "normal"
    return pd.Series(out, index=x.index)


# ====================================================================== reports (2025-26)

def reason_group(reason: str | None) -> str:
    s = reason.lower() if isinstance(reason, str) else ""
    s = s.replace("injury/illness - ", "").replace("injury/illness", "")   # every injury row
    if not s.strip():
        return "none"
    if "reconditioning" in s or "return to competition" in s:
        return "reconditioning"
    if s.startswith("g league") or "two-way" in s:
        return "gleague"
    if s.startswith("rest") or "management" in s or "load" in s:
        return "rest/management"
    if "illness" in s or "sick" in s or "flu" in s or "covid" in s:
        return "illness"
    if re.search(r"hamstring|calf|groin|achilles|adductor|soleus|quad", s):
        return "soft tissue"
    if re.search(r"knee|ankle|foot|toe|heel|acl|mcl|meniscus|patella|plantar|tibia|fibula", s):
        return "knee/ankle/foot"
    return "other"


def load_reports(conn, pg: pd.DataFrame) -> pd.DataFrame:
    rep = pd.read_sql_query(
        "SELECT captured_at, team, player_name, status, reason FROM injury_snapshots "
        "WHERE source='nba_official' AND player_name IS NOT NULL AND player_name <> ''", conn)
    rep["nk"] = rep["player_name"].map(INJ.norm_name)
    p = pg[pg["season"].isin(["2024-25", "2025-26"])][["player_id", "player_name", "team"]]
    p = p.drop_duplicates()
    p["nk"] = p["player_name"].map(INJ.norm_name)
    by_team = p.drop_duplicates(["nk", "team", "player_id"]).groupby(["nk", "team"])["player_id"]
    tmap = {k: v.iloc[0] for k, v in by_team if v.nunique() == 1}
    nmap = {k: v.iloc[0] for k, v in p.groupby("nk")["player_id"] if v.nunique() == 1}
    rep["player_id"] = [tmap.get((k, t), nmap.get(k)) for k, t in zip(rep["nk"], rep["team"])]
    print(f"reports: {len(rep)} rows, matched {rep['player_id'].notna().mean():.3f}")
    rep = rep.dropna(subset=["player_id"])
    rep["player_id"] = rep["player_id"].astype("int64")
    return rep.sort_values("captured_at")


def attach_reasons(df: pd.DataFrame, rep: pd.DataFrame, conn) -> pd.DataFrame:
    """For return rows (gsr 0..15) of 2025-26: the last report row before the return game's tip,
    captured since the day after his last game before the absence."""
    tips = dict(conn.execute("SELECT game_id, tipoff_utc FROM games").fetchall())
    by_p = {pid: g for pid, g in rep.groupby("player_id")}
    reason, status, gday_status = [], [], []
    for r in df.itertuples():
        if r.season != "2025-26" or not (r.gsr == r.gsr) or r.gsr >= NORMAL_GSR:
            reason.append(None); status.append(None); gday_status.append(None)
            continue
        g = by_p.get(int(r.player_id))
        ret_day = r.ret_date
        tip = tips.get(r.game_id) if r.gsr == 0 else None
        hi = (tip or f"{ret_day}T22:00:00Z").replace(" ", "T")
        lo = (pd.Timestamp(ret_day) - pd.Timedelta(days=int(min(r.abs_len, 60) * 2.2 + 3))
              ).strftime("%Y-%m-%d")
        if g is None:
            reason.append(None); status.append(None); gday_status.append(None)
            continue
        w = g[(g["captured_at"] < hi) & (g["captured_at"] >= lo)]
        if w.empty:
            reason.append(None); status.append(None); gday_status.append(None)
            continue
        last = w.iloc[-1]
        reason.append(last["reason"]); status.append(last["status"])
        same_day = w[w["captured_at"] >= f"{ret_day}T13:00:00Z"]       # ~ ET game day
        gday_status.append(same_day.iloc[-1]["status"] if len(same_day) else "not listed")
    df = df.copy()
    df["rep_reason"] = reason
    df["rep_status"] = status
    df["gday_status"] = gday_status
    ret = (df["season"] == "2025-26") & (df["gsr"] < NORMAL_GSR)
    df["reason_g"] = [(reason_group(x) if st is not None else "no report") if r else None
                      for x, st, r in zip(df["rep_reason"], df["rep_status"], ret)]
    return df


# ====================================================================== metrics

def cl_mean(z: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    n = len(z)
    if n == 0:
        return np.nan, np.nan
    m = float(np.mean(z))
    s = pd.Series(z - m).groupby(cl).sum().to_numpy()
    g = len(s)
    se = float(np.sqrt(g / (g - 1) * (s ** 2).sum()) / n) if g > 1 else np.nan
    return m, se


def cl_diff(z1, c1, z0, c0) -> tuple[float, float]:
    """mean(z1) - mean(z0), date-clustered (the two samples share dates)."""
    n1, n0 = len(z1), len(z0)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan
    m1, m0 = np.mean(z1), np.mean(z0)
    e = pd.concat([pd.Series((z1 - m1) / n1, index=c1),
                   pd.Series(-(z0 - m0) / n0, index=c0)]).groupby(level=0).sum().to_numpy()
    g = len(e)
    return float(m1 - m0), float(np.sqrt(g / (g - 1) * (e ** 2).sum()))


def played(d: pd.DataFrame) -> pd.DataFrame:
    """Rows where he played and E[.|plays] is defined; adds the conditional projections."""
    d = d[(d["minutes"] > 0) & (d["p_play"] >= P_MIN) & d["has_proj"].astype(bool)].copy()
    d["min_c"] = d["exp_min"] / d["p_play"]
    d["dk_c"] = d["mean"] / d["p_play"]
    d["e_min"] = d["minutes"] - d["min_c"]
    d["e_dk"] = d["y"] - d["dk_c"]
    return d


def summarize(d: pd.DataFrame, ref: pd.DataFrame | None = None) -> dict:
    """n, players, minutes / DK bias (SE) and MAE; excess bias vs `ref` (same players' normal
    rows) when given."""
    out = {"n": len(d), "players": d["player_id"].nunique() if len(d) else 0}
    if not len(d):
        return out
    cl = d["date"].to_numpy()
    for k, e in (("min", "e_min"), ("dk", "e_dk")):
        z = d[e].to_numpy(float)
        out[f"{k}_bias"], out[f"{k}_se"] = cl_mean(z, cl)
        out[f"{k}_mae"], out[f"{k}_mae_se"] = cl_mean(np.abs(z), cl)
    out["min_proj"] = float(d["min_c"].mean())
    out["min_act"] = float(d["minutes"].mean())
    out["dk_proj"] = float(d["dk_c"].mean())
    out["ratio_min"] = float(d["minutes"].sum() / d["min_c"].sum())
    out["ratio_dk"] = float(d["y"].sum() / d["dk_c"].sum())
    if ref is not None:
        r = ref[ref["player_id"].isin(set(d["player_id"]))]
        out["ref_n"] = len(r)
        for k, e in (("min", "e_min"), ("dk", "e_dk")):
            out[f"{k}_ex"], out[f"{k}_ex_se"] = cl_diff(
                d[e].to_numpy(float), cl, r[e].to_numpy(float), r["date"].to_numpy())
    return out


def table(rows: list[tuple[str, dict]], ex: bool = True) -> str:
    h = ("| group | n (players) | proj → act min | min bias (SE) | min MAE | DK bias (SE) | DK MAE |"
         + (" excess vs own normal: min / DK |" if ex else ""))
    lines = [h, "|" + "---|" * (h.count("|") - 1)]
    for name, s in rows:
        if not s.get("n"):
            lines.append(f"| {name} | 0 |" + " |" * (h.count("|") - 3))
            continue
        line = (f"| {name} | {s['n']:,} ({s['players']}) | {s['min_proj']:.1f} → {s['min_act']:.1f}"
                f" | {s['min_bias']:+.2f} ({s['min_se']:.2f}) | {s['min_mae']:.2f}"
                f" | {s['dk_bias']:+.2f} ({s['dk_se']:.2f}) | {s['dk_mae']:.2f} |")
        if ex:
            line += (f" {s.get('min_ex', np.nan):+.2f} ({s.get('min_ex_se', np.nan):.2f}) / "
                     f"{s.get('dk_ex', np.nan):+.2f} ({s.get('dk_ex_se', np.nan):.2f}) |"
                     if "min_ex" in s else " – |")
        lines.append(line)
    return "\n".join(lines)


# ====================================================================== fix (walk-forward)

CELL = ["gsr_b", "abs_b"]
FIX_GSR = ("0", "1", "2", "3-5")
K_SHRINK = 30.0


def fit_cells(train: pd.DataFrame, by=CELL) -> pd.DataFrame:
    """Shrunk ratio actual / projected (minutes and DK, given played) per cell."""
    t = train[train["gsr_b"].isin(FIX_GSR) & (train["abs_b"] != "")]
    g = t.groupby(by).agg(n=("minutes", "size"), act=("minutes", "sum"), proj=("min_c", "sum"),
                          y=("y", "sum"), dk=("dk_c", "sum"))
    mp = g["proj"] / g["n"]
    mdk = g["dk"] / g["n"]
    g["r_min"] = (g["act"] + K_SHRINK * mp) / (g["proj"] + K_SHRINK * mp)
    g["r_dk"] = (g["y"] + K_SHRINK * mdk) / (g["dk"] + K_SHRINK * mdk)
    return g.reset_index()


def apply_fix(d: pd.DataFrame, cells: pd.DataFrame, by=CELL) -> pd.DataFrame:
    d = d.merge(cells[by + ["r_min", "r_dk"]], on=by, how="left")
    d["r_min"] = d["r_min"].fillna(1.0)
    d["r_dk"] = d["r_dk"].fillna(1.0)
    return d


QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def qscore(y, Q) -> np.ndarray:
    """Quantile-score proxy for CRPS: mean over the 5 stored quantiles of 2 x pinball loss."""
    out = np.zeros(len(y))
    for tau, q in zip(QS, Q):
        u = y - q
        out += 2 * np.where(u >= 0, tau * u, (tau - 1) * u)
    return out / len(QS)


def fix_eval(test: pd.DataFrame, cells: pd.DataFrame, by=CELL) -> dict:
    """Paired before / after on the test arm: DK MAE + quantile score on the standard DK population
    (E[min] >= 10, DNP = 0), minutes MAE on played rows; overall and on the touched rows."""
    d = apply_fix(test, cells, by)
    pop = d[(d["exp_min"] >= 10) & d["has_proj"].astype(bool)].copy()
    y = pop["y"].to_numpy(float)
    Q0 = [pop[f"q{int(t * 100)}"].to_numpy(float) for t in QS]
    Q1 = [q * pop["r_dk"].to_numpy() for q in Q0]
    e0 = np.abs(y - pop["mean"].to_numpy())
    e1 = np.abs(y - pop["mean"].to_numpy() * pop["r_dk"].to_numpy())
    s0, s1 = qscore(y, Q0), qscore(y, Q1)
    touched = (pop["r_dk"] != 1.0).to_numpy()
    cl = pop["date"].to_numpy()
    res = {"n_pop": len(pop), "n_touched": int(touched.sum())}
    for name, m in (("all", np.ones(len(pop), bool)), ("touched", touched)):
        if m.sum() == 0:
            continue
        res[f"dkmae_{name}"] = float(e0[m].mean())
        res[f"d_dkmae_{name}"], res[f"d_dkmae_{name}_se"] = cl_mean((e1 - e0)[m], cl[m])
        res[f"qs_{name}"] = float(s0[m].mean())
        res[f"d_qs_{name}"], res[f"d_qs_{name}_se"] = cl_mean((s1 - s0)[m], cl[m])
        res[f"crps_{name}"] = float(pop["crps"].to_numpy()[m].mean())
    pl = played(d)
    a0 = np.abs(pl["e_min"].to_numpy())
    a1 = np.abs(pl["minutes"].to_numpy() - pl["min_c"].to_numpy() * pl["r_min"].to_numpy())
    t = (pl["r_min"] != 1.0).to_numpy()
    clp = pl["date"].to_numpy()
    res["minmae_all"] = float(a0.mean())
    res["d_minmae_all"], res["d_minmae_all_se"] = cl_mean(a1 - a0, clp)
    if t.sum():
        res["minmae_touched"] = float(a0[t].mean())
        res["d_minmae_touched"], res["d_minmae_touched_se"] = cl_mean((a1 - a0)[t], clp[t])
        res["n_touched_played"] = int(t.sum())
    return res


# ====================================================================== run

def run(out: Path, reuse: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if reuse:
        data = {Path(f).stem[5:]: pd.read_parquet(f) for f in glob.glob(str(out / "rows_*.parquet"))}
        res = report(data)
        (out / "tables.md").write_text(res["md"])
        (out / "results.json").write_text(json.dumps(res["json"], indent=1, default=float))
        print(res["md"])
        return
    conn = connect()
    pg, tg = load_box(conn)
    bs = BoxState(pg, tg)
    arms = load_arms()
    rep = load_reports(conn, pg)
    box_min = pg.set_index(["game_id", "player_id"])["min"]
    data = {}
    for arm, d in arms.items():
        d = attach_state(d, bs)
        if d["season"].iloc[0] in ("2019-20", "2020-21"):          # RotoGuru minutes -> box
            bm = box_min.reindex(pd.MultiIndex.from_arrays([d["game_id"], d["player_id"]]))
            d["minutes"] = np.where(bm.notna().to_numpy(), bm.to_numpy(), d["minutes"])
        if d["season"].iloc[0] == "2025-26":
            d = attach_reasons(d, rep, conn)
        data[arm] = d
        d.to_parquet(out / f"rows_{arm}.parquet")
        print(f"{arm}: {len(d)} rows, return rows (gsr<=5) "
              f"{int(d['gsr_b'].isin(FIX_GSR).sum())}", flush=True)
    res = report(data)
    (out / "tables.md").write_text(res["md"])
    (out / "results.json").write_text(json.dumps(res["json"], indent=1, default=float))
    print(res["md"])


def report(data: dict[str, pd.DataFrame]) -> dict:
    md, js = [], {}
    avail = pd.concat([data[f"avail-{s[2:4]}{s[5:7]}"] for s in AVAIL_SEASONS], ignore_index=True)
    sets = {"avail (5 seasons)": avail, "live+p2": data["live+p2"], "live10+p2": data["live10+p2"],
            "avail 2025-26": data["avail-2526"], "live (p2 off)": data["live"]}
    core = lambda d: d[(d["tier_pre"].isin(["star", "starter", "rotation"]))
                       & (d["team_change"] != 1) & (d["first_season"] == 0)]

    # 1. main: games since return, star/starter/rotation
    for name, d in sets.items():
        p = played(core(d))
        ref = p[p["gsr_b"] == "normal"]
        rows = [("all rows (pre ≥ 15 min)", summarize(p))]
        for g in ("0", "1", "2", "3-5", "6-15", "short", "normal"):
            rows.append((f"gsr {g}" if g not in ("short", "normal") else
                         {"short": "1–2 missed (gsr 0)", "normal": "normal"}[g],
                         summarize(p[p["gsr_b"] == g], ref if g != "normal" else None)))
        rows.append(("gsr 0–2", summarize(p[p["gsr_b"].isin(["0", "1", "2"])], ref)))
        md.append(f"\n### Games since return: {name} (pre-absence ≥ 15 min, played rows)\n")
        md.append(table(rows))
        js[f"gsr|{name}"] = dict(rows)

    # 2. absence length x gsr (0-2 pooled and 0)
    for name in ("avail (5 seasons)", "live+p2", "live10+p2"):
        p = played(core(sets[name]))
        ref = p[p["gsr_b"] == "normal"]
        rows = []
        for a in ("3-5", "6-15", "16+"):
            for g, lab in ((["0"], "0"), (["1", "2"], "1–2"), (["3-5"], "3–5")):
                rows.append((f"abs {a}, gsr {lab}",
                             summarize(p[(p["abs_b"] == a) & p["gsr_b"].isin(g)], ref)))
        md.append(f"\n### Absence length × games since return: {name}\n")
        md.append(table(rows))
        js[f"abs|{name}"] = dict(rows)

    # 3. tier, gsr 0-2
    for name in ("avail (5 seasons)", "live+p2"):
        d = sets[name]
        d = d[(d["team_change"] != 1) & (d["first_season"] == 0)]
        p = played(d)
        rows = []
        for t in ("star", "starter", "rotation", "bench"):
            q = p[p["tier_pre"] == t]
            ref = q[q["gsr_b"] == "normal"]
            rows.append((f"{t}, gsr 0–2", summarize(q[q["gsr_b"].isin(["0", "1", "2"])], ref)))
            rows.append((f"{t}, normal", summarize(ref)))
        md.append(f"\n### Pre-absence tier: {name}\n")
        md.append(table(rows))
        js[f"tier|{name}"] = dict(rows)

    # 4. per season (avail), gsr 0-2
    rows = []
    for s in AVAIL_SEASONS:
        p = played(core(data[f"avail-{s[2:4]}{s[5:7]}"]))
        ref = p[p["gsr_b"] == "normal"]
        rows.append((f"{s} gsr 0", summarize(p[p["gsr_b"] == "0"], ref)))
        rows.append((f"{s} gsr 1–2", summarize(p[p["gsr_b"].isin(["1", "2"])], ref)))
    md.append("\n### By season (avail, context ON), pre-absence ≥ 15 min\n")
    md.append(table(rows))
    js["season"] = dict(rows)

    # 5. reasons (2025-26)
    for name in ("live+p2", "avail 2025-26"):
        p = played(core(sets[name]))
        ref = p[p["gsr_b"] == "normal"]
        q = p[p["gsr_b"].isin(["0", "1", "2"])]
        rows = [(f"{r} (gsr 0–2)", summarize(q[q["reason_g"] == r], ref))
                for r in q["reason_g"].fillna("none").value_counts().index]
        q0 = p[p["gsr_b"] == "0"]
        rows += [(f"{r} (gsr 0)", summarize(q0[q0["reason_g"] == r], ref))
                 for r in q0["reason_g"].fillna("none").value_counts().index]
        md.append(f"\n### Report reason before return (2025-26 slice): {name}\n")
        md.append(table(rows))
        js[f"reason|{name}"] = dict(rows)
        if name == "live+p2":
            rows = [(f"game-day {s} (gsr 0)", summarize(q0[q0["gday_status"] == s], ref))
                    for s in q0["gday_status"].fillna("?").value_counts().index]
            md.append("\n### Game-day report status of the first game back: live+p2\n")
            md.append(table(rows))

    # 6. pregame population at lock-45: candidates whose absence is still running
    for name in ("live+p2", "live10+p2", "live (p2 off)"):
        d = core(sets[name])
        c = d[(d["missed_now"] >= 3) & (d["status"].fillna("") != "out")
              & d["has_proj"].astype(bool)].copy()
        c["pl"] = (c["minutes"] > 0).astype(float)
        lines = ["| status | n | mean P(play) | played | gap (SE) | DK bias, DNP = 0 (SE) |",
                 "|---|---|---|---|---|---|"]
        lab = c["status"].fillna("").replace("", "fringe pool (no status)")
        for st, g in [("all", c)] + list(c.groupby(lab)):
            if len(g) < 5:
                continue
            gap = cl_mean((g["p_play"] - g["pl"]).to_numpy(), g["date"].to_numpy())
            b = cl_mean((g["y"] - g["mean"]).to_numpy(), g["date"].to_numpy())
            lines.append(f"| {st} | {len(g)} | {g['p_play'].mean():.3f} | {g['pl'].mean():.3f}"
                         f" | {gap[0]:+.3f} ({gap[1]:.3f}) | {b[0]:+.2f} ({b[1]:.2f}) |")
        md.append(f"\n### Pregame: absence ≥ 3 still running, not Out ({name})\n")
        md.append("\n".join(lines))

    # 7. fix, walk-forward
    md.append("\n### Fix: multiplier by games-since-return × absence length, walk-forward\n")
    seasons = list(AVAIL_SEASONS[:-1])
    fx_lines = ["| test | train | rows / touched | DK MAE all Δ (SE) | DK QS all Δ (SE) "
                "| touched: DK MAE → Δ (SE) | touched: QS Δ (SE) | touched min MAE → Δ (SE) |",
                "|---|---|---|---|---|---|---|---|"]
    js["fix"] = {}
    tests = [("2020-21", ["2019-20"]), ("2023-24", ["2019-20", "2020-21"]),
             ("2024-25", ["2019-20", "2020-21", "2023-24"])]
    tests += [(f"2025-26 {a}", seasons) for a in ("avail-2526", "live+p2", "live10+p2")]
    tests += [("2025-26 live+p2 (in-sample cells)", ["live+p2"])]
    for by, tag in ((CELL, "gsr×abs"), (CELL + ["tier_pre"], "gsr×abs×tier")):
        cells_all = {}
        for test, train in tests:
            arm = test.split()[1] if test.startswith("2025-26") else f"avail-{test[2:4]}{test[5:7]}"
            trn = [data[a] if a in data else data[f"avail-{a[2:4]}{a[5:7]}"] for a in train]
            tr = played(pd.concat([core(t) for t in trn], ignore_index=True))
            cells = fit_cells(tr, by)
            cells_all[test] = cells
            te = data[arm].copy()
            te.loc[~te.index.isin(core(te).index), "gsr_b"] = "x"      # only touch core rows
            r = fix_eval(te, cells, by)
            js["fix"][f"{tag}|{test}"] = r
            fx_lines.append(
                f"| {test} ({tag}) | {'+'.join(t[2:7] if t[0] == '2' else t for t in train)} "
                f"| {r['n_pop']:,} / {r['n_touched']:,} "
                f"| {r['d_dkmae_all']:+.4f} ({r['d_dkmae_all_se']:.4f}) "
                f"| {r['d_qs_all']:+.4f} ({r['d_qs_all_se']:.4f}) "
                f"| {r.get('dkmae_touched', np.nan):.2f} → {r.get('d_dkmae_touched', np.nan):+.3f}"
                f" ({r.get('d_dkmae_touched_se', np.nan):.3f}) "
                f"| {r.get('d_qs_touched', np.nan):+.3f} ({r.get('d_qs_touched_se', np.nan):.3f}) "
                f"| {r.get('minmae_touched', np.nan):.2f} → {r.get('d_minmae_touched', np.nan):+.3f}"
                f" ({r.get('d_minmae_touched_se', np.nan):.3f}) |")
        c = cells_all["2025-26 live+p2"]
        md.append(f"\nCells fit on the 4 earlier avail seasons ({tag}): "
                  + "; ".join(f"{'/'.join(str(r[b]) for b in by)}: n {int(r['n'])}, "
                              f"min ×{r['r_min']:.3f}, DK ×{r['r_dk']:.3f}"
                              for _, r in c.iterrows()))
    md.append("\n" + "\n".join(fx_lines))

    # 8. per-minute decomposition by games since return
    for name in ("avail (5 seasons)", "live+p2"):
        p = played(core(sets[name]))
        lines = ["| gsr | n | DK/min actual | DK/min projected | rate error % (SE) |",
                 "|---|---|---|---|---|"]
        for g in ("0", "1", "2", "3-5", "6-15", "short", "normal"):
            q = p[p["gsr_b"] == g]
            m, se = ratio_se(q)
            lines.append(f"| {g} | {len(q):,} | {q['y'].sum() / q['minutes'].sum():.3f} | "
                         f"{q['dk_c'].sum() / q['min_c'].sum():.3f} | {100 * m:+.1f} ({100 * se:.1f}) |")
        md.append(f"\n### Per-minute DK rate, played rows (pre ≥ 15 min): {name}\n")
        md.append("\n".join(lines))

    # 9. per-minute error by projected minutes / recent minutes (everyone, not just returners)
    for name in ("avail (5 seasons)", "live+p2"):
        p = played(core(sets[name]))
        p["rb"] = pd.cut(p["min_c"] / p["pre_min"], RATIO_BINS)
        lines = ["| E[min∣plays] / last-10 avg | n (gsr 0–2) | rate err % gsr 0–2 (SE) "
                 "| n (other) | rate err % other (SE) | min bias other | DK bias other |",
                 "|---|---|---|---|---|---|---|"]
        for b, g in p.groupby("rb", observed=True):
            r = g[g["gsr_b"].isin(["0", "1", "2"])]
            o = g[~g["gsr_b"].isin(["0", "1", "2"])]
            mr, sr = ratio_se(r)
            mo, so = ratio_se(o)
            lines.append(f"| {b} | {len(r):,} | {100 * mr:+.1f} ({100 * sr:.1f}) | {len(o):,} | "
                         f"{100 * mo:+.1f} ({100 * so:.1f}) | {o['e_min'].mean():+.2f} | "
                         f"{o['e_dk'].mean():+.2f} |")
        md.append(f"\n### Per-minute DK error by projected / recent minutes: {name}\n")
        md.append("\n".join(lines))

    # 10. fix B: DK correction by projected/recent minutes ratio (all rows), walk-forward
    lines = ["| test | train | rows | DK MAE Δ (SE) | QS Δ (SE) | gsr 0–2 rows: DK MAE Δ (SE) "
             "| ratio < 0.8: DK MAE Δ (SE) | ratio > 1.2: DK MAE Δ (SE) |",
             "|---|---|---|---|---|---|---|---|"]
    js["fixB"] = {}
    for test, train in tests:
        arm = test.split()[1] if test.startswith("2025-26") else f"avail-{test[2:4]}{test[5:7]}"
        trn = [data[a] if a in data else data[f"avail-{a[2:4]}{a[5:7]}"] for a in train]
        tr = played(pd.concat([t[(t["team_change"] != 1) & (t["first_season"] == 0)]
                               for t in trn], ignore_index=True))
        f = fit_ratio(tr)
        r = fixb_eval(data[arm], f)
        js["fixB"][test] = r
        lines.append(f"| {test} | {'+'.join(t[2:7] if t[0] == '2' else t for t in train)} | "
                     f"{r['n']:,} | {r['all'][0]:+.4f} ({r['all'][1]:.4f}) | "
                     f"{r['qs'][0]:+.4f} ({r['qs'][1]:.4f}) | {r['ret'][0]:+.3f} ({r['ret'][1]:.3f}) | "
                     f"{r['lo'][0]:+.3f} ({r['lo'][1]:.3f}) | {r['hi'][0]:+.3f} ({r['hi'][1]:.3f}) |")
        if test == "2025-26 live+p2":
            md.append("\nFix B factors (fit on the 4 earlier avail seasons): " + ", ".join(
                f"{b}: ×{v:.3f}" for b, v in f.items()))
    md.append("\n### Fix B: DK factor by E[min∣plays] / last-10 minutes, walk-forward (all rows)\n")
    md.append("\n".join(lines))
    return {"md": "\n".join(md) + "\n", "json": js}


RATIO_BINS = [0, 0.7, 0.8, 0.9, 0.97, 1.03, 1.1, 1.2, 1.4, 10]


def ratio_se(q: pd.DataFrame) -> tuple[float, float]:
    """Per-minute DK rate error sum(y)/sum(minutes) / (sum(dk_c)/sum(min_c)) - 1, date-clustered
    SE by the delta method (ratio of means of y - k * dk_c * minutes / min_c)."""
    if len(q) < 2:
        return np.nan, np.nan
    ra = q["y"].sum() / q["minutes"].sum()
    rp = q["dk_c"].sum() / q["min_c"].sum()
    z = (q["y"] - ra * q["minutes"]).to_numpy() / q["minutes"].mean() / rp
    _m, se = cl_mean(z, q["date"].to_numpy())
    return ra / rp - 1, se


def fit_ratio(tr: pd.DataFrame) -> dict:
    """DK factor (actual / projected given played, relative to the overall level) per bin of
    E[min | plays] / last-10 minutes, shrunk (K_SHRINK rows at 1)."""
    tr = tr[tr["pre_min"] >= 5].copy()
    tr["rb"] = pd.cut(tr["min_c"] / tr["pre_min"], RATIO_BINS)
    lvl = tr["y"].sum() / tr["dk_c"].sum()
    out = {}
    for b, g in tr.groupby("rb", observed=True):
        m = g["dk_c"].mean()
        out[b] = float((g["y"].sum() / lvl + K_SHRINK * m) / (g["dk_c"].sum() + K_SHRINK * m))
    return out


def fixb_eval(d: pd.DataFrame, f: dict) -> dict:
    d = d[d["has_proj"].astype(bool) & (d["exp_min"] >= 10)].copy()
    ok = (d["pre_min"] >= 5) & (d["team_change"] != 1) & (d["first_season"] == 0)
    ratio = d["exp_min"] / d["p_play"].clip(lower=P_MIN) / d["pre_min"]
    rb = pd.cut(ratio, RATIO_BINS)
    k = np.where(ok, rb.map(f).astype(float).fillna(1.0), 1.0)
    y = d["y"].to_numpy(float)
    Q0 = [d[f"q{int(t * 100)}"].to_numpy(float) for t in QS]
    de = np.abs(y - d["mean"].to_numpy() * k) - np.abs(y - d["mean"].to_numpy())
    dq = qscore(y, [q * k for q in Q0]) - qscore(y, Q0)
    cl = d["date"].to_numpy()
    sub = lambda m: cl_mean(de[m], cl[m]) if m.sum() > 1 else (np.nan, np.nan)
    r = ratio.to_numpy()
    return {"n": len(d), "all": cl_mean(de, cl), "qs": cl_mean(dq, cl),
            "base": {"dk_mae": float(np.abs(y - d["mean"].to_numpy()).mean()),
                     "qs": float(qscore(y, Q0).mean()), "crps": float(d["crps"].mean())},
            "ret": sub(d["gsr_b"].isin(["0", "1", "2"]).to_numpy() & ok.to_numpy()),
            "lo": sub((r < 0.8) & ok.to_numpy()), "hi": sub((r > 1.2) & ok.to_numpy())}


# ====================================================================== news grep

PAT = re.compile(r"minutes? restrict|minutes? limit|limited minutes|restriction|ramp(ing)? up|"
                 r"reconditioning|return to competition", re.I)


def news() -> None:
    """How much 'minutes restriction' text do we store? Local DB + raw ESPN / DK captures."""
    conn = connect()
    for src, n, lo, hi in conn.execute(
            "SELECT source, count(*), min(captured_at), max(captured_at) FROM injury_snapshots "
            "GROUP BY 1"):
        print(f"injury_snapshots {src}: {n} rows {lo} .. {hi}")
    rows = conn.execute("SELECT source, reason FROM injury_snapshots").fetchall()
    hits = {}
    for src, r in rows:
        m = PAT.search(r or "")
        if m:
            hits.setdefault((src, m.group(0).lower()), 0)
            hits[(src, m.group(0).lower())] += 1
    for k, v in sorted(hits.items()):
        print("  hit", k, v)
    import gzip
    for sub in ("espn/injuries", "espn/summary", "draftkings"):
        fs = glob.glob(str(ROOT / "var" / "raw" / sub / "**" / "*"), recursive=True)
        fs = [f for f in fs if Path(f).is_file()]
        n = 0
        for f in fs:
            op = gzip.open if f.endswith(".gz") else open
            try:
                with op(f, "rt", errors="ignore") as fh:
                    n += len(PAT.findall(fh.read()))
            except OSError:
                pass
        print(f"raw {sub}: {len(fs)} files, {n} pattern hits")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=WORK / "out")
    r.add_argument("--reuse", action="store_true", help="re-report from the saved rows")
    sub.add_parser("news")
    a = ap.parse_args(argv)
    if a.cmd == "pull":
        pull()
    elif a.cmd == "run":
        run(a.out, a.reuse)
    else:
        news()


if __name__ == "__main__":
    sys.exit(main())
