"""NBA DFS (DraftKings) empirical correlation study on the Courtside DB.

Run:  cd /home/nicholas/Documents/NBA && uv run python /tmp/dfs_corr/analysis.py [section ...]
Sections: build env blowout absence dist corr stability   (default: all)
Read-only DB access. Single process, pandas/numpy (+scipy for distribution fits).

DK Classic scoring used: pts 1, 3PM +0.5, reb 1.25, ast 1.5, stl 2, blk 2, tov -0.5,
double-double +1.5, triple-double +3 *in addition* (so a triple-double = +4.5 total).
"""
from __future__ import annotations
import sys, sqlite3, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DB = "file:/home/nicholas/Documents/NBA/var/courtside.db?mode=ro"
FIRST_SEASON = "2015-16"
RNG = np.random.default_rng(7)
MIN_GP = 15           # min games for a player-team-season to have a stable mean
ROT_MIN = 12.0        # rotation cutoff: season avg mpg


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        vals = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- data
def dk_points(df: pd.DataFrame) -> pd.Series:
    cats = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    p = (df.pts + 0.5 * df.fg3m + 1.25 * df.reb + 1.5 * df.ast + 2 * df.stl + 2 * df.blk - 0.5 * df.tov)
    return p + 1.5 * (cats >= 2) + 3.0 * (cats >= 3)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = sqlite3.connect(DB, uri=True)
    pg = pd.read_sql(
        """select p.game_id,p.player_id,p.player_name,p.team,p.season,p.game_date,p.min,p.pts,p.fg3m,p.reb,
                  p.ast,p.stl,p.blk,p.tov,a.usg_pct
           from player_games p left join player_games_adv a using(game_id,player_id)
           where p.season_type='Regular Season' and p.season>=? and p.min>0""", c, params=(FIRST_SEASON,))
    tg = pd.read_sql(
        """select t.game_id,t.team,t.season,t.game_date,t.opp,t.is_home,t.pts,t.plus_minus,a.pace,
                  cl.total_close,cl.home_spread_close
           from team_games t left join team_games_adv a using(game_id,team)
           left join (select game_id, avg(total_close) total_close, avg(home_spread_close) home_spread_close
                      from closing_lines group by game_id) cl on cl.game_id=t.game_id
           where t.season_type='Regular Season' and t.season>=?""", c, params=(FIRST_SEASON,))
    pos = pd.read_sql("select player_id, position from player_index", c)
    c.close()
    pg["dk"] = dk_points(pg)
    pg["usg_pct"] = pg.usg_pct.astype(float)
    # position: primary letter G/F/C from current-snapshot player_index
    pos["pos"] = pos.position.str.split("-").str[0]
    pg = pg.merge(pos[["player_id", "pos"]], on="player_id", how="left")
    # stint (player-team-season) stats
    g = pg.groupby(["player_id", "season", "team"])
    st = g.agg(gp=("dk", "size"), mu=("dk", "mean"), mmin=("min", "mean"),
               usg=("usg_pct", "mean")).reset_index()
    pg = pg.merge(st, on=["player_id", "season", "team"])
    pg["res"] = pg.dk - pg.mu
    pg["res_min"] = pg["min"] - pg.mmin
    pg["dkpm"] = pg.dk / pg["min"]
    # tiers (team-season): S = top-2 mean DK among avg>=24mpg; R = >=22 mpg non-S; B = 12-22 mpg
    st = st[(st.gp >= MIN_GP) & (st.mmin >= ROT_MIN)].copy()
    st["rk"] = st.assign(m=np.where(st.mmin >= 24, st.mu, -1)).groupby(["season", "team"]).m.rank(ascending=False, method="first")
    st["tier"] = np.where((st.rk <= 2) & (st.mmin >= 24), "S", np.where(st.mmin >= 22, "R", "B"))
    st["utier"] = pd.cut(st.usg, [0, 0.18, 0.25, 1], labels=["L", "M", "H"]).astype(str)
    pg = pg.merge(st[["player_id", "season", "team", "tier", "utier"]], on=["player_id", "season", "team"], how="inner")
    # team-game context
    opp = tg[["game_id", "team", "pts"]].rename(columns={"team": "opp", "pts": "opp_pts"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")
    tg["margin"] = tg.pts - tg.opp_pts
    tg["amargin"] = tg.margin.abs()
    tg["gtotal"] = tg.pts + tg.opp_pts
    tg["vtotal"] = pd.to_numeric(tg.total_close, errors="coerce")
    tg["vspread"] = np.where(tg.is_home == 1, -pd.to_numeric(tg.home_spread_close, errors="coerce"),
                             pd.to_numeric(tg.home_spread_close, errors="coerce"))  # + = team favored
    tgd = pg.groupby(["game_id", "team"]).dk.sum().rename("team_dk").reset_index()
    tg = tg.merge(tgd, on=["game_id", "team"], how="left")
    pg = pg.merge(tg[["game_id", "team", "opp", "is_home", "amargin", "margin", "gtotal", "pace", "vtotal", "vspread"]],
                  on=["game_id", "team"], how="left")
    return pg, tg


# ----------------------------------------------------------------------------- pair correlations
TIER_ORD = {"S": 0, "R": 1, "B": 2}


def make_pairs(pg: pd.DataFrame, cross: bool) -> pd.DataFrame:
    cols = ["game_id", "team", "player_id", "season", "res", "res_min", "tier", "utier", "pos"]
    a = pg[cols]
    if cross:
        b = pg[["game_id", "team", "player_id", "res", "res_min", "tier", "utier", "pos"]].rename(
            columns={"team": "team_b", "player_id": "pid_b", "res": "res_b", "res_min": "resmin_b",
                     "tier": "tier_b", "utier": "utier_b", "pos": "pos_b"})
        m = a.merge(b, on="game_id")
        m = m[m.team != m.team_b]
    else:
        b = pg[["game_id", "team", "player_id", "res", "res_min", "tier", "utier", "pos"]].rename(
            columns={"player_id": "pid_b", "res": "res_b", "res_min": "resmin_b",
                     "tier": "tier_b", "utier": "utier_b", "pos": "pos_b"})
        m = a.merge(b, on=["game_id", "team"])
        m = m[m.player_id != m.pid_b]
    m["tc"] = m.tier.map(TIER_ORD); m["tcb"] = m.tier_b.map(TIER_ORD)
    keep = (m.tc < m.tcb) | ((m.tc == m.tcb))  # ordered: lower tier index first; same-tier both orders kept
    m = m[keep].copy()
    m["tp"] = m.tier + m.tier_b
    um = {"H": 0, "M": 1, "L": 2}
    m["uc"] = m.utier.map(um); m["ucb"] = m.utier_b.map(um)
    m["up"] = np.where(m.uc <= m.ucb, m.utier + m.utier_b, m.utier_b + m.utier)
    m["pp"] = np.where(m.pos.fillna("U") <= m.pos_b.fillna("U"), m.pos.fillna("U") + m.pos_b.fillna("U"),
                       m.pos_b.fillna("U") + m.pos.fillna("U"))
    m["tg"] = m.game_id + m.team
    return m


def corr_boot(m: pd.DataFrame, cat: str, x="res", y="res_b", nboot=300, min_n=300) -> pd.DataFrame:
    """Pearson r per category with cluster (team-game) bootstrap 95% CI. Also returns pair count."""
    out = []
    tgi = pd.factorize(m.tg)[0]; G = tgi.max() + 1
    w = RNG.multinomial(G, np.ones(G) / G, size=nboot).astype(float)  # nboot x G
    X = m[x].values; Y = m[y].values
    for k, idx in m.groupby(cat).indices.items():
        if len(idx) < min_n:
            continue
        gi = tgi[idx]
        S = np.zeros((G, 6))
        xs, ys = X[idx], Y[idx]
        for j, v in enumerate([np.ones(len(idx)), xs, ys, xs * xs, ys * ys, xs * ys]):
            S[:, j] = np.bincount(gi, weights=v, minlength=G)

        def r(T):
            n, sx, sy, sxx, syy, sxy = T
            return (sxy / n - sx / n * sy / n) / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
        r0 = r(S.sum(0))
        bs = (w @ S)
        rb = np.array([r(t) for t in bs])
        sub = m.iloc[idx]
        key = tgi[idx].astype(np.int64) * 10_000_000_000 + np.minimum(sub.player_id.values, sub.pid_b.values).astype(np.int64) * 100_000 + np.maximum(sub.player_id.values, sub.pid_b.values).astype(np.int64) % 100_000
        out.append((k, len(np.unique(key)), r0, np.nanpercentile(rb, 2.5), np.nanpercentile(rb, 97.5)))
    return pd.DataFrame(out, columns=[cat, "unordered pairs", "r", "lo", "hi"]).set_index(cat)


def sec_corr(pg):
    tm = make_pairs(pg, cross=False)
    print("\n## 1. Teammate residual correlation (DK pts minus own stint mean), regular seasons 2015-16..2025-26")
    print(f"team-games: {tm.tg.nunique():,}; ordered pairs: {len(tm):,}; players in rotation (>=12 mpg, >=15 GP): {pg.player_id.nunique():,}")
    tm["all"] = "all"
    print("\nAll teammates:"); print(md(corr_boot(tm, "all")))
    print("\nBy role pair (S=top-2 DK/g with >=24 mpg, R=>=22 mpg non-S, B=12-22 mpg):")
    print(md(corr_boot(tm, "tp")))
    print("\nBy usage-tier pair (H>=25%, M 18-25%, L<18% season USG):")
    print(md(corr_boot(tm, "up")))
    t2 = tm[tm.pp.str.contains("U") == False]
    print("\nBy primary-position pair (current player_index G/F/C; pairs share last-snapshot positions):")
    print(md(corr_boot(t2, "pp")))
    print("\nMinutes-only residual correlation (teammate minutes) / DK-per-minute residual correlation:")
    t3 = tm.copy(); t3["res_min_b"] = t3["resmin_b"]
    print("  minutes resid r =", round(float(np.corrcoef(t3.res_min, t3.res_min_b)[0, 1]), 3))
    # opponent
    op = make_pairs(pg, cross=True)
    op["all"] = "all"
    print("\n## 2. Opponent residual correlation (player vs. players on the other team, same game)")
    print(f"ordered pairs: {len(op):,}")
    print(md(corr_boot(op, "all", nboot=150)))
    print(md(corr_boot(op, "tp", nboot=150)))
    return tm


def sec_stability(pg, tm=None):
    if tm is None:
        tm = make_pairs(pg, cross=False)
    print("\n## 6. Season-to-season stability of teammate residual correlation")
    rows = []
    for s, d in tm.groupby("season"):
        d = d.copy(); d["all"] = "all"
        a = corr_boot(d, "all", nboot=150).iloc[0].rename({"unordered pairs": "pairs"})
        ss = corr_boot(d, "tp", nboot=150, min_n=200)
        rows.append((s, a.pairs, a.r, a.lo, a.hi, ss.r.get("SS", np.nan), ss.r.get("SR", np.nan), ss.r.get("RR", np.nan), ss.r.get("RB", np.nan), ss.r.get("BB", np.nan)))
    t = pd.DataFrame(rows, columns=["season", "pairs", "r_all", "lo", "hi", "SS", "SR", "RR", "RB", "BB"]).set_index("season")
    print(md(t))
    se = ((t.hi - t.lo) / 3.92).mean()
    print(f"\nr_all: mean {t.r_all.mean():.3f}, SD across seasons {t.r_all.std():.3f}, mean bootstrap SE {se:.3f}, range [{t.r_all.min():.3f},{t.r_all.max():.3f}]")
    q = (((t.r_all - t.r_all.mean()) / ((t.hi - t.lo) / 3.92)) ** 2).sum()
    print(f"Cochran Q (all-pairs) = {q:.1f} on {len(t)-1} df (SE from game-cluster bootstrap)")
    # team-level: same team-season consistency (split halves by odd/even game within team-season)
    d = tm.copy(); d["half"] = pd.factorize(d.game_id)[0] % 2
    d["ts"] = d.season + d.team
    h = d.groupby(["ts", "half"]).apply(lambda x: np.corrcoef(x.res, x.res_b)[0, 1]).unstack()
    print(f"team-season split-half (odd/even games) r of team teammate-corr: {h.corr().iloc[0,1]:.3f} (n={len(h)} team-seasons)")
    return t


# ----------------------------------------------------------------------------- environment & blowout
def ols_r2(y, Xs):
    X = np.column_stack([np.ones(len(y))] + Xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - res.var() / y.var(), b


def sec_env(pg, tg):
    print("\n## 3. Game environment")
    t = tg.dropna(subset=["team_dk", "pace", "gtotal", "amargin"]).copy()
    t["ts"] = t.season + t.team
    for c in ["team_dk", "gtotal", "pace", "amargin", "pts", "vtotal"]:
        t[c + "_d"] = t[c] - t.groupby("ts")[c].transform("mean")   # within team-season
    n = len(t)
    print(f"team-games: {n:,}  (Vegas total available for {t.vtotal.notna().sum():,})")
    rows = []
    def add(name, cols, d=t):
        r2, b = ols_r2(d.team_dk_d.values, [d[c + "_d"].values for c in cols])
        rows.append((name, len(d), r2))
    add("team's own points (mechanical)", ["pts"])
    add("game total pts (actual)", ["gtotal"])
    add("pace (actual)", ["pace"])
    add("game total + pace", ["gtotal", "pace"])
    add("game total + pace + |margin|", ["gtotal", "pace", "amargin"])
    v = t.dropna(subset=["vtotal"])
    add("Vegas closing total (ex-ante)", ["vtotal"], v)
    add("Vegas total + |margin| (actual)", ["vtotal", "amargin"], v)
    print(md(pd.DataFrame(rows, columns=["predictors (team-season demeaned)", "n", "R2 of team DK total"]).set_index("predictors (team-season demeaned)")))
    r2, b = ols_r2(t.team_dk_d.values, [t.gtotal_d.values, t.pace_d.values, t.amargin_d.values])
    print(f"\nteam DK total slopes: per +1 game-total pt {b[1]:.2f} DK; per +1 pace {b[2]:.2f}; per +1 |margin| {b[3]:.2f}")
    # per-player view
    print("\nPlayer-level: R2 of player DK residual on environment, by tier (ex-ante Vegas total / spread, ex-post total, pace, |margin|)")
    p = pg.dropna(subset=["gtotal", "pace", "amargin", "vtotal"]).copy()
    for c in ["gtotal", "pace", "amargin", "vtotal"]:
        p[c + "_d"] = p[c] - p.groupby(["season", "team"])[c].transform("mean")
    rows = []
    for tier, d in [("S", p[p.tier == "S"]), ("R", p[p.tier == "R"]), ("B", p[p.tier == "B"])]:
        r_v, bv = ols_r2(d.res.values, [d.vtotal_d.values])
        r_a, ba = ols_r2(d.res.values, [d.gtotal_d.values, d.pace_d.values, d.amargin_d.values])
        r_m, bm = ols_r2(d.res_min.values, [d.amargin_d.values])
        rows.append((tier, len(d), r_v, bv[1], r_a, ba[1], ba[2], ba[3], r_m))
    print(md(pd.DataFrame(rows, columns=["tier", "n", "R2 Vegas total", "DK per +1 Vegas pt", "R2 total+pace+|m|", "b_total", "b_pace", "b_|margin|", "R2 minutes~|m|"]).set_index("tier")))


def sec_blowout(pg):
    print("\n## 3b. Blowout curve: residual DK and minutes vs final |margin|")
    p = pg.dropna(subset=["amargin"]).copy()
    p["bin"] = pd.cut(p.amargin, [-1, 5, 10, 15, 20, 25, 100], labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26+"])
    def tab(d, label):
        g = d.groupby("bin")
        se = g.res.std() / np.sqrt(g.size())
        t = pd.DataFrame({"n": g.size(), "mean DK resid": g.res.mean(), "95% +/-": 1.96 * se,
                          "mean min resid": g.res_min.mean(), "DK/min resid": g.apply(lambda x: (x.dkpm - x.mu / x.mmin).mean())})
        print(f"\n{label}"); print(md(t))
    tab(p[p.tier == "S"], "S (top-2 DK per team)")
    tab(p[p.tier.isin(["S", "R"])], "Starters (S+R, >=22 mpg)")
    tab(p[p.tier == "B"], "Bench (B, 12-22 mpg)")
    print("\nStarters (S+R) split by game winner/loser at |margin|>=16:")
    d = p[p.tier.isin(["S", "R"]) & (p.amargin >= 16)]
    for lab, dd in [("winning team", d[d.margin > 0]), ("losing team", d[d.margin < 0])]:
        print(f"  {lab}: n={len(dd):,} mean DK resid {dd.res.mean():+.2f} +/- {1.96*dd.res.std()/np.sqrt(len(dd)):.2f}, min resid {dd.res_min.mean():+.2f}")
    d = p[p.tier.isin(["S", "R"])]
    close = d[d.amargin <= 5].res.mean()
    print(f"  baseline close games (|m|<=5) starters resid {close:+.2f}")


# ----------------------------------------------------------------------------- absence
def sec_absence(pg, tg):
    print("\n## 4. Teammate-absence usage bump")
    gi = pg.groupby(["season", "team"]).player_id.nunique()
    nteams = pg.groupby(["player_id", "season"]).team.nunique().rename("nt").reset_index()
    p = pg.merge(nteams, on=["player_id", "season"])
    p = p[p.nt == 1]
    games = tg[["game_id", "season", "team"]].drop_duplicates()
    stint = p.groupby(["season", "team", "player_id"]).agg(gp=("dk", "size"), mmin=("min", "mean"), usg=("usg_pct", "mean")).reset_index()
    rec = []   # one row per (star, player) pair
    for (s, t), st in stint.groupby(["season", "team"]):
        tgames = set(games[(games.season == s) & (games.team == t)].game_id)
        cand = st[(st.mmin >= 28) & (st.gp >= 20)].sort_values("usg", ascending=False).head(2)
        if len(cand) < 2:
            continue
        top5 = st.sort_values("mmin", ascending=False).head(6).player_id.tolist()
        sub = p[(p.season == s) & (p.team == t)]
        played = sub.groupby("player_id").game_id.apply(set)
        for rank, (star, other) in enumerate([(cand.iloc[0].player_id, cand.iloc[1].player_id), (cand.iloc[1].player_id, cand.iloc[0].player_id)], start=1):
            out = tgames - played[star]
            out = out - (tgames - played[other])      # other top-usage star must be available
            inn = played[star] & played[other]
            if len(out) < 3:
                continue
            for pid in top5:
                if pid == star:
                    continue
                d = sub[sub.player_id == pid]
                do, di = d[d.game_id.isin(out)], d[d.game_id.isin(inn)]
                if len(do) < 3 or len(di) < 10:
                    continue
                rec.append(dict(season=s, team=t, star=star, rank=rank, pid=pid, is_other_star=(pid == other),
                                n_out=len(do), n_in=len(di),
                                d_min=do["min"].mean() - di["min"].mean(),
                                d_dk=do.dk.mean() - di.dk.mean(),
                                d_dkpm=do.dk.sum() / do["min"].sum() - di.dk.sum() / di["min"].sum(),
                                dk_in=di.dk.mean(), dkpm_in=di.dk.sum() / di["min"].sum(), min_in=di["min"].mean(),
                                d_usg=do.usg_pct.mean() - di.usg_pct.mean()))
    r = pd.DataFrame(rec)
    r["unit"] = r.season + r.team + r.star.astype(str)
    def agg(d, label):
        w = d.n_out.values
        res = {}
        units = d.unit.values; uu = pd.unique(units)
        for m in ["d_min", "d_dk", "d_dkpm", "d_usg"]:
            x = d[m].values
            est = np.average(x, weights=w)
            bs = []
            idx = {u: np.where(units == u)[0] for u in uu}
            for _ in range(300):
                pick = RNG.choice(len(uu), len(uu))
                ii = np.concatenate([idx[uu[k]] for k in pick])
                bs.append(np.average(x[ii], weights=w[ii]))
            res[m] = (est, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        return dict(group=label, star_seasons=len(uu), player_pairs=len(d), player_games_out=int(w.sum()),
                    base_min=np.average(d.min_in, weights=w), base_dk=np.average(d.dk_in, weights=w),
                    d_min=f"{res['d_min'][0]:+.2f} [{res['d_min'][1]:+.2f},{res['d_min'][2]:+.2f}]",
                    d_dk=f"{res['d_dk'][0]:+.2f} [{res['d_dk'][1]:+.2f},{res['d_dk'][2]:+.2f}]",
                    d_dkpm=f"{res['d_dkpm'][0]:+.3f} [{res['d_dkpm'][1]:+.3f},{res['d_dkpm'][2]:+.3f}]",
                    d_usg=f"{res['d_usg'][0]*100:+.1f}pp")
    rows = [agg(r, "all remaining top-6 (min) players"),
            agg(r[r.is_other_star], "the other top-2-usage star"),
            agg(r[~r.is_other_star], "non-star starters/rotation"),
            agg(r[r["rank"] == 1], "star #1 usage out (all)"),
            agg(r[r["rank"] == 2], "star #2 usage out (all)"),
            agg(r[(r["rank"] == 1) & ~r.is_other_star], "star #1 out, non-star"),
            agg(r[(r["rank"] == 2) & ~r.is_other_star], "star #2 out, non-star"),
            agg(r[(~r.is_other_star) & (r.min_in >= 28)], "non-star with >=28 mpg baseline")]
    print(md(pd.DataFrame(rows).set_index("group"), "{:.1f}"))
    print("(d_* = games with star out minus games with star in, same player & season, player-games weighted; other top-usage star required available; single-team players only; CI = bootstrap over star-seasons)")
    return r


# ----------------------------------------------------------------------------- distribution
def sec_dist(pg):
    from scipy import stats
    print("\n## 5. DK fantasy-point distribution shape per role tier (per player-game, played games)")
    rows = []
    for tier in ["S", "R", "B"]:
        d = pg[pg.tier == tier].dk
        pc = np.percentile(d, [10, 25, 50, 75, 90, 99])
        rows.append((tier, len(d), d.mean(), d.std(), stats.skew(d), stats.kurtosis(d), *pc, (d <= 0).mean()))
    print(md(pd.DataFrame(rows, columns=["tier", "n", "mean", "SD", "skew", "ex.kurt", "p10", "p25", "p50", "p75", "p90", "p99", "P(<=0)"]).set_index("tier"), "{:.2f}"))
    print("\nWithin-player shape: z = DK / own stint mean (removes talent mix). Fit comparison (mean log-lik per obs, higher=better; KS statistic, lower=better), n=60k subsample/tier")
    rows = []
    for tier in ["S", "R", "B"]:
        d = pg[(pg.tier == tier) & (pg.mu > 5) & (pg.dk > 0)]
        z = (d.dk / d.mu).values
        z = RNG.choice(z, size=min(60000, len(z)), replace=False)
        for name, dist, kw in [("normal", stats.norm, {}), ("lognormal", stats.lognorm, {"floc": 0}), ("gamma", stats.gamma, {"floc": 0})]:
            par = dist.fit(z, **kw)
            ll = dist.logpdf(z, *par).mean()
            ks = stats.kstest(z, dist.cdf, args=par).statistic
            rows.append((tier, name, ll, ks, z.mean(), z.std(), stats.skew(z), np.percentile(z, 99)))
    print(md(pd.DataFrame(rows, columns=["tier", "fit", "loglik/obs", "KS", "z mean", "z SD(CV)", "z skew", "z p99"]).set_index("tier")))
    print("\nCoefficient of variation (SD/mean of DK) and p99/mean by season-average DK band, pooled player-games:")
    pg2 = pg.copy(); pg2["band"] = pd.cut(pg2.mu, [0, 10, 20, 30, 40, 50, 80])
    g = pg2.groupby("band")
    t = pd.DataFrame({"n": g.size(), "mean": g.dk.mean(), "SD": g.dk.std(), "CV (within player)": g.apply(lambda x: (x.dk - x.mu).std() / x.mu.mean()),
                      "skew (resid)": g.res.apply(stats.skew), "p10/mean": g.dk.quantile(.1) / g.dk.mean(), "p90/mean": g.dk.quantile(.9) / g.dk.mean(),
                      "p99/mean": g.dk.quantile(.99) / g.dk.mean()})
    print(md(t, "{:.2f}"))
    d = pg[(pg.tier == "S")]
    td = ((d[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 3).mean()
    print(f"\nTriple-double rate among S tier player-games: {td:.3%}; double-double rate: {((d[['pts','reb','ast','stl','blk']] >= 10).sum(axis=1) >= 2).mean():.1%}")


SECTIONS = ["build", "corr", "env", "blowout", "absence", "dist", "stability"]
if __name__ == "__main__":
    want = sys.argv[1:] or SECTIONS
    pg, tg = build()
    print(f"player-games used: {len(pg):,} ({pg.season.min()}..{pg.season.max()}); tiers: {pg.groupby('tier').size().to_dict()}")
    tm = None
    if "corr" in want:
        tm = sec_corr(pg)
    if "env" in want:
        sec_env(pg, tg)
    if "blowout" in want:
        sec_blowout(pg)
    if "absence" in want:
        sec_absence(pg, tg)
    if "dist" in want:
        sec_dist(pg)
    if "stability" in want:
        sec_stability(pg, tm)
