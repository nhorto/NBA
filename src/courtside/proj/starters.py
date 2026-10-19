"""Starters per team-game -> `player_starts` (docs/PLAYERS.md §4.2, T1.2 part A).

Source ladder, per team-game:

  sdv     sportsdataverse lineups (`sources.sportsdataverse.read_lineups`): the first period-1 row
          of each game (lowest action_number), home_player_1..5 / away_player_1..5. Each side is
          mapped to the tricode whose box holds >= 3 of its five (neutral-site games have
          is_home = 0 for both teams in `team_games`; the `games` table only holds the live
          schedule), else to the `team_games.is_home` / `games.home` team.
  proxy   the team's top-5 minutes in that game (box), used only where sdv has no row for the
          game, or to replace an sdv starter who has no `player_games` row with min > 0 for that
          team in that game (sdv lineups are inferred from PBP and occasionally wrong).

Every team-game in `player_games` (2000-01+, all season types) gets exactly 5 rows when the box
has >= 5 players with minutes.

Public API
  extract_first_lineups(lineups) -> DataFrame[game_id, side, player_id]   (pure, testable)
  build_starts(conn, seasons=None, *, log=print) -> summary dict           (writes player_starts)
  validate_starts(conn, seasons=None) -> dict  (box check; cdn + RotoGuru agreement)
  load_starts(conn, seasons=None) -> DataFrame[game_id, player_id, team, source]
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FIRST_SEASON = "2000-01"
HOME_COLS = [f"home_player_{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player_{i}" for i in range(1, 6)]


# --- extraction ------------------------------------------------------------------------------

def extract_first_lineups(lineups: pd.DataFrame) -> pd.DataFrame:
    """First period-1 row per game -> long frame (game_id, side 'home'/'away', player_id).

    Rows whose five-man set has a 0/missing id or duplicates are dropped for that side."""
    if lineups.empty:
        return pd.DataFrame(columns=["game_id", "side", "player_id"])
    df = lineups[lineups["period"] == 1]
    df = df.sort_values(["game_id", "action_number"], kind="mergesort")
    first = df.drop_duplicates("game_id", keep="first")
    out = []
    for side, cols in (("home", HOME_COLS), ("away", AWAY_COLS)):
        v = first[cols].to_numpy()
        ok = np.all(v > 0, axis=1) & np.array([len(set(r)) == 5 for r in v], dtype=bool)
        sub = first.loc[ok, ["game_id"] + cols]
        long = sub.melt(id_vars="game_id", value_vars=cols, value_name="player_id")
        long["side"] = side
        out.append(long[["game_id", "side", "player_id"]])
    res = pd.concat(out, ignore_index=True)
    res["game_id"] = res["game_id"].astype(str)
    res["player_id"] = res["player_id"].astype("int64")
    return res.sort_values(["game_id", "side", "player_id"]).reset_index(drop=True)


def _box(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT game_id, player_id, team, min FROM player_games WHERE season = ?",
        conn, params=(season,))


def _home_away(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """game_id, team, is_home (team_games; 0/0 at neutral sites) plus `games` home/away."""
    tg = pd.read_sql_query(
        "SELECT game_id, team, is_home FROM team_games WHERE season = ?", conn, params=(season,))
    g = pd.read_sql_query("SELECT game_id, home, away FROM games WHERE season = ?", conn,
                          params=(season,))
    if len(g):
        extra = pd.concat([g[["game_id", "home"]].rename(columns={"home": "team"}).assign(is_home=1),
                           g[["game_id", "away"]].rename(columns={"away": "team"}).assign(is_home=0)])
        tg = pd.concat([tg, extra]).drop_duplicates(["game_id", "team"])
    return tg


def _map_sides(first: pd.DataFrame, played: pd.DataFrame, ha: pd.DataFrame) -> pd.DataFrame:
    """game_id, side -> team. The side's five are matched to the team whose box holds most of
    them (robust at neutral sites, where team_games has is_home = 0 for both teams); ties and
    no-overlap fall back to team_games.is_home."""
    ov = first.merge(played[["game_id", "player_id", "team"]], on=["game_id", "player_id"])
    cnt = ov.groupby(["game_id", "side", "team"]).size().rename("n").reset_index()
    cnt = cnt.sort_values(["game_id", "side", "n"], ascending=[True, True, False])
    best = cnt.drop_duplicates(["game_id", "side"])
    best = best[best["n"] >= 3][["game_id", "side", "team"]]
    # both sides mapped to the same team -> drop (fallback)
    dup = best.duplicated(["game_id", "team"], keep=False)
    best = best[~dup]
    sides = first[["game_id", "side"]].drop_duplicates()
    m = sides.merge(best, on=["game_id", "side"], how="left")
    fb = ha.assign(side=np.where(ha["is_home"] == 1, "home", "away"))
    fb = fb[fb["is_home"] == 1]  # only the explicit home team is unambiguous
    fb_home = fb[["game_id", "team"]].rename(columns={"team": "fb"}).assign(side="home")
    m = m.merge(fb_home, on=["game_id", "side"], how="left")
    m["team"] = m["team"].fillna(m["fb"])
    return m.dropna(subset=["team"])[["game_id", "side", "team"]]


def starts_for_season(box: pd.DataFrame, ha: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    """Combine sdv first lineups with the box. Returns game_id, player_id, team, source.

    box: player_games rows (game_id, player_id, team, min) of the season;
    ha: game_id, team, is_home; first: extract_first_lineups output."""
    played = box[box["min"].fillna(0) > 0]
    sides = _map_sides(first, played, ha)
    f = first.merge(sides, on=["game_id", "side"], how="inner")[["game_id", "player_id", "team"]]
    # keep sdv starters who logged minutes for that team in that game
    ok = f.merge(played[["game_id", "player_id", "team"]], on=["game_id", "player_id", "team"],
                 how="inner")
    ok = ok.drop_duplicates(["game_id", "player_id"])
    ok["source"] = "sdv"
    raw = {"sdv_raw_rows": int(len(first)), "sdv_mapped_rows": int(len(f)),
           "sdv_valid_rows": int(len(ok))}
    # fill each team-game up to 5 with the top-minutes players not already starters
    teams = played[["game_id", "team"]].drop_duplicates()
    have = ok.groupby(["game_id", "team"]).size().rename("n").reset_index()
    teams = teams.merge(have, on=["game_id", "team"], how="left").fillna({"n": 0})
    need = teams[teams["n"] < 5]
    if len(need):
        cand = played.merge(need, on=["game_id", "team"])
        cand = cand.merge(ok[["game_id", "player_id"]].assign(_s=1), on=["game_id", "player_id"],
                          how="left")
        cand = cand[cand["_s"].isna()]
        cand = cand.sort_values(["game_id", "team", "min", "player_id"],
                                ascending=[True, True, False, True], kind="mergesort")
        cand["rk"] = cand.groupby(["game_id", "team"]).cumcount()
        fill = cand[cand["rk"] < (5 - cand["n"])][["game_id", "player_id", "team"]].copy()
        fill["source"] = "proxy"
        ok = pd.concat([ok, fill], ignore_index=True)
    ok = ok.sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)
    ok.attrs.update(raw)
    return ok


def _seasons(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM player_games WHERE season >= ? ORDER BY season",
        (FIRST_SEASON,))]


def build_starts(conn: sqlite3.Connection, seasons=None, *, read_lineups: Callable | None = None,
                 validate: bool = True, log: Callable[[str], None] = print) -> dict:
    """Fill `player_starts` for `seasons` (default: every player_games season >= 2000-01).

    Idempotent: each season's rows are replaced in one transaction. Returns a summary with
    per-season counts and (validate=True) the validation report."""
    if read_lineups is None:
        from courtside.sources.sportsdataverse import read_lineups as _rl
        read_lineups = _rl
    seasons = list(seasons) if seasons is not None else _seasons(conn)
    cols = ["game_id", "action_number", "period"] + HOME_COLS + AWAY_COLS
    per = {}
    for s in seasons:
        box = _box(conn, s)
        if box.empty:
            continue
        try:
            lu = read_lineups(s, columns=cols)
            lu = lu[lu["game_id"].astype(str).isin(set(box["game_id"]))]
            first = extract_first_lineups(lu)
            del lu
        except FileNotFoundError:
            first = pd.DataFrame(columns=["game_id", "side", "player_id"])
        st = starts_for_season(box, _home_away(conn, s), first)
        gids = box["game_id"].unique().tolist()
        with conn:
            conn.executemany("DELETE FROM player_starts WHERE game_id = ?", [(g,) for g in gids])
            conn.executemany(
                "INSERT INTO player_starts(game_id, player_id, team, source) VALUES (?,?,?,?)",
                list(zip(st["game_id"], st["player_id"].astype(int), st["team"], st["source"])))
        tgs = box[box["min"].fillna(0) > 0][["game_id", "team"]].drop_duplicates()
        n_tg = len(tgs)
        nsrc = st.groupby(["game_id", "team"])["source"].agg(lambda x: "sdv" if (x == "sdv").all()
                                                             else "proxy")
        per[s] = {"team_games": n_tg, "rows": len(st),
                  "team_games_all_sdv": int((nsrc == "sdv").sum()),
                  "proxy_rows": int((st["source"] == "proxy").sum()), **st.attrs}
        log(f"{s}: {per[s]}")
    raw = sum(v.get("sdv_raw_rows", 0) for v in per.values())
    out = {"seasons": per,
           # (1) share of raw sdv starters (first period-1 row) with a box row, min > 0, same team
           "sdv_box_valid_rate": round(sum(v.get("sdv_valid_rows", 0) for v in per.values())
                                       / raw, 6) if raw else None,
           "rows": sum(v["rows"] for v in per.values()),
           "team_games": sum(v["team_games"] for v in per.values()),
           "proxy_rows": sum(v["proxy_rows"] for v in per.values())}
    if validate:
        out["validation"] = validate_starts(conn, seasons)
    return out


def load_starts(conn: sqlite3.Connection, seasons=None) -> pd.DataFrame:
    """All player_starts rows (optionally restricted to player_games seasons)."""
    if seasons is None:
        q, p = "SELECT game_id, player_id, team, source FROM player_starts", ()
    else:
        seasons = list(seasons)
        q = ("SELECT s.game_id, s.player_id, s.team, s.source FROM player_starts s WHERE "
             "s.game_id IN (SELECT DISTINCT game_id FROM player_games WHERE season IN ("
             + ",".join("?" * len(seasons)) + "))")
        p = tuple(seasons)
    df = pd.read_sql_query(q, conn, params=p)
    df["player_id"] = df["player_id"].astype("int64")
    return df


# --- validation ------------------------------------------------------------------------------

def _cdn_starters(root: Path) -> pd.DataFrame:
    """(game_id, player_id, starter) from archived cdn boxscores with a `starter` field."""
    rows = []
    for f in sorted(root.glob("*/*.json.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                g = json.load(fh).get("game") or {}
        except (OSError, ValueError):
            continue
        gid = g.get("gameId")
        for side in ("homeTeam", "awayTeam"):
            for p in (g.get(side) or {}).get("players") or []:
                if "starter" not in p or p.get("played") not in ("1", 1, True):
                    continue
                rows.append((gid, int(p["personId"]), int(str(p["starter"]) == "1")))
    return pd.DataFrame(rows, columns=["game_id", "player_id", "starter"])


def _agree(ours: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """ref: game_id, player_id, starter (players who played). ours: game_id, player_id."""
    games = set(ours["game_id"])
    ref = ref[ref["game_id"].isin(games)]
    if ref.empty:
        return {"games": 0}
    m = ref.merge(ours[["game_id", "player_id"]].assign(ours=1), on=["game_id", "player_id"],
                  how="left").fillna({"ours": 0})
    acc = float((m["ours"] == m["starter"]).mean())
    ref_st = m[m["starter"] == 1]
    recall = float(ref_st["ours"].mean()) if len(ref_st) else float("nan")
    return {"games": int(m["game_id"].nunique()), "player_rows": len(m),
            "agreement": round(acc, 4), "ref_starters_found": round(recall, 4)}


def validate_starts(conn: sqlite3.Connection, seasons=None, cdn_root: Path | None = None) -> dict:
    """(1) every starter has a box row with min > 0 for that team; (2) agreement with cdn
    boxscores (`starter` == "1") and RotoGuru `dfs_hist_salaries.starter`."""
    from courtside import config
    st = load_starts(conn, seasons)
    box = pd.read_sql_query("SELECT game_id, player_id, team, min FROM player_games "
                            "WHERE season >= ?", conn, params=(FIRST_SEASON,))
    if seasons is not None:
        box = box[box["game_id"].isin(set(st["game_id"]))]
    j = st.merge(box, on=["game_id", "player_id", "team"], how="left")
    box_ok = float((j["min"].fillna(0) > 0).mean()) if len(j) else float("nan")
    per_tg = st.groupby(["game_id", "team"]).size()
    out = {"rows": len(st), "box_check_rate": round(box_ok, 6),
           "team_games": int(len(per_tg)), "team_games_with_5": int((per_tg == 5).sum()),
           "source_share": st["source"].value_counts(normalize=True).round(4).to_dict()}
    sdv_only = st[st["source"] == "sdv"]
    root = cdn_root or (config.data_dir() / "raw" / "nba_cdn" / "boxscore")
    if root.exists():
        cdn = _cdn_starters(root)
        out["cdn"] = _agree(st, cdn)
        out["cdn_team_game_exact"] = _exact(st, cdn)
    rg = pd.read_sql_query(
        "SELECT game_id, player_id, starter FROM dfs_hist_salaries WHERE game_id IS NOT NULL "
        "AND player_id IS NOT NULL AND starter IS NOT NULL AND minutes > 0", conn)
    if len(rg):
        rg["player_id"] = rg["player_id"].astype("int64")
        rg = rg.drop_duplicates(["game_id", "player_id"])
        out["rotoguru"] = _agree(st, rg)
        out["rotoguru_sdv_rows_only"] = _agree(sdv_only, rg)
    return out


def _exact(ours: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Share of team-games where our 5 equal the reference's starters (ref with exactly 5)."""
    r = ref[ref["starter"] == 1].merge(ours[["game_id", "player_id", "team"]],
                                       on=["game_id", "player_id"], how="left")
    rs = ref[ref["starter"] == 1].groupby("game_id")["player_id"].apply(frozenset)
    os_ = ours[ours["game_id"].isin(rs.index)].groupby("game_id")["player_id"].apply(frozenset)
    common = rs.index.intersection(os_.index)
    if not len(common):
        return {"games": 0}
    eq = np.mean([rs[g] == os_[g] for g in common])
    del r
    return {"games": int(len(common)), "both_teams_exact": round(float(eq), 4)}
