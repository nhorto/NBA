"""Historical pre-game lines -> `closing_lines`, matched to `team_games` games.

Sources
  kaggle  CC0 dataset cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024: SBR closing lines
          through 2023-01-16, ESPN BET afterwards. Close only; moneylines missing from 2023-01-17.
  espn    ESPN core API odds (open + close + moneylines), optional/resumable backfill, 2022-23+.

All spreads are stored HOME-signed (negative = home favored). Moneylines are American odds.
Run:  python -m courtside.odds_history kaggle | espn --from 2025-26 --to 2025-26 | sanity
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sqlite3
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import archive, config, db, http

KAGGLE_SLUG = "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}"
SBR_LAST_DATE = "2023-01-16"        # Kaggle rows after this are ESPN BET lines

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/"
             "{eid}/competitions/{eid}/odds")
ESPN_FIRST_SEASON = "2022-23"

# <=3 req/s on the ESPN hosts (http.py default is 0.3s; be explicit).
for _host in ("site.api.espn.com", "sports.core.api.espn.com"):
    http.MIN_INTERVAL.setdefault(_host, 0.34)

# Historical / alternate codes -> one canonical code per franchise (2007+ scope).
ALIASES: dict[str, str] = {
    "NJN": "BKN", "NJ": "BKN", "BRK": "BKN",
    "SEA": "OKC",
    "NOH": "NOP", "NOK": "NOP", "NO": "NOP",
    "CHO": "CHA", "CHH": "CHA",
    "PHO": "PHX", "GS": "GSW", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
    "VAN": "MEM",
}

CL_COLUMNS = ("game_id", "source", "provider", "home_spread_open", "home_spread_close",
              "total_open", "total_close", "home_ml_open", "home_ml_close",
              "away_ml_open", "away_ml_close")


def norm_team(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIASES.get(c, c)


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[2:]}"


def _num(v: Any) -> float | None:
    """'+1.5' / '-108' / 225.5 / 'o222.5' / 'EVEN' / 'PK' -> float, else None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "-", "n/a"):
        return None
    if s in ("even", "ev"):
        return 100.0
    if s in ("pk", "pick", "pickem", "pick'em"):
        return 0.0
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------- game index

class GameIndex:
    """(home, away) franchise codes -> sorted [(date, game_id)], built from team_games."""

    def __init__(self, conn: sqlite3.Connection, seasons: Iterable[str] | None = None):
        sql = ("SELECT game_id, game_date, team, opp, is_home, season FROM team_games "
               "WHERE game_date IS NOT NULL")
        params: list = []
        if seasons:
            seasons = list(seasons)
            sql += f" AND season IN ({','.join('?' * len(seasons))})"
            params = seasons
        self.by_pair: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        self.game_info: dict[str, dict] = {}
        home_rows: dict[str, Any] = {}
        neutral: dict[str, Any] = {}
        for r in conn.execute(sql, params):
            (home_rows if r["is_home"] == 1 else neutral).setdefault(r["game_id"], r)
        for gid, r in home_rows.items():
            h, a = norm_team(r["team"]), norm_team(r["opp"])
            self.by_pair[(h, a)].append((r["game_date"], gid))
            self.game_info[gid] = {"date": r["game_date"], "home": h, "away": a,
                                   "season": r["season"]}
        # neutral-site games (Paris/Mexico City...): both rows is_home=0 -> either orientation
        for gid, r in neutral.items():
            if gid in home_rows:
                continue
            h, a = norm_team(r["team"]), norm_team(r["opp"])
            self.by_pair[(h, a)].append((r["game_date"], gid))
            self.by_pair[(a, h)].append((r["game_date"], gid))
            self.game_info[gid] = {"date": r["game_date"], "home": h, "away": a,
                                   "season": r["season"], "neutral": True}
        for v in self.by_pair.values():
            v.sort()

    def _find(self, home: str, away: str, day: str, tol: int) -> tuple[str | None, int, str]:
        """-> (game_id | None, day_offset, status) where status in ok|none|ambiguous."""
        cands = self.by_pair.get((home, away), [])
        d = date.fromisoformat(day)
        for off in range(0, tol + 1):
            hits = []
            for sgn in ((0,) if off == 0 else (-1, 1)):
                dd = (d + timedelta(days=sgn * off)).isoformat()
                hits += [(gid, sgn * off) for dt, gid in cands if dt == dd]
            if len(hits) == 1:
                return hits[0][0], hits[0][1], "ok"
            if len(hits) > 1:
                return None, 0, "ambiguous"
        return None, 0, "none"

    def match(self, home: str, away: str, day: str, *, tol: int = 1
              ) -> tuple[str | None, str, int]:
        """Match by date + teams. -> (game_id, how, day_offset); how in
        exact | tolerance | swapped | swapped_tolerance | ambiguous | no_game."""
        home, away = norm_team(home), norm_team(away)
        gid, off, st = self._find(home, away, day, tol)
        if st == "ok":
            return gid, ("exact" if off == 0 else "tolerance"), off
        if st == "ambiguous":
            return None, "ambiguous", 0
        gid, off, st = self._find(away, home, day, tol)      # neutral-site designation differs
        if st == "ok":
            return gid, ("swapped" if off == 0 else "swapped_tolerance"), off
        return None, "no_game", 0


# ---------------------------------------------------------------------------- Kaggle

def parse_kaggle_row(r: dict) -> dict | None:
    """One CSV row -> normalized dict (home-signed spread); None if unusable."""
    day = (r.get("date") or "").strip()[:10]
    home, away = norm_team(r.get("home", "")), norm_team(r.get("away", ""))
    if not day or not home or not away:
        return None
    spread = _num(r.get("spread"))
    if spread is not None:
        spread = abs(spread)
        fav = (r.get("whos_favored") or "").strip().lower()
        spread = 0.0 if spread == 0 else (-spread if fav == "home" else spread)
    year = int(r["season"])
    return {
        "date": day, "home": home, "away": away, "season": season_label(year),
        "playoffs": str(r.get("playoffs")).lower() == "true",
        "provider": "sbr" if day <= SBR_LAST_DATE else "espnbet",
        "home_spread_close": spread,
        "total_close": _num(r.get("total")),
        "home_ml_close": _num(r.get("moneyline_home")),
        "away_ml_close": _num(r.get("moneyline_away")),
        "score_home": _num(r.get("score_home")), "score_away": _num(r.get("score_away")),
    }


def iter_kaggle_rows(csv_text: str) -> Iterator[dict]:
    for r in csv.DictReader(io.StringIO(csv_text)):
        p = parse_kaggle_row(r)
        if p:
            yield p


def _kaggle_zip(force: bool) -> Path:
    base = config.data_dir() / "raw" / "kaggle" / "odds"
    existing = sorted(base.glob("*/*.zip")) if base.exists() else []
    if existing and not force:
        return existing[-1]
    data = http.fetch_bytes(KAGGLE_URL, timeout=120)
    if data[:2] != b"PK":
        raise http.FetchError("Kaggle download is not a zip", url=KAGGLE_URL)
    return archive.save_raw_bytes("kaggle", "odds", date.today().isoformat(),
                                  "nba-betting-data", data, ext=".zip")


def _read_kaggle_csv(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(name).decode("utf-8-sig")


def _write_unmatched(rows: list[dict], name: str = "odds_unmatched.csv") -> Path:
    out = config.data_dir() / "reports" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source", "season", "date", "home", "away", "reason", "detail"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return out


def _cl_row(game_id: str, source: str, provider: str | None, **vals) -> dict:
    row = {c: None for c in CL_COLUMNS}
    row.update(game_id=game_id, source=source, provider=provider, **vals)
    return row


def import_kaggle(conn: sqlite3.Connection, *, force: bool = False,
                  zip_path: str | Path | None = None) -> dict:
    """Download (once; archived under raw/kaggle/odds), match to team_games, upsert 'kaggle' rows.

    force=True re-downloads the zip. Idempotent. Returns match stats incl. per-season rates."""
    path = Path(zip_path) if zip_path else _kaggle_zip(force)
    rows = list(iter_kaggle_rows(_read_kaggle_csv(path)))
    idx = GameIndex(conn)
    how = defaultdict(int)
    per_season: dict[str, dict] = defaultdict(lambda: {"rows": 0, "matched": 0, "tolerance": 0})
    unmatched: list[dict] = []
    out: dict[str, dict] = {}
    score_mismatch = 0
    scores = {}
    for r in conn.execute("SELECT game_id, team, pts, is_home FROM team_games WHERE game_date >= '2007-01-01'"):
        scores.setdefault(r["game_id"], {})[r["is_home"]] = r["pts"]
    for k in rows:
        ps = per_season[k["season"]]
        ps["rows"] += 1
        gid, h, off = idx.match(k["home"], k["away"], k["date"])
        how[h] += 1
        if gid is None:
            unmatched.append({"source": "kaggle", "season": k["season"], "date": k["date"],
                              "home": k["home"], "away": k["away"], "reason": h})
            continue
        if gid in out:
            how["duplicate_target"] += 1
            unmatched.append({"source": "kaggle", "season": k["season"], "date": k["date"],
                              "home": k["home"], "away": k["away"], "reason": "duplicate_target",
                              "detail": gid})
            continue
        ps["matched"] += 1
        if off:
            ps["tolerance"] += 1
        swap = h.startswith("swapped")
        sp, hm, am = k["home_spread_close"], k["home_ml_close"], k["away_ml_close"]
        if swap:
            sp = None if sp is None else (0.0 if sp == 0 else -sp)
            hm, am = am, hm
        sc = scores.get(gid, {})
        if k["score_home"] is not None and sc.get(1) is not None:
            a, b = (k["score_away"], k["score_home"]) if swap else (k["score_home"], k["score_away"])
            if (a, b) != (sc.get(1), sc.get(0)):
                score_mismatch += 1
        out[gid] = _cl_row(gid, "kaggle", k["provider"], home_spread_close=sp,
                           total_close=k["total_close"], home_ml_close=hm, away_ml_close=am)
    # replace the whole kaggle slice so reruns can't leave stale rows behind
    conn.execute("DELETE FROM closing_lines WHERE source = 'kaggle'")
    db.upsert_rows(conn, "closing_lines", list(out.values()))
    report = _write_unmatched(unmatched)
    n, m = len(rows), len(out)
    for ps in per_season.values():
        ps["rate"] = round(ps["matched"] / ps["rows"], 4) if ps["rows"] else None
    return {"zip": str(path), "rows": n, "matched": m, "rate": round(m / n, 4) if n else None,
            "how": dict(how), "tolerance_used": how["tolerance"] + how["swapped_tolerance"],
            "score_mismatch": score_mismatch, "per_season": dict(sorted(per_season.items())),
            "unmatched_csv": str(report)}


# ---------------------------------------------------------------------------- ESPN

PROVIDER_PREFERENCE = ("DraftKings", "ESPN BET", "ESPN Bet", "Caesars Sportsbook", "consensus")


def _is_live(name: str) -> bool:
    return "live odds" in name.lower()


def _amer(block: dict | None, key: str) -> float | None:
    """block[key] is e.g. {'american': '-110'} or {'alternateDisplayValue': ..}."""
    v = (block or {}).get(key)
    if isinstance(v, dict):
        return _num(v.get("american", v.get("alternateDisplayValue")))
    return _num(v)


def _ok_spread(v: float | None) -> float | None:
    return v if v is not None and abs(v) <= 40 else None


def _ok_total(v: float | None) -> float | None:
    return v if v is not None and 150 <= v <= 300 else None


def _ok_ml(v: float | None) -> float | None:
    return v if v is not None and abs(v) >= 100 else None


def _first(*vals: float | None) -> float | None:
    return next((v for v in vals if v is not None), None)


def parse_espn_odds(payload: dict) -> dict | None:
    """ESPN core `odds` payload -> closing_lines fields (without game_id/source), or None.

    Provider choice: valid `close` block first, then any `close` block, then `current` only;
    preference DraftKings / ESPN BET / Caesars / consensus; never 'Live Odds'.
    GOTCHA: in some 2022-23-era payloads `close.pointSpread` / `close.total` hold the JUICE
    (-105, -110) instead of the line; those fail the range check and `current` (the true
    pre-game line for finished games) is used instead; provider is then tagged '[current]'."""
    best = None
    for it in (payload or {}).get("items") or []:
        name = ((it.get("provider") or {}).get("name") or "").strip()
        if not name or _is_live(name):
            continue
        home = it.get("homeTeamOdds") or {}
        away = it.get("awayTeamOdds") or {}
        h_close, h_cur = home.get("close") or {}, home.get("current") or {}
        spread_close_blk = _ok_spread(_amer(h_close, "pointSpread"))
        spread_cur = _first(_ok_spread(_amer(h_cur, "pointSpread")), _ok_spread(_num(it.get("spread"))))
        if spread_close_blk is None and spread_cur is None:
            continue
        tier = 0 if spread_close_blk is not None else (1 if h_close else 2)
        pref = next((i for i, p in enumerate(PROVIDER_PREFERENCE) if p.lower() == name.lower()),
                    len(PROVIDER_PREFERENCE))
        pri = (it.get("provider") or {}).get("priority", 99)
        rank = (tier, pref, pri if isinstance(pri, (int, float)) else 99)
        if best is None or rank < best[0]:
            best = (rank, name, it, spread_close_blk, spread_cur)
    if best is None:
        return None
    _, name, it, spread_close_blk, spread_cur = best
    home, away = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
    h_open, h_close, h_cur = home.get("open") or {}, home.get("close") or {}, home.get("current") or {}
    a_open, a_close, a_cur = away.get("open") or {}, away.get("close") or {}, away.get("current") or {}
    t_open, t_close, t_cur = it.get("open") or {}, it.get("close") or {}, it.get("current") or {}
    spread_close = _first(spread_close_blk, spread_cur)
    total_close = _first(_ok_total(_amer(t_close, "total")), _ok_total(_amer(t_cur, "total")),
                         _ok_total(_num(it.get("overUnder"))))
    return {
        "provider": name if spread_close_blk is not None else f"{name} [current]",
        "home_spread_open": _ok_spread(_amer(h_open, "pointSpread")),
        "home_spread_close": spread_close,
        "total_open": _ok_total(_amer(t_open, "total")),
        "total_close": total_close,
        "home_ml_open": _ok_ml(_amer(h_open, "moneyLine")),
        "home_ml_close": _first(_ok_ml(_amer(h_close, "moneyLine")),
                                _ok_ml(_amer(h_cur, "moneyLine")), _ok_ml(_num(home.get("moneyLine")))),
        "away_ml_open": _ok_ml(_amer(a_open, "moneyLine")),
        "away_ml_close": _first(_ok_ml(_amer(a_close, "moneyLine")),
                                _ok_ml(_amer(a_cur, "moneyLine")), _ok_ml(_num(away.get("moneyLine")))),
    }


def _season_range(a: str, b: str) -> list[str]:
    y0, y1 = int(a[:4]), int(b[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(y0, y1 + 1)]


def _cached(source: str, kind: str, day: str, key: str, resume: bool, fetch, stats: dict):
    """Load from the raw archive when resuming, else fetch (and archive). None on failure."""
    p = archive.raw_path(source, kind, day, key)
    if resume and p.exists():
        stats["cached"] += 1
        return archive.load_raw(p)
    try:
        payload = fetch()
    except http.FetchError as e:
        if e.status == 404:
            payload = {"_status": 404, "count": 0, "items": [], "events": []}
        else:
            stats["errors"] += 1
            print(f"espn: {kind} {key}: {e}", file=sys.stderr)
            return None
    archive.save_raw(source, kind, day, key, payload)
    stats["requests"] += 1
    return payload


def backfill_espn(conn: sqlite3.Connection, seasons: Iterable[str], *, resume: bool = True,
                  progress: bool = True) -> dict:
    """Fetch ESPN scoreboards + odds for every game date in `seasons`, upsert source='espn'.

    resume=True reuses archived raw responses (no network for anything already saved), so an
    interrupted run restarts where it stopped. Only dates <= today are requested."""
    from .sources import espn as espn_src

    seasons = list(seasons)
    idx = GameIndex(conn, seasons)
    today = date.today().isoformat()
    stats = {"requests": 0, "cached": 0, "errors": 0, "events": 0, "matched": 0, "with_odds": 0,
             "no_odds": 0, "unmatched_events": 0, "per_season": {}}
    unmatched: list[dict] = []
    for season in seasons:
        days = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM team_games WHERE season = ? AND game_date <= ? "
            "ORDER BY game_date", (season, today))]
        ps = stats["per_season"][season] = {"games": 0, "with_odds": 0, "with_close": 0}
        ps["games"] = sum(1 for g in idx.game_info.values() if g["season"] == season)
        for n, day in enumerate(days, 1):
            sb = _cached("espn_odds", "scoreboard", day, "scoreboard", resume,
                         lambda d=day: http.fetch_json(
                             ESPN_SCOREBOARD, params={"dates": d.replace("-", "")}), stats)
            if sb is None:
                continue
            batch = []
            for ev in espn_src.scoreboard_events(sb):
                stats["events"] += 1
                gid, how, _ = idx.match(ev["home"], ev["away"], ev["nba_day"])
                if gid is None or how.startswith("swapped"):
                    stats["unmatched_events"] += 1
                    unmatched.append({"source": "espn", "season": season, "date": ev["nba_day"],
                                      "home": ev["home"], "away": ev["away"], "reason": how,
                                      "detail": ev["espn_id"]})
                    continue
                stats["matched"] += 1
                eid = ev["espn_id"]
                payload = _cached("espn_odds", "odds", ev["nba_day"], eid, resume,
                                  lambda e=eid: http.fetch_json(ESPN_ODDS.format(eid=e)), stats)
                parsed = parse_espn_odds(payload) if payload else None
                if not parsed:
                    stats["no_odds"] += 1
                    continue
                stats["with_odds"] += 1
                ps["with_odds"] += 1
                ps["with_close"] += parsed["home_spread_close"] is not None
                batch.append({"game_id": gid, "source": "espn", **parsed})
            db.upsert_rows(conn, "closing_lines", batch)
            if progress and n % 20 == 0:
                print(f"espn {season}: {n}/{len(days)} days, {stats['requests']} requests",
                      file=sys.stderr, flush=True)
    _write_unmatched(unmatched, "espn_unmatched.csv")
    return stats


# ---------------------------------------------------------------------------- consensus

def consensus_line(conn: sqlite3.Connection, game_id: str) -> dict | None:
    """Best available closing line for a game: ESPN close for 2022-23+, else Kaggle.

    Returns {source, provider, home_spread, total, home_ml, away_ml, *_open} or None."""
    rows = {r["source"]: r for r in conn.execute(
        "SELECT cl.*, tg.season FROM closing_lines cl "
        "LEFT JOIN team_games tg ON tg.game_id = cl.game_id AND tg.is_home = 1 "
        "WHERE cl.game_id = ?", (game_id,))}
    if not rows:
        return None
    season = next((r["season"] for r in rows.values() if r["season"]), None)
    recent = season is not None and int(season[:4]) >= int(ESPN_FIRST_SEASON[:4])
    order = ["espn", "kaggle"] if recent else ["kaggle", "espn"]
    usable = [rows[s] for s in order if s in rows and (
        rows[s]["home_spread_close"] is not None or rows[s]["home_ml_close"] is not None)]
    if not usable:
        return None
    r = usable[0]
    return {"game_id": game_id, "source": r["source"], "provider": r["provider"],
            "home_spread": r["home_spread_close"], "total": r["total_close"],
            "home_ml": r["home_ml_close"], "away_ml": r["away_ml_close"],
            "home_spread_open": r["home_spread_open"], "total_open": r["total_open"],
            "home_ml_open": r["home_ml_open"], "away_ml_open": r["away_ml_open"]}


# ---------------------------------------------------------------------------- sanity

def ml_to_prob(ml: float) -> float:
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def devig_home_prob(home_ml: float, away_ml: float) -> float | None:
    if home_ml is None or away_ml is None or home_ml == 0 or away_ml == 0:
        return None
    ph, pa = ml_to_prob(home_ml), ml_to_prob(away_ml)
    return ph / (ph + pa)


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def sanity(conn: sqlite3.Connection, source: str | None = None, *,
           regular_only: bool = True) -> dict:
    """Per-season market sanity stats. source=None -> consensus_line (ESPN 2022-23+, else Kaggle).

    Keys per season: n, fav_win_rate, home_spread_mean, corr_spread_margin, n_ml, brier,
    logloss, plus an 'ALL_2012-13_2025-26' aggregate for the ML metrics."""
    sql = ("SELECT h.game_id, h.season, h.pts - a.pts AS margin FROM team_games h "
           "JOIN team_games a ON a.game_id = h.game_id AND a.is_home = 0 "
           "WHERE h.is_home = 1 AND h.pts IS NOT NULL AND a.pts IS NOT NULL")
    if regular_only:
        sql += " AND h.season_type = 'Regular Season'"
    games = conn.execute(sql).fetchall()
    if source:
        lines = {r["game_id"]: r for r in conn.execute(
            "SELECT game_id, home_spread_close AS home_spread, home_ml_close AS home_ml, "
            "away_ml_close AS away_ml FROM closing_lines WHERE source = ?", (source,))}
        get = lines.get
    else:
        def get(gid):
            return consensus_line(conn, gid)
    acc: dict[str, dict] = defaultdict(lambda: {"fav": [], "spread": [], "margin": [],
                                                "briers": [], "lls": []})
    for g in games:
        ln = get(g["game_id"])
        if not ln:
            continue
        a = acc[g["season"]]
        sp, margin = ln["home_spread"], g["margin"]
        if sp is not None:
            a["spread"].append(sp)
            a["margin"].append(margin)
            if sp != 0 and margin != 0:
                a["fav"].append(1 if (margin > 0) == (sp < 0) else 0)
        p = devig_home_prob(ln["home_ml"], ln["away_ml"])
        if p is not None and margin != 0:
            y = 1.0 if margin > 0 else 0.0
            a["briers"].append((p - y) ** 2)
            pc = min(max(p, 1e-9), 1 - 1e-9)
            a["lls"].append(-(y * math.log(pc) + (1 - y) * math.log(1 - pc)))

    def summ(a: dict) -> dict:
        mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
        c = _corr(a["spread"], a["margin"])
        return {"n": len(a["spread"]), "fav_win_rate": mean(a["fav"]),
                "home_spread_mean": mean(a["spread"]),
                "corr_spread_margin": None if c is None else round(c, 4),
                "n_ml": len(a["briers"]), "brier": mean(a["briers"]), "logloss": mean(a["lls"])}

    out = {s: summ(a) for s, a in sorted(acc.items())}
    agg = {"fav": [], "spread": [], "margin": [], "briers": [], "lls": []}
    for s, a in acc.items():
        if "2012-13" <= s <= "2025-26":
            for k in agg:
                agg[k] += a[k]
    out["ALL_2012-13_2025-26"] = summ(agg)
    return out


# ---------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m courtside.odds_history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kaggle", help="import the Kaggle closing-line dataset")
    k.add_argument("--force", action="store_true", help="re-download the zip")
    e = sub.add_parser("espn", help="ESPN open/close backfill (optional, resumable)")
    e.add_argument("--from", dest="start", default=ESPN_FIRST_SEASON)
    e.add_argument("--to", dest="end", default=None)
    e.add_argument("--no-resume", action="store_true", help="refetch even if archived")
    s = sub.add_parser("sanity", help="market sanity stats per season")
    s.add_argument("--source", choices=["kaggle", "espn"], default=None)
    args = ap.parse_args(argv)
    conn = db.connect()
    if args.cmd == "kaggle":
        res = import_kaggle(conn, force=args.force)
    elif args.cmd == "espn":
        res = backfill_espn(conn, _season_range(args.start, args.end or args.start),
                            resume=not args.no_resume)
    else:
        res = sanity(conn, args.source)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
