"""sportsdataverse-data bulk downloads (GitHub releases): lineups, possessions, officials.

Release tags (one per dataset; each holds one file per season in csv / parquet / rds, 1997-2026):
  nba_stats_game_lineups  nba_lineups_<Y>.parquet     on-court 5v5 per play-by-play action
  nba_stats_possessions   nba_possessions_<Y>.parquet possession-level rows
  nba_stats_officials     officials_<Y>.parquet       referees per game
``<Y>`` is the season's END year (2023-24 -> 2024). Only the parquet files are fetched (about
120 MB for all three; csv/rds would add ~300 MB). Files land in
``<data>/raw/sportsdataverse/<tag>/<asset>`` and are skipped when already present at the right size.

Nothing is loaded into SQLite: use ``read_lineups(season)`` etc. (pandas + pyarrow).
Asset lists come from the GitHub API (api.github.com/repos/sportsdataverse/sportsdataverse-data/
releases/tags/<tag>), so the real names are never hardcoded.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Callable

from courtside import config
from courtside.http import fetch_bytes, fetch_json

REPO = "sportsdataverse/sportsdataverse-data"
API = f"https://api.github.com/repos/{REPO}/releases/tags"
TAGS = {"lineups": "nba_stats_game_lineups", "possessions": "nba_stats_possessions",
        "officials": "nba_stats_officials"}
FIRST_SEASON_END, LAST_SEASON_END = 1997, 2026


def root() -> Path:
    return config.data_dir() / "raw" / "sportsdataverse"


def asset_path(tag: str, asset: str) -> Path:
    return root() / tag / asset


def season_end_year(season: str | int) -> int:
    """'2023-24' -> 2024; 2024 -> 2024; '2024' -> 2024."""
    if isinstance(season, int):
        return season
    m = re.fullmatch(r"(\d{4})-(\d{2})", season)
    if m:
        return int(m[1]) + 1
    if re.fullmatch(r"\d{4}", season):
        return int(season)
    raise ValueError(f"invalid season {season!r}, expected '2023-24' or 2024")


def list_assets(tag: str, *, get: Callable = fetch_json) -> list[dict]:
    """Parquet assets of a release: [{name, size, url}], sorted by name."""
    rel = get(f"{API}/{tag}", headers={"Accept": "application/vnd.github+json"})
    return sorted(({"name": a["name"], "size": a["size"], "url": a["browser_download_url"]}
                   for a in rel.get("assets", []) if a["name"].endswith(".parquet")),
                  key=lambda a: a["name"])


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def download(datasets=tuple(TAGS), since: int = FIRST_SEASON_END, until: int = LAST_SEASON_END,
             force: bool = False, *, get: Callable = fetch_json,
             get_bytes: Callable = fetch_bytes, log: Callable[[str], None] = print) -> dict:
    """Download the parquet assets (season end years since..until) of each dataset.

    Returns {"downloaded", "skipped", "failed", "bytes", "errors"}. Re-runs skip files whose size
    matches the release metadata; a size mismatch after download counts as a failure."""
    out: dict = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0, "errors": []}
    for name in datasets:
        tag = TAGS.get(name, name)
        assets = list_assets(tag, get=get)
        log(f"{tag}: {len(assets)} parquet assets")
        for a in assets:
            m = re.search(r"_(\d{4})\.parquet$", a["name"])
            if not m or not (since <= int(m[1]) <= until):
                continue
            path = asset_path(tag, a["name"])
            if path.exists() and path.stat().st_size == a["size"] and not force:
                out["skipped"] += 1
                continue
            try:
                data = get_bytes(a["url"], timeout=180.0)
                if len(data) != a["size"]:
                    raise ValueError(f"size {len(data)} != expected {a['size']}")
                _atomic_write(path, data)
                out["downloaded"] += 1
                out["bytes"] += len(data)
            except Exception as e:  # noqa: BLE001 - keep going; a rerun resumes
                out["failed"] += 1
                out["errors"].append(f"{a['name']}: {type(e).__name__}: {e}")
                log(f"  FAILED {a['name']}: {e}")
    return out


def row_count(tag_or_name: str, season: str | int) -> int:
    """Row count from the parquet footer (no data read)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(_season_file(tag_or_name, season)).metadata.num_rows


def _season_file(name: str, season: str | int) -> Path:
    tag = TAGS.get(name, name)
    y = season_end_year(season)
    stem = {"nba_stats_game_lineups": "nba_lineups", "nba_stats_possessions": "nba_possessions",
            "nba_stats_officials": "officials"}[tag]
    p = asset_path(tag, f"{stem}_{y}.parquet")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run sportsdataverse.download()")
    return p


def _read(name: str, season: str | int, columns=None):
    import pandas as pd
    return pd.read_parquet(_season_file(name, season), columns=columns)


def read_lineups(season: str | int, columns=None):
    """Lineups for one season ('2023-24' or end year 2024) as a DataFrame.

    One row per play-by-play action: game_id (10 chars, e.g. '0022300061'), action_number,
    period, home_player_1..5 / away_player_1..5 (the five on court after that action), season."""
    return _read("lineups", season, columns)


def read_possessions(season: str | int, columns=None):
    return _read("possessions", season, columns)


def read_officials(season: str | int, columns=None):
    return _read("officials", season, columns)


def compare_with_stints(sdv_game, events, stints) -> dict:
    """Agreement between sportsdataverse lineups and our stints for ONE game.

    ``sdv_game``: the rows of ``read_lineups`` for the game; ``events``/``stints`` from
    ``analytics.pbp.normalize`` / ``analytics.stints.build_rotation(...).stints``. Rows are matched
    on ``action_number``; an action agrees when both five-man sets (home and away) are identical.
    Returns counts for all matched actions and for non-substitution actions only, plus a few
    disagreeing examples."""
    from courtside.analytics.stints import lineup_at
    # ~5% of sportsdataverse rows are exact duplicates (same game_id + action_number).
    sdv_game = sdv_game.drop_duplicates("action_number")
    by_action = {int(r.action_number): r for r in sdv_game.itertuples(index=False)}
    res = {"sdv_rows": len(sdv_game), "events": len(events), "matched": 0, "agree": 0,
           "matched_nonsub": 0, "agree_nonsub": 0, "no_stint": 0, "examples": []}
    for e in events:
        r = by_action.get(e.action_number)
        if r is None:
            continue
        st = lineup_at(stints, e.idx)
        if st is None:
            res["no_stint"] += 1
            continue
        theirs_h = tuple(sorted(int(getattr(r, f"home_player_{i}")) for i in range(1, 6)))
        theirs_a = tuple(sorted(int(getattr(r, f"away_player_{i}")) for i in range(1, 6)))
        ok = theirs_h == tuple(st.home) and theirs_a == tuple(st.away)
        res["matched"] += 1
        res["agree"] += ok
        if not e.is_sub:
            res["matched_nonsub"] += 1
            res["agree_nonsub"] += ok
        if not ok and len(res["examples"]) < 5:
            res["examples"].append({"action": e.action_number, "period": e.period,
                                    "clock": e.clock, "is_sub": e.is_sub,
                                    "ours": (st.home, st.away), "theirs": (theirs_h, theirs_a)})
    for k in ("", "_nonsub"):
        m = res[f"matched{k}"]
        res[f"rate{k}"] = round(res[f"agree{k}"] / m, 4) if m else None
    return res
