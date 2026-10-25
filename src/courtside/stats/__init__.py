"""Query layer for the pro stats pages (docs/STATS_SITE.md).

Pure functions that take a `sqlite3.Connection` and return plain dicts/lists: no FastAPI, no
templates. The web routes live in `courtside.web.stats_*`.

  db.py       read-only connections + `cached(key, fn, conn_or_path)` (mtime/size keyed memo)
  seasons.py  current season, the 3-season window, labels, season types
  search.py   the global search box (`/api/search`): players of the last 3 seasons + 30 teams
"""
