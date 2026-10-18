"""Play-by-play analytics: events, lineups/stints, game story, and the per-game facts document.

All functions are pure: they take raw cdn.nba.com payloads (or objects derived from them) and never
touch the network or the DB. See docs/ANALYTICS.md for definitions and thresholds.
"""
