# Courtside: the NBA Morning Brief

A box score is a snapshot. Courtside tells you what last night meant and what to expect tonight:

- **Last night:** each game's story, the turning point (the biggest win-probability swing), the role player who quietly won it, clutch play, and our pregame pick against the result.
- **Team and player pulse:** who's rising or sliding, who's hot or cold against their own baseline, role changes, and the leaders over the last 10 games. Every flag is calibrated against history so noise doesn't make the cut.
- **Injury watch:** new absences, returns, and how they move tonight's odds.
- **Tonight:** win probability, projected margin and total, our independent number against the market, and the key factors (absences, rest, travel).
- **Scorecard:** our season record and calibration, shown openly next to the market's.

It runs locally, with an always-on Mac mini and a Linux box as backup, and is served over Tailscale.

## How good is the model?

Results are walk-forward by season. Regular season, 2012-13 through 2025-26:

| Model | Log-loss | Brier | Accuracy |
|---|---|---|---|
| Elo + schedule | 0.6131 | 0.2126 | 66.1% |
| Player-based strength (leak-safe) | 0.6029 | 0.2081 | 67.0% |
| **gbm_opinion** (our independent number) | 0.6022 | 0.2079 | 67.2% |
| **gbm_market** (best estimate) | 0.5933 | 0.2041 | 68.1% |
| Vegas closing line | 0.5918 | 0.2035 | 68.1% |

Our best estimate ties the closing line, and we have no betting edge (about 50% against the spread). The Brief presents "our number vs the market" as information, not advice. See [docs/MODEL.md](docs/MODEL.md).

## Quick start

```bash
uv sync
uv run courtside db migrate
uv run courtside backfill gamelogs --from 2000-01 --to 2025-26   # ~10 min, one-time
uv run courtside collect pregame          # snapshots odds + injuries, predicts today, builds the Brief
uv run courtside collect postgame         # archives last night, computes game facts, builds tomorrow's Brief
uv run courtside serve                    # http://127.0.0.1:8765
uv run pytest -q
```

To run it on a schedule, see [deploy/README.md](deploy/README.md) (launchd on macOS, systemd on Linux). All state lives under `COURTSIDE_DATA`, which defaults to `./var/`.

## Docs

- [DESIGN](docs/DESIGN.md): the product spec and architecture
- [MODEL](docs/MODEL.md): the prediction ladder, backtests, and market comparison
- [ANALYTICS](docs/ANALYTICS.md): play-by-play, stints, runs, clutch, impact, and win probability
- [TRENDS](docs/TRENDS.md): team and player pulse, and the noise calibration
- [BRIEF](docs/BRIEF.md): the Brief JSON schema
- [STYLE](docs/STYLE.md): the writer's house style
- [WEB](docs/WEB.md): the web app

The original 2023-24 notebook project lives in [legacy/](legacy/).

**Data:** NBA.com and ESPN public endpoints for personal, local use; historical lines from a CC0 Kaggle dataset. A paid public product would need licensed data feeds.
