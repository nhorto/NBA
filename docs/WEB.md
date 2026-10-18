# Courtside web front end

A server-rendered, mobile-first reader for the daily Brief. It uses FastAPI and Jinja2, with one
hand-written stylesheet and about 100 lines of vanilla JS. There is no build step, no Node, no
CDN and no tracker: fonts, CSS, JS and charts are all served by the app.

Code: `src/courtside/web/`. Tests: `tests/test_web.py`.

## Run it

```bash
uv run courtside serve                          # http://127.0.0.1:8765
uv run courtside serve --host 0.0.0.0 --port 8765
uv run courtside serve --reload                 # dev: reload on .py/.html/.css/.js changes
```

`courtside serve` is handled before the main argparse tree, so it doesn't appear in
`courtside --help`. Use `courtside serve --help` for its options.

- The app reads Briefs from `$COURTSIDE_DATA` (default `<repo>/var`): `briefs/<day>.json` plus
  the `briefs` table in `courtside.db`. The DB is opened **read-only** and never migrated, so it
  is safe to run the server while collectors write. When a day exists in both places, the newer
  `meta.built_at` wins.
- It runs as a single process. Parsed Briefs are cached by file mtime and size (or by the DB
  `built_at`), and so are the derived view models. A warm page renders in about 2–10 ms; the
  first hit after start-up is about 80 ms because the templates compile. Responses carry a
  `Server-Timing: app;dur=…` header and are gzipped.
- Rebuilding a Brief (`courtside brief build`) shows up on the next request. No restart is needed.

## Routes

| route | what |
|---|---|
| `/` | the latest stored Brief |
| `/brief/<YYYY-MM-DD>` | the Brief for a day, with prev/next navigation |
| `/archive` | every stored Brief, grouped by month |
| `/game/<game_id>` | deep-dive for a finished game (linescore, lead-margin chart, runs, clutch, impact table, unsung-hero candidates, notable lines, incidents); a preview page if the id is only on a "tonight" slate |
| `/healthz` | `{"ok", "status", "briefs", "latest_day", "latest_built_at", "latest_age_hours", "data_dir"}` for monitoring |
| `/api/brief/<day>` or `/api/brief/latest` | the stored Brief JSON, unchanged |

## View it from your phone over Tailscale

The server binds to `127.0.0.1` by default, which only the host itself can reach. Pick one of
these options:

1. **`tailscale serve`** (recommended: HTTPS, tailnet-only, and the app stays on localhost):

   ```bash
   uv run courtside serve                       # keep the default 127.0.0.1:8765
   tailscale serve --bg 8765                    # https://<host>.<tailnet>.ts.net/
   tailscale serve status                       # show the URL
   tailscale serve reset                        # stop serving
   ```

2. **Bind to the tailnet address** (plain HTTP, reachable from tailnet devices only):

   ```bash
   uv run courtside serve --host "$(tailscale ip -4)"     # http://100.x.y.z:8765/
   ```

   Avoid `--host 0.0.0.0` on untrusted networks: it also exposes the app on your LAN.

On iOS, Safari's *Share → Add to Home Screen* gives the Brief an app-like icon. The pages respect
the phone's light/dark setting; the toggle in the top bar pins a theme, and that choice is saved
per device.

To keep it running, use a `systemd --user` service (Linux) or a launchd agent (macOS) running
`uv run courtside serve`, alongside the collector timers in `deploy/`.

## Design notes

- **Type.** Noto Serif Display Bold carries the editorial voice: masthead, section titles, card
  headlines. Inter (variable) carries the UI and the data, with tabular figures wherever numbers
  line up. Both fonts are self-hosted WOFF2 files, subset to Latin (`static/fonts/`, SIL OFL; see
  `OFL.txt`). The type scale is fluid (`clamp()`), with a 16 px body.
- **Colour.** A warm paper background with white cards (light mode), and near-black with
  graphite cards (dark mode). Each theme has its own palette, not an automatic inversion. Text
  tokens meet WCAG AA on their surfaces. One accent (court orange) is used for brand, upset
  watch and links. Green and red are reserved for good/bad signals (hot/cold, rising/sliding,
  right/wrong picks), and a text label always accompanies them. Team identity comes from tricode
  badges in each team's primary colour with a secondary-colour stripe (`web/teams.py`). No logos
  are used.
- **Charts** are inline SVG generated on the server (`web/charts.py`): win-probability bars with
  a market tick, the calibration plot (with a table view), "then → now" dumbbells for player and
  team form, and a lead-margin chart. The lead chart uses `facts.score_timeline` when it exists;
  otherwise it rebuilds the line from score checkpoints (quarter ends, runs, largest leads,
  comeback, clutch plays) and says so in its caption. The game page ships a wide and a narrow
  SVG so that labels stay legible on both phone and desktop.
- **Null text.** Every writer slot is optional. When a slot is `null`, `web/view.py` builds a
  plain fallback from the structured facts only. For example, the card headline is "Clippers edge
  Nuggets behind Mathurin's 38", and the story is the turning-point run, its time and its effect
  on the score, plus the top lines. The masthead headline comes from the most notable game; the
  "what you need to know" line comes from the counts, the pick record, upset watch and the
  biggest absence. A test checks that every number in a fallback story appears in the facts.
- **Empty states.** Every section has its own message (off day, early season, no injury
  snapshots, no graded picks). Sub-panels with nothing to show are dropped, not drawn as empty
  boxes.
- **JS** (`static/app.js`) does three things: the theme toggle, accessible tabs (last-10
  leaders), and highlighting the active section in the sticky section nav. Everything still
  works without JS (`<details>` for collapsibles).

## Visual QA

Screenshots come from a headless Chromium that Playwright launches itself. It never attaches to
an existing browser. The shots are taken at 390×844 and 1440×900, in light and dark, and saved to
`var/screenshots/`.

```bash
uv run --with playwright python -m playwright install chromium-headless-shell
# then drive http://127.0.0.1:<port>/ with a short sync_playwright script (color_scheme="dark"|"light")
```
