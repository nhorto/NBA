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
| `/tonight`, `/tonight/<day>` | Tonight's players: projections, availability and prop cards (`players_view.py`); `/tonight` redirects to today. Was `/players/<day>`, which now 301-redirects here (query kept) |
| `/games`, `/games/<day>` | stats dashboard: the scoreboard for a night (default: the latest night with finals) (`stats_games.py`) |
| `/teams` | stats dashboard: league table of all 30 teams, `?season=` (`stats_teams.py`) |
| `/team/<ABBR>` | stats dashboard: team page; lower-case codes 301 to upper case, unknown codes 404 (`stats_teams.py`) |
| `/players` | stats dashboard: player directory; `?q=` is also the no-JS fallback of the top-bar search (`stats_players.py`) |
| `/player/<player_id>` | stats dashboard: player page; unknown or non-numeric ids 404 (`stats_players.py`) |
| `/leaders` | stats dashboard: league and team leaders, `?season=` (`stats_leaders.py`) |
| `/api/search?q=` | JSON for the top-bar search: `{"q", "results": [{type, id, name, team, url, sub, ...}]}`, players of the last 3 seasons plus the 30 teams, accent-insensitive, at most 10 (`stats/search.py`) |
| `/health` | Model Health: status light, per-model scoreboard vs the market, rolling log-loss/accuracy, calibration, over/under-prediction, drift, alerts, data checks and model history (reads the MLOps tables; empty states when they are empty) |
| `/api/health` | JSON summary of the same: status, champions, alerts, models, calibration, bias, drift, checks |
| `/healthz` | `{"ok", "status", "briefs", "latest_day", "latest_built_at", "latest_age_hours", "data_dir"}` for monitoring |
| `/api/brief/<day>` or `/api/brief/latest` | the stored Brief JSON, unchanged |

The site menu is Today · Tonight · Games · Teams · Players · Leaders · DFS · Health, plus a
global search box ("/" focuses it). The stats dashboard pages (spec: `docs/STATS_SITE.md`) use
the dark look in `static/pro.css`, the macros in `templates/_stats.html`, `static/stats.js`
(sortable tables, segmented controls, tooltips, search) and the query layer in
`src/courtside/stats/`. Their content is being built; the routes above render placeholders
until then.

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

## The home page is a brief, not a dump

The home page should give someone the whole night in roughly 6–10 phone screens. Depth lives on
`/game/<id>`, or behind `<details>`/`<summary>` disclosures that work without JS. All
collapsed content is still in the HTML. Measured on the 2026-02-20 Brief with everything
collapsed: **phone (390 px) 7,945 px** (was ~29,600) and **desktop (1440 px) 4,574 px** (was
~15,300).

- **Masthead.** The headline, a four-number strip (games, picks, tonight, season), and **the
  60-second version**: 3–5 takeaways written deterministically from the facts by
  `view.sixty_view`. They are the game of the night, the strongest team trend (or a hot player),
  the biggest injury plus a high-impact new absence, tonight's marquee game (best combined power
  rank and closest odds, skipping the upset game), and the top upset-watch game. A Scorecard
  line fills in when there are fewer than five. **Writer hook:** if the Brief carries
  `sixty_seconds: {"bullets": [str, ...]}`, those bullets replace the deterministic ones. The
  old "what you need to know" dek is now the page's `<meta name="description">`.
- **Last Night is ordered by interest** (`view.interest`). When the analytics provide them,
  `facts.excitement` (sum of |ΔWP|) and `facts.comeback_wp.min_wp` drive the order. Otherwise
  margin, overtime and the score comeback do. Upsets (the market favourite lost, or our pick
  missed) and 40/50-point nights or triple-doubles add to the score. Cards get up to two tags
  (Thriller, Overtime, Comeback, Upset, 40-point night...). Layout by tier (`view.card_tiers`):
  - The top four competitive games get full cards. If there are five or fewer, all of them do.
  - The remaining competitive games become compact rows under **Also last night**.
  - Wins by 20+ points, without a star line, become compact rows under **Decided early**.
- **Compact card:** score, our pick ✓/✗, the headline, and the story *lead*. The lead is the
  first sentence, plus the second when the first is short; `view.story_lead` only splits on
  sentence boundaries. Then two beat lines: the turning point, and the featured player. "Read
  more" opens the rest of the story, the full turning point, the hero's "why" chips, clutch,
  notable lines, the betting result and incidents. "Full breakdown →" goes to `/game/<id>`.
- **Unsung-hero guard** (`view.hero_qualifies`). Single-game on/off is noisy. The home card only
  features a candidate with **≥ 15 minutes AND (pts + reb + ast + stocks ≥ 10 OR plus-minus
  ≥ +8)**. It takes the first candidate that qualifies; if none does, it shows the winning side's
  top performer, labelled "Top line". The game page still lists every candidate.
- **Team Pulse:** the three strongest trends (by |z|) as rows. Each row shows record, last 10,
  streak, net rating and a one-line why, and opens to show the dumbbell, splits, blurb and
  absences. "See all" holds the rest. The power ranking is a top-10 strip that scrolls sideways
  on phones. Streaks and movers are chips.
- **Player Pulse:** the top three hot and top three cold; "See all" holds the rest. Leaders, role
  changes, streaks and career highs sit in one disclosure.
- **Injury Watch:** up to five impact players out tonight (Courtside Impact ≥ 10, or ≥ 24 mpg
  when impact is unknown), and the three biggest changes since yesterday. The full lists are
  behind "See all".
- **Tonight:** one row per game: matchup, our win-probability bar with the market tick, upset
  watch, tip-off, the line, and one key factor (the side missing the most minutes, else
  schedule). Tap a row to see our line vs the market's, the edge, the preview, every factor and
  form.
- **Scorecard:** one card with the record, Brier vs market, last night, a hit-rate-by-confidence
  chart and a one-line verdict. The calibration chart, its table and the full us-vs-market table
  are collapsed.
- **Section nav:** a sticky horizontal scroller (fades at the right edge on phones). It
  highlights the section in view, and a link to a collapsed game opens it.

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
- **JS** (`static/app.js`) does four things: the theme toggle, accessible tabs (last-10
  leaders), opening a collapsed row when a link targets it, and highlighting the active section
  in the sticky section nav. Everything still works without JS: every collapsible is a native
  `<details>`, and a summary is a keyboard-focusable control with a visible focus ring.
  Chevron transitions are disabled under `prefers-reduced-motion`.

## Visual QA

Screenshots come from a headless Chromium that Playwright launches itself. It never attaches to
an existing browser. The shots are taken at 390×844 and 1440×900, in light and dark, and saved to
`var/screenshots/` (`var/screenshots/v2/` for the compact home page: collapsed, one card
expanded, and an empty off day). Run your QA server on a spare port (for example
`COURTSIDE_DATA=/tmp/qa uv run courtside serve --port 8790`) so you don't disturb a running
instance.

```bash
uv run --with playwright python -m playwright install chromium-headless-shell
# then drive http://127.0.0.1:<port>/ with a short sync_playwright script (color_scheme="dark"|"light")
```
