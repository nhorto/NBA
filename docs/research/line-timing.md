# Line timing: train on the close, run on an earlier line (IMPROVEMENTS G1)

*2026-10-23. Switch `COURTSIDE_NEWS_SINCE_LINE` (default OFF, version tag `+nsl`), code in
`src/courtside/model/line_news.py` and `predict.price_day`. Scripts: `research/g1_staleness.py`,
`research/g1_line_news_replay.py` (Mac scratch, `~/Developer/courtside-compute/g1/`),
`research/g1_line_news_eval.py`. Outputs: `var/tmp/g1/` on geekom1 (`eval.json`, `tables.md`,
`wf_*.csv.gz`, `staleness.json`, `days/`). Honest numbers only; see MODEL.md for the model.*

## 0. Summary

- **The opinion model's injury value explains line moves, but at less than half its size.**
  Regressing the ESPN open→close move on the change in `gbm_opinion`'s margin between the
  reports at the open and at the tip gives a slope of **0.45 (SE 0.05)** on 879 games, R² 0.13.
  The market moves about 0.45 pts for every point our model says the news is worth. That is 12 SE
  below 1. Totals show no relationship.
- **Live, the line is not stale when it is captured.** The pregame job captures odds and prices in
  the same run, so the median line is 1 minute old. What is stale is the *run*: a 10 AM line sits
  0.5 pts (mean |Δ spread|) from the last pre-tip line, and a line from 2–4 h before tip sits
  0.33 pts from it.
- **The switch does not clear any bar.** Fed the opening line, `gbm_market` loses 0.008 log loss
  to the same model fed the close. The news shift recovers −0.0003 (SE 0.0012) of that with the
  pre-registered open time, and −0.0010 (SE 0.0008) at best. The CRPS gain on margin is about
  1 SE. **Recommendation: keep it OFF.** See §4.

## 1. Data and method

- **Lines.** ESPN `closing_lines` (DraftKings): open and close spread, total and moneyline,
  3,865 games from 2023-24. ESPN's open has no timestamp.
- **Injuries.** The official report archive (`injury_snapshots`, `nba_official`), 22 Dec 2025 →
  13 Jun 2026, with `game_date` (migration 16). It has no batches between about 03:00Z and
  16:00Z, so the 03:00Z state also stands for a 10 AM ET run. The overlap is **899 games**
  (Regular Season 22 Dec → 12 Apr, plus the play-in and playoffs), and 879 have the state at
  every open proxy.
- **Replay.** `predict.predict_days` runs on a read-only scratch copy of the database (schema 16
  with the `game_date` backfill). It fits the production pair every day on games before the
  day, with frozen configs. `price_day` is replaced by a dumper that writes the fitted scorers,
  the day's feature rows, and the expected-lineup strengths from `injuries.availability` (the
  production function) at 8 cutoffs per game:
  - d1_17, d1_20, d1_23: 17:00, 20:00 and 23:00 UTC the day before
  - d0_03, d0_17, d0_21: 03:00, 17:00 and 21:00 UTC on the day
  - tip90: 90 minutes before tip
  - tip: the tip-off itself

  Each cutoff is capped at the game's tip. Two Mac lanes ran it in 68 min.
- **News value.** Δ = opinion margin (lineups at time B) − opinion margin (lineups at time A),
  holding everything else in the feature row fixed.
  - **raw:** production semantics. A team the newest batch does not list counts as healthy.
  - **guard:** a side counts only if the newest batch listed its team at both times. Otherwise it
    keeps its time-B lineup at time A too. This is what the switch does. Without it, a team that
    had not filed yet looks healthy at the open, and every standing absence turns into "news".
- **Open proxy.** **d0_03** (the evening before, about 10–11 PM ET) was pre-registered as the
  primary proxy before looking at results. The other proxies are sensitivity checks.
- **SEs.** Slopes use date-clustered sandwich SEs. Paired ON − OFF differences are reported with
  two SEs: date-clustered and two-way (date × home team, `proj.metrics.paired_cluster2`).

## 2. Phase 1: does our injury value explain line moves?

`move = close home-margin line − open home-margin line`. Δ is the opinion margin at the tip minus
at the open proxy. b is the slope through the origin; b (int) is the slope with an intercept.

| rule | open proxy | n | b (SE) | b with intercept (SE) | R² | corr | mean \|move\| | mean \|Δ\| | share \|Δ\| ≥ 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| raw | d1_17 | 881 | 0.27 (0.04) | 0.28 (0.04) | 0.10 | 0.31 | 1.52 | 1.76 | 0.72 |
| raw | d1_20 | 881 | 0.26 (0.04) | 0.27 (0.04) | 0.09 | 0.30 | 1.52 | 1.76 | 0.71 |
| raw | d1_23 | 881 | 0.30 (0.06) | 0.30 (0.06) | 0.08 | 0.28 | 1.52 | 1.33 | 0.59 |
| raw | d0_03 | 879 | 0.45 (0.05) | 0.45 (0.05) | 0.13 | 0.36 | 1.53 | 1.10 | 0.54 |
| raw | d0_17 | 899 | 0.44 (0.05) | 0.44 (0.05) | 0.10 | 0.32 | 1.53 | 1.00 | 0.51 |
| raw | d0_21 | 899 | 0.30 (0.15) | 0.30 (0.15) | 0.02 | 0.13 | 1.53 | 0.54 | 0.34 |
| guard | d1_17 | 881 | 0.44 (0.06) | 0.45 (0.06) | 0.07 | 0.26 | 1.52 | 0.47 | 0.19 |
| guard | d1_20 | 881 | 0.41 (0.07) | 0.41 (0.07) | 0.06 | 0.25 | 1.52 | 0.51 | 0.22 |
| guard | d1_23 | 881 | 0.44 (0.05) | 0.44 (0.05) | 0.11 | 0.33 | 1.52 | 0.93 | 0.45 |
| **guard** | **d0_03** | 879 | **0.45 (0.05)** | 0.45 (0.05) | 0.13 | 0.35 | 1.53 | 1.09 | 0.53 |
| guard | d0_17 | 899 | 0.50 (0.09) | 0.50 (0.09) | 0.06 | 0.23 | 1.53 | 0.63 | 0.38 |
| guard | d0_21 | 899 | 0.27 (0.16) | 0.27 (0.16) | 0.01 | 0.11 | 1.53 | 0.52 | 0.34 |

- **Totals:** the slope is −0.8 (SE 0.9) at d0_03 and indistinguishable from 0 at every proxy
  (R² ≈ 0). The opinion total barely responds to absences (mean |Δ| 0.07 pts), while totals move
  2.6 pts open→close. `BETA_TOTAL = 0`.
- **Reading.** Our news explains about 13% of the open→close variance, and the market prices it
  at 0.45 of our size. Three possible causes, none separable with these data:
  - Our absence values may be too large relative to the market's.
  - The open may already anticipate part of the reported news (a "questionable" player who was
    expected to sit).
  - Errors in our Δ shrink the slope toward 0: Questionable is priced at P = 0.5, and name
    matching can miss.

  The raw rule at early proxies has a much larger mean |Δ| (1.76) and a lower slope (0.27). That
  is the not-yet-filed artefact the guard removes. With the guard, the slope is stable at
  0.41–0.50 from d1_17 to d0_17.
- **Answer to G1's question:** the slope is not ≈ 1. It is 0.45.

## 3. How stale is the live line? (2026-27, prod snapshot of 23 Oct)

`research/g1_staleness.py` on the nightly prod copy, 88 pre-tip champion `gbm_market` rows on the
16 final games of 20–22 Oct 2026 (85 with a snapshot line). Columns:
- **age:** the row's created_at minus the line's `captured_at`.
- **moved, mean |Δ|:** the line used compared with the last pre-tip snapshot (what
  `market_line(gid, tip)` returns, captured a median 60 min before tip).

| run (hours to tip) | rows | games | line age, median (p90) | share moved | mean \|Δ spread\| (p90) | mean \|Δ total\| |
|---|---|---|---|---|---|---|
| 0–2 h | 21 | 16 | 0.8 min (1.5) | 0.00 | 0.00 (0.0) | 0.10 |
| 2–4 h | 12 | 11 | 1.2 min (1.4) | 0.33 | 0.33 (1.0) | 0.25 |
| 4–8 h | 18 | 16 | 0.6 min (1.3) | 0.28 | 0.39 (1.0) | 0.56 |
| 8–13 h (10 AM / 1:30 PM ET) | 18 | 15 | 1.1 min (2.1) | 0.39 | 0.50 (1.0) | 0.83 |
| 13 h+ (3 AM ET run) | 18 | 16 | 303 min (303) | 0.44 | 0.88 (2.5) | 1.33 (n = 3) |
| all | 88 | 16 | 1.1 min (303) | 0.27 | 0.40 (1.0) | 0.47 |

For comparison, ESPN open→close on the 2025-26 games above moves 1.53 pts on average.

- The line is captured in the same run as the injury report, so "news since the line" is live
  only for the minutes between the two captures, plus the 03:00 ET run (a 5-hour-old line). The
  live problem is that a morning run is hours before the close, and its line and its reports are
  equally early. A shift computed from news since the line cannot fix that.
- The sample is three days. Re-run once 2026-27 has a month of snapshots.

## 4. Phase 2: ON vs OFF, walk-forward

- **Setup.** The "line available at prediction time" is the **ESPN opening line**: no intraday
  snapshots exist for 2025-26, and 2026-27 has three days.
  - `gbm_market` is fed the open spread and total, with injury lineups at the run time T (the
    tip, or 21:00Z, about 4–5 PM ET).
  - **OFF:** what production does with that line today.
  - **ON:** the open line shifted by b_d × Δ(T vs the open proxy, guarded). b_d is fit on games
    of earlier days only: OLS through the origin, expanding window, 0 until 150 games, clipped
    to [0, 1.5]. It ends at 0.45.
- **References:**
  - **close-fed:** the same model fed the close (what replays assume).
  - **open ML, close ML:** the de-vigged moneylines.
- **Scores:**
  - Log loss and Brier.
  - CRPS of margin under Normal(μ, the day's `oof_resid_sd_`).
  - CRPS of total under Normal(μ, sd of OFF total residuals on earlier days). Both arms use the
    same σ.
- **SEs** are date-clustered / two-way (date × home team).

| scenario (open proxy → run time) | n | mean \|shift\| | LL OFF | LL ON | ON − OFF LL | ON − OFF Brier | ON − OFF CRPS margin | ON − OFF CRPS total | LL close-fed | LL open ML | LL close ML |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **d0_03 → tip** (pre-registered) | 879 | 0.41 | 0.5764 | 0.5761 | **−0.00033 (0.00124 / 0.00126)** | −0.00021 (0.00049 / 0.00051) | −0.013 (0.011 / 0.012) | +0.0003 (0.0016 / 0.0019) | 0.5681 | 0.5773 | 0.5656 |
| d0_03 → tip, Regular Season | 791 | 0.43 | 0.5655 | 0.5647 | −0.00072 (0.00133 / 0.00135) | −0.00040 (0.00053 / 0.00055) | −0.019 (0.012 / 0.014) | +0.0002 (0.0018 / 0.0021) | 0.5580 | 0.5674 | 0.5563 |
| d0_03 → tip, Playoffs | 82 | 0.19 | 0.6713 | 0.6712 | −0.00012 (0.00129 / 0.00115) | +0.00006 | +0.015 (0.014 / 0.015) | +0.0009 | 0.6547 | 0.6615 | 0.6434 |
| d1_17 → tip | 881 | 0.20 | 0.5772 | 0.5784 | +0.00120 (0.00085 / 0.00092) | +0.00045 | +0.004 (0.008 / 0.008) | −0.0013 | 0.5688 | 0.5780 | 0.5663 |
| d1_20 → tip | 881 | 0.18 | 0.5772 | 0.5783 | +0.00108 (0.00078 / 0.00080) | +0.00038 | +0.004 (0.007 / 0.007) | +0.0013 | 0.5688 | 0.5780 | 0.5663 |
| d1_23 → tip | 881 | 0.35 | 0.5772 | 0.5768 | −0.00037 (0.00115 / 0.00113) | −0.00022 | −0.009 (0.010 / 0.011) | +0.0035 | 0.5688 | 0.5780 | 0.5663 |
| d0_17 → tip | 899 | 0.27 | 0.5757 | 0.5747 | −0.00097 (0.00082 / 0.00082) | −0.00042 | −0.012 (0.009 / 0.009) | +0.0005 | 0.5673 | 0.5766 | 0.5646 |
| d0_21 → tip | 899 | 0.13 | 0.5757 | 0.5754 | −0.00028 (0.00044 / 0.00043) | −0.00013 | −0.001 (0.005 / 0.006) | +0.0016 | 0.5673 | 0.5766 | 0.5646 |
| d0_03 → 21:00Z run | 879 | 0.29 | 0.5772 | 0.5773 | +0.00012 (0.00099 / 0.00101) | −0.00001 | −0.010 (0.008 / 0.008) | −0.0003 | 0.5686 | 0.5773 | 0.5656 |
| d0_17 → 21:00Z run | 899 | 0.12 | 0.5763 | 0.5755 | −0.00088 (0.00048 / 0.00056) | −0.00040 | −0.011 (0.005 / 0.005) | −0.0006 | 0.5677 | 0.5766 | 0.5646 |

Full tables, including every proxy at the 21:00Z run and the comparisons with the open
moneyline, are in `var/tmp/g1/tables.md`.

- **The stale-line cost is real.** The same model fed the open instead of the close loses
  **0.0083 LL** (0.5764 vs 0.5681). Fed the open, `gbm_market` is slightly better than the open
  moneyline itself (−0.0009, SE 0.0024).
- **The news shift recovers little of it.** At the pre-registered proxy it recovers 0.0003 of
  the 0.0083 (4%). The best cell (d0_17 → the 21:00Z run) is −0.0009 at 1.6 two-way SE, with
  margin CRPS −0.011 at 2.1 SE. That cell was picked after looking at 10, so it is not
  confirmatory. Using an early proxy makes things *worse* (+0.0011 to +0.0012, about 1.3 SE),
  because a wrong capture time turns old news into a shift. Live, the capture time is known, so
  that failure mode does not apply there. But live, the news since capture is also near zero
  (§3).
- **Totals:** no gain anywhere. β_total = 0, so ON totals differ from OFF only through the
  margin shift's effect on the trees.
- **Not every season is tested.** The injury archive is only 2025-26, so this is one partial
  season. The usual "no season worse by > 1 SE" rule cannot be checked.

## 5. Implementation

- **Capture time.** `predict.market_line` already returns each snapshot's `captured_at`.
  `price_day` now reads it next to the line it feeds the model (`line_news.capture_time`):
  snapshot lines only, since a closing line has no capture time.
- **ON** (`predict._news_since_line`):
  1. The capture-time state is `injuries.availability` at min(capture time, the game's injury
     cutoff). Lineups come from `production.lineup_strengths` with the day's engine state, and
     the guard is applied.
  2. Δ is the opinion margin (and total) now minus at capture, on the same feature rows.
  3. The shift is `BETA_MARGIN` = 0.45 times the margin Δ (and `BETA_TOTAL` = 0 times the total
     Δ). It is added to `mkt_margin` / `mkt_total` for the market scorers only; the opinion
     model sees unshifted features.
  4. Market rows get `model_version + "+nsl"` and `inputs.news_since_line` (line_at,
     injuries_at_line, sides, deltas, shifts, betas).
  5. With no news since the capture, the two lineups are computed by the same function, so the
     shift is exactly 0.
- **OFF:** `price_day` never calls the new code; a test monkeypatches it to raise. The OFF output
  is **byte-identical to HEAD's `predict.py`**: stripped rows of three synthetic worlds (shift,
  guard, closing fallback) and a real past day (2026-01-15, 9 games, official report priced)
  were dumped as JSON under both versions and matched with `cmp`. `tests/test_news_since_line.py`
  (10 tests) covers:
  - OFF determinism, and that OFF never enters the new code
  - ON with β = 0 equals OFF
  - ON shift = β × the opinion news, and the headline equals OFF fed the shifted line
  - the guard
  - no shift before any report or after the news
  - closing lines are never moved

## 6. Recommendation

1. **Keep `COURTSIDE_NEWS_SINCE_LINE` OFF.** With the pipeline as it runs, live news since line
   capture is minutes. The backtest gain, under the most favourable stale-line framing, is
   ≤ 1.6 SE on one partial season.
2. **The live gap is run timing, not line staleness.** The last pregame run (≤ 2 h before tip)
   already prices a line within 0 pts of the last snapshot. If the Brief's headline should
   match the close more closely, re-price closer to tip, or show the newest pre-tip row, which
   `latest_pretip` already does. Do not extrapolate the close.
3. **Revisit** once 2026-27 has about 60 days of intraday snapshots. Then score the 10 AM run
   against the last pre-tip snapshot with the real capture times. The scripts take
   `--db`/`--days`, so this is a rerun.
4. **Our absence values run about 2× the market's** (slope 0.45). That is worth a separate look
   on the opinion side, under G2's "no edge vs the close" framing.

## Reproduce

```
# Mac scratch (2 lanes, nice 10; read-only DB with schema 16 + game_date backfill)
run.sh code/research/g1_line_news_replay.py --db <scratch-gd.db> --start 2025-12-22 --end 2026-02-28 --out out/days
run.sh code/research/g1_line_news_replay.py --db <scratch-gd.db> --start 2026-03-01 --end 2026-06-30 --out out/days
# geekom1 (light)
uv run python research/g1_line_news_eval.py --days var/tmp/g1/days --db var/courtside.db \
    --min-fit 150 --out var/tmp/g1/eval.json --md var/tmp/g1/tables.md
uv run python research/g1_staleness.py --db <prod copy> --out var/tmp/g1/staleness.json
```
