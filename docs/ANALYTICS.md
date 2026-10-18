# Courtside analytics: play-by-play core

The engine behind the Brief's *turning point*, *unsung hero*, clutch play, runs and standout lines.
Code: `src/courtside/analytics/`. Everything is a pure function of the raw cdn.nba.com
`boxscore_{id}.json` and `playbyplay_{id}.json` payloads: no DB, no network.

```
pbp.normalize(pbp, box)            -> list[Event]
stints.build_rotation(events, box) -> Rotation(stints, starters, anomalies)
stints.validate_stints(box, stints)-> ValidationReport          (the correctness gate)
game_story.*                       -> runs, leads, comeback, clutch, garbage time, impact, turning point
summarize.game_facts(box, pbp)     -> dict (JSON facts document for the Brief writer)
```

## 1. Events (`pbp.py`)

- Order is by `orderNumber`, the feed's canonical order. `actionNumber` is not monotonic after edits.
- `clock` is the number of seconds remaining in the period. `elapsed` is game seconds since tip-off.
  Periods 1–4 are 720 s each and overtime periods are 300 s, so 2OT ends at 3480 s.
- `points` is derived from the shot type and result: a made 2pt is 2, a made 3pt is 3, a made FT is 1.
  `score_home` and `score_away` are the feed's running score *after* the event. Tests check that the
  two always agree.
- `heave` actions (team end-of-period heaves) are **not** field-goal attempts. The boxscore doesn't
  count them either.
- Steals and blocks are credited from the separate `steal` and `block` actions. Their `teamId` is the
  defending team.

### Possessions (counting method)

Possessions are **counted**, not estimated. An event ends a possession (`ends_possession`, with the
offense in `poss_side`) when it is:

| event | rule |
|---|---|
| made FG | ends it, unless it's an and-one (a non-technical "1 of 1" FT by the same team follows at the same clock) |
| made last FT of a trip | ends it, unless the trip keeps possession (technical, flagrant, transition take, away-from-play, clear path) |
| defensive rebound | ends it if it rebounds a missed FG or a missed *final* FT. Team rebounds count |
| turnover | always ends it, for player and team turnovers |
| period end | ends it if the offense did something since the last possession end (shot, foul, OREB, heave, jump ball). The owner comes from the feed's `possession` field |

Check: in every fixture, home and away counts are within 3 of each other, and within 8% of the box
estimate `FGA − OREB + TOV + 0.44·FTA`. The estimate runs about 4% high, which is expected.

## 2. Stints and lineups (`stints.py`)

A **stint** is a stretch with no substitutions. It runs between substitution *blocks* (runs of
consecutive sub events) and is cut at period boundaries. Each stint records:
`period, start, end (elapsed s), home[5], away[5], home_pts, away_pts, home_poss, away_poss`.

**Period starters:**
1. **Period 1** uses the boxscore `starter == "1"` players.
2. **Later periods** use first evidence within the period. A player whose first appearance is `SUB in`
   started on the bench. A player whose first appearance is `SUB out` or any on-court action (shot,
   rebound, foul, turnover, assist, steal, block, foul drawn, jump ball) started on the floor.
   Technical fouls and ejections don't count as evidence, because bench players can draw them.
3. **Fewer than 5 found.** The missing players played the *whole* period without recording anything.
   A provisional pass computes on-court seconds, and the missing slots go to the silent players with
   the largest boxscore residual (box seconds minus reconstructed seconds). Ties go to whoever was on
   the floor at the end of the previous period. The test strips a full-period player's evidence from
   real games and checks that he is recovered.
4. **More than 5 found** means a sub is missing from the feed. The earliest 5 are kept and an anomaly
   is logged.

**Free-throw credit:** following the NBA's own +/-, FT points (and FT possession ends) are credited to
the lineup **on the floor when the foul was committed**: the most recent foul at the same period and
clock. Subs made between the foul and the free throws don't change the credit. Without this rule, 67
of 108 player +/- values in the fixtures were off by 1–3.

**Validation gate** (`validate_stints`), run in tests for every fixture:
- on-court seconds must be within 1 s of the boxscore `minutes`;
- on-court +/- must equal `plusMinusPoints` exactly.

| fixture | game | players | max min diff | +/- mismatches |
|---|---|---|---|---|
| 0022500001 | HOU @ OKC 124–125, 2OT | 20 | 0.0 s | 0 |
| 0022500095 | SAS @ NOP 120–116, OT | 19 | 0.0 s | 0 |
| 0022500083 | PHI @ BOS 117–116 | 20 | 0.0 s | 0 |
| 0022500622 | BKN @ NYK 66–120 (blowout) | 27 | 0.0 s | 0 |
| 0022501066 | LAC @ IND 114–113 (LAC down 24) | 22 | 0.0 s | 0 |

## 3. Game story (`game_story.py`)

**Runs:**
- An **unanswered run** (`unanswered_runs`) is a maximal sequence of scoring events by one team with at
  least `min_run` points (default 8).
- An **"X–Y run"** (`stretch_runs`) starts and ends with the team scoring, and must satisfy all of:
  - X ≥ `min_run`;
  - X − Y ≥ `min_run`;
  - Y ≤ 0.4·X;
  - it spans ≤ 6 minutes of game clock.
- Stretch runs don't overlap. They are picked greedily by net margin, with ties going to the shorter run.
- Duration is measured from the first basket to the last.
- `scoring_runs` returns both kinds. The facts document folds an unanswered core into its enclosing
  stretch ("19-5, includes 14-0").
- Check: the largest unanswered run matches the boxscore's `biggestScoringRun` / `biggestScoringRunScore`.

**Lead changes and ties:**
- A lead change is counted when the leading team switches, even with a tie in between.
- A tie is counted each time the score becomes level, excluding 0–0.
- Both match the boxscore `leadChanges` and `timesTied` exactly in all fixtures.

**Largest lead** is computed per team, at the first moment the lead was reached, and matches `biggestLead`.

**Comeback** is the largest deficit the eventual winner faced, and when it happened.

**Clutch** follows the NBA definition: the last 5:00 of the 4th quarter or any OT, with the margin
within 5.
- The window is **inclusive**: a moment at exactly 5:00 is in it; one at 5:00.1 is not.
- An event is in the window if its *pre-event* state qualifies. So the basket that stretches a lead
  from 5 to 7 is still a clutch basket, and the window closes right after it.
- Clutch seconds are measured over the piecewise-constant game state.
- **Per player:** points, FG, 3P, FT, AST, STL, BLK, TOV, on-court +/- and seconds.
- **Key plays:**
  - a **go-ahead** score turns a tie or deficit into a lead;
  - a **tying** score erases a deficit exactly;
  - steals and blocks are listed only inside the final 2:00.
  - The winner's last go-ahead score, after which they never trailed or were tied, is marked
    `decisive`.
  - The facts document keeps the 8 most recent key plays.

**Garbage time** follows the Cleaning the Glass style, and applies to the 4th quarter only.

| time left in Q4 | margin needed |
|---|---|
| 12:00–9:01 | ≥ 25 |
| 9:00–6:01 | ≥ 20 |
| 6:00–0:00 | ≥ 10 |

There is also a starter condition: **≤ 2 of the game's starters (both teams combined)** on the floor.
- The rule is evaluated continuously. Garbage time ends if the margin drops under the threshold or
  starters come back, and it can start again later.
- **Difference from CTG:** CTG treats garbage time as lasting to the final buzzer once it starts.
- **Effect:** garbage time is excluded from the non-garbage +/-, on/off, minutes, the unsung hero
  pool and turning-point candidates.
- **In the fixtures:** the NYK blowout has the whole 4th quarter flagged (720 s). The other games have
  0 s.

**Player impact** (`player_game_impact`) gives, for everyone who played:
- minutes, points and box stats;
- TS% = PTS / (2·(FGA + 0.44·FTA));
- Hollinger Game Score;
- stocks (STL + BLK);
- on-court +/-, for the whole game and excluding garbage time;
- **on/off net rating differential**: the team's net per 100 possessions with the player on minus with
  him off, excluding garbage time. Possessions are (own + opponent possessions) / 2. It is `None` if
  either side has fewer than 10 possessions;
- **winning time**: the stints, seconds and +/- the player was on the floor for during his team's
  *decisive run*. That run is the turning-point run for the winner, and each team's largest
  non-garbage run otherwise.

**Turning point** (`turning_point(events, runs=None, win_prob=None, exclude=garbage)`):
- **Placeholder** (`method: "run"`). Among the winner's non-garbage runs, ranked by net margin, it takes
  the first that does one of these:
  - flips the lead (`flipped_lead`);
  - creates a lead never relinquished (`created_winning_margin`);
  - levels the game from behind (`levelled`).

  Otherwise it falls back to the winner's largest run.
- **Win-probability interface**: pass `win_prob=[(event_idx, p_home), ...]`. It returns the largest
  swing toward the eventual winner between consecutive points (`method: "win_probability"`,
  `wp_swing`), plus the run containing it.

## 4. Facts document (`summarize.game_facts`)

`game_facts` returns a deterministic, JSON-serializable document, about 4.5–8.6 KB per game. Scores are
always keyed by tricode. Times look like `{"label": "Q4", "clock": "3:52", "score": {...}}`. A golden
snapshot lives in `tests/fixtures/analytics/facts_0022501066.json`; regenerate it with
`COURTSIDE_UPDATE_GOLDEN=1`.

Keys: `schema, game_id, date, arena, home, away (team, name, score, periods[]), winner, margin,
overtimes, top_performers, unsung_hero, runs, turning_point, clutch, lead_changes, times_tied,
largest_lead, comeback, garbage_time, notable_lines, incidents, validation`.

- **Top performers** are the two best Game Scores per team.
- **Unsung hero candidates** (up to 3; winning team first, then by score):
  - **Eligible** players have:
    - ≥ 12 non-garbage minutes;
    - ≤ 20 points;
    - not one of their team's top-2 scorers;
    - a score ≥ 2.0.
  - **Score:**
    ```
    0.5·(non-garbage +/-)
    + 0.1·on_off·on_poss/(on_poss+40)                                  # shrunk on/off
    + (STL + 0.7·BLK + 0.7·AST + 0.7·OREB + 0.3·DREB − TOV)            # hustle / non-scoring box
    + 0.5·(PTS − 2·0.57·(FGA + 0.44·FTA))                              # points over par TS
    + 1.0 if off the bench
    ```
  - Each candidate carries plain-English `why` strings that the writer can quote.
- **Notable lines** are flagged at these thresholds:
  - 30/40/50 points (highest only);
  - triple-double (10+ in three of PTS/REB/AST/STL/BLK);
  - 20–10 (20+ PTS with 10+ REB or AST);
  - 15+ REB, 12+ AST, 5+ threes, 5+ stocks;
  - perfect FG (≥ 5 FGA) and perfect FT (≥ 8 FTA);
  - efficient scoring (25+ PTS at TS ≥ 70%);
  - cold shooting (≥ 15 FGA at FG% ≤ 30%);
  - bench scoring (20+ from a reserve);
  - 7+ turnovers.
- **Incidents:**
  - **Ejections**: an explicit `ejection` action, a second non-defensive-3-second technical, or a
    flagrant 2.
  - **Foul-outs**: the 6th personal foul. Checked against the boxscore.
  - **Explicit injury mentions**: any action text matching `injur`.
  - **Heuristic `possible_injuries`**: a starter with < 20 minutes, last on the floor before 6:00 of
    the 3rd quarter (outside garbage time), while at least three teammate starters played on more than
    2 minutes longer. The note says "left the game and did not return". **Verify before calling it an
    injury.**
- **validation** carries the stint-gate summary, so the writer and the fact-check pass can refuse
  games that fail it.

## Known limitations

- **Injuries:** the cdn PBP has no injury event. The heuristic misses 2nd-half injuries and can flag
  coach's decisions. It should be cross-checked against the ESPN summary and injury reports.
- **Ejections:** detection is rule-based and wasn't exercised by a real fixture (only synthetic).
  The `ejection` subType is passed through as the reason verbatim.
- **Possession counting** is approximate around jump balls, held balls, and possession arrows after
  offsetting fouls. Period-end possessions depend on the feed's `possession` field.
- **Garbage time** uses continuous evaluation, unlike CTG's "lasts to the end" rule, and is 4th quarter
  only, so a 30-point 3rd-quarter rout is never garbage time.
- **On/off is noisy** in a single game. Shrinkage dampens it but doesn't fix it. Season-level
  aggregation is the real use.
- **Turning point** is run-based until the win-probability model exists. A big early run that
  "flipped the lead" in a blowout (e.g. NYK's 20–5 in Q1) is selected even though the game was
  decided gradually.
- **Starters** come from the boxscore. If the feed ever omits `starter` flags, period 1 falls back to
  evidence plus residual fill.

## Next

- A **win-probability model** trained on historical PBP (score margin, time, possession, pregame spread
  prior), feeding `turning_point(win_prob=...)` and "biggest swing" plays.
- **Season-level player impact** aggregation (rolling on/off, non-garbage +/- per 100, TS% vs baseline)
  for the Player-pulse hot/cold section.
- **Role-change detection** from stints: new starter, minutes and closing-lineup share jumping, and
  clutch-minutes share.
- Lineup-level net ratings, and closing-lineup identification, for the story text.
