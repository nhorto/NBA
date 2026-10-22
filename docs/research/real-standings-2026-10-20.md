# First real standings: what the 2026-10-20 contests say about our field and chances

*2026-10-22. One slate: DK main slate 154575 (3 games), four contests the owner played, 31,621
real lineups. Script: `research/real_standings_calib.py` (commands `own`, `field`, `chances`,
`slates`, `hist`; JSON outputs in geekom1 `var/reports/real_standings_20261022/`, run on a scratch
copy of the Mac prod DB). Report command: `courtside dfs field-calib`. Versions after this note:
`own-proxy-1.1`, `field-proxy-1.1`.*

## Plain-language summary

- **The modelled field is about as strong as the real ones.** Scored with what the players
  really did that night, our modelled field's cash lines landed inside the spread of the real
  ones: double-ups 244 (now 248) against real lines of 237.25, 238.75 and 247.75; the 29.7k GPP
  253.6 (now 254.4) against 253.25. The old note that "a modelled field is much weaker than a real
  DraftKings field" is not what this night showed, so the contest-builder and slate-page notes
  were reworded.
- **Real double-up fields are far more bunched up than ours were.** In the two $2 double-ups,
  30-35 % of all entries were the *same lineup* and almost 9 in 10 entries shared their lineup
  with someone else. Our modelled double-up field had at most 1.4 % on one lineup. The 10 most
  popular players filled 6 to 6.7 of the 8 roster spots in the double-ups, but only 4.1 in the
  GPP. We changed the double-up field to concentrate and repeat more (and the large-GPP field to
  repeat less, which is what the real GPP did), and every field now spends the salary cap the way
  real entrants do (97 % of real lineups used at least $49,500).
- **Our chances read high, but one night cannot say by how much.** Our lineups were given about a
  70 % chance to finish in the money in a double-up; about half of them did (3 of the owner's 5
  double-up entries). With the recalibrated field the same lineups get about 54 % on average
  (62 % before), against 48 % realised over the three double-ups. Where the night landed inside
  our simulated range (its percentile) moved from about the 27th-42nd to the 35th-52nd: an
  ordinary night either way. One slate is a single draw, so there is still no calibration: bands
  and the "simulated, not real" labels stay until 10 slates are uploaded.
- **Ownership.** The proxy ranks players reasonably (rank correlation 0.72-0.87) but misses the
  biggest names: the public piled into players our inputs did not single out (Dylan Harper 66-93 %
  owned vs our 13-15 %, Jayson Tatum 36-84 % vs 12-16 %). Nothing in our inputs explains those
  misses (no injury tags, no replacement flags, DK FPPG present), so they are left for more
  slates; the one supported change is that double-up ownership is much more concentrated than GPP
  ownership on the same slate.
- **The "cash" lineup was not the problem.** The lock-10 cash lineup *was* our Top-N #1 (same
  eight players); the cash rule (mean minus 0.3 x downside) almost always picks the Top-N #1. Over
  288 RotoGuru slates the cash lineup scored 271.5 vs 270.8 for Top-N #1. 226.75 was a bad night
  for that one lineup. No change to the cash objective.
- **Grader.** The two $2 double-ups came only from the uploads (no payout table), so the grader
  had no cash line for them. DK double-ups pay round(0.435 x entries) places (exact on all three:
  125/287, 500/1,149, 200/459), so the grader and the contest list now infer that table and label
  the line "inferred double-up".

## What one night can and cannot show

Can: anything measured over thousands of real lineups *within* the night and independent of who
the popular players were: how bunched up a double-up field is versus a GPP on the same slate, how
much of the cap entrants spend, how many lineups are unique, how the score distribution's shape
(top-1 % line vs cash line) compares with ours. Also a direct check of field strength: score both
fields with the same real box scores.

Cannot: anything that depends on *which* players the public liked (one set of names), on slate
size (this was a 3-game slate, where crowds bunch up more than on 11 games), or on luck (every
lineup that night shares the same 3 games, so all our P(cash) numbers are tested by one draw).
The ownership weights (`fit_from_uploads`) stay gated at 10 slates; the per-slate report below is
how the next slates will confirm or undo this calibration.

## 1. The real contests

| contest | entries | paid | cash line | median | top 1 % | winner | unique lineups | entries in duplicated lineups | most common lineup | mean salary | >= $49,500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $2 Double Up 196970732 | 287 | 125 (inferred) | 237.25 | 234.75 | 283.5 | 290.25 | 19 % | 89 % | 34.5 % (99) | $49,893 | 96.5 % |
| SE $5 Double Up 196456027 | 459 | 200 | 238.75 | 235.5 | 283.5 | 293.0 | 45 % | 62 % | 17.9 % (82) | $49,818 | 96.5 % |
| $2 Double Up 196456023 | 1,149 | 500 (inferred) | 247.75 | 241.75 | 283.5 | 292.25 | 16 % | 87 % | 29.9 % (343) | $49,918 | 97.3 % |
| $100K Four Point Play 196456024 | 29,726 | 7,513 | 253.25 | 239.0 | 290.5 | 320.25 | 66 % (82 % in 3,000-entry draws) | 43 % (28 %) | 0.8 % (234) | $49,835 | 94.5 % |

All but 85 of 31,621 lineups parsed to our pool (names). Re-scoring the real lineups with our
box-score DK points reproduces DK's points exactly (MAE 0.00).

The most common lineup in every contest was the same one: Tatum, Towns, Brunson, Pritchard,
Anunoby, Ausar Thompson, Harper, Wallace ($50,000), rated 267.1 by our public projection and
264.1 by ours (our best lineups were rated ~280). It scored 234.75, just under the double-up
lines; the three double-ups' lines differ by 10.5 points mostly because of where such blocks of
identical lineups landed. Our modelled field cannot know that lineup's identity.

## 2. Field realism at the actual box scores

Our field built the way production builds it (lock-45 run c15398ddd87f inputs, the contest's
archetype, medium strength), every lineup scored with the real DK points, 3 seeds. Structure at
the real entry count (draws of 287 / 459 / 1,149 / 3,000 entries).

| | cash line | median | top 1 % | mean salary | >= $49.5k | unique | in duplicates | most common | mean public / our projection |
|---|---|---|---|---|---|---|---|---|---|
| real double-ups | 237.25 / 238.75 / 247.75 | 234.8-241.8 | 283.5 | 49,818-49,918 | 96.5-97.3 % | 16-45 % | 62-89 % | 18-35 % | 263-266 / 257-261 |
| field-proxy-1.0 (cash) | 244.0 | 240.0 | 293.3 | 49,646 | 73.5 % | 67-87 % | 23-50 % | 0.9-1.4 % | 263.6 / 257.1 |
| field-proxy-1.1 (cash) | 248.2 | 243.5 | 285.7 | 49,787 | 97.8 % | 56-66 % | 40-54 % | 8.4-8.6 % | 267.6 / 261.7 |
| real 4PP GPP | 253.25 | 239.0 | 290.5 | 49,835 | 94.5 % | 82 % | 28 % | 0.8 % | 256.0 / 247.6 |
| field-proxy-1.0 (large GPP) | 253.6 | 237.0 | 294.5 | 49,576 | 69.6 % | 62 % | 48 % | 0.5 % | 257.5 / 249.5 |
| field-proxy-1.1 (large GPP) | 254.4 | 238.2 | 293.8 | 49,710 | 94.5 % | 80 % | 31 % | 0.3 % | 257.8 / 249.6 |

Grid explored (`field` command; full output in the script's JSON): mix 0.2 / 0.4 / 0.55 / 0.7,
public blend w_ours 0.25 / 0.5 / 0.75, cash optimizer noise 0.10 / 0.05 / 0.03 / 0.02, cash
concentration 1.25 / 1.8 / 2.4, large-GPP concentration 0.9 / 1.1. Findings:

- **Strength.** At the real box scores every setting put the double-up cash line between 241.6
  and 257.5 and the GPP's between 251.5 and 255.6. The medium settings sit inside the real
  spread. A more optimizer-heavy cash field (mix 0.55-0.7) pushed the line to 246-254, above two
  of the three real lines, so the cash mix stays 0.40. There is no evidence here for a "tougher"
  field.
- **Double-up duplication and top tail.** Optimizer noise is what makes a field repeat: at 0.03
  the most common lineup reaches ~8 % and the top-1 % line falls from 293 to 285 (real: 283.5 in
  all three double-ups). Even 0.02 only reaches 14 % on one lineup: the real 30 % blocks come
  from a consensus the public shares and we cannot see.
- **GPP duplication.** The real GPP is mostly unique lineups (82 % unique at 3,000 entries). Our
  40 % optimizer share resampled from ~200 solves made it 62 %; at 20 % it is 80 %, with the same
  cash line and a mean projection closer to the real field's.
- **Salary use.** Real entrants spend more than our model assumed (mean $49,820-49,920; 95-97 %
  at $49,500+). New: expected spend 0.997 of the cap, 95-97 % of sampled lineups held to 99 %,
  the rest to 98 % / 96 %.
- **Stacking.** Most players from one team: real 2.8-3.0 (4+ from one team in 5-13 % of
  lineups); ours 2.7-2.9 (13-17 %). Close; no change.

## 3. Ownership

Proxy vs real %Drafted, lock-45 inputs (players above 0.1 % in either):

| contest | MAE (pp) | rank corr | top-10 MAE (pp) | top-10 bias | real top-10 sum | proxy top-10 sum |
|---|---|---|---|---|---|---|
| $2 DU 287 (cash 1.0 -> 1.1) | 10.4 -> 10.9 | 0.72 -> 0.69 | 31.5 -> 29.8 | -31.5 -> -29.2 | 666 % | 448 % -> 507 % |
| SE $5 DU 459 | 7.7 -> 8.1 | 0.82 -> 0.83 | 24.1 -> 23.5 | -23.2 -> -20.5 | 587 % | |
| $2 DU 1,149 | 10.1 -> 10.8 | 0.82 -> 0.79 | 28.3 -> 27.0 | -28.3 -> -26.0 | 635 % | |
| 4PP GPP (large, unchanged) | 4.6 | 0.87 | 12.6 | -12.5 | 410 % | 368 % |

- **Concentration by contest type.** Within the slate, double-up ownership is GPP ownership
  "sharpened": real DU %Drafted ~ (real GPP %Drafted)^k with k = 2.05 / 2.20 / 3.00 (MAE-best
  power fit, `field.sharpen_exponent`). The proxy had cash / large-GPP concentration 1.25 / 0.9 =
  1.39. New: cash 1.60 (ratio 1.78). Higher values match the *shape* better (2.5 reproduces the
  real top-10 sums) but put even more weight on the players our proxy wrongly favours (MAE gets
  worse: 1.25 -> 2.0 raises cash MAE from 9.4 to 10.5 pp), so 1.6 is a compromise until more
  slates. Large-GPP concentration stays 0.9 (lowest MAE; 1.1 matches the shape but MAE 5.0).
- **Public projection.** w_ours (our share of the DK-FPPG blend) 0.5 is best or tied (GPP MAE
  4.57 at 0.5 and 0.75, 6.0 at 0); unchanged.
- **The misses.** Under: Harper ($5,100; FPPG 24.8, ours 23.5; proxy 13 % vs 66-93 % real),
  Tatum ($9,000; FPPG 47.8 from 23 games last season; 12-16 % vs 36-84 %), Jalen Williams
  ($6,600; ours 25.7 at 23 projected minutes vs FPPG 32.3 from 39 games; 5-7 % vs 8-27 %), Cason
  Wallace ($4,600; 18 % vs 25-47 %). Over: Holmgren ($5,900; 42-53 % vs 18-22 %), Fox ($6,100;
  36-47 % vs 13-21 %), Hart ($6,200; 35-36 % vs 12-20 %). What they share: none had a Q tag
  (9 Q players on the slate, none of these), no injury-replacement flags fired, every one had a
  DK FPPG. The public's own projections were simply higher for young players with growing roles
  (Harper, Wallace, Thompson) and for stars back from long absences with short FPPG samples
  (Tatum, J. Williams). Preseason per-minute production points the same way for Harper (26.3 DK
  in 21.8 min), J. Williams and Wallace but the wrong way for others (Duren, Towns), so it was not
  added from one slate. By salary, real fields put more on $4.5-5.5k and $7.5k+ players and less
  on $5.5-6.5k (double-ups: 3.8-4.1 vs our 2.7 slots at $4.5-5.5k, 0.9-1.1 vs 2.3 at $5.5-6.5k);
  that is the same handful of names, so it waits for more slates.
- **Starter flags.** At lock-10 Pritchard was flagged a non-starter (bench penalty) yet 73-82 %
  owned; limiting the penalty to players projected under 24 minutes barely moved the fit (MAE
  -0.3 pp, rank correlation -0.02), so unchanged. Lock-10 inputs fit worse than lock-45 overall
  (cash top-10 MAE 33-37 vs 24-31).

## 4. Chances against reality

All 50 distinct cash / Top-N lineups we built for the slate, scored on the lock-45 run's sims with
the calibrated truth (`portfolio.truth_sims`), against each real contest (real payout tables;
inferred for the $2 double-ups), 3 seeds. "PIT" = where the night's (lineup - cash line) fell in
the simulated distribution (0.5 = typical).

| contest | real cash line | realised in the money | mean P(cash) 1.0 -> 1.1 | PIT median 1.0 -> 1.1 | simulated cash line p50 |
|---|---|---|---|---|---|
| $2 DU 287 | 237.25 | 60 % | 0.62 -> 0.54 | 0.42 -> 0.52 | 263.4 -> 269.6 |
| SE $5 DU 459 | 238.75 | 54 % | 0.62 -> 0.54 | 0.40 -> 0.49 | 263.5 -> 269.6 |
| $2 DU 1,149 | 247.75 | 30 % | 0.62 -> 0.54 | 0.27 -> 0.35 | 263.5 -> 269.7 |
| 4PP GPP (top 25 %) | 253.25 | 30 % | 0.50 -> 0.50 | 0.32 -> 0.33 | 272.7 -> 272.7 |

The owner's entries: lineup 333 (266.5; stored P(cash) 0.76) finished 13th of 287 and 60th of
1,149 (both in the money); 394 (247.5; 0.72) 98th of 287 (in) and 10,150th of 29,726 (out);
395 (232.5; 0.71) 345th of 459 and 18,953rd (out). No lineup of ours reached a real top-1 % line
in the GPP (290.5; best 287.0). Simulated P(top 1 %) for these lineups is 2-6 %.

The simulated cash lines (~264-270) sit far above the real ones (~241) because the whole night
scored below our projections (our Top-N lineups were projected ~280 and scored ~246 on average);
that shift hits our lineups and the field alike, which is why the PIT, not the raw line, is the
check. The recalibrated double-up field moves P(cash) about 8 points down, toward what happened,
with a typical-looking PIT; nothing here separates a modest bias from one bad draw. The live
"truth" shift toward DK FPPG (TRUTH_BETA 0.18 from 46k historical player-games) estimated on the
two live nights so far is 0.45 +/- 0.18 (146 player-games, game-clustered SE): higher, not
significantly. It is the main lever to watch as standings accumulate; unchanged.

## 5. The cash lineup

- The runner's cash lineup maximises mean - 0.3 x (mean - 25th percentile) (`runner.CASH_LAMBDA`),
  the PLAYERS.md §5.3 fallback; it is not chosen by P(cash) or by the field. On 10-20 it equalled
  the Top-N #1 at lock-45 and lock-10 (and in 7 of the slate's 9 builds).
- Its simulated P(cash) was mid-pack (6th-7th of the lock-10 Top N, old and new field). The Top-N lineup with the highest
  P(cash), under both the old and the new field, was the same lineup (lock-45 #4 / lock-10 #10):
  it scored 258.0 and would have been in the money in all four contests. The recalibration does
  not change which lineup the simulations prefer.
- History (RotoGuru 2019-20 and 2020-21, `var/reports/dfsbt_20261019T132321.txt`): cash lineup
  271.5 vs Top-N #1 270.8 mean actual DK points over 288 slates; real-points cash rate against the
  proxy double-up line 0.729 vs 0.721 (108 slates, `cash_calibration_20261020.txt`). The cash rule
  does not cost points. Switching to a P(cash)-maximising pick needs its own walk-forward test
  (the T2.13 harness in `research/cash_calibration.py`) before it replaces the rule; one night is
  not that test. `optimizer.py` / `topn.py` are unchanged.

## 6. Grader cash line

`grade.contest_results` (154575 was graded on 10-21, before the uploads; its rows have source
'none' and are re-graded automatically at the next grading because standings now exist):

| contest | before | after |
|---|---|---|
| $2 DU 196970732 (upload only) | no line (no table, no paid rows) | 237.25, `line_source` 'inferred double-up' |
| $2 DU 196456023 (upload only) | no line | 247.75, inferred |
| SE $5 DU 196456027 | 238.75 (paid rows) | 238.75 |
| 4PP 196456024 | 253.25 (paid rows) | 253.25 |

The cash lineup is now graded against the slate's *largest* double-up (before: the first one
whose name said "double up", which skipped the unnamed uploads). Names and fees of upload-only
contests come from the owner's contest-history rows. `field.load_contest` uses the same inference
so these contests classify as cash (before: small / large GPP, which would also have mislabelled
them in the ownership refit).

## 7. What changed (and the evidence for each)

| change | evidence (this note) |
|---|---|
| cash `conc` 1.25 -> 1.60 (cash / large GPP 1.39 -> 1.78) | §3: DU ~ GPP^k, k 2.05-3.0, same slate |
| cash optimizer `noise` 0.10 -> 0.03 | §2: duplication (1 % -> 8 % on one lineup; real 18-35 %) and top-1 % line (293 -> 286; real 283.5) |
| large GPP default `mix` 0.40 -> 0.20 (cash / small GPP stay 0.40; "medium" = the archetype's mix) | §2: GPP uniqueness 62 % -> 80 % (real 82 %), duplicated entries 48 % -> 31 % (real 28 %), same cash line |
| `SPEND` 0.99 -> 0.997; `full_spend` 0.90 / 0.82 / 0.78 -> 0.97 / 0.95 / 0.95; floors 0.98 / 0.96 / 0.92 -> 0.99 / 0.98 / 0.96 | §1-2: real mean salary $49,820-49,920, 95-97 % at $49,500+ |
| uploaded double-ups without a payout table: DK's round(0.435 x entries) table (`field.load_contest`, `grade.contest_results`) | §6: exact on all three real double-ups |
| cash lineup graded against the slate's largest double-up | §6 |
| `slate_inputs(as_of=)`, `prepare(as_of=)`, `score_run(as_of=)`; the ownership refit uses the capture at lock | the refit used the newest (post-lock) capture |
| `courtside dfs field-calib [--fit [--save]]` (`field.calibration_report`) | makes the per-slate check repeatable |
| caveat text in `dfs_contest.html` and `dfs.html` | §2 and §4: the field is not "much weaker"; chances ran high on the one slate |
| versions `own-proxy-1.1`, `field-proxy-1.1` | stored ownership rows and run details carry them |

Not changed: ownership weights (b's), w_ours, large-GPP concentration, bench / Q / replacement
terms, TRUTH_BETA / TRUTH_TAU, the cash objective, bands, `MIN_FIT_SLATES` = 10, the "simulated
against a modelled field" labels.

## 8. Before / after on other slates (no standings yet)

Slate page numbers (`field.score_run(write=False)` at the run's capture):

| slate (run) | cash reference contest | simulated cash line p50 | P(cash), cash lineup | P(cash), Top N mean (range) | P(top 1 %), mean |
|---|---|---|---|---|---|
| 155078, 11 games (lock-45 d514cb25b711) | SE $5 Double Up (689) | 271.1 -> 283.7 | 0.91 -> 0.86 | 0.87 -> 0.78 (0.70-0.86) | 0.130 -> 0.131 (vs the 95k GPP) |
| 155316, 3 games (lock-10 9418d9ee3778) | preset double-up (1,000) | 278.7 -> 283.3 | 0.74 -> 0.67 | 0.73 -> 0.66 (0.63-0.71) | 0.041 -> 0.041 (preset 10k GPP) |

Contest builder, chances by number of entries, slate 155078 (10,000 sims; nested sets as on the
page; "1st" = the single top prize):

NBA $5 Double Up 196981149 (229 entries, max 6):

| entries | in the money 1.0 -> 1.1 | top 1 % | top 0.1 % / 1st |
|---|---|---|---|
| 1 | 0.91 -> 0.86 | 0.083 -> 0.063 | 0.030 -> 0.025 |
| 3 | 0.985 -> 0.977 | 0.21 -> 0.20 | 0.090 -> 0.099 |
| 6 | 0.998 -> 0.996 | 0.27 -> 0.27 | 0.123 -> 0.139 |

NBA $40K Sharpshooter 196981122 (15,854 entries, max 20):

| entries | in the money | top 1 % | top 0.1 % | 1st |
|---|---|---|---|---|
| 1 | 0.738 -> 0.737 | 0.126 -> 0.127 | 0.039 -> 0.031 | 0.013 -> 0.007 |
| 5 | 0.978 -> 0.980 | 0.34 -> 0.36 | 0.114 -> 0.094 | 0.040 -> 0.022 |
| 20 | 0.999 -> 0.999 | 0.63 -> 0.64 | 0.27 -> 0.23 | 0.109 -> 0.059 |

On an 11-game slate the simulations still give our lineups ~80-86 % in a double-up, well above
what one 3-game night delivered (about half). That gap is the reason the chances stay banded and
labelled until real standings from 10 slates calibrate them. In the big GPP the less-repetitive
field makes first place and the top 0.1 % roughly half as likely (more distinct lineups near the
top), while in-the-money and top-1 % hardly move.

RotoGuru history (`hist` command, 58 dates 2019-21, 1,000-entry double-up, public = season
average, availability known, scored at the actual points): the new cash field raises the modelled
cash line by +1.7 +/- 0.5 points (250.5 -> 252.2) and lowers its top-1 % line by -3.5 +/- 1.0
(307.0 -> 303.5); share of the field at >= 98 % of the cap 0.965 -> 0.986. Same direction as on
10-20 (+4.2 / -7.6), smaller on the bigger slates of the archive. The T2.13 harness
(`research/cash_calibration.py`, simulated vs realised P(cash) against the proxy line) was not
re-run; a 1.7-point higher line moves both sides of that check the same way.

## 9. Re-fitting as uploads accumulate

1. Upload each contest's standings ("Export lineups to CSV") and the contest history on
   `/dfs/results` (or `courtside dfs results-import FILE`).
2. `courtside dfs field-calib`: per contest, the real lineup structure and concentration next to
   the proxy's errors, and per slate the double-up vs GPP concentration exponent. If k keeps
   landing near 2-3 the cash `conc` should rise further; if it falls back toward 1.4-1.8 it is
   right.
3. On a scratch DB copy, `research/real_standings_calib.py field|own|chances --slate S --run R`
   (R = the slate's lock-45 run) repeats §2-4 for the new slate.
4. From 10 slates: `courtside dfs field-calib --fit` reports the ownership refit
   (`fit_from_uploads`: Nelder-Mead on the b's at each contest's archetype, leave-one-slate-out
   MAE); `--fit --save` writes `var/models/dfs_ownership_params.json`, which the proxy then uses
   (version `own-proxy-1.1-fit-<date>-<n>s`). The payout display gate (`portfolio.payout_calibrated`)
   opens at the same 10 slates.

## 10. What still needs more slates

- **How high the chances really run.** The double-up P(cash) gap (simulated ~0.54-0.62 vs 0.48 on
  10-20; 0.78-0.86 simulated on the 11-game 10-21 slate) can only be measured over many nights.
  The candidate lever is the truth shift toward the public number (TRUTH_BETA; live estimate
  0.45 +/- 0.18 vs 0.18 used).
- **Cash concentration level** (k = 2.05-3.0 on this slate vs 1.78 in the proxy) and whether it
  depends on slate size (this was 3 games).
- **Which players the public piles into**: young players in bigger roles, stars back from long
  absences, the salary "barbell". Needs several slates before any input (e.g. preseason
  per-minute production) is added to the proxy.
- **Double-up lumpiness.** Real double-ups had 18-35 % on one lineup; ours ~8 %. A "consensus
  lineup" share could be added once a few more slates show how stable that share is.
- **P(cash)-maximising cash lineup**: walk-forward test on the T2.13 harness before any switch.
- **Ownership refit** (`fit_from_uploads`) and dollar / ROI display: from 10 uploaded slates.
