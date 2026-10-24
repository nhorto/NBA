# X factor v2: leverage against the modelled GPP field (D3)

2026-10-23. IMPROVEMENTS.md D3. Code: `proj/dfs/xfactor.py` (switch `COURTSIDE_XF_V2`, default
OFF, version `xf-v2`; `top1_counts`, `leverage`, `noisy_fields`, `public_value_ranks`,
`is_xfactor_v2`, `run_leverage`, `apply_v2`, `builder_leverage`, `tilt_scores`), the candidate
solves of `proj/dfs/portfolio.py` (`candidates(lev=)`, `build_portfolio`), the DFS pool tag
(`web/dfs_view.py`, `templates/dfs.html`). Tests `tests/test_xfactor_v2.py` (+ golden
`tests/fixtures/xfactor_v2_off_golden.json`). Eval `research/xfactor_v2_eval.py` (`slate`,
`opener`, `rg`, `rgsum`, `builder`). Data: the 2026-10-23 nightly prod copy (geekom1
`var/tmp/d3/`, deleted after; Mac scratch `~/Developer/courtside-compute/d3/`, outputs in
`d3/out/`, copies in geekom1 `var/reports/xfactor_v2_20261023/`). Everything here is simulated against a modelled field, except the 10-20 real
standings (section 4).

## Plain-language summary

- **What changed.** v1 marks bench and role players with a high chance of a big night and a
  low estimated popularity. Those are mostly cheap value plays the public finds too. v2
  (switch `COURTSIDE_XF_V2`, OFF by default) marks players who show up in the best 1 % of
  entries of our simulated contests much more often than the modelled field drafts them. The
  field is drawn 4 times with different popularity guesses, so a "low" popularity is not taken
  at face value.
- **Walk-forward on 281 RotoGuru dates (2019-21).** Against the modelled field scored on the
  actual box scores:
  - v2's tags were in the top 1 % **1.9 times** as often as their popularity (v1: 1.2; an
    average player: 1.0). The difference is clear in both seasons.
  - v2's tags are pricier (about $5.5k vs $4.1k), a third are stars, and they scored more DK
    points (29.7 vs 23.6).
  - They reach 6x salary a little less often (38 % vs 43 %).
- **The one real contest (10-20, 3 games).** Neither version tagged anyone who mattered. v2
  tagged 0 and 1 players across two runs; v1 tagged 0. One night cannot tell them apart.
- **10-21 example (11 games).** v2 tags Garland, Naji Marshall, Randle, Brandon Miller and
  McCollum. v1 tags Tre Jones, Suggs, Hachimura, Plowden and Mashack. There is no overlap.
- **Contest builder (same switch).** It adds 100 extra candidate lineups tilted toward
  high-leverage players.
  - In simulation against the modelled field, the 150-entry sets gain 4-6 % expected payout on
    rows they did not see. 2-3 % of that comes from the tilt itself; the rest is just more
    candidates.
  - At 20 entries the gain is noisier; on the 20-max 4PP it is flat.
- **Recommendation.** Turning the tag ON is a meaning change for the owner to OK. The builder
  part is promising but needs a few more slates. OFF is bit-identical (golden tests).

## 1. Design

**Goal (owner):** find players most people won't pick who still have a big night. v1 picks
players with a high chance of 6x salary and a low proxy popularity. Those are mostly cheap value
plays, which the public finds too.

**Leverage.** For a stored run (UI) or a build (builder):

1. **Fields.** `LEV_K` = 4 noisy fields of the slate's reference GPP (`field.default_contests`;
   the builder uses its own contest). Each is drawn with the D5 ownership noise
   (`field.perturb_ownership`, log-odds sd 0.86 at the GPP archetype, x2 for D4 late-news
   beneficiaries via `run_own_uncertainty`). This is done explicitly with K = 4 whatever
   `COURTSIDE_FIELD_K` says. The 4 fields share one set of optimizer-like solves. The UI samples
   10,000 entries per field. The builder reuses its own field when that is already a 4-field
   FieldSet; otherwise its single field lends its solves and its sample size to 4 noisy fields.
2. **Sims.** 4,000 rows of the builder's calibrated scoring truth (`portfolio.truth_sims`: the
   beta shift toward the public number, or the role blend when the sims carry it, plus the tau
   widening). Row s is scored against field s mod 4, as `field.evaluate` does for a FieldSet. In
   the builder these are the generation rows (G + S in the shipped `gs` split), never R.
3. **Top 1 %.** Per row, every unique field lineup is scored in `field.evaluate`'s exact integer
   units. The line is the score of the ceil(1 % x F)-th best entry, count-weighted by sample
   multiplicity, with ties at the line included. A player's **top-1 % share** is the mean over
   rows of the share of those entries that hold him.
4. **Ownership** (the denominator) is his realised share of the 4 fields' entries. This differs
   from the stored proxy (`dfs_ownership_proxy`), because the field's optimizer-like part (20 %
   in a large GPP) drafts the public's value plays more than the proxy says (10-21: Brook Lopez
   is 3.5 % in the proxy and 9.0 % in the field). Using the field's own number keeps share /
   ownership a like-for-like ratio.
5. **Shrinkage:** `lev = (share + a) / (own + a)`, with a = `LEV_PRIOR` = 0.25 %. This is the
   posterior mean of his per-entry top-1 % rate under a Beta(a F q, a F (1 - q)) prior centred on
   the field's base rate q = 1 %, i.e. a F = 100 pseudo-entries at leverage 1 at the default
   4 x 10,000 sample. His own C = own x F entries are the trials. The binomial variance with C
   trials is an upper bound on the sampling noise of the rate: the rows average each lineup's
   chance, but which C lineups the field happens to hold is the remaining noise. Effect: a
   0.1 %-owned player keeps 29 % of his raw excess leverage, a 1 % player 80 %, a 5 % player
   95 %. The size was chosen before any evaluation (one quarter of a percent ~ the smallest
   ownership the proxy resolves on big slates). Section 3 shows it helps a little (prior 0: worse).

**Tag rule (`is_xfactor_v2`, then the cap):**
- P(play) >= 0.85;
- leverage >= `LEV_MIN` = 1.5;
- top-1 % share >= `SHARE_MIN` = 2 % (no 0.1 %-owned scrubs: a scrub can have a high ratio and
  still be in almost none of the winning entries);
- modelled-field popularity < `OWN_MAX` = 10 % ("most people won't pick");
- not among the slate's 10 best **public** projections per $1k (`public_value_ranks`: the
  proxy's `pub_proj` / salary, among players it drafts), instead of v1's "our" value rank;
- at most 5 per slate, highest leverage first (ties: lower popularity).

The tier filter (v1: bench or role) is dropped: an under-owned star with a high share of the
winning entries is what the owner asked for.

**Shown on tagged players** (hover text of the tag): his top-1 % share, field popularity and the
ratio; the P10 teammate split (`scenarios.card_splits`, the teammate whose absence moves him
most, with the big-night chance in both cases) when the run's sims carry scenarios; and a
pointer to the D4 "Teammate out" tag when he has one (that tag renders on its own as before).

**UI.** Switch ON: the pool's X factor tag uses v2, with a per-player hover text and a v2
footnote. The "Big-night chance" column is unchanged (it is the v1 chance and still useful).
Wording follows STYLE.md: information only, no "edge", "lean", "over" or "play" in the v2 text
(tests check the footnote and the hover text).

**Cost.** The first view of a run builds the 4 fields: 14-18 s on geekom1 for an 11-game slate,
3-5 s for 3-4 games. The result is cached on disk per run and sims mtime
(`data_dir()/cache/xfactor/<run>.<params>.json`), and in memory as before.

**Builder (ON only).** `build_portfolio` computes the leverage on its generation rows (above).
`candidates(lev=)` then adds round(100 x scale) **leverage-tilted scenario solves**,
w mu_i + (1 - w) FP_{i,s} + lambda clip(log lev_i, +-1.5), with s a uniform generation row and
lambda ~ U(0.25, 1.0) sd(mu). They sit next to the 100 ownership-penalised solves.
- **Why a scenario base.** The first version mirrored the ownpen solves exactly
  (mu_i + lambda log lev_i, lambda ~ U(0, 0.5 sd(mu))). On the real slates it added 0-2 unique
  lineups per build, and in the first Mac run (N = 20 and 150, 3 slates, 3 seeds) the greedy picked
  a tilted lineup once in 14 builds. A tilt of mu alone moves every solve along one direction,
  so 100 solves collapse onto the Top-N-by-mean lineups. The ownpen solves have the same
  problem: they add 0-1 unique candidates on 155078 / 155482. A random scenario row per solve
  gives ~100 unique tilted lineups.
- **Isolation.** The tilted solves draw from their own rng stream (seed + 31) and are appended
  after every other job, so all other candidates are exactly OFF's. ON only adds candidates; the
  greedy decides.
- **Bookkeeping.** `info.version` gets `+xf2`. `info.xfactor_v2` holds K, rows, the top-5
  leverage, seconds and how many chosen lineups are tilted.
- **Cost:** +5-10 s per build at N = 20 (fields 1-4 s, 100 solves).

## 2. OFF is bit-identical

- **Golden** (`tests/test_xfactor_v2.py`, Linux only). Written from the HEAD (102ed53) modules
  loaded in place: `python tests/test_xfactor_v2.py golden <HEAD portfolio.py>` and
  `XF2_GOLDEN_FROM=<HEAD xfactor.py> pytest -k golden_table`.
  - Builder: three synthetic builds (split off, `gs`, `gs` with K = 2 fields), every lineup,
    summary, exposure and info field (floats by repr, timings dropped).
  - `run_table`: the full xf-v1 table of the web test fixture (incl. splits).
  Both reproduce exactly OFF, and an OFF table carries no v2 keys.
- The existing goldens (`test_portfolio_split.py`, `test_field_k.py`) and the xfactor / web DFS
  tests pass unchanged. The template's OFF branch renders the same text (the v1 sentence moved
  into an `{% else %}`, the trailing "These are estimates, not advice." is shared).
- ON tests: share / line / ownership by hand on toy fields (count-weighted line, ties),
  row-to-field routing, the shrinkage formula, shares summing to 8 per entry on a real
  FieldSet, field reuse, the rule and the cap, the public value rank, the tilt, run_table ON
  (v2 flag, v1 kept as `x_v1`, disk cache, fallback to v1 on failure), the pool page text, and
  the builder (only extra solves, own stream, the other sources' candidates equal OFF's, tilted
  lineups appear, reuse of a K = 4 FieldSet, survives a leverage failure).

## 3. Walk-forward on RotoGuru 2019-20 / 2020-21 (proxy ownership)

**Method** (`xfactor_v2_eval.py rg`, Mac scratch, 2 lanes, `nice 10`; arm = current tree,
`COURTSIDE_CONTEXT=1 COURTSIDE_PPLAY=v2` as the xf harness, `protocol='avail'`, 4,000 sims, bundle
fit on seasons < S). Per date, every game of the day:
- **public proxy:** `field.hist_inputs` (season-to-date DK average, RotoGuru starters,
  availability known), public number 1/2 ours + 1/2 season average as live;
- **field:** K = 4 noisy large-GPP fields of a preset 10,000-entry GPP, 10,000 sampled entries
  each;
- **v2 tags:** from the truth sims;
- **v1 tags:** from the raw sims (bench / role, proxy band Low, calibrated P(DK >= 6 x salary/1000)
  >= 1/3, our value rank > 10, top 5). The v1 `STRETCH` was fit on all six RotoGuru seasons,
  2019-21 included, so v1 gets a small look-ahead advantage.
- **Hindsight top 1 % (no real standings exist for these seasons):** the same 4 modelled fields
  are scored on the players' ACTUAL DK points. The top 1 % of each field's entries (count-weighted
  line, ties in) gives each player's actual top-1 % share, averaged over the fields. **Realised
  leverage** = actual share / his share of the fields' entries.
  - This is the top 1 % of the entries the proxy field actually built, not the hindsight-optimal
    lineups. A player nobody drafts cannot appear, which is the point: a real top 1 % is made of
    real entries.
  - The baseline over all eligible players is 1.00.
- **SE:** pooled ratio, clustered by date (delta method). Paired v2 - v1: bootstrap over dates
  (2,000 resamples).

281 dates (143 + 138), 0 failures. 2019-10-22 / 2020-12-22 were skipped: no season average yet.

| set | tags / date | realised leverage (± SE) | predicted leverage | actual top-1 % share | field popularity | in any top-1 % entry | DK >= 6x salary | mean salary / actual DK |
|---|---|---|---|---|---|---|---|---|
| 2019-20 v1 | 1.97 | 1.14 ± 0.09 | 1.20 | 6.3 % | 5.5 % | 91 % | 43 % | $4,179 / 23.6 |
| 2019-20 v2 | 2.33 | **1.99 ± 0.15** | 1.73 | 7.9 % | 4.0 % | 79 % | 42 % | $5,451 / 30.1 |
| 2020-21 v1 | 1.62 | 1.31 ± 0.12 | 1.22 | 5.7 % | 4.3 % | 85 % | 44 % | $4,018 / 23.6 |
| 2020-21 v2 | 2.59 | **1.79 ± 0.16** | 1.72 | 6.5 % | 3.6 % | 83 % | 34 % | $5,464 / 29.4 |
| pooled v1 | 1.80 | 1.20 ± 0.07 | 1.21 | 6.0 % | 5.0 % | 89 % | 43 % | $4,108 / 23.6 |
| pooled v2 | 2.46 | **1.89 ± 0.11** | 1.72 | 7.2 % | 3.8 % | 81 % | 38 % | $5,458 / 29.7 |

**Paired v2 - v1:** 2019-20 +0.85 ± 0.18, 2020-21 +0.48 ± 0.19, pooled **+0.69 ± 0.13**. The
bootstrap never has v2 at or below v1 in 2019-20; P = 0.002 in 2020-21.

- **v2 does what it says.** Its tags sit in the hindsight top 1 % about 1.9 times as often as
  their popularity, against 1.2 for v1 and 1.0 for an average eligible player. Its prediction
  (1.72) is honest, even a little low.
- **v1 mostly finds the field's own picks.** Its tags reach 6x salary as often as promised (43 %),
  but they are also the cheap players the proxy field drafts. Leverage 1.2 means they are barely
  under-owned for how often they win.
- **v2 picks different players:** 44 of 506 v1 tags overlap. v2's tags are
  - 32 % stars (v1: none) at about $5.5k vs $4.1k;
  - less popular in the field (3.8 % vs 5.0 %), but more often in the winners (7.2 % vs 6.0 %).
  - They reach 6x salary a little less often (38 % vs 43 %): 6x is a cheap player's mark, and
    a $7k player's big night is a smaller multiple.
  - There are tags on 257 of 281 dates (v1: 149).
- **Big slates (>= 150 players):** v2 2.05 vs v1 1.29. **Smaller:** 1.77 vs 1.08.
- **Per player-night,** predicted vs realised leverage has a rank correlation of only 0.10 (v1's
  chance: 0.12). A single night is mostly noise, but the predicted top-1 % share correlates 0.72
  with the actual share. The gain shows in the tagged set, not as a ranking of everybody.

**Sensitivity** (pooled, the rule re-applied to the stored rows; NOT used to pick the constants,
which were set before the run):

| variant | tags / date | realised leverage |
|---|---|---|
| shipped (lev >= 1.5, share >= 2 %, a = 0.25 %, popularity < 10 %, public top-10 out) | 2.46 | 1.89 |
| lev >= 1.3 | 4.52 | 1.62 |
| lev >= 2.0 | 0.23 | 2.62 |
| share >= 1 % / 3 % / none | 2.60 / 2.16 / 2.62 | 1.86 / 1.87 / 1.86 |
| no shrinkage (a = 0) | 2.84 | 1.82 |
| no shrinkage, no share floor | 3.20 | 1.82 |
| a = 1 % | 1.49 | 1.83 |
| popularity < 20 % | 2.58 | 1.84 |
| no public-value exclusion | 2.91 | 1.82 |

Each guard adds a little, and none is critical. The threshold is the main dial: lev >= 2 is
sharper but tags about one player a week.

**Caveat (circularity).** The hindsight top 1 % is built from the same modelled field, so this
measures whether v2 finds players who are under-owned *by that field* relative to their real
nights. It does not test against real ownership, which these seasons do not have. Section 4 is
the only real check, and it is one night.

## 4. The 10-20 uploaded GPP (4PP, 29,726 entries, real %Drafted)

`xfactor_v2_eval.py opener`, geekom1, on the prod copy. v2 leverage is computed against THAT
contest's modelled field.
- **Real top 1 %:** the 298 entries ranked within 1 % of the field; 84 of 29,726 lineups could
  not be parsed.
- **Real leverage** = a player's share of those entries / his real %Drafted.

| run | v1 tags | v2 tags | v1 rule without the 1-in-3 floor (top 5 by chance) | rank corr. predicted vs real leverage (41 players, P(play) >= 0.85) | predicted vs real top-1 % share |
|---|---|---|---|---|---|
| lock-45 c15398ddd87f | 0 | 0 | 5 tags, real leverage 0.43 (0 at 6x) | -0.01 | 0.66 |
| lock-10 42da71bcf37a | 0 | 1: Mikal Bridges (field 5.4 %, real 11.1 %, lev 1.58), in **no** real top-1 % entry (11.25 DK) | 5 tags, 0.42 | -0.03 | 0.71 |

- A 3-game slate compresses leverage. The best eligible lock-45 players were Champagnie 1.63
  (public value #5, excluded), Hartenstein 1.60 (#6), Ausar Thompson 1.56 (#4, 27 % proxy) and
  Bridges 1.50.
- Real leverage was driven by Paul George (in 98 % of the real top-1 % entries at 18.6 %
  drafted, 60.75 DK) and Harper (84 % at 66 %). v2 had George at 1.07-1.27 and Harper at 0.93-0.96.
- One night with 41 eligible players cannot rank v1 against v2. Like v1 (xfactor.md §0), the
  owner's examples Drummond and Green were P(play) misses that no tail or ownership rule catches.

## 5. Example: 10-21 main slate (11 games), lock-45 run d514cb25b711

Reference GPP: $400K Zone Special (95,124 entries); 18 s cold on geekom1. Actual DK points come
from the run's grades.

| tag | player | salary | proxy | field | top-1 % share | leverage | chance 6x | public / our value rank | actual DK |
|---|---|---|---|---|---|---|---|---|---|
| v2 | Darius Garland | 7,000 | 10.6 % | 5.6 % | 13.1 % | 2.26 | 0.52 | 16 / 5 | 39.25 |
| v2 | Naji Marshall | 4,800 | 3.4 % | 2.6 % | 5.3 % | 1.97 | 0.38 | 19 / 25 | 29.5 (6.1x) |
| v2 | Julius Randle | 7,100 | 4.4 % | 2.4 % | 4.8 % | 1.94 | 0.29 | 49 / 52 | 55.0 (7.7x) |
| v2 | Brandon Miller | 7,500 | 5.2 % | 3.0 % | 5.7 % | 1.83 | 0.42 | 40 / 15 | 25.25 |
| v2 | CJ McCollum | 5,800 | 5.7 % | 5.0 % | 9.4 % | 1.83 | 0.38 | 29 / 22 | 26.5 |
| v1 | Tre Jones | 4,800 | 8.4 % | 5.9 % | 10.6 % | 1.78 | 0.38 | 14 / 19 | 16.75 |
| v1 | Jalen Suggs | 5,600 | 6.9 % | 6.7 % | 10.7 % | 1.58 | 0.41 | 18 / 16 | 20.25 |
| v1 | Rui Hachimura | 5,400 | 2.5 % | 1.4 % | 1.6 % | 1.11 | 0.38 | 102 / 21 | 26.5 |
| v1 | Daeqwon Plowden | 3,000 | 6.9 % | 5.9 % | 6.0 % | 1.00 | 0.40 | 7 / 14 | 10.0 |
| v1 | Jahmai Mashack | 3,000 | 5.2 % | 3.4 % | 3.0 % | 0.89 | 0.41 | 21 / 12 | 0.0 |

The two sets do not overlap. Plowden and Mashack are $3,000 value plays that v1 likes and the
modelled field also drafts (leverage about 1). v2 picks mid-priced players the field leaves
under-owned. On 10-22 (lock-45 076afee975c9, 4 games) v2 tags nobody: the best eligible
leverage is 1.33, and v1 also tags nobody. No tagged player had a P10 split or a Teammate-out
flag on these runs (the 10-21 / 10-22 sims predate scenario storage; D4 had 0 flags).

## 6. Builder: leverage-tilted solves, E[payout] on the R rows (`gs` split)

`xfactor_v2_eval.py builder`, Mac scratch (2 lanes, `nice 10`, 1 thread per build), prod copy,
lock-45 runs, `COURTSIDE_ROLE_BLEND=0` (the stored sims predate +rb) and `COURTSIDE_FIELD_K=0`.
- Each build sees the first 20,000 stored rows. The `gs` split puts 11,000 R rows in the report
  and leaves 9,000 for generation and selection. **Hold** = the chosen set scored on the last
  5,000 stored rows (fresh truth noise), which no arm saw.
- Same seed means the same rows and the same truth noise, so differences are paired per row.
  Mean over 3 seeds; SE = max(within-seed, between-seed).
- **Arms:** OFF; ON (100 x scale tilted scenario solves); **ctl** = the same extra solves from
  the same stream with leverage 1 (no tilt). ON - ctl therefore isolates the leverage from
  "more candidates".

| contest | N | E[payout] OFF (R) | ON - OFF, R | ON - ctl, R | ON - OFF, hold | ON - ctl, hold | tilted picked | P(any top 1 %) OFF / ON | s OFF / ON |
|---|---|---|---|---|---|---|---|---|---|
| $60K Zone (10-20, 3 games) | 20 | 362 | +6.4 ± 2.9 % | +4.6 ± 3.3 % | +5.3 ± 3.2 % | +6.5 ± 3.9 % | 7.0 | 0.388 / 0.394 | 21 / 27 |
| $100K Fadeaway (10-21, 11) | 20 | 3,554 | +5.4 ± 2.5 % | +4.7 ± 3.3 % | +3.7 ± 1.5 % | +2.8 ± 2.4 % | 5.7 | 0.641 / 0.642 | 48 / 57 |
| $40K Zone (10-22, 4) | 20 | 478 | +7.3 ± 2.3 % | +3.2 ± 1.8 % | +8.9 ± 3.1 % | +7.6 ± 2.6 % | 9.3 | 0.346 / 0.330 | 15 / 16 |
| 4PP (10-20, 20 max) | 20 | 243 | +1.9 ± 3.7 % | +3.7 ± 2.8 % | -4.2 ± 3.2 % | -0.2 ± 3.7 % | 7.7 | 0.393 / 0.388 | 16 / 19 |
| $60K Zone | 150 | 1,898 | +8.8 ± 0.9 % | +4.9 ± 0.8 % | +6.5 ± 1.2 % | +2.4 ± 1.1 % | 56 | 0.886 / 0.878 | 79 / 90 |
| $100K Fadeaway | 150 | 13,742 | +4.4 ± 0.4 % | +2.6 ± 0.4 % | +4.4 ± 0.5 % | +3.1 ± 0.5 % | 62 | 0.952 / 0.951 | 97 / 113 |
| $40K Zone | 150 | 2,245 | +6.3 ± 1.0 % | +2.3 ± 1.6 % | +5.4 ± 1.3 % | +3.1 ± 1.0 % | 58 | 0.804 / 0.783 | 33 / 37 |

- **N = 150:** ON beats OFF on all three contests by 4.4-8.8 % on R and 4.4-6.5 % on the holdout.
  40-70 % of that survives against the control: +2.4 to +3.1 % on the holdout, each at about
  2-6 SE. The greedy picks ~60 of 150 lineups from the tilted solves.
- **N = 20:** the same sign on the three 150-max GPPs (holdout +3.7 to +8.9 % vs OFF) but noisier.
  The 20-max 4PP is flat to negative (-4.2 ± 3.2 % on the holdout).
- **What changes.** P(any top 1 %) does not move (or drops by up to 0.02 on the $40K Zone). So
  the gain sits above the top-1 % line, in the top-heavy tiers (first place, top 0.1 %) and in
  less prize-splitting. This was not decomposed here.
- **The control gains too:** +1.5 to +4 % for 100 plain extra scenario solves. Today's candidate
  pool is short of scenario lineups (the ownpen solves add ~0 unique). That is a separate,
  cheap improvement that does not need the switch.
- **Caveats.**
  - Everything is against the modelled field, and the leverage comes from that same field, so
    this measures self-consistency, not real dollars.
  - Three slates. Dollar E[payout] stays "uncalibrated" in the UI until 10 uploaded slates.
- **Cost:** +2 to +16 s per build (+5-15 %).

## 7. Recommendation

- **X factor tag:** turn `COURTSIDE_XF_V2` ON for the DFS pool once the owner agrees with the new
  meaning.
  - For: it answers the owner's question ("most people won't pick, still a big night") directly.
    Walk-forward, its tags were in the hindsight top 1 % 1.9x as often as their popularity, vs
    1.2x for v1 (paired +0.69 ± 0.13, both seasons). The prediction is calibrated.
  - Against: it tags stars and mid-priced players, not only cheap bench players. It reaches 6x
    salary a little less often (38 % vs 43 %). It often tags nobody on 2-4-game slates. The one
    real night (10-20) shows nothing either way.
  - Revisit LEV_MIN and OWN_MAX with real %Drafted after ~10 uploaded GPPs: run
    `research/xfactor_v2_eval.py opener` per slate.
- **Builder:** the tilted solves add +2-3 % honest E[payout] beyond "more candidates" at N = 150,
  in the model. They ship with the same switch. Before turning it on for builds, confirm on 3+
  more slates, and consider giving the OFF builder more scenario solves anyway (the control's
  +1.5-4 %).
- **Not done / follow-ups:**
  - precompute the v2 table in the pregame job (`pipeline.py` / `runner.py` are other agents'
    files), so the first page view does not pay the 14-18 s;
  - the ownpen solves add ~0 unique candidates on real slates (section 1), so they are worth
    fixing or dropping;
  - with scenarios now stored (P10), the split text on tagged players appears only for runs from
    today on.

## Addendum (2026-10-23): ship prep, the table is precomputed in the DFS run path

**Design.** With `COURTSIDE_XF_V2` ON, `runner._execute` now stores the run's leverage table
with the run. It is a 2-line hook right after the late-news tag (`xfactor.store_safe`). The
table goes in `dfs_runs.detail["xfactor_v2"]`, so no migration is needed.
- It holds the `run_leverage` result plus the sims mtime and `_params_key()`.
- It is computed with the run's `now` (the `created_at` the run is saved with) and the run's
  own late-news tag (D5 ownership uncertainty). Those are the inputs a later page view reads
  back, so the stored table equals a fresh computation.
- `apply_v2` reads it in this order: the stored table (if version, params and mtime match),
  then the disk cache, then a fresh computation (which it caches).
- Showdown runs and runs without sims store nothing. Swap runs are skipped, like the late-news
  tag. A failure is logged and never fails the run.
- **OFF:** `store_safe` returns before doing anything, and the detail is unchanged.

**Mac scratch** (`courtside-compute/xfprep`, a read-only copy of the prod DB, one copy per arm,
prod defaults). `research/xf_ship_prep.py build` replays a lock10 build with
`runner.run_slate(refresh=False, force=True)`. `compare` diffs the two arms.

| slate | games | build OFF | build ON | hook (added) | stored = fresh `run_leverage` | lineups / projections / sims hash / late news |
|---|---|---|---|---|---|---|
| 155482 (10-22 main), lock10 22:45Z | 4 | 24.3 s | 18.6 s | **2.3 s** | yes (51 players) | identical |
| 155078 (10-21 main), lock10 22:50Z | 11 | 41.2 s | 52.0 s | **10.9 s** | yes (286 players) | identical |

- The OFF and ON detail differ only in timing fields (`field.*seconds`) and, ON, the new key.
- The stored table makes the detail 7 KB → 60 KB (11 games).
- The page then reads it in well under a second. A fresh computation took 10.1 s on the Mac
  (14–18 s on geekom1).
- The hook is about the cost of one cold page view. It scales with Mac load, like the rest of
  the build.

**Tests:** `tests/test_xf_ship_prep.py`.
- OFF: `store_safe` never computes, and a lock10 build gives the same lineups, detail and late
  news in two fresh DBs with the switch OFF and ON.
- ON: the run's `now` and late-news tag are passed through; the page reads the stored table
  without building fields; a stale table (params / mtime / version) is ignored; a hook failure
  never fails the run.
