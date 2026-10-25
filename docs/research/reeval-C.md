# Re-evaluation C: field and ownership vs real standings (H8)

2026-10-25. Workstream C of `docs/REEVAL_2026-10.md`. Switch **`COURTSIDE_OWN_CAL`** (OFF by
default; ON = `own-proxy-1.3` / `field-proxy-1.3`). Code: `proj/dfs/field.py` (`OWN_CAL_*`,
`cal_utility`, `own_cal_w`, `cal_spec_at_w`, `own_sd_table`, `OWN_SD_CAL`, `recent_dk`),
`proj/dfs/portfolio.py` (`strength_field`, `_k_bands` sd label), `proj/dfs/xfactor.py` (version
label). Tests: `tests/test_own_cal.py` (+ golden `tests/fixtures/own_cal_off_golden.json`).
Research: `research/reeval_c.py` (`data`, `diag`, `loso`, `grid`, `noise`, `lines`, `chances`,
`build`). Data: geekom1 `var/courtside.db`, a read-only snapshot taken at 05:42 ET today. Outputs:
`var/tmp/reeval-c/*.json` and `*.txt`. Everything was computed on geekom1 with `nice 10`. The only
Mac access was a copy of 6 sim files (44 MB).

Everything "simulated" is against a modelled field. The standings, %Drafted and the owner's
results are real. **There are 3 slates and 11 contests, and the 3 double-ups are all on one slate.**
Treat every number here as a direction, not an estimate.

## 0. Plain-language summary

- **The big-GPP "top 10 under-estimated by 11-15 points" is mostly not a concentration
  problem.**
  - When you rank players by their *real* ownership, any imperfect guess reads low on the top 10:
    the most-owned players are partly the ones the public surprised us on. A guess with no bias
    and errors of our size would read about -11 points (`field-k.md` §2).
  - Ranked by *our* guess, our top 10 is **too high by 6-21 points**. The real-vs-proxy log-odds
    slope is 0.74-0.90 on all 8 GPPs (below 1 means the guess is too steep).
  - So our GPP guess is too *peaked* on the players it likes. What it misses is *who* the public
    piles onto: value plays where a news-aware projection is far above DK's early-season FPPG
    (Suggs, Schröder, Marshall, Trae Young) and a rookie on a hot streak (Harper).
- **Double-ups are different.** They really are more concentrated than our guess (sum of squared
  ownership 4.2-5.1 real vs 3.5), and our guess also picks the wrong chalk (John Collins at 97 %
  vs 4 % real; Harper at 8 % vs 90-93 %). The DU data is one slate, so nothing is refit for cash.
- **The fix (switch ON), GPPs only.** It lowers the GPP concentration (0.90 → 0.65) and makes the
  public projection lean more on our news-aware projection (w_ours 0.50 → 0.70). Fit on two
  slates and tested on the third:
  - ownership error falls from 4.96 to 4.38 points;
  - rank correlation rises from 0.86 to 0.90;
  - **the modelled GPP field's cash line at the actual box scores moves from 4.1 to 1.7 points
    off reality, and the top-1 % line from 6.1 to 2.8 points off**. On 10-23 the OFF field was
    6-13 points too weak.
  - The 10-20 opener alone gets slightly worse (ownership error 5.26 → 5.34).
- **D5 field K (ownership noise):** on the 1.2 guess, K fields make GPP fields *too* chalky
  (top-10 ownership sum 4.0-4.5 vs 3.3-4.1 real). On the ON guess, a noise sd of about 1.0
  reproduces the real chalk almost exactly. So if the owner wants K fields, turn them on together
  with ON.
- **Our chances barely move:** the owner's entries' simulated GPP P(cash) goes 0.63 → 0.65 ON.
  The "0.41-0.47 real vs 0.63 simulated" gap is one bad night: 0 of 2 matched GPP entries cashed
  on 10-20, against 22 of 27 on 10-23 and 8 of 10 on 10-24. The line PITs are 0.22 / 0.61 / 0.50.
  That is not a calibration signal yet, and the ownership fix is not what would close it.
- **Recommendation:** ON is reasonable to turn on for GPP fields (owner decision), with
  `COURTSIDE_FIELD_K` either OFF or ON together with it, never K alone. Re-check after 3 more
  uploaded slates (§7).

## 1. Data

The last pre-lock run per slate is the lock10 run made before lock (`standings._reference_run_before`).
Inputs come from the salary capture at or before lock, as the public saw them.

| slate | day | games | run (lock10) | made | lock | pool |
|---|---|---|---|---|---|---|
| 154575 | 10-20 | 3 | 42da71bcf37a | 18:45Z | 19:00Z | 125 |
| 155547 | 10-23 | 6 | 685785c2c04b | 22:50Z | 23:00Z | 207 |
| 155576 | 10-24 | 5 | 1f04e52b1979 | 23:15Z | 23:30Z | 175 |

Contests: 3 double-ups (287 / 459 / 1,149, all on 10-20) and 8 large GPPs (5,112-83,234 entries).
There is no small GPP (≤ 1,000), so `small_gpp` keeps the large-GPP value × the 1.2 ratio.

**Report bug fixed (report only, not a model change).** `field.calibration_report` (the ownership
section of `real-calib`) classified uploaded contests that have no stored field size or payout
table as *small* GPPs (field 0). That hit 4 of the 10-23 GPPs, and their proxy used conc 1.0
instead of 0.9. They are now classified by the standings' entry count, and the report's
"small_gpp 5+ games" row became "large_gpp".

## 2. Diagnosis (OFF = own-proxy-1.2 at each contest's real type)

### 2.1 Per contest

"top-10 bias" = mean (proxy − real) over the **real** top 10 (the metric `real-calib` reports).
"proxy-top-10 bias" = the same over **our** top 10. Slope = OLS of logit(real) on logit(proxy);
1 means right steepness, below 1 means too steep.

| contest | type | entries | MAE pp | rank corr | top-10 bias | proxy-top-10 bias | logit slope | Σ own² real / proxy |
|---|---|---|---|---|---|---|---|---|
| 10-20 $100K 4PP | GPP | 29,726 | 5.26 | 0.88 | -14.6 | +7.4 | 0.89 | 2.43 / 2.26 |
| 10-20 SE $5 DU | cash | 459 | 9.70 | 0.84 | -25.7 | +13.6 | 0.76 | 4.21 / 3.47 |
| 10-20 $2 DU | cash | 1,149 | 13.65 | 0.72 | -30.3 | +14.1 | 0.69 | 4.73 / 3.47 |
| 10-20 $2 DU | cash | 287 | 13.37 | 0.69 | -33.4 | +12.3 | 0.71 | 5.09 / 3.47 |
| 10-23 $40K Sharpshooter | GPP | 15,854 | 4.94 | 0.84 | -14.3 | +17.9 | 0.81 | 1.79 / 2.23 |
| 10-23 $2.1K mini-MAX | GPP | 5,112 | 5.69 | 0.76 | -11.8 | +20.9 | 0.74 | 1.68 / 2.23 |
| 10-23 $20K And-One | GPP | 23,781 | 4.74 | 0.85 | -13.1 | +17.8 | 0.81 | 1.71 / 2.23 |
| 10-23 $35K mini-MAX | GPP | 83,234 | 5.38 | 0.79 | -13.3 | +19.5 | 0.77 | 1.74 / 2.23 |
| 10-23 $4K Quarter Jukebox | GPP | 19,024 | 4.68 | 0.85 | -12.4 | +18.0 | 0.81 | 1.66 / 2.23 |
| 10-24 $4K Quarter Jukebox | GPP | 16,812 | 4.59 | 0.88 | -10.8 | +6.0 | 0.90 | 1.97 / 1.98 |
| 10-24 $20K And-One | GPP | 23,781 | 4.46 | 0.88 | -10.7 | +6.1 | 0.90 | 1.95 / 1.98 |

### 2.2 By group (rows above 0.1 % in proxy or real; mean real vs proxy, pp)

| group | GPP n | GPP real / proxy (bias) | cash n | cash real / proxy (bias) |
|---|---|---|---|---|
| salary < 4k | 183 | 1.0 / 1.9 (+0.9) | 39 | 1.4 / 0.5 (-0.9) |
| salary 4-6k | 405 | 7.5 / 6.5 (-1.0) | 74 | 17.3 / 16.0 (-1.4) |
| salary 6-8k | 160 | 12.9 / 14.4 (+1.4) | 36 | 21.3 / 24.7 (+3.4) |
| salary 8k+ | 88 | 11.8 / 12.5 (+0.8) | 21 | 13.6 / 14.9 (+1.3) |
| public value rank 1-5 | 40 | 30.2 / 40.9 (**+10.7**) | 15 | 30.7 / 48.5 (**+17.8**) |
| value rank 6-10 | 40 | 18.2 / 28.2 (**+10.0**) | 15 | 47.4 / 45.7 (-1.8) |
| value rank 11-20 | 80 | 18.5 / 18.3 (-0.2) | 30 | 20.7 / 19.1 (-1.5) |
| value rank 21-40 | 159 | 10.6 / 8.6 (-2.0) | 54 | 10.2 / 7.5 (-2.8) |
| public projection rank 1-5 | 40 | 13.5 / 32.2 (**+18.7**) | 15 | 17.7 / 16.9 (-0.8) |
| projection rank 6-10 | 40 | 21.8 / 18.0 (-3.8) | 15 | 32.5 / 21.5 (**-11.0**) |
| teammate-out news (a ≥ 20-min teammate OUT and +3 min) | 71 | 17.3 / 13.8 (-3.4) | 9 | 4.9 / 34.9 (+30.0) |
| 1.2 replacement flag (+6 min) | 24 | 17.7 / 15.1 (-2.6) | 3 | 7.7 / 97.0 (+89.3) |
| ours > DK FPPG by 10 %+ | 254 | 4.5 / 2.6 (-1.8) | 12 | 14.2 / 37.9 (+23.6) |

- **GPPs.** The proxy over-owns its own favourites (the top-5 value plays and the top-5
  projections by 11-19 points). It under-owns news and replacement plays and players our
  projection rates above FPPG. Salary tier is flat (within ±1.5 points).
- **Double-ups (one slate).** The replacement flag fired on John Collins (proxy 97 %, real 4 %).
  The real chalk was Harper 90-93 %, Tatum 62-84 %, Towns 69-84 % and Pritchard 76-82 %, all
  under-owned by the proxy.

### 2.3 Residual log-odds regression (logit real - logit proxy; SE clustered by player-slate)

| term | GPP (836 rows, 296 player-slates) | cash (170 rows, 1 slate) |
|---|---|---|
| logit proxy | **-0.29 (SE 0.07)** | -0.04 (0.10) |
| (ours - FPPG) / FPPG | **+0.54 (0.15)** | -0.46 (0.60) |
| recent form (last 3 DK vs public projection) | -0.13 (0.13) | +0.94 (0.50) |
| teammate-out news | +0.42 (0.27) | -1.14 (1.19) |
| 1.2 replacement flag | +0.16 (0.47) | **-4.20 (1.29)** |
| z(salary) | **+0.42 (0.10)** | -0.22 (0.26) |
| confirmed bench | -0.27 (0.13) | +0.53 (0.38) |

### 2.4 Verdict: temperature or features?

- **GPP: both, but the temperature runs the opposite way to the headline.**
  - The proxy is too steep (logit coefficient -0.29, slopes < 1), so its concentration should go
    *down*.
  - It also misses drivers. The clearest one is how much the public trusts a news-aware
    projection over DK FPPG (w_ours). The salary and teammate-out terms are suggestive in-sample
    but not stable out of sample (§3.2).
  - Sharpening the proxy to chase the real-top-10 bias would make its own top 10 even more
    wrong. The real-top-10 under-read is the price of per-player errors, and its honest
    treatment is ownership noise in the field (D5), not a steeper point guess.
- **Cash: a features ("who") problem.** It is under-concentrated in total (Σ own² 3.5 vs 4.2-5.1),
  but every error metric improves as the cash proxy *flattens*, because its chalk is the wrong
  players. One slate cannot separate the two, so cash is left unchanged.

## 3. The fix (`COURTSIDE_OWN_CAL`, OFF by default)

### 3.1 What ON changes (`field.OWN_CAL_SPEC`)

| | OFF (1.2) | ON (1.3) |
|---|---|---|
| large-GPP concentration | 0.90 | **0.65** |
| small-GPP concentration | 1.00 | 0.72 (keeps the 1.2 ratio; no data) |
| GPP public projection weight on ours (w_ours) | 0.50 | **0.70** |
| cash concentration / w_ours | 1.60 / 0.50 | unchanged |
| K-field log-odds sd (only with `COURTSIDE_FIELD_K`) | `OWN_SD` 0.86 GPP / 2.65 cash | `OWN_SD_CAL` **1.00** GPP / **2.00** cash |
| versions | own-proxy-1.2, field-proxy-1.2 | own-proxy-1.3, field-proxy-1.3 (+k) |

- **Contest builder.** Field strengths shift w_ours by ±0.25 around 0.70 instead of 0.50
  (`own_cal_w`, `cal_spec_at_w`).
- **`fit_from_uploads`** (the gated refit at 10+ slates) stays on the 1.2 utility (`cal=False`).
  At 10 slates, refit the spec instead.
- **`cal_utility` also carries three more terms**: z(salary), recent form (`recent_dk`, last 3
  box scores) and teammate-out news (`b_sal`, `b_form`, `b_news`). All three are 0 in the shipped
  spec. `slate_inputs` reads recent DK only if `b_form` ≠ 0, so ON adds no query.
- **Rollback:** unset `COURTSIDE_OWN_CAL`, or set it to 0.

### 3.2 How it was chosen: leave-one-slate-out, GPPs (8 contests, 3 slates)

The fit loss weights slates equally, with each slate's contests averaged first. Nelder-Mead
fits per variant, held-out means over the 3 slates:

| variant (free parameters) | loss | MAE pp | rank corr | logit MAE | proxy-top-10 bias | fitted (all 3 slates) |
|---|---|---|---|---|---|---|
| OFF | — | 4.96 | 0.858 | 0.622 | +10.8 | — |
| concentration only | pp | 4.40 | 0.895 | 0.618 | +6.2 | conc 0.67 |
| conc + w_ours | pp | **4.33** | **0.897** | **0.600** | +4.2 | conc 0.66, w 0.64 |
| w + salary + form + news (1.2 conc) | pp | 4.84 | 0.872 | 0.594 | +9.3 | b_form +1.25, b_news -0.32 |
| all five | pp | 4.36 | 0.895 | 0.602 | +7.1 | b_form +0.63, b_news -0.15 |
| conc + w_ours | logit | 5.12 | 0.868 | 0.627 | +11.5 | conc 0.94, w 0.71 |
| all five | logit | 5.24 | 0.867 | 0.625 | +10.1 | b_form -0.11, b_news +0.93 |

- The feature weights flip sign between losses and folds, so they are not identifiable from 3
  slates and are not used.
- **w_ours went up in every variant** (0.62-0.72).
- **The concentration fit is consistently 0.64-0.77 under the pp loss.**

**The shipped values come from a transparent grid** (`grid`: conc 0.50-1.20 × w 0.30-1.00, picked
on the mean MAE over the training slates):
- each fold's pick: 10-20 held out → (0.65, 0.70); 10-23 → (0.70, 0.60); 10-24 → (0.65, 0.70);
- all 3 slates → **(0.65, 0.70)**.

Held-out results per slate (OFF → the fold's pick):

| held-out slate | MAE pp | rank corr | logit MAE | real-top-10 bias | proxy-top-10 bias | logit slope |
|---|---|---|---|---|---|---|
| 10-20 (1 GPP) | 5.26 → 5.34 | 0.878 → 0.880 | 0.566 → 0.625 | -14.6 → -16.3 | +7.4 → +4.1 | 0.89 → 0.93 |
| 10-23 (5 GPPs) | 5.09 → **4.05** | 0.817 → **0.906** | 0.744 → 0.644 | -13.0 → -15.7 | +18.8 → +9.9 | 0.79 → 0.96 |
| 10-24 (2 GPPs) | 4.52 → **3.74** | 0.880 → 0.902 | 0.557 → 0.546 | -10.8 → -15.8 | +6.0 → -0.5 | 0.90 → 0.97 |
| **mean** | **4.96 → 4.38** | **0.858 → 0.896** | 0.622 → 0.605 | -12.8 → -15.9 | +10.8 → +4.5 | 0.86 → 0.95 |

The real-top-10 bias gets *worse*, as §2.4 predicts: a flatter point guess reads lower on
whatever the public happened to pile onto. That gap is carried by the K fields (§4.2), not by
the point guess.

**Cash (10-20 only, in-sample).** The double-ups' MAE falls steadily as the cash concentration
drops (1.6 → 0.8: 12.2 → 9.5 points), while the field's concentration moves further from the real
one. w_ours 0.7 also hurts cash on this slate (12.2 → 13.2). Cash is left at 1.2 until a second
double-up slate exists.

## 4. Leave-one-slate-out results

### 4.1 Ownership (`real-calib` metrics; slate-weighted)

| | large GPP OFF | large GPP ON (held out) | cash OFF = ON |
|---|---|---|---|
| MAE pp | 4.96 | **4.38** | 12.24 (1 slate) |
| rank corr | 0.858 | **0.896** | 0.751 |
| top-10 bias (real top 10) | -12.8 | -15.9 | -29.8 |
| proxy-top-10 bias | +10.8 | **+4.5** | +13.3 |

ON in-sample, as `courtside dfs real-calib` will show it with the switch on: large GPP 5+ games
MAE 4.8 → 3.7, rank corr 0.85 → 0.91.

### 4.2 Ownership noise (D5 field K) refit

Method as `field-k.md` §2: the sd whose noisy draws reproduce each slate's real MAE. Coverage is
real ownership against 400 draws, at the sd fitted on the other two slates. Large GPP:

| proxy | fitted sd per slate (10-20 / 10-23 / 10-24) | held-out 90 % coverage | 50 % coverage | draws' top-10 own sum vs real (per slate) |
|---|---|---|---|---|
| OFF | 1.05 / 1.63 / 1.06 | 0.81 / 0.83 / 0.89 | 0.63 / 0.45 / 0.55 | 4.99 / 4.43 / 4.72 vs 4.10 / 3.33 / 3.54 |
| OFF at the live `OWN_SD` 0.86 | — | 0.78 / 0.76 / 0.75 | 0.39 / 0.38 / 0.38 | 4.45 / 4.25 / 4.22 vs the same |
| ON | 1.04 / 1.14 / 0.94 | 0.80 / 0.78 / 0.74 | 0.48 / 0.35 / 0.48 | **4.39 / 3.50 / 3.80** vs the same |
| ON at 0.86 | — | 0.75 / 0.70 / 0.65 | 0.36 / 0.29 / 0.36 | **4.17 / 3.35 / 3.49** vs the same |

- **Field K on the 1.2 proxy over-chalks GPP fields** (Σ own² 2.6-3.2 vs 1.7-2.4 real).
- On the ON proxy, sd 0.86-1.0 reproduces the real chalk. `OWN_SD_CAL` GPP = **1.00**, the
  D5-method value fitted leave-one-slate-out.
- The tails are heavier than log-normal (90 % intervals cover 65-80 %).
- **Cash (one slate, in-sample):** sd 2.0 reproduces the double-ups' top-10 sum (6.22 vs 6.30)
  and Σ own² (4.63 vs 4.68); the D5 value 2.65 over-shoots (6.54 / 5.01). `OWN_SD_CAL` cash = 2.00.

### 4.3 Modelled field vs the real standings, at the actual box scores

Cash line = the score at the contest's paid fraction; top-1 % line likewise. Each value is the
mean of 2 seeds; the field is built exactly as production builds it (200 optimizer solves, up to
20k lineups). Errors are modelled minus real. **ON for 10-23 uses its held-out fold (conc 0.70,
w 0.60).** The 10-20 and 10-24 folds equal the shipped spec, so every ON number is out of sample.

| contest | real cash | OFF | ON | OFF+K | ON+K | real top 1 % | OFF | ON | OFF+K | ON+K |
|---|---|---|---|---|---|---|---|---|---|---|
| 10-20 4PP | 253.25 | +3.1 | +1.2 | +2.4 | +0.9 | 290.50 | +3.5 | +1.2 | +2.6 | +0.8 |
| 10-23 Sharpshooter | 245.50 | -6.4 | -1.2 | -6.0 | -1.0 | 292.25 | -8.8 | -1.6 | -7.8 | -1.2 |
| 10-23 $2.1K mini-MAX | 257.50 | -13.4 | -8.9 | -13.4 | -8.0 | 302.75 | -19.5 | -14.1 | -17.6 | -11.9 |
| 10-23 And-One | 243.75 | -5.6 | -1.2 | -6.0 | -0.5 | 293.25 | -9.5 | -3.1 | -8.8 | -2.4 |
| 10-23 $35K mini-MAX | 252.25 | -9.0 | -3.8 | -9.1 | -3.5 | 296.50 | -12.8 | -6.4 | -12.0 | -5.6 |
| 10-23 Quarter Jukebox | 247.00 | -6.5 | -1.5 | -6.5 | -1.4 | 293.75 | -9.2 | -4.4 | -9.6 | -3.2 |
| 10-24 Quarter Jukebox | 277.00 | -0.6 | +0.6 | -3.1 | -2.5 | 318.25 | -2.8 | +1.5 | -5.0 | -0.6 |
| 10-24 And-One | 278.00 | -1.5 | -0.1 | -4.1 | -3.2 | 318.25 | -2.9 | +1.1 | -5.1 | -0.2 |
| **GPP slate-mean abs error** | | **4.1** | **1.7** | 4.7 | 2.2 | | **6.1** | **2.8** | 6.3 | **2.0** |
| $5 SE DU (cash, ON = OFF) | 238.75 | +13.9 | +13.9 | +11.0 | +11.5 | 283.50 | +2.1 | +2.1 | -1.8 | +0.8 |
| $2 DU 1,149 | 247.75 | +4.9 | +4.9 | +2.1 | +2.5 | 283.50 | +2.1 | +2.1 | -1.8 | +0.8 |
| $2 DU 287 | 237.25 | +15.4 | +15.4 | +12.5 | +13.0 | 283.50 | +2.1 | +2.1 | -1.8 | +0.8 |

- **GPP: ON is much closer.** The OFF field was too weak on 10-23: its chalk was the wrong
  players, and real fields rostered the news-driven value.
- The 150-max mini-MAXes (sharper fields) stay the weakest fit.
- Field realism (top-10 realised ownership vs real; 10-23 real 3.21-3.45):
  - OFF 3.69, too chalky;
  - ON alone 3.00, too flat;
  - **ON+K 3.44, right**;
  - OFF+K 3.89.
- **Double-ups:** the modelled cash line is 5-15 points too high on 10-20 with this lock10 run.
  The 10-20 report had 248 with the lock45 run, so the line is run-sensitive. K fields trim it by
  about 3 points. Nothing here refits cash.

### 4.4 Simulated chances of the owner's real entries (`real_calib.contest_sims`, 2,000 sims, local copies of the stored sims)

| config | GPP sim P(cash), slate mean | GPP real in the money, slate mean | GPP place PIT | DU sim P(cash) | DU real | DU line PIT |
|---|---|---|---|---|---|---|
| OFF | 0.629 | 0.472 | 0.445 | 0.647 | 0.60 | 0.468 |
| ON | 0.653 | 0.472 | 0.442 | 0.647 | 0.60 | 0.468 |
| OFF+K | 0.640 | 0.472 | 0.439 | 0.636 | 0.60 | 0.468 |
| ON+K | 0.659 | 0.472 | 0.431 | 0.637 | 0.60 | 0.468 |

Per slate (matched lineups, OFF): 10-20 GPP 0 of 2 (sim 0.61), 10-23 22 of 27 (sim 0.64), 10-24 8
of 10 (sim 0.63). Line PITs are 0.23 / 0.61 / 0.51.

The headline gap is the 2 entries of one night. Pooled over entries, reality (30 of 39) is above
the simulation. With the slate as the unit, 3 slates say nothing yet (E5: about 33 slates to see a
0.10 shift). The ownership fix moves simulated P(cash) by +0.02-0.04, the wrong way for that
headline. If a real gap persists, look at `TRUTH_BETA` / `TRUTH_TAU` (how our lineups are
simulated), not at the field.

### 4.5 Contest builder, OFF vs ON (one build per slate, geekom1, nice 10)

Settings: N = 20, 5,000 sims, no curve or strengths, the lock10 run, `now` = run time, not saved.

| slate / contest | config | P(any cash) | P(any top 1 %) | E[lineups in the money] | mean lineup P(cash) | same lineups vs OFF | actual mean / best of the 20 | seconds |
|---|---|---|---|---|---|---|---|---|
| 10-23 $40K Sharpshooter | OFF | 0.992 | 0.641 | 11.3 | 0.566 | — | 239.2 / 315.2 | 20 |
| | ON | 0.996 | 0.582 | 11.4 | 0.572 | 11 of 20 | 237.3 / 295.0 | 21 |
| | ON+K | 0.997 | 0.601 | 11.6 | 0.579 | | 239.4 / 297.2 | 22 |
| 10-24 $50K Four Point Play | OFF | 0.987 | 0.462 | 9.7 | 0.484 | — | 273.7 / 322.2 | 17 |
| | ON | 0.999 | 0.478 | 10.2 | 0.511 | 9 of 20 | 270.0 / 322.8 | 17 |
| | ON+K | 0.992 | 0.510 | 10.1 | 0.504 | | 269.8 / 322.8 | 17 |

- **The builder picks noticeably different sets:** about half the lineups change, because the
  leverage and ownership terms see different chalk.
- **The chances change little:** P(any cash) +0.5-1.2 points, P(any top 1 %) -6 to +5 points.
- On these two nights the ON sets scored slightly less in reality (-2 to -4 points mean). The
  10-23 OFF set had one 315 lineup that ON did not keep.
- **One night per slate is anecdote, not evidence.**
  - Lineups at or above the real Sharpshooter cash line (245.5): OFF 9 / ON 7 / ON+K 11 of 20.
  - On 10-24 the Four Point Play was not uploaded. Against the And-One line (278): OFF 8 / ON 9 /
    ON+K 10 of 20.

## 5. OFF is bit-identical

- **Golden (Linux):** `test_off_bit_identical_to_pre_change_golden`.
  - It covers the proxy at all 3 archetypes on a synthetic slate with Out / Q players,
    replacements and our projection ≠ FPPG (own, by_slot, λ, spend, version, public
    projection), a sampled field, and the builder's 3 strength proxies and fields.
  - The golden was written from HEAD's `field.py` / `portfolio.py` (94b9712) loaded in place, and
    the current OFF path reproduces it exactly.
  - The D5 golden (`test_field_k.py`) and the builder-split golden (`test_portfolio_split.py`)
    still pass.
- **Real data:** current OFF vs the pre-change modules, by hashes of the arrays, on all 3
  slates. **24 of 24 identical:**
  - `slate_inputs` (same frame and columns: no `recent_dk`);
  - the proxy at cash / small / large (own, by_slot, λ, version);
  - `build_field` and 2-field `build_fields` (lineups, counts, versions);
  - the builder's soft / sharp `strength_field`.
- **Tests:** `tests/test_own_cal.py` has 10 tests:
  - the switch;
  - the OFF golden;
  - OFF carries no cal;
  - the 1.2 values written as a spec reproduce OFF bit for bit;
  - the ON versions and field tags;
  - ON GPP is flatter and cash unchanged;
  - builder strengths around the refit w_ours;
  - the form and news terms;
  - `recent_dk`;
  - the report classification fix.

  These 10, plus `test_dfs_field`, `test_dfs_field_calib`, `test_field_k`, `test_field_rowcap`,
  `test_e5_real_calib`, `test_dfs_portfolio`, `test_portfolio_split`, `test_dfs_xfactor`,
  `test_xfactor_v2` and `test_dfs_showdown`, all pass (152 tests).

## 6. Caveats

- **3 slates.** The held-out MAE gain comes from 10-23 and 10-24. The 10-20 opener is slightly
  worse (first day, FPPG = last season, a 3-game slate, rookies), and there are 5 near-duplicate
  contests on 10-23.
- Every SE that matters has 3 clusters. None of these numbers clears a formal bar.
- **Cash is one slate (the opener).** Nothing for cash is fitted out of sample. `OWN_SD_CAL`
  cash 2.0 is in-sample.
- **The proxy's top-5 GPP projections are still over-owned ON** (+4.8 points), and news plays
  still under-owned (-5.1). Those are D7 feature items that need ~10 slates.
- **The public-projection weight stands in for "the public reads news-aware projections".** A
  late scratch after our lock10 run, or a news source we don't have, is not captured.
- The field-at-actual check is one draw of the games per slate, so a single night's lines carry
  game-luck noise. The contest-builder comparison is one build per slate.
- The sims used for §4.4 / §4.5 are local copies of the stored runs. The runs were not recomputed.

## 7. When to re-check

- **After 3 more uploaded slates with a big GPP (6 in total):**
  - re-run `research/reeval_c.py data grid loso noise lines`;
  - keep ON only if the held-out GPP MAE and the cash / top-1 % line errors still beat OFF on
    the new slates.
- **The first uploaded double-up on a second slate:** revisit cash concentration and w_ours
  (and `OWN_SD_CAL` cash).
- **About 10 slates:** D7 features (form, news, salary, consensus lineup) and an E8 log-odds fit
  of the whole utility (`fit_from_uploads`, refitting the spec).
- **About 33 slates:** DU P(cash) calibration via the line PIT (E5).

## 8. Commands for tonight (Mac; not during 12:00-12:50 or 15:45-19:45 ET; nice 10)

Nothing is committed, so copy the working tree. Workstreams A and B have uncommitted changes in
the same tree; they do not touch the stored sims or the DFS code used here.

    # from geekom1
    ssh nicholashorton@mac-mini 'mkdir -p ~/Developer/courtside-compute/reeval-c'
    rsync -a --exclude var --exclude .venv --exclude __pycache__ ~/Documents/NBA/ \
        nicholashorton@mac-mini:Developer/courtside-compute/reeval-c/code/
    # on the Mac
    cd ~/Developer/courtside-compute/reeval-c
    sqlite3 ~/Developer/NBA/var/courtside.db ".backup courtside.db"
    cd code && export PYTHONPATH=src:research
    R="nice -n 10 uv run python research/reeval_c.py --db ../courtside.db --out ../out"
    $R data && $R grid && $R loso && $R noise --draws 60
    $R lines --seeds 4
    $R lines --seeds 4 --fold-spec 0.70,0.60 --slates 155547
    $R chances --sims 10000
    # builder at web settings (curve + strengths, 25k sims), N = 20, then N = 150 (runtime vs §5.8)
    $R build --sims 25000 --full --variants off,on,on+k
    $R build --sims 25000 --full --n 150 --variants off,on,on+k \
        --builds 155547:197052786:685785c2c04b,155576:197082678:1f04e52b1979

(The script reads the Mac's stored sim paths directly when `../out/sims` has no copy.)
