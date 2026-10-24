# Early-season prior (`+ep`, IMPROVEMENTS.md P3)

2026-10-23. Switch `COURTSIDE_EARLY_PRIOR` (default OFF, `pipeline.set_early_prior`). Bundle
suffix `_ep` (after `_bo`, before `_mr` / `_c1`), model version `<arm>+ep`
(`pathA-v0.3+mf+ep+p2+rx+rb`), same seeds. Code: `src/courtside/proj/early_prior.py`, the `ep=`
hook of `state.rate_frame` / `walk_forward`, `pipeline._load_bundle_ep` / `_ep_input`.
**Verdict: recommend ship.** First-30-days DK CRPS is better on all 4 sets that cover October,
by ≥ 2 SE (two-way) on 3 of them. The full season is better on all 5 sets, and no tier is worse by
2 SE.

## 1. What was tried (state level, given actual minutes)

`research/early_prior_eval.py state S --parts …` fits on seasons < S (the S bundle's base rate
params fixed) and compares the walk-forward state of S. The metrics are DK points error given
actual minutes and the NB log-lik of the DK stats. This is cheap (≈ 20 s a season, geekom1), so
every piece was screened here before the harness.

The EB rate is split into this season's games (cur), earlier seasons' games (prev) and this
season's preseason games (pre):

    r = (Xcur + w·v·Xpre + g·d·Xprev + κ·m·μ) / (Ncur + w·Npre + g·Nprev + κ)

First-30-days z of |DK error| (two-way date × player; negative = better), by piece:

| piece | 2019-20 | 2020-21 | 2023-24 | 2024-25 | 2025-26 | kept |
|---|---|---|---|---|---|---|
| d: age delta on prev, per stat × 11 season-age buckets | +0.14 | −0.24 | −0.10 | −0.57 | **+2.07** | no |
| d centred (league drift removed) | −0.75 | −1.42 | −0.27 | −0.59 | +1.95 | no |
| m: rookie prior × draft bucket (1-3/4-10/11-20/21-30/31+/undrafted) | −0.99 | −0.64 | +0.23 | −0.71 | −1.24 | no |
| g: prev weight by role ratio (cur mpg / last mpg; ≥ 2 games) | +0.02 | −0.45 | +1.06 | −0.93 | −0.09 | no |
| w: preseason weight | −2.86 | −2.06 | −1.51 | −1.32 | −2.85 | – |
| **w + v: preseason weight and per-minute level** | **−3.76** | **−3.42** | **−3.64** | **−3.12** | **−3.06** | **yes** |

- **Age delta.** Per-bucket deltas are small (±5 %), they change sign between fit windows, and they
  hurt 2025-26. The H decay (pts 90 days) already discounts last season by ~4× over the summer.
  Centring the deltas to remove league-wide drift did not rescue it.
- **Rookie prior and role change** move almost nothing: rookies' prior is overwritten within a
  few games (κ 25-250 min), and players with a big minutes change already re-weight through the
  tier of the cell prior. The role blend (`+rb`) handles the mean part of role changes.
- **Preseason.** Without v, preseason inflates per-minute rates (first-30 bias +0.05 → +0.25). The
  fitted level v ≈ 0.875 for pts / ast / fg3m (bench-heavy preseason minutes against bench
  units). With v the bias stays flat and the gain grows. Fitted w is 0.8-1.25 (interior on a
  grid to 2.0): one preseason game counts about as much as a regular-season game, decayed with
  the stat's half-life. Full-season |err| also improves (z −2.7 to −4.6), because early rates
  feed the decayed state for weeks.

**Shipped form:** `early_prior.PARTS = ("w", "v")`. `fit` (≈ 10 s) writes w, v per stat. d, m and g
stay neutral; the code for them remains for re-tests (`fit(parts=…)`, letter `c` = centred d).
- **Preseason rows:** `preseason_player_games` with `scrimmage = 0`, `vs_non_nba = 0`, NBA teams
  only, `game_date` before the season's first regular-season day. This drops the 2019-20 bubble
  seeding games. Only the row's own season counts, strictly before the projected day.
- **Fit:** seasons max(S−5, 2015-16) … S−1. Max NB log-lik of each training game's count given its
  minutes and the minutes table, with the base H / ρ / κ / φ of S fixed. A joint grid over
  w ∈ {0 … 2.0} × v ∈ {0.70 … 1.15}.
- **Output:** only `r_<stat>` / `r_dk` change. `n_`, `p_`, `mpg`, tier and grp stay the base values,
  so the context layer's other inputs and the star picks' tiers are unchanged.

2026-27 params (`bundle_2026-27_v1_cx_mf_ep.json`, fit 2021-22..2025-26): pts w 1.0 v 0.875,
reb 1.0 / 0.925, ast 1.0 / 0.875, dreb 0.8 / 0.9.

## 2. OFF is bit-identical

- **Frame level, geekom1:** `research/ep_state_identity.py` compares `git show HEAD:state.py` with the
  working tree on the real DB. walk_forward 2019-20 / 2023-24 / 2025-26 and build_state 2023-10-26
  / 2024-01-10 / 2026-06-01 are `assert_frame_equal(check_exact=True)` identical.
- **Pipeline level, Mac `courtside-compute/ep/lane_id.sh`:** the base tree is the working tree with
  `git show HEAD:` state.py and pipeline.py and no early_prior.py. The code tree is this change.
  Live arm, switch OFF, same DB copy and bundles.
  - `id_driver` hist 2023-10-26, 2024-01-10 and 2020-01-15, plus slate 154575 at 2026-10-20T22:15Z:
    players, stats, every sim array and meta are identical (4/4).
  - `p2_identity` live / live10 / avail 2026-01-15, opener 2026-10-20 and avail 2024-01-10: sha1
    values are identical (5/5).
- `tests/test_early_prior.py` (13 tests):
  - the switch and env flag;
  - version / suffix composition, with OFF strings unchanged;
  - bundle JSON has no `early_prior` key when OFF;
  - OFF never calls `early_prior.rates`, and `ep=None` equals no `ep` exactly;
  - `_ep_input` is None unless ON and fitted;
  - neutral params reproduce the base rates (1e-11);
  - ON changes only `r_*`;
  - the preseason formula by hand;
  - strictly-before and own-season rules;
  - scrimmage, non-NBA and post-start rows are dropped;
  - the fit recovers informative preseason games and gains out of sample;
  - the other pieces run;
  - `_load_bundle_ep` adds only `state["early_prior"]` and caches.
  - Exact checks compare two in-process computations (platform-free); there are no float goldens.

## 3. Harness results (4,000 sims; ON − OFF; SE two-way date × player, `paired_cluster2`)

- **Arm:** `pathA-v0.3+mf+p2+rx+rb` (live), same seeds.
- **Population:** E[min] ≥ 10 in either arm, DNP = 0. Tiers by E[min | plays].
- **First 30 days:** dates before the first regular-season day + 30. The 2025-26 live slice starts
  2025-12-22, so it has no October rows and is full-season only.
- **Per-row outputs:** Mac `~/Developer/courtside-compute/ep/out/rows_*`; `dk_*.json` per set.

| set | first-30 n | CRPS off | ΔCRPS (SE) | z | ΔMAE z | full n | ΔCRPS (SE) | z | ΔMAE z |
|---|---|---|---|---|---|---|---|---|---|
| 2023-24 harness | 3895 | 5.535 | −0.0166 (0.0049) | **−3.39** | −3.47 | 21886 | −0.0028 (0.0012) | −2.36 | −1.37 |
| 2024-25 harness | 3941 | 5.518 | −0.0145 (0.0051) | **−2.84** | −2.49 | 22045 | −0.0023 (0.0012) | −1.92 | −1.27 |
| 2019-20 RotoGuru | 3748 | 5.660 | −0.0119 (0.0064) | −1.87 | −0.73 | 18858 | −0.0013 (0.0015) | −0.84 | +0.67 |
| 2020-21 RotoGuru | 3665 | 5.608 | −0.0156 (0.0053) | **−2.94** | −2.27 | 19266 | −0.0020 (0.0012) | −1.61 | −0.33 |
| 2025-26 live (lock-45) | – | – | – | – | – | 15108 | −0.0015 (0.0007) | −2.11 | −1.40 |

CRPS z by tier (first 30 days / full season):

| set | star | starter | rotation | bench |
|---|---|---|---|---|
| 2023-24 | −2.40 / −2.21 | −0.60 / +0.24 | −1.83 / −1.52 | −1.24 / −1.19 |
| 2024-25 | −1.99 / −1.30 | −0.45 / −0.46 | −2.64 / −1.73 | +0.40 / +0.55 |
| 2019-20 | −1.82 / −1.73 | −1.16 / +0.29 | −1.46 / −0.97 | +1.59 / +1.17 |
| 2020-21 | −1.54 / −0.82 | −2.22 / −1.27 | −2.21 / −1.38 | +1.66 / +1.50 |
| live | – / −0.63 | – / −0.90 | – / −1.96 | – / −1.09 |

- **Acceptance:** first-30 ≥ 2 SE better on 3 of 4 sets (needed 2), and negative on all 4. No tier
  is worse by 2 SE in either window.
  - **Bench caveat:** bench (E[min] < 15, ~1.2-1.6k rows) leans worse on three sets, z +0.4 to
    +1.7. 2024-25 bench full-season MAE is z +2.03 (CRPS z +0.55).
  - Preseason gives fringe players their only current-season per-minute evidence. Preseason
    bench minutes come against other benches, so this is the place to watch.
- **Where the gain is:**
  - Players ≤ 23: first-30 CRPS z −2.0 to −3.1. This is the development the age delta could not
    fit as a curve; it shows up player by player in the preseason.
  - Stars: z −1.5 to −2.4.
  - Rookies: −1.2 to −2.0 on ~300 rows.
  - The mean shift is ~0 (+0.003 to +0.037 DK), so this is allocation, not level. Mean |Δ| is
    0.30-0.35 DK in the first 30 days and 0.13-0.16 over the full season.
- **Live 2026-27 smoke test:** slate 154575 at 2026-10-20T22:15Z, ON vs OFF. Minutes are identical.
  Over 106 players the mean move is +0.002 DK and the mean |move| 0.20, with a range of −1.0 to +0.8.

## 4. Ship notes

- **Pre-fit bundle:** Mac `~/Developer/courtside-compute/ep/data_ship/proj/bundle_2026-27_v1_cx_mf_ep.json`.
  It is the prod `bundle_2026-27_v1_cx_mf.json` (md5 269ada9d…) plus `state["early_prior"]`.
  1. Copy it into prod `var/proj/`.
  2. Set `early_prior.EARLY_PRIOR_DEFAULT = True`.
  3. Roll back with `COURTSIDE_EARLY_PRIOR=0`.
  - Without the copy, the first ON run fits it in ≈ 12 s.
- **Live inputs:** the live path reads the season's preseason rows from `preseason_player_games`
  (prod has 2026-27: 1,991 rows through 2026-10-16). It also reads bio from
  `team_rosters` / `player_index`; with the shipped parts only the preseason matters.
- **Value fades:** the gain is largest in the first weeks and fades as regular-season games
  accumulate. By ~December it is ≈ 0.0015-0.003 CRPS.
- **Harness cache:** `backtest.CODE_FILES` does not cover `early_prior.py`. ON caches are keyed by
  the model version and the bundle JSON (which carries the params). Clear
  `reports/backtest/cache/*/<S>/` after editing `early_prior.py` code alone.
- **Mac scratch DBs** (`rb/data_ev`, `data_id`) had empty preseason tables (they predate the
  backfill). `ep/` copies were filled from geekom1's DB, whose counts match prod. Nothing was
  written to prod.

## 5. Reproduce

```
# state level (geekom1, seconds): one row per piece set
python research/early_prior_eval.py state S --parts {d,dc,w,wv,m,g,dwmg}
python research/ep_state_identity.py 2019-20 2023-24 2025-26          # OFF frame identity
# Mac ~/Developer/courtside-compute/ep: run.sh TREE SCRIPT (env EP=0/1), worker.sh over jobs/*.job
python research/early_prior_eval.py harness S {avail|live}           # rows_<P>_<S>_<MV>.parquet
python research/early_prior_eval.py dk OFF ON LABEL                  # first-30 / full, tiers
# identity: lane_id.sh (base = HEAD state.py + pipeline.py, code = this change)
```
