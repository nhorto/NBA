# NBA Data Preprocessor Usage Guide

## Overview

This guide explains how to use `preprocessor.py` (the updated version) to process your NBA data files.

## Key Insight: Derived Metrics

**IMPORTANT**: Many "derived" metrics like `EFG_PCT`, `TS_PCT`, `AST_PCT`, `OREB_PCT`, `DREB_PCT`, `REB_PCT`, and `TM_TOV_PCT` are **already included** in the Advanced Box Scores dataset from the NBA API.

You only need to calculate derived metrics when:
1. You want to use **only traditional box scores** (without advanced stats)
2. You want custom metrics not provided by the NBA API

## When to Use Advanced Box Scores vs Derived Metrics

### Scenario 1: Use Advanced Box Scores (RECOMMENDED)
- **Best for**: Most machine learning tasks
- **Advantage**: Official NBA calculations, no division-by-zero issues
- **Use**: Just load the advanced box scores directly
- **Skip**: The derived metrics calculation step

### Scenario 2: Calculate Derived Metrics from Traditional Box Scores
- **Best for**: When you want full control over calculations or don't have advanced box scores
- **Advantage**: Can customize formulas
- **Use**: Traditional box scores + derived metric definitions

---

## Example 1: Using Advanced Box Scores (Recommended Approach)

This is the simplest and most reliable approach.

```python
import sys
sys.path.append('/Users/nicholashorton/Documents/NBA Data')

from preprocessor import ProcessingConfig, DataProcessor

# Define which raw stats to process
all_processable_columns = [
    # Raw counting stats (good for cumulative)
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
    'TOV', 'PF', 'PTS', 'PLUS_MINUS',

    # Advanced metrics already calculated (good for rolling)
    'OFF_RATING', 'DEF_RATING', 'NET_RATING',
    'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING',
    'PACE', 'E_PACE', 'PACE_PER40', 'POSS', 'PIE',
    'AST_PCT', 'AST_TOV', 'AST_RATIO',
    'OREB_PCT', 'DREB_PCT', 'REB_PCT',
    'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT'
]

# Create configuration - NO DERIVED METRICS NEEDED!
config = ProcessingConfig(
    all_processable_columns=all_processable_columns,
    rolling_window=5,
    derived_metrics={},  # Empty - we already have these in advanced box scores!

    # Elo settings (optional to customize)
    initial_elo=1500,
    k_factor=20,
    home_advantage=100,
    elo_width=400
)

# Initialize processor
processor = DataProcessor(config)

# Process data - ONE SIMPLE CALL!
df = processor.process(
    advanced_path='Data/Advanced_box_scores_2023.csv',
    traditional_path='Data/traditional_box_scores_2023.csv',
    player_boxscores='Data/player_box_scores_2023.csv'
)

# Save processed data
df.to_csv('Data/final_preprocessed_data.csv', index=False)
print(f"Processed {len(df)} games")
```

---

## Example 2: Calculate Derived Metrics (Without Advanced Box Scores)

Use this when you DON'T have advanced box scores or want custom calculations.

```python
import sys
sys.path.append('/Users/nicholashorton/Documents/NBA Data')

from preprocessor import ProcessingConfig, DataProcessor, DerivedMetric

# Define calculation functions
def calculate_fg_pct(FGM, FGA):
    return FGM / FGA

def calculate_fg3_pct(FG3M, FG3A):
    return FG3M / FG3A

def calculate_ft_pct(FTM, FTA):
    return FTM / FTA

def calculate_ast_to_ratio(AST, TOV):
    return AST / TOV

def calculate_efg_pct(FGM, FG3M, FGA):
    return (FGM + 0.5 * FG3M) / FGA

def calculate_ts_pct(PTS, FGA, FTA):
    return PTS / (2 * (FGA + 0.44 * FTA))

# Define derived metrics
derived_metrics = {
    'FG_PCT': DerivedMetric(
        columns=['FGM', 'FGA'],
        formula=calculate_fg_pct
    ),
    'FG3_PCT': DerivedMetric(
        columns=['FG3M', 'FG3A'],
        formula=calculate_fg3_pct
    ),
    'FT_PCT': DerivedMetric(
        columns=['FTM', 'FTA'],
        formula=calculate_ft_pct
    ),
    'AST_TO_RATIO': DerivedMetric(
        columns=['AST', 'TOV'],
        formula=calculate_ast_to_ratio
    ),
    'EFG_PCT': DerivedMetric(
        columns=['FGM', 'FG3M', 'FGA'],
        formula=calculate_efg_pct
    ),
    'TS_PCT': DerivedMetric(
        columns=['PTS', 'FGA', 'FTA'],
        formula=calculate_ts_pct
    )
}

# Base columns to process (from traditional box scores only)
all_processable_columns = [
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
    'TOV', 'PF', 'PTS', 'PLUS_MINUS'
]

# Keep base columns that go into derived calculations
base_columns = [
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'AST', 'TOV', 'PTS', 'OREB', 'DREB'
]

# Create configuration WITH derived metrics
config = ProcessingConfig(
    all_processable_columns=all_processable_columns,
    base_columns=base_columns,
    rolling_window=5,
    derived_metrics=derived_metrics,
    initial_elo=1500,
    k_factor=20,
    home_advantage=100,
    elo_width=400
)

# Initialize processor
processor = DataProcessor(config)

# Process data
df = processor.process(
    advanced_path='Data/Advanced_box_scores_2023.csv',  # Still needed for merge
    traditional_path='Data/traditional_box_scores_2023.csv',
    player_boxscores='Data/player_box_scores_2023.csv'
)

# Save
df.to_csv('Data/preprocessed_with_derived_metrics.csv', index=False)
```

---

## What the Processor Does Automatically

When you call `processor.process()`, it runs these steps in order:

1. **Load & Merge**: Combines traditional and advanced box scores
2. **Add Context**: Creates home/away indicators, converts W/L to binary
3. **Calculate Rolling & Cumulative**:
   - Creates `{column}_cumulative` for all processable columns
   - Creates `{column}_rolling` for all processable columns (default 5-game window)
4. **Calculate Derived Metrics**: If specified, creates rolling/cumulative versions
5. **Prepare Game Data**: Splits into home/away teams and merges into game-level rows
6. **Calculate Elo**: Adds pre-game and post-game Elo ratings
7. **Process Player Availability**: Calculates missing players and their impact
8. **Calculate Differences**: Creates difference columns for key rolling metrics

---

## Output Structure

After processing, your dataframe will have:

### Game Information
- `GAME_ID`, `GAME_DATE`, `WL` (home team won = 1)

### Team IDs
- `TEAM_ID_home`, `TEAM_ID_away`

### Elo Ratings
- `HOME_ELO_PRE`, `AWAY_ELO_PRE` (ratings before game)
- `HOME_ELO_POST`, `AWAY_ELO_POST` (ratings after game)
- `HOME_WIN_PROB` (home team win probability based on Elo)

### Player Availability
- `missing_players_home/away` (count of rotation players missing)
- `missing_impact_home/away` (importance score of missing players)
- `missing_star_home/away` (1 if star player missing, 0 otherwise)

### Rolling Stats (5-game average for each stat)
- Format: `{stat}_rolling_home`, `{stat}_rolling_away`
- Example: `PTS_rolling_home`, `OFF_RATING_rolling_home`

### Cumulative Stats (season total before this game)
- Format: `{stat}_cumulative_home`, `{stat}_cumulative_away`
- Example: `PTS_cumulative_home`, `FGA_cumulative_home`

### Difference Metrics (calculated automatically)
- The processor drops most home/away pairs and keeps only difference columns for key metrics
- These are specified in the `calculate_difference_metrics()` method

---

## Post-Processing: Filtering Nonsensical Columns

**IMPORTANT**: The preprocessor creates cumulative AND rolling versions of ALL columns in `all_processable_columns`. However, **cumulative only makes sense for counting stats**, not for rates or percentages.

You should filter the output to remove cumulative versions of rate/percentage stats.

### Columns to KEEP (Cumulative Makes Sense)

**Counting stats** - these accumulate over a season:
```python
keep_cumulative_stats = [
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',      # Shooting volume
    'OREB', 'DREB', 'REB',                            # Rebounds
    'AST', 'STL', 'BLK', 'TOV', 'PF',                # Other counting stats
    'PTS',                                            # Points
    'PLUS_MINUS',                                     # Cumulative plus/minus
    'POSS'                                            # Total possessions
]
```

### Columns to DROP (Cumulative Doesn't Make Sense)

**Rate/percentage stats** - cumulative versions are meaningless:
```python
drop_cumulative_stats = [
    # Rating stats (per 100 possessions)
    'OFF_RATING', 'DEF_RATING', 'NET_RATING',
    'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING',

    # Pace stats (rate metrics)
    'PACE', 'E_PACE', 'PACE_PER40',

    # Percentage stats
    'PIE',                                            # Player Impact Estimate
    'AST_PCT', 'AST_TOV', 'AST_RATIO',               # Assist metrics
    'OREB_PCT', 'DREB_PCT', 'REB_PCT',               # Rebound percentages
    'TM_TOV_PCT',                                     # Turnover percentage
    'EFG_PCT', 'TS_PCT',                              # Shooting efficiency
    'USG_PCT', 'E_USG_PCT',                           # Usage percentage
    'FG_PCT', 'FG3_PCT', 'FT_PCT'                     # Basic percentages (if using derived metrics)
]
```

### Filtering Code

Add this AFTER processing to clean up your dataframe:

```python
def filter_nonsensical_columns(df):
    """
    Remove cumulative versions of rate/percentage stats.
    Cumulative only makes sense for counting stats.
    """
    # Define rate/percentage stats that shouldn't have cumulative versions
    rate_stats = [
        'OFF_RATING', 'DEF_RATING', 'NET_RATING',
        'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING',
        'PACE', 'E_PACE', 'PACE_PER40',
        'PIE',
        'AST_PCT', 'AST_TOV', 'AST_RATIO',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT',
        'TM_TOV_PCT',
        'EFG_PCT', 'TS_PCT',
        'USG_PCT', 'E_USG_PCT',
        'FG_PCT', 'FG3_PCT', 'FT_PCT'
    ]

    # Build list of columns to drop
    cols_to_drop = []

    for stat in rate_stats:
        # Drop cumulative versions (both home and away)
        cumulative_home = f'{stat}_cumulative_home'
        cumulative_away = f'{stat}_cumulative_away'

        if cumulative_home in df.columns:
            cols_to_drop.append(cumulative_home)
        if cumulative_away in df.columns:
            cols_to_drop.append(cumulative_away)

    # Drop the columns
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} nonsensical cumulative columns...")
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped: {', '.join(cols_to_drop[:5])}{'...' if len(cols_to_drop) > 5 else ''}")

    return df

# Usage:
df = processor.process(
    advanced_path='Data/Advanced_box_scores_2023.csv',
    traditional_path='Data/traditional_box_scores_2023.csv',
    player_boxscores='Data/player_box_scores_2023.csv'
)

# Filter out nonsensical columns
df = filter_nonsensical_columns(df)
```

### What Gets Kept vs Dropped

**Example: OFF_RATING**
- ✅ KEEP: `OFF_RATING_rolling_home`, `OFF_RATING_rolling_away` (5-game avg makes sense)
- ❌ DROP: `OFF_RATING_cumulative_home`, `OFF_RATING_cumulative_away` (cumulative rating is meaningless)

**Example: PTS (Points)**
- ✅ KEEP: `PTS_rolling_home`, `PTS_rolling_away` (recent scoring trend)
- ✅ KEEP: `PTS_cumulative_home`, `PTS_cumulative_away` (total season points)

**Example: FG_PCT (Field Goal Percentage)**
- ✅ KEEP: `FG_PCT_rolling_home`, `FG_PCT_rolling_away` (recent shooting %)
- ❌ DROP: `FG_PCT_cumulative_home`, `FG_PCT_cumulative_away` (cumulative % doesn't make sense)

### Final Column Count

After filtering, you'll have approximately:
- **Before filtering**: ~300-400 columns (depending on configuration)
- **After filtering**: ~250-300 columns (removes ~50-100 nonsensical cumulative columns)

This makes your dataset cleaner and more interpretable!

---

## Tips

1. **Start with Example 1**: Use advanced box scores, it's simpler and more reliable
2. **Adjust rolling window**: Change `rolling_window=5` to any value (e.g., 3, 10)
3. **Customize Elo**: Modify `k_factor`, `home_advantage`, or `elo_width` for your needs
4. **Handle NaN values**: The first few games will have limited historical data
5. **Check data leakage**: All rolling/cumulative stats use `.shift(1)` to prevent future data leakage

---

## Common Issues

### Division by Zero
If you see infinite values in derived metrics, it's because you're dividing by zero (e.g., FGA=0). The preprocessor replaces `inf` with `NaN` automatically.

### Missing Columns
If you specify a column in `all_processable_columns` that doesn't exist in your data, it will be skipped.

### Slow Processing
The player availability step can be slow for large datasets. Consider processing one season at a time.

---

## Next Steps

After preprocessing, you can:
1. Drop rows with too many NaN values (early season games)
2. Impute missing values
3. Scale/normalize features
4. Select important features
5. Train your model!
