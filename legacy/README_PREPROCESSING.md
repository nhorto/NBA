# NBA Data Preprocessing - Complete Guide

## Quick Start

You asked for help understanding your preprocessing pipeline. Here's what you need to know:

### Your Questions - Answered

#### 1. **How do I run the preprocessor?**

**Simple answer**: Run `example_with_advanced_boxscores.py`

```bash
cd "/Users/nicholashorton/Documents/NBA Data"
python example_with_advanced_boxscores.py
```

This will process all three data files (advanced box scores, traditional box scores, player box scores) and create a ready-to-use dataset.

#### 2. **Why were you creating derived metrics if they're already in advanced box scores?**

**You're right to be confused!** Many of the metrics you were calculating (like `EFG_PCT`, `TS_PCT`, `AST_PCT`, `OREB_PCT`, `DREB_PCT`, `REB_PCT`, `TM_TOV_PCT`) **are already in the Advanced Box Scores**.

You were likely:
- Testing different calculation methods
- Not realizing the NBA API already provides these
- Using old code when you only had traditional box scores

**Bottom line**: If you have advanced box scores, you DON'T need to calculate these metrics.

#### 3. **Should I use the same columns for rolling, cumulative, and difference?**

**No!** Different features work better for different transformations:

- **Cumulative** (season totals): Use COUNTING STATS
  - Examples: FGM, FGA, PTS, REB, AST (things that accumulate)

- **Rolling** (recent averages): Use RATES and PERCENTAGES
  - Examples: FG_PCT, NET_RATING, OFF_RATING, PIE (things that show trends)

- **Difference** (home vs away): Use PREDICTIVE ROLLING METRICS
  - Examples: NET_RATING, OFF_RATING, DEF_RATING (things that show advantage)

See `COLUMN_RECOMMENDATIONS.md` for detailed lists.

---

## Files Created For You

I've created several files to help you:

### 1. **README_PREPROCESSING.md** (this file)
Quick overview and answers to your questions

### 2. **PREPROCESSOR_USAGE_GUIDE.md**
Comprehensive guide on how to use preprocessor.py with examples

### 3. **COLUMN_RECOMMENDATIONS.md**
Detailed recommendations on which columns to use for rolling/cumulative/difference
Includes three configurations (Comprehensive, Essential, Traditional-only)

### 4. **example_with_advanced_boxscores.py**
Ready-to-run script using advanced box scores (RECOMMENDED)

### 5. **example_with_derived_metrics.py**
Example showing how to calculate metrics from scratch (if you didn't have advanced box scores)

---

## The Two Approaches

### Approach A: Use Advanced Box Scores (RECOMMENDED)

**When**: You have all three files
- `Advanced_box_scores_2023.csv`
- `traditional_box_scores_2023.csv`
- `player_box_scores_2023.csv`

**Why**: Simpler, faster, more reliable - metrics already calculated by NBA

**How**: Run `example_with_advanced_boxscores.py`

**Result**: Creates `Data/final_preprocessed_data.csv` with:
- Rolling stats (5-game averages)
- Cumulative stats (season totals)
- Elo ratings
- Player availability metrics
- Home vs away differences

---

### Approach B: Calculate Derived Metrics

**When**: You don't have advanced box scores or want custom formulas

**Why**: Full control over calculations

**How**: Run `example_with_derived_metrics.py`

**Result**: Creates `Data/preprocessed_with_derived_metrics.csv` with calculated metrics

---

## What The Preprocessor Does

When you run `processor.process()`, it automatically:

1. **Loads & Merges** advanced + traditional box scores
2. **Adds Context** (home/away indicator, W/L binary)
3. **Calculates Rolling** (5-game moving average for all specified columns)
4. **Calculates Cumulative** (season-to-date totals for all specified columns)
5. **Calculates Derived Metrics** (if specified - e.g., shooting percentages)
6. **Prepares Game Data** (splits home/away into single game-level rows)
7. **Calculates Elo Ratings** (pre-game, post-game, win probability)
8. **Processes Player Availability** (missing players, missing impact, missing stars)
9. **Calculates Differences** (home - away for key metrics)

---

## Output Structure

Your final dataframe will have these column types:

### Core Game Info
- `GAME_ID`, `GAME_DATE`, `TEAM_ID_home`, `TEAM_ID_away`, `WL`

### Elo Features
- `HOME_ELO_PRE`, `AWAY_ELO_PRE` (before game)
- `HOME_ELO_POST`, `AWAY_ELO_POST` (after game)
- `HOME_WIN_PROB` (predicted probability based on Elo)

### Player Availability Features
- `missing_players_home`, `missing_players_away`
- `missing_impact_home`, `missing_impact_away`
- `missing_star_home`, `missing_star_away`

### Rolling Features (5-game averages)
- Format: `{stat}_rolling_home`, `{stat}_rolling_away`
- Examples: `PTS_rolling_home`, `NET_RATING_rolling_home`

### Cumulative Features (season totals)
- Format: `{stat}_cumulative_home`, `{stat}_cumulative_away`
- Examples: `PTS_cumulative_home`, `FGA_cumulative_home`

### Some Difference Features
- The processor automatically calculates some key differences
- These are hard-coded in `calculate_difference_metrics()` method

---

## Common Questions

### Q: Can I change the rolling window?
**A**: Yes! In the config, set `rolling_window=10` (or any number)

### Q: How do I handle NaN values?
**A**: Early season games will have NaNs (not enough history). Either:
1. Drop games with NaNs
2. Impute with mean/median
3. Use `min_periods=1` in rolling (already done)

### Q: Why are some values infinite?
**A**: Division by zero (e.g., 0 FGA). The preprocessor automatically converts `inf` to `NaN`.

### Q: Should I normalize features?
**A**: Yes! After preprocessing, use StandardScaler or MinMaxScaler before modeling.

### Q: Can I use multiple seasons?
**A**: Yes, but be careful:
1. Process each season separately
2. Reset Elo ratings between seasons OR
3. Carry over Elo with regression to the mean

---

## Recommended Workflow

```python
# 1. Process data (includes automatic filtering of nonsensical columns)
python example_with_advanced_boxscores.py

# This script automatically:
# - Processes all data (rolling, cumulative, Elo, player availability)
# - Filters out cumulative versions of rate/percentage stats
# - Saves clean data to 'Data/final_preprocessed_data.csv'

# 2. Load processed data
import pandas as pd
df = pd.read_csv('Data/final_preprocessed_data.csv')

# 3. Handle missing values
df = df.dropna()  # or impute

# 4. Select features
# The data is already filtered (nonsensical cumulative columns removed)
# Use feature importance to select best features

# 5. Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train model
from sklearn.model_selection import train_test_split
# ... your modeling code ...
```

---

## Key Differences: Preprocess.py vs preprocessor.py

### Preprocess.py (OLD)
- Class name: `NBADataProcessor`
- Separate methods for each step
- Manual orchestration needed
- No player availability features
- Less comprehensive

### preprocessor.py (NEW - RECOMMENDED)
- Class name: `DataProcessor`
- Single `.process()` method does everything
- Includes player availability
- More features (Elo, player impact)
- Better error handling
- This is what you should use!

---

## Quick Reference: Column Recommendations

### For CUMULATIVE (season totals):
```python
['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
 'TOV', 'PF', 'PTS', 'POSS']
```

### For ROLLING (recent form):
```python
['OFF_RATING', 'DEF_RATING', 'NET_RATING',
 'PACE', 'PIE', 'EFG_PCT', 'TS_PCT',
 'AST_PCT', 'AST_TOV', 'REB_PCT', 'PLUS_MINUS']
```

### For DIFFERENCE (home - away):
```python
['NET_RATING', 'OFF_RATING', 'DEF_RATING',
 'PACE', 'PIE', 'EFG_PCT', 'TS_PCT', 'REB_PCT']
```

---

## Next Steps

1. **Test the preprocessor**:
   ```bash
   python example_with_advanced_boxscores.py
   ```

2. **Review the output**:
   ```python
   import pandas as pd
   df = pd.read_csv('Data/final_preprocessed_data.csv')
   print(df.info())
   print(df.describe())
   ```

3. **Explore column recommendations**:
   - Read `COLUMN_RECOMMENDATIONS.md`
   - Decide which configuration fits your needs

4. **Build your model**:
   - Use the processed data
   - Start with essential features
   - Use feature selection to optimize

5. **Iterate**:
   - Adjust rolling window
   - Try different Elo parameters
   - Experiment with feature engineering

---

## Need Help?

Check these files:
- **Full usage guide**: `PREPROCESSOR_USAGE_GUIDE.md`
- **Column guide**: `COLUMN_RECOMMENDATIONS.md`
- **Example scripts**: `example_with_advanced_boxscores.py` and `example_with_derived_metrics.py`

---

## Summary

- ✅ Use `preprocessor.py` (not `Preprocess.py`)
- ✅ Run `example_with_advanced_boxscores.py` to get started
- ✅ You DON'T need to calculate derived metrics (they're in advanced box scores)
- ✅ Use different columns for cumulative vs rolling vs difference (see recommendations)
- ✅ The `.process()` method does everything automatically
- ✅ Output is ready for machine learning after basic cleaning

**Happy modeling!** 🏀📊
