"""
Example: Using preprocessor.py WITH advanced box scores
This is the RECOMMENDED approach - simpler and more reliable
"""

import sys
sys.path.append('/Users/nicholashorton/Documents/NBA Data')

from preprocessor import ProcessingConfig, DataProcessor
import pandas as pd

# ============================================================================
# STEP 1: Define which columns to process
# ============================================================================

all_processable_columns = [
    # === COUNTING STATS (for cumulative season totals + rolling averages) ===
    'FGM', 'FGA',           # Field goals made/attempted
    'FG3M', 'FG3A',         # Three-pointers made/attempted
    'FTM', 'FTA',           # Free throws made/attempted
    'OREB', 'DREB', 'REB',  # Offensive/Defensive/Total rebounds
    'AST',                  # Assists
    'STL',                  # Steals
    'BLK',                  # Blocks
    'TOV',                  # Turnovers
    'PF',                   # Personal fouls
    'PTS',                  # Points
    'PLUS_MINUS',           # Plus/minus

    # === ADVANCED METRICS (already in advanced box scores!) ===
    'OFF_RATING',           # Points per 100 possessions
    'DEF_RATING',           # Points allowed per 100 possessions
    'NET_RATING',           # Net rating
    'E_OFF_RATING',         # Estimated offensive rating
    'E_DEF_RATING',         # Estimated defensive rating
    'E_NET_RATING',         # Estimated net rating
    'PACE',                 # Pace (possessions per 48 min)
    'E_PACE',               # Estimated pace
    'PACE_PER40',           # Pace per 40 minutes
    'POSS',                 # Total possessions
    'PIE',                  # Player Impact Estimate

    # === PERCENTAGE/RATIO METRICS (already in advanced box scores!) ===
    'AST_PCT',              # Assist percentage
    'AST_TOV',              # Assist to turnover ratio
    'AST_RATIO',            # Assist ratio
    'OREB_PCT',             # Offensive rebound percentage
    'DREB_PCT',             # Defensive rebound percentage
    'REB_PCT',              # Total rebound percentage
    'TM_TOV_PCT',           # Team turnover percentage
    'EFG_PCT',              # Effective field goal percentage
    'TS_PCT',               # True shooting percentage
    'USG_PCT',              # Usage percentage
    'E_USG_PCT'             # Estimated usage percentage
]

# ============================================================================
# STEP 2: Create configuration
# ============================================================================

config = ProcessingConfig(
    all_processable_columns=all_processable_columns,
    rolling_window=5,       # Use 5-game rolling average (adjustable)

    # NO DERIVED METRICS NEEDED - we already have them in advanced box scores!
    derived_metrics={},

    # Elo rating parameters (optional to customize)
    initial_elo=1500,
    k_factor=20,
    home_advantage=100,
    elo_width=400
)

# ============================================================================
# STEP 3: Initialize processor and run
# ============================================================================

print("Initializing DataProcessor...")
processor = DataProcessor(config)

print("\nProcessing data...")
print("This will:")
print("  1. Load and merge advanced + traditional box scores")
print("  2. Add home/away context")
print("  3. Calculate rolling (5-game avg) and cumulative stats")
print("  4. Prepare game-level dataset (home vs away)")
print("  5. Calculate Elo ratings")
print("  6. Process player availability")
print("  7. Calculate difference metrics")

df = processor.process(
    advanced_path='Data/Advanced_box_scores_2023.csv',
    traditional_path='Data/traditional_box_scores_2023.csv',
    player_boxscores='Data/player_box_scores_2023.csv'
)

# ============================================================================
# STEP 4: Filter nonsensical columns
# ============================================================================

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
        print(f"\nFiltering out {len(cols_to_drop)} nonsensical cumulative columns...")
        df = df.drop(columns=cols_to_drop)
        print(f"Examples dropped: {', '.join(cols_to_drop[:5])}{'...' if len(cols_to_drop) > 5 else ''}")

    return df

print("\nFiltering nonsensical columns...")
print("(Removing cumulative versions of rate/percentage stats)")
df = filter_nonsensical_columns(df)

# ============================================================================
# STEP 5: Inspect results
# ============================================================================

print(f"\n{'='*70}")
print(f"Processing complete!")
print(f"{'='*70}")
print(f"Total games processed: {len(df)}")
print(f"Total features: {len(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(2))

print(f"\nColumn categories:")
print(f"  - Elo columns: {len([c for c in df.columns if 'ELO' in c])}")
print(f"  - Rolling columns: {len([c for c in df.columns if '_rolling_' in c])}")
print(f"  - Cumulative columns: {len([c for c in df.columns if '_cumulative_' in c])}")
print(f"  - Missing player columns: {len([c for c in df.columns if 'missing_' in c])}")

# ============================================================================
# STEP 6: Save processed data
# ============================================================================

output_path = 'Data/final_preprocessed_data.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# ============================================================================
# STEP 7: Quick data quality check
# ============================================================================

print(f"\n{'='*70}")
print("Data Quality Check:")
print(f"{'='*70}")

# Check for missing values
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 10]

if len(high_missing) > 0:
    print(f"\n⚠️  Columns with >10% missing values:")
    for col, pct in high_missing.items():
        print(f"  {col}: {pct:.1f}%")
else:
    print("\n✅ No columns with >10% missing values")

# Check for infinite values
inf_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and
            (df[col] == float('inf')).any() or (df[col] == float('-inf')).any()]

if len(inf_cols) > 0:
    print(f"\n⚠️  Columns with infinite values: {inf_cols}")
else:
    print("✅ No infinite values detected")

print(f"\n{'='*70}")
print("Ready for modeling!")
print(f"{'='*70}")
