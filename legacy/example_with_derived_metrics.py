"""
Example: Using preprocessor.py WITHOUT advanced box scores
Calculate derived metrics from traditional box scores only

Use this when:
- You don't have advanced box scores
- You want full control over metric calculations
- You want custom formulas
"""

import sys
sys.path.append('/Users/nicholashorton/Documents/NBA Data')

from preprocessor import ProcessingConfig, DataProcessor, DerivedMetric
import pandas as pd

# ============================================================================
# STEP 1: Define calculation functions for derived metrics
# ============================================================================

def calculate_fg_pct(FGM, FGA):
    """Field goal percentage: FGM / FGA"""
    return FGM / FGA

def calculate_fg3_pct(FG3M, FG3A):
    """Three-point percentage: FG3M / FG3A"""
    return FG3M / FG3A

def calculate_ft_pct(FTM, FTA):
    """Free throw percentage: FTM / FTA"""
    return FTM / FTA

def calculate_ast_to_ratio(AST, TOV):
    """Assist to turnover ratio: AST / TOV"""
    return AST / TOV

def calculate_efg_pct(FGM, FG3M, FGA):
    """Effective field goal percentage: (FGM + 0.5 * FG3M) / FGA
    Accounts for 3-pointers being worth more"""
    return (FGM + 0.5 * FG3M) / FGA

def calculate_ts_pct(PTS, FGA, FTA):
    """True shooting percentage: PTS / (2 * (FGA + 0.44 * FTA))
    Most accurate shooting efficiency metric"""
    return PTS / (2 * (FGA + 0.44 * FTA))

def calculate_reb_ratio(OREB, DREB):
    """Rebound ratio: (OREB + DREB) / DREB"""
    return (OREB + DREB) / DREB

def calculate_ast_pct(AST, POSS):
    """Assist percentage: AST / POSS"""
    return AST / POSS

def calculate_oreb_pct(OREB, POSS):
    """Offensive rebound percentage: OREB / POSS"""
    return OREB / POSS

def calculate_dreb_pct(DREB, POSS):
    """Defensive rebound percentage: DREB / POSS"""
    return DREB / POSS

def calculate_reb_pct(REB, POSS):
    """Total rebound percentage: REB / POSS"""
    return REB / POSS

def calculate_tm_tov_pct(TOV, POSS):
    """Team turnover percentage: TOV / POSS"""
    return TOV / POSS

# ============================================================================
# STEP 2: Define derived metrics
# ============================================================================

derived_metrics = {
    # Shooting percentages
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

    # Advanced shooting efficiency
    'EFG_PCT': DerivedMetric(
        columns=['FGM', 'FG3M', 'FGA'],
        formula=calculate_efg_pct
    ),
    'TS_PCT': DerivedMetric(
        columns=['PTS', 'FGA', 'FTA'],
        formula=calculate_ts_pct
    ),

    # Ball handling
    'AST_TO_RATIO': DerivedMetric(
        columns=['AST', 'TOV'],
        formula=calculate_ast_to_ratio
    ),

    # Rebounding
    'REB_RATIO': DerivedMetric(
        columns=['OREB', 'DREB'],
        formula=calculate_reb_ratio
    ),

    # Advanced metrics (require POSS from advanced box scores or estimate)
    # Note: These will only work if POSS column exists
    'AST_PCT': DerivedMetric(
        columns=['AST', 'POSS'],
        formula=calculate_ast_pct
    ),
    'OREB_PCT': DerivedMetric(
        columns=['OREB', 'POSS'],
        formula=calculate_oreb_pct
    ),
    'DREB_PCT': DerivedMetric(
        columns=['DREB', 'POSS'],
        formula=calculate_dreb_pct
    ),
    'REB_PCT': DerivedMetric(
        columns=['REB', 'POSS'],
        formula=calculate_reb_pct
    ),
    'TM_TOV_PCT': DerivedMetric(
        columns=['TOV', 'POSS'],
        formula=calculate_tm_tov_pct
    )
}

# ============================================================================
# STEP 3: Define columns to process
# ============================================================================

# Only counting stats from traditional box scores
all_processable_columns = [
    'FGM', 'FGA',           # Field goals
    'FG3M', 'FG3A',         # Three-pointers
    'FTM', 'FTA',           # Free throws
    'OREB', 'DREB', 'REB',  # Rebounds
    'AST',                  # Assists
    'STL',                  # Steals
    'BLK',                  # Blocks
    'TOV',                  # Turnovers
    'PF',                   # Personal fouls
    'PTS',                  # Points
    'PLUS_MINUS',           # Plus/minus
    'POSS'                  # Possessions (if available from advanced)
]

# Base columns that will be kept for derived calculations
base_columns = [
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'AST', 'TOV', 'PTS', 'OREB', 'DREB', 'REB', 'POSS'
]

# ============================================================================
# STEP 4: Create configuration
# ============================================================================

config = ProcessingConfig(
    all_processable_columns=all_processable_columns,
    base_columns=base_columns,
    rolling_window=5,
    derived_metrics=derived_metrics,

    # Elo parameters
    initial_elo=1500,
    k_factor=20,
    home_advantage=100,
    elo_width=400
)

# ============================================================================
# STEP 5: Initialize processor and run
# ============================================================================

print("Initializing DataProcessor with derived metrics...")
processor = DataProcessor(config)

print("\nProcessing data with custom metric calculations...")
print("This will:")
print("  1. Load and merge advanced + traditional box scores")
print("  2. Add home/away context")
print("  3. Calculate rolling (5-game avg) and cumulative stats")
print("  4. Calculate DERIVED metrics (FG_PCT, EFG_PCT, TS_PCT, etc.)")
print("  5. Prepare game-level dataset (home vs away)")
print("  6. Calculate Elo ratings")
print("  7. Process player availability")
print("  8. Calculate difference metrics")

df = processor.process(
    advanced_path='Data/Advanced_box_scores_2023.csv',
    traditional_path='Data/traditional_box_scores_2023.csv',
    player_boxscores='Data/player_box_scores_2023.csv'
)

# ============================================================================
# STEP 6: Inspect results
# ============================================================================

print(f"\n{'='*70}")
print(f"Processing complete!")
print(f"{'='*70}")
print(f"Total games processed: {len(df)}")
print(f"Total features: {len(df.columns)}")

# Check which derived metrics were created
derived_cols = [col for col in df.columns if any(metric in col for metric in derived_metrics.keys())]
print(f"\nDerived metric columns created: {len(derived_cols)}")
for col in sorted(derived_cols)[:20]:  # Show first 20
    print(f"  - {col}")

if len(derived_cols) > 20:
    print(f"  ... and {len(derived_cols) - 20} more")

# ============================================================================
# STEP 7: Save processed data
# ============================================================================

output_path = 'Data/preprocessed_with_derived_metrics.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# ============================================================================
# BONUS: Compare derived vs original metrics
# ============================================================================

print(f"\n{'='*70}")
print("Comparing Derived vs Original Metrics (if both exist):")
print(f"{'='*70}")

# If we have both advanced box scores AND derived metrics, compare them
comparison_metrics = ['EFG_PCT', 'TS_PCT']

for metric in comparison_metrics:
    # Look for rolling versions (most useful)
    derived_col = f"{metric}_rolling_home"
    original_col = f"{metric}_rolling_home"

    if derived_col in df.columns:
        print(f"\n{metric}:")
        print(f"  Derived stats:")
        print(f"    Mean: {df[derived_col].mean():.4f}")
        print(f"    Std:  {df[derived_col].std():.4f}")
        print(f"    Min:  {df[derived_col].min():.4f}")
        print(f"    Max:  {df[derived_col].max():.4f}")

print(f"\n{'='*70}")
print("Note: Some derived metrics may have NaN or inf values")
print("This happens when dividing by zero (e.g., FGA=0)")
print("The preprocessor automatically replaces inf with NaN")
print(f"{'='*70}")
