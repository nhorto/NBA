"""
Quick test to verify the preprocessor fix works
"""

import sys
sys.path.append('/Users/nicholashorton/Documents/NBA Data')

from preprocessor import ProcessingConfig, DataProcessor

# Test configuration with base_columns = None (this was causing the error)
config = ProcessingConfig(
    all_processable_columns=[
        'FGM', 'FGA', 'PTS', 'REB', 'AST',
        'OFF_RATING', 'DEF_RATING', 'NET_RATING'
    ],
    rolling_window=5,
    derived_metrics={},  # Empty dict
    # base_columns is NOT set - defaults to None
)

print("Creating DataProcessor with config...")
print(f"  all_processable_columns: {len(config.all_processable_columns)} columns")
print(f"  base_columns: {config.base_columns}")
print(f"  derived_metrics: {config.derived_metrics}")

processor = DataProcessor(config)

print("\nNow running the full process...")
print("If this works without error, the fix is successful!\n")

try:
    df = processor.process(
        advanced_path='Data/Advanced_box_scores_2023.csv',
        traditional_path='Data/traditional_box_scores_2023.csv',
        player_boxscores='Data/player_box_scores_2023.csv'
    )

    print("\n" + "="*70)
    print("✅ SUCCESS! The preprocessor works correctly now.")
    print("="*70)
    print(f"Processed {len(df)} games with {len(df.columns)} features")

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR: The fix didn't work completely")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
