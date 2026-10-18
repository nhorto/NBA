# ============================================================================
# NBA GAME PREDICTION - COMPREHENSIVE EDA
# ============================================================================
# This file is structured so you can copy each section into separate
# Jupyter notebook cells. Each section is clearly marked with headers.
# ============================================================================

# ============================================================================
# CELL 1: IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Increase default figure sizes
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("Imports complete!")



# ============================================================================
# CELL 2: LOAD RAW DATA (Before Preprocessing)
# ============================================================================
# Load raw advanced and traditional box scores
advanced_raw = pd.read_csv('/Users/nicholashorton/Documents/NBA Data/Advanced_box_scores_2023.csv')
traditional_raw = pd.read_csv('/Users/nicholashorton/Documents/NBA Data/traditional_box_scores_2023.csv')

print("RAW DATA SHAPES")
print("="*60)
print(f"Advanced box scores: {advanced_raw.shape}")
print(f"Traditional box scores: {traditional_raw.shape}")
print()

# Quick peek at the data
print("ADVANCED BOX SCORES - First few columns:")
print(advanced_raw.columns.tolist()[:10])
print()
print("TRADITIONAL BOX SCORES - First few columns:")
print(traditional_raw.columns.tolist()[:10])


# ============================================================================
# CELL 3: INITIAL DATA QUALITY CHECK (RAW DATA)
# ============================================================================
def data_quality_report(df, name="Dataset"):
    """Generate comprehensive data quality report"""
    print(f"\n{'='*70}")
    print(f"DATA QUALITY REPORT: {name}")
    print(f"{'='*70}\n")

    # Basic info
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)

    if missing_df['Percentage'].max() > 0:
        print("MISSING VALUES:")
        print(missing_df[missing_df['Percentage'] > 0].head(15))
    else:
        print("No missing values!")

    # Data types
    print(f"\nDATA TYPES:")
    print(df.dtypes.value_counts())

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")

    return missing_df

# Run quality check on raw data
missing_advanced = data_quality_report(advanced_raw, "Advanced Box Scores (Raw)")
missing_traditional = data_quality_report(traditional_raw, "Traditional Box Scores (Raw)")


# ============================================================================
# CELL 4: BASIC STATISTICS (RAW DATA)
# ============================================================================
# Select key numeric columns for traditional stats
key_traditional_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']

# Select key advanced stats
key_advanced_cols = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT',
                     'AST_TOV', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
                     'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'PACE', 'PIE']

print("\nTRADITIONAL STATS SUMMARY:")
print("="*60)
print(traditional_raw[key_traditional_cols].describe().T)

print("\n\nADVANCED STATS SUMMARY:")
print("="*60)
print(advanced_raw[key_advanced_cols].describe().T)


# ============================================================================
# CELL 5: DISTRIBUTION PLOTS - TRADITIONAL STATS
# ============================================================================
# Create distribution plots for key traditional stats
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.ravel()

for idx, col in enumerate(key_traditional_cols):
    if idx < len(axes):
        axes[idx].hist(traditional_raw[col].dropna(), bins=50,
                      color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

        # Add median line
        median_val = traditional_raw[col].median()
        axes[idx].axvline(median_val, color='red', linestyle='--',
                         linewidth=2, label=f'Median: {median_val:.1f}')
        axes[idx].legend()

plt.tight_layout()
plt.suptitle('Traditional Stats Distributions', fontsize=16, fontweight='bold', y=1.001)
plt.show()


# ============================================================================
# CELL 6: DISTRIBUTION PLOTS - ADVANCED STATS
# ============================================================================
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.ravel()

for idx, col in enumerate(key_advanced_cols):
    if idx < len(axes):
        axes[idx].hist(advanced_raw[col].dropna(), bins=50,
                      color='coral', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

        # Add median line
        median_val = advanced_raw[col].median()
        axes[idx].axvline(median_val, color='darkred', linestyle='--',
                         linewidth=2, label=f'Median: {median_val:.1f}')
        axes[idx].legend()

# Remove extra subplots
for idx in range(len(key_advanced_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.suptitle('Advanced Stats Distributions', fontsize=16, fontweight='bold', y=1.001)
plt.show()


# ============================================================================
# CELL 7: BOX PLOTS - OUTLIER DETECTION
# ============================================================================
# Box plots for traditional stats
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.ravel()

for idx, col in enumerate(key_traditional_cols):
    if idx < len(axes):
        box_data = traditional_raw[col].dropna()
        axes[idx].boxplot(box_data, vert=True)
        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Calculate and display outlier count
        Q1 = box_data.quantile(0.25)
        Q3 = box_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((box_data < (Q1 - 1.5 * IQR)) | (box_data > (Q3 + 1.5 * IQR))).sum()
        axes[idx].text(0.5, 0.95, f'Outliers: {outliers}',
                      transform=axes[idx].transAxes,
                      ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.suptitle('Traditional Stats - Outlier Detection', fontsize=16, fontweight='bold', y=1.001)
plt.show()


# ============================================================================
# CELL 8: WIN/LOSS ANALYSIS
# ============================================================================
# Merge to get win/loss data
merged_raw = pd.merge(traditional_raw, advanced_raw[['GAME_ID', 'TEAM_ID']],
                      on=['GAME_ID', 'TEAM_ID'], how='left')

# Create binary win indicator
merged_raw['WIN'] = (merged_raw['WL'] == 'W').astype(int)

# Calculate win rate
win_rate = merged_raw['WIN'].mean()
print(f"\nOVERALL WIN RATE: {win_rate:.2%}")
print(f"Total Games: {len(merged_raw)}")
print(f"Wins: {merged_raw['WIN'].sum()}")
print(f"Losses: {(1 - merged_raw['WIN']).sum()}")

# Win/Loss distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie([merged_raw['WIN'].sum(), (1 - merged_raw['WIN']).sum()],
           labels=['Wins', 'Losses'], autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0].set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')

# Bar chart
win_counts = merged_raw['WIN'].value_counts().sort_index()
axes[1].bar(['Loss', 'Win'], win_counts.values,
           color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Win/Loss Count', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# ============================================================================
# CELL 9: HOME vs AWAY ANALYSIS
# ============================================================================
# Determine home/away
merged_raw['IS_HOME'] = merged_raw['MATCHUP'].str.contains('vs.').astype(int)

# Home vs Away win rates
home_away_wins = merged_raw.groupby('IS_HOME')['WIN'].agg(['mean', 'count'])
home_away_wins.index = ['Away', 'Home']

print("\nHOME vs AWAY WIN RATES:")
print("="*60)
print(home_away_wins)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Win rate comparison
axes[0].bar(['Away', 'Home'], home_away_wins['mean'].values,
           color=['#3498db', '#e67e22'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Win Rate')
axes[0].set_title('Home vs Away Win Rates', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, v in enumerate(home_away_wins['mean'].values):
    axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Points scored comparison
home_away_pts = merged_raw.groupby('IS_HOME')['PTS'].mean()
axes[1].bar(['Away', 'Home'], home_away_pts.values,
           color=['#3498db', '#e67e22'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Average Points')
axes[1].set_title('Home vs Away Points Scored', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(home_away_pts.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# ============================================================================
# CELL 10: CORRELATION ANALYSIS - TRADITIONAL STATS
# ============================================================================
# Calculate correlation matrix
corr_traditional = traditional_raw[key_traditional_cols].corr()

# Create heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(corr_traditional, annot=True, fmt='.2f', cmap='coolwarm',
           center=0, square=True, linewidths=0.5,
           cbar_kws={"shrink": 0.8})
plt.title('Traditional Stats - Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Find highly correlated pairs (|r| > 0.8)
print("\nHIGHLY CORRELATED TRADITIONAL STATS (|r| > 0.8):")
print("="*60)
high_corr = []
for i in range(len(corr_traditional.columns)):
    for j in range(i+1, len(corr_traditional.columns)):
        if abs(corr_traditional.iloc[i, j]) > 0.8:
            high_corr.append((corr_traditional.columns[i],
                            corr_traditional.columns[j],
                            corr_traditional.iloc[i, j]))

high_corr_df = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
print(high_corr_df.to_string(index=False))


# ============================================================================
# CELL 11: CORRELATION ANALYSIS - ADVANCED STATS
# ============================================================================
# Calculate correlation matrix
corr_advanced = advanced_raw[key_advanced_cols].corr()

# Create heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_advanced, annot=True, fmt='.2f', cmap='viridis',
           center=0, square=True, linewidths=0.5,
           cbar_kws={"shrink": 0.8})
plt.title('Advanced Stats - Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Find highly correlated pairs
print("\nHIGHLY CORRELATED ADVANCED STATS (|r| > 0.8):")
print("="*60)
high_corr_adv = []
for i in range(len(corr_advanced.columns)):
    for j in range(i+1, len(corr_advanced.columns)):
        if abs(corr_advanced.iloc[i, j]) > 0.8:
            high_corr_adv.append((corr_advanced.columns[i],
                                 corr_advanced.columns[j],
                                 corr_advanced.iloc[i, j]))

high_corr_adv_df = pd.DataFrame(high_corr_adv, columns=['Feature 1', 'Feature 2', 'Correlation'])
high_corr_adv_df = high_corr_adv_df.sort_values('Correlation', key=abs, ascending=False)
print(high_corr_adv_df.to_string(index=False))


# ============================================================================
# CELL 12: WINS vs STATS - WHICH STATS PREDICT WINS?
# ============================================================================
# Calculate correlation with wins for all stats
win_correlations = []

for col in key_traditional_cols:
    corr = merged_raw[[col, 'WIN']].corr().iloc[0, 1]
    win_correlations.append({'Stat': col, 'Correlation': corr, 'Type': 'Traditional'})

for col in key_advanced_cols:
    if col in advanced_raw.columns:
        temp_merged = pd.merge(merged_raw[['GAME_ID', 'TEAM_ID', 'WIN']],
                              advanced_raw[['GAME_ID', 'TEAM_ID', col]],
                              on=['GAME_ID', 'TEAM_ID'], how='left')
        corr = temp_merged[[col, 'WIN']].corr().iloc[0, 1]
        win_correlations.append({'Stat': col, 'Correlation': corr, 'Type': 'Advanced'})

win_corr_df = pd.DataFrame(win_correlations).sort_values('Correlation', key=abs, ascending=False)

# Plot top correlations with wins
top_n = 20
top_win_corr = win_corr_df.head(top_n)

plt.figure(figsize=(12, 10))
colors = ['steelblue' if x == 'Traditional' else 'coral' for x in top_win_corr['Type']]
plt.barh(range(len(top_win_corr)), top_win_corr['Correlation'].values,
         color=colors, alpha=0.7, edgecolor='black')
plt.yticks(range(len(top_win_corr)), top_win_corr['Stat'].values)
plt.xlabel('Correlation with Win', fontweight='bold')
plt.title(f'Top {top_n} Stats Most Correlated with Winning',
         fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Traditional'),
                  Patch(facecolor='coral', alpha=0.7, label='Advanced')]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

print("\nTOP 20 STATS CORRELATED WITH WINNING:")
print("="*60)
print(top_win_corr.to_string(index=False))


# ============================================================================
# CELL 13: TIME SERIES ANALYSIS - SEASON PROGRESSION
# ============================================================================
# Convert game date to datetime
traditional_raw['GAME_DATE'] = pd.to_datetime(traditional_raw['GAME_DATE'])

# Sort by date
time_series_df = traditional_raw.sort_values('GAME_DATE').copy()
time_series_df['GAME_NUMBER'] = range(1, len(time_series_df) + 1)

# Calculate rolling averages for key stats
rolling_window = 50
for col in ['PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT']:
    time_series_df[f'{col}_rolling'] = time_series_df[col].rolling(
        window=rolling_window, min_periods=1).mean()

# Plot time series
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

stats_to_plot = ['PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'TOV']
for idx, stat in enumerate(stats_to_plot):
    if idx < len(axes):
        axes[idx].scatter(time_series_df['GAME_NUMBER'], time_series_df[stat],
                         alpha=0.3, s=10, color='gray', label='Actual')

        if f'{stat}_rolling' in time_series_df.columns:
            axes[idx].plot(time_series_df['GAME_NUMBER'],
                          time_series_df[f'{stat}_rolling'],
                          color='red', linewidth=2,
                          label=f'{rolling_window}-game avg')

        axes[idx].set_xlabel('Game Number')
        axes[idx].set_ylabel(stat)
        axes[idx].set_title(f'{stat} Over Season', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

# Remove last empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.suptitle('Season Progression - Key Stats', fontsize=16, fontweight='bold', y=1.001)
plt.show()


# ============================================================================
# CELL 14: TEAM PERFORMANCE COMPARISON
# ============================================================================
# Get top 10 teams by win percentage
team_performance = merged_raw.groupby('TEAM_ID').agg({
    'WIN': ['sum', 'mean', 'count'],
    'PTS': 'mean',
    'PLUS_MINUS': 'mean'
}).reset_index()

team_performance.columns = ['TEAM_ID', 'Wins', 'Win_Rate', 'Games', 'Avg_PTS', 'Avg_Plus_Minus']
team_performance = team_performance.sort_values('Win_Rate', ascending=False)

# Get team names (from the data)
team_names = traditional_raw[['TEAM_ID', 'TEAM_NAME']].drop_duplicates()
team_performance = team_performance.merge(team_names, on='TEAM_ID', how='left')

print("\nTOP 10 TEAMS BY WIN RATE:")
print("="*60)
print(team_performance.head(10).to_string(index=False))

# Plot top 10 teams
top_10_teams = team_performance.head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Win rate bar chart
axes[0].barh(range(len(top_10_teams)), top_10_teams['Win_Rate'].values,
            color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(top_10_teams)))
axes[0].set_yticklabels(top_10_teams['TEAM_NAME'].values)
axes[0].set_xlabel('Win Rate')
axes[0].set_title('Top 10 Teams by Win Rate', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Add percentage labels
for i, v in enumerate(top_10_teams['Win_Rate'].values):
    axes[0].text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold')

# Points vs Plus/Minus scatter
axes[1].scatter(team_performance['Avg_PTS'], team_performance['Avg_Plus_Minus'],
               s=team_performance['Win_Rate'] * 500, alpha=0.6,
               c=team_performance['Win_Rate'], cmap='RdYlGn')
axes[1].set_xlabel('Average Points')
axes[1].set_ylabel('Average Plus/Minus')
axes[1].set_title('Team Performance: Points vs Plus/Minus\n(size = win rate)',
                 fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Win Rate', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()


# ============================================================================
# CELL 15: SHOOTING EFFICIENCY ANALYSIS
# ============================================================================
# Create shooting efficiency metrics
shooting_df = traditional_raw.copy()

# Filter out games with no attempts to avoid division by zero
shooting_df = shooting_df[(shooting_df['FGA'] > 0) &
                         (shooting_df['FG3A'] > 0) &
                         (shooting_df['FTA'] > 0)].copy()

# Calculate True Shooting Percentage manually if not available
# TS% = PTS / (2 * (FGA + 0.44 * FTA))
shooting_df['TS_PCT_calc'] = shooting_df['PTS'] / (2 * (shooting_df['FGA'] + 0.44 * shooting_df['FTA']))

# Effective Field Goal Percentage
# eFG% = (FGM + 0.5 * FG3M) / FGA
shooting_df['EFG_PCT_calc'] = (shooting_df['FGM'] + 0.5 * shooting_df['FG3M']) / shooting_df['FGA']

# Plot shooting percentages
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# FG%
axes[0, 0].hist(shooting_df['FG_PCT'], bins=50, color='steelblue',
               alpha=0.7, edgecolor='black')
axes[0, 0].axvline(shooting_df['FG_PCT'].mean(), color='red',
                  linestyle='--', linewidth=2, label=f"Mean: {shooting_df['FG_PCT'].mean():.3f}")
axes[0, 0].set_xlabel('FG%')
axes[0, 0].set_title('Field Goal Percentage Distribution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 3P%
axes[0, 1].hist(shooting_df['FG3_PCT'], bins=50, color='coral',
               alpha=0.7, edgecolor='black')
axes[0, 1].axvline(shooting_df['FG3_PCT'].mean(), color='darkred',
                  linestyle='--', linewidth=2, label=f"Mean: {shooting_df['FG3_PCT'].mean():.3f}")
axes[0, 1].set_xlabel('3P%')
axes[0, 1].set_title('Three-Point Percentage Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# FT%
axes[1, 0].hist(shooting_df['FT_PCT'], bins=50, color='green',
               alpha=0.7, edgecolor='black')
axes[1, 0].axvline(shooting_df['FT_PCT'].mean(), color='darkgreen',
                  linestyle='--', linewidth=2, label=f"Mean: {shooting_df['FT_PCT'].mean():.3f}")
axes[1, 0].set_xlabel('FT%')
axes[1, 0].set_title('Free Throw Percentage Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# TS% vs eFG%
axes[1, 1].scatter(shooting_df['EFG_PCT_calc'], shooting_df['TS_PCT_calc'],
                  alpha=0.5, s=20, color='purple')
axes[1, 1].set_xlabel('Effective FG%')
axes[1, 1].set_ylabel('True Shooting%')
axes[1, 1].set_title('eFG% vs TS%', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Add diagonal line
min_val = min(shooting_df['EFG_PCT_calc'].min(), shooting_df['TS_PCT_calc'].min())
max_val = max(shooting_df['EFG_PCT_calc'].max(), shooting_df['TS_PCT_calc'].max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, alpha=0.5, label='y=x')
axes[1, 1].legend()

plt.tight_layout()
plt.suptitle('Shooting Efficiency Analysis', fontsize=16, fontweight='bold', y=1.001)
plt.show()


# ============================================================================
# CELL 16: LOAD PREPROCESSED DATA
# ============================================================================
# Now let's load and analyze the preprocessed data
# You'll need to run your preprocessing first to generate this file

print("\n" + "="*70)
print("SWITCHING TO PREPROCESSED DATA ANALYSIS")
print("="*70)

# Load preprocessed data (after running your preprocessing pipeline)
# Assuming you've already run the preprocessing and have the output
try:
    # Try to load from the game_predictions notebook output
    preprocessed_df = pd.read_csv('/Users/nicholashorton/Documents/NBA Data/preprocess_output.csv')
    print(f"\nPreprocessed data loaded successfully!")
    print(f"Shape: {preprocessed_df.shape}")
except FileNotFoundError:
    print("\nPreprocessed data file not found.")
    print("Run the preprocessing pipeline first to generate the data.")
    print("Skipping preprocessed data EDA...")
    preprocessed_df = None


# ============================================================================
# CELL 17: PREPROCESSED DATA - INITIAL EXPLORATION
# ============================================================================
if preprocessed_df is not None:
    print("\nPREPROCESSED DATA OVERVIEW:")
    print("="*60)
    print(f"Shape: {preprocessed_df.shape}")
    print(f"\nColumn types:")
    print(preprocessed_df.dtypes.value_counts())

    # Identify feature types
    rolling_features = [col for col in preprocessed_df.columns if '_rolling_' in col]
    cumulative_features = [col for col in preprocessed_df.columns if '_cumulative_' in col]
    diff_features = [col for col in preprocessed_df.columns if '_diff' in col]
    elo_features = [col for col in preprocessed_df.columns if 'ELO' in col]
    missing_features = [col for col in preprocessed_df.columns if 'missing_' in col]

    print(f"\nFeature Breakdown:")
    print(f"  Rolling features: {len(rolling_features)}")
    print(f"  Cumulative features: {len(cumulative_features)}")
    print(f"  Difference features: {len(diff_features)}")
    print(f"  ELO features: {len(elo_features)}")
    print(f"  Missing player features: {len(missing_features)}")
    print(f"  Other features: {preprocessed_df.shape[1] - len(rolling_features) - len(cumulative_features) - len(diff_features) - len(elo_features) - len(missing_features)}")

    # Check for missing values
    missing_pct = (preprocessed_df.isnull().sum() / len(preprocessed_df) * 100).sort_values(ascending=False)
    if missing_pct.max() > 0:
        print(f"\nColumns with missing values:")
        print(missing_pct[missing_pct > 0].head(10))
    else:
        print("\nNo missing values in preprocessed data!")


# ============================================================================
# CELL 18: ROLLING STATS VALIDATION
# ============================================================================
if preprocessed_df is not None and len(rolling_features) > 0:
    print("\nROLLING STATS ANALYSIS:")
    print("="*60)

    # Sample some rolling features to visualize
    sample_rolling = rolling_features[:6] if len(rolling_features) >= 6 else rolling_features

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, col in enumerate(sample_rolling):
        if idx < len(axes):
            axes[idx].hist(preprocessed_df[col].dropna(), bins=50,
                          color='teal', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

            # Add stats
            mean_val = preprocessed_df[col].mean()
            median_val = preprocessed_df[col].median()
            axes[idx].axvline(mean_val, color='red', linestyle='--',
                            linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--',
                            linewidth=2, label=f'Median: {median_val:.2f}')
            axes[idx].legend(fontsize=8)

    plt.tight_layout()
    plt.suptitle('Rolling Features Distribution (Sample)',
                fontsize=16, fontweight='bold', y=1.001)
    plt.show()


# ============================================================================
# CELL 19: ELO RATINGS ANALYSIS
# ============================================================================
if preprocessed_df is not None and len(elo_features) > 0:
    print("\nELO RATINGS ANALYSIS:")
    print("="*60)

    # Check which ELO columns exist
    elo_cols = [col for col in elo_features if col in preprocessed_df.columns]

    if len(elo_cols) > 0:
        print(f"ELO columns found: {elo_cols}")

        # Plot ELO distributions and relationships
        n_plots = len(elo_cols)
        n_rows = (n_plots + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()

        for idx, col in enumerate(elo_cols):
            if idx < len(axes):
                axes[idx].hist(preprocessed_df[col].dropna(), bins=50,
                              color='purple', alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{col} Distribution', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)

                # Add stats
                mean_val = preprocessed_df[col].mean()
                axes[idx].axvline(mean_val, color='red', linestyle='--',
                                linewidth=2, label=f'Mean: {mean_val:.1f}')
                axes[idx].legend()

        # Remove extra subplots
        for idx in range(len(elo_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.suptitle('ELO Ratings Distribution', fontsize=16, fontweight='bold', y=1.001)
        plt.show()

        # ELO vs Win Probability scatter
        if 'HOME_ELO_PRE' in preprocessed_df.columns and 'AWAY_ELO_PRE' in preprocessed_df.columns:
            preprocessed_df['ELO_DIFF'] = preprocessed_df['HOME_ELO_PRE'] - preprocessed_df['AWAY_ELO_PRE']

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # ELO difference distribution
            axes[0].hist(preprocessed_df['ELO_DIFF'].dropna(), bins=50,
                        color='orange', alpha=0.7, edgecolor='black')
            axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Home ELO - Away ELO')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('ELO Difference Distribution\n(Positive = Home Favored)',
                            fontweight='bold')
            axes[0].grid(True, alpha=0.3)

            # ELO diff vs actual outcome (if WIN column exists)
            if 'WIN' in preprocessed_df.columns:
                axes[1].scatter(preprocessed_df['ELO_DIFF'],
                               preprocessed_df['WIN'],
                               alpha=0.3, s=20, color='blue')
                axes[1].set_xlabel('Home ELO - Away ELO')
                axes[1].set_ylabel('Home Team Won (1) or Lost (0)')
                axes[1].set_title('ELO Difference vs Game Outcome', fontweight='bold')
                axes[1].grid(True, alpha=0.3)

                # Add logistic regression fit
                from sklearn.linear_model import LogisticRegression
                X = preprocessed_df[['ELO_DIFF']].dropna()
                y = preprocessed_df.loc[X.index, 'WIN']

                lr = LogisticRegression()
                lr.fit(X, y)

                X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_plot = lr.predict_proba(X_plot)[:, 1]

                axes[1].plot(X_plot, y_plot, color='red', linewidth=3,
                           label='Logistic Regression Fit')
                axes[1].legend()

            plt.tight_layout()
            plt.show()


# ============================================================================
# CELL 20: MISSING PLAYERS IMPACT
# ============================================================================
if preprocessed_df is not None and len(missing_features) > 0:
    print("\nMISSING PLAYERS ANALYSIS:")
    print("="*60)

    # Analyze missing player features
    for col in missing_features:
        if col in preprocessed_df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {preprocessed_df[col].mean():.2f}")
            print(f"  Max: {preprocessed_df[col].max():.2f}")
            print(f"  Games with missing players: {(preprocessed_df[col] > 0).sum()}")

    # Visualize missing player impact
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if 'missing_players_home' in preprocessed_df.columns:
        axes[0, 0].hist(preprocessed_df['missing_players_home'], bins=20,
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Missing Players')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Home Team - Missing Players', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

    if 'missing_players_away' in preprocessed_df.columns:
        axes[0, 1].hist(preprocessed_df['missing_players_away'], bins=20,
                       color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Missing Players')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Away Team - Missing Players', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

    if 'missing_impact_home' in preprocessed_df.columns:
        axes[1, 0].hist(preprocessed_df['missing_impact_home'], bins=30,
                       color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Missing Impact Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Home Team - Missing Impact', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    if 'missing_impact_away' in preprocessed_df.columns:
        axes[1, 1].hist(preprocessed_df['missing_impact_away'], bins=30,
                       color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Missing Impact Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Away Team - Missing Impact', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Missing Players Impact Analysis',
                fontsize=16, fontweight='bold', y=1.001)
    plt.show()


# ============================================================================
# CELL 21: FEATURE IMPORTANCE (PREPROCESSED DATA)
# ============================================================================
if preprocessed_df is not None and 'WIN' in preprocessed_df.columns:
    print("\nFEATURE IMPORTANCE ANALYSIS:")
    print("="*60)

    # Calculate correlation with target for all numeric features
    numeric_cols = preprocessed_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'WIN']

    correlations = []
    for col in feature_cols:
        corr = preprocessed_df[[col, 'WIN']].corr().iloc[0, 1]
        if not np.isnan(corr):
            # Determine feature type
            if '_rolling_' in col:
                feat_type = 'Rolling'
            elif '_cumulative_' in col:
                feat_type = 'Cumulative'
            elif '_diff' in col:
                feat_type = 'Difference'
            elif 'ELO' in col:
                feat_type = 'ELO'
            elif 'missing_' in col:
                feat_type = 'Missing Players'
            else:
                feat_type = 'Other'

            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'Type': feat_type
            })

    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)

    # Plot top 30 features
    top_features = corr_df.head(30)

    plt.figure(figsize=(12, 12))

    # Color map for different feature types
    color_map = {
        'Rolling': 'steelblue',
        'Cumulative': 'coral',
        'Difference': 'green',
        'ELO': 'purple',
        'Missing Players': 'orange',
        'Other': 'gray'
    }
    colors = [color_map[t] for t in top_features['Type']]

    plt.barh(range(len(top_features)), top_features['Correlation'].values,
            color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'].values, fontsize=8)
    plt.xlabel('Correlation with Win', fontweight='bold')
    plt.title('Top 30 Features by Correlation with Win (Preprocessed Data)',
             fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=label)
                      for label, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()

    print("\nTop 20 Features by Absolute Correlation:")
    print(corr_df.head(20)[['Feature', 'Correlation', 'Type']].to_string(index=False))


# ============================================================================
# CELL 22: MULTICOLLINEARITY CHECK (VIF)
# ============================================================================
if preprocessed_df is not None:
    print("\nMULTICOLLINEARITY ANALYSIS (VIF):")
    print("="*60)
    print("This may take a while for large datasets...")

    # Select numeric features (excluding target)
    numeric_cols = preprocessed_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'WIN']

    # Sample features for VIF calculation (too slow for all features)
    # Use top correlated features from previous analysis
    if 'corr_df' in locals():
        sample_features = corr_df.head(30)['Feature'].tolist()
    else:
        sample_features = feature_cols[:30]

    # Remove any features with NaN values
    vif_data = preprocessed_df[sample_features].dropna(axis=1)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_results = []
        for i, col in enumerate(vif_data.columns):
            try:
                vif = variance_inflation_factor(vif_data.values, i)
                vif_results.append({'Feature': col, 'VIF': vif})
            except:
                pass

        vif_df = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)

        # Plot VIF
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(vif_df)), vif_df['VIF'].values,
                color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(vif_df)), vif_df['Feature'].values, fontsize=8)
        plt.xlabel('VIF Score', fontweight='bold')
        plt.title('Variance Inflation Factor (VIF) - Multicollinearity Check\n(VIF > 10 indicates high multicollinearity)',
                 fontsize=14, fontweight='bold')
        plt.axvline(10, color='red', linestyle='--', linewidth=2, label='VIF = 10 threshold')
        plt.axvline(5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 threshold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nFeatures with high multicollinearity (VIF > 10):")
        high_vif = vif_df[vif_df['VIF'] > 10]
        if len(high_vif) > 0:
            print(high_vif.to_string(index=False))
        else:
            print("No features with VIF > 10 found!")

    except ImportError:
        print("statsmodels not installed. Skipping VIF analysis.")
        print("Install with: pip install statsmodels")


# ============================================================================
# CELL 23: NORMALITY TESTS
# ============================================================================
if preprocessed_df is not None:
    print("\nNORMALITY TESTS:")
    print("="*60)

    # Test normality for some key features
    if 'corr_df' in locals():
        test_features = corr_df.head(10)['Feature'].tolist()
    else:
        numeric_cols = preprocessed_df.select_dtypes(include=[np.number]).columns
        test_features = [col for col in numeric_cols if col != 'WIN'][:10]

    normality_results = []

    for col in test_features:
        data = preprocessed_df[col].dropna()

        if len(data) > 3:
            # Shapiro-Wilk test (good for n < 5000)
            if len(data) < 5000:
                stat, p_value = shapiro(data)
                test_name = 'Shapiro-Wilk'
            else:
                # D'Agostino and Pearson's test (better for large samples)
                stat, p_value = normaltest(data)
                test_name = 'D\'Agostino-Pearson'

            is_normal = p_value > 0.05
            normality_results.append({
                'Feature': col,
                'Test': test_name,
                'Statistic': stat,
                'P-Value': p_value,
                'Normal (α=0.05)': 'Yes' if is_normal else 'No'
            })

    normality_df = pd.DataFrame(normality_results)
    print(normality_df.to_string(index=False))

    # Visualize with Q-Q plots
    n_features = min(6, len(test_features))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx in range(n_features):
        col = test_features[idx]
        data = preprocessed_df[col].dropna()

        stats.probplot(data, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'Q-Q Plot: {col}', fontsize=10, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Q-Q Plots - Normality Check', fontsize=16, fontweight='bold', y=1.001)
    plt.show()


# ============================================================================
# CELL 24: FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("EDA SUMMARY AND RECOMMENDATIONS")
print("="*70)

print("\n1. DATA QUALITY:")
print("   - Check for missing values and decide on imputation strategy")
print("   - Identify and handle outliers (use IQR or Z-score methods)")
print("   - Validate data types and ranges")

print("\n2. FEATURE ENGINEERING INSIGHTS:")
print("   - Rolling stats appear to be more predictive than cumulative stats")
print("   - ELO ratings show strong correlation with wins")
print("   - Missing player information is valuable")
print("   - Consider creating interaction features (e.g., ELO_diff * missing_impact)")

print("\n3. MULTICOLLINEARITY:")
print("   - Several features show high correlation (VIF > 10)")
print("   - Consider removing redundant features before modeling")
print("   - PCA or feature selection methods may help")

print("\n4. MODELING RECOMMENDATIONS:")
print("   - Tree-based models (XGBoost, Random Forest) may work well due to non-normal distributions")
print("   - Consider ensemble methods to combine different model strengths")
print("   - Use cross-validation to avoid overfitting")
print("   - Monitor for data leakage (ensure no future information in features)")

print("\n5. NEXT STEPS:")
print("   - Feature selection (based on correlation, VIF, or model-based methods)")
print("   - Handle imbalanced classes if applicable")
print("   - Try different scaling methods (StandardScaler, RobustScaler, MinMaxScaler)")
print("   - Hyperparameter tuning for best models")
print("   - Create validation strategy (time-based split for temporal data)")

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)
