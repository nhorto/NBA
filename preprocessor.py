import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Callable, Tuple

@dataclass
class DerivedMetric:
    """Class to define how to calculate a derived metric"""
    columns: List[str]  # List of columns needed for calculation
    formula: Callable  # Function that implements the calculation

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    rolling_columns: Optional[List[str]] = None
    cumulative_columns: Optional[List[str]] = None
    difference_columns: Optional[List[str]] = None
    all_processable_columns: List[str] = None
    base_columns: Optional[List[str]] = None
    rolling_window: int = 5
    derived_metrics: Optional[Dict[str, DerivedMetric]] = None
    calculate_diffs: bool = False

    # Elo configuration parameters
    initial_elo: float = 1500
    k_factor: float = 20
    home_advantage: float = 100
    elo_width: float = 400

@dataclass
class DataProcessor:
    """Class to process data based on the provided configuration"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.team_elos = {}
        self.player_box_scores = None
    
    def load_and_merge_data(self, advanced_path: str, traditional_path: str, player_boxscores: str) -> pd.DataFrame:
        """Load and merge the advanced and traditional box score data"""
        self.player_box_scores = player_boxscores

        # Load the data
        ad = pd.read_csv(advanced_path)
        tr = pd.read_csv(traditional_path)
        
        tr = tr.drop(columns=['TEAM_ABBREVIATION', 'MIN', 'VIDEO_AVAILABLE'])
        ad = ad.drop(columns=['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'MIN'])
        return pd.merge(tr, ad, on=['GAME_ID', 'TEAM_ID'], how='right')
    
    def add_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game context features"""
        df = df.copy()

        # Create home/away indicator
        conditions = [df['MATCHUP'].str.contains('vs.'), df['MATCHUP'].str.contains('@')]
        choices = ['home', 'away']
        df['home_away'] = np.select(conditions, choices, default='unknown')
        
        # Convert WL to binary
        df['WL'] = df['WL'].map({'W': 1, 'L': 0})
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

        group = df.groupby('TEAM_ID')
        DF = []

        for _, group_df in group:
            temp_df = group_df.copy()
            temp_df = temp_df.sort_values(by='GAME_DATE')

            # Calculate cumulative stats and rolling stats
            for col in self.config.all_processable_columns:
                if col in temp_df.columns:
                    temp_df[f'{col}_cumulative'] = temp_df[col].cumsum().shift(1)
                    temp_df[f'{col}_rolling'] = temp_df[col].shift(1).rolling(window=5, min_periods=1).mean()

            # Append after processing all columns
            DF.append(temp_df)

        df = pd.concat(DF, ignore_index=True)

        for metric_name, metric_info in self.config.derived_metrics.items():
            columns_needed = metric_info.columns
            formula = metric_info.formula
            
            # For cumulative metrics
            cumulative_cols = [f'{col}_cumulative' for col in columns_needed]
            if all(col in df.columns for col in cumulative_cols):
                # Create a dictionary of column_name -> column_data
                col_data = {original_col: df[cum_col] 
                        for original_col, cum_col in zip(columns_needed, cumulative_cols)}
                
                # Call the formula with keyword arguments
                df[f'{metric_name}_cumulative'] = formula(**col_data)
                print(f"✅ Calculated {metric_name}_cumulative (dynamic)")
            
            # For rolling metrics
            rolling_cols = [f'{col}_rolling' for col in columns_needed]
            if all(col in df.columns for col in rolling_cols):
                # Create a dictionary of column_name -> column_data
                col_data = {original_col: df[roll_col] 
                        for original_col, roll_col in zip(columns_needed, rolling_cols)}
                
                # Call the formula with keyword arguments
                df[f'{metric_name}_rolling'] = formula(**col_data)
                print(f"✅ Calculated {metric_name}_rolling (dynamic)")

        # Verify what we created
        derived_cumulative_cols = [col for col in df.columns if col.endswith('_cumulative') and any(metric in col for metric in self.config.derived_metrics.keys())]
        derived_rolling_cols = [col for col in df.columns if col.endswith('_rolling') and any(metric in col for metric in self.config.derived_metrics.keys())]

        print(f"\nCreated {len(derived_cumulative_cols)} cumulative derived metrics:")
        for col in derived_cumulative_cols:
            print(f"  - {col}")

        print(f"\nCreated {len(derived_rolling_cols)} rolling derived metrics:")
        for col in derived_rolling_cols:
            print(f"  - {col}")

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Check for any issues
        problematic_cols = []
        for col in df.columns:
            if col.endswith('_cumulative') or col.endswith('_rolling'):
                if df[col].isna().all():
                    problematic_cols.append(col)
                elif df[col].isin([np.inf, -np.inf]).any():
                    problematic_cols.append(col)
        print(f"⚠️  Warning: These columns have issues: {problematic_cols}")

        return df
    
    def prepare_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create game-level dataset with optional home/away differences"""
        # Split into home and away
        home_data = df[df['home_away'] == 'home'].copy()
        away_data = df[df['home_away'] == 'away'].copy()
        
        # Rename columns
        home_data = home_data.rename(columns=lambda x: f"{x}_home" if x not in ['GAME_ID', 'WL', 'GAME_DATE'] else x)
        away_data = away_data.rename(columns=lambda x: f"{x}_away" if x not in ['GAME_ID', 'WL', 'GAME_DATE'] else x)
        
        # Merge home and away data
        game_data = pd.merge(home_data, away_data, on='GAME_ID')
        
        return game_data
    
    def calculate_elo_probability(self, home_elo, away_elo) -> float:
        """Calculate expected probability of home team winning"""
        return 1.0 / (1 + 10 ** ((away_elo - (home_elo + self.config.home_advantage)) / self.config.elo_width))

    def update_elos(self, home_team, away_team, home_won, margin):
        """Update Elo ratings for both teams after a game"""
        # Get current Elo ratings (or default if not existing)
        home_elo = self.team_elos.get(home_team, self.config.initial_elo)
        away_elo = self.team_elos.get(away_team, self.config.initial_elo)
        
        # Calculate expected win probability
        expected = self.calculate_elo_probability(home_elo, away_elo)
        
        # Calculate actual outcome (1 for home win, 0 for home loss)
        actual = float(home_won)
        
        # Optionally, you can use margin to scale Elo change (optional, but common)
        margin_multiplier = max(1, abs(margin) / 10.0)
        elo_change = self.config.k_factor * (actual - expected) * margin_multiplier
        
        # Update team Elos
        self.team_elos[home_team] = home_elo + elo_change
        self.team_elos[away_team] = away_elo - elo_change
        
        return self.team_elos[home_team], self.team_elos[away_team]
    
    def calculate_elo_ratings(self, df):
        """Calculate Elo ratings for all games"""
        df = df.copy()

        # Sort games chronologically
        df = df.sort_values(['GAME_DATE_x', 'GAME_ID']).copy()
        
        # Create columns for Elo ratings
        df['HOME_ELO_PRE'] = np.nan
        df['AWAY_ELO_PRE'] = np.nan
        df['HOME_ELO_POST'] = np.nan
        df['AWAY_ELO_POST'] = np.nan
        df['HOME_WIN_PROB'] = np.nan
        
        # Keep track of all teams we've seen
        seen_teams = set()
        
        # Process each game
        for idx in df.index:
            home_team = df.loc[idx, 'TEAM_ID_home']
            away_team = df.loc[idx, 'TEAM_ID_away']
            print(home_team, idx)
            home_score = df.loc[idx, 'PTS_home']
            away_score = df.loc[idx, 'PTS_away']
            margin = home_score - away_score  # <-- This is the margin!
            
            # Check if this is first appearance for either team
            if home_team not in seen_teams:
                self.team_elos[home_team] = self.config.initial_elo
                seen_teams.add(home_team)
            if away_team not in seen_teams:
                self.team_elos[away_team] = self.config.initial_elo
                seen_teams.add(away_team)
                
            # Get pre-game Elos
            home_elo = self.team_elos[home_team]
            away_elo = self.team_elos[away_team]
            
            # Store pre-game Elos
            df.loc[idx, 'HOME_ELO_PRE'] = home_elo
            df.loc[idx, 'AWAY_ELO_PRE'] = away_elo
            
            # Calculate win probability
            win_prob = self.calculate_elo_probability(home_elo, away_elo)
            df.loc[idx, 'HOME_WIN_PROB'] = win_prob
            
            # Update Elos based on game result
            home_won = df.loc[idx, 'WL_x']
            new_home_elo, new_away_elo = self.update_elos(home_team, away_team, home_won, margin)
            
            # Store post-game Elos
            df.loc[idx, 'HOME_ELO_POST'] = new_home_elo
            df.loc[idx, 'AWAY_ELO_POST'] = new_away_elo

            for col in self.config.base_columns:
                home = col + "_home"
                away = col + "_away"

                if home in df.columns:
                    df.drop(columns=home, inplace=True)

                if away in df.columns:
                    df.drop(columns=away, inplace=True)

            df = df.drop(columns=['Unnamed: 0_x_home','Unnamed: 0_x_away','SEASON_ID_away','GAME_DATE_y','MATCHUP_away','MATCHUP_home','WL_y','home_away_away','Unnamed: 0_y_away','Unnamed: 0_y_home',
                                'home_away_home'])

            df = df.rename(columns={
                'GAME_DATE_x':'GAME_DATE',
                'WL_x':'WL'})

            drop_cols = ['FG_PCT','FG3_PCT','FT_PCT']

            for col in drop_cols:
                home = col+"_home"
                away = col+"_away"

                if home in df.columns:
                    df.drop(columns=home, inplace=True)

                if away in df.columns:
                    df.drop(columns=away, inplace=True)
        
        return df 

    def normalize(self,series):
        """Normalize a series to 0-1 range"""
        if len(series) == 0:
            return series
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series * 0  # Return zeros if all values are the same
        return (series - min_val) / (max_val - min_val)
    
    def get_player_info(self, player_boxscores, cutoff_date=None, game_id=None, team_id=None, home_away=None):
        """Calculate importance score for each player based on season averages up to cutoff_date"""
        
        # Make a copy to avoid modifying original data
        player_data = player_boxscores.copy()
        
        # Filter by cutoff date for importance calculation (historical data only)
        if cutoff_date is not None:
            historical_data = player_data[player_data['GAME_DATE'] < cutoff_date].copy()
        else:
            historical_data = player_data.copy()
        
        # If no historical data, return zeros
        if historical_data.empty:
            return {
                f'missing_players_{home_away}': 0,
                f'missing_impact_{home_away}': 0.0,
                f'missing_star_{home_away}': False
            }
        
        # Clean minutes column for historical data
        historical_data['MIN'] = historical_data['MIN'].astype(str).str.split('.').str[0]
        historical_data['MIN'] = historical_data['MIN'].fillna(0)
        historical_data['MIN'] = pd.to_numeric(historical_data['MIN'], errors='coerce').fillna(0)
        
        # Calculate season stats from historical data only
        season_stats = historical_data.groupby('PLAYER_ID').agg({
            'MIN': 'mean',
            'PLUS_MINUS': 'mean',
            'PTS': 'mean',
            'AST': 'mean',
            'REB': 'mean',
            'TEAM_ID': 'last'
        }).reset_index()
        
        # Calculate importance scores
        season_stats['importance_score'] = (
            self.normalize(season_stats['MIN']) * 0.3 +
            self.normalize(season_stats['PLUS_MINUS']) * 0.2 +
            self.normalize(season_stats['PTS']) * 0.3 +
            self.normalize(season_stats['AST'] + season_stats['REB']) * 0.2
        )
        
        # Get players who actually played in this specific game (can look ahead for this)
        game_players = player_data[
            (player_data['GAME_ID'] == game_id) &
            (player_data['TEAM_ID'] == team_id)
        ]['PLAYER_ID'].unique()
        
        # Get team's regular rotation players (from historical data)
        rotation_players = season_stats[
            (season_stats['TEAM_ID'] == team_id) &
            (season_stats['MIN'] >= 15)
        ]['PLAYER_ID'].values
        
        # Find missing players
        missing_players = set(rotation_players) - set(game_players)
        
        # Calculate impact of missing players
        missing_impact = season_stats[
            season_stats['PLAYER_ID'].isin(missing_players)
        ]['importance_score'].sum()
        
        # Check if team's most important player is missing
        team_top_score = season_stats[
            season_stats['TEAM_ID'] == team_id
        ]['importance_score'].max() if not season_stats[season_stats['TEAM_ID'] == team_id].empty else 0
        
        missing_star = False
        if team_top_score > 0:
            missing_star = any(
                season_stats[season_stats['PLAYER_ID'].isin(missing_players)]['importance_score'] >= team_top_score * 0.9
            )
        
        return {
            f'missing_players_{home_away}': len(missing_players),
            f'missing_impact_{home_away}': missing_impact,
            f'missing_star_{home_away}': missing_star
        }

    def process_player_availability(self, df):
        """Process player availability for all games"""
        
        # Load player data once
        bs = pd.read_csv(self.player_box_scores)
        
        # Make sure GAME_DATE is in both dataframes and is datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Add GAME_DATE to player boxscores if not present
        if 'GAME_DATE' not in bs.columns:
            # Extract game dates from your main df
            game_dates = df[['GAME_ID', 'GAME_DATE']].drop_duplicates()
            bs = bs.merge(game_dates, on='GAME_ID', how='left')
        
        bs['GAME_DATE'] = pd.to_datetime(bs['GAME_DATE'])
        
        # Initialize result columns
        df['missing_players_home'] = 0
        df['missing_impact_home'] = 0.0
        df['missing_star_home'] = False
        df['missing_players_away'] = 0
        df['missing_impact_away'] = 0.0
        df['missing_star_away'] = False
        
        # Process each game
        for idx, row in df.iterrows():
            cutoff_date = row['GAME_DATE']
            game_id = row['GAME_ID']
            team_id_home = row['TEAM_ID_home']
            team_id_away = row['TEAM_ID_away']
            
            # Get home team results
            home_results = self.get_player_info(
                bs, cutoff_date=cutoff_date, game_id=game_id, 
                team_id=team_id_home, home_away='home'
            )
            
            # Get away team results
            away_results = self.get_player_info(
                bs, cutoff_date=cutoff_date, game_id=game_id, 
                team_id=team_id_away, home_away='away'
            )
            
            # Update dataframe with results for this specific row
            for key, value in home_results.items():
                df.at[idx, key] = value
            
            for key, value in away_results.items():
                df.at[idx, key] = value

        df['missing_star_away'] = df['missing_star_away'].map({True: 1, False: 0})
        df['missing_star_home'] = df['missing_star_home'].map({True: 1, False: 0})
        df['WIN'] = (df['WL'] == 1).astype(int)
        
        return df
    
    def calculate_difference_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate difference metrics between home and away teams"""
        df = df.copy()
        
        rolling_basenames = [
            # Team strength & efficiency
            "NET_RATING", "OFF_RATING", "DEF_RATING",
            "E_NET_RATING", "E_OFF_RATING", "E_DEF_RATING",
            "PIE",

            # Shooting efficiency
            "FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT",

            # Possession & tempo
            "PACE", "E_PACE", "PACE_PER40", "POSS",

            # Ball movement & turnovers
            "AST_TO_RATIO", "AST_PCT", "AST_TOV", "TM_TOV_PCT",

            # Rebounding
            "OREB_PCT", "DREB_PCT", "REB_PCT", "REB_RATIO",

            # Game control
            "PLUS_MINUS", "PTS"
        ]
 
        # keep track of original cols we’ll remove
        cols_to_drop = []         
        for base in rolling_basenames:
            home_col = f"{base}_rolling_home"
            away_col = f"{base}_rolling_away"
            diff_col = f"{base}_diff"

            # Skip if either column is missing (protects against typos / schema drift)
            if home_col in df.columns and away_col in df.columns:
                df[diff_col] = df[home_col] - df[away_col]
                cols_to_drop.extend([home_col, away_col])
            else:
                print(f"⚠️  Skipping {base}: expected columns not found in calculate difference metrics.")

        elo_pairs = [
            ("HOME_ELO_PRE", "AWAY_ELO_PRE", "ELO_DIFF"),   # pre-game
            ("HOME_ELO_POST", "AWAY_ELO_POST", "ELO_POST_DIFF")  # optional
        ]

        for home_col, away_col, diff_col in elo_pairs:
            if home_col in df.columns and away_col in df.columns:
                df[diff_col] = df[home_col] - df[away_col]
                cols_to_drop.extend([home_col, away_col])
            else:
                print(f"⚠️  Skipping ELO pair: {home_col}, {away_col} not found in calculate difference metrics.")

        df.drop(columns=cols_to_drop, inplace=True)

        return df

    def process(self, advanced_path: str, traditional_path: str, player_boxscores: str) -> pd.DataFrame:
        """Main processing function to load, merge, and process data"""
        # Load and merge data
        print("Loading and merging data...")
        df = self.load_and_merge_data(advanced_path, traditional_path, player_boxscores)

        print("Adding context features...")
        # Add context features
        df = self.add_context(df)
        
        print("Preparing game-level dataset...")
        # Prepare game-level dataset
        df = self.prepare_game_data(df)
        
        print("Calculating Elo ratings...")
        # Calculate Elo ratings
        df = self.calculate_elo_ratings(df)
        
        print("Calculating derived metrics...")
        # Process player availability
        df = self.process_player_availability(df)

        print("Calculating derived metrics...")
        # Calculate difference metrics
        df = self.calculate_difference_metrics(df)
        
        return df