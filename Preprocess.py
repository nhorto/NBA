from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np

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
    rolling_window: int = 5
    derived_metrics: Optional[Dict[str, DerivedMetric]] = None
    calculate_diffs: bool = True

    # Elo configuration parameters
    initial_elo: float = 1500
    k_factor: float = 20
    home_advantage: float = 100
    elo_width: float = 400


class NBADataProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.team_elos = {}

    def calculate_elo_probability(self, home_elo: float, away_elo: float) -> float:
        """Calculate expected probability of home team winning"""
        return 1.0 / (1 + 10 ** ((away_elo - (home_elo + self.config.home_advantage)) / self.config.elo_width))
    
    def update_elos(self, home_team: int, away_team: int, home_won: int, 
                   margin: float = None) -> Tuple[float, float]:
        """Update Elo ratings for both teams after a game"""
        # Get current Elo ratings (or default if not existing)
        home_elo = self.team_elos.get(home_team, self.config.initial_elo)
        away_elo = self.team_elos.get(away_team, self.config.initial_elo)
        
        # Calculate expected win probability
        expected = self.calculate_elo_probability(home_elo, away_elo)
        
        # Calculate actual outcome (1 for home win, 0 for home loss)
        actual = float(home_won)
        
        # Calculate basic Elo change
        elo_change = self.config.k_factor * (actual - expected)
        
        # Update team Elos
        self.team_elos[home_team] = home_elo + elo_change
        self.team_elos[away_team] = away_elo - elo_change
        
        return self.team_elos[home_team], self.team_elos[away_team]
    
    def calculate_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Elo ratings for all games"""
        # Reset Elo ratings at start of season
        self.team_elos = {}
        
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
            
            # Check if this is first appearance for either team
            if home_team not in seen_teams:
                self.team_elos[home_team] = self.config.initial_elo
                seen_teams.add(home_team)
            if away_team not in seen_teams:
                self.team_elos[away_team] = self.config.initial_elo
                seen_teams.add(away_team)
                
            # Get pre-game Elos
            home_elo = self.team_elos[home_team]  # Now we can use direct lookup
            away_elo = self.team_elos[away_team]
            
            # Store pre-game Elos
            df.loc[idx, 'HOME_ELO_PRE'] = home_elo
            df.loc[idx, 'AWAY_ELO_PRE'] = away_elo
            
            # Calculate win probability
            win_prob = self.calculate_elo_probability(home_elo, away_elo)
            df.loc[idx, 'HOME_WIN_PROB'] = win_prob
            
            # Update Elos based on game result
            home_won = df.loc[idx, 'WL_x']
            new_home_elo, new_away_elo = self.update_elos(home_team, away_team, home_won)
            
            # Store post-game Elos
            df.loc[idx, 'HOME_ELO_POST'] = new_home_elo
            df.loc[idx, 'AWAY_ELO_POST'] = new_away_elo
        
        return df
        
    def load_and_merge_data(self, advanced_path: str, traditional_path: str) -> pd.DataFrame:
        """Load and merge the advanced and traditional box score data"""
        ad = pd.read_csv(advanced_path)
        tr = pd.read_csv(traditional_path)
        
        tr = tr.drop(columns=['TEAM_ABBREVIATION', 'MIN', 'VIDEO_AVAILABLE'])
        ad = ad.drop(columns=['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'MIN'])
        return pd.merge(tr, ad, on=['GAME_ID', 'TEAM_ID'], how='right')
    
    def add_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game context features"""
        # Create home/away indicator
        conditions = [df['MATCHUP'].str.contains('vs.'), df['MATCHUP'].str.contains('@')]
        choices = ['home', 'away']
        df['home_away'] = np.select(conditions, choices, default='unknown')
        
        # Convert WL to binary
        df['WL'] = df['WL'].map({'W': 1, 'L': 0})
        
        return df
    
    def calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages for specified columns"""
        if not self.config.rolling_columns:
            return df
            
        # Create a copy of the original dataframe
        df_processed = df.copy()
        
        # Make sure date is in datetime format and sort
        df_processed['GAME_DATE'] = pd.to_datetime(df_processed['GAME_DATE'])
        df_processed = df_processed.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Process each team separately
        for team in df_processed['TEAM_ID'].unique():
            mask = df_processed['TEAM_ID'] == team
            
            for col in self.config.rolling_columns:
                if col in df_processed.columns:
                    df_processed.loc[mask, f'{col}_rolling'] = (
                        df_processed.loc[mask, col]
                        .shift(1)  # Shift to exclude current game
                        .rolling(
                            window=self.config.rolling_window,
                            min_periods=1
                        )
                        .mean()
                    )
        
        return df_processed
    
    def calculate_cumulative_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative stats for specified columns"""
        if not self.config.cumulative_columns:
            return df
            
        # Create a copy of the original dataframe
        df_processed = df.copy()
        
        # Make sure date is in datetime format and sort
        df_processed['GAME_DATE'] = pd.to_datetime(df_processed['GAME_DATE'])
        df_processed = df_processed.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Process each team separately
        for team in df_processed['TEAM_ID'].unique():
            # Get team's data
            mask = df_processed['TEAM_ID'] == team
            
            # Calculate cumulative stats for each column
            for col in self.config.cumulative_columns:
                if col in df_processed.columns:
                    df_processed.loc[mask, f'{col}_cumulative'] = (
                        df_processed.loc[mask, col].cumsum().shift(1)
                    )
        
        return df_processed
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics with complex formulas"""
        if not self.config.derived_metrics:
            return df
            
        df_processed = df.copy()
        
        for team in df_processed['TEAM_ID'].unique():
            mask = df_processed['TEAM_ID'] == team
            
            for metric_name, metric_info in self.config.derived_metrics.items():
                # For cumulative metrics
                if self.config.cumulative_columns:
                    # Get cumulative columns needed for calculation
                    cumulative_cols = {col: df_processed.loc[mask, f'{col}_cumulative'] 
                                    for col in metric_info.columns 
                                    if f'{col}_cumulative' in df_processed.columns}
                    
                    if len(cumulative_cols) == len(metric_info.columns):
                        # Apply the formula to cumulative values
                        df_processed.loc[mask, f'{metric_name}_cumulative'] = (
                            metric_info.formula(**cumulative_cols)
                        )
        
        return df_processed
    
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
        
        # Calculate differences only if specified
        if self.config.calculate_diffs and self.config.difference_columns:
            # Get all possible processed column patterns
            processed_patterns = []
            if self.config.rolling_columns:
                processed_patterns.append("_rolling")
            if self.config.cumulative_columns:
                processed_patterns.append("_cumulative")
            
            # Calculate differences for specified columns
            for base_col in self.config.difference_columns:
                # Check each processing pattern
                for pattern in processed_patterns:
                    col = f"{base_col}{pattern}"
                    home_col = f'{col}_home'
                    away_col = f'{col}_away'
                    if home_col in game_data.columns and away_col in game_data.columns:
                        game_data[f'{col}_diff'] = game_data[home_col] - game_data[away_col]
        return game_data
    
    def process_data(self, advanced_path: str, traditional_path: str) -> pd.DataFrame:
        """Run complete preprocessing pipeline including Elo calculations"""
        df = self.load_and_merge_data(advanced_path, traditional_path)
        df = self.add_game_context(df)
        
        # Apply the requested processing steps
        if self.config.cumulative_columns:
            df = self.calculate_cumulative_stats(df)
        if self.config.derived_metrics:
            df = self.calculate_derived_metrics(df)
            
        # Calculate Elo ratings
        game_data = self.prepare_game_data(df)
        game_data = self.calculate_elo_ratings(game_data)
        
        return game_data