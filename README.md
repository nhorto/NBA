# NBA Game Prediction

A machine learning project that predicts NBA game outcomes using historical game data, advanced statistics, and ELO ratings. This project demonstrates end-to-end data science workflow including API integration, feature engineering, and predictive modeling.

## Project Overview

This project pulls historical NBA data from the official NBA API, engineers features including team performance metrics and ELO ratings, and uses machine learning models to predict game outcomes. The system achieves approximately 69% accuracy in predicting game winners, outperforming baseline models.

---

## NBA_API.py

A custom Python class for retrieving NBA statistics from NBA.com's official API. This class handles data extraction for players and teams across multiple seasons.

### Key Features:
- **Automatic retry logic**: All API calls include a retry decorator to handle transient network issues
- **Comprehensive data retrieval**: Pull career stats, player box scores, team box scores, and season-wide data
- **Rate limiting**: Built-in delays to respect API rate limits

### Main Methods:
- `get_player_career_stats(player_name)`: Retrieves complete career statistics for a specified player
- `get_player_boxscores(player_name, season)`: Gets game-by-game box scores for a player in a specific season
- `get_team_boxscores(team_name, season)`: Retrieves team box scores for all games in a season
- `get_all_players_boxscores(season, advanced_boxscore=False)`: Pulls box score data for all players across an entire season, with option for advanced statistics
- `get_all_teams_boxscore(season, advanced_boxscore=False)`: Gets box score data for all teams in a season, with option for advanced statistics

### Dependencies:
- `nba_api` - Official NBA statistics API wrapper
- `pandas` - Data manipulation
- `numpy` - Numerical operations

---

## preprocessor.py

The core data processing pipeline that transforms raw NBA box score data into machine learning-ready features. This module handles feature engineering, ELO rating calculations, and player availability tracking.

### What is ELO Rating?

ELO is a rating system originally developed for chess that quantifies the relative skill levels of competitors. In this project, ELO ratings are used to track team strength over time:

- **Initial Rating**: Each team starts with an ELO of 1500
- **K-Factor**: Controls how much ratings change after each game (set to 20)
- **Home Advantage**: Home teams receive a +100 ELO boost when calculating win probability
- **Rating Updates**: After each game, the winner's ELO increases and the loser's decreases based on the expected outcome
- **Probability Calculation**: The system calculates win probability using the ELO difference between teams

ELO ratings are particularly valuable for NBA predictions because they:
- Adapt dynamically as team performance changes
- Account for strength of schedule implicitly
- Provide a single metric that captures team quality
- Include home court advantage in predictions

### Configuration Classes

#### ProcessingConfig
Defines all parameters for the data processing pipeline:
- **all_processable_columns**: Currently processes all available statistical columns (temporary approach while optimizing feature selection)
- **base_columns**: Core statistics columns from box scores
- **rolling_window**: Number of games to include in rolling averages (default: 5)
- **derived_metrics**: Dictionary of custom metrics calculated from base statistics
- **ELO parameters**: Initial ELO, K-factor, home advantage, and ELO width

> **Note on Column Processing**: The current implementation uses `all_processable_columns` to process all available statistics while the optimal feature set is being determined. Additionally, advanced metrics are derived from base statistics because the NBA API does not provide historical advanced box scores.

### DataProcessor Class

The main preprocessing engine with the following methods:

#### Data Loading & Merging
- **`load_and_merge_data(advanced_path, traditional_path, player_boxscores)`**: Loads and merges advanced and traditional box score CSV files, removing redundant columns

#### Feature Engineering
- **`add_context(df)`**:
  - Creates home/away indicators from matchup strings
  - Converts win/loss to binary format
  - Calculates cumulative statistics for all processable columns
  - Calculates rolling averages (default 5-game window)
  - Computes derived metrics from cumulative and rolling stats
  - Handles infinite values and validates data quality

- **`prepare_game_data(df)`**:
  - Splits data into separate home and away team records
  - Renames columns with `_home` and `_away` suffixes
  - Merges home and away data to create game-level dataset with both teams' statistics

#### ELO Rating System
- **`calculate_elo_probability(home_elo, away_elo)`**:
  - Calculates the expected probability of the home team winning
  - Incorporates home court advantage into probability calculation

- **`update_elos(home_team, away_team, home_won, margin)`**:
  - Updates ELO ratings for both teams after a game
  - Adjusts rating changes based on point margin
  - Returns new ELO ratings for both teams

- **`calculate_elo_ratings(df)`**:
  - Processes all games chronologically to calculate ELO ratings
  - Creates pre-game and post-game ELO columns
  - Generates win probability for each game
  - Cleans up unnecessary columns

#### Player Availability Analysis
- **`normalize(series)`**:
  - Normalizes a statistical series to 0-1 range for importance scoring

- **`get_player_info(player_boxscores, cutoff_date, game_id, team_id, home_away)`**:
  - Calculates player importance scores based on minutes, plus/minus, points, assists, and rebounds
  - Identifies rotation players (minimum 15 minutes per game)
  - Detects missing players by comparing expected rotation to actual game participants
  - Calculates total impact of missing players
  - Flags when star players (top 90% importance) are absent

- **`process_player_availability(df)`**:
  - Processes player availability data for all games
  - Adds features for number of missing players, missing impact score, and missing star indicators
  - Creates separate metrics for home and away teams

#### Advanced Metrics
- **`calculate_difference_metrics(df)`**:
  - Calculates the difference between home and away team statistics
  - Focuses on key metrics like net rating, offensive/defensive rating, pace, efficiency metrics, and rebounding percentages
  - Reduces dimensionality by creating single differential features

#### Pipeline Orchestration
- **`process(advanced_path, traditional_path, player_boxscores)`**:
  - Main processing function that orchestrates the entire pipeline
  - Executes all steps in correct order: load → context → game preparation → ELO → player availability
  - Returns analysis-ready dataset with all engineered features

---

## predict_games_test.ipynb

A Jupyter notebook that tests various machine learning models for predicting NBA game outcomes.

### Workflow:

1. **Data Preparation**:
   - Loads preprocessed data from the DataProcessor pipeline
   - Filters out nonsensical cumulative columns (e.g., cumulative percentages)
   - Removes rate statistics that don't make sense when summed
   - Performs train/test split (80/20)

2. **Dimensionality Reduction**:
   - Applies PCA (Principal Component Analysis) to reduce features
   - Retains 27 components explaining 95% of variance
   - Visualizes feature loadings to understand component composition

3. **Baseline Models**:
   - **Always predict home win**: 57.2% accuracy
   - **Predict higher ELO wins**: 62.6% accuracy

4. **Machine Learning Models Tested**:
   - **Logistic Regression**: 66% accuracy, 0.72 ROC AUC
   - **Support Vector Machine (SVM)**: 66% accuracy, 0.70 ROC AUC
   - **XGBoost**: 64% accuracy, 0.69 ROC AUC
   - **Naive Bayes**: 64% accuracy, 0.70 ROC AUC

5. **Ensemble Methods**:
   - Tests stacking classifiers with various combinations of base models
   - Experiments with different meta-classifiers (Logistic Regression, XGBoost, SVM)
   - Identifies optimal model combinations

### Key Results:
- Logistic Regression achieves the best single-model performance (66% accuracy)
- All models significantly outperform the baseline approaches
- ELO ratings prove to be strong predictive features
- Rolling averages and cumulative statistics provide valuable temporal context

---

## Project Structure

```
NBA Data/
├── NBA_API.py                  # API wrapper for data collection
├── preprocessor.py             # Feature engineering pipeline
├── Preprocess.py              # Alternative preprocessing implementation
├── predict_games_test.ipynb   # Model training and evaluation
├── Data/                      # Raw and processed datasets
└── Notebooks/                 # Additional analysis notebooks
```

---

## Future Improvements

- Optimize feature selection by testing subsets of `all_processable_columns`
- Incorporate additional features like rest days, back-to-back games, and travel distance
- Experiment with deep learning models (LSTM for sequence prediction)
- Add real-time prediction capabilities for upcoming games
- Develop a web interface for model predictions

---

## Technologies Used

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Data Visualization**: matplotlib, seaborn
- **API Integration**: nba_api
