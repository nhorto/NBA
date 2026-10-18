# to us the NBA API you will need to install it: pip install nba_api

from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv2, boxscoreadvancedv2, playercareerstats, teamgamelogs, playergamelogs
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import players, teams
import numpy as np
import pandas as pd
import time

def retry(func):
    '''
    This is a decorator that will take a function and allow it to be retried up up to three times if it raises an exception. If the number of retries is exhausted, it raises
    an exception indicating that all retry attempts have failed.
    
    Parameters:
    - func (callable): This is the function to be retried. This will be retried up to three times. 
    
    Returns:
    - A wrapper function that includes the retry logic. It will return the result of the func parameter if it succeds within the retry attempts. 
    
    Raises:
    - Execption: An exception is raised indicating that all retry attempts have failed
      or a persistent error has occured.
    '''
    def retry_wrapper(*args, **kwargs):
        retries = 3 # can change the number of retries if needed
        attempts = 0 # counter
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except Exception as e: 
                raise Exception(f"persistent error encountered: {e}")
            except Exception as e:
                print(f"Attempt {attempts + 1} fialed with error: {e}")
                time.sleep(10) 
                attempts +=1
        raise Exception("All retry attempts failed.")
    return retry_wrapper

class NBA_Data:
    def __init__(self):
        '''
        This is a class for pulling NBA statistics from NBA.com's API. 

        This class provides methods to retrieve various types of NBA data such as player career stats,
        player box scores, team box scores, and comprehensive box scores for all players or teams within a season.
        Each method automatically handles retries upon failure, ensuring data retrieval is robust against transient issues.

        Methods are decorated with a `retry` mechanism to attempt a specified number of retries upon encountering
        exceptions, enhancing reliability in the face of temporary network or API issues.
        The methods provided will pull various types of data that can be found on the NBA.com statistics page such as 
        player career stats, player box scores, team box scores, and comprehensive box scores for all players or teams within a season.
        Each method will handle rerued if it fails to pull the data. 

        Methods:
        - get_player_career_stats(player_name): Gets career statistics for a given player.
        - get_player_boxscores(player_name, season): Gets box scores for a specific player and season.
        - get_team_boxscores(team_name, season): Gets box scores for a specific team and season.
        - get_all_players_boxscores(season, advanced_boxscore=False): Gets box scores for all players in a given season,
          with a option for getting advanced box scores.
        - get_all_teams_boxscore(season, advanced_boxscore=False): Gets box scores for all teams in a specified season,
          with a option for getting advanced statistics.

        Usage:
        - nba_data = NBA_Data()
        - player_career_stats = nba_data.get_player_career_stats('Stephen Curry')
        - team_boxscores = nba_data.get_team_boxscores('Golden State Warriors', 2023)

        Dependencies:
        - For this class to run you will need to install the following packages:
            -Pandas
            -Time
            -NBA API 
        - For the NBA API you will need the following endpoints:
            -from nba_api.stats.endpoints import, leaguegamelog, boxscoretraditionalv2, boxscoreadvancedv2, playercareerstats, teamgamelogs, playergamelogs
            -from nba_api.stats.static import players, teams

        There are no parameters that are required for initializing this class
        '''
    pass

    @retry
    def get_player_career_stats(self, player_name):
        '''
        This function gets a players career statistics. 
        
        Parameters:
        - player_name (str): The full name of the NBA player. Example: 'Stephen Curry'
        
        Returns:
        -DataFrame: A pandas data frame that contains a players career statistics.
        '''
        try:
            # Get players ID
            nba_players =players.get_players()
            player = [player for player in nba_players if player['full_name'] == player_name][0]
            playerID = player['id']

            # Get the stats
            carreer_stats = playercareerstats.PlayerCareerStats(player_id=playerID).get_data_frames()[0]
            return carreer_stats
                              
        except IndexError:
            return f"No data fround for player: {player_name}"
        except Exception as e:
            return f"An error occurred: {e}"
    
    @retry
    def get_player_boxscores(self, player_name, season):
        '''
        This function gets a players box score from a season. 
        
        Parameters:
        - player_name (str): The full name of the NBA player. Example: 'Stephen Curry'
        - season (int): The season to get the box score for. Example: 2023.
        
        Returns:
        -DataFrame: A pandas data frame that contains a players box score for a season.
        '''
        season = str(season) + "-" + str(season+1)[-2:] # Convert year to season format ie. 2020 -> 2020-21
        try:
            # Get players ID
            nba_players =players.get_players()
            player = [player for player in nba_players if player['full_name'] == player_name][0]
            playerID = player['id']

            # Get the stats
            player_boxscore = playergamelogs.PlayerGameLogs(player_id_nullable=playerID, season_nullable=season).get_data_frames()[0]
            return player_boxscore

        except IndexError:
            return f"No data fround for player: {player_name}"
        except Exception as e:
            return f"An error occurred: {e}"
                              
        

    @retry
    def get_team_boxscores(self, team_name, season): 
        '''
        This function gets a players box score from a season. 
        
        Parameters:
        - player_name (str): The full name of the NBA player. Example: 'Stephen Curry'
        - season (int): The season to get the box score for. Example: 2023.
        
        Returns:
        -DataFrame: A pandas data frame that contains a players box score for a season.
        '''
        try:
            season = str(season) + "-" + str(season+1)[-2:] # Convert year to season format ie. 2020 -> 2020-21

            # Get team ID
            nba_teams = teams.get_teams()
            team = [team for team in nba_teams if team["full_name"] == team_name][0]
            teamID = team['id']

            # Get the stats
            teamGameStats = teamgamelogs.TeamGameLogs(team_id_nullable=teamID, season_nullable=season).get_data_frames()[0]
            return teamGameStats
        
        except IndexError:
            return f"No data found for team: {team_name}"
        except Exception as e:
            return f"An error occurred: {e}"
        
    @retry
    def get_all_players_boxscores(self, season, advanced_boxscore=False):
        '''
        Retrieves box score statistics for all games played by every player in a given NBA season.
        
        Parameters:
        - season (int): The season to get the box scores for. Example: 2023.
        - advanced_boxscore (bool, optional): If this is set to True, the advanced boxscore statistics will be pulled in stead of the traditional boxscore statistics

        Returns:
        - pandas.DataFrame: A DataFrame with box score statistics for each game for every player in a specified season. 
        '''
        try:
            season_format = str(season) + "-" + str(season+1)[-2:]
            season_games = leaguegamelog.LeagueGameLog(season=season_format).get_data_frames()[0]
            game_id_list = season_games['GAME_ID'].unique()

            all_games_box_score = []

            for game_id in game_id_list:
                if advanced_boxscore:
                    adv_box_score = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0] # 0 returns the player data from the JSON
                    all_games_box_score.append(adv_box_score)
                    
                else:
                    box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0] # 0 returns the player data from the JSON
                    all_games_box_score.append(box_score)
                time.sleep(3)
                                               
            boxscore_combined = pd.concat(all_games_box_score, ignore_index=True)
            return boxscore_combined
                                            
        except IndexError:
            return f"No data found for season: {season}"
        except Exception as e:
            return f"An error occurred: {e}"


    @retry
    def get_all_teams_boxscore(self, season, advanced_boxscore=False):
        '''
        Retrieves box score statistics for all games played by every team in a given NBA season.
        
        Parameters:
        - season (int): The season to get the box scores for. Example: 2023.
        - advanced_boxscore (bool, optional): If this is set to True, the advanced boxscore statistics will be pulled in stead of the traditional boxscore statistics

        Returns:
        - pandas.DataFrame: A DataFrame with box score statistics for each game for every team in a specified season. 
        '''
        try:
            season_format = str(season) + "-" + str(season+1)[-2:]
            season_games = leaguegamelog.LeagueGameLog(season=season_format).get_data_frames()[0]
            game_id_list = season_games['GAME_ID'].unique()

            if advanced_boxscore:
                all_games_box_score = []
                for game_id in game_id_list:
                    box_score = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)#.get_data_frames()[1] # 1 returns the team data from the JSON
                    team_stats = box_score.team_stats.get_data_frame()
                    all_games_box_score.append(team_stats)
                    time.sleep(5)
                advanced_boxscore = pd.concat(all_games_box_score, ignore_index=True)
                return advanced_boxscore
     
            else:
                return season_games
    
        except IndexError:
            return f"No data found for season: {season}"
        except Exception as e:
            return f"An error occurred: {e}"