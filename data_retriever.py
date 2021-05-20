# By Sammy Rao
import os.path
from custom_headers import custom_headers
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, leaguestandingsv3
from nba_api.stats.endpoints import scoreboard, teamdashboardbygeneralsplits
from datetime import datetime, timedelta

# Function to retrieve the teams
def retvieve_teams():

    # No need to download the teams data if it already exists
    if os.path.exists('data/teams/teams.csv'):

        print('\nTeams CSV file already exists!')

        # Operation successful
        return 1

    else:

        print('\nRetrieving the teams . . .')   

        # Retrieve the teams
        teams_df = pd.DataFrame(teams.get_teams())

        # Sort the teams by their full names
        teams_sorted_df = teams_df.sort_values('full_name', ascending = True)

        print('Number of teams retrieved: {}'.format(len(teams_sorted_df)))
        
        # Rename the columns
        teams_sorted_df = teams_sorted_df.rename(columns = {'id'              :   'ID', 
                                                            'full_name'       :   'FULL_NAME',
                                                            'abbreviation'    :   'ABBREVIATION',
                                                            'nickname'        :   'NICKNAME',
                                                            'city'            :   'CITY',
                                                            'state'           :   'STATE',
                                                            'year_founded'    :   'YEAR_FOUNDED'})

        print('Done!')

        # Save the data as CSV
        teams_sorted_df.to_csv('data/teams/teams.csv', index = False)

        print('Teams CSV file Saved!')

        # Operation successful
        return 1

# Function to retrieve the past games between two teams
def retrieve_past_games(team_a_id, team_a, team_b_id, team_b):

    # No need to download the past games data if it already exists
    # 'team_a_Vs_team_b' is the same set of games as 'team_b_Vs_team_a'
    if os.path.exists('data/past_games/' + team_a + '_Vs_' + team_b + '.csv') or os.path.exists('data/past_games/' + team_b + '_Vs_' + team_a + '.csv'):

        print('\nPast games CSV file already exists!')

        # Operation successful
        return 1

    else:

        print('\nRetrieving past games for ' + team_a + ' Vs ' + team_b + ' . . .')

        # Read the seasons CSV file
        seasons_CSV_df = pd.read_csv('data/seasons/seasons.csv')

        games = []        

        # Loop through the seasons and get the past games
        for index, row in seasons_CSV_df.iterrows():

            index = index # To silence 'Unused variable' warning
            season = row['SEASON']
            season_start_date = datetime.strptime(row['START_DATE'], '%m/%d/%Y').date() # Format as date
            season_start_date = datetime.strftime(season_start_date, '%Y-%m-%d') # Format as string for date problem detection later on

            print('     Retrieving ' + season + ' season games . . .')

            season_games = leaguegamefinder.LeagueGameFinder(team_id_nullable = team_a_id, 
                                                             vs_team_id_nullable = team_b_id, 
                                                             season_type_nullable = 'Regular Season', 
                                                             season_nullable = season, 
                                                             headers = custom_headers, 
                                                             timeout = 120)
            season_games = season_games.get_data_frames()[0]

            # For date problem detection later on
            game_date = pd.Series(season_games['GAME_DATE']) # Get the series
            game_date = pd.to_datetime(game_date) # Convert the series data to date 
            date_before_game = game_date - timedelta(days = 1) # Get the date before the game date
            date_before_game = date_before_game.dt.strftime('%Y-%m-%d') # Format as string

            # Insert 'SEASON', 'SEASON_START_DATE' and 'DATE_BEFORE_GAME' columns into the dataframe
            season_games.insert(1, 'SEASON', season)
            season_games.insert(2, 'SEASON_START_DATE', season_start_date)
            season_games.insert(6, 'TEAM_B_ID', team_b_id)
            season_games.insert(7, 'TEAM_B_NAME', team_b)
            season_games.insert(8, 'DATE_BEFORE_GAME', date_before_game)               

            # Append the data frame to the list
            games.append(season_games)

        games = pd.concat(games)    

        games_df = pd.DataFrame.from_dict(games)

        # Perform cleanup
        # Take only the relevant columns and all rows
        games_df = games_df.iloc[:, [1, 2, 3, 5, 6, 7, 8, 10, 11, 12]]
        # Rename the 'TEAM_ID' column to 'TEAM_A_ID'
        games_df.rename(columns = {'TEAM_ID' : 'TEAM_A_ID'}, inplace = True) 
        # Rename the 'TEAM_NAME' column to 'TEAM_A_NAME'
        games_df.rename(columns = {'TEAM_NAME' : 'TEAM_A_NAME'}, inplace = True)         
        # Replace the longer team name with the shorter nick name
        games_df['TEAM_A_NAME'] = team_a 
        # Rename the 'MATCHUP' column to 'HOME_TEAM'
        games_df.rename(columns = {'MATCHUP' : 'HOME_TEAM'}, inplace = True)  
        # Replace home and away games by 1 and 0 respectively
        games_df['HOME_TEAM'] = games_df['HOME_TEAM'].replace(['(vs)', '(@)'], [1, 0], regex = True)           
        # Rename the 'WL' column to 'WINNER'
        games_df.rename(columns = {'WL' : 'WINNER'}, inplace = True) 
        # Replace wins and losses by 1 and 0 respectively
        games_df['WINNER'] = games_df['WINNER'].replace(['W', 'L'], [1, 0])         
        # An error will occur later on while retrieving pre-game stats for the teams if the pre-game date
        # is on or before the starting date of the season as there will be no data available on that date for the teams
        # So, take only the rows without possible date problems to avoid future errors
        games_df = games_df[games_df.DATE_BEFORE_GAME > games_df.SEASON_START_DATE]

        print('Done!')

        print('Number of past games retrieved: {}'.format(len(games_df)))

        # Save the data as CSV
        games_df.to_csv('data/past_games/' + team_a + '_Vs_' + team_b + '.csv', index = False)

        print('Past games CSV file Saved!')

        # Operation successful
        return 1

# Function to launch the recursive retrieval of the stats a team had going into past games
def retrieve_past_pregame_stats(team_a_id, team_a, team_b):

    # No need to download the past pre-game stats data if it already exists
    if os.path.exists('data/past_pregame_stats/' + team_a + '_pregame_stats_Vs_' + team_b + '.csv'):

        print('\nPast pre-game stats CSV file already exists!')

        # Operation successful
        return 1    

    else:

        # Since the past games CSV file is saved under one name (Either 'the_first_team_Vs_the_second_team' 
        # or 'the_second_team_Vs_the_first_team'), the file will not be found depending on which team is 
        # passed to this function as 'team_a'
        # The try block will check this and reverse the names to read the CSV file if the second team is passed as 'team_a'
        try:
            
            # The first team is passed as 'team_a' 
            # Read the past games CSV file
            games_CSV_df = pd.read_csv('data/past_games/' + team_a + '_Vs_' + team_b + '.csv') 

        except FileNotFoundError:

            # The second team is passed as 'team_a' 
            # Read the past games CSV file
            games_CSV_df = pd.read_csv('data/past_games/' + team_b + '_Vs_' + team_a + '.csv')  

        print('\nRetrieving past pre-game stats for ' + team_a + ' Vs ' + team_b + ' . . .')

        team_stats = []

        # Loop through the past games and get the team stats a day before the game date
        for index, row in games_CSV_df.iterrows():

            index = index # To silence 'Unused variable' warning
            season = row['SEASON']
            season_start_date = row['SEASON_START_DATE']
            date_before_game = datetime.strptime(row['DATE_BEFORE_GAME'], '%Y-%m-%d').date() # Format as date

            print('     Retrieving ' + season + ' season stats from {}'.format(season_start_date) + ' to {}'.format(date_before_game) + ' . . .')

            # Append the returned result to the list
            team_stats.append(retrieve_team_stats(team_a_id, team_a, season_start_date, date_before_game, season))

        team_stats_df = pd.DataFrame.from_dict(team_stats)

        print('Done!')

        print('Number of records retrieved: {}'.format(len(team_stats_df)))

        # Save the data as CSV
        team_stats_df.to_csv('data/past_pregame_stats/' + team_a + '_pregame_stats_Vs_' + team_b + '.csv', index = False)

        print('Past pre-game stats CSV file Saved!')

        # Operation successful
        return 1

# Function to retrieve the stats for a team
def retrieve_team_stats(team_id, team_name, date_from, date_to, season):

    # The stats from the 'Traditional' section
    general_team_info = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id = team_id, 
                                                                                  per_mode_detailed = 'Per100Possessions', 
                                                                                  season_type_all_star = 'Regular Season', 
                                                                                  date_from_nullable = date_from, 
                                                                                  date_to_nullable = date_to, 
                                                                                  season = season, 
                                                                                  headers = custom_headers, 
                                                                                  timeout = 120)
    general_team_dict = general_team_info.get_normalized_dict()
    general_team_dashboard = general_team_dict['OverallTeamDashboard'][0]

    games_played = general_team_dashboard['GP']
    points = general_team_dashboard['PTS']
    field_goal_percentage = general_team_dashboard['FG_PCT']
    three_point_percentage = general_team_dashboard['FG3_PCT']
    free_throw_percentage = general_team_dashboard['FT_PCT']
    offensive_rebounds = general_team_dashboard['OREB']
    defensive_rebounds = general_team_dashboard['DREB']
    assists = general_team_dashboard['AST']
    turnovers = general_team_dashboard['TOV']
    plus_minus = general_team_dashboard['PLUS_MINUS']

    # The stats from the 'Advanced' section
    advanced_team_info = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id = team_id, 
                                                                                   measure_type_detailed_defense = 'Advanced', 
                                                                                   season_type_all_star = 'Regular Season', 
                                                                                   date_from_nullable = date_from, 
                                                                                   date_to_nullable = date_to, 
                                                                                   season = season, 
                                                                                   headers = custom_headers, 
                                                                                   timeout = 120)
    advanced_team_dict = advanced_team_info.get_normalized_dict()
    advanced_team_dashboard = advanced_team_dict['OverallTeamDashboard'][0]

    offensive_rating = advanced_team_dashboard['OFF_RATING']
    defensive_rating = advanced_team_dashboard['DEF_RATING']
    assist_percentage = advanced_team_dashboard['AST_PCT']
    assist_to_turnover_ratio = advanced_team_dashboard['AST_TO']
    assist_ratio = advanced_team_dashboard['AST_RATIO']
    effective_field_goal_percentage = advanced_team_dashboard['EFG_PCT']
    true_shooting_percentage = advanced_team_dashboard['TS_PCT']
    player_impact_estimate = advanced_team_dashboard['PIE']
    
    # Dictionary to hold the stats
    team_stats = {'GP'          :   games_played,
                  'PTS'         :   points,
                  'FG_PCT'      :   field_goal_percentage,
                  'FG3_PCT'     :   three_point_percentage,
                  'FT_PCT'      :   free_throw_percentage,   
                  'OREB'        :   offensive_rebounds,
                  'DREB'        :   defensive_rebounds,
                  'AST'         :   assists,
                  'TOV'         :   turnovers,
                  'PLUS_MINUS'  :   plus_minus,
                  'OFF_RATING'  :   offensive_rating,
                  'DEF_RATING'  :   defensive_rating,
                  'AST_PCT'     :   assist_percentage,
                  'AST_TO'      :   assist_to_turnover_ratio,
                  'AST_RATIO'   :   assist_ratio,
                  'EFG_PCT'     :   effective_field_goal_percentage,
                  'TS_PCT'      :   true_shooting_percentage,
                  'PIE'         :   player_impact_estimate}

    return(team_stats)   

# Function to retrieve the games schedule
def retrieve_games_schedule(date):

    # Read the teams CSV file
    teams_CSV_df = pd.read_csv('data/teams/teams.csv')

    print('\nRetrieving the games scheduled for ' + date + ' . . .')

    # Get the games scheduled for the specified date
    games_schedule = scoreboard.Scoreboard(game_date = date, 
                                           headers = custom_headers, 
                                           timeout = 120)
    games_schedule_dict = games_schedule.get_normalized_dict()
    games = games_schedule_dict['GameHeader']

    games_df = pd.DataFrame(games) 

    # Map the IDs of the teams with thier nicknames
    home_teams = games_df.HOME_TEAM_ID.map(teams_CSV_df.set_index('ID')['NICKNAME'])
    visitor_teams = games_df.VISITOR_TEAM_ID.map(teams_CSV_df.set_index('ID')['NICKNAME'])

    # Insert the nickname columns next to the ID columns
    games_df.insert(loc = 7, column = 'HOME_TEAM_NICKNAME', value = home_teams)
    games_df.insert(loc = 9, column = 'VISITOR_TEAM_NICKNAME', value = visitor_teams)

    # Take only the relevant columns and all rows
    games_df = games_df.iloc[:, [2, 6, 7, 8, 9]]

    print('Done!')

    print('Number of games retrieved: ' + str(len(games_df)))

    return games_df

# Function to retrieve the league standings of the teams
def retrieve_league_standings():

    print('\nRetrieving league standings . . .')

    league_standings = leaguestandingsv3.LeagueStandingsV3(season = '2020-21',
                                                           season_type = 'Regular Season',
                                                           headers = custom_headers, 
                                                           timeout = 120)  
    league_standings = league_standings.get_normalized_dict()
    league_standings_dict = league_standings['Standings']

    league_standings_df = pd.DataFrame(league_standings_dict)

    # Perform cleanup
    # Take only the relevant columns and all rows
    league_standings_df = league_standings_df.iloc[:, [4, 8, 15, 17, 18, 19, 37]]
    # Rename the 'TeamName' column to 'TEAM_NAME'
    league_standings_df.rename(columns = {'TeamName' : 'TEAM_NAME'}, inplace = True) 
    # Rename the 'PlayoffRank' column to 'PLAYOFF_RANK'
    league_standings_df.rename(columns = {'PlayoffRank' : 'PLAYOFF_RANK'}, inplace = True)  
    # Rename the 'WinPCT' column to 'WIN_PERCENTAGE'
    league_standings_df.rename(columns = {'WinPCT' : 'WIN_PERCENTAGE'}, inplace = True)
    # Rename the 'Record' column to 'LEAGUE_RECORD'
    league_standings_df.rename(columns = {'Record' : 'LEAGUE_RECORD'}, inplace = True)
    # Rename the 'HOME' column to 'HOME_RECORD'
    league_standings_df.rename(columns = {'HOME' : 'HOME_RECORD'}, inplace = True)
    # Rename the 'ROAD' column to 'ROAD_RECORD'
    league_standings_df.rename(columns = {'ROAD' : 'ROAD_RECORD'}, inplace = True)  
    # Rename the 'strCurrentStreak' column to 'CURRENT_STREAK'
    league_standings_df.rename(columns = {'strCurrentStreak' : 'CURRENT_STREAK'}, inplace = True)          

    print('Done!\n')

    return league_standings_df 
                   