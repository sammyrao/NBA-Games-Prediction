# By Sammy Rao
import os.path
import pandas as pd
import data_retriever as dr

# Function to launch the data collection
def start_retrieval(team_a, team_b):

    # Retrieve the teams
    teams_retrieval_result = dr.retvieve_teams()

    # If successful, continue
    if teams_retrieval_result == 1:

        # Read the teams CSV file
        teams_CSV_df = pd.read_csv('data/teams/teams.csv')

        # Map the nicknames of the teams with their IDs
        team_a_id = teams_CSV_df.loc[teams_CSV_df['NICKNAME'] == team_a, 'ID'].values[0]
        team_b_id = teams_CSV_df.loc[teams_CSV_df['NICKNAME'] == team_b, 'ID'].values[0]

        # Retrieve the past games between the two teams
        get_past_games(team_a_id, team_a, team_b_id, team_b)

# Function to launch the past games and stats data collection
def get_past_games(team_a_id, team_a, team_b_id, team_b):

    # Retrieve the past games between the two teams
    games_retrieval_result = dr.retrieve_past_games(team_a_id, team_a, team_b_id, team_b)

    # If successful, continue
    if games_retrieval_result == 1:

        # Retrieve the stats the first team had going into the past games
        dr.retrieve_past_pregame_stats(team_a_id, team_a, team_b)

        # Retrieve the stats the second team had going into the past games
        dr.retrieve_past_pregame_stats(team_b_id, team_b, team_a)

        # Combine the past games and stats data
        combine_stats_and_games(team_a, team_b)

# Function to combine the collected games and stats data
def combine_stats_and_games(team_a, team_b):

    # No need to to combine the collected games and stats data if it already exists
    # 'team_a_Vs_team_b' is the same set of games as 'team_b_Vs_team_a'
    if os.path.exists('data/combined_games_and_stats/combined_' + team_a + '_Vs_' + team_b + '.csv') or os.path.exists('data/combined_games_and_stats/combined_' + team_b + '_Vs_' + team_a + '.csv'):

        print('\nCombined data CSV file already exists!')

    else:    

        print('\nReading the CSV files . . .')   

        # Read the past games CSV file for the teams
        past_games_CSV_df = pd.read_csv('data/past_games/' + team_a + '_Vs_' + team_b + '.csv', usecols = ['HOME_TEAM', 'WINNER'])

        # Read the past pre-game stats CSV file for 'team_a'
        team_a_stats_CSV_df = pd.read_csv('data/past_pregame_stats/' + team_a + '_pregame_stats_Vs_' + team_b + '.csv')

        # Read the past pre-game stats CSV file for 'team_b'
        team_b_stats_CSV_df = pd.read_csv('data/past_pregame_stats/' + team_b + '_pregame_stats_Vs_' + team_a + '.csv')                

        print('Combining the CSV files . . .')

        # Get the stats differentials onto a new data frame
        combined_data_df = pd.DataFrame(team_a_stats_CSV_df.values - team_b_stats_CSV_df.values)
        # Give names to the columns
        combined_data_df.columns = team_a_stats_CSV_df.columns
        # Add a prefix to the columns
        combined_data_df = combined_data_df.add_suffix('_DIFF')

        # Join the two data frames to create the combined data
        combined_data_df = combined_data_df.join(past_games_CSV_df)

        print('Done!')

        # Save the data as CSV
        combined_data_df.to_csv('data/combined_games_and_stats/combined_' + team_a + '_Vs_' + team_b + '.csv', index = False)

        print('Combined data CSV file Saved!')              

if __name__ == '__main__':

    # Read the teams CSV file
    teams_CSV_df = pd.read_csv('data/teams/teams.csv', usecols = ['NICKNAME'])

    # Get two random teams
    random_teams_df = teams_CSV_df.sample(n = 2, replace = False)

    # Extract the teams' nicknames as strings
    team_a = str(random_teams_df.head(1).values)[3:-3]
    team_b = str(random_teams_df.tail(1).values)[3:-3]
    
    # Start the chain of operations to gather the data required for training
    start_retrieval(team_a, team_b)