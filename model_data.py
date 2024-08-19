#%%
import os
import pandas as pd

# build model data for one gw in three steps
# step 1: extract relevant data from relevant gw.csv file
# step 2: extract opponent difficulty data from fixtures.csv
# step 3: extract lagged data from prior gw.csv files
# turn the script into a function, loop the function to run for each week needed to build the dataset for
 
# step 1 
season = '2023-24'
gw = 1

season_path = f'data/{season}/'
gw_path = season_path + f'gws/'
gw_filepath = gw_path + f'gw{gw}.csv'
params = ['name', 'position', 'team', 'total_points', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes', 'was_home', 'opponent_team']

gw_df = pd.read_csv(gw_filepath)
gw_df = gw_df.loc[:, params]

#step 2
fixs_df = pd.read_csv(f'{season_path}fixtures.csv')
gw_fixs_df = fixs_df[fixs_df['event'] == gw]

h_diff = gw_fixs_df.loc[:, ['team_h', 'team_h_difficulty']].rename(columns = {'team_h':'team', 'team_h_difficulty':'difficulty'})
a_diff = gw_fixs_df.loc[:, ['team_a', 'team_a_difficulty']].rename(columns = {'team_a':'team', 'team_a_difficulty':'difficulty'})
team_diff = pd.concat([h_diff, a_diff]).sort_values('team')
team_diff = dict(team_diff.values)
gw_df['opponent_team_difficulty']  = gw_df['opponent_team'].map(team_diff)
#%%
#step 3
r = range(gw - 1, gw - 1 - 18, -1)

def get_prev_season(season):

    yr1 = int(season.split('-')[0])
    yr2 = int(season.split('-')[1])
    
    return f'{yr1 -1 }-{yr2 -1}'                           

prev_season_gw_path = f'data/{get_prev_season(season)}/gws/'

lagged_gw_filepaths = []

for i in r:
    if i > 0:
        lagged_gw_filepaths.append(gw_filepath + f'gw{i}.csv')
    else:
        lagged_gw_filepaths.append(prev_season_gw_path + f'gw{i + 38}.csv')
 
lagged_data_df = pd.DataFrame()

for filepath in lagged_gw_filepaths:
    df = pd.read_csv(filepath)
    lagged_data_df = pd.concat([lagged_data_df, df], axis = 0)

lag_params = ['name', 'position', 'team', 'total_points', 'xP', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes']
lagged_data_df = lagged_data_df.loc[:, lag_params]             

players = list(set(gw_df['name']))

def get_model_gw_df(lagged_data_df, gw_df, players):
    df = pd.DataFrame()
    for player in players:
        lagged_player_data = lagged_data_df[lagged_data_df['name']==player].loc[:, 'xP':'minutes'].sum()
        lagged_player_data_90 = lagged_player_data.iloc[:-1].div(lagged_player_data.iloc[-1])
        lagged_player_data_90['mins_p_match'] = lagged_player_data['minutes'] / 18
        lagged_player_data_90 = lagged_player_data_90.to_frame().T
        gw_player_data = gw_df[gw_df['name'] == player]
        lagged_player_data_90.index = gw_player_data.index
        player_model_data = pd.concat([gw_player_data, lagged_player_data_90], axis = 1)
        df = pd.concat([df, player_model_data], axis = 0)
    return df
    

    

# %%
