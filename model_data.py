#%%
import os
import pandas as pd

# build model data for one gw in three steps
# step 1: extract relevant data from relevant gw.csv file
# step 2: extract opponent difficulty data 
# step 3: extract lagged data from prior gw.csv files
 
# step 1 
season = '2023-24'
gw = 1

season_path = f'data/{season}/'
gw_path = season_path + f'gws/'
gw_filepath = gw_path + f'gw{gw}.csv'

fixs_df = pd.read_csv(f'{season_path}fixtures.csv')
fixs_df = fixs_df[fixs_df['event'] == gw]

#step 2

h_diff = fixs_df.loc[:, ['team_h', 'team_h_difficulty']].rename(columns = {'team_h':'team', 'team_h_difficulty':'difficulty'})
a_diff = fixs_df.loc[:, ['team_a', 'team_a_difficulty']].rename(columns = {'team_a':'team', 'team_a_difficulty':'difficulty'})
team_diff = pd.concat([h_diff, a_diff]).sort_values('team')
team_diff = dict(team_diff.values)


params = ['name', 'position', 'team', 'total_points', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes', 'was_home', 'opponent_team']
gw_df = pd.read_csv(gw_filepath)
gw_df = gw_df.loc[:, params]
gw_df['opponent_team_difficulty']  = gw_df['opponent_team'].map(team_diff)

#step 3
r = range(gw - 1, gw - 1 - 18, -1)

def get_prev_season(season):

    yr1 = int(season.split('-')[0])
    yr2 = int(season.split('-')[1])
    
    return f'{yr1 -1 }-{yr2 -1}'                           

prev_season_gw_path = f'data/{get_prev_season(season)}/gws/'

lags = []

for i in r:
    if i > 0:
        lags.append(gw_filepath + f'gw{i}.csv')
    else:
        lags.append(prev_season_gw_path + f'gw{i + 36}.csv')

lagged_params = ['name', 'position', 'team', 'total_points', 'xP', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes']

lagged_df = pd.DataFrame()

for lag in lags:
    df = pd.read_csv(lag)
    lagged_df = pd.concat([lagged_df, df], axis = 1)
             
## calculate trailing data for each player

players = set(gw_df['name'])

#test lagged data for player[0] ('Marcus Tavernier')

lagged_player_data = list(players)[0].loc[:, 'total_points':'minutes'].sum()
laged_player_data_90 = lagged_player_data.iloc[:-1].div(lagged_player_data.iloc[-1])

# %%
