#%%
import os
import pandas as pd
import numpy as np

# build model data for one gw in three steps
# step 1: extract relevant data from relevant gw.csv file
# step 2: extract opponent difficulty data from fixtures.csv
# step 3: extract lagged data from prior gw.csv files
# turn the script into a function, loop the function to run for each week needed to build the dataset for

pd.set_option('future.no_silent_downcasting', True) #to prevent FutureWarning: Downcasting behaviour in 'replace' is deprecated ...

def get_prev_season(season):
    """
    Smmary:
        Generate prior season for given season

    Args:
        season (str): format YYYY-YY

    Returns:
        string
    """
    yr1 = int(season.split('-')[0])
    yr2 = int(season.split('-')[1])
    
    return f'{yr1 -1 }-{yr2 -1}' 

def get_prev_gw(season, gw):
    """
    Return tuple of (season, gw) for previous gw

    """
    yr1 = int(season.split('-')[0])
    yr2 = int(season.split('-')[1])
    
    if gw - 1 < 1:
        gw = gw - 1 + 38
        season = f'{yr1 - 1}-{yr2 -1}'
    else:
        gw = gw - 1
        season = season
    return (season, gw)


def get_gw_data(season, gw):

    season_path = f'data/{season}/'
    gw_path = season_path + f'gws/'
    gw_filepath = gw_path + f'gw{gw}.csv'
    
    #step 1: get gameweek df
    gw_df = pd.read_csv(gw_filepath)
    params = ['name', 'position', 'team', 'value', 'total_points', 'xP', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes', 'was_home', 'opponent_team']
    missing_params = [param for param in params if param not in gw_df.columns]
    for col in missing_params:
        gw_df[col] = np.nan
    gw_df = gw_df.loc[:, params]

    #step 2: add opponent difficulty from fixtures.csv to gameweek df
    fixs_df = pd.read_csv(f'{season_path}fixtures.csv')
    gw_fixs_df = fixs_df[fixs_df['event'] == gw]

    h_diff = gw_fixs_df.loc[:, ['team_h', 'team_h_difficulty']].rename(columns = {'team_h':'team', 'team_h_difficulty':'difficulty'})
    a_diff = gw_fixs_df.loc[:, ['team_a', 'team_a_difficulty']].rename(columns = {'team_a':'team', 'team_a_difficulty':'difficulty'})
    team_diff = pd.concat([h_diff, a_diff]).sort_values('team')
    team_diff = dict(team_diff.values)
    gw_df['opponent_team_difficulty']  = gw_df['opponent_team'].map(team_diff)
    
    return gw_df.reindex(columns = params)

def get_trailing_data(season, gw, lags = 19):
    """_summary_

    Args:
        season (_type_): _description_
        gw (_type_): _description_
        lags (int, optional): _description_. Defaults to 19.

    Returns:
        _type_: _description_
    """

    season_path = f'data/{season}/'
    gw_path = season_path + f'gws/'
    gw_filepath = gw_path + f'gw{gw}.csv'
    
    r = range(gw - 1, gw - 1 - lags, -1)

    trailing_gw_filepaths = []
    for i in r:
        if i > 0:
            trailing_gw_filepaths.append(gw_path + f'gw{i}.csv')
        else:
            prev_season_gw_path = f'data/{get_prev_season(season)}/gws/'
            trailing_gw_filepaths.append(prev_season_gw_path + f'gw{i + 38}.csv')
 
    
    df_list = [pd.read_csv(filepath) for filepath in trailing_gw_filepaths]

    # make sure the first df in the list isn't empty
    i = 0 # counter for index location of first non-empty df
    for df in df_list:
        if df.empty == False:
            lagged_data_df = df
            break
        else:
            i += 1
    
    for df in df_list[i:]:
        if df.empty or df.isna().all().all():
            continue
        lagged_data_df = pd.concat([lagged_data_df, df], axis=0)

    lag_params = ['name', 'position', 'team', 'value', 'total_points', 'xP', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes']
    
    missing_params = [param for param in lag_params if param not in lagged_data_df.columns]
    for col in missing_params:
        lagged_data_df[col] = np.nan
    
    return lagged_data_df.loc[:, lag_params].reindex(columns = lag_params)             
    

def combine_gw_trailing(gw_data, trailing_data):
    """_summary_

    Args:
        gw_data (_type_): _description_
        trailing_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    players = list(set(gw_data['name']))
 
    for player in players:
        
        gw_player_data = gw_data[gw_data['name'] == player]
        player_rows = gw_player_data.shape[0]
        
        if player_rows == 0 or player_rows > 2:
            continue

        elif player_rows == 2: # some gw have teams play twice, recalculate lag data accordingly
#            print(player, player_rows)
            lagged_data_to_input = gw_player_data.iloc[0, 4:16].to_frame().T
            first_gw_match_lagged_player_data = trailing_data[trailing_data['name']==player].loc[:, 'total_points':'minutes']
            second_gw_match_lagged_player_data = pd.concat([lagged_data_to_input, first_gw_match_lagged_player_data.iloc[:-1]])
            aggregated_first = first_gw_match_lagged_player_data.sum().to_frame().T
            aggregated_second = second_gw_match_lagged_player_data.sum().to_frame().T
            agg_lagged_player_data = pd.concat([aggregated_first, aggregated_second], axis=0)
            agg_lagged_player_data['minutes'] = agg_lagged_player_data['minutes'].replace([0], np.nan)
            agg_lagged_player_data_90 = agg_lagged_player_data.iloc[:, :-1].div(agg_lagged_player_data.iloc[:, -1], axis = 0)
            
        else:
#            print(player, player_rows)
            agg_lagged_player_data = trailing_data[trailing_data['name']==player].loc[:, 'total_points':'minutes'].sum().to_frame().T
            agg_lagged_player_data['minutes'] = agg_lagged_player_data['minutes'].replace([0], np.nan)
            agg_lagged_player_data_90 = agg_lagged_player_data.iloc[:, :-1].div(agg_lagged_player_data.iloc[:, -1], axis=0)

        agg_lagged_player_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        n = trailing_data[trailing_data['name']==player].shape[0]
        if n > 0:
            agg_lagged_player_data_90['minutes'] = agg_lagged_player_data['minutes'] / n
        else:
            agg_lagged_player_data_90['minutes'] = 0

        agg_lagged_player_data_90.columns = 'tr_' + agg_lagged_player_data_90.columns
        agg_lagged_player_data_90.index = gw_player_data.index

        if agg_lagged_player_data_90.isna().all().all(): #concatenating empty or #NA dfs will soon be discontinued, this line ensures the code will continue to run then
            continue

        player_model_data = pd.concat([gw_player_data, agg_lagged_player_data_90], axis = 1)

        try:
            combined_data = pd.concat([player_model_data, combined_data], axis = 0)
        except NameError:
            combined_data = player_model_data


    return combined_data


#%%
season = '2019-20'
seasons_to_run = 3

for i in range(seasons_to_run):

    if season == '2019-20':
        gw = 47
        runs = 47
    else:
        gw = 38
        runs = 38
    lags = 19

    path = f'data/{season}/'
    
    for i in range(runs):
        print(f'compiling data for gameweek {gw}, season {season}')
        gw_df = get_gw_data(season, gw)
        if gw_df.empty:
            continue
        trailing_df = get_trailing_data(season, gw, lags)
        comb = combine_gw_trailing(gw_df, trailing_df)
        
        if i == 0:
            model_data = comb
        else:
            model_data = pd.concat([comb, model_data], axis=0)
        
        season, gw = get_prev_gw(season, gw)
    
    model_data.to_csv(f'{path}model_data.csv')

print('Data gathering complete')


# %%
# debug gw and lagged data for cov season 2020-21
# this will help fix the lagged data function to skip empty lagged data, and finish when #lags are complete
# lst = [0, 1, 'x', 'y', 2, 3]

# new = []

# for e in lst:
#     if type(e) == int:
#         new.append(e)
#         if len(new) == 4:
#             break


season = '2019-20'
gw = 38
lags=19

gw_data = get_gw_data(season, gw)
trailing_data = get_trailing_data(season, gw, lags)
#comb = combine_gw_trailing(gw_data, trailing_data)

players = list(set(gw_data['name']))

for player in players:
    
    gw_player_data = gw_data[gw_data['name'] == player]
    player_rows = gw_player_data.shape[0]
    
    if player_rows == 0 or player_rows > 2:
        continue

    elif player_rows == 2: # some gw have teams play twice, recalculate lag data accordingly
        print(player, player_rows)
        lagged_data_to_input = gw_player_data.iloc[0, 4:16].to_frame().T
        first_gw_match_lagged_player_data = trailing_data[trailing_data['name']==player].loc[:, 'total_points':'minutes']
        second_gw_match_lagged_player_data = pd.concat([lagged_data_to_input, first_gw_match_lagged_player_data.iloc[:-1]])
        aggregated_first = first_gw_match_lagged_player_data.sum().to_frame().T
        aggregated_second = second_gw_match_lagged_player_data.sum().to_frame().T
        agg_lagged_player_data = pd.concat([aggregated_first, aggregated_second], axis=0)
        agg_lagged_player_data['minutes'] = agg_lagged_player_data['minutes'].replace([0], np.nan)
        agg_lagged_player_data_90 = agg_lagged_player_data.iloc[:, :-1].div(agg_lagged_player_data.iloc[:, -1], axis = 0)
        
    else:
        print(player, player_rows)
        agg_lagged_player_data = trailing_data[trailing_data['name']==player].loc[:, 'total_points':'minutes'].sum().to_frame().T
        agg_lagged_player_data['minutes'] = agg_lagged_player_data['minutes'].replace([0], np.nan)
        agg_lagged_player_data_90 = agg_lagged_player_data.iloc[:, :-1].div(agg_lagged_player_data.iloc[:, -1], axis=0)

    agg_lagged_player_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    n = trailing_data[trailing_data['name']==player].shape[0]
    if n > 0:
        agg_lagged_player_data_90['minutes'] = agg_lagged_player_data['minutes'] / n
    else:
        agg_lagged_player_data_90['minutes'] = 0

    agg_lagged_player_data_90.columns = 'tr_' + agg_lagged_player_data_90.columns
    agg_lagged_player_data_90.index = gw_player_data.index

    if agg_lagged_player_data_90.isna().all().all(): #concatenating empty or #NA dfs will soon be discontinued, this line ensures the code will continue to run then
        continue

    player_model_data = pd.concat([gw_player_data, agg_lagged_player_data_90], axis = 1)

    try:
        combined_data = pd.concat([player_model_data, combined_data], axis = 0)
    except NameError:
        combined_data = player_model_data



# %%
