#%%
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True) #to prevent FutureWarning: Downcasting behaviour in 'replace' is deprecated ...

def get_runs(season):
    """return maximum number of runs (ie gws) in a season allowing for covid disruption
    """
    if season == '2019-20':
        return 47
    else:
        return 38
  
def get_prev_season(season):
    """
    Smmary:
        Generate prior season string for given season

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
        season = f'{yr1 - 1}-{yr2 -1}'
        if season == '2019-20': #cov season
            gws_in_season = 47
        else:
            gws_in_season = 38
        gw = gw - 1 + gws_in_season
    else:
        season = season
        gw = gw - 1

    return (season, gw)

def get_gw_data(season, gw):
    """
    Summary:
        reads gw{i} data for a given season and week i, adding 'opponent difficulty' from seperate fixtures.csv file

    Args:
        season (str): season in 'YYYY-YY'
        gw (int): gameweek

    Returns:
        type(df): cleaned gw dataframe
    """
    season_path = f'data/{season}/'
    gw_path = season_path + f'gws/'
    gw_filepath = gw_path + f'gw{gw}.csv'
    
    #step 1: get gameweek df
    gw_df = pd.read_csv(gw_filepath, encoding = 'ISO-8859-1')
    gw_df['gw'] = f'{season}-{gw}'
    params = ['gw', 'name', 'position', 'team', 'value', 'total_points', 'xP', 'goals_scored', 'assists', 'goals_conceded', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'influence', 'creativity', 'threat', 'minutes', 'was_home', 'opponent_team']
    missing_params = [param for param in params if param not in gw_df.columns]
    for col in missing_params:
        gw_df[col] = np.nan
    gw_df = gw_df.loc[:, params]

    #step 2: add opponent difficulty from fixtures.csv to gameweek df
    fixs_df = pd.read_csv(f'{season_path}fixtures.csv', encoding = 'ISO-8859-1')
    gw_fixs_df = fixs_df[fixs_df['event'] == gw]

    h_diff = gw_fixs_df.loc[:, ['team_h', 'team_h_difficulty']].rename(columns = {'team_h':'team', 'team_h_difficulty':'difficulty'})
    a_diff = gw_fixs_df.loc[:, ['team_a', 'team_a_difficulty']].rename(columns = {'team_a':'team', 'team_a_difficulty':'difficulty'})
    team_diff = pd.concat([h_diff, a_diff]).sort_values('team')
    team_diff = dict(team_diff.values)
    gw_df['opponent_team_difficulty']  = gw_df['opponent_team'].map(team_diff)
    
    return gw_df.reindex(columns = params)

def build_lagged_file_list(season, gw, lags):
    counter = 0

    while counter < lags:
        season, gw = get_prev_gw(season, gw)
        season_path = f'data/{season}/'
        gw_path = season_path + f'gws/'
        gw_data = get_gw_data(season, gw)

        if gw_data.empty == False:
            yield gw_path + f'gw{gw}.csv'
            counter += 1

def get_trailing_data(season, gw, lags = 19):
    """_summary_

    Args:
        season (_type_): _description_
        gw (_type_): _description_
        lags (int, optional): _description_. Defaults to 19.

    Returns:
        _type_: _description_
    """

    trailing_gw_filepaths = list(build_lagged_file_list(season, gw, lags))
    
    df_list = [pd.read_csv(filepath, encoding = 'ISO-8859-1') for filepath in trailing_gw_filepaths]

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
    
    # ensure elements in the 'name' columns are fomatted the same 
    # season 2019-20 and before names = 'First_Second_i' vs 'First Second' afterwards
    # ==> since players is derived from gw_data, searching for each player in players will return null from trailing_data
    trailing_data['name'] = trailing_data['name'].str.replace(r'[\d]+', '', regex=True).str.replace('_', ' ').str.strip()

 
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

def update_current_season_dataset():
    '''
    Summary:
        updates model_data.csv in the 2024-25 season folder with latest data from the gw folders
        NB can only update model_data.csv to be as current as the gw folder
    '''
    
    season = '2024-25'
    filepath = f'data/{season}/model_data.csv'
    pattern = r'^gw\d+\.csv$' # regex to match any file with 'gw*.csv'

    if not os.path.exists(filepath): #if model_data doesn't exist, create it
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
    try:
        model_data = pd.read_csv(filepath, index_col=0)
    except pd.errors.EmptyDataError:
        model_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    #number of gw csv files already downloaded
    gw_folder_contents = os.listdir(f'data/{season}/gws')
    gw_files = [filename for filename in gw_folder_contents if re.match(pattern, filename)]
    gw_count = len(gw_files)
    
    finish = gw_count

    if model_data.empty:
        start = 0
    else:
        gws_already_saved = len(model_data.groupby('gw').last())
        start = gw_count - gws_already_saved
        if gw_count == gws_already_saved:
            print(f'gw model data uptodate. Latest gw = {gw_count}; gws already saved = {gws_already_saved}.')
            return
        
    for i in range(start + 1, finish + 1):
        gw_df = get_gw_data(season, i)
        trailing_df = get_trailing_data(season, i)
        comb = combine_gw_trailing(gw_df, trailing_df)
        
        if model_data.empty:
            model_data = comb
        else:
            model_data = pd.concat([model_data, comb], axis = 0)
    
    model_data.to_csv(filepath)
    print(f'Model data updated with {finish-start} new gws to gw{finish}, season {season}')
    return    

def generate_full_season_dataset(starting_season, seasons_to_run = 1, lags_for_trailing_data = 19):
    """ Creates and stores all player all season datbase in model_data.csv for as many seasons to run as input, 
        Stores in model_data.csv in each season's folder
        NB/ ONLY WORKS WHEN GW FOLDER IS COMPLETE, REWRITE WLD BE REQUIRED TO READ INCOMPLETE GW FOLDERS 

    Args:
        starting_season (str): season to start gathering data
        seasons_to_run (int, optional): number of prior seasons to build databases for
        lags_for_trailing_data (int, optional): lags to us in calculation of trailing data (defaults to 19).
        
    """
    season = starting_season
    gw = get_runs(season) 
    seasons_to_run = seasons_to_run

    for i in range(seasons_to_run): # required data unavailable in season 2017-18 and prior; last full season the code will run for is 2018-19

        runs = get_runs(season)
        lags = lags_for_trailing_data

        path = f'data/{season}/'
        
        for i in range(runs):
            print(f'compiling data for gameweek {gw}, season {season}')
            try:
                gw_df = get_gw_data(season, gw)
                if gw_df.empty:
                    season, gw = get_prev_gw(season, gw)
                    continue
                trailing_df = get_trailing_data(season, gw, lags)
            except FileNotFoundError as e:
                print(f'Terminated at gameweek: {gw}; season: {season}; because: {e}')
                break

            comb = combine_gw_trailing(gw_df, trailing_df)
            
            if i == 0:
                model_data = comb
            model_data = pd.concat([comb, model_data], axis=0)
            
            season, gw = get_prev_gw(season, gw)
        
        model_data.to_csv(f'{path}model_data.csv')
    print('Data gathering complete')


