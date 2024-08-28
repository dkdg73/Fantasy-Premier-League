#%%
import pandas as pd
import numpy as np
import os
import csv
import re

from model_dataset_functions import get_runs, get_prev_season, get_prev_gw, get_gw_data, build_lagged_file_list, get_trailing_data, combine_gw_trailing

pd.set_option('future.no_silent_downcasting', True) #to prevent FutureWarning: Downcasting behaviour in 'replace' is deprecated ...

# build full data for one gw in 4 steps
# - step 1: extract relevant data from relevant gw.csv file
# - step 2: extract opponent difficulty data from fixtures.csv
# - step 3: extract lagged data from prior gw.csv files, normalizing sums to per90mins over the entire lag period
# - step 4: concat gw_data with trailing_data into one dataframe showing each players gw performance + lagged performance per90mins
# generate dataset for n seasons by running the following loop

# generate full season dataset

# %% test full season generating function
def update_current_season_dataset():
    season = '2024-25'
    filepath = f'data/{season}/model_data.csv'
    pattern = r'^gw\d+\.csv$' # regex to match any file with 'gw*.csv'

    if not os.path.exists(filepath): #if model_data doesn't exist, create it
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
    try:
        model_data = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        model_data = pd.DataFrame()  # Initialize an empty DataFrame
    
    #number of gw csv files already downloaded
    gw_folder_contents = os.listdir(f'data/{season}/gws')
    gw_files = [filename for filename in gw_folder_contents if re.match(pattern, filename)]
    gw_count = len(gw_files)
    
    start = gw_count

    if model_data.empty:
        finish = 0
    else:
        gws_already_saved = len(model_data.groupby('gw').last())
        finish = gw_count - gws_already_saved
        if gw_count == gws_already_saved:
            print(f'gw model data uptodate, gws already saved = {gws_already_saved}')
            return
        
    for i in range(start, finish, -1): #previous code builds model_data in declining gws
        gw_df = get_gw_data(season, i)
        trailing_df = get_trailing_data(season, 1)
        comb = combine_gw_trailing(gw_df, trailing_df)
        
        if model_data.empty:
            model_data = comb
        else:
            model_data = pd.concat([comb, model_data], axis = 0)
    
    print('Model data uptodate to gw{start}, season {season}')
    return        
        

def generate_full_season_dataset(starting_season, seasons_to_run = 1, lags_for_trailing_data = 19):
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



#%%
season = '2023-24'
gw = 38
seasons_to_run = 4

for i in range(seasons_to_run): # required data unavailable in season 2017-18 and prior; last full season the code will run for is 2018-19

    runs = get_runs(season)
    lags = 19

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

# %%
