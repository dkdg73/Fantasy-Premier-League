#%%
import pandas as pd
import numpy as np

from model_dataset_functions import get_runs, get_prev_season, get_prev_gw, get_gw_data, build_lagged_file_list, get_trailing_data, combine_gw_trailing

pd.set_option('future.no_silent_downcasting', True) #to prevent FutureWarning: Downcasting behaviour in 'replace' is deprecated ...

# build full data for one gw in 4 steps
# - step 1: extract relevant data from relevant gw.csv file
# - step 2: extract opponent difficulty data from fixtures.csv
# - step 3: extract lagged data from prior gw.csv files, normalizing sums to per90mins over the entire lag period
# - step 4: concat gw_data with trailing_data into one dataframe showing each players gw performance + lagged performance per90mins
# generate dataset for n seasons by running the following loop

# generate dataset
season = '2024-25'
gw = 1
seasons_to_run = 1

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
