#%%
import os
import pandas as pd
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from model_dataset_functions import get_prev_season

path = 'data/'
contents = os.listdir(path)
folders = [found for found in contents if os.path.isdir(path + found)]

start = folders.index('2020-21')
end = folders.index('2024-25')

seasons = folders[start:end]
filepaths = [f'{path}{season}/model_data.csv' for season in seasons]
dfs = [pd.read_csv(filepath) for filepath in filepaths]

season_data_dic = dict(zip(seasons, dfs))

all_data = pd.concat(dfs, axis=0)

def build_position_data_dic(season_data_dic, position):
    """
    Suumary:
        Filters {k:v} = {season: data} dictionary according to given position

    Args:
        season_data_dic (dic): {season: data} dictionary for al players
        position (str): player position
        minute_filter (int): filter to apply to average mins per match over the trailing period

    Raises:
        NameError: Player position string must be in ['GK', 'DEF', 'FWD', 'MID']

    Returns:
        dic: {season: data} dictionary by position
    """
    allowed_positions = ['GK', 'DEF', 'FWD', 'MID']
    if position not in allowed_positions:
        raise NameError(f'Position must be one of {allowed_positions}')
    
    if position == 'GK':
        data_dic = {
            season: season_data_dic[season][(season_data_dic[season]['position'] == position) | (season_data_dic[season]['position'] == 'GKP')].copy() 
            for season in season_data_dic.keys()
            }
    else:
        data_dic = {
            season: season_data_dic[season][season_data_dic[season]['position'] == position].copy() 
    for season in season_data_dic.keys()
        }
    
    return data_dic
 

def run_regression(df, y, x, minute_filter = 0):
    """
    Summary:
        Returns sm.OLS object prints summary of a regression taking y as dependent, x as independent variables from the DataFrame df
    """ 
    if type(y) != str:
        raise TypeError('Dependent variable y must be str.')
    if type(y) == str:
        y = y
    elif type(y) == list:
        y = y
    else:
        raise TypeError('Independent variables x must be str or list.')
    if 'tr_minutes' in df.columns:       
        df = df[df['tr_minutes'] > minute_filter].copy()
    
    y = df.loc[:, y] 
    x = df.loc[:, x]
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    print(model.summary())
    return model

# %% GK regressions

gk_data = build_position_data_dic(season_data_dic, 'GK')
gk_value_models = {}

for season in seasons:
    gk_data[season]['tr_rel'] = gk_data[season].loc[:, 'tr_total_points'] - gk_data[season].loc[:, 'tr_xP']
    gk_value_models[season] = run_regression(gk_data[season], 'value', ['tr_goals_conceded', 'tr_influence'], minute_filter=80)
    
    

#%%
dfd_data = build_position_data_dic(season_data_dic, 'DEF')
mid_data = build_position_data_dic(season_data_dic, 'MID')
fwd_data = build_position_data_dic(season_data_dic, 'FWD')



    

# %%
