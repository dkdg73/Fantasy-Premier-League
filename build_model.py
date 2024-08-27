#%%
import os
import pandas as pd
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from model_dataset_functions import get_prev_season

# the folowing functions help build datasets by season and position 
# which can then be used to build regression models on
# most of the inputs and outputs are based on dictionaries:
#   - season (str) are typically keys
#   - data (df) value 
 
def build_position_dic(season_data_dic, position):
    """
    Summary:
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
        
    for season in season_data_dic.keys():
        data_dic[season]['tr_rel'] = data_dic[season].loc[:, 'tr_total_points'] - data_dic[season].loc[:, 'tr_xP']
   
    return data_dic
 

def run_regression(df, y, x, minute_filter = 0, print_output = False):
    """
    Summary:
        Returns sm.OLS(x, y).fit() object prints summary of a regression taking y as dependent, x as independent variables from the DataFrame df
    """ 
    if type(y) != str:
        raise TypeError('Dependent variable y must be str.')
    if type(x) == str:
        x = x
    elif type(x) == list:
        x = x
    else:
        raise TypeError('Independent variables x must be str or list.')
    if 'tr_minutes' in df.columns:       
        df = df[df['tr_minutes'] > minute_filter].copy()
    
    y = df.loc[:, y] 
    x = df.loc[:, x]
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()

    if print_output == True:
        print(model.summary())
        print('\n\n\n')

    return model

def build_model_dic(position, data_dic, y, x, minute_filter = 80, print_output=False):
    """
    Summary:
        generate a dictionry of {season: statsmodel object} a given position, y and x
        
    Args:
        position (str): GK, DEF, FWD, MID
        data_dic (dic): dictionary of {season: data_df}
        y (str or lst): dependent variable
        x (str or lst): independent variables
        minute_filter (int): to exclude players with trailing ave mpg below given threshold
        print_output (bool): print model runs in output terminal
    
    Output:
        models dictionary {season: statsmodel.OLS} for given position and filters 
    """
    
    seasons = data_dic.keys()
    position_dic = build_position_dic(data_dic, position)
    
    models_dic = {}
    
    for season in seasons:
        models_dic[season] = run_regression(position_dic[season], y, x, minute_filter=minute_filter, print_output=print_output)
 
    return models_dic

def generate_params_df(models_dic, save_name = None):
    """_summary_

    Args:
        models_dic (_type_): _description_
    """
            
    coeffs = pd.concat(
        [models_dic[season].params for season in models_dic.keys()], 
        axis=1
        )
    
    ses = pd.concat(
        [models_dic[season].bse for season in models_dic.keys()], 
        axis=1
        )
    
    pvals = pd.concat(
        [models_dic[season].pvalues for season in models_dic.keys()], 
        axis=1
        )

    model_results = pd.concat([coeffs, ses, pvals], keys=['Coefficients', 'Std Errors', 'p-vals'])
    model_results.columns = [key for key in models_dic.keys()]
    if save_name != None:
        model_results.to_csv(f'models/{save_name}.csv')
    
    return model_results


#%%

path = 'data/'
contents = os.listdir(path)
folders = [found for found in contents if os.path.isdir(path + found)]

start = folders.index('2020-21')
end = folders.index('2024-25')

seasons = folders[start:end]
filepaths = [f'{path}{season}/model_data.csv' for season in seasons]
dfs = [pd.read_csv(filepath) for filepath in filepaths]

season_data_dic = dict(zip(seasons, dfs))

#%%
#build models for each season
gk_params = ['value', 'tr_goals_conceded', 'tr_influence', 'tr_rel']
gk_model = build_model_dic('GK', season_data_dic, 'value', gk_params, print_output=False)
gk_model_df = generate_params_df(gk_model, save_name='gk')

gk_coeffs = gk_model_df.loc['Coefficients'].median(axis=1)
gk_data = build_position_dic(season_data_dic, 'GK')
gk_valuation_23 = gk_data['2023-24'].loc[:, ['name'] + gk_params]


# %%
