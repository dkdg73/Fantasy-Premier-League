#%%
import pandas as pd
import numpy as np

from model_dataset_functions import generate_full_season_dataset

seasons = ['2023-24', '2022-23', '2021-22', '2020-21']

for season in seasons:
    generate_full_season_dataset(season)
