import yaml
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
from pytorch_lightning import seed_everything


class Context:

  def __init__(self):
    with open('config/epi_config.yaml', 'r', encoding='utf-8') as file:
      self.__epi_config = yaml.load(file, Loader=yaml.FullLoader)

    with open('config/model_config.yaml', 'r', encoding='utf-8') as file:
      self.__model_config = yaml.load(file, Loader=yaml.FullLoader)

    with open('config/model_features.yaml', 'r', encoding='utf-8') as file:
      self.__model_features = yaml.load(file, Loader=yaml.FullLoader)

    # Setting the seed for this file
    seed_everything(self.__model_config['seed'])

  def epi_config(self):
    return self.__epi_config

  def model_config(self):
    return self.__model_config

  def model_features(self):
    return self.__model_features

  @staticmethod
  def __iso_to_continent(iso_code):
    """Given an iso code return the alpha2 representation"""

    unknown = {
      'ESH': 'AF',
      'SXM': 'NA'
    }
    # If the iso_code is unknown for `pycountry`
    if iso_code in unknown:
      return unknown[iso_code]

    alpha_2 = pycountry.countries.get(alpha_3=iso_code).alpha_2

    try:
      continent = pc.country_alpha2_to_continent_code(alpha_2)
    except KeyError:
      return 'unknown'

    return continent

  @staticmethod
  def __filter_valid_countries(data, min_ratio):
    """Filter the dataframe to check if there are sufficient days with r_estim"""
    iso_codes = []

    for iso_code in data.iso_code.unique():
      data_country = data[data.iso_code == iso_code].r_estim.dropna()
      n_values = data_country.shape[0]


      # If rows exist for that country with a valid r_estim value
      if n_values > 0:
        n_days = (data_country.index[-1] - data_country.index[0]).days + 1
        ratio = n_values / n_days

        if ratio > min_ratio:
          iso_codes.append(iso_code)


    return data[data['iso_code'].isin(iso_codes)]

  def get_model_data(self, train_cols, target_col, continents=None,
                     dropna=True):
    # Loading the dataframe
    data = pd.read_csv('data/merged_data/model_data_owid.csv',
                       parse_dates=['date']).set_index('date')

    #data = pd.read_csv('data/merged_data/economic_cli.csv',
                        #parse_dates=['date']).set_index('date')

    # Taking only countries for which the number of NOT NAN in r_estim is > min ratio
    data = Context.__filter_valid_countries(data,
                                            self.__epi_config['min_ratio'])

    # Putting NaN where the reported r is larger than max_r
    data['r_estim'] = data['r_estim'].apply(
      lambda x: np.nan if x >= self.__epi_config['max_r'] else x)

    # Generate shifted columns for r estimations (11 days)
    # On average the reported cases are referred to 11 days before
    data['shifted_r_estim'] = data['r_estim'].shift(
      self.__epi_config['r_shift']).where(
      data['iso_code'].eq(data['iso_code'].shift(self.__epi_config['r_shift'])))

    # Generate weekdays data
    weekdays = ['monday', 'tuesday', 'wednesday', 'thursday',
                'friday', 'saturday', 'sunday']
    reference_monday = pd.Timestamp('20200227')

    data['weekday'] = pd.Series(
      [weekdays[i] for i in list((data.index - reference_monday).days % 7)])

    # Generate dummies for weekday and continents
    data = pd.get_dummies(data, prefix='', columns=['weekday', 'continent'])

    # Saving some variables before dropping columns
    cumul_cases = data['cumul_case']

    # Remove unused columns
    data = data[['iso_code'] + train_cols + [target_col]]

    #data = data[(data.index >= '2020-01-01')] #2021-02-02

    # If policy is not defined (wasn't applied at that time), its level is 0
    for x in data.columns:
      if 'level' in x:
        data[x] = data[x].fillna(0)

    # Selecting all countries which surpassed `min_cases`
    iso_codes = data[cumul_cases > self.__epi_config['min_cases']].iso_code.unique()

    # Getting only rows related to those countries
    data = data[data['iso_code'].isin(iso_codes)]

    # If continents filter is given
    if continents is not None:
      data = data[data.apply(lambda row:
                             Context.__iso_to_continent(
                               row.iso_code) in continents,
                             axis=1)]

    # Remove lines for which we don't have complete data
    if dropna:
      data = data.dropna(subset=train_cols)

    data = data[data.index >= '2020-04-01']
    print(f"Number of considered countries: {len(data.iso_code.unique())}")
    return data
