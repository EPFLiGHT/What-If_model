from classes.context import Context
from classes.pipeline import Pipeline
import argparse

context = Context()


features = context.model_features()

train_cols = features['demography'] + \
             features['sanitary'] + \
             features['weather'] + \
             features['policies']

target_col = 'shifted_r_estim'

# Set Dropna=False, important
data = context.get_model_data(train_cols, target_col, dropna=False)

parser = argparse.ArgumentParser(description='IGH - WhatIf')
parser.add_argument('-g', '--gpu', help="Select GPU")
parser.add_argument('-c', '--country', required=True,
                    help='Select target country ISO code')

gpu = parser.parse_args().gpu
target_iso = parser.parse_args().country.upper()

if target_iso not in data['iso_code'].unique():
  raise Exception('Invalid country')

if gpu is not None:
  gpu = int(gpu)

pipeline = Pipeline(data, train_cols, target_col, target_iso, context, gpu_id=gpu)
pred, std = pipeline.fit_predict_pipeline(save_model=False)

# Std is None if MC_Dropout in fit_predict_pipeline is false.
mse = pipeline.plot_results(pred, std, save_path='./plots/')


print(f'The mean average error was {mse}')



