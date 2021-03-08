from classes.context import Context
from classes.pipeline import Pipeline
import argparse

context = Context()
context.set_seed()

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

gpu = parser.parse_args().gpu
if gpu is not None:
  gpu = int(gpu)

pipeline = Pipeline(data, train_cols, target_col, 'CHE', context, gpu_id=gpu)
pred, std = pipeline.fit_predict_pipeline(save_model=False)

print(pred, std)
