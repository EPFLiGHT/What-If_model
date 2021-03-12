from classes.context import Context
from classes.pipeline import Pipeline
import argparse
from classes.plot import Plot


# Instantiating context (configurations and data)
context = Context()
features = context.model_features()

# Run parameters
parser = argparse.ArgumentParser(description='IGH - WhatIf')
parser.add_argument('-g', '--gpu', help="Select GPU")
parser.add_argument('-c', '--country', required=True,
                    help='Select target country ISO code')
parser.add_argument('-l', '--load_checkpoint', action='store_true',
                    help='Load a model from a previous checkpoint')


gpu = parser.parse_args().gpu
target_iso = parser.parse_args().country.upper()

# Training columns
train_cols = features['demography'] + \
             features['sanitary'] + \
             features['weather'] + \
             features['policies'] + \
             features['mobi_google']



# Target column
target_col = 'shifted_r_estim'

# Set Dropna=False, important. Data is the dataframe cleaned
data = context.get_model_data(train_cols, target_col, dropna=False)

# If the selected country is not valid
if target_iso not in data['iso_code'].unique():
  raise Exception('Invalid country')

# If we want to run the model on gpu
if gpu is not None:
  gpu = int(gpu)

# Instantiating the training and testing pipeline
pipeline = Pipeline(data, train_cols, target_col, target_iso, context, gpu_id=gpu)

# If we want to restart from a checkpoint
if parser.parse_args().load_checkpoint:
  pipeline.load_from_checkpoint()
else:
  # Fitting the model
  pipeline.fit_pipeline(save_model=False)

# Making the prediction on the selected country
prediction = pipeline.predict()

# Plotting the results: main plot (prediction against ground truth)
plot = Plot(data, target_col, target_iso, save_path="./plots/")
plot.plot_results(prediction, show=False)

plot.plot_shap(pipeline.get_model(), pipeline.get_cols(), pipeline.get_data()[0], pipeline.get_data()[1], show=False)
plot.plot_shap(pipeline.get_model(), pipeline.get_cols(), pipeline.get_data()[0], pipeline.get_data()[1], plot_bars=True, show=False)