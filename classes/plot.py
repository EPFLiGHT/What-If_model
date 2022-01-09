import numpy as np
import torch
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Plot:
  def __init__(self, df, target_col, target_country, country_name, save_path):
    self.__df = df
    self.__target_col = target_col
    self.__target_country = target_country
    self.__save_path = save_path
    self.__country_name = country_name

  @staticmethod
  def __compute_diagonal_mean(values):
    """ Each day will contribute to a prediction (with all its features) 7 times, but each time its features will have a
    different importance. So we do a diagonal mean over the importances. """

    diagonal_mean = []
    for feature in range(values.shape[2]):

      # 331 x 7, submatrix of the single feature.
      submatrix = values[:, :, feature]
      f = []
      for i in range(-submatrix.shape[0] + 1, submatrix.shape[1]):
        f.append(submatrix.diagonal(i).mean())
      diagonal_mean.append(f)

    return np.array(diagonal_mean).T

  @staticmethod
  def __compute_shap(fitted_model, train_data, validation_data):
    """ Function to compute the feature importances using the Gradient Explainer method of the SHAP library """

    # Instantiating the explainer
    explainer = shap.GradientExplainer(fitted_model, [train_data[0], train_data[1]])

    # Computing the SHAP values on the prediction
    shap_values = explainer.shap_values([validation_data[0], validation_data[1]])

    # Variable features: computing the diagonal mean
    variable_val_data_mean = Plot.__compute_diagonal_mean(validation_data[1].numpy())
    variable_shap_mean = Plot.__compute_diagonal_mean(shap_values[1])

    return validation_data[0].numpy(), shap_values[0], variable_val_data_mean, variable_shap_mean

  def __plot_shap_bars(self, df_shap, df, ax):
    # Making a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index', axis=1)

    # Determining the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
      b = np.corrcoef(shap_v[i], df_v[i])[1][0]
      corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)

    # Making a dataframe: column 1 is the feature, and column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, '#eb3456', '#388cf3')

    # Plotting it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']
    k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, legend=False, ax=ax)
    ax.set_title(f"Policy impact for {self.__target_country}")
    ax.set_xlabel("SHAP Value (Blue = Lower reproduction rate)")

  def plot_shap(self, fitted_model, const_col_names, var_col_names, train_data, validation_data, cols_to_drop=[],
                plot_bars=False, show=True):

    # Computing shap values for constant and variable features, on the validation data
    const_val_data, const_shap_values, var_val_data, var_shap_values = self.__compute_shap(fitted_model, train_data,
                                                                                           validation_data)

    const_val_data_df = pd.DataFrame(data=const_val_data, columns=const_col_names)
    var_val_data_df = pd.DataFrame(data=var_val_data, columns=var_col_names)
    var_val_data_df.drop(columns=cols_to_drop, inplace=True)

    var_shap_df = pd.DataFrame(data=var_shap_values, columns=var_col_names)
    var_shap_df.drop(columns=cols_to_drop, inplace=True)

    if plot_bars:
      fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
      self.__plot_shap_bars(const_shap_values, const_val_data_df, ax1)

      fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
      self.__plot_shap_bars(var_shap_df.values, var_val_data_df, ax2)

      if not show:
        fig1.savefig(f"{self.__save_path}{self.__target_country}_barplot_const.png", bbox_inches='tight')
        fig2.savefig(f"{self.__save_path}{self.__target_country}_barplot_var.png", bbox_inches='tight')
      else:
        fig1.show()
        fig2.show()

    else:
      if not show:
        shap.summary_plot(const_shap_values, const_val_data_df, show=False, plot_size = (15, 15))
        plt.savefig(f"{self.__save_path}{self.__target_country}_beeswarm_const.png", bbox_inches='tight')
        plt.clf()
        shap.summary_plot(var_shap_df.values, var_val_data_df, show=False, plot_size = (15, 15))
        plt.savefig(f"{self.__save_path}{self.__target_country}_beeswarm_var.png", bbox_inches='tight')
        plt.clf()
      else:
        plt.title(f"Features importances for {self.__target_country}")
        shap.summary_plot(const_shap_values, const_val_data_df)
        plt.title(f"Features importances for {self.__target_country}")
        shap.summary_plot(var_shap_df.values, var_val_data_df)

  def plot_results(self, pred, std=None, target_name='R_E', plot_error=True, show=True, plot_ci = False, axis=None, show_x_label=False, show_legend=False):
    """Plot a target prediction for a given country"""

    # Computing the ground truth (true R of the val set)
    test_indices = self.__df['iso_code'] == self.__target_country
    index = self.__df.loc[test_indices].index
    ground = self.__df.loc[test_indices][self.__target_col]

    # Mean squared error of the prediction vs the true rep. rate
    mse = np.square(ground - pred).mean()
    print(f'The mean average error was {mse}')

    # Absolute error for every prediction
    error_curve = np.abs(ground - pred)

    if axis is None:
      fig = plt.figure(figsize=(12, 3), dpi=100)
      axis = plt.gca()

    # Plot curves
    axis.set_title(f'Predictions for {self.__country_name} ({self.__target_country})')
    axis.plot(index, ground, label=f'Reported {target_name}')
    axis.plot(index, pred, label=f'Predicted {target_name}')

    if plot_error:
      axis.plot(index, error_curve, label='Absolute error')

    # Default confidence interval, not computed
    if plot_ci:
      lower = self.__df[test_indices]['epiforecasts_effective_reproduction_number_lower_90']
      upper = self.__df[test_indices]['epiforecasts_effective_reproduction_number_upper_90']
      axis.fill_between(index, lower, upper, facecolor='r', alpha=.3)

    if std is not None:
      # 95% Confidence interval. Std is present only if we predict with Monte Carlo Dropout
      ci = 1.96 * std
      axis.fill_between(index, (pred - ci), (pred + ci), facecolor='r', alpha=.3)

    # Setup x ticks
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=28))
    axis.xaxis.set_tick_params(rotation=90)

    axis.set_ylabel(target_name)

    # Legend and labels
    if show_legend:
      axis.legend()

    if show_x_label:
      axis.set_xlabel('Date')
    #axis.axhline(color='black', lw=1, ls='--', y=1)
    #axis.axhline(color='black', lw=1)

    if axis is None:
      # Save plot in memory
      if not show:
        plt.savefig(self.__save_path + self.__target_country + '.pdf', bbox_inches='tight')
      else:
        plt.show()

      plt.close()
