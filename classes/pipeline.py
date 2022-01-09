import numpy as np
import pandas as pd
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from classes.context import Context
from classes.hybrid import HybridLSTM


class Pipeline:

  def __init__(self, df, train_cols, target_col, target_country, window_size,
               context: Context, discount_col=None, gpu_id=None):

    # Setting the seed for this file
    seed_everything(context.model_config()['seed'])

    self.__context = context
    self.__df = df
    self.__train_cols = train_cols
    self.__target_col = target_col

    # Used to assign a weight to each datapoint in the economic predictions
    self.__discount_col = discount_col

    self.__target_country = target_country
    self.__model = None
    self.__train_data = None  # Data used for training, will be returned by split_train_val
    self.__val_data = None  # Data used for testing / validation.
    self.__const_cols = None  # Name of the constant features
    self.__var_cols = None  # Name of the variable features
    # Mask used to remember for which days it was not possible to make a prediction.
    # Check the `inject_nans` function
    self.__prediction_mask = None
    self.__window_size = window_size



    self.__train_mean, self.__train_std = 0, 0

    if gpu_id is not None:
      self.__gpus = [gpu_id]
      torch.cuda.set_device(gpu_id)

    else:
      self.__gpus = 0

  def __split_features(self):
    """ Split in constant and variable cols"""

    constant_cols = set()
    variable_cols = set()

    iso_code_list = self.__df.iso_code.unique()

    for col in self.__train_cols:
      is_constant = True
      for iso_code in iso_code_list:
        test = self.__df[self.__df.iso_code == iso_code][col].dropna()

        if len(test) == 0:
          continue

        if test.max() - test.min() != 0:
          variable_cols.add(col)
          is_constant = False
          break
      if is_constant:
        constant_cols.add(col)

    self.__const_cols = sorted(list(constant_cols))
    self.__var_cols = sorted(list(variable_cols))


  def __normalize(self):
    """ Normalizing data """

    # Decide what columns to normalize
    train_ndata = self.__df[self.__df['iso_code'] != self.__target_country]
    stds = train_ndata.std().values
    norm_cols = [c for c, s in zip(self.__train_cols, stds)
                 if (c not in self.__context.model_config()['no_norm_cols'])
                 and (s != 0)]

    # Get train weights on train countries to avoid info leak
    self.__train_mean = train_ndata[norm_cols].mean()
    self.__train_std = train_ndata[norm_cols].std()

    # Normalizing both (train and test) using only the columns to normalize
    self.__df[norm_cols] = (self.__df[norm_cols] - self.__train_mean) / self.__train_std

  def __sliced_hybrid(self, group, const_cols, var_cols):
    """ Slices a df to generate hybrid_lstm training data (stride=1), assumes the df is sorted by date and has no date
    dropped """

    # Regular slicing
    train_cols = sorted(var_cols) + sorted(const_cols)
    dates = group.index
    slices = np.array([group[train_cols].values[i:i + self.__window_size] for i in
                       range(len(group) - self.__window_size + 1)])
    targets = group[self.__target_col].values[self.__window_size - 1:]

    # Mask for training and prediction
    valid_inp = np.array([not np.isnan(x).any() for x in slices])
    valid_tar = np.array([not np.isnan(y) for y in targets])
    valid_dates = valid_inp & valid_tar
    slices = slices[valid_dates]
    targets = targets[valid_dates]

    # Pop Mean of const features
    const_features = slices[:, :, -len(const_cols):].mean(axis=1)
    var_features = slices[:, :, :len(var_cols)]

    if self.__discount_col is not None:
      discounts = group[self.__discount_col].values[self.__window_size - 1:]
      discounts = discounts[valid_dates]
      return const_features, var_features, targets, valid_dates, discounts

    return const_features, var_features, targets, valid_dates

  def __split_train_val(self):
    """Trains a model for a given country in leave-one-out and make a prediction"""

    self.__split_features()  # Split features into constant and variable
    self.__normalize()

    # Get train and test indices for given df
    train_indices = self.__df['iso_code'] != self.__target_country
    test_indices = ~train_indices
    
    # Train and validation data
    grouped = self.__df.loc[train_indices].groupby('iso_code').apply(
      lambda group: self.__sliced_hybrid(group, self.__const_cols, self.__var_cols))

    # Constant and variable features and target for each country
    # for the training (without keeping the `iso_code`)
    X_train_const = torch.from_numpy(
      np.array([x for g in grouped for x in g[0]]))
    X_train_var = torch.from_numpy(np.array([x for g in grouped for x in g[1]]))
    y_train = torch.from_numpy(np.array([x for g in grouped for x in g[2]]))

    discount_train = None
    if self.__discount_col is not None:
      discount_train = torch.from_numpy(np.array([x for g in grouped for x in g[4]]))

    self.__train_data = (X_train_const, X_train_var, y_train, discount_train)

    # Constant and variable features and target for each country
    # for the validation (only one country)
    sliced = self.__sliced_hybrid(self.__df[test_indices], self.__const_cols, self.__var_cols)

    discount_val = None
    if self.__discount_col is not None:
      discount_val = torch.from_numpy(np.array(sliced[4]))

    self.__val_data = (torch.from_numpy(sliced[0]), torch.from_numpy(sliced[1]), torch.from_numpy(sliced[2]),
                       discount_val)

    self.__prediction_mask = sliced[3]
    
    print(self.__train_data)
    print("\n\nValidation data\n\n")
    print(self.__val_data)

  def __save_model(self, trainer):
    """Save the model in the current parameter state"""

    save_dir = f'./models/{self.__target_country}'
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    trainer.save_checkpoint(os.path.join(save_dir, "model.ckpt"))

  def __train_model(self, save=True):

    np.random.seed(42)
    torch.manual_seed(42)

    patience = self.__context.model_config()['patience']
    max_epochs = self.__context.model_config()['max_epochs']
    auto_lr_find = self.__context.model_config()['auto_lr_find']
    auto_scale_batch_size = self.__context.model_config()['auto_scale_batch_size']

    # Create model
    self.__model = HybridLSTM(self.__context, {"var_cols": self.__var_cols,
                                               "const_cols": self.__const_cols,
                                               "target_col": self.__target_col,
                                               "country": self.__target_country,
                                               "past_window": self.__window_size})
    self.__model.create_dataloaders(self.__train_data, self.__val_data)
    

    # Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=patience,
                                        verbose=True,
                                        mode='min')

    # Training
    logger = TensorBoardLogger('tb_logs',
                               name=f'hybrid_{self.__target_country}')

    trainer = pl.Trainer(gpus=self.__gpus, min_epochs=1, max_epochs=max_epochs,
                         auto_lr_find=auto_lr_find,
                         auto_scale_batch_size=auto_scale_batch_size,
                         progress_bar_refresh_rate=0,
                         logger=logger) #callbacks=[early_stop_callback], 

    trainer.fit(self.__model)

    # Save model
    if save:
      self.__save_model(trainer)

  def __inject_nans(self, pred):
    """Fill array with Nans, assume it has appended nans already to account for past_window size"""

    idx_to_fill = [e for e, x in enumerate(self.__prediction_mask) if x]

    # Count appended nan and remove them
    nb_appended = np.isnan(pred).sum()
    pred = pred[~np.isnan(pred)]

    new_pred = np.full(len(self.__prediction_mask), np.nan)

    for p, idx in zip(pred, idx_to_fill):
      new_pred[idx] = p

    # Restore nan
    new_pred = np.append([np.nan] * nb_appended, new_pred)
    return new_pred

  def single_prediction(self, var_names, const_names, var_features, const_features, norm_const=False, norm_var=False):

    df_vars = pd.DataFrame(var_features, columns=var_names)
    df_const = pd.DataFrame(const_features, columns=const_names)

    if norm_var:
      for var_feat_name in df_vars.columns:
        df_vars[var_feat_name] = (df_vars[var_feat_name] - self.__train_mean[var_feat_name]) / self.__train_std[var_feat_name]

    if norm_const:
      for const_feat_name in df_const.columns:
        df_const[const_feat_name] = (df_const[const_feat_name] - self.__train_mean[const_feat_name]) / self.__train_std[const_feat_name]

    pred = self.__model.eval()(torch.from_numpy(df_const.values.reshape(1,-1)),
                               torch.from_numpy(df_vars.values).unsqueeze(0)).detach()

    return pred

  def predict(self):
    # Generate Final Prediction
    pred = self.__model.eval()(*(self.__val_data[0], self.__val_data[1])).detach()
    pred = pred.reshape(pred.size(0)).numpy()
    pred = np.append([np.nan] * (self.__window_size - 1), pred).flatten()

    # Appending NaNs for impossible predictions (missing data in features)
    pred = self.__inject_nans(pred)

    return pred

  def predict_mcdropout(self, n_samples=20):
    mean_pred, std_pred = self.__model.sample_predict(self.__val_data[0], self.__val_data[1],
                                                      n_samples)

    mean_pred = mean_pred.reshape(mean_pred.size(0)).numpy()
    mean_pred = np.append([np.nan] * (self.__window_size - 1), mean_pred).flatten()

    std_pred = std_pred.reshape(std_pred.size(0)).numpy()
    std_pred = np.append([np.nan] * (self.__window_size - 1), std_pred).flatten()

    # Appending NaNs for impossible predictions (missing data in features)
    mean_pred = self.__inject_nans(mean_pred)
    std_pred = self.__inject_nans(std_pred)

    return mean_pred, std_pred

  def fit_pipeline(self, save_model=True):
    print("Training the model...\n")


    # Split trainig and validation
    self.__split_train_val()
    # Train the model
    self.__train_model(save_model)

  def load_from_checkpoint(self):
    print("Loading the model from a previous checkpoint...\n")
    self.__model = HybridLSTM.load_from_checkpoint( f"./models/{self.__target_country}/{self.__target_country}-policies-benchmarked.ckpt",
                                                    torch.device(f'cuda:{self.__gpus}' if self.__gpus else 'cpu'),
                                                    context=self.__context)
    self.__split_train_val()

  
  def get_data(self):
    return self.__train_data, self.__val_data

  def get_cols(self):
    return self.__const_cols, self.__var_cols

  def get_model(self):
    return self.__model

  def get_denorm_data(self, names, values):

    df = pd.DataFrame(values, columns=names)

    for var_feat_name in df.columns:
      df[var_feat_name] = (df[var_feat_name] * self.__train_std[var_feat_name]) + self.__train_mean[var_feat_name]

    return df.values