import numpy as np
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from classes.context import Context
from classes.hybrid import HybridLSTM


class Pipeline:

  def __init__(self, df, train_cols, target_col, target_country,
               context: Context, gpu_id=None):
    self.__context = context
    self.__df = df
    self.__train_cols = train_cols
    self.__target_col = target_col
    self.__target_country = target_country
    self.__model = None

    if gpu_id is not None:
      self.__gpus = 1
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

    constant_cols = sorted(list(constant_cols))
    variable_cols = sorted(list(variable_cols))

    return constant_cols, variable_cols

  def __normalize(self):
    """ Normalizing data """

    # Decide what columns to normalize
    train_ndata = self.__df[self.__df['iso_code'] != self.__target_country]
    stds = train_ndata.std().values
    norm_cols = [c for c, s in zip(self.__train_cols, stds)
                 if (c not in self.__context.model_config()['no_norm_cols'])
                 and (s != 0)]

    # Get train weights on train countries to avoid info leak
    train_mean = train_ndata[norm_cols].mean()
    train_std = train_ndata[norm_cols].std()

    # Normalizing both (train and test) using only the columns to normalize
    self.__df[norm_cols] = (self.__df[norm_cols] - train_mean) / train_std

  def __sliced_hybrid(self, group, const_cols, var_cols):
    """Slices a df to generate hybrid_lstm training data (stride=1), assumes the df is sorted by date and has no date dropped"""

    past_window = self.__context.model_config()['past_window']

    # Regular slicing
    train_cols = var_cols + const_cols
    dates = group.index
    slices = np.array([group[train_cols].values[i:i + past_window] for i in
                       range(len(group) - past_window + 1)])
    targets = group[self.__target_col].values[past_window - 1:]

    # Mask for training and prediction
    valid_inp = np.array([not np.isnan(x).any() for x in slices])
    valid_tar = np.array([not np.isnan(y) for y in targets])
    valid_dates = valid_inp & valid_tar
    slices = slices[valid_dates]
    targets = targets[valid_dates]

    # Pop Mean of const features
    const_features = slices[:, :, -len(const_cols):].mean(axis=1)
    var_features = slices[:, :, :len(var_cols)]

    return const_features, var_features, targets, valid_dates

  def __split_train_val(self):
    """Trains a model for a given country in leave-one-out and make a prediction"""

    const_cols, var_cols = self.__split_features()
    self.__normalize()

    # Get train and test indices for given df
    train_indices = self.__df['iso_code'] != self.__target_country
    test_indices = ~train_indices

    # Train and validation data
    grouped = self.__df.loc[train_indices].groupby('iso_code').apply(
      lambda group: self.__sliced_hybrid(group, const_cols, var_cols))

    # Constant and variable features and target for each country
    # for the training (without keeping the `iso_code`)
    X_train_const = torch.from_numpy(
      np.array([x for g in grouped for x in g[0]]))
    X_train_var = torch.from_numpy(np.array([x for g in grouped for x in g[1]]))
    y_train = torch.from_numpy(np.array([x for g in grouped for x in g[2]]))
    train_data = (X_train_const, X_train_var, y_train)

    # Constant and variable features and target for each country
    # for the validation (only one country)
    sliced = self.__sliced_hybrid(self.__df[test_indices], const_cols, var_cols)

    val_data = (torch.from_numpy(sliced[0]), torch.from_numpy(sliced[1]),
                torch.from_numpy(sliced[2]))

    # The last returned value is the mask used to remember for which days it was
    # not possible to make a prediction. Check the `inject_nans` function
    # sliced[3] is the mask
    return train_data, val_data, const_cols, var_cols, sliced[3]

  def __save_model(self, trainer):

    """Save the model in the current parameter state"""
    save_dir = f'./models/{self.__target_country}'
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    trainer.save_checkpoint(os.path.join(save_dir, "model.ckpt"))

  def __train_model(self, var_cols, const_cols, train_data, val_data,
                    save=True):

    np.random.seed(42)
    torch.manual_seed(42)

    # Same as train_data[1].size(1)
    past_window = self.__context.model_config()['past_window']
    patience = self.__context.model_config()['patience']
    max_epochs = self.__context.model_config()['max_epochs']
    auto_lr_find = self.__context.model_config()['auto_lr_find']
    auto_scale_batch_size = self.__context.model_config()[
      'auto_scale_batch_size']

    # Create model
    model = HybridLSTM(self.__context, {"var_cols": var_cols,
                                        "const_cols": const_cols,
                                        "target_col": self.__target_col,
                                        "country": self.__target_country,
                                        "past_window": past_window})
    model.create_dataloaders(train_data, val_data)

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
                         callbacks=[early_stop_callback], logger=logger)

    trainer.fit(model)

    self.__model = model

    # Save model
    if save:
      self.__save_model(trainer)

  def __inject_nans(self, pred, valid_mask):

    """
    Fill array with Nans, assume it has appended nans already to account
    for past_window size
    """

    idx_to_fill = [e for e, x in enumerate(valid_mask) if x]

    # Count appended nan and remove them
    nb_appended = np.isnan(pred).sum()
    pred = pred[~np.isnan(pred)]

    new_pred = np.full(len(valid_mask), np.nan)

    for p, idx in zip(pred, idx_to_fill):
      new_pred[idx] = p

    # Restore nan
    new_pred = np.append([np.nan] * nb_appended, new_pred)
    return new_pred

  def __predict(self, val_data):

    past_window = self.__context.model_config()['past_window']

    # Generate Final Prediction
    pred = self.__model.eval()(*(val_data[0], val_data[1])).detach()
    pred = pred.reshape(pred.size(0)).numpy()
    pred = np.append([np.nan] * (past_window - 1), pred).flatten()

    return pred, None

  def __predict_mcdropout(self, val_data, n_samples=20):

    past_window = self.__context.model_config()['past_window']

    mean_pred, std_pred = self.__model.sample_predict(val_data[0], val_data[1],
                                                      n_samples)

    mean_pred = mean_pred.reshape(mean_pred.size(0)).numpy()
    mean_pred = np.append([np.nan] * (past_window - 1), mean_pred).flatten()

    std_pred = std_pred.reshape(std_pred.size(0)).numpy()
    std_pred = np.append([np.nan] * (past_window - 1), std_pred).flatten()

    return mean_pred, std_pred

  def fit_predict_pipeline(self, save_model=True, mc_dropout=False,
                           nb_samples=10):

    train_data, val_data, const_cols, var_cols, mask = self.__split_train_val()

    self.__train_model(var_cols, const_cols, train_data, val_data, save_model)

    # Compute pred and inject nans where input was invalid
    if mc_dropout:
      pred, std = self.__predict_mcdropout(val_data, n_samples=nb_samples)
    else:
      pred, std = self.__predict(val_data)

    # Inject nans in pred
    pred = self.__inject_nans(pred, mask)
    if mc_dropout:
      std = self.__inject_nans(std, mask)

    return pred, std
