import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace
from classes.context import Context


################## Model Definition ##################

def MC_dropout(act_vec, p=0.5, mask=True):
  return F.dropout(act_vec, p=p, training=mask, inplace=True)


class HybridLSTM(pl.LightningModule):

  def __init__(self, context: Context, hparams):

    # Setting the seed for this file
    seed_everything(context.model_config()['seed'])

    super().__init__()

    if isinstance(hparams, dict):
      hparams = Namespace(**hparams)
    self.hparams = hparams

    self.hidden_sizes = context.model_config()['hidden_size']
    self.dropout = context.model_config()['dropout']
    self.lr = context.model_config()['default_lr']
    self.batch_size = context.model_config()['default_batch_size']

    self.lstm_1 = nn.LSTM(input_size=len(self.hparams.var_cols),
                          hidden_size=self.hidden_sizes[0],
                          num_layers=1,
                          batch_first=True).double()

    self.linear_1 = nn.Linear(len(self.hparams.const_cols),
                              self.hidden_sizes[1]).double()
    self.linear_2 = torch.nn.ReLU()

    self.mixed_1 = nn.Linear(self.hidden_sizes[0] + self.hidden_sizes[1],
                             self.hidden_sizes[2]).double()
    self.mixed_2 = torch.nn.ReLU()

    self.mixed_3 = nn.Linear(self.hidden_sizes[2], 1).double()
    self.mixed_4 = torch.nn.ReLU()


  def create_dataloaders(self, train_data, val_data):

    # Register data
    self.train_data = train_data
    self.val_data = val_data

  def forward(self, x_mlp, x_lstm, sample=False):
    # if training or sampling, mc dropout will apply random binary mask
    # Otherwise, for regular test set evaluation, we can just scale activations
    mask = self.training or sample

    # x_mlp, x_lstm = x
    # x = (x_mlp, x_lstm)

    x1, _ = self.lstm_1(x_lstm)
    x1 = x1[:, x1.size(1) - 1, :].clone()
    x1 = MC_dropout(x1, p=self.dropout, mask=mask)

    x2 = self.linear_1(x_mlp)
    x2 = self.linear_2(x2)
    x2 = MC_dropout(x2, p=self.dropout, mask=mask)

    x = torch.cat((x1, x2), dim=1)

    x = self.mixed_1(x)
    x = self.mixed_2(x)
    x = MC_dropout(x, p=self.dropout, mask=mask)

    x = self.mixed_3(x)
    x = self.mixed_4(x)

    return x

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)

  def train_dataloader(self):
    dataset = HybridDataset(self.train_data)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=16,
                      shuffle=True)

  def val_dataloader(self):
    dataset = HybridDataset(self.val_data)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=1)

  def training_step(self, batch, batch_nb):
    x_mlp, x_lstm, y = batch
    pred = self(x_mlp, x_lstm)
    pred = pred.reshape(pred.size(0))
    loss = nn.functional.mse_loss(pred, y)
    self.log('train_loss', loss)
    return {'loss': loss}

  def validation_step(self, batch, batch_nb):
    x_mlp, x_lstm, y = batch
    pred = self(x_mlp, x_lstm)
    pred = pred.reshape(pred.size(0))
    loss = nn.functional.mse_loss(pred, y)
    self.log('val_loss', loss)
    return {'val_loss': loss}

  def validation_epoch_end(self, outputs):
    val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
    self.log('val_loss', val_loss_mean.item())
    print('.', end='')

  def sample_predict(self, x_const, x_var, Nsamples):
    # Vector of predictions
    predictions = x_const.data.new(Nsamples, x_const.shape[0], 1)

    for i in range(Nsamples):
      y = self.forward(x_const, x_var, sample=True)
      predictions[i] = y

    mean_pred = predictions.mean(axis=0).detach()
    std_pred = predictions.std(axis=0).detach()

    return mean_pred, std_pred


################## Data-related functions and classes ##################

class HybridDataset(Dataset):
  def __init__(self, data):
    self.X_mlp, self.X_lstm, self.y = data

  def __getitem__(self, index):
    return self.X_mlp[index], self.X_lstm[index], self.y[index]

  def __len__(self):
    return self.y.size()[0]
