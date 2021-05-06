import torch
from torch import nn
from classes.ddpg.utility import WEIGHTS_FINAL_INIT, BIAS_FINAL_INIT, fan_in_uniform_init


class Critic(nn.Module):
  def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=256, init=True):  # hidden1=256, hidden2=128
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(nb_states, hidden1)
    self.ln1 = nn.LayerNorm(hidden1)

    self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
    self.ln2 = nn.LayerNorm(hidden2)

    self.fc3 = nn.Linear(hidden2, 1)
    self.relu = nn.ReLU()

    if init:
      # Weight Init
      fan_in_uniform_init(self.fc1.weight)
      fan_in_uniform_init(self.fc1.bias)

      fan_in_uniform_init(self.fc2.weight)
      fan_in_uniform_init(self.fc2.bias)

      nn.init.uniform_(self.fc3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
      nn.init.uniform_(self.fc3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

  def forward(self, x, a):
    out = self.fc1(x)
    out = self.ln1(out)
    out = self.relu(out)

    # concatenate along columns
    c_in = torch.cat([out, a], len(a.shape) - 1)
    out = self.fc2(c_in)
    out = self.ln2(out)
    out = self.relu(out)

    out = self.fc3(out)

    return out
