from torch import nn
from classes.ddpg.utility import WEIGHTS_FINAL_INIT, BIAS_FINAL_INIT, fan_in_uniform_init


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=256, init=True):  # hidden1=256, hidden2=128
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(nb_states, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)

        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()

        if init:
            # Weight Init
            fan_in_uniform_init(self.fc1.weight)
            fan_in_uniform_init(self.fc1.bias)

            fan_in_uniform_init(self.fc2.weight)
            fan_in_uniform_init(self.fc2.bias)

            nn.init.uniform_(self.fc3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
            nn.init.uniform_(self.fc3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)

        out = self.fc3(out)
        # out = self.tanh(out)
        out = self.softsign(out)
        return out
