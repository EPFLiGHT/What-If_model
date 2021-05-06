import numpy as np
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from classes.ddpg.actor import Actor
from classes.ddpg.critic import Critic
from classes.ddpg.memory import Memory
from classes.ddpg.ou_noise import OUNoise


class DDPGAgent:
    def __init__(self, state_space, action_space, actor_learning_rate=3e-4, critic_learning_rate=3e-4,
                 gamma=0.99, tau=0.005, max_memory_size=6000000, batch_size=64, seed=42, weight_decay=1e-4,
                 epsilon_decay=80000):
        # Params
        # self.num_states = env.observation_space.shape[0]
        # self.num_actions = env.action_space.shape[0]
        self.seed(seed)

        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Linear decay rate of exploration policy
        self.depsilon = 1.0 / epsilon_decay
        # Initial exploration rate
        self.epsilon = 1.0

        # Random noise
        self.noise = OUNoise(size=self.action_space)

        # Networks
        self.actor = Actor(self.state_space, self.action_space).double()
        self.actor_target = Actor(self.state_space, self.action_space).double()
        self.critic = Critic(self.state_space, self.action_space).double()
        self.critic_target = Critic(self.state_space, self.action_space).double()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)

    def reset(self):
        self.noise.reset()

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def random_action(self):
        action = np.random.uniform(-1., 1., self.action_space)
        return action

    def select_action(self, s_t, decay_epsilon=True):
        with torch.no_grad():
            state = Variable(torch.from_numpy(s_t).double().unsqueeze(0))
            action = self.actor(state)
            action = action.numpy()[0]

        action_prev = np.copy(action)
        n = max(self.epsilon, 0) * self.noise.evolve_state()
        action += n
        action = np.clip(action, -1., 1.)

        #print(action_prev, action, n)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        return action

    def select_target_action(self, s_t):
        return self.actor_target(s_t)

    def update(self):
        states, actions, rewards, next_states, done_batch = self.memory.sample(self.batch_size)
        states = torch.DoubleTensor(states)
        actions = torch.DoubleTensor(actions)
        rewards = torch.DoubleTensor(rewards)
        next_states = torch.DoubleTensor(next_states)
        done_batch = torch.IntTensor(done_batch).unsqueeze(1)

        with torch.no_grad():
            next_actions = torch.DoubleTensor(self.select_target_action(next_states))
            next_Q = self.critic_target(next_states, next_actions)
            Qprime = rewards + (1.0 - done_batch) * self.gamma * next_Q
        
        # Critic loss
        self.critic_optimizer.zero_grad()

        Qvals = self.critic(states, actions)
        critic_loss = self.critic_criterion(Qvals, Qprime)

        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        self.actor_optimizer.zero_grad()

        policy_loss = -self.critic(states, self.actor(states)).mean()

        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
