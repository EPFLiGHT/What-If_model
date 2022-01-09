import numpy as np


class OUNoise(object):
  def __init__(self, space_low=-1, space_high=1, size=1, dt=1e-2, mu=0.0, theta=0.15, sigma=1): #sigma = 0.2
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.size = size
    self.low = space_low
    self.dt = dt
    self.high = space_high

    self.state = 0
    self.reset()

  def reset(self):
    self.state = np.ones(self.size) * self.mu

  def evolve_state(self):
    x = self.state
    # print(np.random.normal(size=self.size))
    dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
    self.state = x + dx

    return self.state
