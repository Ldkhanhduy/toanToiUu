import numpy as np

class NesterovOptimizer:
    def __init__(self, lr: float=0.001, gamma: float=0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = None

    def update(self, theta, gradient):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.gamma * self.v + gradient
        step = gradient + self.gamma * self.v
        theta_new = theta - self.lr * step

        return theta_new