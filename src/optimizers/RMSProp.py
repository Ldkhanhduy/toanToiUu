import numpy as np

class RMSPropOtimizer:
    def __init__(self, lr: float=0.001, beta: float=0.9, epsilon: float=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.v = None

    def update(self, theta, gradient):
        if self.v is None:
            self.v = np.zeros_like(theta)

        self.v = self.beta * self.v + (1 - self.beta) * gradient**2

        theta_new = theta - (self.lr/np.sqrt(self.v + self.epsilon)) * gradient

        return theta_new