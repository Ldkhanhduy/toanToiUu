import numpy as np

class MomentumOptimizerHandMade:
    """
    Công thức:
        v(t) = momentum*v(t-1) - learning_rate*gradients
        theta(t) = theta(t-1) + v(t)
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def __str__(self):
        return (
            f"MomentumOptimizerHandMade("
            f"learning_rate={self.learning_rate}, "
            f"momentum={self.momentum})"
        )

    def reset(self):
        self.velocity = None
    
    def update(self, theta, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

        self.velocity = self.momentum*self.velocity - self.learning_rate*gradients
        theta += self.velocity

        return theta