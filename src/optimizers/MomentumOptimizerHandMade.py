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
            self.velocity = [0.0 for _ in theta]

        for i in range(len(theta)):
            self.velocity[i] = (self.momentum*self.velocity[i] - self.learning_rate*gradients[i])
            theta[i] += self.velocity[i]

        return theta