import numpy as np

class QuantileRegression:
    def __init__(self, tau=0.9, optimizer=None):
        if not (0 < tau < 1):
            raise ValueError('tau must be between 0 and 1')
        
        self.tau = tau
        self.optimizer = optimizer
        self.theta = None

    def predict(self, x):
        return np.dot(x, self.theta)

    def pinball_loss(self, y, y_pred):
        return np.mean(np.where(y >= y_pred, self.tau * (y - y_pred), (1-self.tau) * (y_pred - y)))

    def sub_gradient(self, x, y, y_pred):
        m = x.shape[0]
        error = y - y_pred
        pen = np.where(error >= 0, -self.tau, (1 - self.tau))

        return np.dot(x.T, pen) / m

    def fit(self, x, y, epochs=1000, verbose=100):
        if self.optimizer is None:
            raise ValueError('Optimizer is not specified')
        
        n = x.shape[1]

        self.theta = np.zeros(n)
        loss = []
        
        for i in range(epochs):
            y_pred = self.predict(x)

            gradient = self.sub_gradient(x, y, y_pred)
            self.theta = self.optimizer.update(self.theta, gradient)

            cur_loss = self.pinball_loss(y, y_pred)
            loss.append(cur_loss)

            if i % verbose == 0 or i == epochs - 1:
                print(f"Epoch {i}/{epochs}, Loss: {cur_loss:.4f}")
        
        return loss