import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        #trạng thái adam
        self.v = None   #trung bình gradient bậc 1
        self.s = None   #trung bình bình phương gradient bậc 2
        self.t = 0      

    #khởi tạo trạng thái m, v với kích thước của trọng số theta
    def _init_states(self, shape: tuple): 
        self.v = np.zeros(shape)
        self.s = np.zeros(shape)

    #cap nhat trong so theta
    def update(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self._init_states(theta.shape)
        
        self.t += 1 #tang mot time-step sau moi lan cap nhat mini-batch

        #buoc 1: tính toán trung bình gradient bac 1 (momemtum bac 1)
        self.v = self.beta1 * self.v + (1.0 - self.beta1) * grad
        #tính toán trung bình bình phương gradient (momemtum bac 2)
        self.s = self.beta2 * self.s + (1.0 - self.beta2)* (grad ** 2)
        #buoc 2: hieu chinh chech (bias)
        v_hat = self.v / (1.0 - self.beta1 ** self.t)
        s_hat = self.s / (1.0 - self.beta2 ** self.t)

        #buoc 3: cap nhat trong so 
        theta_new = theta - self.lr * v_hat / (np.sqrt(s_hat) + self.epsilon)
        return theta_new
