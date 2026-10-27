import numpy as np


class Optimizer:
    def update(self, layer):
        raise NotImplementedError


class SGDMomentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                "v_W": np.zeros_like(layer.W),
                "v_b": np.zeros_like(layer.b),
            }

        v = self.velocities[layer_id]

        grad_W = layer.grad_W + self.weight_decay * layer.W
        grad_b = layer.grad_b

        v["v_W"] = self.momentum * v["v_W"] + self.lr * grad_W
        v["v_b"] = self.momentum * v["v_b"] + self.lr * grad_b

        layer.W -= v["v_W"]
        layer.b -= v["v_b"]


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        self.t = 0

    def step_epoch(self):
        self.t += 1

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.state:
            self.state[layer_id] = {
                "m_W": np.zeros_like(layer.W),
                "v_W": np.zeros_like(layer.W),
                "m_b": np.zeros_like(layer.b),
                "v_b": np.zeros_like(layer.b),
            }

        s = self.state[layer_id]
        t = max(1, self.t)

        grad_W = layer.grad_W + self.weight_decay * layer.W
        grad_b = layer.grad_b

        # First moment
        s["m_W"] = self.beta1 * s["m_W"] + (1 - self.beta1) * grad_W
        s["m_b"] = self.beta1 * s["m_b"] + (1 - self.beta1) * grad_b

        # Second moment
        s["v_W"] = self.beta2 * s["v_W"] + (1 - self.beta2) * (grad_W**2)
        s["v_b"] = self.beta2 * s["v_b"] + (1 - self.beta2) * (grad_b**2)

        # Bias correction
        m_W_hat = s["m_W"] / (1 - self.beta1**t)
        m_b_hat = s["m_b"] / (1 - self.beta1**t)
        v_W_hat = s["v_W"] / (1 - self.beta2**t)
        v_b_hat = s["v_b"] / (1 - self.beta2**t)

        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
