import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int, init_method: str = "he"):
        if init_method == "he":
            std = np.sqrt(2.0 / in_features)
        elif init_method == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            std = 0.01

        self.W = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_W = np.dot(self.input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)
