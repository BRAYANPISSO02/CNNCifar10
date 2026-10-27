import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.input > 0)


class SoftmaxCrossEntropy:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.probs = None
        self.y_true = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        # Subtract max along class axis for numerical stability
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        self.y_true = y_true

        m = logits.shape[0]
        if y_true.ndim == 1:
            correct_log_probs = -np.log(self.probs[np.arange(m), y_true] + self.eps)
        else:
            correct_log_probs = -np.sum(y_true * np.log(self.probs + self.eps), axis=-1)
        return float(np.mean(correct_log_probs))

    def backward(self) -> np.ndarray:
        m = self.probs.shape[0]
        grad = self.probs.copy()
        if self.y_true.ndim == 1:
            grad[np.arange(m), self.y_true] -= 1.0
        else:
            grad -= self.y_true
        return grad / m
