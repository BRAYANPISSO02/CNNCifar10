import os
import time
import numpy as np
from src.from_scratch.activations import ReLU, SoftmaxCrossEntropy
from src.from_scratch.layers import Dense
from src.from_scratch.optimizers import Adam, Optimizer


class NeuralNetwork:
    def __init__(self, layers: list | None = None):
        self.layers = layers or []
        self.loss_fn = SoftmaxCrossEntropy()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true: np.ndarray):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        preds = np.argmax(self.loss_fn.probs, axis=-1)
        acc = float(np.mean(preds == y))
        return loss, acc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 64,
        optimizer: Optimizer | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        if optimizer is None:
            optimizer = Adam(lr=0.001)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        num_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            if hasattr(optimizer, "step_epoch"):
                optimizer.step_epoch()

            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []
            epoch_correct = 0

            for i in range(0, num_samples, batch_size):
                xb = X_shuffled[i : i + batch_size]
                yb = y_shuffled[i : i + batch_size]

                logits = self.forward(xb)
                loss = self.loss_fn.forward(logits, yb)
                self.backward(yb)

                for layer in self.layers:
                    if isinstance(layer, Dense):
                        optimizer.update(layer)

                epoch_losses.append(loss)
                preds = np.argmax(self.loss_fn.probs, axis=-1)
                epoch_correct += np.sum(preds == yb)

            train_loss = float(np.mean(epoch_losses))
            train_acc = float(epoch_correct / num_samples)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            elapsed = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} - {elapsed:.2f}s - "
                    f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

        return history

    def save_weights(self, file_path: str):
        dirname = os.path.dirname(os.path.abspath(file_path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        weights = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                weights[f"W_{idx}"] = layer.W
                weights[f"b_{idx}"] = layer.b
        np.savez(file_path, **weights)

    def load_weights(self, file_path: str):
        data = np.load(file_path)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.W = data[f"W_{idx}"]
                layer.b = data[f"b_{idx}"]


def build_scratch_mlp(num_classes: int = 3, in_features: int = 3072) -> NeuralNetwork:
    return NeuralNetwork(
        [
            Dense(in_features=in_features, out_features=256, init_method="he"),
            ReLU(),
            Dense(in_features=256, out_features=64, init_method="he"),
            ReLU(),
            Dense(in_features=64, out_features=num_classes, init_method="xavier"),
        ]
    )
