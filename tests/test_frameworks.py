import numpy as np
import torch
from src.frameworks.pytorch_cnn import PyTorchCNN
from src.frameworks.tensorflow_cnn import build_tensorflow_cnn


def test_pytorch_cnn_forward():
    model = PyTorchCNN(num_classes=3)
    batch = torch.randn(8, 3, 32, 32)
    output = model(batch)
    assert output.shape == (8, 3)


def test_tensorflow_cnn_forward():
    model = build_tensorflow_cnn(num_classes=3)
    batch = np.random.randn(8, 32, 32, 3).astype(np.float32)
    output = model(batch, training=False)
    assert output.shape == (8, 3)
