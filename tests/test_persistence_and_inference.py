import numpy as np
import pytest
import torch
from PIL import Image
from predict import (
    load_and_preprocess_image,
    predict_pytorch,
    predict_scratch,
    predict_tensorflow,
)
from src.frameworks.pytorch_cnn import PyTorchCNN, load_pytorch_model, save_pytorch_model
from src.frameworks.tensorflow_cnn import (
    build_tensorflow_cnn,
    load_tensorflow_model,
    save_tensorflow_model,
)
from src.from_scratch.network import build_scratch_mlp


def test_pytorch_save_and_load(tmp_path):
    save_file = str(tmp_path / "pt_model.pth")
    model = PyTorchCNN(num_classes=3)
    model.eval()
    save_pytorch_model(model, save_file)

    loaded_model = load_pytorch_model(save_file, num_classes=3, device="cpu")

    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        orig_out = model(dummy_input)
        loaded_out = loaded_model(dummy_input)

    torch.testing.assert_close(orig_out, loaded_out)


def test_tensorflow_save_and_load(tmp_path):
    save_file = str(tmp_path / "tf_model.keras")
    model = build_tensorflow_cnn(num_classes=3)
    save_tensorflow_model(model, save_file)

    loaded_model = load_tensorflow_model(save_file, num_classes=3)

    dummy_input = np.random.randn(2, 32, 32, 3).astype(np.float32)
    orig_out = model(dummy_input, training=False).numpy()
    loaded_out = loaded_model(dummy_input, training=False).numpy()

    np.testing.assert_allclose(orig_out, loaded_out, rtol=1e-5, atol=1e-5)


def test_scratch_save_and_load(tmp_path):
    save_file = str(tmp_path / "scratch_model.npz")
    model = build_scratch_mlp(num_classes=3, in_features=3072)
    model.save_weights(save_file)

    loaded_model = build_scratch_mlp(num_classes=3, in_features=3072)
    loaded_model.load_weights(save_file)

    dummy_input = np.random.randn(2, 3072).astype(np.float32)
    orig_out = model.forward(dummy_input)
    loaded_out = loaded_model.forward(dummy_input)

    np.testing.assert_allclose(orig_out, loaded_out, rtol=1e-6, atol=1e-6)


def test_load_and_preprocess_image(tmp_path):
    img_path = str(tmp_path / "sample.jpg")
    img = Image.new("RGB", (128, 96), color=(200, 100, 50))
    img.save(img_path)

    arr = load_and_preprocess_image(img_path)
    assert arr.shape == (32, 32, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() <= arr.max() <= 1.0


def test_predict_functions(tmp_path):
    img_path = str(tmp_path / "test_img.png")
    Image.new("RGB", (64, 64), color=(50, 150, 250)).save(img_path)
    img_arr = load_and_preprocess_image(img_path)
    class_names = ["airplane", "automobile", "bird"]

    # 1. PyTorch
    pt_path = str(tmp_path / "pt.pth")
    pt_model = PyTorchCNN(num_classes=3)
    save_pytorch_model(pt_model, pt_path)
    pt_probs = predict_pytorch(img_arr, pt_path, class_names, device="cpu")
    assert pt_probs.shape == (3,)
    assert pytest.approx(float(pt_probs.sum()), rel=1e-4) == 1.0

    # 2. TensorFlow
    tf_path = str(tmp_path / "tf.keras")
    tf_model = build_tensorflow_cnn(num_classes=3)
    save_tensorflow_model(tf_model, tf_path)
    tf_probs = predict_tensorflow(img_arr, tf_path, class_names)
    assert tf_probs.shape == (3,)
    assert pytest.approx(float(tf_probs.sum()), rel=1e-4) == 1.0

    # 3. Scratch
    scratch_path = str(tmp_path / "scratch.npz")
    scratch_model = build_scratch_mlp(num_classes=3, in_features=3072)
    scratch_model.save_weights(scratch_path)
    scratch_probs = predict_scratch(img_arr, scratch_path, class_names)
    assert scratch_probs.shape == (3,)
    assert pytest.approx(float(scratch_probs.sum()), rel=1e-4) == 1.0
