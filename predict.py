import argparse
import os
import sys
import numpy as np
from PIL import Image
from src.data.dataset import CIFAR10_CLASSES, DEFAULT_TARGET_CLASSES


DEFAULT_WEIGHTS = {
    "pytorch": "checkpoints/pytorch_cnn.pth",
    "tensorflow": "checkpoints/tensorflow_cnn.keras",
    "scratch": "checkpoints/scratch_mlp.npz",
}


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr


def predict_scratch(image_arr: np.ndarray, weights_path: str, class_names: list[str]) -> np.ndarray:
    from src.from_scratch.network import build_scratch_mlp

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Train the model first with 'python train_scratch.py'."
        )

    model = build_scratch_mlp(num_classes=len(class_names), in_features=3072)
    model.load_weights(weights_path)

    x = image_arr.reshape(1, -1)
    probs = model.predict_proba(x)[0]
    return probs


def predict_pytorch(
    image_arr: np.ndarray,
    weights_path: str,
    class_names: list[str],
    device: str | None = None,
) -> np.ndarray:
    import torch
    from src.frameworks.pytorch_cnn import load_pytorch_model

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Train the model first with 'python train_pytorch.py'."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pytorch_model(weights_path, num_classes=len(class_names), device=device)
    model.eval()

    # Channels-first: (1, 3, 32, 32)
    x = torch.tensor(image_arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def predict_tensorflow(
    image_arr: np.ndarray,
    weights_path: str,
    class_names: list[str],
) -> np.ndarray:
    import tensorflow as tf
    from src.frameworks.tensorflow_cnn import load_tensorflow_model

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Train the model first with 'python train_tensorflow.py'."
        )

    model = load_tensorflow_model(weights_path, num_classes=len(class_names))

    # Channels-last: (1, 32, 32, 3)
    x = np.expand_dims(image_arr, axis=0)

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits).numpy()[0]
    return probs


def main():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 inference on a single image without retraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image file (JPEG, PNG, etc.)")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pytorch", "tensorflow", "scratch"],
        default="pytorch",
        help="Model architecture / framework to use",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to serialized weights file. Defaults to framework default in checkpoints/",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device for PyTorch (cuda or cpu)")
    args = parser.parse_args()

    class_names = [CIFAR10_CLASSES[c] for c in DEFAULT_TARGET_CLASSES]
    weights_path = args.weights or DEFAULT_WEIGHTS[args.model]

    try:
        image_arr = load_and_preprocess_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    print(" CIFAR-10 INFERENCE")
    print("=" * 50)
    print(f"Image:       {args.image}")
    print(f"Model:       {args.model.upper()}")
    print(f"Weights:     {weights_path}")
    print("-" * 50)

    try:
        if args.model == "pytorch":
            probs = predict_pytorch(image_arr, weights_path, class_names, device=args.device)
        elif args.model == "tensorflow":
            probs = predict_tensorflow(image_arr, weights_path, class_names)
        else:
            probs = predict_scratch(image_arr, weights_path, class_names)
    except FileNotFoundError as e:
        print(f"\n[!] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Unexpected error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    print(f"\nTop Prediction: {pred_class.upper()} ({confidence:.2f}% confidence)")
    print("\nClass Probabilities:")
    for idx, name in enumerate(class_names):
        bar = "█" * int(probs[idx] * 30)
        print(f"  [{idx}] {name:<12}: {probs[idx] * 100:6.2f}% | {bar}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
