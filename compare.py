import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.dataset import (
    DEFAULT_TARGET_CLASSES,
    get_pytorch_dataloaders,
    get_tf_datasets,
    load_filtered_cifar10,
)
from src.frameworks.pytorch_cnn import PyTorchCNN, evaluate_pytorch_model, train_pytorch_model
from src.frameworks.tensorflow_cnn import (
    build_tensorflow_cnn,
    evaluate_tensorflow_model,
    train_tensorflow_model,
)
from src.from_scratch.activations import ReLU
from src.from_scratch.layers import Dense
from src.from_scratch.network import NeuralNetwork
from src.from_scratch.optimizers import Adam as ScratchAdam


def count_parameters(model, framework: str) -> int:
    if framework == "scratch":
        return sum(layer.W.size + layer.b.size for layer in model.layers if isinstance(layer, Dense))
    elif framework == "pytorch":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif framework == "tensorflow":
        return int(model.count_params())
    return 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Scratch vs PyTorch vs TensorFlow")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    results = []

    # 1. Scratch MLP
    print("\n[1/3] Benchmarking From-Scratch Network (NumPy)...")
    (X_train_s, y_train_s), (X_test_s, y_test_s), class_names = load_filtered_cifar10(
        target_classes=DEFAULT_TARGET_CLASSES, flatten=True, normalize=True
    )
    scratch_model = NeuralNetwork(
        [
            Dense(3072, 256, init_method="he"),
            ReLU(),
            Dense(256, 64, init_method="he"),
            ReLU(),
            Dense(64, len(class_names), init_method="xavier"),
        ]
    )
    scratch_params = count_parameters(scratch_model, "scratch")

    t0 = time.time()
    scratch_history = scratch_model.fit(
        X_train_s,
        y_train_s,
        X_test_s,
        y_test_s,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=ScratchAdam(lr=args.lr),
        verbose=False,
    )
    scratch_time = time.time() - t0
    scratch_loss, scratch_acc = scratch_model.evaluate(X_test_s, y_test_s)
    results.append({
        "Model": "Scratch (NumPy MLP)",
        "Params": scratch_params,
        "Time (s)": round(scratch_time, 2),
        "Test Loss": round(scratch_loss, 4),
        "Test Acc (%)": round(scratch_acc * 100, 2),
    })

    # 2. PyTorch CNN
    print("\n[2/3] Benchmarking PyTorch CNN...")
    pt_train_loader, pt_test_loader, _ = get_pytorch_dataloaders(
        target_classes=DEFAULT_TARGET_CLASSES, batch_size=args.batch_size
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_model = PyTorchCNN(num_classes=len(class_names))
    pt_params = count_parameters(pt_model, "pytorch")

    t0 = time.time()
    pt_history = train_pytorch_model(
        pt_model,
        pt_train_loader,
        pt_test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        verbose=False,
    )
    pt_time = time.time() - t0
    pt_loss, pt_acc = evaluate_pytorch_model(pt_model, pt_test_loader, device=device)
    results.append({
        "Model": f"PyTorch CNN ({device.upper()})",
        "Params": pt_params,
        "Time (s)": round(pt_time, 2),
        "Test Loss": round(pt_loss, 4),
        "Test Acc (%)": round(pt_acc * 100, 2),
    })

    # 3. TensorFlow CNN
    print("\n[3/3] Benchmarking TensorFlow CNN...")
    tf_train_ds, tf_test_ds, _ = get_tf_datasets(
        target_classes=DEFAULT_TARGET_CLASSES, batch_size=args.batch_size
    )
    tf_model = build_tensorflow_cnn(num_classes=len(class_names))
    tf_params = count_parameters(tf_model, "tensorflow")

    t0 = time.time()
    tf_history = train_tensorflow_model(
        tf_model,
        tf_train_ds,
        tf_test_ds,
        epochs=args.epochs,
        lr=args.lr,
        verbose=0,
    )
    tf_time = time.time() - t0
    tf_loss, tf_acc = evaluate_tensorflow_model(tf_model, tf_test_ds)
    results.append({
        "Model": "TensorFlow CNN (Keras)",
        "Params": tf_params,
        "Time (s)": round(tf_time, 2),
        "Test Loss": round(tf_loss, 4),
        "Test Acc (%)": round(tf_acc * 100, 2),
    })

    # Display comparison table
    print("\n" + "=" * 75)
    print(f"{'Model':<26} {'Params':<12} {'Time (s)':<12} {'Test Loss':<12} {'Test Acc (%)':<12}")
    print("-" * 75)
    for r in results:
        print(f"{r['Model']:<26} {r['Params']:<12,d} {r['Time (s)']:<12.2f} {r['Test Loss']:<12.4f} {r['Test Acc (%)']:<12.2f}")
    print("=" * 75 + "\n")

    # Plot Comparison Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, args.epochs + 1)

    # Loss comparison
    ax1.plot(epochs_range, scratch_history["val_loss"], "o-", label="Scratch Val Loss")
    ax1.plot(epochs_range, pt_history["val_loss"], "s-", label="PyTorch Val Loss")
    ax1.plot(epochs_range, tf_history["val_loss"], "^-", label="TensorFlow Val Loss")
    ax1.set_title("Validation Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy comparison
    ax2.plot(epochs_range, [x * 100 for x in scratch_history["val_acc"]], "o-", label="Scratch Val Acc")
    ax2.plot(epochs_range, [x * 100 for x in pt_history["val_acc"]], "s-", label="PyTorch Val Acc")
    ax2.plot(epochs_range, [x * 100 for x in tf_history["val_acc"]], "^-", label="TensorFlow Val Acc")
    ax2.set_title("Validation Accuracy (%) Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=150)
    print("Benchmark plot saved to: benchmark_comparison.png")


if __name__ == "__main__":
    main()
