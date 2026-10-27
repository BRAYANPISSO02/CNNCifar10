import matplotlib.pyplot as plt
import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    report = {}

    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

        print(f"{name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10d}")

    total_samples = np.sum(supports)
    macro_acc = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    macro_f1 = np.mean(f1s)

    print("-" * 65)
    print(f"{'Accuracy':<15} {'':<12} {'':<12} {macro_acc:<12.4f} {total_samples:<10d}")
    print(f"{'Macro Avg':<15} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {macro_f1:<12.4f} {total_samples:<10d}\n")

    return report


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training and Validation Curves",
    save_path: str | None = None,
):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--", label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r--", label="Val Acc")
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    plt.close()
