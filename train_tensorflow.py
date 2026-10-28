import argparse
import numpy as np
from src.data.dataset import DEFAULT_TARGET_CLASSES, get_tf_datasets
from src.frameworks.tensorflow_cnn import (
    build_tensorflow_cnn,
    evaluate_tensorflow_model,
    save_tensorflow_model,
    train_tensorflow_model,
)
from src.utils.metrics import plot_training_history, print_classification_report


def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow CNN on CIFAR-10 (3 classes)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--plot", type=str, default="tensorflow_curves.png")
    parser.add_argument("--save-weights", type=str, default="checkpoints/tensorflow_cnn.keras", help="Path to save trained weights")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(" TRAINING TENSORFLOW CNN")
    print("=" * 50)

    train_ds, test_ds, class_names = get_tf_datasets(
        target_classes=DEFAULT_TARGET_CLASSES, batch_size=args.batch_size
    )

    model = build_tensorflow_cnn(num_classes=len(class_names))

    history = train_tensorflow_model(
        model=model,
        train_ds=train_ds,
        val_ds=test_ds,
        epochs=args.epochs,
        lr=args.lr,
        verbose=1,
    )

    test_loss, test_acc = evaluate_tensorflow_model(model, test_ds)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc * 100:.2f}%")

    all_preds = []
    all_targets = []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0).argmax(axis=-1)
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

    print_classification_report(np.array(all_targets), np.array(all_preds), class_names)

    if args.plot:
        plot_training_history(history, title="TensorFlow CNN", save_path=args.plot)

    if args.save_weights:
        save_tensorflow_model(model, args.save_weights)
        print(f"Weights saved to: {args.save_weights}")


if __name__ == "__main__":
    main()
