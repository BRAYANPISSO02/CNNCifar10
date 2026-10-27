import argparse
from src.data.dataset import DEFAULT_TARGET_CLASSES, load_filtered_cifar10
from src.from_scratch.activations import ReLU
from src.from_scratch.layers import Dense
from src.from_scratch.network import NeuralNetwork
from src.from_scratch.optimizers import Adam, SGDMomentum
from src.utils.metrics import plot_training_history, print_classification_report


def main():
    parser = argparse.ArgumentParser(description="Train Neural Network From Scratch (NumPy)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--plot", type=str, default="scratch_curves.png")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(" TRAINING NEURAL NETWORK FROM SCRATCH (NUMPY)")
    print("=" * 50)

    (X_train, y_train), (X_test, y_test), class_names = load_filtered_cifar10(
        target_classes=DEFAULT_TARGET_CLASSES, flatten=True, normalize=True
    )
    print(f"Dataset loaded: {len(class_names)} classes {class_names}")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    model = NeuralNetwork(
        [
            Dense(in_features=3072, out_features=256, init_method="he"),
            ReLU(),
            Dense(in_features=256, out_features=64, init_method="he"),
            ReLU(),
            Dense(in_features=64, out_features=len(class_names), init_method="xavier"),
        ]
    )

    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = SGDMomentum(lr=args.lr, momentum=0.9, weight_decay=1e-4)

    history = model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=optimizer,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = model.predict(X_test)
    print_classification_report(y_test, y_pred, class_names)

    if args.plot:
        plot_training_history(history, title="Scratch MLP (NumPy)", save_path=args.plot)


if __name__ == "__main__":
    main()
