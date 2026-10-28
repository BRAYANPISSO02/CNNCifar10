import argparse
import numpy as np
import torch
from src.data.dataset import DEFAULT_TARGET_CLASSES, get_pytorch_dataloaders
from src.frameworks.pytorch_cnn import (
    PyTorchCNN,
    evaluate_pytorch_model,
    save_pytorch_model,
    train_pytorch_model,
)
from src.utils.metrics import plot_training_history, print_classification_report


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch CNN on CIFAR-10 (3 classes)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", type=str, default="pytorch_curves.png")
    parser.add_argument("--save-weights", type=str, default="checkpoints/pytorch_cnn.pth", help="Path to save trained weights")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(" TRAINING PYTORCH CNN")
    print("=" * 50)

    train_loader, test_loader, class_names = get_pytorch_dataloaders(
        target_classes=DEFAULT_TARGET_CLASSES, batch_size=args.batch_size
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Classes: {class_names}")

    model = PyTorchCNN(num_classes=len(class_names))

    history = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    test_loss, test_acc = evaluate_pytorch_model(model, test_loader, device=device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc * 100:.2f}%")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    print_classification_report(np.array(all_targets), np.array(all_preds), class_names)

    if args.plot:
        plot_training_history(history, title="PyTorch CNN", save_path=args.plot)

    if args.save_weights:
        save_pytorch_model(model, args.save_weights)
        print(f"Weights saved to: {args.save_weights}")


if __name__ == "__main__":
    main()
