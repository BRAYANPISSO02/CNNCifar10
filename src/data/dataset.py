import os
import pickle
import tarfile
import urllib.request
import numpy as np

CIFAR10_URLS = [
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_TARGET_CLASSES = [0, 1, 2]


def download_and_extract_cifar10(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isdir(cifar_dir):
        return cifar_dir

    archive_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(archive_path):
        success = False
        for url in CIFAR10_URLS:
            try:
                print(f"Downloading CIFAR-10 from {url}...")
                urllib.request.urlretrieve(url, archive_path)
                success = True
                print("Download complete.")
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}. Retrying with next mirror...")
        if not success:
            raise RuntimeError("Unable to download CIFAR-10 from any mirror.")

    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")
    return cifar_dir


def _load_batch(file_path: str):
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
        data = entry[b"data"]
        labels = entry[b"labels"]
        # CIFAR-10 stored as (10000, 3072) where 3072 is 3x32x32 (R, G, B channels)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
        labels = np.array(labels, dtype=np.int64)
        return data, labels


def load_cifar10_raw(data_dir: str = "data"):
    cifar_dir = download_and_extract_cifar10(data_dir)

    train_data = []
    train_labels = []
    for batch_idx in range(1, 6):
        batch_file = os.path.join(cifar_dir, f"data_batch_{batch_idx}")
        d, l = _load_batch(batch_file)
        train_data.append(d)
        train_labels.append(l)

    X_train = np.concatenate(train_data, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    test_file = os.path.join(cifar_dir, "test_batch")
    X_test, y_test = _load_batch(test_file)

    return (X_train, y_train), (X_test, y_test)


def filter_classes(X: np.ndarray, y: np.ndarray, target_classes: list[int]):
    mask = np.isin(y, target_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Remap selected classes to contiguous range [0, len(target_classes) - 1]
    class_map = {orig_cls: new_idx for new_idx, orig_cls in enumerate(target_classes)}
    y_remapped = np.array([class_map[val] for val in y_filtered], dtype=np.int64)

    return X_filtered, y_remapped


def load_filtered_cifar10(
    data_dir: str = "data",
    target_classes: list[int] = DEFAULT_TARGET_CLASSES,
    flatten: bool = False,
    normalize: bool = True,
):
    (X_train, y_train), (X_test, y_test) = load_cifar10_raw(data_dir)
    X_train, y_train = filter_classes(X_train, y_train, target_classes)
    X_test, y_test = filter_classes(X_test, y_test, target_classes)

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    class_names = [CIFAR10_CLASSES[c] for c in target_classes]
    return (X_train, y_train), (X_test, y_test), class_names


def get_pytorch_dataloaders(
    data_dir: str = "data",
    target_classes: list[int] = DEFAULT_TARGET_CLASSES,
    batch_size: int = 64,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    (X_train, y_train), (X_test, y_test), class_names = load_filtered_cifar10(
        data_dir=data_dir, target_classes=target_classes, flatten=False, normalize=True
    )

    # PyTorch expects channels-first: (N, C, H, W)
    X_train_t = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, class_names


def get_tf_datasets(
    data_dir: str = "data",
    target_classes: list[int] = DEFAULT_TARGET_CLASSES,
    batch_size: int = 64,
):
    import tensorflow as tf

    (X_train, y_train), (X_test, y_test), class_names = load_filtered_cifar10(
        data_dir=data_dir, target_classes=target_classes, flatten=False, normalize=True
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds, class_names
