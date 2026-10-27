import tensorflow as tf
from tensorflow.keras import layers, models


def build_tensorflow_cnn(num_classes: int = 3, input_shape: tuple = (32, 32, 3)) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes),
        ]
    )
    return model


def train_tensorflow_model(
    model: models.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 15,
    lr: float = 0.001,
    verbose: int = 1,
) -> dict[str, list[float]]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    fit_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=verbose,
    )

    history = {
        "train_loss": [float(x) for x in fit_history.history["loss"]],
        "train_acc": [float(x) for x in fit_history.history["acc"]],
        "val_loss": [float(x) for x in fit_history.history["val_loss"]],
        "val_acc": [float(x) for x in fit_history.history["val_acc"]],
    }
    return history


def evaluate_tensorflow_model(
    model: models.Model,
    val_ds: tf.data.Dataset,
) -> tuple[float, float]:
    results = model.evaluate(val_ds, verbose=0)
    loss = float(results[0])
    acc = float(results[1])
    return loss, acc
