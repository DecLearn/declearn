"""Standard MNIST CNN built with TensorFlow Keras."""

import tensorflow as tf  # type: ignore

from declearn.model.api import Model
from declearn.model.tensorflow import TensorflowModel

__all__ = ["build_model"]


def build_model() -> Model:
    """Return a `TensorflowModel` wrapping a small MNIST CNN.

    Expects inputs of shape `(28, 28, 1)` (the `hwc` data layout) with
    integer targets and uses `sparse_categorical_crossentropy`.
    """
    stack = [
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, 1, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
    network = tf.keras.models.Sequential(stack)
    return TensorflowModel(network, loss="sparse_categorical_crossentropy")
