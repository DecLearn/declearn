"""Simple TensorFlow-backed CNN model for the MNIST quickrun example."""

import tensorflow as tf

from declearn.model.tensorflow import TensorflowModel

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

# This needs to be called "model"; otherwise, a different name must be
# specified via the experiment's TOML configuration file.
model = TensorflowModel(network, loss="sparse_categorical_crossentropy")
