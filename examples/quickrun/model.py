"""Wrapping a simple CNN for the MNIST example"""

import tensorflow as tf

from declearn.model.tensorflow import TensorflowModel

stack = [
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, 1, activation="relu"),
    tf.keras.layers.Conv2D(64, 3, 1, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
]
model = tf.keras.models.Sequential(stack)

MyModel = TensorflowModel(model, loss="sparse_categorical_crossentropy")
