"""Wrapping a torch model"""

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from declearn.model.tensorflow import TensorflowModel
from declearn.model.torch import TorchModel


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
MyCustomModel = TensorflowModel(model, loss="sparse_categorical_crossentropy")
