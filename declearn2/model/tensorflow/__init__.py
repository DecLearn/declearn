# coding: utf-8

"""Tensorflow models interfacing tools.

This submodule provides with a generic interface to wrap up
any TensorFlow `keras.Model` instance that is to be trained
through gradient descent.

This module exposes:
* TensorflowModel: Model subclass to wrap tensorflow.keras.Model objects
* TensorflowVector: Vector subclass to wrap tensorflow.Tensor objects
"""

from ._vector import TensorflowVector
from ._model import TensorflowModel
