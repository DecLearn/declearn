# coding: utf-8

"""Haiku models interfacing tools.

This submodule provides with a generic interface to wrap up
any Haiku module instance that is to be trained
through gradient descent.

This module exposes:
* HaikuModel: Model subclass to wrap haiku.Model objects
* JaxNumpyVector: Vector subclass to wrap jax.numpy.ndarray objects
"""

from . import utils
from ._vector import JaxNumpyVector
from ._model import HaikuModel
