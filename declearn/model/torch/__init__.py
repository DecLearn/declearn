# coding: utf-8

"""Tensorflow models interfacing tools.

This submodule provides with a generic interface to wrap up
any PyTorch `nn.Module` instance that is to be trained with
gradient descent.

This module exposes:
* TorchModel: Model subclass to wrap torch.nn.Module objects
* TorchVector: Vector subclass to wrap torch.Tensor objects
"""

from ._vector import TorchVector
from ._model import TorchModel
