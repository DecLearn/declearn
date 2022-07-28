# coding: utf-8

"""Model abstraction API."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from declearn2.model.api._vector import Vector
from declearn2.model.api._np_vec import NumpyVector
from declearn2.typing import Batch


__all__ = [
    'Model',
]


class Model(metaclass=ABCMeta):
    """Abstract class defining an API to manipulate a ML model.

    A 'Model' is an abstraction that defines a generic interface
    to access a model's parameters and perform operations (such
    as computing gradients or metrics over some data), enabling
    writing algorithms and operations agnostic to the framework
    in which the underlying model is implemented (e.g. PyTorch,
    TensorFlow, Scikit-Learn...).
    """

    def __init__(
            self,
            model: Any,
        ) -> None:
        """Instantiate a Model interface wrapping a 'model' object."""
        self._model = model

    @abstractmethod
    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""
        return NotImplemented

    @classmethod
    @abstractmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'Model':
        """Instantiate a model from a configuration dict."""
        return NotImplemented

    @abstractmethod
    def get_weights(
            self,
        ) -> NumpyVector:
        """Return the model's trainable weights."""
        return NotImplemented

    @abstractmethod
    def set_weights(
            self,
            weights: NumpyVector,
        ) -> None:
        """Assign values to the model's trainable weights."""
        return None

    @abstractmethod
    def compute_batch_gradients(
            self,
            batch: Batch,
        ) -> Vector:
        """Compute and return the model's gradients over a data batch."""
        return NotImplemented

    @abstractmethod
    def apply_updates(
            self,
            updates: Vector,
        ) -> None:
        """Apply updates to the model's weights."""
        return None
