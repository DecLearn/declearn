# coding: utf-8

"""Model abstraction API."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, Set

from declearn.model.api._vector import Vector
from declearn.model.api._np_vec import NumpyVector
from declearn.typing import Batch
from declearn.utils import create_types_registry


__all__ = [
    "Model",
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

    @property
    @abstractmethod
    def required_data_info(
        self,
    ) -> Set[str]:
        """List of 'data_info' fields required to initialize this model.

        Note: these fields should match a registered specification
              (see `declearn.data_info` submodule)
        """
        return NotImplemented

    @abstractmethod
    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        """Initialize the model based on data specifications.

        Parameters
        ----------
        data_info: dict[str, any]
            Data specifications, presenting values for all fields
            listed under `self.required_data_info`

        Raises
        ------
        KeyError:
            If some fields in `required_data_info` are missing.

        Notes
        -----
        See the `aggregate_data_info` method to derive `data_info`
        from client-wise dict.
        """
        return None

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
    ) -> "Model":
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
        """Compute and return gradients computed over a given data batch.

        Compute the average gradients of the model's loss with respect
        to its trainable parameters for the given data batch.

        Parameters
        ----------
        batch: declearn.typing.Batch
            Tuple wrapping input data, (opt.) target values and (opt.)
            sample weights to be applied to the loss function.

        Returns
        -------
        gradients: Vector
            Batch-averaged gradients, wrapped into a Vector (using
            a suited Vector subclass depending on the Model class).
        """
        return NotImplemented

    @abstractmethod
    def apply_updates(
        self,
        updates: Vector,
    ) -> None:
        """Apply updates to the model's weights."""
        return None

    @abstractmethod
    def compute_loss(
        self,
        dataset: Iterable[Batch],
    ) -> float:
        """Compute the average loss of the model on a given dataset.

        Parameters
        ----------
        dataset: iterable of batches
            Iterable yielding batch structures that are to be unpacked
            into (input_features, target_labels, [sample_weights]).
            If set, sample weights will affect the loss averaging.

        Returns
        -------
        loss: float
            Average value of the model's loss over samples.
        """
        return NotImplemented


create_types_registry("Model", Model)
