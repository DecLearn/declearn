# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model abstraction API."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api._vector import Vector
from declearn.typing import Batch
from declearn.utils import DevicePolicy, create_types_registry


__all__ = [
    "Model",
]


@create_types_registry
class Model(metaclass=ABCMeta):
    """Abstract class defining an API to manipulate a ML model.

    A 'Model' is an abstraction that defines a generic interface
    to access a model's parameters and perform operations (such
    as computing gradients or metrics over some data), enabling
    writing algorithms and operations agnostic to the framework
    in which the underlying model is implemented (e.g. PyTorch,
    TensorFlow, Scikit-Learn...).

    Device-placement (i.e. running computations on CPU or GPU)
    is also handled as part of Model classes' backend, mapping
    the generic `declearn.utils.DevicePolicy` parameters to any
    required framework-specific instruction to adequately pick
    the device to use and ensure the wrapped model, input data
    and interfaced computations are placed there.
    """

    def __init__(
        self,
        model: Any,
    ) -> None:
        """Instantiate a Model interface wrapping a 'model' object."""
        self._model = model

    @property
    @abstractmethod
    def device_policy(
        self,
    ) -> DevicePolicy:
        """Return the device-placement policy currently used by this model."""

    @property
    @abstractmethod
    def required_data_info(
        self,
    ) -> Set[str]:
        """List of 'data_info' fields required to initialize this model.

        Note: these fields should match a registered specification
              (see `declearn.data_info` submodule)
        """

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

    @abstractmethod
    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a model from a configuration dict."""

    @abstractmethod
    def get_weights(
        self,
        trainable: bool = False,
    ) -> Vector:
        """Return the model's weights, optionally excluding frozen ones.

        Parameters
        ----------
        trainable: bool, default=False
            Whether to restrict the returned weights to the trainable ones,
            or include those that are frozen, i.e. are not updates as part
            of the training process.

        Returns
        -------
        weights: Vector
            Vector wrapping the named weights data arrays.
            The concrete type of the returned Vector depends on the concrete
            `Model`, and is the same as with `compute_batch_gradients`.
        """

    @abstractmethod
    def set_weights(
        self,
        weights: Vector,
        trainable: bool = False,
    ) -> None:
        """Assign values to the model's weights.

        This method can only be used to update the values of *all*
        model weights, with the optional exception of frozen (i.e.
        non-trainable) ones. It cannot be used to alter the values
        of a subset of weight tensors.

        Parameters
        ----------
        weights: Vector
            Vector wrapping the named data arrays that should replace
            the current weights' values.
            The concrete type of Vector depends on the Model class,
            and matches the `get_weights` method's return type.
        trainable: bool, default=False
            Whether the assigned weights only cover the trainable ones,
            or include those that are frozen, i.e. are not updated as
            part of the training process.

        Raises
        ------
        KeyError:
            If the input weights do not match the expected number and
            names of weight tensors.
        TypeError:
            If the input weights are of unproper concrete Vector type.
        """

    @abstractmethod
    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> Vector:
        """Compute and return gradients computed over a given data batch.

        Compute the average gradients of the model's loss with respect
        to its trainable parameters for the given data batch.
        Optionally clip sample-wise gradients before batch-averaging.

        Parameters
        ----------
        batch: declearn.typing.Batch
            Tuple wrapping input data, (opt.) target values and (opt.)
            sample weights to be applied to the loss function.
        max_norm: float or None, default=None
            Maximum L2-norm of sample-wise gradients, beyond which to
            clip them before computing the batch-average gradients.
            If None, batch-averaged gradients are computed directly,
            which is less costful in computational time and memory.

        Returns
        -------
        gradients: Vector
            Batch-averaged gradients, wrapped into a Vector (using
            a suited Vector subclass depending on the Model class).
        """

    @abstractmethod
    def apply_updates(
        self,
        updates: Vector,
    ) -> None:
        """Apply updates to the model's weights."""

    @abstractmethod
    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute and return model predictions on given inputs.

        This method is designed to return numpy arrays independently
        from the wrapped model's actual framework, for compatibility
        purposed with the `declearn.metrics.Metric` API.

        Note that in most cases, the returned `y_true` and `s_wght`
        are directly taken from the input batch. Their inclusion in
        the inputs and outputs of this method aims to enable using
        some non-standard data-flow schemes, such as that of auto-
        encoder models, that re-use their inputs as labels.

        Parameters
        ----------
        batch: declearn.typing.Batch
            Tuple wrapping input data, (opt.) target values and (opt.)
            sample weights. Note that in general, predictions should
            only be computed from input data - but the API is flexible
            for edge cases, e.g. auto-encoder models, as target labels
            are equal to the input data.

        Returns
        -------
        y_true: np.ndarray
            Ground-truth labels, to which predictions are aligned
            and should be compared for loss (and other evaluation
            metrics) computation.
        y_pred: np.ndarray
            Output model predictions (scores or labels), wrapped as
            a (>=1)-d numpy array, batched along the first axis.
        s_wght: np.ndarray or None
            Optional sample weights to be used to weight metrics.
        """

    @abstractmethod
    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute the model's sample-wise loss from labels and predictions.

        This method is designed to be used when evaluating the model,
        to compute a sample-wise loss from the predictions output by
        `self.compute_batch_predictions`.

        It may further be wrapped as an ad-hoc samples-averaged Metric
        instance so as to mutualize the inference computations between
        the loss's and other evaluation metrics' computation.

        Parameters
        ----------
        y_true: np.ndarray
            Target values or labels, wrapped as a (>=1)-d numpy array,
            the first axis of which is the batching one.
        y_pred: np.ndarray
            Predicted values or scores, as a (>=1)-d numpy array aligned
            with the `y_true` one.

        Returns
        -------
        s_loss: np.ndarray
            Sample-wise loss values, as a 1-d numpy array.
        """

    @abstractmethod
    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        """Update the device-placement policy of this model.

        This method is designed to be called after a change in the global
        device-placement policy (e.g. to disable using a GPU, or move to
        a specific one), so as to place pre-existing Model instances and
        avoid policy inconsistencies that might cause repeated memory or
        runtime costs from moving data or weights around each time they
        are used. You should otherwise not worry about a Model's device-
        placement, as it is handled at instantiation based on the global
        device policy (see `declearn.utils.set_device_policy`).

        Parameters
        ----------
        policy: DevicePolicy or None, default=None
            Optional DevicePolicy dataclass instance to be used.
            If None, use the global device policy, accessed via
            `declearn.utils.get_device_policy`.
        """
