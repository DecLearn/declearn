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

"""Model subclass to wrap TensorFlow models."""

import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import tensorflow as tf  # type: ignore
from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model
from declearn.model.tensorflow.utils import (
    build_keras_loss,
    move_layer_to_device,
    select_device,
)
from declearn.model.tensorflow._vector import TensorflowVector
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type


__all__ = [
    "TensorflowModel",
]


@register_type(name="TensorflowModel", group="Model")
class TensorflowModel(Model):
    """Model wrapper for TensorFlow Model instances.

    This `Model` subclass is designed to wrap a `tf.keras.Model`
    instance to be learned federatively.

    Notes regarding device management (CPU, GPU, etc.):
    * By default, tensorflow places data and operations on GPU whenever one
      is available.
    * Instead, `TensorflowModel` consults the device-placement policy (via
      `declearn.utils.get_device_policy`), places the wrapped keras model's
      weights there, and runs computations defined under public methods in
      a `tensorflow.device` context, to enforce that policy.
    * Note that there is no guarantee that calling a private method directly
      will result in abiding by that policy. Hence, be careful when writing
      custom code, and use your own context managers to get guarantees.
    * Note that if the global device-placement policy is updated, this will
      only be propagated to existing instances by manually calling their
      `update_device_policy` method.
    """

    def __init__(
        self,
        model: tf.keras.layers.Layer,
        loss: Union[str, tf.keras.losses.Loss],
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None,
        _from_config: bool = False,
        **kwargs: Any,
    ) -> None:
        """Instantiate a Model interface wrapping a tensorflow.keras model.

        Parameters
        ----------
        model: tf.keras.layers.Layer
            Keras Layer (or Model) instance that defines the model's
            architecture. If a Layer is provided, it will be wrapped
            into a keras Sequential Model.
        loss: tf.keras.losses.Loss or str
            Keras Loss instance, or name of one. If a function (name)
            is provided, it will be converted to a Loss instance, and
            an exception may be raised if that fails.
        metrics: list[str or tf.keras.metrics.Metric] or None
            List of keras Metric instances, or their names. These are
            compiled with the model and computed using the `evaluate`
            method of the returned TensorflowModel instance.
        **kwargs: Any
            Any additional keyword argument to `tf.keras.Model.compile`
            may be passed.
        """
        # Type-check the input Model and wrap it up.
        if not isinstance(model, tf.keras.layers.Layer):
            raise TypeError(
                "'model' should be a tf.keras.layers.Layer instance."
            )
        if not isinstance(model, tf.keras.Model):
            model = tf.keras.Sequential([model])
        super().__init__(model)
        # Ensure the loss is a keras.Loss object and set its reduction to none.
        loss = build_keras_loss(loss, reduction=tf.keras.losses.Reduction.NONE)
        # Select the device where to place computations and move the model.
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        if not _from_config:
            self._model = move_layer_to_device(self._model, self._device)
        # Finalize initialization using the selected device.
        with tf.device(self._device):
            # Compile the wrapped model and retain compilation arguments.
            kwargs.update({"loss": loss, "metrics": metrics})
            self._model.compile(**kwargs)
            self._kwargs = kwargs
            # Instantiate a SGD optimizer to apply updates as-provided.
            self._sgd = tf.keras.optimizers.SGD(learning_rate=1.0)

    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        """Update the device-placement policy of this model.

        Parameters
        ----------
        policy: DevicePolicy or None, default=None
            Optional DevicePolicy dataclass instance to be used.
            If None, use the global device policy, accessed via
            `declearn.utils.get_device_policy`.
        """
        if policy is None:
            policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # When needed, re-create the model to force moving it to the device.
        if self._device is not device:
            self._device = device
            self._model = move_layer_to_device(self._model, self._device)

    @property
    def device_policy(
        self,
    ) -> DevicePolicy:
        """Return the device-placement policy currently used by this model."""
        device = self._device
        try:
            idx = int(device.name.rsplit(":", 1)[-1])
        except ValueError:
            idx = None
        return DevicePolicy(gpu=(device.device_type == "GPU"), idx=idx)

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        return set() if self._model.built else {"input_shape"}

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        if not self._model.built:
            data_info = aggregate_data_info([data_info], {"input_shape"})
            with tf.device(self._device):
                self._model.build(data_info["input_shape"])
        # Warn about frozen weights.
        # similar to TorchModel warning; pylint: disable=duplicate-code
        if len(self._model.trainable_weights) < len(self._model.weights):
            warnings.warn(
                "'TensorflowModel' wraps a model with frozen weights.\n"
                "This is not fully compatible with declearn v2.0.x: the "
                "use of weight decay and/or of a loss-regularization "
                "plug-in in an Optimizer will fail to produce updates "
                "for this model.\n"
                "This issue will be fixed in declearn v2.1.0."
            )
        # pylint: enable=duplicate-code

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = tf.keras.layers.serialize(self._model)  # type: Dict[str, Any]
        kwargs = deepcopy(self._kwargs)
        loss = tf.keras.losses.serialize(kwargs.pop("loss"))
        return {"model": config, "loss": loss, "kwargs": kwargs}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a TensorflowModel from a configuration dict."""
        for key in ("model", "loss", "kwargs"):
            if key not in config.keys():
                raise KeyError(f"Missing key '{key}' in the config dict.")
        # Set up the device policy.
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Deserialize the model and loss keras objects on the device.
        with tf.device(device):
            model = tf.keras.layers.deserialize(config["model"])
            loss = tf.keras.losses.deserialize(config["loss"])
        # Instantiate the TensorflowModel, avoiding device-to-device copies.
        return cls(model, loss, **config["kwargs"], _from_config=True)

    def get_weights(
        self,
    ) -> TensorflowVector:
        # REVISE: only return trainable weights? add flag to select?
        return TensorflowVector(
            {var.name: var.value() for var in self._model.weights}
        )

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: TensorflowVector,
    ) -> None:
        if not isinstance(weights, TensorflowVector):
            raise TypeError(
                "TensorflowModel requires TensorflowVector weights."
            )
        self._verify_weights_compatibility(weights, trainable_only=False)
        variables = {var.name: var for var in self._model.weights}
        with tf.device(self._device):
            for name, value in weights.coefs.items():
                variables[name].assign(value, read_value=False)

    def _verify_weights_compatibility(
        self,
        vector: TensorflowVector,
        trainable_only: bool = False,
    ) -> None:
        """Verify that a vector has the same names as the model's weights.

        Parameters
        ----------
        vector: TensorflowVector
            Vector wrapping weight-related coefficients (e.g. weight
            values or gradient-based updates).
        trainable_only: bool, default=False
            Whether to restrict the comparision to the model's trainable
            weights rather than to all of its weights.

        Raises
        ------
        KeyError:
            In case some expected keys are missing, or additional keys
            are present. Be verbose about the identified mismatch(es).
        """
        # Gather the variables to compare to the input vector.
        if trainable_only:
            weights = self._model.trainable_weights
        else:
            weights = self._model.weights
        variables = {var.name: var for var in weights}
        # Raise a verbose KeyError in case inputs do not match weights.
        if set(vector.coefs).symmetric_difference(variables):
            missing = set(variables).difference(vector.coefs)
            unexpct = set(vector.coefs).difference(variables)
            raise KeyError(
                "Mismatch between input and model weights' names:\n"
                + f"Missing key(s) in inputs: {missing}\n" * bool(missing)
                + f"Unexpected key(s) in inputs: {unexpct}\n" * bool(unexpct)
            )

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> TensorflowVector:
        with tf.device(self._device):
            data = self._unpack_batch(batch)
            if max_norm is None:
                grads = self._compute_batch_gradients(*data)
            else:
                norm = tf.constant(max_norm)
                grads = self._compute_clipped_gradients(*data, norm)
        grads_and_vars = zip(grads, self._model.trainable_weights)
        return TensorflowVector(
            {var.name: grad for grad, var in grads_and_vars}
        )

    def _unpack_batch(
        self,
        batch: Batch,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Unpack and enforce Tensor conversion to an input data batch."""
        # fmt: off
        # Define an array-to-tensor conversion routine.
        def convert(data: Optional[ArrayLike]) -> Optional[tf.Tensor]:
            if (data is None) or tf.is_tensor(data):
                return data
            return tf.convert_to_tensor(data)
        # Apply it to the the batched elements.
        return tf.nest.map_structure(convert, batch)

    @tf.function  # optimize tensorflow runtime
    def _compute_batch_gradients(
        self,
        inputs: tf.Tensor,
        y_true: Optional[tf.Tensor],
        s_wght: Optional[tf.Tensor],
    ) -> List[tf.Tensor]:
        """Compute and return batch-averaged gradients of trainable weights."""
        with tf.GradientTape() as tape:
            y_pred = self._model(inputs, training=True)
            loss = self._model.compute_loss(inputs, y_true, y_pred, s_wght)
            loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, self._model.trainable_weights)
        return grad

    @tf.function  # optimize tensorflow runtime
    def _compute_clipped_gradients(
        self,
        inputs: tf.Tensor,
        y_true: Optional[tf.Tensor],
        s_wght: Optional[tf.Tensor],
        max_norm: Union[tf.Tensor, float],
    ) -> List[tf.Tensor]:
        """Compute and return sample-wise-clipped batch-averaged gradients."""
        grad = self._compute_samplewise_gradients(inputs, y_true)
        if s_wght is None:
            s_wght = tf.cast(1, grad[0].dtype)
        grad = self._clip_and_average_gradients(grad, max_norm, s_wght)
        return grad

    @tf.function  # optimize tensorflow runtime
    def _compute_samplewise_gradients(
        self,
        inputs: tf.Tensor,
        y_true: Optional[tf.Tensor],
    ) -> List[tf.Tensor]:
        """Compute and return sample-wise gradients for a given batch."""
        with tf.GradientTape() as tape:
            y_pred = self._model(inputs, training=True)
            loss = self._model.compute_loss(inputs, y_true, y_pred)
            grad = tape.jacobian(loss, self._model.trainable_weights)
        return grad

    @staticmethod
    @tf.function  # optimize tensorflow runtime
    def _clip_and_average_gradients(
        gradients: List[tf.Tensor],
        max_norm: Union[tf.Tensor, float],
        s_wght: tf.Tensor,
    ) -> List[tf.Tensor]:
        """Clip sample-wise gradients then batch-average them."""
        outp = []  # type: List[tf.Tensor]
        for grad in gradients:
            dims = list(range(1, grad.shape.rank))
            grad = tf.clip_by_norm(grad, max_norm, axes=dims)
            outp.append(tf.reduce_mean(grad * s_wght, axis=0))
        return outp

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: TensorflowVector,
    ) -> None:
        self._verify_weights_compatibility(updates, trainable_only=True)
        with tf.device(self._device):
            # Delegate updates' application to a tensorflow Optimizer.
            values = (-1 * updates).coefs.values()
            zipped = zip(values, self._model.trainable_weights)
            upd_op = self._sgd.apply_gradients(zipped)
            # Ensure ops have been performed before exiting.
            with tf.control_dependencies([upd_op]):
                return None

    def evaluate(
        self,
        dataset: Iterable[Batch],
    ) -> Dict[str, float]:
        """Compute the model's built-in evaluation metrics on a given dataset.

        Parameters
        ----------
        dataset: iterable of batches
            Iterable yielding batch structures that are to be unpacked
            into (input_features, target_labels, [sample_weights]).
            If set, sample weights will affect metrics' averaging.

        Returns
        -------
        metrics: dict[str, float]
            Dictionary associating evaluation metrics' values to their name.
        """
        with tf.device(self._device):
            return self._model.evaluate(dataset, return_dict=True)

    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        with tf.device(self._device):
            inputs, y_true, s_wght = self._unpack_batch(batch)
            if y_true is None:
                raise TypeError(
                    "`TensorflowModel.compute_batch_predictions` received a "
                    "batch with `y_true=None`, which is unsupported. Please "
                    "correct the inputs, or override this method to support "
                    "creating labels from the base inputs."
                )
            y_pred = self._model(inputs, training=False).numpy()
        y_true = y_true.numpy()
        s_wght = s_wght.numpy() if s_wght is not None else s_wght
        return y_true, y_pred, s_wght

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        with tf.device(self._device):
            tns_true = tf.convert_to_tensor(y_true)
            tns_pred = tf.convert_to_tensor(y_pred)
            s_loss = self._model.compute_loss(y=tns_true, y_pred=tns_pred)
        return s_loss.numpy().squeeze()
