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
from declearn.model._utils import raise_on_stringsets_mismatch
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type


__all__ = [
    "TensorflowModel",
]


@register_type(name="TensorflowModel", group="Model")
class TensorflowModel(Model):
    """Model wrapper for TensorFlow Model instances.

    This `Model` subclass is designed to wrap a `tf.keras.Model` instance
    to be trained federatively.

    Notes regarding device management (CPU, GPU, etc.):
    * By default, tensorflow places data and operations on GPU whenever one
      is available.
    * Our `TensorflowModel` instead consults the device-placement policy (via
      `declearn.utils.get_device_policy`), places the wrapped keras model's
      weights there, and runs computations defined under public methods in
      a `tensorflow.device` context, to enforce that policy.
    * Note that there is no guarantee that calling a private method directly
      will result in abiding by that policy. Hence, be careful when writing
      custom code, and use your own context managers to get guarantees.
    * Note that if the global device-placement policy is updated, this will
      only be propagated to existing instances by manually calling their
      `update_device_policy` method.
    * You may consult the device policy currently enforced by a TensorflowModel
      instance by accessing its `device_policy` property.
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
        # Compile the wrapped model and retain compilation arguments.
        with tf.device(self._device):
            kwargs.update({"loss": loss, "metrics": metrics})
            self._model.compile(**kwargs)
            self._kwargs = kwargs

    @property
    def device_policy(
        self,
    ) -> DevicePolicy:
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
        trainable: bool = False,
    ) -> TensorflowVector:
        variables = (
            self._model.trainable_weights if trainable else self._model.weights
        )
        return TensorflowVector({var.name: var.value() for var in variables})

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: TensorflowVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, TensorflowVector):
            raise TypeError(
                "TensorflowModel requires TensorflowVector weights."
            )
        self._verify_weights_compatibility(weights, trainable=trainable)
        variables = {var.name: var for var in self._model.weights}
        with tf.device(self._device):
            for name, value in weights.coefs.items():
                variables[name].assign(value, read_value=False)

    def _verify_weights_compatibility(
        self,
        vector: TensorflowVector,
        trainable: bool = False,
    ) -> None:
        """Verify that a vector has the same names as the model's weights.

        Parameters
        ----------
        vector: TensorflowVector
            Vector wrapping weight-related coefficients (e.g. weight
            values or gradient-based updates).
        trainable: bool, default=False
            Whether to restrict the comparision to the model's trainable
            weights rather than to all of its weights.

        Raises
        ------
        KeyError:
            In case some expected keys are missing, or additional keys
            are present. Be verbose about the identified mismatch(es).
        """
        variables = (
            self._model.trainable_weights if trainable else self._model.weights
        )
        raise_on_stringsets_mismatch(
            received=set(vector.coefs),
            expected={var.name for var in variables},
            context="model weights",
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
        self._verify_weights_compatibility(updates, trainable=True)
        with tf.device(self._device):
            for var in self._model.trainable_weights:
                updt = updates.coefs[var.name]
                if isinstance(updt, tf.IndexedSlices):
                    var.scatter_add(updt)
                else:
                    var.assign_add(updt, read_value=False)

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

    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        # Select the device to use based on the provided or global policy.
        if policy is None:
            policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # When needed, re-create the model to force moving it to the device.
        if self._device is not device:
            self._device = device
            self._model = move_layer_to_device(self._model, self._device)
