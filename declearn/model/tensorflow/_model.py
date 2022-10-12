# coding: utf-8

"""Model subclass to wrap TensorFlow models."""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import tensorflow as tf  # type: ignore
from numpy.typing import ArrayLike

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model, NumpyVector
from declearn.model.tensorflow._utils import build_keras_loss
from declearn.model.tensorflow._vector import TensorflowVector
from declearn.typing import Batch
from declearn.utils import register_type


@register_type(name="TensorflowModel", group="Model")
class TensorflowModel(Model):
    """Model wrapper for TensorFlow Model instances.

    This `Model` subclass is designed to wrap a `tf.keras.Model`
    instance to be learned federatively.
    """

    def __init__(
        self,
        model: tf.keras.layers.Layer,
        loss: Optional[Union[str, tf.keras.losses.Loss]],
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]],
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
            Any addition keyword argument to `tf.keras.Model.compile`
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
        # Compile the wrapped model and retain compilation arguments.
        kwargs.update({"loss": loss, "metrics": metrics})
        model.compile(**kwargs)
        self._kwargs = kwargs
        # Instantiate a SGD optimizer to apply updates as-provided.
        self._sgd = tf.keras.optimizers.SGD(learning_rate=1.0)

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
    ) -> "TensorflowModel":
        """Instantiate a TensorflowModel from a configuration dict."""
        for key in ("model", "loss", "kwargs"):
            if key not in config.keys():
                raise KeyError(f"Missing key '{key}' in the config dict.")
        model = tf.keras.layers.deserialize(config["model"])
        loss = tf.keras.losses.deserialize(config["loss"])
        return cls(model, loss, **config["kwargs"])

    def get_weights(
        self,
    ) -> NumpyVector:
        return NumpyVector(
            {str(i): arr for i, arr in enumerate(self._model.get_weights())}
        )

    def set_weights(
        self,
        weights: NumpyVector,
    ) -> None:
        self._model.set_weights(list(weights.coefs.values()))

    def compute_batch_gradients(
        self,
        batch: Batch,
    ) -> TensorflowVector:
        data = self._unpack_batch(batch)
        grad = self._compute_batch_gradients(*data)
        return TensorflowVector({str(i): tns for i, tns in enumerate(grad)})

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
        return tf.nest.map_structure(convert, batch)  # type: ignore

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
        return grad  # type: ignore

    def apply_updates(  # type: ignore  # future: revise
        self,
        updates: TensorflowVector,
    ) -> None:
        # Delegate updates' application to a tensorflow Optimizer.
        values = (-1 * updates).coefs.values()
        zipped = zip(values, self._model.trainable_weights)
        upd_op = self._sgd.apply_gradients(zipped)
        # Ensure ops have been performed before exiting.
        with tf.control_dependencies([upd_op]):
            return None

    def compute_loss(
        self,
        dataset: Iterable[Batch],
    ) -> float:
        total = 0.0
        n_btc = 0.0
        for batch in dataset:
            inputs, y_true, s_wght = self._unpack_batch(batch)
            y_pred = self._model(inputs, training=False)
            loss = self._model.compute_loss(inputs, y_true, y_pred, s_wght)
            total += loss.numpy().mean()
            n_btc += 1 if s_wght is None else s_wght.numpy().mean()
        return total / n_btc

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
            Dictionary associating evaluation metrics' values to their
            name.
        """
        return self._model.evaluate(dataset, return_dict=True)  # type: ignore
