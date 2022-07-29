# coding: utf-8

"""Model subclass to wrap TensorFlow models."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import tensorflow as tf  # type: ignore
from numpy.typing import ArrayLike

from declearn2.model.api import Model, NumpyVector
from declearn2.model.tensorflow._vector import TensorflowVector
from declearn2.typing import Batch


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
            **kwargs: Any
        ) -> None:
        """Instantiate a Model interface wrapping a 'model' object."""
        # Type-check the input Model and wrap it up.
        if not isinstance(model, tf.keras.layers.Layer):
            raise TypeError(
                "'model' should be a tf.keras.layers.Layer instance."
            )
        if not isinstance(model, tf.keras.Model):
            model = tf.keras.Sequential([model])
        super().__init__(model)
        # Compile the wrapped model and retain compilation arguments.
        kwargs.update({"loss": loss, "metrics": metrics})
        model.compile(**kwargs)
        self._compile_kwargs = kwargs
        # Instantiate a SGD optimizer to apply updates as-provided.
        self._sgd = tf.keras.optimizers.SGD(learning_rate=1.)

    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""
        config = tf.keras.layers.serialize(self._model)  # type: Dict[str, Any]
        return {"model": config, "compile": self._compile_kwargs}

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'Model':
        """Instantiate a model from a configuration dict."""
        for key in ("model", "compile"):
            if key not in config.keys():
                raise KeyError(f"Missing key '{key}' in the config dict.")
        model = tf.keras.layers.deserialize(config["model"])
        return cls(model, **config["compile"])

    def get_weights(
            self,
        ) -> NumpyVector:
        """Return the model's trainable weights."""
        return NumpyVector({
            str(i): arr for i, arr in enumerate(self._model.get_weights())
        })

    def set_weights(
            self,
            weights: NumpyVector,
        ) -> None:
        """Assign values to the model's trainable weights."""
        self._model.set_weights(list(weights.coefs.values()))

    def compute_batch_gradients(
            self,
            batch: Batch,
        ) -> TensorflowVector:
        """Compute and return the model's gradients over a data batch."""
        inputs, y_true, s_wght = self._verify_batch(batch)
        with tf.GradientTape() as tape:
            y_pred = self._model(inputs)
            loss = self._model.compute_loss(inputs, y_true, y_pred, s_wght)
            grad = tape.gradient(loss, self._model.trainable_weights)
        return TensorflowVector({str(i): tns for i, tns in enumerate(grad)})

    def _verify_batch(
            self,
            batch: Batch,
        ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Enforce Tensor conversion to batched data."""
        # Define an array-to-tensor conversion routine.
        def convert(data: Optional[ArrayLike]) -> Optional[tf.Tensor]:
            if (data is None) or tf.is_tensor(data):
                return data
            return tf.convert_to_tensor(data)
        # Apply it to the the batched elements.
        return tf.nest.map_structure(convert, batch)

    def apply_updates(  # type: ignore  # future: revise
            self,
            updates: TensorflowVector,
        ) -> None:
        """Apply updates to the model's weights."""
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
        """Compute the average loss of the model on a given dataset.

        dataset: iterable of batches
            Iterable yielding batch structures that are to be unpacked
            into (input_features, target_labels, [sample_weights]).
            If set, sample weights will affect the loss averaging.

        Return the average value of the model's loss over samples.
        """
        total = 0.
        n_btc = 0
        for batch in dataset:
            inputs, y_true, s_wght = batch
            y_pred = self._model(inputs)
            loss = self._model.compute_loss(inputs, y_true, y_pred, s_wght)
            total += loss.numpy()
            n_btc += 1
        return total / n_btc
