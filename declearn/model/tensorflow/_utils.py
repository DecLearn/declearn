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

"""Backend utils for the declearn.model.tensorflow module."""

import inspect

from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf  # type: ignore


__all__ = [
    "build_keras_loss",
]


# alias for loss functions' signature in keras
CallableLoss = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@tf.keras.utils.register_keras_serializable(package="declearn")
class LossFunction(tf.keras.losses.Loss):
    """Generic loss function container enabling reduction strategy control."""

    def __init__(
        self,
        loss_fn: Union[str, CallableLoss],
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(reduction, name)
        self.loss_fn = tf.keras.losses.deserialize(loss_fn)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        # inherited docstring; pylint: disable=missing-docstring
        return self.loss_fn(y_true, y_pred)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        # inherited docstring; pylint: disable=missing-docstring
        config = super().get_config()  # type: Dict[str, Any]
        config["loss_fn"] = tf.keras.losses.serialize(self.loss_fn)
        return config


def build_keras_loss(
    loss: Union[str, tf.keras.losses.Loss, CallableLoss],
    reduction: str = tf.keras.losses.Reduction.NONE,
) -> tf.keras.losses.Loss:
    """Type-check, deserialize and/or wrap a keras loss into a Loss object.

    Parameters
    ----------
    loss: str or tf.keras.losses.Loss or function(y_true, y_pred)->loss
        Either a keras Loss object, the name of a keras loss, or a loss
        function that needs wrapping into a Loss object.
    reduction: str, default=`tf.keras.losses.Reduction.NONE`
        Reduction scheme to apply on point-wise loss values.

    Returns
    -------
    loss_obj: tf.keras.losses.Loss
        Loss object, configured to apply the `reduction` scheme.
    """
    # Case when 'loss' is already a Loss object.
    if isinstance(loss, tf.keras.losses.Loss):
        loss.reduction = reduction
    # Case when 'loss' is a string.
    elif isinstance(loss, str):
        cls = tf.keras.losses.deserialize(loss)
        # Case when the string was deserialized into a function.
        if inspect.isfunction(cls):
            # Try altering the string to gather its object counterpart.
            loss = "".join(word.capitalize() for word in loss.split("_"))
            try:
                loss = tf.keras.losses.deserialize(loss)
                loss.reduction = reduction
            # If this failed, try wrapping the function using LossFunction.
            except ValueError:
                loss = LossFunction(cls)
        # Case when the string was deserialized into a class.
        else:
            loss = cls(reduction=reduction)
    # Case when 'loss' is a function: wrap it up using LossFunction.
    elif inspect.isfunction(loss):
        loss = LossFunction(loss, reduction=reduction)
    # Case when 'loss' is of invalid type: raise a TypeError.
    if not isinstance(loss, tf.keras.losses.Loss):
        raise TypeError(
            "'loss' should be a keras Loss object or the name of one."
        )
    # Otherwise, properly configure the reduction scheme and return.
    return loss
