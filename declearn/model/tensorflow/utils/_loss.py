# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Function to parse and/or wrap a keras loss for use with declearn."""

import inspect
from typing import Any, Callable, Dict, Optional, Union

# fmt: off
# pylint: disable=import-error,no-name-in-module
import tensorflow as tf  # type: ignore
import tensorflow.keras as tf_keras  # type: ignore

# pylint: enable=import-error,no-name-in-module
# fmt: on


__all__ = [
    "build_keras_loss",
]


# alias for loss functions' signature in keras
CallableLoss = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@tf_keras.utils.register_keras_serializable(package="declearn")
class LossFunction(tf_keras.losses.Loss):
    """Generic loss function container enabling reduction strategy control."""

    def __init__(
        self,
        loss_fn: Union[str, CallableLoss],
        reduction: str = tf_keras.losses.Reduction.NONE,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(reduction=reduction, name=name)  # type: ignore
        self.loss_fn = tf_keras.losses.deserialize(loss_fn)  # type: ignore

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
        config: Dict[str, Any] = super().get_config()
        config["loss_fn"] = tf_keras.losses.serialize(self.loss_fn)
        return config


def build_keras_loss(
    loss: Union[str, tf_keras.losses.Loss, CallableLoss],
    reduction: str = tf_keras.losses.Reduction.NONE,
) -> tf_keras.losses.Loss:
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
    loss_obj: tf_keras.losses.Loss
        Loss object, configured to apply the `reduction` scheme.
    """
    # Case when 'loss' is already a Loss object.
    if isinstance(loss, tf_keras.losses.Loss):
        loss.reduction = reduction  # type: ignore
    # Case when 'loss' is a string: deserialize and/or wrap into a Loss object.
    elif isinstance(loss, str):
        loss = get_keras_loss_from_string(name=loss, reduction=reduction)
    # Case when 'loss' is a function: wrap it up using LossFunction.
    elif inspect.isfunction(loss):
        loss = LossFunction(loss, reduction=reduction)
    # Case when 'loss' is of invalid type: raise a TypeError.
    if not isinstance(loss, tf_keras.losses.Loss):
        raise TypeError(
            "'loss' should be a keras Loss object or the name of one."
        )
    # Otherwise, properly configure the reduction scheme and return.
    return loss


def get_keras_loss_from_string(
    name: str,
    reduction: str,
) -> tf_keras.losses.Loss:
    """Instantiate a keras Loss object from a registered string identifier.

    - If `name` matches a Loss registration name, return an instance.
    - If it matches a loss function registration name, return either
      an instance from its name-matching Loss subclass, or a custom
      Loss subclass instance wrapping the function.
    - If it does not match anything, raise a ValueError.
    """
    loss = tf_keras.losses.deserialize(name)
    if isinstance(loss, tf_keras.losses.Loss):
        loss.reduction = reduction  # type: ignore
        return loss
    if inspect.isfunction(loss):
        try:
            name = "".join(word.capitalize() for word in name.split("_"))
            return get_keras_loss_from_string(name, reduction)
        except ValueError:
            return LossFunction(
                loss, reduction=reduction, name=getattr(loss, "__name__", None)
            )
    raise ValueError(
        f"Name '{loss}' cannot be deserialized into a keras loss."
    )
