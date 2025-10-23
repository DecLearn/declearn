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

"""Hacky OptiModule subclass enabling the use of a keras Optimizer."""

from typing import Any, Dict, List, Union

# fmt: off
# pylint: disable=import-error,no-name-in-module
import tensorflow as tf  # type: ignore
import tensorflow.keras as tf_keras  # type: ignore

from declearn.model.api import Vector
from declearn.model.tensorflow._vector import TensorflowVector
from declearn.model.tensorflow.utils import select_device
from declearn.optimizer.modules import OptiModule
from declearn.utils import get_device_policy

# pylint: enable=import-error,no-name-in-module
# fmt: on


__all__ = [
    "TensorflowOptiModule",
]


class TensorflowOptiModule(OptiModule):
    """Framework-specific OptiModule to wrap tensorflow-keras optimizers.

    This tensorflow-only OptiModule enables wrapping a tensorflow-keras
    `keras.optimizers.Optimizer` to make it part of a declearn Optimizer
    pipeline, where it may be combined with other, framework-agnostic
    tools (notably FL-specific ones such as the FedProx loss regularizer).

    The wrapped keras Optimizer states will be placed on a device (CPU
    or GPU) selected automatically based on the global device policy.
    This device will also be used to place all wrapped computations.
    The `reset` and `set_state` methods both result in consulting the
    policy anew and therefore updating the placement of internal states
    and computations. `reset` also drops internal states' values.

    Please note that this relies on a hack that may have unforeseen side
    effects on the optimization algorithm if used carelessly and will at
    any rate cause some memory overhead. Thus it should be used sparingly,
    taking into account the following constraints and limitations:

    - The wrapped optimizer's learning rate will be forced to 1.0, so that
      updates' scaling remains the responsibility of the wrapping declearn
      Optimizer.
    - The wrapped optimizer should not make use of the updated variables'
      values, only of their gradients, because it will in fact operate on
      artificial, zero-valued variables at each step.
    - If the module is to be used by the clients, the wrapped optimizer
      class must have been imported from a third-party package that is
      also available to the clients (e.g. tensorflow).

    This class is mostly provided for experimental use of algorithms that
    are not natively available in declearn, for users that do not want to
    put in (or reserve for later) the effort of writing a custom, dedicated,
    framework-agnostic OptiModule subclass implementing that algorithm.
    If you encounter issues, please report to the declearn developers, and
    we will be happy to assist with debugging the present module and/or
    implementing the desired algorithm as a proper OptiModule.

    Finally, please note that some keras optimizers use different formulas
    than other reference implementations, including the declearn ones (e.g.
    for Adam, Adagrad or RMSProp). As a result, switching a keras optimizer
    instead of a declearn one can lead to diverging results.
    """

    name = "tensorflow-optim"

    def __init__(
        self,
        optim: Union[tf_keras.optimizers.Optimizer, str, Dict[str, Any]],
    ) -> None:
        """Instantiate a hacky tensorflow optimizer plug-in module.

        Parameters
        ----------
        optim: tf.keras.optimizers.Optimizer or dict[str, any] or str
            Keras optimizer instance that needs wrapping, or configuration
            dict or string identifier of one, enabling its retrieval using
            `tensorflow.keras.optimizer.get`.
            Note that if an instance is provided, a copy will be created.

        Note that the wrapped optimizer's base learning rate will be forced
        to be 1.0 and be constant. EMA and weight decay will also be forced
        not to be used due to the wrapped optimizer not accessing the actual
        model parameters; to implement these, please use the `weight_decay`
        parameter of `declearn.optimizer.Optimizer` and/or the `EWMAModule`
        plug-in.
        """
        # Select the device where to place the wrapped states and computations.
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Wrap the provided optimizer, enforcing a fixed learning rate of 1.
        # Also prevent the use of weight-decay or built-in ema (~momentum).
        self.optim = tf_keras.optimizers.get(optim)
        config = self.optim.get_config()
        config["weight_decay"] = 0
        config["use_ema"] = False
        if "learning_rate" in config:
            config["learning_rate"] = 1.0
        # Force the use of a brand-new optimizer instance.
        with tf.device(self._device):
            self.optim = self.optim.from_config(config)
        # Create a container for artificial, zero-valued variables.
        self._vars: Dict[str, tf.Variable] = {}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Run input gradients through the wrapped keras Optimizer.

        Parameters
        ----------
        gradients: TensorflowVector
            Input gradients that are to be processed and updated.

        Raises
        ------
        TypeError
            If `gradients` are not a TensorflowVector (this module is
            a framework-specific hack).
        KeyError
            If `gradients` have an inconsistent spec with the first
            ones ever processed by this module. Use `reset` if you
            wish to start back from the beginning.

        Returns
        -------
        gradients: TensorflowVector
            Modified input gradients. The output Vector should be
            fully compatible with the input one - only the values
            of the wrapped coefficients may have changed.
        """
        # Run type and specs verifications. Initialize variables if needed.
        if not isinstance(gradients, TensorflowVector):
            raise TypeError(
                "TensorflowOptiModule only supports TensorflowVector "
                "input gradients."
            )
        if not self._vars:
            self._init_variables(gradients)
        if gradients.coefs.keys() != self._vars.keys():
            raise KeyError(
                "Mismatch between input gradients and stored parameters."
            )
        # Perform the optimization step on the policy-defined device.
        with tf.device(self._device):
            # Zero-out the artificial variables.
            for var in self._vars.values():
                var.assign_sub(var, read_value=False)
            # Zip gradients and variables, then compute and apply updates.
            grads_and_vars = [
                (gradients.coefs[key], var) for key, var in self._vars.items()
            ]
            self.optim.apply_gradients(grads_and_vars)
            # Collect the updates, sparsifying back IndexedSlices.
            coefs = {key: -var.value() for key, var in self._vars.items()}
            for key, val in gradients.coefs.items():
                if isinstance(val, tf.IndexedSlices):
                    values = tf.gather(coefs[key], val.indices)
                    coefs[key] = tf.IndexedSlices(
                        values, val.indices, val.dense_shape
                    )
        return TensorflowVector(coefs)

    def _init_variables(self, gradients: TensorflowVector) -> None:
        """Create zero-valued variables based on input gradients' specs."""
        with tf.device(self._device):
            self._vars = {
                key: tf.Variable(tf.zeros_like(grad), name=key)
                for key, grad in gradients.coefs.items()
            }
            self.optim.build(list(self._vars.values()))

    def reset(self) -> None:
        """Reset this module to its uninitialized state.

        Discard the wrapped tensorflow Variables (that define a required
        specification of input gradients), and replace the optimizer with
        a new, uninitialized one. As a consequence, the next call to `run`
        will result in setting a new required input specification.

        This method also updates the device-placement policy of the states
        and computations wrapped by this OptiModule, based on the global
        policy accessed via `declearn.utils.get_device_policy`.
        """
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        with tf.device(self._device):
            self._vars.clear()
            self.optim = self.optim.from_config(self.optim.get_config())

    def get_config(
        self,
    ) -> Dict[str, Any]:
        optim = tf_keras.optimizers.serialize(self.optim)
        return {"optim": optim}

    def get_state(
        self,
    ) -> Dict[str, Any]:
        specs = {
            key: (val.shape.as_list(), val.dtype.name)
            for key, val in self._vars.items()
        }
        variables = self._get_optimizer_variables()
        state = TensorflowVector(
            {str(i): v.value() for i, v in enumerate(variables)}
        )
        return {"specs": specs, "state": state}

    def _get_optimizer_variables(
        self,
    ) -> List[tf.Variable]:
        """Access wrapped optimizer's variables as 'tf.Variable' instances."""
        if hasattr(tf_keras, "version") and tf_keras.version().startswith("3"):
            return [var.value for var in self.optim.variables]
        return self.optim.variables()

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("specs", "state"):
            if key not in state:
                raise KeyError(
                    "Missing required key in input TensorflowOptiModule "
                    f"state dict: '{key}'."
                )
        # Restore weight variables' specifications from the input state dict.
        self.reset()  # note: this also updates the device policy
        with tf.device(self._device):
            self._vars = {
                key: tf.Variable(tf.zeros(shape, dtype), name=key)
                for key, (shape, dtype) in state["specs"].items()
            }
            self.optim.build(list(self._vars.values()))
        # Restore optimizer variables' values from the input state dict.
        opt_vars = self._get_optimizer_variables()
        with tf.device(self._device):
            for var, val in zip(
                opt_vars, state["state"].coefs.values(), strict=False
            ):
                var.assign(val, read_value=False)
