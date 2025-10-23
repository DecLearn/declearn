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

"""Model subclass to wrap Haiku models."""

import functools
import inspect
import io
import warnings
from random import SystemRandom
from typing import Any, Callable, Dict, List, Optional, Self, Set, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import joblib  # type: ignore
import numpy as np

from declearn.data_info import aggregate_data_info
from declearn.model._utils import raise_on_stringsets_mismatch
from declearn.model.api import Model
from declearn.model.haiku._vector import JaxNumpyVector
from declearn.model.haiku.utils import select_device
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type

__all__ = [
    "HaikuModel",
]

# alias for unpacked Batch structures, converted to jax arrays
# FUTURE: add support for lists of inputs
JaxBatch = Tuple[List[jax.Array], Optional[jax.Array], Optional[jax.Array]]


@register_type(name="HaikuModel", group="Model")
class HaikuModel(Model):
    """Model wrapper for Haiku Model instances.

    This `Model` subclass is designed to wrap a `hk.Module`
    instance to be learned federatively.

    Notes regarding device management (CPU, GPU, etc.):

    * By default, jax places data and operations on GPU whenever one
      is available.
    * Our `HaikuModel` instead consults the device-placement policy (via
      `declearn.utils.get_device_policy`), places the wrapped haiku model's
      weights there, and runs computations defined under public methods on
      that device.
    * Note that there is no guarantee that calling a private method directly
      will result in abiding by that policy. Hence, be careful when writing
      custom code, and use your own context managers to get guarantees.
    * Note that if the global device-placement policy is updated, this will
      only be propagated to existing instances by manually calling their
      `update_device_policy` method.
    * You may consult the device policy enforced by a HaikuModel
      instance by accessing its `device_policy` property.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model: Callable[[jax.Array], jax.Array],
        loss: Callable[[jax.Array, jax.Array], jax.Array],
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate a Model interface wrapping a jax-haiku model.

        Parameters
        ----------
        model: callable(jax.Array) -> jax.Array
            Function encapsulating a `haiku.Module` such that `model(x)`
            returns `haiku_module(x)`, constituting a model's forward.
        loss: callable(jax.Array, jax.Array) -> jax.Array
            Jax-compatible function that defines the model's loss.
            It must expect `y_pred` and `y_true` as input arguments (in
            that order) and return sample-wise loss values.
        seed: int or None, default=None
            Random seed used to initialize the haiku-wrapped Pseudo-random
            number generator. If none is provided, draw an integer between
            0 and 10^6 using `random.SystemRandom`.
        """
        # Transform the input function using Haiku, and wrap the result.
        super().__init__(hk.transform(model))
        # Assign the loss function, and initial model one.
        self._model_fn = model
        self._loss_fn = loss
        # Select the device where to place computations.
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Create model state attributes.
        self._params: hk.Params = {}
        self._pnames: List[str] = []
        self._trainable: List[str] = []
        # Initialize the PRNG.
        if seed is None:
            seed = int(SystemRandom().random() * 10e6)
        self._rng_gen = hk.PRNGSequence(
            jax.device_put(np.array([0, seed], dtype="uint32"), self._device)
        )
        # Initialized and data_info utils
        self._initialized = False
        self.data_info: Dict[str, Any] = {}

    @property
    def device_policy(
        self,
    ) -> DevicePolicy:
        device = self._device
        return DevicePolicy(gpu=(device.platform == "gpu"), idx=device.id)

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        return set() if self._initialized else {"data_type", "features_shape"}

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        if self._initialized:
            return
        # Check that required fields are available and of valid type.
        self.data_info = aggregate_data_info(
            [data_info], self.required_data_info
        )
        # Initialize the model parameters, using zero-valued inputs.
        inputs = jax.device_put(
            np.zeros((1, *data_info["features_shape"])), self._device
        ).astype(data_info["data_type"])
        with warnings.catch_warnings():  # jax.jit(device=...) is deprecated
            warnings.simplefilter("ignore", DeprecationWarning)
            params = jax.jit(  # pylint: disable=[not-callable]
                self._model.init, device=self._device
            )(
                next(self._rng_gen), inputs
            )  # NOTE: jit is used to force haiku's device selection
        self._params = jax.device_put(params, self._device)
        self._pnames = [
            f"{layer}:{weight}"
            for layer, weights in self._params.items()
            for weight in weights
        ]
        self._trainable = self._pnames.copy()
        self._initialized = True

    def get_config(
        self,
    ) -> Dict[str, Any]:
        warnings.warn(
            "Our custom Haiku serialization relies on pickle,"
            "which may be unsafe.",
            stacklevel=2,
        )
        with io.BytesIO() as buffer:
            joblib.dump(self._model_fn, buffer)
            model = buffer.getbuffer().hex()
        with io.BytesIO() as buffer:
            joblib.dump(self._loss_fn, buffer)
            loss = buffer.getbuffer().hex()
        return {
            "model": model,
            "loss": loss,
            "data_info": self.data_info,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = joblib.load(buffer)
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = joblib.load(buffer)
        model = cls(model=model, loss=loss)
        if config.get("data_info"):
            model.initialize(config["data_info"])
        return model

    def get_weights(
        self,
        trainable: bool = False,
    ) -> JaxNumpyVector:
        params = {
            f"{layer}:{wname}": value
            for layer, weights in self._params.items()
            for wname, value in weights.items()
        }
        if trainable:
            params = {k: v for k, v in params.items() if k in self._trainable}
        return JaxNumpyVector(params)

    def set_weights(
        self,
        weights: JaxNumpyVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector weights.")
        self._verify_weights_compatibility(weights, trainable=trainable)
        for key, val in weights.coefs.items():
            layer, weight = key.rsplit(":", 1)
            self._params[layer][weight] = val.copy()  # type: ignore

    def _verify_weights_compatibility(
        self,
        vector: JaxNumpyVector,
        trainable: bool = False,
    ) -> None:
        """Verify that a vector has the same names as the model's weights.

        Parameters
        ----------
        vector: JaxNumpyVector
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
        received = set(vector.coefs)
        expected = set(self._trainable if trainable else self._pnames)
        raise_on_stringsets_mismatch(
            received, expected, context="model weights"
        )

    def set_trainable_weights(
        self,
        criterion: Union[
            Callable[[str, str, jax.Array], bool],
            Dict[str, Dict[str, Any]],
            List[str],
        ],
    ) -> None:
        """Sets the index of trainable weights.

        The split can be done by providing a functions applying conditions on
        the named weights, as haiku users are used to do, but can also accept
        an explicit dict of names or even the index of the parameter leaves
        stored by our HaikuModel.

        Notes
        -----
        - The model needs to be initialized for this method to work.
        - The list of model weight names (general, or restricted to trainable
          ones) may be consulted using the `get_weight_names` method.

        Usage
        -----

        Let us pretend the model is made of a single linear layer; we want
        to freeze its bias, leaving only the kernel weights trainable.
        ```
        >>> # Display current names of trainable model weights.
        >>> self.get_weight_names(trainable=True)
        ["linear/~/w", "linear/~/b"]
        ```
        - (A) Using a list of weight names:
        ```
        >>> criterion = ["linear/~/w"]
        >>> self.set_trainable_weights(criterion)
        ```
        - (B) Using a function as input:
        ```
        >>> criterion = lambda layer, name, value: name == 'w'
        >>> self.set_trainable_weights(criterion)
        ```
        - (C) Using a dictionnary or pytree:
        ```
        >>> criterion = {'linear': {'b': None}}
        >>> self.set_trainable_weights(criterion)
        ```
        - In all three cases, we can verify the results.
        ```
        >>> self.get_weight_names(trainable=True)
        ["linear/~/w"]
        ```

        Parameters
        ----------
        criterion: Callable or dict(str,dict(str,any)) or list(int)
            Criterion to be used to identify trainable params.

            - If a list of strings, should represent the names of weights to
              keep as trainable (freezing each and every other one).
            - If callable, must be a function taking in the name of the module
              (e.g. layer name), the element name (e.g. parameter name) and the
              corresponding data and returning a boolean.
              See [the haiku doc](https://tinyurl.com/3v28upaz) for details.
            - If a dict, should be formatted as a pytree, the keys of which
              are the nodes/leaves that should remain trainable.
        """
        if not self._initialized:
            raise ValueError("Model needs to be initialized first")
        if (
            isinstance(criterion, list)
            and all(isinstance(c, str) for c in criterion)
            and all(c in self._pnames for c in criterion)
        ):
            self._trainable = criterion
        else:
            self._trainable = []  # reset if needed
            if inspect.isfunction(criterion):
                include_fn: Callable[[str, str, jax.Array], bool] = criterion
            elif isinstance(criterion, dict):
                include_fn = self._build_include_fn(criterion)
            else:
                raise TypeError(
                    "The provided criterion does not conform "
                    "to the expected format and or type."
                )
            gen = hk.data_structures.traverse(self._params)
            for layer, name, value in gen:
                if include_fn(layer, name, value):
                    self._trainable.append(f"{layer}:{name}")

    @staticmethod
    def _build_include_fn(
        node_names: Dict[str, Dict[str, Any]],
    ) -> Callable[[str, str, jax.Array], bool]:
        """Build an equality-checking function for parameters' traversal."""

        def include_fn(layer: str, name: str, value: jax.Array) -> bool:
            # mandatory signature; pylint: disable=unused-argument
            if layer in list(node_names.keys()):
                return name in list(node_names[layer].keys())
            return False

        return include_fn

    def get_weight_names(
        self,
        trainable: bool = False,
    ) -> List[str]:
        """Return the list of names of the model's weights.

        Parameters
        ----------
        trainable: bool
            Whether to return only the names of trainable weights,
            rather than including both trainable and frozen ones.

        Returns
        -------
        names:
            Ordered list of model weights' names.
        """
        return self._trainable.copy() if trainable else self._pnames.copy()

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> JaxNumpyVector:
        # Unpack input batch and prepare model parameters.
        inputs = self._unpack_batch(batch)
        train_params, fixed_params = hk.data_structures.partition(
            predicate=lambda l, w, _: f"{l}:{w}" in self._trainable,  # noqa: E741
            structure=self._params,
        )
        rng = next(self._rng_gen)
        # Compute batch-averaged gradients, opt. clipped on a per-sample basis.
        if max_norm:
            grads, loss = self._clipped_grads_and_loss_fn(  # pylint: disable=[not-callable, line-too-long]
                train_params, fixed_params, rng, inputs, max_norm
            )
            grads = [value.mean(0) for value in grads]
        else:
            loss, grads_tree = self._loss_and_grads_fn(  # pylint: disable=[not-callable, line-too-long]
                train_params, fixed_params, rng, inputs
            )
            grads = jax.tree_util.tree_leaves(grads_tree)
        # Record the batch-averaged loss value.
        self._loss_history.append(float(np.array(loss).mean()))
        # Return the gradients, flattened into a JaxNumpyVector container.
        return JaxNumpyVector(dict(zip(self._trainable, grads, strict=False)))

    @functools.cached_property
    def _loss_and_grads_fn(
        self,
    ) -> Callable[
        [hk.Params, hk.Params, jax.Array, JaxBatch],
        Tuple[jax.Array, hk.Params],
    ]:
        """Lazy-built jax function to compute batch-averaged gradients."""
        return jax.jit(jax.value_and_grad(self._forward))

    def _forward(
        self,
        train_params: hk.Params,
        fixed_params: hk.Params,
        rng: jax.Array,
        batch: JaxBatch,
    ) -> jax.Array:
        """The forward pass chaining the model to the loss as a pure function.

        Parameters
        -------
        params: haiku.Params
            The model parameters, as a nested dict of jax arrays.
        rng: jax.Array
            A jax pseudo-random number generator (PRNG) key.
        batch: (list[jax.Array], jax.Array, optional[jax.Array])
            Batch of jax-converted inputs, comprising (in that order)
            input data, ground-truth labels and optional sample weights.

        Returns
        -------
        loss: jax.Array
            The mean loss over the input data provided.
        """
        inputs, y_true, s_wght = batch
        params = hk.data_structures.merge(train_params, fixed_params)
        y_pred = self._model.apply(params, rng, *inputs)
        s_loss = self._loss_fn(y_pred, y_true)  # type: ignore
        if s_wght is not None:
            s_loss = s_loss * s_wght
        return jnp.mean(s_loss)

    @functools.cached_property
    def _clipped_grads_and_loss_fn(
        self,
    ) -> Callable[
        [hk.Params, hk.Params, jax.Array, JaxBatch, float],
        Tuple[List[jax.Array], jax.Array],
    ]:
        """Lazy-built jax function to compute clipped sample-wise gradients.

        Note: The vmap in_axis parameters work thank to the jax feature of
        applying optional parameters to pytrees.
        """

        def clipped_grads_and_loss_fn(
            train_params: hk.Params,
            fixed_params: hk.Params,
            rng: jax.Array,
            batch: JaxBatch,
            max_norm: float,
        ) -> Tuple[List[jax.Array], jax.Array]:
            """Compute and clip gradients wrt parameters for a sample."""
            inputs, y_true, s_wght = batch
            batch = (inputs, y_true, None)
            loss, grads = jax.value_and_grad(self._forward)(
                train_params, fixed_params, rng, batch
            )
            grads_flat = [
                grad / jnp.maximum(jnp.linalg.norm(grad) / max_norm, 1.0)
                for grad in jax.tree_util.tree_leaves(grads)
            ]
            if s_wght is not None:
                grads_flat = [g * s_wght for g in grads_flat]
            return grads_flat, loss

        in_axes = [None, None, None, 0, None]  # map on inputs' first dimension
        return jax.jit(jax.vmap(clipped_grads_and_loss_fn, in_axes))

    def _unpack_batch(
        self,
        batch: Batch,
    ) -> JaxBatch:
        """Unpack and enforce jnp.array conversion to an input data batch."""

        def convert(data: Any) -> Optional[jax.Array]:
            if data is None:
                return data
            if isinstance(data, (jax.Array, np.ndarray)):
                return jax.device_put(data, self._device)
            raise TypeError("HaikuModel requires numpy or jax.numpy data.")

        # similar code to TorchModel; pylint: disable=duplicate-code
        # Convert batched data to jax Arrays.
        inputs, y_true, s_wght = batch
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        output = [list(map(convert, inputs)), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def apply_updates(
        self,
        updates: JaxNumpyVector,
    ) -> None:
        if not isinstance(updates, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector updates.")
        self._verify_weights_compatibility(updates, trainable=True)
        for key, val in updates.coefs.items():
            layer, weight = key.rsplit(":", 1)
            self._params[layer][weight] += val  # type: ignore

    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        inputs, y_true, s_wght = self._unpack_batch(batch)
        if y_true is None:
            raise TypeError(
                "`HaikuModel.compute_batch_predictions` received a "
                "batch with `y_true=None`, which is unsupported. Please "
                "correct the inputs, or override this method to support "
                "creating labels from the base inputs."
            )
        y_pred = self._predict_fn(  # pylint: disable=[not-callable]
            self._params, next(self._rng_gen), *inputs
        )
        return (
            np.asarray(y_true),
            np.asarray(y_pred),
            None if s_wght is None else np.asarray(s_wght),
        )

    @functools.cached_property
    def _predict_fn(
        self,
    ) -> Callable[[hk.Params, jax.Array, jax.Array], jax.Array]:
        """Lazy-built jax function to run in inference on a given device."""
        with warnings.catch_warnings():  # jax.jit(device=...) is deprecated
            warnings.simplefilter("ignore", DeprecationWarning)
            return jax.jit(self._model.apply, device=self._device)

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        s_loss = self._loss_fn(
            jax.device_put(y_pred, self._device),
            jax.device_put(y_true, self._device),
        )
        return np.array(s_loss).squeeze()

    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        # similar code to tensorflow Model; pylint: disable=duplicate-code
        # Select the device to use based on the provided or global policy.
        if policy is None:
            policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # When needed, re-create the model to force moving it to the device.
        if self._device is not device:
            self._device = device
            self._params = jax.device_put(self._params, self._device)
            # Delete a cached, device-committed JIT function.
            del self._predict_fn
