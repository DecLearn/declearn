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

"""Model subclass to wrap Haiku models."""

import functools
import inspect
import io
import warnings
from copy import deepcopy
from random import SystemRandom
from typing import (
    # fmt: off
    Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
)

import haiku as hk
import jax
import jax.numpy as jnp
import joblib  # type: ignore
import numpy as np
from typing_extensions import Self

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
        super().__init__(model)
        # Assign loss module.
        self._loss_fn = loss
        # Get pure functions from haiku transform.
        self._model_fn = model
        self._transformed_model = hk.transform(model)
        # Select the device where to place computations.
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Create model state attributes
        self._pleaves = []  # type: List[jax.Array]
        self._treedef = None  # type: Optional[jax.tree_util.PyTreeDef]
        self._trainable = []  # type: List[int]
        # Initialize the PRNG
        if seed is None:
            seed = int(SystemRandom().random() * 10e6)
        self._rng_gen = hk.PRNGSequence(seed)
        # Initialized and data_info utils
        self._initialized = False
        self.data_info = {}  # type: Dict[str, Any]

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
        # initialize.
        params = self._transformed_model.init(
            next(self._rng_gen),
            jnp.zeros(
                (1, *data_info["features_shape"]), data_info["data_type"]
            ),
        )
        params = jax.device_put(params, self._device)
        pleaves, treedef = jax.tree_util.tree_flatten(params)
        self._treedef = treedef
        self._pleaves = pleaves
        self._trainable = list(range(len(pleaves)))
        self._initialized = True

    def get_config(
        self,
    ) -> Dict[str, Any]:
        warnings.warn(
            "Our custom Haiku serialization relies on pickle,"
            "which may be unsafe."
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
        params = {str(k): v for k, v in enumerate(self._pleaves)}
        if trainable:
            params = {str(idx): params[str(idx)] for idx in self._trainable}
        return JaxNumpyVector(params)

    def get_named_weights(
        self,
        trainable: bool = False,
    ) -> Dict[str, Dict[str, jax.Array]]:
        """Access the weights of the haiku model as a nested dict.

        Return type is any due to `jax.tree_util.tree_unflatten`.

        trainable: bool, default=False
            If True, restrict the returned weights to the trainable ones,
            else return all weights.
        """
        assert self._treedef is not None, "uninitialized JaxModel"
        params = jax.tree_util.tree_unflatten(self._treedef, self._pleaves)
        if trainable:
            pop_idx = set(range(len(self._pleaves))) - set(self._trainable)
            for i, (layer, name, _) in enumerate(self._traverse_params()):
                if i in pop_idx:
                    params[layer].pop(name)
                if len(params[layer]) == 0:
                    params.pop(layer)
        return params

    def _get_fixed_named_weights(self) -> Any:
        """Access the fixed weights of the model as a nested dict, if any.

        Return type is any due to `jax.tree_util.tree_unflatten`.
        """
        assert self._treedef is not None, "uninitialized JaxModel"
        if len(self._trainable) == len(self._pleaves):
            return {}
        params = jax.tree_util.tree_unflatten(self._treedef, self._pleaves)
        pop_idx = set(self._trainable)
        for i, (layer, name, _) in enumerate(self._traverse_params()):
            if i in pop_idx:
                params[layer].pop(name)
            if len(params[layer]) == 0:
                params.pop(layer)
        return params

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: JaxNumpyVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector weights.")
        self._verify_weights_compatibility(weights, trainable=trainable)
        coefs_copy = deepcopy(weights.coefs)
        if trainable:
            for idx in self._trainable:
                self._pleaves[idx] = jax.device_put(
                    coefs_copy[str(idx)], self._device
                )
        else:
            self._pleaves = [
                jax.device_put(v, self._device) for v in coefs_copy.values()
            ]

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
        if trainable:
            expected = {str(i) for i in self._trainable}
        else:
            expected = {str(i) for i in range(len(self._pleaves))}
        raise_on_stringsets_mismatch(
            received, expected, context="model weights"
        )

    def set_trainable_weights(
        self,
        criterion: Union[
            Callable[[str, str, jax.Array], bool],
            Dict[str, Dict[str, Any]],
            List[int],
        ],
    ) -> None:
        """Sets the index of trainable weights.

        The split can be done by providing a functions applying conditions on
        the named weights, as haiku users are used to do, but can also accept
        an explicit dict of names or even the index of the parameter leaves
        stored by our HaikuModel.

        Example use :
            >>> self.get_named_weights() = {'linear': {'w': None, 'b': None}}
        Using a function as input
            >>> criterion = lambda layer, name, value: name == 'w'
            >>> self.set_trainable_weights(criterion)
            >>> self._trainable
            [0]
        Using a dictionnary or pytree
            >>> criterion = {'linear': {'b': None}}
            >>> self.set_trainable_weights(criterion)
            >>> self._trainable
            [1]

        Note : model needs to be initialized

        Arguments
        --------
        criterion : Callable or dict(str,dict(str,any)) or list(int)
            Criterion to be used to identify trainable params. If Callable,
            must be a function taking in the name of the module (e.g.
            layer name), the element name (e.g. parameter name) and the
            corresponding data and returning a boolean. See
            [the haiku doc](https://tinyurl.com/3v28upaz)
            for details. If a list of integers, should represent the index of
            trainable  parameters in the parameter tree leaves. If a dict,
            should be formatted as a pytree.

        """
        if not self._initialized:
            raise ValueError("Model needs to be initialized first")
        if isinstance(criterion, list) and isinstance(criterion[0], int):
            self._trainable = criterion
        else:
            self._trainable = []  # reset if needed
            if inspect.isfunction(criterion):
                include_fn = (
                    criterion
                )  # type: Callable[[str, str, jax.Array], bool]
            elif isinstance(criterion, dict):
                include_fn = self._build_include_fn(criterion)
            else:
                raise TypeError(
                    "The provided criterion does not conform"
                    "to the expected format and or type"
                )
            for idx, (layer, name, value) in enumerate(
                self._traverse_params()
            ):
                if include_fn(layer, name, value):
                    self._trainable.append(idx)

    def _traverse_params(self) -> Iterator[Tuple[str, str, jax.Array]]:
        """Traverse the pytree of a model's named weights.

        Yield (layer_name, weight_name, weight_value) tuples from
        traversing the pytree left-to-right, depth-first.
        """
        assert self._treedef is not None, "uninitialized JaxModel"
        params = jax.tree_util.tree_unflatten(self._treedef, self._pleaves)
        for layer in params:
            bundle = params[layer]
            for name in bundle:
                value = bundle[name]
                yield layer, name, value

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
        batch: (jax.Array, jax.Array, optional[jax.Array])
            Batch of jax-converted inputs, comprising (in that order)
            input data, ground-truth labels and optional sample weights.

        Returns
        -------
        loss: jax.Array
            The mean loss over the input data provided.
        """
        # FUTURE: add support for lists of inputs
        inputs, y_true, s_wght = batch
        params = hk.data_structures.merge(train_params, fixed_params)
        y_pred = self._transformed_model.apply(params, rng, *inputs)
        s_loss = self._loss_fn(y_pred, y_true)  # type: ignore
        if s_wght is not None:
            s_loss = s_loss * s_wght
        return jnp.mean(s_loss)

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> JaxNumpyVector:
        if max_norm:
            return self._compute_clipped_gradients(batch, max_norm)
        return self._compute_batch_gradients(batch)

    def _compute_batch_gradients(
        self,
        batch: Batch,
    ) -> JaxNumpyVector:
        """Compute and return batch-averaged gradients of trainable weights."""
        # Unpack input batch and unflatten model parameters.
        inputs = self._unpack_batch(batch)
        train_params = self.get_named_weights(trainable=True)
        fixed_params = self._get_fixed_named_weights()
        # Run the forward and backward passes to compute gradients.
        grads = self._grad_fn(
            train_params,
            fixed_params,
            next(self._rng_gen),
            inputs,
        )
        # Flatten the gradients and return them in a Vector container.
        flat_grad, _ = jax.tree_util.tree_flatten(grads)
        return JaxNumpyVector({str(k): v for k, v in enumerate(flat_grad)})

    @functools.cached_property
    def _grad_fn(
        self,
    ) -> Callable[[hk.Params, hk.Params, jax.Array, JaxBatch], hk.Params]:
        """Lazy-built jax function to compute batch-averaged gradients."""
        return jax.jit(jax.grad(self._forward))

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: float,
    ) -> JaxNumpyVector:
        """Compute and return sample-wise clipped, batch-averaged gradients."""
        # Unpack input batch and unflatten model parameters.
        inputs = self._unpack_batch(batch)
        train_params = self.get_named_weights(trainable=True)
        fixed_params = self._get_fixed_named_weights()
        # Get flattened, per-sample, clipped gradients and aggregate them.
        clipped_grads = self._clipped_grad_fn(
            train_params,
            fixed_params,
            next(self._rng_gen),
            inputs,
            max_norm,
        )
        grads = [g.sum(0) for g in clipped_grads]
        # Return them in a Vector container.
        return JaxNumpyVector({str(k): v for k, v in enumerate(grads)})

    @functools.cached_property
    def _clipped_grad_fn(
        self,
    ) -> Callable[
        [hk.Params, hk.Params, jax.Array, JaxBatch, float], List[jax.Array]
    ]:
        """Lazy-built jax function to compute clipped sample-wise gradients.

        Note : The vmap in_axis parameters work thank to the jax feature of
        applying optional parameters to pytrees."""

        def clipped_grad_fn(
            train_params: hk.Params,
            fixed_params: hk.Params,
            rng: jax.Array,
            batch: JaxBatch,
            max_norm: float,
        ) -> List[jax.Array]:
            """Compute and clip gradients wrt parameters for a sample."""
            inputs, y_true, s_wght = batch
            batch = (inputs, y_true, None)
            grads = jax.grad(self._forward)(
                train_params,
                fixed_params,
                rng,
                batch,
            )
            grads_flat = [
                grad / jnp.maximum(jnp.linalg.norm(grad) / max_norm, 1.0)
                for grad in jax.tree_util.tree_leaves(grads)
            ]
            if s_wght is not None:
                grads_flat = [g * s_wght for g in grads_flat]
            return grads_flat

        in_axes = [None, None, None, 0, None]  # map on inputs' first dimension
        return jax.jit(jax.vmap(clipped_grad_fn, in_axes))

    @staticmethod
    def _unpack_batch(batch: Batch) -> JaxBatch:
        """Unpack and enforce jnp.array conversion to an input data batch."""

        def convert(data: Any) -> Optional[jax.Array]:
            if (data is None) or isinstance(data, jax.Array):
                return data
            if isinstance(data, np.ndarray):
                return jnp.array(data)  # pylint: disable=no-member
            raise TypeError("HaikuModel requires numpy or jax.numpy data.")

        # similar code to TorchModel; pylint: disable=duplicate-code
        # Convert batched data to jax Arrays.
        inputs, y_true, s_wght = batch
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        output = [list(map(convert, inputs)), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: JaxNumpyVector,
    ) -> None:
        if not isinstance(updates, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector updates.")
        self._verify_weights_compatibility(updates, trainable=True)
        for key, upd in updates.coefs.items():
            self._pleaves[int(key)] += upd

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
        params = self.get_named_weights()
        y_pred = np.asarray(
            self._transformed_model.apply(params, next(self._rng_gen), *inputs)
        )
        y_true = np.asarray(y_true)  # type: ignore
        s_wght = None if s_wght is None else np.asarray(s_wght)  # type: ignore
        return y_true, y_pred, s_wght  # type: ignore

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        s_loss = self._loss_fn(jnp.array(y_pred), jnp.array(y_true))
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
            self._pleaves = jax.device_put(self._pleaves, self._device)
