# coding: utf-8

"""Model subclass to wrap Haiku models."""

import io
import functools
import warnings
from copy import deepcopy
from random import SystemRandom
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import joblib  # type: ignore
import numpy as np
from typing_extensions import Self

from declearn.data_info import aggregate_data_info
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
JaxBatch = Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]


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
        self._params = []  # type: List[jax.Array]
        self._treedef = None  # type: Optional[jax.tree_util.PyTreeDef]
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
        # Check that required fields are available and of valid type.
        data_info = aggregate_data_info([data_info], self.required_data_info)
        self.data_info = deepcopy(data_info)
        # initialize.
        params = self._transformed_model.init(
            next(self._rng_gen),
            jnp.zeros(
                (1, *data_info["features_shape"]), data_info["data_type"]
            ),
        )
        params = jax.device_put(params, self._device)
        flat_params = jax.tree_util.tree_flatten(params)
        self._treedef = deepcopy(flat_params[1])
        self._params = flat_params[0]
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
        params = {str(k): v for k, v in enumerate(self._params)}
        return JaxNumpyVector(params)

    def get_named_weights(self) -> Any:
        """Access the weights of the haiku model as a nested dict.

        Return type is any due to `jax.tree_util.tree_unflatten`.
        """
        assert self._treedef is not None, "uninitialized JaxModel"
        return jax.tree_util.tree_unflatten(self._treedef, self._params)

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: JaxNumpyVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector weights.")
        coefs_copy = deepcopy(weights.coefs)
        self._params = [
            jax.device_put(v, self._device) for v in coefs_copy.values()
        ]

    def _forward(
        self,
        params: hk.Params,
        rng: jax.Array,
        batch: JaxBatch
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
        y_pred = self._transformed_model.apply(params, rng, inputs)
        s_loss = self._loss_fn(y_pred, y_true)
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
        assert self._treedef is not None, "uninitialized JaxModel"
        inputs = self._unpack_batch(batch)
        params = jax.tree_util.tree_unflatten(self._treedef, self._params)
        # Run the forward and backward passes to compute gradients.
        grads = self._grad_fn(params, next(self._rng_gen), inputs)
        # Flatten the gradients and return them in a Vector container
        flat_grad = jax.tree_util.tree_flatten(grads)
        grads = {str(k): v for k, v in enumerate(flat_grad[0])}
        return JaxNumpyVector(grads)

    @functools.cached_property
    def _grad_fn(self) -> (
        Callable[[hk.Params, jax.Array, JaxBatch], hk.Params]
    ):
        """Lazy-built jax function to compute batch-averaged gradients."""
        return jax.jit(jax.grad(self._forward))

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: float,
    ) -> JaxNumpyVector:
        """Compute and return sample-wise clipped, batch-averaged gradients."""
        # Unpack input batch and unflatten model parameters.
        assert self._treedef is not None, "uninitialized JaxModel"
        inputs = self._unpack_batch(batch)
        params = jax.tree_util.tree_unflatten(self._treedef, self._params)
        # Get flattened, per-sample, clipped gradients and aggregate them.
        clipped_grads = self._clipped_grad_fn(
            params, next(self._rng_gen), inputs, max_norm
        )
        grads = [g.sum(0) for g in clipped_grads]
        # Return them in a Vector container.
        return JaxNumpyVector({str(k): v for k, v in enumerate(grads)})

    @functools.cached_property
    def _clipped_grad_fn(self) -> (
        Callable[[hk.Params, jax.Array, JaxBatch, float], jax.Array]
    ):
        """Lazy-built jax function to compute clipped sample-wise gradients."""

        def clipped_grad_fn(
            params: hk.Params, rng: jax.Array, batch: JaxBatch, max_norm: float
        ) -> List[jax.Array]:
            """Compute and clip gradients wrt parameters for a sample."""
            inputs, y_pred, s_wght = batch
            data = (inputs, y_pred, None)
            grads = jax.grad(self._forward)(params, rng, data)
            grads_flat = [
                grad / jnp.maximum(jnp.linalg.norm(grad) / max_norm, 1.0)
                for grad in jax.tree_util.tree_leaves(grads)
            ]
            if s_wght is not None:
                grads_flat = [g * s_wght for g in grads_flat]
            return grads_flat

        in_axes = [None, None, 0, None]  # map on inputs' first dimension
        return jax.jit(jax.vmap(clipped_grad_fn, in_axes))

    @staticmethod
    def _unpack_batch(batch: Batch) -> JaxBatch:
        """Unpack and enforce jnp.array conversion to an input data batch."""

        def convert(data: Any) -> Optional[jax.Array]:
            if isinstance(data, (list, tuple)):
                if len(data) == 1:
                    data = data[0]
                else:
                    raise TypeError(
                        "HaikuModel does not support multi-arrays inputs."
                    )
            if (data is None) or isinstance(data, jax.Array):
                return data
            if isinstance(data, np.ndarray):
                return jnp.array(data)  # pylint: disable=no-member
            raise TypeError("HaikuModel requires numpy or jax.numpy data.")

        # Convert batched data to jax Arrays.
        inputs, y_true, s_wght = batch
        output = [convert(inputs), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: JaxNumpyVector,
    ) -> None:
        if not isinstance(updates, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector updates.")
        try:
            for key, upd in updates.coefs.items():
                self._params[int(key)] += upd
        except KeyError as exc:
            raise KeyError(
                "Invalid model parameter name(s) found in updates."
            ) from exc

    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],]:
        inputs, y_true, s_wght = self._unpack_batch(batch)
        if y_true is None:
            raise TypeError(
                "`HaikuModel.compute_batch_predictions` received a "
                "batch with `y_true=None`, which is unsupported. Please "
                "correct the inputs, or override this method to support "
                "creating labels from the base inputs."
            )
        assert self._treedef is not None, "uninitialized JaxModel"
        params = jax.tree_util.tree_unflatten(self._treedef, self._params)
        y_pred = np.asarray(
            self._transformed_model.apply(params, next(self._rng_gen), inputs)
        )
        y_true = np.asarray(y_true)  # type: ignore
        if isinstance(s_wght, jax.Array):
            s_wght = np.asarray(s_wght)  # type: ignore
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
            self._params = jax.device_put(self._params, self._device)
