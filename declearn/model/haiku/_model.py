# coding: utf-8

"""Model subclass to wrap Haiku models."""

import io
import warnings
from copy import deepcopy
from random import SystemRandom
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import joblib  # type: ignore
import numpy as np
from jax import grad, jit, vmap
from jax.tree_util import tree_flatten, tree_unflatten
from jaxtyping import Array  # future: import ArrayLike from jax.typing
from typing_extensions import Self

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model
from declearn.model.haiku import JaxNumpyVector
from declearn.model.haiku.utils import select_device
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type

SEED = int(SystemRandom().random() * 10e6)

# alias for unpacked Batch structures, converted to jax objects
# input, optional label, optional weights
JaxBatch = Tuple[List[Array], Optional[Array], Optional[Array]]


@register_type(name="HaikuModel", group="Model")
class HaikuModel(Model):
    """
    Model wrapper for Haiku Model instances.

    This `Model` subclass is designed to wrap a `hk.Module`
    instance to be learned federatively.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model: Callable,
        loss: Callable,
        seed: int = SEED,
    ) -> None:
        """
        Instantiate a Model interface wrapping a torch.nn.Module.

        Parameters
        ----------
        model: Callable
            A function encapsulating a hk.Module such that `model(x)`
            returns `hk.Module(x)
        loss: Callable
            A user-defined loss function
        seed: Optional int
            Random seed used to initialize the haiku-wrapped Pseudo-random
            number generator. If none is provided, use an integer between
            0 and 10e6 provided by SystemRandom.
        """
        super().__init__(model)
        # Assign loss module.
        self._loss_fn = loss
        # Get pure functions from haiku transform
        self._model_fn = model
        self._transformed_model = hk.transform(model)
        # Select the device where to place computations.
        policy = get_device_policy()
        self._device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Create model state attributes
        self._params_leaves = None  # type: Optional[List[Array]]
        self._params_treedef = None  # type: Optional[Any]
        # Initialize the PRNG
        self._rng_gen = hk.PRNGSequence(seed)
        # Initialized and data_info utils
        self._initialized = False
        self.data_info = {}

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
        flat_params = tree_flatten(params)
        self._params_treedef = deepcopy(flat_params[1])
        self._params_leaves = flat_params[0]
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
        params = {
            str(k): v
            for k, v in enumerate(self._params_leaves)  # type: ignore
        }
        return JaxNumpyVector(params)

    def get_named_weights(self) -> Any:
        """Utility function to access the weights of the haiku model as a
        nested dict, using the appropriate naming. Return type is any to
        follow the typing of the `jax.tree_util.tree_unflatten`.
        """
        return tree_unflatten(
            self._params_treedef, self._params_leaves  # type: ignore
        )

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: JaxNumpyVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector weights.")
        coefs_copy = deepcopy(weights.coefs)
        self._params_leaves = [
            jax.device_put(v, self._device) for v in coefs_copy.values()
        ]

    def _forward(
        self,
        params: Dict[str, Array],
        inputs: Array,
        y_true: Optional[Array] = None,
        s_wght: Optional[Array] = None,
        rng: Optional[Array] = None,
    ) -> Array:
        """#TODO Document order of arguments"""
        # pylint: disable=too-many-arguments
        y_pred = self._transformed_model.apply(
            params, rng, inputs
        )  # list-inputs : *inputs
        loss = self._compute_loss(y_pred, y_true, s_wght)
        return jnp.mean(loss)

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
        # Unflatten the parameters, run forward to compute gradients.
        params = tree_unflatten(
            self._params_treedef, self._params_leaves  # type: ignore
        )
        grad_fn = jit(grad(self._forward))
        inputs, y_true, s_wght = self._unpack_batch(batch)
        grads = grad_fn(params, inputs, y_true, s_wght, next(self._rng_gen))
        # Flatten the gradients and return them in a Vector container
        flat_grad = tree_flatten(grads)
        grads = {str(k): v for k, v in enumerate(flat_grad[0])}
        return JaxNumpyVector(grads)

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> JaxNumpyVector:
        """Compute and return smpla-wise clipped, batch-averaged gradients
        of trainable weights."""
        # Unflatten parameters, run forward to compute per sample gradients.
        params = tree_unflatten(
            self._params_treedef, self._params_leaves  # type: ignore
        )
        inputs, y_true, s_wght = self._unpack_batch(batch)
        # Get  flatten, per-sample, clipped gradients and aggregate them
        in_axes = [
            None,
            None,
            0,
            None if y_true is None else 0,
            None if s_wght is None else 0,
            None,
        ]
        grad_fn = jit(vmap(self._clipped_grad, in_axes))
        clipped_grads = grad_fn(
            params, next(self._rng_gen), inputs, y_true, s_wght, max_norm
        )
        grads = [g.sum(0) for g in clipped_grads]
        # Return them in a Vector container.
        return JaxNumpyVector({str(k): v for k, v in enumerate(grads)})

    def _clipped_grad(
        self,
        params: Dict[str, Array],
        rng: Array,
        inputs: Array,
        y_true: Optional[Array],
        s_wght: Optional[Array],
        max_norm: Optional[float] = None,
    ) -> List[Array]:
        """Evaluate gradient for a single-example batch and clip its
        grad norm."""
        # pylint: disable=too-many-arguments
        grads = grad(self._forward)(params, inputs, y_true, s_wght, rng)
        nonempty_grads = jax.tree_util.tree_leaves(grads)
        grad_norm = [jnp.linalg.norm(g) for g in nonempty_grads]
        divisor = [jnp.maximum(g / max_norm, 1.0) for g in grad_norm]
        clipped_grads = [
            nonempty_grads[i] / divisor[i] for i in range(len(grad_norm))
        ]
        return clipped_grads

    @staticmethod
    def _unpack_batch(batch: Batch) -> JaxBatch:
        """Unpack and enforce jnp.array conversion to an input data batch."""

        def convert(data: Any) -> Optional[Array]:
            if (data is None) or isinstance(data, Array):
                return data
            if isinstance(data, np.ndarray):
                return jnp.array(data)  # pylint: disable=no-member
            raise TypeError("HaikuModel requires numpy or jax.numpy data.")

        # Ensure inputs is a list.
        inputs, y_true, s_wght = batch
        # Ensure output data was converted to Tensor.
        output = [convert(inputs), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def _compute_loss(
        self,
        y_pred: Array,
        y_true: Optional[Array],
        s_wght: Optional[Array] = None,
    ) -> Array:
        """Compute the average (opt. weighted) loss over given predictions."""
        loss = self._loss_fn(y_pred, y_true)
        if s_wght is not None:
            loss = loss * s_wght
        return loss  # type: ignore

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: JaxNumpyVector,
    ) -> None:
        if not isinstance(updates, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector updates.")
        try:
            for key, upd in updates.coefs.items():
                self._params_leaves[int(key)] += upd  # type: ignore
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
        params = tree_unflatten(
            self._params_treedef, self._params_leaves  # type: ignore
        )
        y_pred = np.asarray(self._transformed_model.apply(params, next(self._rng_gen),inputs))
        y_true = np.asarray(y_true)  # type: ignore
        if isinstance(s_wght, Array):
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
        # Select the device to use based on the provided or global policy.
        if policy is None:
            policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # When needed, re-create the model to force moving it to the device.
        if self._device is not device:
            self._device = device
            self._params_leaves = jax.device_put(
                self._params_leaves, self._device
            )
