# coding: utf-8

"""Model subclass to wrap Haiku models."""

import io
import warnings
from copy import deepcopy
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Set, Tuple)

import haiku as hk
import jax
import jax.numpy as jnp
import joblib
import numpy as np
from jax import grad, jit, vmap

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model
from declearn.model.haiku import JaxNumpyVector
from declearn.typing import Batch
from declearn.utils import register_type

RAND_SEQ = hk.PRNGSequence(jax.random.PRNGKey(42))

# alias for unpacked Batch structures, converted to jax objects
# input, optional label, optional weights
JaxBatch = Tuple[
    List[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]
]

#TODO compare performance with list comprehension in or out the vmap for clipped grads
#TODO add type checking of loss and model at __init__
#TODO Allow for proper use of random seed at apply 

@register_type(name="HaikuModel", group="Model")
class HaikuModel(Model):
    """
    Model wrapper for Haiku Model instances.

    This `Model` subclass is designed to wrap a `hk.Module`
    instance to be learned federatively.
    """
    
    def __init__(
        self,
        model: Callable,
        loss: Callable,
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
        """
        super().__init__(model)
        # Assign loss module.
        self._loss_fn = loss
        # Get pure functions from haiku transform
        self._model_fn = model
        self._transformed_model = hk.transform(model)
        # Store model state
        self._params_leaves = None
        self._params_treedef = None

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        return {"input_shape","data_type"}

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        # Check that required fields are available and of valid type.
        data_info = aggregate_data_info([data_info], self.required_data_info)
        # initialize.
        params = self._transformed_model.init(
            next(RAND_SEQ),
            jnp.zeros((1,*data_info["input_shape"][1:]),*[x.__name__ for x in data_info["data_type"]]) #CHECK
        )
        flat_params = jax.tree_util.tree_flatten(params)
        self._params_treedef = flat_params[1]
        self._params_leaves = flat_params[0]

    def get_config(
        self,
    ) -> Dict[str, Any]:
        warnings.warn(
            "Our custom Haiku serialization relies on pickle, which may be unsafe."
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
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "HaikuModel": 
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = joblib.load(buffer)
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = joblib.load(buffer)
        return cls(model=model, loss=loss)

    def get_weights(
        self,
    ) -> JaxNumpyVector:
        params = dict(enumerate(self._params_leaves))
        return JaxNumpyVector(params)

    def get_named_weights(self) -> Any:
        """ Utility function to access the weights of the haiku model as a nested 
        dict, using the appropriate naming"""
        return jax.tree_util.tree_unflatten(self._params_treedef,self._params_leaves)

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: JaxNumpyVector,
    ) -> None:
        if not isinstance(weights, JaxNumpyVector):
            raise TypeError("HaikuModel requires JaxNumpyVector weights.")
        coefs_copy = deepcopy(weights.coefs)
        self._params_leaves = list(coefs_copy.values())

    def _forward(self,
                 params: Dict[str,jnp.ndarray],
                 rng: Optional[hk.PRNGSequence],
                 inputs: jnp.ndarray,
                 y_true: Optional[jnp.ndarray],
                 s_wght: Optional[jnp.ndarray],
        ):
        y_pred = self._transformed_model.apply(params,rng,inputs) #list-inputs : *inputs 
        loss = self._compute_loss(y_pred, y_true, s_wght)
        return loss.mean()

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
        params = jax.tree_util.tree_unflatten(self._params_treedef,self._params_leaves)
        grad_fn = jit(grad(self._forward))
        inputs, y_true, s_wght = self._unpack_batch(batch)
        grads = grad_fn(params, None, inputs, y_true, s_wght)
        # Flatten the gradients and return them in a Vector container
        flat_grad = jax.tree_util.tree_flatten(grads)
        grads = dict(enumerate(flat_grad[0]))
        return JaxNumpyVector(grads)

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> JaxNumpyVector:
        """Compute and return smpla-wise clipped, batch-averaged gradients of trainable weights."""
        # Unflatten the parameters, run forward to compute per sample gradients.
        params = jax.tree_util.tree_unflatten(self._params_treedef,self._params_leaves)
        inputs, y_true, s_wght = self._unpack_batch(batch)
        # Get  flatten, per-sample, clipped gradients and aggregate them
        in_axes = [
            None,
            0,
            None if y_true is None else 0,
            None if s_wght is None else 0,
            None,]
        grad_fn = jit(vmap(self._clipped_grad, in_axes))
        clipped_grads = grad_fn(params, inputs, y_true, s_wght, max_norm)
        grads = [g.sum(0) for g in clipped_grads] 
        # Return them in a Vector container.
        return JaxNumpyVector(dict(enumerate(grads)))
        
    def _clipped_grad(self, 
                params: Dict[str,jnp.ndarray],
                inputs: jnp.ndarray,
                y_true: Optional[jnp.ndarray],
                s_wght: Optional[jnp.ndarray],
                max_norm: Optional[float] = None,
        ):
        """Evaluate gradient for a single-example batch and clip its grad norm."""
        grads = grad(self._forward)(params, None, inputs, y_true, s_wght)
        nonempty_grads = jax.tree_util.tree_leaves(grads)
        grad_norm = [jnp.linalg.norm(g) for g in nonempty_grads] 
        divisor = [jnp.maximum(g / max_norm, 1.) for g in grad_norm]
        clipped_grads = [nonempty_grads[i] / divisor[i] for i in range(len(grad_norm))]
        return clipped_grads
    
    @staticmethod
    def _unpack_batch(batch: Batch) -> JaxBatch:
        """Unpack and enforce jnp.array conversion to an input data batch."""
        # fmt: off
        def convert(data: Any) -> jnp.ndarray:
            if (data is None) or isinstance(data, jnp.ndarray):
                return data
            elif isinstance(data, np.ndarray):
                return jnp.array(data)  # pylint: disable=no-member
            else :
                raise TypeError("HaikuModel requires numpy or jax.numpy data.")

        # Ensure inputs is a list.
        inputs, y_true, s_wght = batch
        # Ensure output data was converted to Tensor.
        output = [convert(inputs), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def _compute_loss(
        self,
        y_pred: jnp.array,
        y_true: Optional[jnp.array],
        s_wght: Optional[jnp.array] = None,
    ) -> jnp.array:
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
                self._params_leaves[key] += upd
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
        params = jax.tree_util.tree_unflatten(self._params_treedef,self._params_leaves)
        y_pred = self._transformed_model.apply(params,*inputs)
        y_true = np.array(y_true)
        s_wght = np.array(s_wght) if s_wght is not None else s_wght
        return y_true, y_pred, s_wght  # type: ignore

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        s_loss = self._loss_fn(jnp.array(y_pred), jnp.array(y_true))
        return np.array(s_loss).squeeze()
