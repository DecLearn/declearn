# coding: utf-8

"""Model subclass to wrap PyTorch models.

# TODO

Plan : 
- [ ] Do minimal functionality, untyped unoptimized
    - [ ] Extend `_unpack_batch` type support
    - [ ] Add kwargs to _compute_loss and loss_function to cover all cases of optax loss
    - [x] I need to have a testing file where I can incrementally work rather than blindly go forward
    - [ ] Accept optax losses, do the mapping on loss inputs (inspect.get_signature)
    - [ ] Look at refine flattening of params https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
    - [ ] Test including regularizer -> sign issue ?
    - [ ] Solve flattening/unflattening issue
- [ ] add clipping : use optax clipping per sample
- [ ] Add jit
- [ ] Add proper randomizatoin 
- [ ] Add typing (chex vs jnp)
- [ ] check hopw to make model genral enough to handle statefulness, multiinput, etc


Questions 
- init assumes input is numpy array > check potential issues
- Check fixed input size limitations on haiku and interaction with poisson sampling\
- Import for loss : use the torch optimodule, will need to depack 
- unpack batch for torch implicitely requires np data
-  @abstractmethod loss_function() requires you output np.array
- What happens when y_true not there in compute loss, since no default to noneq
 - why output list
 - Inspect.get-source, inspect.get_user 
 - Where to use pytrees, besides manipulating _params ? DO I need to addd support in jax numpy vector ?
 - Can we do better than get_buffer.hex in terms of weights 
 - How do deal with multuiple model input (data_type at init, inputs as list )
 - RIght now, model needs to be initialized to use set_weights, not just __init_-
 - are model params always dict of dict or can they be more nested than this ? For typing purposes, 
 see for instance get_named_weights


Notes :
- how to make fake input and does type count ? 
    - Do I absolutely need the corrsct batch size ? no, see for instance main() in 
    https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_vae.html
- get inspiration from functorch example 
- can I assign my jax functions to self.stuff ? 
    - I think I can, but at execution need to be functionnaly pure,
    so should not update things in-place at any point but output things clearly 
    - so do something like self.state = self.function(self.state)
- How to handle different data inputs ? Check how it is done for other models, 
as in should I add a conversion to jnp arr on top of my models ?
- Dealing with mixed prececsion > will first default to float32 type for params but 
need to check in more details down the line 
- Use OPtax defined-losses, but provide the option to 
- Will need t TrainingState class like the haiku mnist example ?
- check if staeteful is needed ? I think yes because we have params. 
Looking at provided example, it looks like no. Let me try without and see. I need to revise this 
down the ine by thinking about how to dealwith models that ned to be stateful
- Should I init with a numpy of jax array ? Found both in docs, I'll use jnp just in case
- How to deal with JaxBatxch ? why does the _unpack utility deal with list for  input ?
- Need to create a function model https://lightrun.com/answers/deepmind-dm-haiku-allow-creating-module-instances-outside-hktransform
- How to go about saving : https://github.com/deepmind/dm-haiku/issues/18

Issue with flattening: 
- Cost of flattening O(n) in terms of space (need to copy), same inm terms of complexity
- Could be optimized by 
    - popping dict elements as I go, whre maximum buffer is one element of the gradient 
    - using the input of the tree flattening util, depending on its source code (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/pytree.cc, in c++)
- Keep the flat version > need to convert at every hk.apply operation, so grad and predict  
- Keep the nested version > need to convert at every Vector operation
- Selected solution : 
    - keep nested, flatten using jax implem, make deftree a class attribute 
    - Chekc if anny issue with deepcopies ?
    - Make sure no issues with abstract tracers when adding jit 

# todo check time for flatten vs unflatten 

"""

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
# TODO : check needed
# class TrainingState(NamedTuple):
#   """Container for the training state."""
#   params: hk.Params
#   rng: jnp.DeviceArray
#   step: jnp.DeviceArray

@register_type(name="HaikuModel", group="Model")
class HaikuModel(Model):
    """
    Model wrapper for Haiku Model instances.

    This `Model` subclass is designed to wrap a `hk.Module`
    instance to be learned federatively.
    """
    #TODO checktyping (use jnp.float?)
    #TODO add type checking of loss and model ?
    def __init__(
        self,
        model: Callable,
        loss: Callable,
    ) -> None:
        """
        # TODO 
        """
        super().__init__(model)
        # Assign loss module.
        self._loss_fn = loss
        # Get pure fucntions from model'
        self._model_fn = model
        self._transformed_model = hk.transform(model)
        # Store model state
        self._params_leaves = None
        self._params_treedef = None
        # Utility hidden attributes
        self._initialized = False

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

        # Mark the model as initialized.
        self._initialized = True #CHECK if useful

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
    # give the import string, using torch optimodule trick 

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "HaikuModel": # QUESTION is that self typing ?
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = joblib.load(buffer)
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = joblib.load(buffer)
        return cls(model=model, loss=loss)

    def get_weights(
        self,
    ) -> JaxNumpyVector:
        params = {k: p for k, p in enumerate(self._params_leaves)} 
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

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> JaxNumpyVector:
        # if max_norm:
        #     return self._compute_clipped_gradients(batch, max_norm)
        return self._compute_batch_gradients(batch)

    def _forward(self, params, inputs, y_true, s_wght):
        y_pred = self._transformed_model.apply(params,next(RAND_SEQ),*inputs) 
        loss = self._compute_loss(y_pred, y_true, s_wght)
        return loss 
    
    def _compute_batch_gradients(
        self,
        batch: Batch,
    ) -> JaxNumpyVector:
        """Compute and return batch-averaged gradients of trainable weights."""
        # Run the forward and backward pass to compute gradients.
        params = jax.tree_util.tree_unflatten(self._params_treedef,self._params_leaves)
        grads = jax.grad(self._forward)(params, *self._unpack_batch(batch))
        # Collect weights' gradients and return them in a Vector container.
        flat_grad = jax.tree_util.tree_flatten(grads)
        grads = {k: p for k, p in enumerate(flat_grad[0])} 
        return JaxNumpyVector(grads)
    
        # grads = {k: p for k, p in flatten_params(self._params,sep='/').items()} 
    

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
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        # Ensure output data was converted to Tensor.
        output = [list(map(convert, inputs)), convert(y_true), convert(s_wght)]
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
        return loss.mean()  # type: ignore

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
                "`TorchModel.compute_batch_predictions` received a "
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
