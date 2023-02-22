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

"""Model subclass to wrap PyTorch models."""

import io
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import functorch  # type: ignore
import numpy as np
import torch
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Model
from declearn.model.torch.utils import AutoDeviceModule, select_device
from declearn.model.torch._vector import TorchVector
from declearn.model._utils import raise_on_stringsets_mismatch
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type


__all__ = [
    "TorchModel",
]


# alias for unpacked Batch structures, converted to torch.Tensor objects
TensorBatch = Tuple[
    List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]


@register_type(name="TorchModel", group="Model")
class TorchModel(Model):
    """Model wrapper for PyTorch Model instances.

    This `Model` subclass is designed to wrap a `torch.nn.Module` instance
    to be trained federatively.

    Notes regarding device management (CPU, GPU, etc.):
    * By default torch operates on CPU, and it does not automatically move
      tensors between devices. This means users have to be careful where
      tensors are placed to avoid operations between tensors on different
      devices, leading to runtime errors.
    * Our `TorchModel` instead consults the global device-placement policy
      (via `declearn.utils.get_device_policy`), places the wrapped torch
      modules' weights there, and automates the placement of input data on
      the same device as the wrapped model.
    * Note that if the global device-placement policy is updated, this will
      only be propagated to existing instances by manually calling their
      `update_device_policy` method.
    * You may consult the device policy currently enforced by a TorchModel
      instance by accessing its `device_policy` property.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
    ) -> None:
        """Instantiate a Model interface wrapping a torch.nn.Module.

        Parameters
        ----------
        model: torch.nn.Module
            Torch Module instance that defines the model's architecture.
        loss: torch.nn.Module
            Torch Module instance that defines the model's loss, that
            is to be minimized through training. Note that it will be
            altered when wrapped.
        """
        # Type-check the input model.
        if not isinstance(model, torch.nn.Module):
            raise TypeError("'model' should be a torch.nn.Module instance.")
        # Select the device where to place computations, and wrap the model.
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        model = AutoDeviceModule(model, device=device)
        super().__init__(model)
        # Assign loss module and set it not to reduce sample-wise values.
        if not isinstance(loss, torch.nn.Module):
            raise TypeError("'loss' should be a torch.nn.Module instance.")
        loss.reduction = "none"  # type: ignore
        self._loss_fn = AutoDeviceModule(loss, device=device)
        # Compute and assign a functional version of the model.
        self._func_model = functorch.make_functional(self._model)[0]

    @property
    def device_policy(
        self,
    ) -> DevicePolicy:
        device = self._model.device
        return DevicePolicy(gpu=(device.type == "cuda"), idx=device.index)

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        return set()

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        return None

    def get_config(
        self,
    ) -> Dict[str, Any]:
        warnings.warn(
            "PyTorch JSON serialization relies on pickle, which may be unsafe."
        )
        with io.BytesIO() as buffer:
            torch.save(self._model.module, buffer)
            model = buffer.getbuffer().hex()
        with io.BytesIO() as buffer:
            torch.save(self._loss_fn.module, buffer)
            loss = buffer.getbuffer().hex()
        return {"model": model, "loss": loss}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a TorchModel from a configuration dict."""
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = torch.load(buffer)
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = torch.load(buffer)
        return cls(model=model, loss=loss)

    def get_weights(
        self,
        trainable: bool = False,
    ) -> TorchVector:
        params = self._model.named_parameters()
        if trainable:
            weights = {k: p.data for k, p in params if p.requires_grad}
        else:
            weights = {k: p.data for k, p in params}
        # Note: calling `tensor.clone()` to return a copy rather than a view.
        return TorchVector({k: t.detach().clone() for k, t in weights.items()})

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: TorchVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, TorchVector):
            raise TypeError("TorchModel requires TorchVector weights.")
        self._verify_weights_compatibility(weights, trainable=trainable)
        if trainable:
            state_dict = self._model.state_dict()
            state_dict.update(weights.coefs)
        else:
            state_dict = weights.coefs
        # NOTE: this preserves the device placement of current states
        self._model.load_state_dict(state_dict)

    def _verify_weights_compatibility(
        self,
        vector: TorchVector,
        trainable: bool = False,
    ) -> None:
        """Verify that a vector has the same names as the model's weights.

        Parameters
        ----------
        vector: TorchVector
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
        expected = {
            name
            for name, param in self._model.named_parameters()
            if (not trainable) or param.requires_grad
        }
        raise_on_stringsets_mismatch(
            received, expected, context="model weights"
        )

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> TorchVector:
        self._model.train()
        if max_norm:
            return self._compute_clipped_gradients(batch, max_norm)
        return self._compute_batch_gradients(batch)

    def _compute_batch_gradients(
        self,
        batch: Batch,
    ) -> TorchVector:
        """Compute and return batch-averaged gradients of trainable weights."""
        # Unpack inputs and clear gradients' history.
        inputs, y_true, s_wght = self._unpack_batch(batch)
        self._model.zero_grad()
        # Run the forward and backward pass to compute gradients.
        y_pred = self._model(*inputs)
        loss = self._compute_loss(y_pred, y_true, s_wght)
        loss.backward()
        # Collect weights' gradients and return them in a Vector container.
        grads = {
            k: p.grad.detach().clone()
            for k, p in self._model.named_parameters()
            if p.requires_grad
        }
        return TorchVector(grads)

    @staticmethod
    def _unpack_batch(batch: Batch) -> TensorBatch:
        """Unpack and enforce Tensor conversion to an input data batch."""
        # fmt: off
        # Define an array-to-tensor conversion routine.
        def convert(data: Any) -> Optional[torch.Tensor]:
            if (data is None) or isinstance(data, torch.Tensor):
                return data
            return torch.from_numpy(data)  # pylint: disable=no-member
        # Ensure inputs is a list.
        inputs, y_true, s_wght = batch
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        # Ensure output data was converted to Tensor.
        output = [list(map(convert, inputs)), convert(y_true), convert(s_wght)]
        return output  # type: ignore

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor],
        s_wght: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the average (opt. weighted) loss over given predictions."""
        loss = self._loss_fn(y_pred, y_true)
        if s_wght is not None:
            loss.mul_(s_wght.to(loss.device))
        return loss.mean()

    def _compute_samplewise_gradients(
        self,
        batch: Batch,
    ) -> TorchVector:
        """Compute and return stacked sample-wise gradients from a batch."""
        # Delegate preparation of the gradients-computing function.
        # fmt: off
        grads_fn, data, params, pnames, in_axes = (
            self._prepare_samplewise_gradients_computations(batch)
        )
        # Vectorize the function to compute sample-wise gradients.
        with torch.no_grad():
            grads = functorch.vmap(grads_fn, in_axes)(*data, *params)
        # Wrap the results into a TorchVector and return it.
        return TorchVector(dict(zip(pnames, grads)))

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: float,
    ) -> TorchVector:
        """Compute and return batch-averaged sample-wise-clipped gradients."""
        # Delegate preparation of the gradients-computing function.
        # fmt: off
        grads_fn, data, params, pnames, in_axes = (
            self._prepare_samplewise_gradients_computations(batch)
        )
        # Compose it to clip output gradients on the way.
        def clipped_grads_fn(inputs, y_true, s_wght, *params):
            grads = grads_fn(inputs, y_true, None, *params)
            for grad in grads:
                # future: use torch.linalg.norm when supported by functorch
                norm = torch.norm(grad, p=2, keepdim=True)
                # false-positive; pylint: disable=no-member
                grad.mul_(torch.clamp(max_norm / norm, max=1))
                if s_wght is not None:
                    grad.mul_(s_wght.to(grad.device))
            return grads
        # Vectorize the function to compute sample-wise clipped gradients.
        with torch.no_grad():
            grads = functorch.vmap(clipped_grads_fn, in_axes)(*data, *params)
        # Wrap batch-averaged results into a TorchVector and return it.
        return TorchVector(
            {name: grad.mean(dim=0) for name, grad in zip(pnames, grads)}
        )

    def _prepare_samplewise_gradients_computations(
        self,
        batch: Batch,
    ) -> Tuple[
        Callable[..., List[torch.Tensor]],
        TensorBatch,
        List[torch.nn.Parameter],
        List[str],
        Tuple[Any, ...],
    ]:
        """Prepare a function an parameters to compute sample-wise gradients.

        Note: this method is merely implemented as a way to avoid code
        redundancies between the `_compute_samplewise_gradients` method
        and the `_compute_clipped_gradients` ones.

        Parameters
        ----------
        batch: declearn.typing.Batch
            Batch structure wrapping the input data, target labels and
            optional sample weights based on which to compute gradients.

        Returns
        -------
        grads_fn: function(*data, *params) -> List[torch.Tensor]
            Functorch-issued gradients computation function.
        data: tuple([torch.Tensor], torch.Tensor, torch.Tensor or None)
            Tensor-converted data unpacked from `batch`.
        params: list[torch.nn.Parameter]
            Input parameters of the model, some of which require grads.
        pnames: list[str]
            Names of the parameters that require gradients.
        in_axes: tuple(...)
            Prepared `in_axes` parameter to `functorch.vmap`, suitable
            to distribute `grads_fn` (or any compose that shares its
            input signature) over a batch so as to compute sample-wise
            gradients in a computationally-efficient manner.
        """
        # fmt: off
        # Unpack and validate inputs.
        data = (inputs, y_true, s_wght) = self._unpack_batch(batch)
        # Gather parameters and list those that require gradients.
        idxgrd = []  # type: List[int]
        pnames = []  # type: List[str]
        params = []  # type: List[torch.nn.Parameter]
        for idx, (name, param) in enumerate(self._model.named_parameters()):
            params.append(param)
            if param.requires_grad:
                pnames.append(name)
                idxgrd.append(idx)
        # Define a differentiable function wrapping the forward pass.
        def forward(inputs, y_true, s_wght, *params):
            y_pred = self._func_model(params, *inputs)
            return self._compute_loss(y_pred, y_true, s_wght)
        # Transform it into a sample-wise-gradients-computing function.
        grads_fn = functorch.grad(forward, argnums=tuple(i+3 for i in idxgrd))
        # Prepare `functools.vmap` parameter to slice through data and params.
        in_axes = [
            [0] * len(inputs),
            None if y_true is None else 0,
            None if s_wght is None else 0,
        ]
        in_axes.extend([None] * len(params))
        # Return all this prepared material.
        return grads_fn, data, params, pnames, tuple(in_axes)

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: TorchVector,
    ) -> None:
        if not isinstance(updates, TorchVector):
            raise TypeError("TorchModel requires TorchVector updates.")
        self._verify_weights_compatibility(updates, trainable=True)
        with torch.no_grad():
            for key, upd in updates.coefs.items():
                tns = self._model.get_parameter(key)
                tns.add_(upd.to(tns.device))

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
        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(*inputs).cpu().numpy()
        y_true = y_true.cpu().numpy()
        s_wght = None if s_wght is None else s_wght.cpu().numpy()
        return y_true, y_pred, s_wght  # type: ignore

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        tns_pred = torch.from_numpy(y_pred)  # pylint: disable=no-member
        tns_true = torch.from_numpy(y_true)  # pylint: disable=no-member
        s_loss = self._loss_fn(tns_pred, tns_true)
        return s_loss.cpu().numpy().squeeze()

    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        # Select the device to use based on the provided or global policy.
        if policy is None:
            policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Place the wrapped model and loss function modules on that device.
        self._model.set_device(device)
        self._loss_fn.set_device(device)
