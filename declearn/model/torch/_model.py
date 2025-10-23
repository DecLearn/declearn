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

"""Model subclass to wrap PyTorch models."""

import functools
import io
import warnings
from typing import Any, Dict, List, Optional, Self, Set, Tuple

import numpy as np
import torch

from declearn.model._utils import raise_on_stringsets_mismatch
from declearn.model.api import Model
from declearn.model.torch._samplewise import (
    GetGradientsFunction,
    build_samplewise_grads_fn,
)
from declearn.model.torch._vector import TorchVector
from declearn.model.torch.utils import AutoDeviceModule, select_device
from declearn.typing import Batch
from declearn.utils import DevicePolicy, get_device_policy, register_type

__all__ = [
    "TorchModel",
]


@register_type(name="TorchModel", group="Model")
class TorchModel(Model):
    """Model wrapper for PyTorch Model instances.

    This `Model` subclass is designed to wrap a `torch.nn.Module` instance
    to be trained federatively.

    Notes regarding device management (CPU, GPU, etc.):

    - By default torch operates on CPU, and it does not automatically move
      tensors between devices. This means users have to be careful where
      tensors are placed to avoid operations between tensors on different
      devices, leading to runtime errors.
    - Our `TorchModel` instead consults the global device-placement policy
      (via `declearn.utils.get_device_policy`), places the wrapped torch
      modules' weights there, and automates the placement of input data on
      the same device as the wrapped model.
    - Note that if the global device-placement policy is updated, this will
      only be propagated to existing instances by manually calling their
      `update_device_policy` method.
    - You may consult the device policy currently enforced by a TorchModel
      instance by accessing its `device_policy` property.

    Notes regarding `torch.compile` support (torch >=2.0):

    - If you want the wrapped model to be optimized via `torch.compile`, it
      should be so _prior_ to being wrapped using `TorchModel`.
    - The compilation will not be used when computing sample-wise-clipped
      gradients, as `torch.func` and `torch.compile` do not play along yet.
    - The information that the module was compiled will be saved as part of
      the `TorchModel` config, so that using `TorchModel.from_config` will
      trigger it again when possible; this is however limited to calling
      `torch.compile`, meaning that any other argument will be lost.
    - Note that the former point notably affects the way clients will run
      a server-emitted `TorchModel` as part of a FL process: client that
      run Torch 1.X will be able to use the un-optimized module, while
      clients running Torch 2.0 will use compilation, but in a rather crude
      flavor, that may not be suitable for some specific/advanced cases.
    - Enhanced support for `torch.compile` is on the roadmap. If you run
      into issues and/or have requests or advice on that topic, feel free
      to let us know by contacting us via mail or GitLab.
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
            altered when wrapped. It must expect `y_pred` and `y_true`
            as input arguments (in that order) and will be used to get
            sample-wise loss values (by removing any reduction scheme).
        """
        # Type-check the input model.
        if not isinstance(model, torch.nn.Module):
            raise TypeError("'model' should be a torch.nn.Module instance.")
        # Select the device where to place computations, and wrap the model.
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        super().__init__(AutoDeviceModule(model, device=device))
        # Assign loss module and set it not to reduce sample-wise values.
        if not isinstance(loss, torch.nn.Module):
            raise TypeError("'loss' should be a torch.nn.Module instance.")
        loss.reduction = "none"  # type: ignore
        self._loss_fn = AutoDeviceModule(loss, device=device)
        # Detect torch-compiled models and extract underlying module.
        self._raw_model = self._model
        if hasattr(torch, "compile") and hasattr(model, "_orig_mod"):
            self._raw_model = AutoDeviceModule(
                module=model._orig_mod,
                device=self._model.device,
            )

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
            "PyTorch JSON serialization relies on pickle, which may be unsafe.",
            stacklevel=2,
        )
        with io.BytesIO() as buffer:
            torch.save(self._raw_model.module, buffer)
            model = buffer.getbuffer().hex()
        with io.BytesIO() as buffer:
            torch.save(self._loss_fn.module, buffer)
            loss = buffer.getbuffer().hex()
        return {
            "model": model,
            "loss": loss,
            "compile": self._raw_model is not self._model,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a TorchModel from a configuration dict."""
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = torch.load(buffer, weights_only=False)
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = torch.load(buffer, weights_only=False)
        if config.get("compile", False) and hasattr(torch, "compile"):
            model = torch.compile(model)
        return cls(model=model, loss=loss)

    def get_weights(
        self,
        trainable: bool = False,
    ) -> TorchVector:
        params = self._raw_model.named_parameters()
        if trainable:
            weights = {k: p.data for k, p in params if p.requires_grad}
        else:
            weights = {k: p.data for k, p in params}
        # Note: calling `tensor.clone()` to return a copy rather than a view.
        return TorchVector({k: t.detach().clone() for k, t in weights.items()})

    def set_weights(
        self,
        weights: TorchVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, TorchVector):
            raise TypeError("TorchModel requires TorchVector weights.")
        self._verify_weights_compatibility(weights, trainable=trainable)
        if trainable:
            state_dict = self._raw_model.state_dict()
            state_dict.update(weights.coefs)
        else:
            state_dict = weights.coefs
        # NOTE: this preserves the device placement of current states
        self._raw_model.load_state_dict(state_dict)

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
        KeyError
            In case some expected keys are missing, or additional keys
            are present. Be verbose about the identified mismatch(es).
        """
        params = self._raw_model.named_parameters()
        received = set(vector.coefs)
        expected = {n for n, p in params if (not trainable) or p.requires_grad}
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
        self._loss_history.append(float(loss.detach().cpu().numpy().mean()))
        # Collect weights' gradients and return them in a Vector container.
        grads = {
            k: p.grad.detach().clone()
            for k, p in self._raw_model.named_parameters()
            if p.requires_grad
        }
        return TorchVector(grads)

    @staticmethod
    def _unpack_batch(
        batch: Batch,
    ) -> Tuple[
        List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Unpack and enforce Tensor conversion to an input data batch."""

        # fmt: off
        # Define an array-to-tensor conversion routine.
        def convert(data: Any) -> Optional[torch.Tensor]:
            if (data is None) or isinstance(data, torch.Tensor):
                return data
            return torch.from_numpy(data)
        # Ensure inputs is a list.
        inputs, y_true, s_wght = batch
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        # Ensure output data was converted to Tensor.
        output = (list(map(convert, inputs)), convert(y_true), convert(s_wght))
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

    def _compute_clipped_gradients(
        self,
        batch: Batch,
        max_norm: float,
    ) -> TorchVector:
        """Compute and return batch-averaged sample-wise-clipped gradients."""
        # Compute sample-wise clipped gradients, using functional torch.
        grads = self._compute_samplewise_gradients(batch, clip=max_norm)
        # Batch-average the resulting sample-wise gradients.
        return TorchVector(
            {name: tensor.mean(dim=0) for name, tensor in grads.coefs.items()}
        )

    def _compute_samplewise_gradients(
        self,
        batch: Batch,
        clip: Optional[float],
    ) -> TorchVector:
        """Compute and return stacked sample-wise gradients over a batch."""
        inputs, y_true, s_wght = self._unpack_batch(batch)
        grads_fn = self._build_samplewise_grads_fn(
            inputs=len(inputs),
            y_true=(y_true is not None),
            s_wght=(s_wght is not None),
        )
        with torch.no_grad():
            grads, loss = grads_fn(inputs, y_true, s_wght, clip=clip)  # type: ignore
            self._loss_history.append(float(loss.cpu().numpy().mean()))
        return TorchVector(grads)

    # TODO: move usage of lru_cache to prevent memory leaks
    # (see: https://docs.astral.sh/ruff/rules/cached-instance-method/)
    @functools.lru_cache  # noqa: B019
    def _build_samplewise_grads_fn(
        self,
        inputs: int,
        y_true: bool,
        s_wght: bool,
    ) -> GetGradientsFunction:
        """Build an optimizer sample-wise gradients-computation function.

        This function is cached, i.e. repeated calls with the same parameters
        will return the same object - enabling to reduce runtime costs due to
        building and (when available) compiling the output function.

        Returns
        -------
        grads_fn: callable[[inputs, y_true, s_wght, clip], grads]
            Function to efficiently compute and return sample-wise gradients
            wrt trainable model parameters based on a batch of inputs, with
            opt. clipping based on a maximum l2-norm value `clip`.
        """
        # NOTE: torch.func is not compatible with torch.compile yet
        return build_samplewise_grads_fn(
            self._raw_model, self._loss_fn, inputs, y_true, s_wght
        )

    def apply_updates(
        self,
        updates: TorchVector,
    ) -> None:
        if not isinstance(updates, TorchVector):
            raise TypeError("TorchModel requires TorchVector updates.")
        self._verify_weights_compatibility(updates, trainable=True)
        with torch.no_grad():
            for key, upd in updates.coefs.items():
                tns = self._raw_model.get_parameter(key)
                tns.add_(upd.to(tns.device))

    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        inputs, y_true, s_wght = self._unpack_batch(batch)
        if y_true is None:
            raise TypeError(
                "`TorchModel.compute_batch_predictions` received a "
                "batch with `y_true=None`, which is unsupported. Please "
                "correct the inputs, or override this method to support "
                "creating labels from the base inputs."
            )
        self._model.eval()
        self._handle_torch_compile_eval_issue(inputs)
        with torch.no_grad():
            y_pred = self._model(*inputs).cpu().numpy()
        y_true = y_true.cpu().numpy()
        s_wght = None if s_wght is None else s_wght.cpu().numpy()
        return y_true, y_pred, s_wght  # type: ignore

    def _handle_torch_compile_eval_issue(
        self,
        inputs: List[torch.Tensor],
    ) -> None:
        """Clumsily handle issues with `torch.compile` and `torch.no_grad`.

        As of Torch 2.0.1, running a compiled model's first forward pass
        within a `torch.no_grad` context results in the model's future
        weights updates not being properly taken into account.

        Therefore, when wrapping a compiled model, this method runs a lost
        forward pass outside of a no-grad context on its first call (later
        it does nothing).
        """
        if (self._raw_model is self._model) or hasattr(self, "__eval_called"):
            return
        self._model(*inputs)
        setattr(self, "__eval_called", True)

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        tns_pred = torch.from_numpy(y_pred)
        tns_true = torch.from_numpy(y_true)
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
