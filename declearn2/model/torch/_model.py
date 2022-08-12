# coding: utf-8

"""Model subclass to wrap PyTorch models."""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from declearn2.model.api import Model, NumpyVector
from declearn2.model.torch._vector import TorchVector
from declearn2.typing import Batch
from declearn2.utils import register_type


@register_type(name="TorchModel", group="Model")
class TorchModel(Model):
    """Model wrapper for PyTorch Model instances.

    This `Model` subclass is designed to wrap a `torch.nn.Module`
    instance to be learned federatively.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            #metrics: ,
        ) -> None:
        """Instantiate a Model interface wrapping a 'model' object."""
        # Type-check the input Model and wrap it up.
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "'model' should be a torch.nn.Module instance."
            )
        super().__init__(model)
        # Compile the wrapped model and retain compilation arguments.
        self._loss_fn = loss
        self._loss_fn.reduction = 'none'  # type: ignore

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
        """Return the model's parameters as a JSON-serializable dict."""
        raise NotImplementedError(
            "PyTorch does not implement JSON serialization."
        )

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'Model':
        """Instantiate a model from a configuration dict."""
        raise NotImplementedError(
            "PyTorch does not implement JSON serialization."
        )

    def get_weights(
            self,
        ) -> NumpyVector:
        """Return the model's trainable weights."""
        return NumpyVector({
            key: tns.numpy().copy()  # NOTE: otherwise, view on Tensor data
            for key, tns in self._model.state_dict().items()
        })

    def set_weights(
            self,
            weights: NumpyVector,
        ) -> None:
        """Assign values to the model's trainable weights."""
        # false-positive on torch.from_numpy; pylint: disable=no-member
        self._model.load_state_dict({
            key: torch.from_numpy(arr) for key, arr in weights.coefs.items()
        })

    def compute_batch_gradients(
            self,
            batch: Batch,
        ) -> TorchVector:
        """Compute and return the model's gradients over a data batch."""
        # Unpack inputs and clear gradients' history.
        inputs, y_true, s_wght = self._unpack_batch(batch)
        self._model.zero_grad()
        # Run the forward and backward pass to compute gradients.
        y_pred = self._model(*inputs)
        loss = self._loss_fn(y_pred, y_true)
        if s_wght is not None:
            loss.mul_(s_wght)
        loss = loss.mean()
        loss.backward()
        # Collect weights' gradients and return them in a Vector container.
        return TorchVector({
            str(i): p.grad
            for i, p in enumerate(self._model.parameters())
            if p.grad is not None
        })

    @staticmethod
    def _unpack_batch(
            batch: Batch
        ) -> Tuple[
            List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
        ]:
        """Unpack an input data batch for use in `compute_batch_gradients`."""
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

    def apply_updates(  # type: ignore  # future: revise
            self,
            updates: TorchVector,
        ) -> None:
        """Apply updates to the model's weights."""
        with torch.no_grad():
            for idx, par in enumerate(self._model.parameters()):
                upd = updates.coefs.get(str(idx))
                if upd is not None:
                    par.add_(upd)
