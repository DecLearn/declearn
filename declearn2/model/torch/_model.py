# coding: utf-8

"""Model subclass to wrap PyTorch models."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from numpy.typing import ArrayLike

from declearn2.model.api import Model, NumpyVector
from declearn2.model.torch._vector import TorchVector
from declearn2.utils import unpack_batch


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
        ) -> NumpyVector:  # revise: could be TorchVector
        """Return the model's trainable weights."""
        return NumpyVector({
            key: tns.numpy() for key, tns in self._model.state_dict().items()
        })

    def set_weights(
            self,
            weights: NumpyVector,  # revise: could be TorchVector
        ) -> None:
        """Assign values to the model's trainable weights."""
        self._model.load_state_dict({
            key: torch.Tensor(arr) for key, arr in weights.coefs.items()
        })

    def compute_batch_gradients(
            self,
            batch: Union[ArrayLike, List[Optional[ArrayLike]]],
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
            batch: Union[ArrayLike, List[Optional[ArrayLike]]]
        ) -> Tuple[
            List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
        ]:
        """Unpack an input data batch for use in `compute_batch_gradients`."""
        # Perform basic unpacking.
        inparr, y_true, s_wght = unpack_batch(batch)
        # Convert inputs into a list of Tensor objects.
        if isinstance(inparr, (tuple, list)):
            inputs = [torch.Tensor(arr) for arr in inparr]
        else:
            inputs = [torch.Tensor(inparr)]
        # Ensure defined arrays are converted to Tensor objects.
        y_true = None if (y_true is None) else torch.Tensor(y_true)
        s_wght = None if (s_wght is None) else torch.Tensor(s_wght)
        # Return the prepared data.
        return inputs, y_true, s_wght

    def apply_updates(  # type: ignore  # future: revise
            self,
            updates: TorchVector,
        ) -> None:
        """Apply updates to the model's weights."""
        for idx, par in self._model.parameters():
            upd = updates.coefs.get(str(idx))
            if upd is not None:
                par.add_(upd)
