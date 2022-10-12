# coding: utf-8

"""Model subclass to wrap PyTorch models."""

import io
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

from declearn.model.api import Model, NumpyVector
from declearn.model.torch._vector import TorchVector
from declearn.typing import Batch
from declearn.utils import register_type


# alias for unpacked Batch structures, converted to torch.Tensor objects
TensorBatch = Tuple[
    List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]


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
        # metrics: ,
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
        # Type-check the input Model and wrap it up.
        if not isinstance(model, torch.nn.Module):
            raise TypeError("'model' should be a torch.nn.Module instance.")
        super().__init__(model)
        # Assign loss module and set it not to reduce sample-wise values.
        self._loss_fn = loss
        self._loss_fn.reduction = "none"  # type: ignore

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
            torch.save(self._model, buffer)
            model = buffer.getbuffer().hex()
        with io.BytesIO() as buffer:
            torch.save(self._loss_fn, buffer)
            loss = buffer.getbuffer().hex()
        return {
            "model": model,
            "loss": loss,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "TorchModel":
        """Instantiate a TorchModel from a configuration dict."""
        with io.BytesIO(bytes.fromhex(config["model"])) as buffer:
            model = torch.load(buffer)  # type: ignore
        with io.BytesIO(bytes.fromhex(config["loss"])) as buffer:
            loss = torch.load(buffer)  # type: ignore
        return cls(model=model, loss=loss)

    def get_weights(
        self,
    ) -> NumpyVector:
        weights = {
            key: tns.numpy().copy()  # NOTE: otherwise, view on Tensor data
            for key, tns in self._model.state_dict().items()
        }
        return NumpyVector(weights)

    def set_weights(
        self,
        weights: NumpyVector,
    ) -> None:
        # false-positive on torch.from_numpy; pylint: disable=no-member
        self._model.load_state_dict(
            {key: torch.from_numpy(arr) for key, arr in weights.coefs.items()}
        )

    def compute_batch_gradients(
        self,
        batch: Batch,
    ) -> TorchVector:
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
        loss.backward()  # type: ignore
        # Collect weights' gradients and return them in a Vector container.
        grads = {
            str(i): p.grad.detach().clone()
            for i, p in enumerate(self._model.parameters())
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
            loss.mul_(s_wght)
        return loss.mean()  # type: ignore

    def apply_updates(  # type: ignore  # future: revise
        self,
        updates: TorchVector,
    ) -> None:
        with torch.no_grad():
            for idx, par in enumerate(self._model.parameters()):
                upd = updates.coefs.get(str(idx))
                if upd is not None:
                    par.add_(upd)

    def compute_loss(
        self,
        dataset: Iterable[Batch],
    ) -> float:
        total = 0.0
        n_btc = 0.0
        try:
            self._model.eval()
            with torch.no_grad():
                for batch in dataset:
                    inputs, y_true, s_wght = self._unpack_batch(batch)
                    y_pred = self._model(*inputs)
                    loss = self._compute_loss(y_pred, y_true, s_wght)
                    total += loss.numpy()
                    n_btc += 1 if s_wght is None else s_wght.mean().numpy()
        finally:
            self._model.train()
        return total / n_btc
