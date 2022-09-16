# coding: utf-8

"""Model and metrics checkpointing util."""

import json
import os
from typing import List, Optional

import numpy as np

from declearn2.model.api import Model, NumpyVector
from declearn2.utils import json_pack, serialize_object


__all__ = [
    'Checkpointer',
]


class Checkpointer:
    """Model and metrics checkpointing class."""

    def __init__(
            self,
            model: Model,
            folder: Optional[str] = None,
        ) -> None:
        """Instantiate the checkpointer.

        Parameters
        ----------
        model: Model
            Model, the config and weights from which to checkpoint.
        folder: str or None, default=None
            Optional folder where to write output files, such as
            the loss values or the model's checkpointed weights.
            If None, record losses in memory, as well as weights
            having yielded the lowest loss only.
        """
        self.model = model
        self.folder = folder
        self._best = None  # type: Optional[NumpyVector]
        self._loss = []    # type: List[float]

    def save_model(
            self,
        ) -> None:
        """Save the wrapped model's configuration to a JSON file."""
        if self.folder is not None:
            path = os.path.join(self.folder, "model.json")
            serialize_object(self.model).to_json(path)

    def checkpoint(
            self,
            loss: float,
        ) -> None:
        """Checkpoint the loss value and the model's weights.

        If `self.folder is not None`, append the loss value to
        the "losses.txt" file and record model weights under a
        "weights_{i}.json" file.
        Otherwise, retain the loss, and the model's weights if
        the loss is at its lowest.

        Parameters
        ----------
        loss: float
            Loss value associated with the current model state.
        """
        self._loss.append(loss)
        if loss <= np.min(self._loss):
            self._best = self.model.get_weights()
        if self.folder is not None:
            # Save the model's weights to a JSON file.
            indx = len(self._loss)
            path = os.path.join(self.folder, f"weights_{indx}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.model.get_weights(), file, default=json_pack)
            # Append the loss to a txt file.
            path = os.path.join(self.folder, "losses.txt")
            mode = "a" if indx > 1 else "w"
            with open(path, mode, encoding="utf-8") as file:
                file.write(f"{indx}: {loss}\n")

    def reset_best_weights(
            self,
        ) -> None:
        """Restore the model's weights associated with the lowest past loss."""
        if self._best is not None:
            self.model.set_weights(self._best)

    def get_loss(
            self,
            index: int,
        ) -> float:
        """Return the loss value recorded at a given index."""
        return self._loss[index]
