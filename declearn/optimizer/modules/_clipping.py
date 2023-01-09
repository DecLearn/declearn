# coding: utf-8

"""Batch-averaged gradients clipping plug-in modules."""

from typing import Any, Dict


from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule


__all__ = ["L2Clipping"]


class L2Clipping(OptiModule):
    """Fixed-threshold L2-norm gradient clipping module.

    This module implements the following algorithm:
        Init(max_norm):
        Step(max_norm):
            norm = euclidean_norm(grads)
            clip = min(norm, max_norm)
            grads *= clip / max_norm

    In other words, (batch-averaged) gradients are clipped
    based on their L2 (euclidean) norm, based on a single,
    fixed threshold (as opposed to more complex algorithms
    that may use parameter-wise and/or adaptive clipping).

    This may notably be used to bound the contribution of
    batch-based gradients to model updates, notably so as
    to bound the sensitivity associated to that action.
    """

    name = "l2-clipping"

    def __init__(
        self,
        max_norm: float = 1.0,
    ) -> None:
        """Instantiate the L2-norm gradient-clipping module.

        Parameters
        ----------
        max_norm: float, default=1.0
            Clipping threshold of the L2 (euclidean) norm of
            input (batch-averaged) gradients.
        """
        self.max_norm = max_norm

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Apply L2-norm clipping to input gradients."""
        l2_norm = (gradients**2).sum() ** 0.5
        c_scale = (l2_norm / self.max_norm).minimum(1.0)
        return gradients * c_scale

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters."""
        return {"max_norm": self.max_norm}