# coding: utf-8

"""API and base FedAvg-like class to implement local updates' aggregation."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional


from declearn.model.api import Vector
from declearn.utils import create_types_registry, register_type


__all__ = [
    'Aggregator',
    'AverageAggregator',
]


class Aggregator(metaclass=ABCMeta):
    """Base class to implement an aggregation method."""

    @abstractmethod
    def aggregate(
            self,
            updates: Dict[str, Vector],
            n_steps: Dict[str, int],  # revise: abstract~generalize kwargs use
        ) -> Vector:
        """Aggregate input vectors into a single one.

        Parameters
        ----------
        updates: dict[str, Vector]
            Client-wise updates, as a dictionary with clients' names as
            string keys and updates as Vector values.
        n_steps: dict[str, int]
            Client-wise number of local training steps performed during
            the training round having produced the updates.

        Returns
        -------
        gradients: Vector
            Aggregated updates, as a Vector - treated as gradients by
            the server-side optimizer.
        """
        return NotImplemented

    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this object's parameters."""
        return {}

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'Aggregator':
        """Instantiate an Aggregator from its configuration dict."""
        return cls(**config)


create_types_registry("Aggregator", Aggregator)


@register_type(name="Average", group="Aggregator")
class AverageAggregator(Aggregator):
    """Average-based-aggregation Aggregator subclass.

    This class implements local updates' averaging, with optional
    client-based and/or number-of-training-steps-based weighting.

    It may therefore be used to implement FedAvg and derivatives
    that use simple weighting schemes.
    """

    def __init__(
            self,
            steps_weighted: bool = True,
            client_weights: Optional[Dict[str, float]] = None,
        ) -> None:
        """Instantiate an averaging aggregator.

        Parameters
        ----------
        steps_weighted: bool, default=True
            Whether to weight updates based on the number of optimization
            steps taken by the clients (relative to one another).
        client_weights: dict[str, float] or None, default=None
            Optional dict of client-wise base weights to use.
            If None, homogeneous base weights are used.

        Notes
        -----
        * One may specify `client_weights` and use `steps_weighted=True`.
          In that case, the product of the client's base weight and their
          number of training steps taken will be used (and unit-normed).
        * One may use incomplete `client_weights`. In that case, unknown-
          clients' base weights will be set to 1.
        """
        self.steps_weighted = steps_weighted
        self.client_weights = client_weights or {}

    def get_config(
            self,
        ) -> Dict[str, Any]:
        return {
            "steps_weighted": self.steps_weighted,
            "client_weights": self.client_weights,
        }

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'AverageAggregator':
        return cls(**config)

    def aggregate(
            self,
            updates: Dict[str, Vector],
            n_steps: Dict[str, int],
        ) -> Vector:
        if not updates:
            raise TypeError("Cannot aggregate an empty set of updates.")
        weights = self.compute_client_weights(updates, n_steps)
        agg = sum(grads * weights[client] for client, grads in updates.items())
        return agg  # type: ignore

    def compute_client_weights(
            self,
            updates: Dict[str, Vector],
            n_steps: Dict[str, int],
        ) -> Dict[str, float]:
        """Compute weights to use when averaging a given set of updates.

        Parameters
        ----------
        updates: dict[str, Vector]
            Client-wise updates, as a dictionary with clients' names as
            string keys and updates as Vector values.
        n_steps: dict[str, int]
            Client-wise number of local training steps performed during
            the training round having produced the updates.

        Returns
        -------
        weights: dict[str, float]
            Client-wise updates-averaging weights, suited to the input
            parameters and normalized so that they sum to 1.
        """
        if self.steps_weighted:
            weights = {
                client: steps * self.client_weights.get(client, 1.)
                for client, steps in n_steps.items()
            }
        else:
            weights = {
                client: self.client_weights.get(client, 1.)
                for client in updates
            }
        total = sum(weights.values())
        return {client: weight / total for client, weight in weights.items()}
