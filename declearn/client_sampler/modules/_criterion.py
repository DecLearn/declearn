from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Set, Union

import numpy as np

from ...messaging import TrainReply
from .._base import ClientSampler


class Criterion(metaclass=ABCMeta):
    """
    Client sampling criterion.
    """

    @abstractmethod
    def compute(
        self, client_replies: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        """
        Compute the value of a criterion based on the train replies of clients.
        """

    @staticmethod
    def wrap(obj: Any):
        """
        Makes sure a native Python object is wrapped in a Criterion.

        Parameters
        ----------
            obj : Any
                object to be wrapped in a Criterion

        Raises
        ------
            ValueError
                If object is not a bool, int, float or Criterion.
        """
        if obj is None or isinstance(obj, (int, float, bool)):
            return ConstantCriterion(value=obj)
        if isinstance(obj, Criterion):
            return obj

        raise ValueError(f"Criterion cannot wrap {type(obj)}.")

    def __add__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__add__, self, self.wrap(other))

    def __radd__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__radd__, self, self.wrap(other))

    def __sub__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__sub__, self, self.wrap(other))

    def __rsub__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__rsub__, self, self.wrap(other))

    def __mul__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__mul__, self, self.wrap(other))

    def __rmul__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__rmul__, self, self.wrap(other))

    def __truediv__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__truediv__, self, self.wrap(other))

    def __rtruediv__(self, other: Any) -> Criterion:
        return CompositionCriterion(float.__rtruediv__, self, self.wrap(other))

    def __pow__(self, power, modulo=None):
        return CompositionCriterion(
            float.__pow__, self, self.wrap(power), self.wrap(modulo)
        )


class CompositionCriterion(Criterion):
    """
    Allows to apply operations between criteria to compose them.
    """

    def __init__(self, operation: Callable, *parents: Criterion):
        self.operation = operation
        self.parents = parents

    def compute(
        self, client_replies: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        if self.operation is None:
            raise ValueError(
                "Criterion value cannot be computed with no operation."
            )

        parent_values = [
            parent.compute(client_replies) for parent in self.parents
        ]
        out_result = {}
        for client_name in client_replies.keys():
            client_parent = [parent[client_name] for parent in parent_values]
            out_result[client_name] = self.operation(*client_parent)

        return out_result


class ConstantCriterion(Criterion):
    """
    Wraps a native Python object (int, float, bool or None) in a Criterion.
    """

    def __init__(self, value: Union[int, float, bool, None]):
        super().__init__()
        self.value = value

    def compute(
        self, client_replies: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        return {client_name: self.value for client_name in client_replies}


class GradientNormCriterion(Criterion):
    """
    Retrieves the norm of the gradients from the TrainReply message of clients.
    """

    def compute(
        self, client_replies: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        criterion_dict: Dict[str, float] = {}
        for client_name, reply in client_replies.items():
            flattened_gradients, _ = reply.updates.updates.flatten()
            criterion_dict[client_name] = np.linalg.norm(flattened_gradients)

        return criterion_dict


class CriterionClientSampler(ClientSampler):
    """
    Samples participants with the highest criterion values.
    """

    secagg_compatible = False

    def __init__(  # noqa: PLR0913
        self,
        clients: Set[str],
        n_samples: int,
        criterion: Criterion,
        prior_weights: Optional[Dict[str, float]] = None,
        initialization_round: bool = False,
        missing_weights_policy: Optional[
            Literal["priority", "equal"]
        ] = "priority",
    ):
        super().__init__(
            clients, n_samples, prior_weights, initialization_round
        )
        if missing_weights_policy not in ["priority", "equal"]:
            raise NotImplementedError(
                f"Missing weights policy {missing_weights_policy} "
                f"is not implemented."
            )

        self.criterion = criterion
        self.learnt_weights: Dict[str, Optional[float]] = {
            client_name: None for client_name in clients
        }
        self.missing_weights_policy = missing_weights_policy

    def convert_missing_weights(self) -> Dict[str, float]:
        """
        Convert missing weights such as each client gets a weight which is
        not None.
        """
        if self.missing_weights_policy == "priority":
            replacement_weight = float("inf")
        elif self.missing_weights_policy == "equal":
            replacement_weight = 1 / len(self.clients)
        else:
            raise NotImplementedError(
                f"Missing weights policy {self.missing_weights_policy} "
                f"is not implemented."
            )

        return {
            client_name: weight if weight is not None else replacement_weight
            for client_name, weight in self.learnt_weights.items()
        }

    def cls_sample(self, input_clients: Set[str]) -> Set[str]:
        learnt_weights = self.convert_missing_weights()

        weights_subset = {
            client_name: self.prior_weights[client_name]
            * learnt_weights[client_name]
            for client_name in input_clients
        }
        ordered_clients_criterion = dict(
            sorted(
                weights_subset.items(), key=lambda item: item[1], reverse=True
            )
        )
        best_clients = set(
            list(ordered_clients_criterion.keys())[: self.n_samples]
        )
        return best_clients

    def update(self, results: Dict[str, TrainReply]):
        learnt_weights = self.criterion.compute(results)
        self.learnt_weights.update(learnt_weights)
