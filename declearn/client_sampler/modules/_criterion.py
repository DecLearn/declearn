from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

from declearn.client_sampler._base import ClientSampler
from declearn.messaging import TrainReply
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_from_attr,
)

MissingWeightPolicy = Literal["priority", "equal"]


@create_types_registry(name="ClientSamplerCriterion")
class Criterion(metaclass=ABCMeta):
    """
    Abstract class for client sampling criterion.

    TODO details
    """

    name: ClassVar[str]
    """Name identifier of the class, unique across client sampler Criterion
    classes.
    """

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register Criterion subclasses if enabled."""
        super().__init_subclass__(**kwargs)
        if register:
            register_from_attr(cls, "name", "ClientSamplerCriterion")

    @abstractmethod
    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
    ) -> Dict[str, float]:
        """
        Compute the criterion value for each client based on the train replies
        of clients.

        Parameters
        ----------
        client_to_reply: Dict[str, TrainReply]
            Dictionary mapping a client name to their reply

        Returns
        -------
        Dictionary mapping a client name to their criterion value
        """

    @staticmethod
    def wrap(obj: Any):
        """
        Make sure a native Python object is wrapped in a Criterion.

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

    @staticmethod
    def from_specs(name: str, **kwargs: Any) -> Criterion:
        """
        TODO
        """
        try:
            cls = access_registered(name, group="ClientSamplerCriterion")
        except KeyError as e:
            raise ValueError(
                f"Unknown client sampler criterion name '{name}'"
            ) from e

        try:
            return cls._from_specs(**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Invalid client sampler criterion specifications: {e}"
            ) from e

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> Criterion:
        """
        TODO
        """
        return cls(**kwargs)


class CompositionCriterion(Criterion):
    """
    Allow to apply operations between criteria to compose them.
    """

    name = "composition"

    def __init__(self, operation: Callable, *parents: Criterion):
        self.operation = operation
        self.parents: Tuple[Criterion, ...] = parents

    def compute(
        self, client_to_reply: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        if self.operation is None:
            raise ValueError(
                "Criterion value cannot be computed with no operation."
            )

        cli_to_val_list = [
            parent.compute(client_to_reply) for parent in self.parents
        ]  # list of mappings between client and value for each parent
        client_to_composed_val = {}
        for client in client_to_reply.keys():
            client_values = [
                cli_to_val[client] for cli_to_val in cli_to_val_list
            ]  # list of values of *this client* for each parent
            client_to_composed_val[client] = self.operation(*client_values)

        return client_to_composed_val

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> Criterion:
        """
        TODO
        raises ...
        """
        operation_str = kwargs["operation"]
        if not isinstance(operation_str, str):
            raise ValueError("Criterion 'operation' value must be a string")

        op_str_to_func = {
            "add": float.__add__,
            "+": float.__add__,
            "sub": float.__sub__,
            "-": float.__sub__,
            "mul": float.__mul__,
            "*": float.__mul__,
            "div": float.__truediv__,
            "truediv": float.__truediv__,
            "/": float.__truediv__,
            "radd": float.__radd__,
            "rsub": float.__rsub__,
            "rmul": float.__rmul__,
            "rtruediv": float.__rtruediv__,
            "pow": float.__pow__,
        }

        if operation_str not in op_str_to_func:
            raise ValueError(
                f"Unsupported criterion operation '{operation_str}'"
            )
        operation = op_str_to_func[operation_str]

        parsed_parents = []
        for parent in kwargs["parents"]:
            if isinstance(parent, Criterion):
                parsed_parent = parent
            elif isinstance(parent, dict):
                parsed_parent = Criterion.from_specs(**parent)
            else:
                raise ValueError(
                    f"Unsupported criterion type '{type(parent)}' "
                    "in 'parents' list"
                )
            parsed_parents.append(parsed_parent)
        return cls(operation, *parsed_parents)


class ConstantCriterion(Criterion):
    """
    Wrap a native Python object (int, float, bool or None) in a Criterion.
    """

    name = "constant"

    def __init__(self, value: Union[int, float, bool, None]):
        super().__init__()
        self.value = value

    def compute(
        self, client_to_reply: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        return {client_name: self.value for client_name in client_to_reply}


class GradientNormCriterion(Criterion):
    """
    Retrieve the L2-norm of the gradients from the TrainReply message of clients.
    """

    name = "gradient_norm"

    def compute(
        self, client_to_reply: Dict[str, TrainReply]
    ) -> Dict[str, float]:
        client_to_norm: Dict[str, float] = {}
        for client, reply in client_to_reply.items():
            flattened_gradients, _ = reply.updates.updates.flatten()
            client_to_norm[client] = np.linalg.norm(flattened_gradients)

        return client_to_norm


class CriterionClientSampler(ClientSampler):
    """
    Sample participants with the highest criterion values.

    This implementation sets and uses a client metadata named "weight"
    to perform the sample. A client weight is this client criterion value
    if already computed ; else, it is a default value depending on the
    missing_weights_policy.

    TODO doc
    """

    strategy = "criterion"

    def __init__(
        self,
        n_samples: int,
        criterion: Criterion,
        missing_weights_policy: Optional[MissingWeightPolicy] = "priority",
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        super().__init__(max_retries=max_retries)
        if missing_weights_policy not in MissingWeightPolicy.__args__:
            raise NotImplementedError(
                f"Missing weights policy {missing_weights_policy} "
                f"is not implemented."
            )
        self.n_samples = n_samples
        self.criterion = criterion
        self.missing_weights_policy = missing_weights_policy

    @property
    def secagg_compatible(self) -> bool:
        return False

    def init_clients(self, clients: Set[str]) -> None:
        super().init_clients(clients)
        for client in clients:
            self.client_to_metadata[client].setdefault("weight", None)

    def convert_missing_weights(self) -> Dict[str, float]:
        """
        Access client weights in metadata, and convert missing weights such that
        each client gets a non-None weight.
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

        client_to_weight = {
            client: self.client_to_metadata[client]["weight"]
            for client in self.client_to_metadata.keys()
        }
        return {
            client: weight if weight is not None else replacement_weight
            for client, weight in client_to_weight.items()
        }

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        client_to_weight = self.convert_missing_weights()

        eligible_client_to_weight = {
            client: weight
            for client, weight in client_to_weight.items()
            if client in eligible_clients
        }

        ordered_client_to_weight = dict(
            sorted(
                eligible_client_to_weight.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )  # ordered by highest criterion weight
        best_clients = set(
            list(ordered_client_to_weight.keys())[: self.n_samples]
        )
        return best_clients

    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        updated_client_to_weight = self.criterion.compute(client_to_reply)
        for client, weight in updated_client_to_weight.items():
            self.client_to_metadata[client]["weight"] = weight

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        TODO
        """
        criterion = kwargs["criterion"]
        if isinstance(criterion, Criterion):
            pass  # nothing to do
        elif isinstance(criterion, dict):
            kwargs["criterion"] = Criterion.from_specs(**criterion)
        else:
            raise ValueError(
                f"Unsupported criterion type '{type(criterion)}' used as "
                "'criterion' value"
            )
        return cls(**kwargs)
