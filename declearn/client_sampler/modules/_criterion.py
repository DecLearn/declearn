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
from declearn.model.api import Model
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_from_attr,
)

MissingScorePolicy = Literal["priority", "equal"]


@create_types_registry(name="ClientSamplerCriterion")
class Criterion(metaclass=ABCMeta):
    """
    Abstract class for client sampling criterion.

    Criterion objects are used by CriterionClientSampler objects to select the
    best clients regarding the value of a criterion calculated on clients
    (e.g. highest norm of gradient).

    Attributes
    ----------
    - name: str class attribute
        See details in the Abstract section.

    Abstract
    --------
    The following attributes and methods must be implemented by any
    non-abstract child class:

    - name: str class attribute
        Identifier name of the criterion, should match the class name and be
        unique across client sampler Criterion.
        e.g. "constant" for `ConstantCriterion`.

    Overridable
    -----------
    - _from_specs(cls, **kwargs: Any):
        Class method, backend of the `from_specs` method, can be overriden by
        subclass if specific mechanisms are needed to allow a proper
        instanciation from specifications.

    Inheritance
    -----------
    When a subclass inheriting from `Criterion` is declared, it is
    automatically registered under the "Criterion" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(Criterion, register=False)`)
    See `declearn.utils.register_type` for details on types registration.

    Key methods
    -----------
    - compute(client_to_reply, server_model):
        Instance method that computes the criterion score for each client based
        on the client train replies and server model.
    """

    name: ClassVar[str]

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
        server_model: Model,
    ) -> Dict[str, float]:
        """
        Compute the criterion value (score) for each client based on the client
        train replies and the server model.

        Note: The parameters must be considered read-only, do not modify them
        when defining the concrete method.

        Parameters
        ----------
        client_to_reply:
            Dictionary mapping a client name to their reply.
        server_model:
            Central server model.

        Returns
        -------
        Dictionary mapping a client name to their criterion value.
        """

    @staticmethod
    def wrap(obj: Any):
        """
        Make sure a native Python object is wrapped in a Criterion.

        Parameters
        ----------
        obj:
            object to be wrapped in a Criterion.

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
        Instantiate a 'Criterion' from its specifications.

        Parameters
        ----------
        name:
            Name of the criterion associated with the target Criterion subclass.
        **kwargs:
            Any additional instantiation keyword argument (general or
            criterion-specific).

        Returns
        -------
        criterion:
            `Criterion` instance matching input specifications.

        Raises
        ------
        ValueError
            If `name` does not match any registered `Criterion` type,
            or more generally if specifications are invalid.
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
        Backend of the from_specs method, specific to the subclass.
        Can be overriden.
        """
        return cls(**kwargs)


class CompositionCriterion(Criterion):
    """
    Allow to apply operations between `Criterion` objects to compose them.
    """

    name = "composition"

    def __init__(self, operation: Callable, *parents: Criterion):
        self.operation = operation
        self.parents: Tuple[Criterion, ...] = parents

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, float]:
        if self.operation is None:
            raise ValueError(
                "Criterion value cannot be computed with no operation."
            )

        cli_to_val_list = [
            parent.compute(client_to_reply, server_model)
            for parent in self.parents
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
        Backend of the from_specs method, specific to the
        `CompositionCriterion`.

        TODO precise what operations are supported in specs

        Raises
        ------
        ValueError
            If specifications are invalid.
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
    Wrap a native Python object (int, float, bool or None) in a `Criterion`
    object.
    """

    name = "constant"

    def __init__(self, value: Union[int, float, bool, None]):
        super().__init__()
        self.value = value

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, float]:
        return {client_name: self.value for client_name in client_to_reply}


class GradientNormCriterion(Criterion):
    """
    Criterion subclass where the criterion value is the L2-norm of the client
    "gradients" (model updates).
    """

    name = "gradient_norm"

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, float]:
        client_to_norm: Dict[str, float] = {}
        for client, reply in client_to_reply.items():
            flattened_updates, _ = reply.updates.updates.flatten()
            client_to_norm[client] = np.linalg.norm(flattened_updates).item()
        return client_to_norm


class NormalizedDivCriterion(Criterion):
    """
    Criterion subclass where the criterion value is the normalized model
    divergence (average difference between the model weights in client i
    and the global model).

    Note: Only the trainable weights are compared.

    Raises
    ------
    ValueError:
        If the number of trainable weights in server model and in a client
        updates object are different.

    Reference
    ---------
    [1] Fu et al., 2023.
        Client Selection in Federated Learning: Principles, Challenges, and
        Opportunities.
        Section IV.A.2.
        https://arxiv.org/abs/2211.01549
    """

    name = "normalized_divergence"

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, float]:
        client_to_div: Dict[str, float] = {}
        w_server = np.array(
            server_model.get_weights(trainable=True).flatten()[0]
        )  # server weights
        size_w = len(w_server)  # model size (nb trainable parameters)
        eps = 1e-8  # epsilon added to denominator to avoid zero-division error
        for client, reply in client_to_reply.items():
            w_updates = np.array(reply.updates.updates.flatten()[0])
            # client weight updates
            size_upd = len(w_updates)
            if size_upd != size_w:
                raise ValueError(
                    f"Flattened server model weights size ({size_w}) and "
                    f"client model updates size ({size_upd}) must be equal."
                )
            client_to_div[client] = (
                1 / size_w * np.sum(np.abs(w_updates / (w_server + eps)))
            ).item()
        return client_to_div


class CriterionClientSampler(ClientSampler):
    """
    Sample participants with the highest criterion score.

    TODO : precise that criterion are on server weights / client updates
    and why not compatible with secagg (uses client training info)

    This implementation sets and uses a client metadata named "score"
    to perform the sampling. A client score is the criterion value associated
    to them if already computed ; otherwise, it is a default value depending on
    the missing_scores_policy.

    Attributes
    ----------
    n_samples: int
        Number of clients to be sampled.
    criterion: Criterion
        The criterion to be used to select the best clients.
    missing_scores_policy:  Optional[MissingScorePolicy]
        String that identifies a missing scores policy, i.e. a strategy to
        attribute a criterion score to a client if it is missing (e.g. because
        of a missing train reply).
        Supported values are :
            "priority": prioritizes the clients with a missing score, by setting
            the score to infinity.
            "equal": sets the missing scores to 1 / number_of_clients.
    """

    strategy = "criterion"

    def __init__(
        self,
        n_samples: int,
        criterion: Criterion,
        missing_scores_policy: Optional[MissingScorePolicy] = "priority",
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        """
        Instantiate the criterion client sampler.

        Raises
        ------
        ValueError:
            If the provided missing scores policy is not supported.
        """
        super().__init__(max_retries=max_retries)
        if missing_scores_policy not in MissingScorePolicy.__args__:
            raise ValueError(
                f"Missing scores policy {missing_scores_policy} "
                f"is not supported."
            )
        self.n_samples = n_samples
        self.criterion = criterion
        self.missing_scores_policy = missing_scores_policy

    @property
    def secagg_compatible(self) -> bool:
        return False

    def init_clients(self, clients: Set[str]) -> None:
        """
        Initialize clients common metadata and then set each client's criterion
        score to None.
        """
        super().init_clients(clients)
        for client in clients:
            self.client_to_metadata[client].setdefault("score", None)

    def convert_missing_scores(self) -> Dict[str, float]:
        """
        Access client scores in metadata, and convert missing scores such that
        each client gets a non-None score.

        Raises
        ------
        ValueError:
            If the string identifying the missing score policy is not supported.
        """
        if self.missing_scores_policy == "priority":
            replacement_score = float("inf")
        elif self.missing_scores_policy == "equal":
            replacement_score = 1 / len(self.clients)
        else:
            raise ValueError(
                f"Missing scores policy {self.missing_scores_policy} "
                f"is not supported."
            )

        client_to_score = {
            client: self.client_to_metadata[client]["score"]
            for client in self.client_to_metadata.keys()
        }
        return {
            client: score if score is not None else replacement_score
            for client, score in client_to_score.items()
        }

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        """
        Back-end of the sampling method for criterion client sampler.

        If there are more than `n_samples` clients in `eligible_clients`,
        this method selects the `n_samples` clients with the highest criterion
        scores. Otherwise, they are all selected.
        """
        if self.n_samples >= len(eligible_clients):
            return eligible_clients

        client_to_score = self.convert_missing_scores()

        eligible_client_to_score = {
            client: score
            for client, score in client_to_score.items()
            if client in eligible_clients
        }

        ordered_client_to_score = dict(
            sorted(
                eligible_client_to_score.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )  # ordered by highest criterion score
        best_clients = set(
            list(ordered_client_to_score.keys())[: self.n_samples]
        )
        self._logger.debug(f"Client scores: {ordered_client_to_score}.")
        return best_clients

    def update(
        self, client_to_reply: Dict[str, TrainReply], server_model: Model
    ) -> None:
        """
        Update clients metadata and sampler internal state according
        to each client training reply and the server model.

        Concretely, compute and update each client criterion score.
        """
        updated_client_to_score = self.criterion.compute(
            client_to_reply, server_model
        )
        for client, score in updated_client_to_score.items():
            self.client_to_metadata[client]["score"] = score

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        Backend of the from_specs method, specific to
        'CriterionClientSampler'.
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
