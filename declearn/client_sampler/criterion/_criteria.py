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

"""
Client sampling `Criterion` API and concrete subclasses used by
`CriterionClientSampler`.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)

import numpy as np

from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_from_attr,
)

PrimitiveType = Union[int, float, bool, None]


@create_types_registry(name="ClientSamplerCriterion")
class Criterion(metaclass=ABCMeta):
    """
    Abstract class for client sampling criterion.

    `Criterion` objects are used by `CriterionClientSampler` objects to select
    the best clients regarding the value of a criterion score that can be
    derived from clients replies and server model (e.g. highest norm of client
    gradients).

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
        unique across client sampler `Criterion`,
        e.g. "constant" for `ConstantCriterion`.

    - compute(client_to_reply, server_model):
        Instance method that computes the criterion score for each client based
        on the client train replies and server model.

    Overridable
    -----------
    - _from_specs(cls, **kwargs):
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
    """

    name: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register `Criterion` subclasses if enabled."""
        super().__init_subclass__(**kwargs)
        if register:
            register_from_attr(cls, "name", "ClientSamplerCriterion")

    @abstractmethod
    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the criterion score for each client listed in `client_to_reply`.
        The score can be derived from information in the client train replies
        and the server model.

        Notes
        -----
        The parameters must be considered read-only, do not modify them
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
    def wrap(obj: Any) -> Criterion:
        """
        Make sure a native Python object is wrapped in a `Criterion`.

        Parameters
        ----------
        obj:
            object to be wrapped in a `Criterion`.

        Returns
        -------
        criterion:
            `Criterion` wrapping the input object.

        Raises
        ------
        ValueError
            If object is not a bool, int, float or `Criterion`.
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
        Instantiate a `Criterion` from its specifications.

        Parameters
        ----------
        name:
            Name of the criterion associated with the target `Criterion` subclass.
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


class ConstantCriterion(Criterion):
    """
    `Criterion` implementation to wrap a native constant Python object
    (int, float, bool or None) in a `Criterion` object.

    Attributes
    ----------
    value: PrimitiveType
        The native object value wrapped into the `ConstantCriterion` instance.
    """

    name = "constant"

    def __init__(self, value: PrimitiveType):
        super().__init__()
        self.value = value

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
        if self.value is None:
            value = None
        else:
            value = float(self.value)
        return {client_name: value for client_name in client_to_reply}


class CompositionCriterion(Criterion):
    """
    `Criterion` implementation that contains other `Criterion` objects, and allows
    to apply operations between criteria to compose them.

    Attributes
    ----------
    operation: Callable[..., Optional[float]]
        Operation function that will be used to build the composed criterion.
        It must take one or more arguments of type PrimitiveType, and return
        a float or None.
    parents: Tuple[Criterion, ...]
        Tuple of parent criteria, i.e. criteria whose computed values will be
        composed using the `operation` attribute. Thus, the number of parents
        must be compatible with the number of arguments accepted by the
        operation.
    """

    name = "composition"

    def __init__(
        self, operation: Callable[..., Optional[float]], *parents: Criterion
    ):
        self.operation = operation
        self.parents: Tuple[Criterion, ...] = parents

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
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

        Notes
        -----
        The supported operation strings in the specifications are the
        following :
            - "add", "+"
            - "sub", "-"
            - "mul", "*"
            - "div", "truediv", "/"
            - "radd"
            - "rsub"
            - "rmul"
            - "rtruediv"
            - "pow"

        To use other operations in a CompositionClientSampler, you cannot
        use specifications and the `from_specs` method. You must
        instantiate the client sampler via the Python API, passing the
        operation as a Callable.

        Raises
        ------
        ValueError
            If specifications are invalid.
        """
        operation_str = kwargs["operation"]
        if not isinstance(operation_str, str):
            raise ValueError("Criterion 'operation' value must be a string")

        op_str_to_func: Dict[str, Callable] = {
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


class GradientNormCriterion(Criterion):
    """
    `Criterion` implementation where the criterion score is the L2-norm of the
    client "gradients" (model updates).
    """

    name = "gradient_norm"

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
        client_to_norm: Dict[str, Optional[float]] = {}
        for client, reply in client_to_reply.items():
            flattened_updates, _ = reply.updates.updates.flatten()
            client_to_norm[client] = np.linalg.norm(flattened_updates).item()
        return client_to_norm


class NormalizedDivCriterion(Criterion):
    """
    `Criterion` implementation where the criterion score is the normalized model
    divergence (average difference between the model weights in client i
    and the global model).

    Notes
    -----
    Only the trainable weights are compared.

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
    ) -> Dict[str, Optional[float]]:
        client_to_div: Dict[str, Optional[float]] = {}
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
            score = float(
                1 / size_w * np.sum(np.abs(w_updates / (w_server + eps)))
            )
            client_to_div[client] = score
        return client_to_div


class TrainTimeCriterion(Criterion):
    """
    `Criterion` implementation where the criterion score is computed from the
    last round training time (in seconds) spent by the client.

    Attributes
    ----------
    lower_is_better: bool
        If True (default), a lower time leads to a better score (score will be
        `- time`). Otherwise, a higher time leads to a better score (score will
        be `+ time`).
    """

    name = "train_time"

    def __init__(self, lower_is_better: bool = True):
        self.lower_is_better = lower_is_better

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
        sign = -1 if self.lower_is_better else 1
        return {
            client: sign * reply.t_spent
            for client, reply in client_to_reply.items()
        }


class TrainTimeHistoryCriterion(Criterion):
    """
    `Criterion` implementation where the criterion score is computed from the
    history of all past rounds' training times (in seconds) spent by the client.

    Attributes
    ----------
    lower_is_better: bool
        If True (default), a lower value for aggregated times leads to a better
        score (score will be `- aggregated_times`). Otherwise, a higher value
        for aggregated times of leads to a better score (score will be
        `+ aggregated_times`).
    agg: AggregateFunc
        Name of a method to aggregate the history values into a float, e.g.
        average, sum.
    history: Dict[str, List[float]], read-only instance property
        Dictionary mapping each client to its training time history (time values
        for all past training rounds).

    Notes
    -----
    Beware that the time history will be updated every time a call to `compute`
    is made, assuming that this call matches a new training round.
    Thus, you should only call this class' `compute` method once per round
    (passing the new round client replies as argument).
    """

    name = "train_time_history"

    AggregateFunc = Literal["average", "sum"]

    def __init__(
        self, lower_is_better: bool = True, agg: AggregateFunc = "average"
    ):
        if agg not in get_args(self.AggregateFunc):
            raise ValueError(f"Unsupported aggregate function '{agg}'.")

        self.lower_is_better = lower_is_better
        self.agg = agg
        self._history: Dict[str, List[float]] = {}

    @property
    def history(self):
        return self._history

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> Dict[str, Optional[float]]:
        sign = -1 if self.lower_is_better else 1
        for client, reply in client_to_reply.items():
            if client in self._history:
                self._history[client].append(reply.t_spent)
            else:
                self._history[client] = [reply.t_spent]  # init history

        # aggregate all times in each client history
        if self.agg == "average":

            def agg_fn(hist):
                return sum(hist) / len(hist)
        elif self.agg == "sum":
            agg_fn = sum
        else:
            raise ValueError(f"Unsupported aggregate function '{self.agg}'.")

        return {
            client: sign * agg_fn(self._history[client])
            for client in client_to_reply
        }
