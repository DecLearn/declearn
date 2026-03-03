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

"""Client sampling `Criterion` API and its core subclasses."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Tuple,
    Union,
)

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
    """Abstract class for client sampling criterion.

    `Criterion` objects are used by `CriterionClientSampler` objects to select
    the best clients regarding the value of a criterion score that can be
    derived from clients replies and global model (e.g. highest norm of client
    gradients).

    Attributes
    ----------
    name: str class attribute
        See details in the Abstract section.

    Abstract
    --------
    The following attributes and methods must be implemented by any
    non-abstract child class:

    - name: str class attribute
        Identifier name of the criterion, should match the class name and be
        unique across client sampler `Criterion`,
        e.g. "constant" for `ConstantCriterion`.

    - compute(client_to_reply, global_model):
        Instance method that computes the criterion score for each client based
        on the client train replies and global model.

    Overridable
    -----------
    - from_specs(cls, **kwargs):
        Class method, create an instance from specifications, can be overriden
        by subclass if specific mechanisms are needed for instantiation.

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
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        """Compute the criterion score for each client listed in `client_to_reply`.
        The score can be derived from information in the client train replies
        and the global model.

        Notes
        -----
        The parameters must be considered read-only, do not modify them
        when defining the concrete method.

        Parameters
        ----------
        client_to_reply:
            Dictionary mapping a client name to their reply.
        global_model:
            Global model hold by the server.

        Returns
        -------
        Dictionary mapping a client name to their criterion value.
        """

    @staticmethod
    def wrap(obj: Any) -> Criterion:
        """Make sure a native Python object is wrapped in a `Criterion`.

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

    @classmethod
    def from_specs(cls, **kwargs: Any) -> Criterion:
        """Instantiate a `Criterion` subclass from its specifications."""
        return cls(**kwargs)


class ConstantCriterion(Criterion):
    """`Criterion` implementation to wrap a native constant Python object
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
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        if self.value is None:
            value = None
        else:
            value = float(self.value)
        return {client_name: value for client_name in client_to_reply}


class CompositionCriterion(Criterion):
    """`Criterion` implementation that contains other `Criterion` objects, and
    allows to apply operations between criteria to compose them.

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
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        if self.operation is None:
            raise ValueError(
                "Criterion value cannot be computed with no operation."
            )

        cli_to_val_list = [
            parent.compute(client_to_reply, global_model)
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
    def from_specs(cls, **kwargs: Any) -> Criterion:
        """Instantiate a `CompositionCriterion` from specifications.

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
                parsed_parent = instantiate_criterion(**parent)
            else:
                raise ValueError(
                    f"Unsupported criterion type '{type(parent)}' "
                    "in 'parents' list"
                )
            parsed_parents.append(parsed_parent)
        return cls(operation, *parsed_parents)


def instantiate_criterion(name: str, **kwargs: Any) -> Criterion:
    """Instantiate a `Criterion` from its specifications.

    Parameters
    ----------
    name:
        Name of the criterion associated with the target `Criterion`
        subclass.
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
        return cls.from_specs(**kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid client sampler criterion specifications: {e}"
        ) from e
