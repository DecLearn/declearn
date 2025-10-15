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

"""Base API for plug-in gradients-alteration algorithms."""

import abc
import dataclasses
from typing import Any, ClassVar, Dict, Generic, Optional, Self, Type, TypeVar

from declearn.model.api import Vector
from declearn.utils import (
    Aggregate,
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "AuxVar",
    "OptiModule",
]


T = TypeVar("T")


@dataclasses.dataclass
class AuxVar(Aggregate, base_cls=True, register=False, metaclass=abc.ABCMeta):
    """Abstract base class for OptiModule auxiliary variables.

    Each and every `OptiModule` subclass that requires information to be
    exchanged is expected to be coupled with one (or multiple) `AuxVar`
    subtype(s). These may be used to transmit information from a server
    to its clients, and/or to exchange and aggregate data from clients
    into a single `AuxVar` instance to be processed by the server.

    This class also defines whether contents are compatible with secure
    aggregation, and whether some fields should remain in cleartext no
    matter what.

    Note that subclasses are automatically type-registered, and should be
    decorated as `dataclasses.dataclass`. To prevent registration, simply
    pass `register=False` at inheritance.
    """

    _group_key = "AuxVar"


AuxVarT = TypeVar("AuxVarT", bound=AuxVar)


@create_types_registry
class OptiModule(Generic[AuxVarT], metaclass=abc.ABCMeta):
    """Abstract class defining an API to implement gradients adaptation tools.

    The aim of this abstraction (which itself operates on the Vector
    abstraction, so as to provide framework-agnostic algorithms) is
    to enable implementing unitary gradients-adaptation bricks that
    can easily and modularly be composed into complex algorithms.

    The `declearn.optimizer.Optimizer` class defines the main tools
    and routines for computing and applying gradients-based updates.
    `OptiModule` instances are designed to be "plugged in" such an
    `Optimizer` instance to add intermediary operations between the
    moment gradients are obtained and that when they are applied as
    updates. Note that learning-rate use and optional (decoupled)
    weight-decay mechanisms are implemented at `Optimizer` level.

    Abstract
    --------
    The following attribute and method require to be overridden
    by any non-abstract child class of `OptiModule`:

    - name: str class attribute
        Name identifier of the class (should be unique across existing
        OptiModule classes). Also used for automatic types-registration
        of the class (see `Inheritance` section below).
    - run(gradients: Vector) -> Vector:
        Apply an adaptation algorithm to input gradients and return
        them. This is the main method for any `OptiModule`.

    Overridable
    -----------
    The following methods may be overridden to implement information-
    passing and parallel behaviors between client/server module pairs.
    As defined at `OptiModule` level, they have no effect and may thus
    be safely ignored when implementing self-contained algorithms.

    - collect_aux_var() -> Optional[AuxVar]:
        Emit an `AuxVar` instance holding auxiliary variables,
        that may be shared with peers, aggregated across them,
        and eventually processed by a counterpart module on the
        other side of the client/server relationship.
    - process_aux_var(AuxVar) -> None:
        Process auxiliary variables received from a counterpart
        module on the other side of the client/server relationship.
    - aux_name: optional[str] class attribute, default=None
        Name to use when sending or receiving auxiliary variables
        between synchronous client/server modules, that therefore
        need to share the *same* `aux_name`.
    - auxvar_cls: optional[type[AuxVar]] class attribute, default=None
        Type of `AuxVar` used by this module (defining the actual
        signature of `collect_aux_var` and `process_aux_var`).

    Inheritance
    -----------
    When a subclass inheriting from `OptiModule` is declared, it is
    automatically registered under the "OptiModule" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(OptiModule, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str] = NotImplemented
    """Name identifier of the class, unique across OptiModule classes."""

    aux_name: ClassVar[Optional[str]] = None
    """Optional aux-var-sharing identifier of the class.

    This name may be shared by a pair of OptiModule classes, designed
    to operate on the client and server side respectively. It should
    be unique to that pair of classes across all OptiModule classes.
    """

    auxvar_cls: Optional[Type[AuxVar]] = None
    """Optional `AuxVar` subtype used by this module and its counterpart."""

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register OptiModule subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.name, group="OptiModule")

    @abc.abstractmethod
    def run(
        self,
        gradients: Vector[T],
    ) -> Vector[T]:
        """Apply the module's algorithm to input gradients.

        Please refer to the module's main docstring for details
        on the implemented algorithm and the way it transforms
        input gradients.

        Parameters
        ----------
        gradients: Vector
            Input gradients that are to be processed and updated.

        Returns
        -------
        gradients: Vector
            Modified input gradients. The output Vector should be
            fully compatible with the input one - only the values
            of the wrapped coefficients may have changed.
        """

    def collect_aux_var(
        self,
    ) -> Optional[AuxVarT]:
        """Return auxiliary variables that need to be shared between nodes.

        Returns
        -------
        aux_var: Optional[AuxVar]
            Optional `AuxVar` instance holding auxiliary variables that
            are to be shared with a counterpart OptiModule on the other
            side of the client-server relationship.

        Notes
        -----
        The calling context depend ons whether the module is part of a
        client's optimizer or of the server's one:

        - Client:
            - `collect_aux_var` is expected to happen after taking a series
              of local optimization steps, before sending the local updates
              to the server for aggregation and further processing.
        - Server:
            - `collect_aux_var` is expected to happen when the global model
              weights are ready to be shared with clients, i.e. either at
              the very end or very beginning of a training round.
        """
        return None

    def process_aux_var(
        self,
        aux_var: AuxVarT,
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Parameters
        ----------
        aux_var:
            Auxiliary variables that are to be processed by this module,
            emitted by a counterpart OptiModule on the other side of the
            client-server relationship.

        Notes
        -----
        The calling context depends on whether the module is part of a
        client's optimizer or of the server's one:

        - Client:
            - `process_aux_var` is expected to happen at the beginning of
              a training round to define gradients' processing during the
              local optimization steps taken through that round.
        - Server:
            - `process_aux_var` is expected to happen upon receiving local
              updates (and, thus, aux_var), before the aggregated updates
              are computed and passed through the server optimizer (which
              comprises this module).

        Raises
        ------
        KeyError
            If received auxiliary variables lack some required data.
        NotImplementedError
            If auxiliary variables are passed to a module that is not meant
            to receive any.
        TypeError
            If `aux_var` or one of its fields has unproper type.
        """
        if aux_var is not None:  # pragma: no cover
            raise NotImplementedError(
                f"'{self.__class__.__name__}.process_aux_var' was called, but"
                " this class is not designed to receive auxiliary variables."
            )

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters.

        The counterpart to this method is the `from_config` classmethod.
        To access the module's inner states, see the `get_state` method.

        Returns
        -------
        config: Dict[str, Any]
            JSON-serializable dict storing this module's instantiation
            configuration.
        """
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate an OptiModule from its configuration dict.

        The counterpart to this classmethod is the `get_config` method.
        To restore the module's inner states, see its `get_state` method.

        Parameters
        ----------
        config: dict[str, Any]
            Dict storing the module's instantiation configuration.
            This must match the target subclass's requirements.

        Raises
        ------
        KeyError
            If the provided `config` lacks some required parameters
            and/or contains some unused ones.
        """
        return cls(**config)

    @staticmethod
    def from_specs(
        name: str,
        config: Dict[str, Any],
    ) -> "OptiModule":
        """Instantiate an OptiModule from its specifications.

        Parameters
        ----------
        name: str
            Name based on which the module can be retrieved.
            Available as a class attribute.
        config: dict[str, any]
            Configuration dict of the module, that is to be
            passed to its `from_config` class constructor.
        """
        cls = access_registered(name, group="OptiModule")
        assert issubclass(cls, OptiModule)  # force-tested by access_registered
        return cls.from_config(config)

    def get_state(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's state(s).

        The counterpart to this method is the `set_state` one.

        Returns
        -------
        state: Dict[str, Any]
            JSON-serializable dict storing this module's inner state
            variables.
        """
        return {}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        """Load a state dict into an instantiated module.

        The counterpart to this method is the `get_state` one.

        Parameters
        ----------
        state: dict[str, any]
            Dict storing values to assign to this module's inner
            state variables.

        Raises
        ------
        KeyError
            If an expected state variable is missing from `state`.
        """
        if state:
            raise KeyError(
                f"'{self.__class__.__name__}.set_state' received some data, "
                "but it is not implemented to actually use any."
            )
