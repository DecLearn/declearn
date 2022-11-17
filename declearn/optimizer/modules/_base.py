# coding: utf-8

"""Base API and common examples of plug-in gradients-alteration algorithms."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

from declearn.model.api import Vector
from declearn.utils import (
    ObjectConfig,
    access_registered,
    create_types_registry,
    deserialize_object,
    serialize_object,
)

__all__ = [
    "OptiModule",
]


@create_types_registry
class OptiModule(metaclass=ABCMeta):
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

    name: str class attribute
        Keyword naming this module. This has an effect when passing
        auxiliary variables (see section below) between synchronous
        client/server modules, which should share the *same* name.
    run(gradients: Vector) -> Vector:
        Apply an adaptation algorithm to input gradients and return
        them. This is the main method for any `OptiModule`.

    Overridable
    -----------
    The following methods may be overridden to implement information-
    passing and parallel behaviors between client/server module pairs.
    As defined at `OptiModule` level, they have no effect and may thus
    be safely ignored when implementing self-contained algorithms.

    collect_aux_var() -> Optional[Dict[str, Any]]:
        Emit a JSON-serializable dict of auxiliary variables,
        to be received by a counterpart of this module on the
        other side of the client/server relationship.
    process_aux_var(Dict[str, Any]) -> None:
        Process a dict of auxiliary variables, received from
        a counterpart to this module on the other side of the
        client/server relationship.
    """

    name: str = NotImplemented

    @abstractmethod
    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Apply an adaptation algorithm to input gradients.

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
        return NotImplemented

    def collect_aux_var(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return auxiliary variables that need to be shared between nodes.

        Returns
        -------
        aux_var: dict[str, any] or None
            Optional JSON-serializable dict of auxiliary variables that
            are to be shared with a similarly-named OptiModule on the
            other side of the client-server relationship.

        Notes
        -----
        Specfications for the output and calling context depend on whether
        the module is part of a client's optimizer or of the server's one:
        * Client:
          - aux_var is dict[str, any] or None.
          - `collect_aux_var` is expected to happen after taking a series
            of local optimization steps, before sending the local updates
            to the server for aggregation and further processing.
        * Server:
          - aux_var may be None ; dict[str, any] (to send the same values
            to each and every client) ; or dict[str, dict[str, any]] with
            clients' names as keys and client-wise new aux_var as values
            so as to send distinct values to the clients.
          - `collect_aux_var` is expected to happen when the global model
            weights are ready to be shared with clients, i.e. at the very
            end of a training round or at the beginning of the training
            process.
        """
        return None

    def process_aux_var(
        self,
        aux_var: Dict[str, Any],
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Parameters
        ----------
        aux_var: dict[str, any]
            JSON-serializable dict of auxiliary variables that are to be
            processed by this module at the start of a training round (on
            the client side) or before processing global updates (on the
            server side).

        Notes
        -----
        Specfications for the inputs and calling context depend on whether
        the module is part of a client's optimizer or of the server's one:
        * Client:
          - aux_var is dict[str, any] and may be client-specific.
          - `process_aux_var` is expected to happen at the beginning of
            a training round to define gradients' processing during the
            local optimization steps taken through that round.
        * Server:
          - aux_var is dict[str, dict[str, any]] with clients' names as
            primary keys and client-wise collected aux_var as values.
          - `process_aux_var` is expected to happen upon receiving local
            updates (and, thus, aux_var), before the aggregated updates
            are computed and passed through the server optimizer (which
            comprises this module).

        Raises
        ------
        KeyError:
            If an expected auxiliary variable is missing.
        TypeError:
            If a variable is of unproper type, or if aux_var
            is not formatted as it should be.
        """
        # API-defining method; pylint: disable=unused-argument
        return None

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters."""
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "OptiModule":
        """Instantiate an OptiModule from its configuration dict."""
        return cls(**config)

    def serialize(
        self,
    ) -> ObjectConfig:
        """Return an ObjectConfig serialization of this instance."""
        return serialize_object(self, group="OptiModule")

    @classmethod
    def deserialize(
        cls,
        config: Union[str, ObjectConfig],
    ) -> "OptiModule":
        """Instantiate an OptiModule from a JSON configuration file or dict."""
        obj = deserialize_object(config, custom=None)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Configuration specifies a '{type(obj).__name__}' object, "
                f"which is not a subclass of '{cls.__name__}'."
            )
        return obj
