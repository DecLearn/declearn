# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Model updates aggregation API."""

import abc
import dataclasses
import warnings
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar, Union

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Vector
from declearn.utils import (
    access_types_mapping,
    create_types_registry,
    register_type,
)

__all__ = [
    "Aggregator",
    "ModelUpdates",
]


T = TypeVar("T")


@dataclasses.dataclass
class ModelUpdates:
    """Base dataclass for model updates' sharing and aggregation."""

    updates: Vector
    weights: Union[int, float]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this instance."""
        return dataclasses.asdict(self)

    def __add__(
        self,
        other: Any,
    ) -> Self:
        """Overload the sum operator to aggregate multiple instances."""
        try:
            return self.aggregate(other)
        except TypeError:
            return NotImplemented

    def __radd__(
        self,
        other: Any,
    ) -> Self:
        """Enable `0 + ModelUpdates`, to support `sum(List[ModelUpdates])`."""
        if isinstance(other, int) and not other:
            return self
        return NotImplemented

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate this with another instance of the same class.

        Parameters
        ----------
        other:
            Another instance of the same type as `self`.

        Returns
        -------
        aggregated:
            An instance of the same class containing aggregated values.

        Raises
        ------
        TypeError
            If `other` is of unproper type.
        ValueError
            If any field's aggregation fails.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"'{self.__class__.__name__}.aggregate' received a wrongful "
                f"'other'  argument: excepted same type, got '{type(other)}'."
            )
        # Run the fields' aggregation, wrapping any exception as ValueError.
        try:
            results = {
                field.name: (
                    getattr(self, field.name) + getattr(other, field.name)
                )
                for field in dataclasses.fields(self)
            }
        except Exception as exc:
            raise ValueError(
                "Exception encountered while aggregating two instances "
                f"of '{self.__class__.__name__}': {repr(exc)}."
            ) from exc
        # If everything went right, return the resulting ModelUpdates.
        return self.__class__(**results)


ModelUpdatesT = TypeVar("ModelUpdatesT", bound=ModelUpdates)


@create_types_registry
class Aggregator(Generic[ModelUpdatesT], metaclass=abc.ABCMeta):
    """Abstract class defining an API for model updates aggregation.

    The aim of this abstraction (which itself operates on the Vector
    abstraction, so as to provide framework-agnostic algorithms) is
    to enable implementing arbitrary aggregation rules to turn local
    model updates into global updates in a federated or decentralized
    learning context.

    An Aggregator has three main purposes:

    - Preparing and packaging data that is to be shared with peers
      based on local model updates into a `ModelUpdates` container
      that implements aggregation, usually via summation.
    - Finalizing such a data container into model updates.

    Abstract
    --------
    The following class attributes and methods must be implemented
    by any non-abstract child class of `Aggregator`:

    - name:
        Name identifier of the class (should be unique across Aggregator
        classes). Also used for automatic type-registration of the class
        (see `Inheritance` section below).
    - updates_cls:
        Type of `ModelUpdates` on which this class is designed to operate.

    - prepare_for_sharing(updates: Vector, n_steps: int) -> ModelUpdates:
        Pre-process and package local model updates for aggregation.
    - finalize_updates(updates: ModelUpdates):
        Finalize pre-aggregated data into global model updates.

    Overridable
    -----------
    - get_config() -> Dict[str, Any]:
        Return a JSON-serializable configuration dict of an instance.
    - from_config(Dict[str, Any]) -> Aggregator:
        Classmethod to instantiate an Aggregator from a config dict.

    Inheritance
    -----------
    When a subclass inheriting from `Aggregator` is declared, it is
    automatically registered under the "Aggregator" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(Aggregator, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str]
    """Name identifier of the class, unique across Aggregator classes."""

    updates_cls: ClassVar[Type[ModelUpdates]] = ModelUpdates
    """Type of 'ModelUpdates' on which this class is designed to operate."""

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register Aggregator subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.name, group="Aggregator")

    @abc.abstractmethod
    def prepare_for_sharing(
        self,
        updates: Vector,
        n_steps: int,  # revise: generalize kwargs?
    ) -> ModelUpdatesT:
        """Pre-process and package local model updates for aggregation.

        Parameters
        ----------
        updates:
            Local model updates, as a Vector value.
        n_steps:
            Number of local training steps taken to produce `updates`.

        Returns
        -------
        updates:
            Data to be shared with peers, wrapped into a `ModelUpdates`
            (subclass) instance.
        """

    @abc.abstractmethod
    def finalize_updates(
        self,
        updates: ModelUpdatesT,
    ) -> Vector:
        """Finalize pre-aggregated data into global model updates.

        Parameters
        ----------
        updates:
            `ModelUpdates` instance holding aggregated data to finalize,
            resulting from peers' shared instances' sum-aggregation.
        """

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this object's parameters."""
        return {}  # pragma: no cover

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate an Aggregator from its configuration dict."""
        return cls(**config)

    def aggregate(
        self,
        updates: Dict[str, Vector[T]],
        n_steps: Dict[str, int],  # revise: abstract~generalize kwargs use
    ) -> Vector[T]:
        """DEPRECATED - Aggregate input vectors into a single one.

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

        Raises
        ------
        TypeError
            If the input `updates` are an empty dict.
        """
        warnings.warn(
            "'Aggregator.aggregate' was deprecated in DecLearn v2.4 in favor "
            "of new API methods. It will be removed in DecLearn v2.6 and/or "
            "v3.0.",
            DeprecationWarning,
        )
        if not updates:
            raise TypeError("'Aggregator.aggregate' received an empty dict.")
        partials = [
            self.prepare_for_sharing(updates[client], n_steps[client])
            for client in updates
        ]
        aggregated = sum(partials[1:], start=partials[0])
        return self.finalize_updates(aggregated)


def list_aggregators() -> Dict[str, Type[Aggregator]]:
    """Return a mapping of registered Aggregator subclasses.

    This function aims at making it easy for end-users to list and access
    all available Aggregator classes at any given time. The returned dict
    uses unique identifier keys, which may be used to specify the desired
    algorithm as part of a federated learning process without going through
    the fuss of importing and instantiating it manually.

    Note that the mapping will include all declearn-provided plug-ins,
    but also registered plug-ins provided by user or third-party code.

    See also
    --------
    * [declearn.aggregator.Aggregator][]:
        API-defining abstract base class for the aggregation algorithms.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to `Aggregator` class
        constructors.
    """
    return access_types_mapping("Aggregator")
