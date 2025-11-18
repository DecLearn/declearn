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

"""Model updates aggregation API."""

import abc
import dataclasses
from typing import Any, ClassVar, Dict, Generic, Self, Type, TypeVar, Union

from declearn.model.api import Vector
from declearn.utils import (
    Aggregate,
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
class ModelUpdates(Aggregate, base_cls=True, register=True):
    """Base dataclass for model updates' sharing and aggregation.

    Each and every `Aggregator` subclass is expected to be coupled with
    one (or multiple) `ModelUpdates` (sub)type(s), that define which data
    is exchanged and how it is aggregated across a network of peers. An
    instance resulting from the aggregation of multiple peers' data may
    be passed to an appropriate `Aggregator` instance for finalization
    into a `Vector` of model updates.

    This class also defines whether contents are compatible with secure
    aggregation, and whether some fields should remain in cleartext no
    matter what.

    Note that subclasses are automatically type-registered, and should be
    decorated as `dataclasses.dataclass`. To prevent registration, simply
    pass `register=False` at inheritance.
    """

    updates: Vector
    weights: Union[int, float]

    _group_key = "ModelUpdates"


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

    - name: str class attribute
        Name identifier of the class (should be unique across Aggregator
        classes). Also used for automatic type-registration of the class
        (see `Inheritance` section below).
    - prepare_for_sharing(updates: Vector, n_steps: int) -> ModelUpdates:
        Pre-process and package local model updates for aggregation.
    - finalize_updates(updates: ModelUpdates):
        Finalize pre-aggregated data into global model updates.

    Overridable
    -----------
    - updates_cls: type[ModelUpdates] class attribute
        Type of 'ModelUpdates' data structure used by this Aggregator class.
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
    """Type of 'ModelUpdates' data structure used by this Aggregator class."""

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
