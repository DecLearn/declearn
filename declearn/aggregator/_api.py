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

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Vector
from declearn.utils import create_types_registry, register_type

__all__ = [
    "Aggregator",
]


@create_types_registry
class Aggregator(metaclass=ABCMeta):
    """Abstract class defining an API for Vector aggregation.

    The aim of this abstraction (which itself operates on the Vector
    abstraction, so as to provide framework-agnostic algorithms) is
    to enable implementing arbitrary aggregation rules to turn local
    model updates into global updates in a federated learning context.

    An Aggregator is typically meant to be used on a round-wise basis
    by the orchestrating server of a centralized federated learning
    process, to aggregate the client-wise model updated into a Vector
    that may then be used as "gradients" by the server's Optimizer to
    update the global model.

    Abstract
    --------
    The following attribute and method require to be overridden
    by any non-abstract child class of `Aggregator`:

    name: str class attribute
        Name identifier of the class (should be unique across existing
        Aggregator classes). Also used for automatic types-registration
        of the class (see `Inheritance` section below).
    aggregate(updates: Dict[str, Vector], n_steps: Dict[str, int]) -> Vector:
        Aggregate input vectors into a single one.
        This is the main method for any `Aggregator`.

    Overridable
    -----------
    get_config() -> Dict[str, Any]:
        Return a JSON-serializable configuration dict of an instance.
    from_config(Dict[str, Any]) -> Aggregator:
        Classmethod to instantiate an Aggregator from a config dict.

    Inheritance
    -----------
    When a subclass inheriting from `Aggregator` is declared, it is
    automatically registered under the "Aggregator" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(Aggregator, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str] = NotImplemented

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register Aggregator subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.name, group="Aggregator")

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

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this object's parameters."""
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate an Aggregator from its configuration dict."""
        return cls(**config)
