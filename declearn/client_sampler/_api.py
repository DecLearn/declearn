# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Client sampler abstraction API."""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Set,
    Type,
)

from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.utils import (
    access_registered,
    access_types_mapping,
    create_types_registry,
    register_from_attr,
)


@create_types_registry
class ClientSampler(metaclass=ABCMeta):
    """Abstract base class for client sampler.

    The aim of this abstraction is to enable implementing client
    sampling strategies, to sample a subset of clients at each
    round instead of selecting them all.

    Attributes
    ----------
    strategy: str class attribute
        See details in the Abstract section.

    secagg_compatible: boolean read-only class property
        See details in the Abstract section.

    clients: Set[str]
        Set of clients among which sampling is done.

    max_retries: int
        Maximum number of consecutive retries performed by the sampler if
        the sampling fails (i.e. if no client is selected).

    logger: logging.Logger
        The logger associated to this class. By default, it is named
        "declearn.server.client_sampler" for any ClientSampler instance.

    Key methods
    -----------
    - sample(eligible_clients):
        Perform the client sampling.
    - update(client_to_reply, global_model):
        Update sampler internal state.

    Abstract
    --------
    The following attributes and methods must be implemented by any
    non-abstract child class:

    - strategy: str class attribute
        Name of the client sampler strategy, should match the class name and be
        unique accross `ClientSampler` classes,
        e.g. "default" for `DefaultClientSampler`.
    - secagg_compatible(): boolean read-only class property
        Indicate if the client sampler is compatible with secure aggregation
    - cls_sample(eligible_clients):
        Class-specific back-end of the common sampling method `sample`.

    Overridable
    -----------
    - init_clients(clients):
        Instance method that initializes clients and their metadata in the
        sampler.
        Can be overriden (or extended) to precisely initialize
        some metadata used in the strategy of the sampler
        subclass.

    - update(client_to_reply, global_model):
        Instance method that update the sampler internal state.
        Can be overriden to update internal states that are specific to the
        client sampler subclass.

    - from_specs(cls, **kwargs):
        Class method, create an instance from specifications, can be overriden
        by subclass if specific mechanisms are needed to allow a proper
        instanciation from specifications.

    Inheritance
    -----------
    When a subclass inheriting from `ClientSampler` is declared, it is
    automatically registered under the "ClientSampler" group using its
    class-attribute `strategy`. This can be prevented by adding
    `register=False` to the inheritance specs
    (e.g. `class MyCls(ClientSampler, register=False)`).
    See `declearn.utils.register_type` for details on types registration.

    Notes
    -----
    You can access and configure this class logger using
    `logger = logging.getLogger("declearn.server.client_sampler")`, and then
    adjust it as needed (e.g. `logger.setLevel(...)`).
    """

    DEFAULT_MAX_RETRIES = 5
    """Default maximum number of retries for the init method."""

    strategy: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register `ClientSampler` subclasses if
        enabled.
        """
        super().__init_subclass__(**kwargs)
        if register:
            register_from_attr(cls, "strategy", group="ClientSampler")

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Instantiate a client sampler.

        Parameters
        ----------
        max_retries:
            Maximum number of consecutive retries performed by the sampler if
            the sampling fails (i.e. if no client is selected).
        """
        self.clients: Set[str] = set()
        self.max_retries = max_retries
        self.logger = logging.getLogger(
            "declearn.server.client_sampler",
        )

    @property
    @abstractmethod
    def secagg_compatible(self) -> bool:
        """Class read-only property to indicate if the client sampler is
        compatible with secure aggregation.
        """

    def init_clients(self, clients: Set[str]) -> None:
        """Initialize clients in the sampler.

        This method can be overriden by subclasses, but if so, it should
        ideally be extended (call to super().init_clients() at first, then add
        new code).

        Parameters
        ----------
        clients:
            Set of all clients involved in the federated process.
        """
        self.clients.update(clients)

    def sample(
        self,
        eligible_clients: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Sample clients among the provided eligible clients.

        It is the entrypoint method for the client sampling, including
        mechanisms common to all client samplers (e.g. retry capability).

        If eligible_client is None, samples among the full clients set.

        Notes
        -----
        If no client is selected after the sampling action, it is
        retried until at least one client is sampled or until the number of
        max_retries (instance attribute) is reached.

        Caution: If the number of max_retries is reached, all the clients
        that have been registered in the federated process will be selected by
        default (legacy behavior before the integration of client sampling).

        Parameters
        ----------
        eligible_clients:
            Optional subset of all clients among which the
            sampling has to be made, if None: all clients are considered.

            This parameter allows to constrain the sampling to a subset of
            clients, which is useful for client sampler composition.

        Returns
        -------
        Subset of eligible clients, containing the sampled clients.

        Raises
        ------
        AttributeError
            If clients attribute is not initialized (is empty).

        ValueErrror
            If the provided clients set is not a subset of the 'clients'
            attribute.
        """
        if self.clients == set():
            raise AttributeError(
                "The clients set is empty, it must be initialized before "
                "calling the sample method."
            )

        if eligible_clients is not None and not eligible_clients.issubset(
            self.clients
        ):
            raise ValueError(
                f"The given client subset {eligible_clients} is not a subset "
                f"of {self.clients}."
            )

        if eligible_clients is None:
            eligible_clients = self.clients

        nb_retries = 0
        retry = True
        while retry:
            sampled_clients = self.cls_sample(eligible_clients)
            if len(sampled_clients) > 0:
                retry = False
            elif nb_retries < self.max_retries:
                nb_retries += 1
            else:  # no client sampled and max number of retries reached
                self.logger.warning(
                    f"No client was sampled after {self.max_retries} "
                    "attempts. Falling back to selecting all provided clients."
                )
                sampled_clients = eligible_clients
                retry = False
        return sampled_clients

    @abstractmethod
    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        """Class-specific back-end of the `sample` method.

        Implementation of the precise client sampling algorithm, specific to
        the subclass.
        """

    def update(
        self, client_to_reply: Dict[str, TrainReply], global_model: Model
    ) -> None:
        """Update sampler internal state using clients training reply and
        the global model.

        By default, this method does nothing. If your concrete client sampler
        subclass needs to update its internal state, override it with a
        proper implementation.

        Notes
        -----
        The parameters must be considered read-only, do not modify them
        if you override this method.

        Parameters
        ----------
        client_to_reply:
            Dictionary mapping each client to their training reply.

        global_model:
            Global model hold by the server.
        """
        return None

    @classmethod
    def from_specs(cls, **kwargs: Any) -> ClientSampler:
        """Instantiate a client sampler of the given class from its
        specifications.

        Can be overriden by subclass if specific mechanisms are needed to
        allow a proper instanciation from specifications.
        """
        return cls(**kwargs)


def list_client_samplers() -> Dict[str, Type[ClientSampler]]:
    """Return a mapping of registered `ClientSampler` subclasses.

    This function aims at making it easy for end-users to list and access
    all available `ClientSampler` classes at any given time.

    Note that the mapping will include all declearn-provided client samplers,
    but also registered one provided by user or third-party code.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to `ClientSampler`
        class constructors.
    """
    return access_types_mapping("ClientSampler")


def instantiate_client_sampler(strategy: str, **kwargs: Any) -> ClientSampler:
    """Instantiate a `ClientSampler` from its specifications.

    The value of the `strategy` argument identifies which subclass to
    instantiate, by matching against each subclass's `strategy` class variable.

    Parameters
    ----------
    strategy:
        Name of the strategy associated with the target `ClientSampler`
        subclass.
    **kwargs:
        Any additional instantiation keyword argument (general or
        strategy-specific).

    Returns
    -------
    client_sampler:
        `ClientSampler` instance matching input specifications.

    Raises
    ------
    ValueError
        If `strategy` does not match any registered `ClientSampler` type,
        or more generally if specifications are invalid.
    """
    try:
        cls = access_registered(strategy, group="ClientSampler")
    except KeyError as e:
        raise ValueError(
            f"Unknown client sampler strategy '{strategy}'."
        ) from e

    try:
        return cls.from_specs(**kwargs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid client sampler specifications: {e}.") from e
