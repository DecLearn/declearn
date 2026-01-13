"""Client Sampler abstraction API"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
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
    """
    Abstract base class for client sampler.

    The aim of this abstraction is to enable implementing client
    sampling strategies, to sample a subset of clients at each
    round instead of selecting them all.

    Attributes
    ----------
    - strategy: str class attribute
        See details in the Abstract section.

    - secagg_compatible: boolean read-only class property
        See details in the Abstract section.

    - clients: Set[str]
        Set of clients among which sampling is done.

    - client_to_metadata: Dict[str, Dict[str, Any]]
        Dictionary mapping each client name with its metadata dictionary,
        itself mapping the metadata name with its value (of arbitrary type).
        This metadata could be used in a selection strategy.

    - max_retries: int
        Maximum number of consecutive retries performed by the sampler if
        the sampling fails (i.e. if no client is selected).

    Abstract
    --------
    The following attributes and methods must be implemented by any
    non-abstract child class:

    - strategy: str class attribute
        Name of the client sampler strategy, should match the class name and be
        unique accross `ClientSampler` classes,
        e.g. "default" for `DefaultClientSampler
    - secagg_compatible(): boolean read-only class property
        Indicate if the client sampler is compatible with secure
        aggregation
    - _sample():
        Back-end of the sampling method.
    - update(client_to_reply: Dict[str, Message], server_model: Model):
        Update clients metadata and sampler internal state.

    Overridable
    -----------
    - init_clients(clients):
        Instance method that initializes clients and their metadata in the
        sampler.
        Can be overriden (or extended) to precisely initialize
        some metadata used in the strategy of the sampler
        subclass.

    - _from_specs(cls, **kwargs):
        Class method, backend of the `from_specs` method, can be overriden by
        subclass if specific mechanisms are needed to allow a proper
        instanciation from specifications.

    Inheritance
    -----------
    When a subclass inheriting from `ClientSampler` is declared, it is
    automatically registered under the "ClientSampler" group using its
    class-attribute `strategy`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(ClientSampler, register=False)`)
    See `declearn.utils.register_type` for details on types registration.
    """

    DEFAULT_MAX_RETRIES = 5
    """Default maximum number of retries for the init method"""

    strategy: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register ClientSampler subclasses if enabled."""
        super().__init_subclass__(**kwargs)
        if register:
            register_from_attr(cls, "strategy", "ClientSampler")

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Instantiate a client sampler.

        Parameters
        ----------
        max_retries: int
            Maximum number of consecutive retries performed by the sampler if
            the sampling fails (i.e. if no client is selected).
        """
        self.clients: Set[str] = {}
        self.client_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_retries = max_retries
        self._logger = logging.getLogger(
            "FederatedServer.client_sampler",
        )  # FIXME to adapt during rework on the logging system
        # because for now, if the parent logger is not named "FederatedServer"
        # this logger won't be attach to it

    @property
    @abstractmethod
    def secagg_compatible(self) -> bool:
        """
        Class read-only property to indicate if the client sampler is compatible with secure
        aggregation.
        """

    def init_clients(self, clients: Set[str]) -> None:
        """
        Initialize clients and their metadata in the sampler.

        This method can be overriden by subclasses, but if so, it should
        ideally be extended (call to super().init_clients() at first, then add
        new code).

        Parameters
        ----------
        clients: Set[str]
            Set of all clients involved in the federated process
        """
        self.clients: Set[str] = clients
        self.client_to_metadata: Dict[str, Dict[str, Any]] = {
            client: {} for client in clients
        }

    def sample(
        self,
        eligible_clients: Optional[Set[str]] = None,
    ) -> Set[str]:
        """
        Samples clients among the provided eligible clients, or among
        the full clients set.

        Note : If no client is selected after the sampling action, it is
        retried until at least one client is sampled or until the number of
        max_retries (instance attribute) is reached.

        TODO explain when max retries reached, take them all (arbitrary choice)

        Parameters
        ----------
        eligible_clients: Optional[Set[str]]
            optional subset of all clients among which the
            sampling has to be made, if None: all clients are considered.

            This parameter allows to constrain the sampling to a subset of
            clients, e.g. useful if we use two samplers consecutively: the first
            would sample among all clients, the second sampler among the clients
            that have not been selected by the first sampler.

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
        if self.clients == {}:
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
            sampled_clients = self._sample(eligible_clients)
            if len(sampled_clients) > 0:
                retry = False
            elif nb_retries < self.max_retries:
                nb_retries += 1
            else:  # no client sampled and max number of retries reached
                self._logger.warning(
                    f"No client was sampled after {self.max_retries} attempts. "
                    "Falling back to selecting all provided clients."
                )
                sampled_clients = eligible_clients
                retry = False
        return sampled_clients

    @abstractmethod
    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        """
        Back-end of the sampling method.
        """

    @abstractmethod
    def update(
        self, client_to_reply: Dict[str, TrainReply], server_model: Model
    ) -> None:
        """
        Update clients metadata and sampler internal state according
        to each client training reply and the server model.

        Note: The parameters must be considered read-only, do not modify them
        when defining the concrete method.

        Parameters
        ----------
        client_to_reply: Dict[str, Message]
            Dictionary mapping each client to their training reply.

        server_model: Model
            Central server model.
        """

    @staticmethod
    def from_specs(strategy: str, **kwargs: Any) -> ClientSampler:
        """
        Instantiate a 'ClientSampler' from its specifications.

        Parameters
        ----------
        strategy:
            Name of the strategy associated with the target ClientSampler
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
            return cls._from_specs(**kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid client sampler specifications: {e}."
            ) from e

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        Backend of the from_specs method, specific to the subclass.

        Can be overriden by subclass if specific mechanisms are needed to
        allow a proper instanciation from specifications.
        """
        return cls(**kwargs)


class CompositionClientSampler(ClientSampler):
    """
    Class allowing the composition of a list of samplers, i.e. the use of
    multiple samplers consecutively.

    The composition mechanism works the following way: the first sampler
    selects some client(s), then the second one selects other(s) among the
    remaining ones, and so on.
    At the end, the clients selected by the 'CompositionClientSampler' are the
    union of client sets selected by each sampler, consecutively.

    Attributes
    ----------
    samplers: List[ClientSampler, ...]
        list of client samplers to be combined.
    """

    strategy = "composition"

    def __init__(
        self,
        samplers: List[ClientSampler],
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        super().__init__(max_retries=max_retries)
        self.samplers = samplers

    @property
    def secagg_compatible(self) -> bool:
        """
        Composition client sampler is secagg-compatible if all of its
        samplers are.
        """
        return all([sampler.secagg_compatible for sampler in self.samplers])

    def init_clients(self, clients: Set[str]) -> None:
        super().init_clients(clients)
        for sampler in self.samplers:
            sampler.init_clients(clients)

    def _sample(self, eligible_clients: Set[str]):
        total_sampled_clients = set()
        for sampler in self.samplers:
            sampler_clients = sampler.sample(eligible_clients)
            total_sampled_clients.update(sampler_clients)
            eligible_clients = eligible_clients - sampler_clients

        return total_sampled_clients

    def update(
        self, client_to_reply: Dict[str, TrainReply], server_model: Model
    ):
        for sampler in self.samplers:
            sampler.update(client_to_reply, server_model)

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        Backend of the from_specs method, specific to
        'CompositionClientSampler'.

        Note: each sampler composing the 'samplers' list can be either
        a dictionnary of valid sampler specification, or an instance of
        ClientSampler.
        """
        samplers = kwargs["samplers"]
        parsed_samplers = []
        for sampler in samplers:
            if isinstance(sampler, ClientSampler):
                parsed_samplers.append(sampler)
            elif isinstance(sampler, dict):
                parsed_samplers.append(ClientSampler.from_specs(**sampler))
            else:
                raise ValueError(
                    f"Unsupported sampler type '{type(sampler)}' in "
                    "samplers list"
                )
        kwargs["samplers"] = parsed_samplers
        return cls(**kwargs)


def list_client_samplers() -> Dict[str, Type[ClientSampler]]:
    """Return a mapping of registered ClientSampler subclasses.

    This function aims at making it easy for end-users to list and access
    all available ClientSampler classes at any given time.

    Note that the mapping will include all declearn-provided client samplers,
    but also registered one provided by user or third-party code.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to `ClientSampler`
        class constructors.
    """
    return access_types_mapping("ClientSampler")
