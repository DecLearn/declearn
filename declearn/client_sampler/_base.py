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

    TODO info on max_retries

    Attributes
    ----------
    clients: Set[str]
        Set of clients among which sampling is done.

    client_to_metadata: Dict[str, Dict[str, Any]]
        Dictionary mapping each client name with its metadata dictionary,
        itself mapping the metadata name with its value (of arbitrary type).
        This metadata could be used in a selection strategy.

    secagg_compatible: boolean
        Flag that is True if the client sampler is compatible with secure
        aggregation, False otherwise. Must be defined by the subclass.

    Abstract
    --------
    The following attributes and methods must be implemented by any
    non-abstract child class:

    - strategy: str class attribute
        Name of the client sampler strategy, should match the class name,
        ex: "default"
    - secagg_compatible(): boolean class property
        Indicate if the client sampler is compatible with secure
        aggregation
    - _sample():
        Back-end of the sampling method.
    - update(results: Dict[str, Message]):
        Update clients metadata and sampler internal state.

    Overridable
    -----------
    - init_clients(clients: Set[str]):
        Initialize clients and their metadata in the sampler.
        Can be overriden (extended) to precisely initialize
        some metadata used by in the strategy of the sampler
        subclass.

    Inheritance
    -----------
    When a subclass inheriting from `ClientSampler` is declared, it is
    automatically registered under the "ClientSampler" group using its
    class-attribute `strategy`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(ClientSampler, register=False)`)
    See `declearn.utils.register_type` for details on types registration.
    """

    DEFAULT_MAX_RETRIES = 5

    strategy: ClassVar[str]
    """Name identifier of the class, unique across ClientSampler classes."""

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
        TODO
        """
        self.max_retries = max_retries
        self.clients: Set[str] = {}
        self.client_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(
            "client_sampler"
        )  # FIXME to adapt during rework on the logging system

    @property
    @abstractmethod
    def secagg_compatible(self) -> bool:
        """
        Property to indicate if the client sampler is compatible with secure
        aggregation
        """

    def init_clients(self, clients: Set[str]) -> None:
        """
        Initialize clients and their metadata in the sampler.

        This method can be overriden by subclasses, but if so, it should
        ideally be extended (call to super().init_clients() at first, then add
        new code)

        Parameters
        ----------
            clients: the set of all clients involved in the federated
            process
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
        the full clients set

        TODO info on retries

        Parameters:
        eligible_clients: optional subset of all clients among which the
        sampling has to be made, if None: all clients are considered.

        This parameter allows to constrain the sampling to a subset of clients,
        e.g. useful if we use consecutively two samplers, the first would
        sample among all clients, the second sampler among the clients that
        have not been selected by the first sampler.

        Returns
        -------
        Subset of eligible clients, containing the sampled clients.

        Raises
        ------
        AttributeError
            If clients attribute is not initialized (= is empty)

        ValueErrror
            If the provided clients set is not a subset of the 'clients'
            attribute
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
                self.logger.warning(
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
    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        """
        Update clients metadata and sampler internal state according
        to each client training reply.

        Parameters
        ----------
            client_to_reply: dict[str, Message]
                dictionary mapping each client to their training reply
        """

    @staticmethod
    def from_specs(strategy: str, **kwargs: Any) -> ClientSampler:
        """
        TODO
        Raises : ...
        """
        try:
            cls = access_registered(strategy, group="ClientSampler")
        except KeyError as e:
            raise ValueError(
                f"Unknown client sampler strategy '{strategy}'"
            ) from e

        try:
            return cls._from_specs(**kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid client sampler specifications: {e}"
            ) from e

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        TODO
        """
        return cls(**kwargs)


class CompositionClientSampler(ClientSampler):
    """
    Class allowing the composition of a list of samplers, i.e. the use of
    multiple samplers consecutively

    TODO: order convention to be detailed
    (first sampler sample, then the others does among the remaining ones)

    Parameters
    ----------
        samplers: List[ClientSampler, ...]
            list of client samplers to be combined.

    Raises
    ------
        ValueError
            if all samplers do not have the same clients set.
            if all samplers do not have the same initialization policy.

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
        samplers are
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

    def update(self, results: Dict[str, TrainReply]):
        for sampler in self.samplers:
            sampler.update(results)

    @classmethod
    def _from_specs(cls, **kwargs: Any) -> ClientSampler:
        """
        TODO
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
