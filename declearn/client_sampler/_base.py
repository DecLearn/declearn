"""Client Sampler abstraction API"""

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

from declearn.messaging import TrainReply
from declearn.utils import create_types_registry, register_type


@create_types_registry
class ClientSampler(metaclass=ABCMeta):
    """
    Abstract base class for client sampler.

    The aim of this abstraction is to enable implementing client
    sampling strategies, to sample a subset of clients at each
    round instead of selecting them all.

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

    - name: str class attribute
        Name of the client sampler strategy, should match the class name,
        ex: "default"
    - secagg_compatible(): boolean class property
        Indicate if the client sampler is compatible with secure
        aggregation
    - cls_sample():
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
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(ClientSampler, register=False)`)
    See `declearn.utils.register_type` for details on types registration.
    """

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register ClientSampler subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            if not getattr(cls, "name", None):
                raise TypeError(
                    f"{cls.__name__} must define a class attribute 'name'"
                )

            register_type(cls, cls.name, group="ClientSampler")

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
        input_clients: Optional[Set[str]] = None,
    ) -> Set[str]:
        """
        Samples clients a subset of clients.
        TODO explain why input_clients parameter

        Parameters:
        TODO

        Returns
        -------
        Subset of input clients, containing the sampled clients.

        Raises
        ------
        AttributeError
            If clients attribute is not initialized

        ValueErrror
            If the provided clients is not a subset of the attribute 'clients'
        """
        if self.clients is None:
            raise AttributeError(
                "The attribute 'clients' must be initialized before calling "
                "the sample method."
            )
        # check that input subset is valid
        if input_clients is not None and not input_clients.issubset(
            self.clients
        ):
            raise ValueError(
                f"The given client subset {input_clients} is not a subset "
                f"of {self.clients}."
            )

        return self.cls_sample(
            input_clients if input_clients is not None else self.clients
        )

    @abstractmethod
    def cls_sample(self, input_clients: Set[str]) -> Set[str]:
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


class CompositionClientSampler(ClientSampler):
    """
    Class allowing the composition of a list of samplers

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

    name = "composition"

    def __init__(self, *samplers: ClientSampler):
        super().__init__()
        self.samplers: List[ClientSampler] = list(samplers)

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

    def cls_sample(self, input_clients: Set[str]):
        total_sampled_clients = set()
        for sampler in self.samplers:
            sampler_clients = sampler.sample(input_clients)
            total_sampled_clients.update(sampler_clients)
            input_clients = input_clients - sampler_clients

        return total_sampled_clients

    def update(self, results: Dict[str, TrainReply]):
        for sampler in self.samplers:
            sampler.update(results)
