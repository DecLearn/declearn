"""Client Sampler abstraction API"""

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Dict,
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
    to the inheritance specs (e.g. `class MyCls(ClientSampler, register=False)`).
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
        ideally be extended (call to super().init_clients() at first, then add new
        code)

        Parameters
        ----------
            clients: the set of all clients involved in the federated
            process
        """
        self.clients: Set[str] = clients
        self.client_to_metadata: Dict[str, Dict[str, Any]] = {
            client: {} for client in clients
        }

    def sample(self) -> Set[str]:
        """
        Samples a subset of clients.

        Returns
        -------
            sampled_clients: subset of sampled clients.

        Raises
        ------aises
            AttributeError
                If clients attribute is not initialized or equals zero
        """
        # check that clients attribute is not unassigned
        if self.clients is None or len(self.clients) == 0:
            raise AttributeError(
                "The client sampler must have a non-empty set of clients "
                "before performing the sampling."
            )
        return self.cls_sample()

    @abstractmethod
    def cls_sample(self) -> Set[str]:
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


# TODO handle composition
# class CompositionClientSampler(ClientSampler):
#     """
#     Class allowing the composition of a list of samplers

#     TODO: order convention to be defined.

#     Parameters
#     ----------
#         samplers: tuple[ClientSampler, ...]
#             list of client samplers to be combined.

#     Raises
#     ------
#         ValueError
#             if all samplers do not have the same clients set.
#             if all samplers do not have the same initialization policy.

#     """

#     name = "composition"

#     def __init__(self, *samplers: ClientSampler):
#         self.check_composition_homogeneity(*samplers)
#         self.samplers = samplers

#         clients = samplers[0].clients
#         n_samples = sum((sampler.n_samples for sampler in samplers))
#         initialization_round = samplers[0].initialization_round

#         super().__init__(
#             clients,
#             n_samples,
#             initialization_round=initialization_round,
#         )

#     @staticmethod
#     def check_composition_homogeneity(
#         *samplers: ClientSampler,
#     ):
#         """
#         Checks that the composition of given samplers is possible.

#         Parameters
#         ----------
#         samplers: tuple[ClientSampler]
#             Tuple of client samplers to be composed together

#         Raises
#         ------
#             ValueError
#                 if all samplers do not have the same clients set.
#                 if all samplers do not have the same initialization policy.
#         """
#         if not all(
#             (sampler.clients == samplers[0].clients for sampler in samplers)
#         ):
#             raise ValueError(
#                 "All samplers composed together should have the same set of "
#                 "clients."
#             )

#         if not all(
#             (
#                 sampler.initialization_round
#                 == samplers[0].initialization_round
#                 for sampler in samplers
#             )
#         ):
#             raise ValueError(
#                 "All samplers composed together should have the same "
#                 "initialization policy."
#             )

#     def _sample(self, input_clients: Set[str]):
#         total_sampled_clients = set()
#         for sampler in self.samplers:
#             sampler_clients = sampler._sample(input_clients)
#             total_sampled_clients.update(sampler_clients)
#             input_clients = input_clients - sampler_clients

#         return total_sampled_clients

#     def update(self, results: Dict[str, Message]):
#         for sampler in self.samplers:
#             sampler.update(results)
