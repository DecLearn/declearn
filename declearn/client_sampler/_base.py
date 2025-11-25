"""Implementation of Client Sampler"""

from abc import ABCMeta, abstractmethod
from typing import (
    Dict,
    Optional,
    Set,
)

from declearn.messaging import Message


class ClientSampler(metaclass=ABCMeta):
    """

    Attributes
    ----------
    # TODO fix
    clients: Optional[Set[str]]
        Set of clients among which sampling is done.
    n_samples: int TODO remove
        Number of clients sampled each round.
    prior_weights: dict[str, float] or None, default=None
        Prior weights attributed to each client.
        Higher weight for a client should mean higher chances of being
        sampled, depends on the implementation class.
        Default will assign the same weights to each client.
    initialization_round: bool, default=False
        If True, all clients will be used during the first round.
    secagg_compatible : TODO

    TODO : Abstract, Overridable, Inheritance

    """

    def __init__(
        self,
        clients: Optional[Set[str]] = None,
        prior_weights: Optional[Dict[str, float]] = None,
        initialization_round: bool = False,
    ):
        """
        TODO full doc
        clients: expected set of all clients in the federated process

        Raises
        ------
        KeyError
            If the weights keys do not match the set of clients.
        ValueError
            If the number of samples is higher that the number of clients.
        """
        if clients is None:
            if prior_weights is not None:
                raise ValueError(
                    "'prior_weights' cannot be set if 'clients' is not set"
                )
        elif (
            prior_weights is not None and set(prior_weights.keys()) != clients
        ):
            raise KeyError(
                f"The weights keys {prior_weights.keys()} do not match "
                f"clients {clients}."
            )

        # TODO : move constraint below elsewhere
        # if n_samples > len(clients):
        #     raise ValueError(
        #         f"The number of samples {n_samples} is higher than the "
        #         f"number of clients {len(clients)}."
        #     )
        self.clients = clients
        self.prior_weights = prior_weights
        self.initialization_round = initialization_round

    def init_check_clients(self, actual_clients: Set[str]) -> None:
        """
        Initialize or check clients attribute in the sampler

        If 'clients' attribute is not set, initialize it with provided
        clients.
        Else, check that already-initialized clients match provided
        clients.

        Parameters
        ----------
            actual_clients: the set of all clients involved in the federated
            process

        Raises
        ------

        """
        if self.clients is None:
            self.clients = actual_clients
        elif self.clients != actual_clients:
            raise AttributeError(
                f"Initialized set of clients {self.clients} does not match "
                f"actual set of clients {actual_clients}"
            )

    def sample(
        self,
        input_clients: Optional[Set[str]] = None,
    ) -> Set[str]:
        """
        Samples a subset of clients.

        Parameters
        ----------
            input_clients: subset of clients to sample from.
            Default will use the complete set of clients.

        Returns
        -------
            sampled_clients: subset of sampled clients.

        TODO Raises
        """
        if self.initialization_round:
            self.initialization_round = False
            return self.clients

        # check that clients attribute is not unassigned
        if self.clients is None or len(self.clients) == 0:
            raise AttributeError(
                "The client sampler must have a non-empty set of clients "
                "before performing the sampling."
            )
        # check that input subset is valid
        elif input_clients is not None and not input_clients.issubset(
            self.clients
        ):
            raise ValueError(
                f"The given client subset {input_clients} is not a subset "
                f"of {self.clients}."
            )

        return self._sample(
            input_clients if input_clients is not None else self.clients
        )

    @abstractmethod
    def _sample(self, input_clients: Set[str]) -> Set[str]:
        """
        Back-end to the sample method.
        """

    @abstractmethod
    def update(self, results: Dict[str, Message]) -> None:
        """
        Updates the learnt weights.

        Parameters
        ----------
            results: dict[str, Message]
                reply of clients after training
        """


class DefaultClientSampler(ClientSampler):
    """
    Default client sampler which actually don't sample, because
    it selects all clients.
    """

    secagg_compatible = True

    def _sample(self, input_clients: Set[str]) -> Set[str]:
        return input_clients

    def update(self, results: Dict[str, Message]) -> None:
        pass


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
