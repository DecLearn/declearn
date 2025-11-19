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
    clients: set[str]
        Set of clients among which sampling is done.
    n_samples: int
        Number of clients sampled each round.
    prior_weights: dict[str, float] or None, default=None
        Prior weights attributed to each client. Default will assign the
        same weights to each client.
    initialization_round: bool, default=False
        If True, all clients will be used during the first round.

    Raises
    ------
    KeyError
        If the weights keys do not match the set of clients.
    ValueError
        If the number of samples is higher that the number of clients.
    """

    def __init__(
        self,
        clients: Set[str],
        n_samples: int,
        prior_weights: Optional[Dict[str, float]] = None,
        initialization_round: bool = False,
    ):
        if set(prior_weights.keys()) != clients:
            raise KeyError(
                f"The weights keys {prior_weights.keys()} do not match "
                f"clients {clients}."
            )

        if n_samples > len(clients):
            raise ValueError(
                f"The number of samples {n_samples} is higher than the "
                f"number of clients {len(clients)}."
            )

        self.clients = clients
        self.n_samples = n_samples
        self.prior_weights = prior_weights
        self.initialization_round = initialization_round

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
        """
        if self.initialization_round:
            self.initialization_round = False
            return self.clients

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
        Back-end to the sample method.
        """

    @abstractmethod
    def update(self, results: Dict[str, Message]):
        """
        Updates the learnt weights.

        Parameters
        ----------
            results: dict[str, Message]
                reply of clients after training
        """


class CompositionClientSampler(ClientSampler):
    """
    Class allowing the composition of a list of samplers

    TODO: order convention to be defined.

    Parameters
    ----------
        samplers: tuple[ClientSampler, ...]
            list of client samplers to be combined.

    Raises
    ------
        ValueError
            if all samplers do not have the same clients set.
            if all samplers do not have the same initialization policy.

    """

    def __init__(self, *samplers: ClientSampler):
        self.check_composition_homogeneity(*samplers)
        self.samplers = samplers

        clients = samplers[0].clients
        n_samples = sum((sampler.n_samples for sampler in samplers))
        initialization_round = samplers[0].initialization_round

        super().__init__(
            clients,
            n_samples,
            initialization_round=initialization_round,
        )

    @staticmethod
    def check_composition_homogeneity(
        *samplers: ClientSampler,
    ):
        """
        Checks that the composition of given samplers is possible.

        Parameters
        ----------
        samplers: tuple[ClientSampler]
            Tuple of client samplers to be composed together

        Raises
        ------
            ValueError
                if all samplers do not have the same clients set.
                if all samplers do not have the same initialization policy.
        """
        if not all(
            (sampler.clients == samplers[0].clients for sampler in samplers)
        ):
            raise ValueError(
                "All samplers composed together should have the same set of "
                "clients."
            )

        if not all(
            (
                sampler.initialization_round
                == samplers[0].initialization_round
                for sampler in samplers
            )
        ):
            raise ValueError(
                "All samplers composed together should have the same "
                "initialization policy."
            )

    def cls_sample(self, input_clients: Set[str]):
        total_sampled_clients = set()
        for sampler in self.samplers:
            sampler_clients = sampler.cls_sample(input_clients)
            total_sampled_clients.update(sampler_clients)
            input_clients = input_clients - sampler_clients

        return total_sampled_clients

    def update(self, results: Dict[str, Message]):
        for sampler in self.samplers:
            sampler.update(results)
