"""Unit tests for the 'ClientSampler' subclasses."""

from typing import Set

import pytest

from declearn.client_sampler.modules import (
    CriterionClientSampler,
    DefaultClientSampler,
    GradientNormCriterion,
    UniformClientSampler,
)


@pytest.fixture
def clients() -> Set[str]:
    return set(["client1", "client2", "client3"])


class TestClientSampler:
    """Shared unit tests suite for 'ClientSampler' subclasses."""

    def test_default_sampling(self, clients):
        sampler = DefaultClientSampler()
        sampler.init_clients(clients)
        sampled_clients = sampler.sample()
        assert clients == sampled_clients

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_uniform_sampling(self, n_samples, clients):
        sampler = UniformClientSampler(n_samples=n_samples)
        sampler.init_clients(clients)
        sampled_client = sampler.sample()
        assert len(sampled_client) == n_samples
        assert sampled_client.issubset(clients)

    @pytest.mark.parametrize("framework", ["torch"])
    def test_criterion_sampling(self, clients, train_replies, monkeypatch):
        """
        Note: uses the train_replies fixture with one arbitrary fixed framework: torch
        Overrides the "compute" method thanks to the pytest feature "monkeypatch"
        """
        criterion = GradientNormCriterion()

        # we associate artificially each client to a norm equal to its id
        # e.g. client1 has a norm of 1, client2 a norm of 2
        fake_client_to_norm = {
            client: float(i + 1) for i, client in enumerate(sorted(clients))
        }
        # mock (override) the method "compute" of the criterion
        # to return the fake norms
        monkeypatch.setattr(
            criterion, "compute", lambda *_: fake_client_to_norm
        )

        sampler = CriterionClientSampler(
            n_samples=2,
            criterion=criterion,
            missing_weights_policy="priority",
        )
        sampler.init_clients(clients)
        # update the weights using the fake gradient norms
        sampler.update(train_replies)
        sampled_client = sampler.sample()
        assert len(sampled_client) == 2
        # client 2 and 3 have the highest weights (2 and 3)
        # so they must be chosen
        assert sampled_client == {"client2", "client3"}
