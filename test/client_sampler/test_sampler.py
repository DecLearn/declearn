"""Unit tests for the 'ClientSampler' subclasses."""

from typing import Set

import pytest

from declearn.client_sampler import CompositionClientSampler
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
        Test gradient norm criterion client sampling

        Notes: uses the train_replies fixture with one arbitrary fixed framework:
        torch
        Overrides the "compute" method thanks to the pytest feature
        "monkeypatch"
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
        sampled_clients = sampler.sample()
        assert len(sampled_clients) == 2
        # client 2 and 3 have the highest weights (2 and 3)
        # so they must be chosen
        assert sampled_clients == {"client2", "client3"}

    @pytest.mark.parametrize("framework", ["torch"])
    def test_compo_crit_unif_sampling(
        self, clients, train_replies, monkeypatch
    ):
        """
        Test composition client sampler with a gradient norm criterion client
        sampling and then a uniform sampling

        Note: we use the same mocking method (with monkeypatch) as in
        'test_criterion_sampling'
        """
        criterion = GradientNormCriterion()
        fake_client_to_norm = {
            client: float(i + 1) for i, client in enumerate(sorted(clients))
        }
        monkeypatch.setattr(
            criterion, "compute", lambda *_: fake_client_to_norm
        )

        crit_sampler = CriterionClientSampler(
            n_samples=1,
            criterion=criterion,
            missing_weights_policy="priority",
        )
        unif_sampler = UniformClientSampler(n_samples=1)
        compo_sampler = CompositionClientSampler(crit_sampler, unif_sampler)

        compo_sampler.init_clients(clients)
        # update the weights using the fake gradient norms
        compo_sampler.update(train_replies)
        sampled_clients = compo_sampler.sample()

        # first, the criterion sampler should have selected client3 and then
        # the uniform sampler should have picked randomly one among the others
        assert len(sampled_clients) == 2
        assert "client3" in sampled_clients
