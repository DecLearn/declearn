import pytest

from declearn.client_sampler import DefaultClientSampler
from declearn.client_sampler.modules import UniformClientSampler


class TestClientSampler:
    """Shared unit tests suite for 'ClientSampler' subclasses."""

    def test_default_sampling(self):
        sampler = DefaultClientSampler()
        clients = set(["client_1", "client_2", "client_3"])
        sampler.init_check_clients(clients)
        sampled_clients = sampler.sample()
        assert clients == sampled_clients

    def test_check_clients_unmatch(self):
        expected_clients = set(["client_1", "client_2", "client_3"])
        sampler = DefaultClientSampler(clients=expected_clients)
        actual_clients = set(["client_1", "client_2", "client_4"])
        with pytest.raises(AttributeError):
            sampler.init_check_clients(actual_clients)

    def test_uniform_sampling(self):
        sampler = UniformClientSampler(n_samples=1)
        clients = set(["client_1", "client_2", "client_3"])
        sampler.init_check_clients(clients)
        sampled_client = sampler.sample()
        assert len(sampled_client) == 1
        assert sampled_client.issubset(clients)

    def test_weighted_sampling(self):
        clients = set(["client_1", "client_2", "client_3"])
        prior_weights = {
            "client_1": 1,
            "client_2": 0,
            "client_3": 0,
        }
        sampler = UniformClientSampler(
            n_samples=1, clients=clients, prior_weights=prior_weights
        )
        sampler.init_check_clients(clients)
        sampled_client = sampler.sample()
        assert len(sampled_client) == 1
        assert sampled_client.pop() == "client_1"
