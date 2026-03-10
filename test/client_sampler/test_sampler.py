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

"""Unit tests for the 'ClientSampler' subclasses."""

import math
from collections import Counter
from typing import Dict, Set

import pytest

from declearn.client_sampler import (
    CompositionClientSampler,
    CriterionClientSampler,
    DefaultClientSampler,
    UniformClientSampler,
    WeightedClientSampler,
)
from declearn.client_sampler.criterion import GradientNormCriterion
from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.test_utils import FailClientSampler


class TestClientSampler:
    """Shared unit tests suite for 'ClientSampler' subclasses."""

    def test_default_sampling(self, clients: Set[str]):
        sampler = DefaultClientSampler()
        sampler.init_clients(clients)
        sampled_clients = sampler.sample()
        assert clients == sampled_clients

    def test_sampling_fail(self, clients: Set[str]):
        fail_sampler = FailClientSampler()
        fail_sampler.init_clients(clients)
        sampled_clients = fail_sampler.sample()
        assert sampled_clients == clients

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_uniform_sampling(self, n_samples: int, clients: Set[str]):
        sampler = UniformClientSampler(n_samples=n_samples)
        sampler.init_clients(clients)
        sampled_client = sampler.sample()
        assert len(sampled_client) == n_samples
        assert sampled_client.issubset(clients)

    def test_weighted_sampling(self, clients: Set[str]):
        client_to_weight = {
            "client1": 0,  # zero-valued weight, so should not be sampled
            "client2": 1,
            "client3": 2,
            "client4": 3,  # anticipated client, not in the actual one
        }
        sampler = WeightedClientSampler(
            n_samples=2, client_to_weight=client_to_weight
        )
        sampler.init_clients(clients)
        sampled_clients = sampler.sample()
        expected_clients = {"client2", "client3"}
        assert sampled_clients == expected_clients

    def test_weighted_sampling_check_proportions(self, clients: Set[str]):
        """
        In this test, we repeat many times the independent sampling of one
        client and store the successive results in a list to check at the end
        that the proportions of each client match approximately the
        user-provided weights.
        """
        client_to_weight = {
            "client1": 0,  # zero-valued weight, so should not be sampled
            "client2": 2,
            "client3": 8,
        }
        sampler = WeightedClientSampler(
            n_samples=1, client_to_weight=client_to_weight
        )
        sampler.init_clients(clients)
        sampled_clients = []
        for _ in range(10_000):
            sampled_client = sampler.sample().pop()
            sampled_clients.append(sampled_client)
        counts = Counter(sampled_clients)
        total = counts.total()
        proportions = {k: v / total for k, v in counts.items()}
        assert "client1" not in counts
        assert math.isclose(2 / 10, proportions["client2"], abs_tol=0.05)
        assert math.isclose(8 / 10, proportions["client3"], abs_tol=0.05)

    def test_weighted_sampling_inconsistent_clients(self, clients: Set[str]):
        client_to_weight = {
            "client1": 0,  # zero-valued weight, so should not be sampled
            "client2": 1,
            "client4": 2,
        }
        sampler = WeightedClientSampler(
            n_samples=2, client_to_weight=client_to_weight
        )
        with pytest.raises(ValueError):
            sampler.init_clients(clients)

    @pytest.mark.parametrize("framework", ["torch"])
    def test_criterion_sampling(
        self,
        clients: Set[str],
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
        monkeypatch,
    ):
        """
        Test gradient norm criterion client sampling

        Notes: uses the train_replies fixture with one arbitrary fixed
        framework: torch.
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
            missing_scores_policy="priority",
        )
        sampler.init_clients(clients)
        # update the scores using the fake gradient norms
        sampler.update(client_to_reply, global_model)
        sampled_clients = sampler.sample()
        assert len(sampled_clients) == 2
        # client 2 and 3 have the highest scores (2 and 3)
        # so they must be chosen
        assert sampled_clients == {"client2", "client3"}

    @pytest.mark.parametrize("framework", ["torch"])
    def test_compo_crit_unif_sampling(
        self,
        clients: Set[str],
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
        monkeypatch,
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
            missing_scores_policy="priority",
        )
        unif_sampler = UniformClientSampler(n_samples=1)
        compo_sampler = CompositionClientSampler([crit_sampler, unif_sampler])

        compo_sampler.init_clients(clients)
        # update the scores using the fake gradient norms
        compo_sampler.update(client_to_reply, global_model)
        sampled_clients = compo_sampler.sample()

        # first, the criterion sampler should have selected client3 and then
        # the uniform sampler should have picked randomly one among the others
        assert len(sampled_clients) == 2
        assert "client3" in sampled_clients
