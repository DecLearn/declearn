# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for `declearn.main.privacy.DPTrainingManager`."""

import os
from typing import Any, Optional

import pytest

try:
    from opacus.accountants import RDPAccountant  # type: ignore
    from opacus.accountants.utils import get_noise_multiplier  # type: ignore
except ModuleNotFoundError:
    pytest.skip("Opacus is unavailable", allow_module_level=True)

from declearn import messaging
from declearn.dataset import DataSpecs
from declearn.optimizer.modules import GaussianNoiseModule
from declearn.test_utils import make_importable
from declearn.training.dp import DPTrainingManager

with make_importable(os.path.dirname(__file__)):
    from test_train_manager import BATCHES, build_manager, build_train_request


BATCHES["poisson"] = True  # mock the use of Poisson sampling out of coherence


def build_dp_manager(n_batch: int) -> Any:  # DPTrainingManager with Mock attrs
    """Return a DPTrainingManager instance with Mock attributes."""
    base = build_manager(n_batch)
    # Enable accessing the (emulated) number of samples in the mock dataset.
    base.train_data.get_data_specs.return_value = DataSpecs(
        n_samples=n_batch * BATCHES["batch_size"],
        features_shape=(8,),  # unused
    )
    # Enable accessing the `modules` attribute of the mock Optimizer.
    base.optim.modules = []
    # Replace the base TrainingManager with a DPTrainingManager.
    return DPTrainingManager(
        model=base.model,
        optim=base.optim,
        aggrg=base.aggrg,
        train_data=base.train_data,
        valid_data=base.valid_data,
    )


def build_privacy_request(
    rounds: int = 1,
    n_epoch: Optional[int] = None,
    n_steps: Optional[int] = None,
) -> messaging.PrivacyRequest:
    """Return a PrivacyRequest with specified number of rounds and steps."""
    return messaging.PrivacyRequest(
        budget=(2.0, 1e-05),
        sclip_norm=2.0,
        accountant="rdp",
        use_csprng=False,
        seed=0,
        rounds=rounds,
        n_epoch=n_epoch,
        n_steps=n_steps,
        batches=BATCHES,
    )


@pytest.mark.filterwarnings(
    # Silence opacus warnings about the alphas used to convert
    # between (epsilon, delta) and Renyi differential privacy.
    "ignore: Optimal order is the largest alpha."
)
class TestDPTrainingManager:
    """Unit tests for `declearn.main.privacy.DPTrainingManager`."""

    def test_nonprivate(self):
        """Test that by a vanilla DPTrainingManager acts as its parent does."""
        # Test that at instantiation a DPTrainingManager does not implement DP.
        manager = build_dp_manager(n_batch=100)
        assert manager.accountant is None
        assert manager.sclip_norm is None
        assert not manager.optim.modules  # empty list
        assert manager.get_noise_multiplier() is None
        with pytest.raises(RuntimeError):
            manager.get_privacy_spent()
        # Test that the training routine works (implementing simple SGD).
        reply = manager.training_round(build_train_request(n_steps=20))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert reply.n_steps == 20

    def test_make_private(self):
        """Test that the `DPTrainingManager.make_private` method works."""
        # Create a DPTrainingManager and call its make_private method.
        manager = build_dp_manager(n_batch=100)
        request = build_privacy_request(rounds=1, n_epoch=1)
        manager.make_private(request)
        # Check that expected attribute changes have occurred.
        assert isinstance(manager.accountant, RDPAccountant)
        assert manager.sclip_norm == request.sclip_norm
        assert isinstance(manager.optim.modules[0], GaussianNoiseModule)
        # Compute the expected noise multiplier and verify it is correct.
        noise = get_noise_multiplier(
            target_epsilon=request.budget[0],
            target_delta=request.budget[1],
            sample_rate=0.01,
            steps=100,
        )
        assert manager.get_noise_multiplier() == noise
        assert manager.optim.modules[0].std == noise * request.sclip_norm
        # Check that initially not budget has been spent (but delta is set).
        assert manager.get_privacy_spent() == (0, request.budget[1])

    def test_dp_budget_constraint_1(self):
        """Test that the DP budget overspending is properly prevented.

        Case 1: saturating the budget with a full round (blocking the second).
        """
        # Create a DPTrainingManager and call its make_private method.
        manager = build_dp_manager(n_batch=10)
        request = build_privacy_request(rounds=1, n_epoch=1)
        manager.make_private(request)
        # Check that the first round runs properly, spending the budget.
        reply = manager.training_round(build_train_request(n_epoch=1))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert reply.n_steps == 10
        budget_spent = manager.get_privacy_spent()
        assert budget_spent[0] <= request.budget[0]
        assert budget_spent[1] == request.budget[1]
        # Check that no further step is authorized, as budget is saturated.
        reply = manager.training_round(build_train_request(n_epoch=1))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert reply.n_steps == 0
        assert manager.get_privacy_spent() == budget_spent

    def test_dp_budget_constraint_2(self):
        """Test that the DP budget overspending is properly prevented.

        Case 2: saturating the budget with half a round (interrupting it).
        """
        # Create a DPTrainingManager and call its make_private method.
        manager = build_dp_manager(n_batch=100)
        request = build_privacy_request(rounds=1, n_steps=50)
        manager.make_private(request)
        # Check that the round is interrupted once the budget was spent.
        reply = manager.training_round(build_train_request(n_steps=100))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert 50 <= reply.n_steps < 100
        budget_spent = manager.get_privacy_spent()
        assert budget_spent[0] <= request.budget[0]
        assert budget_spent[1] == request.budget[1]
