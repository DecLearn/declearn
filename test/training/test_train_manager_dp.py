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

"""Unit tests for `declearn.main.privacy.DPTrainingManager`."""

import os
from typing import Any, Optional

import pytest

try:
    from opacus.accountants import (  # type: ignore
        RDPAccountant,
        create_accountant,
    )
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
    accountant: str = "rdp",
) -> messaging.PrivacyRequest:
    """Return a PrivacyRequest with specified number of rounds and steps."""
    return messaging.PrivacyRequest(
        budget=(2.0, 1e-05),
        sclip_norm=2.0,
        accountant=accountant,
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

    def test_make_private_with_invalid_accountant(self):
        """Test that `make_private` rejects an unsupported accountant.

        The allowlist guards `_compute_max_steps_for_round`, whose probe-
        history logic only holds for opacus's "rdp"/"gdp"/"prv" accountants,
        so any other value must raise rather than be silently accepted.
        """
        manager = build_dp_manager(n_batch=100)
        request = build_privacy_request(
            rounds=1, n_epoch=1, accountant="unsupported"
        )
        with pytest.raises(ValueError):
            manager.make_private(request)

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

    def test_precompute_preserves_accountant_history(self):
        """Test that precomputing max-steps does not mutate the accountant.

        The precompute binary-searches the largest allowed step count by
        temporarily overwriting `accountant.history` with synthetic tuples.
        It must restore the real history exactly, so that the accountant
        only ever reflects steps that actually occurred.
        """
        manager = build_dp_manager(n_batch=100)
        request = build_privacy_request(rounds=1, n_steps=50)
        manager.make_private(request)
        noise = manager.get_noise_multiplier()
        srate = BATCHES["batch_size"] / (100 * BATCHES["batch_size"])
        # Case A: empty history (round start, before any step).
        snapshot = list(manager.accountant.history)
        max_steps = manager._compute_max_steps_for_round(noise, srate)
        assert isinstance(max_steps, int) and max_steps > 0
        assert manager.accountant.history == snapshot
        # Case B: non-empty history (some steps already accounted for).
        for _ in range(5):
            manager.accountant.step(noise_multiplier=noise, sample_rate=srate)
        snapshot = list(manager.accountant.history)
        manager._compute_max_steps_for_round(noise, srate)
        assert manager.accountant.history == snapshot

    def test_precompute_matches_canonical_interruption(self):
        """Test precompute interrupts at the same step as a per-step check.

        The whole point of the precompute is byte-identical behavior to the
        canonical per-step `get_epsilon` check: the precomputed bound must
        equal the number of steps a naive per-step accountant would accept
        before the budget is exceeded. This pins that equivalence as a
        regression guard.
        """
        manager = build_dp_manager(n_batch=100)
        request = build_privacy_request(rounds=1, n_steps=50)
        manager.make_private(request)
        noise = manager.get_noise_multiplier()
        srate = BATCHES["batch_size"] / (100 * BATCHES["batch_size"])
        budget_eps, budget_delta = request.budget
        # Reference: count how many steps a per-step check would accept.
        reference = RDPAccountant()
        accepted = 0
        while accepted < 1000:
            reference.step(noise_multiplier=noise, sample_rate=srate)
            if reference.get_epsilon(delta=budget_delta) > budget_eps:
                break  # this step would overspend; canonical rejects it
            accepted += 1
        # The precomputed bound must equal that step count exactly.
        assert manager._compute_max_steps_for_round(noise, srate) == accepted
        # And running an over-long round must stop after exactly that many.
        reply = manager.training_round(
            build_train_request(n_steps=accepted + 50)
        )
        assert reply.n_steps == accepted

@pytest.mark.parametrize("accountant", ["rdp", "gdp", "prv"])
@pytest.mark.parametrize("prev", [0, 30])
def test_precompute_matches_canonical_with_prior_history(self, accountant, prev):
    """Precompute matches a per-step `get_epsilon` check across accountants.

    The bound must equal the number of *additional* steps a per-step check
    accepts, continuing from already-spent budget. `prev` covers both the
    empty-history case (0) and a real prior round (30); the former proves
    the merge reduces cleanly to the no-history bound. The bound is also
    checked tight: exactly `accepted` steps fit, `accepted + 1` overspends.

    Regression guard: the probe merges into the trailing history tuple
    rather than appending a new one. GaussianAccountant ("gdp") reads only
    the last tuple in `get_epsilon`, so appending would ignore prior-round
    budget and over-authorize; merging stays correct and is equivalent for
    the additively-composing rdp/prv.
    """
    manager = build_dp_manager(n_batch=100)
    request = build_privacy_request(rounds=1, n_steps=50)
    manager.make_private(request)
    # Swap in the accountant under test (make_private defaults to rdp).
    manager.accountant = create_accountant(accountant)
    noise = manager.get_noise_multiplier()
    srate = BATCHES["batch_size"] / (100 * BATCHES["batch_size"])
    budget_eps, budget_delta = request.budget
    # Spend some budget first (emulate a previous round of `prev` steps).
    for _ in range(prev):
        manager.accountant.step(noise_multiplier=noise, sample_rate=srate)
    # Reference: additional steps a per-step check would still accept,
    # continuing from the exact same already-spent state.
    reference = create_accountant(accountant)
    for _ in range(prev):
        reference.step(noise_multiplier=noise, sample_rate=srate)
    accepted = 0
    while accepted < 1000:
        reference.step(noise_multiplier=noise, sample_rate=srate)
        if reference.get_epsilon(delta=budget_delta) > budget_eps:
            reference.history.pop()  # undo the rejected probe step
            break
        accepted += 1
    # The precomputed additional-step bound must match exactly, and the
    # real history must be left untouched by the search.
    snapshot = list(manager.accountant.history)
    assert manager._compute_max_steps_for_round(noise, srate) == accepted
    assert manager.accountant.history == snapshot

    # Boundary exactness: the bound must be tight on both sides. Rebuild
    # from scratch each time so the probe shares no state or history-surgery
    # with the code under test.
    def overspends(extra_steps):
        probe = create_accountant(accountant)
        for _ in range(prev + extra_steps):
            probe.step(noise_multiplier=noise, sample_rate=srate)
        return probe.get_epsilon(delta=budget_delta) > budget_eps

    assert not overspends(accepted), "the bound itself must stay within budget"
    assert overspends(accepted + 1), "one step past the bound must overspend"