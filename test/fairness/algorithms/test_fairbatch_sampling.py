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

"""Unit tests for FairBatch sampling probability controllers."""

import pytest

from declearn.fairness.api import FairnessFunction
from declearn.fairness.fairbatch import (
    FairbatchSamplingController,
    setup_fairbatch_controller,
    setup_fedfb_controller,
)

ALPHA = 0.05
COUNTS = {(0, 0): 30, (0, 1): 15, (1, 0): 35, (1, 1): 20}
F_TYPES = [
    "demographic_parity",
    "equality_of_opportunity",
    "equalized_odds",
]


@pytest.fixture(name="controller")
def controller_fixture(
    f_type: str,
    fedfb: bool,
) -> FairbatchSamplingController:
    """Fixture providing with a given 'FairbatchSamplingController'."""
    if fedfb:
        return setup_fedfb_controller(f_type, counts=COUNTS, alpha=ALPHA)
    return setup_fairbatch_controller(f_type, counts=COUNTS, alpha=ALPHA)


@pytest.mark.parametrize("fedfb", [False, True], ids=["FedFairBatch", "FedFB"])
@pytest.mark.parametrize("f_type", F_TYPES)
class TestFairbatchSamplingController:
    """Shared unit tests for all 'FairbatchSamplingController' subclasses."""

    def test_init(
        self,
        controller: FairbatchSamplingController,
    ) -> None:
        """Test that the instantiated controller is coherent."""
        # Assert that groups and counts attributes have expected specs.
        assert isinstance(controller.groups, dict)
        assert isinstance(controller.counts, dict)
        assert all(isinstance(key, str) for key in controller.groups)
        assert set(controller.groups.values()) == set(controller.counts)
        assert controller.counts == COUNTS
        # Verify that other hyper-parameters were properly passed.
        assert controller.alpha == ALPHA
        assert isinstance(controller.f_func, FairnessFunction)
        assert controller.f_func.f_type == controller.f_type

    def test_compute_initial_states(
        self,
        controller: FairbatchSamplingController,
    ) -> None:
        """Test that 'compute_initial_states' has proper output types."""
        states = controller.compute_initial_states()
        assert all(
            isinstance(key, str) and isinstance(val, (int, float))
            for key, val in states.items()
        )
        assert controller.states == states  # initial states

    def test_get_sampling_probas(
        self,
        controller: FairbatchSamplingController,
    ) -> None:
        """Test that 'get_sampling_probas' outputs coherent values."""
        probas = controller.get_sampling_probas()
        # Assert that probabilities are a dict with expected keys.
        assert isinstance(probas, dict)
        assert set(probas.keys()) == set(controller.counts)
        # Assert that values are floats and sum to one (up to a small epsilon).
        assert all(isinstance(val, float) for val in probas.values())
        assert abs(1 - sum(probas.values())) < 0.001

    def test_update_from_losses(
        self,
        controller: FairbatchSamplingController,
    ) -> None:
        """Test that 'update_from_losses' alters states and output probas.

        This test does not verify that the maths match initial papers.
        """
        # Record initial states and sampling probabilities.
        initial_states = controller.states.copy()
        initial_probas = controller.get_sampling_probas()
        # Perform an update with arbitrary loss values.
        losses = {group: float(idx) for idx, group in enumerate(COUNTS)}
        controller.update_from_losses(losses)
        # Verify that states and probas have changed.
        assert controller.states != initial_states
        probas = controller.get_sampling_probas()
        assert probas.keys() == initial_probas.keys()
        assert probas != initial_probas
        # Verify that output probabilities sum to one (up to a small epsilon).
        assert abs(1 - sum(probas.values())) < 0.001

    def test_update_from_federated_losses(
        self,
        controller: FairbatchSamplingController,
    ) -> None:
        """Test that 'update_from_federated_losses' has expected outputs."""
        # Update from arbitrary countes-scaled losses and gather states.
        losses = {group: float(idx) for idx, group in enumerate(COUNTS)}
        controller.update_from_federated_losses(
            {key: val * controller.counts[key] for key, val in losses.items()}
        )
        states = controller.states.copy()
        # Reset states and use unscaled values via basic update method.
        controller.states = controller.compute_initial_states()
        controller.update_from_losses(losses)
        # Assert that resulting states are the same.
        assert controller.states == states


@pytest.mark.parametrize("fedfb", [False, True], ids=["FedFairBatch", "FedFB"])
@pytest.mark.parametrize("f_type", F_TYPES)
def test_setup_controller_parameters(
    f_type: str,
    fedfb: bool,
) -> None:
    """Test that controller setup properly passes input parameters."""
    function = setup_fedfb_controller if fedfb else setup_fairbatch_controller
    controller = function(f_type=f_type, counts=COUNTS.copy(), target=0)
    assert controller.f_type == f_type
    if f_type == "equality_of_opportunity":
        assert controller.f_func.get_specs()["target"] == [0]


@pytest.mark.parametrize("fedfb", [False, True], ids=["FedFairBatch", "FedFB"])
def test_setup_controller_invalid_ftype(
    fedfb: bool,
) -> None:
    """Test that controller setup raises a KeyError on invalid 'f_type'."""
    function = setup_fedfb_controller if fedfb else setup_fairbatch_controller
    with pytest.raises(KeyError):
        function(f_type="invalid_f_type", counts=COUNTS.copy())


@pytest.mark.parametrize("fedfb", [False, True], ids=["FedFairBatch", "FedFB"])
def test_setup_controller_invalid_groups(
    fedfb: bool,
) -> None:
    """Test that controller setup raises a ValueError on invalid groups."""
    function = setup_fedfb_controller if fedfb else setup_fairbatch_controller
    # Case when there are more than 4 groups.
    counts = COUNTS.copy()
    counts[(2, 0)] = counts[(2, 1)] = 5  # add a third target label value
    with pytest.raises(ValueError):
        function(f_type="demographic_parity", counts=counts)
    # Case when there are 4 ill-defined groups.
    counts = {(0, 0): 10, (1, 0): 10, (2, 0): 10, (2, 1): 10}
    with pytest.raises(ValueError):
        function(f_type="demographic_parity", counts=counts)


@pytest.mark.parametrize("fedfb", [False, True], ids=["FedFairBatch", "FedFB"])
def test_setup_controller_invalid_target(
    fedfb: bool,
) -> None:
    """Test that controller setup raises a ValueError on invalid target."""
    function = setup_fedfb_controller if fedfb else setup_fairbatch_controller
    with pytest.raises(ValueError):
        function(f_type="demographic_parity", counts=COUNTS.copy(), target=2)
