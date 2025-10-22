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

"""Unit tests for FairGrad weights computation controller."""

from unittest import mock

import numpy as np
import pytest

from declearn.fairness.api import FairnessFunction
from declearn.fairness.fairgrad import FairgradWeightsController

# pylint: disable=duplicate-code
COUNTS = {(0, 0): 30, (0, 1): 15, (1, 0): 35, (1, 1): 20}
F_TYPES = [
    "accuracy_parity",
    "demographic_parity",
    "equality_of_opportunity",
    "equalized_odds",
]
# pylint: enable=duplicate-code


class TestFairgradWeightsController:
    """Unit tests for 'FairgradWeightsController'.

    These tests cover both the formal behavior of methods
    and the correctness of the wrapped math operations.
    """

    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_init(
        self,
        f_type: str,
    ) -> None:
        """Test that instantiation hyper-parameters are properly passed."""
        eta = mock.create_autospec(float, instance=True)
        eps = mock.create_autospec(float, instance=True)
        controller = FairgradWeightsController(
            counts=COUNTS.copy(),
            f_type=f_type,
            eta=eta,
            eps=eps,
        )
        assert controller.eta is eta
        assert controller.eps is eps
        assert controller.total == sum(COUNTS.values())
        assert isinstance(controller.function, FairnessFunction)
        assert controller.function.f_type == f_type
        assert (controller.f_k == 0).all()

    def test_get_current_weights_initial(self) -> None:
        """Test that initial weights are properly computed / accessed."""
        controller = FairgradWeightsController(
            counts=COUNTS.copy(), f_type="accuracy_parity"
        )
        # Verify that initial weights match expectations (i.e. P(T_k)).
        weights = controller.get_current_weights(norm_nk=False)
        expectw = [val / controller.total for val in COUNTS.values()]
        assert weights == expectw
        # Verify that 'norm_nk' parameter has proper effect.
        weights = controller.get_current_weights(norm_nk=True)
        expectw = [1 / controller.total] * len(COUNTS)
        assert weights == expectw

    @pytest.mark.parametrize("exact", [True, False], ids=["exact", "epsilon"])
    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_update_weights_based_on_accuracy(
        self,
        f_type: str,
        exact: bool,
    ) -> None:
        """Test that weights update works properly."""
        # Setup a controller and update its weights using arbitrary values.
        controller = FairgradWeightsController(
            counts=COUNTS.copy(), f_type=f_type, eps=0.0 if exact else 0.01
        )
        accuracy = {group: 0.2 * idx for idx, group in enumerate(COUNTS)}
        controller.update_weights_based_on_accuracy(accuracy)
        # Verify that proper values were assigned as current fairness.
        f_k = controller.function.compute_from_federated_group_accuracy(
            accuracy
        )
        assert (controller.f_k == np.array(list(f_k.values()))).all()
        # Verify that expected weights are returned.
        c_kk = controller.function.constants[1]
        p_tk = controller.counts / controller.total
        if exact:
            w_tk = controller.eta * controller.f_k
        else:
            w_tk = np.abs(controller.f_k)
            w_tk = controller.eta * np.where(
                w_tk > controller.eps, w_tk - controller.eps, 0.0
            )
        expectw = p_tk + np.dot(w_tk, c_kk)
        weights = controller.get_current_weights(norm_nk=False)
        assert np.allclose(expectw, np.array(weights), atol=0.001)
        # Same check with 'norm_nk = True'.
        expectw /= controller.counts
        weights = controller.get_current_weights(norm_nk=True)
        assert np.allclose(expectw, np.array(weights), atol=0.001)

    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_get_current_fairness(
        self,
        f_type: str,
    ) -> None:
        """Test that access to current fairness values works properly."""
        controller = FairgradWeightsController(
            counts=COUNTS.copy(), f_type=f_type
        )
        accuracy = {group: 0.2 * idx for idx, group in enumerate(COUNTS)}
        controller.update_weights_based_on_accuracy(accuracy)
        fairness = controller.get_current_fairness()
        assert isinstance(fairness, dict)
        assert fairness == dict(
            zip(controller.function.groups, controller.f_k, strict=False)
        )
