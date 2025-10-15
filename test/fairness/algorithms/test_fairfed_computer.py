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

"""Unit tests for FairFed-specific fairness value computer."""

import warnings
from typing import Any, List, Tuple

import pytest

from declearn.fairness.fairfed import FairfedValueComputer

GROUPS_BINARY: List[Tuple[Any, ...]] = [
    (target, s_attr) for target in (0, 1) for s_attr in (0, 1)
]
GROUPS_EXTEND: List[Tuple[Any, ...]] = [
    (tgt, s_a, s_b) for tgt in (0, 1, 2) for s_a in (0, 1) for s_b in (1, 2)
]
F_TYPES = [
    "accuracy_parity",
    "demographic_parity",
    "equality_of_opportunity",
    "equalized_odds",
]


class TestFairfedValueComputer:
    """Unit tests for 'declearn.fairness.fairfed.FairfedValueComputer'."""

    @pytest.mark.parametrize("target", [1, 0], ids=["target1", "target0"])
    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_identify_key_groups_binary(
        self,
        f_type: str,
        target: int,
    ) -> None:
        """Test 'identify_key_groups' with binary target and attribute."""
        computer = FairfedValueComputer(f_type, strict=True, target=target)
        if f_type == "accuracy_parity":
            with pytest.warns(RuntimeWarning):
                key_groups = computer.identify_key_groups(GROUPS_BINARY.copy())
        else:
            key_groups = computer.identify_key_groups(GROUPS_BINARY.copy())
        assert key_groups == ((target, 0), (target, 1))

    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_identify_key_groups_extended_exception(
        self,
        f_type: str,
    ) -> None:
        """Test 'identify_key_groups' exception raising with extended groups.

        'Extended' groups arise from a non-binary label intersected with
        two distinct binary sensitive groups.
        """
        computer = FairfedValueComputer(f_type, strict=True, target=1)
        with pytest.raises(RuntimeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                computer.identify_key_groups(GROUPS_EXTEND.copy())

    @pytest.mark.parametrize("f_type", F_TYPES)
    def test_identify_key_groups_hybrid_exception(
        self,
        f_type: str,
    ) -> None:
        """Test 'identify_key_groups' exception raising with 'hybrid' groups.

        'Hybrid' groups are groups that seemingly arise from a categorical
        target that does not cross all sensitive attribute modalities.
        """
        computer = FairfedValueComputer(f_type, strict=True, target=1)
        with pytest.raises(KeyError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                computer.identify_key_groups([(0, 0), (0, 1), (1, 0), (2, 1)])

    @pytest.mark.parametrize("binary", [True, False], ids=["binary", "extend"])
    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "free"])
    @pytest.mark.parametrize("f_type", F_TYPES[1:])  # avoid warning on AccPar
    def test_initialize(
        self,
        f_type: str,
        strict: bool,
        binary: bool,
    ) -> None:
        """Test that 'initialize' raises an exception in expected cases."""
        computer = FairfedValueComputer(f_type, strict=strict, target=1)
        groups = (GROUPS_BINARY if binary else GROUPS_EXTEND).copy()
        if strict and not binary:
            with pytest.raises(RuntimeError):
                computer.initialize(groups)
        else:
            computer.initialize(groups)

    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "free"])
    def test_compute_synthetic_fairness_value_binary(
        self,
        strict: bool,
    ) -> None:
        """Test 'compute_synthetic_fairness_value' with 4 groups.

        This test only applies to both strict and non-strict modes.
        """
        # Compute a synthetic value using arbitrary inputs.
        fairness = {
            group: float(idx) for idx, group in enumerate(GROUPS_BINARY)
        }
        computer = FairfedValueComputer(
            f_type="demographic_parity",
            strict=strict,
            target=1,
        )
        computer.initialize(list(fairness))
        value = computer.compute_synthetic_fairness_value(fairness)
        # Verify that the ouput value matches expectations.
        if strict:
            expected = fairness[(1, 0)] - fairness[(1, 1)]
        else:
            expected = sum(fairness.values()) / len(fairness)
        assert value == expected

    def test_compute_synthetic_fairness_value_extended(
        self,
    ) -> None:
        """Test 'compute_synthetic_fairness_value' with many groups.

        This test only applies to the non-strict mode.
        """
        # Compute a synthetic value using arbitrary inputs.
        fairness = {
            group: float(idx) for idx, group in enumerate(GROUPS_EXTEND)
        }
        computer = FairfedValueComputer(
            f_type="demographic_parity",
            strict=False,
        )
        computer.initialize(list(fairness))
        value = computer.compute_synthetic_fairness_value(fairness)
        # Verify that the ouput value matches expectations.
        expected = sum(fairness.values()) / len(fairness)
        assert value == expected
