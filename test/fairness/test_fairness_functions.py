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

"""Unit tests for built-in group-fairness functions."""

import abc
from typing import Any, ClassVar, Dict, List, Tuple, Type, Union

import numpy as np
import pytest

from declearn.fairness.api import (
    FairnessFunction,
    instantiate_fairness_function,
)
from declearn.fairness.core import (
    AccuracyParityFunction,
    DemographicParityFunction,
    EqualityOfOpportunityFunction,
    EqualizedOddsFunction,
    list_fairness_functions,
)
from declearn.test_utils import assert_dict_equal


class FairnessFunctionTestSuite(metaclass=abc.ABCMeta):
    """Shared unit tests suite for 'FairnessFunction' subclasses."""

    cls_func: ClassVar[Type[FairnessFunction]]

    @property
    @abc.abstractmethod
    def expected_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        """Expected values for fairness constants with `counts`."""

    @property
    @abc.abstractmethod
    def expected_fairness(self) -> Dict[Tuple[Any, ...], float]:
        """Expected fairness values with `counts` and `accuracy`."""

    @property
    def counts(self) -> Dict[Tuple[Any, ...], int]:
        """Deterministic sensitive group counts for unit tests."""
        return {(0, 0): 40, (0, 1): 20, (1, 0): 30, (1, 1): 10}

    @property
    def accuracy(self) -> Dict[Tuple[Any, ...], float]:
        """Deterministic group-wise accuracy values for unit tests."""
        return {(0, 0): 0.7, (0, 1): 0.4, (1, 0): 0.8, (1, 1): 0.6}

    def setup_function(
        self,
    ) -> FairnessFunction:
        """Return a FairnessFunction used in shared unit tests."""
        return self.cls_func(counts=self.counts.copy())

    def test_properties(
        self,
    ) -> None:
        """Test that properties are properly defined."""
        func = self.setup_function()
        assert func.groups == list(self.counts)
        expected = func.compute_fairness_constants()
        assert (func.constants[0] == expected[0]).all()
        assert (func.constants[1] == expected[1]).all()

    def test_compute_fairness_constants(
        self,
    ) -> None:
        """Test that 'compute_fairness_constants' returns expected values."""
        func = self.setup_function()
        c_k0, c_kk = func.compute_fairness_constants()
        # Assert that types and shapes match generic expectations.
        n_groups = len(self.counts)
        assert isinstance(c_k0, np.ndarray)
        assert c_k0.shape in ((1,), (n_groups,))
        assert isinstance(c_kk, np.ndarray)
        assert c_kk.shape == (n_groups, n_groups)
        # Assert that values match expectations.
        expected = self.expected_constants
        assert np.allclose(c_k0, expected[0])
        assert np.allclose(c_kk, expected[1])

    def test_compute_from_group_accuracy(
        self,
    ) -> None:
        """Test that fairness computations work properly."""
        func = self.setup_function()
        fairness = func.compute_from_group_accuracy(self.accuracy)
        # Assert that types and keys match generic expectations.
        assert isinstance(fairness, dict)
        assert list(fairness) == func.groups
        assert all(isinstance(x, float) for x in fairness.values())
        # Assert that values match expectations.
        expected = self.expected_fairness
        deltas = [abs(fairness[k] - expected[k]) for k in func.groups]
        assert all(x < 1e-10 for x in deltas), deltas

    def test_compute_from_group_accuracy_error(
        self,
    ) -> None:
        """Test that fairness computations fail on improper inputs."""
        func = self.setup_function()
        accuracy = self.accuracy
        red_accr = {key: accuracy[key] for key in list(accuracy)[:2]}
        with pytest.raises(KeyError):
            func.compute_from_group_accuracy(red_accr)

    def test_compute_from_federated_group_accuracy(
        self,
    ) -> None:
        """Test that pseudo-federated fairness computations work properly."""
        func = self.setup_function()
        counts = self.counts
        accuracy = {
            key: val * counts[key] for key, val in self.accuracy.items()
        }
        fairness = func.compute_from_federated_group_accuracy(accuracy)
        expected = self.expected_fairness
        deltas = [abs(fairness[k] - expected[k]) for k in func.groups]
        assert all(x < 1e-10 for x in deltas)

    def test_get_specs(
        self,
    ) -> None:
        """Test that 'get_specs' works properly."""
        func = self.setup_function()
        specs = func.get_specs()
        assert isinstance(specs, dict)
        assert specs["f_type"] == func.f_type
        assert specs["counts"] == self.counts

    def test_instantiation_from_specs(
        self,
    ) -> None:
        """Test that instantiation from specs works properly."""
        func = self.setup_function()
        fbis = instantiate_fairness_function(**func.get_specs())
        assert isinstance(fbis, func.__class__)
        assert_dict_equal(fbis.get_specs(), func.get_specs())


class TestAccuracyParityFunction(FairnessFunctionTestSuite):
    """Unit tests for 'AccuracyParityFunction'."""

    cls_func = AccuracyParityFunction

    @property
    def expected_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        c_k0 = np.array(0.0)
        # fmt: off
        c_kk = [  # (n_k' / n) - 1{s == s'}*(n_k' / n_s)
            [0.4 - 4/7, 0.2 - 0.0, 0.3 - 3/7, 0.1 - 0.0],
            [0.4 - 0.0, 0.2 - 2/3, 0.3 - 0.0, 0.1 - 1/3],
            [0.4 - 4/7, 0.2 - 0.0, 0.3 - 3/7, 0.1 - 0.0],
            [0.4 - 0.0, 0.2 - 2/3, 0.3 - 0.0, 0.1 - 1/3],
        ]
        # fmt: on
        return c_k0, np.array(c_kk)

    @property
    def expected_fairness(self) -> Dict[Tuple[Any, ...], float]:
        c_kk = self.expected_constants[1]
        accuracy = self.accuracy
        acc = [accuracy[k] for k in ((0, 0), (0, 1), (1, 0), (1, 1))]
        f_s0 = -sum(c * a for c, a in zip(c_kk[0], acc, strict=False))
        f_s1 = -sum(c * a for c, a in zip(c_kk[1], acc, strict=False))
        return {(0, 0): f_s0, (0, 1): f_s1, (1, 0): f_s0, (1, 1): f_s1}


class TestDemographicParityFunction(FairnessFunctionTestSuite):
    """Unit tests for 'DemographicParityFunction'."""

    cls_func = DemographicParityFunction

    @property
    def expected_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        # (n_k / n_s) - (n_y / n)
        c_k0 = [
            # fmt: off
            4 / 7 - 0.6,
            2 / 3 - 0.6,
            3 / 7 - 0.4,
            1 / 3 - 0.4,
        ]
        # diagonal: (n_k / n) - (n_k / n_s)
        # reverse-diagonal: -n_k' / n
        # c_(y,s)^(y,s'): n_k' / n
        # c_(y,s)^(y',s): (n_k' / n_s) - (n_k' / n)
        c_kk = [
            # fmt: off
            [0.4 - 4 / 7, 0.2 - 0.0, 3 / 7 - 0.3, 0.0 - 0.1],
            [0.4 - 0.0, 0.2 - 2 / 3, 0.0 - 0.3, 1 / 3 - 0.1],
            [4 / 7 - 0.4, 0.0 - 0.2, 0.3 - 3 / 7, 0.1 - 0.0],
            [0.0 - 0.4, 2 / 3 - 0.2, 0.3 - 0.0, 0.1 - 1 / 3],
        ]
        return np.array(c_k0), np.array(c_kk)

    @property
    def expected_fairness(self) -> Dict[Tuple[Any, ...], float]:
        c_k0, c_kk = self.expected_constants
        accuracy = self.accuracy
        acc = [accuracy[k] for k in ((0, 0), (0, 1), (1, 0), (1, 1))]
        f_00 = (
            c_k0[0]
            + c_kk[0].sum()
            - sum(c * a for c, a in zip(c_kk[0], acc, strict=False))
        )
        f_01 = (
            c_k0[1]
            + c_kk[1].sum()
            - sum(c * a for c, a in zip(c_kk[1], acc, strict=False))
        )
        return {(0, 0): f_00, (0, 1): f_01, (1, 0): -f_00, (1, 1): -f_01}

    def test_error_nonbinary_attribute(self) -> None:
        """Test that a ValueError is raised on non-binary target labels."""
        with pytest.raises(ValueError):
            DemographicParityFunction(
                counts={(y, s): 1 for y in (0, 1, 2) for s in (0, 1)}
            )


class TestEqualizedOddsFunction(FairnessFunctionTestSuite):
    """Unit tests for 'EqualizedOddsFunction'."""

    cls_func = EqualizedOddsFunction

    @property
    def expected_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        c_k0 = np.array(0.0)
        # diagonal: (n_k / n_y) - 1
        # otherwise: 1{y == y'} * (n_k' / n_y)
        c_kk = [
            # fmt: off
            [4 / 6 - 1.0, 2 / 6 - 0.0, 0.0 - 0.0, 0.0 - 0.0],
            [4 / 6 - 0.0, 2 / 6 - 1.0, 0.0 - 0.0, 0.0 - 0.0],
            [0.0 - 0.0, 0.0 - 0.0, 3 / 4 - 1.0, 1 / 4 - 0.0],
            [0.0 - 0.0, 0.0 - 0.0, 3 / 4 - 0.0, 1 / 4 - 1.0],
        ]
        return c_k0, np.array(c_kk)

    @property
    def expected_fairness(self) -> Dict[Tuple[Any, ...], float]:
        c_kk = self.expected_constants[1]
        accuracy = self.accuracy
        groups = ((0, 0), (0, 1), (1, 0), (1, 1))
        acc = [accuracy[k] for k in groups]
        return {
            group: -sum(c * a for c, a in zip(c_kk[i], acc, strict=False))
            for i, group in enumerate(groups)
        }


@pytest.mark.parametrize("target", [0, 1, [0, 1]])
class TestEqualityOfOpportunity(TestEqualizedOddsFunction):
    """ABC for 'EqualityOfOpportunityFunction' unit tests."""

    cls_func = EqualityOfOpportunityFunction

    target: Union[int, List[int]]  # set via fixture

    @pytest.fixture(autouse=True)
    def init_attrs(self, target: Union[int, List[int]]) -> None:
        """Set up the desired 'target' parametrizing the fairness function."""
        self.target = target

    def setup_function(self) -> EqualityOfOpportunityFunction:
        return EqualityOfOpportunityFunction(
            counts=self.counts.copy(),
            target=self.target,
        )

    @property
    def expected_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        c_k0, c_kk = super().expected_constants
        if self.target == 0:
            c_kk[:, 2:] = 0.0
        elif self.target == 1:
            c_kk[:, :2] = 0.0
        return c_k0, c_kk

    def test_error_wrong_target_value(self) -> None:
        """Test that a ValueError is raised if 'target' is misspecified."""
        with pytest.raises(ValueError):
            EqualityOfOpportunityFunction(counts=self.counts, target=2)

    def test_error_wrong_target_type(self) -> None:
        """Test that a TypeError is raised if 'target' has unproper type."""
        with pytest.raises(TypeError):
            EqualityOfOpportunityFunction(
                counts=self.counts,
                target="wrong-type",  # type: ignore
            )


def test_list_fairness_functions() -> None:
    """Test 'declearn.fairness.core.list_fairness_functions'."""
    mapping = list_fairness_functions()
    assert isinstance(mapping, dict)
    assert all(
        isinstance(key, str) and issubclass(val, FairnessFunction)
        for key, val in mapping.items()
    )
    for cls in (
        AccuracyParityFunction,
        DemographicParityFunction,
        EqualityOfOpportunityFunction,
        EqualizedOddsFunction,
    ):
        assert mapping.get(cls.f_type) is cls  # type: ignore
