# coding: utf-8

"""Unit tests for Regularizer subclasses.

This script implements unit tests that are automatically run
for each and every Regularizer subclass type-registered under
the "Regularizer" group name.

These tests verify that API-defined methods can be run and
have the expected behaviour from the API's point of view -
in other words, algorithmic correctness is *not* tested as
it requires plugin-specific testing code.

However, these tests assert that the plugins' `run` method
effectively support gradients from a variety of frameworks
(NumPy, TensorFlow, PyTorch) and that the outputs have the
same values (up to reasonable numerical precision) for all
of these.
"""

import sys
from typing import Type

import pytest

from declearn.optimizer.regularizers import Regularizer
from declearn.utils._register import REGISTRIES


# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from optim_testing import PluginTestBase

# unproper but efficient way to list plugins; pylint: disable=protected-access
REGULARIZER_SUBCLASSES = REGISTRIES["Regularizer"]._reg
# pylint: enable=protected-access


@pytest.mark.parametrize(
    "cls", REGULARIZER_SUBCLASSES.values(), ids=REGULARIZER_SUBCLASSES.keys()
)
class TestRegularizer(PluginTestBase):
    """Unit tests for declearn.optimizer.regularizer.Regularizer subclasses."""

    def test_on_round_start(self, cls: Type[Regularizer]) -> None:
        """Test that a Regularizer's on_round_start method can be called."""
        regularizer = cls()
        assert regularizer.on_round_start() is None  # type: ignore
