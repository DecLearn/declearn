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

import os
from typing import Type

import pytest

from declearn.optimizer import list_optim_regularizers
from declearn.optimizer.regularizers import Regularizer
from declearn.test_utils import make_importable

# relative imports from `optim_testing.py`
with make_importable(os.path.dirname(__file__)):
    from optim_testing import PluginTestBase


REGULARIZER_SUBCLASSES = list_optim_regularizers()


@pytest.mark.parametrize(
    "cls", REGULARIZER_SUBCLASSES.values(), ids=REGULARIZER_SUBCLASSES.keys()
)
class TestRegularizer(PluginTestBase):
    """Unit tests for declearn.optimizer.regularizer.Regularizer subclasses."""

    def test_on_round_start(self, cls: Type[Regularizer]) -> None:
        """Test that a Regularizer's on_round_start method can be called."""
        regularizer = cls()
        assert regularizer.on_round_start() is None  # type: ignore
