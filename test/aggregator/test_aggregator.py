# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for the 'Aggregator' subclasses."""

import typing
from typing import Dict, Type

import pytest

from declearn.aggregator import Aggregator, list_aggregators
from declearn.model.api import Vector
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    assert_dict_equal,
    assert_json_serializable_dict,
)


AGGREGATOR_CLASSES = list_aggregators()
VECTOR_FRAMEWORKS = typing.get_args(FrameworkType)


@pytest.fixture(name="updates")
def updates_fixture(
    framework: FrameworkType,
    n_clients: int = 3,
) -> Dict[str, Vector]:
    """Fixture providing with deterministic sets of updates Vector."""
    return {
        str(idx): GradientsTestCase(framework, seed=idx).mock_gradient
        for idx in range(n_clients)
    }


@pytest.mark.parametrize(
    "agg_cls", AGGREGATOR_CLASSES.values(), ids=AGGREGATOR_CLASSES.keys()
)
class TestAggregator:
    """Shared unit tests suite for 'Aggregator' subclasses."""

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_aggregate(
        self,
        agg_cls: Type[Aggregator],
        updates: Dict[str, Vector],
    ) -> None:
        """Test that the 'aggregate' method works properly."""
        agg = agg_cls()
        n_steps = {key: 10 for key in updates}
        outputs = agg.aggregate(updates, n_steps)
        ref_vec = list(updates.values())[0]
        assert isinstance(outputs, type(ref_vec))
        assert outputs.shapes() == ref_vec.shapes()
        assert outputs.dtypes() == ref_vec.dtypes()

    def test_aggregate_empty(
        self,
        agg_cls: Type[Aggregator],
    ) -> None:
        """Test that 'aggregate' raises the expected error on empty inputs."""
        agg = agg_cls()
        with pytest.raises(TypeError):
            agg.aggregate(updates={}, n_steps={})

    def test_get_config(self, agg_cls: Type[Aggregator]) -> None:
        """Test that the 'get_config' method works properly."""
        agg = agg_cls()
        cfg = agg.get_config()
        assert_json_serializable_dict(cfg)

    def test_from_config(self, agg_cls: Type[Aggregator]) -> None:
        """Test that the 'from_config' method works properly."""
        agg = agg_cls()
        cfg = agg.get_config()
        bis = agg_cls.from_config(cfg)
        assert isinstance(bis, agg_cls)
        assert_dict_equal(cfg, bis.get_config())
