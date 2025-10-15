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

"""Unit tests for the 'Aggregator' subclasses."""

from typing import Dict, Type

import pytest

from declearn.aggregator import Aggregator, ModelUpdates, list_aggregators
from declearn.model.api import Vector
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    assert_dict_equal,
    assert_json_serializable_dict,
    list_available_frameworks,
)
from declearn.utils import set_device_policy

AGGREGATOR_CLASSES = list_aggregators()
VECTOR_FRAMEWORKS = list_available_frameworks()


@pytest.fixture(name="updates")
def updates_fixture(
    framework: FrameworkType,
    n_clients: int = 3,
) -> Dict[str, Vector]:
    """Fixture providing with deterministic sets of updates Vector."""
    set_device_policy(gpu=False)
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
    def test_prepare_for_sharing(
        self,
        agg_cls: Type[Aggregator],
        updates: Dict[str, Vector],
    ) -> None:
        """Test that 'prepare_for_sharing' returns a proper-type instance.

        Also test that the output:
            - is JSON-serializable in dict representation
            - can properly be recovered from its dict representation
        """
        aggregator = agg_cls()
        shared_upd = aggregator.prepare_for_sharing(updates["0"], n_steps=10)
        assert issubclass(aggregator.updates_cls, ModelUpdates)
        assert isinstance(shared_upd, aggregator.updates_cls)
        assert_json_serializable_dict(shared_upd.to_dict())
        assert shared_upd == aggregator.updates_cls(**shared_upd.to_dict())

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_finalize_updates(
        self,
        agg_cls: Type[Aggregator],
        updates: Dict[str, Vector],
    ) -> None:
        """Test that 'finalize_updates' produces expected-type outputs."""
        # Test that sum-aggregation and finalization of partial updates work.
        aggregator = agg_cls()
        aggregated = sum(
            aggregator.prepare_for_sharing(vec, n_steps=10)
            for vec in updates.values()
        )
        assert isinstance(aggregated, aggregator.updates_cls)
        output = aggregator.finalize_updates(aggregated)
        # Test that the output Vector matches expected specifications.
        ref_vec = list(updates.values())[0]
        assert isinstance(output, type(ref_vec))
        assert output.shapes() == ref_vec.shapes()
        assert output.dtypes() == ref_vec.dtypes()

    def test_get_config(self, agg_cls: Type[Aggregator]) -> None:
        """Test that the 'get_config' method works properly."""
        aggregator = agg_cls()
        agg_config = aggregator.get_config()
        assert_json_serializable_dict(agg_config)

    def test_from_config(self, agg_cls: Type[Aggregator]) -> None:
        """Test that the 'from_config' method works properly."""
        aggregator = agg_cls()
        agg_config = aggregator.get_config()
        replica = agg_cls.from_config(agg_config)
        assert isinstance(replica, agg_cls)
        assert_dict_equal(agg_config, replica.get_config())

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_secagg_compatibility(
        self,
        agg_cls: Type[Aggregator],
        updates: Dict[str, Vector],
    ) -> None:
        """Test that the aggregator support secure aggregation."""
        # Setup an Aggregator and produce two sets of ModelUpdates.
        aggregator = agg_cls()
        updates_a = aggregator.prepare_for_sharing(updates["0"], n_steps=10)
        updates_b = aggregator.prepare_for_sharing(updates["1"], n_steps=20)
        # Extract fields as if for SecAgg and verify type correctness.
        sec_a, clr_a = updates_a.prepare_for_secagg()
        sec_b, clr_b = updates_b.prepare_for_secagg()
        assert isinstance(sec_a, dict) and isinstance(sec_b, dict)
        assert isinstance(clr_a, dict) or clr_a is None
        assert isinstance(clr_b, type(clr_b))
        # Conduct their aggregation as defined for SecAgg (but in cleartext).
        secagg = {key: val + sec_b[key] for key, val in sec_a.items()}
        if clr_a is None:
            clrtxt = {}
        else:
            clrtxt = {
                key: getattr(
                    updates_a, f"aggregate_{key}", updates_a.default_aggregate
                )(val, clr_b[key])
                for key, val in clr_a.items()
            }
        output = type(updates_a)(**secagg, **clrtxt)
        # Verify that results match expectations.
        result = aggregator.finalize_updates(output)
        expect = aggregator.finalize_updates(updates_a + updates_b)
        assert result == expect
