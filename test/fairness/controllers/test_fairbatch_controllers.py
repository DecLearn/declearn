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

"""Unit tests for Fed-FairBatch controllers."""

import os
from unittest import mock

import pytest

from declearn.aggregator import Aggregator, SumAggregator
from declearn.fairness.fairbatch import (
    FairbatchControllerClient,
    FairbatchControllerServer,
    FairbatchDataset,
)
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import (
        FairnessControllerTestSuite,
        TOTAL_COUNTS,
    )


class TestFairbatchControllers(FairnessControllerTestSuite):
    """Unit tests for Fed-FairBatch / FedFB controllers."""

    server_cls = FairbatchControllerServer
    client_cls = FairbatchControllerClient

    def setup_server_controller(self) -> FairbatchControllerServer:
        return self.server_cls(f_type="equalized_odds")

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_setup(
        self,
        use_secagg: bool,
    ) -> None:
        aggregator = mock.create_autospec(Aggregator, instance=True)
        with pytest.warns(RuntimeWarning, match="SumAggregator"):
            agg_final, server, clients = (
                await self.run_finalize_fairness_setup(aggregator, use_secagg)
            )
        # Verify that aggregators were replaced with a SumAggregator.
        assert isinstance(agg_final, SumAggregator)
        assert all(
            isinstance(client.manager.aggrg, SumAggregator)
            for client in clients
        )
        # Verify that the sampling controller was properly instantiated.
        assert isinstance(server, FairbatchControllerServer)
        assert server.sampling_controller.counts == TOTAL_COUNTS
        # Verify that FairBatch sampling probas were shared and applied.
        probas = server.sampling_controller.get_sampling_probas()
        for client in clients:
            dst = client.manager.train_data
            assert isinstance(dst, FairbatchDataset)
            total = sum(probas[group] for group in dst.groups)
            expected = {
                group: probas[group] / total for group in dst.groups
            }
            assert dst.get_sampling_probabilities() == expected
