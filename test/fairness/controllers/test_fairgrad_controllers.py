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

"""Unit tests for Fed-FairGrad controllers."""

import os
from unittest import mock

import pytest

from declearn.aggregator import Aggregator, SumAggregator
from declearn.fairness.api import FairnessDataset
from declearn.fairness.fairgrad import (
    FairgradControllerClient,
    FairgradControllerServer,
)
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import FairnessControllerTestSuite


class TestFairgradControllers(FairnessControllerTestSuite):
    """Unit tests for Fed-FairGrad controllers."""

    server_cls = FairgradControllerServer
    client_cls = FairgradControllerClient

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
        # Verify that FairGrad weights were shared and applied.
        assert isinstance(server, FairgradControllerServer)
        weights = server.weights_controller.get_current_weights(norm_nk=True)
        expectw = dict(zip(server.groups, weights))
        for client in clients:
            mock_dst = client.manager.train_data
            assert isinstance(mock_dst, FairnessDataset)
            assert isinstance(mock_dst, mock.NonCallableMagicMock)
            mock_dst.set_sensitive_group_weights.assert_called_once_with(
                weights=expectw, adjust_by_counts=True
            )
