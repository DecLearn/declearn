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

"""Unit tests for Fed-FairGrad controllers."""

import asyncio
import os
from typing import List
from unittest import mock

import pytest

from declearn.aggregator import Aggregator, SumAggregator
from declearn.communication.utils import ErrorMessageException
from declearn.fairness.api import (
    FairnessControllerClient,
    FairnessControllerServer,
    FairnessDataset,
)
from declearn.fairness.fairgrad import (
    FairgradControllerClient,
    FairgradControllerServer,
    FairgradWeightsController,
)
from declearn.test_utils import make_importable, setup_mock_network_endpoints

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import (
        CLIENT_COUNTS,
        TOTAL_COUNTS,
        FairnessControllerTestSuite,
    )


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
            (
                agg_final,
                server,
                clients,
            ) = await self.run_finalize_fairness_setup(aggregator, use_secagg)
        # Verify that aggregators were replaced with a SumAggregator.
        assert isinstance(agg_final, SumAggregator)
        assert all(
            isinstance(client.manager.aggrg, SumAggregator)
            for client in clients
        )
        # Verify that FairgradWeights were shared and applied.
        self.verify_fairgrad_weights_coherence(server, clients)

    def verify_fairgrad_weights_coherence(
        self,
        server: FairnessControllerServer,
        clients: List[FairnessControllerClient],
    ) -> None:
        """Verify that FairGrad weights were shared to clients and applied."""
        assert isinstance(server, FairgradControllerServer)
        weights = server.weights_controller.get_current_weights(norm_nk=True)
        expectw = dict(zip(server.groups, weights, strict=False))
        for client in clients:
            mock_dst = client.manager.train_data
            assert isinstance(mock_dst, FairnessDataset)
            assert isinstance(mock_dst, mock.NonCallableMagicMock)
            mock_dst.set_sensitive_group_weights.assert_called_with(
                weights=expectw, adjust_by_counts=True
            )

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_round(
        self,
        use_secagg: bool,
    ) -> None:
        with mock.patch.object(
            FairgradWeightsController,
            "update_weights_based_on_accuracy",
        ) as patch_update_weights:
            server, clients, metrics = await self.run_finalize_fairness_round(
                use_secagg
            )
        self.verify_fairness_round_metrics(metrics)
        patch_update_weights.assert_called_once()
        self.verify_fairgrad_weights_coherence(server, clients)

    @pytest.mark.asyncio
    async def test_finalize_fairness_setup_error(
        self,
    ) -> None:
        """Test that FairGrad weights setup error-catching works properly."""
        n_peers = len(CLIENT_COUNTS)
        # Instantiate the fairness controllers.
        server = self.setup_server_controller()
        clients = [
            self.setup_client_controller_from_server(server, idx)
            for idx in range(n_peers)
        ]
        # Assign expected group definitions and counts.
        # Have client datasets fail upon receiving sensitive group weights.
        server.groups = sorted(list(TOTAL_COUNTS))
        for client in clients:
            client.groups = server.groups.copy()
            mock_dst = client.manager.train_data
            assert isinstance(mock_dst, mock.NonCallableMagicMock)
            mock_dst.set_sensitive_group_weights.side_effect = Exception
        counts = [TOTAL_COUNTS[group] for group in server.groups]
        # Run setup coroutines, using mock network endpoints.
        async with setup_mock_network_endpoints(n_peers) as network:
            coro_server = server.finalize_fairness_setup(
                netwk=network[0],
                secagg=None,
                counts=counts,
                aggregator=mock.create_autospec(SumAggregator, instance=True),
            )
            coro_clients = [
                client.finalize_fairness_setup(
                    netwk=network[1][idx],
                    secagg=None,
                )
                for idx, client in enumerate(clients)
            ]
            exc_server, *exc_clients = await asyncio.gather(
                coro_server, *coro_clients, return_exceptions=True
            )
        # Assert that expected exceptions were raised.
        assert isinstance(exc_server, ErrorMessageException)
        assert all(isinstance(exc, RuntimeError) for exc in exc_clients)
