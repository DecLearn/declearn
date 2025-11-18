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

"""Unit tests for Fed-FairBatch controllers."""

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
)
from declearn.fairness.fairbatch import (
    FairbatchControllerClient,
    FairbatchControllerServer,
    FairbatchDataset,
    FairbatchSamplingController,
)
from declearn.test_utils import make_importable, setup_mock_network_endpoints

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import (
        CLIENT_COUNTS,
        TOTAL_COUNTS,
        FairnessControllerTestSuite,
    )


class TestFairbatchControllers(FairnessControllerTestSuite):
    """Unit tests for Fed-FairBatch / FedFB controllers."""

    # similar code to FairGrad and parent code; pylint: disable=duplicate-code

    server_cls = FairbatchControllerServer
    client_cls = FairbatchControllerClient

    mock_client_metrics = [
        {
            "accuracy": {group: 1.0 for group in CLIENT_COUNTS[idx]},
            "loss": {group: 0.05 for group in CLIENT_COUNTS[idx]},
        }
        for idx in range(len(CLIENT_COUNTS))
    ]

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
        # Verify that the sampling controller was properly instantiated.
        assert isinstance(server, FairbatchControllerServer)
        assert server.sampling_controller.counts == TOTAL_COUNTS
        # Verify that FairBatch sampling probas were shared and applied.
        self.verify_fairbatch_sampling_probas_coherence(server, clients)

    def verify_fairbatch_sampling_probas_coherence(
        self,
        server: FairnessControllerServer,
        clients: List[FairnessControllerClient],
    ) -> None:
        """Verify that FairBatch sampling probas were shared and applied."""
        assert isinstance(server, FairbatchControllerServer)
        probas = server.sampling_controller.get_sampling_probas()
        for client in clients:
            dst = client.manager.train_data
            assert isinstance(dst, FairbatchDataset)
            total = sum(probas[group] for group in dst.groups)
            expected = {group: probas[group] / total for group in dst.groups}
            assert dst.get_sampling_probabilities() == expected

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_round(
        self,
        use_secagg: bool,
    ) -> None:
        with mock.patch.object(
            FairbatchSamplingController,
            "update_from_federated_losses",
        ) as patch_update_sampling_probas:
            server, clients, metrics = await self.run_finalize_fairness_round(
                use_secagg
            )
        self.verify_fairness_round_metrics(metrics)
        patch_update_sampling_probas.assert_called_once()
        self.verify_fairbatch_sampling_probas_coherence(server, clients)

    def test_init_fedfb_param(self) -> None:
        """Test that server-side 'fedfb' parameter is enforced."""
        with mock.patch(
            "declearn.fairness.fairbatch._server.setup_fairbatch_controller"
        ) as patch_setup_fairbatch:
            controller = FairbatchControllerServer(
                f_type="demographic_parity",
                fedfb=False,
            )
            assert not controller.fedfb
            patch_setup_fairbatch.assert_called_once()
        with mock.patch(
            "declearn.fairness.fairbatch._server.setup_fedfb_controller"
        ) as patch_setup_fedfb:
            controller = FairbatchControllerServer(
                f_type="demographic_parity",
                fedfb=True,
            )
            assert controller.fedfb
            patch_setup_fedfb.assert_called_once()

    def test_init_alpha_param(self) -> None:
        """Test that server-side 'fedfb' parameter is enforced."""
        alpha = mock.MagicMock()
        server = FairbatchControllerServer(
            f_type="demographic_parity", alpha=alpha
        )
        assert server.sampling_controller.alpha is alpha

    @pytest.mark.asyncio
    async def test_finalize_fairness_setup_error(
        self,
    ) -> None:
        """Test that FairBatch probas update error-catching works properly."""
        n_peers = len(CLIENT_COUNTS)
        # Instantiate the fairness controllers.
        server = self.setup_server_controller()
        clients = [
            self.setup_client_controller_from_server(server, idx)
            for idx in range(n_peers)
        ]
        # Assign expected group definitions and counts.
        server.groups = sorted(list(TOTAL_COUNTS))
        for client in clients:
            client.groups = server.groups.copy()
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
            # Have the sampling probabilities' assignment fail.
            with mock.patch.object(
                FairbatchDataset,
                "set_sampling_probabilities",
                side_effect=Exception,
            ) as patch_set_sampling_probabilities:
                exc_server, *exc_clients = await asyncio.gather(
                    coro_server, *coro_clients, return_exceptions=True
                )
        # Assert that expected exceptions were raised.
        assert isinstance(exc_server, ErrorMessageException)
        assert all(isinstance(exc, RuntimeError) for exc in exc_clients)
        assert patch_set_sampling_probabilities.call_count == n_peers
