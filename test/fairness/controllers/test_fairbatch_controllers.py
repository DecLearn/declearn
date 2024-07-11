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
from typing import List
from unittest import mock

import pytest

from declearn.aggregator import Aggregator, SumAggregator
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
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import (
        FairnessControllerTestSuite,
        CLIENT_COUNTS,
        TOTAL_COUNTS,
    )


class TestFairbatchControllers(FairnessControllerTestSuite):
    """Unit tests for Fed-FairBatch / FedFB controllers."""

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
            FairbatchControllerServer(
                f_type="demographic_parity",
                fedfb=False,
            )
            patch_setup_fairbatch.assert_called_once()
        with mock.patch(
            "declearn.fairness.fairbatch._server.setup_fedfb_controller"
        ) as patch_setup_fedfb:
            FairbatchControllerServer(
                f_type="demographic_parity",
                fedfb=True,
            )
            patch_setup_fedfb.assert_called_once()

    def test_init_alpha_param(self) -> None:
        """Test that server-side 'fedfb' parameter is enforced."""
        alpha = mock.MagicMock()
        server = FairbatchControllerServer(
            f_type="demographic_parity", alpha=alpha
        )
        assert server.sampling_controller.alpha is alpha
