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

"""Unit tests for FairFed controllers."""

import os
from typing import Dict, List, Union
from unittest import mock

import numpy as np
import pytest

from declearn.aggregator import Aggregator
from declearn.fairness.fairfed import (
    FairfedAggregator,
    FairfedControllerClient,
    FairfedControllerServer,
    FairfedValueComputer,
)
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from fairness_controllers_testing import (
        CLIENT_COUNTS,
        FairnessControllerTestSuite,
    )


class TestFairfedControllers(FairnessControllerTestSuite):
    """Unit tests for FairFed controllers."""

    server_cls = FairfedControllerServer
    client_cls = FairfedControllerClient

    def setup_server_controller(self) -> FairfedControllerServer:
        return self.server_cls(f_type="equality_of_opportunity", strict=False)

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_setup(
        self,
        use_secagg: bool,
    ) -> None:
        aggregator = mock.create_autospec(Aggregator, instance=True)
        with mock.patch.object(
            FairfedValueComputer, "initialize"
        ) as patch_initialize_computer:
            with mock.patch.object(
                FairfedAggregator, "initialize_local_weight"
            ) as patch_initialize_aggregator:
                with pytest.warns(RuntimeWarning, match="Aggregator"):
                    (
                        agg_final,
                        server,
                        clients,
                    ) = await self.run_finalize_fairness_setup(
                        aggregator, use_secagg
                    )
        # Verify that aggregators were replaced with a FairfedAggregator.
        assert isinstance(agg_final, FairfedAggregator)
        assert all(
            isinstance(client.manager.aggrg, FairfedAggregator)
            for client in clients
        )
        # Verify that all FairFed computers were initialized.
        calls = [
            mock.call(groups=client.fairness_function.groups)
            for client in clients
        ]
        calls.append(mock.call(groups=server.groups))
        patch_initialize_computer.assert_has_calls(calls, any_order=True)
        # Verify that all FairFed aggregators were initialized.
        calls = [
            mock.call(n_samples=sum(client_counts.values()))
            for client_counts in CLIENT_COUNTS
        ]
        patch_initialize_aggregator.assert_has_calls(calls, any_order=True)

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_round(
        self,
        use_secagg: bool,
    ) -> None:
        # Run the routine.
        with mock.patch.object(
            FairfedAggregator, "update_local_weight"
        ) as patch_update_fairfed_local_weight:
            server, clients, metrics = await self.run_finalize_fairness_round(
                use_secagg
            )
        # Verify output metrics, including coherence of fairfed-specific ones.
        self.verify_fairness_round_metrics(metrics)
        # Verify that expected client-wise weights update occurred.
        calls = [
            mock.call(
                delta_loc=client_metrics["fairfed_delta"],
                delta_avg=client_metrics["fairfed_deltavg"],
            )
            for client_metrics in metrics[1:]
        ]
        patch_update_fairfed_local_weight.assert_has_calls(
            calls, any_order=True
        )
        # Verify that fairfed synthetic values were properly computed.
        assert isinstance(server, FairfedControllerServer)
        fairness = {
            group: float(metrics[0][f"{server.f_type}_{group}"])
            for group in server.groups
        }
        assert metrics[0]["fairfed_value"] == (
            server.fairfed_computer.compute_synthetic_fairness_value(fairness)
        )
        for client, client_metrics in zip(clients, metrics[1:], strict=False):
            assert isinstance(client, FairfedControllerClient)
            fairness = {
                group: float(client_metrics[f"{server.f_type}_{group}"])
                for group in client.fairness_function.groups
            }
            assert client_metrics["fairfed_value"] == (
                client.fairfed_computer.compute_synthetic_fairness_value(
                    fairness
                )
            )

    def verify_fairness_round_metrics(
        self,
        metrics: List[Dict[str, Union[float, np.ndarray]]],
    ) -> None:
        # Perform basic verifications.
        super().verify_fairness_round_metrics(metrics)
        # Verify that computed fairfed delta values are coherent.
        server = metrics[0]
        clients = metrics[1:]
        for client in clients:
            assert client["fairfed_delta"] == abs(
                client["fairfed_value"] - server["fairfed_value"]
            )
            assert client["fairfed_deltavg"] == server["fairfed_deltavg"]
        assert server["fairfed_deltavg"] == (
            sum(client["fairfed_delta"] for client in clients) / len(clients)
        )

    @pytest.mark.parametrize(
        "strict", [True, False], ids=["strict", "extended"]
    )
    def test_init_params(
        self,
        strict: bool,
    ) -> None:
        """Test that instantiation parameters are properly passed."""
        rng = np.random.default_rng()
        beta = abs(rng.normal())
        target = int(rng.choice(2))
        controller = FairfedControllerServer(
            f_type="demographic_parity",
            beta=beta,
            strict=strict,
            target=target,
        )
        assert controller.beta == beta
        assert controller.fairfed_computer.f_type == "demographic_parity"
        assert controller.strict is strict
        assert controller.fairfed_computer.strict is strict
        assert controller.fairfed_computer.target is target
        # Verify that parameters are transmitted to clients.
        client = self.setup_client_controller_from_server(controller, idx=0)
        assert isinstance(client, FairfedControllerClient)
        assert client.beta == controller.beta
        assert client.fairfed_computer.f_type == "demographic_parity"
        assert client.strict is strict
        assert client.fairfed_computer.strict is strict
