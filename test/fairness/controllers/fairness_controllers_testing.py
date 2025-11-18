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

"""Shared unit tests for Fairness controllers."""

import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest import mock

import numpy as np
import pytest

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import (
    FairnessControllerClient,
    FairnessControllerServer,
    FairnessDataset,
)
from declearn.messaging import (
    FairnessQuery,
    FairnessReply,
    FairnessSetupQuery,
    SerializedMessage,
)
from declearn.metrics import MeanMetric
from declearn.model.api import Model
from declearn.secagg.api import Decrypter, Encrypter
from declearn.secagg.messaging import SecaggFairnessReply
from declearn.test_utils import (
    assert_dict_equal,
    build_secagg_controllers,
    setup_mock_network_endpoints,
)
from declearn.training import TrainingManager

# Define arbitrary group definitions and sample counts.
CLIENT_COUNTS = [
    {(0, 0): 10, (0, 1): 10, (1, 0): 10, (1, 1): 10},
    {(0, 0): 10, (1, 0): 15, (1, 1): 10},
    {(0, 0): 10, (0, 1): 5, (1, 0): 10},
]
TOTAL_COUNTS = {(0, 0): 30, (0, 1): 15, (1, 0): 35, (1, 1): 20}


def build_mock_dataset(idx: int) -> mock.Mock:
    """Return a mock FairnessDataset with deterministic group counts."""
    counts = CLIENT_COUNTS[idx]
    dataset = mock.create_autospec(FairnessDataset, instance=True)
    dataset.get_sensitive_group_definitions.return_value = list(counts)
    dataset.get_sensitive_group_counts.return_value = counts
    return dataset


class FairnessControllerTestSuite:
    """Shared test suite for Fairness controllers."""

    # Types of controllers associated with a given test suite subclass.
    server_cls: Type[FairnessControllerServer]
    client_cls: Type[FairnessControllerClient]

    # Default expected local computed metrics. May be overloaded by subclasses.
    mock_client_metrics = [
        {"accuracy": {group: 1.0 for group in CLIENT_COUNTS[idx]}}
        for idx in range(len(CLIENT_COUNTS))
    ]

    def setup_server_controller(self) -> FairnessControllerServer:
        """Instantiate and return a server-side fairness controller."""
        return self.server_cls(f_type="accuracy_parity")

    def setup_mock_training_manager(
        self,
        idx: int,
    ) -> mock.MagicMock:
        """Setup and return a mock TrainingManager for a given client."""
        manager = mock.create_autospec(TrainingManager, instance=True)
        manager.aggrg = mock.create_autospec(Aggregator, instance=True)
        manager.logger = mock.create_autospec(logging.Logger, instance=True)
        manager.model = mock.create_autospec(Model, instance=True)
        manager.train_data = build_mock_dataset(idx)
        return manager

    def test_setup_server_from_specs(
        self,
    ) -> None:
        """Test instantiating a server-side controller 'from_specs'."""
        server = self.server_cls.from_specs(
            algorithm=self.server_cls.algorithm,
            f_type="demographic_parity",
        )
        assert isinstance(server, self.server_cls)

    def test_setup_client_from_setup_query(
        self,
    ) -> None:
        """Test that the server's setup query results in a proper client."""
        server = self.setup_server_controller()
        query = server.prepare_fairness_setup_query()
        assert isinstance(query, FairnessSetupQuery)
        manager = self.setup_mock_training_manager(idx=0)
        client = FairnessControllerClient.from_setup_query(query, manager)
        assert isinstance(client, self.client_cls)
        assert client.manager is manager
        assert client.fairness_function.f_type == server.f_type

    def setup_client_controller_from_server(
        self,
        server: FairnessControllerServer,
        idx: int,
    ) -> FairnessControllerClient:
        """Instantiate and return a client-side fairness controller."""
        manager = self.setup_mock_training_manager(idx)
        query = server.prepare_fairness_setup_query()
        return FairnessControllerClient.from_setup_query(query, manager)

    def setup_fairness_controllers_and_secagg(
        self,
        n_peers: int,
        use_secagg: bool,
    ) -> Tuple[
        FairnessControllerServer,
        List[FairnessControllerClient],
        Optional[Decrypter],
        Union[List[Encrypter], List[None]],
    ]:
        """Instantiate fairness and (optional) secagg controllers."""
        # Instantiate the server and client controllers.
        server = self.setup_server_controller()
        clients = [
            self.setup_client_controller_from_server(server, idx)
            for idx in range(n_peers)
        ]
        # Optionally set up SecAgg controllers, then return.
        if use_secagg:
            decrypter, encrypters = build_secagg_controllers(n_peers)
            return server, clients, decrypter, encrypters  # type: ignore
        return server, clients, None, [None for _ in range(n_peers)]

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_exchange_sensitive_groups_list_and_counts(
        self,
        use_secagg: bool,
    ) -> None:
        """Test that sensitive groups' definitions and counts works."""
        n_peers = len(CLIENT_COUNTS)
        # Instantiate the fairness and optional secagg controllers.
        server, clients, decrypter, encrypters = (
            self.setup_fairness_controllers_and_secagg(n_peers, use_secagg)
        )
        # Run setup coroutines, using mock network endpoints.
        async with setup_mock_network_endpoints(n_peers) as network:
            coro_server = server.exchange_sensitive_groups_list_and_counts(
                netwk=network[0], secagg=decrypter
            )
            coro_clients = [
                client.exchange_sensitive_groups_list_and_counts(
                    netwk=network[1][idx], secagg=encrypters[idx]
                )
                for idx, client in enumerate(clients)
            ]
            counts, *_ = await asyncio.gather(coro_server, *coro_clients)
        # Verify that expected attributes were assigned with expected values.
        assert isinstance(counts, list) and len(counts) == len(TOTAL_COUNTS)
        assert dict(zip(server.groups, counts, strict=False)) == TOTAL_COUNTS
        assert all(client.groups == server.groups for client in clients)

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_setup(
        self,
        use_secagg: bool,
    ) -> None:
        """Test that 'finalize_fairness_setup' works properly.

        This test should be overridden by subclasses to perform
        algorithm-specific verification (and warnings-catching).
        """
        aggregator = mock.create_autospec(Aggregator, instance=True)
        agg_final, *_ = await self.run_finalize_fairness_setup(
            aggregator, use_secagg
        )
        # Verify that the server returns an Aggregator.
        assert isinstance(agg_final, Aggregator)

    async def run_finalize_fairness_setup(
        self,
        aggregator: Aggregator,
        use_secagg: bool,
    ) -> Tuple[
        Aggregator, FairnessControllerServer, List[FairnessControllerClient]
    ]:
        """Run 'finalize_fairness_setup' and return controllers."""
        n_peers = len(CLIENT_COUNTS)
        # Instantiate the fairness and optional secagg controllers.
        server, clients, decrypter, encrypters = (
            self.setup_fairness_controllers_and_secagg(n_peers, use_secagg)
        )
        # Assign expected group definitions and counts.
        server.groups = sorted(list(TOTAL_COUNTS))
        for client in clients:
            client.groups = server.groups.copy()
        counts = [TOTAL_COUNTS[group] for group in server.groups]
        # Run setup coroutines, using mock network endpoints.
        async with setup_mock_network_endpoints(n_peers) as network:
            coro_server = server.finalize_fairness_setup(
                netwk=network[0],
                secagg=decrypter,
                counts=counts,
                aggregator=aggregator,
            )
            coro_clients = [
                client.finalize_fairness_setup(
                    netwk=network[1][idx],
                    secagg=encrypters[idx],
                )
                for idx, client in enumerate(clients)
            ]
            agg_final, *_ = await asyncio.gather(coro_server, *coro_clients)
        # Return the resulting aggregator and controllers.
        return agg_final, server, clients

    def test_setup_fairness_metrics(
        self,
    ) -> None:
        """Test that 'setup_fairness_metrics' has proper output type."""
        server = self.setup_server_controller()
        client = self.setup_client_controller_from_server(server, idx=0)
        metrics = client.setup_fairness_metrics()
        assert isinstance(metrics, list)
        assert all(isinstance(metric, MeanMetric) for metric in metrics)

    @pytest.mark.parametrize("idx", list(range(len(CLIENT_COUNTS))))
    def test_compute_fairness_metrics(
        self,
        idx: int,
    ) -> None:
        """Test that metrics computation works for a given client."""
        server = self.setup_server_controller()
        client = self.setup_client_controller_from_server(server, idx)
        client.groups = list(TOTAL_COUNTS)
        # Run mock computations.
        with mock.patch.object(
            client.computer, "compute_groupwise_metrics"
        ) as patch_compute:
            patch_compute.return_value = self.mock_client_metrics[idx].copy()
            share_values, local_values = client.compute_fairness_measures(32)
        # Verify that expected shareable values were output.
        patch_compute.assert_called_once()
        assert isinstance(share_values, list)
        expected_share = [
            group_values.get(group, 0.0) * CLIENT_COUNTS[idx].get(group, 0.0)
            for group_values in self.mock_client_metrics[idx].values()
            for group in client.groups
        ]
        assert share_values == expected_share
        # Verify that expected local values were output.
        assert isinstance(local_values, dict)
        expected_local = self.mock_client_metrics[idx].copy()
        if "accuracy" in expected_local:
            expected_local[client.fairness_function.f_type] = (
                client.fairness_function.compute_from_group_accuracy(
                    expected_local["accuracy"]
                )
            )
        assert_dict_equal(local_values, expected_local)

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_receive_and_aggregate_fairness_metrics(
        self,
        use_secagg: bool,
    ) -> None:
        """Test that server-side aggregation of metrics works properly."""
        # Setup a server controller and optionally some secagg controllers.
        n_peers = len(CLIENT_COUNTS)
        server = self.setup_server_controller()
        server.groups = list(TOTAL_COUNTS)
        decrypter, encrypters = (
            build_secagg_controllers(n_peers) if use_secagg else (None, None)
        )
        # Setup a mock network endpoint receiving local metrics.
        netwk = mock.create_autospec(NetworkServer, instance=True)
        replies = {
            f"client_{idx}": FairnessReply(
                [
                    g_val.get(group, 0.0) * CLIENT_COUNTS[idx].get(group, 0.0)
                    for g_val in self.mock_client_metrics[idx].values()
                    for group in list(TOTAL_COUNTS)
                ]
            )
            for idx in range(len(self.mock_client_metrics))
        }
        if encrypters:
            secagg_replies = {
                key: SecaggFairnessReply.from_cleartext_message(
                    cleartext=val, encrypter=encrypters[idx]
                )
                for idx, (key, val) in enumerate(replies.items())
            }
            netwk.wait_for_messages.return_value = {
                key: SerializedMessage.from_message_string(val.to_string())
                for key, val in secagg_replies.items()
            }
        else:
            netwk.wait_for_messages.return_value = {
                key: SerializedMessage.from_message_string(val.to_string())
                for key, val in replies.items()
            }
        # Run the reception and (secure-)aggregation of these replies.
        aggregated = await server.receive_and_aggregate_fairness_measures(
            netwk=netwk, secagg=decrypter
        )
        # Verify that outputs match expectations.
        assert isinstance(aggregated, list)
        expected = [
            sum(rv)
            for rv in zip(
                *[rep.values for rep in replies.values()], strict=False
            )
        ]
        assert np.allclose(np.array(aggregated), np.array(expected))

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_finalize_fairness_round(
        self,
        use_secagg: bool,
    ) -> None:
        """Test that 'finalize_fairness_round' works properly.

        This test should be overridden by subclasses to perform
        algorithm-specific verification.
        """
        _, _, metrics = await self.run_finalize_fairness_round(use_secagg)
        self.verify_fairness_round_metrics(metrics)

    async def run_finalize_fairness_round(
        self,
        use_secagg: bool,
    ) -> Tuple[
        FairnessControllerServer,
        List[FairnessControllerClient],
        List[Dict[str, Union[float, np.ndarray]]],
    ]:
        """Run 'finalize_fairness_round' after mocking previous steps.

        Return the server and client controllers, as well as the list
        of output metrics dictionary returned by the executed routines.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, server, clients = await self.run_finalize_fairness_setup(
                mock.MagicMock(),
                use_secagg,
            )
        # Run mock client computations and compute expected aggregate.
        share_vals: List[List[float]] = []
        local_vals: List[Dict[str, Dict[Tuple[Any, ...], float]]] = []
        for idx, client in enumerate(clients):
            with mock.patch.object(
                client.computer,
                "compute_groupwise_metrics",
                return_value=self.mock_client_metrics[idx].copy(),
            ):
                client_values = client.compute_fairness_measures(32)
                share_vals.append(client_values[0])
                local_vals.append(client_values[1])
        server_values = [
            float(sum(values)) for values in zip(*share_vals, strict=False)
        ]
        # Setup optional SecAgg and mock network communication endpoints.
        # Run the tested method.
        n_peers = len(clients)
        decrypter, encrypters = (
            build_secagg_controllers(n_peers)
            if use_secagg
            else (None, [None] * n_peers)  # type: ignore
        )
        async with setup_mock_network_endpoints(n_peers) as netwk:
            metrics = await asyncio.gather(
                server.finalize_fairness_round(
                    netwk=netwk[0],
                    secagg=decrypter,
                    values=server_values,
                ),
                *[
                    client.finalize_fairness_round(
                        netwk=netwk[1][idx],
                        secagg=encrypters[idx],
                        values=local_vals[idx],
                    )
                    for idx, client in enumerate(clients)
                ],
            )
        return server, clients, metrics

    def verify_fairness_round_metrics(
        self,
        metrics: List[Dict[str, Union[float, np.ndarray]]],
    ) -> None:
        """Verify that metrics output by fairness rounds match expectations.

        Input `metrics` contain the server-side metrics followed by each and
        every client-side ones, all formatted as dictionaries.
        """
        # Verify that all output metrics are dict with proper inner types.
        for m_dict in metrics:
            assert isinstance(m_dict, dict)
            assert all(isinstance(key, str) for key in m_dict)
            assert all(
                isinstance(val, (float, np.ndarray)) for val in m_dict.values()
            )
        # Verify that client dictionaries have the same keys.
        keys = list(metrics[1].keys())
        assert all(set(m_dict).issubset(keys) for m_dict in metrics[2:])

    @pytest.mark.parametrize(
        "use_secagg", [False, True], ids=["clrtxt", "secagg"]
    )
    @pytest.mark.asyncio
    async def test_fairness_end2end(
        self,
        use_secagg: bool,
    ) -> None:
        """Test that running both fairness setup and round routines works.

        This end-to-end test is about verifying that running all unit-tested
        components together does not raise exceptions. Details about unitary
        operations are left up to unit tests.
        """
        # Instantiate the fairness and optional secagg controllers.
        n_peers = len(CLIENT_COUNTS)
        decrypter: Optional[Decrypter] = None
        encrypters: List[Optional[Encrypter]] = [None] * n_peers
        if use_secagg:
            decrypter, encrypters = build_secagg_controllers(  # type: ignore
                n_peers
            )
        # Run end-to-end routines using mock communication endpoints.
        async with setup_mock_network_endpoints(n_peers=n_peers) as netwk:

            async def server_routine() -> None:
                """Server-side fairness setup and round routine."""
                nonlocal decrypter, netwk
                server = self.setup_server_controller()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    await server.setup_fairness(
                        netwk=netwk[0],
                        aggregator=mock.create_autospec(
                            Aggregator, instance=True
                        ),
                        secagg=decrypter,
                    )
                await netwk[0].broadcast_message(FairnessQuery(round_i=0))
                await server.run_fairness_round(
                    netwk=netwk[0],
                    secagg=decrypter,
                )

            async def client_routine(idx: int) -> None:
                """Client-side fairness setup and round routine."""
                nonlocal encrypters, netwk
                # Instantiate the client-side controller.
                received = await netwk[1][idx].recv_message()
                setup_query = await verify_server_message_validity(
                    netwk[1][idx], received, FairnessSetupQuery
                )
                client = FairnessControllerClient.from_setup_query(
                    setup_query, manager=self.setup_mock_training_manager(idx)
                )
                # Run the fairness setup routine.
                await client.setup_fairness(netwk[1][idx], encrypters[idx])
                # Run the fairness round routine.
                received = await netwk[1][idx].recv_message()
                round_query = await verify_server_message_validity(
                    netwk[1][idx], received, FairnessQuery
                )
                await client.run_fairness_round(
                    netwk[1][idx], round_query, encrypters[idx]
                )

            await asyncio.gather(
                server_routine(),
                *[client_routine(idx) for idx in range(n_peers)],
            )
