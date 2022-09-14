# coding: utf-8

"""Declearn demonstration / testing code."""

import multiprocessing as mp
import tempfile
import warnings
from typing import Dict, Optional

import numpy as np
import pytest
with warnings.catch_warnings():  # silence tensorflow import-time warnings
    warnings.simplefilter("ignore")
    import tensorflow as tf  # type: ignore
import torch
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.communication import build_client, build_server
from declearn2.communication.api import Client, Server
from declearn2.dataset import InMemoryDataset
from declearn2.model.api import Model
from declearn2.model.sklearn import SklearnSGDModel
from declearn2.model.tensorflow import TensorflowModel
from declearn2.model.torch import TorchModel
from declearn2.main import FederatedClient, FederatedServer
from declearn2.strategy import FedAvg, FedAvgM, Scaffold, ScaffoldM


class DeclearnTestCase:
    """Test-case for the "main" federated learning orchestrating classes."""
    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            kind: Literal['Reg', 'Bin', 'Clf'],
            framework: Literal['Sksgd', 'Tflow', 'Torch'],
            strategy: Literal['FedAvg', 'FedAvgM', 'Scaffold', 'ScaffoldM'],
            nb_clients: int,
            protocol: Literal['grpc', 'websockets'],
            use_ssl: bool,
            ssl_cert: Dict[str, str],
            rounds: int = 5,
        ) -> None:
        # arguments provide modularity; pylint: disable=too-many-arguments
        self.kind = kind
        self.framework = framework
        self.strategy = {
            cls.__name__: cls for cls in (FedAvg, FedAvgM, Scaffold, ScaffoldM)
        }[strategy]
        self.nb_clients = nb_clients
        self.protocol = protocol
        self.use_ssl = use_ssl
        self.ssl_cert = ssl_cert
        self.rounds = rounds
        self.coefs = np.random.normal(
            size=(32, 4 if kind == 'Clf' else 1)
        ).astype(np.float32)

    def build_model(
            self,
        ) -> Model:
        """Return a Model suitable for the learning task and framework."""
        if self.framework.lower() == 'sksgd':
            return SklearnSGDModel.from_parameters(
                kind=("regressor" if self.kind == 'Reg' else "classifier")
            )
        if self.framework.lower() == 'tflow':
            return self._build_tflow_model()
        if self.framework.lower() == 'torch':
            return self._build_torch_model()
        raise ValueError("Invalid 'framework' attribute.")

    def _build_tflow_model(
            self,
        ) -> TensorflowModel:
        """Return a TensorflowModel suitable for the learning task."""
        if self.kind == 'Reg':
            output_layer = tf.keras.layers.Dense(1)
            loss = 'mse'
        elif self.kind == 'Bin':
            output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
            loss = 'binary_crossentropy'
        elif self.kind == 'Clf':
            output_layer = tf.keras.layers.Dense(4, activation='softmax')
            loss = 'sparse_categorical_crossentropy'
        else:
            raise ValueError("Invalid 'kind' attribute.")
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((32,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            output_layer,
        ])
        return TensorflowModel(model, loss, metrics=None)

    def _build_torch_model(
            self,
        ) -> TorchModel:
        """Return a TorchModel suitable for the learning task."""
        stack =  [
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
        ]
        if self.kind == 'Reg':
            stack.append(torch.nn.Linear(8, 1))
            loss = torch.nn.MSELoss()  # type: torch.nn.Module
        elif self.kind == 'Bin':
            stack.append(torch.nn.Linear(8, 1))
            stack.append(torch.nn.Sigmoid())
            loss = torch.nn.BCELoss()
        elif self.kind == 'Clf':
            stack.append(torch.nn.Linear(8, 4))
            stack.append(torch.nn.Softmax(-1))
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid 'kind' attribute.")
        model = torch.nn.Sequential(*stack)
        return TorchModel(model, loss)

    def build_dataset(
            self,
        ) -> InMemoryDataset:
        """Return an in-memory dataset suitable for the learning task."""
        features = np.random.normal(size=(1000, 32)).astype(np.float32)
        if self.kind in ("Reg", "Bin"):
            noise = np.random.normal(size=(1000,)).astype(np.float32)
            target = np.matmul(features, self.coefs)[:, 0] + noise
            if self.kind == "Bin":
                target = (np.tanh(target).round() > 0).astype(np.float32)
                if self.framework.lower() == "torch":
                    target = np.expand_dims(target, 1)
        else:
            noise = np.random.normal(size=(1000, 4)).astype(np.float32)
            target = np.matmul(features, self.coefs) + noise
            target = target.argmax(axis=1)
        return InMemoryDataset(
            features, target, s_wght=None, expose_classes=(self.kind != "Reg")
        )

    def build_netwk_server(
            self,
        ) -> Server:
        """Return a communication Server."""
        return build_server(
            self.protocol, host="localhost", port=8765,
            certificate=self.ssl_cert["server_cert"] if self.use_ssl else None,
            private_key=self.ssl_cert["server_pkey"] if self.use_ssl else None,
        )

    def build_netwk_client(
            self,
            name: str = "client",
        ) -> Client:
        """Return a communication Client."""
        server_uri = "localhost:8765"
        if self.protocol == "websockets":
            server_uri = f"ws{'s' * self.use_ssl}://" + server_uri
        return build_client(
            self.protocol, server_uri, name=name,
            certificate=self.ssl_cert["client_cert"] if self.use_ssl else None,
        )

    def run_federated_server(
            self,
        ) -> None:
        """Set up and run a FederatedServer."""
        model = self.build_model()
        netwk = self.build_netwk_server()
        strat = self.strategy(eta_l=0.01)
        server = FederatedServer(model, netwk, strat, batch_size=100)
        server.run(rounds=self.rounds, min_clients=self.nb_clients)

    def run_federated_client(
            self,
            name: str = "client",
        ) -> None:
        """Set up and run a FederatedClient."""
        netwk = self.build_netwk_client(name)
        dataset = self.build_dataset()
        with tempfile.TemporaryDirectory() as folder:
            client = FederatedClient(netwk, dataset, folder=folder)
            client.run()


def run_test_case(
        kind: Literal['Reg', 'Bin', 'Clf'],
        framework: Literal['Sksgd', 'Tflow', 'Torch'],
        strategy: Literal['FedAvg', 'FedAvgM', 'Scaffold', 'ScaffoldM'],
        nb_clients: int,
        protocol: Literal['grpc', 'websockets'],
        use_ssl: bool,
        ssl_cert: Optional[Dict[str, str]] = None,
        rounds: int = 2,
    ) -> None:
    """Run a given test case, using processes to isolate server and clients."""
    # arguments provide modularity; pylint: disable=too-many-arguments
    test_case = DeclearnTestCase(
        kind, framework, strategy, nb_clients, protocol, use_ssl,
        ssl_cert or {}, rounds
    )
    server = mp.Process(target=test_case.run_federated_server)
    clients = [
        mp.Process(target=test_case.run_federated_client, args=(f"cli_{i}",))
        for i in range(nb_clients)
    ]
    try:
        # Start all processes.
        server.start()
        for process in clients:
            process.start()
        # Regularly check for any failed process (exit if so).
        while server.is_alive() or any(p.is_alive() for p in clients):
            if server.exitcode or any(p.exitcode for p in clients):
                break
            server.join(timeout=1)
        # Assert that all processes exited properly.
        assert server.exitcode == 0
        assert all(p.exitcode == 0 for p in clients)
    finally:
        # Ensure that all processes are terminated.
        server.terminate()
        for process in clients:
            process.terminate()


@pytest.mark.parametrize("strategy", ['FedAvg', 'FedAvgM', 'Scaffold'])
@pytest.mark.parametrize("framework", ['Sksgd', 'Tflow', 'Torch'])
@pytest.mark.parametrize("kind", ['Reg', 'Bin', 'Clf'])
@pytest.mark.filterwarnings("ignore: PyTorch JSON serialization")
def test_declearn(
        kind: Literal['Reg', 'Bin', 'Clf'],
        framework: Literal['Sksgd', 'Tflow', 'Torch'],
        strategy: Literal['FedAvg', 'FedAvgM', 'Scaffold', 'ScaffoldM'],
    ) -> None:
    """Pytest-collected functional test of declearn's main classes.

    Note: Run 2 training rounds with 2 clients.
    Note: Use unsecured websockets communication, which are less
          costful to establish than gRPC and/or SSL-secured ones
          (the latter due to the certificates-generation costs).
    """
    run_test_case(
        kind, framework, strategy,
        nb_clients=2, protocol="websockets", use_ssl=False,
        ssl_cert=None, rounds=2,
    )