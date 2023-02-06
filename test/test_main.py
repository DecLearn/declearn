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

"""Declearn demonstration / testing code."""

import tempfile
import warnings
from typing import Any, Dict, Literal, Optional

import numpy as np
import pytest

from declearn.communication import (
    build_client,
    build_server,
    list_available_protocols,
)
from declearn.communication.api import NetworkClient, NetworkServer
from declearn.dataset import InMemoryDataset
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.main import FederatedClient, FederatedServer
from declearn.test_utils import run_as_processes

# Select the subset of tests to run, based on framework availability.
# Note: TensorFlow and Torch (-related) imports are delayed due to this.
# pylint: disable=ungrouped-imports
FRAMEWORKS = ["Sksgd", "Tflow", "Torch"]
try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    FRAMEWORKS.remove("Tflow")
else:
    from declearn.model.tensorflow import TensorflowModel
try:
    import torch
except ModuleNotFoundError:
    FRAMEWORKS.remove("Torch")
else:
    from declearn.model.torch import TorchModel


class DeclearnTestCase:
    """Test-case for the "main" federated learning orchestrating classes."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        kind: Literal["Reg", "Bin", "Clf"],
        framework: Literal["Sksgd", "Tflow", "Torch"],
        strategy: Literal["FedAvg", "FedAvgM", "Scaffold", "ScaffoldM"],
        nb_clients: int,
        protocol: Literal["grpc", "websockets"],
        use_ssl: bool,
        ssl_cert: Dict[str, str],
        rounds: int = 5,
    ) -> None:
        # arguments provide modularity; pylint: disable=too-many-arguments
        self.kind = kind
        self.framework = framework
        self.strategy = strategy
        self.nb_clients = nb_clients
        self.protocol = protocol
        self.use_ssl = use_ssl
        self.ssl_cert = ssl_cert
        self.rounds = rounds
        size = (32, 4 if kind == "Clf" else 1)
        self.coefs = np.random.normal(size=size).astype(np.float32)

    def build_model(
        self,
    ) -> Model:
        """Return a Model suitable for the learning task and framework."""
        if self.framework.lower() == "sksgd":
            return SklearnSGDModel.from_parameters(
                kind=("regressor" if self.kind == "Reg" else "classifier")
            )
        if self.framework.lower() == "tflow":
            return self._build_tflow_model()
        if self.framework.lower() == "torch":
            return self._build_torch_model()
        raise ValueError("Invalid 'framework' attribute.")

    def _build_tflow_model(
        self,
    ) -> Model:
        """Return a TensorflowModel suitable for the learning task."""
        if self.kind == "Reg":
            output_layer = tf.keras.layers.Dense(1)
            loss = "mse"
        elif self.kind == "Bin":
            output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
            loss = "binary_crossentropy"
        elif self.kind == "Clf":
            output_layer = tf.keras.layers.Dense(4, activation="softmax")
            loss = "sparse_categorical_crossentropy"
        else:
            raise ValueError("Invalid 'kind' attribute.")
        stack = [
            tf.keras.layers.InputLayer((32,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            output_layer,
        ]
        model = tf.keras.Sequential(stack)
        return TensorflowModel(model, loss, metrics=None)

    def _build_torch_model(
        self,
    ) -> Model:
        """Return a TorchModel suitable for the learning task."""
        # Build the model and return it.
        stack = [
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
        ]
        if self.kind == "Reg":
            stack.append(torch.nn.Linear(8, 1))
            loss = torch.nn.MSELoss()  # type: torch.nn.Module
        elif self.kind == "Bin":
            stack.append(torch.nn.Linear(8, 1))
            stack.append(torch.nn.Sigmoid())
            loss = torch.nn.BCELoss()
        elif self.kind == "Clf":
            stack.append(torch.nn.Linear(8, 4))
            stack.append(torch.nn.Softmax(-1))
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid 'kind' attribute.")
        model = torch.nn.Sequential(*stack)
        return TorchModel(model, loss)

    def build_dataset(
        self,
        size: int = 1000,
    ) -> InMemoryDataset:
        """Return an in-memory dataset suitable for the learning task."""
        features = np.random.normal(size=(size, 32)).astype(np.float32)
        if self.kind in ("Reg", "Bin"):
            noise = np.random.normal(size=(size,)).astype(np.float32)
            target = np.matmul(features, self.coefs)[:, 0] + noise
            if self.kind == "Bin":
                target = (np.tanh(target).round() > 0).astype(np.float32)
                if self.framework.lower() == "torch":
                    target = np.expand_dims(target, 1)
        else:
            noise = np.random.normal(size=(size, 4)).astype(np.float32)
            target = np.matmul(features, self.coefs) + noise
            target = target.argmax(axis=1)
        return InMemoryDataset(
            features, target, s_wght=None, expose_classes=(self.kind != "Reg")
        )

    def build_netwk_server(
        self,
    ) -> NetworkServer:
        """Return a NetworkServer instance."""
        return build_server(
            self.protocol,
            host="127.0.0.1",
            port=8765,
            certificate=self.ssl_cert["server_cert"] if self.use_ssl else None,
            private_key=self.ssl_cert["server_pkey"] if self.use_ssl else None,
        )

    def build_netwk_client(
        self,
        name: str = "client",
    ) -> NetworkClient:
        """Return a NetworkClient instance."""
        server_uri = "localhost:8765"
        if self.protocol == "websockets":
            server_uri = f"ws{'s' * self.use_ssl}://" + server_uri
        return build_client(
            self.protocol,
            server_uri,
            name=name,
            certificate=self.ssl_cert["client_cert"] if self.use_ssl else None,
        )

    def build_optim_config(self) -> Dict[str, Any]:
        """Return parameters to instantiate a FLOptimConfig."""
        client_modules = []
        server_modules = []
        if self.strategy == "Scaffold":
            client_modules.append("scaffold-client")
            server_modules.append("scaffold-server")
        if self.strategy in ("FedAvgM", "ScaffoldM"):
            server_modules.append("momentum")
        return {
            "aggregator": "averaging",
            "client_opt": {"lrate": 0.01, "modules": client_modules},
            "server_opt": {"lrate": 1.0, "modules": server_modules},
        }

    def run_federated_server(
        self,
    ) -> None:
        """Set up and run a FederatedServer."""
        model = self.build_model()
        netwk = self.build_netwk_server()
        optim = self.build_optim_config()
        with tempfile.TemporaryDirectory() as folder:
            server = FederatedServer(model, netwk, optim, checkpoint=folder)
            config = {
                "rounds": self.rounds,
                "register": {"max_clients": self.nb_clients, "timeout": 20},
                "training": {"batch_size": 100},
            }
            server.run(config)

    def run_federated_client(
        self,
        name: str = "client",
    ) -> None:
        """Set up and run a FederatedClient."""
        netwk = self.build_netwk_client(name)
        train = self.build_dataset(size=1000)
        valid = self.build_dataset(size=250)
        with tempfile.TemporaryDirectory() as folder:
            client = FederatedClient(netwk, train, valid, folder)
            client.run()


def run_test_case(
    kind: Literal["Reg", "Bin", "Clf"],
    framework: Literal["Sksgd", "Tflow", "Torch"],
    strategy: Literal["FedAvg", "FedAvgM", "Scaffold", "ScaffoldM"],
    nb_clients: int,
    protocol: Literal["grpc", "websockets"],
    use_ssl: bool,
    ssl_cert: Optional[Dict[str, str]] = None,
    rounds: int = 2,
) -> None:
    """Run a given test case, using processes to isolate server and clients."""
    # arguments provide modularity; pylint: disable=too-many-arguments
    # Set up a test case object.
    # fmt: off
    test_case = DeclearnTestCase(
        kind, framework, strategy, nb_clients, protocol, use_ssl,
        ssl_cert or {}, rounds,
    )
    # fmt: on
    # Prepare the server and clients routines.
    server = (test_case.run_federated_server, tuple())  # type: ignore
    clients = [
        (test_case.run_federated_client, (f"cli_{i}",))
        for i in range(nb_clients)
    ]
    # Run them concurrently using multiprocessing.
    exitcodes = run_as_processes(server, *clients)
    # Verify that all processes ended without error nor interruption.
    assert all(code == 0 for code in exitcodes)


@pytest.mark.parametrize("strategy", ["FedAvg", "FedAvgM", "Scaffold"])
@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("kind", ["Reg", "Bin", "Clf"])
@pytest.mark.filterwarnings("ignore: PyTorch JSON serialization")
def test_declearn(
    kind: Literal["Reg", "Bin", "Clf"],
    framework: Literal["Sksgd", "Tflow", "Torch"],
    strategy: Literal["FedAvg", "FedAvgM", "Scaffold", "ScaffoldM"],
    fulltest: bool,
) -> None:
    """Pytest-collected functional test of declearn's main classes.

    Note: Run 2 training rounds with 2 clients.
    Note: Use unsecured websockets communication, which are less
          costful to establish than gRPC and/or SSL-secured ones
          (the latter due to the certificates-generation costs).
    Note: if websockets is unavailable, use gRPC (warn) or fail.
    """
    if not fulltest:
        if (kind != "Reg") or (strategy == "FedAvgM"):
            pytest.skip("skip scenario (no --fulltest option)")
    protocol = "websockets"  # type: Literal["grpc", "websockets"]
    if "websockets" not in list_available_protocols():
        if "grpc" not in list_available_protocols():
            pytest.fail("Both 'grpc' and 'websockets' are unavailable.")
        protocol = "grpc"
        warnings.warn("Using 'grpc' as 'websockets' is unavailable.")
    # fmt: off
    run_test_case(
        kind, framework, strategy,
        nb_clients=2, protocol=protocol, use_ssl=False, rounds=2,
    )
