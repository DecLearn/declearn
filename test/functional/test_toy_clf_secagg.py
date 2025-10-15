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

"""Integration test using SecAgg (and opt. Scaffold) on a toy problem.

* Set up a toy classification dataset, with some client heterogeneity.
* Run a FedAvg or Scaffold federated learning experiment in cleartext.
* Re-run the experiment adding SecAgg, and verify that convergence is
  not significatively altered due to it.
"""

import asyncio
import json
import os
import tempfile
from typing import List, Optional, Tuple, Union

import pytest
import sklearn.cluster  # type: ignore
import sklearn.datasets  # type: ignore
import sklearn.model_selection  # type: ignore
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.sklearn import SklearnSGDModel
from declearn.secagg.masking import (
    MaskingSecaggConfigClient,
    MaskingSecaggConfigServer,
)
from declearn.secagg.utils import IdentityKeys

SEED = 0


def generate_toy_dataset(
    n_train: int = 300,
    n_valid: int = 150,
    n_clients: int = 3,
) -> List[Tuple[InMemoryDataset, InMemoryDataset]]:
    """Generate toy data for a centralized regression problem.

    Parameters
    ----------
    n_train:
        Total number of training samples.
    n_valid:
        Total number of validation samples.
    n_clients:
        Number of clients.

    Returns
    -------
    datasets:
        List of client-wise training and validation samples.
    """
    # Generate a toy classification dataset, with some clustered samples.
    n_samples = n_train + n_valid
    # false-positive; pylint: disable=unbalanced-tuple-unpacking
    inputs, target = sklearn.datasets.make_classification(
        n_samples,
        n_features=100,
        n_classes=2,
        n_informative=4,
        n_clusters_per_class=2,
        random_state=SEED,
    )
    inputs, target = inputs.astype("float32"), target.astype("int32")
    # Cluster samples based on features and assign them to clients thereof.
    # Also split client-wise data: 80% for training and 20% for validation.
    kclust = sklearn.cluster.KMeans(
        n_clusters=n_clients, init="random", n_init="auto", random_state=SEED
    ).fit_predict(inputs)
    datasets: List[Tuple[InMemoryDataset, InMemoryDataset]] = []
    for i in range(n_clients):
        arrays = sklearn.model_selection.train_test_split(
            inputs[kclust == i],
            target[kclust == i],
            test_size=0.20,
            random_state=SEED,
        )
        train = InMemoryDataset(arrays[0], arrays[2], expose_classes=True)
        valid = InMemoryDataset(arrays[1], arrays[3])
        datasets.append((train, valid))
    return datasets


async def async_run_server(
    folder: str,
    scaffold: bool,
    secagg: bool,
    n_clients: int = 3,
) -> None:
    """Server-side routine."""
    # Set up the FederatedServer.
    model = SklearnSGDModel.from_parameters(
        kind="classifier",
        loss="log_loss",
        penalty="none",
        dtype="float32",
    )
    netwk = NetworkServerConfig.from_params(
        protocol="websockets", host="127.0.0.1", port=8765, heartbeat=0.1
    )
    modules = ["l2-global-clipping"]
    optim = FLOptimConfig.from_params(
        aggregator="averaging",
        client_opt={
            "lrate": 0.1,
            "modules": ["scaffold-client", *modules] if scaffold else modules,
        },
        server_opt={
            "lrate": 1.0,
            "modules": ["scaffold-server"] if scaffold else None,
        },
    )
    secagg_config = (
        MaskingSecaggConfigServer(bitsize=64, clipval=1e8) if secagg else None
    )
    server = FederatedServer(
        model=model,
        netwk=netwk,
        optim=optim,
        metrics=["binary-classif"],
        secagg=secagg_config,
        checkpoint={"folder": folder, "max_history": 1},
    )
    # Set up hyper-parameters and run training.
    config = FLRunConfig.from_params(
        rounds=8,
        register={"min_clients": n_clients, "timeout": 2},
        training={"n_epoch": 1, "batch_size": 1, "drop_remainder": False},
        evaluate={"frequency": 8},  # only evaluate the last model
    )
    await server.async_run(config)


async def async_run_client(
    train: InMemoryDataset,
    valid: InMemoryDataset,
    id_keys: Optional[IdentityKeys],
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    netwk = NetworkClientConfig.from_params(
        protocol="websockets", server_uri="ws://localhost:8765", name="client"
    )
    secagg = MaskingSecaggConfigClient(id_keys=id_keys) if id_keys else None
    client = FederatedClient(
        netwk=netwk,
        train_data=train,
        valid_data=valid,
        secagg=secagg,
        verbose=False,
    )
    await client.async_run()


def setup_masking_idkeys(
    secagg: bool,
    n_clients: int,
) -> Union[List[IdentityKeys], List[None]]:
    """Setup identity keys for SecAgg, or a list of None values."""
    if not secagg:
        return [None for _ in range(n_clients)]
    prv_keys = [Ed25519PrivateKey.generate() for _ in range(n_clients)]
    pub_keys = [key.public_key() for key in prv_keys]
    return [IdentityKeys(key, trusted=pub_keys) for key in prv_keys]


async def run_declearn_experiment(
    scaffold: bool,
    secagg: bool,
    datasets: List[Tuple[InMemoryDataset, InMemoryDataset]],
) -> float:
    """Run a FL experiment using DecLearn, with opt. Scaffold and/or SecAgg.

    Train a linear model with log-loss, using simple SGD with global-l2-norm
    gradient clipping. Run 1 epoch per round, with single-sample batches.

    Parameters
    ----------
    scaffold:
        Whether to use Scaffold rather than vanilla FedAvg.
    secagg:
        Whether to use (masking-based) SecAgg.
    datasets:
        List of client-wise (training, validation) datasets.
        These must hold data to a binary classification task.

    Returns
    -------
    accuracy:
        Accuracy of the trained model (after 10 rounds) on the validation set.
    """
    # Set up the toy dataset(s) and optional identity keys (for SecAgg).
    n_clients = len(datasets)
    id_keys = setup_masking_idkeys(secagg=secagg, n_clients=n_clients)
    with tempfile.TemporaryDirectory() as folder:
        # Set up the server and client coroutines.
        coro_server = async_run_server(folder, scaffold, secagg, n_clients)
        coro_clients = [
            async_run_client(train=train, valid=valid, id_keys=id_keys[i])
            for i, (train, valid) in enumerate(datasets)
        ]
        # Run the coroutines concurrently using asyncio.
        output = await asyncio.gather(
            coro_server, *coro_clients, return_exceptions=True
        )
        # Assert that no exceptions occurred during the process.
        errors = "\n".join(repr(e) for e in output if isinstance(e, Exception))
        assert not errors, f"The FL process failed:\n{errors}"
        # Assert that the experiment ran properly.
        with open(
            os.path.join(folder, "metrics.json"), encoding="utf-8"
        ) as file:
            metrics = json.load(file)
        # Return the last validation accuracy reached.
        return list(metrics.values())[-1]["accuracy"]


@pytest.mark.parametrize("scaffold", [False, True], ids=["fedavg", "scaffold"])
@pytest.mark.asyncio
async def test_toy_classif_secagg(
    scaffold: bool,
) -> None:
    """Test that using SecAgg does not hurt convergence on a toy problem."""
    datasets = generate_toy_dataset(n_clients=3)
    acc_clrtxt = await run_declearn_experiment(
        scaffold=scaffold, secagg=False, datasets=datasets
    )
    acc_secagg = await run_declearn_experiment(
        scaffold=scaffold, secagg=True, datasets=datasets
    )
    assert acc_clrtxt >= 0.70
    assert abs(acc_clrtxt - acc_secagg) < 0.0001
