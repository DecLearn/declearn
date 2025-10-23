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

"""Integration test using fairness algorithms (and opt. SecAgg) on toy data.

* Set up a toy classification dataset with a sensitive attribute, and
  some client heterogeneity.
* Run a federated learning experiment...
"""

import asyncio
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from declearn.dataset.utils import split_multi_classif_dataset
from declearn.fairness.api import FairnessControllerServer
from declearn.fairness.core import FairnessInMemoryDataset
from declearn.fairness.fairbatch import FairbatchControllerServer
from declearn.fairness.fairfed import FairfedControllerServer
from declearn.fairness.fairgrad import FairgradControllerServer
from declearn.fairness.monitor import FairnessMonitorServer
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLRunConfig
from declearn.model.sklearn import SklearnSGDModel
from declearn.secagg.utils import IdentityKeys
from declearn.test_utils import (
    MockNetworkClient,
    MockNetworkServer,
    make_importable,
)

with make_importable(os.path.dirname(__file__)):
    from test_toy_clf_secagg import setup_masking_idkeys


SEED = 0


def generate_toy_dataset(
    n_train: int = 300,
    n_valid: int = 150,
    n_clients: int = 3,
) -> List[Tuple[FairnessInMemoryDataset, FairnessInMemoryDataset]]:
    """Generate datasets to a toy fairness-aware classification problem."""
    # Generate a toy classification dataset with a sensitive attribute.
    n_samples = n_train + n_valid
    inputs, s_attr, target = _generate_toy_data(n_samples)
    # Split samples uniformly across clients, with 80%/20% train/valid splits.
    shards = split_multi_classif_dataset(
        dataset=(np.concatenate([inputs, s_attr], axis=1), target.ravel()),
        n_shards=n_clients,
        scheme="iid",
        p_valid=0.2,
        seed=SEED,
    )
    # Wrap the resulting data as fairness in-memory datasets and return them.
    return [
        (
            FairnessInMemoryDataset(
                # fmt: off
                data=x_train[:, :-1],
                s_attr=x_train[:, -1:],
                target=y_train,
                expose_classes=True,
            ),
            FairnessInMemoryDataset(
                data=x_valid[:, :-1], s_attr=x_valid[:, -1:], target=y_valid
            ),
        )
        for (x_train, y_train), (x_valid, y_valid) in shards
    ]


def _generate_toy_data(
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a toy classification dataset with a binary sensitive attribute.

    - Draw random normal features X, random coefficients B and random noise N.
    - Compute L = XB + N, min-max normalize it into [0, 1] probabilities P.
    - Draw random binary sensitive attribute values S.
    - Define Y = 1{P >= 0.8}*1{S == 1} + 1{P >= 0.5}*1{S == 0}.

    Return X, S and Y matrices, as numpy arrays.
    """
    rng = np.random.default_rng(SEED)
    x_dat = rng.normal(size=(n_samples, 10), scale=10.0)
    s_dat = rng.choice(2, size=(n_samples, 1))
    theta = rng.normal(size=(10, 1), scale=5.0)
    noise = rng.normal(size=(n_samples, 1), scale=5.0)
    logit = np.matmul(x_dat, theta) + noise
    y_dat = (logit - logit.min()) / (logit.max() - logit.min())
    y_dat = (y_dat >= np.where(s_dat == 1, 0.8, 0.5)).astype("float32")
    return x_dat.astype("float32"), s_dat.astype("float32"), y_dat


async def server_routine(
    fairness: FairnessControllerServer,
    secagg: bool,
    folder: str,
    n_clients: int = 3,
) -> None:
    """Run the FL routine of the server."""
    # similar to SecAgg functional test; pylint: disable=duplicate-code
    model = SklearnSGDModel.from_parameters(
        kind="classifier",
        loss="log_loss",
        penalty="none",
        dtype="float32",
    )
    netwk = MockNetworkServer(
        host="localhost",
        port=8765,
        heartbeat=0.1,
    )
    optim = {
        "client_opt": 0.05,
        "server_opt": 1.0,
        "fairness": fairness,
    }
    server = FederatedServer(
        model,
        netwk=netwk,
        optim=optim,
        metrics=["binary-classif"],
        secagg={"secagg_type": "masking"} if secagg else None,
        checkpoint={"folder": folder, "max_history": 1},
    )
    config = FLRunConfig.from_params(
        rounds=5,
        register={"min_clients": n_clients, "timeout": 2},
        training={"n_epoch": 1, "batch_size": 10},
        evaluate={"frequency": 5},  # only evaluate the last model
        fairness={"batch_size": 50},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        await server.async_run(config)


async def client_routine(
    train_dst: FairnessInMemoryDataset,
    valid_dst: FairnessInMemoryDataset,
    id_keys: Optional[IdentityKeys],
) -> None:
    """Run the FL routine of a given client."""
    netwk = MockNetworkClient(
        server_uri="mock://localhost:8765",
        name="client",
    )
    secagg = (
        {"secagg_type": "masking", "id_keys": id_keys} if id_keys else None
    )
    client = FederatedClient(
        netwk=netwk,
        train_data=train_dst,
        valid_data=valid_dst,
        verbose=False,
        secagg=secagg,
    )
    await client.async_run()


@pytest.fixture(name="fairness")
def fairness_fixture(
    algorithm: str,
    f_type: str,
) -> FairnessControllerServer:
    """Server-side fairness controller providing fixture."""
    if algorithm == "fairbatch":
        return FairbatchControllerServer(f_type, alpha=0.005, fedfb=False)
    if algorithm == "fairfed":
        return FairfedControllerServer(f_type, beta=1.0, strict=True)
    if algorithm == "fairgrad":
        return FairgradControllerServer(f_type, eta=0.5, eps=1e-6)
    return FairnessMonitorServer(f_type)


@pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
@pytest.mark.parametrize("f_type", ["demographic_parity", "equalized_odds"])
@pytest.mark.parametrize(
    "algorithm", ["fairbatch", "fairfed", "fairgrad", "monitor"]
)
@pytest.mark.asyncio
async def test_toy_classif_fairness(
    fairness: FairnessControllerServer,
    secagg: bool,
    tmp_path: str,
) -> None:
    """Test a given fairness-aware federated learning algorithm on toy data.

    Set up a toy dataset for fairness-aware federated learning.
    Use a given algorithm, with a given group-fairness definition.
    Run training for 5 rounds. Optionally use SecAgg.

    When using mere monitoring, verify that hardcoded accuracy
    and (un)fairness levels, taken as a baseline, are achieved.
    When using another algorithm, verify that is achieves some
    degraded accuracy, and better fairness than the baseline.
    """
    # Set up the toy dataset and optional identity keys for SecAgg.
    datasets = generate_toy_dataset(n_clients=3)
    clients_id_keys = setup_masking_idkeys(secagg, n_clients=3)
    # Set up and run the fairness-aware federated learning experiment.
    coro_server = server_routine(fairness, secagg, folder=tmp_path)
    coro_clients = [
        client_routine(train_dst, valid_dst, id_keys)
        for (train_dst, valid_dst), id_keys in zip(
            datasets, clients_id_keys, strict=False
        )
    ]
    outputs = await asyncio.gather(
        coro_server, *coro_clients, return_exceptions=True
    )
    # Assert that no exceptions occurred during the process.
    errors = "\n".join(repr(e) for e in outputs if isinstance(e, Exception))
    assert not errors, f"The FL process failed:\n{errors}"
    # Load and parse utility and fairness metrics at the final round.
    u_metrics = pd.read_csv(os.path.join(tmp_path, "metrics.csv"))
    f_metrics = pd.read_csv(os.path.join(tmp_path, "fairness_metrics.csv"))
    accuracy = u_metrics.iloc[-1]["accuracy"]
    fairness_cols = [f"{fairness.f_type}_{group}" for group in fairness.groups]
    fairness_mean_abs = f_metrics.iloc[-1][fairness_cols].abs().mean()
    # Verify that the FedAvg baseline matches expected accuracy and fairness,
    # or that other algorithms achieve lower accuracy and better fairness.
    # Note that FairFed is bound to match the FedAvg baseline due to the
    # split across clients being uniform.
    expected_fairness = {
        "demographic_parity": 0.025,
        "equalized_odds": 0.142,
    }
    if fairness.algorithm == "monitor":
        assert accuracy >= 0.76
        assert fairness_mean_abs > expected_fairness[fairness.f_type]
    elif fairness.algorithm == "fairfed":
        assert accuracy >= 0.72
        assert fairness_mean_abs > expected_fairness[fairness.f_type]
    else:
        assert 0.76 > accuracy > 0.54
        assert fairness_mean_abs < expected_fairness[fairness.f_type]
