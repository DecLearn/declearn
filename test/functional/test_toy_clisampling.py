"""
Integration test using client sampling (= ClientSampler subclasses)
on a toy problem.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, List, Tuple

import pytest

from declearn.client_sampler import ClientSampler, CompositionClientSampler
from declearn.client_sampler.modules import (
    CriterionClientSampler,
    DefaultClientSampler,
    GradientNormCriterion,
    UniformClientSampler,
)
from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.main.utils import IncompatibleConfigsError
from declearn.model.sklearn import SklearnSGDModel
from declearn.secagg.masking import MaskingSecaggConfigServer
from declearn.test_utils import make_importable
from declearn.utils import get_logger

with make_importable(os.path.dirname(__file__)):
    from test_toy_clf_secagg import generate_toy_dataset

#### Experiment functions ####


async def async_run_server(
    folder: str,
    client_sampler: ClientSampler,
    n_clients: int = 3,
    secagg: bool = False,
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
            "modules": modules,
        },
        server_opt={
            "lrate": 1.0,
            "modules": None,
        },
    )
    secagg_config = (
        MaskingSecaggConfigServer(bitsize=64, clipval=1e8) if secagg else None
    )
    logger = get_logger("logger", logging.DEBUG)
    server = FederatedServer(
        model=model,
        netwk=netwk,
        optim=optim,
        metrics=["binary-classif"],
        client_sampler=client_sampler,
        secagg=secagg_config,
        checkpoint={"folder": folder, "max_history": 1},
        logger=logger,
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
    name: str,
    train: InMemoryDataset,
    valid: InMemoryDataset,
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    netwk = NetworkClientConfig.from_params(
        protocol="websockets",
        server_uri="ws://localhost:8765",
        name=name,
    )
    client = FederatedClient(
        netwk=netwk,
        train_data=train,
        valid_data=valid,
        verbose=False,
    )
    await client.async_run()


async def run_declearn_experiment(
    client_sampler: ClientSampler,
    datasets: List[Tuple[InMemoryDataset, InMemoryDataset]],
    secagg: bool = False,
) -> float:
    """Run a FL experiment using DecLearn, with opt. Scaffold and/or SecAgg.

    Train a linear model with log-loss, using simple SGD with global-l2-norm
    gradient clipping. Run 1 epoch per round, with single-sample batches.

    Parameters
    ----------
    client_sampler:
        Client sampler to use
    datasets:
        List of client-wise (training, validation) datasets.
        These must hold data to a binary classification task.
    secagg:
        If secagg is enabled or not
    Returns
    -------
    accuracy:
        Accuracy of the trained model (after 10 rounds) on the validation set.
    """
    # Set up the toy dataset(s)
    n_clients = len(datasets)
    with tempfile.TemporaryDirectory() as folder:
        # Set up the server and client coroutines.
        coro_server = async_run_server(
            folder,
            client_sampler,
            n_clients,
            secagg,
        )
        coro_clients = [
            async_run_client(name=f"client_{i}", train=train, valid=valid)
            for i, (train, valid) in enumerate(datasets)
        ]
        # Run the coroutines concurrently using asyncio.
        output = await asyncio.gather(
            coro_server, *coro_clients, return_exceptions=False
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


#### TESTS ####

# client samplers involved in the integration tests
CLIENT_SAMPLERS: Dict[str, ClientSampler] = {
    "Default": DefaultClientSampler(),
    "Uniform": UniformClientSampler(n_samples=2),
    "Criterion": CriterionClientSampler(
        n_samples=2, criterion=GradientNormCriterion()
    ),
    "Composition": CompositionClientSampler(
        CriterionClientSampler(n_samples=1, criterion=GradientNormCriterion()),
        UniformClientSampler(n_samples=1),
    ),
}
# TODO add tests with a config and a dict when supported, /!\ update typing


@pytest.mark.parametrize(
    "client_sampler",
    list(CLIENT_SAMPLERS.values()),
    ids=list(CLIENT_SAMPLERS.keys()),
)
@pytest.mark.asyncio
async def test_toy_classif_client_sampling(
    client_sampler: ClientSampler,
) -> None:
    """Test that client sampling is correctly integrated for all
    client sampler subclasses, and that the resulting accuracy for a
    toy problem is not too much impacted
    """
    datasets = generate_toy_dataset(n_clients=3)
    acc = await run_declearn_experiment(
        client_sampler=client_sampler,
        datasets=datasets,
        secagg=False,
    )
    acc_threshold = 0.3
    assert acc is not None and acc > acc_threshold


@pytest.mark.asyncio
async def test_clisamp_secagg_incompatibility() -> None:
    """Test that enabling secagg with a secagg-incompatible client sampler
    will raise an exception
    """
    datasets = generate_toy_dataset(n_clients=3)
    client_sampler = CriterionClientSampler(
        n_samples=2, criterion=GradientNormCriterion()
    )
    with pytest.raises(IncompatibleConfigsError):
        await run_declearn_experiment(
            client_sampler=client_sampler,
            datasets=datasets,
            secagg=True,
        )
