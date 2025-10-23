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

"""Script to run convergence tests on a toy regression problem.

This script sets up a toy problem, and three experiments based on it.

* Problem description:
  - Toy dataset, generated using (seeded) `make_regression` from scikit-learn.
  - Dimensions: 100 features, 10 of which are informative, and 1 target.
  - For each client: 100 training samples, 50 validation ones.

* Centralized baselines:
  - Concatenate the client-wise datasets into a single (train, valid) pair.
  - Either use declearn or raw scikit-learn to perform training:
    - `run_declearn_baseline`:
      Use the declearn Model and Optimizer APIs to train the model.
    - `run_sklearn_baseline`:
      Use the built-in training routine of scikit-learn's SGDRegressor.

* Federated experiments:
  - Split the dataset into client-wise (train, valid) dataset pairs.
  - Use declearn to perform federated training:
    - `run_declearn_experiment`:
      Use multiprocessing to run an actual FL process.

The convergence results of those experiments is then compared.

"""

import asyncio
import dataclasses
import json
import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest
import sklearn.datasets  # type: ignore
import sklearn.linear_model  # type: ignore

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import Dataset, InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.metrics import RSquared
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.test_utils import FrameworkType
from declearn.utils import set_device_policy

# optional frameworks' dependencies pylint: disable=ungrouped-imports
# pylint: disable=duplicate-code

# tensorflow imports
try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pass
else:
    import tensorflow.keras as tf_keras  # type: ignore

    from declearn.dataset.tensorflow import TensorflowDataset
    from declearn.model.tensorflow import TensorflowModel, TensorflowVector
# torch imports
try:
    import torch
except ModuleNotFoundError:
    pass
else:
    from declearn.dataset.torch import TorchDataset
    from declearn.model.torch import TorchModel, TorchVector
# haiku and jax imports
try:
    import haiku as hk
    import jax
except ModuleNotFoundError:
    pass
else:
    from declearn.model.haiku import HaikuModel, JaxNumpyVector

    def haiku_model_fn(inputs: jax.Array) -> jax.Array:
        """Simple linear model implemented with Haiku."""
        return hk.Linear(1)(inputs)

    def haiku_loss_fn(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
        """Sample-wise squared error loss function."""
        y_pred = jax.numpy.squeeze(y_pred)
        return (y_pred - y_true) ** 2


# pylint: enable=duplicate-code, ungrouped-imports


SEED = 0
R2_THRESHOLD = 0.9999


def get_model(framework: FrameworkType) -> Model:
    """Set up a simple toy regression model, with zero-valued weights."""
    set_device_policy(gpu=False)  # disable GPU use to avoid concurrence
    if framework == "numpy":
        return _get_model_numpy()
    if framework == "tensorflow":
        return _get_model_tflow()
    if framework == "torch":
        return _get_model_torch()
    if framework == "jax":
        return _get_model_haiku()
    raise ValueError(f"Unrecognised model framework: '{framework}'.")


def _get_model_numpy() -> Model:
    """Return a linear model with MSE loss in Sklearn, with zero weights."""
    np.random.seed(SEED)  # set seed
    model = SklearnSGDModel.from_parameters(
        kind="regressor", loss="squared_error", penalty="none"
    )
    return model


def _get_model_tflow() -> Model:
    """Return a linear model with MSE loss in TensorFlow, with zero weights."""
    tf.random.set_seed(SEED)  # set seed
    tfmod = tf_keras.Sequential([tf_keras.layers.Dense(units=1)])
    tfmod.build([None, 100])
    model = TensorflowModel(tfmod, loss="mean_squared_error")
    with tf.device("CPU"):
        zeros = {
            key: tf.zeros_like(val)
            for key, val in model.get_weights().coefs.items()
        }
    model.set_weights(TensorflowVector(zeros))
    return model


def _get_model_torch() -> Model:
    """Return a linear model with MSE loss in Torch, with zero weights."""
    torch.manual_seed(SEED)  # set seed
    torchmod = torch.nn.Sequential(
        torch.nn.Linear(100, 1, bias=True),
        torch.nn.Flatten(0),
    )
    model = TorchModel(torchmod, loss=torch.nn.MSELoss())
    zeros = {
        key: torch.zeros_like(val)
        for key, val in model.get_weights().coefs.items()
    }
    model.set_weights(TorchVector(zeros))
    return model


def _get_model_haiku() -> Model:
    """Return a linear model with MSE loss in Haiku, with zero weights."""
    model = HaikuModel(haiku_model_fn, loss=haiku_loss_fn)
    model.initialize({"data_type": "float32", "features_shape": (100,)})
    zeros_like = jax.jit(jax.numpy.zeros_like, backend="cpu")
    zeros = {
        key: zeros_like(val)  # pylint: disable=not-callable
        for key, val in model.get_weights().coefs.items()
    }
    model.set_weights(JaxNumpyVector(zeros))
    return model


def get_dataset(
    framework: FrameworkType,
    inputs: np.ndarray,
    labels: np.ndarray,
) -> Dataset:
    """Return a framework-appropriate dataset"""
    if framework == "torch":
        pt_inp = torch.from_numpy(inputs)
        pt_lab = torch.from_numpy(labels)
        return TorchDataset(torch.utils.data.TensorDataset(pt_inp, pt_lab))
    if framework == "tensorflow":
        with tf.device("CPU"):
            tf_inp = tf.convert_to_tensor(inputs)
            tf_lab = tf.convert_to_tensor(labels)
            tf_dst = tf.data.Dataset.from_tensor_slices((tf_inp, tf_lab))
        return TensorflowDataset(tf_dst)
    return InMemoryDataset(inputs, labels, expose_data_type=True)


def prep_full_dataset(
    n_train: int = 300,
    n_valid: int = 150,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate toy data for a centralized regression problem.

    Parameters
    ----------
    n_train: int, default=30
        Number of training samples per client.
    n_valid: int, default=30
        Number of validation samples per client.

    Returns
    -------
    datasets: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
        np.ndarray]]
        Tuple of ((train_data,train_label), (valid_data,vlaid_target)).
    """
    n_samples = n_train + n_valid
    # false-positive; pylint: disable=unbalanced-tuple-unpacking
    inputs, target = sklearn.datasets.make_regression(
        n_samples, n_features=100, n_informative=10, random_state=SEED
    )
    inputs, target = inputs.astype("float32"), target.astype("float32")
    # Wrap up the data into client-wise pairs of dataset
    out = (
        (inputs[:n_train], target[:n_train]),
        (inputs[n_train:], target[n_train:]),
    )
    return out


def test_sklearn_baseline(
    lrate: float = 0.04,
    rounds: int = 10,
    b_size: int = 10,
) -> None:
    """Run a baseline using scikit-learn to emulate a centralized setting.

    This function does not use declearn. It sets up a single sklearn
    model and performs training on the full dataset.

    Parameters
    ----------
    lrate: float, default=0.01
        Learning rate of the SGD algorithm.
    rounds: int, default=10
        Number of training rounds to perform, i.e. number of epochs.
    b_size: int, default=10
        Batch size fot the training (and validation) data.
        Batching will be performed without shuffling nor replacement,
        and the final batch may be smaller than the others (no drop).
    """
    # Generate the client datasets, then centralize them into numpy arrays.
    train, valid = prep_full_dataset()
    # Set up a scikit-learn model, implementing step-wise gradient descent.
    sgd = sklearn.linear_model.SGDRegressor(
        loss="squared_error",
        penalty="l1",
        alpha=0.1,
        eta0=lrate / b_size,  # adjust learning rate for (dropped) batch size
        learning_rate="constant",  # disable scaling, unused in declearn
        max_iter=rounds,
    )
    # Iteratively train the model, evaluating it after each epoch.
    for _ in range(rounds):
        sgd.partial_fit(train[0], train[1])
    assert sgd.score(valid[0], valid[1]) > R2_THRESHOLD


def test_declearn_baseline(
    framework: FrameworkType,
    lrate: float = 0.02,
    rounds: int = 10,
    b_size: int = 10,
) -> None:
    """Run a baseline uing declearn APIs to emulate a centralized setting.

    This function uses declearn but sets up a single model and performs
    training on the entire toy regression dataset.

    Parameters
    ----------
    framework: str
        Framework of the model to train and evaluate.
    lrate: float, default=0.02
        Learning rate of the SGD algorithm.
    rounds: int, default=10
        Number of training rounds to perform, i.e. number of epochs.
    b_size: int, default=10
        Batch size fot the training (and validation) data.
        Batching will be performed without shuffling nor replacement,
        and the final batch may be smaller than the others (no drop).
    """
    # Generate the client datasets, then centralize them into numpy arrays.
    train, valid = prep_full_dataset()
    dst_train = get_dataset(framework, *train)
    # Set up a declearn model and a SGD optimizer with Lasso regularization.
    model = get_model(framework)
    model.initialize(dataclasses.asdict(dst_train.get_data_specs()))
    optim = Optimizer(
        lrate=lrate if framework != "numpy" else (lrate * 2),
        regularizers=[("lasso", {"alpha": 0.1})],
    )
    # Iteratively train the model and evaluate it between rounds.
    r_sq = RSquared()
    scores: List[float] = []
    for _ in range(rounds):
        for batch in dst_train.generate_batches(
            batch_size=b_size, drop_remainder=False
        ):
            optim.run_train_step(model, batch)
        pred = model.compute_batch_predictions((*valid, None))
        r_sq.reset()
        r_sq.update(*pred)
        scores.append(r_sq.get_result()["r2"])  # type: ignore
    # Check that the R2 increased through epochs to reach a high value.
    print(scores)
    assert all(scores[i + 1] >= scores[i] for i in range(rounds - 1))
    assert scores[-1] >= R2_THRESHOLD


def prep_client_datasets(
    framework: FrameworkType,
    clients: int = 3,
    n_train: int = 100,
    n_valid: int = 50,
) -> List[Tuple[Dataset, Dataset]]:
    """Generate and split data for a toy sparse regression problem.

    Parameters
    ----------
    framework:
        Name of the framework being tested, based on which the Dataset
        class choice may be adjusted as well.
    clients:
        Number of clients, i.e. of dataset shards to generate.
    n_train:
        Number of training samples per client.
    n_valid:
        Number of validation samples per client.

    Returns
    -------
    datasets:
        List of client-wise tuple of (train, valid) Dataset instances.
    """
    train, valid = prep_full_dataset(
        n_train=clients * n_train,
        n_valid=clients * n_valid,
    )
    # Wrap up the data into client-wise pairs of dataset.
    out: List[Tuple[Dataset, Dataset]] = []
    for idx in range(clients):
        # Gather the client's training dataset.
        srt = n_train * idx
        end = n_train + srt
        dst_train = get_dataset(
            framework=framework,
            inputs=train[0][srt:end],
            labels=train[1][srt:end],
        )
        # Gather the client's validation dataset.
        srt = n_valid * idx
        end = n_valid + srt
        dst_valid = get_dataset(
            framework=framework,
            inputs=valid[0][srt:end],
            labels=valid[1][srt:end],
        )
        # Store both datasets into the output list.
        out.append((dst_train, dst_valid))
    return out


async def async_run_server(  # noqa: PLR0913
    folder: str,
    framework: FrameworkType,
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 10,
    clients: int = 3,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # Set up the FederatedServer.
    model = get_model(framework)
    netwk = NetworkServerConfig.from_params(
        protocol="websockets", host="127.0.0.1", port=8765, heartbeat=0.1
    )
    optim = FLOptimConfig.from_params(
        aggregator="averaging",
        client_opt={
            "lrate": lrate if framework != "numpy" else (lrate * 2),
            "regularizers": [("lasso", {"alpha": 0.1})],
        },
        server_opt=1.0,
    )
    server = FederatedServer(
        model,
        netwk,
        optim,
        metrics=["r2"],
        checkpoint={"folder": folder, "max_history": 1},
    )
    # Set up hyper-parameters and run training.
    config = FLRunConfig.from_params(
        rounds=rounds,
        register={"min_clients": clients, "timeout": 10},
        training={
            "n_epoch": 1,
            "batch_size": b_size,
            "drop_remainder": False,
        },
        evaluate={"frequency": 10},  # only evaluate the last model
    )
    await server.async_run(config)


async def async_run_client(
    train: Dataset,
    valid: Dataset,
    name: str = "client",
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    netwk = NetworkClientConfig.from_params(
        protocol="websockets", server_uri="ws://localhost:8765", name=name
    )
    client = FederatedClient(netwk, train, valid)
    await client.async_run()


# similar structure in other functional tests; pylint: disable=duplicate-code


@pytest.mark.asyncio
async def test_declearn_federated(
    framework: FrameworkType,
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 1,
    clients: int = 3,
) -> None:
    """Run an experiment using declearn to perform a federative training.

    This function runs the experiment using declearn.
    It sets up and runs the server and client-wise routines in separate
    processes, to enable their concurrent execution.

    Parameters
    ----------
    framework: str
        Framework of the model to train and evaluate.
    lrate: float, default=0.01
        Learning rate of the SGD algorithm performed by clients.
    rounds: int, default=10
        Number of FL training rounds to perform.
        At each round, each client will perform a full epoch of training.
    b_size: int, default=10
        Batch size fot the training (and validation) data.
        Batching will be performed without shuffling nor replacement,
        and the final batch may be smaller than the others (no drop).
    clients: int, default=3
        Number of federated clients to set up and run.
    """
    datasets = prep_client_datasets(framework, clients)
    with tempfile.TemporaryDirectory() as folder:
        # Set up the server and client coroutines.
        coro_server = async_run_server(
            folder, framework, lrate, rounds, b_size, clients
        )
        coro_clients = [
            async_run_client(train, valid, name=f"client_{i}")
            for i, (train, valid) in enumerate(datasets)
        ]
        # Run the coroutines concurrently using asyncio.
        outputs = await asyncio.gather(
            coro_server, *coro_clients, return_exceptions=True
        )
        # Assert that no exceptions occurred during the process.
        errors = "\n".join(
            repr(exc) for exc in outputs if isinstance(exc, Exception)
        )
        assert not errors, f"The FL process failed:\n{errors}"
        # Assert that the federated model converged above an expected value.
        with open(
            os.path.join(folder, "metrics.json"), encoding="utf-8"
        ) as file:
            metrics = json.load(file)
        best_r2 = max(values["r2"] for values in metrics.values())
        assert best_r2 >= R2_THRESHOLD, "The FL training did not converge"
