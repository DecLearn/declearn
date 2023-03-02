# coding: utf-8

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

import json
import tempfile
from typing import List, Tuple

import numpy as np
import tensorflow as tf  # type: ignore
import torch
from sklearn.datasets import make_regression  # type: ignore
from sklearn.linear_model import SGDRegressor  # type: ignore

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.metrics import RSquared
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.model.tensorflow import TensorflowModel
from declearn.model.torch import TorchModel
from declearn.optimizer import Optimizer
from declearn.test_utils import FrameworkType, run_as_processes

SEED = 0
R2_THRESHOLD = 0.999

# pylint: disable=too-many-function-args


def get_model(framework: FrameworkType) -> Model:
    """Set up a simple toy regression model."""
    if framework == "numpy":
        np.random.seed(SEED)  # set seed
        model = SklearnSGDModel.from_parameters(
            kind="regressor", loss="squared_error", penalty="none"
        )  # type: Model
    elif framework == "tensorflow":
        tf.random.set_seed(SEED)  # set seed
        tfmod = tf.keras.Sequential(tf.keras.layers.Dense(units=1))
        tfmod.build([None, 100])
        model = TensorflowModel(tfmod, loss="mean_squared_error")
    elif framework == "torch":
        torch.manual_seed(SEED)  # set seed
        torchmod = torch.nn.Sequential(
            torch.nn.Linear(100, 1, bias=True),
            torch.nn.Flatten(0),
        )
        model = TorchModel(torchmod, loss=torch.nn.MSELoss())
    else:
        raise ValueError("unrecognised framework")
    return model


def prep_client_datasets(
    clients: int = 3,
    n_train: int = 100,
    n_valid: int = 50,
) -> List[Tuple[InMemoryDataset, InMemoryDataset]]:
    """Generate and split toy data for a regression problem.

    Parameters
    ----------
    clients: int, default=3
        Number of clients, i.e. of dataset shards to generate.
    n_train: int, default=30
        Number of training samples per client.
    n_valid: int, default=30
        Number of validation samples per client.

    Returns
    -------
    datasets: list[(InMemoryDataset, InMemoryDataset)]
        List of client-wise (train, valid) pair of datasets.
    """
    n_samples = clients * (n_train + n_valid)
    # false-positive; pylint: disable=unbalanced-tuple-unpacking
    inputs, target = make_regression(
        n_samples, n_features=100, n_informative=10, random_state=SEED
    )
    inputs, target = inputs.astype("float32"), target.astype("float32")
    # Wrap up the data into client-wise pairs of dataset.
    out = []  # type: List[Tuple[InMemoryDataset, InMemoryDataset]]
    for idx in range(clients):
        start = (n_train + n_valid) * idx
        mid = start + n_train
        end = mid + n_valid
        train = InMemoryDataset(inputs[start:mid], target[start:mid])
        valid = InMemoryDataset(inputs[mid:end], target[mid:end])
        out.append((train, valid))
    return out


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
    inputs, target = make_regression(
        n_samples, n_features=100, n_informative=10, random_state=SEED
    )
    inputs, target = inputs.astype("float32"), target.astype("float32")
    # Wrap up the data into client-wise pairs of dataset
    out = (
        (inputs[:n_train], target[:n_train]),
        (inputs[n_train:], target[n_train:]),
    )
    return out


def test_declearn_experiment(
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
    # pylint: disable=too-many-locals
    with tempfile.TemporaryDirectory() as folder:
        # Set up a (func, args) tuple specifying the server process.
        p_server = (
            _server_routine,
            (folder, framework, lrate, rounds, b_size, clients),
        )
        # Set up the (func, args) tuples specifying client-wise processes.
        datasets = prep_client_datasets(clients)
        p_client = []
        for i, data in enumerate(datasets):
            client = (_client_routine, (data[0], data[1], f"client_{i}"))
            p_client.append(client)
        # Run each and every process in parallel.
        exitcodes = run_as_processes(p_server, *p_client)
        if not all(code == 0 for code in exitcodes):
            raise RuntimeError("The FL experiment failed.")
        # Assert convergence
        with open(f"{folder}/metrics.json", encoding="utf-8") as file:
            r2_dict = json.load(file)
            last_r2_dict = r2_dict.get(max(r2_dict.keys()))
            final_r2 = float(last_r2_dict.get("r2"))
            assert final_r2 > R2_THRESHOLD


def _server_routine(
    folder: str,
    framework: FrameworkType,
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 10,
    clients: int = 3,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    # pylint: disable=too-many-arguments
    # Set up the FederatedServer.
    model = get_model(framework)
    netwk = NetworkServerConfig("websockets", "127.0.0.1", 8765)
    optim = FLOptimConfig.from_params(
        aggregator="averaging",
        client_opt={
            "lrate": lrate,
            "regularizers": [("lasso", {"alpha": 0.1})],
        },
        server_opt=1.0,
    )

    server = FederatedServer(
        model,
        netwk,
        optim,
        metrics=["r2"],
        checkpoint=folder,
    )
    # Set up hyper-parameters and run training.
    config = FLRunConfig.from_params(
        rounds=rounds,
        register={"min_clients": clients},
        training={
            "n_epoch": 1,
            "batch_size": b_size,
            "drop_remainder": False,
        },
    )
    server.run(config)


def _client_routine(
    train: InMemoryDataset,
    valid: InMemoryDataset,
    name: str = "client",
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # Run the declearn FL client routine.
    netwk = NetworkClientConfig("websockets", "ws://localhost:8765", name)
    client = FederatedClient(netwk, train, valid)
    client.run()


def test_declearn_baseline(
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 1,
) -> None:
    """Run a baseline uing declearn APIs to emulate a centralized setting.

    This function uses declearn but sets up a single model and performs
    training on the concatenation of "client-wise" datasets.

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
    d_train = InMemoryDataset(train[0], train[1])
    # Set up a declearn model and a vanilla SGD optimizer.
    model = get_model("numpy")
    model.initialize({"n_features": d_train.data.shape[1]})
    opt = Optimizer(lrate=lrate, regularizers=[("lasso", {"alpha": 0.1})])
    # Iteratively train the model, evaluating it after each epoch.
    for _ in range(rounds):
        # Run the training round.
        for batch in d_train.generate_batches(batch_size=b_size):
            grads = model.compute_batch_gradients(batch)
            opt.apply_gradients(model, grads)
    # Check the final R2 value.
    r_sq = RSquared()
    r_sq.update(*model.compute_batch_predictions((valid[0], valid[1], None)))
    assert r_sq.get_result()["r2"] > R2_THRESHOLD


def test_sklearn_baseline(
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 1,
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
    sgd = SGDRegressor(
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
