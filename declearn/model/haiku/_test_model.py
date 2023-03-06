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

from typing import Any, Dict, List, Tuple, Union
from datetime import datetime

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.haiku import HaikuModel
from declearn.model.sklearn import NumpyVector, SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.test_utils import run_as_processes
from declearn.utils import deserialize_object, get_logger, serialize_object

SEED = 1
R2_THRESHOLD = 0.999
RAND_SEQ = hk.PRNGSequence(jax.random.PRNGKey(SEED))

# pylint: disable protected-access

# chnages made :
# removed the list apstec of inputs
# removed the weighting for now
# changed the net output and the loss function (now the latter squeezes)

def net_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Simple linear regression model"""
    sgd = hk.Sequential([hk.Linear(1)])
    return sgd(x)

def loss_fn(y_pred: jnp.ndarray,y_true: jnp.ndarray)-> jnp.ndarray:
    """Per-sample mean square error loss"""
    y_pred = jnp.squeeze(y_pred)
    errors = (y_pred - y_true) if (y_true is not None) else y_pred
    return 0.5 * (errors)**2

class ToyModel:
    """A simple toy regression model"""

    def __init__(self, framework):
        self.framework = framework

    @property
    def model(self):
        if self.framework == "haiku":
            model = HaikuModel(net_fn, loss=loss_fn)
        else:
            raise ValueError("unrecognised framework")
        return model


def prep_datasets(
    clients: int = 3,
    n_train: int = 100,
    n_valid: int = 50,
    full: bool = False,
) -> Union[
    List[Tuple[InMemoryDataset, InMemoryDataset]],
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
]:
    """Generate and optionally split toy data for a regression problem.

    Parameters
    ----------
    clients: int, default=3
        Number of clients, i.e. of dataset shards to generate.
    n_train: int, default=30
        Number of training samples per client.
    n_valid: int, default=30
        Number of validation samples per client.
    full: bool, default=False
        Wether or not to split the dataset and encapsulate it in declearn
        InMemoryDataset objects

    Returns
    -------
    datasets: list[(InMemoryDataset, InMemoryDataset)]
        List of client-wise (train, valid) pair of datasets.
    """
    n_samples = clients * (n_train + n_valid)
    # false-positive; pylint: disable=unbalanced-tuple-unpacking
    inputs, target = make_regression(
        n_samples, n_features=20, n_informative=10, random_state=SEED
    )
    # Wrap up the data into client-wise pairs of dataset.
    if full:
        mid = n_train * clients
        out = (
            (inputs[:mid], target[:mid]),
            (inputs[mid:], target[mid:]),
        )
        return out
    else:
        out = []  # type: List[Tuple[InMemoryDataset, InMemoryDataset]]
        for idx in range(clients):
            start = (n_train + n_valid) * idx
            mid = start + n_train
            end = mid + n_valid
            train = InMemoryDataset(inputs[start:mid], target[start:mid])
            valid = InMemoryDataset(inputs[mid:end], target[mid:end])
            out.append((train, valid))
        return out


def test_declearn_experiment(
    framework: str,
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 10,
    clients: int = 3,
) -> None:
    """Run an experiment using declearn to perform a federative training.

    This function runs the experiment using declearn.
    It sets up and runs the server and client-wise routines in separate
    processes, to enable their concurrent execution.

    Parameters
    ----------
    framework: str
        The framework to be used to build the model
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
    # Set up a (func, args) tuple specifying the server process.
    p_server = (_server_routine, (framework, lrate, rounds, b_size, clients))
    # Set up the (func, args) tuples specifying client-wise processes.
    datasets = prep_datasets(clients)
    p_client = []
    for i, data in enumerate(datasets):
        client = (_client_routine, (data[0], data[1], f"client_{i}"))
        p_client.append(client)
    # Run each and every process in parallel.
    exitcodes = run_as_processes(p_server, *p_client)
    if not all(code == 0 for code in exitcodes):
        raise RuntimeError("The FL experiment failed.")


def _server_routine(
    framework: str,
    lrate: float = 0.01,
    rounds: int = 10,
    b_size: int = 10,
    clients: int = 3,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    # Set up the FederatedServer.
    model = ToyModel(framework).model
    netwk = NetworkServerConfig(
        protocol="websockets", host="localhost", port=8888
    )
    optim = FLOptimConfig.from_params(
        aggregator="averaging",
        client_opt={
            "lrate": lrate,
            # "regularizers": [("lasso", {"alpha": 0.1})],
        },
        server_opt=1.0,
    )
    server = FederatedServer(
        model, netwk, optim, metrics=["r2"]
    ) 
    # Set up hyper-parameters and run training.
    config = FLRunConfig.from_params(
        rounds=rounds,
        register={"min_clients": clients},
        training={"n_epoch": 1, "batch_size": b_size, "drop_remainder": False},
    )
    server.run(config)
    assert server.metrics.metrics[0].get_result()["r2"] > R2_THRESHOLD


def _client_routine(
    train: InMemoryDataset,
    valid: InMemoryDataset,
    name: str = "client",
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # Run the declearn FL client routine.
    netwk = NetworkClientConfig(
        protocol="websockets", server_uri="ws://localhost:8888", name=name
    )
    client = FederatedClient(netwk, train, valid)
    client.run()



def test_declearn_baseline(
    lrate: float = 0.01, rounds: int = 10, b_size: int = 3
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
    train, valid = prep_datasets(full=True)
    d_train = InMemoryDataset(train[0], train[1])
    # Set up a declearn model and a vanilla SGD optimizer.
    model = ToyModel("haiku").model
    model.initialize(
        {"input_shape": (1, d_train.data.shape[1]), "data_type": np.float32}
    )
    opt = Optimizer.from_config(
        {
            "lrate": lrate,
            # "regularizers": [("lasso", {"alpha": 0.1})],
        }
    )
    # Iteratively train the model, evaluating it after each epoch
    stime = datetime.now()
    for _ in range(rounds):
        # Run the training round.
        for batch in d_train.generate_batches(batch_size=b_size):
            grads = model.compute_batch_gradients(batch,max_norm=200.0)
            opt.apply_gradients(model, grads)
    etime = datetime.now()

    # Check the final R2 value
    
    params = jax.tree_util.tree_unflatten(
        model._params_treedef, model._params_leaves
    )
    y_pred = model._transformed_model.apply(params, next(RAND_SEQ), valid[0])
    print(f'Final R2 score is {r2_score(valid[1], y_pred)}')
    print(f'Training took {(etime-stime).total_seconds()} seconds')


if __name__ == "__main__":
    test_declearn_baseline()
