# Quickstart

This section provides with demonstration code on how to run a simple federated
learning task using declearn, that requires minimal adjustments to be run for
real (mainly, to provide with a valid network configuration and actual data).

You may find even more concrete examples on our gitlab repository
[here](https://gitlab.inria.fr/magnet/declearn/declearn2/examples).
The Heart UCI example may notably be run as-is, either locally or on a
real-life network with minimal command-line parametrization.

## Setting

Here is a quickstart example on how to set up a federated learning process
to learn a LASSO logistic regression model (using a scikit-learn backend)
using pre-processed data, formatted as csv files with a "label" column,
where each client has two files: one for training, the other for validation.

Here, the code uses:

- standard FedAvg strategy (SGD for local steps, averaging of updates weighted
  by clients' training dataset size, no modifications of server-side updates)
- 10 rounds of training, with 5 local epochs performed at each round and
  128-samples batch size
- at least 1 and at most 3 clients, awaited for 180 seconds by the server
- network communications using gRPC, on host "example.com" and port 8888

Note that this example code may easily be adjusted to suit use cases, using
other types of models, alternative federated learning algorithms and/or
modifying the communication, training and validation hyper-parameters.
Please refer to the [Hands-on usage](./user-guide/usage.md) section for a more
detailed and general description of how to set up a federated learning
task and process with declearn.

## Server-side script

```python
import declearn

model = declearn.model.sklearn.SklearnSGDModel.from_parameters(
    kind="classifier", loss="log_loss", penalty="l1"
)
netwk = declearn.communication.NetworkServerConfig(
    protocol="grpc", host="example.com", port=8888,
    certificate="path/to/certificate.pem",
    private_key="path/to/private_key.pem"
)
optim = declearn.main.config.FLOptimConfig.from_params(
    aggregator="averaging",
    client_opt=0.001,
)
server = declearn.main.FederatedServer(
    model, netwk, optim, checkpoint="outputs"
)
config = declearn.main.config.FLRunConfig.from_params(
    rounds=10,
    register={"min_clients": 1, "max_clients": 3, "timeout": 180},
    training={"n_epoch": 5, "batch_size": 128, "drop_remainder": False},
)
server.run(config)
```

## Client-side script

```python
import declearn

netwk = declearn.communication.NetworkClientConfig(
    protocol="grpc",
    server_uri="example.com:8888",
    name="client_name",
    certificate="path/to/root_ca.pem"
)
train = declearn.dataset.InMemoryDataset(
    "path/to/train.csv", target="label",
    expose_classes=True  # enable sharing of unique target values
)
valid = declearn.dataset.InMemoryDataset("path/to/valid.csv", target="label")
client = declearn.main.FederatedClient(
    netwk, train, valid, checkpoint="outputs"
)
client.run()
```

## Simulating this experiment locally

To simulate the previous experiment on a single computer, you may set up
network communications to go through the localhost, and resort to one of
two possibilities:
- 1. Run the server and client-wise scripts parallelly, e.g. in distinct
     terminals.
- 2. Use declearn-provided tools to run the server and clients' routines
     concurrently using multiprocessing.

While technically similar (both solutions resolve on isolating the agents'
routines in separate python processes that communicate over the localhost),
the second solution offers more practicality in terms of offering a single
entrypoint for your experiment, and optionally automatically stopping any
running agent in case one of the other has failed.
To find out more about this solution, please have a look at the Heart UCI
example [implemented here](https://gitlab.inria.fr/magnet/declearn/declearn2/examples/heart-uci/readme.md).
