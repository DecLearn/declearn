# Quickstart

**Here's where to start if you want to quickly understand what `declearn`
does**. This tutorial expects a basic understanding of
[federated learning](https://en.wikipedia.org/wiki/Federated_learning).

We show different ways to use `declearn` on a well-known example, the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)
(see [section 1](#1-federated-learning-on-the-mnist-dataset)).
We then look at how to use declearn on your own problem
(see [section 2](#2-federated-learning-on-your-own-dataset)).

## 1. Federated learning on the MNIST dataset

**We are going to train a common model between three simulated clients on the
classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/)**. The input of the
model is a set of images of handwritten digits, and the model needs to
determine to which number between $0$ and $9$ each image corresponds.
We show two ways to use `declearn` on this problem.

### 1.1. Quickrun mode

**The quickrun mode is the simplest way to simulate a federated learning
process on a single machine with `declearn`**. It does not require to
understand the details of the `declearn` implementation. It requires a basic
understanding of federated learning.

---
**To test this on the MNIST example**, you can follow along the jupyter
notebook provided
[here](https://gitlab.inria.fr/magnet/declearn/declearn2/-/blob/develop/examples/mnist_quickrun/mnist.ipynb),
which we recommend running on [Google Colab](https://colab.research.google.com)
to skip on setting up git, python, a virtual environment, etc.

You may find a (possibly not entirely up-to-date) pre-hosted version of that
notebook
[here](https://colab.research.google.com/drive/13sBDOQeorI6dfziSoyRpU4q4iGuESIPo?usp=sharing).

---

**If you want to run this locally**, the detailed notebook can be boiled down
to five shell commands. Set up a dedicated `conda` or `venv` environment, and
run:

```bash
git clone https://gitlab.inria.fr/magnet/declearn/declearn2 &&
cd declearn2 &&
pip install .[tensorflow,websockets] &&
declearn-split --folder "examples/mnist_quickrun" &&
declearn-quickrun --config "examples/mnist_quickrun/config.toml"
```

**To better understand the details** of what happens under the hood you can
look at what the key element of the declearn process are in
[section 1.2](#12-python-script). To understand how to use the quickrun mode
in practice, see [section 2.1](#21-quickrun-on-your-problem).

### 1.2. Python script

#### MNIST

**The quickrun mode abstracts away a lot of important elements** of the
process, and is only designed to simulate an FL experiment: the clients all
run on the same machine. In real life deployment, a `declearn` experiment is
built in python.

---
**To see what this looks like in practice**, you can head to the all-python
MNIST example `examples/mnist/` in the `declearn` repository, which you can
access [here](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/develop/examples/mnist/).

This version of the example may either be used to run a simulated process on
a single computer, or to deploy the example over a real-life network.

---

#### Stylized structure

At a very high-level, declearn is structured around two key objects. The
`Clients` hold the data and perform calculations locally. The `Server` owns
the model and the global training process. They communicate over a `network`,
the central endpoint of which is hosted by the `Server`.

We provide below a stylized view of the main elements of the `Server` and
`Client` scripts. For more details, you can look at the hands-on usage
[section](user-guide/usage.md) of the documentation.

We show what a `Client` and `Server` script can look like on a hypothetical
LASSO logistic regression model, using a scikit-learn backend and
pre-processed data. The data is in csv files with a "label" column,
where each client has two files: one for training, the other for validation.

Here, the code uses:

* **Aggregation**: the standard `FedAvg` strategy.
* **Optimizer**: standard SGD for both client and server.
* **Training**:  10 rounds of training, with 5 local epochs performed at each
  round and 128-samples batch size. At least 1 and at most 3 clients, awaited
  for at most 180 seconds by the server.
* **Network**: communications using `websockets`.

The server-side script:

```python
import declearn

model = declearn.model.sklearn.SklearnSGDModel.from_parameters(
    kind="classifier", loss="log_loss", penalty="l1"
)
netwk = declearn.communication.NetworkServerConfig(
    protocol="websockets", host="127.0.0.1", port=8888,
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

The client-side script

```python
import declearn

netwk = declearn.communication.NetworkClientConfig(
    protocol="websockets",
    server_uri="127.0.0.1:8888",
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

## 2. Federated learning on your own dataset

### 2.1. Quickrun on your problem

Using the mode `declearn-quickrun` requires a configuration file, some data,
and a model file:

* A TOML file, to store your experiment configurations.
  In the MNIST example: `examples/mnist_quickrun/config.toml`.
* A folder with your data, split by client.
  In the MNIST example: `examples/mnist_quickrun/data_iid`
  (after running `declearn-split --folder "examples/mnist_quickrun"`).
* A Python model file, to declare your model wrapped in a `declearn` object.
  In the MNIST example: `examples/mnist_quickrun/model.py`.

#### The TOML file

TOML is a minimal, human-readable configuration file format.
We use is to store all the configurations of an FL experiment.
The TOML is parsed by python as dictionary with each `[header]`
as a key. For more details, see the [TOML doc](https://toml.io/en/)

This file is your main entry point to everything else.
The absolute path to this file should be given as an argument in:

```bash
declearn-quickrun --config <path_to_toml_file>
```

The TOML file has six sections, some of which are optional. Note that the order
does not matter, and that we give illustrative, not necessarily functional
examples.

**`[network]`: Network configuration** used by both client and server,
most notably the port, host, and ssl certificates. An example:

```toml
[network]
    protocol = "websockets" # Protocol used, to keep things simple use websocket
    host = "127.0.0.1" # Address used, works as is on most set ups
    port = 8765 # Port used, works as is on most set ups
```

This section is parsed as the initialization arguments to the `NetworkServer`
class. Check its [documentation][declearn.communication.api.NetworkServer]
to see all available fields. Note it is also used to initialize a
[`NetworkClient`][declearn.communication.api.NetworkClient], mirroring the
server.

**`[data]`: Where to find your data**. This is particularly useful if you have
split your data yourself, using custom names for files and folders. An example:

```toml
[data]
    data_folder = "./custom/data_custom" # Your main data folder
    client_names = ["client_a", "client_b", "client_c"] # The names of your client folders

    [data.dataset_names] # The names of train and test datasets
    train_data = "cifar_train"
    train_target = "label_train"
    valid_data = "cifar_valid"
    valid_target = "label_valid"
```

This section is parsed as the fields of a `DataSourceConfig` dataclass.
Check its [documentation][declearn.quickrun/DataSourceConfig] to see
all available fields. This `DataSourceConfig` is then parsed by the
[`parse_data_folder`][declearn.quickrun.parse_data_folder] function.

**`[optim]`: Optimization options** for both client and server, with
three distinct sub-sections: the server-side aggregator (i) and optimizer (ii),
and the client optimizer (iii). An example:

```python
[optim]
    aggregator = "averaging" # The basic server aggregation strategy

    [optim.server_opt] # Server optimization strategy
    lrate = 1.0 # Server learning rate

    [optim.client_opt] # Client optimization strategy
    lrate = 0.001 # Client learning rate
    modules = [["momentum", {"beta" = 0.9}]] # List of optimizer modules used
    regularizers = [["lasso", {alpha = 0.1}]] # List of regularizer modules
```

This section is parsed as the fields of a `FLOptimConfig` dataclass. Check its
[documentation][declearn.main.config.FLOptimConfig] to see more details on
these three sub-sections. For more details on available fields within those
subsections, you can navigate inside the documentation of the
[`Aggregator`][declearn.aggregator.Aggregator] and
[`Optimizer`][declearn.optimizer.Optimizer] classes.

**`[run]`: Training process option** for both client and server. Most notably,
includes the number of rounds as well as the registration, training, and
evaluation parameters. An example:

```toml
[run]
    rounds = 10 # Number of overall training rounds

    [run.register] # Client registration options
    min_clients = 1 # Minimum of clients that need to connect
    max_clients = 6 # The maximum number of clients that can connect
    timeout = 5 # How long to wait for clients, in seconds

    [run.training] # Client training procedure
    n_epoch = 1 # Number of local epochs
    batch_size = 48 # Training batch size
    drop_remainder = false # Whether to drop the last training examples

    [run.evaluate]
    batch_size = 128 # Evaluation batch size
```

This section is parsed as the fields of a `FLRunConfig` dataclass. Check its
[documentation][declearn.main.config.FLOptimConfig] to see more details on the
sub-sections. For more details on available fields within those subsections,
you can navigate inside the documentation of `FLRunConfig` to the relevant
dataclass, for instance [`TrainingConfig`][declearn.main.config.TrainingConfig].

**`[model]`: Optional section**, where to find the model. An example:

```toml
[model]
# The location to a model file
model_file = "./custom/model_custom.py"
# The name of your model file, if different from "MyModel"
model_name = "MyCustomModel"
```

This section is parsed as the fields of a `ModelConfig` dataclass. Check its
[documentation][declearn.quickrun.ModelConfig] to see all available fields.

**`[experiment]`: Optional section**, what to report during the experiment and
where to report it. An example:

```toml
[experiment]
metrics = [
    # Multi-label Accuracy, Precision, Recall and F1-Score.
    ["multi-classif", {labels = [0,1,2,3,4,5,6,7,8,9]}]
]
checkpoint = "./result_custom" # Custom location for results
```

This section is parsed as the fields of a `ExperimentConfig` dataclass.
Check its [documentation][declearn.quickrun.ExperimentConfig] to see all
available fields.

#### The data

Your data, in a standard tabular format, split by client. Within each client
folder, we expect four files : training data and labels, validation data and
labels.

If your data is not already split by client, we are developing an experimental
data splitting utility. It currently has a limited scope, only dealing
with classification tasks, excluding multi-label. You can call it using
`declearn-split --folder <path_to_original_data>`. For more details, refer to
the [documentation][declearn.dataset.split_data].

#### The Model file

The model file should just contain the model you built for
your data, e.g. a `torch` model, wrapped in a declearn object.
See `examples/mnist_quickrun/model.py` for an example.

The wrapped model should be named "model" by default. If you use any other
name, you have to specify it in the TOML file, as demonstrated in
`./custom/config_custom.toml`.

### 2.2. Using declearn full capabilities

To upgrade your experimental setting beyond the `quickrun` mode, you may move
on to the hands-on usage [section](user-guide/usage.md) of the documentation.
