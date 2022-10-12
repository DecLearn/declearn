# Declearn: a modular and extensible framework for Federated Learning.

- [Introduction](#introduction)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Usage of the Python API](#usage-of-the-python-api)
  - [Overview of the Federated Learning process](#overview-of-the-federated-learning-process)
  - [Overview of the declearn API](#overview-of-the-declearn-api)
  - [Hands-on usage](#hands-on-usage)
- [Developers](#developers)

--------------------

## Introduction

`declearn` is a python package providing with a framework to perform federated
learning, i.e. to train machine learning models by distributing computations
across a set of data owners that, consequently, only have to share aggregated
information (rather than individual data samples) with an orchestrating server
(and, by extension, with each other).

The aim of `declearn` is to provide both real-world end-users and algorithm
researchers with a modular and extensible framework that:
* builds on **abstractions** general enough to write backbone algorithmic code
  agnostic to the actual computation framework, statistical model details
  or network communications setup
* designs **modular and combinable** objects, so that algorithmic features, and
  more generally any specific implementation of a component (the model, network
  protocol, client or server optimizer...) may easily be plugged into the main
  federated learning process - enabling users to experiment with configurations
  that intersect unitary features
* provides with functioning tools that may be used **out-of-the-box** to set up
  federated learning tasks using some popular computation frameworks (scikit-
  learn, tensorflow, pytorch...) and federated learning algorithms (FedAvg,
  Scaffold, FedYogi...)
* provides with tools that enable **extending** the support of existing tools
  and APIs to custom functions and classes without having to hack into the
  source code, merely adding new features (tensor libraries, model classes,
  optimization plug-ins, orchestration algorithms, communication protocols...)
  to the party

At the moment, `declearn` has been focused on so-called "centralized" federated
learning that implies a central server orchestrating computations, but it might
become more oriented towards decentralized processes in the future, that remove
the use of a central agent.


## Setup
### Requirements

* python >= 3.8
* pip

Third-party requirements are specified (and automatically installed) as part
of the installation process, and may be consulted from the `pyproject.toml`
file.

### Optional requirements

Some third-party requirements are optional, and may not be installed. These
are also specified as part of the `pyproject.toml` file, and may be divided
into two categories:<br/>
(a) dependencies of optional, applied declearn components
(such as the TensorFlow and PyTorch tensor libraries) that are not imported
with declern by default ;<br/>
(b) dependencies for running tests on the package (mainly pytest and some of
its plug-ins)

The second category is more developer-oriented, while the first may or may not
be relevant depending on the use case to which you wish to apply `declearn`.

### Using a virtual environment (optional)

It is generally advised to use a virtual environment, to avoid any dependency
conflict between declearn and packages you might use in separate projects. To
do so, you may for example use python's built-in
[venv](https://docs.python.org/3/library/venv.html), or the third-party tool
[conda](https://docs.conda.io/en/latest/).

Venv instructions (example):
```bash
python -m venv ~/.venvs/declearn
source ~/.venvs/declearn/bin/activate
```

Conda instructions (example):
```bash
conda create -n declearn python=3.8 pip
conda activate declearn
```

_Note: at the moment, conda installation is not recommended, because the
package's installation is made slightly harder due to some dependencies being
installable via conda while other are only available via pip/pypi, which can
lead to dependency-tracking trouble._

### Installation

To install the package, simply clone the git repository and run `pip install .`
from its root folder.

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn.git
cd declearn
pip install .  # or pip install -e .
```

To also install optional requirements, add the name of the extras between
brackets to the `pip install` command, _e.g._ running one of the following:

```bash
pip install .[torch]  # install declearn.model.torch submodule dependencies
pip install .[tensorflow,torch]  # install both tensorflow and torch
pip install .[tests]  # install all optional dependencies plus testing ones
```

**Notes**:
* If you are not using a virtual environment, select carefully the `pip`
  binary being called (e.g. use `python -m pip`), and/or add a `--user`
  flag to the pip command.
* Developers may have better installing the package in editable mode,
  using `pip install -e .`
* If you are installing the package within a conda environment, it may
  be better to run `pip install --no-deps .` so as to only install the
  package, and then to manually install the dependencies listed in the
  `pyproject.toml` file, using `conda install` rather than `pip install`
  whenever it is possible.

## Quickstart

### Setting

Here is a quickstart example on how to set up a federated learning process
to learn a LASSO logistic regression model (using a scikit-learn backend)
using pre-processed data, formatted as csv files with a "label" column,
where each client has two files: one for training, the other for validation.

Here, the code uses:
* standard FedAvg strategy (SGD for local steps, averaging of updates weighted
  by clients' training dataset size, no modifications of server-side updates)
* 10 rounds of training, with 5 local epochs performed at each round and
  128-samples batch size
* at least 1 and at most 3 clients, awaited for 180 seconds by the server
* network communications using gRPC, on host "example.com" and port 8888

Note that this example code may easily be adjusted to suit use cases, using
other types of models, alternative federated learning algorithms and/or
modifying the communication, training and validation hyper-parameters.
Please refer to the [Hands-on usage](#hands-on-usage) section for a more
general and detailed description of how to set up a federated learning
task and process with declearn.

### Server-side script

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
strat = declearn.strategy.FedAvg()
server = declearn.main.FederatedServer(model, netwk, strat, folder="outputs")
server.run(
    rounds=10,
    regst_cfg={"min_clients": 1, "max_clients": 3, "timeout": 180},
    train_cfg={"n_epochs": 5, "batch_size": 128, "drop_remainder": False}
)
```

### Client-side script

```python
import declearn

netwk = declearn.communication.NetworkClientConfig(
    server_uri="example.com:8888",
    name="client_name",
    certificate="path/to/client_cert.pem"
)
train = declearn.dataset.InMemoryDataset(
    "path/to/train.csv", target="label",
    expose_classes=True  # enable sharing of unique target values
)
valid = declearn.dataset.InMemoryDataset("path/to/valid.csv", target="label")
client = declearn.main.FederatedClient(netwk, train, valid, folder="outputs")
client.run()
```

### Note on dependency sharing

One important issue however that is not handled by declearn itself is that
of ensuring that clients have loaded all dependencies that may be required
to unpack the Model and Optimizer instances transmitted at initialization.
At the moment, it is therefore left to users to agree on the dependencies
that need to be imported as part of the client-side launching script.

For example, if the trained model is an artificial neural network that uses
PyTorch as implementation backend, clients will need to add the
`import declearn.model.torch` statement in their code (and, obviously, to
have `torch` installed). Similarly, if a custom declearn `OptiModule` was
written to modify the way updates are computed locally by clients, it will
need to be shared with clients - either as a package to be imported (like
torch previously), or as a bit of source code to add on top of the script.


## Usage of the Python API

### Overview of the Federated Learning process

This overview describes the way the `declearn.main.FederatedServer`
and `declearn.main.FederatedClient` pair of classes implement the
federated learning process. It is however possible to subclass
these and/or implement alternative orchestrating classes to define
alternative overall algorithmic processes - notably by overriding
or extending methods that define the sub-components of the process
exposed here.

#### Overall process orchestrated by the server

- Initially:
  - have the clients connect and register for training
  - prepare model and optimizer objects on both sides
- Iteratively:
  - perform a training round
  - perform an evaluation round
  - decide whether to continue, based on the number of
    rounds taken or on the evolution of the global loss
- Finally:
  - notify clients that training is over, so they can disconnect
    and run their final routine (e.g. model saving)
  - optionally save the model (through a checkpointer)
  - close the network server and end the process


#### Detail of the process phases

- **Registration process**:
  - Server:
    - open up registration (stop rejecting all received messages)
    - handle and respond to client-emitted registration requests
    - await criteria to have been met (exact or min/max number of clients
      registered, optionally under a given timeout delay)
    - close registration (reject future requests)
  - Client:
    - gather metadata about the local training dataset
      (_e.g._ dimensions and unique labels)
    - connect to the server and send a request to join training,
      including the former information
    - await the server's response (retry after a timeout if the request
      came in too soon, i.e. registration is not opened yet)
  - messaging : (JoinRequest <-> JoinReply)


* **Post-registration initialization**
  - Server:
    - validate and aggregate clients-transmitted metadata
    - finalize the model's initialization using those metadata
    - send the model and local optimizer's specs to all clients
  - Client:
    - instantiate the model and optimizer based on server instructions
  - messaging: (InitRequest <-> InitReply)


* **Training round**:
  - Server:
    - select clients that are to participate
    - send data-batching and effort constraints parameters
    - send shared model weights and (opt. client-specific) auxiliary variables
  - Client:
    - update model weights and optimizer auxiliary variables
    - perform training steps based on effort constraints
    - step: compute gradients over a batch; compute updates; apply them
    - finally, send back local model weights and auxiliary variables
  - messaging: (TrainRequest <-> TrainReply)
  - Server:
    - unpack and aggregate clients' model weights into global updates
    - unpack and process clients' auxiliary variables
    - run global updates through the server's optimizer to modify and finally
      apply them


* **Evaluation round**:
  - Server:
    - select clients that are to participate
    - send data-batching parameters and shared model weights
    - (_send effort constraints, unused for now_)
  - Client:
    - update model weights
    - compute the model's loss over the entire validation dataset
    - checkpoint the model, then send back the loss to the server
  - messaging: (EvaluateRequest <-> EvaluateReply)
  - Server:
    - aggregate local loss values into a global loss metric
    - checkpoint the model and the global loss

### Overview of the declearn API

#### Package structure

The package is organized into the following submodules:
* `communication`:<br/>
  &emsp; Client-Server network communications API and implementations.
* `data_info`:<br/>
  &emsp; Tools to write and extend shareable metadata fields specifications.
* `dataset`:<br/>
  &emsp; Data interfacing API and implementations.
* `main`:<br/>
  &emsp; Main classes implementing a Federated Learning process.
* `model`:<br/>
  &emsp; Model interfacing API and implementations.
* `optimizer`:<br/>
  &emsp; Framework-agnostic optimizer and algorithmic plug-ins API and tools.
* `typing`:<br/>
  &emsp; Type hinting utils, defined and exposed for code readability purposes.
* `utils`:<br/>
  &emsp; Shared utils used (extensively) across all of declearn.

#### Main abstractions

This section lists the main abstractions implemented as part of
`declearn`, exposing their main object and usage, some examples
of ready-to-use implementations that are part of `declearn`, as
well as references on how to extend the support of `declearn`
backend (notably, (de)serialization and configuration utils) to
new custom concrete implementations inheriting the abstraction.

* `declearn.model.api.Model`:
  - Object: Interface framework-specific machine learning models.
  - Usage: Compute gradients, apply updates, compute loss...
  - Examples:
    - `declearn.model.sklearn.SklearnSGDModel`
    - `declearn.model.tensorflow.TensorflowModel`
    - `declearn.model.torch.TorchModel`
  - Extend: use `declearn.utils.register_type(group="Model")`


* `declearn.model.api.Vector`:
  - Object: Interface framework-specific data structures.
  - Usage: Wrap and operate on model weights, gradients, updates...
  - Examples:
    - `declearn.model.api.NumpyVector`
    - `declearn.model.tensorflow.TensorflowVector`
    - `declearn.model.torch.TorchVector`
  - Extend: use `declearn.model.api.register_vector_type`


* `declearn.optimizer.modules.OptiModule`:
  - Object: Define optimization algorithm bricks.
  - Usage: Plug into a `declearn.optimizer.Optimizer`.
  - Examples:
    - `declearn.optimizer.modules.AdagradModule`
    - `declearn.optimizer.modules.MomentumModule`
    - `declearn.optimizer.modules.ScaffoldClientModule`
    - `declearn.optimizer.modules.ScaffoldServerModule`
  - Extend: use `declearn.utils.register_type(group="OptiModule")`


* `declearn.communication.api.Client`:
  - Object: Instantiate a network communication client endpoint.
  - Usage: Register for training, send and receive messages.
  - Examples:
    - `declearn.communication.grpc.GrpcClient`
    - `declearn.communication.websockets.WebsocketsClient`
  - Extend: use `declearn.utils.register_type(group="Client")`


* `declearn.communication.api.Server`:
  - Object: Instantiate a network communication server endpoint.
  - Usage: Receive clients' requests, send and receive messages.
  - Examples:
    - `declearn.communication.grpc.GrpcServer`
    - `declearn.communication.websockets.WebsocketsServer`
  - Extend: use `declearn.utils.register_type(group="Server")`


* `declearn.dataset.Dataset`:
  - Object: Interface data sources agnostic to their format.
  - Usage: Yield (inputs, labels, weights) data batches, expose metadata.
  - Examples:
    - `declearn.dataset.InMemoryDataset`
  - Extend: use `declearn.utils.register_type(group="Dataset")`


### Hands-on usage

Here are details on how to set up server-side and client-side programs
that will run together to perform a federated learning process. Generic
remarks from the [Quickstart](#quickstart) section hold here as well, the
former section being an overly simple examplification of the present one.

#### Server setup instructions

* Define a Model:
  - Set up a machine learning model in a given framework
    (_e.g._ a `torch.nn.Module`).
  - Select the appropriate `declearn.model.api.Model` subclass to wrap it up.
  - Either instantiate the `Model` or provide a JSON-serialized configuration.
* Define a Strategy:
  - Select an out-of-the-box `declearn.strategy.Strategy` subclass that
    defines the aggregation and optimization strategies for the process
    (_e.g._ `declearn.strategy.FedAvg` or `declearn.strategy.Scaffold`)
  - Parameterize and instantiate it.
  <br/>**- OR -**
  - Select and parameterize a `declearn.strategy.Aggregator` (subclass)
    instance to define how clients' updates are to be aggregated into
    global-model updates on the server side.
  - Parameterize a `declearn.optimizer.Optimizer` (possibly using a selected
    pipeline of `declearn.optimizer.modules.OptiModule` plug-ins) to be
    used by clients to derive local step-wise updates from model gradients.
  - Similarly, parameterize an `Optimizer` to be used by the server to
    (optionally) refine the aggregated model updates before applying them.
  - Wrap these three objects into a custom `Strategy` using
    `declearn.strategiy.strategy_from_config`. Use instantiated objects'
    `get_config` method if needed to abide by the former function's specs.
* Define a communication Server:
  - Select a communication protocol (_e.g._ "grpc" or "websockets").
  - Select the host address and port to use.
  - Optionally provide paths to PEM files storing SSL-required information.
  - Wrap this into a config dict or use `declearn.communication.build_server`
    to instantiate a `declearn.communication.api.Server` that will be used.
* Instantiate a `declearn.main.FederatedServer`:
  - Provide the Model, Strategy and Server objects or configurations.
  - Optionally provide the path to a folder where to write output files
    (model checkpoints and global loss history).
* Call the server's `run` method, further specifying:
  - Registration parameters: exact or min/max number of clients to have
    and optional timeout delay spent waiting for said clients to join.
  - Training parameters: data-batching parameters and effort constraints
    (number of local epochs and/or steps to take, and optional timeout).
  - Evaluation parameters: data-batching parameters (_as of now, effort
    constraints are not yet used by the clients_).

#### Clients setup instructions

* Interface training data:
  - Select and parameterize a `declearn.dataset.Dataset` subclass that
    will interface the local training dataset.
  - Ensure its `get_data_sepcs` method exposes the metadata that is to
    be shared with the server (and nothing else, to prevent data leak).
* Interface validation data (optional):
  - Optionally set up a second Dataset interfacing a validation dataset,
    used in evaluation rounds. Otherwise, those rounds will be run using
    the training dataset - which can be slow and/or lead to overfitting.
* Define a communication Client:
  - Select the communication protocol used (_e.g._ "grpc" or "websockets").
  - Provide the server URI to connect to.
  - Optionally provide path to a PEM file storing SSL-required information.
  - Wrap this into a config dict or use `declearn.communication.build_client`
    to instantiate a `declearn.communication.api.Client` that will be used.
* Run any necessary import statement:
  - If optional or third-party dependencies are known to be required, import
    them (_e.g._ `import declearn.model.torch`).
* Instantiate a `declearn.main.FederatedClient` and run it:
  - Provide the Client and Dataset objects or configurations.
  - Optionally provide the path to a folder where to write output files
    (model checkpoints and local loss history).
  - Call the client's `run` method and let the magic happen.

#### Logging

Note that this section and the quickstart example both left apart the option
to configure logging associated with the federated client and server, and/or
the network communication handlers they make use of. One may simply set up
custom `logging.Logger` instances and pass them as arguments to the class
constructors to replace the default, console-only, loggers.

The `declearn.utils.get_logger` function may be used to facilitate the setup
of such logger instances, defining their name, verbosity level, and whether
messages should be logged to the console and/or to an output file.

## Developers

### Contributions

Contributions to `declearn` are welcome, whether to provide fixes, suggest
new features (_e.g._ new subclasses of the core abstractions) or even push
forward framework evolutions and API revisions.

To contribute directly to the code (beyond posting issues on gitlab), please
create a dedicated branch, and submit a **Merge Request** once you want your
work reviewed and further processed to end up integrated into the main branch.

The **coding rules** are fairly simple:
* abide by [PEP 8](https://peps.python.org/pep-0008/), in a way that is
  coherent with the practices already at work in declearn
* abide by [PEP 257](https://peps.python.org/pep-0257/), _i.e._ write
  docstrings **everywhere** (unless inheriting from a method, the behaviour
  and signature of which are unmodified), again using formatting that is
  coherent with the declearn practices
* type-hint the code, abiding by [PEP 484](https://peps.python.org/pep-0484/);
  note that the use of Any and of "type: ignore" comments is authorized, but
  should be remain sparse
* lint your code with [mypy](http://mypy-lang.org/) (for static type checking)
  and [pylint](https://pylint.pycqa.org/en/latest/) (for more general linting);
  do use "type: ..." and "pylint: disable=..." comments where you think it
  relevant, preferably with some side explanations
  (see dedicated sub-section [below](#running-black-to-format-the-code))
* reformat your code using [black](https://github.com/psf/black); do use
  (sparingly) "fmt: off/on" comments when you think it relevant
  (see dedicated sub-section [below](#running-pylint-to-check-the-code))

### Unit tests and code analysis

Unit tests, as well as more-involved functional ones, are implemented under
the `test/` folder of the present repository.
They are implemented using the [PyTest](https://docs.pytest.org) framework,
as well as some third-party plug-ins (refer to [Setup][#setup] for details).

Additionally, code analysis tools are configured through the `pyproject.toml`
file, and used to control code quality upon merging to the main branch. These
tools are [black](https://github.com/psf/black) for code formatting, and
[pylint](https://pylint.pycqa.org/) for static code analysis.

#### Running the test suite using tox

The third-party [tox](https://tox.wiki/en/latest/) tools may be used to run
the entire test suite within a dedicated virtual environment. Simply run `tox`
from the commandline with the root repo folder as working directory. You may
optionally specify the python version(s) with which you want to run tests.

```bash
tox           # run with default py 3.8
tox -e py310  # override to use py 3.10
```

Note that additional parameters for `pytest` may be passed as well, by adding
`--` followed by any set of options you want at the end of the `tox` command.
For example, to use the declearn-specific `--fulltest` option (see the section
below), run:
```bash
tox [tox options] -- --fulltest
```

#### Running unit tests using pytest

To run all the tests, simply use:
```bash
pytest test
```

To run the tests under a given module (here, "model"):
```bash
pytest test/model
```

To run the tests under a given file (here, "test_main.py"):
```bash
pytest test/test_main.py
```

Note that by default, some test scenarios that are considered somewhat
superfluous~redundant will be skipped in order to save time. To avoid
skipping these, and therefore run a more complete test suite, add the
`--fulltest` option to pytest:
```bash
pytest --fulltest test  # or any more-specific target you want
```

#### Running black to format the code

The [black](https://github.com/psf/black) code formatter is used to enforce
uniformity of the source code's formatting style. It is configured to have
a maximum line length of 79 (as per [PEP 8](https://peps.python.org/pep-0008/))
and ignore auto-generated protobuf files, but will otherwise modify files
in-place when executing the following commands from the repository's root
folder:
```bash
black declearn  # reformat the package
black test      # reformat the tests
```

Note that it may also be called on individual files or folders.
One may "blindly" run black, however it is actually advised to have a look
at the reformatting operated, and act on any readability loss due to it. A
couple of advice:
1. Use `#fmt: off` / `#fmt: on` comments sparingly, but use them.
<br/>It is totally okay to protect some (limited) code blocks from
reformatting if you already spent some time and effort in achieving a
readable code that black would disrupt. Please consider refactoring as
an alternative (e.g. limiting the nest-depth of a statement).

2. Pre-format functions and methods' signature to ensure style homogeneity.
<br/>When a signature is short enough, black may attempt to flatten it as a
one-liner, whereas the norm in declearn is to have one line per argument,
all of which end with a trailing comma (for diff minimization purposes). It
may sometimes be necessary to manually write the code in the latter style
for black not to reformat it.


Finally, note that the test suite run with tox comprises code-checking by
black, and will fail if some code is deemed to require alteration by that
tool. You may run this check manually:
```bash
black --check declearn  # or any specific file or folder
```

#### Running pylint to check the code

The [pylint](https://pylint.pycqa.org/) linter is expected to be used for
static code analysis. As a consequence, `# pylint: disable=[some-warning]`
comments can be found (and added) to the source code, preferably with some
indication as to the rationale for silencing the warning (or error).

A minimal amount of non-standard hyper-parameters are configured via the
`pyproject.toml` file and will automatically be used by pylint when run
from within the repository's folder.

Most code editors enable integrating the linter to analyze the code as it is
being edited. To lint the entire package (or some specific files or folders)
one may simply run `pylint`:
```bash
pylint declearn  # analyze the package
pylint test      # analyze the tests
```

Note that the test suite run with tox comprises the previous two commands,
which both result in a score associated with the analyzed code. If the score
does not equal 10/10, the test suite will fail - notably preventing acceptance
of merge requests.
