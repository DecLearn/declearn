# Hands-on usage

Here are details on how to set up server-side and client-side programs
that will run together to perform a federated learning process. Generic
remarks from the [Quickstart](#quickstart) section hold here as well, the
former section being an overly simple exemplification of the present one.

You can follow along on a concrete example that uses the UCI heart disease
dataset, that is stored in the `examples/uci-heart` folder. You may refer
to the `server.py` and `client.py` example scripts, that comprise comments
indicating how the code relates to the steps described below. For further
details on this example and on how to run it, please refer to its own
`readme.md` file.

## Server setup instructions

**1. Define a Model**

  - Set up a machine learning model in a given framework
    (_e.g._ a `torch.nn.Module`).
  - Select the appropriate `declearn.model.api.Model` subclass to wrap it up.
  - Either instantiate the `Model` or provide a JSON-serialized configuration.

**2. Define a FLOptimConfig**

  - Select a `declearn.aggregator.Aggregator` (subclass) instance to define
    how clients' updates are to be aggregated into global-model updates on
    the server side.
  - Parameterize a `declearn.optimizer.Optimizer` (possibly using a selected
    pipeline of `declearn.optimizer.modules.OptiModule` plug-ins and/or a
    pipeline of `declearn.optimizer.regularizers.Regularizer` ones) to be
    used by clients to derive local step-wise updates from model gradients.
  - Similarly, parameterize an `Optimizer` to be used by the server to
    (optionally) refine the aggregated model updates before applying them.
  - Optionally, parametrize a `FairnessControllerServer`, defining an
    algorithm to enforce fairness constraints to the model being trained.
  - Wrap these objects into a `declearn.main.config.FLOptimConfig`,
    possibly using its `from_config` method to specify the former three
    components via configuration dicts rather than actual instances.
  - Alternatively, write up a TOML configuration file that specifies these
    components (note that 'aggregator' and 'server_opt' have default values
    and may therefore be left unspecified).

**3. Define a communication server endpoint**

  - Select a communication protocol (_e.g._ "grpc" or "websockets").
  - Select the host address and port to use.
  - Preferably provide paths to PEM files storing SSL-required information.
  - Wrap this into a config dict or use `declearn.communication.build_server`
    to instantiate a `declearn.communication.api.NetworkServer` to be used.

**4. Instantiate and run a FederatedServer**

  - Instantiate a `declearn.main.FederatedServer`:
    - Provide the Model, FLOptimConfig and Server objects or configurations.
    - Optionally provide a MetricSet object or its specs (i.e. a list of
      Metric instances, identifier names of (name, config) tuples), that
      defines metrics to be computed by clients on their validation data.
    - Optionally provide the path to a folder where to write output files
      (model checkpoints and global loss history).
    - Optionally parameterize and provide with a `SecaggConfigServer` or its
      configuration, to set up and use secure aggregation for all quantities
      that support it (model weights, metrics and metadata).
  - Instantiate a `declearn.main.config.FLRunConfig` to specify the process:
    - Maximum number of training and evaluation rounds to run.
    - Registration parameters: exact or min/max number of clients to have
      and optional timeout delay spent waiting for said clients to join.
    - Training parameters: data-batching parameters and effort constraints
      (number of local epochs and/or steps to take, and optional timeout).
    - Evaluation parameters: data-batching parameters and effort constraints
      (optional maximum number of steps (<=1 epoch) and optional timeout);
      optional frequency (to only evaluate after every N training rounds).
    - Early-stopping parameters (optionally): patience, tolerance, etc. as
      to the global model loss's evolution throughout rounds.
    - Local Differential-Privacy parameters (optionally): (epsilon, delta)
      budget, type of accountant, clipping norm threshold, RNG parameters.
    - Fairness evaluation parameters (optionally): computational constraints
      and optionally frequency of fairness rounds; only used if fairness is
      set up in the `FLOptimConfig`, and automatically/dynamically-filled if
      left untouched.
  - Alternatively, write up a TOML configuration file that specifies all of
    the former hyper-parameters.
  - Call the server's `run` method, passing it the former config object,
    the path to the TOML configuration file, or dictionaries of keyword
    arguments to be parsed into a `FLRunConfig` instance.

## Clients setup instructions

**1. Interface training data**

  - Select and parameterize a `declearn.dataset.Dataset` subclass that
    will interface the local training dataset.
  - Ensure its `get_data_specs` method exposes the metadata that is to
    be shared with the server (and nothing else, to prevent data leak).

**2. Interface validation data (optional)**

   - Optionally set up a second Dataset interfacing a validation dataset,
     used in evaluation rounds. Otherwise, those rounds will be run using
     the training dataset - which can be slow and/or lead to overfitting.

**3. Define a communication client endpoint**

  - Select the communication protocol used (_e.g._ "grpc" or "websockets").
  - Provide the server URI to connect to.
  - Preferable provide the path to a PEM file storing SSL-required information
    (matching those used on the Server side).
  - Wrap this into a config dict or use `declearn.communication.build_client`
    to instantiate a `declearn.communication.api.NetworkClient` to be used.

**4. Run any necessary import statement**

  - If optional or third-party dependencies are known to be required, import
    them (_e.g._ `import declearn.model.torch`).
  - Read more about this point [below](#dependency-sharing).

**5. Instantiate a FederatedClient and run it**

  - Instantiate a `declearn.main.FederatedClient`:
    - Provide the NetworkClient and Dataset objects or configurations.
    - Optionally specify `share_metrics=False` to prevent sharing evaluation
      metrics (apart from the aggregated loss) with the server out of privacy
      concerns.
    - Optionally provide the path to a folder where to write output files
      (model checkpoints and local loss history).
    - Optionally parameterize and provide with a `SecaggConfigClient` or its
      configuration, to set up and use secure aggregation for all quantities
      that support it (model weights, metrics and metadata).
  - Call the client's `run` method and let the magic happen.

## Logging

Note that this section and the quickstart example both left apart the option
to configure logging associated with the federated client and server, and/or
the network communication handlers they make use of. One may simply set up
custom `logging.Logger` instances and pass them as arguments to the class
constructors to replace the default, console-only, loggers.

The `declearn.utils.get_logger` function may be used to facilitate the setup
of such logger instances, defining their name, verbosity level, and whether
messages should be logged to the console and/or to an output file.

## Support for GPU acceleration

**TL;DR**: GPU acceleration is natively available in `declearn` for model
frameworks that support it. It may be disabled or configured with one line
of code and without changing your original model.

**Details**:

Most machine learning frameworks, including Tensorflow and Torch, enable
accelerating computations by using computational devices other than CPU.
`declearn` interfaces supported frameworks to be able to set a device policy
in a single line of code, accross frameworks.

`declearn` internalizes the framework-specific code adaptations to place the
data, model weights and computations on such a device. `declearn` provides
with a simple API to define a global device policy. This enables using a
single GPU to accelerate computations, or forcing the use of a CPU.

By default, the policy is set to use the first available GPU, and otherwise
use the CPU, with a warning that can safely be ignored.

Setting the device policy to be used can be done in local scripts, either as a
client or as a server. Device policy is local and is not synchronized between
federated learninng participants.

Here are some examples of the one-liner used:
```python
declearn.utils.set_device_policy(gpu=False)  # disable GPU use
declearn.utils.set_device_policy(gpu=True)  # use any available GPU
declearn.utils.set_device_policy(gpu=True, idx=2)  # specifically use GPU n°2
```

**Known issues**:

- For Haiku / Jax, GPU support must be installed manually by end-users, as it
  is dependent on your local CUDA version, and as such cannot be easily shipped
  as part of declearn's dependencies specification. You most probably will need
  to run `pip install jax[cudaXX_pip]==0.4`, where `XX` is either `11`, `12`,
  or your more recent CUDA version. For more details, please refer to Jax's
  [installation instructions](https://github.com/google/jax#installation).
- For Torch, if you have an unsupported CUDA and/or cuDNN version installed,
  the package may not work at all (even on CPU). This is an issue with Torch,
  and we advise you to report to their documentation or issue tracker if you
  need help fixing it - see for example their installation instructions for
  [version 1.13](https://pytorch.org/get-started/previous-versions/#v1131).

## Dependency sharing

One important issue that is not currently handled by declearn itself is that
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
