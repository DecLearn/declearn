# Overview of the declearn API

## Package structure

The package is organized into the following submodules:

- `aggregator`:<br/>
  &emsp; Model updates aggregating API and implementations.
- `communication`:<br/>
  &emsp; Client-Server network communications API and implementations.
- `data_info`:<br/>
  &emsp; Tools to write and extend shareable metadata fields specifications.
- `dataset`:<br/>
  &emsp; Data interfacing API and implementations.
- `main`:<br/>
  &emsp; Main classes implementing a Federated Learning process.
- `metrics`:<br/>
  &emsp; Iterative and federative evaluation metrics computation tools.
- `model`:<br/>
  &emsp; Model interfacing API and implementations.
- `optimizer`:<br/>
  &emsp; Framework-agnostic optimizer and algorithmic plug-ins API and tools.
- `typing`:<br/>
  &emsp; Type hinting utils, defined and exposed for code readability purposes.
- `utils`:<br/>
  &emsp; Shared utils used (extensively) across all of declearn.

## Main abstractions

This section lists the main abstractions implemented as part of
`declearn`, exposing their main object and usage, some examples
of ready-to-use implementations that are part of `declearn`, as
well as references on how to extend the support of `declearn`
backend (notably, (de)serialization and configuration utils) to
new custom concrete implementations inheriting the abstraction.

### `Model`
- Import: `declearn.model.api.Model`
- Object: Interface framework-specific machine learning models.
- Usage: Compute gradients, apply updates, compute loss...
- Examples:
    - `declearn.model.sklearn.SklearnSGDModel`
    - `declearn.model.tensorflow.TensorflowModel`
    - `declearn.model.torch.TorchModel`
- Extend: use `declearn.utils.register_type(group="Model")`

### `Vector`
- Import: `declearn.model.api.Vector`
- Object: Interface framework-specific data structures.
- Usage: Wrap and operate on model weights, gradients, updates...
- Examples:
    - `declearn.model.sklearn.NumpyVector`
    - `declearn.model.tensorflow.TensorflowVector`
    - `declearn.model.torch.TorchVector`
- Extend: use `declearn.model.api.register_vector_type`

### `OptiModule`
- Import: `declearn.optimizer.modules.OptiModule`
- Object: Define optimization algorithm bricks.
- Usage: Plug into a `declearn.optimizer.Optimizer`.
- Examples:
    - `declearn.optimizer.modules.AdagradModule`
    - `declearn.optimizer.modules.MomentumModule`
    - `declearn.optimizer.modules.ScaffoldClientModule`
    - `declearn.optimizer.modules.ScaffoldServerModule`
- Extend:
    - Simply inherit from `OptiModule` (registration is automated).
    - To avoid it, use `class MyModule(OptiModule, register=False)`.

### `Regularizer`
- Import: `declearn.optimizer.modules.Regularizer`
- Object: Define loss-regularization terms as gradients modifiers.
- Usage: Plug into a `declearn.optimizer.Optimizer`.
- Examples:
    - `declearn.optimizer.regularizer.FedProxRegularizer`
    - `declearn.optimizer.regularizer.LassoRegularizer`
    - `declearn.optimizer.regularizer.RidgeRegularizer`
- Extend:
    - Simply inherit from `Regularizer` (registration is automated).
    - To avoid it, use `class MyRegularizer(Regularizer, register=False)`.

### `Metric`
- Import: `declearn.metrics.Metric`
- Object: Define evaluation metrics to compute iteratively and federatively.
- Usage: Compute local and federated metrics based on local data streams.
- Examples:
    - `declearn.metric.BinaryRocAuc`
    - `declearn.metric.MeanSquaredError`
    - `declearn.metric.MuticlassAccuracyPrecisionRecall`
- Extend:
    - Simply inherit from `Metric` (registration is automated).
    - To avoid it, use `class MyMetric(Metric, register=False)`

### `NetworkClient`
- Import: `declearn.communication.api.NetworkClient`
- Object: Instantiate a network communication client endpoint.
- Usage: Register for training, send and receive messages.
- Examples:
    - `declearn.communication.grpc.GrpcClient`
    - `declearn.communication.websockets.WebsocketsClient`
- Extend:
    - Simply inherit from `NetworkClient` (registration is automated).
    - To avoid it, use `class MyClient(NetworkClient, register=False)`.

### `NetworkServer`
- Import: `declearn.communication.api.NetworkServer`
- Object: Instantiate a network communication server endpoint.
- Usage: Receive clients' requests, send and receive messages.
- Examples:
    - `declearn.communication.grpc.GrpcServer`
    - `declearn.communication.websockets.WebsocketsServer`
- Extend:
    - Simply inherit from `NetworkServer` (registration is automated).
    - To avoid it, use `class MyServer(NetworkServer, register=False)`.

### `Dataset`
- Import: `declearn.dataset.Dataset`
- Object: Interface data sources agnostic to their format.
- Usage: Yield (inputs, labels, weights) data batches, expose metadata.
- Examples:
    - `declearn.dataset.InMemoryDataset`
- Extend: use `declearn.utils.register_type(group="Dataset")`

## Full API Reference

The full API reference, which is generated automatically from the code's
internal documentation, can be found [here](../api-reference/index.md).
