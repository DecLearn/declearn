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
- `messaging`:<br/>
  &emsp; API and default classes to define parsable messages for applications.
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
- `version`:<br/>
  &emsp; DecLearn version information, as hard-coded constants.

## Main abstractions

This section lists the main abstractions implemented as part of
`declearn`, exposing their main object and usage, some examples
of ready-to-use implementations that are part of `declearn`, as
well as references on how to extend the support of `declearn`
backend (notably, (de)serialization and configuration utils) to
new custom concrete implementations inheriting the abstraction.

### Model and Tensors

#### `Model`
- Import: `declearn.model.api.Model`
- Object: Interface framework-specific machine learning models.
- Usage: Compute gradients, apply updates, compute loss...
- Examples:
    - `declearn.model.sklearn.SklearnSGDModel`
    - `declearn.model.tensorflow.TensorflowModel`
    - `declearn.model.torch.TorchModel`
- Extend: use `declearn.utils.register_type(group="Model")`

#### `Vector`
- Import: `declearn.model.api.Vector`
- Object: Interface framework-specific data structures.
- Usage: Wrap and operate on model weights, gradients, updates...
- Examples:
    - `declearn.model.sklearn.NumpyVector`
    - `declearn.model.tensorflow.TensorflowVector`
    - `declearn.model.torch.TorchVector`
- Extend: use `declearn.model.api.register_vector_type`

### Federated Optimization

You may learn more about our (non-abstract) `Optimizer` API by reading our
[Optimizer guide](./optimizer.md).

#### `Aggregator`
- Import: `declearn.aggregator.Aggregator`
- Object: Define model updates aggregation algorithms.
- Usage: Post-process client updates; finalize aggregated global ones.
- Examples:
    - `declearn.aggregator.AveragingAggregator`
    - `declearn.aggregator.GradientMaskedAveraging`
- Extend:
    - Simply inherit from `Aggregator` (registration is automated).
    - To avoid it, use `class MyAggregator(Aggregator, register=False)`.

#### `ModelUpdates`
- Import: `declearn.aggregator.ModelUpdates`
- Object: Define exchanged model updates data and their aggregation.
- Usage: Share and aggregate client's updates for a given `Aggregator`.
- Examples:
    - Each `Aggregator` has its own dedicated/supported `ModelUpdates` type(s).
- Extend:
    - Simply inherit from `ModelUpdates` (registration is automated).
    - Define a `name` class attribute and decorate as a `dataclass`.
    - To avoid it, use `class MyModelUpdates(ModelUpdates, register=False)`.

#### `OptiModule`
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

#### `Regularizer`
- Import: `declearn.optimizer.regularizers.Regularizer`
- Object: Define loss-regularization terms as gradients modifiers.
- Usage: Plug into a `declearn.optimizer.Optimizer`.
- Examples:
    - `declearn.optimizer.regularizers.FedProxRegularizer`
    - `declearn.optimizer.regularizers.LassoRegularizer`
    - `declearn.optimizer.regularizers.RidgeRegularizer`
- Extend:
    - Simply inherit from `Regularizer` (registration is automated).
    - To avoid it, use `class MyRegularizer(Regularizer, register=False)`.

#### `AuxVar`
- Import: `declearn.optimizer.modules.AuxVar`
- Object: Define exchanged data between a pair of `OptiModules` across the
  clients/server boundary, and their aggregation.
- Usage: Share information from server to clients and reciprocally.
- Examples:
    - `declearn.optimizer.modules.ScaffoldAuxVar`
- Extend:
    - Simply inherit from `AuxVar` (registration is automated).
    - Define a `name` class attribute and decorate as a `dataclass`.
    - To avoid it, use `class MyAuxVar(AuxVar, register=False)`.

### Evaluation Metrics

#### `Metric`
- Import: `declearn.metrics.Metric`
- Object: Define evaluation metrics to compute iteratively and federatively.
- Usage: Compute local and federated metrics based on local data streams.
- Examples:
    - `declearn.metric.BinaryRocAuc`
    - `declearn.metric.MeanSquaredError`
    - `declearn.metric.MuticlassAccuracyPrecisionRecall`
- Extend:
    - Simply inherit from `Metric` (registration is automated).
    - To avoid it, use `class MyMetric(Metric, register=False)`.

#### `MetricState`
- Import: `declearn.metrics.MetricState`
- Object: Define exchanged data to compute a `Metric` and their aggregation.
- Usage: Share locally-computed metrics for their aggregation into global ones.
- Examples:
    - Each `Metric` has its own dedicated/supported `MetricState` type(s).
- Extend:
    - Simply inherit from `MetricState` (registration is automated).
    - Define a `name` class attribute and decorate as a `dataclass`.
    - To avoid it, use `class MyMetricState(MetricState, register=False)`.

### Network communication

#### `NetworkClient`
- Import: `declearn.communication.api.NetworkClient`
- Object: Instantiate a network communication client endpoint.
- Usage: Register for training, send and receive messages.
- Examples:
    - `declearn.communication.grpc.GrpcClient`
    - `declearn.communication.websockets.WebsocketsClient`
- Extend:
    - Simply inherit from `NetworkClient` (registration is automated).
    - To avoid it, use `class MyClient(NetworkClient, register=False)`.

#### `NetworkServer`
- Import: `declearn.communication.api.NetworkServer`
- Object: Instantiate a network communication server endpoint.
- Usage: Receive clients' requests, send and receive messages.
- Examples:
    - `declearn.communication.grpc.GrpcServer`
    - `declearn.communication.websockets.WebsocketsServer`
- Extend:
    - Simply inherit from `NetworkServer` (registration is automated).
    - To avoid it, use `class MyServer(NetworkServer, register=False)`.

#### `Message`
- Import: `declearn.messaging.Message`
- Object: Define serializable/parsable message types and their data.
- Usage: Exchanged via communication endpoints to transmit data and
  trigger behaviors based on type analysis.
- Examples:
    - `declearn.messages.TrainRequest`
    - `declearn.messages.TrainReply`
    - `declearn.messages.Error`
- Extend:
    - Simply inherit from `Message` (registration is automated).
    - To avoid it, use `class MyMessage(Message, register=False)`.

### Dataset

#### `Dataset`
- Import: `declearn.dataset.Dataset`
- Object: Interface data sources agnostic to their format.
- Usage: Yield (inputs, labels, weights) data batches, expose metadata.
- Examples:
    - `declearn.dataset.InMemoryDataset`
    - `declearn.dataset.tensorflow.TensorflowDataset`
    - `declearn.dataset.torch.TorchDataset`
- Extend: use `declearn.utils.register_type(group="Dataset")`.

## Full API Reference

The full API reference, which is generated automatically from the code's
internal documentation, can be found [here](../api-reference/index.md).
