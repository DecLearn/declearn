# Declearn Optimizer - API, design principles and practical how-tos

This guide provides a comprehensive introduction to the Declearn `Optimizer`,
which is one of the core features of the package. It is aimed at both end-users
and developers, as it tackles both the practical use of our API, its advanced
details (including some limitations), its underlying design principles and the
way to build upon it to implement your own federated optimization algorithms
and applications.

As an alternative or a complement to this guide, one may refer to the
[API reference](../api-reference/optimizer/index.md) of `declearn.optimizer`
for a code-driven and docstrings-based exhaustive view on the abstractions,
concrete classes and utils exposed by DecLearn. Here are links to the abstract
base classes' API reference:
[`Optimizer`](../api-reference/optimizer/Optimizer.md),
[`OptiModule`](../api-reference/optimizer/modules/OptiModule.md),
[`Regularizer`](../api-reference/optimizer/regularizers/Regularizer.md).

## Introduction

In short, `declearn.optimizers.Optimizer` is a class designed to provide with
a single entry-point to define SGD-based optimization algorithms, agnostic to
the machine learning framework used to define the model and its weights' data
structure. It is designed around a plug-in system that enables combining some
unitary algorithm pieces into complex optimizers, that leave the opportunity
for both developers and end-users to write up their own plug-ins. It is also
meant to provide with capabilities that are specific to the federated or even
decentralized learning settings, while remaining compatible with the "basic"
centralized setting.

Although it was designed as part of Declearn and in articulation with the rest
of our APIs, our `Optimizer` may be re-used in other projects, and has notably
been made part of [Fed-BioMed](https://fedbiomed.org/), another Inria-spawned
project that implements a Federated Learning solution specifically targetted at
cross-silo settings for healthcare institutions and applications.

## Overview

### Structure

Declearn provides with a unified entry-point to define SGD-based optimizers:
`declearn.optimizer.Optimizer`.

- It provides with the most basic SGD algorithm pieces: it scales input
  gradients into updates based on a learning rate, and optionally adds a
  decoupled weight decay term.
- In addition, it enables setting up pipelines of plug-ins, that are applied
  sequentially to the input gradients so as to refine them prior to applying
  the learning rate scaling and adding the weight decay term.
- Furthermore, the learning rate and weight decay factor may be made to
  evolve throughout training using a time-based scheduler (that counts and
  may use both training steps and rounds).

There are two types of plug-ins to an `Optimizer`:

- `declearn.optimizer.regularizers.Regularizer`:
    - A `Regularizer` plug-in implements a loss regularization term.
    - It receives gradients and model weights as inputs, and returns the modified
      gradients (usually: gradients to which a weight-based term was added).
    - Examples include generic regularization terms, such as the Lasso (L1) and
      Ridge (L2) ones, but also FL-specific ones, such as FedProx.
    - You can list currently-available regularizers (and their names) by calling
      `declearn.optimizer.list_optim_regularizers()`.
    - In most cases, a `Regularizer` actually computes the derivative of the
      desired regularization term based on model weights and/or some internal
      state, and adds the results to input gradients.

- `declearn.optimizer.modules.OptiModule`:
    - An `OptiModule` plug-in implements a given gradients-altering algorithm.
    - It receives gradients as inputs, and returns them after some processing.
    - Examples include norm-based gradient clipping, adaptive algorithms such as
      RMSProp or Adam, or the FL-specific Scaffold algorithm, that manages and
      applies a correction term based on quantities computed federatively during
      training.
    - You can list currently-available optimodules (and their names) by calling
      `declearn.optimizer.list_optim_modules()`.
    - Additional mechanisms enable setting up some information sharing between
      paired client-side and server-side modules in the federated context.
      These can be used to set up FL-specific algorithms such as Scaffold.

To which can be added the learning and weight decay rate schedulers:

- `declearn.optimizer.schedulers.Scheduler`:
  - A `Scheduler` implements a time-based rule to have a float value evolve.
  - It keeps track of and may use both the number of training steps and rounds
    to compute the next scheduled value.
  - Examples include various types of decays, cyclic rules, and linear warmup
    (that may wrap another scheduler to make use of once warmup is over).
  - You can list currently-available scheduleres (and their names) by calling
    `declearn.optimizer.list_rate_schedulers()`.

### Practical use

The syntax to set up an `Optimizer` instance is:

- `optim = Optimizer(lrate, w_decay, regularizers, modules)`
- By default, `w_decay=0.0` (no weight decay), and `regularizers` and `modules`
  are `None`, resulting in a Vanilla SGD optimizer.
- Plug-ins (whether `regularizers` or `modules`) are input as a list, each
  element of which specifies a plug-in, either as:
    - an instance of the desired plug-in
      (e.g. `AdamModule(beta_1=0.9, beta_2=0.99)`)
    - a tuple with the name and hyper-parameters of the plug-in
      (e.g. `("adam", {"beta_1": 0.9, "beta_2": 0.99})`)
    - a string providing only the plug-in's name, resulting in default
      hyper-parameter values to be used (e.g. `"adam"`)
- Both `lrate` and `w_decay` may be input either as:
    - a float value (denoting a constant value)
    - a `Scheduler` instance
    - a tuple with name and hyper-parameters for the desired `Scheduler` (e.g.
      `("cosine-annealing", {"base": 0.01, "max_lr": 0.1, "n_steps": 100000})`)

An `Optimizer` has a configuration and a state, that may be (de)serialized:

- `optim.get_config` may be used to return a JSON-serializable configuration.
- `Optimizer.from_config` may be used to instantiate an optimizer from its
  configuration dict.
- `optim.get_state` may be used to access the current states of an optimizer
  (made recursively of those of its plug-ins).
- `optim.set_state` may be used to reset the states of an optimizer to given
  values.
- The `declearn.utils.json_load` and `json_dump` utils may be used to save and
  load the configuration and state dictionaries to and from JSON files.

### Framework agnosticity

Both the `Optimizer` and its plug-in components operate on the `Model` and
`Vector` APIs of Declearn, enabling their code to be written agnostic to the
machine learning framework (and actual model architecture), so that the (exact)
same algorithms are made available for all supported frameworks.

The only exceptions to that principle are a few framework-specific plug-ins
that provide end-users with adapters to interface framework-specific optimizer
tools, which were designed to facilitate the transition to Declearn as well as
ease efforts to experiment with using less-usual algorithms in a federated
context - in hope that such efforts will eventually result in implementing such
algorithms into "proper" framework-agnostic Declearn plug-ins.

## Design principles

The design choices for the Declearn `Optimizer` were based on the following
objectives and rationale:

1. Provide with framework-agnostic rather than framework-wise implementations
2. Provide with combinable bricks rather than a myriad of optimizer subclasses
3. Enable end-users to write up their own bricks with full tooling support

### Provide with framework-agnostic implementations

The first objective comes from the observation that while each of our supported
machine learning frameworks (notably TensorFlow and Torch) provide with their
own optimization API and implementations for a number of algorithms, using them
directly would yield costs of three distinct natures. First, it would mean that
any algorithm that would not be available out-of-the-box (_e.g._ ones that are
specific to federated learning) would have to be implemented once per supported
framework. Second, it would mean that adding support for a new maching learning
framework would require re-implementing each and every pre-existing algorithm
(or suffer some asymetry of capabilities across frameworks, which goes against
the ambition of Declearn). Finally, it would mean that discrepancies between
framework-specific third-party implementations would be borne in Declearn -
_e.g._ the fact that the Torch and TensorFlow implementations of Adam differ,
or that thay do not abide by the same definition of weight decay, would be
kept, which again goes against our ambition to see the choice of framework as a
mere implementation detail.

### Provide with combinable bricks

The second objective comes from the fact that many research papers on federated
optimization provide with specifications to modify a given point in SGD-based
or existing algorithms, that are in fact open to be combined with other bricks
and/or re-use a common algorithmic backbone. While many frameworks (including
Torch and TensorFlow) choose to implement optimization algorithms as subclasses
of an abstract optimizer (with possible shared backend code), Declearn takes a
reverse approach, where a single `Optimizer` class is provided, the instances
of which are populated with plug-ins that implement unitary algorithmic bricks
and are combinable into complex algorithms. One may for instance use a FedProx
loss regularization term together with some gradient clipping and an adaptive
algorithm that scales the resulting gradients based on some momentum. While
this means that end-users can and may write up non-sensical algorithms, we
decided in favor of bearing this risk rather than limit or complexify the
process through which one may configure their desired optimizer and/or run
some experiments on how some bricks interact with each other.

### Enable end-users to write up their own bricks

The third objective comes from the fact that we do not expect Declearn to ever
cover the entire range of algorithms that end-users may want to use; especially
as it aims to be used for research purposes. It is therefore primordial to us
that end-users may easily write up their own plug-ins and use them, whether in
simulated FL experiments or in real-life deployments, with full support by the
rest of the Declearn machinery. A type-registration system was therefore built
to enable offering the same support for third-party plug-ins as for the ones
that are integrated as part of the main Declearn package. In practice, this
means that new plug-ins can be defined in the context of a specific use-cases
as well as distributed via third-party Declearn add-on packages (in the vein
of what `tensorflow_addons` used to do for TensorFlow). It is also possible to
have our unit tests run for third-party plug-ins (as detailed below in this
guide).

## Practical examples

### SGD-M optimizer

This sets up an optimizer that uses SGD with Momentum, using the default
hyper-parameters of the `MomentumModule`:

```python
import declearn

optim = declearn.optimizer.Optimizer(lrate=0.01, modules=["momentum"])
```

This sets up a similar optimizer, with 0.8 Nesterov Momentum:

```python
import declearn
from declearn.optimizer.modules import MomentumModule

momentum = declearn.optimizer.modules.MomentumModule(beta=0.8, nesterov=True)
optim = declearn.optimizer.Optimizer(lrate=0.01, modules=[momentum])
```

This does exactly the same, with a different, just-as-valid syntax:

```python
import declearn

momentum = ("momentum", {"beta": 0.8, "nesterov": True})
optim = declearn.optimizer.Optimizer(lrate=0.01, modules=[momentum])
```

### SGDR optimizer

This sets up an optimizer that implements Stochastic Gradient Descent with
Warm Restarts (SGDR), _i.e._ SGD with cosine annealing cycles of the learning
rate.

```python
import declearn

scheduler = declearn.optimizer.schedulers.CosineAnnealingWarmRestarts(
    base=0.001,  # anneal towards 0.001 learning rate
    max_lr=0.1,  # anneal from 0.1 at the first cycle
    period=100,  # warm restart every 100 training steps
    t_mult=0.5,  # halve max_lr at the end of each cycle
    step_level=True,  # use step as time unit, not round (default)
)
optim = declearn.optimizer.Optimizer(lrate=scheduler)
```

This does exactly the same, with a different, just-as-valid syntax:

```python
import declearn

config = (
    "cosine-annealing-warm-restarts",
    {"base": 0.001, "max_lr": 0.1, "period": 100, "t_mult": 0.5}
)
optim = declearn.optimizer.Optimizer(lrate=config)
```

### Complex optimizer (FedProx & AdamW with gradient clipping)

This sets up an Optimizer with FedProx regularization (to constraint client
drift related to data heterogeneity), L2-norm-based gradient clipping (to avoid
exploding gradients), and an AdamW adaptive algorithm (made of Adam and a
decoupled weight decay term).

In this case, we use a 0.001 learning rate, 0.01 weight decay, a 1.0 clipping
threshold, 0.01 alpha for FedProx (mu in the original paper) and default values
for Adam's hyper-parameters (beta1=0.9, beta2=0.99 and epsilon=1e-7).

```python
import declearn

optim = declearn.optimizer.Optimizer(
    lrate=0.001,
    w_decay=0.01,
    regularizers=[("fedprox", {"alpha": 0.01})]
    modules=[
        ("l2-clipping", {"max_norm": 1.0}),
        ("adam", {"beta1": 0.9, "beta2": 0.99, "eps": 1e-7}),
    ],
)
```

Since all plug-ins hyper-parameters specified here are the default ones, we
could be less explicit in the instantiation instructions and go with:

```python
import declearn

optim = declearn.optimizer.Optimizer(
    lrate=0.001,
    w_decay=0.01,
    regularizers=["fedprox"],
    modules=["l2-clipping", "adam"],
)
```

Of course, one may be even more explicit by manually instantiating the plug-ins
and passing them to the `Optimizer` constructor:

```python
import declearn

fedprox = declearn.optimizers.regularizers.FedProx(alpha=0.01)
l2_clip = declearn.optimizers.modules.L2Clipping(max_norm=1.0)
adam = declearn.optimizers.modules.AdamModule(beta1=0.9, beta2=0.99, eps=1e-7)

optim = declearn.optimizer.Optimizer(
    lrate=0.001,
    w_decay=0.01,
    regularizers=[fedprox],
    modules=[l2_clip, adam],
)
```

### Scaffold

[Scaffold](https://arxiv.org/abs/1910.06378) is a Federated Learning algorithm
that aims at improving over vanilla FedAvg in contexts where data heterogeneity
leads to clients having distinct optimal model solutions, the average of which
does not correspond to the global optimum. It relies on the introduction of
correction terms to the locally-computed gradients, that are computed based on
both a shared global state and client-wise ones.

To implement Scaffold in Declearn, one needs to set up both server-side and
client-side OptiModule plug-ins. The client-side module is in charge of both
correcting input gradients and computing the required quantities to update the
states at the end of each training round, while the server-side module merely
manages the computation and distribution of the global referencestate.

The following snippet sets up a pair of client-side and server-side optimizers
that implement Scaffold, here with a 0.001 learning rate on the client side and
a 0.9 one on the server side, which are both arbitrary values to adjust to your
actual use-case.

```python
import declearn

client_opt = declearn.optimizer.Optimizer(
    lrate=0.001,
    modules=[("scaffold-client")],
)

server_opt = declearn.optimizer.Optimizer(
    lrate=0.9,
    modules=[("scaffold-server")],
)
```

In practice, both of these instances (or their configuration) should be wrapped
as part of the `declearn.main.config.FLOptimConfig` that would be provided at
instantiation to the server-side `declearn.main.FederatedServer` instance used
for orchestrating the federated learning process.

For more details on how the paired modules exchange information, see the
section on [auxiliary variables](#optimodule-auxiliary-variables) below.

### Interface a TensorFlow or Torch Optimizer

In general, we strongly advise end-users to make use of the Declearn-specific
optimizer plug-ins, whether provided as part of Declearn or written as part of
your use-case code or a third-party extension library. There might however be
cases where one _really_ wants to use a specific optimizer written using the
framework-specific API of TensorFlow or Torch, _e.g._ because it is a very
specific, experimental and/or complex algorithm that is either too hard or
not mature enough to be worth the effort re-implementing using the Declearn
syntax. In that case, you may use a Declearn-provided interface to wrap that
framework-specific object into an `OptiModule` plug-in instance.

Note that in both cases, the learning rate defined as part of the TensorFlow
or Torch optimizer will be overridden in favor of that defined by the Declearn
optimizer into which it is being plugged.

Here is an example code snippet in Torch:

```python
import declearn
import declearn.model.torch
import torch


plugin = declearn.model.torch.TorchOptiModule(
    torch.optim.RAdam,  # note that this is a type, not an instance
    # you may pass any RAdam instantiation kwargs here
)

optim = declearn.optimizer.Optimizer(
    lrate=0.001,
    modules=[plugin],
)
```

And here is an example code snippet in TensorFlow:

```python
import declearn
import declearn.model.tensorflow
import tensorflow as tf


tf_opt = tf.optimizers.Nadam()
plugin = declearn.model.tensorflow.TensorFlowOptiModule(tf_opt)

optim = declearn.optimizer.Optimizer(
    lrate=0.001,
    modules=[plugin],
)
```

## Integration in the Declearn Federated Learning process

In Declearn, the Federated Learning optimization problem is formalized as
follows:

- An orchestrating server is in charge of learning a set of model parameters
  $\theta$ using gradient descent, based on data that is distributed among an
  ensemble of clients, each of which holds a dataset $\mathcal{D}_i$.

- At each training round:
    - (a subset of) the clients receive the current model
      weights $\theta^{(t)}$, and perform a number of stochastic gradient
      descent steps based on their dataset, that result in $K_i$ local weights
      updates:
        - $\theta_i^{(t, k + 1)} = \theta^{(t, k)} - client\_opt(\theta_i^{(t, k)}, \nabla\theta_i^{(t, k)}(B_{i, k}))$.
    - the server collects the resulting local models $\theta_i^{t + 1}$,
      performs an aggregation into a single set of weights, and then conducts
      an optimization step, using these aggregated weights as a proxy to the
      actual gradients of the model:
        - $\hat{\nabla}\theta^{(t)}(\cup_i \mathcal{D}_i) = aggregator(\{\theta_i^{(t + 1)}\}_i)$
        - $\theta^{(t+1)} = \theta^{(t)} - server\_opt(\theta^{(t)}, \hat{\nabla}\theta^{(t)}(\cup_i \mathcal{D}_i))$

As such, a federated optimzation comprises three structural components:

- An [`Aggregator`](../api-reference/aggregator/Aggregator.md), that
  defines how client-wise model updates are combined into the approximate
  gradients of the global model. In vanilla FedAvg, client-wise updates are
  averaged, with optional weighting based on the number of steps taken by
  the clients.
- A client-side `Optimizer`, that defines the algorithm used to conduct
  local stochastic gradient descent steps. In vanilla FedAvg, this is a
  vanilla SGD optimizer with a given learning rate.
- A server-side `Optimizer`, that defines how the aggregated weights are
  transformed into updates to the global model at the end of the round. In
  vanilla FedAvg, the model is replaced with the average of client-wise new
  ones; this is equivalent to using a vanilla SGD optimzier with a 1.0
  learning rate.

In addition to these components, a number of additional hyper-parameters may
intervene, such as the number of training rounds, the number of training steps
per round (which may be defined in number of steps, number of epochs or even in
training duration), or the clients-selection strategy (something which is yet
to be extended in Declearn).

In Declearn, we use the following configuration-specification tools to set up
the federated optimization algorithm, which is entirely defined by the server
and applied by its clients:

- [`FLOptimConfig`](../api-reference/main/config/FLOptimConfig.md) defines the
  aggregator, client-side optimizer and server-side one.
- [`FLRunConfig`](../api-reference/main/config/FLRunConfig.md) defines other
  hyper-parameters, such as the number of rounds and their effort constraints.

If you are not using our orchestration classes, bearing in mind the way how we
have structured and formalized federated learning should be helpful in using
properly our `Optimizer` (and optionally `Aggregator`) APIs as part of your
own processing flow.

## Advanced topics

### `OptiModule` auxiliary variables

#### What are auxiliary variables?

In most cases, optimization is merely a function of a model's weights, its
gradients based on some inputs, some hyper-parameters and some local state
variables. In federated learning however, some algorithms require the use of
state variables or hyper-parameters that may change through time co-dependently
between clients. An example is the [Scaffold](https://arxiv.org/abs/1910.06378)
algorithm, that requires each and every client to apply a specific correction
term to their local gradients, that is updated based on both local computations
and a shared state that depends on the aggregation of client-wise statistics.

Declearn introduces the notion of "auxiliary variables" to cover such cases:

- Each and every `OptiModule` subclass may define a pair of routines to emit
  and receive such variables, which are structured information of any nature.
  This is done by implementing the `collect_aux_var` and `process_aux_var`
  API-defined methods.
- When needed, a pair of server-side and client-side modules may be implemented
  and made to exchange their auxiliary variables with each other. This is done
  by having these paired subclasses share the same (unique) `aux_name` string
  class attribute.
- The packaging and distribution of module-wise auxiliary variables is done by
  `Optimizer.collect_aux_var` and `process_aux_var`, which orchestrate calls to
  the plugged-in modules' methods of the same name.
- Exchanged information is formatted via dedicated `AuxVar` data structures
  (inheriting `declearn.optimizer.modules.AuxVar`) that define how to aggregate
  peers' data, and indicate how to use secure aggregation on top of it (when it
  is possible to do so).

#### OptiModule and Optimizer auxiliary variables API

At the level of any `OptiModule`:

- `OptiModule.collect_aux_var` should output either `None` or an instance of
  a module-specific `AuxVar` subclass wrapping data to be shared.

- `OptiModule.process_aux_var` should expect a dict that has the same structure
  as that emitted by `collect_aux_var` (of this module class, or of a
  counterpart paired one).

At the level of a wrapping `Optimizer`:

- `Optimizer.collect_aux_var` outputs a `{module_aux_name: module_aux_var}`
  dict to be shared.

- `Optimizer.process_aux_var` expects a `{module_aux_name: module_aux_var}`
  dict as well, containing either server-emitted or aggregated clients-emitted
  data.

As a consequence, you should note that:

- An `Optimizer` should not contain multiple auxiliary-variables-using modules
  that have the same `name` or `aux_name`.
- If you are using our `Optimizer` within your own orchestration code (_i.e._
  outside of our `FederatedServer` / `FederatedClient` main classes), it is up
  to you to handle the aggregation of client-wise auxiliary variables into the
  module-wise single instance that the server should receive.

#### Integration to the Declearn FL process

On the server side:

- `Optimizer.collect_aux_var` is called at the start of a training round,
  to emit auxiliary variables that should be sent to participating clients
  alongside with the current model weights.
- `Optimizer.process_aux_var` is called at the end of a training round, to
  process client-emitted information prior to post-processing aggregated
  model weights into updates to the global model's weights.

On the client side:

- `Optimizer.process_aux_var` is called at the start of a training round;
  to process server-emitted information prior to processing the sequence
  of locally-computed gradients in order to update the local model weights.
- `Optimizer.collect_aux_var` is called at the end of a training round, to
  emit auxiliary variables that should be known by the server and used to
  process the aggregated model weights and/or send back new information at
  the start of the next training round.

### The `Regularizer.on_round_start` method

`Regularizer` plug-ins expose an `on_round_start` method, that takes no
argument and is designed to be called at the beginning of each training round.
Depending on the implemented algorithm, is may be used to reset or update some
internal state(s); it is notably used in the FedProx plug-in.

As a developer, you may use this feature in your custom `Regularizer` plug-ins.

As an end-user, you do not need to worry about it when using the Declearn main
orchestration tools. If you write training loops or FL processes on your own,
you should however not omit to call that method.

### Differential Privacy

If you want to use Differential Privacy in Declearn, we strongly advise that
you read [our dedicated guide](./local_dp.md). In short, when DP features are
set up as part of the configuration of a federated learning process conducted
using our main orchestration classes, the adjustment of everything that needs
be is automated, _including_ the setup of a properly-calibrated noise-addition
module as part of the optimizers used.

If you are _not_ using our orchestrating class, _nor_ our DP-specific training
manager (`declearn.main.utils.DPTrainingManager`), then you may want to set up
noise addition as part of your `Optimizer`. This can be done by plugging in a
`NoiseModule` subclass instance (e.g. `GaussianNoiseModule`), which should in
general be placed at the very beginning of your modules pipeline. You are then
in charge of conducting gradient clipping to control the sensitivity of your
inputs, _e.g._ by using the `Model.compute_batch_gradients` `max_norm` kwarg
for sample-level DP, by plugging a `L2Clipping` module prior to the noise
addition one to clip batch-averaged gradients based on their L2 norm, or by
using your own solutions to compute and clip gradients that are input to the
declearn-provided `Optimizer` you use.

### Use in Fed-BioMed

[Fed-BioMed](https://fedbiomed.org/) is another Inria-spawned project that
implements a Federated Learning solution in Python, that specifically targets
cross-silo settings for healthcare institutions and applications.

Starting with version 4.4, Fed-BioMed has been integrating the Declearn
`Optimizer` as an alternative to framework-specific optimization tools.
Moreover, some algorithms specific to Federated Learning, such as FedProx and
Scaffold, are being delegated to Declearn (which takes over custom
implementations that are being deprecated).

This is illustrative of how the Declearn optimization API can benefit other
projects, even without adopting all of the other things Declearn has to offer.
In the case of Fed-BioMed, a custom interface that wraps our Optimizer has been
implemented (with our direct support), enabling to bound our API behind theirs,
and therefore to have both APIs evolve at their own pace in the future.

### How to implement and register your own plug-in

#### Implementing a Regularizer or an OptiModule

To implement a plug-in (whether a `Regularizer` or an `OptiModule`), one should
simply inherit from the abstract base class (or from any intermediate class),
declare the abstract `name` class attribute (with one that is not already in
use) and implement the abstract `run` method, and optionally define, overload
or overcharge any method that needs be.

For pairs of `OptiModule` classes that are designed to be plugged on the client
and server side respectively, and exchange auxiliary variables with each other
between training rounds, one will also need to define the `aux_name` class
attribute, and implement the `process_aux_var` and `collect_aux_var` methods
of both classes so that their specifications match.

#### Type-registration system

For (de)serialization purposes, Declearn maintains so-called type registries,
which are dynamic mappings that link some custom types with some unique names.
This mechanism notably enables specifying optimizer plug-ins as a string and a
dict of kwargs, which are mappable into an instance of the target class - this
can be used when instantiating an `Optimizer`, and is most importantly used to
share the client-side `Optimizer`'s configuration from the orchestrating server
to its clients as part of a Federated Learning process.

Custom, _i.e._ non-declearn-provided plug-in classes should therefore be added
to that registry, so that they can be handled as part of these mechanisms just
as any declearn-provided class would. Luckily, this is automated so that it is
done by default whenever a subclass of `OptiModule` or `Regularizer` is
declared. In other words, **your custom plug-in subtypes will be registered by
default**, under their `name` class attribute (hence the requirement for it to
be unique). However, if for some reason you want to prevent type-registration,
you may do it by adding `register=False` to the inheritance instructions of the
class; e.g. `class MyModule(OptiModule, register=False):`.

Note that for your custom types to be parsable by clients, they will need to
execute the code that declares them as part of the script that sets up and runs
their `declearn.main.FederatedClient`. This can be done by distributing your
types as part of a third-party library (somehow acting as a Declearn add-on)
that is imported as part of clients' script, or by ensuring that their source
code is included as part of these scripts. This may feel complicated, but is in
fact a design choice not to introduce too hastily mechanisms by which clients
would be made to execute arbitrary code or imports based on server demands. As
a result, and for now, we rely on end-users agreeing with each other on what
code will be executed by clients, using processes that are external to Declearn
and expected to match security and trust constraints that are unknown to us and
that ought to differ depending on the application setting.

## Caveats and Open Topics

Here is a non-exhaustive list of things that are not available in Declearn, but
that we are aware of and considering adding in the future. If you want to
contribute ideas, code or new topics to discuss, feel free to let us know,
either by opening a GitHub or GitLab issue, or by sending us an e-mail!

### Layer-wise Learning Rate

In some applications, it may make sense to apply distinct learning rates to
subsets of your model's weights. This is notably something that Torch allows
for deep neural networks. At the moment, such an approach is not possible in
Declearn, as the same learning rate (and optimization algorithms) will be
applied to all of the model weights.

A work-around that may be used, but requires tailoring to your application and
model architecture, would be to implement a custom `OptiModule` that operates
on the input gradients' `Vector.coefs` data array values, filtering them by
name. This would neither be elegant nor practical, but would work.

### Loss-based Learning Rate Scheduling

Learning rate (and weight decay factor) scheduling was recently added to
DecLearn (v2.6.0), via the `Scheduler` API. It is however restricted to
time-based rules, that may build on training steps and/or training rounds
indices to compute scheduled values. In some cases, it may be interesting
to use schedulers that operate based on feedback from the training process
(typically, adjusting the rate when the loss reaches a plateau). This is
not yet possible but may be made available in the future.

There is no simple workaround to that problem, save for using the available
early stopping mechanism to detect plateaus, stop an ongoing experiment,
and then manually relaunch it with a distinct base learning rate, but with
reloaded starting model weights, optimizer states, etc.

## How-Tos

Here is a non-exhaustive list of things that are available in Declearn, but
that new-comers may not find out about as easily as they wish. If you have
questions, or see things that you think should be listed here, feel free to
let us know either by pushing changes to this documentation's source files,
opening an issue on GitLab or GitHub, or sending us an e-mail!

### Gradient-Clipping

In some cases, you might want to clip your batch-averaged gradients, _e.g._ to
prevent exploding gradients issues. This is possible in Declearn, thanks to a
couple of `OptiModule` subclasses: `L2Clipping` (name: `'l2-clipping'`) clips
arrays of weights based on their L2-norm, while `L2GlobalClipping` (name:
`'l2-global-clipping'`) clips all weights based on their global L2-norm (as if
concatenated into a single array).
