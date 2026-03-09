# Client Sampling

## Overview

### What is Client Sampling ?

In federated learning, client sampling (also known as client selection or
participant selection) is the process used to decide which clients are chosen
to participate in each training round of the global federated process.

### General capabilities

In DecLearn, this process is handled by a `ClientSampler` attributed to the
`FederatedServer`.

In the running phase of the server, the client sampler is called before each
training round to select the client that will be involved. If no client sampler
is provided to the server, all clients registered in the federated process will
be involved in all training rounds by default.

A ClientSampler is binded to a specific strategy to sample clients. For
instance, this strategy can rely on :

- a probability law (e.g. pick two clients randomly)
- user preferences (e.g. pick a specific client more often)
- information from clients model training or the global model (e.g. gradients,
  training time)

DecLearn implements both a generic API for ClientSampler and some practical
samplers with their concrete strategy that are ready-to-use.

### Details

In the FederatedServer, the ClientSampler :

- is setup during the server initialization (hyperparameters, registered
  clients, etc.)
- is called before each training round to select clients that will participate
  to it
- is called in each training round, after receiving client replies, to update
  its internal state (e.g. compute the value of a selection score for next
  round sampling). This update uses information from the last client replies
  and from the global model (before this round global update).

Each implemented client sampler must indicate if it is compatible with secure
aggregation (secagg).  
For instance, a client sampler that uses in its selection strategy a quantity
computed from clients gradients is logically not compatible with secure
aggregation, as client gradients are obfuscated from the server's perspective
by secagg. If a secagg-incompatible client sampler is instantiated in a secagg
context, an error will be raised.

### Caveats

At the moment in the API, the client sampling can only be performed on the
server side. It means that the server requires the clients to participate in a
given round. A client is currently not able to refuse to participate.

## How to setup and use a ClientSampler

To allow client sampling in your federated learning experiment :

- Instantiate a `ClientSampler` or use a valid client sampler configuration
  (`ClientSamplerConfig` or dictionary with the proper keys)
- Pass it as the `client_sampler` argument in the `FederatedServer` used in
  your experiment

See examples below.

### Available client sampling strategies

- `DefaultClientSampler` : client sampler that always select all registered
  clients.

- `UniformClientSampler` : client sampler that randomly selects n clients among
  all following the uniform probability law.

- `WeightedClientSampler` : client sampler that randomly selects n clients
  among all following a probability law built from user-provided weights (if
  there are two clients, and client 1 has a weight of 1 where client 2 has a
  weight of 2 : then client 2 is twice as likely to be selected).

- `CriterionClientSampler` : client sampler that selects the n clients that
  have the highest score, according to a specific criterion (often
  deterministic).
  The criterion has to be specified by the user through an instance of a
  `Criterion` subclass, passed to the sampler at construction. The criterion
  score is usually computed from information in the client training replies and
  from the global model.  
  Available `Criterion` subclasses :
    - `GradientNormCriterion` : the criterion score is the L2-norm of the
    client "gradients" (updates).

    - `NormalizedDivCriterion` : the criterion score is the client "normalized 
    model divergence" computed from client updates and global server weights :
      $$ \frac{1}{|w|} \sum_{j=1}^{|w|} 
      \left| \frac{w_{ij} - \bar{w}_j}{\bar{w}_j} \right| $$
    
    (Where $w$ represents the weights of a model, $\bar{w}$ represents the
    weights of the global model, $w_{ij}$ and $\bar{w}_{j}$ are the $j$ th
    weights of client $i$ and the global model, respectively).

    - `TrainTimeCriterion` : the criterion score is computed from the client 
    training time in the current round. By default, a lower time means a higher 
    criterion score.

    - `TrainTimeHistoryCriterion` : the criterion score is computed from all 
    past client training times (until the current round). For instance, it can 
    be computed from the sum or the average of all past times. By default, a 
    lower aggregated time means a higher criterion score.

    - Composition with math operators is made possible thanks to the subclasses 
    `ConstantCriterion` and `CompositionCriterion`, meaning that it is possible 
    to define a criterion like : 
    `(GradientNormCriterion() + NormalizedDivCriterion())**2 / 2` in a 
    straightforward way.

- `CompositionClientSampler` : client sampler that contains others, allowing to
compose the strategies to select clients. For instance, assuming that we have 
5 registered clients, it is possible to define a client sampler performing the 
following selection before each training round :
    - first, selects 2 clients over the 5 based on a deterministic criterion
    - then, selects randomly 1 client over the 3 that remains

To achieve this, we just need to create a `CompositionClientSampler` containing 
a `CriterionClientSampler` and a `UniformClientSampler`. See the "composition 
example section" below for precise implementation.


### Examples

#### Basic example

Instantiate a client sampler that randomly selects 2 clients among all
following a uniform probability law (seeded with 42) :

```python
from declearn.main import FederatedServer
from declearn.client_sampler import UniformClientSampler

# define your sampler
client_sampler = UniformClientSampler(n_samples=2, seed=42)

# define other objects used in the federated server
model = ...
netwk = ...
optim = ...

# define your server
server = FederatedServer(
    model=model,
    netwk=netwk,
    optim=optim,
    client_sampler=client_sampler,
)
```

Then, everything is ready to use the client sampler in your experiment when the
server will run.  
To know how to run a complete experiment, please refer to the 
[Quickstart page](../quickstart.md#12-python-script).

##### Instantiate from a specification dictionary

DecLearn API supports the use of specifications to build client sampler
instances (thanks to the ClientSampler `from_specs` method) without importing
and using directly the Python objects. You just need to define a dictionary of
valid client sampler specifications and pass it at server's construction.  

Thus, to instantiate the same client sampler as in previous example, but using
a dictionary :
```python
from declearn.main import FederatedServer

# define your sampler specs dictionary
clisamp_specs = {
    "strategy": "uniform",
    "n_samples": 2,
    "seed": 42,
}

# define other objects used in the federated server
model = ...
netwk = ...
optim = ...

# define your server
server = FederatedServer(
    model=model,
    netwk=netwk,
    optim=optim,
    client_sampler=clisamp_specs,
)
```

Note that the `strategy` key is mandatory in all specifications, it allows to
decide which ClientSampler subclass will be instantiated.
To know which value matches your desired sampler, refer to the value of the
`strategy` class attribute (in the class definition). This attribute has to be
defined for each `ClientSampler` subclass.

##### Instantiate from a TOML configuration file

You can also use a ClientSamplerConfig built from a TOML configuration file in
the federated server.
For instance, assuming you have the file `config.toml` below :
```toml
[client_sampler]
strategy = "uniform"

[client_sampler.params]
n_samples = 2
seed = 42
```

You can build a ClientSamplerConfig this way :

```python
from declearn.main import FederatedServer
from declearn.client_sampler import ClientSamplerConfig

# build your sampler configuration
clisamp_config = ClientSamplerConfig.from_toml(
    "config.toml", 
    use_section="client_sampler",
    warn_user=False,
)

# define other objects used in the federated server
model = ...
netwk = ...
optim = ...

# define your server
server = FederatedServer(
    model=model,
    netwk=netwk,
    optim=optim,
    client_sampler=clisamp_config,
)
```

#### Criterion-based example

In this section, you can find an example of how to use a 
`ClientSampler` object and a `Criterion` object to build a criterion client
sampler. Here we use the  L2-norm of the client gradients as criterion :

```python
from declearn.client_sampler import CriterionClientSampler
from declearn.client_sampler.criterion import GradientNormCriterion

client_sampler = CriterionClientSampler(
  n_samples=2, 
  criterion=GradientNormCriterion(),
)
```

Alternatively, the following is an equivalent of this criterion client sampler,
defined as a specifications dictionary :

```python
clisamp_specs = {
    "strategy": "criterion",
    "n_samples": 2,
    "criterion": {
        "name": "gradient_norm",
    },
}
```

#### Composition example

In this section, you can find an example of how to sequentially compose two
client samplers in one, that, in each round :
- first, selects the 2 clients that have the lowest training time in the last
round local training
- then, among the remaining clients, select 1 client randomly based on a
uniform probability law

```python
from declearn.client_sampler import (
  CompositionClientSampler,
  CriterionClientSampler,
  UniformClientSampler,
)
from declearn.client_sampler.criterion import TrainTimeCriterion

cs1 = CriterionClientSampler(n_samples=2, criterion=TrainTimeCriterion())
cs2 = UniformClientSampler(n_samples=1, seed=42)

client_sampler = CompositionClientSampler([cs1, cs2])
```

Alternatively, the following is an equivalent of this composed client sampler,
defined as a specifications dictionary :

```python
clisamp_specs = {
    "strategy": "composition",
    "samplers": [
          {
              "strategy": "criterion",
              "n_samples": 2,
              "criterion": {
                  "name": "train_time",
              },
          },
          {
              "strategy": "uniform",
              "n_samples": 1,
              "seed": 42,
          },
    ],
}
```

## How to implement a new client sampling method

If you want to define your own client selection strategy, you can define a new
`ClientSampler` subclass. However, note that if you just want to define a new
way to select clients using a criterion derived from client replies and the
global model, it may not be necessary to define a new `ClientSampler` subclass,
but only a new `Criterion` subclass (integrated into the existing
`CriterionClientSampler`).  

The two sections below detail the two approaches.

### Implementing a new Criterion to be used by CriterionClientSampler

#### When is it needed ?
You may need to define your own criterion only, instead of a new client sampler
if your selection strategy matches the properties below :
- if you want to select the "best" clients regarding a specific criterion
derived from its last / past local training(s), and optionnaly using the global
model metadata (e.g. weights).  
- if you are able to define your criterion (e.g. through a math formula)
and compute a float-valued score from it (a higher score = a better client).

To understand what is possible with a `Criterion`, the documentation and code
of the existing `Criterion` subclasses may be valuable resources.

#### Inheriting from `Criterion`
To define your own criterion, you must create a subclass of `Criterion`. When
doing so, the following steps are mandatory :
- Defining a string class attribute `name` with a value matching your class 
name, e.g. `"train_time"` for the class `TrainTimeCriterion`.
- Implementing the method `compute` in which your criterion-based selection
logic is defined.

See the `Criterion` API reference and subclasses implementation for more 
details.

### Implementing a new ClientSampler

#### When is it needed ?
You may need to define your own client sampler if you want a specific strategy
involving for example randomness with currently unsupported probability law ;
or any complex strategies that cannot be implemented with the existing DecLearn
client sampling bricks, typically the `CriterionClientSampler` and `Criterion`
APIs. 

#### Inheriting from `ClientSampler`

To define your own client sampler, you must create a subclass of
`ClientSampler`. When doing so, the following steps are mandatory :
- Defining a string class attribute `strategy` with a value matching your class
name, e.g. `"uniform"` for the class `UniformClientSampler`.
- Defining the boolean class property `secagg_compatible` (just returning True
or False) to precise if your strategy is compatible with secure aggregation.
- Implementing the method `cls_sample` in which your custom client sampling
logic is defined.

Optionally, if your sampler needs to update its internal state after each
round : 
- You can implement the method `update` in which you update the sampler
internal state (e.g. criterion score computation). This method is called
by the `FederatedServer` at each global round, after collecting all involved
clients' reply.  
> Note : if no update is needed for your sampler, you don't need to override the
`update` method.

See the `ClientSampler` API reference and subclasses implementation for more
details.
