# Client Sampling

## Overview

### What is Client Sampling ?

In federated learning, client sampling (also known as client selection or
participant selection) is the process used to decide which clients are chosen to
participate in each training round of the global federated process.

### General capabilities

In DecLearn, this process is handled by a ClientSampler attributed to the
FederatedServer.

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
  its internal state (e.g. compute the value of a selection score for next round
  sampling). This update uses information from the last client replies and from
  the global model (before this round global update).

Each implemented client sampler must indicate if it is compatible with secure
aggregation (secagg).  
For instance, a client sampler that uses in its selection strategy a quantity
computed from clients gradients is logically not compatible with secure
aggregation, as client gradients are obfuscated from the server's perspective by
secagg. If a secagg-incompatible client sampler is instantiated in a secagg
context, an error will be raised.

### Caveats

At the moment in the API, the client sampling can only be performed on the
server side. It means that the server requires the clients to participate in a
given round. A client is currently not able to refuse to participate.

## How to setup and use a ClientSampler

To allow client sampling in your federated learning experiment :

- Instantiate a `ClientSampler` or use a valid client sampler configuration
  (`ClientSamplerConfig` or dictionary with the proper keys)
- Pass it as the `client_sampler` argument in the `FederatedServer` used in your
  experiment

See examples below.

### Available client sampling strategies

- `DefaultClientSampler` : client sampler that always select all registered
  clients

- `UniformClientSampler` : client sampler that randomly selects n clients among
  all following the uniform probability law.

- `WeightedClientSampler` : client sampler that randomly selects n clients among
  all following a probability law built from user-provided weights (if there are
  two clients and client 1 has a weight of 1 and client 2 has a weight of 2 :
  then client 2 is twice as likely to be selected).

- `CriterionClientSampler` : client sampler that selects the n clients that have
  the highest score, according to a specific criterion (often deterministic).
  The criterion has to be specified by the user through an instance of a
  `Criterion` subclass, passed to the sampler at construction. The criterion
  score is usually computed from information in the client training replies and
  from the global model.  
  Available `Criterion` subclasses :
    - `GradientNormCriterion` : the criterion score is the L2-norm of the client "gradients" (updates)

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

    - Arithmetic on criteria is made possible thanks to the subclasses 
    `ConstantCriterion` and `CompositionCriterion`, meaning that it is possible 
    to define a criterion like : 
    `(GradientNormCriterion() + NormalizedDivCriterion())**2 / 2` in a 
    straightforward way.

- `CompositionClientSampler` : client sampler that contains other ones and 
allow to compose the strategies to select clients. Thus, assuming that we have 
5 registered clients, it is possible to define a client sampler performing the 
following selection before each training round :
    - first, selects 2 clients over the 5 based on a deterministic criterion
    - then, selects randomly 1 client over the 3 that remains

To achieve this, we just need to create a `CompositionClientSampler` containing 
a `CriterionClientSampler` and a `UniformClientSampler`. See examples below for 
details.


### Examples

#### Basic example

Instantiate a client sampler that randomly selects 2 clients among all following
a uniform probability law :

```python
from declearn.client_sampler import UniformClientSampler
from declearn.main import FederatedServer

# define your sampler
client_sampler = UniformClientSampler(n_samples=2)

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

Then, everything is ready to use the client sampler in your experiment when the server will run.  
To know how to run a complete experiment, please refer to the 
[Quickstart page](../quickstart.md#12-python-script).

##### Instantiate from configuration / TOML
TODO

##### Instantiate from a dictionary
TODO

#### Criterion-based example
TODO

TODO specs

#### Composition example
TODO

TODO specs

## How to implement a new client sampling method

### Implement a new ClientSampler

TODO link to the following section if criterion is sufficient

### Implement a new Criterion to be used by CriterionClientSampler

TODO