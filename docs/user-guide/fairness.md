# Fairness

DecLearn comes up with algorithms and tools to measure and (attempt to)
enforce fairness constraints when learning a machine learning model
federatively. This guide introduces what (group) fairness is, which
settings and algorithms are available in DecLearn, how to use them,
and how to implement custom algorithms that fit into the API.

## Overview

### What is Fairness?

Fairness in machine learning is a wide area of research and algorithms that
aim at formalizing, measuring and correcting various algorithmic biases that
are deemed undesirable, notably when they result in models under-performing
for some individuals or groups in a way that is correlated with attributes
such as gender, ethnicity or other socio-demographic characteristics that
can also be the source of unfair discrimination in real life.

Defining what fairness is and how to formalize it is a research topic _per se_,
that has been open and active for the past few years. So is the understanding
of both the causes and consequences of unfairness in machine learning.

### What is Group Fairness?

Group Fairness is one of the main families of approaches to defining fairness
in machine learning. It applies to classification problems, and to data that
can be divided into non-overlapping subsets, designated as sensitive groups,
defined by the intersected values of one or more categorical attributes
(designated as sensitive attributes) and (usually, but not always) the target
label.

For instance, when learning a classifier over a human population, sensitive
attributes may include gender, ethnicity, age groups, etc. Defining relevant
attributes and assigning samples to them can be a sensible issue, which may
motivate a recourse to other families of fairness approaches.

Formally, we can note $\mathcal{Y}$ the set of values for the target label,
$\mathcal{S}$ the set of (intersected) sensitive attribute values, $S$ the
random variable over $\mathcal{S}$ denoting a sample's sensitive attribute
values, and $Y$ the random variable over $\mathcal{Y}$ denoting its true
target label and $\hat{Y}$ its predicted target label by the evaluated
classifier.

Various group fairness definitions exist, that can overall be summarized as
achieving a balance between the group-wise accuracy scores of the evaluated
model. The detail of that balance varies with the definitions; some common
choices include:

- Demographic Parity (also known as Statistical Parity):
    $$
    \forall a, b \in \mathcal{S}, \forall y \in \mathcal{Y},
    \mathbb{P}(\hat{Y} = y | S = a) = \mathbb{P}(\hat{Y} = y | S = b)
    $$
- Accuracy Parity:
    $$
    \forall a \in \mathcal{S}, \forall y \in \mathcal{Y},
    \mathbb{P}(\hat{Y} = y | S = a) = \mathbb{P}(\hat{Y} = y)
    $$
- Equalized Odds:
    $$
    \forall a \in \mathcal{S}, \forall y \in \mathcal{Y},
    \mathbb{P}(\hat{Y} = y | Y = y) = \mathbb{P}(\hat{Y} = y | Y = y, S = a)
    $$

### Fairness in DecLearn

Starting with version 2.6, and following a year-long collaboration with
researcher colleagues to develop and evaluate fairness-enforcing federated
learning algorithms, DecLearn is providing with an API and algorithms that
(attempt to) enforce group fairness by altering the model training itself
(as opposed to pre-processing and post-processing methods, that can be
applied outside of DecLearn).

The dedicated API, shared tools and provided algorithms are implemented
under the `declearn.fairness` submodule, and integrated into the main
`declearn.main.FederatedServer` and `declearn.main.FederatedClient` classes,
enabling to plug a group-fairness algorithm into any federated learning
process.

Currently, the implemented features enable end-users to choose among a
variety of group fairness definitions, to measure it throughout model
training, and optionally to use one of various algorithms that aim at
enforcing fairness while training a model federatively. The provided
algorithms are either taken from the litterature, original algorithms
awaiting publication, or adaptations of algorithms from the centralized
to the federated learning setting.

It is worth noting that the fairness API and DecLearn-provided algorithms
are fully-compatible with the use of secure-aggregation (protecting
fairness-related values such as group-wise sample counts, accuracy and
fairness scores), and of any advanced federated optimization strategy (apart
from some algorithms forcing (with a warning) the choice of aggregation rule).

Local differential privacy can also be used, but the accounting might not
be correct for all algorithms (namely, Fed-FairBatch/FedFB), hence we would
advise careful use after informed analysis of the selected algorithm.

## Details and caveats

### Overall setting

Currently, the DecLearn fairness API is designed so that the fairness being
measured and optimized is computed over the union of all training datasets
held by clients.

The API is designed to be compatible with any number of sensitive groups,
with regimes where individual clients do not necessarily hold samples to
each and every group, and with all group fairness definitions that can be
expressed in a form that was introduced in the FairGrad paper (Maheshwari
& Perrot, 2023). However, some additional restrictions may be enforced by
concrete definitions and/or algorithms.

### Available group-fairness definitions

As of version 2.6.0, DecLearn provides with the following group-fairness
definitions:

- **Accuracy Parity**, achieved when the model's accuracy is independent
  from the sensitive attribute(s) - but not necessarily balanced across
  target labels.
- **Demographic Parity**, achieved when the probability to predict a given
  label is independent from the sensitive attribute(s) - regardless of
  whether that label is accurate or not. _In DecLearn, it is restricted to
  binary classification tasks._
- **Equalized Odds**, achieved when the probability to predict the correct
  label is independent from the sensitive attribute(s).
- **Equality of Opportunity**, which is similar to Equalized Odds but is
  restricted to an arbitrary subset of target labels.

### Available algorithms

As of version 2.6.0, DecLearn provides with the following algorithms, that
can each impose restrictions as to the supported group-fairness definition
and/or number of sensitive groups:

- [**Fed-FairGrad**][declearn.fairness.fairgrad], an adaptation of FairGrad
  (Maheshwari & Perrot, 2023) to the federated learning setting.<br/>
  This algorithm reweighs the training loss based on the current fairness
  levels of the model, so that advantaged groups contribute less than
  disadvantaged ones, and may even contribute negatively (effectively trading
  accuracy off in favor of fairness).
- [**Fed-FairBatch**][declearn.fairness.fairbatch], a custom adaptation of
  FairBatch (Roh et al., 2020) to the federated learning setting.<br/>
  This algorithm alters the way training data batches are drawn, enforcing
  sampling probabilities that are based on the current fairness levels of the
  model, so that advantaged groups are under-represented and disadvantaged
  groups are over-represented relatively to raw group-wise sample counts.
- **FedFB** (Zeng et al., 2022), an arXiv-published alternative adaptation
  of FairBatch that is similar to Fed-FairBatch but introduces further
  formula changes with respect to the original FairBatch.
- [**FairFed**][declearn.fairness.fairfed] (Ezzeldin et al., 2021), an
  algorithm designed for federated learning, with the caveat that authors
  designed it to be combined with local fairness-enforcing algorithms,
  something that is not yet effortlessly-available in DecLearn.<br/>
  This algorithm modifies the aggregation rule based on the discrepancy
  between client-wise fairness levels, so that clients for which the model
  is more unfair weigh more in the overall model updates than clients for
  which the model is fairer.

### Shared algorithm structure

The current API sets up a shared structure for all implemented algorithms,
that is divided between two phases. Each of these comprises a basic part
that is shared across algorithms, and an algorithm-specific part that has
varying computation and communication costs depending on the algorithm.

- The **setup** phase, that occurs as an additional step of the overall
  federated learning initialization phase. During that phase:
    - Each client sends the list of sensitive group definitions for which they
      have samples to the server.
    - The server sends back the ordered list of sensitive group definitions
      across the union of client datasets.
    - Each client communicates the (optionally encrypted) sample counts
      associated with these definitions.
    - The server (secure-)aggregates these sample counts to initialize the
      fairness function on the union of client datasets.
    - Any algorithm-specific additional steps occur. For this, the controllers
      have access to the network communication endpoints and optional secure
      aggregation controllers. On the server side, the `Aggregator` may be
      changed (with a warning). On the client side, side effects may occur
      on the `TrainingManager` (hence altering future training rounds).

- The **fairness round**, that is designed to occur prior to training rounds
  (and implemented as such as part of `FederatedServer`). During that phase:
    - Each client evaluates the fairness of the current (global) model on their
      training set. By default, group-wise accuracy scores are computed and
      sent to the server, while the local fairness scores are computed and
      kept locally. Specific algorithms may change the metrics computed and
      shared. Shared metrics are encrypted when secure aggregation is set up.
    - The server (secure-)aggregates client-emitted metrics. It will usually
      compute the global fairness scores of the model based on group-wise
      accuracy scores.
    - Any algorithm-specific additional steps occur. For this, the controllers
      have access to the network communication endpoints and optional secure
      aggregation controllers. On the client side, side effects may occur on
      the `TrainingManager` (hence altering future training rounds).
    - On both sides, computed metrics are returned, so that they can be
      checkpointed as part of the overall federated learning process.

### A word to fellow researchers in machine learning fairness

If you are an end-user with a need for fairness, we hope that the current
state of things offers a suitable set of ready-for-use state-of-the-art
algorithms. If you are a researcher however, it is likely that you will
run into limitations of the current API at some point, whether because
you need to change some assumptions or use an algorithm that has less
in common with the structure of the currently-available one than we have
anticipated. Maybe you even want to work on something different from the
group fairness family of approaches.

As with the rest of DecLearn features, if you run into issues, have trouble
figuring out how to implement something or have suggestions as to how and/or
why the API should evolve, you are welcome to contact us by e-mail or by
opening an issue on our GitLab or GitHub repository, so that we can figure
out solutions and plan evolutions in future DecLearn versions.

On our side, we plan to keep collaborating with colleagues to design and
evaluate fairness-aware federated learning methods. As in the past, this
will likely happen first in parallel versions of DecLearn prior to being
robustified and integrated in the stable branch - but at any rate, we are
aware and want to make clear that the current API and its limitations are
but the product of a first iteration to come up with solid implementations
of existing algorithms and lay the ground for more research-driven iterations.

### Caveats and future work

The current API was abstracted from an ensemble of concrete algorithms.
As such, it may be limiting for end-users that would like to implement
alternative algorithms to tackle fairness-aware federated learning.

The current API assumes that fairness is to be estimated over the training
dataset of clients, and without client sampling. More generally, available
algorithms are mostly designed assuming that all clients participate to
each fairness evaluation and model training round. The current API offers
some space to use and test algorithms with client sampling, and with limited
computational effort put into the fairness evaluation, but it is not as
modular as one might want, in part because we believe further theoretical
analysis of the implications at hand is required to inform implementation
choices.

The current API requires fairness definitions to be defined using the form
introduced in the FairGrad paper. While we do believe this form to be a clever
choice, and use it as leverage to be consistent in the way computations are
performed and limit the room left for formula implementation errors, we are
open to revising it in case we are signalled that definitions that do not fit
into that form are desired by someone using DecLearn.

## How to set up fairness-aware federated learning

### Description

Making a federated learning process fairness-aware only requires a couple
of changes to the code executed by the server and clients.

First, clients need to interface their training data using a subclass of
`declearn.fairness.api.FairnessDataset`, which extends the base
`declearn.dataset.Dataset` API to define sensitive attributes and access
their definitions, sample counts and group-wise sub-dataset. As with the
base `Dataset`, DecLearn provides with a `FairnessInMemoryDataset` that
is suited for tabular data that fits in memory. For other uses, end-users
need to write their own subclass, implementing the various abstract methods
regarding metadata fetching and data batching.

Second, the server needs to select and configure a fairness algorithm, and
make it part of the federated optimization configuration (wrapped or parsed
using `declearn.main.config.FLOptimConfig`). Some hyper-parameters enabling
to control computational efforts put into evaluating the model's fairness
throughout training may also be specified as part of the run configuration
(`declearn.main.config.FLRunConfig`), which is otherwise auto-filled to use
the same batch size as in evaluation rounds, and the entire training dataset
to compute as robust fairness estimates as possible.

### Hands-on example

For this example, we are going to use the
[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
dataset, for which we already provide a [base example](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/develop/examples/heart-uci/)
implemented via Python scripts.

This is a binary classification task, for which we are going to define a single
binary sensitive attribute: patients' biological sex.

**1. Interface training datasets to define sensitive groups**

On the client side, we simply need to wrap the training dataset as an
`InMemoryFairnessDataset` rather than a base `InMemoryDataset`. This
results in a simple edit in steps (1-2) of the initial
[client script](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/develop/examples/heart-uci/client.py):

```python
from declearn.fairness.core import InMemoryFairnessDataset

train = InMemoryFairnessDataset(
    data=data.iloc[:n_tr],  # unchanged
    target=target,          # unchanged
    s_attr=["sex"],         # define sensitive attribute(s)
    sensitive_target=True,  # define fairness relative to Y x S (default)
)
```

instead of the initial

```python
train = InMemoryDataset(
    data=data.iloc[:n_tr],
    target=target,
)
```

Note that:

- The validation dataset does not need to be wrapped as a `FairnessDataset`,
  as fairness is only evaluated on the training dataset during the process.
- By default, sensitive attribute columns are _not_ excluded from feature
  columns. The `f_cols` parameter may be used to exclude it. Alternatively,
  one could pop the sensitive attribute column(s) from `data` and pass the
  resulting DataFrame or numpy array directly as `s_attr`.
- It is important that clients order the sensitive attributes in the same
  order. This is also true of feature columns in general in DecLearn.

**2. Configure a fairness algorithm on the server side**

On the server side, a `declearn.fairness.api.FairnessControllerServer` subclass
must be selected, instantiated and plugged into the `FLOptimConfig` object (or
dict input as `optim` instantiation parameter to `FederatedServer`).

For instance, to merely measure the model's fairness wihtout altering the
training process (typically to assess the fairness of a baseline approach),
one may edit step 2 of the initial
[server script](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/develop/examples/heart-uci/server.py)
as follows:

```python
from declearn.fairness.monitor import FairnessMonitorServer

fairness = FairnessMonitorServer(
    f_type="demographic_parity",  # choose any fairness definition
)

optim = FLOptimConfig.from_params(
    aggregator=aggregator,  # unchanged
    client_opt=client_opt,  # unchanged
    server_opt=server_opt,  # unchanged
    fairness=fairness,      # set up the fairness monitoring algorithm
)
```

To use a fairness-enforcing algorithm, simply instantiate another type of
controller. For instance, to use Fed-FairGrad:

```python
from declearn.fairness.fairgrad import FairgradControllerServer

fairness = FairgradControllerServer(
    f_type="demographic_parity",  # choose any fairness definition
    eta=0.1,  # adjust this based on the SGD learning rate and empirical tuning
    eps=0.0,  # change this to configure epsilon-fairness
)
```

Equivalently, the choice of fairness controller class and parameters may be
specified using a configuration dict (that may be parsed from a TOML file):

```python
fairness = {
    # mandatory parmaeters:
    "algorithm": "fairgrad",         # name of the algorithm
    "f_type": "demographic_parity",  # name of the group-fairness definition
    # optional, algorithm-dependent hyper-parameters:
    "eta": 0.1,
    "eps": 0.0,
}
```

Notes:

- `declearn.fairness.core.list_fairness_functions` may be used to review all
  available fairness definitions and their registration name.
- `FairnessControllerServer` all expose an optional `f_args` parameter that
  can be used to parameterize the fairness definition, _e.g._ to define
  target labels on which to focus when using "equality_of_opportunity".
- `FLRunConfig` exposes an optional `fairness` field that can be used to
  reduce the computational effort put in evaluating fairness. Otherwise,
  it is automatically filled with a mix of default values and values parsed
  from the training and/or evaluation round configuration.

## How to implement custom fairness definitions and algorithms

As with most DecLearn components, group fairness definitions and algorithms
are designed to be extendable by third-party developers and end-users. On
the one hand, new definitions of group fairness may be implemented by
subclassing `declearn.fairness.api.FairnessFunction`. On the other hand,
new algorithms for fairness-aware federated learning may be implemented by
subclassing both `declearn.fairness.api.FairnessControllerServer` and
`declearn.fairness.api.FairnessControllerClient`.

### Implement a new group fairness definition

Group fairness definitions are implemented by subclassing the API-defining
abstract base class [declearn.fairness.api.FairnessFunction][]. Extensive
details can be found in the API docs of that class. Overall, the process
is the following:

- Declare a `FairnessFunction` subclass.
- Define its `f_type` string class attribute, that must be unique across
  subclasses, and will be used to type-register this class and make it
  available in controllers.
- Define its `compute_fairness_contants` method, which must return $C_k^{k'}$
  constants defining the group-wise fairness level computations based on
  group-wise sample counts.

The latter method, and overall API, echo the generic formulation for fairness
functions introduced in the FairGrad paper (Maheshwari & Perrot, 2023). If
this is limiting for your application, please let us know. If you are using
definitions that are specific to your custom algorithm, you may "simply" tweak
around the API when implementing controllers (see the following section).

### Implement a new fairness-enforcing algorithm

Fairness enforcing algorithms are implemented by subclassing API-defining
abstract base classes [declearn.fairness.api.FairnessControllerServer][]
and [declearn.fairness.api.FairnessControllerClient][]. Extensive details
can be found in the API docs of these classes. Overall, the process is
the following:

- Declare the paired subclasses.
- Define their `algorithm` string class attribute, which must be the same
  for paired classes, and distinct from that of other subclasses. It is
  used to register the types and make serializable generic instantation
  instructions.
- Define their respective `finalize_fairness_setup` methods, to take any
  algorithm-specific steps once sensitive group definitions and sample
  counts have been exchanged.
- Define their respected `finalize_fairness_round` methods, to take any
  algorithm-specific steps once fairness-related metrics have been computed
  by clients and (secure-)aggregated by the server.
- Optionally, overload or override the client's `setup_fairness_metrics`
  method, that defines the fairness-related metrics being computed and
  shared as part of fairness rounds.

When overloading the `__init__` method of subclasses, you may add additional
restrictions as to the supported fairness definitions and/or number and
nature of sensitive groups.

## References

- Maheshwari & Perrot (2023).
  FairGrad: Fairness Aware Gradient Descent.
  [https://openreview.net/forum?id=0f8tU3QwWD]()
- Roh et al. (2020).
  FairBatch: Batch Selection for Model Fairness.
  [https://arxiv.org/abs/2012.01696]()
- Zeng et al. (2022).
  Improving Fairness via Federated Learning.
  [https://arxiv.org/abs/2110.15545]()
- Ezzeldin et al. (2021).
  FairFed: Enabling Group Fairness in Federated Learning
  [https://arxiv.org/abs/2110.00857]()
