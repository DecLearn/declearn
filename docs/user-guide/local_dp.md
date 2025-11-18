# Local Differential Privacy

## Basics

`declearn` comes with the possibility to train models using local differential
privacy, as described in the centralized case by Abadi et al, 2016,
[Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133).
This means that training can provide per-client privacy guarantees with regard
to the central server.

In practice, this can be done by simply adding a privacy field to the config
file, object or input dict to the `run` method of `FederatedServer`. Taking
the Heart UCI example, one simply has one line to add to the server-side
script (`examples/heart-uci/server.py`) in order to implement local DP,
here using Renyi-DP with epsilon=5, delta=0.00001 and a sample-wise gradient
clipping parameter that binds their euclidean norm below 3:

```python
# These are the last statements in the `run_server` function.
run_cfg = FLRunConfig.from_params(
    # The following lines come from the base example:
    rounds=20,
    register={"min_clients": nb_clients},
    training={"batch_size": 30, "drop_remainder": False},
    evaluate={"batch_size": 50, "drop_remainder": False},
    early_stop={"tolerance": 0.0, "patience": 5, "relative": False},
    # DP-specific instructions (in their absence, do not use local DP):
    privacy={"accountant": "rdp", "budget": (5, 10e-5), "sclip_norm": 3},
)
server.run(run_cfg)  # this is unaltered
```

This implementation can breach privacy garantees for some standard model
architecture and training processes, see the _Warnings and limits_ section.

## More details on the backend

Implementing local DP requires to change four key elements, which are
automatically handled in declearn based on the provided privacy configuration:

- **Add a privacy accountant**. We use the `Opacus` library, to set up a
privacy accountant. The accountant is used in two key ways :
    - To calculate how much noise to add to the gradient at each trainig step
    to provide an $`(\epsilon-\delta)`$-DP guarantee over the total number of
    steps planned. This is where the heavily lifting is done, as estimating
    the tighest bounds on the privacy loss is a non-trivial problem. We default
    to the Renyi-DP accountant used in the original paper, but Opacus provides
    an evolving list of options, since this is an active area of research. For
    more details see the documentation of `declearn.main.utils.PrivacyConfig`.
    - To keep track of the privacy budget spent as training progresses, in
    particular in case of early stopping.

- **Implement per-sample gradient clipping**. Clipping bounds the sensitivity
of samples' contributions to model updates. It is performed using the
`max_norm` parameter of `Model.compute_batch_gradients`.

- **Implement noise-addition to applied gradients**. A gaussian noise with a
tailored variance is drawn and added to the batch-averaged gradients based on
which the local model is updated at each and every training step.

- **Use Poisson sampling to draw batches**. This is done at the `Dataset` level
using the `poisson` argument of `Dataset.generate_batches`.
    - As stated in the Opacus documentation, "Minibatches should be formed by
    uniform sampling, i.e. on each training step, each sample from the dataset
    is included with a certain probability p. Note that this is different from
    standard approach of dataset being shuffled and split into batches: each
    sample has a non-zero probability of appearing multiple times in a given
    epoch, or not appearing at all."
    - For more details, see Zhu and Wang, 2019,
    [Poisson Subsampled Renyi Differential Privacy](http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf)

## Warnings and limits

Under certain model and training specifications, two silent breaches of formal
privacy guarantees can occur. Some can be handled automatically if working
with `torch`, but need to be manually checked for in other frameworks.

- **Neural net layers that breach DP**. Standard architectures can lead
to information leaking between batch samples. Know examples include batch
normalization layers, LSTM, and multi-headed attention modules. In `torch`,
checking a module for DP-compliance can be done using Opacus, by running:

  ```python
  #given an NN.module to be tested
  from opacus import PrivacyEngine
  dp_compatible_module = PrivacyEngine.get_compatible_module(module)
  ```

- **Gradient accumulation**. This feature is not used in standard declearn
models and training tools, but users that might try to write custom hacks
to simulate large batches by setting a smaller batch size and executing the
optimization step every N steps over the accumulated sum of output gradients
should be aware that this is not compatible with Poisson sampling.

Finally, note that at this stage the DP implementation in declearn is taken
directly from the centralized training case, and as such does not account for
nor make use of some specifities of the Federated Learning process, such as
privacy amplification by iteration.
