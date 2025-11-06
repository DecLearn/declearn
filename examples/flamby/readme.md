# Demo training task : FLamby (TCGA_BRCA)

## Overview

**We are going to train a common model between three simulated clients on the
[TCGA_BRCA dataset](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_tcga_brca/README.md)
provided by [FLamby](https://github.com/owkin/FLamby)**. The input of the model
is a set of clinical tabular data, and the task involved is a survival
analysis.  
We use the FLamby API to manipulate the dataset and instantiate the model.

## Setup

To be able to experiment with this tutorial:

- Clone the declearn repo (you may specify a given release branch or tag):

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn2.git declearn
```

- Clone the FLamby repo:

```bash
git clone https://github.com/owkin/FLamby.git
```

- Create a dedicated virtual environment.

- Install declearn in it from the local repo:

```bash
cd declearn && pip install ".[websockets,torch]" && cd ..
```

- Install FLamby (in editable mode) in it from the local repo:

```bash
cd FLamby && pip install -e ".[tcga]" && cd ..
```

## Contents

This script runs a FL experiment using FLamby TCGA_BRCA dataset. The folder is
structured the following way:

```
flamby/
│   generate_ssl.py - generate self-signed ssl certificates
│   metric.py       - Metric implementation specific to this learning task
|   run_client.py   - set up and launch a federated-learning client
│   run_demo.py     - simulate the entire FL process in a single session
│   run_server.py   - set up and launch a federated-learning server
└─── results_<time> - saved results from training procedures
```

## Run training routine

The simplest way to run the demo is to run it locally, using multiprocessing.

### Locally, for testing and experimentation

**To simply run the demo**, use the bash command below.  
You will need to **press y** when prompted if you accept the dataset license agreement.

```bash
cd declearn/examples/flamby/
python run_demo.py  # note: python declearn/examples/flamby/run_demo.py works as well
```

The `run_demo.py` scripts collects the server and client routines defined under
the `run_server.py` and `run_client.py` scripts, and runs them concurrently
under a single python session using multiprocessing.

This is the easiest way to launch the demo, e.g. to see the effects of tweaking
some learning parameters (by editing the `run_server.py` script).

### On separate terminals or machines

For something closer to real life implementation, i.e. to run the examples from
different terminals or machines, you can draw inspiration from the dedicated
section in [the MNIST example](../mnist/readme.md) which mainly uses the same
scripts as here.  
For details on scripts usage, you can use the command
`python [SCRIPT_NAME].py --help`.

## Notes

- FLamby provides several datasets. Here, for simplification purpose, we provide
  a demo that uses only one of them : the TCGA_BRCA dataset.  
  If you want to adapt this example to another FLamby dataset, especially one
  that requires a proper download, you may find useful resources in
  [FLamby quickstart](https://github.com/owkin/FLamby/blob/main/Quickstart.md).

- In particular, this demo only supports up to 6 clients.

- In this FLamby TCGA_BRCA example, we also provide an example of how to
  implement a subclass of Declearn `Metric` abstraction (see `metric.py`). To
  build it, we started from
  [FLamby metric function for TCGA_BRCA](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_tcga_brca/metric.py),
  and then we adapted it for a Federated context following the
  [Declearn Metric API](https://gitlab.inria.fr/magnet/declearn/declearn2/-/blob/develop/declearn/metrics/_api.py?ref_type=heads).  
  You may need to do a similar implementation if you want to use another FLamby
  dataset and if the metric involved is not already implemented by Declearn (see
  the
  [already-implemented metric](https://magnet.gitlabpages.inria.fr/declearn/docs/latest/api-reference/metrics/)).
