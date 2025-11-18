# Demo NLP training task : HuggingFace transformer on IMDb dataset

## Overview

This is a demo of a Natural Language Processing (NLP) training task with Declearn.  
**We are going to train a common model between two simulated clients on the
[IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
provided by HuggingFace**.  

This dataset contains texts that are reviews of highly polar movies. They are associated with labels : 0 for a negative review and 1 for a positive review. Indeed, the task involved is binary sentiment classification.  

We fine-tune the [DistilBERT base (uncased)](https://huggingface.co/distilbert/distilbert-base-uncased) model, a compact transformer architecture with approximately 67 million parameters. Specifically, we use its [sequence classification variant](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertForSequenceClassification) for this task.

## Setup

To be able to experiment with this tutorial:

- Clone the declearn repo (you may specify a given release branch or tag):

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn2.git declearn
```

- Create a dedicated virtual environment.

- Install declearn in it from the local repo:

```bash
cd declearn && pip install ".[websockets,torch,nlp]" && cd ..
```

## Contents

The main script `run_demo.py` runs a FL experiment using IMDb dataset. The folder is
structured the following way:

```
nlp/
│   dataset.py      - contains a torch Dataset implementation adapted to this task
│   generate_ssl.py - generate self-signed ssl certificates
│   model.py        - contains a torch module implementation adapted to this task
|   prepare_data.py - fetch and split the IMDb dataset for FL use
|   run_client.py   - set up and launch a federated-learning client
│   run_demo.py     - simulate the entire FL process in a single session
│   run_server.py   - set up and launch a federated-learning server
└─── data           - data folder, containing raw and split IMDb data
└─── results_<time> - saved results from training procedures
```

## Run training routine

The simplest way to run the demo is to run it locally, using multiprocessing.

### Locally, for testing and experimentation

**To simply run the demo**, use the bash command below.  
You will need to **press y** when prompted if you accept the dataset license agreement.

```bash
cd declearn/examples/nlp/
python run_demo.py  # note: python declearn/examples/nlp/run_demo.py works as well
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

- This example performs the fine-tuning (a few steps at least) of a transformer model. As the number of trained parameters remains substantial, it requires more resources and execution time than other quick examples of this project. Therefore, we recommend running this example on a machine with a GPU. By default, the example is configured to use one, if available.
- For execution time purpose, we have used a minimal configuration (few clients, few FL rounds, few data), if you adapt this example, it should be necessary to change some parameters to get proper results. 
- As is, running on a GPU, the example takes several tens of minutes to complete.
