# Demo training task : MNIST in Quickrun Mode

## Overview

**We are going to use the declearn-quickrun tool to easily run a simulated
federated learning experiment on the classic
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)**. The input of the model
is a set of images of handwritten digits, and the model needs to determine to
which digit between $0$ and $9$ each image corresponds.

## Setup

A Jupyter Notebook tutorial is provided, that you may import and run on Google
Colab so as to avoid having to set up a local python environment.

Alternatively, you may run the notebook on your personal computer, or follow
its instructions to install declearn and operate the quickrun tools directly
from a shell command-line.

## Contents

This example's folder is structured the following way:

```
mnist/
│    config.toml - configuration file for the quickrun FL experiment
|    mnist.ipynb - tutorial for this example, as a jupyter notebook
|    model.py    - python file declaring the model to be trained
└─── data_iid    - mnist data generated with `declearn-split`
└─── results_*   - results generated after running `declearn-quickrun`
```
