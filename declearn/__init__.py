# coding: utf-8

"""Declearn - a python package for decentralized learning.

Declearn is a framework providing with tools to set up and
run Federated Learning processes. It is being developed by
the MAGNET team of INRIA Lille, with the aim of providing
users with a modular and extensible framework to implement
federated learning algorithms and apply them to real-world
(or simulated) data using any model-defining framework one
might want to use.

Declearn provides with abstractions that enable algorithms
to be written agnostic to the actual computation framework
as well as with workable interfaces that cover some of the
most popular frameworks, such as Scikit-Learn, TensorFlow
and PyTorch.

The package is organized into the following submodules:
* aggregator:
    Model updates aggregating API and implementations.
* communication:
    Client-Server network communications API and implementations.
* data_info:
    Tools to write and extend shareable metadata fields specifications.
* dataset:
    Data interfacing API and implementations.
* main:
    Main classes implementing a Federated Learning process.
* model:
    Model interfacing API and implementations.
* optimizer:
    Framework-agnostic optimizer and algorithmic plug-ins API and tools.
* strategy:
    Interface to gather an Aggregator and a pair of Optimizer into a strategy.
* typing:
    Type hinting utils, defined and exposed for code readability purposes.
* utils:
    Shared utils used (extensively) across all of declearn.
"""

from . import typing
from . import utils
from . import communication
from . import data_info
from . import dataset
from . import model
from . import optimizer
from . import aggregator
from . import strategy
from . import main

__version__ = "2.0.0b2"
