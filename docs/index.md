# Declearn: a modular and extensible framework for Federated Learning

## Introduction

[declearn](https://magnet.gitlabpages.inria.fr/declearn/docs/latest/)
is a python package providing with a framework to perform federated
learning, i.e. to train machine learning models by distributing computations
across a set of data owners that, consequently, only have to share aggregated
information (rather than individual data samples) with an orchestrating server
(and, by extension, with each other).

The aim of `declearn` is to provide both real-world end-users and algorithm
researchers with a modular and extensible framework that:

- builds on **abstractions** general enough to write backbone algorithmic code
  agnostic to the actual computation framework, statistical model details
  or network communications setup
- designs **modular and combinable** objects, so that algorithmic features, and
  more generally any specific implementation of a component (the model, network
  protocol, client or server optimizer...) may easily be plugged into the main
  federated learning process - enabling users to experiment with configurations
  that intersect unitary features
- provides with functioning tools that may be used **out-of-the-box** to set up
  federated learning tasks using some popular computation frameworks (scikit-
  learn, tensorflow, pytorch...) and federated learning algorithms (FedAvg,
  Scaffold, FedYogi...)
- provides with tools that enable **extending** the support of existing tools
  and APIs to custom functions and classes without having to hack into the
  source code, merely adding new features (tensor libraries, model classes,
  optimization plug-ins, orchestration algorithms, communication protocols...)
  to the party

At the moment, `declearn` has been focused on so-called "centralized" federated
learning that implies a central server orchestrating computations, but it might
become more oriented towards decentralized processes in the future, that remove
the use of a central agent.

## Explore the documentation

The documentation is structured this way:

- [Installation guide](./setup.md):<br/>
  Learn how to set up for and install declearn.
- [Quickstart example](./quickstart.md):<br/>
  See in a glance what end-user declearn code looks like.
- [User guide](./user-guide/index.md):<br/>
  Learn about declearn's take on Federated Learning, its current capabilities,
  how to implement your own use case, and the API's structure and key points.
- [API Reference](./api-reference/index.md):<br/>
  Full API documentation, auto-generated from the source code.
- [Developer guide](./devs-guide/index.md):<br/>
  Information on how to contribute, codings rules and how to run tests.

## Copyright

Declearn is an open-source software developed by people from the
[Magnet](https://team.inria.fr/magnet/) team at [Inria](https://www.inria.fr/).

### Authors

Current core developers are listed under the `pyproject.toml` file. A more
detailed acknowledgement and history of authors and contributors to declearn
can be found in the `AUTHORS` file.

### License

Declearn distributed under the Apache-2.0 license. All code files should
therefore contain the following mention, which also applies to the present
README file:
```
Copyright 2025 Inria (Institut National de Recherche en Informatique
et Automatique)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
