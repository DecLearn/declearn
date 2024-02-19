# Installation guide

This guide provides with all the required information to install `declearn`.

**TL;DR**:<br/>
If you want to install the latest stable version with all of its optional
dependencies, simply run `pip install declearn[all]` from your desired
python (preferably virtual) environment.

**Important note**:<br/>
When running a federated process with DecLearn, the server and all clients
should use the same `major.minor` version; otherwise, clients' registration
will fail verbosely, prompting to install the same version as the server's.

## Requirements

- python >= 3.8
- pip

Third-party requirements are specified (and automatically installed) as part
of the installation process, and may be consulted from the `pyproject.toml`
file.

## Optional requirements

Some third-party requirements are optional, and may not be installed. These
are also specified as part of the `pyproject.toml` file, and may be divided
into two categories:<br/>
(a) dependencies of optional, applied declearn components (such as the PyTorch
and Tensorflow tensor libraries, or the gRPC and websockets network
communication backends) that are not imported with declearn by default<br/>
(b) dependencies for developers, e.g. to run tests on the package (mainly
pytest and some of its plug-ins), or building its documentation (with mkdocs)

The second category is more developer-oriented, while the first may or may not
be relevant depending on the use case to which you wish to apply `declearn`.

In the `pyproject.toml` file, the `[project.optional-dependencies]` table
`all` lists all dependencies from the first category, with additional tables
redundantly listing them thematically, enabling end-users to cherry-pick the
optional components they want to install. For developers, the "tests" and
"docs" tables specify tooling dependencies.

## Using a virtual environment (optional)

It is generally advised to use a virtual environment, to avoid any dependency
conflict between declearn and packages you might use in separate projects. To
do so, you may for example use python's built-in
[venv](https://docs.python.org/3/library/venv.html), or the third-party tool
[conda](https://docs.conda.io/en/latest/).

Venv instructions (example):

```bash
python -m venv ~/.venvs/declearn
source ~/.venvs/declearn/bin/activate
```

Conda instructions (example):

```bash
conda create -n declearn python=3.8 pip
conda activate declearn
```

_Note: at the moment, conda installation is not recommended, because the
package's installation is made slightly harder due to some dependencies being
installable via conda while other are only available via pip/pypi, which caninstall
lead to dependency-tracking trouble._

## Installation

### Install from PyPI

Stable releases of the package are uploaded to
[PyPI](https://pypi.org/project/declearn/), enabling one to install with:

```bash
pip install declearn  # optionally with version constraints and/or extras
```

### Install from source

Alternatively, to install from source, one may clone the git repository (or
download the source code from a release) and run `pip install .` from its
root folder.

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn.git
cd declearn
pip install .  # or pip install -e .
```

### Install extra dependencies

To also install optional requirements, add the name of the extras between
brackets to the `pip install` command, _e.g._ running one of the following:

```bash
# Examples of cherry-picked installation instructions.
pip install declearn[grpc]   # install dependencies to use gRPC communications
pip install declearn[torch]  # install `declearn.model.torch` dependencies
pip install declearn[tensorflow,torch]  # install both tensorflow and torch

# Instructions to install bundles of optional components.
pip install declearn[all]    # install all extra dependencies, save for testing
pip install declearn[all,tests]  # install all extra and testing dependencies
```

### Notes

- If you are not using a virtual environment, select carefully the `pip` binary
  being called (e.g. use `python -m pip`), and/or add a `--user` flag to the
  pip command.
- Developers may have better installing the package in editable mode, using
  `pip install -e .` from the repository's root folder.
- If you are installing the package within a conda environment, it may be
  better to run `pip install --no-deps declearn` so as to only install the
  package, and then to manually install the dependencies listed in the
  `pyproject.toml` file, using `conda install` rather than `pip install`
  whenever it is possible.
- On some systems, the square brackets used our pip install are not properly
  parsed. Try replacing `[` by `\[` and `]` by `\]`, or putting the instruction
  between quotes (`pip install "declearn[...]"`).
- Regarding Torch: declearn currently supports both late 1.10-1.13 versions and
  2.X ones. You may use either one freely, but may run into issues regarding
  co-dependent versions of `torch`, `functorch` (in 1.10-1.13) and `opacus` (if
  you want to use [differential privacy](./user-guide/local_dp.md) features).
  You may use the `torch1` or `torch2` extra dependency specifier to explicitly
  target either the 1.13 or latest 2.X torch version and install the proper
  versions of the other packages (including opacus).
