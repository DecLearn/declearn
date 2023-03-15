# Unit tests and code analysis

Unit tests, as well as more-involved functional ones, are implemented under
the `test/` folder of the declearn gitlab repository. Functional tests are
isolated in the `test/functional/` subfolder to enable one to easily exclude
them in order to run unit tests only.

Tests are implemented using the [PyTest](https://docs.pytest.org) framework,
as well as some third-party plug-ins that are automatically installed with
the package when using `pip install declearn[tests]`.

Additionally, code analysis tools are configured through the `pyproject.toml`
file, and used to control code quality upon merging to the main branch. These
tools are [black](https://github.com/psf/black) for code formatting,
[pylint](https://pylint.pycqa.org/) for overall static code analysis and
[mypy](https://mypy.readthedocs.io/) for static type-cheking.


## Running the unit tests suite

### Running the test suite using tox

The third-party [tox](https://tox.wiki/en/latest/) tool may be used to run
the entire test suite within a dedicated virtual environment. Simply run `tox`
from the commandline with the root repo folder as working directory. You may
optionally specify the python version(s) with which you want to run tests.

```bash
tox           # run with default python 3.8
tox -e py310  # override to use python 3.10
```

Note that additional parameters for `pytest` may be passed as well, by adding
`--` followed by any set of options you want at the end of the `tox` command.
For example, to use the declearn-specific `--fulltest` option (see the section
below), run:

```bash
tox [tox options] -- --fulltest
```

The tests pipeline specified under the `tox.ini` file runs the following:

- install declearn in an isolated environment
- run unit tests
- run functional tests
- run pylint on declearn, then on the tests' code
- run mypy on declearn
- run black on declearn, in check mode

### Running unit tests using pytest

To run all the tests, simply use:

```bash
pytest test
```

To run the tests under a given module (here, "model"):

```bash
pytest test/model
```

To run the tests under a given file (here, "test_regression.py"):

```bash
pytest test/functional/test_regression.py
```

Note that by default, some test scenarios that are considered somewhat
superfluous~redundant will be skipped in order to save time. To avoid
skipping these, and therefore run a more complete test suite, add the
`--fulltest` option to pytest:

```bash
pytest --fulltest test  # or any more-specific target you want
```

For more details on how to run targetted tests, please refer to the
[pytest](https://docs.pytest.org/) documentation.

You may also arguments to compute and export coverage statistics, using the
[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/index.html) plug-in:

```bash
# Run all tests and export coverage information in HTML format.
pytest --cov=declearn --cov-report=html tests/
```

## Running black to format the code

The [black](https://github.com/psf/black) code formatter is used to enforce
uniformity of the source code's formatting style. It is configured to have
a maximum line length of 79 (as per [PEP 8](https://peps.python.org/pep-0008/))
and ignore auto-generated protobuf files, but will otherwise modify files
in-place when executing the following commands from the repository's root
folder:

```bash
black declearn  # reformat the package
black test      # reformat the tests
```

Note that it may also be called on individual files or folders.
One may "blindly" run black, however it is actually advised to have a look
at the reformatting operated, and act on any readability loss due to it. A
couple of advice:

1. Use `#fmt: off` / `#fmt: on` comments sparingly, but use them.
<br/>It is totally okay to protect some (limited) code blocks from
reformatting if you already spent some time and effort in achieving a
readable code that black would disrupt. Please consider refactoring as
an alternative (e.g. limiting the nest-depth of a statement).

2. Pre-format functions and methods' signature to ensure style homogeneity.
<br/>When a signature is short enough, black may attempt to flatten it as a
one-liner, whereas the norm in declearn is to have one line per argument,
all of which end with a trailing comma (for diff minimization purposes). It
may sometimes be necessary to manually write the code in the latter style
for black not to reformat it.

Finally, note that the test suite run with tox comprises code-checking by
black, and will fail if some code is deemed to require alteration by that
tool. You may run this check manually:

```bash
black --check declearn  # or any specific file or folder
```

## Running pylint to check the code

The [pylint](https://pylint.pycqa.org/) linter is expected to be used for
static code analysis. As a consequence, `# pylint: disable=[some-warning]`
comments can be found (and added) to the source code, preferably with some
indication as to the rationale for silencing the warning (or error).

A minimal amount of non-standard hyper-parameters are configured via the
`pyproject.toml` file and will automatically be used by pylint when run
from within the repository's folder.

Most code editors enable integrating the linter to analyze the code as it is
being edited. To lint the entire package (or some specific files or folders)
one may simply run `pylint`:

```bash
pylint declearn  # analyze the package
pylint test      # analyze the tests
```

Note that the test suite run with tox comprises the previous two commands,
which both result in a score associated with the analyzed code. If the score
does not equal 10/10, the test suite will fail - notably preventing acceptance
of merge requests.

## Running mypy to type-check the code

The [mypy](https://mypy.readthedocs.io/) linter is expected to be used for
static type-checking code analysis. As a consequence, `# type: ignore` comments
can be found (and added) to the source code, as sparingly as possible (mostly,
to silence warnings about untyped third-party dependencies, false-positives,
or locally on closure functions that are obvious enough to read from context).

Code should be type-hinted as much and as precisely as possible - so that mypy
actually provides help in identifying (potential) errors and mistakes, with
code clarity as final purpose, rather than being another linter to silence off.

A minimal amount of parameters are configured via the `pyproject.toml` file,
and some of the strictest rules are disabled as per their default value (e.g.
Any expressions are authorized - but should be used sparingly).

Most code editors enable integrating the linter to analyze the code as it is
being edited. To lint the entire package (or some specific files or folders)
one may simply run `mypy`:

```bash
mypy declearn
```

Note that the test suite run with tox comprises the previous command. If mypy
identifies errors, the test suite will fail - notably preventing acceptance
of merge requests.
