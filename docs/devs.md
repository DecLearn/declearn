# Developer Guide

## Contributions

Contributions to `declearn` are welcome, whether to provide fixes, suggest
new features (_e.g._ new subclasses of the core abstractions) or even push
forward framework evolutions and API revisions.

To contribute directly to the code (beyond posting issues on gitlab), please
create a dedicated branch, and submit a **Merge Request** once you want your
work reviewed and further processed to end up integrated into the package.

The **git branching strategy** is the following:

- The 'develop' branch is the main one and should receive all finalized changes
  to the source code. Release branches are then created and updated by cherry-
  picking from that branch. It therefore acts as a nightly stable version.
- The 'rX.Y' branches are release branches for each and every X.Y versions.
  For past versions, these branches enable pushing patches towards a subminor
  version release (hence being version `X.Y.(Z+1)-dev`). For future versions,
  these branches enable cherry-picking commits from main to build up an alpha,
  beta, release-candidate and eventually stable `X.Y.0` version to release.
- Feature branches should be created at will to develop features, enhancements,
  or even hotfixes that will later be merged into 'main' and eventually into
  one or multiple release branches.
- It is legit to write up poc branches, as well as to split the development of
  a feature into multiple branches that will incrementally be merged into an
  intermediate feature branch that will eventually be merged into 'main'.

The **coding rules** are fairly simple:

- Abide by [PEP 8](https://peps.python.org/pep-0008/), in a way that is
  coherent with the practices already at work in declearn.
- Abide by [PEP 257](https://peps.python.org/pep-0257/), _i.e._ write
  docstrings **everywhere** (unless inheriting from a method, the behaviour
  and signature of which are unmodified), again using formatting that is
  coherent with the declearn practices.
- Type-hint the code, abiding by [PEP 484](https://peps.python.org/pep-0484/);
  note that the use of Any and of "type: ignore" comments is authorized, but
  should be remain sparse.
- Lint your code with [mypy](http://mypy-lang.org/) (for static type checking)
  and [pylint](https://pylint.pycqa.org/en/latest/) (for more general linting);
  do use "type: ..." and "pylint: disable=..." comments where you think it
  relevant, preferably with some side explanations.
  (see dedicated sub-sections below: [pylint](#running-pylint-to-check-the-code)
  and [mypy](#running-mypy-to-type-check-the-code))
- Reformat your code using [black](https://github.com/psf/black); do use
  (sparingly) "fmt: off/on" comments when you think it relevant
  (see dedicated sub-section [below](#running-black-to-format-the-code)).
- Abide by [semver](https://semver.org/) when implementing new features or
  changing the existing APIs; try making changes non-breaking, document and
  warn about deprecations or behavior changes, or make a point for API-breaking
  changes, which we are happy to consider but might take time to be released.

## Unit tests and code analysis

Unit tests, as well as more-involved functional ones, are implemented under
the `test/` folder of the present repository.
They are implemented using the [PyTest](https://docs.pytest.org) framework,
as well as some third-party plug-ins (refer to [Setup](./setup.md) for details).

Additionally, code analysis tools are configured through the `pyproject.toml`
file, and used to control code quality upon merging to the main branch. These
tools are [black](https://github.com/psf/black) for code formatting,
[pylint](https://pylint.pycqa.org/) for overall static code analysis and
[mypy](https://mypy.readthedocs.io/) for static type-cheking.

### Running the test suite using tox

The third-party [tox](https://tox.wiki/en/latest/) tools may be used to run
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

### Running unit tests using pytest

To run all the tests, simply use:

```bash
pytest test
```

To run the tests under a given module (here, "model"):

```bash
pytest test/model
```

To run the tests under a given file (here, "test_main.py"):

```bash
pytest test/test_main.py
```

Note that by default, some test scenarios that are considered somewhat
superfluous~redundant will be skipped in order to save time. To avoid
skipping these, and therefore run a more complete test suite, add the
`--fulltest` option to pytest:

```bash
pytest --fulltest test  # or any more-specific target you want
```

### Running black to format the code

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

### Running pylint to check the code

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

### Running mypy to type-check the code

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
