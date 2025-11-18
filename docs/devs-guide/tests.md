# Unit tests and code analysis

Unit tests, as well as more-involved functional ones, are implemented under the
`test/` folder of the declearn gitlab repository. Integration tests are isolated
in the `test/functional/` subfolder to enable one to easily exclude them in
order to run unit tests only.

Tests are implemented using the [PyTest](https://docs.pytest.org) framework, as
well as some third-party plug-ins that are automatically installed with the
package when using `pip install declearn[tests]`.

Additionally, code analysis tools are configured through the `pyproject.toml`
file, and used to control code quality upon merging to the main branch. These
tools are [ruff](https://github.com/astral-sh/ruff) for overall static code
analysis and code formatting, and [mypy](https://mypy.readthedocs.io/) for
static type-cheking.

## Running the test suite

### Running the test suite using tox

The third-party [tox](https://tox.wiki/en/latest/) tool may be used to run the
entire test suite within a dedicated virtual environment. Simply run `tox` from
the commandline with the root repo folder as working directory. You may
optionally specify the python version(s) with which you want to run tests.

```bash
tox           # run with default python 3.11
tox -e py312  # override to use python 3.12
```

Note that additional parameters for `pytest` may be passed as well, by adding
`--` followed by any set of options you want at the end of the `tox` command.
For example, to use the declearn-specific `--fulltest` option (see the section
below), run:

```bash
tox [tox options] -- --fulltest
```

You may alternatively trigger some specific categories of tests, using one of:

```bash
tox -e py{version}-tests       # run unit and integration tests
tox -e py{version}-lint_code   # run static code analysis on the source code
tox -e py{version}-lint_tests  # run static code analysis on the tests' code
```

Note that calling all three commands is equivalent to running the basic
`tox -e py{version}` job and is somewhat less efficient, as an isolated virtual
environment will be created for each and every one of them (in spite of
containing the same package and dependencies).

### Running the test suite using a bash script

Under the hood, `tox` makes calls to the `scripts/run_tests.sh` bash script,
where the categories of tests and associate commands are defined. End-users may
skip the build isolation offered by tox and call that script directly; it is
however not advised as it will require you to first install declearn (_not_ in
editable mode for tests' code's linting) and will not protect you from side
effects of packages pre-installed in your current python environment.

If you want to call that script, call one of:

```bash
bash scripts/run_tests.sh lint_code
bash scripts/run_tests.sh lint_tests
bash scripts/run_tests.sh run_tests  # optionally adding pytest flags
```

The actual takeaway from this section is that developers that want to edit or
expand the tests suite by altering or adding commands to be run should have a
look at that bash script and edit it. Further information and instructions are
provided as part of its internal documentation.

### Running the test suite components manually

You may prefer to run test suite components manually, e.g. because you want to
investigate a specific type of test or run tests for a single part of the code
(notably when something fails and requires fixing). To do so, please refer to
the next section about running targetted tests by manually calling the various
tools that make up the test suite.

## Running targetted tests

In this section, it is assumed that you are in a dedicated Python (virtual)
environment with the test/lint tools installed.  
If you use uv to manage the project, you can prefix all the following commands
by `uv run` (e.g. `uv run pytest [args]` instead of `pytest [args]`).

### Running unit and integration tests with pytest

To run all the unit and integration tests, simply use:

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
superfluous~redundant will be skipped in order to save time. To avoid skipping
these, and therefore run a more complete test suite, add the `--fulltest` option
to pytest:

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

### Running ruff to format the code

The [ruff](https://github.com/astral-sh/ruff) code formatter (which works the
same as [black](https://github.com/psf/black)) is used to enforce uniformity of
the source code's formatting style. It is configured to have a maximum line
length of 79 (as per [PEP 8](https://peps.python.org/pep-0008/)) and ignore
auto-generated protobuf files, but will otherwise modify files in-place when
executing the following commands from the repository's root folder:

```bash
ruff format declearn  # reformat the package
ruf format test       # reformat the tests
```

Note that it may also be called on individual files or folders. One may
"blindly" run the formatter, however it is actually advised to have a look at
the reformatting operated, and act on any readability loss due to it. A couple
of advice:

1. Use `#fmt: off` / `#fmt: on` comments sparingly, but use them. <br/>It is
   totally okay to protect some (limited) code blocks from reformatting if you
   already spent some time and effort in achieving a readable code that the
   formatter would disrupt. Please consider refactoring as an alternative (e.g.
   limiting the nest-depth of a statement).

2. Pre-format functions and methods' signature to ensure style homogeneity.
   <br/>When a signature is short enough, the formatter may attempt to flatten
   it as a one-liner, whereas the norm in declearn is to have one line per
   argument, all of which end with a trailing comma (for diff minimization
   purposes). It may sometimes be necessary to manually write the code in the
   latter style for the formatter not to reformat it.

Finally, note that the test suite run with tox comprises code-checking by ruff
format, and will fail if some code is deemed to require alteration by that tool.
You may run this check manually:

```bash
ruff format --check declearn  # or any specific file or folder
```

### Running ruff to check the code quality

[ruff](https://github.com/astral-sh/ruff) is above all a code linter, it is
expected to be used for static code analysis. As a consequence,
`# noqa: [some-warning-code]` comments can be found (and added) to the source
code, preferably with some indication as to the rationale for silencing the
warning (or error).

A minimal amount of non-standard hyper-parameters are configured via the
`pyproject.toml` file and will automatically be used by ruff when run from
within the repository's folder.

Most code editors enable integrating the linter to analyze the code as it is
being edited. To lint the entire package (or some specific files or folders) one
may simply run `ruff check`:

```bash
ruff check declearn  # analyze the package
ruff check test      # analyze the tests
```

Note that the test suite run with tox comprises the previous two commands, if
any of the commands fail, it prevent from accepting a merge request for example.

Note : Pylint was previously used in the project to lint the code, that is why
you can still find Pylint-related comments in the codebase.  
Most of the important lints provided by Pylint are now included in the rules
checked by ruff (Pylint rules, prefixed by `PL`, are activated in our ruff
settings).  
Ruff allows to extend the set of lint rules that are considered, and the
checks are executed much faster than using its predecessor.

### Running mypy to type-check the code

The [mypy](https://mypy.readthedocs.io/) linter is expected to be used for
static type-checking code analysis. As a consequence, `# type: ignore` comments
can be found (and added) to the source code, as sparingly as possible (mostly,
to silence warnings about untyped third-party dependencies, false-positives, or
locally on closure functions that are obvious enough to read from context).

Code should be type-hinted as much and as precisely as possible - so that mypy
actually provides help in identifying (potential) errors and mistakes, with code
clarity as final purpose, rather than being another linter to silence off.

A minimal amount of parameters are configured via the `pyproject.toml` file, and
some of the strictest rules are disabled as per their default value (e.g. Any
expressions are authorized - but should be used sparingly).

Most code editors enable integrating the linter to analyze the code as it is
being edited. To lint the entire package (or some specific files or folders) one
may simply run `mypy`:

```bash
mypy declearn
```

Note that the test suite run with tox comprises the previous command. If mypy
identifies errors, the test suite will fail - notably preventing acceptance of
merge requests.

## Notes regarding the GitLab CI/CD

Our GitLab repository is doted with CI/CD (continuous integration / continuous
delivery) tools, that have the test suite be automatically run when commits are
pushed to the repository under certain conditions. The rules for this can be
founder under the `.gitlab-ci.yml` YAML file, which should be somewhat readable
by someone with limited or no prior knowledge of the GitLab CI/CD tools (that
are otherwise [documented here](https://docs.gitlab.com/ee/ci/)).

To summarize the current CI/CD configuration:

- The test suite is run when commits are pushed to the development or a release
  branch (including on merge commits).
- The test suite is run with some restrictions (no GPU use, limited number of
  integration tests) on commits to a branch with an open, non-draft Merge
  Request (MR).
- Both forms (minimal/maximal) of the test suite may be manually triggered for
  commits made to a branch with an open, draft MR.
- Tox is used when running jobs, with some re-use of the created environments
  but isolation across branches.
- Some manual and/or automated jobs enable removing the tox cache, to collect
  unused files and/or force the full environment recreation.

Note that we currently rely on a self-hosted GitLab Runner that uses a Docker
executor to run jobs and has access to a GPU, that some tests make use of.
