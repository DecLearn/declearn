# Contributions guide

Contributions to `declearn` are welcome, whether to provide fixes, suggest new
features (_e.g._ new subclasses of the core abstractions) or even push forward
framework evolutions and API revisions.

## GitLab and GitHub repositories

At the moment, declearn is being published on two mirrored public repositories:

- [Inria's GitLab](https://gitlab.inria.fr/magnet/declearn/declearn2) is where
  the code is primarily hosted and developed.

- [GitHub](https://github.com/declearn/declearn) hosts a mirroring repository
  where only the main and release branches are copied from the GitLab source.

Contributions are welcome on both platforms:

- GitHub is purposed to facilitate the interaction with end-users, that may
  easily open issues to report bugs, request new features or ask questions about
  the package.
- GitLab remains the place where core developers operate, notably as our CI/CD
  tooling has been developed for that platform. In the future, it may be (more
  or less progressively) replaced with GitHub as main development place if the
  onboarding of external contributors proves too difficult.

If you want to contribute directly to the code, you may open a Merge Request (on
GitLab) or Pull Request (on GitHub) to submit your code and ideas for review and
eventual integration into the package. GitHub-posted contributions will need to
transit via GitLab to be integrated. If you want an account for the Inria
GitLab, feel free to let us know (as it is unfortunately not yet possible to
register without an invitation).

## Git branching strategy

- The 'develop' branch is the main one and should receive all finalized changes
  to the source code. Release branches are then created and updated by cherry-
  picking from that branch. It therefore acts as a nightly stable version.
- The 'rX.Y' branches are release branches for each and every X.Y versions. For
  past versions, these branches enable pushing patches towards a subminor
  version release (hence being version `X.Y.(Z+1)-dev`). For future versions,
  these branches enable cherry-picking commits from main to build up an alpha,
  beta, release-candidate and eventually stable `X.Y.0` version to release.
- Feature branches should be created at will to develop features, enhancements,
  or even hotfixes that will later be merged into 'main' and eventually into one
  or multiple release branches.
- It is legit to write up poc branches, as well as to split the development of a
  feature into multiple branches that will incrementally be merged into an
  intermediate feature branch that will eventually be merged into 'main'.

## Coding rules

The **coding rules** are fairly simple:

- Abide by [PEP 8](https://peps.python.org/pep-0008/), in a way that is coherent
  with the practices already at work in declearn.
- Abide by [PEP 257](https://peps.python.org/pep-0257/), _i.e._ write docstrings
  **everywhere** (unless inheriting from a method, the behaviour and signature
  of which are unmodified). The formatting rules for docstrings are detailed in
  the [docstrings style guide](./docs-style.md).
- Type-hint the code, abiding by [PEP 484](https://peps.python.org/pep-0484/);
  note that the use of Any and of `type: ignore` comments is authorized, but
  should remain parsimonious.
- Lint your code with [mypy](http://mypy-lang.org/) (for static type checking)
  and [ruff](https://github.com/astral-sh/ruff) (for more general linting); do
  use `type: ...` (mypy) and `noqa: [some-warning-code]` (to ignore ruff
  linting) comments where you think it relevant, preferably with some side
  explanations. (see dedicated sections:
  [ruff](./tests.md#running-ruff-to-check-the-code-quality) and
  [mypy](./tests.md#running-mypy-to-type-check-the-code))
- Reformat your code using [ruff format](https://github.com/astral-sh/ruff); do
  use (sparingly) "fmt: off/on" comments when you think it relevant (see
  dedicated section: [ruff format](./tests.md#running-ruff-to-format-the-code)).
- Abide by [semver](https://semver.org/) when implementing new features or
  changing the existing APIs; try making changes non-breaking, document and warn
  about deprecations or behavior changes, or make a point for API-breaking
  changes, which we are happy to consider but might take time to be released.

## Environment / Project manager

As a developer or contributor of Declearn, the recommended tools to handle
project dependencies are either :

- pip + a virtual environment
- [uv](https://docs.astral.sh/uv/)

### Details on uv usage

During development phase, uv can show some benefits compared to pip, among which
:

- dependency installation speed
- automatic isolation : a virtual environment is created and handled by uv, you
  don't need to create or activate it manually, just invoke uv commands to
  install dependencies, run scripts, lint tools, tests...
- reproducibility (thanks to its `uv.lock` file)
- test the code with different Python versions in an isolated way

If you want to use uv, you can quick install it
[here](https://docs.astral.sh/uv/getting-started/installation/).

---

If you use uv as a Declearn contributor, you can find below some useful
commands.

- Install all main dependencies of the project useful for users and devs, it excludes
  example-specific dependencies (e.g. flamby):

```bash
uv sync --extra all
```

- Install all dependencies of the project, with no exception :
```bash
uv sync --all-extras
```

- Add / remove dependencies in the project :  
  It is recommended that you add or remove them in the `pyproject.toml` with the
  wanted version marker and in the wanted dependency section, and then apply the
  changes in the environment using `uv sync`.  
  But if you want to do it programmatically, you can use the commands
  `uv add [DEPENDENCY]` and `uv remove [DEPENDENCY]`, see
  [uv documentation](https://docs.astral.sh/uv/reference/cli/) for the detailed
  arguments.

- Upgrade all dependencies of the project, in agreement with version conditions
  precised in the `pyproject.toml` :

```bash
uv lock --upgrade
uv sync --extra all
```

- Upgrade one dependency of the project (e.g. numpy), in agreement with version
  conditions precised in the `pyproject.toml` :

```bash
uv lock --upgrade-package [PACKAGE]
uv sync --extra all
```

- Run a tool that is present in dependencies (e.g. pytest, ruff, mypy) :

```bash
uv run [TOOL] [ARGS]
```

- See the installed project dependencies (with versions), organized in a tree
  structure :

```bash
uv tree
```

- See the version of an installed package (e.g. numpy) :

```bash
uv tree --package [PACKAGE]
```

- Show which Python version is used in the project virtual environment :

```bash
uv run python --version
```

- Install another python version (e.g. 3.12) in the project virtual environment
  :

```bash
uv venv --python 3.12
```

## Pre-commit

A [pre-commit](https://pre-commit.com/) configuration is provided in the
project, in `.pre-commit-config.yaml`.

At the moment, it is used to auto-format your code with ruff before each
commit.  
Its usage by the developer is totally optional (by default it is not used).

If activated, it prevents you from committing unformatted code : it blocks the
commit, format the necessary files among your changes, and let you re-add and
re-commit manually those file (allowing you to see the changes applied by
pre-commit).

To activate pre-commit :

- make sure the dev dependencies are installed in your environment, `pre-commit`
  is one of them
- run `pre-commit install` (prefixed by `uv run` if you use uv)

This tool is integrated into the project to free developers from formatting
concerns, while still allowing them to review it before committing their code.  
The pre-commit checks are intentionally lightweight to remain non-blocking and
minimize developer frustration during commits.

## CI/CD pipelines

The **continuous development** (CI/CD) tools of GitLab are used:

- The [test suite](./tests.md) is run remotely when pushing new commits to the
  'develop' or to a release branch.
- It is also triggered when pushing to a feature branch that is the object of an
  open merge request that is not tagged to be a draft and that targets the
  develop or a release branch.
- It may be triggered manually for any merge request commit, whether draft or
  not, via the online gitlab interface.

Resources relative to CI/CD are :

- the `.gitlab-ci.yml` configuration file
- the resources in the `ci/` directory (e.g. `Dockerfile` to build the CI/CD
  custom Docker image)
