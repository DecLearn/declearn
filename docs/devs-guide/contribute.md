# Contributions guide

Contributions to `declearn` are welcome, whether to provide fixes, suggest
new features (_e.g._ new subclasses of the core abstractions) or even push
forward framework evolutions and API revisions.

## GitLab and GitHub repositories

At the moment, declearn is being published on two mirrored public repositories:

- [Inria's GitLab](https://gitlab.inria.fr/magnet/declearn/declearn2) is where
  the code is primarily hosted and developed.

- [GitHub](https://github.com/declearn/declearn) hosts a mirroring repository
  where only the main and release branches are copied from the GitLab source.

Contributions are welcome on both platforms:

- GitHub is purposed to facilitate the interaction with end-users, that may
  easily open issues to report bugs, request new features or ask questions
  about the package.
- GitLab remains the place where core developers operate, notably as our CI/CD
  tooling has been developed for that platform. In the future, it may be (more
  or less progressively) replaced with GitHub as main development place if the
  onboarding of external contributors proves too difficult.

If you want to contribute directly to the code, you may open a Merge Request
(on GitLab) or Pull Request (on GitHub) to submit your code and ideas for
review and eventual integration into the package. GitHub-posted contributions
will need to transit via GitLab to be integrated. If you want an account for
the Inria GitLab, feel free to let us know (as it is unfortunately not yet
possible to register without an invitation).

## Git branching strategy

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

## Coding rules

The **coding rules** are fairly simple:

- Abide by [PEP 8](https://peps.python.org/pep-0008/), in a way that is
  coherent with the practices already at work in declearn.
- Abide by [PEP 257](https://peps.python.org/pep-0257/), _i.e._ write
  docstrings **everywhere** (unless inheriting from a method, the behaviour
  and signature of which are unmodified). The formatting rules for docstrings
  are detailed in the [docstrings style guide](./docs-style.md).
- Type-hint the code, abiding by [PEP 484](https://peps.python.org/pep-0484/);
  note that the use of Any and of "type: ignore" comments is authorized, but
  should remain parsimonious.
- Lint your code with [mypy](http://mypy-lang.org/) (for static type checking)
  and [pylint](https://pylint.pycqa.org/en/latest/) (for more general linting);
  do use "type: ..." and "pylint: disable=..." comments where you think it
  relevant, preferably with some side explanations. (see dedicated sections:
  [pylint](./tests.md#running-pylint-to-check-the-code)
  and [mypy](./tests.md/#running-mypy-to-type-check-the-code))
- Reformat your code using [black](https://github.com/psf/black); do use
  (sparingly) "fmt: off/on" comments when you think it relevant (see dedicated
  section: [black](./tests.md/#running-black-to-format-the-code)).
- Abide by [semver](https://semver.org/) when implementing new features or
  changing the existing APIs; try making changes non-breaking, document and
  warn about deprecations or behavior changes, or make a point for API-breaking
  changes, which we are happy to consider but might take time to be released.

## CI/CD pipelines

The **continuous development** (CI/CD) tools of GitLab are used:

- The [test suite](./tests.md) is run remotely when pushing new commits to the
  'develop' or to a release branch.
- It is also triggered when pushing to a feature branch that is the object of
  an open merge request that is not tagged to be a draft and that targets the
  develop or a release branch.
- It may be triggered manually for any merge request commit, whether draft or
  not, via the online gitlab interface.
