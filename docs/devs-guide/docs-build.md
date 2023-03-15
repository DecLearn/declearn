# Building the documentation

The documentation rendered on our website is built automatically every time
a new release of the package occurs. You may however want to build and render
the website locally, notably to check that docstrings you wrote are properly
parsed and rendered.

## Generating markdown files

The markdown documentation of declearn, based on which our website is rendered,
is split between a static part (including the file you are currently reading),
and a dynamically-generated one: namely, the full API reference, as well as
part of the home index file.

The `gen_docs.py` script may be used to generate the dynamically-created
markdown files:

- `docs/index.md` is created based on a hard-coded template and contents
  parsed from the "README.md" file (to avoid discrepancies).
- `docs/api-reference/` files and subfolders are generated procedurally
  based on the exploration of the source code, using the `griffe` third-
  party tool that `mkdocstrings` also makes use of.

## Generating the website

The website is automatically generated using [mkdocs](https://www.mkdocs.org/)
and [mkdocstrings](https://mkdocstrings.github.io/) - the latter being used to
render the minimal API reference markdown files into actual content parsed from
the code's docstrings and type annotations.

To build the website locally, you may use the following instructions:

```bash
# Clone the gitlab repository (optionally targetting a given branch).
git clone https://gitlab.inria.fr/magnet/declearn/declearn2.git declearn
cd declearn

# Install the required dependencies (preferably in a dedicated venv).
# You may find the up-to-date list of dependencies in the `pyproject.toml`
# file, under [project.optional-dependencies], as the "docs" table.
pip install \
  mkdocstrings[python] \
  mkdocs-autorefs \
  mkdocs-literate-nav \
  mkdocs-material

# Auto-generate the API reference and home index markdown files.
python scripts/gen_docs.py

# Build the docs and serve them on your localhost.
mkdocs build
mkdocs serve  # by default, serve on localhost:8000
```

In practice, the actual documentation website is built using
[mike](https://github.com/jimporter/mike), so as to preserve access to the
documentation of past releases. This is however out of scope for building
and testing the documentation at a given code point locally.

## Contributing to the documentation

You may contribute changes to the out-of-code documentation by modifying
static markdown files and opening a merge request, just as you would for
source code modifications. Note that modifications to the home page should
be contributed to the `gen_docs.py` script and/or `README.md` file; other
markdown files' modifications may be proposed directly.

Contributions should be pushed to the main repository (i.e. the one that
holds the source code) rather than to the website one, as the latter is
periodically updated by pulling from the former.
