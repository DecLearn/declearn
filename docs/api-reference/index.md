The API reference is automatically built from the source code and uploaded
to our [website](https://declearn.gitlabpages.inria.fr). This file is just
a placeholder for the docs that are included as part of the project's main
gitlab repository.

You may build this doc locally:
```bash
# Clone the gitlab repository (optionally targetting a given branch).
git clone https://gitlab.inria.fr/magnet/declearn/declearn2.git declearn
cd declearn
# Install the required dependencies (preferably in a dedicated venv).
pip install -U pip
pip install \
  mkdocstrings[python] \
  mkdocs-autorefs \
  mkdocs-literate-nav \
  mkdocs-material
# Auto-generate the API reference and main index markdown files.
python gen_docs.py
# Build the docs and serve them on your localhost.
mkdocs build
mkdocs serve  # by default, serve on localhost:8000
```
