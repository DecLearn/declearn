# coding: utf-8

"""Script to auto-generate the docs' markdown files."""

import os
import re
import shutil
from typing import Dict, Tuple

import griffe


ROOT_FOLDER = os.path.dirname(__file__)
DOCS_INDEX = """{title}

## Introduction

{intro}

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
- [Developer guide](./devs.md):<br/>
  Information on how to contribute, codings rules and how to run tests.

## Copyright

{rights}
"""



def generate_index():
    """Fill-in the main index file based on the README one."""
    # Parse contents from the README file and write up the index.md one.
    title, readme = _parse_readme()
    # TODO: parse the existing index.md and fill rather then overwrite?
    docidx = DOCS_INDEX.format(
        title=title,
        intro=readme["Introduction"],
        rights=readme["Copyright"],
    )
    # Write up the index.md file.
    path = os.path.join(ROOT_FOLDER, "docs", "index.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(docidx)


def _parse_readme() -> Tuple[str, Dict[str, str]]:
    """Parse contents from the declearn README file."""
    path = os.path.join(ROOT_FOLDER, "README.md")
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    title, text = text.split("\n", 1)
    content = re.split(r"\n(## \w+\n+)", text)
    readme = dict(zip(content[1::2], content[2::2]))
    readme = {k.strip("# \n"): v.strip("\n") for k, v in readme.items()}
    return title, readme


def generate_api_docs():
    """Auto-generate the API Reference docs' markdown files."""
    # Build the API reference docs folder.
    docdir = os.path.join(ROOT_FOLDER, "docs")
    os.makedirs(docdir, exist_ok=True)
    docdir = os.path.join(docdir, "api-reference")
    if os.path.isdir(docdir):
        shutil.rmtree(docdir)
    os.makedirs(docdir)
    # Recursively generate the module-wise files.
    module = griffe.load(
        os.path.join(ROOT_FOLDER, "declearn"),
        submodules=True,
        try_relative_path=True,
    )
    parse_module(module, docdir, root=True)


def parse_module(
    module: griffe.dataclasses.Module,
    docdir: str,
    root: bool = False,
) -> str:
    """Recursively auto-generate markdown files for a module."""
    # Case of file-based public module (`module.py`).
    if not module.is_init_module:
        path = os.path.join(docdir, f"{module.name}.md")
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"::: {module.path}")
        return f"{module.name}.md"
    # Case of folder-based public module (`module/`)
    # Create a dedicated folder.
    if not root:  # skip for the main folder
        docdir = os.path.join(docdir, module.name)
        os.makedirs(docdir)
    # Create files or folders for public and private submodules.
    pub_mod = {}
    prv_mod = []
    for key, mod in module.modules.items():
        # Create a dedicated file for private submodules.
        # TODO: function or class-wise page with grouped index?
        if key.startswith("_"):
            name = key.strip("_")
            path = os.path.join(docdir, f"{name}.md")
            with open(path, "w", encoding="utf-8") as file:
                file.write(f"::: {mod.path}")
            prv_mod.append(name)
        # Create a dedicated folder for public submodules.
        else:
            pub_mod[key] = parse_module(mod, docdir)
    # Write up an overview file based on the '__init__.py' docs,
    # that also links to the public and private submodules.
    path = os.path.join(docdir, "index.md")
    code = f"::: {module.path}"
    if prv_mod:
        code += "\n\n## Exported contents\n" + "\n".join(
            f"- [{name}](./{name}.md)" for name in prv_mod
        )
    if pub_mod:
        code += "\n\n## Public submodules\n" + "\n".join(
            f"- [{name}](./{path})" for name, path in pub_mod.items()
        )
    with open(path, "w", encoding="utf-8") as file:
        file.write(code)
    return f"{module.name}/index.md"


if __name__ == "__main__":
    generate_index()
    generate_api_docs()
