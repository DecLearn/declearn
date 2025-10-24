# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to auto-generate the docs' markdown files."""

import os
import re
import shutil
from typing import Dict, Tuple

import griffe

ROOT_FOLDER = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
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
- [Developer guide](./devs-guide/index.md):<br/>
  Information on how to contribute, codings rules and how to run tests.

## Copyright

{rights}
"""


def generate_index():
    """Fill-in the main index file based on the README one."""
    # Parse contents from the README file and write up the index.md one.
    title, readme = _parse_readme()
    # FUTURE: parse the existing index.md and fill it rather then overwrite?
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
    readme = dict(zip(content[1::2], content[2::2], strict=False))
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
    generate_module_docs(module, docdir, root=True)


def generate_module_docs(
    module: griffe.Module,
    docdir: str,
    root: bool = False,
) -> str:
    """Recursively generate markdown files for a module's documentation."""
    # Case of file-based public module (`module.py`).
    if not module.is_init_module:
        return _generate_public_module_file_docs(module, docdir)
    # Case of folder-based public module (`module/`)
    return _generate_public_module_folder_docs(module, docdir, root)


def _generate_public_module_file_docs(
    module: griffe.Module,
    docdir: str,
) -> str:
    """Create a markdown file for a public single-file module."""
    path = os.path.join(docdir, f"{module.name}.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"::: {module.path}")
    return f"{module.name}.md"


def _generate_public_module_folder_docs(
    module: griffe.Module,
    docdir: str,
    root: bool = False,
) -> str:
    """Create markdown files for a public folder-based module."""
    # Create a dedicated folder.
    if not root:  # skip for the main folder
        docdir = os.path.join(docdir, module.name)
        os.makedirs(docdir)
    # Parse through public and private submodules to generate doc files.
    pub_mod = _generate_public_submodules_doc(module, docdir)
    pub_obj = _generate_private_submodules_content_doc(module, docdir, pub_mod)
    # Generate summary files.
    _generate_overview_file(module, docdir)
    _generate_literate_nav_summary(module, docdir, pub_obj, pub_mod)
    return f"{module.name}/"


def _generate_public_submodules_doc(
    module: griffe.Module,
    docdir: str,
) -> Dict[str, str]:
    """Create files for public submodules of a base module."""
    pub_mod = {}
    for key, mod in module.modules.items():
        # local copies, to avoid updating loop variables
        _key = key
        _mod = mod
        if not _key.startswith("_"):
            if isinstance(_mod, griffe.Alias):
                _key = f"{_key} (alias re-export)"
                _mod = _mod.target
            pub_mod[_key] = generate_module_docs(_mod, docdir)
    return pub_mod


def _generate_private_submodules_content_doc(
    module: griffe.Module,
    docdir: str,
    pub_mod: Dict[str, str],
) -> Dict[str, str]:
    """Create files for public contents from a module's private submodules."""
    pub_obj = {}
    if module.exports is None:
        members = module.members
    else:
        members = {str(k): module.members[str(k)] for k in module.exports}
    for key, obj in members.items():
        if obj.is_module or obj.module.name in pub_mod or key.startswith("_"):
            continue
        if not (obj.docstring or obj.is_class or obj.is_function):
            continue
        path = os.path.join(docdir, f"{obj.name}.md")
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"#`{obj.path}`\n::: {obj.path}")
        pub_obj[key] = f"{obj.name}.md"
    return pub_obj


def _generate_overview_file(
    module: griffe.Module,
    docdir: str,
) -> None:
    """Write up an overview file based on a module's '__init__.py'."""
    path = os.path.join(docdir, "index.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"::: {module.path}")


def _generate_literate_nav_summary(
    module: griffe.Module,
    docdir: str,
    pub_obj: Dict[str, str],
    pub_mod: Dict[str, str],
) -> None:
    """Write up a literate-nav summary file based on created doc files."""
    index = rf"- [\[{module.name}\]](./index.md)"
    index += "".join(f"\n- [{k}](./{pub_obj[k]})" for k in sorted(pub_obj))
    index += "".join(f"\n- [{k}](./{pub_mod[k]})" for k in sorted(pub_mod))
    path = os.path.join(docdir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(index)


if __name__ == "__main__":
    generate_index()
    generate_api_docs()
