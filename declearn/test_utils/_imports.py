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

"""Context manager to perform relative/local imports."""

import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def make_importable(dirname: str = ".") -> Iterator[None]:
    """Context manager to perform relative/local imports.

    This functions establishes a context in which a given directory
    is added to the `sys.path` variable, enabling one to import the
    python code files it contains even without `__init__.py` files.

    The default usage is to import from the local folder ("."); in
    a test or script file launched from its own directory:
    >>> with make_importable():
    >>>     import local_file

    It is however possible to specify another directory, and/or to
    use an explicit path, including the one relative to the script
    file:
    >>> with make_importable(os.path.dirname(__file__)):
    >>>     import local_file
    """
    remove = False
    try:
        if dirname not in sys.path:
            sys.path.append(dirname)
            remove = True
        yield None
    finally:
        if remove:
            sys.path.remove(dirname)
