# coding: utf-8

"""Model interfacing submodule, defining an API an derived applications.

This declearn submodule provides with:
* Model and Vector abstractions, used as an API to design FL algorithms
* Submodules implementing interfaces to various frameworks and models.
"""

from . import api
from . import sklearn
