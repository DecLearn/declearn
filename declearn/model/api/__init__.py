# coding: utf-8

"""Model Vector abstractions submodule."""

from ._vector import Vector, register_vector_type
from ._model import Model

__all__ = [
    "Model",
    "Vector",
    "register_vector_type",
]
