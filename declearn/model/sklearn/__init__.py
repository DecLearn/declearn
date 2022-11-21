# coding: utf-8

"""Scikit-Learn models interfacing tools.

Due to the variety of model classes provided by scikit-learn
and to the way their learning process is implemented, model-
specific interfaces are required for declearn compatibility.

This module currently implements:
* NumpyVector: Vector subclass to wrap numpy.ndarray objects
* SklearnSGDModel: interface to SGD-based linear models
"""

from ._np_vec import NumpyVector
from ._sgd import SklearnSGDModel
