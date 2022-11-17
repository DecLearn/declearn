# coding: utf-8

"""Optimizer loss-regularization algorithms, implemented as plug-in modules.

Base class implemented here:
* Regularizer: base API for loss-regularization plug-ins

Common regularization terms:
* LassoRegularizer : L1 regularization, aka Lasso penalization
* RidgeRegularizer : L2 regularization, aka Ridge penalization

Federated-Learning-specific regularizers:
* FedProxRegularizer : FedProx algorithm, as a proximal term regularizer
"""

from ._api import Regularizer
from ._base import (
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)
