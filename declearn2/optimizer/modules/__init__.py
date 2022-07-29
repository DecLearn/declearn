# coding: utf-8

"""Optimizer gradients-alteration algorithms, implemented as plug-in modules.

Base class implemented here:
* OptiModule: base API for optimizer plug-in algorithms

Adaptive learning-rate algorithms:
* AdaGradModule : AdaGrad algorithm
* AdamModule    : Adam and AMSGrad algorithms
* RMSPropModule : RMSProp algorithm
* YogiModule    : Yogi algorithm, with Adam or AMSGrad base
"""

from ._base import (
    MomentumModule,
    OptiModule,
)
from ._adaptive import (
    AdaGradModule,
    AdamModule,
    RMSPropModule,
    YogiModule,
)
