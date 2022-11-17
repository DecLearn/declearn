# coding: utf-8

"""Optimizer gradients-alteration algorithms, implemented as plug-in modules.

Base class implemented here:
* OptiModule: base API for optimizer plug-in algorithms

Adaptive learning-rate algorithms:
* AdaGradModule : AdaGrad algorithm
* AdamModule    : Adam and AMSGrad algorithms
* RMSPropModule : RMSProp algorithm
* YogiModule    : Yogi algorithm, with Adam or AMSGrad base

Momentum algorithms:
* EWMAModule         : Exponentially-Weighted Moving Average module
* MomentumModule     : Momentum (and Nesterov) acceleration module
* YogiMomentumModule : Yogi-specific EWMA-like module

SCAFFOLD algorithm, as a pair of complementary modules:
* ScaffoldClientModule : client-side module
* ScaffoldServerModule : server-side module
"""

from ._api import (
    OptiModule,
)

from ._adaptive import (
    AdaGradModule,
    AdamModule,
    RMSPropModule,
    YogiModule,
)

from ._momentum import (
    EWMAModule,
    MomentumModule,
    YogiMomentumModule,
)

from ._scaffold import (
    ScaffoldClientModule,
    ScaffoldServerModule,
)
