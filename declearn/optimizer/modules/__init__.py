# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Optimizer gradients-alteration algorithms, implemented as plug-in modules.

Base class implemented here:
* OptiModule: base API for optimizer plug-in algorithms

Adaptive learning-rate algorithms:
* AdaGradModule : AdaGrad algorithm
* AdamModule    : Adam and AMSGrad algorithms
* RMSPropModule : RMSProp algorithm
* YogiModule    : Yogi algorithm, with Adam or AMSGrad base

Gradient clipping algorithms:
* L2Clipping : Fixed-threshold L2-norm gradient clipping module

Momentum algorithms:
* EWMAModule         : Exponentially-Weighted Moving Average module
* MomentumModule     : Momentum (and Nesterov) acceleration module
* YogiMomentumModule : Yogi-specific EWMA-like module

Noise-addition mechanisms:
* NoiseModule         : abstract base class for noise-addition modules
* GaussianNoiseModule : Gaussian noise-addition module

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

from ._clipping import (
    L2Clipping,
)

from ._momentum import (
    EWMAModule,
    MomentumModule,
    YogiMomentumModule,
)

from ._noise import (
    GaussianNoiseModule,
    NoiseModule,
)

from ._scaffold import (
    ScaffoldClientModule,
    ScaffoldServerModule,
)
