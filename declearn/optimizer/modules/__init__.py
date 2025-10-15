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

"""Optimizer gradients-alteration algorithms, implemented as plug-in modules.

API base classes
----------------
* [AuxVar][declearn.optimizer.modules.AuxVar]:
    Abstract base class for OptiModule auxiliary variables.
* [OptiModule][declearn.optimizer.modules.OptiModule]:
    Abstact base class for optimizer plug-in algorithms.

Adaptive learning-rate algorithms
---------------------------------
* [AdaGradModule][declearn.optimizer.modules.AdaGradModule]:
    AdaGrad algorithm.
* [AdamModule][declearn.optimizer.modules.AdamModule]:
    Adam and AMSGrad algorithms.
* [RMSPropModule][declearn.optimizer.modules.RMSPropModule]:
    RMSProp algorithm.
* [YogiModule][declearn.optimizer.modules.YogiModule]:
    Yogi algorithm, with Adam or AMSGrad base.

Gradient clipping algorithms
----------------------------
* [L2Clipping][declearn.optimizer.modules.L2Clipping]:
    Fixed-threshold, per-parameter-L2-norm gradient clipping module.
* [L2GlobalClipping][declearn.optimizer.modules.L2GlobalClipping]:
    Fixed-threshold, global-L2-norm gradient clipping module.

Momentum algorithms
-------------------
* [EWMAModule][declearn.optimizer.modules.EWMAModule]:
    Exponentially-Weighted Moving Average module.
* [MomentumModule][declearn.optimizer.modules.MomentumModule]:
    Momentum (and Nesterov) acceleration module.
* [YogiMomentumModule][declearn.optimizer.modules.YogiMomentumModule]:
    Yogi-specific EWMA-like module.

Noise-addition mechanisms
-------------------------
* [NoiseModule][declearn.optimizer.modules.NoiseModule]:
    Abstract base class for noise-addition modules.
* [GaussianNoiseModule][declearn.optimizer.modules.GaussianNoiseModule]:
    Gaussian noise-addition module.

SCAFFOLD algorithm
------------------
Scaffold is implemented as a pair of complementary modules:

* [ScaffoldClientModule][declearn.optimizer.modules.ScaffoldClientModule]:
    Client-side Scaffold module.
* [ScaffoldServerModule][declearn.optimizer.modules.ScaffoldServerModule]:
    Server-side Scaffold module.
* [ScaffoldAuxVar][declearn.optimizer.modules.ScaffoldAuxVar]:
    AuxVar subclass for Scaffold modules.
"""

from ._adaptive import (
    AdaGradModule,
    AdamModule,
    RMSPropModule,
    YogiModule,
)
from ._api import (
    AuxVar,
    OptiModule,
)
from ._clipping import (
    L2Clipping,
    L2GlobalClipping,
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
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule,
)
