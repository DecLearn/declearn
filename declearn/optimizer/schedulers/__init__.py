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

"""Time-based schedulers for learning rate or weight decay.

This submodule provides with an extensible API and a number of
standard concrete implementations for time-based scheduling of
learning or weight decay rates throughout training.

The rules implemented here are mostly designed to operate based
on steps, but may also operate based on training rounds (a unit
which may not be equivalent to an epoch, but is thought to be
more appropriate for the federated learning setting).

API-defining base class
-----------------------
* [Scheduler][declearn.optimizer.schedulers.Scheduler]:
    Abstract base class for time-based learning rate schedulers.

Basic decay rules
-----------------
These decay schedulers may be parameterized to take either training
steps or rounds as time unit.

* [CosineAnnealing][declearn.optimizer.schedulers.CosineAnnealing]:
    Cosine Annealing scheduler.
* [ExponentialDecay][declearn.optimizer.schedulers.ExponentialDecay]:
    Exponential decay scheduler.
* [InverseScaling][declearn.optimizer.schedulers.InverseScaling]:
    Inverse-scaling decay scheduler.
* [LinearDecay][declearn.optimizer.schedulers.LinearDecay]:
    Linear decay scheduler.
* [PiecewiseDecay][declearn.optimizer.schedulers.PiecewiseDecay]:
    Piecewise-constant exponential decay scheduler.
* [PolynomialDecay][declearn.optimizer.schedulers.PolynomialDecay]:
    Polynomial decay scheduler.

Cyclic rate rules
-----------------
* [CosineAnnealingWarmRestarts]
[declearn.optimizer.schedulers.CosineAnnealingWarmRestarts]:
    Cosine Annealing with Warm Restarts scheduler (aka SGDR).
* [CyclicExpRange][declearn.optimizer.schedulers.CyclicExpRange]:
    Cyclic Learning Rate (CLR) scheduling policy with exponential decay.
* [CyclicTriangular][declearn.optimizer.schedulers.CyclicTriangular]:
    Cyclic Learning Rate (CLR) scheduling policy with triangular cycle.

Warmup schedulers
-----------------
* [Warmup][declearn.optimizer.schedulers.Warmup]:
    Scheduler (wrapper) setting up a linear warmup over steps.
* [WarmupRounds][declearn.optimizer.schedulers.WarmupRounds]:
    Scheduler (wrapper) setting up a linear warmup over rounds.
"""

from ._api import Scheduler
from ._cosine import (
    CosineAnnealing,
    CosineAnnealingWarmRestarts,
)
from ._cyclic import (
    CyclicExpRange,
    CyclicTriangular,
)
from ._decay import (
    ExponentialDecay,
    InverseScaling,
    LinearDecay,
    PiecewiseDecay,
    PolynomialDecay,
)
from ._warmup import (
    Warmup,
    WarmupRounds,
)
