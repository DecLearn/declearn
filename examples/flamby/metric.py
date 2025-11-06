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

"""
Here we implement a custom subclass of the Declearn "Metric" abstraction.
It aims at representing a metric specific to the task related to the
TCGA-BRCA dataset provided by FLamby.
"""

import dataclasses
from typing import Dict, Optional, Union

import lifelines
import numpy as np

from declearn.metrics._api import Metric, MetricState


@dataclasses.dataclass
class CIndexState(MetricState):
    """Concordance index (c-index) 'MetricState'.
    We store in this state the weighted sum of c-indices in
    local_cindex_sum, and the number of samples in n_samples
    to compute a weighted average of the c-index.
    """

    local_cindex_sum: float
    n_samples: int


class CIndexMetric(Metric[CIndexState]):
    """
    Calculates the concordance index (c-index) between a series of event
    times and a predicted score.
    The c-index is the average of how often a model says X is greater than Y
    when, in the observed data, X is indeed greater than Y.
    The c-index also handles how to handle censored values.

    Note: This implementation uses a weighted average version of the metric
    from the FLamby TCGA-BRCA dataset example, see :
    https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_tcga_brca/metric.py

    Caveat: For simplicity, this implementation relies directly on
    `lifelines.utils.concordance_index`.
    A more rigorous approach would compute from scratch and store intermediate
    aggregates in CIndexState to obtain a global c-index over all
    (y_true, y_pred) pairs, rather than a weighted average of subset-level
    c-indices.
    """

    name = "c_index"
    state_cls = CIndexState

    def build_initial_states(self) -> CIndexState:
        return CIndexState(local_cindex_sum=0.0, n_samples=0)

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        if self._states.n_samples == 0:
            return {"c_index": np.nan}
        return {
            "c_index": self._states.local_cindex_sum / self._states.n_samples
        }

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        # y_true: [n_samples, 2] (event, time)
        # y_pred: [n_samples, 1]
        c_idx = lifelines.utils.concordance_index(
            y_true[:, 1],
            -y_pred,
            y_true[:, 0],
        )
        self._states.local_cindex_sum += c_idx * len(y_true)
        self._states.n_samples += len(y_true)
