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

"""Server-side controller to monitor fairness without altering training."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.fairness.api import (
    FairnessControllerServer,
    instantiate_fairness_function,
)
from declearn.secagg.api import Decrypter

__all__ = [
    "FairnessMonitorServer",
]


class FairnessMonitorServer(FairnessControllerServer):
    """Server-side controller to monitor fairness without altering training.

    This controller, together with its client-side counterpart,
    does not alter the training procedure of the model, but adds
    computation and communication steps to measure its fairness
    level at the start of each and every training round.

    It is compatible with any group-fairness definition implemented
    in DecLearn, and any number of sensitive groups compatible with
    the chosen definition.
    """

    algorithm = "monitor"

    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f_type, f_args)
        # Assign a temporary fairness functions, replaced at setup time.
        self.function = instantiate_fairness_function(
            f_type="accuracy_parity", counts={}
        )

    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        self.function = instantiate_fairness_function(
            f_type=self.f_type,
            counts=dict(zip(self.groups, counts, strict=False)),
            **self.f_args,
        )
        return aggregator

    async def finalize_fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        values: List[float],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack group-wise accuracy metrics and compute fairness ones.
        accuracy = dict(zip(self.groups, values, strict=False))
        fairness = self.function.compute_from_federated_group_accuracy(
            accuracy
        )
        # Package and return these metrics.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }
        metrics.update(
            {f"{self.f_type}_{key}": val for key, val in fairness.items()}
        )
        return metrics
