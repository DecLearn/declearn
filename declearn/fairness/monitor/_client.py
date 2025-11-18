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

"""Client-side controller to monitor fairness without altering training."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.fairness.api import FairnessControllerClient
from declearn.secagg.api import Encrypter

__all__ = [
    "FairnessMonitorClient",
]


class FairnessMonitorClient(FairnessControllerClient):
    """Client-side controller to monitor fairness without altering training."""

    algorithm = "monitor"

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        pass

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
        values: Dict[str, Dict[Tuple[Any, ...], float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        return {
            f"{metric}_{group}": value
            for metric, m_dict in values.items()
            for group, value in m_dict.items()
        }
