# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Several client sampling `Criterion` concrete classes."""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Literal,
    Optional,
    get_args,
)

import numpy as np

from declearn.client_sampler.criterion._api import Criterion
from declearn.messaging import TrainReply
from declearn.model.api import Model


class GradientNormCriterion(Criterion):
    """Criterion where the score is the L2-norm of the client "gradients"
    (model updates).
    """

    name = "gradient_norm"

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        client_to_norm: Dict[str, Optional[float]] = {}
        for client, reply in client_to_reply.items():
            flattened_updates, _ = reply.updates.updates.flatten()
            client_to_norm[client] = np.linalg.norm(flattened_updates).item()
        return client_to_norm


class NormalizedDivCriterion(Criterion):
    r"""Criterion where the score is the normalized model divergence.

    Normalized model divergence is the average difference between the model
    weights in client i and the global model) :

    $$ \frac{1}{|w|} \sum_{j=1}^{|w|}
      \left| \frac{w_{ij} - \bar{w}_j}{\bar{w}_j} \right| $$

    Where $w$ represents the weights of a model, $\bar{w}$ represents the
    weights of the global model, $w_{ij}$ and $\bar{w}_{j}$ are the $j$ th
    weights of client $i$ and the global model, respectively.

    Notes
    -----
    Only the trainable weights are compared.

    Raises
    ------
    ValueError:
        If the number of trainable weights in the global model and in a client
        updates object are different.

    Reference
    ---------
    [1] Fu et al., 2023.
        Client Selection in Federated Learning: Principles, Challenges, and
        Opportunities.
        Section IV.A.2.
        https://arxiv.org/abs/2211.01549
    """

    name = "normalized_divergence"

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        client_to_div: Dict[str, Optional[float]] = {}
        w_server = np.array(
            global_model.get_weights(trainable=True).flatten()[0]
        )  # server weights
        size_w = len(w_server)  # model size (nb trainable parameters)
        eps = 1e-8  # epsilon added to denominator to avoid zero-division error
        for client, reply in client_to_reply.items():
            w_updates = np.array(reply.updates.updates.flatten()[0])
            # client weight updates
            size_upd = len(w_updates)
            if size_upd != size_w:
                raise ValueError(
                    f"Flattened global model weights size ({size_w}) and "
                    f"client model updates size ({size_upd}) must be equal."
                )
            score = float(
                1 / size_w * np.sum(np.abs(w_updates / (w_server + eps)))
            )
            client_to_div[client] = score
        return client_to_div


class TrainTimeCriterion(Criterion):
    """Criterion where the score is computed from the last round client's
    training time (in seconds).

    Attributes
    ----------
    lower_is_better: bool
        If True (default), a lower time leads to a better score (score will be
        `- time`). Otherwise, a higher time leads to a better score (score will
        be `+ time`).
    """

    name = "train_time"

    def __init__(self, lower_is_better: bool = True):
        self.lower_is_better = lower_is_better

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        sign = -1 if self.lower_is_better else 1
        return {
            client: sign * reply.t_spent
            for client, reply in client_to_reply.items()
        }


class TrainTimeHistoryCriterion(Criterion):
    """Criterion where the score is computed from all past rounds client's
    training time (in seconds).

    Attributes
    ----------
    lower_is_better: bool
        If True (default), a lower value for aggregated times leads to a better
        score (score will be `- aggregated_times`). Otherwise, a higher value
        for aggregated times of leads to a better score (score will be
        `+ aggregated_times`).
    agg: AggregateFunc
        Name of a method to aggregate the history values into a float, e.g.
        average, sum.
    history: Dict[str, List[float]], read-only instance property
        Dictionary mapping each client to its training time history (time
        values for all past training rounds).

    Notes
    -----
    Beware that the time history will be updated every time a call to `compute`
    is made, assuming that this call matches a new training round.
    Thus, you should only call this class' `compute` method once per round
    (passing the new round client replies as argument).
    """

    name = "train_time_history"

    AggregateFunc = Literal["average", "sum"]

    def __init__(
        self, lower_is_better: bool = True, agg: AggregateFunc = "average"
    ):
        if agg not in get_args(self.AggregateFunc):
            raise ValueError(f"Unsupported aggregate function '{agg}'.")

        self.lower_is_better = lower_is_better
        self.agg = agg
        self._history: Dict[str, List[float]] = {}

    @property
    def history(self):
        return self._history

    def compute(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> Dict[str, Optional[float]]:
        sign = -1 if self.lower_is_better else 1
        for client, reply in client_to_reply.items():
            if client in self._history:
                self._history[client].append(reply.t_spent)
            else:
                self._history[client] = [reply.t_spent]  # init history

        # aggregate all times in each client history
        if self.agg == "average":

            def agg_fn(hist):
                return sum(hist) / len(hist)
        elif self.agg == "sum":
            agg_fn = sum
        else:
            raise ValueError(f"Unsupported aggregate function '{self.agg}'.")

        return {
            client: sign * agg_fn(self._history[client])
            for client in client_to_reply
        }
