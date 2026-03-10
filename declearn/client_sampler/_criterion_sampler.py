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

"""`ClientSampler` implementation that selects clients based on a criterion
derived from client training replies and the global model.
"""

from typing import Any, Dict, Literal, Optional, Set, get_args

import pandas as pd

from declearn.client_sampler import ClientSampler
from declearn.client_sampler.criterion import Criterion, instantiate_criterion
from declearn.messaging import TrainReply
from declearn.model.api import Model

MissingScorePolicy = Literal["priority", "equal"]


class CriterionClientSampler(ClientSampler):
    """Client sampler selecting the n clients with the highest criterion score.

    The criterion must be chosen by the user and passed to the sampler, it
    defines how the criterion score is computed. It can be computed
    using data from clients training replies and possibly from the server
    model.

    A client score is the criterion value if already computed ; otherwise,
    it is a default value depending on the missing_scores_policy.

    Attributes
    ----------
    n_samples: int
        Number of clients to be sampled.
    criterion: Criterion
        The criterion to be used to select the best clients.
    client_to_score: Dict[str, Optional[float]]
        Dictionary mapping each client to its criterion score (or None if not
        computed).
    missing_scores_policy:  Optional[MissingScorePolicy]
        String that identifies a missing scores policy, i.e. a strategy to
        attribute a criterion score to a client if it is missing (e.g. because
        of a missing train reply).
        Supported values are :
            "priority": prioritizes the clients with a missing score, by
            setting the score to infinity.
            "equal": sets the missing scores to 1 / number_of_clients.
    """

    strategy = "criterion"

    def __init__(
        self,
        n_samples: int,
        criterion: Criterion,
        missing_scores_policy: Optional[MissingScorePolicy] = "priority",
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        """Instantiate the criterion client sampler.

        Raises
        ------
        ValueError:
            If the provided missing scores policy is not supported.
        """
        super().__init__(max_retries=max_retries)
        if missing_scores_policy not in get_args(MissingScorePolicy):
            raise ValueError(
                f"Missing scores policy {missing_scores_policy} "
                f"is not supported."
            )
        self.n_samples = n_samples
        self.criterion = criterion
        self.client_to_score: Dict[str, Optional[float]] = {}
        self.missing_scores_policy = missing_scores_policy

    @property
    def secagg_compatible(self) -> bool:
        return False

    def init_clients(self, clients: Set[str]) -> None:
        """Initialize clients common metadata and set each client's criterion
        score to None.
        """
        super().init_clients(clients)
        for client in clients:
            self.client_to_score[client] = None

    def convert_missing_scores(self) -> Dict[str, float]:
        """Access client scores in metadata, and convert missing scores such
        that each client gets a non-None score.

        Raises
        ------
        ValueError:
            If the string identifying the missing score policy is not
            supported.
        """
        if self.missing_scores_policy == "priority":
            replacement_score = float("inf")
        elif self.missing_scores_policy == "equal":
            replacement_score = 1 / len(self.clients)
        else:
            raise ValueError(
                f"Missing scores policy {self.missing_scores_policy} "
                f"is not supported."
            )

        return {
            client: score if score is not None else replacement_score
            for client, score in self.client_to_score.items()
        }

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        """Back-end of the sampling method for criterion client sampler.

        If there are more than `n_samples` clients in `eligible_clients`,
        this method selects the `n_samples` clients with the highest criterion
        scores. Otherwise, they are all selected.
        """
        if self.n_samples >= len(eligible_clients):
            return eligible_clients

        cli_to_score = self.convert_missing_scores()
        eligible_cli_to_score = {
            client: score
            for client, score in cli_to_score.items()
            if client in eligible_clients
        }

        ordered_cli_to_score = pd.Series(eligible_cli_to_score).sort_values(
            ascending=False
        )
        self.logger.debug(
            "Client scores: %s", ordered_cli_to_score.astype(float).to_dict()
        )

        best_clients = set(ordered_cli_to_score[: self.n_samples].index)
        return best_clients

    def update(
        self, client_to_reply: Dict[str, TrainReply], global_model: Model
    ) -> None:
        """Update clients metadata and sampler internal state according to each
        client training reply and the global model.

        Concretely, compute and update each client criterion score.
        """
        updated_client_to_score = self.criterion.compute(
            client_to_reply, global_model
        )
        for client, score in updated_client_to_score.items():
            self.client_to_score[client] = score

    @classmethod
    def from_specs(cls, **kwargs: Any) -> ClientSampler:
        """Instantiate a `CriterionClientSampler` from specifications."""
        criterion = kwargs["criterion"]
        if isinstance(criterion, Criterion):
            pass  # nothing to do
        elif isinstance(criterion, dict):
            kwargs["criterion"] = instantiate_criterion(**criterion)
        else:
            raise ValueError(
                f"Unsupported criterion type '{type(criterion)}' used as "
                "'criterion' value"
            )
        return cls(**kwargs)
