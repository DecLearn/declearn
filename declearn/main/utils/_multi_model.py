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


import copy
from typing import Any, Callable, Dict, List, Optional, TypeVar

from declearn import messaging
from declearn.dataset._base import Dataset
from declearn.main.utils._error import InvalidDatasetType

RequestT = TypeVar("RequestT", bound=messaging.Message)
T = TypeVar("T")
MultiReply = TypeVar("MultiReply")
Reply = TypeVar("Reply")
R = TypeVar("R")


__all__ = [
    "validate_optional_dataset_list",
    "dispatch_replies",
    "copy_requests_with_weights",
]


def validate_optional_dataset_list(
    data: Optional[Dataset | List[Dataset]] = None,
    name: str = "",
    reference_length: Optional[int] = None,
) -> Any:
    """Validate and normalize an optional list of Datasets.

    - If data is None, returns a list of None of length `reference_length`.
    - If data is a list, checks that:
        - each element is None or a Dataset
        - if reference_length is given, len(data) == reference_length
    Returns a fresh list (copy) to avoid external mutation.

    Parameters
    ----------
    data: optional[Dataset | List[Dataset]], default to None,
        Data to be validated.
    name: str, default to ""
        Dataset class, ideally, either train or valid
    reference_length: Optional[int], default to None.
        The length of reference to validate the dataset length against.

    Returns
    -------
        List of validated data.
    """
    # check if the dataset is a single element, convert it to a single
    # element list, if a reference length is provided
    # NOTE necessary change to not break the old api
    if data is None:
        if reference_length is None:
            raise ValueError(
                f"{name} is None and no reference_length provided"
            )
        return [None] * reference_length
    data_list = [data] if isinstance(data, Dataset) else data
    for idx, item in enumerate(data_list):
        if item is not None and not isinstance(item, Dataset):
            raise InvalidDatasetType(f"{name}[{idx}]", type(item).__name__)

    if reference_length is not None and len(data_list) != reference_length:
        raise ValueError(
            f"{name} must have length {reference_length};got {len(data_list)}"
        )
    return data_list


def dispatch_replies(
    results: dict[str, MultiReply],
    get_replies: Callable[[MultiReply], list[Reply]],
) -> list[dict[str, Reply]]:
    """Dispatch multi-model replies by model index.

    Parameters
    ----------
    results: dict[str, MultiReply]
        Dictionary of MultiReply per client.
    get_replies: Callable[[MultiReply], list[Reply]]
        A callback (function) that takes a MultiReply and returns a list of
        replies.

    Returns
    -------
        a list of dictionaries in which the keys are client names and
        values are the Reply (TrainReply, EvaluationReply .. etc)
    """
    results_per_model: list[dict[str, Reply]] = []

    for client, multi_reply in results.items():
        replies = get_replies(multi_reply)

        if not results_per_model:
            results_per_model = [{} for _ in replies]
        elif len(replies) != len(results_per_model):
            raise ValueError(
                "All clients must provide the same number of replies."
            )

        for idx, reply in enumerate(replies):
            results_per_model[idx][client] = reply

    return results_per_model


def copy_requests_with_weights(
    requests: List[RequestT],
    client: str,
    needs_weights: Dict[int, set[str]],
    model_weights: Dict[int, Any],
) -> list[RequestT]:
    """Copy requests and attach weights when needed while maintaining the
    requests relevant type to avoid mypy complains.

    Parameters
    ----------
    requests: list[RequestT].
        List of requests to which the relevant model weights are copied.
    client: str
        Client name.
    needs_weights: dict[int, set[str]]
        Dictionary of clients who need the weights
    model_weights: dict[int, set[str]]
        Dictionary containing the model weights for every client

    Returns
    -------
        A list of type-casted requests with the relevant RequestT type

    """
    copied_requests: list[RequestT] = []

    for idx, rqst in enumerate(requests):
        request = copy.copy(rqst)
        if client in needs_weights[idx]:
            request.weights = model_weights[idx]  # type: ignore [attr-defined]
        copied_requests.append(request)
    return copied_requests
