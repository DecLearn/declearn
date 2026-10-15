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


from typing import Callable, Iterable, List, Optional, TypeVar

from declearn.dataset._base import Dataset
from declearn.main.utils._error import InvalidDatasetType

T = TypeVar("T")

MultiReply = TypeVar("MultiReply")
Reply = TypeVar("Reply")


__all__ = ["validate_optional_dataset_list", "dispatch_replies"]


def validate_optional_dataset_list(
    data: Optional[Dataset | List[Dataset]] = None,
    name: str = "",
    reference_length: Optional[int] = None,
) -> List[Optional[T]]:
    """Validate and normalize an optional list of Datasets.

    - If data is None, returns a list of None of length `reference_length`.
    - If data is a list, checks that:
        - each element is None or a Dataset
        - if reference_length is given, len(data) == reference_length
    Returns a fresh list (copy) to avoid external mutation.
    """
    # check if the dataset is a single element, convert it to a single
    # element list
    # NOTE necessary change to not break the old api

    if data is None:
        if reference_length is None:
            raise ValueError(
                f"{name} is None and no reference_length provided"
            )
        return [None] * reference_length
    data_list = [data] if isinstance(data, Dataset) else data
    # Validate element types
    for idx, item in enumerate(data_list):
        if item is not None and not isinstance(item, Dataset):
            raise InvalidDatasetType(f"{name}[{idx}]", type(item).__name__)

    # Validate length against reference if provided
    if reference_length is not None and len(data_list) != reference_length:
        raise ValueError(
            f"{name} must have length {reference_length};got {len(data_list)}"
        )

    return data_list


def dispatch_replies(
    results: dict[str, MultiReply],
    get_replies: Callable[[MultiReply], list[Reply]],
) -> list[dict[str, Reply]]:
    """Dispatch multi-model replies by model index."""
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
