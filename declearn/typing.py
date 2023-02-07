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

"""Type hinting utils, defined and exposed for code readability purposes."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)


__all__ = [
    "Batch",
    "SupportsConfig",
]

# Data batches specification: (inputs, labels, weights), where:
# - inputs and labels may be an array or a list of arrays:
# - labels and/or weights may ne None
Batch = Tuple[
    Union[ArrayLike, List[ArrayLike]],
    Optional[Union[ArrayLike, List[ArrayLike]]],
    Optional[ArrayLike],
]


class SupportsConfig(Protocol, metaclass=ABCMeta):
    """Protocol for type annotation of objects with get/from_config methods.

    This class is primarily designed to be used for type annotation,
    but may also be used to implement `get_config` and `from_config`
    the former of which requires overriding.
    """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable config dict representing this object."""
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate an object from its JSON-serializable config dict."""
        return cls(**config)
