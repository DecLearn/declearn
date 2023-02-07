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

"""DataInfoField subclasses specifying common 'data_info' metadata fields."""

from typing import Any, ClassVar, List, Optional, Set, Tuple, Type

import numpy as np

from declearn.data_info._base import DataInfoField, register_data_info_field

__all__ = [
    "ClassesField",
    "InputShapeField",
    "NbFeaturesField",
    "NbSamplesField",
]


@register_data_info_field
class ClassesField(DataInfoField):
    """Specifications for 'classes' data_info field."""

    field: ClassVar[str] = "classes"
    types: ClassVar[Tuple[Type, ...]] = (list, set, tuple, np.ndarray)
    doc: ClassVar[str] = "Set of classification targets, combined by union."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        if isinstance(value, np.ndarray):
            return value.ndim == 1
        return super().is_valid(value)

    @classmethod
    def combine(
        cls,
        *values: Any,
    ) -> Set[Any]:
        super().combine(*values)  # type-check inputs
        return set.union(*map(set, values))


@register_data_info_field
class InputShapeField(DataInfoField):
    """Specifications for 'input_shape' data_info field."""

    field: ClassVar[str] = "input_shape"
    types: ClassVar[Tuple[Type, ...]] = (tuple, list)
    doc: ClassVar[str] = "Input features' batched shape, checked to be equal."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        return (
            isinstance(value, cls.types)
            and (len(value) >= 2)
            and all(isinstance(val, int) or (val is None) for val in value)
        )

    @classmethod
    def combine(
        cls,
        *values: Any,
    ) -> List[Optional[int]]:
        # Type check each and every input shape.
        super().combine(*values)
        # Check that all shapes are of same length.
        unique = list({len(shp) for shp in values})
        if len(unique) != 1:
            raise ValueError(
                f"Cannot combine '{cls.field}': inputs have various lengths."
            )
        # Fill-in the unified shape: except all-None or (None or unique) value.
        # Note: batching dimension is set to None by default (no check).
        shape = [None] * unique[0]  # type: List[Optional[int]]
        for i in range(1, unique[0]):
            val = [shp[i] for shp in values if shp[i] is not None]
            if not val:  # all None
                shape[i] = None
            elif len(set(val)) > 1:
                raise ValueError(
                    f"Cannot combine '{cls.field}': provided shapes differ."
                )
            else:
                shape[i] = val[0]
        # Return the combined shape.
        return shape


@register_data_info_field
class NbFeaturesField(DataInfoField):
    """Specifications for 'n_features' data_info field."""

    field: ClassVar[str] = "n_features"
    types: ClassVar[Tuple[Type, ...]] = (int,)
    doc: ClassVar[str] = "Number of input features, checked to be equal."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        return isinstance(value, int) and (value > 0)

    @classmethod
    def combine(
        cls,
        *values: Any,
    ) -> int:
        unique = list(set(values))
        if len(unique) != 1:
            raise ValueError(
                f"Cannot combine '{cls.field}': non-unique inputs."
            )
        if not cls.is_valid(unique[0]):
            raise ValueError(
                f"Cannot combine '{cls.field}': invalid unique value."
            )
        return unique[0]


@register_data_info_field
class NbSamplesField(DataInfoField):
    """Specifications for 'n_samples' data_info field."""

    field: ClassVar[str] = "n_samples"
    types: ClassVar[Tuple[Type, ...]] = (int,)
    doc: ClassVar[str] = "Number of data samples, combined by summation."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        return isinstance(value, int) and (value > 0)

    @classmethod
    def combine(
        cls,
        *values: Any,
    ) -> int:
        super().combine(*values)  # type-check inputs
        return sum(values)
