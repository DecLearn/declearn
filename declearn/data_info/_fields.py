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

"""DataInfoField subclasses specifying common 'data_info' metadata fields."""

from typing import Any, Optional, Set, Tuple

import numpy as np

from declearn.data_info._base import DataInfoField, register_data_info_field

__all__ = [
    "ClassesField",
    "DataTypeField",
    "FeaturesShapeField",
    "NbSamplesField",
]


@register_data_info_field
class ClassesField(DataInfoField):
    """Specifications for 'classes' data_info field."""

    field = "classes"
    types = (list, set, tuple, np.ndarray)
    doc = "Set of classification targets, combined by union."

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
class DataTypeField(DataInfoField):
    """Specifications for 'data_type' data_info field."""

    field = "data_type"
    types = (str,)
    doc = "Type of dataset(s)."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        if not isinstance(value, str):
            return False
        try:
            np.dtype(value)
        except TypeError:
            return False
        return True

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
class FeaturesShapeField(DataInfoField):
    """Specifications for 'features_shape' data_info field."""

    field = "features_shape"
    types = (tuple, list)
    doc = "Input features' shape, excluding batch size, checked to be equal."

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        return isinstance(value, cls.types) and all(
            (isinstance(val, int) and val > 0) or (val is None)
            for val in value
        )

    @classmethod
    def combine(
        cls,
        *values: Any,
    ) -> Tuple[Optional[int], ...]:
        # Type check each and every input shape.
        super().combine(*values)
        # Check that all shapes are the same.
        unique_shapes = list({tuple(shp) for shp in values})
        if len(unique_shapes) != 1:
            raise ValueError(
                f"Cannot combine '{cls.field}': non-unique shapes."
            )
        return unique_shapes[0]


@register_data_info_field
class NbSamplesField(DataInfoField):
    """Specifications for 'n_samples' data_info field."""

    field = "n_samples"
    types = (int,)
    doc = "Number of data samples, combined by summation."

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
