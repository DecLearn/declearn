# coding: utf-8

"""DataInfoField subclasses specifying common 'data_info' metadata fields."""

from typing import Any, List, Optional, Set

import numpy as np

from declearn2.data_info._base import DataInfoField, register_data_info_field


__all__ = [
    'ClassesField',
    'InputShapeField',
    'NbFeaturesField',
    'NbSamplesField',
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
class InputShapeField(DataInfoField):
    """Specifications for 'input_shape' data_info field."""

    field = "input_shape"
    types = (tuple, list)
    doc = "Input features' batched shape, checked to be equal."

    @classmethod
    def is_valid(
            cls,
            value: Any,
        ) -> bool:
        return isinstance(value, cls.types) and (len(value) >= 2) and all(
            isinstance(val, int) or (val is None) for val in value
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

    field = "n_features"
    types = (int,)
    doc = "Number of input features, checked to be equal."

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
        return unique[0]  # type: ignore


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
