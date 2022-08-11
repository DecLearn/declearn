# coding: utf-8

"""DataInfoField subclasses specifying common 'data_info' metadata fields."""

from typing import Any, Set

import numpy as np

from declearn2.data_info._base import DataInfoField, register_data_info_field


__all__ = [
    'ClassesField',
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
