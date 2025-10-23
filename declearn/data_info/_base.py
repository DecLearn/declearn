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

"""Tools to write 'data_info' metadata fields specifications.

The 'data_info' dictionaries are a discrete yet important component of
declearn's federated learning API. They convey aggregated information
about clients' data to the server, which in turns validates, combines
and passes the values to tools that require them - e.g. to initialize
a Model or parametrize an Optimizer's OptiModule plug-ins.

This (private) submodule implements a small API and a pair of functions
that enable writing specifications for expected 'data_info' fields and
automating their use to validate and combine individual values into a
dict of "final" aggregated values.

The `DataInfoField` class defines (somewhat-verbosely) an API to write
field-wise specifications for values' type-verification and combination.

The `register_data_info_field` decorator should be used to decorate any
subclass of `DataInfoField`, so that it is effectively used by derived
tools.

The `aggregate_data_info` function builds on the previous specifications
to provide a simple way to derive an aggregated 'data_info' dict from a
list of individual (e.g. client-wise) ones.

The `get_data_info_fields_documentation` function lists the fields that
have a registered specification, and provide with a short documentation
for the latter (taken from the `DataInfoField.cls` class attribute).

Note that some subclasses, providing specifications for the most common
data_info fields, are implemented (although unexposed) here.
"""

import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type

__all__ = [
    "DataInfoField",
    "aggregate_data_info",
    "get_data_info_fields_documentation",
    "register_data_info_field",
]


class DataInfoField(metaclass=ABCMeta):
    """Abstract base class to implement 'data_info' fields specifications.

    This class defines a small API to write specifications for expected
    metadata fields shared by clients in a federated learning process.

    Subclasses are not meant to be instantiated, but act as a namespace,
    and must implement the following class attributes and class methods:

    Attributes
    ----------
    field: str
        Name of the field that is being specified.
    types: tuple of types
        Supported types for individual field values.
    doc: str
        Short summary of the field's specification (accessible by
        using the `get_data_info_fields_documentation` function).

    Methods
    -------
    is_valid: any -> bool
        Check that a given value is valid for this field.
        If not overridden, run isinstance(value, `cls.types`).
    combine: *any -> any
        Combine multiple field values into a single one.
        This method needs extension or overridding. If super
        is called, run `is_valid` on each and every input.
    """

    field: ClassVar[str] = NotImplemented
    types: ClassVar[Tuple[Type, ...]] = NotImplemented
    doc: ClassVar[str] = NotImplemented

    @classmethod
    def is_valid(
        cls,
        value: Any,
    ) -> bool:
        """Check that a given value may belong to this field."""
        # false-pos; pylint: disable=isinstance-second-argument-not-valid-type
        return isinstance(value, cls.types)

    @classmethod
    @abstractmethod
    def combine(
        cls,
        *values: Any,
    ) -> Any:
        """Combine multiple field values into a single one.

        Raise a ValueError if input values are invalid or incompatible.
        """
        if not all(cls.is_valid(val) for val in values):
            raise ValueError(
                f"Cannot combine '{cls.field}': invalid values encountered."
            )


DATA_INFO_FIELDS: Dict[str, Type[DataInfoField]] = {}


def register_data_info_field(
    cls: Type[DataInfoField],
) -> Type[DataInfoField]:
    """Decorator to register DataInfoField subclasses."""
    if not issubclass(cls, DataInfoField):
        raise TypeError(
            f"Cannot register '{cls}': not a DataInfoField subclass."
        )
    if cls.field in DATA_INFO_FIELDS:
        raise KeyError(f"DataInfoField name '{cls.field}' is already used.")
    DATA_INFO_FIELDS[cls.field] = cls
    return cls


def aggregate_data_info(
    clients_data_info: List[Dict[str, Any]],
    required_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Combine individual 'data_info' dict into a single one.

    Parameters
    ----------
    clients_data_info: list[dict[str, any]]
        List of client-wise data specifications.
    required_fields: set[str] or None, default=None
        Optional set of fields to target among provided information.
        If set, raise if a field is missing from any client, and use
        only these fields in the returned dict.

    Returns
    -------
    data_info: dict[str, any]
        Aggregated data specifications derived from individual ones.
        Fields are either `required_fields` or the intersection of
        the individual dicts' fields.
        Values are the result of `DataInfoField.combine(...)` called
        on individual values, provided a `DataInfoField` subclass is
        associated with the field's name.

    Raises
    ------
    KeyError
        If a field in `required_fields` is missing from at least one
        dict included in `clients_data_info`.
    ValueError
        If any value of a shared field is invalid, or if values from
        a field are incompatible for combination.

    Warns
    -----
    UserWarning
        If one of the return fields has no corresponding registered
        `DataInfoField` specification class.
        In that case, return the list of individual values in lieu
        of aggregated value.

    Notes
    -----
    See `declearn.data_info.register_data_info_field` for details on
    how to register a `DataInfoField` subclass. See the latter (also
    part of `declearn.data_info`) for the field specification API.
    """
    # Select shared fields across clients, or required ones.
    fields = set.intersection(*[set(info) for info in clients_data_info])
    if required_fields is not None:
        missing = set(required_fields).difference(fields)
        if missing:
            raise KeyError(
                f"Missing required fields in at least one dict: {missing}"
            )
        fields = required_fields
    # Gather and spec-based-aggregate individual values.
    data_info: Dict[str, Any] = {}
    for field in fields:
        values = [info[field] for info in clients_data_info]
        spec = DATA_INFO_FIELDS.get(field)
        if spec is None:
            warnings.warn(
                f"Unspecified 'data_info' field '{field}': "
                "returning list of individual values.",
                stacklevel=2,
            )
            data_info[field] = values
        else:
            data_info[field] = spec.combine(*values)
    # Return aggregated information.
    return data_info


def get_data_info_fields_documentation(
    display: bool = True,
) -> Dict[str, str]:
    """Return the documentation of all registered `DataInfoField` subclasses.

    Parameters
    ----------
    display: bool, default=True
        Whether to print a docstring-like version of the returned dict.

    Returns
    -------
    documentation: dict[str, str]
        Dict with specified 'data_info' fields as keys, and associated
        documentation (from `DataInfoField.doc`) as values.
    """
    documentation = {cls.field: cls.doc for cls in DATA_INFO_FIELDS.values()}
    if display:
        msg = "\n".join(
            f"{field}:\n    {doc}" for field, doc in documentation.items()
        )
        print(msg)
    return documentation
