# coding: utf-8

"""Tools to write 'data_info' metadata fields specifications.

The 'data_info' dictionaries are a discrete yet important component of
declearn's federated learning API. They convey aggregated information
about clients' data to the server, which in turns validates, combines
and passes the values to tools that require them - e.g. to initialize
a Model or parametrize an Optimizer's OptiModule plug-ins.

This module implements a small API and a pair of functions that enable
writing specifications for expected 'data_info' fields, and automating
their use to validate and combine individual 'data_info' dicts into an
aggregated one.

DataInfoField API tools
-----------------------
* DataInfoField:
    Abstract class defining an API to write field-wise specifications.
* register_data_info_field:
    Decorator for DataInfoField subclasses, enabling their effective use.
* aggregate_data_info:
    Turn a list of individual 'data_info' dicts into an aggregated one.
* get_data_info_fields_documentation:
    Gather documentation for all fields that have registered specs.

Field specifications
--------------------
* ClassesField:
    Specification for the 'classes' field.
* InputShapeField:
    Specification for the 'input_shape' field.
* NbFeaturesField:
    Specification for the 'n_features' field.
* NbSamplesField:
    Specification for the 'n_samples' field.
"""

from ._base import (
    DataInfoField,
    aggregate_data_info,
    get_data_info_fields_documentation,
    register_data_info_field,
)
from ._fields import (
    ClassesField,
    InputShapeField,
    NbFeaturesField,
    NbSamplesField,
)
