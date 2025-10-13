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

This module implements a small API and a pair of functions that enable
writing specifications for expected 'data_info' fields, and automating
their use to validate and combine individual 'data_info' dicts into an
aggregated one.

DataInfoField API tools
-----------------------

* [DataInfoField][declearn.data_info.DataInfoField]:
    Abstract class defining an API to write field-wise specifications.
* [register_data_info_field][declearn.data_info.register_data_info_field]:
    Decorator for DataInfoField subclasses, enabling their effective use.
* [aggregate_data_info][declearn.data_info.aggregate_data_info]:
    Turn a list of individual 'data_info' dicts into an aggregated one.
* [get_data_info_fields_documentation]\
[declearn.data_info.get_data_info_fields_documentation]:
    Gather documentation for all fields that have registered specs.

Field specifications
--------------------

* [ClassesField][declearn.data_info.ClassesField]:
    Specification for the 'classes' field.
* [DataTypeField][declearn.data_info.DataTypeField]:
    Specification for the 'data_type' field.
* [FeaturesShapeField][declearn.data_info.FeaturesShapeField]:
    Specification for the 'features_shape' field.
* [NbSamplesField][declearn.data_info.NbSamplesField]:
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
    DataTypeField,
    FeaturesShapeField,
    NbSamplesField,
)
