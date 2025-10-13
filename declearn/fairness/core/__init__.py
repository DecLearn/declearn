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

"""Built-in concrete components for fairness-aware federated learning.

Dataset subclass
----------------

* [FairnessInMemoryDataset][declearn.fairness.core.FairnessInMemoryDataset]:
    Fairness-aware `InMemoryDataset` subclass.


Group-fairness functions
------------------------
Concrete implementations of various fairness functions:

* [AccuracyParityFunction][declearn.fairness.core.AccuracyParityFunction]:
    Accuracy Parity group-fairness function.
* [DemographicParityFunction]\
[declearn.fairness.core.DemographicParityFunction]:
    Demographic Parity group-fairness function for binary classifiers..
* [EqualityOfOpportunityFunction]\
[declearn.fairness.core.EqualityOfOpportunityFunction]:
    Equality of Opportunity group-fairness function.
* [EqualizedOddsFunction][declearn.fairness.core.EqualizedOddsFunction]:
    Equalized Odds group-fairness function.

Abstraction and generic constructor may be found in [declearn.fairness.api][].
An additional util may be used to list available functions, either declared
here or by third-party and end-user code:

* [list_fairness_functions][declearn.fairness.core.list_fairness_functions]:
    Return a mapping of registered FairnessFunction subclasses.
"""

from ._functions import (
    AccuracyParityFunction,
    DemographicParityFunction,
    EqualityOfOpportunityFunction,
    EqualizedOddsFunction,
    list_fairness_functions,
)
from ._inmemory import FairnessInMemoryDataset
