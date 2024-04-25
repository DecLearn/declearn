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

"""Core components and utils for fairness-aware (federated) machine learning.

Group-fairness functions
------------------------

API-defining ABC and generic constructor:

* [FairnessFunction][declearn.fairness.core.FairnessFunction]:
    Abstract base class for group-fairness functions.
* [instantiate_fairness_function]\
[declearn.fairness.core.instantiate_fairness_function]:
    Instantiate a FairnessFunction from its specifications.

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
"""

from ._fair_func import FairnessFunction, instantiate_fairness_function
from ._functions import (
    AccuracyParityFunction,
    DemographicParityFunction,
    EqualityOfOpportunityFunction,
    EqualizedOddsFunction,
)
