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

"""Iterative and federative evaluation metrics computation tools.

This module provides with Metric, an abstract base class that defines
an API to iteratively and/or federatively compute evaluation metrics,
as well as a number of concrete standard machine learning metrics.

Abstractions
------------
* [Metric][declearn.metrics.Metric]:
    Abstract base class defining an API for metrics' computation.
* [MeanMetric][declearn.metrics.MeanMetric]:
    Abstract class that defines a template for simple scores' averaging.

Utils
-----
* [MetricSet][declearn.metrics.MetricSet]:
    Wrapper to bind together an ensemble of Metric instances.
* MetricInputType:
    Type alias for valid inputs to specify a metric for `MetricSet`.
    Equivalent to `Union[Metric, str, Tuple[str, Dict[str, Any]]]`.

Classification metrics
----------------------
* [BinaryAccuracyPrecisionRecall]\
[declearn.metrics.BinaryAccuracyPrecisionRecall]:
    Accuracy, precision, recall and confusion matrix for binary classif.
    Identifier name: "binary-classif".
* [MulticlassAccuracyPrecisionRecall]\
[declearn.metrics.MulticlassAccuracyPrecisionRecall]:
    Accuracy, precision, recall and confusion matrix for multiclass classif.
    Identifier name: "multi-classif".
* [BinaryRocAUC][declearn.metrics.BinaryRocAUC]:
    Receiver Operator Curve and its Area Under the Curve for binary classif.
    Identifier name: "binary-roc".

Regression metrics
------------------
* [MeanAbsoluteError][declearn.metrics.MeanAbsoluteError]:
    Mean absolute error, averaged across all samples (and channels).
    Identifier name: "mae".
* [MeanSquaredError][declearn.metrics.MeanSquaredError]:
    Mean squared error, averaged across all samples (and channels).
    Identifier name: "mse".
* [RSquared][declearn.metrics.RSquared]:
    R^2 (R-Squared, coefficient of determination) regression metric.
    Identifier name: "r2".

"""

from ._api import Metric
from ._classif import (
    BinaryAccuracyPrecisionRecall,
    MulticlassAccuracyPrecisionRecall,
)
from ._mean import MeanMetric, MeanAbsoluteError, MeanSquaredError
from ._roc_auc import BinaryRocAUC
from ._rsquared import RSquared
from ._wrapper import MetricInputType, MetricSet
