# coding: utf-8

"""Iterative and federative evaluation metrics computation tools.

This module provides with Metric, an abstract base class that defines
an API to iteratively and/or federatively compute evaluation metrics,
as well as a number of concrete standard machine learning metrics.

Abstractions:
* Metric:
    Abstract base class defining an API for metrics' computation.
* MeanMetric:
    Abstract class that defines a template for simple scores' averaging.

Utils:
* MetricSet:
    Wrapper to bind together an ensemble of Metric instances.
* MetricInputType:
    Type alias for valid inputs to specify a metric for `MetricSet`.
    Equivalent to `Union[Metric, str, Tuple[str, Dict[str, Any]]]`.

Classification metrics:
* BinaryAccuracyPrecisionRecall
    Accuracy, precision, recall and confusion matrix for binary classif.
    Identifier name: "binary-classif".
* MulticlassAccuracyPrecisionRecall
    Accuracy, precision, recall and confusion matrix for multiclass classif.
    Identifier name: "multi-classif".
* BinaryRocAuc:
    Receiver Operator Curve and its Area Under the Curve for binary classif.
    Identified name: "binary-roc"

Regression metrics:
* MeanAbsoluteError:
    Mean absolute error, averaged across all samples (and channels).
* MeanSquaredError:
    Mean squared error, averaged across all samples (and channels).
"""

from ._api import Metric
from ._classif import (
    BinaryAccuracyPrecisionRecall,
    MulticlassAccuracyPrecisionRecall,
)
from ._mean import MeanMetric, MeanAbsoluteError, MeanSquaredError
from ._roc_auc import BinaryRocAUC
from ._wrapper import MetricInputType, MetricSet
