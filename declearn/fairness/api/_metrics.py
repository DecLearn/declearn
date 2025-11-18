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

"""Utility dataset-handler to compute group-wise model evaluation metrics."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.fairness.api._dataset import FairnessDataset
from declearn.metrics import Accuracy, MeanMetric, MetricSet
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel

__all__ = [
    "FairnessMetricsComputer",
]


class ModelLoss(MeanMetric, register=False):
    """Metric container to compute a model's loss iteratively."""

    name = "loss"

    def __init__(
        self,
        model: Model,
    ) -> None:
        super().__init__()
        self.model = model

    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return self.model.loss_function(y_true, y_pred)


class FairnessMetricsComputer:
    """Utility dataset-handler to compute group-wise evaluation metrics.

    This class aims at making fairness evaluation of models readable,
    by internalizing the computation of group-wise accuracy (or loss)
    metrics, that may then be passed to a `FairnessFunction` instance
    so as to compute the associate fairness values.

    In federated contexts, clients' group-wise accuracy scores should
    be weighted by their group-wise counts, sum-aggregated and passed
    to `FairnessFunction.compute_from_federated_group_accuracy`.

    Attributes
    ----------
    counts: Dict[Tuple[Any, ...], int]
        Category-wise number of data samples.
    g_data: Dict[Tuple[Any, ...], FairnessDataset]
        Category-wise sub-datasets, over which the accuracy of a model
        may be computed via the `compute_groupwise_accuracy` method.
    """

    def __init__(
        self,
        dataset: FairnessDataset,
    ) -> None:
        """Wrap up a `FairnessDataset` to facilitate metrics computation.

        Parameters
        ----------
        dataset:
            `FairnessDataset` instance, that wraps samples over which
            to estimate models' evaluation metrics, and defines the
            partition of that data into sensitive groups.
        """
        self.counts = dataset.get_sensitive_group_counts()
        self.g_data = {
            group: dataset.get_sensitive_group_subset(group)
            for group in dataset.get_sensitive_group_definitions()
        }

    def compute_groupwise_metrics(
        self,
        metrics: List[MeanMetric],
        model: Model,
        batch_size: int = 32,
        n_batch: Optional[int] = None,
    ) -> Dict[str, Dict[Tuple[Any, ...], float]]:
        """Compute an ensemble of mean metrics over group-wise sub-datasets.

        Parameters
        ----------
        metrics:
            List of `MeanMetric` instances defining metrics to compute,
            that are required to be scalar float values.
        model:
            Model that is to be evaluated.
        batch_size: int, default=32
            Number of samples per batch when computing predictions.
        n_batch: int or None, default=None
            Optional maximum number of batches to draw per group.
            If None, use the entire wrapped dataset.

        Returns
        -------
        metrics:
            Computed group-wise metrics, as a nested dictionary
            with `{metric.name: {group: value}}` structure.
        """
        metricset = MetricSet(metrics)
        output: Dict[str, Dict[Tuple[Any, ...], float]] = {
            metric.name: {} for metric in metrics
        }
        for group in self.g_data:
            values = self.compute_metrics_over_sensitive_group(
                group, metricset, model, batch_size, n_batch
            )
            for metric in metrics:
                output[metric.name][group] = float(values[metric.name])
        return output

    # pylint: disable-next=too-many-positional-arguments
    def compute_metrics_over_sensitive_group(
        self,
        group: Tuple[Any, ...],
        metrics: MetricSet,
        model: Model,
        batch_size: int = 32,
        n_batch: Optional[int] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute some metrics for a given model and sensitive group.

        Parameters
        ----------
        group: tuple
            Tuple of sensitive attribute values defining the group,
            the accuracy of the model over which to compute.
        metrics: MetricSet
            Ensemble of metrics that need to be computed.
        model: Model
            Model that is to be evaluated.
        batch_size: int, default=32
            Number of samples per batch when computing predictions.
        n_batch: int or None, default=None
            Optional maximum number of batches to draw.
            If None, use the entire wrapped dataset.

        Returns
        -------
        metrics:
            Dict storing resulting metrics.

        Raises
        ------
        KeyError:
            If `category` is an invalid key to the existing combinations
            of sensitive attribute values.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Prepare to iterate over batches from the target group.
        if group not in self.g_data:
            raise KeyError(f"Invalid sensitive group: '{group}'.")
        gen_batches = self.g_data[group].generate_batches(
            batch_size, shuffle=(n_batch is not None), drop_remainder=False
        )
        # Iteratively evaluate the model.
        metrics.reset()
        for idx, batch in enumerate(gen_batches):
            if n_batch and (idx == n_batch):
                break
            # Run the model in inference, and round up output scores.
            batch_predictions = model.compute_batch_predictions(batch)
            metrics.update(*batch_predictions)
        # Return the computed metrics.
        return metrics.get_result()

    def setup_accuracy_metric(
        self,
        model: Model,
        thresh: Optional[float] = None,
    ) -> MeanMetric:
        """Return a Metric object to compute a model's accuracy.

        Parameters
        ----------
        model: Model
            Model that needs to be evaluated.
        thresh: int or None, default=None
            Optional binarization threshold for binary classification
            models' output scores. If None, use 0.5 by default, or 0.0
            for `SklearnSGDModel` instances.
            Unused for multinomial classifiers (argmax over scores).

        Returns
        -------
        metric:
            `MeanMetric` subclass that computes the average accuracy
            from pre-computed model predictions.
        """
        if thresh is None:
            thresh = 0.0 if isinstance(model, SklearnSGDModel) else 0.5
        return Accuracy(thresh=thresh)

    def setup_loss_metric(
        self,
        model: Model,
    ) -> MeanMetric:
        """Compute a model's accuracy and loss over each sensitive group.

        Parameters
        ----------
        model: Model
            Model that needs to be evaluated.

        Returns
        -------
        metric:
            `MeanMetric` subclass that computes the average loss of
            the input `model` based on pre-computed predictions.
        """
        return ModelLoss(model)

    def scale_metrics_by_sample_counts(
        self,
        metrics: Dict[Tuple[Any, ...], float],
    ) -> Dict[Tuple[Any, ...], float]:
        """Scale a dict of computed group-wise metrics by sample counts.

        Parameters
        ----------
        metrics:
            Pre-computed raw metrics, as a `{group_k: score_k}` dict.

        Returns
        -------
        metrics:
            Scaled matrics, as a `{group_k: n_k * score_k}` dict.
        """
        return {key: val * self.counts[key] for key, val in metrics.items()}
