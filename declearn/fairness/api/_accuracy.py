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

"""Utility dataset-handler to compute group-wise accuracy metrics."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn.fairness.api._dataset import FairnessDataset
from declearn.metrics import MeanMetric, MetricSet
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel


__all__ = [
    "FairnessAccuracyComputer",
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


class BinaryAccuracy(MeanMetric, register=False):
    """Metric container to compute binary accuracy iteratively."""

    name = "binary-accuracy"

    def __init__(
        self,
        thresh: float,
    ) -> None:
        super().__init__()
        self.thresh = thresh

    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_pred = (
            y_pred > self.thresh
            if (y_pred.ndim == 1) or (y_pred.shape[1] == 1)
            else y_pred.max(axis=1)
        )
        return y_pred == y_true


class FairnessAccuracyComputer:
    """Utility dataset-handler to compute group-wise accuracy metrics.

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
            `FairnessDataset` instance, that wraps samples over which to
            estimate models' accuracy and/or loss metrics, and defines
            the partition of that data into sensitive groups.
        """
        self.counts = dataset.get_sensitive_group_counts()
        self.g_data = {
            group: dataset.get_sensitive_group_subset(group)
            for group in dataset.get_sensitive_group_definitions()
        }

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
            Model that needs to be evaluated.
        batch_size: int, default=32
            Number of samples per batch over which to run `model` in
            inference and iteratively compute the accuracy metric.
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
        # Return the computed accuracy score.
        return metrics.get_result()

    def compute_groupwise_accuracy(
        self,
        model: Model,
        batch_size: int = 32,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Dict[Tuple[Any, ...], float]:
        """Compute a model's accuracy over each and every sensitive group.

        Parameters
        ----------
        model: Model
            Model that needs to be evaluated.
        batch_size: int, default=32
            Number of samples per batch over which to run `model` in
            inference and iteratively compute the group-wise accuracy.
        n_batch: int or None, default=None
            Optional maximum number of batches to draw per category.
            If None, use the entire wrapped dataset.
        thresh: int or None, default=None
            Optional binarization threshold for binary classification
            models' output scores. If None, use 0.5 by default, or 0.0
            for `SklearnSGDModel` instances.
            Unused for multinomial classifiers (argmax over scores).

        Returns
        -------
        accuracy:
            Group-wise accuracy metrics, as a dict, the keys of which
            are tuples of sensitive attributes values that define the
            sensitive groups.
        """
        # Optionally set up a default binarization threshold.
        if thresh is None:
            thresh = 0.0 if isinstance(model, SklearnSGDModel) else 0.5
        return {
            group: self._compute_accuracy_over_sensitive_group(
                group, model, batch_size, n_batch, thresh
            )
            for group in self.g_data
        }

    def _compute_accuracy_over_sensitive_group(
        self,
        group: Tuple[Any, ...],
        model: Model,
        batch_size: int = 32,
        n_batch: Optional[int] = None,
        thresh: float = 0.5,
    ) -> float:
        """Compute the accuracy of a model for a given sensitive group.

        Parameters
        ----------
        group: tuple
            Tuple of sensitive attribute values defining the group,
            the accuracy of the model over which to compute.
        model: Model
            Model that needs to be evaluated.
        batch_size: int, default=32
            Number of samples per batch over which to run `model` in
            inference and iteratively compute the accuracy metric.
        n_batch: int or None, default=None
            Optional maximum number of batches to draw.
            If None, use the entire wrapped dataset.
        thresh: float, default=0.5
            Binarization threshold for binary classification models'
            output scores. Unused for multinomial classifiers.

        Returns
        -------
        accuracy:
            Scalar float binary accuracy metric, defined as
            `P(Y_true == Y_categ | S = category)`.

        Raises
        ------
        KeyError:
            If `category` is an invalid key to the existing combinations
            of sensitive attribute values.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        metrics = MetricSet([BinaryAccuracy(thresh=thresh)])
        results = self.compute_metrics_over_sensitive_group(
            group, metrics, model, batch_size, n_batch
        )
        return float(results[BinaryAccuracy.name])

    def compute_groupwise_accuracy_and_loss(
        self,
        model: Model,
        batch_size: int = 32,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[Dict[Tuple[Any, ...], float], Dict[Tuple[Any, ...], float]]:
        """Compute a model's accuracy and loss over each sensitive group.

        Parameters
        ----------
        model: Model
            Model that needs to be evaluated.
        batch_size: int, default=32
            Number of samples per batch over which to run `model` in
            inference and iteratively compute the group-wise accuracy.
        n_batch: int or None, default=None
            Optional maximum number of batches to draw per category.
            If None, use the entire wrapped dataset.
        thresh: int or None, default=None
            Optional binarization threshold for binary classification
            models' output scores. If None, use 0.5 by default, or 0.0
            for `SklearnSGDModel` instances.
            Unused for multinomial classifiers (argmax over scores).

        Returns
        -------
        accuracy:
            Group-wise accuracy metrics, as a dict, the keys of which
            are tuples of sensitive attributes values that define the
            sensitive groups.
        loss:
            Group-wise model loss values, as a dict with the same keys
            and format as `accuracy`.
        """
        # Set up metrics to be computed.
        if thresh is None:
            thresh = 0.0 if isinstance(model, SklearnSGDModel) else 0.5
        metrics = MetricSet([BinaryAccuracy(thresh), ModelLoss(model)])
        # Compute group-wise metrics and parse them into output dicts.
        accuracy = {}  # type: Dict[Tuple[Any, ...], float]
        g_losses = {}  # type: Dict[Tuple[Any, ...], float]
        for group in self.g_data:
            results = self.compute_metrics_over_sensitive_group(
                group, metrics, model, batch_size, n_batch
            )
            accuracy[group] = float(results[BinaryAccuracy.name])
            g_losses[group] = float(results[ModelLoss.name])
        # Return the pair of dicts storing results.
        return accuracy, g_losses

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
