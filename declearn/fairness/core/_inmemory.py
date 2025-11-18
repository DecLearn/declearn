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

"""Fairness-aware InMemoryDataset subclass."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse  # type: ignore

from declearn.dataset import InMemoryDataset
from declearn.dataset.utils import load_data_array
from declearn.fairness.api import FairnessDataset
from declearn.typing import DataArray

__all__ = [
    "FairnessInMemoryDataset",
]


class FairnessInMemoryDataset(FairnessDataset, InMemoryDataset):
    """Fairness-aware InMemoryDataset subclass.

    This class extends `declearn.dataset.InMemoryDataset` to
    enable its use in fairness-aware federated learning. New
    parameters are added to its `__init__`: `s_attr` as well
    as `sensitive_target`, that are used to define sensitive
    groups among the held dataset. Additionally, API methods
    from `declearn.fairness.api.FairnessDataset` are defined,
    enabling to access sensitive groups' metadata and samples
    as well as to change sample weights based on the group to
    which samples belong.
    """

    def __init__(  # noqa: PLR0913
        self,
        data: Union[DataArray, str],
        *,
        s_attr: Union[DataArray, str, List[int], List[str]],
        target: Optional[Union[DataArray, str]] = None,
        s_wght: Optional[Union[DataArray, str]] = None,
        f_cols: Optional[Union[List[int], List[str]]] = None,
        sensitive_target: bool = True,
        expose_classes: bool = False,
        expose_data_type: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the memory-fitting-data `FairnessDataset` interface.

        Please refer to `declearn.dataset.InMemoryDataset`, which this class
        extends, for generalities about supported input formats.

        Parameters
        ----------
        data:
            Main data array which contains input features (and possibly
            more), or path to a dump file from which it is to be loaded.
        s_attr:
            Sensitive attributes, that define group-fairness constraints.
            May be a data array, the path to dump file, a list of `data`
            column names or a list of `data` column indices.
        target:
            Optional target labels, as a data array, or as a path to a
            dump file, or as the name of a `data` column.
        s_wght:
            Optional sample weights, as a data array, or as a path to a
            dump file, or as the name of a `data` column.
        f_cols:
            Optional list of columns in `data` to use as input features.
            These may be specified as column names or indices. If None,
            use all non-target, non-sample-weights columns of `data`.

        Other parameters
        ----------------
        sensitive_target:
            Whether to define sensitive groups based on the intersection
            of `s_attr` sensitive attributes and `target` target labels,
            or merely on `s_attr`.
        expose_classes:
            Whether to expose unique target values as part of data specs.
            This should only be used for classification datasets.
        expose_data_type:
            Whether to expose features' dtype, which will be verified to
            be unique, as part of data specs.
        seed:
            Optional seed for the random number generator used for all
            randomness-based operations required to generate batches
            (e.g. to shuffle the data or sample from it).
        """
        # inherited signature; pylint: disable=too-many-arguments
        super().__init__(
            data=data,
            target=target,
            s_wght=s_wght,
            f_cols=f_cols,
            expose_classes=expose_classes,
            expose_data_type=expose_data_type,
            seed=seed,
        )
        # Pre-emptively declare attributes to deal with fairness balancing.
        self.sensitive: pd.Series = pd.Series()
        self._smp_wght: DataArray = self.weights
        # Actually set up sensitive groups based on specific parameters.
        self._set_sensitive_data(sensitive=s_attr, use_label=sensitive_target)

    def _set_sensitive_data(
        self,
        sensitive: Union[DataArray, str, List[int], List[str]],
        use_label: bool = True,
    ) -> None:
        """Define sensitive attributes based on which to filter samples.

        This method updates in-place the `sensitive` attribute of this
        dataset instance.

        Parameters
        ----------
        sensitive:
            Sensitive attributes, either as a pandas DataFrame storing data
            that is aligned with that already interfaced by this Dataset,
            or as a list of columns that are part of `self.data` (only when
            the latter is a pandas DataFrame).
        use_label:
            Whether to use the target labels (when defined) as an additional
            sensitive attribute, placed first in the list. Default: True.

        Raises
        ------
        TypeError
            If the inputs are of unproper type.
        ValueError
            If 'sensitive' is parsed into an unproper-length data array.
        """
        # Gather (and/or validate) sensitive data as a data array.
        s_data = self._parse_sensitive_data(sensitive)
        if len(s_data) != len(self.data):  # type: ignore
            raise ValueError(
                "The passed 'sensitive' data was parsed into a DataFrame with"
                " a number of records that does not match the base data."
            )
        # Optionally add target labels as a first sensitive category.
        if use_label:
            if self.target is None:
                warnings.warn(
                    f"'{self.__class__.__name__}.set_sensitive_data' was"
                    " called with 'use_label=True', but there are no labels"
                    " defined for this instance.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                target = (
                    self.target.rename("target")
                    if isinstance(self.target, pd.Series)
                    else pd.Series(self.target, name="target")
                )
                s_data = pd.concat([target, s_data], axis=1)
        # Wrap sensitive data as a Series of tuples of values.
        self.sensitive = pd.Series(
            zip(*[s_data[c] for c in s_data.columns], strict=False)
        )

    def _parse_sensitive_data(
        self,
        sensitive: Union[DataArray, str, List[int], List[str]],
    ) -> pd.DataFrame:
        """Process inputs to `set_sensitive_data` into a data array."""
        # Handle cases when 'sensitive' is a file path of columns list.
        if isinstance(sensitive, str):
            sensitive = load_data_array(sensitive)
        elif isinstance(sensitive, list):
            if isinstance(self.data, pd.DataFrame) and all(
                col in self.data.columns for col in sensitive
            ):
                sensitive = self.data[sensitive]
            elif all(
                isinstance(col, int) and (col <= self.data.shape[1])
                for col in sensitive
            ):
                sensitive = (
                    self.data.iloc[:, sensitive]  # type: ignore[index]
                    if isinstance(self.data, pd.DataFrame)
                    else self.data[:, sensitive]  # type: ignore[index]
                )
            else:
                raise TypeError(
                    "'sensitive' was passed as a list, but matches neither"
                    " data column names nor indices."
                )
        # Type-check and optionally convert sensitive attributes to pandas.
        if isinstance(sensitive, pd.DataFrame):
            return sensitive
        if isinstance(sensitive, np.ndarray):
            return pd.DataFrame(sensitive)
        if isinstance(sensitive, scipy.sparse.spmatrix):
            return pd.DataFrame(sensitive.toarray())  # type: ignore
        raise TypeError(
            "'sensitive' should be a numpy array, scipy matrix, pandas"
            " DataFrame, path to such a structure's file dump, or list"
            " of 'data' column names or indices to slice off."
        )

    def get_sensitive_group_definitions(
        self,
    ) -> List[Tuple[Any, ...]]:
        return sorted(self.sensitive.unique().tolist())

    def get_sensitive_group_counts(
        self,
    ) -> Dict[Tuple[Any, ...], int]:
        return self.sensitive.value_counts().sort_index().to_dict()

    def get_sensitive_group_subset(
        self,
        group: Tuple[Any, ...],
    ) -> InMemoryDataset:
        mask = self.sensitive == group
        inputs = self.feats[mask]  # type: ignore
        target = (
            None if (self.target is None) else self.target[mask]  # type: ignore
        )
        s_wght = (
            None if self._smp_wght is None else self._smp_wght[mask]  # type: ignore
        )
        return InMemoryDataset(
            data=inputs,
            target=target,
            s_wght=s_wght,
            expose_classes=self.expose_classes,
            expose_data_type=self.expose_data_type,
            seed=self.seed,
        )

    def set_sensitive_group_weights(
        self,
        weights: Dict[Tuple[Any, ...], float],
        adjust_by_counts: bool = False,
    ) -> None:
        # Optionally adjust input weights based on local group-wise counts.
        if adjust_by_counts:
            counts = self.get_sensitive_group_counts()
            weights = {
                key: val * counts.get(key, 0) for key, val in weights.items()
            }
        # Define or adjust sample weights based on sensitive attributes.
        sample_weights = self.sensitive.apply(weights.get)
        if sample_weights.isnull().any():
            raise KeyError(
                f"'{self.__class__.__name__}.set_sensitive_group_weights'"
                " received input weights that seemingly do not cover all"
                " existing sensitive groups."
            )
        if self._smp_wght is not None:
            sample_weights *= self._smp_wght  # type: ignore[assignment]
        self.weights = sample_weights
