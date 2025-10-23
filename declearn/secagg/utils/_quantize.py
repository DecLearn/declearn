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

"""Data quantization utils to enable SecAgg features over float values."""

import functools
import warnings
from typing import List

import numpy as np

from declearn.secagg.utils._numpy import (
    get_numpy_float_dtype,
    get_numpy_uint_dtype,
)

__all__ = [
    "Quantizer",
]


class Quantizer:
    """Data (un)quantization facility for finite-domain int/float conversion.

    This class merely aims at exposing pre-parametrized functions to
    convert floating-point values back and from unsigned integers in
    a finite domain.
    """

    def __init__(
        self,
        val_range: float,
        int_range: int,
    ) -> None:
        """Instantiate the data (un)quantizer.

        Parameters
        ----------
        val_range:
            Absolute value beyond which to clip inputs upon quantization.
            The tighter this range, the less decimal information is lost.
        int_range:
            Upper bound of the target finite integer field. The higher,
            the more memory costly and the most precise the quantization.
            To optimize memory use, set this to `2**b` where `b` is the
            maximum bitsize of the output quantized integers.

        Notes
        -----
        - The input parameters may not be changed after instantiation.
        - If `int_range` is small enough (on most systems, `< 2**64`),
          the (un)quantization operations will be runnable using numpy,
          enabling the use of the `(un)quantize_array` methods, notably
          as backend of the `(un)quantize_list` ones that will therefore
          be faster than if using pure-python loops over input lists.
        - When `2 * val_range / int_range` gets very small (usually in
          the order of 1e-16), limits to float precision may hinder the
          (un)quantization, with multiple int-domain values matching the
          same float-domain one, which can be memory-ineffective.
        """
        self._val_range = val_range
        self._int_range = int_range

    @property
    def val_range(self) -> float:
        """Absolute value limiting the unquantized float domain."""
        return self._val_range

    @property
    def int_range(self) -> int:
        """Upper bound to the quantized finite integer field domain."""
        return self._int_range

    @functools.cached_property
    def _step_size(self) -> float:
        """Pre-computed and cached scalar term for values conversion."""
        return 2 * self.val_range / self.int_range

    @functools.cached_property
    def _float_dtype(self) -> np.dtype:
        """Select the appropriate float dtype for numpy, if any.

        Raise a ValueError if `self.val_range` goes beyond the numpy
        float limit, resulting in much slower pure-python operations.
        """
        return get_numpy_float_dtype(2 * self.val_range)

    @functools.cached_property
    def _uint_dtype(self) -> np.dtype:
        """Select the appropriate uint dtype for numpy, if any.

        Raise a ValueError if `self.int_range` goes beyond the numpy
        uint limit, resulting in much slower pure-python operations.
        """
        return get_numpy_uint_dtype(self.int_range)

    @functools.cached_property
    def numpy_compatible(self) -> bool:
        """Whether this quantizer can operate via numpy or not.

        - If False, `(un)quantize_array` methods will raise ValueError.
        - If True, `(un)quantize_list` methods will make use of their
          numpy counterparts to speed up computations by vectorization.
        """
        try:
            # 1st access triggers checks; pylint: disable=pointless-statement
            self._float_dtype  # noqa: B018
            self._uint_dtype  # noqa: B018
        except ValueError:
            return False
        return True

    def quantize_value(
        self,
        value: float,
    ) -> int:
        """Quantize a given value onto the target finite integer field.

        Parameters
        ----------
        value:
            Float value that needs quantizing.

        Returns
        -------
        quantized:
            Quantized positive integer value.
        """
        clipped = max(min(value, self.val_range), -self.val_range)
        output = int(round((clipped + self.val_range) / self._step_size))
        return min(output, self.int_range)

    def unquantize_value(
        self,
        value: int,
    ) -> float:
        """Unquantize a given integer to a float value.

        Parameters
        ----------
        value:
            Int value that needs unquantizing.

        Returns
        -------
        unquantized:
            Float value recovered from the inputs.
        """
        return value * self._step_size - self.val_range

    def quantize_list(
        self,
        values: List[float],
    ) -> List[int]:
        """Quantize a list of float values onto the target finite int field.

        If the size of the target field allows it, use numpy to accelerate
        computations.

        Parameters
        ----------
        values:
            List of float values that need quantizing.

        Returns
        -------
        quantized:
            List of quantized positive integer value.
        """
        if self.numpy_compatible:
            return self.quantize_array(np.array(values)).tolist()
        return [self.quantize_value(x) for x in values]

    def unquantize_list(
        self,
        values: List[int],
    ) -> List[float]:
        """Unquantize a list of int values back to the initial float domain.

        If the size of the target field allows it, use numpy to accelerate
        computations.

        Parameters
        ----------
        values:
            Lose of int values that need unquantizing.

        Returns
        -------
        unquantized:
            List of float values recovered from the inputs.
        """
        if self.numpy_compatible:
            return self.unquantize_array(np.array(values)).tolist()
        return [self.unquantize_value(x) for x in values]

    def quantize_array(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Quantize a numpy array onto the target finite integer field.

        This method may only be called if `self.int_range <= 2**64 - 1`,
        as numpy does not support integers above unsigned 64-bit ones,
        and if `self.val_range` is lower than the maximum for float128.

        Parameters
        ----------
        values:
            Array of float values that need quantizing.

        Returns
        -------
        values:
            Array of quantized int values, with as low a size as possible.

        Raises
        ------
        ValueError
            If `self.int_range` goes above the maximum size for numpy
            unsigned integer, or `self.float_range` goes above that
            for numpy floats.
        """
        uint_dtype = self._uint_dtype
        float_dtype = (
            max(values.dtype, self._float_dtype)
            if values.dtype.kind == "f"
            else self._float_dtype
        )
        clipped = values.clip(
            min=-self.val_range,
            max=self.val_range,
            dtype=float_dtype,
        )
        outputs = np.round((clipped + self.val_range) / self._step_size)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            outputs = outputs.clip(max=self.int_range).astype(uint_dtype)
        return outputs

    def unquantize_array(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Unquantize a numpy array of quantized values back to float.

        Parameters
        ----------
        values:
            Array of integer values that need unquantizing.

        Returns
        -------
        values:
            Array of recovered float values.
        """
        return (values * self._step_size) - self.val_range
