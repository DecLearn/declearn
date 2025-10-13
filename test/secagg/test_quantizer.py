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

"""Unit tests for 'declearn.secagg.utils.Quantizer'."""

from unittest import mock

import numpy as np
import pytest

from declearn.secagg.utils import Quantizer


@pytest.fixture(name="quantizer")
def quantizer_fixture(
    val_range: float,
    int_range: int,
) -> Quantizer:
    """Fixture providing with a Quantizer instance."""
    return Quantizer(val_range, int_range)


@pytest.mark.parametrize("val_range", [5.0, 100.0])
@pytest.mark.parametrize(
    "int_range", [2**8 - 1, 2**32 - 1, 2**64 - 1, 2**68 - 1]
)
class TestQuantizer:
    """Unit tests for 'declearn.secagg.utils.Quantizer'."""

    def test_quantize_unquantize_value(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test that quantizing and unquantizing a scalar value works."""
        # Generate a random float within the clipping range.
        value = np.random.uniform(
            low=-quantizer.val_range, high=quantizer.val_range
        )
        # Test that it can be quantized into an int.
        quant = quantizer.quantize_value(value)
        assert isinstance(quant, int)
        assert 0 <= quant <= quantizer.int_range
        # Test that it can be recovered from the quantized int,
        # with the guarantee that no other unquantized value is
        # closer to the original one.
        recov = quantizer.unquantize_value(quant)
        assert isinstance(recov, float)
        if recov == value:
            return
        # If recovered <= value, find next greater recoverable value.
        if recov <= value:
            inc = 1
            while (rnext := quantizer.unquantize_value(quant + inc)) == recov:
                inc += 1
            assert recov <= value <= rnext
            assert abs(value - recov) < abs(value - rnext)
        # If recovered >= value, find next smaller recoverable value.
        else:
            dec = 1
            while (rprev := quantizer.unquantize_value(quant - dec)) == recov:
                dec += 1
            assert rprev <= value <= recov
            assert abs(value - recov) < abs(value - rprev)

    def test_quantize_unquantize_extreme_values(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test edge cases with out-of-range floats and edge-of-range ints."""
        max_val = quantizer.val_range
        # Test that edge float values are quantized into edge int ones.
        quant_min = quantizer.quantize_value(-max_val)
        assert quant_min == 0
        quant_max = quantizer.quantize_value(max_val)
        assert quant_max == quantizer.int_range
        # Test that edge int values are unquantized into edge float ones.
        assert quantizer.unquantize_value(0) == -max_val
        assert quantizer.unquantize_value(quantizer.int_range) == max_val
        # Test that out-of-range float values are properly clipped.
        assert quantizer.quantize_value(-max_val - 0.1) == quant_min
        assert quantizer.quantize_value(max_val + 0.1) == quant_max

    def test_numpy_compatible(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test that the `numpy_compatible` property is correct."""
        compatible = any(
            quantizer.int_range <= np.iinfo(dtype).max
            for dtype in np.unsignedinteger.__subclasses__()
        )
        assert quantizer.numpy_compatible == compatible

    def test_quantize_unquantize_array(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test that numpy array (un)quantization works properly."""
        # Generate an array of random floats within the clipping range.
        values = np.random.uniform(
            low=-quantizer.val_range, high=quantizer.val_range, size=(4, 8)
        )
        # Case when the specified quantizer should not be able to use numpy.
        if not quantizer.numpy_compatible:
            with pytest.raises(ValueError):
                quantizer.quantize_array(values)
            return
        # Case when the quantizer should be able to (un)quantize the values.
        # Test that the values can be quantized into a uint array.
        int_val = quantizer.quantize_array(values)
        assert isinstance(int_val, np.ndarray)
        assert issubclass(int_val.dtype.type, np.unsignedinteger)
        assert np.all(int_val <= quantizer.int_range)
        assert int_val.shape == values.shape
        # Test that the values can be unquantized into a float array.
        flt_val = quantizer.unquantize_array(int_val)
        assert isinstance(flt_val, np.ndarray)
        assert issubclass(flt_val.dtype.type, np.floating)
        assert np.all(np.abs(flt_val) <= quantizer.val_range)
        # Test that the values are as close to the initial ones as possible.
        fl_prev = quantizer.unquantize_array(
            np.where(int_val > 0, int_val - 1, int_val)
        )
        fl_next = quantizer.unquantize_array(
            np.where(int_val < quantizer.int_range, int_val + 1, int_val)
        )
        assert np.all(fl_prev <= values)
        assert np.all(fl_next >= values)
        error = np.abs(np.stack([flt_val, fl_prev, fl_next]) - values)
        assert np.all(np.argmin(error, axis=0) == 0)

    def test_quantize_list_backend_choice(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test that list quantization calls the proper backend."""
        with mock.patch.object(quantizer, "quantize_array") as quantize_array:
            with mock.patch.object(
                quantizer, "quantize_value"
            ) as quantize_value:
                quantizer.quantize_list([0.0])
                if quantizer.numpy_compatible:
                    quantize_array.assert_called_once()
                    quantize_value.assert_not_called()
                else:
                    quantize_array.assert_not_called()
                    quantize_value.assert_called_once()

    def test_unquantize_list_backend_choice(
        self,
        quantizer: Quantizer,
    ) -> None:
        """Test that list unquantization calls the proper backend."""
        with mock.patch.object(
            quantizer, "unquantize_array"
        ) as unquantize_array:
            with mock.patch.object(
                quantizer, "unquantize_value"
            ) as unquantize_value:
                quantizer.unquantize_list([0])
                if quantizer.numpy_compatible:
                    unquantize_array.assert_called_once()
                    unquantize_value.assert_not_called()
                else:
                    unquantize_array.assert_not_called()
                    unquantize_value.assert_called_once()
