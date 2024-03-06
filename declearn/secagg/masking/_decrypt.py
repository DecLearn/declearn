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

"""Data decrypter for SecAgg using Joye-Libert homomorphic summation."""

from typing import Any, Dict, List, Union, TypeVar

import numpy as np
from declearn.model.api import Vector, VectorSpec

from declearn.secagg.masking._aggregate import ArraySpec, MaskedAggregate
from declearn.secagg.utils import Quantizer
from declearn.utils import Aggregate

__all__ = [
    "MaskingDecrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskingDecrypter:
    """Controller for the reconstruction of sums of masked values."""

    def __init__(
        self,
        n_peers: int,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        self.n_peers = n_peers
        self.max_int = 2**bitsize
        self.quantizer = Quantizer(val_range=clipval, int_range=2**bitsize - 1)
        if not self.quantizer.numpy_compatible:
            raise ValueError(
                "'MaskingDecrypter' requires 'bitsize' to be low enough for "
                "compatibility with numpy uint dtypes. This usually means a "
                "bitsize <= 64."
            )

    def decrypt_int(
        self,
        value: int,
    ) -> int:
        return value % self.max_int

    def decrypt_float(
        self,
        value: int,
    ) -> float:
        int_val = self.decrypt_int(value)
        return self.quantizer.unquantize_value(int_val)

    def decrypt_numpy_array(
        self,
        values: List[int],
        specs: ArraySpec,
    ) -> np.ndarray:
        s_val = [self.decrypt_int(val) for val in values]
        shape, dtype = specs
        if not issubclass(np.dtype(dtype).type, np.unsignedinteger):
            s_val = self.quantizer.unquantize_list(  # type: ignore[assignment]
                s_val
            )
            if issubclass(np.dtype(dtype).type, np.signedinteger):
                s_val = [round(x) for x in s_val]
        return np.array(s_val, dtype=dtype).reshape(shape)

    def decrypt_vector(
        self,
        values: List[int],
        specs: VectorSpec,
    ) -> Vector:
        int_val = [self.decrypt_int(val) for val in values]
        flt_val = self.quantizer.unquantize_list(int_val)
        return Vector.build_from_specs(flt_val, specs)

    def decrypt_aggregate(
        self,
        value: MaskedAggregate[AggregateT],
    ) -> AggregateT:
        # Perform basic verifications.
        if not isinstance(value, MaskedAggregate):
            raise TypeError(
                f"'{self.__class__.__name__}.decrypt_aggregate' expects "
                f"'MaskedAggregate' inputs but received a '{type(value)}'."
            )
        if value.max_int != self.max_int:
            raise ValueError(
                "Cannot decrypt a 'MaskedAggregate' with mismatching "
                "'max_int'."
            )
        if value.n_aggrg != self.n_peers:
            raise ValueError(
                f"'{self.__class__.__name__}.decrypt_aggregate' expects "
                "input 'MaskedAggregate' to result from the summation of "
                f"{self.n_peers} instances, but it appears {value.n_aggrg} "
                "values were in fact summed."
            )
        # Iteratively decrypt and recover encrypted fields.
        srt = end = 0
        fields = {}  # type: Dict[str, Any]
        for name, size, specs in value.enc_specs:
            end += size
            fields[name] = self._decrypt_value(value.encrypted[srt:end], specs)
            srt = end
        # Instantiate and return from decrypted and cleartext fields.
        return value.agg_cls(**fields, **value.cleartext)

    def _decrypt_value(
        self,
        values: List[int],
        specs: Union[bool, ArraySpec, VectorSpec],
    ) -> Union[int, float, np.ndarray, Vector]:
        if isinstance(specs, (tuple, list)):
            return self.decrypt_numpy_array(values, specs)
        if isinstance(specs, VectorSpec):
            return self.decrypt_vector(values, specs)
        if isinstance(specs, bool):
            func = self.decrypt_float if specs else self.decrypt_int
            return func(values[0])
        raise TypeError(
            f"Cannot decrypt inputs with specs of type '{type(specs)}'."
        )
